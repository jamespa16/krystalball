"""
krystalball -- live webcam next-frame prediction, trained online in real time,
served as a self-hosted web page.

Usage:
    uv sync
    uv run server.py [--port 8000] [--lr 1e-3] [--lstm-hidden-channels 128]
                      [--optimizer adam] [--loss mse_ssim]
    (run `uv run server.py --help` for the full list of flags)

    Then open https://<this-machine's-lan-ip>:<port>/ in a browser on any
    device on the same network -- the browser (not this machine) needs the
    webcam. Browsers only allow camera access over a secure context, so for
    LAN access (anything other than localhost) you need a TLS cert:

        openssl req -x509 -newkey rsa:2048 -nodes -keyout key.pem \\
            -out cert.pem -days 365 -subj "/CN=krystalball"

        uv run server.py --ssl-keyfile key.pem --ssl-certfile cert.pem

    (your browser will show a one-time warning for the self-signed cert --
    that's expected, click through it.)

Web page controls:
    Delay slider -- "Real frame delay (frames)", snaps to whole-frame
        detents, shows the current value. At 0 every frame is trained on
        immediately (original 1:1 behavior). Higher values buffer real
        frames into a small FIFO so training lags real time by that many
        frames but still eventually trains on every frame, nothing dropped.
    Reset button -- clears recurrent hidden state, replay buffer, and
        re-initializes the optimizer, without restarting the server.
    Stop button -- pauses training (state preserved, not cleared); sending
        any new frame (e.g. reloading the page and re-granting camera
        access) automatically resumes it.

What you'll see:
    Two panes side by side: [ real webcam frame | model's current
    prediction ], with a stats readout (frame count, FPS, rolling-average
    training loss, replay-buffer backlog, REAL/FREE-RUN mode). The
    prediction pane should start out noisy/blurry and grow more temporally
    coherent over the first 30-60 seconds as the model learns online.
"""

import asyncio
import json
from contextlib import asynccontextmanager

import torch
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

from config import build_arg_parser
from training import TrainingEngine

args = build_arg_parser().parse_args()

_divisor = 2 ** args.encoder_scales
if args.width % _divisor != 0 or args.height % _divisor != 0:
    raise ValueError(
        f"--width and --height must both be divisible by {_divisor} "
        f"(the encoder/decoder downsample/upsample by 2x, {args.encoder_scales} times)."
    )

if args.device == "auto":
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.mps.is_available()
        else "cpu"
    )
else:
    device = torch.device(args.device)
print(f"[krystalball] device: {device}")

# How long an abandoned connection can go without a new frame before training
# auto-pauses (state preserved, not cleared -- resumes on the next frame).
DISCONNECT_GRACE_SECONDS = 5.0
OUTPUT_PUMP_HZ = 30


@asynccontextmanager
async def lifespan(app: FastAPI):
    engine = TrainingEngine(args, device)
    engine.start()
    app.state.engine = engine
    yield
    engine.stop_and_join()


app = FastAPI(lifespan=lifespan)


async def _handle_control_message(engine: TrainingEngine, text: str):
    try:
        msg = json.loads(text)
    except json.JSONDecodeError:
        return
    msg_type = msg.get("type")
    if msg_type == "set_delay":
        engine.set_delay(msg.get("frames", 0))
    elif msg_type == "reset":
        engine.request_reset()
    elif msg_type == "stop":
        engine.pause()


async def _delayed_pause(engine: TrainingEngine, delay: float):
    await asyncio.sleep(delay)
    engine.pause()


async def _pump_output(websocket: WebSocket, engine: TrainingEngine):
    """Polls the engine's output mailbox and forwards new (prediction,
    stats) pairs to the browser at up to OUTPUT_PUMP_HZ -- fully decoupled
    from the training loop's own rate, and from the browser's send rate."""
    interval = 1.0 / OUTPUT_PUMP_HZ
    while True:
        item = engine.pop_latest_output()
        if item is not None:
            pred_jpeg, stats = item
            await websocket.send_bytes(b"\x01" + pred_jpeg)
            await websocket.send_json(stats)
        await asyncio.sleep(interval)


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    engine: TrainingEngine = websocket.app.state.engine

    await websocket.send_json({
        "type": "config",
        "width": args.width,
        "height": args.height,
        "upscale": args.upscale,
        "max_lag": args.real_frame_interval_max_frames,
    })

    pump_task = asyncio.create_task(_pump_output(websocket, engine))
    try:
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                break
            data_bytes = message.get("bytes")
            if data_bytes is not None:
                engine.submit_frame(data_bytes)
                continue
            data_text = message.get("text")
            if data_text is not None:
                await _handle_control_message(engine, data_text)
    except WebSocketDisconnect:
        pass
    finally:
        pump_task.cancel()
        # Single-user tool: don't leave the model training on stale hidden
        # state forever if the tab is closed and no one reconnects, but don't
        # instantly wipe it either -- any new frame naturally resumes.
        asyncio.create_task(_delayed_pause(engine, DISCONNECT_GRACE_SECONDS))


app.mount("/", StaticFiles(directory="static", html=True), name="static")


if __name__ == "__main__":
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        ssl_keyfile=args.ssl_keyfile,
        ssl_certfile=args.ssl_certfile,
    )
