# krystalball

A Python app that trains a neural network **online, in real time**, on a
live webcam feed to predict the next video frame — and shows you it
learning. It's a self-hosted web page, not a desktop app: the server (which
does the training) doesn't need a webcam of its own — any browser on the
network, e.g. your laptop, supplies the camera and renders two live panes:
`[real frame | model's prediction]`, updating live while the model trains
underneath.

There's no dataset, no offline training, no checkpoints. The model starts
from random weights the moment you run it, and everything you see it
predict is produced by weights that have only ever seen the live camera
feed, up to that instant.

## How it works

Every loop iteration:

1. The model has already made a prediction for "the next frame," based on
   everything it's seen so far.
2. The next real frame arrives (over a WebSocket from the browser). The loss
   between that prediction and the real frame is computed, backpropagated,
   and the optimizer steps.
3. That same real frame is then fed into the model to update its temporal
   state and produce a prediction for the *following* frame.

This predict-then-learn ordering — always one step behind — is the
non-obvious core of the app, implemented explicitly in `training.py`'s
`TrainingEngine`. Hidden state is detached from autograd after every
training step, so backprop never spans more than one timestep no matter how
long the process runs.

The model has real temporal memory (a stacked ConvLSTM core, plus
per-scale recurrence at each encoder skip connection) — a single
frame-in/frame-out network can't represent motion, so this app doesn't use
one. It also predicts a learned optical-flow field each step, warps the
previous frame forward by it, and blends that warped estimate with a
freely generated one to build its prediction — an explicit motion model on
top of raw pixel regression.

### Adjustable real-frame delay

A browser slider ("Real frame delay (frames)") controls how far training is
allowed to lag behind live video, from 0 up to a configurable max (default
60):

- **At 0** (default): every frame is trained on immediately — the
  original 1:1 predict-then-learn loop.
- **Above 0**: incoming frames are pushed into a small in-memory FIFO
  replay buffer instead. Training drains it one frame per iteration, so it
  lags real time by exactly the slider's value — but every frame is still
  eventually trained on. Nothing is discarded, and nothing is persisted;
  the buffer is a scheduling mechanism, not a dataset.

While buffering is active, the *displayed* prediction pane runs its own
separate, periodically-reseeded self-feeding rollout under `torch.no_grad()`
purely so the preview still looks live — it never produces gradients and
never touches the replay buffer or the real training path.

## Requirements

- Python ≥ 3.10
- A webcam **on whatever device you open the page from** (the server itself
  doesn't need one)
- GPU optional on the server — auto-detects CUDA or Apple MPS, falls back
  to CPU

Dependencies (PyTorch, OpenCV, NumPy, FastAPI, Uvicorn) are managed by
[uv](https://docs.astral.sh/uv/).

## Quickstart

```bash
uv sync
uv run server.py
```

Then open `http://localhost:8000/` in a browser **on the same machine** and
grant camera permission. The prediction pane starts noisy/blurry and should
visibly sharpen and become more temporally coherent over the first 30–60
seconds as the model trains.

```bash
uv run server.py --help          # see all CLI flags
uv add <package>                  # add a new dependency
```

### Accessing it from another device (e.g. a laptop, from a desktop server)

Browsers only allow camera access (`getUserMedia`) over a secure context —
`https://` or `localhost`. To open the page from a *different* device than
the one running the server, generate a one-time self-signed TLS cert and
pass it to the server:

```bash
openssl req -x509 -newkey rsa:2048 -nodes -keyout key.pem -out cert.pem -days 365 -subj "/CN=krystalball"
uv run server.py --ssl-keyfile key.pem --ssl-certfile cert.pem
```

Then open `https://<server-machine's-lan-ip>:8000/` from the other device's
browser. You'll get a one-time certificate warning since it's self-signed —
that's expected, click through it.

## Controls

| Control | Effect |
|---|---|
| Stop / Resume button | Pauses training (hidden/optimizer state is preserved, not cleared); clicking again (or reconnecting/reloading) resumes it. Does **not** shut down the server — that's an operator action (Ctrl-C) |
| Reset button | Clears recurrent hidden state, replay buffer, and optimizer state (fresh Adam/SGD), without restarting the process. Use this if training visibly diverges |
| "Real frame delay (frames)" slider | Live-adjustable real-frame delay described above; snaps to whole-frame detents, value echoed in the stats readout |

The stats readout also shows frame count, FPS, rolling-average training
loss, replay-buffer backlog, and whether the displayed frame is currently
anchored to a real frame or free-running.

## Configuration

Everything tunable at server startup is a CLI flag (see
`uv run server.py --help` for the full, current list with defaults) —
nothing is buried in logic. Defaults for any of these flags can also be set
in `config.yaml` (loaded automatically if present; point elsewhere with
`--config path/to/file.yaml`) — an explicit CLI flag always overrides
`config.yaml`, which in turn overrides `config.py`'s own hardcoded
defaults. Handy for pinning a tuned set of values without retyping them on
every `uv run server.py` invocation. Highlights:

- **Server**: `--host`, `--port`, `--ssl-keyfile`/`--ssl-certfile` (for LAN
  HTTPS access)
- **Resolution**: `--width`/`--height` (internal training resolution,
  default 96×72), `--upscale` (display-only)
- **Model size**: `--encoder-base-channels`, `--encoder-scales`,
  `--res-blocks-per-scale`, `--lstm-layers`, `--lstm-hidden-channels`,
  `--lstm-kernel-size`, `--skip-lstm-base-channels` (per-scale skip
  recurrence; 0 disables it)
- **Motion modeling**: `--use-flow` (learned optical-flow head),
  `--flow-hidden-channels`, `--blend-mask` (decoder blends a warped
  estimate with a freely generated one)
- **Training**: `--lr`, `--lr-warmup-steps`, `--optimizer` (adam/sgd),
  `--loss` (mse/ssim/mse_ssim), `--ssim-weight`, `--motion-loss-weight`,
  `--motion-delta-weight`, `--flow-smoothness-weight`, `--grad-clip-norm`
- **Replay buffer**: `--real-frame-interval-frames` (initial delay),
  `--real-frame-interval-max-frames` (slider upper bound)
- **Device**: `--device` (auto/cpu/cuda/mps)

## Project layout

- `server.py` — FastAPI app/entrypoint; creates the one `TrainingEngine`,
  mounts the `static/` frontend, exposes the `/ws` WebSocket endpoint
- `training.py` — `TrainingEngine`: the predict-then-learn training loop as
  a background thread, plus `LatestSlot`, the lock-protected mailbox
  bridging the async WebSocket handler and the sync training loop
- `frame_codec.py` — JPEG bytes ↔ model tensor conversion for the
  WebSocket transport
- `model.py` — multi-scale U-Net-style encoder/decoder with GroupNorm
  residual blocks, stacked ConvLSTM temporal core (plus per-scale skip
  recurrence), a learned optical-flow head, and swappable loss functions
  (MSE / SSIM / blended)
- `static/` — browser frontend (`index.html`/`app.js`/`style.css`):
  webcam capture, WebSocket client, canvas rendering, controls
- `config.py` — all CLI flags/defaults; also layers in `config.yaml` (or
  `--config`'s path) as defaults underneath explicit CLI flags
- `config.yaml` — optional YAML file of flag defaults, one key per CLI flag
  (dashes become underscores); delete a key to fall back to `config.py`'s
  hardcoded default for it
- `idea.md` — the authoritative spec for this project; read it before
  making architectural changes

## Non-goals

By design, this project does **not** do checkpointing, offline/disk-backed
datasets, general-purpose N-steps-ahead rollout as a standalone feature, or
multi-user/auth support (it's a single-operator tool, just browser-fronted
instead of OpenCV-windowed). The replay buffer is a small bounded in-memory
FIFO queue, not a persisted dataset — see `idea.md`'s Non-goals section
before adding any of these.
