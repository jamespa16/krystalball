"""
krystalball -- live webcam next-frame prediction, trained online in real time.

Usage:
    uv sync
    uv run main.py [--camera-index 0] [--width 96] [--height 72] [--upscale 6]
                   [--lr 1e-3] [--hidden-channels 32] [--optimizer adam] [--loss mse]
    (run `uv run main.py --help` for the full list of flags)

Keyboard controls (window must be focused):
    q  -  quit: stops capture, releases the camera, closes the window.
    r  -  reset: clears the recurrent hidden state and re-initializes the
          optimizer (fresh Adam/SGD state), without restarting the process.
          Use this if training visibly diverges or goes unstable.

What you'll see:
    One window, two panes side by side: [ real webcam frame | model's
    current next-frame prediction ], with an overlay of frame count, FPS,
    and rolling-average training loss. The right pane should start out
    noisy/blurry and grow more temporally coherent over the first
    30-60 seconds as the model learns online, frame by frame.
"""

import time
from collections import deque

import cv2
import numpy as np
import torch

from config import build_arg_parser
from model import NextFramePredictor, detach_hidden, get_loss_fn
from webcam import WebcamStream


def preprocess(frame_bgr, width, height, device):
    small = cv2.resize(frame_bgr, (width, height), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).float().div_(255.0)
    return tensor.permute(2, 0, 1).unsqueeze(0).to(device)


def tensor_to_bgr(tensor, disp_w, disp_h):
    frame = tensor.detach().squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy()
    frame = (frame * 255.0).astype(np.uint8)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    return cv2.resize(frame, (disp_w, disp_h), interpolation=cv2.INTER_NEAREST)


def make_optimizer(name, params, lr):
    name = name.lower()
    if name == "adam":
        return torch.optim.Adam(params, lr=lr)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr)
    raise ValueError(f"Unknown optimizer: {name!r}")


def main():
    args = build_arg_parser().parse_args()

    if args.width % 4 != 0 or args.height % 4 != 0:
        raise ValueError("--width and --height must both be divisible by 4 "
                          "(the encoder/decoder each downsample/upsample by 2x twice).")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[krystalball] device: {device}")

    model = NextFramePredictor(hidden_channels=args.hidden_channels).to(device)
    optimizer = make_optimizer(args.optimizer, model.parameters(), args.lr)
    loss_fn = get_loss_fn(args.loss)

    webcam = WebcamStream(args.camera_index)
    print("[krystalball] waiting for first webcam frame...")
    while webcam.read() is None:
        time.sleep(0.01)

    disp_w, disp_h = args.width * args.upscale, args.height * args.upscale
    window_name = "krystalball -- real | prediction"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    hidden = None
    pred_frame = None  # prediction made *last* iteration, awaiting this iteration's ground truth
    loss_history = deque(maxlen=args.loss_window)
    frame_count = 0
    fps = 0.0
    last_tick = time.time()

    try:
        while True:
            raw_frame = webcam.read()
            if raw_frame is None:
                continue
            real_tensor = preprocess(raw_frame, args.width, args.height, device)

            # --- predict-then-learn, one step behind ---
            # `pred_frame` was produced last iteration, before this real frame
            # existed. Now that the real frame has arrived, score that earlier
            # prediction against it, backprop through the single timestep that
            # produced it, and step the optimizer.
            if pred_frame is not None:
                loss = loss_fn(pred_frame, real_tensor)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                loss_history.append(loss.item())
                # Cut the graph here so next timestep's backward can't walk
                # back through this one -- backprop spans exactly one step.
                hidden = detach_hidden(hidden)

            # Now feed the just-arrived real frame in to advance temporal
            # state and produce the prediction for the *next* frame (this
            # is the pred_frame that next iteration's loss will score).
            pred_frame, hidden = model(real_tensor, hidden)

            # --- display ---
            real_disp = cv2.resize(raw_frame, (disp_w, disp_h), interpolation=cv2.INTER_LINEAR)
            pred_disp = tensor_to_bgr(pred_frame, disp_w, disp_h)
            canvas = np.hstack([real_disp, pred_disp])

            frame_count += 1
            now = time.time()
            dt = now - last_tick
            last_tick = now
            if dt > 0:
                inst_fps = 1.0 / dt
                fps = inst_fps if frame_count == 1 else 0.9 * fps + 0.1 * inst_fps
            avg_loss = sum(loss_history) / len(loss_history) if loss_history else 0.0

            cv2.putText(canvas, f"frame {frame_count}  fps {fps:5.1f}  loss {avg_loss:.5f}",
                        (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(canvas, "real", (10, disp_h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(canvas, "prediction", (disp_w + 10, disp_h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.imshow(window_name, canvas)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r"):
                print("[krystalball] reset: clearing hidden state + optimizer state")
                hidden = None
                pred_frame = None
                optimizer = make_optimizer(args.optimizer, model.parameters(), args.lr)
                loss_history.clear()
    finally:
        webcam.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
