"""
krystalball -- live webcam next-frame prediction, trained online in real time.

Usage:
    uv sync
    uv run main.py [--camera-index 0] [--width 96] [--height 72] [--upscale 6]
                   [--lr 1e-3] [--lstm-hidden-channels 128] [--optimizer adam]
                   [--loss mse_ssim] [--real-frame-interval-frames 0]
    (run `uv run main.py --help` for the full list of flags)

Keyboard controls (window must be focused):
    q  -  quit: stops capture, releases the camera, closes the window.
    r  -  reset: clears the recurrent hidden state and re-initializes the
          optimizer (fresh Adam/SGD state), without restarting the process.
          Use this if training visibly diverges or goes unstable.

On-screen slider:
    "Real frame delay (frames)" -- snaps to whole-frame detents; its value
    is shown live both on the slider and in the on-screen overlay ("delay
    Nf"). At 0 (default) every frame is trained on immediately, the
    original 1:1 predict-then-learn behavior. Dragged higher, incoming
    real frames are pushed into a small in-memory replay buffer instead of
    being trained on right away; a separate training pass drains that
    buffer at a steady one-frame-per-iteration pace, so every real frame
    is still eventually trained on -- just delayed by exactly that many
    frames, never discarded. The right (prediction) pane keeps showing a
    live, purely cosmetic self-feeding rollout (reseeded with the real
    frame periodically) so you can watch the model's "imagination" drift
    between anchors; that rollout never affects training, which only ever
    learns from real buffered frames.

What you'll see:
    One window, two panes side by side: [ real webcam frame | model's
    current prediction ], with an overlay of frame count, FPS,
    rolling-average training loss, replay-buffer backlog, and whether the
    display is currently anchored to a REAL frame or FREE-RUNning. The
    right pane should start out noisy/blurry and grow more temporally
    coherent over the first 30-60 seconds as the model learns online.
"""

import time
from collections import deque

import cv2
import numpy as np
import torch

from config import build_arg_parser
from model import NextFramePredictor, detach_hidden, get_loss_fn, motion_weight_map
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


def make_optimizer_and_scheduler(name, params, lr, warmup_steps, warmup_start_factor):
    optimizer = make_optimizer(name, params, lr)
    scheduler = None
    if warmup_steps > 0:
        # Ramps optimizer's lr from warmup_start_factor*lr up to lr over the
        # first `warmup_steps` calls to scheduler.step(), then holds at lr.
        # Rebuilding this (at startup and on every 'r' reset) restarts the
        # ramp from step 0, which matters because the hidden state is most
        # vulnerable to fast-moving weights right when it's freshest.
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_start_factor, end_factor=1.0,
            total_iters=warmup_steps,
        )
    return optimizer, scheduler


def main():
    args = build_arg_parser().parse_args()

    divisor = 2 ** args.encoder_scales
    if args.width % divisor != 0 or args.height % divisor != 0:
        raise ValueError(
            f"--width and --height must both be divisible by {divisor} "
            f"(the encoder/decoder downsample/upsample by 2x, {args.encoder_scales} times)."
        )

    if args.device == "auto":
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.mps.is_available()
            else "cpu"
        )
    else:
        device = torch.device(args.device)
    print(f"[krystalball] device: {device}")

    model = NextFramePredictor(
        encoder_base_channels=args.encoder_base_channels,
        encoder_scales=args.encoder_scales,
        res_blocks_per_scale=args.res_blocks_per_scale,
        lstm_hidden_channels=args.lstm_hidden_channels,
        lstm_layers=args.lstm_layers,
        delta_scale=args.delta_scale,
        use_flow=args.use_flow == "on",
        flow_hidden_channels=args.flow_hidden_channels,
        use_blend_mask=args.blend_mask == "on",
    ).to(device)
    optimizer, lr_scheduler = make_optimizer_and_scheduler(
        args.optimizer, model.parameters(), args.lr,
        args.lr_warmup_steps, args.lr_warmup_start_factor,
    )
    loss_fn = get_loss_fn(args.loss, ssim_weight=args.ssim_weight)

    webcam = WebcamStream(args.camera_index)
    print("[krystalball] waiting for first webcam frame...")
    while webcam.read() is None:
        time.sleep(0.01)

    disp_w, disp_h = args.width * args.upscale, args.height * args.upscale
    window_name = "krystalball -- real | prediction"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    trackbar_name = "Real frame delay (frames)"
    cv2.createTrackbar(
        trackbar_name,
        window_name,
        args.real_frame_interval_frames,
        args.real_frame_interval_max_frames,
        lambda _v: None,
    )

    # `train_hidden`/`train_pred` are the real training path: fed only genuine
    # (buffered) frames, never the model's own predictions.
    train_hidden = None
    train_pred = None  # prediction awaiting the next buffered ground-truth frame
    prev_train_frame = None  # previous training-consumed frame, for motion-weighted loss
    # `display_hidden`/`display_pred` drive the on-screen preview only when
    # buffering is active (delay > 0 frames); periodically reseeded with a
    # real frame and self-fed in between. Purely cosmetic -- no gradient
    # ever flows through this path.
    display_hidden = None
    display_pred = None
    # Replay buffer: real frames waiting to be trained on. At delay=0 it's
    # unused (frames are trained on immediately); above 0 it holds roughly
    # `target_lag` frames, drained one-per-iteration so every real frame is
    # eventually trained on, just delayed.
    buffer = deque()
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

            # --- real-frame interval: how far training lags behind live video ---
            # The trackbar snaps to whole-frame detents and its position *is*
            # the target lag in frames -- at 0 (default) every frame is
            # trained on immediately, i.e. the original 1:1 behavior (no
            # buffering at all). Above 0, real frames are pushed into a
            # replay buffer instead, and training drains that buffer at one
            # frame per iteration, so it lags real time by exactly
            # `target_lag` frames but still eventually trains on every single
            # frame -- nothing is discarded.
            target_lag = cv2.getTrackbarPos(trackbar_name, window_name)
            interval_frames = target_lag + 1

            if interval_frames == 1:
                # --- predict-then-learn, one step behind (no buffering) ---
                if train_pred is not None:
                    weight_map = None
                    if args.motion_loss_weight > 0:
                        weight_map = motion_weight_map(
                            real_tensor, prev_train_frame, args.motion_loss_weight
                        )
                    loss = loss_fn(train_pred, real_tensor, weight_map=weight_map)
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    if args.grad_clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
                    optimizer.step()
                    if lr_scheduler is not None:
                        lr_scheduler.step()
                    loss_history.append(loss.item())
                    # Cut the graph so next timestep's backward can't walk
                    # back through this one -- backprop spans exactly one step.
                    train_hidden = detach_hidden(train_hidden)
                train_pred, train_hidden = model(real_tensor, train_hidden)
                prev_train_frame = real_tensor
                pred_for_display = train_pred
                is_anchor_step = True
            else:
                # --- replay buffer: train on every frame, just delayed ---
                buffer.append(real_tensor)
                # `while`, not `if`: normally this pops at most once per
                # iteration (steady one-frame-per-iteration drain, matching
                # the one frame just appended). But if the slider is dragged
                # to a smaller interval mid-run, target_lag drops and the
                # backlog is briefly larger than the new target -- `while`
                # lets it catch back down over the next few iterations
                # instead of staying oversized forever.
                while len(buffer) > target_lag:
                    ground_truth = buffer.popleft()
                    if train_pred is not None:
                        weight_map = None
                        if args.motion_loss_weight > 0:
                            weight_map = motion_weight_map(
                                ground_truth, prev_train_frame, args.motion_loss_weight
                            )
                        loss = loss_fn(train_pred, ground_truth, weight_map=weight_map)
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        if args.grad_clip_norm > 0:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
                        optimizer.step()
                        if lr_scheduler is not None:
                            lr_scheduler.step()
                        loss_history.append(loss.item())
                        # Same one-timestep-only invariant as above, just
                        # applied to a delayed (buffered) ground truth frame
                        # instead of the live one.
                        train_hidden = detach_hidden(train_hidden)
                    train_pred, train_hidden = model(ground_truth, train_hidden)
                    prev_train_frame = ground_truth

                # --- cosmetic preview: periodic anchor + self-feeding rollout ---
                # Purely for the display pane; runs under no_grad and never
                # touches train_hidden/train_pred, so it can't affect learning.
                is_anchor_step = frame_count % interval_frames == 0
                with torch.no_grad():
                    disp_input = (
                        real_tensor
                        if (is_anchor_step or display_pred is None)
                        # Clamp before self-feeding: the decoder's residual
                        # output can drift outside [0, 1], and feeding an
                        # out-of-distribution frame back in compounds across
                        # rollout steps, so keep it on the manifold the model
                        # was actually trained on.
                        else display_pred.clamp(0, 1)
                    )
                    display_pred, display_hidden = model(disp_input, display_hidden)
                pred_for_display = display_pred

            # --- display ---
            real_disp = cv2.resize(
                raw_frame, (disp_w, disp_h), interpolation=cv2.INTER_LINEAR
            )
            pred_disp = tensor_to_bgr(pred_for_display, disp_w, disp_h)
            canvas = np.hstack([real_disp, pred_disp])

            frame_count += 1
            now = time.time()
            dt = now - last_tick
            last_tick = now
            if dt > 0:
                inst_fps = 1.0 / dt
                fps = inst_fps if frame_count == 1 else 0.9 * fps + 0.1 * inst_fps
            avg_loss = sum(loss_history) / len(loss_history) if loss_history else 0.0
            mode_label = "REAL" if is_anchor_step else "FREE-RUN"

            cv2.putText(
                canvas,
                f"frame {frame_count}  fps {fps:5.1f}  loss {avg_loss:.5f}  "
                f"[{mode_label}]  delay {target_lag}f  buf {len(buffer)}/{target_lag}",
                (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                canvas,
                "real",
                (10, disp_h - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                canvas,
                "prediction",
                (disp_w + 10, disp_h - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

            cv2.imshow(window_name, canvas)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r"):
                print("[krystalball] reset: clearing hidden/replay-buffer/optimizer state")
                train_hidden = None
                train_pred = None
                prev_train_frame = None
                display_hidden = None
                display_pred = None
                buffer.clear()
                optimizer, lr_scheduler = make_optimizer_and_scheduler(
                    args.optimizer, model.parameters(), args.lr,
                    args.lr_warmup_steps, args.lr_warmup_start_factor,
                )
                loss_history.clear()
    finally:
        webcam.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
