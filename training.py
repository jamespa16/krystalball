"""The online predict-then-learn training loop, ported from the old main.py
into a background-thread class so it can run independently of the FastAPI
server's asyncio event loop.

Frame arrival used to be a synchronous cv2.VideoCapture read (webcam.py);
it's now an async WebSocket message. TrainingEngine bridges the two with
LatestSlot, a small lock-protected single-item mailbox that generalizes
webcam.py's old "background thread + lock-protected latest-frame" pattern to
an arbitrary producer (the async WebSocket handler instead of cap.read()).
"""

import threading
import time
from collections import deque

import torch

import frame_codec
from model import (
    NextFramePredictor,
    detach_hidden,
    flow_smoothness_loss,
    get_loss_fn,
    motion_delta_loss,
    motion_weight_map,
)


class LatestSlot:
    """Lock-protected single-item mailbox.

    put() always overwrites -- an unconsumed previous item is simply lost,
    same as WebcamStream overwriting self._frame faster than it's read.
    get() consumes: it returns the current item and clears the slot, so a
    consumer polling faster than the producer sees None (rather than
    reprocessing the same item repeatedly) until a fresh item arrives.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._item = None

    def put(self, item):
        with self._lock:
            self._item = item

    def get(self):
        with self._lock:
            item, self._item = self._item, None
            return item


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
        # Rebuilding this (at startup and on every reset) restarts the ramp
        # from step 0, which matters because the hidden state is most
        # vulnerable to fast-moving weights right when it's freshest.
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_start_factor, end_factor=1.0,
            total_iters=warmup_steps,
        )
    return optimizer, scheduler


def compute_training_loss(loss_fn, pred, target, prev_frame, flow, args):
    """Composes the swappable base photometric loss with the two optional
    additive auxiliary terms (flow smoothness, motion-delta magnitude),
    shared by both the delay=0 and buffered training branches."""
    weight_map = None
    if args.motion_loss_weight > 0:
        weight_map = motion_weight_map(target, prev_frame, args.motion_loss_weight)
    loss = loss_fn(pred, target, weight_map=weight_map)
    if args.flow_smoothness_weight > 0 and flow is not None:
        loss = loss + args.flow_smoothness_weight * flow_smoothness_loss(flow, prev_frame)
    if args.motion_delta_weight > 0:
        loss = loss + args.motion_delta_weight * motion_delta_loss(pred, target, prev_frame)
    return loss


class TrainingEngine:
    """Owns the model/optimizer/hidden-state and runs the predict-then-learn
    loop on a dedicated background thread for the lifetime of the process."""

    def __init__(self, args, device):
        self.args = args
        self.device = device
        self.width = args.width
        self.height = args.height

        self.model = NextFramePredictor(
            encoder_base_channels=args.encoder_base_channels,
            encoder_scales=args.encoder_scales,
            res_blocks_per_scale=args.res_blocks_per_scale,
            lstm_hidden_channels=args.lstm_hidden_channels,
            lstm_layers=args.lstm_layers,
            kernel_size=args.lstm_kernel_size,
            delta_scale=args.delta_scale,
            use_flow=args.use_flow == "on",
            flow_hidden_channels=args.flow_hidden_channels,
            use_blend_mask=args.blend_mask == "on",
            skip_lstm_base_channels=args.skip_lstm_base_channels,
        ).to(device)
        self.optimizer, self.lr_scheduler = make_optimizer_and_scheduler(
            args.optimizer, self.model.parameters(), args.lr,
            args.lr_warmup_steps, args.lr_warmup_start_factor,
        )
        self.loss_fn = get_loss_fn(args.loss, ssim_weight=args.ssim_weight)

        # `train_hidden`/`train_pred` are the real training path: fed only
        # genuine (buffered) frames, never the model's own predictions.
        self.train_hidden = None
        self.train_pred = None  # prediction awaiting the next buffered ground-truth frame
        self.train_flow = None  # flow field from the last train-path forward call
        self.prev_train_frame = None  # previous training-consumed frame, for motion-weighted loss
        # `display_hidden`/`display_pred` drive the on-screen preview only
        # when buffering is active; periodically reseeded with a real frame
        # and self-fed in between. Purely cosmetic -- no gradient ever flows
        # through this path.
        self.display_hidden = None
        self.display_pred = None
        # Replay buffer: real frames waiting to be trained on. At delay=0
        # it's unused; above 0 it holds roughly `target_lag` frames, drained
        # one-per-iteration so every real frame is eventually trained on.
        self._buffer = deque()
        self._loss_history = deque(maxlen=args.loss_window)
        self.frame_count = 0
        self.fps = 0.0
        self._last_tick = time.time()

        self._intake = LatestSlot()  # browser -> engine, real frames (delay=0 path)
        self._output = LatestSlot()  # engine -> browser, (prediction_jpeg, stats)
        self._lock = threading.Lock()  # guards _target_lag only
        self._target_lag = args.real_frame_interval_frames
        self._reset_requested = threading.Event()
        self._paused = threading.Event()
        self._shutdown = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def stop_and_join(self, timeout=2.0):
        self._shutdown.set()
        self._thread.join(timeout=timeout)

    def submit_frame(self, jpeg_bytes):
        """Called from the async WebSocket receive handler. Any inbound
        frame implicitly resumes training if it was paused."""
        self._paused.clear()
        self._intake.put(jpeg_bytes)

    def set_delay(self, frames):
        frames = max(0, min(int(frames), self.args.real_frame_interval_max_frames))
        with self._lock:
            self._target_lag = frames

    def _get_target_lag(self):
        with self._lock:
            return self._target_lag

    def request_reset(self):
        self._reset_requested.set()

    def pause(self):
        self._paused.set()

    def pop_latest_output(self):
        """Returns (prediction_jpeg_bytes, stats_dict) or None."""
        return self._output.get()

    def _do_reset(self):
        print("[krystalball] reset: clearing hidden/replay-buffer/optimizer state")
        self.train_hidden = None
        self.train_pred = None
        self.train_flow = None
        self.prev_train_frame = None
        self.display_hidden = None
        self.display_pred = None
        self._buffer.clear()
        self.optimizer, self.lr_scheduler = make_optimizer_and_scheduler(
            self.args.optimizer, self.model.parameters(), self.args.lr,
            self.args.lr_warmup_steps, self.args.lr_warmup_start_factor,
        )
        self._loss_history.clear()

    def _run(self):
        args = self.args
        while not self._shutdown.is_set():
            if self._paused.is_set():
                time.sleep(0.02)
                continue

            jpeg = self._intake.get()
            if jpeg is None:
                time.sleep(0.005)
                continue
            try:
                real_tensor = frame_codec.decode_jpeg_to_tensor(
                    jpeg, self.width, self.height, self.device
                )
            except ValueError:
                continue

            # --- real-frame interval: how far training lags behind live video ---
            # target_lag is the replay-buffer delay in frames (0 = original
            # 1:1 behavior); above 0, real frames are pushed into a replay
            # buffer instead, and training drains that buffer at one frame
            # per iteration, so it lags real time by exactly `target_lag`
            # frames but still eventually trains on every single frame.
            target_lag = self._get_target_lag()
            interval_frames = target_lag + 1

            if interval_frames == 1:
                # --- predict-then-learn, one step behind (no buffering) ---
                if self.train_pred is not None:
                    loss = compute_training_loss(
                        self.loss_fn, self.train_pred, real_tensor,
                        self.prev_train_frame, self.train_flow, args,
                    )
                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    if args.grad_clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.grad_clip_norm)
                    self.optimizer.step()
                    if self.lr_scheduler is not None:
                        self.lr_scheduler.step()
                    self._loss_history.append(loss.item())
                    # Cut the graph so next timestep's backward can't walk
                    # back through this one -- backprop spans exactly one step.
                    self.train_hidden = detach_hidden(self.train_hidden)
                self.train_pred, self.train_hidden, self.train_flow = self.model(
                    real_tensor, self.train_hidden
                )
                self.prev_train_frame = real_tensor
                pred_for_display = self.train_pred
                is_anchor_step = True
            else:
                # --- replay buffer: train on every frame, just delayed ---
                self._buffer.append(real_tensor)
                # `while`, not `if`: normally pops at most once per iteration,
                # but if the delay was just lowered mid-run, `while` lets the
                # backlog catch back down over the next few iterations.
                while len(self._buffer) > target_lag:
                    ground_truth = self._buffer.popleft()
                    if self.train_pred is not None:
                        loss = compute_training_loss(
                            self.loss_fn, self.train_pred, ground_truth,
                            self.prev_train_frame, self.train_flow, args,
                        )
                        self.optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        if args.grad_clip_norm > 0:
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.grad_clip_norm)
                        self.optimizer.step()
                        if self.lr_scheduler is not None:
                            self.lr_scheduler.step()
                        self._loss_history.append(loss.item())
                        self.train_hidden = detach_hidden(self.train_hidden)
                    self.train_pred, self.train_hidden, self.train_flow = self.model(
                        ground_truth, self.train_hidden
                    )
                    self.prev_train_frame = ground_truth

                # --- cosmetic preview: periodic anchor + self-feeding rollout ---
                # Purely for the display pane; runs under no_grad and never
                # touches train_hidden/train_pred, so it can't affect learning.
                is_anchor_step = self.frame_count % interval_frames == 0
                with torch.no_grad():
                    disp_input = (
                        real_tensor
                        if (is_anchor_step or self.display_pred is None)
                        else self.display_pred.clamp(0, 1)
                    )
                    self.display_pred, self.display_hidden, _ = self.model(
                        disp_input, self.display_hidden
                    )
                pred_for_display = self.display_pred

            self.frame_count += 1
            now = time.time()
            dt = now - self._last_tick
            self._last_tick = now
            if dt > 0:
                inst_fps = 1.0 / dt
                self.fps = inst_fps if self.frame_count == 1 else 0.9 * self.fps + 0.1 * inst_fps
            avg_loss = sum(self._loss_history) / len(self._loss_history) if self._loss_history else 0.0
            mode_label = "REAL" if is_anchor_step else "FREE-RUN"

            pred_jpeg = frame_codec.tensor_to_jpeg(pred_for_display)
            stats = {
                "type": "stats",
                "frame_count": self.frame_count,
                "fps": round(self.fps, 1),
                "avg_loss": avg_loss,
                "mode_label": mode_label,
                "buffer_len": len(self._buffer),
                "target_lag": target_lag,
            }
            self._output.put((pred_jpeg, stats))

            if self._reset_requested.is_set():
                self._do_reset()
                self._reset_requested.clear()
