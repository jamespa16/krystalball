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
    Discriminator,
    NextFramePredictor,
    detach_hidden,
    flow_smoothness_loss,
    get_loss_fn,
    lsgan_d_loss,
    lsgan_g_loss,
    motion_delta_loss,
    motion_weight_map,
    world_model_consistency_loss,
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


def compute_training_loss(loss_fn, pred, target, prev_frame, flow, flow_velocity,
                           bottleneck_latent, world_model_target, world_model_head,
                           discriminator, adv_weight_eff, args):
    """Composes the swappable base photometric loss with the optional
    additive auxiliary terms (flow smoothness, flow-acceleration
    smoothness, motion-delta magnitude, world-model latent consistency,
    adversarial), shared by both the delay=0 and buffered training
    branches.

    `bottleneck_latent`/`world_model_target` are one tick STALE relative to
    `pred`/`target` here -- both come from the SAME earlier forward call
    that produced `pred` (see TrainingEngine._predict_and_learn), so this
    combines into the same single backward() as the pixel loss rather than
    a second backward() through that already-used graph.

    The adversarial term does NOT detach `pred`: gradient must flow from
    the discriminator's verdict back into the generator. Backpropagating
    through `discriminator`'s own forward graph as a side effect populates
    stray `.grad` on ITS parameters too -- harmless only because the
    discriminator's own optimizer step (in TrainingEngine._predict_and_learn)
    unconditionally zero_grad()s before its own separate backward pass."""
    weight_map = None
    if args.motion_loss_weight > 0:
        weight_map = motion_weight_map(target, prev_frame, args.motion_loss_weight)
    loss = loss_fn(pred, target, weight_map=weight_map)
    if args.flow_smoothness_weight > 0 and flow is not None:
        loss = loss + args.flow_smoothness_weight * flow_smoothness_loss(flow, prev_frame)
    if args.flow_accel_smoothness_weight > 0 and flow_velocity is not None:
        loss = loss + args.flow_accel_smoothness_weight * flow_smoothness_loss(flow_velocity, prev_frame)
    if args.motion_delta_weight > 0:
        loss = loss + args.motion_delta_weight * motion_delta_loss(pred, target, prev_frame)
    if args.world_model_weight > 0 and world_model_target is not None:
        pred_latent = world_model_head(bottleneck_latent, flow, flow_velocity)
        loss = loss + args.world_model_weight * world_model_consistency_loss(pred_latent, world_model_target)
    if adv_weight_eff > 0:
        loss = loss + adv_weight_eff * lsgan_g_loss(discriminator(pred))
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
            use_flow_acceleration=args.flow_acceleration == "on",
            st_memory_channels=args.st_memory_channels,
            world_model_hidden_channels=args.world_model_hidden_channels,
        ).to(device)
        self.optimizer, self.lr_scheduler = make_optimizer_and_scheduler(
            args.optimizer, self.model.parameters(), args.lr,
            args.lr_warmup_steps, args.lr_warmup_start_factor,
        )
        self.loss_fn = get_loss_fn(args.loss, ssim_weight=args.ssim_weight)

        # Adversarial discriminator: lives directly on TrainingEngine, NOT as
        # a NextFramePredictor submodule -- it must never be in
        # self.model.parameters(), or the generator's optimizer step would
        # also update the discriminator using the generator's own gradients,
        # defeating the adversarial setup entirely.
        self.discriminator = Discriminator(
            in_channels=3, base_channels=args.disc_base_channels, num_layers=args.disc_layers,
        ).to(device)
        self.disc_optimizer = make_optimizer(args.optimizer, self.discriminator.parameters(), args.disc_lr)
        self._adv_step_count = 0  # drives the adversarial-weight warmup ramp, restarted on reset

        # `train_hidden`/`train_pred` are the real training path: fed only
        # genuine (buffered) frames, never the model's own predictions.
        self.train_hidden = None
        self.train_pred = None  # prediction awaiting the next buffered ground-truth frame
        self.train_flow = None  # flow field from the last train-path forward call
        self.train_flow_velocity = None  # flow-velocity field from the last train-path forward call
        self.prev_train_frame = None  # previous training-consumed frame, for motion-weighted loss
        # World-model loss stash: this tick's bottleneck latent + a real
        # future frame's encoder embedding (peeked from _buffer, never
        # popped), consumed one tick later in the SAME backward as the
        # pixel loss -- see compute_training_loss / _predict_and_learn.
        self.pending_bottleneck_latent = None
        self.pending_world_model_target = None
        self._world_model_active = None  # tri-state so __init__'s first check always logs
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
        self._log_world_model_status(self._target_lag)

    def _log_world_model_status(self, target_lag):
        """The world-model loss (see compute_training_loss) sources its
        lookahead target by peeking the replay buffer, so it's only ever
        active while the delay slider is >= --world-model-horizon --
        including never at the default delay of 0. Log transitions so
        that contingency is visible rather than a silent no-op."""
        if self.args.world_model_weight <= 0:
            return
        active = target_lag >= self.args.world_model_horizon
        if active == self._world_model_active:
            return
        self._world_model_active = active
        if active:
            print(f"[krystalball] world-model loss active: real-frame delay ({target_lag}) "
                  f">= --world-model-horizon ({self.args.world_model_horizon})")
        else:
            print(f"[krystalball] world-model loss inactive: real-frame delay ({target_lag}) "
                  f"< --world-model-horizon ({self.args.world_model_horizon})")

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
        self._log_world_model_status(frames)

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
        self.train_flow_velocity = None
        self.prev_train_frame = None
        self.pending_bottleneck_latent = None
        self.pending_world_model_target = None
        self.display_hidden = None
        self.display_pred = None
        self._buffer.clear()
        self.optimizer, self.lr_scheduler = make_optimizer_and_scheduler(
            self.args.optimizer, self.model.parameters(), self.args.lr,
            self.args.lr_warmup_steps, self.args.lr_warmup_start_factor,
        )
        # Discriminator WEIGHTS are preserved (same convention as the
        # predictor's) -- only its optimizer momentum is reset. The
        # adversarial-weight warmup restarts too, for the same reason the LR
        # warmup does: a freshly-reset, under-trained generator shouldn't
        # immediately face a long-trained discriminator at full strength.
        self.disc_optimizer = make_optimizer(
            self.args.optimizer, self.discriminator.parameters(), self.args.disc_lr
        )
        self._adv_step_count = 0
        self._loss_history.clear()

    def _predict_and_learn(self, ground_truth, world_model_horizon=None):
        """One predict-then-learn step against a genuine ground-truth frame
        (either the live frame directly, or one popped off the replay
        buffer) -- shared by both the delay=0 and buffered branches of
        `_run` below, which used to duplicate this block verbatim. If a
        prediction is already pending (from the previous call), score it
        against `ground_truth`, backprop, step the optimizer, and detach
        hidden state (backprop spans exactly one timestep); then run the
        model forward on `ground_truth` to produce the next pending
        prediction.

        `world_model_horizon` (frames), when given and the replay buffer
        already holds that many frames beyond `ground_truth`, peeks
        (never pops) the buffer for a real future frame, encodes it
        (stateless, no_grad) as the world-model loss's target, and stashes
        it alongside this call's bottleneck latent for consumption -- and
        backward() -- one tick later, together with the pixel loss (see
        compute_training_loss)."""
        if self.train_pred is not None:
            # Adversarial-weight warmup, mirroring the LR-warmup pattern but
            # as a manually tracked ramp (it scales a loss weight, not an
            # LR, so LinearLR doesn't directly apply): ramps 0 -> adv_weight
            # over adv_warmup_steps steps, restarted on reset. Mitigates the
            # classic GAN cold-start problem -- an untrained predictor's
            # early noisy output would otherwise give the discriminator a
            # trivial win with near-zero useful gradient back to the
            # generator.
            self._adv_step_count += 1
            adv_weight_eff = self.args.adv_weight * min(
                1.0, self._adv_step_count / max(1, self.args.adv_warmup_steps)
            )
            loss = compute_training_loss(
                self.loss_fn, self.train_pred, ground_truth,
                self.prev_train_frame, self.train_flow, self.train_flow_velocity,
                self.pending_bottleneck_latent, self.pending_world_model_target,
                self.model.world_model_head, self.discriminator, adv_weight_eff, self.args,
            )
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if self.args.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip_norm)
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            self._loss_history.append(loss.item())
            # Cut the graph so next timestep's backward can't walk
            # back through this one -- backprop spans exactly one step.
            self.train_hidden = detach_hidden(self.train_hidden)

            # --- discriminator step: fresh forward passes on (real, fake) ---
            # `self.train_pred` here is still the JUST-SCORED pending
            # prediction (reassigned to the new one below) -- exactly the
            # (ground_truth, pred) pair the generator step above judged.
            self.disc_optimizer.zero_grad(set_to_none=True)
            d_loss = lsgan_d_loss(
                self.discriminator(ground_truth), self.discriminator(self.train_pred.detach())
            )
            d_loss.backward()
            if self.args.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.args.grad_clip_norm)
            self.disc_optimizer.step()
        output = self.model(ground_truth, self.train_hidden)
        self.train_pred, self.train_hidden = output.pred, output.hidden
        self.train_flow, self.train_flow_velocity = output.flow, output.flow_velocity
        self.prev_train_frame = ground_truth

        if world_model_horizon and len(self._buffer) >= world_model_horizon:
            peek_frame = self._buffer[world_model_horizon - 1]
            with torch.no_grad():
                target_latent = self.model.encode_frame(peek_frame)
            self.pending_bottleneck_latent = output.bottleneck_latent
            self.pending_world_model_target = target_latent
        else:
            self.pending_bottleneck_latent = None
            self.pending_world_model_target = None

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
                self._predict_and_learn(real_tensor)
                pred_for_display = self.train_pred
                is_anchor_step = True
            else:
                # --- replay buffer: train on every frame, just delayed ---
                self._buffer.append(real_tensor)
                # World-model loss (see compute_training_loss) sources its
                # lookahead target by peeking this same buffer, so it's only
                # reachable in this branch -- never at delay=0, where there's
                # no buffer to peek ahead into.
                world_model_horizon = args.world_model_horizon if args.world_model_weight > 0 else None
                # `while`, not `if`: normally pops at most once per iteration,
                # but if the delay was just lowered mid-run, `while` lets the
                # backlog catch back down over the next few iterations.
                while len(self._buffer) > target_lag:
                    ground_truth = self._buffer.popleft()
                    self._predict_and_learn(ground_truth, world_model_horizon)

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
                    disp_output = self.model(disp_input, self.display_hidden)
                    self.display_pred, self.display_hidden = disp_output.pred, disp_output.hidden
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
