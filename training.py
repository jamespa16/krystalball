"""The online predict-then-learn training loop, ported from the old main.py
into a background-thread class so it can run independently of the FastAPI
server's asyncio event loop.

Frame arrival used to be a synchronous cv2.VideoCapture read (webcam.py);
it's now an async WebSocket message. TrainingEngine bridges the two with
LatestSlot, a small lock-protected single-item mailbox that generalizes
webcam.py's old "background thread + lock-protected latest-frame" pattern to
an arbitrary producer (the async WebSocket handler instead of cap.read()).
"""

import math
import os
import threading
import time
from collections import defaultdict, deque

import torch

import frame_codec
from model import (
    Discriminator,
    NextFramePredictor,
    detach_hidden,
    flow_smoothness_loss,
    get_loss_fn,
    latent_consistency_loss,
    lsgan_d_loss,
    lsgan_g_loss,
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


def compute_multistep_losses(model, loss_fn, seed_pred, seed_hidden, targets, decay=1.0):
    """Unified multistep self-consistency loss: self-feeds `seed_pred`
    through `len(targets)` more forward calls -- true BPTT, gradient flowing
    across the whole chain, no detach between steps -- and at EACH step
    scores the output BOTH ways against the same real future frame:
      - pixel:  loss_fn(output.pred, target) -- lets the loss teach the
                model to correct for its own compounding prediction errors,
                not just produce a good single next frame from a real input.
      - latent: latent_consistency_loss(model.project_latent(output.bottleneck_latent),
                                         model.encode_frame(target))
                -- scores the recurrent core's OWN predicted state directly
                against a real future frame's stop-gradient encoder
                embedding, rather than routing through a separate
                forecasting network (the old WorldModelHead this replaces).
                project_latent is only a channel-count adapter (bottleneck_
                latent and encode_frame's output generally live at different
                channel counts) -- it does no forward-in-time prediction of
                its own.
    Both accumulate with decay**i into two independent scalar sums sharing
    the chain's ONE autograd graph -- a single .backward() one tick later
    (see TrainingEngine._predict_and_learn) consumes both, plus the primary
    pixel loss. Returns (pixel_total, latent_total)."""
    hidden = seed_hidden
    inp = seed_pred
    pixel_total = 0.0
    latent_total = 0.0
    for i, target in enumerate(targets):
        output = model(inp, hidden)
        pixel_total = pixel_total + (decay ** i) * loss_fn(output.pred, target)
        with torch.no_grad():
            target_latent = model.encode_frame(target)
        latent_total = latent_total + (decay ** i) * latent_consistency_loss(
            model.project_latent(output.bottleneck_latent), target_latent
        )
        hidden, inp = output.hidden, output.pred.clamp(0, 1)
    return pixel_total, latent_total


def compute_training_loss(loss_fn, pred, target, prev_frame, flow, flow_velocity,
                           multistep_losses, discriminator, adv_ramp_fraction,
                           loss_weighter, args):
    """Composes the swappable base photometric loss with the optional
    additive auxiliary terms (flow smoothness, flow-acceleration
    smoothness, motion-delta magnitude, the two halves of the unified
    multistep self-consistency loss, adversarial), shared by both the
    delay=0 and buffered training branches.

    Each auxiliary term's contribution is `loss_weighter.weighted_term(name, raw)`
    -- learned homoscedastic uncertainty weighting (Kendall, Gal & Cipolla
    2018), not a static CLI-weight multiply -- see model.py's
    UncertaintyWeighter. A term is active iff `name in loss_weighter.log_vars`,
    which is decided once at construction from that term's CLI flag (0
    means "never construct a weight for this, it's fully disabled"); the
    base photometric loss itself is deliberately NOT run through the
    weighter and stays an unweighted anchor, so a runaway auxiliary weight
    can never zero out the primary prediction objective.

    `multistep_losses` is `(pixel_total, latent_total)` or None -- a
    one-tick-stale stash from compute_multistep_losses, both scalars
    sharing ONE chain's un-detached autograd graph (see
    TrainingEngine._predict_and_learn) -- this backward() is what finally
    consumes it, combined into the same single backward() as the pixel
    loss rather than a second backward() through an already-used graph.
    None when the chain wasn't built this tick (buffer too short /
    --multistep-every-n-steps cadence gate closed); both multistep_pixel
    and multistep_latent then report 0.0 in the breakdown -- meaningful (it
    shows how often the gate is open at the current delay setting), not a
    bug.

    `adv_ramp_fraction` (0..1, see the warmup comment in
    TrainingEngine._predict_and_learn) scales the WHOLE Kendall-weighted
    adversarial term, including its log-variance regularizer, not just the
    raw GAN loss -- scaling only the raw loss would make the log-variance's
    gradient behave as if the observed loss were ~0 during warmup, which
    drives that weight to its clamp ceiling immediately (the opposite of
    what warmup is for). Ramping the whole term keeps the learned weight
    near its CLI-derived initial value until the discriminator signal is
    real.

    The adversarial term does NOT detach `pred`: gradient must flow from
    the discriminator's verdict back into the generator. Backpropagating
    through `discriminator`'s own forward graph as a side effect populates
    stray `.grad` on ITS parameters too -- harmless only because the
    discriminator's own optimizer step (in TrainingEngine._predict_and_learn)
    unconditionally zero_grad()s before its own separate backward pass.

    Returns `(loss, breakdown)`: `breakdown` is a `{term_name: weighted_value}`
    dict with one entry per active term (regardless of whether it fired
    this particular tick), so callers can log each term's actual share of
    the total rather than only ever seeing the sum."""
    breakdown = {}
    weight_map = None
    if args.motion_loss_weight > 0:
        weight_map = motion_weight_map(target, prev_frame, args.motion_loss_weight)
    loss = loss_fn(pred, target, weight_map=weight_map)
    breakdown["base"] = loss.item()
    if "flow_smoothness" in loss_weighter.log_vars and flow is not None:
        term = loss_weighter.weighted_term("flow_smoothness", flow_smoothness_loss(flow, prev_frame))
        loss = loss + term
        breakdown["flow_smoothness"] = term.item()
    if "flow_accel_smoothness" in loss_weighter.log_vars and flow_velocity is not None:
        term = loss_weighter.weighted_term(
            "flow_accel_smoothness", flow_smoothness_loss(flow_velocity, prev_frame)
        )
        loss = loss + term
        breakdown["flow_accel_smoothness"] = term.item()
    if "motion_delta" in loss_weighter.log_vars:
        term = loss_weighter.weighted_term("motion_delta", motion_delta_loss(pred, target, prev_frame))
        loss = loss + term
        breakdown["motion_delta"] = term.item()
    if "multistep_pixel" in loss_weighter.log_vars:
        if multistep_losses is not None:
            term = loss_weighter.weighted_term("multistep_pixel", multistep_losses[0])
            loss = loss + term
            breakdown["multistep_pixel"] = term.item()
        else:
            breakdown["multistep_pixel"] = 0.0
    if "multistep_latent" in loss_weighter.log_vars:
        if multistep_losses is not None:
            term = loss_weighter.weighted_term("multistep_latent", multistep_losses[1])
            loss = loss + term
            breakdown["multistep_latent"] = term.item()
        else:
            breakdown["multistep_latent"] = 0.0
    if "adversarial" in loss_weighter.log_vars:
        raw = lsgan_g_loss(discriminator(pred))
        term = adv_ramp_fraction * loss_weighter.weighted_term("adversarial", raw)
        loss = loss + term
        breakdown["adversarial"] = term.item()
    return loss, breakdown


class TrainingEngine:
    """Owns the model/optimizer/hidden-state and runs the predict-then-learn
    loop on a dedicated background thread for the lifetime of the process."""

    def __init__(self, args, device):
        self.args = args
        self.device = device
        self.width = args.width
        self.height = args.height
        self.frame_count = 0  # set before load_checkpoint() below, which may override it

        # Static CLI weights for the auxiliary loss terms now only seed each
        # term's INITIAL learned-uncertainty weight (see model.py's
        # UncertaintyWeighter) -- omitting a term here (weight 0) still
        # fully disables it, unchanged from the old static-weight behavior.
        initial_weights = {
            "flow_smoothness": args.flow_smoothness_weight,
            "flow_accel_smoothness": args.flow_accel_smoothness_weight,
            "motion_delta": args.motion_delta_weight,
            "multistep_pixel": args.multistep_pixel_weight,
            "multistep_latent": args.multistep_latent_weight,
            "adversarial": args.adv_weight,
        }
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
            initial_weights=initial_weights,
            uncertainty_clamp=args.uncertainty_clamp,
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

        self._last_checkpoint_time = time.time()
        if not args.fresh_start:
            self.load_checkpoint()

        # `train_hidden`/`train_pred` are the real training path: fed only
        # genuine (buffered) frames, never the model's own predictions.
        self.train_hidden = None
        self.train_pred = None  # prediction awaiting the next buffered ground-truth frame
        self.train_flow = None  # flow field from the last train-path forward call
        self.train_flow_velocity = None  # flow-velocity field from the last train-path forward call
        self.prev_train_frame = None  # previous training-consumed frame, for motion-weighted loss
        # Unified multistep-consistency stash: a fully-composed, un-detached
        # (pixel_total, latent_total) pair built this tick by
        # compute_multistep_losses, consumed one tick later in the SAME
        # backward as the pixel loss -- see compute_training_loss /
        # _predict_and_learn.
        self.pending_multistep_losses = None
        self._multistep_step_count = 0  # drives the --multistep-every-n-steps cadence gate
        self._multistep_active = None  # tri-state so __init__'s first check always logs
        # Delay-driven look-ahead forecast: when buffering is active, the
        # prediction pane shows a genuine N-frame (N = target_lag) forecast
        # generated from the CURRENT (latest-trained) train_hidden/train_pred
        # under no_grad -- see generate() and _run's buffered branch. This
        # list is consumed one frame per real-frame tick; once exhausted, a
        # fresh forecast is generated using whatever weights training has
        # reached by then. No gradient ever flows through this path.
        self._display_rollout = []
        self._display_rollout_idx = 0
        # Replay buffer: real frames waiting to be trained on. At delay=0
        # it's unused; above 0 it holds roughly `target_lag` frames, drained
        # one-per-iteration so every real frame is eventually trained on.
        self._buffer = deque()
        self._loss_history = deque(maxlen=args.loss_window)
        # Per-term rolling history (see compute_training_loss's `breakdown`
        # return value), keyed lazily so only terms with a nonzero CLI
        # weight ever show up -- lets the stats readout show each loss
        # term's actual weighted contribution instead of only their sum.
        self._loss_component_history = defaultdict(lambda: deque(maxlen=args.loss_window))
        # Which auxiliary terms' learned weight (see model.py's
        # UncertaintyWeighter) are currently pinned near --uncertainty-clamp
        # -- tracked so _check_loss_weight_saturation only logs on
        # transitions, not every tick. The checkpoint (see save_checkpoint/
        # load_checkpoint below) doesn't persist this set, so this console
        # warning is still the only way to notice a term has run away during
        # an unattended session.
        self._saturated_loss_weights = set()
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
        self._log_multistep_status(self._target_lag)

    def _log_multistep_status(self, target_lag):
        """The unified multistep loss (see compute_multistep_losses) sources
        its lookahead targets by peeking the replay buffer, so it's only
        ever active while the delay slider is >= --multistep-horizon + 1 --
        including never at the default delay of 0. Log transitions so
        that contingency is visible rather than a silent no-op."""
        if self.args.multistep_pixel_weight <= 0 and self.args.multistep_latent_weight <= 0:
            return
        active = target_lag >= self.args.multistep_horizon + 1
        if active == self._multistep_active:
            return
        self._multistep_active = active
        state = "active" if active else "inactive"
        print(f"[krystalball] multistep loss {state}: real-frame delay ({target_lag}) "
              f"{'>=' if active else '<'} --multistep-horizon + 1 ({self.args.multistep_horizon + 1})")

    def _check_loss_weight_saturation(self, loss_weights):
        """Warn (on transitions only) when a learned auxiliary-loss weight
        (see model.py's UncertaintyWeighter) is pinned near either
        --uncertainty-clamp bound -- the checkpoint doesn't persist this
        history, so this live console warning is the only way to notice a
        term has run away during an unattended session."""
        clamp = self.args.uncertainty_clamp
        lo, hi = math.exp(-clamp), math.exp(clamp)
        # 90% of the way to either clamp bound, in log-var space (where the
        # clamp is actually applied) -- NOT 90% of the weight range, which
        # would be a very different (and wrong) threshold on this log scale.
        lo_thresh, hi_thresh = math.exp(-0.9 * clamp), math.exp(0.9 * clamp)
        currently_saturated = set()
        for name, weight in loss_weights.items():
            if weight <= lo_thresh or weight >= hi_thresh:
                currently_saturated.add(name)
        newly = currently_saturated - self._saturated_loss_weights
        cleared = self._saturated_loss_weights - currently_saturated
        for name in newly:
            print(f"[krystalball] WARNING: learned weight for '{name}' loss term saturated near "
                  f"--uncertainty-clamp bound (weight={loss_weights[name]:.4g}, range "
                  f"[{lo:.4g}, {hi:.4g}]) -- consider raising --uncertainty-clamp or revisiting "
                  f"that term's --*-weight init")
        for name in cleared:
            print(f"[krystalball] learned weight for '{name}' loss term back within normal range")
        self._saturated_loss_weights = currently_saturated

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
        self._log_multistep_status(frames)

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

    def generate(self, num_frames):
        """Open-loop look-ahead forecast: seeds from the CURRENT live
        training state (train_pred/train_hidden -- the same pending
        prediction the predict-then-learn loop is about to score against
        the next real frame) and self-feeds num_frames steps forward under
        no_grad, WITHOUT disturbing train_pred/train_hidden/train_flow/the
        optimizer -- a pure read. Called automatically by _run's buffered
        branch (num_frames = target_lag) to (re)generate the delay-driven
        display forecast each time the previous one is exhausted, so the
        prediction pane always shows the model's current best guess of
        what happens target_lag frames ahead, using the newest trained
        weights available at generation time. Returns [] if there's no
        seed yet (train_pred is None, e.g. before the first frame or right
        after a reset)."""
        if self.train_pred is None:
            return []
        num_frames = max(1, int(num_frames))
        frames = []
        with torch.no_grad():
            hidden, inp = self.train_hidden, self.train_pred.clamp(0, 1)
            for _ in range(num_frames):
                output = self.model(inp, hidden)
                frames.append(output.pred.clamp(0, 1))
                hidden, inp = output.hidden, output.pred.clamp(0, 1)
        return frames

    def _checkpoint_payload(self):
        return {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "discriminator": self.discriminator.state_dict(),
            "disc_optimizer": self.disc_optimizer.state_dict(),
            "frame_count": self.frame_count,
        }

    def save_checkpoint(self, path=None):
        """Persist model/optimizer/discriminator WEIGHTS to disk -- NOT
        hidden state, the replay buffer, or display state, matching
        _do_reset's "weights persist, transient state doesn't" convention.
        Atomic write via temp-file + os.replace, so a process killed
        mid-write can't leave a truncated/corrupt checkpoint file behind."""
        path = path or self.args.checkpoint_path
        tmp_path = f"{path}.tmp"
        torch.save(self._checkpoint_payload(), tmp_path)
        os.replace(tmp_path, path)
        print(f"[krystalball] checkpoint saved: {path} (frame_count={self.frame_count})")

    def load_checkpoint(self, path=None):
        """Called once from __init__ (unless --fresh-start) if the file
        exists. Restores WEIGHTS only -- hidden state/replay buffer/display
        state always start fresh regardless, same convention as save.
        Requires the checkpoint's loss_weighter terms (which auxiliary
        losses had nonzero initial weight) to match this run's flags --
        strict load_state_dict fails loudly on mismatch rather than
        silently corrupting; --fresh-start is the escape hatch if flags
        changed since the checkpoint was saved."""
        path = path or self.args.checkpoint_path
        if not os.path.exists(path):
            return False
        payload = torch.load(path, map_location=self.device)
        self.model.load_state_dict(payload["model"])
        self.optimizer.load_state_dict(payload["optimizer"])
        self.discriminator.load_state_dict(payload["discriminator"])
        self.disc_optimizer.load_state_dict(payload["disc_optimizer"])
        self.frame_count = payload.get("frame_count", 0)
        print(f"[krystalball] checkpoint loaded: {path} (frame_count={self.frame_count})")
        return True

    def _do_reset(self):
        print("[krystalball] reset: clearing hidden/replay-buffer/optimizer state")
        self.train_hidden = None
        self.train_pred = None
        self.train_flow = None
        self.train_flow_velocity = None
        self.prev_train_frame = None
        self.pending_multistep_losses = None
        self._multistep_step_count = 0
        self._display_rollout = []
        self._display_rollout_idx = 0
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
        self._loss_component_history.clear()

    def _predict_and_learn(self, ground_truth, multistep_horizon=None):
        """One predict-then-learn step against a genuine ground-truth frame
        (either the live frame directly, or one popped off the replay
        buffer) -- shared by both the delay=0 and buffered branches of
        `_run` below, which used to duplicate this block verbatim. If a
        prediction is already pending (from the previous call), score it
        against `ground_truth`, backprop, step the optimizer, and detach
        hidden state (backprop spans exactly one timestep); then run the
        model forward on `ground_truth` to produce the next pending
        prediction.

        `multistep_horizon` (K frames), when given and the replay buffer
        already holds K+1 frames beyond `ground_truth`, builds ONE
        true-BPTT self-feeding chain (see compute_multistep_losses) seeded
        from this call's own fresh prediction/hidden state, scored against
        buffer[1..K] (buffer[0] is skipped -- it's exactly what becomes
        next call's `ground_truth`, so scoring it here would double-weight
        the primary loop's own next-step term) BOTH as pixel loss and
        latent consistency, gated by a single --multistep-every-n-steps
        cadence counter. Stashed one-tick-stale, consumed together with
        the primary pixel loss one tick later (see compute_training_loss)."""
        if self.train_pred is not None:
            # Adversarial ramp fraction (0..1), mirroring the LR-warmup
            # pattern but as a manually tracked ramp (it scales a loss
            # weight, not an LR, so LinearLR doesn't directly apply): ramps
            # 0 -> 1 over adv_warmup_steps steps, restarted on reset.
            # Mitigates the classic GAN cold-start problem -- an untrained
            # predictor's early noisy output would otherwise give the
            # discriminator a trivial win with near-zero useful gradient
            # back to the generator. Scales the WHOLE learned-weight
            # adversarial term (see compute_training_loss's docstring), not
            # a fixed --adv-weight, since that's now only the term's
            # initial value under learned uncertainty weighting.
            self._adv_step_count += 1
            adv_ramp_fraction = min(1.0, self._adv_step_count / max(1, self.args.adv_warmup_steps))
            loss, loss_breakdown = compute_training_loss(
                self.loss_fn, self.train_pred, ground_truth,
                self.prev_train_frame, self.train_flow, self.train_flow_velocity,
                self.pending_multistep_losses, self.discriminator, adv_ramp_fraction,
                self.model.loss_weighter, self.args,
            )
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if self.args.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip_norm)
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            self._loss_history.append(loss.item())
            for term_name, term_value in loss_breakdown.items():
                self._loss_component_history[term_name].append(term_value)
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

        if multistep_horizon and len(self._buffer) >= multistep_horizon + 1:
            self._multistep_step_count += 1
            if self._multistep_step_count % self.args.multistep_every_n_steps == 0:
                targets = [self._buffer[i] for i in range(1, multistep_horizon + 1)]
                self.pending_multistep_losses = compute_multistep_losses(
                    self.model, self.loss_fn, output.pred, output.hidden, targets,
                    decay=self.args.multistep_decay,
                )
            else:
                self.pending_multistep_losses = None
        else:
            self.pending_multistep_losses = None

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
                mode_label = "REAL"
                forecast_step, forecast_horizon = 0, 0
            else:
                # --- replay buffer: train on every frame, just delayed ---
                self._buffer.append(real_tensor)
                # Unified multistep loss (see compute_multistep_losses)
                # sources its lookahead targets by peeking this same buffer,
                # so it's only reachable in this branch -- never at delay=0,
                # where there's no buffer to peek ahead into.
                multistep_horizon = (
                    args.multistep_horizon
                    if (args.multistep_pixel_weight > 0 or args.multistep_latent_weight > 0)
                    else None
                )
                # `while`, not `if`: normally pops at most once per iteration,
                # but if the delay was just lowered mid-run, `while` lets the
                # backlog catch back down over the next few iterations.
                while len(self._buffer) > target_lag:
                    ground_truth = self._buffer.popleft()
                    self._predict_and_learn(ground_truth, multistep_horizon)

                # --- delay-driven look-ahead forecast: "delay N frames"
                # means both "train N frames behind" (above) AND "show N
                # frames ahead" here -- one mechanism, not two. The
                # prediction pane displays a genuine target_lag-frame
                # forecast generated from the CURRENT (latest-trained)
                # train_hidden/train_pred, one forecast frame consumed per
                # real-frame tick; once exhausted, a fresh forecast is
                # (re)generated using whatever the model has learned in the
                # meantime -- so the pane always reflects the newest
                # weights. Runs under no_grad inside generate() and never
                # touches train_hidden/train_pred, so it can't affect
                # learning.
                if self._display_rollout_idx >= len(self._display_rollout):
                    self._display_rollout = self.generate(target_lag)
                    self._display_rollout_idx = 0
                if self._display_rollout:
                    pred_for_display = self._display_rollout[self._display_rollout_idx]
                    self._display_rollout_idx += 1
                else:
                    pred_for_display = real_tensor  # no seed yet (e.g. right after reset)
                mode_label = "FORECAST"
                forecast_step = self._display_rollout_idx
                forecast_horizon = len(self._display_rollout)

            self.frame_count += 1
            now = time.time()
            dt = now - self._last_tick
            self._last_tick = now
            if dt > 0:
                inst_fps = 1.0 / dt
                self.fps = inst_fps if self.frame_count == 1 else 0.9 * self.fps + 0.1 * inst_fps
            avg_loss = sum(self._loss_history) / len(self._loss_history) if self._loss_history else 0.0
            # Rolling per-term average of each enabled auxiliary loss's
            # WEIGHTED contribution (see compute_training_loss's `breakdown`)
            # -- lets the stats readout show which term is actually driving
            # `avg_loss` instead of only the summed total.
            loss_breakdown_avg = {
                term_name: sum(history) / len(history)
                for term_name, history in self._loss_component_history.items()
                if history
            }

            # Current (not rolling-averaged -- these drift slowly relative
            # to the per-step loss) learned weight for each active
            # auxiliary term, see model.py's UncertaintyWeighter.
            loss_weights = {
                name: self.model.loss_weighter.weight(name).item()
                for name in self.model.loss_weighter.log_vars
            }
            self._check_loss_weight_saturation(loss_weights)

            pred_jpeg = frame_codec.tensor_to_jpeg(pred_for_display)
            stats = {
                "type": "stats",
                "frame_count": self.frame_count,
                "fps": round(self.fps, 1),
                "avg_loss": avg_loss,
                "loss_breakdown": loss_breakdown_avg,
                "loss_weights": loss_weights,
                "mode_label": mode_label,
                "forecast_step": forecast_step,
                "forecast_horizon": forecast_horizon,
                "buffer_len": len(self._buffer),
                "target_lag": target_lag,
            }
            self._output.put((pred_jpeg, stats))

            now_ckpt = time.time()
            if (args.checkpoint_interval_seconds > 0
                    and now_ckpt - self._last_checkpoint_time >= args.checkpoint_interval_seconds):
                self.save_checkpoint()
                self._last_checkpoint_time = now_ckpt

            if self._reset_requested.is_set():
                self._do_reset()
                self._reset_requested.clear()
