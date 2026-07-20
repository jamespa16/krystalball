"""CLI configuration for krystalball. All tunables live here, not buried in main.py."""

import argparse


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Live webcam next-frame prediction, trained online in real time."
    )
    p.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Interface for the web server to bind to (default: 0.0.0.0, i.e. all interfaces)",
    )
    p.add_argument(
        "--port", type=int, default=8000, help="Port for the web server to listen on (default: 8000)"
    )
    p.add_argument(
        "--ssl-keyfile",
        type=str,
        default=None,
        help="Path to a TLS private key (PEM). Required for browser webcam access "
        "from any origin other than localhost -- see README for how to generate a "
        "self-signed cert. Omit to serve plain HTTP (fine for localhost-only use).",
    )
    p.add_argument(
        "--ssl-certfile",
        type=str,
        default=None,
        help="Path to a TLS certificate (PEM), paired with --ssl-keyfile.",
    )
    p.add_argument(
        "--width",
        type=int,
        default=96,
        help="Internal working width in pixels; must be divisible by 2**--encoder-scales "
        "(default: 96, divisible by 8 for the default 3 scales)",
    )
    p.add_argument(
        "--height",
        type=int,
        default=72,
        help="Internal working height in pixels; must be divisible by 2**--encoder-scales "
        "(default: 72, divisible by 8 for the default 3 scales)",
    )
    p.add_argument(
        "--upscale",
        type=int,
        default=6,
        help="Factor to upscale each pane by for display only (default: 6)",
    )
    p.add_argument(
        "--lr", type=float, default=2e-4, help="Optimizer learning rate (default: 2e-4)"
    )
    p.add_argument(
        "--lr-warmup-steps",
        type=int,
        default=200,
        help="Number of training steps (optimizer.step() calls) to linearly ramp the "
        "LR up from --lr-warmup-start-factor*--lr to the full --lr value; re-triggered "
        "on every 'r' reset (default: 200; set 0 to disable warmup)",
    )
    p.add_argument(
        "--lr-warmup-start-factor",
        type=float,
        default=0.05,
        help="LR at the start of warmup, as a fraction of --lr (default: 0.05)",
    )
    p.add_argument(
        "--encoder-base-channels",
        type=int,
        default=32,
        help="Encoder first-stage channel count; doubles at each subsequent "
        "downsample stage (default: 32, giving 32/64/128 for the default 3 scales)",
    )
    p.add_argument(
        "--encoder-scales",
        type=int,
        default=3,
        help="Number of stride-2 downsample stages in the encoder/decoder "
        "(total spatial downsample = 2**this). --width/--height must be "
        "divisible by 2**this (default: 3, i.e. divisible by 8)",
    )
    p.add_argument(
        "--res-blocks-per-scale",
        type=int,
        default=1,
        help="Number of GroupNorm residual blocks after each encoder downsample "
        "/ before each decoder upsample (default: 1)",
    )
    p.add_argument(
        "--lstm-layers",
        type=int,
        default=2,
        help="Number of bottleneck-depth spatiotemporal-LSTM cells stacked after "
        "the (optional) skip-scale cells in the hierarchical ST-LSTM core "
        "(default: 2) -- the cross-scale memory `m` zigzags through these last, "
        "after every skip scale.",
    )
    p.add_argument(
        "--lstm-hidden-channels",
        type=int,
        default=128,
        help="Hidden/cell channel count of each bottleneck-depth ST-LSTM cell, "
        "uniform across all such cells (default: 128).",
    )
    p.add_argument(
        "--lstm-kernel-size",
        type=int,
        default=3,
        help="Convolution kernel size for every ST-LSTM gate (bottleneck-depth "
        "cells and, if enabled, per-scale skip cells alike) (default: 3). Larger "
        "kernels widen each cell's per-step receptive field at extra compute cost.",
    )
    p.add_argument(
        "--st-memory-channels",
        type=int,
        default=32,
        help="Channel count of the spatiotemporal memory `m` shared across every "
        "scale of the hierarchical ST-LSTM core (default: 32), distinct from each "
        "scale's own hidden/cell channels (--skip-lstm-base-channels / "
        "--lstm-hidden-channels). This is the tensor that gives the recurrent "
        "core genuine cross-scale information flow within a single timestep -- "
        "it zigzags from the finest skip scale down to the bottleneck within one "
        "step, and the bottleneck's final value seeds the finest scale's `m` at "
        "the start of the next step.",
    )
    p.add_argument(
        "--skip-lstm-base-channels",
        type=int,
        default=16,
        help="Hidden channel count of the first (highest-resolution) per-scale "
        "ST-LSTM cell attached to each encoder skip connection, doubling per "
        "scale like the encoder's own channel progression (default: 16). Gives "
        "each scale its own recurrent state (and a place in the cross-scale `m` "
        "zigzag, see --st-memory-channels) so fast/fine motion detail isn't "
        "limited to whatever the single coarsest bottleneck scale can represent. "
        "Set to 0 to disable (feedforward-only skips; the ST-LSTM core then runs "
        "at the bottleneck depth only -- still gains the `m` pathway there "
        "versus a plain stacked ConvLSTM, just without the cross-scale reach).",
    )
    p.add_argument(
        "--delta-scale",
        type=float,
        default=0.6,
        help="Max per-step residual magnitude added to the input frame to form the "
        "prediction (decoder outputs scale*tanh(x), default: 0.6). Lower values "
        "keep free-running rollouts closer to the learned manifold (less drift) "
        "at the cost of reacting more slowly to fast real motion.",
    )
    p.add_argument(
        "--use-flow",
        choices=["on", "off"],
        default="on",
        help="Predict a learned optical-flow field between consecutive frames and warp "
        "the current frame forward by it to form the decoder's residual base (instead "
        "of the raw current frame), plus feed the flow field as extra encoder input "
        "channels (default: on)",
    )
    p.add_argument(
        "--flow-hidden-channels",
        type=int,
        default=16,
        help="Hidden channel count of the flow-prediction submodule (default: 16). "
        "Lower this if --use-flow on measurably hurts CPU framerate.",
    )
    p.add_argument(
        "--flow-smoothness-weight",
        type=float,
        default=0.01,
        help="Edge-aware total-variation regularization weight on the predicted "
        "flow field (units: raw pixel-displacement, a much larger scale than the "
        "photometric loss -- keep this small) (default: 0.01). Too large a value "
        "can collapse flow back toward the zero-init trivial solution. Set to 0 "
        "to disable. Ignored when --use-flow off.",
    )
    p.add_argument(
        "--flow-acceleration",
        choices=["on", "off"],
        default="on",
        help="Extend FlowHead to also predict flow VELOCITY (the rate of change of "
        "the flow field itself) and warp using flow+velocity (constant-acceleration "
        "extrapolation) instead of just flow (constant-velocity extrapolation) "
        "(default: on). Zero-initialized like the rest of FlowHead, so this is "
        "architecturally additive at construction -- a fresh model still predicts "
        "exactly zero flow and zero velocity. Set to off to fall back to today's "
        "constant-velocity-only behavior. Ignored when --use-flow off.",
    )
    p.add_argument(
        "--flow-accel-smoothness-weight",
        type=float,
        default=0.01,
        help="Edge-aware total-variation regularization weight on the predicted "
        "flow-VELOCITY field, analogous to --flow-smoothness-weight but for the "
        "acceleration term (default: 0.01). Velocity is, like flow, supervised "
        "only indirectly through the downstream warp -- without this it can "
        "drift noisily in textureless/occluded regions. Set to 0 to disable. "
        "Ignored unless --flow-acceleration on.",
    )
    p.add_argument(
        "--world-model-weight",
        type=float,
        default=0.1,
        help="Weight for an auxiliary latent-space consistency loss: a small "
        "predictor head maps the recurrent core's current bottleneck latent to "
        "a predicted future latent, compared (cosine distance) against the real "
        "encoder features of a genuine future frame, --world-model-horizon "
        "frames ahead (default: 0.1). Makes the internal representation more "
        "predictive of actual future content, not just good for one-step pixel "
        "reconstruction. Set to 0 to disable. NOTE: only active while the real-"
        "frame delay (--real-frame-interval-frames / the live delay slider) is "
        ">= --world-model-horizon, since the lookahead frame is sourced by "
        "peeking the existing replay buffer rather than a new buffering "
        "mechanism -- inactive (a logged no-op) at the default delay of 0.",
    )
    p.add_argument(
        "--world-model-horizon",
        type=int,
        default=4,
        help="How many frames ahead (peeked from the replay buffer) the world-"
        "model predictor head targets (default: 4). Larger values make the "
        "objective harder (less correlated with the present) but push the "
        "representation to encode longer-range predictive structure; must be "
        "<= --real-frame-interval-max-frames for the feature to ever be "
        "reachable via the delay slider.",
    )
    p.add_argument(
        "--world-model-hidden-channels",
        type=int,
        default=64,
        help="Hidden channel count of the world-model predictor head's internal "
        "conv layers (default: 64). Deliberately a separate small network from "
        "the encoder/ST-LSTM whose output it's trained to match (an asymmetric "
        "predictor head), which is itself part of this loss's defense against "
        "representation collapse.",
    )
    p.add_argument(
        "--rollout-weight",
        type=float,
        default=0.1,
        help="Weight for an auxiliary rollout-consistency loss: after the "
        "primary loop's forward call this tick, the model self-feeds its own "
        "prediction for --rollout-horizon more steps (true BPTT -- gradient "
        "flows across the whole chain, not detached step-by-step), each step "
        "scored against a real future frame peeked (never popped) from the "
        "replay buffer (default: 0.1). Exists because the primary predict-"
        "then-learn loop is strictly teacher-forced and never trains on its "
        "own prior predictions, so low one-step loss is compatible with the "
        "model diverging once it has to run several steps ahead of ground "
        "truth. Set to 0 to disable. NOTE: only active while the real-frame "
        "delay (--real-frame-interval-frames / the live delay slider) is >= "
        "--rollout-horizon + 1 -- inactive (a logged no-op) at the default "
        "delay of 0.",
    )
    p.add_argument(
        "--rollout-horizon",
        type=int,
        default=3,
        help="How many self-fed steps (K) the rollout-consistency loss chains "
        "forward, each compared against a real future frame peeked from the "
        "replay buffer (default: 3). Larger values push harder against "
        "compounding-error drift but cost K extra full model forward passes "
        "per activation and deepen the retained BPTT graph; must be < "
        "--real-frame-interval-max-frames for the feature to ever be "
        "reachable via the delay slider.",
    )
    p.add_argument(
        "--rollout-every-n-steps",
        type=int,
        default=8,
        help="Only run the rollout-consistency K-step chain on every Nth "
        "eligible training tick (default: 8). This is a real-time training "
        "loop, and each activation costs --rollout-horizon extra full model "
        "forward passes, so this bounds the amortized FPS cost. Lower this "
        "for a stronger/more frequent signal at the cost of framerate.",
    )
    p.add_argument(
        "--rollout-decay",
        type=float,
        default=1.0,
        help="Per-step weight multiplier (decay**i) applied across the "
        "rollout-consistency chain's K steps (default: 1.0, i.e. flat -- "
        "every step weighted equally). Values < 1.0 discount later, harder-"
        "to-predict steps relative to earlier ones.",
    )
    p.add_argument(
        "--adv-weight",
        type=float,
        default=0.05,
        help="Weight for the generator's adversarial (LSGAN) loss term against a "
        "discriminator trained alongside it from scratch, encouraging sharper/"
        "less-blurry predictions than photometric loss alone rewards (default: "
        "0.05 -- kept small since this optimizes realism/sharpness, not exact "
        "pixel correctness; too high can let the generator hallucinate detail "
        "unfaithful to actual content). Ramped up from 0 over --adv-warmup-steps. "
        "Set to 0 to disable (the discriminator still trains, just has no effect "
        "on the generator).",
    )
    p.add_argument(
        "--adv-warmup-steps",
        type=int,
        default=500,
        help="Training steps to linearly ramp the adversarial loss weight from 0 "
        "up to --adv-weight, restarted on every reset (default: 500). Mitigates "
        "the classic GAN cold-start problem: an untrained predictor's early "
        "noisy output gives the discriminator a trivial win with near-zero "
        "useful gradient back to the generator -- this keeps the generator "
        "immune to that while the discriminator has time to become a genuinely "
        "useful critic. Set to 0 to apply --adv-weight at full strength immediately.",
    )
    p.add_argument(
        "--disc-lr",
        type=float,
        default=1e-4,
        help="Discriminator's own optimizer learning rate, independent of the "
        "predictor's --lr (default: 1e-4). Balancing generator/discriminator "
        "learning speed is its own tuning axis distinct from the predictor's.",
    )
    p.add_argument(
        "--disc-base-channels",
        type=int,
        default=32,
        help="Discriminator's first-stage channel count, doubling per stride-2 "
        "stage like --encoder-base-channels (default: 32).",
    )
    p.add_argument(
        "--disc-layers",
        type=int,
        default=3,
        help="Number of stride-2 conv stages in the discriminator's PatchGAN "
        "stack (default: 3). Produces a spatial map of real/fake verdicts "
        "rather than one per frame, so gradient concentrates on which regions "
        "look fake instead of an undifferentiated whole-frame judgment.",
    )
    p.add_argument(
        "--blend-mask",
        choices=["on", "off"],
        default="on",
        help="Let the decoder additionally predict a fully-generated pixel estimate and "
        "a per-pixel blend mask, so it can locally fall back away from the warped base "
        "frame (e.g. in occlusion/disocclusion regions warping can't represent) instead "
        "of being limited to a small bounded delta on top of it (default: on). Set to "
        "off if this introduces training instability.",
    )
    p.add_argument(
        "--optimizer",
        choices=["adam", "sgd"],
        default="adam",
        help="Optimizer to use (default: adam)",
    )
    p.add_argument(
        "--loss",
        choices=["mse", "ssim", "mse_ssim"],
        default="mse_ssim",
        help="Loss function between prediction and real frame (default: mse_ssim, "
        "a blend of MSE and 1-SSIM -- see --ssim-weight)",
    )
    p.add_argument(
        "--ssim-weight",
        type=float,
        default=0.5,
        help="Blend weight for --loss mse_ssim: final = (1-w)*MSE + w*(1-SSIM). "
        "Ignored for other --loss choices (default: 0.5)",
    )
    p.add_argument(
        "--motion-loss-weight",
        type=float,
        default=1.0,
        help="Extra per-pixel loss weight in regions that changed from the previous "
        "training frame, on top of a base weight of 1 (default: 1.0, giving moving "
        "regions ~2x the static background's average gradient weight). Set to 0 to "
        "disable (uniform per-pixel weighting, today's behavior).",
    )
    p.add_argument(
        "--motion-delta-weight",
        type=float,
        default=0.5,
        help="Extra loss term matching the magnitude of predicted frame-to-frame "
        "change to actual change (|pred - prev_frame| vs |target - prev_frame|), "
        "distinct from --motion-loss-weight (which only reweights the existing "
        "photometric loss, it doesn't add a new term) -- directly supervises "
        "whether the model predicts the right amount of motion, not just correct "
        "final pixels (default: 0.5). Set to 0 to disable.",
    )
    p.add_argument(
        "--grad-clip-norm",
        type=float,
        default=0.5,
        help="Max L2 norm for gradient clipping, applied after backward() and "
        "before optimizer.step(); set to 0 to disable (default: 0.5)",
    )
    p.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
        help="Compute device; 'auto' picks GPU/MPS if available (default: auto)",
    )
    p.add_argument(
        "--loss-window",
        type=int,
        default=100,
        help="Number of recent losses averaged for the on-screen readout (default: 100)",
    )
    p.add_argument(
        "--real-frame-interval-frames",
        type=int,
        default=0,
        help="Initial real-frame delay in frames; 0 means every frame is real "
        "(default: 0). Adjustable live via the on-screen trackbar, which snaps "
        "to whole-frame detents and shows the current value. Between real "
        "frames the model free-runs on its own predictions.",
    )
    p.add_argument(
        "--real-frame-interval-max-frames",
        type=int,
        default=60,
        help="Upper bound (frames) of the real-frame-interval trackbar (default: 60)",
    )
    return p
