"""CLI configuration for krystalball. All tunables live here, not buried in main.py."""

import argparse


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Live webcam next-frame prediction, trained online in real time."
    )
    p.add_argument(
        "--camera-index", type=int, default=0, help="OpenCV camera index (default: 0)"
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
        help="Number of stacked ConvLSTM layers forming the recurrent temporal "
        "core (default: 2)",
    )
    p.add_argument(
        "--lstm-hidden-channels",
        type=int,
        default=128,
        help="Hidden/cell channel count of each ConvLSTM layer, uniform across "
        "all layers (default: 128)",
    )
    p.add_argument(
        "--lstm-kernel-size",
        type=int,
        default=3,
        help="Convolution kernel size for every ConvLSTM gate (bottleneck stack "
        "and, if enabled, per-scale skip cells) (default: 3). Larger kernels "
        "widen each cell's per-step receptive field at extra compute cost.",
    )
    p.add_argument(
        "--skip-lstm-base-channels",
        type=int,
        default=16,
        help="Hidden channel count of the first (highest-resolution) per-scale "
        "ConvLSTM cell attached to each encoder skip connection, doubling per "
        "scale like the encoder's own channel progression (default: 16). Today "
        "skip connections are purely feedforward with no temporal memory of "
        "their own; this gives each scale its own recurrent state so fast/fine "
        "motion detail isn't limited to whatever the single coarsest bottleneck "
        "scale can represent. Set to 0 to disable (today's feedforward-only "
        "skip behavior).",
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
