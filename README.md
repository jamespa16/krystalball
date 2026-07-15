# krystalball

A Python app that trains a neural network **online, in real time**, on your
live webcam feed to predict the next video frame — and shows you it
learning. One window, two panes: `[real frame | model's prediction]`,
updating live while the model trains underneath.

There's no dataset, no offline training, no checkpoints. The model starts
from random weights the moment you run it, and everything you see it
predict is produced by weights that have only ever seen your webcam, live,
up to that instant.

## How it works

Every loop iteration:

1. The model has already made a prediction for "the next frame," based on
   everything it's seen so far.
2. The next real frame arrives. The loss between that prediction and the
   real frame is computed, backpropagated, and the optimizer steps.
3. That same real frame is then fed into the model to update its temporal
   state and produce a prediction for the *following* frame.

This predict-then-learn ordering — always one step behind — is the
non-obvious core of the app, implemented explicitly in `main.py`. Hidden
state is detached from autograd after every training step, so backprop
never spans more than one timestep no matter how long the process runs.

The model has real temporal memory (a stacked ConvLSTM core, plus
per-scale recurrence at each encoder skip connection) — a single
frame-in/frame-out network can't represent motion, so this app doesn't use
one. It also predicts a learned optical-flow field each step, warps the
previous frame forward by it, and blends that warped estimate with a
freely generated one to build its prediction — an explicit motion model on
top of raw pixel regression.

### Adjustable real-frame delay

An in-window trackbar ("Real frame delay (frames)") controls how far
training is allowed to lag behind live video, from 0 up to a configurable
max (default 60):

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
- A webcam
- GPU optional — auto-detects CUDA or Apple MPS, falls back to CPU

Dependencies (PyTorch, OpenCV, NumPy) are managed by [uv](https://docs.astral.sh/uv/).

## Quickstart

```bash
uv sync
uv run main.py
```

A window opens within a few seconds. The right pane starts noisy/blurry
and should visibly sharpen and become more temporally coherent over the
first 30–60 seconds as the model trains.

```bash
uv run main.py --help          # see all CLI flags
uv add <package>                # add a new dependency
```

## Controls

| Key / control | Effect |
|---|---|
| `q` | Quit — releases the camera and closes the window cleanly |
| `r` | Reset — clears recurrent hidden state, replay buffer, and optimizer state (fresh Adam/SGD), without restarting the process. Use this if training visibly diverges |
| "Real frame delay (frames)" trackbar | Live-adjustable real-frame delay described above; snaps to whole-frame detents, value echoed in the on-screen overlay |

The overlay also shows frame count, FPS, rolling-average training loss,
replay-buffer backlog, and whether the displayed frame is currently
anchored to a real frame or free-running.

## Configuration

Everything tunable is a CLI flag (see `uv run main.py --help` for the full,
current list with defaults) — nothing is buried in logic. Highlights:

- **Capture/display**: `--camera-index`, `--width`/`--height` (internal
  training resolution, default 96×72), `--upscale` (display-only)
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
  `--real-frame-interval-max-frames` (trackbar upper bound)
- **Device**: `--device` (auto/cpu/cuda/mps)

## Project layout

- `main.py` — entrypoint; the predict-then-learn training loop, display
  compositing/overlay, `q`/`r` key handling
- `model.py` — multi-scale U-Net-style encoder/decoder with GroupNorm
  residual blocks, stacked ConvLSTM temporal core (plus per-scale skip
  recurrence), a learned optical-flow head, and swappable loss functions
  (MSE / SSIM / blended)
- `webcam.py` — background-thread webcam capture with a lock-protected
  latest-frame buffer, so `cap.read()` never blocks training
- `config.py` — all CLI flags/defaults
- `idea.md` — the authoritative spec for this project; read it before
  making architectural changes

## Non-goals

By design, this project does **not** do checkpointing, offline/disk-backed
datasets, or general-purpose N-steps-ahead rollout as a standalone
feature. The replay buffer is a small bounded in-memory FIFO queue, not a
persisted dataset — see `idea.md`'s Non-goals section before adding any of
these.
