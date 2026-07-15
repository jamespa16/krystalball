# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status

v1 (immediate 1:1 predict-then-learn), v2 (adjustable real-frame-interval replay buffer), v3 (deeper multi-scale U-Net-style encoder/decoder + stacked ConvLSTM temporal core + GroupNorm residual blocks + SSIM/blended loss + gradient clipping), and v4 (learned optical-flow head that warps the previous frame forward as the decoder's residual base + per-scale skip-connection ConvLSTM recurrence + decoder blend mask fusing warped/generated estimates + motion-weighted loss reweighting + motion-delta magnitude loss + flow-smoothness regularization + LR warmup scheduler) are all implemented across `main.py` (entrypoint/training loop), `model.py`, `webcam.py`, and `config.py`. Dependencies are managed via `pyproject.toml`/`uv.lock`, not `requirements.txt`. Treat `idea.md` as the authoritative spec for any future changes.

## What this project is

A Python application that trains a neural network **online, in real time**, on a live webcam feed to predict the next video frame, displaying a live side-by-side window of `[real frame | model prediction]` while it learns. Full spec: [idea.md](idea.md).

Key behavioral requirements from the spec (do not silently drop these when implementing):

- **Predict-then-learn loop, one step behind**: the model predicts the next frame *before* it arrives; when the ground-truth frame arrives (live, or popped from the replay buffer — see below), loss is computed against the earlier prediction, then backprop/step happens, then that frame is fed in to produce the *next* prediction. This ordering is the non-obvious core of the app — implement and comment it explicitly.
- **Temporal memory is required**: encoder → recurrent temporal core (ConvLSTM or equivalent) → decoder. A single-frame-in/single-frame-out model does not satisfy the spec because it can't represent motion. Recurrence lives at both the bottleneck (`ConvLSTMStack`) *and* each encoder skip-connection scale (`SkipConvLSTMBank`, one independent ConvLSTM cell per scale, toggleable via `--skip-lstm-base-channels 0`) — skip-only-feedforward connections cap how well fast/fine motion detail tracks.
- **Motion is modeled explicitly, not just regressed**: `FlowHead` predicts a per-step optical-flow field (conditioned on its own previous flow estimate, so it refines rather than re-derives each frame) used to warp the current frame/skip activations forward under a constant-velocity assumption; the decoder optionally blends that warped estimate with a freely generated one via a predicted per-pixel mask (`--blend-mask`) so it isn't limited to a small bounded delta in occlusion/disocclusion regions it can't warp correctly.
- **Detach hidden state from autograd after every training step, not just optimizer steps** — backprop must span exactly one timestep, whether that step trained on the live frame or a delayed one popped from the buffer, or the graph grows unbounded.
- **Webcam reads on a background thread** with a lock-protected "latest frame" buffer, so `cap.read()` never blocks training/inference.
- Internally downsample for training (default ~96×72, must stay easily configurable), upscale only for display.
- Auto-detect GPU, fall back to CPU.
- Loss function should be swappable (MSE baseline, but structure so SSIM/perceptual loss can replace it later). The final training loss composes this swappable base loss with three optional additive/reweighting terms (all wired through `compute_training_loss` in `main.py`): per-pixel motion-weighted reweighting of the base loss (`--motion-loss-weight`, via `motion_weight_map`), a motion-delta magnitude term supervising *how much* the frame changed (`--motion-delta-weight`, via `motion_delta_loss`), and edge-aware flow-smoothness regularization (`--flow-smoothness-weight`, via `flow_smoothness_loss`) — the flow head has no other direct supervision.
- Runs indefinitely; `q` quits and releases the camera cleanly, `r` resets recurrent hidden state + optimizer state + replay buffer without restarting the process.
- **Real-frame delay is adjustable live** via an in-window OpenCV trackbar ("Real frame delay (frames)") that snaps to whole-frame detents and shows the current value (also echoed in the on-screen overlay): at 0 the model trains on every frame immediately (original 1:1 behavior); above 0, real frames are pushed into a small in-memory FIFO replay buffer and drained one-per-iteration by training, so training lags real time by exactly that many frames but still eventually trains on *every* frame — nothing is discarded. This is `main.py`'s `train_hidden`/`train_pred`/`buffer` path.
- **The display's prediction pane is cosmetically separate from training** when buffering is active: it runs its own periodically-reseeded self-feeding rollout (`display_hidden`/`display_pred`, under `torch.no_grad()`) purely so the preview looks live; it never produces gradients and never touches the replay buffer. This is the one narrow, sanctioned form of multi-step rollout — see idea.md's Core behavior and Non-goals sections.
- No disk-backed dataset/checkpointing/standalone N-steps-ahead rollout API/config UI beyond the one trackbar above — explicitly out of scope (see idea.md's Non-goals section before adding any of these). The replay buffer itself is in-scope but stays a small bounded in-memory FIFO queue, not a persisted dataset. The learned optical-flow head (`FlowHead`) is in-scope, not a non-goal — see idea.md's Model architecture section; only *additional* auxiliary input modalities beyond it (depth, segmentation, audio, etc.) remain out of scope.

## Stack

- Python, PyTorch (model/training), OpenCV `cv2` (webcam capture + display window).
- Dependencies (torch, opencv-python, numpy) are managed by `uv` via `pyproject.toml`/`uv.lock`.
- Config values (camera index, working resolution, display upscale factor, learning rate + warmup schedule, encoder depth/channel counts, ConvLSTM layer count/hidden channels, per-scale skip-recurrence channels, flow-head channels/smoothness weight, blend-mask toggle, motion-loss/motion-delta weights, optimizer, loss choice/blend weight, gradient-clip norm, real-frame interval) should be exposed as easily editable constants or CLI flags, not buried in logic.

## Commands

This is a `uv` project (`pyproject.toml` + `uv.lock`).

```
uv sync                        # install/sync dependencies into .venv
uv run main.py                 # run with defaults (camera 0, 96x72, upscale 6x)
uv run main.py --help          # see all CLI flags (camera index, resolution, lr, etc.)
uv add <package>                # add a new dependency (updates pyproject.toml + uv.lock)
```

No lint/test tooling exists yet.

## Code layout

- `config.py` — argparse CLI flags / defaults (camera index, resolution, lr + warmup, encoder scales/channels/res-blocks, ConvLSTM layers/hidden channels, skip-recurrence channels, flow-head/blend-mask/motion-loss flags, optimizer, loss choice + SSIM blend weight, gradient-clip norm).
- `model.py` — multi-scale U-Net-style `Encoder` / `Decoder` (GroupNorm `ResBlock`s via `DownBlock`/`UpBlock`, one skip connection per scale, each optionally backed by a `SkipConvLSTMBank` cell) → `ConvLSTMStack` (stacked multi-layer `ConvLSTMCell`s at the bottleneck; hidden state is `list[(h, c)]`, one tuple per layer) → `NextFramePredictor`, which also owns `FlowHead`/`warp_frame` (optical-flow prediction + forward-warp) and the decoder's blend-mask fusion. Plus `detach_hidden()` (generalized to the per-layer hidden-state list), the auxiliary-loss helpers (`motion_weight_map`, `motion_delta_loss`, `flow_smoothness_loss`), and the `get_loss_fn()` swappable-loss factory (`mse` / `ssim` / `mse_ssim`, the last via `SSIMLoss`/`BlendedLoss`).
- `webcam.py` — `WebcamStream`: background-thread capture with a lock-protected latest-frame buffer.
- `main.py` — entrypoint; owns the predict-then-learn training loop (including `compute_training_loss`, which composes the base loss with the auxiliary terms above, and the Adam/SGD + `LinearLR` warmup scheduler built by `make_optimizer_and_scheduler`), display compositing/overlay, and `q`/`r` key handling.
