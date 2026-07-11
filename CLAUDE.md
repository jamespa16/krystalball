# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status

v1 (immediate 1:1 predict-then-learn) and v2 (adjustable real-frame-interval replay buffer) are both implemented across `main.py` (entrypoint/training loop), `model.py`, `webcam.py`, `config.py`, and `requirements.txt`. Treat `idea.md` as the authoritative spec for any future changes.

## What this project is

A Python application that trains a neural network **online, in real time**, on a live webcam feed to predict the next video frame, displaying a live side-by-side window of `[real frame | model prediction]` while it learns. Full spec: [idea.md](idea.md).

Key behavioral requirements from the spec (do not silently drop these when implementing):

- **Predict-then-learn loop, one step behind**: the model predicts the next frame *before* it arrives; when the ground-truth frame arrives (live, or popped from the replay buffer — see below), loss is computed against the earlier prediction, then backprop/step happens, then that frame is fed in to produce the *next* prediction. This ordering is the non-obvious core of the app — implement and comment it explicitly.
- **Temporal memory is required**: encoder → recurrent temporal core (ConvLSTM or equivalent) → decoder. A single-frame-in/single-frame-out model does not satisfy the spec because it can't represent motion.
- **Detach hidden state from autograd after every training step, not just optimizer steps** — backprop must span exactly one timestep, whether that step trained on the live frame or a delayed one popped from the buffer, or the graph grows unbounded.
- **Webcam reads on a background thread** with a lock-protected "latest frame" buffer, so `cap.read()` never blocks training/inference.
- Internally downsample for training (default ~96×72, must stay easily configurable), upscale only for display.
- Auto-detect GPU, fall back to CPU.
- Loss function should be swappable (MSE baseline, but structure so SSIM/perceptual loss can replace it later).
- Runs indefinitely; `q` quits and releases the camera cleanly, `r` resets recurrent hidden state + optimizer state + replay buffer without restarting the process.
- **Real-frame interval is adjustable live** via an in-window OpenCV trackbar ("Real frame every (sec)"): at 0 the model trains on every frame immediately (original 1:1 behavior); above 0, real frames are pushed into a small in-memory FIFO replay buffer and drained one-per-iteration by training, so training lags real time by roughly the slider's interval but still eventually trains on *every* frame — nothing is discarded. This is `main.py`'s `train_hidden`/`train_pred`/`buffer` path.
- **The display's prediction pane is cosmetically separate from training** when buffering is active: it runs its own periodically-reseeded self-feeding rollout (`display_hidden`/`display_pred`, under `torch.no_grad()`) purely so the preview looks live; it never produces gradients and never touches the replay buffer. This is the one narrow, sanctioned form of multi-step rollout — see idea.md's Core behavior and Non-goals sections.
- No disk-backed dataset/checkpointing/optical flow/standalone N-steps-ahead rollout API/config UI beyond the one trackbar above — explicitly out of scope (see idea.md's Non-goals section before adding any of these). The replay buffer itself is in-scope but stays a small bounded in-memory FIFO queue, not a persisted dataset.

## Stack

- Python, PyTorch (model/training), OpenCV `cv2` (webcam capture + display window).
- Dependencies (torch, opencv-python, numpy) are managed by `uv` via `pyproject.toml`/`uv.lock`.
- Config values (camera index, working resolution, display upscale factor, learning rate, hidden channel count, optimizer, real-frame interval) should be exposed as easily editable constants or CLI flags, not buried in logic.

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

- `config.py` — argparse CLI flags / defaults (camera index, resolution, lr, hidden channels, optimizer, loss).
- `model.py` — `Encoder` / `ConvLSTMCell` / `Decoder` / `NextFramePredictor`, plus `detach_hidden()` and the `get_loss_fn()` swappable-loss factory.
- `webcam.py` — `WebcamStream`: background-thread capture with a lock-protected latest-frame buffer.
- `main.py` — entrypoint; owns the predict-then-learn training loop, display compositing/overlay, and `q`/`r` key handling.
