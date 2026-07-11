# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status

v1 is implemented: `main.py` (entrypoint/training loop), `model.py`, `webcam.py`, `config.py`, and `requirements.txt`. Treat `idea.md` as the authoritative spec for any future changes.

## What this project is

A Python application that trains a neural network **online, in real time**, on a live webcam feed to predict the next video frame, displaying a live side-by-side window of `[real frame | model prediction]` while it learns. Full spec: [idea.md](idea.md).

Key behavioral requirements from the spec (do not silently drop these when implementing):

- **Predict-then-learn loop, one step behind**: the model predicts the next frame *before* it arrives; when the real frame arrives, loss is computed against the earlier prediction, then backprop/step happens, then that real frame is fed in to produce the *next* prediction. This ordering is the non-obvious core of the app — implement and comment it explicitly.
- **Temporal memory is required**: encoder → recurrent temporal core (ConvLSTM or equivalent) → decoder. A single-frame-in/single-frame-out model does not satisfy the spec because it can't represent motion.
- **Detach hidden state from autograd after every optimizer step** — backprop must span exactly one timestep, not the whole stream, or the graph grows unbounded.
- **Webcam reads on a background thread** with a lock-protected "latest frame" buffer, so `cap.read()` never blocks training/inference.
- Internally downsample for training (default ~96×72, must stay easily configurable), upscale only for display.
- Auto-detect GPU, fall back to CPU.
- Loss function should be swappable (MSE baseline, but structure so SSIM/perceptual loss can replace it later).
- Runs indefinitely; `q` quits and releases the camera cleanly, `r` resets recurrent hidden state + optimizer state without restarting the process.
- No dataset/replay buffer/checkpointing/multi-step rollout/optical flow/config UI — explicitly out of scope for v1 (see idea.md's Non-goals section before adding any of these).

## Stack

- Python, PyTorch (model/training), OpenCV `cv2` (webcam capture + display window).
- Dependencies (torch, opencv-python, numpy) are managed by `uv` via `pyproject.toml`/`uv.lock`.
- Config values (camera index, working resolution, display upscale factor, learning rate, hidden channel count, optimizer) should be exposed as easily editable constants or CLI flags, not buried in logic.

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
