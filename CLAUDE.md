# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status

This repository is currently empty of code — it contains only `idea.md` (the project spec, untracked in git) and `LICENSE` (Apache 2.0). There is no build system, package manifest, or source tree yet. When asked to start implementing, treat `idea.md` as the authoritative spec and check it back into git along with the code.

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
- Expect a `requirements.txt` (torch, opencv-python, numpy) once the deliverable is created.
- Config values (camera index, working resolution, display upscale factor, learning rate, hidden channel count, optimizer) should be exposed as easily editable constants or CLI flags, not buried in logic.

## Commands

No build/lint/test tooling exists yet. Once a `requirements.txt` and script(s) are added, update this section with actual install/run commands (e.g. `pip install -r requirements.txt`, `python <entrypoint>.py`) — do not invent commands that don't correspond to real files.
