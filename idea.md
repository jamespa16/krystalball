# Project: Live Webcam Next-Frame Prediction

## Goal
Build a Python application that trains a neural network **online, in real time**, on a live webcam feed to predict the next video frame. While training, display a window showing the current webcam frame side-by-side with the model's current prediction, updating live so the user can watch the model learn.

## Core behavior

1. Continuously capture frames from the default webcam.
2. Maintain a model with temporal memory (it must condition on frame history, not just the current frame in isolation — a single-frame-in/single-frame-out model has no way to represent motion).
3. Online training loop, repeated every frame:
   - The model has already produced a prediction for "the next frame" based on frames seen so far.
   - When the new real frame arrives, compute a loss between that prediction and the real frame, backprop, and step the optimizer.
   - Feed the new real frame into the model to update its temporal state and produce a prediction for the *following* frame.
4. Display a single window with two panes side by side: `[current real frame | current prediction]`, updated every loop iteration. Overlay basic live stats: frame count, FPS, and a rolling-average training loss.
5. Run indefinitely until the user quits (keypress `q`). Support a reset keypress (`r`) that clears the model's temporal hidden state and optimizer state without restarting the process, for recovering from instability.

## Technical requirements

- **Language/stack:** Python, PyTorch for the model, OpenCV (`cv2`) for webcam capture and display.
- **Resolution:** Internally downsample webcam frames to a small working resolution (start around 96×72, must be easily configurable) so that training keeps pace with the live video framerate. Upscale only for display, not for training.
- **Model architecture:** A small encoder → recurrent temporal core (ConvLSTM or equivalent) → decoder network that outputs a predicted RGB frame. The recurrent core must carry state across frames so predictions use motion history, not just the current static frame.
- **Device:** Auto-detect and use GPU if available, otherwise fall back to CPU.
- **Threading:** Webcam reads must happen on a background thread with a small lock-protected "latest frame" buffer, so training/inference is never blocked waiting on `cap.read()`.
- **Loss function:** MSE between predicted and actual frame is the baseline; structure the code so the loss function is easy to swap out later (e.g., for an SSIM or perceptual loss).
- **State detachment:** Detach the recurrent hidden state from the autograd graph after each optimizer step, so backprop only spans one timestep (avoid unbounded graph growth / memory blowup from training on an indefinitely long sequence).
- **No dataset/offline data needed** — training data is exclusively the live camera stream, consumed and discarded frame by frame (no frame buffer/dataset persistence required for v1).

## Configuration (should be easily editable constants or CLI flags)
- Camera index
- Working resolution (width, height)
- Display upscale factor
- Learning rate
- Hidden channel count of the recurrent core
- Optimizer choice (default Adam)

## Non-goals for v1 (explicitly out of scope, don't build)
- Multi-step/autoregressive rollout (predicting several frames into the future without ground truth)
- Saving/loading model checkpoints
- Any dataset collection, replay buffer, or offline pretraining
- Optical flow or other auxiliary input features
- Any GUI beyond the single OpenCV display window (no config UI, no web frontend)

## Deliverable
A single runnable Python script (or minimal set of modules if that's cleaner) with:
- A `requirements.txt` or documented `pip install` line (torch, opencv-python, numpy)
- Clear inline comments explaining the online-training loop logic (this is the non-obvious part: predict-then-wait-for-ground-truth-then-backprop, one step behind)
- A short usage note at the top of the file: how to run it, and what the keyboard controls (`q` to quit, `r` to reset) do

## Acceptance criteria
- Running the script opens a webcam window within a few seconds.
- The window shows two panes side by side, both updating live.
- The right pane visibly changes character over the first 30–60 seconds of runtime (starts blurry/noisy, becomes more temporally coherent) — confirming that training is actually happening, not just displaying a static or copied frame.
- Loss value displayed on screen trends downward or stabilizes over time rather than diverging.
- Frame rate is high enough to feel "live" (aim for double-digit FPS on a mid-range GPU; CPU fallback should still run, even if slower).
- `q` cleanly exits and releases the camera; `r` resets state without crashing.