# Project: Live Webcam Next-Frame Prediction

## Goal
Build a Python application that trains a neural network **online, in real time**, on a live webcam feed to predict the next video frame. While training, display a window showing the current webcam frame side-by-side with the model's current prediction, updating live so the user can watch the model learn.

## Core behavior

1. Continuously capture frames from the default webcam.
2. Maintain a model with temporal memory (it must condition on frame history, not just the current frame in isolation — a single-frame-in/single-frame-out model has no way to represent motion).
3. Online training loop:
   - The model has already produced a prediction for "the next frame" based on frames seen so far.
   - When the next ground-truth frame is available (see the replay buffer below — this may be the live frame, or a slightly delayed one), compute a loss between the pending prediction and that frame, backprop, and step the optimizer.
   - Feed that same ground-truth frame into the model to update its temporal state and produce a prediction for the *following* frame.
   - **Adjustable real-frame delay (replay buffer):** a live on-screen slider, snapping to whole-frame detents and displaying the current value, sets how many frames training is allowed to lag behind live video, from 0 (train on every frame immediately, the original 1:1 behavior) up to a configurable max (default 60 frames). Above 0, incoming real frames are appended to a small in-memory FIFO replay buffer instead of being trained on right away; a separate consumption step drains at most one frame per loop iteration once the backlog exceeds the target delay. This means training lags real time by exactly the slider's frame count, but it still eventually trains on *every* real frame — nothing is discarded, unlike a pure free-run/self-feed scheme. (If the slider is lowered mid-run, the buffer drains faster than it fills — more than one frame per iteration — until the backlog catches back down to the new, smaller target.)
   - **Cosmetic preview, decoupled from training:** whenever buffering is active, the displayed prediction pane runs its own self-feeding rollout purely for visual interest — periodically reseeded with the live real frame, self-fed on its own prior prediction in between. This preview path never receives gradients and never touches the replay buffer or the optimizer; it exists only so the display stays "live-looking" while the real training lags behind.
   - Backprop must still span exactly one timestep: hidden state is detached after every training step (whether that step consumed the live frame directly or a delayed frame popped from the buffer), so an extended lag never grows the autograd graph.
4. Display a single window with two panes side by side: `[current real frame | current prediction]`, updated every loop iteration. Overlay basic live stats: frame count, FPS, and a rolling-average training loss.
5. Run indefinitely until the user quits (keypress `q`). Support a reset keypress (`r`) that clears the model's temporal hidden state and optimizer state without restarting the process, for recovering from instability.

## Technical requirements

- **Language/stack:** Python, PyTorch for the model, OpenCV (`cv2`) for webcam capture and display.
- **Resolution:** Internally downsample webcam frames to a small working resolution (start around 96×72, must be easily configurable) so that training keeps pace with the live video framerate. Upscale only for display, not for training.
- **Model architecture:** A small encoder → recurrent temporal core (ConvLSTM or equivalent) → decoder network that outputs a predicted RGB frame, plus a learned optical-flow head that warps the current frame (and encoder skip activations) forward under a constant-velocity assumption to form the decoder's residual base. The recurrent core must carry state across frames so predictions use motion history, not just the current static frame — this now means recurrence at *both* the coarsest bottleneck scale *and* each encoder skip-connection scale (a per-scale bank of independent ConvLSTM cells), not just the bottleneck alone, since skip-only-feedforward connections cap how well fast/fine motion detail can be tracked. The flow head is also fed its own previous flow estimate each step (in addition to the current/previous frame pair) so its estimate is refined/smoothed across steps instead of re-derived from scratch every frame.
- **Device:** Auto-detect and use GPU if available, otherwise fall back to CPU.
- **Threading:** Webcam reads must happen on a background thread with a small lock-protected "latest frame" buffer, so training/inference is never blocked waiting on `cap.read()`.
- **Loss function:** MSE between predicted and actual frame is the baseline; structure the code so the loss function is easy to swap out later (e.g., for an SSIM or perceptual loss). The final training loss is this swappable base loss plus two optional additive auxiliary terms: an edge-aware smoothness/total-variation regularizer on the predicted flow field (the flow head has no direct supervision otherwise, only the indirect signal from downstream photometric reconstruction), and a motion-magnitude term that supervises the predicted *amount* of frame-to-frame change against the actual amount — distinct from the existing per-pixel motion-weighted reweighting, which only reweights the base loss rather than adding a new signal.
- **State detachment:** Detach the recurrent hidden state from the autograd graph after each optimizer step, so backprop only spans one timestep (avoid unbounded graph growth / memory blowup from training on an indefinitely long sequence).
- **No offline/external dataset** — training data is exclusively the live camera stream. Frames are either trained on immediately or briefly held in the bounded in-memory replay buffer described above; nothing is ever persisted to disk or survives a process restart.

## Configuration (should be easily editable constants or CLI flags)
- Camera index
- Working resolution (width, height)
- Display upscale factor
- Learning rate
- Hidden channel count of the recurrent core
- ConvLSTM kernel size
- Per-skip-scale recurrent hidden-channel base (0 disables skip recurrence, falling back to feedforward-only skips)
- Flow-smoothness regularization weight
- Motion-magnitude supervision weight
- Optimizer choice (default Adam)
- Real-frame delay in frames (default 0 = every frame is real, i.e. the original 1:1 behavior), adjustable live via an in-window OpenCV trackbar that snaps to whole-frame detents and shows the current frame-delay value.

## Non-goals (explicitly out of scope, don't build)
- General-purpose long-horizon rollout as a standalone feature (e.g. an API to query "predict N frames ahead" on demand) — the only rollout is the cosmetic display preview described above, which never feeds back into training
- Saving/loading model checkpoints
- Any dataset collection, disk-backed persistence, or offline pretraining from sources other than the live camera. The replay buffer is a small, bounded, in-memory FIFO queue only (sized by the real-frame-interval slider) — it is a latency/scheduling mechanism, not a dataset: it doesn't persist across restarts, isn't shuffled/sampled from, and every frame is consumed in strict arrival order exactly once
- Additional auxiliary input modalities beyond the in-scope learned optical-flow head (e.g. depth, segmentation, audio) — optical flow itself is in-scope, see Model architecture
- Any GUI beyond the single OpenCV display window and its trackbar (no separate config panel, no web frontend)

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