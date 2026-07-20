# Project: Live Webcam Next-Frame Prediction

## Goal
Build a Python application that trains a neural network **online, in real time**, on a live webcam feed to predict the next video frame. While training, serve a self-hosted web page showing the current webcam frame side-by-side with the model's current prediction, updating live so the user can watch the model learn -- the browser (not necessarily the machine doing the training) owns the webcam.

## Core behavior

1. Continuously receive frames captured by the browser client's webcam, streamed to the server over a WebSocket connection (the server itself has no camera of its own -- see Deployment / Architecture below).
2. Maintain a model with temporal memory (it must condition on frame history, not just the current frame in isolation — a single-frame-in/single-frame-out model has no way to represent motion).
3. Online training loop:
   - The model has already produced a prediction for "the next frame" based on frames seen so far.
   - When the next ground-truth frame is available (see the replay buffer below — this may be the live frame, or a slightly delayed one), compute a loss between the pending prediction and that frame, backprop, and step the optimizer.
   - Feed that same ground-truth frame into the model to update its temporal state and produce a prediction for the *following* frame.
   - **Adjustable real-frame delay (replay buffer):** a live on-screen slider, snapping to whole-frame detents and displaying the current value, sets how many frames training is allowed to lag behind live video, from 0 (train on every frame immediately, the original 1:1 behavior) up to a configurable max (default 60 frames). Above 0, incoming real frames are appended to a small in-memory FIFO replay buffer instead of being trained on right away; a separate consumption step drains at most one frame per loop iteration once the backlog exceeds the target delay. This means training lags real time by exactly the slider's frame count, but it still eventually trains on *every* real frame — nothing is discarded, unlike a pure free-run/self-feed scheme. (If the slider is lowered mid-run, the buffer drains faster than it fills — more than one frame per iteration — until the backlog catches back down to the new, smaller target.)
   - **Cosmetic preview, decoupled from training:** whenever buffering is active, the displayed prediction pane runs its own self-feeding rollout purely for visual interest — periodically reseeded with the live real frame, self-fed on its own prior prediction in between. This preview path never receives gradients and never touches the replay buffer or the optimizer; it exists only so the display stays "live-looking" while the real training lags behind.
   - Backprop must still span exactly one timestep: hidden state is detached after every training step (whether that step consumed the live frame directly or a delayed frame popped from the buffer), so an extended lag never grows the autograd graph.
4. Render two panes side by side in a browser page: `[current real frame | current prediction]`, updated live. Show basic live stats: frame count, FPS, and a rolling-average training loss.
5. Run indefinitely until stopped. A "Stop" control pauses training (state preserved, not cleared) without restarting the server process -- sending a new frame (e.g. reloading the page and re-granting camera access) implicitly resumes it. A "Reset" control clears the model's temporal hidden state and optimizer state without restarting the process, for recovering from instability. Actually shutting down the server process remains an operator action (Ctrl-C on the machine running it), not exposed as a web control -- there's no "restart the process" button, deliberately, to avoid a stray browser click taking down shared server-side state.

## Technical requirements

- **Language/stack:** Python/PyTorch for the model (server-side, GPU-capable machine). FastAPI + Uvicorn + WebSockets for the client/server transport. A browser (`getUserMedia` + `<canvas>` + WebSocket) is the client: it captures the webcam and renders both display panes. OpenCV (`cv2`) is used server-side only for JPEG decode/encode and resizing -- no GUI window.
- **Resolution:** Internally downsample webcam frames to a small working resolution (start around 96×72, must be easily configurable) so that training keeps pace with the live video framerate. Upscale only for display, not for training.
- **Model architecture:** A small encoder → recurrent temporal core (ConvLSTM or equivalent) → decoder network that outputs a predicted RGB frame, plus a learned optical-flow head that warps the current frame (and encoder skip activations) forward under a constant-velocity assumption to form the decoder's residual base. The recurrent core must carry state across frames so predictions use motion history, not just the current static frame — this now means recurrence at *both* the coarsest bottleneck scale *and* each encoder skip-connection scale (a per-scale bank of independent ConvLSTM cells), not just the bottleneck alone, since skip-only-feedforward connections cap how well fast/fine motion detail can be tracked. The flow head is also fed its own previous flow estimate each step (in addition to the current/previous frame pair) so its estimate is refined/smoothed across steps instead of re-derived from scratch every frame.
- **Device:** Auto-detect and use GPU if available, otherwise fall back to CPU.
- **Threading:** Training runs in a dedicated background thread with a small lock-protected "latest frame" mailbox (`LatestSlot`) fed by the async WebSocket receive handler, so training/inference is never blocked waiting on the network -- a direct generalization of the original "background thread + lock-protected latest-frame buffer" design, just with the browser as the frame producer instead of `cap.read()`.
- **Loss function:** MSE between predicted and actual frame is the baseline; structure the code so the loss function is easy to swap out later (e.g., for an SSIM or perceptual loss). The final training loss is this swappable base loss plus two optional additive auxiliary terms: an edge-aware smoothness/total-variation regularizer on the predicted flow field (the flow head has no direct supervision otherwise, only the indirect signal from downstream photometric reconstruction), and a motion-magnitude term that supervises the predicted *amount* of frame-to-frame change against the actual amount — distinct from the existing per-pixel motion-weighted reweighting, which only reweights the base loss rather than adding a new signal.
- **State detachment:** Detach the recurrent hidden state from the autograd graph after each optimizer step, so backprop only spans one timestep (avoid unbounded graph growth / memory blowup from training on an indefinitely long sequence).
- **No offline/external dataset** — training data is exclusively the live camera stream. Frames are either trained on immediately or briefly held in the bounded in-memory replay buffer described above; nothing is ever persisted to disk or survives a process restart.

## Configuration (should be easily editable constants or CLI flags)
- Server host/port, and TLS cert/key paths for browser camera access from off-localhost
- Working resolution (width, height)
- Display upscale factor
- Learning rate
- Hidden channel count of the recurrent core
- ConvLSTM kernel size
- Per-skip-scale recurrent hidden-channel base (0 disables skip recurrence, falling back to feedforward-only skips)
- Flow-smoothness regularization weight
- Motion-magnitude supervision weight
- Optimizer choice (default Adam)
- Real-frame delay in frames (default 0 = every frame is real, i.e. the original 1:1 behavior), adjustable live via a browser slider that snaps to whole-frame detents and shows the current frame-delay value.

## Non-goals (explicitly out of scope, don't build)
- General-purpose long-horizon rollout as a standalone feature (e.g. an API to query "predict N frames ahead" on demand) — rollout here is limited to the cosmetic display preview described above (never feeds back into training) plus, as of v7 (see CLAUDE.md), a bounded --rollout-horizon-step self-conditioned rollout used only as an additive auxiliary TRAINING loss, gated off below that many frames of real-frame delay and sourcing its targets by peeking (never popping) the replay buffer, same discipline as the world-model loss
- Saving/loading model checkpoints
- Any dataset collection, disk-backed persistence, or offline pretraining from sources other than the live camera. The replay buffer is a small, bounded, in-memory FIFO queue only (sized by the real-frame-interval slider) — it is a latency/scheduling mechanism, not a dataset: it doesn't persist across restarts, isn't shuffled/sampled from, and every frame is consumed in strict arrival order exactly once
- Additional auxiliary input modalities beyond the in-scope learned optical-flow head (e.g. depth, segmentation, audio) — optical flow itself is in-scope, see Model architecture
- A general-purpose multi-user web service (accounts, auth, session isolation, concurrent independent training runs) -- this remains a single-operator LAN tool, just browser-fronted instead of OpenCV-windowed. One `TrainingEngine` instance lives for the process lifetime; there's no per-connection isolation.

## Deployment / Architecture

The server (`server.py`, FastAPI + Uvicorn) runs on the machine doing the actual training/inference -- typically the one with a GPU. It has no camera of its own. A browser on any device on the same network (e.g. a laptop with a built-in webcam) opens the server's page, captures its own webcam via `getUserMedia`, and streams frames to the server over a WebSocket (`/ws`) as JPEG bytes at the internal working resolution. The server streams prediction frames and stats back over the same socket; the "real" pane needs no round trip since the browser already has its own live camera frame locally.

Browsers only allow camera access (`getUserMedia`) over a secure context -- `https://` or `localhost`. Accessing the server from another machine on the LAN therefore requires a TLS certificate; a self-signed one (`--ssl-keyfile`/`--ssl-certfile`, generated with a single `openssl req` command) is sufficient for this use case -- the browser will show a one-time warning to click through.

## Deliverable
A minimal set of Python modules (server-side) plus a small static HTML/JS/CSS frontend, with:
- Dependencies managed via `uv` (`pyproject.toml`/`uv.lock`) -- torch, opencv-python, numpy, fastapi, uvicorn
- Clear inline comments explaining the online-training loop logic (this is the non-obvious part: predict-then-wait-for-ground-truth-then-backprop, one step behind)
- A short usage note at the top of the server entrypoint: how to run it, how to reach it from a browser, and what the on-page controls (delay slider, Reset, Stop) do

## Acceptance criteria
- Opening the served page in a browser and granting camera permission shows two live panes within a few seconds of the WebSocket connecting.
- Both panes update live.
- The prediction pane visibly changes character over the first 30–60 seconds of runtime (starts blurry/noisy, becomes more temporally coherent) — confirming that training is actually happening, not just displaying a static or copied frame.
- Loss value displayed on screen trends downward or stabilizes over time rather than diverging.
- Frame rate is high enough to feel "live" (aim for double-digit FPS on a mid-range GPU; CPU fallback should still run, even if slower), bounded by network round-trip in addition to compute.
- Stop pauses training without crashing or losing hidden state; Reset clears hidden/optimizer state without crashing.