"use strict";

const video = document.getElementById("video");
const realCanvas = document.getElementById("realCanvas");
const predCanvas = document.getElementById("predCanvas");
const realCtx = realCanvas.getContext("2d");
const predCtx = predCanvas.getContext("2d");
const statsEl = document.getElementById("stats");
const delaySlider = document.getElementById("delaySlider");
const delayValueEl = document.getElementById("delayValue");
const resetBtn = document.getElementById("resetBtn");
const stopBtn = document.getElementById("stopBtn");

const SEND_FPS = 15;
let ws = null;
let sendTimer = null;
let sendCanvas = null;
let sendCtx = null;
let stopped = false;
let cfg = { width: 96, height: 72, upscale: 6, max_lag: 60 };

function applyConfig(newCfg) {
  cfg = newCfg;
  const dispW = cfg.width * cfg.upscale;
  const dispH = cfg.height * cfg.upscale;

  // Real pane: drawn directly from the camera at full display resolution,
  // stays smooth (no round trip to the server needed for this pane).
  realCanvas.width = dispW;
  realCanvas.height = dispH;

  // Prediction pane: buffer is the tiny internal resolution the model
  // actually predicts at; CSS scales it up blockily (see style.css).
  predCanvas.width = cfg.width;
  predCanvas.height = cfg.height;
  predCanvas.style.width = dispW + "px";
  predCanvas.style.height = dispH + "px";

  sendCanvas = document.createElement("canvas");
  sendCanvas.width = cfg.width;
  sendCanvas.height = cfg.height;
  sendCtx = sendCanvas.getContext("2d");

  delaySlider.max = cfg.max_lag;
}

async function initCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
  video.srcObject = stream;
  await video.play();
}

function renderRealLoop() {
  if (video.readyState >= 2) {
    realCtx.drawImage(video, 0, 0, realCanvas.width, realCanvas.height);
  }
  requestAnimationFrame(renderRealLoop);
}

function startSendLoop() {
  if (sendTimer !== null) return;
  sendTimer = setInterval(() => {
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    if (video.readyState < 2 || !sendCanvas) return;
    sendCtx.drawImage(video, 0, 0, sendCanvas.width, sendCanvas.height);
    sendCanvas.toBlob(
      (blob) => {
        if (blob && ws && ws.readyState === WebSocket.OPEN) {
          blob.arrayBuffer().then((buf) => ws.send(buf));
        }
      },
      "image/jpeg",
      0.8
    );
  }, 1000 / SEND_FPS);
}

function stopSendLoop() {
  if (sendTimer !== null) {
    clearInterval(sendTimer);
    sendTimer = null;
  }
}

function connectWS() {
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  ws = new WebSocket(`${proto}//${location.host}/ws`);
  ws.binaryType = "arraybuffer";

  ws.onopen = () => {
    statsEl.textContent = "connected, waiting for first frame...";
  };

  ws.onmessage = (event) => {
    if (typeof event.data === "string") {
      const msg = JSON.parse(event.data);
      if (msg.type === "config") {
        applyConfig(msg);
      } else if (msg.type === "stats") {
        statsEl.textContent =
          `frame ${msg.frame_count}  fps ${msg.fps.toFixed(1)}  ` +
          `loss ${msg.avg_loss.toFixed(5)}  [${msg.mode_label}]  ` +
          `delay ${msg.target_lag}f  buf ${msg.buffer_len}/${msg.target_lag}`;
      }
    } else {
      // Binary prediction frame: 1-byte type tag + JPEG bytes.
      const bytes = new Uint8Array(event.data);
      const jpegBytes = bytes.slice(1);
      const blob = new Blob([jpegBytes], { type: "image/jpeg" });
      createImageBitmap(blob).then((bitmap) => {
        predCtx.drawImage(bitmap, 0, 0);
        bitmap.close();
      });
    }
  };

  ws.onclose = () => {
    statsEl.textContent = "disconnected -- reload to reconnect";
  };
}

delaySlider.addEventListener("input", () => {
  delayValueEl.textContent = delaySlider.value;
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: "set_delay", frames: Number(delaySlider.value) }));
  }
});

resetBtn.addEventListener("click", () => {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: "reset" }));
  }
});

// Stop pauses training server-side *and* stops the client from sending new
// frames -- submit_frame() auto-clears the server's pause flag on any
// inbound frame, so leaving the send loop running would undo Stop almost
// immediately. Clicking again (now labeled "Resume") restarts sending,
// which implicitly resumes training on the very next frame.
stopBtn.addEventListener("click", () => {
  stopped = !stopped;
  if (stopped) {
    stopSendLoop();
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: "stop" }));
    }
    stopBtn.textContent = "Resume";
  } else {
    startSendLoop();
    stopBtn.textContent = "Stop";
  }
});

async function main() {
  applyConfig(cfg); // sensible defaults until the server's config message arrives
  await initCamera();
  requestAnimationFrame(renderRealLoop);
  connectWS();
  startSendLoop();
}

main().catch((err) => {
  statsEl.textContent = `error: ${err.message}`;
  console.error(err);
});
