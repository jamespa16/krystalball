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
const saveCheckpointBtn = document.getElementById("saveCheckpointBtn");

// --- metric-vs-time small multiples ------------------------------------
// Fixed slot order + hue per series (categorical identity assigned by name,
// never by arrival order or rank -- a series that shows up later still gets
// its own permanent color). "fps" is the engine's rolling frame rate;
// "total" is the reported `avg_loss` (the summed, already-weighted training
// loss); the rest mirror training.py's `compute_training_loss` breakdown
// keys and only appear once that term is actually enabled (a CLI weight >
// 0), so the panel set adapts to config.
const METRIC_CHARTS_EL = document.getElementById("metricCharts");
const SERIES_ORDER = [
  "fps",
  "total",
  "base",
  "flow_smoothness",
  "flow_accel_smoothness",
  "motion_delta",
  "multistep_pixel",
  "multistep_latent",
  "adversarial",
];
const SERIES_COLOR = {
  fps: "#4ade80", // matches the app's own --accent, not a categorical slot --
  // fps is never shown side-by-side with a loss term for comparison (each
  // panel is its own single-series facet), so it doesn't need a distinct
  // categorical hue of its own.
  total: "#3987e5",
  base: "#d95926",
  flow_smoothness: "#199e70",
  flow_accel_smoothness: "#c98500",
  motion_delta: "#d55181",
  multistep_pixel: "#008300",
  multistep_latent: "#9085e9",
  adversarial: "#e66767",
};
const SERIES_LABEL = {
  fps: "fps",
  total: "total (avg_loss)",
  base: "base",
  flow_smoothness: "flow smoothness",
  flow_accel_smoothness: "flow accel smoothness",
  motion_delta: "motion delta",
  multistep_pixel: "multistep pixel",
  multistep_latent: "multistep latent",
  adversarial: "adversarial",
};
const SERIES_DECIMALS = { fps: 1 }; // everything else defaults to 5 (loss-scale values)
function formatMetricValue(key, value) {
  return value.toFixed(SERIES_DECIMALS[key] ?? 5);
}
const METRIC_HISTORY_MAX_POINTS = 400; // ~kept per series, oldest dropped first

const metricHistory = new Map(); // key -> number[]
const metricCharts = new Map(); // key -> { canvas, ctx, valueEl }
let metricChartHover = null; // { key, xFrac } | null

function ensureMetricChart(key) {
  if (metricCharts.has(key)) return metricCharts.get(key);

  const panel = document.createElement("div");
  panel.className = "metric-chart";

  const title = document.createElement("div");
  title.className = "metric-chart-title";
  const nameEl = document.createElement("span");
  nameEl.textContent = SERIES_LABEL[key] || key;
  nameEl.style.color = SERIES_COLOR[key] || "var(--chart-secondary)";
  const valueEl = document.createElement("span");
  valueEl.className = "metric-chart-value";
  title.appendChild(nameEl);
  title.appendChild(valueEl);

  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d");

  panel.appendChild(title);
  panel.appendChild(canvas);

  // Insert panels in fixed series order regardless of arrival order, so a
  // term that only starts reporting later doesn't reshuffle earlier panels.
  const order = SERIES_ORDER.indexOf(key);
  let inserted = false;
  for (const child of METRIC_CHARTS_EL.children) {
    const childKey = child.dataset.metricKey;
    if (SERIES_ORDER.indexOf(childKey) > order) {
      METRIC_CHARTS_EL.insertBefore(panel, child);
      inserted = true;
      break;
    }
  }
  if (!inserted) METRIC_CHARTS_EL.appendChild(panel);
  panel.dataset.metricKey = key;

  canvas.addEventListener("mousemove", (ev) => {
    const rect = canvas.getBoundingClientRect();
    const xFrac = Math.min(1, Math.max(0, (ev.clientX - rect.left) / rect.width));
    metricChartHover = { key, xFrac };
    drawMetricChart(key);
  });
  canvas.addEventListener("mouseleave", () => {
    if (metricChartHover && metricChartHover.key === key) {
      metricChartHover = null;
      drawMetricChart(key);
    }
  });

  const entry = { canvas, ctx, valueEl };
  metricCharts.set(key, entry);
  return entry;
}

function pushMetricSample(key, value) {
  let history = metricHistory.get(key);
  if (!history) {
    history = [];
    metricHistory.set(key, history);
  }
  history.push(value);
  if (history.length > METRIC_HISTORY_MAX_POINTS) history.shift();
}

function drawMetricChart(key) {
  const entry = metricCharts.get(key);
  const history = metricHistory.get(key);
  if (!entry || !history || history.length === 0) return;
  const { canvas, ctx, valueEl } = entry;

  // Crisp lines at the canvas's actual device pixel ratio. Measured from the
  // panel (parent)'s content width, not the canvas's own clientWidth, so
  // resizing the canvas's backing store can never feed back into the
  // measurement that drives it (a <canvas>'s intrinsic size is its
  // width/height attributes, which participate in grid/flex sizing).
  const dpr = window.devicePixelRatio || 1;
  const panelStyle = getComputedStyle(canvas.parentElement);
  const cssWidth =
    canvas.parentElement.clientWidth -
      parseFloat(panelStyle.paddingLeft) -
      parseFloat(panelStyle.paddingRight) || 220;
  const cssHeight = 64;
  if (canvas.width !== Math.round(cssWidth * dpr) || canvas.height !== Math.round(cssHeight * dpr)) {
    canvas.width = Math.round(cssWidth * dpr);
    canvas.height = Math.round(cssHeight * dpr);
  }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, cssWidth, cssHeight);

  const latest = history[history.length - 1];
  valueEl.textContent = formatMetricValue(key, latest);

  let min = Math.min(...history);
  let max = Math.max(...history);
  if (min === max) {
    // Flat series (e.g. a term stuck at exactly 0): synthesize a small
    // band around the value so the line still renders instead of
    // collapsing onto the baseline.
    const pad = Math.abs(min) > 0 ? Math.abs(min) * 0.1 : 1;
    min -= pad;
    max += pad;
  } else {
    const pad = (max - min) * 0.08;
    min -= pad;
    max += pad;
  }

  const padTop = 4;
  const padBottom = 4;
  const plotHeight = cssHeight - padTop - padBottom;
  const color = SERIES_COLOR[key] || "#c3c2b7";

  // Recessive hairline gridline at the baseline (most-recent-value level
  // isn't special here, so just mark the vertical midline for scale).
  ctx.strokeStyle = "#2c2c2a";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(0, padTop + plotHeight);
  ctx.lineTo(cssWidth, padTop + plotHeight);
  ctx.stroke();

  const n = history.length;
  const xStep = n > 1 ? cssWidth / (n - 1) : 0;
  const yFor = (v) => padTop + plotHeight - ((v - min) / (max - min)) * plotHeight;

  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.lineJoin = "round";
  ctx.lineCap = "round";
  ctx.beginPath();
  for (let i = 0; i < n; i++) {
    const x = i * xStep;
    const y = yFor(history[i]);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();

  // End-dot marking the current value, with a surface ring so it stays
  // legible where the line runs close to the edge.
  const endX = (n - 1) * xStep;
  const endY = yFor(history[n - 1]);
  ctx.beginPath();
  ctx.arc(endX, endY, 4, 0, Math.PI * 2);
  ctx.fillStyle = "#1b1e26";
  ctx.fill();
  ctx.beginPath();
  ctx.arc(endX, endY, 3, 0, Math.PI * 2);
  ctx.fillStyle = color;
  ctx.fill();

  // Hover crosshair + readout for the nearest sample to the cursor.
  if (metricChartHover && metricChartHover.key === key) {
    const idx = Math.min(n - 1, Math.round(metricChartHover.xFrac * (n - 1)));
    const hx = idx * xStep;
    const hy = yFor(history[idx]);
    ctx.strokeStyle = "#898781";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(hx, 0);
    ctx.lineTo(hx, cssHeight);
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(hx, hy, 3, 0, Math.PI * 2);
    ctx.fillStyle = "#1b1e26";
    ctx.fill();
    ctx.beginPath();
    ctx.arc(hx, hy, 2.5, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.fill();

    const stepsAgo = n - 1 - idx;
    const label = `${formatMetricValue(key, history[idx])}${stepsAgo > 0 ? `  (-${stepsAgo}f)` : ""}`;
    ctx.font = "10px ui-monospace, monospace";
    const textWidth = ctx.measureText(label).width;
    let tx = hx + 6;
    if (tx + textWidth + 4 > cssWidth) tx = hx - textWidth - 6;
    const ty = hy < cssHeight / 2 ? hy + 14 : hy - 8;
    ctx.fillStyle = "#ffffff";
    ctx.fillText(label, tx, ty);
  }
}

function updateMetricCharts(msg) {
  pushMetricSample("fps", msg.fps);
  ensureMetricChart("fps");
  pushMetricSample("total", msg.avg_loss);
  ensureMetricChart("total");
  for (const [name, value] of Object.entries(msg.loss_breakdown || {})) {
    if (!(name in SERIES_COLOR)) continue; // unknown term: ignore rather than guess a color
    pushMetricSample(name, value);
    ensureMetricChart(name);
  }
  for (const key of metricCharts.keys()) drawMetricChart(key);
}

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
        updateMetricCharts(msg);
        const breakdown = Object.entries(msg.loss_breakdown || {})
          .map(([name, value]) => `${name} ${value.toFixed(5)}`)
          .join("  ");
        const weights = Object.entries(msg.loss_weights || {})
          .map(([name, value]) => `${name} ${value.toFixed(4)}`)
          .join("  ");
        // In FORECAST mode the prediction pane is showing a genuine
        // look-ahead generated from the model's current weights, N frames
        // ahead (N = delay) -- forecast_step/forecast_horizon show where
        // in that N-frame window the current displayed frame sits.
        const modeDetail =
          msg.mode_label === "FORECAST"
            ? `${msg.mode_label} ${msg.forecast_step}/${msg.forecast_horizon}`
            : msg.mode_label;
        statsEl.textContent =
          `frame ${msg.frame_count}  fps ${msg.fps.toFixed(1)}  ` +
          `loss ${msg.avg_loss.toFixed(5)}  [${modeDetail}]  ` +
          `delay ${msg.target_lag}f  buf ${msg.buffer_len}/${msg.target_lag}` +
          (breakdown ? `\n${breakdown}` : "") +
          (weights ? `\nweights: ${weights}` : "");
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

saveCheckpointBtn.addEventListener("click", () => {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: "save_checkpoint" }));
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
