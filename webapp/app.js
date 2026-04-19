"use strict";

// ─── Element refs ────────────────────────────────────────────────────────────
const els = {
  emergency:   document.getElementById("emergency"),
  emergencyX:  document.querySelector(".emergency__dismiss"),
  status:      document.getElementById("status"),
  feedMeta:    document.getElementById("feed-meta"),
  hapticMeta:  document.getElementById("haptic-meta"),
  compass:     document.getElementById("compass"),
  hbtns:       Object.fromEntries(
    [...document.querySelectorAll(".hbtn[data-dir]")].map(el => [el.dataset.dir, el])
  ),
  imuMeta:     document.getElementById("imu-meta"),
  imuChart:    document.getElementById("imu-chart"),
  transcript:  document.getElementById("transcript"),
  distState:   document.getElementById("dist-state"),
  connState:   document.getElementById("conn-state"),
  feedFps:     document.getElementById("feed-fps"),
  feedTime:    document.getElementById("feed-time"),
  proxRows:    Object.fromEntries(
    [...document.querySelectorAll(".prox__row[data-dir]")].map(el => [el.dataset.dir, el])
  ),
};

// ─── Hi-DPI canvas setup ────────────────────────────────────────────────────
function fitCanvas(canvas) {
  const dpr  = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  if (rect.width === 0 || rect.height === 0) return null;
  canvas.width  = Math.round(rect.width  * dpr);
  canvas.height = Math.round(rect.height * dpr);
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, w: rect.width, h: rect.height };
}

let imuView = fitCanvas(els.imuChart);

window.addEventListener("resize", () => {
  imuView = fitCanvas(els.imuChart);
  drawIMU();
});

// ─── State ──────────────────────────────────────────────────────────────────
const imuTrace      = { x: [], y: [], z: [] };
const IMU_WINDOW    = 240;
const hapticTimers  = new Map();
const HAPTIC_FADE_MS = 480;
let lastDetections  = [];
let fallTimeout     = null;
let compassFiringId = null;

// Per-direction proximity values (mm). Smoothed display values are tracked
// alongside so we can RAF-interpolate the visible number to the latest reading.
const proxTarget    = { F: null, B: null, L: null, R: null };
const proxDisplay   = { F: null, B: null, L: null, R: null };
let proxAnimRaf     = null;

// ─── WebSocket ──────────────────────────────────────────────────────────────
// Always pull live data from the Pi, even when the HTML is served from a
// different machine (e.g. a teammate running their own local web server).
// Falls back to same-origin when the page is already served by the Pi.
const BELT_PI_HOST = "10.10.9.207:8000";
const wsHost = (location.host === BELT_PI_HOST || location.hostname === "10.10.9.207")
  ? location.host
  : BELT_PI_HOST;
const ws = new WebSocket(`ws://${wsHost}/ws`);

ws.onopen  = () => setConn(true);
ws.onclose = () => setConn(false);
ws.onerror = () => setConn(false);
ws.onmessage = ev => {
  let msg;
  try { msg = JSON.parse(ev.data); } catch { return; }
  dispatch(msg);
};

function setConn(connected) {
  if (!els.connState) return;
  els.connState.className = `nav__conn ${connected ? "connected" : "disconnected"}`;
  els.connState.querySelector(".conn-label").textContent = connected ? "Connected" : "Disconnected";
}

function dispatch(m) {
  switch (m.t) {
    case "detections": onDetections(m.boxes); break;
    case "imu":        onIMU(m);              break;
    case "distance":   onDistance(m);         break;
    case "haptic":     onHaptic(m);           break;
    case "voice":      onVoice(m);            break;
    case "fall":       onFall(m);             break;
    case "health":     onHealth(m);           break;
  }
}

// ─── YOLO overlay ───────────────────────────────────────────────────────────
// Boxes are baked into the /mjpeg-overlay stream server-side, so we no longer
// draw on the canvas — we just use detection events to update the meta line.
function onDetections(boxes) {
  lastDetections = boxes || [];
  updateDetectionSummary();
}

function updateDetectionSummary() {
  const meta = document.getElementById("feed-meta");
  if (!meta) return;
  const n = lastDetections.length;
  if (n === 0) {
    meta.textContent = "No detections";
    return;
  }
  const top = lastDetections[0];
  meta.textContent = n === 1
    ? `${top.cls} · ${(top.conf * 100).toFixed(0)}%`
    : `${n} objects · top: ${top.cls}`;
}

// ─── IMU chart ──────────────────────────────────────────────────────────────
function onIMU(m) {
  pushTrace(imuTrace.x, m.ax);
  pushTrace(imuTrace.y, m.ay);
  pushTrace(imuTrace.z, m.az);
  els.imuMeta.textContent =
    `${fmt(m.ax)}  ${fmt(m.ay)}  ${fmt(m.az)}`;
  drawIMU();
}

function fmt(v) {
  const s = v.toFixed(2);
  return v >= 0 ? `+${s}` : s;
}

function pushTrace(arr, v) {
  arr.push(v);
  if (arr.length > IMU_WINDOW) arr.shift();
}

function drawIMU() {
  if (!imuView) imuView = fitCanvas(els.imuChart);
  if (!imuView) return;
  const { ctx, w, h } = imuView;
  ctx.clearRect(0, 0, w, h);

  // Faint midline
  ctx.strokeStyle = "rgba(0, 0, 0, 0.06)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(0, h / 2);
  ctx.lineTo(w, h / 2);
  ctx.stroke();

  drawSeries(ctx, w, h, imuTrace.x, "#7a9e7e"); // sage
  drawSeries(ctx, w, h, imuTrace.y, "#d4a574"); // honey
  drawSeries(ctx, w, h, imuTrace.z, "#c8847a"); // coral
}

function drawSeries(ctx, w, h, arr, color) {
  if (arr.length < 2) return;
  const range = 12;

  ctx.strokeStyle = color;
  ctx.lineWidth   = 1.6;
  ctx.lineCap     = "round";
  ctx.lineJoin    = "round";
  ctx.beginPath();
  for (let i = 0; i < arr.length; i++) {
    const x = (i / (IMU_WINDOW - 1)) * w;
    const y = h / 2 - (arr[i] / range) * (h / 2 - 6);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();
}

// ─── Proximity (4 directions) ────────────────────────────────────────────────
//
// Backend message shapes accepted (frontend stays compatible while hardware
// grows from one ToF sensor to four):
//   {"t":"distance","mm":420}                  → treated as Front
//   {"t":"distance","dir":"F","mm":420}        → single direction
//   {"t":"distance","F":420,"B":1500,"L":...}  → bulk update
function onDistance(m) {
  if (typeof m.mm === "number" && !m.dir) {
    setProxTarget("F", m.mm);
  } else if (m.dir && typeof m.mm === "number") {
    setProxTarget(m.dir, m.mm);
  } else {
    for (const d of ["F", "B", "L", "R"]) {
      if (typeof m[d] === "number") setProxTarget(d, m[d]);
    }
  }
  if (!proxAnimRaf) animateProx();
  renderProxState();
}

function setProxTarget(dir, mm) {
  proxTarget[dir] = mm;
  if (proxDisplay[dir] === null) proxDisplay[dir] = mm;
}

// Distance range used to compute the proximity bar fill (closer = fuller).
const PROX_RANGE_MM = 2000;

function animateProx() {
  const tick = () => {
    let stillAnimating = false;
    for (const d of ["F", "B", "L", "R"]) {
      const tgt = proxTarget[d];
      if (tgt === null) continue;
      const cur  = proxDisplay[d];
      const diff = tgt - cur;
      if (Math.abs(diff) < 0.5) {
        proxDisplay[d] = tgt;
      } else {
        proxDisplay[d] = cur + diff * 0.2;
        stillAnimating = true;
      }
      renderProxRow(d);
    }
    proxAnimRaf = stillAnimating ? requestAnimationFrame(tick) : null;
  };
  proxAnimRaf = requestAnimationFrame(tick);
}

function renderProxRow(dir) {
  const row = els.proxRows[dir];
  if (!row) return;
  const valueEl = row.querySelector(".prox__value");
  const fillEl  = row.querySelector(".prox__fill");
  const v       = proxDisplay[dir];
  if (v === null) {
    valueEl.textContent = "—";
    if (fillEl) fillEl.style.width = "0%";
    row.className = "prox__row";
    return;
  }
  valueEl.innerHTML = `${Math.round(v)}<span class="prox__unit">mm</span>`;
  if (fillEl) {
    const pct = Math.max(4, Math.min(100, (1 - v / PROX_RANGE_MM) * 100));
    fillEl.style.width = `${pct}%`;
  }
  const danger = v < 400;
  const warn   = v < 800;
  const state  = danger ? "is-danger" : warn ? "is-warn" : "is-ok";
  row.className = `prox__row ${state}`;
}

function renderProxState() {
  if (!els.distState) return;
  const valid = Object.values(proxTarget).filter(v => v !== null);
  if (valid.length === 0) {
    els.distState.textContent = "Standby";
    els.distState.className   = "";
    return;
  }
  const min    = Math.min(...valid);
  const danger = min < 400;
  const warn   = min < 800;
  els.distState.textContent = danger ? "Danger" : warn ? "Caution" : "Clear";
  els.distState.className   = danger ? "is-danger" : warn ? "is-warn" : "is-ok";
}

// ─── Haptic ──────────────────────────────────────────────────────────────────
function onHaptic(m) {
  const dirs = m.dir === "ALL" ? ["F", "B", "L", "R"] : [m.dir];

  for (const d of dirs) {
    const el = els.hbtns[d];
    if (!el) continue;
    el.classList.add("is-active");
    clearTimeout(hapticTimers.get(d));
    hapticTimers.set(
      d,
      setTimeout(() => el.classList.remove("is-active"),
        Math.max(m.duration_ms ?? 400, HAPTIC_FADE_MS))
    );
  }

  if (els.compass) {
    // Re-trigger pulse animation by toggling class
    els.compass.classList.remove("is-firing");
    void els.compass.offsetWidth;
    els.compass.classList.add("is-firing");
    clearTimeout(compassFiringId);
    compassFiringId = setTimeout(() => els.compass.classList.remove("is-firing"), 700);
  }

  els.hapticMeta.textContent = m.dir === "ALL"
    ? "All directions"
    : `${dirName(m.dir)} · ${m.intensity ?? "—"}`;
}

function dirName(d) {
  return { F: "Forward", B: "Back", L: "Left", R: "Right" }[d] ?? d;
}

// ─── Voice ───────────────────────────────────────────────────────────────────
function onVoice(m) {
  const empty = els.transcript.querySelector(".transcript__empty");
  if (empty) empty.remove();

  const div = document.createElement("div");
  div.className = `turn turn--${m.role}`;
  div.textContent = m.text;
  els.transcript.appendChild(div);

  // Smooth auto-scroll
  els.transcript.scrollTo({
    top: els.transcript.scrollHeight,
    behavior: "smooth",
  });
}

// ─── Fall / emergency ────────────────────────────────────────────────────────
function onFall(_m) {
  els.emergency.classList.remove("hidden");
  clearTimeout(fallTimeout);
  fallTimeout = setTimeout(() => els.emergency.classList.add("hidden"), 8000);
}

if (els.emergencyX) {
  els.emergencyX.addEventListener("click", () => {
    els.emergency.classList.add("hidden");
    clearTimeout(fallTimeout);
  });
}

// ─── Health ──────────────────────────────────────────────────────────────────
function onHealth(m) {
  setChip("serial",  m.serial_ok    ? "ok" : "bad");
  setChip("yolo",    m.yolo_fps >= 3 ? "ok" : m.yolo_fps > 0 ? "warn" : "bad");
  setChip("warmup",  m.warmup_ok    ? "ok" : "warn");
  setChip("ollama",  m.ollama_alive ? "ok" : "bad");
  if (els.feedFps) {
    const streamFps = m.stream_fps ?? m.camera_fps ?? 0;
    els.feedFps.textContent = `${streamFps.toFixed(1)} fps`;
    els.feedFps.title = `video ${streamFps.toFixed(1)} · yolo ${m.yolo_fps.toFixed(1)} · cam ${(m.camera_fps ?? 0).toFixed(1)}`;
  }
}

function setChip(key, state) {
  const el = els.status.querySelector(`[data-key="${key}"]`);
  if (!el) return;
  el.classList.remove("is-ok", "is-warn", "is-bad");
  el.classList.add(`is-${state}`);
}

// ─── Feed clock ──────────────────────────────────────────────────────────────
function tickFeedClock() {
  if (!els.feedTime) return;
  const d = new Date();
  const pad = n => String(n).padStart(2, "0");
  els.feedTime.textContent = `${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`;
}
setInterval(tickFeedClock, 1000);
tickFeedClock();
