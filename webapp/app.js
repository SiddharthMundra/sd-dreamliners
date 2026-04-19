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
  fusionCard:  document.querySelector(".card--fusion"),
  fusionStrip: document.getElementById("fusion-strip"),
  fusionCore:  document.getElementById("fusion-core-label"),
  navMap:      document.getElementById("nav-map"),
  navBearing:  document.getElementById("nav-bearing"),
  navThreat:   document.getElementById("nav-threat"),
  navZone:     document.getElementById("nav-zone"),
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
let lastImuHeading  = { ax: 0, ay: 0, az: 1 };
let lastHapticDir   = null;
let lastHapticAt    = 0;

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
    case "demo_nav":   onDemoNav(m);          break;
  }
}

// ─── YOLO overlay ───────────────────────────────────────────────────────────
// Boxes are baked into the /mjpeg-overlay stream server-side, so we no longer
// draw on the canvas — we just use detection events to update the meta line.
function onDetections(boxes) {
  lastDetections = boxes || [];
  updateDetectionSummary();
  updateFusionStrip();
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
  lastImuHeading.ax = m.ax;
  lastImuHeading.ay = m.ay;
  lastImuHeading.az = m.az;
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
// If both mm and F/B/L/R exist (Pi always sends mm + per-sector mm), we must
// apply the bulk fields — legacy code only set Front from mm and dropped L/R.
function onDistance(m) {
  const hasBulk = ["F", "B", "L", "R"].some((d) => typeof m[d] === "number");
  if (hasBulk) {
    for (const d of ["F", "B", "L", "R"]) {
      if (typeof m[d] === "number") setProxTarget(d, m[d]);
    }
  } else if (m.dir && typeof m.mm === "number") {
    setProxTarget(m.dir, m.mm);
  } else if (typeof m.mm === "number") {
    setProxTarget("F", m.mm);
  }
  if (!proxAnimRaf) animateProx();
  renderProxState();
  updateFusionStrip();
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
      updateFusionRadarWedge(d);
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
    els.distState.className   = "card__meta card__meta--fusion";
    if (els.fusionCard) {
      els.fusionCard.classList.remove("fusion--hot", "fusion--warn", "fusion--clear");
    }
    if (els.fusionCore) els.fusionCore.textContent = "SYNC";
    return;
  }
  const min    = Math.min(...valid);
  const danger = min < 400;
  const warn   = min < 800;
  els.distState.textContent = danger ? "Danger" : warn ? "Caution" : "Clear";
  const pill = danger ? "is-danger" : warn ? "is-warn" : "is-ok";
  els.distState.className = `card__meta card__meta--fusion ${pill}`;
  if (els.fusionCard) {
    els.fusionCard.classList.remove("fusion--hot", "fusion--warn", "fusion--clear");
    els.fusionCard.classList.add(danger ? "fusion--hot" : warn ? "fusion--warn" : "fusion--clear");
  }
  if (els.fusionCore) {
    els.fusionCore.textContent = danger ? "ALERT" : warn ? "ARM" : "CALM";
  }
  for (const d of ["F", "R", "B", "L"]) {
    updateFusionRadarWedge(d);
  }
}

function wedgeOpacityMm(mm) {
  if (mm === null || mm === undefined) return 0.05;
  const t = Math.max(0, Math.min(1, 1 - mm / 2000));
  return 0.07 + t * 0.62;
}

function wedgeColor(mm) {
  if (mm === null || mm === undefined) return "rgba(122,158,126,0.25)";
  if (mm < 400) return "rgba(200,132,122,0.85)";
  if (mm < 800) return "rgba(212,165,116,0.75)";
  return "rgba(122,158,126,0.55)";
}

function updateFusionRadarWedge(dir) {
  const el = document.getElementById(`fusion-wedge-${dir}`);
  if (!el) return;
  const v = proxDisplay[dir];
  if (v === null) {
    el.style.opacity = "0.05";
    el.style.fill = "rgba(122,158,126,0.2)";
    return;
  }
  el.style.opacity = String(wedgeOpacityMm(v));
  el.style.fill = wedgeColor(v);
}

function updateFusionStrip() {
  if (!els.fusionStrip) return;
  const parts = [];
  const n = lastDetections.length;
  if (n === 0) parts.push("Vision idle");
  else {
    const top = lastDetections[0];
    parts.push(`${n} det · ${top.cls} ${(top.conf * 100).toFixed(0)}%`);
  }
  const valid = ["F", "R", "B", "L"].map(d => proxTarget[d]).filter(v => v !== null);
  if (valid.length) {
    const mn = Math.round(Math.min(...valid));
    parts.push(`min ${mn} mm`);
  } else parts.push("US pair standby");
  if (lastHapticDir && performance.now() - lastHapticAt < 1800) {
    parts.push(`haptic ${lastHapticDir}`);
  }
  els.fusionStrip.textContent = parts.join(" · ");
}

// ─── Haptic ──────────────────────────────────────────────────────────────────
function onHaptic(m) {
  lastHapticDir = m.dir;
  lastHapticAt = performance.now();
  updateFusionStrip();

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

// ─── Demo nav override ───────────────────────────────────────────────────────
function onDemoNav(m) {
  if (els.navBearing && m.bearing !== undefined) {
    els.navBearing.textContent = `HDG ${String(m.bearing).padStart(3, "0")}°`;
  }
  if (els.navThreat && m.threat) els.navThreat.textContent = m.threat;
  if (els.navZone && m.zone) els.navZone.textContent = m.zone;
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

// ─── Browser PTT mic ────────────────────────────────────────────────────────
const pttBtn = document.getElementById("ptt-btn");
const voiceStatus = document.getElementById("voice-status");
let mediaStream = null;
let mediaRecorder = null;
let audioChunks = [];
let pttActive = false;

async function startPTT() {
  if (pttActive) return;
  pttActive = true;
  audioChunks = [];
  pttBtn.classList.add("is-recording");
  if (voiceStatus) voiceStatus.textContent = "Listening...";
  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: { sampleRate: 16000, channelCount: 1, echoCancellation: true }
    });
    const ctx = new AudioContext({ sampleRate: 16000 });
    const src = ctx.createMediaStreamSource(mediaStream);
    const proc = ctx.createScriptProcessor(4096, 1, 1);
    const pcmBufs = [];
    proc.onaudioprocess = e => {
      const f32 = e.inputBuffer.getChannelData(0);
      const i16 = new Int16Array(f32.length);
      for (let i = 0; i < f32.length; i++) {
        i16[i] = Math.max(-32768, Math.min(32767, Math.round(f32[i] * 32767)));
      }
      pcmBufs.push(i16.buffer);
    };
    src.connect(proc);
    proc.connect(ctx.destination);
    pttBtn._ctx = ctx;
    pttBtn._proc = proc;
    pttBtn._src = src;
    pttBtn._pcmBufs = pcmBufs;
  } catch (e) {
    console.error("mic error:", e);
    if (voiceStatus) voiceStatus.textContent = "Mic denied";
    stopPTT();
  }
}

function stopPTT() {
  if (!pttActive) return;
  pttActive = false;
  pttBtn.classList.remove("is-recording");
  if (voiceStatus) voiceStatus.textContent = "Processing...";
  if (pttBtn._proc) { pttBtn._proc.disconnect(); pttBtn._proc = null; }
  if (pttBtn._src) { pttBtn._src.disconnect(); pttBtn._src = null; }
  if (mediaStream) { mediaStream.getTracks().forEach(t => t.stop()); mediaStream = null; }
  const bufs = pttBtn._pcmBufs || [];
  if (bufs.length > 0 && ws.readyState === WebSocket.OPEN) {
    let total = 0;
    for (const b of bufs) total += b.byteLength;
    const merged = new Uint8Array(total);
    let off = 0;
    for (const b of bufs) { merged.set(new Uint8Array(b), off); off += b.byteLength; }
    ws.send(merged.buffer);
    if (voiceStatus) voiceStatus.textContent = `Sent ${(total/1024).toFixed(0)} KB`;
  } else {
    if (voiceStatus) voiceStatus.textContent = "No audio captured";
  }
  pttBtn._pcmBufs = [];
  if (pttBtn._ctx) { pttBtn._ctx.close(); pttBtn._ctx = null; }
  setTimeout(() => { if (voiceStatus && !pttActive) voiceStatus.textContent = "Tap mic to talk"; }, 3000);
}

if (pttBtn) {
  pttBtn.addEventListener("mousedown", startPTT);
  pttBtn.addEventListener("mouseup", stopPTT);
  pttBtn.addEventListener("mouseleave", stopPTT);
  pttBtn.addEventListener("touchstart", e => { e.preventDefault(); startPTT(); });
  pttBtn.addEventListener("touchend", e => { e.preventDefault(); stopPTT(); });
}

// ─── Fusion radar (initial paint) ─────────────────────────────────────────
for (const d of ["F", "R", "B", "L"]) {
  updateFusionRadarWedge(d);
}
updateFusionStrip();

// ─── Tactical nav map (synthetic city + fusion-driven hazards) ──────────────
const ZONES = ["Grid 7α", "Sector 12", "Block 4θ", "Corridor Ω", "Node 9"];
let navCtx = null;
let navCssW = 400;
let navCssH = 260;

function fitNavMap() {
  const c = els.navMap;
  if (!c || !c.parentElement) return;
  const rect = c.parentElement.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  navCssW = Math.max(200, rect.width);
  navCssH = Math.max(200, rect.height);
  c.width = Math.round(navCssW * dpr);
  c.height = Math.round(navCssH * dpr);
  c.style.width = `${navCssW}px`;
  c.style.height = `${navCssH}px`;
  navCtx = c.getContext("2d");
  navCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
}

function fusionMinMm() {
  const vals = ["F", "R", "B", "L"].map(d => proxDisplay[d]).filter(v => v !== null);
  if (!vals.length) return null;
  return Math.min(...vals);
}

function navThreatText() {
  const mn = fusionMinMm();
  if (mn === null) return "Awaiting ultrasonics";
  if (mn < 400) return `Proximity ${Math.round(mn)} mm`;
  if (mn < 800) return `Buffer ${Math.round(mn)} mm`;
  return "Corridor nominal";
}

function navBearingText() {
  const { ax, ay } = lastImuHeading;
  const deg = ((Math.atan2(ay, ax) * 180) / Math.PI + 360) % 360;
  return `HDG ${deg.toFixed(0).padStart(3, "0")}°`;
}

function drawIsoBlock(ctx, x, y, w, h, base, hue) {
  const skew = 0.42;
  ctx.fillStyle = base;
  ctx.beginPath();
  ctx.moveTo(x, y);
  ctx.lineTo(x + w, y);
  ctx.lineTo(x + w + h * skew, y + h * 0.35);
  ctx.lineTo(x + h * skew, y + h * 0.35);
  ctx.closePath();
  ctx.fill();
  ctx.fillStyle = hue;
  ctx.beginPath();
  ctx.moveTo(x + h * skew, y + h * 0.35);
  ctx.lineTo(x + w + h * skew, y + h * 0.35);
  ctx.lineTo(x + w, y + h);
  ctx.lineTo(x, y + h);
  ctx.closePath();
  ctx.fill();
  ctx.strokeStyle = "rgba(255,255,255,0.04)";
  ctx.lineWidth = 1;
  ctx.stroke();
}

let navBuildings = [];

function seedNavBuildings() {
  navBuildings = [];
  const rnd = (s) => {
    let x = s;
    return () => {
      x = (x * 16807) % 2147483647;
      return (x - 1) / 2147483646;
    };
  };
  const r = rnd(90210);
  for (let i = 0; i < 38; i++) {
    navBuildings.push({
      x: (r() - 0.5) * 2.4,
      y: (r() - 0.5) * 2.4,
      w: 0.08 + r() * 0.12,
      h: 0.06 + r() * 0.1,
      hue: r() * 0.25,
    });
  }
}

function drawNavMapFrame(tSec) {
  if (!navCtx) return;
  const ctx = navCtx;
  const W = navCssW;
  const H = navCssH;
  ctx.fillStyle = "#07090c";
  ctx.fillRect(0, 0, W, H);

  const cx = W * 0.5;
  const cy = H * 0.52;
  const driftX = Math.sin(tSec * 0.09) * 6;
  const driftY = Math.cos(tSec * 0.07) * 4;

  ctx.save();
  ctx.translate(cx + driftX, cy + driftY);

  ctx.strokeStyle = "rgba(122, 158, 126, 0.12)";
  ctx.lineWidth = 1;
  const gridR = Math.max(W, H) * 0.55;
  for (let a = 0; a < 6; a++) {
    const ang = (a / 6) * Math.PI * 2 + tSec * 0.04;
    ctx.beginPath();
    ctx.moveTo(0, 0);
    ctx.lineTo(Math.cos(ang) * gridR, Math.sin(ang) * gridR);
    ctx.stroke();
  }
  for (let rr = 40; rr < gridR; rr += 36) {
    ctx.beginPath();
    ctx.arc(0, 0, rr, 0, Math.PI * 2);
    ctx.strokeStyle = `rgba(122,158,126,${0.05 + (rr % 72) * 0.001})`;
    ctx.stroke();
  }

  const scale = Math.min(W, H) * 0.42;
  for (const b of navBuildings) {
    const px = b.x * scale;
    const py = b.y * scale;
    const bw = b.w * scale;
    const bh = b.h * scale;
    const lum = 18 + b.hue * 40;
    drawIsoBlock(
      ctx,
      px - bw / 2,
      py - bh / 2,
      bw,
      bh,
      `rgb(${lum},${lum + 8},${lum + 14})`,
      `rgba(122,158,126,${0.04 + b.hue * 0.08})`
    );
  }

  const mn = fusionMinMm();
  const pulse = 0.55 + 0.45 * Math.sin(tSec * 3.2);
  const dirs = [
    { d: "F", ang: -Math.PI / 2 },
    { d: "R", ang: -Math.PI / 4 },
    { d: "B", ang: Math.PI / 2 },
    { d: "L", ang: -3 * Math.PI / 4 },
  ];
  for (const { d, ang } of dirs) {
    const mm = proxDisplay[d];
    if (mm === null) continue;
    const threat = Math.max(0, Math.min(1, 1 - mm / 2000));
    const len = 28 + threat * (Math.min(W, H) * 0.38);
    const alpha = 0.12 + threat * 0.55 * pulse;
    let col = "122, 158, 126";
    if (mm < 400) col = "200, 132, 122";
    else if (mm < 800) col = "212, 165, 116";
    const gx = Math.cos(ang) * len;
    const gy = Math.sin(ang) * len;
    const g = ctx.createLinearGradient(0, 0, gx, gy);
    g.addColorStop(0, `rgba(${col},${alpha + 0.15})`);
    g.addColorStop(1, `rgba(${col},0)`);
    ctx.strokeStyle = g;
    ctx.lineWidth = 3 + threat * 5;
    ctx.beginPath();
    ctx.moveTo(0, 0);
    ctx.lineTo(gx, gy);
    ctx.stroke();
  }

  const wx = Math.cos(tSec * 0.31) * scale * 0.55;
  const wy = Math.sin(tSec * 0.27) * scale * 0.5;
  ctx.strokeStyle = "rgba(212, 165, 116, 0.55)";
  ctx.setLineDash([6, 10]);
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(0, 0);
  ctx.lineTo(wx, wy);
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = "rgba(212, 165, 116, 0.9)";
  ctx.beginPath();
  ctx.arc(wx, wy, 5, 0, Math.PI * 2);
  ctx.fill();

  ctx.fillStyle = "#e8f5e9";
  ctx.beginPath();
  ctx.moveTo(0, -12);
  ctx.lineTo(9, 10);
  ctx.lineTo(0, 5);
  ctx.lineTo(-9, 10);
  ctx.closePath();
  ctx.fill();
  ctx.strokeStyle = "rgba(122,158,126,0.5)";
  ctx.lineWidth = 1;
  ctx.stroke();

  ctx.restore();

  ctx.fillStyle = "rgba(255,255,255,0.04)";
  ctx.fillRect(0, 0, W, 22);
  ctx.fillStyle = "rgba(200,210,205,0.25)";
  ctx.font = "600 10px Inter, system-ui, sans-serif";
  ctx.fillText("TACTICAL GRID", 12, 14);
}

function navMapLoop(t) {
  const sec = t * 0.001;
  drawNavMapFrame(sec);
  if (els.navBearing) els.navBearing.textContent = navBearingText();
  if (els.navThreat) els.navThreat.textContent = navThreatText();
  if (els.navZone) {
    const zi = Math.floor(sec * 0.15 + Date.now() * 0.00001) % ZONES.length;
    els.navZone.textContent = ZONES[zi];
  }
  requestAnimationFrame(navMapLoop);
}

function initNavMap() {
  if (!els.navMap) return;
  seedNavBuildings();
  fitNavMap();
  requestAnimationFrame(navMapLoop);
  const p = els.navMap.parentElement;
  if (typeof ResizeObserver !== "undefined" && p) {
    new ResizeObserver(() => fitNavMap()).observe(p);
  }
  setTimeout(fitNavMap, 120);
}

window.addEventListener("resize", () => {
  fitNavMap();
});

initNavMap();
