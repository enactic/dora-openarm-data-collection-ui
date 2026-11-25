const VR_FPS_ERROR = 50;        // expect ~72 Hz from the headset
const VR_JITTER_WARN_MS = 10;
const VR_JITTER_ERROR_MS = 30;
const vrStatsSource = new EventSource("/vr-stats");
vrStatsSource.onmessage = (e) => {
  const s = JSON.parse(e.data);
  document.getElementById("vr-panel").hidden = !s.present;
  if (s.present) rateAlert("vr", "VR", s, VR_FPS_ERROR, VR_JITTER_WARN_MS, VR_JITTER_ERROR_MS);
  else setAlert("vr", null);
  const row = document.querySelector('[data-vr="vr_rx"]');
  if (!row) return;
  const fpsEl = row.querySelector(".cam-fps");
  const jitterEl = row.querySelector(".cam-jitter");
  if (!s || s.fps <= 0) {
    fpsEl.textContent = "-- Hz";
    jitterEl.textContent = "Δ -- ms";
    row.classList.add("stale");
    row.classList.remove("warn", "error");
    return;
  }
  row.classList.remove("stale");
  fpsEl.textContent = s.fps.toFixed(1) + " Hz";
  jitterEl.textContent = "Δ " + s.jitter_ms.toFixed(1) + " ms";
  const isError = s.fps < VR_FPS_ERROR || s.jitter_ms > VR_JITTER_ERROR_MS;
  const isWarn = !isError && s.jitter_ms > VR_JITTER_WARN_MS;
  row.classList.toggle("error", isError);
  row.classList.toggle("warn", isWarn);
};
const KER_FPS_ERROR = 100;      // the leader is driven at 250 Hz
const KER_JITTER_WARN_MS = 10;
const KER_JITTER_ERROR_MS = 30;
const kerStatsSource = new EventSource("/ker-stats");
kerStatsSource.onmessage = (e) => {
  const s = JSON.parse(e.data);
  document.getElementById("ker-panel").hidden = !s.present;
  if (s.present) rateAlert("ker", "KER", s, KER_FPS_ERROR, KER_JITTER_WARN_MS, KER_JITTER_ERROR_MS);
  else setAlert("ker", null);
  const row = document.querySelector('[data-ker="ker_rx"]');
  if (!row) return;
  const name = document.querySelector("#ker-panel .device-name");
  // Sent once at startup, so it is absent for a UI that connected
  // later. The panel is worth showing either way.
  if (name && s.device && (s.device.hw || s.device.fw)) {
    name.textContent =
      "hw " + (s.device.hw ?? "?") + " / fw " + (s.device.fw ?? "?");
  }
  const fpsEl = row.querySelector(".cam-fps");
  const jitterEl = row.querySelector(".cam-jitter");
  if (!s || s.fps <= 0) {
    fpsEl.textContent = "-- Hz";
    jitterEl.textContent = "\u0394 -- ms";
    row.classList.add("stale");
    row.classList.remove("warn", "error");
    return;
  }
  row.classList.remove("stale");
  fpsEl.textContent = s.fps.toFixed(1) + " Hz";
  jitterEl.textContent = "\u0394 " + s.jitter_ms.toFixed(1) + " ms";
  const isError = s.fps < KER_FPS_ERROR || s.jitter_ms > KER_JITTER_ERROR_MS;
  const isWarn = !isError && s.jitter_ms > KER_JITTER_WARN_MS;
  row.classList.toggle("error", isError);
  row.classList.toggle("warn", isWarn);
};
