const FPS_ERROR = 20;
// Receipt minus the frame's dora timestamp: transport and queue delay.
const LATENCY_WARN_MS = 50;
const LATENCY_ERROR_MS = 100;
const JITTER_WARN_MS = 10;
const JITTER_ERROR_MS = 30;
const statsSource = new EventSource("/stats");
statsSource.onmessage = (e) => {
  const stats = JSON.parse(e.data);
  // Only the cameras this dataflow wired are reported; the rest of
  // the rows stay hidden instead of reading "-- Hz" forever.
  for (const row of document.querySelectorAll("[data-cam]")) {
    row.hidden = !(row.dataset.cam in stats);
    const key = "cam:" + row.dataset.cam;
    if (row.hidden) {
      setAlert(key, null);
      continue;
    }
    // Only a camera that has stopped raises the errors panel. A low
    // rate, delay or jitter is shown in the row's colours instead.
    const name = TEXT.camera + " " + row.querySelector(".cam-name").textContent;
    const s = stats[row.dataset.cam];
    if (!s || s.fps <= 0) {
      setAlert(key, "error", `${name}: ${TEXT.camera_not_received}`);
    } else {
      setAlert(key, null);
    }
  }
  document.getElementById("camera-panel").hidden =
    Object.keys(stats).length === 0;
  for (const [cam, s] of Object.entries(stats)) {
    const row = document.querySelector(`[data-cam="${cam}"]`);
    if (!row) continue;
    const fpsEl = row.querySelector(".cam-fps");
    const jitterEl = row.querySelector(".cam-jitter");
    if (!s || s.fps <= 0) {
      fpsEl.textContent = "-- Hz";
      jitterEl.textContent = "Δ -- ms";
      row.querySelector(".cam-latency").textContent = TEXT.latency + " --";
      row.classList.add("stale");
      row.classList.remove("warn", "error");
      continue;
    }
    row.classList.remove("stale");
    fpsEl.textContent = s.fps.toFixed(1) + " Hz";
    jitterEl.textContent = "Δ " + s.jitter_ms.toFixed(1) + " ms";
    const latencyEl = row.querySelector(".cam-latency");
    const latency = s.latency_ms ?? 0;
    const latencyLevel = latency > LATENCY_ERROR_MS ? "error"
      : latency > LATENCY_WARN_MS ? "warn" : null;
    latencyEl.textContent = TEXT.latency + " " + latency.toFixed(0) + " ms";
    latencyEl.classList.toggle("error", latencyLevel === "error");
    latencyEl.classList.toggle("warn", latencyLevel === "warn");
    const isError = s.fps < FPS_ERROR || s.jitter_ms > JITTER_ERROR_MS;
    const isWarn = !isError && s.jitter_ms > JITTER_WARN_MS;
    row.classList.toggle("error", isError);
    row.classList.toggle("warn", isWarn);
  }
};
