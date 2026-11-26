const FPS_ERROR = 20;
const JITTER_WARN_MS = 10;
const JITTER_ERROR_MS = 30;

eventSource.addEventListener("camera-stats", (e) => {
  const stats = JSON.parse(e.data);
  // Only the cameras this dataflow wired are reported; the rest of
  // the rows stay hidden instead of reading "-- Hz" forever.
  for (const row of document.querySelectorAll("[data-cam]")) {
    row.hidden = !(row.dataset.cam in stats);
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
      row.classList.add("stale");
      row.classList.remove("warn", "error");
      continue;
    }
    row.classList.remove("stale");
    fpsEl.textContent = s.fps.toFixed(1) + " Hz";
    jitterEl.textContent = "Δ " + s.jitter_ms.toFixed(1) + " ms";
    const isError = s.fps < FPS_ERROR || s.jitter_ms > JITTER_ERROR_MS;
    const isWarn = !isError && s.jitter_ms > JITTER_WARN_MS;
    row.classList.toggle("error", isError);
    row.classList.toggle("warn", isWarn);
  }
});
