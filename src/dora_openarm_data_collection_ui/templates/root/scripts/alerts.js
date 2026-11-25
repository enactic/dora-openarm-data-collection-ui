// Live warnings from the health streams, listed in the errors
// section (shown while there are any, or any server errors).
const liveAlerts = new Map();  // key -> { level: "error"|"warn", text }
const renderAlerts = () => {
  const items = [...liveAlerts.values()].sort(
    (a, b) => (a.level === "error" ? 0 : 1) - (b.level === "error" ? 0 : 1)
  );
  document.getElementById("live-alerts").replaceChildren(
    ...items.map((a) => {
      const li = document.createElement("li");
      li.className = a.level;
      li.textContent = a.text;
      return li;
    })
  );
  const empty = !hasServerErrors && liveAlerts.size === 0;
  document.getElementById("error-panel").hidden = empty;
  const mark = document.getElementById("error-mark");
  mark.hidden = empty;
  mark.classList.toggle("warn",
    !hasServerErrors && items.every((a) => a.level !== "error"));
};
// level null clears the key.
const setAlert = (key, level, text) => {
  const old = liveAlerts.get(key);
  if (!level) {
    if (!liveAlerts.delete(key)) return;
  } else {
    if (old && old.level === level && old.text === text) return;
    liveAlerts.set(key, { level, text });
  }
  renderAlerts();
};
// A rate row's alert: silent, under the rate, or jittery.
const rateAlert = (key, name, s, fpsMin, jitterWarn, jitterError, silentHint = "") => {
  if (!s || s.fps <= 0) {
    setAlert(key, "error", `${name}: ${TEXT.not_received}${silentHint}`);
  } else if (s.fps < fpsMin || s.jitter_ms > jitterError) {
    setAlert(key, "error",
      `${name}: ${s.fps.toFixed(1)} Hz, Δ ${s.jitter_ms.toFixed(1)} ms`);
  } else if (s.jitter_ms > jitterWarn) {
    setAlert(key, "warn", `${name}: Δ ${s.jitter_ms.toFixed(1)} ms`);
  } else {
    setAlert(key, null);
  }
};
