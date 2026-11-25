// Lifter: moves only while a button is held. The press is repeated
// every LIFTER_KEEPALIVE_MS; the node stops the lifter by itself
// once the repeats stop (LIFTER_HOLD_TIMEOUT_S), should the release
// never reach it.
const LIFTER_KEEPALIVE_MS = 200;
let lifterDirection = null;
let lifterTimer = null;
const sendLifter = (direction) =>
  fetch(`/lifter/${direction}`, { method: "POST", keepalive: true }).catch(() => {});
const lifterStart = (button) => {
  if (lifterDirection || button.disabled) return;
  lifterDirection = button.dataset.lifter;
  button.classList.add("active");
  sendLifter(lifterDirection);
  lifterTimer = setInterval(() => sendLifter(lifterDirection), LIFTER_KEEPALIVE_MS);
};
const lifterStop = () => {
  if (!lifterDirection) return;
  lifterDirection = null;
  clearInterval(lifterTimer);
  lifterTimer = null;
  for (const b of document.querySelectorAll(".lifter-btn.active")) b.classList.remove("active");
  sendLifter("stop");
};
for (const button of document.querySelectorAll("[data-lifter]")) {
  button.addEventListener("pointerdown", (e) => {
    if (e.button !== 0) return;
    // Captured, so letting go anywhere stops it, not only on the button.
    button.setPointerCapture(e.pointerId);
    lifterStart(button);
  });
  button.addEventListener("pointerup", lifterStop);
  button.addEventListener("pointercancel", lifterStop);
  button.addEventListener("lostpointercapture", lifterStop);
  button.addEventListener("contextmenu", (e) => e.preventDefault());
}
// Anything that takes the page away from the operator ends the move,
// the reload after every state change included.
window.addEventListener("blur", lifterStop);
window.addEventListener("pagehide", lifterStop);
document.addEventListener("visibilitychange", () => {
  if (document.hidden) lifterStop();
});
const showLifter = (s) => {
  document.getElementById("lifter-panel").hidden = !s?.present;
  if (!s?.present) return;
  // Buttons under a KER leader only; VR moves it with its stick.
  const panel = document.getElementById("lifter-panel");
  panel.querySelector(".lifter-buttons").hidden = !s.controls;
  panel.querySelector(".lifter-hint").hidden = !s.controls;
  if (!s.controls) lifterStop();
  const known = s.elevation_mm != null;
  document.getElementById("lifter-elevation").textContent =
    known ? s.elevation_mm.toFixed(1) : "--";
  document.querySelector(".lifter-height").classList.toggle("stale", !known);
};
