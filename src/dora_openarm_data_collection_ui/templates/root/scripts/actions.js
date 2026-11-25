const LONG_PRESS_MS = 1000;
// A start->finish toggle is two separate <form> submits (page
// navigations, since these POST and 303-redirect), so we can't just
// debounce the keypress -- we need to remember when recording
// started and refuse to finish too soon after. sessionStorage
// survives the navigation between the two within the same tab.
const MIN_RECORDING_MS = 1000;
const RECORDING_STARTED_KEY = "collectingStartedAt";
let skipTimer = null;
let cancelTimer = null;
let armTimer = null;
let cooldownToastTimer = null;
const focusableElements = [...document.querySelectorAll('input')];
function getRecordingStartedAt() {
  return Number(sessionStorage.getItem(RECORDING_STARTED_KEY) || 0);
}
function setRecordingStartedAt(timestamp) {
  sessionStorage.setItem(RECORDING_STARTED_KEY, String(timestamp));
}
function recordingTooShort() {
  return Date.now() - getRecordingStartedAt() < MIN_RECORDING_MS;
}
function showCooldownToast() {
  const toast = document.getElementById("cooldown-toast");
  if (!toast) return;
  toast.classList.add("show");
  clearTimeout(cooldownToastTimer);
  cooldownToastTimer = setTimeout(() => toast.classList.remove("show"), 800);
}
function submitAction(form) {
  if (form.id === "quit-form") requestQuit(true);
  else form.submit();
}
async function requestQuit(confirmFirst) {
  if (window.quitRequestPending) return;
  if (confirmFirst && !window.confirm(TEXT.confirm_quit)) return;
  window.quitRequestPending = true;
  const form = document.getElementById("quit-form");
  if (form) {
    const button = form.querySelector("input[type=submit]");
    if (button) {
      button.value = TEXT.quitting;
      button.disabled = true;
    }
  }
  document.getElementById("ending-panel").style.display = "flex";
  try {
    await fetch("/quit", {method: "POST"});
  } catch (e) {
    // The UI server shuts down after it sends the Dora quit command.
  }
  // This node ends with the dataflow: once it stops answering, the data
  // collection is over.
  const deadline = Date.now() + 60000;
  while (Date.now() < deadline) {
    try {
      await fetch("/", {cache: "no-store"});
    } catch (e) {
      document.getElementById("ending-title").textContent = TEXT.quit_done_title;
      document.getElementById("ending-hint").textContent = TEXT.quit_done;
      return;
    }
    await new Promise(resolve => setTimeout(resolve, 500));
  }
}
const quitForm = document.getElementById("quit-form");
if (quitForm) {
  quitForm.addEventListener("submit", event => {
    event.preventDefault();
    requestQuit(true);
  });
}
document.addEventListener("keydown", function(event) {
  if (event.repeat || window.quitRequestPending) return;
  if (event.key.toLowerCase() === "c") {
    if (!skipTimer) {
      skipTimer = setTimeout(() => {
        skipTimer = null;
        const skipForm = document.getElementById("skip-form");
        if (skipForm) submitAction(skipForm);
      }, LONG_PRESS_MS);
    }
  }
  if (event.key.toLowerCase() === "b") {
    // `b` is quit (short press) / arm start-pause toggle (long press).
    // The toggle works on both the start and collecting panels.
    if (!armTimer) {
      armTimer = setTimeout(() => {
        armTimer = null;
        const armForm =
          document.getElementById("arm-start-form") ||
          document.getElementById("arm-stop-form");
        if (armForm) submitAction(armForm);
      }, LONG_PRESS_MS);
    }
  }
  if (event.key.toLowerCase() === "a") {
    // While collecting, `a` is fail (short) / cancel (long press).
    if (!document.getElementById("start-form") && !cancelTimer) {
      cancelTimer = setTimeout(() => {
        cancelTimer = null;
        const cancelForm = document.getElementById("cancel-form");
        if (cancelForm) submitAction(cancelForm);
      }, LONG_PRESS_MS);
    }
  }
  if (event.key.toLowerCase() === "h") {
    const current = focusableElements.indexOf(document.activeElement);
    const next = (current + 1) % focusableElements.length;
    focusableElements[next].focus();
  }
  if (event.key.toLowerCase() === "y") {
    const current = focusableElements.indexOf(document.activeElement);
    const prev = (current - 1 + focusableElements.length) % focusableElements.length;
    focusableElements[prev].focus();
  }
  if (event.key.toLowerCase() === "n") {
    const active = document.activeElement;
    const form = active.closest('form');
    if (form) {
      submitAction(form);
    }
  }
});
document.addEventListener("keyup", (event) => {
  if (window.quitRequestPending) return;
  if (event.key.toLowerCase() === "c") {
    if (skipTimer) {
      clearTimeout(skipTimer);
      skipTimer = null;
      const startForm = document.getElementById("start-form");
      const finishForm = document.getElementById("finish-form");
      if (finishForm) {
        if (recordingTooShort()) {
          showCooldownToast();
          return;
        }
        submitAction(finishForm);
      }
      if (startForm) {
        setRecordingStartedAt(Date.now());
        submitAction(startForm);
      }
    }
  }
  if (event.key.toLowerCase() === "b") {
    // Short press (timer still pending): quit. A long press already
    // fired the arm toggle from the timer, so do nothing here.
    if (armTimer) {
      clearTimeout(armTimer);
      armTimer = null;
      const quitForm = document.getElementById("quit-form");
      if (quitForm) requestQuit(false);
    }
  }
  if (event.key.toLowerCase() === "a") {
    if (cancelTimer) {
      clearTimeout(cancelTimer);
      cancelTimer = null;
      const failForm = document.getElementById("fail-form");
      if (failForm) {
        if (recordingTooShort()) {
          showCooldownToast();
          return;
        }
        submitAction(failForm);
      }
    }
  }
});
