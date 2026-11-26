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

document.addEventListener("keydown", function(event) {
  if (event.repeat) return;

  if (event.key.toLowerCase() === "c") {
    if (!skipTimer) {
      skipTimer = setTimeout(() => {
        skipTimer = null;
        const skipForm = document.getElementById("skip-form");
        if (skipForm) skipForm.submit();
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
        if (armForm) armForm.submit();
      }, LONG_PRESS_MS);
    }
  }

  if (event.key.toLowerCase() === "a") {
    // While collecting, `a` is fail (short) / cancel (long press).
    if (!document.getElementById("start-form") && !cancelTimer) {
      cancelTimer = setTimeout(() => {
        cancelTimer = null;
        const cancelForm = document.getElementById("cancel-form");
        if (cancelForm) cancelForm.submit();
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
      form.submit();
    }
  }
});

document.addEventListener("keyup", (event) => {
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
        finishForm.submit();
      }

      if (startForm) {
        setRecordingStartedAt(Date.now());
        startForm.submit();
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
      if (quitForm) quitForm.submit();
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
        failForm.submit();
      }
    }
  }
});
