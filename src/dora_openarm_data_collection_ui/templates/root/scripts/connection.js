// One stream for everything: see _events() in main.py.
const eventSource = new EventSource("/events?since={{ state_version }}&lang={{ language }}");
eventSource.addEventListener("state", () => {
  location.reload();
});
