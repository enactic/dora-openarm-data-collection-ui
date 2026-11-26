// One stream for everything: see _events() in main.py.
const eventSource = new EventSource("/events?since={{ state_version }}");
eventSource.addEventListener("state", () => {
  location.reload();
});
