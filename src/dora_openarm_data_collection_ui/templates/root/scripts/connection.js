const eventSource = new EventSource(`/events?since=${STATE_VERSION}`);
eventSource.onmessage = () => {
  location.reload();
};
// The dataflow dying takes this page's event stream with it. Tell the
// operator on screen instead of leaving a silently frozen page.
const CONNECTION_LOST_AFTER_MS = 4000;
let connectionLostTimer = null;
eventSource.onopen = () => {
  clearTimeout(connectionLostTimer);
  connectionLostTimer = null;
  document.getElementById("connection-lost").style.display = "none";
};
eventSource.onerror = () => {
  if (connectionLostTimer) return;
  connectionLostTimer = setTimeout(() => {
    connectionLostTimer = null;
    if (eventSource.readyState === EventSource.OPEN) return;
    if (window.quitRequestPending) return;
    document.getElementById("connection-lost").style.display = "flex";
  }, CONNECTION_LOST_AFTER_MS);
};
