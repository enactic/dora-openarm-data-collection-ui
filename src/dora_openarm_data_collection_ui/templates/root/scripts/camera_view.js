// Live camera view: one WebSocket carries the picked view's cameras
// (`?view=`) as `name\n<jpeg>` (binary), plus a JSON text message
// saying which cameras are live, the other views' included so their
// buttons can show ⚠. Only that message shows No signal: a camera is
// reported silent when this node stops receiving it, and nothing
// else -- a page reload, an arm starting, this socket reconnecting --
// changes a tile. Closed while the tab is hidden so a forgotten
// window does not keep decoding video next to the recorder.
if (document.getElementById("camera-view")) {
  const cameraView = document.getElementById("camera-view");
  const frameUrls = {};
  const applyCameraStatus = (live) => {
    let shown = 0;
    for (const tile of document.querySelectorAll("[data-cam-view]")) {
      const cam = tile.dataset.camView;
      tile.hidden = !(cam in live);
      if (tile.hidden) continue;
      shown += 1;
      tile.classList.toggle("stale", !live[cam]);
    }
    cameraView.hidden = shown === 0;
    for (const tab of document.querySelectorAll(".camera-tab")) {
      const cams = tab.dataset.cams.split(" ");
      tab.disabled = !cams.some((cam) => cam in live);
      tab.querySelector(".tab-warn").hidden =
        !cams.some((cam) => cam in live && !live[cam]);
    }
  };
  let cameraSocket = null;
  let reconnectTimer = null;
  const closeCameraSocket = () => {
    clearTimeout(reconnectTimer);
    if (cameraSocket) {
      cameraSocket.onclose = null;
      cameraSocket.close();
      cameraSocket = null;
    }
  };
  const openCameraSocket = () => {
    if (cameraSocket || document.hidden) return;
    const scheme = location.protocol === "https:" ? "wss" : "ws";
    const view = encodeURIComponent(cameraView.dataset.view);
    cameraSocket = new WebSocket(
      `${scheme}://${location.host}/cameras?view=${view}`
    );
    cameraSocket.binaryType = "arraybuffer";
    cameraSocket.onmessage = (e) => {
      if (typeof e.data === "string") {
        applyCameraStatus(JSON.parse(e.data));
        return;
      }
      const bytes = new Uint8Array(e.data);
      const split = bytes.indexOf(10);  // "\n"
      if (split < 0) return;
      const cam = new TextDecoder().decode(bytes.subarray(0, split));
      const img = document.querySelector(`[data-cam-view="${cam}"] img`);
      if (!img) return;
      const url = URL.createObjectURL(
        new Blob([bytes.subarray(split + 1)], { type: "image/jpeg" })
      );
      img.src = url;
      if (frameUrls[cam]) URL.revokeObjectURL(frameUrls[cam]);
      frameUrls[cam] = url;
    };
    cameraSocket.onclose = () => {
      cameraSocket = null;
      clearTimeout(reconnectTimer);
      reconnectTimer = setTimeout(openCameraSocket, 1000);
    };
  };
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) {
      closeCameraSocket();
    } else {
      openCameraSocket();
    }
  });
  // Switching views reconnects, so the server only sends the cameras
  // on screen. The choice survives the page's reloads.
  for (const tab of document.querySelectorAll(".camera-tab")) {
    tab.addEventListener("click", () => {
      if (tab.dataset.tab === cameraView.dataset.view) return;
      cameraView.dataset.view = tab.dataset.tab;
      try {
        sessionStorage.setItem("cameraView", tab.dataset.tab);
      } catch (e) {}
      closeCameraSocket();
      openCameraSocket();
    });
  }
  openCameraSocket();
}
