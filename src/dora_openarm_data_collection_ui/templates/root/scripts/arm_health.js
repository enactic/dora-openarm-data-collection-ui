const TEMP_OK_MAX = 40;    // up to here green
const TEMP_WARM_MAX = 70;  // up to here yellow, above it red

const tempClass = (c) =>
  c <= TEMP_OK_MAX ? "ok" : c <= TEMP_WARM_MAX ? "warm" : "hot";

const setTemp = (cell, celsius) => {
  if (!cell) return;
  cell.classList.remove("ok", "warm", "hot", "unknown");
  if (celsius === undefined || celsius === null) {
    cell.textContent = "--";
    cell.classList.add("unknown");
    return;
  }
  cell.textContent = celsius + "°C";
  cell.classList.add(tempClass(celsius));
};

// DISABLED is torque off, not a fault; only the motor's own error
// codes and a motor that stopped answering are worth a colour.
const NORMAL_MOTOR_STATUS = ["ENABLED", "DISABLED"];

const setMotorStatus = (cell, status) => {
  if (!cell) return;
  cell.classList.remove("enabled", "fault", "silent");
  if (!status) {
    cell.textContent = "--";
    return;
  }
  cell.textContent = status;
  if (status === "SILENT") cell.classList.add("silent");
  else if (status === "ENABLED") cell.classList.add("enabled");
  else if (!NORMAL_MOTOR_STATUS.includes(status)) cell.classList.add("fault");
};

const setBusBadge = (panel, bus) => {
  const badge = panel.querySelector(".bus-badge");
  if (!badge || !bus) return;
  badge.classList.remove("ok", "warn", "error");
  badge.textContent =
    "BUS " + bus.state + (bus.count > 0 ? " \u00d7" + bus.count : "");
  if (bus.severity !== "unknown") badge.classList.add(bus.severity);
};

// Joints are named from the data rather than baked into the page. The
// last axis is always the gripper -- OpenArm's motor count is fixed
// whether it is driven in POS_FORCE or MIT mode, so this does not
// depend on which.
const jointLabel = (i, count) => (i === count - 1 ? "GRIP" : "J" + (i + 1));

const syncRows = (panel, count) => {
  const host = panel.querySelector(".temp-rows");
  if (host.children.length === count) return;
  host.replaceChildren();
  for (let i = 0; i < count; i++) {
    const row = document.createElement("div");
    row.className = "temp-row";
    row.dataset.joint = i;
    row.innerHTML =
      '<span class="joint"></span>' +
      '<span class="mos unknown">--</span>' +
      '<span class="rotor unknown">--</span>' +
      '<span class="mstatus">--</span>';
    row.querySelector(".joint").textContent = jointLabel(i, count);
    host.append(row);
  }
};

eventSource.addEventListener("arm-health", (e) => {
  const health = JSON.parse(e.data);
  const anyPresent = Object.values(health).some((s) => s.present);
  document.getElementById("arm-health-panel").hidden = !anyPresent;
  for (const [side, s] of Object.entries(health)) {
    const panel = document.querySelector(`[data-arm="${side}"]`);
    if (!panel) continue;
    panel.hidden = !s.present;
    const count = Math.max(
      s.mos?.length ?? 0,
      s.motor_status?.length ?? 0,
    );
    syncRows(panel, count);
    setBusBadge(panel, s.bus);
    for (const row of panel.querySelectorAll(".temp-row")) {
      const i = Number(row.dataset.joint);
      const status = s.motor_status?.[i];
      // A motor that stopped answering keeps its last reported
      // temperature in the driver, so showing it would present a
      // stale reading as a current one.
      const silent = status === "SILENT";
      setTemp(row.querySelector(".mos"), silent ? null : s.mos?.[i]);
      setTemp(row.querySelector(".rotor"), silent ? null : s.rotor?.[i]);
      setMotorStatus(row.querySelector(".mstatus"), status);
    }
  }
});
