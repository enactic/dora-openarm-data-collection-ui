const TEMP_OK_MAX = 40;    // up to here green
const TEMP_WARM_MAX = 70;  // up to here yellow, above it red
const tempClass = (c) =>
  c <= TEMP_OK_MAX ? "ok" : c <= TEMP_WARM_MAX ? "warn" : "error";
// DISABLED is torque off, not a fault; only the motor's own error
// codes and a motor that stopped answering are worth a colour.
const NORMAL_MOTOR_STATUS = ["ENABLED", "DISABLED"];
const ARM_NAME = { left: TEXT.arm_left, right: TEXT.arm_right };
// Joints are named from the data rather than baked into the page. The
// last axis is always the gripper -- OpenArm's motor count is fixed
// whether it is driven in POS_FORCE or MIT mode, so this does not
// depend on which.
const jointLabel = (i, count) => (i === count - 1 ? "GRIP" : "J" + (i + 1));
const setBusBadge = (line, bus) => {
  const badge = line.querySelector(".bus-badge");
  badge.classList.remove("ok", "warn", "error");
  if (!bus) return;
  badge.textContent =
    "BUS " + bus.state + (bus.count > 0 ? " \u00d7" + bus.count : "");
  if (bus.severity !== "unknown") badge.classList.add(bus.severity);
};
// The follower reports its state at the leader's rate (250 Hz).
const ARM_FPS_ERROR = 100;
const ARM_JITTER_WARN_MS = 10;
const ARM_JITTER_ERROR_MS = 30;
const clearArmAlerts = (side) => {
  for (const kind of ["rate", "bus", "temp", "faults"]) setAlert(`arm:${side}:${kind}`, null);
};
const setArmRate = (line, side, s) => {
  const el = line.querySelector(".arm-rate");
  el.classList.remove("warn", "error", "stale");
  if (!s.fps) {
    el.textContent = "-- Hz";
    el.classList.add("stale");
  } else {
    el.textContent = `${s.fps.toFixed(0)} Hz Δ ${s.jitter_ms.toFixed(1)} ms`;
    const isError = s.fps < ARM_FPS_ERROR || s.jitter_ms > ARM_JITTER_ERROR_MS;
    if (isError) el.classList.add("error");
    else if (s.jitter_ms > ARM_JITTER_WARN_MS) el.classList.add("warn");
  }
  rateAlert(`arm:${side}:rate`, `${ARM_NAME[side]} ${TEXT.arm_rate}`, s,
    ARM_FPS_ERROR, ARM_JITTER_WARN_MS, ARM_JITTER_ERROR_MS);
};
const armHealthSource = new EventSource("/arm-health");
armHealthSource.onmessage = (e) => {
  const { lifter, ...health } = JSON.parse(e.data);
  showLifter(lifter);
  const anyPresent = Object.values(health).some((s) => s.present);
  document.getElementById("arm-health-panel").hidden = !anyPresent;
  for (const [side, s] of Object.entries(health)) {
    const line = document.querySelector(`[data-arm="${side}"]`);
    if (!line) continue;
    line.hidden = !s.present;
    if (!s.present) {
      clearArmAlerts(side);
      continue;
    }
    const arm = ARM_NAME[side];
    setArmRate(line, side, s);
    setBusBadge(line, s.bus);
    const severity = s.bus?.severity;
    setAlert(`arm:${side}:bus`,
      severity === "error" || severity === "warn" ? severity : null,
      `${arm} ${TEXT.can_bus}: ${s.bus?.state}`);
    // Hottest answering motor. A silent motor keeps its last reading
    // in the driver, so that would be a stale value.
    const count = Math.max(s.mos?.length ?? 0, s.motor_status?.length ?? 0);
    let hottest = null;
    const faults = [];
    for (let i = 0; i < count; i++) {
      const status = s.motor_status?.[i];
      if (status && !NORMAL_MOTOR_STATUS.includes(status)) {
        faults.push({ joint: jointLabel(i, count), status });
      }
      if (status === "SILENT") continue;
      const temps = [s.mos?.[i], s.rotor?.[i]].filter((t) => t != null);
      if (!temps.length) continue;
      const t = Math.max(...temps);
      if (!hottest || t > hottest.t) hottest = { joint: jointLabel(i, count), t };
    }
    const hot = line.querySelector(".hottest");
    hot.classList.remove("ok", "warn", "error", "stale");
    hot.textContent = hottest ? `${hottest.joint} ${hottest.t}°C` : "--";
    hot.classList.add(hottest ? tempClass(hottest.t) : "stale");
    setAlert(`arm:${side}:temp`,
      hottest && hottest.t > TEMP_WARM_MAX ? "error" : null,
      hottest ? `${arm} ${hottest.joint}: ${hottest.t}°C` : "");
    const rows = line.querySelector(".temp-rows");
    if (rows.children.length !== count) {
      rows.replaceChildren(...Array.from({ length: count }, (_, i) => {
        const row = document.createElement("div");
        row.className = "temp-row mono";
        row.innerHTML = '<span class="joint"></span><span class="mos tone"></span>' +
          '<span class="rotor tone"></span><span class="mstatus"></span>';
        row.querySelector(".joint").textContent = jointLabel(i, count);
        return row;
      }));
    }
    [...rows.children].forEach((row, i) => {
      const status = s.motor_status?.[i];
      // A silent motor keeps its last reading in the driver: a stale value.
      for (const [cls, t] of [["mos", s.mos?.[i]], ["rotor", s.rotor?.[i]]]) {
        const cell = row.querySelector("." + cls);
        const known = t != null && status !== "SILENT";
        cell.classList.remove("ok", "warn", "error", "stale");
        cell.classList.add(known ? tempClass(t) : "stale");
        cell.textContent = known ? t + "°C" : "--";
      }
      const st = row.querySelector(".mstatus");
      st.className = "mstatus" + (status === "SILENT" ? " silent"
        : status === "ENABLED" ? " enabled"
        : status && !NORMAL_MOTOR_STATUS.includes(status) ? " fault" : "");
      st.textContent = status ?? "--";
    });
    setAlert(`arm:${side}:faults`, faults.length ? "error" : null,
      `${arm} ` + faults.map((f) => `${f.joint} ${f.status}`).join(", "));
  }
};
