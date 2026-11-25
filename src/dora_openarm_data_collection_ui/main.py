# Copyright 2026 Enactic, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""dora-rs node that provides UI to control data collection with OpenArm."""

import argparse
import asyncio
import collections
from contextlib import asynccontextmanager
import dataclasses
import itertools
import datetime
import dora
import json
from collections.abc import AsyncIterable
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    RedirectResponse,
    Response,
)
from fastapi.sse import EventSourceResponse, ServerSentEvent
from fastapi.templating import Jinja2Templates
import os
import pathlib
import pyarrow as pa
import re
import threading
import time
import urllib.parse
import uvicorn
import yaml

base_dir = os.path.dirname(__file__)
templates = Jinja2Templates(directory=f"{base_dir}/templates")

# The UI's text, one YAML file per language (see --language). English is the
# default and the fallback for any key another language does not have.
LOCALES_DIR = pathlib.Path(base_dir) / "locales"
DEFAULT_LANGUAGE = "en"
language = DEFAULT_LANGUAGE
messages: dict[str, str] = {}


def available_languages():
    """Return the languages that have a file under locales/."""
    return sorted(path.stem for path in LOCALES_DIR.glob("*.yaml"))


def load_messages(lang):
    """Use the given language's text, English where it has none."""
    global language, messages
    loaded = load_yaml(LOCALES_DIR / f"{DEFAULT_LANGUAGE}.yaml")
    if lang != DEFAULT_LANGUAGE:
        loaded.update(load_yaml(LOCALES_DIR / f"{lang}.yaml") or {})
    language = lang
    messages = {key: str(value) for key, value in loaded.items()}


def t(key, /, **values):
    """Return the text for key in the UI's language, {name}s filled in."""
    text = messages.get(key, key)
    return text.format(**values) if values else text


templates.env.globals["t"] = t

node = None

auto_open = False
port = None

# Operation manual shown on the task screen (see --manual-file).
manual = None

# Where the reference videos of the tasks are (see --reference-video-dir):
# <dir>/<task set>/<task number from 1>/<any name>.mp4, or the file that
# tasks[].reference_video names.
DEFAULT_REFERENCE_VIDEO_DIR = pathlib.Path("task-reference")
reference_video_dir = DEFAULT_REFERENCE_VIDEO_DIR
REFERENCE_VIDEO_SUFFIXES = (".mp4", ".m4v", ".webm", ".mov")
# The id of the task set being recorded: the metadata's task_set, or the
# metadata file's name.
task_set = ""


# Browser openers, most specific first. "open" is macOS, "xdg-open" is Linux.
_BROWSER_COMMANDS = ("xdg-open", "gio", "open")


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """Open a Web browser automatically if requested."""
    if auto_open:
        url = f"http://127.0.0.1:{port}"
        for command in _BROWSER_COMMANDS:
            argv = [command, "open", url] if command == "gio" else [command, url]
            try:
                await asyncio.create_subprocess_exec(*argv)
                break
            except (FileNotFoundError, OSError):
                continue
    yield


app = FastAPI(lifespan=_lifespan)


@dataclasses.dataclass
class State:
    """The current state."""

    collecting: bool = False
    running: bool = True
    ending: bool = False
    quit_requested: bool = False
    episode_number: int = 0
    task_index: int = 0
    task_title: str = ""
    # The dataset language instruction (tasks[].prompt).
    task_prompt: str = ""
    task_description: str = ""
    # task_description split at its headings (see split_description), one
    # row each in the left panel; empty when it has no headings.
    task_sections: list = dataclasses.field(default_factory=list)
    # Videos of the task done right (tasks[].reference_video, or every video
    # in the task's numbered folder): URLs, or paths that /reference-video
    # serves. Several are tiled. Display only.
    task_reference_videos: list = dataclasses.field(default_factory=list)
    # The reference video named by the task but not found: shown as a quiet
    # note, since recording works without it (the folder may not be there).
    task_reference_missing: str = ""
    arm_status_right: str = "stopped"
    arm_status_left: str = "stopped"
    # Failures to show in red on the task screen: newest last.
    # [{"time", "source", "message", "count"}]
    errors: list = dataclasses.field(default_factory=list)


state = State()

# Keep the newest failures only; the node's log has every one.
MAX_ERRORS = 20


def _record_error(source, message):
    """Remember a failure for the task screen. Returns True when it is new."""
    message = " ".join(str(message).split())
    if not message:
        return False
    now = datetime.datetime.now().strftime("%H:%M:%S")
    for error in state.errors:
        if error["message"] == message:
            # A repeating failure updates its counter instead of piling up.
            error["count"] += 1
            error["time"] = now
            return False
    state.errors.append(
        {"time": now, "source": str(source), "message": message, "count": 1}
    )
    del state.errors[:-MAX_ERRORS]
    return True


_state_changed = asyncio.Condition()

# Monotonically incremented on every state change.
state_version = 0


CAMERA_INPUTS = (
    "camera_wrist_right",
    "camera_wrist_left",
    "camera_head_left",
    "camera_head_right",
    "camera_ceiling",
)

CAMERA_TIMESTAMP_WINDOW = 60
CAMERA_STALE_AFTER_S = 1.0

# Cameras shown live to the operator, by view: one head camera, the head
# with both wrists, or all five. Only the picked view's cameras are sent,
# since every camera shown costs browser CPU on the same PC that records.
CAMERA_VIEWS = {
    "head": ("camera_head_left",),
    "wrists": ("camera_head_left", "camera_wrist_left", "camera_wrist_right"),
    "all": (
        "camera_head_left",
        "camera_head_right",
        "camera_ceiling",
        "camera_wrist_left",
        "camera_wrist_right",
    ),
}
CAMERA_VIEW_DEFAULT = "head"
CAMERA_VIEW_INPUTS = CAMERA_VIEWS["all"]

# The live view is a preview, so it is sent at a fraction of the camera rate
# (see --camera-view-fps). 0 turns it off.
camera_view_fps = 10.0

# The newest JPEG of each viewed camera, kept as the Arrow array it arrived
# in. It is only turned into bytes when a frame is actually sent, so storing
# every 30 fps frame costs nothing.
latest_frames: dict[str, pa.Array] = {}
# Bumped on every stored frame, so a sender can tell a new frame from the
# one it already sent.
frame_seq: dict[str, int] = {name: 0 for name in CAMERA_VIEW_INPUTS}
# When this node last got a frame from each viewed camera (time.monotonic()).
# This is the time of receipt, not the frame's dora timestamp, so the event
# loop falling behind (both arms report at 250 Hz once started) cannot make a
# camera that is still delivering look disconnected.
frame_received_at: dict[str, float] = {}
# A tile shows No signal only once its camera has sent nothing for this long,
# which is far longer than any gap a working camera at 30 fps leaves.
CAMERA_VIEW_SILENT_AFTER_S = 2.0
# How often /cameras repeats which cameras are live, so a tile turns to
# No signal even though no frame will come to tell it.
CAMERA_VIEW_STATUS_INTERVAL_S = 0.5

# dora-openarm status inputs, one per arm. The input id matches the State field
ARM_STATUS_INPUTS = ("arm_status_right", "arm_status_left")

# dora-openarm state inputs, one per arm, carrying per-motor MOSFET and rotor
# temperatures. They arrive at the leader's rate (250 Hz), so the handler only
# stores the latest sample; the browser is fed from /arm-health instead.
ARM_STATE_INPUTS = {"arm_state_right": "right", "arm_state_left": "left"}

# A reading older than this is shown as unknown rather than as the last value,
# so a follower that stopped answering does not look merely cool.
ARM_TEMPERATURE_STALE_AFTER_S = 3.0

# Shortest gap between two parses of an arm's state. The arms report at the
# leader's rate and the browser is fed every 500 ms, so converting every
# message out of Arrow would be ~50x more work than anyone can see.
ARM_STATE_PARSE_INTERVAL_S = 0.2

# The leader's own stream. `ker_metadata` names the device and is sent once
# at startup, so it is remembered rather than polled; `ker_position` is only
# read for its arrival time, the same way the camera rows are.
KER_METADATA_INPUT = "ker_metadata"
KER_POSITION_INPUT = "ker_position"
KER_TIMESTAMP_WINDOW = 120
KER_STALE_AFTER_S = 1.0

# Which inputs have ever delivered something. A dataflow wires up only the
# nodes it uses, and a node is never told what it was wired to, so a panel
# earns its place on screen by having produced data at least once.
seen_inputs: set[str] = set()

# VR packet arrival times (ns) published by udp-receiver as
# `vr_receive_times` or `vr_recv_ts` (deprecated).
VR_RECEIVE_TIMES_INPUTS = ("vr_receive_times", "vr_recv_ts")

VR_TIMESTAMP_WINDOW = 120  # ~1.6 s of history at 72 Hz
VR_STALE_AFTER_S = 1.0

# The Cell lifter (dora-openarm-cell-lifter). Its `elevation_observation` is
# the height in mm, sent on every tick once it has calibrated. The panel
# shows from the start when the metadata's equipment has a lifter (the
# Cell), else once a height has arrived. `lifter_command` drives it like its joystick:
# "lifter-up" / "lifter-down" move it until "lifter-stop".
#
# The buttons are for a KER leader only. A VR or WebXR leader moves the
# lifter with its left stick, which it sends with every packet, so a resting
# stick would cancel a move from the page at once.
LIFTER_OBSERVATION_INPUT = "elevation_observation"
LIFTER_COMMANDS = {"up": "lifter-up", "down": "lifter-down", "stop": "lifter-stop"}
LIFTER_STALE_AFTER_S = 1.0
# The page repeats up/down every 200 ms while a button is held. Should those
# stop arriving (the browser closed or lost the connection mid-press), the
# lifter is stopped here instead of running on to the end of its stroke.
LIFTER_HOLD_TIMEOUT_S = 0.6
LIFTER_WATCHDOG_INTERVAL_S = 0.05


@dataclasses.dataclass
class Lifter:
    """The lifter's last height and the move the task screen holds, if any."""

    elevation_mm: float | None = None
    updated_at: float = 0.0
    # "up" or "down" while a button is held, else None.
    moving: str | None = None
    hold_until: float = 0.0


lifter = Lifter()
# From the metadata (equipment.embodiments and leader): whether the robot
# has a lifter, and the leader.
lifter_equipped = False
leader = None
# Routes run in worker threads while the watchdog runs on the event loop.
_lifter_lock = threading.Lock()


def _lifter_present():
    """Whether the task screen shows the lifter panel."""
    return lifter_equipped or LIFTER_OBSERVATION_INPUT in seen_inputs


def _lifter_controls_enabled():
    """Whether the page may move the lifter: only under a KER leader."""
    return leader == "ker" or KER_POSITION_INPUT in seen_inputs


@dataclasses.dataclass
class StreamStats:
    """Rolling rate / jitter stats for one input stream (camera, VR, leader)."""

    fps: float = 0.0
    jitter_ms: float = 0.0


camera_stats: dict[str, StreamStats] = {name: StreamStats() for name in CAMERA_INPUTS}
camera_timestamps: dict[str, collections.deque] = {
    name: collections.deque(maxlen=CAMERA_TIMESTAMP_WINDOW) for name in CAMERA_INPUTS
}

vr_stats = StreamStats()
vr_timestamps: collections.deque = collections.deque(maxlen=VR_TIMESTAMP_WINDOW)

# How late each frame arrives: receipt time minus its dora timestamp, in ms.
# That is the transport and queue delay, not the camera's own capture delay.
camera_latency_ms: dict[str, collections.deque] = {
    name: collections.deque(maxlen=CAMERA_TIMESTAMP_WINDOW) for name in CAMERA_INPUTS
}

# Arrival times of each arm's state. Its rate and jitter stand in for the CAN
# loop's timing, which nothing measures directly. Computed per snapshot, not
# per message, since the state arrives at 250 Hz.
ARM_STATE_TIMESTAMP_WINDOW = 250
ARM_STATE_STALE_AFTER_S = 1.0
arm_state_timestamps: dict[str, collections.deque] = {
    side: collections.deque(maxlen=ARM_STATE_TIMESTAMP_WINDOW)
    for side in ARM_STATE_INPUTS.values()
}

ker_stats = StreamStats()
ker_timestamps: collections.deque = collections.deque(maxlen=KER_TIMESTAMP_WINDOW)
ker_device: dict = {}


@dataclasses.dataclass
class ArmHealth:
    """Latest per-motor and bus readings for one arm.

    Temperatures are in Celsius: `mos` is the driver MOSFET, `rotor` the
    motor itself. Both are kept because either can be the one that
    overheats -- the MOSFET leads under a sustained current, the rotor lags
    but stays hot for longer.

    `motor_status` is what each motor says about itself, or "SILENT" for one
    that stopped answering. A motor that is merely unplugged raises no bus
    error at all, so this is the only place it shows.

    `bus` holds openarm_can's per-interface counters. They latch, so what an
    operator wants -- what happened during this episode -- is the difference
    against `bus_baseline`, which is reset when an episode starts.
    """

    mos: list[int] = dataclasses.field(default_factory=list)
    rotor: list[int] = dataclasses.field(default_factory=list)
    motor_status: list[str] = dataclasses.field(default_factory=list)
    bus: dict = dataclasses.field(default_factory=dict)
    bus_baseline: dict = dataclasses.field(default_factory=dict)
    updated_at: float = 0.0


arm_health: dict[str, ArmHealth] = {
    side: ArmHealth() for side in ARM_STATE_INPUTS.values()
}

# Bus counters worth surfacing, worst last: the badge shows the heaviest one
# that moved, so the order here is the severity order.
BUS_COUNTER_SEVERITY = (
    ("error_warning", "WARNING"),
    ("tx_overflow", "OVERFLOW"),
    ("rx_overflow", "OVERFLOW"),
    ("error_passive", "ERROR-PASSIVE"),
    ("ack_error", "NO ACK"),
    ("bus_off", "BUS-OFF"),
)


def _event_field(event, name, default=None):
    """Read a field from a dora event, which only supports __getitem__."""
    try:
        value = event[name]
    except (KeyError, TypeError, AttributeError):
        return default
    return default if value is None else value


def _event_ts_to_seconds(ts) -> float:
    """Normalize a dora event timestamp (datetime or ns int) to POSIX seconds."""
    if isinstance(ts, datetime.datetime):
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=datetime.timezone.utc)
        return ts.timestamp()
    if isinstance(ts, (int, float)):
        return float(ts) / 1e9
    return time.time()


def _fold_arrival(series, stats, ts_s: float, stale_after_s: float) -> None:
    """Fold one arrival time into a rolling rate / jitter estimate.

    A gap longer than `stale_after_s` restarts the window, so the rate
    reported after a stall describes the stream now rather than averaging
    across the outage.
    """
    if series and ts_s - series[-1] > stale_after_s:
        series.clear()
    series.append(ts_s)
    if len(series) < 2:
        return
    span = series[-1] - series[0]
    if span <= 0:
        return
    diffs = [series[i] - series[i - 1] for i in range(1, len(series))]
    stats.fps = (len(series) - 1) / span
    stats.jitter_ms = (max(diffs) - min(diffs)) * 1e3


def _series_rate(series, now: float, stale_after_s: float) -> tuple[float, float]:
    """Return (Hz, jitter in ms) of arrival times, or (0, 0) when stale."""
    if len(series) < 2 or now - series[-1] > stale_after_s:
        return 0.0, 0.0
    span = series[-1] - series[0]
    if span <= 0:
        return 0.0, 0.0
    shortest = longest = series[1] - series[0]
    for previous, current in zip(
        series, itertools.islice(series, 1, None), strict=False
    ):
        gap = current - previous
        shortest = min(shortest, gap)
        longest = max(longest, gap)
    return (len(series) - 1) / span, (longest - shortest) * 1e3


def _add_arrival(series, ts_s: float, stale_after_s: float) -> None:
    """Append an arrival time, restarting the window after a stall."""
    if series and ts_s - series[-1] > stale_after_s:
        series.clear()
    series.append(ts_s)


def _update_camera_stats(event_id: str, ts_s: float) -> None:
    _fold_arrival(
        camera_timestamps[event_id],
        camera_stats[event_id],
        ts_s,
        CAMERA_STALE_AFTER_S,
    )


def _store_camera_frame(event_id: str, value, metadata) -> None:
    """Keep the newest frame of a viewed camera for /cameras."""
    if event_id not in CAMERA_VIEW_INPUTS or camera_view_fps <= 0:
        return
    # Every robot module sends JPEG today; anything else would need decoding,
    # which this node deliberately does not do.
    if metadata.get("encoding", "jpeg") not in ("jpeg", "jpg", "jpe"):
        return
    latest_frames[event_id] = value
    frame_seq[event_id] += 1
    frame_received_at[event_id] = time.monotonic()


def _camera_view_status() -> dict[str, bool]:
    """Whether each viewed camera is delivering, keyed by input id.

    A camera that never sent a frame is left out: this dataflow did not wire
    it, so its tile stays hidden rather than reading No signal forever.
    """
    now = time.monotonic()
    return {
        name: now - received_at <= CAMERA_VIEW_SILENT_AFTER_S
        for name, received_at in frame_received_at.items()
    }


def _update_arm_health(side: str, value) -> None:
    """Record one dora-openarm `state` message.

    The payload is a length-1 StructArray. `tmos`, `trotor` and
    `motor_status` are one entry per motor, in the order the driver reports
    them (joints, then gripper); `bus` covers the interface as a whole.
    """
    health = arm_health[side]
    now = time.time()
    if now - health.updated_at < ARM_STATE_PARSE_INTERVAL_S:
        return
    names = value.type.names if pa.types.is_struct(value.type) else ()
    try:
        if "tmos" in names:
            health.mos = [int(v) for v in value.field("tmos")[0].as_py()]
            health.rotor = [int(v) for v in value.field("trotor")[0].as_py()]
        if "motor_status" in names:
            health.motor_status = list(value.field("motor_status")[0].as_py())
        if "bus" in names:
            health.bus = dict(value.field("bus")[0].as_py() or {})
            if not health.bus_baseline:
                health.bus_baseline = dict(health.bus)
        health.updated_at = now
    except Exception:
        # An older (or differently shaped) dora-openarm state must never take
        # the whole node down: _main_dora's event loop has no guard around
        # its dispatch, so an exception here kills every panel, not just this
        # one. Broad on purpose -- some pyarrow errors (ArrowIndexError,
        # ArrowNotImplementedError) fall outside AttributeError/KeyError/
        # TypeError/ValueError, so a narrower catch would still let a
        # malformed message escape.
        pass


def _reset_bus_baselines() -> None:
    """Make the bus badges read "during this episode" from here on.

    The counters latch by design, so that a bus-off the driver recovers from
    within milliseconds is not missed. Without a baseline the badge would go
    red once and stay red for the rest of the session.
    """
    for health in arm_health.values():
        health.bus_baseline = dict(health.bus)


def _bus_state(health: ArmHealth) -> dict:
    """Worst bus condition for one arm, and how often it happened.

    `carrier` is the only instantaneous reading. Everything else is counted,
    so a fault that has since recovered reports as RECOVERED rather than as
    though it were still happening.
    """
    if not health.bus:
        return {"state": "UNKNOWN", "count": 0, "severity": "unknown"}
    if not health.bus.get("carrier", True):
        # IFF_UP is clear only when the interface was taken down; a bus-off
        # with no auto-restart leaves it up but without carrier.
        down = health.bus.get("net_down", 0) - health.bus_baseline.get("net_down", 0)
        return {
            "state": "IF DOWN" if down > 0 else "NO CARRIER",
            "count": 0,
            "severity": "error",
        }

    worst, count = None, 0
    for name, label in BUS_COUNTER_SEVERITY:
        delta = health.bus.get(name, 0) - health.bus_baseline.get(name, 0)
        if delta > 0:
            worst, count = label, delta
    if worst is None:
        return {"state": "OK", "count": 0, "severity": "ok"}
    # It happened, but the link is carrying traffic again.
    return {"state": f"RECOVERED ({worst})", "count": count, "severity": "warn"}


def _update_ker_device(value) -> None:
    """Remember what the leader said about itself.

    Sent once when the leader starts, so a UI that connected later simply
    never sees it. That only costs the device name; the panel itself is
    earned by the position stream.
    """
    try:
        payload = value[0].as_py()
        ker_device.update(json.loads(payload) if isinstance(payload, str) else payload)
    except (AttributeError, IndexError, TypeError, ValueError, json.JSONDecodeError):
        pass


def _update_ker_stats(ts_s: float) -> None:
    """Fold one leader sample arrival time into the rolling stats."""
    _fold_arrival(ker_timestamps, ker_stats, ts_s, KER_STALE_AFTER_S)


def _update_vr_stats(ts_s: float) -> None:
    """Fold one real VR packet arrival time (POSIX seconds) into the rolling stats."""
    series = vr_timestamps
    if series and ts_s - series[-1] > VR_STALE_AFTER_S:
        series.clear()
    series.append(ts_s)
    if len(series) < 2:
        return
    span = series[-1] - series[0]
    if span <= 0:
        return
    vr_stats.fps = (len(series) - 1) / span
    diffs = [series[i] - series[i - 1] for i in range(1, len(series))]
    vr_stats.jitter_ms = (max(diffs) - min(diffs)) * 1e3


async def _notify_state_changed() -> None:
    global state_version
    async with _state_changed:
        state_version += 1
        _state_changed.notify_all()


def apply_task(index):
    """Show the task at the given index, in the UI's language when it has one."""
    task = tasks[index]
    state.task_index = index
    # prompt_<language> / description_<language> are display only: prompt
    # stays as it is so the recorded dataset keeps its language instruction.
    state.task_title = task.get(f"prompt_{language}") or task.get("prompt", "")
    state.task_prompt = task.get("prompt", "")
    state.task_description = task.get(f"description_{language}") or task.get(
        "description", ""
    )
    state.task_sections = split_description(state.task_description)
    video = str(task.get("reference_video") or "").strip()
    found = reference_video_files(index)
    if _is_url(video):
        state.task_reference_videos = [video]
    else:
        # The URL differs per file, so the browser never plays a cached video
        # of another task set (or an older file) at the same index.
        state.task_reference_videos = [
            f"/reference-video/{index}/{number}?v="
            + urllib.parse.quote(f"{task_set}/{path.name}/{_mtime(path)}")
            for number, path in enumerate(found)
        ]
    state.task_reference_missing = (
        video if video and not _is_url(video) and not found else ""
    )


# Folded YAML joins Japanese lines with a space: drop spaces between
# non-ASCII characters.
_CJK_SPACE = re.compile(r"(?<=[^\x00-\x7f]) +(?=[^\x00-\x7f])")
# Where a description is split, whatever the UI's language: 【注意】 style
# headings, and the older 「初期状態: … 完了状態: … 次の周を始める前に、…」
# and English "Initial state: … Done when … Before repeating …" markers.
# "Before repeating" is a lookahead, so it stays in the section's text.
_DESCRIPTION_HEADING = re.compile(
    r"【(?P<bracket>[^】]+)】"
    r"|(?P<marker>初期状態|完了状態|Initial state)\s*[:：]"
    r"|(?P<reset_ja>次の周を始める前に)[、,]?"
    r"|\b(?P<done_en>Done when)\s+"
    r"|(?=\b(?P<reset_en>Before repeating)\b)"
)
# heading -> (key of the label shown, kind). The kind colours the section:
# "warn" must not be missed, "setup" is how the scene starts, "done" is when
# to stop, "reset" is for between rounds. Unlisted 【…】 headings show as
# they are.
_DESCRIPTION_KINDS = {
    "注意": ("heading_caution", "warn"),
    "スキップ": ("heading_skip", "warn"),
    "準備": ("heading_prepare", "setup"),
    "初期状態": ("heading_initial", "setup"),
    "Initial state": ("heading_initial", "setup"),
    "完了": ("heading_done", "done"),
    "完了状態": ("heading_done_when", "done"),
    "Done when": ("heading_done_when", "done"),
    "次の周を始める前に": ("heading_reset", "reset"),
    "Before repeating": ("heading_reset", "reset"),
}


def split_description(text):
    """Split a task description at its headings, one section per heading.

    Returns [{"heading", "text", "kind"}], or [] when the description has
    no headings, so the caller shows it as one paragraph. Text before the
    first heading is a section without a heading, except before a "Done
    when" / 完了状態 marker, where it is the initial state.
    """
    text = _CJK_SPACE.sub("", " ".join(str(text or "").split()))
    sections = []
    heading, start = None, 0
    for match in _DESCRIPTION_HEADING.finditer(text):
        body = text[start : match.start()].strip()
        if heading or body:
            sections.append((heading, body))
        heading = next(v for v in match.groupdict().values() if v)
        start = match.end()
    if not sections and heading is None:
        return []
    body = text[start:].strip()
    if heading or body:
        sections.append((heading, body))
    # In the marker style, the text before "Done when" / 完了状態 is the
    # initial state. Anything else before the first heading stays without
    # one: "Pick a spoon. Before repeating, …" is not an initial state.
    headings = {heading for heading, _ in sections}
    if sections[0][0] is None and headings & {"Done when", "完了状態"}:
        sections[0] = ("Initial state", sections[0][1])
    result = []
    for heading, body in sections:
        if heading in _DESCRIPTION_KINDS:
            key, kind = _DESCRIPTION_KINDS[heading]
            label = t(key)
        else:
            label, kind = heading or "", ""
        if heading == "Done when" and body:
            body = body[0].upper() + body[1:]
        result.append({"heading": label, "text": body, "kind": kind})
    return result


def _mtime(path):
    try:
        return path.stat().st_mtime_ns
    except OSError:
        return 0


def _is_url(value):
    return value.startswith(("http://", "https://"))


def reference_video_files(index):
    """Return the files of the reference videos of the task at `index`.

    A task's `reference_video` is a URL, which the browser plays as is, or a
    file name in reference_video_dir. Without it, every video file in
    <reference_video_dir>/<task set>/<index + 1>/ is the task's, in name
    order, so a folder of numbered folders gives every task of a set its
    videos. /reference-video serves the files.
    """
    video = str(tasks[index].get("reference_video") or "").strip()
    if _is_url(video):
        return []
    if video:
        path = reference_video_dir / video
        return [path] if path.is_file() else []
    folder = reference_video_dir / task_set / str(index + 1)
    try:
        files = sorted(folder.iterdir())
    except OSError:
        return []
    return [
        path
        for path in files
        if path.suffix.lower() in REFERENCE_VIDEO_SUFFIXES
        and not path.name.startswith(".")
        and path.is_file()
    ]


def next_task():
    """Update the state with the next task."""
    index = state.task_index + 1
    if index >= len(tasks):
        index = 0
    apply_task(index)


def _command_start():
    """Start a new episode."""
    # The height is set between episodes, never while recording.
    _stop_lifter_if_moving()
    node.send_output(
        "command",
        pa.array(["start"]),
        {
            "episode_number": state.episode_number,
            "task_index": state.task_index,
        },
    )
    # From here the bus badges report this episode, not the whole session.
    _reset_bus_baselines()
    state.collecting = True


def _command_success():
    """Finish the current episode successfully."""
    node.send_output("command", pa.array(["success"]))
    state.collecting = False
    state.episode_number += 1
    next_task()


def _command_fail():
    """Finish the current episode unsuccessfully."""
    node.send_output("command", pa.array(["fail"]))
    state.collecting = False
    state.episode_number += 1
    next_task()


async def _command_quit():
    """Quit this data collection: Dora closes the dataflow."""
    if state.quit_requested:
        return
    state.quit_requested = True
    state.ending = True
    node.send_output("command", pa.array(["quit"]))
    state.running = False
    await _notify_state_changed()


def _command_arm_start():
    """Start (power on) the arm(s)."""
    node.send_output("arm_command", pa.array(["start"]))


def _command_arm_stop():
    """Pause (stop) the arm(s)."""
    node.send_output("arm_command", pa.array(["stop"]))


def _command_lifter(direction):
    """Move the lifter "up" or "down" until "stop" or the hold times out.

    A repeated up/down only extends the hold; the lifter is told again
    only when the direction changes.
    """
    if direction == "stop":
        # Only a move this page started: a stop otherwise would zero a VR
        # leader's stick for a moment.
        _stop_lifter_if_moving()
        return
    with _lifter_lock:
        lifter.hold_until = time.monotonic() + LIFTER_HOLD_TIMEOUT_S
        if lifter.moving == direction:
            return
        # Moving only once the command is out, so that a failed send leaves
        # nothing for the watchdog or the next start to stop.
        if _send_lifter_command(direction):
            lifter.moving = direction


def _stop_lifter_if_moving(expired_only=False):
    """Stop a move the task screen started, if one is going.

    With `expired_only`, only once its hold has run out: checked under the
    lock, so a keep-alive that just arrived is never cut short.
    """
    with _lifter_lock:
        if lifter.moving is None:
            return
        if expired_only and time.monotonic() <= lifter.hold_until:
            return
        lifter.moving = None
        _send_lifter_command("stop")


def _send_lifter_command(direction):
    """Send a lifter command; return whether it went out.

    A failure is shown on the task screen instead of raised: this runs from
    the watchdog and before every start, which must keep working when the
    dataflow does not wire `lifter_command`.
    """
    try:
        node.send_output("lifter_command", pa.array([LIFTER_COMMANDS[direction]]))
    except Exception as error:  # dora raises plain exceptions here.
        _record_error("ui", t("lifter_command_failed", error=error))
        return False
    return True


def _update_lifter_elevation(value) -> None:
    """Record the lifter's height (mm) from `elevation_observation`."""
    try:
        lifter.elevation_mm = float(value[0].as_py())
        lifter.updated_at = time.time()
    except (IndexError, TypeError, ValueError):
        pass


async def _lifter_watchdog():
    """Stop the lifter when the page holding a button stops saying so."""
    while state.running:
        if lifter.moving is not None:
            _stop_lifter_if_moving(expired_only=True)
        await asyncio.sleep(LIFTER_WATCHDOG_INTERVAL_S)


@app.get("/", response_class=HTMLResponse)
def _root(request: Request):
    """Render the main HTML."""
    return templates.TemplateResponse(
        request=request,
        name="root.html",
        context={
            "language": language,
            # The text the scripts build on the page (TEXT in root.html).
            "messages": messages,
            "state": state,
            "state_version": state_version,
            "manual": manual,
            "camera_view_enabled": camera_view_fps > 0,
            # Rendered into the tiles so that the reload after every state
            # change (an arm starting, say) keeps them as they were.
            "camera_view_status": _camera_view_status(),
            "camera_views": CAMERA_VIEWS,
            "camera_view_default": CAMERA_VIEW_DEFAULT,
            # Rendered for the same reason: the lifter panel would otherwise
            # blink out until /arm-health's first snapshot.
            "lifter_present": _lifter_present(),
            "lifter_controls": _lifter_controls_enabled(),
        },
    )


@app.post("/api/error")
async def _api_error(request: Request):
    """Accept a failure from outside, to show it on the task screen."""
    try:
        payload = await request.json()
    except ValueError:
        payload = {}
    if not isinstance(payload, dict):
        payload = {}
    if _record_error(payload.get("source", "dataflow"), payload.get("message", "")):
        await _notify_state_changed()
    return {"ok": True}


@app.post("/errors/clear")
async def _clear_errors(request: Request):
    """Clear the failures shown on the task screen."""
    state.errors.clear()
    await _notify_state_changed()
    return RedirectResponse(request.url_for("_root"), 303)


@app.post("/start")
def _start(request: Request):
    _command_start()
    return RedirectResponse(request.url_for("_root"), 303)


@app.post("/skip")
def _skip(request: Request):
    """Skip the next task."""
    next_task()
    return RedirectResponse(request.url_for("_root"), 303)


@app.post("/success")
def _success(request: Request):
    _command_success()
    return RedirectResponse(request.url_for("_root"), 303)


@app.post("/fail")
def _fail(request: Request):
    _command_fail()
    return RedirectResponse(request.url_for("_root"), 303)


@app.post("/cancel")
def _cancel(request: Request):
    """Cancel the current episode."""
    node.send_output("command", pa.array(["cancel"]))
    state.collecting = False
    state.episode_number += 1
    return RedirectResponse(request.url_for("_root"), 303)


@app.post("/lifter/{direction}")
def _lifter(direction: str):
    """Move the lifter while a button is held: up / down, repeated, then stop.

    Refused while recording, so that an episode never has the height
    change under it, and without a KER leader (see LIFTER_COMMANDS);
    "stop" is always accepted.
    """
    if direction not in LIFTER_COMMANDS:
        return JSONResponse({"accepted": False, "reason": "unknown"}, status_code=404)
    if direction != "stop" and not _lifter_controls_enabled():
        return JSONResponse({"accepted": False, "reason": "not-ker"}, status_code=409)
    if direction != "stop" and state.collecting:
        return JSONResponse(
            {"accepted": False, "reason": "collecting"}, status_code=409
        )
    _command_lifter(direction)
    return {"accepted": True}


@app.get("/reference-video/{index}/{number}")
def _reference_video(index: int, number: int):
    """Serve a reference video file of a task (seekable: Range requests work)."""
    files = reference_video_files(index) if 0 <= index < len(tasks) else []
    if not 0 <= number < len(files):
        return Response(status_code=404)
    # The same URL can mean another file after the task set changes.
    return FileResponse(files[number], headers={"Cache-Control": "no-cache"})


@app.get("/events", response_class=EventSourceResponse)
async def _events(request: Request) -> AsyncIterable[ServerSentEvent]:
    try:
        last_version = int(request.query_params.get("since"))
    except (TypeError, ValueError):
        last_version = state_version
    while state.running:
        async with _state_changed:
            await _state_changed.wait_for(
                lambda: state_version != last_version or not state.running
            )
        if not state.running:
            break
        last_version = state_version
        yield ServerSentEvent(
            data={
                "collecting": state.collecting,
                "episode_number": state.episode_number,
                "task_index": state.task_index,
                "arm_status_right": state.arm_status_right,
                "arm_status_left": state.arm_status_left,
            },
            id=str(state_version),
        )


@app.get("/stats", response_class=EventSourceResponse)
async def _stats() -> AsyncIterable[ServerSentEvent]:
    """Push camera FPS / jitter snapshots to the browser every 500 ms."""
    while state.running:
        now = time.time()
        snapshot = {}
        for name, s in camera_stats.items():
            if name not in seen_inputs:
                # Not wired in this dataflow; the row stays hidden rather
                # than sitting at "-- Hz" for the whole session.
                continue
            series = camera_timestamps[name]
            if not series or now - series[-1] > CAMERA_STALE_AFTER_S:
                snapshot[name] = {"fps": 0.0, "jitter_ms": 0.0, "latency_ms": 0.0}
            else:
                latency = camera_latency_ms[name]
                snapshot[name] = {
                    "fps": s.fps,
                    "jitter_ms": s.jitter_ms,
                    "latency_ms": sum(latency) / len(latency) if latency else 0.0,
                }
        yield ServerSentEvent(data=snapshot)
        await asyncio.sleep(0.5)


@app.get("/vr-stats", response_class=EventSourceResponse)
async def _vr_stats() -> AsyncIterable[ServerSentEvent]:
    """Push the real VR stream Hz / jitter snapshot to the browser every 500 ms."""
    while state.running:
        now = time.time()
        if not vr_timestamps or now - vr_timestamps[-1] > VR_STALE_AFTER_S:
            snapshot = {"fps": 0.0, "jitter_ms": 0.0}
        else:
            snapshot = {"fps": vr_stats.fps, "jitter_ms": vr_stats.jitter_ms}
        snapshot["present"] = any(i in seen_inputs for i in VR_RECEIVE_TIMES_INPUTS)
        yield ServerSentEvent(data=snapshot)
        await asyncio.sleep(0.5)


@app.get("/ker-stats", response_class=EventSourceResponse)
async def _ker_stats() -> AsyncIterable[ServerSentEvent]:
    """Push the leader stream rate and the device it came from every 500 ms."""
    while state.running:
        now = time.time()
        if not ker_timestamps or now - ker_timestamps[-1] > KER_STALE_AFTER_S:
            snapshot = {"fps": 0.0, "jitter_ms": 0.0}
        else:
            snapshot = {"fps": ker_stats.fps, "jitter_ms": ker_stats.jitter_ms}
        snapshot["present"] = KER_POSITION_INPUT in seen_inputs
        snapshot["device"] = ker_device
        yield ServerSentEvent(data=snapshot)
        await asyncio.sleep(0.5)


@app.get("/arm-health", response_class=EventSourceResponse)
async def _arm_health() -> AsyncIterable[ServerSentEvent]:
    """Push per-motor and bus state, and the lifter's height, every 200 ms.

    Sampled rather than pushed on change: the arms and the lifter report at
    250 Hz. The rate is the lifter's, so that its height follows a move.
    The lifter rides along under "lifter" rather than in a stream of its
    own: the page already holds five, and a sixth would leave no connection
    for its own requests.
    """
    while state.running:
        now = time.time()
        lifter_fresh = (
            lifter.updated_at and now - lifter.updated_at <= LIFTER_STALE_AFTER_S
        )
        snapshot = {
            "lifter": {
                "present": _lifter_present(),
                "elevation_mm": lifter.elevation_mm if lifter_fresh else None,
                "moving": lifter.moving,
                "controls": _lifter_controls_enabled(),
            }
        }
        for side, health in arm_health.items():
            fresh = health.updated_at and now - health.updated_at <= (
                ARM_TEMPERATURE_STALE_AFTER_S
            )
            fps, jitter_ms = _series_rate(
                arm_state_timestamps[side], now, ARM_STATE_STALE_AFTER_S
            )
            snapshot[side] = {
                "present": f"arm_state_{side}" in seen_inputs,
                "fps": fps,
                "jitter_ms": jitter_ms,
                "mos": health.mos if fresh else [],
                "rotor": health.rotor if fresh else [],
                "motor_status": health.motor_status if fresh else [],
                "bus": _bus_state(health)
                if fresh
                else {"state": "UNKNOWN", "count": 0, "severity": "unknown"},
            }
        yield ServerSentEvent(data=snapshot)
        await asyncio.sleep(0.2)


@app.websocket("/cameras")
async def _cameras(websocket: WebSocket):
    """Push each viewed camera's newest frame and whether it is live.

    `?view=` picks the cameras sent (CAMERA_VIEWS); the page reconnects
    when the operator switches views.

    Frames are binary messages: the camera's name, a newline, the JPEG.
    Every CAMERA_VIEW_STATUS_INTERVAL_S a text message carries
    _camera_view_status() as JSON for every camera, including those not
    shown, so the other views' buttons can warn about a dead camera; it
    alone decides No signal.

    One socket carries every camera: the page already holds five SSE
    streams, and one MJPEG stream per camera would exhaust the browser's six
    connections per host. Only the newest frame is ever sent, so a slow
    browser skips frames instead of backing them up in this node, whose
    event loop also serves the dora inputs.
    """
    await websocket.accept()
    if camera_view_fps <= 0:
        await websocket.close()
        return
    view = websocket.query_params.get("view", CAMERA_VIEW_DEFAULT)
    names = CAMERA_VIEWS.get(view, CAMERA_VIEWS[CAMERA_VIEW_DEFAULT])
    sent = {name: 0 for name in names}
    status_sent_at = 0.0
    try:
        while state.running:
            now = time.monotonic()
            if now - status_sent_at >= CAMERA_VIEW_STATUS_INTERVAL_S:
                status_sent_at = now
                await websocket.send_text(json.dumps(_camera_view_status()))
            for name in names:
                seq = frame_seq[name]
                if seq == sent[name] or name not in latest_frames:
                    continue
                sent[name] = seq
                jpeg = latest_frames[name].to_numpy(zero_copy_only=False).tobytes()
                await websocket.send_bytes(name.encode() + b"\n" + jpeg)
            await asyncio.sleep(1.0 / camera_view_fps)
    except WebSocketDisconnect:
        pass


@app.post("/quit")
async def _quit():
    await _command_quit()
    return {"accepted": True}


@app.post("/arm/start")
def _arm_start(request: Request):
    """Start (power on) the arm(s)."""
    _command_arm_start()
    return RedirectResponse(request.url_for("_root"), 303)


@app.post("/arm/stop")
def _arm_stop(request: Request):
    """Pause (stop) the arm(s)."""
    _command_arm_stop()
    return RedirectResponse(request.url_for("_root"), 303)


def load_yaml(path):
    """Load a YAML file."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_manual(path):
    """Load the operation manual shown on the task screen.

    The file is YAML:

        title: Operation manual
        sections:
          - title: Aligning the arms
            steps: ["...", "..."]
            note: "..."
    """
    if not path:
        return None
    path = pathlib.Path(path)
    if not path.exists():
        _record_error("ui", t("manual_not_found", path=path))
        return None
    try:
        loaded = load_yaml(path)
    except yaml.YAMLError as error:
        _record_error("ui", t("manual_unreadable", error=error))
        return None
    if not isinstance(loaded, dict) or not loaded.get("sections"):
        _record_error("ui", t("manual_no_sections", path=path))
        return None
    return loaded


async def _main_uvicorn(server):
    await server.serve()


async def _main_dora(server):
    """Quit the Web application when this dataflow is stopped."""
    # Bring the arm(s) up on boot. dora-openarm no longer auto-starts
    _command_arm_start()
    last_values = {}
    while state.running:
        if node.is_empty():
            await asyncio.sleep(0.001)
            continue
        event = node.next()
        if event["type"] == "STOP":
            state.running = False
        elif event["type"] == "ERROR":
            # Show dora's own failures on the task screen, not just on stderr.
            _record_error("dora", _event_field(event, "error", "unknown dora error"))
            await _notify_state_changed()
        elif event["type"] == "INPUT_CLOSED":
            # A node that ends while we are still running has died.
            _record_error(
                "dora", t("input_stopped", input=_event_field(event, "id", "?"))
            )
            await _notify_state_changed()
        elif event["type"] == "INPUT":
            event_id = event["id"]
            seen_inputs.add(event_id)
            if event_id == KER_METADATA_INPUT:
                _update_ker_device(event["value"])
                continue
            if event_id == KER_POSITION_INPUT:
                _update_ker_stats(
                    _event_ts_to_seconds(event["metadata"].get("timestamp"))
                )
                continue
            if event_id in CAMERA_INPUTS:
                ts_s = _event_ts_to_seconds(event["metadata"].get("timestamp"))
                _update_camera_stats(event_id, ts_s)
                camera_latency_ms[event_id].append((time.time() - ts_s) * 1e3)
                _store_camera_frame(event_id, event["value"], event["metadata"])
                continue
            if event_id in ARM_STATUS_INPUTS:
                value = event["value"][0].as_py()
                # Only notify on an actual change. The follower may publish repeated (heartbeat) status values;
                if getattr(state, event_id) != value:
                    setattr(state, event_id, value)
                    await _notify_state_changed()
                continue
            if event_id in ARM_STATE_INPUTS:
                # Store only. Notifying here would push an SSE frame 250
                # times a second per arm; /arm-health samples instead.
                side = ARM_STATE_INPUTS[event_id]
                _add_arrival(
                    arm_state_timestamps[side],
                    _event_ts_to_seconds(event["metadata"].get("timestamp")),
                    ARM_STATE_STALE_AFTER_S,
                )
                _update_arm_health(side, event["value"])
                continue
            if event_id == LIFTER_OBSERVATION_INPUT:
                # 250 Hz: store only; /arm-health samples it.
                _update_lifter_elevation(event["value"])
                continue
            if event_id in VR_RECEIVE_TIMES_INPUTS:
                for ts_ns in event["value"].to_pylist():
                    _update_vr_stats(float(ts_ns) / 1e9)
                continue
            if event_id not in ("button_a", "button_b"):
                continue

            value = event["value"][0].as_py()
            triggered = value and not last_values.get(event_id, False)
            last_values[event_id] = value
            if not triggered:
                continue

            if state.collecting:
                if event_id == "button_a":
                    _command_success()
                elif event_id == "button_b":
                    _command_fail()
            else:
                if event_id == "button_a":
                    _command_start()
                elif event_id == "button_b":
                    await _command_quit()

            await _notify_state_changed()
    server.should_exit = True


async def _main_async():
    config = uvicorn.Config(app, port=port, log_level="info")
    server = uvicorn.Server(config)

    task_uvicorn = asyncio.create_task(_main_uvicorn(server))
    task_dora = asyncio.create_task(_main_dora(server))
    task_lifter = asyncio.create_task(_lifter_watchdog())

    await task_uvicorn
    # Process may linger when dora exits via SIGTERM,
    # as _main_dora() may not receive a STOP event.
    # Set `state.running = False` when task_uvicorn exits
    # so that _main_dora() also exits.
    state.running = False
    await task_dora
    await task_lifter


def main():
    """Run data collection control Web application."""
    global node
    global tasks
    global camera_view_fps

    parser = argparse.ArgumentParser(description="Record data as OpenArm dataset")
    parser.add_argument(
        "--metadata-file",
        default=os.getenv("METADATA_FILE"),
        help="The metadata file",
        type=pathlib.Path,
    )
    parser.add_argument(
        "--auto-open",
        action=argparse.BooleanOptionalAction,
        default=os.getenv("AUTO_OPEN", "") == "yes",
        help="Open a Web browser automatically",
    )
    default_port = 8000
    parser.add_argument(
        "--port",
        default=int(os.getenv("PORT", default_port)),
        help=f"The port for UI ({default_port})",
        type=int,
    )
    parser.add_argument(
        "--manual-file",
        default=os.getenv("MANUAL_FILE"),
        help="The operation manual (YAML) shown on the task screen",
        type=pathlib.Path,
    )
    parser.add_argument(
        "--reference-video-dir",
        default=os.getenv("REFERENCE_VIDEO_DIR") or DEFAULT_REFERENCE_VIDEO_DIR,
        help=f"Where the tasks' reference videos are ({DEFAULT_REFERENCE_VIDEO_DIR})",
        type=pathlib.Path,
    )
    parser.add_argument(
        "--camera-view-fps",
        default=float(os.getenv("CAMERA_VIEW_FPS", camera_view_fps)),
        help=f"Live camera view rate, 0 to disable ({camera_view_fps:g})",
        type=float,
    )
    # An option only, not an environment variable, so that the dataflow's
    # `args:` alone sets the language.
    parser.add_argument(
        "--language",
        default=DEFAULT_LANGUAGE,
        choices=available_languages(),
        help=f"The language of the UI ({DEFAULT_LANGUAGE})",
    )
    args = parser.parse_args()
    # Before anything below records an error in the UI's language.
    load_messages(args.language)
    camera_view_fps = args.camera_view_fps
    global auto_open
    auto_open = args.auto_open
    global port
    port = args.port
    global manual
    manual = load_manual(args.manual_file)
    global reference_video_dir
    reference_video_dir = args.reference_video_dir.expanduser()
    metadata = load_yaml(args.metadata_file)
    global task_set
    task_set = str(metadata.get("task_set") or args.metadata_file.stem)
    tasks = metadata.get("tasks")
    if not tasks:
        raise SystemExit(f"{args.metadata_file}: `tasks` is missing or empty")
    global lifter_equipped, leader
    try:
        lifter_equipped = "lifter" in metadata["equipment"]["embodiments"]
    except (KeyError, TypeError):
        # A bare task file (no robot equipment) has no lifter.
        lifter_equipped = False
    leader = metadata.get("leader")
    apply_task(state.task_index)

    node = dora.Node()
    asyncio.run(_main_async())


if __name__ == "__main__":
    main()
