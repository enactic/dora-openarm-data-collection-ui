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
import datetime
import dora
import json
from collections.abc import AsyncIterable
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.sse import EventSourceResponse, ServerSentEvent
from fastapi.templating import Jinja2Templates
import os
import pathlib
import pyarrow as pa
import pydantic
import threading
import time
import uvicorn
import webbrowser
import yaml

base_dir = os.path.dirname(__file__)
templates = Jinja2Templates(directory=f"{base_dir}/templates")

# The UI's text, one YAML file per language. English is used for text that
# isn't in the other languages.
LOCALES_DIR = pathlib.Path(base_dir) / "locales"
DEFAULT_LANGUAGE = "en"
# {language: {key: text}}. Languages are in lower case such as "zh-tw".
messages = {}

node = None

auto_open = False
port = None


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """Open a Web browser automatically if requested."""
    if auto_open:
        url = f"http://127.0.0.1:{port}"
        # In a thread because webbrowser.open() blocks until the browser
        # exits with some browsers such as one specified by $BROWSER.
        threading.Thread(target=webbrowser.open, args=(url,), daemon=True).start()
    yield


app = FastAPI(lifespan=_lifespan)


@dataclasses.dataclass
class State:
    """The current state."""

    collecting: bool = False
    # time.monotonic() when the current episode started.
    collecting_started_at: float = 0.0
    running: bool = True
    episode_number: int = 0
    task_index: int = 0
    arm_status_right: str = "stopped"
    arm_status_left: str = "stopped"
    # Errors to show in the errors panel: {what: error}. The newest is
    # the last. See _record_error().
    errors: dict = dataclasses.field(default_factory=dict)


state = State()

# The number of errors to keep. Older errors are forgotten.
MAX_ERRORS = 20


def _record_error(source, message="", key=None, values=None):
    """Remember an error to show it in the errors panel.

    `message` is shown as is. Use `key` and `values` instead to show a
    text in locales/ in the UI's language.

    The same error again counts up the existing one and moves it to
    the newest.

    This doesn't change `state_version`. /events sends the errors
    without reloading the page. A reload cancels long presses such as
    skip.
    """
    # On one line.
    message = " ".join(str(message).split())
    values = {name: str(value) for name, value in (values or {}).items()}
    what = (source, message, key, tuple(values.items()))
    old = state.errors.pop(what, None)
    state.errors[what] = {
        "source": source,
        "message": message,
        "key": key,
        "values": values,
        "count": old["count"] + 1 if old else 1,
        "time": datetime.datetime.now().strftime("%H:%M:%S"),
    }
    if len(state.errors) > MAX_ERRORS:
        del state.errors[next(iter(state.errors))]


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


def _update_camera_stats(event_id: str, ts_s: float) -> None:
    _fold_arrival(
        camera_timestamps[event_id],
        camera_stats[event_id],
        ts_s,
        CAMERA_STALE_AFTER_S,
    )


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


def next_task():
    """Update the state with the next task."""
    state.task_index += 1
    if state.task_index >= len(tasks):
        state.task_index = 0


def _command_start():
    """Start a new episode."""
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
    state.collecting_started_at = time.monotonic()


# Success and fail just after start are ignored. They are probably
# accidental double presses. This is for all of the controller, the
# mouse and other tabs. actions.js also has MIN_RECORDING_MS for the
# keyboard to show a message.
MIN_RECORDING_S = 1.0


def _recording_too_short():
    """Return whether the current episode has just started."""
    return (
        state.collecting
        and time.monotonic() - state.collecting_started_at < MIN_RECORDING_S
    )


def _command_success():
    """Finish the current episode successfully."""
    if _recording_too_short():
        return
    node.send_output("command", pa.array(["success"]))
    state.collecting = False
    state.episode_number += 1
    next_task()


def _command_fail():
    """Finish the current episode unsuccessfully."""
    if _recording_too_short():
        return
    node.send_output("command", pa.array(["fail"]))
    state.collecting = False
    state.episode_number += 1
    next_task()


def _command_quit():
    """Quit this data collection."""
    node.send_output("command", pa.array(["quit"]))
    state.running = False


def _command_arm_start():
    """Start (power on) the arm(s)."""
    node.send_output("arm_command", pa.array(["start"]))


def _command_arm_stop():
    """Pause (stop) the arm(s)."""
    node.send_output("arm_command", pa.array(["stop"]))


def _accepted_languages(header):
    """Return the language tags in an Accept-Language header, preferred first."""
    tags = []
    for item in header.split(","):
        tag, _, parameters = item.partition(";")
        try:
            q = float(parameters.strip().removeprefix("q=") or 1)
        except ValueError:
            continue
        if q > 0:
            tags.append((q, tag.strip()))
    tags.sort(key=lambda tag: -tag[0])
    return [tag for _, tag in tags]


def _parent_languages(language):
    """Return the language and its parents: "zh-hant-tw", "zh-hant", "zh"."""
    subtags = language.lower().split("-")
    return ["-".join(subtags[:i]) for i in range(len(subtags), 0, -1)]


def _find_language(tag):
    """Return the UI's language for a language tag or None.

    The most specific one is used: "zh-Hant-TW", "zh-Hant" and then "zh".
    """
    for language in _parent_languages(tag):
        if language in messages:
            return language
    return None


def _task_title(task, language):
    """Return the task's prompt in the language if it has one.

    The most specific one is used: prompt_zh-TW, prompt_zh and then prompt.
    """
    prompts = {str(key).lower(): value for key, value in task.items()}
    for parent in _parent_languages(language):
        prompt = prompts.get(f"prompt_{parent}")
        if prompt:
            return prompt
    return task["prompt"]


def _language(request):
    """Return the UI's language: ?lang=, Accept-Language or the default."""
    tags = [request.query_params.get("lang", "")]
    tags += _accepted_languages(request.headers.get("accept-language", ""))
    for tag in tags:
        language = _find_language(tag)
        if language:
            return language
    return DEFAULT_LANGUAGE


def _url_for(request, name):
    """Return the URL for the name with ?lang= if it's specified."""
    url = request.url_for(name)
    lang = request.query_params.get("lang")
    if lang:
        url = url.include_query_params(lang=lang)
    return url


@app.get("/", response_class=HTMLResponse)
def _root(request: Request):
    """Render the main HTML."""
    language = _language(request)
    texts = messages[language]

    def t(key, default=None, /, **values):
        """Return the text for the key in the UI's language.

        `{name}`s in the text are filled in with `values`.
        """
        if default is None:
            text = texts[key]
        else:
            text = texts.get(key, default)
        if values:
            text = text.format(**values)
        return text

    return templates.TemplateResponse(
        request=request,
        name="root.html",
        context={
            "language": language,
            "languages": {
                code: messages[code]["language_name"] for code in sorted(messages)
            },
            "t": t,
            # Display only. The recorder uses `prompt`.
            "task_title": _task_title(tasks[state.task_index], language),
            # Keep ?lang= after a form is submitted.
            "url_for": lambda name: _url_for(request, name),
            "state": state,
            "state_version": state_version,
        },
    )


@app.post("/start")
async def _start(request: Request):
    _command_start()
    await _notify_state_changed()
    return RedirectResponse(_url_for(request, "_root"), 303)


@app.post("/skip")
async def _skip(request: Request):
    """Skip the next task."""
    next_task()
    await _notify_state_changed()
    return RedirectResponse(_url_for(request, "_root"), 303)


@app.post("/success")
async def _success(request: Request):
    _command_success()
    await _notify_state_changed()
    return RedirectResponse(_url_for(request, "_root"), 303)


@app.post("/fail")
async def _fail(request: Request):
    _command_fail()
    await _notify_state_changed()
    return RedirectResponse(_url_for(request, "_root"), 303)


@app.post("/cancel")
async def _cancel(request: Request):
    """Cancel the current episode."""
    node.send_output("command", pa.array(["cancel"]))
    state.collecting = False
    state.episode_number += 1
    await _notify_state_changed()
    return RedirectResponse(_url_for(request, "_root"), 303)


class ErrorReport(pydantic.BaseModel):
    """An error from another program."""

    source: str = "api"
    # Not empty nor only spaces.
    message: str = pydantic.Field(pattern=r"\S")


@app.post("/api/errors")
async def _api_errors(error: ErrorReport):
    """Show an error from another program in the errors panel."""
    _record_error(error.source, error.message)
    return {}


@app.post("/errors/clear")
async def _clear_errors(request: Request):
    """Clear the errors panel."""
    state.errors.clear()
    return RedirectResponse(_url_for(request, "_root"), 303)


# How often /events sends the errors, the stream statistics and the arm
# health.
STATS_INTERVAL_S = 0.5


def _state_snapshot() -> dict:
    return {
        "collecting": state.collecting,
        "episode_number": state.episode_number,
        "task_index": state.task_index,
        "arm_status_right": state.arm_status_right,
        "arm_status_left": state.arm_status_left,
    }


def _camera_stats_snapshot() -> dict:
    """Return camera FPS / jitter."""
    now = time.time()
    snapshot = {}
    for name, s in camera_stats.items():
        if name not in seen_inputs:
            # Not wired in this dataflow; the row stays hidden rather
            # than sitting at "-- Hz" for the whole session.
            continue
        series = camera_timestamps[name]
        if not series or now - series[-1] > CAMERA_STALE_AFTER_S:
            snapshot[name] = {"fps": 0.0, "jitter_ms": 0.0}
        else:
            snapshot[name] = {"fps": s.fps, "jitter_ms": s.jitter_ms}
    return snapshot


def _vr_stats_snapshot() -> dict:
    """Return the real VR stream Hz / jitter."""
    now = time.time()
    if not vr_timestamps or now - vr_timestamps[-1] > VR_STALE_AFTER_S:
        snapshot = {"fps": 0.0, "jitter_ms": 0.0}
    else:
        snapshot = {"fps": vr_stats.fps, "jitter_ms": vr_stats.jitter_ms}
    snapshot["present"] = any(i in seen_inputs for i in VR_RECEIVE_TIMES_INPUTS)
    return snapshot


def _ker_stats_snapshot() -> dict:
    """Return the leader stream rate and the device it came from."""
    now = time.time()
    if not ker_timestamps or now - ker_timestamps[-1] > KER_STALE_AFTER_S:
        snapshot = {"fps": 0.0, "jitter_ms": 0.0}
    else:
        snapshot = {"fps": ker_stats.fps, "jitter_ms": ker_stats.jitter_ms}
    snapshot["present"] = KER_POSITION_INPUT in seen_inputs
    snapshot["device"] = ker_device
    return snapshot


def _arm_health_snapshot() -> dict:
    """Return per-motor temperatures, motor status and bus state.

    Sampled rather than pushed on change: the arms report at 250 Hz, and
    none of this needs to reach an operator sooner than this.
    """
    now = time.time()
    snapshot = {}
    for side, health in arm_health.items():
        fresh = health.updated_at and now - health.updated_at <= (
            ARM_TEMPERATURE_STALE_AFTER_S
        )
        snapshot[side] = {
            "present": f"arm_state_{side}" in seen_inputs,
            "mos": health.mos if fresh else [],
            "rotor": health.rotor if fresh else [],
            "motor_status": health.motor_status if fresh else [],
            "bus": _bus_state(health)
            if fresh
            else {"state": "UNKNOWN", "count": 0, "severity": "unknown"},
        }
    return snapshot


def _errors_snapshot(language) -> list:
    """Return the errors, the newest first, in the language."""
    return [
        {
            "time": error["time"],
            "source": error["source"],
            "count": error["count"],
            "message": messages[language][error["key"]].format(**error["values"])
            if error["key"]
            else error["message"],
        }
        for error in reversed(state.errors.values())
    ]


@app.get("/events", response_class=EventSourceResponse)
async def _events(request: Request) -> AsyncIterable[ServerSentEvent]:
    """Push everything the page shows over one stream.

    One stream per page, not one per panel: a browser opens only 6
    connections to a host, shared by all of its tabs. With a stream per
    panel, a second tab used them all up and no form could be submitted.

    A "state" event is sent on every state change. The page reloads on
    it. The others are sent every STATS_INTERVAL_S. Text in them such
    as errors is in the language of `?lang=` or the one that the Web
    browser prefers.
    """
    try:
        last_version = int(request.query_params.get("since"))
    except (TypeError, ValueError):
        last_version = state_version
    language = _language(request)
    stats_at = 0.0
    while state.running:
        if state_version != last_version:
            last_version = state_version
            yield ServerSentEvent(
                event="state", data=_state_snapshot(), id=str(state_version)
            )
        now = time.monotonic()
        if now >= stats_at:
            stats_at = now + STATS_INTERVAL_S
            yield ServerSentEvent(event="errors", data=_errors_snapshot(language))
            yield ServerSentEvent(event="camera-stats", data=_camera_stats_snapshot())
            yield ServerSentEvent(event="vr-stats", data=_vr_stats_snapshot())
            yield ServerSentEvent(event="ker-stats", data=_ker_stats_snapshot())
            yield ServerSentEvent(event="arm-health", data=_arm_health_snapshot())
        async with _state_changed:
            try:
                await asyncio.wait_for(
                    _state_changed.wait_for(
                        lambda: state_version != last_version or not state.running
                    ),
                    max(0.0, stats_at - time.monotonic()),
                )
            except asyncio.TimeoutError:
                pass


@app.post("/quit")
def _quit(request: Request):
    _command_quit()
    return RedirectResponse(_url_for(request, "_root"), 303)


@app.post("/arm/start")
def _arm_start(request: Request):
    """Start (power on) the arm(s)."""
    _command_arm_start()
    return RedirectResponse(_url_for(request, "_root"), 303)


@app.post("/arm/stop")
def _arm_stop(request: Request):
    """Pause (stop) the arm(s)."""
    _command_arm_stop()
    return RedirectResponse(_url_for(request, "_root"), 303)


def load_yaml(path):
    """Load a YAML file."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_messages():
    """Load the UI's text in all languages.

    Text that isn't in a language is taken from its parent languages such
    as "zh" for "zh-TW" and then English.
    """
    files = {path.stem.lower(): load_yaml(path) for path in LOCALES_DIR.glob("*.yaml")}
    loaded = {}
    for language, texts in files.items():
        merged = dict(files[DEFAULT_LANGUAGE])
        for parent in reversed(_parent_languages(language)):
            merged |= files.get(parent, {})
        # Not from other languages: each language has its own name.
        merged["language_name"] = texts.get("language_name", language)
        loaded[language] = merged
    return loaded


def load_metadata(path):
    """Load a metadata file and validate it.

    Raises ValueError when it can't be used.
    """
    metadata = load_yaml(path)
    tasks = metadata.get("tasks") if isinstance(metadata, dict) else None
    if not tasks:
        raise ValueError(f"{path}: `tasks` is missing or empty")
    # Check all tasks now. A broken one would otherwise fail only when
    # it's reached during data collection.
    if not isinstance(tasks, list) or not all(
        isinstance(task, dict) and "prompt" in task for task in tasks
    ):
        raise ValueError(f"{path}: each task in `tasks` must have `prompt`")
    return metadata


async def _main_uvicorn(server):
    await server.serve()


async def _main_dora(server):
    """Quit the Web application when this dataflow is stopped."""
    try:
        await _process_dora_events(server)
    finally:
        # Even when processing failed. Otherwise, the page stays but
        # nothing works, and SIGTERM can't stop this node.
        state.running = False
        # Wake up /events streams. Otherwise, uvicorn waits for them forever.
        await _notify_state_changed()
        server.should_exit = True


async def _process_dora_events(server):
    """Process dora events until this dataflow is stopped."""
    # Bring the arm(s) up on boot. dora-openarm no longer auto-starts
    _command_arm_start()
    last_values = {}
    # uvicorn sets `should_exit` by itself on SIGINT/SIGTERM.
    while state.running and not server.should_exit:
        if node.is_empty():
            await asyncio.sleep(0.001)
            continue
        event = node.next()
        if event["type"] == "STOP":
            state.running = False
        elif event["type"] == "ERROR":
            _record_error("dora", event["error"])
        elif event["type"] == "NODE_FAILED":
            # Not INPUT_CLOSED. It's also sent when a node finishes
            # normally.
            failure = event["value"]
            _record_error(
                "dora",
                key="node_failed",
                values={"node": failure["source_node_id"], "error": failure["error"]},
            )
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
                _update_camera_stats(
                    event_id,
                    _event_ts_to_seconds(event["metadata"].get("timestamp")),
                )
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
                _update_arm_health(ARM_STATE_INPUTS[event_id], event["value"])
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
                    _command_quit()

            await _notify_state_changed()


async def _main_async():
    config = uvicorn.Config(app, port=port, log_level="info")
    server = uvicorn.Server(config)

    task_uvicorn = asyncio.create_task(_main_uvicorn(server))
    task_dora = asyncio.create_task(_main_dora(server))

    await task_uvicorn
    # Process may linger when dora exits via SIGTERM,
    # as _main_dora() may not receive a STOP event.
    # Set `state.running = False` when task_uvicorn exits
    # so that _main_dora() also exits.
    state.running = False
    await task_dora


def main():
    """Run data collection control Web application."""
    global node
    global tasks

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
    args = parser.parse_args()
    global messages
    messages = load_messages()
    global auto_open
    auto_open = args.auto_open
    global port
    port = args.port
    if args.metadata_file is None:
        parser.error("--metadata-file or METADATA_FILE is required")
    try:
        metadata = load_metadata(args.metadata_file)
    except ValueError as error:
        parser.error(str(error))
    tasks = metadata["tasks"]

    node = dora.Node()
    asyncio.run(_main_async())


if __name__ == "__main__":
    main()
