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
from contextlib import asynccontextmanager
import dataclasses
import dora
from collections.abc import AsyncIterable
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.sse import EventSourceResponse, ServerSentEvent
from fastapi.templating import Jinja2Templates
import os
import pathlib
import pyarrow as pa
import uvicorn
import yaml

base_dir = os.path.dirname(__file__)
templates = Jinja2Templates(directory=f"{base_dir}/templates")

node = None

auto_open = False
port = None


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """Open a Web browser automatically if requested."""
    if auto_open:
        url = f"http://127.0.0.1:{port}"
        await asyncio.create_subprocess_exec("open", url)
    yield


app = FastAPI(lifespan=_lifespan)


@dataclasses.dataclass
class State:
    """The current state."""

    collecting: bool = False
    running: bool = True
    episode_number: int = 0
    task_index: int = 0
    task_title: str = ""


state = State()

_state_changed = asyncio.Condition()


async def _notify_state_changed() -> None:
    async with _state_changed:
        _state_changed.notify_all()


def next_task():
    """Update the state with the next task."""
    state.task_index += 1
    if state.task_index >= len(tasks):
        state.task_index = 0
    state.task_title = tasks[state.task_index]["prompt"]


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


def _command_quit():
    """Quit this data collection."""
    node.send_output("command", pa.array(["quit"]))
    state.running = False


@app.get("/", response_class=HTMLResponse)
def _root(request: Request):
    """Render the main HTML."""
    return templates.TemplateResponse(
        request=request, name="root.html", context={"state": state}
    )


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


@app.get("/events", response_class=EventSourceResponse)
async def _events() -> AsyncIterable[ServerSentEvent]:
    while state.running:
        async with _state_changed:
            await _state_changed.wait()
        yield ServerSentEvent(
            data={
                "collecting": state.collecting,
                "episode_number": state.episode_number,
                "task_index": state.task_index,
            }
        )


@app.post("/quit")
def _quit(request: Request):
    _command_quit()
    return RedirectResponse(request.url_for("_root"), 303)


def load_yaml(path):
    """Load a YAML file."""
    with open(path) as f:
        return yaml.safe_load(f)


async def _main_uvicorn(server):
    await server.serve()


async def _main_dora(server):
    """Quit the Web application when this dataflow is stopped."""
    last_values = {}
    while state.running:
        if node.is_empty():
            await asyncio.sleep(0.1)
            continue
        event = node.next()
        if event["type"] == "STOP":
            state.running = False
        elif event["type"] == "INPUT":
            event_id = event["id"]
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
    server.should_exit = True


async def _main_async():
    config = uvicorn.Config(app, port=port, log_level="info")
    server = uvicorn.Server(config)

    task_uvicorn = asyncio.create_task(_main_uvicorn(server))
    task_dora = asyncio.create_task(_main_dora(server))

    await task_uvicorn
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
    global auto_open
    auto_open = args.auto_open
    global port
    port = args.port
    metadata = load_yaml(args.metadata_file)
    tasks = metadata["tasks"]
    state.task_title = tasks[state.task_index]["prompt"]

    node = dora.Node()
    asyncio.run(_main_async())


if __name__ == "__main__":
    main()
