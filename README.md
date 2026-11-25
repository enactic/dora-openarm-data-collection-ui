# dora-openarm-data-collection-ui

A [dora-rs](https://dora-rs.ai/) node that provides UI for data collection with OpenArm.

## Configuration

| Environment variable | Option | Description |
| --- | --- | --- |
| `METADATA_FILE` | `--metadata-file` | Metadata with the task list |
| `MANUAL_FILE` | `--manual-file` | Operation manual (YAML) shown on the task screen |
| `REFERENCE_VIDEO_DIR` | `--reference-video-dir` | Where the tasks' reference videos are (default `task-reference`): `<dir>/<task set>/<task number from 1>/*.mp4`, or the file `tasks[].reference_video` names |
| `PORT` | `--port` | Port of this UI (default `8000`) |
| `AUTO_OPEN` | `--auto-open` | Open a Web browser on start |
| `CAMERA_VIEW_FPS` | `--camera-view-fps` | Rate of the live camera view (default `10`, `0` turns it off) |

The manual file is:

```yaml
title: Operation manual
sections:
  - title: Aligning the arms
    steps: ["...", "..."]     # numbered
    bullets: ["...", "..."]   # unnumbered
    note: "..."               # highlighted warning
```

## Languages

The UI's text is in `src/dora_openarm_data_collection_ui/locales/`, one YAML
file per language, by key. `en.yaml` has every key, and a language falls back
to it for any key it does not have. To add a language, add `<language>.yaml`
with the keys translated; `--language` then accepts it.

The language is set with the `--language` option only, in the dataflow:

```yaml
  - id: ui
    path: dora-openarm-data-collection-ui
    args: --language ja
```

Tasks in the metadata may carry `prompt_<language>` and
`description_<language>` (for example `prompt_ja`). They are shown instead of
`prompt` / `description` in that language. The recorder still writes `prompt`
to the dataset as its language instruction.

## Live camera view

The task screen shows the cameras live next to the task, in one of three
views picked with the tabs above them:

| View | Cameras |
| --- | --- |
| Head (default) | `camera_head_left` |
| Head + wrists | `camera_head_left`, `camera_wrist_left`, `camera_wrist_right` |
| All cameras | `camera_head_left`, `camera_head_right`, `camera_ceiling`, `camera_wrist_left`, `camera_wrist_right` |

A tab shows ⚠ when one of its cameras stops, even while another view is on
screen. A tile shows No signal once its camera has sent nothing for 2 s.

Frames must be JPEG (`encoding` metadata `jpeg`, or none). They are forwarded
as they are over one WebSocket (`/cameras?view=head|wrists|all`), newest frame
only, at `CAMERA_VIEW_FPS`. Only the picked view's cameras are sent, and the
page closes the socket while its tab is hidden.

## Errors

Failures are shown in red on the task screen instead of only on a terminal:

- dora `ERROR` events and inputs that stop while the dataflow runs,
- anything posted to `POST /api/error` with `{"source": ..., "message": ...}`,
  for example from a program that watches the dataflow's output,
- camera streams that stop delivering frames,
- the loss of this node itself.

`POST /errors/clear` clears the panel.

## License

Licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.

Copyright 2026 Enactic, Inc.

## Code of Conduct

All participation in the OpenArm project is governed by our [Code of Conduct](CODE_OF_CONDUCT.md).
