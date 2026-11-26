# dora-openarm-data-collection-ui

A [dora-rs](https://dora-rs.ai/) node that provides UI for data collection with OpenArm.

## Languages

The UI is shown in the language that the Web browser prefers. English is
used for languages that the UI doesn't have. Use the menu at the top right
or add `?lang=<language>` such as `?lang=ja` to the URL to use another
language.

The UI's text is in `src/dora_openarm_data_collection_ui/locales/`, one YAML
file per language. To add a language, add `<language>.yaml` such as
`ja.yaml` or `zh-TW.yaml` with `language_name`, the language's name in
itself. `en.yaml` has all text. Text that isn't in a language is taken from
its parent language such as `zh.yaml` for `zh-TW.yaml` and then `en.yaml`.

Tasks in the metadata may have `prompt_<language>` such as `prompt_ja`. It's
shown instead of `prompt` in that language. Its parent language is also used
such as `prompt_zh` for `zh-TW`. The recorder still uses `prompt`.

## Errors

The errors panel below the task shows:

- errors from dora,
- nodes in the dataflow that failed,
- errors sent to `POST /api/errors` with a JSON object such as
  `{"source": "recorder", "message": "Disk full"}`. It's for other
  programs such as one that watches the dataflow.

```bash
curl -X POST -H "Content-Type: application/json" \
  -d '{"source": "recorder", "message": "Disk full"}' \
  http://127.0.0.1:8000/api/errors
```

The same error again counts up instead of adding a new one. "Clear errors"
in the panel clears them.

## License

Licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.

Copyright 2026 Enactic, Inc.

## Code of Conduct

All participation in the OpenArm project is governed by our [Code of Conduct](CODE_OF_CONDUCT.md).
