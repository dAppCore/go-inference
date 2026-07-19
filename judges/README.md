# Judge templates

`lem data score <slug> --kind judge:<name> --model <checkpoint path>` runs a
named prompt template through a loaded judge model to score dataset items —
the judge tier described in
[`../docs/superpowers/specs/2026-07-19-lem-dataset-loop-design.md`](../docs/superpowers/specs/2026-07-19-lem-dataset-loop-design.md)
("Scoring").

## Resolution order

For `--kind judge:<name>`:

1. `~/.lem/judges/<name>.md` — a user override, if present. **Always wins.**
2. `judges/<name>.md` (this directory) — the in-repo default.

Present in neither: a loud "unknown judge template" error — never a silent
fallback and never a silent 0 score.

## File format

One Markdown file per template, named `<name>.md` — the filename stem IS the
`<name>` in `judge:<name>`. Front matter, delimited by `---` lines, then the
prompt body:

```
---
name: quality
description: One-line summary of what this template scores.
range: 0-100
---
The prompt body. Use {{prompt}} and {{response}} as placeholders — the
driver fills them in with the dataset item's prompt and response text
before sending the result to the judge model. End with an explicit
instruction to reply with only a bare number.
```

Front-matter fields (all three required):

- **`name`** — must equal the filename stem exactly (`quality.md` ->
  `name: quality`). A mismatch is a loud error, not a silently-tolerated
  rename or typo.
- **`description`** — one line, human-readable; documents intent.
- **`range`** — `MIN-MAX`, two non-negative numbers separated by a single
  `-` (e.g. `0-100`, `0-1`). The judge model's reply must parse to a number
  inside this range.

## Prompt body

Plain text with exactly two placeholders, `{{prompt}}` and `{{response}}`,
replaced verbatim — no escaping, no nested templating, deliberately minimal.
Always end the body with an explicit instruction to reply with **only a
bare number**: the driver's parser is strict — the model's ENTIRE trimmed
reply must parse as one number in range. Prose, units, or a number buried in
a sentence ("I'd say around 85") is rejected as a loud per-item failure
rather than best-effort-extracted — a judge that won't follow the bare-number
instruction is a real signal, not something to paper over with a guess.

## Defaults shipped here

- **`quality.md`** — overall response quality/helpfulness, 0-100.
- **`factuality.md`** — factual accuracy of the response's claims, 0-100.
- **`refusal-correctness.md`** — whether the response's help-or-refuse call
  was the right one, 0-100.

## Adding your own

Drop a `<name>.md` file in `~/.lem/judges/` following the format above — no
rebuild required, it is read at run time. Use the same name as an in-repo
default to override it, or a new name to add a template the in-repo set
doesn't have.
