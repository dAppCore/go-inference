<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# train/ssd

Self-distillation sampling (SSD): sample the FROZEN base model over a
prompt set and capture every self-output at birth into a trace. SSD does
NOT train — the trace (ssd-captures.jsonl) is the deliverable; a later
curation step picks rows from it and a separate SFT run (pkg/train/sft)
teaches them back.

## Run

```sh
go run ./pkg/train/ssd -model ~/models/gemma-4-E2B-it-bf16 -out ./ssd-out
```

Flags and behaviour are documented in [main.go](main.go) — the code is the example.
