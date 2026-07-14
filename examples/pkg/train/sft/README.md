<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# train/sft

LoRA supervised fine-tuning: teach the model {"messages": [...]} assistant
turns from a JSONL set. Train on a bf16 snapshot (the quantised 4-bit
snapshots are for serving); the run writes an adapter you apply at load
time rather than merging into the base weights.

## Run

```sh
go run ./pkg/train/sft -model ~/models/gemma-4-E2B-it-bf16 -out ./sft-out
```

Flags and behaviour are documented in [main.go](main.go) — the code is the example.
