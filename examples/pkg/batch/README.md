<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# batch

Batch generation: BatchGenerate runs full autoregressive decoding for every
prompt in one call (parallel, unlike Classify's single-forward-pass fast
path — see pkg/classify). The Result carries one inference.BatchResult per
prompt, each with its own Tokens and a per-prompt Err so one bad prompt
(cancelled context, OOM) doesn't fail the whole batch.

## Run

```sh
go run ./pkg/batch -model ~/models/gemma-4-e2b-it-4bit
```

Flags and behaviour are documented in [main.go](main.go) — the code is the example.
