<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# generate

Raw completion versus chat: Generate continues the prompt text verbatim —
no turn markers, no system/user framing, just the model predicting what
comes next. Chat (see pkg/chat/basic) wraps the same call in the model's
native chat template before it ever reaches the engine. Point this at a
base/pretrained snapshot for the clearest contrast; an instruction-tuned
model will still continue raw text, just less fluently.

## Run

```sh
go run ./pkg/generate -model ~/models/gemma-4-e2b-4bit
```

Flags and behaviour are documented in [main.go](main.go) — the code is the example.
