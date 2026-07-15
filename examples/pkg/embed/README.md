<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# embed

Text embeddings via the host BERT encoder. model/bert is a pure-CPU
float32 forward pass — it never touches the GPU registry the chat/eval
examples blank-import (no metallib, no engine backend needed), so this
example has no `_ "dappco.re/go/inference/examples/internal/engine"` line.

Fetch a snapshot first:

hf download BAAI/bge-small-en-v1.5

## Run

```sh
go run ./pkg/embed -model ~/.cache/huggingface/hub/models--BAAI--bge-small-en-v1.5/snapshots/<rev>
```

Flags and behaviour are documented in [main.go](main.go) — the code is the example.
