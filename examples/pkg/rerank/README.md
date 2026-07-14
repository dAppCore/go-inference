<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# rerank

Document reranking via the host BERT encoder — same bert.Load as pkg/embed.
model/bert never touches the GPU registry the chat/eval examples
blank-import (no metallib, no engine backend), so this example carries no
`_ "dappco.re/go/inference/examples/internal/engine"` line either.

Fetch a snapshot first:

hf download BAAI/bge-small-en-v1.5

## Run

```sh
go run ./pkg/rerank -model ~/.cache/huggingface/hub/models--BAAI--bge-small-en-v1.5/snapshots/<rev>
```

Flags and behaviour are documented in [main.go](main.go) — the code is the example.
