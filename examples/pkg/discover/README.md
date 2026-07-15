<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# discover

Discover walks a directory tree looking for model snapshots — any directory
containing config.json plus at least one *.safetensors file. It needs no
loaded model and no GPU engine, so unlike the other examples in this tree
it does not blank-import examples/internal/engine.

## Run

```sh
go run ./pkg/discover -dir ~/.cache/huggingface/hub
```

Flags and behaviour are documented in [main.go](main.go) — the code is the example.
