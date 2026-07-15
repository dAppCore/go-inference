<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# tokenizer

Tokenise text without loading a model or touching the GPU: tokenizer.json
is a standalone artifact every model snapshot carries, so token counting
and the encode/decode plumbing work with no engine import at all — like
pkg/discover, this example never blank-imports examples/internal/engine.

## Run

```sh
go run ./pkg/tokenizer -model ~/models/gemma-4-e2b-it-4bit
```

Flags and behaviour are documented in [main.go](main.go) — the code is the example.
