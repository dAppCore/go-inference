<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# chat/multiturn

Multi-turn chat: go-inference is stateless per call — there is no
server-side session. Each Chat call resends the FULL message history; the
model only "remembers" earlier turns because they are still in the slice
you pass it.

## Run

```sh
go run ./pkg/chat/multiturn -model ~/models/gemma-4-e2b-it-4bit
```

Flags and behaviour are documented in [main.go](main.go) — the code is the example.
