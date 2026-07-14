<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# chat/stream

Token streaming: print each token the moment it decodes — the shape a
terminal UI or SSE endpoint wants. The Chat iterator IS the stream; there
is no separate streaming API to opt into.

## Run

```sh
go run ./pkg/chat/stream -model ~/models/gemma-4-e2b-it-4bit
```

Flags and behaviour are documented in [main.go](main.go) — the code is the example.
