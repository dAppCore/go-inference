<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# chat/cancel

Cancelling generation: two different mechanisms, easy to conflate.
context.WithCancel stops the ENGINE — the backend observes ctx.Done() and
tears down mid-generation. Breaking the range loop stops the CONSUMER —
the iterator simply isn't asked for another token, and the engine notices
on its next step and winds down; no context involved.

## Run

```sh
go run ./pkg/chat/cancel -model ~/models/gemma-4-e2b-it-4bit
```

Flags and behaviour are documented in [main.go](main.go) — the code is the example.
