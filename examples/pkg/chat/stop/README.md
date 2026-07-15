<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# chat/stop

Stop control: WithStopTokens halts generation the instant a listed token ID
is sampled, WithSuppressTokens removes an ID from the distribution
entirely, and WithMinTokensBeforeStop delays stop-token matching until n
tokens have been emitted. Token IDs are model-specific, so this example
obtains one honestly — from a baseline generation's own Token.ID stream —
rather than guessing a value.

## Run

```sh
go run ./pkg/chat/stop -model ~/models/gemma-4-e2b-it-4bit
```

Flags and behaviour are documented in [main.go](main.go) — the code is the example.
