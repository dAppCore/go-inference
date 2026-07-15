<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# chat/basic

The smallest possible go-inference chat call: load a model directory, run
one user turn through the model's own chat template, print the reply.

Gemma 4 reasons in a thought channel by default; this example turns that
off so the model answers directly (see pkg/chat/thinking for the channel).

## Run

```sh
go run ./pkg/chat/basic -model ~/models/gemma-4-e2b-it-4bit
```

Flags and behaviour are documented in [main.go](main.go) — the code is the example.
