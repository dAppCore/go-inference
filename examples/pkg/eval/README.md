<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# eval

A minimal eval harness on the Classify fast path: batched prefill-only
inference — each prompt gets ONE forward pass and the sampled token at the
last position is the model's one-token answer. That makes label tasks
(sentiment, topic, yes/no) cheap enough to run inside a test suite.

## Run

```sh
go run ./pkg/eval -model ~/models/gemma-4-e2b-it-4bit
```

Flags and behaviour are documented in [main.go](main.go) — the code is the example.
