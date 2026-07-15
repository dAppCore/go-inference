<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# classify

Classify plus WithLogits: each prompt gets one forward pass (see
pkg/eval for the plain top-token version); WithLogits also returns the raw
vocab-sized logits, so a caller can score confidence instead of only
reading the sampled token.

TextModel exposes no tokenizer-encode call, so there is no public way to
turn the strings "positive"/"negative" into vocab ids directly. Instead
this example BOOTSTRAPS the two candidate ids from a calibration Classify
call: two unambiguous reviews whose sampled Token.ID *is* the model's own
id for that label in this exact prompt frame. Those ids then index into
the logits of the real (ambiguous) reviews to compute a margin — how much
the model preferred "positive" over "negative", even when it sampled a
different token.

## Run

```sh
go run ./pkg/classify -model ~/models/gemma-4-e2b-it-4bit
```

Flags and behaviour are documented in [main.go](main.go) — the code is the example.
