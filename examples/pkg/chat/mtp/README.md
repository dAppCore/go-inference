<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# chat/mtp

MTP speculative decoding as a library caller: a Gemma 4 target paired with
its drafter runs draft -> verify -> accept forwards instead of one token
per forward. The pairing convention is a fixed suffix on the target's own
name — target "gemma-4-<size>-it" pairs with drafter
"gemma-4-<size>-it-assistant" (or a quantised variant such as
"...-assistant-bf16"); LoadSpeculativePair takes the two directories
directly rather than auto-detecting the pair (that ladder is serve/generate's
job, not the library seam).

The speculative lane only engages fully greedy: at temperature 0 the verify
is exact (byte-identical to plain decode, just faster), so this example
pins temperature, top_p and top_k all to 0 rather than relying on whatever
defaults the checkpoint declares.

darwin/arm64 only — this file imports engine/metal directly
(LoadSpeculativePair), the same call serve/generate wire in through
serving.SpeculativeLoader.

## Run

```sh
go run ./pkg/chat/mtp -model ~/models/gemma-4-e2b-it-bf16 -draft ~/models/gemma-4-e2b-it-assistant-bf16
```

Flags and behaviour are documented in [main.go](main.go) — the code is the example.
