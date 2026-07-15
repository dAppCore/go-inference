<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# state

Durable conversation state — the no-prompt-replay loop: `lem generate -state
<name>` wakes a named slot from disk (KV restored in ~1-3ms), appends ONLY
the new turn, generates, then sleeps the session back for the next call.
This example calls the same library seam cmd/lem's -state flag delegates to
(dappco.re/go/inference/decode/generate) twice against one state name,
proving the second turn answers from the first turn's content without
resending it.

## Run

```sh
go run ./pkg/state -model ~/models/gemma-4-e2b-it-4bit
```

Flags and behaviour are documented in [main.go](main.go) — the code is the example.
