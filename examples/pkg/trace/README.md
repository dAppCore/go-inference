<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# trace

Per-token phase timing: GenerateConfig.TraceTokenPhases asks the engine to
report where each generated token's wall time went — GPU-busy versus
host-serial (encode/sample/detokenise while the GPU sits idle). There is no
WithTraceTokenPhases option constructor yet; cmd/lem's own -trace flag sets
the field via an inline GenerateOption closure (go/decode/generate/generate.go),
so this example mirrors that exact wiring rather than inventing one.

The budget lands on m.Metrics().DecodePhases, and is nil whenever the active
engine/decode path did not report one — an untraced path, not an error.

## Run

```sh
go run ./pkg/trace -model ~/models/gemma-4-e2b-it-4bit
```

Flags and behaviour are documented in [main.go](main.go) — the code is the example.
