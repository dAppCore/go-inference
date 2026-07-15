<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# info

What loaded: architecture metadata (Info, ModelType) plus the live
capability probes — VisionModel/AudioModel are per-CHECKPOINT type
assertions (a vision-capable family can still ship a text-only snapshot),
and CapabilitiesOf reports the model's full feature surface when it
implements CapabilityReporter, as the engine's TextModel does.

## Run

```sh
go run ./pkg/info -model ~/models/gemma-4-e2b-it-4bit
```

Flags and behaviour are documented in [main.go](main.go) — the code is the example.
