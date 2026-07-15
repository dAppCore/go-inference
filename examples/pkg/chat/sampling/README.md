<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# chat/sampling

Sampling control: Temperature/TopK/TopP/MinP shape the distribution a
token is drawn from; WithSeed pins the draw itself. Same seed plus the
same sampling settings on the same prompt reproduces the same completion —
this example runs it twice and compares.

## Run

```sh
go run ./pkg/chat/sampling -model ~/models/gemma-4-e2b-it-4bit
```

Flags and behaviour are documented in [main.go](main.go) — the code is the example.
