<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# benchmark

Getting the numbers out of go-inference: every completed Generate/Chat call
leaves a GenerateMetrics snapshot behind — token counts, prefill vs decode
split, throughput, peak GPU memory. This is the llama-bench "tg" shape:
one untimed warmup (load + JIT), then a measured N-token generation.

## Run

```sh
go run ./pkg/benchmark -model ~/models/gemma-4-e2b-it-4bit -tokens 256
```

Flags and behaviour are documented in [main.go](main.go) — the code is the example.
