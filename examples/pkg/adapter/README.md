<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# adapter

Loading a LoRA adapter: WithAdapterPath injects an adapter at load time
without fusing it into the base weights — the directory needs
adapter_config.json plus adapter safetensors files, exactly what
pkg/train/sft's -out/adapter directory writes. Train there first, then
point -adapter at its output to see the fine-tune applied at inference.

## Run

```sh
go run ./pkg/train/sft -model ~/models/gemma-4-E2B-it-bf16 -out ./sft-out
go run ./pkg/adapter -model ~/models/gemma-4-E2B-it-bf16 -adapter ./sft-out/adapter
```

Flags and behaviour are documented in [main.go](main.go) — the code is the example.
