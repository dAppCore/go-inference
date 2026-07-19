<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# rwkv7-generate

Library-level generation for the RWKV-7 "Goose" host port: a recurrent SSM
family (token-shift + WKV7 time-mix + channel-mix, no attention, no KV
cache — a fixed-size carried state per layer) served straight from
`model/arch/rwkv7`, bypassing `inference.LoadModel`. This is the same path
`real_checkpoint_test.go` gates against a numpy oracle transcription of the
upstream reference (github.com/fla-org/flash-linear-attention).

The native engine's load hook for `model_type: rwkv7` IS wired
(`engine/metal`'s `LoadModel` routes it through the generic
`model.SessionModel` serve arm), so `inference.LoadModel` + `TextModel.Generate`
now also works unchanged for this checkpoint — see
[`../generate`](../generate) for that path. This example stays as the
low-level demonstration that `model/arch/rwkv7` works standalone, no engine
required.

## Run

```sh
python3 -c "from huggingface_hub import snapshot_download; print(snapshot_download('RWKV/RWKV7-Goose-World2.8-0.1B-HF'))"
go run ./pkg/rwkv7-generate -model <the printed snapshot dir> -prompt "The capital of France is"
```

The RWKV World tokenizer vocab defaults to the tokenizer's own embedded
canonical table (`rwkv7.NewWorldTokenizer()`, backed by
`go/model/arch/rwkv7/data/rwkv_vocab_v20230424.hex`) — every released RWKV-7
checkpoint (0.1B/1.5B/2.9B) shares the same vocabulary, so no per-checkpoint
tokenizer file is needed and no `-vocab` flag is required; pass `-vocab
<path>` only to point at an alternate on-disk fixture. Flags and behaviour
are documented in [main.go](main.go) — the code is the example.
