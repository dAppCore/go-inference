<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# rwkv7-generate

Library-level generation for the RWKV-7 "Goose" host port: a recurrent SSM
family (token-shift + WKV7 time-mix + channel-mix, no attention, no KV
cache — a fixed-size carried state per layer) served straight from
`model/arch/rwkv7`, bypassing `inference.LoadModel` because the native
engine's load hook for `model_type: rwkv7` is not wired yet (see the
package's own doc comments). This is the same path
`real_checkpoint_test.go` gates against a numpy oracle transcription of the
upstream reference (github.com/fla-org/flash-linear-attention).

## Run

```sh
python3 -c "from huggingface_hub import snapshot_download; print(snapshot_download('RWKV/RWKV7-Goose-World2.8-0.1B-HF'))"
go run ./pkg/rwkv7-generate -model <the printed snapshot dir> -prompt "The capital of France is"
```

The RWKV World tokenizer vocab defaults to the fixture this repo ships
(`go/model/arch/rwkv7/testdata/rwkv_vocab_v20230424.hex`) — every released
RWKV-7 checkpoint (0.1B/1.5B/2.9B) shares the same vocabulary, so no
per-checkpoint tokenizer file is needed. Flags and behaviour are documented
in [main.go](main.go) — the code is the example.
