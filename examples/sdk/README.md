# examples/sdk — gemma4 from any language, via the OpenAPI standard

`lem spec` exports go-inference's HTTP surface as an OpenAPI 3.1 document, and
[openapi-generator](https://openapi-generator.tech/docs/generators) turns that
document into a typed client in **any of its ~90 languages** — SDKs for free,
no hand-written client code. These four examples are the proof: the same
`/v1/chat/completions` call, one generated client per language, against a
local `lem serve`.

Tuned for **gemma4 / Lemma** — the engine this repo optimises (thinking
channel split into a typed `thought` field, MTP, prompt reuse). Other models
serve through the same API; results vary, it's OSS.

## Once: generate the clients

```bash
task sdk          # spec → build/sdk/openapi.json → build/sdk/{go,python,rust,typescript,c}
```

(Needs `openapi-generator-cli` + a JRE — `brew install openapi-generator openjdk`.
Add a language by dropping a `sdk-config/<lang>.yaml`.)

## Once: start a serve

```bash
task build && ./bin/lem serve --model ~/.cache/huggingface/hub/models--mlx-community--gemma-4-E2B-it-qat-4bit/snapshots/<hash>
# or from the TUI: lem tui → Service tab → enter
```

Every example reads `LEM_BASE_URL` (default `http://localhost:36911`).

## Run

| Language | Directory | Run |
|----------|-----------|-----|
| Go | [`go/`](go/) | `go run .` |
| Python | [`python/`](python/) | `pip install ../../build/sdk/python && python3 main.py` |
| TypeScript | [`typescript/`](typescript/) | `npm install && npm start` |
| Rust | [`rust/`](rust/) | `cargo run` |
| C | [`c/`](c/) | see [`c/README.md`](c/README.md) — needs a post-generation patch, generator bugs documented in its Friction section |

Each lists the served models, then asks gemma4 one question through the typed
client and prints the answer with token usage. The request's `model` field is
cosmetic on a single-model serve — the loaded model answers every name. The C
example goes further (two-turn memory, thinking-channel probe) — see its README.
