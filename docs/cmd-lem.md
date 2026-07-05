# The `lem` binary

`lem` is Lethean's sovereign inference binary. It hosts an
OpenAI/Anthropic/Ollama-compatible HTTP API for a local model and runs the
training and packaging verbs — and it compiles from **go-inference alone** (no
go-mlx, no go-rocm). Each subcommand is deliberately thin: flag parsing plus one
call into a go-inference library package. The business logic lives in the
libraries (`serving`, `decode/generate`, `train`, `train/tune`, `model/pack`,
`model/modelmgmt`), not in `cmd/lem`.

Source: `go/cmd/lem/`. Build instructions: [build.md](build.md).

## Backends registered at compile time

`main.go` blank-imports two packages so their `init()` hooks register into the
inference registry before any verb runs:

- `dappco.re/go/inference/engine/metal` — the no-cgo Apple "metal" backend
  (darwin/arm64, dispatches Apple MLX's compiled kernels via the Objective-C
  runtime).
- `dappco.re/go/inference/model/builtin` — the built-in architectures
  (gemma3, gemma4, mistral, qwen3).

The invoked binary name is taken from `argv[0]`, so a renamed copy of the binary
prints its own name in usage and notices (the dev binary is often built as
`lthn-mlx`).

## Verbs at a glance

| Verb | What it does | Library |
|------|--------------|---------|
| `serve` | Host the OpenAI/Anthropic/Ollama HTTP API for a loaded model | `serving.RunServe` |
| `generate` | One-shot generate + decode-only tok/s (no HTTP; like-for-like bench) | `decode/generate.RunGenerate` |
| `ssd` | Self-distillation sampling: sample the frozen base, capture the trace | `train.RunSSDCommand` |
| `sft` | LoRA supervised fine-tuning through the engine trainer seam | `train.RunSFTCommand` |
| `tune` | Measure + persist the best MTP draft block as a serve profile | `train/tune.RunTune` |
| `pack` | Build/inspect/list/extract/hash `.model` containers (no weights loaded) | `model/pack` |
| `ebook` | Render a model directory as a valid EPUB3 (weights as base64 plates) | `model/modelmgmt.BuildModelBook` |

Run `lem <verb> -h` for the command-specific flag dump. Boot notices and errors
go to stderr; generation output goes to stdout.

Default runtime paths live under `~/Lethean/data/` (admin token, tuning
profiles, conversation state) — see each verb below.

---

## `serve`

Hosts an OpenAI / Anthropic / Ollama-compatible HTTP API for a model.

```
lem serve --model ~/models/gemma-4-e2b-it-4bit               # OpenAI HTTP on :36911
lem serve --model ~/models/gemma-4-e2b-it-4bit --context 8192
lem serve                                                    # start model-less, load later via admin reload
```

The default port **36911** is Lethean's own, so an Ollama install on 11434 never
collides. Point any OpenAI or Ollama client at `http://localhost:36911`.

### Inference routes

| Route | API |
|-------|-----|
| `POST /v1/chat/completions` | OpenAI chat (streaming + non-streaming) |
| `POST /v1/messages` | Anthropic Messages |
| `POST /api/chat` | Ollama chat |
| `GET /v1/models` | list loaded models |
| `GET /v1/health` | process health probe |

### Admin control plane

The `/v1/admin/*` subtree (machine identity, serve status, hot-swap reload) sits
behind a Bearer wall. The admin token is stored mode `0600` at
`~/Lethean/data/admin.token`. Serve is **fail-closed**: if the token file cannot
be written it refuses to boot rather than binding a listener with an unprotected
admin surface. `POST /v1/admin/serve/reload` hot-swaps the loaded model (and
re-runs the reactive drafter ladder over the new target).

### Flags

| Flag | Default | Meaning |
|------|---------|---------|
| `--addr` | `:36911` | listen address (Lethean's own port) |
| `--model` | `""` | model to load; empty starts the driver model-less (load later via admin reload) |
| `--context` | `0` | override context length; 0 uses the model default |
| `--kv-cache` | `""` | KV cache mode: `paged`, `fp16`, `q8`, `kq8vq4`, `turboquant`; empty loads the model default |
| `--draft` | `auto` | MTP drafter: `auto` detects one beside a Gemma 4 target, a path forces it, `""` disables |
| `--draft-detect` | `true` | reactive drafter detection for Gemma 4 targets |
| `--draft-block` | `0` | MTP draft block; 0 = engine default (5), a tuned profile overrides when present |
| `--no-auto-profile` | `false` | ignore tuned profiles from `lem tune` |
| `--profile-dir` | `""` | tuned-profile directory (default `~/Lethean/data/tuning`) |
| `--state-conversations` | `true` | conversation continuity: wake each chat from its slept state, append only the new turn, no prompt replay |
| `--state-store` | `""` | conversation state store file (default `~/Lethean/data/state/conversations.kv`) |
| `--native` | `false` | serve via the no-cgo native token-loop contract (the default metal engine already is native) |
| `--read-timeout` | `30s` | HTTP read-header timeout |
| `--write-timeout` | `5m` | HTTP write timeout (covers a full streaming response) |
| `--shutdown-timeout` | `10s` | graceful-shutdown deadline after SIGINT/SIGTERM |
| `--print-admin-token` | `false` | print the admin Bearer token and exit (generates if absent) |
| `--rotate-admin-token` | `false` | regenerate the admin Bearer token, print it, and exit |

The two token-management flags are handled before the `--model` check, so an
operator can reveal or rotate the token without a model loaded.

Reactive drafting degrades gracefully: a detected drafter is only armed when the
registered engine exposes a speculative loader; otherwise serve prints an honest
notice and serves plain autoregressive. Conversation continuity likewise
degrades to stateless (with a notice) if the engine exposes no continuity attach
or the state store cannot open — neither ever blocks the serve from coming up.

---

## `generate`

Loads a model and generates from a prompt with **no HTTP serve in the path**,
reporting decode-only tok/s (prefill excluded) for like-for-like benching
against `llama-bench` and friends. Takes exactly one positional model path.

```
lem generate ~/models/gemma-4-e2b-it-4bit
    # one-shot generate + decode tok/s

lem generate -state chat1 -prompt "Hello, who are you?" ~/models/gemma-4-e2b-it-4bit
    # a durable conversation turn (wake -> generate -> sleep)
```

### Flags

| Flag | Default | Meaning |
|------|---------|---------|
| `-prompt` | (a Go linked-list prompt) | user prompt |
| `-max-tokens` | `128` | tokens to generate |
| `-temp` | `1.0` | sampling temperature (0 = greedy/argmax — fastest, fair vs `llama-bench`) |
| `-think` | `false` | enable the thinking channel (off keeps the decode rate clean) |
| `-context` | `0` | context length override (0 = model default) |
| `-kv-cache` | `""` | KV cache mode (`paged`, `fp16`, `q8`, `kq8vq4`, `turboquant`; empty = load default) |
| `-kv-storage` | `""` | retained KV storage dtype (`fp16`, `bf16`; empty = native fp32) |
| `-draft` | `auto` | MTP drafter (as for `serve`) |
| `-draft-block` | `0` | MTP draft block; 0 = engine default (5) |
| `-pipeline` | `true` | one-ahead pipelined decode (false forces the serial loop, for A/B traces) |
| `-native` | `false` | generate via the no-cgo native token-loop contract |
| `-trace` | `false` | print the per-token decode time budget — GPU wait vs host-serial work |
| `-state` | `""` | conversation state name: wake it from the store, generate, sleep it back — the no-prompt-replay turn loop |
| `-state-store` | `""` | state store file (default `~/Lethean/data/state/agent.kv`) |
| `-raw` | `false` | with `-state`: skip chat-framing and run the raw completion-loop turn (ignored without `-state`) |
| `-image` | (repeatable) | image input for a vision model: a local PNG/JPEG path or a base64 `data:` URL; gated on the model's vision capability |
| `-audio` | (repeatable) | reserved — no engine-neutral audio-input seam yet, so passing one errors (follow-up) |

---

## `ssd` — self-distillation sampling

Samples the **frozen** base model over a set of prompts, captures each
self-generated output at birth, and **stops at the trace**. Nothing is taught —
there is no reference answer, no verifier, no training in this verb. The lab
refines the trace into an SFT artifact; a separate `sft` run trains on it.

`--model` and `--data` are required.

```
lem ssd --model ~/models/gemma-4-E2B-it-bf16 --data prompts.jsonl \
        --checkpoint-dir ~/Lethean/data/ssd/run1
```

`--data` is a prompt JSONL — `{"messages":[…]}` or `{"prompt":…}` per line; only
the prompts are read, the responses are self-generated. `--kernel` supplies a
LEK-2 kernel prefix that rides every generation as KV state but never enters the
captured rows (#97).

### Flags

| Flag | Default | Meaning |
|------|---------|---------|
| `-model` | (required) | frozen base model path to self-distil |
| `-data` | (required) | prompt JSONL (only prompts are read) |
| `-kernel` | `""` | file holding the LEK-2 kernel prefix (rides as KV state, never captured) |
| `-sample-max-tokens` | `256` | tokens per self-generated sample |
| `-sample-temp` | `0.7` | sampling temperature (must be ≠ 1.0 — diversity is the point) |
| `-sample-top-k` | `64` | sampling top-k |
| `-sample-top-p` | `0.95` | sampling top-p |
| `-sample-min-p` | `0` | sampling min-p |
| `-rep-penalty` | `1.0` | repetition penalty over self-samples |
| `-filter-shortest` | `10` | drop the shortest N% of self-samples before the trace (0 keeps all) |
| `-score-samples` | `false` | score every self-sample at birth — needs a scorer wired into go-inference (none yet, so this is an honest no-op) |
| `-checkpoint-dir` | `""` | output dir for the scored trace (`ssd-captures.jsonl`) |
| `-context` | `0` | model context override; 0 uses the model default |

---

## `sft` — LoRA supervised fine-tuning

Native LoRA SFT through the engine-neutral trainer seam: the loaded engine opens
a **head-LoRA trainer**, the loop steps it over the training set, checkpoints and
evaluates on a fixed probe set, and saves a reloadable adapter package. Apply the
adapter at load with `serve`/`generate --adapter`.

`--model` and `--data` are required.

```
lem sft --model ~/models/gemma-4-E2B-it-bf16 \
    --data train.jsonl --valid valid.jsonl \
    --rank 16 --epochs 2 --checkpoint-dir ~/Lethean/data/sft/run1
```

### Flags

| Flag | Default | Meaning |
|------|---------|---------|
| `-model` | (required) | model path to fine-tune |
| `-data` | (required) | training JSONL — `{"messages":[{role,content}…]}` per line |
| `-valid` | `""` | validation JSONL; derives eval probes from its first user turns when `-eval-prompts` is absent |
| `-eval-prompts` | `""` | file of eval probes, one per line (overrides `-valid` derivation) |
| `-eval-every` | `25` | run the eval probes every N optimiser steps (0 disables eval) |
| `-eval-max-tokens` | `200` | tokens per eval generation |
| `-eval-probes` | `4` | probes derived from `-valid` when `-eval-prompts` is absent |
| `-eval-temp` | `0` | eval sampling temperature (0 = greedy) |
| `-score-cascade` | `false` | score every eval pass — needs a scorer wired in (none yet; notes honestly and captures only) |
| `-score-window` | `3` | eval passes per windowed composite |
| `-rank` | `16` | LoRA rank |
| `-alpha` | `32` | LoRA alpha |
| `-lr` | `1e-4` | AdamW learning rate |
| `-epochs` | `1` | training epochs |
| `-batch` | `1` | batch size |
| `-grad-accum` | `4` | gradient accumulation steps |
| `-max-seq` | `1024` | max sequence length (longer samples truncate) |
| `-packing` | `false` | sequence packing (no effect on the head-LoRA trainer; noted honestly) |
| `-checkpoint-dir` | `""` | checkpoint directory |
| `-checkpoint-every` | `50` | save a checkpoint every N optimiser steps (0 disables) |
| `-save` | `""` | final adapter path (default `<checkpoint-dir>/adapter` when a dir is set) |
| `-resume` | `""` | resume from a saved adapter checkpoint |
| `-merge` | `false` | merge the adapter into the weights after training (unsupported on head-LoRA; noted honestly) |
| `-context` | `0` | model context override; 0 uses the model default |

---

## `tune` — MTP draft-block profile

Measures plain autoregressive decode against each candidate MTP draft block on
the real model, then persists the winner as a tuning profile that `serve`
auto-applies. `--model` is required.

```
lem tune --model ~/models/gemma-4-e2b-it-4bit --depths 4,5,6
```

**Current status:** the block sweep needs a speculative-pair loader that no
registered go-inference engine exposes yet, so `tune` currently detects the
drafter and reports the plan **without measuring** (it lights up when the engine
seam lands). It reports this honestly rather than faking a measurement.

### Flags

| Flag | Default | Meaning |
|------|---------|---------|
| `-model` | (required) | Gemma 4 target model path |
| `-draft` | `auto` | MTP drafter: `auto` detects one beside the target, a path forces it |
| `-depths` | `4,5,6` | comma-separated draft blocks to sweep |
| `-max-tokens` | `256` | tokens per measurement run |
| `-prompt` | (a Go linked-list prompt) | measurement prompt |
| `-workload` | `chat` | workload the profile is scored + persisted under |
| `-profile-dir` | `""` | tuned-profile directory (default `~/Lethean/data/tuning`) |
| `-json` | `false` | emit JSONL tuning events instead of the text summary |

---

## `pack` — `.model` containers

Builds and reads `.model` containers — the Trix container with magic `"MDL1"`
(`[Magic "MDL1"][Version][Header Length][JSON Header][Payload]`) — **without
loading weights or touching an engine**. Each subcommand is flag parsing plus one
library call.

```
lem pack create ~/models/gemma-4-e2b-it-4bit gemma.model -arch gemma4 -quant 4
lem pack inspect gemma.model
```

### Subcommands

| Subcommand | Synopsis | Notes |
|------------|----------|-------|
| `create` | `<src-dir> <out.model>` | pack a directory into a `.model` container (deterministic tar payload + manifest header) |
| `inspect` | `<file.model>` | print the container manifest (no extraction) |
| `list` | `<file.model>` | list the payload entries (path + size) |
| `extract` | `<file.model> <dest-dir>` | unpack the container back to a directory |
| `hash` | `<src-dir>` | print the canonical model-pack hash of a directory |

`create` flags: `-arch` (architecture id in the manifest), `-quant` (quant bits),
`-source-format` (`safetensors` — default — or `gguf`), `-producer` (default
`lem`). Model identity comes from the flags; no directory scan populates it.
`inspect` and `list` take `-json`; `extract` takes `-overwrite` (refuses a
non-empty destination otherwise). `hash` reads metadata files and safetensors
sizes only — it does not read tensor bytes — and prints the same value `create`
embeds as `Manifest.Model.Hash`.

---

## `ebook` — render a model as an EPUB3

Renders a model directory into a valid EPUB3: the authored foreword (the model's
`README` — the human-speech anchor), a method section, and — by default — the
weights as base64 plates that decode back into a runnable model. This is the PGP
playbook applied to weights: a published, authored book carries the protection of
speech. Pure file I/O — no model is loaded, so it is engine-neutral.

```
lem ebook --model ~/Code/lthn/LEM-Gemma3-1B --out LEM-Gemma3-1B.epub
lem ebook --model <dir> --weights=false     # the readable manifesto, no plates
```

### Flags

| Flag | Default | Meaning |
|------|---------|---------|
| `-model` | (required) | model directory to render |
| `-out` | `<model-name>.epub` | output `.epub` path |
| `-title` | (model dir name) | book title |
| `-author` | `Lethean` | book author — the publishing voice that makes it authored speech |
| `-foreword` | `<model>/README.md` | foreword text file |
| `-weights` | `true` | include the weights as base64 plates; false = manifesto + method only |
| `-chapter-chars` | `0` | base64 characters per weight plate (0 = default 4,000,000) |

On success it reports the output path, chapter count (and how many are in the
table of contents), and the EPUB size in bytes.
