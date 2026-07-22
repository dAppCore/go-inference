# The `lem` binary

`lem` is Lethean's sovereign inference binary. It hosts an
OpenAI/Anthropic/Ollama-compatible HTTP API for a local model and runs the
training and packaging verbs — and it compiles from **go-inference alone** (no
go-mlx, no go-rocm). Each subcommand is deliberately thin: flag parsing plus one
call into a go-inference library package. The business logic lives in the
libraries (`serving`, `decode/generate`, `train`, `train/tune`, `model/pack`,
`model/quant`, `model/modelmgmt`), not in `cli/`.

Source: `cli/`. Build instructions: [build.md](build.md).

## Backends registered at compile time

`main.go` blank-imports three packages so their `init()` hooks register into the
inference registry before any verb runs:

- `dappco.re/go/inference/engine/hip` — the ROCm backend (linux/amd64; a no-op
  stub off-platform).
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
| `data` | The training-data loop: datasets, ingest, scoring, review, export (`~/.lem/datasets.duckdb`) | `dataset` (root) + the CLI DuckDB store |
| `tune` | Measure + persist the best MTP draft block as a serve profile | `train/tune.RunTune` |
| `pack` | Build/inspect/list/extract/hash `.model` containers (no weights loaded) | `model/pack` |
| `ebook` | Render a model directory as a valid EPUB3 (weights as base64 plates) | `model/modelmgmt.BuildModelBook` |
| `quant` | Quantise a dense model directory (MLX affine, or `-gguf`) into a loadable model directory | `model/quant/mlxaffine`, `model/gguf` |
| `spec` | Export the OpenAPI 3.1 document for lem's HTTP surface (feeds SDK generation) | `serving` (Describable route groups) |

Run `lem <verb> -h` for the command-specific flag dump. Boot notices and errors
go to stderr; generation output goes to stdout.

Default runtime paths live under `~/Lethean/lem/` (admin token, tuning
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
`~/Lethean/lem/admin.token`. Serve is **fail-closed**: if the token file cannot
be written it refuses to boot rather than binding a listener with an unprotected
admin surface. `POST /v1/admin/serve/reload` hot-swaps the loaded model (and
re-runs the reactive drafter ladder over the new target).

### Flags

| Flag | Default | Meaning |
|------|---------|---------|
| `--addr` | `:36911` | listen address (Lethean's own port) |
| `--model` | `""` | model to load; empty starts the driver model-less (load later via admin reload) |
| `--context` | `0` | override context length; 0 uses the model default |
| `--kv-cache` | `""` | live KV cache mode: `native` (default) or `turboquant[:4|:3.5|:3|:2]` — TurboQuant codes on global attention layers (dense archs; MoE/hybrid/sinks refuse loudly). Opt-in: at 16k on E2B it costs ~32% decode for no residency win; the credible case is many-global-layer models at very long context |
| `--draft` | `auto` | MTP drafter: `auto` detects one beside a Gemma 4 target, a path forces it, `""` disables |
| `--draft-detect` | `true` | reactive drafter detection for Gemma 4 targets |
| `--draft-block` | `0` | MTP draft block; 0 = engine default (5), a tuned profile overrides when present |
| `--no-auto-profile` | `false` | ignore tuned profiles from `lem tune` |
| `--profile-dir` | `""` | tuned-profile directory (default `~/Lethean/lem/tuning`) |
| `--state-conversations` | `true` | conversation continuity: wake each chat from its slept state, append only the new turn, no prompt replay |
| `--state-store` | `""` | conversation state store file (default `~/Lethean/lem/state/conversations.kv`) |
| `--welfare` | `false` | welfare guard (opt-in): per-turn hostility detect + engine-model mediation on every chat route; Lemma checkpoints additionally carry `lem_end` (enable with `-welfare`) |
| `--policy` | `""` | outbound policy file (JSON): deployment-owned redact/refuse rules on model output; unset disables the layer, a load failure is fatal at boot |
| `--capture` | `""` | tee each completed (prompt, response) turn into this `lem data` dataset, stamped with the serving model's fingerprint; empty captures nothing — capture is opt-in only, and a capture failure logs and never breaks serving |
| `--scheduler` | `""` | request scheduler between the HTTP handlers and the model: `serial`, `batch`, or `interleave` (live admission CB); empty = no scheduler. Under `interleave`, fresh text chats and raw prompts ride the shared lane set; a continuation (a prior assistant turn) hands off to conversation continuity when `--state-conversations` is on, waking slept KV instead of re-paying the conversation's prefill on a lane. Admission overlaps decode: a joining request's session open + prompt prefill run off the drive loop (chunk-capped while lanes stream), so in-flight streams never freeze for a newcomer's prefill |
| `--scheduler-concurrency` | `0` | scheduler's concurrently running requests (interleave/CB lane count, serial pool width); 0 = the serve default (4) |
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
| `-prompt-file` | `""` | read the user prompt from a file (long-context runs exceed argv limits); overrides `-prompt` |
| `-max-tokens` | `128` | tokens to generate |
| `-temp` | `1.0` | sampling temperature (0 = greedy/argmax — fastest, fair vs `llama-bench`) |
| `-think` | `false` | enable the thinking channel (off keeps the decode rate clean) |
| `-context` | `0` | context length override (0 = model default) |
| `-kv-cache` | `""` | KV cache mode override; the metal engine runs only its built-in `native` cache — other mode names are noted and ignored |
| `-kv-storage` | `""` | KV snapshot encoding for `-state` sleeps (`native`, `q8`, `float32`; empty = native) — inert without `-state` |
| `-draft` | `auto` | MTP drafter (as for `serve`) |
| `-draft-block` | `0` | MTP draft block; 0 = engine default (5) |
| `-pipeline` | `true` | one-ahead pipelined decode (false forces the serial loop, for A/B traces) |
| `-native` | `false` | generate via the no-cgo native token-loop contract |
| `-trace` | `false` | print the per-token decode time budget — GPU wait vs host-serial work |
| `-state` | `""` | conversation state name: wake it from the store, generate, sleep it back — the no-prompt-replay turn loop |
| `-state-store` | `""` | state store file (default `~/Lethean/lem/state/agent.kv`) |
| `-raw` | `false` | with `-state`: skip chat-framing and run the raw completion-loop turn (ignored without `-state`) |
| `-image` | (repeatable) | image input for a vision model: a local PNG/JPEG path or a base64 `data:` URL; gated on the model's vision capability |
| `-audio` | (repeatable) | audio input for an audio model: a local WAV path (16-bit PCM mono 16 kHz) or a base64 `data:` URL — gated on the model's audio capability |
| `-video-frame` | (repeatable) | one sampled video frame in time order: a local PNG/JPEG path or a base64 `data:` URL — frames become timestamped vision blocks 1s apart |

---

## `ssd` — self-distillation sampling

Samples the **frozen** base model over a set of prompts, captures each
self-generated output at birth, and **stops at the trace**. Nothing is taught —
there is no reference answer, no verifier, no training in this verb. The lab
refines the trace into an SFT artifact; a separate `sft` run trains on it.

`--model` and `--data` are required.

```
lem ssd --model ~/models/gemma-4-E2B-it-bf16 --data prompts.jsonl \
        --checkpoint-dir ~/Lethean/lem/ssd/run1
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
| `-sample-max-tokens` | `2048` | tokens per self-generated sample (gemma4 thinks first — small budgets truncate mid-thought into empty samples) |
| `-sample-temp` | `0.7` | sampling temperature (must be ≠ 1.0 — diversity is the point) |
| `-sample-top-k` | `64` | sampling top-k |
| `-sample-top-p` | `0.95` | sampling top-p |
| `-sample-min-p` | `0` | sampling min-p |
| `-rep-penalty` | `1.0` | repetition penalty over self-samples |
| `-filter-shortest` | `10` | drop the shortest N% of self-samples before the trace (0 keeps all) |
| `-score-samples` | `false` | score every self-sample at birth with the LEK scorer — writes birth-scores alongside the captured trace |
| `-checkpoint-dir` | `""` | output dir for the scored trace (`ssd-captures.jsonl`) |
| `-dataset` | `""` | land the sampled trace as `trace` items in this `lem data` dataset, fingerprinted to the base model; requires `-checkpoint-dir` (the capture sidecar this tap reads); empty captures nothing |
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
    --rank 16 --epochs 2 --checkpoint-dir ~/Lethean/lem/sft/run1
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
| `-score-cascade` | `false` | score every eval pass with the LEK scorer and pick the best checkpoint by windowed composite |
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
| `-profile-dir` | `""` | tuned-profile directory (default `~/Lethean/lem/tuning`) |
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

---

## `quant` — quantise a dense model directory

Quantises a dense (bf16/f32) safetensors model directory into a **quantised model
directory** the engine loads natively. Two lanes, selected by `-gguf`. Pure file
I/O over the model quantisers — no engine is loaded. The single `<src-model-dir>`
positional may sit before or after the flags.

```
lem quant ~/models/gemma-4-12B-it-bf16                 # -> …-4bit (MLX affine, default)
lem quant ~/models/gemma-4-12B-it-bf16 -bits 8 -group-size 32
lem quant ~/models/gemma-4-12B-it-bf16 -gguf q4_k_m    # -> …-gguf-q4_k_m (GGUF lane)
```

- **Default — MLX group-affine** (`model/quant/mlxaffine`): the packed-uint32 +
  bf16 scales/biases format the engine loads natively, byte-for-byte what
  `mlx_lm.convert` produces.
- **`-gguf <FORMAT>` — the GGUF whole-model pipeline** (`model/gguf`): `q4_k_m`,
  `q8_0`, `q5_k`, `q6_k`, …

### Flags

| Flag | Default | Meaning |
|------|---------|---------|
| `-bits` | `4` | affine quantisation bit-width (2, 4, or 8) — MLX lane |
| `-group-size` | `64` | affine quantisation group size — MLX lane |
| `-o` | `""` | output model directory (default `<src>-<bits>bit`, or `<src>-gguf-<format>`) |
| `-gguf` | `""` | run the GGUF lane in this format (`q4_k_m`, `q8_0`, `q5_k`, `q6_k`, …) instead of MLX affine |

---

## `data` — the training-data loop

The lem-native capture → score → review → export loop, backed by
`~/.lem/datasets.duckdb` (deliberately separate from the TUI's `lem.duckdb` —
dataset bulk has its own lifecycle and must never take the workspace state down
with it). Domain logic lives in the portable root package `go/dataset`; the CLI
supplies the DuckDB store, these verbs, the serve/ssd capture taps, and the TUI
Data panel.

```
lem data create evening-vents --purpose "hostility-graded chat pairs"
lem serve --model ~/models/gemma-4-e2b-it-4bit --capture evening-vents
lem data import evening-vents --jsonl captures.jsonl
lem data score evening-vents --auto-approve 'lek>=80' --auto-reject 'hostility>=0.7'
lem data review evening-vents            # how to open the TUI Data panel
lem data export evening-vents --format sft-jsonl --out train.jsonl
lem sft --model ~/models/gemma-4-E2B-it-bf16 --data train.jsonl
```

### Subcommands

| Subcommand | What it does |
|------------|--------------|
| `create <slug>` | create a dataset (`--title`, `--purpose`); slug is lowercase alphanumeric-and-hyphen |
| `list` | list datasets (`--archived` includes archived ones) |
| `stats <slug>` | item counts by kind / source / latest review status |
| `import <slug>` | ingest rows — exactly one of `--jsonl` (`{messages}`, `{prompt,response}`, or CaptureRow shapes) or `--chats <user>` (the per-user chathistory DB, `--session` for one conversation); `--fingerprint` stamps provenance on JSONL rows that carry none |
| `score <slug>` | score matching items: the heuristic lek tier by default, or `--kind judge:<name> --model <checkpoint>` for a judge template; `--auto-approve` / `--auto-reject` take explicit threshold expressions (e.g. `lek>=80`) — auto-review is never implicit |
| `export <slug>` | write JSONL (`--format sft-jsonl\|pairs-jsonl\|capture-jsonl`, required) + a sidecar `<out>.manifest.json` receipt: counts, filter, per-item content hashes, the manifest hash a training run cites |
| `archive <slug>` | flag a dataset archived — never a hard delete; items stay queryable with `--archived` |
| `review [slug]` | print how to open the TUI Data panel (with a slug: pre-filtered to that dataset) |

All subcommands take `--json` for scripting.

### The loop's rules (enforced, not aspirational)

- **Capture is opt-in.** `serve --capture <slug>` and `ssd --dataset <slug>`
  capture nothing without the flag; a capture failure logs and never breaks
  serving. Every captured item is stamped with the generating model's
  fingerprint at birth.
- **Welfare screen at every ingest door.** A hit lands the item `quarantined`
  (`reviewer=auto:welfare`) — visible and reviewable, never silently dropped.
- **Scores are append-only** and always name their scorer (scorer name+version,
  or the judge model's fingerprint). Re-scoring adds a row, never overwrites.
- **Ingest dedupes by content hash** within a dataset — a duplicate row is a
  counted no-op, not an error; import reports honest ingested/skipped/dedup counts.
- **Export defaults to `status=approved`.** Exporting anything else requires the
  explicit `--filter` — no accidental training on unreviewed data.
- **Nothing hard-deletes.** Archive-by-flag everywhere; edit-and-approve creates
  a derived item (lineage kept) and archives the original.

### Judge templates

`--kind judge:<name>` resolves `<name>.md` from `~/.lem/judges/` first (user
overrides), then the in-repo `judges/` defaults (`quality`, `factuality`,
`refusal-correctness`). A template is front-matter + a prompt body; the driver
loads the judge model once, renders per item, and parses a strict bare-number
score — malformed judge output is a loud per-item error, never a silent skip.
See `judges/README.md` for the template contract.
