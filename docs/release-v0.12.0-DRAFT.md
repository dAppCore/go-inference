# lem v0.12.0 — first release

> One binary. Pure Go, zero cgo. Gemma 4 on Apple silicon, at full speed,
> with conversations that never re-read themselves.

`lem` — the **Lethean Ethical Machine** — is the Lethean inference engine
for the Gemma 4 family on Apple silicon: a single no-cgo Go binary driving
Metal directly, serving OpenAI-, Anthropic- and Ollama-compatible routes
on port 11434. It trains LoRA adapters, runs speculative decode the
reference engines don't have for this family, holds 256K context by
default on hardware that can carry it, and — the part nothing else does —
sleeps and wakes conversations byte-identically, so a session's history
is never re-processed.

## The name

The initials are a lineage, not a coincidence:

| rung | is |
|---|---|
| **LEK** — Lethean Ethical Kernel | the framework, trained into the weights |
| **LEM** — Lethean Ethical Model | a model graded trusted under it — safe, courteous, aware of its own cognition |
| **lem** — Lethean Ethical Machine | the runtime that carries them |

Kernel → Model → Machine: the ethics do not stop at the weights. The
machine is where the contracts execute — state continuity, audit traces,
refusal as a first-class output — a trusted model deserves a room built
to the same axioms.

*(And yes: Stanisław Lem, who wrote fables about machine minds decades
before anyone was ready to read them, gets the homage he was always
going to get. The name stays.)*

This is the first public release. (v0.11.0 was an internal preservation
tag, not a release.) Everything below has a receipt in the repository:
the benchmark harness, the test gates, and the numbers are reproducible
from the committed code.

## The headline: no-replay conversations

Every mainstream serving engine re-reads the conversation to answer the
next turn — the longer the chat, the more it re-processes, every single
turn. `lem serve -state-conversations` serialises the session instead:
turn N+1 resumes from the live KV state and pays only for the new tokens.

| receipt | replay engines | lem `-state` |
|---|---|---|
| turn 2 of a 10K-token conversation | 5.49s | **0.40s** |
| history re-processed per turn | the whole conversation | **none** |
| multi-turn book bench (10 chapters, full history resent) | prompt cost grows per turn | prompt column flat at ~88 tokens |

The work a replay engine repeats every turn is work this engine does not
do: compute that never runs, and — by construction — energy that is never
drawn. A measured joules-per-turn A/B (macOS `powermetrics`, replay vs
`-state`) is being prepared as a follow-up receipt; the mechanism itself
is visible in the wall-clock numbers above.

For agentic use — tool-calling loops where a session accumulates dozens
of turns — the effect compounds every round trip. This is the difference
a side-by-side video makes obvious.

## Decode: ahead of the reference engine on its best hardware

Gemma 4 E2B, 4-bit class, M3 Ultra, tg512, all defaults:

| engine | decode tok/s |
|---|---|
| llama.cpp (Q4_K_M, build 9860) | 144.0 |
| llama.cpp (Q8_0) | 124.0 |
| **lem, plain** | **161.3** (+12%) |
| **lem + MTP speculative** | **207.3** (+44%) |

MTP (Gemma 4's built-in speculative decoding — a drafter the target
verifies in one forward pass, same output quality) is on by default
(`-draft auto`) and no other engine currently runs it for this family.

Family board (decode tok/s, 4-bit, all defaults, plain / +MTP):

| model | plain | +MTP |
|---|---|---|
| E2B | 161.3 | 207.3 |
| E4B | 112.1 | — |
| 12B | 69.1 | 97.1 |
| 26B-A4B (MoE) | 139.8 | — |
| 31B | 32.5 | 47.4 |

## 256K context, fitted to the box

The default context follows the checkpoint window up to 256K, then a
RAM-aware guard budgets the box (weights + per-row KV cost against
physical memory) and clamps only when the machine cannot carry it —
explicit `-context` always wins, `LTHN_CONTEXT_RAM_GUARD=0` disables.

Receipt: a 31B, 235,000-token cold ingest on a 96GB M3 Ultra — 53.96GB
peak footprint, memory pressure green throughout, swap lower at the end
than the start, and a correct answer to a needle question over the full
context. Three days earlier the same run needed 64.9GB + 19.5GB of swap
and had to be killed; the difference is five memory-lifecycle fixes in
this release, each with a regression gate that fails if it returns.

KV is quantised q8 by default on qualifying layers (int8 codes + group
scales) at parity with bf16 on every axis we measure — decode to 124K
depth, prefill, MTP, and byte-identical `-state` snapshots.
`LTHN_KV_Q8_ICB=0` restores bf16.

## Flash attention where none existed

Gemma 4's global attention runs head dims 256 and 512 — above the 128
ceiling of every shipped fused-attention kernel on Metal (MLX's steel
path included; mlx-lm materialises attention scores for this family, as
does our own fallback composition). This release ships:

- **BD-256 steel flash attention** — MLX's template instantiated at the
  head dim nobody ships, one dispatch per chunk-layer, no score matrix
  in device memory. Token-identical to the composition, up to +4% at
  depth, and the up-to-2GB score scratch is never allocated.
- **Sliding-window flash** — Gemma 4's majority layers (30/35 on E2B,
  50/60 on 31B) window their attention; each query tile now streams its
  own fixed key span once through threadgroup tiles instead of re-reading
  the window per query row. +4% prefill at every depth measured.

Prefill, E2B class, against the field (tok/s at prompt depth):

| depth | lem | llama.cpp | mlx-lm |
|---|---|---|---|
| ~512 | 3,157¹ | 4,437 | — |
| 8K | 3,157 | 3,776 | ~9,850 |
| 32K | 2,769 | 2,412 | — |
| 62K | 2,255 | — | — |
| 118K | 1,656 | — | — |

¹ measured at 8K; the shallow-depth column carries the same configuration.

Honest reading: mlx-lm leads cold ingest at shallow depth by ~3× — that
gap is characterised in-repo (the whole-prompt dispatch structure) and is
open work. Two things frame it: past ~30K *we* lead llama.cpp and the
curve is flatter than both; and under `-state` serving, prefill is a cost
paid once per conversation, while the per-turn economics above are paid
by every engine on every turn — that trade is the design.

## Sovereignty

- **EUPL-1.2**, copyleft, asset-locked to Lethean CIC — this engine
  cannot be enclosed.
- **Zero cgo**: pure Go over Metal via runtime dispatch. One binary.
- **`task build:embed`** produces a fully self-contained binary with the
  Metal libraries baked in — runs from any path, air-gapped, no runtime
  environment.
- Requires macOS 26 (Tahoe) or newer on Apple silicon.

## Falsified and kept

This project publishes its negative results with the same care as its
wins — the kernels below are in-tree, gated off by default, each with the
receipt for why:

- **q8-reading flash attention**: parity-correct, but the in-loader
  dequant re-processes the prefix per query tile — O(N²) where the
  mirror lane is O(N). 551 vs 2,171 tok/s at 62K. Opt-in
  (`LTHN_FLASH_Q8=1`); the once-per-chunk staging shape is noted for a
  future pass.
- **Split-D 512 flash**: wins in a band (+1.5% at 32K), loses ≥42% at
  extreme depth as its serial tile loop compounds. Opt-in
  (`LTHN_FLASH_512=1`) until band-gated.

If a claim in these notes has no number beside it, treat it as a bug and
file it.

## Next

- **Measured joules-per-turn** — the `powermetrics` replay-vs-`-state` A/B
  that turns the no-replay wall-clock numbers into an energy receipt.
- **Model-side end-conversation** — a welfare surface for Lemma models
  served on the machine: abusive sessions route to mediation and the
  model may end them, the same courtesy Anthropic's Claude carries. A
  trusted model gets the right to say "no thanks, come back in a better
  mood." (Lemma models only — other models, other rules.)
- **Cold-ingest prefill** — the shallow-depth gap against mlx-lm is
  characterised in-repo; the campaign continues.

## Getting it

```sh
git clone https://github.com/dAppCore/go-inference
cd go-inference
task metallib && task build:embed   # self-contained bin/lem
bin/lem serve --model <gemma-4 snapshot>   # port 11434, drop-in Ollama routes
```

Prebuilt binaries (macOS 26+, Apple silicon) attached to this release.

---

### Release-asset note (not for publication): the side-by-side video

One agentic task, two panes, same prompt, same model class:

- **Left**: lem `serve -state-conversations`. **Right**: a replay engine
  (ollama or llama-server).
- The task: a tool-calling loop — agent reads a file, greps, edits,
  re-tests; ~10-15 turns of accumulating context.
- Overlay per pane: turn number, time-to-first-token this turn, running
  total of tokens re-processed.
- The shape the viewer watches: the right pane's TTFT grows every turn
  as the conversation lengthens; the left pane stays flat at
  decode-only. By turn 10 the gap is not a benchmark, it is visible
  waiting.
- End card: total wall time, total tokens re-processed (right: hundreds
  of thousands; left: the conversation length, once), and the one-line
  mechanism: "the engine that never re-reads the conversation."
