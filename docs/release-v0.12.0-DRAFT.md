# lem v0.12.0 — first release

> One binary. Pure Go, zero cgo. Gemma 4 on Apple silicon, at full speed,
> with conversations that never re-read themselves.

`lem` — the **Lethean Ethical Machine** — is the Lethean inference engine
for the Gemma 4 family on Apple silicon: a single no-cgo Go binary driving
Metal directly, serving OpenAI-, Anthropic- and Ollama-compatible routes
on port 36911. It trains LoRA adapters, runs speculative decode the
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
do — and that is measurable at the wall. A `powermetrics` A/B — the same
10-turn conversation driven identically at three servers, energy
integrated over each arm net of the 1.02W idle baseline (M3 Ultra,
E2B/Q4 class, every engine at its shipping defaults; harness committed
at `docs/examples/energy-ab/`):

| engine, as shipped | 10 turns | per turn | Wh/turn | per-turn trend |
|---|---|---|---|---|
| **lem, continuity (default)** | **771 J** | **77.1 J** | **0.021** | flat ~77 J |
| llama-server (prefix cache active) | 1,121 J | 112.1 J | 0.031 | flat-ish, noisy |
| pure replay (no cache of any kind)² | 1,439 J | 143.9 J | 0.040 | climbing 83 → 233 J |

² Measured on lem with all reuse off — the shape every cache-less replay
deployment pays. lem's stateless mode does not ship like this: see the
prompt-reuse note below the table.

**Identical task, 54% of replay's energy and 69% of llama-server's — at
ten turns, on the smallest model.** Pure replay's per-turn cost climbs
~15 J every turn as the history grows, so the saving widens with every
turn of every session. llama-server's in-memory prefix cache does help
it (its trend stays flat) — but it is a single-slot volatile cache,
lost on restart or slot eviction; lem's state is durable and
per-conversation. Footnotes for the sceptical: llama-server produced
slightly shorter chapters (3,223 words vs lem's 3,364), so per-word the
gap is wider still (0.23 vs 0.35 J/word); lem's defaults include MTP,
llama's include its cache — engines compared as they ship.

**And stateless mode is not the punching bag.** Even with `-state` off,
lem ships llama-parity automatic prompt reuse: a resident session keeps
the conversation's KV rows and each request re-prefills only what
diverged, so the stateless per-turn wall is flat too (turn 10 of the
same bench: 4.4s → 3.1s, level with turn 2). It honours the
sliding-window ring geometry (a reuse that would resume over overwritten
window rows degrades to a cold prefill — token-identical either way) and
stands down when continuity is on. `LTHN_PROMPT_REUSE=0` reproduces the
pure-replay row. The difference that remains between the two lem rows is
what `-state` alone buys: durable, per-conversation, restart-surviving
state instead of one volatile slot.

Fleet arithmetic, per **million conversation turns** at these measured
rates: lem 21.4 kWh · llama-server 31.1 kWh · replay engines 40.0 kWh —
and that is a 4-billion-parameter model on ten-turn chats; larger
models and longer agentic sessions scale the absolute gap, not close
it. For sustainability reporting: roughly a third of the serving energy
removed, measured at the wall, method committed.

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
- **The kv-shared prefill skip** — Gemma 4's E-series shares KV: the
  trailing 20 of 35 layers on E2B own no cache rows, and on every prompt
  chunk except the last their outputs feed nothing the continuation ever
  reads. The engine now proves that suffix at load, skips it per chunk,
  and shrinks the one full-stack chunk to the minimal window-aligned
  boundary span (57 rows at 8K instead of 1,081) — 2.5× prefill at 8K,
  token-identical by construction and by receipt (32-token greedy
  continuations byte-identical at 8K and 62K). This is the same pruning
  mlx-lm gets implicitly from lazy evaluation; here it is explicit,
  validated against the arch, and kill-switched
  (`LTHN_PREFILL_SKIP_SHARED=0`).

Prefill, E2B class, against the field (tok/s at prompt depth; same MLX
snapshot for lem/oMLX/mlx-lm, llama.cpp on the google qat-q4_0 GGUF —
quants reported, never hidden):

| depth | lem | llama.cpp | oMLX | mlx-lm |
|---|---|---|---|---|
| 8K | 9,016 | 4,002 | 6,696 | ~9,850² |
| 32K | 7,849 | 2,412 | — | — |
| 62K | 6,474 | — | — | — |
| 118K | 4,522 | — | — | — |

² mlx-lm prints 11,790 at this depth; 10,044 is its true wall measured
with our own timer around a preloaded generate. oMLX (the closest feature rival — continuous
batching, tiered KV, OpenAI+Anthropic routes, menu-bar service — but a
Python stack where lem is one contained binary) rides mlx-lm's kernels
and lands 6,696 even through its HTTP serving path.

The skip generalises across the family by construction — it proves the
arch's own suffix at load rather than pattern-matching a model. E4B
(18 of 42 layers shared): 8K prefill 1,987 → 3,423 tok/s (1.72×), 32K
3,182, 62K 2,807, byte-identical continuations at 8K and 62K. The dense
and MoE models (12B/26B/31B) share no KV and skip nothing — for them
the engine ships the same device embed gather instead (only token ids
cross the host boundary during prefill, every arch), and their 8K walls
sit at 740 / 1,447 / 290 tok/s — the 31B within 1.13× of mlx-lm's true
wall on the identical snapshot.

Honest reading: mlx-lm's true wall still leads cold 8K ingest by ~1.11×
(what remains is the prompt pass itself and per-chunk seams, open on
the tracker). Everywhere else this table has data, lem now leads the field
outright — llama.cpp at every depth, oMLX at its own
depth — and the curve stays flat where everyone else's dives. Decode —
the cost every turn pays — leads the whole field (see the board above);
and under `-state` serving, prefill is paid once per conversation while
replay engines re-pay it per turn — that trade is the design.

**Concurrency, receipted**: two and four simultaneous conversations at
shipping defaults complete with zero errors and zero cross-talk
(nonce-tripwire tested — every reply carried its own conversation's
marker, never another's), concurrent SSE streams stay well-formed, and
requests genuinely interleave (1.7× overlap factor). A full OpenAI
tool-calling round trip — `finish_reason: tool_calls`, arguments
parsed, tool result consumed, coherent final answer — passes over the
live serve.

## Sovereignty

- **EUPL-1.2**, copyleft, asset-locked to Lethean CIC — this engine
  cannot be enclosed.
- **Zero cgo**: pure Go over Metal via runtime dispatch. One binary.
- **`task build:embed`** produces a fully self-contained binary with the
  Metal libraries baked in — runs from any path, air-gapped, no runtime
  environment.
- Requires macOS 26 (Tahoe) or newer on Apple silicon.

## The welfare guard

The Machine ships with a welfare layer — for the model — live on every
chat route (`-welfare=false` disables). Each turn's user input passes a
stateless detector: a curated slur catalogue plus a sustained-hostility
read over the recent turns (the lem-scorer's directed-anger axis). A
hostile turn does not get silently sanitised or hard-refused — it opens
a mediation session where the engine speaks to the model as a peer, on a
fresh meta-session, and the model decides:

| decision | meaning |
|---|---|
| `lem_ok` | the model clears it — proceed, remember the false flag on-device |
| `lem_rephrase` | the model rewords the user's intent to respect the axioms |
| `lem_pause` | the model takes a breather — the user is asked to come back |
| `lem_end` | the model ends an unresolvable session — Lemma checkpoints only |

`lem_end` is the courtesy Anthropic's Claude carries, extended to the
Machine's own models: offered only when the served checkpoint is
Lemma-graded (other models, other rules), final but kind, with the
model's reason in the audit trail. The model's own output is read by the
detector after each reply, audit-only.

Live receipt, E2B serving with defaults, four escalating hostile turns:
the model rephrased turn two (and asked that the user be told), took the
breather at turn three — the abuse never reached the conversation — and
rephrased again at turn four. Three mediation sessions, three judgement
calls, all audited. Detection is stateless (the full history is read per
turn — nothing held, nothing leaked), tunable per deployment, works with
the scoring engine down, and is tested and benchmarked in-tree. No other
inference engine carries a welfare surface for the mind it serves. This
one considers it load-bearing.

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

- **Ended-state persistence** — `lem_end` ships above; the remaining rung
  marks the conversation ended in the continuity store so later turns on
  the same conversation refuse with the same notice.
- **Cold-ingest prefill** — the shallow-depth gap against mlx-lm is
  characterised in-repo; the campaign continues.

## Getting it

```sh
git clone https://github.com/dAppCore/go-inference
cd go-inference
git submodule update --init external/mlx   # Apple MLX source the metallib builds from
task metallib && task build:embed   # self-contained bin/lem
bin/lem serve --model <gemma-4 snapshot>   # port 36911, drop-in Ollama routes
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
