# HANDOVER — go-inference

Curated 2026-07-14 (wall clock): every completed campaign block pruned; the
full history lives in `git log --follow docs/handover.md`. This file now holds
only (1) what is true, (2) what is open, (3) the operational notes that keep
costing time when forgotten.

## What is true (current state)

- **One repo.** go-inference holds the `lem` binary (`cli/`, module
  `dappco.re/go/inference/cli`, binary name/version via Taskfile
  `BIN_NAME`/`VERSION` and Makefile `CLI_NAME`/`CLI_VERSION`), every engine
  (`go/engine/metal` darwin/arm64 no-cgo; `go/engine/hip` linux/amd64 cgo —
  **codex's fenced lane** until ~2026-08-10), the shared serving/train/decode
  layers, the wails desktop (`gui/`), and `examples/`. go-mlx / go-rocm are
  retired.
- **Branch flow:** work lands on dev, main fast-forwards from dev, push
  origin (github dAppCore) non-force. Codex's box fast-forwards from
  origin/dev; its uncommitted tree is never touched — fetch its *committed*
  dev over ssh (`git remote homelab-box`).
- **Serving surface** (`lem serve`, port 36911 — Lethean's own): OpenAI
  (`/v1/chat/completions|completions|models`), Anthropic (`/v1/messages` —
  string-content shorthand accepted; **typed thinking blocks** + the
  `thinking` request control, streaming `thinking_delta` sequence), Ollama
  (`/api/*`), embeddings/rerank, admin behind Bearer. Thinking defaults ON
  for gemma4 (vendor default); the reasoning rides the RESPONSE ROOT
  `thought` field on OpenAI routes and typed blocks on Anthropic. Schedulers
  serial/batch/interleave all re-arm EnableThinking. **`--cors <origins|*>`**
  (off by default) wraps OUTSIDE the admin wall. `--help` presents GNU
  `--long` flags only (guard test pins all 15 surfaces).
- **Metrics honesty:** GenerateMetrics carries real timing splits AND memory
  (engine.MemoryReporter — metal reads MTLDevice.currentAllocatedSize; peak
  is a device-global per-op watermark). Continuity decode applies declared
  sampling (request-set > model-declared > engine fallback) same as
  stateless.
- **Prompt reuse under q8** works via the canonical per-token landing
  (engine.CanonicalLandingSession, resident reuse lane only) — byte-identical
  receipts; tile-position sensitivity is the root cause, upstream of q8.
- **Training:** GPU cross-entropy (lthn_softmax_xent_rows_f32) — host 457ms
  → GPU 28.6ms at T=128; full SFT step 356ms (was 3.64s). **Coupling: a new
  Metal kernel needs BOTH `task metallib:kernels` AND `task build:embed`** or
  the trainer silently host-falls-back.
- **Perf reference (tg512 @ -context 6144, 4-bit, greedy no-think, M3 Ultra,
  2026-07 binary):** plain / MTP / CB-8-aggregate(per-stream):
  e2b 180.8 / 163.9 / 163.0 (20.5) · e4b 125.8 / 116.0 / 115.7 (14.5) ·
  12b 75.4 / 67.9 / 72.1 (9.0) · 26b 140.9 / 117.1 / **220.7 (28.0)** ·
  31b 35.6 / 32.2 / 33.6 (4.2). MTP is CONTENT-DEPENDENT (counting prompts:
  e2b +28%, e4b +38%, 12b +27%; free prose: all five lose; 26b MoE loses in
  both — verify cost, #372/#392: K=1→K=8 is weight-stream amortisation, not
  removable overhead; matvec/gather 73.5% of GPU, host gap 2.6%). Dense CB-8
  ≈ time-slicing; only the MoE scales.
- **TUI** (`lem tui`, Charm): tabs Chat · Models · Service · Settings ·
  Tools · Modes — Service hosts the API on the loaded model through ONE
  serial scheduler lane (TUI turns + HTTP queue, never race).
- **GUI** (`gui/`, wails alpha2.117): LEM Desktop on the **Lethean design
  pack** — Angular 20 over the `<lthn-*>` Lit elements
  (CUSTOM_ELEMENTS_SCHEMA), foundations tokens vendored, committed
  hash-free `dist/` embedded via `//go:embed all:frontend/dist/browser`.
  Chat view streams via Angular's `resource({stream})` with the typed
  thought channel rendered live; the serve manager launches lem with
  `--cors '*'`. The design pack is the `lethean-design` skill (source of
  truth: claude.ai/design project 6460c5d6…, refresh from its zip export).
- **SDK fleet:** `task sdk` regenerates 17 configs (spec → build/sdk/<lang>)
  and `examples/sdk/` holds 17 live-proven lanes (+ Angular). The spec's
  compat routes use RouteDescription.ResponseRaw (dappco.re/go/api
  go/v0.19.0). Friction sections in each lane README are the changelog.
- **Examples house rule:** a public feature ships WITH its examples/pkg/
  proof (memory: project_prove_features_in_examples_pkg).

## Open items

1. **#382 Release beta** (board): demos, edit pass, binaries, video.
   Release-draft note: the outbound-policy slur catalogue ships EMPTY
   (hostility axis does the live triggering) — snider-curated population
   pending.
2. **io module release due:** its core.Result migration is only on dev —
   and the proxy's io v0.14.0 content differs from the repo's go/v0.14.0
   tag (divergence noticed while re-pinning core/gui submodules). Cut a
   fresh tag and re-pin consumers.
3. **`-state` persistent-session lane:** adopt SetReuseCanonicalLanding IF
   it shows the q8 tile-position wobble (#397 left deliberately unwired).
4. **Welfare mediator output is emitted verbatim** — add a formatting pass
   when welfare is next touched (#376 era).
5. **#378 plans-repo spec write** (docs debt; deferred with snider's ok).
6. **Codex-lane flags (engine/hip, raise at next contact, do not fix
   cross-fence):** `go vet ./engine/hip` exits 1 — unsafe.Pointer ×5 in
   hip_driver_cgo.go (4 pre-existing + 1 from the 2026-07 stint); the
   vision/bidir pooling forward is UNARMED pending a real unified-vision
   receipt; hip now carries `internal/gguf` (minimal reader — fine as
   engine-internal, watch drift vs the shared gguf home).
7. **Parked (composed lane — only if it returns to the hot path):**
   RangeKVBlocks sleep-lane block tiling + live multi-turn serve gate;
   composed multi-turn `-state`; graceful degrade when composed sleep
   declines; the composed fuse ladder (gated-delta/attention/whole-layer).
   The ArchSession mainline superseded this lane; #391 declared the
   chained-vs-host ≤3-ulp residual as accumulation-order noise.

## Operational notes (the ones that keep costing time)

- **cwd drift (rtk):** every compound Bash risks starting elsewhere — pin
  git/go to absolute paths (`git -C`, absolute cd at command start). It has
  produced wrong-repo history reads and a stale-binary receipt this cycle.
- **Pipes mask exits:** `cmd | tail; echo $?` reports tail. Redirect to a
  file and grep it when the exit code matters.
- **rtk swallows -v test output** — receipts via `> file 2>&1` then grep.
- **`task sdk`** needs `JAVA_HOME="$(brew --prefix openjdk)"`. Generated TS
  packages need their own `npm install` (+ explicit `run build` — npm's
  script gate can block `prepare`). The Go SDK demo runs `GOWORK=off`
  (depends on gitignored generated output). The C generator's output needs
  `examples/sdk/c/postgen-fix.sh`.
- **GOWORK=off is the CI truth** for gui-family repos — the workspace build
  masks tagged-dep breaks (caught the whole core.Result migration).
- **Bench method:** bench through the serve with the serial method (two-
  point tg48→tg512), GPU quiescent, one lane at a time; MTP pairs need
  fully-greedy for the drafter to engage; the model's response IS the data
  point — never re-score old responses against a later model state.
- **Mantis:** raw REST PATCH can 500 on status flips — use the mantis MCP
  update_issue. Ticket = self-contained work order (no "see spec at X").
- **Forge:** forge.lthn.sh canonical for lthn repos, github for dAppCore;
  core-repo (cladius home) push was pending forge reachability 2026-07-14.
