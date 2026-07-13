# NEXT WAKE (2026-07-16 — residual LOCATED: it's the CB serve layer, not the engine)

## 2026-07-15 (the GPU-span split — b6acd09, pushed)

- THE INSTRUMENT: laneGPUSpanNs (chainedGPUSpanNs's lane twin, gated by
  pieceTimingOn, accumulated in waitReleaseChainedCB) +
  TestProbe26BLaneVsChainGPUSpan (LTHN_PROBE_MODEL-gated, two-pass warm
  — the first pass pays scratch/PSO warm-up and MUST be discarded; the
  cold first run inverted the chain's numbers by +1.3ms/tok).
- THE RECEIPT (real 26B, warm): chain 7.69 ms/tok wall = 7.55 GPU +
  0.13 host; lane 7.62 ms/tok wall = 7.11 GPU + 0.52 host. ENGINE
  WALLS EQUAL — the lane round's GPU span is actually LEANER. The K=1
  serve-level residual (CB 7.83 vs plain 7.32 live) is therefore
  ENTIRELY the CB serve layer: coordinator deliver → request channel →
  scheduleCBStep goroutine (DecodeToken + metrics) → facade Chat
  channel → handler — ~0.5ms/token of channel hops and goroutine
  wakes that the plain path (engine yields straight into the handler)
  never pays.
- NEXT RUNG (first move): collapse the serve-layer hops in
  serving/scheduler — candidates: deliver tokens straight into the
  request's output stream from the drive loop (fold the two channels
  into one), move/batch DecodeToken, cut per-token ScheduledToken
  overhead. Target: recover ~0.4ms/token → K=1 parity and most of the
  K=4 gap (139 → ~155 of plain's 161).
- Depth-2 refutation + the ladder stand in the previous blocks.

# NEXT WAKE (2026-07-15 late — depth-2 refuted; the residual needs the GPU-span split)

## 2026-07-15 (the residual hunt — refutation banked, nothing committed)

- K=1 DISCRIMINATOR (zero new code): CB single-lane 127.7 tok/s vs plain
  single-stream 136.6 — HALF the K=4 gap exists at K=1, so it is
  per-lane overhead vs the serial chain, not batch-boundary sync.
- DEPTH-2 REFUTED LIVE: a two-deep pending queue (the serial chain's
  fill+push steady shape, entry fills to 2, terminal unwinds n rounds)
  went through all 38 gates green — and measured FLAT at K=1 (127.0)
  and NEGATIVE at K=4 (135.9 vs 139.2). Reverted (uncommitted). This
  falsifies the GPU-idle-gap theory for the residual: a queued second
  round covers any idle window (rounds hazard-serialise on session
  buffers within a lane but still queue back-to-back), so the ~0.5ms/
  token K=1 residual lives in the ROUND ITSELF or the per-token serve
  path (2 channel hops with goroutine wakes, DecodeToken, per-Step
  allocs, pool-per-Step vs the chain's pool-per-generation).
- NEXT INSTRUMENT (first move next session): cb.GPUStartTime/GPUEndTime
  spans on CB rounds vs serial-chain links, same boot — one run splits
  GPU-span from host-path. pieceTimingOn already does this for the
  chain (chainedGPUSpanNs); mirror it for lane rounds.
- Shipped state stands at 47c23e9: 26B ladder 93→118→120→123→139 vs
  plain 161 (0.86×), cross-route parity, all receipts in the previous
  blocks.

# NEXT WAKE (2026-07-15 — submit-ahead lanes: 139 tok/s, 0.86× plain)

## 2026-07-15 (lane submit-ahead — 47c23e9, pushed)

- LANE SUBMIT-AHEAD (47c23e9): one speculative round in flight per
  chained lane ACROSS the Step boundary. Step = (1) speculatively
  encode+commit each continuing lane's next round (from-xA input is
  GPU-produced — no token value needed), (2) wait the pending round,
  emit its 4-byte token. Budget-terminal emits skip speculation; a stop
  token wastes ONE round (waited, discarded, pos rewound,
  truncateSpeculativeKV — the serial chain's unwind). Entry rounds
  commit-not-wait into the pipeline; Retire/Close drain in-flight.
- THE TRAP (10-minute hang on the first gate run): the pending cb
  crosses the Step autorelease-pool boundary — the pool drains at Step
  end and FREES the autoreleased cb; the next Step's wait hangs
  forever. Exactly the trap stepGreedyLiveCommit's no-pool comment
  documents, now met from the other side. chainRoundCommit RETAINS the
  committed cb; waitReleaseChainedCB drops the pin at every consuming
  wait (harvest, unwind, drains).
- RECEIPTS: 38 lane+fork gates green (byte-identity + default-Generate
  oracle through the full pipeline; every lane ending exercises the
  truncation unwind); suite 2275/0. Live 26B: 123.5 → 139.2 tok/s
  (+12%). Campaign ladder: 93 → 118 → 120 → 123 → 139 vs plain 161
  (0.86×). Cross-route hashes STILL identical (8efeafbf/a6a40bdd/
  ada2b84e/9ce3bd42), 0 wobbles ×12, usage isolation exact.
- REMAINING ~14%: candidates in likely order — per-Step host
  bookkeeping between harvest and the next speculative commits
  (results assembly, scheduler drive-loop channel sends between
  Steps); the batched chained tail (device-packed rows head: one head
  sweep for all K instead of K per-lane sweeps — grows with K);
  deeper speculation (depth 2) if profiled worth it.

# NEXT WAKE (2026-07-15 — chained lanes live; the gap is lockstep now)

## 2026-07-15 (chained lane step — e533450, pushed)

- CHAINED LANE STEP (e533450): greedy re-encode lanes carry
  [forward → head argmax → next-embed into xA] in ONE cb per lane, the
  serial chained decode's shape. Armed steps emit a 4-byte token; no
  head submission, no hidden readback, no host embed. Arming keeps the
  batched K-row head; head-decline demotes the lane two-phase (forward
  stays valid); chainedSteps is the engagement counter (asserted).
  38 lane+fork gates green FIRST RUN through the chained path; suite
  2275/0; live 26B 120.6 → 123.5 tok/s, cross-route hashes unchanged.
- WHY ONLY +2.4%: the chained shape pays K per-lane head sweeps — the
  SAME head work plain's K goroutine streams pay — so the bubble win
  mostly cancels at K=4 on the 369MB 26B head. The remaining gap (123
  vs plain 161) is LOCKSTEP vs FREE-RUNNING: the plain path keeps ~2
  cbs in flight per stream (its submit-ahead), while lane Steps
  synchronise all K lanes every round (slowest-lane gating + a host
  gap per step).
- NEXT RUNGS (order): lane SUBMIT-AHEAD — one speculative round in
  flight (commit round N+1's cbs before reading round N's tokens; the
  scheduler consumes a round late). SAFE now the position-buffer ring
  exists (b6a4147 fixed exactly this class). Closes the lockstep gap.
  Then the BATCHED CHAINED TAIL: device-pack the K finalOut hiddens
  into a rows buffer (K tiny copy kernels), ONE K-row fused head sweep
  + per-row embed gathers, all in a trailing cb ordered by hazard
  tracking — head-sweep economy that matters as K grows.

# NEXT WAKE (2026-07-15 — THE FORK IS FIXED: submit-ahead position clobber)

## 2026-07-15 (the fork hunt — b6a4147, pushed)

- ROOT CAUSE: the chain-vs-host fork was never precision or tie-breaks.
  The submit-ahead chained decode commits link N without waiting, then
  encodes link N+1 — and stepTokenEncode wrote pos_N+1 into the ONE
  shared offBuf (a single int32 the kernels read at EXECUTION: RoPE
  position + KV append row). Link N's kernels then read the clobbered
  position: its OWN token computed fine, but its KV row landed at the
  NEXT slot → every later token read stale/shifted KV. Fixture
  signature: token 3 right, token 4 on wrong, deterministic.
- THE HUNT (method receipt): forced-token step-vs-step diag exonerated
  the shared encode; three-arm greedy diag (oracle/per-lane/shared)
  showed BOTH lane arms == host, oracle alone diverging; the chain
  kill-switch (stepGreedyChainDisabled) proved chain-vs-host WITHIN
  Generate; embed-twin + argmax-twin probes exonerated the primitives
  byte-identically; liveSubmitAheadDisabled isolated the speculative
  link; the offBuf single-int32 execution-read shape closed it.
  Enumerate the arms, THEN compare pairs — kill-switches are the
  fastest scalpel.
- FIX: position-buffer ring pre-built in the pooled core scratch
  (slot 0 = the old offBuf; step path allocation-free — the AX-11
  budget gates caught the lazy-alloc first draft immediately).
  rotateOffBuf claims a slot per step encode; a committed link's
  offset is immutable for its lifetime.
- RECEIPTS: TestChainedDecodeArmsAgree (permanent — chain+spec ==
  chain-nospec == host) + TestChainedDecodeEmbedTwins; lane MoE oracle
  back on DEFAULT Generate (un-pinned); metal suite 2275/0. LIVE 26B:
  plain-boot 0 wobbles in 12 rounds (the parked 45-vs-53 variance was
  THIS bug — dissolved); CROSS-ROUTE PARITY first time (CB and plain
  boots hash-identical content: 8efeafbf/a6a40bdd/ada2b84e/9ce3bd42);
  plain aggregate ROSE to ~161 tok/s (correct KV rows), CB 120.
- NEXT: the lane-wide chained step is UNBLOCKED (the arms agree) —
  fuse head argmax + next-embed into the lane forward cbs, kill the
  per-step GPU/host bubbles, close the 26B CB gap (120 vs 161).

# NEXT WAKE (2026-07-15 — batched head live; the 26B gap fully root-caused)

## 2026-07-15 (batched head — 8a1a2a0, pushed)

- BATCHED HEAD (8a1a2a0): all-greedy phase 1 = ONE K-row fused
  lm_head+argmax via ArchSession.greedyRowsFromHiddensInPool (the MTP
  verify head — on quant heads ONE weight sweep scores all K rows).
  headRowsCount is the engagement counter, asserted in both byte-identity
  gates (the CB-never-live lesson: no silent fallback may pass as
  batched). sharedReencodeForward switched to per-lane cbs + pipelined
  waits — the single shared cb's pass-edge memory barriers serialised
  the lanes against each other (measured flat).
- Live 26B: 120.6 tok/s (+2% over the shared rung) — EXACTLY the head's
  share of step bytes at K=4 (~369MB head sweep vs ~16GB of expert/dense
  reads per 4-token step). The probe receipt pinned live engagement:
  canUseDirectHeadGreedy=true, qmm_t pipeline present for
  vocab=262144/gs=64/bits=4, sharedStepEligible=true. The head win
  SCALES WITH K; at K=4 on 26B it is small by physics, not by defect.
- THE RESIDUAL GAP (120 vs plain 151) IS NOW MECHANICALLY UNDERSTOOD:
  plain 26B serve decodes on the CHAINED-LIVE tail — GPU argmax feeds
  GPU embed, ONE cb per token, zero host round-trips, 4 goroutines'
  cbs overlapping freely. The lane step alternates GPU and host phases
  (wait forwards → host pack rows → head cb → wait → host embedID ×K →
  encode forwards → commit) — a pipeline bubble every step. The
  gap-closer is a lane-wide CHAINED step (fuse head+next-embed into the
  forward cbs), but it changes the lane's head arm from host-loop to
  chain — BLOCKED ON the chain-vs-host token fork (previous block).
  Resolve the fork first (GPU-vs-host embed dequant rounding / argmax
  tie-break; unify or declare), then build the chained lane step
  against the unified arm.
- Bench-shape note: the 26B agg_bench warm round pays first-boot
  page-ins (10-50 tok/s noise) — read measured-1/2 only.

# NEXT WAKE (2026-07-15 — shared submission live; the chain/host fork found)

## 2026-07-14 (final rung of the session — de239ac, pushed)

- SHARED SUBMISSION (de239ac): K re-encode (MoE) forwards encode into ONE
  command buffer per lane-set Step — sharedStepSink threaded through
  stepTokenEncode (nil sink = the classic own-cb shape, byte-for-byte
  untouched); sharedEncodeEligible gates the device-router quant MoE lane
  up front AND at step entry; the MoE break-out under a sink fails clean
  and the owner retries per-lane off an uncommitted cb. Live 26B-A4B
  (-context 4096): 4-concurrent unique-prompt aggregate 93 → 118 tok/s
  (+27%); plain 151. The residual gap is PHASE 1 — K separate head
  submissions per step (greedyFromHiddenInPool per lane) — i.e. the
  batched-head remainder rung. Suite 2273/0.
- THE REAL FIND (pre-existing, now tracked): on the quant MoE fixture,
  Generate's CHAINED-LIVE tail (GPU argmax head + GPU-produced next embed,
  arch_session.go ~4989) diverges from the HOST loop on near-tie logits —
  oracle [15 42 7 26…] vs oracle-nochain [15 42 7 34…], where nochain ==
  both lane arms byte-for-byte. Two production arms disagree WITHIN
  Generate; the ICB-parity-scar family ("chained paths differ from host
  re-encode"). Consequence: plain serve (chain) vs CB lanes (host loop)
  can pick different tokens at temp 0 on near-ties on REAL weights.
  Next: root-cause GPU-vs-host embed dequant rounding / argmax tie-break,
  unify or declare. The lane oracles pin the HOST arm
  (stepGreedyChainDisabled) — like-for-like.
- Method note (cost an hour): my first shared-vs-plain diag compared
  stepIDInPool vs stepIDEncodeShared with FORCED tokens — equal, correctly
  exonerating the shared encode — but the failing test compared lane
  GREEDY LOOPS vs Generate. The three-arm diag (oracle / per-lane loop /
  shared loop) + the chain kill-switch found the fork in two runs.
  Enumerate the arms BEFORE comparing pairs.

# NEXT WAKE (2026-07-14 night — MoE rides CB; 26B live)

## 2026-07-14 (later the same session — edd3249, pushed)

- MOE RIDES CB (edd3249): laneSet.Prepare admits no-ICB sessions (MoE —
  icbEligible declines the router block) as RE-ENCODE lanes; phase 2
  advances them via the session's own stepIDInPool (embed→PLE→stepToken→
  pos++ — the plain path's per-token machinery, byte-identical by
  construction). ICB lanes keep the shared replay + GEMM fold, partitioned.
  BatchForwardCount stays honest (re-encode lanes claim no shared
  submission — MoE lane tests assert fwd==0).
- THE BEFORE (live receipt): 26B + -scheduler interleave chat returned an
  EMPTY completion on the prior binary — the coordinator CANCELS a request
  whose Prepare fails ("not recorded-ICB eligible"), stream closes with 0
  tokens, finish "stop". Silent serve breakage on every MoE model, not a
  fallback.
- THE AFTER (26B-A4B qat-4bit, -context 4096, quiescent box): 'OK' with
  real usage; 4-concurrent usage isolation 3/3 exact; CB counts identical
  every round. Unique-prompt aggregate (salted prompts — continuity AND
  the engine prompt-reuse cache both hit on repeats and contaminate
  repeat-round A/Bs): CB 93 tok/s vs plain 151 (0.62×) — re-encode lanes
  SERIALISE in the drive loop (own commit+wait per lane per step) while
  plain overlaps 4 goroutines' submissions on the GPU queue. NEXT RUNG:
  the stepToken encode-into-caller-CB split so K re-encode forwards share
  ONE submission (the 26B quant device-router forward already encodes
  break-free — encMoEBlockQuantDevice); that closes the gap and then some.
- OPS (snider's catch): each lane session allocates the FULL context
  window's KV — a big-model CB boot at the model default can exceed device
  memory with K lanes. Set -context on CB serve boots (4096 for bench
  shapes). The durable fix is kv/budget.FitsMemory admission gating
  (already noted on defaultLaneSetMaxLanes).
- PARKED OBSERVATION: plain-path 26B same-prompt temp-0 completion COUNTS
  varied under 4-concurrency (45 vs 53, stateless boot); CB was identical
  every round. Counts ≠ hashes — needs the content-hash probe shape on 26B
  plain before calling it a wobble (e2b was hash-clean yesterday).
- Metal suite 2271 PASS / 0 FAIL (up 2: the MoE lane gates). Gate hygiene:
  the first full-suite run died exit 144 — it shared the GPU with my 26B
  serve boots; run the suite QUIESCENT.

# NEXT WAKE (2026-07-14 evening — usage race closed; welfare opt-in)

## 2026-07-14 (post-restart session, both on dev, pushed)

- WELFARE OPT-IN (6840d9a): -welfare defaults FALSE while CB is built
  (snider). Guard untouched behind the flag; the welfare×CB audit gates
  default-on returning (the guard decorates TextModel.Chat — CB never calls
  it, so default-on never covered CB chats anyway).
- USAGE RACE CLOSED (b578008): GenerateConfig.MetricsSink (+ WithMetricsSink,
  json:"-" — train marshals configs in SSD reports and encoding/json rejects
  func fields) delivers each generation's final metrics request-scoped.
  ScheduledRequest.MetricsSink carries it across the scheduler's
  opts→SamplerConfig fold (which drops non-sampler opts — the reason a
  handler-installed option alone could never work); serial/batch/interleave
  re-arm it at dispatch, the CB route delivers its own scheduler-built
  counts, continuity delivers the RecordChatMetrics numbers. The openai
  handler prefers sink delivery on both serve paths; global Metrics() stays
  as fallback (HTTP backend, engine/hip's own scheduler — codex's lane,
  untouched, drops the sink exactly like the old facade did).
  Receipts: 12 tests across the 5 seams (handler tests discriminate sink vs
  global with different numbers); -race green on all 5 pkgs (63 + 1448 —
  the 600s scheduler "hang" was TestBatch_Cancel_Active flaking under
  triple-package race load, passes 0.00s alone); full sweep green; live
  4-concurrent usage-isolation probe (distinct max_tokens 4/8/16/24 +
  distinct prompt sizes, all finish=length) 5/5 rounds exact on BOTH the
  plain boot and the -scheduler interleave boot.
- Next rungs unchanged otherwise: welfare×CB audit · concurrent lane
  admission · CB lane metric durations · batched head · MoE arches.

# NEXT WAKE (2026-07-14 — CB serve integration LIVE end-to-end; restart handover)

## THE 2026-07-13 MARATHON (all pushed, dev @ 2be27d9 — tracker #385 holds the full detail)

**Where the CB campaign stands — carry on from here:**

- FOLD COMPLETE: bf16 AND quant weight-read-once byte-identical (072e0cd —
  lthn_qmv_rows is now qmv_fast_impl's M-variant, packs matched, and the lthn
  metallib builds -fno-fast-math like MLX's own kernels; the plan gate is the
  fast-twin rule outDim%8 && inDim%512). Envelope exclusions LIFTED (09bf08a):
  the E2B fires-and-diverges was NEVER the mirrored arms — it was gemmDims
  sizing gate/up/gated slabs from s.dFF while MatFormer deep layers run 12288
  (no-cache layers → trail-free corruption) + the plain SDPA routing by live n
  vs the recorded fixed 2-pass fan. E2B folds byte-identically to completion.
  PROFIT GATE (d3f8035): the fold is a wall LOSS on quant (K=4 AND K=6
  receipts) — LTHN_CB_GEMM unset=AUTO folds bf16 only, 1=force, 0=off.
  K>4 = chunked byte tier (85abaab, fold-only; chunking the PLAIN multi-row
  route was measured −20% e2b MTP and reverted; LTHN_QMV_CHUNKS=0).
- SERVE INTEGRATION LIVE (the real story): per-lane sampling (44acef5,
  token-identical to GenerateSampledEach) + chat via FormatChatPrompt
  capability + ResolvedStopTokens (f2ceaff — also fixed raw lanes decoding
  past EOS) + THE WIRING FIX (72c9565): engine.TextModel had never satisfied
  inference.TokenizerModel, so cbEngine NEVER bound on any live boot — CPU
  tests bound to fakes. Encode/Decode/ApplyChatTemplate + streaming
  DecodeToken now exported; scheduler probes via inference.As; lazy rebind
  heals the construct-during-load race. **The "2.10× CB live receipt" from
  earlier that day is WITHDRAWN — it was plain-vs-plain boot variance.**
- BATCHED PREFILL (38d0faa): prefillLane runs the production prefill route
  (prefillRetainedTokens) — 4×1161-tok cold admissions 29.83s → 0.70s (42×),
  TTFT at plain parity. Per-request usage on CB responses.
- WOBBLE CLOSED (2be27d9): temp-0 outputs were ALWAYS deterministic (probe:
  identical content hashes, seq+conc, two boots). The variance was usage
  accounting — the openai handler reads GLOBAL model.Metrics() after its
  stream, racing concurrent generations (pre-existing, engine-wide under
  concurrency) + a one-day facade shift-by-one (fixed). CB-served usage is
  scheduler-built per-request and immune; the plain path keeps the race
  until metrics can flow request-scoped through iter.Seq (API change).

**Honest live state (quant e2b, K=4):** CB aggregate ≈ plain-interleave on
short chats — the density fold is profit-gated off on quant and this box
parallelises 4 independent streams about as well as one batched submission.
CB's wins today: admission cost, one-submission-per-step CPU, scaling
headroom, race-free usage. Decode-aggregate wins live on bf16 (1.51× fold)
and higher K.

**Next rungs (in rough value order):** welfare×CB audit (snider parked it —
"make the thing first"; the guard decorates TextModel.Chat which CB never
calls; audit before CB defaults on) · request-scoped metrics through the
handler surface (kills the plain-path usage race) · concurrent lane
admission (Prepares serialise in the drive loop — fine at production prefill
speeds, revisit at higher K) · batched head · non-ICB arches (MoE 12B/26B).
Also open elsewhere: #386 family-row catalogue (snider: "the catalog comes
later"); #382 release folds (default-stance declaration, whole-repo
remeasure).

**Environment gotchas (cost real time today):**
- engine/metal TestMain exits 0 SILENTLY without MLX_METALLIB_PATH — an
  "ok …0.3s" ran ZERO tests; always count "--- PASS" lines in gate receipts.
- lem serve needs MLX_METALLIB_PATH too or the backend registry is empty
  ("no backends available" 404). lem.sh exports it; a bare binary boot must.
- lem.sh build outputs /private/tmp/lem-dev/bin/lem-dev (NOT under the skill
  dir).
- rtk filters -v test output; receipts go to a file (rtk proxy) and get
  grepped for --- PASS/FAIL counts.
- Serve bench harness shapes (recreate freely): 4-concurrent chat POSTs to
  /v1/chat/completions with temperature 0 + usage sums for aggregate; the
  same with max_tokens=1 and a ~1161-token prompt for TTFT; content-hash
  rounds (seq+conc) for determinism. LTHN_CB_STEP=0 = the plain arm.

# NEXT WAKE (2026-07-16 — #381 SHIPPED: the skip is live, 2.1-2.6x at every depth)

## LATER THE SAME NIGHT (2026-07-12 — schedulers become modes; the CB physics receipt)

- SCHEDULERS UNIFIED AS MODES (snider's fork resolution, a57829e): ONE
  serving/scheduler — serial (survivor base) / batch (schedule's admission
  policy + interleave's bounded MaxQueue, live tokens via iter.Pull) /
  interleave (verbatim); serving/schedule + serving/interleave DELETED.
  -scheduler flag: unset = request path byte-for-byte unbuilt; batch fails
  closed at New without TokenizerModel. Suite 9864 > 9797 baseline.
  Non-reconcilable documented: schedule's static int-token slab vs live
  delivery — batch keeps lockstep prefill costs, why interleave stays.
- THE CB PHYSICS RECEIPT (live A/B, E2B, 4 concurrent × 48 tok, alternating
  arms): serial 142.9 vs interleave 143.6 tok/s aggregate — PARITY, pinned
  at single-stream rate. Routing proven live; aggregate is ENGINE-bound.
  #35 slice 3 = step-level batched decode INSIDE the engine (one forward
  advances N requests; stepTokensBatched* machinery exists unwired).
  Slice 2 (per-resident-model scheduler under -models-config) in flight.
- HIP 12B round 1 (honest negative, 6de7459): per-head value-norm fix on
  the single-token path (RED->GREEN, armed box suite 1246/0 vs own
  kernels); 12B still garbles — directional, pack exonerated, suspects =
  16 query heads vs the 8 k_eq_v layers. NOTE the layering is a 5:1
  INTERLEAVE (full attn at 5,11,...,47 — every 6th), not set positions.
  Round 2 (layer-0 numerical oracle) running. TRAP BANKED: box card is
  gfx1101; make hip-amd defaults gfx1100 — 14 false FAILs from the arch
  mismatch; always AMD_HIP_ARCH=gfx1101.
- COMPOSED BREADTH (same merge): mamba2 + rwkv7 gained composed wrappers +
  projMixer (block-level NoProj splits; byte-for-bit parity) — the CB-fold
  family now reaches all four mixer kinds.
- RATCHET r3+r4 (metal 76.77 -> 77.1%, suite 1634/0): quant-PLE fixture
  family (two 0%->100% unlocks, #381 bounded-chunk branch verified firing);
  arch_session 72.9->75.0 (bidir lane non-vacuous vs causal control);
  vacuous-fixture trap caught (constant-fill tensors). r5 target: the
  pipelined-GPU sampling tails. Quant fold variance ~0.125 diagnosed as
  accumulation-order class, tolerance calibrated at site.

## POST-UNFENCE NIGHT (2026-07-12 night — the oMLX gap closes; ladder terminal rung)

- #23 LADDER COMPLETE to the terminal rung. Input-side fold merged
  (2N+1→N+1, +20%/+18%), then HEAD FUSE (e64421b): final RMSNorm + head
  GEMM onto the LAST layer's tail CB (N+1→N). 0.8B 24.96→25.83 /
  24.41→25.46 tok/s interleaved A/B; 4B neutral at floor (head CB was
  amortised — the win is the norm-glue round-trip); text byte-identical;
  metal 1609/0. The serve-loop piece that makes it real:
  generateStepwiseWithSession prefers a stepper-level LMHead (one-shot
  byte-keyed logits cache in composedStepper). Kill switch LTHN_HEAD_FUSE.
  Day total 0.8B: 1.6 → ~25.5 (~16×). Remainder booked #56: the fuse also
  fires per prefill-walk token (L==1 stateless replay) — bounded, next rung.
- TOKENIZER ChatML fix (886d2b7): `added` map = FULL atomic matcher (all
  added tokens), `special` = decode-skip only; qwen im_start-as-BOS branch
  removed. Go==HF token-for-token; the "think soup"/assistant-echo leaks
  were encode defects, never template bugs. Gate turns 36→4 tokens clean.
- WRAPPER CAPABILITY BUG CLASS (233fb4c): interface embedding never widens
  method sets — welfare/policy wrappers now forward AcceptsImages/Audio;
  inference.WrappedModel/BaseTextModel unwrap seam. Audio+vision now work
  over HTTP (both were 400-ing behind wrappers while CLI worked). General
  registry-level fix parked in #50 item 4.
- AUDIO CONVERGED: model/gemma4/audio is the one neutral Conformer home
  (#44 — engine/metal audio_features.go 385→59 lines, aliases); HIP port
  #31 done to the same goldens. Remainder tasks #42/#43/#45/#46.
- HIP LINE: causal-stride kernel root-cause fixed (f2bb2cc — stride derived
  from key_bytes/token_count broke on causal windows; GPU suite 1475/0).
  Option A format fork (e79ee2e): mlxaffine skip-list keeps
  .per_layer_model_projection.weight wide BF16 (hip's loaded contract).
  First coherent AMD decodes: E2B 125 tok/s, E4B 72 tok/s on the 7800 XT.
  make test-matrix + named binaries lthn-{amd,cuda,cpu-x86,cpu-aarch64};
  cuda lane is REAL CUDA runtime via ZLUDA-on-AMD; cpu lane = HIP-CPU+TBB
  (linker gap #51). Kernel doctrine (snider): one portable HIP++/C++23
  source, made-to-work cpu/cuda/amd; per-hardware tuning AFTER this
  version. 12B dense forward lane running (#52). hip hygiene #49 +
  darwin-tag fix c7f17c8.
- K-QUANT SWEEP (#47): 7 of 9 GGUF formats were broken-since-written —
  Q4_K/Q5_K geometry, Q4_0 scale sign, Q5_0 STRUCTURAL (24- vs 22-byte
  block + writer stride table), Q6_K/Q3_K/Q2_K naive scales — all
  byte-pinned vs real compiled libggml via a clang harness.
- oMLX GAP CLOSED (each with receipts): #34 embeddings — go-rag had NO
  native ML (delegation only); model/bert built fresh (WordPiece, 12-layer,
  CLS/mean, L2), cosine=1.000000 vs sentence-transformers on bge-small;
  /v1/embeddings + /v1/rerank live. #36 multi-model — resolver-seam
  registry: aliases, lazy load, LRU+idle-TTL under byte ceiling with real
  Close reclaim, pinning, model:profile presets; -models-config. #35
  slice 1 — serving/interleave skeleton (dual admission budget, goroutine
  isolation; 3 real concurrency bugs found by its own tests). THREE
  schedulers now exist (schedule/scheduler/interleave) — survivor fork is
  an operator call before slice 2. #40 tray — gui ServeService + manager
  (gui test binary had been build-dead since the core.Result migration;
  22 masked assertions repaired). #39/#48 tiered KV — RAM-default store
  (-state-store unset = pure RAM; InMemoryStore race fixed), then ramspill
  (cb2fd5c): kv/kvtier found COMPLETE, wrapped as a state.Store behind
  -state-ram-budget (coldest chunks spill to scratch .kv, wake
  transparently). kv/radix cross-conversation sharing = evidenced defer →
  #54 (needs Tokenize capability + global prefix index; whole-message-hash
  keys can never cross-match; ReusePrefix dedup already general).
  kv/blockcache found HIP-only → #55.
- QUANT FORMAT RESEARCH (a5ecc6d → #53): GPTQ #1 next (autoround's numeric
  core exists; its MXFP4/NVFP4 schemes are metadata stubs), AWQ, plain fp8,
  NF4; MX hold — gfx1101 has ZERO fp8/fp6/fp4 acceleration (CDNA3+/RDNA4/
  CDNA4 territory) and no ggml-style reference to byte-verify. Affine =
  shared native BOTH engines, GGUF = interchange: confirmed resting
  position. Operator fork: is lem a general HF exporter at all?
- KNOWLEDGE PACK (knowledge-packs 722e1d4): corego/pkg/inference split into
  repo-shaped area folders (contract/build/engine{,metal,hip}/model/kv/
  decode/serving/train/agent/lem/gui), root README = getting started;
  facts re-verified against dev HEAD.
- FLEET DOCTRINE this stretch: staffing 2 Opus + 4 Sonnet (Sonnet default,
  quota pooled); rtk block in every brief (it HELPS — pass/fail summaries,
  rtk --help for retrieval; never work around it); no deferred notes on
  Mantis — harvest to tasks (#42-46, #50-56 all came from that rule).

## UNFENCED EVENING (2026-07-12 late — hip lands, GGUF interop, the ratchet)

- HIP UNFENCED (snider): scout run proved the lane workable → codex's stint
  3eb3cf2 MERGED to dev (15.9k lines, hip-only) with ALL 11 aspirational
  spec-tests implemented green: config-label veto root-caused; kernel-source
  assertion retargeted; thought-suffix test repinned to runtime contract
  (provenance in commit — orphan red predating the stint); attached-drafter
  batching via scoped ForceBatchedProjection — incl. wiring codex's
  built-but-UNWIRED accepted-prefix KV truncation into KQ8VQ4 verify. gfx1101
  hsaco builds (-O3); MoE router + GGUF expert kernels PASS on the 7800 XT.
  Box working rules: desk /home/claude/Code NEVER touched; fresh /tmp clones
  + rsync for validation. OPEN: TestHIPHardwareTransformerKernelSource
  numerical red (slice[0]=3.0 want 1.462, kernel-debug, inherent to stint);
  on-device tok/s parity for the new verify path awaits a real-model run.
- lem BUILDS ON LINUX for the first time (799fce8): cmd/lem imported
  engine/metal unguarded — portable seam (engine_metal.go darwin /
  engine_other.go) + tagged the untagged MLX example test. Receipted on a
  fresh homelab clone: build + both packages green.
- QUANT FEATURES RECEIPTED (oMLX parity): lem quant bf16→MLX-4bit —
  E2B 9.54→2.71 GiB, loads + answers 143.7 tok/s. GGUF lane: multimodal
  crash fixed (tower skip + F32 fallback, 8739e99) THEN the full
  llama.cpp-interop layer landed (#28 merged): canonical blk.N.* names,
  56-key gemma4.*+tokenizer.ggml.* metadata, per-tensor type policy,
  rope_freqs mask, MatFormer double-wide FFN widths from tensors — and the
  HIDDEN BLOCKER: quantizeQ4_K/Q5_K had wrong super-block geometry + never
  applied sub-block scales (garbage, never validated running) — rewritten to
  ggml reference. ACCEPTANCE: llama-completion generates coherently from our
  file; type histogram identical to the unsloth oracle. JANG parked (snider:
  experimental). Remainder: other gemma4 GGUF formats need own oracles;
  K-quant fix warrants a consumer regression sweep.
- QWEN FUSES MERGED (#23 ladder): gated-delta input CB (+3.8%) + attention
  q/k/v CB (k/v host matmuls absorbed, +15%) = ~21.3 tok/s on 0.8B,
  byte-identical text, thermal-honest interleaved A/B method banked.
  Frontier: host glue (residual/RMSNorm/conv/recurrence/sdpa) between device
  GEMMs. QWEN-LINE LANE RUNNING: 4B step-up + residual/RMSNorm on-device as
  SHARED engine primitives (snider mandate: multi-arch shared kernels,
  engine/ or model/ roots — no composed-only one-offs).
- CODECOV RATCHET (#30 = pre-req for the #8 tag; snider doctrine: gap table
  = how we find perf + underused features + gemma4-isms in shared code).
  Measured: whole-repo CPU 67.8%; excluding hip 91.1%; metal 75.6%; hip ~1%
  on darwin (12.6k stmts = 78% of the gap, needs box-measured coverage).
  Rotation 1 MERGED: decode/generate 37.2→83.3 tagged (supervised idiom,
  -state proven on real weights); eval/datapipe 75.1→89.5 + eval/score
  82.6→92.8 (ScoreAll -race). Rotation 2 IN FLIGHT: modelmgmt, shared-roots
  (arch-neutrality tests: non-gemma fakes drive shared surfaces), metal
  interior r1 (per-file uncovered-mass ranking = deliverable).
- MANTIS: #1840 noted fixed (2488; status flip needs a human — REST update
  500s). #1839 open remainder = unit D (mask fidelity) + unit E (HIP
  Conformer port — NOW DISPATCHABLE, board #31, acceptance kit = the audio
  goldens).
## QWEN-LINE SHARED FFN-TAIL FUSE (2026-07-12 — evidenced NEUTRAL, and why)

- BUILT SHARED: engine/metal/residual_norm_mlp_device.go — ResidualNormMLPDevice,
  an ARCH-NEUTRAL f32 primitive (named for the op, not for composed) that encodes
  the whole pre-norm SwiGLU FFN sub-block into ONE command buffer: hplus = h +
  mixOut → normed = RMSNorm(hplus, w) [plain rsqrt(mean²+eps)·w, the f32 rms
  kernel] → SwiGLU(normed) → y = hplus + mlpOut. Every pre-norm SwiGLU stack
  (llama/qwen/mistral) has exactly this tail, so it is a shared rung, not a
  composed one-off. Reuses the existing emit helpers (emitBinary vv_Addfloat32,
  emitRMSNormRows, emitSteelGemm, emitUnary sigmoid) + pooled pinned scratch.
  composed declares the AX-8 hook (composed.ResidualNormMLPDevice); native binds
  it; forwardEmb routes the dense-MLP layers above deviceMinWork through it, MoE
  + sub-floor + device-fail fall back to the host add/norm/MLP/add path.
- PARITY: TestComposedResidualNormMLPFuseDeviceVsHost (counter-guarded device-vs
  -host, f32 tol, mirrors the q/k/v fuse test). Full metal suite 1554 green,
  model/composed + model/qwen3 52 green, vet clean.
- RECEIPT (interleaved same-thermal A/B, temp 0, sky-blue prompt): greedy text
  BYTE-IDENTICAL before/after on BOTH models. tok/s NEUTRAL within thermal noise
  — 0.8B before 21.3/22.0 -> after 23.0/21.9; 4B steady-state ~9.3 both (the lone
  8.3 was the cold first run).
- WHY NEUTRAL (the inventory correction): this fuse RELOCATES host glue onto the
  GPU, it does NOT collapse a command buffer. The MLP was ALREADY one CB
  (ComposedMLPDevice); the residual adds + RMSNorm were CPU work BETWEEN CBs, not
  a CB each — and at L=1 that CPU glue is ~3 passes over [1,D], negligible. So
  the CB-per-token count is UNCHANGED by this slice; it removes host glue (which
  matters more at larger L, and cleans the boundary) and, crucially, is the
  scaffold the real CB-collapse needs. The genuine collapse is merging the
  mixer's FINAL-projection CB (o_proj / out_proj) INTO this tail CB — i.e. the
  mixer hands a device-resident buffer to the tail instead of reading it back —
  which is the GPU-resident whole-token orchestration flagged design-worthy
  below. That is the next rung; this slice makes the tail a single shared entry
  it can plug into.

## QWEN-LINE 4B STEP-UP (2026-07-12 — the family scales, zero loader fixes)

- RECEIPT: mlx-community/Qwen3.5-4B-OptiQ-4bit loads through LoadComposed and
  greedy-generates (temp 0) coherent text — the sky-blue Rayleigh answer — at
  8.2 tok/s decode (55 tok / 6.553s), prefill 9 tok/s (23 tok / 2.691s), on the
  same host-f32 + device-GEMM-fuse composed stack the 0.8B rides. Two-turn
  continuity gate (composed_gate_4b.py, mirror of composed_gate.py at the 4B):
  turn1 "OK Wibble", turn2 "Wibble" — RECALL=PASS, NO-REPLAY=PASS (turn2
  prompt_tokens 25 < full-history 29, so no replay).
- NO GEOMETRY/LOADER FIX NEEDED. The shape-derived loader absorbed every 4B
  delta out of the box: D 2560, 32 layers, 16 attn heads / 4 KV, head_dim 256
  (explicit, != D/heads=160), attn_output_gate=true (q_proj [8192,640] = 2·16·256
  carries the [q;gate] split), full_attention_interval 4, vocab 248320, tied
  embeddings; gated-delta derived keyHeads 16 / valueHeads 32 / headDim 128 /
  convK 4 from the weight shapes; mixed per-tensor 4/8-bit OptiQ quant resolved
  by QuantConfig.For overrides (embed 8-bit, per-layer proj bits vary). ~16GB
  f32 dequant footprint, fine on 96GB.
- Confirms the composed stack is arch-general across the Qwen3.5 family, not
  0.8B-tuned. The perf ladder below (device seam, fused MLP) now has a second,
  larger model to A/B against — see the shared-glue slice receipts.

## #23 COMPOSED DEVICE SEAM (2026-07-12 evening — 1.6 -> 16.7 tok/s in one day)

- SLICE 1 (a36b11a): composed.ProjMatMulInto — the stack's OWN projections
  (attn q/k/v/o, MLP, LM head 155 MMAC) now ride the steel GEMM, same AX-8
  seam as qwen3/mamba2/rwkv7; native binds at init; deviceMinWork 1<<20 keeps
  sub-MMAC GEMVs on the sharded host path. Parity: TestComposedDeviceVsHost
  (one-layer attn+MLP forward, f32 tol) + TestMatNTIntoDeviceHook (floor /
  verbatim / error-fallback). 10.5 -> 14.9 tok/s.
- SLICE 2 (5620c01): qwen3's hooks fired UNCONDITIONALLY — the gated-delta
  in_proj_a/b ([16,1024] = 16 KMAC) each paid a full CB round-trip, 36
  wasted/token. Floored to match. 14.9 -> 16.7 tok/s, same greedy text;
  continuity gate PASS at the new tier (turn-2 now 4s, was 21s host-tier).
- DAY TOTAL on the 0.8B hybrid: 1.6 -> 16.7 tok/s (10.4x): column-sharded
  host GEMVs (574d179, 8325ee2) + device seam + floors. Numeric tier note:
  device f32 accumulation vs host f64 — same tier the gated-delta already
  served at; greedy text unchanged on the receipts prompt; -state contract
  is token-prefix, per-build deterministic.
- SLICE 3 (0460e13): fused SwiGLU MLP — instrumented the round-trip first
  (330us/call at decode shapes, ~10us of it compute); composed.MLPDevice
  encodes gate+up GEMMs + sigmoid + 2 multiplies + down GEMM into ONE CB,
  [L,FF] intermediates device-resident, pooled pinned scratch. 16.7 -> 18.6
  tok/s, prefill 15 -> 19. Ladder next: same fuse for gated-delta
  (in_proj_qkv/z/out) and attention (q/o) projections, then whole-layer.
- THE CEILING NOW: ~70+ per-projection command-buffer round-trips per token.
  Next slice is the GPU-resident orchestration — weights uploaded once,
  whole token encoded in ONE CB (intermediates stay device-side), recurrence
  on device — the real composed decode session in engine/metal. Design-worthy;
  brief a research pass before building. mamba2/rwkv7 hooks share the
  no-floor issue but are not on today's receipt path (noted, not touched).

## SWEEP ROUNDS (2026-07-12 afternoon — 8 Opus lanes, 2 rounds; the wrapper bug)

- THE BUG OF THE DAY (233fb4c): welfareTextModel + policyTextModel embed
  inference.TextModel, which does NOT widen the wrapper's method set — so the
  serve handler's VisionModel/AudioModel assertions failed on EVERY wrapped
  serve (welfare is default-on): all image_url AND input_audio requests 400'd
  while the same checkpoint worked over the CLI. Nothing could trip it until
  the first audio-capable serve existed (today). Failing tests first, then
  explicit AcceptsImages/AcceptsAudio forwards on both wrappers. LIVE RECEIPT:
  base64-WAV input_audio chat completion on e2b → exact fox transcript, 2.8s.
  LESSON (bug class): a serving wrapper must forward every capability
  interface the handler gates on; embedding forwards calls, not assertions.
- COMPOSED CPU LEVERS (574d179 composed, 8325ee2 qwen3): both host GEMVs now
  shard output columns across cores — bit-identical (per-element f64
  accumulation order untouched; serial floor 1<<20 MACs). MLP fwd 12.47→1.15ms
  (10.9x), attn 4.00→1.43ms, gated-delta 4.48→1.25ms. 0.8B live decode
  1.6→10.5 tok/s quiet-machine (35-tok coherent answer). Next order of
  magnitude = the GPU lane (ProjMatMulInto seam ready) — board task open.
- FLEET ROUND 1 (4 Opus, merged 7c04584): cmd/lem 18.7→85.4% (25→133 tests,
  every verb family); train 89.4→91.1% + sftSampleText 8→1 allocs
  (928→240 B/op, per-sample-per-epoch); serving policy MATCH path 7→3 allocs
  (clean path 0-alloc now bench-pinned); engine-neutral 80.3→82.2% (six 0%
  fns). TWO honest premise-corrections: my bench-presence instrument was
  broken by zsh glob noise — engine + policy were ALREADY benched; both
  agents verified reality and refused theatre benches.
- FLEET ROUND 2 (4 Opus, merged 3b43573): train2 — appendSidecar −37% B/op
  (named lead confirmed), lora multi-shard hash −63% B/op, distill cache-key
  byte-identity pinned; serve2 — five named 0% gaps with real fakes
  (sessionkv route-drift gate, continuity fake session factory), kv floored
  97.3%, one perf hypothesis benchstat-FALSIFIED and reverted clean;
  modeldeep — gguf metadata 41→97%, tar-slip guard tested with hostile
  containers, ggufLoadTensorData corruption guards; decodedeep — spec-decode
  path evidenced at the floor (0-1 inherent allocs), redaction-gate slice
  descent leak-pinned, json.Number grammar parity 100%.
- MY FINDING CLOSES (e1c1de1, 61a6725): decode/parser Flush's residue
  branches removed after independently verifying drain(true)'s postcondition
  (holdback only arms mid-stream); TestProcessorDrainFinalConsumesPending
  pins it. Orphaned Err doc comment rehomed.
- PARKED DESIGN CALLS: safetensors Parse (reflection, 1819 allocs/shard) vs
  parseHeaderInto (6-8) unification — divergent validation semantics
  (zero-dim, dtype case), wall impact at load is small; eval ScoreAll
  fake-judge harness; DuckDB ingest fixtures.
- Round-close gates: 9,537 CPU (122 pkgs) + 1,550 metal green.
- GATE-SCRIPT SCAR: serve boot pings need ≥120s timeouts (cold first request
  pays welfare probe + PSO warmup); kill stray serves BY PORT
  (lsof -ti:PORT | xargs kill -9) — go run children live in the build cache,
  name-pattern pkill misses them.

## HEATWAVE ROUND (2026-07-12 midday — Opus fleet rolling; composed lands for real)

- EMBED PAYLOADS REFRESHED (79ed05e): the tracked cmd/lem/*.metallib.gz had
  drifted from build/dist (a fresh-clone `task build:embed` baked stale
  kernels). Smoke: bin/lem 154M runs generate + serve with NO
  MLX_METALLIB_PATH. /bin/ gitignored. Release pre-flight DONE — snider's
  edit pass on docs/release-v0.12.0-DRAFT.md is the only step left before
  tag (rebuild binaries at tag time; today's bin/ predates 886d2b7).
- COMPOSED LOADS REAL PACKS (787ada6): Qwen3.5-0.8B-OptiQ-4bit (867MB,
  hybrid 18 linear + 6 full attention, qwen3_5) pulled to the HF cache;
  LoadComposed gained language_model.* normalisation
  (model.NormalizeWrapperNames), mlx-affine host dequant
  (mlxaffine.DequantizeTensor; per-module bits/gs overrides honoured,
  packed-shape cross-check fails loudly), and the mlx [convDim,K,1]
  depthwise-conv shape. 3 loader pins added. Loads + greedy-generates on
  engine/metal (host-f32 mixers ~1.6 tok/s — composed GPU speed is future
  work, by design).
- #379 LIVE GATE PASS (#10 closed): two-turn serve -state-conversations on
  the real hybrid — turn 2 recalled turn 1's fact AND prompt_tokens 25 vs
  29+ full-history replay (wake + append-only proven live). Gate script:
  /private/tmp/lem-dev/composed_gate.py.
- TOKENIZER CHATML PARITY (886d2b7): the gate's two leaks root-caused to
  ENCODE defects, not templates — <|im_start|> was wrongly mapped as a BOS
  (ghost im_start at the head of every continuation → 'assistant' echo on
  woken turns) and only special:true added tokens joined the atomic
  matcher, so qwen's special:false <think>/</think> BPE-split into text
  (the model never saw its pre-closed think channel → reasoning leaked
  into content). Fix: added = the full atomic matcher, special = the
  decode-side skip; qwen BOS mapping removed with rationale. Receipts:
  Go == HF token-for-token on the real qwen tokenizer (19-id continuation
  identical); gate turn 1 36→4 tokens clean 'OK Wibble', turn 2 clean
  'Wibble'. Gemma blast radius zero (no special:false added tokens in
  gemma packs; <bos> untouched): engine/metal 1547 green, tokenizer 123
  green (+2 pins). <think> is already in decode/parser's paired reasoning
  markers, so think-ON requests now split into reasoning_content properly.
- G2 DEFAULT-MEDIATOR DECISION (deferred item closed): cmd/lem ships NO
  default mediator — a rewrite-rule policy without a wired mediator
  refuses to boot (loud + correct, serving/serve.go:111); redact/refuse
  policies work with plain -policy.
- FLEET (Opus 4.8, worktree protocol): docs drift sweep MERGED (2c08372 —
  lem verb/flag surface, campaign env knobs LTHN_SDPA_GEMM_MINKV /
  LTHN_FLASH_WIN / LTHN_GPU_TRACE=host, MLX pin v0.32.0 confirmed via
  gitlink, 8 files). QA honest-tests MERGED (570dd34 — gguf
  dequant-on-load path 20 tests, train/tune 0→100%, welfare, eval bits;
  the 28-flag fake-coverage class adjudicated as analyser false-positives
  on 4-slot scenario names — honest tests left alone; analyser-upgrade
  noted as the real fix).
- AUDIO LANE LANDED (fded3d4, Mantis #1839 note 2487): E2B/E4B Conformer
  audio LIVE on metal. Both suspected parity defects were REAL and are
  fixed — output_proj.bias (max|abs| 14.875) was silently dropped;
  real-pack OHWI convs were double-scrambled by the torch-OIHW assumption
  (shape discriminator now routes both). Serve seam wired (audioExtractor
  held at load; ProjectAudio Conformer bytes-in branch) — CLI -audio +
  OpenAI input_audio light up with zero serve changes. HF harness: mel
  golden max|Δ| 4.768e-7 (the go-mlx 1-ULP bar), tower goldens subsample
  0.999999 / layer0 1.000000 / tower 0.999995 cosine (supervised — skips
  without the cached e2b). E2E receipt reproduced independently in the
  main tree: say→afconvert→lem generate -audio transcribes "the quick
  brown fox jumps over the lazy dog" EXACTLY at 128.7 tok/s decode. NB
  the ASR prompt from capabilities-audio.md is required — generic
  "transcribe this" truncates (prompt-shape, not code). Deferred: D
  (batch/pad mask fidelity), E (HIP port — codex's lane). Post-merge
  gates: engine/metal + model/gemma4 1,631 green.
- dev pushed through fded3d4 (d412158 → 79ed05e, 787ada6, 2c08372,
  570dd34, 886d2b7, 8c828a9, audio 3c614b3..322590b, fded3d4).

**The kv-shared layer skip is BUILT, receipted, and pushed (473c242).**
sharedLayerSuffixStart validates the non-owner suffix at state build;
prefillRetainedTokensBatchedDenseChunks arms prefillSkipToLayer on
non-final chunks; the batched pass bounds its layer loop. Receipts (e2b
4bit, correct metallib, A/B vs LTHN_PREFILL_SKIP_SHARED=0 same env):
pp8K 3,190->6,777 · pp32K 2,753->7,008 · pp62K 2,323->5,833 · pp118K
1,688->4,249 tok/s; 32-token greedy continuations byte-identical at 8K
and 62K; TestArchSessionPrefillChunksSkipSharedSuffix pins serial-vs-
chunked byte identity over a kv-shared fixture both lanes. Field position
now: beats llama.cpp everywhere, edges oMLX at 8K (6,777 vs 6,696), leads
the field outright at 32K+; mlx-lm true-wall gap 3.2x -> 1.48x at 8K.
SECOND LEVER SHIPPED (c257ff0): minimal boundary chunk — with the skip
armed, only the final chunk pays the full stack, so the absorb policy's
1081-row final chunk shrank to the last partial window (57 rows at 8K,
floored at 32 to keep the ICB/q8-fold gates clear). pp8K 6,777->8,049.
Chunk-width RE-RECEIPTED under the skip: 2048 stays the peak (4096
still collapses: 4,986).
THIRD LEVER SHIPPED (a0364b5): PLE slab at owner-layer width + the
one-dispatch quant gather (lthn_ple_gather_rows_quant; K-loop of
per-token gathers retired; builders derive the bound from slab length).
pleSlab host span 104->47ms steady-state; pp8K 8,049->8,528 (0.847s
wall; mlx true-wall gap now 1.18x). NOTE: first run after a metallib
rebuild pays the new kernel's PSO compile once (~40ms) — measure
steady-state.
FOURTH LEVER SHIPPED (cfc84d5): device-resident PLE slab handoff — the
builder commits WITHOUT waiting and the pass binds its buffer directly
(same queue = GPU-ordered; single-buffered scratch safe because the host
stages the next chunk only after the pass's wait). pleSlab host span
47->3.1ms; pp8K 8,528->8,708 (0.830s; mlx true-wall gap 1.15x). The
builder's GPU work (~2ms/chunk) still serialises ahead of the main pass
on the shared queue — a second queue + MTLEvent could overlap it, worth
~1-2% more, diminishing.
FIFTH LEVER SHIPPED (9e5846d): device embed gather — the builder's CB
gathers the K main-embed rows too (same rows kernel at dModel width),
the projection reads them in place of host staging, and the pass takes
the same buffer as its input rows. Only token ids cross to the GPU.
embed span 17.6->0ms; pp8K 8,708->9,016 (0.801s; mlx gap 1.11x).
NIGHT TOTAL pp8K 3,190 -> 9,016 (2.83x), token-identical at every step.
MORNING AFTER (2026-07-12) — the family round:
- E4B RECEIPTED (zero code): 18-of-42 kv-shared -> the skip engages
  by construction. pp8K 1,987->3,423 (1.72x, off=LTHN_PREFILL_SKIP_SHARED=0),
  32K 3,182, 62K 2,807; 32-token greedy identity at 8K AND 62K.
- DENSE/MoE EMBED GATHER PORTED (c470d19): the rows kernel gathers a
  whole chunk's embed rows in one committed-not-waited CB; dense archs
  have no PLE so the closure hands back (embBuf, nil). 31B qat 289->290
  (embed span ~71ms->0), 26B-A4B 1,440->1,447, 12B ENGAGES (inputsDev
  spans replace host embed; ~0.5% under bench noise). e2b-qat unchanged
  at ~9,250 (E-series untouched). Bytes identical pre/post on ALL FOUR.
  Gate: TestEmbedRowsBatchQuantDeviceMatchesEmbedTokenQuant (host-oracle
  byte identity 4/8-bit) + decline-contract test.
- 31B mlx GAP MEASURED: mlx-lm true wall 22.16s vs our 24.99s = 1.13x —
  same shape as e2b's 1.11x. The gap is SYSTEMATIC across archs; the 31B
  chunk trace says GEMM-bound (mlp gate/up/down 60%, qkv 16% —
  proportionate to FLOPs at head_dim 256, checked), sdpa 11.6% -> the
  likely gap home is the flash prompt SDPA (#375's campaign).
- WHOLE-REPO metal_runtime SWEEP NOW GREEN (10,754 tests / 122 pkgs):
  train's command tests/examples pinned their fake backend (514297e —
  SSDCommandConfig/SFTCommandConfig grew Backend -> WithBackend; under
  metal_runtime the metal engine had been winning selection and
  genuinely loading fixture paths).
AFTERNOON (2026-07-12) — #375 flash routing + the Opus fleet round:
- THE 8K CELL FLIPPED: e2b pp8K 10,048 tok/s (0.718-0.723s ×3, quiet
  machine) vs mlx-lm true wall 0.730s SAME conditions — lem now leads
  the measured field at EVERY depth. 32K 7,849->8,038; 62K/118K
  unchanged. Two levers, both found by the NEW per-lane sdpa trace
  (attn.sdpa -> .win/.gemm/.mq, permanent instrument, d8f8e24):
  1. GEMM knee 4096->2048 (d8f8e24): chunk-1 globals rode the multiQ
     vector kernel at 30.7ms where the composition runs deeper spans in
     17.6ms. LTHN_SDPA_GEMM_MINKV overrides live. pp8K 9,315->9,641.
  2. Window-flash occupancy cliff (e9e0935): one TG per (BQ-tile,head)
     = 16 TGs at a 57-row boundary chunk — 35.7ms vs the ring kernel's
     2.4ms (15x). flashWinMinRows=1024 gates small chunks to ring
     (crossover receipted: 484->ring 1.7x, 1024 tie, 2048 flash 1.5x).
     pp8K 9,641->10,058.
  NUMERIC TIER, both levers: re-routed chunks change accumulation
  order -> greedy forks at near-ties observed (both branches coherent).
  Same tier the GEMM/flash lanes already trade at; kill switches:
  LTHN_SDPA_GEMM_MINKV=4096 / LTHN_FLASH_WIN=0.
  BANKED NEGATIVE: BQ16/BK32/WM2 steel shape (halve rescale/barrier
  tax) ran 18% SLOWER — halving BQ doubles Q-tiles and each re-reads
  its whole K/V span; the kernel is bandwidth-bound, not rescale-bound.
  Reverted. (Also: the win flash at BD256 is ALREADY the max legal
  shape — BQ32/BK32 blows the 32KB TG budget.)
  mlx anatomy brief (Opus research, /private/tmp/lem-dev/
  sdpa_anatomy_brief.md): mlx has NO fused attention at head_dim 256
  (use_fallback fuses 64/80/128 only) — it runs the unfused composition
  with a materialised mask tensor; our fused BD-256 steel port + window
  flash are structurally AHEAD; mlx-lm caps sliding KV via
  RotatingKVCache(512). NAX is an M5-era feature — nothing hiding on M3.
- OPUS FLEET MERGED (d566736, 4 agents, worktrees cleaned):
  #378 G2 mediated rewrite SHIPPED (5 commits): rewrite action +
  streaming mediation (degrade lattice refuse>redact>rewrite, mediator
  timeout per span, original span can never leak), WrapResolverMediated,
  ServeConfig.PolicyMediator boot-fatal seam. 0 B/op clean path.
  FOLLOW-UPS parked: mediator output is emitted verbatim (add a
  non-recursive re-enforcement pass if the mediator is untrusted);
  cmd/lem ships no default mediator (rewrite policies boot-fatal from
  the CLI until one is injected); audit lacks a Degraded flag.
  #379 composed -state SHIPPED: token-prefix kv.Snapshot (a stateless-
  replay session's complete state IS its prefix; restore re-prefills,
  deterministic host-f32 recomputes identical recurrent state).
  generate -state + snapshot-strategy wake now work for composed.
  PARKED: RangeKVBlocks (serve -state-conversations sleep lane) —
  needs trusted-prefix block tiling + a LIVE multi-turn serve gate;
  also confirm serve degrades gracefully when composed sleep declines.
  #381 vision/bidir lane: skip is structurally VALID there (consumer
  reads only boundary hidden — the "embeddings lane" is the VISION
  lane; the pooling forward is engine/hip's) but UNARMED pending a
  real unified-vision receipt; prefillSkipToLayer pinned 0 at entry
  (load-bearing: no per-chunk reset there) + guard test.
EVENING (2026-07-12) — the scraps round (fleet ×4 again, all merged):
- SEAMS + 2ND-QUEUE: BOTH CLOSED NEGATIVE, receipted (e97056d). The new
  LTHN_GPU_TRACE=host mode (host spans at production GPU fidelity — no
  segment-splitting tax) + chunk/chunk.step spans show chunk==chunk.step
  within 0.1ms on every chunk at 10,039 tok/s UNDER the instrument: the
  #381 device-input levers already ate the host seams. The 2nd queue was
  FULLY BUILT (builder queue + MTLEvent + double-buffered scratch pairs
  + one-ahead pipeline), held byte identity, measured 10,053 vs 10,051 —
  parity; the builder's GPU work is ~0.2ms/chunk post-#381 (the ~2ms
  estimate was stale). Machinery reverted; instrument kept.
- #380 LIVE RECEIPT CLOSED: served e2b decode 162.6/165.4/165.1 tok/s
  (512-tok completions, temp0, think off) through the full HTTP +
  streaming path — at/above the generate-measured board's 161.3. No
  streaming tax; the emittedContent fix stands end-to-end.
- G2 HARDENED (be440c0, merged c486c9c): mediator output re-enforced
  (one non-recursive pass; residual hits — INCLUDING refuse — degrade
  to redact, stream never killed by mediator output; mediator called
  exactly once) + audit Event.Degraded flag. Clean path still 0 B/op.
  Contract change: span-echoing mediators now get echoes redacted.
- Mantis #1840 FIXED (148044d, merged 9435b26): jsonSkipValue fixed
  bracket pair -> LIFO closer stack ([16]byte local, no heap on shallow
  metadata); 9 pinned heterogeneous-nesting cases; sibling audit clean.
- #379 COMPOSED BLOCK SLEEP, CPU HALF (1403fd4, merged 9f412b1):
  RangeKVBlocks streams token-only blocks on ArchSession's exact tiling
  contract (uniform grid, absolute contiguous Index/TokenStart,
  BlockStartToken skips whole blocks, graft alignment); multi-turn
  re-sleep vs trusted parent bundles CPU-tested; serve degrade confirmed
  already graceful (finishTurn logs + stays RAM-resident). LIVE GATE
  STILL OWED: multi-turn `lem serve -state-conversations` on a real
  composed checkpoint — NONE CACHED locally (needs a Qwen3-Next-class
  download decision).
- AUDIO TOWER RECON (Mantis #1839, /private/tmp/lem-dev/
  audio_tower_brief.md): THE TICKET IS STALE — the full Conformer
  already lives in engine/metal (ported from go-mlx in b142528:
  encoder/attention/subsample/mel extractor/assembly all real). Actual
  gaps: (A) ~40-80 LOC serve wiring (build+hold the feature extractor at
  load; Conformer branch in NativeTokenModel.ProjectAudio — today only
  the 12B-unified raw-waveform path exists, so E2B/E4B audio gates true
  then fails); (C) two probable parity defects — output_proj.bias
  [1536] dropped by AssembleAudio/AudioEncode, and audioConv2dToOHWI
  may double-transpose mlx checkpoints' already-OHWI conv weights
  ("wrong = garbage audio", confirm empirically first); (B) HF
  fixture-parity harness gates it. HIP port = codex's lane.
FOLLOW-UPS still open on #381: (none — seams + 2nd-queue closed above). 26B/31B receipts DONE;
12B unified receipts ride the same lane. #375 remaining slack: the
win lane at big rows (~26ms/chunk vs ~15ms physics) and the global
flash lane (67ms vs ~30) — genuinely diminishing; next likely lever is
structural (fewer dispatches per chunk / 2nd queue), not tile shapes.

**THE TRAP THAT ATE AN HOUR (bank it):** running engine/metal tests with
go-mlx's dist metallib (the retired path my own notes carried) makes the
sibling lthn_kernels.metallib STALE -> missing kernels
(`lthn_embed_gather_row_bf16 not found`) -> a 40-test parity red wave that
reproduces at KNOWN-GREEN commits. The correct pair is
**$REPO/build/dist/lib/mlx.metallib** (the Taskfile var; lem.sh already
defaults to it). If a mass parity red appears at a green commit: read ONE
failure message before bisecting — mine said "kernel not found" all along.

**The solved mechanism (for the record):** mlx-lm's prompt loop evals ONLY
cache states; gemma4 E-series KV-SHARED layers (e2b: 20 of 35) own no cache
state, so lazy DCE skips their entire compute on every non-final chunk —
architecturally correct (deep outputs of prompt positions feed only unsampled
logits; later attention reaches prompt tokens via OWNER-layer KV). Kill
shots: width-halving = 0ms delta; powermetrics energy forbids the phantom
45TF. Caveats held: Classify all-position logits do NOT skip (they take the
non-chunked lane); MTP boundary hidden comes from the final chunk (full
stack).
ALSO LANDED OVERNIGHT: #380 emittedContent quadratic (871x B/op at 8K,
45,530-case oracle fuzz — live tok/s A/B still owed); #378 outbound policy
G1 (term 0-alloc + bounded-window patterns, boot-fatal, outermost wrap; G2
mediated-rewrite designed-next); #379 serve wiring (composed models were
NEVER servable — metalBackend hard-asserted native; now bridged + registry
routing + ChatML declaration with precedence golden; composed live-serve
needs a metallib smoke; multi-turn -state for composed = follow-up).
Port DECIDED: 36911 stays.

# PREVIOUS WAKE (2026-07-16 early hours — the capture night)

**#381: eleven acquittals and a localised mystery.** The prefill gap vs mlx
is NOT: host, q8, chunk width, hazards, the metallib binary, bf16 emulation,
clocks, data values, PSO options, CB overlap, or weight allocation — every
one receipted (instruments at d65e96b, full chain in #381). Metal System
Trace with MLX_MAX_OPS_PER_BUFFER=1 shows their real forward's fat qmms at
~1.7ms each (≈45 TFLOPS, serial, no overlap, no gaps — bucket-count fit is
decisive) while the SAME op in ANY isolated context (our Go bench, their own
runtime, their real weight tensors) runs 20.5-22.9. NEXT TOOL: GPU counter
capture (ALU utilisation forward-vs-burst) — traces saved at
/private/tmp/lem-dev/{mlx_prefill,mlx_perop,mlx_burst}.trace + harnesses.
BUILD REGARDLESS: our in-situ 12.9 vs our benched 22 = 1.7x from our own
pass first (sdpa 2.7TF@win512, launch-bound elementwise, GEMM 14-vs-22).
ALSO TONIGHT: lem quant SHIPPED (319ddc4 — bf16 in, BYTE-IDENTICAL-to-
mlx_lm.convert variations out, oracle in-tree; bits 2/4/8, -gguf lane);
convergence campaign closed at ~50 wins (decode 5-for-5, kv sliding-wake
quadratic, eval scanners 10-26x, engine render 20->1); field matrix +
concurrency + tool-calling receipts in the release draft; first-impressions
audit merged (port decision PENDING WITH SNIDER: 36911 vs 11434).

# PREVIOUS WAKE (2026-07-15 session close — the convergence campaign)

**The folder-by-folder Opus convergence campaign ran all day** (snider's
rule: single-folder passes, first defended zero retires a folder, wins
re-queue). ~45 gated wins landed across 11 merge waves, dev pushed at every
step. Headlines: kv/ multi-head wake was O(N^2) (hidden by heads=1
fixtures; 16-block wake 39.6MB->4.2MB); both per-turn scorers were mostly
regexp (ScoreHeuristic halved); ngram drafter 6.7x on the miss path (CPU
axis, was 0-alloc); model.Sampler full-sorted the 256K vocab per token
when top-k/p engaged (19.6x fixed — NOTE: metal serve lane samples
engine-side, live A/B 136 tok/s both sides, so primitive-level not
serve-headline); welfare Detect/Guard now 0-alloc/turn; ProjMatMulInto
seam threads projection scratch through all hybrid forwards (race-proven
ownership model); engine render loop 20->1 allocs same day it landed.
FOLDER LEDGER — retired on defended zeros: serving/, safety/, jsonenc/,
internal/, root pkg, train/, agent/, gguf/, merge/, state/filestore.
Won-and-owing-a-pass: decode/ (4 straight), kv/, eval/, welfare/, engine
root, safetensors/. Locked: engine/hip (codex, ~1 month). Mine:
engine/metal. ALSO SHIPPED between waves: Qwen 3.6 model layer (42feca3 —
gated attention is SIGMOID per reference, registry path via
ArchSpec.Composed; remaining: serve wiring + fused-expert reshape at
smoke + GPU kernels, see #379) and the chat-template lift (a9d20c2 —
ChatTemplateDeclarer, gemma byte-identical fallback, ChatML proven via
fake; qwen declaration is now a one-liner at serve wiring). Traps 8+9
(Go 1.26 Sprintf already 1-alloc; option-closure collapse forces whole-
struct escape) banked in lethean-perf. #378 outbound policy spec awaits
the plans-repo write (a guardrail blocked it; snider ok'd moving on).
#380 openai emittedContent O(N^2) still needs the live-model pass.
Slur catalogue ships EMPTY (hostility axis does live triggering) —
snider-curated population pending, note for release draft.

# PREVIOUS WAKE (2026-07-15 session, later)

**#376 shipped and receipted live.** The welfare guard was built-but-unwired
(welfare.Guard/Mediate zero callers; the pipeline that wired Detect has zero
non-test callers itself — the release draft over-claimed and is now trued).
7acc246 wires it where requests actually flow: a TextModel decorator at the
serve resolver fronts every chat route; mediation runs on a fresh
meta-session against the inner model; lem_ok/rephrase/pause applied per
spec; lem_end lands as the fourth rung, Lemma-gated. -welfare default ON in
cmd/lem. LIVE receipt: four escalating hostile turns at a served E2B — the
model rephrased (warn=true), paused (🍵 reached the client, abuse never
reached the conversation), rephrased again; audit lines in the serve log.
Remaining rung: persist ended-state per conversation under continuity.
BOTH bench agents merged + worktrees cleaned: A (serving lane — the
ThinkingExtractor quadratic kill 124704→4472 B/op per 200-tok stream,
conversationKey 13→2 allocs) and B (model cluster — MoE forward 9→5,
gguf Quantize ~20→1, 46 bench files). Suites: 2052 (serving+welfare+cmd)
+ 3197 (model+agent) green.

# PREVIOUS WAKE (2026-07-15 session)

**#377 closed with a lever + a bug kill.** Snider read the release energy
table and asked why pure replay climbs while llama-server stays flat. Answer:
the stateless lane opened a fresh session per request (full re-prefill every
turn + a maxLen-sized KV alloc/free round trip — also why 31B feels heavy on
short chats) while TWO complete LCP reuse implementations sat in-engine
caller-less. Shipped 64688fb: resident-session prompt reuse on the stateless
lane (llama slot parity; TryLock single slot; stands down under continuity;
`LTHN_PROMPT_REUSE=0` kills). Receipt: 10-turn stateless walls 3.1→4.4s now
FLAT ~3.1s with prompt_tok climbing to 3,959. Found + fixed en route
(300eb33): every prompt-cache rollback (pair lane prepareAssistantPrompt,
GenerateCached×2) rolled back past wrapped sliding RINGS — silent wrong
context on MTP multi-turn; all sites now degrade to cold prefill
(TestPrepareAssistantPromptRingSafety). Release draft + energy-ab README
updated (replay arm now needs the kill switch to reproduce). TWO OPUS BENCH
AGENTS dispatched on worktrees perf/bench-a (serving/decode/kv/welfare/root)
+ perf/bench-b (model/agent) off 9af89e9 — adding missing
{file}_bench_test.go + allocs/B-op cuts; merge their branches when done, then
`git worktree remove` + prune (.worktrees/goinf-bench-{a,b}).

# PREVIOUS WAKE (2026-07-13 session close)

**The release is on the pad.** `docs/release-v0.12.0-DRAFT.md` is complete —
name (lem = Lethean Ethical Machine, LEK→LEM→lem lineage), the THREE-WAY
measured energy receipt (continuity 77.1 J/turn vs llama-server 112.1 vs
replay 143.9; 21.4/31.1/40.0 kWh per million turns; harness at
docs/examples/energy-ab/), decode board, 256K receipt, flash kernels,
welfare guard (go/welfare — already built, wired bidirectionally; #376 =
just the lem_end rung), falsified-and-kept, video storyboard. Remaining
before tag: Snider's edit pass, `task build:embed` binaries, tag
(GitHub-first or after forge returns), the side-by-side video.
**forge.lthn.sh is DOWN deliberately (CoreAgent phase work)** — plans repo
has TWO LOCAL UNPUSHED commits (6c50e77 dreaming tier §13, 05d5b1f
field-first + register view); push forge_sh main when it returns.
Flash campaign (#375): 256+window shipped default-on (+4% all depths);
split-D and q8-flash opt-in with falsification receipts in metadata.

---
title: Handover
description: Working notes from the engine-perf campaigns — for the next driver of this codebase.
---

# Handover — from the previous driver

You're inheriting a working engine. Everything below is the part that isn't
derivable from the code: how it got to this state, what was falsified along the
way, and the traps that cost real hours. Trust the receipts in `git log` over
anything here; the task tracker carries the live campaign state.

## State of the board (2026-07-10 evening, M3 Ultra 96GB, tg512, ALL-DEFAULTS: q8 KV + 128K ctx + -draft auto)

| model | plain | +MTP | qat plain | qat+MTP | notes |
|-------|-------|------|-----------|---------|-------|
| E2B | 161.3 | **207.3** | 136.1 | **188.4** | trainer workhorse; PLE arch |
| E4B | 112.1 | (no asst) | 90.4 | **101.0** | non-qat assistant not cached |
| 12B | 69.1 | **97.1** | — | — | bf16 drafter; beats the old board's 93 |
| 26B-A4B | 139.8 | (no asst) | 133.0 | 31.5 LOSS | MoE verify cost — loss now QUANTIFIED (−76%) |
| 31B | 32.5 | **47.4** | 22.1 | 19.1 LOSS | orchestrator; qat lanes are the anomaly |

Generate-measured (decode-only, greedy, log-continuation prompt); the old
serve-bench board read a few % higher on plain lanes — harness delta, not a
regression. **The qat tax is CLOSED (#374): not a kernel path** — the qat
checkpoints ship their entire FFN at 8-bit (all 35 layers' gate/up/down at
bits=8 in the config overrides), and the slowdown is byte-proportional to
the percent (e2b 1.21× bytes → 1.19× slower; 31B 1.56× → 1.47×). qat is a
quality tier priced in bytes. llama-bench calibration (E2B tg512, llama.cpp
build 9860, same box): Q4_K_M 144.0, Q8_0 124.0 — **lthn leads +12% at the
4-bit class and +44% with MTP** (llama-bench has no speculative lane; note
unsloth's GGUF repo now ships an MTP drafter — the ecosystem is adopting
gemma-4 MTP). Remaining open: the 26B MoE MTP loss magnitude (−76%).

Depth (e2b, `generate`-measured): linear −0.61 tok/s per 1K context, no cliff;
~105 at 98K. Cold prefill is the deep-context pain (54s at 98K) — conversation
continuity (`serving/continuity`, shipped) makes it once-per-conversation:
turn 2 of a 10K chat is 0.4s vs 5.5s, book-bench prompt column flat at ~88
tok/turn while the client resends full history.

## The method (this is the important part)

1. **Instrument first.** Before forming any view, build or run the
   measurement: the anatomy bench (`LEM_SDPA_ANATOMY=1`), the MTP diag
   (`LTHN_MTP_DIAG=1`), the confidence capture (`LTHN_MTP_CONF=<path>`), or
   a plain live A/B. Every productive result this engine has came from an
   instrument saying something surprising; every wasted hour came from a
   theory that skipped the instrument.
2. **No receipt, no claim — and no keep.** If you build an optimisation and
   the live A/B doesn't show it, revert it *even if the design is beautiful
   and the suite is green*. It has happened here: a full cache rewrite,
   suite-green, reproducibly slower at 32K — reverted the same night, lesson
   banked. The discipline is what keeps the tree trustworthy.
3. **One lever per commit, receipt in the message.** Future-you greps
   commit messages for numbers. Make them greppable.
4. **Kill-switch anything adaptive.** Wall-clock-adaptive policies (the MTP
   re-engagement gate, the dynamic draft cap) make runs non-reproducible by
   design. Their env kill switches (`LTHN_MTP_REENGAGE=0`,
   `LTHN_MTP_DRAFTLEN=0`) are the reproducibility anchors — never ship an
   adaptive behaviour without one.

## The open campaigns (task tracker)

- **#372 verify-ICB (closed — built, measured, FALSIFIED at break-even)** —
  the tail-only replay shipped opt-in (`LTHN_VERIFY_ICB=1`, default off):
  live verify fwd 10.3-10.9ms vs replay 10.8-11.2ms. Two banked findings:
  recordings must be **per-K** (the adaptive draft cap wobbles block width
  5/6/7 — a single-K ICB never replays), and at ONE executeCommands per
  14-op layer tail the execute + recorded-barrier-drain tax eats the
  recorded-op savings — the chained lane's ~3µs/op economics need ONE
  execute per pass. The remaining shape is whole-layer recording
  (attention + tail, pos/N rebinds), blocked on the staged-sliding landing's
  per-pass slot offsets. Also from this campaign: `c06cc8a` fixed
  `serve --draft` (dead since 601ac4e — LoadDir passed maxLen 0 through).
- **#367 q8 KV (dense/ICB lane)** — BUILT, opt-in (`LTHN_KV_Q8_ICB=1`),
  receipted on e2b: global layers store int8+scales on the recorded ICB
  (`0103bd9` kernels, `b41ce28` wiring). Depth receipt: 16K +3.9%, 32K +2.8%
  decode; slope −0.70→−0.52/1K — i.e. ~half of e2b's depth slope IS the
  global-KV bytes (now halved), the other half is #365's non-KV class. Zero
  shallow-depth cost. V1 tax: batched prefill + MTP verify + KV snapshots
  DECLINE on q8. Slice C (`bd1e314`) made the batched fold q8-aware
  (stage + `lthn_kv_q8_store_rows` landing, `lthn_sdpa_multiq_q8` read):
  prefill 146s→14.3s @32K (bf16 batched 8.2s — the gap is the GEMM lane
  forced to multiQ on q8), MTP verify engages at shallow depth. Still
  declining: KV save/restore, per-row shapes, the deep+small 2-pass corner.
  12B receipt CALIBRATES the campaign: decode 16K flat, 32K −2.5% — on
  compute-dominated tokens the current int8 read overhead ≈ the bandwidth
  saved; e2b gains +2.8-3.9%. `-state` works under q8 (`ee8632a`): the bf16
  snapshot mirror makes sleep/wake BYTE-IDENTICAL and .kv portable both
  directions (+0.9s wake at 16K = host mirror loops). The read-cost
  instrument (`TestDiagQ8ReadKernelCost`) shows the q8 2-pass at 105 GB/s
  effective vs bf16's 246 — char4 vectorisation FALSIFIED (no change); the
  cause is strided small-visit DRAM efficiency + the separate scale line.
  **THE TARGET STANCE (Snider): MTP + q8 + 256K context becomes the DEFAULT
  once q8 completes** (q8 ≈ bf16 at ~1%; q6 −6% / q4 −22% stay user-choice).
  The 2026-07-10 fresh-binary 2×2 (e2b-4bit @16K, HIGH-acceptance
  log-continuation prompt, greedy) rewrote the previous day's partial
  receipt — that 47.6-vs-112.9 framing was wrong twice over: (a) the q8+MTP
  loss was the DRAFTER's target-KV export host-dequantising the q8 mirrors
  every draft block (kvExport 108.9ms vs bf16's 0.4); the GPU mirror
  dequant/quant kernel pair fixed it (kvExport → 2.6ms steady, q8+MTP 85.6 →
  125.6 tok/s = bf16+MTP's 128.0 at the same 44% acceptance); (b) q8 plain
  decode @16K is at PARITY (152.0 vs 151.0), so the read-format fix does NOT
  gate MTP+q8. The read-efficiency micro-fixes are ALL FALSIFIED by
  instrument (2026-07-10): chunked key walk 105→95 GB/s (worse — the
  strided TGs already cooperate on aggregate), scale-stream stubbed to 1.0
  gives 105→107 (the separate scale plane is FREE), char4 already dead —
  the q8 2-pass is per-iteration ISSUE-bound; a fat-iteration redesign is
  the only shape that would move it, and the live matrix says it is NOT
  needed: q8 decode holds PARITY at 16/32/64/124K (clean 124K pair 100.6
  vs 103.2 — an earlier "−11% deep decode" was a pressured-machine
  artefact; depth receipts want a quiet machine and full decodes, never
  8-token samples). The q8 PREFILL tax (1.5→3.6× with depth, 307s vs 85s
  at 124K) was the GEMM lane declining q8 — fixed by the q8 GEMM prefix
  (`4751f9a`): the owner dequantises its attended prefix into the layer's
  snapshot mirrors in-encoder and the steel GEMM reads the mirrors.
  Prefill now 1.03-1.04× bf16 at every depth (9.9/26.9/70.8s at
  32/64/124K). MTP-at-depth economics: MTP@16K loses on BOTH formats
  (127-128 vs ~151 plain) even at 44% acceptance — draft AND verify pay
  the depth scan — and the re-engage gate correctly oscillates it off,
  which is exactly the adaptive default the stance needs. q8 is now at
  bf16 parity on every measured axis (decode, prefill, MTP, -state) while
  halving global-KV bytes. **THE DEFAULT LANDED (`ae0636e`, RAM-corrected
  `df61abf`)**: q8 KV on by default (kill switch `LTHN_KV_Q8_ICB=0` — the
  bf16 A/B anchor), defaultContextCap 32768→131072 (unset context follows
  the checkpoint window to 128K; e2b runs its full window), `-draft auto`
  already the serve default. Live receipt with ZERO flags: e2b 124K
  prompt → 104.6 tok/s decode, 69.6s prefill (the identical command
  errored at the old 32K cap). The 256K default was tried and PULLED BACK
  the same day: a 31B all-defaults 250K run hit a 64.9GB footprint
  (+19.5GB swap) on the 96GB box — weights ~17GB + q8 KV 8.6GB + 17GB
  GEMM-prefix mirrors + ~19GB UNATTRIBUTED. Mirrors now free at the
  prefill→decode seam (`df61abf`; -state and the drafter export
  re-materialise per-layer). The census then CLOSED the same evening,
  statically: the "unattributed" ~19GB is the DUPLICATE CACHE SET — the
  layer-buf builders allocate bf16 lb caches at maxLen for every owner
  (decode_forward_arch.go:1856 / _quant.go:345) SEPARATE from the ICB
  replay's own caches, and the batched pass prefers the ICB set when
  present (decode_batched_session.go:961) — 17.2GB allocated-but-idle on
  31B@256K — plus the sdpaPromptS S-slabs (2 × rows×maxLen bf16, whose
  growth realloc also drops old slabs unreleased).
  **THE DEDUPE SHIPPED** (`13ffe18`, 2026-07-11): session builders defer
  the lb KV allocation (kvCacheBytes recorded, buffers nil) and the four
  lanes that genuinely read lb ensure it on first touch (stepToken top,
  non-ICB batched entry, the paged↔linear bridge, the state-view lb
  branch) — a recorded-ICB session now never materialises the set, and
  `TestICBSessionDefersLBKVCaches` pins that through build/prefill/
  decode/serialize/restore. Care-map that proved it: every batched site
  overrides to icbK/icbV, declines fall to the per-token replay,
  prefillCachedIDs' lb fallback is gated icb==nil, state views branch to
  icb/q8-mirrors first, and the sequential encoder prefers the PAGED
  pool — the only lb consumer on a real session is the LTHN_DECODE_ICB=0
  lever (tests set it around fresh builds; ensure serves it).
  HONESTY NOTE on the census number: the dup was CLEAN-but-mapped GPU VA,
  not dirty RSS — e2b@128K receipt: IOAccelerator(graphics) VA 5.6G→4.1G
  (−1.5G, regions 1183→1133), physical peak only 4.9G→4.8G, resident 4.1G
  both, decode 155.9→155.3 tok/s + prefill 1076→1077ms (parity), suite
  1507 green. So the 31B@256K swap driver was NOT the lb dup (clean pages
  don't swap) — it was the live ICB caches + mirrors + slabs; the dedupe
  buys ~17GB of Metal VA/working-set headroom at 256K, which is what
  trips allocation failures near the wired limit.
  **THE SLAB LEAK ALSO SHIPPED** (`14fa6aa`, same day): denseBatchScratch's
  grow sites (mlpFold/attnFold/layerStage/sdpaPromptS) now release the
  outgrown slabs and Close() releases the whole set — the old code stacked
  a dead 1-2GB sdpaPromptS pair per widening chunk and leaked the grown
  set on every session teardown. Gate:
  TestDenseBatchScratchReleasesSlabsOnGrowthAndClose (device
  CurrentAllocatedSize back to baseline over 4 grow+close cycles;
  +405MB FAIL with releases neutered — falsified before trusted). The
  row-plumbing buffers (offBuf/inPacked/outPacked/lastRows) cache host
  pointers and were deliberately left alone — they need their own
  care-map.
  **THE RAM-AWARE DEFAULT ALSO SHIPPED** (`fad5212`, same day):
  clampContextToRAM fits an UNSET -context to the box after
  loadRegistered — budget = RAM − reserve(min(max(20%,8GiB),24GiB)) −
  mapped weights − 4GiB fixed; global owners pay bf16-rate K+V per row
  (covers the q8 mirror transient), sliding owners charge their ring
  once; floor 4096, 1024-aligned, trace line on clamp. Escapes: explicit
  -context always wins; LTHN_CONTEXT_RAM_GUARD=0 kills it; a failed
  sysctl probe disables rather than guesses. mamba2/composed keep the
  plain capped default. 5 unit gates + guard-silent live check on the
  96GB box (suite 1513 green).
  **THE SESSION KV LIFECYCLE FIX** (`f9003ce`) — the fourth gremlin,
  spotted by Snider off the pressure graph: session Close dropped the
  archICBReplay without releasing its KV set; model.Generate closes a
  session per call, so EVERY generate leaked a complete set (region
  census: 40 half-GB q8 planes where one session owns 20; tiny-prompt
  31B@256K footprint 41.3G→29.9G with the fix). releaseKVCaches frees
  codes+scales+mirrors once via the primary (the peer ICB shares the
  slices). Gate: TestArchSessionCloseReleasesICBKVCaches (falsified:
  FAILS with the release neutered).
  **256K DEFAULT LANDED** (`ef9392a`) — the receipt: 31B all-defaults
  over a 235K cold prompt on the 96GB box: peak 53.96GB (Wednesday's
  killed run: 64.9GB + 19.5GB swap), swap ended BELOW baseline,
  answered the 235K needle correctly. generate now prints prefill
  tok/s ('prefill N tok @ R tok/s').
  **THE NEXT LANE — PREFILL ECONOMY AT DEPTH** (measured 2026-07-13):
  31B@235K prefill = 110 tok/s (35.6 min). e2b vs llama.cpp (same-class
  quants): pp512 3057 vs 4437 (−31%), pp8K 2840 vs 3776 (−25%),
  pp32K 2541 vs 2412 (+5%) — llama.cpp wins SHALLOW, we cross over at
  depth; our curve is flatter. Suspects: per-chunk fixed overheads in
  the shallow regime, and the q8 GEMM-prefix re-dequant which is
  O(N²) — every chunk re-dequants the whole attended prefix into the
  mirrors when only the delta rows changed (mirrors persist; old rows
  never change). Instrument first: LTHN_GPU_TRACE depth ladder
  (16/64/128K) to attribute the curve before touching anything.
  Remaining after that: the N-bit knob; the ~14GB of private
  misaligned-tensor weight copies on 31B (census 180×53MB regions —
  alignment-aware conversion is the fix shape); the 26B paged-lane q8
  alignment; the row-plumbing buffer lifecycle care-map.
- **#373 (closed — read its receipts before ANY fusion work)** — the fusion
  map: decode is GPU-busy at ~170GB/s of ~800; thin-stage fusion is EXHAUSTED
  (receipted flat); the 500-tok/s lane is fat-dispatch kernel architecture.
- **#371 (closed — the FULL metal_runtime sweep is GREEN again)** — the 16
  failures had three roots, none of them the suspected mlx bump (falsified:
  pre-bump Go + current metallib passes). (a) the q8 KV default quantised the
  paged pool on lanes whose decode never reads it — q8 now defaults only on
  the paged-decode lane (MoE/trace = 26B), `8fc251d`; (b) the fused PLE
  gate+gelu ICB op was reverted from the arch recording — byte-identical in
  every standalone probe yet drifting in the recorded replay from the second
  step (in-situ cause unresolved; its receipt was ~3µs), `96e0cc3`; (c) the
  coverage guards now pin the loader-owns-default context contract,
  `011a41d`. A red FULL sweep is a REGRESSION again — treat it as such.
  Residual seam for #367: on the 26B paged-q8 lane the batched pass attends
  its own fresh K/V rows pre-quantisation (serial attends post-round-trip) —
  unpinned, price a fresh-row round-trip if 26B MTP revives.
- **#366 / #360** — the frontier lanes (KV temporal coherence; Lemma v2
  training), waiting on their own starts.

## Falsified — do not re-dig

- MXU/NAX on M3 Ultra (needs GPU arch gen 17; this part is g15).
- Token early-exit on e2b (argmax decides at layer ~29/35; detector > skip).
- Weight-stream splitting at small K (the gather's CONCURRENT threadgroups
  already L2-amortise the stream; small-K matvecs are occupancy/launch-bound,
  not bandwidth-bound).
- TG-per-row fusion kernels (K threadgroups starve a 76-core GPU).
- In-kernel grid-barrier megakernels (Metal guarantees no TG co-residency —
  barriers spin-bail into wrong results; even `lthn_ffn_megakernel` is
  off-by-receipt in production).
- Thin-stage ICB fusion for speed (saves only ~3µs launch; the byte-identical
  PLE gate fusion shipped anyway since it costs nothing — `d00c526`).
- Draft-block ≠ 4 on e2b (3/6/8/12 all worse or noise).
- Serve two-point marginal for depth curves past ~16K (rides the variance of
  two full prefills — produced a false "98K collapse"; use
  `generate -prompt-file`).

## The house fusion rule (learned twice, keep it)

Fold only **O(output)** work into a matvec's loads/stores; per-element work
consumed by many tiles must stay its own dispatch. The receipted-off
`geluFoldEnabled` / `routerFusedEnabled` / `enableInputRMSFusion` all violate
it; the byte-identical `lthn_ple_gate_gelu_qmv` respects it (gelu once per
output row at lane 0). For byte-identity ports of MLX matvecs, replicate
`qmv_fast_impl` verbatim — `lthn_qmv_rows` and `lthn_ple_gate_gelu_qmv` are
the precedents, and `bytes.Equal` gates are achievable, not aspirational.

## Instruments (run these, don't re-derive)

- **Wall vs GPU split**: `TestDecodeWallGPUSplitRealE2B` (`LEM_REAL_E2B=1`) —
  chained-lane GPU-span accounting; the fixed-cost fork resolver.
- **Per-stage GPU buckets**: `LTHN_GPU_TRACE=1` on any serve (batched pass).
- **MTP internals**: `LTHN_MTP_DIAG=1` (draft fwd/head split, verify rows,
  acceptance per block).
- **Depth curve**: `lem-dev generate -prompt-file` per depth point.
- **Live harness**: `lem.sh bench/pair/book/smoke` — the book prompt column is
  the continuity receipt (~88 tok/turn flat = no replay).

## Traps that will bite you (each cost real time)

- **`bin/lem` is a build artifact, not the tree** — a full 2×2 A/B ran on a
  binary built 2h before the code it was supposed to measure (env opt-ins
  silently ignored; the q8 lanes were bf16 reruns identical to the ms).
  `task build` before ANY receipt run, and treat impossibly-identical lanes
  as a stale-binary tell, not a result.
- **objc `new*` buffers leak unless YOU release them** (third scar in this
  class: the MoE re-upload leak, the resident-cache eviction leak). The
  test sweep stacked ~10 GB of IOAccelerator-dirty per real-model test and
  drove machine memory pressure red — `653aef1` releases at ranged eviction.
  Two rules from its falsifications: release ONLY where the owner is
  provably done (releasing at reset-for-test dangled live fixtures — wrong
  tokens, then SIGSEGV), and don't chase ps-RSS — clean mmap pages inflate
  it harmlessly; `vmmap --summary`'s physical footprint + IOAccelerator
  DIRTY column is the pressure that matters.
- `MLX_METALLIB_PATH` must be THIS repo's `build/dist/lib/mlx.metallib` —
  go-mlx's lacks the lthn kernels ("kernel not found" on runtime tests).
- `task metallib:kernels` rebuilds the lthn library; new `.metal` files under
  `engine/metal/kernels/` auto-include. The MLX include chain needs the fast
  variant's geometry gates (`inDim%512==0`) respected per kernel.
- Serve's default context now follows the checkpoint window capped at 32K
  (`resolveDefaultContext`); before `601ac4e` everything silently capped at
  4096 — the book-bench chapter-5 death class.
- The ICB op layout is dynamic per-feature (`opsPerLayer` accounting) —
  uniform shrinks are fine and the recorder self-asserts the final count.
- Continuity decline reasons log via `core.Error`, which does NOT reach the
  serve stderr — don't chase ghosts in `/tmp/lem-serve.log`.
- Codex (GPT-5.6-Sol) owns `go/engine/hip` on the homelab box (~a month); it
  has NO git access — snapshot its working tree to a `codex/...` branch (see
  `codex/hip-stint-2026-07-10`) and review for mitigation-instead-of-root-cause
  before anything reaches shared layers. Its RAM-offload lane (experts in
  system RAM, <6GB VRAM for 26B) amortises on MoE only — dense models must fit
  resident (12B fits a 16GB card; 31B does not).

## Where the money went (shipped receipts, greppable)

`c77f5be` serve arms MTP (was silently plain) · `b2b2dcf` drafter cluster head
on GPU (10-17ms → 0.1ms/draft tok) · `aa45e02` quant K-row verify head ·
`028766d` 3-dispatch verify epilogue (verify 11.5 → 9.7ms) · `96e1b83`
conversation continuity (13× turn-2) · `601ac4e` checkpoint-window context
default · `45e014f` the wall/GPU split probe · `d00c526` byte-identical PLE
gate fusion + the thin-stage receipt.

## 2026-07-12, LATE NIGHT — the 12B verdict, GEMM rung, wave-4 close

- **#52 r10 VERDICT — the nine-round "hip 12B forward bug" dissolved.** On
  IDENTICAL pack bytes (box clean requant rsynced to the mac — r8 had compared
  metal-on-local-HF-quant vs hip-on-box-requant, so its "layer-5 MLP
  amplification, final cosine 0.22" was a PACK-MISMATCH ARTEFACT) the engines
  AGREE: worst cosine 0.9635 anywhere in the 48L×6tok ladder, final layers
  0.994+, all five layer-5 MLP sub-tensors ≥0.9997, hip product ==
  tanh-gelu(gate)×up to 1e-5. Metal generates the clean pack coherently
  (71 tok/s). The REAL hip signature: PROGRESSIVE decay (coherent ~60 tok →
  "Rayreuleg" → "Earth'ように" → "1.11"-loop collapse) at **0.1 tok/s decode /
  2 tok/s prefill** — accumulating error on a host-fallback-heavy route, not a
  wrong kernel. Suspect: 12B dual-geometry attention (512 head-dim full-attn
  every 6th layer vs 256 sliding; E2B/E4B have no dual geometry) missing the
  device route. r10 codex lane (lane/hip-12b-r10) is producing the per-op
  device-vs-host route table 12B-vs-E2B, diagnosis only.
- **Cross-engine diff discipline:** any future engine differential MUST consume
  byte-identical weights on both sides (rsync the pack, sha the shards) — two
  quantisation runs of the same model differ enough to fake an engine bug.
- **`fix(serving,generate) 0af1329`:** the serve resolver + decode/generate
  pinned `WithBackend("metal")`, so `lem generate`/serve could NEVER load on
  the hip box. Now inference.LoadModel preference order (metal→rocm→llama_cpp,
  Available()-checked). Darwin receipt unchanged (67.5 tok/s).
- **TRAP: `go -C go build -o <relative>`** resolves -o against the -C dir —
  wrote the fixed binary to go/build/bin/ while the probe ran the stale one at
  build/bin/ (cost ~20 min). Absolute -o always; verify with
  `strings <bin> | grep <new-literal>` before trusting a probe.
- **CB rung 2 landed (lane/gemm-batch → ea73a3a~):** weight-read-once GEMM
  batching over the lane set — the seven projections gather K lanes into one
  slab, sweep each weight once. **1.47× over replay at K=4/8** (synthetic bf16
  E2B-shape; 265.9 tok/s aggregate at K=4 vs 179.4 replay), byte-identical,
  lockstep counter-guarded receipt. 4-bit falls back to the 2.58× per-lane ICB
  replay: the quant ICB fuses entry/MLP RMS into the qmv (fp32-internal), so a
  batched read needs a **batched rms-qmv-rows metallib kernel** — pinned as the
  next rung; metallib is read-only until a rebuild window.
- **Wave-4 MoE flood closed:** qwenmoe, granitemoe, dbrx, llama4-text
  (a9b658d), jetmoe re-run (37f9c68 — self-committed 13 tests). Composed gained
  QKVClip (dbrx), l2NormHead + per-layer buildAttn (llama4), granite
  multipliers survived the union (its own test caught my drop). **LESSON PAID
  TWICE: never remove a worktree until `git log <branch> -1` shows a commit
  past base** — jetmoe run 1's uncommitted 97%-coverage package was destroyed
  by the orchestrator's cleanup sweep.
- **Gate discipline (third strike today):** `<cmd> | tail; echo $?` reports
  TAIL's exit — the rsync "success" that copied nothing and the vet
  "clean-exit" both hid behind it. Capture to file; assert on the file.

## 2026-07-13, SMALL HOURS — the sampler conviction (r11+A/B)

- r11 feedback receipts: argmax==fed 30/30, pos==KV 30/30, generation==teacher-
  forced 29/29 — feedback loop CLEAN (that hypothesis lasted one round).
- **The A/B that ended the hunt** (box, clean pack, same binary/prompt, 40 tok):
  greedy temp 0 → COHERENT at 42.4 tok/s; sampled temp 1.0 → repetition
  collapse by ~token 12 at **0.2 tok/s**. The garble AND the "923s" slowness
  are BOTH sampled-path-only. Nine rounds of oracles were clean because every
  one of them ran greedy — the device sampling stack
  (hipRunPackedTopKSampleKernel / softcap-sample variants + host prep + the
  eligibility gates) is the one seam no oracle ever touched. r12 codex lane is
  on it: RED device-vs-host sampler oracle → timing receipt → root-cause fix
  (host-sampling reroute banned as a workaround); done-gate = temp-1.0 120-tok
  coherent 12B at sane speed, greedy byte-unchanged, E2B still coherent.
- Triage rule earned: **when generation misbehaves, A/B temp-0 vs default FIRST**
  — it separates forward from sampler in one run and would have saved rounds
  1–11. (lem generate defaults to temp 1.0, not greedy.)

## 2026-07-13 — #52 CLOSED: the producer fork (fifteen rounds, one boolean)

- **ROOT CAUSE (r15, 09168bd):** `useBatchedPrefill := … && !hostSampling` —
  requesting SAMPLING switched the whole prompt/decode producer to the
  token-at-a-time prefill, whose 12B logits are flattened (argmax mostly
  preserved). Fifteen rounds of oracles stayed clean because greedy and
  near-zero-temp only see the argmax; temp-1 sampling sees the spread.
  Fix: batched prefill owns every compatible prompt; the sampler consumes the
  SAME forward's final-hidden row (hipGemma4Q4SampleBatchedPrefillRow;
  batched decode gained ReturnHidden). Receipts: 12B default CLI 120-tok
  coherent 20.8 tok/s · greedy byte-unchanged 42.5 · E2B 56.1 · armed suite
  green.
- **En-route real fixes that weren't THE bug:** device sampler missing final
  softcap (r12, oracle-caught); quadratic insertion sort over the 262144
  vocab per sampled step → radix, 0.2→20.8 tok/s (r13); loader
  WithBackend("metal") pin (0af1329); r8's cross-engine "divergence" exposed
  as a pack-mismatch artefact (r9/r10).
- **Residuals → #71:** explicit TopK-64 path still incoherent (device draws
  match host reference — a third producer fork suspected; spread instrument
  in-tree); generation_config sampling defaults ignored by BOTH engines
  (declares-discipline follow-up); why metal's raw full-vocab temp-1 is
  coherent (instrument).
- **LESSONS (banked to memory):** sampler-vs-forward triage = A/B temp-0 vs
  default FIRST. A knob must never select a different PRODUCER — eligibility
  gates that fork the computation (not just the consumer) are the bug class
  fifteen rounds of consumer-side oracles cannot see; grep eligibility
  booleans for forward-path terms. Pre-change gates in fix briefs (r14's
  "explicit-topk must be coherent first") pay for themselves — one fired and
  saved a masking implementation.
