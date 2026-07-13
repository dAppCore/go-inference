// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"bytes"
	"context"
	"os"
	"slices"
	"testing"
	"time"

	"dappco.re/go/inference"
	"dappco.re/go/inference/model"
	g4 "dappco.re/go/inference/model/gemma4"
)

// TestLaneSetGEMMByteIdentityHiddens is the counter-guarded byte-identity receipt for
// the weight-read-once GEMM forward: two lane sets over the SAME varied-fill specs are
// advanced in lockstep — one with the batched-GEMM forward armed (gemmMode 1), one
// forced onto the per-lane ICB replay (gemmMode 2, the merged 2.58× path) — and every
// lane's post-stack hidden is compared BYTE-for-byte at every step, not just its
// argmax token. Byte-equal hiddens across the two phase-2 implementations is the whole
// correctness claim: sweeping a weight once for K lanes changes the projection's
// dispatch shape, never a row's arithmetic. The counter proves the GEMM path actually
// fired (gemmFwdCount > 0 on the armed set, 0 on the replay set) — else the proof is
// vacuous (both silently on the replay).
func TestLaneSetGEMMByteIdentityHiddens(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set — GPU decode fixture")
	}
	m := laneSetFixtureModel(t)
	defer m.Close()
	specs := laneSpecFixtures()
	ctx := context.Background()

	open := func(mode int) (*laneSet, []inference.LaneHandle) {
		lsi, err := m.OpenLaneSet(inference.LaneSetConfig{MaxLanes: len(specs)})
		if err != nil {
			t.Fatalf("OpenLaneSet(mode %d): %v", mode, err)
		}
		ls, ok := lsi.(*laneSet)
		if !ok {
			t.Fatalf("OpenLaneSet returned %T, not *laneSet", lsi)
		}
		ls.gemmMode = mode // 1 = batched GEMM armed, 2 = ICB replay only
		handles := make([]inference.LaneHandle, len(specs))
		for i, spec := range specs {
			if handles[i], err = ls.Prepare(ctx, spec); err != nil {
				t.Fatalf("Prepare(mode %d lane %d): %v", mode, i, err)
			}
		}
		return ls, handles
	}

	lsG, hG := open(1) // weight-read-once GEMM forward
	lsR, hR := open(2) // per-lane ICB replay
	defer lsG.Close()
	defer lsR.Close()

	for step := 0; ; step++ {
		sg, err := lsG.Step(ctx)
		if err != nil {
			t.Fatalf("step %d GEMM Step: %v", step, err)
		}
		sr, err := lsR.Step(ctx)
		if err != nil {
			t.Fatalf("step %d replay Step: %v", step, err)
		}
		if len(sg) == 0 && len(sr) == 0 {
			break
		}
		// Per-lane token AND post-stack hidden byte-identity across the two paths.
		for i := range specs {
			lg, lr := lsG.lanes[hG[i].ID], lsR.lanes[hR[i].ID]
			if lg == nil || lr == nil {
				continue // both retired this lane below
			}
			if !bytes.Equal(lg.hidden, lr.hidden) {
				t.Fatalf("step %d lane %d: post-stack hidden bytes differ between GEMM and replay (len %d vs %d)", step, i, len(lg.hidden), len(lr.hidden))
			}
		}
		// Token streams must match step-for-step too (the head reads that hidden).
		tokG := stepTokens(sg, hG)
		tokR := stepTokens(sr, hR)
		if !slices.Equal(tokG, tokR) {
			t.Fatalf("step %d: token vector diverges GEMM %v vs replay %v", step, tokG, tokR)
		}
		retireTerminal(t, lsG, sg)
		retireTerminal(t, lsR, sr)
	}

	if lsG.gemmFwdCount == 0 {
		t.Fatal("gemmFwdCount is zero — the weight-read-once GEMM forward never fired (proof vacuous)")
	}
	if lsR.gemmFwdCount != 0 {
		t.Fatalf("replay set ran %d GEMM forwards — it should be pure ICB replay", lsR.gemmFwdCount)
	}
}

// TestLaneSetGEMMRaggedJoinByteIdentity extends the byte-identity receipt across the
// RAGGED-JOIN surface: lanes are admitted in WAVES (two at step 0, one at step 2, one
// at step 4) into a GEMM-armed set and a replay set driven in lockstep, so the batched
// GEMM forward runs at a K that GROWS under a live set (2 → 3 → 4). This exercises two
// code paths the up-front-admission receipts do not: the gemmSlabs REALLOCATION when a
// later wave pushes K past the current staging (ensureGemmSlabs' grow branch), and a
// GEMM sweep over lanes at HETEROGENEOUS positions (a fresh joiner sits at pos = its
// prompt length while incumbents have already decoded several tokens). Every lane's
// post-stack hidden and token stream must stay byte-identical to the replay path at
// every step, before AND after each join — a joining lane must neither perturb the
// incumbents' bytes nor take a divergent forward for itself. The counter guard proves
// the GEMM path actually fired across the joins (else the proof is vacuous).
func TestLaneSetGEMMRaggedJoinByteIdentity(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set — GPU decode fixture")
	}
	m := laneSetFixtureModel(t)
	defer m.Close()
	specs := laneSpecFixtures()
	ctx := context.Background()

	// waveAt[i] is the step at which spec i is admitted. Lanes 0,1 start together;
	// lane 2 joins after two steps; lane 3 after four — so the GEMM forward runs at
	// K = 2, then 3, then 4, forcing the slab to reallocate under a live set. MaxNew
	// (8) exceeds the last join step, so every lane is admitted well before any
	// retires — the loop covers both ragged JOIN and ragged LEAVE.
	waveAt := []int{0, 0, 2, 4}

	open := func(mode int) *laneSet {
		lsi, err := m.OpenLaneSet(inference.LaneSetConfig{MaxLanes: len(specs)})
		if err != nil {
			t.Fatalf("OpenLaneSet(mode %d): %v", mode, err)
		}
		ls, ok := lsi.(*laneSet)
		if !ok {
			t.Fatalf("OpenLaneSet returned %T, not *laneSet", lsi)
		}
		ls.gemmMode = mode // 1 = batched GEMM armed, 2 = ICB replay only
		return ls
	}
	lsG, lsR := open(1), open(2)
	defer lsG.Close()
	defer lsR.Close()

	admitted := make([]bool, len(specs))
	hG := make([]inference.LaneHandle, len(specs))
	hR := make([]inference.LaneHandle, len(specs))
	admitWave := func(step int) {
		for i := range specs {
			if admitted[i] || waveAt[i] != step {
				continue
			}
			var err error
			if hG[i], err = lsG.Prepare(ctx, specs[i]); err != nil {
				t.Fatalf("GEMM Prepare(lane %d @ step %d): %v", i, step, err)
			}
			if hR[i], err = lsR.Prepare(ctx, specs[i]); err != nil {
				t.Fatalf("replay Prepare(lane %d @ step %d): %v", i, step, err)
			}
			admitted[i] = true
		}
	}

	for step := 0; ; step++ {
		admitWave(step)
		sg, err := lsG.Step(ctx)
		if err != nil {
			t.Fatalf("step %d GEMM Step: %v", step, err)
		}
		sr, err := lsR.Step(ctx)
		if err != nil {
			t.Fatalf("step %d replay Step: %v", step, err)
		}
		allAdmitted := !slices.Contains(admitted, false)
		if len(sg) == 0 && len(sr) == 0 && allAdmitted {
			break
		}
		// Per-lane hidden + token byte-identity for every lane live in BOTH sets.
		for i := range specs {
			if !admitted[i] {
				continue
			}
			lg, lr := lsG.lanes[hG[i].ID], lsR.lanes[hR[i].ID]
			if lg == nil || lr == nil {
				continue // retired this step in one/both sets
			}
			if !bytes.Equal(lg.hidden, lr.hidden) {
				t.Fatalf("step %d lane %d: post-stack hidden bytes differ between GEMM and replay after ragged join (len %d vs %d)", step, i, len(lg.hidden), len(lr.hidden))
			}
		}
		if tokG, tokR := stepTokens(sg, hG), stepTokens(sr, hR); !slices.Equal(tokG, tokR) {
			t.Fatalf("step %d: token vector diverges GEMM %v vs replay %v", step, tokG, tokR)
		}
		retireTerminal(t, lsG, sg)
		retireTerminal(t, lsR, sr)
	}

	if lsG.gemmFwdCount == 0 {
		t.Fatal("gemmFwdCount is zero — the GEMM forward never fired across the ragged joins (proof vacuous)")
	}
	if lsR.gemmFwdCount != 0 {
		t.Fatalf("replay set ran %d GEMM forwards — it should be pure ICB replay", lsR.gemmFwdCount)
	}
}

// TestLaneSetGEMMQuantFallsBackToReplay pins the PROVEN-ENVELOPE fallback: E2B is a
// 4-bit checkpoint whose PLE tower (and sliding window / KV-share layers) sit outside
// the fold's demonstrated envelope, so even with the forward armed (gemmMode 1) the
// lane set runs the per-lane ICB replay — the merged 2.58× path — byte-for-byte, and
// the GEMM counter stays zero. E2B declines for TWO reasons now: the envelope
// exclusions AND the quant projection safety gate — the register-tiled lthn_qmv_rows
// is NOT byte-identical to the per-lane qmv (the hd-256 fold divergence, root-caused
// 2026-07-13), so ANY quant model declines the fold (TestLaneSetGEMMDenseQuantDeclinesFold
// pins the dense-quant case with no envelope features).
func TestLaneSetGEMMQuantFallsBackToReplay(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set — GPU decode fixture")
	}
	if _, err := os.Stat(laneSetABModelDir); err != nil {
		t.Skipf("E2B model dir not present (%s) — set LTHN_CB_AB_MODEL", laneSetABModelDir)
	}
	tm, err := LoadTokenModelDir(laneSetABModelDir, 4096)
	if err != nil {
		t.Fatalf("LoadTokenModelDir: %v", err)
	}
	m, ok := tm.(*NativeTokenModel)
	if !ok {
		if c, ok := tm.(interface{ Close() error }); ok {
			_ = c.Close()
		}
		t.Skipf("loaded model is %T, not *NativeTokenModel", tm)
	}
	defer m.Close()

	const k, maxNew = 4, 6
	ctx := context.Background()
	specs := make([]inference.LaneSpec, k)
	for i := range specs {
		ids := make([]int32, 12)
		for j := range ids {
			ids[j] = int32(2 + (i*29+j*5)%1500)
		}
		specs[i] = inference.LaneSpec{PromptIDs: ids, MaxNew: maxNew}
	}

	run := func(mode int) (map[int][]int32, uint64) {
		lsi, err := m.OpenLaneSet(inference.LaneSetConfig{MaxLanes: k})
		if err != nil {
			t.Fatalf("OpenLaneSet(mode %d): %v", mode, err)
		}
		ls := lsi.(*laneSet)
		ls.gemmMode = mode
		handles := make([]inference.LaneHandle, k)
		for i, spec := range specs {
			if handles[i], err = ls.Prepare(ctx, spec); err != nil {
				t.Fatalf("Prepare(mode %d lane %d): %v", mode, i, err)
			}
		}
		toks, _ := drainLaneSet(t, ls)
		fwd := ls.gemmFwdCount
		byID := map[int][]int32{}
		for i := range specs {
			byID[i] = toks[handles[i].ID]
		}
		_ = ls.Close()
		return byID, fwd
	}

	armedTok, armedFwd := run(1) // GEMM armed — must fall back for quant
	replayTok, replayFwd := run(2)

	if armedFwd != 0 {
		t.Fatalf("quant E2B ran %d batched-GEMM forwards — it must fall back to the replay (PLE/sliding/KV-share proven-envelope exclusion)", armedFwd)
	}
	if replayFwd != 0 {
		t.Fatalf("replay ran %d GEMM forwards", replayFwd)
	}
	for i := range specs {
		if len(armedTok[i]) == 0 {
			t.Fatalf("lane %d produced no tokens", i)
		}
		if !slices.Equal(armedTok[i], replayTok[i]) {
			t.Fatalf("lane %d: armed %v != replay %v (fallback must be transparent)", i, armedTok[i], replayTok[i])
		}
	}
	t.Logf("quant E2B correctly falls back to the ICB replay (0 GEMM forwards), output byte-identical")
}

// TestLaneSetGEMMThroughputAB is the weight-read-once receipt on a realistic-scale
// bf16 gemma4 (E2B-shaped dims: dModel 1536, 16 layers, qDim 2048, dFF 8192 — big
// enough that decode is weight-bandwidth-bound, where reading each weight ONCE for K
// lanes instead of K times is the win). Synthetic weights (the tokens are arbitrary
// but the GPU work and timings are real), so it stands in for a real bf16 checkpoint
// the box does not carry — both installed checkpoints are 4-bit quant, which is
// GEMM-ineligible (see TestLaneSetGEMMQuantFallsBackToReplay). Three modes, K=4:
//
//	serial (1 lane at a time, ICB replay)     — the pre-batching baseline
//	batched replay (K lanes, 1 CB, K weights) — the merged 2.58× CB-count win
//	batched GEMM   (K lanes, 1 CB, 1 weight)  — weight-read-once, on top
//
// All three MUST produce byte-identical tokens (asserted) — the win is scheduling +
// bandwidth, never arithmetic. Not a gate (t.Log): the number is the deliverable.
func TestLaneSetGEMMThroughputAB(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set — GPU throughput receipt")
	}
	// headDim 128 (not 256/512) keeps the ICB KV caches bf16 — a 256/512 head trips
	// the global-layer q8 KV path, which is GEMM-ineligible (a separate rung).
	const dModel, nHeads, nKV, headDim, dFF, vocab, nLayers = 1536, 16, 1, 128, 8192, 4096, 16
	g, arch := gemma4BF16Fixture(t, dModel, nHeads, nKV, headDim, dFF, vocab, nLayers)
	m, err := NewBF16TokenModel(g, arch, 2048)
	if err != nil {
		t.Fatalf("NewBF16TokenModel: %v", err)
	}
	defer m.Close()
	probe, err := m.OpenSession()
	if err != nil {
		t.Fatalf("OpenSession: %v", err)
	}
	as, ok := probe.(*ArchSession)
	if !ok || as.state.icb == nil {
		_ = probe.(interface{ Close() error }).Close()
		t.Skip("scaled fixture not ICB-eligible on this metallib")
	}
	_ = as.Close()
	ctx := context.Background()

	// Scale probe: K=4 (register-tiled tier) and K=8 (still < steelGEMMMinRows, so
	// the batched bf16 gemv stays byte-identical to K single-row gemvs).
	for _, kMaxNew := range [][2]int{{4, 48}, {8, 32}} {
		k, maxNew := kMaxNew[0], kMaxNew[1]
		runGEMMThroughputAB(t, m, ctx, dModel, nHeads, headDim, nLayers, dFF, k, maxNew)
	}
}

func runGEMMThroughputAB(t *testing.T, m *NativeTokenModel, ctx context.Context, dModel, nHeads, headDim, nLayers, dFF, k, maxNew int) {
	specs := make([]inference.LaneSpec, k)
	for i := range specs {
		ids := make([]int32, 24)
		for j := range ids {
			ids[j] = int32(2 + (i*31+j*7)%2000)
		}
		specs[i] = inference.LaneSpec{PromptIDs: ids, MaxNew: maxNew}
	}

	// Serial: one lane at a time (always ICB replay — K=1 is never GEMM-batched).
	serialTokens := make([][]int32, k)
	serialStart := time.Now()
	var serialGen int
	for i, spec := range specs {
		lsi, err := m.OpenLaneSet(inference.LaneSetConfig{MaxLanes: 1})
		if err != nil {
			t.Skipf("OpenLaneSet(serial): %v", err)
		}
		ls := lsi.(*laneSet)
		h, err := ls.Prepare(ctx, spec)
		if err != nil {
			_ = ls.Close()
			t.Skipf("Prepare(serial %d): %v", i, err)
		}
		got, _ := drainLaneSet(t, ls)
		serialTokens[i] = got[h.ID]
		serialGen += len(serialTokens[i])
		_ = ls.Close()
	}
	serialWall := time.Since(serialStart)

	batched := func(mode int) ([][]int32, uint64, uint64, time.Duration) {
		lsi, err := m.OpenLaneSet(inference.LaneSetConfig{MaxLanes: k})
		if err != nil {
			t.Fatalf("OpenLaneSet(batched mode %d): %v", mode, err)
		}
		ls := lsi.(*laneSet)
		ls.gemmMode = mode
		handles := make([]inference.LaneHandle, k)
		start := time.Now()
		for i, spec := range specs {
			if handles[i], err = ls.Prepare(ctx, spec); err != nil {
				t.Fatalf("Prepare(batched %d lane %d): %v", mode, i, err)
			}
		}
		toks, fwd := drainLaneSet(t, ls)
		wall := time.Since(start)
		gemmFwd := ls.gemmFwdCount
		out := make([][]int32, k)
		for i := range specs {
			out[i] = toks[handles[i].ID]
		}
		_ = ls.Close()
		return out, fwd, gemmFwd, wall
	}

	replayTok, replayFwd, replayGemm, replayWall := batched(2) // batched replay
	gemmTok, gemmFwd, gemmGemm, gemmWall := batched(1)         // batched GEMM

	// Byte-identity: every mode's tokens match the serial oracle.
	countGen := func(toks [][]int32) int {
		n := 0
		for _, t := range toks {
			n += len(t)
		}
		return n
	}
	for i := range specs {
		if !slices.Equal(replayTok[i], serialTokens[i]) {
			t.Fatalf("lane %d: batched-replay tokens diverge from serial", i)
		}
		if !slices.Equal(gemmTok[i], serialTokens[i]) {
			t.Fatalf("lane %d: batched-GEMM tokens diverge from serial — the win must not change output\n gemm  =%v\n serial=%v", i, gemmTok[i], serialTokens[i])
		}
	}
	if gemmGemm == 0 {
		t.Fatal("batched-GEMM mode ran 0 weight-read-once forwards — the receipt is vacuous")
	}
	if replayGemm != 0 {
		t.Fatalf("batched-replay mode ran %d GEMM forwards (should be pure replay)", replayGemm)
	}

	serialTokS := float64(serialGen) / serialWall.Seconds()
	replayTokS := float64(countGen(replayTok)) / replayWall.Seconds()
	gemmTokS := float64(countGen(gemmTok)) / gemmWall.Seconds()
	t.Logf("weight-read-once A/B (synthetic bf16 E2B-shape: dModel=%d layers=%d qDim=%d dFF=%d, K=%d, maxNew=%d):", dModel, nLayers, nHeads*headDim, dFF, k, maxNew)
	t.Logf("  serial (1 lane/replay):          %d tok in %v = %.1f tok/s", serialGen, serialWall.Round(time.Millisecond), serialTokS)
	t.Logf("  batched replay (K/1CB/K weights): %d tok in %v = %.1f tok/s (%d fwd) = %.2fx vs serial", countGen(replayTok), replayWall.Round(time.Millisecond), replayTokS, replayFwd, replayTokS/serialTokS)
	t.Logf("  batched GEMM   (K/1CB/1 weight):  %d tok in %v = %.1f tok/s (%d fwd, %d gemm) = %.2fx vs serial, %.2fx vs batched replay", countGen(gemmTok), gemmWall.Round(time.Millisecond), gemmTokS, gemmFwd, gemmGemm, gemmTokS/serialTokS, gemmTokS/replayTokS)
}

// stepTokens flattens a Step's results into a per-spec token vector (0 where a lane
// produced no token this step), keyed by the spec's handle order.
func stepTokens(steps []inference.LaneStep, handles []inference.LaneHandle) []int32 {
	out := make([]int32, len(handles))
	for _, s := range steps {
		for i, h := range handles {
			if h.ID == s.Lane.ID && s.HasToken {
				out[i] = s.Token
			}
		}
	}
	return out
}

func retireTerminal(t *testing.T, ls *laneSet, steps []inference.LaneStep) {
	t.Helper()
	for _, s := range steps {
		if s.Terminal {
			if err := ls.Retire(s.Lane); err != nil {
				t.Fatalf("Retire: %v", err)
			}
		}
	}
}

// laneSetQuantFixtureModel builds a synthetic 4-bit quant gemma4 whose seven
// projection dims all satisfy the register-tiled lthn_qmv_rows plan
// (inDim%256==0, outDim%8==0) at gs=32 — the quant byte-identity fixture for
// the weight-read-once fold. Skips (not fails) when the fixture is not
// recorded-ICB eligible or the tiled plan is unavailable on this metallib.
func laneSetQuantFixtureModel(t *testing.T) *NativeTokenModel {
	return laneSetQuantFixtureModelMax(t, 32)
}

// laneSetQuantFixtureModelMax builds the DENSE 4-bit quant fixture at a
// caller-chosen maxLen (maxLen >= sdpa2PassMinKV records the fixed-fan 2-pass SDPA
// layout on global layers). Used to pin that dense quant DECLINES the fold (the
// register-tiled projection is not byte-identical — see
// TestLaneSetGEMMDenseQuantDeclinesFold); the q8-KV byte-identity receipt runs on
// bf16 weights instead (laneSetBF16Q8FixtureMax).
func laneSetQuantFixtureModelMax(t *testing.T, maxLen int) *NativeTokenModel {
	t.Helper()
	const gs, bits = 32, 4
	// HeadDim 256: the q8 KV lane only arms on head-dim 256/512 (its SDPA
	// kernels exist for the production gemma4 dims), and every projection dim
	// stays on the tiled qmv-rows plan (inDim%256, outDim%8).
	cfg := g4.Config{
		HiddenSize: 256, NumHiddenLayers: 2, IntermediateSize: 512,
		NumAttentionHeads: 2, NumKeyValueHeads: 1, HeadDim: 256, VocabSize: 48, RMSNormEps: 1e-6,
		Quantization: &model.QuantConfig{GroupSize: gs, Bits: bits},
	}
	arch, err := cfg.Arch()
	if err != nil {
		t.Fatalf("Arch: %v", err)
	}
	lm, err := model.Assemble(quantGemma4Tensors(t, arch, gs, bits), arch, model.StandardWeightNames())
	if err != nil {
		t.Fatalf("model.Assemble: %v", err)
	}
	g, err := loadedToQuant(lm, gs, bits)
	if err != nil {
		t.Fatalf("loadedToQuant: %v", err)
	}
	m, err := NewQuantTokenModel(g, arch, maxLen)
	if err != nil {
		t.Fatalf("NewQuantTokenModel: %v", err)
	}
	probe, err := m.OpenSession()
	if err != nil {
		t.Fatalf("OpenSession: %v", err)
	}
	as, ok := probe.(*ArchSession)
	if !ok || as.state.icb == nil {
		_ = probe.(interface{ Close() error }).Close()
		_ = m.Close()
		t.Skip("quant fixture is not recorded-ICB eligible on this metallib")
	}
	// No rowsByteTier guard: it is now always false for a quant projector (the
	// register-tiled kernel is not byte-identical), which is precisely the safety
	// gate the sole caller (TestLaneSetGEMMDenseQuantDeclinesFold) pins. The fold's
	// byte-tier gate declines this fixture regardless of the tiled kernel's
	// availability, so recorded-ICB eligibility is the only precondition.
	_ = as.Close()
	return m
}

// TestLaneSetGEMMDenseQuantDeclinesFold pins the projection SAFETY GATE (receipt
// 2026-07-13): a DENSE 4-bit quant model — no PLE, no sliding, no KV-share, so
// the proven-envelope exclusions do NOT apply — STILL declines the weight-read-
// once fold, because the register-tiled lthn_qmv_rows kernel is not byte-identical
// to the per-lane qmv the replay records (proven 2026-07-13: batched projectRows
// vs per-row project diverged ~1 ulp value-dependently on every projection; the
// divergence surfaced as the hd-256 fold failure at step 6). So even with the
// forward armed (gemmMode 1)
// the lane set falls back to the per-lane ICB replay byte-for-byte and the GEMM
// counter stays zero. This is the load-bearing pin for the gate that protects
// production 12B/31B dense-quant (hd 256/512): if it ever finds the fold firing,
// a byte-identical batched quant kernel landed — lift the gate and restore this
// to the positive byte-identity receipt.
func TestLaneSetGEMMDenseQuantDeclinesFold(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set — GPU decode fixture")
	}
	m := laneSetQuantFixtureModel(t)
	defer m.Close()
	specs := laneSpecFixtures()
	ctx := context.Background()

	open := func(mode int) (*laneSet, []inference.LaneHandle) {
		lsi, err := m.OpenLaneSet(inference.LaneSetConfig{MaxLanes: len(specs)})
		if err != nil {
			t.Fatalf("OpenLaneSet(mode %d): %v", mode, err)
		}
		ls, ok := lsi.(*laneSet)
		if !ok {
			t.Fatalf("OpenLaneSet returned %T, not *laneSet", lsi)
		}
		ls.gemmMode = mode
		handles := make([]inference.LaneHandle, len(specs))
		for i, spec := range specs {
			if handles[i], err = ls.Prepare(ctx, spec); err != nil {
				t.Fatalf("Prepare(mode %d lane %d): %v", mode, i, err)
			}
		}
		return ls, handles
	}

	lsG, hG := open(1) // fold armed — must DECLINE (projection safety gate)
	lsR, hR := open(2) // pure ICB replay
	defer lsG.Close()
	defer lsR.Close()

	for step := 0; ; step++ {
		sg, err := lsG.Step(ctx)
		if err != nil {
			t.Fatalf("step %d GEMM Step: %v", step, err)
		}
		sr, err := lsR.Step(ctx)
		if err != nil {
			t.Fatalf("step %d replay Step: %v", step, err)
		}
		if len(sg) == 0 && len(sr) == 0 {
			break
		}
		// The declined fold must be a TRANSPARENT fallback: byte-identical hiddens
		// and tokens to the pure replay at every step.
		for i := range specs {
			lg, lr := lsG.lanes[hG[i].ID], lsR.lanes[hR[i].ID]
			if lg == nil || lr == nil {
				continue
			}
			if !bytes.Equal(lg.hidden, lr.hidden) {
				t.Fatalf("step %d lane %d: declined-fold hidden bytes differ from replay (fallback not transparent)", step, i)
			}
		}
		tokG := stepTokens(sg, hG)
		tokR := stepTokens(sr, hR)
		if !slices.Equal(tokG, tokR) {
			t.Fatalf("step %d: token vector diverges GEMM %v vs replay %v", step, tokG, tokR)
		}
		retireTerminal(t, lsG, sg)
		retireTerminal(t, lsR, sr)
	}

	if lsG.gemmFwdCount != 0 {
		t.Fatalf("dense quant ran %d batched-GEMM forwards — the projection safety gate must decline it (register-tiled lthn_qmv_rows is not byte-identical)", lsG.gemmFwdCount)
	}
	if lsR.gemmFwdCount != 0 {
		t.Fatalf("replay set ran %d GEMM forwards — it should be pure ICB replay", lsR.gemmFwdCount)
	}
}

// TestLaneSetGEMMEnvelopeExclusionLoadBearing pins that the proven-envelope
// exclusions in gemmEligible protect REAL output (receipt 2026-07-13): with
// gemmEnvelopeLiftForTest armed on real E2B — a 4-bit checkpoint carrying all
// three excluded features at once (PLE tower, sliding windows, KV-share
// layers), q8 KV disabled so this run isolates exactly those arms — the
// weight-read-once forward FIRES and its post-stack hiddens DIVERGE from the
// per-lane ICB replay at step 0. The mirrored single-row PLE/sliding/KV-share
// arms are not byte-identical yet, so the exclusions stay. If this test ever
// finds full identity instead, the arms got fixed: lift the exclusions and
// convert this into the positive promotion receipt.
func TestLaneSetGEMMEnvelopeExclusionLoadBearing(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set — GPU decode fixture")
	}
	if _, err := os.Stat(laneSetABModelDir); err != nil {
		t.Skipf("E2B model dir not present (%s) — set LTHN_CB_AB_MODEL", laneSetABModelDir)
	}
	// The q8 KV cache is its own separate fold rung (gemmEligible declines
	// icb.kvQ8 outright) — load this receipt's model with bf16 KV so the run
	// isolates exactly the three envelope arms under test.
	kvQ8ICBOffForTest = true
	t.Cleanup(func() { kvQ8ICBOffForTest = false })
	tm, err := LoadTokenModelDir(laneSetABModelDir, 4096)
	if err != nil {
		t.Fatalf("LoadTokenModelDir: %v", err)
	}
	m, ok := tm.(*NativeTokenModel)
	if !ok {
		if c, ok := tm.(interface{ Close() error }); ok {
			_ = c.Close()
		}
		t.Skipf("loaded model is %T, not *NativeTokenModel", tm)
	}
	defer m.Close()

	gemmEnvelopeLiftForTest = true
	defer func() { gemmEnvelopeLiftForTest = false }()

	specs := laneSpecFixtures()
	ctx := context.Background()
	open := func(mode int) (*laneSet, []inference.LaneHandle) {
		lsi, err := m.OpenLaneSet(inference.LaneSetConfig{MaxLanes: len(specs)})
		if err != nil {
			t.Fatalf("OpenLaneSet(mode %d): %v", mode, err)
		}
		ls, ok := lsi.(*laneSet)
		if !ok {
			t.Fatalf("OpenLaneSet returned %T, not *laneSet", lsi)
		}
		ls.gemmMode = mode
		handles := make([]inference.LaneHandle, len(specs))
		for i, spec := range specs {
			if handles[i], err = ls.Prepare(ctx, spec); err != nil {
				t.Fatalf("Prepare(mode %d lane %d): %v", mode, i, err)
			}
		}
		return ls, handles
	}

	lsG, hG := open(1)
	lsR, hR := open(2)
	defer lsG.Close()
	defer lsR.Close()

	diverged := false
	for step := 0; !diverged; step++ {
		sg, err := lsG.Step(ctx)
		if err != nil {
			t.Fatalf("step %d GEMM Step: %v", step, err)
		}
		sr, err := lsR.Step(ctx)
		if err != nil {
			t.Fatalf("step %d replay Step: %v", step, err)
		}
		if len(sg) == 0 && len(sr) == 0 {
			break
		}
		for i := range specs {
			lg, lr := lsG.lanes[hG[i].ID], lsR.lanes[hR[i].ID]
			if lg == nil || lr == nil {
				continue
			}
			if !bytes.Equal(lg.hidden, lr.hidden) {
				diverged = true
			}
		}
		retireTerminal(t, lsG, sg)
		retireTerminal(t, lsR, sr)
	}

	if lsG.gemmFwdCount == 0 {
		t.Fatal("gemmFwdCount is zero — the lifted-envelope forward never fired (pin vacuous)")
	}
	if !diverged {
		t.Fatal("lifted-envelope forward matched the replay byte-for-byte — the mirrored arms appear fixed: lift the proven-envelope exclusions and promote this test to the positive receipt")
	}
	t.Logf("envelope exclusion is load-bearing: lifted forward fired (%d) and diverged from the replay", lsG.gemmFwdCount)
}

// laneSetBF16Q8FixtureMax builds a synthetic BF16-weight gemma4 at HeadDim 256
// (global attention, gqa2) so the recorded ICB arms the q8 KV cache
// (allocArchICBCaches: q8 on lhd 256/512 globals). BF16 weights are DELIBERATE:
// the fold's bf16 batched gemv is byte-identical to the per-lane replay
// (bf16Projector.rowsByteTier), so this fixture isolates the q8-KV staging rung
// from the register-tiled QUANT projection, which is NOT byte-identical (the
// hd-256 fold divergence, root-caused 2026-07-13 — see gemmEligible's safety
// gate). A quant-weight q8 fold waits on a byte-identical batched quant kernel.
func laneSetBF16Q8FixtureMax(t *testing.T, maxLen int) *NativeTokenModel {
	t.Helper()
	const dModel, nHeads, nKV, headDim, dFF, vocab, nLayers = 256, 2, 1, 256, 512, 48, 2
	g, arch := gemma4BF16Fixture(t, dModel, nHeads, nKV, headDim, dFF, vocab, nLayers)
	m, err := NewBF16TokenModel(g, arch, maxLen)
	if err != nil {
		t.Fatalf("NewBF16TokenModel: %v", err)
	}
	probe, err := m.OpenSession()
	if err != nil {
		t.Fatalf("OpenSession: %v", err)
	}
	as, ok := probe.(*ArchSession)
	if !ok || as.state.icb == nil {
		_ = probe.(interface{ Close() error }).Close()
		_ = m.Close()
		t.Skip("bf16 hd-256 fixture is not recorded-ICB eligible on this metallib")
	}
	_ = as.Close()
	return m
}

// TestLaneSetGEMMQ8KVByteIdentityHiddens receipts the fold's q8-KV staging rung
// (#367 x #35): with the q8 KV cache armed, the weight-read-once forward mirrors
// the recorded quantise-store + q8-read SDPA ops per lane and must produce
// byte-for-byte identical post-stack hiddens to the per-lane ICB replay,
// counter-guarded. Runs at two maxLens: below the 2-pass knee (single-pass q8
// SDPA layout) and at it (the recorder bakes the fixed 2-pass fan on global
// layers — the fold must reproduce the identical fan for the reduction order
// to match). Weights are BF16 (byte-identical projection path) so the receipt
// isolates the q8 KV ops from the non-byte-identical quant tiled kernel — see
// laneSetBF16Q8FixtureMax.
func TestLaneSetGEMMQ8KVByteIdentityHiddens(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set — GPU decode fixture")
	}
	for _, tc := range []struct {
		name   string
		maxLen int
	}{
		{"singlePass", 32},
		{"twoPassFixedFan", sdpa2PassMinKV},
	} {
		t.Run(tc.name, func(t *testing.T) {
			kvQ8ICBForTest = true
			t.Cleanup(func() { kvQ8ICBForTest = false })
			m := laneSetBF16Q8FixtureMax(t, tc.maxLen)
			defer m.Close()
			specs := laneSpecFixtures()
			ctx := context.Background()

			open := func(mode int) (*laneSet, []inference.LaneHandle) {
				lsi, err := m.OpenLaneSet(inference.LaneSetConfig{MaxLanes: len(specs)})
				if err != nil {
					t.Fatalf("OpenLaneSet(mode %d): %v", mode, err)
				}
				ls, ok := lsi.(*laneSet)
				if !ok {
					t.Fatalf("OpenLaneSet returned %T, not *laneSet", lsi)
				}
				ls.gemmMode = mode
				handles := make([]inference.LaneHandle, len(specs))
				for i, spec := range specs {
					if handles[i], err = ls.Prepare(ctx, spec); err != nil {
						t.Fatalf("Prepare(mode %d lane %d): %v", mode, i, err)
					}
				}
				return ls, handles
			}

			lsG, hG := open(1)
			lsR, hR := open(2)
			defer lsG.Close()
			defer lsR.Close()

			// Pin that the sessions really carry q8 KV — else the receipt is vacuous.
			if st := lsG.lanes[hG[0].ID].sess.state; st.icb == nil || st.icb.kvQ8 == nil {
				t.Fatal("fixture sessions did not arm the q8 KV cache")
			}

			for step := 0; ; step++ {
				sg, err := lsG.Step(ctx)
				if err != nil {
					t.Fatalf("step %d GEMM Step: %v", step, err)
				}
				sr, err := lsR.Step(ctx)
				if err != nil {
					t.Fatalf("step %d replay Step: %v", step, err)
				}
				if len(sg) == 0 && len(sr) == 0 {
					break
				}
				for i := range specs {
					lg, lr := lsG.lanes[hG[i].ID], lsR.lanes[hR[i].ID]
					if lg == nil || lr == nil {
						continue
					}
					if !bytes.Equal(lg.hidden, lr.hidden) {
						t.Fatalf("step %d lane %d: post-stack hidden bytes differ between q8 GEMM and replay", step, i)
					}
				}
				tokG := stepTokens(sg, hG)
				tokR := stepTokens(sr, hR)
				if !slices.Equal(tokG, tokR) {
					t.Fatalf("step %d: token vector diverges GEMM %v vs replay %v", step, tokG, tokR)
				}
				retireTerminal(t, lsG, sg)
				retireTerminal(t, lsR, sr)
			}

			if lsG.gemmFwdCount == 0 {
				t.Fatal("gemmFwdCount is zero — the q8 weight-read-once forward never fired (proof vacuous)")
			}
			if lsR.gemmFwdCount != 0 {
				t.Fatalf("replay set ran %d GEMM forwards, want 0", lsR.gemmFwdCount)
			}
		})
	}
}
