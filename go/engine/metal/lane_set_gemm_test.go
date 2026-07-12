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

// TestLaneSetGEMMQuantFallsBackToReplay documents the evidenced quant blocker as a
// passing guard: a 4-bit checkpoint (E2B) is NOT batched-GEMM-eligible, so even with
// the forward armed (gemmMode 1) the lane set runs the per-lane ICB replay — the
// merged 2.58× path — byte-for-byte, and the GEMM counter stays zero.
//
// WHY the quant path is excluded: the quant ICB FUSES the entry/MLP rms INTO the qmv
// (setRMSQMV, decode_forward_arch_icb_quant.go:677) keeping the normed activation in
// fp32 through the matmul. Batching the weight read across lanes needs a separately
// materialised (bf16-rounded) normed slab fed to qmv-rows, which rounds one ulp
// differently and DIVERGES (a bisect showed byte-identical layer 0, one-ulp drift at
// layer 1, exploding by the first global layer). No batched rms-qmv-rows kernel
// exists and the metallib is read-only, so byte-identity + weight-read-once cannot
// both hold for the fused-rms quant path this rung — it stays on the replay.
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
		t.Fatalf("quant E2B ran %d batched-GEMM forwards — it must fall back to the replay (fused rms-qmv is not byte-identical to batched qmv-rows)", armedFwd)
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
