// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"context"
	"os"
	"slices"
	"testing"

	"dappco.re/go/inference"
)

// laneSetFixtureModel builds a synthetic uniform dense gemma4 token model — no
// sliding, no MoE, simple rope — so every session it opens is recorded-ICB
// eligible, the arch shape the multi-session owner serves.
func laneSetFixtureModel(t *testing.T) *NativeTokenModel {
	t.Helper()
	const dModel, nHeads, nKV, headDim, dFF, vocab, nLayers = 128, 2, 1, 64, 256, 48, 2
	g, arch := gemma4BF16Fixture(t, dModel, nHeads, nKV, headDim, dFF, vocab, nLayers)
	m, err := NewBF16TokenModel(g, arch, 32)
	if err != nil {
		t.Fatalf("NewBF16TokenModel: %v", err)
	}
	// Pin that the fixture really exercises the ICB path (else the owner would
	// have declined it and the whole proof would be vacuous).
	probe, err := m.OpenSession()
	if err != nil {
		t.Fatalf("OpenSession: %v", err)
	}
	as, ok := probe.(*ArchSession)
	if !ok || as.state.icb == nil {
		_ = probe.(interface{ Close() error }).Close()
		t.Skip("fixture model is not recorded-ICB eligible on this metallib — owner path not exercised")
	}
	_ = as.Close()
	return m
}

// laneSpecFixtures are varied prompt fills (content AND length) — the ragged
// admission surface the owner must serve byte-identically.
func laneSpecFixtures() []inference.LaneSpec {
	return []inference.LaneSpec{
		{PromptIDs: []int32{1, 5, 3, 2}, MaxNew: 8},
		{PromptIDs: []int32{7, 7, 1}, MaxNew: 8},
		{PromptIDs: []int32{2, 9, 4, 6, 8, 3}, MaxNew: 8},
		{PromptIDs: []int32{4}, MaxNew: 8},
	}
}

// drainLaneSet runs every admitted lane to completion, returning each lane's
// full token stream keyed by lane id, and the number of batched forwards taken.
func drainLaneSet(t *testing.T, ls inference.LaneSet) (map[int][]int32, uint64) {
	t.Helper()
	ctx := context.Background()
	out := map[int][]int32{}
	for {
		steps, err := ls.Step(ctx)
		if err != nil {
			t.Fatalf("Step: %v", err)
		}
		if len(steps) == 0 {
			break
		}
		for _, s := range steps {
			if s.HasToken {
				out[s.Lane.ID] = append(out[s.Lane.ID], s.Token)
			}
			if s.Terminal {
				if err := ls.Retire(s.Lane); err != nil {
					t.Fatalf("Retire: %v", err)
				}
			}
		}
	}
	return out, ls.BatchForwardCount()
}

// TestLaneSetByteIdentity is the counter-guarded conformance proof for the
// multi-session owner: the SAME lane specs produce the SAME per-lane token
// streams whether each is run ALONE (K==1, one lane per lane set) or ALL
// TOGETHER (K>1, one shared batched forward per step). Byte-identity across that
// difference is the whole correctness claim — fusing K lanes into one command
// buffer changes submission and scheduling, never a lane's arithmetic. The
// counter proves the K-way path actually fired: the batched run advances K lanes
// with the batched-forward count of a SINGLE lane, not K× it (which K disguised
// single-session steps would show).
func TestLaneSetByteIdentity(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set — GPU decode fixture")
	}
	m := laneSetFixtureModel(t)
	defer m.Close()
	specs := laneSpecFixtures()
	ctx := context.Background()

	// Serial oracle: each spec run alone through its own lane set.
	serialTokens := make([][]int32, len(specs))
	var serialFwdSingle uint64
	for i, spec := range specs {
		ls, err := m.OpenLaneSet(inference.LaneSetConfig{MaxLanes: 1})
		if err != nil {
			t.Fatalf("OpenLaneSet(serial %d): %v", i, err)
		}
		h, err := ls.Prepare(ctx, spec)
		if err != nil {
			t.Fatalf("Prepare(serial %d): %v", i, err)
		}
		got, fwd := drainLaneSet(t, ls)
		serialTokens[i] = got[h.ID]
		if len(serialTokens[i]) == 0 {
			t.Fatalf("serial lane %d produced no tokens", i)
		}
		// Every equal-length lane run alone takes the same batched-forward count;
		// capture one to compare against the K-way run below.
		if i == 0 {
			serialFwdSingle = fwd
		}
		_ = ls.Close()
	}

	// Batched: all specs admitted into ONE lane set, advanced together.
	ls, err := m.OpenLaneSet(inference.LaneSetConfig{MaxLanes: len(specs)})
	if err != nil {
		t.Fatalf("OpenLaneSet(batched): %v", err)
	}
	handles := make([]inference.LaneHandle, len(specs))
	for i, spec := range specs {
		if handles[i], err = ls.Prepare(ctx, spec); err != nil {
			t.Fatalf("Prepare(batched %d): %v", i, err)
		}
	}
	batchedTokens, batchedFwd := drainLaneSet(t, ls)
	// Engagement guard for the batched Phase-1 head: the K-way greedy must
	// have taken the K-row fused submission at least once — a silent decline
	// to the per-lane ladder would make this proof vacuous.
	if rows := ls.(*laneSet).headRowsCount; rows == 0 {
		t.Fatal("batched Phase-1 head never engaged (headRowsCount == 0)")
	}
	_ = ls.Close()

	// Per-lane byte-identity: batched stream == serial stream, for every lane.
	for i := range specs {
		want := serialTokens[i]
		got := batchedTokens[handles[i].ID]
		if !slices.Equal(got, want) {
			t.Fatalf("lane %d: batched tokens %v != serial %v", i, got, want)
		}
	}

	// Counter guard: the batched run must have executed strictly FEWER forwards
	// than running the lanes serially would (K lanes advanced per forward, not 1),
	// and no more than a single lane's own forward count — proof of one shared
	// forward per step rather than K disguised single-session steps.
	if batchedFwd == 0 {
		t.Fatal("BatchForwardCount is zero — the batched path never fired")
	}
	if batchedFwd > serialFwdSingle {
		t.Fatalf("batched forwards %d exceed a single lane's %d — lanes were not sharing a forward", batchedFwd, serialFwdSingle)
	}
	serialTotal := uint64(len(specs)) * serialFwdSingle
	if batchedFwd >= serialTotal {
		t.Fatalf("batched forwards %d not fewer than serial total %d — no batching occurred", batchedFwd, serialTotal)
	}
}

// laneSetMoEFixtureModel builds a synthetic gemma4 model whose second layer is
// MoE — icbEligible declines the router block, so no session records an ICB and
// every lane the owner admits is a RE-ENCODE lane: the arch shape the 26B-class
// checkpoints serve. The probe pins that the fixture really lacks the ICB (a
// recorded fixture would exercise the replay path and prove nothing here).
func laneSetMoEFixtureModel(t *testing.T) *NativeTokenModel {
	t.Helper()
	const dModel, nHeads, nKV, headDim, dFF, vocab, nLayers = 128, 2, 1, 64, 256, 48, 2
	g, arch := gemma4BF16Fixture(t, dModel, nHeads, nKV, headDim, dFF, vocab, nLayers)
	g.Layers[1].MoE = buildMoEWeights(4, 2, dModel, dFF, 192, 700)
	arch.Layer[1].MoE = true
	m, err := NewBF16TokenModel(g, arch, 32)
	if err != nil {
		t.Fatalf("NewBF16TokenModel(MoE): %v", err)
	}
	probe, err := m.OpenSession()
	if err != nil {
		t.Fatalf("OpenSession: %v", err)
	}
	as, ok := probe.(*ArchSession)
	if !ok {
		t.Fatal("MoE fixture session is not an ArchSession")
	}
	if as.state.icb != nil {
		_ = as.Close()
		t.Fatal("MoE fixture recorded an ICB — the re-encode lane path would be vacuous")
	}
	_ = as.Close()
	return m
}

// TestLaneSetMoEReencodeByteIdentity is the conformance proof for RE-ENCODE
// lanes on the PER-LANE path (bf16 MoE keeps the host handoff, so it is
// shared-submission INELIGIBLE): the SAME lane specs produce the SAME per-lane
// token streams run alone (K==1) and all together (K>1). No shared submission
// fires for ineligible lanes, so BatchForwardCount must stay ZERO on both
// runs — an honest counter, not a disguised one.
func TestLaneSetMoEReencodeByteIdentity(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set — GPU decode fixture")
	}
	m := laneSetMoEFixtureModel(t)
	defer m.Close()
	specs := laneSpecFixtures()
	ctx := context.Background()

	serialTokens := make([][]int32, len(specs))
	for i, spec := range specs {
		ls, err := m.OpenLaneSet(inference.LaneSetConfig{MaxLanes: 1})
		if err != nil {
			t.Fatalf("OpenLaneSet(serial %d): %v", i, err)
		}
		h, err := ls.Prepare(ctx, spec)
		if err != nil {
			t.Fatalf("Prepare(serial %d): %v", i, err)
		}
		got, fwd := drainLaneSet(t, ls)
		serialTokens[i] = got[h.ID]
		if len(serialTokens[i]) == 0 {
			t.Fatalf("serial MoE lane %d produced no tokens", i)
		}
		if fwd != 0 {
			t.Fatalf("serial MoE lane %d claimed %d batched forwards, want 0 (re-encode lanes share no submission)", i, fwd)
		}
		_ = ls.Close()
	}

	ls, err := m.OpenLaneSet(inference.LaneSetConfig{MaxLanes: len(specs)})
	if err != nil {
		t.Fatalf("OpenLaneSet(batched): %v", err)
	}
	handles := make([]inference.LaneHandle, len(specs))
	for i, spec := range specs {
		if handles[i], err = ls.Prepare(ctx, spec); err != nil {
			t.Fatalf("Prepare(batched %d): %v", i, err)
		}
	}
	batchedTokens, batchedFwd := drainLaneSet(t, ls)
	_ = ls.Close()

	for i := range specs {
		if !slices.Equal(batchedTokens[handles[i].ID], serialTokens[i]) {
			t.Fatalf("MoE lane %d: batched tokens %v != serial %v", i, batchedTokens[handles[i].ID], serialTokens[i])
		}
	}
	if batchedFwd != 0 {
		t.Fatalf("batched MoE set claimed %d batched forwards, want 0 (re-encode lanes share no submission)", batchedFwd)
	}
}

// laneSetQuantMoEFixtureModel builds a synthetic QUANT gemma4 model whose
// second layer is MoE on the fully-encoded device-router lane — no recorded
// ICB (re-encode lanes), but shared-submission ELIGIBLE: the 26B-A4B shape.
// The probes pin both properties, else the shared-path proof is vacuous.
func laneSetQuantMoEFixtureModel(t *testing.T) *NativeTokenModel {
	t.Helper()
	const dModel, nHeads, nKV, headDim, dFF, vocab, nLayers = 128, 2, 1, 64, 256, 48, 2
	const gs, bits = 32, 4
	arch := archFixture(t, dModel, nHeads, nKV, headDim, dFF, vocab, nLayers)
	arch.Layer[1].MoE = true
	qlayers := make([]QuantizedLayerWeights, nLayers)
	for i := range qlayers {
		qlayers[i] = quantizedLayerFixture(t, dModel, nHeads, nKV, headDim, dFF, gs, bits, (i+1)*100)
	}
	moe := quantMoELayerWeightsGuard(t, 4, 2, dModel, dFF, 96, gs, bits)
	qlayers[1].MoE = &moe
	embed := quantWeightFixture(t, vocab, dModel, gs, bits, 11)
	g := &QuantModel{
		Layers: qlayers,
		Embed:  embed.Packed, EmbedScales: embed.Scales, EmbedBiases: embed.Biases,
		// tied: the head reuses the embedding triple (loadedToQuant's shape)
		LMHead: embed.Packed, LMHeadScales: embed.Scales, LMHeadBiases: embed.Biases,
		FinalNorm: toBF16Bytes(syntheticFloat32(dModel, 7)),
		Tied:      true, GroupSize: gs, Bits: bits,
	}
	m, err := NewQuantTokenModel(g, arch, 32)
	if err != nil {
		t.Fatalf("NewQuantTokenModel(MoE): %v", err)
	}
	probe, err := m.OpenSession()
	if err != nil {
		t.Fatalf("OpenSession: %v", err)
	}
	as, ok := probe.(*ArchSession)
	if !ok {
		t.Fatal("quant MoE fixture session is not an ArchSession")
	}
	if as.state.icb != nil {
		_ = as.Close()
		t.Fatal("quant MoE fixture recorded an ICB — re-encode lanes would be vacuous")
	}
	if !as.sharedStepEligible() {
		_ = as.Close()
		t.Fatal("quant MoE fixture is not shared-encode eligible — the shared-submission proof would be vacuous")
	}
	_ = as.Close()
	return m
}

// TestLaneSetMoESharedSubmissionByteIdentity is the conformance proof for the
// SHARED re-encode submission (quant device-router MoE — the 26B-A4B lane):
// identical per-lane token streams run alone (K==1) and all together (K>1),
// with the counter guard proving the shared path actually fired — the batched
// run advances K lanes with a SINGLE lane's forward count, not K× it.
func TestLaneSetMoESharedSubmissionByteIdentity(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set — GPU decode fixture")
	}
	m := laneSetQuantMoEFixtureModel(t)
	defer m.Close()
	specs := laneSpecFixtures()
	ctx := context.Background()

	serialTokens := make([][]int32, len(specs))
	var serialFwdSingle uint64
	for i, spec := range specs {
		ls, err := m.OpenLaneSet(inference.LaneSetConfig{MaxLanes: 1})
		if err != nil {
			t.Fatalf("OpenLaneSet(serial %d): %v", i, err)
		}
		h, err := ls.Prepare(ctx, spec)
		if err != nil {
			t.Fatalf("Prepare(serial %d): %v", i, err)
		}
		got, fwd := drainLaneSet(t, ls)
		serialTokens[i] = got[h.ID]
		if len(serialTokens[i]) == 0 {
			t.Fatalf("serial quant MoE lane %d produced no tokens", i)
		}
		if i == 0 {
			serialFwdSingle = fwd
		}
		_ = ls.Close()
	}

	ls, err := m.OpenLaneSet(inference.LaneSetConfig{MaxLanes: len(specs)})
	if err != nil {
		t.Fatalf("OpenLaneSet(batched): %v", err)
	}
	handles := make([]inference.LaneHandle, len(specs))
	for i, spec := range specs {
		if handles[i], err = ls.Prepare(ctx, spec); err != nil {
			t.Fatalf("Prepare(batched %d): %v", i, err)
		}
	}
	batchedTokens, batchedFwd := drainLaneSet(t, ls)
	// Engagement guard for the batched Phase-1 head (see the dense twin).
	if rows := ls.(*laneSet).headRowsCount; rows == 0 {
		t.Fatal("batched Phase-1 head never engaged (headRowsCount == 0)")
	}
	_ = ls.Close()

	for i := range specs {
		if !slices.Equal(batchedTokens[handles[i].ID], serialTokens[i]) {
			t.Fatalf("quant MoE lane %d: batched tokens %v != serial %v", i, batchedTokens[handles[i].ID], serialTokens[i])
		}
	}
	if batchedFwd == 0 {
		t.Fatal("BatchForwardCount is zero — the shared re-encode submission never fired")
	}
	if batchedFwd > serialFwdSingle {
		t.Fatalf("batched forwards %d exceed a single lane's %d — lanes were not sharing a submission", batchedFwd, serialFwdSingle)
	}
	if serialTotal := uint64(len(specs)) * serialFwdSingle; batchedFwd >= serialTotal {
		t.Fatalf("batched forwards %d not fewer than serial total %d — no sharing occurred", batchedFwd, serialTotal)
	}
}

// TestLaneSetMoESharedMatchesProductionGreedy pins the shared submission to
// the engine's own DEFAULT serial decode — the chained-live tail included.
// (This oracle briefly had to pin the host arm: the submit-ahead link's
// shared position buffer corrupted in-flight links, forking the chain from
// the host loop. rotateOffBuf closed the fork — TestChainedDecodeArmsAgree
// holds the arms together — so the lane gates face the production default
// again.)
func TestLaneSetMoESharedMatchesProductionGreedy(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set — GPU decode fixture")
	}
	m := laneSetQuantMoEFixtureModel(t)
	defer m.Close()
	ctx := context.Background()

	for i, spec := range laneSpecFixtures() {
		prod, err := m.OpenSession()
		if err != nil {
			t.Fatalf("OpenSession(%d): %v", i, err)
		}
		as := prod.(*ArchSession)
		want, err := as.Generate(spec.PromptIDs, spec.MaxNew, -1)
		if err != nil {
			t.Fatalf("Generate(%d): %v", i, err)
		}
		_ = as.Close()

		ls, err := m.OpenLaneSet(inference.LaneSetConfig{MaxLanes: 1})
		if err != nil {
			t.Fatalf("OpenLaneSet(%d): %v", i, err)
		}
		h, err := ls.Prepare(ctx, spec)
		if err != nil {
			t.Fatalf("Prepare(%d): %v", i, err)
		}
		got, _ := drainLaneSet(t, ls)
		_ = ls.Close()

		if !slices.Equal(got[h.ID], want) {
			t.Fatalf("quant MoE lane %d: owner tokens %v != production Generate %v", i, got[h.ID], want)
		}
	}
}

// TestLaneSetMoEReencodeMatchesProductionGreedy pins re-encode lanes to the
// engine's own serial decode: an MoE lane's stream equals ArchSession.Generate
// token-for-token — it IS the plain path's step, run under the owner.
func TestLaneSetMoEReencodeMatchesProductionGreedy(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set — GPU decode fixture")
	}
	m := laneSetMoEFixtureModel(t)
	defer m.Close()
	ctx := context.Background()

	for i, spec := range laneSpecFixtures() {
		prod, err := m.OpenSession()
		if err != nil {
			t.Fatalf("OpenSession(%d): %v", i, err)
		}
		as := prod.(*ArchSession)
		want, err := as.Generate(spec.PromptIDs, spec.MaxNew, -1)
		if err != nil {
			t.Fatalf("Generate(%d): %v", i, err)
		}
		_ = as.Close()

		ls, err := m.OpenLaneSet(inference.LaneSetConfig{MaxLanes: 1})
		if err != nil {
			t.Fatalf("OpenLaneSet(%d): %v", i, err)
		}
		h, err := ls.Prepare(ctx, spec)
		if err != nil {
			t.Fatalf("Prepare(%d): %v", i, err)
		}
		got, _ := drainLaneSet(t, ls)
		_ = ls.Close()

		if !slices.Equal(got[h.ID], want) {
			t.Fatalf("MoE lane %d: owner tokens %v != production Generate %v", i, got[h.ID], want)
		}
	}
}

// TestLaneSetMatchesProductionGreedy is the secondary oracle: the owner's greedy
// decode also equals the production single-session greedy path (ArchSession.
// Generate) token-for-token, so the batched owner is not merely self-consistent
// but tracks the engine's own serial decode.
func TestLaneSetMatchesProductionGreedy(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set — GPU decode fixture")
	}
	m := laneSetFixtureModel(t)
	defer m.Close()
	ctx := context.Background()

	for i, spec := range laneSpecFixtures() {
		// Production greedy over a fresh session.
		prod, err := m.OpenSession()
		if err != nil {
			t.Fatalf("OpenSession(%d): %v", i, err)
		}
		as := prod.(*ArchSession)
		want, err := as.Generate(spec.PromptIDs, spec.MaxNew, -1)
		if err != nil {
			t.Fatalf("Generate(%d): %v", i, err)
		}
		_ = as.Close()

		// Owner greedy over a single lane.
		ls, err := m.OpenLaneSet(inference.LaneSetConfig{MaxLanes: 1})
		if err != nil {
			t.Fatalf("OpenLaneSet(%d): %v", i, err)
		}
		h, err := ls.Prepare(ctx, spec)
		if err != nil {
			t.Fatalf("Prepare(%d): %v", i, err)
		}
		got, _ := drainLaneSet(t, ls)
		_ = ls.Close()

		if !slices.Equal(got[h.ID], want) {
			t.Fatalf("lane %d: owner tokens %v != production Generate %v", i, got[h.ID], want)
		}
	}
}
