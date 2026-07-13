// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference/model"
	g4 "dappco.re/go/inference/model/gemma4"
	"dappco.re/go/inference/model/safetensors"
)

// addPLETensorsBF16 mints a DENSE (bf16) per-layer-input tower for a synthetic
// gemma4 arch — the bf16 twin of addPLETensors' quant tower. E2B/E4B ship PLE,
// and their bf16 checkpoints are exactly the shape the batched dense prefill
// must serve.
func addPLETensorsBF16(t testing.TB, ts map[string]safetensors.Tensor, arch model.Arch) {
	t.Helper()
	vocabPLI, numLayers, pliDim, dModel := arch.PerLayerInputVocab, len(arch.Layer), arch.PerLayerInputHidden, arch.Hidden
	plDim := numLayers * pliDim
	salt := 150
	mk := func(name string, shape []int) {
		elems := 1
		for _, d := range shape {
			elems *= d
		}
		f := make([]float32, elems)
		for i := range f {
			f[i] = float32((i*salt+11)%79-39) * 0.02
		}
		ts[name] = safetensors.Tensor{Dtype: "BF16", Shape: shape, Data: toBF16Bytes(f)}
		salt++
	}
	mk("model.embed_tokens_per_layer.weight", []int{vocabPLI, plDim})
	mk("model.per_layer_model_projection.weight", []int{plDim, dModel})
	mk("model.per_layer_projection_norm.weight", []int{pliDim})
	for i := range numLayers {
		p := core.Sprintf("model.layers.%d", i)
		mk(p+".per_layer_input_gate.weight", []int{pliDim, dModel})
		mk(p+".per_layer_projection.weight", []int{dModel, pliDim})
		mk(p+".post_per_layer_input_norm.weight", []int{dModel})
	}
}

func newBatchedPLEBF16Fixture(t testing.TB) *ArchSession {
	return newBatchedPLEBF16FixtureShared(t, 0)
}

// newBatchedPLEBF16FixtureShared builds the E-family shape at synthetic scale: a bf16
// gemma4 arch with the per-layer-input tower and, when kvShared > 0, a shared-KV tail
// (the last kvShared layers attend an owner's cache — real E2B carries 20 of these).
func newBatchedPLEBF16FixtureShared(t testing.TB, kvShared int) *ArchSession {
	t.Helper()
	const dModel, nHeads, nKV, headDim, dFF, vocab = 128, 2, 1, 64, 256, 32
	const numLayers, pliDim = 4, 64
	const maxLen = 16
	cfg := g4.Config{
		HiddenSize: dModel, NumHiddenLayers: numLayers, IntermediateSize: dFF,
		NumAttentionHeads: nHeads, NumKeyValueHeads: nKV, HeadDim: headDim,
		VocabSize: vocab, RMSNormEps: 1e-6,
		HiddenSizePerLayerInput: pliDim, VocabSizePerLayerInput: vocab,
		NumKVSharedLayers: kvShared,
	}
	arch, err := cfg.Arch()
	if err != nil {
		t.Fatalf("Arch: %v", err)
	}
	ts, _ := gemma4Tensors(arch, false)
	addPLETensorsBF16(t, ts, arch)
	lm, err := model.Assemble(ts, arch, model.StandardWeightNames())
	if err != nil {
		t.Fatalf("model.Assemble: %v", err)
	}
	g := loadedToBF16(lm)
	if !g.HasPLE() {
		t.Fatal("fixture model should have PLE")
	}
	sess, err := NewArchSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("NewArchSession: %v", err)
	}
	t.Cleanup(func() { sess.Close() })
	// bf16 sessions record the arch ICB now (recordArchICBBF16) and route SHORT appends
	// (≤ batchedDenseICBMaxRows) to the chained replay lane — which would decline this
	// suite's 8-token prompts. THIS suite pins the batched-dense prefill body itself (the
	// prompt-scale lane on recorded-ICB sessions, and the whole lane for non-eligible
	// ones), so force the stepToken/batched pair; the ICB lane has its own parity gate
	// (TestArchSessionICBParityBF16*).
	sess.state.icb = nil
	return sess
}

// newBatchedPLEQuantFixture builds a 4-bit gemma4 arch WITH the per-layer-input tower — the
// quant twin of newBatchedPLEBF16Fixture. addPLETensors (per_layer_session_test.go) quantises
// the PLE table AND every per-layer gate/projection weight while the TOWER projection
// (model.per_layer_model_projection.weight) stays bf16-resident — the #381 "plain" e2b/e4b
// profile arch_session.go wires into ArchSession.perLayerInputBatch/-Device at
// NewArchQuantSession's affineBitsSupported && EmbedPerLayerScales>0 && PerLayerModelProjScales==0
// gate. Before r3 nothing drove a quantised-PLE model through the SESSION at batch scale: only
// the bare perLayerInputsBatchQuantIntoSlab/-Device builders (per_layer_batch_slab_test.go) or
// the 3-4 token per-token TestLoadGemma4QuantPLE chain — decode_batched_session.go's PLE-suffixed
// batched dense paths never saw a quant arch, and steelGEMMMinRows(32) was never reached PLE-side.
func newBatchedPLEQuantFixture(t testing.TB) *ArchSession {
	t.Helper()
	requireNativeRuntime(t)
	const dModel, nHeads, nKV, headDim, dFF, vocab = 128, 2, 1, 64, 256, 32
	const numLayers, pliDim, gs, bits = 2, 64, 64, 4
	const maxLen = 96
	cfg := g4.Config{
		HiddenSize: dModel, NumHiddenLayers: numLayers, IntermediateSize: dFF,
		NumAttentionHeads: nHeads, NumKeyValueHeads: nKV, HeadDim: headDim,
		VocabSize: vocab, RMSNormEps: 1e-6,
		HiddenSizePerLayerInput: pliDim, VocabSizePerLayerInput: vocab,
		Quantization: &model.QuantConfig{GroupSize: gs, Bits: bits},
	}
	arch, err := cfg.Arch()
	if err != nil {
		t.Fatalf("Arch: %v", err)
	}
	ts := quantGemma4Tensors(t, arch, gs, bits)
	addPLETensors(t, ts, arch, gs, bits)
	lm, err := model.Assemble(ts, arch, model.StandardWeightNames())
	if err != nil {
		t.Fatalf("model.Assemble: %v", err)
	}
	g, err := loadedToQuant(lm, gs, bits)
	if err != nil {
		t.Fatalf("loadedToQuant: %v", err)
	}
	if !g.HasPLE() {
		t.Fatal("fixture model should have PLE")
	}
	sess, err := NewArchQuantSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("NewArchQuantSession: %v", err)
	}
	t.Cleanup(func() { sess.Close() })
	// Recorded-ICB (quant) sessions batch prompt-scale runs too, but short appends decline the
	// batched-dense lane before any host PLE work (prefillRetainedTokensBatchedDenseOne's
	// icb!=nil guard) — force it off so THIS suite pins the batched-dense body itself, mirroring
	// newBatchedPLEBF16FixtureShared's forced sess.state.icb = nil above.
	sess.state.icb = nil
	return sess
}

// newBatchedPLEQuantFixtureShared is the quant twin of newBatchedPLEBF16FixtureShared: a 4-bit
// gemma4 arch WITH the per-layer-input tower, a clean trailing KV-shared suffix of kvShared
// layers (0 = none, mirroring the bf16 fixture), and an explicit sliding window. Unlike the bf16
// sibling, slidingWindow is a real parameter (not implicit 0): the #381 BOUNDED per-layer-input
// build (per_layer_batch.go's perLayerInputsBatchQuantEncode, nOut < numLayers at ~line 464) only
// ever runs when a prefill crosses the sliding-window ring wrap into multiple chunks — a
// kvShared-only fixture (newBatchedPLEQuantFixture's plain 2-layer shape, or this one called with
// slidingWindow=0) never leaves prefillRetainedTokensBatchedDenseOne, so nOut always equals
// numLayers (the unbounded arm r3 already covers). Layers are ALL sliding_attention (not the
// mixed sliding/full real E2B pattern) so the fixture needs no per-attention-type RoPE — only an
// OWNING sliding layer is required for verifyBatchedCrossesSlidingRingWrap to engage.
func newBatchedPLEQuantFixtureShared(t testing.TB, kvShared, slidingWindow int) *ArchSession {
	t.Helper()
	requireNativeRuntime(t)
	const dModel, nHeads, nKV, headDim, dFF, vocab = 128, 2, 1, 64, 256, 32
	const numLayers, pliDim, gs, bits = 4, 64, 64, 4
	const maxLen = 128
	layerTypes := make([]string, numLayers)
	for i := range layerTypes {
		layerTypes[i] = "sliding_attention"
	}
	cfg := g4.Config{
		HiddenSize: dModel, NumHiddenLayers: numLayers, IntermediateSize: dFF,
		NumAttentionHeads: nHeads, NumKeyValueHeads: nKV, HeadDim: headDim,
		VocabSize: vocab, RMSNormEps: 1e-6,
		HiddenSizePerLayerInput: pliDim, VocabSizePerLayerInput: vocab,
		Quantization:      &model.QuantConfig{GroupSize: gs, Bits: bits},
		NumKVSharedLayers: kvShared,
		SlidingWindow:     slidingWindow,
		LayerTypes:        layerTypes,
	}
	arch, err := cfg.Arch()
	if err != nil {
		t.Fatalf("Arch: %v", err)
	}
	ts := quantGemma4Tensors(t, arch, gs, bits)
	addPLETensors(t, ts, arch, gs, bits)
	lm, err := model.Assemble(ts, arch, model.StandardWeightNames())
	if err != nil {
		t.Fatalf("model.Assemble: %v", err)
	}
	g, err := loadedToQuant(lm, gs, bits)
	if err != nil {
		t.Fatalf("loadedToQuant: %v", err)
	}
	if !g.HasPLE() {
		t.Fatal("fixture model should have PLE")
	}
	sess, err := NewArchQuantSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("NewArchQuantSession: %v", err)
	}
	t.Cleanup(func() { sess.Close() })
	// mirrors newBatchedPLEQuantFixture: force the batched-dense lane on rather than the
	// recorded-ICB replay, so this suite pins the batched-dense body (and, via the sliding
	// window, the chunked skip-suffix body) itself.
	sess.state.icb = nil
	return sess
}

// quantBatchedPLEIDs returns a token batch at/above steelGEMMMinRows: below that floor
// perLayerInputsBatchQuantEncode declines and the quant slab builders never engage, silently
// degrading this suite back to the already-tested per-token PerLayerInputs chain
// (TestLoadGemma4QuantPLE) instead of exercising the batched builders r3 targets.
func quantBatchedPLEIDs() []int32 {
	const vocab = 32
	ids := make([]int32, steelGEMMMinRows+8)
	for i := range ids {
		ids[i] = int32((i*7 + 3) % vocab)
	}
	return ids
}

// quantBatchedPLESharedIDs is quantBatchedPLEIDs parametrised by count: the shared-suffix
// bounded-chunk test needs an EXACT multiple of the sliding window (so the chunk split lands
// where the test expects it), not just steelGEMMMinRows+8.
func quantBatchedPLESharedIDs(n int) []int32 {
	const vocab = 32
	ids := make([]int32, n)
	for i := range ids {
		ids[i] = int32((i*7 + 3) % vocab)
	}
	return ids
}

// TestPrefillRetainedTokensBatchedDenseEngagesPLEAndMatchesSequential pins the
// #252 fix: the batched dense prefill must ENGAGE on a PLE (E2B/E4B-shaped)
// bf16 arch and produce results byte-identical to the sequential per-token
// path. Today prefillRetainedTokensBatchedDenseOne declines any session with
// perLayerInput != nil, so every bf16 E-family prompt falls back to n full
// single-token forwards — O(n^2) total, the measured 44s/200-token prefill
// against metal's 175ms/600 tokens. The PLE input gate is an encoded device
// kernel fed from a per-token input buffer (no host readback), so a batched
// pass can upload one [K x layers*pliDim] slab and encode the same kernel with
// row offsets — the batched contract stays byte-identity with stepToken.
func TestPrefillRetainedTokensBatchedDenseEngagesPLEAndMatchesSequential(t *testing.T) {
	requireNativeRuntime(t)
	control, candidate := newBatchedPLEBF16Fixture(t), newBatchedPLEBF16Fixture(t)
	// The name's claim is executable: the fixture must actually carry the PLE
	// tower, or the parity below silently degrades to a dense-lane test.
	if candidate.arch.PerLayerInputHidden == 0 {
		t.Fatal("fixture lost its per-layer-input tower — this parity no longer exercises PLE")
	}
	batchedPLEParity(t, control, candidate)
}

// TestPrefillRetainedTokensBatchedDenseEngagesSharedKVAndMatchesSequential extends the
// batched-prefill contract to the E-family's OTHER gate: shared-KV tail layers (real E2B
// carries 20). The sharer attends its owner's cache; in the batch the owner's rows are
// encoded at a lower layer index in the same command buffer, so causality holds via the
// per-row SDPA length cap plus Metal's hazard ordering — proven here by byte-identity.
func TestPrefillRetainedTokensBatchedDenseEngagesSharedKVAndMatchesSequential(t *testing.T) {
	requireNativeRuntime(t)
	control, candidate := newBatchedPLEBF16FixtureShared(t, 2), newBatchedPLEBF16FixtureShared(t, 2)
	// The name's claim is executable: at least one layer must read an owner's
	// cache (KVShareFrom != own index), or the parity below silently degrades
	// to an unshared test.
	shared := 0
	for i, l := range candidate.arch.Layer {
		if l.KVShareFrom != i {
			shared++
		}
	}
	if shared == 0 {
		t.Fatal("fixture lost its shared-KV tail — this parity no longer exercises KV sharing")
	}
	batchedPLEParity(t, control, candidate)
}

// TestPrefillPromptRetainedInPoolBatchesLiveBoundaryAppend pins the multi-turn
// serve lane (#252 slice 3): appending a turn to a LIVE session (pos > 0 with a
// retained boundary — the prompt-cache suffix path) must ride the batched dense
// prefill, not fall to n host-synced single-token steps. A +540-token turn on a
// real E2B session took 6m28s down the per-token path while the identical fresh
// prompt batched in ~5s. Byte-identity with the sequential path stays the bar,
// and engagement is asserted via dispatch counts (batched ≈ tens of dispatches;
// per-token ≈ hundreds per appended token).
func TestPrefillPromptRetainedInPoolBatchesLiveBoundaryAppend(t *testing.T) {
	requireNativeRuntime(t)
	turn1 := []int32{3, 9, 17, 24}
	turn2 := []int32{6, 11, 29, 2, 21, 14, 8, 27}

	// both sessions establish the same live boundary: turn 1 + one decoded token.
	control := newBatchedPLEBF16Fixture(t)
	candidate := newBatchedPLEBF16Fixture(t)
	for _, s := range []*ArchSession{control, candidate} {
		if _, err := s.prefillRetainedTokens(turn1, "test.appendSetup"); err != nil {
			t.Fatalf("turn 1 prefill: %v", err)
		}
	}

	// control: the sequential per-token append (the old pool fallback).
	var ctrlHidden []byte
	var err error
	for _, id := range turn2[:len(turn2)-1] {
		if _, err = control.stepIDInPool(id); err != nil {
			t.Fatalf("control step: %v", err)
		}
	}
	if ctrlHidden, err = control.stepIDRetainedInPool(turn2[len(turn2)-1]); err != nil {
		t.Fatalf("control last step: %v", err)
	}

	// engagement: the batched lane must ACCEPT a live-boundary append (a third
	// session, same boundary). This is the lane the pool's gate routes to.
	engaged := newBatchedPLEBF16Fixture(t)
	if _, err := engaged.prefillRetainedTokens(turn1, "test.appendSetup"); err != nil {
		t.Fatalf("engaged turn 1: %v", err)
	}
	if _, ok, err := engaged.prefillRetainedTokensBatchedDense(turn2, "test.appendEngaged"); err != nil {
		t.Fatalf("batched append: %v", err)
	} else if !ok {
		t.Fatal("batched dense prefill DECLINED a live-boundary append — multi-turn serve pays the per-token path every turn (#252 slice 3)")
	}

	// candidate: the pool path the prompt cache calls for a turn suffix.
	hidden, err := candidate.prefillPromptRetainedInPool(turn2)
	if err != nil {
		t.Fatalf("candidate append: %v", err)
	}

	if candidate.Pos() != control.Pos() {
		t.Fatalf("pos after append: candidate=%d control=%d", candidate.Pos(), control.Pos())
	}
	if len(hidden) != len(ctrlHidden) {
		t.Fatalf("hidden sizes differ: candidate=%d control=%d", len(hidden), len(ctrlHidden))
	}
	for i := range hidden {
		if hidden[i] != ctrlHidden[i] {
			t.Fatalf("boundary hidden diverges at byte %d (batched append contract is byte-identity with stepping)", i)
		}
	}
	va, err := control.stateLayerViews()
	if err != nil {
		t.Fatalf("control views: %v", err)
	}
	vb, err := candidate.stateLayerViews()
	if err != nil {
		t.Fatalf("candidate views: %v", err)
	}
	for i := range va {
		for j := range va[i].keyBytes {
			if va[i].keyBytes[j] != vb[i].keyBytes[j] {
				t.Fatalf("layer %d K diverges at byte %d", i, j)
			}
		}
		for j := range va[i].valueBytes {
			if va[i].valueBytes[j] != vb[i].valueBytes[j] {
				t.Fatalf("layer %d V diverges at byte %d", i, j)
			}
		}
	}

}

// TestStepTokensBatchedDenseMLPFoldEngagesAndMatchesPerRow pins the MLP fold (#252): the batched
// dense pass folds each bf16 layer's MLP into one rms-rows + three batched gemvs + one fused gelu
// (grid Z carries the rows, each layer's gate/up/down weights read once instead of K times),
// byte-identical to the per-row interleave — and actually ENGAGES: the folded pass must encode
// strictly fewer dispatches than the per-row pass on the same batch, or the fold is dead code.
func TestStepTokensBatchedDenseMLPFoldEngagesAndMatchesPerRow(t *testing.T) {
	requireNativeRuntime(t)
	ids := []int32{3, 9, 17, 24, 6, 11, 29, 2}

	run := func(s *ArchSession, disableFold bool) ([]byte, int64) {
		t.Helper()
		prevFold, prevTiming := batchedMLPFoldDisabledForTest, pieceTimingOn
		batchedMLPFoldDisabledForTest = disableFold
		pieceTimingOn = true
		dispatchCountForTest = 0
		defer func() {
			batchedMLPFoldDisabledForTest = prevFold
			pieceTimingOn = prevTiming
		}()
		hidden, ok, err := s.prefillRetainedTokensBatchedDense(ids, "test.mlpFold")
		if err != nil {
			t.Fatalf("batched dense prefill (disableFold=%v): %v", disableFold, err)
		}
		if !ok {
			t.Fatalf("batched dense prefill DECLINED (disableFold=%v)", disableFold)
		}
		return append([]byte(nil), hidden...), dispatchCountForTest
	}

	folded := newBatchedPLEBF16FixtureShared(t, 2)
	perRow := newBatchedPLEBF16FixtureShared(t, 2)
	foldedHidden, foldedDispatches := run(folded, false)
	perRowHidden, perRowDispatches := run(perRow, true)

	if foldedDispatches >= perRowDispatches {
		t.Fatalf("MLP fold did not engage: folded pass encoded %d dispatches, per-row %d — the fold must strictly reduce dispatch count", foldedDispatches, perRowDispatches)
	}
	if len(foldedHidden) != len(perRowHidden) {
		t.Fatalf("hidden sizes differ: folded=%d perRow=%d", len(foldedHidden), len(perRowHidden))
	}
	for i := range foldedHidden {
		if foldedHidden[i] != perRowHidden[i] {
			t.Fatalf("boundary hidden diverges at byte %d: folded=%02x perRow=%02x (the fold contract is byte-identity with the per-row interleave)", i, foldedHidden[i], perRowHidden[i])
		}
	}
	va, err := perRow.stateLayerViews()
	if err != nil {
		t.Fatalf("per-row views: %v", err)
	}
	vb, err := folded.stateLayerViews()
	if err != nil {
		t.Fatalf("folded views: %v", err)
	}
	for i := range va {
		for j := range va[i].keyBytes {
			if va[i].keyBytes[j] != vb[i].keyBytes[j] {
				t.Fatalf("layer %d K diverges at byte %d", i, j)
			}
		}
		for j := range va[i].valueBytes {
			if va[i].valueBytes[j] != vb[i].valueBytes[j] {
				t.Fatalf("layer %d V diverges at byte %d", i, j)
			}
		}
	}
}

// TestStepTokensBatchedDenseMultiQSDPAEngagesAndMatchesPerRow pins the multi-query SDPA (#252):
// on a no-evict batch every (head, row) attention runs in ONE dispatch (grid Y carries the rows,
// the causal cap computed in-kernel), byte-identical to the per-row single-query dispatches — and
// it must actually ENGAGE: strictly fewer dispatches than the per-row SDPA path, or the kernel is
// dead code.
func TestStepTokensBatchedDenseMultiQSDPAEngagesAndMatchesPerRow(t *testing.T) {
	requireNativeRuntime(t)
	if !gpuHasSDPAMultiQ(64) {
		t.Fatal("multi-query SDPA kernel missing for headDim 64 — rebuild dist/lib/lthn_kernels.metallib (task build:kernels)")
	}
	ids := []int32{3, 9, 17, 24, 6, 11, 29, 2}

	run := func(s *ArchSession, disable bool) ([]byte, int64) {
		t.Helper()
		prev, prevTiming := sdpaMultiQDisabledForTest, pieceTimingOn
		sdpaMultiQDisabledForTest = disable
		pieceTimingOn = true
		dispatchCountForTest = 0
		defer func() {
			sdpaMultiQDisabledForTest = prev
			pieceTimingOn = prevTiming
		}()
		hidden, ok, err := s.prefillRetainedTokensBatchedDense(ids, "test.multiq")
		if err != nil {
			t.Fatalf("batched dense prefill (disableMultiQ=%v): %v", disable, err)
		}
		if !ok {
			t.Fatalf("batched dense prefill DECLINED (disableMultiQ=%v)", disable)
		}
		return append([]byte(nil), hidden...), dispatchCountForTest
	}

	multiq := newBatchedPLEBF16FixtureShared(t, 2)
	perRow := newBatchedPLEBF16FixtureShared(t, 2)
	mqHidden, mqDispatches := run(multiq, false)
	rowHidden, rowDispatches := run(perRow, true)

	if mqDispatches >= rowDispatches {
		t.Fatalf("multi-query SDPA did not engage: multiq pass encoded %d dispatches, per-row %d — the kernel must strictly reduce dispatch count", mqDispatches, rowDispatches)
	}
	if len(mqHidden) != len(rowHidden) {
		t.Fatalf("hidden sizes differ: multiq=%d perRow=%d", len(mqHidden), len(rowHidden))
	}
	for i := range mqHidden {
		if mqHidden[i] != rowHidden[i] {
			t.Fatalf("boundary hidden diverges at byte %d: multiq=%02x perRow=%02x (the multi-query kernel contract is byte-identity with the single-query dispatches)", i, mqHidden[i], rowHidden[i])
		}
	}
	va, err := perRow.stateLayerViews()
	if err != nil {
		t.Fatalf("per-row views: %v", err)
	}
	vb, err := multiq.stateLayerViews()
	if err != nil {
		t.Fatalf("multiq views: %v", err)
	}
	for i := range va {
		for j := range va[i].keyBytes {
			if va[i].keyBytes[j] != vb[i].keyBytes[j] {
				t.Fatalf("layer %d K diverges at byte %d", i, j)
			}
		}
		for j := range va[i].valueBytes {
			if va[i].valueBytes[j] != vb[i].valueBytes[j] {
				t.Fatalf("layer %d V diverges at byte %d", i, j)
			}
		}
	}
}

// TestStepTokensBatchedDenseBatchedRopeEngagesAndMatchesPerRow pins the batched-rows rope (#252):
// the K per-row fused QK-norm+rope dispatches (Q slab + direct K landing + value norm) fold into
// one dispatch each per layer, byte-identical to the per-row dispatches — and must actually
// ENGAGE (strictly fewer dispatches than the per-row rope path).
func TestStepTokensBatchedDenseBatchedRopeEngagesAndMatchesPerRow(t *testing.T) {
	requireNativeRuntime(t)
	if !gpuHasQKNormRopeRows() {
		t.Fatal("batched-rows qknorm-rope kernel missing — rebuild dist/lib/lthn_kernels.metallib (task build:kernels)")
	}
	ids := []int32{3, 9, 17, 24, 6, 11, 29, 2}

	run := func(s *ArchSession, disable bool) ([]byte, int64) {
		t.Helper()
		prev, prevTiming := batchedRopeDisabledForTest, pieceTimingOn
		batchedRopeDisabledForTest = disable
		pieceTimingOn = true
		dispatchCountForTest = 0
		defer func() {
			batchedRopeDisabledForTest = prev
			pieceTimingOn = prevTiming
		}()
		hidden, ok, err := s.prefillRetainedTokensBatchedDense(ids, "test.batchedRope")
		if err != nil {
			t.Fatalf("batched dense prefill (disableBatchedRope=%v): %v", disable, err)
		}
		if !ok {
			t.Fatalf("batched dense prefill DECLINED (disableBatchedRope=%v)", disable)
		}
		return append([]byte(nil), hidden...), dispatchCountForTest
	}

	batched := newBatchedPLEBF16FixtureShared(t, 2)
	perRow := newBatchedPLEBF16FixtureShared(t, 2)
	bHidden, bDispatches := run(batched, false)
	rHidden, rDispatches := run(perRow, true)

	if bDispatches >= rDispatches {
		t.Fatalf("batched rope did not engage: batched pass encoded %d dispatches, per-row %d — the rows kernel must strictly reduce dispatch count", bDispatches, rDispatches)
	}
	if len(bHidden) != len(rHidden) {
		t.Fatalf("hidden sizes differ: batched=%d perRow=%d", len(bHidden), len(rHidden))
	}
	for i := range bHidden {
		if bHidden[i] != rHidden[i] {
			t.Fatalf("boundary hidden diverges at byte %d: batched=%02x perRow=%02x (the batched rope contract is byte-identity with the per-row dispatches)", i, bHidden[i], rHidden[i])
		}
	}
	va, err := perRow.stateLayerViews()
	if err != nil {
		t.Fatalf("per-row views: %v", err)
	}
	vb, err := batched.stateLayerViews()
	if err != nil {
		t.Fatalf("batched views: %v", err)
	}
	for i := range va {
		for j := range va[i].keyBytes {
			if va[i].keyBytes[j] != vb[i].keyBytes[j] {
				t.Fatalf("layer %d K diverges at byte %d", i, j)
			}
		}
		for j := range va[i].valueBytes {
			if va[i].valueBytes[j] != vb[i].valueBytes[j] {
				t.Fatalf("layer %d V diverges at byte %d", i, j)
			}
		}
	}
}

// TestStepTokensBatchedDenseBatchedEpilogueEngagesAndMatchesPerRow pins the rows-batched layer
// tail (#252): the per-row entry-rms, residuals, PLE gate chain (5 dispatches/row) and layer
// scalar fold into a handful of dispatches per layer over the contiguous row slabs (the PLE slab
// is layer-major so each layer's K token slices batch through one gelu·pli), byte-identical to
// the per-row chain — and must actually ENGAGE (strictly fewer dispatches).
func TestStepTokensBatchedDenseBatchedEpilogueEngagesAndMatchesPerRow(t *testing.T) {
	requireNativeRuntime(t)
	if !gpuHasMulRowsKernel() {
		t.Fatal("rows-multiply kernel missing — rebuild dist/lib/lthn_kernels.metallib (task build:kernels)")
	}
	ids := []int32{3, 9, 17, 24, 6, 11, 29, 2}

	run := func(s *ArchSession, disable bool) ([]byte, int64) {
		t.Helper()
		prev, prevTiming := batchedEpilogueDisabledForTest, pieceTimingOn
		batchedEpilogueDisabledForTest = disable
		pieceTimingOn = true
		dispatchCountForTest = 0
		defer func() {
			batchedEpilogueDisabledForTest = prev
			pieceTimingOn = prevTiming
		}()
		hidden, ok, err := s.prefillRetainedTokensBatchedDense(ids, "test.batchedEpilogue")
		if err != nil {
			t.Fatalf("batched dense prefill (disableBatchedEpilogue=%v): %v", disable, err)
		}
		if !ok {
			t.Fatalf("batched dense prefill DECLINED (disableBatchedEpilogue=%v)", disable)
		}
		return append([]byte(nil), hidden...), dispatchCountForTest
	}

	batched := newBatchedPLEBF16FixtureShared(t, 2)
	perRow := newBatchedPLEBF16FixtureShared(t, 2)
	bHidden, bDispatches := run(batched, false)
	rHidden, rDispatches := run(perRow, true)

	if bDispatches >= rDispatches {
		t.Fatalf("batched epilogue did not engage: batched pass encoded %d dispatches, per-row %d — the rows epilogue must strictly reduce dispatch count", bDispatches, rDispatches)
	}
	if len(bHidden) != len(rHidden) {
		t.Fatalf("hidden sizes differ: batched=%d perRow=%d", len(bHidden), len(rHidden))
	}
	for i := range bHidden {
		if bHidden[i] != rHidden[i] {
			t.Fatalf("boundary hidden diverges at byte %d: batched=%02x perRow=%02x (the batched epilogue contract is byte-identity with the per-row chain)", i, bHidden[i], rHidden[i])
		}
	}
	va, err := perRow.stateLayerViews()
	if err != nil {
		t.Fatalf("per-row views: %v", err)
	}
	vb, err := batched.stateLayerViews()
	if err != nil {
		t.Fatalf("batched views: %v", err)
	}
	for i := range va {
		for j := range va[i].keyBytes {
			if va[i].keyBytes[j] != vb[i].keyBytes[j] {
				t.Fatalf("layer %d K diverges at byte %d", i, j)
			}
		}
		for j := range va[i].valueBytes {
			if va[i].valueBytes[j] != vb[i].valueBytes[j] {
				t.Fatalf("layer %d V diverges at byte %d", i, j)
			}
		}
	}
}

func batchedPLEParity(t *testing.T, control, candidate *ArchSession) {
	t.Helper()
	ids := []int32{3, 9, 17, 24, 6, 11, 29, 2}

	// control: the sequential per-token lane (previously the only bf16 PLE path).
	ctrlHidden, err := control.prefillPromptRetainedInPool(ids)
	if err != nil {
		t.Fatalf("sequential control prefill: %v", err)
	}

	// candidate: the batched dense lane on an identical fresh session.
	hidden, ok, err := candidate.prefillRetainedTokensBatchedDense(ids, "test.batchedPLE")
	if err != nil {
		t.Fatalf("batched dense prefill: %v", err)
	}
	if !ok {
		t.Fatal("batched dense prefill DECLINED the arch — bf16 E2B/E4B prompts have no batched lane (#252: per-token fallback is O(n^2))")
	}
	if candidate.Pos() != control.Pos() {
		t.Fatalf("pos after prefill: batched=%d sequential=%d", candidate.Pos(), control.Pos())
	}
	if len(hidden) != len(ctrlHidden) {
		t.Fatalf("hidden sizes differ: batched=%d sequential=%d", len(hidden), len(ctrlHidden))
	}
	for i := range hidden {
		if hidden[i] != ctrlHidden[i] {
			t.Fatalf("boundary hidden diverges at byte %d: batched=%02x sequential=%02x (batched PLE contract is byte-identity with stepToken)", i, hidden[i], ctrlHidden[i])
		}
	}

	// the caches must match too — every layer, both slabs.
	va, err := control.stateLayerViews()
	if err != nil {
		t.Fatalf("control views: %v", err)
	}
	vb, err := candidate.stateLayerViews()
	if err != nil {
		t.Fatalf("candidate views: %v", err)
	}
	if len(va) != len(vb) {
		t.Fatalf("view counts differ: sequential=%d batched=%d", len(va), len(vb))
	}
	for i := range va {
		for j := range va[i].keyBytes {
			if va[i].keyBytes[j] != vb[i].keyBytes[j] {
				t.Fatalf("layer %d K diverges at byte %d", i, j)
			}
		}
		for j := range va[i].valueBytes {
			if va[i].valueBytes[j] != vb[i].valueBytes[j] {
				t.Fatalf("layer %d V diverges at byte %d", i, j)
			}
		}
	}
}

// TestPrefillRetainedTokensBatchedDenseEngagesQuantPLEAndMatchesSequential is the quant twin of
// TestPrefillRetainedTokensBatchedDenseEngagesPLEAndMatchesSequential (#252/#381): the batched
// dense prefill must ENGAGE on a 4-bit quant PLE (E2B/E4B-shaped) arch, at a batch width
// (quantBatchedPLEIDs, steelGEMMMinRows+8) that actually drives perLayerInputsBatchQuantIntoSlab/
// -Device instead of declining below the steel floor, and must stay CLOSE to genuinely sequential
// per-token stepping. The control walks stepIDInPool/stepIDRetainedInPool directly rather than
// reusing batchedPLEParity's prefillPromptRetainedInPool helper: once state.icb is forced nil
// (this fixture, like newBatchedPLEBF16FixtureShared, forces it so the batched lane isn't
// declined by the short-append ICB guard), prefillPromptRetainedInPool's own pos==0 branch tries
// the batched-dense lane FIRST on both sides, which would make a "control vs candidate"
// comparison self-referential. stepIDInPool/stepIDRetainedInPool skip that branch entirely and
// always call the original per-token stepToken/stepTokenInto.
//
// Unlike the bf16 sibling this is a TOLERANCE, not byte-identity, check — measured, not assumed:
// forcing candidate.perLayerInputBatch/-Device off (so BOTH sides build the PLE slab through the
// identical per-token PerLayerInputs loop) reproduces the exact same divergence, and an equivalent
// non-PLE quant fixture (no per-layer-input tower at all) diverges by the same bounded amount —
// so the source is the general quant batched-dense fold's qmm_t/qmv accumulation-order difference
// (the same class of variance per_layer_batch_slab_test.go's own steel-vs-per-token tests already
// tolerate at 0.05-0.08), not anything specific to the PLE tower or a defect this fixture
// introduced. No prior test drove the quant batched-dense fold at a steel-engaged width
// (K>=steelGEMMMinRows) against sequential at all — the closest existing coverage
// (arch_session_test.go's quant + prefillRetainedTokensBatchedDense tests) stays at K<=5.
func TestPrefillRetainedTokensBatchedDenseEngagesQuantPLEAndMatchesSequential(t *testing.T) {
	requireNativeRuntime(t)
	ids := quantBatchedPLEIDs()
	control := newBatchedPLEQuantFixture(t)
	candidate := newBatchedPLEQuantFixture(t)
	if candidate.arch.PerLayerInputHidden == 0 {
		t.Fatal("fixture lost its per-layer-input tower — this parity no longer exercises quant PLE")
	}
	if candidate.perLayerInputBatch == nil || candidate.perLayerInputBatchDevice == nil {
		t.Fatal("fixture lost its quant PLE batch closures — arch_session's #381 plain-quant gate no longer wires them")
	}

	var ctrlHidden []byte
	var err error
	for _, id := range ids[:len(ids)-1] {
		if _, err = control.stepIDInPool(id); err != nil {
			t.Fatalf("control step: %v", err)
		}
	}
	if ctrlHidden, err = control.stepIDRetainedInPool(ids[len(ids)-1]); err != nil {
		t.Fatalf("control last step: %v", err)
	}

	hidden, ok, err := candidate.prefillRetainedTokensBatchedDense(ids, "test.batchedQuantPLE")
	if err != nil {
		t.Fatalf("batched dense prefill: %v", err)
	}
	if !ok {
		t.Fatal("batched dense prefill DECLINED the quant PLE arch")
	}
	if candidate.Pos() != control.Pos() {
		t.Fatalf("pos after prefill: batched=%d sequential=%d", candidate.Pos(), control.Pos())
	}
	if len(hidden) != len(ctrlHidden) {
		t.Fatalf("hidden sizes differ: batched=%d sequential=%d", len(hidden), len(ctrlHidden))
	}
	// 0.25 gives 2x margin over the measured maxDiff (0.125, bf16-adjacent-bucket-sized, absolute
	// not relative — values up to magnitude ~12 diverge by the same ~0.03-0.125 the near-zero ones
	// do) — tight enough to catch a real regression, loose enough not to flake on kernel updates.
	const hiddenTol = 0.25
	if maxDiff := pleSlabMaxDiff(hidden, ctrlHidden); maxDiff > hiddenTol {
		t.Fatalf("boundary hidden diverges from sequential: maxDiff=%.5f (> %.2f)", maxDiff, hiddenTol)
	}

	// the caches must stay close too — every layer, both slabs (same tolerance class; measured
	// maxDiff sits under 0.04 for both K and V, comfortably inside the hidden-state bound).
	va, err := control.stateLayerViews()
	if err != nil {
		t.Fatalf("control views: %v", err)
	}
	vb, err := candidate.stateLayerViews()
	if err != nil {
		t.Fatalf("candidate views: %v", err)
	}
	if len(va) != len(vb) {
		t.Fatalf("view counts differ: sequential=%d batched=%d", len(va), len(vb))
	}
	for i := range va {
		if kd := pleSlabMaxDiff(va[i].keyBytes, vb[i].keyBytes); kd > hiddenTol {
			t.Fatalf("layer %d K diverges from sequential: maxDiff=%.5f (> %.2f)", i, kd, hiddenTol)
		}
		if vd := pleSlabMaxDiff(va[i].valueBytes, vb[i].valueBytes); vd > hiddenTol {
			t.Fatalf("layer %d V diverges from sequential: maxDiff=%.5f (> %.2f)", i, vd, hiddenTol)
		}
	}
}

// TestVerifyBatchedEngagesQuantPLE exercises the MTP-verify batched lane (verifyBatchedHiddens /
// verifyBatchedInto) on a quant PLE session directly. In production these run only inside a real
// AssistantPair's verify step, but they are plain *ArchSession methods and the PLE-suffixed
// switch arms they select (stepTokensBatchedDenseIntoPLE / -NoResultPLE / -PLE, decode_batched_
// session.go) key on the same s.state.ple tower this fixture carries, so driving them directly is
// a legitimate — not shimmed — way to reach the quant PLE arms of the verify lane. Output is
// checked for real invariants (row/token counts, vocab range, no position mutation) rather than
// pinned bytes: there is no independent oracle for MTP-verify greedy output at this level.
func TestVerifyBatchedEngagesQuantPLE(t *testing.T) {
	requireNativeRuntime(t)
	ids := quantBatchedPLEIDs()

	hiddenSess := newBatchedPLEQuantFixture(t)
	hiddens, ok, err := hiddenSess.verifyBatchedHiddens(ids)
	if err != nil {
		t.Fatalf("verifyBatchedHiddens: %v", err)
	}
	if !ok {
		t.Fatal("verifyBatchedHiddens DECLINED the quant PLE arch")
	}
	if len(hiddens) != len(ids) {
		t.Fatalf("verifyBatchedHiddens returned %d hidden rows, want %d", len(hiddens), len(ids))
	}
	rowBytes := hiddenSess.arch.Hidden * bf16Size
	for i, h := range hiddens {
		if len(h) != rowBytes {
			t.Fatalf("hidden row %d size = %d, want %d", i, len(h), rowBytes)
		}
	}
	if hiddenSess.Pos() != 0 {
		t.Fatalf("verifyBatchedHiddens advanced pos to %d, want 0 (verify must not commit)", hiddenSess.Pos())
	}

	greedySess := newBatchedPLEQuantFixture(t)
	greedys, ok, err := greedySess.verifyBatchedInto(ids, nil)
	if err != nil {
		t.Fatalf("verifyBatchedInto: %v", err)
	}
	if !ok {
		t.Fatal("verifyBatchedInto DECLINED the quant PLE arch")
	}
	if len(greedys) != len(ids) {
		t.Fatalf("verifyBatchedInto returned %d tokens, want %d", len(greedys), len(ids))
	}
	for i, tok := range greedys {
		if tok < 0 || int(tok) >= greedySess.arch.Vocab {
			t.Fatalf("greedy token[%d] = %d outside vocab %d", i, tok, greedySess.arch.Vocab)
		}
	}
	if greedySess.Pos() != 0 {
		t.Fatalf("verifyBatchedInto advanced pos to %d, want 0 (verify must not commit)", greedySess.Pos())
	}
}

// TestPrefillInputsDeviceEmbedDeviceKillSwitch pins the #381 dense/MoE port's
// kill switch (prefillEmbedDeviceOffForTest, the in-process twin of
// LTHN_PREFILL_EMBED_DEVICE=0 — same shape as flash_prompt.go's
// prefillSkipSharedOffForTest): a fixture whose perLayerInputBatchDevice
// closure is wired and would otherwise engage must still fall back to the nil
// (host-path) result the instant the switch is off, and re-engage the moment
// it's back on.
func TestPrefillInputsDeviceEmbedDeviceKillSwitch(t *testing.T) {
	requireNativeRuntime(t)
	sess := newBatchedPLEQuantFixture(t)
	if sess.perLayerInputBatchDevice == nil {
		t.Fatal("fixture lost its quant PLE device closure — arch_session's #381 gate no longer wires it")
	}
	ids := quantBatchedPLEIDs()

	embBuf, pleBuf, err := sess.prefillInputsDevice(ids)
	if err != nil {
		t.Fatalf("prefillInputsDevice (switch on): %v", err)
	}
	if embBuf == nil || pleBuf == nil {
		t.Fatal("prefillInputsDevice (switch on) declined a fixture that should engage")
	}

	prefillEmbedDeviceOffForTest = true
	defer func() { prefillEmbedDeviceOffForTest = false }()
	embBufOff, pleBufOff, err := sess.prefillInputsDevice(ids)
	if err != nil {
		t.Fatalf("prefillInputsDevice (switch off): %v", err)
	}
	if embBufOff != nil || pleBufOff != nil {
		t.Fatal("prefillInputsDevice (switch off) engaged the device path — kill switch had no effect")
	}

	prefillEmbedDeviceOffForTest = false
	embBufBack, pleBufBack, err := sess.prefillInputsDevice(ids)
	if err != nil {
		t.Fatalf("prefillInputsDevice (switch back on): %v", err)
	}
	if embBufBack == nil || pleBufBack == nil {
		t.Fatal("prefillInputsDevice (switch back on) stayed declined")
	}
}

// TestPrefillRetainedTokensBatchedDenseChunksEngagesQuantPLEBoundedSlab is the quant-PLE
// analogue of arch_session_test.go's TestArchSessionPrefillChunksSkipSharedSuffix (#30 r4): a
// 4-bit gemma4 arch WITH the per-layer-input tower, a clean 2-of-4 trailing KV-shared suffix, and
// a sliding window small enough that a steel-floor-sized prompt crosses the ring wrap into TWO
// chunks. The non-final (steelGEMMMinRows-sized) chunk arms state.prefillSkipToLayer =
// sharedSuffix, which prefillInputsDevice threads into perLayerInputBatchDevice's outLayers bound
// — the ONLY route (#381) that drives perLayerInputsBatchQuantEncode with nOut < numLayers on a
// real session (per_layer_batch.go ~464-472: the "plain" quant-PLE profile this fixture builds
// has projPacked == nil, so the bounded arm engages unconditionally once nOut < numLayers).
// TestPrefillRetainedTokensBatchedDenseEngagesQuantPLEAndMatchesSequential (r3, above) runs a
// single 40-row chunk and never sees this branch: nOut == numLayers there because the whole
// prompt lands in one unbounded chunk. LTHN_PREFILL_WINDOWS=1 keeps the "wide" chunk unit at
// exactly one window (sized to steelGEMMMinRows) so the fixture stays at 64 tokens rather than
// the 256+ the production default (up to 8 windows) would need to observe the same split.
func TestPrefillRetainedTokensBatchedDenseChunksEngagesQuantPLEBoundedSlab(t *testing.T) {
	requireNativeRuntime(t)
	t.Setenv("LTHN_PREFILL_WINDOWS", "1")
	const window = steelGEMMMinRows // 32: one window == one steel-floor chunk
	const kvShared = 2
	control := newBatchedPLEQuantFixtureShared(t, kvShared, window)
	candidate := newBatchedPLEQuantFixtureShared(t, kvShared, window)
	if candidate.arch.PerLayerInputHidden == 0 {
		t.Fatal("fixture lost its per-layer-input tower — this parity no longer exercises quant PLE")
	}
	if candidate.perLayerInputBatchDevice == nil {
		t.Fatal("fixture lost its quant PLE device closure — arch_session's #381 plain-quant gate no longer wires it")
	}
	if got := candidate.state.sharedSuffix; got != kvShared {
		t.Fatalf("sharedSuffix = %d, want %d (fixture must have a clean trailing KV-shared suffix)", got, kvShared)
	}

	ids := quantBatchedPLESharedIDs(2 * window) // 64: exactly two steel-floor chunks
	if !candidate.verifyBatchedCrossesSlidingRingWrap(len(ids)) {
		t.Fatal("fixture prompt must cross the sliding ring wrap — the chunked skip-suffix lane never engages otherwise")
	}

	// control: genuinely sequential per-token stepping (oblivious to prefillSkipToLayer, which
	// only the batched chunked lane reads) — the r3 pattern, avoided prefillPromptRetainedInPool
	// because once state.icb is nil its own pos==0 branch tries the batched-dense lane first too.
	var ctrlHidden []byte
	var err error
	for _, id := range ids[:len(ids)-1] {
		if _, err = control.stepIDInPool(id); err != nil {
			t.Fatalf("control step: %v", err)
		}
	}
	if ctrlHidden, err = control.stepIDRetainedInPool(ids[len(ids)-1]); err != nil {
		t.Fatalf("control last step: %v", err)
	}

	hidden, ok, err := candidate.prefillRetainedTokensBatchedDense(ids, "test.batchedQuantPLEBoundedSlab")
	if err != nil {
		t.Fatalf("prefillRetainedTokensBatchedDense: %v", err)
	}
	if !ok {
		t.Fatal("prefillRetainedTokensBatchedDense DECLINED the quant PLE shared-suffix fixture")
	}
	// unlike the causal skip lane's own test (TestArchSessionPrefillChunksSkipSharedSuffix), which
	// asserts this on the ArchSession's exported state, the PLE builders read the SAME field — the
	// leaked-flag hazard is identical, so pin it here too.
	if candidate.state.prefillSkipToLayer != 0 {
		t.Fatalf("prefillSkipToLayer leaked = %d, want 0 after prefill", candidate.state.prefillSkipToLayer)
	}
	if candidate.Pos() != control.Pos() {
		t.Fatalf("pos after prefill: batched=%d sequential=%d", candidate.Pos(), control.Pos())
	}
	if len(hidden) != len(ctrlHidden) {
		t.Fatalf("hidden sizes differ: batched=%d sequential=%d", len(hidden), len(ctrlHidden))
	}
	// same tolerance class as TestPrefillRetainedTokensBatchedDenseEngagesQuantPLEAndMatchesSequential
	// (measured, not assumed: the general quant batched-dense fold's qmm_t/qmv accumulation-order
	// difference, not anything specific to the bounded-slab lane this test adds).
	const hiddenTol = 0.25
	if maxDiff := pleSlabMaxDiff(hidden, ctrlHidden); maxDiff > hiddenTol {
		t.Fatalf("boundary hidden diverges from sequential: maxDiff=%.5f (> %.2f)", maxDiff, hiddenTol)
	}
}
