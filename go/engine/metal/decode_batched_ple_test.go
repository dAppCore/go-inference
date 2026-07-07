// SPDX-Licence-Identifier: EUPL-1.2

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
	return sess
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
