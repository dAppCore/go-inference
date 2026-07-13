// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"bytes"
	"testing"
	"unsafe"

	core "dappco.re/go"
	"dappco.re/go/inference/model"
)

// TestStepTokensBatchedDense asserts the session-level MTP batched verify (K tokens through the whole
// resident layer stack in one pass) is BYTE-IDENTICAL to stepping the same K tokens one at a time with
// stepToken over the same growing cache. This is the bar that lets MTPDecode swap its sequential
// stepGreedy verify for one batched pass without changing the emitted token stream.
func TestStepTokensBatchedDense(t *testing.T) {
	requireNativeRuntime(t)
	const dModel, nHeads, nKV, headDim, dFF = 512, 8, 4, 64, 1024
	const base, scale, eps = float32(10000), float32(0.125), float32(1e-5)
	const nL, maxLen, prefix, K = 3, 32, 5, 4

	layers := make([]DecodeLayerWeights, nL)
	types := make([]string, nL)
	for li := range layers {
		layers[li] = forwardLayer(dModel, nHeads, nKV, headDim, dFF, (li+1)*100)
		types[li] = "full_attention"
	}
	specs := model.DeriveLayers(types, 0)

	emb := func(seed int) []byte {
		f := make([]float32, dModel)
		for j := range f {
			f[j] = float32((j*(seed+3)+5)%97-48) * 0.02
		}
		return toBF16Bytes(f)
	}
	embs := make([][]byte, prefix+K)
	for i := range embs {
		embs[i] = emb(i + 1)
	}

	build := func() *archDecodeState {
		lb, moe, err := buildBF16ArchLayerBufs(layers, specs, dModel, nHeads, nKV, headDim, dFF, maxLen, 0, nil)
		if err != nil {
			t.Fatalf("buildBF16ArchLayerBufs: %v", err)
		}
		st := newArchDecodeState(specs, lb, moe, dModel, nHeads, nKV, headDim, dFF, 0, headDim, headDim, base, base, scale, eps, false, 0)
		return &st
	}

	// sequential reference: prefill the prefix, then step K tokens one at a time.
	var seqOut [][]byte
	withAutoreleasePool(func() {
		st := build()
		for i := range prefix {
			if _, err := st.stepToken(embs[i], i); err != nil {
				t.Fatalf("prefill stepToken %d: %v", i, err)
			}
		}
		for i := range K {
			h, err := st.stepToken(embs[prefix+i], prefix+i)
			if err != nil {
				t.Fatalf("seq stepToken %d: %v", prefix+i, err)
			}
			seqOut = append(seqOut, append([]byte(nil), h...))
		}
	})

	// batched: fresh state, same prefix, then ONE stepTokensBatchedDense over the K tokens.
	var batOut [][]byte
	var ok bool
	withAutoreleasePool(func() {
		st := build()
		for i := range prefix {
			if _, err := st.stepToken(embs[i], i); err != nil {
				t.Fatalf("batched prefill stepToken %d: %v", i, err)
			}
		}
		var err error
		batOut, ok, err = st.stepTokensBatchedDense(embs[prefix:prefix+K], prefix)
		if err != nil {
			t.Fatalf("stepTokensBatchedDense: %v", err)
		}
	})
	if !ok {
		t.Fatal("stepTokensBatchedDense reported !ok for a dense full-attention arch")
	}
	if len(batOut) != K {
		t.Fatalf("batched returned %d rows, want %d", len(batOut), K)
	}
	for i := range K {
		eqBytes(t, core.Sprintf("batched session row %d vs stepToken", i), batOut[i], seqOut[i])
	}
}

func TestStepTokensBatchedDenseUsesPinnedInputRows(t *testing.T) {
	requireNativeRuntime(t)
	const dModel, nHeads, nKV, headDim, dFF = 64, 1, 1, 64, 128
	const base, scale, eps = float32(10000), float32(0.125), float32(1e-5)
	const nL, maxLen, K = 1, 8, 2

	layers := make([]DecodeLayerWeights, nL)
	types := make([]string, nL)
	for li := range layers {
		layers[li] = forwardLayer(dModel, nHeads, nKV, headDim, dFF, (li+1)*100)
		types[li] = "full_attention"
	}
	specs := model.DeriveLayers(types, 0)
	lb, moe, err := buildBF16ArchLayerBufs(layers, specs, dModel, nHeads, nKV, headDim, dFF, maxLen, 0, nil)
	if err != nil {
		t.Fatalf("buildBF16ArchLayerBufs: %v", err)
	}
	st := newArchDecodeState(specs, lb, moe, dModel, nHeads, nKV, headDim, dFF, 0, headDim, headDim, base, base, scale, eps, false, 0)
	defer st.Close()

	embs := make([][]byte, K)
	for i := range embs {
		pinned, err := newPinnedNoCopyBytes(dModel * bf16Size)
		if err != nil {
			t.Fatalf("newPinnedNoCopyBytes(%d): %v", i, err)
		}
		defer pinned.Close()
		copy(pinned.bytes, toBF16Bytes(syntheticFloat32(dModel, i+1)))
		embs[i] = pinned.bytes
	}

	withAutoreleasePool(func() {
		st.denseBatch.rows(K, dModel)
	})
	inPacked := unsafe.Slice((*byte)(st.denseBatch.inPacked.Contents()), K*dModel*bf16Size)
	sentinel := bytes.Repeat([]byte{0x4d}, len(inPacked))
	copy(inPacked, sentinel)

	var ok bool
	withAutoreleasePool(func() {
		ok, err = st.stepTokensBatchedDenseNoResult(embs, 0)
	})
	if err != nil {
		t.Fatalf("stepTokensBatchedDenseNoResult: %v", err)
	}
	if !ok {
		t.Fatal("stepTokensBatchedDenseNoResult reported !ok for a dense full-attention arch")
	}
	if !bytes.Equal(inPacked, sentinel) {
		t.Fatal("stepTokensBatchedDense copied pinned embeddings into packed input scratch")
	}
}

func TestStepTokensBatchedDenseIntoWritesPinnedOutputRowsDirectly(t *testing.T) {
	requireNativeRuntime(t)
	const dModel, nHeads, nKV, headDim, dFF = 64, 1, 1, 64, 128
	const base, scale, eps = float32(10000), float32(0.125), float32(1e-5)
	const nL, maxLen, K = 1, 8, 2

	layers := make([]DecodeLayerWeights, nL)
	types := make([]string, nL)
	for li := range layers {
		layers[li] = forwardLayer(dModel, nHeads, nKV, headDim, dFF, (li+1)*100)
		types[li] = "full_attention"
	}
	specs := model.DeriveLayers(types, 0)
	lb, moe, err := buildBF16ArchLayerBufs(layers, specs, dModel, nHeads, nKV, headDim, dFF, maxLen, 0, nil)
	if err != nil {
		t.Fatalf("buildBF16ArchLayerBufs: %v", err)
	}
	st := newArchDecodeState(specs, lb, moe, dModel, nHeads, nKV, headDim, dFF, 0, headDim, headDim, base, base, scale, eps, false, 0)
	defer st.Close()

	embs := make([][]byte, K)
	dstRows := make([][]byte, K)
	pinned := make([]*pinnedNoCopyBytes, K)
	for i := range embs {
		emb := toBF16Bytes(syntheticFloat32(dModel, i+1))
		embs[i] = append([]byte(nil), emb...)
		pinned[i], err = newPinnedNoCopyBytes(dModel * bf16Size)
		if err != nil {
			t.Fatalf("newPinnedNoCopyBytes(%d): %v", i, err)
		}
		defer pinned[i].Close()
		dstRows[i] = pinned[i].bytes
	}

	withAutoreleasePool(func() {
		st.denseBatch.rows(K, dModel)
	})
	outPacked := unsafe.Slice((*byte)(st.denseBatch.outPacked.Contents()), K*dModel*bf16Size)
	sentinel := bytes.Repeat([]byte{0x6b}, len(outPacked))
	copy(outPacked, sentinel)

	var out [][]byte
	var ok bool
	withAutoreleasePool(func() {
		out, ok, err = st.stepTokensBatchedDenseInto(embs, 0, dstRows)
	})
	if err != nil {
		t.Fatalf("stepTokensBatchedDenseInto: %v", err)
	}
	if !ok {
		t.Fatal("stepTokensBatchedDenseInto reported !ok for a dense full-attention arch")
	}
	if len(out) != K {
		t.Fatalf("stepTokensBatchedDenseInto returned %d rows, want %d", len(out), K)
	}
	for i := range out {
		if len(out[i]) != dModel*bf16Size || unsafe.Pointer(&out[i][0]) != unsafe.Pointer(&dstRows[i][0]) {
			t.Fatalf("output row %d does not reuse caller pinned backing", i)
		}
	}
	if !bytes.Equal(outPacked, sentinel) {
		t.Fatal("stepTokensBatchedDenseInto wrote final rows through packed output scratch")
	}
	if st.denseBatch.lastRows == nil || st.denseBatch.lastRows.GetID() != pinned[0].buf.GetID() {
		t.Fatal("stepTokensBatchedDenseInto did not record pinned output rows as final rows")
	}
}

func TestStepTokensBatchedDenseSyncsLinearCacheAfterPagedStep(t *testing.T) {
	requireNativeRuntime(t)
	const dModel, nHeads, nKV, headDim, dFF = 512, 8, 4, 64, 1024
	const base, scale, eps = float32(10000), float32(0.125), float32(1e-5)
	const nL, maxLen, prefix, K = 3, 32, 6, 4

	layers := make([]DecodeLayerWeights, nL)
	types := make([]string, nL)
	for li := range layers {
		layers[li] = forwardLayer(dModel, nHeads, nKV, headDim, dFF, (li+1)*200)
		types[li] = "full_attention"
	}
	specs := model.DeriveLayers(types, 0)

	emb := func(seed int) []byte {
		f := make([]float32, dModel)
		for j := range f {
			f[j] = float32((j*(seed+7)+11)%89-44) * 0.025
		}
		return toBF16Bytes(f)
	}
	embs := make([][]byte, prefix+1+K)
	for i := range embs {
		embs[i] = emb(i + 1)
	}

	build := func() *archDecodeState {
		lb, moe, err := buildBF16ArchLayerBufs(layers, specs, dModel, nHeads, nKV, headDim, dFF, maxLen, 0, nil)
		if err != nil {
			t.Fatalf("buildBF16ArchLayerBufs: %v", err)
		}
		st := newArchDecodeState(specs, lb, moe, dModel, nHeads, nKV, headDim, dFF, 0, headDim, headDim, base, base, scale, eps, false, 0)
		if err := st.initDevicePagedKV(2); err != nil {
			t.Fatalf("initDevicePagedKV: %v", err)
		}
		return &st
	}

	var seqOut [][]byte
	withAutoreleasePool(func() {
		st := build()
		for i := range prefix + 1 {
			if _, err := st.stepToken(embs[i], i); err != nil {
				t.Fatalf("seq prefix stepToken %d: %v", i, err)
			}
		}
		for i := range K {
			pos := prefix + 1 + i
			h, err := st.stepToken(embs[pos], pos)
			if err != nil {
				t.Fatalf("seq stepToken %d: %v", pos, err)
			}
			seqOut = append(seqOut, append([]byte(nil), h...))
		}
	})

	var batOut [][]byte
	var ok bool
	withAutoreleasePool(func() {
		st := build()
		var err error
		ok, err = st.stepTokensBatchedDenseNoResult(embs[:prefix], 0)
		if err != nil {
			t.Fatalf("dense prefix: %v", err)
		}
		if !ok {
			t.Fatal("dense prefix reported !ok")
		}
		if _, err := st.stepToken(embs[prefix], prefix); err != nil {
			t.Fatalf("paged bonus stepToken: %v", err)
		}
		batOut, ok, err = st.stepTokensBatchedDense(embs[prefix+1:prefix+1+K], prefix+1)
		if err != nil {
			t.Fatalf("stepTokensBatchedDense after paged step: %v", err)
		}
	})
	if !ok {
		t.Fatal("stepTokensBatchedDense after paged step reported !ok")
	}
	for i := range K {
		eqBytes(t, core.Sprintf("batched after paged row %d vs stepToken", i), batOut[i], seqOut[i])
	}
}

// TestStepTokensBatchedDensePLE_Good pins the PLE wrapper itself, not just the
// session prefill route that normally reaches it. The sequential PLE step is
// the oracle: every batch row and every owner cache byte must match because
// the wrapper promises the same per-row gate and cache mutation ordering.
func TestStepTokensBatchedDensePLE_Good(t *testing.T) {
	requireNativeRuntime(t)
	ids := []int32{3, 9, 17, 24}
	control := newBatchedPLEBF16Fixture(t)
	candidate := newBatchedPLEBF16Fixture(t)

	want := make([][]byte, len(ids))
	for i, id := range ids {
		hidden, err := control.stepID(id)
		if err != nil {
			t.Fatalf("control stepID(%d): %v", id, err)
		}
		want[i] = append([]byte(nil), hidden...)
	}

	rowBytes := candidate.arch.Hidden * bf16Size
	embs := make([][]byte, len(ids))
	pinned := make([]*pinnedNoCopyBytes, len(ids))
	for i, id := range ids {
		emb, err := candidate.embed(id)
		if err != nil {
			t.Fatalf("candidate embed(%d): %v", id, err)
		}
		pinned[i], err = newPinnedNoCopyBytes(rowBytes)
		if err != nil {
			t.Fatalf("newPinnedNoCopyBytes(input %d): %v", i, err)
		}
		defer pinned[i].Close()
		copy(pinned[i].bytes, emb)
		embs[i] = pinned[i].bytes
	}
	pleSlab, err := candidate.pleSlabFor(ids, embs)
	if err != nil {
		t.Fatalf("pleSlabFor: %v", err)
	}

	var got [][]byte
	var ok bool
	withAutoreleasePool(func() {
		got, ok, err = candidate.state.stepTokensBatchedDensePLE(embs, pleSlab, candidate.Pos())
	})
	if err != nil {
		t.Fatalf("stepTokensBatchedDensePLE: %v", err)
	}
	if !ok {
		t.Fatal("stepTokensBatchedDensePLE declined a complete bf16 PLE slab")
	}
	if candidate.Pos() != 0 {
		t.Fatalf("direct state batch advanced session pos to %d, want 0", candidate.Pos())
	}
	if len(got) != len(want) {
		t.Fatalf("stepTokensBatchedDensePLE returned %d rows, want %d", len(got), len(want))
	}
	for i := range got {
		eqBytes(t, core.Sprintf("PLE batch row %d vs sequential", i), got[i], want[i])
		// The implementation binds a registered pinned embedding directly. This
		// checks the binding identity, rather than inferring it from the output.
		if candidate.state.denseBatch.inputViewStack[i].buf == nil || candidate.state.denseBatch.inputViewStack[i].buf.GetID() != pinned[i].buf.GetID() {
			t.Fatalf("PLE input row %d was not bound to its registered pinned buffer", i)
		}
	}
	controlViews, err := control.stateLayerViews()
	if err != nil {
		t.Fatalf("control stateLayerViews: %v", err)
	}
	candidateViews, err := candidate.stateLayerViews()
	if err != nil {
		t.Fatalf("candidate stateLayerViews: %v", err)
	}
	if len(candidateViews) != len(controlViews) {
		t.Fatalf("owner cache view count = %d, want %d", len(candidateViews), len(controlViews))
	}
	for li := range controlViews {
		eqBytes(t, core.Sprintf("PLE layer %d K cache", li), candidateViews[li].keyBytes, controlViews[li].keyBytes)
		eqBytes(t, core.Sprintf("PLE layer %d V cache", li), candidateViews[li].valueBytes, controlViews[li].valueBytes)
	}
}

// TestStepTokensBatchedDenseIntoPLE_Good pins the final-layer direct-output
// binding on the PLE path. The returned rows must alias the caller's pinned
// buffers, while their bytes still equal independent sequential PLE steps.
func TestStepTokensBatchedDenseIntoPLE_Good(t *testing.T) {
	requireNativeRuntime(t)
	ids := []int32{6, 11, 29}
	control := newBatchedPLEBF16Fixture(t)
	candidate := newBatchedPLEBF16Fixture(t)

	want := make([][]byte, len(ids))
	for i, id := range ids {
		hidden, err := control.stepID(id)
		if err != nil {
			t.Fatalf("control stepID(%d): %v", id, err)
		}
		want[i] = append([]byte(nil), hidden...)
	}

	rowBytes := candidate.arch.Hidden * bf16Size
	embs := make([][]byte, len(ids))
	for i, id := range ids {
		emb, err := candidate.embed(id)
		if err != nil {
			t.Fatalf("candidate embed(%d): %v", id, err)
		}
		embs[i] = append([]byte(nil), emb...)
	}
	pleSlab, err := candidate.pleSlabFor(ids, embs)
	if err != nil {
		t.Fatalf("pleSlabFor: %v", err)
	}
	dstRows := make([][]byte, len(ids))
	pinned := make([]*pinnedNoCopyBytes, len(ids))
	for i := range dstRows {
		pinned[i], err = newPinnedNoCopyBytes(rowBytes)
		if err != nil {
			t.Fatalf("newPinnedNoCopyBytes(output %d): %v", i, err)
		}
		defer pinned[i].Close()
		dstRows[i] = pinned[i].bytes
	}

	var got [][]byte
	var ok bool
	withAutoreleasePool(func() {
		got, ok, err = candidate.state.stepTokensBatchedDenseIntoPLE(embs, pleSlab, candidate.Pos(), dstRows)
	})
	if err != nil {
		t.Fatalf("stepTokensBatchedDenseIntoPLE: %v", err)
	}
	if !ok {
		t.Fatal("stepTokensBatchedDenseIntoPLE declined a complete bf16 PLE slab")
	}
	if len(got) != len(ids) {
		t.Fatalf("stepTokensBatchedDenseIntoPLE returned %d rows, want %d", len(got), len(ids))
	}
	for i := range got {
		if unsafe.Pointer(&got[i][0]) != unsafe.Pointer(&dstRows[i][0]) {
			t.Fatalf("PLE output row %d did not reuse the caller's pinned backing", i)
		}
		if candidate.state.denseBatch.directOutRows[i] == nil || candidate.state.denseBatch.directOutRows[i].GetID() != pinned[i].buf.GetID() {
			t.Fatalf("PLE output row %d was not bound to its registered pinned buffer", i)
		}
		eqBytes(t, core.Sprintf("PLE direct output row %d vs sequential", i), got[i], want[i])
	}
}

// TestStepTokensBatchedDenseNoResultPLE_Good pins the no-readback PLE route:
// it still has to mutate exactly the same owner caches and publish K final rows
// for the direct-head caller, despite deliberately returning no hidden slices.
func TestStepTokensBatchedDenseNoResultPLE_Good(t *testing.T) {
	requireNativeRuntime(t)
	ids := []int32{2, 21, 14}
	control := newBatchedPLEBF16Fixture(t)
	candidate := newBatchedPLEBF16Fixture(t)

	for _, id := range ids {
		if _, err := control.stepID(id); err != nil {
			t.Fatalf("control stepID(%d): %v", id, err)
		}
	}
	embs := make([][]byte, len(ids))
	for i, id := range ids {
		emb, err := candidate.embed(id)
		if err != nil {
			t.Fatalf("candidate embed(%d): %v", id, err)
		}
		embs[i] = emb
	}
	pleSlab, err := candidate.pleSlabFor(ids, embs)
	if err != nil {
		t.Fatalf("pleSlabFor: %v", err)
	}

	var ok bool
	withAutoreleasePool(func() {
		ok, err = candidate.state.stepTokensBatchedDenseNoResultPLE(embs, pleSlab, candidate.Pos())
	})
	if err != nil {
		t.Fatalf("stepTokensBatchedDenseNoResultPLE: %v", err)
	}
	if !ok {
		t.Fatal("stepTokensBatchedDenseNoResultPLE declined a complete bf16 PLE slab")
	}
	if candidate.state.denseBatch.lastK != len(ids) || candidate.state.denseBatch.lastRows == nil {
		t.Fatalf("no-result PLE batch did not publish %d final rows (lastK=%d rows=%v)", len(ids), candidate.state.denseBatch.lastK, candidate.state.denseBatch.lastRows != nil)
	}
	controlViews, err := control.stateLayerViews()
	if err != nil {
		t.Fatalf("control stateLayerViews: %v", err)
	}
	candidateViews, err := candidate.stateLayerViews()
	if err != nil {
		t.Fatalf("candidate stateLayerViews: %v", err)
	}
	for li := range controlViews {
		eqBytes(t, core.Sprintf("no-result PLE layer %d K cache", li), candidateViews[li].keyBytes, controlViews[li].keyBytes)
		eqBytes(t, core.Sprintf("no-result PLE layer %d V cache", li), candidateViews[li].valueBytes, controlViews[li].valueBytes)
	}
}

// TestStepTokensBatchedDensePLE_Bad derives its failure from the layer-major
// contract: a one-row, one-layer PLE state needs exactly pliDim bf16 values.
// A shorter slab must fail before any command buffer can encode a partial gate.
func TestStepTokensBatchedDensePLE_Bad(t *testing.T) {
	state := &archDecodeState{
		specs:  []model.LayerSpec{{}},
		ple:    []pleLayer{{}},
		dModel: 1,
		pliDim: 1,
	}
	_, ok, err := state.stepTokensBatchedDensePLE([][]byte{{0, 0}}, []byte{0}, 0)
	if err == nil {
		t.Fatal("stepTokensBatchedDensePLE accepted a PLE slab shorter than one layer row")
	}
	if ok {
		t.Fatal("stepTokensBatchedDensePLE reported ok with a malformed PLE slab")
	}
}

// TestStepTokensBatchedDensePLE_Ugly pins the deliberate decline rather than
// guessing at a fallback: an otherwise PLE-shaped state without a host or
// device slab returns !ok and no error, leaving sequential stepping authoritative.
func TestStepTokensBatchedDensePLE_Ugly(t *testing.T) {
	state := &archDecodeState{
		specs:  []model.LayerSpec{{}},
		ple:    []pleLayer{{}},
		dModel: 1,
		pliDim: 1,
	}
	_, ok, err := state.stepTokensBatchedDensePLE([][]byte{{0, 0}}, nil, 0)
	if err != nil {
		t.Fatalf("stepTokensBatchedDensePLE without a slab: %v", err)
	}
	if ok {
		t.Fatal("stepTokensBatchedDensePLE accepted a PLE arch without a slab")
	}
}

// TestDenseBatchScratchAttnFoldGrowsWithRows pins the ~52K long-context corruption's root
// cause: attnFold must reallocate its slabs when the batch row count GROWS, independent of
// mlpFold. The production call order is mlpFold first (which raises the shared-looking
// foldRowCap), then attnFold — the old attnFold keyed its growth check on foldRowCap and so
// skipped the realloc for the one wide tail-absorbed chunk (window + tail rows), leaving every
// attention slab short: rows past the stale capacity read/wrote out of bounds (undefined bytes,
// NaN/garbage varying per process). The wide-chunk shape only occurs when promptLen mod window
// lands in (0, window/2], which is why it escaped every fixed-size fixture.
func TestDenseBatchScratchAttnFoldGrowsWithRows(t *testing.T) {
	requireNativeRuntime(t)
	const dModel, qDim, kvDim, dFF = 2048, 2048, 256, 8192
	s := &denseBatchScratch{}
	// the steady prompt chunks: window-sized batches
	s.mlpFold(512, dModel, dFF)
	normed, q, attn, attnOut, kSt, vSt := s.attnFold(512, dModel, qDim, kvDim)
	if int(bufferLengthFast(q)) < 512*qDim*bf16Size {
		t.Fatalf("baseline q slab too small: %d", bufferLengthFast(q))
	}
	_, _, _, _, _ = normed, attn, attnOut, kSt, vSt
	// the wide tail-absorbed chunk: window + tail rows, mlpFold FIRST (production order)
	const wide = 724
	s.mlpFold(wide, dModel, dFF)
	normed, q, attn, attnOut, kSt, vSt = s.attnFold(wide, dModel, qDim, kvDim)
	check := func(name string, got, want int) {
		t.Helper()
		if got < want {
			t.Fatalf("attnFold %s slab did not grow with the wide chunk: %d bytes, want >= %d — rows past the stale capacity read/write out of bounds", name, got, want)
		}
	}
	check("attnNorm", int(bufferLengthFast(normed)), wide*dModel*bf16Size)
	check("q", int(bufferLengthFast(q)), wide*qDim*bf16Size)
	check("attn", int(bufferLengthFast(attn)), wide*qDim*bf16Size)
	check("attnOut", int(bufferLengthFast(attnOut)), wide*dModel*bf16Size)
	check("kStage", int(bufferLengthFast(kSt)), wide*kvDim*bf16Size)
	check("vStage", int(bufferLengthFast(vSt)), wide*kvDim*bf16Size)
}

// TestDenseBatchScratchReleasesSlabsOnGrowthAndClose pins the slab lifecycle
// (#367 census): every grow-realloc releases the outgrown slabs and Close
// releases the whole set — the old code dropped +1 retained handles at both
// seams, stacking GB-scale sdpaPromptS pairs across widening chunks and
// leaking the grown set on every session teardown. Device-level receipt:
// grow + close in a loop must return CurrentAllocatedSize to ~baseline.
func TestDenseBatchScratchReleasesSlabsOnGrowthAndClose(t *testing.T) {
	requireNativeRuntime(t)
	const dModel, dFF, qDim, kvDim = 512, 1024, 512, 512
	const rowsA, rowsB = 256, 1024
	const sCols = 16384 // sdpaS pair at rowsB: 2 × 1024×16384×2B = 64MB — dominates noise
	baseline := device.CurrentAllocatedSize()
	for range 4 {
		var s denseBatchScratch
		s.mlpFold(rowsA, dModel, dFF)
		s.attnFold(rowsA, dModel, qDim, kvDim)
		s.layerStage(0, 3, rowsA, kvDim)
		s.sdpaPromptS(rowsA, sCols)
		s.q8Stage(0, rowsA, kvDim)
		s.q8Stage(1, rowsA, kvDim)
		// widen: every grow site reallocates (and must release the outgrown set)
		s.mlpFold(rowsB, dModel, dFF)
		s.attnFold(rowsB, dModel, qDim, kvDim)
		s.layerStage(1, 3, rowsB, kvDim)
		s.sdpaPromptS(rowsB, sCols)
		s.q8Stage(0, rowsB*64, kvDim) // q8 staging grows to the attended prefix — force a realloc
		s.q8Stage(1, rowsB*64, kvDim)
		s.Close()
	}
	end := device.CurrentAllocatedSize()
	var grew uint
	if end > baseline {
		grew = end - baseline
	}
	// one leaked generation would exceed 64MB (the sdpaS pair alone); allow slack
	// for allocator noise well below it.
	if grew > 32<<20 {
		t.Fatalf("scratch slabs not reclaimed: device allocation grew %dMB across grow+close cycles", grew>>20)
	}
}
