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
		for i := 0; i < prefix; i++ {
			if _, err := st.stepToken(embs[i], i); err != nil {
				t.Fatalf("prefill stepToken %d: %v", i, err)
			}
		}
		for i := 0; i < K; i++ {
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
		for i := 0; i < prefix; i++ {
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
	for i := 0; i < K; i++ {
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
		embs[i] = emb
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
		for i := 0; i < prefix+1; i++ {
			if _, err := st.stepToken(embs[i], i); err != nil {
				t.Fatalf("seq prefix stepToken %d: %v", i, err)
			}
		}
		for i := 0; i < K; i++ {
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
	for i := 0; i < K; i++ {
		eqBytes(t, core.Sprintf("batched after paged row %d vs stepToken", i), batOut[i], seqOut[i])
	}
}
