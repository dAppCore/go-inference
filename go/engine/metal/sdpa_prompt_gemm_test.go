// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"math"
	"testing"
	"unsafe"
)

// requirePromptSDPAGEMM skips the test unless every pipeline the GEMM SDPA composition
// needs is loadable: the steel GEMM pair from the main metallib and the causal softmax
// from the sibling lthn_kernels.metallib.
func requirePromptSDPAGEMM(t *testing.T) {
	t.Helper()
	requireNativeRuntime(t)
	if customLibrary == nil || customLibrary.GetID() == 0 {
		t.Skip("custom kernel library (lthn_kernels.metallib) not loaded")
	}
	if !gpuHasPromptSDPAGEMM() {
		t.Skip("prompt SDPA GEMM pipelines unavailable - rebuild lthn_kernels.metallib (engine/metal/kernels/README.md)")
	}
}

// TestSdpaPromptGemm_encSDPAPromptGEMM_Good proves the GEMM composition matches the multiQ
// vector kernel row-for-row on the same query-major slab and seq-major caches — the
// closeness bar for the token-identity tier (S stores bf16 between the GEMMs, so byte
// equality is not the claim; per-row cosine ~1 is).
func TestSdpaPromptGemm_encSDPAPromptGEMM_Good(t *testing.T) {
	requirePromptSDPAGEMM(t)
	requireSDPAMultiQKernel(t)

	const nHeads, nKVHeads, headDim = 4, 2, 64
	const kRows, nBase = 8, 71 // unaligned N exercises the bounds-checked steel tiles
	nTotal := nBase + kRows
	qDim := nHeads * headDim
	kvDim := nKVHeads * headDim
	scale := float32(1.0 / math.Sqrt(float64(headDim)))

	qFull := toBF16Bytes(syntheticFloat32(kRows*qDim, 3))
	kCache := toBF16Bytes(syntheticFloat32(nTotal*kvDim, 5))
	vCache := toBF16Bytes(syntheticFloat32(nTotal*kvDim, 7))

	qBuf := sharedBytes(qFull)
	kBuf := sharedBytes(kCache)
	vBuf := sharedBytes(vCache)

	run := func(gemm bool) []byte {
		outBuf := scratchBF16(kRows * qDim)
		cb := commandBufferFast(queue)
		enc := computeCommandEncoderFast(cb)
		var err error
		if gemm {
			gqa := nHeads / nKVHeads
			s0 := scratchBF16(gqa * kRows * nTotal)
			s1 := scratchBF16(gqa * kRows * nTotal)
			err = encSDPAPromptGEMM(enc, qBuf, kBuf, vBuf, outBuf, s0, s1,
				nHeads, nKVHeads, headDim, kRows, nTotal, qDim, kvDim, scale)
		} else {
			err = encSDPAMultiQCausal(enc, qBuf, kBuf, vBuf, outBuf, nHeads, nKVHeads, headDim, kRows, nTotal,
				int64(headDim), int64(kvDim), int64(headDim), int64(kvDim), scale)
		}
		if err != nil {
			endEncodingFast(enc)
			t.Fatalf("encode (gemm=%v): %v", gemm, err)
		}
		endEncodingFast(enc)
		commitCommandBufferFast(cb)
		waitUntilCompletedFast(cb)
		return append([]byte(nil), unsafe.Slice((*byte)(outBuf.Contents()), kRows*qDim*bf16Size)...)
	}

	got := run(true)
	want := run(false)
	rowBytes := qDim * bf16Size
	for s := range kRows {
		gotRow := got[s*rowBytes : (s+1)*rowBytes]
		wantRow := want[s*rowBytes : (s+1)*rowBytes]
		if cos := cosineBF16(gotRow, wantRow); cos < 0.999 {
			t.Fatalf("row %d: GEMM composition cosine=%.6f vs multiQ vector kernel, want ~1", s, cos)
		}
	}
}

// TestSdpaPromptGemm_encSDPAPromptGEMM_Bad proves a query-head count that is not a multiple
// of the kv-head count is rejected before any dispatch.
func TestSdpaPromptGemm_encSDPAPromptGEMM_Bad(t *testing.T) {
	requireNativeRuntime(t)

	const headDim = 64
	q := sharedBytes(toBF16Bytes(syntheticFloat32(3*headDim, 3)))
	k := sharedBytes(toBF16Bytes(syntheticFloat32(2*headDim, 5)))
	v := sharedBytes(toBF16Bytes(syntheticFloat32(2*headDim, 7)))
	out := scratchBF16(3 * headDim)
	s0 := scratchBF16(8)
	s1 := scratchBF16(8)

	cb := commandBufferFast(queue)
	enc := computeCommandEncoderFast(cb)
	defer endEncodingFast(enc)
	if err := encSDPAPromptGEMM(enc, q, k, v, out, s0, s1, 3, 2, headDim, 1, 1, 3*headDim, 2*headDim, 0.125); err == nil {
		t.Fatal("expected encSDPAPromptGEMM to reject nHeads not a multiple of nKVHeads")
	}
}

// TestSdpaPromptGemm_encSoftmaxCausalRows_Ugly pins the causal-cap semantics on the score
// slab directly: row s of a K-row chunk keeps keys [0 .. N-K+s], the masked tail is written
// EXACTLY zero (the following P @ V GEMM reads the full row), and the kept prefix sums to ~1.
func TestSdpaPromptGemm_encSoftmaxCausalRows_Ugly(t *testing.T) {
	requirePromptSDPAGEMM(t)

	const kRows, n = 3, 6
	scores := make([]float32, kRows*n)
	for i := range scores {
		scores[i] = float32(i%7) * 0.25
	}
	sBuf := sharedBytes(toBF16Bytes(scores))

	cb := commandBufferFast(queue)
	enc := computeCommandEncoderFast(cb)
	if err := encSoftmaxCausalRows(enc, sBuf, kRows, kRows, n, 1.0); err != nil {
		endEncodingFast(enc)
		t.Fatalf("encSoftmaxCausalRows: %v", err)
	}
	endEncodingFast(enc)
	commitCommandBufferFast(cb)
	waitUntilCompletedFast(cb)

	out := unsafe.Slice((*byte)(sBuf.Contents()), kRows*n*bf16Size)
	for s := range kRows {
		valid := n - kRows + s + 1
		sum := 0.0
		for i := range n {
			bits := uint16(out[(s*n+i)*2]) | uint16(out[(s*n+i)*2+1])<<8
			v := math.Float32frombits(uint32(bits) << 16)
			if i < valid {
				sum += float64(v)
			} else if v != 0 {
				t.Fatalf("row %d key %d: masked tail = %v, want exactly 0", s, i, v)
			}
		}
		if math.Abs(sum-1.0) > 0.02 {
			t.Fatalf("row %d: kept prefix sums to %.4f, want ~1", s, sum)
		}
	}
}
