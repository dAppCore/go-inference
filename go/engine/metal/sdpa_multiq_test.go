// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"math"
	"testing"
	"unsafe"
)

// requireSDPAMultiQKernel skips the test unless the custom lthn_sdpa_multiq_bf16 kernel is
// loaded (the same lthn_kernels.metallib gate as the paged-SDPA tests).
func requireSDPAMultiQKernel(t *testing.T) {
	t.Helper()
	requireNativeRuntime(t)
	if customLibrary == nil || customLibrary.GetID() == 0 {
		t.Skip("custom kernel library (lthn_kernels.metallib) not loaded")
	}
	if fn := customLibrary.NewFunctionWithName("lthn_sdpa_multiq_bf16_64"); fn == nil || fn.GetID() == 0 {
		t.Skip("custom multi-query causal SDPA kernel not loaded - run `task build:kernels`")
	}
}

// TestSdpaMultiq_sdpaMultiQPipelineForHeadDim_Good proves the pipeline resolves (and caches) for
// a shipped gemma head geometry.
func TestSdpaMultiq_sdpaMultiQPipelineForHeadDim_Good(t *testing.T) {
	requireSDPAMultiQKernel(t)

	pso, ok := sdpaMultiQPipelineForHeadDim(64)
	if !ok || pso == nil {
		t.Fatal("sdpaMultiQPipelineForHeadDim(64): expected a resolved pipeline for a shipped head geometry")
	}
	// second call must hit the cache and return the identical pipeline object.
	pso2, ok2 := sdpaMultiQPipelineForHeadDim(64)
	if !ok2 || pso2 != pso {
		t.Fatal("sdpaMultiQPipelineForHeadDim(64): second call did not return the cached pipeline")
	}
}

// TestSdpaMultiq_sdpaMultiQPipelineForHeadDim_Bad proves an un-instantiated head dimension
// reports unavailable rather than panicking or fabricating a pipeline.
func TestSdpaMultiq_sdpaMultiQPipelineForHeadDim_Bad(t *testing.T) {
	requireNativeRuntime(t)

	if _, ok := sdpaMultiQPipelineForHeadDim(33); ok {
		t.Fatal("sdpaMultiQPipelineForHeadDim(33): expected unavailable for a non-shipped head geometry")
	}
}

// TestSdpaMultiq_gpuHasSDPAMultiQ_Good proves gpuHasSDPAMultiQ agrees with the pipeline
// resolution it wraps.
func TestSdpaMultiq_gpuHasSDPAMultiQ_Good(t *testing.T) {
	requireSDPAMultiQKernel(t)

	if !gpuHasSDPAMultiQ(64) {
		t.Fatal("gpuHasSDPAMultiQ(64): expected true once the kernel library is loaded")
	}
}

// TestSdpaMultiq_encSDPAMultiQCausal_Good proves the batched K-query causal dispatch is
// byte/cosine-identical, row by row, to K single-query encSDPAStrided dispatches over the SAME
// seq-major cache — the fold's whole correctness claim.
func TestSdpaMultiq_encSDPAMultiQCausal_Good(t *testing.T) {
	requireSDPAMultiQKernel(t)

	const nHeads, nKVHeads, headDim = 2, 1, 64
	const kRows, nBase = 3, 2
	nTotal := nBase + kRows
	kvDim := nKVHeads * headDim
	scale := float32(1.0 / math.Sqrt(float64(headDim)))

	qFull := toBF16Bytes(syntheticFloat32(kRows*nHeads*headDim, 3))
	kCache := toBF16Bytes(syntheticFloat32(nTotal*kvDim, 5))
	vCache := toBF16Bytes(syntheticFloat32(nTotal*kvDim, 7))

	qBuf := sharedBytes(qFull)
	kBuf := sharedBytes(kCache)
	vBuf := sharedBytes(vCache)
	outBuf := scratchBF16(kRows * nHeads * headDim)

	khs, kss := int64(headDim), int64(kvDim)
	cb := commandBufferFast(queue)
	enc := computeCommandEncoderFast(cb)
	if err := encSDPAMultiQCausal(enc, qBuf, kBuf, vBuf, outBuf, nHeads, nKVHeads, headDim, kRows, nTotal, khs, kss, khs, kss, scale); err != nil {
		endEncodingFast(enc)
		t.Fatalf("encSDPAMultiQCausal: %v", err)
	}
	endEncodingFast(enc)
	commitCommandBufferFast(cb)
	waitUntilCompletedFast(cb)

	rowBytes := nHeads * headDim * bf16Size
	got := append([]byte(nil), unsafe.Slice((*byte)(outBuf.Contents()), kRows*rowBytes)...)

	for s := 0; s < kRows; s++ {
		n := nBase + s + 1 // causal limit: key i valid iff i <= nTotal-kRows+s = nBase+s
		qRow := sharedBytes(qFull[s*rowBytes : (s+1)*rowBytes])
		rowOut := scratchBF16(nHeads * headDim)

		cb2 := commandBufferFast(queue)
		enc2 := computeCommandEncoderFast(cb2)
		if err := encSDPAStrided(enc2, qRow, kBuf, vBuf, rowOut, nHeads, nKVHeads, headDim, n, khs, kss, khs, kss, scale, 0); err != nil {
			endEncodingFast(enc2)
			t.Fatalf("encSDPAStrided (row %d): %v", s, err)
		}
		endEncodingFast(enc2)
		commitCommandBufferFast(cb2)
		waitUntilCompletedFast(cb2)

		want := unsafe.Slice((*byte)(rowOut.Contents()), rowBytes)
		gotRow := got[s*rowBytes : (s+1)*rowBytes]
		if cos := cosineBF16(gotRow, want); cos < 0.999 {
			t.Fatalf("row %d: multi-query causal cosine=%.6f vs single-query encSDPAStrided (n=%d), want ~1", s, cos, n)
		}
	}
}

// TestSdpaMultiq_encSDPAMultiQCausal_Bad proves the dispatch reports an error rather than
// dispatching against a headDim the kernel library never instantiated.
func TestSdpaMultiq_encSDPAMultiQCausal_Bad(t *testing.T) {
	requireNativeRuntime(t)

	const nHeads, nKVHeads, headDim = 2, 1, 33 // not one of the shipped {64,128,256,512} geometries
	kvDim := nKVHeads * headDim
	q := sharedBytes(toBF16Bytes(syntheticFloat32(nHeads*headDim, 3)))
	k := sharedBytes(toBF16Bytes(syntheticFloat32(kvDim, 5)))
	v := sharedBytes(toBF16Bytes(syntheticFloat32(kvDim, 7)))
	out := scratchBF16(nHeads * headDim)

	cb := commandBufferFast(queue)
	enc := computeCommandEncoderFast(cb)
	defer endEncodingFast(enc)
	khs, kss := int64(headDim), int64(kvDim)
	if err := encSDPAMultiQCausal(enc, q, k, v, out, nHeads, nKVHeads, headDim, 1, 1, khs, kss, khs, kss, 0.125); err == nil {
		t.Fatal("expected encSDPAMultiQCausal to reject a headDim with no compiled kernel")
	}
}
