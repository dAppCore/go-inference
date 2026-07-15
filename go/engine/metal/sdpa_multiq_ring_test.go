// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"math"
	"sync"
	"testing"
	"unsafe"
)

// requireSDPAMultiQRingKernel skips the test unless the custom lthn_sdpa_multiq_ring_bf16 and
// lthn_copy_bf16 kernels are loaded (the same lthn_kernels.metallib gate as the paged/multiq
// SDPA tests).
func requireSDPAMultiQRingKernel(t *testing.T) {
	t.Helper()
	requireNativeRuntime(t)
	if customLibrary == nil || customLibrary.GetID() == 0 {
		t.Skip("custom kernel library (lthn_kernels.metallib) not loaded")
	}
	if fn := customLibrary.NewFunctionWithName("lthn_sdpa_multiq_ring_bf16_64"); fn == nil || fn.GetID() == 0 {
		t.Skip("custom staged-ring multi-query SDPA kernel not loaded - run `task build:kernels`")
	}
}

// TestSdpaMultiqRing_sdpaMultiQRingPipelineForHeadDim_Good proves the pipeline resolves (and
// caches) for a shipped gemma head geometry.
func TestSdpaMultiqRing_sdpaMultiQRingPipelineForHeadDim_Good(t *testing.T) {
	requireSDPAMultiQRingKernel(t)

	pso, ok := sdpaMultiQRingPipelineForHeadDim(64)
	if !ok || pso == nil {
		t.Fatal("sdpaMultiQRingPipelineForHeadDim(64): expected a resolved pipeline for a shipped head geometry")
	}
	pso2, ok2 := sdpaMultiQRingPipelineForHeadDim(64)
	if !ok2 || pso2 != pso {
		t.Fatal("sdpaMultiQRingPipelineForHeadDim(64): second call did not return the cached pipeline")
	}
}

// TestSdpaMultiqRing_sdpaMultiQRingPipelineForHeadDim_Bad proves an un-instantiated head
// dimension reports unavailable rather than panicking or fabricating a pipeline.
func TestSdpaMultiqRing_sdpaMultiQRingPipelineForHeadDim_Bad(t *testing.T) {
	requireNativeRuntime(t)

	if _, ok := sdpaMultiQRingPipelineForHeadDim(33); ok {
		t.Fatal("sdpaMultiQRingPipelineForHeadDim(33): expected unavailable for a non-shipped head geometry")
	}
}

// TestSdpaMultiqRing_gpuHasSDPAMultiQRing_Good proves gpuHasSDPAMultiQRing agrees with the
// pipeline resolution it wraps.
func TestSdpaMultiqRing_gpuHasSDPAMultiQRing_Good(t *testing.T) {
	requireSDPAMultiQRingKernel(t)

	if !gpuHasSDPAMultiQRing(64) {
		t.Fatal("gpuHasSDPAMultiQRing(64): expected true once the kernel library is loaded")
	}
}

// TestSdpaMultiqRing_copyPipeline_Good proves the contiguous bf16 copy pipeline resolves and
// caches (sync.Once), and gpuHasCopyKernel agrees.
func TestSdpaMultiqRing_copyPipeline_Good(t *testing.T) {
	requireSDPAMultiQRingKernel(t)

	pso, err := copyPipeline()
	if err != nil || pso == nil {
		t.Fatalf("copyPipeline: %v (pso=%v)", err, pso)
	}
	pso2, err2 := copyPipeline()
	if err2 != nil || pso2 != pso {
		t.Fatalf("copyPipeline: second call did not return the cached pipeline (err=%v)", err2)
	}
	if !gpuHasCopyKernel() {
		t.Fatal("gpuHasCopyKernel: expected true once the kernel library is loaded")
	}
}

// TestSdpaMultiqRing_encCopyBF16Contig_Good proves the deferred-landing copy is a per-element
// identity: out[0..n) == in[0..n) bf16 elements, both starting at their given byte offsets.
func TestSdpaMultiqRing_encCopyBF16Contig_Good(t *testing.T) {
	requireSDPAMultiQRingKernel(t)

	const n = 32
	in := toBF16Bytes(syntheticFloat32(n, 3))
	inBuf := sharedBytes(in)
	outBuf := scratchBF16(n)

	cb := commandBufferFast(queue)
	enc := computeCommandEncoderFast(cb)
	if err := encCopyBF16Contig(enc, inBuf, outBuf, 0, 0, n); err != nil {
		endEncodingFast(enc)
		t.Fatalf("encCopyBF16Contig: %v", err)
	}
	endEncodingFast(enc)
	commitCommandBufferFast(cb)
	waitUntilCompletedFast(cb)

	got := unsafe.Slice((*byte)(outBuf.Contents()), n*bf16Size)
	eqBytes(t, "encCopyBF16Contig", got, in)
}

// TestSdpaMultiqRing_encCopyBF16Contig_Bad proves the copy dispatch reports an error rather than
// dispatching when the custom kernel library is unavailable — simulated here by nilling
// customLibrary for the duration of the call (restored via defer).
func TestSdpaMultiqRing_encCopyBF16Contig_Bad(t *testing.T) {
	requireNativeRuntime(t)

	prev := customLibrary
	customLibrary = nil
	t.Cleanup(func() {
		customLibrary = prev
		// copyPipeline's sync.Once cached the "unavailable" failure against the nilled
		// library; clear it so later tests re-resolve against the restored, real library.
		copyPSOOnce = sync.Once{}
		copyPSO, copyPSOErr = nil, nil
	})
	// force copyPipeline to re-resolve against the now-nil library rather than returning an
	// already-cached pipeline from an earlier test.
	copyPSOOnce = sync.Once{}
	copyPSO, copyPSOErr = nil, nil

	const n = 8
	in := sharedBytes(toBF16Bytes(syntheticFloat32(n, 3)))
	out := scratchBF16(n)
	cb := commandBufferFast(queue)
	enc := computeCommandEncoderFast(cb)
	defer endEncodingFast(enc)
	if err := encCopyBF16Contig(enc, in, out, 0, 0, n); err == nil {
		t.Fatal("expected encCopyBF16Contig to reject a missing custom kernel library")
	}
}

// TestSdpaMultiqRing_encCopyBF16Contig_Ugly exercises nonzero in/out byte offsets — the deferred
// landing copies a run starting mid-buffer, not just from byte 0.
func TestSdpaMultiqRing_encCopyBF16Contig_Ugly(t *testing.T) {
	requireSDPAMultiQRingKernel(t)

	const total, n, inOff, outOff = 16, 4, 6 * bf16Size, 2 * bf16Size
	in := toBF16Bytes(syntheticFloat32(total, 5))
	inBuf := sharedBytes(in)
	outBuf := scratchBF16(total)
	// sentinel-fill the destination so untouched regions are distinguishable from the copy.
	sentinel := make([]byte, total*bf16Size)
	for i := range sentinel {
		sentinel[i] = 0xA5
	}
	copy(unsafe.Slice((*byte)(outBuf.Contents()), total*bf16Size), sentinel)

	cb := commandBufferFast(queue)
	enc := computeCommandEncoderFast(cb)
	if err := encCopyBF16Contig(enc, inBuf, outBuf, uint(inOff), uint(outOff), n); err != nil {
		endEncodingFast(enc)
		t.Fatalf("encCopyBF16Contig: %v", err)
	}
	endEncodingFast(enc)
	commitCommandBufferFast(cb)
	waitUntilCompletedFast(cb)

	got := unsafe.Slice((*byte)(outBuf.Contents()), total*bf16Size)
	eqBytes(t, "encCopyBF16Contig (offset run)", got[outOff:outOff+n*bf16Size], in[inOff:inOff+n*bf16Size])
	eqBytes(t, "encCopyBF16Contig (untouched prefix)", got[:outOff], sentinel[:outOff])
}

// TestSdpaMultiqRing_encSDPAMultiQRing_Good pins the fresh/empty-ring degenerate case
// (ringLive=0, slideW wider than the batch): with nothing in the pre-batch ring, the two-segment
// attention must reduce to a plain causal fold over just the staged K rows — exactly what
// encSDPAMultiQCausal computes for kRows==nTotal (no base). ringLive=0 is an explicitly
// documented supported shape ("the kernel handles a partial or fresh ring").
func TestSdpaMultiqRing_encSDPAMultiQRing_Good(t *testing.T) {
	requireSDPAMultiQRingKernel(t)

	const nHeads, nKVHeads, headDim = 2, 1, 64
	const kRows, slideW = 3, 64 // slideW wider than the batch: no in-batch window truncation
	kvDim := nKVHeads * headDim
	scale := float32(1.0 / math.Sqrt(float64(headDim)))

	q := toBF16Bytes(syntheticFloat32(kRows*nHeads*headDim, 3))
	stageK := toBF16Bytes(syntheticFloat32(kRows*kvDim, 5))
	stageV := toBF16Bytes(syntheticFloat32(kRows*kvDim, 7))
	// ring buffers sized to slideW rows so any in-kernel read stays in-bounds even though
	// ringLive=0 means none of it should be attended.
	ringK := toBF16Bytes(syntheticFloat32(slideW*kvDim, 11))
	ringV := toBF16Bytes(syntheticFloat32(slideW*kvDim, 13))

	qBuf := sharedBytes(q)
	stageKBuf, stageVBuf := sharedBytes(stageK), sharedBytes(stageV)
	ringKBuf, ringVBuf := sharedBytes(ringK), sharedBytes(ringV)
	outBuf := scratchBF16(kRows * nHeads * headDim)

	khs, kss := int64(headDim), int64(kvDim)
	cb := commandBufferFast(queue)
	enc := computeCommandEncoderFast(cb)
	if err := encSDPAMultiQRing(enc, qBuf, ringKBuf, ringVBuf, stageKBuf, stageVBuf, outBuf,
		nHeads, nKVHeads, headDim, kRows, slideW, 0, 0, khs, kss, khs, kss, scale); err != nil {
		endEncodingFast(enc)
		t.Fatalf("encSDPAMultiQRing: %v", err)
	}
	endEncodingFast(enc)
	commitCommandBufferFast(cb)
	waitUntilCompletedFast(cb)

	rowBytes := nHeads * headDim * bf16Size
	got := append([]byte(nil), unsafe.Slice((*byte)(outBuf.Contents()), kRows*rowBytes)...)

	// oracle: encSDPAMultiQCausal over the SAME staged rows as the sole cache (kRows==nTotal,
	// no base) — proven byte/cosine-identical to per-row single-query SDPA in
	// sdpa_multiq_test.go.
	wantBuf := scratchBF16(kRows * nHeads * headDim)
	cb2 := commandBufferFast(queue)
	enc2 := computeCommandEncoderFast(cb2)
	if err := encSDPAMultiQCausal(enc2, qBuf, stageKBuf, stageVBuf, wantBuf, nHeads, nKVHeads, headDim, kRows, kRows, khs, kss, khs, kss, scale); err != nil {
		endEncodingFast(enc2)
		t.Fatalf("encSDPAMultiQCausal oracle: %v", err)
	}
	endEncodingFast(enc2)
	commitCommandBufferFast(cb2)
	waitUntilCompletedFast(cb2)
	want := append([]byte(nil), unsafe.Slice((*byte)(wantBuf.Contents()), kRows*rowBytes)...)

	if cos := cosineBF16(got, want); cos < 0.999 {
		t.Fatalf("fresh-ring (ringLive=0) staged attention cosine=%.6f vs plain causal fold over the staged rows, want ~1", cos)
	}
}

// TestSdpaMultiqRing_encSDPAMultiQRing_Bad proves the dispatch reports an error rather than
// dispatching against a headDim the ring kernel library never instantiated.
func TestSdpaMultiqRing_encSDPAMultiQRing_Bad(t *testing.T) {
	requireNativeRuntime(t)

	const nHeads, nKVHeads, headDim = 2, 1, 33 // not a shipped {64,128,256,512} geometry
	kvDim := nKVHeads * headDim
	q := sharedBytes(toBF16Bytes(syntheticFloat32(nHeads*headDim, 3)))
	stageK, stageV := sharedBytes(toBF16Bytes(syntheticFloat32(kvDim, 5))), sharedBytes(toBF16Bytes(syntheticFloat32(kvDim, 7)))
	ringK, ringV := sharedBytes(toBF16Bytes(syntheticFloat32(kvDim, 11))), sharedBytes(toBF16Bytes(syntheticFloat32(kvDim, 13)))
	out := scratchBF16(nHeads * headDim)

	cb := commandBufferFast(queue)
	enc := computeCommandEncoderFast(cb)
	defer endEncodingFast(enc)
	khs, kss := int64(headDim), int64(kvDim)
	if err := encSDPAMultiQRing(enc, q, ringK, ringV, stageK, stageV, out, nHeads, nKVHeads, headDim, 1, 1, 0, 0, khs, kss, khs, kss, 0.125); err == nil {
		t.Fatal("expected encSDPAMultiQRing to reject a headDim with no compiled kernel")
	}
}
