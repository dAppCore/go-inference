// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"math"
	"testing"
	"unsafe"

	"github.com/tmc/apple/metal"
)

func resetFlashPromptCachesForTest() {
	steelAttnPSOMu.Lock()
	steelAttnPSOCache = map[steelAttnKey]metal.MTLComputePipelineState{}
	steelAttnPSOMu.Unlock()
	splitDAttnPSOMu.Lock()
	splitDAttnPSOCache = map[steelAttnKey]metal.MTLComputePipelineState{}
	splitDAttnPSOMu.Unlock()
	flashQ8PSOMu.Lock()
	flashQ8PSOCache = map[flashQ8Key]metal.MTLComputePipelineState{}
	flashQ8PSOMu.Unlock()
	flashWinPSOMu.Lock()
	flashWinPSOCache = map[steelAttnKey]metal.MTLComputePipelineState{}
	flashWinPSOMu.Unlock()
	flashPromptPSOMu.Lock()
	flashPromptPSOCache = map[int]metal.MTLComputePipelineState{}
	flashPromptPSOMu.Unlock()
}

func withWrongFlashCustomLibrary(t *testing.T, fn func()) {
	t.Helper()
	requireNativeRuntime(t)
	old := customLibrary
	customLibrary = library
	resetFlashPromptCachesForTest()
	t.Cleanup(func() {
		customLibrary = old
		resetFlashPromptCachesForTest()
	})
	fn()
}

func TestFlashQ8Usable_UnsupportedHeadDim(t *testing.T) {
	if flashQ8Usable(128, splitDAttnMinKV) {
		t.Fatal("flashQ8Usable accepted an unsupported head dimension")
	}
}

func TestFlashQ8Usable_Short512Depth(t *testing.T) {
	if flashQ8Usable(512, splitDAttnMinKV-1) {
		t.Fatal("flashQ8Usable accepted a 512-wide request below its crossover")
	}
}

func TestFlashWinUsable_UnsupportedHeadDim(t *testing.T) {
	if flashWinUsable(512) {
		t.Fatal("flashWinUsable accepted a non-256 head dimension")
	}
}

func TestGPUHasFlashPrompt_UnsupportedHeadDim(t *testing.T) {
	if gpuHasFlashPrompt(128) {
		t.Fatal("gpuHasFlashPrompt accepted an unsupported head dimension")
	}
}

func TestFlashPromptUsable_UnsupportedHeadDim(t *testing.T) {
	if flashPromptUsable(128, splitDAttnMinKV) {
		t.Fatal("flashPromptUsable accepted an unsupported head dimension")
	}
}

func TestSplitDAttnPipeline_BadWrongLibrary(t *testing.T) {
	withWrongFlashCustomLibrary(t, func() {
		if pso, ok := splitDAttnPipeline(false, false); ok || pso != nil {
			t.Fatal("splitDAttnPipeline resolved a kernel from the wrong metallib")
		}
	})
}

func TestEncSteelAttnSplitD_BadWrongLibrary(t *testing.T) {
	withWrongFlashCustomLibrary(t, func() {
		var noEncoder metal.MTLComputeCommandEncoderObject
		err := encSteelAttnSplitD(noEncoder, nil, nil, nil, nil, 2, 1, 512, 3, 19, 1024, 512, 0.044)
		if err == nil {
			t.Fatal("encSteelAttnSplitD accepted an unavailable split-D pipeline")
		}
	})
}

func TestEncFlashPromptQ8_BadWrongLibrary(t *testing.T) {
	withWrongFlashCustomLibrary(t, func() {
		var noEncoder metal.MTLComputeCommandEncoderObject
		err := encFlashPromptQ8(noEncoder, nil, nil, nil, nil, nil, nil, 2, 1, 256, 3, 19, 512, 256, 0.044)
		if err == nil {
			t.Fatal("encFlashPromptQ8 accepted an unavailable q8 pipeline")
		}
	})
}

func TestFlashWinPipeline_BadWrongLibrary(t *testing.T) {
	withWrongFlashCustomLibrary(t, func() {
		if pso, ok := flashWinPipeline(false); ok || pso != nil {
			t.Fatal("flashWinPipeline resolved a kernel from the wrong metallib")
		}
	})
}

func TestEncFlashWindowSDPA_BadWrongLibrary(t *testing.T) {
	withWrongFlashCustomLibrary(t, func() {
		var noEncoder metal.MTLComputeCommandEncoderObject
		err := encFlashWindowSDPA(noEncoder, nil, nil, nil, nil, nil, nil, 2, 1, 256, 3, 32, 2, 2, 512, 256, 0.044)
		if err == nil {
			t.Fatal("encFlashWindowSDPA accepted an unavailable window pipeline")
		}
	})
}

func TestFlashPromptPipeline_BadWrongLibrary(t *testing.T) {
	withWrongFlashCustomLibrary(t, func() {
		if pso, ok := flashPromptPipeline(999); ok || pso != nil {
			t.Fatal("flashPromptPipeline resolved an unsupported kernel from the wrong metallib")
		}
	})
}

func TestEncFlashPromptSDPA_BadWrongLibrary(t *testing.T) {
	withWrongFlashCustomLibrary(t, func() {
		var noEncoder metal.MTLComputeCommandEncoderObject
		err := encFlashPromptSDPA(noEncoder, nil, nil, nil, nil, 2, 1, 999, 3, 19, 1998, 999, 0.044)
		if err == nil {
			t.Fatal("encFlashPromptSDPA accepted an unavailable flash prompt pipeline")
		}
	})
}

func TestEncSteelAttnSplitD_GoodMatchesMultiQReference(t *testing.T) {
	requireNativeRuntime(t)
	if !gpuHasSDPAMultiQ(512) || !flashPromptUsable(512, splitDAttnMinKV) {
		t.Skip("split-D or the 512 reference lane is unavailable")
	}
	const nHeads, nKVHeads, hd, kRows, nTotal = 2, 1, 512, 4, 20
	const qDim, kvDim = nHeads * hd, nKVHeads * hd
	scale := float32(1.0 / math.Sqrt(float64(hd)))
	q := sharedBytes(toBF16Bytes(syntheticFloat32(kRows*qDim, 11)))
	k := sharedBytes(toBF16Bytes(syntheticFloat32(nTotal*kvDim, 13)))
	v := sharedBytes(toBF16Bytes(syntheticFloat32(nTotal*kvDim, 17)))
	splitOut := scratchBF16(kRows * qDim)
	cb := commandBufferFast(queue)
	enc := computeCommandEncoderFast(cb)
	if err := encSteelAttnSplitD(enc, q, k, v, splitOut, nHeads, nKVHeads, hd, kRows, nTotal, qDim, kvDim, scale); err != nil {
		endEncodingFast(enc)
		t.Fatalf("encSteelAttnSplitD: %v", err)
	}
	endEncodingFast(enc)
	commitCommandBufferFast(cb)
	waitUntilCompletedFast(cb)
	got := append([]byte(nil), unsafe.Slice((*byte)(splitOut.Contents()), kRows*qDim*bf16Size)...)

	refOut := scratchBF16(kRows * qDim)
	cbRef := commandBufferFast(queue)
	encRef := computeCommandEncoderFast(cbRef)
	if err := encSDPAMultiQCausal(encRef, q, k, v, refOut, nHeads, nKVHeads, hd, kRows, nTotal, int64(hd), int64(kvDim), int64(hd), int64(kvDim), scale); err != nil {
		endEncodingFast(encRef)
		t.Fatalf("encSDPAMultiQCausal reference: %v", err)
	}
	endEncodingFast(encRef)
	commitCommandBufferFast(cbRef)
	waitUntilCompletedFast(cbRef)
	want := unsafe.Slice((*byte)(refOut.Contents()), kRows*qDim*bf16Size)
	if cos := cosineBF16(got, want); cos < 0.995 {
		t.Fatalf("split-D attention cosine = %.6f vs multiQ reference, want >= 0.995", cos)
	}
}

func TestEncFlashPromptQ8_GoodMatchesQ8Reference(t *testing.T) {
	requireNativeRuntime(t)
	if !flashQ8Usable(256, 19) || !gpuHasSDPAMultiQQ8(256) {
		t.Skip("q8 flash or q8 multi-query reference lane is unavailable")
	}
	const nHeads, nKVHeads, hd, kRows, nTotal = 2, 1, 256, 3, 19
	const qDim, kvDim = nHeads * hd, nKVHeads * hd
	scale := float32(1.0 / math.Sqrt(float64(hd)))
	q := sharedBytes(toBF16Bytes(syntheticFloat32(kRows*qDim, 19)))
	kBF16 := toBF16Bytes(syntheticFloat32(nTotal*kvDim, 23))
	vBF16 := toBF16Bytes(syntheticFloat32(nTotal*kvDim, 29))
	kCodes, kScales := kvQ8QuantiseRows(kBF16, kvDim)
	vCodes, vScales := kvQ8QuantiseRows(vBF16, kvDim)
	kScaleBytes := unsafe.Slice((*byte)(unsafe.Pointer(&kScales[0])), len(kScales)*4)
	vScaleBytes := unsafe.Slice((*byte)(unsafe.Pointer(&vScales[0])), len(vScales)*4)
	kCodesBuf, vCodesBuf := sharedBytes(kCodes), sharedBytes(vCodes)
	kScalesBuf, vScalesBuf := sharedBytes(kScaleBytes), sharedBytes(vScaleBytes)
	gotBuf := scratchBF16(kRows * qDim)
	cb := commandBufferFast(queue)
	enc := computeCommandEncoderFast(cb)
	if err := encFlashPromptQ8(enc, q, kCodesBuf, vCodesBuf, kScalesBuf, vScalesBuf, gotBuf, nHeads, nKVHeads, hd, kRows, nTotal, qDim, kvDim, scale); err != nil {
		endEncodingFast(enc)
		t.Fatalf("encFlashPromptQ8: %v", err)
	}
	endEncodingFast(enc)
	commitCommandBufferFast(cb)
	waitUntilCompletedFast(cb)
	got := unsafe.Slice((*byte)(gotBuf.Contents()), kRows*qDim*bf16Size)

	wantBuf := scratchBF16(kRows * qDim)
	cbRef := commandBufferFast(queue)
	encRef := computeCommandEncoderFast(cbRef)
	if err := encSDPAMultiQCausalQ8(encRef, q, kCodesBuf, vCodesBuf, wantBuf, kScalesBuf, vScalesBuf, nHeads, nKVHeads, hd, kRows, nTotal, int64(hd), int64(kvDim), int64(hd), int64(kvDim), scale); err != nil {
		endEncodingFast(encRef)
		t.Fatalf("encSDPAMultiQCausalQ8 reference: %v", err)
	}
	endEncodingFast(encRef)
	commitCommandBufferFast(cbRef)
	waitUntilCompletedFast(cbRef)
	want := unsafe.Slice((*byte)(wantBuf.Contents()), kRows*qDim*bf16Size)
	if cos := cosineBF16(got, want); cos < 0.995 {
		t.Fatalf("q8 flash attention cosine = %.6f vs q8 reference, want >= 0.995", cos)
	}
}

func TestEncFlashWindowSDPA_GoodMatchesCausalReference(t *testing.T) {
	requireNativeRuntime(t)
	if !flashWinUsable(256) || !gpuHasSDPAMultiQ(256) {
		t.Skip("window flash or multi-query reference lane is unavailable")
	}
	const nHeads, nKVHeads, hd, kRows, basePos = 2, 1, 256, 4, 2
	const nTotal, winW = basePos + kRows, 32
	const qDim, kvDim = nHeads * hd, nKVHeads * hd
	scale := float32(1.0 / math.Sqrt(float64(hd)))
	q := sharedBytes(toBF16Bytes(syntheticFloat32(kRows*qDim, 31)))
	allK := toBF16Bytes(syntheticFloat32(nTotal*kvDim, 37))
	allV := toBF16Bytes(syntheticFloat32(nTotal*kvDim, 41))
	ringK := make([]byte, winW*kvDim*bf16Size)
	ringV := make([]byte, winW*kvDim*bf16Size)
	copy(ringK, allK[:basePos*kvDim*bf16Size])
	copy(ringV, allV[:basePos*kvDim*bf16Size])
	kRing, vRing := sharedBytes(ringK), sharedBytes(ringV)
	kStage := sharedBytes(allK[basePos*kvDim*bf16Size:])
	vStage := sharedBytes(allV[basePos*kvDim*bf16Size:])
	gotBuf := scratchBF16(kRows * qDim)
	cb := commandBufferFast(queue)
	enc := computeCommandEncoderFast(cb)
	if err := encFlashWindowSDPA(enc, q, kRing, vRing, kStage, vStage, gotBuf, nHeads, nKVHeads, hd, kRows, winW, basePos, basePos, qDim, kvDim, scale); err != nil {
		endEncodingFast(enc)
		t.Fatalf("encFlashWindowSDPA: %v", err)
	}
	endEncodingFast(enc)
	commitCommandBufferFast(cb)
	waitUntilCompletedFast(cb)
	got := unsafe.Slice((*byte)(gotBuf.Contents()), kRows*qDim*bf16Size)

	wantBuf := scratchBF16(kRows * qDim)
	cbRef := commandBufferFast(queue)
	encRef := computeCommandEncoderFast(cbRef)
	if err := encSDPAMultiQCausal(encRef, q, sharedBytes(allK), sharedBytes(allV), wantBuf, nHeads, nKVHeads, hd, kRows, nTotal, int64(hd), int64(kvDim), int64(hd), int64(kvDim), scale); err != nil {
		endEncodingFast(encRef)
		t.Fatalf("encSDPAMultiQCausal reference: %v", err)
	}
	endEncodingFast(encRef)
	commitCommandBufferFast(cbRef)
	waitUntilCompletedFast(cbRef)
	want := unsafe.Slice((*byte)(wantBuf.Contents()), kRows*qDim*bf16Size)
	if cos := cosineBF16(got, want); cos < 0.995 {
		t.Fatalf("window flash attention cosine = %.6f vs causal reference, want >= 0.995", cos)
	}
}

func TestEncFlashPromptSDPA_GoodMatchesMultiQReference(t *testing.T) {
	requireNativeRuntime(t)
	if !flashPromptUsable(256, 19) || !gpuHasSDPAMultiQ(256) {
		t.Skip("flash prompt or multi-query reference lane is unavailable")
	}
	const nHeads, nKVHeads, hd, kRows, nTotal = 2, 1, 256, 4, 19
	const qDim, kvDim = nHeads * hd, nKVHeads * hd
	scale := float32(1.0 / math.Sqrt(float64(hd)))
	q := sharedBytes(toBF16Bytes(syntheticFloat32(kRows*qDim, 43)))
	k := sharedBytes(toBF16Bytes(syntheticFloat32(nTotal*kvDim, 47)))
	v := sharedBytes(toBF16Bytes(syntheticFloat32(nTotal*kvDim, 53)))
	gotBuf := scratchBF16(kRows * qDim)
	cb := commandBufferFast(queue)
	enc := computeCommandEncoderFast(cb)
	if err := encFlashPromptSDPA(enc, q, k, v, gotBuf, nHeads, nKVHeads, hd, kRows, nTotal, qDim, kvDim, scale); err != nil {
		endEncodingFast(enc)
		t.Fatalf("encFlashPromptSDPA: %v", err)
	}
	endEncodingFast(enc)
	commitCommandBufferFast(cb)
	waitUntilCompletedFast(cb)
	got := unsafe.Slice((*byte)(gotBuf.Contents()), kRows*qDim*bf16Size)

	wantBuf := scratchBF16(kRows * qDim)
	cbRef := commandBufferFast(queue)
	encRef := computeCommandEncoderFast(cbRef)
	if err := encSDPAMultiQCausal(encRef, q, k, v, wantBuf, nHeads, nKVHeads, hd, kRows, nTotal, int64(hd), int64(kvDim), int64(hd), int64(kvDim), scale); err != nil {
		endEncodingFast(encRef)
		t.Fatalf("encSDPAMultiQCausal reference: %v", err)
	}
	endEncodingFast(encRef)
	commitCommandBufferFast(cbRef)
	waitUntilCompletedFast(cbRef)
	want := unsafe.Slice((*byte)(wantBuf.Contents()), kRows*qDim*bf16Size)
	if cos := cosineBF16(got, want); cos < 0.995 {
		t.Fatalf("flash prompt attention cosine = %.6f vs multiQ reference, want >= 0.995", cos)
	}
}
