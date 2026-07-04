// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"bytes"
	"testing"
	"unsafe"
)

func TestAttentionBlockMatchesComposedPrimitives(t *testing.T) {
	requireNativeRuntime(t)

	const dModel, nHeads, nKV, headDim, kvLen = 64, 1, 1, 64, 2
	const base, scale, offset, eps = float32(10000), float32(0.125), 1, float32(1e-5)
	qDim := nHeads * headDim
	x := toBF16Bytes(syntheticFloat32(dModel, 3))
	normW := toBF16Bytes(syntheticFloat32(dModel, 5))
	wQ := toBF16Bytes(syntheticFloat32(qDim*dModel, 7))
	wO := toBF16Bytes(syntheticFloat32(dModel*qDim, 11))
	kCache := toBF16Bytes(syntheticFloat32(nKV*kvLen*headDim, 13))
	vCache := toBF16Bytes(syntheticFloat32(nKV*kvLen*headDim, 17))

	got, err := AttentionBlock(x, normW, wQ, wO, kCache, vCache, dModel, nHeads, nKV, headDim, kvLen, base, scale, offset, eps)
	if err != nil {
		t.Fatalf("AttentionBlock: %v", err)
	}
	normed, err := RMSNormBF16(x, normW, 1, dModel, eps)
	if err != nil {
		t.Fatalf("RMSNormBF16: %v", err)
	}
	q, err := MatVecBF16(wQ, normed, qDim, dModel)
	if err != nil {
		t.Fatalf("MatVecBF16 q: %v", err)
	}
	qr, err := RoPEBF16(q, 1, nHeads, headDim, base, scale, offset, false)
	if err != nil {
		t.Fatalf("RoPEBF16: %v", err)
	}
	attn, err := SDPA(qr, kCache, vCache, 1, nHeads, nKV, headDim, kvLen, scale)
	if err != nil {
		t.Fatalf("SDPA: %v", err)
	}
	attnOut, err := MatVecBF16(wO, attn, dModel, qDim)
	if err != nil {
		t.Fatalf("MatVecBF16 o: %v", err)
	}
	want, err := AddBF16(x, attnOut)
	if err != nil {
		t.Fatalf("AddBF16: %v", err)
	}
	if !bytes.Equal(got, want) {
		t.Fatalf("AttentionBlock = %v, want composed primitives %v", bf16Floats(got), bf16Floats(want))
	}
}

func TestAttentionBlockIntoReusesOutputBackingAndBypassesScratchOutput(t *testing.T) {
	requireNativeRuntime(t)

	const dModel, nHeads, nKV, headDim, kvLen = 64, 1, 1, 64, 4
	const base, scale, offset, eps = float32(10000), float32(0.125), 1, float32(1e-5)
	layer := decodeLayerFixture(dModel, nHeads, nKV, headDim, 128, 3)
	x := toBF16Bytes(syntheticFloat32(dModel, 5))
	kCache := toBF16Bytes(syntheticFloat32(nKV*kvLen*headDim, 7))
	vCache := toBF16Bytes(syntheticFloat32(nKV*kvLen*headDim, 11))
	want, err := AttentionBlock(x, layer.AttnNormW, layer.WQ, layer.WO, kCache, vCache, dModel, nHeads, nKV, headDim, kvLen, base, scale, offset, eps)
	if err != nil {
		t.Fatalf("AttentionBlock reference: %v", err)
	}
	out := make([]byte, dModel*bf16Size)
	outPtr := unsafe.Pointer(&out[0])
	scratch, err := getQMVBF16Scratch(dModel, dModel)
	if err != nil {
		t.Fatalf("getQMVBF16Scratch: %v", err)
	}
	sentinel := bytes.Repeat([]byte{0xa5}, len(scratch.out.bytes))
	copy(scratch.out.bytes, sentinel)
	putQMVBF16Scratch(scratch)

	got, err := AttentionBlockInto(out, x, layer.AttnNormW, layer.WQ, layer.WO, kCache, vCache, dModel, nHeads, nKV, headDim, kvLen, base, scale, offset, eps)
	if err != nil {
		t.Fatalf("AttentionBlockInto: %v", err)
	}
	if len(got) != dModel*bf16Size || unsafe.Pointer(&got[0]) != outPtr {
		t.Fatal("AttentionBlockInto did not reuse caller-owned output backing")
	}
	eqBytes(t, "AttentionBlockInto", got, want)

	scratch, err = getQMVBF16Scratch(dModel, dModel)
	if err != nil {
		t.Fatalf("getQMVBF16Scratch after call: %v", err)
	}
	defer putQMVBF16Scratch(scratch)
	if !bytes.Equal(scratch.out.bytes, sentinel) {
		t.Fatal("AttentionBlockInto wrote through pooled scratch output instead of caller output")
	}
}

func TestAttentionBlockKeepsFixedWeightsResident(t *testing.T) {
	requireNativeRuntime(t)

	resetResidentBufsForTest()
	defer resetResidentBufsForTest()

	const dModel, nHeads, nKV, headDim, kvLen = 64, 1, 1, 64, 2
	const base, scale, offset, eps = float32(10000), float32(0.125), 1, float32(1e-5)
	qDim := nHeads * headDim
	x := toBF16Bytes(syntheticFloat32(dModel, 3))
	normW := toBF16Bytes(syntheticFloat32(dModel, 5))
	wQ := toBF16Bytes(syntheticFloat32(qDim*dModel, 7))
	wO := toBF16Bytes(syntheticFloat32(dModel*qDim, 11))
	kCache := toBF16Bytes(syntheticFloat32(nKV*kvLen*headDim, 13))
	vCache := toBF16Bytes(syntheticFloat32(nKV*kvLen*headDim, 17))

	if _, err := AttentionBlock(x, normW, wQ, wO, kCache, vCache, dModel, nHeads, nKV, headDim, kvLen, base, scale, offset, eps); err != nil {
		t.Fatalf("AttentionBlock: %v", err)
	}

	key := func(b []byte) uintptr { return uintptr(unsafe.Pointer(&b[0])) }
	residentBufMu.Lock()
	got := len(residentBufs)
	_, hasNorm := residentBufs[key(normW)]
	_, hasQ := residentBufs[key(wQ)]
	_, hasO := residentBufs[key(wO)]
	residentBufMu.Unlock()

	if !hasNorm || !hasQ || !hasO {
		t.Fatalf("AttentionBlock did not keep fixed weights resident (norm=%v q=%v o=%v resident=%d want>=3)", hasNorm, hasQ, hasO, got)
	}
}

func TestAttentionBlockKVScratchPoolKeepsDimensionsResident(t *testing.T) {
	requireNativeRuntime(t)

	small, err := getAttentionBlockKVScratch(128, 128)
	if err != nil {
		t.Fatalf("get small attention KV scratch: %v", err)
	}
	putAttentionBlockKVScratch(small)

	large, err := getAttentionBlockKVScratch(256, 256)
	if err != nil {
		t.Fatalf("get large attention KV scratch: %v", err)
	}
	putAttentionBlockKVScratch(large)
	forceNativeGC()
	forceNativeGC()

	gotSmall, err := getAttentionBlockKVScratch(128, 128)
	if err != nil {
		t.Fatalf("get small attention KV scratch again: %v", err)
	}
	defer putAttentionBlockKVScratch(gotSmall)
	if gotSmall != small {
		t.Fatal("attention KV scratch pool evicted the small scratch after using a larger scratch")
	}

	gotLarge, err := getAttentionBlockKVScratch(256, 256)
	if err != nil {
		t.Fatalf("get large attention KV scratch again: %v", err)
	}
	defer putAttentionBlockKVScratch(gotLarge)
	if gotLarge != large {
		t.Fatal("attention KV scratch pool evicted the large scratch after reusing the small scratch")
	}
}

func TestAttentionBlockKVScratchUsesCallerCacheBacking(t *testing.T) {
	requireNativeRuntime(t)

	const dModel, nHeads, nKV, headDim, kvLen = 64, 1, 1, 64, 3
	const base, scale, offset, eps = float32(10000), float32(0.125), 1, float32(1e-5)
	layer := decodeLayerFixture(dModel, nHeads, nKV, headDim, 128, 3)
	x := toBF16Bytes(syntheticFloat32(dModel, 5))
	kCache := toBF16Bytes(syntheticFloat32(nKV*kvLen*headDim, 7))
	vCache := toBF16Bytes(syntheticFloat32(nKV*kvLen*headDim, 11))
	scratch, err := getAttentionBlockKVScratch(len(kCache), len(vCache))
	if err != nil {
		t.Fatalf("get attention KV scratch: %v", err)
	}
	scratch.closeCacheViews()
	putAttentionBlockKVScratch(scratch)

	if _, err := AttentionBlock(x, layer.AttnNormW, layer.WQ, layer.WO, kCache, vCache, dModel, nHeads, nKV, headDim, kvLen, base, scale, offset, eps); err != nil {
		t.Fatalf("AttentionBlock: %v", err)
	}

	gotScratch, err := getAttentionBlockKVScratch(len(kCache), len(vCache))
	if err != nil {
		t.Fatalf("get attention KV scratch after call: %v", err)
	}
	defer putAttentionBlockKVScratch(gotScratch)
	if gotScratch != scratch {
		t.Fatal("AttentionBlock did not reuse the prepared KV scratch")
	}
	if gotScratch.kViewPtr != uintptr(unsafe.Pointer(&kCache[0])) || gotScratch.vViewPtr != uintptr(unsafe.Pointer(&vCache[0])) {
		t.Fatal("AttentionBlock copied KV cache bytes instead of retaining no-copy cache views")
	}
	if gotScratch.kViewPinned == nil || gotScratch.vViewPinned == nil {
		t.Fatal("AttentionBlock did not keep pinned KV cache lifetimes on the scratch")
	}
}

func TestAttentionBlockKVScratchReusesPinnedOwnerCacheBuffers(t *testing.T) {
	requireNativeRuntime(t)

	const nKV, headDim, kvLen = 1, 64, 3
	cacheBytes := nKV * kvLen * headDim * bf16Size
	kPinned, err := newPinnedNoCopyBytes(cacheBytes)
	if err != nil {
		t.Fatalf("newPinnedNoCopyBytes(k): %v", err)
	}
	vPinned, err := newPinnedNoCopyBytes(cacheBytes)
	if err != nil {
		kPinned.Close()
		t.Fatalf("newPinnedNoCopyBytes(v): %v", err)
	}
	t.Cleanup(func() {
		kPinned.Close()
		vPinned.Close()
	})

	scratch, err := getAttentionBlockKVScratch(len(kPinned.bytes), len(vPinned.bytes))
	if err != nil {
		t.Fatalf("get attention KV scratch: %v", err)
	}
	scratch.closeCacheViews()
	t.Cleanup(func() {
		scratch.closeCacheViews()
		putAttentionBlockKVScratch(scratch)
	})

	kBuf, vBuf, ok, err := scratch.buffersNoCopy(kPinned.bytes, vPinned.bytes)
	if err != nil {
		t.Fatalf("buffersNoCopy: %v", err)
	}
	if !ok {
		t.Fatal("buffersNoCopy did not create no-copy KV cache views")
	}
	requirePinnedOwnerBuffer(t, "attention K cache view", kBuf, kPinned)
	requirePinnedOwnerBuffer(t, "attention V cache view", vBuf, vPinned)
}

func TestAttentionBlockAllocationBudget(t *testing.T) {
	requireNativeRuntime(t)

	const dModel, nHeads, nKV, headDim, kvLen = 64, 1, 1, 64, 4
	const base, scale, offset, eps = float32(10000), float32(0.125), 1, float32(1e-5)
	layer := decodeLayerFixture(dModel, nHeads, nKV, headDim, 128, 3)
	x := toBF16Bytes(syntheticFloat32(dModel, 5))
	kCache := toBF16Bytes(syntheticFloat32(nKV*kvLen*headDim, 7))
	vCache := toBF16Bytes(syntheticFloat32(nKV*kvLen*headDim, 11))
	if _, err := AttentionBlock(x, layer.AttnNormW, layer.WQ, layer.WO, kCache, vCache, dModel, nHeads, nKV, headDim, kvLen, base, scale, offset, eps); err != nil {
		t.Fatalf("AttentionBlock warmup: %v", err)
	}

	var blockErr error
	allocs := testing.AllocsPerRun(5, func() {
		_, blockErr = AttentionBlock(x, layer.AttnNormW, layer.WQ, layer.WO, kCache, vCache, dModel, nHeads, nKV, headDim, kvLen, base, scale, offset, eps)
	})
	if blockErr != nil {
		t.Fatalf("AttentionBlock: %v", blockErr)
	}
	if allocs > 10 {
		t.Fatalf("AttentionBlock allocations = %.0f, want <= 10", allocs)
	}
}
