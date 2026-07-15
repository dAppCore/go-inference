// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"bytes"
	"runtime"
	"testing"
	"unsafe"

	"dappco.re/go/inference/engine/scheme"
)

// TestAttention_AttentionBlock_Good proves the fused on-device attention block (rmsnorm ->
// wQ -> rope -> sdpa -> wO -> residual) equals the same maths run as separate proven primitives.
func TestAttention_AttentionBlock_Good(t *testing.T) {
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

// TestAttention_AttentionBlock_Bad exercises every dimension guard attentionBlockInto validates
// before it touches the GPU.
func TestAttention_AttentionBlock_Bad(t *testing.T) {
	requireNativeRuntime(t)

	const dModel, nHeads, nKV, headDim, kvLen = 64, 1, 1, 64, 2
	const base, scale, offset, eps = float32(10000), float32(0.125), 1, float32(1e-5)
	qDim := nHeads * headDim
	validX := toBF16Bytes(syntheticFloat32(dModel, 3))
	validNormW := toBF16Bytes(syntheticFloat32(dModel, 5))
	validWQ := toBF16Bytes(syntheticFloat32(qDim*dModel, 7))
	validWO := toBF16Bytes(syntheticFloat32(dModel*qDim, 11))
	validK := toBF16Bytes(syntheticFloat32(nKV*kvLen*headDim, 13))
	validV := toBF16Bytes(syntheticFloat32(nKV*kvLen*headDim, 17))

	cases := []struct {
		name                   string
		x, normW, wQ, wO, k, v []byte
	}{
		{"x length mismatch", validX[:len(validX)-2], validNormW, validWQ, validWO, validK, validV},
		{"normWeight length mismatch", validX, validNormW[:len(validNormW)-2], validWQ, validWO, validK, validV},
		{"wQ length mismatch", validX, validNormW, validWQ[:len(validWQ)-2], validWO, validK, validV},
		{"wO length mismatch", validX, validNormW, validWQ, validWO[:len(validWO)-2], validK, validV},
		{"kCache length mismatch", validX, validNormW, validWQ, validWO, validK[:len(validK)-2], validV},
		{"vCache length mismatch", validX, validNormW, validWQ, validWO, validK, validV[:len(validV)-2]},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if _, err := AttentionBlock(c.x, c.normW, c.wQ, c.wO, c.k, c.v, dModel, nHeads, nKV, headDim, kvLen, base, scale, offset, eps); err == nil {
				t.Fatalf("AttentionBlock(%s): expected an error, got none", c.name)
			}
		})
	}
}

// TestAttention_AttentionBlock_Ugly extends the composed-primitives proof past the trivial
// nHeads==nKVHeads==1 base case to real GQA (nHeads>nKVHeads), a longer cache, and a nonzero
// offset — the shape the decode path actually runs.
func TestAttention_AttentionBlock_Ugly(t *testing.T) {
	requireNativeRuntime(t)

	const dModel, nHeads, nKV, headDim, kvLen = 64, 4, 2, 64, 5
	const base, scale, offset, eps = float32(10000), float32(0.125), 3, float32(1e-5)
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
		t.Fatalf("AttentionBlock (GQA) = %v, want composed primitives %v", bf16Floats(got), bf16Floats(want))
	}
}

// TestAttention_AttentionBlockInto_Good proves AttentionBlockInto reuses caller-owned output
// backing and bypasses the pooled scratch output.
func TestAttention_AttentionBlockInto_Good(t *testing.T) {
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

// TestAttention_AttentionBlockInto_Bad mirrors AttentionBlock's dimension guards through the
// caller-output entry point.
func TestAttention_AttentionBlockInto_Bad(t *testing.T) {
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
	out := make([]byte, dModel*bf16Size)

	if _, err := AttentionBlockInto(out, x[:len(x)-2], normW, wQ, wO, kCache, vCache, dModel, nHeads, nKV, headDim, kvLen, base, scale, offset, eps); err == nil {
		t.Fatal("expected AttentionBlockInto to reject an x length mismatch")
	}
	if _, err := AttentionBlockInto(out, x, normW, wQ, wO, kCache[:len(kCache)-2], vCache, dModel, nHeads, nKV, headDim, kvLen, base, scale, offset, eps); err == nil {
		t.Fatal("expected AttentionBlockInto to reject a kCache length mismatch")
	}
}

// TestAttention_AttentionBlockInto_Ugly proves the too-small-capacity path: when cap(out) is
// smaller than dModel*2 bytes, AttentionBlockInto must allocate fresh storage rather than write
// out of bounds, and still return the correct residual output.
func TestAttention_AttentionBlockInto_Ugly(t *testing.T) {
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

	tooSmall := make([]byte, 1)
	got, err := AttentionBlockInto(tooSmall, x, layer.AttnNormW, layer.WQ, layer.WO, kCache, vCache, dModel, nHeads, nKV, headDim, kvLen, base, scale, offset, eps)
	if err != nil {
		t.Fatalf("AttentionBlockInto (undersized out): %v", err)
	}
	eqBytes(t, "AttentionBlockInto (undersized out)", got, want)
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

func TestAttentionBlockKVScratch_Close_Good(t *testing.T) {
	// Close is deliberately safe for a partially built scratch: cleanup may run
	// after its second pinned allocation fails.
	s := &attentionBlockKVScratch{
		kBytes: 8, vBytes: 12,
		k:        &pinnedNoCopyBytes{bytes: []byte{1}},
		v:        &pinnedNoCopyBytes{bytes: []byte{2}},
		kViewPtr: 1, kViewLen: 2, kViewPinned: &pinnedNoCopyBytes{bytes: []byte{3}},
		vViewPtr: 4, vViewLen: 5, vViewPinned: &pinnedNoCopyBytes{bytes: []byte{6}},
	}
	s.Close()
	if s.k != nil || s.v != nil || s.kBytes != 0 || s.vBytes != 0 {
		t.Fatalf("Close left KV ownership behind: %#v", s)
	}
	if s.kView != nil || s.vView != nil || s.kViewPinned != nil || s.vViewPinned != nil || s.kViewPtr != 0 || s.vViewPtr != 0 || s.kViewLen != 0 || s.vViewLen != 0 {
		t.Fatalf("Close left cache views behind: %#v", s)
	}
	var nilScratch *attentionBlockKVScratch
	nilScratch.Close()
}

func TestTemporaryPinnedNoCopyBytes_Good(t *testing.T) {
	requireNativeRuntime(t)

	src := []byte{3, 1, 4, 1}
	var pinner runtime.Pinner
	buf, err := temporaryPinnedNoCopyBytes(src, &pinner)
	if err != nil {
		t.Fatalf("temporaryPinnedNoCopyBytes: %v", err)
	}
	defer func() {
		pinner.Unpin()
		runtime.KeepAlive(src)
	}()
	if buf == nil || buf.GetID() == 0 || unsafe.Pointer(buf.Contents()) != unsafe.Pointer(&src[0]) {
		t.Fatal("temporaryPinnedNoCopyBytes did not return a live view of the caller bytes")
	}
}

func TestTemporaryPinnedNoCopyBytes_Bad(t *testing.T) {
	var pinner runtime.Pinner
	if _, err := temporaryPinnedNoCopyBytes(nil, &pinner); err == nil {
		t.Fatal("temporaryPinnedNoCopyBytes accepted an empty slice")
	}
}

func TestPinnedNoCopyBytes_CopyPrefixBuffer_Good(t *testing.T) {
	requireNativeRuntime(t)

	p, err := newPinnedNoCopyBytes(6)
	if err != nil {
		t.Fatalf("newPinnedNoCopyBytes: %v", err)
	}
	defer p.Close()
	buf, err := p.copyPrefixBuffer([]byte{9, 8, 7})
	if err != nil {
		t.Fatalf("copyPrefixBuffer: %v", err)
	}
	if buf != p.buf {
		t.Fatal("copyPrefixBuffer returned a buffer other than the pinned backing")
	}
	eqBytes(t, "copyPrefixBuffer backing", p.bytes, []byte{9, 8, 7, 0, 0, 0})
}

func TestPinnedNoCopyBytes_CopyPrefixBuffer_Bad(t *testing.T) {
	var nilPinned *pinnedNoCopyBytes
	if _, err := nilPinned.copyPrefixBuffer([]byte{1}); err == nil {
		t.Fatal("copyPrefixBuffer accepted nil pinned storage")
	}
	requireNativeRuntime(t)
	p, err := newPinnedNoCopyBytes(2)
	if err != nil {
		t.Fatalf("newPinnedNoCopyBytes: %v", err)
	}
	defer p.Close()
	if _, err := p.copyPrefixBuffer([]byte{1, 2, 3}); err == nil {
		t.Fatal("copyPrefixBuffer accepted a source larger than its backing")
	}
}

func TestSharedOrNil_Good(t *testing.T) {
	requireNativeRuntime(t)

	if got := sharedOrNil(nil); got != nil {
		t.Fatal("sharedOrNil(nil) returned a buffer")
	}
	src := []byte{7, 11, 13}
	buf := sharedOrNil(src)
	if buf == nil || buf.GetID() == 0 {
		t.Fatal("sharedOrNil(non-empty) returned no buffer")
	}
	eqBytes(t, "sharedOrNil contents", unsafe.Slice((*byte)(buf.Contents()), len(src)), src)
}

func TestEncBinaryDT_Good(t *testing.T) {
	requireNativeRuntime(t)

	a := sharedBytes(toBF16Bytes([]float32{1, 2}))
	b := sharedBytes(toBF16Bytes([]float32{3, 4}))
	out := scratchBF16(2)
	cb := commandBufferFast(queue)
	enc := computeCommandEncoderFast(cb)
	if err := encBinaryDT(enc, "Add", scheme.BFloat16, a, b, out, 2); err != nil {
		endEncodingFast(enc)
		t.Fatalf("encBinaryDT: %v", err)
	}
	endEncodingFast(enc)
	commitCommandBufferFast(cb)
	waitUntilCompletedFast(cb)
	eqBytes(t, "encBinaryDT Add", unsafe.Slice((*byte)(out.Contents()), 2*bf16Size), toBF16Bytes([]float32{4, 6}))
}

func TestEncBinaryDTTo_Good(t *testing.T) {
	requireNativeRuntime(t)

	a := sharedBytes(toBF16Bytes([]float32{99, 1, 2}))
	b := sharedBytes(toBF16Bytes([]float32{88, 3, 4}))
	out := scratchBF16(3)
	cb := commandBufferFast(queue)
	enc := computeCommandEncoderFast(cb)
	if err := encBinaryDTTo(enc, "Add", scheme.BFloat16, a, b, out, bf16Size, bf16Size, bf16Size, 2); err != nil {
		endEncodingFast(enc)
		t.Fatalf("encBinaryDTTo: %v", err)
	}
	endEncodingFast(enc)
	commitCommandBufferFast(cb)
	waitUntilCompletedFast(cb)
	// The encoder promises only the requested output range; no ordering or
	// contents outside [outOff, outOff+n) are implied by the implementation.
	got := unsafe.Slice((*byte)(unsafe.Add(out.Contents(), bf16Size)), 2*bf16Size)
	eqBytes(t, "encBinaryDTTo output range", got, toBF16Bytes([]float32{4, 6}))
}

func TestEncUnaryDTObject_Good(t *testing.T) {
	requireNativeRuntime(t)

	in := sharedBytes(toBF16Bytes([]float32{0, 0}))
	out := scratchBF16(2)
	cb := commandBufferFast(queue)
	enc := computeCommandEncoderFast(cb)
	if err := encUnaryDTObject(enc, "Tanh", scheme.BFloat16, in, out, 2); err != nil {
		endEncodingFast(enc)
		t.Fatalf("encUnaryDTObject: %v", err)
	}
	endEncodingFast(enc)
	commitCommandBufferFast(cb)
	waitUntilCompletedFast(cb)
	eqBytes(t, "encUnaryDTObject Tanh", unsafe.Slice((*byte)(out.Contents()), 2*bf16Size), toBF16Bytes([]float32{0, 0}))
}

func TestEncSDPADecodeAt_Good(t *testing.T) {
	requireNativeRuntime(t)

	const headDim = 64
	rowBytes := headDim * bf16Size
	q := sharedBytes(make([]byte, 2*rowBytes))
	k := sharedBytes(make([]byte, rowBytes))
	want := toBF16Bytes(syntheticFloat32(headDim, 23))
	v := sharedBytes(want)
	out := scratchBF16(2 * headDim)
	cb := commandBufferFast(queue)
	enc := computeCommandEncoderFast(cb)
	if err := encSDPADecodeAt(enc, attnScratch{}, q, uint(rowBytes), k, v, out, uint(rowBytes), 1, 1, headDim, 1, headDim, headDim, headDim, headDim, 1, 0); err != nil {
		endEncodingFast(enc)
		t.Fatalf("encSDPADecodeAt: %v", err)
	}
	endEncodingFast(enc)
	commitCommandBufferFast(cb)
	waitUntilCompletedFast(cb)
	// With one key/value row, softmax has exactly one member, so the output is
	// that value row irrespective of the zero query or scale.
	got := unsafe.Slice((*byte)(unsafe.Add(out.Contents(), rowBytes)), rowBytes)
	eqBytes(t, "encSDPADecodeAt output range", got, want)
}

func TestEncSDPADecodeAt_Ugly(t *testing.T) {
	requireNativeRuntime(t)

	const headDim = 64
	const n = sdpa2PassMinKV
	rowBytes := headDim * bf16Size
	q := sharedBytes(make([]byte, rowBytes))
	k := sharedBytes(make([]byte, n*rowBytes))
	want := toBF16Bytes(syntheticFloat32(headDim, 29))
	vBytes := make([]byte, n*rowBytes)
	for row := range n {
		copy(vBytes[row*rowBytes:(row+1)*rowBytes], want)
	}
	v := sharedBytes(vBytes)
	out := scratchBF16(headDim)
	// The implementation selects the two-pass route only when both the length
	// reaches its knee and this scratch carries all three intermediates.
	sc := newAttnScratch(headDim, headDim, headDim, 1, n)
	if sc.p2Partials == nil || sc.p2Sums == nil || sc.p2Maxs == nil {
		t.Fatal("newAttnScratch did not provision two-pass intermediates at the SDPA knee")
	}
	cb := commandBufferFast(queue)
	enc := computeCommandEncoderFast(cb)
	if err := encSDPADecodeAt(enc, sc, q, 0, k, v, out, 0, 1, 1, headDim, n, headDim, headDim, headDim, headDim, 1, 0); err != nil {
		endEncodingFast(enc)
		t.Fatalf("encSDPADecodeAt two-pass: %v", err)
	}
	endEncodingFast(enc)
	commitCommandBufferFast(cb)
	waitUntilCompletedFast(cb)
	// Every value row is identical, so the softmax weighting and its reduction
	// order cannot change the expected output.
	eqBytes(t, "encSDPADecodeAt two-pass output", unsafe.Slice((*byte)(out.Contents()), rowBytes), want)
}

func TestEncSDPADecodeAt_Bad(t *testing.T) {
	requireNativeRuntime(t)

	// An unsupported head dimension declines while selecting the pipeline, before
	// it can dereference the deliberately absent encoder and buffers.
	if err := encSDPADecodeAt(nil, attnScratch{}, nil, 0, nil, nil, nil, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0); err == nil {
		t.Fatal("encSDPADecodeAt accepted an unsupported head dimension")
	}
}
