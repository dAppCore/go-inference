// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"sync"
	"testing"
	"unsafe"

	"github.com/tmc/apple/metal"
)

func TestNormProjectICBMatchesReencode(t *testing.T) {
	requireNativeRuntime(t)

	x := syntheticFloat32(64, 3)
	normW := syntheticFloat32(64, 5)
	projW := syntheticFloat32(128*64, 7)
	want, err := NormProject(x, normW, projW, 64, 128, 1e-5)
	if err != nil {
		t.Fatalf("NormProject: %v", err)
	}
	got, err := NormProjectICB(x, normW, projW, 64, 128, 1e-5, 1)
	if err != nil {
		t.Fatalf("NormProjectICB: %v", err)
	}
	assertFloat32Near(t, "NormProjectICB", got, want, 0)
}

func TestNormProjectICBAllocationBudget(t *testing.T) {
	requireNativeRuntime(t)

	const dIn, dOut = 128, 256
	x := syntheticFloat32(dIn, 3)
	normW := syntheticFloat32(dIn, 5)
	projW := syntheticFloat32(dOut*dIn, 7)
	if _, err := NormProjectICB(x, normW, projW, dIn, dOut, 1e-5, 1); err != nil {
		t.Fatalf("NormProjectICB warmup: %v", err)
	}

	allocs := testing.AllocsPerRun(5, func() {
		if _, err := NormProjectICB(x, normW, projW, dIn, dOut, 1e-5, 1); err != nil {
			t.Fatalf("NormProjectICB: %v", err)
		}
	})
	if allocs > 134 {
		t.Fatalf("NormProjectICB allocations = %.0f, want <= 134", allocs)
	}
}

func TestAttentionBlockICBMatchesReencode(t *testing.T) {
	requireNativeRuntime(t)

	const dModel, nHeads, nKV, headDim, kvLen = 64, 1, 1, 64, 2
	const base, scale, offset, eps = float32(10000), float32(0.125), 1, float32(1e-5)
	layer := decodeLayerFixture(dModel, nHeads, nKV, headDim, 128, 3)
	x := toBF16Bytes(syntheticFloat32(dModel, 5))
	kCache := toBF16Bytes(syntheticFloat32(nKV*kvLen*headDim, 7))
	vCache := toBF16Bytes(syntheticFloat32(nKV*kvLen*headDim, 11))
	want, err := AttentionBlock(x, layer.AttnNormW, layer.WQ, layer.WO, kCache, vCache, dModel, nHeads, nKV, headDim, kvLen, base, scale, offset, eps)
	if err != nil {
		t.Fatalf("AttentionBlock: %v", err)
	}
	got, err := AttentionBlockICB(x, layer.AttnNormW, layer.WQ, layer.WO, kCache, vCache, dModel, nHeads, nKV, headDim, kvLen, base, scale, offset, eps, 1)
	if err != nil {
		t.Fatalf("AttentionBlockICB: %v", err)
	}
	eqBytes(t, "AttentionBlockICB", got, want)
}

func TestAttentionBlockICBRebindsCallerBuffers(t *testing.T) {
	requireNativeRuntime(t)

	const dModel, nHeads, nKV, headDim, kvLen = 64, 1, 1, 64, 2
	const base, scale, offset, eps = float32(10000), float32(0.125), 1, float32(1e-5)
	layer := decodeLayerFixture(dModel, nHeads, nKV, headDim, 128, 3)
	kCache := toBF16Bytes(syntheticFloat32(nKV*kvLen*headDim, 7))
	vCache := toBF16Bytes(syntheticFloat32(nKV*kvLen*headDim, 11))

	for _, tc := range []struct {
		name string
		x    []byte
	}{
		{name: "first", x: toBF16Bytes(syntheticFloat32(dModel, 5))},
		{name: "second", x: toBF16Bytes(syntheticFloat32(dModel, 17))},
	} {
		want, err := AttentionBlock(tc.x, layer.AttnNormW, layer.WQ, layer.WO, kCache, vCache, dModel, nHeads, nKV, headDim, kvLen, base, scale, offset, eps)
		if err != nil {
			t.Fatalf("%s AttentionBlock: %v", tc.name, err)
		}
		got, err := AttentionBlockICB(tc.x, layer.AttnNormW, layer.WQ, layer.WO, kCache, vCache, dModel, nHeads, nKV, headDim, kvLen, base, scale, offset, eps, 1)
		if err != nil {
			t.Fatalf("%s AttentionBlockICB: %v", tc.name, err)
		}
		eqBytes(t, tc.name+" AttentionBlockICB", got, want)
	}
}

func TestAttentionBlockICBKeepsFixedWeightsResident(t *testing.T) {
	requireNativeRuntime(t)

	resetResidentBufsForTest()
	defer resetResidentBufsForTest()

	const dModel, nHeads, nKV, headDim, kvLen = 64, 1, 1, 64, 2
	const base, scale, offset, eps = float32(10000), float32(0.125), 1, float32(1e-5)
	layer := decodeLayerFixture(dModel, nHeads, nKV, headDim, 128, 3)
	x := toBF16Bytes(syntheticFloat32(dModel, 5))
	kCache := toBF16Bytes(syntheticFloat32(nKV*kvLen*headDim, 7))
	vCache := toBF16Bytes(syntheticFloat32(nKV*kvLen*headDim, 11))

	if _, err := AttentionBlockICB(x, layer.AttnNormW, layer.WQ, layer.WO, kCache, vCache, dModel, nHeads, nKV, headDim, kvLen, base, scale, offset, eps, 1); err != nil {
		t.Fatalf("AttentionBlockICB: %v", err)
	}

	key := func(b []byte) uintptr { return uintptr(unsafe.Pointer(&b[0])) }
	residentBufMu.Lock()
	got := len(residentBufs)
	_, hasNorm := residentBufs[key(layer.AttnNormW)]
	_, hasQ := residentBufs[key(layer.WQ)]
	_, hasO := residentBufs[key(layer.WO)]
	residentBufMu.Unlock()

	if !hasNorm || !hasQ || !hasO {
		t.Fatalf("AttentionBlockICB did not keep fixed weights resident (norm=%v q=%v o=%v resident=%d want>=3)", hasNorm, hasQ, hasO, got)
	}
}

func TestAttentionBlockICBScratchPoolKeepsShapesResident(t *testing.T) {
	attentionBlockICBScratchPools = sync.Map{}
	t.Cleanup(func() { attentionBlockICBScratchPools = sync.Map{} })

	small := &attentionBlockICBScratch{dModel: 64, qDim: 64, nHeads: 1, nKVHeads: 1, headDim: 64, kvLen: 2}
	large := &attentionBlockICBScratch{dModel: 128, qDim: 128, nHeads: 2, nKVHeads: 1, headDim: 64, kvLen: 4}
	smallPool := attentionBlockICBScratchPoolFor(small.dModel, small.qDim, small.nHeads, small.nKVHeads, small.headDim, small.kvLen)
	largePool := attentionBlockICBScratchPoolFor(large.dModel, large.qDim, large.nHeads, large.nKVHeads, large.headDim, large.kvLen)
	if smallPool == largePool {
		t.Fatal("AttentionBlock ICB scratch reused one pool for distinct attention shapes")
	}

	putAttentionBlockICBScratch(small)
	putAttentionBlockICBScratch(large)
	forceNativeGC()
	forceNativeGC()

	if got := smallPool.Get(); got != small {
		t.Fatal("AttentionBlock ICB scratch pool evicted the small shape after using the larger shape")
	}
	if got := largePool.Get(); got != large {
		t.Fatal("AttentionBlock ICB scratch pool evicted the larger shape after reusing the small shape")
	}
}

func TestAttentionBlockICBScratchBuffersUseCallerBacking(t *testing.T) {
	requireNativeRuntime(t)

	const dModel, nHeads, nKV, headDim, kvLen = 64, 1, 1, 64, 4
	const base, scale, offset, eps = float32(10000), float32(0.125), 1, float32(1e-5)
	x := toBF16Bytes(syntheticFloat32(dModel, 5))
	kCache := toBF16Bytes(syntheticFloat32(nKV*kvLen*headDim, 7))
	vCache := toBF16Bytes(syntheticFloat32(nKV*kvLen*headDim, 11))
	scratch, err := getAttentionBlockICBScratch(dModel, nHeads*headDim, nHeads, nKV, headDim, kvLen, base, scale, offset, eps)
	if err != nil {
		t.Fatalf("get AttentionBlockICB scratch: %v", err)
	}
	defer putAttentionBlockICBScratch(scratch)
	xBuf, kBuf, vBuf, _, err := scratch.buffers(x, kCache, vCache)
	if err != nil {
		t.Fatalf("AttentionBlockICB scratch buffers: %v", err)
	}
	if got, want := uintptr(xBuf.Contents()), uintptr(unsafe.Pointer(&x[0])); got != want {
		t.Fatalf("x buffer pointer = %#x, want caller backing %#x", got, want)
	}
	if got, want := uintptr(kBuf.Contents()), uintptr(unsafe.Pointer(&kCache[0])); got != want {
		t.Fatalf("k buffer pointer = %#x, want caller backing %#x", got, want)
	}
	if got, want := uintptr(vBuf.Contents()), uintptr(unsafe.Pointer(&vCache[0])); got != want {
		t.Fatalf("v buffer pointer = %#x, want caller backing %#x", got, want)
	}
}

func TestAttentionBlockICBAllocationBudget(t *testing.T) {
	requireNativeRuntime(t)

	const dModel, nHeads, nKV, headDim, kvLen = 64, 1, 1, 64, 4
	const base, scale, offset, eps = float32(10000), float32(0.125), 1, float32(1e-5)
	layer := decodeLayerFixture(dModel, nHeads, nKV, headDim, 128, 3)
	x := toBF16Bytes(syntheticFloat32(dModel, 5))
	kCache := toBF16Bytes(syntheticFloat32(nKV*kvLen*headDim, 7))
	vCache := toBF16Bytes(syntheticFloat32(nKV*kvLen*headDim, 11))
	if _, err := AttentionBlockICB(x, layer.AttnNormW, layer.WQ, layer.WO, kCache, vCache, dModel, nHeads, nKV, headDim, kvLen, base, scale, offset, eps, 1); err != nil {
		t.Fatalf("AttentionBlockICB warmup: %v", err)
	}

	var blockErr error
	allocs := testing.AllocsPerRun(5, func() {
		_, blockErr = AttentionBlockICB(x, layer.AttnNormW, layer.WQ, layer.WO, kCache, vCache, dModel, nHeads, nKV, headDim, kvLen, base, scale, offset, eps, 1)
	})
	if blockErr != nil {
		t.Fatalf("AttentionBlockICB: %v", blockErr)
	}
	if allocs > 3 {
		t.Fatalf("AttentionBlockICB allocations = %.0f, want <= 3", allocs)
	}
}

func TestAttentionBlockICBReplayAllocationBudget(t *testing.T) {
	requireNativeRuntime(t)

	const dModel, nHeads, nKV, headDim, kvLen, replays = 64, 1, 1, 64, 4, 4
	const base, scale, offset, eps = float32(10000), float32(0.125), 1, float32(1e-5)
	layer := decodeLayerFixture(dModel, nHeads, nKV, headDim, 128, 3)
	x := toBF16Bytes(syntheticFloat32(dModel, 5))
	kCache := toBF16Bytes(syntheticFloat32(nKV*kvLen*headDim, 7))
	vCache := toBF16Bytes(syntheticFloat32(nKV*kvLen*headDim, 11))
	if _, err := AttentionBlockICB(x, layer.AttnNormW, layer.WQ, layer.WO, kCache, vCache, dModel, nHeads, nKV, headDim, kvLen, base, scale, offset, eps, replays); err != nil {
		t.Fatalf("AttentionBlockICB warmup: %v", err)
	}

	var blockErr error
	allocs := testing.AllocsPerRun(5, func() {
		_, blockErr = AttentionBlockICB(x, layer.AttnNormW, layer.WQ, layer.WO, kCache, vCache, dModel, nHeads, nKV, headDim, kvLen, base, scale, offset, eps, replays)
	})
	if blockErr != nil {
		t.Fatalf("AttentionBlockICB: %v", blockErr)
	}
	if allocs > 9 {
		t.Fatalf("AttentionBlockICB replay allocations = %.0f, want <= 9", allocs)
	}
}

func TestAttentionBlockICBRecordAllocationBudget(t *testing.T) {
	requireNativeRuntime(t)

	const dModel, nHeads, nKV, headDim, kvLen = 64, 1, 1, 64, 4
	const base, scale, offset, eps = float32(10000), float32(0.125), 1, float32(1e-5)
	layer := decodeLayerFixture(dModel, nHeads, nKV, headDim, 128, 3)
	qDim := nHeads * headDim
	rmsPSO, err := pipelineForICB("rmsbfloat16")
	if err != nil {
		t.Fatalf("rms pso: %v", err)
	}
	bmQ, bnQ, smQ, snQ, tmQ, tnQ := gemvTiles(dModel, qDim)
	gemvQPSO, err := pipelineForICB(gemvKernelName("bfloat16", bmQ, bnQ, smQ, snQ, tmQ, tnQ))
	if err != nil {
		t.Fatalf("q pso: %v", err)
	}
	bmO, bnO, smO, snO, tmO, tnO := gemvTiles(qDim, dModel)
	gemvOPSO, err := pipelineForICB(gemvKernelName("bfloat16", bmO, bnO, smO, snO, tmO, tnO))
	if err != nil {
		t.Fatalf("o pso: %v", err)
	}
	ropePSO, err := ropePipelineICB(false)
	if err != nil {
		t.Fatalf("rope pso: %v", err)
	}
	sdpaPSO, err := sdpaVectorPipelineICBForHeadDim(headDim)
	if err != nil {
		t.Fatalf("sdpa pso: %v", err)
	}
	addPSO, err := pipelineForICB("vv_Addbfloat16")
	if err != nil {
		t.Fatalf("add pso: %v", err)
	}

	sc, err := getAttentionBlockICBScratch(dModel, qDim, nHeads, nKV, headDim, kvLen, base, scale, offset, eps)
	if err != nil {
		t.Fatalf("scratch: %v", err)
	}
	defer putAttentionBlockICBScratch(sc)
	x0, err := newPinnedNoCopyBytes(dModel * bf16Size)
	if err != nil {
		t.Fatalf("x0: %v", err)
	}
	defer x0.Close()
	x1, err := newPinnedNoCopyBytes(dModel * bf16Size)
	if err != nil {
		t.Fatalf("x1: %v", err)
	}
	defer x1.Close()
	k, err := newPinnedNoCopyBytes(nKV * kvLen * headDim * bf16Size)
	if err != nil {
		t.Fatalf("k: %v", err)
	}
	defer k.Close()
	v, err := newPinnedNoCopyBytes(nKV * kvLen * headDim * bf16Size)
	if err != nil {
		t.Fatalf("v: %v", err)
	}
	defer v.Close()
	nwBuf := residentBytes(layer.AttnNormW)
	wqBuf, woBuf := residentBytes(layer.WQ), residentBytes(layer.WO)
	xBufs := []metal.MTLBuffer{x0.buf, x1.buf}
	idx := 0
	sc.record(rmsPSO, gemvQPSO, gemvOPSO, ropePSO, sdpaPSO, addPSO, xBufs[0], k.buf, v.buf, nwBuf, wqBuf, woBuf, bmQ, bnQ, smQ, tmQ, bmO, bnO, smO, tmO)
	sc.record(rmsPSO, gemvQPSO, gemvOPSO, ropePSO, sdpaPSO, addPSO, xBufs[1], k.buf, v.buf, nwBuf, wqBuf, woBuf, bmQ, bnQ, smQ, tmQ, bmO, bnO, smO, tmO)

	allocs := testing.AllocsPerRun(5, func() {
		idx ^= 1
		sc.record(rmsPSO, gemvQPSO, gemvOPSO, ropePSO, sdpaPSO, addPSO, xBufs[idx], k.buf, v.buf, nwBuf, wqBuf, woBuf, bmQ, bnQ, smQ, tmQ, bmO, bnO, smO, tmO)
	})
	if allocs > 10 {
		t.Fatalf("AttentionBlockICB record allocations = %.0f, want <= 10", allocs)
	}
}
