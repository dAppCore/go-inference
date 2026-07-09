// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"bytes"
	"sync"
	"testing"
	"unsafe"
)

func TestDecodeForwardICBMatchesReencode(t *testing.T) {
	requireNativeRuntime(t)

	const dModel, nHeads, nKV, headDim, dFF, maxLen = 64, 1, 1, 64, 128, 4
	const base, scale, eps = float32(10000), float32(0.125), float32(1e-5)
	inputs := decodeInputsFixture(2, dModel)
	layers := []DecodeLayerWeights{decodeLayerFixture(dModel, nHeads, nKV, headDim, dFF, 3)}

	want, err := DecodeForward(inputs, layers, dModel, nHeads, nKV, headDim, maxLen, dFF, base, scale, eps)
	if err != nil {
		t.Fatalf("DecodeForward: %v", err)
	}
	got, err := DecodeForwardICB(inputs, layers, dModel, nHeads, nKV, headDim, maxLen, dFF, base, scale, eps)
	if err != nil {
		t.Fatalf("DecodeForwardICB: %v", err)
	}
	for i := range want {
		eqBytes(t, "DecodeForwardICB token", got[i], want[i])
	}
}

func TestDecodeForwardICBIntoReusesOutputBacking(t *testing.T) {
	requireNativeRuntime(t)

	const dModel, nHeads, nKV, headDim, dFF, maxLen = 64, 1, 1, 64, 128, 4
	const base, scale, eps = float32(10000), float32(0.125), float32(1e-5)
	inputs := decodeInputsFixture(2, dModel)
	layers := []DecodeLayerWeights{decodeLayerFixture(dModel, nHeads, nKV, headDim, dFF, 3)}
	want, err := DecodeForwardICB(inputs, layers, dModel, nHeads, nKV, headDim, maxLen, dFF, base, scale, eps)
	if err != nil {
		t.Fatalf("DecodeForwardICB reference: %v", err)
	}
	out := [][]byte{
		bytes.Repeat([]byte{0xa5}, dModel*bf16Size),
		bytes.Repeat([]byte{0x5a}, dModel*bf16Size),
	}
	ptrs := []unsafe.Pointer{unsafe.Pointer(&out[0][0]), unsafe.Pointer(&out[1][0])}

	got, err := DecodeForwardICBInto(out, inputs, layers, dModel, nHeads, nKV, headDim, maxLen, dFF, base, scale, eps)
	if err != nil {
		t.Fatalf("DecodeForwardICBInto: %v", err)
	}
	if len(got) != len(want) {
		t.Fatalf("DecodeForwardICBInto returned %d outputs, want %d", len(got), len(want))
	}
	for tok := range want {
		if len(got[tok]) != dModel*bf16Size || unsafe.Pointer(&got[tok][0]) != ptrs[tok] {
			t.Fatalf("DecodeForwardICBInto token %d did not reuse caller-owned output backing", tok)
		}
		eqBytes(t, "DecodeForwardICBInto token", got[tok], want[tok])
	}
}

func TestDecodeForwardICBCoreScratchOutputViewsUseCallerBacking(t *testing.T) {
	requireNativeRuntime(t)

	const dModel, nHeads, nKV, headDim, dFF, nLayers = 64, 1, 1, 64, 128, 1
	sc := newDecodeForwardICBCoreScratch(dModel, nHeads*headDim, nKV*headDim, dFF, nLayers)
	t.Cleanup(sc.closeOutputViews)

	out := [][]byte{
		bytes.Repeat([]byte{0xa5}, dModel*bf16Size),
		bytes.Repeat([]byte{0x5a}, dModel*bf16Size),
	}
	views, ok := sc.outputViews(out, dModel*bf16Size)
	if !ok {
		t.Fatal("outputViews did not create no-copy views for caller-owned outputs")
	}
	for i := range out {
		if views[i] == nil || views[i].Contents() != unsafe.Pointer(&out[i][0]) {
			t.Fatalf("output view %d not backed by caller output slice", i)
		}
	}
	firstID := views[0].GetID()
	reused, ok := sc.outputViews(out, dModel*bf16Size)
	if !ok {
		t.Fatal("outputViews did not reuse no-copy views for unchanged caller outputs")
	}
	if reused[0].GetID() != firstID {
		t.Fatal("outputViews rebuilt an unchanged caller output view")
	}
}

func TestDecodeForwardICBCoreScratchOutputViewsReusePinnedOwnerBuffers(t *testing.T) {
	requireNativeRuntime(t)

	const dModel, nHeads, nKV, headDim, dFF, nLayers = 64, 1, 1, 64, 128, 1
	pinned := make([]*pinnedNoCopyBytes, 2)
	t.Cleanup(func() {
		for _, p := range pinned {
			if p != nil {
				p.Close()
			}
		}
	})
	sc := newDecodeForwardICBCoreScratch(dModel, nHeads*headDim, nKV*headDim, dFF, nLayers)
	t.Cleanup(sc.closeOutputViews)

	outputs := make([][]byte, len(pinned))
	for i := range pinned {
		var err error
		pinned[i], err = newPinnedNoCopyBytes(dModel * bf16Size)
		if err != nil {
			t.Fatalf("newPinnedNoCopyBytes(%d): %v", i, err)
		}
		outputs[i] = pinned[i].bytes
	}
	views, ok := sc.outputViews(outputs, dModel*bf16Size)
	if !ok {
		t.Fatal("outputViews did not create no-copy views for pinned-owner outputs")
	}
	for i := range pinned {
		requirePinnedOwnerBuffer(t, "decode ICB output view", views[i], pinned[i])
	}
}

func TestDecodeForwardICBAllocationBudget(t *testing.T) {
	requireNativeRuntime(t)

	const dModel, nHeads, nKV, headDim, dFF, maxLen = 64, 1, 1, 64, 128, 4
	const base, scale, eps = float32(10000), float32(0.125), float32(1e-5)
	inputs := decodeInputsFixture(2, dModel)
	layers := []DecodeLayerWeights{decodeLayerFixture(dModel, nHeads, nKV, headDim, dFF, 3)}
	if _, err := DecodeForwardICB(inputs, layers, dModel, nHeads, nKV, headDim, maxLen, dFF, base, scale, eps); err != nil {
		t.Fatalf("DecodeForwardICB warmup: %v", err)
	}

	var forwardErr error
	allocs := testing.AllocsPerRun(5, func() {
		_, forwardErr = DecodeForwardICB(inputs, layers, dModel, nHeads, nKV, headDim, maxLen, dFF, base, scale, eps)
	})
	if forwardErr != nil {
		t.Fatalf("DecodeForwardICB: %v", forwardErr)
	}
	if allocs > 235 {
		t.Fatalf("DecodeForwardICB allocations = %.0f, want <= 235", allocs)
	}
}

func TestDecodeForwardICBCoreScratchPoolKeepsShapesResident(t *testing.T) {
	decodeForwardICBCoreScratchPools = sync.Map{}
	t.Cleanup(func() { decodeForwardICBCoreScratchPools = sync.Map{} })

	small := &decodeForwardICBCoreScratch{dModel: 64, qDim: 64, kvDim: 64, dFF: 128, nLayers: 1}
	large := &decodeForwardICBCoreScratch{dModel: 128, qDim: 128, kvDim: 64, dFF: 256, nLayers: 2}
	smallPool := decodeForwardICBCoreScratchPoolFor(small.dModel, small.qDim, small.kvDim, small.dFF, small.nLayers)
	largePool := decodeForwardICBCoreScratchPoolFor(large.dModel, large.qDim, large.kvDim, large.dFF, large.nLayers)
	if smallPool == largePool {
		t.Fatal("DecodeForward ICB core scratch reused one pool for distinct core shapes")
	}

	putDecodeForwardICBCoreScratch(small)
	putDecodeForwardICBCoreScratch(large)
	forceNativeGC()
	forceNativeGC()

	if got := smallPool.Get(); got != small {
		t.Fatal("DecodeForward ICB core scratch pool evicted the small shape after using the larger shape")
	}
	if got := largePool.Get(); got != large {
		t.Fatal("DecodeForward ICB core scratch pool evicted the larger shape after reusing the small shape")
	}
}
