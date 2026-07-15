// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"bytes"
	"math"
	"testing"
	"unsafe"

	core "dappco.re/go"
	"github.com/tmc/apple/metal"
)

func perLayerProjUnbatchedRef(t testing.TB, projW, hidden, perLayer []byte, projScale float32, projNormW []byte, numLayers, pliDim, dModel int, eps float32) []byte {
	t.Helper()
	plDim := numLayers * pliDim
	must := func(b []byte, err error) []byte {
		if err != nil {
			t.Fatalf("perLayerProj unbatched op: %v", err)
		}
		return b
	}
	projected := must(MatVecBF16(projW, hidden, plDim, dModel))
	scaled := must(MulBF16(projected, bf16ConstBytes(plDim, projScale)))
	projNormed := must(RMSNormBF16(scaled, projNormW, numLayers, pliDim, eps))
	combined := must(AddBF16(projNormed, perLayer))
	return must(MulBF16(combined, bf16ConstBytes(plDim, gemma4PerLayerCombineScale)))
}

func TestPerLayerProjBatchedMatchesUnbatchedReference(t *testing.T) {
	requireNativeRuntime(t)
	const numLayers, pliDim, dModel = 2, 8, 16
	const eps = float32(1e-5)
	plDim := numLayers * pliDim
	projScale := float32(1 / math.Sqrt(float64(dModel)))
	hidden := toBF16Bytes(syntheticFloat32(dModel, 1))
	perLayer := toBF16Bytes(syntheticFloat32(plDim, 2))
	projW := toBF16Bytes(syntheticFloat32(plDim*dModel, 3))
	projNormW := toBF16Bytes(syntheticFloat32(pliDim, 4))

	got, err := perLayerProjBatched(copyView(projW), hidden, perLayer, projScale, projNormW, plDim, numLayers, pliDim, dModel, eps)
	if err != nil {
		t.Fatalf("perLayerProjBatched: %v", err)
	}
	want := perLayerProjUnbatchedRef(t, projW, hidden, perLayer, projScale, projNormW, numLayers, pliDim, dModel, eps)
	eqBytes(t, "perLayerProjBatched", got, want)
}

func TestPerLayerProjBatchedResidentBytesMatchesUnbatchedReference(t *testing.T) {
	requireNativeRuntime(t)
	resetResidentBufsForTest()
	defer resetResidentBufsForTest()

	const numLayers, pliDim, dModel = 2, 8, 16
	const eps = float32(1e-5)
	plDim := numLayers * pliDim
	projScale := float32(1 / math.Sqrt(float64(dModel)))
	hidden := toBF16Bytes(syntheticFloat32(dModel, 1))
	perLayer := toBF16Bytes(syntheticFloat32(plDim, 2))
	projW := toBF16Bytes(syntheticFloat32(plDim*dModel, 3))
	projNormW := toBF16Bytes(syntheticFloat32(pliDim, 4))

	got, err := perLayerProjBatched(bufView{buf: residentBytes(projW)}, hidden, perLayer, projScale, projNormW, plDim, numLayers, pliDim, dModel, eps)
	if err != nil {
		t.Fatalf("perLayerProjBatched residentBytes: %v", err)
	}
	want := perLayerProjUnbatchedRef(t, projW, hidden, perLayer, projScale, projNormW, numLayers, pliDim, dModel, eps)
	eqBytes(t, "perLayerProjBatched residentBytes", got, want)
}

func TestPerLayerProjBatchedResidentBytesScratchMatchesUnbatchedReference(t *testing.T) {
	requireNativeRuntime(t)
	resetResidentBufsForTest()
	defer resetResidentBufsForTest()

	const numLayers, pliDim, dModel = 2, 8, 16
	const eps = float32(1e-5)
	plDim := numLayers * pliDim
	projScale := float32(1 / math.Sqrt(float64(dModel)))
	hidden := toBF16Bytes(syntheticFloat32(dModel, 1))
	perLayer := toBF16Bytes(syntheticFloat32(plDim, 2))
	projW := toBF16Bytes(syntheticFloat32(plDim*dModel, 3))
	projNormW := toBF16Bytes(syntheticFloat32(pliDim, 4))
	scratch, err := newPLHostScratch(plDim, dModel, projScale)
	if err != nil {
		t.Fatalf("newPLHostScratch: %v", err)
	}
	defer scratch.Close()

	got, err := perLayerProjBatched(bufView{buf: residentBytes(projW)}, hidden, perLayer, projScale, projNormW, plDim, numLayers, pliDim, dModel, eps, scratch)
	if err != nil {
		t.Fatalf("perLayerProjBatched residentBytes scratch: %v", err)
	}
	want := perLayerProjUnbatchedRef(t, projW, hidden, perLayer, projScale, projNormW, numLayers, pliDim, dModel, eps)
	eqBytes(t, "perLayerProjBatched residentBytes scratch", got, want)
}

func TestPerLayerProjBatchedScratchMatchesDefault(t *testing.T) {
	requireNativeRuntime(t)
	const numLayers, pliDim, dModel = 2, 8, 16
	const eps = float32(1e-5)
	plDim := numLayers * pliDim
	projScale := float32(1 / math.Sqrt(float64(dModel)))
	projW := toBF16Bytes(syntheticFloat32(plDim*dModel, 3))
	projNormW := toBF16Bytes(syntheticFloat32(pliDim, 4))
	projView := copyView(projW)
	scratch, err := newPLHostScratch(plDim, dModel, projScale)
	if err != nil {
		t.Fatalf("newPLHostScratch: %v", err)
	}
	defer scratch.Close()

	for seed := 1; seed <= 2; seed++ {
		hidden := toBF16Bytes(syntheticFloat32(dModel, seed))
		perLayer := toBF16Bytes(syntheticFloat32(plDim, seed+10))
		want, err := perLayerProjBatched(projView, hidden, perLayer, projScale, projNormW, plDim, numLayers, pliDim, dModel, eps)
		if err != nil {
			t.Fatalf("perLayerProjBatched default seed %d: %v", seed, err)
		}
		got, err := perLayerProjBatched(projView, hidden, perLayer, projScale, projNormW, plDim, numLayers, pliDim, dModel, eps, scratch)
		if err != nil {
			t.Fatalf("perLayerProjBatched scratch seed %d: %v", seed, err)
		}
		eqBytes(t, core.Sprintf("perLayerProjBatched scratch seed %d", seed), got, want)
	}
}

func TestPerLayerProjBatchedScratchUsesCallerInputBacking(t *testing.T) {
	requireNativeRuntime(t)
	const numLayers, pliDim, dModel = 2, 8, 16
	const eps = float32(1e-5)
	plDim := numLayers * pliDim
	projScale := float32(1 / math.Sqrt(float64(dModel)))
	hidden := toBF16Bytes(syntheticFloat32(dModel, 1))
	perLayer := toBF16Bytes(syntheticFloat32(plDim, 2))
	projW := toBF16Bytes(syntheticFloat32(plDim*dModel, 3))
	projNormW := toBF16Bytes(syntheticFloat32(pliDim, 4))
	scratch, err := newPLHostScratch(plDim, dModel, projScale)
	if err != nil {
		t.Fatalf("newPLHostScratch: %v", err)
	}
	defer scratch.Close()
	hiddenSentinel := bytes.Repeat([]byte{0xa5}, len(scratch.hidden.bytes))
	perLayerSentinel := bytes.Repeat([]byte{0x5a}, len(scratch.perLayer.bytes))
	copy(scratch.hidden.bytes, hiddenSentinel)
	copy(scratch.perLayer.bytes, perLayerSentinel)

	got, err := perLayerProjBatched(bufView{buf: residentBytes(projW)}, hidden, perLayer, projScale, projNormW, plDim, numLayers, pliDim, dModel, eps, scratch)
	if err != nil {
		t.Fatalf("perLayerProjBatched scratch: %v", err)
	}
	want := perLayerProjUnbatchedRef(t, projW, hidden, perLayer, projScale, projNormW, numLayers, pliDim, dModel, eps)
	eqBytes(t, "perLayerProjBatched scratch", got, want)
	if !bytes.Equal(scratch.hidden.bytes, hiddenSentinel) {
		t.Fatal("perLayerProjBatched copied hidden bytes into pooled scratch instead of using caller backing")
	}
	if !bytes.Equal(scratch.perLayer.bytes, perLayerSentinel) {
		t.Fatal("perLayerProjBatched copied per-layer bytes into pooled scratch instead of using caller backing")
	}
}

func TestPerLayerProjBatchedScratchReusesOutputBacking(t *testing.T) {
	requireNativeRuntime(t)
	const numLayers, pliDim, dModel = 2, 8, 16
	const eps = float32(1e-5)
	plDim := numLayers * pliDim
	projScale := float32(1 / math.Sqrt(float64(dModel)))
	projW := toBF16Bytes(syntheticFloat32(plDim*dModel, 3))
	projNormW := toBF16Bytes(syntheticFloat32(pliDim, 4))
	projView := copyView(projW)
	scratch, err := newPLHostScratch(plDim, dModel, projScale)
	if err != nil {
		t.Fatalf("newPLHostScratch: %v", err)
	}
	defer scratch.Close()

	var firstPtr uintptr
	for seed := 1; seed <= 2; seed++ {
		hidden := toBF16Bytes(syntheticFloat32(dModel, seed))
		perLayer := toBF16Bytes(syntheticFloat32(plDim, seed+10))
		want, err := perLayerProjBatched(projView, hidden, perLayer, projScale, projNormW, plDim, numLayers, pliDim, dModel, eps)
		if err != nil {
			t.Fatalf("perLayerProjBatched default seed %d: %v", seed, err)
		}
		got, err := perLayerProjBatched(projView, hidden, perLayer, projScale, projNormW, plDim, numLayers, pliDim, dModel, eps, scratch)
		if err != nil {
			t.Fatalf("perLayerProjBatched scratch seed %d: %v", seed, err)
		}
		eqBytes(t, core.Sprintf("perLayerProjBatched scratch seed %d", seed), got, want)
		ptr := uintptr(unsafe.Pointer(&got[0]))
		if seed == 1 {
			firstPtr = ptr
			continue
		}
		if ptr != firstPtr {
			t.Fatalf("scratch output backing changed: got %#x, want %#x", ptr, firstPtr)
		}
	}
}

func TestPerLayerProjBatchedScratchWritesDirectlyToHostReadback(t *testing.T) {
	requireNativeRuntime(t)
	const numLayers, pliDim, dModel = 2, 8, 16
	const eps = float32(1e-5)
	plDim := numLayers * pliDim
	projScale := float32(1 / math.Sqrt(float64(dModel)))
	hidden := toBF16Bytes(syntheticFloat32(dModel, 21))
	perLayer := toBF16Bytes(syntheticFloat32(plDim, 22))
	projW := toBF16Bytes(syntheticFloat32(plDim*dModel, 23))
	projNormW := toBF16Bytes(syntheticFloat32(pliDim, 24))
	projView := copyView(projW)
	scratch, err := newPLHostScratch(plDim, dModel, projScale)
	if err != nil {
		t.Fatalf("newPLHostScratch: %v", err)
	}
	defer scratch.Close()

	outScratch := unsafe.Slice((*byte)(scratch.out.Contents()), plDim*bf16Size)
	sentinel := bytes.Repeat([]byte{0xa5}, len(outScratch))
	copy(outScratch, sentinel)

	got, err := perLayerProjBatched(projView, hidden, perLayer, projScale, projNormW, plDim, numLayers, pliDim, dModel, eps, scratch)
	if err != nil {
		t.Fatalf("perLayerProjBatched scratch: %v", err)
	}
	want := perLayerProjUnbatchedRef(t, projW, hidden, perLayer, projScale, projNormW, numLayers, pliDim, dModel, eps)
	eqBytes(t, "perLayerProjBatched scratch direct readback", got, want)
	if !bytes.Equal(outScratch, sentinel) {
		t.Fatal("perLayerProjBatched wrote through pooled scratch output instead of host readback backing")
	}
}

func TestPerLayerProjQuantBatchedScratchWritesDirectlyToHostReadback(t *testing.T) {
	requireNativeRuntime(t)
	const numLayers, pliDim, dModel = 2, 8, 32
	const groupSize, bits = 32, 4
	const eps = float32(1e-5)
	plDim := numLayers * pliDim
	projScale := float32(1 / math.Sqrt(float64(dModel)))
	hidden := toBF16Bytes(syntheticFloat32(dModel, 31))
	perLayer := toBF16Bytes(syntheticFloat32(plDim, 32))
	proj := quantWeightFixture(t, plDim, dModel, groupSize, bits, 33)
	projNormW := toBF16Bytes(syntheticFloat32(pliDim, 34))
	scratch, err := newPLHostScratch(plDim, dModel, projScale)
	if err != nil {
		t.Fatalf("newPLHostScratch: %v", err)
	}
	defer scratch.Close()

	outScratch := unsafe.Slice((*byte)(scratch.out.Contents()), plDim*bf16Size)
	sentinel := bytes.Repeat([]byte{0x5a}, len(outScratch))
	copy(outScratch, sentinel)

	got, err := perLayerProjQuantBatched(proj, hidden, perLayer, projScale, projNormW, plDim, numLayers, pliDim, dModel, groupSize, bits, eps, scratch)
	if err != nil {
		t.Fatalf("perLayerProjQuantBatched scratch: %v", err)
	}
	if len(got) != plDim*bf16Size {
		t.Fatalf("perLayerProjQuantBatched length = %d, want %d", len(got), plDim*bf16Size)
	}
	if !bytes.Equal(outScratch, sentinel) {
		t.Fatal("perLayerProjQuantBatched wrote through pooled scratch output instead of host readback backing")
	}
}

func TestPerLayerProjBatchedResidentBufferUsesNoCopyHiddenInput(t *testing.T) {
	requireNativeRuntime(t)
	const numLayers, pliDim, dModel = 2, 8, 16
	const eps = float32(1e-5)
	plDim := numLayers * pliDim
	projScale := float32(1 / math.Sqrt(float64(dModel)))
	hidden := toBF16Bytes(syntheticFloat32(dModel, 21))
	perLayer := toBF16Bytes(syntheticFloat32(plDim, 22))
	projW := toBF16Bytes(syntheticFloat32(plDim*dModel, 23))
	projNormW := toBF16Bytes(syntheticFloat32(pliDim, 24))
	projView := copyView(projW)
	scratch, err := newPLHostScratch(plDim, dModel, projScale)
	if err != nil {
		t.Fatalf("newPLHostScratch: %v", err)
	}
	defer scratch.Close()
	for i := range scratch.hidden.bytes {
		scratch.hidden.bytes[i] = 0xa5
	}
	wantHidden := append([]byte(nil), scratch.hidden.bytes...)

	want, err := perLayerProjBatched(projView, hidden, perLayer, projScale, projNormW, plDim, numLayers, pliDim, dModel, eps)
	if err != nil {
		t.Fatalf("perLayerProjBatched default: %v", err)
	}
	var got []byte
	err = withPinnedNoCopyBytes(hidden, func(hiddenBuf metal.MTLBuffer) error {
		buf, err := perLayerProjBatchedResidentBuffer(projView, hiddenBuf, perLayer, projScale, projNormW, plDim, numLayers, pliDim, dModel, eps, scratch)
		if err != nil {
			return err
		}
		got = append([]byte(nil), unsafe.Slice((*byte)(buf.Contents()), plDim*bf16Size)...)
		return nil
	})
	if err != nil {
		t.Fatalf("perLayerProjBatchedResidentBuffer: %v", err)
	}
	eqBytes(t, "perLayerProjBatchedResident", got, want)
	if string(scratch.hidden.bytes) != string(wantHidden) {
		t.Fatal("resident buffer path copied hidden into scratch backing; want existing hidden Metal buffer")
	}
}

func TestPerLayerProjBatchedUsesScalarScaleBuffers(t *testing.T) {
	requireNativeRuntime(t)
	const numLayers, pliDim, dModel = 2, 8, 16
	const eps = float32(1e-5)
	plDim := numLayers * pliDim
	projScale := float32(1 / math.Sqrt(float64(dModel)))
	hidden := toBF16Bytes(syntheticFloat32(dModel, 21))
	perLayer := toBF16Bytes(syntheticFloat32(plDim, 22))
	projW := toBF16Bytes(syntheticFloat32(plDim*dModel, 23))
	projNormW := toBF16Bytes(syntheticFloat32(pliDim, 24))

	projKey := bf16ConstKey{n: plDim, v: projScale}
	combineKey := bf16ConstKey{n: plDim, v: gemma4PerLayerCombineScale}
	bf16ConstMu.Lock()
	delete(bf16ConstCache, projKey)
	delete(bf16ConstCache, combineKey)
	bf16ConstMu.Unlock()

	if _, err := perLayerProjBatched(copyView(projW), hidden, perLayer, projScale, projNormW, plDim, numLayers, pliDim, dModel, eps); err != nil {
		t.Fatalf("perLayerProjBatched: %v", err)
	}

	bf16ConstMu.Lock()
	_, projectedScaleCached := bf16ConstCache[projKey]
	_, combineScaleCached := bf16ConstCache[combineKey]
	bf16ConstMu.Unlock()
	if projectedScaleCached || combineScaleCached {
		t.Fatalf("perLayerProjBatched materialized plDim-wide scale buffers (projected=%v combine=%v), want scalar-bound BF16 scales", projectedScaleCached, combineScaleCached)
	}
}

func TestPLHostScratchKeepsScalarBuffersResident(t *testing.T) {
	requireNativeRuntime(t)
	const numLayers, pliDim, dModel = 2, 8, 16
	plDim := numLayers * pliDim
	projScale := float32(1 / math.Sqrt(float64(dModel)))
	first, err := newPLHostScratch(plDim, dModel, projScale)
	if err != nil {
		t.Fatalf("newPLHostScratch first: %v", err)
	}
	defer first.Close()
	second, err := newPLHostScratch(plDim, dModel, projScale)
	if err != nil {
		t.Fatalf("newPLHostScratch second: %v", err)
	}
	defer second.Close()

	if first.projScaleBuf.GetID() != second.projScaleBuf.GetID() {
		t.Fatalf("projection scale buffer was not resident: first=%d second=%d", first.projScaleBuf.GetID(), second.projScaleBuf.GetID())
	}
	if first.combineScaleBuf.GetID() != second.combineScaleBuf.GetID() {
		t.Fatalf("combine scale buffer was not resident: first=%d second=%d", first.combineScaleBuf.GetID(), second.combineScaleBuf.GetID())
	}
}

func TestPLHostScratchPoolKeepsDimensionsResident(t *testing.T) {
	requireNativeRuntime(t)

	smallScale := float32(1 / math.Sqrt(float64(16)))
	small, err := getPLHostScratch(16, 16, smallScale)
	if err != nil {
		t.Fatalf("get small PL host scratch: %v", err)
	}
	putPLHostScratch(small)
	largeScale := float32(1 / math.Sqrt(float64(32)))
	large, err := getPLHostScratch(32, 32, largeScale)
	if err != nil {
		t.Fatalf("get large PL host scratch: %v", err)
	}
	putPLHostScratch(large)
	forceNativeGC()
	forceNativeGC()

	gotSmall, err := getPLHostScratch(16, 16, smallScale)
	if err != nil {
		t.Fatalf("get small PL host scratch again: %v", err)
	}
	defer putPLHostScratch(gotSmall)
	if gotSmall != small {
		t.Fatal("PL host scratch pool evicted the small dimension after using a larger dimension")
	}
	gotLarge, err := getPLHostScratch(32, 32, largeScale)
	if err != nil {
		t.Fatalf("get large PL host scratch again: %v", err)
	}
	defer putPLHostScratch(gotLarge)
	if gotLarge != large {
		t.Fatal("PL host scratch pool evicted the large dimension after reusing the small dimension")
	}
}

func TestPerLayerProjBatchedAllocationBudget(t *testing.T) {
	requireNativeRuntime(t)
	resetResidentBufsForTest()
	defer resetResidentBufsForTest()

	const numLayers, pliDim, dModel = 2, 64, 128
	const eps = float32(1e-5)
	plDim := numLayers * pliDim
	projScale := float32(1 / math.Sqrt(float64(dModel)))
	hidden := toBF16Bytes(syntheticFloat32(dModel, 11))
	perLayer := toBF16Bytes(syntheticFloat32(plDim, 12))
	projW := toBF16Bytes(syntheticFloat32(plDim*dModel, 13))
	projNormW := toBF16Bytes(syntheticFloat32(pliDim, 14))
	projView := copyView(projW)

	if _, err := perLayerProjBatched(projView, hidden, perLayer, projScale, projNormW, plDim, numLayers, pliDim, dModel, eps); err != nil {
		t.Fatalf("perLayerProjBatched warmup: %v", err)
	}

	allocs := testing.AllocsPerRun(5, func() {
		if _, err := perLayerProjBatched(projView, hidden, perLayer, projScale, projNormW, plDim, numLayers, pliDim, dModel, eps); err != nil {
			t.Fatalf("perLayerProjBatched: %v", err)
		}
	})
	if allocs > 145 {
		t.Fatalf("perLayerProjBatched allocations = %.0f, want <= 145", allocs)
	}
}

func TestPerLayerProjBatchedInputGuards(t *testing.T) {
	const numLayers, pliDim, dModel = 2, 3, 4
	plDim := numLayers * pliDim
	hidden := make([]byte, dModel*bf16Size)
	perLayer := make([]byte, plDim*bf16Size)
	projNormW := make([]byte, pliDim*bf16Size)

	tests := []struct {
		name      string
		projView  bufView
		hidden    []byte
		perLayer  []byte
		projNormW []byte
		numLayers int
		pliDim    int
		dModel    int
	}{
		{"zero layers", bufView{}, hidden, perLayer, projNormW, 0, pliDim, dModel},
		{"bad hidden", bufView{}, hidden[:len(hidden)-1], perLayer, projNormW, numLayers, pliDim, dModel},
		{"bad per-layer", bufView{}, hidden, perLayer[:len(perLayer)-1], projNormW, numLayers, pliDim, dModel},
		{"bad norm", bufView{}, hidden, perLayer, projNormW[:len(projNormW)-1], numLayers, pliDim, dModel},
		{"nil resident view", bufView{}, hidden, perLayer, projNormW, numLayers, pliDim, dModel},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			_, err := perLayerProjBatched(tc.projView, tc.hidden, tc.perLayer, 1, tc.projNormW, tc.numLayers*tc.pliDim, tc.numLayers, tc.pliDim, tc.dModel, 1e-5)
			if err == nil {
				t.Fatal("perLayerProjBatched error = nil")
			}
		})
	}
}

// TestEmbedRowsBatchQuantDeviceMatchesEmbedTokenQuant gates the dense-arch
// device embed gather (#381): the one-dispatch rows kernel must reproduce the
// host embedTokenQuant oracle byte-for-byte for every row of the chunk, at
// every affine width the models ship — the batched pass binds these bytes as
// its input rows, so exactness is the whole contract.
func TestEmbedRowsBatchQuantDeviceMatchesEmbedTokenQuant(t *testing.T) {
	requireNativeRuntime(t)
	if _, err := pleGatherRowsQuantPipeline(); err != nil {
		t.Skip("rows-quant kernel not loaded")
	}
	const vocab, dModel, gs = 256, 512, 64
	const scale = float32(0.5)
	for _, bits := range []int{4, 8} {
		packed, scales, biases := embedGatherQuantFixture(vocab, dModel, gs, bits)
		mainEmb := &mainEmbedGather{
			packed: residentBytes(packed), scales: residentBytes(scales), biases: residentBytes(biases),
			gs: gs, bits: bits, scale: scale,
		}
		ids := make([]int32, steelGEMMMinRows+3)
		for i := range ids {
			ids[i] = int32((i*37 + 5) % vocab)
		}
		sc := &embedRowsScratch{}
		embBuf, ok, err := embedRowsBatchQuantDevice(sc, ids, mainEmb, dModel)
		if err != nil || !ok || embBuf == nil {
			t.Fatalf("b%d: embedRowsBatchQuantDevice ok=%v err=%v", bits, ok, err)
		}
		// the production ordering contract: the consumer's command buffer follows
		// on the same queue, so an empty committed-and-waited buffer drains the
		// builder before the host reads the rows.
		withAutoreleasePool(func() {
			cb := commandBufferFast(queue)
			commitCommandBufferFast(cb)
			waitUntilCompletedFast(cb)
		})
		rowBytes := dModel * bf16Size
		got := unsafe.Slice((*byte)(embBuf.Contents()), len(ids)*rowBytes)
		for i, id := range ids {
			want, refErr := embedTokenQuant(packed, scales, biases, id, vocab, dModel, gs, bits, scale)
			if refErr != nil {
				t.Fatalf("b%d row %d: embedTokenQuant: %v", bits, i, refErr)
			}
			if !bytes.Equal(got[i*rowBytes:(i+1)*rowBytes], want) {
				t.Fatalf("b%d row %d (tok %d): device rows gather differs from host embedTokenQuant", bits, i, id)
			}
		}
	}
}

// TestEmbedRowsBatchQuantDeviceDeclinesBelowFloor pins the fallback contract:
// a chunk below the batch floor (and a nil table) reports no-work — never an
// error — so the caller keeps the proven per-token host loop.
func TestEmbedRowsBatchQuantDeviceDeclinesBelowFloor(t *testing.T) {
	requireNativeRuntime(t)
	const vocab, dModel, gs, bits = 64, 128, 64, 4
	packed, scales, biases := embedGatherQuantFixture(vocab, dModel, gs, bits)
	mainEmb := &mainEmbedGather{
		packed: residentBytes(packed), scales: residentBytes(scales), biases: residentBytes(biases),
		gs: gs, bits: bits, scale: 1,
	}
	sc := &embedRowsScratch{}
	ids := make([]int32, steelGEMMMinRows-1)
	if buf, ok, err := embedRowsBatchQuantDevice(sc, ids, mainEmb, dModel); buf != nil || ok || err != nil {
		t.Fatalf("below-floor chunk: want decline, got buf=%v ok=%v err=%v", buf != nil, ok, err)
	}
	full := make([]int32, steelGEMMMinRows)
	if buf, ok, err := embedRowsBatchQuantDevice(sc, full, nil, dModel); buf != nil || ok || err != nil {
		t.Fatalf("nil mainEmb: want decline, got buf=%v ok=%v err=%v", buf != nil, ok, err)
	}
}
