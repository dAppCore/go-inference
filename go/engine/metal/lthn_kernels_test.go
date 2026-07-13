// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"bytes"
	"sync"
	"testing"
	"unsafe"

	"github.com/tmc/apple/metal"
)

func TestGeluKernelCapabilityReflectsLoadedFlag(t *testing.T) {
	old := customLibraryLoaded
	defer func() { customLibraryLoaded = old }()

	customLibraryLoaded = false
	if gpuHasGeluKernel() {
		t.Fatal("gpuHasGeluKernel true when custom library flag is false")
	}
	customLibraryLoaded = true
	if !gpuHasGeluKernel() {
		t.Fatal("gpuHasGeluKernel false when custom library flag is true")
	}
}

func TestBF16ScalarBytes_EncodesValue(t *testing.T) {
	got := bf16ScalarBytes(-2.5)
	want := f32ToBF16(-2.5)
	if got[0] != byte(want) || got[1] != byte(want>>8) {
		t.Fatalf("bf16ScalarBytes = %v, want little-endian %#04x", got, want)
	}
}

func TestQ4LMHeadTopKCandidateCount_Tiles(t *testing.T) {
	got := q4LMHeadTopKCandidateCount(q4LMHeadTopKRowsPerTile*2+1, 3)
	if want := 9; got != want {
		t.Fatalf("q4LMHeadTopKCandidateCount = %d, want %d", got, want)
	}
}

func TestQ4LMHeadTopKCandidatesPerTile_Clamps(t *testing.T) {
	if got := q4LMHeadTopKCandidatesPerTile(q4LMHeadTopKRowsPerTile + 1); got != q4LMHeadTopKRowsPerTile {
		t.Fatalf("q4LMHeadTopKCandidatesPerTile = %d, want %d", got, q4LMHeadTopKRowsPerTile)
	}
	if got := q4LMHeadTopKCandidatesPerTile(3); got != 3 {
		t.Fatalf("q4LMHeadTopKCandidatesPerTile(3) = %d", got)
	}
}

func TestQMVLogitsArgmaxUsable_InvalidGeometry(t *testing.T) {
	if qmvLogitsArgmaxUsable(65, 32, 64, 4) {
		t.Fatal("qmvLogitsArgmaxUsable accepted dModel not divisible by group size")
	}
}

func TestQMVLogitsTopKUsable_InvalidTopK(t *testing.T) {
	if qmvLogitsTopKUsable(64, 8, 64, 4, 9) {
		t.Fatal("qmvLogitsTopKUsable accepted topK larger than vocabulary")
	}
}

func TestQ4LMHeadTopKUsable_InvalidBits(t *testing.T) {
	if q4LMHeadTopKUsable(128, 64, 64, 8, 4) {
		t.Fatal("q4LMHeadTopKUsable accepted a non-q4 weight")
	}
}

func TestEncMulScalarBF16_RejectsNegativeAndAcceptsZeroLength(t *testing.T) {
	if err := encMulScalarBF16(nil, nil, nil, nil, 0, -1); err == nil {
		t.Fatal("encMulScalarBF16 accepted a negative length")
	}
	if err := encMulScalarBF16(nil, nil, nil, nil, 0, 0); err != nil {
		t.Fatalf("encMulScalarBF16 zero length: %v", err)
	}
}

func TestEncMulScalarBF16Object_RejectsNegativeAndAcceptsZeroLength(t *testing.T) {
	var enc metal.MTLComputeCommandEncoderObject
	if err := encMulScalarBF16Object(enc, nil, nil, nil, 0, -1); err == nil {
		t.Fatal("encMulScalarBF16Object accepted a negative length")
	}
	if err := encMulScalarBF16Object(enc, nil, nil, nil, 0, 0); err != nil {
		t.Fatalf("encMulScalarBF16Object zero length: %v", err)
	}
}

func TestEncRouterTopKBF16_RejectsOutOfRangeTopK(t *testing.T) {
	if err := encRouterTopKBF16(nil, nil, nil, nil, nil, 0, 4, 0, false); err == nil {
		t.Fatal("encRouterTopKBF16 accepted topK zero")
	}
	if err := encRouterTopKBF16(nil, nil, nil, nil, nil, 0, 4, 5, false); err == nil {
		t.Fatal("encRouterTopKBF16 accepted topK above numExperts")
	}
}

func TestEncBF16LogitsArgmaxTilesBF16At_RejectsNonPositiveVocab(t *testing.T) {
	if err := encBF16LogitsArgmaxTilesBF16At(nil, nil, nil, nil, nil, 0, 0, 0, 0, 0); err == nil {
		t.Fatal("encBF16LogitsArgmaxTilesBF16At accepted vocab zero")
	}
}

func TestEncBF16LMHeadArgmaxTilesBF16_RejectsNonPositiveDimensions(t *testing.T) {
	if err := encBF16LMHeadArgmaxTilesBF16(nil, nil, nil, nil, nil, nil, 0, 0, 0, 8, 0); err == nil {
		t.Fatal("encBF16LMHeadArgmaxTilesBF16 accepted dModel zero")
	}
	if err := encBF16LMHeadArgmaxTilesBF16(nil, nil, nil, nil, nil, nil, 0, 0, 8, 0, 0); err == nil {
		t.Fatal("encBF16LMHeadArgmaxTilesBF16 accepted vocab zero")
	}
}

func TestEncBF16LMHeadArgmaxTilesRowsBF16_RejectsBatchFloorAndCeiling(t *testing.T) {
	if err := encBF16LMHeadArgmaxTilesRowsBF16(nil, nil, nil, nil, nil, nil, 0, 0, 8, 8, 0, 0, 1); err == nil {
		t.Fatal("encBF16LMHeadArgmaxTilesRowsBF16 accepted k zero")
	}
	if err := encBF16LMHeadArgmaxTilesRowsBF16(nil, nil, nil, nil, nil, nil, 0, 0, 8, 8, 0, 9, 1); err == nil {
		t.Fatal("encBF16LMHeadArgmaxTilesRowsBF16 accepted k above eight")
	}
}

func TestEncArgmaxMergeRowsF32_RejectsNonPositiveDimensions(t *testing.T) {
	if err := encArgmaxMergeRowsF32(nil, nil, nil, nil, 0, 1); err == nil {
		t.Fatal("encArgmaxMergeRowsF32 accepted n zero")
	}
	if err := encArgmaxMergeRowsF32(nil, nil, nil, nil, 1, 0); err == nil {
		t.Fatal("encArgmaxMergeRowsF32 accepted k zero")
	}
}

func TestEncBF16LogitsTopKTilesBF16_RejectsTopKOutsideKernelLimit(t *testing.T) {
	if err := encBF16LogitsTopKTilesBF16(nil, nil, nil, nil, nil, nil, 1, 0, 0, 0, 1, 0); err == nil {
		t.Fatal("encBF16LogitsTopKTilesBF16 accepted topK zero")
	}
	if err := encBF16LogitsTopKTilesBF16(nil, nil, nil, nil, nil, nil, 1, 0, 0, headSampleTopKMaxK+1, 1, 0); err == nil {
		t.Fatal("encBF16LogitsTopKTilesBF16 accepted topK above the kernel limit")
	}
}

func TestEncBF16LogitsTopKTilesBF16Object_RejectsTopKOutsideKernelLimit(t *testing.T) {
	var enc metal.MTLComputeCommandEncoderObject
	if err := encBF16LogitsTopKTilesBF16Object(enc, nil, nil, nil, nil, nil, 1, 0, 0, 0, 1, 0); err == nil {
		t.Fatal("encBF16LogitsTopKTilesBF16Object accepted topK zero")
	}
}

func TestEncQ4LMHeadTopKTilesBF16_RejectsCandidateAndDimensionEdges(t *testing.T) {
	if err := encQ4LMHeadTopKTilesBF16(nil, nil, nil, nil, nil, nil, nil, nil, nil, 0, 0, 0, 0, q4LMHeadTopKBlockSize, 8, 64, 0, 0, 4, 0, 1, 0); err == nil {
		t.Fatal("encQ4LMHeadTopKTilesBF16 accepted candidatesPerTile zero")
	}
	if err := encQ4LMHeadTopKTilesBF16(nil, nil, nil, nil, nil, nil, nil, nil, nil, 0, 0, 0, 0, q4LMHeadTopKBlockSize, 8, 16, 0, 0, 4, 4, 1, 0); err == nil {
		t.Fatal("encQ4LMHeadTopKTilesBF16 accepted an unsupported group size")
	}
	if err := encQ4LMHeadTopKTilesBF16(nil, nil, nil, nil, nil, nil, nil, nil, nil, 0, 0, 0, 0, q4LMHeadTopKBlockSize+64, 8, 64, 0, 0, 4, 4, 1, 0); err == nil {
		t.Fatal("encQ4LMHeadTopKTilesBF16 accepted dModel outside the 512-byte block")
	}
}

func TestEncTopKMergeF32_RejectsInvalidDimensions(t *testing.T) {
	if err := encTopKMergeF32(nil, nil, nil, nil, nil, 0, 1); err == nil {
		t.Fatal("encTopKMergeF32 accepted n zero")
	}
	if err := encTopKMergeF32(nil, nil, nil, nil, nil, 1, 0); err == nil {
		t.Fatal("encTopKMergeF32 accepted topK zero")
	}
}

func TestEncTopKMergeF32Object_RejectsInvalidDimensions(t *testing.T) {
	var enc metal.MTLComputeCommandEncoderObject
	if err := encTopKMergeF32Object(enc, nil, nil, nil, nil, 1, headSampleTopKMaxK+1); err == nil {
		t.Fatal("encTopKMergeF32Object accepted topK above the kernel limit")
	}
}

func TestEncTopKMergeSampleF32_RejectsMissingParams(t *testing.T) {
	if err := encTopKMergeSampleF32(nil, nil, nil, nil, nil); err == nil {
		t.Fatal("encTopKMergeSampleF32 accepted a nil params buffer")
	}
}

func TestEncTopKMergeSampleF32Object_RejectsMissingParams(t *testing.T) {
	var enc metal.MTLComputeCommandEncoderObject
	if err := encTopKMergeSampleF32Object(enc, nil, nil, nil, nil); err == nil {
		t.Fatal("encTopKMergeSampleF32Object accepted a nil params buffer")
	}
}

func TestEncLogitsSampleBF16_RejectsMissingParams(t *testing.T) {
	if err := encLogitsSampleBF16(nil, nil, nil, nil, nil, nil); err == nil {
		t.Fatal("encLogitsSampleBF16 accepted a nil params buffer")
	}
}

func TestEncLogitsSampleBF16Object_RejectsMissingParams(t *testing.T) {
	var enc metal.MTLComputeCommandEncoderObject
	if err := encLogitsSampleBF16Object(enc, nil, nil, nil, nil, nil); err == nil {
		t.Fatal("encLogitsSampleBF16Object accepted a nil params buffer")
	}
}

func TestMulScalarBF16MatchesBroadcastMultiply(t *testing.T) {
	requireNativeRuntime(t)
	if _, err := bf16MulScalarPipeline(); err != nil {
		t.Fatalf("bf16 scalar kernel unavailable: %v", err)
	}

	in := toBF16Bytes([]float32{-2, -0.5, 0, 0.25, 1.5, 3})
	scalar := toBF16Bytes([]float32{0.375})
	got, err := MulScalarBF16(in, scalar)
	if err != nil {
		t.Fatalf("MulScalarBF16: %v", err)
	}
	want, err := MulBF16(in, scalarFillBF16(scalar, len(in)/bf16Size))
	if err != nil {
		t.Fatalf("broadcast MulBF16: %v", err)
	}
	eqBytes(t, "MulScalarBF16", got, want)
}

func TestMulScalarBF16IntoUsesCallerOutput(t *testing.T) {
	requireNativeRuntime(t)
	if _, err := bf16MulScalarPipeline(); err != nil {
		t.Fatalf("bf16 scalar kernel unavailable: %v", err)
	}

	in := toBF16Bytes(syntheticFloat32(1024, 17))
	scalar := toBF16Bytes([]float32{0.375})
	out := make([]byte, len(in))
	for i := range out {
		out[i] = 0xA5
	}

	if err := MulScalarBF16Into(out, in, scalar); err != nil {
		t.Fatalf("MulScalarBF16Into: %v", err)
	}
	want, err := MulBF16(in, scalarFillBF16(scalar, len(in)/bf16Size))
	if err != nil {
		t.Fatalf("broadcast MulBF16: %v", err)
	}
	if !bytes.Equal(out, want) {
		t.Fatal("MulScalarBF16Into output differs from broadcast multiply")
	}
}

func TestMulScalarBF16AllocationBudget(t *testing.T) {
	requireNativeRuntime(t)
	if _, err := bf16MulScalarPipeline(); err != nil {
		t.Fatalf("bf16 scalar kernel unavailable: %v", err)
	}

	in := toBF16Bytes(syntheticFloat32(1024, 17))
	scalar := toBF16Bytes([]float32{0.375})
	if _, err := MulScalarBF16(in, scalar); err != nil {
		t.Fatalf("MulScalarBF16 warmup: %v", err)
	}

	allocs := testing.AllocsPerRun(5, func() {
		if _, err := MulScalarBF16(in, scalar); err != nil {
			t.Fatalf("MulScalarBF16: %v", err)
		}
	})
	if allocs > 10 {
		t.Fatalf("MulScalarBF16 allocations = %.0f, want <= 10", allocs)
	}
}

func TestMulScalarBF16KeepsScalarBufferResident(t *testing.T) {
	requireNativeRuntime(t)
	if _, err := bf16MulScalarPipeline(); err != nil {
		t.Fatalf("bf16 scalar kernel unavailable: %v", err)
	}
	resetResidentBufsForTest()
	defer resetResidentBufsForTest()

	const scalarValue = float32(0.375)
	key := bf16ConstKey{n: 1, v: scalarValue}
	bf16ConstMu.Lock()
	delete(bf16ConstCache, key)
	bf16ConstMu.Unlock()

	in := toBF16Bytes([]float32{-2, -0.5, 0, 0.25, 1.5, 3})
	scalar := toBF16Bytes([]float32{scalarValue})
	if _, err := MulScalarBF16(in, scalar); err != nil {
		t.Fatalf("MulScalarBF16: %v", err)
	}

	bf16ConstMu.Lock()
	_, cached := bf16ConstCache[key]
	bf16ConstMu.Unlock()
	if !cached {
		t.Fatal("MulScalarBF16 did not cache its one-element BF16 scalar buffer")
	}
}

func TestMulRowsPipelineICB_LoadsICBCapablePipeline(t *testing.T) {
	requireNativeRuntime(t)
	pso, err := mulRowsPipelineICB()
	if err != nil {
		t.Fatalf("mulRowsPipelineICB: %v", err)
	}
	if pso == nil || pso.GetID() == 0 {
		t.Fatal("mulRowsPipelineICB returned an empty pipeline")
	}
}

func TestBF16LogitsCandidatesPipeline_LoadsPipeline(t *testing.T) {
	requireNativeRuntime(t)
	pso, err := bf16LogitsCandidatesPipeline()
	if err != nil {
		t.Fatalf("bf16LogitsCandidatesPipeline: %v", err)
	}
	if pso == nil || pso.GetID() == 0 {
		t.Fatal("bf16LogitsCandidatesPipeline returned an empty pipeline")
	}
}

func TestEncArgmaxMergeF32At_ReturnsKnownIndex(t *testing.T) {
	requireNativeRuntime(t)
	values := shared([]float32{1, 9, 3, 7})
	indices := scratch(4)
	copy(unsafe.Slice((*int32)(indices.Contents()), 4), []int32{4, 9, 3, 7})
	out := scalarI32(-1)
	cb := queue.CommandBuffer()
	enc := cb.ComputeCommandEncoder()
	if err := encArgmaxMergeF32At(enc, values, indices, out, 0, 0, 0, 4); err != nil {
		enc.EndEncoding()
		t.Fatalf("encArgmaxMergeF32At: %v", err)
	}
	enc.EndEncoding()
	cb.Commit()
	cb.WaitUntilCompleted()
	if got := *(*int32)(out.Contents()); got != 9 {
		t.Fatalf("encArgmaxMergeF32At output = %d, want 9", got)
	}
}

func TestEncBF16LMHeadCandidatesBF16_MatchesHostDots(t *testing.T) {
	requireNativeRuntime(t)
	const dModel, vocab = 4, 11
	xBytes := toBF16Bytes([]float32{1, -2, 0.5, 3})
	wBytes := toBF16Bytes([]float32{
		1, 2, 3, 4,
		-1, 0.5, 2, -3,
		4, 3, 2, 1,
		-2, -1, 0, 1,
		0.25, 0.5, 0.75, 1,
		-4, 2, -1, 3,
		2, 1, 0, -1,
		-3, 1, 2, 4,
		0, 0.5, 1, 1.5,
		5, -1, 2, 0,
		-2, 4, -3, 1,
	})
	x := sharedBytes(xBytes)
	weight := sharedBytes(wBytes)
	values := scratch((vocab + bf16LMHeadArgmaxRowsPerTile - 1) / bf16LMHeadArgmaxRowsPerTile * bf16LMHeadArgmaxRowsPerTile)
	indices := scratch((vocab + bf16LMHeadArgmaxRowsPerTile - 1) / bf16LMHeadArgmaxRowsPerTile * bf16LMHeadArgmaxRowsPerTile)
	suppress := []int32{3}
	suppressBuf := scratch(len(suppress))
	copy(unsafe.Slice((*int32)(suppressBuf.Contents()), len(suppress)), suppress)
	cb := queue.CommandBuffer()
	enc := cb.ComputeCommandEncoder()
	if err := encBF16LMHeadCandidatesBF16(enc, x, weight, values, indices, suppressBuf, nil, 0, 0, dModel, vocab, len(suppress), 0, 1, 0); err != nil {
		enc.EndEncoding()
		t.Fatalf("encBF16LMHeadCandidatesBF16: %v", err)
	}
	enc.EndEncoding()
	cb.Commit()
	cb.WaitUntilCompleted()
	gotValues := unsafe.Slice((*float32)(values.Contents()), 16)
	gotIDs := unsafe.Slice((*int32)(indices.Contents()), 16)
	xf := bf16Floats(xBytes)
	wf := bf16Floats(wBytes)
	for row := 0; row < vocab; row++ {
		if row == 3 {
			if gotIDs[row] != -1 {
				t.Fatalf("suppressed row id = %d, want -1", gotIDs[row])
			}
			continue
		}
		if gotIDs[row] != int32(row) {
			t.Fatalf("row %d id = %d, want %d", row, gotIDs[row], row)
		}
		var want float32
		for col := 0; col < dModel; col++ {
			want += xf[col] * wf[row*dModel+col]
		}
		if d := gotValues[row] - want; d < -0.02 || d > 0.02 {
			t.Fatalf("row %d value = %v, want %v", row, gotValues[row], want)
		}
	}
}

func TestEncBF16LogitsCandidatesBF16_MatchesLogitsAndSuppression(t *testing.T) {
	requireNativeRuntime(t)
	const vocab = 7
	logitsBytes := toBF16Bytes([]float32{-1, 2.5, 0, 9, -3, 4, 1.25})
	logits := sharedBytes(logitsBytes)
	values := scratch(vocab)
	indices := scratch(vocab)
	suppress := scratch(1)
	*(*int32)(suppress.Contents()) = 3
	cb := queue.CommandBuffer()
	enc := cb.ComputeCommandEncoder()
	if err := encBF16LogitsCandidatesBF16(enc, logits, values, indices, suppress, vocab, 1, 0); err != nil {
		enc.EndEncoding()
		t.Fatalf("encBF16LogitsCandidatesBF16: %v", err)
	}
	enc.EndEncoding()
	cb.Commit()
	cb.WaitUntilCompleted()
	gotValues := unsafe.Slice((*float32)(values.Contents()), vocab)
	gotIDs := unsafe.Slice((*int32)(indices.Contents()), vocab)
	for i, want := range []float32{-1, 2.5, 0, 9, -3, 4, 1.25} {
		if i == 3 {
			if gotIDs[i] != -1 {
				t.Fatalf("suppressed logits id = %d, want -1", gotIDs[i])
			}
			continue
		}
		if gotIDs[i] != int32(i) || gotValues[i] != want {
			t.Fatalf("logits row %d = (%v,%d), want (%v,%d)", i, gotValues[i], gotIDs[i], want, i)
		}
	}
}

func TestEncBF16LogitsTopKTilesBF16_MatchesHostTopKPerTile(t *testing.T) {
	requireNativeRuntime(t)
	const vocab, topK = 300, 3
	vals := make([]float32, vocab)
	for i := range vals {
		vals[i] = -10
	}
	vals[3], vals[7], vals[10] = 9, 8, 7
	vals[260], vals[299], vals[280] = 12, 11, 10
	logits := sharedBytes(toBF16Bytes(vals))
	values := scratch(2 * topK)
	indices := scratch(2 * topK)
	suppress := scratch(1)
	*(*int32)(suppress.Contents()) = 7
	cb := queue.CommandBuffer()
	enc := cb.ComputeCommandEncoder()
	if err := encBF16LogitsTopKTilesBF16(enc, logits, values, indices, suppress, nil, vocab, 1, 0, topK, 1, 0); err != nil {
		enc.EndEncoding()
		t.Fatalf("encBF16LogitsTopKTilesBF16: %v", err)
	}
	enc.EndEncoding()
	cb.Commit()
	cb.WaitUntilCompleted()
	gotValues := unsafe.Slice((*float32)(values.Contents()), 2*topK)
	gotIDs := unsafe.Slice((*int32)(indices.Contents()), 2*topK)
	// Each tile's candidates come out value-sorted descending. Tile 1 (ids
	// 0-255, id 7 suppressed): 9, 7, then the -10 tie winner with the lowest
	// valid id, 0. Tile 2 (ids 256-299): 12, 11, 10.
	wantIDs := []int32{3, 10, 0, 260, 299, 280}
	for i, want := range wantIDs {
		if gotIDs[i] != want {
			t.Fatalf("top-k candidate id[%d] = %d, want %d (all %v)", i, gotIDs[i], want, gotIDs)
		}
	}
	if gotValues[0] != 9 || gotValues[1] != 7 || gotValues[3] != 12 || gotValues[4] != 11 || gotValues[5] != 10 {
		t.Fatalf("top-k candidate values = %v, want tile highs", gotValues)
	}
}

func TestEncQ4LMHeadTopKTilesBF16_MatchesQMVTopIDs(t *testing.T) {
	requireNativeRuntime(t)
	const dModel, vocab, groupSize, bits, topK = 512, 17, 64, 4, 4
	q := quantWeightFixture(t, vocab, dModel, groupSize, bits, 31)
	xBytes := toBF16Bytes(syntheticFloat32(dModel, 17))
	values := scratch(topK)
	indices := scratch(topK)
	qmv, err := QMVBF16(xBytes, q.Packed, q.Scales, q.Biases, vocab, dModel, groupSize, bits)
	if err != nil {
		t.Fatalf("QMVBF16 reference: %v", err)
	}
	qmvTop := make([]int32, topK)
	used := make(map[int32]bool, topK)
	for k := range topK {
		best := int32(-1)
		var bestV float32
		for i := 0; i < vocab; i++ {
			if used[int32(i)] {
				continue
			}
			v := bf16ToF32(qmv[i*bf16Size], qmv[i*bf16Size+1])
			if best < 0 || v > bestV {
				best, bestV = int32(i), v
			}
		}
		qmvTop[k], used[best] = best, true
	}
	cb := queue.CommandBuffer()
	enc := cb.ComputeCommandEncoder()
	if err := encQ4LMHeadTopKTilesBF16(enc, sharedBytes(xBytes), sharedBytes(q.Packed), sharedBytes(q.Scales), sharedBytes(q.Biases), values, indices, nil, nil, 0, 0, 0, 0, dModel, vocab, groupSize, 0, 0, topK, topK, 1, 0); err != nil {
		enc.EndEncoding()
		t.Fatalf("encQ4LMHeadTopKTilesBF16: %v", err)
	}
	enc.EndEncoding()
	cb.Commit()
	cb.WaitUntilCompleted()
	gotIDs := unsafe.Slice((*int32)(indices.Contents()), topK)
	gotValues := unsafe.Slice((*float32)(values.Contents()), topK)
	for i, want := range qmvTop {
		if gotIDs[i] != want {
			t.Fatalf("q4 top-k id[%d] = %d, want qmv id %d (got %v want %v)", i, gotIDs[i], want, gotIDs, qmvTop)
		}
		qv := bf16ToF32(qmv[want*bf16Size], qmv[want*bf16Size+1])
		if d := gotValues[i] - qv; d < -0.25 || d > 0.25 {
			t.Fatalf("q4 top-k value[%d] = %v, want qmv %v", i, gotValues[i], qv)
		}
	}
}

func TestEncTopKMergeF32_ReturnsSortedCandidates(t *testing.T) {
	requireNativeRuntime(t)
	values := shared([]float32{1, 9, 4, 7, 3, 8})
	indices := scratch(6)
	copy(unsafe.Slice((*int32)(indices.Contents()), 6), []int32{10, 11, 12, 13, 14, 15})
	outValues := scratch(3)
	outIndices := scratch(3)
	cb := queue.CommandBuffer()
	enc := cb.ComputeCommandEncoder()
	if err := encTopKMergeF32(enc, values, indices, outValues, outIndices, 6, 3); err != nil {
		enc.EndEncoding()
		t.Fatalf("encTopKMergeF32: %v", err)
	}
	enc.EndEncoding()
	cb.Commit()
	cb.WaitUntilCompleted()
	gotValues := unsafe.Slice((*float32)(outValues.Contents()), 3)
	gotIDs := unsafe.Slice((*int32)(outIndices.Contents()), 3)
	if gotIDs[0] != 11 || gotIDs[1] != 15 || gotIDs[2] != 13 || gotValues[0] != 9 || gotValues[1] != 8 || gotValues[2] != 7 {
		t.Fatalf("encTopKMergeF32 = values %v ids %v, want [9 8 7] [11 15 13]", gotValues, gotIDs)
	}
}

func TestEncTopKMergeSampleF32_ReturnsTopCandidateForZeroDraw(t *testing.T) {
	requireNativeRuntime(t)
	values := shared([]float32{4, 3, 2, 1})
	indices := scratch(4)
	copy(unsafe.Slice((*int32)(indices.Contents()), 4), []int32{40, 41, 42, 43})
	out := scalarI32(-1)
	params := topKSampleKernelParams{n: 4, topK: 3, temperature: 1, topP: 1, draw: 0}
	paramsBytes := unsafe.Slice((*byte)(unsafe.Pointer(&params)), int(unsafe.Sizeof(params)))
	paramsBuf := sharedBytes(paramsBytes)
	cb := queue.CommandBuffer()
	enc := cb.ComputeCommandEncoder()
	if err := encTopKMergeSampleF32(enc, values, indices, out, paramsBuf); err != nil {
		enc.EndEncoding()
		t.Fatalf("encTopKMergeSampleF32: %v", err)
	}
	enc.EndEncoding()
	cb.Commit()
	cb.WaitUntilCompleted()
	if got := *(*int32)(out.Contents()); got != 40 {
		t.Fatalf("encTopKMergeSampleF32 output = %d, want 40", got)
	}
}

func resetLTHNKernelsPSOsForTest() {
	mulRowsPSOOnce, mulRowsPSO, mulRowsPSOErr = sync.Once{}, nil, nil
	mulRowsICBPSOOnce, mulRowsICBPSO, mulRowsICBPSOErr = sync.Once{}, nil, nil
	bf16MulScalarPSOOnce, bf16MulScalarPSO, bf16MulScalarPSOErr = sync.Once{}, nil, nil
	moeWeightedSumPSOOnce, moeWeightedSumPSO, moeWeightedSumPSOErr = sync.Once{}, nil, nil
	moeCombineNormsPSOOnce, moeCombineNormsPSO, moeCombineNormsPSOErr = sync.Once{}, nil, nil
	argmaxMergeRowsF32PSOOnce, argmaxMergeRowsF32PSO, argmaxMergeRowsF32PSOErr = sync.Once{}, nil, nil
	bf16LMHeadArgmaxTilesPSOOnce, bf16LMHeadArgmaxTilesPSO, bf16LMHeadArgmaxTilesPSOErr = sync.Once{}, nil, nil
	bf16LogitsArgmaxTilesPSOOnce, bf16LogitsArgmaxTilesPSO, bf16LogitsArgmaxTilesPSOErr = sync.Once{}, nil, nil
	argmaxMergeF32PSOOnce, argmaxMergeF32PSO, argmaxMergeF32PSOErr = sync.Once{}, nil, nil
	bf16LMHeadCandidatesPSOOnce, bf16LMHeadCandidatesPSO, bf16LMHeadCandidatesPSOErr = sync.Once{}, nil, nil
	bf16LogitsCandidatesPSOOnce, bf16LogitsCandidatesPSO, bf16LogitsCandidatesPSOErr = sync.Once{}, nil, nil
	bf16LogitsTopKTilesPSOOnce, bf16LogitsTopKTilesPSO, bf16LogitsTopKTilesPSOErr = sync.Once{}, nil, nil
	q4LMHeadTopKTilesPSOOnce, q4LMHeadTopKTilesPSO, q4LMHeadTopKTilesPSOErr = sync.Once{}, nil, nil
	topKMergeF32PSOOnce, topKMergeF32PSO, topKMergeF32PSOErr = sync.Once{}, nil, nil
	topKMergeSampleF32PSOOnce, topKMergeSampleF32PSO, topKMergeSampleF32PSOErr = sync.Once{}, nil, nil
	logitsSampleBF16PSOOnce, logitsSampleBF16PSO, logitsSampleBF16PSOErr = sync.Once{}, nil, nil
	ffnMegaBitsPSOMu.Lock()
	ffnMegaBitsPSOCache = map[int]metal.MTLComputePipelineState{}
	ffnMegaBitsPSOMu.Unlock()
	routerTopKPSOMu.Lock()
	routerTopKPSOCache = map[int]metal.MTLComputePipelineState{}
	routerTopKPSOMu.Unlock()
	bf16LMHeadArgmaxTilesRowsPSOMu.Lock()
	bf16LMHeadArgmaxTilesRowsPSOCache = map[int]metal.MTLComputePipelineState{}
	bf16LMHeadArgmaxTilesRowsPSOMu.Unlock()
}

func withWrongLTHNKernelsLibrary(t *testing.T, fn func()) {
	t.Helper()
	requireNativeRuntime(t)
	resetLTHNKernelsPSOsForTest()
	t.Cleanup(resetLTHNKernelsPSOsForTest)
	withWrongCustomLibrary(t, fn)
}

func TestMulRowsPipeline_WrongMetallib(t *testing.T) {
	withWrongLTHNKernelsLibrary(t, func() {
		if _, err := mulRowsPipeline(); err == nil {
			t.Fatal("mulRowsPipeline accepted the main metallib as a custom-kernel library")
		}
	})
}

func TestMulRowsPipelineICB_WrongMetallib(t *testing.T) {
	withWrongLTHNKernelsLibrary(t, func() {
		if _, err := mulRowsPipelineICB(); err == nil {
			t.Fatal("mulRowsPipelineICB accepted the main metallib as a custom-kernel library")
		}
	})
}

func TestBF16MulScalarPipeline_WrongMetallib(t *testing.T) {
	withWrongLTHNKernelsLibrary(t, func() {
		if _, err := bf16MulScalarPipeline(); err == nil {
			t.Fatal("bf16MulScalarPipeline accepted the main metallib as a custom-kernel library")
		}
	})
}

func TestMoeWeightedSumPipeline_WrongMetallib(t *testing.T) {
	withWrongLTHNKernelsLibrary(t, func() {
		if _, err := moeWeightedSumPipeline(); err == nil {
			t.Fatal("moeWeightedSumPipeline accepted the main metallib as a custom-kernel library")
		}
	})
}

func TestMoeCombineNormsPipeline_WrongMetallib(t *testing.T) {
	withWrongLTHNKernelsLibrary(t, func() {
		if _, err := moeCombineNormsPipeline(); err == nil {
			t.Fatal("moeCombineNormsPipeline accepted the main metallib as a custom-kernel library")
		}
	})
}

func TestArgmaxMergeRowsF32Pipeline_WrongMetallib(t *testing.T) {
	withWrongLTHNKernelsLibrary(t, func() {
		if _, err := argmaxMergeRowsF32Pipeline(); err == nil {
			t.Fatal("argmaxMergeRowsF32Pipeline accepted the main metallib as a custom-kernel library")
		}
	})
}

func TestBF16LMHeadArgmaxTilesPipeline_WrongMetallib(t *testing.T) {
	withWrongLTHNKernelsLibrary(t, func() {
		if _, err := bf16LMHeadArgmaxTilesPipeline(); err == nil {
			t.Fatal("bf16LMHeadArgmaxTilesPipeline accepted the main metallib as a custom-kernel library")
		}
	})
}

func TestBF16LogitsArgmaxTilesPipeline_WrongMetallib(t *testing.T) {
	withWrongLTHNKernelsLibrary(t, func() {
		if _, err := bf16LogitsArgmaxTilesPipeline(); err == nil {
			t.Fatal("bf16LogitsArgmaxTilesPipeline accepted the main metallib as a custom-kernel library")
		}
	})
}

func TestArgmaxMergeF32Pipeline_WrongMetallib(t *testing.T) {
	withWrongLTHNKernelsLibrary(t, func() {
		if _, err := argmaxMergeF32Pipeline(); err == nil {
			t.Fatal("argmaxMergeF32Pipeline accepted the main metallib as a custom-kernel library")
		}
	})
}

func TestBF16LMHeadCandidatesPipeline_WrongMetallib(t *testing.T) {
	withWrongLTHNKernelsLibrary(t, func() {
		if _, err := bf16LMHeadCandidatesPipeline(); err == nil {
			t.Fatal("bf16LMHeadCandidatesPipeline accepted the main metallib as a custom-kernel library")
		}
	})
}

func TestBF16LogitsCandidatesPipeline_WrongMetallib(t *testing.T) {
	withWrongLTHNKernelsLibrary(t, func() {
		if _, err := bf16LogitsCandidatesPipeline(); err == nil {
			t.Fatal("bf16LogitsCandidatesPipeline accepted the main metallib as a custom-kernel library")
		}
	})
}

func TestBF16LogitsTopKTilesPipeline_WrongMetallib(t *testing.T) {
	withWrongLTHNKernelsLibrary(t, func() {
		if _, err := bf16LogitsTopKTilesPipeline(); err == nil {
			t.Fatal("bf16LogitsTopKTilesPipeline accepted the main metallib as a custom-kernel library")
		}
	})
}

func TestQ4LMHeadTopKTilesPipeline_WrongMetallib(t *testing.T) {
	withWrongLTHNKernelsLibrary(t, func() {
		if _, err := q4LMHeadTopKTilesPipeline(); err == nil {
			t.Fatal("q4LMHeadTopKTilesPipeline accepted the main metallib as a custom-kernel library")
		}
	})
}

func TestTopKMergeF32Pipeline_WrongMetallib(t *testing.T) {
	withWrongLTHNKernelsLibrary(t, func() {
		if _, err := topKMergeF32Pipeline(); err == nil {
			t.Fatal("topKMergeF32Pipeline accepted the main metallib as a custom-kernel library")
		}
	})
}

func TestTopKMergeSampleF32Pipeline_WrongMetallib(t *testing.T) {
	withWrongLTHNKernelsLibrary(t, func() {
		if _, err := topKMergeSampleF32Pipeline(); err == nil {
			t.Fatal("topKMergeSampleF32Pipeline accepted the main metallib as a custom-kernel library")
		}
	})
}

func TestLogitsSampleBF16Pipeline_WrongMetallib(t *testing.T) {
	withWrongLTHNKernelsLibrary(t, func() {
		if _, err := logitsSampleBF16Pipeline(); err == nil {
			t.Fatal("logitsSampleBF16Pipeline accepted the main metallib as a custom-kernel library")
		}
	})
}

func TestFFNMegaPipelineBits_WrongMetallib(t *testing.T) {
	withWrongLTHNKernelsLibrary(t, func() {
		if _, err := ffnMegaPipelineBits(4); err == nil {
			t.Fatal("ffnMegaPipelineBits accepted the main metallib as a custom-kernel library")
		}
	})
}

func TestRouterTopKPipelineK_WrongMetallib(t *testing.T) {
	withWrongLTHNKernelsLibrary(t, func() {
		if _, err := routerTopKPipelineK(2); err == nil {
			t.Fatal("routerTopKPipelineK accepted the main metallib as a custom-kernel library")
		}
	})
}

func TestBF16LMHeadArgmaxTilesRowsPipeline_WrongMetallib(t *testing.T) {
	withWrongLTHNKernelsLibrary(t, func() {
		if _, err := bf16LMHeadArgmaxTilesRowsPipeline(2); err == nil {
			t.Fatal("bf16LMHeadArgmaxTilesRowsPipeline accepted the main metallib as a custom-kernel library")
		}
	})
}
