// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"bytes"
	"testing"
	"unsafe"
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
