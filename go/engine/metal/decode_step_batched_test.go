// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"bytes"
	"sync"
	"testing"
)

// TestDecodeLayerBatchedKV asserts the MTP batched verify forward is BYTE-IDENTICAL to K sequential
// DecodeStepKV calls over the same growing KV cache: same K output hiddens, same final cache. This is
// the correctness bar for the batched verify — it must produce exactly what stepping the K draft
// tokens one at a time produces, so wiring it into MTPDecode keeps the token stream identical.
func TestDecodeLayerBatchedKV(t *testing.T) {
	requireNativeRuntime(t)
	const (
		dModel   = 256
		nHeads   = 4
		nKVHeads = 2
		headDim  = 64
		maxLen   = 32
		dFF      = 512
		basePos  = 5
		K        = 4
	)
	qDim, kvDim := nHeads*headDim, nKVHeads*headDim
	base, scale, eps := float32(10000), float32(1.0/8.0), float32(1e-6) // 1/sqrt(64)=1/8

	attnNormW := toBF16Bytes(syntheticFloat32(dModel, 1))
	mlpNormW := toBF16Bytes(syntheticFloat32(dModel, 2))
	wQ := toBF16Bytes(syntheticFloat32(qDim*dModel, 3))
	wK := toBF16Bytes(syntheticFloat32(kvDim*dModel, 4))
	wV := toBF16Bytes(syntheticFloat32(kvDim*dModel, 5))
	wO := toBF16Bytes(syntheticFloat32(dModel*qDim, 6))
	wGate := toBF16Bytes(syntheticFloat32(dFF*dModel, 7))
	wUp := toBF16Bytes(syntheticFloat32(dFF*dModel, 8))
	wDown := toBF16Bytes(syntheticFloat32(dModel*dFF, 9))

	// a non-empty resident prefix: basePos rows of K/V already in the cache.
	kCache0 := make([]byte, maxLen*kvDim*bf16Size)
	vCache0 := make([]byte, maxLen*kvDim*bf16Size)
	copy(kCache0, toBF16Bytes(syntheticFloat32(basePos*kvDim, 10)))
	copy(vCache0, toBF16Bytes(syntheticFloat32(basePos*kvDim, 11)))

	xs := toBF16Bytes(syntheticFloat32(K*dModel, 12))
	rowBytes := dModel * bf16Size

	// sequential reference: K DecodeStepKV calls over a copy of the cache.
	kSeq := append([]byte(nil), kCache0...)
	vSeq := append([]byte(nil), vCache0...)
	seqOut := make([]byte, K*rowBytes)
	for i := 0; i < K; i++ {
		h, err := DecodeStepKV(xs[i*rowBytes:(i+1)*rowBytes], attnNormW, wQ, wK, wV, wO, kSeq, vSeq, mlpNormW, wGate, wUp, wDown,
			dModel, nHeads, nKVHeads, headDim, maxLen, dFF, basePos+i, base, scale, eps)
		if err != nil {
			t.Fatalf("DecodeStepKV row %d: %v", i, err)
		}
		copy(seqOut[i*rowBytes:(i+1)*rowBytes], h)
	}

	// batched: one DecodeLayerBatchedKV over a fresh copy of the same prefix.
	kBat := append([]byte(nil), kCache0...)
	vBat := append([]byte(nil), vCache0...)
	batOut, err := DecodeLayerBatchedKV(xs, attnNormW, wQ, wK, wV, wO, kBat, vBat, mlpNormW, wGate, wUp, wDown,
		dModel, nHeads, nKVHeads, headDim, maxLen, dFF, basePos, K, base, scale, eps)
	if err != nil {
		t.Fatalf("DecodeLayerBatchedKV: %v", err)
	}

	eqBytes(t, "batched verify output vs K sequential DecodeStepKV", batOut, seqOut)
	eqBytes(t, "batched verify kCache vs sequential", kBat, kSeq)
	eqBytes(t, "batched verify vCache vs sequential", vBat, vSeq)
}

func TestDecodeLayerBatchedKVAllocationBudget(t *testing.T) {
	requireNativeRuntime(t)

	const dModel, nHeads, nKVHeads, headDim, maxLen, dFF, basePos, K = 256, 4, 2, 64, 32, 512, 5, 4
	const base, scale, eps = float32(10000), float32(1.0 / 8.0), float32(1e-6)
	qDim, kvDim := nHeads*headDim, nKVHeads*headDim
	attnNormW := toBF16Bytes(syntheticFloat32(dModel, 1))
	mlpNormW := toBF16Bytes(syntheticFloat32(dModel, 2))
	wQ := toBF16Bytes(syntheticFloat32(qDim*dModel, 3))
	wK := toBF16Bytes(syntheticFloat32(kvDim*dModel, 4))
	wV := toBF16Bytes(syntheticFloat32(kvDim*dModel, 5))
	wO := toBF16Bytes(syntheticFloat32(dModel*qDim, 6))
	wGate := toBF16Bytes(syntheticFloat32(dFF*dModel, 7))
	wUp := toBF16Bytes(syntheticFloat32(dFF*dModel, 8))
	wDown := toBF16Bytes(syntheticFloat32(dModel*dFF, 9))
	kCache := make([]byte, maxLen*kvDim*bf16Size)
	vCache := make([]byte, maxLen*kvDim*bf16Size)
	copy(kCache, toBF16Bytes(syntheticFloat32(basePos*kvDim, 10)))
	copy(vCache, toBF16Bytes(syntheticFloat32(basePos*kvDim, 11)))
	xs := toBF16Bytes(syntheticFloat32(K*dModel, 12))
	kWarm := append([]byte(nil), kCache...)
	vWarm := append([]byte(nil), vCache...)
	if _, err := DecodeLayerBatchedKV(xs, attnNormW, wQ, wK, wV, wO, kWarm, vWarm, mlpNormW, wGate, wUp, wDown, dModel, nHeads, nKVHeads, headDim, maxLen, dFF, basePos, K, base, scale, eps); err != nil {
		t.Fatalf("DecodeLayerBatchedKV warmup: %v", err)
	}

	var batchedErr error
	allocs := testing.AllocsPerRun(5, func() {
		kc := append([]byte(nil), kCache...)
		vc := append([]byte(nil), vCache...)
		_, batchedErr = DecodeLayerBatchedKV(xs, attnNormW, wQ, wK, wV, wO, kc, vc, mlpNormW, wGate, wUp, wDown, dModel, nHeads, nKVHeads, headDim, maxLen, dFF, basePos, K, base, scale, eps)
	})
	if batchedErr != nil {
		t.Fatalf("DecodeLayerBatchedKV: %v", batchedErr)
	}
	if allocs > 50 {
		t.Fatalf("DecodeLayerBatchedKV allocations = %.0f, want <= 50", allocs)
	}
}

func TestDecodeLayerBatchedKVUsesCallerInputBacking(t *testing.T) {
	requireNativeRuntime(t)

	const dModel, nHeads, nKVHeads, headDim, maxLen, dFF, basePos, K = 64, 1, 1, 64, 8, 128, 2, 2
	const base, scale, eps = float32(10000), float32(1.0 / 8.0), float32(1e-6)
	qDim, kvDim := nHeads*headDim, nKVHeads*headDim
	attnNormW := toBF16Bytes(syntheticFloat32(dModel, 1))
	mlpNormW := toBF16Bytes(syntheticFloat32(dModel, 2))
	wQ := toBF16Bytes(syntheticFloat32(qDim*dModel, 3))
	wK := toBF16Bytes(syntheticFloat32(kvDim*dModel, 4))
	wV := toBF16Bytes(syntheticFloat32(kvDim*dModel, 5))
	wO := toBF16Bytes(syntheticFloat32(dModel*qDim, 6))
	wGate := toBF16Bytes(syntheticFloat32(dFF*dModel, 7))
	wUp := toBF16Bytes(syntheticFloat32(dFF*dModel, 8))
	wDown := toBF16Bytes(syntheticFloat32(dModel*dFF, 9))
	kCache := make([]byte, maxLen*kvDim*bf16Size)
	vCache := make([]byte, maxLen*kvDim*bf16Size)
	copy(kCache, toBF16Bytes(syntheticFloat32(basePos*kvDim, 10)))
	copy(vCache, toBF16Bytes(syntheticFloat32(basePos*kvDim, 11)))
	xs := toBF16Bytes(syntheticFloat32(K*dModel, 12))
	scratch, err := getDecodeLayerBatchedScratch(dModel, qDim, kvDim, nHeads, dFF, K)
	if err != nil {
		t.Fatalf("get DecodeLayerBatched scratch: %v", err)
	}
	sentinel := bytes.Repeat([]byte{0xa5}, len(scratch.xs.bytes))
	copy(scratch.xs.bytes, sentinel)
	putDecodeLayerBatchedScratch(scratch)

	if _, err := DecodeLayerBatchedKV(xs, attnNormW, wQ, wK, wV, wO, kCache, vCache, mlpNormW, wGate, wUp, wDown, dModel, nHeads, nKVHeads, headDim, maxLen, dFF, basePos, K, base, scale, eps); err != nil {
		t.Fatalf("DecodeLayerBatchedKV: %v", err)
	}
	gotScratch, err := getDecodeLayerBatchedScratch(dModel, qDim, kvDim, nHeads, dFF, K)
	if err != nil {
		t.Fatalf("get DecodeLayerBatched scratch after call: %v", err)
	}
	defer putDecodeLayerBatchedScratch(gotScratch)
	if gotScratch != scratch {
		t.Fatal("DecodeLayerBatchedKV did not reuse the prepared scratch")
	}
	if !bytes.Equal(gotScratch.xs.bytes, sentinel) {
		t.Fatal("DecodeLayerBatchedKV copied input rows into pooled scratch instead of using caller backing")
	}
}

func TestDecodeLayerBatchedKVIntoReusesOutputBackingAndBypassesScratchOutput(t *testing.T) {
	requireNativeRuntime(t)

	const dModel, nHeads, nKVHeads, headDim, maxLen, dFF, basePos, K = 64, 1, 1, 64, 8, 128, 2, 2
	const base, scale, eps = float32(10000), float32(1.0 / 8.0), float32(1e-6)
	qDim, kvDim := nHeads*headDim, nKVHeads*headDim
	attnNormW := toBF16Bytes(syntheticFloat32(dModel, 1))
	mlpNormW := toBF16Bytes(syntheticFloat32(dModel, 2))
	wQ := toBF16Bytes(syntheticFloat32(qDim*dModel, 3))
	wK := toBF16Bytes(syntheticFloat32(kvDim*dModel, 4))
	wV := toBF16Bytes(syntheticFloat32(kvDim*dModel, 5))
	wO := toBF16Bytes(syntheticFloat32(dModel*qDim, 6))
	wGate := toBF16Bytes(syntheticFloat32(dFF*dModel, 7))
	wUp := toBF16Bytes(syntheticFloat32(dFF*dModel, 8))
	wDown := toBF16Bytes(syntheticFloat32(dModel*dFF, 9))
	kCache0 := make([]byte, maxLen*kvDim*bf16Size)
	vCache0 := make([]byte, maxLen*kvDim*bf16Size)
	copy(kCache0, toBF16Bytes(syntheticFloat32(basePos*kvDim, 10)))
	copy(vCache0, toBF16Bytes(syntheticFloat32(basePos*kvDim, 11)))
	xs := toBF16Bytes(syntheticFloat32(K*dModel, 12))

	kWant := append([]byte(nil), kCache0...)
	vWant := append([]byte(nil), vCache0...)
	want, err := DecodeLayerBatchedKV(xs, attnNormW, wQ, wK, wV, wO, kWant, vWant, mlpNormW, wGate, wUp, wDown, dModel, nHeads, nKVHeads, headDim, maxLen, dFF, basePos, K, base, scale, eps)
	if err != nil {
		t.Fatalf("DecodeLayerBatchedKV reference: %v", err)
	}

	out := make([]byte, K*dModel*bf16Size)
	outPtr := &out[0]
	scratch, err := getDecodeLayerBatchedScratch(dModel, qDim, kvDim, nHeads, dFF, K)
	if err != nil {
		t.Fatalf("get DecodeLayerBatched scratch: %v", err)
	}
	sentinel := bytes.Repeat([]byte{0x5a}, len(scratch.out.bytes))
	copy(scratch.out.bytes, sentinel)
	putDecodeLayerBatchedScratch(scratch)

	kGot := append([]byte(nil), kCache0...)
	vGot := append([]byte(nil), vCache0...)
	got, err := DecodeLayerBatchedKVInto(out, xs, attnNormW, wQ, wK, wV, wO, kGot, vGot, mlpNormW, wGate, wUp, wDown, dModel, nHeads, nKVHeads, headDim, maxLen, dFF, basePos, K, base, scale, eps)
	if err != nil {
		t.Fatalf("DecodeLayerBatchedKVInto: %v", err)
	}
	if len(got) != len(out) || &got[0] != outPtr {
		t.Fatal("DecodeLayerBatchedKVInto did not reuse caller-owned output backing")
	}
	eqBytes(t, "DecodeLayerBatchedKVInto output", got, want)
	eqBytes(t, "DecodeLayerBatchedKVInto kCache", kGot, kWant)
	eqBytes(t, "DecodeLayerBatchedKVInto vCache", vGot, vWant)

	scratch, err = getDecodeLayerBatchedScratch(dModel, qDim, kvDim, nHeads, dFF, K)
	if err != nil {
		t.Fatalf("get DecodeLayerBatched scratch after call: %v", err)
	}
	defer putDecodeLayerBatchedScratch(scratch)
	if !bytes.Equal(scratch.out.bytes, sentinel) {
		t.Fatal("DecodeLayerBatchedKVInto wrote through pooled scratch output instead of caller output")
	}
}

func TestDecodeLayerBatchedScratchPoolKeepsShapesResident(t *testing.T) {
	decodeLayerBatchedScratchPools = sync.Map{}
	t.Cleanup(func() { decodeLayerBatchedScratchPools = sync.Map{} })

	small := &decodeLayerBatchedScratch{dModel: 128, qDim: 128, kvDim: 64, nHeads: 2, dFF: 256, K: 2}
	large := &decodeLayerBatchedScratch{dModel: 256, qDim: 256, kvDim: 128, nHeads: 4, dFF: 512, K: 4}
	smallPool := decodeLayerBatchedScratchPoolFor(small.dModel, small.qDim, small.kvDim, small.nHeads, small.dFF, small.K)
	largePool := decodeLayerBatchedScratchPoolFor(large.dModel, large.qDim, large.kvDim, large.nHeads, large.dFF, large.K)
	if smallPool == largePool {
		t.Fatal("DecodeLayerBatched scratch reused one pool for distinct batched shapes")
	}

	putDecodeLayerBatchedScratch(small)
	putDecodeLayerBatchedScratch(large)
	forceNativeGC()
	forceNativeGC()

	if got := smallPool.Get(); got != small {
		t.Fatal("DecodeLayerBatched scratch pool evicted the small shape after using the larger shape")
	}
	if got := largePool.Get(); got != large {
		t.Fatal("DecodeLayerBatched scratch pool evicted the larger shape after reusing the small shape")
	}
}
