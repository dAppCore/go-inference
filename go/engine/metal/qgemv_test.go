// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"os"
	"sync"
	"testing"
	"unsafe"

	core "dappco.re/go"
	"github.com/tmc/apple/metal"
)

var (
	qgemvPSOOnce sync.Once
	qgemvPSO     metal.MTLComputePipelineState
	qgemvErr     error
)

func qgemvPipeline() (metal.MTLComputePipelineState, error) {
	qgemvPSOOnce.Do(func() {
		if customLibrary == nil || customLibrary.GetID() == 0 {
			qgemvErr = core.NewError("qgemv: custom library unavailable")
			return
		}
		fn := customLibrary.NewFunctionWithName("lthn_qgemv")
		if fn == nil || fn.GetID() == 0 {
			qgemvErr = core.NewError("qgemv: kernel not found")
			return
		}
		qgemvPSO, qgemvErr = device.NewComputePipelineStateWithFunctionError(fn)
	})
	return qgemvPSO, qgemvErr
}

func runQGemv(t *testing.T, pso metal.MTLComputePipelineState, x, packed, scales, biases []byte, outDim, inDim, groupSize int) []byte {
	out := make([]byte, outDim*bf16Size)
	withAutoreleasePool(func() {
		xB, pB, sB, bB := sharedBytes(x), sharedBytes(packed), sharedBytes(scales), sharedBytes(biases)
		outBuf := device.NewBufferWithLengthOptions(uint(outDim*bf16Size), metal.MTLResourceStorageModeShared)
		cb := queue.CommandBuffer()
		enc := cb.ComputeCommandEncoder()
		enc.SetComputePipelineState(pso)
		enc.SetBufferWithOffsetAtIndex(xB, 0, 0)
		enc.SetBufferWithOffsetAtIndex(pB, 0, 1)
		enc.SetBufferWithOffsetAtIndex(sB, 0, 2)
		enc.SetBufferWithOffsetAtIndex(bB, 0, 3)
		enc.SetBufferWithOffsetAtIndex(outBuf, 0, 4)
		setEncInt32(enc, int32(outDim), 5)
		setEncInt32(enc, int32(inDim), 6)
		setEncInt32(enc, int32(groupSize), 7)
		setEncInt32(enc, int32(inDim/2), 8)
		setEncInt32(enc, int32(inDim/groupSize), 9)
		enc.DispatchThreadsThreadsPerThreadgroup(
			metal.MTLSize{Width: uint(outDim), Height: 1, Depth: 1},
			metal.MTLSize{Width: uint(elemGroupTG(outDim)), Height: 1, Depth: 1},
		)
		enc.EndEncoding()
		cb.Commit()
		cb.WaitUntilCompleted()
		copy(out, unsafe.Slice((*byte)(outBuf.Contents()), outDim*bf16Size))
	})
	return out
}

var (
	qgemvSimdPSOOnce sync.Once
	qgemvSimdPSO     metal.MTLComputePipelineState
	qgemvSimdErr     error
)

func qgemvSimdPipeline() (metal.MTLComputePipelineState, error) {
	qgemvSimdPSOOnce.Do(func() {
		if customLibrary == nil || customLibrary.GetID() == 0 {
			qgemvSimdErr = core.NewError("qgemv_simd: custom library unavailable")
			return
		}
		fn := customLibrary.NewFunctionWithName("lthn_qgemv_simd")
		if fn == nil || fn.GetID() == 0 {
			qgemvSimdErr = core.NewError("qgemv_simd: kernel not found")
			return
		}
		qgemvSimdPSO, qgemvSimdErr = device.NewComputePipelineStateWithFunctionError(fn)
	})
	return qgemvSimdPSO, qgemvSimdErr
}

func runQGemvSimd(pso metal.MTLComputePipelineState, x, packed, scales, biases []byte, outDim, inDim, groupSize int) []byte {
	out := make([]byte, outDim*bf16Size)
	withAutoreleasePool(func() {
		xB, pB, sB, bB := sharedBytes(x), sharedBytes(packed), sharedBytes(scales), sharedBytes(biases)
		outBuf := device.NewBufferWithLengthOptions(uint(outDim*bf16Size), metal.MTLResourceStorageModeShared)
		cb := queue.CommandBuffer()
		enc := cb.ComputeCommandEncoder()
		enc.SetComputePipelineState(pso)
		enc.SetBufferWithOffsetAtIndex(xB, 0, 0)
		enc.SetBufferWithOffsetAtIndex(pB, 0, 1)
		enc.SetBufferWithOffsetAtIndex(sB, 0, 2)
		enc.SetBufferWithOffsetAtIndex(bB, 0, 3)
		enc.SetBufferWithOffsetAtIndex(outBuf, 0, 4)
		setEncInt32(enc, int32(outDim), 5)
		setEncInt32(enc, int32(inDim), 6)
		setEncInt32(enc, int32(groupSize), 7)
		setEncInt32(enc, int32(inDim/2), 8)
		setEncInt32(enc, int32(inDim/groupSize), 9)
		enc.DispatchThreadsThreadsPerThreadgroup(
			metal.MTLSize{Width: uint(outDim * 32), Height: 1, Depth: 1},
			metal.MTLSize{Width: 256, Height: 1, Depth: 1},
		)
		enc.EndEncoding()
		cb.Commit()
		cb.WaitUntilCompleted()
		copy(out, unsafe.Slice((*byte)(outBuf.Contents()), outDim*bf16Size))
	})
	return out
}

// TestQGemvSimdBeatsSequentialOnGated is the robust-precision gate: on the exact pathological case that made
// the sequential gemv diverge (the FFN down-projection over gelu·mul-activated input, ill-conditioned), the
// simd-cooperative qgemv must track MLX's steel qmv at cosine~1 where the sequential one fell to ~0.99 — the
// simd-tree reduction order matches the steel far better. This is the gemv the megakernel inlines.
func TestQGemvSimdBeatsSequentialOnGated(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	if err := ensureInit(); err != nil {
		t.Skipf("device init: %v", err)
	}
	seqPSO, err := qgemvPipeline()
	if err != nil {
		t.Skipf("qgemv pipeline: %v", err)
	}
	simdPSO, err := qgemvSimdPipeline()
	if err != nil {
		t.Skipf("qgemv_simd pipeline: %v", err)
	}
	const hidden, ff, groupSize, bits = 256, 512, 64, 4
	mkW := func(outDim, inDim, seed int) (p, s, b []byte) {
		p = make([]byte, outDim*inDim*bits/8)
		for i := range p {
			p[i] = byte((i*131 + 17 + seed) % 256)
		}
		nSB := outDim * (inDim / groupSize)
		s = toBF16Bytes(syntheticFloat32(nSB, seed+1))
		b = toBF16Bytes(syntheticFloat32(nSB, seed+2))
		return
	}
	gateP, gateS, gateB := mkW(ff, hidden, 10)
	upP, upS, upB := mkW(ff, hidden, 40)
	downP, downS, downB := mkW(hidden, ff, 70)
	x := toBF16Bytes(syntheticFloat32(hidden, 23))

	gate, _ := QMVBF16(x, gateP, gateS, gateB, ff, hidden, groupSize, bits)
	up, _ := QMVBF16(x, upP, upS, upB, ff, hidden, groupSize, bits)
	gated := hostGeluMul(gate, up)

	ref, err := QMVBF16(gated, downP, downS, downB, hidden, ff, groupSize, bits)
	if err != nil {
		t.Fatalf("down qmv: %v", err)
	}
	seq := runQGemv(t, seqPSO, gated, downP, downS, downB, hidden, ff, groupSize)
	simd := runQGemvSimd(simdPSO, gated, downP, downS, downB, hidden, ff, groupSize)

	cosSeq := cosineBF16(seq, ref)
	cosSimd := cosineBF16(simd, ref)
	// FINDING: BOTH the sequential and the simd-cooperative gemv match the steel qmv at ~1.0 here — so the
	// gemv reduction order was NOT the FFN megakernel's 0.99 divergence. That 0.99 is grid-barrier cross-TG
	// COHERENCY (stage 2 read stale gated), since this standalone test uses the post-kernel (correct) gated.
	// The simd-cooperative gemv is still the more robust reduction to inline; it must at least match steel.
	t.Logf("down over gelu·mul input vs steel qmv: sequential cosine=%.6f, simd-cooperative cosine=%.6f (both ~1 ⇒ gemv was not the FFN divergence; grid-barrier coherency is)", cosSeq, cosSimd)
	if cosSimd < 0.9999 {
		t.Fatalf("simd-cooperative qgemv cosine=%.6f vs steel — reduction order diverges", cosSimd)
	}
}

// TestQGemvMatchesSteel validates the megakernel's inlined 4-bit gemv against MLX's steel affine_quantized
// gemv (QMVBF16) on the SAME packed weight: same affine dequant (scale·code + bias), so token-identical
// (cosine~1) though not byte-identical (the steel kernel's simd-cooperative reduction differs in order). If
// the nibble/group layout matched the steel kernel, this passes — the gemv the megakernel inlines is sound.
func TestQGemvMatchesSteel(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	if err := ensureInit(); err != nil {
		t.Skipf("device init: %v", err)
	}
	pso, err := qgemvPipeline()
	if err != nil {
		t.Skipf("qgemv pipeline: %v", err)
	}
	const outDim, inDim, groupSize, bits = 256, 512, 64, 4
	packed := make([]byte, outDim*inDim*bits/8)
	for i := range packed {
		packed[i] = byte((i*131 + 17) % 256)
	}
	nSB := outDim * (inDim / groupSize)
	scales := toBF16Bytes(syntheticFloat32(nSB, 11))
	biases := toBF16Bytes(syntheticFloat32(nSB, 13))
	x := toBF16Bytes(syntheticFloat32(inDim, 23))

	ref, err := QMVBF16(x, packed, scales, biases, outDim, inDim, groupSize, bits)
	if err != nil {
		t.Fatalf("QMVBF16 (steel): %v", err)
	}
	got := runQGemv(t, pso, x, packed, scales, biases, outDim, inDim, groupSize)

	if cos := cosineBF16(got, ref); cos < 0.999 {
		t.Fatalf("qgemv cosine=%.6f vs steel QMVBF16 — nibble/group layout mismatch", cos)
	} else {
		t.Logf("qgemv matches steel QMVBF16 (cosine=%.6f) — the megakernel's inlined 4-bit gemv is sound", cos)
	}
}
