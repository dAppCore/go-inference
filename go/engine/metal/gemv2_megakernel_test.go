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
	gemv2PSOOnce sync.Once
	gemv2PSO     metal.MTLComputePipelineState
	gemv2Err     error
)

func gemv2Pipeline() (metal.MTLComputePipelineState, error) {
	gemv2PSOOnce.Do(func() {
		if customLibrary == nil || customLibrary.GetID() == 0 {
			gemv2Err = core.NewError("gemv2: custom library unavailable")
			return
		}
		fn := customLibrary.NewFunctionWithName("lthn_gemv2_megakernel")
		if fn == nil || fn.GetID() == 0 {
			gemv2Err = core.NewError("gemv2: kernel not found")
			return
		}
		gemv2PSO, gemv2Err = device.NewComputePipelineStateWithFunctionError(fn)
	})
	return gemv2PSO, gemv2Err
}

func bf16BytesToF32(b []byte) []float32 {
	out := make([]float32, len(b)/2)
	for i := range out {
		out[i] = bf16ToF32(b[i*2], b[i*2+1])
	}
	return out
}

// hostGemvBF16 mirrors the megakernel's per-output f32-accumulate-then-round-bf16, same k order.
func hostGemvBF16(wF32, xF32 []float32, outDim, inDim int) []byte {
	out := make([]byte, outDim*bf16Size)
	for o := range outDim {
		var acc float32
		for k := range inDim {
			acc += wF32[o*inDim+k] * xF32[k]
		}
		h := f32ToBF16(acc)
		out[o*bf16Size] = byte(h)
		out[o*bf16Size+1] = byte(h >> 8)
	}
	return out
}

// TestGemv2Megakernel proves the foundational megakernel pattern: two dependent gemvs (out = W2·(W1·x)) in
// ONE dispatch with an in-kernel grid barrier between them must equal the host two-gemv reference. This
// validates the grid sync AND cross-threadgroup coherency (stage 2 reads the `mid` every stage-1 TG wrote)
// — the two primitives a full-layer decode megakernel rests on, with no external barrier between stages.
func TestGemv2Megakernel(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	if err := ensureInit(); err != nil {
		t.Skipf("device init: %v", err)
	}
	pso, err := gemv2Pipeline()
	if err != nil {
		t.Skipf("gemv2 pipeline: %v", err)
	}
	const inDim, midDim, outDim = 128, 256, 128
	const numTG, threadsPerTG = 64, 128
	const maxSpin = int32(1_000_000)

	xB := toBF16Bytes(syntheticFloat32(inDim, 3))
	w1B := toBF16Bytes(syntheticFloat32(midDim*inDim, 7))
	w2B := toBF16Bytes(syntheticFloat32(outDim*midDim, 11))

	// host reference (read the bf16-rounded operand values, same as the kernel sees)
	midRef := hostGemvBF16(bf16BytesToF32(w1B), bf16BytesToF32(xB), midDim, inDim)
	outRef := hostGemvBF16(bf16BytesToF32(w2B), bf16BytesToF32(midRef), outDim, midDim)

	out := make([]byte, outDim*bf16Size)
	withAutoreleasePool(func() {
		x, w1, w2 := sharedBytes(xB), sharedBytes(w1B), sharedBytes(w2B)
		mid := device.NewBufferWithLengthOptions(uint(midDim*bf16Size), metal.MTLResourceStorageModeShared)
		outBuf := device.NewBufferWithLengthOptions(uint(outDim*bf16Size), metal.MTLResourceStorageModeShared)
		arrive := device.NewBufferWithLengthOptions(4, metal.MTLResourceStorageModeShared)
		*(*uint32)(arrive.Contents()) = 0
		cb := queue.CommandBuffer()
		enc := cb.ComputeCommandEncoder()
		enc.SetComputePipelineState(pso)
		enc.SetBufferWithOffsetAtIndex(x, 0, 0)
		enc.SetBufferWithOffsetAtIndex(w1, 0, 1)
		enc.SetBufferWithOffsetAtIndex(w2, 0, 2)
		enc.SetBufferWithOffsetAtIndex(mid, 0, 3)
		enc.SetBufferWithOffsetAtIndex(outBuf, 0, 4)
		enc.SetBufferWithOffsetAtIndex(arrive, 0, 5)
		setEncInt32(enc, inDim, 6)
		setEncInt32(enc, midDim, 7)
		setEncInt32(enc, outDim, 8)
		setEncInt32(enc, numTG, 9)
		setEncInt32(enc, maxSpin, 10)
		enc.DispatchThreadgroupsThreadsPerThreadgroup(
			metal.MTLSize{Width: numTG, Height: 1, Depth: 1},
			metal.MTLSize{Width: threadsPerTG, Height: 1, Depth: 1},
		)
		enc.EndEncoding()
		cb.Commit()
		cb.WaitUntilCompleted()
		copy(out, unsafe.Slice((*byte)(outBuf.Contents()), outDim*bf16Size))
	})

	if cos := cosineBF16(out, outRef); cos < 0.9999 {
		t.Fatalf("gemv2 megakernel cosine=%.6f vs host two-gemv reference — grid sync / coherency broken", cos)
	}
	t.Logf("gemv2 megakernel (grid-barrier between two gemvs) matches host reference — pattern works")
}
