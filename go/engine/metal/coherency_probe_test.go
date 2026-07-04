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
	coherencyPSOOnce sync.Once
	coherencyPSO     metal.MTLComputePipelineState
	coherencyErr     error
)

func coherencyPipeline() (metal.MTLComputePipelineState, error) {
	coherencyPSOOnce.Do(func() {
		if customLibrary == nil || customLibrary.GetID() == 0 {
			coherencyErr = core.NewError("coherency: custom library unavailable")
			return
		}
		fn := customLibrary.NewFunctionWithName("lthn_coherency_probe")
		if fn == nil || fn.GetID() == 0 {
			coherencyErr = core.NewError("coherency: kernel lthn_coherency_probe not found")
			return
		}
		coherencyPSO, coherencyErr = device.NewComputePipelineStateWithFunctionError(fn)
	})
	return coherencyPSO, coherencyErr
}

// TestCrossTGCoherencyPlainVsAtomic is the megakernel-viability find-out: does Metal give reliable cross-
// DISTANT-threadgroup producer→consumer data visibility? Metal has no release/acquire ordering (compile-
// proven: only memory_order_relaxed), so the question is whether ATOMIC (L2-coherent) handoff data works
// where PLAIN (L1-cacheable) data goes stale — the failure mode of the grid-barrier FFN megakernel. Each TG
// writes its tag to slot[tgid] both plain and atomic; after a grid barrier TG 0 reads EVERY slot. If atomic
// reads all numTG tags but plain doesn't, the megakernel's cross-TG dependency IS expressible on Metal by
// routing handoff data through atomics. If atomic is also stale, the cross-TG handoff is genuinely Metal-
// blocked and the path is partial-fusion + streaming + the direct-OS dispatch.
func TestCrossTGCoherencyPlainVsAtomic(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	if err := ensureInit(); err != nil {
		t.Skipf("device init: %v", err)
	}
	pso, err := coherencyPipeline()
	if err != nil {
		t.Skipf("coherency pipeline: %v", err)
	}
	const numTG, threadsPerTG = 64, 128
	const maxSpin = int32(1_000_000)
	withAutoreleasePool(func() {
		plain := device.NewBufferWithLengthOptions(uint(numTG*4), metal.MTLResourceStorageModeShared)
		atom := device.NewBufferWithLengthOptions(uint(numTG*4), metal.MTLResourceStorageModeShared)
		arrive := device.NewBufferWithLengthOptions(4, metal.MTLResourceStorageModeShared)
		result := device.NewBufferWithLengthOptions(8, metal.MTLResourceStorageModeShared)
		for i, s := 0, unsafe.Slice((*uint32)(plain.Contents()), numTG); i < numTG; i++ {
			s[i] = 0
		}
		for i, s := 0, unsafe.Slice((*uint32)(atom.Contents()), numTG); i < numTG; i++ {
			s[i] = 0
		}
		*(*uint32)(arrive.Contents()) = 0
		unsafe.Slice((*uint32)(result.Contents()), 2)[0] = 0
		unsafe.Slice((*uint32)(result.Contents()), 2)[1] = 0

		cb := queue.CommandBuffer()
		enc := cb.ComputeCommandEncoder()
		enc.SetComputePipelineState(pso)
		enc.SetBufferWithOffsetAtIndex(plain, 0, 0)
		enc.SetBufferWithOffsetAtIndex(atom, 0, 1)
		enc.SetBufferWithOffsetAtIndex(arrive, 0, 2)
		enc.SetBufferWithOffsetAtIndex(result, 0, 3)
		setEncInt32(enc, numTG, 4)
		setEncInt32(enc, maxSpin, 5)
		enc.DispatchThreadgroupsThreadsPerThreadgroup(
			metal.MTLSize{Width: numTG, Height: 1, Depth: 1},
			metal.MTLSize{Width: threadsPerTG, Height: 1, Depth: 1},
		)
		enc.EndEncoding()
		cb.Commit()
		cb.WaitUntilCompleted()
		res := unsafe.Slice((*uint32)(result.Contents()), 2)
		t.Logf("cross-TG visibility over %d distant TGs: plain=%d/%d  atomic=%d/%d", numTG, res[0], numTG, res[1], numTG)
		if res[1] == numTG && res[0] < numTG {
			t.Logf("ATOMIC handoff is COHERENT where plain is stale — the megakernel's cross-TG dependency IS expressible on Metal")
		} else if res[1] < numTG {
			t.Logf("ATOMIC also stale (%d/%d) — cross-TG handoff genuinely Metal-blocked; path is partial-fusion + streaming + direct-OS dispatch", res[1], numTG)
		}
	})
}
