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
	gridsyncPSOOnce sync.Once
	gridsyncPSO     metal.MTLComputePipelineState
	gridsyncErr     error
)

func gridsyncPipeline() (metal.MTLComputePipelineState, error) {
	gridsyncPSOOnce.Do(func() {
		if customLibrary == nil || customLibrary.GetID() == 0 {
			gridsyncErr = core.NewError("gridsync: custom library unavailable")
			return
		}
		fn := customLibrary.NewFunctionWithName("lthn_gridsync_probe")
		if fn == nil || fn.GetID() == 0 {
			gridsyncErr = core.NewError("gridsync: kernel not found")
			return
		}
		gridsyncPSO, gridsyncErr = device.NewComputePipelineStateWithFunctionError(fn)
	})
	return gridsyncPSO, gridsyncErr
}

// TestGridSyncFeasibility probes whether Apple Silicon can co-schedule + grid-barrier a given threadgroup
// count — the make-or-break primitive for a full-layer decode megakernel. For each TG count it dispatches
// that many threadgroups, each spinning (bounded) on an atomic arrival counter; if EVERY threadgroup sees
// the counter reach numTG, the grid barrier completed (all co-resident). The largest count that completes
// is the ceiling on how parallel a megakernel's gemvs can be while still grid-syncing in one dispatch.
func TestGridSyncFeasibility(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	if os.Getenv("LEM_GRIDSYNC_PROBE") == "" {
		t.Skip("set LEM_GRIDSYNC_PROBE=1 to run the grid-sync feasibility probe (spins; ~10s)")
	}
	if err := ensureInit(); err != nil {
		t.Skipf("device init: %v", err)
	}
	pso, err := gridsyncPipeline()
	if err != nil {
		t.Skipf("gridsync pipeline: %v", err)
	}
	const threadsPerTG = 256
	const maxSpin = uint32(1_000_000)
	maxOK := 0
	for _, numTG := range []int{32, 64, 128, 256, 512, 1024, 2048, 4096} {
		ok := false
		withAutoreleasePool(func() {
			counter := device.NewBufferWithLengthOptions(4, metal.MTLResourceStorageModeShared)
			*(*uint32)(counter.Contents()) = 0
			out := device.NewBufferWithLengthOptions(uint(numTG*4), metal.MTLResourceStorageModeShared)
			cb := queue.CommandBuffer()
			enc := cb.ComputeCommandEncoder()
			enc.SetComputePipelineState(pso)
			enc.SetBufferWithOffsetAtIndex(counter, 0, 0)
			enc.SetBufferWithOffsetAtIndex(out, 0, 1)
			setEncInt32(enc, int32(numTG), 2)
			setEncInt32(enc, int32(maxSpin), 3)
			enc.DispatchThreadgroupsThreadsPerThreadgroup(
				metal.MTLSize{Width: uint(numTG), Height: 1, Depth: 1},
				metal.MTLSize{Width: threadsPerTG, Height: 1, Depth: 1},
			)
			enc.EndEncoding()
			cb.Commit()
			cb.WaitUntilCompleted()
			vals := unsafe.Slice((*uint32)(out.Contents()), numTG)
			completed := 0
			minSeen := uint32(numTG)
			for _, v := range vals {
				if v >= uint32(numTG) {
					completed++
				}
				if v < minSeen {
					minSeen = v
				}
			}
			ok = completed == numTG
			t.Logf("numTG=%-5d threads/TG=%d -> %d/%d threadgroups reached the barrier (min counter seen=%d) %s",
				numTG, threadsPerTG, completed, numTG, minSeen, map[bool]string{true: "GRID-SYNC OK", false: "WOULD DEADLOCK"}[ok])
		})
		if ok {
			maxOK = numTG
		} else {
			break
		}
	}
	t.Logf("max grid-syncable threadgroups @ %d threads/TG: %d", threadsPerTG, maxOK)
}
