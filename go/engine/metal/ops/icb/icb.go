// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

// Package icb owns the Metal indirect-command-buffer recording primitive.
// Architecture packages supply the operations to record; this package owns the
// architecture-neutral buffer shape and allocation.
package icb

import (
	"github.com/tmc/apple/foundation"
	"github.com/tmc/apple/metal"
)

const commandTypes = metal.MTLIndirectCommandTypeConcurrentDispatch | metal.MTLIndirectCommandTypeConcurrentDispatchThreads

// Recorder is an architecture-neutral Metal indirect command buffer and the
// range covering all commands allocated in it.
type Recorder struct {
	Buffer metal.MTLIndirectCommandBuffer
	Range  foundation.NSRange
}

func commandRange(count uint) foundation.NSRange {
	return foundation.NSRange{Location: 0, Length: count}
}

// New allocates a shared Metal indirect command buffer ready for a caller to
// record count compute commands. The caller supplies the largest kernel-buffer
// binding count used by its operation plan.
func New(device metal.MTLDevice, count, maxKernelBufferBindCount uint) Recorder {
	desc := metal.NewMTLIndirectCommandBufferDescriptor()
	desc.SetCommandTypes(commandTypes)
	desc.SetInheritBuffers(false)
	desc.SetInheritPipelineState(false)
	desc.SetMaxKernelBufferBindCount(maxKernelBufferBindCount)
	return Recorder{
		Buffer: device.NewIndirectCommandBufferWithDescriptorMaxCommandCountOptions(desc, count, metal.MTLResourceStorageModeShared),
		Range:  commandRange(count),
	}
}
