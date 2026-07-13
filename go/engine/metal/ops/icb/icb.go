// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

// Package icb owns the Metal indirect-command-buffer recording primitive.
// Architecture packages supply the operations to record; this package owns the
// architecture-neutral buffer shape and allocation.
//
// # Declared-ops seam (the #57 ops-homes migration)
//
// The goal of the ops-homes slices is that model families DECLARE their op selections on the
// neutral model contract (model.Arch / model.LayerSpec) and the engine BINDS them, so the ICB
// record/replay machinery that lands here carries no architecture-specific knowledge. This ledger
// tracks which replay-state items are already family-declared versus still inferred in the engine,
// so each slice starts from truth.
//
// DECLARED BY THE FAMILY (read from model.Arch / model.LayerSpec; the engine only binds):
//   - attention SDPA scale, embedding scale (Arch.AttnScale / Arch.EmbedScale)
//   - per-layer attention geometry (LayerSpec.HeadDim / LayerSpec.KVHeads)
//   - value normalisation, sliding window, RoPE bases/dims, MoE gating, per-layer-input dims
//   - K==V value-projection sharing (LayerSpec.AttentionKEqV) — slice 2: model.Assemble resolves
//     it from the checkpoint (a layer with no v_proj has its value ride the key projection); the
//     record boundary (recordArchICB* in package native) reads the declared selection instead of an
//     arch-supplied v-proj-index hook. A hand-built caller that does not declare falls back to
//     weight presence at the record boundary, never inside the neutral core.
//   - norm-op selection (LayerSpec.Attention{Q,K}Norm / Post{Attn,FF}Norm) — slice 3:
//     model.Assemble resolves each from checkpoint weight presence, so every family that flows
//     through Assemble (gemma4, qwen-class QK-norm, exaone4/gptneox sandwich norms) declares for
//     free; the record boundary reads the declared selections with the same buffer-presence
//     self-heal for hand-built callers. Declaring a norm without its weight is a caller bug.
//
// STILL ENGINE-INFERRED (candidates for later slices — inferred from weight-buffer presence at the
// record boundary, shared across families):
//   - the fixed per-layer op LAYOUT (opsPerLayer) and the per-layer-input record hooks
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
