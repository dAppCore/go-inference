// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import "math"

// train_softcap.go — the final-logit soft-cap in the TRAINER's host head path (#42 follow-through).
// Serving applies logits = cap·tanh(raw/cap) in the head encoder (arch_session's newHeadEncoder);
// the trainer's own head forward (MatMulF32NT → cross-entropy) previously skipped it, so a capped
// model (gemma4 E2B/E4B declare final_logit_softcapping 30) trained against a loss whose
// distribution disagreed with what serving samples — the reason validatePerLayerLoRAShape refused
// SoftCap outright. Capping the trainer forward and scaling the CE gradient by the cap's derivative
// closes that gap for BOTH the head-adapter and per-layer paths.
//
// Backward: with t = tanh(raw/cap) and capped = cap·t, d(capped)/d(raw) = 1 − t² = 1 − (capped/cap)²
// — the factor is computable from the CAPPED value alone, so the forward can cap in place and the
// backward needs no stash of the raw logits.

// softcapForwardF32 applies logits = cap·tanh(logits/cap) in place. cap == 0 means no cap (the
// uncapped arches' no-op), matching model.Arch.SoftCap's zero-value contract.
func softcapForwardF32(logits []float32, cap float32) {
	if cap == 0 {
		return
	}
	c := float64(cap)
	for i, v := range logits {
		logits[i] = float32(c * math.Tanh(float64(v)/c))
	}
}

// softcapBackwardScaleF32 scales dLogits in place by the cap's derivative 1 − (capped/cap)², where
// capped is the FORWARD-capped logits slice (softcapForwardF32's output). cap == 0 is the no-op.
func softcapBackwardScaleF32(dLogits, capped []float32, cap float32) {
	if cap == 0 {
		return
	}
	c := float64(cap)
	for i := range dLogits {
		t := float64(capped[i]) / c
		dLogits[i] = float32(float64(dLogits[i]) * (1 - t*t))
	}
}
