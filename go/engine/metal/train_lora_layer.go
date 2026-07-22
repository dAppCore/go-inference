// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import core "dappco.re/go"

// train_lora_layer.go is the HOST-SIDE correctness reference for the per-LAYER projection LoRA backward
// (#31) — the maths the trainer refusal (validateHeadLoRATargets) guards until it is wired. A layer
// projection with a LoRA is trained through its EFFECTIVE weight W' = W + (alpha/rank)·B·A: the layer
// forward/backward runs with W' substituted, and the factor gradients fall out of the projection's
// weight gradient (dA = scaling·Bᵀ·dW', dB = scaling·dW'·Aᵀ — the chain rule through W' = W + s·B·A).
// The two small primitives (LoRAEffectiveWeightF32, LoRAFactorGradsF32) are pure host f32 with f64
// accumulation — deterministic, runtime-free, finite-difference-gated. The composed layer backward
// (LayerProjLoRABackwardF32) chains them through the existing gradient-checked block backwards
// (train_backward.go), which model the SIMPLIFIED gemma layer (pre-norm GQA attention + pre-norm
// gated-GELU MLP; no PLE / QK-norm / post-norm) — so this is the reference for the follow-up that
// builds the real-arch layer backward, NOT a trainer path: wiring it into LoRATrainer as-is would
// train silently-wrong gradients on every real (PLE / QK-norm / post-norm) base the trainer opens,
// exactly the failure shape #31 exists to kill.

// Canonical per-layer projection target names — the inference.LoRAConfig.TargetKeys vocabulary
// (the ecosystem's adapter naming; emitted identifiers stay canonical).
const (
	ProjQ    = "q_proj"
	ProjK    = "k_proj"
	ProjV    = "v_proj"
	ProjO    = "o_proj"
	ProjGate = "gate_proj"
	ProjUp   = "up_proj"
	ProjDown = "down_proj"
)

// TrainLayerF32 bundles one simplified gemma layer's frozen f32 weights and geometry for the host-side
// training reference — the widened (bf16→f32) form of DecodeLayerWeights plus the dims the block
// forwards/backwards need. The projection weights are row-major [out,in], the storage every projection
// uses (out_features × in_features).
type TrainLayerF32 struct {
	AttnNormW []float32 // [DModel]
	WQ        []float32 // [Heads·HeadDim, DModel]
	WK, WV    []float32 // [KVHeads·HeadDim, DModel]
	WO        []float32 // [DModel, Heads·HeadDim]
	MLPNormW  []float32 // [DModel]
	WGate     []float32 // [DFF, DModel]
	WUp       []float32 // [DFF, DModel]
	WDown     []float32 // [DModel, DFF]

	T, DModel, DFF           int // rows (tokens), hidden, feed-forward width
	Heads, KVHeads, HeadDim  int // GQA head geometry (Heads % KVHeads == 0)
	RotaryDim                int // rotary width ≤ HeadDim
	RopeBase, AttnScale, Eps float32
	Causal                   bool
}

// projDims returns the [out,in] dimensions of a canonical projection target within L, or an error for
// a target this simplified layer does not carry — the one place the target vocabulary is interpreted.
func (L *TrainLayerF32) projDims(target string) (out, in int, err error) {
	switch target {
	case ProjQ:
		return L.Heads * L.HeadDim, L.DModel, nil
	case ProjK, ProjV:
		return L.KVHeads * L.HeadDim, L.DModel, nil
	case ProjO:
		return L.DModel, L.Heads * L.HeadDim, nil
	case ProjGate, ProjUp:
		return L.DFF, L.DModel, nil
	case ProjDown:
		return L.DModel, L.DFF, nil
	}
	return 0, 0, core.NewError(core.Concat("native.TrainLayerF32: unknown projection target ", core.Sprintf("%q", target),
		" (supported: ", ProjQ, " ", ProjK, " ", ProjV, " ", ProjO, " ", ProjGate, " ", ProjUp, " ", ProjDown, ")"))
}

// projWeight returns the frozen weight slice of a canonical projection target within L (the slice the
// LoRA's effective weight substitutes).
func (L *TrainLayerF32) projWeight(target string) []float32 {
	switch target {
	case ProjQ:
		return L.WQ
	case ProjK:
		return L.WK
	case ProjV:
		return L.WV
	case ProjO:
		return L.WO
	case ProjGate:
		return L.WGate
	case ProjUp:
		return L.WUp
	case ProjDown:
		return L.WDown
	}
	return nil
}

// LoRAEffectiveWeightF32 folds a LoRA into its frozen projection weight: eff[out,in] = w + scaling·(B·A),
// with A [rank,in] and B [out,rank]. Pure host f32 with f64 accumulation — the deterministic correctness
// reference (B starts zero, so an untrained adapter folds to w exactly). The layer forward under a
// projection LoRA is the plain layer forward with this effective weight substituted.
func LoRAEffectiveWeightF32(w, a, b []float32, out, in, rank int, scaling float32) ([]float32, error) {
	if len(w) != out*in || len(a) != rank*in || len(b) != out*rank {
		return nil, core.NewError("native.LoRAEffectiveWeightF32: w[out,in]/A[rank,in]/B[out,rank] size mismatch")
	}
	eff := make([]float32, out*in)
	for o := range out {
		for i := range in {
			acc := float64(w[o*in+i])
			for r := range rank {
				acc += float64(scaling) * float64(b[o*rank+r]) * float64(a[r*in+i])
			}
			eff[o*in+i] = float32(acc)
		}
	}
	return eff, nil
}

// LoRAFactorGradsF32 maps a projection's weight gradient onto the LoRA factors — the chain rule through
// the effective weight W' = W + scaling·(B·A):
//
//	dA [rank,in]  = scaling · Bᵀ · dW
//	dB [out,rank] = scaling · dW · Aᵀ
//
// dW [out,in] is the projection weight gradient a block backward returns (MLPBlockGrads.DWDown,
// AttnBlockGrads.DWQ, …); dA/dB are what AdamW steps. Pure host f32 with f64 accumulation — the
// deterministic correctness reference, finite-difference-gated in train_lora_layer_test.go.
func LoRAFactorGradsF32(dW, a, b []float32, out, in, rank int, scaling float32) (dA, dB []float32, err error) {
	if len(dW) != out*in || len(a) != rank*in || len(b) != out*rank {
		return nil, nil, core.NewError("native.LoRAFactorGradsF32: dW[out,in]/A[rank,in]/B[out,rank] size mismatch")
	}
	dA = make([]float32, rank*in)
	for r := range rank {
		for i := range in {
			var acc float64
			for o := range out {
				acc += float64(b[o*rank+r]) * float64(dW[o*in+i])
			}
			dA[r*in+i] = float32(float64(scaling) * acc)
		}
	}
	dB = make([]float32, out*rank)
	for o := range out {
		for r := range rank {
			var acc float64
			for i := range in {
				acc += float64(dW[o*in+i]) * float64(a[r*in+i])
			}
			dB[o*rank+r] = float32(float64(scaling) * acc)
		}
	}
	return dA, dB, nil
}

// LayerProjLoRABackwardF32 is the host-side backward of ONE simplified gemma layer with a LoRA on the
// named projection target: given the layer's frozen input h [T,DModel] (the residual stream from
// ForwardCaptureHiddens) and the upstream gradient dout [T,DModel] (from the layer above, or the
// head+final-norm backward for the top layer), it substitutes the LoRA's effective weight into the
// layer, backpropagates through the MLP and attention blocks (train_backward.go), and returns the
// factor gradients plus dH — the gradient to the layer below, so a full-stack chain keeps walking.
// The corresponding forward is MultiHeadAttnBlockForwardF32 + MLPBlockForwardF32 with the same
// effective weight substituted. f32; finite-difference-gated per target in train_lora_layer_test.go.
//
// This is the CORRECTNESS REFERENCE for the per-layer projection backward (#31), not a trainer path:
// the blocks model the simplified layer (no PLE / QK-norm / post-norm), so the trainer keeps refusing
// per-layer targets until the real-arch layer backward exists (the follow-up this reference gates).
func LayerProjLoRABackwardF32(dout, h []float32, L *TrainLayerF32, target string, a, b []float32, rank int, scaling float32) (dA, dB, dH []float32, err error) {
	if L == nil {
		return nil, nil, nil, core.NewError("native.LayerProjLoRABackwardF32: nil layer")
	}
	out, in, err := L.projDims(target)
	if err != nil {
		return nil, nil, nil, err
	}
	if len(dout) != L.T*L.DModel || len(h) != L.T*L.DModel {
		return nil, nil, nil, core.NewError("native.LayerProjLoRABackwardF32: dout/h must be [T,DModel]")
	}
	eff, err := LoRAEffectiveWeightF32(L.projWeight(target), a, b, out, in, rank, scaling)
	if err != nil {
		return nil, nil, nil, err
	}

	// The layer with the effective weight substituted at the target projection.
	wQ, wK, wV, wO := L.WQ, L.WK, L.WV, L.WO
	wGate, wUp, wDown := L.WGate, L.WUp, L.WDown
	switch target {
	case ProjQ:
		wQ = eff
	case ProjK:
		wK = eff
	case ProjV:
		wV = eff
	case ProjO:
		wO = eff
	case ProjGate:
		wGate = eff
	case ProjUp:
		wUp = eff
	case ProjDown:
		wDown = eff
	}

	// Forward the attention half to recover the MLP block's input (the only intermediate the block
	// backwards do not recompute themselves), then backpropagate MLP → attention → dH.
	attnOut, err := MultiHeadAttnBlockForwardF32(h, L.AttnNormW, wQ, wK, wV, wO, L.T, L.DModel, L.Heads, L.KVHeads, L.HeadDim, L.RotaryDim, L.RopeBase, L.AttnScale, L.Eps, L.Causal)
	if err != nil {
		return nil, nil, nil, err
	}
	mg, err := MLPBlockBackwardF32(dout, attnOut, L.MLPNormW, wGate, wUp, wDown, L.T, L.DModel, L.DFF, L.Eps)
	if err != nil {
		return nil, nil, nil, err
	}
	ag, err := MultiHeadAttnBlockBackwardF32(mg.DH, h, L.AttnNormW, wQ, wK, wV, wO, L.T, L.DModel, L.Heads, L.KVHeads, L.HeadDim, L.RotaryDim, L.RopeBase, L.AttnScale, L.Eps, L.Causal)
	if err != nil {
		return nil, nil, nil, err
	}

	var dW []float32
	switch target {
	case ProjQ:
		dW = ag.DWQ
	case ProjK:
		dW = ag.DWK
	case ProjV:
		dW = ag.DWV
	case ProjO:
		dW = ag.DWO
	case ProjGate:
		dW = mg.DWGate
	case ProjUp:
		dW = mg.DWUp
	case ProjDown:
		dW = mg.DWDown
	}
	dA, dB, err = LoRAFactorGradsF32(dW, a, b, out, in, rank, scaling)
	if err != nil {
		return nil, nil, nil, err
	}
	return dA, dB, ag.DH, nil
}
