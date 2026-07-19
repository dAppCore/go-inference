// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"os"

	core "dappco.re/go"
)

// ExampleLoRAEffectiveWeightF32 folds a rank-1 LoRA into a 1×2 base weight: eff = w + scaling·(B·A).
// Pure host maths — deterministic output.
func ExampleLoRAEffectiveWeightF32() {
	eff, err := LoRAEffectiveWeightF32([]float32{1, 2}, []float32{1, 1}, []float32{1}, 1, 2, 1, 2)
	if err != nil {
		return
	}
	core.Println(eff)
	// Output: [3 4]
}

// ExampleLoRAFactorGradsF32 maps a projection weight gradient onto the LoRA factors: with dW = [1 1],
// A = [1 0], B = [2] and scaling 1, dA = Bᵀ·dW = [2 2] and dB = dW·Aᵀ = [1]. Pure host maths.
func ExampleLoRAFactorGradsF32() {
	dA, dB, err := LoRAFactorGradsF32([]float32{1, 1}, []float32{1, 0}, []float32{2}, 1, 2, 1, 1)
	if err != nil {
		return
	}
	core.Println(dA, dB)
	// Output: [2 2] [1]
}

// ExampleLayerProjLoRABackwardF32 shows the intended follow-up usage: one simplified layer's frozen
// input plus the upstream gradient in, the LoRA factor gradients (for the optimiser) and dH (for the
// layer below) out. The block backwards dispatch steel GEMMs, so the example guards on the runtime
// (no Output directive — exercised under the FD gate in train_lora_layer_test.go).
func ExampleLayerProjLoRABackwardF32() {
	if os.Getenv(MetallibPathEnv) == "" {
		return
	}
	const T, dModel, dFF, rank = 2, 4, 6, 1
	L := &TrainLayerF32{
		AttnNormW: make([]float32, dModel), WQ: make([]float32, dModel*dModel), WK: make([]float32, dModel*dModel),
		WV: make([]float32, dModel*dModel), WO: make([]float32, dModel*dModel),
		MLPNormW: make([]float32, dModel), WGate: make([]float32, dFF*dModel), WUp: make([]float32, dFF*dModel),
		WDown: make([]float32, dModel*dFF),
		T:     T, DModel: dModel, DFF: dFF, Heads: 1, KVHeads: 1, HeadDim: dModel, RotaryDim: dModel,
		RopeBase: 10000, AttnScale: 0.5, Eps: 1e-5, Causal: true,
	}
	dout := make([]float32, T*dModel)
	h := make([]float32, T*dModel)
	a := make([]float32, rank*dFF)    // A [rank,in] for down_proj: in = dFF
	b := make([]float32, dModel*rank) // B [out,rank]: out = dModel
	dA, dB, dH, err := LayerProjLoRABackwardF32(dout, h, L, ProjDown, a, b, rank, 16)
	if err != nil {
		return
	}
	_ = dA // steps the optimiser
	_ = dB
	_ = dH // chains to the layer below
}
