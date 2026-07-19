// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import core "dappco.re/go"

// exampleRealLayer builds the smallest well-formed real layer (single head, full rotary, global
// window) — the shared fixture of the package examples below. Pure host: no runtime needed.
func exampleRealLayer() *RealTrainLayerF32 {
	const T, dModel, dFF = 2, 4, 6
	return &RealTrainLayerF32{
		AttnNormW: onesF32(dModel), WQ: make([]float32, dModel*dModel), WK: make([]float32, dModel*dModel),
		WV: make([]float32, dModel*dModel), WO: make([]float32, dModel*dModel),
		MLPNormW: onesF32(dModel), WGate: make([]float32, dFF*dModel), WUp: make([]float32, dFF*dModel),
		WDown: make([]float32, dModel*dFF),
		T:     T, DModel: dModel, DFF: dFF, Heads: 1, KVHeads: 1, HeadDim: dModel,
		RopeInvFreq: realRopeInvFreqs(dModel, 10000), RopePairHalf: dModel / 2, RopeScale: 1,
		AttnScale: 0.5, Eps: 1e-5,
	}
}

// ExampleRealLayerForwardF32 runs the host-pure real-layer forward: with all-zero projections every
// branch contributes nothing, so the layer output is the input unchanged — the residual identity.
func ExampleRealLayerForwardF32() {
	L := exampleRealLayer()
	h := []float32{1, 2, 3, 4, 5, 6, 7, 8} // [T=2, DModel=4]
	out, err := RealLayerForwardF32(h, L)
	if err != nil {
		return
	}
	core.Println(out)
	// Output: [1 2 3 4 5 6 7 8]
}

// ExampleRealLayerProjLoRABackwardF32 is the per-layer training seam: one real layer's frozen input
// plus the upstream gradient in, the LoRA factor gradients (for the optimiser) and dH (for the
// layer below) out. With a zero adapter (B = 0) on zero frozen weights, dA is exactly zero — the
// untrained adapter starts as the identity. Pure host, no runtime.
func ExampleRealLayerProjLoRABackwardF32() {
	L := exampleRealLayer()
	const rank = 1
	dout := onesF32(L.T * L.DModel)
	h := onesF32(L.T * L.DModel)
	a := onesF32(rank * L.DFF)           // A [rank,in] for down_proj: in = DFF
	b := make([]float32, L.DModel*rank)  // B [out,rank] starts zero → the adapter is a no-op
	dA, _, dH, err := RealLayerProjLoRABackwardF32(dout, h, L, ProjDown, a, b, rank, 16)
	if err != nil {
		return
	}
	core.Println(len(dA), len(dH), dA[0])
	// Output: 6 8 0
}
