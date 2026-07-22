// SPDX-Licence-Identifier: EUPL-1.2

package mamba2

import core "dappco.re/go"

func ExampleBlockForwardF32() {
	cfg := BlockConfig{NumHeads: 2, HeadDim: 8, StateDim: 8, NumGroups: 1, ConvKernel: 4, Eps: 1e-5}
	const L, D = 5, 8
	w := mkBlockWeights(cfg, D)
	out, _, _, err := BlockForwardF32(syn(L*D, 1), w, cfg, nil, nil, L, D)
	core.Println(err == nil, len(out))
	// Output: true 40
}

func ExampleBlockForwardScratchF32() {
	cfg := BlockConfig{NumHeads: 2, HeadDim: 8, StateDim: 8, NumGroups: 1, ConvKernel: 4, Eps: 1e-5}
	const L, D = 5, 8
	w := mkBlockWeights(cfg, D)
	sc := &BlockScratch{}
	out, _, _, err := BlockForwardScratchF32(syn(L*D, 1), w, cfg, nil, nil, L, D, sc)
	core.Println(err == nil, len(out))
	// Output: true 40
}

func ExampleBlockForwardScratchNoProjF32() {
	cfg := BlockConfig{NumHeads: 2, HeadDim: 8, StateDim: 8, NumGroups: 1, ConvKernel: 4, Eps: 1e-5}
	const L, D = 5, 8
	w := mkBlockWeights(cfg, D)
	gated, dInner, _, _, err := BlockForwardScratchNoProjF32(syn(L*D, 1), w, cfg, nil, nil, L, D, nil)
	core.Println(err == nil, dInner, len(gated))
	// Output: true 16 80
}

func ExampleBlockForwardScratchFromInputF32() {
	cfg := BlockConfig{NumHeads: 2, HeadDim: 8, StateDim: 8, NumGroups: 1, ConvKernel: 4, Eps: 1e-5}
	const L, D = 5, 8
	w := mkBlockWeights(cfg, D)
	x := syn(L*D, 1)
	proj := matNT(x, w.InProj, L, D, cfg.projDim())
	gated, dInner, _, _, err := BlockForwardScratchFromInputF32(proj, w, cfg, nil, nil, L, D, nil)
	core.Println(err == nil, dInner, len(gated))
	// Output: true 16 80
}
