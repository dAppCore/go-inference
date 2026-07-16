// SPDX-Licence-Identifier: EUPL-1.2

package qwen3

import core "dappco.re/go"

func ExampleGatedDeltaConfig_QDim() {
	cfg := GatedDeltaConfig{KeyHeads: 2, HeadDim: 4}
	core.Println(cfg.QDim())
	// Output: 8
}

func ExampleGatedDeltaConfig_VDim() {
	cfg := GatedDeltaConfig{ValueHeads: 4, HeadDim: 4}
	core.Println(cfg.VDim())
	// Output: 16
}

func ExampleGatedDeltaConfig_ConvDim() {
	cfg := GatedDeltaConfig{KeyHeads: 2, ValueHeads: 4, HeadDim: 4}
	core.Println(cfg.ConvDim())
	// Output: 32
}

func ExampleGatedDeltaForwardF32() {
	cfg := GatedDeltaConfig{KeyHeads: 2, ValueHeads: 4, HeadDim: 4, ConvKernel: 3, Eps: 1e-5}
	const L, D = 3, 6
	w := mkGatedDeltaWeights(cfg, D)
	out, _, _, err := GatedDeltaForwardF32(gdSyn(L*D, 1), w, cfg, nil, nil, L, D)
	core.Println(err == nil, len(out))
	// Output: true 18
}

func ExampleGatedDeltaForwardScratchF32() {
	cfg := GatedDeltaConfig{KeyHeads: 2, ValueHeads: 4, HeadDim: 4, ConvKernel: 3, Eps: 1e-5}
	const L, D = 3, 6
	w := mkGatedDeltaWeights(cfg, D)
	sc := &GatedDeltaScratch{}
	out, _, _, err := GatedDeltaForwardScratchF32(gdSyn(L*D, 1), w, cfg, nil, nil, L, D, sc)
	core.Println(err == nil, len(out))
	// Output: true 18
}

func ExampleGatedDeltaForwardScratchNoProjF32() {
	cfg := GatedDeltaConfig{KeyHeads: 2, ValueHeads: 4, HeadDim: 4, ConvKernel: 3, Eps: 1e-5}
	const L, D = 3, 6
	w := mkGatedDeltaWeights(cfg, D)
	gated, vDim, _, _, err := GatedDeltaForwardScratchNoProjF32(gdSyn(L*D, 1), w, cfg, nil, nil, L, D, nil)
	core.Println(err == nil, vDim, len(gated))
	// Output: true 16 48
}

func ExampleGatedDeltaForwardScratchFromInputF32() {
	cfg := GatedDeltaConfig{KeyHeads: 2, ValueHeads: 4, HeadDim: 4, ConvKernel: 3, Eps: 1e-5}
	const L, D = 3, 6
	w := mkGatedDeltaWeights(cfg, D)
	x := gdSyn(L*D, 1)
	vDim, convDim := cfg.vDim(), cfg.convDim()
	qkv := matNT(x, w.InProjQKV, L, D, convDim)
	alpha := matNT(x, w.InProjA, L, D, cfg.ValueHeads)
	beta := matNT(x, w.InProjB, L, D, cfg.ValueHeads)
	zProj := matNT(x, w.InProjZ, L, D, vDim)
	gated, gotVDim, _, _, err := GatedDeltaForwardScratchFromInputF32(qkv, zProj, alpha, beta, w, cfg, nil, nil, L, D, nil)
	core.Println(err == nil, gotVDim, len(gated))
	// Output: true 16 48
}
