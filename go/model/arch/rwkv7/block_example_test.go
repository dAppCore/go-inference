// SPDX-Licence-Identifier: EUPL-1.2

package rwkv7

import core "dappco.re/go"

func ExampleBlockForwardF32() {
	cfg := BlockConfig{NumHeads: 2, KeyDim: 4, ValueDim: 6}
	const L, D = 5, 8
	w := mkBlockWeights(cfg, D)
	out, _, err := BlockForwardF32(syn(L*D, 1), w, cfg, nil, L, D)
	core.Println(err == nil, len(out))
	// Output: true 40
}

func ExampleBlockForwardScratchF32() {
	cfg := BlockConfig{NumHeads: 2, KeyDim: 4, ValueDim: 6}
	const L, D = 5, 8
	w := mkBlockWeights(cfg, D)
	sc := &BlockScratch{}
	out, _, err := BlockForwardScratchF32(syn(L*D, 1), w, cfg, nil, L, D, sc)
	core.Println(err == nil, len(out))
	// Output: true 40
}

func ExampleBlockForwardScratchNoProjF32() {
	cfg := BlockConfig{NumHeads: 2, KeyDim: 4, ValueDim: 6}
	const L, D = 5, 8
	w := mkBlockWeights(cfg, D)
	o, hv, _, err := BlockForwardScratchNoProjF32(syn(L*D, 1), w, cfg, nil, L, D, nil)
	core.Println(err == nil, hv, len(o))
	// Output: true 12 60
}

func ExampleBlockForwardScratchFromInputF32() {
	cfg := BlockConfig{NumHeads: 2, KeyDim: 4, ValueDim: 6}
	const L, D = 5, 8
	w := mkBlockWeights(cfg, D)
	x := syn(L*D, 1)
	hk, hv := cfg.hk(), cfg.hv()
	r := matNT(x, w.RProj, L, D, hk)
	wd := matNT(x, w.WProj, L, D, hk)
	k := matNT(x, w.KProj, L, D, hk)
	v := matNT(x, w.VProj, L, D, hv)
	a := matNT(x, w.AProj, L, D, hk)
	b := matNT(x, w.BProj, L, D, hk)
	o, gotHv, _, err := BlockForwardScratchFromInputF32(r, wd, k, v, a, b, w, cfg, nil, L, D, nil)
	core.Println(err == nil, gotHv, len(o))
	// Output: true 12 60
}
