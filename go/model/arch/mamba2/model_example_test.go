// SPDX-Licence-Identifier: EUPL-1.2

package mamba2

import core "dappco.re/go"

func ExampleNewSession() {
	cfg := BlockConfig{NumHeads: 2, HeadDim: 8, StateDim: 8, NumGroups: 1, ConvKernel: 4, Eps: 1e-5}
	m := mkModel(cfg, 8, 32, 2)
	s := NewSession(m)
	core.Println(len(s.convState), len(s.ssmState))
	// Output: 2 2
}

func ExampleMambaSession_Forward() {
	cfg := BlockConfig{NumHeads: 2, HeadDim: 8, StateDim: 8, NumGroups: 1, ConvKernel: 4, Eps: 1e-5}
	m := mkModel(cfg, 8, 32, 1)
	h, err := NewSession(m).Forward([]int32{1, 2, 3})
	core.Println(err == nil, len(h))
	// Output: true 24
}

func ExampleMambaSession_Generate() {
	cfg := BlockConfig{NumHeads: 2, HeadDim: 8, StateDim: 8, NumGroups: 1, ConvKernel: 4, Eps: 1e-5}
	m := mkModel(cfg, 8, 32, 1)
	gen, err := NewSession(m).Generate([]int32{1, 2, 3}, 4, -1)
	core.Println(err == nil, len(gen))
	// Output: true 4
}
