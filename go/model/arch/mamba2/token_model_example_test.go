// SPDX-Licence-Identifier: EUPL-1.2

package mamba2

import core "dappco.re/go"

func ExampleNewTokenModel() {
	cfg := BlockConfig{NumHeads: 2, HeadDim: 8, StateDim: 8, NumGroups: 1, ConvKernel: 4, Eps: 1e-5}
	m := mkModel(cfg, 8, 32, 1)
	tm := NewTokenModel(m)
	core.Println(tm.Vocab())
	// Output: 32
}

func ExampleMambaTokenModel_Vocab() {
	cfg := BlockConfig{NumHeads: 2, HeadDim: 8, StateDim: 8, NumGroups: 1, ConvKernel: 4, Eps: 1e-5}
	m := mkModel(cfg, 8, 32, 1)
	core.Println(NewTokenModel(m).Vocab())
	// Output: 32
}

func ExampleMambaTokenModel_Embed() {
	cfg := BlockConfig{NumHeads: 2, HeadDim: 8, StateDim: 8, NumGroups: 1, ConvKernel: 4, Eps: 1e-5}
	m := mkModel(cfg, 8, 32, 1)
	emb, err := NewTokenModel(m).Embed(5)
	core.Println(err == nil, len(emb))
	// Output: true 16
}

func ExampleMambaTokenModel_Head() {
	cfg := BlockConfig{NumHeads: 2, HeadDim: 8, StateDim: 8, NumGroups: 1, ConvKernel: 4, Eps: 1e-5}
	m := mkModel(cfg, 8, 32, 1)
	tm := NewTokenModel(m)
	hidden := f32ToBF16Bytes(syn(m.D, 7))
	logits, err := tm.Head(hidden)
	core.Println(err == nil, len(logits))
	// Output: true 64
}

func ExampleMambaTokenModel_DecodeForward() {
	cfg := BlockConfig{NumHeads: 2, HeadDim: 8, StateDim: 8, NumGroups: 1, ConvKernel: 4, Eps: 1e-5}
	m := mkModel(cfg, 8, 32, 1)
	tm := NewTokenModel(m)
	e, _ := tm.Embed(3)
	out, err := tm.DecodeForward([][]byte{e})
	core.Println(err == nil, len(out))
	// Output: true 1
}

func ExampleMambaTokenModel_OpenSession() {
	cfg := BlockConfig{NumHeads: 2, HeadDim: 8, StateDim: 8, NumGroups: 1, ConvKernel: 4, Eps: 1e-5}
	m := mkModel(cfg, 8, 32, 1)
	tm := NewTokenModel(m)
	st, err := tm.OpenSession()
	core.Println(err == nil, st != nil)
	// Output: true true
}
