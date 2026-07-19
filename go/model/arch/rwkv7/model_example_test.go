// SPDX-Licence-Identifier: EUPL-1.2

package rwkv7

import core "dappco.re/go"

func ExampleNewSession() {
	cfg := BlockConfig{NumHeads: 2, KeyDim: 4, ValueDim: 3}
	m := mkRWKV7Model(cfg, 8, 16, 32, 2)
	s := NewSession(m)
	core.Println(len(s.wkv), len(s.shift1), len(s.shift2))
	// Output: 2 2 2
}

func ExampleRWKV7Session_Forward() {
	cfg := BlockConfig{NumHeads: 2, KeyDim: 4, ValueDim: 3}
	m := mkRWKV7Model(cfg, 8, 16, 32, 1)
	h, err := NewSession(m).Forward([]int32{1, 2, 3})
	core.Println(err == nil, len(h))
	// Output: true 24
}

func ExampleRWKV7Session_Generate() {
	cfg := BlockConfig{NumHeads: 2, KeyDim: 4, ValueDim: 3}
	m := mkRWKV7Model(cfg, 8, 16, 32, 1)
	gen, err := NewSession(m).Generate([]int32{1, 2, 3}, 4, -1)
	core.Println(err == nil, len(gen))
	// Output: true 4
}
