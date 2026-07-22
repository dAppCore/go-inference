// SPDX-Licence-Identifier: EUPL-1.2

package rwkv7

import core "dappco.re/go"

func ExampleLoadRWKV7Model() {
	ts := mkCheckpoint(2, 4, 3, 8, 16, 2, 2, 2, 2, 32, 2)
	m, err := LoadRWKV7Model(ts, []byte(`{"norm_eps": 1e-5}`))
	core.Println(err == nil, len(m.Layers), m.D, m.Vocab)
	// Output: true 2 8 32
}
