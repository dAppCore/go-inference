// SPDX-Licence-Identifier: EUPL-1.2

package model

import core "dappco.re/go"

// ExampleTransformerConfig shows the neutral transformer core every architecture's config
// embeds: its JSON tags parse the shared HF config.json fields once, so every arch's
// config struct gets them by embedding TransformerConfig rather than re-declaring them.
func ExampleTransformerConfig() {
	var cfg TransformerConfig
	data := []byte(`{"hidden_size":2048,"num_hidden_layers":24,"num_attention_heads":8}`)
	if r := core.JSONUnmarshal(data, &cfg); !r.OK {
		return
	}
	core.Println(cfg.HiddenSize)
	core.Println(cfg.NumHiddenLayers)
	// Output:
	// 2048
	// 24
}
