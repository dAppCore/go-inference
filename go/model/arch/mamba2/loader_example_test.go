// SPDX-Licence-Identifier: EUPL-1.2

package mamba2

import (
	core "dappco.re/go"
	"dappco.re/go/inference/model/safetensors"
)

// ExampleLoadMambaModel builds a minimal synthetic 1-layer Mamba-2 checkpoint (the standard HF
// backbone.* names/shapes) and loads it, deriving the SSD geometry from the weight shapes alone.
func ExampleLoadMambaModel() {
	const H, headDim, N, K = 2, 8, 8, 4
	const dInner = H * headDim
	const convDim = dInner + 2*N
	const projOut = 2*dInner + 2*N + H
	const D, vocab = 8, 32

	ts := map[string]safetensors.Tensor{
		"backbone.embeddings.weight":              bf16Tensor(syn(vocab*D, 1), vocab, D),
		"backbone.norm_f.weight":                  bf16Tensor(syn(D, 2), D),
		"backbone.layers.0.norm.weight":           bf16Tensor(syn(D, 3), D),
		"backbone.layers.0.mixer.in_proj.weight":  bf16Tensor(syn(projOut*D, 4), projOut, D),
		"backbone.layers.0.mixer.conv1d.weight":   bf16Tensor(syn(convDim*K, 5), convDim, 1, K),
		"backbone.layers.0.mixer.A_log":           bf16Tensor(syn(H, 6), H),
		"backbone.layers.0.mixer.out_proj.weight": bf16Tensor(syn(D*dInner, 7), D, dInner),
	}
	m, err := LoadMambaModel(ts, 1e-5)
	core.Println(err == nil, m.Cfg.NumHeads, m.Cfg.HeadDim, len(m.Layers))
	// Output: true 2 8 1
}
