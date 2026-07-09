// SPDX-Licence-Identifier: EUPL-1.2

package model

import (
	core "dappco.re/go"

	"dappco.re/go/inference/model/safetensors"
)

// ExampleStandardWeightNames shows the canonical HF weight layout: an architecture with
// the plain naming uses it as-is; one that differs overrides only the fields that do.
func ExampleStandardWeightNames() {
	names := StandardWeightNames()
	core.Println(names.Embed)
	core.Println(names.LayerPrefix)
	// Output:
	// model.embed_tokens
	// model.layers.%d
}

// ExampleAssemble shows the generic, arch.Layer-driven weight build: a tensor set plus the
// derived Arch and the arch's WeightNames become a LoadedModel — the same loop serving
// every architecture, quant width, and layer count.
func ExampleAssemble() {
	tensors := map[string]safetensors.Tensor{
		"embed.weight":             {Shape: []int{8, 4}, Data: make([]byte, 8*4*2)},
		"norm.weight":              {Shape: []int{4}, Data: make([]byte, 4*2)},
		"layer.0.attn_norm.weight": {Shape: []int{4}, Data: make([]byte, 4*2)},
		"layer.0.q.weight":         {Shape: []int{4, 4}, Data: make([]byte, 4*4*2)},
		"layer.0.k.weight":         {Shape: []int{4, 4}, Data: make([]byte, 4*4*2)},
		"layer.0.o.weight":         {Shape: []int{4, 4}, Data: make([]byte, 4*4*2)},
		"layer.0.mlp_norm.weight":  {Shape: []int{4}, Data: make([]byte, 4*2)},
		"layer.0.gate.weight":      {Shape: []int{8, 4}, Data: make([]byte, 8*4*2)},
		"layer.0.up.weight":        {Shape: []int{8, 4}, Data: make([]byte, 8*4*2)},
		"layer.0.down.weight":      {Shape: []int{4, 8}, Data: make([]byte, 4*8*2)},
	}
	arch := Arch{Hidden: 4, Heads: 2, FF: 8, Layer: []LayerSpec{{CacheIndex: 0, HeadDim: 2, KVHeads: 2}}}
	names := WeightNames{
		Embed: "embed", FinalNorm: "norm.weight", LayerPrefix: "layer.%d",
		AttnNorm: ".attn_norm.weight", Q: ".q", K: ".k", O: ".o",
		MLPNorm: ".mlp_norm.weight", Gate: ".gate", Up: ".up", Down: ".down",
	}
	m, err := Assemble(tensors, arch, names)
	if err != nil {
		return
	}
	core.Println(len(m.Layers))
	core.Println(m.Tied()) // no lm_head weight supplied → tied to Embed
	// Output:
	// 1
	// true
}
