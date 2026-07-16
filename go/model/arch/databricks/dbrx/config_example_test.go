// SPDX-Licence-Identifier: EUPL-1.2

package dbrx_test

import (
	core "dappco.re/go"
	"dappco.re/go/inference/model/arch/databricks/dbrx"
)

func ExampleConfig() {
	cfg := dbrx.Config{DModel: 8, Heads: 2, Layers: 1, VocabSize: 32, Attention: dbrx.AttentionConfig{KVHeads: 1}, FFN: dbrx.FFNConfig{HiddenSize: 12, Experts: 4, TopK: 2}}
	arch, _ := cfg.Arch()
	core.Println(arch.Hidden, arch.Experts, arch.TopK, arch.NormaliseMoETopK)
	// Output: 8 4 2 false
}

func ExampleConfig_Arch() {
	cfg := dbrx.Config{DModel: 8, Heads: 2, Layers: 1, VocabSize: 32, Attention: dbrx.AttentionConfig{KVHeads: 1}, FFN: dbrx.FFNConfig{HiddenSize: 12, Experts: 4, TopK: 2}}
	arch, _ := cfg.Arch()
	core.Println(arch.HeadDim, arch.KVHeads)
	// Output: 4 1
}

func ExampleConfig_InferFromWeights() {
	cfg := dbrx.Config{DModel: 8}
	cfg.InferFromWeights(nil) // no-op: DBRX declares its geometry
	core.Println(cfg.DModel)
	// Output: 8
}
