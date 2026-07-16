// SPDX-Licence-Identifier: EUPL-1.2

package deepseek

import core "dappco.re/go"

func ExampleConfig_QHeadDim() {
	cfg := Config{QKNoPEHeadDim: 128, QKRoPEHeadDim: 64}
	core.Println(cfg.QHeadDim())
	// Output: 192
}

func ExampleConfig_KVHeadDim() {
	cfg := Config{QKNoPEHeadDim: 128, QKRoPEHeadDim: 64}
	core.Println(cfg.KVHeadDim())
	// Output: 192
}

func ExampleConfig_Validate() {
	cfg := Config{
		HiddenSize: 8, NumHiddenLayers: 1, NumAttentionHeads: 2, VocabSize: 32,
		KVLoRARank: 4, QKNoPEHeadDim: 2, QKRoPEHeadDim: 2, ValueHeadDim: 2,
		NumRoutedExperts: 2, NumExpertsPerTok: 1, MoEIntermediateSize: 4,
	}
	core.Println(cfg.Validate() == nil)
	// Output: true
}

func ExampleConfig_InferFromWeights() {
	cfg := Config{HiddenSize: 8}
	cfg.InferFromWeights(nil) // no-op: DeepSeek declares its MLA geometry
	core.Println(cfg.HiddenSize)
	// Output: 8
}

func ExampleConfig_Arch() {
	cfg := Config{
		HiddenSize: 8, NumHiddenLayers: 1, NumAttentionHeads: 2, VocabSize: 32,
		KVLoRARank: 4, QKNoPEHeadDim: 2, QKRoPEHeadDim: 2, ValueHeadDim: 2,
		NumRoutedExperts: 2, NumExpertsPerTok: 1, MoEIntermediateSize: 4,
	}
	_, err := cfg.Arch() // MLA requires a separate attention implementation; Arch always refuses
	core.Println(err != nil)
	// Output: true
}
