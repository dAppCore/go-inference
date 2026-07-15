// SPDX-Licence-Identifier: EUPL-1.2

package granitemoe_test

import (
	core "dappco.re/go"
	"dappco.re/go/inference/model/arch/ibm-granite/granitemoe"
)

func ExampleParseConfig() {
	r := granitemoe.ParseConfig([]byte(`{"model_type":"granitemoe","num_local_experts":32,"num_experts_per_tok":8}`))
	cfg := r.Value.(*granitemoe.Config)
	core.Println(cfg.ModelType, cfg.NumLocalExperts, cfg.NumExpertsPerTok)
	// Output: granitemoe 32 8
}

func ExampleConfig_Arch() {
	cfg := granitemoe.Config{ModelType: "granitemoe", HiddenSize: 8, IntermediateSize: 4, NumHiddenLayers: 1, NumAttentionHeads: 2, NumKeyValueHeads: 1, NumLocalExperts: 4, NumExpertsPerTok: 2, VocabSize: 32, RMSNormEps: 1e-5, RopeTheta: 10000, LogitsScaling: 6, ResidualMultiplier: .22, EmbeddingMultiplier: 12, AttentionMultiplier: .125}
	arch, err := cfg.Arch()
	core.Println(err == nil, arch.Experts, arch.TopK, arch.NormaliseMoETopK)
	// Output: true 4 2 true
}
