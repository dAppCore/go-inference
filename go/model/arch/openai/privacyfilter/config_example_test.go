// SPDX-Licence-Identifier: EUPL-1.2

package privacyfilter

import core "dappco.re/go"

func ExampleParseConfig() {
	cfg, err := ParseConfig([]byte(`{"model_type":"openai_privacy_filter","hidden_size":640,"num_hidden_layers":8,"num_local_experts":128,"num_experts_per_tok":4}`))
	core.Println(err == nil, cfg.ModelType, cfg.HiddenSize)
	// Output: true openai_privacy_filter 640
}

func ExampleConfig_Arch() {
	cfg := Config{NumLocalExperts: 128, NumExpertsPerTok: 4, ID2Label: map[string]string{"0": "O"}}
	_, err := cfg.Arch()
	core.Println(err != nil)
	// Output: true
}
