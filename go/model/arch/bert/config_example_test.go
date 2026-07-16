// SPDX-Licence-Identifier: EUPL-1.2

package bert_test

import (
	core "dappco.re/go"
	"dappco.re/go/inference/model/arch/bert"
)

// ExampleParseConfig decodes a BertModel config.json and reports the derived
// per-head width used by the attention forward.
func ExampleParseConfig() {
	cfg, err := bert.ParseConfig([]byte(`{
		"model_type": "bert",
		"hidden_size": 384,
		"num_hidden_layers": 12,
		"num_attention_heads": 12,
		"intermediate_size": 1536,
		"vocab_size": 30522,
		"max_position_embeddings": 512,
		"type_vocab_size": 2
	}`))
	if err != nil {
		panic(err)
	}
	println(cfg.HiddenSize, cfg.HeadDim())
	// Output:
}

func ExampleConfig_HeadDim() {
	cfg := bert.Config{HiddenSize: 384, NumAttentionHeads: 12}
	core.Println(cfg.HeadDim())
	// Output: 32
}

func ExampleConfig_IsCrossEncoder() {
	cfg := bert.Config{NumLabels: 1, Architectures: []string{"BertForSequenceClassification"}}
	core.Println(cfg.IsCrossEncoder())
	// Output: true
}
