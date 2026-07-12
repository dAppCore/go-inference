// SPDX-Licence-Identifier: EUPL-1.2

package bert_test

import (
	"dappco.re/go/inference/model/bert"
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

// ExampleNewTokenizer builds a WordPiece tokeniser from a vocab blob and frames
// a sentence as [CLS] … [SEP] ids.
func ExampleNewTokenizer() {
	vocab := "[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nthe\nfox\n"
	tk, err := bert.NewTokenizer([]byte(vocab), true)
	if err != nil {
		panic(err)
	}
	ids := tk.Encode("The fox")
	// ids == [2 5 6 3] -> [CLS] the fox [SEP]
	println(len(ids))
	// Output:
}
