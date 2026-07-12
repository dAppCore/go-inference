// SPDX-Licence-Identifier: EUPL-1.2

package qwen2

import "testing"

var sinkConfig *Config

func BenchmarkParseConfig(b *testing.B) {
	data := []byte(`{"model_type":"qwen2","hidden_size":896,"intermediate_size":4864,"num_hidden_layers":24,"num_attention_heads":14,"num_key_value_heads":2,"vocab_size":151936}`)
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		sinkConfig, _ = ParseConfig(data)
	}
}

func BenchmarkConfig_Arch(b *testing.B) {
	cfg := Config{HiddenSize: 896, IntermediateSize: 4864, NumHiddenLayers: 24, NumAttentionHeads: 14, NumKeyValueHeads: 2, VocabSize: 151936}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_, _ = cfg.Arch()
	}
}
