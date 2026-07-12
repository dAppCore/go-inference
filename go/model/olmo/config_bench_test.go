// SPDX-Licence-Identifier: EUPL-1.2

package olmo

import "testing"

var sinkConfig any

func BenchmarkParseConfig(b *testing.B) {
	data := []byte(`{"model_type":"olmo2","hidden_size":2048,"intermediate_size":8192,"num_hidden_layers":16,"num_attention_heads":16,"vocab_size":100352}`)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		cfg, err := ParseConfig(data)
		if err != nil {
			b.Fatal(err)
		}
		sinkConfig = cfg
	}
}

func BenchmarkConfigArch(b *testing.B) {
	cfg := Config{ModelType: "olmo2", HiddenSize: 2048, IntermediateSize: 8192, NumHiddenLayers: 16, NumAttentionHeads: 16, VocabSize: 100352}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		a, err := cfg.Arch()
		if err != nil {
			b.Fatal(err)
		}
		sinkConfig = a
	}
}
