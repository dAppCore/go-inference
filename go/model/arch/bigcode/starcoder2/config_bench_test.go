// SPDX-Licence-Identifier: EUPL-1.2

package starcoder2

import "testing"

var sinkConfig *Config

func BenchmarkParseConfig(b *testing.B) {
	data := []byte(`{"model_type":"starcoder2","hidden_size":3072,"intermediate_size":12288,"max_position_embeddings":16384,"num_attention_heads":24,"num_hidden_layers":30,"num_key_value_heads":2,"sliding_window":4096,"vocab_size":49152}`)
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		cfg, err := ParseConfig(data)
		if err != nil {
			b.Fatal(err)
		}
		sinkConfig = cfg
	}
}
