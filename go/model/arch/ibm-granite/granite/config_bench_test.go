// SPDX-Licence-Identifier: EUPL-1.2

package granite

import (
	"testing"

	core "dappco.re/go"
)

var sinkConfig core.Result
var sinkArchError error

func BenchmarkParseConfig(b *testing.B) {
	data := []byte(`{"model_type":"granite","hidden_size":2048,"intermediate_size":8192,"num_hidden_layers":40,"num_attention_heads":32,"num_key_value_heads":8,"vocab_size":49152,"rms_norm_eps":0.00001,"rope_theta":10000000,"logits_scaling":8,"residual_multiplier":0.22,"embedding_multiplier":12,"attention_multiplier":0.015625}`)
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		sinkConfig = ParseConfig(data)
	}
}

func BenchmarkConfig_Arch(b *testing.B) {
	r := ParseConfig([]byte(`{"model_type":"granite","hidden_size":2048,"intermediate_size":8192,"num_hidden_layers":40,"num_attention_heads":32,"num_key_value_heads":8,"vocab_size":49152,"rms_norm_eps":0.00001,"rope_theta":10000000,"logits_scaling":8,"residual_multiplier":0.22,"embedding_multiplier":12,"attention_multiplier":0.015625}`))
	cfg := r.Value.(*Config)
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_, sinkArchError = cfg.Arch()
	}
}
