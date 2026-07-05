// SPDX-Licence-Identifier: EUPL-1.2

package lora

import "testing"

// BenchmarkParseAdapterConfig exercises the adapter_config.json parse +
// alias normalisation path. This is metadata-only decode; the LoRA delta
// apply/merge is an engine-side concern this package never performs (AX-11
// evidence, not a load-path hot loop).
func BenchmarkParseAdapterConfig(b *testing.B) {
	data := []byte(`{"r":8,"lora_alpha":16,"target_modules":["q_proj","k_proj","v_proj","o_proj"],"num_layers":32}`)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := ParseAdapterConfig(data); err != nil {
			b.Fatalf("ParseAdapterConfig() error = %v", err)
		}
	}
}
