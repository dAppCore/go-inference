// SPDX-Licence-Identifier: EUPL-1.2

package qwen3

import (
	"testing"

	"dappco.re/go/inference/model"
)

// The register bench baselines the reactive-loader entry qwen3's init() installs (AX-11):
// the ArchSpec parser resolved through model.LookupArch("qwen3") — the per-load path that
// turns a config.json into a model.ArchConfig. This exercises exactly what register.go wires
// up (the JSON parse into a qwen3 Config), so its allocation is the config-parse cost the
// loader pays per qwen3 checkpoint. init() ran when this test binary loaded, so the registry
// is populated. Synthetic config bytes — no checkpoint read.

// BenchmarkQwen3RegisteredParse — the registered parser via the arch registry: LookupArch
// dispatch + the spec's JSON unmarshal into a Config. The unmarshal is the allocation story.
func BenchmarkQwen3RegisteredParse(b *testing.B) {
	spec, ok := model.LookupArch("qwen3")
	if !ok {
		b.Fatal("qwen3 arch not registered — init() did not run")
	}
	data := []byte(`{"model_type":"qwen3","hidden_size":2048,"num_hidden_layers":36,` +
		`"num_attention_heads":16,"num_key_value_heads":2,"head_dim":128,` +
		`"intermediate_size":11008,"vocab_size":151936,"rms_norm_eps":1e-06,"rope_theta":1000000}`)
	b.SetBytes(int64(len(data)))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := spec.Parse(data); err != nil {
			b.Fatal(err)
		}
	}
}
