// SPDX-Licence-Identifier: EUPL-1.2

package model

import "testing"

// The quant-config benches baseline the mlx quantization block (AX-11). UnmarshalJSON is
// the per-load parse — the default group_size/bits plus a per-module override map for
// mixed-precision packs — so its allocation is the config-parse + override-map cost. For is
// the per-TENSOR lookup the assembler runs for every weight to resolve (groupSize, bits),
// the hottest of the three at load. Validate is the once-per-config representation check.

func benchQuantBlock() []byte {
	return []byte(`{"group_size":32,"bits":4,"mode":"affine",` +
		`"language_model.model.layers.0.mlp.gate_proj":{"group_size":64,"bits":8},` +
		`"language_model.model.layers.0.mlp.up_proj":{"group_size":64,"bits":8},` +
		`"language_model.model.layers.1.self_attn.q_proj":{"group_size":32,"bits":6}}`)
}

// BenchmarkQuantConfig_UnmarshalJSON — the per-load parse: a generic map decode + the
// per-module override extraction (prefix strip + ModuleQuant build). The override map is
// the allocation a mixed-precision pack pays at load.
func BenchmarkQuantConfig_UnmarshalJSON(b *testing.B) {
	data := benchQuantBlock()
	b.SetBytes(int64(len(data)))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		var q QuantConfig
		if err := q.UnmarshalJSON(data); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkQuantConfig_For_Override — the per-tensor lookup the assembler runs for every
// weight, hitting a per-module override: one map get, no allocation. The hot load-path cost.
func BenchmarkQuantConfig_For_Override(b *testing.B) {
	var q QuantConfig
	if err := q.UnmarshalJSON(benchQuantBlock()); err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = q.For("model.layers.0.mlp.gate_proj")
	}
}

// BenchmarkQuantConfig_For_Default — the same lookup missing the override map, so it falls
// back to the default (group_size, bits): a failed map get, which must stay as cheap as a hit.
func BenchmarkQuantConfig_For_Default(b *testing.B) {
	var q QuantConfig
	if err := q.UnmarshalJSON(benchQuantBlock()); err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = q.For("model.layers.9.self_attn.o_proj")
	}
}

// BenchmarkQuantConfig_Validate — the representation check an arch's parse runs once on the
// resolved block: a mode normalise + a bits/group_size switch, no allocation on the success path.
func BenchmarkQuantConfig_Validate(b *testing.B) {
	q := &QuantConfig{GroupSize: 32, Bits: 4, Mode: "affine"}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := q.Validate(); err != nil {
			b.Fatal(err)
		}
	}
}
