// SPDX-Licence-Identifier: EUPL-1.2

package model

import "testing"

// The arch benches baseline the per-layer decode-topology derivation (AX-11). DeriveLayers
// is the once-per-load resolve of every layer's attention type + KV-cache-sharing map: it
// allocates the []LayerSpec result and a small type→owner map, so its shape is the
// per-model derivation cost. HasMoE is the per-decode scan a backend runs to route MoE
// archs off fast paths. Realistic input: a 48-layer gemma-style 5-sliding-1-global pattern.

func benchLayerTypes(n int) []string {
	lt := make([]string, n)
	for i := range lt {
		if (i+1)%6 == 0 {
			lt[i] = "full_attention"
		} else {
			lt[i] = "sliding_attention"
		}
	}
	return lt
}

// BenchmarkDeriveLayers — the full per-load derivation: 48 layers resolved to specs with
// the KV-share map threaded through. The []LayerSpec result + the latestByType map are the
// allocation story a model pays once at load.
func BenchmarkDeriveLayers(b *testing.B) {
	lt := benchLayerTypes(48)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = DeriveLayers(lt, 4)
	}
}

// BenchmarkArch_HasMoE — the per-decode routing scan over a dense 48-layer arch (worst
// case: no MoE layer, so the scan runs to the end). No allocation; pins the routing check
// stays a cheap loop.
func BenchmarkArch_HasMoE(b *testing.B) {
	a := Arch{Layer: DeriveLayers(benchLayerTypes(48), 4)}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if a.HasMoE() {
			b.Fatal("dense arch reported MoE")
		}
	}
}
