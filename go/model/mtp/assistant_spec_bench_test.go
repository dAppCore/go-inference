// SPDX-Licence-Identifier: EUPL-1.2

package mtp

import (
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference/model"
)

// The assistant-spec benches baseline the attached-drafter contract (AX-11).
// ParseAssistantConfig is the per-drafter-load path — a JSON model_type probe, a registry
// dispatch, and the spec's own parse — so its allocation is the config-parse cost. LayerType
// is the per-layer KV-stream-matching lookup a decode pairing runs, expected alloc-free.
// Synthetic — a bench-only assistant spec registered with a real JSON parse.

// benchLayerTypes builds a realistic 5-sliding-1-global attention pattern (the gemma-style
// layout) for the bench fixture — a bench-only helper, mirroring the arch benches' input.
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

func benchAssistantConfig() AssistantConfig {
	lt := benchLayerTypes(48)
	return AssistantConfig{ModelType: "bench-assistant", LayerTypes: lt, Arch: model.Arch{Layer: model.DeriveLayers(lt, 0)}}
}

func registerBenchAssistant() {
	RegisterAssistant(AssistantSpec{
		ModelTypes: []string{"bench-assistant"},
		Parse: func(data []byte) (AssistantConfig, error) {
			var probe struct {
				ModelType      string `json:"model_type"`
				BackboneHidden int    `json:"backbone_hidden"`
			}
			if r := core.JSONUnmarshal(data, &probe); !r.OK {
				return AssistantConfig{}, core.NewError("bench parse failed")
			}
			return AssistantConfig{ModelType: probe.ModelType, BackboneHidden: probe.BackboneHidden}, nil
		},
	})
}

// BenchmarkParseAssistantConfig — the per-load path: probe model_type from the config
// bytes, dispatch through the registry, run the spec parse. The JSON unmarshals are the
// allocation story a drafter load pays.
func BenchmarkParseAssistantConfig(b *testing.B) {
	registerBenchAssistant()
	data := []byte(`{"model_type":"bench-assistant","backbone_hidden":2048,"num_hidden_layers":6}`)
	b.SetBytes(int64(len(data)))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := ParseAssistantConfig(data); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkAssistantConfig_LayerType — the per-layer name lookup KV-stream matching runs
// for every drafter layer: a bounds-checked slice index, no allocation.
func BenchmarkAssistantConfig_LayerType(b *testing.B) {
	c := benchAssistantConfig()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = c.LayerType(i % 48)
	}
}
