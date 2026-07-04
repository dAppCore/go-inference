// SPDX-Licence-Identifier: EUPL-1.2

package model

import "testing"

// TestRegisterAssistantAliasesAndLookup proves the reactive assistant registry: every
// ModelTypes alias resolves, the GGUFArch resolves under its "gguf:" prefix, and an
// unregistered id (for either lookup) misses — the dispatch the engine's assistant loader
// reacts to. Mirrors TestRegisterArchAliasesAndLookup for the assistant twin registry.
func TestRegisterAssistantAliasesAndLookup(t *testing.T) {
	RegisterAssistant(AssistantSpec{
		ModelTypes: []string{"fake4_assistant", "fake4_unified_assistant"},
		Parse: func([]byte) (AssistantConfig, error) {
			return AssistantConfig{ModelType: "fake4_assistant"}, nil
		},
		GGUFArch: "fake4-assistant",
		ParseGGUF: func(map[string]any, int) (AssistantConfig, error) {
			return AssistantConfig{ModelType: "fake4_assistant"}, nil
		},
	})
	for _, mt := range []string{"fake4_assistant", "fake4_unified_assistant"} {
		spec, ok := LookupAssistant(mt)
		if !ok {
			t.Fatalf("LookupAssistant(%q) = not found, want registered", mt)
		}
		cfg, err := spec.Parse(nil)
		if err != nil || cfg.ModelType != "fake4_assistant" {
			t.Fatalf("spec.Parse for %q = (%+v, %v), want a fake4_assistant config", mt, cfg, err)
		}
	}
	spec, ok := LookupAssistantGGUF("fake4-assistant")
	if !ok {
		t.Fatal(`LookupAssistantGGUF("fake4-assistant") = not found, want registered`)
	}
	if cfg, err := spec.ParseGGUF(nil, 0); err != nil || cfg.ModelType != "fake4_assistant" {
		t.Fatalf("spec.ParseGGUF = (%+v, %v), want a fake4_assistant config", cfg, err)
	}
	if _, ok := LookupAssistant("unregistered-assistant"); ok {
		t.Fatal("unregistered model_type must not resolve")
	}
	if _, ok := LookupAssistantGGUF("unregistered-gguf-arch"); ok {
		t.Fatal("unregistered GGUF general.architecture must not resolve")
	}
}

// TestRegisterAssistantEmptyModelTypeClaimsLegacy proves a spec MAY claim the empty
// model_type ("") to own checkpoints that predate the field — assistant_spec.go's documented
// behaviour, and the one deliberate divergence from RegisterArch (which skips "" outright:
// see TestRegisterArchAliasesAndLookup's "empty model_type must not register").
func TestRegisterAssistantEmptyModelTypeClaimsLegacy(t *testing.T) {
	RegisterAssistant(AssistantSpec{
		ModelTypes: []string{"fake5_assistant", ""},
		Parse: func([]byte) (AssistantConfig, error) {
			return AssistantConfig{ModelType: "fake5_assistant"}, nil
		},
	})
	spec, ok := LookupAssistant("")
	if !ok {
		t.Fatal(`LookupAssistant("") = not found, want the legacy-default spec registered`)
	}
	cfg, err := spec.Parse(nil)
	if err != nil || cfg.ModelType != "fake5_assistant" {
		t.Fatalf(`legacy "" spec.Parse = (%+v, %v)`, cfg, err)
	}
}

// TestParseAssistantConfigDispatch proves ParseAssistantConfig's probe-then-dispatch: a
// declared model_type routes to its registered spec's Parse, an undeclared/unregistered
// model_type is a clean error, and malformed probe JSON is a clean error too — all before
// any backend-specific parsing runs.
func TestParseAssistantConfigDispatch(t *testing.T) {
	RegisterAssistant(AssistantSpec{
		ModelTypes: []string{"fake6_assistant"},
		Parse: func([]byte) (AssistantConfig, error) {
			return AssistantConfig{ModelType: "fake6_assistant", BackboneHidden: 42}, nil
		},
	})
	cfg, err := ParseAssistantConfig([]byte(`{"model_type":"fake6_assistant"}`))
	if err != nil {
		t.Fatalf("ParseAssistantConfig: %v", err)
	}
	if cfg.BackboneHidden != 42 {
		t.Fatalf("ParseAssistantConfig dispatched the wrong spec: %+v", cfg)
	}
	if _, err := ParseAssistantConfig([]byte(`{"model_type":"nope-assistant"}`)); err == nil {
		t.Fatal("expected an error for an unregistered assistant model_type")
	}
	if _, err := ParseAssistantConfig([]byte(`not json`)); err == nil {
		t.Fatal("expected an error when the probe JSON itself is malformed")
	}
}

// TestAssistantConfigLayerType covers LayerType's declared-vs-derived fallback: an explicit
// LayerTypes entry wins, an entry the config left blank falls back to the Arch layer's own
// TypeName (the KV-stream matching the doc comment describes), and an out-of-range index —
// past both slices — returns "".
func TestAssistantConfigLayerType(t *testing.T) {
	c := AssistantConfig{
		LayerTypes: []string{"sliding_attention", ""},
		Arch: Arch{Layer: []LayerSpec{
			{Attention: SlidingAttention},
			{Attention: GlobalAttention},
		}},
	}
	if got := c.LayerType(0); got != "sliding_attention" {
		t.Fatalf("LayerType(0) = %q, want the declared %q", got, "sliding_attention")
	}
	if got := c.LayerType(1); got != "full_attention" {
		t.Fatalf("LayerType(1) = %q, want the Arch fallback %q (declared entry was blank)", got, "full_attention")
	}
	if got := c.LayerType(5); got != "" {
		t.Fatalf(`LayerType(5) out of range = %q, want ""`, got)
	}
	if got := c.LayerType(-1); got != "" {
		t.Fatalf(`LayerType(-1) negative index = %q, want ""`, got)
	}
}
