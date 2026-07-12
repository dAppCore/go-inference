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

// TestResolveMTPMethod covers the method default: an unset method resolves to
// MTPDraftModel (the only shipped method, and what every legacy checkpoint uses),
// while an explicitly-set method passes through unchanged.
func TestResolveMTPMethod(t *testing.T) {
	if got := resolveMTPMethod(""); got != MTPDraftModel {
		t.Fatalf(`resolveMTPMethod("") = %q, want the default %q`, got, MTPDraftModel)
	}
	if got := resolveMTPMethod(MTPDraftModel); got != MTPDraftModel {
		t.Fatalf("resolveMTPMethod(MTPDraftModel) = %q, want it unchanged", got)
	}
	if got := resolveMTPMethod(MTPMethod("eagle")); got != MTPMethod("eagle") {
		t.Fatalf(`resolveMTPMethod("eagle") = %q, want a set method to pass through`, got)
	}
	// MTPDFlash is the block-diffusion method; its value is the checkpoint's
	// speculators_model_type marker and it passes through unchanged.
	if MTPDFlash != "dflash" {
		t.Fatalf("MTPDFlash should equal the checkpoint marker %q, got %q", "dflash", MTPDFlash)
	}
	if got := resolveMTPMethod(MTPDFlash); got != MTPDFlash {
		t.Fatalf("resolveMTPMethod(MTPDFlash) = %q, want it unchanged", got)
	}
}

// TestParseAssistantConfigStampsMTPMethod proves the MTP method is INFERRED FROM THE
// MODEL: ParseAssistantConfig stamps the registered spec's Method onto the parsed
// AssistantConfig, and a spec that leaves the method unset defaults to MTPDraftModel
// (the separate-drafter method every current checkpoint uses).
func TestParseAssistantConfigStampsMTPMethod(t *testing.T) {
	RegisterAssistant(AssistantSpec{
		ModelTypes: []string{"fake7_assistant"},
		Method:     MTPMethod("eagle"),
		Parse: func([]byte) (AssistantConfig, error) {
			return AssistantConfig{ModelType: "fake7_assistant"}, nil
		},
	})
	cfg, err := ParseAssistantConfig([]byte(`{"model_type":"fake7_assistant"}`))
	if err != nil {
		t.Fatalf("ParseAssistantConfig: %v", err)
	}
	if cfg.Method != MTPMethod("eagle") {
		t.Fatalf("Method = %q, want the spec's declared %q inferred onto the config", cfg.Method, "eagle")
	}
	RegisterAssistant(AssistantSpec{
		ModelTypes: []string{"fake8_assistant"},
		Parse: func([]byte) (AssistantConfig, error) {
			return AssistantConfig{ModelType: "fake8_assistant"}, nil
		},
	})
	cfg, err = ParseAssistantConfig([]byte(`{"model_type":"fake8_assistant"}`))
	if err != nil {
		t.Fatalf("ParseAssistantConfig (no method): %v", err)
	}
	if cfg.Method != MTPDraftModel {
		t.Fatalf("Method = %q, want the %q default when the spec declares none", cfg.Method, MTPDraftModel)
	}
}

// TestAssistantSpec_RegisterAssistant_Good covers the ordinary registration: every
// declared ModelTypes alias AND the "gguf:"-prefixed GGUFArch resolve to the same spec.
func TestAssistantSpec_RegisterAssistant_Good(t *testing.T) {
	RegisterAssistant(AssistantSpec{
		ModelTypes: []string{"good10_assistant"},
		GGUFArch:   "good10-assistant",
	})
	if _, ok := LookupAssistant("good10_assistant"); !ok {
		t.Fatal("LookupAssistant(good10_assistant): not found after registration")
	}
	if _, ok := LookupAssistantGGUF("good10-assistant"); !ok {
		t.Fatal(`LookupAssistantGGUF("good10-assistant"): not found after registration`)
	}
}

// TestAssistantSpec_RegisterAssistant_Bad covers a spec that declares NO GGUFArch: the
// "gguf:" namespace must gain no entry at all (unlike ModelTypes, GGUFArch registration
// is conditional on it being set).
func TestAssistantSpec_RegisterAssistant_Bad(t *testing.T) {
	RegisterAssistant(AssistantSpec{ModelTypes: []string{"bad10_assistant"}})
	if _, ok := LookupAssistantGGUF("bad10_assistant"); ok {
		t.Fatal("LookupAssistantGGUF must not match a bare ModelTypes id (it is not GGUFArch)")
	}
}

// TestAssistantSpec_RegisterAssistant_Ugly covers re-registration: the registry is Open
// (overwrite), so registering the SAME model_type twice replaces the prior spec.
func TestAssistantSpec_RegisterAssistant_Ugly(t *testing.T) {
	RegisterAssistant(AssistantSpec{ModelTypes: []string{"ugly10_assistant"}, Method: MTPMethod("first")})
	RegisterAssistant(AssistantSpec{ModelTypes: []string{"ugly10_assistant"}, Method: MTPMethod("second")})
	spec, ok := LookupAssistant("ugly10_assistant")
	if !ok {
		t.Fatal("LookupAssistant(ugly10_assistant): not found after re-registration")
	}
	if spec.Method != MTPMethod("second") {
		t.Fatalf("re-registration did not overwrite: Method = %q, want %q", spec.Method, "second")
	}
}

// TestAssistantSpec_LookupAssistant_Good covers a registered config.json model_type
// resolving to its exact spec.
func TestAssistantSpec_LookupAssistant_Good(t *testing.T) {
	RegisterAssistant(AssistantSpec{
		ModelTypes: []string{"lookup-good-assistant"},
		Parse:      func([]byte) (AssistantConfig, error) { return AssistantConfig{BackboneHidden: 7}, nil },
	})
	spec, ok := LookupAssistant("lookup-good-assistant")
	if !ok {
		t.Fatal("LookupAssistant(lookup-good-assistant): not found")
	}
	cfg, err := spec.Parse(nil)
	if err != nil || cfg.BackboneHidden != 7 {
		t.Fatalf("resolved spec.Parse = (%+v, %v), want BackboneHidden=7", cfg, err)
	}
}

// TestAssistantSpec_LookupAssistant_Bad covers an unregistered model_type: ok=false and
// a zero AssistantSpec.
func TestAssistantSpec_LookupAssistant_Bad(t *testing.T) {
	spec, ok := LookupAssistant("never-registered-assistant-xyz")
	if ok {
		t.Fatal("LookupAssistant(never-registered-assistant-xyz) = found, want ok=false")
	}
	if spec.Parse != nil {
		t.Fatal("LookupAssistant miss should return the zero AssistantSpec")
	}
}

// TestAssistantSpec_LookupAssistant_Ugly covers a NUL-prefixed sentinel model_type that
// no spec ever claims: LookupAssistant must miss (unlike "" itself, which
// TestRegisterAssistantEmptyModelTypeClaimsLegacy shows CAN resolve once a spec
// deliberately claims it — the empty string is not universally unregistrable, only a
// truly never-claimed id is).
func TestAssistantSpec_LookupAssistant_Ugly(t *testing.T) {
	if _, ok := LookupAssistant("\x00-sentinel-never-registered"); ok {
		t.Fatal("a NUL-prefixed sentinel model_type must never resolve")
	}
}

// TestAssistantSpec_LookupAssistantGGUF_Good covers a registered GGUFArch resolving
// under its "gguf:" prefix.
func TestAssistantSpec_LookupAssistantGGUF_Good(t *testing.T) {
	RegisterAssistant(AssistantSpec{ModelTypes: []string{"gguf-good-assistant"}, GGUFArch: "gguf-good-arch"})
	if _, ok := LookupAssistantGGUF("gguf-good-arch"); !ok {
		t.Fatal("LookupAssistantGGUF(gguf-good-arch): not found")
	}
}

// TestAssistantSpec_LookupAssistantGGUF_Bad covers an unregistered GGUF
// general.architecture: ok=false.
func TestAssistantSpec_LookupAssistantGGUF_Bad(t *testing.T) {
	if _, ok := LookupAssistantGGUF("never-registered-gguf-arch-xyz"); ok {
		t.Fatal("LookupAssistantGGUF(never-registered-gguf-arch-xyz) = found, want ok=false")
	}
}

// TestAssistantSpec_LookupAssistantGGUF_Ugly covers namespace isolation: a ModelTypes id
// must NOT be resolvable through LookupAssistantGGUF even when an identically-spelled
// GGUFArch was never registered — the "gguf:" prefix is a genuinely separate keyspace.
func TestAssistantSpec_LookupAssistantGGUF_Ugly(t *testing.T) {
	RegisterAssistant(AssistantSpec{ModelTypes: []string{"shared-name-assistant"}})
	if _, ok := LookupAssistantGGUF("shared-name-assistant"); ok {
		t.Fatal("LookupAssistantGGUF must not match a bare ModelTypes id sharing its GGUFArch's spelling")
	}
}

// TestAssistantSpec_ParseAssistantConfig_Good covers the ordinary probe-then-dispatch: a
// declared model_type routes to its registered spec's Parse.
func TestAssistantSpec_ParseAssistantConfig_Good(t *testing.T) {
	RegisterAssistant(AssistantSpec{
		ModelTypes: []string{"parse-good-assistant"},
		Parse:      func([]byte) (AssistantConfig, error) { return AssistantConfig{BackboneHidden: 99}, nil },
	})
	cfg, err := ParseAssistantConfig([]byte(`{"model_type":"parse-good-assistant"}`))
	if err != nil {
		t.Fatalf("ParseAssistantConfig: %v", err)
	}
	if cfg.BackboneHidden != 99 {
		t.Fatalf("BackboneHidden = %d, want 99 (dispatched to the registered spec)", cfg.BackboneHidden)
	}
}

// TestAssistantSpec_ParseAssistantConfig_Bad covers an undeclared/unregistered
// model_type: a clean error, before any backend-specific parsing runs.
func TestAssistantSpec_ParseAssistantConfig_Bad(t *testing.T) {
	if _, err := ParseAssistantConfig([]byte(`{"model_type":"never-registered-parse-bad"}`)); err == nil {
		t.Fatal("ParseAssistantConfig with an unregistered model_type: expected an error")
	}
}

// TestAssistantSpec_ParseAssistantConfig_Ugly covers malformed probe JSON itself: a
// clean error rather than a panic deep in json.Unmarshal.
func TestAssistantSpec_ParseAssistantConfig_Ugly(t *testing.T) {
	if _, err := ParseAssistantConfig([]byte(`{not json at all`)); err == nil {
		t.Fatal("ParseAssistantConfig with malformed JSON: expected an error")
	}
}

// TestAssistantSpec_AssistantConfig_LayerType_Good covers the declared-entry win: an
// explicit non-blank LayerTypes entry is returned as-is.
func TestAssistantSpec_AssistantConfig_LayerType_Good(t *testing.T) {
	c := AssistantConfig{LayerTypes: []string{"sliding_attention"}}
	if got := c.LayerType(0); got != "sliding_attention" {
		t.Fatalf("LayerType(0) = %q, want the declared %q", got, "sliding_attention")
	}
}

// TestAssistantSpec_AssistantConfig_LayerType_Bad covers the declared-blank fallback: an
// index within LayerTypes whose entry is "" falls back to the Arch layer's own TypeName.
func TestAssistantSpec_AssistantConfig_LayerType_Bad(t *testing.T) {
	c := AssistantConfig{
		LayerTypes: []string{""},
		Arch:       Arch{Layer: []LayerSpec{{Attention: GlobalAttention}}},
	}
	if got := c.LayerType(0); got != "full_attention" {
		t.Fatalf("LayerType(0) blank declared entry = %q, want the Arch fallback %q", got, "full_attention")
	}
}

// TestAssistantSpec_AssistantConfig_LayerType_Ugly covers both out-of-range directions
// (past the end AND negative): LayerType must return "" rather than index out of range.
func TestAssistantSpec_AssistantConfig_LayerType_Ugly(t *testing.T) {
	c := AssistantConfig{LayerTypes: []string{"sliding_attention"}}
	if got := c.LayerType(50); got != "" {
		t.Fatalf("LayerType(50) out of range = %q, want empty", got)
	}
	if got := c.LayerType(-3); got != "" {
		t.Fatalf("LayerType(-3) negative index = %q, want empty", got)
	}
}
