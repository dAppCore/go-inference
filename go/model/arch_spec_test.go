// SPDX-Licence-Identifier: EUPL-1.2

package model

import (
	"testing"

	"dappco.re/go/inference/model/safetensors"
)

type fakeArchConfig struct{}

func (fakeArchConfig) InferFromWeights(map[string]safetensors.Tensor) {}
func (fakeArchConfig) Arch() (Arch, error)                            { return Arch{}, nil }

// TestRegisterArchAliasesAndLookup proves the reactive registry: every ModelTypes alias resolves, an
// empty id is ignored, and an unregistered id misses — the dispatch the engine's loader reacts to.
func TestRegisterArchAliasesAndLookup(t *testing.T) {
	RegisterArch(ArchSpec{
		ModelTypes: []string{"fake4", "fake4_text", "fake4_unified", ""}, // "" must be ignored
		Parse:      func([]byte) (ArchConfig, error) { return fakeArchConfig{}, nil },
	})
	for _, mt := range []string{"fake4", "fake4_text", "fake4_unified"} {
		spec, ok := LookupArch(mt)
		if !ok {
			t.Fatalf("LookupArch(%q) = not found, want registered", mt)
		}
		cfg, err := spec.Parse(nil)
		if err != nil || cfg == nil {
			t.Fatalf("spec.Parse for %q = (%v, %v), want a config", mt, cfg, err)
		}
	}
	if _, ok := LookupArch(""); ok {
		t.Fatal("empty model_type must not register")
	}
	if _, ok := LookupArch("unregistered"); ok {
		t.Fatal("unregistered model_type must not resolve")
	}
}

// TestArchSpec_RegisterArch_Good covers the ordinary registration: every declared
// ModelTypes alias resolves to the SAME spec.
func TestArchSpec_RegisterArch_Good(t *testing.T) {
	spec := ArchSpec{
		ModelTypes: []string{"good9", "good9_text"},
		Parse:      func([]byte) (ArchConfig, error) { return fakeArchConfig{}, nil },
	}
	RegisterArch(spec)
	for _, mt := range spec.ModelTypes {
		if _, ok := LookupArch(mt); !ok {
			t.Fatalf("LookupArch(%q) after RegisterArch: not found", mt)
		}
	}
}

// TestArchSpec_RegisterArch_Bad covers the documented empty-id guard: RegisterArch must
// NOT register the empty model_type (unlike RegisterAssistant, which deliberately does —
// see TestRegisterAssistantEmptyModelTypeClaimsLegacy).
func TestArchSpec_RegisterArch_Bad(t *testing.T) {
	RegisterArch(ArchSpec{ModelTypes: []string{""}})
	if _, ok := LookupArch(""); ok {
		t.Fatal("RegisterArch must not register the empty model_type")
	}
}

// TestArchSpec_RegisterArch_Ugly covers re-registration: the registry is Open
// (overwrite), so registering the SAME model_type twice replaces the prior spec rather
// than erroring or duplicating.
func TestArchSpec_RegisterArch_Ugly(t *testing.T) {
	RegisterArch(ArchSpec{ModelTypes: []string{"ugly9"}, Weights: WeightNames{Embed: "first"}})
	RegisterArch(ArchSpec{ModelTypes: []string{"ugly9"}, Weights: WeightNames{Embed: "second"}})
	spec, ok := LookupArch("ugly9")
	if !ok {
		t.Fatal("LookupArch(ugly9): not found after re-registration")
	}
	if spec.Weights.Embed != "second" {
		t.Fatalf("re-registration did not overwrite: Weights.Embed = %q, want %q", spec.Weights.Embed, "second")
	}
}

// TestArchSpec_LookupArch_Good covers a registered id resolving to its exact spec (the
// Parse function set on registration, callable through the looked-up value).
func TestArchSpec_LookupArch_Good(t *testing.T) {
	RegisterArch(ArchSpec{
		ModelTypes: []string{"lookup-good"},
		Parse:      func([]byte) (ArchConfig, error) { return fakeArchConfig{}, nil },
	})
	spec, ok := LookupArch("lookup-good")
	if !ok {
		t.Fatal("LookupArch(lookup-good): not found")
	}
	if _, err := spec.Parse(nil); err != nil {
		t.Fatalf("resolved spec.Parse: %v", err)
	}
}

// TestArchSpec_LookupArch_Bad covers an unregistered id: ok=false and a zero ArchSpec.
func TestArchSpec_LookupArch_Bad(t *testing.T) {
	spec, ok := LookupArch("never-registered-arch")
	if ok {
		t.Fatal("LookupArch(never-registered-arch) = found, want ok=false")
	}
	if spec.Parse != nil {
		t.Fatal("LookupArch miss should return the zero ArchSpec")
	}
}

// TestArchSpec_LookupArch_Ugly covers the empty-string query: since RegisterArch never
// registers "", looking it up must always miss, even after many other registrations.
func TestArchSpec_LookupArch_Ugly(t *testing.T) {
	if _, ok := LookupArch(""); ok {
		t.Fatal(`LookupArch("") = found, want ok=false (RegisterArch never registers the empty id)`)
	}
}
