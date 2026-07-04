// SPDX-Licence-Identifier: EUPL-1.2

package model

import (
	"testing"

	"dappco.re/go/inference/safetensors"
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
