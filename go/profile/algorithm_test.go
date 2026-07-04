// SPDX-Licence-Identifier: EUPL-1.2

package profile_test

import (
	"slices"
	"testing"

	"dappco.re/go/inference"
	prof "dappco.re/go/inference/profile"
)

func containsCapabilityProvide(values []string, want string) bool {
	return slices.Contains(values, want)
}

// TestAlgorithm_BuiltinAlgorithmProfiles_Good pins the built-in algorithm
// feature matrix: every advertised capability ID is present exactly once and
// carries a non-empty group and detail, the shape a CapabilityReport renders.
func TestAlgorithm_BuiltinAlgorithmProfiles_Good(t *testing.T) {
	profiles := prof.BuiltinAlgorithmProfiles()
	if len(profiles) == 0 {
		t.Fatal("prof.BuiltinAlgorithmProfiles() returned no profiles")
	}
	seen := map[inference.CapabilityID]bool{}
	for _, p := range profiles {
		if p.ID == "" {
			t.Fatalf("profile missing ID: %+v", p)
		}
		if seen[p.ID] {
			t.Fatalf("duplicate algorithm profile %q", p.ID)
		}
		seen[p.ID] = true
		if p.Group == "" || p.Detail == "" {
			t.Fatalf("profile = %+v, want group and detail", p)
		}
	}
	for _, id := range []inference.CapabilityID{
		inference.CapabilityScheduler,
		inference.CapabilityQuantization,
		inference.CapabilityMoELazyExperts,
		inference.CapabilitySpeculativeDecode,
	} {
		if !seen[id] {
			t.Fatalf("BuiltinAlgorithmProfiles missing capability %q", id)
		}
	}
}

// TestAlgorithm_BuiltinAlgorithmProfiles_Bad pins the defensive-clone contract:
// mutating an element of the returned slice must not leak into the registry, so
// a later call sees the original algorithm name.
func TestAlgorithm_BuiltinAlgorithmProfiles_Bad(t *testing.T) {
	profiles := prof.BuiltinAlgorithmProfiles()
	if len(profiles) == 0 {
		t.Fatal("prof.BuiltinAlgorithmProfiles() returned no profiles")
	}
	original := profiles[0].Algorithm
	profiles[0].Algorithm = "mutated"
	again := prof.BuiltinAlgorithmProfiles()
	if again[0].Algorithm == "mutated" {
		t.Fatal("prof.BuiltinAlgorithmProfiles returned aliased profile data")
	}
	if again[0].Algorithm != original {
		t.Fatalf("BuiltinAlgorithmProfiles()[0].Algorithm = %q, want stable %q", again[0].Algorithm, original)
	}
}

// TestAlgorithm_BuiltinAlgorithmProfiles_Ugly pins per-call independence at the
// element-scalar level: mutating the Detail of a returned profile must not leak
// into a fresh call, and the fresh slice keeps a stable length (no aliasing of
// the backing array between calls).
func TestAlgorithm_BuiltinAlgorithmProfiles_Ugly(t *testing.T) {
	first := prof.BuiltinAlgorithmProfiles()
	if len(first) < 2 {
		t.Fatalf("BuiltinAlgorithmProfiles len = %d, want at least 2 to test element independence", len(first))
	}
	want := len(first)
	id := first[1].ID
	original := first[1].Detail
	first[1].Detail = "mutated-detail"
	second := prof.BuiltinAlgorithmProfiles()
	if len(second) != want {
		t.Fatalf("BuiltinAlgorithmProfiles len = %d, want stable %d", len(second), want)
	}
	if second[1].ID != id || second[1].Detail != original {
		t.Fatalf("BuiltinAlgorithmProfiles[1] = {%q, Detail=%q}, want stable {%q, %q}", second[1].ID, second[1].Detail, id, original)
	}
}

// TestAlgorithm_LookupAlgorithmProfile_Good pins the per-ID resolution and the
// runtime/capability state every built-in algorithm advertises — the metadata
// a backend planner reads to decide native vs experimental vs planned.
func TestAlgorithm_LookupAlgorithmProfile_Good(t *testing.T) {
	cases := []struct {
		id      inference.CapabilityID
		runtime prof.AlgorithmRuntimeStatus
		status  inference.CapabilityStatus
	}{
		{id: inference.CapabilityScheduler, runtime: prof.AlgorithmRuntimeNative, status: inference.CapabilityStatusSupported},
		{id: inference.CapabilityCacheBlocks, runtime: prof.AlgorithmRuntimeNative, status: inference.CapabilityStatusSupported},
		{id: inference.CapabilityReasoningParse, runtime: prof.AlgorithmRuntimeNative, status: inference.CapabilityStatusSupported},
		{id: inference.CapabilityJANGTQ, runtime: prof.AlgorithmRuntimeMetadataOnly, status: inference.CapabilityStatusExperimental},
		{id: inference.CapabilityCodebookVQ, runtime: prof.AlgorithmRuntimeExperimental, status: inference.CapabilityStatusExperimental},
		{id: inference.CapabilityQuantization, runtime: prof.AlgorithmRuntimeExperimental, status: inference.CapabilityStatusExperimental},
		{id: inference.CapabilityEmbeddings, runtime: prof.AlgorithmRuntimeMetadataOnly, status: inference.CapabilityStatusPlanned},
		{id: inference.CapabilityMoERouting, runtime: prof.AlgorithmRuntimeMetadataOnly, status: inference.CapabilityStatusPlanned},
		{id: inference.CapabilityMoELazyExperts, runtime: prof.AlgorithmRuntimeExperimental, status: inference.CapabilityStatusExperimental},
		{id: inference.CapabilitySpeculativeDecode, runtime: prof.AlgorithmRuntimeExperimental, status: inference.CapabilityStatusExperimental},
		{id: inference.CapabilityPromptLookupDecode, runtime: prof.AlgorithmRuntimeExperimental, status: inference.CapabilityStatusExperimental},
	}

	for _, tc := range cases {
		t.Run(string(tc.id), func(t *testing.T) {
			p, ok := prof.LookupAlgorithmProfile(tc.id)
			if !ok {
				t.Fatalf("prof.LookupAlgorithmProfile(%q) ok = false", tc.id)
			}
			if p.RuntimeStatus != tc.runtime || p.CapabilityStatus != tc.status {
				t.Fatalf("profile = %+v, want runtime/status %q/%q", p, tc.runtime, tc.status)
			}
			if p.Group == "" || p.Detail == "" {
				t.Fatalf("profile = %+v, want group and detail", p)
			}
		})
	}

	t.Run("LazyExpertsProvides", func(t *testing.T) {
		p, ok := prof.LookupAlgorithmProfile(inference.CapabilityMoELazyExperts)
		if !ok {
			t.Fatal("missing lazy expert profile")
		}
		if !containsCapabilityProvide(p.Provides, "expert.page_in") || !containsCapabilityProvide(p.Provides, "expert.residency.probe") {
			t.Fatalf("lazy expert provides = %+v, want page-in and probe labels", p.Provides)
		}
	})

	t.Run("AutoRoundProvides", func(t *testing.T) {
		p, ok := prof.LookupAlgorithmProfile(inference.CapabilityQuantization)
		if !ok {
			t.Fatal("missing quantization profile")
		}
		if p.Algorithm != "auto-round" {
			t.Fatalf("quantization profile = %+v, want auto-round", p)
		}
		for _, want := range []string{"quantization.profile.auto-round", "quantization.profile.auto-round-best", "quantization.profile.auto-round-light", "weight_rounding.signround", "packed_weight.write_safetensors_projection", "packed_weight.write_safetensors_pack", "packed_weight.write_native_pack_sidecar", "model_pack.inspect_native_tensor_map"} {
			if !containsCapabilityProvide(p.Provides, want) {
				t.Fatalf("quantization provides = %+v, want %q", p.Provides, want)
			}
		}
	})
}

// TestAlgorithm_LookupAlgorithmProfile_Bad pins the miss path: a capability ID
// that names no built-in algorithm yields ok=false and the zero profile, so
// callers branch on ok rather than reading a guessed value.
func TestAlgorithm_LookupAlgorithmProfile_Bad(t *testing.T) {
	p, ok := prof.LookupAlgorithmProfile(inference.CapabilityID("missing-capability"))
	if ok {
		t.Fatal("prof.LookupAlgorithmProfile(missing) ok = true")
	}
	if p.Algorithm != "" || p.RuntimeStatus != "" {
		t.Fatalf("prof.LookupAlgorithmProfile(missing) = %+v, want zero profile", p)
	}
}

// TestAlgorithm_LookupAlgorithmProfile_Ugly pins the empty-ID edge: an empty
// capability ID is not registered, so the lookup misses cleanly rather than
// matching a zero-keyed entry.
func TestAlgorithm_LookupAlgorithmProfile_Ugly(t *testing.T) {
	if _, ok := prof.LookupAlgorithmProfile(inference.CapabilityID("")); ok {
		t.Fatal("prof.LookupAlgorithmProfile(\"\") ok = true, want false for empty id")
	}
}

// TestAlgorithm_AlgorithmCapabilities_Good pins the folded capability list every
// CapabilityReport appends: each entry is unique, carries a runtime_status
// label, and the experimental/planned families are all present.
func TestAlgorithm_AlgorithmCapabilities_Good(t *testing.T) {
	capabilities := prof.AlgorithmCapabilities()
	seen := map[inference.CapabilityID]bool{}
	for _, capability := range capabilities {
		if seen[capability.ID] {
			t.Fatalf("duplicate algorithm capability %q", capability.ID)
		}
		seen[capability.ID] = true
		if capability.Labels["runtime_status"] == "" {
			t.Fatalf("capability = %+v, want runtime_status label", capability)
		}
	}
	for _, id := range []inference.CapabilityID{
		inference.CapabilitySpeculativeDecode,
		inference.CapabilityPromptLookupDecode,
		inference.CapabilityEmbeddings,
		inference.CapabilityRerank,
		inference.CapabilityMoERouting,
		inference.CapabilityMoELazyExperts,
		inference.CapabilityCodebookVQ,
		inference.CapabilityQuantization,
	} {
		if !seen[id] {
			t.Fatalf("missing algorithm capability %q", id)
		}
	}
}

// TestAlgorithm_AlgorithmCapabilities_Bad pins that the folded capabilities
// carry through the per-profile labels intact: the prompt-lookup entry must
// expose its experimental runtime state and algorithm name, not a blank label.
func TestAlgorithm_AlgorithmCapabilities_Bad(t *testing.T) {
	var found bool
	for _, capability := range prof.AlgorithmCapabilities() {
		if capability.ID != inference.CapabilityPromptLookupDecode {
			continue
		}
		found = true
		if capability.Status != inference.CapabilityStatusExperimental {
			t.Fatalf("prompt-lookup capability = %+v, want experimental status", capability)
		}
		if capability.Labels["runtime_status"] != string(prof.AlgorithmRuntimeExperimental) || capability.Labels["algorithm"] != "prompt-lookup" {
			t.Fatalf("labels = %+v, want experimental runtime_status and prompt-lookup algorithm", capability.Labels)
		}
	}
	if !found {
		t.Fatal("AlgorithmCapabilities() missing prompt-lookup-decode entry")
	}
}

// TestAlgorithm_AlgorithmCapabilities_Ugly pins the count parity edge: the
// folded capability slice has exactly as many entries as the backing profile
// matrix — no entry dropped or duplicated in the fold.
func TestAlgorithm_AlgorithmCapabilities_Ugly(t *testing.T) {
	capabilities := prof.AlgorithmCapabilities()
	profiles := prof.BuiltinAlgorithmProfiles()
	if len(capabilities) != len(profiles) {
		t.Fatalf("AlgorithmCapabilities len = %d, want one per profile (%d)", len(capabilities), len(profiles))
	}
}
