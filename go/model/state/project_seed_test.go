// SPDX-Licence-Identifier: EUPL-1.2

package state

import "testing"

func TestProjectSeed_WakeRequest_Good(t *testing.T) {
	seed := NewProjectSeed(ProjectSeedOptions{
		BaseURI:   "state://lthn/projects",
		ProjectID: "core/go-mlx",
		Title:     "go-mlx seed",
		Labels:    map[string]string{"scope": "repo"},
		Metadata:  map[string]string{"operator": "snider"},
	})

	wake := seed.WakeRequest(ProjectSeedWakeOptions{
		Store:     "store",
		Model:     ModelIdentity{ID: "gemma4", Hash: "model-a"},
		Tokenizer: TokenizerIdentity{Hash: "tok-a"},
		Adapter:   AdapterIdentity{Hash: "adapter-a"},
		Runtime:   RuntimeIdentity{Backend: "metal", CacheMode: "paged-q8"},
	})

	if wake.Store != "store" || wake.EntryURI != "state://lthn/projects/core/go-mlx/seed" || wake.IndexURI != "state://lthn/projects/core/go-mlx/seed/index" {
		t.Fatalf("wake request = %+v, want project seed URIs and store", wake)
	}
	if wake.Model.Hash != "model-a" || wake.Tokenizer.Hash != "tok-a" || wake.Adapter.Hash != "adapter-a" || wake.Runtime.Backend != "metal" {
		t.Fatalf("wake identities = %+v/%+v/%+v/%+v", wake.Model, wake.Tokenizer, wake.Adapter, wake.Runtime)
	}
	if wake.Labels["project_id"] != "core/go-mlx" || wake.Labels["scope"] != "repo" {
		t.Fatalf("wake labels = %+v, want project and caller labels", wake.Labels)
	}

	seed.Labels["scope"] = "mutated"
	if wake.Labels["scope"] != "repo" {
		t.Fatalf("wake request labels aliased seed labels: %+v", wake.Labels)
	}
}

func TestProjectSeed_PlanContinuation_Good(t *testing.T) {
	seed := NewProjectSeed(ProjectSeedOptions{BaseURI: "state://lthn/projects", ProjectID: "core/go-mlx"})
	parent := WakeResult{
		Entry:        Ref{URI: seed.EntryURI, BundleURI: seed.BundleURI, IndexURI: seed.IndexURI},
		PrefixTokens: 42,
	}

	statePlan := seed.PlanContinuation(ProjectSeedContinuationOptions{
		Mode:      ProjectSeedStateCheckpoint,
		Store:     "store",
		EntryURI:  "state://lthn/projects/core/go-mlx/tasks/inspect",
		Title:     "inspect result",
		Parent:    parent,
		Model:     ModelIdentity{ID: "gemma4"},
		Tokenizer: TokenizerIdentity{Hash: "tok-a"},
		Metadata:  map[string]string{"finding_count": "2"},
	})
	if !statePlan.PersistState || statePlan.NeedsSummary || statePlan.ReuseCurrentSeed {
		t.Fatalf("state plan flags = %+v, want state checkpoint", statePlan)
	}
	if statePlan.Sleep.Store != "store" || !statePlan.Sleep.ReuseParentPrefix {
		t.Fatalf("sleep request = %+v, want store and parent prefix reuse", statePlan.Sleep)
	}
	if statePlan.Sleep.ParentEntryURI != seed.EntryURI || statePlan.Sleep.ParentBundleURI != seed.BundleURI || statePlan.Sleep.ParentIndexURI != seed.IndexURI {
		t.Fatalf("sleep parent = %+v, want seed parent refs", statePlan.Sleep)
	}
	if statePlan.Sleep.Metadata["project_id"] != "core/go-mlx" || statePlan.Sleep.Metadata["finding_count"] != "2" {
		t.Fatalf("sleep metadata = %+v, want project and caller metadata", statePlan.Sleep.Metadata)
	}

	summaryPlan := seed.PlanContinuation(ProjectSeedContinuationOptions{Mode: ProjectSeedSummaryWindow})
	if summaryPlan.PersistState || !summaryPlan.NeedsSummary || summaryPlan.Sleep.EntryURI != "" {
		t.Fatalf("summary plan = %+v, want summary-only window", summaryPlan)
	}

	reusePlan := seed.PlanContinuation(ProjectSeedContinuationOptions{Mode: ProjectSeedReuseCurrent})
	if reusePlan.PersistState || reusePlan.NeedsSummary || !reusePlan.ReuseCurrentSeed {
		t.Fatalf("reuse plan = %+v, want current seed reuse", reusePlan)
	}
}

func TestWakeCompatibility_GoodBadUgly(t *testing.T) {
	bundle := Bundle{
		Model:        ModelIdentity{Hash: "model-a", Architecture: "gemma4_text", NumLayers: 28, QuantBits: 4, ContextLength: 4096},
		Tokenizer:    TokenizerIdentity{Hash: "tok-a", ChatTemplate: "chat-a"},
		Adapter:      AdapterIdentity{Hash: "adapter-a", Rank: 8},
		Runtime:      RuntimeIdentity{Backend: "metal", CacheMode: "paged-q8"},
		PromptTokens: 2048,
	}
	req := WakeRequest{
		Model:     ModelIdentity{Hash: "model-a", Architecture: "gemma4_text", NumLayers: 28, QuantBits: 4, ContextLength: 8192},
		Tokenizer: TokenizerIdentity{Hash: "tok-a", ChatTemplate: "chat-a"},
		Adapter:   AdapterIdentity{Hash: "adapter-a", Rank: 8},
		Runtime:   RuntimeIdentity{Backend: "rocm", CacheMode: "paged-q8"},
	}

	report := CheckWakeCompatibility(bundle, req)
	if !report.Compatible || report.SummaryRequired || len(report.Reasons) != 0 {
		t.Fatalf("compatible report = %+v, want wake-compatible", report)
	}
	if len(report.Warnings) == 0 || report.Warnings[0] != "runtime_backend_changed" {
		t.Fatalf("warnings = %+v, want runtime backend warning", report.Warnings)
	}

	req.Tokenizer.Hash = "tok-b"
	req.Adapter = AdapterIdentity{}
	req.Model.ContextLength = 1024
	report = CheckWakeCompatibility(bundle, req)
	if report.Compatible || !report.SummaryRequired {
		t.Fatalf("incompatible report = %+v, want summary fallback", report)
	}
	if !stringSliceContains(report.Reasons, "tokenizer_hash_mismatch") || !stringSliceContains(report.Reasons, "adapter_missing") || !stringSliceContains(report.Reasons, "context_length_too_small") {
		t.Fatalf("reasons = %+v, want tokenizer, adapter, and context blockers", report.Reasons)
	}

	req = WakeRequest{
		Model:     ModelIdentity{Hash: "model-b", Architecture: "qwen3", NumLayers: 28, QuantBits: 8, ContextLength: 8192},
		Tokenizer: TokenizerIdentity{Hash: "tok-a", ChatTemplate: "chat-a"},
		Adapter:   AdapterIdentity{Hash: "adapter-a", Rank: 8},
		Runtime:   RuntimeIdentity{Backend: "metal", CacheMode: "paged-q8"},
	}
	report = CheckWakeCompatibility(bundle, req)
	if report.Compatible || !report.SummaryRequired {
		t.Fatalf("model-incompatible report = %+v, want summary fallback", report)
	}
	for _, want := range []string{"model_hash_mismatch", "model_architecture_mismatch", "model_quantisation_mismatch"} {
		if !stringSliceContains(report.Reasons, want) {
			t.Fatalf("reasons = %+v, want %s", report.Reasons, want)
		}
	}

	req.SkipCompatibilityCheck = true
	report = CheckWakeCompatibility(bundle, req)
	if !report.Compatible || len(report.Warnings) == 0 || report.Warnings[0] != "compatibility_check_skipped" {
		t.Fatalf("skip report = %+v, want forced compatibility warning", report)
	}
}

func stringSliceContains(values []string, want string) bool {
	for _, value := range values {
		if value == want {
			return true
		}
	}
	return false
}

// TestProjectSeed_NewProjectSeed_Ugly proves both ends of the default
// derivation: every field omitted falls back to package defaults, and
// every field supplied is used verbatim with no derivation at all.
func TestProjectSeed_NewProjectSeed_Ugly(t *testing.T) {
	seed := NewProjectSeed(ProjectSeedOptions{})
	if seed.BaseURI != "state://projects" || seed.ProjectID != "default" {
		t.Fatalf("seed = %+v, want default BaseURI/ProjectID", seed)
	}
	if seed.Title != "default project seed" {
		t.Fatalf("seed.Title = %q, want default-derived title", seed.Title)
	}
	if seed.EntryURI != "state://projects/default/seed" {
		t.Fatalf("seed.EntryURI = %q, want derived from defaults", seed.EntryURI)
	}

	explicit := NewProjectSeed(ProjectSeedOptions{
		BaseURI:   "state://lthn/projects",
		ProjectID: "core/go-mlx",
		EntryURI:  "state://custom/entry",
		BundleURI: "state://custom/entry/bundle",
		IndexURI:  "state://custom/entry/index",
		Title:     "custom title",
	})
	if explicit.EntryURI != "state://custom/entry" || explicit.BundleURI != "state://custom/entry/bundle" || explicit.IndexURI != "state://custom/entry/index" || explicit.Title != "custom title" {
		t.Fatalf("explicit seed = %+v, want caller URIs preserved verbatim", explicit)
	}
}

// TestProjectSeed_PlanContinuation_Ugly covers the mode-defaulting branch,
// the Hybrid mode (persist + summary together, still falling through to
// sleepRequest), and sleepRequest's own EntryURI auto-derivation when the
// caller doesn't supply one.
func TestProjectSeed_PlanContinuation_Ugly(t *testing.T) {
	seed := NewProjectSeed(ProjectSeedOptions{BaseURI: "state://lthn/projects", ProjectID: "core/go-mlx"})

	// An empty Mode defaults to state-checkpoint persistence.
	defaulted := seed.PlanContinuation(ProjectSeedContinuationOptions{})
	if defaulted.Mode != ProjectSeedStateCheckpoint || !defaulted.PersistState {
		t.Fatalf("defaulted plan = %+v, want state checkpoint persistence", defaulted)
	}

	// Hybrid mode persists state AND flags a summary is also needed,
	// unlike the mutually-exclusive checkpoint/summary-only modes, and
	// still falls through to sleepRequest (no early return).
	hybrid := seed.PlanContinuation(ProjectSeedContinuationOptions{Mode: ProjectSeedHybrid})
	if !hybrid.PersistState || !hybrid.NeedsSummary || hybrid.ReuseCurrentSeed {
		t.Fatalf("hybrid plan = %+v, want persist+summary both set", hybrid)
	}
	if hybrid.Sleep.EntryURI == "" {
		t.Fatalf("hybrid plan sleep = %+v, want a derived sleep request", hybrid.Sleep)
	}

	// EntryURI omitted from the continuation options — sleepRequest
	// derives it from the seed's base/project/checkpoints/latest path.
	derived := seed.PlanContinuation(ProjectSeedContinuationOptions{Mode: ProjectSeedStateCheckpoint})
	wantEntry := "state://lthn/projects/core/go-mlx/checkpoints/latest"
	if derived.Sleep.EntryURI != wantEntry {
		t.Fatalf("derived sleep EntryURI = %q, want %q", derived.Sleep.EntryURI, wantEntry)
	}
	if derived.Sleep.BundleURI != wantEntry+"/bundle" || derived.Sleep.IndexURI != wantEntry+"/index" {
		t.Fatalf("derived sleep = %+v, want bundle/index appended to the derived entry", derived.Sleep)
	}
}

// TestCompareModelIdentity_Bad exercises the two branches
// CheckWakeCompatibility's existing scenarios never trigger: a NumLayers
// mismatch, and the PromptTokens-only fallback when
// PromptTokens+GeneratedTokens collapses to <= 0.
func TestCompareModelIdentity_Bad(t *testing.T) {
	var report WakeCompatibilityReport
	bundle := Bundle{
		Model:           ModelIdentity{NumLayers: 28},
		PromptTokens:    0,
		GeneratedTokens: 0,
	}
	req := ModelIdentity{NumLayers: 16}

	compareModelIdentity(&report, bundle, req)

	if !stringSliceContains(report.Reasons, "model_layer_mismatch") {
		t.Fatalf("reasons = %+v, want model_layer_mismatch", report.Reasons)
	}
	if stringSliceContains(report.Reasons, "context_length_too_small") {
		t.Fatalf("reasons = %+v, want no context-length reason when prefixTokens collapses to 0", report.Reasons)
	}
}

// TestCompareTokenizerIdentity_Bad exercises the chat-template mismatch
// branch, which no existing CheckWakeCompatibility scenario triggers.
func TestCompareTokenizerIdentity_Bad(t *testing.T) {
	var report WakeCompatibilityReport
	compareTokenizerIdentity(&report, TokenizerIdentity{Hash: "tok-a", ChatTemplate: "chat-a"}, TokenizerIdentity{Hash: "tok-a", ChatTemplate: "chat-b"})

	if !stringSliceContains(report.Reasons, "chat_template_mismatch") {
		t.Fatalf("reasons = %+v, want chat_template_mismatch", report.Reasons)
	}
}

// TestCompareAdapterIdentity_Bad drives every switch case in
// compareAdapterIdentity individually — CheckWakeCompatibility's existing
// scenarios only ever reach the bundleActive&&!reqActive case.
func TestCompareAdapterIdentity_Bad(t *testing.T) {
	cases := []struct {
		name   string
		bundle AdapterIdentity
		req    AdapterIdentity
		reason string
	}{
		{"unexpected", AdapterIdentity{}, AdapterIdentity{Hash: "adapter-x", Rank: 8}, "adapter_unexpected"},
		{"hash mismatch", AdapterIdentity{Hash: "a"}, AdapterIdentity{Hash: "b"}, "adapter_hash_mismatch"},
		{"path mismatch", AdapterIdentity{Hash: "a", Path: "/a"}, AdapterIdentity{Hash: "a", Path: "/b"}, "adapter_path_mismatch"},
		{"rank mismatch", AdapterIdentity{Hash: "a", Path: "/a", Rank: 8}, AdapterIdentity{Hash: "a", Path: "/a", Rank: 16}, "adapter_rank_mismatch"},
	}
	for _, tc := range cases {
		var report WakeCompatibilityReport
		compareAdapterIdentity(&report, tc.bundle, tc.req)
		if !stringSliceContains(report.Reasons, tc.reason) {
			t.Fatalf("%s: reasons = %+v, want %s", tc.name, report.Reasons, tc.reason)
		}
	}
}

// TestCompareRuntimeIdentity_Bad exercises the cache-mode warning branch,
// which no existing CheckWakeCompatibility scenario triggers.
func TestCompareRuntimeIdentity_Bad(t *testing.T) {
	var report WakeCompatibilityReport
	compareRuntimeIdentity(&report, RuntimeIdentity{Backend: "metal", CacheMode: "paged"}, RuntimeIdentity{Backend: "metal", CacheMode: "paged-q4"})

	if !stringSliceContains(report.Warnings, "runtime_cache_mode_changed") {
		t.Fatalf("warnings = %+v, want runtime_cache_mode_changed", report.Warnings)
	}
}

// TestJoinURI_Ugly covers the skip-empty-part branch (in both internal
// passes) and the total-length-zero early return, neither of which the
// NewProjectSeed/PlanContinuation call sites ever trigger.
func TestJoinURI_Ugly(t *testing.T) {
	if got := joinURI("base", "", "tail"); got != "base/tail" {
		t.Fatalf(`joinURI("base", "", "tail") = %q, want "base/tail"`, got)
	}

	if got := joinURI("", "", ""); got != "" {
		t.Fatalf(`joinURI("", "", "") = %q, want ""`, got)
	}
}
