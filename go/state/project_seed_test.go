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
