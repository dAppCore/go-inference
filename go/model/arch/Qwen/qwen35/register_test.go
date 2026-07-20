// SPDX-Licence-Identifier: EUPL-1.2

package qwen35

import (
	"testing"

	"dappco.re/go/inference/model"
)

// qwenHybridReleasedIDs is every model_type string the Qwen 3.6 hybrid has been released under: qwen3_5 /
// qwen3_5_moe (+ nested text_config aliases), qwen3_6 / qwen3_6_moe (the same hybrid, its other released
// name), and qwen3_next (the predecessor release). One architecture, three names, one ArchSpec (#18/#50).
var qwenHybridReleasedIDs = []string{
	"qwen3_5", "qwen3_5_text", "qwen3_5_moe", "qwen3_5_moe_text",
	"qwen3_6", "qwen3_6_moe", "qwen3_next",
}

// TestArchSpecRegistration_Good is the #50 archzoo bar for qwen3_6/qwen3_6_moe/qwen3_next: every released
// name of the hybrid resolves through the registry to a DUAL-route spec — Parse + Weights (the factory
// route, model.Assemble + arch_session) AND a Composed hook (model/composed — the A/B reference / escape
// hatch, unchanged) both present. Parse != nil is precisely the condition model.Load checks before
// rejecting a composed-only arch (load.go: "spec.Composed != nil && spec.Parse == nil"), so this is also
// the direct proof none of these three ids requires model/composed to load any more.
func TestArchSpecRegistration_Good(t *testing.T) {
	want := WeightNames()
	for _, mt := range qwenHybridReleasedIDs {
		spec, ok := model.LookupArch(mt)
		if !ok {
			t.Fatalf("model_type %q not registered", mt)
		}
		if spec.Parse == nil {
			t.Fatalf("model_type %q: Parse is nil — model.Load would still reject this as composed-only", mt)
		}
		if spec.Weights != want {
			t.Fatalf("model_type %q: Weights = %+v, want the shared qwen35 WeightNames()", mt, spec.Weights)
		}
		if spec.Composed == nil {
			t.Fatalf("model_type %q: Composed hook is nil — the composed escape hatch/A-B reference is gone", mt)
		}
	}
}

// TestArchSpecRegistration_Bad proves an unrelated model_type does NOT resolve to qwen35's WeightNames —
// a registry bug (e.g. a stray empty-string ModelTypes entry, which Set silently no-ops on) would make
// this package's Weights leak onto an arch it never declared.
func TestArchSpecRegistration_Bad(t *testing.T) {
	qwen35Weights := WeightNames()
	for _, mt := range []string{"llama", "gemma3", "mixtral", "qwen3_5x", "qwen3_60", "not-a-real-model-type"} {
		spec, ok := model.LookupArch(mt)
		if !ok {
			continue // unregistered (or, for the two near-miss qwen ids, correctly not an alias) — fine
		}
		if spec.Weights == qwen35Weights {
			t.Fatalf("model_type %q resolved to qwen35's exact WeightNames — registration boundary leaked", mt)
		}
	}
}

// TestArchSpecRegistration_Ugly proves qwenHybridReleasedIDs itself has no accidental duplicate — a
// repeated id would silently double-register the same ModelTypes entry (harmless with RegisterArch's
// last-wins Set, but a sign of a copy-paste mistake in the list this test and register.go both carry).
func TestArchSpecRegistration_Ugly(t *testing.T) {
	seen := make(map[string]bool, len(qwenHybridReleasedIDs))
	for _, mt := range qwenHybridReleasedIDs {
		if seen[mt] {
			t.Fatalf("qwenHybridReleasedIDs lists %q twice", mt)
		}
		seen[mt] = true
	}
}
