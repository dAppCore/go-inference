// SPDX-Licence-Identifier: EUPL-1.2

// Internal coverage tests for architecture.go helpers whose final fallback arms
// are unreachable through the public API (no built-in profile drives them) but
// are part of the helper contract — every built-in profile carries a non-empty
// id, so the "empty id" branch can only be reached by calling the helper directly.

package profile

import "testing"

// TestArchitectureCoverageInternal_DefaultChatTemplate_GenericFallback exercises
// the terminal "return generic" arm of architectureDefaultChatTemplate: an
// architecture whose family is not a known template family AND whose id is empty
// falls back to the generic template. The id-bearing default (an unknown family
// with a non-empty id) is already exercised by the phi profile at build time;
// only the empty-id corner needs a direct call.
func TestArchitectureCoverageInternal_DefaultChatTemplate_GenericFallback(t *testing.T) {
	if got := architectureDefaultChatTemplate("unrecognised_family", "", false); got != "generic" {
		t.Fatalf("architectureDefaultChatTemplate(unknown family, empty id) = %q, want generic", got)
	}
	// A known template family is returned verbatim regardless of id, and the
	// embeddings flag forces an empty template — pinned here so the generic
	// fallback is distinguished from those neighbouring arms.
	if got := architectureDefaultChatTemplate("gemma", "", false); got != "gemma" {
		t.Fatalf("architectureDefaultChatTemplate(gemma, empty id) = %q, want gemma", got)
	}
	if got := architectureDefaultChatTemplate("unrecognised_family", "some_id", true); got != "" {
		t.Fatalf("architectureDefaultChatTemplate(embeddings) = %q, want empty", got)
	}
}
