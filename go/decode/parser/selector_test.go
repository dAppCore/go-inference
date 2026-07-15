// SPDX-Licence-Identifier: EUPL-1.2

package parser

import "testing"

// TestSelector_NormaliseKey_Good pins the common case: an already-canonical
// key (lowercase, no `-`/`.`) passes through unchanged.
func TestSelector_NormaliseKey_Good(t *testing.T) {
	if got := NormaliseKey("qwen3"); got != "qwen3" {
		t.Fatalf("NormaliseKey(qwen3) = %q, want qwen3", got)
	}
}

// TestSelector_NormaliseKey_Bad pins the transform case: mixed case plus
// `-`/`.` separators lowercase and normalise to underscores in one pass.
func TestSelector_NormaliseKey_Bad(t *testing.T) {
	if got := NormaliseKey("Qwen-3.5"); got != "qwen_3_5" {
		t.Fatalf("NormaliseKey(Qwen-3.5) = %q, want qwen_3_5", got)
	}
}

// TestSelector_NormaliseKey_Ugly pins the boundary: an empty/whitespace-only
// value normalises to the empty string rather than panicking or echoing
// whitespace.
func TestSelector_NormaliseKey_Ugly(t *testing.T) {
	if got := NormaliseKey(""); got != "" {
		t.Fatalf("NormaliseKey(empty) = %q, want empty", got)
	}
	if got := NormaliseKey("   "); got != "" {
		t.Fatalf("NormaliseKey(whitespace) = %q, want empty", got)
	}
}

// TestSelector_Family_Good pins the direct architecture match.
func TestSelector_Family_Good(t *testing.T) {
	if got := Family(Hint{Architecture: "qwen3"}); got != "qwen" {
		t.Fatalf("Family(qwen3) = %q, want qwen", got)
	}
	if got := Family(Hint{Architecture: "gemma4_text"}); got != "gemma" {
		t.Fatalf("Family(gemma4_text) = %q, want gemma", got)
	}
}

// TestSelector_Family_Bad pins the unresolvable case: an architecture that
// matches no known family falls back to "generic" rather than erroring.
func TestSelector_Family_Bad(t *testing.T) {
	if got := Family(Hint{Architecture: "not-a-real-arch"}); got != "generic" {
		t.Fatalf("Family(unknown) = %q, want generic", got)
	}
}

// TestSelector_Family_Ugly pins the adapter-name fallback: a family that
// only shows up in AdapterName (not Architecture) still resolves, and the
// architecture/adapter pair is scanned independently — a needle can never
// straddle the boundary between them.
func TestSelector_Family_Ugly(t *testing.T) {
	got := Family(Hint{Architecture: "unknown-base", AdapterName: "lora-kimi"})
	if got != "kimi" {
		t.Fatalf("Family(adapter-only match) = %q, want kimi", got)
	}
}
