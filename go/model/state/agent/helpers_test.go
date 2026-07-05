// SPDX-Licence-Identifier: EUPL-1.2

package agent

import (
	"testing"

	"dappco.re/go/inference/model/bundle"
)

// --- firstNonEmpty / firstNonEmptyString ----------------------------------

func TestHelpers_firstNonEmpty_Good(t *testing.T) {
	if got := firstNonEmpty("primary", "fallback"); got != "primary" {
		t.Fatalf("firstNonEmpty = %q, want first non-empty", got)
	}
	if got := firstNonEmpty("", "fallback"); got != "fallback" {
		t.Fatalf("firstNonEmpty = %q, want fallback when first empty", got)
	}
}

func TestHelpers_firstNonEmpty_Bad(t *testing.T) {
	if got := firstNonEmpty(); got != "" {
		t.Fatalf("firstNonEmpty() = %q, want empty for no args", got)
	}
	if got := firstNonEmpty("", "", ""); got != "" {
		t.Fatalf("firstNonEmpty(all empty) = %q, want empty", got)
	}
}

func TestHelpers_firstNonEmpty_Ugly(t *testing.T) {
	// A whitespace-only string is treated as empty (Trim'd), so the
	// later real value wins.
	if got := firstNonEmpty("   ", "\t\n", "real"); got != "real" {
		t.Fatalf("firstNonEmpty(whitespace…) = %q, want real", got)
	}
	// The legacy alias must behave identically.
	if got := firstNonEmptyString("   ", "alias"); got != "alias" {
		t.Fatalf("firstNonEmptyString(whitespace) = %q, want alias", got)
	}
}

// --- stateHash ------------------------------------------------------------

func TestHelpers_stateHash_Good(t *testing.T) {
	h := stateHash("hello")
	if len(h) != 64 {
		t.Fatalf("stateHash length = %d, want 64-hex SHA-256", len(h))
	}
	// Deterministic: matches bundle.HashString (the canonical helper it
	// forwards to).
	if h != bundle.HashString("hello") {
		t.Fatalf("stateHash = %q, want bundle.HashString equivalence", h)
	}
}

func TestHelpers_stateHash_Bad(t *testing.T) {
	// Empty input short-circuits to the empty string (bundle.HashString
	// returns "" for "" rather than hashing the empty byte slice).
	if h := stateHash(""); h != "" {
		t.Fatalf("stateHash(\"\") = %q, want empty string", h)
	}
}

func TestHelpers_stateHash_Ugly(t *testing.T) {
	// Distinct inputs produce distinct digests (collision-free over a
	// trivial pair).
	if stateHash("a") == stateHash("b") {
		t.Fatal("stateHash collided for distinct inputs")
	}
}

// --- stateBundleTokenizer -------------------------------------------------

func TestHelpers_stateBundleTokenizer_Good(t *testing.T) {
	// A fully-populated tokenizer passes through with its hashes intact.
	in := bundle.Tokenizer{Hash: "tok-a", ChatTemplateHash: "chat-a"}
	out := stateBundleTokenizer(in)
	if out.Hash != "tok-a" || out.ChatTemplateHash != "chat-a" {
		t.Fatalf("stateBundleTokenizer = %+v, want hashes preserved", out)
	}
}

func TestHelpers_stateBundleTokenizer_Bad(t *testing.T) {
	// A zero-value tokenizer is normalised; NormaliseTokenizer leaves the
	// hash empty when there is no source path to derive one from, so the
	// result must remain a valid (empty-hash) value rather than panic.
	out := stateBundleTokenizer(bundle.Tokenizer{})
	if out != bundle.NormaliseTokenizer(bundle.Tokenizer{}) {
		t.Fatalf("stateBundleTokenizer(zero) = %+v, want NormaliseTokenizer equivalence", out)
	}
}

// --- cloneStringMap -------------------------------------------------------

func TestHelpers_cloneStringMap_Good(t *testing.T) {
	src := map[string]string{"session_id": "s-1", "agent": "cladius"}
	clone := cloneStringMap(src)
	if len(clone) != 2 || clone["session_id"] != "s-1" || clone["agent"] != "cladius" {
		t.Fatalf("clone = %+v, want copy of src", clone)
	}
	// Mutating the clone must not touch the source.
	clone["agent"] = "mutated"
	if src["agent"] != "cladius" {
		t.Fatalf("source mutated through clone: %q", src["agent"])
	}
}

func TestHelpers_cloneStringMap_Bad(t *testing.T) {
	if clone := cloneStringMap(nil); clone != nil {
		t.Fatalf("cloneStringMap(nil) = %+v, want nil", clone)
	}
}

func TestHelpers_cloneStringMap_Ugly(t *testing.T) {
	// An empty (non-nil) map clones to nil — the helper short-circuits on
	// len == 0 rather than allocating an empty map.
	if clone := cloneStringMap(map[string]string{}); clone != nil {
		t.Fatalf("cloneStringMap(empty) = %+v, want nil", clone)
	}
}
