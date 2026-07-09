// SPDX-Licence-Identifier: EUPL-1.2

package session

import (
	"testing"

	core "dappco.re/go"
)

// TestDefaults_DefaultNewSessionText_Good confirms the engine-facing alias
// stays byte-identical to the Lemma-family seed text it wraps — frameworks
// that read DefaultNewSessionText and code that reads
// DefaultLemmaNewSessionText directly must observe the same seed.
func TestDefaults_DefaultNewSessionText_Good(t *testing.T) {
	if DefaultNewSessionText != DefaultLemmaNewSessionText {
		t.Fatalf("DefaultNewSessionText = %q, want alias of DefaultLemmaNewSessionText %q",
			DefaultNewSessionText, DefaultLemmaNewSessionText)
	}
}

// TestDefaults_DefaultLemmaNewSessionText_Good asserts the seed text carries
// the Lemma/Lethean identity phrasing every caller depends on to open a
// session before the first real user prompt arrives.
func TestDefaults_DefaultLemmaNewSessionText_Good(t *testing.T) {
	if DefaultLemmaNewSessionText == "" {
		t.Fatal("DefaultLemmaNewSessionText is empty, want a non-empty seed")
	}
	if !core.Contains(DefaultLemmaNewSessionText, "Lemma") {
		t.Fatalf("DefaultLemmaNewSessionText = %q, want it to name Lemma", DefaultLemmaNewSessionText)
	}
	if !core.Contains(DefaultLemmaNewSessionText, "Lethean") {
		t.Fatalf("DefaultLemmaNewSessionText = %q, want it to name Lethean", DefaultLemmaNewSessionText)
	}
}
