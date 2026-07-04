// SPDX-Licence-Identifier: EUPL-1.2

package spine

import "testing"

// Tests for PromptChunksToString — the iter.Seq[string] concatenation
// helper. Good = a multi-chunk sequence joins in order; Bad = the nil
// sequence early return; Ugly = a single empty chunk (drains the loop
// but produces nothing).

func TestPrompt_PromptChunksToString_Good(t *testing.T) {
	chunks := func(yield func(string) bool) {
		for _, s := range []string{"foo", "bar", "baz"} {
			if !yield(s) {
				return
			}
		}
	}
	if got := PromptChunksToString(chunks); got != "foobarbaz" {
		t.Fatalf("PromptChunksToString = %q, want %q", got, "foobarbaz")
	}
}

func TestPrompt_PromptChunksToString_Bad(t *testing.T) {
	// Nil sequence → empty string via the early return, no builder.
	if got := PromptChunksToString(nil); got != "" {
		t.Fatalf("PromptChunksToString(nil) = %q, want empty", got)
	}
}

func TestPrompt_PromptChunksToString_Ugly(t *testing.T) {
	// A sequence yielding only an empty chunk drains the loop but appends
	// nothing — distinct from the nil early return.
	empty := func(yield func(string) bool) { yield("") }
	if got := PromptChunksToString(empty); got != "" {
		t.Fatalf("PromptChunksToString(empty chunk) = %q, want empty", got)
	}
}
