// SPDX-Licence-Identifier: EUPL-1.2

package openai

import "testing"

// TestStops_IndexString_Good pins core.Index delegation for the
// ordinary match / no-match cases.
func TestStops_IndexString_Good(t *testing.T) {
	if got := indexString("hello world", "world"); got != 6 {
		t.Fatalf("indexString(match) = %d, want 6", got)
	}
	if got := indexString("hello world", "xyz"); got != -1 {
		t.Fatalf("indexString(no match) = %d, want -1", got)
	}
}

// TestStops_IndexString_Bad covers the empty-needle contract: an
// empty stop string must report "no match" (-1), not
// strings.Index's "match at 0" semantics — otherwise every stream
// would immediately truncate at position 0.
func TestStops_IndexString_Bad(t *testing.T) {
	if got := indexString("hello", ""); got != -1 {
		t.Fatalf("indexString(empty needle) = %d, want -1", got)
	}
}

// TestStops_FirstStopSequenceCut_Good covers the earliest-match-wins
// selection across multiple candidate stop sequences.
func TestStops_FirstStopSequenceCut_Good(t *testing.T) {
	cut, ok := firstStopSequenceCut("The answer is 42. END more text.", []string{"END", "42"})
	if !ok || cut != 14 {
		t.Fatalf("firstStopSequenceCut() = %d, %v, want 14, true (earliest match wins)", cut, ok)
	}
}

// TestStops_FirstStopSequenceCut_Bad covers the fast-exit branches:
// empty content, no stop sequences configured, and no stop sequence
// present in the content.
func TestStops_FirstStopSequenceCut_Bad(t *testing.T) {
	if cut, ok := firstStopSequenceCut("", []string{"END"}); ok || cut != 0 {
		t.Fatalf("firstStopSequenceCut(empty content) = %d, %v, want 0, false", cut, ok)
	}
	if cut, ok := firstStopSequenceCut("hello", nil); ok || cut != 0 {
		t.Fatalf("firstStopSequenceCut(no stops) = %d, %v, want 0, false", cut, ok)
	}
	if cut, ok := firstStopSequenceCut("hello world", []string{"END", "STOP"}); ok || cut != 0 {
		t.Fatalf("firstStopSequenceCut(no match) = %d, %v, want 0, false", cut, ok)
	}
}

// TestStops_TruncateAtStopSequence_Good covers the cut-applied path.
func TestStops_TruncateAtStopSequence_Good(t *testing.T) {
	got := TruncateAtStopSequence("Answer: 42.END trailing", []string{"END"})
	if got != "Answer: 42." {
		t.Fatalf("TruncateAtStopSequence() = %q, want %q", got, "Answer: 42.")
	}
}

// TestStops_TruncateAtStopSequence_Bad covers the pass-through path
// when no stop sequence matches — the original content returns
// unmodified (not a copy check, an identity-of-value check).
func TestStops_TruncateAtStopSequence_Bad(t *testing.T) {
	got := TruncateAtStopSequence("no stop sequence here", []string{"END"})
	if got != "no stop sequence here" {
		t.Fatalf("TruncateAtStopSequence() = %q, want unmodified content", got)
	}
}
