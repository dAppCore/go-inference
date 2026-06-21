// SPDX-Licence-Identifier: EUPL-1.2

package ngram

import "testing"

// eq reports whether two token slices are element-wise equal. A nil slice and an
// empty slice are treated as equal (both mean "no draft proposed").
//
//	if !eq(d.Draft(ctx), []int{4, 5}) { t.Fatal("mismatch") }
func eq(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// TestNgram_Draft_Good is the canonical prompt-lookup case: a phrase repeats, so
// the suffix of the context matches an earlier occurrence and the drafter
// proposes the tokens that followed it last time.
func TestNgram_Draft_Good(t *testing.T) {
	d := New(Config{MaxNgram: 3, MaxDraft: 4})

	// "the quick brown fox jumps ... the quick brown" → predict "fox jumps".
	// tokens:   1   2     3    4    5      9   1   2     3
	// Suffix [1 2 3] last occurred at index 0, followed by [4 5] then the
	// barrier token 9 — so the four tokens after the match are [4 5 9 1].
	ctx := []int{1, 2, 3, 4, 5, 9, 1, 2, 3}
	got := d.Draft(ctx)
	if want := []int{4, 5, 9, 1}; !eq(got, want) {
		t.Fatalf("repeated phrase should predict the following tokens: want %v, got %v", want, got)
	}
}

// TestNgram_Draft_LongestSuffixWins covers the longest-suffix preference: when
// both a short and a long suffix match earlier text but point at DIFFERENT
// continuations, the longest matching n-gram must win (it is the more specific,
// higher-confidence prediction).
func TestNgram_Draft_LongestSuffixWins(t *testing.T) {
	d := New(Config{MaxNgram: 3, MaxDraft: 2})

	// Suffix [2 3] (n=2) last appeared followed by 7.
	// Suffix [5 2 3] (n=3) appeared earlier followed by 4.
	// The trailing context is [... 5 2 3]; n=3 must win → predict 4, not 7.
	//          0  1  2  3  4  5  6  7  8  9
	ctx := []int{5, 2, 3, 4, 8, 2, 3, 7, 5, 2, 3}
	got := d.Draft(ctx)
	if want := []int{4, 8}; !eq(got, want) {
		t.Fatalf("longest matching suffix must win: want %v (n=3 match), got %v", want, got)
	}
}

// TestNgram_Draft_MostRecentOccurrence covers tie-breaking by recency: the SAME
// suffix appears more than once earlier in the context, each followed by a
// different token. Prompt-lookup picks the MOST RECENT earlier occurrence.
func TestNgram_Draft_MostRecentOccurrence(t *testing.T) {
	d := New(Config{MaxNgram: 2, MaxDraft: 1})

	// Suffix [1 2] appears at index 0 (→ 3) and index 4 (→ 9). The trailing
	// [1 2] is index 7. Most-recent earlier occurrence is index 4 → predict 9.
	//          0  1  2  3  4  5  6  7  8
	ctx := []int{1, 2, 3, 0, 1, 2, 9, 1, 2}
	got := d.Draft(ctx)
	if want := []int{9}; !eq(got, want) {
		t.Fatalf("most-recent earlier occurrence should be chosen: want %v, got %v", want, got)
	}
}

// TestNgram_Draft_MaxDraftCaps covers the MaxDraft cap: even when many tokens
// follow the matched occurrence, the drafter proposes at most MaxDraft of them.
func TestNgram_Draft_MaxDraftCaps(t *testing.T) {
	d := New(Config{MaxNgram: 2, MaxDraft: 2})

	// [1 2] first followed by [3 4 5 6]; trailing [1 2] → propose only 2: [3 4].
	//          0  1  2  3  4  5  6  7  8
	ctx := []int{1, 2, 3, 4, 5, 6, 0, 1, 2}
	got := d.Draft(ctx)
	if want := []int{3, 4}; !eq(got, want) {
		t.Fatalf("draft must be capped at MaxDraft: want %v, got %v", want, got)
	}
}

// TestNgram_Draft_FewerThanMaxDraft covers the tail-clamp: when fewer than
// MaxDraft tokens follow the match (the match is near the end), the drafter
// returns only the tokens that actually exist, never reading past the end.
func TestNgram_Draft_FewerThanMaxDraft(t *testing.T) {
	d := New(Config{MaxNgram: 2, MaxDraft: 5})

	// [5 6] first occurs at index 0; the tokens after it run to the end of the
	// context (indices 2..5 = [7 8 5 6]) — only 4 tokens, fewer than MaxDraft 5,
	// so the draft clamps to those 4 and never reads past the end.
	//          0  1  2  3  4  5
	ctx := []int{5, 6, 7, 8, 5, 6}
	got := d.Draft(ctx)
	if want := []int{7, 8, 5, 6}; !eq(got, want) {
		t.Fatalf("draft should clamp to available tokens (fewer than MaxDraft): want %v, got %v", want, got)
	}
}

// TestNgram_Draft_Bad covers the no-match arm: a context with no repeated suffix
// yields an empty draft (the target model just decodes normally).
func TestNgram_Draft_Bad(t *testing.T) {
	d := New(Config{MaxNgram: 3, MaxDraft: 4})

	got := d.Draft([]int{1, 2, 3, 4, 5})
	if len(got) != 0 {
		t.Fatalf("no repeated suffix → empty draft, got %v", got)
	}
}

// TestNgram_Draft_Ugly covers the degenerate inputs that must not panic and must
// return an empty draft: nil context, context shorter than a single-token
// suffix's match window, and a single-element context (no earlier occurrence
// possible).
func TestNgram_Draft_Ugly(t *testing.T) {
	d := New(Config{MaxNgram: 3, MaxDraft: 4})

	if got := d.Draft(nil); len(got) != 0 {
		t.Fatalf("nil context → empty draft, got %v", got)
	}
	if got := d.Draft([]int{}); len(got) != 0 {
		t.Fatalf("empty context → empty draft, got %v", got)
	}
	if got := d.Draft([]int{42}); len(got) != 0 {
		t.Fatalf("single-token context has no earlier occurrence → empty, got %v", got)
	}
	// Context shorter than MaxNgram still drafts via shorter n: [7 7] has a
	// 1-gram suffix [7] whose earlier occurrence (index 0) is followed by 7.
	if got := d.Draft([]int{7, 7}); !eq(got, []int{7}) {
		t.Fatalf("short context should fall back to shorter n: want [7], got %v", got)
	}
}

// TestNgram_Draft_ZeroNgramClampedUgly covers config clamping: MaxNgram <= 0 is
// nonsense and is clamped to 1 (still a usable 1-gram drafter), and MaxDraft <= 0
// is clamped to 1 so a match always proposes at least one token.
func TestNgram_Draft_ZeroNgramClampedUgly(t *testing.T) {
	d := New(Config{MaxNgram: 0, MaxDraft: 0})

	// 1-gram on [5 5]: suffix [5] matched at index 0 → propose 1 token: [5].
	got := d.Draft([]int{5, 5})
	if want := []int{5}; !eq(got, want) {
		t.Fatalf("clamped config should still draft: want %v, got %v", want, got)
	}
}

// TestNgram_Draft_NoSelfMatchUgly guards the off-by-one that would let the
// trailing suffix match ITSELF: with no genuinely-earlier occurrence the draft
// must be empty, never the suffix pointing at its own following position.
func TestNgram_Draft_NoSelfMatchUgly(t *testing.T) {
	d := New(Config{MaxNgram: 2, MaxDraft: 3})

	// [9 8] appears only once (as the trailing suffix). No earlier [9 8] → empty.
	got := d.Draft([]int{1, 2, 3, 9, 8})
	if len(got) != 0 {
		t.Fatalf("trailing suffix must not match itself: want empty, got %v", got)
	}
}

// TestNgram_Update_Good covers the running-context composition: after Update
// appends accepted tokens, DraftNext reflects them — the drafter's internal
// context grows so later drafts see the newly-accepted text.
func TestNgram_Update_Good(t *testing.T) {
	d := New(Config{MaxNgram: 2, MaxDraft: 2})

	// Seed a repeated phrase via Update, then DraftNext should predict from it.
	d.Update([]int{1, 2, 3, 9}) // context = [1 2 3 9]
	d.Update([]int{1, 2})       // context = [1 2 3 9 1 2] → suffix [1 2] → predict 3
	got := d.DraftNext()
	if want := []int{3, 9}; !eq(got, want) {
		t.Fatalf("DraftNext should reflect appended context: want %v, got %v", want, got)
	}
}

// TestNgram_Update_Bad covers DraftNext on an empty running context: with nothing
// appended yet there is no context to draft from, so the result is empty.
func TestNgram_Update_Bad(t *testing.T) {
	d := New(Config{MaxNgram: 3, MaxDraft: 4})

	if got := d.DraftNext(); len(got) != 0 {
		t.Fatalf("DraftNext on empty context → empty, got %v", got)
	}
	if got := d.Context(); len(got) != 0 {
		t.Fatalf("fresh drafter has empty context, got %v", got)
	}
}

// TestNgram_Update_Ugly covers the no-op appends: Update(nil) and Update of an
// empty slice must not change the running context or panic, and Context returns a
// copy so callers cannot mutate the drafter's internal buffer through it.
func TestNgram_Update_Ugly(t *testing.T) {
	d := New(Config{MaxNgram: 2, MaxDraft: 2})

	d.Update(nil)
	d.Update([]int{})
	if got := d.Context(); len(got) != 0 {
		t.Fatalf("no-op Update must leave context empty, got %v", got)
	}

	d.Update([]int{1, 2, 3})
	snap := d.Context()
	if !eq(snap, []int{1, 2, 3}) {
		t.Fatalf("Context should mirror appended tokens: got %v", snap)
	}
	// Mutating the returned snapshot must not corrupt the drafter's buffer.
	snap[0] = 999
	if again := d.Context(); !eq(again, []int{1, 2, 3}) {
		t.Fatalf("Context must return a copy, not the live buffer: got %v", again)
	}
}

// TestNgram_Reset_Ugly covers Reset: it clears the running context so a reused
// drafter starts a fresh sequence without allocating a new one.
func TestNgram_Reset_Ugly(t *testing.T) {
	d := New(Config{MaxNgram: 2, MaxDraft: 2})

	d.Update([]int{1, 2, 1})
	if got := d.DraftNext(); len(got) == 0 {
		t.Fatalf("setup: expected a draft before reset, got empty")
	}
	d.Reset()
	if got := d.Context(); len(got) != 0 {
		t.Fatalf("Reset must clear the context, got %v", got)
	}
	if got := d.DraftNext(); len(got) != 0 {
		t.Fatalf("DraftNext after Reset → empty, got %v", got)
	}
}
