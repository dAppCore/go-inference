// SPDX-Licence-Identifier: EUPL-1.2

package lek

import "testing"

// Coverage-completion tests for metaphone.go. These target the
// per-consonant doubled-letter "skip the second letter" arms (B, K, M,
// N, R, V → i+2), the DG→TK arm, the mid-word XC/ZH/ZZ arms, and the
// truncate clip + unknown-character fall-through that the primary
// metaphone_test.go suite does not reach. Expected codes are locked to
// THIS implementation's actual output (the package diverges from
// textbook Double Metaphone for some edge cases — same convention as
// TestMetaphone_StepCBranches).

// TestMetaphone_DoubledConsonantSkips — each row contains a doubled
// consonant whose handler consumes both letters (returns i+2): BB, KK,
// MM, NN, RR, VV. The second letter is skipped, so the code carries a
// single phoneme for the pair.
func TestMetaphone_DoubledConsonantSkips(t *testing.T) {
	cases := []struct {
		word         string
		wantP, wantS string
	}{
		{"rubber", "RPR", "RPR"},    // BB → P, skip second B
		{"trekker", "TRKR", "TRKR"}, // KK → K, skip second K
		{"summer", "SMR", "SMR"},    // MM → M, skip second M
		{"dinner", "TNR", "TNR"},    // NN → N, skip second N
		{"berry", "PR", "PR"},       // RR → R, skip second R
		{"savvy", "SF", "SF"},       // VV → F, skip second V
	}
	for _, c := range cases {
		p, s, ok := DoubleMetaphone(c.word)
		if !ok {
			t.Errorf("DoubleMetaphone(%q): ok=false, want true", c.word)
			continue
		}
		if p != c.wantP || s != c.wantS {
			t.Errorf("DoubleMetaphone(%q) = (%q,%q), want (%q,%q)",
				c.word, p, s, c.wantP, c.wantS)
		}
	}
}

// TestMetaphone_DG_TK — DG followed by a NON-(E/I/Y) letter takes the
// "TK" arm of step (knowledge/judge take the J arm; this is the other
// side). "hodgkin" (DGK) and "midgut" (DGU) both reach it.
func TestMetaphone_DG_TK(t *testing.T) {
	cases := []struct {
		word         string
		wantP, wantS string
	}{
		{"hodgkin", "HTKK", "HTKK"}, // DGK → TK
		{"midgut", "MTKT", "MTKT"},  // DGU → TK
	}
	for _, c := range cases {
		p, s, ok := DoubleMetaphone(c.word)
		if !ok {
			t.Errorf("DoubleMetaphone(%q): ok=false, want true", c.word)
			continue
		}
		if p != c.wantP || s != c.wantS {
			t.Errorf("DoubleMetaphone(%q) = (%q,%q), want (%q,%q)",
				c.word, p, s, c.wantP, c.wantS)
		}
	}
}

// TestMetaphone_XC_ZH_ZZ — the mid-word X followed by C (XC consumes
// both), ZH → J, and ZZ → S/TS (the Italian /ts/ secondary), each
// taking its dedicated arm in step.
func TestMetaphone_XC_ZH_ZZ(t *testing.T) {
	cases := []struct {
		word         string
		wantP, wantS string
	}{
		{"exceed", "AKST", "AKST"}, // XC → consume both (KS then C handled)
		{"zhao", "J", "J"},         // ZH → J
		{"buzz", "PS", "PTS"},      // ZZ → S primary, TS secondary
		{"jazz", "JS", "ATS"},      // ZZ end-of-word with initial-J alt
	}
	for _, c := range cases {
		p, s, ok := DoubleMetaphone(c.word)
		if !ok {
			t.Errorf("DoubleMetaphone(%q): ok=false, want true", c.word)
			continue
		}
		if p != c.wantP || s != c.wantS {
			t.Errorf("DoubleMetaphone(%q) = (%q,%q), want (%q,%q)",
				c.word, p, s, c.wantP, c.wantS)
		}
	}
}

// TestMetaphone_TruncateClips — truncate's clip arm (len(b) > n) fires
// when the raw code exceeds MetaphoneMaxCode. A long, consonant-dense
// word produces a code that must be cut. Verifies the returned code is
// exactly the cap length (so the clip path, not the pass-through, ran).
func TestMetaphone_TruncateClips(t *testing.T) {
	// Direct unit test of truncate so the clip branch is unambiguous.
	long := []byte("ABCDEFG")
	got := truncate(long, MetaphoneMaxCode)
	if len(got) != MetaphoneMaxCode {
		t.Fatalf("truncate(len=7, %d) len = %d, want %d", MetaphoneMaxCode, len(got), MetaphoneMaxCode)
	}
	if string(got) != "ABCD" {
		t.Errorf("truncate clip = %q, want %q", got, "ABCD")
	}
	// And the pass-through arm (len(b) <= n) returns the slice unchanged.
	short := []byte("AB")
	if g := truncate(short, MetaphoneMaxCode); string(g) != "AB" {
		t.Errorf("truncate pass-through = %q, want %q", g, "AB")
	}
}

// TestMetaphone_EncodeMetaphoneUnknownChar — encodeMetaphone skips
// resetFromRaw's letter filter, so a non-letter byte reaches step and
// falls through its switch to the "unknown character — skip" arm
// (return i+1) without panicking. Letters around it still encode.
func TestMetaphone_EncodeMetaphoneUnknownChar(t *testing.T) {
	// "A1B" — '1' is not a letter; step skips it, A and B encode.
	pri, alt := encodeMetaphone("A1B")
	if string(pri) != "AP" || string(alt) != "AP" {
		t.Errorf("encodeMetaphone(%q) = (%q,%q), want (%q,%q)",
			"A1B", pri, alt, "AP", "AP")
	}
}

// TestMetaphone_StepEdgeArms — step / stepC / stepG / stepJ / stepS
// arms the primary suite leaves out: plain mid/end X (no CX/XX), plain
// Z (no ZH/ZZ), CK→K, GH-after-consonant→K, mid-word GN→KN, and CIO→S/X.
// Codes locked to this implementation's actual output.
func TestMetaphone_StepEdgeArms(t *testing.T) {
	cases := []struct {
		word         string
		wantP, wantS string
	}{
		{"fox", "FKS", "FKS"},      // X with no following C/X → +1 arm
		{"zone", "SN", "TSN"},      // Z with no ZH/ZZ → S / TS, +1 arm
		{"back", "PK", "PK"},       // CK → K (stepC)
		{"afghan", "AFKN", "AFKN"}, // GH after consonant (F) → K (stepG)
		{"signal", "SKNL", "SNL"},  // mid-word GN → KN / N (stepG)
		{"vicious", "FSS", "FXS"},  // CIO → S / X (stepC, Italian)
	}
	for _, c := range cases {
		p, s, ok := DoubleMetaphone(c.word)
		if !ok {
			t.Errorf("DoubleMetaphone(%q): ok=false, want true", c.word)
			continue
		}
		if p != c.wantP || s != c.wantS {
			t.Errorf("DoubleMetaphone(%q) = (%q,%q), want (%q,%q)",
				c.word, p, s, c.wantP, c.wantS)
		}
	}
}

// TestMetaphone_StepSlavoGermanicArms — the SlavoGermanic-context arms
// of stepC / stepG / stepJ / stepS. A leading K or W flags the word
// SlavoGermanic (detectSlavoGermanic), flipping C-before-E/I/Y, GE/GI,
// vowel-J-vowel and SIO/SIA onto their Germanic encodings. These use
// constructed letter tokens (no common English word combines a
// SlavoGermanic trigger with these clusters); the encoder operates on
// the byte pattern, not a dictionary.
func TestMetaphone_StepSlavoGermanicArms(t *testing.T) {
	cases := []struct {
		word         string
		wantP, wantS string
		why          string
	}{
		{"kice", "KS", "KS", "slavo + CI → S/S (not S/X)"},
		{"wcyc", "SK", "SK", "slavo + CY → S/S"},
		{"kgin", "KKN", "KKN", "slavo + GI → K (not J)"},
		{"kgel", "KKL", "KKL", "slavo + GE → K"},
		{"krejak", "KRJK", "KRAK", "slavo vowel-J-vowel → J/A glide"},
		{"ksiok", "KSK", "KSK", "slavo + SIO → S/S (not S/X)"},
	}
	for _, c := range cases {
		p, s, ok := DoubleMetaphone(c.word)
		if !ok {
			t.Errorf("DoubleMetaphone(%q): ok=false, want true (%s)", c.word, c.why)
			continue
		}
		if p != c.wantP || s != c.wantS {
			t.Errorf("DoubleMetaphone(%q) = (%q,%q), want (%q,%q) — %s",
				c.word, p, s, c.wantP, c.wantS, c.why)
		}
	}
}
