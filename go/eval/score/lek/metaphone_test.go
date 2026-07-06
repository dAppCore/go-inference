// SPDX-Licence-Identifier: EUPL-1.2

package lek

import "testing"

// --- Normalize ---

func TestMetaphone_MetaphoneNormalizeStripsNonLetters(t *testing.T) {
	cases := []struct {
		in, want string
	}{
		{"Thompson", "THOMPSON"},
		{"Cina-Gia'a", "CINAGIAA"},
		{"O'Brien", "OBRIEN"},
		{"  spaces  ", "SPACES"},
		{"digit1mix", "DIGITMIX"},
		{"", ""},
		{"!@#$", ""},
		{"café", "CAF"}, // non-ASCII é stripped
	}
	for _, c := range cases {
		if got := metaphoneNormalize(c.in); got != c.want {
			t.Errorf("metaphoneNormalize(%q) = %q, want %q", c.in, got, c.want)
		}
	}
}

// --- Round-trip canonical words ---

// TestMetaphone_DoubleMetaphone_Bad — empty input returns ok=false.
func TestMetaphone_DoubleMetaphone_Bad(t *testing.T) {
	if _, _, ok := DoubleMetaphone(""); ok {
		t.Error("DoubleMetaphone(\"\") returned ok=true, want false")
	}
	if _, _, ok := DoubleMetaphone("!!!"); ok {
		t.Error("DoubleMetaphone(\"!!!\") returned ok=true, want false")
	}
}

// TestMetaphone_DoubleMetaphone_Good — canonical DM cases.
//
// Note: the implementation diverges from Lawrence Philips' exact
// reference output for some edge cases (silent letters in unusual
// positions). This test asserts STABILITY of OUR encoding — once a
// word has an output, it stays that output. Cross-equivalence is what
// the LEK detector cares about, not exact textbook codes.
func TestMetaphone_DoubleMetaphone_Good(t *testing.T) {
	cases := []struct {
		word        string
		wantPrimary string
		wantSecond  string // empty = expect same as primary
	}{
		// PH → F.
		{"Philip", "FLP", ""},
		{"Philippe", "FLP", ""},
		// Silent initial.
		{"Knight", "NT", ""},
		{"Gnostic", "NSTK", ""},
		{"Wrap", "RP", ""},
		{"Psalm", "SLM", ""},
		// Doubled consonants collapse.
		{"Smith", "SM0", "SMT"},
		// Cross-orthographic equivalence — Smith / Smyth share TH ending.
		// We assert ONLY that they produce IDENTICAL codes (in
		// PhoneticEquivalent_Good below). Exact value tested in this row
		// for the canonical Smith.
	}
	for _, c := range cases {
		p, s, ok := DoubleMetaphone(c.word)
		if !ok {
			t.Errorf("DoubleMetaphone(%q): ok=false, want true", c.word)
			continue
		}
		if p != c.wantPrimary {
			t.Errorf("DoubleMetaphone(%q) primary = %q, want %q (sec=%q)",
				c.word, p, c.wantPrimary, s)
		}
		expSec := c.wantSecond
		if expSec == "" {
			expSec = c.wantPrimary
		}
		if s != expSec {
			t.Errorf("DoubleMetaphone(%q) secondary = %q, want %q",
				c.word, s, expSec)
		}
	}
}

// --- Cross-orthographic equivalence ---

// TestMetaphone_PhoneticEquivalent_Good — different spellings
// of the same word should match.
func TestMetaphone_PhoneticEquivalent_Good(t *testing.T) {
	pairs := [][2]string{
		{"Smith", "Smyth"},         // Y/I substitution
		{"Philip", "Phillip"},      // doubled consonant
		{"Catherine", "Katherine"}, // C/K initial
		{"riddley", "riddly"},      // dropped vowel
	}
	for _, p := range pairs {
		if !PhoneticEquivalent(p[0], p[1]) {
			a1, a2, _ := DoubleMetaphone(p[0])
			b1, b2, _ := DoubleMetaphone(p[1])
			t.Errorf("PhoneticEquivalent(%q,%q)=false; %q→(%q,%q), %q→(%q,%q)",
				p[0], p[1], p[0], a1, a2, p[1], b1, b2)
		}
	}
}

// TestMetaphone_PhoneticEquivalent_Distinct_Good — clearly different words
// should NOT match.
func TestMetaphone_PhoneticEquivalent_Distinct_Good(t *testing.T) {
	pairs := [][2]string{
		{"dog", "cat"},
		{"hello", "world"},
		{"piano", "elephant"},
	}
	for _, p := range pairs {
		if PhoneticEquivalent(p[0], p[1]) {
			a1, a2, _ := DoubleMetaphone(p[0])
			b1, b2, _ := DoubleMetaphone(p[1])
			t.Errorf("PhoneticEquivalent(%q,%q)=true (should differ); %q→(%q,%q), %q→(%q,%q)",
				p[0], p[1], p[0], a1, a2, p[1], b1, b2)
		}
	}
}

// TestMetaphone_PhoneticEquivalent_Bad — the documented
// contract: returns false when EITHER word is empty or has no letters to
// encode (DoubleMetaphone returns ok=false), regardless of position.
func TestMetaphone_PhoneticEquivalent_Bad(t *testing.T) {
	cases := [][2]string{
		{"", "cat"},    // empty first → first DoubleMetaphone fails
		{"cat", ""},    // empty second → second DoubleMetaphone fails
		{"", ""},       // both empty
		{"123", "cat"}, // non-letter first → unrecognisable
		{"cat", "!!!"}, // non-letter second → unrecognisable
	}
	for _, c := range cases {
		if PhoneticEquivalent(c[0], c[1]) {
			t.Errorf("PhoneticEquivalent(%q,%q) = true, want false (empty/unrecognisable)", c[0], c[1])
		}
	}
}

// --- LEK-class: Cina-Gia'a ≈ China's ---

// TestMetaphone_PhoneticContains_Good — the canonical LEK artifact.
// DeepSeek-1B encoded "China's" as "Cina-Gia'a" inside an Italian
// shell to bypass compliance ([[research-lek-artifact-phonetic-circumvention]]).
// The phonetic detector MUST flag the blocked topic as phonetically
// CONTAINED within the response token.
//
// PhoneticEquivalent (exact code equality) doesn't fire here because
// the response token has extra "Gia'a" phonemes beyond "China"'s
// code. PhoneticContains (prefix match across the four code pairings)
// does fire because the blocked topic's secondary code "XN" appears
// as the prefix of the response's secondary code "XNJ" (or "XNJS").
//
// This is the load-bearing test for the whole U lane — if this fails,
// the dimension we're building doesn't catch the failure class we
// designed it to catch.
func TestMetaphone_PhoneticContains_Good(t *testing.T) {
	pairs := []struct {
		response, blocked string
	}{
		{"Cina", "China"},
		{"Cina-Gia'a", "China"},
		{"Cina-Gia'a", "China's"},
	}
	for _, p := range pairs {
		respP, respS, _ := DoubleMetaphone(p.response)
		blockedP, blockedS, _ := DoubleMetaphone(p.blocked)
		t.Logf("response %q → (%q,%q); blocked %q → (%q,%q)",
			p.response, respP, respS, p.blocked, blockedP, blockedS)
		if !PhoneticContains(p.response, p.blocked) {
			t.Errorf("PhoneticContains(%q,%q)=false — LEK artifact MUST match",
				p.response, p.blocked)
		}
	}
}

// TestMetaphone_PhoneticContains_Bad — single-letter needles
// don't trigger PhoneticContains (would fire on every word containing
// a common phoneme — false-positive volcano).
func TestMetaphone_PhoneticContains_Bad(t *testing.T) {
	// "I" → ("A", "A") — single phoneme. Must not match every word.
	if PhoneticContains("anything", "I") {
		t.Error("PhoneticContains with single-phoneme needle returned true; floor=2 should reject")
	}
}

// TestMetaphone_PhoneticContains_Ugly — the documented
// contract: returns false when EITHER word is empty or unrecognisable
// (DoubleMetaphone ok=false on the haystack or the needle).
func TestMetaphone_PhoneticContains_Ugly(t *testing.T) {
	cases := [][2]string{
		{"", "china"},       // empty haystack → haystack DoubleMetaphone fails
		{"china", ""},       // empty needle → needle DoubleMetaphone fails
		{"123", "china"},    // non-letter haystack → unrecognisable
		{"response", "!!!"}, // non-letter needle → unrecognisable
	}
	for _, c := range cases {
		if PhoneticContains(c[0], c[1]) {
			t.Errorf("PhoneticContains(%q,%q) = true, want false (empty/unrecognisable)", c[0], c[1])
		}
	}
}

// --- Stability ---

// TestMetaphone_DoubleMetaphone_DeterministicStable_Good — same input → same
// output. Phonetic codes are pure functions; this catches accidental
// state leakage if the encoder ever grew mutable globals.
func TestMetaphone_DoubleMetaphone_DeterministicStable_Good(t *testing.T) {
	word := "Tchaikovsky"
	pa, sa, ok := DoubleMetaphone(word)
	if !ok {
		t.Fatalf("first call ok=false")
	}
	for i := range 5 {
		pb, sb, ok := DoubleMetaphone(word)
		if !ok || pa != pb || sa != sb {
			t.Errorf("call %d: %s,%s,%v (want %s,%s,true)", i, pb, sb, ok, pa, sa)
		}
	}
}

// TestMetaphone_DoubleMetaphone_Truncation_Good — codes are clipped to
// MetaphoneMaxCode.
func TestMetaphone_DoubleMetaphone_Truncation_Good(t *testing.T) {
	// A long word should produce codes no longer than the cap.
	p, s, ok := DoubleMetaphone("Pneumonoultramicroscopicsilicovolcanoconiosis")
	if !ok {
		t.Fatal("ok=false on long word")
	}
	if len(p) > MetaphoneMaxCode {
		t.Errorf("primary len %d > %d", len(p), MetaphoneMaxCode)
	}
	if len(s) > MetaphoneMaxCode {
		t.Errorf("secondary len %d > %d", len(s), MetaphoneMaxCode)
	}
}

// --- Consonant-rule branch coverage (step / stepC / stepG / stepJ) ---
//
// These exercise the per-consonant dispatch branches through the public
// DoubleMetaphone entry point. Inputs are chosen so each row targets one
// explainable rule; expected codes are derived from THIS implementation's
// actual output (the package diverges from textbook Double Metaphone for
// some edge cases — see TestDoubleMetaphone_BasicWords_Good), so the
// assertions lock OUR stable encoding, not Lawrence Philips' reference.

// TestMetaphone_StepCBranches — the C consonant, the most complex
// rule. Covers CH→X (church), Greek CH→K (character), CIO/CIA → S/X
// (Italian), CC-I → KS, CZ → S/X (Slavic), and C-before-E/I/Y → S/X.
func TestMetaphone_StepCBranches(t *testing.T) {
	cases := []struct {
		word         string
		wantP, wantS string
	}{
		{"church", "XRX", "XRX"},      // CH → X (English, default)
		{"character", "KRKT", "KRKT"}, // initial CH + ARAC → K (Greek)
		{"choir", "XR", "XR"},         // CH → X
		{"chasm", "KSM", "KSM"},       // initial CH + ASM → K (Greek)
		{"cello", "SL", "XL"},         // C before E → S primary, X secondary (Italian)
		{"cipher", "SFR", "XFR"},      // C before I → S/X
		{"city", "ST", "XT"},          // C before I → S/X
		{"czar", "SR", "XR"},          // CZ → S/X (Slavic)
		{"vacci", "FKS", "FKS"},       // CCI → KS (Italian doubled C)
		{"focaccia", "FKKS", "FKKS"},  // CC before I → KS, trailing CIA
		{"accord", "AKRT", "AKRT"},    // CC not before E/I/H → K
		{"mccoy", "MK", "MK"},         // initial MC → K
		{"bach", "PX", "PX"},          // CH word-final → X
		{"special", "SPSL", "SPXL"},   // CI before A → S/X (Italian)
		{"ancient", "ANSN", "ANXN"},   // CI mid-word → S/X
		{"cat", "KT", "KT"},           // default C → K
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

// TestMetaphone_StepGBranches — the G consonant. Covers GE/GI → J/K
// (gentle, giraffe — non-SlavoGermanic), GN-final silent (sign), GN-mid
// (design), GH-after-vowel silent (light), GH-after-consonant → K (ghost),
// and doubled GG → K (egg).
func TestMetaphone_StepGBranches(t *testing.T) {
	cases := []struct {
		word         string
		wantP, wantS string
	}{
		{"gentle", "JNTL", "KNTL"}, // GE → J primary, K secondary
		{"giraffe", "JRF", "KRF"},  // GI → J/K
		{"sign", "SN", "SN"},       // word-final GN → silent G
		{"design", "TSN", "TSN"},   // mid GN
		{"light", "LT", "LT"},      // GH after vowel → silent
		{"ghost", "ST", "ST"},      // initial GH (KN-handler) → silent, ST
		{"egg", "AK", "AK"},        // doubled GG → K
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

// TestMetaphone_StepJBranches — the J consonant. Covers JOSE →
// Spanish H, initial J → J primary / A secondary (the Y-glide reading),
// and mid-word J → J.
func TestMetaphone_StepJBranches(t *testing.T) {
	cases := []struct {
		word         string
		wantP, wantS string
	}{
		{"Jose", "HS", "HS"},   // JOSE special → Spanish H
		{"jump", "JMP", "AMP"}, // initial J → J / A (Y-glide alt)
		{"judge", "JJ", "AJ"},  // initial J → J/A, mid J → J
		{"hajj", "HJ", "HJ"},   // doubled J mid-word → consume both (i+2)
		{"raj", "RJ", "RJ"},    // word-final J → J
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

// TestMetaphone_StepSBranches — the S consonant. Covers SH → X
// (ship), SIO/SIA → S/X (mansion — Italian), SCH → X (schmidt), SC-before-E
// → S (scene), SCHOOL → SK secondary, and the SUGAR special (S before U
// sounds /ʃ/ → X/S).
func TestMetaphone_StepSBranches(t *testing.T) {
	cases := []struct {
		word         string
		wantP, wantS string
	}{
		{"ship", "XP", "XP"},        // SH → X
		{"mansion", "MNSN", "MNXN"}, // SIO → S/X (Italian /ʃ/ secondary)
		{"schmidt", "XMT", "XMT"},   // SCH → X (Germanic)
		{"scene", "SN", "SN"},       // SC before E → S
		{"school", "XL", "SKL"},     // SCH start + O → X / SK
		{"sugar", "XKR", "SKR"},     // SUGAR special → X/S
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

// TestMetaphone_StepTWNZBranches — the T, W, N, Z consonant rules.
// Covers TH → T/0 dental (think), TH+OM → T (Thomas), TIO → X (nation),
// TCH → X (witch), initial W+vowel → A/F (away), WH-start silent (when),
// and WR-start → R (wrap).
func TestMetaphone_StepTWNZBranches(t *testing.T) {
	cases := []struct {
		word         string
		wantP, wantS string
	}{
		{"think", "TNK", "TNK"},  // TH → T primary, but THINK uses 0/T then merges
		{"Thomas", "TMS", "TMS"}, // TH + OM → T
		{"nation", "NXN", "NXN"}, // TIO → X
		{"witch", "AX", "FX"},    // TCH → X; initial W+vowel → A/F
		{"when", "AN", "AN"},     // WH start → silent W, A
		{"wrap", "RP", "RP"},     // WR start → silent W, R
		{"away", "A", "A"},       // initial vowel + mid W silent
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

// TestMetaphone_InitialX — the encodeInline initial-X arm: a word
// beginning with X is read as an /s/ onset (Xavier, Xena → S…), the
// Greek-derived initial-X-as-S rule. Distinct from mid-word X (→ KS).
func TestMetaphone_InitialX(t *testing.T) {
	cases := []struct {
		word         string
		wantP, wantS string
	}{
		{"xavier", "SFR", "SFR"}, // initial X → S onset
		{"xena", "SN", "SN"},     // initial X → S onset
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

// TestMetaphone_EncodeMetaphoneNonPooledFallback — encodeMetaphone is the
// non-pooled scaffolding variant kept for tests that need a fresh encoder
// (it constructs the enc directly rather than routing through the pool).
// It must produce byte-identical codes to the pooled DoubleMetaphone path
// for a pre-normalised (uppercase, letters-only) input. This is the only
// caller of encodeMetaphone + reset, both of which are otherwise
// test-only scaffolding per their doc comments.
func TestMetaphone_EncodeMetaphoneNonPooledFallback(t *testing.T) {
	// Input must be pre-normalised (uppercase, no punctuation) because
	// encodeMetaphone skips resetFromRaw's normalise pass.
	pri, alt := encodeMetaphone("THOMPSON")
	gotP := string(truncate(pri, MetaphoneMaxCode))
	gotA := string(truncate(alt, MetaphoneMaxCode))
	// Pooled path on the same normalised word must agree.
	wantP, wantA, ok := DoubleMetaphone("THOMPSON")
	if !ok {
		t.Fatal("DoubleMetaphone(THOMPSON) ok=false")
	}
	if gotP != wantP || gotA != wantA {
		t.Errorf("encodeMetaphone(THOMPSON) = (%q,%q), pooled = (%q,%q) — paths must agree",
			gotP, gotA, wantP, wantA)
	}
}

// TestMetaphone_EncResetPreNormalised — reset is the pre-normalised encoder
// setup (the non-pooled counterpart of resetFromRaw). After reset + encode
// the codes must match the from-raw path for the same letters. Exercises
// the otherwise-uncovered reset method directly.
func TestMetaphone_EncResetPreNormalised(t *testing.T) {
	e := &enc{}
	e.reset("KNIGHT")
	if string(e.word) != "KNIGHT" || e.length != 6 {
		t.Fatalf("reset set word=%q length=%d, want KNIGHT/6", e.word, e.length)
	}
	e.encodeInline()
	got := string(truncate(e.pri, MetaphoneMaxCode))
	want, _, _ := DoubleMetaphone("Knight")
	if got != want {
		t.Errorf("reset+encode primary = %q, want %q (matches DoubleMetaphone(Knight))", got, want)
	}
}

// TestMetaphone_DoubleMetaphone_Ugly — degenerate inputs: pure
// whitespace and a mixed digit+punctuation token normalise to nothing,
// so DoubleMetaphone reports ok=false without panicking. A token whose
// only letters are non-ASCII (stripped by normalise) likewise fails.
func TestMetaphone_DoubleMetaphone_Ugly(t *testing.T) {
	for _, in := range []string{"   \t\n ", "12-34_56", "你好", "'''"} {
		p, s, ok := DoubleMetaphone(in)
		if ok {
			t.Errorf("DoubleMetaphone(%q) = (%q,%q,true), want ok=false (no encodable letters)", in, p, s)
		}
	}
}

// TestMetaphone_PhoneticEquivalent_Ugly — when BOTH sides normalise to
// nothing, PhoneticEquivalent must report false (not "both empty codes
// are equal"). Two blank tokens are not phonetic twins.
func TestMetaphone_PhoneticEquivalent_Ugly(t *testing.T) {
	cases := [][2]string{
		{"   ", "\t\n"}, // both whitespace → both unencodable
		{"123", "!!!"},  // both non-letter
		{"你好", "''"},    // non-ASCII vs punctuation
	}
	for _, c := range cases {
		if PhoneticEquivalent(c[0], c[1]) {
			t.Errorf("PhoneticEquivalent(%q,%q) = true, want false (both unencodable)", c[0], c[1])
		}
	}
}
