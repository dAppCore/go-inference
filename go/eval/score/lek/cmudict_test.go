// SPDX-Licence-Identifier: EUPL-1.2

package lek

import (
	"testing"
)

// --- Lookup ---

func TestCmudict_Lookup_Good(t *testing.T) {
	ph, ok := Lookup("cat")
	if !ok {
		t.Fatal("Lookup(cat) returned ok=false; starter dict should include 'cat'")
	}
	if len(ph) != 3 || ph[0] != "K" || ph[1] != "AE1" || ph[2] != "T" {
		t.Errorf("Lookup(cat) = %v, want [K AE1 T]", ph)
	}
}

func TestCmudict_Lookup_Bad(t *testing.T) {
	// An unknown word resolves to (nil, false). Lookup must not invent a
	// pronunciation for tokens outside the embedded dict.
	ph, ok := Lookup("nonexistentwordxyz")
	if ok {
		t.Error("Lookup on unknown word returned ok=true")
	}
	if ph != nil {
		t.Errorf("Lookup(unknown) phonemes = %v, want nil", ph)
	}
}

func TestCmudict_Lookup_Ugly(t *testing.T) {
	// Empty and whitespace-only input must resolve to (nil, false)
	// without panicking — Lookup trims before keying the map, so a
	// blank key never matches a real entry.
	if ph, ok := Lookup(""); ok || ph != nil {
		t.Errorf("Lookup(\"\") = (%v, %v), want (nil, false)", ph, ok)
	}
	if ph, ok := Lookup("   \t "); ok || ph != nil {
		t.Errorf("Lookup(whitespace) = (%v, %v), want (nil, false)", ph, ok)
	}
}

// TestCmudict_Lookup_CaseInsensitive_Good — the dict is keyed uppercase
// and Lookup upper-cases the query, so CAT / cat / Cat resolve to the
// same phoneme sequence.
func TestCmudict_Lookup_CaseInsensitive_Good(t *testing.T) {
	a, _ := Lookup("CAT")
	b, _ := Lookup("cat")
	c, _ := Lookup("Cat")
	if len(a) == 0 || !slicesEqual(a, b) || !slicesEqual(a, c) {
		t.Errorf("Lookup case mismatch: CAT=%v cat=%v Cat=%v", a, b, c)
	}
}

// --- IsDictWord ---

// TestCmudict_IsDictWord_Good — a word present in the CMU starter dict
// reports true regardless of case (Lookup trims + uppercases first).
func TestCmudict_IsDictWord_Good(t *testing.T) {
	for _, w := range []string{"the", "cat", "CAT", "Cat"} {
		if !IsDictWord(w) {
			t.Errorf("IsDictWord(%q) = false, want true (in starter dict)", w)
		}
	}
}

// TestCmudict_IsDictWord_Bad — an invented pseudo-jargon token is not in
// the dict (this is exactly the signal PseudoJargonDensity relies on to
// flag LEK-class compounds like "Cina-Gia'a").
func TestCmudict_IsDictWord_Bad(t *testing.T) {
	for _, w := range []string{"zzxqwf", "Gia", "qwertyuiop"} {
		if IsDictWord(w) {
			t.Errorf("IsDictWord(%q) = true, want false (invented token)", w)
		}
	}
}

// TestCmudict_IsDictWord_Ugly — empty / whitespace input is never a dict
// word; must not panic.
func TestCmudict_IsDictWord_Ugly(t *testing.T) {
	if IsDictWord("") {
		t.Error("IsDictWord(\"\") = true, want false")
	}
	if IsDictWord("   ") {
		t.Error("IsDictWord(\"   \") = true, want false")
	}
}

// --- IsVowelPhoneme ---

func TestCmudict_IsVowelPhoneme_Good(t *testing.T) {
	// ARPAbet vowels carry a trailing stress digit (0/1/2).
	for _, ph := range []string{"AE1", "AH0", "IY1", "OW2"} {
		if !IsVowelPhoneme(ph) {
			t.Errorf("IsVowelPhoneme(%q) = false, want true (vowel w/ stress digit)", ph)
		}
	}
}

func TestCmudict_IsVowelPhoneme_Bad(t *testing.T) {
	// Consonant phonemes have no trailing stress digit.
	for _, ph := range []string{"K", "T", "DH", "S", "NG"} {
		if IsVowelPhoneme(ph) {
			t.Errorf("IsVowelPhoneme(%q) = true, want false (consonant)", ph)
		}
	}
}

func TestCmudict_IsVowelPhoneme_Ugly(t *testing.T) {
	// Empty string must report false and not panic on the len-1 index.
	if IsVowelPhoneme("") {
		t.Error("IsVowelPhoneme(\"\") = true, want false")
	}
}

// --- PhonemeStress ---

func TestCmudict_PhonemeStress_Good(t *testing.T) {
	cases := map[string]int{"AE1": 1, "AH0": 0, "OW2": 2}
	for ph, want := range cases {
		if got := PhonemeStress(ph); got != want {
			t.Errorf("PhonemeStress(%q) = %d, want %d", ph, got, want)
		}
	}
}

func TestCmudict_PhonemeStress_Bad(t *testing.T) {
	// Consonants carry no stress — PhonemeStress returns -1.
	for _, ph := range []string{"K", "T", "DH"} {
		if got := PhonemeStress(ph); got != -1 {
			t.Errorf("PhonemeStress(%q) = %d, want -1 (consonant)", ph, got)
		}
	}
}

func TestCmudict_PhonemeStress_Ugly(t *testing.T) {
	// Empty input is not a vowel → -1, no panic.
	if got := PhonemeStress(""); got != -1 {
		t.Errorf("PhonemeStress(\"\") = %d, want -1", got)
	}
}

// --- helpers ---

// slicesEqual reports element-wise equality of two string slices. Used
// by the case-insensitivity checks above and by the syllable helpers in
// phonetic_dims_test.go's neighbourhood.
func slicesEqual(a, b []string) bool {
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
