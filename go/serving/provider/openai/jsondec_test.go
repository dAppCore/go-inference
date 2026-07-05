// SPDX-Licence-Identifier: EUPL-1.2

package openai

import (
	"reflect"
	"testing"
)

// TestParseJSONStringList_RoundTrip locks the hand-rolled
// string-or-array walker against the documented input/output
// contract. Cases cover every branch: null literal, plain string,
// empty array, single-element array, multi-element array, and
// every escape form the JSON spec recognises.
func TestParseJSONStringList_RoundTrip(t *testing.T) {
	cases := []struct {
		name string
		in   string
		want []string
	}{
		{"null", "null", nil},
		{"null-with-whitespace", "  null\t", nil},
		{"plain-string", `"END"`, []string{"END"}},
		{"string-with-escapes", `"line1\nline2"`, []string{"line1\nline2"}},
		{"string-with-quote", `"he said \"hi\""`, []string{`he said "hi"`}},
		{"string-with-unicode", `"é"`, []string{"é"}},
		{"empty-array", `[]`, nil},
		{"single-element-array", `["END"]`, []string{"END"}},
		{"multi-element-array", `["A","B","C"]`, []string{"A", "B", "C"}},
		{"array-with-whitespace", ` [ "A" , "B" ] `, []string{"A", "B"}},
		{"array-with-escapes", `["\t","\n"]`, []string{"\t", "\n"}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := parseJSONStringList([]byte(tc.in))
			if err != nil {
				t.Fatalf("parseJSONStringList(%s) error = %v", tc.in, err)
			}
			if !reflect.DeepEqual(got, tc.want) {
				t.Fatalf("parseJSONStringList(%s) = %v, want %v", tc.in, got, tc.want)
			}
		})
	}
}

// TestParseJSONStringList_Invalid asserts the walker rejects
// malformed inputs cleanly — no panics, just errors.
func TestParseJSONStringList_Invalid(t *testing.T) {
	cases := []string{
		"",
		"   ",
		`{`,
		`}`,
		`"unterminated`,
		`[`,
		`["unterminated`,
		`["A"`,
		`["A",]`,
		`[123]`,
		`tru`,
	}
	for _, in := range cases {
		t.Run(in, func(t *testing.T) {
			_, err := parseJSONStringList([]byte(in))
			if err == nil {
				t.Fatalf("parseJSONStringList(%q) returned nil error, want error", in)
			}
		})
	}
}
