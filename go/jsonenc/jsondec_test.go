// SPDX-Licence-Identifier: EUPL-1.2

package jsonenc

import (
	"reflect"
	"testing"
)

// TestParseJSONStringList_RoundTrip mirrors the test in openai/jsondec_test.go —
// when this passes, the openai package's call site is byte-for-byte
// compatible with the lifted primitive.
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
			got, err := ParseJSONStringList([]byte(tc.in))
			if err != nil {
				t.Fatalf("ParseJSONStringList(%s) error = %v", tc.in, err)
			}
			if !reflect.DeepEqual(got, tc.want) {
				t.Fatalf("ParseJSONStringList(%s) = %v, want %v", tc.in, got, tc.want)
			}
		})
	}
}

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
			_, err := ParseJSONStringList([]byte(in))
			if err == nil {
				t.Fatalf("ParseJSONStringList(%q) returned nil error, want error", in)
			}
		})
	}
}

func TestParseJSONString_FastPath(t *testing.T) {
	data := []byte(`"hello world"`)
	s, next, err := ParseJSONString(data, 0)
	if err != nil {
		t.Fatalf("ParseJSONString error = %v", err)
	}
	if s != "hello world" {
		t.Fatalf("got %q want hello world", s)
	}
	if next != len(data) {
		t.Fatalf("next = %d want %d", next, len(data))
	}
}

func TestParseJSONString_Escapes(t *testing.T) {
	cases := []struct {
		in   string
		want string
	}{
		{`"\""`, `"`},
		{`"\\"`, `\`},
		{`"\/"`, "/"},
		{`"\b"`, "\b"},
		{`"\f"`, "\f"},
		{`"\n"`, "\n"},
		{`"\r"`, "\r"},
		{`"\t"`, "\t"},
		{`"A"`, "A"},
		{`"é"`, "é"},
	}
	for _, tc := range cases {
		t.Run(tc.in, func(t *testing.T) {
			s, _, err := ParseJSONString([]byte(tc.in), 0)
			if err != nil {
				t.Fatalf("ParseJSONString(%s) error = %v", tc.in, err)
			}
			if s != tc.want {
				t.Fatalf("got %q want %q", s, tc.want)
			}
		})
	}
}

// TestParseJSONString_UnicodeEscape pins the \uXXXX decode path
// (1/2/3-byte UTF-8 output) — previously uncovered, and the load-
// bearing guard for the strings.Builder rewrite of the escape path.
func TestParseJSONString_UnicodeEscape(t *testing.T) {
	// bs is a single backslash, built from its code point so this
	// source file contains no backslash-u literal (which the test
	// harness would itself interpret). u("00e9") -> `é`.
	bs := string(rune(92))
	u := func(hex string) string { return bs + "u" + hex }
	cases := []struct {
		name string
		in   string
		want string
	}{
		{"1byte", `"` + u("0041") + `"`, "A"},
		{"2byte", `"` + u("00e9") + `"`, "é"},
		{"3byte", `"` + u("20ac") + `"`, "€"},
		{"interleaved", `"a` + u("00e9") + `b` + u("20ac") + `c"`, "aéb€c"},
		{"back2back", `"` + u("0041") + u("0042") + `"`, "AB"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			s, _, err := ParseJSONString([]byte(tc.in), 0)
			if err != nil {
				t.Fatalf("ParseJSONString(%q) error = %v", tc.in, err)
			}
			if s != tc.want {
				t.Fatalf("got %q want %q", s, tc.want)
			}
		})
	}
}

func TestParseJSONInt(t *testing.T) {
	cases := []struct {
		in   string
		want int64
	}{
		{`0`, 0},
		{`1`, 1},
		{`-1`, -1},
		{`123456789`, 123456789},
		{`-987`, -987},
	}
	for _, tc := range cases {
		t.Run(tc.in, func(t *testing.T) {
			n, _, err := ParseJSONInt([]byte(tc.in), 0)
			if err != nil {
				t.Fatalf("ParseJSONInt(%s) error = %v", tc.in, err)
			}
			if n != tc.want {
				t.Fatalf("got %d want %d", n, tc.want)
			}
		})
	}
}

func TestParseJSONInt_Invalid(t *testing.T) {
	cases := []string{"", "-", "a", "+1"}
	for _, in := range cases {
		t.Run(in, func(t *testing.T) {
			_, _, err := ParseJSONInt([]byte(in), 0)
			if err == nil {
				t.Fatalf("ParseJSONInt(%q) returned nil error, want error", in)
			}
		})
	}
}

func TestParseJSONBool(t *testing.T) {
	v, next, err := ParseJSONBool([]byte(`true`), 0)
	if err != nil || v != true || next != 4 {
		t.Fatalf("true: v=%v next=%d err=%v", v, next, err)
	}
	v, next, err = ParseJSONBool([]byte(`false`), 0)
	if err != nil || v != false || next != 5 {
		t.Fatalf("false: v=%v next=%d err=%v", v, next, err)
	}
	_, _, err = ParseJSONBool([]byte(`tru`), 0)
	if err == nil {
		t.Fatalf("ParseJSONBool(tru) returned nil error")
	}
}

func TestIsJSONNull(t *testing.T) {
	if !IsJSONNull([]byte(`null`), 0) {
		t.Fatalf("expected null match")
	}
	if IsJSONNull([]byte(`nul`), 0) {
		t.Fatalf("expected no match on nul")
	}
	if IsJSONNull([]byte(`xnull`), 0) {
		t.Fatalf("expected no match on xnull")
	}
}

func TestSkipJSONValue(t *testing.T) {
	cases := []struct {
		in   string
		want int
	}{
		{`null`, 4},
		{`true`, 4},
		{`false`, 5},
		{`"abc"`, 5},
		{`123`, 3},
		{`-1.5e3`, 6},
		{`{}`, 2},
		{`[]`, 2},
		{`{"a":1}`, 7},
		{`["a","b"]`, 9},
		{`{"a":[1,2,{"b":"c"}]}`, 21},
	}
	for _, tc := range cases {
		t.Run(tc.in, func(t *testing.T) {
			next, err := SkipJSONValue([]byte(tc.in), 0)
			if err != nil {
				t.Fatalf("SkipJSONValue(%s) error = %v", tc.in, err)
			}
			if next != tc.want {
				t.Fatalf("got %d want %d", next, tc.want)
			}
		})
	}
}

func TestMatchObjectAndArrayStart(t *testing.T) {
	i, err := MatchObjectStart([]byte(`  {`), 0)
	if err != nil || i != 3 {
		t.Fatalf("MatchObjectStart: i=%d err=%v", i, err)
	}
	i, err = MatchArrayStart([]byte(`  [`), 0)
	if err != nil || i != 3 {
		t.Fatalf("MatchArrayStart: i=%d err=%v", i, err)
	}
	_, err = MatchObjectStart([]byte(`123`), 0)
	if err == nil {
		t.Fatalf("expected error on non-object")
	}
}

func TestSkipJSONString(t *testing.T) {
	cases := []struct {
		in   string
		want int
	}{
		{`"abc"`, 5},
		{`""`, 2},
		{`"a\nb"`, 6},
		{`"a\"b"`, 6},
		{`"a\\b"`, 6},
		{`"aÿb"`, 6}, // ÿ is 2 UTF-8 bytes inside the quotes
	}
	for _, tc := range cases {
		t.Run(tc.in, func(t *testing.T) {
			next, err := SkipJSONString([]byte(tc.in), 0)
			if err != nil {
				t.Fatalf("SkipJSONString(%s) error = %v", tc.in, err)
			}
			if next != tc.want {
				t.Fatalf("got %d want %d", next, tc.want)
			}
		})
	}
}

func TestParseJSONFloat(t *testing.T) {
	v, _, err := ParseJSONFloat32([]byte(`0.7`), 0)
	if err != nil || v != 0.7 {
		t.Fatalf("ParseJSONFloat32(0.7): v=%v err=%v", v, err)
	}
	v, _, err = ParseJSONFloat32([]byte(`-1.5e2`), 0)
	if err != nil || v != -150 {
		t.Fatalf("ParseJSONFloat32(-1.5e2): v=%v err=%v", v, err)
	}
	d, _, err := ParseJSONFloat64([]byte(`3.14`), 0)
	if err != nil || d != 3.14 {
		t.Fatalf("ParseJSONFloat64(3.14): d=%v err=%v", d, err)
	}
}

func TestCountJSONArrayElements(t *testing.T) {
	cases := []struct {
		in   string
		want int
	}{
		{`]`, 0},
		{`1]`, 1},
		{`1,2,3]`, 3},
		{`"a","b"]`, 2},
		{`{"x":1},{"y":2}]`, 2},
		{`[1,2],[3]]`, 2},
	}
	for _, tc := range cases {
		t.Run(tc.in, func(t *testing.T) {
			got := CountJSONArrayElements([]byte(tc.in), 0)
			if got != tc.want {
				t.Fatalf("got %d want %d", got, tc.want)
			}
		})
	}
}

func TestParseJSONStringRaw(t *testing.T) {
	b, next, err := ParseJSONStringRaw([]byte(`"hello"`), 0)
	if err != nil || string(b) != "hello" || next != 7 {
		t.Fatalf("ParseJSONStringRaw fast path: b=%q next=%d err=%v", b, next, err)
	}
	b, next, err = ParseJSONStringRaw([]byte(`"a\nb"`), 0)
	if err != nil || string(b) != "a\nb" || next != 6 {
		t.Fatalf("ParseJSONStringRaw escape path: b=%q next=%d err=%v", b, next, err)
	}
}
