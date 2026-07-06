// SPDX-Licence-Identifier: EUPL-1.2

package jsonenc

import (
	"reflect"
	"strings"
	"testing"
)

// bs is a single backslash, built from its code point so no source
// file in this package contains a literal `\u` sequence — Go would
// interpret that as its own unicode escape inside a double-quoted
// string, and even inside a raw string it would make the JSON escape
// harder to eyeball against the mnemonic escapes below. u builds a
// `\uXXXX` JSON escape sequence from a 4-hex-digit codepoint.
var bs = string(rune(92))

func u(hex string) string { return bs + "u" + hex }

// TestJsondec_ParseJSONStringList_Good covers the two valid shapes —
// a bare string and an array of strings — plus the null literal and
// surrounding whitespace.
func TestJsondec_ParseJSONStringList_Good(t *testing.T) {
	cases := []struct {
		name string
		in   string
		want []string
	}{
		{"null", "null", nil},
		{"null-with-whitespace", "  null\t", nil},
		{"plain-string", `"END"`, []string{"END"}},
		{"string-with-escapes", `"line1` + u("000a") + `line2"`, []string{"line1\nline2"}},
		{"empty-array", `[]`, nil},
		{"single-element-array", `["END"]`, []string{"END"}},
		{"multi-element-array", `["A","B","C"]`, []string{"A", "B", "C"}},
		{"array-with-whitespace", ` [ "A" , "B" ] `, []string{"A", "B"}},
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

// TestJsondec_ParseJSONStringList_Bad covers malformed input across
// every rejection branch: empty/whitespace-only, wrong bracket kind,
// unterminated string/array, a non-string array element, and an
// "n"-prefixed literal that isn't actually "null".
func TestJsondec_ParseJSONStringList_Bad(t *testing.T) {
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
		`nope`,
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

// TestJsondec_ParseJSONStringList_Ugly covers the array-element
// separator boundary — two string elements with no comma between
// them — which walks a different rejection branch inside
// parseJSONStringArray than a plain unterminated array does.
func TestJsondec_ParseJSONStringList_Ugly(t *testing.T) {
	_, err := ParseJSONStringList([]byte(`["A" "B"]`))
	if err == nil {
		t.Fatalf(`ParseJSONStringList(["A" "B"]) returned nil error, want error`)
	}
}

// TestJsondec_ParseJSONString_Good covers the no-escape fast path,
// which returns a direct slice-to-string conversion.
func TestJsondec_ParseJSONString_Good(t *testing.T) {
	data := []byte(`"hello world"`)
	s, next, err := ParseJSONString(data, 0)
	if err != nil {
		t.Fatalf("error = %v", err)
	}
	if s != "hello world" || next != len(data) {
		t.Fatalf("got (%q, %d) want (%q, %d)", s, next, "hello world", len(data))
	}
}

// TestJsondec_ParseJSONString_Bad covers entry-point rejection (index
// out of range, or the byte at i isn't a quote), an unterminated
// string, and a raw control byte inside the body — all distinct
// rejection branches.
func TestJsondec_ParseJSONString_Bad(t *testing.T) {
	if _, _, err := ParseJSONString([]byte("abc"), 0); err == nil {
		t.Fatalf("no-quote: expected error")
	}
	if _, _, err := ParseJSONString([]byte(`"abc`), 0); err == nil {
		t.Fatalf("unterminated: expected error")
	}
	if _, _, err := ParseJSONString([]byte("\"a\x01b\""), 0); err == nil {
		t.Fatalf("control byte: expected error")
	}
	if _, _, err := ParseJSONString([]byte{}, 5); err == nil {
		t.Fatalf("index out of range: expected error")
	}
}

// TestJsondec_ParseJSONString_Ugly covers the escape path: every
// mnemonic escape, an empty string body, and the \uXXXX decode across
// 1/2/3-byte UTF-8 output plus back-to-back escapes.
func TestJsondec_ParseJSONString_Ugly(t *testing.T) {
	cases := []struct {
		name string
		in   string
		want string
	}{
		{"quote", `"` + bs + `""`, `"`},
		{"backslash", `"` + bs + bs + `"`, bs},
		{"solidus", `"` + bs + `/"`, "/"},
		{"backspace", `"` + bs + `b"`, "\b"},
		{"formfeed", `"` + bs + `f"`, "\f"},
		{"newline", `"` + bs + `n"`, "\n"},
		{"carriage-return", `"` + bs + `r"`, "\r"},
		{"tab", `"` + bs + `t"`, "\t"},
		{"empty", `""`, ""},
		{"1byte-unicode", `"` + u("0041") + `"`, "A"},
		{"2byte-unicode", `"` + u("00e9") + `"`, "é"},
		{"3byte-unicode", `"` + u("20ac") + `"`, "€"},
		{"2byte-unicode-uppercase-hex", `"` + u("00E9") + `"`, "é"},
		{"back2back-unicode", `"` + u("0041") + u("0042") + `"`, "AB"},
		{"interleaved-unicode", `"a` + u("00e9") + `b` + u("20ac") + `c"`, "aéb€c"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			s, _, err := ParseJSONString([]byte(tc.in), 0)
			if err != nil {
				t.Fatalf("ParseJSONString(%s) error = %v", tc.in, err)
			}
			if s != tc.want {
				t.Fatalf("got %q want %q", s, tc.want)
			}
		})
	}
	// Malformed escape shapes the fast/slow split must still reject:
	// an unrecognised escape letter, a truncated \u (fewer than 4 hex
	// digits remaining), an invalid hex digit, a trailing backslash
	// with nothing following it, a raw control byte encountered AFTER
	// the escape decoder has already taken over (parseJSONStringEscaped's
	// own control-byte check, distinct from ParseJSONString's
	// pre-escape scan), and a clean run of bytes after a decoded
	// escape that reaches true EOF with no closing quote at all.
	badCases := []string{
		`"` + bs + `x"`,
		`"` + bs + `u12`,
		`"` + u("12zz") + `"`,
		`"a` + bs,
		`"a` + bs + `nb` + "\x01" + `c"`,
		`"a` + bs + `nbc`,
	}
	for _, in := range badCases {
		t.Run(in, func(t *testing.T) {
			if _, _, err := ParseJSONString([]byte(in), 0); err == nil {
				t.Fatalf("ParseJSONString(%s) returned nil error, want error", in)
			}
		})
	}
}

// TestJsondec_ParseJSONStringRaw_Good covers the no-copy fast path —
// the returned slice must alias data directly (no allocation).
func TestJsondec_ParseJSONStringRaw_Good(t *testing.T) {
	data := []byte(`"hello"`)
	b, next, err := ParseJSONStringRaw(data, 0)
	if err != nil || string(b) != "hello" || next != 7 {
		t.Fatalf("fast path: b=%q next=%d err=%v", b, next, err)
	}
}

// TestJsondec_ParseJSONStringRaw_Bad covers entry-point rejection
// (mirrors ParseJSONString) and an escape decode that fails —
// exercised only via the raw variant's own error-propagation branch.
func TestJsondec_ParseJSONStringRaw_Bad(t *testing.T) {
	if _, _, err := ParseJSONStringRaw([]byte("abc"), 0); err == nil {
		t.Fatalf("no-quote: expected error")
	}
	if _, _, err := ParseJSONStringRaw([]byte(`"a`+bs+`xb"`), 0); err == nil {
		t.Fatalf("invalid escape: expected error")
	}
	if _, _, err := ParseJSONStringRaw([]byte("\"a\x01b\""), 0); err == nil {
		t.Fatalf("raw control byte: expected error")
	}
	if _, _, err := ParseJSONStringRaw([]byte(`"abc`), 0); err == nil {
		t.Fatalf("unterminated: expected error")
	}
}

// TestJsondec_ParseJSONStringRaw_Ugly covers the escape path, which
// forces the one allocation the no-copy variant otherwise avoids —
// the returned bytes must still match what ParseJSONString produces.
func TestJsondec_ParseJSONStringRaw_Ugly(t *testing.T) {
	b, next, err := ParseJSONStringRaw([]byte(`"a`+bs+`nb"`), 0)
	if err != nil || string(b) != "a\nb" || next != 6 {
		t.Fatalf("escape path: b=%q next=%d err=%v", b, next, err)
	}
}

// TestJsondec_SkipJSONWhitespace_Good covers a mix of all four
// whitespace bytes JSON recognises.
func TestJsondec_SkipJSONWhitespace_Good(t *testing.T) {
	i := SkipJSONWhitespace([]byte(" \t\r\nx"), 0)
	if i != 4 {
		t.Fatalf("got %d want 4", i)
	}
}

// TestJsondec_SkipJSONWhitespace_Bad covers the no-op case — no
// leading whitespace at all — returning i unchanged.
func TestJsondec_SkipJSONWhitespace_Bad(t *testing.T) {
	i := SkipJSONWhitespace([]byte("x"), 0)
	if i != 0 {
		t.Fatalf("got %d want 0", i)
	}
}

// TestJsondec_SkipJSONWhitespace_Ugly covers all-whitespace input
// (walks to len(data)) and a starting index already at the end.
func TestJsondec_SkipJSONWhitespace_Ugly(t *testing.T) {
	data := []byte("   ")
	if i := SkipJSONWhitespace(data, 0); i != len(data) {
		t.Fatalf("all-whitespace: got %d want %d", i, len(data))
	}
	if i := SkipJSONWhitespace(data, len(data)); i != len(data) {
		t.Fatalf("start-at-end: got %d want %d", i, len(data))
	}
}

// TestJsondec_ParseJSONInt_Good covers positive integers of varying
// width.
func TestJsondec_ParseJSONInt_Good(t *testing.T) {
	cases := []struct {
		in   string
		want int64
	}{
		{`1`, 1},
		{`123456789`, 123456789},
	}
	for _, tc := range cases {
		t.Run(tc.in, func(t *testing.T) {
			n, _, err := ParseJSONInt([]byte(tc.in), 0)
			if err != nil {
				t.Fatalf("error = %v", err)
			}
			if n != tc.want {
				t.Fatalf("got %d want %d", n, tc.want)
			}
		})
	}
}

// TestJsondec_ParseJSONInt_Bad covers input with no digit at all —
// empty, a lone sign, a non-digit byte, and a leading '+' (JSON
// numbers never carry one).
func TestJsondec_ParseJSONInt_Bad(t *testing.T) {
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

// TestJsondec_ParseJSONInt_Ugly covers zero, a negative value, and a
// digit run wider than int64 can hold. The walker has no overflow
// guard — n = n*10 + digit wraps silently on 64-bit overflow, the
// same way a naive accumulator would. This pins the actual (rather
// than assumed) behaviour rather than asserting an overflow error
// the code does not produce.
func TestJsondec_ParseJSONInt_Ugly(t *testing.T) {
	n, _, err := ParseJSONInt([]byte(`0`), 0)
	if err != nil || n != 0 {
		t.Fatalf("zero: n=%d err=%v", n, err)
	}
	n, _, err = ParseJSONInt([]byte(`-987`), 0)
	if err != nil || n != -987 {
		t.Fatalf("negative: n=%d err=%v", n, err)
	}
	n, next, err := ParseJSONInt([]byte(`99999999999999999999`), 0)
	if err != nil {
		t.Fatalf("overflow digit run: unexpected error %v", err)
	}
	if next != 20 {
		t.Fatalf("overflow digit run: next=%d want 20", next)
	}
	if n != 7766279631452241919 {
		t.Fatalf("overflow digit run: n=%d want the wrapped int64 value 7766279631452241919", n)
	}
	// The digit-scan loop must stop AT the delimiter rather than
	// requiring EOF — the real call shape inside an array/object body
	// where a comma or brace follows the number directly.
	n, next, err = ParseJSONInt([]byte(`123,`), 0)
	if err != nil || n != 123 || next != 3 {
		t.Fatalf("embedded delimiter: n=%d next=%d err=%v", n, next, err)
	}
}

// TestJsondec_ParseJSONBool_Good covers the true and false literals.
func TestJsondec_ParseJSONBool_Good(t *testing.T) {
	v, next, err := ParseJSONBool([]byte(`true`), 0)
	if err != nil || v != true || next != 4 {
		t.Fatalf("true: v=%v next=%d err=%v", v, next, err)
	}
	v, next, err = ParseJSONBool([]byte(`false`), 0)
	if err != nil || v != false || next != 5 {
		t.Fatalf("false: v=%v next=%d err=%v", v, next, err)
	}
}

// TestJsondec_ParseJSONBool_Bad covers a partial literal that matches
// neither branch.
func TestJsondec_ParseJSONBool_Bad(t *testing.T) {
	_, _, err := ParseJSONBool([]byte(`tru`), 0)
	if err == nil {
		t.Fatalf("ParseJSONBool(tru) returned nil error")
	}
}

// TestJsondec_ParseJSONBool_Ugly pins a real gotcha: the matcher only
// checks the 4 (or 5) literal bytes and never checks what follows, so
// "truely" matches the "true" prefix and returns success even though
// the full token is not a valid JSON literal on its own. Callers rely
// on SkipJSONValue/dispatch code to only invoke this at a genuine
// value boundary — this test documents the contract rather than a bug
// to fix.
func TestJsondec_ParseJSONBool_Ugly(t *testing.T) {
	v, next, err := ParseJSONBool([]byte(`truely`), 0)
	if err != nil || v != true || next != 4 {
		t.Fatalf("prefix match: v=%v next=%d err=%v", v, next, err)
	}
}

// TestJsondec_IsJSONNull_Good covers the exact "null" literal.
func TestJsondec_IsJSONNull_Good(t *testing.T) {
	if !IsJSONNull([]byte(`null`), 0) {
		t.Fatalf("expected null match")
	}
}

// TestJsondec_IsJSONNull_Bad covers a too-short prefix and an offset
// mismatch — neither is a null literal at i.
func TestJsondec_IsJSONNull_Bad(t *testing.T) {
	if IsJSONNull([]byte(`nul`), 0) {
		t.Fatalf("expected no match on nul")
	}
	if IsJSONNull([]byte(`xnull`), 0) {
		t.Fatalf("expected no match on xnull")
	}
}

// TestJsondec_IsJSONNull_Ugly pins the same prefix-only contract as
// ParseJSONBool_Ugly — "nullx" matches because only the first 4 bytes
// are checked, and the caller does not advance i itself (IsJSONNull
// never mutates the index).
func TestJsondec_IsJSONNull_Ugly(t *testing.T) {
	if !IsJSONNull([]byte(`nullx`), 0) {
		t.Fatalf("expected prefix match on nullx")
	}
}

// TestJsondec_SkipJSONValue_Good covers every JSON value kind at
// least once, including nested structures.
func TestJsondec_SkipJSONValue_Good(t *testing.T) {
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
		{`{"a":1,"b":2}`, 13}, // two keys — walks the comma-continue branch
	}
	for _, tc := range cases {
		t.Run(tc.in, func(t *testing.T) {
			next, err := SkipJSONValue([]byte(tc.in), 0)
			if err != nil {
				t.Fatalf("error = %v", err)
			}
			if next != tc.want {
				t.Fatalf("got %d want %d", next, tc.want)
			}
		})
	}
}

// TestJsondec_SkipJSONValue_Bad covers structurally malformed input at
// every nesting level: empty data, an unrecognised leading byte, an
// "n"-prefixed literal that isn't "null", an object with an
// unterminated key (SkipJSONString itself fails while walking the
// key), an object missing its colon, a non-string object key, an
// object with a malformed value after the colon, an object missing
// the comma/brace after a value, an array with a malformed element,
// and an array missing its comma.
func TestJsondec_SkipJSONValue_Bad(t *testing.T) {
	cases := []string{
		``,
		`x`,
		`nope`,
		`{"a`,
		`{"a" 1}`,
		`{1:2}`,
		`{"a":}`,
		`{"a":1 "b":2}`,
		`[x]`,
		`[1 2]`,
		`{"a":1`,
		`[1,2`,
	}
	for _, in := range cases {
		t.Run(in, func(t *testing.T) {
			if _, err := SkipJSONValue([]byte(in), 0); err == nil {
				t.Fatalf("SkipJSONValue(%q) returned nil error, want error", in)
			}
		})
	}
}

// TestJsondec_SkipJSONValue_Ugly covers whitespace-padded empty
// containers — the fast-path branch inside skipJSONObject/
// skipJSONArray that returns before the general per-element loop.
func TestJsondec_SkipJSONValue_Ugly(t *testing.T) {
	next, err := SkipJSONValue([]byte(`{  }`), 0)
	if err != nil || next != 4 {
		t.Fatalf("padded empty object: next=%d err=%v", next, err)
	}
	next, err = SkipJSONValue([]byte(`[  ]`), 0)
	if err != nil || next != 4 {
		t.Fatalf("padded empty array: next=%d err=%v", next, err)
	}
}

// TestJsondec_SkipJSONString_Good covers strings of varying content,
// including multi-byte UTF-8.
func TestJsondec_SkipJSONString_Good(t *testing.T) {
	cases := []struct {
		in   string
		want int
	}{
		{`"abc"`, 5},
		{`""`, 2},
		{`"aÿb"`, 6}, // ÿ is 2 UTF-8 bytes inside the quotes
	}
	for _, tc := range cases {
		t.Run(tc.in, func(t *testing.T) {
			next, err := SkipJSONString([]byte(tc.in), 0)
			if err != nil {
				t.Fatalf("error = %v", err)
			}
			if next != tc.want {
				t.Fatalf("got %d want %d", next, tc.want)
			}
		})
	}
}

// TestJsondec_SkipJSONString_Bad covers every rejection branch: entry
// (index out of range or data[i] isn't a quote), a missing closing
// quote, a raw control byte in the body, a trailing backslash with
// nothing after it, and a truncated \uXXXX escape.
func TestJsondec_SkipJSONString_Bad(t *testing.T) {
	if _, err := SkipJSONString([]byte("abc"), 0); err == nil {
		t.Fatalf("no-quote: expected error")
	}
	if _, err := SkipJSONString([]byte(`"a`), 0); err == nil {
		t.Fatalf("unterminated: expected error")
	}
	if _, err := SkipJSONString([]byte("\"a\x01b\""), 0); err == nil {
		t.Fatalf("control byte: expected error")
	}
	if _, err := SkipJSONString([]byte(`"a`+bs), 0); err == nil {
		t.Fatalf("trailing backslash: expected error")
	}
	if _, err := SkipJSONString([]byte(`"a`+bs+`u12`), 0); err == nil {
		t.Fatalf("truncated unicode escape: expected error")
	}
}

// TestJsondec_SkipJSONString_Ugly covers escape shapes that advance
// the index by more than one byte per iteration: mnemonic escapes
// (2-byte advance) and back-to-back \uXXXX escapes (6-byte advance
// each, handled by the j+=5-then-continue branch).
func TestJsondec_SkipJSONString_Ugly(t *testing.T) {
	cases := []struct {
		in   string
		want int
	}{
		{`"a` + bs + `nb"`, 6},
		{`"a` + bs + `"b"`, 6},
		{`"a` + bs + bs + `b"`, 6},
		{`"` + u("0041") + u("0042") + `"`, 14},
	}
	for _, tc := range cases {
		t.Run(tc.in, func(t *testing.T) {
			next, err := SkipJSONString([]byte(tc.in), 0)
			if err != nil {
				t.Fatalf("error = %v", err)
			}
			if next != tc.want {
				t.Fatalf("got %d want %d", next, tc.want)
			}
		})
	}
}

// TestJsondec_MatchObjectStart_Good covers a leading '{' with
// preceding whitespace.
func TestJsondec_MatchObjectStart_Good(t *testing.T) {
	i, err := MatchObjectStart([]byte(`  {`), 0)
	if err != nil || i != 3 {
		t.Fatalf("i=%d err=%v", i, err)
	}
}

// TestJsondec_MatchObjectStart_Bad covers input that is not an
// object.
func TestJsondec_MatchObjectStart_Bad(t *testing.T) {
	_, err := MatchObjectStart([]byte(`123`), 0)
	if err == nil {
		t.Fatalf("expected error on non-object")
	}
}

// TestJsondec_MatchObjectStart_Ugly covers whitespace-only input that
// runs out before any '{' is found — the whitespace skip consumes the
// whole buffer and the boundary check must still reject it.
func TestJsondec_MatchObjectStart_Ugly(t *testing.T) {
	_, err := MatchObjectStart([]byte(``), 0)
	if err == nil {
		t.Fatalf("expected error on empty input")
	}
}

// TestJsondec_MatchArrayStart_Good covers a leading '[' with
// preceding whitespace.
func TestJsondec_MatchArrayStart_Good(t *testing.T) {
	i, err := MatchArrayStart([]byte(`  [`), 0)
	if err != nil || i != 3 {
		t.Fatalf("i=%d err=%v", i, err)
	}
}

// TestJsondec_MatchArrayStart_Bad covers input that is not an array.
func TestJsondec_MatchArrayStart_Bad(t *testing.T) {
	_, err := MatchArrayStart([]byte(`123`), 0)
	if err == nil {
		t.Fatalf("expected error on non-array")
	}
}

// TestJsondec_MatchArrayStart_Ugly covers whitespace-only input that
// runs out before any '[' is found.
func TestJsondec_MatchArrayStart_Ugly(t *testing.T) {
	_, err := MatchArrayStart([]byte(`   `), 0)
	if err == nil {
		t.Fatalf("expected error on whitespace-only input")
	}
}

// TestJsondec_ParseJSONFloat32_Good covers a plain decimal.
func TestJsondec_ParseJSONFloat32_Good(t *testing.T) {
	v, _, err := ParseJSONFloat32([]byte(`0.7`), 0)
	if err != nil || v != 0.7 {
		t.Fatalf("v=%v err=%v", v, err)
	}
}

// TestJsondec_ParseJSONFloat32_Bad covers two distinct rejection
// branches: no digit at all (i never advances past the sign) and a
// digit run strconv itself rejects (two decimal points).
func TestJsondec_ParseJSONFloat32_Bad(t *testing.T) {
	if _, _, err := ParseJSONFloat32([]byte(``), 0); err == nil {
		t.Fatalf("empty: expected error")
	}
	if _, _, err := ParseJSONFloat32([]byte(`1.2.3`), 0); err == nil {
		t.Fatalf("double decimal point: expected error")
	}
}

// TestJsondec_ParseJSONFloat32_Ugly covers scientific notation with a
// negative exponent — the shape "-1.5e2" — and confirms it parses to
// the equivalent expanded value.
func TestJsondec_ParseJSONFloat32_Ugly(t *testing.T) {
	v, _, err := ParseJSONFloat32([]byte(`-1.5e2`), 0)
	if err != nil || v != -150 {
		t.Fatalf("v=%v err=%v", v, err)
	}
	// The digit-scan loop must stop AT a following delimiter rather
	// than requiring EOF — the real call shape inside a JSON array.
	v, next, err := ParseJSONFloat32([]byte(`0.7,`), 0)
	if err != nil || v != 0.7 || next != 3 {
		t.Fatalf("embedded delimiter: v=%v next=%d err=%v", v, next, err)
	}
}

// TestJsondec_ParseJSONFloat64_Good covers a plain decimal and the
// leading-sign branch (a plain positive decimal never touches the
// optional '-' consumption at the top of the walker).
func TestJsondec_ParseJSONFloat64_Good(t *testing.T) {
	d, _, err := ParseJSONFloat64([]byte(`3.14`), 0)
	if err != nil || d != 3.14 {
		t.Fatalf("d=%v err=%v", d, err)
	}
	d, _, err = ParseJSONFloat64([]byte(`-3.14`), 0)
	if err != nil || d != -3.14 {
		t.Fatalf("negative: d=%v err=%v", d, err)
	}
}

// TestJsondec_ParseJSONFloat64_Bad mirrors ParseJSONFloat32_Bad's two
// rejection branches for the float64 variant.
func TestJsondec_ParseJSONFloat64_Bad(t *testing.T) {
	if _, _, err := ParseJSONFloat64([]byte(``), 0); err == nil {
		t.Fatalf("empty: expected error")
	}
	if _, _, err := ParseJSONFloat64([]byte(`1.2.3`), 0); err == nil {
		t.Fatalf("double decimal point: expected error")
	}
}

// TestJsondec_ParseJSONFloat64_Ugly covers scientific notation at
// both exponent extremes.
func TestJsondec_ParseJSONFloat64_Ugly(t *testing.T) {
	v, _, err := ParseJSONFloat64([]byte(`1e20`), 0)
	if err != nil || v != 1e20 {
		t.Fatalf("large: v=%v err=%v", v, err)
	}
	v, _, err = ParseJSONFloat64([]byte(`1e-10`), 0)
	if err != nil || v != 1e-10 {
		t.Fatalf("small: v=%v err=%v", v, err)
	}
	// The digit-scan loop must stop AT a following delimiter rather
	// than requiring EOF.
	v, next, err := ParseJSONFloat64([]byte(`3.14,`), 0)
	if err != nil || v != 3.14 || next != 4 {
		t.Fatalf("embedded delimiter: v=%v next=%d err=%v", v, next, err)
	}
}

// TestJsondec_CountJSONArrayElements_Good covers arrays of scalars,
// strings, and nested containers — each nested value counts as one
// element regardless of its internal structure.
func TestJsondec_CountJSONArrayElements_Good(t *testing.T) {
	cases := []struct {
		in   string
		want int
	}{
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

// TestJsondec_CountJSONArrayElements_Bad covers a body that goes
// malformed partway through. The count returned is however many
// elements parsed cleanly before the failure (2, not 0) — the
// function has no way to signal an error to a caller that only wants
// a count, so this pins the actual partial-count behaviour rather
// than the doc comment's "returns 0" description of the
// all-malformed-immediately case.
func TestJsondec_CountJSONArrayElements_Bad(t *testing.T) {
	got := CountJSONArrayElements([]byte(`1,2,x]`), 0)
	if got != 2 {
		t.Fatalf("got %d want 2 (partial count before the malformed element)", got)
	}
}

// TestJsondec_CountJSONArrayElements_Ugly covers the empty-array fast
// path — no elements at all, immediate ']'.
func TestJsondec_CountJSONArrayElements_Ugly(t *testing.T) {
	got := CountJSONArrayElements([]byte(`]`), 0)
	if got != 0 {
		t.Fatalf("got %d want 0", got)
	}
}

// TestJsondec_writeUTF8_FourByte directly exercises writeUTF8's
// >=0x10000 branch (the internal helper is unexported so this test
// lives in-package). No exported entry point can reach this branch
// today: ParseJSONString's only caller of writeUTF8 is the \uXXXX
// escape decoder, which is capped at a 4-hex-digit codepoint (max
// 0xFFFF) — codepoints above the Basic Multilingual Plane only reach
// JSON via a UTF-16 surrogate pair, which this package does not
// combine. Documented here rather than left silently uncovered.
func TestJsondec_writeUTF8_FourByte(t *testing.T) {
	var sb strings.Builder
	writeUTF8(&sb, 0x1F600) // U+1F600 GRINNING FACE
	got := sb.String()
	want := "\U0001F600"
	if got != want {
		t.Fatalf("got %q want %q", got, want)
	}
}

// TestJsondec_parseJSONUnicodeEscape_WrongLength directly exercises
// the defensive len(hex)!=4 guard (the internal helper is unexported
// so this test lives in-package). Also unreachable via any exported
// entry point: ParseJSONString always slices exactly 4 bytes
// (data[i+2:i+6]) before calling, having already bounds-checked
// i+6<=len(data). Documented rather than left silently uncovered.
func TestJsondec_parseJSONUnicodeEscape_WrongLength(t *testing.T) {
	if _, ok := parseJSONUnicodeEscape([]byte("041")); ok {
		t.Fatalf("expected ok=false for a 3-byte hex slice")
	}
	if _, ok := parseJSONUnicodeEscape([]byte("00410")); ok {
		t.Fatalf("expected ok=false for a 5-byte hex slice")
	}
}
