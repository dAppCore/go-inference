// SPDX-Licence-Identifier: EUPL-1.2

// filestore hand-rolled JSON tests: append-encoder escape coverage and the extractRecordURI scanner's helper walkers.
package filestore

import "testing"

func TestHexChar_Good_Digits(t *testing.T) {
	cases := map[byte]byte{
		0x0: '0', 0x5: '5', 0x9: '9',
		0xa: 'a', 0xc: 'c', 0xf: 'f',
	}
	for v, want := range cases {
		if got := hexChar(v); got != want {
			t.Fatalf("hexChar(%#x) = %q, want %q", v, got, want)
		}
	}
}

func TestHexChar_Good_MasksHighNibble(t *testing.T) {
	// hexChar only consults the low nibble — a high nibble set must not
	// change the result.
	if got := hexChar(0xf0 | 0x3); got != '3' {
		t.Fatalf("hexChar(0xf3) = %q, want %q", got, "3")
	}
}

func TestAppendJSONString_Good_MnemonicEscapes(t *testing.T) {
	got := string(appendJSONString(nil, "a\bb\fc\rd\ne\tf\"g\\h/i"))
	want := `"a\bb\fc\rd\ne\tf\"g\\h/i"`
	if got != want {
		t.Fatalf("appendJSONString() = %s, want %s", got, want)
	}
}

func TestAppendJSONString_Good_ControlCharFallback(t *testing.T) {
	// Built from explicit byte values, not string escapes: the two
	// control bytes (0x01, 0x1f) as input, and the JSON \u00XX escape
	// form (a literal backslash-u-0-0-hex-hex run) as the expected
	// output. Composing both from byte slices sidesteps any ambiguity
	// between Go-string-escape and JSON-escape interpretation of a
	// literal "\x01"-style sequence typed directly into source.
	input := string([]byte{0x01, 0x1f})
	got := string(appendJSONString(nil, input))
	backslash := byte(0x5c)
	want := string([]byte{'"', backslash, 'u', '0', '0', '0', '1', backslash, 'u', '0', '0', '1', 'f', '"'})
	if got != want {
		t.Fatalf("appendJSONString(control chars) = % x, want % x", []byte(got), []byte(want))
	}
}

func TestAppendJSONField_Good_FirstOmitsComma(t *testing.T) {
	got := string(appendJSONField(nil, "k", "v", true))
	if got != `"k":"v"` {
		t.Fatalf("appendJSONField(first) = %s, want %s", got, `"k":"v"`)
	}
}

func TestAppendJSONField_Good_NonFirstPrependsComma(t *testing.T) {
	got := string(appendJSONField([]byte(`"a":"1"`), "k", "v", false))
	if got != `"a":"1","k":"v"` {
		t.Fatalf("appendJSONField(non-first) = %s, want %s", got, `"a":"1","k":"v"`)
	}
}

func TestJsonSkipWS_Good_SkipsLeadingWhitespace(t *testing.T) {
	i, err := jsonSkipWS([]byte(" \t\n\rX"), 0)
	if err != nil {
		t.Fatalf("jsonSkipWS() error = %v", err)
	}
	if i != 4 {
		t.Fatalf("jsonSkipWS() = %d, want 4", i)
	}
}

func TestJsonSkipWS_Good_NoLeadingWhitespace(t *testing.T) {
	i, err := jsonSkipWS([]byte("X"), 0)
	if err != nil || i != 0 {
		t.Fatalf("jsonSkipWS() = (%d, %v), want (0, nil)", i, err)
	}
}

func TestJsonSkipWS_Bad_TruncatedAllWhitespace(t *testing.T) {
	if _, err := jsonSkipWS([]byte("  \t"), 0); err == nil {
		t.Fatal("jsonSkipWS(all whitespace) error = nil, want truncation error")
	}
}

func TestJsonSkipString_Good_NoEscapes(t *testing.T) {
	i, err := jsonSkipString([]byte(`"abc"rest`), 0)
	if err != nil {
		t.Fatalf("jsonSkipString() error = %v", err)
	}
	if i != 5 {
		t.Fatalf("jsonSkipString() = %d, want 5", i)
	}
}

func TestJsonSkipString_Good_WithEscape(t *testing.T) {
	i, err := jsonSkipString([]byte(`"a\"b"rest`), 0)
	if err != nil {
		t.Fatalf("jsonSkipString() error = %v", err)
	}
	if i != 6 {
		t.Fatalf("jsonSkipString() = %d, want 6", i)
	}
}

func TestJsonSkipString_Bad_NotAString(t *testing.T) {
	if _, err := jsonSkipString([]byte("nope"), 0); err == nil {
		t.Fatal("jsonSkipString(not a string) error = nil")
	}
}

func TestJsonSkipString_Bad_TrailingEscape(t *testing.T) {
	if _, err := jsonSkipString([]byte(`"a\`), 0); err == nil {
		t.Fatal("jsonSkipString(trailing escape) error = nil")
	}
}

func TestJsonSkipString_Bad_Unterminated(t *testing.T) {
	if _, err := jsonSkipString([]byte(`"abc`), 0); err == nil {
		t.Fatal("jsonSkipString(unterminated) error = nil")
	}
}

func TestJsonReadString_Good_FastPathNoEscape(t *testing.T) {
	value, end, err := jsonReadString([]byte(`"abc"rest`), 0)
	if err != nil {
		t.Fatalf("jsonReadString() error = %v", err)
	}
	if value != "abc" || end != 5 {
		t.Fatalf("jsonReadString() = (%q, %d), want (\"abc\", 5)", value, end)
	}
}

func TestJsonReadString_Good_SlowPathWithEscape(t *testing.T) {
	value, end, err := jsonReadString([]byte(`"a\tb"rest`), 0)
	if err != nil {
		t.Fatalf("jsonReadString() error = %v", err)
	}
	if value != "a\tb" || end != 6 {
		t.Fatalf("jsonReadString() = (%q, %d), want (\"a\\tb\", 6)", value, end)
	}
}

func TestJsonReadString_Bad_NotAString(t *testing.T) {
	if _, _, err := jsonReadString([]byte("nope"), 0); err == nil {
		t.Fatal("jsonReadString(not a string) error = nil")
	}
}

func TestJsonReadString_Bad_TrailingEscape(t *testing.T) {
	if _, _, err := jsonReadString([]byte(`"a\`), 0); err == nil {
		t.Fatal("jsonReadString(trailing escape) error = nil")
	}
}

func TestJsonReadString_Bad_Unterminated(t *testing.T) {
	if _, _, err := jsonReadString([]byte(`"abc`), 0); err == nil {
		t.Fatal("jsonReadString(unterminated) error = nil")
	}
}

func TestJsonReadString_Bad_InvalidEscapeInValue(t *testing.T) {
	// hasEscape=true routes through jsonUnescape on the slow path;
	// an unrecognised escape must surface as an error there.
	if _, _, err := jsonReadString([]byte(`"a\qb"`), 0); err == nil {
		t.Fatal("jsonReadString(invalid escape) error = nil")
	}
}

func TestJsonUnescape_Good_MnemonicEscapes(t *testing.T) {
	got, err := jsonUnescape([]byte(`\"\\\/\b\f\n\r\t`))
	if err != nil {
		t.Fatalf("jsonUnescape() error = %v", err)
	}
	want := "\"\\/\b\f\n\r\t"
	if got != want {
		t.Fatalf("jsonUnescape() = %q, want %q", got, want)
	}
}

func TestJsonUnescape_Good_UnicodeEscapeRanges(t *testing.T) {
	// Genuine \u-escaped input (not raw UTF-8 bytes) so the hex-decode
	// loop and each UTF-8 emission-length branch actually run. Digit
	// AND letter hex digits are mixed (both cases) across the three
	// reachable emission ranges — a lone \uXXXX escape maxes out at
	// 0xFFFF, so the 4-byte "r >= 0x10000" branch is unreachable by
	// construction (see final report). Built from explicit byte
	// values (see the ControlCharFallback test above for why) rather
	// than a \uXXXX escape typed directly into source.
	bs := byte(0x5c)
	uEscape := func(h0, h1, h2, h3 byte) string {
		return string([]byte{bs, 'u', h0, h1, h2, h3})
	}
	cases := []struct {
		name string
		in   string
		want string
	}{
		{"ascii-digit-hex", uEscape('0', '0', '4', '1'), "A"},          // < 0x80, hex digits only
		{"two-byte-lowercase-hex", uEscape('0', '0', 'e', '9'), "é"},   // < 0x800, lowercase a-f
		{"three-byte-uppercase-hex", uEscape('6', '5', 'E', '5'), "日"}, // < 0x10000, uppercase A-F
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := jsonUnescape([]byte(tc.in))
			if err != nil {
				t.Fatalf("jsonUnescape(%s) error = %v", tc.name, err)
			}
			if got != tc.want {
				t.Fatalf("jsonUnescape(%s) = %q, want %q", tc.name, got, tc.want)
			}
		})
	}
}

func TestJsonUnescape_Good_LiteralPassthrough(t *testing.T) {
	got, err := jsonUnescape([]byte(`plain\ntext`))
	if err != nil {
		t.Fatalf("jsonUnescape() error = %v", err)
	}
	if got != "plain\ntext" {
		t.Fatalf("jsonUnescape() = %q, want %q", got, "plain\ntext")
	}
}

func TestJsonUnescape_Bad_TrailingEscape(t *testing.T) {
	if _, err := jsonUnescape([]byte(`\`)); err == nil {
		t.Fatal("jsonUnescape(trailing escape) error = nil")
	}
}

func TestJsonUnescape_Bad_ShortUnicodeEscape(t *testing.T) {
	if _, err := jsonUnescape([]byte(`\u12`)); err == nil {
		t.Fatal("jsonUnescape(short \\u escape) error = nil")
	}
}

func TestJsonUnescape_Bad_InvalidUnicodeHexDigit(t *testing.T) {
	if _, err := jsonUnescape([]byte(`\u12zz`)); err == nil {
		t.Fatal("jsonUnescape(invalid \\u hex digit) error = nil")
	}
}

func TestJsonUnescape_Bad_UnknownEscape(t *testing.T) {
	if _, err := jsonUnescape([]byte(`\q`)); err == nil {
		t.Fatal("jsonUnescape(unknown escape) error = nil")
	}
}

func TestJsonSkipValue_Good_Kinds(t *testing.T) {
	cases := []struct {
		name string
		in   string
		want int
	}{
		{"string", `"abc"`, 5},
		{"true", `true`, 4},
		{"false", `false`, 5},
		{"null", `null`, 4},
		{"int", `123`, 3},
		{"negative", `-42`, 3},
		{"decimal", `3.14`, 4},
		{"exponent", `1e10`, 4},
		{"empty-object", `{}`, 2},
		{"nested-object", `{"a":{"b":1}}`, 13},
		{"empty-array", `[]`, 2},
		{"nested-array", `[1,[2,3],"x"]`, 13},
		{"array-with-string-containing-brackets", `["[{}]"]`, 8},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			end, err := jsonSkipValue([]byte(tc.in), 0)
			if err != nil {
				t.Fatalf("jsonSkipValue(%s) error = %v", tc.name, err)
			}
			if end != tc.want {
				t.Fatalf("jsonSkipValue(%s) = %d, want %d", tc.name, end, tc.want)
			}
		})
	}
}

func TestJsonSkipValue_Bad_MalformedInputs(t *testing.T) {
	cases := []struct {
		name string
		in   string
	}{
		{"truncated-at-eof", ""},
		{"bad-true", "tru3"},
		{"bad-false", "fals3"},
		{"bad-null", "nul3"},
		// No "empty-number" case: jsonSkipValue only enters the number
		// branch when data[i] is already '-' or a digit, and both
		// sub-paths (leading '-' pre-increments j; a leading digit is
		// re-consumed by the loop's first iteration) always advance j
		// past i by at least one byte. The case's own "if j == i"
		// guard is therefore unreachable from any caller — dead code,
		// left untested (see final report).
		{"unbalanced-object", `{"a":1`},
		{"mismatched-object-close", `{"a":1]`},
		{"mismatched-array-close", `[1}`},
		{"invalid-value", `?`},
		{"nested-string-unterminated", `{"a":"bad`},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := jsonSkipValue([]byte(tc.in), 0); err == nil {
				t.Fatalf("jsonSkipValue(%s) error = nil, want error", tc.name)
			}
		})
	}
}

func TestExtractRecordURI_Good_UnusualShapes(t *testing.T) {
	cases := []struct {
		name string
		in   string
		want string
	}{
		{"uri-not-first-key", `{"kind":"kv","uri":"mlx://a"}`, "mlx://a"},
		{"three-byte-key-not-uri", `{"xyz":"skip","uri":"mlx://b"}`, "mlx://b"},
		{"duplicate-uri-keeps-first", `{"uri":"first","uri":"second"}`, "first"},
		{"uri-value-has-escape", `{"uri":"mlx:\/\/c"}`, "mlx://c"},
		{"no-uri-key", `{"kind":"kv"}`, ""},
		{"whitespace-padded", "{ \"uri\" : \"mlx://d\" }", "mlx://d"},
		// Flat tags object (matching the real recordMeta shape) sat
		// before the uri key — proves jsonSkipValue's object-skip
		// path is used to jump over a non-uri value before the scan
		// reaches uri. Deliberately not a mixed {..[..]..} shape: see
		// the jsonSkipValue depth-tracking note in the final report.
		{"nested-value-before-uri", `{"tags":{"a":"1","b":"2"},"uri":"mlx://e"}`, "mlx://e"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := extractRecordURI([]byte(tc.in))
			if err != nil {
				t.Fatalf("extractRecordURI(%s) error = %v", tc.name, err)
			}
			if got != tc.want {
				t.Fatalf("extractRecordURI(%s) = %q, want %q", tc.name, got, tc.want)
			}
		})
	}
}

func TestExtractRecordURI_Bad_MalformedInputs(t *testing.T) {
	cases := []struct {
		name string
		in   string
	}{
		{"not-an-object", `"just a string"`},
		{"truncated-empty", ``},
		{"missing-comma", `{"a":1 "b":2}`},
		{"key-not-string", `{1:2}`},
		{"missing-colon", `{"a" 1}`},
		{"uri-value-not-string", `{"uri":1}`},
		{"trailing-garbage", `{"uri":"x"} trailing`},
		{"truncated-after-comma", `{"a":1,`},
		{"unterminated-key", `{"unterminated`},
		{"truncated-after-key", `{"a"`},
		{"truncated-after-colon", `{"a":`},
		{"uri-value-unterminated", `{"uri":"unterminated`},
		{"non-uri-value-malformed", `{"other":xyz}`},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := extractRecordURI([]byte(tc.in)); err == nil {
				t.Fatalf("extractRecordURI(%s) error = nil, want error", tc.name)
			}
		})
	}
}
