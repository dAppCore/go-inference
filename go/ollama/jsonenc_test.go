// SPDX-Licence-Identifier: EUPL-1.2

package ollama

import (
	"encoding/json"
	"testing"
)

// TestAppendJSONString_Good pins the escape contract of appendJSONString
// against encoding/json's encoder. Every byte class (mnemonic escapes,
// \u00XX controls, plain ASCII, multi-byte UTF-8) must round-trip
// identically.
func TestAppendJSONString_Good(t *testing.T) {
	cases := []struct {
		name  string
		input string
	}{
		{"empty", ""},
		{"plain ASCII", "answer"},
		{"quote", `say "hi"`},
		{"backslash", `path\to\file`},
		{"mnemonics", "\b\f\n\r\t"},
		{"control 0x01", "\x01\x02\x1f"},
		{"utf8", "café — résumé"},
		{"mixed", "line1\n\"quote\"\tend"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := string(appendJSONString(nil, tc.input))
			want, err := json.Marshal(tc.input)
			if err != nil {
				t.Fatalf("json.Marshal(%q) error: %v", tc.input, err)
			}
			// encoding/json HTML-escapes <, >, &; appendJSONString does not.
			// None of the cases above exercise that branch, so direct
			// compare holds.
			if got != string(want) {
				t.Fatalf("appendJSONString(%q):\n got = %s\nwant = %s", tc.input, got, want)
			}
			// Round-trip back into Go via encoding/json.
			var parsed string
			if err := json.Unmarshal([]byte(got), &parsed); err != nil {
				t.Fatalf("Unmarshal(%s): %v", got, err)
			}
			if parsed != tc.input {
				t.Fatalf("round-trip drift:\n got = %q\nwant = %q", parsed, tc.input)
			}
		})
	}
}

// TestAppendStringField_Good verifies the `"key":"value"` shape with
// and without leading comma.
func TestAppendStringField_Good(t *testing.T) {
	buf := appendStringField(nil, "model", "qwen3", false)
	if got, want := string(buf), `"model":"qwen3"`; got != want {
		t.Fatalf("no-comma: got %s want %s", got, want)
	}
	buf = appendStringField(nil, "role", "assistant", true)
	if got, want := string(buf), `,"role":"assistant"`; got != want {
		t.Fatalf("leading-comma: got %s want %s", got, want)
	}
}

// TestAppendIntField_Good verifies the `"key":N` shape.
func TestAppendIntField_Good(t *testing.T) {
	buf := appendIntField(nil, "index", 0, false)
	if got, want := string(buf), `"index":0`; got != want {
		t.Fatalf("int zero: got %s want %s", got, want)
	}
	buf = appendIntField(nil, "count", 256, true)
	if got, want := string(buf), `,"count":256`; got != want {
		t.Fatalf("int with comma: got %s want %s", got, want)
	}
}

// TestAppendInt64Field_Good covers wide int64 values that the duration
// fields use (nanoseconds, easily >2^31).
func TestAppendInt64Field_Good(t *testing.T) {
	buf := appendInt64Field(nil, "total_duration", 1_500_000_000, false)
	if got, want := string(buf), `"total_duration":1500000000`; got != want {
		t.Fatalf("int64: got %s want %s", got, want)
	}
}

// TestAppendBoolField_Good verifies the Done flag emission shape.
func TestAppendBoolField_Good(t *testing.T) {
	buf := appendBoolField(nil, "done", true, false)
	if got, want := string(buf), `"done":true`; got != want {
		t.Fatalf("bool true: got %s want %s", got, want)
	}
	buf = appendBoolField(nil, "done", false, true)
	if got, want := string(buf), `,"done":false`; got != want {
		t.Fatalf("bool false: got %s want %s", got, want)
	}
}

// TestAppendFloat32_Good verifies sampling-field emission shape for
// the bounded value ranges ollama Options carry (Temperature [0,2],
// TopP [0,1]). encoding/json uses a magnitude-conditional path for
// floats that switches between 'e' and 'f' representations; the 'g'
// path here matches for the in-band sampling-field range, which is
// the only space this primitive serves in this adapter.
func TestAppendFloat32_Good(t *testing.T) {
	cases := []struct {
		in   float32
		want string
	}{
		{0.7, "0.7"},
		{0.95, "0.95"},
		{1.0, "1"},
		{0.0001, "0.0001"},
		{2.0, "2"},
	}
	for _, tc := range cases {
		got := string(appendFloat32(nil, tc.in))
		if got != tc.want {
			t.Fatalf("float32(%v): got %s want %s", tc.in, got, tc.want)
		}
	}
}
