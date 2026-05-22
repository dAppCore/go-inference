// SPDX-Licence-Identifier: EUPL-1.2

package jsonenc

import (
	"encoding/json"
	"strconv"
	"testing"
)

// TestAppendJSONString_RoundTrip pins the escape contract of
// AppendJSONString against encoding/json's encoder. Every byte class
// (mnemonic escapes, \u00XX controls, plain ASCII, multi-byte UTF-8)
// must round-trip identically.
func TestAppendJSONString_RoundTrip(t *testing.T) {
	cases := []struct {
		name  string
		input string
	}{
		{"empty", ""},
		{"plain_ASCII", "answer"},
		{"quote", `say "hi"`},
		{"backslash", `path\to\file`},
		{"mnemonics", "\b\f\n\r\t"},
		{"control_low", "\x01\x02\x1f"},
		{"utf8", "café — résumé"},
		{"mixed", "line1\n\"quote\"\tend"},
		{"long_clean", "the quick brown fox jumps over the lazy dog — repeated bulk-copy fast-path"},
		{"escape_at_end", "clean prefix then\\"},
		{"escape_at_start", "\"quoted prefix"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := string(AppendJSONString(nil, tc.input))
			want, err := json.Marshal(tc.input)
			if err != nil {
				t.Fatalf("json.Marshal(%q) error: %v", tc.input, err)
			}
			// encoding/json HTML-escapes <, >, &; AppendJSONString
			// does not. None of the cases above exercise that branch,
			// so direct compare holds.
			if got != string(want) {
				t.Fatalf("AppendJSONString(%q):\n got = %s\nwant = %s", tc.input, got, want)
			}
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

// TestAppendJSONString_AppendsToExisting verifies the primitive
// appends without clobbering the leading bytes — load-bearing for
// the per-shape encoders that pre-populate `{"key":` before calling.
func TestAppendJSONString_AppendsToExisting(t *testing.T) {
	buf := []byte(`{"key":`)
	buf = AppendJSONString(buf, "value")
	if got, want := string(buf), `{"key":"value"`; got != want {
		t.Fatalf("append-onto: got %s want %s", got, want)
	}
}

// TestAppendStringField verifies the `"key":"value"` shape with and
// without leading comma.
func TestAppendStringField(t *testing.T) {
	buf := AppendStringField(nil, "model", "qwen3", false)
	if got, want := string(buf), `"model":"qwen3"`; got != want {
		t.Fatalf("no-comma: got %s want %s", got, want)
	}
	buf = AppendStringField(nil, "role", "assistant", true)
	if got, want := string(buf), `,"role":"assistant"`; got != want {
		t.Fatalf("leading-comma: got %s want %s", got, want)
	}
	// Escape contract carries through.
	buf = AppendStringField(nil, "content", "line1\n\"q\"", false)
	if got, want := string(buf), `"content":"line1\n\"q\""`; got != want {
		t.Fatalf("escapes: got %s want %s", got, want)
	}
}

// TestAppendIntField verifies the `"key":N` shape.
func TestAppendIntField(t *testing.T) {
	buf := AppendIntField(nil, "index", 0, false)
	if got, want := string(buf), `"index":0`; got != want {
		t.Fatalf("int zero: got %s want %s", got, want)
	}
	buf = AppendIntField(nil, "count", 256, true)
	if got, want := string(buf), `,"count":256`; got != want {
		t.Fatalf("int with comma: got %s want %s", got, want)
	}
	buf = AppendIntField(nil, "neg", -1, false)
	if got, want := string(buf), `"neg":-1`; got != want {
		t.Fatalf("int negative: got %s want %s", got, want)
	}
}

// TestAppendInt64Field covers wide int64 values that duration fields
// use (nanoseconds, easily >2^31).
func TestAppendInt64Field(t *testing.T) {
	buf := AppendInt64Field(nil, "total_duration", 1_500_000_000, false)
	if got, want := string(buf), `"total_duration":1500000000`; got != want {
		t.Fatalf("int64: got %s want %s", got, want)
	}
	buf = AppendInt64Field(nil, "max", 1<<62, true)
	if got, want := string(buf), `,"max":`+strconv.FormatInt(1<<62, 10); got != want {
		t.Fatalf("int64 large: got %s want %s", got, want)
	}
}

// TestAppendBoolField pins the Done-flag emission shape used by
// every per-token streaming chunk.
func TestAppendBoolField(t *testing.T) {
	buf := AppendBoolField(nil, "done", true, false)
	if got, want := string(buf), `"done":true`; got != want {
		t.Fatalf("bool true: got %s want %s", got, want)
	}
	buf = AppendBoolField(nil, "done", false, true)
	if got, want := string(buf), `,"done":false`; got != want {
		t.Fatalf("bool false: got %s want %s", got, want)
	}
}

// TestAppendFloat32Field verifies the inline `"key":F` form used by
// sampling parameters (temperature, top_p).
func TestAppendFloat32Field(t *testing.T) {
	buf := AppendFloat32Field(nil, "temperature", 0.7, false)
	if got, want := string(buf), `"temperature":0.7`; got != want {
		t.Fatalf("float32 field: got %s want %s", got, want)
	}
	buf = AppendFloat32Field(nil, "top_p", 0.95, true)
	if got, want := string(buf), `,"top_p":0.95`; got != want {
		t.Fatalf("float32 field with comma: got %s want %s", got, want)
	}
}

// TestAppendFloat32 verifies the bare-value emission shape used for
// embedding vector elements.
func TestAppendFloat32(t *testing.T) {
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
		got := string(AppendFloat32(nil, tc.in))
		if got != tc.want {
			t.Fatalf("float32(%v): got %s want %s", tc.in, got, tc.want)
		}
	}
}

// TestAppendFloat64 verifies the bare-value emission shape used for
// score / probability outputs.
func TestAppendFloat64(t *testing.T) {
	got := string(AppendFloat64(nil, 0.12345))
	if got != "0.12345" {
		t.Fatalf("float64: got %s want 0.12345", got)
	}
}

// TestHexChar covers the nibble-to-ASCII contract used by the
// \u00XX escape branch.
func TestHexChar(t *testing.T) {
	cases := []struct {
		in   byte
		want byte
	}{
		{0, '0'},
		{9, '9'},
		{10, 'a'},
		{15, 'f'},
		// High nibble masked off — only low 4 bits matter.
		{0xF0, '0'},
		{0xFF, 'f'},
	}
	for _, tc := range cases {
		got := HexChar(tc.in)
		if got != tc.want {
			t.Fatalf("HexChar(%#x): got %q want %q", tc.in, got, tc.want)
		}
	}
}
