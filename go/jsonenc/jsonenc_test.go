// SPDX-Licence-Identifier: EUPL-1.2

package jsonenc

import (
	"encoding/json"
	"math"
	"strconv"
	"testing"
)

// roundTrip checks got against encoding/json's own encoder for the
// same input, then unmarshals got back and checks it matches in.
// Shared by every AppendJSONString variant below.
func roundTrip(t *testing.T, in, got string) {
	t.Helper()
	want, err := json.Marshal(in)
	if err != nil {
		t.Fatalf("json.Marshal(%q) error: %v", in, err)
	}
	if got != string(want) {
		t.Fatalf("AppendJSONString(%q):\n got = %s\nwant = %s", in, got, string(want))
	}
	var parsed string
	if err := json.Unmarshal([]byte(got), &parsed); err != nil {
		t.Fatalf("Unmarshal(%s): %v", got, err)
	}
	if parsed != in {
		t.Fatalf("round-trip drift:\n got = %q\nwant = %q", parsed, in)
	}
}

// TestJsonenc_AppendJSONString_Good covers strings that need no
// escaping — the single bulk-copy fast path — verified against
// encoding/json's own encoder and round-tripped back through
// encoding/json.Unmarshal.
func TestJsonenc_AppendJSONString_Good(t *testing.T) {
	cases := []string{
		"answer",
		"the quick brown fox jumps over the lazy dog — repeated bulk-copy fast-path",
		"café — résumé",
	}
	for _, in := range cases {
		t.Run(in, func(t *testing.T) {
			roundTrip(t, in, string(AppendJSONString(nil, in)))
		})
	}
}

// TestJsonenc_AppendJSONString_Bad covers strings that force the
// escape path — mnemonic escapes, \u00XX controls, and a naive
// injection attempt (an embedded quote) that must come out neutralised
// rather than breaking the surrounding JSON structure.
func TestJsonenc_AppendJSONString_Bad(t *testing.T) {
	cases := []string{
		`say "hi"`,
		`path\to\file`,
		"\b\f\n\r\t",
		"\x01\x02\x1f",
		"line1\n\"quote\"\tend",
		`","injected":"true`,
	}
	for _, in := range cases {
		t.Run(in, func(t *testing.T) {
			roundTrip(t, in, string(AppendJSONString(nil, in)))
		})
	}
}

// TestJsonenc_AppendJSONString_Ugly covers the boundary conditions of
// the scan-then-split logic — empty input, an escape landing on the
// very first byte, and an escape landing on the very last byte —
// where an off-by-one in the split point would corrupt the output.
func TestJsonenc_AppendJSONString_Ugly(t *testing.T) {
	cases := []string{
		"",
		"\"quoted prefix",
		"clean prefix then\\",
	}
	for _, in := range cases {
		t.Run(in, func(t *testing.T) {
			roundTrip(t, in, string(AppendJSONString(nil, in)))
		})
	}
	// Appending onto a pre-populated buffer must not clobber the
	// existing bytes — load-bearing for the per-shape encoders that
	// pre-write `{"key":` before calling.
	buf := []byte(`{"key":`)
	buf = AppendJSONString(buf, "value")
	if got, want := string(buf), `{"key":"value"`; got != want {
		t.Fatalf("append-onto: got %s want %s", got, want)
	}
}

// TestJsonenc_AppendStringField_Good covers the `"key":"value"` shape
// with and without a leading comma, the two call shapes every
// per-field encoder uses.
func TestJsonenc_AppendStringField_Good(t *testing.T) {
	buf := AppendStringField(nil, "model", "qwen3", false)
	if got, want := string(buf), `"model":"qwen3"`; got != want {
		t.Fatalf("no-comma: got %s want %s", got, want)
	}
	buf = AppendStringField(nil, "role", "assistant", true)
	if got, want := string(buf), `,"role":"assistant"`; got != want {
		t.Fatalf("leading-comma: got %s want %s", got, want)
	}
}

// TestJsonenc_AppendStringField_Bad covers a value that forces the
// escape contract to carry through the field wrapper.
func TestJsonenc_AppendStringField_Bad(t *testing.T) {
	buf := AppendStringField(nil, "content", "line1\n\"q\"", false)
	if got, want := string(buf), `"content":"line1\n\"q\""`; got != want {
		t.Fatalf("escapes: got %s want %s", got, want)
	}
}

// TestJsonenc_AppendStringField_Ugly covers empty key and empty value
// — both legal call shapes the function never rejects.
func TestJsonenc_AppendStringField_Ugly(t *testing.T) {
	buf := AppendStringField(nil, "", "", false)
	if got, want := string(buf), `"":""`; got != want {
		t.Fatalf("empty key+value: got %s want %s", got, want)
	}
	buf = AppendStringField(nil, "note", "", true)
	if got, want := string(buf), `,"note":""`; got != want {
		t.Fatalf("empty value: got %s want %s", got, want)
	}
}

// TestJsonenc_AppendIntField_Good covers typical positive values.
func TestJsonenc_AppendIntField_Good(t *testing.T) {
	buf := AppendIntField(nil, "count", 256, true)
	if got, want := string(buf), `,"count":256`; got != want {
		t.Fatalf("got %s want %s", got, want)
	}
}

// TestJsonenc_AppendIntField_Bad covers a negative value.
func TestJsonenc_AppendIntField_Bad(t *testing.T) {
	buf := AppendIntField(nil, "neg", -1, false)
	if got, want := string(buf), `"neg":-1`; got != want {
		t.Fatalf("got %s want %s", got, want)
	}
}

// TestJsonenc_AppendIntField_Ugly covers zero and the platform int
// boundary values — the widest range the type can carry.
func TestJsonenc_AppendIntField_Ugly(t *testing.T) {
	buf := AppendIntField(nil, "index", 0, false)
	if got, want := string(buf), `"index":0`; got != want {
		t.Fatalf("zero: got %s want %s", got, want)
	}
	buf = AppendIntField(nil, "max", math.MaxInt, false)
	if got, want := string(buf), `"max":`+strconv.Itoa(math.MaxInt); got != want {
		t.Fatalf("max int: got %s want %s", got, want)
	}
	buf = AppendIntField(nil, "min", math.MinInt, false)
	if got, want := string(buf), `"min":`+strconv.Itoa(math.MinInt); got != want {
		t.Fatalf("min int: got %s want %s", got, want)
	}
}

// TestJsonenc_AppendInt64Field_Good covers the wide nanosecond-duration
// values every timing field in the adapters carries.
func TestJsonenc_AppendInt64Field_Good(t *testing.T) {
	buf := AppendInt64Field(nil, "total_duration", 1_500_000_000, false)
	if got, want := string(buf), `"total_duration":1500000000`; got != want {
		t.Fatalf("got %s want %s", got, want)
	}
}

// TestJsonenc_AppendInt64Field_Bad covers a negative int64.
func TestJsonenc_AppendInt64Field_Bad(t *testing.T) {
	buf := AppendInt64Field(nil, "delta", -42, true)
	if got, want := string(buf), `,"delta":-42`; got != want {
		t.Fatalf("got %s want %s", got, want)
	}
}

// TestJsonenc_AppendInt64Field_Ugly covers the int64 boundary values —
// beyond the 32-bit range that a naive int-sized encoder would
// truncate.
func TestJsonenc_AppendInt64Field_Ugly(t *testing.T) {
	buf := AppendInt64Field(nil, "max", math.MaxInt64, true)
	if got, want := string(buf), `,"max":`+strconv.FormatInt(math.MaxInt64, 10); got != want {
		t.Fatalf("max int64: got %s want %s", got, want)
	}
	buf = AppendInt64Field(nil, "min", math.MinInt64, false)
	if got, want := string(buf), `"min":`+strconv.FormatInt(math.MinInt64, 10); got != want {
		t.Fatalf("min int64: got %s want %s", got, want)
	}
}

// TestJsonenc_AppendBoolField_Good covers the true-value branch, the
// Done-flag shape used on the final chunk of a streaming response.
func TestJsonenc_AppendBoolField_Good(t *testing.T) {
	buf := AppendBoolField(nil, "done", true, false)
	if got, want := string(buf), `"done":true`; got != want {
		t.Fatalf("got %s want %s", got, want)
	}
}

// TestJsonenc_AppendBoolField_Bad covers the false-value branch — a
// structurally different literal (5 bytes vs 4) worth its own case.
func TestJsonenc_AppendBoolField_Bad(t *testing.T) {
	buf := AppendBoolField(nil, "done", false, true)
	if got, want := string(buf), `,"done":false`; got != want {
		t.Fatalf("got %s want %s", got, want)
	}
}

// TestJsonenc_AppendBoolField_Ugly chains multiple fields onto a
// pre-populated buffer — the shape every real streaming chunk uses —
// to guard the leading-comma boundary across repeated calls.
func TestJsonenc_AppendBoolField_Ugly(t *testing.T) {
	buf := []byte(`{"model":"x"`)
	buf = AppendBoolField(buf, "done", true, true)
	buf = AppendBoolField(buf, "stream", false, true)
	if got, want := string(buf), `{"model":"x","done":true,"stream":false`; got != want {
		t.Fatalf("chained: got %s want %s", got, want)
	}
}

// TestJsonenc_AppendFloat32Field_Good covers the sampling-parameter
// shape (temperature, top_p).
func TestJsonenc_AppendFloat32Field_Good(t *testing.T) {
	buf := AppendFloat32Field(nil, "temperature", 0.7, false)
	if got, want := string(buf), `"temperature":0.7`; got != want {
		t.Fatalf("got %s want %s", got, want)
	}
	buf = AppendFloat32Field(nil, "top_p", 0.95, true)
	if got, want := string(buf), `,"top_p":0.95`; got != want {
		t.Fatalf("got %s want %s", got, want)
	}
}

// TestJsonenc_AppendFloat32Field_Bad covers a negative value.
func TestJsonenc_AppendFloat32Field_Bad(t *testing.T) {
	buf := AppendFloat32Field(nil, "bias", -0.5, false)
	if got, want := string(buf), `"bias":-0.5`; got != want {
		t.Fatalf("got %s want %s", got, want)
	}
}

// TestJsonenc_AppendFloat32Field_Ugly covers zero and a very small
// value — the precision boundary 'g' formatting must not round away.
func TestJsonenc_AppendFloat32Field_Ugly(t *testing.T) {
	buf := AppendFloat32Field(nil, "epsilon", 0.0001, false)
	if got, want := string(buf), `"epsilon":0.0001`; got != want {
		t.Fatalf("small: got %s want %s", got, want)
	}
	buf = AppendFloat32Field(nil, "zero", 0, false)
	if got, want := string(buf), `"zero":0`; got != want {
		t.Fatalf("zero: got %s want %s", got, want)
	}
}

// TestJsonenc_AppendFloat32_Good covers the bare-value emission shape
// used for per-element embedding-vector output.
func TestJsonenc_AppendFloat32_Good(t *testing.T) {
	cases := []struct {
		in   float32
		want string
	}{
		{0.7, "0.7"},
		{0.95, "0.95"},
	}
	for _, tc := range cases {
		got := string(AppendFloat32(nil, tc.in))
		if got != tc.want {
			t.Fatalf("float32(%v): got %s want %s", tc.in, got, tc.want)
		}
	}
}

// TestJsonenc_AppendFloat32_Bad covers a negative value.
func TestJsonenc_AppendFloat32_Bad(t *testing.T) {
	got := string(AppendFloat32(nil, -1.5))
	if got != "-1.5" {
		t.Fatalf("got %s want -1.5", got)
	}
}

// TestJsonenc_AppendFloat32_Ugly covers an integer-valued float (must
// render without a trailing ".0"), a very small value, and the
// largest finite float32.
func TestJsonenc_AppendFloat32_Ugly(t *testing.T) {
	cases := []struct {
		in   float32
		want string
	}{
		{1.0, "1"},
		{2.0, "2"},
		{0.0001, "0.0001"},
		{math.MaxFloat32, "3.4028235e+38"},
	}
	for _, tc := range cases {
		got := string(AppendFloat32(nil, tc.in))
		if got != tc.want {
			t.Fatalf("float32(%v): got %s want %s", tc.in, got, tc.want)
		}
	}
}

// TestJsonenc_AppendFloat64_Good covers the bare-value emission shape
// used for score / probability outputs.
func TestJsonenc_AppendFloat64_Good(t *testing.T) {
	got := string(AppendFloat64(nil, 0.12345))
	if got != "0.12345" {
		t.Fatalf("got %s want 0.12345", got)
	}
}

// TestJsonenc_AppendFloat64_Bad covers a negative value.
func TestJsonenc_AppendFloat64_Bad(t *testing.T) {
	got := string(AppendFloat64(nil, -3.14))
	if got != "-3.14" {
		t.Fatalf("got %s want -3.14", got)
	}
}

// TestJsonenc_AppendFloat64_Ugly covers zero and scientific-notation
// boundaries at both ends of the exponent range.
func TestJsonenc_AppendFloat64_Ugly(t *testing.T) {
	cases := []struct {
		in   float64
		want string
	}{
		{0, "0"},
		{1e20, "1e+20"},
		{1e-10, "1e-10"},
	}
	for _, tc := range cases {
		got := string(AppendFloat64(nil, tc.in))
		if got != tc.want {
			t.Fatalf("float64(%v): got %s want %s", tc.in, got, tc.want)
		}
	}
}

// TestJsonenc_HexChar_Good covers the digit range 0-9.
func TestJsonenc_HexChar_Good(t *testing.T) {
	for v := byte(0); v <= 9; v++ {
		if got, want := HexChar(v), '0'+v; got != want {
			t.Fatalf("HexChar(%d): got %q want %q", v, got, want)
		}
	}
}

// TestJsonenc_HexChar_Bad covers the letter range a-f (10-15) — a
// structurally different branch from the digit range.
func TestJsonenc_HexChar_Bad(t *testing.T) {
	cases := []struct {
		in   byte
		want byte
	}{
		{10, 'a'}, {11, 'b'}, {12, 'c'}, {13, 'd'}, {14, 'e'}, {15, 'f'},
	}
	for _, tc := range cases {
		if got := HexChar(tc.in); got != tc.want {
			t.Fatalf("HexChar(%d): got %q want %q", tc.in, got, tc.want)
		}
	}
}

// TestJsonenc_HexChar_Ugly covers out-of-nibble input — only the low
// 4 bits are significant, so any byte above 15 masks down rather than
// panicking or indexing out of range. The escape branch in
// AppendJSONString only ever calls HexChar with c>>4 or c&0x0f (both
// already <=15); this pins the masking contract for any future caller
// that passes a raw byte.
func TestJsonenc_HexChar_Ugly(t *testing.T) {
	cases := []struct {
		in   byte
		want byte
	}{
		{0xF0, '0'},
		{0xFF, 'f'},
		{0x1A, 'a'},
		{0x2C, 'c'},
	}
	for _, tc := range cases {
		if got := HexChar(tc.in); got != tc.want {
			t.Fatalf("HexChar(%#x): got %q want %q", tc.in, got, tc.want)
		}
	}
}
