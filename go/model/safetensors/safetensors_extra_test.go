// SPDX-Licence-Identifier: EUPL-1.2

package safetensors

import (
	"testing"
)

// TestParseHeaderFieldErrors drives the per-entry validation branches in Parse that the
// existing TestParseErrors leaves uncovered: a tensor whose "dtype" key is absent (or not a
// string), whose "shape" key is absent (or not an array), whose shape holds a non-numeric /
// negative entry, whose "data_offsets" is missing or the wrong length, and whose offsets are
// non-numeric. Each must be rejected with a non-nil error. No model load — synthetic blobs.
func TestParseHeaderFieldErrors(t *testing.T) {
	d := func(n int) []byte { return make([]byte, n) }
	cases := []struct {
		name string
		blob []byte
	}{
		{"dtype missing", blob(`{"a":{"shape":[4],"data_offsets":[0,4]}}`, d(4))},
		{"dtype not a string", blob(`{"a":{"dtype":7,"shape":[4],"data_offsets":[0,4]}}`, d(4))},
		{"shape missing", blob(`{"a":{"dtype":"U8","data_offsets":[0,4]}}`, d(4))},
		// a missing shape whose span COINCIDENTALLY equals dtype×∏(empty)=elem (F32: 4B), so the
		// span check alone would pass it — only the explicit missing-shape guard rejects it.
		{"shape missing, span matches scalar", blob(`{"a":{"dtype":"F32","data_offsets":[0,4]}}`, d(4))},
		{"shape not an array", blob(`{"a":{"dtype":"U8","shape":4,"data_offsets":[0,4]}}`, d(4))},
		{"shape entry non-numeric", blob(`{"a":{"dtype":"U8","shape":["x"],"data_offsets":[0,4]}}`, d(4))},
		{"shape entry negative", blob(`{"a":{"dtype":"U8","shape":[-1],"data_offsets":[0,4]}}`, d(4))},
		{"data_offsets missing", blob(`{"a":{"dtype":"U8","shape":[4]}}`, d(4))},
		{"data_offsets wrong length", blob(`{"a":{"dtype":"U8","shape":[4],"data_offsets":[0]}}`, d(4))},
		{"data_offsets non-numeric", blob(`{"a":{"dtype":"U8","shape":[4],"data_offsets":["a","b"]}}`, d(4))},
	}
	for _, c := range cases {
		if _, err := Parse(c.blob); err == nil {
			t.Fatalf("%s: expected an error, got nil", c.name)
		}
	}
	t.Logf("parse field errors: all %d malformed header entries rejected", len(cases))
}

// TestParseScalarTensor parses a zero-dimensional tensor (shape []): count is the empty-product
// 1, so a 1-element dtype has a 1-byte span. Confirms the shape loop's empty case and that a
// scalar round-trips through Parse with the right (zero-rank) shape and byte.
func TestParseScalarTensor(t *testing.T) {
	ts, err := Parse(blob(`{"s":{"dtype":"U8","shape":[],"data_offsets":[0,1]}}`, []byte{42}))
	if err != nil {
		t.Fatalf("Parse scalar: %v", err)
	}
	s, ok := ts["s"]
	if !ok {
		t.Fatal("scalar tensor s missing")
	}
	if len(s.Shape) != 0 || len(s.Data) != 1 || s.Data[0] != 42 {
		t.Fatalf("scalar: shape %v data %v, want rank-0 and [42]", s.Shape, s.Data)
	}
	t.Logf("parse: rank-0 scalar (shape []) yields a 1-byte tensor")
}

// TestEncodeErrors drives Encode's rejection branches the round-trip test leaves out: a
// __metadata__ key (reserved), an unsupported dtype, and a negative shape dimension.
func TestEncodeErrors(t *testing.T) {
	cases := []struct {
		name string
		in   map[string]Tensor
	}{
		{"metadata reserved", map[string]Tensor{"__metadata__": {Dtype: "U8", Shape: []int{1}, Data: []byte{0}}}},
		{"unsupported dtype", map[string]Tensor{"a": {Dtype: "Q4", Shape: []int{1}, Data: []byte{0}}}},
		{"negative shape", map[string]Tensor{"a": {Dtype: "U8", Shape: []int{-1}, Data: nil}}},
	}
	for _, c := range cases {
		if _, err := Encode(c.in); err == nil {
			t.Fatalf("%s: expected Encode to error, got nil", c.name)
		}
	}
	t.Logf("encode errors: __metadata__, unsupported dtype, negative shape all rejected")
}

// TestEncodeNilShape covers Encode's nil-shape normalisation: a tensor with a nil Shape (a
// scalar) must encode as shape [] and Parse back to a rank-0 tensor whose bytes survive.
func TestEncodeNilShape(t *testing.T) {
	in := map[string]Tensor{"scalar": {Dtype: "F32", Shape: nil, Data: []byte{0, 0, 128, 63}}}
	b, err := Encode(in)
	if err != nil {
		t.Fatalf("Encode nil shape: %v", err)
	}
	out, err := Parse(b)
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	got, ok := out["scalar"]
	if !ok {
		t.Fatal("scalar missing after round-trip")
	}
	if len(got.Shape) != 0 {
		t.Fatalf("nil shape should encode as rank-0, got shape %v", got.Shape)
	}
	if len(got.Data) != 4 || got.Data[3] != 63 {
		t.Fatalf("scalar bytes corrupted: %v", got.Data)
	}
	t.Logf("encode: nil Shape normalised to [] and round-trips as a rank-0 tensor")
}
