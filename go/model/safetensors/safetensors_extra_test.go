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
