// SPDX-Licence-Identifier: EUPL-1.2

package safetensors

import (
	"testing"
)

// blob builds a safetensors byte stream from a header JSON string + the data section
// (8-byte little-endian header length, the JSON, then the data).
func blob(header string, data []byte) []byte {
	h := []byte(header)
	out := make([]byte, 8+len(h)+len(data))
	n := uint64(len(h))
	for i := 0; i < 8; i++ {
		out[i] = byte(n >> (8 * uint(i)))
	}
	copy(out[8:], h)
	copy(out[8+len(h):], data)
	return out
}

// TestParse reads a well-formed two-tensor blob: the names, dtypes, shapes and the
// sub-sliced data must all come back correct, and the reserved __metadata__ key is
// skipped (not returned as a tensor).
func TestParse(t *testing.T) {
	data := make([]byte, 20)
	for i := range data {
		data[i] = byte(i + 1) // 1..20, so a sub-slice mistake is visible
	}
	header := `{"a":{"dtype":"BF16","shape":[2,3],"data_offsets":[0,12]},` +
		`"b":{"dtype":"F32","shape":[2],"data_offsets":[12,20]},` +
		`"__metadata__":{"format":"pt"}}`

	ts, err := Parse(blob(header, data))
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	if len(ts) != 2 {
		t.Fatalf("got %d tensors, want 2 (metadata must be skipped): %v", len(ts), keys(ts))
	}
	a, ok := ts["a"]
	if !ok {
		t.Fatal("tensor a missing")
	}
	if a.Dtype != "BF16" || len(a.Shape) != 2 || a.Shape[0] != 2 || a.Shape[1] != 3 {
		t.Fatalf("a: dtype/shape wrong: %s %v", a.Dtype, a.Shape)
	}
	if len(a.Data) != 12 || a.Data[0] != 1 || a.Data[11] != 12 {
		t.Fatalf("a: data sub-slice wrong: len %d, first %d last %d", len(a.Data), a.Data[0], a.Data[len(a.Data)-1])
	}
	b := ts["b"]
	if b.Dtype != "F32" || len(b.Shape) != 1 || b.Shape[0] != 2 {
		t.Fatalf("b: dtype/shape wrong: %s %v", b.Dtype, b.Shape)
	}
	if len(b.Data) != 8 || b.Data[0] != 13 || b.Data[7] != 20 {
		t.Fatalf("b: data sub-slice wrong: len %d, first %d last %d", len(b.Data), b.Data[0], b.Data[len(b.Data)-1])
	}
	t.Logf("parse: 2 tensors (BF16 [2,3] + F32 [2]) names/dtypes/shapes/data correct, __metadata__ skipped")
}

// TestParseErrors rejects malformed blobs: too short, bad header length, unknown dtype,
// a byte span that disagrees with dtype × shape, and out-of-range offsets.
func TestParseErrors(t *testing.T) {
	d := func(n int) []byte { return make([]byte, n) }
	cases := []struct {
		name string
		blob []byte
	}{
		{"too short", []byte{1, 2, 3}},
		{"header len overflows blob", blob(`xxxxxxxxxxxxxxxx`, nil)[:10]}, // claims a long header it doesn't have
		{"unknown dtype", blob(`{"a":{"dtype":"Q4","shape":[4],"data_offsets":[0,4]}}`, d(4))},
		{"span != dtype*shape", blob(`{"a":{"dtype":"F32","shape":[2],"data_offsets":[0,4]}}`, d(4))}, // F32[2]=8B, span 4
		{"offsets past blob", blob(`{"a":{"dtype":"U8","shape":[8],"data_offsets":[0,8]}}`, d(4))},    // data only 4B
		{"bad json", blob(`{not json`, d(4))},
	}
	for _, c := range cases {
		if _, err := Parse(c.blob); err == nil {
			t.Fatalf("%s: expected an error, got nil", c.name)
		}
	}
	t.Logf("parse errors: all %d malformed blobs rejected", len(cases))
}

func keys(m map[string]Tensor) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	return out
}

// TestEncodeRoundTrip checks Encode is the inverse of Parse: tensors → blob → tensors
// recovers every dtype, shape and the exact bytes.
func TestEncodeRoundTrip(t *testing.T) {
	in := map[string]Tensor{
		"w":     {Dtype: "BF16", Shape: []int{2, 3}, Data: []byte{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}},
		"norm":  {Dtype: "F32", Shape: []int{2}, Data: []byte{13, 14, 15, 16, 17, 18, 19, 20}},
		"codes": {Dtype: "U8", Shape: []int{4}, Data: []byte{21, 22, 23, 24}},
	}
	blob, err := Encode(in)
	if err != nil {
		t.Fatalf("Encode: %v", err)
	}
	out, err := Parse(blob)
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	if len(out) != len(in) {
		t.Fatalf("round-trip tensor count %d != %d", len(out), len(in))
	}
	for name, want := range in {
		got, ok := out[name]
		if !ok {
			t.Fatalf("%s missing after round-trip", name)
		}
		if got.Dtype != want.Dtype || len(got.Shape) != len(want.Shape) {
			t.Fatalf("%s dtype/shape mismatch: %s %v vs %s %v", name, got.Dtype, got.Shape, want.Dtype, want.Shape)
		}
		for i := range want.Shape {
			if got.Shape[i] != want.Shape[i] {
				t.Fatalf("%s shape[%d] %d != %d", name, i, got.Shape[i], want.Shape[i])
			}
		}
		if len(got.Data) != len(want.Data) {
			t.Fatalf("%s data len %d != %d", name, len(got.Data), len(want.Data))
		}
		for i := range want.Data {
			if got.Data[i] != want.Data[i] {
				t.Fatalf("%s data[%d] %d != %d", name, i, got.Data[i], want.Data[i])
			}
		}
	}
	if _, err := Encode(map[string]Tensor{"x": {Dtype: "BF16", Shape: []int{2}, Data: []byte{1, 2}}}); err == nil {
		t.Fatal("expected Encode to reject a byte span != dtype×shape")
	}
	t.Logf("encode: tensors → blob → tensors round-trips dtypes/shapes/bytes; bad-size rejected")
}
