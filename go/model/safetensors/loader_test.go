// SPDX-Licence-Identifier: EUPL-1.2

package safetensors

import (
	"bytes"
	"testing"

	core "dappco.re/go"
)

// TestSafetensors_Encode_Golden pins the EXACT encoded byte stream (length prefix + header JSON +
// name-sorted data section) for a fixed multi-tensor input, gating the codec's buffer restructure on
// byte-identical output — the round-trip test above only proves the tensors survive, not that the
// wire bytes are unchanged.
func TestSafetensors_Encode_Golden(t *testing.T) {
	in := map[string]Tensor{
		"w":     {Dtype: "BF16", Shape: []int{2, 3}, Data: []byte{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}},
		"norm":  {Dtype: "F32", Shape: []int{2}, Data: []byte{13, 14, 15, 16, 17, 18, 19, 20}},
		"codes": {Dtype: "U8", Shape: []int{4}, Data: []byte{21, 22, 23, 24}},
	}
	got, err := Encode(in)
	if err != nil {
		t.Fatalf("Encode: %v", err)
	}
	wantBlob := []byte{172, 0, 0, 0, 0, 0, 0, 0, 123, 34, 99, 111, 100, 101, 115, 34, 58, 123, 34, 100, 116, 121, 112, 101, 34, 58, 34, 85, 56, 34, 44, 34, 115, 104, 97, 112, 101, 34, 58, 91, 52, 93, 44, 34, 100, 97, 116, 97, 95, 111, 102, 102, 115, 101, 116, 115, 34, 58, 91, 48, 44, 52, 93, 125, 44, 34, 110, 111, 114, 109, 34, 58, 123, 34, 100, 116, 121, 112, 101, 34, 58, 34, 70, 51, 50, 34, 44, 34, 115, 104, 97, 112, 101, 34, 58, 91, 50, 93, 44, 34, 100, 97, 116, 97, 95, 111, 102, 102, 115, 101, 116, 115, 34, 58, 91, 52, 44, 49, 50, 93, 125, 44, 34, 119, 34, 58, 123, 34, 100, 116, 121, 112, 101, 34, 58, 34, 66, 70, 49, 54, 34, 44, 34, 115, 104, 97, 112, 101, 34, 58, 91, 50, 44, 51, 93, 44, 34, 100, 97, 116, 97, 95, 111, 102, 102, 115, 101, 116, 115, 34, 58, 91, 49, 50, 44, 50, 52, 93, 125, 125, 21, 22, 23, 24, 13, 14, 15, 16, 17, 18, 19, 20, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}
	if !bytes.Equal(got, wantBlob) {
		t.Fatalf("Encode blob diverged: got %d bytes, want %d bytes\n got=%v\nwant=%v", len(got), len(wantBlob), got, wantBlob)
	}
}

func TestEncodeFP8E4M3RoundTrip(t *testing.T) {
	blob, err := Encode(map[string]Tensor{"weight": {Dtype: "F8_E4M3", Shape: []int{2}, Data: []byte{0x38, 0xb8}}})
	if err != nil {
		t.Fatal(err)
	}
	parsed, err := Parse(blob)
	if err != nil {
		t.Fatal(err)
	}
	if parsed["weight"].Dtype != "F8_E4M3" || len(parsed["weight"].Data) != 2 {
		t.Fatalf("FP8 tensor = %+v", parsed["weight"])
	}
}

// blob builds a safetensors byte stream from a header JSON string + the data section
// (8-byte little-endian header length, the JSON, then the data).
func blob(header string, data []byte) []byte {
	h := []byte(header)
	out := make([]byte, 8+len(h)+len(data))
	n := uint64(len(h))
	for i := range 8 {
		out[i] = byte(n >> (8 * uint(i)))
	}
	copy(out[8:], h)
	copy(out[8+len(h):], data)
	return out
}

// TestLoader_Parse_Good reads a well-formed two-tensor blob: the names, dtypes,
// shapes and the sub-sliced data must all come back correct, and the reserved
// __metadata__ key is skipped (not returned as a tensor).
func TestLoader_Parse_Good(t *testing.T) {
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

// TestLoader_Parse_Bad rejects malformed blobs: too short, bad header length,
// unknown dtype, a byte span that disagrees with dtype × shape, and out-of-range
// offsets.
func TestLoader_Parse_Bad(t *testing.T) {
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

// TestLoader_Encode_Good checks Encode is the inverse of Parse: tensors → blob →
// tensors recovers every dtype, shape and the exact bytes.
func TestLoader_Encode_Good(t *testing.T) {
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
	t.Logf("encode: tensors → blob → tensors round-trips dtypes/shapes/bytes")
}

// TestLoader_Encode_Bad rejects a tensor whose byte span disagrees with its
// declared dtype × shape — the same structural check Parse enforces on the way in.
func TestLoader_Encode_Bad(t *testing.T) {
	if _, err := Encode(map[string]Tensor{"x": {Dtype: "BF16", Shape: []int{2}, Data: []byte{1, 2}}}); err == nil {
		t.Fatal("expected Encode to reject a byte span != dtype×shape")
	}
}

// TestLoader_Encode_Ugly encodes a nil-Shape scalar (Encode must normalise it to
// an empty, non-nil shape) and confirms it round-trips through Parse as rank-0.
func TestLoader_Encode_Ugly(t *testing.T) {
	in := map[string]Tensor{"scalar": {Dtype: "F32", Shape: nil, Data: []byte{0, 0, 128, 63}}}
	blob, err := Encode(in)
	if err != nil {
		t.Fatalf("Encode(nil shape): %v", err)
	}
	out, err := Parse(blob)
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	got, ok := out["scalar"]
	if !ok {
		t.Fatal("scalar missing after round-trip")
	}
	if len(got.Shape) != 0 {
		t.Fatalf("Shape = %v, want empty (rank-0)", got.Shape)
	}
	if len(got.Data) != 4 || got.Data[3] != 63 {
		t.Fatalf("Data = %v, want [0 0 128 63]", got.Data)
	}
}

// TestLoader_Parse_Ugly parses a zero-dimensional (scalar) tensor: shape []
// makes the element count the empty-product 1, so a 1-byte dtype has a 1-byte span.
func TestLoader_Parse_Ugly(t *testing.T) {
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
}

// TestLoader_Load_Good writes a safetensors file via Encode and confirms Load
// reads it back with the same dtype, shape and bytes.
func TestLoader_Load_Good(t *testing.T) {
	dir := t.TempDir()
	path := dir + "/model.safetensors"
	in := map[string]Tensor{"weight": {Dtype: "F32", Shape: []int{2}, Data: []byte{0, 0, 128, 63, 0, 0, 0, 64}}}
	blob, err := Encode(in)
	if err != nil {
		t.Fatalf("Encode: %v", err)
	}
	if result := core.WriteFile(path, blob, 0o644); !result.OK {
		t.Fatalf("WriteFile: %v", result.Value)
	}
	got, err := Load(path)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	w, ok := got["weight"]
	if !ok {
		t.Fatal("weight missing after Load")
	}
	if w.Dtype != "F32" || len(w.Shape) != 1 || w.Shape[0] != 2 {
		t.Fatalf("weight dtype/shape = %s %v, want F32 [2]", w.Dtype, w.Shape)
	}
	if len(w.Data) != 8 || w.Data[3] != 63 {
		t.Fatalf("weight data = %v, want the encoded 8 bytes", w.Data)
	}
}

// TestLoader_Load_Bad confirms a missing file surfaces an error rather than a
// zero-value tensor map.
func TestLoader_Load_Bad(t *testing.T) {
	if _, err := Load(core.PathJoin(t.TempDir(), "missing.safetensors")); err == nil {
		t.Fatal("Load(missing file) error = nil")
	}
}

// TestLoader_Load_Ugly loads a file holding only a __metadata__ block (no real
// tensors) — the smallest legal safetensors file — and confirms it comes back empty,
// not an error.
func TestLoader_Load_Ugly(t *testing.T) {
	dir := t.TempDir()
	path := dir + "/meta_only.safetensors"
	header := []byte(`{"__metadata__":{"format":"pt"}}`)
	buf := make([]byte, 8+len(header))
	for i := range 8 {
		buf[i] = byte(len(header) >> (8 * uint(i)))
	}
	copy(buf[8:], header)
	if result := core.WriteFile(path, buf, 0o644); !result.OK {
		t.Fatalf("WriteFile: %v", result.Value)
	}
	got, err := Load(path)
	if err != nil {
		t.Fatalf("Load(metadata only): %v", err)
	}
	if len(got) != 0 {
		t.Fatalf("Load(metadata only) = %v, want empty", got)
	}
}
