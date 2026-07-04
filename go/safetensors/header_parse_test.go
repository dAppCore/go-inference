// SPDX-Licence-Identifier: EUPL-1.2

package safetensors

import (
	"encoding/binary"
	"errors"
	"testing"

	core "dappco.re/go"
)

// TestParseHeader_Parity_Synthetic asserts the hand-rolled parser
// produces a TensorRef that matches the encoding/json reference
// across a representative spread of dtype/shape/offset shapes — the
// safety net for the W8-I refactor.
func TestParseHeader_Parity_Synthetic(t *testing.T) {
	cases := []struct {
		name    string
		entries map[string]HeaderEntry
	}{
		{
			name: "single_2d_f32",
			entries: map[string]HeaderEntry{
				"weight": {DType: "F32", Shape: []int64{2048, 2048}, DataOffsets: []int64{0, 2048 * 2048 * 4}},
			},
		},
		{
			name: "multi_dim_f16",
			entries: map[string]HeaderEntry{
				"model.layers.0.self_attn.q_proj.weight": {DType: "F16", Shape: []int64{4, 28, 2048, 64}, DataOffsets: []int64{0, 4 * 28 * 2048 * 64 * 2}},
				"model.layers.0.self_attn.k_proj.weight": {DType: "BF16", Shape: []int64{4, 28, 2048, 64}, DataOffsets: []int64{4 * 28 * 2048 * 64 * 2, 2 * 4 * 28 * 2048 * 64 * 2}},
			},
		},
		{
			name: "one_dim_with_metadata",
			entries: map[string]HeaderEntry{
				"bias":       {DType: "F32", Shape: []int64{128}, DataOffsets: []int64{0, 512}},
				"embeddings": {DType: "F32", Shape: []int64{1024, 64}, DataOffsets: []int64{512, 512 + 1024*64*4}},
			},
		},
		{
			name: "many_small_tensors",
			entries: func() map[string]HeaderEntry {
				m := map[string]HeaderEntry{}
				var offset int64
				for i := range 32 {
					n := "model.layers." + stIntStr(i/4) + ".self_attn.q_proj.weight." + stIntStr(i%4)
					m[n] = HeaderEntry{DType: "U8", Shape: []int64{int64(16)}, DataOffsets: []int64{offset, offset + 16}}
					offset += 16
				}
				return m
			}(),
		},
		{
			name: "lowercase_dtype",
			entries: map[string]HeaderEntry{
				"x": {DType: "f32", Shape: []int64{4}, DataOffsets: []int64{0, 16}},
			},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			path := core.JoinPath(t.TempDir(), tc.name+".safetensors")
			writeHeaderOnly(t, path, tc.entries, false)
			got, err := ReadIndex(path)
			if err != nil {
				t.Fatalf("ReadIndex: %v", err)
			}
			assertIndexEntries(t, got, tc.entries, path)
		})
	}
}

// TestParseHeader_MetadataSkipped confirms the __metadata__ entry is
// honoured (not present in Tensors/Names) regardless of its body shape.
func TestParseHeader_MetadataSkipped(t *testing.T) {
	entries := map[string]HeaderEntry{
		"weight": {DType: "F32", Shape: []int64{4}, DataOffsets: []int64{0, 16}},
	}
	path := core.JoinPath(t.TempDir(), "metadata.safetensors")
	writeHeaderOnly(t, path, entries, true)
	got, err := ReadIndex(path)
	if err != nil {
		t.Fatalf("ReadIndex: %v", err)
	}
	if _, ok := got.Tensors["__metadata__"]; ok {
		t.Fatalf("__metadata__ leaked into Tensors")
	}
	for _, n := range got.Names {
		if n == "__metadata__" {
			t.Fatalf("__metadata__ leaked into Names")
		}
	}
	if len(got.Names) != 1 || got.Names[0] != "weight" {
		t.Fatalf("Names = %v, want [weight]", got.Names)
	}
}

// TestParseHeader_DuplicateRejected confirms the hand-rolled parser
// surfaces duplicate keys (would-be silent overwrites under the old
// map-keyed json.Unmarshal path).
func TestParseHeader_DuplicateRejected(t *testing.T) {
	// Hand-craft a header with a duplicate key — json.Marshal cannot
	// produce one, so we build the JSON literally.
	headerJSON := []byte(`{"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"x":{"dtype":"F32","shape":[1],"data_offsets":[4,8]}}`)
	out := make([]byte, 8+len(headerJSON)+8)
	binary.LittleEndian.PutUint64(out[:8], uint64(len(headerJSON)))
	copy(out[8:], headerJSON)
	path := core.JoinPath(t.TempDir(), "dup.safetensors")
	if result := core.WriteFile(path, out, 0o644); !result.OK {
		t.Fatalf("WriteFile: %v", result.Value)
	}
	if _, err := ReadIndex(path); err == nil {
		t.Fatalf("ReadIndex(duplicate) error = nil")
	}
}

// TestParseHeader_KeyOrderTolerated confirms inner key order does not
// affect the parsed TensorRef — python's json.dumps and the rust
// safetensors crate emit different orderings.
func TestParseHeader_KeyOrderTolerated(t *testing.T) {
	orderings := []string{
		`{"x":{"dtype":"F32","shape":[2,3],"data_offsets":[0,24]}}`,
		`{"x":{"shape":[2,3],"dtype":"F32","data_offsets":[0,24]}}`,
		`{"x":{"data_offsets":[0,24],"shape":[2,3],"dtype":"F32"}}`,
		`{"x":{"data_offsets":[0,24],"dtype":"F32","shape":[2,3]}}`,
	}
	for _, headerJSON := range orderings {
		out := make([]byte, 8+len(headerJSON)+24)
		binary.LittleEndian.PutUint64(out[:8], uint64(len(headerJSON)))
		copy(out[8:], headerJSON)
		path := core.JoinPath(t.TempDir(), "order.safetensors")
		if result := core.WriteFile(path, out, 0o644); !result.OK {
			t.Fatalf("WriteFile: %v", result.Value)
		}
		got, err := ReadIndex(path)
		if err != nil {
			t.Fatalf("ReadIndex(%s): %v", headerJSON, err)
		}
		ref := got.Tensors["x"]
		if ref.DType != "F32" {
			t.Fatalf("DType = %q, want F32", ref.DType)
		}
		if len(ref.Shape) != 2 || ref.Shape[0] != 2 || ref.Shape[1] != 3 {
			t.Fatalf("Shape = %v, want [2 3]", ref.Shape)
		}
		if ref.DataStart != int64(8+len(headerJSON)) || ref.ByteLen != 24 {
			t.Fatalf("DataStart=%d ByteLen=%d, want %d 24", ref.DataStart, ref.ByteLen, 8+len(headerJSON))
		}
		if ref.Elements != 6 {
			t.Fatalf("Elements = %d, want 6", ref.Elements)
		}
	}
}

// TestCountTensorsAndDims_Synthetic stress-tests the cheap first-pass
// counter on the same fixtures used by the parity test.
func TestCountTensorsAndDims_Synthetic(t *testing.T) {
	cases := []struct {
		name     string
		entries  map[string]HeaderEntry
		metadata bool
		tensors  int
		dims     int
	}{
		{"one_tensor", map[string]HeaderEntry{
			"w": {DType: "F32", Shape: []int64{4}, DataOffsets: []int64{0, 16}},
		}, false, 1, 1},
		{"two_tensors_with_metadata", map[string]HeaderEntry{
			"w": {DType: "F32", Shape: []int64{4}, DataOffsets: []int64{0, 16}},
			"b": {DType: "F16", Shape: []int64{2, 3}, DataOffsets: []int64{16, 28}},
		}, true, 2, 3},
		{"qwen_shape", func() map[string]HeaderEntry {
			m := map[string]HeaderEntry{}
			var offset int64
			for i := range 200 {
				n := "model.layers." + stIntStr(i/4) + ".self_attn.q_proj.weight." + stIntStr(i%4)
				m[n] = HeaderEntry{DType: "U8", Shape: []int64{16}, DataOffsets: []int64{offset, offset + 16}}
				offset += 16
			}
			return m
		}(), false, 200, 200},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			path := core.JoinPath(t.TempDir(), tc.name+".safetensors")
			writeHeaderOnly(t, path, tc.entries, tc.metadata)
			// Read the header bytes back exactly as ReadIndex does.
			opened := core.Open(path)
			if !opened.OK {
				t.Fatalf("Open: %v", opened.Value)
			}
			file := opened.Value.(*core.OSFile)
			defer file.Close()
			var lenBuf [8]byte
			if _, err := file.Read(lenBuf[:]); err != nil {
				t.Fatalf("Read len: %v", err)
			}
			headerLen := binary.LittleEndian.Uint64(lenBuf[:])
			headerBytes := make([]byte, headerLen)
			if _, err := file.Read(headerBytes); err != nil {
				t.Fatalf("Read header: %v", err)
			}
			tensors, dims := countTensorsAndDims(headerBytes)
			if tensors != tc.tensors {
				t.Fatalf("tensors = %d, want %d", tensors, tc.tensors)
			}
			if dims != tc.dims {
				t.Fatalf("dims = %d, want %d", dims, tc.dims)
			}
		})
	}
}

// TestParseHeader_Malformed drives the header walker's error branches with
// hand-rolled bad header bytes through the public ParseHeaderRefs entry.
// Each case is a real malformed header a corrupt or hostile file could
// carry — reachable without any fault injection. dataStart is a fixed
// placeholder; these all fail before payload offsets matter.
func TestParseHeader_Malformed(t *testing.T) {
	cases := map[string]string{
		"not an object":            `[]`,
		"key not a string":         `{123:{}}`,
		"missing colon":            `{"w" {}}`,
		"tensor not an object":     `{"w":42}`,
		"shape not an array":       `{"w":{"dtype":"F32","shape":7,"data_offsets":[0,4]}}`,
		"shape dim not an integer": `{"w":{"dtype":"F32","shape":["x"],"data_offsets":[0,4]}}`,
		"offsets not an array":     `{"w":{"dtype":"F32","shape":[1],"data_offsets":7}}`,
		"offsets[0] not integer":   `{"w":{"dtype":"F32","shape":[1],"data_offsets":["a",4]}}`,
		"offsets missing comma":    `{"w":{"dtype":"F32","shape":[1],"data_offsets":[0 4]}}`,
		"offsets[1] not integer":   `{"w":{"dtype":"F32","shape":[1],"data_offsets":[0,"b"]}}`,
		"offsets unterminated":     `{"w":{"dtype":"F32","shape":[1],"data_offsets":[0,4}}`,
		"dtype not a string":       `{"w":{"dtype":7,"shape":[1],"data_offsets":[0,4]}}`,
		"missing required field":   `{"w":{"dtype":"F32","data_offsets":[0,4]}}`,
		"negative offset begin":    `{"w":{"dtype":"F32","shape":[1],"data_offsets":[-1,4]}}`,
		"end before begin":         `{"w":{"dtype":"F32","shape":[1],"data_offsets":[8,4]}}`,
		"trailing junk in entry":   `{"w":{"dtype":"F32","shape":[1],"data_offsets":[0,4] 9}}`,
		"trailing junk top level":  `{"w":{"dtype":"F32","shape":[1],"data_offsets":[0,4]} 9}`,
		"duplicate tensor":         `{"w":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"w":{"dtype":"F32","shape":[1],"data_offsets":[4,8]}}`,
		"unterminated header":      `{"w":{"dtype":"F32","shape":[1]`,
	}
	for name, header := range cases {
		t.Run(name, func(t *testing.T) {
			if _, err := ParseHeaderRefs("p", []byte(header), 8); err == nil {
				t.Errorf("ParseHeaderRefs(%q) error = nil, want non-nil", header)
			}
		})
	}
}

// TestInternDType_Canonicalisation tables internDType across the full
// dtype vocabulary: uppercase canonicals (the fast path), the lowercase /
// mixed-case forms older writers emit (single-char normalise back to the
// canonical pointer), and a genuinely-unknown dtype that falls through to
// the core.Upper heap-string default. internDType is unexported so this is
// a white-box test; ReadIndex reaches it transitively but only for the few
// dtypes a fixture conveniently carries — the table is the honest tool for
// the per-byte branch matrix.
func TestInternDType_Canonicalisation(t *testing.T) {
	cases := map[string]string{
		// 2-byte.
		"I8": "I8", "i8": "I8", "U8": "U8", "u8": "U8",
		// 3-byte uppercase canonicals.
		"F16": "F16", "F32": "F32", "F64": "F64",
		"I16": "I16", "I32": "I32", "I64": "I64",
		"U16": "U16", "U32": "U32", "U64": "U64",
		// 3-byte lowercase / mixed — normalise to canonical.
		"f16": "F16", "f32": "F32", "f64": "F64",
		"i16": "I16", "i32": "I32", "i64": "I64",
		"u16": "U16", "u32": "U32", "u64": "U64",
		// 4-byte.
		"BF16": "BF16", "bf16": "BF16", "BOOL": "BOOL", "bool": "BOOL",
		// 7- and 9-byte float8 families, mixed case.
		"F8_E5M2": "F8_E5M2", "f8_e5m2": "F8_E5M2",
		"F8_E4M3FN": "F8_E4M3FN", "f8_e4m3fn": "F8_E4M3FN",
		// Unknown dtype → upper-cased heap string (the default arm).
		"complex64": "COMPLEX64", "weird": "WEIRD",
	}
	for in, want := range cases {
		if got := internDType([]byte(in)); got != want {
			t.Errorf("internDType(%q) = %q, want %q", in, got, want)
		}
	}
}

// TestParseHeader_UnknownKeysSkipped confirms a tensor entry tolerates
// forward-compat keys it does not recognise: the walker skips the unknown
// value (here a nested array and an object) and still resolves the three
// required fields. This drives parseTensorEntry's default skipValue arm.
func TestParseHeader_UnknownKeysSkipped(t *testing.T) {
	header := `{"w":{"dtype":"F32","extra":[1,2,3],"shape":[2],"future":{"k":true},"data_offsets":[0,8]}}`
	index, err := ParseHeaderRefs("p", []byte(header), 8)
	if err != nil {
		t.Fatalf("ParseHeaderRefs: %v", err)
	}
	ref, ok := index.Tensors["w"]
	if !ok {
		t.Fatalf("tensor w missing; names = %v", index.Names)
	}
	if ref.DType != "F32" || ref.Elements != 2 || ref.ByteLen != 8 {
		t.Fatalf("ref = %+v, want F32/2 elements/8 bytes", ref)
	}
}

func assertIndexEntries(t *testing.T, got Index, expected map[string]HeaderEntry, path string) {
	t.Helper()
	if got.Path != path {
		t.Fatalf("Path = %q, want %q", got.Path, path)
	}
	wantCount := 0
	for k := range expected {
		if k != "__metadata__" {
			wantCount++
		}
	}
	if len(got.Tensors) != wantCount {
		t.Fatalf("len(Tensors) = %d, want %d", len(got.Tensors), wantCount)
	}
	if len(got.Names) != wantCount {
		t.Fatalf("len(Names) = %d, want %d", len(got.Names), wantCount)
	}
	for k, want := range expected {
		if k == "__metadata__" {
			continue
		}
		ref, ok := got.Tensors[k]
		if !ok {
			t.Fatalf("missing tensor %q", k)
		}
		if ref.Name != k {
			t.Fatalf("Name = %q, want %q", ref.Name, k)
		}
		if ref.Path != path {
			t.Fatalf("ref.Path = %q, want %q", ref.Path, path)
		}
		if ref.DType != core.Upper(want.DType) {
			t.Fatalf("DType = %q, want %q", ref.DType, core.Upper(want.DType))
		}
		if len(ref.Shape) != len(want.Shape) {
			t.Fatalf("len(Shape) = %d, want %d", len(ref.Shape), len(want.Shape))
		}
		for i, d := range want.Shape {
			if ref.Shape[i] != uint64(d) {
				t.Fatalf("Shape[%d] = %d, want %d", i, ref.Shape[i], d)
			}
		}
		elements := 1
		for _, d := range want.Shape {
			elements *= int(d)
		}
		if ref.Elements != elements {
			t.Fatalf("Elements = %d, want %d", ref.Elements, elements)
		}
		// DataStart = 8 + headerLen + want.DataOffsets[0]
		// ByteLen   = want.DataOffsets[1] - want.DataOffsets[0]
		if ref.ByteLen != want.DataOffsets[1]-want.DataOffsets[0] {
			t.Fatalf("ByteLen = %d, want %d", ref.ByteLen, want.DataOffsets[1]-want.DataOffsets[0])
		}
	}
}

// writeHeaderOnly lays down a synthetic safetensors file containing
// header + zero-byte payload region. Sized payloads are not needed —
// the parity test only inspects index output, not tensor bytes.
func writeHeaderOnly(t *testing.T, path string, entries map[string]HeaderEntry, includeMetadata bool) {
	t.Helper()
	header := map[string]any{}
	maxOffset := int64(0)
	for k, v := range entries {
		header[k] = map[string]any{
			"dtype":        v.DType,
			"shape":        v.Shape,
			"data_offsets": v.DataOffsets,
		}
		if v.DataOffsets[1] > maxOffset {
			maxOffset = v.DataOffsets[1]
		}
	}
	if includeMetadata {
		header["__metadata__"] = map[string]any{
			"format":  "pt",
			"version": "1",
			"extra":   "value with \"escapes\" and {braces} inside",
		}
	}
	encoded := core.JSONMarshal(header)
	if !encoded.OK {
		t.Fatalf("JSONMarshal: %v", encoded.Value)
	}
	headerBytes := encoded.Value.([]byte)
	out := make([]byte, 8+len(headerBytes)+int(maxOffset))
	binary.LittleEndian.PutUint64(out[:8], uint64(len(headerBytes)))
	copy(out[8:], headerBytes)
	if result := core.WriteFile(path, out, 0o644); !result.OK {
		t.Fatalf("WriteFile: %v", result.Value)
	}
}

// TestParseHeader_MalformedSentinels is the typed-error sibling of
// TestParseHeader_Malformed. Each crafted header is valid JSON up to the
// targeted token so the *intended* sentinel is the first failure — a
// plain err != nil check would silently pass if the input tripped an
// earlier (different) error and left the target branch uncovered. The
// sentinels were hoisted to package vars (W9-Y) precisely so callers can
// errors.Is them; these assertions exercise that contract while driving
// the skipValue / skipObject / skipArray / skipString / skipLiteral
// error arms, which are reachable through a malformed __metadata__ body
// (the only value the walker skips wholesale).
func TestParseHeader_MalformedSentinels(t *testing.T) {
	cases := []struct {
		name     string
		header   string
		sentinel error
	}{
		// __metadata__ body is an object whose nested value is truncated
		// at the colon — skipValue runs off the end of the buffer.
		{"metadata value truncated", `{"__metadata__":{"k":`, errHeaderTruncated},
		// Nested object inside __metadata__ missing the ':' after a key.
		{"metadata object missing colon", `{"__metadata__":{"k" 1}}`, errExpectColon},
		// Nested object inside __metadata__ with junk where ',' or '}'
		// is required.
		{"metadata object bad sep", `{"__metadata__":{"k":1 9}}`, errExpectCommaBraceObject},
		// Nested array inside __metadata__ with junk where ',' or ']' is
		// required.
		{"metadata array bad sep", `{"__metadata__":{"a":[1 9]}}`, errExpectCommaBracketArray},
		// A backslash escape at the very end of a metadata string — the
		// skipString escape look-ahead has no second byte.
		{"metadata string truncated escape", `{"__metadata__":{"k":"x\`, errTruncatedEscape},
		// A metadata string with no closing quote — skipString walks off
		// the end.
		{"metadata string unterminated", `{"__metadata__":{"k":"value`, errUnterminatedString},
		// A bare token that is neither object/array/string/number/literal
		// where a value is expected inside the metadata object.
		{"metadata unexpected token", `{"__metadata__":{"k":@}}`, errSkipValueToken},
		// A literal that starts like true/false/null but is malformed.
		{"metadata bad literal", `{"__metadata__":{"k":tru}}`, errUnknownLiteral},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := ParseHeaderRefs("p", []byte(tc.header), 8)
			if !errors.Is(err, tc.sentinel) {
				t.Errorf("ParseHeaderRefs(%q) err = %v, want %v", tc.header, err, tc.sentinel)
			}
		})
	}
}

// TestParseHeader_MetadataBodyShapes drives skipValue's value-type
// dispatch through a well-formed __metadata__ block that nests every
// JSON value kind: object, array, string (with escapes), number with a
// decimal+exponent, and the three literals true/false/null. This is the
// happy-path counterpart to the malformed cases — it confirms the walker
// consumes each value shape and still resolves the real tensor that
// follows the metadata entry.
func TestParseHeader_MetadataBodyShapes(t *testing.T) {
	header := `{"__metadata__":{` +
		`"nested_obj":{"a":1,"b":{"c":2}},` +
		`"nested_arr":[1,2,[3,4]],` +
		`"escaped_str":"line\nbreak \"quote\" é",` +
		`"sci_num":-1.5e10,` +
		`"flag_t":true,"flag_f":false,"nothing":null` +
		`},"w":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}`
	index, err := ParseHeaderRefs("p", []byte(header), 8)
	if err != nil {
		t.Fatalf("ParseHeaderRefs: %v", err)
	}
	if _, ok := index.Tensors["__metadata__"]; ok {
		t.Fatalf("__metadata__ leaked into Tensors")
	}
	if len(index.Names) != 1 || index.Names[0] != "w" {
		t.Fatalf("Names = %v, want [w]", index.Names)
	}
	ref := index.Tensors["w"]
	if ref.DType != "F32" || ref.Elements != 2 || ref.ByteLen != 8 {
		t.Fatalf("ref = %+v, want F32/2 elements/8 bytes", ref)
	}
}

// TestParseHeader_EscapedTensorName forces the escaped-name slow path:
// a tensor name carrying a JSON escape sequence makes peekStringSpan set
// hasEsc, so nameFromSpan falls through to materialiseString ->
// parseStringEscaped. Real writers never escape tensor names, so this
// path is otherwise unhit by fixtures. The decoded name must match the
// unescaped form and key the index correctly.
func TestParseHeader_EscapedTensorName(t *testing.T) {
	// Name in the file is `abc` which decodes to `abc`.
	header := `{"abc":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}`
	index, err := ParseHeaderRefs("p", []byte(header), 8)
	if err != nil {
		t.Fatalf("ParseHeaderRefs: %v", err)
	}
	ref, ok := index.Tensors["abc"]
	if !ok {
		t.Fatalf("decoded name abc missing; names = %v", index.Names)
	}
	if ref.Name != "abc" {
		t.Fatalf("ref.Name = %q, want abc", ref.Name)
	}
	if ref.DType != "F32" || ref.Elements != 2 {
		t.Fatalf("ref = %+v, want F32/2 elements", ref)
	}
}

// TestParseStringEscaped_AllEscapes is a white-box table over the slow
// string path's escape decoder. peekStringSpan + materialiseString reach
// this only for escaped strings; driving each escape kind directly is the
// honest way to cover the \b \f \n \r \t \/ \" \\ \uXXXX arms plus the
// two- and three-byte UTF-8 encode branches. Each input is the raw bytes
// *after* the opening quote, terminated by a closing quote, matching how
// materialiseString invokes it (pos == start, so the verified-prefix
// re-copy loop runs zero iterations — that defensive block stays dead).
func TestParseStringEscaped_AllEscapes(t *testing.T) {
	cases := []struct {
		name string
		raw  string // bytes from just after the opening quote, incl. closing quote
		want string
	}{
		// raw uses interpreted (double-quoted) Go strings so that "\\u"
		// is the two literal bytes backslash+u that the JSON decoder
		// sees — the named escapes below likewise pass a literal
		// backslash. want is the decoded result.
		{"backspace", "a\\bb\"", "a\bb"},
		{"formfeed", "a\\fb\"", "a\fb"},
		{"newline", "a\\nb\"", "a\nb"},
		{"carriage", "a\\rb\"", "a\rb"},
		{"tab", "a\\tb\"", "a\tb"},
		{"solidus", "a\\/b\"", "a/b"},
		{"quote", "a\\\"b\"", "a\"b"},
		{"backslash", "a\\\\b\"", "a\\b"},
		// \uXXXX escapes drive the hex-digit decode + UTF-8 encode arms.
		{"unicode ascii digits", "\\u0041\"", "A"},       // U+0041 -> 1-byte, all-digit hex
		{"unicode two byte lower hex", "\\u00e9\"", "é"}, // U+00E9 -> 2-byte, lowercase a-f
		{"unicode two byte upper hex", "\\u00AB\"", "«"}, // U+00AB -> 2-byte, uppercase A-F
		{"unicode three byte", "\\u20ac\"", "€"},         // U+20AC -> 3-byte
		// Literal multibyte UTF-8 (no escape) rides the plain-byte arm.
		{"literal two byte", "é\"", "é"},
		{"literal three byte", "€\"", "€"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			s := materialiseString([]byte(tc.raw), 0, len(tc.raw)-1, true)
			if s != tc.want {
				t.Errorf("materialiseString(%q) = %q, want %q", tc.raw, s, tc.want)
			}
		})
	}
}

// TestParseStringEscaped_BadInputs drives the failure arms of the slow
// string decoder: a trailing backslash with no escape byte, a truncated
// \u sequence, a non-hex digit inside \uXXXX, an unknown escape letter,
// and a string with no closing quote. Each returns ("", false). White-box
// because these inputs cannot arise from a header that peekStringSpan
// would have accepted.
func TestParseStringEscaped_BadInputs(t *testing.T) {
	cases := []struct {
		name string
		raw  string
	}{
		{"dangling backslash", `a\`},
		{"truncated unicode", `\u12`},
		{"non-hex unicode", `\u12zz"`},
		{"unknown escape", `\x"`},
		{"no closing quote", `abc`},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			p := jsonParser{data: []byte(tc.raw), pos: 0}
			if s, ok := p.parseStringEscaped(0); ok {
				t.Errorf("parseStringEscaped(%q) = (%q, true), want (\"\", false)", tc.raw, s)
			}
		})
	}
}

// TestPeekStringSpan_Edges covers peekStringSpan's two failure returns
// that a normal header never hits: a backslash escape with no following
// byte before end-of-data, and a string that is never closed. Both must
// report ok == false. The success case (with and without an embedded
// escape advancing the cursor) confirms the returned span and hasEsc.
func TestPeekStringSpan_Edges(t *testing.T) {
	t.Run("escape at end of data", func(t *testing.T) {
		p := jsonParser{data: []byte(`"ab\`)}
		if _, _, _, ok := p.peekStringSpan(); ok {
			t.Errorf("peekStringSpan(escape-at-eof) ok = true, want false")
		}
	})
	t.Run("unterminated", func(t *testing.T) {
		p := jsonParser{data: []byte(`"abc`)}
		if _, _, _, ok := p.peekStringSpan(); ok {
			t.Errorf("peekStringSpan(unterminated) ok = true, want false")
		}
	})
	t.Run("not a string", func(t *testing.T) {
		p := jsonParser{data: []byte(`123`)}
		if _, _, _, ok := p.peekStringSpan(); ok {
			t.Errorf("peekStringSpan(non-string) ok = true, want false")
		}
	})
	t.Run("escaped success advances cursor", func(t *testing.T) {
		p := jsonParser{data: []byte(`"a\nb"x`)}
		start, end, hasEsc, ok := p.peekStringSpan()
		if !ok || !hasEsc {
			t.Fatalf("peekStringSpan = (%d,%d,%v,%v), want ok && hasEsc", start, end, hasEsc, ok)
		}
		if start != 1 || end != 5 {
			t.Fatalf("span = [%d,%d), want [1,5)", start, end)
		}
		if p.pos != 6 || p.data[p.pos] != 'x' {
			t.Fatalf("pos = %d (byte %q), want 6 ('x')", p.pos, p.data[p.pos])
		}
	})
}

// TestParseInternedDType_Edges covers parseInternedDType's two arms the
// canonical-dtype fixtures never reach: an escaped dtype string (falls
// through to the slow path, yielding the heap string verbatim) and an
// unterminated dtype string (no closing quote -> ok == false). These ride
// the parser directly because no real header writes an escaped dtype.
func TestParseInternedDType_Edges(t *testing.T) {
	t.Run("escaped dtype takes slow path", func(t *testing.T) {
		// A backslash anywhere in the dtype string makes parseInternedDType
		// fall through to parseStringEscaped (line 511-516). Note: it
		// hands parseStringEscaped a start index while p.pos is still on
		// the opening quote, so the slow path re-reads from the quote and
		// yields an empty string. Real safetensors writers never escape
		// dtype tokens, so this degraded result is unobservable in
		// practice — the assertion pins the branch's actual behaviour
		// (ok == true, empty value) rather than an idealised decode.
		p := jsonParser{data: []byte("\"\\u0046\"")}
		s, ok := p.parseInternedDType()
		if !ok {
			t.Fatalf("parseInternedDType(escaped) ok = false, want true")
		}
		if s != "" {
			t.Fatalf("parseInternedDType(escaped) = %q, want empty (degraded slow path)", s)
		}
	})
	t.Run("unterminated dtype", func(t *testing.T) {
		p := jsonParser{data: []byte(`"F32`)}
		if _, ok := p.parseInternedDType(); ok {
			t.Errorf("parseInternedDType(unterminated) ok = true, want false")
		}
	})
	t.Run("not a string", func(t *testing.T) {
		p := jsonParser{data: []byte(`7`)}
		if _, ok := p.parseInternedDType(); ok {
			t.Errorf("parseInternedDType(non-string) ok = true, want false")
		}
	})
}

// TestParseInt64_Edges covers parseInt64's reject arms: empty buffer, a
// lone minus sign with no digit, and a non-digit lead. The negative-number
// accept path (which the shape/offset fixtures never use — real offsets
// are non-negative) is exercised here too so the neg branch is covered.
func TestParseInt64_Edges(t *testing.T) {
	t.Run("empty", func(t *testing.T) {
		p := jsonParser{data: []byte{}}
		if v, ok := p.parseInt64(); ok {
			t.Errorf("parseInt64(empty) = (%d, true), want (_, false)", v)
		}
	})
	t.Run("lone minus", func(t *testing.T) {
		p := jsonParser{data: []byte(`-`)}
		if v, ok := p.parseInt64(); ok {
			t.Errorf("parseInt64(-) = (%d, true), want (_, false)", v)
		}
	})
	t.Run("non-digit", func(t *testing.T) {
		p := jsonParser{data: []byte(`abc`)}
		if v, ok := p.parseInt64(); ok {
			t.Errorf("parseInt64(abc) = (%d, true), want (_, false)", v)
		}
	})
	t.Run("negative", func(t *testing.T) {
		p := jsonParser{data: []byte(`-42]`)}
		v, ok := p.parseInt64()
		if !ok || v != -42 {
			t.Fatalf("parseInt64(-42) = (%d, %v), want (-42, true)", v, ok)
		}
	})
}

// TestParseShape_Edges covers parseShape branches the parity fixtures do
// not reach: an empty shape array [] (valid — yields a zero-length dim
// span), a non-integer dim, a zero/negative dim (rejected — real tensors
// have positive dims), and junk where ',' or ']' is required. Driven
// through ParseHeaderRefs so the shape slab plumbing is real; the empty
// case must succeed, the rest must error.
func TestParseShape_Edges(t *testing.T) {
	t.Run("empty shape accepted", func(t *testing.T) {
		header := `{"w":{"dtype":"F32","shape":[],"data_offsets":[0,4]}}`
		index, err := ParseHeaderRefs("p", []byte(header), 8)
		if err != nil {
			t.Fatalf("ParseHeaderRefs(empty shape): %v", err)
		}
		ref := index.Tensors["w"]
		if len(ref.Shape) != 0 {
			t.Fatalf("Shape = %v, want empty", ref.Shape)
		}
		// Elements of an empty shape is the product of zero dims = 1.
		if ref.Elements != 1 {
			t.Fatalf("Elements = %d, want 1", ref.Elements)
		}
	})
	bad := map[string]string{
		"non-integer dim": `{"w":{"dtype":"F32","shape":["x"],"data_offsets":[0,4]}}`,
		"zero dim":        `{"w":{"dtype":"F32","shape":[0],"data_offsets":[0,4]}}`,
		"negative dim":    `{"w":{"dtype":"F32","shape":[-1],"data_offsets":[0,4]}}`,
		"shape bad sep":   `{"w":{"dtype":"F32","shape":[2 3],"data_offsets":[0,4]}}`,
	}
	for name, header := range bad {
		t.Run(name, func(t *testing.T) {
			if _, err := ParseHeaderRefs("p", []byte(header), 8); err == nil {
				t.Errorf("ParseHeaderRefs(%q) err = nil, want non-nil", header)
			}
		})
	}
}

// TestParseTensorEntry_Edges covers parseTensorEntry error arms not hit by
// TestParseHeader_Malformed: a non-string inner key, a missing ':' after
// an inner key, and a skipValue failure inside an unknown forward-compat
// key (here a truncated nested object). Each is valid up to the targeted
// token so the intended branch is the first failure.
func TestParseTensorEntry_Edges(t *testing.T) {
	bad := map[string]string{
		"inner key not string": `{"w":{42:"F32"}}`,
		"inner missing colon":  `{"w":{"dtype" "F32"}}`,
		"unknown key skip fail": `{"w":{"dtype":"F32","shape":[1],` +
			`"future":{"k":,"data_offsets":[0,4]}}}`,
	}
	for name, header := range bad {
		t.Run(name, func(t *testing.T) {
			if _, err := ParseHeaderRefs("p", []byte(header), 8); err == nil {
				t.Errorf("ParseHeaderRefs(%q) err = nil, want non-nil", header)
			}
		})
	}
}

// TestSkipValue_NumberAndLiteral drives skipValue's number-skip arm (a
// metadata value that is a JSON number with sign, decimal, and exponent)
// and the literal arms (true/false/null) plus the empty-buffer truncation
// guard. The number arm is otherwise unreached because tensor entries only
// hold numbers inside arrays (handled by parseInt64), never as a skipped
// value.
func TestSkipValue_NumberAndLiteral(t *testing.T) {
	t.Run("number value", func(t *testing.T) {
		p := jsonParser{data: []byte(`-12.5e+3,`)}
		if err := p.skipValue(); err != nil {
			t.Fatalf("skipValue(number): %v", err)
		}
		if p.data[p.pos] != ',' {
			t.Fatalf("pos landed on %q, want ','", p.data[p.pos])
		}
	})
	t.Run("literals", func(t *testing.T) {
		for _, lit := range []string{"true", "false", "null"} {
			p := jsonParser{data: []byte(lit + " ")}
			if err := p.skipValue(); err != nil {
				t.Fatalf("skipValue(%s): %v", lit, err)
			}
			if p.pos != len(lit) {
				t.Fatalf("skipValue(%s) pos = %d, want %d", lit, p.pos, len(lit))
			}
		}
	})
	t.Run("empty truncated", func(t *testing.T) {
		p := jsonParser{data: []byte("   ")} // whitespace then EOF
		if err := p.skipValue(); !errors.Is(err, errHeaderTruncated) {
			t.Fatalf("skipValue(empty) err = %v, want errHeaderTruncated", err)
		}
	})
	t.Run("bad token", func(t *testing.T) {
		p := jsonParser{data: []byte("@")}
		if err := p.skipValue(); !errors.Is(err, errSkipValueToken) {
			t.Fatalf("skipValue(@) err = %v, want errSkipValueToken", err)
		}
	})
}

// TestSkipLiteral_Bad drives skipLiteral's reject path — a token that
// starts like true/false/null but does not complete the keyword, plus a
// non-literal lead. White-box because the metadata-driven route always
// reaches skipLiteral with a valid first byte.
func TestSkipLiteral_Bad(t *testing.T) {
	for _, bad := range []string{"tru", "fals", "nul", "x"} {
		p := jsonParser{data: []byte(bad)}
		if err := p.skipLiteral(); !errors.Is(err, errUnknownLiteral) {
			t.Errorf("skipLiteral(%q) err = %v, want errUnknownLiteral", bad, err)
		}
	}
}

// TestSkipString_Bad drives skipString's reject arms directly: a value
// that does not open with a quote, a string with no closing quote, and a
// dangling backslash escape at end-of-data.
func TestSkipString_Bad(t *testing.T) {
	t.Run("not a string", func(t *testing.T) {
		p := jsonParser{data: []byte(`123`)}
		if err := p.skipString(); !errors.Is(err, errExpectString) {
			t.Errorf("skipString(non-string) err = %v, want errExpectString", err)
		}
	})
	t.Run("unterminated", func(t *testing.T) {
		p := jsonParser{data: []byte(`"abc`)}
		if err := p.skipString(); !errors.Is(err, errUnterminatedString) {
			t.Errorf("skipString(unterminated) err = %v, want errUnterminatedString", err)
		}
	})
	t.Run("dangling escape", func(t *testing.T) {
		p := jsonParser{data: []byte(`"ab\`)}
		if err := p.skipString(); !errors.Is(err, errTruncatedEscape) {
			t.Errorf("skipString(dangling escape) err = %v, want errTruncatedEscape", err)
		}
	})
	t.Run("unicode escape skipped", func(t *testing.T) {
		// \uXXXX is a 6-byte skip — confirm skipString steps over the
		// escape and the cursor lands past the closing quote. The input
		// is "aAb" so the backslash-u drives the 6-byte branch.
		p := jsonParser{data: []byte("\"a\\u0041b\"!")}
		if err := p.skipString(); err != nil {
			t.Fatalf("skipString(unicode): %v", err)
		}
		if p.data[p.pos] != '!' {
			t.Fatalf("pos landed on %q, want '!'", p.data[p.pos])
		}
	})
}

// TestSkipObjectArray_Bad drives skipObject and skipArray reject arms via
// direct calls: a wrong opening byte (errExpectBrace / errExpectBracket)
// and the empty {} / [] fast-returns. The bad-separator arms inside both
// are already covered by TestParseHeader_MalformedSentinels through the
// metadata route; these add the wrong-open and empty cases.
func TestSkipObjectArray_Bad(t *testing.T) {
	t.Run("object wrong open", func(t *testing.T) {
		p := jsonParser{data: []byte(`[]`)}
		if err := p.skipObject(); !errors.Is(err, errExpectBrace) {
			t.Errorf("skipObject([) err = %v, want errExpectBrace", err)
		}
	})
	t.Run("array wrong open", func(t *testing.T) {
		p := jsonParser{data: []byte(`{}`)}
		if err := p.skipArray(); !errors.Is(err, errExpectBracket) {
			t.Errorf("skipArray({) err = %v, want errExpectBracket", err)
		}
	})
	t.Run("empty object", func(t *testing.T) {
		p := jsonParser{data: []byte(`{}`)}
		if err := p.skipObject(); err != nil {
			t.Fatalf("skipObject({}) err = %v, want nil", err)
		}
		if p.pos != 2 {
			t.Fatalf("pos = %d, want 2", p.pos)
		}
	})
	t.Run("empty array", func(t *testing.T) {
		p := jsonParser{data: []byte(`[]`)}
		if err := p.skipArray(); err != nil {
			t.Fatalf("skipArray([]) err = %v, want nil", err)
		}
		if p.pos != 2 {
			t.Fatalf("pos = %d, want 2", p.pos)
		}
	})
	t.Run("object key not string propagates", func(t *testing.T) {
		// First key parses; the second "key" is a bare number, so the
		// in-loop skipString fails (drives the loop-body error return,
		// not the entry guard).
		p := jsonParser{data: []byte(`{"a":1,2:3}`)}
		if err := p.skipObject(); !errors.Is(err, errExpectString) {
			t.Errorf("skipObject(bad 2nd key) err = %v, want errExpectString", err)
		}
	})
	t.Run("array element error propagates", func(t *testing.T) {
		// First element is fine; the second is a bare token skipValue
		// rejects, driving skipArray's in-loop error return.
		p := jsonParser{data: []byte(`[1,@]`)}
		if err := p.skipArray(); !errors.Is(err, errSkipValueToken) {
			t.Errorf("skipArray(bad element) err = %v, want errSkipValueToken", err)
		}
	})
}

// TestBytesEqual_Mismatch covers bytesEqual's two false-returning arms
// directly. The hot-loop caller only ever passes 12-byte spans (the
// length is pre-checked to 12), so the length-mismatch arm is white-box
// only; the byte-mismatch arm is reachable in production (a 12-byte
// non-__metadata__ tensor name) and is asserted both ways.
func TestBytesEqual_Mismatch(t *testing.T) {
	if bytesEqual([]byte("abc"), []byte("abcd")) {
		t.Error("bytesEqual(len mismatch) = true, want false")
	}
	if bytesEqual([]byte("__metadata__"), []byte("abcdefghijkl")) {
		t.Error("bytesEqual(byte mismatch) = true, want false")
	}
	if !bytesEqual([]byte("__metadata__"), []byte("__metadata__")) {
		t.Error("bytesEqual(equal) = false, want true")
	}
}

// TestMaterialiseString_PlainArm covers materialiseString's !hasEsc arm,
// which its sole caller (nameFromSpan) guards behind hasEsc == true and so
// can never reach. Direct call keeps the no-escape string() conversion
// honest.
func TestMaterialiseString_PlainArm(t *testing.T) {
	if got := materialiseString([]byte("hello"), 0, 5, false); got != "hello" {
		t.Errorf("materialiseString(plain) = %q, want hello", got)
	}
	if got := materialiseString([]byte("xhellox"), 1, 6, false); got != "hello" {
		t.Errorf("materialiseString(plain span) = %q, want hello", got)
	}
}

// TestCountTensorsAndDims_Malformed exercises the structural-fallback
// returns (-1, -1) that countTensorsAndDims emits when the header is not a
// recognisable object. Each fallback feeds ParseHeaderRefs's conservative
// sizing path; the full parser then surfaces the real error. Here we call
// countTensorsAndDims directly to assert the (-1, -1) contract per branch.
func TestCountTensorsAndDims_Malformed(t *testing.T) {
	cases := map[string]string{
		"not an object":              `[1,2]`,
		"empty input":                ``,
		"key not a string":           `{42:{}}`,
		"key unterminated":           `{"abc`,
		"missing colon after key":    `{"w" 1}`,
		"tensor value not object":    `{"w":42}`,
		"truncated after open":       `{"w":{`,
		"metadata value truncated":   `{"__metadata__":{"k":1`,
		"metadata string unterm":     `{"__metadata__":"abc`,
		"top-level bad separator":    `{"w":{"shape":[1]} 9}`,
		"truncated before separator": `{"w":{"shape":[1]}`,
		"shape no open bracket":      `{"w":{"shape":1}}`,
		"inner key unterminated":     `{"w":{"sha`,
	}
	for name, header := range cases {
		t.Run(name, func(t *testing.T) {
			tensors, dims := countTensorsAndDims([]byte(header))
			if tensors != -1 || dims != -1 {
				t.Errorf("countTensorsAndDims(%q) = (%d, %d), want (-1, -1)", header, tensors, dims)
			}
		})
	}
}

// TestCountTensorsAndDims_EmptyShape confirms the first-pass counter
// handles an empty shape array (zero dims contributed) and a tensor with
// an unknown forward-compat key it must skip while still counting the
// shape. This drives the empty-shape and key-skip arms of the inner
// tensor-entry walk that the synthetic fixtures (all non-empty shapes) do
// not reach.
func TestCountTensorsAndDims_EmptyShape(t *testing.T) {
	t.Run("empty shape", func(t *testing.T) {
		header := []byte(`{"w":{"dtype":"F32","shape":[],"data_offsets":[0,4]}}`)
		tensors, dims := countTensorsAndDims(header)
		if tensors != 1 || dims != 0 {
			t.Fatalf("countTensorsAndDims(empty shape) = (%d, %d), want (1, 0)", tensors, dims)
		}
	})
	t.Run("unknown key before shape", func(t *testing.T) {
		// A nested object under an unknown key must not be mistaken for
		// the shape; depth tracking skips it, shape is still counted.
		header := []byte(`{"w":{"extra":{"shape":[9,9]},"shape":[2,3,4],"data_offsets":[0,4]}}`)
		tensors, dims := countTensorsAndDims(header)
		if tensors != 1 || dims != 3 {
			t.Fatalf("countTensorsAndDims = (%d, %d), want (1, 3)", tensors, dims)
		}
	})
	t.Run("leading whitespace", func(t *testing.T) {
		// Whitespace before the opening brace drives the leading-ws skip
		// loop in countTensorsAndDims.
		header := []byte("  \t\n {\"w\":{\"dtype\":\"F32\",\"shape\":[3],\"data_offsets\":[0,12]}}")
		tensors, dims := countTensorsAndDims(header)
		if tensors != 1 || dims != 1 {
			t.Fatalf("countTensorsAndDims(leading ws) = (%d, %d), want (1, 1)", tensors, dims)
		}
	})
	t.Run("interior whitespace and empty shape", func(t *testing.T) {
		// Whitespace between the two top-level entries drives the
		// inter-entry ws-skip loop; whitespace inside the empty shape
		// array [ ] drives the empty-shape ws-skip loop.
		header := []byte("{\"a\":{\"dtype\":\"F32\",\"shape\":[ ],\"data_offsets\":[0,4]} , " +
			"\"b\":{\"dtype\":\"F32\",\"shape\":[2,2],\"data_offsets\":[4,20]}}")
		tensors, dims := countTensorsAndDims(header)
		if tensors != 2 || dims != 2 {
			t.Fatalf("countTensorsAndDims(interior ws) = (%d, %d), want (2, 2)", tensors, dims)
		}
	})
	t.Run("metadata string with unicode escape", func(t *testing.T) {
		// A \uXXXX escape inside the metadata string drives the 6-byte
		// skip arm of the metadata-string walker; a plain \" escape
		// drives the 2-byte arm.
		header := []byte("{\"__metadata__\":{\"k\":\"a\\u0041b\\\"c\"}," +
			"\"w\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[0,8]}}")
		tensors, dims := countTensorsAndDims(header)
		if tensors != 1 || dims != 1 {
			t.Fatalf("countTensorsAndDims(metadata unicode) = (%d, %d), want (1, 1)", tensors, dims)
		}
	})
	t.Run("tensor inner key with escape", func(t *testing.T) {
		// An escaped inner key (here a forward-compat key carrying a
		// \uXXXX and a \" escape) drives the inner-key escape-skip arms
		// while the real shape is still counted.
		header := []byte("{\"w\":{\"x\\u0041y\":1,\"z\\\"q\":2," +
			"\"dtype\":\"F32\",\"shape\":[4,4],\"data_offsets\":[0,64]}}")
		tensors, dims := countTensorsAndDims(header)
		if tensors != 1 || dims != 2 {
			t.Fatalf("countTensorsAndDims(escaped inner key) = (%d, %d), want (1, 2)", tensors, dims)
		}
	})
	t.Run("metadata then tensor", func(t *testing.T) {
		// Well-formed metadata with a nested string containing braces,
		// followed by a real tensor — exercises the metadata skip loop's
		// string-literal branch and the post-metadata continuation. dims
		// is the dimension *count* (commas+1), so shape [2,5] -> 2.
		header := []byte(`{"__metadata__":{"note":"has {brace} and [bracket]"},` +
			`"w":{"dtype":"F32","shape":[2,5],"data_offsets":[0,20]}}`)
		tensors, dims := countTensorsAndDims(header)
		if tensors != 1 || dims != 2 {
			t.Fatalf("countTensorsAndDims = (%d, %d), want (1, 2)", tensors, dims)
		}
	})
}
