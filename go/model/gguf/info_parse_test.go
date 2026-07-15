// SPDX-Licence-Identifier: EUPL-1.2

package gguf

import (
	"bytes"
	"encoding/binary"
	"math"
	"testing"

	core "dappco.re/go"
)

func TestInfoParse_parseGGUF_Good(t *testing.T) {
	dir := t.TempDir()
	path := core.PathJoin(dir, "model.gguf")
	writeTestGGUF(t, path, []ggufMetaSpec{
		{Key: "general.architecture", ValueType: ValueTypeString, Value: "llama"},
		{Key: "general.file_type", ValueType: ValueTypeUint32, Value: uint32(7)},
		{Key: "general.is_test", ValueType: ValueTypeBool, Value: true},
		{Key: "general.scale", ValueType: ValueTypeFloat32, Value: float32(1.5)},
		{Key: "tokenizer.ggml.tokens", ValueType: ValueTypeArray, Value: ggufArraySpec{
			ElementType: ValueTypeString,
			Values:      []any{"a", "b", "c"},
		}},
	}, []ggufTensorSpec{
		{Name: "blk.0.attn_q.weight", Type: TensorTypeF32, Dims: []uint64{4, 4}},
	})

	metadata, tensors, err := parseGGUF(path)
	if err != nil {
		t.Fatalf("parseGGUF: %v", err)
	}
	if metadata["general.architecture"] != "llama" {
		t.Errorf("architecture = %v, want llama", metadata["general.architecture"])
	}
	if metadata["general.file_type"] != uint32(7) {
		t.Errorf("file_type = %v, want 7", metadata["general.file_type"])
	}
	if metadata["general.is_test"] != true {
		t.Errorf("is_test = %v, want true", metadata["general.is_test"])
	}
	if metadata["general.scale"] != float32(1.5) {
		t.Errorf("scale = %v, want 1.5", metadata["general.scale"])
	}
	if got := metadataArrayLen(metadata["tokenizer.ggml.tokens"]); got != 3 {
		t.Errorf("tokens array len = %d, want 3", got)
	}
	if len(tensors) != 1 || tensors[0].Name != "blk.0.attn_q.weight" {
		t.Errorf("tensors = %+v, want one blk.0.attn_q.weight entry", tensors)
	}
}

func TestInfoParse_parseGGUF_Bad(t *testing.T) {
	dir := t.TempDir()
	path := core.PathJoin(dir, "empty.gguf")
	if result := core.WriteFile(path, []byte{}, 0o644); !result.OK {
		t.Fatalf("write fixture: %v", result.Value)
	}
	if _, _, err := parseGGUF(path); err == nil {
		t.Fatalf("parseGGUF(empty file): want error, got nil")
	}

	if _, _, err := parseGGUF(core.PathJoin(dir, "missing.gguf")); err == nil {
		t.Fatalf("parseGGUF(missing file): want error, got nil")
	}
}

func TestInfoParse_readStringIntoArena_Good(t *testing.T) {
	value := "general.architecture" // an interned key
	buf := new(bytes.Buffer)
	mustWriteLenPrefixed(t, buf, value)

	var scratch [64]byte
	var arena []byte
	got, err := readStringIntoArena(buf, scratch[:], &arena)
	if err != nil {
		t.Fatalf("readStringIntoArena: %v", err)
	}
	if got != value {
		t.Errorf("readStringIntoArena = %q, want %q", got, value)
	}
}

func TestInfoParse_readStringIntoArena_Ugly(t *testing.T) {
	// Arena too small to hold the string forces the scratch/heap fallback
	// path; result must still be correct.
	value := "a-fresh-uninterned-tensor-name"
	buf := new(bytes.Buffer)
	mustWriteLenPrefixed(t, buf, value)

	var scratch [64]byte
	arena := make([]byte, 0, 1) // deliberately tiny
	got, err := readStringIntoArena(buf, scratch[:], &arena)
	if err != nil {
		t.Fatalf("readStringIntoArena: %v", err)
	}
	if got != value {
		t.Errorf("readStringIntoArena = %q, want %q", got, value)
	}
}

func TestInfoParse_readGGUFString_Good(t *testing.T) {
	value := "general.name"
	buf := new(bytes.Buffer)
	mustWriteLenPrefixed(t, buf, value)

	var scratch [8]byte // forces the heap-buffer branch (len(scratch) < len(value))
	got, err := readGGUFString(buf, scratch[:])
	if err != nil {
		t.Fatalf("readGGUFString: %v", err)
	}
	if got != value {
		t.Errorf("readGGUFString = %q, want %q", got, value)
	}
}

func TestInfoParse_skipGGUFString_Good(t *testing.T) {
	buf := new(bytes.Buffer)
	mustWriteLenPrefixed(t, buf, "discarded-value")
	mustWriteLenPrefixed(t, buf, "next-value")

	var scratch [8]byte // smaller than either string, forces the discard/heap loop
	if err := skipGGUFString(buf, scratch[:]); err != nil {
		t.Fatalf("skipGGUFString: %v", err)
	}
	got, err := readGGUFString(buf, scratch[:])
	if err != nil {
		t.Fatalf("readGGUFString after skip: %v", err)
	}
	if got != "next-value" {
		t.Errorf("value after skip = %q, want next-value", got)
	}
}

func TestInfoParse_readGGUFValue_AllScalarTypes_Good(t *testing.T) {
	cases := []struct {
		name      string
		valueType uint32
		write     func(*bytes.Buffer)
		want      any
	}{
		{"uint8", ggufValueTypeUint8, func(b *bytes.Buffer) { b.WriteByte(0xAB) }, uint8(0xAB)},
		{"int8", ggufValueTypeInt8, func(b *bytes.Buffer) { b.WriteByte(0xFF) }, int8(-1)},
		{"uint16", ggufValueTypeUint16, func(b *bytes.Buffer) { b.Write(binary.LittleEndian.AppendUint16(nil, 0xABCD)) }, uint16(0xABCD)},
		{"int16", ggufValueTypeInt16, func(b *bytes.Buffer) { b.Write(binary.LittleEndian.AppendUint16(nil, 0xFFFF)) }, int16(-1)},
		{"uint32", ValueTypeUint32, func(b *bytes.Buffer) { b.Write(binary.LittleEndian.AppendUint32(nil, 0xDEADBEEF)) }, uint32(0xDEADBEEF)},
		{"int32", ggufValueTypeInt32, func(b *bytes.Buffer) { b.Write(binary.LittleEndian.AppendUint32(nil, 0xFFFFFFFF)) }, int32(-1)},
		{"float32", ValueTypeFloat32, func(b *bytes.Buffer) { b.Write(binary.LittleEndian.AppendUint32(nil, math.Float32bits(1.5))) }, float32(1.5)},
		{"uint64", ggufValueTypeUint64, func(b *bytes.Buffer) { b.Write(binary.LittleEndian.AppendUint64(nil, 0x0123456789ABCDEF)) }, uint64(0x0123456789ABCDEF)},
		{"int64", ggufValueTypeInt64, func(b *bytes.Buffer) { b.Write(binary.LittleEndian.AppendUint64(nil, 0xFFFFFFFFFFFFFFFF)) }, int64(-1)},
		{"float64", ggufValueTypeFloat64, func(b *bytes.Buffer) { b.Write(binary.LittleEndian.AppendUint64(nil, math.Float64bits(2.25))) }, float64(2.25)},
		{"bool-true", ValueTypeBool, func(b *bytes.Buffer) { b.WriteByte(1) }, true},
		{"bool-false", ValueTypeBool, func(b *bytes.Buffer) { b.WriteByte(0) }, false},
	}
	for _, tc := range cases {
		buf := new(bytes.Buffer)
		tc.write(buf)
		var scratch [8]byte
		got, err := readGGUFValue(buf, tc.valueType, scratch[:], nil)
		if err != nil {
			t.Fatalf("%s: readGGUFValue: %v", tc.name, err)
		}
		if got != tc.want {
			t.Errorf("%s: readGGUFValue = %v (%T), want %v (%T)", tc.name, got, got, tc.want, tc.want)
		}
	}
}

func TestInfoParse_readGGUFValue_UnsupportedType_Bad(t *testing.T) {
	buf := new(bytes.Buffer)
	var scratch [8]byte
	if _, err := readGGUFValue(buf, 0xFFFF, scratch[:], nil); err == nil {
		t.Fatalf("readGGUFValue(unsupported type): want error, got nil")
	}
}

// TestInfoParse_readGGUFValue_String_Good exercises the two string branches of
// the value switch: the arena path (strArena != nil) and the standalone
// readGGUFString path (strArena == nil). Both must decode the same bytes.
func TestInfoParse_readGGUFValue_String_Good(t *testing.T) {
	value := "a-fresh-uninterned-string"

	arenaBuf := new(bytes.Buffer)
	mustWriteLenPrefixed(t, arenaBuf, value)
	var scratch [64]byte
	var arena []byte
	got, err := readGGUFValue(arenaBuf, ValueTypeString, scratch[:], &arena)
	if err != nil {
		t.Fatalf("readGGUFValue(string, arena): %v", err)
	}
	if got != value {
		t.Errorf("readGGUFValue(string, arena) = %v, want %q", got, value)
	}

	plainBuf := new(bytes.Buffer)
	mustWriteLenPrefixed(t, plainBuf, value)
	got, err = readGGUFValue(plainBuf, ValueTypeString, scratch[:], nil)
	if err != nil {
		t.Fatalf("readGGUFValue(string, no arena): %v", err)
	}
	if got != value {
		t.Errorf("readGGUFValue(string, no arena) = %v, want %q", got, value)
	}
}

// TestInfoParse_readGGUFValue_NumericArray_Good decodes a numeric-element array,
// which materialises every element into a []any (unlike the string-element array
// which is counted only). Verifies element type, count and values.
func TestInfoParse_readGGUFValue_NumericArray_Good(t *testing.T) {
	buf := new(bytes.Buffer)
	buf.Write(binary.LittleEndian.AppendUint32(nil, ValueTypeUint32)) // element type
	buf.Write(binary.LittleEndian.AppendUint64(nil, 3))               // length
	for _, v := range []uint32{10, 20, 30} {
		buf.Write(binary.LittleEndian.AppendUint32(nil, v))
	}

	var scratch [8]byte
	got, err := readGGUFValue(buf, ValueTypeArray, scratch[:], nil)
	if err != nil {
		t.Fatalf("readGGUFValue(numeric array): %v", err)
	}
	values, ok := got.([]any)
	if !ok {
		t.Fatalf("readGGUFValue(numeric array) = %T, want []any", got)
	}
	if len(values) != 3 || values[0] != uint32(10) || values[2] != uint32(30) {
		t.Errorf("readGGUFValue(numeric array) = %v, want [10 20 30]", values)
	}
}

// TestInfoParse_readGGUFValue_StringArray_Good exercises the vocab-skip fast
// path: a string-element array is parsed for its COUNT only and returned as a
// ggufStringArrayLen, never materialising the (potentially 200k-token) strings.
func TestInfoParse_readGGUFValue_StringArray_Good(t *testing.T) {
	buf := new(bytes.Buffer)
	buf.Write(binary.LittleEndian.AppendUint32(nil, ValueTypeString)) // element type
	buf.Write(binary.LittleEndian.AppendUint64(nil, 4))               // length
	for _, tok := range []string{"a", "bb", "ccc", "dddd"} {
		mustWriteLenPrefixed(t, buf, tok)
	}

	var scratch [8]byte
	got, err := readGGUFValue(buf, ValueTypeArray, scratch[:], nil)
	if err != nil {
		t.Fatalf("readGGUFValue(string array): %v", err)
	}
	if got != ggufStringArrayLen(4) {
		t.Errorf("readGGUFValue(string array) = %v (%T), want ggufStringArrayLen(4)", got, got)
	}
	if n := metadataArrayLen(got); n != 4 {
		t.Errorf("metadataArrayLen = %d, want 4", n)
	}
}

// TestInfoParse_readGGUFValue_ArrayTooLong_Bad rejects an array whose declared
// length exceeds maxGGUFCollectionEntries — the guard against a corrupt header
// asking the parser to allocate a giant slice.
func TestInfoParse_readGGUFValue_ArrayTooLong_Bad(t *testing.T) {
	buf := new(bytes.Buffer)
	buf.Write(binary.LittleEndian.AppendUint32(nil, ValueTypeUint32))
	buf.Write(binary.LittleEndian.AppendUint64(nil, maxGGUFCollectionEntries+1))

	var scratch [8]byte
	if _, err := readGGUFValue(buf, ValueTypeArray, scratch[:], nil); err == nil {
		t.Fatalf("readGGUFValue(over-length array): want error, got nil")
	}
}

// TestInfoParse_readGGUFValue_Truncated_Ugly feeds a reader that ends before the
// value is fully read. Every scalar branch, plus the array element-type and
// length headers, must surface the io.ReadFull error rather than returning a
// half-decoded value.
func TestInfoParse_readGGUFValue_Truncated_Ugly(t *testing.T) {
	cases := []struct {
		name      string
		valueType uint32
		payload   []byte // deliberately shorter than the type needs
	}{
		{"uint8", ggufValueTypeUint8, nil},
		{"int8", ggufValueTypeInt8, nil},
		{"uint16", ggufValueTypeUint16, []byte{0x01}},
		{"int16", ggufValueTypeInt16, []byte{0x01}},
		{"uint32", ValueTypeUint32, []byte{0x01, 0x02}},
		{"int32", ggufValueTypeInt32, []byte{0x01, 0x02}},
		{"float32", ValueTypeFloat32, []byte{0x01, 0x02}},
		{"bool", ValueTypeBool, nil},
		{"uint64", ggufValueTypeUint64, []byte{0x01, 0x02, 0x03}},
		{"int64", ggufValueTypeInt64, []byte{0x01, 0x02, 0x03}},
		{"float64", ggufValueTypeFloat64, []byte{0x01, 0x02, 0x03}},
		{"array-elemtype", ValueTypeArray, []byte{0x01}},                                         // < 4 bytes for element type
		{"array-length", ValueTypeArray, binary.LittleEndian.AppendUint32(nil, ValueTypeUint32)}, // element type ok, length truncated
	}
	for _, tc := range cases {
		var scratch [8]byte
		if _, err := readGGUFValue(bytes.NewReader(tc.payload), tc.valueType, scratch[:], nil); err == nil {
			t.Errorf("%s: readGGUFValue(truncated): want error, got nil", tc.name)
		}
	}
}

// mustWriteLenPrefixed writes a GGUF-style [uint64 length][bytes] string
// to buf.
func mustWriteLenPrefixed(t *testing.T, buf *bytes.Buffer, value string) {
	t.Helper()
	var lenBytes [8]byte
	for i := range lenBytes {
		lenBytes[i] = byte(len(value) >> (8 * i))
	}
	buf.Write(lenBytes[:])
	buf.WriteString(value)
}
