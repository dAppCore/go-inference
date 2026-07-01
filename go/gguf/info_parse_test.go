// SPDX-Licence-Identifier: EUPL-1.2

package gguf

import (
	"bytes"
	"testing"

	core "dappco.re/go"
)

func TestInfoParse_parseGGUF_Good(t *testing.T) {
	dir := t.TempDir()
	path := core.PathJoin(dir, "model.gguf")
	writeTestGGUF(t, path, []ggufMetaSpec{
		{Key: "general.architecture", ValueType: ValueTypeString, Value: "llama"},
		{Key: "general.file_type", ValueType: ValueTypeUint32, Value: uint32(7)},
		{Key: "general.is_test", ValueType: ggufValueTypeBool, Value: true},
		{Key: "general.scale", ValueType: ggufValueTypeFloat32, Value: float32(1.5)},
		{Key: "tokenizer.ggml.tokens", ValueType: ggufValueTypeArray, Value: ggufArraySpec{
			ElementType: ValueTypeString,
			Values:      []any{"a", "b", "c"},
		}},
	}, []ggufTensorSpec{
		{Name: "blk.0.attn_q.weight", Type: ggufTensorTypeF32, Dims: []uint64{4, 4}},
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
		{"bool-true", ggufValueTypeBool, func(b *bytes.Buffer) { b.WriteByte(1) }, true},
		{"bool-false", ggufValueTypeBool, func(b *bytes.Buffer) { b.WriteByte(0) }, false},
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
