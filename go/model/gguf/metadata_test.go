// SPDX-Licence-Identifier: EUPL-1.2

package gguf

import (
	"testing"

	core "dappco.re/go"
)

func TestMetadata_Good(t *testing.T) {
	dir := t.TempDir()
	path := core.PathJoin(dir, "model.gguf")
	writeTestGGUF(t, path, []ggufMetaSpec{
		{Key: "general.architecture", ValueType: ValueTypeString, Value: "phi"},
		{Key: "general.file_type", ValueType: ValueTypeUint32, Value: uint32(1)},
	}, nil)

	meta, err := Metadata(path)
	if err != nil {
		t.Fatalf("Metadata: %v", err)
	}
	if meta["general.architecture"] != "phi" {
		t.Errorf("architecture = %v, want phi", meta["general.architecture"])
	}
	if meta["general.file_type"] != uint32(1) {
		t.Errorf("file_type = %v, want 1", meta["general.file_type"])
	}
	if len(meta) != 2 {
		t.Errorf("len(meta) = %d, want 2", len(meta))
	}
}

func TestMetadata_Bad(t *testing.T) {
	_, err := Metadata(core.PathJoin(t.TempDir(), "missing.gguf"))
	if err == nil {
		t.Fatalf("Metadata(missing file): want error, got nil")
	}
}

func TestMetadata_MetadataSubset_Good(t *testing.T) {
	// A header mixing every skip class — fixed-width scalars, strings,
	// a bool, a float32, and a string array — with a keep filter that
	// selects two keys. Skipped entries must not surface in the map;
	// the tensor count comes from the header, not the map.
	dir := t.TempDir()
	path := core.PathJoin(dir, "model.gguf")
	writeTestGGUF(t, path, []ggufMetaSpec{
		{Key: "general.architecture", ValueType: ValueTypeString, Value: "qwen3"},
		{Key: "general.file_type", ValueType: ValueTypeUint32, Value: uint32(15)},
		{Key: "general.is_test", ValueType: ValueTypeBool, Value: true},
		{Key: "general.scale", ValueType: ValueTypeFloat32, Value: float32(1.5)},
		{Key: "tokenizer.ggml.tokens", ValueType: ValueTypeArray, Value: ggufArraySpec{
			ElementType: ValueTypeString,
			Values:      []any{"a", "b", "c"},
		}},
	}, []ggufTensorSpec{
		{Name: "blk.0.attn_q.weight", Type: TensorTypeQ8_0, Dims: []uint64{32}},
	})

	meta, tensorCount, err := MetadataSubset(path, func(key string) bool {
		return key == "general.architecture" || key == "general.file_type"
	})
	if err != nil {
		t.Fatalf("MetadataSubset: %v", err)
	}
	if len(meta) != 2 {
		t.Errorf("len(meta) = %d, want 2 (skipped entries must not land in the map)", len(meta))
	}
	if meta["general.architecture"] != "qwen3" {
		t.Errorf("architecture = %v, want qwen3", meta["general.architecture"])
	}
	if meta["general.file_type"] != uint32(15) {
		t.Errorf("file_type = %v, want 15", meta["general.file_type"])
	}
	if tensorCount != 1 {
		t.Errorf("tensorCount = %d, want 1", tensorCount)
	}
}

func TestMetadata_MetadataSubset_Bad(t *testing.T) {
	_, _, err := MetadataSubset(core.PathJoin(t.TempDir(), "missing.gguf"), func(string) bool { return true })
	if err == nil {
		t.Fatalf("MetadataSubset(missing file): want error, got nil")
	}
}

func TestMetadata_MetadataSubset_Ugly(t *testing.T) {
	// keep(nothing): every value skipped, empty map, counts still read.
	dir := t.TempDir()
	path := core.PathJoin(dir, "model.gguf")
	writeTestGGUF(t, path, []ggufMetaSpec{
		{Key: "general.architecture", ValueType: ValueTypeString, Value: "llama"},
		{Key: "general.lengths", ValueType: ValueTypeArray, Value: ggufArraySpec{
			ElementType: ValueTypeUint32,
			Values:      []any{uint32(1), uint32(2), uint32(3)},
		}},
	}, nil)

	meta, tensorCount, err := MetadataSubset(path, func(string) bool { return false })
	if err != nil {
		t.Fatalf("MetadataSubset: %v", err)
	}
	if len(meta) != 0 {
		t.Errorf("len(meta) = %d, want 0", len(meta))
	}
	if tensorCount != 0 {
		t.Errorf("tensorCount = %d, want 0", tensorCount)
	}
}
