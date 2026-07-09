// SPDX-Licence-Identifier: EUPL-1.2

package gguf

import (
	"testing"

	core "dappco.re/go"
)

func TestQuantizeWriter_ggufQuantizeMetadata_Good(t *testing.T) {
	source := Source{Architecture: "llama", VocabSize: 32000, HiddenSize: 4096, NumLayers: 32, ContextLength: 8192}
	entries := ggufQuantizeMetadata(source, QuantizeQ4_K, map[string]string{"z": "last", "a": "first"})

	find := func(key string) (ggufMetadataEntry, bool) {
		for _, e := range entries {
			if e.Key == key {
				return e, true
			}
		}
		return ggufMetadataEntry{}, false
	}

	if e, ok := find("general.architecture"); !ok || e.Value != "llama" {
		t.Errorf("general.architecture = %+v, ok=%v, want llama", e, ok)
	}
	if e, ok := find("general.file_type"); !ok || e.Value != uint32(15) {
		t.Errorf("general.file_type = %+v, ok=%v, want 15 (Q4_K resolves to q4_k_m file_type)", e, ok)
	}
	if e, ok := find("llama.vocab_size"); !ok || e.Value != uint32(32000) {
		t.Errorf("llama.vocab_size = %+v, ok=%v, want 32000", e, ok)
	}
	if e, ok := find("llama.block_count"); !ok || e.Value != uint32(32) {
		t.Errorf("llama.block_count = %+v, ok=%v, want 32", e, ok)
	}
	// Labels are sorted by key.
	labelIdx := map[string]int{}
	for i, e := range entries {
		if core.HasPrefix(e.Key, "gguf.label.") {
			labelIdx[e.Key] = i
		}
	}
	if labelIdx["gguf.label.a"] >= labelIdx["gguf.label.z"] {
		t.Errorf("labels not sorted: %+v", entries)
	}
}

func TestQuantizeWriter_assignGGUFTensorOffsets_Good(t *testing.T) {
	tensors := []ggufQuantizedTensor{
		{Name: "a", Data: make([]byte, 10)},
		{Name: "b", Data: make([]byte, 3)},
	}
	assignGGUFTensorOffsets(tensors, 32)
	if tensors[0].Offset != 0 {
		t.Errorf("tensors[0].Offset = %d, want 0", tensors[0].Offset)
	}
	// Second tensor must start at the next 32-byte-aligned offset after
	// the first tensor's 10 data bytes.
	if tensors[1].Offset != 32 {
		t.Errorf("tensors[1].Offset = %d, want 32", tensors[1].Offset)
	}
}

func TestQuantizeWriter_alignPadding_Good(t *testing.T) {
	cases := []struct {
		offset, alignment, want uint64
	}{
		{0, 32, 0},
		{1, 32, 31},
		{32, 32, 0},
		{10, 0, 0},
	}
	for _, tc := range cases {
		if got := alignPadding(tc.offset, tc.alignment); got != tc.want {
			t.Errorf("alignPadding(%d,%d) = %d, want %d", tc.offset, tc.alignment, got, tc.want)
		}
	}
}

func TestQuantizeWriter_writeGGUFStringValue_Good(t *testing.T) {
	path := core.PathJoin(t.TempDir(), "value.bin")
	created := core.Create(path)
	if !created.OK {
		t.Fatalf("create: %v", created.Value)
	}
	file := created.Value.(*core.OSFile)

	short := "general.architecture"
	long := core.Repeat("x", 512) // forces the heap-buffer branch
	if err := writeGGUFStringValue(file, short); err != nil {
		t.Fatalf("writeGGUFStringValue(short): %v", err)
	}
	if err := writeGGUFStringValue(file, long); err != nil {
		t.Fatalf("writeGGUFStringValue(long): %v", err)
	}
	file.Close()

	read := core.ReadFile(path)
	if !read.OK {
		t.Fatalf("read back: %v", read.Value)
	}
	data := read.Value.([]byte)
	wantLen := 8 + len(short) + 8 + len(long)
	if len(data) != wantLen {
		t.Fatalf("written length = %d, want %d", len(data), wantLen)
	}
}

func TestQuantizeWriter_writePadding_Good(t *testing.T) {
	path := core.PathJoin(t.TempDir(), "padding.bin")
	created := core.Create(path)
	if !created.OK {
		t.Fatalf("create: %v", created.Value)
	}
	file := created.Value.(*core.OSFile)
	n := uint64(len(ggufPaddingZeros)) + 10 // forces the multi-iteration loop
	if err := writePadding(file, n); err != nil {
		t.Fatalf("writePadding: %v", err)
	}
	file.Close()

	stat := core.Stat(path)
	if !stat.OK {
		t.Fatalf("stat: %v", stat.Value)
	}
	if size := stat.Value.(core.FsFileInfo).Size(); uint64(size) != n {
		t.Errorf("padded file size = %d, want %d", size, n)
	}
}

func TestQuantizeWriter_writeQuantizedGGUF_Good(t *testing.T) {
	path := core.PathJoin(t.TempDir(), "written.gguf")
	metadata := []ggufMetadataEntry{
		{Key: "general.architecture", ValueType: ValueTypeString, Value: "llama"},
		{Key: "general.file_type", ValueType: ValueTypeUint32, Value: uint32(7)},
	}
	tensors := []ggufQuantizedTensor{
		{Name: "blk.0.weight", Type: TensorTypeQ8_0, Shape: []uint64{32}, Data: quantizeQ8_0(rampBlock(32))},
	}

	if err := writeQuantizedGGUF(path, metadata, tensors); err != nil {
		t.Fatalf("writeQuantizedGGUF: %v", err)
	}

	parsedMeta, parsedTensors, err := parseGGUF(path)
	if err != nil {
		t.Fatalf("parseGGUF(written file): %v", err)
	}
	if parsedMeta["general.architecture"] != "llama" {
		t.Errorf("architecture = %v, want llama", parsedMeta["general.architecture"])
	}
	if len(parsedTensors) != 1 || parsedTensors[0].Name != "blk.0.weight" || parsedTensors[0].Type != TensorTypeQ8_0 {
		t.Errorf("parsed tensors = %+v, want one blk.0.weight Q8_0 entry", parsedTensors)
	}
}

func TestQuantizeWriter_WriteFile_Good(t *testing.T) {
	// The public writer front door: string + uint32 + float32 metadata and
	// one tensor, read back through this package's own parser. Also locks
	// the canonical data-section shape — gguf-py pads every tensor
	// including the last, so a data-carrying file's length is 32-aligned.
	path := core.PathJoin(t.TempDir(), "written.gguf")
	metadata := []MetadataEntry{
		{Key: "general.architecture", ValueType: ValueTypeString, Value: "gemma3"},
		{Key: "general.file_type", ValueType: ValueTypeUint32, Value: uint32(7)},
		{Key: "adapter.lora.alpha", ValueType: ValueTypeFloat32, Value: float32(16)},
	}
	tensors := []Tensor{
		{Name: "blk.0.weight", Type: TensorTypeQ8_0, Shape: []uint64{32}, Data: quantizeQ8_0(rampBlock(32))},
	}

	if err := WriteFile(path, metadata, tensors); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}

	parsedMeta, parsedTensors, err := parseGGUF(path)
	if err != nil {
		t.Fatalf("parseGGUF(written file): %v", err)
	}
	if parsedMeta["general.architecture"] != "gemma3" {
		t.Errorf("architecture = %v, want gemma3", parsedMeta["general.architecture"])
	}
	if parsedMeta["adapter.lora.alpha"] != float32(16) {
		t.Errorf("adapter.lora.alpha = %v, want float32 16", parsedMeta["adapter.lora.alpha"])
	}
	if len(parsedTensors) != 1 || parsedTensors[0].Name != "blk.0.weight" || parsedTensors[0].Type != TensorTypeQ8_0 {
		t.Errorf("parsed tensors = %+v, want one blk.0.weight Q8_0 entry", parsedTensors)
	}

	stat := core.Stat(path)
	if !stat.OK {
		t.Fatalf("stat: %v", stat.Value)
	}
	if size := stat.Value.(core.FsFileInfo).Size(); size%32 != 0 {
		t.Errorf("file size = %d, want a 32-byte-aligned length (trailing data-section padding)", size)
	}
}

func TestQuantizeWriter_writeGGUFMetadataValue_Float32(t *testing.T) {
	// Round-trip: a float32 metadata entry written by this package's writer
	// must come back bit-exact through this package's own parser — the case
	// modelmgmt's GGUF LoRA adapter (adapter.lora.alpha) depends on.
	path := core.PathJoin(t.TempDir(), "float32.gguf")
	metadata := []ggufMetadataEntry{
		{Key: "general.architecture", ValueType: ValueTypeString, Value: "llama"},
		{Key: "adapter.lora.alpha", ValueType: ValueTypeFloat32, Value: float32(16.5)},
	}

	if err := writeQuantizedGGUF(path, metadata, nil); err != nil {
		t.Fatalf("writeQuantizedGGUF: %v", err)
	}

	parsed, _, err := parseGGUF(path)
	if err != nil {
		t.Fatalf("parseGGUF(written file): %v", err)
	}
	if got, ok := parsed["adapter.lora.alpha"].(float32); !ok || got != 16.5 {
		t.Errorf("adapter.lora.alpha = %v (%T), want float32 16.5", parsed["adapter.lora.alpha"], parsed["adapter.lora.alpha"])
	}
}

func TestQuantizeWriter_writeGGUFMetadataValue_Float32_Bad(t *testing.T) {
	path := core.PathJoin(t.TempDir(), "value.bin")
	created := core.Create(path)
	if !created.OK {
		t.Fatalf("create: %v", created.Value)
	}
	file := created.Value.(*core.OSFile)
	defer file.Close()

	if err := writeGGUFMetadataValue(file, ValueTypeFloat32, "not-a-float32"); err == nil {
		t.Fatalf("writeGGUFMetadataValue(float32 type, string value): want error, got nil")
	}
}

func TestQuantizeWriter_writeGGUFMetadataValue_Bad(t *testing.T) {
	path := core.PathJoin(t.TempDir(), "value.bin")
	created := core.Create(path)
	if !created.OK {
		t.Fatalf("create: %v", created.Value)
	}
	file := created.Value.(*core.OSFile)
	defer file.Close()

	if err := writeGGUFMetadataValue(file, ValueTypeString, 123); err == nil {
		t.Fatalf("writeGGUFMetadataValue(string type, int value): want error, got nil")
	}
	if err := writeGGUFMetadataValue(file, ValueTypeUint32, "not-a-uint32"); err == nil {
		t.Fatalf("writeGGUFMetadataValue(uint32 type, string value): want error, got nil")
	}
	if err := writeGGUFMetadataValue(file, 0xFFFF, nil); err == nil {
		t.Fatalf("writeGGUFMetadataValue(unsupported type): want error, got nil")
	}
}
