// SPDX-Licence-Identifier: EUPL-1.2

package gguf

import (
	"testing"

	core "dappco.re/go"
)

func TestInfo_Valid_Good(t *testing.T) {
	info := Info{ValidationIssues: nil}
	if !info.Valid() {
		t.Fatalf("Valid() = false, want true for no issues")
	}
}

func TestInfo_Valid_Bad(t *testing.T) {
	info := Info{ValidationIssues: []ValidationIssue{
		{Severity: GGUFValidationError, Code: "invalid_tensor_shape"},
	}}
	if info.Valid() {
		t.Fatalf("Valid() = true, want false when an error-severity issue is present")
	}
}

func TestInfo_Valid_Ugly(t *testing.T) {
	// Warning-severity issues alone must not flip Valid() to false.
	info := Info{ValidationIssues: []ValidationIssue{
		{Severity: GGUFValidationWarning, Code: "unusual_shape"},
	}}
	if !info.Valid() {
		t.Fatalf("Valid() = false, want true when only warnings are present")
	}
}

func TestInfo_ReadInfo_Good(t *testing.T) {
	dir := t.TempDir()
	path := core.PathJoin(dir, "model.gguf")
	writeTestGGUF(t, path, []ggufMetaSpec{
		{Key: "general.architecture", ValueType: ValueTypeString, Value: "qwen3"},
		{Key: "general.file_type", ValueType: ValueTypeUint32, Value: uint32(15)},
		{Key: "qwen3.block_count", ValueType: ValueTypeUint32, Value: uint32(2)},
		{Key: "qwen3.context_length", ValueType: ValueTypeUint32, Value: uint32(40960)},
		{Key: "qwen3.embedding_length", ValueType: ValueTypeUint32, Value: uint32(2048)},
	}, []ggufTensorSpec{
		{Name: "blk.0.attn_q.weight", Type: ggufTensorTypeQ4K, Dims: []uint64{256, 4}},
		{Name: "blk.1.attn_q.weight", Type: ggufTensorTypeQ4K, Dims: []uint64{256, 4}},
	})

	info, err := ReadInfo(path)
	if err != nil {
		t.Fatalf("ReadInfo: %v", err)
	}
	if info.Architecture != "qwen3" {
		t.Errorf("Architecture = %q, want qwen3", info.Architecture)
	}
	if info.NumLayers != 2 {
		t.Errorf("NumLayers = %d, want 2", info.NumLayers)
	}
	if info.ContextLength != 40960 {
		t.Errorf("ContextLength = %d, want 40960", info.ContextLength)
	}
	if info.HiddenSize != 2048 {
		t.Errorf("HiddenSize = %d, want 2048", info.HiddenSize)
	}
	if info.QuantBits != 4 {
		t.Errorf("QuantBits = %d, want 4", info.QuantBits)
	}
	if info.QuantType != "q4_k_m" {
		// general.file_type=15 resolves via ggufFileTypeQuantizationTable to
		// "q4_k_m" and takes priority over the tensor-type majority vote
		// ("q4_k") in inferGGUFQuantization's firstNonEmpty ordering.
		t.Errorf("QuantType = %q, want q4_k_m", info.QuantType)
	}
	if info.TensorCount != 2 {
		t.Errorf("TensorCount = %d, want 2", info.TensorCount)
	}
	if info.MetadataCount != 5 {
		t.Errorf("MetadataCount = %d, want 5", info.MetadataCount)
	}
	if !info.Valid() {
		t.Errorf("Valid() = false, want true: %v", info.ValidationIssues)
	}
	if info.Path == "" {
		t.Errorf("Path is empty")
	}
}

func TestInfo_ReadInfo_Bad(t *testing.T) {
	_, err := ReadInfo(core.PathJoin(t.TempDir(), "missing.gguf"))
	if err == nil {
		t.Fatalf("ReadInfo: want error for missing file, got nil")
	}
}

func TestInfo_ReadInfo_Ugly(t *testing.T) {
	// A tensor with a zero-length shape dimension is a validation error,
	// but ReadInfo itself must still succeed — Valid() carries the finding.
	dir := t.TempDir()
	path := core.PathJoin(dir, "model.gguf")
	writeTestGGUF(t, path, []ggufMetaSpec{
		{Key: "general.architecture", ValueType: ValueTypeString, Value: "llama"},
	}, []ggufTensorSpec{
		{Name: "broken.weight", Type: ggufTensorTypeF32, Dims: []uint64{0}},
	})

	info, err := ReadInfo(path)
	if err != nil {
		t.Fatalf("ReadInfo: %v", err)
	}
	if info.Valid() {
		t.Fatalf("Valid() = true, want false for a zero-dimension tensor")
	}
	if !ggufValidationHasCode(info.ValidationIssues, "invalid_tensor_dimension") {
		t.Errorf("ValidationIssues = %+v, want invalid_tensor_dimension", info.ValidationIssues)
	}
}

func TestInfo_ReadInfo_InvalidMagic_Bad(t *testing.T) {
	dir := t.TempDir()
	path := core.PathJoin(dir, "model.gguf")
	if result := core.WriteFile(path, []byte("NOPE12345678901234567890"), 0o644); !result.OK {
		t.Fatalf("write corrupt gguf: %v", result.Value)
	}

	_, err := ReadInfo(path)
	if err == nil {
		t.Fatalf("ReadInfo: want error for invalid magic, got nil")
	}
}

func TestInfo_ReadInfo_ConfigJSONFallback_Good(t *testing.T) {
	dir := t.TempDir()
	if result := core.WriteFile(core.PathJoin(dir, "config.json"), []byte(`{
		"model_type": "gemma3",
		"vocab_size": 262208,
		"hidden_size": 3072,
		"num_hidden_layers": 26,
		"max_position_embeddings": 8192
	}`), 0o644); !result.OK {
		t.Fatalf("write config.json: %v", result.Value)
	}
	path := core.PathJoin(dir, "model.gguf")
	// No general.architecture / dimension keys in the GGUF metadata —
	// ReadInfo must fall back to the sibling config.json.
	writeTestGGUF(t, path, []ggufMetaSpec{
		{Key: "general.name", ValueType: ValueTypeString, Value: "test model"},
	}, nil)

	info, err := ReadInfo(path)
	if err != nil {
		t.Fatalf("ReadInfo: %v", err)
	}
	if info.Architecture != "gemma3" {
		t.Errorf("Architecture = %q, want gemma3 (from config.json)", info.Architecture)
	}
	if info.VocabSize != 262208 {
		t.Errorf("VocabSize = %d, want 262208", info.VocabSize)
	}
	if info.HiddenSize != 3072 {
		t.Errorf("HiddenSize = %d, want 3072", info.HiddenSize)
	}
	if info.NumLayers != 26 {
		t.Errorf("NumLayers = %d, want 26", info.NumLayers)
	}
	if info.ContextLength != 8192 {
		t.Errorf("ContextLength = %d, want 8192", info.ContextLength)
	}
}

func TestInfo_ReadInfo_LayerCountFromTensorNames_Good(t *testing.T) {
	dir := t.TempDir()
	path := core.PathJoin(dir, "model.gguf")
	// No block_count metadata — NumLayers must be inferred from the
	// highest "blk.<N>." tensor-name index seen (+1).
	writeTestGGUF(t, path, []ggufMetaSpec{
		{Key: "general.architecture", ValueType: ValueTypeString, Value: "llama"},
	}, []ggufTensorSpec{
		{Name: "blk.0.attn_q.weight", Type: ggufTensorTypeF32, Dims: []uint64{4, 4}},
		{Name: "blk.3.attn_q.weight", Type: ggufTensorTypeF32, Dims: []uint64{4, 4}},
	})

	info, err := ReadInfo(path)
	if err != nil {
		t.Fatalf("ReadInfo: %v", err)
	}
	if info.NumLayers != 4 {
		t.Errorf("NumLayers = %d, want 4 (max blk index 3 + 1)", info.NumLayers)
	}
}

func TestInfo_resolveGGUFFile_Good(t *testing.T) {
	dir := t.TempDir()
	direct := core.PathJoin(dir, "model.GGUF")
	if result := core.WriteFile(direct, []byte("x"), 0o644); !result.OK {
		t.Fatalf("write fixture: %v", result.Value)
	}
	got, err := resolveGGUFFile(direct)
	if err != nil {
		t.Fatalf("resolveGGUFFile(direct path): %v", err)
	}
	if got != direct {
		t.Errorf("resolveGGUFFile(direct path) = %q, want %q", got, direct)
	}

	single := t.TempDir()
	target := core.PathJoin(single, "only.gguf")
	if result := core.WriteFile(target, []byte("x"), 0o644); !result.OK {
		t.Fatalf("write fixture: %v", result.Value)
	}
	got, err = resolveGGUFFile(single)
	if err != nil {
		t.Fatalf("resolveGGUFFile(dir with one .gguf): %v", err)
	}
	if got != target {
		t.Errorf("resolveGGUFFile(dir) = %q, want %q", got, target)
	}
}

func TestInfo_resolveGGUFFile_Bad(t *testing.T) {
	_, err := resolveGGUFFile(t.TempDir())
	if err != errGGUFNoFile {
		t.Fatalf("resolveGGUFFile(empty dir) error = %v, want errGGUFNoFile", err)
	}
}

func TestInfo_resolveGGUFFile_Ugly(t *testing.T) {
	dir := t.TempDir()
	for _, name := range []string{"a.gguf", "b.gguf"} {
		if result := core.WriteFile(core.PathJoin(dir, name), []byte("x"), 0o644); !result.OK {
			t.Fatalf("write fixture: %v", result.Value)
		}
	}
	_, err := resolveGGUFFile(dir)
	if err != errGGUFMultipleFiles {
		t.Fatalf("resolveGGUFFile(dir with two .gguf) error = %v, want errGGUFMultipleFiles", err)
	}
}

func TestInfo_ResolveFile_Good(t *testing.T) {
	dir := t.TempDir()
	target := core.PathJoin(dir, "only.gguf")
	if result := core.WriteFile(target, []byte("x"), 0o644); !result.OK {
		t.Fatalf("write fixture: %v", result.Value)
	}
	got, err := ResolveFile(dir)
	if err != nil {
		t.Fatalf("ResolveFile(dir with one .gguf): %v", err)
	}
	if got != target {
		t.Errorf("ResolveFile(dir) = %q, want %q", got, target)
	}
}

func TestInfo_ResolveFile_Bad(t *testing.T) {
	_, err := ResolveFile(t.TempDir())
	if err != errGGUFNoFile {
		t.Fatalf("ResolveFile(empty dir) error = %v, want errGGUFNoFile", err)
	}
}

func TestInfo_ResolveFile_Ugly(t *testing.T) {
	// A .gguf-suffixed path resolves to itself without touching the disk.
	got, err := ResolveFile("/nonexistent/model.gguf")
	if err != nil {
		t.Fatalf("ResolveFile(.gguf path): %v", err)
	}
	if got != "/nonexistent/model.gguf" {
		t.Errorf("ResolveFile(.gguf path) = %q, want the input path", got)
	}
}

func TestInfo_architectureFromTransformersName_Good(t *testing.T) {
	cases := []struct {
		name string
		want string
	}{
		{"Qwen3ForCausalLM", "qwen3"},
		{"LlamaForCausalLM", "llama"},
		{"Gemma3ForConditionalGeneration", "gemma3"},
		{"BertForSequenceClassification", "bert"},
		{"", ""},
	}
	for _, tc := range cases {
		if got := architectureFromTransformersName(tc.name); got != tc.want {
			t.Errorf("architectureFromTransformersName(%q) = %q, want %q", tc.name, got, tc.want)
		}
	}
}

func TestInfo_normalizeArchitectureName_Good(t *testing.T) {
	cases := []struct {
		value string
		want  string
	}{
		{"Qwen3", "qwen3"},
		{" gemma-3 ", "gemma_3"},
		{"MiniMax.M2", "minimax_m2"},
	}
	for _, tc := range cases {
		if got := normalizeArchitectureName(tc.value); got != tc.want {
			t.Errorf("normalizeArchitectureName(%q) = %q, want %q", tc.value, got, tc.want)
		}
	}
}

func TestInfo_hasASCIIInsensitiveSuffix_Good(t *testing.T) {
	if !hasASCIIInsensitiveSuffix("model.GGUF", ".gguf") {
		t.Errorf("want case-insensitive suffix match")
	}
	if hasASCIIInsensitiveSuffix("model.bin", ".gguf") {
		t.Errorf("want no match for unrelated suffix")
	}
	if hasASCIIInsensitiveSuffix("guf", ".gguf") {
		t.Errorf("want no match when input shorter than suffix")
	}
}

func ggufValidationHasCode(issues []ValidationIssue, code string) bool {
	for _, issue := range issues {
		if issue.Code == code {
			return true
		}
	}
	return false
}
