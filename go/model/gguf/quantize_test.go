// SPDX-Licence-Identifier: EUPL-1.2

package gguf

import (
	"context"
	"math"
	"testing"

	core "dappco.re/go"
)

// buildQuantizeSource writes a minimal but real safetensors source pack
// (256-element tensor — block-aligned for every supported QuantizeFormat)
// plus a sibling config.json, and returns a Source describing it.
func buildQuantizeSource(t *testing.T) Source {
	t.Helper()
	dir := t.TempDir()
	if result := core.WriteFile(core.PathJoin(dir, "config.json"), []byte(`{"model_type":"llama"}`), 0o644); !result.OK {
		t.Fatalf("write config.json: %v", result.Value)
	}
	weightPath := core.PathJoin(dir, "model.safetensors")
	writeTestSafetensors(t,
		weightPath,
		map[string][]float32{"weight": rampBlock(qkBlockSize)},
		// A flat 256-element shape divides evenly by every supported
		// QuantizeFormat's block size (32 for Q8_0/Q4_0/Q5_0, 256 for the
		// K-quants) — quantizeGGUFTensor requires Shape[0] % blockSize == 0.
		map[string][]int{"weight": {qkBlockSize}},
	)
	return Source{
		Root:          dir,
		Architecture:  "llama",
		VocabSize:     32000,
		HiddenSize:    32,
		NumLayers:     1,
		ContextLength: 2048,
		WeightFiles:   []string{weightPath},
	}
}

func TestQuantize_QuantizeModelPack_Good(t *testing.T) {
	source := buildQuantizeSource(t)
	output := core.PathJoin(t.TempDir(), "out")

	result, err := QuantizeModelPack(context.Background(), QuantizeOptions{
		SourcePack: source,
		OutputPath: output,
		Format:     QuantizeQ8_0,
		Labels:     map[string]string{"note": "unit-test"},
	})
	if err != nil {
		t.Fatalf("QuantizeModelPack: %v", err)
	}
	if result.Format != QuantizeQ8_0 {
		t.Errorf("Format = %q, want q8_0", result.Format)
	}
	if result.TensorCount != 1 || result.QuantizedTensors != 1 {
		t.Errorf("TensorCount/QuantizedTensors = %d/%d, want 1/1", result.TensorCount, result.QuantizedTensors)
	}
	if !result.Info.Valid() {
		t.Errorf("generated GGUF failed validation: %v", result.Info.ValidationIssues)
	}
	if result.Info.Architecture != "llama" {
		t.Errorf("Info.Architecture = %q, want llama", result.Info.Architecture)
	}
	if result.Info.QuantType != "q8_0" {
		t.Errorf("Info.QuantType = %q, want q8_0", result.Info.QuantType)
	}
	// config.json must have been copied alongside the generated weights.
	if stat := core.Stat(core.PathJoin(output, "config.json")); !stat.OK {
		t.Errorf("config.json was not copied into the output pack")
	}
}

func TestQuantize_QuantizeModelPack_Good_KQuant(t *testing.T) {
	source := buildQuantizeSource(t)
	output := core.PathJoin(t.TempDir(), "out")

	result, err := QuantizeModelPack(context.Background(), QuantizeOptions{
		SourcePack: source,
		OutputPath: output,
		Format:     QuantizeQ4_K_M,
	})
	if err != nil {
		t.Fatalf("QuantizeModelPack: %v", err)
	}
	if result.RequestedFormat != "q4_k_m" {
		t.Errorf("RequestedFormat = %q, want q4_k_m", result.RequestedFormat)
	}
	if result.Format != QuantizeQ4_K {
		t.Errorf("Format = %q, want q4_k (q4_k_m resolves to the q4_k GGML type)", result.Format)
	}
	if !result.Info.Valid() {
		t.Errorf("generated GGUF failed validation: %v", result.Info.ValidationIssues)
	}
}

// buildQuantizeSourceSharded is buildQuantizeSource's multi-shard sibling: it writes two
// block-aligned tensors ("weight" and "weight2") across two safetensors shard files instead of
// one model.safetensors, simulating a real HF sharded checkpoint
// (model.safetensors.index.json + N shard files) — the layout QuantizeModelPack now streams
// tensor-by-tensor through Source.WeightFiles (see loadDenseSafetensors) rather than requiring
// one blob per source.
func buildQuantizeSourceSharded(t *testing.T) Source {
	t.Helper()
	dir := t.TempDir()
	if result := core.WriteFile(core.PathJoin(dir, "config.json"), []byte(`{"model_type":"llama"}`), 0o644); !result.OK {
		t.Fatalf("write config.json: %v", result.Value)
	}
	shard0 := core.PathJoin(dir, "model-00001-of-00002.safetensors")
	shard1 := core.PathJoin(dir, "model-00002-of-00002.safetensors")
	writeTestSafetensors(t, shard0, map[string][]float32{"weight": rampBlock(qkBlockSize)}, map[string][]int{"weight": {qkBlockSize}})
	writeTestSafetensors(t, shard1, map[string][]float32{"weight2": rampBlock(qkBlockSize)}, map[string][]int{"weight2": {qkBlockSize}})
	return Source{
		Root:          dir,
		Architecture:  "llama",
		VocabSize:     32000,
		HiddenSize:    32,
		NumLayers:     1,
		ContextLength: 2048,
		WeightFiles:   []string{shard0, shard1},
	}
}

// TestQuantize_QuantizeModelPack_Sharded proves a genuinely multi-shard source (2 safetensors
// files, one tensor each) quantises correctly — the sharded-set gap tracker #34 flagged as
// unimplemented. Both tensors must be resolved to their own shard and quantised.
func TestQuantize_QuantizeModelPack_Sharded(t *testing.T) {
	source := buildQuantizeSourceSharded(t)
	output := core.PathJoin(t.TempDir(), "out")

	result, err := QuantizeModelPack(context.Background(), QuantizeOptions{
		SourcePack: source,
		OutputPath: output,
		Format:     QuantizeQ8_0,
	})
	if err != nil {
		t.Fatalf("QuantizeModelPack: %v", err)
	}
	if result.TensorCount != 2 || result.QuantizedTensors != 2 {
		t.Errorf("TensorCount/QuantizedTensors = %d/%d, want 2/2", result.TensorCount, result.QuantizedTensors)
	}
	if !result.Info.Valid() {
		t.Errorf("generated GGUF failed validation: %v", result.Info.ValidationIssues)
	}
}

func TestQuantize_QuantizeModelPack_Bad(t *testing.T) {
	if _, err := QuantizeModelPack(context.Background(), QuantizeOptions{}); err == nil {
		t.Fatalf("QuantizeModelPack(no source root): want error, got nil")
	}

	if _, err := QuantizeModelPack(context.Background(), QuantizeOptions{
		SourcePack: Source{Root: t.TempDir()},
	}); err == nil {
		t.Fatalf("QuantizeModelPack(no output path): want error, got nil")
	}

	source := buildQuantizeSource(t)
	if _, err := QuantizeModelPack(context.Background(), QuantizeOptions{
		SourcePack: source,
		OutputPath: core.PathJoin(t.TempDir(), "out.gguf"),
	}); err == nil {
		t.Fatalf("QuantizeModelPack(.gguf output path): want error, got nil")
	}
}

func TestQuantize_QuantizeModelPack_Ugly(t *testing.T) {
	source := buildQuantizeSource(t)

	// Output path identical to the source root.
	if _, err := QuantizeModelPack(context.Background(), QuantizeOptions{
		SourcePack: source,
		OutputPath: source.Root,
	}); err == nil {
		t.Fatalf("QuantizeModelPack(output == source root): want error, got nil")
	}

	// A weight file that is not a .safetensors path.
	badSource := source
	badSource.WeightFiles = []string{core.PathJoin(source.Root, "model.bin")}
	if _, err := QuantizeModelPack(context.Background(), QuantizeOptions{
		SourcePack: badSource,
		OutputPath: core.PathJoin(t.TempDir(), "out"),
	}); err == nil {
		t.Fatalf("QuantizeModelPack(non-safetensors weight file): want error, got nil")
	}

	// Output directory that already contains model weights.
	output := t.TempDir()
	if result := core.WriteFile(core.PathJoin(output, "existing.safetensors"), []byte("x"), 0o644); !result.OK {
		t.Fatalf("seed existing weight file: %v", result.Value)
	}
	if _, err := QuantizeModelPack(context.Background(), QuantizeOptions{
		SourcePack: source,
		OutputPath: output,
	}); err == nil {
		t.Fatalf("QuantizeModelPack(output already has weights): want error, got nil")
	}

	// A cancelled context must short-circuit before doing any work.
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := QuantizeModelPack(ctx, QuantizeOptions{
		SourcePack: source,
		OutputPath: core.PathJoin(t.TempDir(), "out"),
	}); err == nil {
		t.Fatalf("QuantizeModelPack(cancelled context): want error, got nil")
	}
}

func TestQuantize_ValidationSummary_Good(t *testing.T) {
	issues := []ValidationIssue{
		{Code: "invalid_tensor_shape", Tensor: "blk.0.weight"},
		{Code: "unknown_tensor_type"},
	}
	got := ValidationSummary(issues)
	want := "invalid_tensor_shape:blk.0.weight, unknown_tensor_type"
	if got != want {
		t.Errorf("ValidationSummary = %q, want %q", got, want)
	}
}

func TestQuantize_ValidationSummary_Bad(t *testing.T) {
	if got := ValidationSummary(nil); got != "unknown validation failure" {
		t.Errorf("ValidationSummary(nil) = %q, want %q", got, "unknown validation failure")
	}
}

func TestQuantize_resolveGGUFQuantizeFormat_Good(t *testing.T) {
	cases := []struct {
		in        QuantizeFormat
		requested QuantizeFormat
		resolved  QuantizeFormat
	}{
		{"", QuantizeQ8_0, QuantizeQ8_0},
		{"Q4_K_M", "q4_k_m", QuantizeQ4_K},
		{QuantizeQ2_K_M, QuantizeQ2_K_M, QuantizeQ2_K},
		{QuantizeQ2_K, QuantizeQ2_K, QuantizeQ2_K},
	}
	for _, tc := range cases {
		requested, resolved, _, err := resolveGGUFQuantizeFormat(tc.in)
		if err != nil {
			t.Fatalf("resolveGGUFQuantizeFormat(%q): %v", tc.in, err)
		}
		if requested != tc.requested || resolved != tc.resolved {
			t.Errorf("resolveGGUFQuantizeFormat(%q) = (%q, %q), want (%q, %q)", tc.in, requested, resolved, tc.requested, tc.resolved)
		}
	}
}

func TestQuantize_resolveGGUFQuantizeFormat_Bad(t *testing.T) {
	if _, _, _, err := resolveGGUFQuantizeFormat("not-a-format"); err == nil {
		t.Fatalf("resolveGGUFQuantizeFormat(unsupported): want error, got nil")
	}
}

func TestQuantize_loadDenseSafetensors_Good(t *testing.T) {
	dir := t.TempDir()
	pathA := core.PathJoin(dir, "a.safetensors")
	pathB := core.PathJoin(dir, "b.safetensors")
	writeTestSafetensors(t, pathA, map[string][]float32{"a.weight": {1, 2, 3, 4}}, map[string][]int{"a.weight": {4}})
	writeTestSafetensors(t, pathB, map[string][]float32{"b.weight": {5, 6}}, map[string][]int{"b.weight": {2}})

	tensors, err := loadDenseSafetensors([]string{pathA, pathB})
	if err != nil {
		t.Fatalf("loadDenseSafetensors: %v", err)
	}
	if len(tensors) != 2 {
		t.Fatalf("len(tensors) = %d, want 2", len(tensors))
	}
	if tensors[0].Name != "a.weight" || tensors[1].Name != "b.weight" {
		t.Errorf("tensors = %+v, want sorted [a.weight, b.weight]", tensors)
	}
}

func TestQuantize_loadDenseSafetensors_Bad(t *testing.T) {
	if _, err := loadDenseSafetensors(nil); err == nil {
		t.Fatalf("loadDenseSafetensors(nil): want error, got nil")
	}
	if _, err := loadDenseSafetensors([]string{core.PathJoin(t.TempDir(), "missing.safetensors")}); err == nil {
		t.Fatalf("loadDenseSafetensors(missing file): want error, got nil")
	}
}

func TestQuantize_loadDenseSafetensors_Ugly(t *testing.T) {
	dir := t.TempDir()
	pathA := core.PathJoin(dir, "a.safetensors")
	pathB := core.PathJoin(dir, "b.safetensors")
	writeTestSafetensors(t, pathA, map[string][]float32{"dup.weight": {1, 2}}, map[string][]int{"dup.weight": {2}})
	writeTestSafetensors(t, pathB, map[string][]float32{"dup.weight": {3, 4}}, map[string][]int{"dup.weight": {2}})

	if _, err := loadDenseSafetensors([]string{pathA, pathB}); err == nil {
		t.Fatalf("loadDenseSafetensors(duplicate tensor across shards): want error, got nil")
	}
}

func TestQuantize_quantizeGGUFTensor_Good(t *testing.T) {
	tensor := DenseSafetensor{Name: "t", Shape: []uint64{32}, Data: rampBlock(32)}
	quantized, err := quantizeGGUFTensor(tensor, QuantizeQ8_0)
	if err != nil {
		t.Fatalf("quantizeGGUFTensor: %v", err)
	}
	if quantized.Type != TensorTypeQ8_0 || len(quantized.Data) != 34 {
		t.Errorf("quantized = %+v, want Type=Q8_0 len(Data)=34", quantized)
	}
}

func TestQuantize_quantizeGGUFTensor_Bad(t *testing.T) {
	// Block-incompatible tensors (a multimodal pack's scalar clips, tiny
	// biases) fall back to raw F32 storage — llama.cpp's own quantizer does
	// the same — rather than failing the whole model.
	tensor := DenseSafetensor{Name: "t", Shape: []uint64{10}, Data: []float32{0, 1, -2, 3.5, 4, 5, 6, 7, 8, 9}}
	got, err := quantizeGGUFTensor(tensor, QuantizeQ8_0)
	if err != nil {
		t.Fatalf("quantizeGGUFTensor(non-block-aligned): want F32 fallback, got error %v", err)
	}
	if got.Type != TensorTypeF32 {
		t.Fatalf("fallback Type = %d, want F32 (%d)", got.Type, TensorTypeF32)
	}
	if len(got.Data) != 40 {
		t.Fatalf("fallback bytes = %d, want 40 (10 raw f32)", len(got.Data))
	}
	bits := uint32(got.Data[12]) | uint32(got.Data[13])<<8 | uint32(got.Data[14])<<16 | uint32(got.Data[15])<<24
	if math.Float32frombits(bits) != 3.5 {
		t.Fatalf("fallback data[3] = %v, want 3.5 (little-endian raw f32)", math.Float32frombits(bits))
	}
}

func TestQuantize_isMultimodalTowerTensor_Good(t *testing.T) {
	for name, want := range map[string]bool{
		"audio_tower.layers.0.feed_forward1.ffw_layer_1.input_max": true,
		"vision_tower.patch_embedder.input_proj.weight":            true,
		"embed_audio.embedding_projection.weight":                  true,
		"model.layers.0.self_attn.q_proj.weight":                   false,
		"lm_head.weight":                                           false,
	} {
		if got := IsMultimodalTowerTensor(name); got != want {
			t.Fatalf("IsMultimodalTowerTensor(%q) = %v, want %v", name, got, want)
		}
	}
}

func TestQuantize_ggufQuantizeLayout_Good(t *testing.T) {
	tensorType, blockSize, bytesPerBlock, err := ggufQuantizeLayout(QuantizeQ6_K)
	if err != nil {
		t.Fatalf("ggufQuantizeLayout(Q6_K): %v", err)
	}
	if tensorType != TensorTypeQ6K || blockSize != 256 || bytesPerBlock != 210 {
		t.Errorf("ggufQuantizeLayout(Q6_K) = (%d,%d,%d), want (%d,256,210)", tensorType, blockSize, bytesPerBlock, TensorTypeQ6K)
	}
}

func TestQuantize_ggufQuantizeLayout_Bad(t *testing.T) {
	if _, _, _, err := ggufQuantizeLayout("nonsense"); err == nil {
		t.Fatalf("ggufQuantizeLayout(unsupported): want error, got nil")
	}
}

func TestQuantize_ensureEmptyGGUFQuantizeDestination_Good(t *testing.T) {
	if err := ensureEmptyGGUFQuantizeDestination(core.PathJoin(t.TempDir(), "does-not-exist-yet")); err != nil {
		t.Fatalf("ensureEmptyGGUFQuantizeDestination(new path): %v", err)
	}
}

func TestQuantize_ensureEmptyGGUFQuantizeDestination_Bad(t *testing.T) {
	dir := t.TempDir()
	if result := core.WriteFile(core.PathJoin(dir, "model.gguf"), []byte("x"), 0o644); !result.OK {
		t.Fatalf("seed fixture: %v", result.Value)
	}
	if err := ensureEmptyGGUFQuantizeDestination(dir); err == nil {
		t.Fatalf("ensureEmptyGGUFQuantizeDestination(dir with .gguf): want error, got nil")
	}
}

func TestQuantize_samePath_Good(t *testing.T) {
	dir := t.TempDir()
	if !samePath(dir, dir) {
		t.Errorf("samePath(dir, dir) = false, want true")
	}
	if samePath(dir, dir+"-other") {
		t.Errorf("samePath(dir, dir-other) = true, want false")
	}
}

func TestQuantize_copyModelPackMetadata_Good(t *testing.T) {
	source := t.TempDir()
	dest := t.TempDir()
	if result := core.WriteFile(core.PathJoin(source, "config.json"), []byte(`{}`), 0o644); !result.OK {
		t.Fatalf("write config.json: %v", result.Value)
	}
	if result := core.WriteFile(core.PathJoin(source, "weights.safetensors"), []byte("weights"), 0o644); !result.OK {
		t.Fatalf("write weights: %v", result.Value)
	}

	if err := copyModelPackMetadata(source, dest); err != nil {
		t.Fatalf("copyModelPackMetadata: %v", err)
	}
	if stat := core.Stat(core.PathJoin(dest, "config.json")); !stat.OK {
		t.Errorf("config.json was not copied")
	}
	if stat := core.Stat(core.PathJoin(dest, "weights.safetensors")); stat.OK {
		t.Errorf("weights.safetensors should not have been copied")
	}
}
