// SPDX-Licence-Identifier: EUPL-1.2

package gguf

import (
	"encoding/binary"
	"math"
	"testing"

	core "dappco.re/go"
	basegguf "dappco.re/go/inference/model/gguf"
)

func TestGemma3Convert_isGemma3Config(t *testing.T) {
	cases := map[string]bool{
		`{"model_type":"gemma3_text"}`: true,
		`{"model_type":"gemma3"}`:      true,
		`{"model_type":"gemma4"}`:      false,
		`{"model_type":"llama"}`:       false,
		`not json`:                     false,
	}
	for config, want := range cases {
		if got := isGemma3Config([]byte(config)); got != want {
			t.Errorf("isGemma3Config(%s) = %v, want %v", config, got, want)
		}
	}
}

func TestGemma3Convert_isGemma3SupportedQuantizeFormat(t *testing.T) {
	supported := []basegguf.QuantizeFormat{
		basegguf.QuantizeQ4_K_M, basegguf.QuantizeQ8_0, basegguf.QuantizeQ6_K,
		basegguf.QuantizeQ5_K_M, basegguf.QuantizeQ3_K_M, basegguf.QuantizeQ2_K_M,
	}
	for _, f := range supported {
		if !isGemma3SupportedQuantizeFormat(f) {
			t.Errorf("isGemma3SupportedQuantizeFormat(%s) = false, want true", f)
		}
	}
	for _, f := range []basegguf.QuantizeFormat{basegguf.QuantizeQ4_0, "gptq", "awq"} {
		if isGemma3SupportedQuantizeFormat(f) {
			t.Errorf("isGemma3SupportedQuantizeFormat(%s) = true, want false", f)
		}
	}
}

func TestGemma3Convert_gemma3AddScalar(t *testing.T) {
	in := []float32{4.09375, 2.90625, -1.0}
	out := gemma3AddScalar(in, 1.0)
	want := []float32{5.09375, 3.90625, 0.0}
	for i := range want {
		if out[i] != want[i] {
			t.Errorf("out[%d] = %v, want %v", i, out[i], want[i])
		}
	}
	// The source slice must not be mutated.
	if in[0] != 4.09375 {
		t.Errorf("input mutated: in[0] = %v, want 4.09375", in[0])
	}
}

func TestGemma3Convert_encodeGemma3TensorData_F32(t *testing.T) {
	data := []float32{1.5, -2.25}
	encoded, err := encodeGemma3TensorData(data, basegguf.TensorTypeF32)
	if err != nil {
		t.Fatalf("encodeGemma3TensorData: %v", err)
	}
	if len(encoded) != 8 {
		t.Fatalf("len = %d, want 8", len(encoded))
	}
	for i, want := range data {
		got := math.Float32frombits(binary.LittleEndian.Uint32(encoded[i*4:]))
		if got != want {
			t.Errorf("f32[%d] = %v, want %v", i, got, want)
		}
	}
}

func TestGemma3Convert_encodeGemma3TensorData_Unsupported(t *testing.T) {
	// BF16 has no gemma3 encoder (gemma3 stores norms as F32, weights as quants).
	if _, err := encodeGemma3TensorData([]float32{1, 2}, basegguf.TensorTypeBF16); err == nil {
		t.Fatal("encodeGemma3TensorData accepted an unsupported type, want error")
	}
}

func TestGemma3Convert_gemma3ModelName(t *testing.T) {
	cases := map[string]string{
		"/models/LEM-Gemma3-1B-bf16": "LEM-Gemma3-1B",
		"/models/gemma3-1b-f32":      "gemma3-1b",
		"/models/gemma3-1b/":         "gemma3-1b",
	}
	for root, want := range cases {
		if got := gemma3ModelName(root); got != want {
			t.Errorf("gemma3ModelName(%s) = %s, want %s", root, got, want)
		}
	}
}

// TestGemma3Convert_quantizeGemma3ModelPack_NormShift runs the full lane on a
// tiny synthetic pack and checks the two load-bearing transforms: canonical
// tensor naming and the gemma "(1 + weight)" RMS-norm fold baked into every norm
// weight (a norm stored without +1 mis-scales llama.cpp's RMSNorm and produces
// garbage).
func TestGemma3Convert_quantizeGemma3ModelPack_NormShift(t *testing.T) {
	dir := t.TempDir()
	gemma3WriteTestPack(t, dir)
	// Overwrite config.json with the hyperparameters the metadata builder needs
	// (gemma3WriteTestPack writes a tokenizer-only config).
	if r := core.WriteFile(core.PathJoin(dir, "config.json"), []byte(`{
	  "model_type":"gemma3_text","num_hidden_layers":1,"max_position_embeddings":8192,
	  "hidden_size":32,"intermediate_size":64,"num_attention_heads":4,"num_key_value_heads":1,
	  "head_dim":8,"rms_norm_eps":1e-06,"rope_theta":1000000,"rope_local_base_freq":10000,
	  "sliding_window":512,"sliding_window_pattern":6,
	  "bos_token_id":2,"eos_token_id":[1,106],"pad_token_id":0
	}`), 0o644); !r.OK {
		t.Fatalf("write config: %v", r.Err())
	}

	normValues := []float32{1, 2, 3}
	tensors := []basegguf.DenseSafetensor{
		{Name: "model.embed_tokens.weight", Shape: []uint64{7, 32}, Data: make([]float32, 7*32)},
		{Name: "model.layers.0.input_layernorm.weight", Shape: []uint64{3}, Data: normValues},
		{Name: "model.layers.0.self_attn.q_proj.weight", Shape: []uint64{32, 32}, Data: make([]float32, 32*32)},
	}
	quantized, metadata, err := quantizeGemma3ModelPack(
		basegguf.Source{Root: dir}, mustReadConfig(t, dir), tensors, basegguf.QuantizeQ8_0)
	if err != nil {
		t.Fatalf("quantizeGemma3ModelPack: %v", err)
	}

	byName := map[string]basegguf.Tensor{}
	for _, tensor := range quantized {
		byName[tensor.Name] = tensor
	}
	// Canonical naming.
	for _, want := range []string{"token_embd.weight", "blk.0.attn_norm.weight", "blk.0.attn_q.weight"} {
		if _, ok := byName[want]; !ok {
			t.Errorf("missing canonical tensor %q", want)
		}
	}
	// The norm tensor is F32 with +1 folded in.
	norm := byName["blk.0.attn_norm.weight"]
	if norm.Type != basegguf.TensorTypeF32 {
		t.Fatalf("attn_norm type = %d, want F32(%d)", norm.Type, basegguf.TensorTypeF32)
	}
	for i, base := range normValues {
		got := math.Float32frombits(binary.LittleEndian.Uint32(norm.Data[i*4:]))
		if got != base+1.0 {
			t.Errorf("attn_norm[%d] = %v, want %v (base %v + 1)", i, got, base+1.0, base)
		}
	}
	// The projection is Q8_0 (32-divisible), not F32.
	if byName["blk.0.attn_q.weight"].Type != basegguf.TensorTypeQ8_0 {
		t.Errorf("attn_q type = %d, want Q8_0(%d)", byName["blk.0.attn_q.weight"].Type, basegguf.TensorTypeQ8_0)
	}
	// The header carries the architecture + a tokenizer block.
	if !gemma3HasKey(metadata, "general.architecture") || !gemma3HasKey(metadata, "tokenizer.ggml.model") {
		t.Error("metadata missing general.architecture or tokenizer.ggml.model")
	}
}

func mustReadConfig(t *testing.T, dir string) []byte {
	t.Helper()
	read := core.ReadFile(core.PathJoin(dir, "config.json"))
	if !read.OK {
		t.Fatalf("read config: %v", read.Err())
	}
	return read.Value.([]byte)
}

func gemma3HasKey(entries []basegguf.MetadataEntry, key string) bool {
	for _, e := range entries {
		if e.Key == key {
			return true
		}
	}
	return false
}
