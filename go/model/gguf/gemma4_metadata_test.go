// SPDX-Licence-Identifier: EUPL-1.2

package gguf

import "testing"

// gemma4TestConfig is a compact but structurally faithful gemma-4 config.json:
// six text layers with full attention at index 4 (as the real 35-layer model
// spaces its full-attention layers), plus the hyperparameters the header reads.
const gemma4TestConfig = `{
  "model_type": "gemma4",
  "text_config": {
    "num_hidden_layers": 6,
    "max_position_embeddings": 131072,
    "hidden_size": 1536,
    "intermediate_size": 6144,
    "num_attention_heads": 8,
    "num_key_value_heads": 1,
    "rms_norm_eps": 1e-06,
    "global_head_dim": 512,
    "head_dim": 256,
    "final_logit_softcapping": 30.0,
    "sliding_window": 512,
    "num_kv_shared_layers": 20,
    "hidden_size_per_layer_input": 256,
    "layer_types": [
      "sliding_attention", "sliding_attention", "sliding_attention",
      "sliding_attention", "full_attention", "sliding_attention"
    ],
    "rope_parameters": {
      "full_attention": { "rope_theta": 1000000.0 },
      "sliding_attention": { "rope_theta": 10000.0 }
    }
  }
}`

func gemma4FindEntry(t *testing.T, entries []MetadataEntry, key string) MetadataEntry {
	t.Helper()
	for _, e := range entries {
		if e.Key == key {
			return e
		}
	}
	t.Fatalf("metadata key %q not found", key)
	return MetadataEntry{}
}

// TestGemma4Metadata_gemma4Metadata_Scalars checks the scalar hyperparameters
// map from config.json to the canonical gemma4.* keys with correct types.
func TestGemma4Metadata_gemma4Metadata_Scalars(t *testing.T) {
	entries, err := gemma4Metadata([]byte(gemma4TestConfig), 15, "Gemma-4-E2B-It")
	if err != nil {
		t.Fatalf("gemma4Metadata: %v", err)
	}
	wantU32 := map[string]uint32{
		"gemma4.block_count":                      6,
		"gemma4.context_length":                   131072,
		"gemma4.embedding_length":                 1536,
		"gemma4.attention.head_count":             8,
		"gemma4.attention.head_count_kv":          1,
		"gemma4.attention.key_length":             512,
		"gemma4.attention.value_length":           512,
		"gemma4.attention.sliding_window":         512,
		"gemma4.attention.shared_kv_layers":       20,
		"gemma4.embedding_length_per_layer_input": 256,
		"gemma4.attention.key_length_swa":         256,
		"gemma4.attention.value_length_swa":       256,
		"gemma4.rope.dimension_count":             512,
		"gemma4.rope.dimension_count_swa":         256,
		"general.file_type":                       15,
		"general.quantization_version":            2,
	}
	for key, want := range wantU32 {
		e := gemma4FindEntry(t, entries, key)
		if e.ValueType != ValueTypeUint32 || e.Value.(uint32) != want {
			t.Errorf("%s = %v (type %d), want uint32 %d", key, e.Value, e.ValueType, want)
		}
	}
	wantF32 := map[string]float32{
		"gemma4.rope.freq_base":                   1000000.0,
		"gemma4.rope.freq_base_swa":               10000.0,
		"gemma4.attention.layer_norm_rms_epsilon": 1e-06,
		"gemma4.final_logit_softcapping":          30.0,
	}
	for key, want := range wantF32 {
		e := gemma4FindEntry(t, entries, key)
		if e.ValueType != ValueTypeFloat32 || e.Value.(float32) != want {
			t.Errorf("%s = %v (type %d), want float32 %g", key, e.Value, e.ValueType, want)
		}
	}
	name := gemma4FindEntry(t, entries, "general.name")
	if name.Value.(string) != "Gemma-4-E2B-It" {
		t.Errorf("general.name = %v, want Gemma-4-E2B-It", name.Value)
	}
	arch := gemma4FindEntry(t, entries, "general.architecture")
	if arch.Value.(string) != "gemma4" {
		t.Errorf("general.architecture = %v, want gemma4", arch.Value)
	}
}

// TestGemma4Metadata_gemma4Metadata_Arrays checks the per-layer feed-forward
// length array (broadcast intermediate_size) and the sliding-window pattern
// (sliding true / full false) are built at block_count length.
func TestGemma4Metadata_gemma4Metadata_Arrays(t *testing.T) {
	entries, err := gemma4Metadata([]byte(gemma4TestConfig), 15, "")
	if err != nil {
		t.Fatalf("gemma4Metadata: %v", err)
	}
	ffn := gemma4FindEntry(t, entries, "gemma4.feed_forward_length")
	ffnArr, ok := ffn.Value.([]int32)
	if !ok || len(ffnArr) != 6 {
		t.Fatalf("feed_forward_length = %v (%T), want []int32 len 6", ffn.Value, ffn.Value)
	}
	for i, v := range ffnArr {
		if v != 6144 {
			t.Errorf("feed_forward_length[%d] = %d, want 6144", i, v)
		}
	}
	pat := gemma4FindEntry(t, entries, "gemma4.attention.sliding_window_pattern")
	patArr, ok := pat.Value.([]bool)
	if !ok || len(patArr) != 6 {
		t.Fatalf("sliding_window_pattern = %v (%T), want []bool len 6", pat.Value, pat.Value)
	}
	want := []bool{true, true, true, true, false, true}
	for i := range want {
		if patArr[i] != want[i] {
			t.Errorf("sliding_window_pattern[%d] = %v, want %v", i, patArr[i], want[i])
		}
	}
}

// TestGemma4Metadata_gemma4Metadata_NoName omits general.name when modelName is
// empty rather than writing a blank string.
func TestGemma4Metadata_gemma4Metadata_NoName(t *testing.T) {
	entries, err := gemma4Metadata([]byte(gemma4TestConfig), 15, "")
	if err != nil {
		t.Fatalf("gemma4Metadata: %v", err)
	}
	for _, e := range entries {
		if e.Key == "general.name" {
			t.Errorf("general.name present with empty modelName: %v", e.Value)
		}
	}
}

// TestGemma4Metadata_gemma4Metadata_Bad rejects malformed / incomplete configs
// rather than emitting a half-built header.
func TestGemma4Metadata_gemma4Metadata_Bad(t *testing.T) {
	for name, cfg := range map[string]string{
		"not json":             "{ not json",
		"missing hyperparams":  `{"text_config": {}}`,
		"layer_types mismatch": `{"text_config": {"num_hidden_layers": 6, "hidden_size": 1536, "num_attention_heads": 8, "layer_types": ["full_attention"]}}`,
	} {
		if _, err := gemma4Metadata([]byte(cfg), 15, ""); err == nil {
			t.Errorf("gemma4Metadata(%s): want error, got nil", name)
		}
	}
}
