// SPDX-Licence-Identifier: EUPL-1.2

package gguf

import (
	"testing"

	basegguf "dappco.re/go/inference/model/gguf"
)

// gemma3TestConfig is a compact gemma3_text config.json carrying every
// hyperparameter the header reads, with gemma-3-1B's real values.
const gemma3TestConfig = `{
  "model_type": "gemma3_text",
  "num_hidden_layers": 26,
  "max_position_embeddings": 32768,
  "hidden_size": 1152,
  "intermediate_size": 6912,
  "num_attention_heads": 4,
  "num_key_value_heads": 1,
  "head_dim": 256,
  "rms_norm_eps": 1e-06,
  "rope_theta": 1000000,
  "rope_local_base_freq": 10000,
  "sliding_window": 512,
  "sliding_window_pattern": 6
}`

// gemma3MetadataEntry returns the entry with key, or a zero entry if absent.
func gemma3MetadataEntry(entries []basegguf.MetadataEntry, key string) (basegguf.MetadataEntry, bool) {
	for _, e := range entries {
		if e.Key == key {
			return e, true
		}
	}
	return basegguf.MetadataEntry{}, false
}

func TestGemma3Metadata_gemma3Metadata_Good(t *testing.T) {
	entries, err := gemma3Metadata([]byte(gemma3TestConfig), 15, "LEM-Gemma3-1B")
	if err != nil {
		t.Fatalf("gemma3Metadata: %v", err)
	}
	wantU32 := map[string]uint32{
		"gemma3.block_count":              26,
		"gemma3.context_length":           32768,
		"gemma3.embedding_length":         1152,
		"gemma3.feed_forward_length":      6912,
		"gemma3.attention.head_count":     4,
		"gemma3.attention.head_count_kv":  1,
		"gemma3.attention.key_length":     256,
		"gemma3.attention.value_length":   256,
		"gemma3.attention.sliding_window": 512,
	}
	for key, want := range wantU32 {
		e, ok := gemma3MetadataEntry(entries, key)
		if !ok {
			t.Errorf("missing key %q", key)
			continue
		}
		if got, _ := e.Value.(uint32); got != want {
			t.Errorf("%s = %d, want %d", key, got, want)
		}
	}
	if e, ok := gemma3MetadataEntry(entries, "general.architecture"); !ok || e.Value.(string) != "gemma3" {
		t.Errorf("general.architecture = %v, want gemma3", e.Value)
	}
	if e, ok := gemma3MetadataEntry(entries, "gemma3.rope.freq_base"); !ok || e.Value.(float32) != 1000000 {
		t.Errorf("rope.freq_base = %v, want 1000000", e.Value)
	}
	if e, ok := gemma3MetadataEntry(entries, "gemma3.rope.freq_base_swa"); !ok || e.Value.(float32) != 10000 {
		t.Errorf("rope.freq_base_swa = %v, want 10000", e.Value)
	}
	if e, ok := gemma3MetadataEntry(entries, "general.name"); !ok || e.Value.(string) != "LEM-Gemma3-1B" {
		t.Errorf("general.name = %v, want LEM-Gemma3-1B", e.Value)
	}
}

func TestGemma3Metadata_gemma3Metadata_Bad(t *testing.T) {
	// Missing num_hidden_layers — a header of zeros would mis-build llama.cpp's
	// graph, so this must be a loud error.
	const noLayers = `{"model_type":"gemma3_text","hidden_size":1152,"num_attention_heads":4,"intermediate_size":6912,"head_dim":256}`
	if _, err := gemma3Metadata([]byte(noLayers), 15, ""); err == nil {
		t.Fatal("gemma3Metadata accepted config with no num_hidden_layers, want error")
	}
}

func TestGemma3Metadata_gemma3Metadata_Ugly(t *testing.T) {
	// head_dim absent — the attention key/value length would silently be zero.
	const noHeadDim = `{"model_type":"gemma3_text","num_hidden_layers":26,"hidden_size":1152,"num_attention_heads":4,"intermediate_size":6912}`
	if _, err := gemma3Metadata([]byte(noHeadDim), 15, ""); err == nil {
		t.Fatal("gemma3Metadata accepted config with no head_dim, want error")
	}
}

func TestGemma3Metadata_gemma3Metadata_NoName(t *testing.T) {
	entries, err := gemma3Metadata([]byte(gemma3TestConfig), 15, "")
	if err != nil {
		t.Fatalf("gemma3Metadata: %v", err)
	}
	if _, ok := gemma3MetadataEntry(entries, "general.name"); ok {
		t.Error("general.name present with empty modelName, want omitted")
	}
}

func TestGemma3Metadata_gemma3Metadata_GlobalOnlyOmitsSliding(t *testing.T) {
	// sliding_window_pattern == 1 (every layer global) → no sliding_window key,
	// and no rope_local_base_freq → no freq_base_swa key.
	const global = `{"model_type":"gemma3_text","num_hidden_layers":4,"hidden_size":1152,"num_attention_heads":4,"intermediate_size":6912,"head_dim":256,"sliding_window":512,"sliding_window_pattern":1}`
	entries, err := gemma3Metadata([]byte(global), 15, "")
	if err != nil {
		t.Fatalf("gemma3Metadata: %v", err)
	}
	if _, ok := gemma3MetadataEntry(entries, "gemma3.attention.sliding_window"); ok {
		t.Error("sliding_window emitted for an all-global model, want omitted")
	}
	if _, ok := gemma3MetadataEntry(entries, "gemma3.rope.freq_base_swa"); ok {
		t.Error("rope.freq_base_swa emitted without rope_local_base_freq, want omitted")
	}
}

func TestGemma3Metadata_gemma3FileType(t *testing.T) {
	cases := []struct {
		format basegguf.QuantizeFormat
		want   uint32
	}{
		{basegguf.QuantizeQ4_K_M, 15},
		{basegguf.QuantizeQ8_0, 7},
		{basegguf.QuantizeQ6_K, 18},
		{basegguf.QuantizeQ5_K_M, 17},
		{basegguf.QuantizeQ3_K_M, 12},
		{basegguf.QuantizeQ2_K_M, 10},
	}
	for _, c := range cases {
		if got := gemma3FileType(c.format); got != c.want {
			t.Errorf("gemma3FileType(%s) = %d, want %d", c.format, got, c.want)
		}
	}
}
