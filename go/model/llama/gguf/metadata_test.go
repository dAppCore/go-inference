// SPDX-Licence-Identifier: EUPL-1.2

package gguf

import (
	"testing"

	basegguf "dappco.re/go/inference/model/gguf"
)

func TestMetadata_llamaMetadata_Good(t *testing.T) {
	config := []byte(`{"model_type":"llama","max_position_embeddings":131072,"hidden_size":2048,"intermediate_size":8192,"num_hidden_layers":16,"num_attention_heads":32,"num_key_value_heads":8,"rms_norm_eps":1e-5,"rope_theta":500000,"head_dim":64,"vocab_size":128256,"bos_token_id":128000,"eos_token_id":128001}`)
	entries, err := llamaMetadata(config, 15, "Llama-3.2-1B")
	if err != nil {
		t.Fatalf("llamaMetadata: %v", err)
	}
	want := map[string]any{
		"general.architecture":                   "llama",
		"general.file_type":                      uint32(15),
		"llama.block_count":                      uint32(16),
		"llama.context_length":                   uint32(131072),
		"llama.embedding_length":                 uint32(2048),
		"llama.feed_forward_length":              uint32(8192),
		"llama.attention.head_count":             uint32(32),
		"llama.attention.head_count_kv":          uint32(8),
		"llama.rope.dimension_count":             uint32(64),
		"llama.attention.layer_norm_rms_epsilon": float32(1e-5),
	}
	for _, entry := range entries {
		if expected, ok := want[entry.Key]; ok {
			if entry.Value != expected {
				t.Errorf("%s = %#v, want %#v", entry.Key, entry.Value, expected)
			}
			delete(want, entry.Key)
		}
	}
	if len(want) != 0 {
		t.Fatalf("missing metadata: %v", want)
	}
	_ = basegguf.ValueTypeUint32
}

func TestMetadata_llamaMetadata_Bad(t *testing.T) {
	if _, err := llamaMetadata([]byte(`{"model_type":"llama"}`), 15, "bad"); err == nil {
		t.Fatal("llamaMetadata incomplete config: want error")
	}
}

func TestMetadata_llamaMetadata_Ugly(t *testing.T) {
	if _, err := llamaMetadata([]byte(`{`), 15, "bad"); err == nil {
		t.Fatal("llamaMetadata malformed config: want error")
	}
}
