// SPDX-Licence-Identifier: EUPL-1.2

package dbrx

import (
	"testing"

	"dappco.re/go/inference/model"
)

// Databricks DBRX Instruct config fixture:
// https://huggingface.co/databricks/dbrx-instruct/blob/main/config.json
const instructConfig = `{"model_type":"dbrx","d_model":6144,"n_heads":48,"n_layers":40,"vocab_size":100352,"tie_word_embeddings":false,"attn_config":{"clip_qkv":8,"kv_n_heads":8,"rope_theta":500000},"ffn_config":{"ffn_hidden_size":10752,"moe_num_experts":16,"moe_top_k":4}}`

func TestConfig_Arch_Good(t *testing.T) {
	var cfg Config
	if r := coreJSONUnmarshal([]byte(instructConfig), &cfg); !r {
		t.Fatal("real DBRX config did not parse")
	}
	arch, err := cfg.Arch()
	if err != nil {
		t.Fatalf("Arch: %v", err)
	}
	if arch.Hidden != 6144 || arch.HeadDim != 128 || arch.KVHeads != 8 || arch.Experts != 16 || arch.TopK != 4 || arch.ExpertFF != 10752 {
		t.Fatalf("DBRX geometry = %+v", arch)
	}
	if arch.MoEGating != model.MoEGatingSoftmax || arch.NormaliseMoETopK || arch.SharedExperts != 0 || !arch.HasMoE() {
		t.Fatalf("DBRX routing = gating %q normalise %v shared %d MoE %v", arch.MoEGating, arch.NormaliseMoETopK, arch.SharedExperts, arch.HasMoE())
	}
	if !arch.LayerNorm || arch.QKVClip != 8 {
		t.Fatalf("attention quirks = layer norm %v qkv clip %g", arch.LayerNorm, arch.QKVClip)
	}
}

func TestConfig_Arch_Bad(t *testing.T) {
	if _, err := (Config{DModel: 7, Heads: 2, Layers: 1, VocabSize: 8, Attention: AttentionConfig{KVHeads: 1}, FFN: FFNConfig{HiddenSize: 4, Experts: 2, TopK: 1}}).Arch(); err == nil {
		t.Fatal("indivisible hidden size accepted")
	}
}

func TestConfig_Arch_Ugly(t *testing.T) {
	if _, err := (Config{}).Arch(); err == nil {
		t.Fatal("empty config accepted")
	}
}

func TestConfig_InferFromWeights_Good(t *testing.T) {
	cfg := Config{DModel: 8}
	cfg.InferFromWeights(nil)
	if cfg.DModel != 8 {
		t.Fatalf("InferFromWeights changed config: %+v", cfg)
	}
}

func TestConfig_InferFromWeights_Bad(t *testing.T) {
	cfg := Config{}
	cfg.InferFromWeights(nil)
	if _, err := cfg.Arch(); err == nil {
		t.Fatal("empty config became valid after InferFromWeights")
	}
}

// TestConfig_InferFromWeights_Ugly proves the no-op does not paper over the
// moe_top_k-exceeds-moe_num_experts guard — distinct from _Bad's all-zero
// rejection.
func TestConfig_InferFromWeights_Ugly(t *testing.T) {
	cfg := Config{DModel: 8, Heads: 2, Layers: 1, VocabSize: 8, Attention: AttentionConfig{KVHeads: 1}, FFN: FFNConfig{HiddenSize: 4, Experts: 2, TopK: 3}}
	cfg.InferFromWeights(nil)
	if _, err := cfg.Arch(); err == nil {
		t.Fatal("moe_top_k exceeding moe_num_experts became valid after InferFromWeights")
	}
}
