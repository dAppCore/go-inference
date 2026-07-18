// SPDX-Licence-Identifier: EUPL-1.2

package qwen35

import (
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference/model"
)

// hybridLayerTypesJSON builds a JSON array of n layer types in the Qwen 3.6 3:1 schedule (every interval-th
// layer full_attention, the rest linear_attention) — the real checkpoints' layer_types.
func hybridLayerTypesJSON(n, interval int) string {
	s := "["
	for i := range n {
		if i > 0 {
			s += ","
		}
		if (i+1)%interval == 0 {
			s += `"full_attention"`
		} else {
			s += `"linear_attention"`
		}
	}
	return s + "]"
}

// qwen35_27B is the real Qwen/Qwen3.6-27B config.json shape (model_type qwen3_5, nested text_config
// qwen3_5_text): a dense-FFN 3:1 GatedDeltaNet/full-attention hybrid with gated attention, head_dim 256,
// kv heads 4, partial rotary 0.25, and rope under rope_parameters.
func qwen35_27B() []byte {
	return []byte(core.Sprintf(`{
		"architectures": ["Qwen3_5ForConditionalGeneration"],
		"model_type": "qwen3_5",
		"tie_word_embeddings": false,
		"text_config": {
			"attn_output_gate": true,
			"full_attention_interval": 4,
			"head_dim": 256,
			"hidden_size": 5120,
			"intermediate_size": 17408,
			"layer_types": %s,
			"linear_conv_kernel_dim": 4,
			"linear_key_head_dim": 128,
			"linear_num_key_heads": 16,
			"linear_num_value_heads": 48,
			"linear_value_head_dim": 128,
			"model_type": "qwen3_5_text",
			"num_attention_heads": 24,
			"num_hidden_layers": 64,
			"num_key_value_heads": 4,
			"output_gate_type": "swish",
			"partial_rotary_factor": 0.25,
			"rms_norm_eps": 1e-06,
			"rope_parameters": { "partial_rotary_factor": 0.25, "rope_theta": 10000000 },
			"tie_word_embeddings": false,
			"vocab_size": 248320
		}
	}`, hybridLayerTypesJSON(64, 4)))
}

// qwen35_35B_A3B is the real Qwen/Qwen3.6-35B-A3B config.json shape (model_type qwen3_5_moe, nested
// text_config qwen3_5_moe_text): the MoE variant — 256 experts top-8, moe_intermediate_size 512, one shared
// expert (512), head_dim 256, kv heads 2. It omits norm_topk_prob (defaults true).
func qwen35_35B_A3B() []byte {
	return []byte(core.Sprintf(`{
		"architectures": ["Qwen3_5MoeForConditionalGeneration"],
		"model_type": "qwen3_5_moe",
		"tie_word_embeddings": false,
		"text_config": {
			"attn_output_gate": true,
			"full_attention_interval": 4,
			"head_dim": 256,
			"hidden_size": 2048,
			"layer_types": %s,
			"linear_conv_kernel_dim": 4,
			"linear_key_head_dim": 128,
			"linear_num_key_heads": 16,
			"linear_num_value_heads": 32,
			"linear_value_head_dim": 128,
			"model_type": "qwen3_5_moe_text",
			"moe_intermediate_size": 512,
			"num_attention_heads": 16,
			"num_experts": 256,
			"num_experts_per_tok": 8,
			"num_hidden_layers": 40,
			"num_key_value_heads": 2,
			"partial_rotary_factor": 0.25,
			"rms_norm_eps": 1e-06,
			"rope_parameters": { "partial_rotary_factor": 0.25, "rope_theta": 10000000 },
			"shared_expert_intermediate_size": 512,
			"tie_word_embeddings": false,
			"vocab_size": 248320
		}
	}`, hybridLayerTypesJSON(40, 4)))
}

// TestConfigArch is the table-driven Arch() test over the two real Qwen 3.6 config shapes. It asserts the
// hybrid schedule (every full_attention_interval-th layer is MixerAttention with a KV cache + resolved
// geometry, the rest MixerGatedDelta with no cache), the MoE-per-layer flag, and the neutral Arch
// declarations (experts/top-k/shared-expert, SiLU activation, gated attention, partial rotary, rope theta).
func TestConfigArch(t *testing.T) {
	cases := []struct {
		name                                              string
		cfg                                               []byte
		layers, interval, heads, kvHeads, headDim, vocab  int
		experts, topK, expertFF, sharedExperts, rotaryDim int
		moe                                               bool
	}{
		{
			name: "qwen3_5_dense_27B", cfg: qwen35_27B(),
			layers: 64, interval: 4, heads: 24, kvHeads: 4, headDim: 256, vocab: 248320,
			experts: 0, topK: 0, expertFF: 0, sharedExperts: 0, rotaryDim: 64, moe: false,
		},
		{
			name: "qwen3_5_moe_35B_A3B", cfg: qwen35_35B_A3B(),
			layers: 40, interval: 4, heads: 16, kvHeads: 2, headDim: 256, vocab: 248320,
			experts: 256, topK: 8, expertFF: 512, sharedExperts: 1, rotaryDim: 64, moe: true,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var cfg Config
			if r := core.JSONUnmarshal(tc.cfg, &cfg); !r.OK {
				t.Fatalf("config.json parse failed")
			}
			arch, err := cfg.Arch()
			if err != nil {
				t.Fatalf("Arch() error: %v", err)
			}

			// Neutral dims + declarations.
			if arch.Hidden == 0 || arch.Heads != tc.heads || arch.KVHeads != tc.kvHeads || arch.HeadDim != tc.headDim {
				t.Errorf("dims heads/kv/headDim = %d/%d/%d, want %d/%d/%d", arch.Heads, arch.KVHeads, arch.HeadDim, tc.heads, tc.kvHeads, tc.headDim)
			}
			if arch.Vocab != tc.vocab {
				t.Errorf("vocab = %d, want %d", arch.Vocab, tc.vocab)
			}
			if arch.RotaryDim != tc.rotaryDim || arch.RotaryDimLocal != tc.rotaryDim {
				t.Errorf("rotaryDim = %d/%d, want %d (head_dim %d * partial 0.25)", arch.RotaryDim, arch.RotaryDimLocal, tc.rotaryDim, tc.headDim)
			}
			if arch.RopeBase != 1e7 {
				t.Errorf("ropeBase = %g, want 1e7 (rope_parameters.rope_theta)", arch.RopeBase)
			}
			if arch.Activation != "silu" {
				t.Errorf("activation = %q, want silu", arch.Activation)
			}
			if !arch.AttnOutputGate {
				t.Error("AttnOutputGate = false, want true (attn_output_gate)")
			}
			if arch.EmbedScale != 1 {
				t.Errorf("embedScale = %g, want 1 (qwen is llama-family, not gemma)", arch.EmbedScale)
			}

			// MoE declarations.
			if arch.Experts != tc.experts || arch.TopK != tc.topK || arch.ExpertFF != tc.expertFF || arch.SharedExperts != tc.sharedExperts {
				t.Errorf("moe experts/topK/expertFF/shared = %d/%d/%d/%d, want %d/%d/%d/%d",
					arch.Experts, arch.TopK, arch.ExpertFF, arch.SharedExperts, tc.experts, tc.topK, tc.expertFF, tc.sharedExperts)
			}
			if tc.moe {
				if arch.MoEGating != model.MoEGatingSoftmax {
					t.Errorf("moeGating = %q, want softmax", arch.MoEGating)
				}
				if !arch.NormaliseMoETopK {
					t.Error("NormaliseMoETopK = false, want true (norm_topk_prob absent ⇒ default true)")
				}
			}

			// Per-layer hybrid schedule: interval-th layer is gated full attention, the rest gated-delta.
			if len(arch.Layer) != tc.layers {
				t.Fatalf("layers = %d, want %d", len(arch.Layer), tc.layers)
			}
			cacheOwners := 0
			for i, l := range arch.Layer {
				wantFull := (i+1)%tc.interval == 0
				if wantFull {
					if l.Mixer != model.MixerAttention {
						t.Errorf("layer %d: mixer = %v, want MixerAttention (full_attention)", i, l.Mixer)
					}
					if l.HeadDim != tc.headDim || l.KVHeads != tc.kvHeads {
						t.Errorf("layer %d: attn geometry headDim/kv = %d/%d, want %d/%d", i, l.HeadDim, l.KVHeads, tc.headDim, tc.kvHeads)
					}
					if l.CacheIndex < 0 {
						t.Errorf("layer %d: full-attention layer owns no KV cache slot", i)
					} else {
						cacheOwners++
					}
				} else {
					if l.Mixer != model.MixerGatedDelta {
						t.Errorf("layer %d: mixer = %v, want MixerGatedDelta (linear_attention)", i, l.Mixer)
					}
					if l.CacheIndex != -1 {
						t.Errorf("layer %d: gated-delta layer holds KV cache slot %d, want none (-1)", i, l.CacheIndex)
					}
				}
				if l.MoE != tc.moe {
					t.Errorf("layer %d: MoE = %v, want %v", i, l.MoE, tc.moe)
				}
			}
			if wantOwners := tc.layers / tc.interval; cacheOwners != wantOwners {
				t.Errorf("full-attention cache owners = %d, want %d", cacheOwners, wantOwners)
			}
		})
	}
}
