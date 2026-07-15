// SPDX-Licence-Identifier: EUPL-1.2

package composed

import (
	"testing"

	core "dappco.re/go"
)

// hybridLayerTypesJSON builds a JSON array of n layer types in the Qwen 3.6 3:1 schedule (every
// interval-th layer is full_attention, the rest linear_attention) — the real checkpoints' layer_types.
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
// qwen3_5_text + a sibling vision_config): a dense-FFN 3:1 GatedDeltaNet/full-attention hybrid with gated
// attention (output_gate_type swish), head_dim 256, kv heads 4, and rope under rope_parameters.
func qwen35_27B() []byte {
	return []byte(core.Sprintf(`{
		"architectures": ["Qwen3_5ForConditionalGeneration"],
		"image_token_id": 248056,
		"model_type": "qwen3_5",
		"tie_word_embeddings": false,
		"text_config": {
			"attn_output_gate": true,
			"bos_token_id": 248044,
			"eos_token_id": 248044,
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
			"max_position_embeddings": 262144,
			"model_type": "qwen3_5_text",
			"mtp_num_hidden_layers": 1,
			"num_attention_heads": 24,
			"num_hidden_layers": 64,
			"num_key_value_heads": 4,
			"output_gate_type": "swish",
			"partial_rotary_factor": 0.25,
			"rms_norm_eps": 1e-06,
			"rope_parameters": {
				"mrope_interleaved": true,
				"mrope_section": [11, 11, 10],
				"partial_rotary_factor": 0.25,
				"rope_theta": 10000000,
				"rope_type": "default"
			},
			"tie_word_embeddings": false,
			"vocab_size": 248320
		},
		"vision_config": {
			"depth": 27,
			"hidden_size": 1152,
			"model_type": "qwen3_5",
			"out_hidden_size": 5120,
			"patch_size": 16
		}
	}`, hybridLayerTypesJSON(64, 4)))
}

// qwen35_35B_A3B is the real Qwen/Qwen3.6-35B-A3B config.json shape (model_type qwen3_5_moe, nested
// text_config qwen3_5_moe_text): the MoE variant — 256 experts top-8, moe_intermediate_size 512, a shared
// expert (512), kv heads 2, linear value heads 32. It omits output_gate_type and norm_topk_prob (both
// default).
func qwen35_35B_A3B() []byte {
	return []byte(core.Sprintf(`{
		"architectures": ["Qwen3_5MoeForConditionalGeneration"],
		"image_token_id": 248056,
		"model_type": "qwen3_5_moe",
		"tie_word_embeddings": false,
		"text_config": {
			"attn_output_gate": true,
			"bos_token_id": 248044,
			"eos_token_id": 248044,
			"full_attention_interval": 4,
			"head_dim": 256,
			"hidden_size": 2048,
			"layer_types": %s,
			"linear_conv_kernel_dim": 4,
			"linear_key_head_dim": 128,
			"linear_num_key_heads": 16,
			"linear_num_value_heads": 32,
			"linear_value_head_dim": 128,
			"max_position_embeddings": 262144,
			"model_type": "qwen3_5_moe_text",
			"moe_intermediate_size": 512,
			"mtp_num_hidden_layers": 1,
			"num_attention_heads": 16,
			"num_experts": 256,
			"num_experts_per_tok": 8,
			"num_hidden_layers": 40,
			"num_key_value_heads": 2,
			"partial_rotary_factor": 0.25,
			"rms_norm_eps": 1e-06,
			"rope_parameters": {
				"mrope_interleaved": true,
				"mrope_section": [11, 11, 10],
				"partial_rotary_factor": 0.25,
				"rope_theta": 10000000,
				"rope_type": "default"
			},
			"shared_expert_intermediate_size": 512,
			"tie_word_embeddings": false,
			"vocab_size": 248320
		},
		"vision_config": {
			"depth": 27,
			"hidden_size": 1152,
			"model_type": "qwen3_5_moe",
			"out_hidden_size": 2048,
			"patch_size": 16
		}
	}`, hybridLayerTypesJSON(40, 4)))
}

// TestParseHybridConfig is the table-driven config-parse test over the two real Qwen 3.6 config.json
// shapes (dense 27B, MoE 35B-A3B). It asserts the wrapper→text_config resolution (effective()), the gated
// attention flags, the linear_attention geometry, rope-under-rope_parameters (theta + mrope), the MoE
// fields, and the token ids — every field the composed loader consumes or validates.
func TestParseHybridConfig(t *testing.T) {
	type want struct {
		attnGate                                          bool
		outputGateType                                    string
		headDim, kvHeads, heads, layers, fullAttnInterval int
		linearKeyHeads, linearValueHeads, convKernel      int
		vocab, bos, eos, mtp                              int
		numExperts, topK, moeInter, sharedInter           int
		ropeTheta, partialRotary                          float32
		mropeInterleaved, normTopK                        bool
		mropeSection                                      []int
		visionOut                                         int
	}
	cases := []struct {
		name string
		cfg  []byte
		want want
	}{
		{
			name: "qwen3_5_dense_27B",
			cfg:  qwen35_27B(),
			want: want{
				attnGate: true, outputGateType: "swish",
				headDim: 256, kvHeads: 4, heads: 24, layers: 64, fullAttnInterval: 4,
				linearKeyHeads: 16, linearValueHeads: 48, convKernel: 4,
				vocab: 248320, bos: 248044, eos: 248044, mtp: 1,
				numExperts: 0, topK: 0, moeInter: 0, sharedInter: 0,
				ropeTheta: 1e7, partialRotary: 0.25,
				mropeInterleaved: true, normTopK: true, mropeSection: []int{11, 11, 10},
				visionOut: 5120,
			},
		},
		{
			name: "qwen3_5_moe_35B_A3B",
			cfg:  qwen35_35B_A3B(),
			want: want{
				attnGate: true, outputGateType: "",
				headDim: 256, kvHeads: 2, heads: 16, layers: 40, fullAttnInterval: 4,
				linearKeyHeads: 16, linearValueHeads: 32, convKernel: 4,
				vocab: 248320, bos: 248044, eos: 248044, mtp: 1,
				numExperts: 256, topK: 8, moeInter: 512, sharedInter: 512,
				ropeTheta: 1e7, partialRotary: 0.25,
				mropeInterleaved: true, normTopK: true, mropeSection: []int{11, 11, 10},
				visionOut: 2048,
			},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var raw loaderConfig
			if r := core.JSONUnmarshal(tc.cfg, &raw); !r.OK {
				t.Fatalf("parse failed")
			}
			eff := raw.effective()
			w := tc.want
			if eff == &raw {
				t.Fatal("effective() must resolve to the nested text_config, not the wrapper")
			}
			if eff.AttnOutputGate != w.attnGate {
				t.Errorf("attn_output_gate = %v, want %v", eff.AttnOutputGate, w.attnGate)
			}
			if eff.OutputGateType != w.outputGateType {
				t.Errorf("output_gate_type = %q, want %q", eff.OutputGateType, w.outputGateType)
			}
			if eff.HeadDim != w.headDim || eff.NumKeyValueHeads != w.kvHeads || eff.NumAttentionHeads != w.heads {
				t.Errorf("attn geometry head_dim/kv/heads = %d/%d/%d, want %d/%d/%d",
					eff.HeadDim, eff.NumKeyValueHeads, eff.NumAttentionHeads, w.headDim, w.kvHeads, w.heads)
			}
			if eff.NumHiddenLayers != w.layers || len(eff.LayerTypes) != w.layers {
				t.Errorf("layers = %d, layer_types len = %d, want both %d", eff.NumHiddenLayers, len(eff.LayerTypes), w.layers)
			}
			if eff.FullAttentionInterval != w.fullAttnInterval {
				t.Errorf("full_attention_interval = %d, want %d", eff.FullAttentionInterval, w.fullAttnInterval)
			}
			if eff.LinearNumKeyHeads != w.linearKeyHeads || eff.LinearNumValueHeads != w.linearValueHeads {
				t.Errorf("linear key/value heads = %d/%d, want %d/%d",
					eff.LinearNumKeyHeads, eff.LinearNumValueHeads, w.linearKeyHeads, w.linearValueHeads)
			}
			if eff.LinearKeyHeadDim != 128 || eff.LinearValueHeadDim != 128 || eff.LinearConvKernelDim != w.convKernel {
				t.Errorf("linear key/value head dim + conv kernel = %d/%d/%d, want 128/128/%d",
					eff.LinearKeyHeadDim, eff.LinearValueHeadDim, eff.LinearConvKernelDim, w.convKernel)
			}
			if eff.VocabSize != w.vocab || int(eff.BosTokenID) != w.bos || int(eff.EosTokenID) != w.eos {
				t.Errorf("vocab/bos/eos = %d/%d/%d, want %d/%d/%d", eff.VocabSize, int(eff.BosTokenID), int(eff.EosTokenID), w.vocab, w.bos, w.eos)
			}
			if eff.MTPNumHiddenLayers != w.mtp {
				t.Errorf("mtp_num_hidden_layers = %d, want %d", eff.MTPNumHiddenLayers, w.mtp)
			}
			if eff.NumExperts != w.numExperts || eff.NumExpertsPerTok != w.topK ||
				eff.MoEIntermediateSize != w.moeInter || eff.SharedExpertIntermediateSize != w.sharedInter {
				t.Errorf("moe experts/topk/inter/shared = %d/%d/%d/%d, want %d/%d/%d/%d",
					eff.NumExperts, eff.NumExpertsPerTok, eff.MoEIntermediateSize, eff.SharedExpertIntermediateSize,
					w.numExperts, w.topK, w.moeInter, w.sharedInter)
			}
			if eff.ropeTheta() != w.ropeTheta {
				t.Errorf("ropeTheta() = %v, want %v (from rope_parameters, no flat key)", eff.ropeTheta(), w.ropeTheta)
			}
			if eff.partialRotary() != w.partialRotary {
				t.Errorf("partialRotary() = %v, want %v", eff.partialRotary(), w.partialRotary)
			}
			if eff.RopeParameters == nil || eff.RopeParameters.MRopeInterleaved != w.mropeInterleaved {
				t.Errorf("mrope_interleaved missing/wrong, want %v", w.mropeInterleaved)
			}
			if eff.RopeParameters == nil || len(eff.RopeParameters.MRopeSection) != 3 ||
				eff.RopeParameters.MRopeSection[0] != w.mropeSection[0] ||
				eff.RopeParameters.MRopeSection[1] != w.mropeSection[1] ||
				eff.RopeParameters.MRopeSection[2] != w.mropeSection[2] {
				t.Errorf("mrope_section = %v, want %v", eff.RopeParameters, w.mropeSection)
			}
			if eff.normTopKProb() != w.normTopK {
				t.Errorf("normTopKProb() = %v, want %v (absent ⇒ default true)", eff.normTopKProb(), w.normTopK)
			}
			if raw.VisionConfig == nil || raw.VisionConfig.OutHiddenSize != w.visionOut {
				t.Errorf("vision_config carried wrong: %v, want out_hidden_size %d", raw.VisionConfig, w.visionOut)
			}
		})
	}
}

// TestNormTopKProbExplicitFalse pins that an explicit norm_topk_prob:false is honoured (the pointer field
// distinguishes an absent key — default true — from an explicit false), so the routing behaviour is
// config-driven rather than assumed.
func TestNormTopKProbExplicitFalse(t *testing.T) {
	var c loaderConfig
	if r := core.JSONUnmarshal([]byte(`{"norm_topk_prob": false}`), &c); !r.OK {
		t.Fatal("parse failed")
	}
	if c.normTopKProb() {
		t.Fatal("explicit norm_topk_prob:false must resolve to false")
	}
	var absent loaderConfig
	if r := core.JSONUnmarshal([]byte(`{}`), &absent); !r.OK {
		t.Fatal("parse failed")
	}
	if !absent.normTopKProb() {
		t.Fatal("absent norm_topk_prob must default to true")
	}
}
