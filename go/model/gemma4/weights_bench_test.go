// SPDX-Licence-Identifier: EUPL-1.2

package gemma4

import (
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference/model/safetensors"
)

// benchRawWeights builds a realistic mixed checkpoint tensor set: an audio Conformer tower, a
// SigLIP vision tower, and a text backbone at the given per-tower layer counts. Only the tensor
// NAMES matter to the sanitise walks (they key the output map by canonical name and copy the
// tensor struct — a slice-header copy, no payload), so the tensors carry no data. This is the
// per-load input the SanitizeAudioWeights / SanitizeVisionWeights canonicalise walks consume.
func benchRawWeights(audioLayers, visionLayers, textLayers int) map[string]safetensors.Tensor {
	raw := make(map[string]safetensors.Tensor)
	add := func(name string) { raw[name] = safetensors.Tensor{Dtype: "BF16"} }

	add("embed_audio.embedding_projection.weight")
	for i := range audioLayers {
		b := core.Sprintf("audio_tower.layers.%d.", i)
		for _, n := range []string{
			"self_attn.q_proj.linear.weight", "self_attn.k_proj.linear.weight",
			"self_attn.v_proj.linear.weight", "self_attn.post.linear.weight",
			"self_attn.relative_k_proj.weight", "self_attn.per_dim_scale",
			"lconv1d.linear_start.linear.weight", "lconv1d.linear_end.linear.weight",
			"lconv1d.depthwise_conv1d.weight", "lconv1d.pre_layer_norm.weight",
			"norm_pre_attn.weight", "norm_post_attn.weight", "norm_out.weight",
		} {
			add(b + n)
		}
	}

	add("multi_modal_projector.proj.weight")
	for i := range visionLayers {
		b := core.Sprintf("vision_tower.encoder.layers.%d.", i)
		for _, n := range []string{
			"self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight",
			"self_attn.o_proj.weight", "input_layernorm.weight", "post_attention_layernorm.weight",
			"pre_feedforward_layernorm.weight", "post_feedforward_layernorm.weight",
			"mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight",
		} {
			add(b + n)
		}
	}

	add("model.embed_tokens.weight")
	add("model.norm.weight")
	for i := range textLayers {
		b := core.Sprintf("model.layers.%d.", i)
		for _, n := range []string{
			"self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight",
			"self_attn.o_proj.weight", "input_layernorm.weight", "post_attention_layernorm.weight",
			"mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight",
		} {
			add(b + n)
		}
	}
	return raw
}

// BenchmarkSanitizeAudioWeights measures the per-load audio canonicalise walk: every raw tensor
// name is wrapper-trimmed and prefix-tested, and the audio_tower./embed_audio. matches keyed into
// a fresh map. Realistic tower sizes (12 audio / 27 vision / 48 text layers).
func BenchmarkSanitizeAudioWeights(b *testing.B) {
	raw := benchRawWeights(12, 27, 48)
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		_ = SanitizeAudioWeights(raw)
	}
}

// BenchmarkSanitizeVisionWeights measures the per-load vision canonicalise walk over the same
// realistic mixed checkpoint.
func BenchmarkSanitizeVisionWeights(b *testing.B) {
	raw := benchRawWeights(12, 27, 48)
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		_ = SanitizeVisionWeights(raw)
	}
}
