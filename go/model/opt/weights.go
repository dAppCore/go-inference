// SPDX-Licence-Identifier: EUPL-1.2

package opt

import "dappco.re/go/inference/model"

// WeightNames maps Hugging Face OPT tensors onto the neutral assembler roles.
func WeightNames() model.WeightNames {
	return model.WeightNames{
		Embed: "model.decoder.embed_tokens", PositionEmbed: "model.decoder.embed_positions",
		EmbedProjectionIn: "model.decoder.project_in", EmbedProjectionOut: "model.decoder.project_out",
		FinalNorm: "model.decoder.final_layer_norm.weight", LayerPrefix: "model.decoder.layers.%d",
		AttnNorm: ".self_attn_layer_norm.weight", Q: ".self_attn.q_proj", K: ".self_attn.k_proj",
		V: ".self_attn.v_proj", O: ".self_attn.out_proj", MLPNorm: ".final_layer_norm.weight",
		Gate: ".fc1", Up: ".fc1", Down: ".fc2",
	}
}
