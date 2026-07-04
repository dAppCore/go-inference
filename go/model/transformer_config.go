// SPDX-Licence-Identifier: EUPL-1.2

package model

// TransformerConfig is the neutral transformer core EVERY model architecture's config embeds — the
// shared dimensions the engine reads regardless of architecture family.
// It lives at the pkg/model root, not in any arch package: an arch's config struct embeds it and adds
// only its own fields, so the core is defined once and never re-rolled per architecture. The JSON tags
// drive config.json parsing for the embedded fields. (Copied from pkg/metal's shared TransformerConfig.)
type TransformerConfig struct {
	ModelType             string  `json:"model_type"`
	HiddenSize            int32   `json:"hidden_size"`
	NumHiddenLayers       int32   `json:"num_hidden_layers"`
	IntermediateSize      int32   `json:"intermediate_size"`
	NumAttentionHeads     int32   `json:"num_attention_heads"`
	NumKeyValueHeads      int32   `json:"num_key_value_heads"`
	HeadDim               int32   `json:"head_dim"`
	VocabSize             int32   `json:"vocab_size"`
	RMSNormEps            float32 `json:"rms_norm_eps"`
	MaxPositionEmbeddings int32   `json:"max_position_embeddings"`
}
