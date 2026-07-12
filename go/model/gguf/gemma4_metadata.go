// SPDX-Licence-Identifier: EUPL-1.2

package gguf

import core "dappco.re/go"

// gemma4Arch is the GGUF general.architecture value (and the metadata key
// prefix) llama.cpp dispatches the gemma-4 text graph on.
const gemma4Arch = "gemma4"

// gemma4Config holds the gemma-4 text-stack hyperparameters the GGUF header
// needs, decoded from the checkpoint's config.json. Only the text_config
// fields the graph reads are declared; the audio/vision towers and cosmetic
// fields are ignored (a text GGUF carries the language model only).
type gemma4Config struct {
	TextConfig struct {
		NumHiddenLayers         int      `json:"num_hidden_layers"`
		MaxPositionEmbeddings   int      `json:"max_position_embeddings"`
		HiddenSize              int      `json:"hidden_size"`
		IntermediateSize        int      `json:"intermediate_size"`
		NumAttentionHeads       int      `json:"num_attention_heads"`
		NumKeyValueHeads        int      `json:"num_key_value_heads"`
		RMSNormEps              float32  `json:"rms_norm_eps"`
		GlobalHeadDim           int      `json:"global_head_dim"`
		HeadDim                 int      `json:"head_dim"`
		FinalLogitSoftcapping   float32  `json:"final_logit_softcapping"`
		SlidingWindow           int      `json:"sliding_window"`
		NumKVSharedLayers       int      `json:"num_kv_shared_layers"`
		HiddenSizePerLayerInput int      `json:"hidden_size_per_layer_input"`
		LayerTypes              []string `json:"layer_types"`
		RopeParameters          struct {
			FullAttention struct {
				RopeTheta float32 `json:"rope_theta"`
			} `json:"full_attention"`
			SlidingAttention struct {
				RopeTheta float32 `json:"rope_theta"`
			} `json:"sliding_attention"`
		} `json:"rope_parameters"`
	} `json:"text_config"`
}

// gemma4Metadata builds the general.* and gemma4.* header metadata for a
// gemma-4 text GGUF from the checkpoint's config.json, mirroring the unsloth
// gemma-4-E2B-it-Q4_K_M oracle's hyperparameter key set. Every value is derived
// from config.json; the cosmetic keys the oracle also carries (general.name
// details, licence, base-model provenance, imatrix stats, default sampling
// params from generation_config.json) are omitted — none are read by the graph
// builder. modelName, when non-empty, is written as general.name.
//
// The tokenizer.ggml.* set is built separately (gemma4Tokenizer); this covers
// the architecture hyperparameters only.
func gemma4Metadata(configJSON []byte, fileType uint32, modelName string) ([]MetadataEntry, error) {
	var config gemma4Config
	if r := core.JSONUnmarshal(configJSON, &config); !r.OK {
		return nil, core.E("gemma4Metadata", "parse config.json", r.Err())
	}
	text := config.TextConfig
	if text.NumHiddenLayers <= 0 || text.HiddenSize <= 0 || text.NumAttentionHeads <= 0 {
		return nil, core.NewError("gguf: gemma4 config.json missing text_config hyperparameters (num_hidden_layers/hidden_size/num_attention_heads)")
	}
	if len(text.LayerTypes) != text.NumHiddenLayers {
		return nil, core.Errorf("gguf: gemma4 layer_types length %d != num_hidden_layers %d", len(text.LayerTypes), text.NumHiddenLayers)
	}

	// Per-layer feed_forward_length: gemma-4's MLP width is uniform, so the
	// array broadcasts intermediate_size across every block.
	feedForward := make([]int32, text.NumHiddenLayers)
	for i := range feedForward {
		feedForward[i] = int32(text.IntermediateSize)
	}
	// Sliding-window pattern: llama.cpp marks sliding-attention layers true and
	// full-attention layers false (layer 4, 9, 14, … are full attention).
	pattern := make([]bool, text.NumHiddenLayers)
	for i, kind := range text.LayerTypes {
		pattern[i] = kind == "sliding_attention"
	}

	u32 := func(key string, v int) MetadataEntry {
		return MetadataEntry{Key: key, ValueType: ValueTypeUint32, Value: uint32(v)}
	}
	f32 := func(key string, v float32) MetadataEntry {
		return MetadataEntry{Key: key, ValueType: ValueTypeFloat32, Value: v}
	}

	metadata := []MetadataEntry{
		{Key: "general.architecture", ValueType: ValueTypeString, Value: gemma4Arch},
		{Key: "general.type", ValueType: ValueTypeString, Value: "model"},
		{Key: "general.quantization_version", ValueType: ValueTypeUint32, Value: uint32(2)},
		{Key: "general.file_type", ValueType: ValueTypeUint32, Value: fileType},
	}
	if modelName != "" {
		metadata = append(metadata, MetadataEntry{Key: "general.name", ValueType: ValueTypeString, Value: modelName})
	}
	metadata = append(metadata,
		u32(gemma4Arch+".block_count", text.NumHiddenLayers),
		u32(gemma4Arch+".context_length", text.MaxPositionEmbeddings),
		u32(gemma4Arch+".embedding_length", text.HiddenSize),
		MetadataEntry{Key: gemma4Arch + ".feed_forward_length", ValueType: ggufValueTypeArray, Value: feedForward},
		u32(gemma4Arch+".attention.head_count", text.NumAttentionHeads),
		u32(gemma4Arch+".attention.head_count_kv", text.NumKeyValueHeads),
		f32(gemma4Arch+".rope.freq_base", text.RopeParameters.FullAttention.RopeTheta),
		f32(gemma4Arch+".rope.freq_base_swa", text.RopeParameters.SlidingAttention.RopeTheta),
		f32(gemma4Arch+".attention.layer_norm_rms_epsilon", text.RMSNormEps),
		u32(gemma4Arch+".attention.key_length", text.GlobalHeadDim),
		u32(gemma4Arch+".attention.value_length", text.GlobalHeadDim),
		f32(gemma4Arch+".final_logit_softcapping", text.FinalLogitSoftcapping),
		u32(gemma4Arch+".attention.sliding_window", text.SlidingWindow),
		u32(gemma4Arch+".attention.shared_kv_layers", text.NumKVSharedLayers),
		u32(gemma4Arch+".embedding_length_per_layer_input", text.HiddenSizePerLayerInput),
		MetadataEntry{Key: gemma4Arch + ".attention.sliding_window_pattern", ValueType: ggufValueTypeArray, Value: pattern},
		u32(gemma4Arch+".attention.key_length_swa", text.HeadDim),
		u32(gemma4Arch+".attention.value_length_swa", text.HeadDim),
		u32(gemma4Arch+".rope.dimension_count", text.GlobalHeadDim),
		u32(gemma4Arch+".rope.dimension_count_swa", text.HeadDim),
	)
	return metadata, nil
}
