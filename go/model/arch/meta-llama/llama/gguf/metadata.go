// SPDX-Licence-Identifier: EUPL-1.2

package gguf

import (
	core "dappco.re/go"
	basegguf "dappco.re/go/inference/model/gguf"
)

const llamaArch = "llama"

type llamaConfig struct {
	ModelType             string  `json:"model_type"`
	MaxPositionEmbeddings int     `json:"max_position_embeddings"`
	HiddenSize            int     `json:"hidden_size"`
	IntermediateSize      int     `json:"intermediate_size"`
	NumHiddenLayers       int     `json:"num_hidden_layers"`
	NumAttentionHeads     int     `json:"num_attention_heads"`
	NumKeyValueHeads      int     `json:"num_key_value_heads"`
	HeadDim               int     `json:"head_dim"`
	VocabSize             int     `json:"vocab_size"`
	RMSNormEps            float32 `json:"rms_norm_eps"`
	RopeTheta             float32 `json:"rope_theta"`
	BOSTokenID            int     `json:"bos_token_id"`
	EOSTokenID            int     `json:"eos_token_id"`
	TieWordEmbeddings     bool    `json:"tie_word_embeddings"`
}

func parseLlamaConfig(configJSON []byte) (llamaConfig, error) {
	var config llamaConfig
	if result := core.JSONUnmarshal(configJSON, &config); !result.OK {
		return config, core.E("parseLlamaConfig", "parse config.json", result.Err())
	}
	if config.ModelType != llamaArch || config.MaxPositionEmbeddings <= 0 || config.HiddenSize <= 0 || config.IntermediateSize <= 0 || config.NumHiddenLayers <= 0 || config.NumAttentionHeads <= 0 || config.VocabSize <= 0 {
		return config, core.NewError("gguf: llama config.json missing required hyperparameters")
	}
	if config.NumKeyValueHeads == 0 {
		config.NumKeyValueHeads = config.NumAttentionHeads
	}
	if config.HeadDim == 0 {
		if config.HiddenSize%config.NumAttentionHeads != 0 {
			return config, core.NewError("gguf: llama hidden_size is not divisible by num_attention_heads")
		}
		config.HeadDim = config.HiddenSize / config.NumAttentionHeads
	}
	if config.RMSNormEps == 0 {
		config.RMSNormEps = 1e-6
	}
	if config.RopeTheta == 0 {
		config.RopeTheta = 10000
	}
	return config, nil
}

func llamaMetadata(configJSON []byte, fileType uint32, modelName string) ([]basegguf.MetadataEntry, error) {
	config, err := parseLlamaConfig(configJSON)
	if err != nil {
		return nil, err
	}
	u32 := func(key string, value int) basegguf.MetadataEntry {
		return basegguf.MetadataEntry{Key: key, ValueType: basegguf.ValueTypeUint32, Value: uint32(value)}
	}
	f32 := func(key string, value float32) basegguf.MetadataEntry {
		return basegguf.MetadataEntry{Key: key, ValueType: basegguf.ValueTypeFloat32, Value: value}
	}
	metadata := []basegguf.MetadataEntry{
		{Key: "general.architecture", ValueType: basegguf.ValueTypeString, Value: llamaArch},
		{Key: "general.type", ValueType: basegguf.ValueTypeString, Value: "model"},
		{Key: "general.quantization_version", ValueType: basegguf.ValueTypeUint32, Value: uint32(2)},
		{Key: "general.file_type", ValueType: basegguf.ValueTypeUint32, Value: fileType},
	}
	if modelName != "" {
		metadata = append(metadata, basegguf.MetadataEntry{Key: "general.name", ValueType: basegguf.ValueTypeString, Value: modelName})
	}
	return append(metadata,
		u32("llama.block_count", config.NumHiddenLayers),
		u32("llama.context_length", config.MaxPositionEmbeddings),
		u32("llama.embedding_length", config.HiddenSize),
		u32("llama.feed_forward_length", config.IntermediateSize),
		u32("llama.attention.head_count", config.NumAttentionHeads),
		u32("llama.attention.head_count_kv", config.NumKeyValueHeads),
		u32("llama.rope.dimension_count", config.HeadDim),
		f32("llama.rope.freq_base", config.RopeTheta),
		f32("llama.attention.layer_norm_rms_epsilon", config.RMSNormEps),
	), nil
}

func llamaFileType(format basegguf.QuantizeFormat) uint32 {
	if format == basegguf.QuantizeQ8_0 {
		return 7
	}
	return 15
}
