// SPDX-Licence-Identifier: EUPL-1.2

package gguf

import (
	core "dappco.re/go"
	basegguf "dappco.re/go/inference/model/gguf"
)

// gemma3Arch is the GGUF general.architecture value (and the metadata key
// prefix) llama.cpp dispatches the gemma-3 text graph on. HF spells the
// model_type "gemma3_text"; the GGUF ecosystem name is "gemma3".
const gemma3Arch = "gemma3"

// gemma3Config holds the gemma-3 text hyperparameters the GGUF header needs,
// decoded from the checkpoint's config.json. gemma3_text config.json is flat
// (no text_config nesting). Only the fields the header carries are declared;
// cosmetic and derived fields are ignored.
type gemma3Config struct {
	NumHiddenLayers       int     `json:"num_hidden_layers"`
	MaxPositionEmbeddings int     `json:"max_position_embeddings"`
	HiddenSize            int     `json:"hidden_size"`
	IntermediateSize      int     `json:"intermediate_size"`
	NumAttentionHeads     int     `json:"num_attention_heads"`
	NumKeyValueHeads      int     `json:"num_key_value_heads"`
	HeadDim               int     `json:"head_dim"`
	RMSNormEps            float32 `json:"rms_norm_eps"`
	RopeTheta             float32 `json:"rope_theta"`
	RopeLocalBaseFreq     float32 `json:"rope_local_base_freq"`
	SlidingWindow         int     `json:"sliding_window"`
	SlidingWindowPattern  int     `json:"sliding_window_pattern"`
}

// gemma3Metadata builds the general.* and gemma3.* header metadata for a
// gemma-3 text GGUF from the checkpoint's config.json, mirroring the key set a
// real llama.cpp-loadable gemma3 GGUF carries (block_count, context_length,
// embedding_length, feed_forward_length as a scalar, the attention head counts
// and key/value lengths, the global + sliding rope bases, the RMS epsilon and
// the sliding-window size). Every value is derived from config.json; the
// cosmetic keys a full checkpoint also carries (licence, provenance, sampling
// defaults) are omitted — none are read by llama.cpp's graph builder. modelName,
// when non-empty, is written as general.name.
//
// A config.json missing a load-bearing hyperparameter (layers, hidden size,
// heads, MLP width, head dim) is a loud error rather than a header of zeros —
// llama.cpp would reject or mis-build the graph from a zeroed key, so the export
// fails at the source instead.
func gemma3Metadata(configJSON []byte, fileType uint32, modelName string) ([]basegguf.MetadataEntry, error) {
	var config gemma3Config
	if r := core.JSONUnmarshal(configJSON, &config); !r.OK {
		return nil, core.E("gemma3Metadata", "parse config.json", r.Err())
	}
	if config.NumHiddenLayers <= 0 || config.HiddenSize <= 0 || config.NumAttentionHeads <= 0 {
		return nil, core.NewError("gguf: gemma3 config.json missing hyperparameters (num_hidden_layers/hidden_size/num_attention_heads)")
	}
	if config.IntermediateSize <= 0 {
		return nil, core.NewError("gguf: gemma3 config.json missing intermediate_size (feed_forward_length)")
	}
	if config.HeadDim <= 0 {
		return nil, core.NewError("gguf: gemma3 config.json missing head_dim (attention key/value length)")
	}

	u32 := func(key string, v int) basegguf.MetadataEntry {
		return basegguf.MetadataEntry{Key: key, ValueType: basegguf.ValueTypeUint32, Value: uint32(v)}
	}
	f32 := func(key string, v float32) basegguf.MetadataEntry {
		return basegguf.MetadataEntry{Key: key, ValueType: basegguf.ValueTypeFloat32, Value: v}
	}

	metadata := []basegguf.MetadataEntry{
		{Key: "general.architecture", ValueType: basegguf.ValueTypeString, Value: gemma3Arch},
		{Key: "general.type", ValueType: basegguf.ValueTypeString, Value: "model"},
		{Key: "general.quantization_version", ValueType: basegguf.ValueTypeUint32, Value: uint32(2)},
		{Key: "general.file_type", ValueType: basegguf.ValueTypeUint32, Value: fileType},
	}
	if modelName != "" {
		metadata = append(metadata, basegguf.MetadataEntry{Key: "general.name", ValueType: basegguf.ValueTypeString, Value: modelName})
	}
	metadata = append(metadata,
		u32(gemma3Arch+".block_count", config.NumHiddenLayers),
		u32(gemma3Arch+".context_length", config.MaxPositionEmbeddings),
		u32(gemma3Arch+".embedding_length", config.HiddenSize),
		u32(gemma3Arch+".feed_forward_length", config.IntermediateSize),
		u32(gemma3Arch+".attention.head_count", config.NumAttentionHeads),
		u32(gemma3Arch+".attention.head_count_kv", config.NumKeyValueHeads),
		f32(gemma3Arch+".rope.freq_base", config.RopeTheta),
		f32(gemma3Arch+".attention.layer_norm_rms_epsilon", config.RMSNormEps),
		u32(gemma3Arch+".attention.key_length", config.HeadDim),
		u32(gemma3Arch+".attention.value_length", config.HeadDim),
	)
	// The sliding rope base is only written when the checkpoint carries it —
	// gemma-3 uses a separate, smaller base for its local (sliding-window)
	// layers than for the global ones.
	if config.RopeLocalBaseFreq > 0 {
		metadata = append(metadata, f32(gemma3Arch+".rope.freq_base_swa", config.RopeLocalBaseFreq))
	}
	// llama.cpp only reads the sliding-window size when the model actually
	// interleaves sliding layers (pattern != 1, i.e. not every layer global);
	// convert_hf_to_gguf gates the key the same way.
	if config.SlidingWindow > 0 && config.SlidingWindowPattern != 1 {
		metadata = append(metadata, u32(gemma3Arch+".attention.sliding_window", config.SlidingWindow))
	}
	return metadata, nil
}

// gemma3FileType returns the GGUF general.file_type enum value for format — the
// LLAMA_FTYPE_MOSTLY_* constant llama.cpp records for the recipe. The key is
// cosmetic (llama.cpp displays it; it does not validate tensors against it), so
// it names the requested format's nominal type even where the divisibility
// fallback (gemma3TensorType) stores some rows at a higher-precision 32-block
// type.
func gemma3FileType(format basegguf.QuantizeFormat) uint32 {
	switch format {
	case basegguf.QuantizeQ8_0:
		return 7
	case basegguf.QuantizeQ3_K_M:
		return 12
	case basegguf.QuantizeQ2_K_M: // QuantizeQ2_K is the same "q2_k" string
		return 10
	case basegguf.QuantizeQ5_K_M, basegguf.QuantizeQ5_K:
		return 17
	case basegguf.QuantizeQ6_K:
		return 18
	default: // q4_k_m
		return 15
	}
}
