// SPDX-Licence-Identifier: EUPL-1.2

package gguf

import (
	"encoding/binary"
	"math"
	"strconv"

	core "dappco.re/go"
)

// TransformerLaneSpec declares the llama.cpp names and config variants for a
// conventional decoder-only transformer GGUF export lane.
type TransformerLaneSpec struct {
	Architecture string
	ModelTypes   []string
	TokenizerPre string
	TopLevel     map[string]string
	Layer        map[string]string
}

type transformerLaneConfig struct {
	ModelType             string                 `json:"model_type"`
	MaxPositionEmbeddings int                    `json:"max_position_embeddings"`
	HiddenSize            int                    `json:"hidden_size"`
	IntermediateSize      int                    `json:"intermediate_size"`
	NumHiddenLayers       int                    `json:"num_hidden_layers"`
	NumAttentionHeads     int                    `json:"num_attention_heads"`
	NumKeyValueHeads      int                    `json:"num_key_value_heads"`
	HeadDim               int                    `json:"head_dim"`
	VocabSize             int                    `json:"vocab_size"`
	RMSNormEps            float32                `json:"rms_norm_eps"`
	LayerNormEps          float32                `json:"layer_norm_eps"`
	RopeTheta             float32                `json:"rope_theta"`
	BOSTokenID            int                    `json:"bos_token_id"`
	EOSTokenID            int                    `json:"eos_token_id"`
	TieWordEmbeddings     bool                   `json:"tie_word_embeddings"`
	TextConfig            *transformerLaneConfig `json:"text_config"`
}

// NewTransformerQuantizeLane builds a q4_k_m/q8_0 lane from a canonical
// llama.cpp architecture declaration.
func NewTransformerQuantizeLane(spec TransformerLaneSpec) QuantizeLane {
	matches := func(modelType string) bool {
		for _, candidate := range spec.ModelTypes {
			if modelType == candidate {
				return true
			}
		}
		return false
	}
	return QuantizeLane{
		Detect: func(data []byte) bool {
			var cfg transformerLaneConfig
			return core.JSONUnmarshal(data, &cfg).OK && matches(cfg.ModelType)
		},
		SupportsFormat: func(format QuantizeFormat) bool {
			return format == QuantizeQ4_K_M || format == QuantizeQ8_0
		},
		UnsupportedFormatError: func(format QuantizeFormat) error {
			return core.NewError("gguf: " + spec.Architecture + " GGUF conversion does not support " + string(format) + " (supported: q4_k_m, q8_0)")
		},
		Quantize: func(source Source, configJSON []byte, tensors []DenseSafetensor, format QuantizeFormat) ([]Tensor, []MetadataEntry, error) {
			return quantizeTransformerLane(spec, matches, source, configJSON, tensors, format)
		},
	}
}

func quantizeTransformerLane(spec TransformerLaneSpec, matches func(string) bool, source Source, configJSON []byte, tensors []DenseSafetensor, format QuantizeFormat) ([]Tensor, []MetadataEntry, error) {
	var wrapper transformerLaneConfig
	if r := core.JSONUnmarshal(configJSON, &wrapper); !r.OK {
		return nil, nil, core.E("quantizeTransformerLane", "parse config.json", r.Err())
	}
	cfg := &wrapper
	if cfg.TextConfig != nil {
		cfg = cfg.TextConfig
	}
	if !matches(wrapper.ModelType) || cfg.MaxPositionEmbeddings <= 0 || cfg.HiddenSize <= 0 || cfg.IntermediateSize <= 0 || cfg.NumHiddenLayers <= 0 || cfg.NumAttentionHeads <= 0 || cfg.VocabSize <= 0 {
		return nil, nil, core.NewError("gguf: " + spec.Architecture + " config.json missing required hyperparameters")
	}
	if cfg.NumKeyValueHeads == 0 {
		cfg.NumKeyValueHeads = cfg.NumAttentionHeads
	}
	if cfg.HeadDim == 0 {
		if cfg.HiddenSize%cfg.NumAttentionHeads != 0 {
			return nil, nil, core.NewError("gguf: hidden_size is not divisible by num_attention_heads")
		}
		cfg.HeadDim = cfg.HiddenSize / cfg.NumAttentionHeads
	}
	eps := cfg.RMSNormEps
	if eps == 0 {
		eps = cfg.LayerNormEps
	}
	if eps == 0 {
		eps = 1e-6
	}
	if cfg.RopeTheta == 0 {
		cfg.RopeTheta = 10000
	}
	quantized := make([]Tensor, 0, len(tensors))
	for _, tensor := range tensors {
		if IsMultimodalTowerTensor(tensor.Name) {
			continue
		}
		canonical, layer, err := transformerCanonicalName(spec, tensor.Name)
		if err != nil {
			return nil, nil, err
		}
		tensorType := transformerTensorType(format, canonical, layer, cfg.NumHiddenLayers, cfg.TieWordEmbeddings)
		data, err := encodeTransformerTensor(tensor.Data, tensorType)
		if err != nil {
			return nil, nil, core.E("quantizeTransformerLane", "encode "+canonical, err)
		}
		quantized = append(quantized, Tensor{Name: canonical, Type: tensorType, Shape: reverseTransformerShape(tensor.Shape), Data: data})
	}
	if len(quantized) == 0 {
		return nil, nil, core.NewError("gguf: no " + spec.Architecture + " tensors found in source")
	}
	metadata := transformerMetadata(spec.Architecture, *cfg, eps, transformerFileType(format), core.TrimSuffix(core.TrimSuffix(core.PathBase(core.TrimSuffix(source.Root, "/")), "-bf16"), "-f32"))
	tokenizerRead := core.ReadFile(core.PathJoin(source.Root, "tokenizer.json"))
	if !tokenizerRead.OK {
		return nil, nil, core.E("quantizeTransformerLane", "read tokenizer.json", tokenizerRead.Err())
	}
	tokenizer, err := transformerTokenizer(tokenizerRead.Value.([]byte), spec.TokenizerPre, cfg.BOSTokenID, cfg.EOSTokenID)
	if err != nil {
		return nil, nil, err
	}
	metadata = append(metadata, tokenizer...)
	return quantized, metadata, nil
}

func transformerCanonicalName(spec TransformerLaneSpec, source string) (string, int, error) {
	if unwrapped, ok := core.CutPrefix(source, "language_model."); ok {
		source = unwrapped
	}
	if name, ok := spec.TopLevel[source]; ok {
		return name, -1, nil
	}
	rest, ok := core.CutPrefix(source, "model.layers.")
	if ok {
		index, suffix, found := core.Cut(rest, ".")
		layer, err := strconv.Atoi(index)
		if found && err == nil {
			if name, exists := spec.Layer[suffix]; exists {
				return core.Concat("blk.", index, ".", name), layer, nil
			}
		}
	}
	return "", -1, core.NewError("gguf: " + spec.Architecture + " has no canonical GGUF name for tensor " + source)
}

func transformerTensorType(format QuantizeFormat, name string, layer, layers int, tied bool) uint32 {
	if core.HasSuffix(name, "_norm.weight") || core.HasSuffix(name, "_norm.bias") || core.HasSuffix(name, ".bias") {
		return TensorTypeF32
	}
	if format == QuantizeQ8_0 {
		return TensorTypeQ8_0
	}
	moreBits := layer < layers/8 || layer >= 7*layers/8 || (layer-layers/8)%3 == 2
	if name == "output.weight" || tied && name == "token_embd.weight" || moreBits && (core.HasSuffix(name, ".attn_v.weight") || core.HasSuffix(name, ".ffn_down.weight")) {
		return TensorTypeQ6K
	}
	return TensorTypeQ4K
}

func encodeTransformerTensor(values []float32, tensorType uint32) ([]byte, error) {
	if tensorType == TensorTypeF32 {
		out := make([]byte, len(values)*4)
		for i, value := range values {
			binary.LittleEndian.PutUint32(out[i*4:], math.Float32bits(value))
		}
		return out, nil
	}
	format, block := QuantizeQ8_0, 32
	if tensorType != TensorTypeQ8_0 {
		block = 256
		switch tensorType {
		case TensorTypeQ4K:
			format = QuantizeQ4_K
		case TensorTypeQ6K:
			format = QuantizeQ6_K
		default:
			return nil, core.Errorf("gguf: unsupported transformer tensor type %d", tensorType)
		}
	}
	if len(values)%block != 0 {
		return nil, core.Errorf("gguf: quantized tensor has %d elements, not a multiple of %d", len(values), block)
	}
	return Quantize(format, values)
}

func reverseTransformerShape(shape []uint64) []uint64 {
	out := make([]uint64, len(shape))
	for i, value := range shape {
		out[len(shape)-1-i] = value
	}
	return out
}

func transformerFileType(format QuantizeFormat) uint32 {
	if format == QuantizeQ8_0 {
		return 7
	}
	return 15
}

func transformerMetadata(arch string, cfg transformerLaneConfig, eps float32, fileType uint32, name string) []MetadataEntry {
	u32 := func(key string, value int) MetadataEntry {
		return MetadataEntry{Key: key, ValueType: ValueTypeUint32, Value: uint32(value)}
	}
	f32 := func(key string, value float32) MetadataEntry {
		return MetadataEntry{Key: key, ValueType: ValueTypeFloat32, Value: value}
	}
	out := []MetadataEntry{{Key: "general.architecture", ValueType: ValueTypeString, Value: arch}, {Key: "general.type", ValueType: ValueTypeString, Value: "model"}, {Key: "general.quantization_version", ValueType: ValueTypeUint32, Value: uint32(2)}, {Key: "general.file_type", ValueType: ValueTypeUint32, Value: fileType}}
	if name != "" {
		out = append(out, MetadataEntry{Key: "general.name", ValueType: ValueTypeString, Value: name})
	}
	return append(out, u32(arch+".block_count", cfg.NumHiddenLayers), u32(arch+".context_length", cfg.MaxPositionEmbeddings), u32(arch+".embedding_length", cfg.HiddenSize), u32(arch+".feed_forward_length", cfg.IntermediateSize), u32(arch+".attention.head_count", cfg.NumAttentionHeads), u32(arch+".attention.head_count_kv", cfg.NumKeyValueHeads), u32(arch+".rope.dimension_count", cfg.HeadDim), f32(arch+".rope.freq_base", cfg.RopeTheta), f32(arch+".attention.layer_norm_rms_epsilon", eps))
}

type transformerTokenizerJSON struct {
	Model struct {
		Vocab  map[string]int `json:"vocab"`
		Merges [][]string     `json:"merges"`
	} `json:"model"`
}

func transformerTokenizer(data []byte, pre string, bos, eos int) ([]MetadataEntry, error) {
	var tokenizer transformerTokenizerJSON
	if r := core.JSONUnmarshal(data, &tokenizer); !r.OK || len(tokenizer.Model.Vocab) == 0 {
		return nil, core.NewError("gguf: tokenizer.json has an empty or invalid vocab")
	}
	tokens := make([]string, len(tokenizer.Model.Vocab))
	for token, id := range tokenizer.Model.Vocab {
		if id < 0 || id >= len(tokens) || tokens[id] != "" {
			return nil, core.NewError("gguf: tokenizer vocab has invalid ids")
		}
		tokens[id] = token
	}
	merges := make([]string, len(tokenizer.Model.Merges))
	for i, merge := range tokenizer.Model.Merges {
		if len(merge) != 2 {
			return nil, core.NewError("gguf: tokenizer merge is not a pair")
		}
		merges[i] = core.Concat(merge[0], " ", merge[1])
	}
	types, scores := make([]int32, len(tokens)), make([]float32, len(tokens))
	for i := range types {
		types[i] = 1
	}
	u32 := func(key string, value int) MetadataEntry {
		return MetadataEntry{Key: key, ValueType: ValueTypeUint32, Value: uint32(value)}
	}
	return []MetadataEntry{{Key: "tokenizer.ggml.model", ValueType: ValueTypeString, Value: "gpt2"}, {Key: "tokenizer.ggml.pre", ValueType: ValueTypeString, Value: pre}, {Key: "tokenizer.ggml.tokens", ValueType: ValueTypeArray, Value: tokens}, {Key: "tokenizer.ggml.scores", ValueType: ValueTypeArray, Value: scores}, {Key: "tokenizer.ggml.token_type", ValueType: ValueTypeArray, Value: types}, {Key: "tokenizer.ggml.merges", ValueType: ValueTypeArray, Value: merges}, u32("tokenizer.ggml.bos_token_id", bos), u32("tokenizer.ggml.eos_token_id", eos), {Key: "tokenizer.ggml.add_bos_token", ValueType: ValueTypeBool, Value: true}}, nil
}
