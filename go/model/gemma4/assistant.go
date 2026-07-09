// SPDX-Licence-Identifier: EUPL-1.2

package gemma4

import (
	core "dappco.re/go"
	"dappco.re/go/inference/model"
)

// assistant.go declares the gemma4 MTP assistant (attached speculative drafter) to the
// engine's reactive assistant loader (model.RegisterAssistant) — the checkpoint-format
// knowledge that used to squat inside the engine: which model_type ids are gemma4
// assistants, how their config.json parses (backbone_hidden_size + the ordered-embedding
// head + a nested/flat text_config), how the GGUF export of the same drafter spells its
// metadata (general.architecture "gemma4-assistant", keys under that prefix) and tensor
// names. The engine consumes only the neutral model.AssistantConfig this produces — it
// never keys on "gemma4".

const assistantGGUFArch = "gemma4-assistant"

func init() {
	model.RegisterAssistant(model.AssistantSpec{
		// "" claims checkpoints that predate the model_type field — the legacy default
		// this format shipped with.
		ModelTypes:     []string{"gemma4_assistant", "gemma4_unified_assistant", ""},
		Method:         model.MTPDraftModel, // gemma4 -assistant is a separate draft model
		Parse:          ParseAssistantConfig,
		GGUFArch:       assistantGGUFArch,
		ParseGGUF:      AssistantConfigFromGGUF,
		GGUFWeightName: AssistantGGUFWeightName,
	})
}

// assistantConfig is the raw config.json shape of a gemma4 assistant checkpoint: the
// target-attachment dims + the ordered-embedding head declaration at the top level, with
// the drafter's own text arch nested under text_config (or flat, in early exports).
type assistantConfig struct {
	ModelType                string             `json:"model_type"`
	BackboneHiddenSize       int                `json:"backbone_hidden_size"`
	NumCentroids             int                `json:"num_centroids"`
	CentroidIntermediateTopK int                `json:"centroid_intermediate_top_k"`
	UseOrderedEmbeddings     bool               `json:"use_ordered_embeddings"`
	Quantization             *model.QuantConfig `json:"quantization"` // quantised exports put the block at the TOP level (mlx convention), beside text_config
	TextConfig               Config             `json:"text_config"`
}

// ParseAssistantConfig parses a gemma4 assistant config.json into the neutral
// model.AssistantConfig: resolves the nested-or-flat text_config, validates the
// load-bearing dims, and derives the drafter's own Arch. Registered as the spec's
// config.json parser.
func ParseAssistantConfig(data []byte) (model.AssistantConfig, error) {
	var raw assistantConfig
	if r := core.JSONUnmarshal(data, &raw); !r.OK {
		return model.AssistantConfig{}, core.NewError("gemma4.assistant config parse failed: " + r.Error())
	}
	textConfig := raw.TextConfig
	if textConfig.HiddenSize <= 0 && textConfig.NumHiddenLayers <= 0 {
		// early exports carry the text arch FLAT rather than under text_config.
		var flatText Config
		if r := core.JSONUnmarshal(data, &flatText); !r.OK {
			return model.AssistantConfig{}, core.NewError("gemma4.assistant config parse failed: " + r.Error())
		}
		if flatText.HiddenSize > 0 || flatText.NumHiddenLayers > 0 {
			textConfig = flatText
		}
	}
	modelType := raw.ModelType
	if modelType == "" {
		modelType = "gemma4_assistant"
	}
	cfg, err := buildAssistantConfig(modelType, raw.BackboneHiddenSize, raw.NumCentroids,
		raw.CentroidIntermediateTopK, raw.UseOrderedEmbeddings, textConfig)
	if err != nil {
		return model.AssistantConfig{}, err
	}
	if raw.Quantization != nil {
		// the top-level block outranks a nested text_config one, matching
		// Config.ResolvedQuant's convention for the multimodal wrapper.
		cfg.Quant = raw.Quantization
	}
	return cfg, nil
}

// buildAssistantConfig validates the parsed dims and derives the neutral config — shared
// by the config.json and GGUF paths so both enforce the same invariants.
func buildAssistantConfig(modelType string, backbone, numCentroids, centroidTopK int, ordered bool, text Config) (model.AssistantConfig, error) {
	if modelType != "gemma4_assistant" && modelType != "gemma4_unified_assistant" {
		return model.AssistantConfig{}, core.NewError("gemma4.assistant config has unsupported model_type: " + modelType)
	}
	if backbone <= 0 {
		return model.AssistantConfig{}, core.NewError("gemma4.assistant config has invalid backbone_hidden_size")
	}
	if text.HiddenSize <= 0 {
		return model.AssistantConfig{}, core.NewError("gemma4.assistant config has invalid hidden_size")
	}
	if text.NumHiddenLayers <= 0 {
		return model.AssistantConfig{}, core.NewError("gemma4.assistant config has invalid num_hidden_layers")
	}
	if text.NumAttentionHeads <= 0 {
		return model.AssistantConfig{}, core.NewError("gemma4.assistant config has invalid num_attention_heads")
	}
	if text.HeadDim <= 0 {
		return model.AssistantConfig{}, core.NewError("gemma4.assistant config has invalid head_dim")
	}
	if ordered && numCentroids <= 0 {
		return model.AssistantConfig{}, core.NewError("gemma4.assistant ordered embeddings require num_centroids")
	}
	arch, err := text.Arch()
	if err != nil {
		return model.AssistantConfig{}, core.E("gemma4.assistant", "derive arch", err)
	}
	return model.AssistantConfig{
		ModelType:         modelType,
		BackboneHidden:    backbone,
		NumCentroids:      numCentroids,
		CentroidTopK:      centroidTopK,
		OrderedEmbeddings: ordered,
		LayerTypes:        text.LayerTypes,
		Arch:              arch,
		Quant:             text.ResolvedQuant(),
	}, nil
}

// AssistantConfigFromGGUF builds the neutral config from a GGUF drafter's metadata
// (general.architecture "gemma4-assistant", dims under that prefix). vocabHint carries the
// embed-tensor-derived vocab for exports that omit vocab_size (0 = no hint). Registered as
// the spec's GGUF parser.
func AssistantConfigFromGGUF(meta map[string]any, vocabHint int) (model.AssistantConfig, error) {
	if arch, _ := meta["general.architecture"].(string); arch != assistantGGUFArch {
		return model.AssistantConfig{}, core.E("gemma4.assistant.gguf", "general.architecture is not gemma4-assistant", nil)
	}
	const p = assistantGGUFArch + "."
	layers := ggufMetaInt(meta, p+"block_count")
	hidden := ggufMetaInt(meta, p+"embedding_length")
	heads := ggufMetaInt(meta, p+"attention.head_count")
	headDim := ggufMetaInt(meta, p+"attention.key_length")
	if layers <= 0 || hidden <= 0 || heads <= 0 || headDim <= 0 {
		return model.AssistantConfig{}, core.E("gemma4.assistant.gguf",
			"drafter gguf is missing block_count / embedding_length / head_count / key_length metadata", nil)
	}
	backbone := ggufMetaInt(meta, p+"embedding_length_out")
	if backbone <= 0 {
		backbone = hidden
	}
	pattern := ggufMetaInt(meta, p+"attention.sliding_window_pattern")
	if pattern <= 0 {
		pattern = 1
	}
	layerTypes := make([]string, layers)
	for i := range layerTypes {
		if (i+1)%pattern == 0 {
			layerTypes[i] = "full_attention"
		} else {
			layerTypes[i] = "sliding_attention"
		}
	}
	eps := ggufMetaFloat(meta, p+"attention.layer_norm_rms_epsilon")
	if eps == 0 {
		eps = 1e-6
	}
	freqBase := ggufMetaFloat(meta, p+"rope.freq_base")
	if freqBase == 0 {
		freqBase = 1000000
	}
	freqBaseSWA := ggufMetaFloat(meta, p+"rope.freq_base_swa")
	if freqBaseSWA == 0 {
		freqBaseSWA = 10000
	}
	rotaryFactor := func(dimKey string) float32 {
		if dims := ggufMetaInt(meta, dimKey); dims > 0 && headDim > 0 {
			return float32(dims) / float32(headDim)
		}
		return 1
	}
	text := Config{
		HiddenSize:              hidden,
		NumHiddenLayers:         layers,
		IntermediateSize:        ggufMetaInt(meta, p+"feed_forward_length"),
		NumAttentionHeads:       heads,
		NumKeyValueHeads:        ggufMetaInt(meta, p+"attention.head_count_kv"),
		HeadDim:                 headDim,
		VocabSize:               ggufMetaInt(meta, p+"vocab_size"),
		RMSNormEps:              eps,
		SlidingWindow:           ggufMetaInt(meta, p+"attention.sliding_window"),
		MaxPositionEmbeddings:   ggufMetaInt(meta, p+"context_length"),
		NumKVSharedLayers:       ggufMetaInt(meta, p+"attention.shared_kv_layers"),
		HiddenSizePerLayerInput: ggufMetaInt(meta, p+"embedding_length_per_layer_input"),
		LayerTypes:              layerTypes,
		RopeParameters: map[string]RopeParam{
			"full_attention": {
				RopeTheta:           freqBase,
				RopeType:            "default",
				Factor:              1,
				PartialRotaryFactor: rotaryFactor(p + "rope.dimension_count"),
			},
			"sliding_attention": {
				RopeTheta:           freqBaseSWA,
				RopeType:            "default",
				Factor:              1,
				PartialRotaryFactor: rotaryFactor(p + "rope.dimension_count_swa"),
			},
		},
	}
	if text.NumKeyValueHeads <= 0 {
		text.NumKeyValueHeads = heads
	}
	if text.VocabSize == 0 {
		text.VocabSize = vocabHint
	}
	return buildAssistantConfig("gemma4_assistant", backbone, 0, 0, false, text)
}

// AssistantGGUFWeightName maps a gemma4-assistant GGUF tensor name onto the canonical
// checkpoint name the engine's assistant forward reads ("" = not part of the format).
// Registered as the spec's weight-name map.
func AssistantGGUFWeightName(name string) string {
	switch name {
	case "token_embd.weight":
		return "model.embed_tokens.weight"
	case "output_norm.weight":
		return "model.norm.weight"
	case "nextn.pre_projection.weight":
		return "pre_projection.weight"
	case "nextn.post_projection.weight":
		return "post_projection.weight"
	}
	if !core.HasPrefix(name, "blk.") {
		return ""
	}
	rest := core.TrimPrefix(name, "blk.")
	dot := -1
	for i := 0; i < len(rest); i++ {
		if rest[i] == '.' {
			dot = i
			break
		}
	}
	if dot <= 0 {
		return ""
	}
	layer, leaf := rest[:dot], rest[dot+1:]
	prefix := "model.layers." + layer
	switch leaf {
	case "attn_norm.weight":
		return prefix + ".input_layernorm.weight"
	case "post_attention_norm.weight":
		return prefix + ".post_attention_layernorm.weight"
	case "ffn_norm.weight":
		return prefix + ".pre_feedforward_layernorm.weight"
	case "post_ffw_norm.weight":
		return prefix + ".post_feedforward_layernorm.weight"
	case "attn_q.weight":
		return prefix + ".self_attn.q_proj.weight"
	case "attn_q_norm.weight":
		return prefix + ".self_attn.q_norm.weight"
	case "attn_output.weight":
		return prefix + ".self_attn.o_proj.weight"
	case "ffn_gate.weight":
		return prefix + ".mlp.gate_proj.weight"
	case "ffn_up.weight":
		return prefix + ".mlp.up_proj.weight"
	case "ffn_down.weight":
		return prefix + ".mlp.down_proj.weight"
	case "layer_output_scale.weight":
		return prefix + ".layer_scalar.weight"
	}
	return ""
}

func ggufMetaInt(meta map[string]any, key string) int {
	switch v := meta[key].(type) {
	case uint32:
		return int(v)
	case int32:
		return int(v)
	case uint64:
		return int(v)
	case int64:
		return int(v)
	case int:
		return v
	case float64:
		return int(v)
	}
	return 0
}

func ggufMetaFloat(meta map[string]any, key string) float32 {
	switch v := meta[key].(type) {
	case float32:
		return v
	case float64:
		return float32(v)
	case uint32:
		return float32(v)
	case int32:
		return float32(v)
	}
	return 0
}
