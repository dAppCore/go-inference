// SPDX-Licence-Identifier: EUPL-1.2

package composed

import (
	core "dappco.re/go"
	"dappco.re/go/inference/model"
	"dappco.re/go/inference/model/mtp"
)

// assistant.go declares the Qwen 3.5/3.6 multi-token-prediction head to the engine's reactive assistant
// registry (mtp.RegisterAssistant) — the composed twin of gemma4/assistant.go. It is what turns the
// register.go MTP model_types from a clean refusal into a RECOGNISED drafter: mtp.LookupAssistant
// ("qwen3_5_mtp") now succeeds, so the neutral "is this an attached speculative drafter?" contract answers
// yes for the Qwen hybrid exactly as it does for gemma4.
//
// The composed head is NOT loaded through the metal AssistantModel path (that loader is gemma4-tensor
// shaped — pre_projection / post_projection / ordered embeddings), the same reason the composed BASE is
// loaded by the composed loader rather than the transformer Assemble: the Qwen hybrid's drafter is a small
// composed transformer that shares the base's embedding + LM head. So the spec's Parse produces the neutral
// mtp.AssistantConfig (recognition + the target-attachment dim the pair validates against) while the
// composed loader (LoadMTPHeadDir) owns the actual tensor realisation — declarative recognition here, the
// composed-specific build there.
//
// Real drafter checkpoint (mlx-community/Qwen3.6-27B-MTP-4bit): model_type "qwen3_5_mtp", the head dims
// nested under text_config (hidden_size == the base's, mtp_num_hidden_layers deep), mtp_use_dedicated_
// embeddings false (shares the base embed_tokens) and tie_word_embeddings false (shares the base lm_head).
func init() {
	mtp.RegisterAssistant(mtp.AssistantSpec{
		ModelTypes: []string{"qwen3_5_mtp", "qwen3_5_mtp_text", "qwen3_6_mtp"},
		Method:     mtp.MTPDraftModel, // the Qwen MTP head is a separate draft model, verified by its base
		Parse:      ParseMTPAssistantConfig,
	})
}

// mtpAssistantConfig is the raw config.json shape of a Qwen MTP head checkpoint: the head-attachment dims
// live under text_config (with the whole-checkpoint quantization block at the top level, the mlx
// convention), alongside the MTP-specific block_size + head depth.
type mtpAssistantConfig struct {
	ModelType    string             `json:"model_type"`
	BlockSize    int                `json:"block_size"`
	Quantization *model.QuantConfig `json:"quantization"`
	TextConfig   mtpTextConfig      `json:"text_config"`
}

// mtpTextConfig is the arch-relevant subset of the head's text_config. hidden_size IS the base backbone
// hidden the head projects from (mtp_use_dedicated_embeddings false → it shares the base embed), and
// mtp_num_hidden_layers is the head's own transformer depth.
type mtpTextConfig struct {
	HiddenSize         int      `json:"hidden_size"`
	MTPNumHiddenLayers int      `json:"mtp_num_hidden_layers"`
	NumAttentionHeads  int      `json:"num_attention_heads"`
	NumKeyValueHeads   int      `json:"num_key_value_heads"`
	HeadDim            int      `json:"head_dim"`
	VocabSize          int      `json:"vocab_size"`
	LayerTypes         []string `json:"layer_types"`
}

// ParseMTPAssistantConfig parses a Qwen MTP head config.json into the neutral mtp.AssistantConfig: it
// validates the load-bearing dims and carries the backbone hidden the pair matches against the base. It
// deliberately does NOT derive a model.Arch — the composed head is not a transformer-Assemble arch, so the
// composed loader (not the neutral Arch machinery) realises its tensors. Registered as the spec's parser.
func ParseMTPAssistantConfig(data []byte) (mtp.AssistantConfig, error) {
	var raw mtpAssistantConfig
	if r := core.JSONUnmarshal(data, &raw); !r.OK {
		return mtp.AssistantConfig{}, core.NewError("composed.mtp assistant config parse failed: " + r.Error())
	}
	if !isComposedMTPModelType(raw.ModelType) {
		return mtp.AssistantConfig{}, core.NewError("composed.mtp assistant config has unsupported model_type: " + raw.ModelType)
	}
	text := raw.TextConfig
	if text.HiddenSize <= 0 {
		return mtp.AssistantConfig{}, core.NewError("composed.mtp assistant config has invalid text_config.hidden_size")
	}
	if text.MTPNumHiddenLayers <= 0 {
		return mtp.AssistantConfig{}, core.NewError("composed.mtp assistant config has invalid mtp_num_hidden_layers")
	}
	return mtp.AssistantConfig{
		ModelType:      raw.ModelType,
		Method:         mtp.MTPDraftModel,
		BackboneHidden: text.HiddenSize, // the base hidden the head's fc projection consumes (shared embed)
		LayerTypes:     text.LayerTypes,
		Quant:          raw.Quantization,
	}, nil
}

// isComposedMTPModelType reports whether modelType is one of the Qwen MTP head ids this package claims —
// the single normalisation table the spec registration, the config parse and the register.go refusal all
// key on, so a new Qwen MTP id is added in exactly one place.
func isComposedMTPModelType(modelType string) bool {
	switch modelType {
	case "qwen3_5_mtp", "qwen3_5_mtp_text", "qwen3_6_mtp":
		return true
	}
	return false
}
