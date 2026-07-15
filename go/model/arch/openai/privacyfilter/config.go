// SPDX-Licence-Identifier: EUPL-1.2

// Package privacyfilter declares openai/privacy-filter (OpenAIPrivacyFilterForTokenClassification) to the
// backend-neutral reactive model loader. It is a PII token-classifier — a MoE transformer whose head emits
// a BIOES tag per token across 33 classes (O plus B/I/E/S for account numbers, addresses, dates, emails,
// names, phone numbers, URLs and secrets) — not a causal-LM decoder. Registering it here gives the engine
// direction (model.LookupArch succeeds, config.json parses) without claiming lem can serve it: lem has no
// token-classification/NER serving path yet, so Config.Arch refuses with that explanation. A classification
// serving path (BIOES decode, label head, span aggregation) is a separate lane.
//
// Source: https://huggingface.co/openai/privacy-filter/resolve/main/config.json
package privacyfilter

import (
	core "dappco.re/go"
	"dappco.re/go/inference/model"
	"dappco.re/go/inference/model/safetensors"
)

// RopeParameters is the YaRN long-context extension openai/privacy-filter carries under the newer
// "rope_parameters" config key — RopeTheta lives nested here, not at Config's top level.
type RopeParameters struct {
	RopeType                      string  `json:"rope_type"`
	RopeTheta                     float32 `json:"rope_theta"`
	Factor                        float32 `json:"factor"`
	BetaFast                      float32 `json:"beta_fast"`
	BetaSlow                      float32 `json:"beta_slow"`
	OriginalMaxPositionEmbeddings int     `json:"original_max_position_embeddings"`
	Truncate                      bool    `json:"truncate"`
}

// Config is the architecture-relevant subset of an OpenAIPrivacyFilterForTokenClassification config.json.
type Config struct {
	ModelType             string            `json:"model_type"`
	HiddenAct             string            `json:"hidden_act"`
	HiddenSize            int               `json:"hidden_size"`
	IntermediateSize      int               `json:"intermediate_size"`
	NumHiddenLayers       int               `json:"num_hidden_layers"`
	NumAttentionHeads     int               `json:"num_attention_heads"`
	NumKeyValueHeads      int               `json:"num_key_value_heads"`
	HeadDim               int               `json:"head_dim"`
	VocabSize             int               `json:"vocab_size"`
	RMSNormEps            float32           `json:"rms_norm_eps"`
	MaxPositionEmbeddings int               `json:"max_position_embeddings"`
	InitialContextLength  int               `json:"initial_context_length"`
	DefaultNCtx           int               `json:"default_n_ctx"`
	SlidingWindow         int               `json:"sliding_window"`
	NumLocalExperts       int               `json:"num_local_experts"`
	NumExpertsPerTok      int               `json:"num_experts_per_tok"`
	RouterAuxLossCoef     float32           `json:"router_aux_loss_coef"`
	OutputRouterLogits    bool              `json:"output_router_logits"`
	RopeParameters        RopeParameters    `json:"rope_parameters"`
	ID2Label              map[string]string `json:"id2label"`
	Label2ID              map[string]int    `json:"label2id"`
	TieWordEmbeddings     *bool             `json:"tie_word_embeddings"`
}

// ParseConfig parses an openai/privacy-filter (OpenAIPrivacyFilterForTokenClassification) Hugging Face
// config.
func ParseConfig(data []byte) (*Config, error) {
	var cfg Config
	if r := core.JSONUnmarshal(data, &cfg); !r.OK {
		return nil, core.NewError("privacyfilter.ParseConfig: config.json parse failed")
	}
	return &cfg, nil
}

// InferFromWeights is a no-op: openai/privacy-filter's config.json declares every geometry field, and
// Config.Arch refuses before any dimension would be consulted for an Assemble — there is nothing to
// resolve from the checkpoint's weight shapes yet (the don't-guess rule; see qwen2.Config.InferFromWeights
// for the pattern a future classification serving path would follow).
func (c *Config) InferFromWeights(map[string]safetensors.Tensor) {}

// Arch always refuses: openai/privacy-filter is a PII token-classification model (BIOES tagging over
// c.ID2Label's classes), not a generative causal-LM, so there is no neutral decode Arch to derive. The
// model_type is registered (model.LookupArch succeeds) so a user pointing lem at this checkpoint gets a
// clean, specific explanation rather than "unknown model architecture"; a classification serving path is a
// separate lane.
func (c *Config) Arch() (model.Arch, error) {
	return model.Arch{}, core.NewError(core.Sprintf(
		"privacyfilter.Config.Arch: openai_privacy_filter is a PII token-classification model "+
			"(%d-class BIOES tagging, %d-expert/%d-per-token MoE encoder), not a generative arch — "+
			"lem has no token-classification/NER serving path yet; recognised for direction, "+
			"classification serving is a separate lane",
		len(c.ID2Label), c.NumLocalExperts, c.NumExpertsPerTok))
}
