// SPDX-Licence-Identifier: EUPL-1.2

// Package bert is a host f32 BERT/BGE-class encoder: it loads a BertModel HF
// snapshot (config.json + vocab.txt + model.safetensors) and runs a real
// transformer forward pass on the CPU to produce sentence embeddings, then
// pools (CLS or mean) and L2-normalises them. It implements
// inference.EmbeddingModel and inference.RerankModel so the OpenAI-compatible
// /v1/embeddings and /v1/rerank routes can serve a small encoder without a GPU.
//
// Correctness is the round-1 goal: the maths is naive host float32 and matches
// sentence-transformers within cosine >= 0.999 per vector (see the parity test).
// The device-GEMM path is a later optimisation — the forward is written so the
// linear/attention stages can be swapped for engine kernels without changing
// the tokeniser, pooling, or wiring.
//
//	m, err := bert.Load("/path/to/bge-small-en-v1.5/snapshot")
//	res, err := m.Embed(ctx, inference.EmbeddingRequest{Input: []string{"hello"}, Normalize: true})
//	vec := res.Vectors[0] // 384-dim unit vector
package bert

import (
	core "dappco.re/go"
)

// Config is the subset of a BertModel config.json the host encoder consumes.
// Field names follow the HF config keys so JSON binding is a straight decode.
//
//	cfg, err := bert.ParseConfig(configBytes)
type Config struct {
	ModelType             string  `json:"model_type"`
	HiddenSize            int     `json:"hidden_size"`
	NumHiddenLayers       int     `json:"num_hidden_layers"`
	NumAttentionHeads     int     `json:"num_attention_heads"`
	IntermediateSize      int     `json:"intermediate_size"`
	VocabSize             int     `json:"vocab_size"`
	MaxPositionEmbeddings int     `json:"max_position_embeddings"`
	TypeVocabSize         int     `json:"type_vocab_size"`
	LayerNormEps          float64 `json:"layer_norm_eps"`
	HiddenAct             string  `json:"hidden_act"`
	PadTokenID            int     `json:"pad_token_id"`
}

// HeadDim is the per-head width — hidden size split across the attention heads.
// Requires NumAttentionHeads > 0 and HiddenSize divisible by it (validated in
// Load); a zero head count returns 0 rather than dividing by zero.
func (c Config) HeadDim() int {
	if c.NumAttentionHeads == 0 {
		return 0
	}
	return c.HiddenSize / c.NumAttentionHeads
}

// ParseConfig decodes a BertModel config.json into a Config and checks the
// shape fields an encoder cannot run without. An unknown model_type is allowed
// (BGE, GTE, and E5 checkpoints all report "bert") but the dimensions must be
// positive and the heads must divide the hidden size evenly.
//
//	cfg, err := bert.ParseConfig(core.ReadFile(path).Value.([]byte))
func ParseConfig(data []byte) (Config, error) {
	var cfg Config
	if result := core.JSONUnmarshal(data, &cfg); !result.OK {
		return Config{}, core.E("bert.ParseConfig", "decode config.json", result.Err())
	}
	if cfg.LayerNormEps == 0 {
		cfg.LayerNormEps = 1e-12
	}
	if cfg.HiddenAct == "" {
		cfg.HiddenAct = "gelu"
	}
	if err := cfg.validate(); err != nil {
		return Config{}, err
	}
	return cfg, nil
}

func (c Config) validate() error {
	switch {
	case c.HiddenSize <= 0:
		return core.E("bert.Config", "hidden_size must be positive", nil)
	case c.NumHiddenLayers <= 0:
		return core.E("bert.Config", "num_hidden_layers must be positive", nil)
	case c.NumAttentionHeads <= 0:
		return core.E("bert.Config", "num_attention_heads must be positive", nil)
	case c.HiddenSize%c.NumAttentionHeads != 0:
		return core.E("bert.Config", "hidden_size must be divisible by num_attention_heads", nil)
	case c.IntermediateSize <= 0:
		return core.E("bert.Config", "intermediate_size must be positive", nil)
	case c.VocabSize <= 0:
		return core.E("bert.Config", "vocab_size must be positive", nil)
	case c.MaxPositionEmbeddings <= 0:
		return core.E("bert.Config", "max_position_embeddings must be positive", nil)
	}
	return nil
}
