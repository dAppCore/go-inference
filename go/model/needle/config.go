// SPDX-Licence-Identifier: EUPL-1.2

// Package needle is a pure-Go, host-side f32 reference forward for Cactus
// Compute's "Needle" model — a 26M-parameter encoder-decoder ("Simple Attention
// Network") distilled for on-device function calling. It has no engine, no cgo
// and no GPU: it loads the published safetensors weights, widens bf16 to f32,
// and runs the encoder + cross-attending decoder in plain Go. Its purpose is to
// de-risk the encoder-decoder + cross-attention direction before any accelerated
// port — a readable oracle whose only claim to correctness is that it reproduces
// the reference model's tokens.
//
//	m, err := needle.Load("/path/to/needle-snapshot")
//	if err != nil {
//		return err
//	}
//	out := m.Generate("What is the weather in San Francisco?",
//		`[{"name":"get_weather","parameters":{"location":"string"}}]`, 64)
//	// out == ` [{"name":"get_weather","arguments":{"location":"San Francisco"}}]`
//
// The architecture (228 weights, verified against the safetensors header):
//   - Shared embed_tokens[vocab,d] scaled by sqrt(d); tied lm_head.
//   - Encoder x12: input_layernorm -> GQA self-attn (per-head QK-norm, RoPE,
//     bidirectional) -> scalar sigmoid gate on the residual. No FFN. final_norm.
//   - Decoder x8: input_layernorm -> masked self-attn (gate) ->
//     encoder_attn_layer_norm -> cross-attn to the FINAL encoder output (own
//     k/v, no RoPE, gate). No FFN. norm.
package needle

import (
	core "dappco.re/go"
	coreio "dappco.re/go/io"
)

// Config holds the Needle hyper-parameters read from a checkpoint's config.json.
// The zero value is not usable; construct via LoadConfig or DefaultConfig.
type Config struct {
	VocabSize        int     // token count (8192)
	HiddenSize       int     // d_model (512)
	NumHeads         int     // query heads (8)
	NumKVHeads       int     // key/value heads for GQA (4)
	NumEncoderLayers int     // encoder depth (12)
	NumDecoderLayers int     // decoder depth (8)
	RopeTheta        float64 // RoPE base (10000)
	RMSNormEps       float64 // ZCRMSNorm epsilon (1e-6)
	PadTokenID       int     // 0
	EosTokenID       int     // 1 — also the decoder start token
	BosTokenID       int     // 2
	ToolsTokenID     int     // 5 — <tools> separator between query and tools
}

// HeadDim is the per-head width (hidden / heads = 64).
//
//	cfg.HeadDim() // 64
func (c Config) HeadDim() int { return c.HiddenSize / c.NumHeads }

// KVDim is the concatenated key/value projection width (kv_heads * head_dim = 256).
//
//	cfg.KVDim() // 256
func (c Config) KVDim() int { return c.NumKVHeads * c.HeadDim() }

// DefaultConfig returns the published Needle configuration. It is the fallback
// when a checkpoint ships no config.json, and the source of truth for the token
// ids the tokenizer and decoder rely on.
//
//	cfg := needle.DefaultConfig()
//	cfg.HiddenSize // 512
func DefaultConfig() Config {
	return Config{
		VocabSize:        8192,
		HiddenSize:       512,
		NumHeads:         8,
		NumKVHeads:       4,
		NumEncoderLayers: 12,
		NumDecoderLayers: 8,
		RopeTheta:        10000,
		RMSNormEps:       1e-6,
		PadTokenID:       0,
		EosTokenID:       1,
		BosTokenID:       2,
		ToolsTokenID:     5,
	}
}

// configJSON mirrors the subset of config.json this reference needs. Fields the
// checkpoint omits keep their DefaultConfig value.
type configJSON struct {
	VocabSize        *int     `json:"vocab_size"`
	HiddenSize       *int     `json:"hidden_size"`
	NumHeads         *int     `json:"num_heads"`
	NumKVHeads       *int     `json:"num_kv_heads"`
	NumEncoderLayers *int     `json:"num_encoder_layers"`
	NumDecoderLayers *int     `json:"num_decoder_layers"`
	RopeTheta        *float64 `json:"rope_theta"`
	RMSNormEps       *float64 `json:"rms_norm_eps"`
	PadTokenID       *int     `json:"pad_token_id"`
	EosTokenID       *int     `json:"eos_token_id"`
	BosTokenID       *int     `json:"bos_token_id"`
}

// LoadConfig reads config.json from a checkpoint directory, falling back to
// DefaultConfig for any field the file omits. A missing config.json is not an
// error — the published defaults are returned — so the reference runs against a
// bare weights+tokenizer pack.
//
//	cfg, err := needle.LoadConfig("/models/needle")
func LoadConfig(dir string) (Config, error) {
	cfg := DefaultConfig()
	path := dir + "/config.json"
	raw, err := coreio.Local.Read(path)
	if err != nil {
		// A checkpoint without config.json is served by the published defaults.
		return cfg, nil
	}
	var j configJSON
	if r := core.JSONUnmarshal(core.AsBytes(raw), &j); !r.OK {
		return cfg, core.E("needle.LoadConfig", "parse "+path, r.Err())
	}
	if j.VocabSize != nil {
		cfg.VocabSize = *j.VocabSize
	}
	if j.HiddenSize != nil {
		cfg.HiddenSize = *j.HiddenSize
	}
	if j.NumHeads != nil {
		cfg.NumHeads = *j.NumHeads
	}
	if j.NumKVHeads != nil {
		cfg.NumKVHeads = *j.NumKVHeads
	}
	if j.NumEncoderLayers != nil {
		cfg.NumEncoderLayers = *j.NumEncoderLayers
	}
	if j.NumDecoderLayers != nil {
		cfg.NumDecoderLayers = *j.NumDecoderLayers
	}
	if j.RopeTheta != nil {
		cfg.RopeTheta = *j.RopeTheta
	}
	if j.RMSNormEps != nil {
		cfg.RMSNormEps = *j.RMSNormEps
	}
	if j.PadTokenID != nil {
		cfg.PadTokenID = *j.PadTokenID
	}
	if j.EosTokenID != nil {
		cfg.EosTokenID = *j.EosTokenID
	}
	if j.BosTokenID != nil {
		cfg.BosTokenID = *j.BosTokenID
	}
	return cfg, nil
}
