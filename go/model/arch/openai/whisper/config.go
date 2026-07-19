// SPDX-Licence-Identifier: EUPL-1.2

// Package whisper declares openai/whisper-* (WhisperForConditionalGeneration) to the backend-neutral
// reactive model loader AND implements its ASR forward: an audio encoder (log-mel spectrogram in, 1500
// source positions, num_mel_bins channels) feeds a cross-attending text decoder that emits transcript
// tokens — is_encoder_decoder is true in every Whisper config.json. That is architecturally distinct from
// lem's existing audio support (gemma4's audio-INPUT path, which conditions a decoder-only causal-LM on
// an audio embedding): Whisper's ASR forward (encoder tower + cross-attention decoder + the task/language
// special-token machinery) is a genuinely different shape, so — like mamba2 — it never enters
// model.Assemble; it registers its own loader (Load in transcribe.go) and is driven through `lem
// transcribe`, not `lem generate`/`serve`. Registering the model_type here still gives the engine
// direction (model.LookupArch succeeds, config.json parses) for the generate/serve causal-LM path: a user
// pointing THOSE commands at a Whisper checkpoint gets Config.Arch's clean, specific redirect rather than
// "unknown model architecture" or a confusing mid-Assemble failure.
//
// Source: https://huggingface.co/openai/whisper-tiny/resolve/main/config.json
package whisper

import (
	core "dappco.re/go"
	"dappco.re/go/inference/model"
	"dappco.re/go/inference/model/safetensors"
)

// Config is the architecture-relevant subset of a WhisperForConditionalGeneration config.json.
type Config struct {
	ModelType             string `json:"model_type"`
	IsEncoderDecoder      bool   `json:"is_encoder_decoder"`
	ActivationFunction    string `json:"activation_function"`
	DModel                int    `json:"d_model"`
	EncoderLayers         int    `json:"encoder_layers"`
	EncoderAttentionHeads int    `json:"encoder_attention_heads"`
	EncoderFFNDim         int    `json:"encoder_ffn_dim"`
	DecoderLayers         int    `json:"decoder_layers"`
	DecoderAttentionHeads int    `json:"decoder_attention_heads"`
	DecoderFFNDim         int    `json:"decoder_ffn_dim"`
	NumMelBins            int    `json:"num_mel_bins"`
	MaxSourcePositions    int    `json:"max_source_positions"`
	MaxTargetPositions    int    `json:"max_target_positions"`
	VocabSize             int    `json:"vocab_size"`
	BOSTokenID            int    `json:"bos_token_id"`
	EOSTokenID            int    `json:"eos_token_id"`
	PadTokenID            int    `json:"pad_token_id"`
	DecoderStartTokenID   int    `json:"decoder_start_token_id"`
}

// ParseConfig parses a Whisper (WhisperForConditionalGeneration) Hugging Face config.
func ParseConfig(data []byte) (*Config, error) {
	var cfg Config
	if r := core.JSONUnmarshal(data, &cfg); !r.OK {
		return nil, core.NewError("whisper.ParseConfig: config.json parse failed")
	}
	return &cfg, nil
}

// InferFromWeights is a no-op: Whisper's config.json declares every encoder/decoder dimension, and
// Config.Arch refuses before any dimension would be consulted for an Assemble — there is nothing to
// resolve from the checkpoint's weight shapes yet (the don't-guess rule; see qwen2.Config.InferFromWeights
// for the pattern a future ASR serving path would follow).
func (c *Config) InferFromWeights(map[string]safetensors.Tensor) {}

// Arch always refuses: Whisper is an ASR encoder-decoder, not the decoder-only causal-LM shape
// model.Arch declares — there is no neutral decode Arch to derive from an encoder tower plus a
// cross-attending decoder, so `lem generate`/`lem serve` (which both resolve a model through Arch before
// model.Assemble) cannot drive one. The model_type is registered (model.LookupArch succeeds) so a user
// pointing THOSE commands at a Whisper checkpoint gets this clean, specific redirect rather than "unknown
// model architecture" or a confusing mid-Assemble failure — transcription itself is fully implemented,
// through `lem transcribe` (see Load/Model.Transcribe in transcribe.go), which never calls Arch.
func (c *Config) Arch() (model.Arch, error) {
	return model.Arch{}, core.NewError(core.Sprintf(
		"whisper.Config.Arch: whisper is a Whisper ASR encoder-decoder (speech-to-text; %d encoder + "+
			"%d decoder layers, d_model %d, %d mel bins), not a decoder-only causal-LM — there is no neutral "+
			"decode Arch to derive; use `lem transcribe --model <dir> <audio.wav>` instead of "+
			"generate/serve (lem's existing audio path is gemma4 audio-INPUT to a decoder, a different "+
			"shape from Whisper's ASR encoder-decoder forward)",
		c.EncoderLayers, c.DecoderLayers, c.DModel, c.NumMelBins))
}
