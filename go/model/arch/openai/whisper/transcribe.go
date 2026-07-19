// SPDX-Licence-Identifier: EUPL-1.2

package whisper

import (
	core "dappco.re/go"
	"dappco.re/go/inference/decode/tokenizer"
	"dappco.re/go/inference/model/safetensors"
)

// transcribe.go is the top-level orchestration `lem transcribe` drives (cli/transcribe.go): Load a
// checkpoint directory once, Transcribe as many WAV clips as needed. Whisper never enters
// model.Assemble/model.LookupArch here — this mirrors mamba2's "own loader, own session" shape
// (arch/mamba2/loader.go's doc comment), because an encoder-decoder ASR forward does not fit the
// decoder-only causal-LM contract model.Load's registered path assembles.

// Model is a fully loaded Whisper checkpoint, ready to transcribe. Weights are read-only, host-owned f32
// slices (LoadWeights copies every tensor out of the safetensors mmap — see Load's doc comment), so one
// Model is safe to call Transcribe on repeatedly and concurrently from multiple goroutines (each call is
// independent: its own mel/encoder-output/cross-KV/decode-loop state).
type Model struct {
	Config     *Config
	Generation *GenerationConfig
	Features   *FeatureConfig
	Weights    *Weights
	Tokenizer  *tokenizer.Tokenizer
}

// Load reads a Whisper checkpoint directory: config.json, generation_config.json,
// preprocessor_config.json, tokenizer.json, and the safetensors weights — everything Transcribe needs.
// A directory whose config.json is not model_type "whisper" is refused cleanly (the capability-refusal
// pattern serve's vision 400 route uses) rather than failing deeper with a confusing tensor-name error.
func Load(dir string) (*Model, error) {
	configRead := core.ReadFile(core.PathJoin(dir, "config.json"))
	if !configRead.OK {
		return nil, core.E("whisper.Load", "read config.json", resultErr(configRead))
	}
	cb, ok := configRead.Value.([]byte)
	if !ok {
		return nil, core.NewError("whisper.Load: config.json read returned non-byte data")
	}
	cfg, err := ParseConfig(cb)
	if err != nil {
		return nil, err
	}
	if cfg.ModelType != "whisper" {
		return nil, core.NewError("whisper.Load: " + dir + " is not a Whisper checkpoint (model_type " +
			core.Sprintf("%q", cfg.ModelType) + ") — lem transcribe only accepts WhisperForConditionalGeneration directories")
	}

	gen, err := LoadGenerationConfig(dir)
	if err != nil {
		return nil, err
	}
	feat, err := LoadFeatureConfig(dir)
	if err != nil {
		return nil, err
	}
	tok, err := tokenizer.LoadTokenizer(core.PathJoin(dir, "tokenizer.json"))
	if err != nil {
		return nil, core.E("whisper.Load", "load tokenizer.json", err)
	}

	dm, err := safetensors.LoadDirMmap(dir)
	if err != nil {
		return nil, core.E("whisper.Load", "load safetensors", err)
	}
	defer func() { _ = dm.Close() }() // LoadWeights copies every tensor to an owned f32 slice — the mmap needn't outlive it
	w, err := LoadWeights(dm.Tensors, cfg)
	if err != nil {
		return nil, err
	}
	return &Model{Config: cfg, Generation: gen, Features: feat, Weights: w, Tokenizer: tok}, nil
}

// Result is one transcription's output.
type Result struct {
	Text     string
	Language string // the resolved bare language code ("en") — auto-detected or Options.Language echoed back
}

// Options configures one Transcribe call.
type Options struct {
	// Language forces the source language (a bare code "en" or the bracketed "<|en|>" form); ""
	// auto-detects via one decoder step (DetectLanguage).
	Language string
}

// Transcribe runs the full ASR forward on one WAV clip (16-bit PCM, mono, 16 kHz — DecodeWAV16Mono's
// contract): decode WAV → pad/refuse against the v1 single-window bound → mel front end → encode once →
// cross-K/V precompute once (the standard Whisper serving trick — cross-attention K/V depend only on the
// encoder output) → resolve language (auto-detect or Options.Language) → greedy decode → tokenizer
// decode. Audio longer than the checkpoint's fixed window (30 s for every published Whisper checkpoint)
// is a named refusal stating the bound, never silent truncation (the design's documented v1 scope —
// chunked long-form transcription is v2).
func (m *Model) Transcribe(wavBytes []byte, opts Options) (Result, error) {
	if m == nil {
		return Result{}, core.NewError("whisper.Transcribe: nil model")
	}
	samples, err := DecodeWAV16Mono(wavBytes)
	if err != nil {
		return Result{}, err
	}
	windowSamples := m.Features.NSamples
	if len(samples) > windowSamples {
		windowSeconds := float64(windowSamples) / float64(m.Features.SamplingRate)
		gotSeconds := float64(len(samples)) / float64(m.Features.SamplingRate)
		return Result{}, core.NewError(core.Sprintf(
			"whisper.Transcribe: audio is %.2fs, exceeds the v1 single-window bound of %.0fs (%d samples > %d) — "+
				"chunked long-form transcription is not implemented in this lane; trim the clip to %.0fs or shorter",
			gotSeconds, windowSeconds, len(samples), windowSamples, windowSeconds))
	}
	if len(samples) < windowSamples {
		padded := make([]float32, windowSamples)
		copy(padded, samples)
		samples = padded
	}

	melFeatures, err := m.Features.ExtractLogMel(samples)
	if err != nil {
		return Result{}, err
	}
	encOut, err := EncodeAudio(melFeatures, m.Weights, m.Config)
	if err != nil {
		return Result{}, err
	}
	tenc := m.Config.MaxSourcePositions
	crossKV := PrecomputeCrossKV(encOut, tenc, m.Weights)

	initTokens, langCode, err := BuildInitTokens(crossKV, tenc, m.Weights, m.Config, m.Generation, opts.Language)
	if err != nil {
		return Result{}, err
	}
	contentIDs, err := GreedyDecode(crossKV, tenc, m.Weights, m.Config, m.Generation, initTokens)
	if err != nil {
		return Result{}, err
	}
	text := m.Tokenizer.Decode(contentIDs)
	return Result{Text: text, Language: langCode}, nil
}
