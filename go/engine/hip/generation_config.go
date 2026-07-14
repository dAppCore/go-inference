// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

// generation_config.go reads engine/hip's checkpoint sampling intent — the hip
// analogue of engine/metal's generation_config.go. It delegates the byte-level
// parse (the pointer-presence semantics, the scalar-eos-beside-suppress_tokens
// shape) to the shared engine.ParseGenerationConfigSampling, so engine/hip and
// engine/metal fold the identical generation_config.json shape; this file only
// locates and reads the file (the engine package does no file I/O) and exposes
// the capability method the shared engine.TextModel folds at the decode seam.
package hip

import (
	core "dappco.re/go"
	"dappco.re/go/inference/engine"
)

// generationConfigStops parses the scalar or array eos_token_id form shipped
// by Hugging Face generation_config.json files. Malformed and absent values
// remain soft-optional so tokenizer-derived stops still apply.
func generationConfigStops(data []byte) []int32 {
	var cfg struct {
		EOSTokenID any `json:"eos_token_id"`
	}
	if result := core.JSONUnmarshal(data, &cfg); !result.OK {
		return nil
	}
	switch value := cfg.EOSTokenID.(type) {
	case float64:
		return []int32{int32(value)}
	case []any:
		ids := make([]int32, 0, len(value))
		for _, candidate := range value {
			if id, ok := candidate.(float64); ok {
				ids = append(ids, int32(id))
			}
		}
		if len(ids) > 0 {
			return ids
		}
	}
	return nil
}

func loadGenerationConfigStops(dir string) []int32 {
	read := core.ReadFile(core.PathJoin(dir, "generation_config.json"))
	if !read.OK {
		return nil
	}
	data, ok := read.Value.([]byte)
	if !ok {
		return nil
	}
	return generationConfigStops(data)
}

// loadGenerationConfigSamplingDefaults reads dir/generation_config.json and
// returns its declared sampling defaults, or the zero value when the file is
// absent — the soft-optional convention engine/metal's generation_config reader
// uses. The parse lives in the shared engine package; this shell only reads the
// file.
func loadGenerationConfigSamplingDefaults(dir string) engine.SamplingDefaults {
	read := core.ReadFile(core.PathJoin(dir, "generation_config.json"))
	if !read.OK {
		return engine.SamplingDefaults{}
	}
	data, ok := read.Value.([]byte)
	if !ok {
		return engine.SamplingDefaults{}
	}
	return engine.ParseGenerationConfigSampling(data)
}

// DeclaredSamplingDefaults reports the checkpoint's generation_config sampling
// intent (engine.SamplingDefaultsDeclarer) — the zero value when the checkpoint
// declares none. engine.TextModel folds it into each request's resolved
// GenerateConfig at the decode seam under request-set > model-declared >
// engine fallback, so a hip-served model honours its own declared
// temperature/top_k/top_p/min_p/suppress_tokens exactly as engine/metal does.
func (m *hipTokenModel) DeclaredSamplingDefaults() engine.SamplingDefaults {
	if m == nil {
		return engine.SamplingDefaults{}
	}
	return m.declaredSampling
}

// DeclaredStopTokens reports the checkpoint's complete eos_token_id set to the
// shared engine. The clone keeps model-lifetime declarations immutable to
// callers that retain and modify the returned slice.
func (m *hipTokenModel) DeclaredStopTokens() []int32 {
	if m == nil {
		return nil
	}
	return append([]int32(nil), m.declaredStops...)
}
