// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

// generation_config.go reads engine/hip's checkpoint sampling intent — the hip
// analogue of engine/metal's generation_config.go. It delegates the byte-level
// parse (the pointer-presence semantics, the scalar-eos-beside-suppress_tokens
// shape) to the shared engine.ParseGenerationConfigSampling, so engine/hip and
// engine/metal fold the identical generation_config.json shape; this file only
// locates and reads the file (the engine package does no file I/O) and exposes
// the capability method the shared engine.TextModel folds at the decode seam.
//
// Scope note: engine/hip declares its stop set separately (eos_token_id via the
// model pack) and does NOT yet implement engine.StopTokenDeclarer — this lane
// wires sampling defaults only; the stop-declarer parity is a follow-up.
package hip

import (
	core "dappco.re/go"
	"dappco.re/go/inference/engine"
)

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
