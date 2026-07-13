// SPDX-Licence-Identifier: EUPL-1.2

package engine

import "dappco.re/go/inference"

// sampling_defaults.go is the sampling sibling of the generation_config stop
// set (StopTokenDeclarer, model.go): HF checkpoints ship generation_config.json
// with a do_sample/temperature/top_p/top_k/min_p/suppress_tokens block
// declaring how the model wants to be sampled. A request that leaves a sampling
// parameter unset should get the model's declared default rather than a
// hardcoded engine fallback — the same declares-discipline precedence stop
// tokens already follow: request-set > model-declared > engine fallback.

// SamplingDefaults is a checkpoint's declared generation_config.json sampling
// intent. Pointer fields report "the file declared this key" (non-nil) versus
// "the file said nothing" (nil) — the JSON-presence signal a parser can always
// recover. That is the model side of the same request-set > model-declared >
// engine fallback precedence [SamplingDefaultsDeclarer] describes; the REQUEST
// side reports "was this set" through inference.GenerateConfig's companion
// TemperatureSet/TopKSet/TopPSet/MinPSet flags.
type SamplingDefaults struct {
	DoSample       *bool
	Temperature    *float32
	TopP           *float32
	TopK           *int
	MinP           *float32
	SuppressTokens []int32
}

// Apply folds these declared defaults into a request's resolved GenerateConfig
// under the declares-discipline precedence — request-set wins, the declared
// value applies only where the request left the field unset, and a field
// neither the request nor the model set keeps its engine fallback (the zero
// value). It is the single home of the precedence rule so every decode path
// (the plain [TextModel] seam and the speculative pair) folds identically
// rather than re-deriving it.
//
// "Unset" is read per field from the signal that field actually carries:
//   - Temperature/TopP/TopK/MinP: the request's companion *Set flag is false.
//     The scalar's zero value is a meaningful explicit request (Temperature 0 =
//     greedy, the others 0 = disabled), so the flag — not the value — is the
//     only honest unset signal (see inference.GenerateConfig).
//   - SuppressTokens: the request's slice is empty. A nil slice has no other
//     meaning here (inference.WithSuppressTokens always clones a real slice).
//
// DoSample is parsed and carried but NOT folded: GenerateConfig has no
// caller-intent counterpart for it (the engine derives sample-vs-greedy from
// the resolved Temperature/MinP/RepeatPenalty), so folding a declared
// Temperature is what actually flips a request into the model's intended
// sampling mode.
func (d SamplingDefaults) Apply(cfg inference.GenerateConfig) inference.GenerateConfig {
	if !cfg.TemperatureSet && d.Temperature != nil {
		cfg.Temperature = *d.Temperature
	}
	if !cfg.TopPSet && d.TopP != nil {
		cfg.TopP = *d.TopP
	}
	if !cfg.TopKSet && d.TopK != nil {
		cfg.TopK = *d.TopK
	}
	if !cfg.MinPSet && d.MinP != nil {
		cfg.MinP = *d.MinP
	}
	if len(cfg.SuppressTokens) == 0 && len(d.SuppressTokens) > 0 {
		cfg.SuppressTokens = d.SuppressTokens
	}
	return cfg
}

// SamplingDefaultsDeclarer is the optional [TokenModel] capability for
// checkpoints that declare sampling defaults (generation_config.json's
// do_sample/temperature/top_p/top_k/min_p/suppress_tokens).
//
// [TextModel] folds the declared values into a request's resolved
// inference.GenerateConfig at the decode seam via [SamplingDefaults.Apply],
// honouring request-set > model-declared > engine fallback. The request-side
// "was this set" signal is GenerateConfig's TemperatureSet/TopKSet/TopPSet/
// MinPSet flags (set by the With* option setters), which resolve the
// zero-value-vs-unset ambiguity these scalar fields would otherwise carry — a
// request explicitly asking Temperature 0 on a model declaring 0.7 stays
// greedy because its TemperatureSet flag is true.
type SamplingDefaultsDeclarer interface {
	DeclaredSamplingDefaults() SamplingDefaults
}

// resolveDeclaredSampling reads a TokenModel's declared sampling defaults (via
// SamplingDefaultsDeclarer) once — the same construction-time resolution
// NewTextModel already applies to thoughtSuppressor/chatTmpl, so the common
// per-request path pays no type assertion. The zero value when tm does not
// declare the capability.
func resolveDeclaredSampling(tm TokenModel) SamplingDefaults {
	if d, ok := tm.(SamplingDefaultsDeclarer); ok {
		return d.DeclaredSamplingDefaults()
	}
	return SamplingDefaults{}
}

// applyDeclaredSampling folds the model's declared sampling defaults into cfg —
// the [TextModel] seam over [SamplingDefaults.Apply], reading the defaults
// resolved once at construction. With no generation_config declaration, or a
// request that set every field, cfg is returned unchanged (byte-for-byte).
func (m *TextModel) applyDeclaredSampling(cfg inference.GenerateConfig) inference.GenerateConfig {
	return m.declaredSampling.Apply(cfg)
}
