// SPDX-Licence-Identifier: EUPL-1.2

package engine

import "dappco.re/go/inference"

// sampling_defaults.go is the sampling sibling of the generation_config stop
// set (StopTokenDeclarer, model.go): HF checkpoints ship generation_config.json
// with a do_sample/temperature/top_p/top_k/suppress_tokens block declaring how
// the model wants to be sampled. A request that leaves a sampling parameter
// unset should get the model's declared default rather than a hardcoded engine
// fallback — the same declares-discipline precedence stop tokens already
// follow: request-set > model-declared > engine fallback.

// SamplingDefaults is a checkpoint's declared generation_config.json sampling
// intent. Pointer fields report "the file declared this key" (non-nil) versus
// "the file said nothing" (nil) — the JSON-presence signal a parser can always
// recover. That is a DIFFERENT problem from the one described on
// [SamplingDefaultsDeclarer]: whether a REQUEST left a field unset.
type SamplingDefaults struct {
	DoSample       *bool
	Temperature    *float32
	TopP           *float32
	TopK           *int
	SuppressTokens []int32
}

// SamplingDefaultsDeclarer is the optional [TokenModel] capability for
// checkpoints that declare sampling defaults (generation_config.json's
// do_sample/temperature/top_p/top_k/suppress_tokens).
//
// [TextModel] folds a declared value into a request's resolved
// inference.GenerateConfig only where "the caller left this field unset" is
// unambiguous — true today for SuppressTokens alone, because a nil slice has
// no other meaning (inference.WithSuppressTokens always clones a real slice,
// so nil can only mean "never called").
//
// Temperature, TopP, and TopK do NOT get the same treatment, and this is a
// deliberate gap, not an oversight: inference.GenerateConfig represents each as
// a plain zero-value float32/int, and the zero value is ALSO the documented,
// meaningful explicit request — Temperature 0 is greedy, TopP/TopK 0 is
// disabled (see options.go). A request that never touched the field and one
// that explicitly asked for greedy/disabled are the identical bit pattern by
// the time it reaches this capability, so applying the declared value on sight
// would silently overrule an explicit caller request — precisely the silent
// decision this fold must not make. Distinguishing the two needs a companion
// "was this set" signal on GenerateConfig (Seed already carries one, SeedSet)
// or pointer-typed option fields; that is a public API change to the inference
// package, outside what an engine-level capability gets to decide.
//
// DoSample has no GenerateConfig counterpart at all: the engine derives
// sample-vs-greedy from the resolved Temperature/MinP/RepeatPenalty rather than
// a caller intent flag, so there is nothing to fold it into. It is parsed and
// carried on [SamplingDefaults] regardless, so the day GenerateConfig gains
// per-field "set" flags, wiring the rest of this capability in is a
// consumption-side change only — the declared data is already flowing.
type SamplingDefaultsDeclarer interface {
	DeclaredSamplingDefaults() SamplingDefaults
}

// declaredSuppressTokens resolves a TokenModel's declared suppress_tokens (via
// SamplingDefaultsDeclarer) once — the same construction-time resolution
// NewTextModel already applies to thoughtSuppressor/chatTmpl, so the common
// per-request path pays no type assertion. Empty when tm does not declare, or
// declares an empty/absent list.
func declaredSuppressTokens(tm TokenModel) []int32 {
	d, ok := tm.(SamplingDefaultsDeclarer)
	if !ok {
		return nil
	}
	return d.DeclaredSamplingDefaults().SuppressTokens
}

// applyDeclaredSuppressTokens folds the checkpoint's declared suppress_tokens
// into cfg when the request carries none of its own — request-set wins
// outright; the declared list applies only to an unset request; with neither,
// cfg is returned unchanged (today's behaviour, byte-for-byte). See
// [SamplingDefaultsDeclarer] for why Temperature/TopP/TopK/DoSample stay
// unfolded here.
func (m *TextModel) applyDeclaredSuppressTokens(cfg inference.GenerateConfig) inference.GenerateConfig {
	if len(cfg.SuppressTokens) == 0 && len(m.declaredSuppressTokens) > 0 {
		cfg.SuppressTokens = m.declaredSuppressTokens
	}
	return cfg
}
