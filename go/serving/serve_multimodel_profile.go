// SPDX-Licence-Identifier: EUPL-1.2

// serve_multimodel_profile.go carries the named generation-profile surface for
// multi-model serving. A ProfileConfig is a declarative preset of generation
// options (temperature, top-p, max-tokens, …) exposed as a `model:profile` id in
// /v1/models and selectable in a request's `model` field. profileModel applies a
// resolved preset AHEAD of the caller's own options — the caller's options win
// (last-set), so a profile establishes defaults a request can still override.
//
// This "profile" is deliberately distinct from serve_profile.go's tuning
// profile: that one resolves the MTP draft block from disk; this one is a
// per-model generation preset. Kept in separate files with disjoint Go types so
// the shared wire term "profile" never collides in the code.
//
// profileModel forwards the full TextModel surface (embedded interface) plus the
// vision/audio capability gates and Unwrap, so a profile never strips a
// capability — inference.BaseTextModel walks past it to reach the base model's
// EmbeddingModel/RerankModel at the /v1/embeddings and /v1/rerank gate (the
// capability-stripping bug class, guarded here by construction).

package serving

import (
	"context"
	"iter"

	core "dappco.re/go"
	"dappco.re/go/inference"
)

// ProfileConfig is a declarative generation preset. Pointer fields distinguish
// "unset" (leave the model/default) from a zero value the caller intended (e.g.
// Temperature 0.0 for greedy), so a preset overrides exactly the knobs it names.
//
//	precise := ProfileConfig{Temperature: ptrFloat32(0.0)}
//	creative := ProfileConfig{Temperature: ptrFloat32(0.9), TopP: ptrFloat32(0.95), MaxTokens: ptrInt(512)}
type ProfileConfig struct {
	Temperature   *float32 `json:"temperature,omitempty"`
	TopP          *float32 `json:"top_p,omitempty"`
	TopK          *int     `json:"top_k,omitempty"`
	MinP          *float32 `json:"min_p,omitempty"`
	MaxTokens     *int     `json:"max_tokens,omitempty"`
	RepeatPenalty *float32 `json:"repeat_penalty,omitempty"`
	Seed          *uint64  `json:"seed,omitempty"`
}

// Options renders the preset as GenerateOptions in a stable field order. Only
// set fields emit an option, so the preset overrides exactly the knobs it names
// and leaves the rest at the model/engine default.
//
//	opts := creative.Options() // WithTemperature(0.9), WithTopP(0.95), WithMaxTokens(512)
func (p ProfileConfig) Options() []inference.GenerateOption {
	var opts []inference.GenerateOption
	if p.Temperature != nil {
		opts = append(opts, inference.WithTemperature(*p.Temperature))
	}
	if p.TopP != nil {
		opts = append(opts, inference.WithTopP(*p.TopP))
	}
	if p.TopK != nil {
		opts = append(opts, inference.WithTopK(*p.TopK))
	}
	if p.MinP != nil {
		opts = append(opts, inference.WithMinP(*p.MinP))
	}
	if p.MaxTokens != nil {
		opts = append(opts, inference.WithMaxTokens(*p.MaxTokens))
	}
	if p.RepeatPenalty != nil {
		opts = append(opts, inference.WithRepeatPenalty(*p.RepeatPenalty))
	}
	if p.Seed != nil {
		opts = append(opts, inference.WithSeed(*p.Seed))
	}
	return opts
}

// profileModel decorates an inference.TextModel with a preset of generation
// options applied ahead of every Generate/Chat/Classify/BatchGenerate call. The
// preset options are prepended, so a caller option for the same knob wins (Go
// option application is last-set-wins). It forwards the full TextModel surface
// via the embedded interface and re-exposes the optional capability gates.
type profileModel struct {
	inference.TextModel
	preset []inference.GenerateOption
}

// wrapProfile decorates model so every generation call carries preset ahead of
// the caller's options. An empty preset returns the model unwrapped — no reason
// to pay a decorator hop for a profile that sets nothing.
func wrapProfile(model inference.TextModel, preset []inference.GenerateOption) inference.TextModel {
	if len(preset) == 0 {
		return model
	}
	return &profileModel{TextModel: model, preset: preset}
}

// merge returns the preset options followed by the caller's, so a caller option
// for the same knob is applied last and wins.
func (m *profileModel) merge(opts []inference.GenerateOption) []inference.GenerateOption {
	merged := make([]inference.GenerateOption, 0, len(m.preset)+len(opts))
	merged = append(merged, m.preset...)
	merged = append(merged, opts...)
	return merged
}

// Generate streams tokens with the preset options applied ahead of the caller's.
func (m *profileModel) Generate(ctx context.Context, prompt string, opts ...inference.GenerateOption) iter.Seq[inference.Token] {
	return m.TextModel.Generate(ctx, prompt, m.merge(opts)...)
}

// Chat streams a conversation turn with the preset options applied ahead of the
// caller's.
func (m *profileModel) Chat(ctx context.Context, messages []inference.Message, opts ...inference.GenerateOption) iter.Seq[inference.Token] {
	return m.TextModel.Chat(ctx, messages, m.merge(opts)...)
}

// Classify runs prefill-only inference with the preset options applied ahead of
// the caller's.
func (m *profileModel) Classify(ctx context.Context, prompts []string, opts ...inference.GenerateOption) core.Result {
	return m.TextModel.Classify(ctx, prompts, m.merge(opts)...)
}

// BatchGenerate runs batched generation with the preset options applied ahead of
// the caller's.
func (m *profileModel) BatchGenerate(ctx context.Context, prompts []string, opts ...inference.GenerateOption) core.Result {
	return m.TextModel.BatchGenerate(ctx, prompts, m.merge(opts)...)
}

// AcceptsImages forwards the vision-capability gate to the wrapped model. The
// embedded interface does not widen this decorator's method set, so without the
// explicit forward a profiled vision checkpoint 400s at the serve gate.
func (m *profileModel) AcceptsImages() bool {
	v, ok := m.TextModel.(inference.VisionModel)
	return ok && v.AcceptsImages()
}

// AcceptsAudio forwards the audio-capability gate to the wrapped model — the
// audio twin of AcceptsImages.
func (m *profileModel) AcceptsAudio() bool {
	a, ok := m.TextModel.(inference.AudioModel)
	return ok && a.AcceptsAudio()
}

// Unwrap exposes the wrapped model so the serving layer can reach optional
// capabilities this profile decorator does not itself re-expose (embeddings,
// rerank). A generation preset has no bearing on an embedding call, so serving
// it through the base model is correct. Without this, a profiled embedder would
// be stripped of its EmbeddingModel/RerankModel interface at the capability gate.
func (m *profileModel) Unwrap() inference.TextModel {
	return m.TextModel
}
