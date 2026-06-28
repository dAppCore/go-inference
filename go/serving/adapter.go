// SPDX-Licence-Identifier: EUPL-1.2

package serving

import (
	"context"

	"dappco.re/go"
	"dappco.re/go/inference"
)

// InferenceAdapter bridges a go-inference TextModel (iter.Seq[Token]) to the
// serving.Backend and serving.StreamingBackend interfaces (string returns / TokenCallback).
//
// This is the key adapter for Phase 1: any go-inference backend (MLX Metal,
// ROCm, llama.cpp) can be wrapped to satisfy the serving.Backend contract.
type InferenceAdapter struct {
	model inference.TextModel
	name  string
}

// Compile-time checks.
var _ Backend = (*InferenceAdapter)(nil)
var _ StreamingBackend = (*InferenceAdapter)(nil)
var _ inference.CapabilityReporter = (*InferenceAdapter)(nil)

// NewInferenceAdapter wraps a go-inference TextModel as an serving.Backend and
// serving.StreamingBackend. The name is used for Backend.Name() (e.g. "mlx").
func NewInferenceAdapter(model inference.TextModel, name string) *InferenceAdapter {
	return &InferenceAdapter{model: model, name: name}
}

// Generate collects all tokens from the model's iterator into a single string.
//
//	r := a.Generate(ctx, "hello", serving.DefaultGenOpts())
//	if !r.OK { return r }
//	resp := r.Value.(serving.Result)
func (a *InferenceAdapter) Generate(ctx context.Context, prompt string, opts GenOpts) core.Result {
	inferOpts := convertOpts(opts)
	b := core.NewBuilder()
	for tok := range a.model.Generate(ctx, prompt, inferOpts...) {
		b.WriteString(tok.Text)
	}
	text := applyStopSequences(b.String(), opts.StopSequences)
	if r := a.model.Err(); !r.OK {
		return r
	}
	return core.Ok(newResult(text, metricsPtr(a.model)))
}

// Chat sends a multi-turn conversation to the underlying TextModel and collects
// all tokens. Since serving.Message is now a type alias for inference.Message, no
// conversion is needed.
//
//	r := a.Chat(ctx, messages, serving.DefaultGenOpts())
//	if !r.OK { return r }
//	resp := r.Value.(serving.Result)
func (a *InferenceAdapter) Chat(ctx context.Context, messages []Message, opts GenOpts) core.Result {
	inferOpts := convertOpts(opts)
	b := core.NewBuilder()
	for tok := range a.model.Chat(ctx, messages, inferOpts...) {
		b.WriteString(tok.Text)
	}
	text := applyStopSequences(b.String(), opts.StopSequences)
	if r := a.model.Err(); !r.OK {
		return r
	}
	return core.Ok(newResult(text, metricsPtr(a.model)))
}

// GenerateStream forwards each generated token's text to the callback.
// Returns nil on success, the callback's error if it stops early, or the
// model's error if generation fails.
//
//	r := a.GenerateStream(ctx, "hello", opts, func(tok string) error { ... })
//	if !r.OK { return r }
func (a *InferenceAdapter) GenerateStream(ctx context.Context, prompt string, opts GenOpts, cb TokenCallback) core.Result {
	inferOpts := convertOpts(opts)
	if len(opts.StopSequences) == 0 {
		for tok := range a.model.Generate(ctx, prompt, inferOpts...) {
			if err := cb(tok.Text); err != nil {
				return core.Fail(err)
			}
		}
		return a.model.Err()
	}

	full := core.NewBuilder()
	emitted := 0
	for tok := range a.model.Generate(ctx, prompt, inferOpts...) {
		full.WriteString(tok.Text)
		truncated := applyStopSequences(full.String(), opts.StopSequences)
		if len(truncated) > emitted {
			if err := cb(truncated[emitted:]); err != nil {
				return core.Fail(err)
			}
			emitted = len(truncated)
		}
		if len(truncated) < full.Len() {
			return a.model.Err()
		}
	}
	return a.model.Err()
}

// ChatStream forwards each generated chat token's text to the callback.
// Since serving.Message is now a type alias for inference.Message, no conversion
// is needed.
//
//	r := a.ChatStream(ctx, messages, opts, func(tok string) error { ... })
//	if !r.OK { return r }
func (a *InferenceAdapter) ChatStream(ctx context.Context, messages []Message, opts GenOpts, cb TokenCallback) core.Result {
	inferOpts := convertOpts(opts)
	if len(opts.StopSequences) == 0 {
		for tok := range a.model.Chat(ctx, messages, inferOpts...) {
			if err := cb(tok.Text); err != nil {
				return core.Fail(err)
			}
		}
		return a.model.Err()
	}

	full := core.NewBuilder()
	emitted := 0
	for tok := range a.model.Chat(ctx, messages, inferOpts...) {
		full.WriteString(tok.Text)
		truncated := applyStopSequences(full.String(), opts.StopSequences)
		if len(truncated) > emitted {
			if err := cb(truncated[emitted:]); err != nil {
				return core.Fail(err)
			}
			emitted = len(truncated)
		}
		if len(truncated) < full.Len() {
			return a.model.Err()
		}
	}
	return a.model.Err()
}

// Name returns the backend identifier set at construction.
func (a *InferenceAdapter) Name() string { return a.name }

// Available always returns true — the model is already loaded.
func (a *InferenceAdapter) Available() bool { return true }

// Close delegates to the underlying TextModel.Close(), releasing GPU memory
// and other resources.
func (a *InferenceAdapter) Close() core.Result {
	return a.model.Close()
}

// Model returns the underlying go-inference TextModel for direct access
// to Classify, BatchGenerate, Metrics, Info, etc.
func (a *InferenceAdapter) Model() inference.TextModel { return a.model }

// Capabilities returns the shared feature report for the wrapped inference model.
func (a *InferenceAdapter) Capabilities() inference.CapabilityReport {
	if a == nil || a.model == nil {
		name := ""
		if a != nil {
			name = a.name
		}
		return inference.CapabilityReport{Runtime: inference.RuntimeIdentity{Backend: name}}
	}
	report, ok := inference.CapabilitiesOf(a.model)
	if !ok {
		report = inference.TextModelCapabilities(inference.RuntimeIdentity{Backend: a.name}, a.model)
	}
	if report.Runtime.Backend == "" {
		report.Runtime.Backend = a.name
	}
	report.Available = a.Available()
	return report
}

// InspectAttention delegates to the underlying TextModel if it implements
// inference.AttentionInspector. Returns an error if the backend does not support
// attention inspection.
//
//	r := a.InspectAttention(ctx, "hello")
//	if !r.OK { return r }
//	snap := r.Value.(*inference.AttentionSnapshot)
func (a *InferenceAdapter) InspectAttention(ctx context.Context, prompt string, opts ...inference.GenerateOption) core.Result {
	inspector, ok := a.model.(inference.AttentionInspector)
	if !ok {
		return core.Fail(core.E("serving.InferenceAdapter.InspectAttention", core.Sprintf("backend %q does not support attention inspection", a.name), nil))
	}
	snap, err := inspector.InspectAttention(ctx, prompt, opts...)
	return core.ResultOf(snap, err)
}

// convertOpts maps serving.GenOpts to go-inference functional options.
func convertOpts(opts GenOpts) []inference.GenerateOption {
	// Count the active options first so the backing array is allocated once
	// at the exact capacity, instead of paying append's geometric regrowth
	// (nil→1→2→4→8) on every request. The With* closures remain — they are
	// allocated inside the inference package's functional-options API.
	n := 0
	if opts.Temperature != 0 {
		n++
	}
	if opts.MaxTokens != 0 {
		n++
	}
	if opts.TopK > 0 {
		n++
	}
	if opts.TopP > 0 {
		n++
	}
	if opts.RepeatPenalty > 0 {
		n++
	}
	if len(opts.StopTokens) > 0 {
		n++
	}
	if n == 0 {
		return nil
	}

	out := make([]inference.GenerateOption, 0, n)
	if opts.Temperature != 0 {
		out = append(out, inference.WithTemperature(float32(opts.Temperature)))
	}
	if opts.MaxTokens != 0 {
		out = append(out, inference.WithMaxTokens(opts.MaxTokens))
	}
	if opts.TopK > 0 {
		out = append(out, inference.WithTopK(opts.TopK))
	}
	if opts.TopP > 0 {
		out = append(out, inference.WithTopP(float32(opts.TopP)))
	}
	if opts.RepeatPenalty > 0 {
		out = append(out, inference.WithRepeatPenalty(float32(opts.RepeatPenalty)))
	}
	if len(opts.StopTokens) > 0 {
		out = append(out, inference.WithStopTokens(opts.StopTokens...))
	}
	// GenOpts.Model is ignored — the model is already loaded.
	return out
}

// metricsPtr returns a copy of the model's latest metrics, or nil if unavailable.
func metricsPtr(m inference.TextModel) *inference.GenerateMetrics {
	met := m.Metrics()
	return &met
}
