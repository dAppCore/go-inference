// SPDX-Licence-Identifier: EUPL-1.2

package serving

import (
	"context"
	"iter"

	"dappco.re/go"
	"dappco.re/go/inference"
)

// HTTPTextModel wraps an HTTPBackend with CoreGo Result-returning helpers.
// This enables cross-platform consistency — HTTP backends can be used anywhere
// that expects a go-inference TextModel (e.g. go-ai, go-i18n).
//
// Generate and Chat yield the entire HTTP response as a single Token since
// the OpenAI-compatible API returns complete responses (non-streaming).
type HTTPTextModel struct {
	http    *HTTPBackend
	lastErr error
}

// NewHTTPTextModel wraps an HTTPBackend as an inference.TextModel.
func NewHTTPTextModel(backend *HTTPBackend) *HTTPTextModel {
	return &HTTPTextModel{http: backend}
}

// Generate sends a single prompt to the HTTP backend and yields the entire
// response as a single Token.
func (m *HTTPTextModel) Generate(ctx context.Context, prompt string, opts ...inference.GenerateOption) iter.Seq[inference.Token] {
	return func(yield func(inference.Token) bool) {
		cfg := inference.ApplyGenerateOpts(opts)
		genOpts := GenOpts{
			Temperature: float64(cfg.Temperature),
			MaxTokens:   cfg.MaxTokens,
			Model:       m.http.Model(),
		}
		r := m.http.Generate(ctx, prompt, genOpts)
		if !r.OK {
			m.lastErr = r.Value.(error)
			return
		}
		m.lastErr = nil
		yield(inference.Token{Text: r.Value.(Result).Text})
	}
}

// Chat sends a multi-turn conversation to the HTTP backend and yields the
// entire response as a single Token.
func (m *HTTPTextModel) Chat(ctx context.Context, messages []inference.Message, opts ...inference.GenerateOption) iter.Seq[inference.Token] {
	return func(yield func(inference.Token) bool) {
		cfg := inference.ApplyGenerateOpts(opts)
		genOpts := GenOpts{
			Temperature: float64(cfg.Temperature),
			MaxTokens:   cfg.MaxTokens,
			Model:       m.http.Model(),
		}

		// ml.Message is now a type alias for inference.Message — no conversion needed.
		r := m.http.Chat(ctx, messages, genOpts)
		if !r.OK {
			m.lastErr = r.Value.(error)
			return
		}
		m.lastErr = nil
		yield(inference.Token{Text: r.Value.(Result).Text})
	}
}

// Classify is not supported by HTTP backends. Returns an error.
func (m *HTTPTextModel) Classify(_ context.Context, _ []string, _ ...inference.GenerateOption) core.Result {
	return core.Fail(core.E("ml.HTTPTextModel.Classify", "classify not supported by HTTP backend", nil))
}

// BatchGenerate processes multiple prompts sequentially via Generate.
func (m *HTTPTextModel) BatchGenerate(ctx context.Context, prompts []string, opts ...inference.GenerateOption) core.Result {
	results := make([]inference.BatchResult, len(prompts))
	for i, prompt := range prompts {
		var tokens []inference.Token
		for tok := range m.Generate(ctx, prompt, opts...) {
			tokens = append(tokens, tok)
		}
		results[i] = inference.BatchResult{
			Tokens: tokens,
			Err:    m.lastErr,
		}
	}
	return core.Ok(results)
}

// ModelType returns the configured model name from the underlying HTTPBackend.
func (m *HTTPTextModel) ModelType() string {
	model := m.http.Model()
	if model == "" {
		return "http"
	}
	return model
}

// Info returns minimal model metadata for an HTTP backend.
func (m *HTTPTextModel) Info() inference.ModelInfo {
	return inference.ModelInfo{Architecture: "http"}
}

// Metrics returns zero metrics — HTTP backends don't track token-level performance.
func (m *HTTPTextModel) Metrics() inference.GenerateMetrics {
	return inference.GenerateMetrics{}
}

// Err returns the error from the last Generate or Chat call, if any.
func (m *HTTPTextModel) Err() core.Result {
	return core.ResultOf(nil, m.lastErr)
}

// Close is a no-op for HTTP backends — there are no resources to release.
func (m *HTTPTextModel) Close() core.Result {
	return core.Ok(nil)
}

// LlamaTextModel wraps a LlamaBackend as an inference.TextModel. It embeds
// HTTPTextModel for Generate/Chat but overrides ModelType and Close to
// reflect the managed llama-server process.
type LlamaTextModel struct {
	*HTTPTextModel
	llama *LlamaBackend
}

// NewLlamaTextModel wraps a LlamaBackend as an inference.TextModel.
func NewLlamaTextModel(backend *LlamaBackend) *LlamaTextModel {
	return &LlamaTextModel{
		HTTPTextModel: NewHTTPTextModel(backend.http),
		llama:         backend,
	}
}

// ModelType returns "llama" to identify this as a managed llama-server backend.
func (m *LlamaTextModel) ModelType() string {
	return "llama"
}

// Close stops the managed llama-server process.
func (m *LlamaTextModel) Close() core.Result {
	return m.llama.Stop()
}
