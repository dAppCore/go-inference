// SPDX-License-Identifier: EUPL-1.2

// Package openai provides an outbound OpenAI-compatible provider backend for
// inference consumers. It implements the shared inference contracts without
// importing local GPU runtimes or core/api.
package openai

import (
	"context"
	"io"
	"iter"
	"net/http"
	"sync"
	"time"

	core "dappco.re/go"
	"dappco.re/go/inference"
	openaicompat "dappco.re/go/inference/openai"
)

const (
	defaultProviderName = "openai"
	defaultHTTPTimeout  = 60 * time.Second
)

// Limiter is satisfied by *ratelimit.RateLimiter without forcing this package
// to own quota policy.
type Limiter interface {
	WaitForCapacity(context.Context, string, int) error
	RecordUsage(model string, promptTokens, outputTokens int)
}

// ContextAssembler optionally injects retrieval/context-pack material before a
// provider request. go-rag adapters can satisfy this shape without creating a
// dependency cycle.
type ContextAssembler interface {
	AssembleContext(context.Context, []inference.Message) core.Result
}

// ContextAssemblerFunc adapts a function to ContextAssembler.
type ContextAssemblerFunc func(context.Context, []inference.Message) core.Result

func (fn ContextAssemblerFunc) AssembleContext(ctx context.Context, messages []inference.Message) core.Result {
	if fn == nil {
		return core.Ok("")
	}
	return fn(ctx, messages)
}

// Config describes one OpenAI-compatible external provider.
type Config struct {
	Name             string
	BaseURL          string
	APIKey           string
	Organisation     string
	Project          string
	DefaultModel     string
	HTTPClient       *http.Client
	Limiter          Limiter
	ContextAssembler ContextAssembler
	EstimateTokens   func([]inference.Message, inference.GenerateConfig) int
}

// Backend implements inference.Backend for an external OpenAI-compatible
// provider.
type Backend struct {
	cfg Config
}

var _ inference.Backend = (*Backend)(nil)
var _ inference.CapabilityReporter = (*Backend)(nil)

// NewBackend creates an outbound OpenAI-compatible provider backend.
func NewBackend(cfg Config) *Backend {
	cfg.Name = defaultString(cfg.Name, defaultProviderName)
	cfg.BaseURL = trimTrailingSlash(cfg.BaseURL)
	return &Backend{cfg: cfg}
}

// Register creates and registers an outbound provider backend with the shared
// inference registry.
func Register(cfg Config) *Backend {
	backend := NewBackend(cfg)
	inference.Register(backend)
	return backend
}

// Name implements inference.Backend.
func (b *Backend) Name() string {
	if b == nil {
		return defaultProviderName
	}
	return defaultString(b.cfg.Name, defaultProviderName)
}

// Available reports whether the provider has enough static configuration to
// attempt requests.
func (b *Backend) Available() bool {
	return b != nil && core.Trim(b.cfg.BaseURL) != "" && core.Trim(b.cfg.DefaultModel) != ""
}

// LoadModel creates a lightweight model handle for the requested provider
// model. path is interpreted as the provider model id; an empty path uses
// Config.DefaultModel.
func (b *Backend) LoadModel(path string, _ ...inference.LoadOption) core.Result {
	if b == nil {
		return core.Fail(core.E("ai.openai.LoadModel", "backend is nil", nil))
	}
	modelID := core.Trim(path)
	if modelID == "" {
		modelID = core.Trim(b.cfg.DefaultModel)
	}
	if modelID == "" {
		return core.Fail(core.E("ai.openai.LoadModel", "model id is required", nil))
	}
	if core.Trim(b.cfg.BaseURL) == "" {
		return core.Fail(core.E("ai.openai.LoadModel", "base URL is required", nil))
	}
	return core.Ok(&Model{
		backend: b,
		modelID: modelID,
		client:  httpClient(b.cfg.HTTPClient),
	})
}

// Capabilities implements inference.CapabilityReporter.
func (b *Backend) Capabilities() inference.CapabilityReport {
	baseURL := ""
	if b != nil {
		baseURL = core.Trim(b.cfg.BaseURL)
	}
	return inference.CapabilityReport{
		Runtime: inference.RuntimeIdentity{
			Backend:       b.Name(),
			Device:        "external",
			NativeRuntime: false,
			Labels: map[string]string{
				"provider": "openai-compatible",
				"base_url": baseURL,
			},
		},
		Available: b.Available(),
		Capabilities: []inference.Capability{
			inference.SupportedCapability(inference.CapabilityModelLoad, inference.CapabilityGroupRuntime),
			inference.SupportedCapability(inference.CapabilityGenerate, inference.CapabilityGroupModel),
			inference.SupportedCapability(inference.CapabilityChat, inference.CapabilityGroupModel),
		},
	}
}

// Model is a loaded external provider model handle.
type Model struct {
	backend *Backend
	modelID string
	client  *http.Client

	mu      sync.Mutex
	lastErr error
	metrics inference.GenerateMetrics
}

var _ inference.TextModel = (*Model)(nil)
var _ inference.CapabilityReporter = (*Model)(nil)

type completionResult struct {
	content string
	metrics inference.GenerateMetrics
}

// Generate implements inference.TextModel.
func (m *Model) Generate(ctx context.Context, prompt string, opts ...inference.GenerateOption) iter.Seq[inference.Token] {
	return m.Chat(ctx, []inference.Message{{Role: "user", Content: prompt}}, opts...)
}

// Chat implements inference.TextModel.
func (m *Model) Chat(ctx context.Context, messages []inference.Message, opts ...inference.GenerateOption) iter.Seq[inference.Token] {
	return func(yield func(inference.Token) bool) {
		result := m.complete(ctx, messages, opts...)
		if !result.OK {
			m.setResult(inference.GenerateMetrics{}, result)
			return
		}
		completion := result.Value.(completionResult)
		m.setResult(completion.metrics, core.Ok(nil))
		if completion.content == "" {
			return
		}
		yield(inference.Token{Text: completion.content})
	}
}

// Classify is not exposed for external chat providers yet.
func (m *Model) Classify(context.Context, []string, ...inference.GenerateOption) core.Result {
	return core.Fail(core.E("ai.openai.Classify", "classification is not supported by this provider backend", nil))
}

// BatchGenerate runs Generate sequentially for each prompt.
func (m *Model) BatchGenerate(ctx context.Context, prompts []string, opts ...inference.GenerateOption) core.Result {
	results := make([]inference.BatchResult, 0, len(prompts))
	for _, prompt := range prompts {
		var tokens []inference.Token
		for token := range m.Generate(ctx, prompt, opts...) {
			tokens = append(tokens, token)
		}
		batch := inference.BatchResult{Tokens: tokens}
		if errResult := m.Err(); !errResult.OK {
			if err, ok := errResult.Value.(error); ok {
				batch.Err = err
			} else {
				batch.Err = core.E("ai.openai.BatchGenerate", errResult.Error(), nil)
			}
		}
		results = append(results, batch)
	}
	return core.Ok(results)
}

// ModelType implements inference.TextModel.
func (m *Model) ModelType() string {
	return "openai-compatible"
}

// Info implements inference.TextModel.
func (m *Model) Info() inference.ModelInfo {
	return inference.ModelInfo{Architecture: "openai-compatible"}
}

// Metrics implements inference.TextModel.
func (m *Model) Metrics() inference.GenerateMetrics {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.metrics
}

// Err implements inference.TextModel.
func (m *Model) Err() core.Result {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.lastErr != nil {
		return core.Fail(m.lastErr)
	}
	return core.Ok(nil)
}

// Close implements inference.TextModel.
func (m *Model) Close() core.Result {
	return core.Ok(nil)
}

// Capabilities implements inference.CapabilityReporter.
func (m *Model) Capabilities() inference.CapabilityReport {
	backendName := defaultProviderName
	baseURL := ""
	if m != nil && m.backend != nil {
		backendName = m.backend.Name()
		baseURL = core.Trim(m.backend.cfg.BaseURL)
	}
	modelID := ""
	if m != nil {
		modelID = m.modelID
	}
	return inference.CapabilityReport{
		Runtime: inference.RuntimeIdentity{
			Backend:       backendName,
			Device:        "external",
			NativeRuntime: false,
			Labels: map[string]string{
				"provider": "openai-compatible",
				"base_url": baseURL,
			},
		},
		Model: inference.ModelIdentity{
			ID:           modelID,
			Architecture: "openai-compatible",
			Labels: map[string]string{
				"provider": "openai-compatible",
			},
		},
		Available: true,
		Capabilities: []inference.Capability{
			inference.SupportedCapability(inference.CapabilityGenerate, inference.CapabilityGroupModel),
			inference.SupportedCapability(inference.CapabilityChat, inference.CapabilityGroupModel),
		},
	}
}

func (m *Model) complete(ctx context.Context, messages []inference.Message, opts ...inference.GenerateOption) core.Result {
	if m == nil || m.backend == nil {
		return core.Fail(core.E("ai.openai.complete", "model is nil", nil))
	}
	cfg := inference.ApplyGenerateOpts(opts)
	contextResult := m.contextMessages(ctx, messages)
	if !contextResult.OK {
		return contextResult
	}
	messages = contextResult.Value.([]inference.Message)
	if limiter := m.backend.cfg.Limiter; limiter != nil {
		if err := limiter.WaitForCapacity(ctx, m.modelID, m.estimateTokens(messages, cfg)); err != nil {
			return core.Fail(err)
		}
	}

	req := openaicompat.ChatCompletionRequest{
		Model:    m.modelID,
		Messages: openaiMessages(messages),
		Stream:   false,
	}
	if cfg.MaxTokens > 0 {
		req.MaxTokens = &cfg.MaxTokens
	}
	req.Temperature = &cfg.Temperature
	if cfg.TopP > 0 {
		req.TopP = &cfg.TopP
	}
	if cfg.TopK > 0 {
		req.TopK = &cfg.TopK
	}

	started := time.Now()
	responseResult := m.doRequest(ctx, req)
	if !responseResult.OK {
		return responseResult
	}
	response := responseResult.Value.(openaicompat.ChatCompletionResponse)
	metrics := inference.GenerateMetrics{
		PromptTokens:    response.Usage.PromptTokens,
		GeneratedTokens: response.Usage.CompletionTokens,
		TotalDuration:   time.Since(started),
	}
	if limiter := m.backend.cfg.Limiter; limiter != nil {
		limiter.RecordUsage(m.modelID, response.Usage.PromptTokens, response.Usage.CompletionTokens)
	}
	if len(response.Choices) == 0 {
		return core.Fail(core.E("ai.openai.complete", "provider response contained no choices", nil))
	}
	return core.Ok(completionResult{content: response.Choices[0].Message.Content, metrics: metrics})
}

func (m *Model) contextMessages(ctx context.Context, messages []inference.Message) core.Result {
	// Resolve assembler before cloning — the no-assembler path is the
	// common configuration when callers don't opt into context injection
	// and the caller's slice can be handed straight through. The clone
	// only matters when an assembler runs (to protect the caller from
	// in-place mutation) or when a context message is prepended (the
	// prepend already builds a fresh slice).
	assembler := m.backend.cfg.ContextAssembler
	if assembler == nil {
		return core.Ok(messages)
	}
	out := append([]inference.Message(nil), messages...)
	contextResult := assembler.AssembleContext(ctx, out)
	if !contextResult.OK {
		return contextResult
	}
	contextText, _ := contextResult.Value.(string)
	contextText = core.Trim(contextText)
	if contextText == "" {
		return core.Ok(out)
	}
	contextMessage := inference.Message{
		Role:    "system",
		Content: core.Concat("Context:\n", contextText),
	}
	out = append([]inference.Message{contextMessage}, out...)
	return core.Ok(out)
}

func (m *Model) doRequest(ctx context.Context, req openaicompat.ChatCompletionRequest) core.Result {
	payload := core.JSONMarshalString(req)
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, chatCompletionsURL(m.backend.cfg.BaseURL), core.NewReader(payload))
	if err != nil {
		return core.Fail(core.E("ai.openai.doRequest", "create request", err))
	}
	httpReq.Header.Set("Content-Type", "application/json")
	if key := core.Trim(m.backend.cfg.APIKey); key != "" {
		httpReq.Header.Set("Authorization", core.Concat("Bearer ", key))
	}
	if organisation := core.Trim(m.backend.cfg.Organisation); organisation != "" {
		httpReq.Header.Set("OpenAI-Organization", organisation)
	}
	if project := core.Trim(m.backend.cfg.Project); project != "" {
		httpReq.Header.Set("OpenAI-Project", project)
	}

	resp, err := m.client.Do(httpReq)
	if err != nil {
		return core.Fail(core.E("ai.openai.doRequest", "provider request", err))
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return core.Fail(core.E("ai.openai.doRequest", "read provider response", err))
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return providerError(resp.StatusCode, string(body))
	}
	var out openaicompat.ChatCompletionResponse
	result := core.JSONUnmarshalString(string(body), &out)
	if !result.OK {
		if err, ok := result.Value.(error); ok {
			return core.Fail(core.E("ai.openai.doRequest", "decode provider response", err))
		}
		return core.Fail(core.E("ai.openai.doRequest", result.Error(), nil))
	}
	return core.Ok(out)
}

func (m *Model) estimateTokens(messages []inference.Message, cfg inference.GenerateConfig) int {
	if estimate := m.backend.cfg.EstimateTokens; estimate != nil {
		return estimate(messages, cfg)
	}
	totalRunes := 0
	for _, msg := range messages {
		totalRunes += core.RuneCount(msg.Content)
	}
	estimate := totalRunes / 4
	if estimate < 1 {
		estimate = 1
	}
	if cfg.MaxTokens > 0 {
		estimate += cfg.MaxTokens
	}
	return estimate
}

func (m *Model) setResult(metrics inference.GenerateMetrics, status core.Result) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.metrics = metrics
	if status.OK {
		m.lastErr = nil
		return
	}
	if err, ok := status.Value.(error); ok {
		m.lastErr = err
		return
	}
	m.lastErr = core.E("ai.openai.result", status.Error(), nil)
}

func openaiMessages(messages []inference.Message) []openaicompat.ChatMessage {
	out := make([]openaicompat.ChatMessage, 0, len(messages))
	for _, msg := range messages {
		out = append(out, openaicompat.ChatMessage{Role: msg.Role, Content: msg.Content})
	}
	return out
}

func chatCompletionsURL(baseURL string) string {
	// Native + over core.Concat for 2-string join: native concat allocates
	// the result once at exact length (1 alloc, len(a)+len(b) bytes); the
	// Builder behind core.Concat does 2 allocs because its first grow is
	// not pre-sized for the joined result.
	return trimTrailingSlash(baseURL) + openaicompat.DefaultChatCompletionsPath
}

func providerError(status int, body string) core.Result {
	// Empty body: skip JSON parse + status-only message.
	if body == "" {
		return core.Fail(core.E("ai.openai.provider", core.Sprintf("provider returned HTTP %d", status), nil))
	}
	// Non-JSON body (typical 5xx HTML / plain text): skip the JSON parser
	// allocs/error path entirely. Real provider errors are JSON objects
	// starting with '{'.
	if body[0] == '{' {
		var payload openaicompat.ErrorResponse
		if result := core.JSONUnmarshalString(body, &payload); result.OK && payload.Error.Message != "" {
			return core.Fail(core.E("ai.openai.provider", core.Sprintf("provider returned HTTP %d: %s", status, payload.Error.Message), nil))
		}
	}
	return core.Fail(core.E("ai.openai.provider", core.Sprintf("provider returned HTTP %d: %s", status, body), nil))
}

func httpClient(client *http.Client) *http.Client {
	if client != nil {
		return client
	}
	return &http.Client{Timeout: defaultHTTPTimeout}
}

func defaultString(value, fallback string) string {
	if core.Trim(value) == "" {
		return fallback
	}
	return value
}

func trimTrailingSlash(value string) string {
	value = core.Trim(value)
	for core.HasSuffix(value, "/") {
		value = core.TrimSuffix(value, "/")
	}
	return value
}
