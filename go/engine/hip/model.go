//go:build linux && amd64 && rocm_legacy_server

package hip

import (
	"context"
	"iter"
	"sync"
	"time"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/inference/engine/hip/internal/llamacpp"
)

// rocmModel implements inference.TextModel using a llama-server subprocess.
type rocmModel struct {
	server        *server
	modelPath     string
	modelType     string
	modelInfo     inference.ModelInfo
	contextLength int

	stateMutex  sync.Mutex
	lastError   error
	lastMetrics inference.GenerateMetrics
}

// Generate streams tokens for the given prompt via llama-server's /v1/completions endpoint.
func (m *rocmModel) Generate(ctx context.Context, prompt string, opts ...inference.GenerateOption) iter.Seq[inference.Token] {
	m.clearLastError()

	if !m.server.alive() {
		m.setServerExitErr()
		return func(yield func(inference.Token) bool) {}
	}

	generateConfig := inference.ApplyGenerateOpts(opts)
	request := newCompletionRequest(prompt, generateConfig)
	promptTokens := approximatePromptTokens(prompt)

	start := time.Now()
	chunks, streamError := m.server.llamaClient.Complete(ctx, request)

	return func(yield func(inference.Token) bool) {
		var count int
		var firstTokenAt time.Time
		for text := range chunks {
			if firstTokenAt.IsZero() {
				firstTokenAt = time.Now()
			}
			count++
			if !yield(inference.Token{Text: text}) {
				break
			}
		}
		if err := streamError(); err != nil {
			m.setLastFailure(err)
		}
		m.recordMetrics(promptTokens, count, start, firstTokenAt)
	}
}

// Chat streams tokens from a multi-turn conversation via llama-server's /v1/chat/completions endpoint.
func (m *rocmModel) Chat(ctx context.Context, messages []inference.Message, opts ...inference.GenerateOption) iter.Seq[inference.Token] {
	m.clearLastError()

	if !m.server.alive() {
		m.setServerExitErr()
		return func(yield func(inference.Token) bool) {}
	}

	generateConfig := inference.ApplyGenerateOpts(opts)
	promptTokens := approximateMessageTokens(messages)

	chatMsgs := make([]llamacpp.ChatMessage, len(messages))
	for i, msg := range messages {
		chatMsgs[i] = llamacpp.ChatMessage{
			Role:    msg.Role,
			Content: msg.Content,
		}
	}
	request := newChatRequest(chatMsgs, generateConfig)

	start := time.Now()
	chunks, streamError := m.server.llamaClient.ChatComplete(ctx, request)

	return func(yield func(inference.Token) bool) {
		var count int
		var firstTokenAt time.Time
		for text := range chunks {
			if firstTokenAt.IsZero() {
				firstTokenAt = time.Now()
			}
			count++
			if !yield(inference.Token{Text: text}) {
				break
			}
		}
		if err := streamError(); err != nil {
			m.setLastFailure(err)
		}
		m.recordMetrics(promptTokens, count, start, firstTokenAt)
	}
}

// Classify runs batched prefill-only inference via llama-server.
// Each prompt gets a single-token completion (max_tokens=1) while honoring
// the sampling settings from opts. llama-server has no native classify
// endpoint, so this simulates it.
func (m *rocmModel) Classify(ctx context.Context, prompts []string, opts ...inference.GenerateOption) core.Result {
	return core.ResultOf(m.classifyResults(ctx, prompts, opts...))
}

func (m *rocmModel) classifyResults(ctx context.Context, prompts []string, opts ...inference.GenerateOption) (
	[]inference.ClassifyResult,
	error,
) {
	if !m.server.alive() {
		m.setServerExitErr()
		return nil, m.currentError()
	}

	generateConfig := inference.ApplyGenerateOpts(opts)
	results := make([]inference.ClassifyResult, len(prompts))
	totalPromptTokens := 0
	totalGenerated := 0
	var totalPrefill time.Duration
	var totalDecode time.Duration

	for promptIndex, prompt := range prompts {
		if contextError := ctx.Err(); contextError != nil {
			m.recordMetricsDurations(totalPromptTokens, totalGenerated, totalPrefill, totalDecode)
			return nil, core.E("rocm.Classify", core.Sprintf("classify cancelled before prompt %d", promptIndex), contextError)
		}

		totalPromptTokens += approximatePromptTokens(prompt)
		request := newCompletionRequest(prompt, generateConfig)
		request.MaxTokens = 1

		requestStart := time.Now()
		chunks, streamError := m.server.llamaClient.Complete(ctx, request)
		text := core.NewBuilder()
		var firstTokenAt time.Time
		var generated int
		for chunk := range chunks {
			if firstTokenAt.IsZero() {
				firstTokenAt = time.Now()
			}
			generated++
			text.WriteString(chunk)
		}
		requestEnd := time.Now()
		prefill, decode := splitDurations(requestStart, firstTokenAt, requestEnd)
		totalPrefill += prefill
		totalDecode += decode
		totalGenerated += generated

		if err := streamError(); err != nil {
			m.recordMetricsDurations(totalPromptTokens, totalGenerated, totalPrefill, totalDecode)
			return nil, core.E("rocm.Classify", core.Sprintf("classify prompt %d", promptIndex), err)
		}

		results[promptIndex] = inference.ClassifyResult{
			Token: inference.Token{Text: text.String()},
		}
	}

	m.recordMetricsDurations(totalPromptTokens, totalGenerated, totalPrefill, totalDecode)
	return results, nil
}

// BatchGenerate runs batched autoregressive generation via llama-server.
// Each prompt is decoded sequentially up to MaxTokens.
func (m *rocmModel) BatchGenerate(ctx context.Context, prompts []string, opts ...inference.GenerateOption) core.Result {
	return core.ResultOf(m.batchGenerateResults(ctx, prompts, opts...))
}

func (m *rocmModel) batchGenerateResults(ctx context.Context, prompts []string, opts ...inference.GenerateOption) (
	[]inference.BatchResult,
	error,
) {
	if !m.server.alive() {
		m.setServerExitErr()
		return nil, m.currentError()
	}

	generateConfig := inference.ApplyGenerateOpts(opts)
	results := make([]inference.BatchResult, len(prompts))
	totalPromptTokens := 0
	var totalGenerated int
	var totalPrefill time.Duration
	var totalDecode time.Duration

	for promptIndex, prompt := range prompts {
		if contextError := ctx.Err(); contextError != nil {
			results[promptIndex].Err = core.E("rocm.BatchGenerate", core.Sprintf("batch prompt %d cancelled before start", promptIndex), contextError)
			continue
		}

		totalPromptTokens += approximatePromptTokens(prompt)
		request := newCompletionRequest(prompt, generateConfig)

		requestStart := time.Now()
		chunks, streamError := m.server.llamaClient.Complete(ctx, request)
		var tokens []inference.Token
		var firstTokenAt time.Time
		for text := range chunks {
			if firstTokenAt.IsZero() {
				firstTokenAt = time.Now()
			}
			tokens = append(tokens, inference.Token{Text: text})
		}
		requestEnd := time.Now()
		prefill, decode := splitDurations(requestStart, firstTokenAt, requestEnd)
		totalPrefill += prefill
		totalDecode += decode
		results[promptIndex].Tokens = tokens
		totalGenerated += len(tokens)

		if err := streamError(); err != nil {
			results[promptIndex].Err = core.E("rocm.BatchGenerate", core.Sprintf("batch prompt %d", promptIndex), err)
		}
	}

	m.recordMetricsDurations(totalPromptTokens, totalGenerated, totalPrefill, totalDecode)
	return results, nil
}

// ModelType returns the architecture identifier (e.g. "gemma3", "qwen3", "llama3").
func (m *rocmModel) ModelType() string { return m.modelType }

// Info returns metadata about the loaded model.
func (m *rocmModel) Info() inference.ModelInfo {
	if m == nil {
		return inference.ModelInfo{}
	}
	info := m.modelInfo
	architecture := firstNonEmptyString(info.Architecture, m.ModelType())
	if info == (inference.ModelInfo{}) && architecture == "" && m.modelPath == "" {
		return inference.ModelInfo{}
	}
	identity := inference.ModelIdentity{
		Path:          m.modelPath,
		Architecture:  architecture,
		VocabSize:     info.VocabSize,
		NumLayers:     info.NumLayers,
		HiddenSize:    info.HiddenSize,
		QuantBits:     info.QuantBits,
		QuantGroup:    info.QuantGroup,
		ContextLength: m.contextLength,
	}
	return modelInfoFromIdentity(rocmGemma4ModelWithInferredPathQuant(identity))
}

func modelInfoFromIdentity(model inference.ModelIdentity) inference.ModelInfo {
	return inference.ModelInfo{
		Architecture: normalizeROCmArchitecture(model.Architecture),
		VocabSize:    model.VocabSize,
		NumLayers:    model.NumLayers,
		HiddenSize:   model.HiddenSize,
		QuantBits:    model.QuantBits,
		QuantGroup:   model.QuantGroup,
	}
}

func (m *rocmModel) ModelIdentity() inference.ModelIdentity {
	if m == nil {
		return inference.ModelIdentity{}
	}
	return rocmCloneModelIdentity(m.modelIdentity())
}

func (m *rocmModel) modelIdentity() inference.ModelIdentity {
	info := m.Info()
	if info.Architecture == "" {
		info.Architecture = m.ModelType()
	}
	return rocmGemma4ModelWithInferredPathQuant(inference.ModelIdentity{
		Path:          m.modelPath,
		Architecture:  normalizeROCmArchitecture(info.Architecture),
		VocabSize:     info.VocabSize,
		NumLayers:     info.NumLayers,
		HiddenSize:    info.HiddenSize,
		QuantBits:     info.QuantBits,
		QuantGroup:    info.QuantGroup,
		ContextLength: m.contextLength,
	})
}

func (m *rocmModel) ModelProfile() ROCmModelProfile {
	if m == nil {
		return ROCmModelProfile{}
	}
	identity := m.modelIdentity()
	profile, ok := ResolveROCmModelProfile(identity.Path, identity)
	if !ok {
		return ROCmModelProfile{}
	}
	return profile
}

func (m *rocmModel) ModelRoutePlan() ROCmModelRoutePlan {
	profile := m.ModelProfile()
	if !profile.Matched() {
		return ROCmModelRoutePlan{}
	}
	plan := ROCmModelRoutePlanForProfile(profile)
	return rocmModelRoutePlanWithLiveCacheProfile(plan, m)
}

func (m *rocmModel) Capabilities() inference.CapabilityReport {
	if m == nil {
		return inference.CapabilityReport{Runtime: inference.RuntimeIdentity{Backend: "rocm"}}
	}
	identity := m.modelIdentity()
	profile := m.ModelProfile()
	available := m.server != nil && m.server.alive()
	runtimeStatus := "unavailable"
	if available {
		runtimeStatus = "available"
	}
	labels := rocmLegacyMergeStringMaps(map[string]string{
		"backend":                      "rocm",
		"native_runtime":               "llama_server",
		"runtime_status":               runtimeStatus,
		"production_requires_env_gate": "false",
		"production_requires_cli_flag": "false",
	}, identity.Labels)
	if profile.Matched() {
		labels = ApplyROCmModelProfileLabels(labels, profile)
		labels = ApplyROCmModelRoutePlanLabels(labels, ROCmModelRoutePlanForProfileAndModel(profile, m))
	}
	capabilities := []inference.Capability{
		inference.SupportedCapability(inference.CapabilityModelLoad, inference.CapabilityGroupRuntime),
		inference.SupportedCapability(inference.CapabilityGenerate, inference.CapabilityGroupModel),
		inference.SupportedCapability(inference.CapabilityChat, inference.CapabilityGroupModel),
		inference.SupportedCapability(inference.CapabilityClassify, inference.CapabilityGroupModel),
		inference.SupportedCapability(inference.CapabilityBatchGenerate, inference.CapabilityGroupModel),
	}
	if profile.Matched() {
		for _, id := range profile.EngineFeatures.EnabledCapabilities() {
			capabilities = rocmLegacySetCapability(capabilities, inference.SupportedCapability(id, inference.CapabilityGroupModel))
		}
	}
	for index := range capabilities {
		capabilities[index].Labels = cloneStringMap(labels)
	}
	return inference.CapabilityReport{
		Runtime: inference.RuntimeIdentity{
			Backend:       "rocm",
			NativeRuntime: false,
			Labels: map[string]string{
				"native_runtime":               "llama_server",
				"production_requires_env_gate": "false",
				"production_requires_cli_flag": "false",
			},
		},
		Model:        rocmCloneModelIdentity(identity),
		Available:    available,
		Capabilities: capabilities,
		Labels:       cloneStringMap(labels),
	}
}

func rocmLegacySetCapability(capabilities []inference.Capability, capability inference.Capability) []inference.Capability {
	if capability.ID == "" {
		return capabilities
	}
	for index := range capabilities {
		if capabilities[index].ID == capability.ID {
			capabilities[index] = capability
			return capabilities
		}
	}
	return append(capabilities, capability)
}

func rocmLegacyMergeStringMaps(left, right map[string]string) map[string]string {
	out := cloneStringMap(left)
	if out == nil {
		out = map[string]string{}
	}
	for key, value := range right {
		out[key] = value
	}
	return out
}

// Metrics returns performance metrics from the last inference operation.
func (m *rocmModel) Metrics() inference.GenerateMetrics {
	m.stateMutex.Lock()
	defer m.stateMutex.Unlock()
	return m.lastMetrics
}

// Err returns the error from the last Generate/Chat call, if any.
func (m *rocmModel) Err() core.Result {
	return core.ResultOf(nil, m.currentError())
}

func (m *rocmModel) currentError() error {
	m.stateMutex.Lock()
	defer m.stateMutex.Unlock()
	return m.lastError
}

// Close releases the llama-server subprocess and all associated resources.
func (m *rocmModel) Close() core.Result {
	return core.ResultOf(nil, m.closeModel())
}

func (m *rocmModel) closeModel() error {
	if m == nil || m.server == nil {
		return nil
	}
	return m.server.stop()
}

// setServerExitErr stores an appropriate error when the server is dead.
func (m *rocmModel) setServerExitErr() {
	m.stateMutex.Lock()
	defer m.stateMutex.Unlock()
	if m.server == nil {
		m.lastError = core.E("rocm.setServerExitErr", "server is not started", nil)
		return
	}
	if m.server.processExitError != nil {
		m.lastError = m.server.processFailure("rocm.setServerExitErr", "server has exited", m.server.processExitError)
	} else {
		m.lastError = core.E("rocm.setServerExitErr", m.server.messageWithProcessOutput("server has exited unexpectedly"), nil)
	}
}

// recordMetrics captures timing data from an inference operation.
func (m *rocmModel) recordMetrics(promptTokens, generatedTokens int, start, firstTokenAt time.Time) {
	prefill, decode := splitDurations(start, firstTokenAt, time.Now())
	m.recordMetricsDurations(promptTokens, generatedTokens, prefill, decode)
}

func (m *rocmModel) recordMetricsDurations(promptTokens, generatedTokens int, prefill, decode time.Duration) {
	if prefill < 0 {
		prefill = 0
	}
	if decode < 0 {
		decode = 0
	}
	total := prefill + decode

	metrics := inference.GenerateMetrics{
		PromptTokens:    promptTokens,
		GeneratedTokens: generatedTokens,
		PrefillDuration: prefill,
		DecodeDuration:  decode,
		TotalDuration:   total,
	}
	if prefill > 0 && promptTokens > 0 {
		metrics.PrefillTokensPerSec = float64(promptTokens) / prefill.Seconds()
	}
	if decode > 0 && generatedTokens > 0 {
		metrics.DecodeTokensPerSec = float64(generatedTokens) / decode.Seconds()
	}

	// Try to get VRAM stats — best effort.
	if vram, err := GetVRAMInfo(); err == nil {
		metrics.PeakMemoryBytes = vram.Used
		metrics.ActiveMemoryBytes = vram.Used
	}

	m.stateMutex.Lock()
	m.lastMetrics = metrics
	m.stateMutex.Unlock()
}

func (m *rocmModel) clearLastError() {
	m.setLastFailure(nil)
}

func (m *rocmModel) setLastFailure(
	err error,
) {
	m.stateMutex.Lock()
	m.lastError = err
	m.stateMutex.Unlock()
}

func newCompletionRequest(prompt string, generateConfig inference.GenerateConfig) llamacpp.CompletionRequest {
	return llamacpp.CompletionRequest{
		Prompt:        prompt,
		MaxTokens:     generateConfig.MaxTokens,
		Temperature:   generateConfig.Temperature,
		TopK:          generateConfig.TopK,
		TopP:          generateConfig.TopP,
		RepeatPenalty: generateConfig.RepeatPenalty,
	}
}

func newChatRequest(messages []llamacpp.ChatMessage, generateConfig inference.GenerateConfig) llamacpp.ChatRequest {
	return llamacpp.ChatRequest{
		Messages:      messages,
		MaxTokens:     generateConfig.MaxTokens,
		Temperature:   generateConfig.Temperature,
		TopK:          generateConfig.TopK,
		TopP:          generateConfig.TopP,
		RepeatPenalty: generateConfig.RepeatPenalty,
	}
}

func splitDurations(start, firstTokenAt, end time.Time) (time.Duration, time.Duration) {
	if start.IsZero() || end.Before(start) {
		return 0, 0
	}
	if firstTokenAt.IsZero() || firstTokenAt.Before(start) || firstTokenAt.After(end) {
		return end.Sub(start), 0
	}
	return firstTokenAt.Sub(start), end.Sub(firstTokenAt)
}

// llama-server's streaming API does not expose prompt token counts, so metrics
// use a lightweight whitespace-token approximation for prefill throughput.
func approximatePromptTokens(prompt string) int {
	trimmed := core.Trim(prompt)
	if trimmed == "" {
		return 0
	}
	return len(core.Split(trimmed, " "))
}

func approximateMessageTokens(messages []inference.Message) int {
	total := 0
	for _, msg := range messages {
		total += approximatePromptTokens(msg.Content)
	}
	return total
}
