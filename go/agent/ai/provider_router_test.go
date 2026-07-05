// SPDX-License-Identifier: EUPL-1.2

package ai

import (
	"context"
	"iter"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference"
)

func TestProviderRouter_NewProviderRouter_Good_ClonesRoutes(t *testing.T) {
	model := &routerFakeModel{modelType: "external", output: "ok"}
	route := ProviderRoute{Name: "openai", ModelID: "gpt-test", Model: model, Labels: map[string]string{"tier": "remote"}}

	result := NewProviderRouter(route)
	if !result.OK {
		t.Fatalf("NewProviderRouter() error = %s", result.Error())
	}
	router := result.Value.(*ProviderRouter)

	route.Labels["tier"] = "mutated"
	providers := router.Providers()
	if len(providers) != 1 {
		t.Fatalf("Providers() len = %d, want 1", len(providers))
	}
	if providers[0].Name != "openai" || providers[0].ModelID != "gpt-test" {
		t.Fatalf("Providers()[0] = %+v, want registered route", providers[0])
	}
	if providers[0].Labels["tier"] != "remote" {
		t.Fatalf("Providers()[0].Labels = %+v, want cloned labels", providers[0].Labels)
	}
}

func TestProviderRouter_NewProviderRouter_Bad_RejectsNilModel(t *testing.T) {
	result := NewProviderRouter(ProviderRoute{Name: "broken", ModelID: "missing"})
	if result.OK {
		t.Fatal("NewProviderRouter() OK = true, want validation failure")
	}
	if !core.Contains(result.Error(), "model is required") {
		t.Fatalf("NewProviderRouter() error = %q, want model validation", result.Error())
	}
}

func TestProviderRouter_NewProviderRouter_Ugly_RejectsEmptyRoutes(t *testing.T) {
	result := NewProviderRouter()
	if result.OK {
		t.Fatal("NewProviderRouter() OK = true, want empty route failure")
	}
	if !core.Contains(result.Error(), "at least one provider") {
		t.Fatalf("NewProviderRouter() error = %q, want empty route validation", result.Error())
	}
}

func TestProviderRouter_Chat_Good_UsesFirstHealthyProvider(t *testing.T) {
	first := &routerFakeModel{modelType: "mlx", output: "local ok", metrics: inference.GenerateMetrics{PromptTokens: 3, GeneratedTokens: 2}}
	second := &routerFakeModel{modelType: "openai", output: "remote ok"}
	router := mustProviderRouter(t,
		ProviderRoute{Name: "mlx", ModelID: "gemma", Model: first},
		ProviderRoute{Name: "openai", ModelID: "gpt", Model: second},
	)

	result := router.Chat(context.Background(), ProviderChatRequest{
		Prompt:      "hello",
		MaxTokens:   8,
		Temperature: 0.2,
	})
	if !result.OK {
		t.Fatalf("Chat() error = %s", result.Error())
	}
	response := result.Value.(ProviderChatResponse)
	if response.Text != "local ok" || response.Provider != "mlx" || response.ModelID != "gemma" {
		t.Fatalf("Chat() = %+v, want first provider response", response)
	}
	if len(response.Attempts) != 1 || !response.Attempts[0].OK {
		t.Fatalf("Attempts = %+v, want one successful attempt", response.Attempts)
	}
	if first.calls != 1 || second.calls != 0 {
		t.Fatalf("calls first=%d second=%d, want first only", first.calls, second.calls)
	}
	if first.lastMessages[0].Role != "user" || first.lastMessages[0].Content != "hello" {
		t.Fatalf("messages = %+v, want prompt converted to user message", first.lastMessages)
	}
	if first.lastConfig.MaxTokens != 8 || first.lastConfig.Temperature != 0.2 {
		t.Fatalf("config = %+v, want request options", first.lastConfig)
	}
	if response.Metrics.PromptTokens != 3 || response.Metrics.GeneratedTokens != 2 {
		t.Fatalf("Metrics = %+v, want model metrics", response.Metrics)
	}
}

func TestProviderRouter_Chat_Good_PrependsRouterContext(t *testing.T) {
	model := &routerFakeModel{modelType: "mlx", output: "context ok"}
	router := mustProviderRouterWithOptions(t,
		ProviderRouterOptions{
			ContextAssembler: ProviderContextAssemblerFunc(func(_ context.Context, messages []inference.Message) core.Result {
				if len(messages) != 1 || messages[0].Content != "question" {
					t.Fatalf("assembler messages = %+v, want original user message", messages)
				}
				return core.Ok("retrieved context")
			}),
		},
		ProviderRoute{Name: "mlx", ModelID: "gemma", Model: model},
	)

	result := router.Chat(context.Background(), ProviderChatRequest{Prompt: "question"})
	if !result.OK {
		t.Fatalf("Chat() error = %s", result.Error())
	}
	response := result.Value.(ProviderChatResponse)
	if !response.ContextInjected || response.ContextBytes == 0 {
		t.Fatalf("ContextInjected=%v ContextBytes=%d, want injected context metadata", response.ContextInjected, response.ContextBytes)
	}
	if len(model.lastMessages) != 2 {
		t.Fatalf("messages len = %d, want context + user", len(model.lastMessages))
	}
	if model.lastMessages[0].Role != "system" || !core.Contains(model.lastMessages[0].Content, "retrieved context") {
		t.Fatalf("context message = %+v, want system context", model.lastMessages[0])
	}
	if model.lastMessages[1].Role != "user" || model.lastMessages[1].Content != "question" {
		t.Fatalf("user message = %+v, want original prompt preserved", model.lastMessages[1])
	}
}

func TestProviderRouter_Chat_Good_RequestContextOverridesRouterContext(t *testing.T) {
	model := &routerFakeModel{modelType: "mlx", output: "context ok"}
	router := mustProviderRouterWithOptions(t,
		ProviderRouterOptions{
			ContextAssembler: ProviderContextAssemblerFunc(func(context.Context, []inference.Message) core.Result {
				return core.Ok("router context")
			}),
		},
		ProviderRoute{Name: "mlx", ModelID: "gemma", Model: model},
	)

	result := router.Chat(context.Background(), ProviderChatRequest{
		Prompt: "question",
		ContextAssembler: ProviderContextAssemblerFunc(func(context.Context, []inference.Message) core.Result {
			return core.Ok("request context")
		}),
	})
	if !result.OK {
		t.Fatalf("Chat() error = %s", result.Error())
	}
	if !core.Contains(model.lastMessages[0].Content, "request context") || core.Contains(model.lastMessages[0].Content, "router context") {
		t.Fatalf("context message = %+v, want request context override", model.lastMessages[0])
	}
}

func TestProviderRouter_Chat_Bad_ContextAssemblerErrorFailsBeforeProvider(t *testing.T) {
	model := &routerFakeModel{modelType: "mlx", output: "should not run"}
	router := mustProviderRouterWithOptions(t,
		ProviderRouterOptions{
			ContextAssembler: ProviderContextAssemblerFunc(func(context.Context, []inference.Message) core.Result {
				return core.Fail(core.E("fake.Context", "retrieval failed", nil))
			}),
		},
		ProviderRoute{Name: "mlx", ModelID: "gemma", Model: model},
	)

	result := router.Chat(context.Background(), ProviderChatRequest{Prompt: "question"})
	if result.OK {
		t.Fatal("Chat() OK = true, want context assembler failure")
	}
	if !core.Contains(result.Error(), "retrieval failed") {
		t.Fatalf("Chat() error = %q, want context failure", result.Error())
	}
	if model.calls != 0 {
		t.Fatalf("model calls = %d, want provider untouched after context failure", model.calls)
	}
}

func TestProviderRouter_Chat_Bad_FallsBackAfterProviderError(t *testing.T) {
	first := &routerFakeModel{modelType: "mlx", err: core.E("fake.Chat", "local offline", nil)}
	second := &routerFakeModel{modelType: "openai", output: "remote ok"}
	router := mustProviderRouter(t,
		ProviderRoute{Name: "mlx", ModelID: "gemma", Model: first},
		ProviderRoute{Name: "openai", ModelID: "gpt", Model: second},
	)

	result := router.Chat(context.Background(), ProviderChatRequest{Messages: []inference.Message{{Role: "user", Content: "hello"}}})
	if !result.OK {
		t.Fatalf("Chat() error = %s", result.Error())
	}
	response := result.Value.(ProviderChatResponse)
	if response.Text != "remote ok" || response.Provider != "openai" {
		t.Fatalf("Chat() = %+v, want fallback provider response", response)
	}
	if len(response.Attempts) != 2 || response.Attempts[0].OK || response.Attempts[1].OK != true {
		t.Fatalf("Attempts = %+v, want failed first and successful second", response.Attempts)
	}
	if !core.Contains(response.Attempts[0].Error, "local offline") {
		t.Fatalf("first attempt error = %q, want provider error", response.Attempts[0].Error)
	}
}

func TestProviderRouter_Chat_Ugly_ReturnsFailureWhenAllProvidersFail(t *testing.T) {
	router := mustProviderRouter(t,
		ProviderRoute{Name: "mlx", ModelID: "gemma", Model: &routerFakeModel{err: core.E("fake.Chat", "local offline", nil)}},
		ProviderRoute{Name: "openai", ModelID: "gpt", Model: &routerFakeModel{err: core.E("fake.Chat", "remote offline", nil)}},
	)

	result := router.Chat(context.Background(), ProviderChatRequest{Prompt: "hello"})
	if result.OK {
		t.Fatal("Chat() OK = true, want all-provider failure")
	}
	if !core.Contains(result.Error(), "remote offline") {
		t.Fatalf("Chat() error = %q, want last provider error", result.Error())
	}
}

func TestProviderRouter_ProviderContextAssemblerFunc_AssembleContext_Good(t *testing.T) {
	assembler := ProviderContextAssemblerFunc(func(context.Context, []inference.Message) core.Result {
		return core.Ok("router context")
	})
	result := assembler.AssembleContext(context.Background(), nil)

	if !result.OK || result.Value.(string) != "router context" {
		t.Fatalf("ProviderContextAssemblerFunc.AssembleContext() = %#v, want context text", result)
	}
}

func TestProviderRouter_ProviderContextAssemblerFunc_AssembleContext_Bad(t *testing.T) {
	var assembler ProviderContextAssemblerFunc
	result := assembler.AssembleContext(context.Background(), nil)

	if !result.OK || result.Value.(string) != "" {
		t.Fatalf("ProviderContextAssemblerFunc.AssembleContext() = %#v, want empty context", result)
	}
}

func TestProviderRouter_ProviderContextAssemblerFunc_AssembleContext_Ugly(t *testing.T) {
	assembler := ProviderContextAssemblerFunc(func(context.Context, []inference.Message) core.Result {
		return core.Fail(core.E("test.context", "failed", nil))
	})
	result := assembler.AssembleContext(context.Background(), nil)

	if result.OK || !core.Contains(result.Error(), "failed") {
		t.Fatalf("ProviderContextAssemblerFunc.AssembleContext() = %#v, want failure", result)
	}
}

func TestProviderRouter_NewProviderRouter_Good(t *testing.T) {
	result := NewProviderRouter(ProviderRoute{Name: "local", ModelID: "model", Model: &routerFakeModel{modelType: "mlx"}})

	if !result.OK {
		t.Fatalf("NewProviderRouter() error = %s", result.Error())
	}
	if providers := result.Value.(*ProviderRouter).Providers(); len(providers) != 1 || providers[0].Name != "local" {
		t.Fatalf("NewProviderRouter() providers = %+v, want local provider", providers)
	}
}

func TestProviderRouter_NewProviderRouter_Bad(t *testing.T) {
	result := NewProviderRouter(ProviderRoute{Name: "broken"})

	if result.OK || !core.Contains(result.Error(), "model is required") {
		t.Fatalf("NewProviderRouter() = %#v, want missing model failure", result)
	}
}

func TestProviderRouter_NewProviderRouter_Ugly(t *testing.T) {
	result := NewProviderRouter()

	if result.OK || !core.Contains(result.Error(), "at least one provider") {
		t.Fatalf("NewProviderRouter() = %#v, want empty routes failure", result)
	}
}

func TestProviderRouter_NewProviderRouterWithOptions_Good(t *testing.T) {
	result := NewProviderRouterWithOptions(ProviderRouterOptions{ContextRole: "developer"}, ProviderRoute{
		Name: "local", ModelID: "model", Model: &routerFakeModel{modelType: "mlx"},
	})

	if !result.OK {
		t.Fatalf("NewProviderRouterWithOptions() error = %s", result.Error())
	}
	if role := result.Value.(*ProviderRouter).options.ContextRole; role != "developer" {
		t.Fatalf("NewProviderRouterWithOptions() ContextRole = %q, want developer", role)
	}
}

func TestProviderRouter_NewProviderRouterWithOptions_Bad(t *testing.T) {
	result := NewProviderRouterWithOptions(ProviderRouterOptions{}, ProviderRoute{Name: "broken"})

	if result.OK || !core.Contains(result.Error(), "model is required") {
		t.Fatalf("NewProviderRouterWithOptions() = %#v, want missing model failure", result)
	}
}

func TestProviderRouter_NewProviderRouterWithOptions_Ugly(t *testing.T) {
	result := NewProviderRouterWithOptions(ProviderRouterOptions{ContextRole: "  "}, ProviderRoute{
		Name: "local", ModelID: "model", Model: &routerFakeModel{modelType: "mlx"},
	})

	if !result.OK {
		t.Fatalf("NewProviderRouterWithOptions() error = %s", result.Error())
	}
	if role := result.Value.(*ProviderRouter).options.ContextRole; role != "" {
		t.Fatalf("NewProviderRouterWithOptions() ContextRole = %q, want trimmed empty role", role)
	}
}

func TestProviderRouter_ProviderRouter_Providers_Good(t *testing.T) {
	router := mustProviderRouter(t, ProviderRoute{Name: "local", ModelID: "model", Model: &routerFakeModel{modelType: "mlx"}})
	providers := router.Providers()

	if len(providers) != 1 || providers[0].Name != "local" {
		t.Fatalf("ProviderRouter.Providers() = %+v, want local provider", providers)
	}
}

func TestProviderRouter_ProviderRouter_Providers_Bad(t *testing.T) {
	var router *ProviderRouter

	if providers := router.Providers(); providers != nil {
		t.Fatalf("ProviderRouter.Providers() = %+v, want nil for nil router", providers)
	}
}

func TestProviderRouter_ProviderRouter_Providers_Ugly(t *testing.T) {
	labels := map[string]string{"tier": "remote"}
	router := mustProviderRouter(t, ProviderRoute{Name: "local", ModelID: "model", Model: &routerFakeModel{modelType: "mlx"}, Labels: labels})
	providers := router.Providers()
	providers[0].Labels["tier"] = "mutated"

	if again := router.Providers(); again[0].Labels["tier"] != "remote" {
		t.Fatalf("ProviderRouter.Providers() leaked labels = %+v", again[0].Labels)
	}
}

func TestProviderRouter_ProviderRouter_Chat_Good(t *testing.T) {
	router := mustProviderRouter(t, ProviderRoute{Name: "local", ModelID: "model", Model: &routerFakeModel{modelType: "mlx", output: "ok"}})
	result := router.Chat(context.Background(), ProviderChatRequest{Prompt: "hello"})

	if !result.OK || result.Value.(ProviderChatResponse).Text != "ok" {
		t.Fatalf("ProviderRouter.Chat() = %#v, want ok response", result)
	}
}

func TestProviderRouter_ProviderRouter_Chat_Bad(t *testing.T) {
	router := mustProviderRouter(t, ProviderRoute{Name: "local", ModelID: "model", Model: &routerFakeModel{err: core.E("fake.Chat", "offline", nil)}})
	result := router.Chat(context.Background(), ProviderChatRequest{Prompt: "hello"})

	if result.OK || !core.Contains(result.Error(), "offline") {
		t.Fatalf("ProviderRouter.Chat() = %#v, want provider failure", result)
	}
}

func TestProviderRouter_ProviderRouter_Chat_Ugly(t *testing.T) {
	router := mustProviderRouter(t, ProviderRoute{Name: "local", ModelID: "model", Model: &routerFakeModel{modelType: "mlx", output: "ok"}})
	result := router.Chat(context.Background(), ProviderChatRequest{})

	if result.OK || !core.Contains(result.Error(), "prompt or messages") {
		t.Fatalf("ProviderRouter.Chat() = %#v, want missing prompt failure", result)
	}
}

func mustProviderRouter(t *testing.T, routes ...ProviderRoute) *ProviderRouter {
	t.Helper()
	result := NewProviderRouter(routes...)
	if !result.OK {
		t.Fatalf("NewProviderRouter() error = %s", result.Error())
	}
	return result.Value.(*ProviderRouter)
}

func mustProviderRouterWithOptions(t *testing.T, options ProviderRouterOptions, routes ...ProviderRoute) *ProviderRouter {
	t.Helper()
	result := NewProviderRouterWithOptions(options, routes...)
	if !result.OK {
		t.Fatalf("NewProviderRouterWithOptions() error = %s", result.Error())
	}
	return result.Value.(*ProviderRouter)
}

type routerFakeModel struct {
	modelType string
	output    string
	tokens    []string // when set, yielded in order instead of single `output`
	err       error
	metrics   inference.GenerateMetrics

	calls        int
	lastMessages []inference.Message
	lastConfig   inference.GenerateConfig
	lastErr      error
}

func (m *routerFakeModel) Generate(ctx context.Context, prompt string, opts ...inference.GenerateOption) iter.Seq[inference.Token] {
	return m.Chat(ctx, []inference.Message{{Role: "user", Content: prompt}}, opts...)
}

func (m *routerFakeModel) Chat(ctx context.Context, messages []inference.Message, opts ...inference.GenerateOption) iter.Seq[inference.Token] {
	return func(yield func(inference.Token) bool) {
		m.calls++
		m.lastMessages = append([]inference.Message(nil), messages...)
		m.lastConfig = inference.ApplyGenerateOpts(opts)
		if ctx.Err() != nil {
			m.lastErr = ctx.Err()
			return
		}
		m.lastErr = m.err
		if m.err != nil {
			return
		}
		if len(m.tokens) > 0 {
			for _, tok := range m.tokens {
				if !yield(inference.Token{Text: tok}) {
					return
				}
			}
			return
		}
		if m.output == "" {
			return
		}
		yield(inference.Token{Text: m.output})
	}
}

func (m *routerFakeModel) Classify(context.Context, []string, ...inference.GenerateOption) core.Result {
	return core.Fail(core.E("fake.Classify", "not implemented", nil))
}

func (m *routerFakeModel) BatchGenerate(ctx context.Context, prompts []string, opts ...inference.GenerateOption) core.Result {
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
				batch.Err = core.E("fake.BatchGenerate", errResult.Error(), nil)
			}
		}
		results = append(results, batch)
	}
	return core.Ok(results)
}

func (m *routerFakeModel) ModelType() string { return m.modelType }

func (m *routerFakeModel) Info() inference.ModelInfo {
	return inference.ModelInfo{Architecture: m.modelType}
}

func (m *routerFakeModel) Metrics() inference.GenerateMetrics { return m.metrics }

func (m *routerFakeModel) Err() core.Result {
	if m.lastErr != nil {
		return core.Fail(m.lastErr)
	}
	return core.Ok(nil)
}

func (m *routerFakeModel) Close() core.Result { return core.Ok(nil) }
