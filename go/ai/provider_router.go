// SPDX-License-Identifier: EUPL-1.2

package ai

import (
	"context"

	core "dappco.re/go"
	"dappco.re/go/inference"
)

// ProviderRoute describes one local or external model that can satisfy a chat
// request through the shared inference contract.
type ProviderRoute struct {
	Name    string
	ModelID string
	Model   inference.TextModel
	Labels  map[string]string
}

// ProviderChatRequest is the package-level chat shape used by the inference stack routing
// policy. It remains backend-neutral: local runtimes and external providers
// both arrive here as inference.TextModel implementations.
type ProviderChatRequest struct {
	Messages []inference.Message
	Prompt   string

	MaxTokens   int
	Temperature float32
	TopK        int
	TopP        float32
	Options     []inference.GenerateOption

	ContextAssembler ProviderContextAssembler
	ContextRole      string
	ContextPrefix    string
	DisableContext   bool

	Labels map[string]string
}

// ProviderContextAssembler optionally adds retrieval/context-pack material to
// a routed request before the selected provider sees it.
type ProviderContextAssembler interface {
	AssembleContext(context.Context, []inference.Message) core.Result
}

// ProviderContextAssemblerFunc adapts a function to ProviderContextAssembler.
type ProviderContextAssemblerFunc func(context.Context, []inference.Message) core.Result

func (fn ProviderContextAssemblerFunc) AssembleContext(ctx context.Context, messages []inference.Message) core.Result {
	if fn == nil {
		return core.Ok("")
	}
	return fn(ctx, messages)
}

// ProviderRouterOptions carries policy that applies across provider fallback
// attempts. It stays in the inference stack because context assembly is product policy, not a
// go-inference primitive.
type ProviderRouterOptions struct {
	ContextAssembler ProviderContextAssembler
	ContextRole      string
	ContextPrefix    string
}

// ProviderAttempt records each provider tried by ProviderRouter.Chat.
type ProviderAttempt struct {
	Provider string
	ModelID  string
	OK       bool
	Error    string
}

// ProviderChatResponse carries the selected provider output and enough route
// metadata for callers to audit fallback behaviour.
type ProviderChatResponse struct {
	Text     string
	Provider string
	ModelID  string
	Metrics  inference.GenerateMetrics
	Attempts []ProviderAttempt
	Labels   map[string]string

	ContextInjected bool
	ContextBytes    int
}

// ProviderRouter applies the inference stack provider policy over shared inference models.
type ProviderRouter struct {
	routes  []ProviderRoute
	options ProviderRouterOptions
}

// NewProviderRouter creates a fallback router over local and external models.
func NewProviderRouter(routes ...ProviderRoute) core.Result {
	return NewProviderRouterWithOptions(ProviderRouterOptions{}, routes...)
}

// NewProviderRouterWithOptions creates a fallback router with shared the inference stack
// policy such as optional retrieval context injection.
func NewProviderRouterWithOptions(options ProviderRouterOptions, routes ...ProviderRoute) core.Result {
	if len(routes) == 0 {
		return core.Fail(core.E("ai.NewProviderRouter", "at least one provider route is required", nil))
	}

	cloned := make([]ProviderRoute, 0, len(routes))
	for i, route := range routes {
		if route.Model == nil {
			return core.Fail(core.E("ai.NewProviderRouter", core.Sprintf("provider route %d model is required", i), nil))
		}
		cloned = append(cloned, normaliseProviderRoute(route, i))
	}
	return core.Ok(&ProviderRouter{routes: cloned, options: normaliseProviderRouterOptions(options)})
}

// Providers returns the configured route order.
func (r *ProviderRouter) Providers() []ProviderRoute {
	if r == nil || len(r.routes) == 0 {
		return nil
	}
	out := make([]ProviderRoute, 0, len(r.routes))
	for _, route := range r.routes {
		out = append(out, cloneProviderRoute(route))
	}
	return out
}

// Chat tries each provider in order until one completes without a model error.
func (r *ProviderRouter) Chat(ctx context.Context, request ProviderChatRequest) core.Result {
	if r == nil || len(r.routes) == 0 {
		return core.Fail(core.E("ai.ProviderRouter.Chat", "provider router has no routes", nil))
	}

	messages := request.normalisedMessages()
	if len(messages) == 0 {
		return core.Fail(core.E("ai.ProviderRouter.Chat", "prompt or messages are required", nil))
	}
	contextResult := r.contextMessages(ctx, request, messages)
	if !contextResult.OK {
		return contextResult
	}
	contextState := contextResult.Value.(providerContextState)
	messages = contextState.messages

	options := request.generateOptions()
	attempts := make([]ProviderAttempt, 0, len(r.routes))
	lastFailure := core.Result{}

	for _, route := range r.routes {
		if err := ctx.Err(); err != nil {
			return core.Fail(core.E("ai.ProviderRouter.Chat", "request cancelled", err))
		}

		providerResult := chatProvider(ctx, route, messages, options)
		attempt := ProviderAttempt{Provider: route.Name, ModelID: route.ModelID}
		if !providerResult.OK {
			attempt.Error = providerResult.Error()
			attempts = append(attempts, attempt)
			lastFailure = providerResult
			continue
		}
		providerResponse := providerResult.Value.(chatProviderResponse)

		attempt.OK = true
		attempts = append(attempts, attempt)
		return core.Ok(ProviderChatResponse{
			Text:     providerResponse.text,
			Provider: route.Name,
			ModelID:  route.ModelID,
			Metrics:  providerResponse.metrics,
			Attempts: attempts,
			Labels:   core.MapClone(request.Labels),

			ContextInjected: contextState.injected,
			ContextBytes:    contextState.bytes,
		})
	}

	if !lastFailure.OK && lastFailure.Value == nil {
		lastFailure = core.Fail(core.E("ai.ProviderRouter.Chat", "all providers failed", nil))
	}
	if err, ok := lastFailure.Value.(error); ok {
		return core.Fail(core.E("ai.ProviderRouter.Chat", core.Sprintf("all providers failed: %s", err.Error()), err))
	}
	return core.Fail(core.E("ai.ProviderRouter.Chat", core.Sprintf("all providers failed: %s", lastFailure.Error()), nil))
}

func (r ProviderChatRequest) normalisedMessages() []inference.Message {
	if len(r.Messages) > 0 {
		return append([]inference.Message(nil), r.Messages...)
	}
	prompt := core.Trim(r.Prompt)
	if prompt == "" {
		return nil
	}
	return []inference.Message{{Role: "user", Content: prompt}}
}

func (r ProviderChatRequest) generateOptions() []inference.GenerateOption {
	options := make([]inference.GenerateOption, 0, len(r.Options)+4)
	if r.MaxTokens > 0 {
		options = append(options, inference.WithMaxTokens(r.MaxTokens))
	}
	if r.Temperature != 0 {
		options = append(options, inference.WithTemperature(r.Temperature))
	}
	if r.TopK > 0 {
		options = append(options, inference.WithTopK(r.TopK))
	}
	if r.TopP > 0 {
		options = append(options, inference.WithTopP(r.TopP))
	}
	options = append(options, r.Options...)
	return options
}

type providerContextState struct {
	messages []inference.Message
	injected bool
	bytes    int
}

func (r *ProviderRouter) contextMessages(ctx context.Context, request ProviderChatRequest, messages []inference.Message) core.Result {
	// Resolve assembler before cloning — when no context is going to be
	// injected (DisableContext, or no assembler configured) we can hand
	// the caller's slice straight through. The downstream chatProvider
	// path is read-only; cloning here is wasted work that fires on every
	// router.Chat call in the hot-path bench. The clone is only needed
	// when an assembler runs (to protect the caller from in-place
	// mutation) or when a context message is prepended (the prepend
	// already builds a fresh slice).
	if request.DisableContext {
		return core.Ok(providerContextState{messages: messages})
	}

	assembler := request.ContextAssembler
	if assembler == nil {
		assembler = r.options.ContextAssembler
	}
	if assembler == nil {
		return core.Ok(providerContextState{messages: messages})
	}

	// Clone before exposing to the assembler so a mutating implementation
	// can't leak changes back to the caller's slice.
	out := append([]inference.Message(nil), messages...)

	contextResult := assembler.AssembleContext(ctx, out)
	if !contextResult.OK {
		if err, ok := contextResult.Value.(error); ok {
			return core.Fail(core.E("ai.ProviderRouter.Chat", "assemble context", err))
		}
		return core.Fail(core.E("ai.ProviderRouter.Chat", contextResult.Error(), nil))
	}
	contextText, _ := contextResult.Value.(string)
	contextText = core.Trim(contextText)
	if contextText == "" {
		return core.Ok(providerContextState{messages: out})
	}

	role := firstNonEmpty(request.ContextRole, r.options.ContextRole, "system")
	prefix := firstNonEmpty(request.ContextPrefix, r.options.ContextPrefix, "Context:\n")
	contextMessage := inference.Message{
		Role:    role,
		Content: core.Concat(prefix, contextText),
	}
	out = append([]inference.Message{contextMessage}, out...)
	return core.Ok(providerContextState{
		messages: out,
		injected: true,
		bytes:    len([]byte(contextText)),
	})
}

type chatProviderResponse struct {
	text    string
	metrics inference.GenerateMetrics
}

func chatProvider(ctx context.Context, route ProviderRoute, messages []inference.Message, options []inference.GenerateOption) core.Result {
	// Use a Builder to aggregate the streamed token sequence. The old
	// shape did text = core.Concat(text, token.Text) per yielded token
	// which is O(N^2): each iteration allocates a progressively larger
	// joined string and copies the prior contents in. Builder grows the
	// internal buffer amortised O(1) per write.
	b := core.NewBuilder()
	for token := range route.Model.Chat(ctx, messages, options...) {
		b.WriteString(token.Text)
	}
	if errResult := route.Model.Err(); !errResult.OK {
		return errResult
	}
	return core.Ok(chatProviderResponse{text: b.String(), metrics: route.Model.Metrics()})
}

func normaliseProviderRouterOptions(options ProviderRouterOptions) ProviderRouterOptions {
	out := options
	out.ContextRole = core.Trim(out.ContextRole)
	return out
}

func normaliseProviderRoute(route ProviderRoute, index int) ProviderRoute {
	out := cloneProviderRoute(route)
	if core.Trim(out.Name) == "" {
		out.Name = core.Trim(out.Model.ModelType())
	}
	if core.Trim(out.Name) == "" {
		out.Name = core.Sprintf("provider-%d", index+1)
	}
	if core.Trim(out.ModelID) == "" {
		info := out.Model.Info()
		out.ModelID = core.Trim(info.Architecture)
	}
	if core.Trim(out.ModelID) == "" {
		out.ModelID = out.Name
	}
	return out
}

func cloneProviderRoute(route ProviderRoute) ProviderRoute {
	route.Labels = core.MapClone(route.Labels)
	return route
}

func firstNonEmpty(values ...string) string {
	for _, value := range values {
		if core.Trim(value) != "" {
			return value
		}
	}
	return ""
}
