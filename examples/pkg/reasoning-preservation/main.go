// SPDX-Licence-Identifier: EUPL-1.2

// The gemma4 canonical-template reasoning-preservation rule (2026-07-09): a stateless client
// replaying agentic history echoes back each assistant turn's chain of thought (OpenAI-wire
// reasoning / reasoning_content); the serving layer re-frames it into the native thought span —
// ALWAYS for a turn after the last user message (the live in-flight exchange keeps its own
// thinking), and, under chat_template_kwargs.preserve_thinking, for a tool-calling assistant turn
// ANYWHERE in history (an agentic tool loop keeps its reasoning across the replay). Earlier
// plain-answer turns always replay clean — the model never sees stale chain-of-thought it didn't
// just produce.
//
// requestMessages (go/serving/provider/openai/handler.go), which does this re-framing, is
// unexported — this example drives it the only way an external caller can: through the public
// HTTP handler, httptest, and a fake model whose Chat method records exactly what it was called
// with (the same fake-model pattern go/serving/compat's own tests use). One replayed agentic
// history is POSTed twice — preserve_thinking false and true — and the two renders are compared.
//
//	go run ./pkg/reasoning-preservation
package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"iter"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/inference/decode/parser"
	openaicompat "dappco.re/go/inference/serving/provider/openai"
)

// Indices into history below — named so the receipts read as claims, not magic numbers.
const (
	preToolCallTurn  = 1 // pre-last-user assistant turn WITH a tool call
	prePlainTurn     = 3 // pre-last-user assistant turn with NO tool call
	postToolCallTurn = 5 // post-last-user assistant turn (the live in-flight exchange)
)

func main() {
	// A replayed agentic history: two user turns bracket an earlier tool-calling exchange (which
	// preserve_thinking alone can rescue) and an earlier plain answer (which it can't — only
	// tool-calling turns qualify), then the SAME user turn's own in-flight assistant response
	// (always kept, no flag needed).
	history := []openaicompat.ChatMessage{
		{Role: "user", Content: "What's the weather in London?"},
		{ // preToolCallTurn: pre-last-user, HAS a tool call
			Role:      "assistant",
			Reasoning: "The user wants London's weather; I should call the weather tool.",
			ToolCalls: []openaicompat.ToolCall{{ID: "call_1", Type: "function", Function: openaicompat.ToolCallFunction{Name: "get_weather", Arguments: `{"city":"London"}`}}},
		},
		{Role: "tool", Content: "15C, cloudy"},
		{ // prePlainTurn: pre-last-user, no tool call
			Role:      "assistant",
			Content:   "It's 15C and cloudy in London.",
			Reasoning: "I have the tool result, so I can answer directly now.",
		},
		{Role: "user", Content: "Now check Paris too, and tell me which city is warmer."},
		{ // postToolCallTurn: post-last-user — the live in-flight turn
			Role:      "assistant",
			Reasoning: "Paris is likely warmer this time of year; I'll call the tool to confirm before comparing.",
			ToolCalls: []openaicompat.ToolCall{{ID: "call_2", Type: "function", Function: openaicompat.ToolCallFunction{Name: "get_weather", Arguments: `{"city":"Paris"}`}}},
		},
		{Role: "tool", Content: "22C, sunny"},
	}

	model := &fakeModel{}
	server := httptest.NewServer(openaicompat.NewHandler(openaicompat.NewStaticResolver(map[string]inference.TextModel{
		"test-model": model,
	})))
	defer server.Close()

	withoutPreserve, err := chat(server.URL, model, history, false)
	if err != nil {
		fmt.Fprintln(os.Stderr, "chat (preserve_thinking=false):", err)
		os.Exit(1)
	}
	withPreserve, err := chat(server.URL, model, history, true)
	if err != nil {
		fmt.Fprintln(os.Stderr, "chat (preserve_thinking=true):", err)
		os.Exit(1)
	}

	fmt.Println("=== preserve_thinking=false — rendered history model.Chat received ===")
	printMessages(withoutPreserve)
	fmt.Println("=== preserve_thinking=true — rendered history model.Chat received ===")
	printMessages(withPreserve)

	fmt.Println("--- receipts ---")
	ok := true
	ok = check("pre-last-user tool-call turn, preserve_thinking=false: reasoning DROPPED", !hasThought(withoutPreserve[preToolCallTurn].Content)) && ok
	ok = check("pre-last-user tool-call turn, preserve_thinking=true:  reasoning KEPT", hasThought(withPreserve[preToolCallTurn].Content)) && ok
	ok = check("pre-last-user plain turn, preserve_thinking=true:      reasoning still DROPPED (not a tool-call turn)", !hasThought(withPreserve[prePlainTurn].Content)) && ok
	ok = check("post-last-user turn, preserve_thinking=false:          reasoning KEPT regardless", hasThought(withoutPreserve[postToolCallTurn].Content)) && ok
	ok = check("post-last-user turn, preserve_thinking=true:           reasoning KEPT regardless", hasThought(withPreserve[postToolCallTurn].Content)) && ok
	if !ok {
		os.Exit(1)
	}
}

// chat POSTs history to the running handler with the given preserve_thinking setting and returns
// the []inference.Message model.Chat was actually called with — requestMessages' re-framed
// output, captured via the fake model rather than read back off the wire (the handler never
// echoes the rendered prompt in its response).
func chat(serverURL string, model *fakeModel, history []openaicompat.ChatMessage, preserveThinking bool) ([]inference.Message, error) {
	req := openaicompat.ChatCompletionRequest{
		Model:              "test-model",
		Messages:           history,
		ChatTemplateKwargs: &openaicompat.ChatTemplateKwargs{PreserveThinking: &preserveThinking},
	}
	body, err := json.Marshal(req)
	if err != nil {
		return nil, err
	}
	resp, err := http.Post(serverURL+openaicompat.DefaultChatCompletionsPath, "application/json", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("HTTP %d: %s", resp.StatusCode, respBody)
	}
	return model.received, nil
}

// hasThought reports whether content carries a re-framed native thought span — the marker
// requestMessages prefixes onto a kept turn's content.
func hasThought(content string) bool {
	return strings.Contains(content, parser.ChannelOpenMarker+"thought\n")
}

func printMessages(messages []inference.Message) {
	for i, m := range messages {
		fmt.Printf("[%d] role=%s\n%s\n\n", i, m.Role, m.Content)
	}
}

func check(label string, cond bool) bool {
	status := "ok"
	if !cond {
		status = "FAIL"
	}
	fmt.Printf("[%s] %s\n", status, label)
	return cond
}

// fakeModel is a minimal inference.TextModel stand-in — enough to satisfy the interface and drive
// Handler.ServeHTTP end to end with no engine loaded. Chat's captured argument IS this example's
// receipt: the re-framed message list requestMessages built from the wire history, exactly what a
// real model's chat template would render from next.
type fakeModel struct {
	received []inference.Message
}

func (m *fakeModel) Generate(context.Context, string, ...inference.GenerateOption) iter.Seq[inference.Token] {
	return func(func(inference.Token) bool) {}
}

func (m *fakeModel) Chat(_ context.Context, messages []inference.Message, _ ...inference.GenerateOption) iter.Seq[inference.Token] {
	m.received = messages
	return func(yield func(inference.Token) bool) { yield(inference.Token{ID: 1, Text: "ok"}) }
}

func (m *fakeModel) Classify(context.Context, []string, ...inference.GenerateOption) core.Result {
	return core.Ok(nil)
}

func (m *fakeModel) BatchGenerate(context.Context, []string, ...inference.GenerateOption) core.Result {
	return core.Ok(nil)
}

func (m *fakeModel) ModelType() string                  { return "gemma4" }
func (m *fakeModel) Info() inference.ModelInfo          { return inference.ModelInfo{Architecture: "gemma4"} }
func (m *fakeModel) Metrics() inference.GenerateMetrics { return inference.GenerateMetrics{} }
func (m *fakeModel) Err() core.Result                   { return core.Ok(nil) }
func (m *fakeModel) Close() core.Result                 { return core.Ok(nil) }
