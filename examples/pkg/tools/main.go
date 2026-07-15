// SPDX-Licence-Identifier: EUPL-1.2

// The full function-calling round trip on a Gemma 4 checkpoint: declare a
// tool, let the model decide to call it, run the call locally, and feed the
// result back for a final answer — the "lem can call your Go functions"
// story.
//
//	go run ./pkg/tools -model ~/models/gemma-4-e2b-it-4bit
//
// Gemma 4's native tool syntax is a special-token grammar the decoder
// preserves verbatim (decode/parser/gemma_tools.go); this example renders a
// declaration, parses the model's own call span out of its reply, and wraps
// the tool's answer in the matching <|tool_response> span before asking for
// the final turn.
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"strings"

	"dappco.re/go/inference"
	"dappco.re/go/inference/decode/parser"
	_ "dappco.re/go/inference/examples/internal/engine" // registers the platform engine
)

func main() {
	model := flag.String("model", os.Getenv("LEM_MODEL"), "model snapshot directory (config.json + *.safetensors)")
	flag.Parse()
	if *model == "" {
		fmt.Fprintln(os.Stderr, "set -model (or LEM_MODEL) to a Gemma 4 model snapshot directory")
		os.Exit(2)
	}

	r := inference.LoadModel(*model)
	if !r.OK {
		fmt.Fprintln(os.Stderr, "load:", r.Value)
		os.Exit(1)
	}
	m := r.Value.(inference.TextModel)
	defer m.Close()

	// Gemma 4's grammar is the only native tool syntax this package implements
	// today (Llama-family checkpoints use Meta's JSON/<|python_tag|> grammar
	// instead) — reject early rather than send declarations the checkpoint was
	// never trained to read.
	if !parser.SupportsToolSyntax(m.ModelType()) {
		fmt.Fprintln(os.Stderr, "model", m.ModelType(), "has no native tool syntax in this package")
		os.Exit(2)
	}

	tools := []parser.ToolDecl{{
		Name:        "get_weather",
		Description: "Gets the current weather for a city.",
		Properties: map[string]parser.ToolParam{
			"city": {Type: "string", Description: "the city name"},
		},
		Required: []string{"city"},
	}}
	declarations := parser.RenderGemmaToolDeclarations(tools)

	// Tool declarations ride the leading system turn (the same framing the
	// serve handlers use — see serving/provider/openai/handler.go
	// requestMessages), ahead of the user's question.
	messages := []inference.Message{
		{Role: "system", Content: declarations},
		{Role: "user", Content: "What's the weather like in Paris right now?"},
	}

	reply := chat(m, messages)
	if er := m.Err(); !er.OK {
		fmt.Fprintln(os.Stderr, "generate:", er.Value)
		os.Exit(1)
	}

	calls, visible := parser.ParseGemmaToolCalls(reply)
	if len(calls) == 0 {
		fmt.Println(visible)
		return
	}

	call := calls[0]
	var args struct {
		City string `json:"city"`
	}
	_ = json.Unmarshal([]byte(call.ArgumentsJSON), &args)
	result := weather(args.City)
	fmt.Printf("model called %s(%s) -> %s\n", call.Name, call.ArgumentsJSON, result)

	// Re-render the model's own call into its wire form (RenderGemmaToolCall)
	// and answer it with a role:"tool" message via RenderGemmaToolResponse —
	// the exact convention serving/provider/openai and /anthropic use to reply
	// to a tool_result turn. This example is a stateless client replaying full
	// history, so both turns are appended rather than relying on server-side
	// KV continuity.
	messages = append(messages,
		inference.Message{Role: "assistant", Content: visible + parser.RenderGemmaToolCall(call.Name, call.ArgumentsJSON)},
		inference.Message{Role: "tool", Content: parser.RenderGemmaToolResponse(result)},
	)
	final := chat(m, messages)
	if er := m.Err(); !er.OK {
		fmt.Fprintln(os.Stderr, "generate:", er.Value)
		os.Exit(1)
	}
	fmt.Println(final)
}

// chat runs one turn to completion — greedy, thinking off — and returns the
// collected reply text.
func chat(m inference.TextModel, messages []inference.Message) string {
	off := false
	var reply strings.Builder
	for tok := range m.Chat(context.Background(), messages,
		inference.WithMaxTokens(256),
		inference.WithTemperature(0),
		inference.WithEnableThinking(&off),
	) {
		reply.WriteString(tok.Text)
	}
	return reply.String()
}

// weather is the toy local function the model's tool call resolves to — a
// real integration would hit a forecast API here.
func weather(city string) string {
	switch city {
	case "Paris":
		return "14C, light rain, wind 12 km/h"
	case "Tokyo":
		return "26C, clear skies"
	default:
		return "forecast unavailable for " + city
	}
}
