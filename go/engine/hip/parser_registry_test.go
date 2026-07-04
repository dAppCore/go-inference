// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import (
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference"
)

func TestParserRegistry_Good_QwenThinkTags(t *testing.T) {
	result, err := NewParserRegistry("qwen3").ParseReasoning(nil, "<think>hidden</think>visible")

	core.RequireNoError(t, err)
	core.AssertEqual(t, "visible", result.VisibleText)
	core.AssertEqual(t, "hidden", result.Reasoning[0].Text)
}

func TestParserRegistry_Good_UsesArchitectureProfileParserID(t *testing.T) {
	registry := NewParserRegistry("Qwen3_5MoeForConditionalGeneration")
	if registry.architecture != "qwen3_6_moe" || registry.parserID != "qwen" {
		t.Fatalf("registry = %+v, want canonical qwen3_6_moe with qwen parser id", registry)
	}
	result, err := registry.ParseReasoning(nil, "<think>hidden</think>visible")

	core.RequireNoError(t, err)
	core.AssertEqual(t, "visible", result.VisibleText)
	core.AssertEqual(t, "hidden", result.Reasoning[0].Text)
}

func TestParserRegistry_Good_GemmaChannels(t *testing.T) {
	result, err := NewParserRegistry("gemma3").ParseReasoning(nil, "<analysis>hidden</analysis>visible")

	core.RequireNoError(t, err)
	core.AssertEqual(t, "visible", result.VisibleText)
	core.AssertEqual(t, "analysis", result.Reasoning[0].Kind)
}

func TestParserRegistry_Good_Gemma4E2BTurnMarkers(t *testing.T) {
	for _, architecture := range []string{"gemma4", "gemma4_text", "Gemma4ForCausalLM"} {
		result, err := NewParserRegistry(architecture).ParseReasoning(nil, "<start_of_turn>analysis\nhidden<end_of_turn>visible")

		core.RequireNoError(t, err)
		core.AssertEqual(t, "visible", result.VisibleText)
		core.AssertEqual(t, "analysis", result.Reasoning[0].Kind)
		core.AssertEqual(t, "hidden", result.Reasoning[0].Text)
	}
}

func TestParserRegistry_Good_DeepSeekR1Thinking(t *testing.T) {
	result, err := NewParserRegistry("DeepSeek-R1").ParseReasoning(nil, "answer <think>chain</think> final")

	core.RequireNoError(t, err)
	core.AssertEqual(t, "answer  final", result.VisibleText)
	core.AssertEqual(t, "chain", result.Reasoning[0].Text)
}

func TestParserRegistry_Good_MiniMaxThinking(t *testing.T) {
	result, err := NewParserRegistry("MiniMax-M2").ParseReasoning(nil, "<think>chain</think>final")

	core.RequireNoError(t, err)
	core.AssertEqual(t, "final", result.VisibleText)
	core.AssertEqual(t, "chain", result.Reasoning[0].Text)
}

func TestParserRegistry_Good_GPTOSSChannels(t *testing.T) {
	result, err := NewParserRegistry("gpt-oss").ParseReasoning(nil, "<|channel>analysis\nplan<|channel>final\nanswer")

	core.RequireNoError(t, err)
	core.AssertEqual(t, "answer", result.VisibleText)
	core.AssertEqual(t, "analysis", result.Reasoning[0].Kind)
	core.AssertEqual(t, "plan", result.Reasoning[0].Text)
}

func TestParserRegistry_Good_KimiAndGLMAnalysisFinal(t *testing.T) {
	for _, architecture := range []string{"Kimi-K2-Instruct", "GLM4ForCausalLM"} {
		result, err := NewParserRegistry(architecture).ParseReasoning(nil, "<think>hidden plan</think>visible answer")

		core.RequireNoError(t, err)
		core.AssertEqual(t, "visible answer", result.VisibleText)
		core.AssertEqual(t, "thinking", result.Reasoning[0].Kind)
		core.AssertEqual(t, "hidden plan", result.Reasoning[0].Text)
	}
}

func TestParserRegistry_Good_JSONToolCalls(t *testing.T) {
	result, err := NewParserRegistry("mistral").ParseTools(nil, `{"tool_calls":[{"id":"call-1","type":"function","function":{"name":"search","arguments":{"q":"rocm"}}}]}`)

	core.RequireNoError(t, err)
	core.AssertEqual(t, "", result.VisibleText)
	core.AssertEqual(t, "search", result.Calls[0].Name)
	core.AssertContains(t, result.Calls[0].ArgumentsJSON, "rocm")
}

func TestParserRegistry_Good_MistralToolCallsArray(t *testing.T) {
	result, err := NewParserRegistry("mistral").ParseTools(nil, `[{"name":"search","arguments":{"q":"rocm"}}]`)

	core.RequireNoError(t, err)
	core.AssertEqual(t, "search", result.Calls[0].Name)
	core.AssertContains(t, result.Calls[0].ArgumentsJSON, "rocm")
}

func TestParserRegistry_Good_MistralToolCallsPrefix(t *testing.T) {
	result, err := NewParserRegistry("mistral").ParseTools(nil, `<tool_calls>[{"name":"lookup","arguments":{"id":7}}]</tool_calls>`)

	core.RequireNoError(t, err)
	core.AssertEqual(t, "lookup", result.Calls[0].Name)
	core.AssertContains(t, result.Calls[0].ArgumentsJSON, "7")
}

func TestParserRegistry_Good_HermesAndGraniteJSONTools(t *testing.T) {
	for _, architecture := range []string{"Nous-Hermes-2", "GraniteForCausalLM"} {
		result, err := NewParserRegistry(architecture).ParseTools(nil, `{"name":"lookup","arguments":{"id":42}}`)

		core.RequireNoError(t, err)
		core.AssertEqual(t, "lookup", result.Calls[0].Name)
		core.AssertContains(t, result.Calls[0].ArgumentsJSON, "42")
	}
}

func TestParserRegistry_Good_GenericXMLToolCall(t *testing.T) {
	result, err := NewParserRegistry("unknown").ParseTools(nil, `<tool_call>{"name":"lookup","arguments":{"a":1}}</tool_call>`)

	core.RequireNoError(t, err)
	core.AssertEqual(t, "lookup", result.Calls[0].Name)
	core.AssertEqual(t, `{"a":1}`, result.Calls[0].ArgumentsJSON)
}

func TestParserRegistry_Bad_UnknownModelLeavesTextVisible(t *testing.T) {
	reasoning, err := NewParserRegistry("unknown").ParseReasoning(nil, "plain text")
	core.RequireNoError(t, err)
	core.AssertEqual(t, "plain text", reasoning.VisibleText)

	tools, err := NewParserRegistry("unknown").ParseTools(nil, "plain text")
	core.RequireNoError(t, err)
	core.AssertEqual(t, "plain text", tools.VisibleText)
	core.AssertEqual(t, 0, len(tools.Calls))
}

func TestParserRegistry_Good_RocmModelImplementsParserContracts(t *testing.T) {
	var _ inference.ReasoningParser = (*rocmModel)(nil)
	var _ inference.ToolParser = (*rocmModel)(nil)

	model := &rocmModel{modelInfo: inference.ModelInfo{Architecture: "qwen3"}}
	result, err := model.ParseReasoning(nil, "<think>x</think>y")

	core.RequireNoError(t, err)
	core.AssertEqual(t, "y", result.VisibleText)
}

func TestParserRegistry_Good_RocmModelUsesModelTypeFallback(t *testing.T) {
	model := &rocmModel{modelType: "qwen3"}

	result, err := model.ParseReasoning(nil, "<think>x</think>y")

	core.RequireNoError(t, err)
	core.AssertEqual(t, "y", result.VisibleText)
	core.AssertEqual(t, "x", result.Reasoning[0].Text)
}

func TestParserRegistry_Bad_RocmModelParseToolsRecordsErrAndSuccessClears_Bad(t *testing.T) {
	model := &rocmModel{modelInfo: inference.ModelInfo{Architecture: "mistral"}}

	_, err := model.ParseTools(nil, `<tool_call>{bad}</tool_call>`)

	core.AssertError(t, err)
	if resultError(model.Err()) == nil {
		t.Fatal("ParseTools failure Err() = nil")
	}
	core.AssertContains(t, resultError(model.Err()).Error(), "parse JSON")

	result, err := model.ParseTools(nil, `<tool_call>{"name":"search","arguments":{"q":"rocm"}}</tool_call>`)

	core.RequireNoError(t, err)
	core.AssertEqual(t, "search", result.Calls[0].Name)
	if resultError(model.Err()) != nil {
		t.Fatalf("ParseTools success Err() = %v, want nil", resultError(model.Err()))
	}
}

func TestParserRegistry_Good_RocmModelParseReasoningClearsStaleErr(t *testing.T) {
	model := &rocmModel{modelInfo: inference.ModelInfo{Architecture: "qwen3"}}
	model.setLastFailure(core.NewError("stale failure"))

	result, err := model.ParseReasoning(nil, "<think>x</think>y")

	core.RequireNoError(t, err)
	core.AssertEqual(t, "y", result.VisibleText)
	if resultError(model.Err()) != nil {
		t.Fatalf("ParseReasoning success Err() = %v, want nil", resultError(model.Err()))
	}
}
