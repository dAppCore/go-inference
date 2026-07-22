// SPDX-Licence-Identifier: EUPL-1.2

// Benchmarks for the OpenAI-compatible chat-completions wire primitives.
// Per AX-11 — these surfaces fire on every served chat request:
//   * DecodeRequest + ValidateRequest at request entry
//   * GenerateOptions / NormalizeStopSequences after validation
//   * ChatMessageDelta.MarshalJSON per streamed delta
//   * indexString + firstStopSequenceCut per delta in the SSE loop
//   * TruncateAtStopSequence at end-of-stream
//   * ThinkingExtractor.Process per token (channel + paired-marker scan)
//
// Run:    go test -bench='BenchmarkOpenAI' -benchtime=100ms -benchmem -run='^$' .

package openai

import (
	"strings"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference"
)

// Sinks defeat compiler DCE.
var (
	openAISinkChatRequest  ChatCompletionRequest
	openAISinkChatResponse ChatCompletionResponse
	openAISinkChunk        ChatCompletionChunk
	openAISinkOptions      []inference.GenerateOption
	openAISinkErr          error
	openAISinkStops        []string
	openAISinkString       string
	openAISinkStopList     StopList
	openAISinkInt          int
	openAISinkBool         bool
	openAISinkBytes        []byte
	openAISinkContent      string
	openAISinkThought      string
	openAISinkResult       core.Result
)

// --- Fixture bodies ---

// openAISingleTurnBody mirrors the typical chat-completions request the
// handler decodes at request entry.
const openAISingleTurnBody = `{"model":"qwen3","messages":[{"role":"system","content":"You are a helpful assistant."},{"role":"user","content":"Please summarise the following paragraph for me in one sentence."}],"temperature":0.7,"top_p":0.95,"max_tokens":256,"stream":true,"stop":["<|im_end|>"]}`

// openAIFiveTurnBody is the realistic chat-history shape — 1 system + 4
// user/assistant pairs.
const openAIFiveTurnBody = `{"model":"qwen3","messages":[{"role":"system","content":"You are a helpful assistant."},{"role":"user","content":"What is 2+2?"},{"role":"assistant","content":"4"},{"role":"user","content":"Are you sure?"},{"role":"assistant","content":"Yes."},{"role":"user","content":"Why?"}],"temperature":0.7,"max_tokens":256,"stream":true}`

// openAITwentyTurnBody — long-running session shape, exercises the
// slice-grow path inside the ChatMessage decode loop.
var openAITwentyTurnBody = buildOpenAITurnsBody(20)

func buildOpenAITurnsBody(turns int) string {
	out := core.NewBuilder()
	out.WriteString(`{"model":"qwen3","messages":[`)
	out.WriteString(`{"role":"system","content":"You are a helpful assistant."}`)
	user := `,{"role":"user","content":"How many tokens does this paragraph contain when measured against the GPT-2 tokeniser?"}`
	assistant := `,{"role":"assistant","content":"That depends on the precise tokeniser implementation but is approximately 32."}`
	for i := range turns {
		if i%2 == 0 {
			out.WriteString(user)
		} else {
			out.WriteString(assistant)
		}
	}
	out.WriteString(`],"max_tokens":1024,"stream":true}`)
	return out.String()
}

// buildChatRequest mirrors a decoded ChatCompletionRequest with the
// requested turn count. Used for Marshal benches.
func buildChatRequest(turns int) ChatCompletionRequest {
	temperature := float32(0.7)
	topP := float32(0.95)
	topK := 64
	maxTokens := 256
	req := ChatCompletionRequest{
		Model:       "qwen3",
		Temperature: &temperature,
		TopP:        &topP,
		TopK:        &topK,
		MaxTokens:   &maxTokens,
		Stream:      true,
		Stop:        StopList{"<|im_end|>", "<|eot_id|>"},
	}
	req.Messages = append(req.Messages, ChatMessage{Role: "system", Content: "You are a helpful assistant."})
	for i := range turns {
		if i%2 == 0 {
			req.Messages = append(req.Messages, ChatMessage{Role: "user", Content: "Summarise the paragraph in one sentence."})
		} else {
			req.Messages = append(req.Messages, ChatMessage{Role: "assistant", Content: "The summary captures the key claim."})
		}
	}
	return req
}

// --- DecodeRequest — front-of-handler JSON decode ---

func BenchmarkOpenAI_DecodeRequest_SingleTurn(b *testing.B) {
	body := openAISingleTurnBody
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		openAISinkChatRequest, openAISinkErr = DecodeRequest(strings.NewReader(body))
	}
}

func BenchmarkOpenAI_DecodeRequest_FiveTurn(b *testing.B) {
	body := openAIFiveTurnBody
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		openAISinkChatRequest, openAISinkErr = DecodeRequest(strings.NewReader(body))
	}
}

func BenchmarkOpenAI_DecodeRequest_TwentyTurn(b *testing.B) {
	body := openAITwentyTurnBody
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		openAISinkChatRequest, openAISinkErr = DecodeRequest(strings.NewReader(body))
	}
}

func BenchmarkOpenAI_DecodeRequest_StopAsString(b *testing.B) {
	body := `{"model":"qwen3","messages":[{"role":"user","content":"hi"}],"stop":"END"}`
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		openAISinkChatRequest, openAISinkErr = DecodeRequest(strings.NewReader(body))
	}
}

func BenchmarkOpenAI_DecodeRequest_StopAsArray(b *testing.B) {
	body := `{"model":"qwen3","messages":[{"role":"user","content":"hi"}],"stop":["END","<|eot_id|>","</response>"]}`
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		openAISinkChatRequest, openAISinkErr = DecodeRequest(strings.NewReader(body))
	}
}

// --- StopList.UnmarshalJSON — direct-call bench bypasses the wrapping
// JSON decoder, isolating the variant-parse cost. ---

func BenchmarkOpenAI_StopList_UnmarshalJSON_String(b *testing.B) {
	data := []byte(`"END"`)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		var sl StopList
		openAISinkErr = sl.UnmarshalJSON(data)
		openAISinkStopList = sl
	}
}

func BenchmarkOpenAI_StopList_UnmarshalJSON_Array(b *testing.B) {
	data := []byte(`["<|im_end|>","<|eot_id|>","</response>"]`)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		var sl StopList
		openAISinkErr = sl.UnmarshalJSON(data)
		openAISinkStopList = sl
	}
}

// --- ValidateRequest — request-shape validation after decode ---

func BenchmarkOpenAI_ValidateRequest_SingleTurn(b *testing.B) {
	req := buildChatRequest(1)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		openAISinkErr = ValidateRequest(req)
	}
}

func BenchmarkOpenAI_ValidateRequest_TwentyTurn(b *testing.B) {
	req := buildChatRequest(20)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		openAISinkErr = ValidateRequest(req)
	}
}

// --- GenerateOptions — sampling-field projection ---

func BenchmarkOpenAI_GenerateOptions_AllFieldsSet(b *testing.B) {
	req := buildChatRequest(1)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		openAISinkOptions, openAISinkErr = GenerateOptions(req)
	}
}

func BenchmarkOpenAI_GenerateOptions_DefaultsOnly(b *testing.B) {
	req := ChatCompletionRequest{
		Model:    "qwen3",
		Messages: []ChatMessage{{Role: "user", Content: "hi"}},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		openAISinkOptions, openAISinkErr = GenerateOptions(req)
	}
}

// --- NormalizeStopSequences — per-request stop-sequence projection ---

func BenchmarkOpenAI_NormalizeStopSequences_Empty(b *testing.B) {
	stops := StopList{}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		openAISinkStops, openAISinkErr = NormalizeStopSequences(stops)
	}
}

func BenchmarkOpenAI_NormalizeStopSequences_Typical(b *testing.B) {
	stops := StopList{"<|im_end|>", "<|eot_id|>", "</response>"}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		openAISinkStops, openAISinkErr = NormalizeStopSequences(stops)
	}
}

// --- ChatMessageDelta.MarshalJSON — per-streamed-delta encode ---
// Hits every SSE frame the streaming handler emits.

func BenchmarkOpenAI_ChatMessageDelta_Marshal_ContentOnly(b *testing.B) {
	delta := ChatMessageDelta{Content: "Answer"}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		openAISinkBytes, openAISinkErr = delta.MarshalJSON()
	}
}

func BenchmarkOpenAI_ChatMessageDelta_Marshal_RolePriming(b *testing.B) {
	delta := ChatMessageDelta{Role: "assistant"}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		openAISinkBytes, openAISinkErr = delta.MarshalJSON()
	}
}

func BenchmarkOpenAI_ChatMessageDelta_Marshal_Empty(b *testing.B) {
	delta := ChatMessageDelta{}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		openAISinkBytes, openAISinkErr = delta.MarshalJSON()
	}
}

// TestChatMessageDelta_Marshal_AllocBudget locks the no-escape hot path
// at one allocation per call: the make([]byte, 0, size) for the output
// buffer. A second alloc indicates the size estimate undersized and the
// append-grow ran — happened twice historically because the envelope
// math forgot the leading-comma + closing-quote bytes. Lock the budget
// so future tweaks don't silently regress.
func TestChatMessageDelta_Marshal_AllocBudget(t *testing.T) {
	cases := []struct {
		name  string
		delta ChatMessageDelta
		want  float64
	}{
		{"content-only", ChatMessageDelta{Content: "Answer"}, 1},
		{"role-priming", ChatMessageDelta{Role: "assistant"}, 1},
		{"both", ChatMessageDelta{Role: "assistant", Content: "Yes."}, 1},
		{"empty", ChatMessageDelta{}, 0}, // returns shared emptyDeltaBytes
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			allocs := testing.AllocsPerRun(100, func() {
				openAISinkBytes, openAISinkErr = c.delta.MarshalJSON()
			})
			if allocs != c.want {
				t.Fatalf("%s: expected %.0f allocs/op, got %.2f", c.name, c.want, allocs)
			}
		})
	}
}

// --- ChatCompletionChunk — full SSE frame marshal ---
// What writeChunk runs once per streamed token plus the terminal frame.

func BenchmarkOpenAI_MarshalChatCompletionChunk_Delta(b *testing.B) {
	chunk := ChatCompletionChunk{
		ID:      "chatcmpl-bench",
		Object:  "chat.completion.chunk",
		Created: 1700000000,
		Model:   "qwen3",
		Choices: []ChatChunkChoice{{
			Index: 0,
			Delta: ChatMessageDelta{Content: "Answer"},
		}},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		openAISinkString = core.JSONMarshalString(chunk)
	}
}

func BenchmarkOpenAI_MarshalChatCompletionChunk_Final(b *testing.B) {
	finish := "stop"
	chunk := ChatCompletionChunk{
		ID:      "chatcmpl-bench",
		Object:  "chat.completion.chunk",
		Created: 1700000000,
		Model:   "qwen3",
		Choices: []ChatChunkChoice{{
			Index:        0,
			Delta:        ChatMessageDelta{},
			FinishReason: &finish,
		}},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		openAISinkString = core.JSONMarshalString(chunk)
	}
}

// --- Hand-rolled chunk-as-SSE-frame — the streaming hot path ---
// Fires per token. The single-buffer frame builder replaces the
// JSONMarshalString + Concat + []byte conversion three-allocation
// chain that the streaming handler used pre-W9-D.

func BenchmarkOpenAI_AppendChatCompletionChunkSSE_Priming(b *testing.B) {
	chunk := ChatCompletionChunk{
		ID:      "chatcmpl-bench",
		Object:  "chat.completion.chunk",
		Created: 1700000000,
		Model:   "qwen3",
		Choices: []ChatChunkChoice{{Index: 0, Delta: ChatMessageDelta{Role: "assistant"}}},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		openAISinkBytes = appendChatCompletionChunkSSE(make([]byte, 0, chunkSSEFrameSize(chunk)), chunk)
	}
}

func BenchmarkOpenAI_AppendChatCompletionChunkSSE_Delta(b *testing.B) {
	chunk := ChatCompletionChunk{
		ID:      "chatcmpl-bench",
		Object:  "chat.completion.chunk",
		Created: 1700000000,
		Model:   "qwen3",
		Choices: []ChatChunkChoice{{Index: 0, Delta: ChatMessageDelta{Content: "Answer"}}},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		openAISinkBytes = appendChatCompletionChunkSSE(make([]byte, 0, chunkSSEFrameSize(chunk)), chunk)
	}
}

func BenchmarkOpenAI_AppendChatCompletionChunkSSE_Final(b *testing.B) {
	finish := "stop"
	chunk := ChatCompletionChunk{
		ID:      "chatcmpl-bench",
		Object:  "chat.completion.chunk",
		Created: 1700000000,
		Model:   "qwen3",
		Choices: []ChatChunkChoice{{Index: 0, Delta: ChatMessageDelta{}, FinishReason: &finish}},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		openAISinkBytes = appendChatCompletionChunkSSE(make([]byte, 0, chunkSSEFrameSize(chunk)), chunk)
	}
}

// --- ChatCompletionResponse — non-streaming response marshal ---

// AppendChatCompletionResponse — hand-rolled fast path used by
// writeJSON for the canonical non-streaming response shape.
func BenchmarkOpenAI_AppendChatCompletionResponse_Typical(b *testing.B) {
	resp := ChatCompletionResponse{
		ID:      "chatcmpl-bench",
		Object:  "chat.completion",
		Created: 1700000000,
		Model:   "qwen3",
		Choices: []ChatChoice{{
			Index:        0,
			Message:      ChatMessage{Role: "assistant", Content: "The summary is concise and faithful to the original text."},
			FinishReason: "stop",
		}},
		Usage: ChatUsage{PromptTokens: 200, CompletionTokens: 32, TotalTokens: 232},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		openAISinkBytes = appendChatCompletionResponse(make([]byte, 0, chatCompletionResponseSize(resp)), resp)
	}
}

func BenchmarkOpenAI_MarshalChatCompletionResponse_Typical(b *testing.B) {
	resp := ChatCompletionResponse{
		ID:      "chatcmpl-bench",
		Object:  "chat.completion",
		Created: 1700000000,
		Model:   "qwen3",
		Choices: []ChatChoice{{
			Index:        0,
			Message:      ChatMessage{Role: "assistant", Content: "The summary is concise and faithful to the original text."},
			FinishReason: "stop",
		}},
		Usage: ChatUsage{PromptTokens: 200, CompletionTokens: 32, TotalTokens: 232},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		openAISinkString = core.JSONMarshalString(resp)
	}
}

// --- indexString — primitive substring scan used by stop-sequence cut ---

func BenchmarkOpenAI_IndexString_Miss(b *testing.B) {
	content := strings.Repeat("answer fragment ", 32) // ~512 chars
	needle := "<|im_end|>"
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		openAISinkInt = indexString(content, needle)
	}
}

func BenchmarkOpenAI_IndexString_EarlyHit(b *testing.B) {
	content := "<|im_end|>" + strings.Repeat("answer fragment ", 32)
	needle := "<|im_end|>"
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		openAISinkInt = indexString(content, needle)
	}
}

func BenchmarkOpenAI_IndexString_LateHit(b *testing.B) {
	content := strings.Repeat("answer fragment ", 32) + "<|im_end|>"
	needle := "<|im_end|>"
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		openAISinkInt = indexString(content, needle)
	}
}

// --- firstStopSequenceCut — per-delta scan in the SSE loop ---
// Scales O(content × |stops|) so multi-stop request shapes pay more.

func BenchmarkOpenAI_FirstStopSequenceCut_Miss(b *testing.B) {
	content := strings.Repeat("answer fragment ", 32)
	stops := []string{"<|im_end|>", "<|eot_id|>"}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		openAISinkInt, openAISinkBool = firstStopSequenceCut(content, stops)
	}
}

func BenchmarkOpenAI_FirstStopSequenceCut_LateHit(b *testing.B) {
	content := strings.Repeat("answer fragment ", 32) + "<|im_end|>"
	stops := []string{"<|im_end|>", "<|eot_id|>"}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		openAISinkInt, openAISinkBool = firstStopSequenceCut(content, stops)
	}
}

func BenchmarkOpenAI_FirstStopSequenceCut_EarlyHit(b *testing.B) {
	content := "<|im_end|>" + strings.Repeat("answer fragment ", 32)
	stops := []string{"<|im_end|>", "<|eot_id|>"}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		openAISinkInt, openAISinkBool = firstStopSequenceCut(content, stops)
	}
}

// --- TruncateAtStopSequence — end-of-stream guard ---

func BenchmarkOpenAI_TruncateAtStopSequence_NoMatch(b *testing.B) {
	content := strings.Repeat("answer fragment ", 32)
	stops := []string{"<|im_end|>", "<|eot_id|>"}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		openAISinkString = TruncateAtStopSequence(content, stops)
	}
}

func BenchmarkOpenAI_TruncateAtStopSequence_Match(b *testing.B) {
	content := strings.Repeat("answer fragment ", 32) + "<|im_end|> ignored"
	stops := []string{"<|im_end|>", "<|eot_id|>"}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		openAISinkString = TruncateAtStopSequence(content, stops)
	}
}

// --- ThinkingExtractor — per-token reasoning split ---
// Runs on every token of every chat completion. The marker scans inside
// Process are where the cost sits.

func BenchmarkOpenAI_ThinkingExtractor_Process_PlainTokenShort(b *testing.B) {
	tokens := []inference.Token{{Text: "Answer"}}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		extractor := NewThinkingExtractor()
		openAISinkContent, openAISinkThought = extractor.Process(tokens[0])
	}
}

func BenchmarkOpenAI_ThinkingExtractor_Process_PairedThinkBlock(b *testing.B) {
	tokens := []inference.Token{{Text: "<think>plan</think>Answer"}}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		extractor := NewThinkingExtractor()
		openAISinkContent, openAISinkThought = extractor.Process(tokens[0])
		c, t := extractor.Flush()
		openAISinkContent = c
		openAISinkThought = t
	}
}

func BenchmarkOpenAI_ThinkingExtractor_Process_ChannelMarker(b *testing.B) {
	token := inference.Token{Text: "<|channel>thought hidden<|channel>assistant Answer"}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		extractor := NewThinkingExtractor()
		openAISinkContent, openAISinkThought = extractor.Process(token)
		c, t := extractor.Flush()
		openAISinkContent = c
		openAISinkThought = t
	}
}

// Long delta — 256 chars without any marker substrate, hits the
// hot-path scan-then-emit branch for every streamed token.
func BenchmarkOpenAI_ThinkingExtractor_Process_LongPlainDelta(b *testing.B) {
	token := inference.Token{Text: strings.Repeat("answer fragment ", 16)} // 256 chars
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		extractor := NewThinkingExtractor()
		openAISinkContent, openAISinkThought = extractor.Process(token)
	}
}

// --- requestMessages — wire→internal conversion ---

func BenchmarkOpenAI_RequestMessages_SingleTurn(b *testing.B) {
	messages := []ChatMessage{
		{Role: "system", Content: "You are a helpful assistant."},
		{Role: "user", Content: "Summarise the paragraph."},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = requestMessages(messages, nil, false)
	}
}

func BenchmarkOpenAI_RequestMessages_TwentyTurn(b *testing.B) {
	req := buildChatRequest(20)
	messages := req.Messages
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = requestMessages(messages, nil, false)
	}
}

// --- completionID — request-level ID generator ---

func BenchmarkOpenAI_CompletionID(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		openAISinkString = completionID()
	}
}

// AX-11: alloc budget for ThinkingExtractor.Process on a plain non-
// marker token — the streaming hot path. Every model that doesn't
// emit reasoning markers hits this path on every token. The drain
// builder pair is lazy-allocated so the no-thought channel doesn't
// pay; a regression here scales per token (a thousand-token stream
// pays 1000x).
func TestAllocBudget_OpenAI_ThinkingExtractor_PlainToken(t *testing.T) {
	tokens := []inference.Token{{Text: "Answer"}}
	avg := testing.AllocsPerRun(5, func() {
		extractor := NewThinkingExtractor()
		openAISinkContent, openAISinkThought = extractor.Process(tokens[0])
	})
	// Floor: 1 alloc for &ThinkingExtractor{} + 1 for the lazy
	// contentDelta builder (allocated only when first written). The
	// no-thought channel adds zero — saves per-token bytes on plain
	// streams.
	const budget = 2.0
	if avg > budget {
		t.Fatalf("ThinkingExtractor.Process plain-token alloc budget exceeded: %.1f allocs/call (budget=%.0f)\n"+
			"This is the per-token streaming hot path. A regression here scales\n"+
			"per token — a 1000-token stream pays 1000x.\n"+
			"Profile: go test -bench=BenchmarkOpenAI_ThinkingExtractor_Process_PlainTokenShort -benchmem -memprofile=/tmp/te.mem",
			avg, budget)
	}
}
