// SPDX-Licence-Identifier: EUPL-1.2

package openai

import (
	"context"
	"iter"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"dappco.re/go/inference"
)

type stubModel struct {
	tokens  []inference.Token
	metrics inference.GenerateMetrics
	err     error
}

func (m *stubModel) Generate(context.Context, string, ...inference.GenerateOption) iter.Seq[inference.Token] {
	return m.seq()
}

func (m *stubModel) Chat(context.Context, []inference.Message, ...inference.GenerateOption) iter.Seq[inference.Token] {
	return m.seq()
}

func (m *stubModel) Classify(context.Context, []string, ...inference.GenerateOption) ([]inference.ClassifyResult, error) {
	return nil, nil
}

func (m *stubModel) BatchGenerate(context.Context, []string, ...inference.GenerateOption) ([]inference.BatchResult, error) {
	return nil, nil
}

func (m *stubModel) ModelType() string { return "stub" }

func (m *stubModel) Info() inference.ModelInfo { return inference.ModelInfo{Architecture: "qwen3"} }

func (m *stubModel) Metrics() inference.GenerateMetrics { return m.metrics }

func (m *stubModel) Err() error { return m.err }

func (m *stubModel) Close() error { return nil }

func (m *stubModel) seq() iter.Seq[inference.Token] {
	return func(yield func(inference.Token) bool) {
		for _, token := range m.tokens {
			if !yield(token) {
				return
			}
		}
	}
}

func TestOpenAI_DecodeRequest_Good_StopStringAndDefaults(t *testing.T) {
	body := strings.NewReader(`{"model":"qwen","messages":[{"role":"user","content":"hi"}],"stop":"END"}`)

	req, err := DecodeRequest(body)
	if err != nil {
		t.Fatalf("DecodeRequest() error = %v", err)
	}
	if req.Model != "qwen" || len(req.Messages) != 1 {
		t.Fatalf("DecodeRequest() = %+v", req)
	}
	stops, err := NormalizeStopSequences(req.Stop)
	if err != nil {
		t.Fatalf("NormalizeStopSequences() error = %v", err)
	}
	if len(stops) != 1 || stops[0] != "END" {
		t.Fatalf("stops = %#v, want END", stops)
	}

	opts, err := GenerateOptions(req)
	if err != nil {
		t.Fatalf("GenerateOptions() error = %v", err)
	}
	cfg := inference.ApplyGenerateOpts(opts)
	if cfg.Temperature != DefaultTemperature || cfg.TopP != DefaultTopP || cfg.TopK != DefaultTopK || cfg.MaxTokens != DefaultMaxTokens {
		t.Fatalf("defaults = %+v", cfg)
	}
}

func TestOpenAI_GenerateOptions_Good_HonoursExplicitZero(t *testing.T) {
	zeroFloat := float32(0)
	zeroInt := 0
	req := ChatCompletionRequest{
		Model:       "qwen",
		Messages:    []ChatMessage{{Role: "user", Content: "hi"}},
		Temperature: &zeroFloat,
		TopP:        &zeroFloat,
		TopK:        &zeroInt,
		MaxTokens:   &zeroInt,
	}

	opts, err := GenerateOptions(req)
	if err != nil {
		t.Fatalf("GenerateOptions() error = %v", err)
	}
	cfg := inference.ApplyGenerateOpts(opts)
	if cfg.Temperature != 0 || cfg.TopP != 0 || cfg.TopK != 0 || cfg.MaxTokens != 0 {
		t.Fatalf("explicit zero options = %+v", cfg)
	}
}

func TestOpenAI_ThinkingExtractor_Good_CapturesQwenAndChannelMarkers(t *testing.T) {
	extractor := NewThinkingExtractor()

	visible, thought := extractor.Process(inference.Token{Text: "A <thi"})
	visible2, thought2 := extractor.Process(inference.Token{Text: "nk>hidden</think> B <|channel>thought plan"})
	visible3, thought3 := extractor.Process(inference.Token{Text: "<|channel>assistant C"})
	visible4, thought4 := extractor.Flush()

	gotVisible := visible + visible2 + visible3 + visible4
	gotThought := thought + thought2 + thought3 + thought4
	if gotVisible != "A  B  C" {
		t.Fatalf("visible = %q", gotVisible)
	}
	if gotThought != "hidden plan" {
		t.Fatalf("thought = %q", gotThought)
	}
	if extractor.Content() != gotVisible || extractor.Thinking() != gotThought {
		t.Fatalf("extractor content/thought = %q/%q", extractor.Content(), extractor.Thinking())
	}
}

func TestOpenAI_ThinkingExtractor_Ugly_IncompleteChannelMarkerDoesNotHang(t *testing.T) {
	extractor := NewThinkingExtractor()
	done := make(chan struct{})
	go func() {
		extractor.Process(inference.Token{Text: "<|channel>"})
		close(done)
	}()

	select {
	case <-done:
	case <-time.After(100 * time.Millisecond):
		t.Fatal("Process() hung on incomplete channel marker")
	}
	visible, thought := extractor.Flush()
	if visible != "<|channel>" || thought != "" {
		t.Fatalf("Flush() = %q/%q", visible, thought)
	}
}

func TestOpenAI_ThinkingExtractor_Good_Gemma4ChannelCloseSwitchesToContent(t *testing.T) {
	// Gemma4 terminates its reasoning with the <channel|> CLOSE marker —
	// distinct from gpt-oss's <|channel> OPEN. Everything after the close
	// is the visible answer and must reach content, not be swallowed as
	// thinking (which left chat-completions content empty). go-mlx #48.
	extractor := NewThinkingExtractor()

	visible, thought := extractor.Process(inference.Token{Text: "<|channel>thought\nadd two and two<channel|>4"})
	visible2, thought2 := extractor.Flush()

	gotVisible := visible + visible2
	gotThought := thought + thought2
	if gotVisible != "4" {
		t.Fatalf("visible = %q, want %q", gotVisible, "4")
	}
	if gotThought != "\nadd two and two" {
		t.Fatalf("thought = %q, want %q", gotThought, "\nadd two and two")
	}
	if extractor.Content() != "4" {
		t.Fatalf("Content() = %q, want %q", extractor.Content(), "4")
	}
}

func TestOpenAI_ThinkingExtractor_Ugly_Gemma4ChannelCloseSplitAcrossTokens(t *testing.T) {
	// The <channel|> close can straddle a streaming token boundary. The
	// safe-suffix split must hold a partial close marker back so it is
	// recognised once complete, not mis-emitted as thinking. go-mlx #48.
	extractor := NewThinkingExtractor()

	v1, th1 := extractor.Process(inference.Token{Text: "<|channel>thought\nadd<chan"})
	v2, th2 := extractor.Process(inference.Token{Text: "nel|>4"})
	v3, th3 := extractor.Flush()

	gotVisible := v1 + v2 + v3
	gotThought := th1 + th2 + th3
	if gotVisible != "4" {
		t.Fatalf("visible = %q, want %q", gotVisible, "4")
	}
	if gotThought != "\nadd" {
		t.Fatalf("thought = %q, want %q", gotThought, "\nadd")
	}
}

func TestOpenAI_StaticResolver_Good_CaseInsensitiveModelLookup(t *testing.T) {
	model := &stubModel{}
	resolver := NewStaticResolver(map[string]inference.TextModel{"Qwen3": model})

	got, err := resolver.ResolveModel(context.Background(), "qwen3")
	if err != nil {
		t.Fatalf("ResolveModel() error = %v", err)
	}
	if got != model {
		t.Fatalf("ResolveModel() = %p, want %p", got, model)
	}
}

func TestOpenAI_Handler_Good_NonStreamingResponseIncludesThoughtAndUsage(t *testing.T) {
	model := &stubModel{
		tokens: []inference.Token{
			{Text: "<think>plan</think>Answer END ignored"},
		},
		metrics: inference.GenerateMetrics{PromptTokens: 3, GeneratedTokens: 4},
	}
	handler := NewHandler(NewStaticResolver(map[string]inference.TextModel{"qwen": model}))
	body := strings.NewReader(`{"model":"qwen","messages":[{"role":"user","content":"hi"}],"stop":["END"]}`)
	req := httptest.NewRequest(http.MethodPost, DefaultChatCompletionsPath, body)
	rec := httptest.NewRecorder()

	handler.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s", rec.Code, rec.Body.String())
	}
	if !strings.Contains(rec.Body.String(), `"content":"Answer "`) {
		t.Fatalf("response missing visible content: %s", rec.Body.String())
	}
	if !strings.Contains(rec.Body.String(), `"thought":"plan"`) {
		t.Fatalf("response missing thought: %s", rec.Body.String())
	}
	if !strings.Contains(rec.Body.String(), `"total_tokens":7`) {
		t.Fatalf("response missing usage: %s", rec.Body.String())
	}
}

func TestOpenAI_Handler_Good_StreamingResponseEmitsSSEChunks(t *testing.T) {
	model := &stubModel{tokens: []inference.Token{{Text: "Hel"}, {Text: "lo"}}}
	handler := NewHandler(NewStaticResolver(map[string]inference.TextModel{"qwen": model}))
	body := strings.NewReader(`{"model":"qwen","messages":[{"role":"user","content":"hi"}],"stream":true}`)
	req := httptest.NewRequest(http.MethodPost, DefaultChatCompletionsPath, body)
	rec := httptest.NewRecorder()

	handler.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s", rec.Code, rec.Body.String())
	}
	if got := rec.Header().Get("Content-Type"); !strings.Contains(got, "text/event-stream") {
		t.Fatalf("content-type = %q", got)
	}
	bodyText := rec.Body.String()
	if !strings.Contains(bodyText, `"role":"assistant","content":""`) {
		t.Fatalf("stream missing priming chunk: %s", bodyText)
	}
	if !strings.Contains(bodyText, `"content":"Hel"`) || !strings.Contains(bodyText, `"content":"lo"`) {
		t.Fatalf("stream missing content deltas: %s", bodyText)
	}
	if !strings.Contains(bodyText, "data: [DONE]") {
		t.Fatalf("stream missing DONE: %s", bodyText)
	}
}
