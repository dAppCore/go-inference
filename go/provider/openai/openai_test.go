// SPDX-Licence-Identifier: EUPL-1.2

package openai

import (
	"context"
	"encoding/json"
	"iter"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	core "dappco.re/go"
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

func (m *stubModel) Classify(context.Context, []string, ...inference.GenerateOption) core.Result {
	return core.Ok([]inference.ClassifyResult(nil))
}

func (m *stubModel) BatchGenerate(context.Context, []string, ...inference.GenerateOption) core.Result {
	return core.Ok([]inference.BatchResult(nil))
}

func (m *stubModel) ModelType() string { return "stub" }

func (m *stubModel) Info() inference.ModelInfo { return inference.ModelInfo{Architecture: "qwen3"} }

func (m *stubModel) Metrics() inference.GenerateMetrics { return m.metrics }

func (m *stubModel) Err() core.Result { return core.ResultOf(nil, m.err) }

func (m *stubModel) Close() core.Result { return core.Ok(nil) }

func (m *stubModel) seq() iter.Seq[inference.Token] {
	return func(yield func(inference.Token) bool) {
		for _, token := range m.tokens {
			if !yield(token) {
				return
			}
		}
	}
}

// TestOpenAI_StopList_UnmarshalJSON_Good_Bad_Ugly drives StopList's own
// json.Unmarshaler implementation directly via encoding/json.Unmarshal.
// The request-decode hot path (ChatCompletionRequest.unmarshalField,
// unmarshal.go) calls jsonenc.ParseJSONStringList directly on the
// "stop" field's raw bytes rather than dispatching through this
// method — see the package note on openai.go's StopList type — so
// this method is otherwise unreachable from DecodeRequest. It remains
// part of StopList's public contract (json.Unmarshaler) for any
// consumer that decodes a StopList value on its own.
func TestOpenAI_StopList_UnmarshalJSON_Good_Bad_Ugly(t *testing.T) {
	var single StopList
	if err := json.Unmarshal([]byte(`"END"`), &single); err != nil {
		t.Fatalf("Unmarshal(string) error = %v", err)
	}
	if len(single) != 1 || single[0] != "END" {
		t.Fatalf("Unmarshal(string) = %#v, want [END]", single)
	}

	var multi StopList
	if err := json.Unmarshal([]byte(`["a","b"]`), &multi); err != nil {
		t.Fatalf("Unmarshal(array) error = %v", err)
	}
	if len(multi) != 2 || multi[0] != "a" || multi[1] != "b" {
		t.Fatalf("Unmarshal(array) = %#v, want [a b]", multi)
	}

	var nulled StopList
	if err := json.Unmarshal([]byte(`null`), &nulled); err != nil {
		t.Fatalf("Unmarshal(null) error = %v", err)
	}
	if nulled != nil {
		t.Fatalf("Unmarshal(null) = %#v, want nil", nulled)
	}

	var bad StopList
	if err := json.Unmarshal([]byte(`42`), &bad); err == nil {
		t.Fatal("Unmarshal(42) returned nil error, want a shape rejection")
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
