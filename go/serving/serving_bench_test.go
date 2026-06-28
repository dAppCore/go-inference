// SPDX-Licence-Identifier: EUPL-1.2

package serving

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/inference/provider/openai"
)

// Package-level sinks defeat dead-code elimination so the benchmarked work
// is not optimised away.
var (
	benchResult  core.Result
	benchString  string
	benchInt     int
	benchOpts    []inference.GenerateOption
	benchOAI     []openai.ChatMessage
	benchMetrics *inference.GenerateMetrics
	benchGenOpts GenOpts
	benchResultV Result
)

// --- Realistic request/response fixtures ---

// benchMessages is a typical 3-turn chat conversation.
func benchMessages() []Message {
	return []Message{
		{Role: "system", Content: "You are a concise, helpful assistant."},
		{Role: "user", Content: "Summarise the key points of distributed consensus."},
		{Role: "assistant", Content: "Consensus protocols ensure replicas agree on an ordered log."},
	}
}

// benchTokens is a realistic short model response split into tokens.
func benchTokens() []inference.Token {
	parts := []string{"Distributed", " consensus", " lets", " a", " set", " of",
		" nodes", " agree", " on", " a", " single", " value", " despite", " failures", "."}
	toks := make([]inference.Token, len(parts))
	for i, p := range parts {
		toks[i] = inference.Token{ID: int32(i + 1), Text: p}
	}
	return toks
}

const benchGenerated = "Distributed consensus lets a set of nodes agree on a single " +
	"value despite failures. Protocols such as Raft and Paxos elect a leader and " +
	"replicate an ordered log so every replica converges. <|im_end|>"

// --- DefaultGenOpts ---

func BenchmarkDefaultGenOpts(b *testing.B) {
	b.ReportAllocs()
	for range b.N {
		benchGenOpts = DefaultGenOpts()
	}
}

// --- convertOpts (per Generate/Chat/Stream request) ---

func BenchmarkConvertOpts_Default(b *testing.B) {
	opts := DefaultGenOpts() // Temperature=0.1 → 1 option
	b.ReportAllocs()
	for range b.N {
		benchOpts = convertOpts(opts)
	}
}

func BenchmarkConvertOpts_Typical(b *testing.B) {
	opts := GenOpts{Temperature: 0.7, MaxTokens: 256} // 2 options
	b.ReportAllocs()
	for range b.N {
		benchOpts = convertOpts(opts)
	}
}

func BenchmarkConvertOpts_Full(b *testing.B) {
	opts := GenOpts{
		Temperature:   0.7,
		MaxTokens:     256,
		TopK:          40,
		TopP:          0.9,
		RepeatPenalty: 1.1,
		StopTokens:    []int32{2},
	} // 6 options
	b.ReportAllocs()
	for range b.N {
		benchOpts = convertOpts(opts)
	}
}

func BenchmarkConvertOpts_Empty(b *testing.B) {
	opts := GenOpts{} // 0 options → nil
	b.ReportAllocs()
	for range b.N {
		benchOpts = convertOpts(opts)
	}
}

// --- applyStopSequences (response post-processing) ---

func BenchmarkApplyStopSequences_NoStops(b *testing.B) {
	b.ReportAllocs()
	for range b.N {
		benchString = applyStopSequences(benchGenerated, nil)
	}
}

func BenchmarkApplyStopSequences_Hit(b *testing.B) {
	stops := []string{"<|im_end|>", "\nUser:"}
	b.ReportAllocs()
	for range b.N {
		benchString = applyStopSequences(benchGenerated, stops)
	}
}

func BenchmarkApplyStopSequences_Miss(b *testing.B) {
	stops := []string{"NEVER_PRESENT", "ALSO_ABSENT"}
	b.ReportAllocs()
	for range b.N {
		benchString = applyStopSequences(benchGenerated, stops)
	}
}

// --- indexSubstr ---

func BenchmarkIndexSubstr_Hit(b *testing.B) {
	b.ReportAllocs()
	for range b.N {
		benchInt = indexSubstr(benchGenerated, "<|im_end|>")
	}
}

func BenchmarkIndexSubstr_Miss(b *testing.B) {
	b.ReportAllocs()
	for range b.N {
		benchInt = indexSubstr(benchGenerated, "NEVER_PRESENT")
	}
}

// --- openaiMessages (request building) ---

func BenchmarkOpenaiMessages(b *testing.B) {
	msgs := benchMessages()
	b.ReportAllocs()
	for range b.N {
		benchOAI = openaiMessages(msgs)
	}
}

// --- newResult / metricsPtr ---

func BenchmarkNewResult(b *testing.B) {
	b.ReportAllocs()
	for range b.N {
		benchResultV = newResult("hello world", nil)
	}
}

func BenchmarkMetricsPtr(b *testing.B) {
	m := &mockTextModel{}
	b.ReportAllocs()
	for range b.N {
		benchMetrics = metricsPtr(m)
	}
}

// --- InferenceAdapter (full request path over a mock model) ---

func BenchmarkInferenceAdapter_Generate(b *testing.B) {
	adapter := NewInferenceAdapter(&mockTextModel{tokens: benchTokens()}, "bench")
	ctx := context.Background()
	opts := DefaultGenOpts()
	b.ReportAllocs()
	for range b.N {
		benchResult = adapter.Generate(ctx, "explain consensus", opts)
	}
}

func BenchmarkInferenceAdapter_Chat(b *testing.B) {
	adapter := NewInferenceAdapter(&mockTextModel{tokens: benchTokens()}, "bench")
	ctx := context.Background()
	msgs := benchMessages()
	opts := DefaultGenOpts()
	b.ReportAllocs()
	for range b.N {
		benchResult = adapter.Chat(ctx, msgs, opts)
	}
}

func BenchmarkInferenceAdapter_GenerateStream(b *testing.B) {
	adapter := NewInferenceAdapter(&mockTextModel{tokens: benchTokens()}, "bench")
	ctx := context.Background()
	opts := DefaultGenOpts()
	sink := func(string) error { return nil }
	b.ReportAllocs()
	for range b.N {
		benchResult = adapter.GenerateStream(ctx, "explain consensus", opts, sink)
	}
}

func BenchmarkInferenceAdapter_GenerateStream_StopSeq(b *testing.B) {
	adapter := NewInferenceAdapter(&mockTextModel{tokens: benchTokens()}, "bench")
	ctx := context.Background()
	opts := DefaultGenOpts()
	opts.StopSequences = []string{"<|im_end|>"}
	sink := func(string) error { return nil }
	b.ReportAllocs()
	for range b.N {
		benchResult = adapter.GenerateStream(ctx, "explain consensus", opts, sink)
	}
}

func BenchmarkInferenceAdapter_ChatStream(b *testing.B) {
	adapter := NewInferenceAdapter(&mockTextModel{tokens: benchTokens()}, "bench")
	ctx := context.Background()
	msgs := benchMessages()
	opts := DefaultGenOpts()
	sink := func(string) error { return nil }
	b.ReportAllocs()
	for range b.N {
		benchResult = adapter.ChatStream(ctx, msgs, opts, sink)
	}
}

// --- HTTPBackend (full request path over a loopback httptest server) ---

// benchHTTPServer returns a server that answers chat completions with a fixed
// assistant reply, plus the precomputed response body bytes it writes.
func benchHTTPServer(b *testing.B) *httptest.Server {
	resp := openai.ChatCompletionResponse{
		Choices: []openai.ChatChoice{{
			Message: openai.ChatMessage{Role: "assistant", Content: "Consensus lets nodes agree on an ordered log."},
		}},
	}
	body := core.JSONMarshalString(resp)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.WriteString(w, body)
	}))
	b.Cleanup(srv.Close)
	return srv
}

func BenchmarkHTTPBackend_Chat(b *testing.B) {
	srv := benchHTTPServer(b)
	backend := NewHTTPBackend(srv.URL, "bench-model")
	ctx := context.Background()
	msgs := benchMessages()
	opts := DefaultGenOpts()
	b.ReportAllocs()
	for range b.N {
		benchResult = backend.Chat(ctx, msgs, opts)
	}
}

func BenchmarkHTTPBackend_Generate(b *testing.B) {
	srv := benchHTTPServer(b)
	backend := NewHTTPBackend(srv.URL, "bench-model")
	ctx := context.Background()
	opts := DefaultGenOpts()
	b.ReportAllocs()
	for range b.N {
		benchResult = backend.Generate(ctx, "explain consensus", opts)
	}
}

func BenchmarkHTTPBackend_doRequest(b *testing.B) {
	srv := benchHTTPServer(b)
	backend := NewHTTPBackend(srv.URL, "bench-model")
	ctx := context.Background()
	req := openai.ChatCompletionRequest{
		Model:    "bench-model",
		Messages: openaiMessages(benchMessages()),
	}
	body := []byte(core.JSONMarshalString(req))
	b.ReportAllocs()
	for range b.N {
		benchResult = backend.doRequest(ctx, body)
	}
}

// --- HTTPTextModel ---

func BenchmarkHTTPTextModel_Generate(b *testing.B) {
	srv := benchHTTPServer(b)
	model := NewHTTPTextModel(NewHTTPBackend(srv.URL, "bench-model"))
	ctx := context.Background()
	b.ReportAllocs()
	for range b.N {
		for tok := range model.Generate(ctx, "explain consensus") {
			benchString = tok.Text
		}
	}
}

func BenchmarkHTTPTextModel_Chat(b *testing.B) {
	srv := benchHTTPServer(b)
	model := NewHTTPTextModel(NewHTTPBackend(srv.URL, "bench-model"))
	ctx := context.Background()
	msgs := benchMessages()
	b.ReportAllocs()
	for range b.N {
		for tok := range model.Chat(ctx, msgs) {
			benchString = tok.Text
		}
	}
}

func BenchmarkHTTPTextModel_BatchGenerate(b *testing.B) {
	srv := benchHTTPServer(b)
	model := NewHTTPTextModel(NewHTTPBackend(srv.URL, "bench-model"))
	ctx := context.Background()
	prompts := []string{"explain consensus", "explain replication", "explain quorum"}
	b.ReportAllocs()
	for range b.N {
		benchResult = model.BatchGenerate(ctx, prompts)
	}
}
