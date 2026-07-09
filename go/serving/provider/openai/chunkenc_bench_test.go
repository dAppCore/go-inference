// SPDX-Licence-Identifier: EUPL-1.2

package openai

import (
	"strings"
	"testing"
)

// AX-11 baseline benchmarks for the chunkenc hot path. These encoders
// fire on the streaming serve path — serveStreaming emits one
// ChatCompletionChunk per content/thought delta in the SSE loop plus
// a priming + terminating chunk. Per-token cost matters because every
// adapter consumer (lthn-mlx, openai-compat proxies, the OpenAI-shaped
// MCP bridge) shells through these encoders for every token streamed.
//
// AX-11 RFC § "What counts as a hot path" lists "Per-token scoring"
// at the top of the hot table — these are per-token. No bench
// coverage existed for the private append* helpers before this file.
//
// Run:
//   go test -bench=. -benchmem -benchtime=300ms ./openai/...

var (
	chunkBenchSink []byte
	chunkBenchInt  int
)

// fixtures — sized to match realistic SSE bodies. Most tokens are
// 1-4 chars (BPE tokenisation); the encoder hot loop reflects that
// shape.

func benchPrimingChunk() ChatCompletionChunk {
	return ChatCompletionChunk{
		ID:      "chatcmpl-bench0001",
		Object:  "chat.completion.chunk",
		Created: 1714291200,
		Model:   "qwen3-7b",
		Choices: []ChatChunkChoice{{
			Index: 0,
			Delta: ChatMessageDelta{Role: "assistant"},
		}},
	}
}

func benchDeltaChunk(token string) ChatCompletionChunk {
	return ChatCompletionChunk{
		ID:      "chatcmpl-bench0001",
		Object:  "chat.completion.chunk",
		Created: 1714291200,
		Model:   "qwen3-7b",
		Choices: []ChatChunkChoice{{
			Index: 0,
			Delta: ChatMessageDelta{Content: token},
		}},
	}
}

func benchTerminatingChunk() ChatCompletionChunk {
	stop := "stop"
	return ChatCompletionChunk{
		ID:      "chatcmpl-bench0001",
		Object:  "chat.completion.chunk",
		Created: 1714291200,
		Model:   "qwen3-7b",
		Choices: []ChatChunkChoice{{
			Index:        0,
			FinishReason: &stop,
		}},
	}
}

// --- appendChatCompletionChunk — JSON body only, no SSE framing ---

// Priming chunk — first frame of the stream. Same shape as a delta
// chunk but with a role marker instead of content. Fires once per
// streamed response.
func BenchmarkChunkEnc_AppendChunk_Priming(b *testing.B) {
	chunk := benchPrimingChunk()
	buf := make([]byte, 0, 512)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		chunkBenchSink = appendChatCompletionChunk(buf, chunk)
	}
}

// Per-token delta — the in-loop hot path. Single short token (BPE
// average), one ChatChunkChoice with a 1-byte Content delta. This is
// the bench number that scales with tokens-per-second.
func BenchmarkChunkEnc_AppendChunk_Delta_ShortToken(b *testing.B) {
	chunk := benchDeltaChunk("e")
	buf := make([]byte, 0, 512)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		chunkBenchSink = appendChatCompletionChunk(buf, chunk)
	}
}

// Long-token delta — chunk-shipped multi-word strings (e.g. when
// the streamer batches several tokens or a single token decodes to
// a long word). Catches per-byte string-copy cost differences.
func BenchmarkChunkEnc_AppendChunk_Delta_LongToken(b *testing.B) {
	chunk := benchDeltaChunk("antidisestablishmentarianism")
	buf := make([]byte, 0, 512)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		chunkBenchSink = appendChatCompletionChunk(buf, chunk)
	}
}

// Terminating chunk — last frame of the stream with the FinishReason
// pointer set instead of Delta.Content. Fires once per response.
func BenchmarkChunkEnc_AppendChunk_Terminating(b *testing.B) {
	chunk := benchTerminatingChunk()
	buf := make([]byte, 0, 512)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		chunkBenchSink = appendChatCompletionChunk(buf, chunk)
	}
}

// --- appendChatCompletionChunkSSE — JSON body + SSE framing ---
// The actual function the streaming serve path calls per token.
// Includes `data: ` prefix + `\n\n` suffix.

func BenchmarkChunkEnc_AppendChunkSSE_Delta_ShortToken(b *testing.B) {
	chunk := benchDeltaChunk("e")
	buf := make([]byte, 0, 512)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		chunkBenchSink = appendChatCompletionChunkSSE(buf, chunk)
	}
}

func BenchmarkChunkEnc_AppendChunkSSE_Delta_LongToken(b *testing.B) {
	chunk := benchDeltaChunk(strings.Repeat("token", 8)) // 40 chars
	buf := make([]byte, 0, 512)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		chunkBenchSink = appendChatCompletionChunkSSE(buf, chunk)
	}
}

// --- chunkSSEFrameSize — pre-allocation helper ---
// Used by callers that want to size their buffer before encoding.
// Worth benchmarking because a wrong size estimate forces a grow
// during the encode loop.

func BenchmarkChunkEnc_FrameSize_Delta(b *testing.B) {
	chunk := benchDeltaChunk("e")
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		chunkBenchInt = chunkSSEFrameSize(chunk)
	}
}

// AX-11: zero-alloc budget for the per-token SSE encode path. With
// a pre-sized caller buffer, every Append* function must stay at
// zero allocations — that's the whole point of the caller-bound
// buffer pattern. A regression here would scale per-token, meaning
// a stream of 1000 tokens would suddenly pay 1000× a new alloc.
func TestAllocBudget_ChunkEnc_AppendNoAllocs(t *testing.T) {
	priming := benchPrimingChunk()
	delta := benchDeltaChunk("e")
	terminating := benchTerminatingChunk()
	cases := []struct {
		name string
		fn   func([]byte) []byte
	}{
		{"AppendChunk_Priming", func(buf []byte) []byte {
			return appendChatCompletionChunk(buf, priming)
		}},
		{"AppendChunk_Delta_ShortToken", func(buf []byte) []byte {
			return appendChatCompletionChunk(buf, delta)
		}},
		{"AppendChunk_Terminating", func(buf []byte) []byte {
			return appendChatCompletionChunk(buf, terminating)
		}},
		{"AppendChunkSSE_Delta_ShortToken", func(buf []byte) []byte {
			return appendChatCompletionChunkSSE(buf, delta)
		}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			buf := make([]byte, 0, 1024)
			avg := testing.AllocsPerRun(5, func() {
				chunkBenchSink = tc.fn(buf)
			})
			const budget = 0.0
			if avg > budget {
				t.Fatalf("%s alloc budget exceeded: %.1f allocs/call (budget=%.0f)\n"+
					"This is per-token streaming hot path — every token pays this.\n"+
					"A 1000-token stream pays 1000× this regression.\n"+
					"Profile: go test -bench=BenchmarkChunkEnc -benchmem -memprofile=/tmp/c.mem",
					tc.name, avg, budget)
			}
		})
	}
}

// TestChunkSSEFrameSize_NeverUnderCounts locks the safety property of
// the SSE-frame size estimator: for every realistic chunk shape, the
// estimate must be >= the actual emit length. Any under-count would
// trigger a grow during appendChatCompletionChunkSSE, defeating the
// whole point of pre-sizing the caller buffer.
//
// The estimator was tightened (W12) to drop the int64-worst-case
// reserves on `created` (10 digits → year 2286) and `index` (≤4
// digits → 9999 n-best), pulling the per-frame buffer from the 240/256
// allocator size class down to the 192/208 class. This test guards
// the tightening so a future "just shave one more byte" change can't
// silently underflow.
func TestChunkSSEFrameSize_NeverUnderCounts(t *testing.T) {
	finish := "stop"
	longContent := strings.Repeat("token-", 100)
	longThought := strings.Repeat("reflection-", 50)
	cases := []struct {
		name  string
		chunk ChatCompletionChunk
	}{
		{"priming", benchPrimingChunk()},
		{"delta-short", benchDeltaChunk("e")},
		{"delta-long", benchDeltaChunk(longContent)},
		{"terminating", benchTerminatingChunk()},
		{"finish-with-reason", ChatCompletionChunk{
			ID: "x", Object: "y", Created: 1700000000, Model: "qwen3",
			Choices: []ChatChunkChoice{{Index: 0, FinishReason: &finish}},
		}},
		{"large-index", ChatCompletionChunk{
			ID: "x", Object: "y", Created: 1700000000, Model: "qwen3",
			Choices: []ChatChunkChoice{{Index: 9999, Delta: ChatMessageDelta{Role: "assistant"}}},
		}},
		{"multi-choice", ChatCompletionChunk{
			ID: "x", Object: "y", Created: 1700000000, Model: "qwen3",
			Choices: []ChatChunkChoice{
				{Index: 0, Delta: ChatMessageDelta{Content: "A"}},
				{Index: 1, Delta: ChatMessageDelta{Content: "B"}},
			},
		}},
		{"with-thought", ChatCompletionChunk{
			ID: "x", Object: "y", Created: 1700000000, Model: "qwen3",
			Choices: []ChatChunkChoice{{Index: 0, Delta: ChatMessageDelta{Content: "Hi"}}},
			Thought: &longThought,
		}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			actual := len(appendChatCompletionChunkSSE(nil, tc.chunk))
			est := chunkSSEFrameSize(tc.chunk)
			if est < actual {
				t.Fatalf("chunkSSEFrameSize=%d under-counts actual emit=%d — "+
					"the pre-sized buffer would force a grow on every frame",
					est, actual)
			}
		})
	}
}
