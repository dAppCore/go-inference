// SPDX-Licence-Identifier: EUPL-1.2

// Benchmarks for the Anthropic Messages streaming event encoders. Per AX-11 —
// AppendContentBlockDeltaEvent fires ONCE PER TOKEN on every streamed Messages
// completion (the content_block_delta payload), so its per-token allocation
// profile is paid on every token of every Claude-compatible answer; the buffer
// is reused across tokens by contract, so the target is zero allocations per
// token. The message_start / message_delta wrappers fire once per stream.
//
// Run:    go test -bench='Anthropic.*Event' -benchmem -run='^$' .
package anthropic

import "testing"

// Sink defeats compiler DCE.
var anthropicStreamBenchSinkBuf []byte

const anthropicBenchDelta = "the quick brown fox "

// --- AppendContentBlockDeltaEvent (per-token hot path) -------------------

func BenchmarkAnthropic_AppendContentBlockDeltaEvent(b *testing.B) {
	buf := make([]byte, 0, 128)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		buf = AppendContentBlockDeltaEvent(buf[:0], 0, anthropicBenchDelta)
	}
	anthropicStreamBenchSinkBuf = buf
}

// Escaped delta: a code/markup token exercising the JSON escape path.
func BenchmarkAnthropic_AppendContentBlockDeltaEvent_Escaped(b *testing.B) {
	buf := make([]byte, 0, 128)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		buf = AppendContentBlockDeltaEvent(buf[:0], 0, `if a < b && "x" > y`)
	}
	anthropicStreamBenchSinkBuf = buf
}

// --- AppendInputJSONDeltaEvent (per-tool-args-chunk) ---------------------

func BenchmarkAnthropic_AppendInputJSONDeltaEvent(b *testing.B) {
	buf := make([]byte, 0, 128)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		buf = AppendInputJSONDeltaEvent(buf[:0], 0, `{"query":"go allocations","limit":5}`)
	}
	anthropicStreamBenchSinkBuf = buf
}

// --- Per-stream wrapper events -------------------------------------------

func BenchmarkAnthropic_AppendMessageStartEvent(b *testing.B) {
	msg := MessageResponse{
		ID: "msg_bench", Type: "message", Role: "assistant", Model: "gemma-4-31b",
		Usage: Usage{InputTokens: 16},
	}
	buf := make([]byte, 0, 256)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		buf = AppendMessageStartEvent(buf[:0], msg)
	}
	anthropicStreamBenchSinkBuf = buf
}

func BenchmarkAnthropic_AppendMessageDeltaEvent(b *testing.B) {
	buf := make([]byte, 0, 128)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		buf = AppendMessageDeltaEvent(buf[:0], "end_turn", "", 48)
	}
	anthropicStreamBenchSinkBuf = buf
}
