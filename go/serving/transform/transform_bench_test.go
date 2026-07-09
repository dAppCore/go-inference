// SPDX-Licence-Identifier: EUPL-1.2

// Allocation benchmarks for the middle-out transform (RFC.md §6.11, §6.13).
// MiddleOut runs per request (the budget layer calls it whenever a prompt
// overflows the chosen endpoint's window), so its steady-state allocation
// profile is on the hot path: every over-window request builds a fresh
// compressed conversation, and the shrink loop rebuilds a candidate per
// iteration. One benchmark per load-path shape — the fitting no-op, a modest
// compress, a deep shrink, the irreducible cannot-fit, and a content-sensitive
// counter — with a realistic multi-turn conversation and a zero-alloc counter
// so the profile attributes every alloc to the transform itself.
//
// Run: go test -bench=. -benchmem -run='^$' ./transform/
package transform_test

import (
	core "dappco.re/go"
	chat "dappco.re/go/inference/serving/chat"
	"dappco.re/go/inference/serving/transform"
)

// Package-level sinks defeat dead-code elimination: each benchmark writes its
// result to a sink of the matching type so the optimiser cannot drop the call.
var (
	sinkMsgs []chat.Message
	sinkBool bool
	sinkErr  error
)

// benchPerMsg sizes a conversation at a fixed cost per message regardless of
// content — zero-alloc, so the timed path's allocations are the transform's own.
type benchPerMsg int

func (c benchPerMsg) Count(messages []chat.Message) int { return len(messages) * int(c) }

// benchLenCtr sizes a conversation by the summed length of its content, the
// other counter shape the spec calls out. Single-text-block messages make
// Message.Text a zero-alloc fast path, so this counter is zero-alloc too.
type benchLenCtr struct{}

func (benchLenCtr) Count(messages []chat.Message) int {
	total := 0
	for _, m := range messages {
		total += len(m.Text())
	}
	return total
}

// benchConvo builds a realistic conversation: one system preamble followed by
// pairs user/assistant turns carrying sentence-length content, the shape a real
// chat thread takes by the time it overflows a 16 GB-GPU endpoint's window.
func benchConvo(pairs int) []chat.Message {
	out := make([]chat.Message, 0, 1+2*pairs)
	out = append(out, msg(chat.System, "You are a helpful assistant. Answer concisely and cite the relevant subsystem when you can."))
	for i := range pairs {
		out = append(out, msg(chat.User, "Question "+core.Itoa(i)+": how does the router decide which endpoint should serve this particular request?"))
		out = append(out, msg(chat.Assistant, "Answer "+core.Itoa(i)+": it scores each endpoint by free context window and current load, then places the request on the best fit."))
	}
	return out
}

func msg(role chat.Role, content string) chat.Message {
	return chat.Message{Role: role, Content: []chat.ContentBlock{chat.Text(content)}}
}

// BenchmarkMiddleOut_Fits — the common case: the conversation already sits inside
// the window, so MiddleOut counts once and hands the input straight back. Should
// be the zero-allocation floor of the package.
func BenchmarkMiddleOut_Fits(b *core.B) {
	convo := benchConvo(15)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sinkMsgs, sinkBool, sinkErr = transform.MiddleOut(convo, benchPerMsg(10), 1000)
	}
}

// BenchmarkMiddleOut_Compress — a modestly over-window conversation: the shrink
// loop drops a handful of middle turns before the tail fits. The typical
// transform.
func BenchmarkMiddleOut_Compress(b *core.B) {
	convo := benchConvo(15) // 31 messages, 310 tokens at 10/msg
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sinkMsgs, sinkBool, sinkErr = transform.MiddleOut(convo, benchPerMsg(10), 280)
	}
}

// BenchmarkMiddleOut_Shrink — a deeply over-window conversation: the loop
// rebuilds and re-measures many candidates before one fits, amplifying the
// per-iteration allocation of the placeholder.
func BenchmarkMiddleOut_Shrink(b *core.B) {
	convo := benchConvo(15)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sinkMsgs, sinkBool, sinkErr = transform.MiddleOut(convo, benchPerMsg(10), 120)
	}
}

// BenchmarkMiddleOut_CannotFit — irreducible: even head + placeholder + one tail
// turn overflows, so the loop exhausts and the best-effort set is returned with
// ErrCannotFit (the worst-case candidate-build count).
func BenchmarkMiddleOut_CannotFit(b *core.B) {
	convo := benchConvo(15)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sinkMsgs, sinkBool, sinkErr = transform.MiddleOut(convo, benchPerMsg(10), 25)
	}
}

// BenchmarkMiddleOut_ContentSensitive — the length-based counter shape: the
// placeholder's own text now counts toward the window, exercising the transform
// under a content-sensitive measure rather than a fixed per-message cost.
func BenchmarkMiddleOut_ContentSensitive(b *core.B) {
	convo := benchConvo(15)
	window := benchLenCtr{}.Count(convo) / 2
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sinkMsgs, sinkBool, sinkErr = transform.MiddleOut(convo, benchLenCtr{}, window)
	}
}
