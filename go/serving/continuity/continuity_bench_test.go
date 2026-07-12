// SPDX-Licence-Identifier: EUPL-1.2

// Benchmarks for the continuity manager's pure per-request helpers. Per AX-11
// — every stateless chat request that reaches the interceptor pays these:
// conversationTurnSplit finds the prefix/tail boundary, conversationKey hashes
// the whole retained prefix into the state-store lookup key (TWICE per turn —
// once to acquire, once to sleep under the grown key), and messagesCarryMedia
// scans every turn to decline media conversations. conversationKey is the
// heavy one: it assembles the full prefix into a builder then SHA-256s it, so
// its allocation profile is what a deep conversation pays on every turn.
//
// Run:    go test -bench=. -benchmem -run='^$' ./serving/continuity/
package continuity

import (
	"strconv"
	"testing"

	"dappco.re/go/inference"
)

// Sinks defeat compiler DCE.
var (
	contBenchSinkString string
	contBenchSinkInt    int
	contBenchSinkBool   bool
)

// benchConversation builds a realistic n-turn transcript (system preamble +
// alternating user/assistant turns), each turn a paragraph-sized body — the
// shape a long chat presents to the interceptor on turn n.
func benchConversation(turns int) []inference.Message {
	const body = "Let me think about that carefully. The trade-off here is between " +
		"latency and throughput, and the honest answer depends on the batch shape " +
		"you are actually serving in production rather than a synthetic micro-bench."
	msgs := make([]inference.Message, 0, turns*2+1)
	msgs = append(msgs, inference.Message{Role: "system", Content: "You are a careful assistant."})
	for i := 0; i < turns; i++ {
		msgs = append(msgs, inference.Message{Role: "user", Content: "Question " + strconv.Itoa(i) + ": " + body})
		msgs = append(msgs, inference.Message{Role: "assistant", Content: "Answer " + strconv.Itoa(i) + ": " + body})
	}
	return msgs
}

// --- conversationKey (the per-request hot path) --------------------------

// Shallow: a 2-turn prefix, the common short-chat case.
func BenchmarkConversationKey_Shallow(b *testing.B) {
	msgs := benchConversation(1)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		contBenchSinkString = conversationKey(msgs, false)
	}
}

// Deep: a 20-turn prefix — the case continuity exists to make cheap, and where
// the builder's growth cost dominates the key computation.
func BenchmarkConversationKey_Deep(b *testing.B) {
	msgs := benchConversation(20)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		contBenchSinkString = conversationKey(msgs, false)
	}
}

// Thinking mode prepends the switch — same growth, verifies the flag path.
func BenchmarkConversationKey_DeepThinking(b *testing.B) {
	msgs := benchConversation(20)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		contBenchSinkString = conversationKey(msgs, true)
	}
}

// --- conversationTurnSplit -----------------------------------------------

func BenchmarkConversationTurnSplit(b *testing.B) {
	msgs := benchConversation(20)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		contBenchSinkInt = conversationTurnSplit(msgs)
	}
}

// --- messagesCarryMedia (scans every turn on every request) --------------

func BenchmarkMessagesCarryMedia(b *testing.B) {
	msgs := benchConversation(20)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		contBenchSinkBool = messagesCarryMedia(msgs)
	}
}

// --- normaliseRole -------------------------------------------------------

func BenchmarkNormaliseRole(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		contBenchSinkString = normaliseRole(" Assistant ")
	}
}
