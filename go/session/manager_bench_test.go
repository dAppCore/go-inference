// SPDX-Licence-Identifier: EUPL-1.2

package session_test

import (
	core "dappco.re/go"
	chat "dappco.re/go/inference/chat"
	session "dappco.re/go/inference/session"
)

// Allocation benchmarks for the conversation registry (RFC §6.10). The session
// package runs per request/turn — Append on every turn, Continue on every
// continued request — so its per-call heap churn is on the hot serving path.
// Each benchmark calls b.ReportAllocs() and parks its result in a package-level
// sink so the optimiser can't elide the work being measured.
//
// The id generator is injected (pooledGen) to keep the registry maps bounded and
// to keep crypto/rand (defaultIDGen → core.RandomString, which lives in the core
// module, not this package) out of the numbers, isolating the allocations this
// package can actually act on.

// Package-level sinks defeat dead-code elimination of benchmark results.
var (
	benchSessionSink session.Session
	benchRespSink    string
	benchErrSink     error
)

// benchTurnCount is the multi-turn fixture depth — a short but realistic
// conversation (alternating user/assistant), enough that the per-turn clone
// copies a meaningful slice rather than one element.
const benchTurnCount = 8

// pooledGen returns a deterministic, allocation-free id generator: it hands back
// pre-built ids from a fixed pool, wrapping at n. Wrapping keeps the responses /
// sessions maps bounded (~n entries) so the benchmark measures steady-state
// per-call cost, not unbounded map growth, and never pays a per-call string
// allocation the way a counter+Itoa generator would.
func pooledGen(n int) func() string {
	pool := make([]string, n)
	for i := range pool {
		pool[i] = "bench-id-" + core.Itoa(i)
	}
	i := 0
	return func() string {
		id := pool[i%n]
		i++
		return id
	}
}

// benchTurns builds n alternating user/assistant single-text turns — the
// canonical multi-turn fixture the registry stores and copies.
func benchTurns(n int) []chat.Message {
	turns := make([]chat.Message, n)
	for i := range turns {
		role, label := chat.User, "user asks a moderately detailed question number "
		if i%2 == 1 {
			role, label = chat.Assistant, "assistant gives a moderately detailed answer number "
		}
		turns[i] = chat.Message{Role: role, Content: []chat.ContentBlock{chat.Text(label + core.Itoa(i))}}
	}
	return turns
}

// buildBenchSession is a stored-session fixture with n turns already present, so
// a benchmark can measure an operation against an established conversation.
func buildBenchSession(id, model string, n int) session.Session {
	now := core.Now()
	return session.Session{ID: id, Model: model, Created: now, Updated: now, Turns: benchTurns(n)}
}

// BenchmarkManager_Open measures opening a fresh session: mint an id, store it,
// hand back a copy. A fresh session has no turns, so the clones are no-ops.
func BenchmarkManager_Open(b *core.B) {
	m := session.NewManager(session.NewMemoryStore(), session.WithIDGen(pooledGen(4096)))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSessionSink = m.Open("lemma")
	}
}

// BenchmarkManager_Append measures adding one turn to an established
// benchTurnCount-turn session. The stored session is reset to the fixture each
// iteration (untimed) so every measured Append runs against the same depth
// rather than an ever-growing transcript.
func BenchmarkManager_Append(b *core.B) {
	store := session.NewMemoryStore()
	m := session.NewManager(store, session.WithIDGen(pooledGen(4096)))
	base := buildBenchSession("bench-append", "lemma", benchTurnCount)
	turn := chat.Message{Role: chat.User, Content: []chat.ContentBlock{chat.Text("one more turn")}}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		b.StopTimer()
		_ = store.Put(base) // reset to benchTurnCount turns; Put clones, not timed
		b.StartTimer()
		benchRespSink, benchErrSink = m.Append(base.ID, turn)
	}
}

// BenchmarkManager_Continue measures resolving a previous_response_id back to its
// session + context (0% replay) — the hot continued-request path. Steady state:
// the session sits at benchTurnCount turns and the same response id is resolved
// each iteration.
func BenchmarkManager_Continue(b *core.B) {
	m := session.NewManager(session.NewMemoryStore(), session.WithIDGen(pooledGen(4096)))
	s := m.Open("lemma")
	var resp string
	for _, t := range benchTurns(benchTurnCount) {
		resp, _ = m.Append(s.ID, t)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSessionSink, benchErrSink = m.Continue(resp)
	}
}

// BenchmarkManager_Get measures fetching the current session (a copy) for a known
// id at benchTurnCount turns.
func BenchmarkManager_Get(b *core.B) {
	m := session.NewManager(session.NewMemoryStore(), session.WithIDGen(pooledGen(4096)))
	s := m.Open("lemma")
	for _, t := range benchTurns(benchTurnCount) {
		_, _ = m.Append(s.ID, t)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSessionSink, benchErrSink = m.Get(s.ID)
	}
}

// BenchmarkManager_SetStateHandle measures attaching the opaque go-mlx KV handle:
// a read-modify-write through the Store at benchTurnCount turns.
func BenchmarkManager_SetStateHandle(b *core.B) {
	m := session.NewManager(session.NewMemoryStore(), session.WithIDGen(pooledGen(4096)))
	s := m.Open("lemma")
	for _, t := range benchTurns(benchTurnCount) {
		_, _ = m.Append(s.ID, t)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchErrSink = m.SetStateHandle(s.ID, "mlx-kv://node-a/slab/42")
	}
}

// BenchmarkManager_Delete measures removing a session and forgetting the response
// ids that point at it. The session is re-created each iteration (untimed) so
// every measured Delete actually removes a present session.
func BenchmarkManager_Delete(b *core.B) {
	store := session.NewMemoryStore()
	m := session.NewManager(store, session.WithIDGen(pooledGen(4096)))
	base := buildBenchSession("bench-delete", "lemma", benchTurnCount)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		b.StopTimer()
		_ = store.Put(base)
		b.StartTimer()
		benchErrSink = m.Delete(base.ID)
	}
}

// BenchmarkMemoryStore_Get measures the backing store's read — the mandatory
// isolation clone of a benchTurnCount-turn session.
func BenchmarkMemoryStore_Get(b *core.B) {
	store := session.NewMemoryStore()
	_ = store.Put(buildBenchSession("bench-store", "lemma", benchTurnCount))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSessionSink, benchErrSink = store.Get("bench-store")
	}
}

// BenchmarkMemoryStore_Put measures the backing store's write — the mandatory
// isolation clone on the way in.
func BenchmarkMemoryStore_Put(b *core.B) {
	store := session.NewMemoryStore()
	base := buildBenchSession("bench-store", "lemma", benchTurnCount)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchErrSink = store.Put(base)
	}
}

// BenchmarkMemoryStore_Delete measures the backing store's delete (a map delete;
// the session is re-created untimed each iteration).
func BenchmarkMemoryStore_Delete(b *core.B) {
	store := session.NewMemoryStore()
	base := buildBenchSession("bench-store", "lemma", benchTurnCount)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		b.StopTimer()
		_ = store.Put(base)
		b.StartTimer()
		benchErrSink = store.Delete("bench-store")
	}
}

// BenchmarkManager_Conversation is the end-to-end load-path shape: open a
// session, append a full benchTurnCount-turn exchange, then continue from the
// last response. One iteration = one realistic multi-turn request lifecycle.
func BenchmarkManager_Conversation(b *core.B) {
	m := session.NewManager(session.NewMemoryStore(), session.WithIDGen(pooledGen(512)))
	turns := benchTurns(benchTurnCount)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		s := m.Open("lemma")
		var resp string
		for _, t := range turns {
			resp, _ = m.Append(s.ID, t)
		}
		benchSessionSink, benchErrSink = m.Continue(resp)
	}
}
