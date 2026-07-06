// SPDX-License-Identifier: EUPL-1.2

package chathistory_test

import (
	"path/filepath"
	"strconv"
	"testing"

	"dappco.re/go/inference/serving/chathistory"
)

// Package-level sinks defeat dead-code elimination so the compiler
// cannot drop the work under benchmark.
var (
	sinkTurns   []chathistory.Turn
	sinkSummary []chathistory.ConversationSummary
	sinkString  string
	sinkInt     int
	sinkErr     error
)

// newBenchHistory opens a fresh archive in a temp dir for benchmarking.
func newBenchHistory(b *testing.B) *chathistory.History {
	b.Helper()
	path := filepath.Join(b.TempDir(), "chats.duckdb")
	h, err := chathistory.Open("snider", path)
	if err != nil {
		b.Fatalf("Open: %v", err)
	}
	b.Cleanup(func() { _ = h.Close() })
	return h
}

// seedConversation creates a conversation with n turns alternating
// user/assistant and returns its id. Content is a realistic multi-turn
// message body, not a one-word stub.
func seedConversation(b *testing.B, h *chathistory.History, n int) string {
	b.Helper()
	id, err := h.StartConversation(chathistory.NewConversation{
		Title:     "evening vent",
		ModelID:   "lemer-lite",
		BaseModel: "gemma-4-e2b-it-4bit",
		Tags:      []string{"life", "vent"},
	})
	if err != nil {
		b.Fatalf("StartConversation: %v", err)
	}
	for i := range n {
		role := "user"
		if i%2 == 1 {
			role = "assistant"
		}
		if _, err := h.WriteTurn(id, chathistory.NewTurn{
			Role:      role,
			Content:   "this is a realistic multi-turn message body of moderate length, turn " + strconv.Itoa(i),
			TokensIn:  16,
			TokensOut: 8,
		}); err != nil {
			b.Fatalf("WriteTurn[%d]: %v", i, err)
		}
	}
	return id
}

// BenchmarkLoadTurns — the per-request context-replay path: load every
// turn of a 20-turn conversation in ordinal order.
func BenchmarkLoadTurns(b *testing.B) {
	h := newBenchHistory(b)
	convID := seedConversation(b, h, 20)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sinkTurns, sinkErr = h.LoadTurns(convID)
	}
}

// BenchmarkWriteTurn — the per-message append path (ordinal lookup +
// insert). Seeds a short history first so the ordinal MAX() scans rows.
func BenchmarkWriteTurn(b *testing.B) {
	h := newBenchHistory(b)
	convID := seedConversation(b, h, 4)
	t := chathistory.NewTurn{
		Role:      "user",
		Content:   "another realistic turn body for the write path",
		TokensIn:  16,
		TokensOut: 8,
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sinkString, sinkErr = h.WriteTurn(convID, t)
	}
}

// BenchmarkStartConversation — opens a new conversation with tags
// (exercises the JSON marshal of the tags slice).
func BenchmarkStartConversation(b *testing.B) {
	h := newBenchHistory(b)
	c := chathistory.NewConversation{
		Title:     "evening vent",
		ModelID:   "lemer-lite",
		BaseModel: "gemma-4-e2b-it-4bit",
		Tags:      []string{"life", "vent"},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sinkString, sinkErr = h.StartConversation(c)
	}
}

// BenchmarkRecentConversations — the "pick up where you left off" list
// path over an archive of 20 conversations, newest 10 returned.
func BenchmarkRecentConversations(b *testing.B) {
	h := newBenchHistory(b)
	for range 20 {
		seedConversation(b, h, 2)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sinkSummary, sinkErr = h.RecentConversations(10)
	}
}

// BenchmarkExportJSONL — the full export path over a small archive of
// several multi-turn conversations, written to a fresh dest each time.
func BenchmarkExportJSONL(b *testing.B) {
	h := newBenchHistory(b)
	for range 5 {
		seedConversation(b, h, 8)
	}
	dest := filepath.Join(b.TempDir(), "export.jsonl")
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sinkErr = h.ExportJSONL(dest)
	}
}

// BenchmarkCountTurns — the lightweight aggregate path.
func BenchmarkCountTurns(b *testing.B) {
	h := newBenchHistory(b)
	seedConversation(b, h, 20)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sinkInt, sinkErr = h.CountTurns()
	}
}
