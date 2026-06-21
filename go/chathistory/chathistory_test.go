// SPDX-License-Identifier: EUPL-1.2

package chathistory

import (
	"path/filepath"
	"testing"
)

// TestRoundtrip — open a fresh archive, write a 4-turn conversation,
// verify counts + export to .duckdb + JSONL.
func TestRoundtrip(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "chats.duckdb")

	h, err := Open("snider", path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer h.Close()

	convID, err := h.StartConversation(NewConversation{
		Title:     "evening vent",
		ModelID:   "lemer-lite",
		BaseModel: "gemma-4-e2b-it-4bit",
		Tags:      []string{"life", "vent"},
	})
	if err != nil {
		t.Fatalf("StartConversation: %v", err)
	}
	if convID == "" {
		t.Fatal("StartConversation returned empty id")
	}

	turns := []NewTurn{
		{Role: "user", Content: "hey lemma"},
		{Role: "assistant", Content: "hey, what's up?", TokensIn: 8, TokensOut: 6},
		{Role: "user", Content: "rough day"},
		{Role: "assistant", Content: "tell me about it", TokensIn: 16, TokensOut: 4},
	}
	turnIDs := make([]string, len(turns))
	for i, t0 := range turns {
		id, err := h.WriteTurn(convID, t0)
		if err != nil {
			t.Fatalf("WriteTurn[%d]: %v", i, err)
		}
		turnIDs[i] = id
	}

	if err := h.SetSignal(turnIDs[1], "liked"); err != nil {
		t.Fatalf("SetSignal: %v", err)
	}
	if err := h.EndConversation(convID); err != nil {
		t.Fatalf("EndConversation: %v", err)
	}

	if n, err := h.CountConversations(); err != nil || n != 1 {
		t.Fatalf("CountConversations: got (%d, %v) want (1, nil)", n, err)
	}
	if n, err := h.CountTurns(); err != nil || n != 4 {
		t.Fatalf("CountTurns: got (%d, %v) want (4, nil)", n, err)
	}

	// Export to duckdb copy
	duckDest := filepath.Join(dir, "export.duckdb")
	if err := h.CopyTo(duckDest); err != nil {
		t.Fatalf("CopyTo: %v", err)
	}
	exported, err := Open("snider", duckDest)
	if err != nil {
		t.Fatalf("Open exported: %v", err)
	}
	defer exported.Close()
	if n, err := exported.CountConversations(); err != nil || n != 1 {
		t.Fatalf("exported.CountConversations: got (%d, %v) want (1, nil)", n, err)
	}
	if n, err := exported.CountTurns(); err != nil || n != 4 {
		t.Fatalf("exported.CountTurns: got (%d, %v) want (4, nil)", n, err)
	}

	// Export to JSONL
	jsonlDest := filepath.Join(dir, "export.jsonl")
	if err := h.ExportJSONL(jsonlDest); err != nil {
		t.Fatalf("ExportJSONL: %v", err)
	}
}

// TestWriteTurnAutoIncrement — verify ordinals start at 0 and increment.
func TestWriteTurnAutoIncrement(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "chats.duckdb")
	h, err := Open("snider", path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer h.Close()

	convID, err := h.StartConversation(NewConversation{ModelID: "lemer-lite"})
	if err != nil {
		t.Fatalf("StartConversation: %v", err)
	}
	for i := 0; i < 5; i++ {
		if _, err := h.WriteTurn(convID, NewTurn{Role: "user", Content: "msg"}); err != nil {
			t.Fatalf("WriteTurn[%d]: %v", i, err)
		}
	}
	row := h.db.QueryRow(
		`SELECT MIN(ordinal), MAX(ordinal) FROM turns WHERE conversation_id = ?`, convID,
	)
	var lo, hi int
	if err := row.Scan(&lo, &hi); err != nil {
		t.Fatalf("scan: %v", err)
	}
	if lo != 0 || hi != 4 {
		t.Fatalf("ordinals: got [%d..%d] want [0..4]", lo, hi)
	}
}

// TestRequiredFields — Open / WriteTurn reject empty required args.
func TestRequiredFields(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "chats.duckdb")

	if _, err := Open("", path); err == nil {
		t.Fatal("Open with empty user_id: want error, got nil")
	}
	if _, err := Open("snider", ""); err == nil {
		t.Fatal("Open with empty path: want error, got nil")
	}

	h, _ := Open("snider", path)
	defer h.Close()
	if _, err := h.WriteTurn("", NewTurn{Role: "user", Content: "x"}); err == nil {
		t.Fatal("WriteTurn with empty conversation_id: want error, got nil")
	}

	convID, _ := h.StartConversation(NewConversation{ModelID: "lemer-lite"})
	if _, err := h.WriteTurn(convID, NewTurn{Role: "", Content: "x"}); err == nil {
		t.Fatal("WriteTurn with empty role: want error, got nil")
	}
}
