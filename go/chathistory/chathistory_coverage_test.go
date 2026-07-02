// SPDX-License-Identifier: EUPL-1.2

package chathistory

import (
	"path/filepath"
	"testing"
	"time"

	core "dappco.re/go"
)

// openTemp returns a History over a fresh temp-dir archive, registering
// Close on test cleanup. Lifts the open boilerplate out of the coverage
// cases below so each test stays focused on the behaviour under test.
//
//	h := openTemp(t)
//	conv, _ := h.StartConversation(NewConversation{ModelID: "lemer-lite"})
func openTemp(t *testing.T) *History {
	t.Helper()
	path := filepath.Join(t.TempDir(), "chats.duckdb")
	h, err := Open("snider", path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	t.Cleanup(func() { _ = h.Close() })
	return h
}

// TestChatHistory_PathUserID_Good — the path + user-id getters return the
// archive's identity exactly as constructed, with no DB or disk needed.
func TestChatHistory_PathUserID_Good(t *testing.T) {
	h := &History{path: "/x/chats.duckdb", userID: "snider"}
	core.AssertEqual(t, "/x/chats.duckdb", h.Path())
	core.AssertEqual(t, "snider", h.UserID())
}

// TestChatHistory_Close_Good — Close on a live handle releases cleanly, and
// Close on a nil handle is a harmless no-op (the h==nil guard branch).
func TestChatHistory_Close_Good(t *testing.T) {
	h := openTemp(t)
	core.AssertEqual(t, nil, h.Close())

	var nilH *History
	core.AssertEqual(t, nil, nilH.Close())
}

// TestChatHistory_Open_Bad_MkdirParent — Open fails loudly when the parent
// directory cannot be created because a path component is a regular file
// (the mkdir-parent error branch).
func TestChatHistory_Open_Bad_MkdirParent(t *testing.T) {
	dir := t.TempDir()
	fileAsParent := filepath.Join(dir, "afile")
	if r := core.WriteFile(fileAsParent, []byte("x"), 0o644); !r.OK {
		t.Fatalf("WriteFile: %v", r.Value)
	}
	_, err := Open("snider", filepath.Join(fileAsParent, "sub", "chats.duckdb"))
	core.AssertTrue(t, err != nil)
}

// TestChatHistory_Open_Bad_NotADuckdbFile — Open fails at the driver level
// when the target path already holds bytes that aren't a valid DuckDB file.
// Covers the "open duckdb" branch, distinct from the mkdir-parent branch.
func TestChatHistory_Open_Bad_NotADuckdbFile(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "garbage.duckdb")
	if r := core.WriteFile(path, []byte("not a duckdb file, just bytes"), 0o644); !r.OK {
		t.Fatalf("WriteFile: %v", r.Value)
	}
	_, err := Open("snider", path)
	core.AssertTrue(t, err != nil)
}

// TestChatHistory_ClosedGuards_Bad — every method short-circuits on a nil
// handle with a "history closed" error rather than dereferencing a nil db.
func TestChatHistory_ClosedGuards_Bad(t *testing.T) {
	var h *History
	dir := t.TempDir()

	if _, err := h.StartConversation(NewConversation{ModelID: "x"}); err == nil {
		t.Fatal("StartConversation: want error on nil handle")
	}
	if _, err := h.WriteTurn("conv", NewTurn{Role: "user", Content: "x"}); err == nil {
		t.Fatal("WriteTurn: want error on nil handle")
	}
	if err := h.EndConversation("conv"); err == nil {
		t.Fatal("EndConversation: want error on nil handle")
	}
	if err := h.SetSignal("turn", "liked"); err == nil {
		t.Fatal("SetSignal: want error on nil handle")
	}
	if _, err := h.CountConversations(); err == nil {
		t.Fatal("CountConversations: want error on nil handle")
	}
	if _, err := h.CountTurns(); err == nil {
		t.Fatal("CountTurns: want error on nil handle")
	}
	if _, err := h.RecentConversations(5); err == nil {
		t.Fatal("RecentConversations: want error on nil handle")
	}
	if _, err := h.LoadTurns("conv"); err == nil {
		t.Fatal("LoadTurns: want error on nil handle")
	}
	if err := h.CopyTo(filepath.Join(dir, "x.duckdb")); err == nil {
		t.Fatal("CopyTo: want error on nil handle")
	}
	if err := h.ExportJSONL(filepath.Join(dir, "x.jsonl")); err == nil {
		t.Fatal("ExportJSONL: want error on nil handle")
	}
}

// TestChatHistory_ClosedDB_Ugly — once Close has released the file, further
// operations against the still-non-nil handle surface the driver's
// closed-db error through the wrapped scope rather than panicking. Distinct
// from ClosedGuards_Bad: here h.db is a real (but closed) *sql.DB, so these
// exercise the Exec/Query error branches, not the h==nil guard.
func TestChatHistory_ClosedDB_Ugly(t *testing.T) {
	dir := t.TempDir()
	h, err := Open("snider", filepath.Join(dir, "chats.duckdb"))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	if err := h.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	if _, err := h.CountConversations(); err == nil {
		t.Fatal("CountConversations on closed db: want error")
	}
	if _, err := h.CountTurns(); err == nil {
		t.Fatal("CountTurns on closed db: want error")
	}
	if _, err := h.RecentConversations(5); err == nil {
		t.Fatal("RecentConversations on closed db: want error")
	}
	if _, err := h.LoadTurns("conv"); err == nil {
		t.Fatal("LoadTurns on closed db: want error")
	}
	if _, err := h.WriteTurn("conv", NewTurn{Role: "user", Content: "x"}); err == nil {
		t.Fatal("WriteTurn on closed db: want error")
	}
	if err := h.SetSignal("turn", "liked"); err == nil {
		t.Fatal("SetSignal on closed db: want error")
	}
	if err := h.EndConversation("conv"); err == nil {
		t.Fatal("EndConversation on closed db: want error")
	}
	if _, err := h.StartConversation(NewConversation{ModelID: "x"}); err == nil {
		t.Fatal("StartConversation on closed db: want error")
	}
	if err := h.CopyTo(filepath.Join(dir, "copy.duckdb")); err == nil {
		t.Fatal("CopyTo on closed db: want error")
	}
	if err := h.ExportJSONL(filepath.Join(dir, "out.jsonl")); err == nil {
		t.Fatal("ExportJSONL on closed db: want error")
	}
}

// TestChatHistory_StartConversation_Good_TagsMetadata — the tags-present and
// metadata-present branches, plus an explicit non-zero ConsentVersion,
// round-trip to storage. Metadata + consent_version aren't exposed by any
// public read path, so this asserts directly against the stored row.
func TestChatHistory_StartConversation_Good_TagsMetadata(t *testing.T) {
	h := openTemp(t)
	conv, err := h.StartConversation(NewConversation{
		Title:          "evening vent",
		ModelID:        "lemer-lite",
		BaseModel:      "gemma-4-e2b-it-4bit",
		AdapterID:      "lek2",
		Tags:           []string{"life", "vent"},
		Metadata:       []byte(`{"client":"desktop"}`),
		ConsentVersion: 3,
	})
	if err != nil {
		t.Fatalf("StartConversation: %v", err)
	}
	core.AssertTrue(t, conv != "")

	var metadata string
	var consent int
	row := h.db.QueryRow(`SELECT metadata, consent_version FROM conversations WHERE id = ?`, conv)
	if err := row.Scan(&metadata, &consent); err != nil {
		t.Fatalf("scan: %v", err)
	}
	core.AssertEqual(t, `{"client":"desktop"}`, metadata)
	core.AssertEqual(t, 3, consent)
}

// TestChatHistory_StartConversation_Good_DefaultConsent — a zero
// ConsentVersion persists as 1, not 0 (the consent==0 default branch).
func TestChatHistory_StartConversation_Good_DefaultConsent(t *testing.T) {
	h := openTemp(t)
	conv, err := h.StartConversation(NewConversation{ModelID: "lemer-lite"})
	if err != nil {
		t.Fatalf("StartConversation: %v", err)
	}
	var consent int
	row := h.db.QueryRow(`SELECT consent_version FROM conversations WHERE id = ?`, conv)
	if err := row.Scan(&consent); err != nil {
		t.Fatalf("scan: %v", err)
	}
	core.AssertEqual(t, 1, consent)
}

// TestChatHistory_WriteTurn_Good_ToolFieldsAndTokens — the tool_calls,
// tool_results and token-count columns persist (nullableJSON / nullableInt
// non-empty branches) and read back through LoadTurns.
func TestChatHistory_WriteTurn_Good_ToolFieldsAndTokens(t *testing.T) {
	h := openTemp(t)
	conv, err := h.StartConversation(NewConversation{ModelID: "lemer-lite"})
	if err != nil {
		t.Fatalf("StartConversation: %v", err)
	}
	turnID, err := h.WriteTurn(conv, NewTurn{
		Role:        "assistant",
		Content:     "calling a tool",
		ToolCalls:   []byte(`[{"name":"search"}]`),
		ToolResults: []byte(`[{"hits":2}]`),
		TokensIn:    16,
		TokensOut:   8,
	})
	if err != nil {
		t.Fatalf("WriteTurn: %v", err)
	}
	core.AssertTrue(t, turnID != "")

	turns, err := h.LoadTurns(conv)
	core.AssertEqual(t, nil, err)
	core.AssertEqual(t, 1, len(turns))
	core.AssertEqual(t, "assistant", turns[0].Role)
}

// TestChatHistory_WriteTurn_Ugly_MissingConversation — a syntactically valid
// but nonexistent conversation id passes the ordinal lookup (no turns yet,
// COALESCE gives ordinal 0) but fails the insert on the turns→conversations
// foreign key. Distinct from the ordinal-lookup-error branch covered by
// ClosedDB_Ugly — this hits the later insert-error branch on an open db.
func TestChatHistory_WriteTurn_Ugly_MissingConversation(t *testing.T) {
	h := openTemp(t)
	_, err := h.WriteTurn("no-such-conversation", NewTurn{Role: "user", Content: "x"})
	core.AssertTrue(t, err != nil)
}

// TestChatHistory_EndConversation_Good_Idempotent — EndConversation on an
// open conversation stamps ended_at, and a second call is a harmless no-op:
// the "AND ended_at IS NULL" guard means the timestamp doesn't move.
func TestChatHistory_EndConversation_Good_Idempotent(t *testing.T) {
	h := openTemp(t)
	conv, err := h.StartConversation(NewConversation{ModelID: "lemer-lite"})
	if err != nil {
		t.Fatalf("StartConversation: %v", err)
	}
	if err := h.EndConversation(conv); err != nil {
		t.Fatalf("EndConversation (first): %v", err)
	}
	var first time.Time
	row := h.db.QueryRow(`SELECT ended_at FROM conversations WHERE id = ?`, conv)
	if err := row.Scan(&first); err != nil {
		t.Fatalf("scan (first): %v", err)
	}
	core.AssertTrue(t, !first.IsZero())

	if err := h.EndConversation(conv); err != nil {
		t.Fatalf("EndConversation (idempotent): %v", err)
	}
	var second time.Time
	row = h.db.QueryRow(`SELECT ended_at FROM conversations WHERE id = ?`, conv)
	if err := row.Scan(&second); err != nil {
		t.Fatalf("scan (second): %v", err)
	}
	core.AssertTrue(t, first.Equal(second))
}

// TestChatHistory_RecentConversations_Good — results come back newest-first,
// each row carrying id/title/model_id, and LIMIT caps the count below the
// total available. Timestamps are pinned via direct UPDATE rather than
// relying on wall-clock gaps between fast successive inserts.
func TestChatHistory_RecentConversations_Good(t *testing.T) {
	h := openTemp(t)
	base := time.Date(2026, 1, 1, 12, 0, 0, 0, time.UTC)
	titles := []string{"first", "second", "third"}
	ids := make([]string, len(titles))
	for i, title := range titles {
		id, err := h.StartConversation(NewConversation{Title: title, ModelID: "lemer-lite"})
		if err != nil {
			t.Fatalf("StartConversation[%d]: %v", i, err)
		}
		ts := base.Add(time.Duration(i) * time.Hour)
		if _, err := h.db.Exec(`UPDATE conversations SET started_at = ? WHERE id = ?`, ts, id); err != nil {
			t.Fatalf("set started_at[%d]: %v", i, err)
		}
		ids[i] = id
	}

	recents, err := h.RecentConversations(2)
	if err != nil {
		t.Fatalf("RecentConversations: %v", err)
	}
	core.AssertEqual(t, 2, len(recents))
	core.AssertEqual(t, "third", recents[0].Title)
	core.AssertEqual(t, ids[2], recents[0].ID)
	core.AssertEqual(t, "lemer-lite", recents[0].ModelID)
	core.AssertEqual(t, "second", recents[1].Title)
	core.AssertEqual(t, ids[1], recents[1].ID)
}

// TestChatHistory_RecentConversations_Good_UserFilter — a row seeded for a
// different user_id (inserted directly, bypassing StartConversation's own
// h.userID binding) never appears in this user's recents — proves the
// WHERE user_id = ? filter rather than merely "there's one row total".
func TestChatHistory_RecentConversations_Good_UserFilter(t *testing.T) {
	h := openTemp(t)
	mine, err := h.StartConversation(NewConversation{Title: "mine", ModelID: "lemer-lite"})
	if err != nil {
		t.Fatalf("StartConversation: %v", err)
	}
	if _, err := h.db.Exec(
		`INSERT INTO conversations (id, user_id, title, started_at, consent_version)
		 VALUES (?, ?, ?, ?, ?)`,
		"other-conv-id", "otheruser", "theirs", time.Now().UTC(), 1,
	); err != nil {
		t.Fatalf("seed other user row: %v", err)
	}

	recents, err := h.RecentConversations(10)
	if err != nil {
		t.Fatalf("RecentConversations: %v", err)
	}
	core.AssertEqual(t, 1, len(recents))
	core.AssertEqual(t, mine, recents[0].ID)
}

// TestChatHistory_RecentConversations_Good_DefaultLimit — limit<=0 falls
// back to 10 rather than returning every conversation or none (the
// limit<=0 default branch).
func TestChatHistory_RecentConversations_Good_DefaultLimit(t *testing.T) {
	h := openTemp(t)
	for i := 0; i < 12; i++ {
		if _, err := h.StartConversation(NewConversation{ModelID: "lemer-lite"}); err != nil {
			t.Fatalf("StartConversation[%d]: %v", i, err)
		}
	}
	recents, err := h.RecentConversations(0)
	if err != nil {
		t.Fatalf("RecentConversations: %v", err)
	}
	core.AssertEqual(t, 10, len(recents))
}

// TestChatHistory_LoadTurns_Good — turns come back in ordinal order with the
// role + content + ordinal triple the consumer replays into the next call.
func TestChatHistory_LoadTurns_Good(t *testing.T) {
	h := openTemp(t)
	conv, err := h.StartConversation(NewConversation{ModelID: "lemer-lite"})
	if err != nil {
		t.Fatalf("StartConversation: %v", err)
	}
	want := []NewTurn{
		{Role: "user", Content: "first"},
		{Role: "assistant", Content: "second"},
		{Role: "user", Content: "third"},
	}
	for i, nt := range want {
		if _, err := h.WriteTurn(conv, nt); err != nil {
			t.Fatalf("WriteTurn[%d]: %v", i, err)
		}
	}

	turns, err := h.LoadTurns(conv)
	if err != nil {
		t.Fatalf("LoadTurns: %v", err)
	}
	core.AssertEqual(t, len(want), len(turns))
	for i, tn := range turns {
		core.AssertEqual(t, i, tn.Ordinal)
		core.AssertEqual(t, want[i].Role, tn.Role)
		core.AssertEqual(t, want[i].Content, tn.Content)
	}
}

// TestChatHistory_LoadTurns_Good_Empty — an unknown conversation id yields
// zero turns and no error (the iterate-nothing branch).
func TestChatHistory_LoadTurns_Good_Empty(t *testing.T) {
	h := openTemp(t)
	turns, err := h.LoadTurns("no-such-conversation")
	core.AssertEqual(t, nil, err)
	core.AssertEqual(t, 0, len(turns))
}

// TestChatHistory_LoadTurns_Bad_EmptyID — an empty conversation id is
// rejected before any query runs.
func TestChatHistory_LoadTurns_Bad_EmptyID(t *testing.T) {
	h := openTemp(t)
	_, err := h.LoadTurns("")
	core.AssertTrue(t, err != nil)
}

// TestChatHistory_LoadTurns_Bad_Closed — a nil or zero-value history errors
// instead of dereferencing a nil db (LoadTurns' own copy of the guard, since
// it also has the extra empty-id check ClosedGuards_Bad doesn't exercise).
func TestChatHistory_LoadTurns_Bad_Closed(t *testing.T) {
	var nilH *History
	_, err := nilH.LoadTurns("conv-1")
	core.AssertTrue(t, err != nil)
	_, err = (&History{}).LoadTurns("conv-1")
	core.AssertTrue(t, err != nil)
}

// TestChatHistory_NullableJSON — the helper maps empty/nil bytes to a SQL
// NULL and non-empty bytes to their string form.
func TestChatHistory_NullableJSON(t *testing.T) {
	core.AssertEqual(t, nil, nullableJSON(nil))
	core.AssertEqual(t, nil, nullableJSON([]byte{}))
	core.AssertEqual(t, `{"a":1}`, nullableJSON([]byte(`{"a":1}`)))
}
