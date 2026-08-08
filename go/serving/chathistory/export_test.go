// SPDX-License-Identifier: EUPL-1.2

package chathistory

import (
	"path/filepath"
	"testing"

	core "dappco.re/go"
)

// TestExport_CopyTo_Bad_EmptyDest — an empty destination is rejected before
// any file work happens.
func TestExport_CopyTo_Bad_EmptyDest(t *testing.T) {
	h := openTemp(t)
	core.AssertTrue(t, h.CopyTo("") != nil)
}

// TestExport_CopyTo_Good_NestedDest — CopyTo creates a missing parent
// directory for the destination, then writes the checkpointed file there,
// and the copy is independently openable with the same row counts.
func TestExport_CopyTo_Good_NestedDest(t *testing.T) {
	h := openTemp(t)
	conv, err := h.StartConversation(NewConversation{ModelID: "lemer-lite"})
	if err != nil {
		t.Fatalf("StartConversation: %v", err)
	}
	if _, err := h.WriteTurn(conv, NewTurn{Role: "user", Content: "hey"}); err != nil {
		t.Fatalf("WriteTurn: %v", err)
	}

	dest := filepath.Join(t.TempDir(), "deep", "nested", "copy.duckdb")
	if err := h.CopyTo(dest); err != nil {
		t.Fatalf("CopyTo: %v", err)
	}
	core.AssertTrue(t, core.Stat(dest).OK)

	exported, err := Open("snider", dest)
	if err != nil {
		t.Fatalf("Open copy: %v", err)
	}
	defer exported.Close()
	n, err := exported.CountTurns()
	core.AssertEqual(t, nil, err)
	core.AssertEqual(t, 1, n)
}

// TestExport_CopyTo_Bad_MkdirParent — the destination's parent can't be
// created because a path component is a regular file.
func TestExport_CopyTo_Bad_MkdirParent(t *testing.T) {
	h := openTemp(t)
	dir := t.TempDir()
	fileAsParent := filepath.Join(dir, "afile")
	if r := core.WriteFile(fileAsParent, []byte("x"), 0o644); !r.OK {
		t.Fatalf("WriteFile: %v", r.Value)
	}
	err := h.CopyTo(filepath.Join(fileAsParent, "sub", "copy.duckdb"))
	core.AssertTrue(t, err != nil)
}

// TestExport_CopyTo_Ugly_DestIsDirectory — the destination path already
// exists as a directory, so creating the destination file fails.
func TestExport_CopyTo_Ugly_DestIsDirectory(t *testing.T) {
	h := openTemp(t)
	destDir := filepath.Join(t.TempDir(), "adir")
	if r := core.MkdirAll(destDir, 0o755); !r.OK {
		t.Fatalf("MkdirAll: %v", r.Value)
	}
	err := h.CopyTo(destDir)
	core.AssertTrue(t, err != nil)
}

// TestExport_CopyTo_Ugly_StaleDestReplaced — exporting twice to the same path
// replaces the file rather than merging into it. ATTACH opens an existing
// database instead of truncating it, so without the clear step the second
// export would hit the first one's rows and fail the primary key, or silently
// accumulate. The second archive here holds a different conversation, so a
// merge would show up as two.
func TestExport_CopyTo_Ugly_StaleDestReplaced(t *testing.T) {
	dir := t.TempDir()
	dest := filepath.Join(dir, "export.duckdb")

	first, err := Open("snider", filepath.Join(dir, "first.duckdb"))
	if err != nil {
		t.Fatalf("Open(first): %v", err)
	}
	if _, err := first.StartConversation(NewConversation{ModelID: "a"}); err != nil {
		t.Fatalf("StartConversation(first): %v", err)
	}
	if err := first.CopyTo(dest); err != nil {
		t.Fatalf("CopyTo(first): %v", err)
	}
	if err := first.Close(); err != nil {
		t.Fatalf("Close(first): %v", err)
	}

	second, err := Open("snider", filepath.Join(dir, "second.duckdb"))
	if err != nil {
		t.Fatalf("Open(second): %v", err)
	}
	defer second.Close()
	wantID, err := second.StartConversation(NewConversation{ModelID: "b"})
	if err != nil {
		t.Fatalf("StartConversation(second): %v", err)
	}
	if err := second.CopyTo(dest); err != nil {
		t.Fatalf("CopyTo(second) over a stale export: %v", err)
	}

	exported, err := Open("snider", dest)
	if err != nil {
		t.Fatalf("Open(export): %v", err)
	}
	defer exported.Close()
	conversations, err := exported.RecentConversations(10)
	if err != nil {
		t.Fatalf("RecentConversations(export): %v", err)
	}
	if len(conversations) != 1 || conversations[0].ID != wantID {
		t.Fatalf("re-export holds %d conversations (%+v), want only the second archive's (%s)",
			len(conversations), conversations, wantID)
	}
}

// TestExport_CopyTo_Bad_ClosedHistory — a History whose database has been
// closed cannot export. The connection cannot be acquired, so the failure is
// reported before anything touches the destination path.
func TestExport_CopyTo_Bad_ClosedHistory(t *testing.T) {
	dir := t.TempDir()
	h, err := Open("snider", filepath.Join(dir, "chats.duckdb"))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	if err := h.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	dest := filepath.Join(dir, "export.duckdb")
	if err := h.CopyTo(dest); err == nil {
		t.Fatal("CopyTo on a closed history = nil, want an error")
	}
	if stat := core.Stat(dest); stat.OK {
		t.Fatal("CopyTo on a closed history created the destination; it must fail before touching it")
	}
}

// TestExport_CopyTo_Ugly_ForeignKeyChain is the direct receipt for the copy
// ORDER. The schema chains conversations <- turns <- embeddings, and DuckDB's
// own COPY FROM DATABASE does not topologically sort: pointed at this schema
// it aborts partway with "Violates foreign key constraint because key ... does
// not exist in the referenced table". CopyTo therefore writes the tables out
// itself, parents first.
//
// embeddings has no public writer — it is a sidecar reserved for a future
// embedding model — so the row is seeded directly. Without that the last link
// in the chain is copied only ever empty, and an ordering regression there
// would pass unnoticed.
func TestExport_CopyTo_Ugly_ForeignKeyChain(t *testing.T) {
	dir := t.TempDir()
	h, err := Open("snider", filepath.Join(dir, "chats.duckdb"))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer h.Close()

	conversationID, err := h.StartConversation(NewConversation{ModelID: "gemma"})
	if err != nil {
		t.Fatalf("StartConversation: %v", err)
	}
	turnID, err := h.WriteTurn(conversationID, NewTurn{Role: "user", Content: "hello"})
	if err != nil {
		t.Fatalf("WriteTurn: %v", err)
	}
	if _, err := h.db.Exec(
		`INSERT INTO embeddings (turn_id, embedding_model, vector) VALUES (?, ?, NULL)`,
		turnID, "test-embed"); err != nil {
		t.Fatalf("seed embeddings: %v", err)
	}

	dest := filepath.Join(dir, "export.duckdb")
	if err := h.CopyTo(dest); err != nil {
		t.Fatalf("CopyTo: %v", err)
	}

	exported, err := Open("snider", dest)
	if err != nil {
		t.Fatalf("Open(export): %v", err)
	}
	defer exported.Close()

	// Every link in the chain must have survived, and the child rows must
	// still point at parents that exist — which is what the FK would have
	// refused had the order been wrong.
	for _, tc := range []struct {
		query string
		want  int
	}{
		{`SELECT count(*) FROM conversations`, 1},
		{`SELECT count(*) FROM turns`, 1},
		{`SELECT count(*) FROM embeddings`, 1},
		{`SELECT count(*) FROM turns t JOIN conversations c ON t.conversation_id = c.id`, 1},
		{`SELECT count(*) FROM embeddings e JOIN turns t ON e.turn_id = t.id`, 1},
	} {
		var got int
		if err := exported.db.QueryRow(tc.query).Scan(&got); err != nil {
			t.Fatalf("%s: %v", tc.query, err)
		}
		if got != tc.want {
			t.Errorf("%s = %d, want %d", tc.query, got, tc.want)
		}
	}

	// The export must be schema-identical to a live archive, not a
	// constraint-free CREATE TABLE AS SELECT: the foreign key has to still be
	// there, or a tool opening the file gets a weaker database than the one
	// it was exported from.
	var constraints int
	if err := exported.db.QueryRow(
		`SELECT count(*) FROM duckdb_constraints()
		  WHERE table_name = 'turns' AND constraint_type = 'FOREIGN KEY'`).Scan(&constraints); err != nil {
		t.Fatalf("duckdb_constraints: %v", err)
	}
	if constraints != 1 {
		t.Errorf("exported turns carries %d foreign keys, want 1", constraints)
	}
}

// TestExport_CopyTo_Ugly_SourceRemoved — the source file is unlinked after
// Open. The driver keeps its own descriptor, so the archive is still fully
// readable, and the export therefore SUCCEEDS with the data intact.
//
// This assertion is the inverse of the one it replaces. The old CopyTo
// re-opened the source by path to byte-copy it, so an unlinked file made the
// export fail — the previous test pinned that "open source" branch by name.
// Re-opening by path is exactly what does not work on Windows, where DuckDB
// holds its file without FILE_SHARE_READ; the engine now copies from its own
// open database, so a path that has gone away no longer costs the user their
// export. Failing here would now be the bug.
func TestExport_CopyTo_Ugly_SourceRemoved(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "chats.duckdb")
	h, err := Open("snider", path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer h.Close()
	conversationID, err := h.StartConversation(NewConversation{ModelID: "x"})
	if err != nil {
		t.Fatalf("StartConversation: %v", err)
	}
	if r := core.Remove(path); !r.OK {
		t.Fatalf("remove source: %v", r.Value)
	}

	dest := filepath.Join(dir, "copy.duckdb")
	if err := h.CopyTo(dest); err != nil {
		t.Fatalf("CopyTo after source unlink: %v", err)
	}

	// The export is not merely present — it carries the row written before
	// the unlink, which is the claim that matters.
	copied, err := Open("snider", dest)
	if err != nil {
		t.Fatalf("Open(export): %v", err)
	}
	defer copied.Close()
	conversations, err := copied.RecentConversations(10)
	if err != nil {
		t.Fatalf("RecentConversations(export): %v", err)
	}
	if len(conversations) != 1 || conversations[0].ID != conversationID {
		t.Fatalf("export carried %d conversations (%+v), want the one written before the unlink (%s)",
			len(conversations), conversations, conversationID)
	}
}

// TestExport_ExportJSONL_Bad_EmptyDest — an empty destination is rejected.
func TestExport_ExportJSONL_Bad_EmptyDest(t *testing.T) {
	h := openTemp(t)
	core.AssertTrue(t, h.ExportJSONL("") != nil)
}

// TestExport_ExportJSONL_Bad_MkdirParent — the destination's parent can't
// be created because a path component is a regular file.
func TestExport_ExportJSONL_Bad_MkdirParent(t *testing.T) {
	h := openTemp(t)
	dir := t.TempDir()
	fileAsParent := filepath.Join(dir, "afile")
	if r := core.WriteFile(fileAsParent, []byte("x"), 0o644); !r.OK {
		t.Fatalf("WriteFile: %v", r.Value)
	}
	err := h.ExportJSONL(filepath.Join(fileAsParent, "sub", "out.jsonl"))
	core.AssertTrue(t, err != nil)
}

// TestExport_ExportJSONL_Ugly_DestIsDirectory — the destination already
// exists as a directory, so creating the destination file fails.
func TestExport_ExportJSONL_Ugly_DestIsDirectory(t *testing.T) {
	h := openTemp(t)
	destDir := filepath.Join(t.TempDir(), "adir")
	if r := core.MkdirAll(destDir, 0o755); !r.OK {
		t.Fatalf("MkdirAll: %v", r.Value)
	}
	err := h.ExportJSONL(destDir)
	core.AssertTrue(t, err != nil)
}

// TestExport_ExportJSONL_Good_AllFields — a fully-populated conversation
// (ended, tagged, with tool fields + tokens + signal) exports a JSONL line
// that carries every optional field through the nullable-scan branches.
func TestExport_ExportJSONL_Good_AllFields(t *testing.T) {
	h := openTemp(t)
	conv, err := h.StartConversation(NewConversation{
		Title:          "vent",
		ModelID:        "lemer-lite",
		BaseModel:      "gemma-4-e2b-it-4bit",
		AdapterID:      "lek2",
		Tags:           []string{"life"},
		ConsentVersion: 2,
	})
	if err != nil {
		t.Fatalf("StartConversation: %v", err)
	}
	turnID, err := h.WriteTurn(conv, NewTurn{
		Role:        "assistant",
		Content:     "hi",
		ToolCalls:   []byte(`[{"name":"search"}]`),
		ToolResults: []byte(`[{"hits":1}]`),
		TokensIn:    5,
		TokensOut:   7,
	})
	if err != nil {
		t.Fatalf("WriteTurn: %v", err)
	}
	if err := h.SetSignal(turnID, "liked"); err != nil {
		t.Fatalf("SetSignal: %v", err)
	}
	if err := h.EndConversation(conv); err != nil {
		t.Fatalf("EndConversation: %v", err)
	}

	dest := filepath.Join(t.TempDir(), "out.jsonl")
	if err := h.ExportJSONL(dest); err != nil {
		t.Fatalf("ExportJSONL: %v", err)
	}

	r := core.ReadFile(dest)
	if !r.OK {
		t.Fatalf("ReadFile: %v", r.Value)
	}
	var line JSONLConversation
	if u := core.JSONUnmarshal(firstLine(r.Value.([]byte)), &line); !u.OK {
		t.Fatalf("JSONUnmarshal: %v", u.Value)
	}

	core.AssertEqual(t, conv, line.ID)
	core.AssertEqual(t, "snider", line.UserID)
	core.AssertEqual(t, "vent", line.Title)
	core.AssertEqual(t, "lemer-lite", line.ModelID)
	core.AssertEqual(t, "gemma-4-e2b-it-4bit", line.BaseModel)
	core.AssertEqual(t, "lek2", line.AdapterID)
	core.AssertEqual(t, 2, line.ConsentVersion)
	core.AssertTrue(t, line.EndedAt != nil)
	core.AssertEqual(t, 1, len(line.Tags))
	core.AssertEqual(t, 1, len(line.Turns))

	turn := line.Turns[0]
	core.AssertEqual(t, "assistant", turn.Role)
	core.AssertEqual(t, "hi", turn.Content)
	core.AssertEqual(t, 5, turn.TokensIn)
	core.AssertEqual(t, 7, turn.TokensOut)
	core.AssertEqual(t, "liked", turn.Signal)
	core.AssertTrue(t, len(turn.ToolCalls) > 0)
	core.AssertTrue(t, len(turn.ToolResults) > 0)
}

// TestExport_ExportJSONL_Good_Empty — an archive with no conversations
// exports an empty file without error (the loop-body-never-runs path).
func TestExport_ExportJSONL_Good_Empty(t *testing.T) {
	h := openTemp(t)
	dest := filepath.Join(t.TempDir(), "empty.jsonl")
	if err := h.ExportJSONL(dest); err != nil {
		t.Fatalf("ExportJSONL: %v", err)
	}
	r := core.ReadFile(dest)
	if !r.OK {
		t.Fatalf("ReadFile: %v", r.Value)
	}
	core.AssertEqual(t, 0, len(r.Value.([]byte)))
}

// TestExport_ExportJSONL_Good_NoTagsNotEnded — a conversation with no tags
// that hasn't been ended exercises the false arms of the tags-present and
// endedAt.Valid branches (Good_AllFields only exercises the true arms).
func TestExport_ExportJSONL_Good_NoTagsNotEnded(t *testing.T) {
	h := openTemp(t)
	conv, err := h.StartConversation(NewConversation{ModelID: "lemer-lite"})
	if err != nil {
		t.Fatalf("StartConversation: %v", err)
	}
	dest := filepath.Join(t.TempDir(), "out.jsonl")
	if err := h.ExportJSONL(dest); err != nil {
		t.Fatalf("ExportJSONL: %v", err)
	}
	r := core.ReadFile(dest)
	if !r.OK {
		t.Fatalf("ReadFile: %v", r.Value)
	}
	var line JSONLConversation
	if u := core.JSONUnmarshal(firstLine(r.Value.([]byte)), &line); !u.OK {
		t.Fatalf("JSONUnmarshal: %v", u.Value)
	}
	core.AssertEqual(t, conv, line.ID)
	core.AssertTrue(t, line.EndedAt == nil)
	core.AssertEqual(t, 0, len(line.Tags))
}

// TestExport_ExportJSONL_Ugly_GarbageTags — invalid JSON in the tags column
// (bypassing StartConversation's own marshal, e.g. from an external write or
// partial migration) logs a warning and still produces a usable export
// rather than failing the whole run — partial export beats no export.
func TestExport_ExportJSONL_Ugly_GarbageTags(t *testing.T) {
	h := openTemp(t)
	conv, err := h.StartConversation(NewConversation{ModelID: "lemer-lite", Tags: []string{"a"}})
	if err != nil {
		t.Fatalf("StartConversation: %v", err)
	}
	if _, err := h.db.Exec(`UPDATE conversations SET tags = ? WHERE id = ?`, "{not valid json", conv); err != nil {
		t.Fatalf("corrupt tags: %v", err)
	}

	dest := filepath.Join(t.TempDir(), "out.jsonl")
	if err := h.ExportJSONL(dest); err != nil {
		t.Fatalf("ExportJSONL: %v", err)
	}
	r := core.ReadFile(dest)
	if !r.OK {
		t.Fatalf("ReadFile: %v", r.Value)
	}
	var line JSONLConversation
	if u := core.JSONUnmarshal(firstLine(r.Value.([]byte)), &line); !u.OK {
		t.Fatalf("JSONUnmarshal: %v", u.Value)
	}
	core.AssertEqual(t, conv, line.ID)
	core.AssertEqual(t, 0, len(line.Tags))
}

// firstLine returns the bytes up to (not including) the first newline, so a
// single-record JSONL export can be unmarshalled directly.
func firstLine(b []byte) []byte {
	for i, c := range b {
		if c == '\n' {
			return b[:i]
		}
	}
	return b
}
