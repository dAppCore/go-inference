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

// TestExport_CopyTo_Ugly_SourceRemoved — the source file is removed from
// disk after Open (the driver keeps its own fd, so CHECKPOINT still
// succeeds), so the fresh-by-path open of the source for copying fails —
// the "open source" branch, distinct from the checkpoint-error branch
// covered by ClosedDB_Ugly.
func TestExport_CopyTo_Ugly_SourceRemoved(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "chats.duckdb")
	h, err := Open("snider", path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer h.Close()
	if _, err := h.StartConversation(NewConversation{ModelID: "x"}); err != nil {
		t.Fatalf("StartConversation: %v", err)
	}
	if r := core.Remove(path); !r.OK {
		t.Fatalf("remove source: %v", r.Value)
	}
	err = h.CopyTo(filepath.Join(dir, "copy.duckdb"))
	core.AssertTrue(t, err != nil)
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
