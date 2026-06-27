// SPDX-License-Identifier: EUPL-1.2

package chathistory

import (
	"database/sql"
	"io"
	"time"

	core "dappco.re/go"
)

// CopyTo copies the live DuckDB file to dest. The user-friendly export
// path: hand them a single .duckdb they can open in any tool. The
// source file is checkpointed first to ensure all WAL writes are
// flushed into the main file.
//
// This is the simplest export — the file IS the format. For tools
// that prefer line-delimited records, ExportJSONL.
//
//	if err := h.CopyTo("/Users/snider/Downloads/snider-chats-2026-05-26.duckdb"); err != nil { ... }
func (h *History) CopyTo(dest string) error {
	if h == nil || h.db == nil {
		return core.E("chathistory.CopyTo", "history closed", nil)
	}
	if core.Trim(dest) == "" {
		return core.E("chathistory.CopyTo", "dest required", nil)
	}
	if _, err := h.db.Exec(`CHECKPOINT`); err != nil {
		return core.E("chathistory.CopyTo", "checkpoint", err)
	}
	srcResult := core.Open(h.path)
	if !srcResult.OK {
		return core.E("chathistory.CopyTo", "open source", srcResult.Value.(error))
	}
	src := srcResult.Value.(*core.OSFile)
	defer src.Close()
	if dir := core.PathDir(dest); dir != "" {
		if r := core.MkdirAll(dir, 0o755); !r.OK {
			return core.E("chathistory.CopyTo", "mkdir dest parent", r.Value.(error))
		}
	}
	dstResult := core.Create(dest)
	if !dstResult.OK {
		return core.E("chathistory.CopyTo", "create dest", dstResult.Value.(error))
	}
	dst := dstResult.Value.(*core.OSFile)
	// Close error matters on the success path — disk-full /
	// network-drive errors often surface only at Close, not during
	// Write. Defer here would discard them. Explicit Close after
	// Copy means a partial-file failure becomes a returned error
	// rather than a "succeeded but file is corrupt" surprise.
	if _, err := io.Copy(dst, src); err != nil {
		_ = dst.Close()
		return core.E("chathistory.CopyTo", "copy bytes", err)
	}
	if err := dst.Close(); err != nil {
		return core.E("chathistory.CopyTo", "close dest", err)
	}
	return nil
}

// JSONLConversation is one record line in the JSONL export. Shape is
// self-describing — any tool that reads JSONL can consume the archive
// without DuckDB. Future LoRA training data prep should prefer the
// .duckdb (richer query surface), but JSONL is the non-technical
// user's option.
type JSONLConversation struct {
	ID             string      `json:"id"`
	UserID         string      `json:"user_id"`
	Title          string      `json:"title,omitempty"`
	StartedAt      time.Time   `json:"started_at"`
	EndedAt        *time.Time  `json:"ended_at,omitempty"`
	ModelID        string      `json:"model_id,omitempty"`
	BaseModel      string      `json:"base_model,omitempty"`
	AdapterID      string      `json:"adapter_id,omitempty"`
	Tags           []string    `json:"tags,omitempty"`
	ConsentVersion int         `json:"consent_version"`
	Turns          []JSONLTurn `json:"turns"`
}

// JSONLTurn is one message inside a conversation's `turns` array.
type JSONLTurn struct {
	ID          string          `json:"id"`
	Ordinal     int             `json:"ordinal"`
	Role        string          `json:"role"`
	Content     string          `json:"content"`
	ToolCalls   core.RawMessage `json:"tool_calls,omitempty"`
	ToolResults core.RawMessage `json:"tool_results,omitempty"`
	CreatedAt   time.Time       `json:"created_at"`
	TokensIn    int             `json:"tokens_in,omitempty"`
	TokensOut   int             `json:"tokens_out,omitempty"`
	Signal      string          `json:"signal,omitempty"`
}

// ExportJSONL writes one conversation per line to dest. Each line is
// a JSONLConversation with all turns inlined. Order is by started_at.
//
//	if err := h.ExportJSONL("/Users/snider/Downloads/chats.jsonl"); err != nil { ... }
func (h *History) ExportJSONL(dest string) error {
	if h == nil || h.db == nil {
		return core.E("chathistory.ExportJSONL", "history closed", nil)
	}
	if core.Trim(dest) == "" {
		return core.E("chathistory.ExportJSONL", "dest required", nil)
	}
	if dir := core.PathDir(dest); dir != "" {
		if r := core.MkdirAll(dir, 0o755); !r.OK {
			return core.E("chathistory.ExportJSONL", "mkdir dest parent", r.Value.(error))
		}
	}
	fResult := core.Create(dest)
	if !fResult.OK {
		return core.E("chathistory.ExportJSONL", "create dest", fResult.Value.(error))
	}
	f := fResult.Value.(*core.OSFile)
	// Belt-and-braces — defer guarantees the fd never leaks on
	// any return path; the success path below ALSO calls Close
	// explicitly so the writer's flush failure (disk full,
	// network drive, etc.) becomes a returned error instead of
	// being silently swallowed by the defer.
	defer f.Close()

	convRows, err := h.db.Query(
		`SELECT id, user_id, title, started_at, ended_at, model_id, base_model,
		        adapter_id, tags, consent_version
		   FROM conversations
		  ORDER BY started_at`,
	)
	if err != nil {
		return core.E("chathistory.ExportJSONL", "query conversations", err)
	}
	defer convRows.Close()

	for convRows.Next() {
		var c JSONLConversation
		var title, modelID, baseModel, adapterID sql.NullString
		var endedAt sql.NullTime
		var tagsJSON sql.NullString
		if err := convRows.Scan(
			&c.ID, &c.UserID, &title, &c.StartedAt, &endedAt,
			&modelID, &baseModel, &adapterID, &tagsJSON, &c.ConsentVersion,
		); err != nil {
			return core.E("chathistory.ExportJSONL", "scan conversation", err)
		}
		c.Title = title.String
		c.ModelID = modelID.String
		c.BaseModel = baseModel.String
		c.AdapterID = adapterID.String
		if endedAt.Valid {
			c.EndedAt = &endedAt.Time
		}
		if tagsJSON.Valid && tagsJSON.String != "" {
			// A decode failure here means the tags column carries
			// garbage JSON (external write, partial migration, disk
			// corruption). Don't fail the export — partial export
			// with logged drift beats refusing to ship anything —
			// but log so audit / activity can correlate later when
			// the user notices missing tags on a re-imported file.
			if r := core.JSONUnmarshal([]byte(tagsJSON.String), &c.Tags); !r.OK {
				core.Warn("chathistory.export.tags_decode_failed",
					"conversation_id", c.ID, "error", r.Error())
			}
		}

		turnRows, err := h.db.Query(
			`SELECT id, ordinal, role, content, tool_calls, tool_results,
			        created_at, tokens_in, tokens_out, signal
			   FROM turns
			  WHERE conversation_id = ?
			  ORDER BY ordinal`,
			c.ID,
		)
		if err != nil {
			return core.E("chathistory.ExportJSONL", "query turns", err)
		}
		for turnRows.Next() {
			var t JSONLTurn
			var toolCalls, toolResults sql.NullString
			var tokensIn, tokensOut sql.NullInt32
			var signal sql.NullString
			if err := turnRows.Scan(
				&t.ID, &t.Ordinal, &t.Role, &t.Content,
				&toolCalls, &toolResults, &t.CreatedAt,
				&tokensIn, &tokensOut, &signal,
			); err != nil {
				turnRows.Close()
				return core.E("chathistory.ExportJSONL", "scan turn", err)
			}
			if toolCalls.Valid {
				t.ToolCalls = core.RawMessage(toolCalls.String)
			}
			if toolResults.Valid {
				t.ToolResults = core.RawMessage(toolResults.String)
			}
			if tokensIn.Valid {
				t.TokensIn = int(tokensIn.Int32)
			}
			if tokensOut.Valid {
				t.TokensOut = int(tokensOut.Int32)
			}
			t.Signal = signal.String
			c.Turns = append(c.Turns, t)
		}
		// turnRows.Next() returns false on both natural end-of-stream
		// AND iterator error. Without Err() a mid-stream DB blip
		// silently truncates a conversation's turn list inside an
		// otherwise-completed export — user gets a "successful" JSONL
		// missing turns from one record with no signal.
		if err := turnRows.Err(); err != nil {
			turnRows.Close()
			return core.E("chathistory.ExportJSONL", "turn rows", err)
		}
		turnRows.Close()

		marshalled := core.JSONMarshal(c)
		if !marshalled.OK {
			return core.E("chathistory.ExportJSONL", "marshal conversation", marshalled.Value.(error))
		}
		line := marshalled.Value.([]byte)
		if _, err := f.Write(line); err != nil {
			return core.E("chathistory.ExportJSONL", "write line", err)
		}
		if _, err := f.Write([]byte{'\n'}); err != nil {
			return core.E("chathistory.ExportJSONL", "write newline", err)
		}
	}
	// Same iterator-error trap on the outer convRows loop — without
	// this a mid-export DB blip silently produces a JSONL with the
	// LATER conversations missing entirely.
	if err := convRows.Err(); err != nil {
		return core.E("chathistory.ExportJSONL", "conversation rows", err)
	}
	// Explicit Close on the success path — surfaces flush failures
	// (disk-full, network drive, etc.) that would otherwise be
	// swallowed by the deferred Close above. The deferred Close
	// still runs but Close-on-closed-file is a no-op error we
	// ignore (the meaningful error already returned here).
	if err := f.Close(); err != nil {
		return core.E("chathistory.ExportJSONL", "close dest", err)
	}
	return nil
}
