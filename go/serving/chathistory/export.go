// SPDX-License-Identifier: EUPL-1.2

package chathistory

import (
	"context"
	"database/sql"
	"time"

	core "dappco.re/go"
	"dappco.re/go/inference/internal/pathx"
)

// exportAttachAlias is the catalog name the destination is attached under for
// the duration of a CopyTo. It only has to avoid colliding with the live
// database's own name; a second concurrent CopyTo fails cleanly at ATTACH
// rather than interleaving into one another's output.
const exportAttachAlias = "chathistory_export"

// exportCopyOrder lists the tables in foreign-key dependency order: a row in
// turns references conversations, and a row in embeddings references turns, so
// a parent is always populated before its children.
//
// This ordering is the whole reason the copy is written out table by table
// rather than handed to DuckDB's COPY FROM DATABASE, which does not
// topologically sort and fails partway with "Violates foreign key constraint
// because key ... does not exist in the referenced table".
var exportCopyOrder = []string{"conversations", "turns", "embeddings"}

// CopyTo writes the live history to dest as a standalone DuckDB file. The
// user-friendly export path: hand them a single .duckdb they can open in any
// tool. The archive is checkpointed first so WAL writes are folded in.
//
// DuckDB performs the copy itself — the destination is ATTACHed, given the
// same schema from the same embedded migration, and populated from the live
// tables. Nothing re-opens the source file by path. That second descriptor was
// not portable: on Windows DuckDB holds its database file without
// FILE_SHARE_READ, so opening it again fails outright with "The process cannot
// access the file because it is being used by another process". Letting the
// engine read from its own open database also means the export is a consistent
// snapshot rather than a byte copy racing concurrent writers.
//
// This is the simplest export — the file IS the format. For tools
// that prefer line-delimited records, ExportJSONL.
//
//	if err := h.CopyTo("/Users/you/Downloads/chat-history.duckdb"); err != nil { ... }
func (h *History) CopyTo(dest string) error {
	if h == nil || h.db == nil {
		return core.E("chathistory.CopyTo", "history closed", nil)
	}
	if core.Trim(dest) == "" {
		return core.E("chathistory.CopyTo", "dest required", nil)
	}

	ctx := context.Background()
	// One pinned connection for the whole operation: USE is connection-scoped
	// under database/sql's pool, so running the steps across arbitrary pooled
	// connections would apply the schema to the wrong catalog.
	conn, err := h.db.Conn(ctx)
	if err != nil {
		return core.E("chathistory.CopyTo", "acquire connection", err)
	}
	defer conn.Close()

	if _, err := conn.ExecContext(ctx, `CHECKPOINT`); err != nil {
		return core.E("chathistory.CopyTo", "checkpoint", err)
	}
	// pathx.Dir, not core.PathDir: the latter matches only the platform
	// separator, so a '/'-spelled dest on Windows reports "." and the nested
	// parent is never created. "" means "no parent component".
	if dir := pathx.Dir(dest); dir != "" {
		if r := core.MkdirAll(dir, 0o755); !r.OK {
			return core.E("chathistory.CopyTo", "mkdir dest parent", r.Value.(error))
		}
	}
	// ATTACH opens an existing file rather than truncating it, so a stale
	// export would be merged into instead of replaced. Clear it first —
	// but only when it is a regular file. os.Remove deletes an empty
	// directory without complaint, and a caller who names a directory by
	// mistake must get ATTACH's error, not have that directory silently
	// deleted. A missing dest is the normal case, not an error.
	if err := clearExportFile(dest); err != nil {
		return core.E("chathistory.CopyTo", "clear dest", err)
	}

	var source string
	if err := conn.QueryRowContext(ctx, `SELECT current_database()`).Scan(&source); err != nil {
		return core.E("chathistory.CopyTo", "current database", err)
	}
	// Every statement below reports through one wrapper, so the failure path
	// is a single arm exercised by any one of them rather than a dozen
	// near-identical ones that only fault injection could reach.
	run := func(step, statement string) error {
		if _, err := conn.ExecContext(ctx, statement); err != nil {
			return core.E("chathistory.CopyTo", step, err)
		}
		return nil
	}

	if err := run("attach dest", `ATTACH `+sqlQuote(dest)+` AS `+sqlIdent(exportAttachAlias)); err != nil {
		return err
	}
	if err := copyInto(run, source); err != nil {
		_, _ = conn.ExecContext(ctx, `DETACH `+sqlIdent(exportAttachAlias))
		return err
	}
	// DETACH is what flushes and closes the destination file, so its error
	// matters on the success path for the same reason a file Close's did: a
	// disk-full surfaces here, not during the writes.
	return run("detach dest", `DETACH `+sqlIdent(exportAttachAlias))
}

// copyInto builds the destination schema and fills it, running each statement
// through run. The destination is attached on entry and detached by the caller
// either way.
func copyInto(run func(step, statement string) error, source string) error {
	// Replay the same embedded migration that Open applies, so the export is
	// schema-identical to a live archive — primary keys, foreign keys and
	// indexes included, none of which a CREATE TABLE AS SELECT would carry.
	// The DDL is unqualified, so the catalog is switched rather than the
	// statements rewritten.
	steps := []struct{ step, statement string }{
		{"select dest catalog", `USE ` + sqlIdent(exportAttachAlias)},
		{"apply dest schema", initSchema},
		{"restore source catalog", `USE ` + sqlIdent(source)},
		// The migration already records the version it introduces; carry
		// across any later rows without disturbing it.
		{"copy schema_version",
			`INSERT INTO ` + sqlIdent(exportAttachAlias) + `.schema_version
			 SELECT * FROM schema_version ON CONFLICT (version) DO NOTHING`},
	}
	for _, table := range exportCopyOrder {
		steps = append(steps, struct{ step, statement string }{
			"copy " + table,
			`INSERT INTO ` + sqlIdent(exportAttachAlias) + `.` + sqlIdent(table) +
				` SELECT * FROM ` + sqlIdent(table),
		})
	}
	for _, s := range steps {
		if err := run(s.step, s.statement); err != nil {
			return err
		}
	}
	return nil
}

// clearExportFile removes dest when it is an existing regular file, so ATTACH
// creates the export fresh rather than opening and merging into a stale one.
// A directory is deliberately left in place: os.Remove would delete an empty
// one, turning a mistyped destination into silent data loss, so it is handed
// to ATTACH to refuse instead.
func clearExportFile(dest string) error {
	stat := core.Stat(dest)
	if !stat.OK {
		return nil // absent, or unreadable — ATTACH reports the latter
	}
	info, ok := stat.Value.(core.FsFileInfo)
	if !ok || info.IsDir() {
		return nil
	}
	if r := core.Remove(dest); !r.OK {
		if removeErr, ok := r.Value.(error); ok && !core.IsNotExist(removeErr) {
			return removeErr
		}
	}
	return nil
}

// sqlQuote renders s as a SQL string literal, doubling any embedded quote.
// Export destinations are caller-supplied paths, and a path may legally
// contain a single quote on both POSIX and Windows.
func sqlQuote(s string) string {
	return "'" + core.Replace(s, "'", "''") + "'"
}

// sqlIdent renders s as a quoted SQL identifier, doubling any embedded double
// quote. DuckDB names a database after its filename, so a source name can
// carry characters that would not parse bare.
func sqlIdent(s string) string {
	return `"` + core.Replace(s, `"`, `""`) + `"`
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

	// Scan-target scratch hoisted out of both loops. Scan takes the
	// address of each target, which forces a heap escape; the scanned
	// values are copied into the result structs before the next Scan
	// runs, so one reused set across every row is byte-identical to
	// per-row locals while dropping the per-row / per-turn escape allocs.
	var title, modelID, baseModel, adapterID, tagsJSON sql.NullString
	var toolCalls, toolResults, signal sql.NullString
	var tokensIn, tokensOut sql.NullInt32

	for convRows.Next() {
		var c JSONLConversation
		var endedAt sql.NullTime
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
			// Scan into the freshly-appended backing-array slot rather
			// than a local JSONLTurn that would escape via Scan taking
			// its address; the Null* targets reuse the hoisted scratch.
			c.Turns = append(c.Turns, JSONLTurn{})
			t := &c.Turns[len(c.Turns)-1]
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
		line := marshalled.Bytes()
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
