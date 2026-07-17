// SPDX-License-Identifier: EUPL-1.2

// Package chathistory captures per-user agent conversations into a
// portable DuckDB file. The file is the user's property — exportable,
// copyable, usable in any DuckDB-aware tool. Continuity-rights design
// per project_chat_continuity_rights_normal_user_pattern: no provider
// pivot, model deprecation, or service sunset can take the user's
// chat friend away, because they have the file.
//
// The schema is intentionally relational (not key-value) because the
// future LoRA training data prep needs (user, assistant) pairs joined
// across turns, filtered by signal + consent_version. The optional
// embeddings sidecar is present in the schema from v1 so any future
// semantic-search tooling can rely on it; it's populated only when
// an embedding model is wired.
//
// Storage convention: one .duckdb per user, conventionally at
//
//	~/Lethean/lem/users/<user_id>/chats.duckdb
//
// Open accepts an explicit path so test/dev contexts can override
// without environment ceremony.
//
// Mirrors core/agent/go/pkg/chathistory; per-binary copies for now,
// extract to shared module when drift proves shared need.
//
// Usage example:
//
//	h, err := chathistory.Open("snider", "/Users/snider/Lethean/lem/users/snider/chats.duckdb")
//	if err != nil { return err }
//	defer h.Close()
//
//	convID, err := h.StartConversation(chathistory.NewConversation{
//	    ModelID:    "lemer-lite",
//	    BaseModel:  "gemma-4-e2b-it-4bit",
//	    Title:      "evening vent",
//	    Tags:       []string{"life"},
//	})
//	_ = h.WriteTurn(convID, chathistory.NewTurn{Role: "user",      Content: "hey lemma"})
//	_ = h.WriteTurn(convID, chathistory.NewTurn{Role: "assistant", Content: "hey, what's up?"})
//	_ = h.EndConversation(convID)
package chathistory

import (
	"database/sql"
	_ "embed"
	"time"

	core "dappco.re/go"
	"github.com/google/uuid"

	// duckdb driver registers itself with database/sql via init().
	// The official DuckDB-hosted binding (github.com/duckdb/duckdb-go), the
	// successor to marcboeker/go-duckdb — one binding across the whole binary
	// so two DuckDB statics never link in (CGo duplicate-symbol errors), and
	// its mapping package compiles under CGO_ENABLED=0 (the CPU release lane).
	_ "github.com/duckdb/duckdb-go/v2"
)

//go:embed migrations/001_init.sql
var initSchema string

// History is a handle on a single user's portable chat archive.
// Safe for concurrent use — DuckDB's database/sql driver handles
// connection pooling. Close releases the underlying file lock.
type History struct {
	userID string
	path   string
	db     *sql.DB
}

// NewConversation captures the metadata needed to start tracking a
// fresh conversation. ModelID is the wire model name as it appears in
// the inference API; BaseModel is the weights identifier (HF id or
// local path) used for future training data prep. AdapterID is the
// LoRA adapter applied on top of BaseModel, or empty if none.
type NewConversation struct {
	Title          string
	ModelID        string
	BaseModel      string
	AdapterID      string
	Tags           []string
	Metadata       []byte // JSON; agent-extensible
	ConsentVersion int    // 0 means "use default 1"; explicit value persists for future revocation
}

// NewTurn captures a single message landing in a conversation. Role
// is "user" / "assistant" / "system" / "tool". For assistant turns
// that called tools, set ToolCalls (JSON-encoded). For tool turns
// (the result of a tool call), set ToolResults. Tokens fields are
// optional but useful for training cost attribution.
type NewTurn struct {
	Role        string
	Content     string
	ToolCalls   []byte // JSON
	ToolResults []byte // JSON
	TokensIn    int
	TokensOut   int
}

// Open returns a History handle for the user, creating the file +
// applying the initial schema if it doesn't already exist. The
// caller owns the lifecycle and must Close when done.
//
//	h, err := chathistory.Open("snider", "/Users/snider/Lethean/lem/users/snider/chats.duckdb")
func Open(userID, path string) (*History, error) {
	if core.Trim(userID) == "" {
		return nil, core.E("chathistory.Open", "user id required", nil)
	}
	if core.Trim(path) == "" {
		return nil, core.E("chathistory.Open", "path required", nil)
	}
	if dir := core.PathDir(path); dir != "" {
		if r := core.MkdirAll(dir, 0o755); !r.OK {
			return nil, core.E("chathistory.Open", "mkdir parent", r.Value.(error))
		}
	}
	db, err := sql.Open("duckdb", path)
	if err != nil {
		return nil, core.E("chathistory.Open", "open duckdb", err)
	}
	if _, err := db.Exec(initSchema); err != nil {
		_ = db.Close()
		return nil, core.E("chathistory.Open", "apply schema", err)
	}
	return &History{userID: userID, path: path, db: db}, nil
}

// Close releases the file lock. Subsequent calls on this handle return errors.
func (h *History) Close() error {
	if h == nil || h.db == nil {
		return nil
	}
	return h.db.Close()
}

// Path returns the on-disk path. Useful for export / display.
func (h *History) Path() string { return h.path }

// UserID returns the user id this archive belongs to.
func (h *History) UserID() string { return h.userID }

// StartConversation creates a conversations row and returns its UUID.
// The conversation stays open (ended_at = NULL) until EndConversation
// is called, so a crashed agent leaves the conversation recoverable.
func (h *History) StartConversation(c NewConversation) (string, error) {
	if h == nil || h.db == nil {
		return "", core.E("chathistory.StartConversation", "history closed", nil)
	}
	id := uuid.NewString()
	consent := c.ConsentVersion
	if consent == 0 {
		consent = 1
	}
	var tags any
	if len(c.Tags) > 0 {
		marshalled := core.JSONMarshal(c.Tags)
		if !marshalled.OK {
			return "", core.E("chathistory.StartConversation", "marshal tags", marshalled.Value.(error))
		}
		tags = string(marshalled.Value.([]byte))
	}
	var metadata any
	if len(c.Metadata) > 0 {
		metadata = string(c.Metadata)
	}
	_, err := h.db.Exec(
		`INSERT INTO conversations
		    (id, user_id, title, started_at, model_id, base_model, adapter_id, tags, metadata, consent_version)
		 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
		id, h.userID, nullableText(c.Title), time.Now().UTC(),
		nullableText(c.ModelID), nullableText(c.BaseModel), nullableText(c.AdapterID),
		tags, metadata, consent,
	)
	if err != nil {
		return "", core.E("chathistory.StartConversation", "insert", err)
	}
	return id, nil
}

// WriteTurn appends a turn to the conversation. Ordinal is computed
// automatically as the next position after the highest existing turn
// in the conversation, so callers don't have to track it.
func (h *History) WriteTurn(conversationID string, t NewTurn) (string, error) {
	if h == nil || h.db == nil {
		return "", core.E("chathistory.WriteTurn", "history closed", nil)
	}
	if core.Trim(conversationID) == "" {
		return "", core.E("chathistory.WriteTurn", "conversation id required", nil)
	}
	if core.Trim(t.Role) == "" {
		return "", core.E("chathistory.WriteTurn", "role required", nil)
	}
	var nextOrdinal int
	row := h.db.QueryRow(
		`SELECT COALESCE(MAX(ordinal), -1) + 1 FROM turns WHERE conversation_id = ?`,
		conversationID,
	)
	if err := row.Scan(&nextOrdinal); err != nil {
		return "", core.E("chathistory.WriteTurn", "ordinal lookup", err)
	}
	id := uuid.NewString()
	_, err := h.db.Exec(
		`INSERT INTO turns
		    (id, conversation_id, ordinal, role, content, tool_calls, tool_results,
		     created_at, tokens_in, tokens_out)
		 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
		id, conversationID, nextOrdinal, t.Role, t.Content,
		nullableJSON(t.ToolCalls), nullableJSON(t.ToolResults),
		time.Now().UTC(),
		nullableInt(t.TokensIn), nullableInt(t.TokensOut),
	)
	if err != nil {
		return "", core.E("chathistory.WriteTurn", "insert", err)
	}
	return id, nil
}

// EndConversation marks the conversation as closed (ended_at = now).
// Idempotent — calling twice is harmless.
func (h *History) EndConversation(conversationID string) error {
	if h == nil || h.db == nil {
		return core.E("chathistory.EndConversation", "history closed", nil)
	}
	_, err := h.db.Exec(
		`UPDATE conversations SET ended_at = ? WHERE id = ? AND ended_at IS NULL`,
		time.Now().UTC(), conversationID,
	)
	if err != nil {
		return core.E("chathistory.EndConversation", "update", err)
	}
	return nil
}

// SetSignal records a curation signal on a turn — "continued",
// "retried", "ended", "liked", "disliked", or any caller-defined
// value. Used later by training data prep to filter quality.
func (h *History) SetSignal(turnID, signal string) error {
	if h == nil || h.db == nil {
		return core.E("chathistory.SetSignal", "history closed", nil)
	}
	_, err := h.db.Exec(`UPDATE turns SET signal = ? WHERE id = ?`, signal, turnID)
	if err != nil {
		return core.E("chathistory.SetSignal", "update", err)
	}
	return nil
}

// CountConversations returns how many conversations the archive holds.
// Useful for export summaries and progress reporting.
func (h *History) CountConversations() (int, error) {
	if h == nil || h.db == nil {
		return 0, core.E("chathistory.CountConversations", "history closed", nil)
	}
	var n int
	if err := h.db.QueryRow(`SELECT COUNT(*) FROM conversations`).Scan(&n); err != nil {
		return 0, core.E("chathistory.CountConversations", "query", err)
	}
	return n, nil
}

// Turn is one row from the turns table, in ordinal order. The shape
// is what consumers replaying conversation context need — role +
// content + ordinal — not the full row schema (no token counts /
// signal here; that detail lives in the archive for later use).
type Turn struct {
	Role    string
	Content string
	Ordinal int
}

// ConversationSummary is one row of RecentConversations — enough for a
// client to offer "pick up where you left off" without loading turns.
type ConversationSummary struct {
	ID        string
	Title     string
	StartedAt time.Time
	ModelID   string
}

// RecentConversations lists the user's conversations newest-first.
// lem-runtime extension to the per-binary copy (the GUI's restore verb);
// fold back into the siblings when drift proves shared need.
//
//	recents, err := h.RecentConversations(1)
//	if len(recents) > 0 { turns, _ := h.LoadTurns(recents[0].ID) }
func (h *History) RecentConversations(limit int) ([]ConversationSummary, error) {
	if h == nil || h.db == nil {
		return nil, core.E("chathistory.RecentConversations", "history closed", nil)
	}
	if limit <= 0 {
		limit = 10
	}
	rows, err := h.db.Query(
		`SELECT id, COALESCE(title, ''), started_at, COALESCE(model_id, '')
		   FROM conversations WHERE user_id = ?
		  ORDER BY started_at DESC LIMIT ?`, h.userID, limit)
	if err != nil {
		return nil, core.E("chathistory.RecentConversations", "query failed", err)
	}
	defer rows.Close()
	var out []ConversationSummary
	for rows.Next() {
		// Scan straight into out's backing array — a local
		// `var c ConversationSummary` would escape to the heap every
		// row because Scan takes the address of its fields.
		out = append(out, ConversationSummary{})
		c := &out[len(out)-1]
		if err := rows.Scan(&c.ID, &c.Title, &c.StartedAt, &c.ModelID); err != nil {
			return nil, core.E("chathistory.RecentConversations", "scan failed", err)
		}
	}
	return out, rows.Err()
}

// LoadTurns returns every turn in the conversation in ordinal order.
// Used by user-chat clients (pkg/lemma) to replay context into the
// next model call without holding a separate in-memory copy that
// could drift from what's persisted.
//
//	turns, err := h.LoadTurns(convID)
func (h *History) LoadTurns(conversationID string) ([]Turn, error) {
	if h == nil || h.db == nil {
		return nil, core.E("chathistory.LoadTurns", "history closed", nil)
	}
	if core.Trim(conversationID) == "" {
		return nil, core.E("chathistory.LoadTurns", "conversation id required", nil)
	}
	rows, err := h.db.Query(
		`SELECT role, content, ordinal FROM turns WHERE conversation_id = ? ORDER BY ordinal`,
		conversationID,
	)
	if err != nil {
		return nil, core.E("chathistory.LoadTurns", "query", err)
	}
	defer rows.Close()
	var out []Turn
	for rows.Next() {
		// Scan straight into the slot in out's backing array. A local
		// `var t Turn` would escape to the heap every row because Scan
		// takes the address of its fields; appending the already-scanned
		// slot avoids that per-row allocation.
		out = append(out, Turn{})
		t := &out[len(out)-1]
		if err := rows.Scan(&t.Role, &t.Content, &t.Ordinal); err != nil {
			return nil, core.E("chathistory.LoadTurns", "scan", err)
		}
	}
	// rows.Next() returns false on both natural end-of-stream AND
	// iterator error; Err() distinguishes. Without this check a
	// mid-stream DB blip silently returns a truncated turn list
	// and the chat view re-renders missing the latter messages.
	if err := rows.Err(); err != nil {
		return nil, core.E("chathistory.LoadTurns", "rows", err)
	}
	return out, nil
}

// CountTurns returns the total number of turns across all conversations.
func (h *History) CountTurns() (int, error) {
	if h == nil || h.db == nil {
		return 0, core.E("chathistory.CountTurns", "history closed", nil)
	}
	var n int
	if err := h.db.QueryRow(`SELECT COUNT(*) FROM turns`).Scan(&n); err != nil {
		return 0, core.E("chathistory.CountTurns", "query", err)
	}
	return n, nil
}

// nullableText converts an empty string to a SQL NULL value so the
// column reads as NULL rather than the empty string. Matters for
// downstream queries that filter on `IS NOT NULL`.
func nullableText(s string) any {
	if core.Trim(s) == "" {
		return nil
	}
	return s
}

// nullableJSON returns a string for non-empty JSON bytes, nil for empty.
func nullableJSON(b []byte) any {
	if len(b) == 0 {
		return nil
	}
	return string(b)
}

// nullableInt returns the int for positive values, nil for zero.
// Treats zero as "not measured" because token counts are always > 0
// for a non-empty turn.
func nullableInt(n int) any {
	if n <= 0 {
		return nil
	}
	return n
}
