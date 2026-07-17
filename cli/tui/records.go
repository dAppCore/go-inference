// SPDX-License-Identifier: EUPL-1.2

package tui

import (
	"time"

	"dappco.re/go/orm"
	"github.com/google/uuid"
)

type schemaVersionRecord struct {
	Version   int64     `json:"version"`
	AppliedAt time.Time `json:"applied_at"`
}

func (schemaVersionRecord) Schema() orm.Schema {
	return orm.Define(func(builder *orm.Builder) {
		builder.Name("lem_schema_versions")
		builder.PK("version")
		builder.Int64("version").NotNull()
		builder.Time("applied_at").NotNull()
	})
}

type sessionRecord struct {
	ID             string    `json:"id"`
	Title          string    `json:"title"`
	Status         string    `json:"status"`
	PreferredModel string    `json:"preferred_model"`
	Mode           string    `json:"mode"`
	GenerationJSON string    `json:"generation_json"`
	ToolsJSON      string    `json:"tools_json"`
	CreatedAt      time.Time `json:"created_at"`
	UpdatedAt      time.Time `json:"updated_at"`
	LastOpenedAt   time.Time `json:"last_opened_at"`
	Archived       bool      `json:"archived"`
	ArchivedAt     time.Time `json:"archived_at"`
}

func (sessionRecord) Schema() orm.Schema {
	return orm.Define(func(builder *orm.Builder) {
		builder.Name("lem_sessions")
		builder.PK("id")
		builder.String("id").NotNull()
		builder.String("title").NotNull()
		builder.String("status").NotNull()
		builder.String("preferred_model").NotNull()
		builder.String("mode").NotNull()
		builder.String("generation_json").NotNull()
		builder.String("tools_json").NotNull()
		builder.Time("created_at").NotNull()
		builder.Time("updated_at").NotNull()
		builder.Time("last_opened_at").NotNull()
		builder.Bool("archived").NotNull()
		builder.Time("archived_at").NotNull()
		builder.Index("archived", "last_opened_at")
	})
}

type turnRecord struct {
	ID             string    `json:"id"`
	SessionID      string    `json:"session_id"`
	Sequence       int64     `json:"sequence"`
	Role           string    `json:"role"`
	Visible        string    `json:"visible"`
	Thought        string    `json:"thought"`
	ToolName       string    `json:"tool_name"`
	ToolCallJSON   string    `json:"tool_call_json"`
	ToolResultJSON string    `json:"tool_result_json"`
	Model          string    `json:"model"`
	CreatedAt      time.Time `json:"created_at"`
	UpdatedAt      time.Time `json:"updated_at"`
}

func (turnRecord) Schema() orm.Schema {
	return orm.Define(func(builder *orm.Builder) {
		builder.Name("lem_turns")
		builder.PK("id")
		builder.String("id").NotNull()
		builder.String("session_id").NotNull()
		builder.Int64("sequence").NotNull()
		builder.String("role").NotNull()
		builder.String("visible").NotNull()
		builder.String("thought").NotNull()
		builder.String("tool_name").NotNull()
		builder.String("tool_call_json").NotNull()
		builder.String("tool_result_json").NotNull()
		builder.String("model").NotNull()
		builder.Time("created_at").NotNull()
		builder.Time("updated_at").NotNull()
		builder.Index("session_id", "sequence")
	})
}

type eventRecord struct {
	ID          string    `json:"id"`
	SessionID   string    `json:"session_id"`
	WorkItemID  string    `json:"work_item_id"`
	JobID       string    `json:"job_id"`
	Kind        string    `json:"kind"`
	Status      string    `json:"status"`
	Title       string    `json:"title"`
	Detail      string    `json:"detail"`
	PayloadJSON string    `json:"payload_json"`
	CreatedAt   time.Time `json:"created_at"`
}

func (eventRecord) Schema() orm.Schema {
	return orm.Define(func(builder *orm.Builder) {
		builder.Name("lem_events")
		builder.PK("id")
		builder.String("id").NotNull()
		builder.String("session_id").NotNull()
		builder.String("work_item_id").NotNull()
		builder.String("job_id").NotNull()
		builder.String("kind").NotNull()
		builder.String("status").NotNull()
		builder.String("title").NotNull()
		builder.String("detail").NotNull()
		builder.String("payload_json").NotNull()
		builder.Time("created_at").NotNull()
		builder.Index("session_id", "created_at")
	})
}

type generationJobRecord struct {
	ID           string    `json:"id"`
	SessionID    string    `json:"session_id"`
	PromptTurnID string    `json:"prompt_turn_id"`
	AnswerTurnID string    `json:"answer_turn_id"`
	Status       string    `json:"status"`
	Model        string    `json:"model"`
	Error        string    `json:"error"`
	MetricsJSON  string    `json:"metrics_json"`
	CreatedAt    time.Time `json:"created_at"`
	StartedAt    time.Time `json:"started_at"`
	FinishedAt   time.Time `json:"finished_at"`
}

func (generationJobRecord) Schema() orm.Schema {
	return orm.Define(func(builder *orm.Builder) {
		builder.Name("lem_generation_jobs")
		builder.PK("id")
		builder.String("id").NotNull()
		builder.String("session_id").NotNull()
		builder.String("prompt_turn_id").NotNull()
		builder.String("answer_turn_id").NotNull()
		builder.String("status").NotNull()
		builder.String("model").NotNull()
		builder.String("error").NotNull()
		builder.String("metrics_json").NotNull()
		builder.Time("created_at").NotNull()
		builder.Time("started_at").NotNull()
		builder.Time("finished_at").NotNull()
		builder.Index("session_id", "status", "created_at")
	})
}

type workItemRecord struct {
	ID         string    `json:"id"`
	ExternalID string    `json:"external_id"`
	Source     string    `json:"source"`
	Title      string    `json:"title"`
	Status     string    `json:"status"`
	Agent      string    `json:"agent"`
	Repo       string    `json:"repo"`
	Org        string    `json:"org"`
	Task       string    `json:"task"`
	Branch     string    `json:"branch"`
	Runtime    string    `json:"runtime"`
	Question   string    `json:"question"`
	PRURL      string    `json:"pr_url"`
	SessionID  string    `json:"session_id"`
	StartedAt  time.Time `json:"started_at"`
	UpdatedAt  time.Time `json:"updated_at"`
	Archived   bool      `json:"archived"`
	ArchivedAt time.Time `json:"archived_at"`
}

func (workItemRecord) Schema() orm.Schema {
	return orm.Define(func(builder *orm.Builder) {
		builder.Name("lem_work_items")
		builder.PK("id")
		builder.String("id").NotNull()
		builder.String("external_id").NotNull().Unique()
		builder.String("source").NotNull()
		builder.String("title").NotNull()
		builder.String("status").NotNull()
		builder.String("agent").NotNull()
		builder.String("repo").NotNull()
		builder.String("org").NotNull()
		builder.String("task").NotNull()
		builder.String("branch").NotNull()
		builder.String("runtime").NotNull()
		builder.String("question").NotNull()
		builder.String("pr_url").NotNull()
		builder.String("session_id").NotNull()
		builder.Time("started_at").NotNull()
		builder.Time("updated_at").NotNull()
		builder.Bool("archived").NotNull()
		builder.Time("archived_at").NotNull()
		builder.Index("archived", "status", "updated_at")
	})
}

type artifactRecord struct {
	ID           string    `json:"id"`
	SessionID    string    `json:"session_id"`
	WorkItemID   string    `json:"work_item_id"`
	Kind         string    `json:"kind"`
	Path         string    `json:"path"`
	Title        string    `json:"title"`
	MetadataJSON string    `json:"metadata_json"`
	CreatedAt    time.Time `json:"created_at"`
	Archived     bool      `json:"archived"`
	ArchivedAt   time.Time `json:"archived_at"`
}

func (artifactRecord) Schema() orm.Schema {
	return orm.Define(func(builder *orm.Builder) {
		builder.Name("lem_artifacts")
		builder.PK("id")
		builder.String("id").NotNull()
		builder.String("session_id").NotNull()
		builder.String("work_item_id").NotNull()
		builder.String("kind").NotNull()
		builder.String("path").NotNull()
		builder.String("title").NotNull()
		builder.String("metadata_json").NotNull()
		builder.Time("created_at").NotNull()
		builder.Bool("archived").NotNull()
		builder.Time("archived_at").NotNull()
		builder.Index("session_id", "created_at")
	})
}

type attachmentRecord struct {
	ID            string    `json:"id"`
	SessionID     string    `json:"session_id"`
	SourcePath    string    `json:"source_path"`
	Title         string    `json:"title"`
	ContentHash   string    `json:"content_hash"`
	Snapshot      string    `json:"snapshot"`
	AddedAt       time.Time `json:"added_at"`
	LastCheckedAt time.Time `json:"last_checked_at"`
	Stale         bool      `json:"stale"`
	Archived      bool      `json:"archived"`
	ArchivedAt    time.Time `json:"archived_at"`
}

func (attachmentRecord) Schema() orm.Schema {
	return orm.Define(func(builder *orm.Builder) {
		builder.Name("lem_attachments")
		builder.PK("id")
		builder.String("id").NotNull()
		builder.String("session_id").NotNull()
		builder.String("source_path").NotNull()
		builder.String("title").NotNull()
		builder.String("content_hash").NotNull()
		builder.String("snapshot").NotNull()
		builder.Time("added_at").NotNull()
		builder.Time("last_checked_at").NotNull()
		builder.Bool("stale").NotNull()
		builder.Bool("archived").NotNull()
		builder.Time("archived_at").NotNull()
		builder.Index("session_id", "archived", "added_at")
	})
}

func newRecordID() string {
	return uuid.NewString()
}

func unsetRecordTime() time.Time {
	return time.Unix(0, 0).UTC()
}

func workspaceRecordSchemas() []orm.Schema {
	return []orm.Schema{
		(schemaVersionRecord{}).Schema(),
		(sessionRecord{}).Schema(),
		(turnRecord{}).Schema(),
		(eventRecord{}).Schema(),
		(generationJobRecord{}).Schema(),
		(workItemRecord{}).Schema(),
		(artifactRecord{}).Schema(),
		(attachmentRecord{}).Schema(),
	}
}
