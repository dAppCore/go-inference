// SPDX-License-Identifier: EUPL-1.2

package tui

import (
	"testing"
	"time"

	"dappco.re/go/orm"
	"dappco.re/go/store"
)

func TestMigrations_Good(t *testing.T) {
	path := t.TempDir() + "/lem.duckdb"
	opened := openWorkspaceDatabase(path)
	if !opened.OK {
		t.Fatalf("openWorkspaceDatabase(%q) failed: %v", path, opened.Value)
	}
	database, ok := opened.Value.(*workspaceDatabase)
	if !ok {
		t.Fatalf("openWorkspaceDatabase value = %T, want *workspaceDatabase", opened.Value)
	}
	defer func() {
		if result := database.Close(); !result.OK {
			t.Errorf("close workspace database: %v", result.Value)
		}
	}()

	second := applyWorkspaceMigrations(database.store, workspaceMigrations, func() time.Time {
		return time.Date(2040, time.January, 2, 3, 4, 5, 0, time.UTC)
	})
	if !second.OK {
		t.Fatalf("second migration pass failed: %v", second.Value)
	}

	var versions int
	if err := database.store.Conn().QueryRow("SELECT COUNT(*) FROM lem_schema_versions").Scan(&versions); err != nil {
		t.Fatalf("count schema versions: %v", err)
	}
	if versions != 2 {
		t.Fatalf("schema version rows = %d, want 2", versions)
	}

	var tables int
	if err := database.store.Conn().QueryRow(`
		SELECT COUNT(*)
		FROM information_schema.tables
		WHERE table_schema = 'main'
		  AND table_name IN (
			'lem_schema_versions', 'lem_sessions', 'lem_turns', 'lem_events',
			'lem_generation_jobs', 'lem_work_items', 'lem_artifacts', 'lem_attachments'
		  )`).Scan(&tables); err != nil {
		t.Fatalf("count workspace tables: %v", err)
	}
	if tables != 8 {
		t.Fatalf("workspace tables = %d, want 8", tables)
	}

	for _, table := range []string{
		"lem_schema_versions", "lem_sessions", "lem_turns", "lem_events",
		"lem_generation_jobs", "lem_work_items", "lem_artifacts", "lem_attachments",
	} {
		if result := orm.OfTable(database.runtime, table).Count(); !result.OK {
			t.Errorf("ORM schema %q was not registered: %v", table, result.Value)
		}
	}
}

func TestMigrations_Bad(t *testing.T) {
	database := openTestMigrationStore(t, t.TempDir()+"/lem.duckdb")
	defer closeTestMigrationStore(t, database)

	first := applyWorkspaceMigrations(database, workspaceMigrations, time.Now)
	if !first.OK {
		t.Fatalf("apply base migrations: %v", first.Value)
	}
	bad := []workspaceMigration{{
		Version: 3,
		Statements: []string{
			"CREATE TABLE lem_rollback_probe (id BIGINT)",
			"THIS IS NOT VALID SQL",
		},
	}}
	result := applyWorkspaceMigrations(database, bad, time.Now)
	if result.OK {
		t.Fatalf("applyWorkspaceMigrations(invalid) = %#v, want failure", result.Value)
	}

	var versions int
	if err := database.Conn().QueryRow("SELECT COUNT(*) FROM lem_schema_versions WHERE version = 3").Scan(&versions); err != nil {
		t.Fatalf("count failed schema version: %v", err)
	}
	if versions != 0 {
		t.Fatalf("failed schema version rows = %d, want 0", versions)
	}
	var probeTables int
	if err := database.Conn().QueryRow(`
		SELECT COUNT(*) FROM information_schema.tables
		WHERE table_schema = 'main' AND table_name = 'lem_rollback_probe'`).Scan(&probeTables); err != nil {
		t.Fatalf("count rollback probe table: %v", err)
	}
	if probeTables != 0 {
		t.Fatalf("rollback probe tables = %d, want 0", probeTables)
	}
}

func TestMigrations_Ugly(t *testing.T) {
	database := openTestMigrationStore(t, t.TempDir()+"/lem.duckdb")
	defer closeTestMigrationStore(t, database)

	appliedAt := time.Date(2020, time.February, 3, 4, 5, 6, 0, time.UTC)
	if result := database.Exec(`CREATE TABLE lem_schema_versions (
		version BIGINT PRIMARY KEY,
		applied_at TIMESTAMP NOT NULL
	)`); !result.OK {
		t.Fatalf("create schema version fixture: %v", result.Value)
	}
	if result := database.Exec(
		"INSERT INTO lem_schema_versions (version, applied_at) VALUES (?, ?)",
		int64(1), appliedAt,
	); !result.OK {
		t.Fatalf("insert schema version fixture: %v", result.Value)
	}

	result := applyWorkspaceMigrations(database, workspaceMigrations, func() time.Time {
		return time.Date(2050, time.December, 30, 1, 2, 3, 0, time.UTC)
	})
	if !result.OK {
		t.Fatalf("applyWorkspaceMigrations(existing version): %v", result.Value)
	}
	var got time.Time
	if err := database.Conn().QueryRow(
		"SELECT applied_at FROM lem_schema_versions WHERE version = 1",
	).Scan(&got); err != nil {
		t.Fatalf("read preserved schema version: %v", err)
	}
	if !got.Equal(appliedAt) {
		t.Fatalf("existing applied_at = %v, want %v", got, appliedAt)
	}
}

func TestMigrations_AgentGood(t *testing.T) {
	path := t.TempDir() + "/lem.duckdb"
	first := openTestMigrationStore(t, path)
	if result := applyWorkspaceMigrations(first, workspaceMigrations, time.Now); !result.OK {
		t.Fatalf("apply agent migrations: %v", result.Value)
	}
	if result := applyWorkspaceMigrations(first, workspaceMigrations, time.Now); !result.OK {
		t.Fatalf("reapply agent migrations: %v", result.Value)
	}
	var versions int
	if err := first.Conn().QueryRow("SELECT COUNT(*) FROM lem_schema_versions").Scan(&versions); err != nil {
		t.Fatalf("count schema versions: %v", err)
	}
	if versions != 2 {
		t.Fatalf("schema versions = %d, want 2", versions)
	}
	var tables int
	if err := first.Conn().QueryRow(`
		SELECT COUNT(*) FROM information_schema.tables
		WHERE table_schema = 'main' AND table_name IN (
			'agent_projects', 'agent_runs', 'agent_events', 'agent_log_chunks',
			'agent_questions', 'agent_answers', 'agent_acceptances',
			'agent_queue_state', 'agent_provider_state'
		)`).Scan(&tables); err != nil {
		t.Fatalf("count agent tables: %v", err)
	}
	if tables != 9 {
		t.Fatalf("agent tables = %d, want 9", tables)
	}
	var indexes int
	if err := first.Conn().QueryRow(`SELECT COUNT(*) FROM duckdb_indexes()
		WHERE index_name IN ('agent_runs_work_number_attempt_idx', 'agent_runs_work_idx',
		'agent_runs_status_idx', 'agent_events_run_idx', 'agent_acceptances_work_idx')`).Scan(&indexes); err != nil {
		t.Fatalf("count agent indexes: %v", err)
	}
	if indexes != 5 {
		t.Fatalf("agent indexes = %d, want 5", indexes)
	}
	closeTestMigrationStore(t, first)

	reopened := openTestMigrationStore(t, path)
	defer closeTestMigrationStore(t, reopened)
	if result := applyWorkspaceMigrations(reopened, workspaceMigrations, time.Now); !result.OK {
		t.Fatalf("migrate reopened database: %v", result.Value)
	}
	if result := reopened.Exec(
		"INSERT INTO agent_queue_state (id, status, reason, updated_at) VALUES (?, ?, ?, ?)",
		"default", "paused", "invalid", time.Now().UTC(),
	); result.OK {
		t.Fatal("agent_queue_state accepted status outside its check constraint")
	}
}

func TestMigrations_AgentBad(t *testing.T) {
	database := openTestMigrationStore(t, t.TempDir()+"/lem.duckdb")
	defer closeTestMigrationStore(t, database)
	if result := applyWorkspaceMigrations(database, workspaceMigrations[:1], time.Now); !result.OK {
		t.Fatalf("apply version 1: %v", result.Value)
	}
	broken := []workspaceMigration{{Version: 2, Statements: []string{
		"CREATE TABLE agent_rollback_probe (id BIGINT)",
		"CREATE TABLE agent_broken (",
	}}}
	if result := applyWorkspaceMigrations(database, broken, time.Now); result.OK {
		t.Fatal("broken agent migration succeeded")
	}
	var tables int
	if err := database.Conn().QueryRow(`SELECT COUNT(*) FROM information_schema.tables
		WHERE table_schema = 'main' AND table_name = 'agent_rollback_probe'`).Scan(&tables); err != nil {
		t.Fatalf("count rollback probe: %v", err)
	}
	if tables != 0 {
		t.Fatalf("agent rollback probe tables = %d, want 0", tables)
	}
	var versions int
	if err := database.Conn().QueryRow("SELECT COUNT(*) FROM lem_schema_versions WHERE version = 2").Scan(&versions); err != nil {
		t.Fatalf("count version 2: %v", err)
	}
	if versions != 0 {
		t.Fatalf("failed version rows = %d, want 0", versions)
	}
}

func TestMigrations_AgentUgly(t *testing.T) {
	path := t.TempDir() + "/lem.duckdb"
	database := openTestMigrationStore(t, path)
	if result := applyWorkspaceMigrations(database, workspaceMigrations[:1], time.Now); !result.OK {
		t.Fatalf("apply version 1 fixture: %v", result.Value)
	}
	closeTestMigrationStore(t, database)

	database = openTestMigrationStore(t, path)
	defer closeTestMigrationStore(t, database)
	if result := applyWorkspaceMigrations(database, workspaceMigrations, time.Now); !result.OK {
		t.Fatalf("upgrade version 1 database: %v", result.Value)
	}
	var durableDefault string
	if result := database.Exec(`INSERT INTO agent_runs (
		id, work_id, project_id, parent_run_id, provider, model, source_revision,
		execution_revision, accepted_revision, branch, worktree, command_receipt,
		run_number, attempt, process_id, status, exit_code, failure_reason,
		queued_at, started_at, finished_at, updated_at
	) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
		"legacy", "work", "project", "", "codex", "", "source", "", "", "branch", "tree", "",
		1, 1, 0, "queued", 0, "", time.Now(), time.Time{}, time.Time{}, time.Now(),
	); !result.OK {
		t.Fatalf("insert legacy-shaped run: %v", result.Value)
	}
	if err := database.Conn().QueryRow("SELECT durable_revision FROM agent_runs WHERE id = ?", "legacy").Scan(&durableDefault); err != nil {
		t.Fatalf("read durable revision default: %v", err)
	}
	if durableDefault != "" {
		t.Fatalf("legacy durable revision = %q, want empty", durableDefault)
	}
}

func openTestMigrationStore(t *testing.T, path string) *store.DuckDB {
	t.Helper()
	result := store.OpenDuckDBReadWrite(path)
	if !result.OK {
		t.Fatalf("open DuckDB %q: %v", path, result.Value)
	}
	database, ok := result.Value.(*store.DuckDB)
	if !ok {
		t.Fatalf("OpenDuckDBReadWrite value = %T, want *store.DuckDB", result.Value)
	}
	return database
}

func closeTestMigrationStore(t *testing.T, database *store.DuckDB) {
	t.Helper()
	if result := database.Close(); !result.OK {
		t.Errorf("close DuckDB: %v", result.Value)
	}
}
