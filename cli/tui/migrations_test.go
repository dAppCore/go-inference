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
	if versions != 1 {
		t.Fatalf("schema version rows = %d, want 1", versions)
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
		Version: 2,
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
	if err := database.Conn().QueryRow("SELECT COUNT(*) FROM lem_schema_versions WHERE version = 2").Scan(&versions); err != nil {
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
