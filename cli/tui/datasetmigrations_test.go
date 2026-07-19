// SPDX-License-Identifier: EUPL-1.2

package tui

import (
	"testing"
	"time"
)

func TestDatasetMigrations_Good(t *testing.T) {
	path := t.TempDir() + "/datasets.duckdb"
	opened := openDatasetDatabase(path)
	if !opened.OK {
		t.Fatalf("openDatasetDatabase(%q) failed: %v", path, opened.Value)
	}
	database, ok := opened.Value.(*datasetDatabase)
	if !ok {
		t.Fatalf("openDatasetDatabase value = %T, want *datasetDatabase", opened.Value)
	}
	defer func() {
		if result := database.Close(); !result.OK {
			t.Errorf("close dataset database: %v", result.Value)
		}
	}()

	var tables int
	if err := database.store.Conn().QueryRow(`
		SELECT COUNT(*)
		FROM information_schema.tables
		WHERE table_schema = 'main'
		  AND table_name IN (
			'dataset_schema_versions', 'dataset_datasets', 'dataset_items',
			'dataset_scores', 'dataset_reviews', 'dataset_exports'
		  )`).Scan(&tables); err != nil {
		t.Fatalf("count dataset tables: %v", err)
	}
	if tables != 6 {
		t.Fatalf("dataset tables = %d, want 6", tables)
	}

	var views int
	if err := database.store.Conn().QueryRow(`
		SELECT COUNT(*) FROM information_schema.tables
		WHERE table_schema = 'main' AND table_name IN ('dataset_latest_scores', 'dataset_latest_reviews')
	`).Scan(&views); err != nil {
		t.Fatalf("count dataset views: %v", err)
	}
	if views != 2 {
		t.Fatalf("dataset views = %d, want 2", views)
	}

	var versions int
	if err := database.store.Conn().QueryRow("SELECT COUNT(*) FROM dataset_schema_versions").Scan(&versions); err != nil {
		t.Fatalf("count schema versions: %v", err)
	}
	if versions != 1 {
		t.Fatalf("schema version rows = %d, want 1", versions)
	}
}

// TestDatasetMigrations_Bad proves a failing statement rolls back the
// whole migration — no partial table, no recorded version — mirroring
// TestMigrations_Bad's shape for the workspace migrations.
func TestDatasetMigrations_Bad(t *testing.T) {
	database := openTestDatasetStore(t, t.TempDir()+"/datasets.duckdb")
	defer closeTestDatasetStore(t, database)

	first := applyDatasetMigrations(database, datasetMigrations, time.Now)
	if !first.OK {
		t.Fatalf("apply base migrations: %v", first.Value)
	}
	bad := []datasetMigration{{
		Version: 2,
		Statements: []string{
			"CREATE TABLE dataset_rollback_probe (id BIGINT)",
			"THIS IS NOT VALID SQL",
		},
	}}
	result := applyDatasetMigrations(database, bad, time.Now)
	if result.OK {
		t.Fatalf("applyDatasetMigrations(invalid) = %#v, want failure", result.Value)
	}

	var versions int
	if err := database.Conn().QueryRow("SELECT COUNT(*) FROM dataset_schema_versions WHERE version = 2").Scan(&versions); err != nil {
		t.Fatalf("count failed schema version: %v", err)
	}
	if versions != 0 {
		t.Fatalf("failed schema version rows = %d, want 0", versions)
	}
	var probeTables int
	if err := database.Conn().QueryRow(`
		SELECT COUNT(*) FROM information_schema.tables
		WHERE table_schema = 'main' AND table_name = 'dataset_rollback_probe'`).Scan(&probeTables); err != nil {
		t.Fatalf("count rollback probe table: %v", err)
	}
	if probeTables != 0 {
		t.Fatalf("rollback probe tables = %d, want 0", probeTables)
	}
}

// TestDatasetMigrations_Ugly proves migration idempotence explicitly: the
// same path opened, migrated, closed, and reopened records version 1
// exactly once — a reopen must be a no-op, never a duplicate insert or a
// second CREATE TABLE failure.
func TestDatasetMigrations_Ugly(t *testing.T) {
	path := t.TempDir() + "/datasets.duckdb"

	first := openTestDatasetStore(t, path)
	if result := applyDatasetMigrations(first, datasetMigrations, time.Now); !result.OK {
		t.Fatalf("apply dataset migrations: %v", result.Value)
	}
	closeTestDatasetStore(t, first)

	reopened := openTestDatasetStore(t, path)
	defer closeTestDatasetStore(t, reopened)
	if result := applyDatasetMigrations(reopened, datasetMigrations, time.Now); !result.OK {
		t.Fatalf("reapply dataset migrations on reopened database: %v", result.Value)
	}

	var versions int
	if err := reopened.Conn().QueryRow("SELECT COUNT(*) FROM dataset_schema_versions").Scan(&versions); err != nil {
		t.Fatalf("count schema versions: %v", err)
	}
	if versions != 1 {
		t.Fatalf("schema version rows after reopen = %d, want 1 (migration must be idempotent)", versions)
	}
	var version int64
	if err := reopened.Conn().QueryRow("SELECT version FROM dataset_schema_versions LIMIT 1").Scan(&version); err != nil {
		t.Fatalf("read recorded version: %v", err)
	}
	if version != 1 {
		t.Fatalf("recorded version = %d, want 1", version)
	}

	// Re-opening through the full store constructor (not just the raw
	// migration function) must also be a no-op — the path a real caller
	// takes via newDuckDatasetStore.
	reopenedStore := newTestDuckDatasetStore(t, path)
	defer closeTestDuckDatasetStore(t, reopenedStore)
	var versionsViaStore int
	if err := reopenedStore.conn().QueryRow("SELECT COUNT(*) FROM dataset_schema_versions").Scan(&versionsViaStore); err != nil {
		t.Fatalf("count schema versions via store: %v", err)
	}
	if versionsViaStore != 1 {
		t.Fatalf("schema version rows via newDuckDatasetStore reopen = %d, want 1", versionsViaStore)
	}
}
