// SPDX-Licence-Identifier: EUPL-1.2

package tui

import (
	"time"

	core "dappco.re/go"
	"dappco.re/go/store"
)

// datasetMigration mirrors workspaceMigration's shape (see migrations.go)
// but versions its own dataset_schema_versions table — a separate
// sequence from the workspace migrations, because datasets.duckdb is a
// wholly separate file from lem.duckdb (the design's bulk/lifecycle/
// blast-radius rationale: a bloated or damaged dataset file must never
// take the agent/TUI state down with it).
type datasetMigration struct {
	Version    int64
	Statements []string
}

// datasetMigrations is versioned exactly like workspaceMigrations: each
// entry runs once, inside one transaction, recorded in
// dataset_schema_versions so a later reopen is a no-op.
//
// Two "latest per kind" views back the score-expression and review-status
// filters Items() compiles ([duckDatasetStore.Items]):
//   - dataset_latest_scores: one row per (item_id, kind) — the most
//     recent Score row by (created_at, seq), matching
//     [dataset.ScoreExpression.Matches]'s "latest value, ties broken by
//     later insertion" rule.
//   - dataset_latest_reviews: one row per item_id — the most recent
//     Review row by (created_at, seq), matching [dataset.Store]'s
//     "latest row wins" Review semantics.
//
// seq is a monotonically increasing column populated by the store (see
// duckDatasetStore.nextSeqLocked) purely to break created_at ties in
// insertion order — Review carries no ID in the domain model, and Score
// IDs are random UUIDs uncorrelated with insertion order, so created_at
// alone cannot reproduce dataset.MemoryStore's append-order tie-break.
//
// dataset_scores.id and dataset_exports.id are deliberately NOT primary
// keys, and seq (guaranteed unique — derived under duckDatasetStore.mu)
// carries dataset_scores' and dataset_reviews' uniqueness instead. This
// mirrors dataset.MemoryStore precisely: Store's interface doc promises
// "fails if ID already exists" only for DatasetCreate and ItemAppend —
// ScoreAppend and ExportAppend's docs say only "fails if
// DatasetID/ItemID does not exist", and MemoryStore's implementations
// confirm it (append.Scores/append.Exports have no id-collision check).
// A conformance run caught this: an early draft PRIMARY KEY'd both id
// columns and TestDatasetStoreConformance/MemoryStore/ExportLifecycle
// failed — MemoryStore silently accepts a re-appended Export id, so
// rejecting it here would have been a real divergence, not a safety
// improvement.
var datasetMigrations = []datasetMigration{
	{
		Version: 1,
		Statements: []string{
			"CREATE TABLE IF NOT EXISTS dataset_schema_versions (version BIGINT PRIMARY KEY, applied_at TIMESTAMP NOT NULL)",
			"CREATE TABLE dataset_datasets (id TEXT PRIMARY KEY, slug TEXT NOT NULL UNIQUE, title TEXT NOT NULL, purpose TEXT NOT NULL, created_at TIMESTAMP NOT NULL, archived BOOLEAN NOT NULL, archived_at TIMESTAMP NOT NULL)",
			"CREATE TABLE dataset_items (id TEXT PRIMARY KEY, dataset_id TEXT NOT NULL, kind TEXT NOT NULL, content BLOB NOT NULL, source TEXT NOT NULL, source_ref TEXT NOT NULL, model_fingerprint TEXT NOT NULL, content_hash TEXT NOT NULL, parent_item_id TEXT NOT NULL, archived BOOLEAN NOT NULL, archived_at TIMESTAMP NOT NULL, created_at TIMESTAMP NOT NULL)",
			"CREATE TABLE dataset_scores (id TEXT NOT NULL, item_id TEXT NOT NULL, kind TEXT NOT NULL, value DOUBLE NOT NULL, payload BLOB NOT NULL, scorer_name TEXT NOT NULL, scorer_version TEXT NOT NULL, judge_fingerprint TEXT NOT NULL, created_at TIMESTAMP NOT NULL, seq BIGINT PRIMARY KEY)",
			"CREATE TABLE dataset_reviews (item_id TEXT NOT NULL, status TEXT NOT NULL, reviewer TEXT NOT NULL, note TEXT NOT NULL, created_at TIMESTAMP NOT NULL, seq BIGINT PRIMARY KEY)",
			"CREATE TABLE dataset_exports (id TEXT NOT NULL, dataset_id TEXT NOT NULL, format TEXT NOT NULL, filter_description TEXT NOT NULL, item_count BIGINT NOT NULL, output_path TEXT NOT NULL, manifest_hash TEXT NOT NULL, created_at TIMESTAMP NOT NULL)",
			"CREATE INDEX IF NOT EXISTS dataset_datasets_archived_idx ON dataset_datasets(archived, created_at)",
			"CREATE INDEX IF NOT EXISTS dataset_items_dataset_idx ON dataset_items(dataset_id, archived, created_at)",
			"CREATE INDEX IF NOT EXISTS dataset_items_hash_idx ON dataset_items(dataset_id, content_hash)",
			"CREATE INDEX IF NOT EXISTS dataset_scores_item_idx ON dataset_scores(item_id, kind, created_at)",
			"CREATE INDEX IF NOT EXISTS dataset_reviews_item_idx ON dataset_reviews(item_id, created_at)",
			"CREATE INDEX IF NOT EXISTS dataset_exports_dataset_idx ON dataset_exports(dataset_id, created_at)",
			`CREATE VIEW dataset_latest_scores AS
				SELECT id, item_id, kind, value, payload, scorer_name, scorer_version, judge_fingerprint, created_at, seq
				FROM (
					SELECT *, ROW_NUMBER() OVER (PARTITION BY item_id, kind ORDER BY created_at DESC, seq DESC) AS rn
					FROM dataset_scores
				) ranked
				WHERE rn = 1`,
			`CREATE VIEW dataset_latest_reviews AS
				SELECT item_id, status, reviewer, note, created_at, seq
				FROM (
					SELECT *, ROW_NUMBER() OVER (PARTITION BY item_id ORDER BY created_at DESC, seq DESC) AS rn
					FROM dataset_reviews
				) ranked
				WHERE rn = 1`,
		},
	},
}

// datasetDatabase is the opened+migrated datasets.duckdb handle — the
// dataset-store analogue of workspaceDatabase, but scoped to its own
// file and its own migration sequence. It deliberately does not mount
// the orm medium/runtime: every duckDatasetStore query is hand-rolled
// SQL (see datasetstore.go), matching duckAgentStore's precedent for
// implementing an external Store contract, because the score-expression
// and latest-review-wins filters need window-function views the orm
// DuckDB medium cannot express (no subquery/alias support in its query
// builder).
type datasetDatabase struct {
	store *store.DuckDB
}

// openDatasetDatabase opens (creating if absent) the DuckDB file at path
// and applies datasetMigrations. path must be non-empty — callers resolve
// it from appPaths.Datasets (the medium-backed path contract: only this
// adapter receives the resolved filename).
func openDatasetDatabase(path string) core.Result {
	path = core.Trim(path)
	if path == "" {
		return core.Fail(core.E("tui.openDatasetDatabase", "database path is required", nil))
	}

	storeResult := store.OpenDuckDBReadWrite(path)
	if !storeResult.OK {
		return core.Fail(core.E("tui.openDatasetDatabase", core.Concat("open DuckDB: ", path), resultError(storeResult)))
	}
	databaseStore, ok := storeResult.Value.(*store.DuckDB)
	if !ok {
		return core.Fail(core.E("tui.openDatasetDatabase", "invalid go-store DuckDB result", nil))
	}
	if result := applyDatasetMigrations(databaseStore, datasetMigrations, time.Now); !result.OK {
		closeStoreAfterFailure(databaseStore)
		return core.Fail(core.E("tui.openDatasetDatabase", core.Concat("migrate DuckDB: ", path), resultError(result)))
	}

	return core.Ok(&datasetDatabase{store: databaseStore})
}

// applyDatasetMigrations runs migrations against database, skipping any
// version already recorded in dataset_schema_versions. Mirrors
// applyWorkspaceMigrations exactly (own transaction per version, full
// rollback on any failed statement) but targets dataset_schema_versions
// instead of lem_schema_versions — the two migration sequences never
// share a version table because they never share a file.
func applyDatasetMigrations(database *store.DuckDB, migrations []datasetMigration, now func() time.Time) core.Result {
	if database == nil || database.Conn() == nil {
		return core.Fail(core.E("tui.applyDatasetMigrations", "DuckDB connection is required", nil))
	}
	if now == nil {
		now = time.Now
	}

	for _, migration := range migrations {
		if migration.Version < 1 {
			return core.Fail(core.E("tui.applyDatasetMigrations", "migration version must be positive", nil))
		}
		applied, err := datasetMigrationApplied(database, migration.Version)
		if err != nil {
			return core.Fail(core.E(
				"tui.applyDatasetMigrations",
				core.Sprintf("read migration version %d", migration.Version),
				err,
			))
		}
		if applied {
			continue
		}

		transaction, err := database.Conn().Begin()
		if err != nil {
			return core.Fail(core.E(
				"tui.applyDatasetMigrations",
				core.Sprintf("begin migration version %d", migration.Version),
				err,
			))
		}
		for _, statement := range migration.Statements {
			if core.Trim(statement) == "" {
				rollbackWorkspaceMigration(transaction)
				return core.Fail(core.E(
					"tui.applyDatasetMigrations",
					core.Sprintf("migration version %d contains an empty statement", migration.Version),
					nil,
				))
			}
			if _, err := transaction.Exec(statement); err != nil {
				rollbackWorkspaceMigration(transaction)
				return core.Fail(core.E(
					"tui.applyDatasetMigrations",
					core.Sprintf("execute migration version %d", migration.Version),
					err,
				))
			}
		}
		if _, err := transaction.Exec(
			"INSERT INTO dataset_schema_versions (version, applied_at) VALUES (?, ?)",
			migration.Version,
			now().UTC(),
		); err != nil {
			rollbackWorkspaceMigration(transaction)
			return core.Fail(core.E(
				"tui.applyDatasetMigrations",
				core.Sprintf("record migration version %d", migration.Version),
				err,
			))
		}
		if err := transaction.Commit(); err != nil {
			rollbackWorkspaceMigration(transaction)
			return core.Fail(core.E(
				"tui.applyDatasetMigrations",
				core.Sprintf("commit migration version %d", migration.Version),
				err,
			))
		}
	}

	return core.Ok(nil)
}

func datasetMigrationApplied(database *store.DuckDB, version int64) (bool, error) {
	var tableCount int
	if err := database.Conn().QueryRow(`
		SELECT COUNT(*)
		FROM information_schema.tables
		WHERE table_schema = 'main' AND table_name = 'dataset_schema_versions'
	`).Scan(&tableCount); err != nil {
		return false, err
	}
	if tableCount == 0 {
		return false, nil
	}

	var versionCount int
	if err := database.Conn().QueryRow(
		"SELECT COUNT(*) FROM dataset_schema_versions WHERE version = ?",
		version,
	).Scan(&versionCount); err != nil {
		return false, err
	}
	return versionCount > 0, nil
}

// Close releases the underlying DuckDB connection. Safe to call on a nil
// receiver or an already-closed database.
func (database *datasetDatabase) Close() core.Result {
	if database == nil || database.store == nil {
		return core.Ok(nil)
	}
	result := database.store.Close()
	database.store = nil
	return result
}
