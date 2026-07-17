// SPDX-License-Identifier: EUPL-1.2

package tui

import (
	"database/sql"
	"time"

	core "dappco.re/go"
	"dappco.re/go/orm"
	"dappco.re/go/store"
)

type workspaceMigration struct {
	Version    int64
	Statements []string
}

var workspaceMigrations = []workspaceMigration{
	{
		Version: 1,
		Statements: []string{
			"CREATE TABLE IF NOT EXISTS lem_schema_versions (version BIGINT PRIMARY KEY, applied_at TIMESTAMP NOT NULL)",
			"CREATE TABLE IF NOT EXISTS lem_sessions (id TEXT PRIMARY KEY, title TEXT NOT NULL, status TEXT NOT NULL, preferred_model TEXT NOT NULL, mode TEXT NOT NULL, generation_json TEXT NOT NULL, tools_json TEXT NOT NULL, created_at TIMESTAMP NOT NULL, updated_at TIMESTAMP NOT NULL, last_opened_at TIMESTAMP NOT NULL, archived BOOLEAN NOT NULL, archived_at TIMESTAMP NOT NULL)",
			"CREATE TABLE IF NOT EXISTS lem_turns (id TEXT PRIMARY KEY, session_id TEXT NOT NULL, sequence BIGINT NOT NULL, role TEXT NOT NULL, visible TEXT NOT NULL, thought TEXT NOT NULL, tool_name TEXT NOT NULL, tool_call_json TEXT NOT NULL, tool_result_json TEXT NOT NULL, model TEXT NOT NULL, created_at TIMESTAMP NOT NULL, updated_at TIMESTAMP NOT NULL, UNIQUE(session_id, sequence))",
			"CREATE TABLE IF NOT EXISTS lem_events (id TEXT PRIMARY KEY, session_id TEXT NOT NULL, work_item_id TEXT NOT NULL, job_id TEXT NOT NULL, kind TEXT NOT NULL, status TEXT NOT NULL, title TEXT NOT NULL, detail TEXT NOT NULL, payload_json TEXT NOT NULL, created_at TIMESTAMP NOT NULL)",
			"CREATE TABLE IF NOT EXISTS lem_generation_jobs (id TEXT PRIMARY KEY, session_id TEXT NOT NULL, prompt_turn_id TEXT NOT NULL, answer_turn_id TEXT NOT NULL, status TEXT NOT NULL, model TEXT NOT NULL, error TEXT NOT NULL, metrics_json TEXT NOT NULL, created_at TIMESTAMP NOT NULL, started_at TIMESTAMP NOT NULL, finished_at TIMESTAMP NOT NULL)",
			"CREATE TABLE IF NOT EXISTS lem_work_items (id TEXT PRIMARY KEY, external_id TEXT NOT NULL UNIQUE, source TEXT NOT NULL, title TEXT NOT NULL, status TEXT NOT NULL, agent TEXT NOT NULL, repo TEXT NOT NULL, org TEXT NOT NULL, task TEXT NOT NULL, branch TEXT NOT NULL, runtime TEXT NOT NULL, question TEXT NOT NULL, pr_url TEXT NOT NULL, session_id TEXT NOT NULL, started_at TIMESTAMP NOT NULL, updated_at TIMESTAMP NOT NULL, archived BOOLEAN NOT NULL, archived_at TIMESTAMP NOT NULL)",
			"CREATE TABLE IF NOT EXISTS lem_artifacts (id TEXT PRIMARY KEY, session_id TEXT NOT NULL, work_item_id TEXT NOT NULL, kind TEXT NOT NULL, path TEXT NOT NULL, title TEXT NOT NULL, metadata_json TEXT NOT NULL, created_at TIMESTAMP NOT NULL, archived BOOLEAN NOT NULL, archived_at TIMESTAMP NOT NULL)",
			"CREATE TABLE IF NOT EXISTS lem_attachments (id TEXT PRIMARY KEY, session_id TEXT NOT NULL, source_path TEXT NOT NULL, title TEXT NOT NULL, content_hash TEXT NOT NULL, snapshot TEXT NOT NULL, added_at TIMESTAMP NOT NULL, last_checked_at TIMESTAMP NOT NULL, stale BOOLEAN NOT NULL, archived BOOLEAN NOT NULL, archived_at TIMESTAMP NOT NULL)",
			"CREATE INDEX IF NOT EXISTS lem_sessions_recent_idx ON lem_sessions(archived, last_opened_at)",
			"CREATE INDEX IF NOT EXISTS lem_turns_session_idx ON lem_turns(session_id, sequence)",
			"CREATE INDEX IF NOT EXISTS lem_events_session_idx ON lem_events(session_id, created_at)",
			"CREATE INDEX IF NOT EXISTS lem_jobs_session_idx ON lem_generation_jobs(session_id, status, created_at)",
			"CREATE INDEX IF NOT EXISTS lem_work_status_idx ON lem_work_items(archived, status, updated_at)",
			"CREATE INDEX IF NOT EXISTS lem_artifacts_session_idx ON lem_artifacts(session_id, created_at)",
			"CREATE INDEX IF NOT EXISTS lem_attachments_session_idx ON lem_attachments(session_id, archived, added_at)",
		},
	},
}

type workspaceDatabase struct {
	runtime *core.Core
	store   *store.DuckDB
	medium  *orm.DuckDBMedium
}

func openWorkspaceDatabase(path string) core.Result {
	path = core.Trim(path)
	if path == "" {
		return core.Fail(core.E("tui.openWorkspaceDatabase", "database path is required", nil))
	}

	storeResult := store.OpenDuckDBReadWrite(path)
	if !storeResult.OK {
		return core.Fail(core.E("tui.openWorkspaceDatabase", core.Concat("open DuckDB: ", path), resultError(storeResult)))
	}
	databaseStore, ok := storeResult.Value.(*store.DuckDB)
	if !ok {
		return core.Fail(core.E("tui.openWorkspaceDatabase", "invalid go-store DuckDB result", nil))
	}
	if result := applyWorkspaceMigrations(databaseStore, workspaceMigrations, time.Now); !result.OK {
		closeStoreAfterFailure(databaseStore)
		return core.Fail(core.E("tui.openWorkspaceDatabase", core.Concat("migrate DuckDB: ", path), resultError(result)))
	}

	mediumResult := orm.NewDuckDB(path)
	if !mediumResult.OK {
		closeStoreAfterFailure(databaseStore)
		return core.Fail(core.E("tui.openWorkspaceDatabase", core.Concat("open ORM DuckDB: ", path), resultError(mediumResult)))
	}
	medium, ok := mediumResult.Value.(*orm.DuckDBMedium)
	if !ok {
		closeStoreAfterFailure(databaseStore)
		return core.Fail(core.E("tui.openWorkspaceDatabase", "invalid ORM DuckDB result", nil))
	}

	runtime := core.New()
	if result := orm.Mount(runtime, "default", medium); !result.OK {
		closeDatabaseAfterFailure(databaseStore, medium, runtime)
		return core.Fail(core.E("tui.openWorkspaceDatabase", "mount ORM DuckDB", resultError(result)))
	}
	for _, schema := range workspaceRecordSchemas() {
		medium.RegisterTable(schema.Name, schema)
		if result := orm.RegisterSchema(runtime, schema); !result.OK {
			closeDatabaseAfterFailure(databaseStore, medium, runtime)
			return core.Fail(core.E("tui.openWorkspaceDatabase", core.Concat("register ORM schema: ", schema.Name), resultError(result)))
		}
	}

	return core.Ok(&workspaceDatabase{
		runtime: runtime,
		store:   databaseStore,
		medium:  medium,
	})
}

func applyWorkspaceMigrations(database *store.DuckDB, migrations []workspaceMigration, now func() time.Time) core.Result {
	if database == nil || database.Conn() == nil {
		return core.Fail(core.E("tui.applyWorkspaceMigrations", "DuckDB connection is required", nil))
	}
	if now == nil {
		now = time.Now
	}

	for _, migration := range migrations {
		if migration.Version < 1 {
			return core.Fail(core.E("tui.applyWorkspaceMigrations", "migration version must be positive", nil))
		}
		applied, err := workspaceMigrationApplied(database, migration.Version)
		if err != nil {
			return core.Fail(core.E(
				"tui.applyWorkspaceMigrations",
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
				"tui.applyWorkspaceMigrations",
				core.Sprintf("begin migration version %d", migration.Version),
				err,
			))
		}
		for _, statement := range migration.Statements {
			if core.Trim(statement) == "" {
				rollbackWorkspaceMigration(transaction)
				return core.Fail(core.E(
					"tui.applyWorkspaceMigrations",
					core.Sprintf("migration version %d contains an empty statement", migration.Version),
					nil,
				))
			}
			if _, err := transaction.Exec(statement); err != nil {
				rollbackWorkspaceMigration(transaction)
				return core.Fail(core.E(
					"tui.applyWorkspaceMigrations",
					core.Sprintf("execute migration version %d", migration.Version),
					err,
				))
			}
		}
		if _, err := transaction.Exec(
			"INSERT INTO lem_schema_versions (version, applied_at) VALUES (?, ?)",
			migration.Version,
			now().UTC(),
		); err != nil {
			rollbackWorkspaceMigration(transaction)
			return core.Fail(core.E(
				"tui.applyWorkspaceMigrations",
				core.Sprintf("record migration version %d", migration.Version),
				err,
			))
		}
		if err := transaction.Commit(); err != nil {
			rollbackWorkspaceMigration(transaction)
			return core.Fail(core.E(
				"tui.applyWorkspaceMigrations",
				core.Sprintf("commit migration version %d", migration.Version),
				err,
			))
		}
	}

	return core.Ok(nil)
}

func workspaceMigrationApplied(database *store.DuckDB, version int64) (bool, error) {
	var tableCount int
	if err := database.Conn().QueryRow(`
		SELECT COUNT(*)
		FROM information_schema.tables
		WHERE table_schema = 'main' AND table_name = 'lem_schema_versions'
	`).Scan(&tableCount); err != nil {
		return false, err
	}
	if tableCount == 0 {
		return false, nil
	}

	var versionCount int
	if err := database.Conn().QueryRow(
		"SELECT COUNT(*) FROM lem_schema_versions WHERE version = ?",
		version,
	).Scan(&versionCount); err != nil {
		return false, err
	}
	return versionCount > 0, nil
}

func rollbackWorkspaceMigration(transaction *sql.Tx) {
	if transaction == nil {
		return
	}
	if err := transaction.Rollback(); err != nil && err != sql.ErrTxDone {
		core.Warn("tui.migration.rollback", "error", err)
	}
}

func (database *workspaceDatabase) Close() core.Result {
	if database == nil {
		return core.Ok(nil)
	}
	if database.runtime != nil {
		orm.Remove(database.runtime)
		database.runtime = nil
	}

	result := core.Ok(nil)
	if database.medium != nil {
		if closeResult := database.medium.Close(); !closeResult.OK {
			result = closeResult
		}
		database.medium = nil
	}
	if database.store != nil {
		if closeResult := database.store.Close(); !closeResult.OK && result.OK {
			result = closeResult
		}
		database.store = nil
	}
	return result
}

func closeStoreAfterFailure(database *store.DuckDB) {
	if database == nil {
		return
	}
	if result := database.Close(); !result.OK {
		core.Warn("tui.database.close_after_failure", "error", result.Value)
	}
}

func closeDatabaseAfterFailure(databaseStore *store.DuckDB, medium *orm.DuckDBMedium, runtime *core.Core) {
	if runtime != nil {
		orm.Remove(runtime)
	}
	if medium != nil {
		if result := medium.Close(); !result.OK {
			core.Warn("tui.orm.close_after_failure", "error", result.Value)
		}
	}
	closeStoreAfterFailure(databaseStore)
}
