// SPDX-License-Identifier: EUPL-1.2

package tui

import (
	"database/sql"
	"sync"
	"time"

	core "dappco.re/go"
	"dappco.re/go/inference/agent/orchestrator"
	"dappco.re/go/inference/agent/work"
)

type duckAgentStore struct {
	connection *sql.DB
	mu         sync.Mutex
	writeMu    *sync.Mutex
}

var _ orchestrator.Store = (*duckAgentStore)(nil)

func newDuckAgentStore(repository workspaceRepository) core.Result {
	provider, ok := repository.(workspaceConnectionProvider)
	if !ok {
		return core.Fail(core.E("tui.newDuckAgentStore", "workspace repository has no open SQL connection", nil))
	}
	connection := provider.workspaceConnection()
	writeMutex := provider.workspaceWriteMutex()
	if connection == nil || writeMutex == nil {
		return core.Fail(core.E("tui.newDuckAgentStore", "workspace repository has no open SQL connection", nil))
	}
	return core.Ok(&duckAgentStore{connection: connection, writeMu: writeMutex})
}

func (store *duckAgentStore) Recover(at time.Time) core.Result {
	if result := store.ready("Recover"); !result.OK {
		return result
	}
	if at.IsZero() {
		return agentStoreFailure("Recover", "recovery time is required", nil)
	}
	store.writeMu.Lock()
	defer store.writeMu.Unlock()
	store.mu.Lock()
	defer store.mu.Unlock()
	result, err := store.connection.Exec(`UPDATE agent_runs
		SET status = ?, finished_at = ?, updated_at = ?
		WHERE status IN (?, ?, ?, ?)`,
		work.RunInterrupted, at.UTC(), at.UTC(), work.RunQueued, work.RunPreparing, work.RunRunning, work.RunCancelling)
	if err != nil {
		return agentStoreFailure("Recover", "interrupt active runs", err)
	}
	count, err := result.RowsAffected()
	if err != nil {
		return agentStoreFailure("Recover", "count interrupted runs", err)
	}
	return core.Ok(int(count))
}

func (store *duckAgentStore) Commit(commit orchestrator.Commit) core.Result {
	if result := store.ready("Commit"); !result.OK {
		return result
	}
	if commit.Project == nil && commit.Run == nil && commit.Event == nil && len(commit.Logs) == 0 && commit.Question == nil && commit.Answer == nil && commit.Acceptance == nil && commit.Queue == nil && commit.Provider == nil {
		return agentStoreFailure("Commit", "commit requires at least one record", nil)
	}
	if commit.CreateRun && (commit.Run == nil || commit.ExpectedStatus != nil) {
		return agentStoreFailure("Commit", "run creation requires a run and no expected status", nil)
	}
	if !commit.CreateRun && commit.Run != nil && commit.ExpectedStatus == nil {
		return agentStoreFailure("Commit", "run update requires expected status", nil)
	}
	if commit.Run == nil && commit.ExpectedStatus != nil {
		return agentStoreFailure("Commit", "expected status requires a run update", nil)
	}
	previous := int64(0)
	for _, chunk := range commit.Logs {
		if chunk.Sequence <= previous {
			return agentStoreFailure("Commit", "log sequences must be positive and increasing", nil)
		}
		previous = chunk.Sequence
	}

	store.writeMu.Lock()
	defer store.writeMu.Unlock()
	store.mu.Lock()
	defer store.mu.Unlock()
	transaction, err := store.connection.Begin()
	if err != nil {
		return agentStoreFailure("Commit", "begin transaction", err)
	}
	rollback := func(operation string, cause error) core.Result {
		rollbackWorkspaceMigration(transaction)
		return agentStoreFailure("Commit", operation, cause)
	}

	if project := commit.Project; project != nil {
		_, err = transaction.Exec(`INSERT INTO agent_projects
			(id, source_path, repository_root, source_branch, source_revision, repository_name, clone_path, created_at, updated_at)
			VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
			ON CONFLICT (id) DO UPDATE SET source_path = excluded.source_path, repository_root = excluded.repository_root,
			source_branch = excluded.source_branch, source_revision = excluded.source_revision,
			repository_name = excluded.repository_name, clone_path = excluded.clone_path, updated_at = excluded.updated_at`,
			project.ID, project.SourcePath, project.RepositoryRoot, project.SourceBranch, project.SourceRevision,
			project.RepositoryName, project.ClonePath, project.CreatedAt.UTC(), project.UpdatedAt.UTC())
		if err != nil {
			return rollback("write project", err)
		}
	}
	if run := commit.Run; run != nil {
		if commit.CreateRun {
			_, err = transaction.Exec(`INSERT INTO agent_runs
				(id, work_id, project_id, parent_run_id, provider, model, source_revision, durable_revision,
				execution_revision, accepted_revision, branch, worktree, command_receipt, run_number, attempt,
				process_id, status, exit_code, failure_reason, queued_at, started_at, finished_at, updated_at)
				VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`, runValues(run)...)
			if err != nil {
				return rollback("create run", err)
			}
		} else {
			arguments := runValues(run)
			arguments = append(arguments, run.ID, *commit.ExpectedStatus)
			updated, updateErr := transaction.Exec(`UPDATE agent_runs SET
				id = ?, work_id = ?, project_id = ?, parent_run_id = ?, provider = ?, model = ?, source_revision = ?,
				durable_revision = ?, execution_revision = ?, accepted_revision = ?, branch = ?, worktree = ?,
				command_receipt = ?, run_number = ?, attempt = ?, process_id = ?, status = ?, exit_code = ?,
				failure_reason = ?, queued_at = ?, started_at = ?, finished_at = ?, updated_at = ?
				WHERE id = ? AND status = ?`, arguments...)
			if updateErr != nil {
				return rollback("update run", updateErr)
			}
			count, countErr := updated.RowsAffected()
			if countErr != nil {
				return rollback("count updated runs", countErr)
			}
			if count != 1 {
				return rollback("compare-and-swap run status", core.NewError("stale expected run status"))
			}
		}
	}
	if event := commit.Event; event != nil {
		_, err = transaction.Exec(`INSERT INTO agent_events
			(id, run_id, work_id, kind, title, detail, detail_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)`,
			event.ID, event.RunID, event.WorkID, event.Kind, event.Title, event.Detail, event.DetailJSON, event.CreatedAt.UTC())
		if err != nil {
			return rollback("write event", err)
		}
	}
	for _, chunk := range commit.Logs {
		var maximum int64
		if err = transaction.QueryRow("SELECT COALESCE(MAX(sequence), 0) FROM agent_log_chunks WHERE run_id = ?", chunk.RunID).Scan(&maximum); err != nil {
			return rollback("read log sequence", err)
		}
		if chunk.Sequence <= maximum {
			return rollback("write log chunk", core.NewError("log sequence is not monotonic"))
		}
		_, err = transaction.Exec(`INSERT INTO agent_log_chunks
			(run_id, sequence, stream, text, created_at) VALUES (?, ?, ?, ?, ?)`,
			chunk.RunID, chunk.Sequence, chunk.Stream, chunk.Text, chunk.CreatedAt.UTC())
		if err != nil {
			return rollback("write log chunk", err)
		}
	}
	if question := commit.Question; question != nil {
		_, err = transaction.Exec(`INSERT INTO agent_questions
			(id, run_id, text, created_at) VALUES (?, ?, ?, ?)`, question.ID, question.RunID, question.Text, question.CreatedAt.UTC())
		if err != nil {
			return rollback("write question", err)
		}
	}
	if answer := commit.Answer; answer != nil {
		_, err = transaction.Exec(`INSERT INTO agent_answers
			(id, question_id, resume_run_id, text, created_at) VALUES (?, ?, ?, ?, ?)`,
			answer.ID, answer.QuestionID, answer.ResumeRunID, answer.Text, answer.CreatedAt.UTC())
		if err != nil {
			return rollback("write answer", err)
		}
	}
	if acceptance := commit.Acceptance; acceptance != nil {
		_, err = transaction.Exec(`INSERT INTO agent_acceptances
			(id, work_id, run_id, source_base, agent_base, agent_tip, integration_branch, integration_worktree,
			result_revision, status, validation_json, failure_reason, created_at, updated_at)
			VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`, acceptance.ID, acceptance.WorkID, acceptance.RunID,
			acceptance.SourceBase, acceptance.AgentBase, acceptance.AgentTip, acceptance.IntegrationBranch,
			acceptance.IntegrationWorktree, acceptance.ResultRevision, acceptance.Status, acceptance.ValidationJSON,
			acceptance.FailureReason, acceptance.CreatedAt.UTC(), acceptance.UpdatedAt.UTC())
		if err != nil {
			return rollback("write acceptance", err)
		}
	}
	if queue := commit.Queue; queue != nil {
		_, err = transaction.Exec(`INSERT INTO agent_queue_state (id, status, reason, updated_at) VALUES (?, ?, ?, ?)
			ON CONFLICT (id) DO UPDATE SET status = excluded.status, reason = excluded.reason, updated_at = excluded.updated_at`,
			queue.ID, queue.Status, queue.Reason, queue.UpdatedAt.UTC())
		if err != nil {
			return rollback("write queue state", err)
		}
	}
	if provider := commit.Provider; provider != nil {
		_, err = transaction.Exec(`INSERT INTO agent_provider_state
			(provider, backoff_reason, last_run_id, backoff_until, last_started_at, window_started_at, window_admissions, updated_at)
			VALUES (?, ?, ?, ?, ?, ?, ?, ?)
			ON CONFLICT (provider) DO UPDATE SET backoff_reason = excluded.backoff_reason, last_run_id = excluded.last_run_id,
			backoff_until = excluded.backoff_until, last_started_at = excluded.last_started_at,
			window_started_at = excluded.window_started_at, window_admissions = excluded.window_admissions,
			updated_at = excluded.updated_at`, provider.Provider, provider.BackoffReason, provider.LastRunID,
			provider.BackoffUntil.UTC(), provider.LastStartedAt.UTC(), provider.WindowStartedAt.UTC(),
			provider.WindowAdmissions, provider.UpdatedAt.UTC())
		if err != nil {
			return rollback("write provider state", err)
		}
	}
	if err = transaction.Commit(); err != nil {
		return rollback("commit transaction", err)
	}
	return core.Ok(nil)
}

func (store *duckAgentStore) Project(id string) core.Result {
	if result := store.ready("Project"); !result.OK {
		return result
	}
	if core.Trim(id) == "" {
		return agentStoreFailure("Project", "project ID is required", nil)
	}
	store.mu.Lock()
	defer store.mu.Unlock()
	return scanAgentProject(store.connection.QueryRow(`SELECT id, source_path, repository_root, source_branch,
		source_revision, repository_name, clone_path, created_at, updated_at FROM agent_projects WHERE id = ?`, id), "Project")
}

func (store *duckAgentStore) ProjectBySource(source string) core.Result {
	if result := store.ready("ProjectBySource"); !result.OK {
		return result
	}
	if core.Trim(source) == "" {
		return agentStoreFailure("ProjectBySource", "source path is required", nil)
	}
	store.mu.Lock()
	defer store.mu.Unlock()
	var project work.Project
	err := store.connection.QueryRow(`SELECT id, source_path, repository_root, source_branch,
		source_revision, repository_name, clone_path, created_at, updated_at FROM agent_projects WHERE source_path = ?`, source).Scan(
		&project.ID, &project.SourcePath, &project.RepositoryRoot, &project.SourceBranch, &project.SourceRevision,
		&project.RepositoryName, &project.ClonePath, &project.CreatedAt, &project.UpdatedAt)
	if err == sql.ErrNoRows {
		return core.Ok(nil)
	}
	if err != nil {
		return agentStoreFailure("ProjectBySource", "scan project", err)
	}
	return core.Ok(project)
}

func (store *duckAgentStore) Run(id string) core.Result {
	if result := store.ready("Run"); !result.OK {
		return result
	}
	if core.Trim(id) == "" {
		return agentStoreFailure("Run", "run ID is required", nil)
	}
	store.mu.Lock()
	defer store.mu.Unlock()
	return scanAgentRun(store.connection.QueryRow(runSelect+" WHERE id = ?", id), "Run")
}

func (store *duckAgentStore) NextRunNumber(workID string) core.Result {
	if result := store.ready("NextRunNumber"); !result.OK {
		return result
	}
	if core.Trim(workID) == "" {
		return agentStoreFailure("NextRunNumber", "work ID is required", nil)
	}
	store.mu.Lock()
	defer store.mu.Unlock()
	var next int
	if err := store.connection.QueryRow("SELECT COALESCE(MAX(run_number), 0) + 1 FROM agent_runs WHERE work_id = ?", workID).Scan(&next); err != nil {
		return agentStoreFailure("NextRunNumber", "derive next run number", err)
	}
	return core.Ok(next)
}

func (store *duckAgentStore) Continuation(runID string) core.Result {
	if result := store.ready("Continuation"); !result.OK {
		return result
	}
	if core.Trim(runID) == "" {
		return agentStoreFailure("Continuation", "run ID is required", nil)
	}
	store.mu.Lock()
	defer store.mu.Unlock()
	runResult := scanAgentRun(store.connection.QueryRow(runSelect+" WHERE id = ?", runID), "Continuation")
	if !runResult.OK {
		return runResult
	}
	continuation := work.Continuation{Run: runResult.Value.(work.Run)}
	var task string
	err := store.connection.QueryRow(`SELECT detail FROM agent_events WHERE run_id = ? AND kind = 'queued'
		ORDER BY created_at, id LIMIT 1`, runID).Scan(&task)
	if err != nil && err != sql.ErrNoRows {
		return agentStoreFailure("Continuation", "read task event", err)
	}
	continuation.Task = task
	logs, err := queryAgentLogs(store.connection, "WHERE run_id = ? ORDER BY sequence", runID)
	if err != nil {
		return agentStoreFailure("Continuation", "read logs", err)
	}
	continuation.Logs = logs
	err = store.connection.QueryRow(`SELECT id, run_id, text, created_at FROM agent_questions WHERE run_id = ?`, runID).Scan(
		&continuation.Question.ID, &continuation.Question.RunID, &continuation.Question.Text, &continuation.Question.CreatedAt)
	if err != nil && err != sql.ErrNoRows {
		return agentStoreFailure("Continuation", "read question", err)
	}
	if continuation.Question.ID != "" {
		err = store.connection.QueryRow(`SELECT id, question_id, resume_run_id, text, created_at FROM agent_answers WHERE question_id = ?`, continuation.Question.ID).Scan(
			&continuation.Answer.ID, &continuation.Answer.QuestionID, &continuation.Answer.ResumeRunID, &continuation.Answer.Text, &continuation.Answer.CreatedAt)
		if err != nil && err != sql.ErrNoRows {
			return agentStoreFailure("Continuation", "read answer", err)
		}
	}
	return core.Ok(continuation)
}

func (store *duckAgentStore) Snapshot(workID string) core.Result {
	if result := store.ready("Snapshot"); !result.OK {
		return result
	}
	store.mu.Lock()
	defer store.mu.Unlock()
	snapshot := work.Snapshot{}
	projects, err := queryAgentProjects(store.connection)
	if err != nil {
		return agentStoreFailure("Snapshot", "read projects", err)
	}
	snapshot.Projects = projects
	runs, err := queryAgentRuns(store.connection, workID)
	if err != nil {
		return agentStoreFailure("Snapshot", "read runs", err)
	}
	snapshot.Runs = runs
	events, err := queryAgentEvents(store.connection, workID)
	if err != nil {
		return agentStoreFailure("Snapshot", "read events", err)
	}
	snapshot.Events = events
	logs, err := querySnapshotLogs(store.connection, workID)
	if err != nil {
		return agentStoreFailure("Snapshot", "read logs", err)
	}
	snapshot.Logs = logs
	questions, err := queryAgentQuestions(store.connection, workID)
	if err != nil {
		return agentStoreFailure("Snapshot", "read questions", err)
	}
	snapshot.Questions = questions
	acceptances, err := queryAgentAcceptances(store.connection, workID)
	if err != nil {
		return agentStoreFailure("Snapshot", "read acceptances", err)
	}
	snapshot.Acceptances = acceptances
	err = store.connection.QueryRow(`SELECT id, status, reason, updated_at FROM agent_queue_state WHERE id = 'default'`).Scan(
		&snapshot.Queue.ID, &snapshot.Queue.Status, &snapshot.Queue.Reason, &snapshot.Queue.UpdatedAt)
	if err != nil && err != sql.ErrNoRows {
		return agentStoreFailure("Snapshot", "read queue state", err)
	}
	providers, err := queryAgentProviders(store.connection)
	if err != nil {
		return agentStoreFailure("Snapshot", "read provider states", err)
	}
	snapshot.Providers = providers
	return core.Ok(snapshot)
}

const runSelect = `SELECT id, work_id, project_id, parent_run_id, provider, model, source_revision,
	durable_revision, execution_revision, accepted_revision, branch, worktree, command_receipt, run_number,
	attempt, process_id, status, exit_code, failure_reason, queued_at, started_at, finished_at, updated_at FROM agent_runs`

func runValues(run *work.Run) []any {
	return []any{run.ID, run.WorkID, run.ProjectID, run.ParentRunID, run.Provider, run.Model, run.SourceRevision,
		run.DurableRevision, run.ExecutionRevision, run.AcceptedRevision, run.Branch, run.Worktree, run.CommandReceipt,
		run.Number, run.Attempt, run.ProcessID, run.Status, run.ExitCode, run.FailureReason, run.QueuedAt.UTC(),
		run.StartedAt.UTC(), run.FinishedAt.UTC(), run.UpdatedAt.UTC()}
}

type rowScanner interface {
	Scan(...any) error
}

func scanAgentProject(row rowScanner, operation string) core.Result {
	var project work.Project
	if err := row.Scan(&project.ID, &project.SourcePath, &project.RepositoryRoot, &project.SourceBranch,
		&project.SourceRevision, &project.RepositoryName, &project.ClonePath, &project.CreatedAt, &project.UpdatedAt); err != nil {
		return agentStoreFailure(operation, "scan project", err)
	}
	return core.Ok(project)
}

func scanAgentRun(row rowScanner, operation string) core.Result {
	var run work.Run
	if err := row.Scan(&run.ID, &run.WorkID, &run.ProjectID, &run.ParentRunID, &run.Provider, &run.Model,
		&run.SourceRevision, &run.DurableRevision, &run.ExecutionRevision, &run.AcceptedRevision, &run.Branch,
		&run.Worktree, &run.CommandReceipt, &run.Number, &run.Attempt, &run.ProcessID, &run.Status, &run.ExitCode,
		&run.FailureReason, &run.QueuedAt, &run.StartedAt, &run.FinishedAt, &run.UpdatedAt); err != nil {
		return agentStoreFailure(operation, "scan run", err)
	}
	return core.Ok(run)
}

func queryAgentProjects(connection *sql.DB) ([]work.Project, error) {
	rows, err := connection.Query(`SELECT id, source_path, repository_root, source_branch, source_revision,
		repository_name, clone_path, created_at, updated_at FROM agent_projects ORDER BY id`)
	if err != nil {
		return nil, err
	}
	defer closeAgentRows(rows)
	projects := make([]work.Project, 0)
	for rows.Next() {
		result := scanAgentProject(rows, "Snapshot")
		if !result.OK {
			return nil, result.Err()
		}
		projects = append(projects, result.Value.(work.Project))
	}
	return projects, rows.Err()
}

func queryAgentRuns(connection *sql.DB, workID string) ([]work.Run, error) {
	query := runSelect
	arguments := make([]any, 0, 1)
	if workID != "" {
		query += " WHERE work_id = ?"
		arguments = append(arguments, workID)
	}
	query += " ORDER BY queued_at, id"
	rows, err := connection.Query(query, arguments...)
	if err != nil {
		return nil, err
	}
	defer closeAgentRows(rows)
	runs := make([]work.Run, 0)
	for rows.Next() {
		result := scanAgentRun(rows, "Snapshot")
		if !result.OK {
			return nil, result.Err()
		}
		runs = append(runs, result.Value.(work.Run))
	}
	return runs, rows.Err()
}

func queryAgentEvents(connection *sql.DB, workID string) ([]work.Event, error) {
	query := "SELECT id, run_id, work_id, kind, title, detail, detail_json, created_at FROM agent_events"
	arguments := make([]any, 0, 1)
	if workID != "" {
		query += " WHERE work_id = ?"
		arguments = append(arguments, workID)
	}
	query += " ORDER BY created_at, id"
	rows, err := connection.Query(query, arguments...)
	if err != nil {
		return nil, err
	}
	defer closeAgentRows(rows)
	values := make([]work.Event, 0)
	for rows.Next() {
		var value work.Event
		if err := rows.Scan(&value.ID, &value.RunID, &value.WorkID, &value.Kind, &value.Title, &value.Detail, &value.DetailJSON, &value.CreatedAt); err != nil {
			return nil, err
		}
		values = append(values, value)
	}
	return values, rows.Err()
}

func queryAgentLogs(connection *sql.DB, suffix string, arguments ...any) ([]work.LogChunk, error) {
	rows, err := connection.Query("SELECT run_id, sequence, stream, text, created_at FROM agent_log_chunks "+suffix, arguments...)
	if err != nil {
		return nil, err
	}
	defer closeAgentRows(rows)
	values := make([]work.LogChunk, 0)
	for rows.Next() {
		var value work.LogChunk
		if err := rows.Scan(&value.RunID, &value.Sequence, &value.Stream, &value.Text, &value.CreatedAt); err != nil {
			return nil, err
		}
		values = append(values, value)
	}
	return values, rows.Err()
}

func querySnapshotLogs(connection *sql.DB, workID string) ([]work.LogChunk, error) {
	if workID == "" {
		return queryAgentLogs(connection, "ORDER BY sequence, run_id")
	}
	return queryAgentLogs(connection, `INNER JOIN agent_runs ON agent_runs.id = agent_log_chunks.run_id
		WHERE agent_runs.work_id = ? ORDER BY agent_log_chunks.sequence, agent_log_chunks.run_id`, workID)
}

func queryAgentQuestions(connection *sql.DB, workID string) ([]work.Question, error) {
	query := `SELECT agent_questions.id, agent_questions.run_id, agent_questions.text, agent_questions.created_at
		FROM agent_questions`
	arguments := make([]any, 0, 1)
	if workID != "" {
		query += " INNER JOIN agent_runs ON agent_runs.id = agent_questions.run_id WHERE agent_runs.work_id = ?"
		arguments = append(arguments, workID)
	}
	query += " ORDER BY agent_questions.created_at, agent_questions.id"
	rows, err := connection.Query(query, arguments...)
	if err != nil {
		return nil, err
	}
	defer closeAgentRows(rows)
	values := make([]work.Question, 0)
	for rows.Next() {
		var value work.Question
		if err := rows.Scan(&value.ID, &value.RunID, &value.Text, &value.CreatedAt); err != nil {
			return nil, err
		}
		values = append(values, value)
	}
	return values, rows.Err()
}

func queryAgentAcceptances(connection *sql.DB, workID string) ([]work.Acceptance, error) {
	query := `SELECT id, work_id, run_id, source_base, agent_base, agent_tip, integration_branch,
		integration_worktree, result_revision, status, validation_json, failure_reason, created_at, updated_at
		FROM agent_acceptances`
	arguments := make([]any, 0, 1)
	if workID != "" {
		query += " WHERE work_id = ?"
		arguments = append(arguments, workID)
	}
	query += " ORDER BY updated_at, id"
	rows, err := connection.Query(query, arguments...)
	if err != nil {
		return nil, err
	}
	defer closeAgentRows(rows)
	values := make([]work.Acceptance, 0)
	for rows.Next() {
		var value work.Acceptance
		if err := rows.Scan(&value.ID, &value.WorkID, &value.RunID, &value.SourceBase, &value.AgentBase,
			&value.AgentTip, &value.IntegrationBranch, &value.IntegrationWorktree, &value.ResultRevision,
			&value.Status, &value.ValidationJSON, &value.FailureReason, &value.CreatedAt, &value.UpdatedAt); err != nil {
			return nil, err
		}
		values = append(values, value)
	}
	return values, rows.Err()
}

func queryAgentProviders(connection *sql.DB) ([]work.ProviderState, error) {
	rows, err := connection.Query(`SELECT provider, backoff_reason, last_run_id, backoff_until, last_started_at,
		window_started_at, window_admissions, updated_at FROM agent_provider_state ORDER BY provider`)
	if err != nil {
		return nil, err
	}
	defer closeAgentRows(rows)
	values := make([]work.ProviderState, 0)
	for rows.Next() {
		var value work.ProviderState
		if err := rows.Scan(&value.Provider, &value.BackoffReason, &value.LastRunID, &value.BackoffUntil,
			&value.LastStartedAt, &value.WindowStartedAt, &value.WindowAdmissions, &value.UpdatedAt); err != nil {
			return nil, err
		}
		values = append(values, value)
	}
	return values, rows.Err()
}

func closeAgentRows(rows *sql.Rows) {
	if rows == nil {
		return
	}
	if err := rows.Close(); err != nil {
		core.Warn("tui.agentstore.rows_close", "error", err)
	}
}

func (store *duckAgentStore) ready(operation string) core.Result {
	if store == nil || store.connection == nil || store.writeMu == nil {
		return agentStoreFailure(operation, "agent store is closed", nil)
	}
	return core.Ok(nil)
}

func agentStoreFailure(operation, message string, cause error) core.Result {
	return core.Fail(core.E(core.Concat("tui.duckAgentStore.", operation), message, cause))
}
