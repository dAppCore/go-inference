// SPDX-License-Identifier: EUPL-1.2

package tui

import (
	"time"

	core "dappco.re/go"
	"dappco.re/go/orm"
)

type sessionSearchHit struct {
	Session sessionRecord
	Snippet string
}

type workspaceRepository interface {
	Close() core.Result
	Session(id string) core.Result
	ListSessions(includeArchived bool) core.Result
	SearchSessions(query string, limit int) core.Result
	SaveSession(record sessionRecord) core.Result
	Turns(sessionID string) core.Result
	SaveTurn(record turnRecord) core.Result
	Events(sessionID string) core.Result
	SaveEvent(record eventRecord) core.Result
	Jobs(sessionID string) core.Result
	SaveJob(record generationJobRecord) core.Result
	InterruptActiveJobs(at time.Time) core.Result
	ListWorkItems(includeArchived bool) core.Result
	WorkItemByExternalID(externalID string) core.Result
	SaveWorkItem(record workItemRecord) core.Result
	Artifacts(sessionID string) core.Result
	SaveArtifact(record artifactRecord) core.Result
	Attachments(sessionID string) core.Result
	SaveAttachment(record attachmentRecord) core.Result
}

type duckRepository struct {
	database *workspaceDatabase
}

func openDuckRepository(path string) core.Result {
	opened := openWorkspaceDatabase(path)
	if !opened.OK {
		return opened
	}
	database, ok := opened.Value.(*workspaceDatabase)
	if !ok {
		return core.Fail(core.E("tui.openDuckRepository", "invalid workspace database result", nil))
	}
	return core.Ok(&duckRepository{database: database})
}

func (repository *duckRepository) Close() core.Result {
	if repository == nil || repository.database == nil {
		return core.Ok(nil)
	}
	result := repository.database.Close()
	repository.database = nil
	return result
}

func (repository *duckRepository) Session(id string) core.Result {
	if result := repository.ready("Session"); !result.OK {
		return result
	}
	if core.Trim(id) == "" {
		return repositoryFailure("Session", "session ID is required")
	}
	return orm.Of[sessionRecord](repository.database.runtime).Find(id)
}

func (repository *duckRepository) ListSessions(includeArchived bool) core.Result {
	if result := repository.ready("ListSessions"); !result.OK {
		return result
	}
	query := orm.Of[sessionRecord](repository.database.runtime)
	if !includeArchived {
		query = query.Where("archived", "=", false)
	}
	return query.Order("last_opened_at", "desc").Get()
}

func (repository *duckRepository) SearchSessions(query string, limit int) core.Result {
	if result := repository.ready("SearchSessions"); !result.OK {
		return result
	}
	query = core.Trim(query)
	if query == "" {
		return core.Ok([]sessionSearchHit{})
	}
	if limit < 1 {
		limit = 20
	}
	if limit > 100 {
		limit = 100
	}
	pattern := core.Concat("%", query, "%")
	rows, err := repository.database.store.Conn().Query(`
		WITH ranked_turn_matches AS (
			SELECT
				session_id,
				visible,
				ROW_NUMBER() OVER (PARTITION BY session_id ORDER BY sequence) AS match_rank
			FROM lem_turns
			WHERE LOWER(visible) LIKE LOWER(?)
		)
		SELECT
			s.id,
			s.title,
			s.status,
			s.preferred_model,
			s.mode,
			s.generation_json,
			s.tools_json,
			s.created_at,
			s.updated_at,
			s.last_opened_at,
			s.archived,
			s.archived_at,
			COALESCE(matches.visible, s.title) AS snippet
		FROM lem_sessions AS s
		LEFT JOIN ranked_turn_matches AS matches
			ON matches.session_id = s.id AND matches.match_rank = 1
		WHERE s.archived = FALSE
			AND (LOWER(s.title) LIKE LOWER(?) OR matches.session_id IS NOT NULL)
		ORDER BY s.last_opened_at DESC
		LIMIT ?
	`, pattern, pattern, limit)
	if err != nil {
		return core.Fail(core.E("tui.duckRepository.SearchSessions", "query sessions", err))
	}
	defer func() {
		if closeErr := rows.Close(); closeErr != nil {
			core.Warn("tui.repository.search_rows_close", "error", closeErr)
		}
	}()

	hits := make([]sessionSearchHit, 0)
	for rows.Next() {
		var hit sessionSearchHit
		if err := rows.Scan(
			&hit.Session.ID,
			&hit.Session.Title,
			&hit.Session.Status,
			&hit.Session.PreferredModel,
			&hit.Session.Mode,
			&hit.Session.GenerationJSON,
			&hit.Session.ToolsJSON,
			&hit.Session.CreatedAt,
			&hit.Session.UpdatedAt,
			&hit.Session.LastOpenedAt,
			&hit.Session.Archived,
			&hit.Session.ArchivedAt,
			&hit.Snippet,
		); err != nil {
			return core.Fail(core.E("tui.duckRepository.SearchSessions", "scan session", err))
		}
		hits = append(hits, hit)
	}
	if err := rows.Err(); err != nil {
		return core.Fail(core.E("tui.duckRepository.SearchSessions", "iterate sessions", err))
	}
	return core.Ok(hits)
}

func (repository *duckRepository) SaveSession(record sessionRecord) core.Result {
	if result := repository.ready("SaveSession"); !result.OK {
		return result
	}
	if record.ArchivedAt.IsZero() {
		record.ArchivedAt = unsetRecordTime()
	}
	return orm.Of[sessionRecord](repository.database.runtime).Save(&record)
}

func (repository *duckRepository) Turns(sessionID string) core.Result {
	if result := repository.ready("Turns"); !result.OK {
		return result
	}
	if core.Trim(sessionID) == "" {
		return repositoryFailure("Turns", "session ID is required")
	}
	return orm.Of[turnRecord](repository.database.runtime).
		Where("session_id", "=", sessionID).
		Order("sequence", "asc").
		Get()
}

func (repository *duckRepository) SaveTurn(record turnRecord) core.Result {
	if result := repository.ready("SaveTurn"); !result.OK {
		return result
	}
	return orm.Of[turnRecord](repository.database.runtime).Save(&record)
}

func (repository *duckRepository) Events(sessionID string) core.Result {
	if result := repository.ready("Events"); !result.OK {
		return result
	}
	if core.Trim(sessionID) == "" {
		return repositoryFailure("Events", "session ID is required")
	}
	return orm.Of[eventRecord](repository.database.runtime).
		Where("session_id", "=", sessionID).
		Order("created_at", "asc").
		Get()
}

func (repository *duckRepository) SaveEvent(record eventRecord) core.Result {
	if result := repository.ready("SaveEvent"); !result.OK {
		return result
	}
	return orm.Of[eventRecord](repository.database.runtime).Save(&record)
}

func (repository *duckRepository) Jobs(sessionID string) core.Result {
	if result := repository.ready("Jobs"); !result.OK {
		return result
	}
	if core.Trim(sessionID) == "" {
		return repositoryFailure("Jobs", "session ID is required")
	}
	return orm.Of[generationJobRecord](repository.database.runtime).
		Where("session_id", "=", sessionID).
		Order("created_at", "asc").
		Get()
}

func (repository *duckRepository) SaveJob(record generationJobRecord) core.Result {
	if result := repository.ready("SaveJob"); !result.OK {
		return result
	}
	if record.StartedAt.IsZero() {
		record.StartedAt = unsetRecordTime()
	}
	if record.FinishedAt.IsZero() {
		record.FinishedAt = unsetRecordTime()
	}
	return orm.Of[generationJobRecord](repository.database.runtime).Save(&record)
}

func (repository *duckRepository) InterruptActiveJobs(at time.Time) core.Result {
	if result := repository.ready("InterruptActiveJobs"); !result.OK {
		return result
	}
	return repository.database.store.Exec(`
		UPDATE lem_generation_jobs
		SET status = ?, finished_at = ?
		WHERE status IN (?, ?)
	`, "interrupted", at.UTC(), "queued", "generating")
}

func (repository *duckRepository) ListWorkItems(includeArchived bool) core.Result {
	if result := repository.ready("ListWorkItems"); !result.OK {
		return result
	}
	query := orm.Of[workItemRecord](repository.database.runtime)
	if !includeArchived {
		query = query.Where("archived", "=", false)
	}
	return query.Order("updated_at", "desc").Get()
}

func (repository *duckRepository) WorkItemByExternalID(externalID string) core.Result {
	if result := repository.ready("WorkItemByExternalID"); !result.OK {
		return result
	}
	if core.Trim(externalID) == "" {
		return repositoryFailure("WorkItemByExternalID", "external ID is required")
	}
	return orm.Of[workItemRecord](repository.database.runtime).
		Where("external_id", "=", externalID).
		First()
}

func (repository *duckRepository) SaveWorkItem(record workItemRecord) core.Result {
	if result := repository.ready("SaveWorkItem"); !result.OK {
		return result
	}
	if record.ArchivedAt.IsZero() {
		record.ArchivedAt = unsetRecordTime()
	}
	return orm.Of[workItemRecord](repository.database.runtime).Save(&record)
}

func (repository *duckRepository) Artifacts(sessionID string) core.Result {
	if result := repository.ready("Artifacts"); !result.OK {
		return result
	}
	if core.Trim(sessionID) == "" {
		return repositoryFailure("Artifacts", "session ID is required")
	}
	return orm.Of[artifactRecord](repository.database.runtime).
		Where("session_id", "=", sessionID).
		Where("archived", "=", false).
		Order("created_at", "asc").
		Get()
}

func (repository *duckRepository) SaveArtifact(record artifactRecord) core.Result {
	if result := repository.ready("SaveArtifact"); !result.OK {
		return result
	}
	if record.ArchivedAt.IsZero() {
		record.ArchivedAt = unsetRecordTime()
	}
	return orm.Of[artifactRecord](repository.database.runtime).Save(&record)
}

func (repository *duckRepository) Attachments(sessionID string) core.Result {
	if result := repository.ready("Attachments"); !result.OK {
		return result
	}
	if core.Trim(sessionID) == "" {
		return repositoryFailure("Attachments", "session ID is required")
	}
	return orm.Of[attachmentRecord](repository.database.runtime).
		Where("session_id", "=", sessionID).
		Where("archived", "=", false).
		Order("added_at", "asc").
		Get()
}

func (repository *duckRepository) SaveAttachment(record attachmentRecord) core.Result {
	if result := repository.ready("SaveAttachment"); !result.OK {
		return result
	}
	if record.ArchivedAt.IsZero() {
		record.ArchivedAt = unsetRecordTime()
	}
	return orm.Of[attachmentRecord](repository.database.runtime).Save(&record)
}

func (repository *duckRepository) ready(operation string) core.Result {
	if repository == nil || repository.database == nil || repository.database.runtime == nil || repository.database.store == nil {
		return repositoryFailure(operation, "repository is closed")
	}
	return core.Ok(nil)
}

func repositoryFailure(operation, message string) core.Result {
	return core.Fail(core.E(core.Concat("tui.duckRepository.", operation), message, nil))
}
