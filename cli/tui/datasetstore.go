// SPDX-Licence-Identifier: EUPL-1.2

package tui

import (
	"database/sql"
	"sync"
	"time"

	core "dappco.re/go"
	"dappco.re/go/inference/dataset"
)

// duckDatasetStore is the DuckDB-backed [dataset.Store] implementation —
// datasets.duckdb, opened separately from lem.duckdb per the design's
// bulk/lifecycle/blast-radius decision. mu serialises every method
// (reads and writes alike): datasets.duckdb is single-writer, and several
// methods are check-then-act (existence checks before insert, MAX(seq)
// before append) that must not interleave with a concurrent call.
type duckDatasetStore struct {
	database *datasetDatabase
	mu       sync.Mutex
}

var _ dataset.Store = (*duckDatasetStore)(nil)

// newDuckDatasetStore opens (creating and migrating if absent) the
// datasets.duckdb file at path and returns a ready [dataset.Store].
//
//	opened := newDuckDatasetStore(paths.Datasets)
//	store := opened.Value.(*duckDatasetStore)
//	defer store.Close()
func newDuckDatasetStore(path string) core.Result {
	opened := openDatasetDatabase(path)
	if !opened.OK {
		return opened
	}
	database, ok := opened.Value.(*datasetDatabase)
	if !ok {
		return core.Fail(core.E("tui.newDuckDatasetStore", "invalid dataset database result", nil))
	}
	return core.Ok(&duckDatasetStore{database: database})
}

// Close releases the underlying DuckDB connection. Safe to call on a nil
// receiver or an already-closed store.
func (s *duckDatasetStore) Close() core.Result {
	if s == nil || s.database == nil {
		return core.Ok(nil)
	}
	result := s.database.Close()
	s.database = nil
	return result
}

func (s *duckDatasetStore) conn() *sql.DB {
	if s == nil || s.database == nil || s.database.store == nil {
		return nil
	}
	return s.database.store.Conn()
}

func (s *duckDatasetStore) ready(operation string) core.Result {
	if s.conn() == nil {
		return datasetStoreFailure(operation, "dataset store is closed", nil)
	}
	return core.Ok(nil)
}

func datasetStoreFailure(operation, message string, cause error) core.Result {
	return core.Fail(core.E(core.Concat("tui.duckDatasetStore.", operation), message, cause))
}

// datasetNotFoundErr builds the exact "dataset.<kind>.notfound" Code
// [dataset.MemoryStore] produces (its unexported notFound helper in
// go/dataset/store.go) — conformance depends on this string matching
// byte-for-byte, since dataset.Store callers branch on Result.Code().
func datasetNotFoundErr(op, kind, id string) error {
	return &core.Err{Operation: op, Message: core.Concat(kind, " not found: ", id), Code: core.Concat("dataset.", kind, ".notfound")}
}

// sqlComparisonOp maps a dataset.ComparisonOp to its SQL operator.
// Everything but OpEQ ("==") is already valid SQL; OpEQ needs folding to
// a single "=". The switch is a closed whitelist — never string-built
// from filter input — so there is no injection surface even though the
// result is concatenated directly into a query string.
func sqlComparisonOp(op dataset.ComparisonOp) (string, error) {
	switch op {
	case dataset.OpGTE:
		return ">=", nil
	case dataset.OpLTE:
		return "<=", nil
	case dataset.OpGT:
		return ">", nil
	case dataset.OpLT:
		return "<", nil
	case dataset.OpEQ:
		return "=", nil
	case dataset.OpNEQ:
		return "!=", nil
	default:
		return "", core.NewError(core.Concat("dataset: unsupported comparison operator: ", string(op)))
	}
}

// ---- scan helpers ----

const datasetSelect = `SELECT id, slug, title, purpose, created_at, archived, archived_at FROM dataset_datasets`

func scanDataset(row rowScanner) (dataset.Dataset, error) {
	var d dataset.Dataset
	err := row.Scan(&d.ID, &d.Slug, &d.Title, &d.Purpose, &d.CreatedAt, &d.Archived, &d.ArchivedAt)
	return d, err
}

const itemSelect = `SELECT id, dataset_id, kind, content, source, source_ref, model_fingerprint, content_hash, parent_item_id, archived, archived_at, created_at FROM dataset_items`

func scanItem(row rowScanner) (dataset.Item, error) {
	var item dataset.Item
	err := row.Scan(&item.ID, &item.DatasetID, &item.Kind, &item.Content, &item.Source, &item.SourceRef,
		&item.ModelFingerprint, &item.ContentHash, &item.ParentItemID, &item.Archived, &item.ArchivedAt, &item.CreatedAt)
	return item, err
}

const scoreSelect = `SELECT id, item_id, kind, value, payload, scorer_name, scorer_version, judge_fingerprint, created_at FROM dataset_scores`

func scanScore(row rowScanner) (dataset.Score, error) {
	var sc dataset.Score
	err := row.Scan(&sc.ID, &sc.ItemID, &sc.Kind, &sc.Value, &sc.Payload, &sc.ScorerName, &sc.ScorerVersion, &sc.JudgeFingerprint, &sc.CreatedAt)
	return sc, err
}

const exportSelect = `SELECT id, dataset_id, format, filter_description, item_count, output_path, manifest_hash, created_at FROM dataset_exports`

func scanExport(row rowScanner) (dataset.Export, error) {
	var e dataset.Export
	err := row.Scan(&e.ID, &e.DatasetID, &e.Format, &e.FilterDescription, &e.ItemCount, &e.OutputPath, &e.ManifestHash, &e.CreatedAt)
	return e, err
}

// ---- existence checks (referential integrity, mirrors MemoryStore's
// hasDatasetLocked/hasItemLocked) ----

func (s *duckDatasetStore) datasetExistsLocked(id string) (bool, error) {
	var count int
	if err := s.conn().QueryRow("SELECT COUNT(*) FROM dataset_datasets WHERE id = ?", id).Scan(&count); err != nil {
		return false, err
	}
	return count > 0, nil
}

func (s *duckDatasetStore) itemExistsLocked(id string) (bool, error) {
	var count int
	if err := s.conn().QueryRow("SELECT COUNT(*) FROM dataset_items WHERE id = ?", id).Scan(&count); err != nil {
		return false, err
	}
	return count > 0, nil
}

// nextSeqLocked derives the next value of the insertion-order tiebreak
// column for table ("dataset_scores" or "dataset_reviews" — always a
// package-internal literal, never external input). Computed under s.mu,
// so the read-then-insert pair here is race-free: no interleaved append
// can observe or claim the same seq value.
func (s *duckDatasetStore) nextSeqLocked(table string) (int64, error) {
	var next int64
	if err := s.conn().QueryRow("SELECT COALESCE(MAX(seq), 0) + 1 FROM " + table).Scan(&next); err != nil {
		return 0, err
	}
	return next, nil
}

// ---- Dataset ----

func (s *duckDatasetStore) DatasetCreate(d dataset.Dataset) core.Result {
	if r := dataset.ValidateDataset(d); !r.OK {
		return r
	}
	if r := s.ready("DatasetCreate"); !r.OK {
		return r
	}
	s.mu.Lock()
	defer s.mu.Unlock()

	var idCount int
	if err := s.conn().QueryRow("SELECT COUNT(*) FROM dataset_datasets WHERE id = ?", d.ID).Scan(&idCount); err != nil {
		return datasetStoreFailure("DatasetCreate", "check existing dataset id", err)
	}
	if idCount > 0 {
		return datasetStoreFailure("DatasetCreate", "a dataset with this id already exists", nil)
	}
	var slugCount int
	if err := s.conn().QueryRow("SELECT COUNT(*) FROM dataset_datasets WHERE slug = ?", d.Slug).Scan(&slugCount); err != nil {
		return datasetStoreFailure("DatasetCreate", "check existing dataset slug", err)
	}
	if slugCount > 0 {
		return datasetStoreFailure("DatasetCreate", "a dataset with this slug already exists", nil)
	}

	archivedAt := d.ArchivedAt
	if archivedAt.IsZero() {
		archivedAt = unsetRecordTime()
	}
	if _, err := s.conn().Exec(
		`INSERT INTO dataset_datasets (id, slug, title, purpose, created_at, archived, archived_at) VALUES (?, ?, ?, ?, ?, ?, ?)`,
		d.ID, d.Slug, d.Title, d.Purpose, d.CreatedAt.UTC(), d.Archived, archivedAt.UTC(),
	); err != nil {
		return datasetStoreFailure("DatasetCreate", "insert dataset", err)
	}
	return core.Ok(d)
}

func (s *duckDatasetStore) Dataset(id string) core.Result {
	if r := s.ready("Dataset"); !r.OK {
		return r
	}
	if core.Trim(id) == "" {
		return datasetStoreFailure("Dataset", "dataset id is required", nil)
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	d, err := scanDataset(s.conn().QueryRow(datasetSelect+" WHERE id = ?", id))
	if err == sql.ErrNoRows {
		return core.Fail(datasetNotFoundErr("tui.duckDatasetStore.Dataset", "dataset", id))
	}
	if err != nil {
		return datasetStoreFailure("Dataset", "scan dataset", err)
	}
	return core.Ok(d)
}

func (s *duckDatasetStore) DatasetBySlug(slug string) core.Result {
	if r := s.ready("DatasetBySlug"); !r.OK {
		return r
	}
	if core.Trim(slug) == "" {
		return datasetStoreFailure("DatasetBySlug", "dataset slug is required", nil)
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	d, err := scanDataset(s.conn().QueryRow(datasetSelect+" WHERE slug = ?", slug))
	if err == sql.ErrNoRows {
		return core.Fail(datasetNotFoundErr("tui.duckDatasetStore.DatasetBySlug", "dataset", slug))
	}
	if err != nil {
		return datasetStoreFailure("DatasetBySlug", "scan dataset", err)
	}
	return core.Ok(d)
}

func (s *duckDatasetStore) Datasets(includeArchived bool) core.Result {
	if r := s.ready("Datasets"); !r.OK {
		return r
	}
	s.mu.Lock()
	defer s.mu.Unlock()

	query := datasetSelect
	args := make([]any, 0, 1)
	if !includeArchived {
		query += " WHERE archived = ?"
		args = append(args, false)
	}
	query += " ORDER BY created_at, id"

	rows, err := s.conn().Query(query, args...)
	if err != nil {
		return datasetStoreFailure("Datasets", "query datasets", err)
	}
	defer closeAgentRows(rows)
	out := make([]dataset.Dataset, 0)
	for rows.Next() {
		d, err := scanDataset(rows)
		if err != nil {
			return datasetStoreFailure("Datasets", "scan dataset", err)
		}
		out = append(out, d)
	}
	if err := rows.Err(); err != nil {
		return datasetStoreFailure("Datasets", "iterate datasets", err)
	}
	return core.Ok(out)
}

func (s *duckDatasetStore) DatasetArchive(id string) core.Result {
	if r := s.ready("DatasetArchive"); !r.OK {
		return r
	}
	if core.Trim(id) == "" {
		return datasetStoreFailure("DatasetArchive", "dataset id is required", nil)
	}
	s.mu.Lock()
	defer s.mu.Unlock()

	result, err := s.conn().Exec("UPDATE dataset_datasets SET archived = ?, archived_at = ? WHERE id = ?", true, time.Now().UTC(), id)
	if err != nil {
		return datasetStoreFailure("DatasetArchive", "archive dataset", err)
	}
	count, err := result.RowsAffected()
	if err != nil {
		return datasetStoreFailure("DatasetArchive", "count archived rows", err)
	}
	if count == 0 {
		return core.Fail(datasetNotFoundErr("tui.duckDatasetStore.DatasetArchive", "dataset", id))
	}
	d, err := scanDataset(s.conn().QueryRow(datasetSelect+" WHERE id = ?", id))
	if err != nil {
		return datasetStoreFailure("DatasetArchive", "scan archived dataset", err)
	}
	return core.Ok(d)
}

// ---- Item ----

func (s *duckDatasetStore) ItemAppend(item dataset.Item) core.Result {
	if r := dataset.ValidateItem(item); !r.OK {
		return r
	}
	if r := s.ready("ItemAppend"); !r.OK {
		return r
	}
	s.mu.Lock()
	defer s.mu.Unlock()

	exists, err := s.datasetExistsLocked(item.DatasetID)
	if err != nil {
		return datasetStoreFailure("ItemAppend", "check dataset exists", err)
	}
	if !exists {
		return core.Fail(datasetNotFoundErr("tui.duckDatasetStore.ItemAppend", "dataset", item.DatasetID))
	}
	itemExists, err := s.itemExistsLocked(item.ID)
	if err != nil {
		return datasetStoreFailure("ItemAppend", "check existing item", err)
	}
	if itemExists {
		return datasetStoreFailure("ItemAppend", "an item with this id already exists", nil)
	}

	archivedAt := item.ArchivedAt
	if archivedAt.IsZero() {
		archivedAt = unsetRecordTime()
	}
	if _, err := s.conn().Exec(
		`INSERT INTO dataset_items (id, dataset_id, kind, content, source, source_ref, model_fingerprint, content_hash, parent_item_id, archived, archived_at, created_at)
		 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
		item.ID, item.DatasetID, string(item.Kind), item.Content, string(item.Source), item.SourceRef,
		item.ModelFingerprint, item.ContentHash, item.ParentItemID, item.Archived, archivedAt.UTC(), item.CreatedAt.UTC(),
	); err != nil {
		return datasetStoreFailure("ItemAppend", "insert item", err)
	}
	return core.Ok(item)
}

func (s *duckDatasetStore) Item(id string) core.Result {
	if r := s.ready("Item"); !r.OK {
		return r
	}
	if core.Trim(id) == "" {
		return datasetStoreFailure("Item", "item id is required", nil)
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	item, err := scanItem(s.conn().QueryRow(itemSelect+" WHERE id = ?", id))
	if err == sql.ErrNoRows {
		return core.Fail(datasetNotFoundErr("tui.duckDatasetStore.Item", "item", id))
	}
	if err != nil {
		return datasetStoreFailure("Item", "scan item", err)
	}
	return core.Ok(item)
}

// Items compiles filter to one SQL query. The dataset id is a mandatory
// WHERE; every other dimension is an optional AND. Status and Score
// filters join the "latest per kind/item" views datasetMigrations
// defines — window-function views the orm DuckDB medium's query builder
// cannot express (its declared Subqueries/Aliases capabilities are not
// actually wired into DuckDBMedium.Read), so this method talks to
// *sql.DB directly, matching duckAgentStore's precedent.
func (s *duckDatasetStore) Items(filter dataset.ItemFilter) core.Result {
	if r := s.ready("Items"); !r.OK {
		return r
	}
	if core.Trim(filter.DatasetID) == "" {
		return datasetStoreFailure("Items", "item filter requires a dataset id", nil)
	}
	s.mu.Lock()
	defer s.mu.Unlock()

	query := "SELECT i.id, i.dataset_id, i.kind, i.content, i.source, i.source_ref, i.model_fingerprint, i.content_hash, i.parent_item_id, i.archived, i.archived_at, i.created_at FROM dataset_items i"
	args := make([]any, 0, 8)

	if filter.Status != "" {
		query += " LEFT JOIN dataset_latest_reviews lr ON lr.item_id = i.id"
	}
	if filter.Score != nil {
		query += " JOIN dataset_latest_scores ls ON ls.item_id = i.id AND ls.kind = ?"
		args = append(args, string(filter.Score.Kind))
	}

	query += " WHERE i.dataset_id = ?"
	args = append(args, filter.DatasetID)

	if !filter.IncludeArchived {
		query += " AND i.archived = ?"
		args = append(args, false)
	}
	if filter.Kind != "" {
		query += " AND i.kind = ?"
		args = append(args, string(filter.Kind))
	}
	if filter.Source != "" {
		query += " AND i.source = ?"
		args = append(args, string(filter.Source))
	}
	if filter.ContentHash != "" {
		query += " AND i.content_hash = ?"
		args = append(args, filter.ContentHash)
	}
	if filter.Status != "" {
		query += " AND COALESCE(lr.status, ?) = ?"
		args = append(args, string(dataset.StatusPending), string(filter.Status))
	}
	if filter.Score != nil {
		op, opErr := sqlComparisonOp(filter.Score.Op)
		if opErr != nil {
			return datasetStoreFailure("Items", "unsupported score expression operator", opErr)
		}
		query += " AND ls.value " + op + " ?"
		args = append(args, filter.Score.Threshold)
	}
	query += " ORDER BY i.created_at, i.id"

	rows, err := s.conn().Query(query, args...)
	if err != nil {
		return datasetStoreFailure("Items", "query items", err)
	}
	defer closeAgentRows(rows)
	out := make([]dataset.Item, 0)
	for rows.Next() {
		item, err := scanItem(rows)
		if err != nil {
			return datasetStoreFailure("Items", "scan item", err)
		}
		out = append(out, item)
	}
	if err := rows.Err(); err != nil {
		return datasetStoreFailure("Items", "iterate items", err)
	}
	return core.Ok(out)
}

// ---- Score ----

func (s *duckDatasetStore) ScoreAppend(score dataset.Score) core.Result {
	if r := dataset.ValidateScore(score); !r.OK {
		return r
	}
	if r := s.ready("ScoreAppend"); !r.OK {
		return r
	}
	s.mu.Lock()
	defer s.mu.Unlock()

	exists, err := s.itemExistsLocked(score.ItemID)
	if err != nil {
		return datasetStoreFailure("ScoreAppend", "check item exists", err)
	}
	if !exists {
		return core.Fail(datasetNotFoundErr("tui.duckDatasetStore.ScoreAppend", "item", score.ItemID))
	}

	seq, err := s.nextSeqLocked("dataset_scores")
	if err != nil {
		return datasetStoreFailure("ScoreAppend", "derive score sequence", err)
	}
	if _, err := s.conn().Exec(
		`INSERT INTO dataset_scores (id, item_id, kind, value, payload, scorer_name, scorer_version, judge_fingerprint, created_at, seq)
		 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
		score.ID, score.ItemID, string(score.Kind), score.Value, score.Payload, score.ScorerName, score.ScorerVersion,
		score.JudgeFingerprint, score.CreatedAt.UTC(), seq,
	); err != nil {
		return datasetStoreFailure("ScoreAppend", "insert score", err)
	}
	return core.Ok(score)
}

func (s *duckDatasetStore) Scores(itemID string) core.Result {
	if r := s.ready("Scores"); !r.OK {
		return r
	}
	if core.Trim(itemID) == "" {
		return datasetStoreFailure("Scores", "item id is required", nil)
	}
	s.mu.Lock()
	defer s.mu.Unlock()

	exists, err := s.itemExistsLocked(itemID)
	if err != nil {
		return datasetStoreFailure("Scores", "check item exists", err)
	}
	if !exists {
		return core.Fail(datasetNotFoundErr("tui.duckDatasetStore.Scores", "item", itemID))
	}

	rows, err := s.conn().Query(scoreSelect+" WHERE item_id = ? ORDER BY created_at, id", itemID)
	if err != nil {
		return datasetStoreFailure("Scores", "query scores", err)
	}
	defer closeAgentRows(rows)
	out := make([]dataset.Score, 0)
	for rows.Next() {
		sc, err := scanScore(rows)
		if err != nil {
			return datasetStoreFailure("Scores", "scan score", err)
		}
		out = append(out, sc)
	}
	if err := rows.Err(); err != nil {
		return datasetStoreFailure("Scores", "iterate scores", err)
	}
	return core.Ok(out)
}

// ---- Review ----

func (s *duckDatasetStore) ReviewAppend(review dataset.Review) core.Result {
	if r := dataset.ValidateReview(review); !r.OK {
		return r
	}
	if r := s.ready("ReviewAppend"); !r.OK {
		return r
	}
	s.mu.Lock()
	defer s.mu.Unlock()

	exists, err := s.itemExistsLocked(review.ItemID)
	if err != nil {
		return datasetStoreFailure("ReviewAppend", "check item exists", err)
	}
	if !exists {
		return core.Fail(datasetNotFoundErr("tui.duckDatasetStore.ReviewAppend", "item", review.ItemID))
	}

	seq, err := s.nextSeqLocked("dataset_reviews")
	if err != nil {
		return datasetStoreFailure("ReviewAppend", "derive review sequence", err)
	}
	if _, err := s.conn().Exec(
		`INSERT INTO dataset_reviews (item_id, status, reviewer, note, created_at, seq) VALUES (?, ?, ?, ?, ?, ?)`,
		review.ItemID, string(review.Status), review.Reviewer, review.Note, review.CreatedAt.UTC(), seq,
	); err != nil {
		return datasetStoreFailure("ReviewAppend", "insert review", err)
	}
	return core.Ok(review)
}

func (s *duckDatasetStore) ReviewLatest(itemID string) core.Result {
	if r := s.ready("ReviewLatest"); !r.OK {
		return r
	}
	if core.Trim(itemID) == "" {
		return datasetStoreFailure("ReviewLatest", "item id is required", nil)
	}
	s.mu.Lock()
	defer s.mu.Unlock()

	exists, err := s.itemExistsLocked(itemID)
	if err != nil {
		return datasetStoreFailure("ReviewLatest", "check item exists", err)
	}
	if !exists {
		return core.Fail(datasetNotFoundErr("tui.duckDatasetStore.ReviewLatest", "item", itemID))
	}

	var review dataset.Review
	err = s.conn().QueryRow(
		"SELECT item_id, status, reviewer, note, created_at FROM dataset_latest_reviews WHERE item_id = ?", itemID,
	).Scan(&review.ItemID, &review.Status, &review.Reviewer, &review.Note, &review.CreatedAt)
	if err == sql.ErrNoRows {
		return core.Ok(dataset.Review{ItemID: itemID, Status: dataset.StatusPending})
	}
	if err != nil {
		return datasetStoreFailure("ReviewLatest", "scan latest review", err)
	}
	return core.Ok(review)
}

// ---- Export ----

func (s *duckDatasetStore) ExportAppend(export dataset.Export) core.Result {
	if r := dataset.ValidateExport(export); !r.OK {
		return r
	}
	if r := s.ready("ExportAppend"); !r.OK {
		return r
	}
	s.mu.Lock()
	defer s.mu.Unlock()

	exists, err := s.datasetExistsLocked(export.DatasetID)
	if err != nil {
		return datasetStoreFailure("ExportAppend", "check dataset exists", err)
	}
	if !exists {
		return core.Fail(datasetNotFoundErr("tui.duckDatasetStore.ExportAppend", "dataset", export.DatasetID))
	}

	// No id-uniqueness check here — matches dataset.MemoryStore.ExportAppend
	// exactly (see the datasetMigrations doc comment): Store's interface
	// only documents "fails if ID already exists" for DatasetCreate and
	// ItemAppend. dataset_exports.id is not a primary key for the same
	// reason.
	if _, err := s.conn().Exec(
		`INSERT INTO dataset_exports (id, dataset_id, format, filter_description, item_count, output_path, manifest_hash, created_at)
		 VALUES (?, ?, ?, ?, ?, ?, ?, ?)`,
		export.ID, export.DatasetID, string(export.Format), export.FilterDescription, export.ItemCount,
		export.OutputPath, export.ManifestHash, export.CreatedAt.UTC(),
	); err != nil {
		return datasetStoreFailure("ExportAppend", "insert export", err)
	}
	return core.Ok(export)
}

func (s *duckDatasetStore) Exports(datasetID string) core.Result {
	if r := s.ready("Exports"); !r.OK {
		return r
	}
	if core.Trim(datasetID) == "" {
		return datasetStoreFailure("Exports", "dataset id is required", nil)
	}
	s.mu.Lock()
	defer s.mu.Unlock()

	exists, err := s.datasetExistsLocked(datasetID)
	if err != nil {
		return datasetStoreFailure("Exports", "check dataset exists", err)
	}
	if !exists {
		return core.Fail(datasetNotFoundErr("tui.duckDatasetStore.Exports", "dataset", datasetID))
	}

	rows, err := s.conn().Query(exportSelect+" WHERE dataset_id = ? ORDER BY created_at, id", datasetID)
	if err != nil {
		return datasetStoreFailure("Exports", "query exports", err)
	}
	defer closeAgentRows(rows)
	out := make([]dataset.Export, 0)
	for rows.Next() {
		e, err := scanExport(rows)
		if err != nil {
			return datasetStoreFailure("Exports", "scan export", err)
		}
		out = append(out, e)
	}
	if err := rows.Err(); err != nil {
		return datasetStoreFailure("Exports", "iterate exports", err)
	}
	return core.Ok(out)
}

// ---- CLI-facing entry point ----

// DatasetStore is the [dataset.Store] handle [OpenDatasetStore] returns.
// Callers close it exactly once when done (typically via defer) — the
// interface exists purely so a caller outside this package (cli/data.go,
// the serve/ssd capture taps) can hold and close the concrete
// *duckDatasetStore [OpenDatasetStore] builds without naming its
// unexported type: Go's structural typing lets an unexported
// implementation satisfy an exported interface asserted from outside the
// package, exactly as dataset.Store itself already does.
type DatasetStore interface {
	dataset.Store
	// Close releases the underlying DuckDB connection. Safe to call on an
	// already-closed store.
	Close() core.Result
}

// OpenDatasetStore opens (creating the ~/.lem layout and migrating
// datasets.duckdb if either is absent) the one dataset store every `lem
// data` verb and the serve/ssd capture taps share — "open the DuckDB
// store via the cli/tui dataset machinery" per the dataset loop design,
// so no caller outside this package ever resolves the ~/.lem path
// itself (see appPathsAt's medium-backed path contract). There is no
// --home override: the root always resolves from $HOME via
// defaultAppPaths, matching every other ~/.lem consumer in this binary —
// tests redirect it with t.Setenv("HOME", ...), the same seam
// cli/serve_test.go already uses for the admin-token path.
//
//	opened := tui.OpenDatasetStore()
//	if !opened.OK { return opened }
//	store := opened.Value.(tui.DatasetStore)
//	defer store.Close()
func OpenDatasetStore() core.Result {
	pathsResult := defaultAppPaths()
	if !pathsResult.OK {
		return pathsResult
	}
	paths, ok := pathsResult.Value.(appPaths)
	if !ok {
		return core.Fail(core.E("tui.OpenDatasetStore", "invalid application paths result", nil))
	}
	// openAppFilesAt ensures the ~/.lem layout exists (root + the standard
	// subdirectories) before the DuckDB adapter ever touches it — the same
	// preparation loadDefaultWorkspace runs for lem.duckdb, applied here to
	// datasets.duckdb's separate root-relative path.
	if ensured := openAppFilesAt(paths.Root); !ensured.OK {
		return core.Fail(core.E("tui.OpenDatasetStore", "ensure application layout", resultError(ensured)))
	}
	return newDuckDatasetStore(paths.Datasets)
}
