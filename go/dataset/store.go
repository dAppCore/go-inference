// SPDX-Licence-Identifier: EUPL-1.2

package dataset

import (
	"sort"
	"sync"
	"time"

	core "dappco.re/go"
)

// ComparisonOp is the tiny comparison grammar [ScoreExpression] uses —
// nothing beyond these six operators.
type ComparisonOp string

const (
	OpGTE ComparisonOp = ">="
	OpLTE ComparisonOp = "<="
	OpGT  ComparisonOp = ">"
	OpLT  ComparisonOp = "<"
	OpEQ  ComparisonOp = "=="
	OpNEQ ComparisonOp = "!="
)

// knownComparisonOp reports whether op is one ScoreExpression accepts.
func knownComparisonOp(op ComparisonOp) bool {
	switch op {
	case OpGTE, OpLTE, OpGT, OpLT, OpEQ, OpNEQ:
		return true
	default:
		return false
	}
}

// ScoreExpression is a structural score threshold/filter predicate — the
// parsed form of the tiny explicit grammar "<field> <op> <number>" (see
// ParseScoreExpression in score.go): does the LATEST score of Kind
// satisfy Op against Threshold. Used both by Store item-list filters
// (ItemFilter.Score) and by --auto-approve/--auto-reject.
type ScoreExpression struct {
	Kind      ScoreKind
	Op        ComparisonOp
	Threshold float64
}

// String renders the expression back in its source grammar, e.g.
// "lek>=80" — used as the human-readable filter description an Export
// manifest records.
func (e ScoreExpression) String() string {
	return core.Sprintf("%s%s%v", e.Kind, e.Op, e.Threshold)
}

// Matches reports whether scores contains a Kind-matching row whose
// LATEST value (by CreatedAt, ties broken by later slice position —
// i.e. insertion order) satisfies the expression. An item that has
// never received a score of Kind never matches — a threshold cannot be
// satisfied by an absent measurement.
//
//	expr := dataset.ScoreExpression{Kind: dataset.ScoreKindLEK, Op: dataset.OpGTE, Threshold: 80}
//	if expr.Matches(scores) { approve() }
func (e ScoreExpression) Matches(scores []Score) bool {
	found := false
	var latest Score
	for _, s := range scores {
		if s.Kind != e.Kind {
			continue
		}
		if !found || !s.CreatedAt.Before(latest.CreatedAt) {
			latest = s
			found = true
		}
	}
	if !found {
		return false
	}
	switch e.Op {
	case OpGTE:
		return latest.Value >= e.Threshold
	case OpLTE:
		return latest.Value <= e.Threshold
	case OpGT:
		return latest.Value > e.Threshold
	case OpLT:
		return latest.Value < e.Threshold
	case OpEQ:
		return latest.Value == e.Threshold
	case OpNEQ:
		return latest.Value != e.Threshold
	default:
		return false
	}
}

// ItemFilter scopes an Items query. DatasetID is required; every other
// field is an optional narrowing dimension — its zero value ("" / nil /
// false) means "any". Status filters on the item's LATEST review status
// (StatusPending when no Review row exists yet).
type ItemFilter struct {
	DatasetID       string
	Kind            ItemKind
	Source          ItemSource
	Status          ReviewStatus
	ContentHash     string
	Score           *ScoreExpression
	IncludeArchived bool
}

// Store is the persistence boundary a driver implements — the CLI
// module's DuckDB-backed store for production, [MemoryStore] for
// root-module tests. Mirrors the shape
// dappco.re/go/inference/agent/orchestrator.Store established for the
// CoreAgent port: every method returns core.Result so validation and
// not-found both flow through the one channel.
//
// Every Append method validates its record (see ValidateDataset /
// ValidateItem / ValidateScore / ValidateReview / ValidateExport) and
// checks referential integrity (an Item's DatasetID, a Score/Review's
// ItemID, an Export's DatasetID must already exist) before persisting —
// a conforming implementation enforces both, exactly as [MemoryStore]
// does, so CLI-side Store conformance tests can drive the same behaviour
// against either.
type Store interface {
	// DatasetCreate persists a new Dataset. Fails if ID or Slug already
	// exist.
	DatasetCreate(Dataset) core.Result
	// Dataset returns the Dataset with id, or a "dataset.notfound" Result.
	Dataset(id string) core.Result
	// DatasetBySlug returns the Dataset with slug, or a
	// "dataset.notfound" Result.
	DatasetBySlug(slug string) core.Result
	// Datasets lists all datasets (optionally including archived ones),
	// ordered by (created_at, id).
	Datasets(includeArchived bool) core.Result
	// DatasetArchive flags the dataset archived (idempotent) and
	// returns the updated Dataset.
	DatasetArchive(id string) core.Result

	// ItemAppend persists a new Item. Fails if DatasetID does not exist
	// or ID already exists.
	ItemAppend(Item) core.Result
	// Item returns the Item with id, or an "dataset.item.notfound"
	// Result.
	Item(id string) core.Result
	// Items lists items matching filter, ordered by (created_at, id).
	Items(ItemFilter) core.Result

	// ScoreAppend persists a new Score row. Append-only: re-scoring adds
	// a row, it never overwrites. Fails if ItemID does not exist.
	ScoreAppend(Score) core.Result
	// Scores returns every Score row for itemID, oldest first. Fails if
	// itemID does not exist.
	Scores(itemID string) core.Result

	// ReviewAppend persists a new Review row. Append-only — the latest
	// row (by created_at) is the item's current status. Fails if
	// ItemID does not exist.
	ReviewAppend(Review) core.Result
	// ReviewLatest returns the most recent Review for itemID, or a
	// synthetic StatusPending Review when none exists yet. Fails if
	// itemID does not exist.
	ReviewLatest(itemID string) core.Result

	// ExportAppend persists a new Export receipt. Fails if DatasetID
	// does not exist.
	ExportAppend(Export) core.Result
	// Exports lists export receipts for datasetID, ordered by
	// (created_at, id). Fails if datasetID does not exist.
	Exports(datasetID string) core.Result
}

// ValidateDataset checks the required fields a Store implementation must
// enforce before persisting a Dataset.
func ValidateDataset(d Dataset) core.Result {
	if core.Trim(d.ID) == "" {
		return core.Fail(core.NewError("dataset: dataset id is required"))
	}
	if !validSlug(d.Slug) {
		return core.Fail(core.NewError("dataset: slug must be a non-empty lowercase alphanumeric-and-hyphen string, not starting or ending with a hyphen"))
	}
	if core.Trim(d.Title) == "" {
		return core.Fail(core.NewError("dataset: title is required"))
	}
	if d.CreatedAt.IsZero() {
		return core.Fail(core.NewError("dataset: created_at is required"))
	}
	return core.Ok(d)
}

// validSlug reports whether slug is a valid Dataset.Slug: non-empty,
// lowercase ascii letters/digits/hyphens only, not starting or ending
// with a hyphen.
func validSlug(slug string) bool {
	if slug == "" {
		return false
	}
	if slug[0] == '-' || slug[len(slug)-1] == '-' {
		return false
	}
	for i := 0; i < len(slug); i++ {
		c := slug[i]
		switch {
		case c >= 'a' && c <= 'z':
		case c >= '0' && c <= '9':
		case c == '-':
		default:
			return false
		}
	}
	return true
}

// ValidateItem checks the required fields a Store implementation must
// enforce before persisting an Item, including its Content shape (see
// [ValidateItemContent]).
func ValidateItem(item Item) core.Result {
	if core.Trim(item.ID) == "" {
		return core.Fail(core.NewError("dataset: item id is required"))
	}
	if core.Trim(item.DatasetID) == "" {
		return core.Fail(core.NewError("dataset: item dataset id is required"))
	}
	if !knownItemKind(item.Kind) {
		return core.Fail(core.NewError("dataset: item kind is unknown"))
	}
	if !knownItemSource(item.Source) {
		return core.Fail(core.NewError("dataset: item source is unknown"))
	}
	if core.Trim(item.ContentHash) == "" {
		return core.Fail(core.NewError("dataset: item content hash is required"))
	}
	if item.CreatedAt.IsZero() {
		return core.Fail(core.NewError("dataset: item created_at is required"))
	}
	if r := ValidateItemContent(item.Kind, item.Content); !r.OK {
		return r
	}
	return core.Ok(item)
}

// ValidateScore checks the required fields a Store implementation must
// enforce before persisting a Score.
func ValidateScore(s Score) core.Result {
	if core.Trim(s.ID) == "" {
		return core.Fail(core.NewError("dataset: score id is required"))
	}
	if core.Trim(s.ItemID) == "" {
		return core.Fail(core.NewError("dataset: score item id is required"))
	}
	if !knownScoreKind(s.Kind) {
		return core.Fail(core.NewError("dataset: score kind is unknown"))
	}
	if core.Trim(s.ScorerName) == "" {
		return core.Fail(core.NewError("dataset: score scorer name is required"))
	}
	if s.CreatedAt.IsZero() {
		return core.Fail(core.NewError("dataset: score created_at is required"))
	}
	return core.Ok(s)
}

// ValidateReview checks the required fields a Store implementation must
// enforce before persisting a Review.
func ValidateReview(r Review) core.Result {
	if core.Trim(r.ItemID) == "" {
		return core.Fail(core.NewError("dataset: review item id is required"))
	}
	if !knownReviewStatus(r.Status) {
		return core.Fail(core.NewError("dataset: review status is unknown"))
	}
	if core.Trim(r.Reviewer) == "" {
		return core.Fail(core.NewError("dataset: review reviewer is required"))
	}
	if r.CreatedAt.IsZero() {
		return core.Fail(core.NewError("dataset: review created_at is required"))
	}
	return core.Ok(r)
}

// ValidateExport checks the required fields a Store implementation must
// enforce before persisting an Export receipt.
func ValidateExport(e Export) core.Result {
	if core.Trim(e.ID) == "" {
		return core.Fail(core.NewError("dataset: export id is required"))
	}
	if core.Trim(e.DatasetID) == "" {
		return core.Fail(core.NewError("dataset: export dataset id is required"))
	}
	if !knownExportFormat(e.Format) {
		return core.Fail(core.NewError("dataset: export format is unknown"))
	}
	if core.Trim(e.OutputPath) == "" {
		return core.Fail(core.NewError("dataset: export output path is required"))
	}
	if core.Trim(e.ManifestHash) == "" {
		return core.Fail(core.NewError("dataset: export manifest hash is required"))
	}
	if e.CreatedAt.IsZero() {
		return core.Fail(core.NewError("dataset: export created_at is required"))
	}
	return core.Ok(e)
}

// MemoryStore is the in-memory [Store] fake — the root-module test
// double, and the reference behaviour a CLI-side DuckDB Store
// implementation's conformance tests drive against (Task 5). Safe for
// concurrent use.
type MemoryStore struct {
	mu       sync.Mutex
	datasets []Dataset
	items    []Item
	scores   []Score
	reviews  []Review
	exports  []Export
}

var _ Store = (*MemoryStore)(nil)

// NewMemoryStore returns an empty MemoryStore.
//
//	store := dataset.NewMemoryStore()
func NewMemoryStore() *MemoryStore {
	return &MemoryStore{}
}

func notFound(op, kind, id string) error {
	return &core.Err{Operation: op, Message: core.Concat(kind, " not found: ", id), Code: core.Concat("dataset.", kind, ".notfound")}
}

func (m *MemoryStore) hasDatasetLocked(id string) bool {
	for _, d := range m.datasets {
		if d.ID == id {
			return true
		}
	}
	return false
}

func (m *MemoryStore) hasItemLocked(id string) bool {
	for _, item := range m.items {
		if item.ID == id {
			return true
		}
	}
	return false
}

func (m *MemoryStore) scoresForLocked(itemID string) []Score {
	var out []Score
	for _, s := range m.scores {
		if s.ItemID == itemID {
			out = append(out, s)
		}
	}
	return out
}

// latestReviewLocked returns itemID's latest Review (ties broken by
// slice/append order), or a synthetic StatusPending Review when none
// exists — pending is the implicit starting state, not an error.
func (m *MemoryStore) latestReviewLocked(itemID string) Review {
	found := false
	var latest Review
	for _, r := range m.reviews {
		if r.ItemID != itemID {
			continue
		}
		if !found || !r.CreatedAt.Before(latest.CreatedAt) {
			latest = r
			found = true
		}
	}
	if !found {
		return Review{ItemID: itemID, Status: StatusPending}
	}
	return latest
}

func (m *MemoryStore) DatasetCreate(d Dataset) core.Result {
	if r := ValidateDataset(d); !r.OK {
		return r
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	for _, existing := range m.datasets {
		if existing.ID == d.ID {
			return core.Fail(core.NewError("dataset: a dataset with this id already exists"))
		}
		if existing.Slug == d.Slug {
			return core.Fail(core.NewError("dataset: a dataset with this slug already exists"))
		}
	}
	m.datasets = append(m.datasets, d)
	return core.Ok(d)
}

func (m *MemoryStore) Dataset(id string) core.Result {
	m.mu.Lock()
	defer m.mu.Unlock()
	for _, d := range m.datasets {
		if d.ID == id {
			return core.Ok(d)
		}
	}
	return core.Fail(notFound("dataset.Dataset", "dataset", id))
}

func (m *MemoryStore) DatasetBySlug(slug string) core.Result {
	m.mu.Lock()
	defer m.mu.Unlock()
	for _, d := range m.datasets {
		if d.Slug == slug {
			return core.Ok(d)
		}
	}
	return core.Fail(notFound("dataset.DatasetBySlug", "dataset", slug))
}

func (m *MemoryStore) Datasets(includeArchived bool) core.Result {
	m.mu.Lock()
	defer m.mu.Unlock()
	out := make([]Dataset, 0, len(m.datasets))
	for _, d := range m.datasets {
		if d.Archived && !includeArchived {
			continue
		}
		out = append(out, d)
	}
	sortByCreated(out, func(d Dataset) (time.Time, string) { return d.CreatedAt, d.ID })
	return core.Ok(out)
}

func (m *MemoryStore) DatasetArchive(id string) core.Result {
	m.mu.Lock()
	defer m.mu.Unlock()
	for i, d := range m.datasets {
		if d.ID == id {
			m.datasets[i].Archived = true
			m.datasets[i].ArchivedAt = time.Now()
			return core.Ok(m.datasets[i])
		}
	}
	return core.Fail(notFound("dataset.DatasetArchive", "dataset", id))
}

func (m *MemoryStore) ItemAppend(item Item) core.Result {
	if r := ValidateItem(item); !r.OK {
		return r
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	if !m.hasDatasetLocked(item.DatasetID) {
		return core.Fail(notFound("dataset.ItemAppend", "dataset", item.DatasetID))
	}
	if m.hasItemLocked(item.ID) {
		return core.Fail(core.NewError("dataset: an item with this id already exists"))
	}
	m.items = append(m.items, item)
	return core.Ok(item)
}

func (m *MemoryStore) Item(id string) core.Result {
	m.mu.Lock()
	defer m.mu.Unlock()
	for _, item := range m.items {
		if item.ID == id {
			return core.Ok(item)
		}
	}
	return core.Fail(notFound("dataset.Item", "item", id))
}

func (m *MemoryStore) Items(filter ItemFilter) core.Result {
	if core.Trim(filter.DatasetID) == "" {
		return core.Fail(core.NewError("dataset: item filter requires a dataset id"))
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	out := make([]Item, 0, len(m.items))
	for _, item := range m.items {
		if item.DatasetID != filter.DatasetID {
			continue
		}
		if item.Archived && !filter.IncludeArchived {
			continue
		}
		if filter.Kind != "" && item.Kind != filter.Kind {
			continue
		}
		if filter.Source != "" && item.Source != filter.Source {
			continue
		}
		if filter.ContentHash != "" && item.ContentHash != filter.ContentHash {
			continue
		}
		if filter.Status != "" && m.latestReviewLocked(item.ID).Status != filter.Status {
			continue
		}
		if filter.Score != nil && !filter.Score.Matches(m.scoresForLocked(item.ID)) {
			continue
		}
		out = append(out, item)
	}
	sortByCreated(out, func(item Item) (time.Time, string) { return item.CreatedAt, item.ID })
	return core.Ok(out)
}

func (m *MemoryStore) ScoreAppend(score Score) core.Result {
	if r := ValidateScore(score); !r.OK {
		return r
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	if !m.hasItemLocked(score.ItemID) {
		return core.Fail(notFound("dataset.ScoreAppend", "item", score.ItemID))
	}
	m.scores = append(m.scores, score)
	return core.Ok(score)
}

func (m *MemoryStore) Scores(itemID string) core.Result {
	m.mu.Lock()
	defer m.mu.Unlock()
	if !m.hasItemLocked(itemID) {
		return core.Fail(notFound("dataset.Scores", "item", itemID))
	}
	out := m.scoresForLocked(itemID)
	sortByCreated(out, func(s Score) (time.Time, string) { return s.CreatedAt, s.ID })
	return core.Ok(out)
}

func (m *MemoryStore) ReviewAppend(review Review) core.Result {
	if r := ValidateReview(review); !r.OK {
		return r
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	if !m.hasItemLocked(review.ItemID) {
		return core.Fail(notFound("dataset.ReviewAppend", "item", review.ItemID))
	}
	m.reviews = append(m.reviews, review)
	return core.Ok(review)
}

func (m *MemoryStore) ReviewLatest(itemID string) core.Result {
	m.mu.Lock()
	defer m.mu.Unlock()
	if !m.hasItemLocked(itemID) {
		return core.Fail(notFound("dataset.ReviewLatest", "item", itemID))
	}
	return core.Ok(m.latestReviewLocked(itemID))
}

func (m *MemoryStore) ExportAppend(export Export) core.Result {
	if r := ValidateExport(export); !r.OK {
		return r
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	if !m.hasDatasetLocked(export.DatasetID) {
		return core.Fail(notFound("dataset.ExportAppend", "dataset", export.DatasetID))
	}
	m.exports = append(m.exports, export)
	return core.Ok(export)
}

func (m *MemoryStore) Exports(datasetID string) core.Result {
	m.mu.Lock()
	defer m.mu.Unlock()
	if !m.hasDatasetLocked(datasetID) {
		return core.Fail(notFound("dataset.Exports", "dataset", datasetID))
	}
	out := make([]Export, 0)
	for _, e := range m.exports {
		if e.DatasetID == datasetID {
			out = append(out, e)
		}
	}
	sortByCreated(out, func(e Export) (time.Time, string) { return e.CreatedAt, e.ID })
	return core.Ok(out)
}

// sortByCreated stably sorts items in place by (created_at, id), ties on
// created_at broken by id — the deterministic ordering every list method
// promises.
func sortByCreated[T any](items []T, key func(T) (time.Time, string)) {
	sort.SliceStable(items, func(i, j int) bool {
		ta, ia := key(items[i])
		tb, ib := key(items[j])
		if !ta.Equal(tb) {
			return ta.Before(tb)
		}
		return ia < ib
	})
}
