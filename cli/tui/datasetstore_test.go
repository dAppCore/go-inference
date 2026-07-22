// SPDX-License-Identifier: EUPL-1.2

package tui

import (
	"os"
	"testing"
	"time"

	core "dappco.re/go"
	"dappco.re/go/inference/dataset"
	"dappco.re/go/store"
)

// ---- test-only construction helpers ----

func conformancePairItem(datasetID, prompt, response string, at time.Time) dataset.Item {
	content := core.MustCast[[]byte](core.JSONMarshal(dataset.PairContent{Prompt: prompt, Response: response}))
	hash := core.MustCast[string](dataset.ContentHash(dataset.KindPair, content))
	return dataset.Item{
		ID: dataset.NewID(), DatasetID: datasetID, Kind: dataset.KindPair, Content: content,
		Source: dataset.SourceImportJSONL, ContentHash: hash, CreatedAt: at,
	}
}

func conformanceScore(itemID string, kind dataset.ScoreKind, value float64, at time.Time) dataset.Score {
	return dataset.Score{ID: dataset.NewID(), ItemID: itemID, Kind: kind, Value: value, ScorerName: "conformance-scorer", ScorerVersion: "1", CreatedAt: at}
}

func indexOfDataset(list []dataset.Dataset, id string) int {
	for i, d := range list {
		if d.ID == id {
			return i
		}
	}
	return -1
}

// ---- store open/close test helpers ----

func openTestDatasetStore(t *testing.T, path string) *store.DuckDB {
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

func closeTestDatasetStore(t *testing.T, database *store.DuckDB) {
	t.Helper()
	if result := database.Close(); !result.OK {
		t.Errorf("close DuckDB: %v", result.Value)
	}
}

func newTestDuckDatasetStore(t *testing.T, path string) *duckDatasetStore {
	t.Helper()
	opened := newDuckDatasetStore(path)
	if !opened.OK {
		t.Fatalf("newDuckDatasetStore(%q) failed: %v", path, opened.Value)
	}
	s, ok := opened.Value.(*duckDatasetStore)
	if !ok {
		t.Fatalf("newDuckDatasetStore value = %T, want *duckDatasetStore", opened.Value)
	}
	return s
}

func closeTestDuckDatasetStore(t *testing.T, s *duckDatasetStore) {
	t.Helper()
	if result := s.Close(); !result.OK {
		t.Errorf("close dataset store: %v", result.Value)
	}
}

// ---- CONFORMANCE: the same behavioural table driven over both
// dataset.MemoryStore (the root package's reference behaviour) and
// duckDatasetStore (this package's DuckDB implementation). Every
// sub-test uses its own dataset slug / item namespace so the two runs
// never collide within one shared store instance. ----

func TestDatasetStoreConformance(t *testing.T) {
	backends := []struct {
		name string
		open func(t *testing.T) dataset.Store
	}{
		{name: "MemoryStore", open: func(t *testing.T) dataset.Store {
			return dataset.NewMemoryStore()
		}},
		{name: "DuckDB", open: func(t *testing.T) dataset.Store {
			path := t.TempDir() + "/datasets.duckdb"
			s := newTestDuckDatasetStore(t, path)
			t.Cleanup(func() { closeTestDuckDatasetStore(t, s) })
			return s
		}},
	}

	for _, backend := range backends {
		t.Run(backend.name, func(t *testing.T) {
			st := backend.open(t)
			t.Run("DatasetLifecycle", func(t *testing.T) { testDatasetLifecycle(t, st) })
			t.Run("ItemLifecycleAndFilters", func(t *testing.T) { testItemLifecycleAndFilters(t, st) })
			t.Run("ScoreAppendAndExpressionMatching", func(t *testing.T) { testScoreAppendAndExpressionMatching(t, st) })
			t.Run("ReviewLatestWins", func(t *testing.T) { testReviewLatestWins(t, st) })
			t.Run("ExportLifecycle", func(t *testing.T) { testExportLifecycle(t, st) })
			t.Run("NotFoundCodes", func(t *testing.T) { testNotFoundCodes(t, st) })
		})
	}
}

func testDatasetLifecycle(t *testing.T, st dataset.Store) {
	t.Helper()
	at := time.Date(2026, time.July, 19, 10, 0, 0, 0, time.UTC)

	dsA := dataset.Dataset{ID: dataset.NewID(), Slug: "conformance-ds-lifecycle-a", Title: "A", CreatedAt: at}
	dsB := dataset.Dataset{ID: dataset.NewID(), Slug: "conformance-ds-lifecycle-b", Title: "B", CreatedAt: at.Add(time.Second)}
	core.RequireTrue(t, st.DatasetCreate(dsA).OK)
	core.RequireTrue(t, st.DatasetCreate(dsB).OK)

	core.AssertFalse(t, st.DatasetCreate(dataset.Dataset{ID: dsA.ID, Slug: "conformance-ds-lifecycle-c", Title: "dup id", CreatedAt: at}).OK, "duplicate id must fail")
	core.AssertFalse(t, st.DatasetCreate(dataset.Dataset{ID: dataset.NewID(), Slug: dsA.Slug, Title: "dup slug", CreatedAt: at}).OK, "duplicate slug must fail")

	got := core.MustCast[dataset.Dataset](st.Dataset(dsA.ID))
	core.AssertEqual(t, dsA.Slug, got.Slug)
	got = core.MustCast[dataset.Dataset](st.DatasetBySlug(dsB.Slug))
	core.AssertEqual(t, dsB.ID, got.ID)

	notFound := st.Dataset("missing-" + dataset.NewID())
	core.AssertFalse(t, notFound.OK)
	core.AssertEqual(t, "dataset.dataset.notfound", notFound.Code())

	list := core.MustCast[[]dataset.Dataset](st.Datasets(false))
	idxA, idxB := indexOfDataset(list, dsA.ID), indexOfDataset(list, dsB.ID)
	core.AssertTrue(t, idxA >= 0 && idxB >= 0, "both datasets must be listed")
	core.AssertTrue(t, idxA < idxB, "created_at ordering: dsA before dsB")

	archived := core.MustCast[dataset.Dataset](st.DatasetArchive(dsA.ID))
	core.AssertTrue(t, archived.Archived)
	core.AssertFalse(t, archived.ArchivedAt.IsZero())
	core.RequireTrue(t, st.DatasetArchive(dsA.ID).OK, "archiving twice must be idempotent")

	listAfter := core.MustCast[[]dataset.Dataset](st.Datasets(false))
	core.AssertTrue(t, indexOfDataset(listAfter, dsA.ID) < 0, "archived dataset excluded by default")
	listAll := core.MustCast[[]dataset.Dataset](st.Datasets(true))
	core.AssertTrue(t, indexOfDataset(listAll, dsA.ID) >= 0, "includeArchived surfaces the archived dataset")

	core.AssertFalse(t, st.DatasetArchive("missing-"+dataset.NewID()).OK, "archiving an unknown dataset must fail")
}

func testItemLifecycleAndFilters(t *testing.T, st dataset.Store) {
	t.Helper()
	at := time.Date(2026, time.July, 19, 11, 0, 0, 0, time.UTC)
	ds := dataset.Dataset{ID: dataset.NewID(), Slug: "conformance-items", Title: "Items", CreatedAt: at}
	core.RequireTrue(t, st.DatasetCreate(ds).OK)

	item1 := conformancePairItem(ds.ID, "hi", "hello-1", at.Add(time.Second))
	item2 := conformancePairItem(ds.ID, "hi", "hello-2", at.Add(2*time.Second))
	core.RequireTrue(t, st.ItemAppend(item1).OK)
	core.RequireTrue(t, st.ItemAppend(item2).OK)

	core.AssertFalse(t, st.ItemAppend(conformancePairItem("missing-"+dataset.NewID(), "x", "y", at)).OK, "unknown dataset id must fail")
	core.AssertFalse(t, st.ItemAppend(item1).OK, "duplicate item id must fail")

	got := core.MustCast[dataset.Item](st.Item(item1.ID))
	core.AssertEqual(t, item1.ContentHash, got.ContentHash)

	notFound := st.Item("missing-" + dataset.NewID())
	core.AssertFalse(t, notFound.OK)
	core.AssertEqual(t, "dataset.item.notfound", notFound.Code())

	core.AssertFalse(t, st.Items(dataset.ItemFilter{}).OK, "a filter with no dataset id must fail")

	list := core.MustCast[[]dataset.Item](st.Items(dataset.ItemFilter{DatasetID: ds.ID}))
	core.AssertLen(t, list, 2)
	core.AssertEqual(t, item1.ID, list[0].ID, "created_at ordering: earlier item first")

	core.AssertLen(t, core.MustCast[[]dataset.Item](st.Items(dataset.ItemFilter{DatasetID: ds.ID, Kind: dataset.KindMessages})), 0, "no messages-kind items exist")
	core.AssertLen(t, core.MustCast[[]dataset.Item](st.Items(dataset.ItemFilter{DatasetID: ds.ID, ContentHash: item1.ContentHash})), 1)
	core.AssertLen(t, core.MustCast[[]dataset.Item](st.Items(dataset.ItemFilter{DatasetID: ds.ID, Source: dataset.SourceCaptureServe})), 0, "no capture:serve items exist")
	core.AssertLen(t, core.MustCast[[]dataset.Item](st.Items(dataset.ItemFilter{DatasetID: ds.ID, Source: item1.Source})), 2)

	core.AssertLen(t, core.MustCast[[]dataset.Item](st.Items(dataset.ItemFilter{DatasetID: ds.ID, Status: dataset.StatusPending})), 2, "unreviewed items are implicitly pending")
	core.AssertLen(t, core.MustCast[[]dataset.Item](st.Items(dataset.ItemFilter{DatasetID: ds.ID, Status: dataset.StatusApproved})), 0, "no reviews yet — nothing approved")
	core.RequireTrue(t, st.ReviewAppend(dataset.Review{ItemID: item1.ID, Status: dataset.StatusApproved, Reviewer: "snider", CreatedAt: at.Add(10 * time.Second)}).OK)
	core.AssertLen(t, core.MustCast[[]dataset.Item](st.Items(dataset.ItemFilter{DatasetID: ds.ID, Status: dataset.StatusApproved})), 1, "the approved item now matches")
	core.AssertLen(t, core.MustCast[[]dataset.Item](st.Items(dataset.ItemFilter{DatasetID: ds.ID, Status: dataset.StatusPending})), 1, "the other item is still pending")

	archivedItem := conformancePairItem(ds.ID, "hi", "hello-archived", at.Add(20*time.Second))
	archivedItem.Archived = true
	archivedItem.ArchivedAt = at.Add(21 * time.Second)
	core.RequireTrue(t, st.ItemAppend(archivedItem).OK)
	list = core.MustCast[[]dataset.Item](st.Items(dataset.ItemFilter{DatasetID: ds.ID}))
	for _, it := range list {
		core.AssertNotEqual(t, archivedItem.ID, it.ID, "archived item excluded by default")
	}
	listAll := core.MustCast[[]dataset.Item](st.Items(dataset.ItemFilter{DatasetID: ds.ID, IncludeArchived: true}))
	core.AssertLen(t, listAll, 3, "IncludeArchived surfaces the archived item too")
}

func testScoreAppendAndExpressionMatching(t *testing.T, st dataset.Store) {
	t.Helper()
	at := time.Date(2026, time.July, 19, 12, 0, 0, 0, time.UTC)
	ds := dataset.Dataset{ID: dataset.NewID(), Slug: "conformance-scores", Title: "Scores", CreatedAt: at}
	core.RequireTrue(t, st.DatasetCreate(ds).OK)
	item := conformancePairItem(ds.ID, "hi", "hello", at.Add(time.Second))
	core.RequireTrue(t, st.ItemAppend(item).OK)

	core.AssertFalse(t, st.ScoreAppend(conformanceScore("missing-"+dataset.NewID(), dataset.ScoreKindLEK, 80, at)).OK, "unknown item id must fail")
	invalid := conformanceScore(item.ID, dataset.ScoreKindLEK, 80, at)
	invalid.ScorerName = ""
	core.AssertFalse(t, st.ScoreAppend(invalid).OK, "an invalid score must fail validation before touching storage")

	core.RequireTrue(t, st.ScoreAppend(conformanceScore(item.ID, dataset.ScoreKindLEK, 40, at.Add(2*time.Second))).OK)
	core.RequireTrue(t, st.ScoreAppend(conformanceScore(item.ID, dataset.ScoreKindLEK, 90, at.Add(3*time.Second))).OK)
	scores := core.MustCast[[]dataset.Score](st.Scores(item.ID))
	core.AssertLen(t, scores, 2)
	core.AssertEqual(t, 40.0, scores[0].Value, "oldest first")

	core.AssertFalse(t, st.Scores("missing-"+dataset.NewID()).OK, "scores for an unknown item must fail")

	// ScoreExpression latest-wins, exercised through Items(Score: expr) —
	// the only Store-level surface for ScoreExpression.Matches: the
	// LATEST lek score (90) satisfies >=80 even though an earlier score
	// (40) does not.
	matching := core.MustCast[[]dataset.Item](st.Items(dataset.ItemFilter{
		DatasetID: ds.ID, Score: &dataset.ScoreExpression{Kind: dataset.ScoreKindLEK, Op: dataset.OpGTE, Threshold: 80},
	}))
	core.AssertLen(t, matching, 1, "the latest lek score (90) satisfies >=80")
	notMatching := core.MustCast[[]dataset.Item](st.Items(dataset.ItemFilter{
		DatasetID: ds.ID, Score: &dataset.ScoreExpression{Kind: dataset.ScoreKindLEK, Op: dataset.OpGTE, Threshold: 95},
	}))
	core.AssertLen(t, notMatching, 0, "95 exceeds even the latest score")
	noSycophancy := core.MustCast[[]dataset.Item](st.Items(dataset.ItemFilter{
		DatasetID: ds.ID, Score: &dataset.ScoreExpression{Kind: dataset.ScoreKindSycophancy, Op: dataset.OpGTE, Threshold: 0},
	}))
	core.AssertLen(t, noSycophancy, 0, "an absent score kind never matches, even threshold >=0")

	// Tie-break: two scores at the SAME created_at — the later-appended
	// (second ScoreAppend call) value must win, matching
	// dataset.ScoreExpression.Matches' insertion-order tie-break rule.
	// A naive "ORDER BY created_at DESC LIMIT 1" SQL implementation with
	// no tiebreak column would get this non-deterministic.
	item2 := conformancePairItem(ds.ID, "hi", "hello-2", at.Add(4*time.Second))
	core.RequireTrue(t, st.ItemAppend(item2).OK)
	tie := at.Add(5 * time.Second)
	core.RequireTrue(t, st.ScoreAppend(conformanceScore(item2.ID, dataset.ScoreKindLEK, 10, tie)).OK)
	core.RequireTrue(t, st.ScoreAppend(conformanceScore(item2.ID, dataset.ScoreKindLEK, 90, tie)).OK)
	tieMatch := core.MustCast[[]dataset.Item](st.Items(dataset.ItemFilter{
		DatasetID: ds.ID, Score: &dataset.ScoreExpression{Kind: dataset.ScoreKindLEK, Op: dataset.OpEQ, Threshold: 90},
	}))
	found := false
	for _, it := range tieMatch {
		if it.ID == item2.ID {
			found = true
		}
	}
	core.AssertTrue(t, found, "on a created_at tie the later-inserted score (90) must win, matching MemoryStore's append-order rule")
}

func testReviewLatestWins(t *testing.T, st dataset.Store) {
	t.Helper()
	at := time.Date(2026, time.July, 19, 13, 0, 0, 0, time.UTC)
	ds := dataset.Dataset{ID: dataset.NewID(), Slug: "conformance-reviews", Title: "Reviews", CreatedAt: at}
	core.RequireTrue(t, st.DatasetCreate(ds).OK)
	item := conformancePairItem(ds.ID, "hi", "hello", at.Add(time.Second))
	core.RequireTrue(t, st.ItemAppend(item).OK)
	other := conformancePairItem(ds.ID, "hi", "hello-other", at.Add(2*time.Second))
	core.RequireTrue(t, st.ItemAppend(other).OK)

	pending := core.MustCast[dataset.Review](st.ReviewLatest(item.ID))
	core.AssertEqual(t, dataset.StatusPending, pending.Status, "pending is the implicit starting state")

	core.AssertFalse(t, st.ReviewAppend(dataset.Review{ItemID: "missing-" + dataset.NewID(), Status: dataset.StatusApproved, Reviewer: "snider", CreatedAt: at}).OK, "unknown item id must fail")
	core.AssertFalse(t, st.ReviewLatest("missing-"+dataset.NewID()).OK, "reviewing an unknown item must fail")

	core.RequireTrue(t, st.ReviewAppend(dataset.Review{ItemID: item.ID, Status: dataset.StatusQuarantined, Reviewer: "auto:welfare", CreatedAt: at.Add(3 * time.Second)}).OK)
	core.RequireTrue(t, st.ReviewAppend(dataset.Review{ItemID: item.ID, Status: dataset.StatusApproved, Reviewer: "snider", CreatedAt: at.Add(4 * time.Second)}).OK)
	latest := core.MustCast[dataset.Review](st.ReviewLatest(item.ID))
	core.AssertEqual(t, dataset.StatusApproved, latest.Status, "the later-timestamped review wins")

	core.RequireTrue(t, st.ReviewAppend(dataset.Review{ItemID: other.ID, Status: dataset.StatusRejected, Reviewer: "snider", CreatedAt: at.Add(8 * time.Second)}).OK)

	// Tie-break: two reviews at the SAME created_at — the later-appended
	// (second ReviewAppend call) status must win, matching
	// MemoryStore.latestReviewLocked's insertion-order rule. Review
	// carries no ID in the domain model, so this is the one place a
	// naive SQL implementation (ORDER BY created_at DESC LIMIT 1, no
	// tiebreak) would be genuinely non-deterministic rather than merely
	// wrong.
	tie := at.Add(9 * time.Second)
	core.RequireTrue(t, st.ReviewAppend(dataset.Review{ItemID: item.ID, Status: dataset.StatusRejected, Reviewer: "auto:threshold", CreatedAt: tie}).OK)
	core.RequireTrue(t, st.ReviewAppend(dataset.Review{ItemID: item.ID, Status: dataset.StatusApproved, Reviewer: "snider", CreatedAt: tie}).OK)
	tieLatest := core.MustCast[dataset.Review](st.ReviewLatest(item.ID))
	core.AssertEqual(t, dataset.StatusApproved, tieLatest.Status, "on a created_at tie the later-inserted review wins")

	otherLatest := core.MustCast[dataset.Review](st.ReviewLatest(other.ID))
	core.AssertEqual(t, dataset.StatusRejected, otherLatest.Status, "the other item's own review is unaffected by item's tie-break append")
}

func testExportLifecycle(t *testing.T, st dataset.Store) {
	t.Helper()
	at := time.Date(2026, time.July, 19, 14, 0, 0, 0, time.UTC)
	ds := dataset.Dataset{ID: dataset.NewID(), Slug: "conformance-exports", Title: "Exports", CreatedAt: at}
	core.RequireTrue(t, st.DatasetCreate(ds).OK)

	exp1 := dataset.Export{ID: dataset.NewID(), DatasetID: ds.ID, Format: dataset.FormatPairsJSONL, OutputPath: "/tmp/a.jsonl", ManifestHash: "hash-a", CreatedAt: at.Add(2 * time.Second)}
	exp2 := dataset.Export{ID: dataset.NewID(), DatasetID: ds.ID, Format: dataset.FormatSFTJSONL, OutputPath: "/tmp/b.jsonl", ManifestHash: "hash-b", CreatedAt: at.Add(time.Second)}
	core.RequireTrue(t, st.ExportAppend(exp1).OK)
	core.RequireTrue(t, st.ExportAppend(exp2).OK)

	core.AssertFalse(t, st.ExportAppend(dataset.Export{ID: dataset.NewID(), DatasetID: "missing-" + dataset.NewID(), Format: dataset.FormatPairsJSONL, OutputPath: "/tmp/c.jsonl", ManifestHash: "hash-c", CreatedAt: at}).OK, "unknown dataset id must fail")

	// Unlike DatasetCreate/ItemAppend, ExportAppend's interface doc only
	// promises "fails if DatasetID does not exist" — no id-uniqueness
	// clause — and MemoryStore's implementation has no id-collision
	// check either. Re-appending the same Export id must succeed on both
	// backends (a real conformance finding, not an assumption).
	redundant := core.MustCast[dataset.Export](st.ExportAppend(exp1))
	core.AssertEqual(t, exp1.ID, redundant.ID)

	list := core.MustCast[[]dataset.Export](st.Exports(ds.ID))
	core.AssertLen(t, list, 3, "ExportAppend does not enforce id uniqueness, matching MemoryStore")
	core.AssertEqual(t, exp2.ID, list[0].ID, "oldest first")

	core.AssertFalse(t, st.Exports("missing-"+dataset.NewID()).OK, "exports for an unknown dataset must fail")
}

func testNotFoundCodes(t *testing.T, st dataset.Store) {
	t.Helper()
	at := time.Date(2026, time.July, 19, 15, 0, 0, 0, time.UTC)
	cases := []struct {
		name string
		code string
		r    core.Result
	}{
		{"Dataset", "dataset.dataset.notfound", st.Dataset("missing-" + dataset.NewID())},
		{"DatasetBySlug", "dataset.dataset.notfound", st.DatasetBySlug("missing-slug-" + dataset.NewID())},
		{"DatasetArchive", "dataset.dataset.notfound", st.DatasetArchive("missing-" + dataset.NewID())},
		{"ItemAppend", "dataset.dataset.notfound", st.ItemAppend(conformancePairItem("missing-"+dataset.NewID(), "x", "y", at))},
		{"Item", "dataset.item.notfound", st.Item("missing-" + dataset.NewID())},
		{"Scores", "dataset.item.notfound", st.Scores("missing-" + dataset.NewID())},
		{"ScoreAppend", "dataset.item.notfound", st.ScoreAppend(conformanceScore("missing-"+dataset.NewID(), dataset.ScoreKindLEK, 1, at))},
		{"ReviewLatest", "dataset.item.notfound", st.ReviewLatest("missing-" + dataset.NewID())},
		{"ReviewAppend", "dataset.item.notfound", st.ReviewAppend(dataset.Review{ItemID: "missing-" + dataset.NewID(), Status: dataset.StatusApproved, Reviewer: "snider", CreatedAt: at})},
		{"Exports", "dataset.dataset.notfound", st.Exports("missing-" + dataset.NewID())},
		{"ExportAppend", "dataset.dataset.notfound", st.ExportAppend(dataset.Export{ID: dataset.NewID(), DatasetID: "missing-" + dataset.NewID(), Format: dataset.FormatPairsJSONL, OutputPath: "/tmp/x.jsonl", ManifestHash: "h", CreatedAt: at})},
	}
	for _, c := range cases {
		core.AssertFalse(t, c.r.OK, c.name+" must fail")
		core.AssertEqual(t, c.code, c.r.Code(), c.name+" error code")
	}
}

// ---- temp-root isolation ----

// TestDuckDatasetStore_TempRootIsolation proves two DuckDB dataset stores
// opened at different roots never share state — a fresh t.TempDir() per
// store is a real isolation boundary — and that datasets.duckdb lands at
// the path paths.go resolves without ever touching lem.duckdb (the
// design's "separate file" decision, checked structurally, not just by
// naming convention).
func TestDuckDatasetStore_TempRootIsolation(t *testing.T) {
	rootA := t.TempDir()
	rootB := t.TempDir()

	pathsA := core.MustCast[appPaths](appPathsAt(rootA))
	pathsB := core.MustCast[appPaths](appPathsAt(rootB))
	core.AssertNotEqual(t, pathsA.Datasets, pathsB.Datasets, "each root resolves its own datasets.duckdb path")

	storeA := newTestDuckDatasetStore(t, pathsA.Datasets)
	defer closeTestDuckDatasetStore(t, storeA)
	storeB := newTestDuckDatasetStore(t, pathsB.Datasets)
	defer closeTestDuckDatasetStore(t, storeB)

	if _, err := os.Stat(pathsA.Datasets); err != nil {
		t.Fatalf("datasets.duckdb was not created at the resolved path: %v", err)
	}
	if _, err := os.Stat(pathsB.Datasets); err != nil {
		t.Fatalf("datasets.duckdb was not created at the resolved path: %v", err)
	}

	at := time.Date(2026, time.July, 19, 16, 0, 0, 0, time.UTC)
	core.RequireTrue(t, storeA.DatasetCreate(dataset.Dataset{ID: dataset.NewID(), Slug: "isolation-only-in-a", Title: "A", CreatedAt: at}).OK)

	core.AssertFalse(t, storeB.DatasetBySlug("isolation-only-in-a").OK, "store B must not see store A's dataset")
	core.AssertLen(t, core.MustCast[[]dataset.Dataset](storeA.Datasets(false)), 1)
	core.AssertLen(t, core.MustCast[[]dataset.Dataset](storeB.Datasets(false)), 0, "store B's dataset list must stay empty")

	if _, err := os.Stat(pathsA.Database); !os.IsNotExist(err) {
		t.Fatalf("opening the dataset store must not create lem.duckdb: stat err = %v", err)
	}
}

// ---- OpenDatasetStore: the CLI-facing entry point (cli/data.go, the
// serve/ssd capture taps) ----

// TestOpenDatasetStore_Good proves the exported entry point resolves the
// $HOME/.lem root (no --home flag surface — HOME is the only seam,
// exactly as cli/serve_test.go redirects the admin-token path), creates
// datasets.duckdb there, and that a second open against the SAME HOME
// reopens the same file rather than a fresh one (migrations are
// idempotent on reopen; a dataset created in the first open is still
// there in the second).
func TestOpenDatasetStore_Good(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)

	first := OpenDatasetStore()
	core.RequireTrue(t, first.OK, "OpenDatasetStore first open")
	store1, ok := first.Value.(DatasetStore)
	if !ok {
		t.Fatalf("OpenDatasetStore value = %T, want DatasetStore", first.Value)
	}

	at := time.Date(2026, time.July, 19, 17, 0, 0, 0, time.UTC)
	core.RequireTrue(t, store1.DatasetCreate(dataset.Dataset{ID: dataset.NewID(), Slug: "reopen-proof", Title: "Reopen proof", CreatedAt: at}).OK)
	core.RequireTrue(t, store1.Close().OK)

	wantPath := core.Path(home, ".lem", "datasets.duckdb")
	if _, err := os.Stat(wantPath); err != nil {
		t.Fatalf("datasets.duckdb missing at the resolved default path %s: %v", wantPath, err)
	}

	second := OpenDatasetStore()
	core.RequireTrue(t, second.OK, "OpenDatasetStore second open")
	store2, ok := second.Value.(DatasetStore)
	if !ok {
		t.Fatalf("OpenDatasetStore value = %T, want DatasetStore", second.Value)
	}
	defer func() { core.RequireTrue(t, store2.Close().OK) }()

	core.RequireTrue(t, store2.DatasetBySlug("reopen-proof").OK, "the dataset created through the first handle must survive a reopen")
}

// TestOpenDatasetStore_Bad proves a root that cannot be prepared (HOME
// points at a regular file, so $HOME/.lem cannot be created) fails
// closed rather than opening a bogus store — mirrors
// TestAppFiles_Ugly's fixture trick one layer up.
func TestOpenDatasetStore_Bad(t *testing.T) {
	blocker := core.Path(t.TempDir(), "home-is-a-file")
	if err := os.WriteFile(blocker, []byte("not a directory"), 0o600); err != nil {
		t.Fatalf("write blocker fixture: %v", err)
	}
	t.Setenv("HOME", blocker)

	result := OpenDatasetStore()
	core.AssertFalse(t, result.OK, "OpenDatasetStore over a blocked root must fail")
}
