// SPDX-Licence-Identifier: EUPL-1.2

package dataset

import (
	core "dappco.re/go"
)

// ---- fixtures ----

func fixturePairContent(prompt, response string) []byte {
	return core.JSONMarshal(PairContent{Prompt: prompt, Response: response}).Value.([]byte)
}

func fixtureDataset(id, slug string) Dataset {
	return Dataset{ID: id, Slug: slug, Title: "Test dataset " + slug, CreatedAt: timeAt(1000)}
}

func fixtureItem(id, datasetID string, at int64) Item {
	content := fixturePairContent("hi", "hello-"+id)
	hash := ContentHash(KindPair, content).Value.(string)
	return Item{
		ID: id, DatasetID: datasetID, Kind: KindPair, Content: content,
		Source: SourceImportJSONL, ContentHash: hash, CreatedAt: timeAt(at),
	}
}

func fixtureScore(id, itemID string, kind ScoreKind, value float64, at int64) Score {
	return Score{ID: id, ItemID: itemID, Kind: kind, Value: value, ScorerName: "test-scorer", ScorerVersion: "1", CreatedAt: timeAt(at)}
}

func fixtureReview(itemID string, status ReviewStatus, at int64) Review {
	return Review{ItemID: itemID, Status: status, Reviewer: "snider", CreatedAt: timeAt(at)}
}

func fixtureExport(id, datasetID string, at int64) Export {
	return Export{ID: id, DatasetID: datasetID, Format: FormatPairsJSONL, OutputPath: "/tmp/out.jsonl", ManifestHash: "abc123", CreatedAt: timeAt(at)}
}

// seededStore returns a MemoryStore with one dataset ("ds-1") and one
// item ("item-1") already present — the common starting point most
// Store method tests build on.
func seededStore(t *core.T) (*MemoryStore, Dataset, Item) {
	store := NewMemoryStore()
	ds := fixtureDataset("ds-1", "evening-vents")
	core.RequireTrue(t, store.DatasetCreate(ds).OK)
	item := fixtureItem("item-1", ds.ID, 2000)
	core.RequireTrue(t, store.ItemAppend(item).OK)
	return store, ds, item
}

// ---- ScoreExpression ----

func TestKnownComparisonOp(t *core.T) {
	for _, op := range []ComparisonOp{OpGTE, OpLTE, OpGT, OpLT, OpEQ, OpNEQ} {
		core.AssertTrue(t, knownComparisonOp(op), "known op must report true: "+string(op))
	}
	core.AssertFalse(t, knownComparisonOp(ComparisonOp("~=")), "unknown op must report false")
}

func TestScoreExpression_String_Good(t *core.T) {
	e := ScoreExpression{Kind: ScoreKindLEK, Op: OpGTE, Threshold: 80}
	core.AssertEqual(t, "lek>=80", e.String())
}

func TestScoreExpression_Matches_Good(t *core.T) {
	scores := []Score{
		fixtureScore("s1", "item-1", ScoreKindLEK, 60, 1000),
		fixtureScore("s2", "item-1", ScoreKindLEK, 85, 2000), // latest lek
		fixtureScore("s3", "item-1", ScoreKindHostility, 0.9, 1500),
	}
	expr := ScoreExpression{Kind: ScoreKindLEK, Op: OpGTE, Threshold: 80}
	core.AssertTrue(t, expr.Matches(scores), "the latest lek score (85) satisfies >=80")

	hostileExpr := ScoreExpression{Kind: ScoreKindHostility, Op: OpLT, Threshold: 0.5}
	core.AssertFalse(t, hostileExpr.Matches(scores), "hostility 0.9 does not satisfy <0.5")

	core.AssertTrue(t, (ScoreExpression{Kind: ScoreKindLEK, Op: OpLTE, Threshold: 85}).Matches(scores), "85 <= 85")
	core.AssertTrue(t, (ScoreExpression{Kind: ScoreKindLEK, Op: OpGT, Threshold: 80}).Matches(scores), "85 > 80")
	core.AssertTrue(t, (ScoreExpression{Kind: ScoreKindHostility, Op: OpNEQ, Threshold: 0}).Matches(scores), "0.9 != 0")
}

func TestScoreExpression_Matches_Bad(t *core.T) {
	scores := []Score{fixtureScore("s1", "item-1", ScoreKindLEK, 60, 1000)}
	expr := ScoreExpression{Kind: ScoreKindSycophancy, Op: OpGTE, Threshold: 0}
	core.AssertFalse(t, expr.Matches(scores), "an absent score kind never matches, even threshold >=0")
}

func TestScoreExpression_Matches_Ugly(t *core.T) {
	core.AssertFalse(t, ScoreExpression{Kind: ScoreKindLEK, Op: OpGTE, Threshold: 0}.Matches(nil), "no scores at all never matches")

	// Ties on created_at: the later slice entry (insertion order) wins.
	tie := timeAt(5000)
	scores := []Score{
		{ID: "a", ItemID: "item-1", Kind: ScoreKindLEK, Value: 10, ScorerName: "x", CreatedAt: tie},
		{ID: "b", ItemID: "item-1", Kind: ScoreKindLEK, Value: 90, ScorerName: "x", CreatedAt: tie},
	}
	core.AssertTrue(t, (ScoreExpression{Kind: ScoreKindLEK, Op: OpEQ, Threshold: 90}).Matches(scores), "on a created_at tie the later-inserted score wins")

	unknownOp := ScoreExpression{Kind: ScoreKindLEK, Op: ComparisonOp("~="), Threshold: 0}
	core.AssertFalse(t, unknownOp.Matches([]Score{fixtureScore("s1", "item-1", ScoreKindLEK, 60, 1000)}), "an unknown operator never matches")
}

// ---- Validate* ----

func TestValidateDataset_Good(t *core.T) {
	r := ValidateDataset(fixtureDataset("ds-1", "evening-vents"))
	core.AssertTrue(t, r.OK)
}

func TestValidateDataset_Bad(t *core.T) {
	core.AssertFalse(t, ValidateDataset(Dataset{Slug: "x", Title: "t", CreatedAt: timeAt(1)}).OK, "missing id must fail")
	core.AssertFalse(t, ValidateDataset(Dataset{ID: "d", Title: "t", CreatedAt: timeAt(1)}).OK, "missing slug must fail")
	core.AssertFalse(t, ValidateDataset(Dataset{ID: "d", Slug: "x", CreatedAt: timeAt(1)}).OK, "missing title must fail")
	core.AssertFalse(t, ValidateDataset(Dataset{ID: "d", Slug: "x", Title: "t"}).OK, "missing created_at must fail")
}

func TestValidateDataset_Ugly(t *core.T) {
	core.AssertFalse(t, ValidateDataset(Dataset{ID: "d", Slug: "Not-Lower", Title: "t", CreatedAt: timeAt(1)}).OK, "uppercase slug must fail")
	core.AssertFalse(t, ValidateDataset(Dataset{ID: "d", Slug: "-leading", Title: "t", CreatedAt: timeAt(1)}).OK, "leading hyphen slug must fail")
	core.AssertFalse(t, ValidateDataset(Dataset{ID: "d", Slug: "trailing-", Title: "t", CreatedAt: timeAt(1)}).OK, "trailing hyphen slug must fail")
	core.AssertFalse(t, ValidateDataset(Dataset{ID: "d", Slug: "has space", Title: "t", CreatedAt: timeAt(1)}).OK, "space in slug must fail")
}

func TestValidateItem_Good(t *core.T) {
	r := ValidateItem(fixtureItem("item-1", "ds-1", 2000))
	core.AssertTrue(t, r.OK)
}

func TestValidateItem_Bad(t *core.T) {
	item := fixtureItem("item-1", "ds-1", 2000)

	noID := item
	noID.ID = ""
	core.AssertFalse(t, ValidateItem(noID).OK, "missing id must fail")

	noDataset := item
	noDataset.DatasetID = ""
	core.AssertFalse(t, ValidateItem(noDataset).OK, "missing dataset id must fail")

	noHash := item
	noHash.ContentHash = ""
	core.AssertFalse(t, ValidateItem(noHash).OK, "missing content hash must fail")

	noCreated := item
	noCreated.CreatedAt = timeZero
	core.AssertFalse(t, ValidateItem(noCreated).OK, "missing created_at must fail")
}

func TestValidateItem_Ugly(t *core.T) {
	item := fixtureItem("item-1", "ds-1", 2000)

	badKind := item
	badKind.Kind = ItemKind("bogus")
	core.AssertFalse(t, ValidateItem(badKind).OK, "unknown kind must fail")

	badSource := item
	badSource.Source = ItemSource("bogus")
	core.AssertFalse(t, ValidateItem(badSource).OK, "unknown source must fail")

	badContent := item
	badContent.Content = []byte(`{}`) // valid JSON, invalid pair shape
	core.AssertFalse(t, ValidateItem(badContent).OK, "content that fails ValidateItemContent must fail")
}

func TestValidateScore_Good(t *core.T) {
	r := ValidateScore(fixtureScore("s1", "item-1", ScoreKindLEK, 80, 1000))
	core.AssertTrue(t, r.OK)
	r = ValidateScore(fixtureScore("s2", "item-1", JudgeScoreKind("helpfulness"), 80, 1000))
	core.AssertTrue(t, r.OK, "a well-formed judge kind must validate")
}

func TestValidateScore_Bad(t *core.T) {
	s := fixtureScore("s1", "item-1", ScoreKindLEK, 80, 1000)

	noID := s
	noID.ID = ""
	core.AssertFalse(t, ValidateScore(noID).OK)

	noItem := s
	noItem.ItemID = ""
	core.AssertFalse(t, ValidateScore(noItem).OK)

	noScorer := s
	noScorer.ScorerName = ""
	core.AssertFalse(t, ValidateScore(noScorer).OK)
}

func TestValidateScore_Ugly(t *core.T) {
	s := fixtureScore("s1", "item-1", ScoreKind("bogus"), 80, 1000)
	core.AssertFalse(t, ValidateScore(s).OK, "unknown kind must fail")

	s = fixtureScore("s2", "item-1", ScoreKind("judge:"), 80, 1000)
	core.AssertFalse(t, ValidateScore(s).OK, "a bare judge prefix must fail")

	noCreated := fixtureScore("s3", "item-1", ScoreKindLEK, 80, 1000)
	noCreated.CreatedAt = timeZero
	core.AssertFalse(t, ValidateScore(noCreated).OK)
}

func TestValidateReview_Good(t *core.T) {
	r := ValidateReview(fixtureReview("item-1", StatusApproved, 1000))
	core.AssertTrue(t, r.OK)
}

func TestValidateReview_Bad(t *core.T) {
	noItem := fixtureReview("", StatusApproved, 1000)
	core.AssertFalse(t, ValidateReview(noItem).OK)

	noReviewer := fixtureReview("item-1", StatusApproved, 1000)
	noReviewer.Reviewer = ""
	core.AssertFalse(t, ValidateReview(noReviewer).OK)
}

func TestValidateReview_Ugly(t *core.T) {
	bad := fixtureReview("item-1", ReviewStatus("bogus"), 1000)
	core.AssertFalse(t, ValidateReview(bad).OK, "unknown status must fail")

	noCreated := fixtureReview("item-1", StatusApproved, 1000)
	noCreated.CreatedAt = timeZero
	core.AssertFalse(t, ValidateReview(noCreated).OK)
}

func TestValidateExport_Good(t *core.T) {
	r := ValidateExport(fixtureExport("exp-1", "ds-1", 1000))
	core.AssertTrue(t, r.OK)
}

func TestValidateExport_Bad(t *core.T) {
	noID := fixtureExport("exp-1", "ds-1", 1000)
	noID.ID = ""
	core.AssertFalse(t, ValidateExport(noID).OK)

	noDataset := fixtureExport("exp-1", "ds-1", 1000)
	noDataset.DatasetID = ""
	core.AssertFalse(t, ValidateExport(noDataset).OK)

	noPath := fixtureExport("exp-1", "ds-1", 1000)
	noPath.OutputPath = ""
	core.AssertFalse(t, ValidateExport(noPath).OK)

	noManifest := fixtureExport("exp-1", "ds-1", 1000)
	noManifest.ManifestHash = ""
	core.AssertFalse(t, ValidateExport(noManifest).OK)
}

func TestValidateExport_Ugly(t *core.T) {
	badFormat := fixtureExport("exp-1", "ds-1", 1000)
	badFormat.Format = ExportFormat("bogus")
	core.AssertFalse(t, ValidateExport(badFormat).OK, "unknown format must fail")

	noCreated := fixtureExport("exp-1", "ds-1", 1000)
	noCreated.CreatedAt = timeZero
	core.AssertFalse(t, ValidateExport(noCreated).OK)
}

// ---- MemoryStore: Dataset ----

func TestMemoryStore_DatasetCreate_Good(t *core.T) {
	store := NewMemoryStore()
	r := store.DatasetCreate(fixtureDataset("ds-1", "evening-vents"))
	core.AssertTrue(t, r.OK)
	core.AssertEqual(t, "ds-1", r.Value.(Dataset).ID)
}

func TestMemoryStore_DatasetCreate_Bad(t *core.T) {
	store := NewMemoryStore()
	core.RequireTrue(t, store.DatasetCreate(fixtureDataset("ds-1", "evening-vents")).OK)

	r := store.DatasetCreate(fixtureDataset("ds-1", "different-slug"))
	core.AssertFalse(t, r.OK, "duplicate id must fail")

	r = store.DatasetCreate(fixtureDataset("ds-2", "evening-vents"))
	core.AssertFalse(t, r.OK, "duplicate slug must fail")
}

func TestMemoryStore_DatasetCreate_Ugly(t *core.T) {
	store := NewMemoryStore()
	r := store.DatasetCreate(Dataset{})
	core.AssertFalse(t, r.OK, "an invalid dataset must fail validation before touching storage")
}

func TestMemoryStore_Dataset_Good(t *core.T) {
	store, ds, _ := seededStore(t)
	r := store.Dataset(ds.ID)
	core.AssertTrue(t, r.OK)
	core.AssertEqual(t, ds.Slug, r.Value.(Dataset).Slug)
}

func TestMemoryStore_Dataset_Bad(t *core.T) {
	store, _, _ := seededStore(t)
	r := store.Dataset("missing")
	core.AssertFalse(t, r.OK)
	core.AssertEqual(t, "dataset.dataset.notfound", r.Code())
}

func TestMemoryStore_DatasetBySlug_Good(t *core.T) {
	store, ds, _ := seededStore(t)
	r := store.DatasetBySlug(ds.Slug)
	core.AssertTrue(t, r.OK)
	core.AssertEqual(t, ds.ID, r.Value.(Dataset).ID)
}

func TestMemoryStore_DatasetBySlug_Bad(t *core.T) {
	store, _, _ := seededStore(t)
	r := store.DatasetBySlug("missing-slug")
	core.AssertFalse(t, r.OK)
}

func TestMemoryStore_Datasets_Good(t *core.T) {
	store := NewMemoryStore()
	a := fixtureDataset("ds-a", "a")
	a.CreatedAt = timeAt(200)
	b := fixtureDataset("ds-b", "b")
	b.CreatedAt = timeAt(100)
	core.RequireTrue(t, store.DatasetCreate(a).OK)
	core.RequireTrue(t, store.DatasetCreate(b).OK)

	r := store.Datasets(false)
	core.AssertTrue(t, r.OK)
	list := r.Value.([]Dataset)
	core.AssertLen(t, list, 2)
	core.AssertEqual(t, "ds-b", list[0].ID, "created_at ordering: earlier dataset first")
	core.AssertEqual(t, "ds-a", list[1].ID)
}

func TestMemoryStore_Datasets_TieBreak(t *core.T) {
	store := NewMemoryStore()
	tie := timeAt(500)
	b := fixtureDataset("ds-b", "b")
	b.CreatedAt = tie
	a := fixtureDataset("ds-a", "a")
	a.CreatedAt = tie
	core.RequireTrue(t, store.DatasetCreate(b).OK)
	core.RequireTrue(t, store.DatasetCreate(a).OK)

	list := core.MustCast[[]Dataset](store.Datasets(false))
	core.AssertLen(t, list, 2)
	core.AssertEqual(t, "ds-a", list[0].ID, "on a created_at tie, ordering falls back to id")
	core.AssertEqual(t, "ds-b", list[1].ID)
}

func TestMemoryStore_Datasets_Bad(t *core.T) {
	store, ds, _ := seededStore(t)
	core.RequireTrue(t, store.DatasetArchive(ds.ID).OK)

	r := store.Datasets(false)
	core.AssertTrue(t, r.OK)
	core.AssertLen(t, r.Value.([]Dataset), 0, "an archived dataset is excluded by default")
}

func TestMemoryStore_Datasets_Ugly(t *core.T) {
	store, ds, _ := seededStore(t)
	core.RequireTrue(t, store.DatasetArchive(ds.ID).OK)

	r := store.Datasets(true)
	core.AssertTrue(t, r.OK)
	core.AssertLen(t, r.Value.([]Dataset), 1, "includeArchived=true must surface the archived dataset")
}

func TestMemoryStore_DatasetArchive_Good(t *core.T) {
	store, ds, _ := seededStore(t)
	r := store.DatasetArchive(ds.ID)
	core.AssertTrue(t, r.OK)
	archived := r.Value.(Dataset)
	core.AssertTrue(t, archived.Archived)
	core.AssertFalse(t, archived.ArchivedAt.IsZero())
}

func TestMemoryStore_DatasetArchive_Bad(t *core.T) {
	store, _, _ := seededStore(t)
	r := store.DatasetArchive("missing")
	core.AssertFalse(t, r.OK)
}

func TestMemoryStore_DatasetArchive_Ugly(t *core.T) {
	store, ds, _ := seededStore(t)
	core.RequireTrue(t, store.DatasetArchive(ds.ID).OK)
	// Archiving twice is idempotent, not an error.
	r := store.DatasetArchive(ds.ID)
	core.AssertTrue(t, r.OK)
}

// ---- MemoryStore: Item ----

func TestMemoryStore_ItemAppend_Good(t *core.T) {
	store, ds, _ := seededStore(t)
	item := fixtureItem("item-2", ds.ID, 3000)
	r := store.ItemAppend(item)
	core.AssertTrue(t, r.OK)
}

func TestMemoryStore_ItemAppend_Bad(t *core.T) {
	store, ds, item := seededStore(t)

	r := store.ItemAppend(item)
	core.AssertFalse(t, r.OK, "duplicate item id must fail")

	r = store.ItemAppend(fixtureItem("item-3", "missing-dataset", 3000))
	core.AssertFalse(t, r.OK, "unknown dataset id must fail")
	_ = ds
}

func TestMemoryStore_ItemAppend_Ugly(t *core.T) {
	store, ds, _ := seededStore(t)
	invalid := fixtureItem("item-4", ds.ID, 3000)
	invalid.Kind = ItemKind("bogus")
	r := store.ItemAppend(invalid)
	core.AssertFalse(t, r.OK, "an invalid item must fail validation before touching storage")
}

func TestMemoryStore_Item_Good(t *core.T) {
	store, _, item := seededStore(t)
	r := store.Item(item.ID)
	core.AssertTrue(t, r.OK)
	core.AssertEqual(t, item.ContentHash, r.Value.(Item).ContentHash)
}

func TestMemoryStore_Item_Bad(t *core.T) {
	store, _, _ := seededStore(t)
	r := store.Item("missing")
	core.AssertFalse(t, r.OK)
	core.AssertEqual(t, "dataset.item.notfound", r.Code())
}

func TestMemoryStore_Items_Good(t *core.T) {
	store, ds, item := seededStore(t)
	later := fixtureItem("item-2", ds.ID, 5000)
	core.RequireTrue(t, store.ItemAppend(later).OK)

	r := store.Items(ItemFilter{DatasetID: ds.ID})
	core.AssertTrue(t, r.OK)
	list := r.Value.([]Item)
	core.AssertLen(t, list, 2)
	core.AssertEqual(t, item.ID, list[0].ID, "created_at ordering: earlier item first")
}

func TestMemoryStore_Items_Bad(t *core.T) {
	store, _, _ := seededStore(t)
	r := store.Items(ItemFilter{})
	core.AssertFalse(t, r.OK, "a filter with no dataset id must fail")
}

func TestMemoryStore_Items_Ugly(t *core.T) {
	store, ds, item := seededStore(t)

	// Kind/Source/ContentHash/Status/Score dimensions all narrow.
	r := store.Items(ItemFilter{DatasetID: ds.ID, Kind: KindMessages})
	core.AssertTrue(t, r.OK)
	core.AssertLen(t, r.Value.([]Item), 0, "no messages-kind items exist")

	r = store.Items(ItemFilter{DatasetID: ds.ID, ContentHash: item.ContentHash})
	core.AssertTrue(t, r.OK)
	core.AssertLen(t, r.Value.([]Item), 1)

	r = store.Items(ItemFilter{DatasetID: ds.ID, Status: StatusApproved})
	core.AssertTrue(t, r.OK)
	core.AssertLen(t, r.Value.([]Item), 0, "no reviews yet — nothing is approved")

	r = store.Items(ItemFilter{DatasetID: ds.ID, Status: StatusPending})
	core.AssertTrue(t, r.OK)
	core.AssertLen(t, r.Value.([]Item), 1, "an unreviewed item is implicitly pending")

	r = store.Items(ItemFilter{DatasetID: ds.ID, Source: SourceCaptureServe})
	core.AssertTrue(t, r.OK)
	core.AssertLen(t, r.Value.([]Item), 0, "no capture:serve-sourced items exist")

	r = store.Items(ItemFilter{DatasetID: ds.ID, Source: item.Source})
	core.AssertTrue(t, r.OK)
	core.AssertLen(t, r.Value.([]Item), 1)

	core.RequireTrue(t, store.ScoreAppend(fixtureScore("s1", item.ID, ScoreKindLEK, 90, 3000)).OK)
	r = store.Items(ItemFilter{DatasetID: ds.ID, Score: &ScoreExpression{Kind: ScoreKindLEK, Op: OpGTE, Threshold: 80}})
	core.AssertTrue(t, r.OK)
	core.AssertLen(t, r.Value.([]Item), 1, "the score filter admits the item once it clears the threshold")

	r = store.Items(ItemFilter{DatasetID: ds.ID, Score: &ScoreExpression{Kind: ScoreKindLEK, Op: OpGTE, Threshold: 95}})
	core.AssertTrue(t, r.OK)
	core.AssertLen(t, r.Value.([]Item), 0, "the score filter excludes the item below the threshold")

	archived := fixtureItem("item-archived", ds.ID, 6000)
	archived.Archived = true
	archived.ArchivedAt = timeAt(6001)
	core.RequireTrue(t, store.ItemAppend(archived).OK)
	r = store.Items(ItemFilter{DatasetID: ds.ID})
	core.AssertTrue(t, r.OK)
	for _, listed := range r.Value.([]Item) {
		core.AssertNotEqual(t, "item-archived", listed.ID, "an archived item is excluded by default")
	}
	r = store.Items(ItemFilter{DatasetID: ds.ID, IncludeArchived: true})
	core.AssertTrue(t, r.OK)
	core.AssertLen(t, r.Value.([]Item), 2, "IncludeArchived surfaces the archived item too")
}

// ---- MemoryStore: Score ----

func TestMemoryStore_ScoreAppend_Good(t *core.T) {
	store, _, item := seededStore(t)
	r := store.ScoreAppend(fixtureScore("s1", item.ID, ScoreKindLEK, 80, 3000))
	core.AssertTrue(t, r.OK)
}

func TestMemoryStore_ScoreAppend_Bad(t *core.T) {
	store, _, _ := seededStore(t)
	r := store.ScoreAppend(fixtureScore("s1", "missing-item", ScoreKindLEK, 80, 3000))
	core.AssertFalse(t, r.OK, "unknown item id must fail")

	invalid := fixtureScore("s2", "item-1", ScoreKindLEK, 80, 3000)
	invalid.ScorerName = ""
	r = store.ScoreAppend(invalid)
	core.AssertFalse(t, r.OK, "an invalid score must fail validation before touching storage")
}

func TestMemoryStore_ScoreAppend_Ugly(t *core.T) {
	store, _, item := seededStore(t)
	// Append-only: two scores of the same kind for the same item both
	// land — re-scoring never overwrites.
	core.RequireTrue(t, store.ScoreAppend(fixtureScore("s1", item.ID, ScoreKindLEK, 40, 3000)).OK)
	core.RequireTrue(t, store.ScoreAppend(fixtureScore("s2", item.ID, ScoreKindLEK, 90, 4000)).OK)
	r := store.Scores(item.ID)
	core.AssertTrue(t, r.OK)
	core.AssertLen(t, r.Value.([]Score), 2)
}

func TestMemoryStore_Scores_Good(t *core.T) {
	store, _, item := seededStore(t)
	core.RequireTrue(t, store.ScoreAppend(fixtureScore("s1", item.ID, ScoreKindLEK, 40, 4000)).OK)
	core.RequireTrue(t, store.ScoreAppend(fixtureScore("s2", item.ID, ScoreKindLEK, 90, 3000)).OK)
	r := store.Scores(item.ID)
	core.AssertTrue(t, r.OK)
	list := r.Value.([]Score)
	core.AssertLen(t, list, 2)
	core.AssertEqual(t, "s2", list[0].ID, "oldest first")
}

func TestMemoryStore_Scores_Bad(t *core.T) {
	store, _, _ := seededStore(t)
	r := store.Scores("missing-item")
	core.AssertFalse(t, r.OK)
}

func TestMemoryStore_Scores_Ugly(t *core.T) {
	store, _, item := seededStore(t)
	r := store.Scores(item.ID)
	core.AssertTrue(t, r.OK)
	core.AssertLen(t, r.Value.([]Score), 0, "an item with no scores yet returns an empty list, not an error")
}

// ---- MemoryStore: Review ----

func TestMemoryStore_ReviewAppend_Good(t *core.T) {
	store, _, item := seededStore(t)
	r := store.ReviewAppend(fixtureReview(item.ID, StatusApproved, 3000))
	core.AssertTrue(t, r.OK)
}

func TestMemoryStore_ReviewAppend_Bad(t *core.T) {
	store, _, _ := seededStore(t)
	r := store.ReviewAppend(fixtureReview("missing-item", StatusApproved, 3000))
	core.AssertFalse(t, r.OK)
}

func TestMemoryStore_ReviewAppend_Ugly(t *core.T) {
	store, _, item := seededStore(t)
	invalid := fixtureReview(item.ID, StatusApproved, 3000)
	invalid.Reviewer = ""
	r := store.ReviewAppend(invalid)
	core.AssertFalse(t, r.OK, "an invalid review must fail validation before touching storage")
}

func TestMemoryStore_ReviewLatest_Good(t *core.T) {
	store, _, item := seededStore(t)
	core.RequireTrue(t, store.ReviewAppend(fixtureReview(item.ID, StatusQuarantined, 3000)).OK)
	core.RequireTrue(t, store.ReviewAppend(fixtureReview(item.ID, StatusApproved, 4000)).OK)
	r := store.ReviewLatest(item.ID)
	core.AssertTrue(t, r.OK)
	core.AssertEqual(t, StatusApproved, r.Value.(Review).Status)
}

func TestMemoryStore_ReviewLatest_Bad(t *core.T) {
	store, _, _ := seededStore(t)
	r := store.ReviewLatest("missing-item")
	core.AssertFalse(t, r.OK)
}

func TestMemoryStore_ReviewLatest_Ugly(t *core.T) {
	store, _, item := seededStore(t)
	r := store.ReviewLatest(item.ID)
	core.AssertTrue(t, r.OK, "no review rows yet is not an error")
	core.AssertEqual(t, StatusPending, r.Value.(Review).Status, "pending is the implicit starting state")

	// A second item's reviews must never leak into the first item's
	// latest-review read.
	other := fixtureItem("item-other", item.DatasetID, 2500)
	core.RequireTrue(t, store.ItemAppend(other).OK)
	core.RequireTrue(t, store.ReviewAppend(fixtureReview(other.ID, StatusRejected, 8000)).OK)

	// Ties on created_at: the later-appended row wins, same rule as
	// ScoreExpression.Matches.
	tie := timeAt(9000)
	core.RequireTrue(t, store.ReviewAppend(Review{ItemID: item.ID, Status: StatusRejected, Reviewer: "auto:threshold", CreatedAt: tie}).OK)
	core.RequireTrue(t, store.ReviewAppend(Review{ItemID: item.ID, Status: StatusApproved, Reviewer: "snider", CreatedAt: tie}).OK)
	r = store.ReviewLatest(item.ID)
	core.AssertTrue(t, r.OK)
	core.AssertEqual(t, StatusApproved, r.Value.(Review).Status, "on a created_at tie the later-inserted review wins")

	otherLatest := core.MustCast[Review](store.ReviewLatest(other.ID))
	core.AssertEqual(t, StatusRejected, otherLatest.Status, "the other item's own review is unaffected")
}

// ---- MemoryStore: Export ----

func TestMemoryStore_ExportAppend_Good(t *core.T) {
	store, ds, _ := seededStore(t)
	r := store.ExportAppend(fixtureExport("exp-1", ds.ID, 3000))
	core.AssertTrue(t, r.OK)
}

func TestMemoryStore_ExportAppend_Bad(t *core.T) {
	store, _, _ := seededStore(t)
	r := store.ExportAppend(fixtureExport("exp-1", "missing-dataset", 3000))
	core.AssertFalse(t, r.OK)
}

func TestMemoryStore_ExportAppend_Ugly(t *core.T) {
	store, ds, _ := seededStore(t)
	invalid := fixtureExport("exp-1", ds.ID, 3000)
	invalid.ManifestHash = ""
	r := store.ExportAppend(invalid)
	core.AssertFalse(t, r.OK)
}

func TestMemoryStore_Exports_Good(t *core.T) {
	store, ds, _ := seededStore(t)
	core.RequireTrue(t, store.ExportAppend(fixtureExport("exp-1", ds.ID, 5000)).OK)
	core.RequireTrue(t, store.ExportAppend(fixtureExport("exp-2", ds.ID, 3000)).OK)
	r := store.Exports(ds.ID)
	core.AssertTrue(t, r.OK)
	list := r.Value.([]Export)
	core.AssertLen(t, list, 2)
	core.AssertEqual(t, "exp-2", list[0].ID, "oldest first")
}

func TestMemoryStore_Exports_Bad(t *core.T) {
	store, _, _ := seededStore(t)
	r := store.Exports("missing-dataset")
	core.AssertFalse(t, r.OK)
}

func TestMemoryStore_Exports_Ugly(t *core.T) {
	store, ds, _ := seededStore(t)
	r := store.Exports(ds.ID)
	core.AssertTrue(t, r.OK)
	core.AssertLen(t, r.Value.([]Export), 0, "a dataset with no exports yet returns an empty list, not an error")
}
