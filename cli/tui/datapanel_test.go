// SPDX-Licence-Identifier: EUPL-1.2

package tui

import (
	"strings"
	"testing"
	"time"

	"github.com/charmbracelet/lipgloss"
	"github.com/charmbracelet/x/ansi"

	core "dappco.re/go"
	"dappco.re/go/inference/dataset"
)

// ---- fixtures ----

func newTestDataPanelStore(t *testing.T) *duckDatasetStore {
	t.Helper()
	path := t.TempDir() + "/datasets.duckdb"
	s := newTestDuckDatasetStore(t, path)
	t.Cleanup(func() { closeTestDuckDatasetStore(t, s) })
	return s
}

func seedDataDataset(t *testing.T, store dataset.Store, slug string, at time.Time) dataset.Dataset {
	t.Helper()
	ds := dataset.Dataset{ID: dataset.NewID(), Slug: slug, Title: slug, CreatedAt: at}
	if result := store.DatasetCreate(ds); !result.OK {
		t.Fatalf("DatasetCreate(%s): %v", slug, result.Value)
	}
	return ds
}

func seedDataItem(t *testing.T, store dataset.Store, datasetID, prompt, response string, at time.Time) dataset.Item {
	t.Helper()
	item := conformancePairItem(datasetID, prompt, response, at)
	if result := store.ItemAppend(item); !result.OK {
		t.Fatalf("ItemAppend: %v", result.Value)
	}
	return item
}

func seedDataScore(t *testing.T, store dataset.Store, itemID string, value float64, at time.Time) dataset.Score {
	t.Helper()
	score := dataset.Score{ID: dataset.NewID(), ItemID: itemID, Kind: dataset.ScoreKindLEK, Value: value, ScorerName: "test", ScorerVersion: "1", CreatedAt: at}
	if result := store.ScoreAppend(score); !result.OK {
		t.Fatalf("ScoreAppend: %v", result.Value)
	}
	return score
}

// ---- construction / state machine ----

func TestNewDataPanel_Good(t *testing.T) {
	store := newTestDataPanelStore(t)
	at := time.Date(2026, time.July, 19, 9, 0, 0, 0, time.UTC)
	ds := seedDataDataset(t, store, "evening-vents", at)
	seedDataItem(t, store, ds.ID, "hello", "hi there", at)
	seedDataItem(t, store, ds.ID, "second", "second reply", at.Add(time.Minute))

	opened := newDataPanel(store, nil, sequenceIDs("unused"), func() time.Time { return at })
	if !opened.OK {
		t.Fatalf("newDataPanel: %v", opened.Value)
	}
	panel := opened.Value.(*dataPanel)
	if len(panel.rows) != 2 {
		t.Fatalf("rows = %d, want 2", len(panel.rows))
	}
	if panel.rows[0].Item.CreatedAt.Before(panel.rows[1].Item.CreatedAt) {
		t.Fatalf("default sort is not newest-first: %#v", panel.rows)
	}
	for _, row := range panel.rows {
		if row.Review.Status != dataset.StatusPending {
			t.Fatalf("fresh item review status = %q, want pending", row.Review.Status)
		}
		if row.HasScore {
			t.Fatalf("fresh item unexpectedly scored: %#v", row)
		}
	}
}

func TestNewDataPanel_Bad(t *testing.T) {
	if opened := newDataPanel(nil, nil, nil, nil); opened.OK {
		t.Fatal("newDataPanel(nil store) succeeded")
	}
}

func TestDataPanel_SelectionPreservedAcrossRefresh(t *testing.T) {
	store := newTestDataPanelStore(t)
	at := time.Now()
	ds := seedDataDataset(t, store, "vents", at)
	first := seedDataItem(t, store, ds.ID, "first", "first", at)
	seedDataItem(t, store, ds.ID, "second", "second", at.Add(time.Second))

	opened := newDataPanel(store, nil, nil, nil)
	if !opened.OK {
		t.Fatalf("newDataPanel: %v", opened.Value)
	}
	panel := opened.Value.(*dataPanel)
	panel.selectItem(first.ID)
	if selected, ok := panel.Selected(); !ok || selected.Item.ID != first.ID {
		t.Fatalf("selected = %#v ok=%v", selected, ok)
	}
	if result := panel.Refresh(); !result.OK {
		t.Fatalf("Refresh: %v", result.Value)
	}
	if selected, ok := panel.Selected(); !ok || selected.Item.ID != first.ID {
		t.Fatalf("selection lost after refresh: %#v ok=%v", selected, ok)
	}
}

func TestDataPanel_ToggleSort(t *testing.T) {
	store := newTestDataPanelStore(t)
	at := time.Date(2026, time.July, 19, 9, 0, 0, 0, time.UTC)
	ds := seedDataDataset(t, store, "vents", at)
	// Deliberately seeded so date order and score order disagree: "low" is
	// newer (date-sorts first) but scores lower (score-sorts second).
	low := seedDataItem(t, store, ds.ID, "low", "low", at.Add(time.Minute))
	high := seedDataItem(t, store, ds.ID, "high", "high", at)
	seedDataScore(t, store, low.ID, 10, at)
	seedDataScore(t, store, high.ID, 90, at)

	opened := newDataPanel(store, nil, nil, nil)
	if !opened.OK {
		t.Fatalf("newDataPanel: %v", opened.Value)
	}
	panel := opened.Value.(*dataPanel)
	if panel.SortMode() != dataSortDate {
		t.Fatalf("default sort = %d, want dataSortDate", panel.SortMode())
	}
	if panel.rows[0].Item.ID != low.ID {
		t.Fatalf("date sort rows = %#v, want low (newer) first", panel.rows)
	}
	if result := panel.ToggleSort(); !result.OK {
		t.Fatalf("ToggleSort: %v", result.Value)
	}
	if panel.SortMode() != dataSortScore || panel.rows[0].Item.ID != high.ID {
		t.Fatalf("score sort mode=%d rows=%#v, want dataSortScore with high first", panel.SortMode(), panel.rows)
	}
	if result := panel.ToggleSort(); !result.OK {
		t.Fatalf("ToggleSort back: %v", result.Value)
	}
	if panel.SortMode() != dataSortDate || panel.rows[0].Item.ID != low.ID {
		t.Fatalf("sort after second toggle = %d rows=%#v", panel.SortMode(), panel.rows)
	}
}

// ---- filter grammar + round-trips ----

func TestParseDataFilterExpr_Good(t *testing.T) {
	for _, tc := range []struct {
		name string
		expr string
		want dataFilterState
	}{
		{"empty", "", dataFilterState{}},
		{"dataset", "dataset=evening-vents", dataFilterState{DatasetSlug: "evening-vents"}},
		{"status", "status=approved", dataFilterState{Status: dataset.StatusApproved}},
		{"kind", "kind=pair", dataFilterState{Kind: dataset.KindPair}},
		{"source", "source=capture:serve", dataFilterState{Source: dataset.SourceCaptureServe}},
		{"score", "lek>=80", dataFilterState{Score: &dataset.ScoreExpression{Kind: dataset.ScoreKindLEK, Op: dataset.OpGTE, Threshold: 80}}},
		{"combined", "dataset=evening-vents,status=pending,kind=pair,source=capture:serve,lek>=80", dataFilterState{
			DatasetSlug: "evening-vents", Status: dataset.StatusPending, Kind: dataset.KindPair, Source: dataset.SourceCaptureServe,
			Score: &dataset.ScoreExpression{Kind: dataset.ScoreKindLEK, Op: dataset.OpGTE, Threshold: 80},
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, err := parseDataFilterExpr(tc.expr)
			if err != nil {
				t.Fatalf("parseDataFilterExpr(%q): %v", tc.expr, err)
			}
			if !got.Equal(tc.want) {
				t.Fatalf("parseDataFilterExpr(%q) = %#v, want %#v", tc.expr, got, tc.want)
			}
		})
	}
}

func TestParseDataFilterExpr_Bad(t *testing.T) {
	for _, expr := range []string{"unknownfield=x", "notascoreexpr", "lek>=notanumber"} {
		if _, err := parseDataFilterExpr(expr); err == nil {
			t.Fatalf("parseDataFilterExpr(%q) succeeded, want an error", expr)
		}
	}
}

func TestParseDataFilterExpr_Ugly(t *testing.T) {
	// Blank/whitespace-only clauses between commas are tolerated (skipped),
	// mirroring cli/data.go's parseItemFilter idiom.
	got, err := parseDataFilterExpr(" , status=approved ,, ")
	if err != nil {
		t.Fatalf("parseDataFilterExpr: %v", err)
	}
	if got.Status != dataset.StatusApproved {
		t.Fatalf("got = %#v", got)
	}
}

func TestDataFilterState_StringRoundTrip(t *testing.T) {
	score := dataset.ScoreExpression{Kind: dataset.ScoreKindHostility, Op: dataset.OpLT, Threshold: 0.5}
	original := dataFilterState{DatasetSlug: "vents", Status: dataset.StatusQuarantined, Kind: dataset.KindMessages, Source: dataset.SourceImportJSONL, Score: &score}
	roundTripped, err := parseDataFilterExpr(original.String())
	if err != nil {
		t.Fatalf("parseDataFilterExpr(%q): %v", original.String(), err)
	}
	if !roundTripped.Equal(original) {
		t.Fatalf("round trip = %#v, want %#v (expr %q)", roundTripped, original, original.String())
	}
}

func TestDataPanel_SetFilterExpr_RoundTrip(t *testing.T) {
	store := newTestDataPanelStore(t)
	at := time.Now()
	dsA := seedDataDataset(t, store, "alpha", at)
	dsB := seedDataDataset(t, store, "beta", at)
	seedDataItem(t, store, dsA.ID, "a-prompt", "a-response", at)
	seedDataItem(t, store, dsB.ID, "b-prompt", "b-response", at)

	opened := newDataPanel(store, nil, nil, nil)
	if !opened.OK {
		t.Fatalf("newDataPanel: %v", opened.Value)
	}
	panel := opened.Value.(*dataPanel)
	if len(panel.rows) != 2 {
		t.Fatalf("unfiltered rows = %d, want 2", len(panel.rows))
	}

	if result := panel.SetFilterExpr("dataset=alpha"); !result.OK {
		t.Fatalf("SetFilterExpr: %v", result.Value)
	}
	if len(panel.rows) != 1 || panel.rows[0].Dataset.Slug != "alpha" {
		t.Fatalf("filtered rows = %#v", panel.rows)
	}
	if got := panel.FilterExpr(); got != "dataset=alpha" {
		t.Fatalf("FilterExpr() = %q, want %q", got, "dataset=alpha")
	}
	// The exact round trip: re-apply the panel's own rendered expression.
	if result := panel.SetFilterExpr(panel.FilterExpr()); !result.OK {
		t.Fatalf("SetFilterExpr(FilterExpr()): %v", result.Value)
	}
	if len(panel.rows) != 1 || panel.rows[0].Dataset.Slug != "alpha" {
		t.Fatalf("round-tripped filter rows = %#v", panel.rows)
	}

	if result := panel.SetFilterExpr(""); !result.OK {
		t.Fatalf("SetFilterExpr(\"\"): %v", result.Value)
	}
	if len(panel.rows) != 2 {
		t.Fatalf("cleared filter rows = %d, want 2", len(panel.rows))
	}
}

func TestDataPanel_SetFilterExpr_Bad(t *testing.T) {
	store := newTestDataPanelStore(t)
	opened := newDataPanel(store, nil, nil, nil)
	panel := opened.Value.(*dataPanel)
	before := panel.Filter()
	if result := panel.SetFilterExpr("garbage clause"); result.OK {
		t.Fatal("SetFilterExpr(garbage) succeeded")
	}
	if !panel.Filter().Equal(before) {
		t.Fatalf("a rejected filter expression mutated the panel's filter: %#v", panel.Filter())
	}
}

// ---- actions ----

func TestDataPanel_Approve(t *testing.T) {
	store := newTestDataPanelStore(t)
	at := time.Now()
	ds := seedDataDataset(t, store, "vents", at)
	item := seedDataItem(t, store, ds.ID, "p", "r", at)
	opened := newDataPanel(store, nil, nil, func() time.Time { return at })
	panel := opened.Value.(*dataPanel)

	if result := panel.Approve(item.ID); !result.OK {
		t.Fatalf("Approve: %v", result.Value)
	}
	latest := core.MustCast[dataset.Review](store.ReviewLatest(item.ID))
	if latest.Status != dataset.StatusApproved || latest.Reviewer == "" {
		t.Fatalf("approved review = %#v", latest)
	}
}

func TestDataPanel_Reject(t *testing.T) {
	store := newTestDataPanelStore(t)
	at := time.Now()
	ds := seedDataDataset(t, store, "vents", at)
	item := seedDataItem(t, store, ds.ID, "p", "r", at)
	opened := newDataPanel(store, nil, nil, func() time.Time { return at })
	panel := opened.Value.(*dataPanel)

	if result := panel.Reject(item.ID); !result.OK {
		t.Fatalf("Reject: %v", result.Value)
	}
	latest := core.MustCast[dataset.Review](store.ReviewLatest(item.ID))
	if latest.Status != dataset.StatusRejected {
		t.Fatalf("rejected review = %#v", latest)
	}
}

func TestDataPanel_QuarantineClear_Good(t *testing.T) {
	store := newTestDataPanelStore(t)
	at := time.Now()
	ds := seedDataDataset(t, store, "vents", at)
	item := seedDataItem(t, store, ds.ID, "p", "r", at)
	if result := store.ReviewAppend(dataset.Review{ItemID: item.ID, Status: dataset.StatusQuarantined, Reviewer: dataset.ReviewerAutoWelfare, Note: "slur match", CreatedAt: at}); !result.OK {
		t.Fatalf("seed quarantine: %v", result.Value)
	}
	opened := newDataPanel(store, nil, nil, func() time.Time { return at.Add(time.Minute) })
	panel := opened.Value.(*dataPanel)

	if result := panel.QuarantineClear(item.ID, "reviewed: clinical context, not a real hit"); !result.OK {
		t.Fatalf("QuarantineClear: %v", result.Value)
	}
	latest := core.MustCast[dataset.Review](store.ReviewLatest(item.ID))
	if latest.Status != dataset.StatusApproved || !core.Contains(latest.Note, "clinical context") {
		t.Fatalf("cleared review = %#v", latest)
	}
}

func TestDataPanel_QuarantineClear_Bad(t *testing.T) {
	store := newTestDataPanelStore(t)
	at := time.Now()
	ds := seedDataDataset(t, store, "vents", at)
	item := seedDataItem(t, store, ds.ID, "p", "r", at)
	opened := newDataPanel(store, nil, nil, func() time.Time { return at })
	panel := opened.Value.(*dataPanel)

	if result := panel.QuarantineClear(item.ID, "   "); result.OK {
		t.Fatal("QuarantineClear with a blank note succeeded")
	}
	if result := panel.QuarantineClear(item.ID, ""); result.OK {
		t.Fatal("QuarantineClear with an empty note succeeded")
	}
	reviews := core.MustCast[[]dataset.Review](store.ReviewHistory(item.ID))
	if len(reviews) != 0 {
		t.Fatalf("a note-less QuarantineClear wrote a review: %#v", reviews)
	}
}

func TestDataPanel_Tag_Good(t *testing.T) {
	store := newTestDataPanelStore(t)
	at := time.Now()
	ds := seedDataDataset(t, store, "vents", at)
	item := seedDataItem(t, store, ds.ID, "p", "r", at)
	opened := newDataPanel(store, nil, nil, func() time.Time { return at })
	panel := opened.Value.(*dataPanel)

	if result := panel.Approve(item.ID); !result.OK {
		t.Fatalf("Approve: %v", result.Value)
	}
	if result := panel.Tag(item.ID, "favourite"); !result.OK {
		t.Fatalf("Tag: %v", result.Value)
	}
	latest := core.MustCast[dataset.Review](store.ReviewLatest(item.ID))
	if latest.Status != dataset.StatusApproved || latest.Note != "tag: favourite" {
		t.Fatalf("tag review = %#v, want status preserved + tag note", latest)
	}
}

func TestDataPanel_Tag_Bad(t *testing.T) {
	store := newTestDataPanelStore(t)
	at := time.Now()
	ds := seedDataDataset(t, store, "vents", at)
	item := seedDataItem(t, store, ds.ID, "p", "r", at)
	opened := newDataPanel(store, nil, nil, func() time.Time { return at })
	panel := opened.Value.(*dataPanel)

	if result := panel.Tag(item.ID, "  "); result.OK {
		t.Fatal("Tag with a blank label succeeded")
	}
	reviews := core.MustCast[[]dataset.Review](store.ReviewHistory(item.ID))
	if len(reviews) != 0 {
		t.Fatalf("a blank-label Tag wrote a review: %#v", reviews)
	}
}

func TestDataPanel_EditAsDerived_Good(t *testing.T) {
	store := newTestDataPanelStore(t)
	at := time.Date(2026, time.July, 19, 9, 0, 0, 0, time.UTC)
	ds := seedDataDataset(t, store, "vents", at)
	original := seedDataItem(t, store, ds.ID, "original prompt", "original response", at)

	opened := newDataPanel(store, nil, sequenceIDs("derived-1"), func() time.Time { return at.Add(time.Hour) })
	panel := opened.Value.(*dataPanel)

	result := panel.EditAsDerived(original, "original prompt", "edited response")
	if !result.OK {
		t.Fatalf("EditAsDerived: %v", result.Value)
	}
	derived := result.Value.(dataset.Item)
	if derived.ID != "derived-1" || derived.ParentItemID != original.ID || derived.DatasetID != ds.ID || derived.ContentHash == original.ContentHash {
		t.Fatalf("derived item = %#v", derived)
	}
	_, editedResponse, ok := dataItemExchange(derived)
	if !ok || editedResponse != "edited response" {
		t.Fatalf("derived content = %s, ok=%v", derived.Content, ok)
	}

	archived := core.MustCast[dataset.Item](store.Item(original.ID))
	if !archived.Archived {
		t.Fatalf("original was not archived: %#v", archived)
	}
	review := core.MustCast[dataset.Review](store.ReviewLatest(derived.ID))
	if review.Status != dataset.StatusApproved {
		t.Fatalf("derived review = %#v", review)
	}
}

func TestDataPanel_EditAsDerived_Bad(t *testing.T) {
	// dataset.MemoryStore does not implement ItemArchiver — EditAsDerived
	// must refuse loudly rather than create a derived item it cannot
	// finish superseding the original for.
	store := dataset.NewMemoryStore()
	at := time.Now()
	ds := dataset.Dataset{ID: dataset.NewID(), Slug: "vents", Title: "vents", CreatedAt: at}
	if result := store.DatasetCreate(ds); !result.OK {
		t.Fatalf("DatasetCreate: %v", result.Value)
	}
	original := conformancePairItem(ds.ID, "p", "r", at)
	if result := store.ItemAppend(original); !result.OK {
		t.Fatalf("ItemAppend: %v", result.Value)
	}

	opened := newDataPanel(store, nil, nil, nil)
	panel := opened.Value.(*dataPanel)

	if result := panel.EditAsDerived(original, "p", "edited"); result.OK {
		t.Fatal("EditAsDerived succeeded against a store without ItemArchiver")
	}
	items := core.MustCast[[]dataset.Item](store.Items(dataset.ItemFilter{DatasetID: ds.ID, IncludeArchived: true}))
	if len(items) != 1 {
		t.Fatalf("EditAsDerived left extra items behind: %#v", items)
	}
	if items[0].Archived {
		t.Fatal("EditAsDerived archived the original despite failing")
	}
}

// ---- bulk apply ----

func TestDataPanel_BulkApply_Good(t *testing.T) {
	store := newTestDataPanelStore(t)
	at := time.Now()
	ds := seedDataDataset(t, store, "vents", at)
	itemA := seedDataItem(t, store, ds.ID, "a", "a", at)
	itemB := seedDataItem(t, store, ds.ID, "b", "b", at.Add(time.Second))
	itemC := seedDataItem(t, store, ds.ID, "c", "c", at.Add(2*time.Second))

	opened := newDataPanel(store, nil, nil, func() time.Time { return at })
	panel := opened.Value.(*dataPanel)
	if len(panel.rows) != 3 {
		t.Fatalf("seeded rows = %d, want 3", len(panel.rows))
	}

	result := panel.BulkApply(dataActionApprove, "")
	if !result.OK {
		t.Fatalf("BulkApply approve: %v", result.Value)
	}
	if applied := result.Value.(int); applied != 3 {
		t.Fatalf("BulkApply applied = %d, want 3", applied)
	}
	for _, item := range []dataset.Item{itemA, itemB, itemC} {
		review := core.MustCast[dataset.Review](store.ReviewLatest(item.ID))
		if review.Status != dataset.StatusApproved {
			t.Fatalf("item %s status = %q, want approved", item.ID, review.Status)
		}
	}
}

func TestDataPanel_BulkApply_Good_TagPreservesStatus(t *testing.T) {
	store := newTestDataPanelStore(t)
	at := time.Now()
	ds := seedDataDataset(t, store, "vents", at)
	approved := seedDataItem(t, store, ds.ID, "a", "a", at)
	pending := seedDataItem(t, store, ds.ID, "b", "b", at.Add(time.Second))
	if result := store.ReviewAppend(dataset.Review{ItemID: approved.ID, Status: dataset.StatusApproved, Reviewer: "snider", CreatedAt: at}); !result.OK {
		t.Fatalf("seed approved: %v", result.Value)
	}

	opened := newDataPanel(store, nil, nil, func() time.Time { return at.Add(time.Hour) })
	panel := opened.Value.(*dataPanel)

	if result := panel.BulkApply(dataActionTag, "batch-1"); !result.OK {
		t.Fatalf("BulkApply tag: %v", result.Value)
	}
	approvedReview := core.MustCast[dataset.Review](store.ReviewLatest(approved.ID))
	if approvedReview.Status != dataset.StatusApproved || approvedReview.Note != "tag: batch-1" {
		t.Fatalf("approved item after bulk tag = %#v, want status preserved", approvedReview)
	}
	pendingReview := core.MustCast[dataset.Review](store.ReviewLatest(pending.ID))
	if pendingReview.Status != dataset.StatusPending || pendingReview.Note != "tag: batch-1" {
		t.Fatalf("pending item after bulk tag = %#v, want status preserved", pendingReview)
	}
}

func TestDataPanel_BulkApply_Bad(t *testing.T) {
	store := newTestDataPanelStore(t)
	at := time.Now()
	ds := seedDataDataset(t, store, "vents", at)
	original := seedDataItem(t, store, ds.ID, "a", "a", at)

	opened := newDataPanel(store, nil, nil, nil)
	panel := opened.Value.(*dataPanel)

	if result := panel.BulkApply(dataActionEditAsDerived, ""); result.OK {
		t.Fatal("BulkApply(edit-as-derived) succeeded — it has no bulk form")
	}
	if result := panel.BulkApply(dataActionTag, "  "); result.OK {
		t.Fatal("BulkApply(tag) with a blank note succeeded")
	}
	if result := panel.BulkApply(dataActionQuarantineClear, ""); result.OK {
		t.Fatal("BulkApply(quarantine-clear) with an empty note succeeded")
	}
	reviews := core.MustCast[[]dataset.Review](store.ReviewHistory(original.ID))
	if len(reviews) != 0 {
		t.Fatalf("a rejected bulk call wrote a review: %#v", reviews)
	}
}

// ---- capabilities (the agentcap pattern, honest unavailable states) ----

func TestDataPanel_Capabilities_Good(t *testing.T) {
	store := newTestDataPanelStore(t)
	at := time.Now()
	ds := seedDataDataset(t, store, "vents", at)
	item := seedDataItem(t, store, ds.ID, "p", "r", at)
	opened := newDataPanel(store, nil, nil, nil)
	panel := opened.Value.(*dataPanel)
	panel.selectItem(item.ID)

	byKey := dataCapabilityIndex(panel.Capabilities())
	if !byKey["approve"].Available {
		t.Fatalf("approve unavailable with a selection: %#v", byKey["approve"])
	}
	if byKey["quarantine-clear"].Available {
		t.Fatalf("quarantine-clear available on a pending item: %#v", byKey["quarantine-clear"])
	}
	if !byKey["edit-as-derived"].Available {
		t.Fatalf("edit-as-derived unavailable against a real ItemArchiver store: %#v", byKey["edit-as-derived"])
	}
	if !byKey["bulk.approve"].Available {
		t.Fatalf("bulk approve unavailable with 1 filtered item: %#v", byKey["bulk.approve"])
	}

	// Clearing the quarantine flag flips availability the other way.
	if result := store.ReviewAppend(dataset.Review{ItemID: item.ID, Status: dataset.StatusQuarantined, Reviewer: dataset.ReviewerAutoWelfare, CreatedAt: at}); !result.OK {
		t.Fatalf("seed quarantine: %v", result.Value)
	}
	if result := panel.Refresh(); !result.OK {
		t.Fatalf("Refresh: %v", result.Value)
	}
	panel.selectItem(item.ID)
	byKey = dataCapabilityIndex(panel.Capabilities())
	if !byKey["quarantine-clear"].Available {
		t.Fatalf("quarantine-clear unavailable on a quarantined item: %#v", byKey["quarantine-clear"])
	}
}

func TestDataPanel_Capabilities_Bad(t *testing.T) {
	var nilPanel *dataPanel
	for _, capability := range nilPanel.Capabilities() {
		if capability.Available || capability.Reason == "" {
			t.Fatalf("nil-panel capability rendered available or reasonless: %#v", capability)
		}
	}

	store := dataset.NewMemoryStore()
	opened := newDataPanel(store, nil, nil, nil)
	empty := opened.Value.(*dataPanel)
	for _, capability := range empty.Capabilities() {
		if capability.Bulk {
			if capability.Available {
				t.Fatalf("bulk %#v available with zero filtered items", capability)
			}
			continue
		}
		if capability.Available {
			t.Fatalf("single-item %#v available with no selection", capability)
		}
	}

	at := time.Now()
	ds := dataset.Dataset{ID: dataset.NewID(), Slug: "vents", Title: "vents", CreatedAt: at}
	if result := store.DatasetCreate(ds); !result.OK {
		t.Fatalf("DatasetCreate: %v", result.Value)
	}
	item := conformancePairItem(ds.ID, "p", "r", at)
	if result := store.ItemAppend(item); !result.OK {
		t.Fatalf("ItemAppend: %v", result.Value)
	}
	if result := empty.Refresh(); !result.OK {
		t.Fatalf("Refresh: %v", result.Value)
	}
	empty.selectItem(item.ID)
	byKey := dataCapabilityIndex(empty.Capabilities())
	if byKey["edit-as-derived"].Available {
		t.Fatalf("edit-as-derived available against a MemoryStore-backed panel: %#v", byKey["edit-as-derived"])
	}
}

func dataCapabilityIndex(capabilities []dataCapability) map[string]dataCapability {
	byKey := make(map[string]dataCapability, len(capabilities))
	for _, capability := range capabilities {
		key := dataActionSlug(capability.Action)
		if capability.Bulk {
			key = "bulk." + key
		}
		byKey[key] = capability
	}
	return byKey
}

// ---- render snapshots (the layout-test idiom: structural assertions +
// a hard width bound, not literal golden files) ----

func TestDataPanel_View_Ugly(t *testing.T) {
	store := newTestDataPanelStore(t)
	at := time.Date(2026, time.July, 19, 9, 0, 0, 0, time.UTC)
	ds := seedDataDataset(t, store, "evening-vents", at)
	item := seedDataItem(t, store, ds.ID, "hello there", "hi general kenobi", at)
	seedDataScore(t, store, item.ID, 87, at)
	if result := store.ReviewAppend(dataset.Review{ItemID: item.ID, Status: dataset.StatusQuarantined, Reviewer: dataset.ReviewerAutoWelfare, Note: "flagged", CreatedAt: at}); !result.OK {
		t.Fatalf("seed quarantine: %v", result.Value)
	}

	opened := newDataPanel(store, newMarkdownRenderer("dark"), nil, nil)
	if !opened.OK {
		t.Fatalf("newDataPanel: %v", opened.Value)
	}
	panel := opened.Value.(*dataPanel)
	panel.selectItem(item.ID)

	// height=90: renderDetail (like workPanel.renderDetail) has no
	// internal viewport/scroll — a short pane height clips the tail of the
	// detail pane, exactly as it does for the Work panel, so this needs
	// enough room to fit every section for the presence assertions below
	// to be meaningful. The width-overflow bound is independent of that
	// and holds regardless of height.
	styles := newUIStyles(midnightTheme())
	for _, width := range []int{72, 132} {
		view := panel.View(width, 90, styles)
		plain := ansi.Strip(view)
		for _, want := range []string{
			"DATA", "QUARANTINED", "evening-vents", "pair",
			"CONTENT", "hello there", "hi general kenobi",
			"SCORES", "lek", "REVIEW", "WELFARE FLAG", "LINEAGE", "no lineage",
		} {
			if !strings.Contains(plain, want) {
				t.Fatalf("width %d missing %q:\n%s", width, want, plain)
			}
		}
		for line, text := range strings.Split(view, "\n") {
			if got := lipgloss.Width(text); got > width {
				t.Fatalf("width %d line %d overflows at %d: %q", width, line, got, text)
			}
		}
	}
}

func TestDataPanel_View_Empty(t *testing.T) {
	store := dataset.NewMemoryStore()
	opened := newDataPanel(store, nil, nil, nil)
	panel := opened.Value.(*dataPanel)
	view := panel.View(100, 24, newUIStyles(midnightTheme()))
	if !strings.Contains(view, "No items match this filter") || !strings.Contains(view, "Select an item") {
		t.Fatalf("empty view:\n%s", view)
	}
}

func TestDataPanel_View_ZeroSize(t *testing.T) {
	store := dataset.NewMemoryStore()
	opened := newDataPanel(store, nil, nil, nil)
	panel := opened.Value.(*dataPanel)
	if view := panel.View(0, 24, newUIStyles(midnightTheme())); view != "" {
		t.Fatalf("zero-width view = %q, want empty", view)
	}
	var nilPanel *dataPanel
	if view := nilPanel.View(80, 24, newUIStyles(midnightTheme())); view != "" {
		t.Fatalf("nil-panel view = %q, want empty", view)
	}
}
