// SPDX-Licence-Identifier: EUPL-1.2

package dataset_test

import (
	"time"

	core "dappco.re/go"
	"dappco.re/go/inference/dataset"
)

// ExampleScoreExpression_String renders an expression back in its source
// grammar — the text an Export manifest or an auto-threshold Review note
// records.
func ExampleScoreExpression_String() {
	e := dataset.ScoreExpression{Kind: dataset.ScoreKindLEK, Op: dataset.OpGTE, Threshold: 80}
	core.Println(e.String())
	// Output:
	// lek>=80
}

// ExampleScoreExpression_Matches evaluates a threshold against an
// item's current scores, using the LATEST score of the matching kind.
func ExampleScoreExpression_Matches() {
	scores := []dataset.Score{
		{ItemID: "item-1", Kind: dataset.ScoreKindLEK, Value: 60, ScorerName: "lek", CreatedAt: time.Unix(1000, 0)},
		{ItemID: "item-1", Kind: dataset.ScoreKindLEK, Value: 85, ScorerName: "lek", CreatedAt: time.Unix(2000, 0)},
	}
	expr := dataset.ScoreExpression{Kind: dataset.ScoreKindLEK, Op: dataset.OpGTE, Threshold: 80}
	core.Println(expr.Matches(scores))
	// Output:
	// true
}

// ExampleValidateItem validates a fully-formed Item before it is
// persisted — the check every Store.ItemAppend implementation runs.
func ExampleValidateItem() {
	content := core.JSONMarshal(dataset.PairContent{Prompt: "hi", Response: "hello"}).Value.([]byte)
	item := dataset.Item{
		ID: "item-1", DatasetID: "ds-1", Kind: dataset.KindPair, Content: content,
		Source: dataset.SourceImportJSONL, ContentHash: "deadbeef", CreatedAt: time.Unix(1000, 0),
	}
	r := dataset.ValidateItem(item)
	core.Println(r.OK)
	// Output:
	// true
}

// ExampleNewMemoryStore walks the root Store contract end to end: create
// a dataset, append an item, read it back through a filter, and check
// its implicit pending review status.
func ExampleNewMemoryStore() {
	store := dataset.NewMemoryStore()

	ds := dataset.Dataset{ID: "ds-1", Slug: "evening-vents", Title: "Evening vents", CreatedAt: time.Unix(1000, 0)}
	core.MustCast[dataset.Dataset](store.DatasetCreate(ds))

	content := core.JSONMarshal(dataset.PairContent{Prompt: "hi", Response: "hello"}).Value.([]byte)
	hash := core.MustCast[string](dataset.ContentHash(dataset.KindPair, content))
	item := dataset.Item{
		ID: "item-1", DatasetID: ds.ID, Kind: dataset.KindPair, Content: content,
		Source: dataset.SourceImportJSONL, ContentHash: hash, CreatedAt: time.Unix(2000, 0),
	}
	core.MustCast[dataset.Item](store.ItemAppend(item))

	items := core.MustCast[[]dataset.Item](store.Items(dataset.ItemFilter{DatasetID: ds.ID}))
	review := core.MustCast[dataset.Review](store.ReviewLatest(items[0].ID))
	core.Println(len(items), review.Status)
	// Output:
	// 1 pending
}
