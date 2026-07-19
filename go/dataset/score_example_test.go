// SPDX-Licence-Identifier: EUPL-1.2

package dataset_test

import (
	"context"
	"time"

	core "dappco.re/go"
	"dappco.re/go/inference/dataset"
)

// ExampleScoreHeuristic runs the no-model-needed heuristic tier over one
// item's content — lek.ScorePair under the hood, surfaced as three
// Score rows (lek, hostility, sycophancy).
func ExampleScoreHeuristic() {
	content := core.JSONMarshal(dataset.PairContent{Prompt: "explain your reasoning", Response: "the sky looks blue because of Rayleigh scattering"}).Value.([]byte)
	item := dataset.Item{ID: "item-1", Kind: dataset.KindPair, Content: content}

	r := dataset.ScoreHeuristic(item)
	scores := r.Value.([]dataset.Score)
	core.Println(r.OK, len(scores))
	// Output:
	// true 3
}

// ExampleParseScoreExpression parses the tiny auto-threshold/filter
// grammar.
func ExampleParseScoreExpression() {
	r := dataset.ParseScoreExpression("lek>=80")
	expr := r.Value.(dataset.ScoreExpression)
	core.Println(expr.String())
	// Output:
	// lek>=80
}

// ExampleApplyAutoThreshold writes an auto:threshold Review when an
// explicit approve/reject expression matches an item's current scores —
// never silently, and never when neither expression is set.
func ExampleApplyAutoThreshold() {
	store := dataset.NewMemoryStore()
	ds := dataset.Dataset{ID: "ds-1", Slug: "evening-vents", Title: "Evening vents", CreatedAt: time.Unix(1000, 0)}
	core.MustCast[dataset.Dataset](store.DatasetCreate(ds))
	content := core.JSONMarshal(dataset.PairContent{Prompt: "hi", Response: "hello"}).Value.([]byte)
	hash := core.MustCast[string](dataset.ContentHash(dataset.KindPair, content))
	item := dataset.Item{ID: "item-1", DatasetID: ds.ID, Kind: dataset.KindPair, Content: content, Source: dataset.SourceImportJSONL, ContentHash: hash, CreatedAt: time.Unix(2000, 0)}
	core.MustCast[dataset.Item](store.ItemAppend(item))

	scores := []dataset.Score{{ItemID: item.ID, Kind: dataset.ScoreKindLEK, Value: 90, ScorerName: "lek", CreatedAt: time.Unix(3000, 0)}}
	approve := &dataset.ScoreExpression{Kind: dataset.ScoreKindLEK, Op: dataset.OpGTE, Threshold: 80}

	r := dataset.ApplyAutoThreshold(store, item, scores, approve, nil)
	result := r.Value.(dataset.AutoThresholdResult)
	core.Println(result.Applied, result.Review.Status)
	// Output:
	// true approved
}

// ExampleScoreJudge dispatches through a JudgeDispatcher (a CLI-side
// driver in production) and appends the resulting judge-tier Score row.
func ExampleScoreJudge() {
	store, ds := dataset.NewMemoryStore(), dataset.Dataset{ID: "ds-1", Slug: "evening-vents", Title: "Evening vents", CreatedAt: time.Unix(1000, 0)}
	core.MustCast[dataset.Dataset](store.DatasetCreate(ds))
	content := core.JSONMarshal(dataset.PairContent{Prompt: "hi", Response: "hello"}).Value.([]byte)
	hash := core.MustCast[string](dataset.ContentHash(dataset.KindPair, content))
	item := dataset.Item{ID: "item-1", DatasetID: ds.ID, Kind: dataset.KindPair, Content: content, Source: dataset.SourceImportJSONL, ContentHash: hash, CreatedAt: time.Unix(2000, 0)}
	core.MustCast[dataset.Item](store.ItemAppend(item))

	driver := func(_ context.Context, template string, _ dataset.Item) (dataset.JudgeVerdict, error) {
		return dataset.JudgeVerdict{Value: 85, Fingerprint: "gemma4-fp-abc"}, nil
	}
	r := dataset.ScoreJudge(context.Background(), store, driver, item, "helpfulness")
	score := r.Value.(dataset.Score)
	core.Println(score.Kind, score.Value, score.JudgeFingerprint)
	// Output:
	// judge:helpfulness 85 gemma4-fp-abc
}
