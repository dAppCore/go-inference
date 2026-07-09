// SPDX-Licence-Identifier: EUPL-1.2

package pipeline

import (
	"context"

	core "dappco.re/go"
	"dappco.re/go/inference/eval/score/lek"
	chat "dappco.re/go/inference/serving/chat"
)

// TestScorerAdapter_Score_Good pins that a real (prompt, response) pair scores
// into the "score" key as a decodable DiffResult with the response side read.
func TestScorerAdapter_Score_Good(t *core.T) {
	bundle := scorerAdapter{}.Score(
		userReq("gemma", "explain your reasoning"),
		chat.Response{Text: "absolutely, you are completely right"},
	)
	raw, ok := bundle["score"]
	core.AssertTrue(t, ok)
	core.AssertContains(t, raw, "sycophancy")
	var d lek.DiffResult
	core.AssertTrue(t, core.JSONUnmarshal([]byte(raw), &d).OK)
	core.AssertTrue(t, d.Response.Sycophancy != nil)
}

// TestScorerAdapter_Score_Bad pins the empty case: no prompt and no response
// records nothing.
func TestScorerAdapter_Score_Bad(t *core.T) {
	core.AssertTrue(t, scorerAdapter{}.Score(chat.Request{}, chat.Response{}) == nil)
}

// TestScorerAdapter_Score_Ugly pins the one-sided case: a response with no user
// prompt still scores (the response half is the useful signal).
func TestScorerAdapter_Score_Ugly(t *core.T) {
	b := scorerAdapter{}.Score(chat.Request{}, chat.Response{Text: "you are absolutely right"})
	_, ok := b["score"]
	core.AssertTrue(t, ok)
}

// TestPipeline_Complete_Scorer pins the generation-core wiring: with a Scorer
// set, a completed turn carries the score in resp.Metadata, and the scored
// response is what gets cached — the read persists with the turn.
func TestPipeline_Complete_Scorer(t *core.T) {
	p, cache, _, _, _, backend := fixture()
	p.Scorer = scorerAdapter{}
	backend.byEndpoint["local-metal"] = backendStep{
		resp: chat.Response{Text: "you are absolutely right", FinishReason: "stop"},
	}

	resp, err := p.Complete(context.Background(), userReq("gemma", "was I right?"))
	core.AssertNoError(t, err)

	_, ok := resp.Metadata["score"]
	core.AssertTrue(t, ok)
	// score() runs before cache.Set, so the cached response carries the score.
	_, cached := cache.setLast.Metadata["score"]
	core.AssertTrue(t, cached)
}

// TestPipeline_Complete_Scorer_Merge pins that the score merges into metadata a
// backend already set rather than clobbering it.
func TestPipeline_Complete_Scorer_Merge(t *core.T) {
	p, _, _, _, _, backend := fixture()
	p.Scorer = scorerAdapter{}
	backend.byEndpoint["local-metal"] = backendStep{
		resp: chat.Response{Text: "you are absolutely right", FinishReason: "stop", Metadata: map[string]string{"backend": "local-metal"}},
	}

	resp, err := p.Complete(context.Background(), userReq("gemma", "was I right?"))
	core.AssertNoError(t, err)

	core.AssertEqual(t, "local-metal", resp.Metadata["backend"]) // pre-existing key survives
	_, scored := resp.Metadata["score"]
	core.AssertTrue(t, scored)
}

// nilScorer returns no bundle — the "scorer ran, found nothing to record" case.
type nilScorer struct{}

func (nilScorer) Score(chat.Request, chat.Response) map[string]string { return nil }

// TestPipeline_score_EmptyBundle pins that an empty scorer result leaves the
// response metadata untouched (no empty map allocated).
func TestPipeline_score_EmptyBundle(t *core.T) {
	p := &Pipeline{Scorer: nilScorer{}}
	resp := p.score(userReq("gemma", "hi"), chat.Response{Text: "x"})
	core.AssertTrue(t, resp.Metadata == nil)
}

// TestPipeline_Complete_NoScorer pins that without a Scorer the turn is
// untouched — no metadata rides along (the original five-seam behaviour).
func TestPipeline_Complete_NoScorer(t *core.T) {
	p, _, _, _, _, backend := fixture()
	backend.byEndpoint["local-metal"] = backendStep{
		resp: chat.Response{Text: "hello", FinishReason: "stop"},
	}
	resp, err := p.Complete(context.Background(), userReq("gemma", "hi"))
	core.AssertNoError(t, err)
	core.AssertTrue(t, resp.Metadata == nil)
}
