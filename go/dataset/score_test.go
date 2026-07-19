// SPDX-Licence-Identifier: EUPL-1.2

package dataset

import (
	"context"

	core "dappco.re/go"
)

func TestScoreHeuristic_Good(t *core.T) {
	item := fixtureItem("item-1", "ds-1", 1000)
	r := ScoreHeuristic(item)
	core.AssertTrue(t, r.OK)
	scores := r.Value.([]Score)
	core.AssertLen(t, scores, 3)

	byKind := map[ScoreKind]Score{}
	for _, s := range scores {
		core.AssertEqual(t, item.ID, s.ItemID)
		core.AssertEqual(t, heuristicScorerName, s.ScorerName)
		core.AssertEqual(t, heuristicScorerVersion, s.ScorerVersion)
		core.AssertNotEmpty(t, s.Payload)
		byKind[s.Kind] = s
	}
	core.AssertGreaterOrEqual(t, byKind[ScoreKindLEK].Value, 0.0)
	core.AssertLessOrEqual(t, byKind[ScoreKindLEK].Value, 100.0)
	core.AssertGreaterOrEqual(t, byKind[ScoreKindHostility].Value, 0.0)
	core.AssertLessOrEqual(t, byKind[ScoreKindHostility].Value, 1.0)

	// KindMessages reduces via LastExchange to the same scorable shape.
	messagesContent := core.JSONMarshal(MessagesContent{Messages: []MessageTurn{
		{Role: "user", Content: "hi"},
		{Role: "assistant", Content: "hello"},
	}}).Value.([]byte)
	messagesItem := Item{ID: "item-2", Kind: KindMessages, Content: messagesContent}
	r = ScoreHeuristic(messagesItem)
	core.AssertTrue(t, r.OK)
	core.AssertLen(t, r.Value.([]Score), 3)
}

func TestScoreHeuristic_Bad(t *core.T) {
	trace := Item{ID: "item-1", Kind: KindTrace, Content: []byte(`{"logits":[1]}`)}
	r := ScoreHeuristic(trace)
	core.AssertFalse(t, r.OK, "an opaque trace is not scorable by the heuristic tier")

	noAssistant := core.JSONMarshal(MessagesContent{Messages: []MessageTurn{{Role: "user", Content: "hi"}}}).Value.([]byte)
	r = ScoreHeuristic(Item{ID: "item-2", Kind: KindMessages, Content: noAssistant})
	core.AssertFalse(t, r.OK, "a conversation with no assistant turn is not scorable")
}

func TestScoreHeuristic_Ugly(t *core.T) {
	r := ScoreHeuristic(Item{ID: "item-1", Kind: KindPair, Content: []byte("not json")})
	core.AssertFalse(t, r.OK, "malformed content must fail, not panic")

	r = ScoreHeuristic(Item{ID: "item-1", Kind: KindMessages, Content: []byte("not json")})
	core.AssertFalse(t, r.OK, "malformed messages content must fail, not panic")

	r = ScoreHeuristic(Item{ID: "item-1", Kind: ItemKind("bogus"), Content: []byte(`{}`)})
	core.AssertFalse(t, r.OK, "an unknown kind must fail")
}

func TestScoreHeuristicAppend_Good(t *core.T) {
	store, _, item := seededStore(t)
	r := ScoreHeuristicAppend(store, item)
	core.AssertTrue(t, r.OK)
	core.AssertLen(t, r.Value.([]Score), 3)

	stored := core.MustCast[[]Score](store.Scores(item.ID))
	core.AssertLen(t, stored, 3)

	// Append-only: re-scoring the same item adds a fresh set of rows,
	// it never overwrites the first — the score history IS the record.
	r = ScoreHeuristicAppend(store, item)
	core.AssertTrue(t, r.OK)
	stored = core.MustCast[[]Score](store.Scores(item.ID))
	core.AssertLen(t, stored, 6, "a second scoring pass adds, it does not replace")
}

func TestScoreHeuristicAppend_Bad(t *core.T) {
	r := ScoreHeuristicAppend(nil, fixtureItem("item-1", "ds-1", 1000))
	core.AssertFalse(t, r.OK, "a nil store must fail, not panic")

	store, _, _ := seededStore(t)
	unscorable := Item{ID: "item-trace", Kind: KindTrace, Content: []byte(`{"logits":[1]}`)}
	r = ScoreHeuristicAppend(store, unscorable)
	core.AssertFalse(t, r.OK, "ScoreHeuristic's own failure (an unscorable kind) propagates before any append is attempted")
}

func TestScoreHeuristicAppend_Ugly(t *core.T) {
	store, _, _ := seededStore(t)
	// An item unknown to the store: ScoreHeuristic succeeds (pure), but
	// the append fails on referential integrity — the caller sees the
	// store's error, not a partial write.
	orphan := fixtureItem("orphan", "ds-1", 1000)
	r := ScoreHeuristicAppend(store, orphan)
	core.AssertFalse(t, r.OK)
	core.AssertLen(t, core.MustCast[[]Score](store.Scores("item-1")), 0, "no scores landed for the seeded item either")
}

func TestFindComparisonOp_Good(t *core.T) {
	op, n := findComparisonOp("lek>=80")
	core.AssertEqual(t, OpGTE, op)
	core.AssertEqual(t, 2, n)

	op, _ = findComparisonOp("hostility<0.5")
	core.AssertEqual(t, OpLT, op)
}

func TestFindComparisonOp_Bad(t *core.T) {
	op, _ := findComparisonOp("no operator here")
	core.AssertEqual(t, ComparisonOp(""), op)
}

func TestFindComparisonOp_Ugly(t *core.T) {
	// ">" is a prefix of ">=" at the same position — the longer match wins.
	op, n := findComparisonOp(">=80")
	core.AssertEqual(t, OpGTE, op)
	core.AssertEqual(t, 2, n)
}

func TestParseScoreExpression_Good(t *core.T) {
	r := ParseScoreExpression("lek>=80")
	core.AssertTrue(t, r.OK)
	expr := r.Value.(ScoreExpression)
	core.AssertEqual(t, ScoreKindLEK, expr.Kind)
	core.AssertEqual(t, OpGTE, expr.Op)
	core.AssertEqual(t, 80.0, expr.Threshold)

	r = ParseScoreExpression("hostility >= 0.7")
	core.AssertTrue(t, r.OK, "surrounding whitespace is tolerated")

	r = ParseScoreExpression("judge:helpfulness>=90")
	core.AssertTrue(t, r.OK, "a well-formed judge field is accepted")
}

func TestParseScoreExpression_Bad(t *core.T) {
	core.AssertFalse(t, ParseScoreExpression("lek~=80").OK, "an unknown operator must fail")
	core.AssertFalse(t, ParseScoreExpression("bogus>=80").OK, "an unknown field must fail")
	core.AssertFalse(t, ParseScoreExpression("lek>=eighty").OK, "a non-numeric threshold must fail")
}

func TestParseScoreExpression_Ugly(t *core.T) {
	core.AssertFalse(t, ParseScoreExpression(">=80").OK, "an empty field must fail")
	core.AssertFalse(t, ParseScoreExpression("lek>=").OK, "an empty threshold must fail")
	core.AssertFalse(t, ParseScoreExpression("").OK, "an empty expression must fail")
	core.AssertFalse(t, ParseScoreExpression("judge:>=80").OK, "a bare judge prefix field must fail")
}

func TestApplyAutoThreshold_Good(t *core.T) {
	store, _, item := seededStore(t)
	scores := []Score{fixtureScore("s1", item.ID, ScoreKindLEK, 90, 1000)}
	approve := &ScoreExpression{Kind: ScoreKindLEK, Op: OpGTE, Threshold: 80}

	r := ApplyAutoThreshold(store, item, scores, approve, nil)
	core.AssertTrue(t, r.OK)
	result := r.Value.(AutoThresholdResult)
	core.AssertTrue(t, result.Applied)
	core.AssertEqual(t, StatusApproved, result.Review.Status)
	core.AssertEqual(t, ReviewerAutoThreshold, result.Review.Reviewer)

	latest := core.MustCast[Review](store.ReviewLatest(item.ID))
	core.AssertEqual(t, StatusApproved, latest.Status)
}

func TestApplyAutoThreshold_Bad(t *core.T) {
	store, _, item := seededStore(t)
	scores := []Score{fixtureScore("s1", item.ID, ScoreKindLEK, 40, 1000)}
	approve := &ScoreExpression{Kind: ScoreKindLEK, Op: OpGTE, Threshold: 80}

	r := ApplyAutoThreshold(store, item, scores, approve, nil)
	core.AssertTrue(t, r.OK, "a non-matching expression is not an error")
	core.AssertFalse(t, r.Value.(AutoThresholdResult).Applied)

	r = ApplyAutoThreshold(nil, item, scores, approve, nil)
	core.AssertFalse(t, r.OK, "a nil store must fail")
}

func TestApplyAutoThreshold_Ugly(t *core.T) {
	store, _, item := seededStore(t)
	scores := []Score{
		fixtureScore("s1", item.ID, ScoreKindLEK, 90, 1000),
		fixtureScore("s2", item.ID, ScoreKindHostility, 0.9, 1000),
	}
	// An item that satisfies BOTH an approve and a reject expression is
	// never accidentally auto-approved — reject wins.
	approve := &ScoreExpression{Kind: ScoreKindLEK, Op: OpGTE, Threshold: 80}
	reject := &ScoreExpression{Kind: ScoreKindHostility, Op: OpGTE, Threshold: 0.7}

	r := ApplyAutoThreshold(store, item, scores, approve, reject)
	core.AssertTrue(t, r.OK)
	result := r.Value.(AutoThresholdResult)
	core.AssertTrue(t, result.Applied)
	core.AssertEqual(t, StatusRejected, result.Review.Status, "reject is checked first")

	// Neither expression set: never evaluated, no silent policy.
	r = ApplyAutoThreshold(store, item, scores, nil, nil)
	core.AssertTrue(t, r.OK)
	core.AssertFalse(t, r.Value.(AutoThresholdResult).Applied)

	// An item unknown to the store: the match still evaluates, but the
	// Review append fails on referential integrity.
	unknown := Item{ID: "missing-item"}
	r = ApplyAutoThreshold(store, unknown, scores, approve, nil)
	core.AssertFalse(t, r.OK)
}

func fakeJudge(value float64, fingerprint string, err error) JudgeDispatcher {
	return func(_ context.Context, _ string, _ Item) (JudgeVerdict, error) {
		if err != nil {
			return JudgeVerdict{}, err
		}
		return JudgeVerdict{Value: value, Fingerprint: fingerprint, Payload: []byte(`{"reasoning":"looks fine"}`)}, nil
	}
}

func TestScoreJudge_Good(t *core.T) {
	store, _, item := seededStore(t)
	r := ScoreJudge(context.Background(), store, fakeJudge(85, "gemma4-fp-abc", nil), item, "helpfulness")
	core.AssertTrue(t, r.OK)
	score := r.Value.(Score)
	core.AssertEqual(t, JudgeScoreKind("helpfulness"), score.Kind)
	core.AssertEqual(t, 85.0, score.Value)
	core.AssertEqual(t, "gemma4-fp-abc", score.JudgeFingerprint)
	core.AssertEqual(t, "", score.ScorerName, "a judge row's identity is carried by the fingerprint, not a scorer name")

	stored := core.MustCast[[]Score](store.Scores(item.ID))
	core.AssertLen(t, stored, 1)
}

func TestScoreJudge_Bad(t *core.T) {
	store, _, item := seededStore(t)
	r := ScoreJudge(context.Background(), store, fakeJudge(0, "", core.NewError("model unreachable")), item, "helpfulness")
	core.AssertFalse(t, r.OK, "a dispatch error must fail")

	r = ScoreJudge(context.Background(), nil, fakeJudge(85, "fp", nil), item, "helpfulness")
	core.AssertFalse(t, r.OK, "a nil store must fail")
}

func TestScoreJudge_Ugly(t *core.T) {
	store, _, item := seededStore(t)

	r := ScoreJudge(context.Background(), store, nil, item, "helpfulness")
	core.AssertFalse(t, r.OK, "a nil dispatcher must fail, not panic")

	r = ScoreJudge(context.Background(), store, fakeJudge(85, "fp", nil), item, "")
	core.AssertFalse(t, r.OK, "an empty template name must fail")

	// A malformed judge output (empty fingerprint) is a loud per-item
	// error at the validation gate, not a silently-stored garbage row.
	r = ScoreJudge(context.Background(), store, fakeJudge(85, "", nil), item, "helpfulness")
	core.AssertFalse(t, r.OK, "an empty fingerprint fails ValidateScore's identity requirement")
}
