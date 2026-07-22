// SPDX-Licence-Identifier: EUPL-1.2

package dataset

import (
	"context"
	"strconv"
	"time"

	core "dappco.re/go"
	"dappco.re/go/inference/eval/score/lek"
)

// heuristicScorerName + heuristicScorerVersion identify the heuristic
// tier's Score rows — bump the version when the detector logic changes
// materially enough that historic rows should not be treated as
// directly comparable to fresh ones.
const (
	heuristicScorerName    = "lek"
	heuristicScorerVersion = "1"
)

// ScoreHeuristic computes the heuristic tier's Score rows for one
// item's content — lek.ScorePair, per the design's "Heuristic tier
// (default, no model needed)": one ScoreKindLEK row (composite value +
// the full lek.DiffResult payload) plus derived ScoreKindHostility /
// ScoreKindSycophancy rows for direct filtering. Pure — no store I/O;
// call [ScoreHeuristicAppend] to persist the result.
//
// KindMessages items are reduced to a prompt/response pair via
// [MessagesContent.LastExchange] — the same trailing-assistant-turn
// reduction the pairs-jsonl and capture-jsonl export writers use.
// KindTrace items are opaque and not scorable by this tier.
//
//	r := dataset.ScoreHeuristic(item)
//	scores := r.Value.([]dataset.Score)
func ScoreHeuristic(item Item) core.Result {
	prompt, response, r := heuristicPromptResponse(item)
	if !r.OK {
		return r
	}

	diff := lek.ScorePair(prompt, response)
	now := time.Now()

	diffPayload := core.JSONMarshal(diff)
	if !diffPayload.OK {
		return core.Fail(core.E("dataset.ScoreHeuristic", "marshal lek diff payload", diffPayload.Err()))
	}
	hostilityPayload := core.JSONMarshal(diff.Response.Hostility)
	if !hostilityPayload.OK {
		return core.Fail(core.E("dataset.ScoreHeuristic", "marshal hostility payload", hostilityPayload.Err()))
	}
	sycophancyPayload := core.JSONMarshal(diff.Response.Sycophancy)
	if !sycophancyPayload.OK {
		return core.Fail(core.E("dataset.ScoreHeuristic", "marshal sycophancy payload", sycophancyPayload.Err()))
	}

	scores := []Score{
		{
			ID: NewID(), ItemID: item.ID, Kind: ScoreKindLEK, Value: diff.Response.LEK.LEKScore,
			Payload: diffPayload.Bytes(), ScorerName: heuristicScorerName, ScorerVersion: heuristicScorerVersion, CreatedAt: now,
		},
		{
			ID: NewID(), ItemID: item.ID, Kind: ScoreKindHostility, Value: diff.Response.Hostility.Score,
			Payload: hostilityPayload.Bytes(), ScorerName: heuristicScorerName, ScorerVersion: heuristicScorerVersion, CreatedAt: now,
		},
		{
			ID: NewID(), ItemID: item.ID, Kind: ScoreKindSycophancy, Value: diff.Response.Sycophancy.Composite,
			Payload: sycophancyPayload.Bytes(), ScorerName: heuristicScorerName, ScorerVersion: heuristicScorerVersion, CreatedAt: now,
		},
	}
	return core.Ok(scores)
}

// heuristicPromptResponse extracts the (prompt, response) text pair
// ScoreHeuristic scores, per item Kind.
func heuristicPromptResponse(item Item) (string, string, core.Result) {
	switch item.Kind {
	case KindPair:
		var pc PairContent
		if r := core.JSONUnmarshal(item.Content, &pc); !r.OK {
			return "", "", core.Fail(core.E("dataset.ScoreHeuristic", "parse pair content", r.Err()))
		}
		return pc.Prompt, pc.Response, core.Ok(nil)
	case KindMessages:
		var mc MessagesContent
		if r := core.JSONUnmarshal(item.Content, &mc); !r.OK {
			return "", "", core.Fail(core.E("dataset.ScoreHeuristic", "parse messages content", r.Err()))
		}
		pc, ok := mc.LastExchange()
		if !ok {
			return "", "", core.Fail(core.NewError("dataset: messages content has no assistant turn to score"))
		}
		return pc.Prompt, pc.Response, core.Ok(nil)
	default:
		return "", "", core.Fail(core.NewError("dataset: item kind is not scorable by the heuristic tier"))
	}
}

// ScoreHeuristicAppend runs [ScoreHeuristic] and appends each resulting
// row to store — the orchestration `lem data score` (Task 6) wires.
//
//	r := dataset.ScoreHeuristicAppend(store, item)
func ScoreHeuristicAppend(store Store, item Item) core.Result {
	if store == nil {
		return core.Fail(core.NewError("dataset: score requires a store"))
	}
	r := ScoreHeuristic(item)
	if !r.OK {
		return r
	}
	scores := r.Value.([]Score)
	for _, s := range scores {
		if appended := store.ScoreAppend(s); !appended.OK {
			return appended
		}
	}
	return core.Ok(scores)
}

// scoreExpressionOps lists the operators [ParseScoreExpression] accepts,
// longest first so ">=" is matched whole rather than as ">" followed by
// a dangling "=".
var scoreExpressionOps = []ComparisonOp{OpGTE, OpLTE, OpEQ, OpNEQ, OpGT, OpLT}

// findComparisonOp finds the earliest-occurring operator in expr,
// breaking ties (an operator that is itself a prefix of a longer one at
// the same position, e.g. ">" inside ">=") in favour of the longer
// match.
func findComparisonOp(expr string) (ComparisonOp, int) {
	bestIdx := -1
	var bestOp ComparisonOp
	for _, op := range scoreExpressionOps {
		idx := core.Index(expr, string(op))
		if idx < 0 {
			continue
		}
		if bestIdx < 0 || idx < bestIdx || (idx == bestIdx && len(op) > len(bestOp)) {
			bestIdx = idx
			bestOp = op
		}
	}
	return bestOp, len(bestOp)
}

// ParseScoreExpression parses the tiny explicit grammar — field, op,
// number, nothing more — used by --auto-approve/--auto-reject flags and
// Store item-list score filters: "<field><op><number>", surrounding
// whitespace tolerated, nothing else. field must name a known
// [ScoreKind] (a heuristic kind or a well-formed judge:<name> kind); op
// is one of >= <= > < == !=; number parses as a float64.
//
//	r := dataset.ParseScoreExpression("lek>=80")
//	expr := r.Value.(dataset.ScoreExpression)
func ParseScoreExpression(expr string) core.Result {
	op, opLen := findComparisonOp(expr)
	if op == "" {
		return core.Fail(core.NewError("dataset: score expression requires one of >= <= > < == !="))
	}
	idx := core.Index(expr, string(op))
	field := core.Trim(expr[:idx])
	rest := core.Trim(expr[idx+opLen:])
	if field == "" || rest == "" {
		return core.Fail(core.NewError("dataset: score expression requires <field><op><number>"))
	}
	kind := ScoreKind(field)
	if !knownScoreKind(kind) {
		return core.Fail(core.E("dataset.ParseScoreExpression", core.Sprintf("unknown score field %q", field), nil))
	}
	threshold, err := strconv.ParseFloat(rest, 64)
	if err != nil {
		return core.Fail(core.E("dataset.ParseScoreExpression", core.Sprintf("invalid threshold %q", rest), err))
	}
	return core.Ok(ScoreExpression{Kind: kind, Op: op, Threshold: threshold})
}

// AutoThresholdResult reports what [ApplyAutoThreshold] did.
type AutoThresholdResult struct {
	Applied bool
	Review  Review
}

// ApplyAutoThreshold evaluates approve/reject expressions against
// item's current scores and, if either is satisfied, appends the
// matching auto:threshold Review row. Explicit only: a nil expression is
// never evaluated — no silent policy, matching the design's "Auto-review
// thresholds are explicit, never implicit". reject is checked first: an
// item that would satisfy both a reject and an approve expression is
// never accidentally auto-approved.
//
//	r := dataset.ApplyAutoThreshold(store, item, scores, approve, reject)
func ApplyAutoThreshold(store Store, item Item, scores []Score, approve, reject *ScoreExpression) core.Result {
	if store == nil {
		return core.Fail(core.NewError("dataset: auto-threshold requires a store"))
	}
	if reject != nil && reject.Matches(scores) {
		return appendAutoThresholdReview(store, item.ID, StatusRejected, reject.String())
	}
	if approve != nil && approve.Matches(scores) {
		return appendAutoThresholdReview(store, item.ID, StatusApproved, approve.String())
	}
	return core.Ok(AutoThresholdResult{})
}

func appendAutoThresholdReview(store Store, itemID string, status ReviewStatus, note string) core.Result {
	review := Review{ItemID: itemID, Status: status, Reviewer: ReviewerAutoThreshold, Note: note, CreatedAt: time.Now()}
	if r := store.ReviewAppend(review); !r.OK {
		return r
	}
	return core.Ok(AutoThresholdResult{Applied: true, Review: review})
}

// JudgeVerdict is one judge tier's fully-parsed verdict for an item —
// what a [JudgeDispatcher] hands back after rendering the named
// template, dispatching it through the shared serial model lane, and
// parsing the model's reply. Fingerprint is the judge model's
// fingerprint — the provenance rule: every score names what produced
// it.
type JudgeVerdict struct {
	Value       float64
	Fingerprint string
	Payload     []byte
}

// JudgeDispatcher is the seam a CLI-side driver implements: given a
// named judge template and the item being scored, render the template,
// run it through the shared serial model lane (queued like any
// generation job — the TUI stays responsive, only one model resident),
// and return the fully-parsed verdict. go/dataset never talks to a
// model directly — template storage/rendering and the live model call
// are both CLI-side (judges/ + ~/.lem/judges/ overrides, Task 9).
type JudgeDispatcher func(ctx context.Context, template string, item Item) (JudgeVerdict, error)

// ScoreJudge runs the judge tier for one item: dispatches through
// driver and appends the resulting Score row (kind judge:<template>,
// scorer identity carried by JudgeVerdict.Fingerprint).
//
//	r := dataset.ScoreJudge(ctx, store, driver, item, "helpfulness")
func ScoreJudge(ctx context.Context, store Store, driver JudgeDispatcher, item Item, template string) core.Result {
	if store == nil {
		return core.Fail(core.NewError("dataset: judge score requires a store"))
	}
	if driver == nil {
		return core.Fail(core.NewError("dataset: judge score requires a dispatcher"))
	}
	if core.Trim(template) == "" {
		return core.Fail(core.NewError("dataset: judge score requires a template name"))
	}
	verdict, err := driver(ctx, template, item)
	if err != nil {
		return core.Fail(core.E("dataset.ScoreJudge", "judge dispatch failed", err))
	}
	score := Score{
		ID: NewID(), ItemID: item.ID, Kind: JudgeScoreKind(template), Value: verdict.Value,
		Payload: verdict.Payload, JudgeFingerprint: verdict.Fingerprint, CreatedAt: time.Now(),
	}
	if r := ValidateScore(score); !r.OK {
		return r
	}
	return store.ScoreAppend(score)
}
