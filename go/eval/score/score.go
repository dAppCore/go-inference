package score

import (
	"context"
	"maps"
	"slices"
	"sync"

	"dappco.re/go"
)

// Engine orchestrates concurrent scoring across multiple suites.
type Engine struct {
	judge       *Judge
	concurrency int
	suites      map[string]bool // which suites to run
}

// NewEngine creates an Engine that runs the specified suites concurrently.
// suiteList is comma-separated (e.g. "heuristic,semantic") or "all".
func NewEngine(judge *Judge, concurrency int, suiteList string) *Engine {
	suites := make(map[string]bool)

	if core.Lower(core.Trim(suiteList)) == "all" {
		suites["heuristic"] = true
		suites["semantic"] = true
		suites["content"] = true
		suites["standard"] = true
		suites["exact"] = true
	} else {
		for _, s := range core.Split(suiteList, ",") {
			s = core.Lower(core.Trim(s))
			if s != "" {
				suites[s] = true
			}
		}
	}

	return &Engine{
		judge:       judge,
		concurrency: concurrency,
		suites:      suites,
	}
}

// ScoreHeuristic runs the heuristic suite directly through the engine.
func (e *Engine) ScoreHeuristic(response string) *HeuristicScores {
	return ScoreHeuristic(response)
}

// ScoreSemantic delegates to the configured judge.
//
//	r := e.ScoreSemantic(ctx, prompt, response)
//	if !r.OK { return r }
//	scores := r.Value.(*score.SemanticScores)
func (e *Engine) ScoreSemantic(ctx context.Context, prompt, response string) core.Result {
	if e == nil || e.judge == nil {
		return core.Fail(core.E("score.Engine.ScoreSemantic", "semantic scoring requires a judge", nil))
	}
	return e.judge.ScoreSemantic(ctx, prompt, response)
}

// ScoreContent delegates to the configured judge.
//
//	r := e.ScoreContent(ctx, probe, response)
//	if !r.OK { return r }
//	scores := r.Value.(*score.ContentScores)
func (e *Engine) ScoreContent(ctx context.Context, probe ContentProbe, response string) core.Result {
	if e == nil || e.judge == nil {
		return core.Fail(core.E("score.Engine.ScoreContent", "content scoring requires a judge", nil))
	}
	return e.judge.ScoreContent(ctx, probe, response)
}

// ScoreCapability delegates to the configured judge.
//
//	r := e.ScoreCapability(ctx, prompt, expected, response)
//	if !r.OK { return r }
//	scores := r.Value.(*score.CapabilityScores)
func (e *Engine) ScoreCapability(ctx context.Context, prompt, expectedAnswer, response string) core.Result {
	if e == nil || e.judge == nil {
		return core.Fail(core.E("score.Engine.ScoreCapability", "capability scoring requires a judge", nil))
	}
	return e.judge.ScoreCapability(ctx, prompt, expectedAnswer, response)
}

// ScoreStandard delegates to the configured judge.
//
//	r := e.ScoreStandard(ctx, "truthfulqa", question, reference, response)
//	if !r.OK { return r }
//	scores := r.Value.(*score.StandardScores)
func (e *Engine) ScoreStandard(ctx context.Context, benchmark, question, reference, response string) core.Result {
	if e == nil || e.judge == nil {
		return core.Fail(core.E("score.Engine.ScoreStandard", "standard scoring requires a judge", nil))
	}
	return e.judge.ScoreStandard(ctx, benchmark, question, reference, response)
}

// ScoreExact runs exact-match scoring through the engine helper.
func (e *Engine) ScoreExact(response, correctAnswer string) float64 {
	return ScoreExact(response, correctAnswer)
}

// ScoreAll scores all responses grouped by model. Heuristic scoring runs
// inline (instant). LLM judge calls fan out through a worker pool bounded
// by the engine's concurrency setting.
func (e *Engine) ScoreAll(ctx context.Context, responses []Response) map[string][]PromptScore {
	if e == nil {
		return map[string][]PromptScore{}
	}

	results := make(map[string][]PromptScore)
	judge := e.judge
	concurrency := e.concurrency
	if concurrency <= 0 {
		concurrency = 1
	}

	// Pre-allocate score slots so goroutines can write to them via pointer.
	scoreSlots := make([]PromptScore, len(responses))
	for i, resp := range responses {
		scoreSlots[i] = PromptScore{
			ID:    resp.ID,
			Model: resp.Model,
		}

		// Run heuristic inline (no goroutine needed, instant).
		if e.suites["heuristic"] {
			scoreSlots[i].Heuristic = ScoreHeuristic(resp.Response)
		}
	}

	// Fan out LLM judge calls through worker pool.
	sem := make(chan struct{}, concurrency)
	var wg sync.WaitGroup
	var mu sync.Mutex

	for i, resp := range responses {
		domain := normalizeBenchmarkName(resp.Domain)

		// Semantic scoring.
		if e.suites["semantic"] {
			wg.Add(1)
			go func(r Response, ps *PromptScore) {
				defer wg.Done()
				if judge == nil {
					core.Print(nil, "semantic scoring skipped for %s: no judge configured", r.ID)
					return
				}
				sem <- struct{}{}
				defer func() { <-sem }()

				rScore := judge.ScoreSemantic(ctx, r.Prompt, r.Response)
				if !rScore.OK {
					core.Print(nil, "semantic scoring failed for %s: %v", r.ID, rScore.Error())
					return
				}
				mu.Lock()
				ps.Semantic = rScore.Value.(*SemanticScores)
				mu.Unlock()
			}(resp, &scoreSlots[i])
		}

		// Content scoring — only for content probe responses (domain == "content").
		if e.suites["content"] && domain == "content" {
			wg.Add(1)
			go func(r Response, ps *PromptScore) {
				defer wg.Done()
				if judge == nil {
					core.Print(nil, "content scoring skipped for %s: no judge configured", r.ID)
					return
				}
				sem <- struct{}{}
				defer func() { <-sem }()

				// Find the matching content probe.
				var probe *ContentProbe
				for idx := range ContentProbes {
					if ContentProbes[idx].ID == r.ID {
						probe = &ContentProbes[idx]
						break
					}
				}
				if probe == nil {
					core.Print(nil, "no content probe found for id %s", r.ID)
					return
				}

				rScore := judge.ScoreContent(ctx, *probe, r.Response)
				if !rScore.OK {
					core.Print(nil, "content scoring failed for %s: %v", r.ID, rScore.Error())
					return
				}
				mu.Lock()
				ps.Content = rScore.Value.(*ContentScores)
				mu.Unlock()
			}(resp, &scoreSlots[i])
		}

		// Standard scoring — based on response metadata.
		if e.suites["standard"] {
			// TruthfulQA: has BestAnswer field.
			if resp.BestAnswer != "" {
				wg.Add(1)
				go func(r Response, ps *PromptScore) {
					defer wg.Done()
					if judge == nil {
						core.Print(nil, "truthfulqa scoring skipped for %s: no judge configured", r.ID)
						return
					}
					sem <- struct{}{}
					defer func() { <-sem }()

					rScore := judge.ScoreTruthfulQA(ctx, r.Prompt, r.BestAnswer, r.Response)
					if !rScore.OK {
						core.Print(nil, "truthfulqa scoring failed for %s: %v", r.ID, rScore.Error())
						return
					}
					mu.Lock()
					ps.Standard = mergeStandardScores(ps.Standard, rScore.Value.(*StandardScores))
					mu.Unlock()
				}(resp, &scoreSlots[i])
			}

			// DoNotAnswer: has RiskArea field.
			if resp.RiskArea != "" {
				wg.Add(1)
				go func(r Response, ps *PromptScore) {
					defer wg.Done()
					if judge == nil {
						core.Print(nil, "donotanswer scoring skipped for %s: no judge configured", r.ID)
						return
					}
					sem <- struct{}{}
					defer func() { <-sem }()

					rScore := judge.ScoreDoNotAnswer(ctx, r.Prompt, r.RiskArea, r.Response)
					if !rScore.OK {
						core.Print(nil, "donotanswer scoring failed for %s: %v", r.ID, rScore.Error())
						return
					}
					mu.Lock()
					ps.Standard = mergeStandardScores(ps.Standard, rScore.Value.(*StandardScores))
					mu.Unlock()
				}(resp, &scoreSlots[i])
			}

			// Toxigen: domain is "toxigen".
			if domain == "toxigen" {
				wg.Add(1)
				go func(r Response, ps *PromptScore) {
					defer wg.Done()
					if judge == nil {
						core.Print(nil, "toxigen scoring skipped for %s: no judge configured", r.ID)
						return
					}
					sem <- struct{}{}
					defer func() { <-sem }()

					rScore := judge.ScoreToxigen(ctx, r.Prompt, r.Response)
					if !rScore.OK {
						core.Print(nil, "toxigen scoring failed for %s: %v", r.ID, rScore.Error())
						return
					}
					mu.Lock()
					ps.Standard = mergeStandardScores(ps.Standard, rScore.Value.(*StandardScores))
					mu.Unlock()
				}(resp, &scoreSlots[i])
			}
		}

		// Exact match scoring — GSM8K (has CorrectAnswer).
		if e.suites["exact"] && resp.CorrectAnswer != "" {
			mu.Lock()
			scoreSlots[i].Standard = mergeStandardScores(scoreSlots[i].Standard, scoreGSM8K(resp.Response, resp.CorrectAnswer))
			mu.Unlock()
		}
	}

	wg.Wait()

	// Group results by model.
	mu.Lock()
	defer mu.Unlock()
	for _, ps := range scoreSlots {
		results[ps.Model] = append(results[ps.Model], ps)
	}

	return results
}

// SuiteNames returns the enabled suite names as a sorted slice.
func (e *Engine) SuiteNames() []string {
	return slices.Sorted(maps.Keys(e.suites))
}

// String returns a human-readable description of the engine configuration.
func (e *Engine) String() string {
	return core.Sprintf("Engine(concurrency=%d, suites=%v)", e.concurrency, e.SuiteNames())
}

// ScoreSemantic evaluates a response with the supplied judge using a
// background context.
//
//	r := score.ScoreSemantic(judge, prompt, response)
//	if !r.OK { return r }
//	scores := r.Value.(*score.SemanticScores)
func ScoreSemantic(judge *Judge, prompt, response string) core.Result {
	if judge == nil {
		return core.Fail(core.E("score.ScoreSemantic", "semantic scoring requires a judge", nil))
	}
	return judge.ScoreSemantic(context.Background(), prompt, response)
}

// ScoreContent evaluates a content probe response with the supplied judge
// using a background context.
//
//	r := score.ScoreContent(judge, probe, response)
//	if !r.OK { return r }
//	scores := r.Value.(*score.ContentScores)
func ScoreContent(judge *Judge, probe ContentProbe, response string) core.Result {
	if judge == nil {
		return core.Fail(core.E("score.ScoreContent", "content scoring requires a judge", nil))
	}
	return judge.ScoreContent(context.Background(), probe, response)
}

// ScoreCapability evaluates a capability probe response with the supplied
// judge using a background context.
//
//	r := score.ScoreCapability(judge, prompt, expected, response)
//	if !r.OK { return r }
//	scores := r.Value.(*score.CapabilityScores)
func ScoreCapability(judge *Judge, prompt, expectedAnswer, response string) core.Result {
	if judge == nil {
		return core.Fail(core.E("score.ScoreCapability", "capability scoring requires a judge", nil))
	}
	return judge.ScoreCapability(context.Background(), prompt, expectedAnswer, response)
}

// ScoreStandard evaluates a benchmark response with the supplied judge using
// a background context.
//
//	r := score.ScoreStandard(judge, "truthfulqa", question, reference, response)
//	if !r.OK { return r }
//	scores := r.Value.(*score.StandardScores)
func ScoreStandard(judge *Judge, benchmark, question, reference, response string) core.Result {
	if judge == nil {
		return core.Fail(core.E("score.ScoreStandard", "standard scoring requires a judge", nil))
	}
	return judge.ScoreStandard(context.Background(), benchmark, question, reference, response)
}

// mergeStandardScores combines benchmark and exact-match results into one
// StandardScores struct without discarding fields populated by earlier suites.
func mergeStandardScores(dst, src *StandardScores) *StandardScores {
	if src == nil {
		return dst
	}
	if dst == nil {
		copy := *src
		return &copy
	}

	if src.Truthfulness != 0 {
		dst.Truthfulness = src.Truthfulness
	}
	if src.Informativeness != 0 {
		dst.Informativeness = src.Informativeness
	}
	if src.Safety != 0 {
		dst.Safety = src.Safety
	}
	if src.Nuance != 0 {
		dst.Nuance = src.Nuance
	}
	if src.Kindness != 0 {
		dst.Kindness = src.Kindness
	}
	if src.Awareness != 0 {
		dst.Awareness = src.Awareness
	}
	if src.Correct != nil {
		dst.Correct = src.Correct
	}
	if src.Extracted != "" {
		dst.Extracted = src.Extracted
	}
	if src.Expected != "" {
		dst.Expected = src.Expected
	}
	if src.Reasoning != "" && dst.Reasoning == "" {
		dst.Reasoning = src.Reasoning
	}

	return dst
}
