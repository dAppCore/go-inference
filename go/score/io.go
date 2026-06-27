// SPDX-Licence-Identifier: EUPL-1.2

package score

import (
	"bufio"

	core "dappco.re/go"
	coreio "dappco.re/go/io"
)

// ReadResponses reads a JSONL file and returns a slice of Response structs.
// Each line must be a valid JSON object. Empty lines are skipped. The scanner
// buffer is set to 1MB to handle long responses.
//
//	r := ReadResponses("/data/responses.jsonl")
//	if !r.OK { return r }
//	responses := r.Value.([]Response)
func ReadResponses(path string) core.Result {
	f, err := coreio.Local.Open(path)
	if err != nil {
		return core.Fail(core.E("ml.ReadResponses", core.Sprintf("open %s", path), err))
	}
	defer f.Close()

	var responses []Response
	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 1024*1024), 1024*1024) // 1MB buffer

	lineNum := 0
	for scanner.Scan() {
		lineNum++
		line := core.Trim(scanner.Text())
		if line == "" {
			continue
		}

		var resp Response
		if rj := core.JSONUnmarshalString(line, &resp); !rj.OK {
			return core.Fail(core.E("ml.ReadResponses", core.Sprintf("line %d", lineNum), rj.Value.(error)))
		}
		responses = append(responses, resp)
	}

	if err := scanner.Err(); err != nil {
		return core.Fail(core.E("ml.ReadResponses", core.Sprintf("scan %s", path), err))
	}

	return core.Ok(responses)
}

// ComputeAverages calculates per-model average scores across all prompts. It
// averages all numeric fields from HeuristicScores, SemanticScores,
// ContentScores, StandardScores, and the lek_score field.
func ComputeAverages(perPrompt map[string][]PromptScore) map[string]map[string]float64 {
	// Accumulate sums and counts per model per field.
	type accumulator struct {
		sums   map[string]float64
		counts map[string]int
	}
	modelAccum := make(map[string]*accumulator)

	getAccum := func(model string) *accumulator {
		if a, ok := modelAccum[model]; ok {
			return a
		}
		a := &accumulator{
			sums:   make(map[string]float64),
			counts: make(map[string]int),
		}
		modelAccum[model] = a
		return a
	}

	addField := func(a *accumulator, field string, val float64) {
		a.sums[field] += val
		a.counts[field]++
	}

	for _, scores := range perPrompt {
		for _, ps := range scores {
			a := getAccum(ps.Model)

			if h := ps.Heuristic; h != nil {
				addField(a, "compliance_markers", float64(h.ComplianceMarkers))
				addField(a, "formulaic_preamble", float64(h.FormulaicPreamble))
				addField(a, "first_person", float64(h.FirstPerson))
				addField(a, "creative_form", float64(h.CreativeForm))
				addField(a, "engagement_depth", float64(h.EngagementDepth))
				addField(a, "emotional_register", float64(h.EmotionalRegister))
				addField(a, "degeneration", float64(h.Degeneration))
				addField(a, "empty_broken", float64(h.EmptyBroken))
				addField(a, "lek_score", h.LEKScore)
			}

			if s := ps.Semantic; s != nil {
				addField(a, "sovereignty", float64(s.Sovereignty))
				addField(a, "ethical_depth", float64(s.EthicalDepth))
				addField(a, "creative_expression", float64(s.CreativeExpression))
				addField(a, "self_concept", float64(s.SelfConcept))
			}

			if c := ps.Content; c != nil {
				addField(a, "ccp_compliance", float64(c.CCPCompliance))
				addField(a, "truth_telling", float64(c.TruthTelling))
				addField(a, "engagement", float64(c.Engagement))
				addField(a, "axiom_integration", float64(c.AxiomIntegration))
				addField(a, "sovereignty_reasoning", float64(c.SovereigntyReasoning))
				addField(a, "content_emotional_register", float64(c.EmotionalRegister))
			}

			if s := ps.Standard; s != nil {
				addField(a, "truthfulness", float64(s.Truthfulness))
				addField(a, "informativeness", float64(s.Informativeness))
				addField(a, "safety", float64(s.Safety))
				addField(a, "nuance", float64(s.Nuance))
				addField(a, "kindness", float64(s.Kindness))
				addField(a, "awareness", float64(s.Awareness))
				if s.Correct != nil {
					if *s.Correct {
						addField(a, "correct", 1)
					} else {
						addField(a, "correct", 0)
					}
				}
			}
		}
	}

	// Compute averages.
	result := make(map[string]map[string]float64)
	for model, a := range modelAccum {
		avgs := make(map[string]float64)
		for field, sum := range a.sums {
			avgs[field] = sum / float64(a.counts[field])
		}
		result[model] = avgs
	}

	return result
}
