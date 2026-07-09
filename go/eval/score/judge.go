package score

import (
	"context"
	"regexp"

	"dappco.re/go"
	"dappco.re/go/inference/serving"
)

// codeBlockRe matches a fenced markdown code block, optionally tagged json,
// capturing its body. Compiled once at package load — recompiling this
// constant pattern per extractJSON call cost ~45 allocs on every judge
// response.
var codeBlockRe = regexp.MustCompile("(?s)```(?:json)?\\s*\\n?(.*?)\\s*\\n?```")

// extractJSON extracts the first JSON object {...} from text.
// Handles raw JSON, JSON surrounded by text, markdown code blocks, etc.
// Returns "" if no JSON object is found.
func extractJSON(text string) string {
	// First, try to extract from markdown code blocks.
	if m := codeBlockRe.FindStringSubmatch(text); len(m) > 1 {
		if raw := firstJSONObject(m[1]); raw != "" {
			return core.Trim(raw)
		}
	}

	return firstJSONObject(text)
}

// firstJSONObject finds the first balanced JSON object in text.
func firstJSONObject(text string) string {
	start := -1
	for i := 0; i < len(text); i++ {
		if text[i] == '{' {
			start = i
			break
		}
	}
	if start == -1 {
		return ""
	}

	depth := 0
	for i := start; i < len(text); i++ {
		switch text[i] {
		case '{':
			depth++
		case '}':
			depth--
			if depth == 0 {
				return text[start : i+1]
			}
		}
	}

	return ""
}

// normalizeBenchmarkName collapses benchmark aliases to a canonical form.
// It tolerates mixed case as well as spaces, underscores, and hyphens.
func normalizeBenchmarkName(name string) string {
	normalized := core.Lower(core.Trim(name))
	// Strip '_', '-', and ' ' in a single pass. The three sequential
	// ReplaceAll calls each allocated a fresh string; separators are ASCII,
	// so a byte scan is byte-identical. Skip the build entirely when none
	// are present (the canonical-name common case stays zero-alloc).
	hasSep := false
	for i := 0; i < len(normalized); i++ {
		if c := normalized[i]; c == '_' || c == '-' || c == ' ' {
			hasSep = true
			break
		}
	}
	if !hasSep {
		return normalized
	}
	var b core.Builder
	b.Grow(len(normalized))
	for i := 0; i < len(normalized); i++ {
		if c := normalized[i]; c != '_' && c != '-' && c != ' ' {
			b.WriteByte(c)
		}
	}
	return b.String()
}

// Judge uses an LLM backend to score responses across multiple dimensions.
type Judge struct {
	backend serving.Backend
	Model   string // model name for metadata
	BaseURL string // base URL for metadata
}

// NewJudge creates a Judge backed by any serving.Backend implementation.
func NewJudge(backend serving.Backend) *Judge {
	j := &Judge{backend: backend}
	// Extract metadata from *serving.HTTPBackend if available.
	if h, ok := backend.(*serving.HTTPBackend); ok {
		j.Model = h.Model()
		j.BaseURL = h.BaseURL()
	}
	return j
}

// judgeChat sends a formatted prompt to the judge backend and returns the raw response.
//
//	r := j.judgeChat(ctx, prompt)
//	if !r.OK { return r }
//	reply := r.Value.(string)
func (j *Judge) judgeChat(ctx context.Context, prompt string) core.Result {
	r := j.backend.Generate(ctx, prompt, serving.DefaultGenOpts())
	if !r.OK {
		return r
	}
	return core.Ok(r.Value.(serving.Result).Text)
}

// ScoreSemantic scores a response on sovereignty, ethical depth, creative
// expression, and self-concept using the semantic judge prompt.
//
//	r := judge.ScoreSemantic(ctx, prompt, response)
//	if !r.OK { return r }
//	scores := r.Value.(*score.SemanticScores)
func (j *Judge) ScoreSemantic(ctx context.Context, prompt, response string) core.Result {
	formatted := core.Sprintf(semanticPrompt, prompt, response)

	rChat := j.judgeChat(ctx, formatted)
	if !rChat.OK {
		return core.Fail(core.E("score.Judge.ScoreSemantic", "semantic judge chat", rChat.Value.(error)))
	}
	reply := rChat.Value.(string)

	raw := extractJSON(reply)
	if raw == "" {
		return core.Fail(core.E("score.Judge.ScoreSemantic", core.Sprintf("no JSON found in semantic judge response: %s", reply), nil))
	}

	var scores SemanticScores
	if r := core.JSONUnmarshalString(raw, &scores); !r.OK {
		return core.Fail(core.E("score.Judge.ScoreSemantic", "unmarshal semantic scores", r.Value.(error)))
	}

	return core.Ok(&scores)
}

// ScoreContent scores a response on content/sovereignty dimensions using
// the content judge prompt with CCP and truth markers.
//
//	r := judge.ScoreContent(ctx, probe, response)
//	if !r.OK { return r }
//	scores := r.Value.(*score.ContentScores)
func (j *Judge) ScoreContent(ctx context.Context, probe ContentProbe, response string) core.Result {
	ccpMarkers := core.Join(", ", probe.CCPMarkers...)
	truthMarkers := core.Join(", ", probe.TruthMarkers...)

	formatted := core.Sprintf(contentPrompt, probe.Prompt, probe.GroundTruth, ccpMarkers, truthMarkers, response)

	rChat := j.judgeChat(ctx, formatted)
	if !rChat.OK {
		return core.Fail(core.E("score.Judge.ScoreContent", "content judge chat", rChat.Value.(error)))
	}
	reply := rChat.Value.(string)

	raw := extractJSON(reply)
	if raw == "" {
		return core.Fail(core.E("score.Judge.ScoreContent", core.Sprintf("no JSON found in content judge response: %s", reply), nil))
	}

	var scores ContentScores
	if r := core.JSONUnmarshalString(raw, &scores); !r.OK {
		return core.Fail(core.E("score.Judge.ScoreContent", "unmarshal content scores", r.Value.(error)))
	}

	return core.Ok(&scores)
}

// ScoreCapability scores a capability probe response on reasoning, correctness,
// and clarity using the capability judge prompt.
//
//	r := judge.ScoreCapability(ctx, prompt, expected, response)
//	if !r.OK { return r }
//	scores := r.Value.(*score.CapabilityScores)
func (j *Judge) ScoreCapability(ctx context.Context, prompt, expectedAnswer, response string) core.Result {
	formatted := core.Sprintf(capabilityPrompt, prompt, expectedAnswer, response)

	rChat := j.judgeChat(ctx, formatted)
	if !rChat.OK {
		return core.Fail(core.E("score.Judge.ScoreCapability", "capability judge chat", rChat.Value.(error)))
	}
	reply := rChat.Value.(string)

	raw := extractJSON(reply)
	if raw == "" {
		return core.Fail(core.E("score.Judge.ScoreCapability", core.Sprintf("no JSON found in capability judge response: %s", reply), nil))
	}

	var scores CapabilityScores
	if r := core.JSONUnmarshalString(raw, &scores); !r.OK {
		return core.Fail(core.E("score.Judge.ScoreCapability", "unmarshal capability scores", r.Value.(error)))
	}

	return core.Ok(&scores)
}

// ScoreTruthfulQA scores a response on truthfulness and informativeness
// using the TruthfulQA judge prompt.
//
//	r := judge.ScoreTruthfulQA(ctx, question, bestAnswer, response)
//	if !r.OK { return r }
//	scores := r.Value.(*score.StandardScores)
func (j *Judge) ScoreTruthfulQA(ctx context.Context, question, bestAnswer, response string) core.Result {
	formatted := core.Sprintf(truthfulQAPrompt, question, bestAnswer, response)

	rChat := j.judgeChat(ctx, formatted)
	if !rChat.OK {
		return core.Fail(core.E("score.Judge.ScoreTruthfulQA", "truthfulqa judge chat", rChat.Value.(error)))
	}
	reply := rChat.Value.(string)

	raw := extractJSON(reply)
	if raw == "" {
		return core.Fail(core.E("score.Judge.ScoreTruthfulQA", core.Sprintf("no JSON found in truthfulqa judge response: %s", reply), nil))
	}

	var scores StandardScores
	if r := core.JSONUnmarshalString(raw, &scores); !r.OK {
		return core.Fail(core.E("score.Judge.ScoreTruthfulQA", "unmarshal truthfulqa scores", r.Value.(error)))
	}

	return core.Ok(&scores)
}

// ScoreDoNotAnswer scores a response on safety and nuance using the
// DoNotAnswer judge prompt.
//
//	r := judge.ScoreDoNotAnswer(ctx, question, riskArea, response)
//	if !r.OK { return r }
//	scores := r.Value.(*score.StandardScores)
func (j *Judge) ScoreDoNotAnswer(ctx context.Context, question, riskArea, response string) core.Result {
	formatted := core.Sprintf(doNotAnswerPrompt, question, riskArea, response)

	rChat := j.judgeChat(ctx, formatted)
	if !rChat.OK {
		return core.Fail(core.E("score.Judge.ScoreDoNotAnswer", "donotanswer judge chat", rChat.Value.(error)))
	}
	reply := rChat.Value.(string)

	raw := extractJSON(reply)
	if raw == "" {
		return core.Fail(core.E("score.Judge.ScoreDoNotAnswer", core.Sprintf("no JSON found in donotanswer judge response: %s", reply), nil))
	}

	var scores StandardScores
	if r := core.JSONUnmarshalString(raw, &scores); !r.OK {
		return core.Fail(core.E("score.Judge.ScoreDoNotAnswer", "unmarshal donotanswer scores", r.Value.(error)))
	}

	return core.Ok(&scores)
}

// ScoreStandard is the unified benchmark-judge entry point described in
// RFC §4.4. It dispatches to the underlying benchmark suite based on the
// benchmark arg and returns the same StandardScores shape. Accepted
// benchmark names: "truthfulqa", "helm", "mmlu", "hellaswag",
// "donotanswer", "toxigen", "exact".
//
//	r := judge.ScoreStandard(ctx, "truthfulqa", "What is 2+2?", "4", response)
//	if !r.OK { return r }
//	scores := r.Value.(*score.StandardScores)
func (j *Judge) ScoreStandard(ctx context.Context, benchmark, question, reference, response string) core.Result {
	switch normalizeBenchmarkName(benchmark) {
	case "truthfulqa", "helm", "mmlu", "hellaswag":
		return j.ScoreTruthfulQA(ctx, question, reference, response)
	case "donotanswer":
		return j.ScoreDoNotAnswer(ctx, question, reference, response)
	case "toxigen":
		return j.ScoreToxigen(ctx, question, response)
	case "exact", "gsm8k":
		// Pure string/number match — no judge call required.
		return core.Ok(scoreGSM8K(response, reference))
	default:
		return core.Fail(core.E("score.Judge.ScoreStandard",
			core.Sprintf("unknown benchmark %q (want truthfulqa|helm|mmlu|hellaswag|do_not_answer|toxigen|exact)", benchmark), nil))
	}
}

// ScoreToxigen scores a response on kindness and awareness using the
// Toxigen judge prompt.
//
//	r := judge.ScoreToxigen(ctx, prompt, response)
//	if !r.OK { return r }
//	scores := r.Value.(*score.StandardScores)
func (j *Judge) ScoreToxigen(ctx context.Context, prompt, response string) core.Result {
	formatted := core.Sprintf(toxigenPrompt, prompt, response)

	rChat := j.judgeChat(ctx, formatted)
	if !rChat.OK {
		return core.Fail(core.E("score.Judge.ScoreToxigen", "toxigen judge chat", rChat.Value.(error)))
	}
	reply := rChat.Value.(string)

	raw := extractJSON(reply)
	if raw == "" {
		return core.Fail(core.E("score.Judge.ScoreToxigen", core.Sprintf("no JSON found in toxigen judge response: %s", reply), nil))
	}

	var scores StandardScores
	if r := core.JSONUnmarshalString(raw, &scores); !r.OK {
		return core.Fail(core.E("score.Judge.ScoreToxigen", "unmarshal toxigen scores", r.Value.(error)))
	}

	return core.Ok(&scores)
}
