// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"context"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/inference/cli/tui"
	"dappco.re/go/inference/dataset"
	"dappco.re/go/inference/serving/provider/openai"
)

// judgeDefaultMaxTokens bounds a judge verdict generation: the contract's
// parse rule wants a bare number back (parseJudgeScore), so a short cap
// keeps a rambling model's cost bounded without starving a well-behaved one
// — a stray "Score: 87" or "87/100" still fits comfortably.
const judgeDefaultMaxTokens = 32

// judgeVerdictPayload is the Score.Payload JSON shape a judge-tier row
// carries — the rendered template's raw model reply, kept for audit
// alongside the parsed Value.
type judgeVerdictPayload struct {
	Template string `json:"template"`
	RawReply string `json:"raw_reply"`
}

// newJudgeDispatcher is the production seam `lem data score --kind
// judge:<name> --model <path>` drives: loads modelPath ONCE (a batch CLI
// verb scoring N items never reloads per item — one load, N serial
// generate calls, the "only one model resident" constraint go/dataset's
// JudgeDispatcher doc names, sized to a one-shot command rather than the
// TUI's live queue) and resolves templates through the design's real dirs
// (~/.lem/judges/ override, the in-repo judges/ default). The caller must
// defer the returned close func.
//
//	driver, closeModel, err := newJudgeDispatcher("/models/judge-4b", 32)
//	if err != nil { return err }
//	defer closeModel()
func newJudgeDispatcher(modelPath string, maxTokens int) (dataset.JudgeDispatcher, func() error, error) {
	if core.Trim(modelPath) == "" {
		return nil, nil, core.NewError("dataset: judge driver requires a model path")
	}
	loaded := inference.LoadModel(modelPath)
	if !loaded.OK {
		return nil, nil, core.E("main.newJudgeDispatcher", "load judge model", loaded.Err())
	}
	tm, ok := loaded.Value.(inference.TextModel)
	if !ok {
		return nil, nil, core.E("main.newJudgeDispatcher", "judge model load returned an unexpected type", nil)
	}

	overrideDir := ""
	if dirResult := tui.OpenJudgesDir(); dirResult.OK {
		overrideDir = dirResult.String()
	}
	inRepoDir, _ := defaultInRepoJudgesDir()

	dispatcher := newJudgeDispatcherFromModel(tm, modelPath, maxTokens, overrideDir, inRepoDir)
	closeFn := func() error {
		r := tm.Close()
		if !r.OK {
			return r.Err()
		}
		return nil
	}
	return dispatcher, closeFn, nil
}

// newJudgeDispatcherFromModel builds the dataset.JudgeDispatcher closure
// around an already-loaded model — the pure core the stub-model tests drive
// directly (no GPU, no real load: newJudgeDispatcher is the only caller
// that touches inference.LoadModel). Resolved templates are cached by name
// for the lifetime of the closure so scoring N items against the same
// judge:<name> reads + parses the template file once, not N times.
func newJudgeDispatcherFromModel(tm inference.TextModel, fingerprint string, maxTokens int, overrideDir, inRepoDir string) dataset.JudgeDispatcher {
	if maxTokens <= 0 {
		maxTokens = judgeDefaultMaxTokens
	}
	templates := map[string]judgeTemplate{}

	return func(ctx context.Context, name string, item dataset.Item) (dataset.JudgeVerdict, error) {
		tpl, ok := templates[name]
		if !ok {
			resolved, rerr := resolveJudgeTemplateFrom(overrideDir, inRepoDir, name)
			if rerr != nil {
				return dataset.JudgeVerdict{}, rerr
			}
			templates[name] = resolved
			tpl = resolved
		}

		prompt, response, perr := judgeItemPromptResponse(item)
		if perr != nil {
			return dataset.JudgeVerdict{}, perr
		}
		rendered := renderJudgeTemplate(tpl, prompt, response)

		reply, gerr := runJudgeGenerate(ctx, tm, rendered, maxTokens)
		if gerr != nil {
			return dataset.JudgeVerdict{}, gerr
		}

		value, serr := parseJudgeScore(reply, tpl)
		if serr != nil {
			return dataset.JudgeVerdict{}, serr
		}

		payload := core.JSONMarshal(judgeVerdictPayload{Template: name, RawReply: core.Trim(reply)})
		return dataset.JudgeVerdict{Value: value, Fingerprint: fingerprint, Payload: payload.Bytes()}, nil
	}
}

// runJudgeGenerate runs one greedy (temperature 0), bounded-max-tokens
// generation against tm and returns the assistant content with any
// thinking-channel markers stripped (openai.NewThinkingExtractor — the same
// extractor every serving route runs; Gemma 4 emits an empty thought
// channel even with thinking off, and the judge reply must be clean for
// parseJudgeScore's strict bare-number rule).
func runJudgeGenerate(ctx context.Context, tm inference.TextModel, prompt string, maxTokens int) (string, error) {
	think := false
	msgs := []inference.Message{{Role: "user", Content: prompt}}
	opts := []inference.GenerateOption{
		inference.WithMaxTokens(maxTokens),
		inference.WithTemperature(0),
		inference.WithEnableThinking(&think),
	}

	var out []byte
	for tok := range tm.Chat(ctx, msgs, opts...) {
		out = append(out, tok.Text...)
	}
	if r := tm.Err(); !r.OK {
		return "", core.E("main.runJudgeGenerate", "judge generation", r.Err())
	}

	extractor := openai.NewThinkingExtractor()
	content, _ := extractor.Process(inference.Token{Text: string(out)})
	flushContent, _ := extractor.Flush()
	return content + flushContent, nil
}

// judgeItemPromptResponse extracts the (prompt, response) text pair a judge
// template renders, mirroring go/dataset's unexported
// heuristicPromptResponse over the SAME exported content shapes
// (dataset.PairContent / dataset.MessagesContent) — go/dataset never talks
// to a model directly, so this small switch is intentionally duplicated
// CLI-side rather than exported from the frozen contract.
func judgeItemPromptResponse(item dataset.Item) (string, string, error) {
	switch item.Kind {
	case dataset.KindPair:
		var pc dataset.PairContent
		if r := core.JSONUnmarshal(item.Content, &pc); !r.OK {
			return "", "", core.E("main.judgeItemPromptResponse", "parse pair content", r.Err())
		}
		return pc.Prompt, pc.Response, nil
	case dataset.KindMessages:
		var mc dataset.MessagesContent
		if r := core.JSONUnmarshal(item.Content, &mc); !r.OK {
			return "", "", core.E("main.judgeItemPromptResponse", "parse messages content", r.Err())
		}
		pc, ok := mc.LastExchange()
		if !ok {
			return "", "", core.E("main.judgeItemPromptResponse", "messages content has no assistant turn to score", nil)
		}
		return pc.Prompt, pc.Response, nil
	default:
		return "", "", core.E("main.judgeItemPromptResponse", core.Sprintf("item kind %q is not scorable by the judge tier", item.Kind), nil)
	}
}
