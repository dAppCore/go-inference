// SPDX-Licence-Identifier: EUPL-1.2

// Reasoning split: separates a think-block from the answer (the parse reasoning concern).

package parse

import core "dappco.re/go"

// ReasoningParser splits a `<think>…</think>`-style reasoning block from the
// answer. The token pair is configurable; ForceReasoning makes the leading text
// reasoning even with no opener (DeepSeek-R1 style — the model starts thinking
// immediately). It mirrors SGLang's BaseReasoningFormatDetector.detect_and_parse.
//
//	p := parse.ReasoningParser{ThinkStart: "<think>", ThinkEnd: "</think>"}
//	reasoning, answer := p.Parse(out)
type ReasoningParser struct {
	ThinkStart     string
	ThinkEnd       string
	ForceReasoning bool
}

// Gemma4Reasoning is a ReasoningParser with the default think tokens. SGLang's
// own Gemma4 reasoning detector uses obscure `<|channel>`/`<channel|>` tokens
// plus a "thought\n" self-label; the task brief calls for the conventional
// `<think>`/`</think>` pair and a clean design, so this constructor uses those
// (the field is configurable for callers that need the channel tokens).
//
//	reasoning, answer := parse.Gemma4Reasoning().Parse(out)
func Gemma4Reasoning() ReasoningParser {
	return ReasoningParser{ThinkStart: "<think>", ThinkEnd: "</think>", ForceReasoning: false}
}

// Parse returns the reasoning block and the answer content. With no reasoning
// (no opener and not forced) reasoning is "" and content is the whole text. A
// block that opens but never closes is treated as truncated reasoning: all of it
// is reasoning, content is "". Leading repeats of ThinkStart are stripped before
// the split, matching the detector's `while startswith` loop.
//
//	r, c := p.Parse("<think>weigh it</think>answer") // r="weigh it", c="answer"
func (p ReasoningParser) Parse(text string) (reasoning string, content string) {
	inReasoning := p.ForceReasoning || core.Contains(text, p.ThinkStart)
	if !inReasoning {
		return "", text
	}

	// Strip any leading ThinkStart openers (the block may echo it more than once).
	processed := text
	for core.HasPrefix(processed, p.ThinkStart) {
		processed = processed[len(p.ThinkStart):]
	}

	end := core.Index(processed, p.ThinkEnd)
	if end == -1 {
		// Reasoning was truncated before the end token — it's all reasoning.
		return processed, ""
	}
	reasoning = processed[:end]
	content = processed[end+len(p.ThinkEnd):]
	return reasoning, content
}
