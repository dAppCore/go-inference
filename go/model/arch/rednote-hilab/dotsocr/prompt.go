// SPDX-Licence-Identifier: EUPL-1.2

package dotsocr

import core "dappco.re/go"

// buildPrompt constructs DOTS-OCR's list-content user-turn prompt string EXACTLY as
// transformers' apply_chat_template renders a {"role":"user","content":[{"type":"image"},
// {"type":"text","text":prompt}]} message — this checkpoint's own README-documented usage
// (Usage with transformers §, messages built with an image part + a text part). EMPIRICALLY
// confirmed against the real chat_template.json rendered through the real processor (see
// testdata/prompt_golden.json and prompt_test.go), not derived by reading the Jinja source alone:
// {%- elif m.role == 'user' %}{% if m.content is string %}{{- '<|user|>' + m.content +
// '<|endofuser|>' }}{% else %} {% for content in m.content %}... — the LIST-content branch
// (the `{% else %}` arm, taken whenever content is a list of parts rather than a bare string)
// wraps with NEITHER "<|user|>" nor "<|endofuser|>", unlike the plain-string branch, and emits
// one literal leading space before "<|img|>" (an unguarded space between "{% else %}" and the
// image loop in the template source). A first attempt at deriving this purely by reading the
// Jinja template got it wrong (assumed the <|user|>/<|endofuser|> wrapper always applies); only
// cross-checking against the processor's actual rendered output caught it — see this package's
// golden-capture notes. numImageTokens is the count apply_chat_template's caller
// (Qwen2_5_VLProcessor.__call__) expands the single "<|imgpad|>" placeholder into: grid_t·
// grid_h·grid_w / spatial_merge_size² for ONE image (this package only ever encodes one image per
// OCR call).
func buildPrompt(promptText string, numImageTokens int) string {
	sb := core.NewBuilder()
	sb.WriteString(" <|img|>")
	sb.WriteString(core.Repeat("<|imgpad|>", numImageTokens))
	sb.WriteString("<|endofimg|>")
	sb.WriteString(promptText)
	sb.WriteString("<|assistant|>")
	return sb.String()
}
