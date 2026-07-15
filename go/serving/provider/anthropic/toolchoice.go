// SPDX-Licence-Identifier: EUPL-1.2

// tool_choice resolution — the Anthropic-side mirror of the OpenAI provider's
// toolchoice.go, mapping onto the same shared agent/tools.ToolChoice /
// tools.Resolve seam so the "which tools does the model actually see this
// turn" decision has one implementation, not two.
package anthropic

import (
	core "dappco.re/go"
	"dappco.re/go/inference/agent/tools"
)

// ToolChoice controls whether — and which — of a request's declared tools the
// model is offered this turn (Anthropic's tool_choice — always an object, no
// bare-string wire form the way OpenAI's is): {"type":"auto"} (the default —
// the zero value decodes the same way when the field is omitted entirely),
// {"type":"any"} (the model must call one of the offered tools),
// {"type":"tool","name":"X"} (forces that one tool), or {"type":"none"} (no
// tools this turn).
type ToolChoice struct {
	Type string `json:"type"`
	Name string `json:"name,omitempty"`
}

// resolve maps the wire ToolChoice onto the engine-neutral
// agent/tools.ToolChoice. A nil choice (tool_choice omitted) is auto — every
// declared tool is offered, matching the pre-tool_choice behaviour byte-for-byte.
func (c *ToolChoice) resolve() tools.ToolChoice {
	if c == nil {
		return tools.ChoiceAuto()
	}
	switch core.Lower(core.Trim(c.Type)) {
	case "none":
		return tools.ChoiceNone()
	case "any":
		return tools.ChoiceRequired()
	case "tool":
		return tools.ChoiceTool(c.Name)
	default:
		return tools.ChoiceAuto()
	}
}

// ResolveOfferedTools maps a request's declared tools + tool_choice into the
// tools actually rendered into the prompt this turn, via the shared
// agent/tools.Resolve — the same rules as the OpenAI provider's
// resolveOfferedTools (toolchoice.go): "none" offers nothing, a named choice
// narrows to that one declared tool, and "auto"/"any"/absent offer every
// declared tool unchanged.
//
// Exported (unlike the OpenAI provider's private helper) so serving/compat's
// HTTP handler can validate tool_choice and gate on tool-calling capability
// using the exact same resolution InferenceMessages applies internally,
// without duplicating the mapping logic — InferenceMessages' own signature
// can't take an error return (see its doc comment), so the HTTP layer is
// where a contradictory tool_choice actually becomes a 4xx.
//
// A tool_choice naming an undeclared tool, or requiring one when none were
// declared, is a caller error (agent/tools.Resolve returns it).
func ResolveOfferedTools(declared []Tool, choice *ToolChoice) ([]Tool, error) {
	declaredTools := make([]tools.Tool, len(declared))
	for i, t := range declared {
		declaredTools[i] = tools.Tool{Name: t.Name}
	}
	resolved, err := tools.Resolve(choice.resolve(), declaredTools)
	if err != nil {
		return nil, err
	}
	if len(resolved) == len(declared) {
		return declared, nil // common case: auto/any offers everything, unchanged
	}
	offered := make([]Tool, 0, len(resolved))
	for _, r := range resolved {
		for _, t := range declared {
			if t.Name == r.Name {
				offered = append(offered, t)
				break
			}
		}
	}
	return offered, nil
}
