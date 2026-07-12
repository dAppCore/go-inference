// SPDX-Licence-Identifier: EUPL-1.2

// tool_choice resolution: which of a request's declared tools are actually
// offered to the model this turn. Decoding lives in jsondec.go (ToolChoice's
// variant string-or-object wire shape); this file maps the decoded choice onto
// the shared engine-neutral agent/tools.ToolChoice and filters the declared
// Tool slice accordingly, so the resolution logic itself is not duplicated
// per provider.
package openai

import (
	core "dappco.re/go"
	"dappco.re/go/inference/agent/tools"
)

// ToolChoice controls whether — and which — of a request's declared tools the
// model is offered this turn (RFC §6.4). The wire form is either a bare string
// ("auto" the default, "none", or "required") or an object naming one function
// tool — {"type":"function","function":{"name":"X"}} — both decoded into this
// one Mode/Name pair by UnmarshalJSON (jsondec.go).
type ToolChoice struct {
	Mode string // "auto" | "none" | "required" | "function" (Name set)
	Name string // the forced tool name, set only when Mode == "function"
}

// resolve maps the wire ToolChoice onto the engine-neutral
// agent/tools.ToolChoice. A nil choice (tool_choice omitted from the request)
// is auto — every declared tool is offered, matching the pre-tool_choice
// behaviour byte-for-byte.
func (c *ToolChoice) resolve() tools.ToolChoice {
	if c == nil {
		return tools.ChoiceAuto()
	}
	switch core.Lower(core.Trim(c.Mode)) {
	case "none":
		return tools.ChoiceNone()
	case "required":
		return tools.ChoiceRequired()
	case "function", "tool":
		return tools.ChoiceTool(c.Name)
	default:
		return tools.ChoiceAuto()
	}
}

// resolveOfferedTools maps a request's declared tools + tool_choice into the
// tools actually rendered into the prompt this turn, via the shared
// agent/tools.Resolve: "none" offers nothing (the model is never told tools
// exist this turn, so it cannot call one), a named choice narrows to that one
// declared tool, and "auto"/"required"/absent offer every declared tool
// unchanged. ("required"'s "the model must call something" obligation travels
// as documented on agent/tools.Resolve — it is not a prompt-level or
// generation-time constraint this package can enforce without engine-level
// grammar support, so it offers the same full set as auto rather than
// half-fabricating enforcement.)
//
// A tool_choice naming an undeclared tool, or requiring a tool when none were
// declared, is a caller error (agent/tools.Resolve returns it) — ServeHTTP
// turns that into a 400, matching every other request-shape validation error.
func resolveOfferedTools(declared []Tool, choice *ToolChoice) ([]Tool, error) {
	// No early return for len(declared)==0: tools.Resolve must still see a
	// required/named choice against zero declared tools so it can report the
	// caller error (ChoiceModeRequired / ChoiceModeTool both reject that
	// combination) — shortcutting here would silently swallow it.
	declaredTools := make([]tools.Tool, len(declared))
	for i, t := range declared {
		declaredTools[i] = tools.Tool{Name: t.Function.Name}
	}
	resolved, err := tools.Resolve(choice.resolve(), declaredTools)
	if err != nil {
		return nil, err
	}
	if len(resolved) == len(declared) {
		return declared, nil // common case: auto/required offers everything, unchanged
	}
	offered := make([]Tool, 0, len(resolved))
	for _, r := range resolved {
		for _, t := range declared {
			if t.Function.Name == r.Name {
				offered = append(offered, t)
				break
			}
		}
	}
	return offered, nil
}
