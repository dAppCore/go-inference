// SPDX-Licence-Identifier: EUPL-1.2

// Package tools is the pure-Go tool-calling orchestration (RFC.md §6.4).
// A chat request declares function tools and a tool_choice; the model answers
// with tool calls; the runner dispatches each call to a registered executor and
// feeds the results back. None of that needs a model loaded — it is plain Go
// glue — so it lives here, separate from the heavy inference packages.
//
// tools.go holds the declarations: Tool (a function or server tool) and
// ToolChoice (auto / none / required / named) with Resolve, which decides which
// tools a turn offers or forces. parse.go turns a model's structured output into
// ToolCall values. dispatch.go runs those calls through a Registry of Executors,
// sequentially or in parallel, collecting ToolResults in input order.
//
//	offered, err := tools.Resolve(tools.ChoiceAuto(), declared)
//	calls, err := tools.ParseToolCalls(modelOutput)
//	results := tools.Dispatch(ctx, calls, registry, true)
package tools

import core "dappco.re/go"

// Tool declares one tool the model may call. A function tool sets Name,
// Description, and Parameters (a JSON-schema document, given either as a raw
// string or a map[string]any — both round-trip through core.JSON*). A server
// tool additionally sets ServerKind to a marker like "web_search", "web_fetch",
// "code_interpreter", or "mcp", so tools that run inside the pipeline (§6.4) are
// representable in the same list as caller-resolved function tools.
//
//	fn := tools.Tool{Name: "get_weather", Description: "current weather",
//	    Parameters: `{"type":"object","properties":{"city":{"type":"string"}}}`}
//	srv := tools.Tool{Name: "web_search", ServerKind: tools.ServerWebSearch}
type Tool struct {
	Name        string     // the tool's stable name — what the model calls
	Description string     // what the tool does, for the model's selection
	Parameters  any        // JSON-schema for the arguments: string or map[string]any
	ServerKind  ServerTool // non-empty → a server tool that runs in-pipeline
}

// IsServer reports whether the tool runs inside the pipeline (a server tool)
// rather than round-tripping its call back to the caller.
//
//	if t.IsServer() { /* dispatched to a registered in-pipeline executor */ }
func (t Tool) IsServer() bool { return t.ServerKind != "" }

// ServerTool is the kind marker for a server tool — a tool the pipeline runs
// itself (§6.4) instead of handing the call back to the caller. The named
// constants below are the kinds the spec lists; the type is an open string so a
// new server tool needs no change here.
type ServerTool string

// The server-tool kinds from RFC.md §6.4. the own MCP server (§4.6) is one
// of these (ServerMCP), so its tools are callable through the same request.
const (
	ServerWebSearch       ServerTool = "web_search"
	ServerWebFetch        ServerTool = "web_fetch"
	ServerFileSearch      ServerTool = "file_search"
	ServerCodeInterpreter ServerTool = "code_interpreter"
	ServerShell           ServerTool = "shell"
	ServerTextEditor      ServerTool = "text_editor"
	ServerApplyPatch      ServerTool = "apply_patch"
	ServerComputerUse     ServerTool = "computer_use"
	ServerBrowserUse      ServerTool = "browser_use"
	ServerImageGen        ServerTool = "image_generation"
	ServerDatetime        ServerTool = "datetime"
	ServerSearchModels    ServerTool = "search_models"
	ServerMemory          ServerTool = "memory"
	ServerToolSearch      ServerTool = "tool_search"
	ServerMCP             ServerTool = "mcp"
)

// ChoiceMode is how the model is told to use the offered tools (§6.4).
type ChoiceMode string

const (
	ChoiceModeAuto     ChoiceMode = "auto"     // model may call any offered tool, or none
	ChoiceModeNone     ChoiceMode = "none"     // model may call no tools this turn
	ChoiceModeRequired ChoiceMode = "required" // model must call at least one offered tool
	ChoiceModeTool     ChoiceMode = "tool"     // model must call the named tool
)

// ToolChoice is the tool_choice field (§6.4): auto, none, required, or a single
// named tool. The zero value is auto, so a request that omits tool_choice still
// behaves sanely. Build one with the helper constructors rather than by hand.
//
//	tools.ChoiceAuto()          // let the model decide
//	tools.ChoiceRequired()      // force a call, model picks which
//	tools.ChoiceTool("fetch")   // force this exact tool
type ToolChoice struct {
	Mode ChoiceMode // auto (zero value) / none / required / tool
	Name string     // the forced tool, when Mode is ChoiceModeTool
}

// ChoiceAuto lets the model call any offered tool or none — the default.
func ChoiceAuto() ToolChoice { return ToolChoice{Mode: ChoiceModeAuto} }

// ChoiceNone offers no tools for this turn (the model answers in prose).
func ChoiceNone() ToolChoice { return ToolChoice{Mode: ChoiceModeNone} }

// ChoiceRequired forces the model to call at least one of the offered tools.
func ChoiceRequired() ToolChoice { return ToolChoice{Mode: ChoiceModeRequired} }

// ChoiceTool forces the model to call exactly the named tool.
//
//	tools.ChoiceTool("web_search")
func ChoiceTool(name string) ToolChoice { return ToolChoice{Mode: ChoiceModeTool, Name: name} }

// Resolve turns a choice plus the declared tools into the set actually offered
// to the model for this turn:
//
//   - auto / required → every declared tool (the model picks; required means it
//     must pick one — that constraint travels in the choice value, not the set);
//   - none → no tools (an empty, non-nil slice);
//   - tool(name) → only that tool, and only if it was declared.
//
// A named choice for an undeclared tool, or required with no tools, is a caller
// error — the model would be told to call something that can't run — so Resolve
// returns a core.E rather than silently degrading.
//
//	offered, err := tools.Resolve(choice, declared)
//	if err != nil { return err } // contradictory tool_choice
func Resolve(choice ToolChoice, declared []Tool) ([]Tool, error) {
	switch choice.Mode {
	case ChoiceModeNone:
		return []Tool{}, nil

	case ChoiceModeTool:
		for _, t := range declared {
			if t.Name == choice.Name {
				return []Tool{t}, nil
			}
		}
		return nil, core.E("tools", "tool_choice names a tool that was not declared: "+choice.Name, nil)

	case ChoiceModeRequired:
		if len(declared) == 0 {
			return nil, core.E("tools", "tool_choice is required but no tools were declared", nil)
		}
		return cloneTools(declared), nil

	default: // ChoiceModeAuto and the zero value
		return cloneTools(declared), nil
	}
}

// cloneTools returns a fresh, non-nil slice over the declared tools so a caller
// can't mutate the request's tool list through the resolved set.
func cloneTools(declared []Tool) []Tool {
	out := make([]Tool, len(declared))
	copy(out, declared)
	return out
}
