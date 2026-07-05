// SPDX-Licence-Identifier: EUPL-1.2

package tools

import core "dappco.re/go"

// ToolCall is one tool invocation the model emitted (§6.4): an ID the result is
// correlated back by, the Name of the tool to run, and its Arguments as a raw
// JSON string (the executor decodes them against the tool's schema). Arguments
// stays a string deliberately — the orchestration layer never needs to inspect
// it, only hand it to the executor.
type ToolCall struct {
	ID        string `json:"id"`
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

// ParseToolCalls extracts the tool calls from a model's structured output. It
// accepts either a JSON array of call objects or a single call object (the
// common one-call shape), decoding via core.JSONUnmarshalString.
//
// Empty or whitespace-only input means the model called no tools — that returns
// an empty slice and no error, so the runner's len==0 loop doesn't have to treat
// "no calls" as a failure. Malformed JSON, or a call missing its tool name, IS
// an error: the model returned something undispatchable.
//
//	calls, err := tools.ParseToolCalls(modelOutput)
//	if err != nil { return err }          // the model emitted junk
//	if len(calls) == 0 { /* no tools — answer is final */ }
func ParseToolCalls(raw string) ([]ToolCall, error) {
	trimmed := core.Trim(raw)
	if trimmed == "" {
		return []ToolCall{}, nil
	}

	// A single object is the one-call shape; wrap it so one decode path handles
	// both. Anything else is decoded as the array it claims to be.
	if core.HasPrefix(trimmed, "{") {
		trimmed = "[" + trimmed + "]"
	}

	var calls []ToolCall
	if r := core.JSONUnmarshalString(trimmed, &calls); !r.OK {
		return nil, core.E("tools", "parse tool calls", resultErr(r))
	}

	// A call with no name can't be routed to any executor — reject the batch
	// rather than dispatch a nameless call that's guaranteed to "unknown tool".
	for _, c := range calls {
		if core.Trim(c.Name) == "" {
			return nil, core.E("tools", "tool call is missing its tool name", nil)
		}
	}

	if calls == nil {
		calls = []ToolCall{}
	}
	return calls, nil
}

// resultErr pulls the underlying error out of a failed core.Result so it can be
// chained as the cause of a core.E. core's JSON decoders always carry the
// json.Unmarshal error in Result.Value on failure (core/json.go returns
// Result{err, false}), so a failed parse always has an error to chain. A
// not-OK Result that somehow carried no error would have an empty message
// anyway, so falling back to a fresh core.E built from r.Error() (also empty)
// is unreachable through this package's one call site — hence resultErr keeps
// only the live extraction and lets a malformed Result chain a nil cause, which
// core.E tolerates.
func resultErr(r core.Result) error {
	err, _ := r.Value.(error)
	return err
}
