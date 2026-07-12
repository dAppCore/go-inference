// SPDX-Licence-Identifier: EUPL-1.2

// Hand-rolled JSON-decoding adapters for the openai variant-shape
// unmarshallers. The walker primitives now live in jsonenc/ so that
// anthropic + ollama field-dispatch UnmarshalJSON paths can share
// the same byte-pump (lifted from this file in W11-B). The shapes
// this file owns — StopList / EmbeddingInput — both reduce to
// `ParseJSONStringList`, so the helpers here are thin variant-shape
// dispatchers.
//
// Per-call performance unchanged from the W10-M baseline — the
// underlying byte walker is identical.

package openai

import (
	core "dappco.re/go"
	"dappco.re/go/inference/jsonenc"
)

// parseJSONStringList walks data as either a JSON string (e.g.
// `"END"`) or an array of JSON strings (e.g. `["END","</s>"]`) and
// returns a []string with the inner values unescaped.
//
// Forwards to jsonenc.ParseJSONStringList — kept under the package-
// local name so existing call sites (StopList / EmbeddingInput) need
// no churn.
func parseJSONStringList(data []byte) ([]string, error) {
	return jsonenc.ParseJSONStringList(data)
}

// UnmarshalJSON decodes tool_choice's two wire shapes: a bare string
// ("auto"/"none"/"required") sets Mode directly; an object naming one
// function tool — {"type":"function","function":{"name":"X"}} — sets Mode
// from "type" (defaulting to "function" when type is itself absent, since an
// object with a function name always means a forced function tool) and Name
// from "function.name". null / empty leaves the zero value (auto).
//
// This is a cold-path field (tool_choice only appears on agentic requests, off
// the hot chat path — same rationale as the "tools" field in unmarshal.go), so
// it decodes via the reflect path (core.JSONUnmarshal) rather than a
// hand-rolled walk.
func (c *ToolChoice) UnmarshalJSON(data []byte) error {
	i := jsonenc.SkipJSONWhitespace(data, 0)
	if i >= len(data) || jsonenc.IsJSONNull(data, i) {
		return nil
	}
	if data[i] == '"' {
		var mode string
		if res := core.JSONUnmarshal(data, &mode); !res.OK {
			return res.Err()
		}
		c.Mode = mode
		return nil
	}
	var obj struct {
		Type     string `json:"type"`
		Function struct {
			Name string `json:"name"`
		} `json:"function"`
	}
	if res := core.JSONUnmarshal(data, &obj); !res.OK {
		return res.Err()
	}
	c.Mode = core.Coalesce(obj.Type, "function")
	c.Name = obj.Function.Name
	return nil
}
