// SPDX-Licence-Identifier: EUPL-1.2

package parser

import (
	"testing"
)

func TestTools_ParseTools_Good(t *testing.T) {
	p := ForHint(Hint{Architecture: "hermes3"})

	tagged, err := p.ParseTools(nil, `before <tool_call>{"name":"search","arguments":{"q":"core"}}</tool_call> after`)
	if err != nil {
		t.Fatalf("ParseTools(tagged) error = %v", err)
	}
	if tagged.VisibleText != "before  after" {
		t.Fatalf("tagged visible = %q", tagged.VisibleText)
	}
	if len(tagged.Calls) != 1 || tagged.Calls[0].Name != "search" || tagged.Calls[0].ArgumentsJSON != `{"q":"core"}` {
		t.Fatalf("tagged calls = %+v", tagged.Calls)
	}

	jsonFallback, err := p.ParseTools(nil, `{"tool_calls":[{"id":"call_1","type":"function","function":{"name":"lookup","arguments":{"id":7}}}]}`)
	if err != nil {
		t.Fatalf("ParseTools(json) error = %v", err)
	}
	if jsonFallback.VisibleText != "" {
		t.Fatalf("json visible = %q, want empty", jsonFallback.VisibleText)
	}
	if len(jsonFallback.Calls) != 1 || jsonFallback.Calls[0].ID != "call_1" || jsonFallback.Calls[0].Name != "lookup" || jsonFallback.Calls[0].ArgumentsJSON != `{"id":7}` {
		t.Fatalf("json calls = %+v", jsonFallback.Calls)
	}
}

func TestTools_BadAndUglyPayloads(t *testing.T) {
	p := ForHint(Hint{Architecture: "qwen3"})
	if _, err := p.ParseTools(nil, `<tool_call>{bad}</tool_call>`); err == nil {
		t.Fatal("ParseTools(malformed tagged JSON) error = nil")
	}
	unclosed, err := p.ParseTools(nil, `before <tool_call>{"name":"search"}`)
	if err != nil {
		t.Fatalf("ParseTools(unclosed tag) error = %v", err)
	}
	if unclosed.VisibleText != `before <tool_call>{"name":"search"}` || len(unclosed.Calls) != 0 {
		t.Fatalf("unclosed tool parse = %+v, want visible passthrough", unclosed)
	}
	if calls, err := parseToolPayload(`[{"name":"search","arguments_json":"{\"q\":\"core\"}"},{"name":""}]`); err != nil || len(calls) != 1 || calls[0].ArgumentsJSON != `{"q":"core"}` {
		t.Fatalf("parseToolPayload(array) = %+v/%v, want one call with existing args JSON", calls, err)
	}
	if calls, err := parseToolPayload(`{"calls":[{"name":"lookup","arguments":"{\"id\":7}"}]}`); err != nil || len(calls) != 1 || calls[0].ArgumentsJSON != `{"id":7}` {
		t.Fatalf("parseToolPayload(calls) = %+v/%v, want string arguments normalised", calls, err)
	}
	if calls, err := parseToolPayload(`{"type":"function"}`); err != nil || len(calls) != 0 {
		t.Fatalf("parseToolPayload(no name) = %+v/%v, want no call", calls, err)
	}
	if _, err := parseToolPayload(`{bad}`); err == nil {
		t.Fatal("parseToolPayload(bad JSON) error = nil")
	}
}
