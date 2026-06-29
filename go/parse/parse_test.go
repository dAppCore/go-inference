// SPDX-Licence-Identifier: EUPL-1.2

package parse

import (
	"testing"

	core "dappco.re/go"
	tools "dappco.re/go/inference/tools"
)

// decode turns a ToolCall.Arguments JSON string back into a map so assertions
// don't depend on Go's map-key ordering when it marshals.
//
//	args := decode(t, calls[0].Arguments)
//	if args["city"] != "Paris" { t.Fatal(...) }
func decode(t *testing.T, raw string) map[string]any {
	t.Helper()
	var m map[string]any
	if r := core.JSONUnmarshalString(raw, &m); !r.OK {
		t.Fatalf("arguments are not valid JSON: %q", raw)
	}
	return m
}

// --- Gemma 4 tool-call detector ---------------------------------------------

func TestParse_Gemma4Tools_Good(t *testing.T) {
	// Single call, leading normal text, a string arg wrapped in <|"|> and a
	// bare number arg. The text before the first tool-call token is normalText.
	in := `Let me check.<|tool_call>call:get_weather{city: <|"|>Paris<|"|>, days: 3}<tool_call|>`

	calls, normal, err := ParseGemma4ToolCalls(in)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if normal != "Let me check." {
		t.Fatalf("normalText = %q, want %q", normal, "Let me check.")
	}
	if len(calls) != 1 {
		t.Fatalf("got %d calls, want 1", len(calls))
	}
	if calls[0].Name != "get_weather" {
		t.Fatalf("name = %q, want get_weather", calls[0].Name)
	}
	args := decode(t, calls[0].Arguments)
	if args["city"] != "Paris" {
		t.Fatalf("city = %v, want Paris", args["city"])
	}
	// JSON numbers decode to float64.
	if args["days"] != float64(3) {
		t.Fatalf("days = %v (%T), want 3", args["days"], args["days"])
	}
}

func TestParse_Gemma4Tools_Good_MultipleCalls(t *testing.T) {
	// Two calls back to back, no normal text. Every span is extracted in order.
	in := `<|tool_call>call:a{x: 1}<tool_call|><|tool_call>call:b{y: <|"|>hi<|"|>}<tool_call|>`

	calls, normal, err := ParseGemma4ToolCalls(in)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if normal != "" {
		t.Fatalf("normalText = %q, want empty", normal)
	}
	if len(calls) != 2 {
		t.Fatalf("got %d calls, want 2", len(calls))
	}
	if calls[0].Name != "a" || calls[1].Name != "b" {
		t.Fatalf("names = %q,%q want a,b", calls[0].Name, calls[1].Name)
	}
	if decode(t, calls[0].Arguments)["x"] != float64(1) {
		t.Fatalf("call a x wrong: %s", calls[0].Arguments)
	}
	if decode(t, calls[1].Arguments)["y"] != "hi" {
		t.Fatalf("call b y wrong: %s", calls[1].Arguments)
	}
}

func TestParse_Gemma4Tools_Good_AllArgKinds(t *testing.T) {
	// Exercise every value kind: string, int, float, bool true/false, array of
	// strings, array of mixed/nested object, nested object, and a bare string.
	in := `<|tool_call>call:complex{` +
		`name: <|"|>Ada<|"|>, ` +
		`count: 42, ` +
		`ratio: 1.5, ` +
		`active: true, ` +
		`hidden: false, ` +
		`tags: [<|"|>a<|"|>, <|"|>b<|"|>], ` +
		`nums: [1, 2, 3], ` +
		`meta: {role: <|"|>admin<|"|>, level: 9}, ` +
		`people: [{n: <|"|>x<|"|>}, {n: <|"|>y<|"|>}], ` +
		`grid: [[1, 2], [3, 4]], ` +
		`raw: bareword` +
		`}<tool_call|>`

	calls, _, err := ParseGemma4ToolCalls(in)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(calls) != 1 {
		t.Fatalf("got %d calls, want 1", len(calls))
	}
	a := decode(t, calls[0].Arguments)

	if a["name"] != "Ada" {
		t.Errorf("name = %v", a["name"])
	}
	if a["count"] != float64(42) {
		t.Errorf("count = %v", a["count"])
	}
	if a["ratio"] != float64(1.5) {
		t.Errorf("ratio = %v", a["ratio"])
	}
	if a["active"] != true {
		t.Errorf("active = %v", a["active"])
	}
	if a["hidden"] != false {
		t.Errorf("hidden = %v", a["hidden"])
	}
	if a["raw"] != "bareword" {
		t.Errorf("raw = %v", a["raw"])
	}

	tags, ok := a["tags"].([]any)
	if !ok || len(tags) != 2 || tags[0] != "a" || tags[1] != "b" {
		t.Errorf("tags = %v", a["tags"])
	}
	nums, ok := a["nums"].([]any)
	if !ok || len(nums) != 3 || nums[2] != float64(3) {
		t.Errorf("nums = %v", a["nums"])
	}
	meta, ok := a["meta"].(map[string]any)
	if !ok || meta["role"] != "admin" || meta["level"] != float64(9) {
		t.Errorf("meta = %v", a["meta"])
	}
	people, ok := a["people"].([]any)
	if !ok || len(people) != 2 {
		t.Fatalf("people = %v", a["people"])
	}
	p0, ok := people[0].(map[string]any)
	if !ok || p0["n"] != "x" {
		t.Errorf("people[0] = %v", people[0])
	}
	grid, ok := a["grid"].([]any)
	if !ok || len(grid) != 2 {
		t.Fatalf("grid = %v", a["grid"])
	}
	row0, ok := grid[0].([]any)
	if !ok || len(row0) != 2 || row0[1] != float64(2) {
		t.Errorf("grid[0] = %v", grid[0])
	}
}

func TestParse_Gemma4Tools_Good_EmptyArgs(t *testing.T) {
	// A call with no arguments yields an empty-object JSON string "{}".
	in := `<|tool_call>call:ping{}<tool_call|>`

	calls, _, err := ParseGemma4ToolCalls(in)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(calls) != 1 || calls[0].Name != "ping" {
		t.Fatalf("calls = %+v", calls)
	}
	if calls[0].Arguments != "{}" {
		t.Fatalf("arguments = %q, want {}", calls[0].Arguments)
	}
}

func TestParse_Gemma4Tools_Bad_NoToolCall(t *testing.T) {
	// No tool-call token at all: zero calls, the whole text is normalText.
	in := "Just a plain answer with no tools."

	calls, normal, err := ParseGemma4ToolCalls(in)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(calls) != 0 {
		t.Fatalf("got %d calls, want 0", len(calls))
	}
	if normal != in {
		t.Fatalf("normalText = %q, want the whole input", normal)
	}
}

func TestParse_Gemma4Tools_Bad_StartButNoEnd(t *testing.T) {
	// A start token with no matching end token: SGLang bails and returns the
	// whole text as normalText with no calls.
	in := `prefix<|tool_call>call:x{a: 1}`

	calls, normal, err := ParseGemma4ToolCalls(in)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(calls) != 0 {
		t.Fatalf("got %d calls, want 0", len(calls))
	}
	if normal != in {
		t.Fatalf("normalText = %q, want whole input", normal)
	}
}

func TestParse_Gemma4Tools_Bad_SpanWithoutCallPrefix(t *testing.T) {
	// A well-formed span whose inner text does not start with "call:" produces no
	// matches, so SGLang's detect_and_parse returns the WHOLE text as normalText
	// (the `if not matches: return normal_text=text` branch), not the prefix.
	in := `<|tool_call>noprefix{a: 1}<tool_call|>`

	calls, normal, err := ParseGemma4ToolCalls(in)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(calls) != 0 {
		t.Fatalf("got %d calls, want 0 (no call: prefix)", len(calls))
	}
	if normal != in {
		t.Fatalf("normalText = %q, want the whole input", normal)
	}
}

func TestParse_Gemma4Tools_Bad_CallPrefixNoBrace(t *testing.T) {
	// "call:" present but no opening brace inside the span: no matches, so the
	// whole text is normalText (same no-matches branch as above).
	in := `<|tool_call>call:lonely<tool_call|>tail`

	calls, normal, err := ParseGemma4ToolCalls(in)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(calls) != 0 {
		t.Fatalf("got %d calls, want 0 (no brace)", len(calls))
	}
	if normal != in {
		t.Fatalf("normalText = %q, want the whole input", normal)
	}
}

func TestParse_Gemma4Tools_Ugly_UnterminatedString(t *testing.T) {
	// Inside a closed span, a string value that never closes its <|"|>: the
	// parser takes the rest of the args as that value (matching _parse_gemma4_args).
	in := `<|tool_call>call:f{note: <|"|>never closes}<tool_call|>`

	calls, _, err := ParseGemma4ToolCalls(in)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(calls) != 1 {
		t.Fatalf("got %d calls, want 1", len(calls))
	}
	a := decode(t, calls[0].Arguments)
	if a["note"] != "never closes}" {
		t.Fatalf("note = %q, want the rest of the args", a["note"])
	}
}

func TestParse_Gemma4Tools_Ugly_KeyWithNoValue(t *testing.T) {
	// A trailing key with a ':' but nothing after it: value is "" (the Python
	// "i >= n after ':'" branch). Brace-balance still closes the span.
	in := `<|tool_call>call:f{a: 1, b:}<tool_call|>`

	calls, _, err := ParseGemma4ToolCalls(in)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(calls) != 1 {
		t.Fatalf("got %d calls, want 1", len(calls))
	}
	a := decode(t, calls[0].Arguments)
	if a["a"] != float64(1) {
		t.Errorf("a = %v", a["a"])
	}
	if a["b"] != "" {
		t.Errorf("b = %v, want empty string", a["b"])
	}
}

func TestParse_Gemma4Tools_Ugly_KeyOnlyNoColon(t *testing.T) {
	// Args content that is only a key with no ':' at all — the key-scan runs off
	// the end and the loop breaks with no entry recorded. Empty object.
	in := `<|tool_call>call:f{justkey}<tool_call|>`

	calls, _, err := ParseGemma4ToolCalls(in)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(calls) != 1 {
		t.Fatalf("got %d calls, want 1", len(calls))
	}
	if calls[0].Arguments != "{}" {
		t.Fatalf("arguments = %q, want {}", calls[0].Arguments)
	}
}

func TestParse_Gemma4Tools_Ugly_StringWithBraces(t *testing.T) {
	// Braces *inside* a delimited string must not affect brace balance — the
	// span closes at the real outer brace, not one buried in the string.
	in := `<|tool_call>call:f{q: <|"|>a {nested} brace<|"|>}<tool_call|>`

	calls, _, err := ParseGemma4ToolCalls(in)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(calls) != 1 {
		t.Fatalf("got %d calls, want 1", len(calls))
	}
	a := decode(t, calls[0].Arguments)
	if a["q"] != "a {nested} brace" {
		t.Fatalf("q = %q, want the string with literal braces", a["q"])
	}
}

func TestParse_Gemma4Tools_Ugly_UnterminatedStringInsideObjectBalance(t *testing.T) {
	// A nested object whose string delimiter never closes: the brace-matcher's
	// "delimiter run to end" branch fires and the span is treated as not
	// closing — SGLang's _find_matching_brace returns -1, so the args become the
	// whole remainder. Still one call, value parsing tolerant.
	in := `<|tool_call>call:f{meta: {k: <|"|>open}<tool_call|>`

	calls, normal, err := ParseGemma4ToolCalls(in)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// The end token IS present, so a span is extracted; brace match fails and
	// args_content (whole remainder) is parsed. We only assert it does not panic
	// and yields a single call with a meta key.
	if len(calls) != 1 {
		t.Fatalf("got %d calls, want 1", len(calls))
	}
	if _, ok := decode(t, calls[0].Arguments)["meta"]; !ok {
		t.Fatalf("expected a meta key, got %s", calls[0].Arguments)
	}
	if normal != "" {
		t.Fatalf("normalText = %q, want empty", normal)
	}
}

func TestParse_Gemma4Tools_Ugly_ArrayUnterminatedString(t *testing.T) {
	// An array element string that never closes — _parse_gemma4_array takes the
	// rest of the array content as the element and stops.
	in := `<|tool_call>call:f{xs: [<|"|>one<|"|>, <|"|>two]}<tool_call|>`

	calls, _, err := ParseGemma4ToolCalls(in)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	a := decode(t, calls[0].Arguments)
	xs, ok := a["xs"].([]any)
	if !ok || len(xs) != 2 {
		t.Fatalf("xs = %v, want 2 elements", a["xs"])
	}
	if xs[0] != "one" || xs[1] != "two]" {
		t.Fatalf("xs = %v, want [one, two]] (unterminated tail)", xs)
	}
}

func TestParse_Gemma4Tools_Ugly_NormalTextOnlyBeforeStart(t *testing.T) {
	// content_end > 0 path: text before the first start token is the normalText,
	// even with multiple calls following.
	in := `Working on it. <|tool_call>call:go{}<tool_call|>`

	calls, normal, err := ParseGemma4ToolCalls(in)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if normal != "Working on it. " {
		t.Fatalf("normalText = %q", normal)
	}
	if len(calls) != 1 {
		t.Fatalf("got %d calls", len(calls))
	}
}

func TestParse_Gemma4Tools_Ugly_DispatchShape(t *testing.T) {
	// The returned slice must be the sibling tools.ToolCall type so it plugs
	// straight into tools.Dispatch — assert the concrete field set.
	in := `<|tool_call>call:noop{}<tool_call|>`
	calls, _, err := ParseGemma4ToolCalls(in)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	var _ []tools.ToolCall = calls
	if calls[0].ID != "" {
		t.Fatalf("ID = %q, want empty (caller assigns)", calls[0].ID)
	}
}

// --- white-box edge branches (same package) ---------------------------------

func TestParse_indexFrom_Bounds(t *testing.T) {
	// Offset clamped below 0 and a past-the-end offset returns -1 — the defensive
	// guards the internal callers never trip, exercised directly.
	if got := indexFrom("aXbX", "X", -5); got != 1 {
		t.Fatalf("indexFrom negative offset = %d, want 1", got)
	}
	if got := indexFrom("abc", "a", 99); got != -1 {
		t.Fatalf("indexFrom past-end offset = %d, want -1", got)
	}
	if got := indexFrom("abc", "z", 0); got != -1 {
		t.Fatalf("indexFrom missing sub = %d, want -1", got)
	}
	if got := indexFrom("aXbX", "X", 2); got != 3 {
		t.Fatalf("indexFrom = %d, want 3", got)
	}
}

func TestParse_findMatchingBrace_NeverCloses(t *testing.T) {
	// Opens more braces than it closes — depth never returns to zero, so -1.
	if got := findMatchingBrace("a: {b"); got != -1 {
		t.Fatalf("findMatchingBrace unbalanced = %d, want -1", got)
	}
}

func TestParse_parseGemma4Args_OnlySeparators(t *testing.T) {
	// Content that is non-empty (so the Trim guard passes) but only separators:
	// the entry loop skips them all and breaks with an empty map.
	got := parseGemma4Args(",")
	if len(got) != 0 {
		t.Fatalf("parseGemma4Args(\",\") = %v, want empty", got)
	}
}

func TestParse_parseGemma4Args_KeyTrailingSpaceAfterColon(t *testing.T) {
	// "key: " — colon then only trailing whitespace, hitting the post-skip
	// end-of-input branch that records an empty-string value.
	got := parseGemma4Args("b: ")
	if v, ok := got["b"]; !ok || v != "" {
		t.Fatalf("parseGemma4Args = %v, want b->\"\"", got)
	}
}

func TestParse_parseGemma4Args_BareValueEmpty(t *testing.T) {
	// A value position that starts on a terminator (',') yields an empty bare
	// value — parseGemma4Value("") returns "" (its empty-after-trim branch).
	got := parseGemma4Args("k: ,x: 1")
	if v, ok := got["k"]; !ok || v != "" {
		t.Fatalf("k = %v, want empty bare value", got["k"])
	}
	if got["x"] != int64(1) && got["x"] != float64(1) {
		// parseGemma4Args stores numbers as float64 (via parseNumber); allow
		// either in case the int path is ever swapped in.
		switch got["x"].(type) {
		case float64, int64:
		default:
			t.Fatalf("x = %v (%T)", got["x"], got["x"])
		}
	}
}

func TestParse_parseGemma4Array_OnlySeparators(t *testing.T) {
	// Array body of only separators -> empty slice (the i>=n break after skip).
	got := parseGemma4Array(" , ")
	if len(got) != 0 {
		t.Fatalf("parseGemma4Array = %v, want empty", got)
	}
}

func TestParse_parseGemma4Array_TripleNested(t *testing.T) {
	// Three-deep nesting forces the inner '[' depth++ branch inside the nested
	// array scanner.
	got := parseGemma4Array("[[1]]")
	if len(got) != 1 {
		t.Fatalf("outer len = %d, want 1", len(got))
	}
	mid, ok := got[0].([]any)
	if !ok || len(mid) != 1 {
		t.Fatalf("mid = %v", got[0])
	}
	inner, ok := mid[0].([]any)
	if !ok || len(inner) != 1 || inner[0] != float64(1) {
		t.Fatalf("inner = %v", mid[0])
	}
}

func TestParse_parseGemma4Value_Kinds(t *testing.T) {
	// Direct coverage of the value classifier, including the empty-after-trim
	// branch and a non-numeric bare string.
	if parseGemma4Value("  ") != "" {
		t.Fatalf("blank value should trim to empty string")
	}
	if parseGemma4Value("true") != true {
		t.Fatalf("true mis-parsed")
	}
	if parseGemma4Value("false") != false {
		t.Fatalf("false mis-parsed")
	}
	if parseGemma4Value("12") != float64(12) {
		t.Fatalf("int mis-parsed: %v", parseGemma4Value("12"))
	}
	if parseGemma4Value("-2.5") != float64(-2.5) {
		t.Fatalf("float mis-parsed: %v", parseGemma4Value("-2.5"))
	}
	if parseGemma4Value("hello") != "hello" {
		t.Fatalf("bare word mis-parsed: %v", parseGemma4Value("hello"))
	}
	// JSON null is not a number — falls through to a bare string.
	if parseGemma4Value("null") != "null" {
		t.Fatalf("null should stay a bare string: %v", parseGemma4Value("null"))
	}
}

func TestParse_Gemma4Tools_Ugly_ArgsBraceNeverCloses(t *testing.T) {
	// Full-span path where the args body opens a brace that never closes inside
	// the span: findMatchingBrace returns -1, so args_str is the whole remainder
	// (SGLang's `args_content if match_idx == -1` branch). One call, no panic.
	in := `<|tool_call>call:f{a: {b<tool_call|>`

	calls, _, err := ParseGemma4ToolCalls(in)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(calls) != 1 || calls[0].Name != "f" {
		t.Fatalf("calls = %+v", calls)
	}
}
