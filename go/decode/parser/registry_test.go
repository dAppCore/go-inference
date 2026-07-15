// SPDX-Licence-Identifier: EUPL-1.2

package parser

import (
	"testing"

	"dappco.re/go/inference"
)

func TestRegistry_DefaultLookup_Good_ModelFamilies(t *testing.T) {
	cases := map[string]string{
		"qwen3":       "qwen",
		"gemma4_text": "gemma",
		"minimax_m2":  "minimax",
		"deepseek_r1": "deepseek-r1",
		"gpt_oss":     "gpt-oss",
		"mistral":     "mistral",
		"kimi_k2":     "kimi",
		"glm4":        "glm",
		"hermes3":     "hermes",
		"granite":     "granite",
		"unknown":     "generic",
	}

	for arch, want := range cases {
		p := ForHint(Hint{Architecture: arch})
		if p == nil {
			t.Fatalf("ForHint(%q) returned nil", arch)
		}
		if p.ParserID() != want {
			t.Fatalf("ForHint(%q) = %q, want %q", arch, p.ParserID(), want)
		}
	}
}

func TestRegistry_RegisterCustomParser_Good(t *testing.T) {
	registry := NewRegistry()
	registry.Register(customOutputParser{}, "custom-family")

	p, ok := registry.Lookup("custom-family")
	if !ok {
		t.Fatal("Lookup(custom-family) = false")
	}
	got, err := p.ParseReasoning(nil, "answer")
	if err != nil {
		t.Fatalf("ParseReasoning() error = %v", err)
	}
	if p.ParserID() != "custom" || got.VisibleText != "custom:answer" {
		t.Fatalf("parser/result = %q %+v", p.ParserID(), got)
	}
}

func TestRegistry_FallbacksAndNilReceivers_Ugly(t *testing.T) {
	var nilRegistry *Registry
	if p, ok := nilRegistry.Lookup("qwen"); ok || p != nil {
		t.Fatalf("nil Lookup() = %+v/%v, want nil/false", p, ok)
	}
	p := nilRegistry.LookupHint(Hint{Architecture: "qwen3"})
	if p == nil || p.ParserID() != "qwen" {
		t.Fatalf("nil LookupHint() = %v, want default qwen parser", p)
	}
	registry := &Registry{}
	registry.Register(nil, "ignored")
	if p := registry.LookupHint(Hint{}); p == nil || p.ParserID() != "generic" {
		t.Fatalf("empty registry LookupHint() = %v, want generic fallback", p)
	}
	registry.Register(customOutputParser{}, "", "custom.alias")
	if p, ok := registry.Lookup("custom-alias"); !ok || p.ParserID() != "custom" {
		t.Fatalf("Lookup(custom-alias) = %v/%v, want custom parser", p, ok)
	}

	var nilParser *builtinOutputParser
	if nilParser.ParserID() != "generic" {
		t.Fatalf("nil builtin ParserID() = %q, want generic", nilParser.ParserID())
	}
	reasoning, err := nilParser.ParseReasoning(nil, "<analysis>plan</analysis>answer")
	if err != nil || reasoning.VisibleText != "answer" || len(reasoning.Reasoning) != 1 {
		t.Fatalf("nil builtin ParseReasoning() = %+v/%v, want generic parse", reasoning, err)
	}
}

// TestRegistry_NewRegistry_Good pins the constructed shape: a fresh Registry
// carries the generic parser both as a lookup entry and as its fallback.
func TestRegistry_NewRegistry_Good(t *testing.T) {
	registry := NewRegistry()
	p, ok := registry.Lookup("generic")
	if !ok || p == nil || p.ParserID() != "generic" {
		t.Fatalf("Lookup(generic) = %v/%v, want the generic parser", p, ok)
	}
}

// TestRegistry_NewRegistry_Bad pins the narrow surface: a fresh Registry has
// no builtin family parsers beyond generic — an unrelated key misses.
func TestRegistry_NewRegistry_Bad(t *testing.T) {
	registry := NewRegistry()
	if _, ok := registry.Lookup("qwen3"); ok {
		t.Fatal("fresh NewRegistry() must not carry the qwen family parser")
	}
}

// TestRegistry_NewRegistry_Ugly pins independence: two NewRegistry() calls
// return distinct instances — registering on one must not leak into the
// other (unlike Default(), which is a shared singleton).
func TestRegistry_NewRegistry_Ugly(t *testing.T) {
	a := NewRegistry()
	b := NewRegistry()
	a.Register(customOutputParser{}, "only-on-a")
	if _, ok := b.Lookup("only-on-a"); ok {
		t.Fatal("NewRegistry() instances must not share state")
	}
}

// TestRegistry_Default_Good pins the shared registry's builtin coverage —
// every family buildDefaultRegistry wires in resolves.
func TestRegistry_Default_Good(t *testing.T) {
	registry := Default()
	for _, name := range []string{"qwen3", "gemma4_text", "generic"} {
		if _, ok := registry.Lookup(name); !ok {
			t.Fatalf("Default().Lookup(%q) = false, want a registered builtin parser", name)
		}
	}
}

// TestRegistry_Default_Bad pins the miss path on the shared registry: an
// unrecognised key still falls through to a safe false, not a panic.
func TestRegistry_Default_Bad(t *testing.T) {
	if p, ok := Default().Lookup("not-a-real-arch"); ok || p != nil {
		t.Fatalf("Default().Lookup(unknown) = %v/%v, want nil/false", p, ok)
	}
}

// TestRegistry_Default_Ugly pins the singleton contract: Default() is built
// once via core.Once, so repeated calls return the identical pointer.
func TestRegistry_Default_Ugly(t *testing.T) {
	// Two separate calls compared for pointer identity — deliberately not hoisted
	// to a single variable, since the thing under test is whether repeated calls
	// return the SAME instance. Not a copy-paste dupe.
	if Default() != Default() {
		t.Fatal("Default() must return the same shared instance on every call")
	}
}

// TestRegistry_Register_Good pins the multi-alias insert: every alias passed
// to one Register call resolves back to the same parser instance.
func TestRegistry_Register_Good(t *testing.T) {
	registry := NewRegistry()
	parser := customOutputParser{}
	registry.Register(parser, "alias-one", "alias-two")
	for _, alias := range []string{"alias-one", "alias-two"} {
		p, ok := registry.Lookup(alias)
		if !ok || p.ParserID() != "custom" {
			t.Fatalf("Lookup(%q) = %v/%v, want the custom parser", alias, p, ok)
		}
	}
}

// TestRegistry_Register_Bad pins the nil-parser guard: registering a nil
// parser is a silent no-op, not a panic or a map entry.
func TestRegistry_Register_Bad(t *testing.T) {
	registry := NewRegistry()
	registry.Register(nil, "should-not-appear")
	if _, ok := registry.Lookup("should-not-appear"); ok {
		t.Fatal("Register(nil, ...) must not insert a lookup entry")
	}
}

// TestRegistry_Register_Ugly pins the empty-alias guard: a blank alias in
// the variadic list is skipped while its siblings still register.
func TestRegistry_Register_Ugly(t *testing.T) {
	registry := NewRegistry()
	registry.Register(customOutputParser{}, "", "real-alias")
	if _, ok := registry.Lookup(""); ok {
		t.Fatal("an empty alias must never become a lookup key")
	}
	if _, ok := registry.Lookup("real-alias"); !ok {
		t.Fatal("siblings of a skipped empty alias must still register")
	}
}

// TestRegistry_Lookup_Good pins the hit path on the shared registry.
func TestRegistry_Lookup_Good(t *testing.T) {
	p, ok := Default().Lookup("qwen3")
	if !ok || p.ParserID() != "qwen" {
		t.Fatalf("Lookup(qwen3) = %v/%v, want the qwen parser", p, ok)
	}
}

// TestRegistry_Lookup_Bad pins the miss path: an unregistered key returns
// false with a nil parser, never a zero-value stand-in.
func TestRegistry_Lookup_Bad(t *testing.T) {
	registry := NewRegistry()
	p, ok := registry.Lookup("never-registered")
	if ok || p != nil {
		t.Fatalf("Lookup(miss) = %v/%v, want nil/false", p, ok)
	}
}

// TestRegistry_Lookup_Ugly pins the nil-receiver guard: Lookup on a nil
// *Registry returns false rather than dereferencing a nil map.
func TestRegistry_Lookup_Ugly(t *testing.T) {
	var registry *Registry
	if p, ok := registry.Lookup("qwen3"); ok || p != nil {
		t.Fatalf("nil Lookup() = %v/%v, want nil/false", p, ok)
	}
}

// TestRegistry_LookupHint_Good pins the direct family resolution path.
func TestRegistry_LookupHint_Good(t *testing.T) {
	if p := Default().LookupHint(Hint{Architecture: "gemma4_text"}); p.ParserID() != "gemma" {
		t.Fatalf("LookupHint(gemma4_text) = %q, want gemma", p.ParserID())
	}
}

// TestRegistry_LookupHint_Bad pins the unresolvable-hint path: a Registry
// with its own fallback set still answers an unknown architecture safely —
// with the fallback parser, not an error.
func TestRegistry_LookupHint_Bad(t *testing.T) {
	registry := NewRegistry()
	p := registry.LookupHint(Hint{Architecture: "not-a-real-arch"})
	if p == nil || p.ParserID() != "generic" {
		t.Fatalf("LookupHint(unknown) = %v, want the generic fallback", p)
	}
}

// TestRegistry_LookupHint_Ugly pins the nil-receiver delegation: a nil
// *Registry routes LookupHint through Default() rather than panicking.
func TestRegistry_LookupHint_Ugly(t *testing.T) {
	var registry *Registry
	if p := registry.LookupHint(Hint{Architecture: "qwen3"}); p == nil || p.ParserID() != "qwen" {
		t.Fatalf("nil LookupHint() = %v, want the default qwen parser", p)
	}
}

// TestRegistry_ForHint_Good pins the convenience wrapper against the shared
// registry's qwen family.
func TestRegistry_ForHint_Good(t *testing.T) {
	if p := ForHint(Hint{Architecture: "qwen3"}); p.ParserID() != "qwen" {
		t.Fatalf("ForHint(qwen3) = %q, want qwen", p.ParserID())
	}
}

// TestRegistry_ForHint_Bad pins the unresolvable-architecture path: it
// answers with the shared registry's generic fallback, not an error.
func TestRegistry_ForHint_Bad(t *testing.T) {
	if p := ForHint(Hint{Architecture: "not-a-real-arch"}); p.ParserID() != "generic" {
		t.Fatalf("ForHint(unknown) = %q, want generic", p.ParserID())
	}
}

// TestRegistry_ForHint_Ugly pins the zero-value boundary: an entirely empty
// Hint resolves the same way an unknown architecture does.
func TestRegistry_ForHint_Ugly(t *testing.T) {
	if p := ForHint(Hint{}); p.ParserID() != "generic" {
		t.Fatalf("ForHint(zero Hint) = %q, want generic", p.ParserID())
	}
}

// TestRegistry_HintFromInference_Good pins the field mapping: Architecture
// carries straight through.
func TestRegistry_HintFromInference_Good(t *testing.T) {
	if h := HintFromInference(inference.ModelInfo{Architecture: "qwen3"}); h.Architecture != "qwen3" {
		t.Fatalf("HintFromInference(qwen3) = %+v, want Architecture=qwen3", h)
	}
}

// TestRegistry_HintFromInference_Bad pins the zero-value input: an empty
// ModelInfo maps to an empty Hint, not a sentinel.
func TestRegistry_HintFromInference_Bad(t *testing.T) {
	if h := HintFromInference(inference.ModelInfo{}); h != (Hint{}) {
		t.Fatalf("HintFromInference(zero) = %+v, want zero Hint", h)
	}
}

// TestRegistry_HintFromInference_Ugly pins the narrow surface: fields beyond
// Architecture (VocabSize, NumLayers, ...) never leak into the Hint —
// AdapterName has no ModelInfo source and stays empty regardless.
func TestRegistry_HintFromInference_Ugly(t *testing.T) {
	info := inference.ModelInfo{Architecture: "gemma4_text", VocabSize: 32000, NumLayers: 26, HiddenSize: 4096, QuantBits: 4, QuantGroup: 32}
	h := HintFromInference(info)
	if h.AdapterName != "" {
		t.Fatalf("HintFromInference AdapterName = %q, want empty (no ModelInfo source)", h.AdapterName)
	}
}

type customOutputParser struct{}

func (customOutputParser) ParserID() string { return "custom" }

func (customOutputParser) ParseReasoning(_ []inference.Token, text string) (inference.ReasoningParseResult, error) {
	return inference.ReasoningParseResult{VisibleText: "custom:" + text}, nil
}

func (customOutputParser) ParseTools(_ []inference.Token, text string) (inference.ToolParseResult, error) {
	return inference.ToolParseResult{VisibleText: text}, nil
}
