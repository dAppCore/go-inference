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

type customOutputParser struct{}

func (customOutputParser) ParserID() string { return "custom" }

func (customOutputParser) ParseReasoning(_ []inference.Token, text string) (inference.ReasoningParseResult, error) {
	return inference.ReasoningParseResult{VisibleText: "custom:" + text}, nil
}

func (customOutputParser) ParseTools(_ []inference.Token, text string) (inference.ToolParseResult, error) {
	return inference.ToolParseResult{VisibleText: text}, nil
}
