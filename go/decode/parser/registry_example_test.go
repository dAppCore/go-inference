// SPDX-Licence-Identifier: EUPL-1.2

package parser

import (
	core "dappco.re/go"
	"dappco.re/go/inference"
)

func ExampleNewRegistry() {
	registry := NewRegistry()
	p, ok := registry.Lookup("generic")
	core.Println(ok, p.ParserID())
	// Output: true generic
}

func ExampleDefault() {
	registry := Default()
	p, ok := registry.Lookup("qwen3")
	core.Println(ok, p.ParserID())
	// Output: true qwen
}

func ExampleRegistry_Register() {
	registry := NewRegistry()
	registry.Register(newBuiltinOutputParser("custom", genericMarkers()), "custom-family")
	p, ok := registry.Lookup("custom-family")
	core.Println(ok, p.ParserID())
	// Output: true custom
}

func ExampleRegistry_Lookup() {
	p, ok := Default().Lookup("gemma4_text")
	core.Println(ok, p.ParserID())
	// Output: true gemma
}

func ExampleRegistry_LookupHint() {
	p := Default().LookupHint(Hint{Architecture: "qwen3"})
	core.Println(p.ParserID())
	// Output: qwen
}

func ExampleForHint() {
	p := ForHint(Hint{Architecture: "gemma4_text"})
	core.Println(p.ParserID())
	// Output: gemma
}

func ExampleHintFromInference() {
	hint := HintFromInference(inference.ModelInfo{Architecture: "qwen3"})
	core.Println(hint.Architecture)
	// Output: qwen3
}
