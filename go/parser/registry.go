// SPDX-Licence-Identifier: EUPL-1.2

package parser

import (
	core "dappco.re/go"
	"dappco.re/go/inference"
)

//	type custom struct{ /* ... */ }
//	func (custom) ParserID() string { return "custom" }
//	// implement inference.ReasoningParser + inference.ToolParser
type OutputParser interface {
	ParserID() string
	inference.ReasoningParser
	inference.ToolParser
}

//	reg := parser.NewRegistry()
//	reg.Register(customParser, "custom", "custom-v2")
type Registry struct {
	parsers  map[string]OutputParser
	fallback OutputParser
}

//	reg := parser.NewRegistry()
func NewRegistry() *Registry {
	generic := newBuiltinOutputParser("generic", genericMarkers())
	return &Registry{
		parsers:  map[string]OutputParser{"generic": generic},
		fallback: generic,
	}
}

// Default returns the process-wide built-in parser registry. Built
// once via core.Once — every Processor / ForHint call shares the same
// instance instead of rebuilding all 11 parsers + their marker
// slices. The registry is read-only after construction (Register is
// safe on bespoke Registries created via NewRegistry, not on the
// shared default).
//
//	reg := parser.Default()
//	out := reg.LookupHint(parser.Hint{Architecture: "qwen3"})
func Default() *Registry {
	defaultOnce.Do(func() { defaultRegistry = buildDefaultRegistry() })
	return defaultRegistry
}

var (
	defaultRegistry *Registry
	defaultOnce     core.Once
)

func buildDefaultRegistry() *Registry {
	registry := NewRegistry()
	registry.Register(newBuiltinOutputParser("qwen", qwenMarkers()), "qwen", "qwen2", "qwen3")
	registry.Register(newBuiltinOutputParser("gemma", gemmaMarkers()), "gemma", "gemma3", "gemma4", "gemma4_text")
	registry.Register(newBuiltinOutputParser("minimax", qwenMarkers()), "minimax", "minimax_m2", "minimax-m2")
	registry.Register(newBuiltinOutputParser("deepseek-r1", qwenMarkers()), "deepseek", "deepseek_r1", "deepseek-r1")
	registry.Register(newBuiltinOutputParser("gpt-oss", gptOSSMarkers()), "gpt-oss", "gpt_oss", "gptoss")
	registry.Register(newBuiltinOutputParser("mistral", genericMarkers()), "mistral", "mixtral")
	registry.Register(newBuiltinOutputParser("kimi", qwenMarkers()), "kimi", "kimi_k2", "moonshot")
	registry.Register(newBuiltinOutputParser("glm", qwenMarkers()), "glm", "glm4", "chatglm")
	registry.Register(newBuiltinOutputParser("hermes", genericMarkers()), "hermes", "hermes2", "hermes3")
	registry.Register(newBuiltinOutputParser("granite", genericMarkers()), "granite", "ibm-granite")
	return registry
}

//	reg.Register(myParser, "alias1", "alias2")
func (registry *Registry) Register(parser OutputParser, aliases ...string) {
	if registry == nil || parser == nil {
		return
	}
	if registry.parsers == nil {
		registry.parsers = map[string]OutputParser{}
	}
	registry.parsers[NormaliseKey(parser.ParserID())] = parser
	for _, alias := range aliases {
		key := NormaliseKey(alias)
		if key == "" {
			continue
		}
		registry.parsers[key] = parser
	}
	if registry.fallback == nil {
		registry.fallback = parser
	}
}

//	if p, ok := reg.Lookup("qwen3"); ok { /* use p */ }
func (registry *Registry) Lookup(name string) (OutputParser, bool) {
	if registry == nil {
		return nil, false
	}
	parser, ok := registry.parsers[NormaliseKey(name)]
	return parser, ok
}

//	p := reg.LookupHint(parser.Hint{Architecture: "qwen3"})
func (registry *Registry) LookupHint(hint Hint) OutputParser {
	if registry == nil {
		return Default().LookupHint(hint)
	}
	if parser, ok := registry.Lookup(Family(hint)); ok {
		return parser
	}
	if registry.fallback != nil {
		return registry.fallback
	}
	return newBuiltinOutputParser("generic", genericMarkers())
}

//	p := parser.ForHint(parser.Hint{Architecture: "qwen3"})
func ForHint(hint Hint) OutputParser {
	return Default().LookupHint(hint)
}

//	hint := parser.HintFromInference(model.Info())
func HintFromInference(info inference.ModelInfo) Hint {
	return Hint{Architecture: info.Architecture}
}
