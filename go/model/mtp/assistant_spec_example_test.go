// SPDX-Licence-Identifier: EUPL-1.2

package mtp

import (
	core "dappco.re/go"
	"dappco.re/go/inference/model"
)

// ExampleRegisterAssistant shows the reactive assistant registration: a model package's
// init() claims its config.json model_type(s) AND, when it also exports GGUF, its
// general.architecture id — resolvable through LookupAssistant / LookupAssistantGGUF
// respectively.
func ExampleRegisterAssistant() {
	RegisterAssistant(AssistantSpec{
		ModelTypes: []string{"example4_assistant"},
		GGUFArch:   "example4-assistant",
	})
	_, byModelType := LookupAssistant("example4_assistant")
	_, byGGUF := LookupAssistantGGUF("example4-assistant")
	core.Println(byModelType)
	core.Println(byGGUF)
	// Output:
	// true
	// true
}

// ExampleLookupAssistant shows the config.json dispatch: a registered model_type
// resolves; an unregistered one misses (ok=false).
func ExampleLookupAssistant() {
	RegisterAssistant(AssistantSpec{ModelTypes: []string{"example5_assistant"}})
	_, ok := LookupAssistant("example5_assistant")
	core.Println(ok)
	_, ok = LookupAssistant("never-registered")
	core.Println(ok)
	// Output:
	// true
	// false
}

// ExampleLookupAssistantGGUF shows the GGUF general.architecture dispatch — a separate
// keyspace ("gguf:"-prefixed) from the config.json model_type lookup.
func ExampleLookupAssistantGGUF() {
	RegisterAssistant(AssistantSpec{ModelTypes: []string{"example6_assistant"}, GGUFArch: "example6-arch"})
	_, ok := LookupAssistantGGUF("example6-arch")
	core.Println(ok)
	// Output: true
}

// ExampleParseAssistantConfig shows the whole reactive config.json load in one call: probe
// the model_type, dispatch to the registered spec's Parse, stamp the inferred MTP method.
func ExampleParseAssistantConfig() {
	RegisterAssistant(AssistantSpec{
		ModelTypes: []string{"example7_assistant"},
		Parse: func([]byte) (AssistantConfig, error) {
			return AssistantConfig{ModelType: "example7_assistant"}, nil
		},
	})
	cfg, err := ParseAssistantConfig([]byte(`{"model_type":"example7_assistant"}`))
	if err != nil {
		return
	}
	core.Println(cfg.Method) // no Method declared on the spec → the MTPDraftModel default
	// Output: draft-model
}

// ExampleAssistantConfig_LayerType shows the declared-vs-derived fallback: an explicit
// LayerTypes entry wins; a blank one falls back to the drafter's own Arch layer TypeName
// (the KV-stream matching a target attachment reads).
func ExampleAssistantConfig_LayerType() {
	c := AssistantConfig{
		LayerTypes: []string{"sliding_attention", ""},
		Arch:       model.Arch{Layer: []model.LayerSpec{{}, {Attention: model.GlobalAttention}}},
	}
	core.Println(c.LayerType(0)) // the declared entry
	core.Println(c.LayerType(1)) // blank declared entry → falls back to the Arch layer
	// Output:
	// sliding_attention
	// full_attention
}
