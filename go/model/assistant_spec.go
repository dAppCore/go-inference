// SPDX-Licence-Identifier: EUPL-1.2

package model

import (
	core "dappco.re/go"
)

// assistant_spec.go is the REACTIVE attached-drafter contract, the assistant-side twin of
// arch_spec.go: a model package declares how its MTP assistant checkpoints parse (config.json
// and/or GGUF metadata) and the engine's assistant loader REACTS to that declaration. An
// attached drafter (an "assistant" in the HF assisted-generation sense) is the speculative
// draft head that projects from the TARGET model's hidden state ([token embed ⊕ target
// hidden] → its own small decode stack → draft logits) and shares the target's KV streams —
// which is why its config carries the target-facing dims alongside its own Arch. The engine
// never keys on a model name: it probes model_type / general.architecture and dispatches to
// whatever spec claimed it. gemma4's -assistant checkpoints are the shipping example.

// AssistantConfig is one attached drafter's parsed, validated declaration, backend-agnostic:
// the drafter's own decode Arch plus the target-attachment dims the pairing validation needs.
// Produced by a registered AssistantSpec parser; consumed blind by a backend's assistant
// loader.
type AssistantConfig struct {
	ModelType         string
	BackboneHidden    int          // the TARGET hidden size the drafter's input projection consumes
	NumCentroids      int          // ordered-embedding head: centroid count (0 = plain LM head)
	CentroidTopK      int          // ordered-embedding head: intermediate top-K
	OrderedEmbeddings bool         // logits via the ordered-embedding (centroid) head
	LayerTypes        []string     // per-layer attention type names — matched against the target's KV streams
	Arch              Arch         // the drafter's OWN decode architecture, fully derived
	Quant             *QuantConfig // quantization block (nil = bf16) — quantised tensor-shape validation reads it
}

// LayerType returns the declared attention-type name for layer idx, falling back to the
// Arch layer's own TypeName when the config declared none — the name the target KV-stream
// matching keys on.
func (c AssistantConfig) LayerType(idx int) string {
	if idx >= 0 && idx < len(c.LayerTypes) && c.LayerTypes[idx] != "" {
		return c.LayerTypes[idx]
	}
	if idx >= 0 && idx < len(c.Arch.Layer) {
		return c.Arch.Layer[idx].TypeName()
	}
	return ""
}

// AssistantSpec is the declaration a model package registers from its init(): how to
// recognise and parse its assistant checkpoints. Parse handles a config.json; the GGUF
// trio handles a single-file GGUF export of the same drafter (GGUFArch is the
// general.architecture value the spec claims, GGUFWeightName maps the GGUF tensor names
// onto the canonical checkpoint names, ParseGGUF builds the config from GGUF metadata —
// vocabHint carries the embed-derived vocab for exports that omit vocab_size).
type AssistantSpec struct {
	ModelTypes     []string
	Parse          func(data []byte) (AssistantConfig, error)
	GGUFArch       string
	ParseGGUF      func(meta map[string]any, vocabHint int) (AssistantConfig, error)
	GGUFWeightName func(name string) string
}

// assistantSpecs is the engine's assistant registry — the same core.NewRegistry primitive
// the arch registry uses. A model package Set()s its spec from init().
var assistantSpecs = core.NewRegistry[AssistantSpec]()

// RegisterAssistant registers spec under each of its ModelTypes (and its GGUFArch, prefixed
// "gguf:", when set); a later registration for the same id overrides. Call from a model
// package's init() so the loader needs no central switch. A spec may claim the empty
// model_type ("") to own checkpoints that predate the field.
func RegisterAssistant(spec AssistantSpec) {
	for _, mt := range spec.ModelTypes {
		assistantSpecs.Set(mt, spec)
	}
	if spec.GGUFArch != "" {
		assistantSpecs.Set("gguf:"+spec.GGUFArch, spec)
	}
}

// LookupAssistant returns the spec registered for a config.json model_type, or ok=false.
func LookupAssistant(modelType string) (AssistantSpec, bool) {
	if r := assistantSpecs.Get(modelType); r.OK {
		return r.Value.(AssistantSpec), true
	}
	return AssistantSpec{}, false
}

// LookupAssistantGGUF returns the spec registered for a GGUF general.architecture, or ok=false.
func LookupAssistantGGUF(arch string) (AssistantSpec, bool) {
	if r := assistantSpecs.Get("gguf:" + arch); r.OK {
		return r.Value.(AssistantSpec), true
	}
	return AssistantSpec{}, false
}

// ParseAssistantConfig probes data's model_type and dispatches to the registered spec —
// the whole reactive load in one call for the config.json path.
func ParseAssistantConfig(data []byte) (AssistantConfig, error) {
	var probe struct {
		ModelType string `json:"model_type"`
	}
	if r := core.JSONUnmarshal(data, &probe); !r.OK {
		return AssistantConfig{}, core.NewError("assistant config probe failed: " + r.Error())
	}
	spec, ok := LookupAssistant(probe.ModelType)
	if !ok {
		return AssistantConfig{}, core.NewError("assistant config declares no registered model_type: " + probe.ModelType)
	}
	return spec.Parse(data)
}
