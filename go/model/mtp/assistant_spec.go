// SPDX-Licence-Identifier: EUPL-1.2

package mtp

import (
	core "dappco.re/go"
	"dappco.re/go/inference/model"
)

// assistant_spec.go is the REACTIVE attached-drafter contract, the assistant-side twin of
// model.arch_spec.go: a model package declares how its MTP assistant checkpoints parse (config.json
// and/or GGUF metadata) and the engine's assistant loader REACTS to that declaration. An
// attached drafter (an "assistant" in the HF assisted-generation sense) is the speculative
// draft head that projects from the TARGET model's hidden state ([token embed ⊕ target
// hidden] → its own small decode stack → draft logits) and shares the target's KV streams —
// which is why its config carries the target-facing dims alongside its own Arch. The engine
// never keys on a model name: it probes model_type / general.architecture and dispatches to
// whatever spec claimed it. gemma4's -assistant checkpoints are the shipping example.

// MTPMethod is the speculative-decode method a drafter uses. It is INFERRED FROM
// THE MTP MODEL — the drafter's registered AssistantSpec declares it, and
// ParseAssistantConfig stamps it onto the AssistantConfig — so the decode path
// dispatches on the method instead of assuming one. Today only the separate
// draft-model method ships; EAGLE-style feature drafters, Medusa-style parallel
// heads, in-model MTP heads, and n-gram / prompt-lookup each earn their own
// constant plus a decode branch as they land.
type MTPMethod string

const (
	// MTPDraftModel: a standalone assistant model proposes tokens the target
	// verifies — it projects [token embed ⊕ target hidden] through its own small
	// decode stack while sharing the target's KV streams. gemma4's -assistant
	// checkpoints are this method, and it is the default for any drafter whose
	// spec leaves the method unset (every checkpoint predating this field).
	MTPDraftModel MTPMethod = "draft-model"

	// MTPDFlash: a block-diffusion draft model (DFlash, arXiv 2602.06036)
	// proposes a whole block of tokens in ONE parallel forward, conditioned on
	// fused hidden states drawn from several verifier layers, which the target
	// verifies with the ordinary greedy prefix-accept. The value matches the
	// checkpoint's speculators_model_type marker. The drafter/verify contract
	// itself lives, model-free and provably lossless, in decode/dflash; the
	// engine-side block-diffusion draft forward + fused multi-layer hidden
	// extraction are an evidenced gap (docs/design-dflash.md), so serving
	// recognises a DFlash checkpoint and reports the gap rather than faking a
	// lane.
	MTPDFlash MTPMethod = "dflash"
)

// resolveMTPMethod defaults an unset method to MTPDraftModel — the only method
// shipped today and the one every legacy checkpoint uses.
func resolveMTPMethod(m MTPMethod) MTPMethod {
	if m == "" {
		return MTPDraftModel
	}
	return m
}

// AssistantConfig is one attached drafter's parsed, validated declaration, backend-agnostic:
// the drafter's own decode Arch plus the target-attachment dims the pairing validation needs.
// Produced by a registered AssistantSpec parser; consumed blind by a backend's assistant
// loader. Method carries the speculative method inferred from the drafter (see MTPMethod).
type AssistantConfig struct {
	ModelType         string
	Method            MTPMethod    // speculative method inferred from the drafter (default MTPDraftModel)
	BackboneHidden    int          // the TARGET hidden size the drafter's input projection consumes
	NumCentroids      int          // ordered-embedding head: centroid count (0 = plain LM head)
	CentroidTopK      int          // ordered-embedding head: intermediate top-K
	OrderedEmbeddings bool         // logits via the ordered-embedding (centroid) head
	LayerTypes        []string           // per-layer attention type names — matched against the target's KV streams
	Arch              model.Arch         // the drafter's OWN decode architecture, fully derived
	Quant             *model.QuantConfig // quantization block (nil = bf16) — quantised tensor-shape validation reads it
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
	Method         MTPMethod // speculative method this arch's drafters use (empty = MTPDraftModel)
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
	cfg, err := spec.Parse(data)
	if err != nil {
		return cfg, err
	}
	cfg.Method = resolveMTPMethod(spec.Method)
	return cfg, nil
}
