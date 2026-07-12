// SPDX-Licence-Identifier: EUPL-1.2

package model

import (
	core "dappco.re/go"
	"dappco.re/go/inference/model/safetensors"
)

// arch_spec.go is the REACTIVE architecture contract: a model package declares itself once — its config
// parser and (with model.Assemble) its weight-name conventions — and the engine's loader REACTS to that
// declaration. Adding an architecture becomes a config + a registration, not a re-implementation of the
// load path. It supersedes the dispatch-only model_type loader registry
// to react across the WHOLE load:
//
//	read config.json → probe model_type → LookupArch → spec.Parse → cfg.InferFromWeights →
//	cfg.Arch() → model.Assemble(tensors, arch, spec.Weights)
//
// model.Load (engine move 3) is that orchestration, and it lives here in the backend-agnostic root —
// so every backend (native, go-rocm) inherits ONE reactive loader rather than re-rolling it.

// ArchConfig is one architecture's parsed, validated config as the loader drives it: it resolves any
// dimension the config omits from the weight SHAPES (the don't-guess rule — see InferHeadDim), then
// derives the neutral decode Arch. InferFromWeights is a no-op for an architecture that declares every
// dimension; the dim-from-shape SELECTION (which weight, which attention-typed layer) is
// genuine per-arch logic, so it is a method here rather than declared data.
type ArchConfig interface {
	InferFromWeights(weights map[string]safetensors.Tensor)
	Arch() (Arch, error)
}

// ArchSpec is the declaration a model package registers from its init(): how to parse its config, and the
// weight-name conventions model.Assemble reacts to (StandardWeightNames + the arch's overrides).
type ArchSpec struct {
	ModelTypes []string                         // config.json "model_type" ids (incl. multimodal wrapper aliases)
	Parse      func([]byte) (ArchConfig, error) // the architecture's own parse: wrapper-merge / validation / defaults
	Weights    WeightNames                      // logical weight role → tensor name; model.Assemble reacts to it
	// Composed builds a hybrid (non-Assemble) TokenModel straight from the checkpoint tensors + config
	// bytes — a config-composed stack (Qwen 3.6 gated-delta + full attention) whose linear_attention
	// layers have no q/k/v for the reactive transformer Assemble to react to. When set, the arch is routed
	// here (model.LoadComposedDir) instead of Assemble; the returned model is already serve-ready. nil for
	// a standard transformer arch. This lets a backend reach a hybrid loader through the registry rather
	// than a hardcoded model_type switch.
	Composed  func(map[string]safetensors.Tensor, []byte) (TokenModel, error)
	Normalize func(map[string]safetensors.Tensor) map[string]safetensors.Tensor
	// NormalizeConfig handles fused layouts whose split depends on parsed geometry.
	NormalizeConfig func(map[string]safetensors.Tensor, ArchConfig) map[string]safetensors.Tensor
	Vision          func(map[string]safetensors.Tensor, ArchConfig) (*LoadedVision, error)
	// UnifiedVision assembles the encoder-free vision payload (gemma4_unified);
	// packs with an encoder tower return nil here and populate Vision instead.
	UnifiedVision func(map[string]safetensors.Tensor, ArchConfig) (*LoadedUnifiedVision, error)
	Audio         func(map[string]safetensors.Tensor, ArchConfig) (*LoadedAudio, error)
	Diffusion     func(map[string]safetensors.Tensor, ArchConfig) (*LoadedDiffusion, error)
}

// archSpecs is the engine's architecture registry — the same core.NewRegistry primitive pkg/scheme
// and pkg/model/quant.go use, not a hand-rolled map. A model package Set()s its spec from init().
var archSpecs = core.NewRegistry[ArchSpec]()

// RegisterArch registers spec under each of its ModelTypes; a later registration for the same id
// overrides. Call from a model package's init() so the reactive loader needs no central switch.
func RegisterArch(spec ArchSpec) {
	for _, mt := range spec.ModelTypes {
		if mt != "" {
			archSpecs.Set(mt, spec)
		}
	}
}

// LookupArch returns the spec registered for a model_type, or ok=false when none is.
func LookupArch(modelType string) (ArchSpec, bool) {
	if r := archSpecs.Get(modelType); r.OK {
		return r.Value.(ArchSpec), true
	}
	return ArchSpec{}, false
}
