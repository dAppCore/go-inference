// SPDX-Licence-Identifier: EUPL-1.2

package gguf

import core "dappco.re/go"

// QuantizeLane is an architecture's own dense-safetensors→GGUF conversion —
// canonical tensor names, per-tensor type policy, and the full metadata +
// tokenizer header a loader like llama.cpp expects — registered by a
// model/<arch> package (e.g. model/gemma4/gguf) from its init(). This lets
// QuantizeModelPack defer an architecture's special-cased export entirely to
// the arch's own package instead of a hardcoded per-arch branch living here,
// so model/gguf never imports an arch package (AX-8, lib never imports
// consumer) — the reactive registration model.RegisterArch already
// established for the engine's load path (model/arch_spec.go), one level
// down for the export path.
type QuantizeLane struct {
	// Detect reports whether configJSON (the source pack's config.json) is
	// this lane's architecture. QuantizeModelPack tries every registered
	// lane's Detect, in registration order, and defers to the first match.
	Detect func(configJSON []byte) bool

	// SupportsFormat reports whether the lane can export the requested
	// QuantizeFormat. QuantizeModelPack calls UnsupportedFormatError and
	// bails before calling Quantize when this returns false.
	SupportsFormat func(format QuantizeFormat) bool

	// UnsupportedFormatError builds the error QuantizeModelPack returns when
	// SupportsFormat rejects the requested format — the lane owns its own
	// wording (which formats it supports, worded however it likes) rather
	// than this package guessing at arch-specific phrasing.
	UnsupportedFormatError func(format QuantizeFormat) error

	// Quantize converts the dense source tensors into the lane's canonical
	// GGUF tensor and metadata records. tensors is the same
	// []DenseSafetensor QuantizeModelPack already decoded from the source
	// pack's safetensors shards — the lane does not re-read them.
	Quantize func(source Source, configJSON []byte, tensors []DenseSafetensor, format QuantizeFormat) ([]Tensor, []MetadataEntry, error)
}

// quantizeLanes holds each registered architecture's dedicated GGUF export
// lane, keyed by a diagnostic name (Names()/Each() bookkeeping only — Detect
// is what actually selects a lane for a given config.json).
var quantizeLanes = core.NewRegistry[QuantizeLane]()

// RegisterQuantizeLane records an architecture's dedicated GGUF export lane
// under name (a label for diagnostics only, not looked up directly). Call
// from the architecture package's init(), mirroring model.RegisterArch's
// reactive registration one level up.
//
//	gguf.RegisterQuantizeLane("gemma4", gguf.QuantizeLane{Detect: isGemma4Config, ...})
func RegisterQuantizeLane(name string, lane QuantizeLane) {
	quantizeLanes.Set(name, lane)
}

// lookupQuantizeLane returns the first registered lane (in registration
// order) whose Detect reports configJSON as its architecture, or ok=false if
// none does (including when no lane has registered at all).
func lookupQuantizeLane(configJSON []byte) (lane QuantizeLane, ok bool) {
	quantizeLanes.Each(func(_ string, candidate QuantizeLane) {
		if ok || candidate.Detect == nil {
			return
		}
		if candidate.Detect(configJSON) {
			lane, ok = candidate, true
		}
	})
	return lane, ok
}
