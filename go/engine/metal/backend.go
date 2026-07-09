// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	core "dappco.re/go"
	"dappco.re/go/inference/model"
)

// NativeBackend is the no-cgo Metal implementation of model.Backend: it binds a gemma4
// Arch + the layer weights (bf16 OR 4-bit) and routes DecodeForward to the matching
// arch forward — re-encode or ICB replay, bf16 or qmv. It automatically falls back to
// the re-encode path for a MoE arch (the ICB replay can't host the router's host top-k).
// All four forwards share runArchDecode / decodeForwardArchICBCore via the projector
// seam; this backend is the single object the engine drives through model.Backend.
type NativeBackend struct {
	arch            model.Arch
	bf16            []DecodeLayerWeights    // set unless isQuant
	quant           []QuantizedLayerWeights // set when isQuant
	isQuant         bool
	useICB          bool
	maxLen          int
	pagedKVPageSize int
	pagedKVPrealloc bool
}

var _ model.Backend = (*NativeBackend)(nil)

// BackendOption configures a NativeBackend.
type BackendOption func(*NativeBackend)

// WithICB selects the ICB encode-bypass replay path (record once, replay per token).
// A MoE arch still uses the re-encode path (the ICB can't host the router readback).
func WithICB() BackendOption { return func(b *NativeBackend) { b.useICB = true } }

func withPagedKVPageSize(n int) BackendOption {
	return func(b *NativeBackend) { b.pagedKVPageSize = n }
}

func withPagedKVPrealloc(enabled bool) BackendOption {
	return func(b *NativeBackend) { b.pagedKVPrealloc = enabled }
}

// NewBF16Backend binds a bf16-weight gemma4 model behind model.Backend; len(layers)
// must equal the arch's layer count.
func NewBF16Backend(arch model.Arch, layers []DecodeLayerWeights, maxLen int, opts ...BackendOption) (*NativeBackend, error) {
	if len(layers) != len(arch.Layer) {
		return nil, core.NewError("native.NewBF16Backend: layers length must equal arch.Layer count")
	}
	if err := resolveSequenceSchemes(); err != nil {
		return nil, err
	}
	b := &NativeBackend{arch: arch, bf16: layers, maxLen: maxLen}
	for _, o := range opts {
		o(b)
	}
	return b, nil
}

// NewQuantBackend binds a 4-bit-weight gemma4 model behind model.Backend; len(qlayers)
// must equal the arch's layer count.
func NewQuantBackend(arch model.Arch, qlayers []QuantizedLayerWeights, maxLen int, opts ...BackendOption) (*NativeBackend, error) {
	if len(qlayers) != len(arch.Layer) {
		return nil, core.NewError("native.NewQuantBackend: layers length must equal arch.Layer count")
	}
	if err := resolveSequenceSchemes(); err != nil {
		return nil, err
	}
	b := &NativeBackend{arch: arch, quant: qlayers, isQuant: true, maxLen: maxLen}
	for _, o := range opts {
		o(b)
	}
	return b, nil
}

// DecodeForward runs the arch decode, routing to the fastest correct path for the
// backend's weights + arch. The attention scale is the standard 1/√headDim (a config
// query_pre_attn_scalar override is a later refinement); base/eps come from the arch.
func (b *NativeBackend) DecodeForward(inputs [][]byte) ([][]byte, error) {
	a := b.arch
	if a.PerLayerInputHidden > 0 {
		// PLE (E2B/E4B) needs the token id per layer; the whole-sequence forward has
		// only embeddings. model.Generate uses the incremental session (StepWithID) for these.
		return nil, core.NewError("native.NativeBackend.DecodeForward: per-layer-input models need the incremental session path, not whole-sequence decode")
	}
	dModel, nHeads, nKVHeads, headDim, dFF := a.Hidden, a.Heads, a.KVHeads, a.HeadDim, a.FF
	base, eps := a.RopeBase, a.Eps
	scale := attnScaleOf(a)
	sw := a.SlidingWindow
	icb := b.useICB && !a.HasMoE() // ICB can't host the MoE router → re-encode for MoE
	switch {
	case b.isQuant && icb:
		return DecodeForwardArchICBQuant(inputs, b.quant, a.Layer, dModel, nHeads, nKVHeads, headDim, b.maxLen, dFF, sw, base, scale, eps, a.ValueNorm)
	case b.isQuant:
		return DecodeForwardArchQuant(inputs, b.quant, a.Layer, dModel, nHeads, nKVHeads, headDim, b.maxLen, dFF, sw, base, scale, eps, a.ValueNorm)
	case icb:
		return DecodeForwardArchICB(inputs, b.bf16, a.Layer, dModel, nHeads, nKVHeads, headDim, b.maxLen, dFF, sw, base, scale, eps, a.ValueNorm)
	default:
		return DecodeForwardArch(inputs, b.bf16, a.Layer, dModel, nHeads, nKVHeads, headDim, b.maxLen, dFF, sw, base, scale, eps, a.ValueNorm)
	}
}

// DecodeForwardInto is DecodeForward with caller-owned output storage. Native
// arch routes write through their Into executors so backend callers avoid the
// allocate-then-copy compatibility path.
func (b *NativeBackend) DecodeForwardInto(outputs [][]byte, inputs [][]byte) ([][]byte, error) {
	a := b.arch
	if a.PerLayerInputHidden > 0 {
		return nil, core.NewError("native.NativeBackend.DecodeForwardInto: per-layer-input models need the incremental session path, not whole-sequence decode")
	}
	dModel, nHeads, nKVHeads, headDim, dFF := a.Hidden, a.Heads, a.KVHeads, a.HeadDim, a.FF
	base, eps := a.RopeBase, a.Eps
	scale := attnScaleOf(a)
	sw := a.SlidingWindow
	icb := b.useICB && !a.HasMoE()
	switch {
	case b.isQuant && icb:
		return DecodeForwardArchICBQuantInto(outputs, inputs, b.quant, a.Layer, dModel, nHeads, nKVHeads, headDim, b.maxLen, dFF, sw, base, scale, eps, a.ValueNorm)
	case b.isQuant:
		return DecodeForwardArchQuantInto(outputs, inputs, b.quant, a.Layer, dModel, nHeads, nKVHeads, headDim, b.maxLen, dFF, sw, base, scale, eps, a.ValueNorm)
	case icb:
		return DecodeForwardArchICBInto(outputs, inputs, b.bf16, a.Layer, dModel, nHeads, nKVHeads, headDim, b.maxLen, dFF, sw, base, scale, eps, a.ValueNorm)
	default:
		return DecodeForwardArchInto(outputs, inputs, b.bf16, a.Layer, dModel, nHeads, nKVHeads, headDim, b.maxLen, dFF, sw, base, scale, eps, a.ValueNorm)
	}
}
