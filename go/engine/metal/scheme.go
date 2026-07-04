// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	core "dappco.re/go"
	"dappco.re/go/inference/scheme"
)

// scheme.go is native's consumption of the shared pkg/scheme registries (R4 + R5): it registers the
// gemma4 sequence-mixer + KV-cache IDENTITIES and resolves them at backend construction, enforcing
// the mixer-owns-state contract. The pkg/scheme interfaces are identity + state-kind only (Mixer =
// Kind+State, CacheScheme = Mode+Serves) — the COMPUTE (the attention kernels, the growing-KV cache
// buffers) lives in native's decode (attention.go, decode_step.go). This mirrors the metal reference
// (pkg/metal/model/gemma4/softmax_mixer.go + pkg/metal/cache_scheme.go); native can't import pkg/metal
// (no-cgo), so it declares the same gemma4 identities here.

const (
	mixerSoftmaxHybrid = "softmax-hybrid" // gemma4's mixer kind (the FLA/SSM families register their own)
	cacheModeDefault   = ""               // KVCacheModeDefault — the full bf16 K/V cache
)

// softmaxHybridMixer is gemma4's sequence mixer identity: softmax attention with the hybrid
// sliding/global layer pattern, needing a standard KV cache. Kind matches the metal reference so a
// combined build resolves one scheme.
type softmaxHybridMixer struct{}

func (softmaxHybridMixer) Kind() string            { return mixerSoftmaxHybrid }
func (softmaxHybridMixer) State() scheme.StateKind { return scheme.StateKVCache }

// nativeKVCache is the full growing-KV cache identity (the default mode), serving the KV state the
// softmax mixer needs. The buffers + per-token grow live in the decode (DecodeStepKV).
type nativeKVCache struct{}

func (nativeKVCache) Mode() string             { return cacheModeDefault }
func (nativeKVCache) Serves() scheme.StateKind { return scheme.StateKVCache }

// init registers the identities, but only if absent — a combined metal+native build (the cross-engine
// parity test) already has metal's richer registrations (which also carry the compute surface), and
// the identity is the same either way, so native must not clobber them. In a native-only (no-cgo)
// build these are the registrations.
func init() {
	if _, ok := scheme.MixerFor(mixerSoftmaxHybrid); !ok {
		scheme.RegisterMixer(softmaxHybridMixer{})
	}
	if _, ok := scheme.CacheFor(cacheModeDefault); !ok {
		scheme.RegisterCache(nativeKVCache{})
	}
}

// resolveSequenceSchemes consumes the registries at backend construction (R4/R5): it resolves the
// registered sequence mixer + the KV-cache scheme and enforces scheme.Compatible (the mixer-owns-
// state contract), refusing a mismatched pairing at load rather than miscomputing. It is called
// unconditionally by NewBF16Backend/NewQuantBackend for any model.Arch — currently gemma4 is the
// only registered mixer kind (softmax-hybrid), but this function itself is arch-agnostic (it never
// names a model); the FLA/SSM/gated-delta families register + resolve through the same seam. The
// resolved schemes are identity + state-kind; native's decode owns the matching compute.
func resolveSequenceSchemes() error {
	m, ok := scheme.MixerFor(mixerSoftmaxHybrid)
	if !ok {
		return core.NewError("native: no sequence mixer registered for " + mixerSoftmaxHybrid)
	}
	c, ok := scheme.CacheFor(cacheModeDefault)
	if !ok {
		return core.NewError("native: no cache scheme registered for the default KV mode")
	}
	if !scheme.Compatible(m, c) {
		return core.NewError("native: mixer " + m.Kind() + " (" + m.State().String() +
			") incompatible with the resolved cache (serves " + c.Serves().String() + ")")
	}
	return nil
}
