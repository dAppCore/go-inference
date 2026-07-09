// SPDX-Licence-Identifier: EUPL-1.2

// Package scheme is ROCm's pure component-contract layer: sequence mixers,
// cache/state holders, and weight-quant schemes. It mirrors the reactive
// registry shape used by go-mlx while keeping ROCm's runtime choices separate
// from any specific HIP, CUDA, or CPU implementation.
package scheme

import (
	"strings"

	core "dappco.re/go"
)

const RegistryContract = "rocm-scheme-registry-v1"

const (
	CacheModeDefault        = "default"
	CacheModeRecurrent      = "recurrent"
	CacheModeMLALatent      = "mla-latent"
	CacheModeCompaction     = "compaction"
	CacheModeCompactionFull = "compaction-full"
)

// StateKind is the state shape a sequence mixer requires from the cache layer.
type StateKind int

const (
	StateNone StateKind = iota
	StateKVCache
	StateRecurrent
)

func (state StateKind) String() string {
	switch state {
	case StateKVCache:
		return "kv-cache"
	case StateRecurrent:
		return "recurrent"
	default:
		return "none"
	}
}

// StateKindForString resolves state names used by model/profile labels.
func StateKindForString(state string) StateKind {
	switch strings.ToLower(strings.TrimSpace(state)) {
	case "kv", "kv-cache", "kvcache":
		return StateKVCache
	case "recurrent", "state", "recurrent-state":
		return StateRecurrent
	default:
		return StateNone
	}
}

// Mixer identifies a sequence-mixing scheme and the state holder it needs.
type Mixer interface {
	Kind() string
	State() StateKind
}

// CacheScheme identifies a state/cache holder and what state kind it serves.
type CacheScheme interface {
	Mode() string
	Serves() StateKind
}

// CacheModer is the optional mixer-owned cache factory override. Mixers with a
// bespoke state holder can name it directly; other mixers resolve by StateKind.
type CacheModer interface {
	CacheMode() string
}

// QuantScheme identifies a weight-quantization scheme and nominal bit width.
type QuantScheme interface {
	Kind() string
	Bits() int
}

var (
	mixers = core.NewRegistry[Mixer]()
	caches = core.NewRegistry[CacheScheme]()
	quants = core.NewRegistry[QuantScheme]()
)

// RegisterMixer registers or replaces a sequence-mixer scheme by Kind.
func RegisterMixer(mixer Mixer) core.Result {
	if mixer == nil || strings.TrimSpace(mixer.Kind()) == "" {
		return core.Result{}
	}
	return mixers.Set(normalizeToken(mixer.Kind()), mixer)
}

// RegisterCache registers or replaces a cache/state scheme by Mode.
func RegisterCache(cache CacheScheme) core.Result {
	if cache == nil || strings.TrimSpace(cache.Mode()) == "" {
		return core.Result{}
	}
	return caches.Set(normalizeToken(cache.Mode()), cache)
}

// RegisterQuant registers or replaces a weight-quant scheme by Kind.
func RegisterQuant(quant QuantScheme) core.Result {
	if quant == nil || strings.TrimSpace(quant.Kind()) == "" {
		return core.Result{}
	}
	return quants.Set(normalizeToken(quant.Kind()), quant)
}

// MixerFor resolves a sequence-mixer scheme by kind.
func MixerFor(kind string) (Mixer, bool) {
	if result := mixers.Get(normalizeToken(kind)); result.OK {
		if mixer, ok := result.Value.(Mixer); ok {
			return mixer, true
		}
	}
	return nil, false
}

// CacheFor resolves a cache/state scheme by mode.
func CacheFor(mode string) (CacheScheme, bool) {
	if result := caches.Get(normalizeToken(mode)); result.OK {
		if cache, ok := result.Value.(CacheScheme); ok {
			return cache, true
		}
	}
	return nil, false
}

// QuantFor resolves a weight-quant scheme by kind.
func QuantFor(kind string) (QuantScheme, bool) {
	if result := quants.Get(normalizeToken(kind)); result.OK {
		if quant, ok := result.Value.(QuantScheme); ok {
			return quant, true
		}
	}
	return nil, false
}

func MixerKinds() []string { return mixers.Names() }

func CacheModes() []string { return caches.Names() }

func QuantKinds() []string { return quants.Names() }

// CacheModeForMixer returns the cache scheme mode a mixer requires. A mixer may
// declare a bespoke mode; otherwise recurrent mixers get the recurrent holder
// and KV mixers get the default KV cache holder.
func CacheModeForMixer(mixer Mixer) string {
	if mixer == nil {
		return ""
	}
	if cacheMode, ok := mixer.(CacheModer); ok {
		if mode := normalizeToken(cacheMode.CacheMode()); mode != "" {
			return mode
		}
	}
	if mixer.State() == StateRecurrent {
		return CacheModeRecurrent
	}
	return CacheModeDefault
}

// CacheForMixer resolves the cache scheme a mixer requires.
func CacheForMixer(mixer Mixer) (CacheScheme, bool) {
	mode := CacheModeForMixer(mixer)
	if mode == "" {
		return nil, false
	}
	return CacheFor(mode)
}

// Compatible checks the mixer-owned-state contract.
func Compatible(mixer Mixer, cache CacheScheme) bool {
	if mixer == nil || cache == nil {
		return false
	}
	return mixer.State() == cache.Serves()
}

func normalizeToken(value string) string {
	value = strings.ToLower(strings.TrimSpace(value))
	return value
}
