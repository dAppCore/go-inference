// SPDX-Licence-Identifier: EUPL-1.2

// Package scheme is the pluggable-component contract layer every inference
// engine shares: the three registries an engine resolves a model's components
// from — weight quant, KV/state cache, and sequence mixer. A model's config
// declares a kind for each; the engine looks it up and reacts, so adding a
// family member is "register a scheme", never an engine branch. A features
// probe says WHAT a model is; the scheme registries say HOW the engine
// provides each piece.
//
//	q, _ := scheme.QuantFor(cfg.QuantKind)   // "affine", "q4_0", "mxfp4", …
//	m, _ := scheme.MixerFor(cfg.MixerKind)   // "softmax-hybrid", "gla", "mamba2", …
//	c, _ := scheme.CacheFor(cfg.KVCacheMode) // "q8", "turboquant", "compaction", …
//	if !scheme.Compatible(m, c) { /* mixer needs a state this cache can't hold */ }
//
// Pure Go by design — these contracts carry no driver tensor type, so every
// Engine (metal on Apple, rocm on AMD/CUDA/CPU) inherits this one scheme
// catalogue. A driver attaches the compute by registering a value that also
// satisfies its driver-side compute interface; new families (the
// flash-linear-attention mixers, TurboQuant, q4_0, Attention-Matching
// compaction) register in their own file — no edit to any engine.
package scheme

import core "dappco.re/go"

// StateKind is what a sequence mixer needs the cache layer to hold for it. The
// mixer OWNS its state — it is the single truth of what it needs; a cache
// scheme only allocates, persists, and streams that state. This contract is
// what lets a Mamba/RWKV model load beside a softmax-attention one: each mixer
// declares its state kind, and the engine pairs it with a cache scheme that
// can serve that kind.
type StateKind int

const (
	StateNone      StateKind = iota // stateless mixer
	StateKVCache                    // softmax attention: a growing per-layer K/V cache (weight quant + compaction operate here)
	StateRecurrent                  // linear-attention / SSM: a fixed-size recurrent state, no growing KV
)

// String renders a StateKind for logs and error messages.
func (s StateKind) String() string {
	switch s {
	case StateKVCache:
		return "kv-cache"
	case StateRecurrent:
		return "recurrent"
	default:
		return "none"
	}
}

// Mixer identifies a sequence-mixing scheme — softmax attention, GLA, RetNet,
// DeltaNet, Mamba, RWKV, GSA, NSA, MoBA, … — and declares the state it needs.
// A driver registers a value implementing this together with its own compute
// interface; the contract here is identity + the mixer-owns-state declaration.
type Mixer interface {
	Kind() string     // the config token a model declares (e.g. "softmax-hybrid", "mamba2")
	State() StateKind // the state shape the mixer requires the cache layer to hold
}

// CacheScheme is how a mixer's state is stored, compressed, and streamed: full
// K/V, q8, k-q8-v-q4, paged, TurboQuant, Attention-Matching compaction, or a
// recurrent-state holder. Serves reports which StateKind it can hold so the
// engine can reject a cache/mixer pairing whose kinds disagree.
type CacheScheme interface {
	Mode() string      // the KVCacheMode token (e.g. "q8", "turboquant", "compaction")
	Serves() StateKind // the state kind this scheme can hold
}

// CacheWidth is the capability a KV-cache scheme adds when its per-element
// storage cost is a known exact rational — the byte ratio a memory planner
// sizes a KV cache from, in place of a per-mode byte table. Registered
// alongside the identity contract, so "what does this KV mode cost" is a
// registry lookup, never a switch. A recurrent-state holder serves no growing
// KV, carries no width, and the probe simply misses — exactly as fp16/paged
// miss kv.CacheProvider in #261.
//
//	if w, ok := cacheScheme.(scheme.CacheWidth); ok {
//	    num, den, roundUp := w.KVBytesPerElement() // 7, 16, true for turboquant
//	}
type CacheWidth interface {
	CacheScheme
	// KVBytesPerElement is the per-element KV storage cost as an exact rational
	// num/den (bytes per element) plus whether a fractional remainder rounds up:
	// fp16/default/paged/fixed 2/1, q8 1/1, k-q8-v-q4 3/4 (truncated), turboquant
	// 7/16 rounded up (= 3.5 bits per element). num/den are neutral facts of the
	// wire format; the rounding is the format's own (k-q8-v-q4 truncates, the
	// TurboQuant ring rounds up), not a caller policy.
	KVBytesPerElement() (num, den uint64, roundUp bool)
}

// QuantScheme is a weight-quantisation format — affine (mlx group-affine),
// q4_0, mxfp4, nvfp4, autoround, … It loads packed weights, runs the packed
// matmul, and (for the quantize verb) packs a dense weight. The contract here
// is identity + nominal bit-width; the driver attaches the ops.
type QuantScheme interface {
	Kind() string // the quantization.kind a model declares ("affine", "q4_0", …)
	Bits() int    // nominal bit-width; 0 means "the model's config declares it"
}

// DType is an activation/compute dtype scheme — the storage format of the LIVE
// tensors the engine moves between ops (the residual stream, the gelu
// intermediates), as opposed to QuantScheme which is a WEIGHT storage format.
// Apple GPUs compute in fp32; bf16 is the narrow storage the engine rounds to on
// every store, so it is a registered dtype scheme exactly as a weight quant is —
// not a "bfloat16" suffix hardcoded at the op call-sites. Name is the canonical
// token a model config declares (torch_dtype) and the metallib kernel-name
// suffix; Bytes is the element size.
//
//	dt, _ := scheme.DTypeFor(cfg.TorchDType) // "bfloat16", "float32", …
//	kernel := "vv_Multiply" + dt.Name()      // the elementwise multiply for that dtype
type DType interface {
	Name() string // config token + metallib kernel-name suffix: "bfloat16", "float32"
	Bytes() int   // element size in bytes (bfloat16=2, float32=4)
}

// The four registries — each mirrors the model/backend registry (one named
// collection, insertion-ordered, thread-safe). A new scheme is one Set().
var (
	mixers = core.NewRegistry[Mixer]()
	caches = core.NewRegistry[CacheScheme]()
	quants = core.NewRegistry[QuantScheme]()
	dtypes = core.NewRegistry[DType]()
)

// RegisterMixer adds (or overwrites) a sequence-mixer scheme by its Kind.
//
//	func init() { scheme.RegisterMixer(gla{}) }
func RegisterMixer(m Mixer) core.Result { return mixers.Set(m.Kind(), m) }

// RegisterCache adds (or overwrites) a cache scheme by its Mode.
func RegisterCache(c CacheScheme) core.Result { return caches.Set(c.Mode(), c) }

// RegisterQuant adds (or overwrites) a weight-quant scheme by its Kind.
func RegisterQuant(q QuantScheme) core.Result { return quants.Set(q.Kind(), q) }

// RegisterDType adds (or overwrites) an activation/compute dtype scheme by its Name.
func RegisterDType(d DType) core.Result { return dtypes.Set(d.Name(), d) }

// MixerFor resolves a registered sequence mixer by kind.
func MixerFor(kind string) (Mixer, bool) {
	if r := mixers.Get(kind); r.OK {
		return r.Value.(Mixer), true
	}
	return nil, false
}

// CacheFor resolves a registered cache scheme by mode.
func CacheFor(mode string) (CacheScheme, bool) {
	if r := caches.Get(mode); r.OK {
		return r.Value.(CacheScheme), true
	}
	return nil, false
}

// QuantFor resolves a registered weight-quant scheme by kind.
func QuantFor(kind string) (QuantScheme, bool) {
	if r := quants.Get(kind); r.OK {
		return r.Value.(QuantScheme), true
	}
	return nil, false
}

// DTypeFor resolves a registered activation/compute dtype by name.
func DTypeFor(name string) (DType, bool) {
	if r := dtypes.Get(name); r.OK {
		return r.Value.(DType), true
	}
	return nil, false
}

// MixerKinds, CacheModes, QuantKinds, DTypeNames list the registered names in
// registration order — the engine's "what can I load" catalogue.
func MixerKinds() []string { return mixers.Names() }
func CacheModes() []string { return caches.Names() }
func QuantKinds() []string { return quants.Names() }
func DTypeNames() []string { return dtypes.Names() }

// Compatible enforces the mixer-owns-state contract: a cache scheme may serve a
// mixer only if it holds the state kind the mixer declares it needs. The engine
// calls this at load and refuses a mismatched pairing rather than miscomputing.
func Compatible(m Mixer, c CacheScheme) bool {
	if m == nil || c == nil {
		return false
	}
	return c.Serves() == m.State()
}
