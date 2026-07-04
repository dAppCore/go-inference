// SPDX-Licence-Identifier: EUPL-1.2

package scheme

// builtin.go registers, as the catalogue's entry-one, the schemes the engine
// already implements — identity + state contract only. A driver attaches the
// compute later by registering a value that also satisfies its driver-side
// interface (same Kind/Mode overwrites this metadata entry). The population —
// the flash-linear-attention mixers, TurboQuant, q4_0/mxfp4/nvfp4, the
// Attention-Matching compaction cache — registers alongside in its own file,
// with no edit here. That is the whole point of the registry: this file never
// grows a branch.

// info is the metadata-only scheme value: it satisfies all three contracts so
// one tiny type seeds the catalogue. Compute-bearing schemes are their own
// types in the driver.
type mixerInfo struct {
	kind  string
	state StateKind
}

func (m mixerInfo) Kind() string     { return m.kind }
func (m mixerInfo) State() StateKind { return m.state }

type cacheInfo struct {
	mode   string
	serves StateKind
}

func (c cacheInfo) Mode() string      { return c.mode }
func (c cacheInfo) Serves() StateKind { return c.serves }

// kvCacheInfo is a KV-cache scheme value that also carries its per-element byte
// width — the exact rational a memory planner sizes a KV cache from. It embeds
// cacheInfo (identity + StateKVCache) and satisfies CacheWidth. A recurrent
// holder stays a plain cacheInfo, so the width probe misses it: knownness and
// sizing both key off this one capability, not a duplicated mode list.
type kvCacheInfo struct {
	cacheInfo
	num, den uint64
	roundUp  bool
}

func (k kvCacheInfo) KVBytesPerElement() (num, den uint64, roundUp bool) {
	return k.num, k.den, k.roundUp
}

type quantInfo struct {
	kind string
	bits int
}

func (q quantInfo) Kind() string { return q.kind }
func (q quantInfo) Bits() int    { return q.bits }

type dtypeInfo struct {
	name  string
	bytes int
}

func (d dtypeInfo) Name() string { return d.name }
func (d dtypeInfo) Bytes() int   { return d.bytes }

// The activation/compute dtypes the engine's op layer moves tensors in — bf16 the
// narrow storage the residual stream rounds to, f32 the width Apple GPUs compute
// in. Exported (unlike the other builtins) because they are compile-time
// foundational to every backend's elementwise ops, like the StateKind constants.
var (
	BFloat16 DType = dtypeInfo{"bfloat16", 2}
	Float32  DType = dtypeInfo{"float32", 4}
)

func init() {
	// Sequence mixer the engine implements today: Gemma-4 hybrid softmax
	// attention (sliding-window local + periodic global, shared-KV) → KV cache.
	RegisterMixer(mixerInfo{"softmax-hybrid", StateKVCache})

	// KV-cache schemes the engine implements today (the KVCacheMode enum in
	// pkg/metal/cache.go — "" maps to "default"). All hold a growing K/V cache,
	// and each carries its per-element byte width (CacheWidth): the exact
	// rational a memory planner sizes a cache from. fp16/default/paged/fixed are
	// full-precision (2 bytes/element); q8 is 1; k-q8-v-q4 is 3/4 truncated;
	// turboquant is 7/16 rounded up (3.5 bits/element). The rounding is per
	// format — k-q8-v-q4 truncates, the TurboQuant ring rounds up.
	for _, kv := range []kvCacheInfo{
		{cacheInfo{"default", StateKVCache}, 2, 1, false},
		{cacheInfo{"fp16", StateKVCache}, 2, 1, false},
		{cacheInfo{"q8", StateKVCache}, 1, 1, false},
		{cacheInfo{"k-q8-v-q4", StateKVCache}, 3, 4, false},
		{cacheInfo{"paged", StateKVCache}, 2, 1, false},
		{cacheInfo{"fixed", StateKVCache}, 2, 1, false},
		{cacheInfo{"turboquant", StateKVCache}, 7, 16, true},
	} {
		RegisterCache(kv)
	}
	// Recurrent-state holder for SSM / linear-attention mixers — registered so
	// the contract exists; the first flash-linear-attention mixer task lands the
	// compute. It holds no growing KV, so it carries no CacheWidth: the planner's
	// width probe misses it and it is not a KV-cache mode.
	RegisterCache(cacheInfo{"recurrent", StateRecurrent})

	// Weight quant every engine implements today: group-affine. Bits 0 = the
	// model's config declares the width (4/6/8); the affine scheme reads it.
	RegisterQuant(quantInfo{"affine", 0})

	// Activation/compute dtypes the engine's op layer moves tensors in. Registered
	// so the elementwise kernels resolve their dtype through the scheme (the
	// "vv_Multiply"+Name suffix) instead of hardcoding it; fp8/… register alongside.
	RegisterDType(BFloat16)
	RegisterDType(Float32)
}
