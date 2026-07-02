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
	// pkg/metal/cache.go — "" maps to "default"). All hold a growing K/V cache.
	for _, mode := range []string{"default", "fp16", "q8", "k-q8-v-q4", "paged", "fixed", "turboquant"} {
		RegisterCache(cacheInfo{mode, StateKVCache})
	}
	// Recurrent-state holder for SSM / linear-attention mixers — registered so
	// the contract exists; the first flash-linear-attention mixer task lands
	// the compute.
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
