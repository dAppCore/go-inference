// SPDX-Licence-Identifier: EUPL-1.2

// arch.go is the backend-agnostic decode-architecture declaration — the "what"
// (transformer dims + per-layer cache topology + the layer derivation), separated from
// any one backend's imperative Forward (the "how"). It is architecture-neutral: every
// arch describes itself as an Arch over the backend contract, and every executor
// (pkg/native, pkg/metal, future go-rocm) consumes it.
//
// It lives at the pkg/model ROOT, next to Backend / TokenModel / Sampler — NOT in a
// model subpackage. A model-named home is exactly what makes one arch import another
// just to get a general type; keeping the neutral contract neutral is what stops that recurring.
package model

// AttentionType is a layer's attention span.
type AttentionType uint8

const (
	GlobalAttention  AttentionType = iota // full_attention — attends the whole context
	SlidingAttention                      // sliding_attention — windowed
)

// QKNormalization declares the per-head operation applied to projected queries
// and keys before rotary position encoding. It is an architecture property, not
// inferred from the presence of weights: Cohere's LayerNorm may be enabled by a
// config switch, while most families use no QK operation.
type QKNormalization string

const (
	QKNone      QKNormalization = ""
	QKLayerNorm QKNormalization = "layer_norm"
)

// LayerSpec declares one decode layer's structure, backend-agnostic.
type LayerSpec struct {
	Attention   AttentionType
	KVShareFrom int  // index of the layer whose KV cache this layer reads (== own index if it owns its cache)
	CacheIndex  int  // cache slot for an owner; -1 if this layer shares another's cache
	MoE         bool // sparse-expert MLP instead of dense (derivation: a later slice)
	// HeadDim / KVHeads are this layer's RESOLVED attention geometry. Some archs use a
	// LARGER head_dim on full_attention layers than on sliding (e.g. sliding head_dim
	// 256, full global_head_dim 512), and may carry a different KV head count on full
	// layers (num_global_key_value_heads). Filled by Config.Arch
	// per the layer's attention type; a backend reads these per layer rather than the
	// single Arch.HeadDim. == the sliding/default values when the config draws no
	// distinction (synthetic + uniform packs).
	HeadDim int
	KVHeads int
}

// OwnsCache reports whether this layer holds its own KV cache (vs sharing).
func (l LayerSpec) OwnsCache() bool { return l.CacheIndex >= 0 }

// TypeName is the layer's attention type as configs spell it — the inverse of the
// DeriveLayers mapping, so KV-stream matching (e.g. drafter layer → target stream)
// speaks the config vocabulary.
func (l LayerSpec) TypeName() string {
	if l.Attention == SlidingAttention {
		return "sliding_attention"
	}
	return "full_attention"
}

// HasMoE reports whether any layer is a MoE (sparse-expert) layer — an arch may apply MoE
// uniformly, but the check is per-layer so a backend can route MoE archs off fast paths
// that can't host the router (the ICB replay).
func (a Arch) HasMoE() bool {
	for _, l := range a.Layer {
		if l.MoE {
			return true
		}
	}
	return false
}

// MoEGating is the router's expert-scoring/combination method — the sparse-expert
// analog of a dense FFN's fixed shape. INFERRED FROM THE MODEL: an arch's config
// declares it and the engine applies it, never assumes (the same DECLARES discipline
// as Arch.AttnScale / EmbedScale). Today the metal router ships softmax top-k; sigmoid
// gating, top-k weight renormalisation (norm_topk_prob), routed-scaling, and always-on
// shared experts — the deepseek / qwen3 / composed variants — each earn a value plus a
// router branch as they land (model/composed/moe.go already implements softmax +
// norm-topk + shared expert on the reference path).
type MoEGating string

// NormPlacement declares where a transformer's block norms sit relative to
// attention and feed-forward sublayers. It is architecture data: executors
// consume it rather than inferring residual order from optional weight names.
type NormPlacement string

const (
	NormPlacementUnspecified NormPlacement = ""
	NormPlacementPre         NormPlacement = "pre"
	NormPlacementPost        NormPlacement = "post"
)

const (
	// MoEGatingSoftmax: softmax over the top-k selected experts' scores (optionally
	// scaled per-expert). gemma4's MoE and the metal router's shipping path, and the
	// default for any MoE arch whose config leaves the gating unset.
	MoEGatingSoftmax MoEGating = "softmax"
)

// resolveMoEGating defaults an unset gating to MoEGatingSoftmax — the only router
// variant the metal engine ships today, and gemma4's method.
func resolveMoEGating(g MoEGating) MoEGating {
	if g == "" {
		return MoEGatingSoftmax
	}
	return g
}

// Arch is the full backend-agnostic decode declaration: the neutral transformer dims
// + the arch-specific extras + the derived per-layer specs. Built from a model config;
// consumed by a backend executor. (Dims are plain fields the loader fills from config;
// the per-layer derivation is DeriveLayers.)
type Arch struct {
	Hidden, EmbeddingDim, Heads, KVHeads, HeadDim, FF, Vocab int       // EmbeddingDim differs from Hidden when a checkpoint projects tied embeddings in/out; HeadDim/KVHeads are the sliding/default geometry
	GlobalHeadDim, GlobalKVHeads                             int       // full_attention head_dim / kv-head count (== HeadDim / KVHeads when the config draws no distinction)
	Experts, TopK, ExpertFF                                  int       // MoE dims (Experts == 0 → dense model); ExpertFF is the experts' intermediate size
	MoEGating                                                MoEGating // router expert-scoring method the model DECLARES (empty → softmax); see MoEGating
	FuseExpertGateUp                                         bool      // model opts its MoE experts into the fused gate+up path — a separate-gate/up checkpoint gets ExpGateUp synthesised at load (~34% MoE speed, trades the weights' mmap zero-copy for a heap copy)
	Eps                                                      float32
	AttnScale                                                float32   // attention SDPA scale the model DECLARES (the engine applies it, never assumes): e.g. 1.0 when a QK-norm IS the scaling, else 1/√headDim
	EmbedScale                                               float32   // token-embedding multiplier the model DECLARES (gemma-family √hidden; llama-family 1.0); 0 = undeclared → backends fall back to √hidden
	LogitsScaling                                            float32   // final-logit divisor the model DECLARES; 0 = no division
	LogitScale                                               float32   // final-logit multiplier (cohere logit_scale); 0 = none
	ResidualMultiplier                                       float32   // attention and MLP residual-branch multiplier; 0 = undeclared → 1
	RopeBase, RopeScale                                      float32   // RopeBase = global-attention RoPE theta
	RopeLocalBase                                            float32   // sliding-attention RoPE theta (an arch may use a smaller local theta)
	RotaryDim, RotaryDimLocal                                int       // rotated dims/head (partial rotary, e.g. full_attention=0.25·GlobalHeadDim); global / sliding
	RopeFreqs                                                []float32 // explicit per-dim inverse frequencies (YaRN long-context remap); len RotaryDim/2; nil ⇒ derive uniformly from RopeBase
	RopeShortFreqs                                           []float32 // short-context inverse frequencies for position-dependent LongRoPE
	RopeOriginalContext                                      int       // positions below this boundary use RopeShortFreqs; 0 = one static table
	SoftCap                                                  float32   // final logit soft-cap (0 = none)
	SlidingWindow                                            int
	PerLayerInputVocab, PerLayerInputHidden                  int             // per-layer-input aux embedding (0 = absent)
	AttentionKEqV                                            bool            // K == V (shared projection)
	ValueNorm                                                bool            // an arch may apply a no-scale per-head RMSNorm to V (metal's RMSNormNoScale); most don't
	ParallelResidual                                         bool            // attention and MLP consume the same normalised input, then both outputs join the residual
	ALiBi                                                    bool            // attention uses linear position bias instead of rotary embeddings
	TieWordEmbeddings                                        *bool           // nil = checkpoint presence decides; non-nil validates lm_head against config.json
	LearnedAbsolutePositions                                 bool            // token embeddings are offset by a learned position table
	PositionOffset                                           int             // learned-position table offset (OPT reserves positions 0 and 1)
	LayerNormBefore                                          bool            // attention and MLP are pre-norm; false declares post-norm
	NoFinalNorm                                              bool            // layer stack has no final norm (OPT post-norm checkpoints)
	MultiQueryAttention                                      bool            // one K/V head is shared by every query head
	Activation                                               string          // declared feed-forward activation (for example gelu_new)
	QKNormalization                                          QKNormalization // per-head Q/K normalisation before position encoding
	NormPlacement                                            NormPlacement   // declared norm-placement strategy (OLMo generations differ)
	NonParametricLayerNorm                                   bool            // LayerNorm has no learned scale or bias (OLMo 1)
	Layer                                                    []LayerSpec
}

// MaxHeadDim is the larger of the sliding and full head_dim — the head_dim a backend
// sizes per-head buffers (Q/K/V scratch, the KV cache row stride) to so both layer
// types fit. == HeadDim when the config draws no sliding/full distinction.
func (a Arch) MaxHeadDim() int {
	if a.GlobalHeadDim > a.HeadDim {
		return a.GlobalHeadDim
	}
	return a.HeadDim
}

// MaxKVHeads is the larger of the sliding and full KV-head count — the count a backend
// sizes KV-cache rows to. == KVHeads when the config draws no distinction.
func (a Arch) MaxKVHeads() int {
	if a.GlobalKVHeads > a.KVHeads {
		return a.GlobalKVHeads
	}
	return a.KVHeads
}

// DeriveLayers resolves the per-layer attention type and KV-cache-sharing map from a
// config — a faithful backend-agnostic lift of the metal model package's KV-cache-layout
// logic plus the layer_types rule. layerTypes is the config's
// per-layer "sliding_attention"/"full_attention"; numKVShared is
// num_kv_shared_layers. Rule: the first (n − numKVShared) layers OWN their cache;
// each later layer SHARES the KV cache of the most recent owner of the same
// attention type (and is itself promoted to owner if no such owner exists yet — the
// toy-config edge). Parity-gated against the metal impl (no model load needed).
func DeriveLayers(layerTypes []string, numKVShared int) []LayerSpec {
	n := len(layerTypes)
	specs := make([]LayerSpec, n)
	firstShared := min(max(n-numKVShared, 0), n)
	latestByType := map[AttentionType]int{}
	nextCache := 0
	for i := range n {
		at := GlobalAttention
		if layerTypes[i] == "sliding_attention" {
			at = SlidingAttention
		}
		specs[i] = LayerSpec{Attention: at, KVShareFrom: i, CacheIndex: -1}
		owns := i < firstShared
		if !owns {
			if prev, ok := latestByType[at]; ok {
				specs[i].KVShareFrom = prev
			} else {
				owns = true // first layer of this type lands in the shared region → promote to owner
			}
		}
		if owns {
			specs[i].KVShareFrom = i
			latestByType[at] = i
			specs[i].CacheIndex = nextCache
			nextCache++
		}
	}
	return specs
}
