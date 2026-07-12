// SPDX-Licence-Identifier: EUPL-1.2

package model

import core "dappco.re/go"

// ExampleQKNormalization shows how an architecture declares Cohere's
// mean-centred per-head Q/K operation without tying it to a family package.
func ExampleQKNormalization() {
	a := Arch{QKNormalization: QKLayerNorm}
	core.Println(a.QKNormalization)
	// Output: layer_norm
}

// ExampleLayerSpec_OwnsCache shows the owner/sharer distinction: a non-negative
// CacheIndex means this layer holds its own KV cache; -1 means it reads another
// layer's (see KVShareFrom).
func ExampleLayerSpec_OwnsCache() {
	owner := LayerSpec{CacheIndex: 0}
	sharer := LayerSpec{CacheIndex: -1, KVShareFrom: 0}
	core.Println(owner.OwnsCache())
	core.Println(sharer.OwnsCache())
	// Output:
	// true
	// false
}

// ExampleLayerSpec_TypeName shows the config-vocabulary spelling DeriveLayers derives
// from and KV-stream matching reads back.
func ExampleLayerSpec_TypeName() {
	core.Println(LayerSpec{Attention: SlidingAttention}.TypeName())
	core.Println(LayerSpec{Attention: GlobalAttention}.TypeName())
	// Output:
	// sliding_attention
	// full_attention
}

// ExampleArch_HasMoE shows the per-layer MoE check: true as soon as ANY layer is a
// sparse-expert layer, not requiring every layer to be one (a dense/MoE mixed arch).
func ExampleArch_HasMoE() {
	a := Arch{Layer: []LayerSpec{{MoE: false}, {MoE: true}}}
	core.Println(a.HasMoE())
	// Output: true
}

// ExampleArch_MaxHeadDim shows the backend buffer-sizing rule: the LARGER of the
// sliding and full (global) head_dim, so per-head scratch and KV-cache row strides fit
// both layer types.
func ExampleArch_MaxHeadDim() {
	a := Arch{HeadDim: 256, GlobalHeadDim: 512}
	core.Println(a.MaxHeadDim())
	// Output: 512
}

// ExampleArch_MaxKVHeads mirrors MaxHeadDim for the KV-head count a backend sizes cache
// rows to.
func ExampleArch_MaxKVHeads() {
	a := Arch{KVHeads: 2, GlobalKVHeads: 8}
	core.Println(a.MaxKVHeads())
	// Output: 8
}

// ExampleDeriveLayers shows the per-layer attention-type and KV-cache-sharing
// derivation: the first (n-numKVShared) layers own their cache; a later layer shares
// the most recent owner of the same attention type.
func ExampleDeriveLayers() {
	specs := DeriveLayers([]string{"sliding_attention", "full_attention", "sliding_attention"}, 1)
	core.Println(specs[0].OwnsCache()) // outside the shared region
	core.Println(specs[2].OwnsCache()) // in the shared region, shares layer 0's sliding cache
	core.Println(specs[2].KVShareFrom)
	// Output:
	// true
	// false
	// 0
}
