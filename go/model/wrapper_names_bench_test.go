// SPDX-Licence-Identifier: EUPL-1.2

package model

import (
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference/model/safetensors"
)

// The NormalizeWrapperNames benches baseline the multimodal-wrapper alias pass (AX-11): at
// load, a "language_model."-prefixed tensor set is rebuilt so every prefixed name is ALSO
// addressable by its stripped "model.…" name. The flat (no-prefix) case returns the input
// untouched — the zero-alloc fast path — while the wrapper case allocates one map sized to
// the input plus the extra stripped aliases. Realistic input: a few-hundred-tensor set.

func benchTensorSet(n int, prefix string) map[string]safetensors.Tensor {
	m := make(map[string]safetensors.Tensor, n)
	for i := 0; i < n; i++ {
		name := prefix + "model.layers." + core.Sprintf("%d", i) + ".self_attn.q_proj.weight"
		m[name] = safetensors.Tensor{Shape: []int{4096, 4096}}
	}
	return m
}

// BenchmarkNormalizeWrapperNames_Flat — a text-only pack with no wrapper prefix: the scan
// finds no prefix and returns the input map unchanged. Expected zero allocs — the floor a
// flat checkpoint pays.
func BenchmarkNormalizeWrapperNames_Flat(b *testing.B) {
	t := benchTensorSet(300, "")
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = NormalizeWrapperNames(t)
	}
}

// BenchmarkNormalizeWrapperNames_Wrapped — a multimodal wrapper layout: every tensor is
// prefixed, so the pass allocates a new map and doubles the entries (prefixed + stripped).
// The map build + the stripped-key inserts are the whole cost.
func BenchmarkNormalizeWrapperNames_Wrapped(b *testing.B) {
	t := benchTensorSet(300, "language_model.")
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = NormalizeWrapperNames(t)
	}
}
