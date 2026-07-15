// SPDX-Licence-Identifier: EUPL-1.2

package mamba2

import "testing"

// BenchmarkCausalConv1dF32 measures the causal depthwise conv1d over a prefill chunk. convDim spans the
// xBC stream (dInner + 2·N for nGroups=1), kernel K.
func BenchmarkCausalConv1dF32(b *testing.B) {
	const L, convDim, K = 256, benchH*benchP + 2*benchN, 4
	in := syn(L*convDim, 1)
	w := syn(convDim*K, 2)
	bias := syn(convDim, 3)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, _, err := CausalConv1dF32(in, w, bias, nil, L, convDim, K); err != nil {
			b.Fatal(err)
		}
	}
}
