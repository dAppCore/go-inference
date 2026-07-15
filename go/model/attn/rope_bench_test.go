// SPDX-Licence-Identifier: EUPL-1.2

package attn

import "testing"

var sinkRotaryDim int

func BenchmarkRopeParamsRotaryDim(b *testing.B) {
	p := RopeParams{HeadDim: 80, PartialRotaryFactor: 0.4}
	b.ReportAllocs()
	for b.Loop() {
		sinkRotaryDim, _ = p.RotaryDim()
	}
}
