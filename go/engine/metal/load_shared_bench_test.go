// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"testing"

	"dappco.re/go/inference/model"
)

func BenchmarkQWMapsLinear(b *testing.B) {
	lin := &model.Linear{
		Weight:    []byte{1, 2, 3},
		Scales:    []byte{4, 5},
		Biases:    []byte{6, 7},
		GroupSize: 64,
		Bits:      4,
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = qw(lin)
	}
}
