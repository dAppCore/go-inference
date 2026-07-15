// SPDX-Licence-Identifier: EUPL-1.2

package opt

import "testing"

var sinkWeightNames any

func BenchmarkWeightNames(b *testing.B) {
	for i := 0; i < b.N; i++ {
		sinkWeightNames = WeightNames()
	}
}
