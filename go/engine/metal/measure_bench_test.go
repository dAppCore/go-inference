// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import "testing"

func BenchmarkRebindCostProbe64(b *testing.B) {
	requireNativeRuntime(b)

	for i := 0; i < b.N; i++ {
		if _, err := rebindCostProbe(64); err != nil {
			b.Fatal(err)
		}
	}
}
