// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import "testing"

func BenchmarkEnsureInitCached(b *testing.B) {
	requireNativeRuntime(b)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := ensureInit(); err != nil {
			b.Fatal(err)
		}
	}
}
