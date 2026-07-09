// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import "testing"

func BenchmarkDispatchProfileOneBy64(b *testing.B) {
	requireNativeRuntime(b)

	for i := 0; i < b.N; i++ {
		if _, _, _, err := dispatchProfile(1, 64); err != nil {
			b.Fatal(err)
		}
	}
}
