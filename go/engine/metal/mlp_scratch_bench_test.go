// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import "testing"

func BenchmarkMLPScratchComposed64x128(b *testing.B) {
	requireNativeRuntime(b)
	old := customLibraryLoaded
	customLibraryLoaded = false
	defer func() { customLibraryLoaded = old }()
	resetResidentBufsForTest()
	defer resetResidentBufsForTest()

	b.ReportAllocs()
	b.SetBytes(128 * bf16Size)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sc := newMLPScratch(64, 128)
		if sc.c044 == nil || sc.c079 == nil || sc.c1 == nil || sc.c05 == nil {
			b.Fatal("missing composed constants")
		}
	}
}
