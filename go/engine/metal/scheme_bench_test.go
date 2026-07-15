// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import "testing"

func BenchmarkResolveSequenceSchemes(b *testing.B) {
	for i := 0; i < b.N; i++ {
		if err := resolveSequenceSchemes(); err != nil {
			b.Fatal(err)
		}
	}
}
