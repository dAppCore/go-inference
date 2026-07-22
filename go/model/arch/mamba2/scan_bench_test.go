// SPDX-Licence-Identifier: EUPL-1.2

package mamba2

import "testing"

// benchScanDims is a moderate SSD geometry for the scan benches (H heads, head_dim P, state N).
const benchH, benchP, benchN = 8, 64, 128

func benchScanInputs(L int) (x, dt, a, b, c, d []float32) {
	return syn(L*benchH*benchP, 1), syn(L*benchH, 2), syn(benchH, 3), syn(L*benchH*benchN, 4), syn(L*benchH*benchN, 5), syn(benchH, 6)
}

// BenchmarkSSDScanF32Prefill measures the selective scan over a prefill chunk (L tokens at once).
func BenchmarkSSDScanF32Prefill(b *testing.B) {
	const L = 256
	x, dt, a, bb, c, d := benchScanInputs(L)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, _, err := SSDScanF32(x, dt, a, bb, c, d, nil, L, benchH, benchP, benchN); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkSSDScanF32Decode measures the single-step (L=1) decode scan with a carried state — the
// per-token cost in streaming generation.
func BenchmarkSSDScanF32Decode(b *testing.B) {
	x, dt, a, bb, c, d := benchScanInputs(1)
	prior := syn(benchH*benchP*benchN, 7)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, _, err := SSDScanF32(x, dt, a, bb, c, d, prior, 1, benchH, benchP, benchN); err != nil {
			b.Fatal(err)
		}
	}
}
