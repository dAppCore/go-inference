// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"os"
	"testing"
)

// BenchmarkGelu measures the composed float32 GELU over a dFF-sized buffer: ~10
// kernel dispatches plus the commit+wait per iteration. Synthetic (AX-11): no
// model load, dFF-sized buffer only.
func BenchmarkGelu(b *testing.B) {
	if os.Getenv(MetallibPathEnv) == "" {
		b.Skip("metallib not set")
	}
	if err := ensureInit(); err != nil {
		b.Fatal(err)
	}
	const dFF = 8192
	x := make([]float32, dFF)
	for i := range x {
		x[i] = float32((i*37)%160-80) * 0.05
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := Gelu(x); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkGeluGateMul measures gelu(gate)*up over a dFF-sized buffer — gemma's
// MLP gate composed on the float32 native path. Synthetic (AX-11).
func BenchmarkGeluGateMul(b *testing.B) {
	if os.Getenv(MetallibPathEnv) == "" {
		b.Skip("metallib not set")
	}
	if err := ensureInit(); err != nil {
		b.Fatal(err)
	}
	const dFF = 8192
	gate := make([]float32, dFF)
	up := make([]float32, dFF)
	for i := range gate {
		gate[i] = float32((i*31)%120-60) * 0.05
		up[i] = float32((i*17)%90-45) * 0.04
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := GeluGateMul(gate, up); err != nil {
			b.Fatal(err)
		}
	}
}
