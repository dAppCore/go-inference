// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import "testing"

func BenchmarkRMSNormResidualBF16Axis1536(b *testing.B) {
	requireNativeRuntime(b)
	if !gpuHasGeluKernel() {
		b.Skip("custom kernel library (lthn_kernels.metallib) not loaded")
	}

	const axisSize = 1536
	const eps = float32(1e-6)
	x, w, res := rmsNormResidualFixture(axisSize)
	b.SetBytes(int64(len(x) + len(w) + len(res)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := RMSNormResidualBF16(x, w, res, axisSize, eps); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkRMSNormResidualBF16IntoAxis1536(b *testing.B) {
	requireNativeRuntime(b)
	if !gpuHasGeluKernel() {
		b.Skip("custom kernel library (lthn_kernels.metallib) not loaded")
	}

	const axisSize = 1536
	const eps = float32(1e-6)
	x, w, res := rmsNormResidualFixture(axisSize)
	out := make([]byte, axisSize*bf16Size)
	b.SetBytes(int64(len(x) + len(w) + len(res)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := RMSNormResidualBF16Into(out, x, w, res, axisSize, eps); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkRMSNormResidualBF16AlternatingAxis(b *testing.B) {
	requireNativeRuntime(b)
	if !gpuHasGeluKernel() {
		b.Skip("custom kernel library (lthn_kernels.metallib) not loaded")
	}

	const eps = float32(1e-6)
	cases := []struct {
		axis      int
		x, w, res []byte
	}{
		{axis: 512},
		{axis: 1536},
	}
	for i := range cases {
		cases[i].x, cases[i].w, cases[i].res = rmsNormResidualFixture(cases[i].axis)
		if _, err := RMSNormResidualBF16(cases[i].x, cases[i].w, cases[i].res, cases[i].axis, eps); err != nil {
			b.Fatalf("warmup axis %d: %v", cases[i].axis, err)
		}
	}
	b.SetBytes(int64(len(cases[0].x) + len(cases[0].w) + len(cases[0].res) + len(cases[1].x) + len(cases[1].w) + len(cases[1].res)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		c := cases[i&1]
		if _, err := RMSNormResidualBF16(c.x, c.w, c.res, c.axis, eps); err != nil {
			b.Fatal(err)
		}
	}
}
