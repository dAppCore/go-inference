// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import "testing"

func BenchmarkForwardCaptureHiddensDense(b *testing.B) {
	requireNativeRuntime(b)
	mk := newMTPDecodeFixture(b)
	ids := []int32{1, 2, 3, 4, 5, 6, 7, 8}
	sess := mk()
	if _, _, err := sess.ForwardCaptureHiddens(ids); err != nil {
		b.Fatalf("ForwardCaptureHiddens warmup: %v", err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, _, err := sess.ForwardCaptureHiddens(ids); err != nil {
			b.Fatalf("ForwardCaptureHiddens: %v", err)
		}
	}
}

func BenchmarkForwardCaptureHiddensICB(b *testing.B) {
	requireNativeRuntime(b)
	g, arch, maxLen := icbSessionStateFixture(b)
	ids := []int32{1, 5, 3, 2, 4, 6, 7, 8}
	sess := newICBSessionStateFixture(b, g, arch, maxLen)
	if _, _, err := sess.ForwardCaptureHiddens(ids); err != nil {
		b.Fatalf("ForwardCaptureHiddens ICB warmup: %v", err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, _, err := sess.ForwardCaptureHiddens(ids); err != nil {
			b.Fatalf("ForwardCaptureHiddens ICB: %v", err)
		}
	}
}
