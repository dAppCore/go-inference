package mlservice

import (
	"testing"

	core "dappco.re/go"
)

func BenchmarkHelpers_fprintf(b *testing.B) {
	buf := core.NewBuffer(nil)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		buf.Reset()
		fprintf(buf, "  %-13s %-9s %9s %7s  loss=%.3f\n", "gemma4", "running", "120/500", "24.0%", 1.234)
	}
}

func BenchmarkHelpers_readAll(b *testing.B) {
	data := []byte("the quick brown fox jumps over the lazy dog, repeated payload body")
	r := core.NewBufferReader(data)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = r.Seek(0, 0)
		benchResult = readAll(r)
	}
}
