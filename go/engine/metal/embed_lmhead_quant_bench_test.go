// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64 && metal_runtime

package native

import (
	"os"
	"syscall"
	"testing"
)

// maxRSSBytes is the process resident high-water mark (darwin: bytes). The per-token
// generation balloon is METAL device memory, invisible to -benchmem's Go-heap counters,
// so AX-11 measures it here: a per-call device-buffer re-allocation that is never freed
// shows as Maxrss growing ~linearly with iterations (rss-grow-B/op ≈ the weight size);
// a clean path keeps it flat (≈ one call's transient ÷ N).
func maxRSSBytes() int64 {
	var ru syscall.Rusage
	_ = syscall.Getrusage(syscall.RUSAGE_SELF, &ru)
	return int64(ru.Maxrss)
}

// BenchmarkLMHeadQuant measures the per-token cost of the quantised LM head — gemma4's
// output projection, run once per generated token over the (tied) [vocab × dModel] 4-bit
// embedding. This is the serve hot path the memory balloon was observed on. The
// rss-grow-B/op metric is the tell: if LMHeadQuant re-uploads the packed weight into a
// fresh Metal buffer every call and it isn't released, rss-grow-B/op ≈ the packed size.
func BenchmarkLMHeadQuant(b *testing.B) {
	if os.Getenv(MetallibPathEnv) == "" {
		b.Skip("metallib not set")
	}
	const vocab, dModel, groupSize, bits = 32768, 2048, 64, 4
	packedBytes := vocab * dModel * bits / 8 // 4-bit packed weight (~33 MB here)
	packed := make([]byte, packedBytes)
	for i := range packed {
		packed[i] = byte(i*7 + 1)
	}
	sb := make([]byte, vocab*(dModel/groupSize)*bf16Size)
	for i := range sb {
		sb[i] = byte(i*3 + 2)
	}
	scales := sb
	biases := append([]byte(nil), sb...)
	finalNorm := bf16ConstBytes(dModel, 1.0)
	hidden := bf16ConstBytes(dModel, 0.01)
	b.Logf("packed weight = %.1f MB resident candidate", float64(packedBytes)/(1<<20))

	b.ResetTimer()
	rss0 := maxRSSBytes()
	for i := 0; i < b.N; i++ {
		if _, err := LMHeadQuant(hidden, finalNorm, packed, scales, biases, dModel, vocab, groupSize, bits, 1e-6, 0); err != nil {
			b.Fatal(err)
		}
	}
	b.StopTimer()
	b.ReportMetric(float64(maxRSSBytes()-rss0)/float64(b.N), "rss-grow-B/op")
}
