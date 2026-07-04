// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"testing"
	"time"

	"github.com/tmc/apple/metal"
)

// TestHeadQMVTiming isolates the head lm_head qmv (the 17.7%-of-wall piece): a 4-bit affine_qmv_fast over
// [262144 × 1536], resident weights, timed per call. It's memory-bound — ~251 MB of weights+scales+biases
// read per call — so the achieved GB/s vs the M3 Ultra's ~819 GB/s peak says whether the kernel is at the
// roofline (no win — it's mlx's own affine_qmv_fast kernel) or leaving bandwidth on the table (a dispatch win).
func TestHeadQMVTiming(t *testing.T) {
	if err := ensureInit(); err != nil {
		t.Fatal(err)
	}
	const outDim, inDim, groupSize, bits = 262144, 1536, 32, 4
	groups := inDim / groupSize
	wq := make([]byte, outDim*inDim*bits/8)        // 201 MB packed 4-bit
	scales := make([]byte, outDim*groups*bf16Size) // 25 MB
	biases := make([]byte, outDim*groups*bf16Size) // 25 MB
	x := make([]byte, inDim*bf16Size)
	bytesRead := float64(len(wq) + len(scales) + len(biases))
	var per time.Duration
	withAutoreleasePool(func() {
		wBuf, sBuf, bBuf := sharedBytes(wq), sharedBytes(scales), sharedBytes(biases)
		xBuf := sharedBytes(x)
		outBuf := device.NewBufferWithLengthOptions(uint(outDim*bf16Size), metal.MTLResourceStorageModeShared)
		run := func() {
			cb := queue.CommandBuffer()
			enc := cb.ComputeCommandEncoder()
			if err := encQMVBF16(enc, wBuf, sBuf, bBuf, xBuf, outBuf, 0, 0, 0, 0, outDim, inDim, groupSize, bits); err != nil {
				t.Fatal(err)
			}
			enc.EndEncoding()
			cb.Commit()
			cb.WaitUntilCompleted()
		}
		for i := 0; i < 20; i++ { // warmup
			run()
		}
		const N = 100
		start := time.Now()
		for i := 0; i < N; i++ {
			run()
		}
		per = time.Since(start) / N
	})
	t.Logf("head lm_head qmv [%d×%d] 4-bit: %v/call", outDim, inDim, per)
	t.Logf("  reads %.0f MB → ideal @819 GB/s = %.2fms; achieved %.0f GB/s (%.0f%% of peak)",
		bytesRead/1e6, bytesRead/819e9*1000, bytesRead/per.Seconds()/1e9, 100*bytesRead/per.Seconds()/819e9)
}
