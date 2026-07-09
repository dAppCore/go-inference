// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"os"
	"testing"

	"github.com/tmc/apple/metal"
)

// TestSDPA2PassDepthAnatomy is the #365 ICB-LANE instrument: it dispatches JUST
// the linear-cache SDPA kernels the recorded ICB replays for dense models (MLX
// sdpa_vector single-pass, and the sdpa_vector_2pass_1/2 pair global layers use
// at depth) and wall-clocks the GPU span directly (GPUStartTime/EndTime — no
// profiler encoders, no model). The paged anatomy bench (#356) priced the PAGED
// lane; dense hot decode never runs those kernels, so the deep-context tax
// (E2B ICB 176 -> 151 tok/s, +0.94 ms/tok at 16K) needs its own pass-level
// numbers. The variant table separates the candidate costs the live profile
// cannot: pass-1's scan (achieved GB/s against the read-once K+V floor),
// pass-2's merge, and the single-pass kernel's serialisation for contrast.
// Gated like every metallib test; LEM_SDPA_ANATOMY=1 to run.
func TestSDPA2PassDepthAnatomy(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	if os.Getenv("LEM_SDPA_ANATOMY") == "" {
		t.Skip("set LEM_SDPA_ANATOMY=1 to run the ICB-lane SDPA depth anatomy bench")
	}
	if err := ensureInit(); err != nil {
		t.Fatalf("ensureInit: %v", err)
	}
	const reps = 64

	shapes := []struct {
		name             string
		nHeads, nKVHeads int
		headDim          int
	}{
		{"e2b-global  8q/1kv/256", 8, 1, 256},
		{"12b-global 16q/8kv/256", 16, 8, 256},
	}
	depths := []int{256, 1024, 4096, 16384}

	for _, sh := range shapes {
		kvDim := sh.nKVHeads * sh.headDim
		q := scratchBF16(sh.nHeads * sh.headDim)
		out := scratchBF16(sh.nHeads * sh.headDim)
		scale := float32(0.0625)
		t.Logf("=== %s — ms/scan over %d reps (one cb, GPU span) ===", sh.name, reps)
		for _, n := range depths {
			k := scratchBF16(n * kvDim)
			v := scratchBF16(n * kvDim)
			if q == nil || out == nil || k == nil || v == nil {
				t.Fatalf("buffer alloc failed at n=%d", n)
			}
			blocks := int(sdpa2PassBlocks(n, sh.nKVHeads))
			partials := scratchF32(sh.nHeads * blocks * sh.headDim)
			sums := scratchF32(sh.nHeads * blocks)
			maxs := scratchF32(sh.nHeads * blocks)
			// seq-major cache [seq][kvh][dim]: head stride = headDim, seq stride = kvDim
			kh, ks := int64(sh.headDim), int64(kvDim)

			pso1, err := sdpaVector2Pass1PipelineForHeadDim(sh.headDim, int32(blocks))
			if err != nil {
				t.Fatalf("2pass1 pso: %v", err)
			}
			pso2, err := sdpaVector2Pass2PipelineForHeadDim(sh.headDim)
			if err != nil {
				t.Fatalf("2pass2 pso: %v", err)
			}
			psoSingle, err := sdpaVectorPipelineForHeadDim(sh.headDim)
			if err != nil {
				t.Fatalf("single pso: %v", err)
			}

			// read-once floor bytes: K+V streamed once per token (bf16)
			scanBytes := float64(n) * float64(kvDim) * 2 * 2

			time1 := func(name string, emit func(enc metal.MTLComputeCommandEncoder)) {
				var gpuMs float64
				for pass := 0; pass < 2; pass++ { // warmup, then measured
					cb := commandBufferFast(queue)
					enc := computeCommandEncoderFast(cb)
					for r := 0; r < reps; r++ {
						emit(metal.MTLComputeCommandEncoder(enc))
					}
					endEncodingFast(enc)
					commitCommandBufferFast(cb)
					waitUntilCompletedFast(cb)
					if pass == 1 {
						gpuMs = (cb.GPUEndTime() - cb.GPUStartTime()) * 1e3 / float64(reps)
					}
				}
				gbs := scanBytes / (gpuMs * 1e-3) / 1e9
				t.Logf("  n=%5d %-14s %8.4f ms  (%6.1f GB/s of read-once K+V, blocks=%d)",
					n, name, gpuMs, gbs, blocks)
			}

			time1("2pass pair", func(enc metal.MTLComputeCommandEncoder) {
				sink := encSink{enc}
				emitSDPA2Pass1(sink, pso1, q, k, v, partials, sums, maxs, 0, 1, sh.nHeads, sh.nKVHeads, n, blocks, kh, ks, kh, ks, scale)
				emitSDPA2Pass2(sink, pso2, partials, sums, maxs, out, 1, sh.nHeads, blocks)
			})
			time1("2pass P1 only", func(enc metal.MTLComputeCommandEncoder) {
				emitSDPA2Pass1(encSink{enc}, pso1, q, k, v, partials, sums, maxs, 0, 1, sh.nHeads, sh.nKVHeads, n, blocks, kh, ks, kh, ks, scale)
			})
			time1("2pass P2 only", func(enc metal.MTLComputeCommandEncoder) {
				emitSDPA2Pass2(encSink{enc}, pso2, partials, sums, maxs, out, 1, sh.nHeads, blocks)
			})
			time1("single-pass", func(enc metal.MTLComputeCommandEncoder) {
				emitSDPA(encSink{enc}, psoSingle, q, k, v, out, 0, nil, sh.nHeads, sh.nKVHeads, n, kh, ks, kh, ks, scale)
			})
		}
	}
}

// TestSDPA2PassBlocksSweep prices the blocks ladder for the OCCUPANCY-starved
// shape the depth anatomy exposed: E2B's single KV head gives the 2-pass grid
// only (1 x blocks) threadgroups, and at the ladder's 128 blocks pass 1 reaches
// ~286 GB/s where the 12B shape (8 KV heads x 128 blocks) saturates ~830. The
// sweep asks whether fanning blocks wider (the only axis this shape HAS) buys
// the scan back, and what the pass-2 merge pays for it — the data for a ladder
// that keys on kvLen x nKVHeads instead of kvLen alone.
// LEM_SDPA_ANATOMY=1 to run.
func TestSDPA2PassBlocksSweep(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	if os.Getenv("LEM_SDPA_ANATOMY") == "" {
		t.Skip("set LEM_SDPA_ANATOMY=1 to run the 2-pass blocks sweep")
	}
	if err := ensureInit(); err != nil {
		t.Fatalf("ensureInit: %v", err)
	}
	const (
		nHeads   = 8
		nKVHeads = 1
		headDim  = 256
		reps     = 64
	)
	kvDim := nKVHeads * headDim
	q := scratchBF16(nHeads * headDim)
	out := scratchBF16(nHeads * headDim)
	scale := float32(0.0625)
	kh, ks := int64(headDim), int64(kvDim)

	for _, n := range []int{4096, 16384, 32768} {
		k := scratchBF16(n * kvDim)
		v := scratchBF16(n * kvDim)
		if q == nil || out == nil || k == nil || v == nil {
			t.Fatalf("buffer alloc failed at n=%d", n)
		}
		scanBytes := float64(n) * float64(kvDim) * 2 * 2
		t.Logf("=== e2b-global 8q/1kv/256 n=%d — blocks sweep (pair ms, %d reps) ===", n, reps)
		for _, blocks := range []int{64, 128, 256, 512, 1024} {
			pso1, err := sdpaVector2Pass1PipelineForHeadDim(headDim, int32(blocks))
			if err != nil {
				t.Fatalf("2pass1 pso blocks=%d: %v", blocks, err)
			}
			pso2, err := sdpaVector2Pass2PipelineForHeadDim(headDim)
			if err != nil {
				t.Fatalf("2pass2 pso: %v", err)
			}
			partials := scratchF32(nHeads * blocks * headDim)
			sums := scratchF32(nHeads * blocks)
			maxs := scratchF32(nHeads * blocks)
			var pairMs, p1Ms float64
			for pass := 0; pass < 2; pass++ {
				cb := commandBufferFast(queue)
				enc := computeCommandEncoderFast(cb)
				for r := 0; r < reps; r++ {
					sink := encSink{metal.MTLComputeCommandEncoder(enc)}
					emitSDPA2Pass1(sink, pso1, q, k, v, partials, sums, maxs, 0, 1, nHeads, nKVHeads, n, blocks, kh, ks, kh, ks, scale)
					emitSDPA2Pass2(sink, pso2, partials, sums, maxs, out, 1, nHeads, blocks)
				}
				endEncodingFast(enc)
				commitCommandBufferFast(cb)
				waitUntilCompletedFast(cb)
				if pass == 1 {
					pairMs = (cb.GPUEndTime() - cb.GPUStartTime()) * 1e3 / float64(reps)
				}
			}
			for pass := 0; pass < 2; pass++ {
				cb := commandBufferFast(queue)
				enc := computeCommandEncoderFast(cb)
				for r := 0; r < reps; r++ {
					emitSDPA2Pass1(encSink{metal.MTLComputeCommandEncoder(enc)}, pso1, q, k, v, partials, sums, maxs, 0, 1, nHeads, nKVHeads, n, blocks, kh, ks, kh, ks, scale)
				}
				endEncodingFast(enc)
				commitCommandBufferFast(cb)
				waitUntilCompletedFast(cb)
				if pass == 1 {
					p1Ms = (cb.GPUEndTime() - cb.GPUStartTime()) * 1e3 / float64(reps)
				}
			}
			t.Logf("  blocks=%4d  pair %8.4f ms  P1 %8.4f ms (%6.1f GB/s)  P2 %8.4f ms",
				blocks, pairMs, p1Ms, scanBytes/(p1Ms*1e-3)/1e9, pairMs-p1Ms)
		}
	}
}
