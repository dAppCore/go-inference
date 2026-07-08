// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"os"
	"testing"

	"github.com/tmc/apple/metal"
)

// TestSDPAPagedDepthAnatomy is the #356 micro-bench: it dispatches JUST the
// paged P1/P2 pair at the 26B's deep-scan shape (one global layer at 16K:
// 8 pages x 2048 rows, 8 KV heads, 16 query heads, headDim 256) and wall-clocks
// the GPU span directly (GPUStartTime/GPUEndTime, no profiler encoders, no
// model). The variant table separates the candidate latency sources the
// live profile cannot: per-dispatch launch overhead (8 pages vs the same rows
// in ONE page), the GQA-shared vs per-head kernel, and the split grain.
// Gated like every metallib test; LEM_SDPA_ANATOMY=1 to run.
func TestSDPAPagedDepthAnatomy(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	if os.Getenv("LEM_SDPA_ANATOMY") == "" {
		t.Skip("set LEM_SDPA_ANATOMY=1 to run the paged-SDPA depth anatomy bench")
	}
	if err := ensureInit(); err != nil {
		t.Fatalf("ensureInit: %v", err)
	}
	const (
		nHeads   = 16
		nKVHeads = 8
		headDim  = 256
		rows     = 16384
		reps     = 64
	)
	kvDim := nKVHeads * headDim
	scale := float32(0.0625)

	newPages := func(pageRows int) (kp, vp []metal.MTLBuffer, lens, heads, seqs []int) {
		n := (rows + pageRows - 1) / pageRows
		for p := 0; p < n; p++ {
			r := min(pageRows, rows-p*pageRows)
			kb := scratchBF16(r * kvDim)
			vb := scratchBF16(r * kvDim)
			if kb == nil || vb == nil {
				t.Fatalf("page alloc failed")
			}
			kp = append(kp, kb)
			vp = append(vp, vb)
			lens = append(lens, r)
			heads = append(heads, headDim) // head-major within a row: [row][kvh][dim]
			seqs = append(seqs, kvDim)
			_ = p
		}
		return
	}
	q := scratchBF16(nHeads * headDim)
	out := scratchBF16(nHeads * headDim)
	scratch, err := newSDPAPagedDecodeScratch(nHeads, headDim)
	if err != nil {
		t.Fatalf("scratch: %v", err)
	}

	run := func(name string, pageRows, splitOverride int, forcePerHead bool) {
		kp, vp, lens, heads, seqs := newPages(pageRows)
		saveOverride := sdpaPagedSplitRowsOverride
		sdpaPagedSplitRowsOverride = splitOverride
		defer func() { sdpaPagedSplitRowsOverride = saveOverride }()

		plan, perr := buildSDPAPagedDecodePlan(q, kp, vp, lens, heads, seqs, heads, seqs, out, scratch, nHeads, nKVHeads, headDim, scale)
		if perr != nil {
			t.Fatalf("%s: plan: %v", name, perr)
		}
		if forcePerHead && plan.gqaShared {
			pso, gerr := sdpaPagedP1Pipeline()
			if gerr != nil {
				t.Fatalf("%s: per-head pso: %v", name, gerr)
			}
			plan.p1PSO, plan.gqaShared = pso, false
		}
		// warmup + measured pass, all reps in ONE command buffer: the GPU span
		// covers exactly the dispatch stream the live decode would issue.
		for pass := 0; pass < 2; pass++ {
			cb := commandBufferFast(queue)
			enc := computeCommandEncoderFast(cb)
			for r := 0; r < reps; r++ {
				plan.emitP1s(enc)
				plan.emitP2(enc)
			}
			endEncodingFast(enc)
			commitCommandBufferFast(cb)
			waitUntilCompletedFast(cb)
			if pass == 1 {
				gpuMs := (cb.GPUEndTime() - cb.GPUStartTime()) * 1e3 / float64(reps)
				splits := 0
				for _, l := range lens {
					if l > 0 {
						splits += (l + plan.splitRows - 1) / plan.splitRows
					}
				}
				tgs := splits * nHeads
				if plan.gqaShared {
					tgs = splits * nKVHeads
				}
				t.Logf("%-34s %7.3f ms/layer-scan  (pages=%d splitRows=%d cells=%d TGs=%d gqa=%v)",
					name, gpuMs, len(kp), plan.splitRows, plan.cellCount, tgs, plan.gqaShared)
			}
		}
	}

	run("gqa2 8x2048 (production)", 2048, 0, false)
	run("gqa2 1x16384 (dispatch probe)", rows, 0, false)
	run("perhead 8x2048 (2x traffic)", 2048, 0, true)
	run("perhead 1x16384", rows, 0, true)
	run("gqa2 8x2048 grain64", 2048, 64, false)
	run("gqa2 8x2048 grain256", 2048, 256, false)
	run("gqa2 1x16384 grain64", rows, 64, false)
}
