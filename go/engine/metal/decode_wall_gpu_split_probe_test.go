// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64 && metal_runtime

package native

import (
	"os"
	"testing"
	"time"
)

// TestDecodeWallGPUSplitRealE2B decomposes the live-decode "fixed cost" the
// family board exposes (~130µs/layer residual on EVERY model — E2B pays it on
// 80% of its token, so its effective bandwidth reads 159GB/s on an ~800GB/s
// part). The residual conflates three different costs; this probe splits the
// first fork with the chained (serial submit+wait) ICB lane's GPU-span
// accounting: GPU-BUSY per token vs the host+sync GAP per token.
//   - gap dominates  -> the lever is submit/sync (pipeline depth, encode-bypass)
//   - busy dominates -> the lever is in-kernel (dispatch ramps + small-read
//     latency: E2B reads ~27MB/layer in ~20 pieces — sizes that cannot hide
//     DRAM latency at full bandwidth) -> the fusion lane
//
// The pipelined (production) lane is timed alongside as the anchor.
//
//	LEM_REAL_E2B=1 MLX_METALLIB_PATH=... go test -run TestDecodeWallGPUSplitRealE2B -v ./engine/metal/
func TestDecodeWallGPUSplitRealE2B(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	if os.Getenv("LEM_REAL_E2B") == "" {
		t.Skip("set LEM_REAL_E2B=1 to run the real e2b-4bit wall/GPU split probe (loads ~2.7GB)")
	}
	dir := resolveE2B4bitDir(t)
	lm, dm, err := loadRegistered(dir)
	if err != nil {
		t.Fatalf("loadRegistered: %v", err)
	}
	defer func() { _ = dm.Close() }()
	sb, err := buildShardBuffers(dm)
	if err != nil {
		t.Fatalf("buildShardBuffers: %v", err)
	}
	defer func() { _ = sb.Close() }()
	qm, err := loadedToQuant(lm, lm.Embed.GroupSize, lm.Embed.Bits)
	if err != nil {
		t.Fatalf("loadedToQuant: %v", err)
	}

	prompt := []int32{2, 106, 1645, 108, 21356, 603, 573, 6875, 576, 235248}
	const maxLen, warm, N = 2048, 8, 64

	oldPipe, oldTiming, oldSpan := pipelinedGPUDecodeEnabled, pieceTimingOn, chainedGPUSpanNs
	defer func() {
		pipelinedGPUDecodeEnabled, pieceTimingOn, chainedGPUSpanNs = oldPipe, oldTiming, oldSpan
	}()

	run := func(pipelined bool) (wallPerTok, gpuPerTok float64) {
		pipelinedGPUDecodeEnabled = pipelined
		sess, serr := newArchQuantSessionShards(qm, lm.Arch, maxLen, sb)
		if serr != nil {
			t.Fatalf("session (pipelined=%v): %v", pipelined, serr)
		}
		if _, gerr := sess.GenerateOneShot(prompt, warm, -1); gerr != nil {
			t.Fatalf("warmup (pipelined=%v): %v", pipelined, gerr)
		}
		pieceTimingOn = true
		chainedGPUSpanNs = 0
		t0 := time.Now()
		gen, gerr := sess.GenerateOneShot(prompt, N, -1)
		wall := time.Since(t0)
		pieceTimingOn = false
		if gerr != nil {
			t.Fatalf("generate (pipelined=%v): %v", pipelined, gerr)
		}
		if len(gen) < N/2 {
			t.Fatalf("generate (pipelined=%v) produced %d tokens, want ~%d", pipelined, len(gen), N)
		}
		return float64(wall.Microseconds()) / float64(len(gen)),
			float64(chainedGPUSpanNs) / 1e3 / float64(len(gen))
	}

	chainedWall, chainedGPU := run(false)
	pipeWall, _ := run(true)

	nLayers := len(lm.Arch.Layer)
	gap := chainedWall - chainedGPU
	t.Logf("=== e2b-4bit live decode wall/GPU split (%d layers, %d tokens) ===", nLayers, N)
	t.Logf("  chained lane:   wall %7.1f µs/tok · GPU-busy %7.1f µs (%2.0f%%) · host+sync gap %7.1f µs", chainedWall, chainedGPU, 100*chainedGPU/chainedWall, gap)
	t.Logf("  per layer:      GPU-busy %5.1f µs · gap %5.1f µs", chainedGPU/float64(nLayers), gap/float64(nLayers))
	t.Logf("  pipelined lane: wall %7.1f µs/tok (production ≈ %.0f tok/s) — hides %5.1f µs of the gap", pipeWall, 1e6/pipeWall, chainedWall-pipeWall)
	t.Logf("  achieved bandwidth during GPU-busy ≈ %.0f GB/s of ~800 (0.93GB/token weight+KV reads)", 0.93e6/chainedGPU)
}
