// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64 && metal_runtime

package native

import (
	"os"
	"testing"

	"dappco.re/go/inference/model"
)

// TestKVScanBandwidthFloorRealE2B computes the #365 KV-scan bandwidth FLOOR from real
// e2b geometry (no decode) and prints it beside the measured per-token cost, to settle
// whether the deep-context tax is fundamental bytes (→ quant is the honest lever) or the
// 2-pass sdpa running OVER its floor (→ fix the kernel: it may degrade more than it
// should). Per token a global layer scans its whole K+V; sliding layers scan only the
// window. floor = bytes / peak-bandwidth; compare its 256→16K delta to the measured
// ICB delta (~0.94 ms/token, 176→151 tok/s).
//
//	LEM_REAL_E2B=1 MLX_METALLIB_PATH=... go test -run TestKVScanBandwidthFloorRealE2B -v ./engine/metal/
func TestKVScanBandwidthFloorRealE2B(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	if os.Getenv("LEM_REAL_E2B") == "" {
		t.Skip("set LEM_REAL_E2B=1 to run the real e2b KV-scan floor (loads ~2.7GB)")
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
	const maxLen = 16640
	sess, err := newArchQuantSessionShards(qm, lm.Arch, maxLen, sb)
	if err != nil {
		t.Fatalf("newArchQuantSessionShards: %v", err)
	}

	sw := sess.state.slidingWindow
	nGlobal, nSliding := 0, 0
	for li := range sess.state.specs {
		if sess.state.specs[li].Attention == model.GlobalAttention {
			nGlobal++
		} else {
			nSliding++
		}
	}
	scanBytes := func(d int) (total, global int64) {
		for li := range sess.state.specs {
			sp := sess.state.specs[li]
			eff := d
			if sp.Attention != model.GlobalAttention && sw > 0 && sw < d {
				eff = sw
			}
			b := int64(eff) * int64(sp.KVHeads) * int64(sp.HeadDim) * 2 * 2 // bf16, K+V
			total += b
			if sp.Attention == model.GlobalAttention {
				global += b
			}
		}
		return
	}

	const bw = 800e9 // M3 Ultra unified memory ~800 GB/s peak
	t.Logf("=== #365 KV-scan bandwidth floor — e2b: %d layers (%d global, %d sliding, window %d) ===",
		len(sess.state.specs), nGlobal, nSliding, sw)
	var floor256, floor16k float64
	for _, d := range []int{256, 16384} {
		tot, glob := scanBytes(d)
		ms := float64(tot) / bw * 1e3
		if d == 256 {
			floor256 = ms
		} else {
			floor16k = ms
		}
		t.Logf("  depth %5d: KV scan %6.1f MB/tok (global %5.1f MB) -> floor %.3f ms @ %.0f GB/s",
			d, float64(tot)/1e6, float64(glob)/1e6, ms, bw/1e9)
	}
	t.Logf("  floor delta 256->16K: %.3f ms/tok  vs  MEASURED ICB delta ~0.94 ms/tok", floor16k-floor256)
	t.Logf("  => floor≈measured: bandwidth-bound (quant is the lever). measured>>floor: kernel over floor (fix the sdpa)")
}
