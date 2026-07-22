// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64 && metal_runtime

package native

import (
	"os"
	"testing"
	"time"
)

// TestDecodeTokPerSecVsDepthRealE2B quantifies the #365 deep-context tax: decode
// tok/s as a function of context depth. Every decoded token scans the whole KV
// cache, so tok/s falls as context grows — this measures BY HOW MUCH on real
// e2b-4bit — the felt curve ("short ctx ~180, 8K ~120"). It's the receipt
// that says whether the KV-bytes lever (q8/q4 KV) is worth building, and the
// baseline any KV-quant A/B is measured against. Also logs the achieved position
// per depth (a prior probe saw the cache cap short of the prompt length).
//
//	LEM_REAL_E2B=1 MLX_METALLIB_PATH=... go test -run TestDecodeTokPerSecVsDepthRealE2B -v ./engine/metal/
func TestDecodeTokPerSecVsDepthRealE2B(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	if os.Getenv("LEM_REAL_E2B") == "" {
		t.Skip("set LEM_REAL_E2B=1 to run the real e2b-4bit decode-tok/s-vs-depth curve (loads ~2.7GB)")
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

	prevICB := icbDisabledForTest
	defer func() { icbDisabledForTest = prevICB }()

	depths := []int{256, 1024, 4096, 8192, 16384}
	const maxLen = 16384 + 256
	const N = 48 // timed decode tokens per depth

	// A/B the two decode paths: ICB replay (production for dense E2B) vs stepToken
	// re-encode. Same KV scan; if ICB degrades with depth but stepToken doesn't, the
	// tax is in the replay machinery (re-encode/host), not the fundamental scan.
	t.Logf("=== #365 decode tok/s vs context depth — real e2b-4bit (bf16 KV) ===")
	for _, icbOff := range []bool{false, true} {
		icbDisabledForTest = icbOff
		mode := "ICB (production)"
		if icbOff {
			mode = "stepToken"
		}
		t.Logf("--- %s path ---", mode)
		for _, d := range depths {
			sess, serr := newArchQuantSessionShards(qm, lm.Arch, maxLen, sb)
			if serr != nil {
				t.Fatalf("session (depth %d): %v", d, serr)
			}
			prompt := make([]int32, d)
			prompt[0] = 2
			for i := 1; i < d; i++ {
				prompt[i] = int32(100 + (i*131)%3000) // in-vocab, diverse
			}
			if perr := sess.PrefillTokens(prompt); perr != nil {
				t.Fatalf("prefill %d: %v", d, perr)
			}
			if _, gerr := sess.GenerateFromCache(4, -1); gerr != nil { // warmup, untimed
				t.Fatalf("warmup (depth %d): %v", d, gerr)
			}
			t0 := time.Now()
			if _, gerr := sess.GenerateFromCache(N, -1); gerr != nil {
				t.Fatalf("timed decode (depth %d): %v", d, gerr)
			}
			wall := time.Since(t0)
			t.Logf("  depth %5d: %5.1f tok/s  (%d tok / %.2fs)", d, float64(N)/wall.Seconds(), N, wall.Seconds())
		}
	}
	t.Logf("  => compare the two slopes: a steeper ICB fall = replay-machinery tax; equal falls = the scan itself")
}
