// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"os"
	"testing"
	"time"
)

// TestRealMoE26BHostProfile decodes the real gemma-4-26B-A4B MoE checkpoint so a CPU profile can show where
// its per-token time goes. Dense e2b decode is GPU-bound (cgocall ≈ GPU wait, 99% gpu-busy); the MoE arch
// can't use the recorded-ICB path (the router top-k forces a host readback), so MoEBlockQuant orchestrates
// ~a dozen separately-host-synced Metal calls per layer per token. If this path is HOST-bound (cgocall a
// small fraction, native orchestration large), that's the reclaimable headroom the dense path doesn't have.
// Gated behind LEM_REAL_MOE (loads ~15 GB). Run with -cpuprofile to read the split.
func TestRealMoE26BHostProfile(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	if os.Getenv("LEM_REAL_MOE") == "" {
		t.Skip("set LEM_REAL_MOE=1 to run the real 26B-A4B MoE decode profile (loads ~15GB)")
	}
	dir := resolveMoE26BDir(t)
	const maxLen, warmup, N = 320, 4, 24

	lm, dm, err := loadRegistered(dir)
	if err != nil {
		t.Fatalf("loadRegistered: %v", err)
	}
	defer func() { _ = dm.Close() }()
	if !quantised(lm) {
		t.Fatalf("expected a quantised 26B checkpoint")
	}
	sb, err := buildShardBuffers(dm)
	if err != nil {
		t.Fatalf("buildShardBuffers: %v", err)
	}
	defer func() { _ = sb.Close() }()
	qm, err := loadedToQuant(lm, lm.Embed.GroupSize, lm.Embed.Bits)
	if err != nil {
		t.Fatalf("loadedToQuant: %v", err)
	}
	sess, err := newArchQuantSessionShards(qm, lm.Arch, maxLen, sb)
	if err != nil {
		t.Fatalf("newArchQuantSessionShards: %v", err)
	}

	prompt := []int32{2, 1000, 2500, 4000, 8000, 16000}
	if err := sess.PrefillTokens(prompt); err != nil {
		t.Fatalf("prefill: %v", err)
	}
	if _, err := sess.GenerateFromCache(warmup, -1); err != nil {
		t.Fatalf("warmup: %v", err)
	}
	t0 := time.Now()
	if _, err := sess.GenerateFromCache(N, -1); err != nil {
		t.Fatalf("generate: %v", err)
	}
	wall := time.Since(t0)
	t.Logf("real 26B-A4B MoE decode (tg%d): %.1f tok/s (%.2f ms/token) — run under -cpuprofile; low cgocall ⇒ host-bound",
		N, float64(N)/wall.Seconds(), wall.Seconds()*1000/float64(N))
}

func resolveMoE26BDir(t *testing.T) string {
	home := os.Getenv("HOME")
	base := home + "/.cache/huggingface/hub/models--mlx-community--gemma-4-26B-A4B-it-qat-4bit/snapshots"
	entries, err := os.ReadDir(base)
	if err != nil {
		t.Skipf("26B-A4B snapshot dir not found (%v)", err)
	}
	for _, e := range entries {
		if !e.IsDir() {
			continue
		}
		dir := base + "/" + e.Name()
		if _, serr := os.Stat(dir + "/config.json"); serr == nil {
			return dir
		}
	}
	t.Skip("no 26B-A4B snapshot with config.json")
	return ""
}
