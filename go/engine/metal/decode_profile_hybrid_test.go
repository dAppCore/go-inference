// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"os"
	"testing"
	"time"

	"dappco.re/go/inference/model"
)

// TestComposedDecodeProfileReal is the S0 instrument of the hybrid-recurrence campaign
// (docs/design-hybrid-recurrence.md): a bare multi-token decode loop over a REAL composed checkpoint,
// built to run under -cpuprofile so pprof splits the per-token wall between the host recurrence
// (deltanet.GatedDeltaRuleF32), the conv/norm/gate host loops, the quant matvec seams and the
// command-buffer waits — the time-domain complement to TestComposedDecodeRoundTripCensus's CB counts.
// It measures and reports; it changes nothing. Skips without LTHN_COMPOSED_PROFILE_MODEL so the suite
// never pays a 27B load; point it at any composed snapshot:
//
//	MLX_METALLIB_PATH=$PWD/build/dist/lib/mlx.metallib \
//	LTHN_COMPOSED_PROFILE_MODEL=~/.cache/huggingface/hub/models--mlx-community--Qwen3.6-27B-4bit/snapshots/<sha> \
//	go test -tags metal_runtime ./engine/metal -run TestComposedDecodeProfileReal -v -cpuprofile cpu.out
func TestComposedDecodeProfileReal(t *testing.T) {
	dir := os.Getenv("LTHN_COMPOSED_PROFILE_MODEL")
	if dir == "" {
		t.Skip("LTHN_COMPOSED_PROFILE_MODEL not set — composed decode profile instrument")
	}
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set — composed decode profile instrument")
	}
	if err := ensureInit(); err != nil {
		t.Skipf("Metal runtime unavailable — composed decode profile instrument: %v", err)
	}
	tm, err := LoadTokenModelDir(dir, 1024)
	if err != nil {
		t.Fatalf("LoadTokenModelDir(%s): %v", dir, err)
	}
	sm, ok := tm.(model.SessionModel)
	if !ok {
		t.Fatalf("loaded model is %T, want model.SessionModel", tm)
	}
	sess, err := sm.OpenSession()
	if err != nil {
		t.Fatalf("OpenSession: %v", err)
	}
	bp, ok := sess.(model.BatchPrefillStepper)
	if !ok {
		t.Fatalf("session %T lacks BatchPrefillStepper", sess)
	}

	const promptLen = 8
	prompt := make([]int32, promptLen)
	for i := range prompt {
		prompt[i] = int32(16 + (i*37)%2048)
	}
	embs := make([][]byte, len(prompt))
	for i, id := range prompt {
		if embs[i], err = tm.Embed(id); err != nil {
			t.Fatalf("Embed(%d): %v", id, err)
		}
	}
	if _, err := bp.PrefillBatch(embs); err != nil {
		t.Fatalf("PrefillBatch: %v", err)
	}

	// One untimed warm token (first-step lazies: pools, pipelines), then the timed loop. 24 tokens is
	// enough profile mass at ~300 ms/token on the 27B while keeping the run under 10 s.
	if _, err := sess.Step(embs[0]); err != nil {
		t.Fatalf("warm Step: %v", err)
	}
	const tokens = 24
	start := time.Now()
	for i := 0; i < tokens; i++ {
		if _, err := sess.Step(embs[i%promptLen]); err != nil {
			t.Fatalf("decode Step %d: %v", i, err)
		}
	}
	wall := time.Since(start)
	t.Logf("COMPOSED DECODE PROFILE (%s): %d tokens in %s = %.2f ms/token (%.2f tok/s)",
		dir, tokens, wall, float64(wall.Milliseconds())/tokens, float64(tokens)/wall.Seconds())
}
