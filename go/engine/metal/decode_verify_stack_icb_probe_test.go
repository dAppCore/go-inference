// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64 && metal_runtime

package native

import (
	"os"
	"testing"
	"time"

	"dappco.re/go/inference/internal/enginegate"
)

// TestRealE2BVerifyStackFwdProbe is an INSTRUMENT, not a gate: it times the
// verify fold's forward wall at a fixed K on the real checkpoint — the live
// fold, the recording pass and the recorded replays — because the shipped
// LTHN_MTP_DIAG verify-rows log caps at the first three rounds, which under
// the lane are all recording passes. LTHN_VERIFY_STACK_PROBE=1 arms it.
func TestRealE2BVerifyStackFwdProbe(t *testing.T) {
	if os.Getenv("LTHN_VERIFY_STACK_PROBE") == "" {
		t.Skip("probe instrument — set LTHN_VERIFY_STACK_PROBE=1 to run")
	}
	requireNativeRuntime(t)
	targetDir := enginegate.HFModelPath(t, "mlx-community/gemma-4-e2b-it-4bit")

	const k, blocks = 6, 12
	run := func(disable bool) []time.Duration {
		t.Helper()
		verifyStackICBDisabledForTest = disable
		defer func() { verifyStackICBDisabledForTest = false }()
		target, err := LoadDir(targetDir, 4096)
		if err != nil {
			t.Fatalf("LoadDir: %v", err)
		}
		defer func() { _ = target.Close() }()
		prompt := realE2BAssistantPrompt(t, targetDir)
		if err := target.prepareAssistantPrompt(prompt); err != nil {
			t.Fatalf("prepareAssistantPrompt: %v", err)
		}
		ids := make([]int32, k)
		for i := range ids {
			ids[i] = int32(100 + i)
		}
		target.state.verifyFoldSmallK = true
		defer func() { target.state.verifyFoldSmallK = false }()
		var walls []time.Duration
		for b := 0; b < blocks; b++ {
			t0 := time.Now()
			_, ok, err := target.verifyBatchedHiddens(ids)
			if err != nil {
				t.Fatalf("block %d: verifyBatchedHiddens: %v", b, err)
			}
			if !ok {
				t.Fatalf("block %d: batched verify declined", b)
			}
			walls = append(walls, time.Since(t0))
			target.pos += k
		}
		return walls
	}

	base := verifyStackReplays.Load()
	lane := run(false)
	replays := verifyStackReplays.Load() - base
	live := run(true)
	for b := 0; b < blocks; b++ {
		t.Logf("block %2d  lane=%6.2fms  live=%6.2fms", b, float64(lane[b].Microseconds())/1000, float64(live[b].Microseconds())/1000)
	}
	t.Logf("lane replays engaged: %d/%d blocks (block 0 records)", replays, blocks)
}
