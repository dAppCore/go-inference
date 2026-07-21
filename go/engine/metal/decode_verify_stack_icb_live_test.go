// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64 && metal_runtime

package native

import (
	"os"
	"testing"

	"dappco.re/go/inference/internal/enginegate"
)

// TestRealE2BVerifyStackICBTokensMatchLive pins the whole-stack verify ICB on
// the real cached MTP pair: the SAME prompt generated with the lane on and
// with it force-disabled must produce the identical token stream (the verify
// fold is deterministic, so the recorded interior must be indistinguishable),
// and the engagement counter must prove the lane actually replayed — an
// unengaged parity proves nothing.
func TestRealE2BVerifyStackICBTokensMatchLive(t *testing.T) {
	if os.Getenv("LTHN_VERIFY_STACK_ICB") != "1" {
		t.Skip("the ENGINE's re-engagement bistability flips a near-tied token with the lane disabled in both arms (see verifyStackICBDisabled + TestRealE2BVerifyStackKVDiff) — set LTHN_VERIFY_STACK_ICB=1 to run; flips always-on when the engine flake is fixed and this holds under -count=10")
	}
	requireNativeRuntime(t)
	targetDir := enginegate.HFModelPath(t, "mlx-community/gemma-4-e2b-it-4bit")
	assistantDir := enginegate.HFModelPath(t, "mlx-community/gemma-4-E2B-it-assistant-bf16")

	run := func(disable bool) ([]int32, int64) {
		t.Helper()
		verifyStackICBDisabledForTest = disable
		defer func() { verifyStackICBDisabledForTest = false }()
		target, err := LoadDir(targetDir, 4096)
		if err != nil {
			t.Fatalf("LoadDir: %v", err)
		}
		defer func() { _ = target.Close() }()
		pair, err := LoadAssistantPairDirs(targetDir, assistantDir)
		if err != nil {
			t.Fatalf("LoadAssistantPairDirs: %v", err)
		}
		defer pair.Close()
		prompt := realE2BAssistantPrompt(t, targetDir)
		base := verifyStackReplays.Load()
		res, err := pair.GenerateFromSession(target, prompt, 96, -1, 4, nil)
		if err != nil {
			t.Fatalf("GenerateFromSession: %v", err)
		}
		return append([]int32(nil), res.Tokens...), verifyStackReplays.Load() - base
	}

	live, liveReplays := run(true)
	if liveReplays != 0 {
		t.Fatalf("force-disabled run replayed %d times — the kill lever leaks", liveReplays)
	}
	lane, laneReplays := run(false)
	if laneReplays == 0 {
		t.Fatal("whole-stack lane never replayed on the real pair — the parity below proves nothing")
	}
	if len(lane) != len(live) {
		t.Fatalf("token count %d != live %d", len(lane), len(live))
	}
	for i := range lane {
		if lane[i] != live[i] {
			t.Fatalf("token %d: lane %d != live %d — the recorded interior diverged", i, lane[i], live[i])
		}
	}
}
