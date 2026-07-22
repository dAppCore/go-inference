// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"context"
	"testing"

	core "dappco.re/go"
	inference "dappco.re/go/inference"
)

// TestBatchedDenseFoldE4BPlain4BitNaN_Ugly is the live-stage bisect for the
// invalid-argmax (all-NaN verify hiddens) the uniform-4-bit lean E4B
// conversion (mlx-community/gemma-4-e4b-it-4bit) produces on the small-K
// batched fold under the REAL pair decode lane (recorded-ICB q8 session —
// a bare PrefillTokens session serves the same verify clean, which is why
// this drives the full LoadSpeculativePair + Generate path). The qat
// conversion (8-bit MLP overrides) is clean on the same lane, and every
// projection's qmm_t is byte-clean at these dims in isolation
// (TestQMMTAt31BDims e4b cases). Flips each fold-stage lever in turn and
// reports which stage's removal clears the failure. Env-gated on the two
// snapshots; skips without them.
func TestBatchedDenseFoldE4BPlain4BitNaN_Ugly(t *testing.T) {
	snap := core.Getenv("LTHN_E4B_PLAIN_SNAPSHOT")
	drafter := core.Getenv("LTHN_E4B_BF16_ASSISTANT")
	if core.Trim(snap) == "" || core.Trim(drafter) == "" {
		t.Skip("set LTHN_E4B_PLAIN_SNAPSHOT (gemma-4-e4b-it-4bit) and LTHN_E4B_BF16_ASSISTANT (gemma-4-e4b-it-assistant-bf16)")
	}
	restoreMTPFoldLevers(t)
	mtpVerifyFoldForced, mtpVerifyFoldDisabled = false, false

	run := func(label string) (failed bool) {
		t.Helper()
		tm, err := LoadSpeculativePair(snap, drafter, 0)
		if err != nil {
			t.Fatalf("%s: load pair: %v", label, err)
		}
		defer func() { _ = tm.Close() }()
		for range tm.Generate(context.Background(), "Count from 1 to 20, numbers only.",
			inference.WithMaxTokens(24), inference.WithTemperature(0)) {
		}
		if r := tm.Err(); r.Err() != nil {
			t.Logf("%s: FAILED: %v", label, r.Err())
			return true
		}
		t.Logf("%s: clean", label)
		return false
	}

	if !run("fold-on") {
		t.Log("fold-on is clean — the repro no longer fires on this build; keep as the regression pin")
		return
	}

	saveMLP, saveRope, saveEpi, saveMQ := batchedMLPFoldDisabledForTest, batchedRopeDisabledForTest, batchedEpilogueDisabledForTest, sdpaMultiQDisabledForTest
	saveVT, saveVS := verifyTailICBDisabledForTest, verifyStackICBDisabledForTest
	t.Cleanup(func() {
		batchedMLPFoldDisabledForTest, batchedRopeDisabledForTest, batchedEpilogueDisabledForTest, sdpaMultiQDisabledForTest = saveMLP, saveRope, saveEpi, saveMQ
		verifyTailICBDisabledForTest, verifyStackICBDisabledForTest = saveVT, saveVS
	})
	// The four fold prerequisites decline the WHOLE batched pass when flipped
	// (the q8 gate requires them), so they only separate fold-vs-interleave;
	// the two recorder levers cut INSIDE the fold: the #372 verify-tail replay
	// and the whole-stack verify ICB.
	levers := []struct {
		name string
		set  func(bool)
	}{
		{"verify-tail-icb-off", func(v bool) { verifyTailICBDisabledForTest = v }},
		{"verify-stack-icb-off", func(v bool) { verifyStackICBDisabledForTest = v }},
		{"mlp-fold-off", func(v bool) { batchedMLPFoldDisabledForTest = v }},
		{"rope-rows-off", func(v bool) { batchedRopeDisabledForTest = v }},
		{"epilogue-off", func(v bool) { batchedEpilogueDisabledForTest = v }},
		{"multiq-off", func(v bool) { sdpaMultiQDisabledForTest = v }},
	}
	for _, lv := range levers {
		lv.set(true)
		failed := run(lv.name)
		lv.set(false)
		if !failed {
			t.Logf("STAGE ISOLATED: %s clears the failure", lv.name)
		}
	}
}
