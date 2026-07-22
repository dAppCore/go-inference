// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"context"
	"testing"

	core "dappco.re/go"
	inference "dappco.re/go/inference"
)

// TestBatchedDenseFold12BPairGreedyParity_Ugly is #73's live-stage bisect:
// the 12B pair (gemma-4-12B-it-4bit + bf16 assistant) diverges from the
// per-row exact verify at the FIRST boundary under the batched fold (greedy,
// temp 0), where E2B/E4B/26B hold byte parity — and the same drift rejects
// the drafter (25% acceptance, low-accept bail). The per-row lane
// (LTHN_MTP_VERIFY_FOLD=0 live) is the parity reference; each fold-stage
// lever then runs in turn — the lever whose removal restores the reference
// token stream names the diverging stage. Env-gated on the two snapshots.
func TestBatchedDenseFold12BPairGreedyParity_Ugly(t *testing.T) {
	snap := core.Getenv("LTHN_12B_PLAIN_SNAPSHOT")
	drafter := core.Getenv("LTHN_12B_BF16_ASSISTANT")
	if core.Trim(snap) == "" || core.Trim(drafter) == "" {
		t.Skip("set LTHN_12B_PLAIN_SNAPSHOT (gemma-4-12B-it-4bit) and LTHN_12B_BF16_ASSISTANT (gemma-4-12B-it-assistant-bf16)")
	}
	restoreMTPFoldLevers(t)

	const prompt = "Count the integers from 1 to 800, separated by single spaces."
	run := func(label string) []int32 {
		t.Helper()
		tm, err := LoadSpeculativePair(snap, drafter, 0)
		if err != nil {
			t.Fatalf("%s: load pair: %v", label, err)
		}
		defer func() { _ = tm.Close() }()
		var ids []int32
		for tok := range tm.Generate(context.Background(), prompt,
			inference.WithMaxTokens(48), inference.WithTemperature(0)) {
			ids = append(ids, tok.ID)
		}
		if r := tm.Err(); r.Err() != nil {
			t.Fatalf("%s: generate: %v", label, r.Err())
		}
		return ids
	}
	firstDiff := func(a, b []int32) int {
		n := min(len(a), len(b))
		for i := 0; i < n; i++ {
			if a[i] != b[i] {
				return i
			}
		}
		if len(a) != len(b) {
			return n
		}
		return -1
	}

	// reference: the per-row byte-exact verify lane.
	mtpVerifyFoldForced, mtpVerifyFoldDisabled = false, true
	ref := run("per-row-reference")

	mtpVerifyFoldForced, mtpVerifyFoldDisabled = false, false
	base := run("fold-on")
	if d := firstDiff(ref, base); d < 0 {
		t.Log("fold-on matches the per-row reference — the repro no longer fires; keep as the regression pin")
		return
	} else {
		t.Logf("fold-on diverges from the per-row reference at token %d", d)
	}

	saveMLP, saveRope, saveEpi, saveMQ := batchedMLPFoldDisabledForTest, batchedRopeDisabledForTest, batchedEpilogueDisabledForTest, sdpaMultiQDisabledForTest
	saveVT, saveVS := verifyTailICBDisabledForTest, verifyStackICBDisabledForTest
	t.Cleanup(func() {
		batchedMLPFoldDisabledForTest, batchedRopeDisabledForTest, batchedEpilogueDisabledForTest, sdpaMultiQDisabledForTest = saveMLP, saveRope, saveEpi, saveMQ
		verifyTailICBDisabledForTest, verifyStackICBDisabledForTest = saveVT, saveVS
	})
	levers := []struct {
		name string
		set  func(bool)
	}{
		{"verify-stack-icb-off", func(v bool) { verifyStackICBDisabledForTest = v }},
		{"multiq-off", func(v bool) { sdpaMultiQDisabledForTest = v }},
		{"rope-rows-off", func(v bool) { batchedRopeDisabledForTest = v }},
		{"epilogue-off", func(v bool) { batchedEpilogueDisabledForTest = v }},
		{"mlp-fold-off", func(v bool) { batchedMLPFoldDisabledForTest = v }},
	}
	for _, lv := range levers {
		lv.set(true)
		got := run(lv.name)
		lv.set(false)
		if d := firstDiff(ref, got); d < 0 {
			t.Logf("%s: PARITY RESTORED — this stage diverges on 12B", lv.name)
		} else {
			t.Logf("%s: still diverges at token %d", lv.name, d)
		}
	}
	t.Error("fold-on diverges from the per-row reference (see stage logs above) — #73 open until parity holds")
}
