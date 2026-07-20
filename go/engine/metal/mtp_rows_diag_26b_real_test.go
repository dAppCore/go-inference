// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"testing"

	"dappco.re/go/inference/internal/enginegate"
)

// mtp_rows_diag_26b_real_test.go — the #53 receipt this diagnostic exists to produce. Two live
// hypotheses stood after the byte-exact layer-major driver (mtp_rows_driver.go) landed and the
// real 26B pair STILL measured -43% vs plain (81.5 vs 142.6 tok/s):
//
//	A: the driver DECLINES in the live serve session for a reason the unit fixtures miss (the
//	   eligibility walk in mtpRowsDriverDeclineReason + the verifyRowsMoEBatchedHiddens entry
//	   checks — nothing logged WHICH condition fired, until now);
//	B: the driver ENGAGES but grouped-by-expert batching degenerates on real routing — K·topK
//	   pairs scatter across many experts, mostly singleton groups, so the batched gather/scatter
//	   pays overhead with no matching per-expert weight-read saving (the unit receipts only proved
//	   grouping on a tiny-expert-count fixture).
//
// WHY THIS DRIVES verifyAssistantDraftRows DIRECTLY RATHER THAN THROUGH inference.TextModel.Chat/
// Generate: a first attempt drove the full production pair loop (LoadSpeculativePair wired to
// inference.TextModel.Chat/Generate — speculative_model.go's speculate(), the SAME seam
// `lem serve --draft` uses) and observed ZERO verify rounds over 32-48 real emitted tokens across
// two different prompts. The reason is upstream of #53 entirely:
// verifyDraftBlockFromSessionWithSuppress (assistant_load.go) rejects the WHOLE drafted block the
// moment draftTokens[0] != the target's boundary greedy, WITHOUT ever calling
// verifyAssistantDraftRows — and the tiny qat-assistant drafter's first proposed token essentially
// never matched the 26B target's on these prompts (naturalD0Mismatches below counts this live).
// That is a real, separate finding, but it is the MTP accept-economy's problem, not the #53
// driver's: verify's engage/decline and expert-grouping behaviour depend ONLY on the target's own
// state and the K drafted rows' TOKEN IDENTITY (embed -> attention -> MoE), never on which model
// proposed those ids or whether they will later be judged "accepted" (mtp.go's own contract: "a
// wrong draft token merely falls back to the token the target would have emitted anyway" — verify
// computes the identical bytes regardless). So this test still loads and exercises the REAL
// drafter every round (pair.DraftBlockFromSession — proving the real pair attaches and really
// proposes real tokens on these two real checkpoints), but calls target.verifyAssistantDraftRows
// directly with those real proposed tokens instead of going through the natural loop's
// d0-must-match gate — guaranteeing the #53 seam runs every round, on real weights, real routing,
// real 26B geometry, independent of the drafter's hit rate. Each round's target boundary seed is
// the PREVIOUS round's verify output (rows[K-1], the target's own greedy after the whole block) —
// both sessions' caches advance by K positions every round, in lockstep, exactly like a real
// always-accepted MTP round.
//
// Skips cleanly when either checkpoint is absent from the local Hugging Face cache
// (enginegate.HFModelPath's house pattern), so CI stays green off this machine. The pair load is
// ~15GB and several verify rounds at 26B-A4B scale take minutes, not seconds — deliberately
// excluded from a bare `go test ./...` by requiring -run (see real_checkpoint_gpu_test.go /
// moe_26b_real_test.go for the same shape). THE PRINTED NUMBERS ARE THE DELIVERABLE — they decide
// hypothesis A vs B; this test does not fix routing or perf, only instruments and reports.
func TestRealCheckpointGPU_26BMTPRowsDiag_Good(t *testing.T) {
	targetDir := enginegate.HFModelPath(t, "mlx-community/gemma-4-26b-a4b-it-4bit")
	draftDir := enginegate.HFModelPath(t, "mlx-community/gemma-4-26B-A4B-it-qat-assistant-4bit")

	const (
		k      = 5 // draftBlock — matches LoadSpeculativePair's shipped default
		rounds = 6 // "a few greedy verify rounds", per the brief
	)
	// moe_26b_real_test.go's own proven 26B-A4B fixture prompt (resolveMoE26BDir's sibling test
	// uses the identical ids) — real, in-vocab, already known to prefill and decode cleanly on
	// this checkpoint family.
	prompt := []int32{2, 1000, 2500, 4000, 8000, 16000}
	maxLen := len(prompt) + rounds*k + 64

	target, err := LoadDir(targetDir, maxLen)
	if err != nil {
		t.Fatalf("LoadDir(target): %v", err)
	}
	defer func() { _ = target.Close() }()

	pair, err := LoadAssistantPairDirs(targetDir, draftDir)
	if err != nil {
		t.Fatalf("LoadAssistantPairDirs: %v", err)
	}
	defer func() { _ = pair.Close() }()
	t.Logf("real pair attached: target=%s draft=%s method=%v", targetDir, draftDir, pair.Method())

	if err := target.PrepareAssistantPrompt(prompt); err != nil {
		t.Fatalf("PrepareAssistantPrompt: %v", err)
	}
	boundaryLogits, err := target.BoundaryLogits()
	if err != nil {
		t.Fatalf("BoundaryLogits: %v", err)
	}
	lastToken, err := greedyBF16Suppressed(boundaryLogits, target.arch.Vocab, nil)
	if err != nil {
		t.Fatalf("greedyBF16Suppressed: %v", err)
	}

	// Force EVERY round's "mtp-diag rows-moe" line (not just the session's first — the default
	// serve-noise cadence) — this receipt wants the round-by-round trend, not one sample.
	prevDiag := mtpDiagForTest
	mtpDiagForTest = true
	defer func() { mtpDiagForTest = prevDiag }()
	prevRounds := mtpRowsDiagRoundsSeen
	mtpRowsDiagRoundsSeen = 0
	defer func() { mtpRowsDiagRoundsSeen = prevRounds }()
	engagedBefore := mtpRowsDriverEngaged.Load()
	// mtpRowsMoEMaxGroupSize (and the histogram beside it) are "most recent value" gauges, not
	// monotonic counters — TestMTPRowsDriverVerifyMatchesRowMajor_Good (mtp_rows_driver_test.go)
	// does a before-vs-after EQUALITY check on mtpRowsMoEMaxGroupSize against its own tiny fixture,
	// which deterministically produces maxGroup=4 (NumExperts=TopK=2, K=4: every row routes to
	// both experts). This file sorts alphabetically before mtp_rows_driver_test.go, so leaving
	// these gauges at THIS test's real 26B values would poison that test's "before" baseline the
	// moment it happens to match 4 too — restore them exactly as found.
	prevMaxGroup := mtpRowsMoEMaxGroupSize.Load()
	prevHist1, prevHist2, prevHist3, prevHist4Plus := mtpRowsMoEGroupHist1.Load(), mtpRowsMoEGroupHist2.Load(), mtpRowsMoEGroupHist3.Load(), mtpRowsMoEGroupHist4Plus.Load()
	defer func() {
		mtpRowsMoEMaxGroupSize.Store(prevMaxGroup)
		mtpRowsMoEGroupHist1.Store(prevHist1)
		mtpRowsMoEGroupHist2.Store(prevHist2)
		mtpRowsMoEGroupHist3.Store(prevHist3)
		mtpRowsMoEGroupHist4Plus.Store(prevHist4Plus)
	}()

	var lastSnap mtpRowsDiagSnapshot
	var naturalD0Mismatches int
	trace := captureNativeTraceLog(t, func() {
		for r := range rounds {
			draftRes, derr := pair.DraftBlockFromSession(target, lastToken, k)
			if derr != nil {
				t.Fatalf("round %d: DraftBlockFromSession: %v", r, derr)
			}
			draftTokens := draftRes.Tokens
			if len(draftTokens) == 0 {
				t.Fatalf("round %d: drafter proposed zero tokens", r)
			}
			if len(draftTokens) > k {
				draftTokens = draftTokens[:k]
			}
			if draftTokens[0] != lastToken {
				// Informational only (see the file header): this is exactly the comparison
				// verifyDraftBlockFromSessionWithSuppress makes to decide an immediate,
				// whole-block reject in the NATURAL pair loop — counted here to quantify why
				// that loop alone never reached the #53 seam on these checkpoints.
				naturalD0Mismatches++
			}
			rows, _, verr := target.verifyAssistantDraftRows(draftTokens, nil)
			if verr != nil {
				t.Fatalf("round %d: verifyAssistantDraftRows: %v", r, verr)
			}
			if len(rows) != len(draftTokens) {
				t.Fatalf("round %d: verify returned %d rows, want %d", r, len(rows), len(draftTokens))
			}
			lastSnap = mtpRowsDiagLast
			lastToken = rows[len(rows)-1]
			t.Logf("round %d: drafted=%v engaged=%v reason=%q", r, draftTokens, lastSnap.Engaged, lastSnap.Reason)
		}
	})

	roundsSeen := mtpRowsDiagRoundsSeen
	engagedDelta := mtpRowsDriverEngaged.Load() - engagedBefore

	t.Logf("mtp-diag rows-moe trace (%d rounds seen, %d/%d natural-loop d0 mismatches):\n%s", roundsSeen, naturalD0Mismatches, rounds, trace)
	t.Logf("VERDICT counters: rounds=%d driverEngagedRounds=%d lastRound{engaged=%v reason=%q K=%d hist{1=%d 2=%d 3=%d 4+=%d} maxGroup=%d wall{attn=%s moe=%s}}",
		roundsSeen, engagedDelta, lastSnap.Engaged, lastSnap.Reason, lastSnap.K,
		lastSnap.Hist1, lastSnap.Hist2, lastSnap.Hist3, lastSnap.Hist4Plus, lastSnap.MaxGroup, lastSnap.AttnWall, lastSnap.MoEWall)

	if roundsSeen == 0 {
		t.Fatal("the #53 instrument never saw a verify round — verifyRowsMoEBatchedHiddens's seam was never reached; the receipt itself is broken, not just the driver")
	}

	if engagedDelta == 0 {
		// Hypothesis A: the driver declined every real round. The instrument's whole point is
		// naming WHICH condition — a blank reason means mtpRowsDriverDeclineReason drifted from
		// the eligibility walk it mirrors.
		if lastSnap.Reason == "" {
			t.Fatal("driver never engaged this run but the last round's decline reason is empty — mtpRowsDriverDeclineReason is out of sync with the eligibility walk")
		}
		t.Logf("HYPOTHESIS A CONFIRMED: the layer-major driver declined every real verify round on gemma-4-26B-A4B-it-4bit — condition: %s", lastSnap.Reason)
		return
	}

	// Hypothesis B check: the driver engaged — the group histogram is the receipt for whether
	// real routing degenerated to mostly-singleton groups.
	total := lastSnap.Hist1 + lastSnap.Hist2 + lastSnap.Hist3 + lastSnap.Hist4Plus
	if total == 0 {
		t.Fatal("driver engaged but the last round's group histogram is all-zero — mtpRowsMoEGroupHistBump wiring is broken")
	}
	if lastSnap.MaxGroup < 1 {
		t.Fatalf("driver engaged but mtpRowsMoEMaxGroupSize = %d, want >= 1", lastSnap.MaxGroup)
	}
	multiRow := lastSnap.Hist2 + lastSnap.Hist3 + lastSnap.Hist4Plus
	t.Logf("HYPOTHESIS B check: engaged=%d/%d rounds; last round singleton-groups=%d multi-row-groups=%d (of %d touched experts) maxGroup=%d — attn=%s moe=%s",
		engagedDelta, roundsSeen, lastSnap.Hist1, multiRow, total, lastSnap.MaxGroup, lastSnap.AttnWall, lastSnap.MoEWall)
	if multiRow == 0 {
		t.Logf("VERDICT: real 26B routing scattered every touched expert into a singleton group this round — grouped batching has no weight-read saving to offer (hypothesis B, scatter form)")
	} else {
		t.Logf("VERDICT: real 26B routing produced %d multi-row expert group(s) this round (max %d) — the batching DOES fold weight reads; if -43%% persists, the cost is elsewhere (gather/scatter overhead or the attention half)", multiRow, lastSnap.MaxGroup)
	}
}
