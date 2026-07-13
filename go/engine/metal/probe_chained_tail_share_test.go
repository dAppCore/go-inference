// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"context"
	"os"
	"testing"

	"dappco.re/go/inference"
)

// TestProbeChainedTailShare (LTHN_PROBE_MODEL-gated) measures the GPU span
// of (a) a full chained round (forward + head argmax + next-embed) vs (b)
// the bare forward on the same lane session. The delta is the head+embed
// tail's per-round GPU cost — the budget a BATCHED (rendezvous) tail could
// save at K lanes is (K-1)/K of the head part, against the slowest-lane
// alignment it re-introduces. The 26B receipt (2026-07-16): round 7.18 ms,
// bare 6.51 ms, tail 0.67 ms (9.3%) — at K=4 the ~0.5 ms/round ceiling is
// roughly cancelled by re-introduced alignment, and the rendezvous reverses
// the ragged free-run that banked +6%: REFUTED at K≤4.
//
// K≥8 RE-CHECK (2026-07-16, closes the question): probe reproduced at
// 0.563 ms (7.9%); the LIVE A/B (LTHN_CB_CHAIN=0 — two-phase rendezvous'd
// batched Phase-1 heads — vs the chained default; 26B ctx4096, salted
// ksweep) measured chained 107/143/173/196 tok/s at K=1/2/4/8 vs two-phase
// 98/116/137/168: the rendezvous LOSES at every K, −14% at K=8. The ~8%
// theoretical tail ceiling never survives giving up the ragged free-run +
// submit-ahead. REFUTED at all K ≤ 8; do not revisit without a design that
// batches the tail WITHOUT a rendezvous.
func TestProbeChainedTailShare(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	dir := os.Getenv("LTHN_PROBE_MODEL")
	if dir == "" {
		t.Skip("LTHN_PROBE_MODEL not set")
	}
	tm, err := LoadTokenModelDir(dir, 4096)
	if err != nil {
		t.Fatalf("LoadTokenModelDir: %v", err)
	}
	ntm := tm.(*NativeTokenModel)
	defer ntm.Close()
	lsI, err := ntm.OpenLaneSet(inference.LaneSetConfig{MaxLanes: 1})
	if err != nil {
		t.Fatalf("OpenLaneSet: %v", err)
	}
	ls := lsI.(*laneSet)
	h, err := ls.Prepare(context.Background(), inference.LaneSpec{PromptIDs: []int32{2, 651, 2364, 573, 5715, 576, 25175, 235265}, MaxNew: 4})
	if err != nil {
		t.Fatalf("Prepare: %v", err)
	}
	lane := ls.lanes[h.ID]
	lane.pendingToken = 651

	span := func(cb interface {
		GPUStartTime() float64
		GPUEndTime() float64
	}) float64 {
		return (cb.GPUEndTime() - cb.GPUStartTime()) * 1e3 // ms
	}

	const n = 24
	// Arm A: full chained rounds.
	fullMS := 0.0
	for i := 0; i < n; i++ {
		cb, scr, ok, err := ls.chainRoundEncode(lane, i > 0)
		if err != nil || !ok {
			t.Fatalf("chainRoundEncode(%d): ok=%v err=%v", i, ok, err)
		}
		ls.commitChainedRound(lane, cb)
		waitUntilCompletedFast(cb)
		if i >= 4 { // skip warm-up rounds
			fullMS += span(cb)
		}
		cb.Release()
		lane.sess.headEnc.putGreedyScratch(scr)
	}
	// Arm B: bare forwards (no tail) — same session, continuing positions.
	bareMS := 0.0
	for i := 0; i < n; i++ {
		sink := &sharedStepSink{cb: commandBufferFast(queue)}
		sink.enc = computeCommandEncoderFast(sink.cb)
		if err := lane.sess.stepEncodeSharedChained(sink); err != nil {
			t.Fatalf("bare step(%d): %v", i, err)
		}
		endEncodingFast(sink.enc)
		commitCommandBufferFast(sink.cb)
		lane.sess.pos++
		waitUntilCompletedFast(sink.cb)
		if i >= 4 {
			bareMS += span(sink.cb)
		}
	}
	m := float64(n - 4)
	t.Logf("chained round GPU span: %.3f ms/round; bare forward: %.3f ms/round; tail (head+embed) share: %.3f ms/round (%.1f%%)",
		fullMS/m, bareMS/m, (fullMS-bareMS)/m, 100*(fullMS-bareMS)/(fullMS))
	_ = ls.Close()
}
