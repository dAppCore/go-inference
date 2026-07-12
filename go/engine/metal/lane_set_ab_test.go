// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"context"
	"os"
	"slices"
	"testing"
	"time"

	"dappco.re/go/inference"
)

// laneSetABModelDir is the E2B checkpoint the throughput receipt runs against.
// Overridable so the receipt can point at another ICB-eligible dense/PLE dir.
var laneSetABModelDir = envOr("LTHN_CB_AB_MODEL", "/Users/snider/Lethean/data/models/gemma4e2b")

func envOr(k, def string) string {
	if v := os.Getenv(k); v != "" {
		return v
	}
	return def
}

// TestLaneSetThroughputAB is the ONE honest throughput receipt on a real model:
// four concurrent greedy (temperature-0) requests decoded SERIALLY (one lane at
// a time) vs through the multi-session owner (four lanes advanced by one shared
// batched forward per step). It reports both aggregate tok/s and the speedup,
// and asserts the two paths produce byte-identical per-request tokens — the win,
// if any, is pure scheduling, never a change in output.
//
// Not a gate (t.Log, no throughput assertion): the number is the deliverable,
// and whether the dense ICB path is CB-count-bound at K=4 is exactly the open
// question docs/design-continuous-batching.md §d flags. It runs only with the
// checkpoint present and the metallib set; skipped otherwise.
func TestLaneSetThroughputAB(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set — GPU throughput receipt")
	}
	if _, err := os.Stat(laneSetABModelDir); err != nil {
		t.Skipf("A/B model dir not present (%s) — set LTHN_CB_AB_MODEL", laneSetABModelDir)
	}
	tm, err := LoadTokenModelDir(laneSetABModelDir, 4096)
	if err != nil {
		t.Fatalf("LoadTokenModelDir(%s): %v", laneSetABModelDir, err)
	}
	m, ok := tm.(*NativeTokenModel)
	if !ok {
		if closer, ok := tm.(interface{ Close() error }); ok {
			_ = closer.Close()
		}
		t.Skipf("loaded model is %T, not *NativeTokenModel — cannot open a lane set", tm)
	}
	defer m.Close()

	const k, maxNew = 4, 64
	ctx := context.Background()
	specs := make([]inference.LaneSpec, k)
	for i := range specs {
		ids := make([]int32, 24) // varied but valid prompt fills
		for j := range ids {
			ids[j] = int32(2 + (i*31+j*7)%2000)
		}
		specs[i] = inference.LaneSpec{PromptIDs: ids, MaxNew: maxNew}
	}

	// Serial: one lane at a time, each run to completion.
	serialTokens := make([][]int32, k)
	serialStart := time.Now()
	var serialGen int
	for i, spec := range specs {
		ls, err := m.OpenLaneSet(inference.LaneSetConfig{MaxLanes: 1})
		if err != nil {
			t.Skipf("OpenLaneSet(serial): %v (model may not be ICB-eligible)", err)
		}
		h, err := ls.Prepare(ctx, spec)
		if err != nil {
			_ = ls.Close()
			t.Skipf("Prepare(serial %d): %v", i, err)
		}
		got, _ := drainLaneSet(t, ls)
		serialTokens[i] = got[h.ID]
		serialGen += len(serialTokens[i])
		_ = ls.Close()
	}
	serialWall := time.Since(serialStart)

	// Batched: all K lanes in one owner, advanced together.
	ls, err := m.OpenLaneSet(inference.LaneSetConfig{MaxLanes: k})
	if err != nil {
		t.Fatalf("OpenLaneSet(batched): %v", err)
	}
	handles := make([]inference.LaneHandle, k)
	batchStart := time.Now()
	for i, spec := range specs {
		if handles[i], err = ls.Prepare(ctx, spec); err != nil {
			t.Fatalf("Prepare(batched %d): %v", i, err)
		}
	}
	batchedTokens, batchedFwd := drainLaneSet(t, ls)
	batchWall := time.Since(batchStart)
	_ = ls.Close()

	var batchGen int
	for i := range specs {
		got := batchedTokens[handles[i].ID]
		batchGen += len(got)
		if !slices.Equal(got, serialTokens[i]) {
			t.Fatalf("lane %d: batched tokens diverge from serial — throughput win must not change output\n batched=%v\n serial =%v", i, got, serialTokens[i])
		}
	}

	serialTokS := float64(serialGen) / serialWall.Seconds()
	batchTokS := float64(batchGen) / batchWall.Seconds()
	t.Logf("A/B (E2B, %d concurrent, temp 0, maxNew %d, prompt 24 tok — prefill included):", k, maxNew)
	t.Logf("  serial (one lane at a time): %d tok in %v = %.1f tok/s aggregate", serialGen, serialWall.Round(time.Millisecond), serialTokS)
	t.Logf("  batched (%d lanes, one shared forward/step): %d tok in %v = %.1f tok/s aggregate (%d batched forwards)", k, batchGen, batchWall.Round(time.Millisecond), batchTokS, batchedFwd)
	if serialTokS > 0 {
		t.Logf("  aggregate speedup: %.2fx", batchTokS/serialTokS)
	}
}
