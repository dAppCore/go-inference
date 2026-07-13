// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"context"
	"os"
	"testing"
	"time"

	"dappco.re/go/inference"
)

// TestProbe26BLaneVsChainGPUSpan — TEMPORARY: on the real snapshot named by
// LTHN_PROBE_MODEL, decodes n tokens through (a) the serial chained decode
// and (b) a K=1 lane set, accumulating each arm's GPU execution span. Wall −
// Σspan = the arm's per-token host/sync gap: one run splits the K=1 CB
// residual into GPU-span vs host-path.
func TestProbe26BLaneVsChainGPUSpan(t *testing.T) {
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
	prompt := []int32{2, 651, 2364, 573, 5715, 576, 25175, 235265}
	const n = 128

	pieceTimingOn = true
	defer func() { pieceTimingOn = false }()

	// Arm (a): the serial chained decode (the plain serve's shape). Run
	// twice, measure the second — the first pays scratch/PSO warm-up.
	var gen []int32
	var wallChain time.Duration
	var spanChain int64
	for pass := 0; pass < 2; pass++ {
		s1, err := ntm.OpenSession()
		if err != nil {
			t.Fatalf("OpenSession: %v", err)
		}
		as1 := s1.(*ArchSession)
		chainedGPUSpanNs = 0
		t0 := time.Now()
		gen, err = as1.Generate(prompt, n, -1)
		wallChain = time.Since(t0)
		if err != nil {
			t.Fatalf("Generate: %v", err)
		}
		_ = as1.Close()
		spanChain = chainedGPUSpanNs
	}

	// Arm (b): a K=1 lane set (the CB serve's shape) — same two-pass shape.
	var wallLane time.Duration
	var spanLane int64
	got := 0
	for pass := 0; pass < 2; pass++ {
		ls, err := ntm.OpenLaneSet(inference.LaneSetConfig{MaxLanes: 1})
		if err != nil {
			t.Fatalf("OpenLaneSet: %v", err)
		}
		laneGPUSpanNs = 0
		got = 0
		t1 := time.Now()
		if _, err := ls.Prepare(context.Background(), inference.LaneSpec{PromptIDs: prompt, MaxNew: n}); err != nil {
			t.Fatalf("Prepare: %v", err)
		}
		for {
			steps, err := ls.Step(context.Background())
			if err != nil {
				t.Fatalf("Step: %v", err)
			}
			if len(steps) == 0 {
				break
			}
			for _, s := range steps {
				if s.HasToken {
					got++
				}
				if s.Terminal {
					_ = ls.Retire(s.Lane)
				}
			}
		}
		wallLane = time.Since(t1)
		_ = ls.Close()
		spanLane = laneGPUSpanNs
	}

	perTok := func(d time.Duration, count int) float64 { return float64(d.Nanoseconds()) / float64(count) / 1e6 }
	t.Logf("chain: %d tok, wall %.2fs (%.2f ms/tok), GPU span %.2f ms/tok, host gap %.2f ms/tok",
		len(gen), wallChain.Seconds(), perTok(wallChain, len(gen)),
		float64(spanChain)/float64(len(gen))/1e6,
		perTok(wallChain, len(gen))-float64(spanChain)/float64(len(gen))/1e6)
	t.Logf("lane:  %d tok, wall %.2fs (%.2f ms/tok), GPU span %.2f ms/tok, host gap %.2f ms/tok",
		got, wallLane.Seconds(), perTok(wallLane, got),
		float64(spanLane)/float64(got)/1e6,
		perTok(wallLane, got)-float64(spanLane)/float64(got)/1e6)
}
