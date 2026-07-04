// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"os"
	"testing"

	"dappco.re/go/inference/model"
)

// TestBF16VsQ4PerLayer localises the bf16-decode bug WITHOUT a metal reference (the cross-engine
// harness rejects PLE on the metal side). It runs e2b-bf16 and e2b-4bit — the WORKING quant — through
// the native session over the SAME token ids and diffs their per-layer hiddens. The 4-bit weights are
// a quantised copy of the bf16, so a structurally-correct bf16 decode tracks the 4-bit at ~quant-error
// cosine (~0.97-0.99 + accumulation); a STRUCTURAL bf16 bug shows a sharp drop at the offending layer.
// Set E2B_BF16_DIR + E2B_Q4_DIR to the two snapshot dirs.
func TestBF16VsQ4PerLayer(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	bf16Dir, q4Dir := os.Getenv("E2B_BF16_DIR"), os.Getenv("E2B_Q4_DIR")
	if bf16Dir == "" || q4Dir == "" {
		t.Skip("set E2B_BF16_DIR + E2B_Q4_DIR")
	}
	const maxLen = 64
	nmB, err := LoadTokenModelDir(bf16Dir, maxLen)
	if err != nil {
		t.Fatalf("bf16 load: %v", err)
	}
	nsB, err := nmB.(model.SessionModel).OpenSession()
	if err != nil {
		t.Fatalf("bf16 session: %v", err)
	}
	nmQ, err := LoadTokenModelDir(q4Dir, maxLen)
	if err != nil {
		t.Fatalf("q4 load: %v", err)
	}
	nsQ, err := nmQ.(model.SessionModel).OpenSession()
	if err != nil {
		t.Fatalf("q4 session: %v", err)
	}

	ids := make([]int32, 8)
	for i := range ids {
		ids[i] = int32(1000 + i*131)
	}
	const captureStep = 3
	for i, id := range ids {
		capturedLayerHiddens = nil
		captureLayerHiddens = true
		eB, _ := nmB.Embed(id)
		hB, serr := nsB.(*ArchSession).StepWithID(id, eB)
		if serr != nil {
			t.Fatalf("bf16 step %d: %v", i, serr)
		}
		lB := capturedLayerHiddens

		capturedLayerHiddens = nil
		eQ, _ := nmQ.Embed(id)
		hQ, serr := nsQ.(*ArchSession).StepWithID(id, eQ)
		if serr != nil {
			t.Fatalf("q4 step %d: %v", i, serr)
		}
		lQ := capturedLayerHiddens
		captureLayerHiddens = false

		t.Logf("pos %d: embCos=%.4f finalHidCos=%.4f", i, cosineBF16(eB, eQ), cosineBF16(hB, hQ))
		if i == captureStep {
			n := len(lB)
			if len(lQ) < n {
				n = len(lQ)
			}
			worst, worstL := 2.0, -1
			for L := 0; L < n; L++ {
				c := cosineBF16(lB[L], lQ[L])
				t.Logf("  L%2d bf16-vs-q4 cosine=%.4f", L, c)
				if c < worst {
					worst, worstL = c, L
				}
			}
			t.Logf("  worst layer %d cosine=%.4f", worstL, worst)
		}
	}
}
