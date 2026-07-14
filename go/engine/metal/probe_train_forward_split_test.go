// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"os"
	"testing"
	"time"

	"dappco.re/go/inference"
)

// TestProbeTrainForwardWallSplit (LTHN_PROBE_MODEL-gated; needs a bf16 snapshot) measures
// where a head-LoRA SFT step's wall time goes on a REAL model at a training-shaped
// sequence length — the evidence for the #57 "bf16 as op option" remainder (Lemma trains
// in bf16; the option is worth building where the wall actually is):
//
//	A  ForwardCaptureHiddens(ids)      — the trainer's forward TODAY: a serial per-token
//	                                     walk of the whole sequence (re-run every step).
//	B  PrefillTokens(ids), fresh sess  — the engine's batched-dense route over the SAME
//	                                     ids: the ceiling a batched capture forward could
//	                                     approach (weight-read-once GEMMs, layer-major).
//	C  LoRATrainer.Step(batch)         — one full SFT step; C − A ≈ the HOST half (f32
//	                                     head matmul against [vocab,dModel], softmax
//	                                     backward over [T,vocab], LoRA grads, AdamW).
//
// The A/B ratio is the batched-capture upside; the (C−A)/C share is the host-half tax
// that no forward-route change touches (its fix is GPU head/loss ops, a separate rung).
//
// VERDICT (2026-07-14, E2B-it-bf16, T=128): A 3.08s vs A2 67ms — 46× — and the SFT step
// fell 3.64s → 0.77s (4.8×). The identity check found the SERIAL capture divergent on the
// real model: serial-vs-batched differed on 128/128 rows (worst |Δ| 48.6), and against the
// SERVING prefill's boundary hidden the batched capture matched EXACTLY (|Δ| 0) while the
// serial ICB capture was off by |Δ| 34 — the old training forward disagreed with what the
// engine serves on the PLE arch (the loss shift 15.15 → 10.10 at B=0 is the trainer
// finally seeing the served model's true CE). The remaining step wall is the HOST half
// (~91%): the f32 head matmul against [vocab,dModel], the [T,vocab] softmax backward and
// the LoRA grads — the GPU head/loss rung (#390).
//
// #391 ROOT CAUSE + FIX (same day): stepBodyCapture carved the recorded ICB at a uniform
// li·opsPerLayer stride, but global layers' 2-pass SDPA records an INLINE extra op (E2B @
// maxLen 1024: 7 of them), so every layer after the first global executed a misaligned
// range and the stream's last 7 ops never ran. Carving by the recorded layerOpStarts
// boundaries dropped the serial-vs-serving delta 34 → 0.094 (and serial-vs-batched worst
// to 0.75 ≈ 3 bf16 ULP) — the residual is cross-route accumulation-order noise, present
// on any serial-replay-vs-batched-fold comparison, not a defect. Regression gate:
// TestForwardCaptureHiddens2PassBoundaries.
func TestProbeTrainForwardWallSplit(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	dir := os.Getenv("LTHN_PROBE_MODEL")
	if dir == "" {
		t.Skip("LTHN_PROBE_MODEL not set")
	}
	tm, err := LoadTokenModelDir(dir, 1024)
	if err != nil {
		t.Fatalf("LoadTokenModelDir: %v", err)
	}
	ntm := tm.(*NativeTokenModel)
	defer ntm.Close()
	if ntm.bf16 == nil {
		t.Skip("probe needs a bf16 snapshot (the trainer's precision)")
	}

	const T = 128
	ids := make([]int32, T)
	for i := range ids {
		ids[i] = int32(2 + i%97)
	}

	// A — the serial capture forward the trainer runs today.
	stepper, err := ntm.OpenSession()
	if err != nil {
		t.Fatalf("OpenSession(A): %v", err)
	}
	sessA := stepper.(*ArchSession)
	start := time.Now()
	_, perLayer, err := sessA.ForwardCaptureHiddens(ids)
	if err != nil {
		t.Fatalf("ForwardCaptureHiddens: %v", err)
	}
	aWall := time.Since(start)
	if len(perLayer) == 0 {
		t.Fatal("capture returned no layers")
	}
	sessA.Close()

	// B — the batched-dense inference route over the same ids on a FRESH session.
	stepper, err = ntm.OpenSession()
	if err != nil {
		t.Fatalf("OpenSession(B): %v", err)
	}
	sessB := stepper.(*ArchSession)
	start = time.Now()
	if err := sessB.PrefillTokens(ids); err != nil {
		t.Fatalf("PrefillTokens: %v", err)
	}
	bWall := time.Since(start)
	// The SERVING truth: the boundary hidden of the last prompt token, exactly what
	// decode continues from. The capture route that matches THIS is the correct one.
	serveBoundary := append([]byte(nil), sessB.retainedHidden...)
	if len(serveBoundary) == 0 {
		serveBoundary = append(serveBoundary, sessB.sampleHidden...)
	}
	sessB.Close()

	// A2 — the batched capture forward the trainer rides now (ForwardCaptureFinalHidden).
	stepper, err = ntm.OpenSession()
	if err != nil {
		t.Fatalf("OpenSession(A2): %v", err)
	}
	sessA2 := stepper.(*ArchSession)
	engagedBefore := captureFinalHiddenBatchedChunksForTest
	start = time.Now()
	if _, err := sessA2.ForwardCaptureFinalHidden(ids); err != nil {
		t.Fatalf("ForwardCaptureFinalHidden: %v", err)
	}
	a2Wall := time.Since(start)
	a2Engaged := captureFinalHiddenBatchedChunksForTest > engagedBefore
	sessA2.Close()

	// C — one full head-LoRA SFT step (its own fresh frozen session inside the trainer).
	trainer, err := NewLoRATrainer(ntm, inference.TrainingConfig{})
	if err != nil {
		t.Fatalf("NewLoRATrainer: %v", err)
	}
	defer trainer.Close()
	start = time.Now()
	loss, err := trainer.Step(inference.Batch{TokenIDs: [][]int32{ids}})
	if err != nil {
		t.Fatalf("Step: %v", err)
	}
	cWall := time.Since(start)

	// Identity check on the REAL model: the serial capture's final layer vs the batched
	// capture's rows. The fixture gate proves byte-identity on the synthetic session;
	// this reports what the real E2B does (rows differing + worst |Δ| in f32).
	wantLast := perLayer[len(perLayer)-1]
	gotLast, err := func() ([]byte, error) {
		st, e := ntm.OpenSession()
		if e != nil {
			return nil, e
		}
		sp := st.(*ArchSession)
		defer sp.Close()
		return sp.ForwardCaptureFinalHidden(ids)
	}()
	if err != nil {
		t.Fatalf("ForwardCaptureFinalHidden(identity): %v", err)
	}
	rowB := len(wantLast) / T
	diffRows, worst := 0, 0.0
	for r := 0; r < T; r++ {
		wr, gr := wantLast[r*rowB:(r+1)*rowB], gotLast[r*rowB:(r+1)*rowB]
		rowDiff := false
		for i := 0; i+1 < len(wr); i += 2 {
			a := bf16ToF32(wr[i], wr[i+1])
			b := bf16ToF32(gr[i], gr[i+1])
			d := float64(a - b)
			if d < 0 {
				d = -d
			}
			if d > worst {
				worst = d
			}
			if wr[i] != gr[i] || wr[i+1] != gr[i+1] {
				rowDiff = true
			}
		}
		if rowDiff {
			diffRows++
		}
	}
	t.Logf("T=%d  identity serial-vs-batched: differing rows %d/%d, worst |Δ| %.6g", T, diffRows, T, worst)
	// Which route agrees with SERVING? Compare each capture's LAST row against the
	// serving prefill's boundary hidden.
	rowDelta := func(row, ref []byte) float64 {
		w := 0.0
		for i := 0; i+1 < len(row) && i+1 < len(ref); i += 2 {
			d := float64(bf16ToF32(row[i], row[i+1]) - bf16ToF32(ref[i], ref[i+1]))
			if d < 0 {
				d = -d
			}
			if d > w {
				w = d
			}
		}
		return w
	}
	if len(serveBoundary) > 0 {
		t.Logf("T=%d  vs SERVING boundary: serial worst |Δ| %.6g, batched worst |Δ| %.6g",
			T, rowDelta(wantLast[(T-1)*rowB:], serveBoundary), rowDelta(gotLast[(T-1)*rowB:], serveBoundary))
	} else {
		t.Log("serving boundary hidden unavailable on this session shape")
	}

	host := cWall - a2Wall
	t.Logf("T=%d  A  serial-capture-forward  = %v", T, aWall)
	t.Logf("T=%d  B  batched-dense-prefill   = %v  (A/B = %.1fx)", T, bWall, float64(aWall)/float64(bWall))
	t.Logf("T=%d  A2 batched-capture-forward = %v  (engaged=%v, A/A2 = %.1fx)", T, a2Wall, a2Engaged, float64(aWall)/float64(a2Wall))
	t.Logf("T=%d  C  full SFT step           = %v  (loss %.4f, was 3.64s pre-batched-capture)", T, cWall, loss)
	t.Logf("T=%d  C-A2 host half             = %v  (%.0f%% of the step)", T, host, 100*float64(host)/float64(cWall))
}
