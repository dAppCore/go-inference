// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"math"
	"testing"
)

// TestCrossEntropyBackwardF32Device is the #390 correctness receipt: the fused GPU softmax-xent
// (crossEntropyBackwardF32Device / lthn_softmax_xent_rows_f32) must reproduce the host reference
// oracle CrossEntropyBackwardF32 — same mean loss and same [rows,vocab] gradient — within the f32
// tolerance the loss-trajectory bar allows. The host is exact-in-f64 and single-goroutine; the device
// runs f32 with a tree-reduced vocab sum, so the two differ only by f32 rounding (~1e-6 relative on the
// loss, ~1e-6 absolute on the gradient), well inside #390's 1e-5 loss bar. Good/Bad/Ugly are the
// numeric edges that break a naive kernel: aligned vocab, a stride-remainder vocab with a last-index
// target, and a single peaked row where one dominating logit stresses the max-subtraction.
func TestCrossEntropyBackwardF32Device(t *testing.T) {
	requireNativeRuntime(t)
	if !gpuHasSoftmaxXent() {
		t.Skip("lthn_softmax_xent kernel unavailable (custom metallib not loaded)")
	}
	cases := []struct {
		name        string
		rows, vocab int
		peaked      bool // scale one logit per row up, driving softmax toward one-hot
	}{
		{"Good_aligned_vocab", 8, 2048, false},    // vocab a multiple of the 1024 threadgroup
		{"Bad_stride_remainder", 6, 4099, false},  // vocab not a multiple of 1024 → the loop tail
		{"Ugly_single_peaked_row", 1, 3000, true}, // one row, one dominating logit
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			logits := syntheticFloat32(tc.rows*tc.vocab, 5)
			targets := make([]int32, tc.rows)
			for r := range targets {
				targets[r] = int32((r*37 + 11) % tc.vocab)
				if r == 0 {
					targets[r] = int32(tc.vocab - 1) // exercise a last-index target
				}
			}
			if tc.peaked {
				for r := 0; r < tc.rows; r++ {
					logits[r*tc.vocab+(r*13)%tc.vocab] += 20 // dwarf the rest of the row
				}
			}

			wantLoss, wantDL, err := CrossEntropyBackwardF32(logits, targets, tc.rows, tc.vocab)
			if err != nil {
				t.Fatalf("host CrossEntropyBackwardF32: %v", err)
			}
			gotLoss, gotDL, err := crossEntropyBackwardF32Device(logits, targets, tc.rows, tc.vocab)
			if err != nil {
				t.Fatalf("crossEntropyBackwardF32Device: %v", err)
			}

			lossRel := math.Abs(float64(gotLoss-wantLoss)) / (1 + math.Abs(float64(wantLoss)))
			if lossRel > 1e-5 {
				t.Fatalf("loss %.8f vs host %.8f: relative %.3g > 1e-5", gotLoss, wantLoss, lossRel)
			}
			if len(gotDL) != len(wantDL) {
				t.Fatalf("dLogits length %d, want %d", len(gotDL), len(wantDL))
			}
			var worstDL float64
			for i := range wantDL {
				d := math.Abs(float64(gotDL[i] - wantDL[i]))
				if d > worstDL {
					worstDL = d
				}
			}
			if worstDL > 5e-6 {
				t.Fatalf("dLogits worst |Δ| %.3g > 5e-6", worstDL)
			}
			t.Logf("rows=%d vocab=%d: loss %.6f (rel %.2g), dLogits worst |Δ| %.2g",
				tc.rows, tc.vocab, gotLoss, lossRel, worstDL)
		})
	}
}

// TestCrossEntropyBackwardF32Auto pins the trainer's dispatcher: with the kernel loadable it agrees
// with the host oracle within the loss tolerance (the device path), and with LTHN_TRAIN_GPU_CE=0 it IS
// the host reference byte-for-byte (the oracle stays selectable). This is the guard that flipping the
// gate cannot change correctness, only where the work runs.
func TestCrossEntropyBackwardF32Auto(t *testing.T) {
	requireNativeRuntime(t)
	const rows, vocab = 4, 2048
	logits := syntheticFloat32(rows*vocab, 9)
	targets := []int32{0, 511, 1234, vocab - 1}

	wantLoss, wantDL, err := CrossEntropyBackwardF32(logits, targets, rows, vocab)
	if err != nil {
		t.Fatalf("host reference: %v", err)
	}

	// Gate OFF → byte-identical to the host oracle regardless of kernel availability.
	t.Setenv("LTHN_TRAIN_GPU_CE", "0")
	offLoss, offDL, err := CrossEntropyBackwardF32Auto(logits, targets, rows, vocab)
	if err != nil {
		t.Fatalf("auto (gate off): %v", err)
	}
	if offLoss != wantLoss {
		t.Fatalf("gate off loss %v != host %v (must be the host path exactly)", offLoss, wantLoss)
	}
	for i := range wantDL {
		if offDL[i] != wantDL[i] {
			t.Fatalf("gate off dLogits[%d] %v != host %v (must be the host path exactly)", i, offDL[i], wantDL[i])
		}
	}

	// Gate ON → device when loadable (within tolerance), else still the host (fallback).
	t.Setenv("LTHN_TRAIN_GPU_CE", "1")
	onLoss, onDL, err := CrossEntropyBackwardF32Auto(logits, targets, rows, vocab)
	if err != nil {
		t.Fatalf("auto (gate on): %v", err)
	}
	lossRel := math.Abs(float64(onLoss-wantLoss)) / (1 + math.Abs(float64(wantLoss)))
	if lossRel > 1e-5 {
		t.Fatalf("gate on loss %.8f vs host %.8f: relative %.3g > 1e-5", onLoss, wantLoss, lossRel)
	}
	var worstDL float64
	for i := range wantDL {
		d := math.Abs(float64(onDL[i] - wantDL[i]))
		if d > worstDL {
			worstDL = d
		}
	}
	if worstDL > 5e-6 {
		t.Fatalf("gate on dLogits worst |Δ| %.3g > 5e-6", worstDL)
	}
	t.Logf("dispatcher: gate off = host exactly; gate on rel %.2g, dLogits worst |Δ| %.2g (kernel=%v)",
		lossRel, worstDL, gpuHasSoftmaxXent())
}
