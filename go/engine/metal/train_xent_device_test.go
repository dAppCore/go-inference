// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"math"
	"testing"

	"dappco.re/go/inference/model"
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

// TestCrossEntropyBackwardF32DeviceInRealSessionSFT is the #390 loss-trajectory receipt: the GPU CE
// dropped into the FULL head-LoRA SFT loop over a real (synthetic-weight) gemma ArchSession — the same
// engine forward + LoRA + AdamW as TestRealSessionHeadLoRASFT — must track the host-CE trajectory
// step-for-step. Two adapters train from identical init, one on the host CE, one on the device CE; the
// step-0 CE must agree within #390's 1e-5 bar (same first-step loss at B=0), the whole trajectory must
// stay in agreement, and both must fall. No model download — a small synthetic gemma stack.
func TestCrossEntropyBackwardF32DeviceInRealSessionSFT(t *testing.T) {
	requireNativeRuntime(t)
	if !gpuHasSoftmaxXent() {
		t.Skip("lthn_softmax_xent kernel unavailable (custom metallib not loaded)")
	}
	const dModel, nHeads, nKV, headDim, dFF = 512, 8, 4, 64, 1024
	const vocab, nL, maxLen, rank, steps = 64, 3, 64, 8, 80
	scaling := float32(16.0 / rank)
	eps := float32(1e-5)

	layers := make([]DecodeLayerWeights, nL)
	types := make([]string, nL)
	for li := range layers {
		layers[li] = forwardLayer(dModel, nHeads, nKV, headDim, dFF, (li+1)*100)
		types[li] = "full_attention"
	}
	g := &BF16Model{Layers: layers, Embed: toBF16Bytes(syntheticFloat32(vocab*dModel, 21)),
		FinalNorm: toBF16Bytes(syntheticFloat32(dModel, 22))}
	g.LMHead, g.Tied = g.Embed, true
	arch := model.Arch{
		Hidden: dModel, Heads: nHeads, KVHeads: nKV, HeadDim: headDim, FF: dFF, Vocab: vocab,
		GlobalHeadDim: headDim, GlobalKVHeads: nKV,
		Eps: eps, AttnScale: 0.125, RopeBase: 10000, RopeScale: 1, RopeLocalBase: 10000,
		RotaryDim: headDim, RotaryDimLocal: headDim, Layer: model.DeriveLayers(types, 0),
	}
	sess, err := NewArchSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("NewArchSession: %v", err)
	}
	ids := []int32{1, 2, 3, 4, 5, 6, 7, 8}
	T := len(ids)

	_, perLayer, err := sess.ForwardCaptureHiddens(ids)
	if err != nil {
		t.Fatalf("ForwardCaptureHiddens: %v", err)
	}
	normed := rmsNormForwardF32(bf16ToF32Slice(perLayer[nL-1]), bf16ToF32Slice(g.FinalNorm), T, dModel, eps)
	baseLogits, err := MatMulF32NT(normed, bf16ToF32Slice(g.LMHead), T, dModel, vocab)
	if err != nil {
		t.Fatalf("base logits: %v", err)
	}
	targets := make([]int32, T)
	for i := range targets {
		targets[i] = int32((i * 7) % vocab)
	}

	// Two adapters from IDENTICAL init: one trained on the host CE, one on the device CE.
	a0 := syntheticFloat32(rank*dModel, 11)
	for i := range a0 {
		a0[i] *= 0.2
	}
	aHost := append([]float32(nil), a0...)
	aDev := append([]float32(nil), a0...)
	bHost := make([]float32, vocab*rank)
	bDev := make([]float32, vocab*rank)
	optAHost, optBHost := NewAdamW(rank*dModel, 0.05, 0.0), NewAdamW(vocab*rank, 0.05, 0.0)
	optADev, optBDev := NewAdamW(rank*dModel, 0.05, 0.0), NewAdamW(vocab*rank, 0.05, 0.0)

	// stepCE runs one SFT step with the given CE implementation and returns the step loss.
	stepCE := func(a, b []float32, optA, optB *AdamW, ce func([]float32, []int32, int, int) (float32, []float32, error)) (float32, error) {
		xA, delta, e := LoRAForwardF32(normed, a, b, T, dModel, vocab, rank, scaling)
		if e != nil {
			return 0, e
		}
		logits := make([]float32, T*vocab)
		for i := range logits {
			logits[i] = baseLogits[i] + delta[i]
		}
		loss, dLogits, e := ce(logits, targets, T, vocab)
		if e != nil {
			return 0, e
		}
		dA, dB, _, e := LoRABackwardF32(dLogits, normed, a, b, xA, T, dModel, vocab, rank, scaling)
		if e != nil {
			return 0, e
		}
		if e := optA.Step(a, dA); e != nil {
			return 0, e
		}
		if e := optB.Step(b, dB); e != nil {
			return 0, e
		}
		return loss, nil
	}

	var firstHost, firstDev, lastHost, lastDev float32
	var worstRel float64
	for s := range steps {
		lh, e := stepCE(aHost, bHost, optAHost, optBHost, CrossEntropyBackwardF32)
		if e != nil {
			t.Fatalf("host step %d: %v", s, e)
		}
		ld, e := stepCE(aDev, bDev, optADev, optBDev, crossEntropyBackwardF32Device)
		if e != nil {
			t.Fatalf("device step %d: %v", s, e)
		}
		rel := math.Abs(float64(ld-lh)) / (1 + math.Abs(float64(lh)))
		if rel > worstRel {
			worstRel = rel
		}
		if s == 0 {
			firstHost, firstDev = lh, ld
			if rel > 1e-5 {
				t.Fatalf("step-0 CE host %.8f vs GPU %.8f: relative %.3g > 1e-5", lh, ld, rel)
			}
		}
		lastHost, lastDev = lh, ld
	}
	if lastHost >= firstHost || lastDev >= firstDev {
		t.Fatalf("SFT did not reduce loss: host %.4f→%.4f, device %.4f→%.4f", firstHost, lastHost, firstDev, lastDev)
	}
	if worstRel > 1e-3 {
		t.Fatalf("host/device loss trajectories diverged: worst per-step relative %.3g > 1e-3 over %d steps", worstRel, steps)
	}
	t.Logf("trajectory intact over %d steps: host %.4f→%.4f, device %.4f→%.4f, worst per-step rel %.2g",
		steps, firstHost, lastHost, firstDev, lastDev, worstRel)
}
