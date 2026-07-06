// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"math"
	"testing"
)

// TestCrossEntropyBackwardF32 verifies the loss gradient against central finite differences of the
// mean softmax cross-entropy.
func TestCrossEntropyBackwardF32(t *testing.T) {
	const rows, vocab = 3, 5
	logits := syntheticFloat32(rows*vocab, 1)
	targets := []int32{0, 2, 4}
	lossOf := func(l []float32) float64 {
		loss, _, err := CrossEntropyBackwardF32(l, targets, rows, vocab)
		if err != nil {
			t.Fatal(err)
		}
		return float64(loss)
	}
	_, dLogits, err := CrossEntropyBackwardF32(logits, targets, rows, vocab)
	if err != nil {
		t.Fatalf("CrossEntropyBackwardF32: %v", err)
	}
	const eps = 1.0 / 2048
	for i := range logits {
		orig := logits[i]
		logits[i] = orig + eps
		lp := lossOf(logits)
		logits[i] = orig - eps
		lm := lossOf(logits)
		logits[i] = orig
		fd := (lp - lm) / (2 * eps)
		if math.Abs(fd-float64(dLogits[i])) > 1e-2*(1+math.Abs(fd)) {
			t.Errorf("dLogits[%d]: analytic %.5f vs finite-diff %.5f", i, dLogits[i], fd)
		}
	}
	t.Logf("cross-entropy VJP matches finite differences: dLogits[%d] within tol", len(dLogits))
}

// TestTrainStepReducesLoss is the end-to-end proof that native training works: a linear classifier
// (logits = X·Wᵀ) trained on fixed targets with cross-entropy + the linear VJP + AdamW must drive the
// loss DOWN over steps. A gradient check proves a gradient; this proves the whole loop — forward (steel
// GEMM), loss/grad, backward (LinearBackwardF32), optimiser (AdamW) — actually learns.
func TestTrainStepReducesLoss(t *testing.T) {
	requireNativeRuntime(t)
	const rows, d, vocab, steps = 16, 8, 4, 200
	x := syntheticFloat32(rows*d, 7)
	targets := make([]int32, rows)
	for i := range targets {
		targets[i] = int32(i % vocab)
	}
	w := syntheticFloat32(vocab*d, 9)
	opt := NewAdamW(vocab*d, 0.1, 0.0)

	var first, last float32
	for s := range steps {
		logits, err := MatMulF32NT(x, w, rows, d, vocab) // [rows,vocab]
		if err != nil {
			t.Fatalf("forward step %d: %v", s, err)
		}
		loss, dLogits, err := CrossEntropyBackwardF32(logits, targets, rows, vocab)
		if err != nil {
			t.Fatalf("loss step %d: %v", s, err)
		}
		if s == 0 {
			first = loss
		}
		last = loss
		_, dW, err := LinearBackwardF32(dLogits, x, w, rows, d, vocab)
		if err != nil {
			t.Fatalf("backward step %d: %v", s, err)
		}
		if err := opt.Step(w, dW); err != nil {
			t.Fatalf("optimiser step %d: %v", s, err)
		}
	}
	if last >= first*0.3 {
		t.Fatalf("training did not reduce loss enough: first=%.4f last=%.4f", first, last)
	}
	t.Logf("native training step works: cross-entropy loss %.4f → %.4f over %d AdamW steps", first, last, steps)
}
