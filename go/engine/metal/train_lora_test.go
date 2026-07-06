// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"math"
	"testing"
)

// TestLoRABackwardF32 gradient-checks the LoRA factor gradients (and the input gradient) against finite
// differences of the delta path delta = scaling·(x·Aᵀ)·Bᵀ, with L = Σ delta·dy.
func TestLoRABackwardF32(t *testing.T) {
	requireNativeRuntime(t)
	const M, in, out, rank = 3, 6, 5, 2
	scaling := float32(2.0)
	x := syntheticFloat32(M*in, 1)
	a := syntheticFloat32(rank*in, 2)
	b := syntheticFloat32(out*rank, 3)
	dy := syntheticFloat32(M*out, 4)

	deltaOf := func() []float32 {
		_, delta, err := LoRAForwardF32(x, a, b, M, in, out, rank, scaling)
		if err != nil {
			t.Fatal(err)
		}
		return delta
	}
	loss := func() float64 {
		delta := deltaOf()
		var s float64
		for i := range delta {
			s += float64(delta[i]) * float64(dy[i])
		}
		return s
	}
	xA, _, err := LoRAForwardF32(x, a, b, M, in, out, rank, scaling)
	if err != nil {
		t.Fatal(err)
	}
	dA, dB, dX, err := LoRABackwardF32(dy, x, a, b, xA, M, in, out, rank, scaling)
	if err != nil {
		t.Fatalf("LoRABackwardF32: %v", err)
	}
	const eps = 1.0 / 1024
	check := func(name string, params, grad []float32) {
		for i := range params {
			orig := params[i]
			params[i] = orig + eps
			lp := loss()
			params[i] = orig - eps
			lm := loss()
			params[i] = orig
			fd := (lp - lm) / (2 * eps)
			if math.Abs(fd-float64(grad[i])) > 2e-2*(1+math.Abs(fd)) {
				t.Errorf("%s[%d]: analytic %.5f vs finite-diff %.5f", name, i, grad[i], fd)
			}
		}
	}
	check("dA", a, dA)
	check("dB", b, dB)
	check("dX", x, dX)
	t.Logf("LoRA VJP matches finite differences: dA[%d] dB[%d] dX[%d] within tol", len(dA), len(dB), len(dX))
}

// TestLoRASFTReducesLoss is the end-to-end proof that native LoRA SFT works: a FROZEN base classifier
// plus trainable LoRA factors (A random, B zero — standard init, so the delta starts at 0 and the model
// starts at the base loss) is trained with cross-entropy + AdamW on A,B only, and the loss must fall.
// This is the SFT loop in miniature — frozen base + LoRA adapter + the gradients flowing only to A,B.
func TestLoRASFTReducesLoss(t *testing.T) {
	requireNativeRuntime(t)
	const rows, d, vocab, rank, steps = 16, 8, 4, 4, 300
	scaling := float32(8.0 / rank)
	x := syntheticFloat32(rows*d, 7)
	w := syntheticFloat32(vocab*d, 9) // FROZEN base
	targets := make([]int32, rows)
	for i := range targets {
		targets[i] = int32((i * 3) % vocab)
	}
	a := syntheticFloat32(rank*d, 11) // A: small random
	for i := range a {
		a[i] *= 0.2
	}
	b := make([]float32, vocab*rank) // B: zero init → delta starts at 0
	optA := NewAdamW(rank*d, 0.05, 0.0)
	optB := NewAdamW(vocab*rank, 0.05, 0.0)

	base, err := MatMulF32NT(x, w, rows, d, vocab) // frozen base logits, computed once
	if err != nil {
		t.Fatal(err)
	}
	var first, last float32
	for s := range steps {
		xA, delta, err := LoRAForwardF32(x, a, b, rows, d, vocab, rank, scaling)
		if err != nil {
			t.Fatalf("lora forward %d: %v", s, err)
		}
		logits := make([]float32, rows*vocab)
		for i := range logits {
			logits[i] = base[i] + delta[i]
		}
		loss, dLogits, err := CrossEntropyBackwardF32(logits, targets, rows, vocab)
		if err != nil {
			t.Fatalf("loss %d: %v", s, err)
		}
		if s == 0 {
			first = loss
		}
		last = loss
		dA, dB, _, err := LoRABackwardF32(dLogits, x, a, b, xA, rows, d, vocab, rank, scaling)
		if err != nil {
			t.Fatalf("lora backward %d: %v", s, err)
		}
		if err := optA.Step(a, dA); err != nil {
			t.Fatal(err)
		}
		if err := optB.Step(b, dB); err != nil {
			t.Fatal(err)
		}
	}
	if last >= first*0.6 {
		t.Fatalf("LoRA SFT did not reduce loss enough: first=%.4f last=%.4f", first, last)
	}
	t.Logf("native LoRA SFT works: frozen base + trainable A/B, cross-entropy %.4f → %.4f over %d AdamW steps", first, last, steps)
}
