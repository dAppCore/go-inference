// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"math"

	core "dappco.re/go"
)

// train_optim.go is the loss + optimiser half of native training (12-14): the cross-entropy objective
// the SFT loop minimises and the AdamW step that applies the gradients the VJPs (train_backward.go)
// produce. With these, the backward primitives, and the steel-GEMM forward, native can run a real
// gradient-descent step end to end — TestTrainStepReducesLoss drives the whole loop and asserts the
// loss falls, the only honest proof that "training" works (a passing gradient check proves a gradient,
// not that the system learns). f32 throughout, matching metal's optimiser precision.

// CrossEntropyBackwardF32 computes the mean softmax cross-entropy loss over `rows` samples of `vocab`
// logits against integer targets, and its gradient w.r.t. the logits — the standard dL/dlogits =
// (softmax(logits) − onehot(target)) / rows. Returns (meanLoss, dLogits[rows,vocab]). This is the head
// of the training graph; dLogits flows back through lm_head and the layers via the VJPs.
func CrossEntropyBackwardF32(logits []float32, targets []int32, rows, vocab int) (float32, []float32, error) {
	if len(logits) != rows*vocab || len(targets) != rows {
		return 0, nil, core.NewError("native.CrossEntropyBackwardF32: logits must be [rows,vocab] and targets [rows]")
	}
	dLogits := make([]float32, rows*vocab)
	var lossSum float64
	inv := 1.0 / float64(rows)
	for r := range rows {
		lr := logits[r*vocab : (r+1)*vocab]
		dr := dLogits[r*vocab : (r+1)*vocab]
		mx := lr[0]
		for _, v := range lr {
			if v > mx {
				mx = v
			}
		}
		var sum float64
		for _, v := range lr {
			sum += math.Exp(float64(v - mx))
		}
		logSum := math.Log(sum) + float64(mx)
		t := int(targets[r])
		if t < 0 || t >= vocab {
			return 0, nil, core.NewError("native.CrossEntropyBackwardF32: target out of range")
		}
		lossSum += logSum - float64(lr[t]) // −log softmax[t]
		for i := range vocab {
			p := math.Exp(float64(lr[i]-mx)) / sum
			g := p
			if i == t {
				g -= 1
			}
			dr[i] = float32(g * inv)
		}
	}
	return float32(lossSum * inv), dLogits, nil
}

// AdamW is the decoupled-weight-decay Adam optimiser state for one parameter tensor: the first/second
// moment running averages and the step counter, with the usual hyper-parameters. One AdamW per trained
// tensor (each LoRA factor); Step applies one update in place.
type AdamW struct {
	M, V         []float32
	T            int
	Beta1, Beta2 float32
	LR, Eps, WD  float32
}

// NewAdamW builds the optimiser state for a parameter tensor of length n with metal's SFT defaults
// (β1=0.9, β2=0.999, ε=1e-8); lr and weight decay are passed per the training config.
func NewAdamW(n int, lr, weightDecay float32) *AdamW {
	return &AdamW{
		M: make([]float32, n), V: make([]float32, n),
		Beta1: 0.9, Beta2: 0.999, LR: lr, Eps: 1e-8, WD: weightDecay,
	}
}

// Step applies one AdamW update to params in place from grads (same length): the bias-corrected moment
// estimates drive the step, and decoupled weight decay is applied directly to the parameter (not the
// gradient), exactly as AdamW (Loshchilov & Hutter) and metal's optim.go do.
func (a *AdamW) Step(params, grads []float32) error {
	if len(params) != len(grads) || len(params) != len(a.M) {
		return core.NewError("native.AdamW.Step: params/grads/state length mismatch")
	}
	a.T++
	b1, b2 := float64(a.Beta1), float64(a.Beta2)
	bc1 := 1 - math.Pow(b1, float64(a.T))
	bc2 := 1 - math.Pow(b2, float64(a.T))
	for i := range params {
		g := float64(grads[i])
		m := b1*float64(a.M[i]) + (1-b1)*g
		v := b2*float64(a.V[i]) + (1-b2)*g*g
		a.M[i], a.V[i] = float32(m), float32(v)
		mhat := m / bc1
		vhat := v / bc2
		upd := mhat/(math.Sqrt(vhat)+float64(a.Eps)) + float64(a.WD)*float64(params[i])
		params[i] = float32(float64(params[i]) - float64(a.LR)*upd)
	}
	return nil
}
