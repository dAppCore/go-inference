// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"math"
	"testing"
)

// TestSoftcapForwardF32_Good pins the forward: logits = cap·tanh(raw/cap), and cap==0 is the
// exact no-op (the uncapped arches' contract).
func TestSoftcapForwardF32_Good(t *testing.T) {
	l := []float32{-40, -1, 0, 2, 90}
	want := make([]float32, len(l))
	for i, v := range l {
		want[i] = float32(30 * math.Tanh(float64(v)/30))
	}
	softcapForwardF32(l, 30)
	for i := range l {
		if math.Abs(float64(l[i]-want[i])) > 1e-6 {
			t.Fatalf("capped[%d] = %v, want %v", i, l[i], want[i])
		}
	}
	unchanged := []float32{-40, -1, 0, 2, 90}
	orig := append([]float32(nil), unchanged...)
	softcapForwardF32(unchanged, 0)
	for i := range unchanged {
		if unchanged[i] != orig[i] {
			t.Fatalf("cap=0 must be a no-op; [%d] changed %v -> %v", i, orig[i], unchanged[i])
		}
	}
}

// TestSoftcapHeadBackward_FD finite-difference-gates the capped head loss end to end:
// loss(raw) = CE(softcap(raw)); the analytic gradient is the CE backward scaled by the cap's
// derivative (softcapBackwardScaleF32). Central differences at eps 1/512, tolerance
// 2e-2·(1+|fd|) — the train_real_layer.go bar.
func TestSoftcapHeadBackward_FD(t *testing.T) {
	const rows, vocab = 3, 7
	const cap = 30.0
	raw := make([]float32, rows*vocab)
	for i := range raw {
		raw[i] = float32(math.Sin(float64(i)*0.7)*20 + math.Cos(float64(i))*5)
	}
	targets := []int32{2, 5, 0}

	lossAt := func(x []float32) float64 {
		capped := append([]float32(nil), x...)
		softcapForwardF32(capped, cap)
		loss, _, err := CrossEntropyBackwardF32Auto(capped, targets, rows, vocab)
		if err != nil {
			t.Fatalf("CE: %v", err)
		}
		return float64(loss)
	}

	capped := append([]float32(nil), raw...)
	softcapForwardF32(capped, cap)
	_, dLogits, err := CrossEntropyBackwardF32Auto(capped, targets, rows, vocab)
	if err != nil {
		t.Fatalf("CE backward: %v", err)
	}
	softcapBackwardScaleF32(dLogits, capped, cap)

	const eps = 1.0 / 512
	for i := range raw {
		up := append([]float32(nil), raw...)
		dn := append([]float32(nil), raw...)
		up[i] += eps
		dn[i] -= eps
		fd := (lossAt(up) - lossAt(dn)) / (2 * eps)
		got := float64(dLogits[i])
		if math.Abs(got-fd) > 2e-2*(1+math.Abs(fd)) {
			t.Fatalf("dRaw[%d] analytic %.6f vs FD %.6f — out of band", i, got, fd)
		}
	}
}

// TestSoftcapBackwardScaleF32_Ugly pins the saturation edge: a deeply-capped logit (|raw| >> cap,
// capped ≈ ±cap) has derivative ≈ 0 — the gradient through a saturated cap must vanish, not blow up.
func TestSoftcapBackwardScaleF32_Ugly(t *testing.T) {
	capped := []float32{29.9999, -29.9999}
	d := []float32{1, 1}
	softcapBackwardScaleF32(d, capped, 30)
	for i := range d {
		if math.Abs(float64(d[i])) > 1e-3 {
			t.Fatalf("saturated derivative [%d] = %v, want ~0", i, d[i])
		}
	}
}
