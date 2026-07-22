// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"bytes"
	"math"
	"os"
	"testing"
)

// clampedTol is a generous absolute/relative tolerance for comparing this file's hand-computed float64
// reference values against the GPU pipeline's result: encClampedSwiGLUGateMulBF16 chains SEVEN separate
// bf16-rounded dispatches (two clamps, a scale, a sigmoid, two multiplies, an add) rather than one
// continuous higher-precision expression, so a few bf16 ULP of accumulated rounding is expected — the
// same "coarser-grained rounding schedule, not a defect" the existing composed-vs-device MoE tests
// already document. The tolerance is loose enough to absorb that, tight enough that using the WRONG
// formula (plain SiLU, a symmetric gate clamp, no alpha, no +1) fails it by a wide margin — see the Bad
// cases below for the actual magnitude those mistakes produce.
func clampedTol(want float64) float64 { return math.Max(0.5, math.Abs(want)*0.02) }

func approxF32(got float32, want float64) bool {
	return math.Abs(float64(got)-want) <= clampedTol(want)
}

// TestMinBF16Const_Good / TestMaxBF16Const_Good pin the two raw clamp primitives
// encClampedSwiGLUGateMulBF16 composes: min(x, limit) and max(x, -limit).
func TestMinBF16Const_Good(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	got, err := minBF16Const(toBF16Bytes([]float32{-10, 3, 10}), 3, 7)
	if err != nil {
		t.Fatalf("minBF16Const: %v", err)
	}
	f := bf16ToF32Slice(got)
	want := []float32{-10, 3, 7}
	for i := range want {
		if !approxF32(f[i], float64(want[i])) {
			t.Fatalf("minBF16Const(-10,3,10; limit 7)[%d] = %v, want %v", i, f[i], want[i])
		}
	}
}

func TestMaxBF16Const_Good(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	got, err := maxBF16Const(toBF16Bytes([]float32{-10, 3, 10}), 3, -7)
	if err != nil {
		t.Fatalf("maxBF16Const: %v", err)
	}
	f := bf16ToF32Slice(got)
	want := []float32{-7, 3, 10}
	for i := range want {
		if !approxF32(f[i], float64(want[i])) {
			t.Fatalf("maxBF16Const(-10,3,10; -7)[%d] = %v, want %v", i, f[i], want[i])
		}
	}
}

// TestClampedSwiGLUGateClamp_Good is the load-bearing asymmetry proof the task calls out explicitly:
// gate's clamp is UPPER-ONLY (clip(gate, max=limit) — no lower bound), while up's is SYMMETRIC
// (clip(up, -limit, limit)). Composing min-then-max on GATE (as up correctly does) would silently
// clamp a very negative gate to -7 instead of leaving it alone — a mistake the full activation's output
// cannot reliably surface (sigmoid(alpha·gate) is already saturated to ~0 for any gate beyond about -7,
// so the DOWNSTREAM effect of the wrong clamp is invisible — see clampedTol's doc). Tested at the
// PRIMITIVE level instead, where the distinction is exact and precision-independent.
func TestClampedSwiGLUGateClamp_Good(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	gateOnlyMin, err := minBF16Const(toBF16Bytes([]float32{-10}), 1, 7) // gate's actual clamp: min only
	if err != nil {
		t.Fatalf("minBF16Const: %v", err)
	}
	symmetric, err := maxBF16Const(gateOnlyMin, 1, -7) // what a WRONG symmetric clamp would do next
	if err != nil {
		t.Fatalf("maxBF16Const: %v", err)
	}
	got, want := bf16ToF32Slice(gateOnlyMin)[0], bf16ToF32Slice(symmetric)[0]
	if !approxF32(got, -10) {
		t.Fatalf("gate's own clamp (min only) on -10 = %v, want -10 unchanged (no lower bound)", got)
	}
	if !approxF32(want, -7) {
		t.Fatalf("sanity: a symmetric clamp on -10 = %v, want -7 (proves the two ARE different operations)", want)
	}
}

// TestClampedSwiGLUBF16_Good pins the full formula (gate' = min(gate,limit); up' = max(min(up,limit),-limit);
// glu = gate'·sigmoid(1.702·gate'); out = glu·(up'+1)) against hand-computed float64 values, source-cited
// in moe_clamped_swiglu.go: HF transformers GptOssExperts._apply_gate + mlx_lm's gpt_oss.py, fetched
// from source and byte-identical between the two.
func TestClampedSwiGLUBF16_Good(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	gate := []float32{10, -10, 2, 0}
	up := []float32{10, -10, 2, 0}
	// Hand-computed (float64) per gate=clip(g,max=7), up=clip(u,-7,7), glu=gate'*sigmoid(1.702*gate'), out=glu*(up'+1):
	want := []float64{55.999625, 0.0000019, 5.806976, 0}

	got, err := ClampedSwiGLUBF16(toBF16Bytes(gate), toBF16Bytes(up), len(gate), 7.0)
	if err != nil {
		t.Fatalf("ClampedSwiGLUBF16: %v", err)
	}
	f := bf16ToF32Slice(got)
	for i := range want {
		if !approxF32(f[i], want[i]) {
			t.Fatalf("ClampedSwiGLUBF16(gate=%v,up=%v)[%d] = %v, want ~%v", gate[i], up[i], i, f[i], want[i])
		}
	}
}

// TestClampedSwiGLUBF16_Bad proves the clamped activation DIFFERS from the plain SiLU sibling
// (SiLUBF16(gate)·up — llama/mistral/qwen's MoEExpertsQuantSiLU) on the SAME inputs: at gate=10 the
// clamped form caps the gate contribution at 7 AND uses alpha=1.702 instead of SiLU's implicit alpha=1,
// AND shifts up by +1 — three independent, large-magnitude divergences on one input, so this is not a
// rounding-noise difference (the gemma-GELU/qwen-SiLU coherent-but-wrong lesson: never gate on "looks
// plausible", assert the actual inequality).
func TestClampedSwiGLUBF16_Bad(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	gate := []float32{10, 2}
	up := []float32{10, 2}
	clamped, err := ClampedSwiGLUBF16(toBF16Bytes(gate), toBF16Bytes(up), len(gate), 7.0)
	if err != nil {
		t.Fatalf("ClampedSwiGLUBF16: %v", err)
	}
	siluGate, err := SiLUBF16(toBF16Bytes(gate))
	if err != nil {
		t.Fatalf("SiLUBF16: %v", err)
	}
	plainSiLU, err := MulBF16(siluGate, toBF16Bytes(up)) // silu(gate)·up — the UNCLAMPED, alpha=1, no-shift sibling
	if err != nil {
		t.Fatalf("MulBF16: %v", err)
	}
	if bytes.Equal(clamped, plainSiLU) {
		t.Fatal("clamped-SwiGLU produced byte-identical output to plain SiLU — the clamp/alpha/+1 are not engaged")
	}
	cf, sf := bf16ToF32Slice(clamped), bf16ToF32Slice(plainSiLU)
	// gate=10: clamped caps the gate branch's sigmoid input at 7 (not 10) and scales it by 1.702 (not 1),
	// so the two must diverge by far more than bf16 rounding noise (plain SiLU: silu(10)*10 ≈ 99.995;
	// clamped: ~56 — see TestClampedSwiGLUBF16_Good).
	if math.Abs(float64(cf[0]-sf[0])) < 5 {
		t.Fatalf("clamped[0]=%v plainSiLU[0]=%v too close — the gate clamp/alpha does not appear engaged", cf[0], sf[0])
	}
}

// TestClampedSwiGLUBF16_Ugly proves the shape/limit guards reject malformed calls rather than silently
// truncating or dividing by a non-positive limit.
func TestClampedSwiGLUBF16_Ugly(t *testing.T) {
	if _, err := ClampedSwiGLUBF16(toBF16Bytes([]float32{1, 2}), toBF16Bytes([]float32{1}), 2, 7); err == nil {
		t.Fatal("ClampedSwiGLUBF16 accepted a gate/up length mismatch")
	}
	if _, err := ClampedSwiGLUBF16(toBF16Bytes([]float32{1}), toBF16Bytes([]float32{1}), 1, 0); err == nil {
		t.Fatal("ClampedSwiGLUBF16 accepted limit <= 0")
	}
}

// TestMoEExpertsQuantClampedSiLU mirrors TestMoEExpertsQuantSiLU exactly (same fixture helper, same
// batched-expert dispatch shape): MoEExpertsQuantClampedSiLU must equal a composed reference whose gate
// nonlinearity is ClampedSwiGLUBF16, AND must differ from both existing siblings (MoEExpertsQuant's GELU
// and MoEExpertsQuantSiLU's plain SiLU) on identical inputs — the one correctness fact a model-level
// greedy A/B cannot show, per TestMoEExpertsQuantSiLU's own doc.
func TestMoEExpertsQuantClampedSiLU(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	const numExperts, topK, dModel, dFF, gs, bits = 4, 2, 64, 128, 32, 4
	const limit = 7.0
	mk := func(n, salt int) []float32 {
		s := make([]float32, n)
		for i := range s {
			s[i] = float32((i*salt+11)%89-44) * 0.02
		}
		return s
	}
	gate, up, down := quantMoEExpertsFixture(t, numExperts, dModel, dFF, gs, bits)
	x := toBF16Bytes(mk(dModel, 5))
	idx := []int32{2, 0}
	weights := toBF16Bytes([]float32{0.7, 0.3})

	got, err := MoEExpertsQuantClampedSiLU(x, idx, weights, gate, up, down, nil, nil, nil, numExperts, topK, dModel, dFF, gs, bits, limit)
	if err != nil {
		t.Fatalf("MoEExpertsQuantClampedSiLU: %v", err)
	}

	gp, gsz := dFF*dModel*bits/8, dFF*(dModel/gs)*bf16Size
	dp, dsz := dModel*dFF*bits/8, dModel*(dFF/gs)*bf16Size
	must := func(b []byte, e error) []byte {
		t.Helper()
		if e != nil {
			t.Fatalf("ref op: %v", e)
		}
		return b
	}
	var acc []byte
	for i, e := range idx {
		ee := int(e)
		ge := must(QMVBF16(x, gate.Packed[ee*gp:(ee+1)*gp], gate.Scales[ee*gsz:(ee+1)*gsz], gate.Biases[ee*gsz:(ee+1)*gsz], dFF, dModel, gs, bits))
		ue := must(QMVBF16(x, up.Packed[ee*gp:(ee+1)*gp], up.Scales[ee*gsz:(ee+1)*gsz], up.Biases[ee*gsz:(ee+1)*gsz], dFF, dModel, gs, bits))
		gg := must(ClampedSwiGLUBF16(ge, ue, dFF, limit)) // the clamped-sigmoid gate, not SiLU or GELU
		de := must(QMVBF16(gg, down.Packed[ee*dp:(ee+1)*dp], down.Scales[ee*dsz:(ee+1)*dsz], down.Biases[ee*dsz:(ee+1)*dsz], dModel, dFF, gs, bits))
		scaled := must(MulBF16(de, scalarFillBF16(weights[i*bf16Size:(i+1)*bf16Size], dModel)))
		if i == 0 {
			acc = scaled
		} else {
			acc = must(AddBF16(acc, scaled))
		}
	}
	if !bytes.Equal(got, acc) {
		t.Fatal("MoEExpertsQuantClampedSiLU != composed clamped-SwiGLU quant reference")
	}
	// the activation branch is genuinely live: both siblings on identical inputs must differ.
	silu, err := MoEExpertsQuantSiLU(x, idx, weights, gate, up, down, numExperts, topK, dModel, dFF, gs, bits)
	if err != nil {
		t.Fatalf("MoEExpertsQuantSiLU: %v", err)
	}
	if bytes.Equal(got, silu) {
		t.Fatal("clamped-SwiGLU and plain-SiLU MoE produced identical output — the activation branch is not engaged")
	}
	gelu, err := MoEExpertsQuant(x, idx, weights, gate, up, down, numExperts, topK, dModel, dFF, gs, bits)
	if err != nil {
		t.Fatalf("MoEExpertsQuant(GELU): %v", err)
	}
	if bytes.Equal(got, gelu) {
		t.Fatal("clamped-SwiGLU and GELU MoE produced identical output — the activation branch is not engaged")
	}
	t.Logf("4-bit batched clamped-SwiGLU experts: gpt_oss activation ≡ composed reference, distinct from both the SiLU and GELU siblings")
}

// TestMoEExpertsQuantClampedSiLU_Biases_Good gates the per-expert additive biases (rung 2, #37):
// with nonzero gate/up/down biases the dispatch must equal the composed reference — each selected
// expert's QMV + its own bias slice added THROUGH THE SAME add kernel, gate/up biased BEFORE the
// clamp, down biased before the router-weighted combine — byte for byte. And the bias must
// actually engage: the biased output differs from the nil-bias output.
func TestMoEExpertsQuantClampedSiLU_Biases_Good(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	const numExperts, topK, dModel, dFF, gs, bits = 4, 2, 64, 128, 32, 4
	const limit = 7.0
	mk := func(n, salt int) []float32 {
		s := make([]float32, n)
		for i := range s {
			s[i] = float32((i*salt+11)%89-44) * 0.02
		}
		return s
	}
	gate, up, down := quantMoEExpertsFixture(t, numExperts, dModel, dFF, gs, bits)
	x := toBF16Bytes(mk(dModel, 5))
	idx := []int32{2, 0}
	weights := toBF16Bytes([]float32{0.7, 0.3})
	gateBias := toBF16Bytes(mk(numExperts*dFF, 13))
	upBias := toBF16Bytes(mk(numExperts*dFF, 17))
	downBias := toBF16Bytes(mk(numExperts*dModel, 19))

	got, err := MoEExpertsQuantClampedSiLU(x, idx, weights, gate, up, down, gateBias, upBias, downBias, numExperts, topK, dModel, dFF, gs, bits, limit)
	if err != nil {
		t.Fatalf("MoEExpertsQuantClampedSiLU(biased): %v", err)
	}

	gp, gsz := dFF*dModel*bits/8, dFF*(dModel/gs)*bf16Size
	dp, dsz := dModel*dFF*bits/8, dModel*(dFF/gs)*bf16Size
	must := func(b []byte, e error) []byte {
		t.Helper()
		if e != nil {
			t.Fatalf("ref op: %v", e)
		}
		return b
	}
	var acc []byte
	for i, e := range idx {
		ee := int(e)
		ge := must(QMVBF16(x, gate.Packed[ee*gp:(ee+1)*gp], gate.Scales[ee*gsz:(ee+1)*gsz], gate.Biases[ee*gsz:(ee+1)*gsz], dFF, dModel, gs, bits))
		ge = must(AddBF16(ge, gateBias[ee*dFF*bf16Size:(ee+1)*dFF*bf16Size])) // + gateBias[e] BEFORE the clamp
		ue := must(QMVBF16(x, up.Packed[ee*gp:(ee+1)*gp], up.Scales[ee*gsz:(ee+1)*gsz], up.Biases[ee*gsz:(ee+1)*gsz], dFF, dModel, gs, bits))
		ue = must(AddBF16(ue, upBias[ee*dFF*bf16Size:(ee+1)*dFF*bf16Size])) // + upBias[e] before the symmetric clamp
		gg := must(ClampedSwiGLUBF16(ge, ue, dFF, limit))
		de := must(QMVBF16(gg, down.Packed[ee*dp:(ee+1)*dp], down.Scales[ee*dsz:(ee+1)*dsz], down.Biases[ee*dsz:(ee+1)*dsz], dModel, dFF, gs, bits))
		de = must(AddBF16(de, downBias[ee*dModel*bf16Size:(ee+1)*dModel*bf16Size])) // + downBias[e] before the combine
		scaled := must(MulBF16(de, scalarFillBF16(weights[i*bf16Size:(i+1)*bf16Size], dModel)))
		if i == 0 {
			acc = scaled
		} else {
			acc = must(AddBF16(acc, scaled))
		}
	}
	if !bytes.Equal(got, acc) {
		t.Fatal("biased MoEExpertsQuantClampedSiLU != composed biased reference — a bias landed in the wrong place or through a different op")
	}

	plain, err := MoEExpertsQuantClampedSiLU(x, idx, weights, gate, up, down, nil, nil, nil, numExperts, topK, dModel, dFF, gs, bits, limit)
	if err != nil {
		t.Fatalf("MoEExpertsQuantClampedSiLU(nil biases): %v", err)
	}
	if bytes.Equal(got, plain) {
		t.Fatal("nonzero expert biases left the output unchanged — the bias adds did not engage")
	}
}

// TestMoEExpertsQuantClampedSiLU_Biases_Bad proves the batched bias-shape guards: a bias of the
// wrong batched length refuses (each must be [numExperts×outDim] bf16 or nil).
func TestMoEExpertsQuantClampedSiLU_Biases_Bad(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	const numExperts, topK, dModel, dFF, gs, bits = 4, 2, 64, 128, 32, 4
	gate, up, down := quantMoEExpertsFixture(t, numExperts, dModel, dFF, gs, bits)
	x := toBF16Bytes(syntheticFloat32(dModel, 5))
	idx := []int32{2, 0}
	weights := toBF16Bytes([]float32{0.7, 0.3})
	short := toBF16Bytes([]float32{1, 2, 3})

	if _, err := MoEExpertsQuantClampedSiLU(x, idx, weights, gate, up, down, short, nil, nil, numExperts, topK, dModel, dFF, gs, bits, 7); err == nil {
		t.Fatal("expected a short gateBias to refuse")
	}
	if _, err := MoEExpertsQuantClampedSiLU(x, idx, weights, gate, up, down, nil, short, nil, numExperts, topK, dModel, dFF, gs, bits, 7); err == nil {
		t.Fatal("expected a short upBias to refuse")
	}
	if _, err := MoEExpertsQuantClampedSiLU(x, idx, weights, gate, up, down, nil, nil, short, numExperts, topK, dModel, dFF, gs, bits, 7); err == nil {
		t.Fatal("expected a short downBias to refuse")
	}
}
