// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"os"
	"testing"

	"dappco.re/go/inference/model"
)

// buildBiasedQwen2QuantLayer is a conditioned quant layer PLUS Qwen2/2.5's distinguishing
// trait — a non-zero additive bias on the q/k/v projections (bias=True) and NONE on o_proj
// (bias=False). The bias magnitude (~±1.1) is comparable to the projection output on the
// conditioned fixture, so a dropped bias moves the hidden well past the 0.999 cosine bar.
func buildBiasedQwen2QuantLayer(t *testing.T, dModel, nHeads, nKV, headDim, dFF, gs, bits, salt int) QuantizedLayerWeights {
	t.Helper()
	ql := buildConditionedQuantLayer(t, dModel, nHeads, nKV, headDim, dFF, gs, bits, salt)
	qDim, kvDim := nHeads*headDim, nKV*headDim
	bias := func(n, s int) []byte {
		f := make([]float32, n)
		for i := range f {
			f[i] = float32((i*s+3)%29-14) * 0.08 // ±~1.1, varied per element
		}
		return toBF16Bytes(f)
	}
	ql.BQ = bias(qDim, salt+91)
	ql.BK = bias(kvDim, salt+97)
	ql.BV = bias(kvDim, salt+103)
	// o_proj carries no additive bias in Qwen2/2.5 — the fixture must not fabricate one.
	return ql
}

func qkvBiasInputs(t *testing.T, n, dModel, salt int) [][]byte {
	t.Helper()
	in := make([][]byte, n)
	for i := range in {
		f := make([]float32, dModel)
		for j := range f {
			f[j] = float32((j*(i+salt)+11)%83-41) * 0.02
		}
		in[i] = toBF16Bytes(f)
	}
	return in
}

// TestDecodeForwardArchQuantQKVBias proves the native quant decode APPLIES Qwen2/2.5's additive
// QKV projection bias, by anchoring it to the independent float host reference (which shares no
// emission or kernel code with the GPU lanes). Two halves:
//
//   - Good (parity): GPU-with-bias ≡ host-with-bias to cosine ≥ 0.999 — the bias is applied, and
//     applied correctly (right value, right offset, before qk-norm/rope).
//   - Bad (load-bearing control): GPU-with-bias must DIVERGE from the host WITHOUT the bias. This
//     is what makes the parity gate meaningful: if the engine dropped the bias (the pre-fix
//     behaviour), the GPU would instead match the no-bias reference and the parity half would fail.
func TestDecodeForwardArchQuantQKVBias(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	const dModel, nHeads, nKV, headDim, dFF, gs, bits = 512, 8, 2, 64, 1024, 64, 4
	const base, scale, eps = float32(10000), float32(0.125), float32(1e-5)
	const maxLen, T = 8, 4

	specs := model.DeriveLayers([]string{"full_attention"}, 0)
	specs[0].KVHeads, specs[0].HeadDim = nKV, headDim
	ql := []QuantizedLayerWeights{buildBiasedQwen2QuantLayer(t, dModel, nHeads, nKV, headDim, dFF, gs, bits, 700)}
	if len(ql[0].BQ) == 0 || len(ql[0].BK) == 0 || len(ql[0].BV) == 0 {
		t.Fatal("fixture lacks QKV bias — the test cannot prove bias application")
	}
	inputs := qkvBiasInputs(t, T, dModel, 7)

	got, err := DecodeForwardArchQuant(inputs, ql, specs, dModel, nHeads, nKV, headDim, maxLen, dFF, 0, base, scale, eps, false)
	if err != nil {
		t.Fatalf("DecodeForwardArchQuant: %v", err)
	}

	// Good: the GPU forward matches the host reference that applies the same bias.
	want := hostArchQuantReference(t, inputs, ql, specs, dModel, nHeads, nKV, headDim, dFF, 0, base, scale, eps, false)
	for tok := range T {
		cos := cosineBF16(got[tok], want[tok])
		t.Logf("with-bias   tok %d cosine=%.6f", tok, cos)
		if cos < 0.999 {
			t.Fatalf("tok %d: GPU quant decode diverges from the biased host reference (cosine=%.6f) — QKV bias mis-applied", tok, cos)
		}
	}

	// Bad (control): clear the bias on a copy and confirm the GPU (with bias) does NOT match the
	// no-bias reference — the bias must actually move the hidden, else the parity above is blind
	// to a dropped bias.
	qlNoBias := []QuantizedLayerWeights{ql[0]}
	qlNoBias[0].BQ, qlNoBias[0].BK, qlNoBias[0].BV = nil, nil, nil
	wantNoBias := hostArchQuantReference(t, inputs, qlNoBias, specs, dModel, nHeads, nKV, headDim, dFF, 0, base, scale, eps, false)
	anyDiverged := false
	for tok := range T {
		cos := cosineBF16(got[tok], wantNoBias[tok])
		t.Logf("no-bias-ref tok %d cosine=%.6f", tok, cos)
		if cos < 0.999 {
			anyDiverged = true
		}
	}
	if !anyDiverged {
		t.Fatal("GPU-with-bias matched the no-bias reference on every token — the bias is not load-bearing, so the parity gate cannot detect a dropped bias")
	}
}
