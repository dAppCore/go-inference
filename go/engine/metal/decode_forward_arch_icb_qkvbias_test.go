// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"os"
	"testing"

	"dappco.re/go/inference/model"
)

// TestDecodeForwardArchICBQuantQKVBias proves the RECORDED ICB quant decode applies Qwen2/2.5's
// additive q/k/v projection bias — the Wall 3 gate. It mirrors TestDecodeForwardArchQuantQKVBias
// (the plain-encode gate) but drives DecodeForwardArchICBQuant (record the arch ICB + replay it),
// so it exercises the bias-add ops woven into the recorder + the per-token V-bias cache-row rebind,
// which the plain-encode path never touches. The host reference shares no emission/kernel code with
// the GPU lanes.
//
//   - Good (parity): ICB-with-bias ≡ host-with-bias to cosine ≥ 0.999 — the bias reaches the recorded
//     hot decode path (right value, right offset, before qk-norm/rope; V-bias rebound to the cache row).
//   - Bad (control): ICB-with-bias must DIVERGE from the no-bias host reference — else the gate is blind
//     to a dropped bias (the pre-fix ICB behaviour, which reached only the plain-encode path).
func TestDecodeForwardArchICBQuantQKVBias(t *testing.T) {
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

	got, err := DecodeForwardArchICBQuant(inputs, ql, specs, dModel, nHeads, nKV, headDim, maxLen, dFF, 0, base, scale, eps, false)
	if err != nil {
		t.Fatalf("DecodeForwardArchICBQuant: %v", err)
	}

	// Good: the recorded ICB forward matches the host reference that applies the same bias.
	want := hostArchQuantReference(t, inputs, ql, specs, dModel, nHeads, nKV, headDim, dFF, 0, base, scale, eps, false)
	for tok := range T {
		cos := cosineBF16(got[tok], want[tok])
		t.Logf("icb with-bias   tok %d cosine=%.6f", tok, cos)
		if cos < 0.999 {
			t.Fatalf("tok %d: ICB quant decode diverges from the biased host reference (cosine=%.6f) — QKV bias mis-applied in the recorder", tok, cos)
		}
	}

	// Bad (control): the ICB (with bias) must NOT match the no-bias reference — the bias must move the hidden.
	qlNoBias := []QuantizedLayerWeights{ql[0]}
	qlNoBias[0].BQ, qlNoBias[0].BK, qlNoBias[0].BV = nil, nil, nil
	wantNoBias := hostArchQuantReference(t, inputs, qlNoBias, specs, dModel, nHeads, nKV, headDim, dFF, 0, base, scale, eps, false)
	anyDiverged := false
	for tok := range T {
		cos := cosineBF16(got[tok], wantNoBias[tok])
		t.Logf("icb no-bias-ref tok %d cosine=%.6f", tok, cos)
		if cos < 0.999 {
			anyDiverged = true
		}
	}
	if !anyDiverged {
		t.Fatal("ICB-with-bias matched the no-bias reference on every token — the bias is not load-bearing in the recorder, so the gate cannot detect a dropped bias")
	}
}
