// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"os"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference/model"
)

// archValueNormRef is the oracle for gemma4's value normalisation + K==V, built from the
// parity-proven value ops. It mirrors archDenseNormRef but: (valueNorm) applies a no-scale
// per-head RMSNorm to V (metal's RMSNormNoScale, expressed as RMSNormBF16 with a ones
// weight), and (kEqV) takes V from the k-proj weight rather than a v_proj — exactly the two
// things encAttnHalfKV does for gemma4. All-owner, all-global dense.
func archValueNormRef(t *testing.T, layers []DecodeLayerWeights, inputs [][]byte, dModel, nHeads, nKV, headDim, dFF, maxLen int, base, scale, eps float32, valueNorm, kEqV bool) [][]byte {
	t.Helper()
	qDim, kvDim := nHeads*headDim, nKV*headDim
	rowBytes := kvDim * bf16Size
	nL, T := len(layers), len(inputs)
	onesF := make([]float32, headDim)
	for i := range onesF {
		onesF[i] = 1
	}
	onesW := toBF16Bytes(onesF) // no-scale value norm = (x/rms)·1
	must := func(b []byte, err error) []byte {
		if err != nil {
			t.Fatalf("archValueNormRef op: %v", err)
		}
		return b
	}
	kC := make([][]byte, nL)
	vC := make([][]byte, nL)
	for li := range layers {
		kC[li] = make([]byte, maxLen*rowBytes)
		vC[li] = make([]byte, maxLen*rowBytes)
	}
	out := make([][]byte, T)
	for tok := 0; tok < T; tok++ {
		x := inputs[tok]
		for li := 0; li < nL; li++ {
			w := layers[li]
			normed := must(RMSNormBF16(x, w.AttnNormW, 1, dModel, eps))
			q := must(MatVecBF16(w.WQ, normed, qDim, dModel))
			if w.QNormW != nil {
				q = must(RMSNormBF16(q, w.QNormW, nHeads, headDim, eps))
			}
			qr := must(RoPEBF16(q, 1, nHeads, headDim, base, scale, tok, false))
			k := must(MatVecBF16(w.WK, normed, kvDim, dModel))
			if w.KNormW != nil {
				k = must(RMSNormBF16(k, w.KNormW, nKV, headDim, eps))
			}
			knew := must(RoPEBF16(k, 1, nKV, headDim, base, scale, tok, false))
			vW := w.WV
			if kEqV { // gemma4 K==V: V is the k-proj output (pre-knorm/rope), value-normed
				vW = w.WK
			}
			vnew := must(MatVecBF16(vW, normed, kvDim, dModel))
			if valueNorm {
				vnew = must(RMSNormBF16(vnew, onesW, nKV, headDim, eps))
			}
			copy(kC[li][tok*rowBytes:(tok+1)*rowBytes], knew)
			copy(vC[li][tok*rowBytes:(tok+1)*rowBytes], vnew)
			n := tok + 1
			attn := must(SDPA(qr, seqToHeadMajor(kC[li], nKV, headDim, n), seqToHeadMajor(vC[li], nKV, headDim, n), 1, nHeads, nKV, headDim, n, scale))
			wo := must(MatVecBF16(w.WO, attn, dModel, qDim))
			if w.PostAttnNormW != nil {
				wo = must(RMSNormBF16(wo, w.PostAttnNormW, 1, dModel, eps))
			}
			h := must(AddBF16(x, wo))
			mlpNormed := must(RMSNormBF16(h, w.MLPNormW, 1, dModel, eps))
			ff := must(mlpTransformBF16(mlpNormed, w.WGate, w.WUp, w.WDown, dModel, dFF))
			if w.PostFFNormW != nil {
				ff = must(RMSNormBF16(ff, w.PostFFNormW, 1, dModel, eps))
			}
			x = must(AddBF16(h, ff))
		}
		out[tok] = x
	}
	return out
}

func valueNormInputs(dModel, T int) [][]byte {
	inputs := make([][]byte, T)
	for i := range inputs {
		f := make([]float32, dModel)
		for j := range f {
			f[j] = float32((j*(i+4)+9)%83-41) * 0.02
		}
		inputs[i] = toBF16Bytes(f)
	}
	return inputs
}

// TestValueNorm gates the value normalisation: a re-encode forward with valueNorm set
// is byte-for-byte the reference that applies a no-scale per-head RMSNorm to V, AND differs
// from the same forward without it (value-norm is genuinely live, not ignored).
func TestValueNorm(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	const dModel, nHeads, nKV, headDim, dFF = 512, 8, 4, 64, 1024
	const base, scale, eps = float32(10000), float32(1.0), float32(1e-5) // gemma4 SDPA scale = 1.0
	const T, maxLen, nL = 4, 8, 3

	inputs := valueNormInputs(dModel, T)
	layers := make([]DecodeLayerWeights, nL)
	types := make([]string, nL)
	for li := range layers {
		layers[li] = forwardLayer(dModel, nHeads, nKV, headDim, dFF, (li+1)*100)
		types[li] = "full_attention"
	}
	specs := model.DeriveLayers(types, 0)

	got, err := DecodeForwardArch(inputs, layers, specs, dModel, nHeads, nKV, headDim, maxLen, dFF, 0, base, scale, eps, true)
	if err != nil {
		t.Fatalf("DecodeForwardArch valueNorm: %v", err)
	}
	want := archValueNormRef(t, layers, inputs, dModel, nHeads, nKV, headDim, dFF, maxLen, base, scale, eps, true, false)
	for tok := 0; tok < T; tok++ {
		eqBytes(t, core.Sprintf("value-norm forward vs ref tok%d", tok), got[tok], want[tok])
	}

	gotNo, err := DecodeForwardArch(inputs, layers, specs, dModel, nHeads, nKV, headDim, maxLen, dFF, 0, base, scale, eps, false)
	if err != nil {
		t.Fatalf("DecodeForwardArch no value-norm: %v", err)
	}
	if !lastTokenDiffers(got, gotNo) {
		t.Fatal("value-norm made no difference to the output — the no-scale value RMSNorm was not applied")
	}
	t.Logf("gemma4 value-norm: re-encode forward with the no-scale per-head V RMSNorm ≡ composed reference, and differs from without (live)")
}

// TestAttentionKEqV gates the K==V path (gemma4 12B/31B: attention_k_eq_v, no v_proj): a forward
// whose layers carry NO v_proj weight (V taken from the k-proj via the projector's hasV()==false)
// is byte-for-byte a forward whose v_proj IS the k-proj weight, and byte-for-byte the oracle that
// takes V from the k-proj — both value-normed. Proves V rides the k-proj output, not a separate
// projection, with no model load.
func TestAttentionKEqV(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	const dModel, nHeads, nKV, headDim, dFF = 512, 8, 4, 64, 1024
	const base, scale, eps = float32(10000), float32(1.0), float32(1e-5)
	const T, maxLen, nL = 4, 8, 3

	inputs := valueNormInputs(dModel, T)
	types := make([]string, nL)
	// explicit reference: v_proj weight set EQUAL to k_proj (so V = k-proj output, as K==V does).
	explicit := make([]DecodeLayerWeights, nL)
	for li := range explicit {
		explicit[li] = forwardLayer(dModel, nHeads, nKV, headDim, dFF, (li+1)*100)
		explicit[li].WV = explicit[li].WK
		types[li] = "full_attention"
	}
	specs := model.DeriveLayers(types, 0)

	wantExplicit, err := DecodeForwardArch(inputs, explicit, specs, dModel, nHeads, nKV, headDim, maxLen, dFF, 0, base, scale, eps, true)
	if err != nil {
		t.Fatalf("DecodeForwardArch explicit v=k: %v", err)
	}

	// K==V: drop v_proj entirely; the decode must route V through wK (hasV()==false).
	keqv := make([]DecodeLayerWeights, nL)
	copy(keqv, explicit)
	for li := range keqv {
		keqv[li].WV = nil
	}
	gotKEqV, err := DecodeForwardArch(inputs, keqv, specs, dModel, nHeads, nKV, headDim, maxLen, dFF, 0, base, scale, eps, true)
	if err != nil {
		t.Fatalf("DecodeForwardArch K==V (no v_proj): %v", err)
	}
	for tok := 0; tok < T; tok++ {
		eqBytes(t, core.Sprintf("K==V vs explicit v=k tok%d", tok), gotKEqV[tok], wantExplicit[tok])
	}

	want := archValueNormRef(t, keqv, inputs, dModel, nHeads, nKV, headDim, dFF, maxLen, base, scale, eps, true, true)
	for tok := 0; tok < T; tok++ {
		eqBytes(t, core.Sprintf("K==V vs oracle tok%d", tok), gotKEqV[tok], want[tok])
	}
	t.Logf("gemma4 K==V: no-v_proj forward (V via wK, value-normed) ≡ explicit v_proj=k_proj ≡ composed reference — the 12B/31B attention_k_eq_v path is correct, no model load")
}
