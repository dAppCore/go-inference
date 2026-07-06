// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"os"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference/model"
)

// archDenseNormRef is an all-owner, all-global dense forward that applies the gemma4
// post-attention and post-feed-forward norms. Its post-norm residual helper mirrors
// encResidualMaybeNorm: when the fused custom kernel is available, production uses the
// fused RMS+Residual numerics to stay byte-equal with ICB replay; otherwise it uses
// the composed RMSNormBF16 then AddBF16 path. (QK-norm is a later slice; this gates
// the two dModel post-norms only.)
func archDenseNormRef(t *testing.T, layers []DecodeLayerWeights, inputs [][]byte, dModel, nHeads, nKV, headDim, dFF, maxLen int, base, scale, eps float32) [][]byte {
	t.Helper()
	qDim, kvDim := nHeads*headDim, nKV*headDim
	rowBytes := kvDim * bf16Size
	nL, T := len(layers), len(inputs)
	must := func(b []byte, err error) []byte {
		if err != nil {
			t.Fatalf("archDenseNormRef op: %v", err)
		}
		return b
	}
	residualMaybeNorm := func(res, branch, norm []byte) []byte {
		if norm == nil {
			return must(AddBF16(res, branch))
		}
		if gpuHasGeluKernel() {
			return must(RMSNormResidualBF16(branch, norm, res, dModel, eps))
		}
		return must(AddBF16(res, must(RMSNormBF16(branch, norm, 1, dModel, eps))))
	}
	kC := make([][]byte, nL)
	vC := make([][]byte, nL)
	for li := range layers {
		kC[li] = make([]byte, maxLen*rowBytes)
		vC[li] = make([]byte, maxLen*rowBytes)
	}
	out := make([][]byte, T)
	for tok := range T {
		x := inputs[tok]
		for li := range nL {
			w := layers[li]
			normed := must(RMSNormBF16(x, w.AttnNormW, 1, dModel, eps))
			q := must(MatVecBF16(w.WQ, normed, qDim, dModel))
			if w.QNormW != nil { // gemma4 per-head QK-norm before RoPE (rows = nHeads)
				q = must(RMSNormBF16(q, w.QNormW, nHeads, headDim, eps))
			}
			qr := must(RoPEBF16(q, 1, nHeads, headDim, base, scale, tok, false))
			k := must(MatVecBF16(w.WK, normed, kvDim, dModel))
			if w.KNormW != nil {
				k = must(RMSNormBF16(k, w.KNormW, nKV, headDim, eps))
			}
			knew := must(RoPEBF16(k, 1, nKV, headDim, base, scale, tok, false))
			vnew := must(MatVecBF16(w.WV, normed, kvDim, dModel))
			copy(kC[li][tok*rowBytes:(tok+1)*rowBytes], knew)
			copy(vC[li][tok*rowBytes:(tok+1)*rowBytes], vnew)
			n := tok + 1
			attn := must(SDPA(qr, seqToHeadMajor(kC[li], nKV, headDim, n), seqToHeadMajor(vC[li], nKV, headDim, n), 1, nHeads, nKV, headDim, n, scale))
			wo := must(MatVecBF16(w.WO, attn, dModel, qDim))
			h := residualMaybeNorm(x, wo, w.PostAttnNormW)
			mlpNormed := must(RMSNormBF16(h, w.MLPNormW, 1, dModel, eps))
			ff := must(mlpTransformBF16(mlpNormed, w.WGate, w.WUp, w.WDown, dModel, dFF))
			x = residualMaybeNorm(h, ff, w.PostFFNormW)
		}
		out[tok] = x
	}
	return out
}

// TestDecodePostNorms gates the gemma4 post-attention + post-feed-forward norm wiring:
// a re-encode arch forward with the two norms set is byte-for-byte the reference that
// applies them under the production fused/composed residual-norm semantics, AND differs
// from the same forward with the norms dropped (the norms are genuinely live, not ignored).
func TestDecodePostNorms(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	const dModel, nHeads, nKV, headDim, dFF = 512, 8, 4, 64, 1024
	const base, scale, eps = float32(10000), float32(0.125), float32(1e-5)
	const T, maxLen, nL = 4, 8, 3

	inputs := make([][]byte, T)
	for i := range inputs {
		f := make([]float32, dModel)
		for j := range f {
			f[j] = float32((j*(i+3)+5)%97-48) * 0.02
		}
		inputs[i] = toBF16Bytes(f)
	}
	normW := func(salt int) []byte {
		f := make([]float32, dModel)
		for j := range f {
			f[j] = float32((j*salt+3)%29-14) * 0.03
		}
		return toBF16Bytes(f)
	}
	layers := make([]DecodeLayerWeights, nL)
	types := make([]string, nL)
	for li := range layers {
		layers[li] = forwardLayer(dModel, nHeads, nKV, headDim, dFF, (li+1)*100)
		layers[li].PostAttnNormW = normW(li*7 + 1)
		layers[li].PostFFNormW = normW(li*7 + 2)
		types[li] = "full_attention"
	}
	specs := model.DeriveLayers(types, 0)

	got, err := DecodeForwardArch(inputs, layers, specs, dModel, nHeads, nKV, headDim, maxLen, dFF, 0, base, scale, eps, false)
	if err != nil {
		t.Fatalf("DecodeForwardArch with post-norms: %v", err)
	}
	want := archDenseNormRef(t, layers, inputs, dModel, nHeads, nKV, headDim, dFF, maxLen, base, scale, eps)
	for tok := range T {
		eqBytes(t, core.Sprintf("post-norm forward vs ref tok%d", tok), got[tok], want[tok])
	}

	// non-vacuous: dropping the post-norms changes the output (they are genuinely live).
	bare := make([]DecodeLayerWeights, nL)
	copy(bare, layers)
	for li := range bare {
		bare[li].PostAttnNormW = nil
		bare[li].PostFFNormW = nil
	}
	gotBare, err := DecodeForwardArch(inputs, bare, specs, dModel, nHeads, nKV, headDim, maxLen, dFF, 0, base, scale, eps, false)
	if err != nil {
		t.Fatalf("DecodeForwardArch bare: %v", err)
	}
	if !lastTokenDiffers(got, gotBare) {
		t.Fatal("post-norms made no difference to the output — the norms were not applied")
	}
	t.Logf("gemma4 post-norms: re-encode forward with post-attn + post-FF ≡ composed reference, and differs from without (norms live)")
}

// TestDecodeQKNorm gates the per-head QK-norm: a re-encode forward with q_norm/k_norm set
// (applied per attention head, headDim-wide, before RoPE) is byte-for-byte the reference
// that does the same, and differs from the same forward with QK-norm dropped.
func TestDecodeQKNorm(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	const dModel, nHeads, nKV, headDim, dFF = 512, 8, 4, 64, 1024
	const base, scale, eps = float32(10000), float32(0.125), float32(1e-5)
	const T, maxLen, nL = 4, 8, 3

	inputs := make([][]byte, T)
	for i := range inputs {
		f := make([]float32, dModel)
		for j := range f {
			f[j] = float32((j*(i+2)+7)%89-44) * 0.02
		}
		inputs[i] = toBF16Bytes(f)
	}
	headNormW := func(salt int) []byte { // a [headDim] q/k-norm weight
		f := make([]float32, headDim)
		for j := range f {
			f[j] = float32((j*salt+5)%23-11) * 0.04
		}
		return toBF16Bytes(f)
	}
	layers := make([]DecodeLayerWeights, nL)
	types := make([]string, nL)
	for li := range layers {
		layers[li] = forwardLayer(dModel, nHeads, nKV, headDim, dFF, (li+1)*100)
		layers[li].QNormW = headNormW(li*5 + 1)
		layers[li].KNormW = headNormW(li*5 + 2)
		types[li] = "full_attention"
	}
	specs := model.DeriveLayers(types, 0)

	got, err := DecodeForwardArch(inputs, layers, specs, dModel, nHeads, nKV, headDim, maxLen, dFF, 0, base, scale, eps, false)
	if err != nil {
		t.Fatalf("DecodeForwardArch with QK-norm: %v", err)
	}
	want := archDenseNormRef(t, layers, inputs, dModel, nHeads, nKV, headDim, dFF, maxLen, base, scale, eps)
	for tok := range T {
		eqBytes(t, core.Sprintf("QK-norm forward vs ref tok%d", tok), got[tok], want[tok])
	}

	bare := make([]DecodeLayerWeights, nL)
	copy(bare, layers)
	for li := range bare {
		bare[li].QNormW = nil
		bare[li].KNormW = nil
	}
	gotBare, err := DecodeForwardArch(inputs, bare, specs, dModel, nHeads, nKV, headDim, maxLen, dFF, 0, base, scale, eps, false)
	if err != nil {
		t.Fatalf("DecodeForwardArch bare: %v", err)
	}
	if !lastTokenDiffers(got, gotBare) {
		t.Fatal("QK-norm made no difference — the per-head norm was not applied")
	}
	t.Logf("gemma4 QK-norm: per-head RMSNorm on Q/K before RoPE ≡ composed reference (RMSNormBF16 rows=nHeads), and differs from without (live); the re-encode dense path is now gemma4-norm-complete")
}
