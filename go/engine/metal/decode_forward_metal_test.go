// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64 && metal_runtime

package native

import (
	"os"
	"testing"

	core "dappco.re/go"
)

// quantW and buildQuantLayer (this file's synthetic quantised-layer builders) now live in
// test_helpers_test.go, reimplemented in pure Go (no cgo/metal) — they are shared by several other
// untagged test files across the package (backend_test.go, decode_forward_arch_quant_test.go, and
// others), so they can't depend on the metal_runtime lane.

// quantRefForward is the oracle: the same N-layer × T-token growing-cache forward
// composed from the parity-proven STANDALONE ops (QMVBF16 projections on the same
// packed bytes, RMSNormBF16, RoPEBF16, head-major SDPA on the assembled window,
// AddBF16, GeluGateMulBF16). It mirrors encAttnHalfKV ▸ encMLPHalfBF16 op-for-op,
// so DecodeForwardQuant must equal it byte-for-byte.
func quantRefForward(t *testing.T, ql []QuantizedLayerWeights, inputs [][]byte, dModel, nHeads, nKV, headDim, dFF, maxLen int, base, scale, eps float32) [][]byte {
	t.Helper()
	qDim, kvDim := nHeads*headDim, nKV*headDim
	rowBytes := kvDim * bf16Size
	nLayers, T := len(ql), len(inputs)
	gs, bits := ql[0].GroupSize, ql[0].Bits
	kC := make([][]byte, nLayers)
	vC := make([][]byte, nLayers)
	for l := range kC {
		kC[l] = make([]byte, maxLen*rowBytes)
		vC[l] = make([]byte, maxLen*rowBytes)
	}
	qmv := func(x, w QuantWeight, vec []byte, outDim, inDim int) []byte {
		o, err := QMVBF16(vec, w.Packed, w.Scales, w.Biases, outDim, inDim, gs, bits)
		if err != nil {
			t.Fatalf("QMVBF16: %v", err)
		}
		return o
	}
	must := func(b []byte, err error) []byte {
		if err != nil {
			t.Fatalf("ref op: %v", err)
		}
		return b
	}
	out := make([][]byte, T)
	for tok := 0; tok < T; tok++ {
		x := inputs[tok]
		for l := 0; l < nLayers; l++ {
			w := ql[l]
			// attention half
			normed := must(RMSNormBF16(x, w.AttnNormW, 1, dModel, eps))
			qr := must(RoPEBF16(qmv(QuantWeight{}, w.Q, normed, qDim, dModel), 1, nHeads, headDim, base, scale, tok, false))
			knew := must(RoPEBF16(qmv(QuantWeight{}, w.K, normed, kvDim, dModel), 1, nKV, headDim, base, scale, tok, false))
			vnew := qmv(QuantWeight{}, w.V, normed, kvDim, dModel)
			copy(kC[l][tok*rowBytes:(tok+1)*rowBytes], knew)
			copy(vC[l][tok*rowBytes:(tok+1)*rowBytes], vnew)
			L := tok + 1
			attn := must(SDPA(qr, seqToHeadMajor(kC[l], nKV, headDim, L), seqToHeadMajor(vC[l], nKV, headDim, L), 1, nHeads, nKV, headDim, L, scale))
			h := must(AddBF16(x, qmv(QuantWeight{}, w.O, attn, dModel, qDim)))
			// MLP half
			mlpNormed := must(RMSNormBF16(h, w.MLPNormW, 1, dModel, eps))
			gg := must(GeluGateMulBF16(qmv(QuantWeight{}, w.Gate, mlpNormed, dFF, dModel), qmv(QuantWeight{}, w.Up, mlpNormed, dFF, dModel)))
			x = must(AddBF16(h, qmv(QuantWeight{}, w.Down, gg, dModel, dFF)))
		}
		out[tok] = x
	}
	return out
}

// TestDecodeForwardQuant gates the 4-bit-quantised forward against the composed
// proven ops: a 2-layer × 3-token growing-cache forward with affine_qmv_bfloat16_t
// projections must equal quantRefForward byte-for-byte (GQA 8/4). This is the whole
// 4-bit decode path verified end to end with no mlx-c on the runtime path.
func TestDecodeForwardQuant(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	const dModel, nHeads, nKV, headDim, dFF, gs, bits = 512, 8, 4, 64, 1024, 64, 4
	const base, scale, eps = float32(10000), float32(0.125), float32(1e-5)
	const nLayers, T, maxLen = 2, 3, 8

	ql := make([]QuantizedLayerWeights, nLayers)
	for l := range ql {
		ql[l] = buildQuantLayer(t, dModel, nHeads, nKV, headDim, dFF, gs, bits, (l+1)*100)
	}
	inputs := make([][]byte, T)
	for i := range inputs {
		f := make([]float32, dModel)
		for j := range f {
			f[j] = float32((j*(i+3)+5)%97-48) * 0.02
		}
		inputs[i] = toBF16Bytes(f)
	}

	got, err := DecodeForwardQuant(inputs, ql, dModel, nHeads, nKV, headDim, maxLen, dFF, base, scale, eps)
	if err != nil {
		t.Fatalf("DecodeForwardQuant: %v", err)
	}
	ref := quantRefForward(t, ql, inputs, dModel, nHeads, nKV, headDim, dFF, maxLen, base, scale, eps)
	for tok := 0; tok < T; tok++ {
		eqBytes(t, core.Sprintf("DecodeForwardQuant tok%d", tok), got[tok], ref[tok])
	}
	t.Logf("DecodeForwardQuant(%d layers × %d tokens, 4-bit gs%d, GQA %d/%d, growing cache): byte-identical to composed proven ops — whole 4-bit decode off mlx-c", nLayers, T, gs, nHeads, nKV)
}

// TestDecodeForwardICBQuant gates the stacked quant-ICB: replaying the recorded
// N-layer 4-bit stack per token (bumping offBuf/nBuf + each layer's K-rope and
// V-qmv cache-write offsets) must equal the proven re-encode DecodeForwardQuant
// byte-for-byte, over a growing cache. 1 and 3 layers (per-layer rebind +
// cross-layer residual ping-pong), GQA 8/4, 4-bit gs64.
func TestDecodeForwardICBQuant(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	const dModel, nHeads, nKV, headDim, dFF, gs, bits = 512, 8, 4, 64, 1024, 64, 4
	const base, scale, eps = float32(10000), float32(0.125), float32(1e-5)
	const T, maxLen = 5, 8

	for _, nLayers := range []int{1, 3} {
		ql := make([]QuantizedLayerWeights, nLayers)
		for l := range ql {
			ql[l] = buildQuantLayer(t, dModel, nHeads, nKV, headDim, dFF, gs, bits, (l+1)*100)
		}
		inputs := make([][]byte, T)
		for i := range inputs {
			f := make([]float32, dModel)
			for j := range f {
				f[j] = float32((j*(i+3)+5)%97-48) * 0.02
			}
			inputs[i] = toBF16Bytes(f)
		}

		ref, err := DecodeForwardQuant(inputs, ql, dModel, nHeads, nKV, headDim, maxLen, dFF, base, scale, eps)
		if err != nil {
			t.Fatalf("DecodeForwardQuant (%d layers): %v", nLayers, err)
		}
		got, err := DecodeForwardICBQuant(inputs, ql, dModel, nHeads, nKV, headDim, maxLen, dFF, base, scale, eps)
		if err != nil {
			t.Fatalf("DecodeForwardICBQuant (%d layers): %v", nLayers, err)
		}
		for tok := 0; tok < T; tok++ {
			eqBytes(t, core.Sprintf("DecodeForwardICBQuant L%d tok%d", nLayers, tok), got[tok], ref[tok])
		}
		t.Logf("DecodeForwardICBQuant(%d layers × %d tokens, 4-bit, growing cache): byte-identical to re-encode DecodeForwardQuant — both levers stacked, off mlx-c", nLayers, T)
	}
}
