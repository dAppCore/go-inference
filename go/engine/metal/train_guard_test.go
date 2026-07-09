// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import "testing"

func requireTrainingGuardError(t *testing.T, name string, err error) {
	t.Helper()
	if err == nil {
		t.Fatalf("%s error = nil", name)
	}
}

func TestLoRAInputGuards(t *testing.T) {
	const M, in, out, rank = 2, 3, 4, 2
	x := make([]float32, M*in)
	a := make([]float32, rank*in)
	b := make([]float32, out*rank)
	xA := make([]float32, M*rank)
	dy := make([]float32, M*out)

	_, _, err := LoRAForwardF32(x[:len(x)-1], a, b, M, in, out, rank, 1)
	requireTrainingGuardError(t, "LoRAForwardF32 bad x", err)
	_, _, err = LoRAForwardF32(x, a[:len(a)-1], b, M, in, out, rank, 1)
	requireTrainingGuardError(t, "LoRAForwardF32 bad A", err)
	_, _, err = LoRAForwardF32(x, a, b[:len(b)-1], M, in, out, rank, 1)
	requireTrainingGuardError(t, "LoRAForwardF32 bad B", err)

	_, _, _, err = LoRABackwardF32(dy[:len(dy)-1], x, a, b, xA, M, in, out, rank, 1)
	requireTrainingGuardError(t, "LoRABackwardF32 bad dy", err)
	_, _, _, err = LoRABackwardF32(dy, x[:len(x)-1], a, b, xA, M, in, out, rank, 1)
	requireTrainingGuardError(t, "LoRABackwardF32 bad x", err)
	_, _, _, err = LoRABackwardF32(dy, x, a[:len(a)-1], b, xA, M, in, out, rank, 1)
	requireTrainingGuardError(t, "LoRABackwardF32 bad A", err)
	_, _, _, err = LoRABackwardF32(dy, x, a, b[:len(b)-1], xA, M, in, out, rank, 1)
	requireTrainingGuardError(t, "LoRABackwardF32 bad B", err)
	_, _, _, err = LoRABackwardF32(dy, x, a, b, xA[:len(xA)-1], M, in, out, rank, 1)
	requireTrainingGuardError(t, "LoRABackwardF32 bad xA", err)
}

func TestTrainingBackwardInputGuards(t *testing.T) {
	const M, K, N = 2, 3, 4
	dyLinear := make([]float32, M*N)
	xLinear := make([]float32, M*K)
	wLinear := make([]float32, N*K)
	_, _, err := LinearBackwardF32(dyLinear[:len(dyLinear)-1], xLinear, wLinear, M, K, N)
	requireTrainingGuardError(t, "LinearBackwardF32 bad dy", err)
	_, _, err = LinearBackwardF32(dyLinear, xLinear[:len(xLinear)-1], wLinear, M, K, N)
	requireTrainingGuardError(t, "LinearBackwardF32 bad x", err)
	_, _, err = LinearBackwardF32(dyLinear, xLinear, wLinear[:len(wLinear)-1], M, K, N)
	requireTrainingGuardError(t, "LinearBackwardF32 bad w", err)

	const rows, width = 2, 5
	dyNorm := make([]float32, rows*width)
	xNorm := make([]float32, rows*width)
	normW := make([]float32, width)
	_, _, err = RMSNormBackwardF32(dyNorm[:len(dyNorm)-1], xNorm, normW, rows, width, 1e-5)
	requireTrainingGuardError(t, "RMSNormBackwardF32 bad dy", err)
	_, _, err = RMSNormBackwardF32(dyNorm, xNorm[:len(xNorm)-1], normW, rows, width, 1e-5)
	requireTrainingGuardError(t, "RMSNormBackwardF32 bad x", err)
	_, _, err = RMSNormBackwardF32(dyNorm, xNorm, normW[:len(normW)-1], rows, width, 1e-5)
	requireTrainingGuardError(t, "RMSNormBackwardF32 bad g", err)

	gated := make([]float32, width)
	gate := make([]float32, width)
	up := make([]float32, width)
	_, _, err = GeluGateMulBackwardF32(gated[:len(gated)-1], gate, up, width)
	requireTrainingGuardError(t, "GeluGateMulBackwardF32 bad dgated", err)
	_, _, err = GeluGateMulBackwardF32(gated, gate[:len(gate)-1], up, width)
	requireTrainingGuardError(t, "GeluGateMulBackwardF32 bad gate", err)
	_, _, err = GeluGateMulBackwardF32(gated, gate, up[:len(up)-1], width)
	requireTrainingGuardError(t, "GeluGateMulBackwardF32 bad up", err)

	const dModel, dFF = 4, 6
	h := make([]float32, M*dModel)
	norm := make([]float32, dModel)
	wGate := make([]float32, dFF*dModel)
	wUp := make([]float32, dFF*dModel)
	wDown := make([]float32, dModel*dFF)
	_, err = MLPBlockForwardF32(h[:len(h)-1], norm, wGate, wUp, wDown, M, dModel, dFF, 1e-5)
	requireTrainingGuardError(t, "MLPBlockForwardF32 bad h", err)
	_, err = MLPBlockForwardF32(h, norm[:len(norm)-1], wGate, wUp, wDown, M, dModel, dFF, 1e-5)
	requireTrainingGuardError(t, "MLPBlockForwardF32 bad normW", err)
	_, err = MLPBlockForwardF32(h, norm, wGate[:len(wGate)-1], wUp, wDown, M, dModel, dFF, 1e-5)
	requireTrainingGuardError(t, "MLPBlockForwardF32 bad wGate", err)

	doutMLP := make([]float32, M*dModel)
	_, err = MLPBlockBackwardF32(doutMLP[:len(doutMLP)-1], h, norm, wGate, wUp, wDown, M, dModel, dFF, 1e-5)
	requireTrainingGuardError(t, "MLPBlockBackwardF32 bad dout", err)
	_, err = MLPBlockBackwardF32(doutMLP, h[:len(h)-1], norm, wGate, wUp, wDown, M, dModel, dFF, 1e-5)
	requireTrainingGuardError(t, "MLPBlockBackwardF32 bad h", err)
	_, err = MLPBlockBackwardF32(doutMLP, h, norm[:len(norm)-1], wGate, wUp, wDown, M, dModel, dFF, 1e-5)
	requireTrainingGuardError(t, "MLPBlockBackwardF32 bad normW", err)
	_, err = MLPBlockBackwardF32(doutMLP, h, norm, wGate[:len(wGate)-1], wUp, wDown, M, dModel, dFF, 1e-5)
	requireTrainingGuardError(t, "MLPBlockBackwardF32 bad wGate", err)
	_, err = MLPBlockBackwardF32(doutMLP, h, norm, wGate, wUp[:len(wUp)-1], wDown, M, dModel, dFF, 1e-5)
	requireTrainingGuardError(t, "MLPBlockBackwardF32 bad wUp", err)
	_, err = MLPBlockBackwardF32(doutMLP, h, norm, wGate, wUp, wDown[:len(wDown)-1], M, dModel, dFF, 1e-5)
	requireTrainingGuardError(t, "MLPBlockBackwardF32 bad wDown", err)

	probs := make([]float32, rows*width)
	_, err = SoftmaxBackwardF32(dyNorm[:len(dyNorm)-1], probs, rows, width)
	requireTrainingGuardError(t, "SoftmaxBackwardF32 bad dy", err)
	_, err = SoftmaxBackwardF32(dyNorm, probs[:len(probs)-1], rows, width)
	requireTrainingGuardError(t, "SoftmaxBackwardF32 bad y", err)

	ropeDy := make([]float32, 2*8)
	_, err = RoPEBackwardF32(ropeDy[:len(ropeDy)-1], 0, 2, 8, 4, 10000)
	requireTrainingGuardError(t, "RoPEBackwardF32 bad dy", err)
	_, err = RoPEBackwardF32(ropeDy, 0, 2, 8, 9, 10000)
	requireTrainingGuardError(t, "RoPEBackwardF32 rotary too wide", err)
	_, err = RoPEBackwardF32(ropeDy, 0, 2, 8, 5, 10000)
	requireTrainingGuardError(t, "RoPEBackwardF32 odd rotary", err)

	const L, headDim = 3, 4
	single := make([]float32, L*headDim)
	_, _, _, err = AttnSingleHeadBackwardF32(single[:len(single)-1], single, single, single, L, headDim, 0.5, true)
	requireTrainingGuardError(t, "AttnSingleHeadBackwardF32 bad dOut", err)
	_, _, _, err = AttnSingleHeadBackwardF32(single, single[:len(single)-1], single, single, L, headDim, 0.5, true)
	requireTrainingGuardError(t, "AttnSingleHeadBackwardF32 bad q", err)
	_, _, _, err = AttnSingleHeadBackwardF32(single, single, single[:len(single)-1], single, L, headDim, 0.5, true)
	requireTrainingGuardError(t, "AttnSingleHeadBackwardF32 bad k", err)
	_, _, _, err = AttnSingleHeadBackwardF32(single, single, single, single[:len(single)-1], L, headDim, 0.5, true)
	requireTrainingGuardError(t, "AttnSingleHeadBackwardF32 bad v", err)

	const H, Hkv = 4, 2
	q := make([]float32, L*H*headDim)
	kv := make([]float32, L*Hkv*headDim)
	_, _, err = QKNormBackwardF32(q[:len(q)-1], q, make([]float32, headDim), L, H, headDim, 1e-5)
	requireTrainingGuardError(t, "QKNormBackwardF32 bad dy", err)
	_, _, err = QKNormBackwardF32(q, q[:len(q)-1], make([]float32, headDim), L, H, headDim, 1e-5)
	requireTrainingGuardError(t, "QKNormBackwardF32 bad x", err)
	_, _, err = QKNormBackwardF32(q, q, make([]float32, headDim-1), L, H, headDim, 1e-5)
	requireTrainingGuardError(t, "QKNormBackwardF32 bad normW", err)

	_, _, _, err = MultiHeadAttnBackwardF32(q, q, kv, kv, L, 3, 2, headDim, 0.5, true)
	requireTrainingGuardError(t, "MultiHeadAttnBackwardF32 bad GQA", err)
	_, _, _, err = MultiHeadAttnBackwardF32(q[:len(q)-1], q, kv, kv, L, H, Hkv, headDim, 0.5, true)
	requireTrainingGuardError(t, "MultiHeadAttnBackwardF32 bad dOut", err)
	_, _, _, err = MultiHeadAttnBackwardF32(q, q[:len(q)-1], kv, kv, L, H, Hkv, headDim, 0.5, true)
	requireTrainingGuardError(t, "MultiHeadAttnBackwardF32 bad q", err)
	_, _, _, err = MultiHeadAttnBackwardF32(q, q, kv[:len(kv)-1], kv, L, H, Hkv, headDim, 0.5, true)
	requireTrainingGuardError(t, "MultiHeadAttnBackwardF32 bad k", err)
	_, _, _, err = MultiHeadAttnBackwardF32(q, q, kv, kv[:len(kv)-1], L, H, Hkv, headDim, 0.5, true)
	requireTrainingGuardError(t, "MultiHeadAttnBackwardF32 bad v", err)
}

func TestTrainingBackwardKernelFailureGuards(t *testing.T) {
	requireNativeRuntime(t)

	withWrongMainLibrary(t, func() {
		const M, K, N = 2, 3, 4
		if _, _, err := LinearBackwardF32(
			syntheticFloat32(M*N, 101),
			syntheticFloat32(M*K, 103),
			syntheticFloat32(N*K, 107),
			M, K, N,
		); err == nil {
			t.Fatal("LinearBackwardF32(wrong library) error = nil")
		}
		resetNativePipelineCachesForCoverage()

		const dModel, dFF = 4, 6
		h := syntheticFloat32(M*dModel, 109)
		norm := syntheticFloat32(dModel, 111)
		wGate := syntheticFloat32(dFF*dModel, 113)
		wUp := syntheticFloat32(dFF*dModel, 115)
		wDown := syntheticFloat32(dModel*dFF, 117)
		if _, err := MLPBlockForwardF32(h, norm, wGate, wUp, wDown, M, dModel, dFF, 1e-5); err == nil {
			t.Fatal("MLPBlockForwardF32(wrong library) error = nil")
		}
		resetNativePipelineCachesForCoverage()
		if _, err := MLPBlockBackwardF32(syntheticFloat32(M*dModel, 119), h, norm, wGate, wUp, wDown, M, dModel, dFF, 1e-5); err == nil {
			t.Fatal("MLPBlockBackwardF32(wrong library) error = nil")
		}
		resetNativePipelineCachesForCoverage()

		const L, headDim = 2, 4
		single := syntheticFloat32(L*headDim, 121)
		if _, _, _, err := AttnSingleHeadBackwardF32(single, single, single, single, L, headDim, 0.5, true); err == nil {
			t.Fatal("AttnSingleHeadBackwardF32(wrong library) error = nil")
		}
		resetNativePipelineCachesForCoverage()

		const H, Hkv = 2, 1
		q := syntheticFloat32(L*H*headDim, 123)
		kv := syntheticFloat32(L*Hkv*headDim, 125)
		if _, _, _, err := MultiHeadAttnBackwardF32(q, q, kv, kv, L, H, Hkv, headDim, 0.5, true); err == nil {
			t.Fatal("MultiHeadAttnBackwardF32(wrong library) error = nil")
		}
		resetNativePipelineCachesForCoverage()

		qDim, kvDim := H*headDim, Hkv*headDim
		hBlock := syntheticFloat32(L*dModel, 127)
		wQ := syntheticFloat32(qDim*dModel, 129)
		wK := syntheticFloat32(kvDim*dModel, 131)
		wV := syntheticFloat32(kvDim*dModel, 133)
		wO := syntheticFloat32(dModel*qDim, 135)
		if _, err := MultiHeadAttnBlockForwardF32(hBlock, norm, wQ, wK, wV, wO, L, dModel, H, Hkv, headDim, headDim, 10000, 0.5, 1e-5, true); err == nil {
			t.Fatal("MultiHeadAttnBlockForwardF32(wrong library) error = nil")
		}
		resetNativePipelineCachesForCoverage()
		if _, err := MultiHeadAttnBlockBackwardF32(syntheticFloat32(L*dModel, 137), hBlock, norm, wQ, wK, wV, wO, L, dModel, H, Hkv, headDim, headDim, 10000, 0.5, 1e-5, true); err == nil {
			t.Fatal("MultiHeadAttnBlockBackwardF32(wrong library) error = nil")
		}
		resetNativePipelineCachesForCoverage()

		wSingle := syntheticFloat32(headDim*dModel, 139)
		wOSingle := syntheticFloat32(dModel*headDim, 141)
		if _, err := AttnBlockBackwardF32(syntheticFloat32(L*dModel, 143), hBlock, norm, wSingle, wSingle, wSingle, wOSingle, L, dModel, headDim, headDim, 10000, 0.5, 1e-5, true); err == nil {
			t.Fatal("AttnBlockBackwardF32(wrong library) error = nil")
		}
	})
}

func TestTrainingBlockInputGuards(t *testing.T) {
	const L, dModel, H, Hkv, headDim, rotaryDim = 2, 8, 4, 2, 2, 2
	qDim, kvDim := H*headDim, Hkv*headDim
	h := make([]float32, L*dModel)
	norm := make([]float32, dModel)
	wQ := make([]float32, qDim*dModel)
	wK := make([]float32, kvDim*dModel)
	wV := make([]float32, kvDim*dModel)
	wO := make([]float32, dModel*qDim)
	dout := make([]float32, L*dModel)

	_, err := MultiHeadAttnBlockForwardF32(h, norm, wQ[:len(wQ)-1], wK, wV, wO, L, dModel, H, Hkv, headDim, rotaryDim, 10000, 0.5, 1e-5, true)
	requireTrainingGuardError(t, "MultiHeadAttnBlockForwardF32 bad wQ", err)
	_, err = MultiHeadAttnBlockBackwardF32(dout[:len(dout)-1], h, norm, wQ, wK, wV, wO, L, dModel, H, Hkv, headDim, rotaryDim, 10000, 0.5, 1e-5, true)
	requireTrainingGuardError(t, "MultiHeadAttnBlockBackwardF32 bad dout", err)
	_, err = MultiHeadAttnBlockBackwardF32(dout, h[:len(h)-1], norm, wQ, wK, wV, wO, L, dModel, H, Hkv, headDim, rotaryDim, 10000, 0.5, 1e-5, true)
	requireTrainingGuardError(t, "MultiHeadAttnBlockBackwardF32 bad h", err)
	_, err = MultiHeadAttnBlockBackwardF32(dout, h, norm[:len(norm)-1], wQ, wK, wV, wO, L, dModel, H, Hkv, headDim, rotaryDim, 10000, 0.5, 1e-5, true)
	requireTrainingGuardError(t, "MultiHeadAttnBlockBackwardF32 bad normW", err)
	_, err = MultiHeadAttnBlockBackwardF32(dout, h, norm, wQ[:len(wQ)-1], wK, wV, wO, L, dModel, H, Hkv, headDim, rotaryDim, 10000, 0.5, 1e-5, true)
	requireTrainingGuardError(t, "MultiHeadAttnBlockBackwardF32 bad wQ", err)
	_, err = MultiHeadAttnBlockBackwardF32(dout, h, norm, wQ, wK[:len(wK)-1], wV, wO, L, dModel, H, Hkv, headDim, rotaryDim, 10000, 0.5, 1e-5, true)
	requireTrainingGuardError(t, "MultiHeadAttnBlockBackwardF32 bad wK", err)
	_, err = MultiHeadAttnBlockBackwardF32(dout, h, norm, wQ, wK, wV[:len(wV)-1], wO, L, dModel, H, Hkv, headDim, rotaryDim, 10000, 0.5, 1e-5, true)
	requireTrainingGuardError(t, "MultiHeadAttnBlockBackwardF32 bad wV", err)
	_, err = MultiHeadAttnBlockBackwardF32(dout, h, norm, wQ, wK, wV, wO[:len(wO)-1], L, dModel, H, Hkv, headDim, rotaryDim, 10000, 0.5, 1e-5, true)
	requireTrainingGuardError(t, "MultiHeadAttnBlockBackwardF32 bad wO", err)

	wSingle := make([]float32, headDim*dModel)
	wOSingle := make([]float32, dModel*headDim)
	_, err = AttnBlockBackwardF32(dout[:len(dout)-1], h, norm, wSingle, wSingle, wSingle, wOSingle, L, dModel, headDim, rotaryDim, 10000, 0.5, 1e-5, true)
	requireTrainingGuardError(t, "AttnBlockBackwardF32 bad dout", err)
	_, err = AttnBlockBackwardF32(dout, h[:len(h)-1], norm, wSingle, wSingle, wSingle, wOSingle, L, dModel, headDim, rotaryDim, 10000, 0.5, 1e-5, true)
	requireTrainingGuardError(t, "AttnBlockBackwardF32 bad h", err)
	_, err = AttnBlockBackwardF32(dout, h, norm[:len(norm)-1], wSingle, wSingle, wSingle, wOSingle, L, dModel, headDim, rotaryDim, 10000, 0.5, 1e-5, true)
	requireTrainingGuardError(t, "AttnBlockBackwardF32 bad normW", err)
	_, err = AttnBlockBackwardF32(dout, h, norm, wSingle[:len(wSingle)-1], wSingle, wSingle, wOSingle, L, dModel, headDim, rotaryDim, 10000, 0.5, 1e-5, true)
	requireTrainingGuardError(t, "AttnBlockBackwardF32 bad wQ", err)
	_, err = AttnBlockBackwardF32(dout, h, norm, wSingle, wSingle[:len(wSingle)-1], wSingle, wOSingle, L, dModel, headDim, rotaryDim, 10000, 0.5, 1e-5, true)
	requireTrainingGuardError(t, "AttnBlockBackwardF32 bad wK", err)
	_, err = AttnBlockBackwardF32(dout, h, norm, wSingle, wSingle, wSingle[:len(wSingle)-1], wOSingle, L, dModel, headDim, rotaryDim, 10000, 0.5, 1e-5, true)
	requireTrainingGuardError(t, "AttnBlockBackwardF32 bad wV", err)
	_, err = AttnBlockBackwardF32(dout, h, norm, wSingle, wSingle, wSingle, wOSingle[:len(wOSingle)-1], L, dModel, headDim, rotaryDim, 10000, 0.5, 1e-5, true)
	requireTrainingGuardError(t, "AttnBlockBackwardF32 bad wO", err)
}

func TestTrainingOptimiserInputGuards(t *testing.T) {
	const rows, vocab = 2, 4
	logits := make([]float32, rows*vocab)
	targets := []int32{0, 3}
	_, _, err := CrossEntropyBackwardF32(logits[:len(logits)-1], targets, rows, vocab)
	requireTrainingGuardError(t, "CrossEntropyBackwardF32 bad logits", err)
	_, _, err = CrossEntropyBackwardF32(logits, targets[:len(targets)-1], rows, vocab)
	requireTrainingGuardError(t, "CrossEntropyBackwardF32 bad targets", err)
	_, _, err = CrossEntropyBackwardF32(logits, []int32{0, int32(vocab)}, rows, vocab)
	requireTrainingGuardError(t, "CrossEntropyBackwardF32 target out of range", err)

	opt := NewAdamW(3, 0.1, 0)
	err = opt.Step(make([]float32, 2), make([]float32, 2))
	requireTrainingGuardError(t, "AdamW.Step bad state", err)
	err = opt.Step(make([]float32, 3), make([]float32, 2))
	requireTrainingGuardError(t, "AdamW.Step bad grads", err)
}
