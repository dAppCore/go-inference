// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import core "dappco.re/go"

// train_lora.go adds the LoRA adapter — the actual trainable parameters of a native SFT (the base
// weights stay frozen). A LoRA linear is y = x·Wᵀ + (alpha/rank)·(x·Aᵀ)·Bᵀ, with A [rank,in] and
// B [out,rank] the only tensors the optimiser steps (W frozen). Its forward and backward compose two
// of the linear primitives (train_backward.go); the backward returns the gradients of A and B (and the
// gradient to the layer input, so a full-stack SFT can keep backpropagating). With the per-op VJPs, the
// block backwards, cross-entropy and AdamW, this completes the trainable-parameter path; chaining it
// through every layer is the SFT loop. f32.

// LoRAForwardF32 computes the LoRA delta path output xA = x·Aᵀ [M,rank] and the scaled delta
// [M,out] = (alpha/rank)·(xA·Bᵀ). It returns both (xA is needed by the backward). The caller adds the
// delta to the frozen base output x·Wᵀ. x is [M,in], A [rank,in], B [out,rank].
func LoRAForwardF32(x, a, b []float32, M, in, out, rank int, scaling float32) (xA, delta []float32, err error) {
	if len(x) != M*in || len(a) != rank*in || len(b) != out*rank {
		return nil, nil, core.NewError("native.LoRAForwardF32: x[M,in]/A[rank,in]/B[out,rank] size mismatch")
	}
	xA, err = MatMulF32NT(x, a, M, in, rank) // x·Aᵀ → [M,rank]
	if err != nil {
		return nil, nil, err
	}
	delta, err = MatMulF32NT(xA, b, M, rank, out) // xA·Bᵀ → [M,out]
	if err != nil {
		return nil, nil, err
	}
	for i := range delta {
		delta[i] *= scaling
	}
	return xA, delta, nil
}

// LoRABackwardF32 is the VJP of the LoRA delta path. Given the upstream gradient dy [M,out] (the
// gradient of the layer output, which the delta is added into) and the recomputed xA from the forward,
// it returns the gradients of the trainable factors plus the gradient to x:
//
//	dInner = dy·scaling        (delta = scaling·(xA·Bᵀ))
//	dXA, dB = linearVJP(dInner, xA, B)     // inner = xA·Bᵀ
//	dX,  dA = linearVJP(dXA,   x,  A)      // xA    = x·Aᵀ
//
// dA [rank,in] and dB [out,rank] are what AdamW steps; dX [M,in] flows to the previous op. f32.
func LoRABackwardF32(dy, x, a, b, xA []float32, M, in, out, rank int, scaling float32) (dA, dB, dX []float32, err error) {
	if len(dy) != M*out || len(x) != M*in || len(a) != rank*in || len(b) != out*rank || len(xA) != M*rank {
		return nil, nil, nil, core.NewError("native.LoRABackwardF32: size mismatch")
	}
	dInner := make([]float32, M*out)
	for i := range dInner {
		dInner[i] = dy[i] * scaling
	}
	dXA, dB, err := LinearBackwardF32(dInner, xA, b, M, rank, out) // inner = xA·Bᵀ
	if err != nil {
		return nil, nil, nil, err
	}
	dX, dA, err = LinearBackwardF32(dXA, x, a, M, in, rank) // xA = x·Aᵀ
	if err != nil {
		return nil, nil, nil, err
	}
	return dA, dB, dX, nil
}
