// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	core "dappco.re/go"
	"dappco.re/go/inference/model"
)

// composed_bf16_backend.go — the dense bf16 matvec seam (#26): the exact quant-seam shape
// (composed_quant_backend.go) for a checkpoint's UNQUANTISED bf16 projections. The weight bytes
// stay the mmap view (residentBytes caches the device buffer per slice); activations cross the
// seam f32 exactly like the quant seam, cast to bf16 for the kernel. This is what stops the dense
// composed lane widening every projection to f32 — the ×2 on bytes streamed per token AND on the
// resident set that put the official bf16 Qwen exports at ×4-6 behind mlx-lm.
func init() {
	// Bound in composed_backend.go alongside the other hooks once the lib seam lands (slice 1);
	// the op below is complete and parity-gated now so consumers migrate onto a proven primitive.
}

// MatMulBF16WF32NTInto computes out[M,N] = x[M,K] @ wᵀ for a dense bf16 weight (w = raw row-major
// bf16 bytes [N,K]) with f32 activations at the seam. M=1 (decode) rides the MLX bf16 gemv over the
// resident weight bytes; M>1 currently loops rows through the same gemv (correct; the batched
// prefill slab is the follow-up). out is reused when cap(out) >= M*N.
func MatMulBF16WF32NTInto(out, x []float32, w []byte, M, K, N int) ([]float32, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	if M <= 0 || N <= 0 || K <= 0 {
		return nil, core.NewError("native.MatMulBF16WF32NTInto: M, N, K must be positive")
	}
	if len(x) != M*K {
		return nil, core.NewError("native.MatMulBF16WF32NTInto: len(x) must equal M*K")
	}
	if len(w) != N*K*bf16Size {
		return nil, core.NewError("native.MatMulBF16WF32NTInto: len(w) must equal N*K*2 bytes")
	}
	if cap(out) < M*N {
		out = make([]float32, M*N)
	} else {
		out = out[:M*N]
	}
	xb := f32sToBF16Bytes(x)
	var ob []byte
	for m := 0; m < M; m++ {
		row, err := MatVecBF16Into(ob, w, xb[m*K*bf16Size:(m+1)*K*bf16Size], N, K)
		if err != nil {
			return nil, err
		}
		ob = row
		for n := 0; n < N; n++ {
			out[m*N+n] = bf16ToF32(row[n*bf16Size], row[n*bf16Size+1])
		}
	}
	return out, nil
}

// MatMulBF16WeightF32NTInto is MatMulBF16WF32NTInto over the lib's BF16Weight form — the signature
// the composed/qwen3 hooks bind.
func MatMulBF16WeightF32NTInto(out, x []float32, w *model.BF16Weight, M, K, N int) ([]float32, error) {
	if w == nil || w.OutDim != N || w.InDim != K {
		return nil, core.NewError("native.MatMulBF16WeightF32NTInto: weight geometry mismatch")
	}
	return MatMulBF16WF32NTInto(out, x, w.Data, M, K, N)
}
