// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import "dappco.re/go/inference/model/mamba2"

// mamba2_backend.go wires native's device GEMM into the engine-neutral Mamba-2 block. The block's in/out
// projections are its compute hot spot (dense GEMM — ~all of BlockForwardF32's time per the mamba2
// benches); the SSM scan + conv are cheap. mamba2 declares the ProjMatMul hook and runs the pure-Go host
// matNT by default (AX-8 — the lib never imports the backend); importing native binds the hook to the
// steel GEMM, so a native serve runs the projections on the GPU. MatMulF32NT(x, w, M, K, N) computes
// x[M,K] @ w[N,K]ᵀ — exactly the projection y = x @ Wᵀ for a [N,K] weight, and is byte-identical to
// metal's projection matmul, so the device path matches a metal serve.
func init() {
	mamba2.ProjMatMul = MatMulF32NT
}
