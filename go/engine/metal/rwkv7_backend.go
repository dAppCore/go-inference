// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import "dappco.re/go/inference/model/arch/rwkv7"

// rwkv7_backend.go wires native's device GEMM into the engine-neutral RWKV-7 block's projections (its
// compute hot spot; the WKV7 recurrence is cheap), the same seam as mamba2_backend.go. rwkv7 declares the
// ProjMatMul hook and runs the pure-Go host matNT by default (AX-8 — the lib never imports the backend);
// importing native binds it to the steel GEMM. MatMulF32NT(x, w, M, K, N) = x[M,K] @ w[N,K]ᵀ = the
// projection y = x @ Wᵀ, byte-identical to metal's projection matmul. (RWKV-7 is a per-layer mixer for a
// gemma4-shaped backbone, not yet wired into a servable native model — this readies the block's projections
// for whenever the mixer-decode path lands; it is not itself a serve path.)
func init() {
	rwkv7.ProjMatMul = MatMulF32NT
	rwkv7.ProjMatMulInto = MatMulF32NTInto // write-into sibling: skips the projection-output alloc, byte-identical
}
