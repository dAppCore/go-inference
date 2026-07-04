// SPDX-Licence-Identifier: EUPL-1.2

package mamba2

// backend.go is the device-acceleration seam for the Mamba-2 block's projections. The in/out projections
// are the block's compute hot spot (dense GEMM, ~all of BlockForwardF32's time per the benches); the SSM
// scan + conv are cheap by comparison. ProjMatMul lets a backend run those projections on its accelerator
// while the block's structure + the scan/conv stay engine-neutral pure Go. The lib never imports the
// backend (AX-8): mamba2 DECLARES the hook, the backend (pkg/native) SETS it from init.

// ProjMatMul, when set by a backend, runs a block projection y = x[M,K] @ w[N,K]ᵀ on-device. nil ⇒ the
// host-f32 matNT default (pure Go — go-rocm, tests, and any caller that hasn't wired a backend). native
// sets it to its steel GEMM, which is byte-identical to metal's projection matmul — so a native serve runs
// the projections on the GPU and matches metal, while the pure-Go path stays the higher-precision (f64
// accumulation) reference.
var ProjMatMul func(x, w []float32, M, K, N int) ([]float32, error)

// projMatMul runs y = x[M,K] @ w[N,K]ᵀ through the backend hook when set, else the host matNT.
func projMatMul(x, w []float32, M, K, N int) ([]float32, error) {
	if ProjMatMul != nil {
		return ProjMatMul(x, w, M, K, N)
	}
	return matNT(x, w, M, K, N), nil
}
