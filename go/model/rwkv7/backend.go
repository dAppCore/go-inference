// SPDX-Licence-Identifier: EUPL-1.2

package rwkv7

// backend.go is the device-acceleration seam for the RWKV-7 block's projections (its compute hot spot —
// dense GEMM; the WKV7 recurrence is cheap by comparison). ProjMatMul lets a backend run the projections
// on its accelerator while the block + recurrence stay engine-neutral pure Go. The lib never imports the
// backend (AX-8): rwkv7 DECLARES the hook, pkg/native SETS it from init to its steel GEMM (byte-identical
// to metal's projection matmul).

// ProjMatMul, when set by a backend, runs a block projection y = x[M,K] @ w[N,K]ᵀ on-device. nil ⇒ the
// host-f32 matNT default (pure Go — go-rocm, tests, the higher-precision f64 reference).
var ProjMatMul func(x, w []float32, M, K, N int) ([]float32, error)

// projMatMul runs y = x[M,K] @ w[N,K]ᵀ through the backend hook when set, else the host matNT.
func projMatMul(x, w []float32, M, K, N int) ([]float32, error) {
	if ProjMatMul != nil {
		return ProjMatMul(x, w, M, K, N)
	}
	return matNT(x, w, M, K, N), nil
}
