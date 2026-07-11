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

// ProjMatMulInto is the OPTIONAL write-into sibling of ProjMatMul: a backend that can target a
// caller-owned output buffer sets this so the projection GEMM skips its per-call output alloc (the
// dominant per-token decode cost). nil ⇒ not injected — the caller falls back to ProjMatMul, then the
// host matNTInto. Into is preferred when set and the legacy ProjMatMul stays the fallback, so a backend
// that wired only the old hook keeps working. native sets it to MatMulF32NTInto (byte-identical to the
// fresh-buffer steel GEMM — Into changes only where the result lands).
var ProjMatMulInto func(out, x, w []float32, M, K, N int) ([]float32, error)

// projMatMul runs y = x[M,K] @ w[N,K]ᵀ through the backend hook when set, else the host matNT.
func projMatMul(x, w []float32, M, K, N int) ([]float32, error) {
	if ProjMatMul != nil {
		return ProjMatMul(x, w, M, K, N)
	}
	return matNT(x, w, M, K, N), nil
}

// projMatMulInto runs y = x[M,K] @ w[N,K]ᵀ into out (reused when cap(out) ≥ M·N, else a fresh slab).
// It prefers the write-into hook, then the legacy fresh-buffer hook (out ignored — correctness kept, no
// reuse), then the host matNTInto. The RETURNED slice is authoritative; callers store it back into their
// scratch to retain any growth.
func projMatMulInto(out, x, w []float32, M, K, N int) ([]float32, error) {
	if ProjMatMulInto != nil {
		return ProjMatMulInto(out, x, w, M, K, N)
	}
	if ProjMatMul != nil {
		return ProjMatMul(x, w, M, K, N)
	}
	return matNTInto(out, x, w, M, K, N), nil
}
