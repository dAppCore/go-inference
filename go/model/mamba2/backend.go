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
