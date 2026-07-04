// SPDX-Licence-Identifier: EUPL-1.2

// Package mamba2 is the native (no-cgo) Mamba-2 mixer: the selective state-space (SSD) scan and the
// pieces a Mamba-2 block composes from it. It is the linear-attention counterpart to the transformer
// arches — the first SSM family ported off metal. The scan here mirrors the metal SSDScan recurrence
// exactly (github.com/state-spaces/mamba ssd_minimal, scalar-A form) but as pure Go over f32 host
// slices, so it is engine-neutral (native + go-rocm) and verifiable with plain `go test`. A device /
// chunked-parallel form is a later optimisation over this exact O(L) recurrence.
package mamba2

import (
	"math"

	core "dappco.re/go"
)

// SSDScanF32 runs the Mamba-2 selective scan for one sequence (batch 1) and returns the mixed output
// y [L,H,P] and the advanced SSM state [H,P,N]. The per-timestep recurrence, with x_t [H,P], B_t/C_t
// [H,N], a per-head decay scalar A [H] and step Δ [L,H] (softplus-activated, ≥0):
//
//	dA_t       = exp(Δ_t · A)                       // [H] scalar decay per head
//	state_t    = state_{t-1} · dA_t  +  x_t ⊗ (Δ_t·B_t)   // [H,P,N] outer product
//	y_t        = state_t @ C_t  +  D ⊙ x_t          // [H,P]
//
// prior is the carried state [H,P,N] from the previous chunk (decode) or nil for a fresh sequence
// (prefill, zero state). d is the per-head skip scalar [H] or nil for no skip. Layouts are row-major:
// x[t*H*P + h*P + p], b/c[t*H*N + h*N + n], state[h*P*N + p*N + n]. f32 throughout (the SSM accumulates
// in f32, the precision the reference and metal's scan keep through the recurrence).
func SSDScanF32(x, dt, a, b, c, d, prior []float32, L, H, P, N int) (y, state []float32, err error) {
	if L <= 0 || H <= 0 || P <= 0 || N <= 0 {
		return nil, nil, core.NewError("mamba2.SSDScanF32: L,H,P,N must be > 0")
	}
	if len(x) != L*H*P || len(dt) != L*H || len(a) != H || len(b) != L*H*N || len(c) != L*H*N {
		return nil, nil, core.NewError("mamba2.SSDScanF32: x[L,H,P]/dt[L,H]/a[H]/b,c[L,H,N] size mismatch")
	}
	if d != nil && len(d) != H {
		return nil, nil, core.NewError("mamba2.SSDScanF32: d must be [H] or nil")
	}
	if prior != nil && len(prior) != H*P*N {
		return nil, nil, core.NewError("mamba2.SSDScanF32: prior state must be [H,P,N] or nil")
	}
	y = make([]float32, L*H*P)
	state = make([]float32, H*P*N)
	if prior != nil {
		copy(state, prior)
	}
	for t := 0; t < L; t++ {
		for h := 0; h < H; h++ {
			dth := float64(dt[t*H+h])
			dA := math.Exp(dth * float64(a[h])) // scalar decay for this head this step
			bRow := b[t*H*N+h*N : t*H*N+h*N+N]
			cRow := c[t*H*N+h*N : t*H*N+h*N+N]
			for p := 0; p < P; p++ {
				xtp := float64(x[t*H*P+h*P+p])
				base := h*P*N + p*N
				var yp float64
				for n := 0; n < N; n++ {
					st := float64(state[base+n])*dA + xtp*dth*float64(bRow[n]) // decay + (Δ·B)⊗x
					state[base+n] = float32(st)
					yp += st * float64(cRow[n]) // state @ C
				}
				if d != nil {
					yp += float64(d[h]) * xtp // D skip
				}
				y[t*H*P+h*P+p] = float32(yp)
			}
		}
	}
	return y, state, nil
}
