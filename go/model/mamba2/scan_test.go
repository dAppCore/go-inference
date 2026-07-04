// SPDX-Licence-Identifier: EUPL-1.2

package mamba2

import (
	"math"
	"testing"
)

// syn is a deterministic synthetic vector (seeded), values in [-1, 1).
func syn(n, seed int) []float32 {
	out := make([]float32, n)
	for i := range out {
		out[i] = float32((i*seed+7)%101-50) * 0.02
	}
	return out
}

// TestSSDScanL1ClosedForm checks the scan against the closed form for a single step from a zero state:
// y[h,p] = Δ_h · x[h,p] · (B_h·C_h) + D_h · x[h,p], where B_h·C_h = Σ_n B[h,n]·C[h,n].
func TestSSDScanL1ClosedForm(t *testing.T) {
	const H, P, N = 3, 4, 5
	x := syn(H*P, 1)
	dt := syn(H, 2)
	a := syn(H, 3)
	b := syn(H*N, 4)
	c := syn(H*N, 5)
	d := syn(H, 6)
	y, _, err := SSDScanF32(x, dt, a, b, c, d, nil, 1, H, P, N)
	if err != nil {
		t.Fatalf("SSDScanF32: %v", err)
	}
	for h := 0; h < H; h++ {
		var bc float64
		for n := 0; n < N; n++ {
			bc += float64(b[h*N+n]) * float64(c[h*N+n])
		}
		for p := 0; p < P; p++ {
			xtp := float64(x[h*P+p])
			want := float64(dt[h])*xtp*bc + float64(d[h])*xtp
			if got := float64(y[h*P+p]); math.Abs(got-want) > 1e-4*(1+math.Abs(want)) {
				t.Errorf("y[%d,%d] = %v, closed form = %v", h, p, got, want)
			}
		}
	}
	t.Log("SSD scan L=1 matches the closed form Δ·(B·C)·x + D·x")
}

// TestSSDScanChunkCarry proves the decode-boundary invariant: scanning a sequence in one pass equals
// scanning it as two chunks where the first chunk's final state is carried into the second. This MUST
// be bit-exact (the per-step recurrence is identical regardless of where the chunk boundary falls) — it
// is what makes Mamba-2 decode (carry the SSM state across calls) correct.
func TestSSDScanChunkCarry(t *testing.T) {
	const L, split, H, P, N = 7, 4, 2, 3, 5
	x := syn(L*H*P, 1)
	dt := syn(L*H, 2)
	a := syn(H, 3) // (sign irrelevant to the carry property)
	b := syn(L*H*N, 4)
	c := syn(L*H*N, 5)
	d := syn(H, 6)

	yFull, sFull, err := SSDScanF32(x, dt, a, b, c, d, nil, L, H, P, N)
	if err != nil {
		t.Fatalf("full: %v", err)
	}
	y1, s1, err := SSDScanF32(x[:split*H*P], dt[:split*H], a, b[:split*H*N], c[:split*H*N], d, nil, split, H, P, N)
	if err != nil {
		t.Fatalf("chunk1: %v", err)
	}
	rem := L - split
	y2, s2, err := SSDScanF32(x[split*H*P:], dt[split*H:], a, b[split*H*N:], c[split*H*N:], d, s1, rem, H, P, N)
	if err != nil {
		t.Fatalf("chunk2: %v", err)
	}

	for i := range y1 {
		if y1[i] != yFull[i] {
			t.Fatalf("chunk1 y[%d] = %v != full %v", i, y1[i], yFull[i])
		}
	}
	for i := range y2 {
		if y2[i] != yFull[split*H*P+i] {
			t.Fatalf("chunk2 y[%d] = %v != full %v", i, y2[i], yFull[split*H*P+i])
		}
	}
	for i := range s2 {
		if s2[i] != sFull[i] {
			t.Fatalf("carried state[%d] = %v != full %v", i, s2[i], sFull[i])
		}
	}
	t.Logf("SSD scan chunk-carry bit-exact: split %d|%d, y and state identical to the one-pass scan (decode boundary correct)", split, rem)
}
