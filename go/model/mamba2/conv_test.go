// SPDX-Licence-Identifier: EUPL-1.2

package mamba2

import (
	"math"
	"testing"
)

// TestCausalConv1dKnown checks the causal conv against a hand window: with K=3 and a fresh (zero-padded)
// start, out[0]=w2·x0, out[1]=w1·x0+w2·x1, out[2]=w0·x0+w1·x1+w2·x2 (per channel), + bias.
func TestCausalConv1dKnown(t *testing.T) {
	const L, convDim, K = 4, 2, 3
	in := syn(L*convDim, 1)
	w := syn(convDim*K, 2)
	bias := syn(convDim, 3)
	out, _, err := CausalConv1dF32(in, w, bias, nil, L, convDim, K)
	if err != nil {
		t.Fatalf("conv: %v", err)
	}
	x := func(t, ch int) float64 {
		if t < 0 {
			return 0
		}
		return float64(in[t*convDim+ch])
	}
	for ch := 0; ch < convDim; ch++ {
		w0, w1, w2 := float64(w[ch*K+0]), float64(w[ch*K+1]), float64(w[ch*K+2])
		bb := float64(bias[ch])
		for tt := 0; tt < L; tt++ {
			want := bb + w0*x(tt-2, ch) + w1*x(tt-1, ch) + w2*x(tt, ch)
			if got := float64(out[tt*convDim+ch]); math.Abs(got-want) > 1e-4*(1+math.Abs(want)) {
				t.Errorf("out[%d,%d] = %v, want %v", tt, ch, got, want)
			}
		}
	}
	t.Log("causal conv1d matches the hand window (weight[K-1] = current input)")
}

// TestCausalConv1dCarry proves the conv-state ring invariant: conv'ing a sequence in one pass is
// BIT-EXACT to conv'ing it as two chunks carrying the last K-1 inputs across the boundary — the
// decode-streaming correctness for the conv.
func TestCausalConv1dCarry(t *testing.T) {
	const L, split, convDim, K = 9, 5, 3, 4
	in := syn(L*convDim, 1)
	w := syn(convDim*K, 2)
	bias := syn(convDim, 3)

	full, _, err := CausalConv1dF32(in, w, bias, nil, L, convDim, K)
	if err != nil {
		t.Fatalf("full: %v", err)
	}
	o1, s1, err := CausalConv1dF32(in[:split*convDim], w, bias, nil, split, convDim, K)
	if err != nil {
		t.Fatalf("chunk1: %v", err)
	}
	rem := L - split
	o2, _, err := CausalConv1dF32(in[split*convDim:], w, bias, s1, rem, convDim, K)
	if err != nil {
		t.Fatalf("chunk2: %v", err)
	}
	for i := range o1 {
		if o1[i] != full[i] {
			t.Fatalf("chunk1 out[%d] = %v != full %v", i, o1[i], full[i])
		}
	}
	for i := range o2 {
		if o2[i] != full[split*convDim+i] {
			t.Fatalf("chunk2 out[%d] = %v != full %v", i, o2[i], full[split*convDim+i])
		}
	}
	t.Logf("causal conv1d conv-state carry bit-exact: split %d|%d, output identical to the one-pass conv", split, rem)
}
