// SPDX-Licence-Identifier: EUPL-1.2

package rwkv7

import "testing"

// mkChannelMixWeights builds a synthetic channel-mix layer's weights.
func mkChannelMixWeights(D, FF, seed int) *channelMixWeights {
	return &channelMixWeights{
		XK:        syn(D, seed+1),
		KeyProj:   syn(FF*D, seed+2),
		ValueProj: syn(D*FF, seed+3),
	}
}

func TestChannelmix_channelMixForward_Good(t *testing.T) {
	const L, D, FF = 4, 6, 10
	w := mkChannelMixWeights(D, FF, 1)
	x := syn(L*D, 11)

	out, newShift, err := channelMixForward(x, w, nil, L, D, FF)
	if err != nil {
		t.Fatalf("channelMixForward: %v", err)
	}
	if len(out) != L*D {
		t.Fatalf("out len %d, want %d", len(out), L*D)
	}
	if len(newShift) != D {
		t.Fatalf("newShift len %d, want %d", len(newShift), D)
	}
	for i := range D {
		if newShift[i] != x[(L-1)*D+i] {
			t.Fatalf("newShift[%d] = %v, want x's last row %v", i, newShift[i], x[(L-1)*D+i])
		}
	}
}

func TestChannelmix_channelMixForward_Bad(t *testing.T) {
	if _, _, err := channelMixForward(syn(4*6, 1), nil, nil, 4, 6, 10); err == nil {
		t.Fatal("nil weights accepted")
	}
}

// TestChannelmix_channelMixForward_Ugly proves sqrelu is genuinely relu(x)^2, not sqr(relu-less) or
// plain relu: a negative-heavy hidden must zero out (relu) AND the surviving values must be squared
// (verified by comparing to a hand-computed single-channel case where KeyProj/ValueProj are unit
// projections through a 1-wide intermediate).
func TestChannelmix_channelMixForward_Ugly(t *testing.T) {
	const L, D, FF = 1, 1, 1
	w := &channelMixWeights{
		XK:        []float32{0}, // no token-shift contribution ⇒ xk == x exactly
		KeyProj:   []float32{1}, // hidden = x
		ValueProj: []float32{1}, // out = sqrelu(hidden)
	}
	neg, _, err := channelMixForward([]float32{-3}, w, nil, L, D, FF)
	if err != nil {
		t.Fatalf("negative case: %v", err)
	}
	if neg[0] != 0 {
		t.Fatalf("sqrelu(-3) = %v, want 0 (relu must zero negatives)", neg[0])
	}
	pos, _, err := channelMixForward([]float32{3}, w, nil, L, D, FF)
	if err != nil {
		t.Fatalf("positive case: %v", err)
	}
	if pos[0] != 9 {
		t.Fatalf("sqrelu(3) = %v, want 9 (relu(3)^2)", pos[0])
	}
}
