// SPDX-Licence-Identifier: EUPL-1.2

package mamba2

import "testing"

func mkModel(cfg BlockConfig, D, vocab, nLayers int) *MambaModel {
	layers := make([]MambaLayer, nLayers)
	for li := range layers {
		layers[li] = MambaLayer{Norm: syn(D, li*9+1), W: mkBlockWeights(cfg, D)}
	}
	return &MambaModel{
		Embed:  syn(vocab*D, 100),
		NormF:  syn(D, 101),
		LMHead: nil, // tied to Embed
		Layers: layers,
		Cfg:    cfg,
		D:      D,
		Vocab:  vocab,
	}
}

// TestMambaDecodeEqualsPrefill is the recurrent-decode correctness: stepping a sequence one token at a
// time through a fresh session (each step O(1), threading the per-layer conv + SSM state) produces hidden
// states BIT-EXACT to a single prefill pass over the whole sequence. This is the SSM analogue of the KV
// cache being byte-faithful — what makes streaming Mamba-2 decode correct.
func TestMambaDecodeEqualsPrefill(t *testing.T) {
	cfg := BlockConfig{NumHeads: 2, HeadDim: 8, StateDim: 8, NumGroups: 1, ConvKernel: 4, Eps: 1e-5}
	const D, vocab, nLayers = 8, 32, 2
	m := mkModel(cfg, D, vocab, nLayers)
	tokens := []int32{1, 5, 9, 2, 7, 3}

	prefill, err := NewSession(m).Forward(tokens)
	if err != nil {
		t.Fatalf("prefill: %v", err)
	}

	dec := NewSession(m)
	for t0, tok := range tokens {
		h, err := dec.Forward([]int32{tok})
		if err != nil {
			t.Fatalf("decode step %d: %v", t0, err)
		}
		for i := range D {
			if h[i] != prefill[t0*D+i] {
				t.Fatalf("token %d hidden[%d] = %v != prefill %v (recurrent decode diverged)", t0, i, h[i], prefill[t0*D+i])
			}
		}
	}
	t.Logf("mamba2 recurrent decode == prefill bit-exact over %d tokens, %d layers", len(tokens), nLayers)
}

// TestMambaForwardRunsEveryLayer guards the layer loop in forwardEmb against an early-exit
// regression — the class of bug the decode==prefill tests structurally CANNOT catch, because
// both sides of that comparison call the same forwardEmb: a loop that stopped after layer 0
// would still be self-consistent between prefill and decode. Two independent probes: every
// layer's recurrent state must have advanced after a forward (an early break leaves the later
// slots nil), and a 2-layer model must produce different hiddens from a 1-layer model built
// from the identical weights (mkModel's block weights are seed-fixed, so layer 0 is shared —
// any difference is layer 1's contribution).
func TestMambaForwardRunsEveryLayer(t *testing.T) {
	cfg := BlockConfig{NumHeads: 2, HeadDim: 8, StateDim: 8, NumGroups: 1, ConvKernel: 4, Eps: 1e-5}
	const D, vocab = 8, 32
	tokens := []int32{1, 5, 9}

	two := mkModel(cfg, D, vocab, 2)
	s := NewSession(two)
	outTwo, err := s.Forward(tokens)
	if err != nil {
		t.Fatalf("2-layer forward: %v", err)
	}
	for li := range two.Layers {
		if s.convState[li] == nil || s.ssmState[li] == nil {
			t.Fatalf("layer %d recurrent state not advanced — forwardEmb's layer loop exited early", li)
		}
	}

	one := mkModel(cfg, D, vocab, 1) // identical layer-0 weights (fixed seeds), one layer fewer
	outOne, err := NewSession(one).Forward(tokens)
	if err != nil {
		t.Fatalf("1-layer forward: %v", err)
	}
	same := true
	for i := range outTwo {
		if outTwo[i] != outOne[i] {
			same = false
			break
		}
	}
	if same {
		t.Fatal("2-layer hiddens identical to 1-layer — layer 1 contributed nothing (layer-loop regression)")
	}
	t.Logf("forwardEmb runs every layer: all %d state slots advanced, layer 1 changes the hiddens", len(two.Layers))
}

// TestMambaGenerateEOSStops covers Generate's eos early-stop branch (every other call site
// passes eosID = -1, leaving the branch dead in tests): with eosID set to the first token the
// unconstrained run emits, generation must stop after exactly that one token — the eos token
// itself included in the output (appended before the break, per the documented behaviour).
func TestMambaGenerateEOSStops(t *testing.T) {
	cfg := BlockConfig{NumHeads: 2, HeadDim: 8, StateDim: 8, NumGroups: 1, ConvKernel: 4, Eps: 1e-5}
	m := mkModel(cfg, 8, 32, 2)
	prompt := []int32{1, 2, 3}
	free, err := NewSession(m).Generate(prompt, 6, -1) // deterministic reference run
	if err != nil {
		t.Fatalf("reference generate: %v", err)
	}
	stopped, err := NewSession(m).Generate(prompt, 6, int(free[0]))
	if err != nil {
		t.Fatalf("eos generate: %v", err)
	}
	if len(stopped) != 1 || stopped[0] != free[0] {
		t.Fatalf("Generate(eos=%d) = %v, want [%d] (stop immediately after emitting the eos token)", free[0], stopped, free[0])
	}
}

// TestMambaGenerate checks the greedy generate loop runs and is deterministic (same prompt → same
// tokens), exercising prefill + the per-token recurrent decode + the LM head.
func TestMambaGenerate(t *testing.T) {
	cfg := BlockConfig{NumHeads: 2, HeadDim: 8, StateDim: 8, NumGroups: 1, ConvKernel: 4, Eps: 1e-5}
	m := mkModel(cfg, 8, 32, 2)
	prompt := []int32{1, 2, 3}
	g1, err := NewSession(m).Generate(prompt, 6, -1)
	if err != nil {
		t.Fatalf("generate: %v", err)
	}
	if len(g1) != 6 {
		t.Fatalf("generated %d tokens, want 6", len(g1))
	}
	g2, _ := NewSession(m).Generate(prompt, 6, -1)
	for i := range g1 {
		if g1[i] != g2[i] {
			t.Fatalf("non-deterministic generate at %d: %d != %d", i, g1[i], g2[i])
		}
	}
	t.Logf("mamba2 Generate: prefill→recurrent decode→head produced %v (deterministic)", g1)
}
