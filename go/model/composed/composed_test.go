// SPDX-Licence-Identifier: EUPL-1.2

package composed

import (
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference/model"
)

func syn(n, seed int) []float32 {
	out := make([]float32, n)
	for i := range out {
		out[i] = float32((i*seed+7)%101-50) * 0.02
	}
	return out
}

// countingMixer wraps a Mixer that also implements projMixer, counting calls to each method — used to
// prove forwardEmb's dispatch actually takes the isProj path (forwardNoProj) rather than the standard
// path (Forward) for a given composed config. Unlike engine/metal's composed_backend_test.go pattern
// (which counts a bound DEVICE hook), there is no device hook at this level to intercept: projMixer is
// composed's own arch-neutral dispatch contract, so the count is taken directly on the mixer.
type countingMixer struct {
	inner interface {
		Mixer
		projMixer
	}
	forwardCalls, noProjCalls int
}

func (c *countingMixer) Kind() string { return c.inner.Kind() }

// CloneState delegates transparently (like Kind) — the counting wrapper only instruments Forward/
// forwardNoProj, so a Layer.Mixer holding a *countingMixer still satisfies the (now-larger) Mixer
// interface without pretending to add its own state-cloning behaviour.
func (c *countingMixer) CloneState(prior any) any { return c.inner.CloneState(prior) }

func (c *countingMixer) Forward(h []float32, L, D int, prior any) ([]float32, any, error) {
	c.forwardCalls++
	return c.inner.Forward(h, L, D, prior)
}

func (c *countingMixer) forwardNoProj(h []float32, L, D int, prior any) (mixerHidden, projW []float32, mixCols int, next any, err error) {
	c.noProjCalls++
	return c.inner.forwardNoProj(h, L, D, prior)
}

func mkGatedDeltaMixer(cfg model.GatedDeltaConfig, D, seed int) Mixer {
	qd, vd, cd := cfg.KeyHeads*cfg.HeadDim, cfg.ValueHeads*cfg.HeadDim, 2*cfg.KeyHeads*cfg.HeadDim+cfg.ValueHeads*cfg.HeadDim
	_ = qd
	w := &model.GatedDeltaWeights{
		InProjQKV:  syn(cd*D, seed+1),
		ConvWeight: syn(cd*cfg.ConvKernel, seed+2),
		ConvBias:   syn(cd, seed+3),
		InProjA:    syn(cfg.ValueHeads*D, seed+4),
		ALog:       syn(cfg.ValueHeads, seed+5),
		DtBias:     syn(cfg.ValueHeads, seed+6),
		InProjB:    syn(cfg.ValueHeads*D, seed+7),
		InProjZ:    syn(vd*D, seed+8),
		Norm:       syn(cfg.HeadDim, seed+9),
		OutProj:    syn(D*vd, seed+10),
	}
	return NewGatedDeltaMixer(w, cfg)
}

func mkComposedModel(nLayers, D, vocab, FF int) *ComposedModel {
	cfg := model.GatedDeltaConfig{KeyHeads: 2, ValueHeads: 4, HeadDim: 8, ConvKernel: 4, Eps: 1e-5}
	layers := make([]Layer, nLayers)
	for li := range layers {
		layers[li] = Layer{
			InputNorm:    syn(D, li*13+1),
			Mixer:        mkGatedDeltaMixer(cfg, D, li*13+20),
			PostAttnNorm: syn(D, li*13+2),
			MLP:          &MLP{Gate: syn(FF*D, li*13+3), Up: syn(FF*D, li*13+4), Down: syn(D*FF, li*13+5), FF: FF},
		}
	}
	return &ComposedModel{
		Embed: syn(vocab*D, 100), Layers: layers, NormF: syn(D, 101), Output: nil,
		D: D, Vocab: vocab, Eps: 1e-5,
	}
}

// TestComposedDecodeEqualsPrefill is the orchestration correctness: stepping a sequence one token at a
// time through a fresh session (each layer threading its gated-delta state) produces hidden states
// BIT-EXACT to a single prefill pass — the layer loop (norm → mixer → residual → norm → SwiGLU → residual)
// plus the recurrent state threading reproduce prefill, the requirement for streaming hybrid decode.
func TestComposedDecodeEqualsPrefill(t *testing.T) {
	const D, vocab, nLayers, FF = 8, 32, 3, 16
	m := mkComposedModel(nLayers, D, vocab, FF)
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
				t.Fatalf("token %d hidden[%d] = %v != prefill %v (composed decode diverged)", t0, i, h[i], prefill[t0*D+i])
			}
		}
	}
	t.Logf("composed decode == prefill bit-exact over %d tokens, %d gated-delta layers + SwiGLU", len(tokens), nLayers)
}

// TestComposedGenerate checks the greedy generate loop runs and is deterministic.
func TestComposedGenerate(t *testing.T) {
	m := mkComposedModel(2, 8, 32, 16)
	prompt := []int32{1, 2, 3}
	g1, err := NewSession(m).Generate(prompt, 5, -1)
	if err != nil {
		t.Fatalf("generate: %v", err)
	}
	if len(g1) != 5 {
		t.Fatalf("generated %d, want 5", len(g1))
	}
	g2, _ := NewSession(m).Generate(prompt, 5, -1)
	for i := range g1 {
		if g1[i] != g2[i] {
			t.Fatalf("non-deterministic at %d: %d != %d", i, g1[i], g2[i])
		}
	}
	t.Logf("composed Generate: prefill→recurrent decode→head produced %v (deterministic)", g1)
}

// TestComposedSessionSnapshotRestoreByteIdentical proves the ComposedSession snapshot/rollback contract
// that Qwen MTP speculative decode needs: Snapshot() taken mid-decode is independent of the live session
// (a later mutation of the session's own state can never reach back and perturb an already-taken
// snapshot), and Restore() rolls the session back exactly — replaying the SAME block of tokens after
// Restore reproduces byte-identical hidden output to the first time that block ran. Exercises a REAL
// recurrent mixer (mkComposedModel's layers are all gated-delta — the mixer family MTP speculative decode
// cares about), not nil/empty state: block1 is run first specifically to make every layer's state non-nil
// before Snapshot is ever called.
func TestComposedSessionSnapshotRestoreByteIdentical(t *testing.T) {
	const D, vocab, nLayers, FF = 8, 32, 3, 16
	m := mkComposedModel(nLayers, D, vocab, FF)
	sess := NewSession(m)

	block1 := []int32{1, 5, 9, 2}
	if _, err := sess.Forward(block1); err != nil {
		t.Fatalf("block1 forward: %v", err)
	}
	for li, st := range sess.states {
		if st == nil {
			t.Fatalf("layer %d state is nil after block1 — precondition (real recurrent state) failed", li)
		}
	}

	snap := sess.Snapshot()

	// Independence: directly corrupt the LIVE session's state backing array (bypassing Forward entirely,
	// so this holds regardless of whether any mixer's Forward ever mutates a prior state in place) and
	// confirm the snapshot taken a moment ago is unaffected — proves CloneState copied into a fresh
	// backing array rather than aliasing s.states. A snapshot that merely re-boxed the same slices would
	// fail this even though it would still pass the coarser "advance and compare" check below.
	live0, ok := sess.states[0].(gatedDeltaState)
	if !ok || len(live0.conv) == 0 {
		t.Fatalf("layer 0 state not a populated gatedDeltaState: %#v", sess.states[0])
	}
	sentinel := live0.conv[0]
	live0.conv[0] = sentinel + 12345 // mutates the shared backing array — visible via sess.states[0] too
	if snapConv := snap[0].(gatedDeltaState).conv[0]; snapConv == sentinel+12345 {
		t.Fatalf("Snapshot aliased the live session's state — corrupting sess.states changed the snapshot")
	}
	live0.conv[0] = sentinel // undo the corruption so the block2 runs below start from the true state

	block2 := []int32{7, 3, 4, 6}
	out1, err := sess.Forward(block2)
	if err != nil {
		t.Fatalf("block2 run 1: %v", err)
	}
	out1 = append([]float32(nil), out1...) // defensive copy before the session advances further

	sess.Restore(snap)

	// Restore must not alias the caller's snap slice either: the restored per-layer state must be a
	// distinct backing array from the one still reachable through snap.
	restored0, ok := sess.states[0].(gatedDeltaState)
	if !ok {
		t.Fatalf("layer 0 state not gatedDeltaState after Restore: %#v", sess.states[0])
	}
	if len(restored0.conv) > 0 && len(snap[0].(gatedDeltaState).conv) > 0 &&
		&restored0.conv[0] == &snap[0].(gatedDeltaState).conv[0] {
		t.Fatalf("Restore aliased s.states with the caller's snap slice")
	}

	out2, err := sess.Forward(block2)
	if err != nil {
		t.Fatalf("block2 run 2 (post-restore): %v", err)
	}
	if len(out1) != len(out2) {
		t.Fatalf("output length mismatch: %d vs %d", len(out1), len(out2))
	}
	for i := range out1 {
		if out1[i] != out2[i] {
			t.Fatalf("hidden[%d] = %v after restore, want %v (byte-identical to the pre-restore run) — snapshot/restore diverged", i, out2[i], out1[i])
		}
	}
	t.Logf("Snapshot/Restore byte-identical over %d gated-delta layers, block2 replayed after rollback", nLayers)
}

// TestMatNTIntoDeviceHook pins the ProjMatMulInto seam: the hook fires only at
// or above the deviceMinWork floor, its result is returned verbatim, and a
// device error falls back to the host path (deterministic either way).
func TestMatNTIntoDeviceHook(t *testing.T) {
	defer func() { ProjMatMulInto = nil }()

	var calls int
	sentinel := []float32{42}
	ProjMatMulInto = func(out, x, w []float32, M, K, N int) ([]float32, error) {
		calls++
		return sentinel, nil
	}
	// Below the floor: 4·4·4 MACs — hook must NOT fire.
	small := matNTInto(nil, make([]float32, 16), make([]float32, 16), 4, 4, 4)
	if calls != 0 {
		t.Fatalf("device hook fired below deviceMinWork (calls=%d)", calls)
	}
	if len(small) != 16 {
		t.Fatalf("host path returned %d elems, want 16", len(small))
	}
	// At the floor: 1<<20 MACs (M=1, K=1024, N=1024) — hook result verbatim.
	in := make([]float32, 1024)
	w := make([]float32, 1024*1024)
	got := matNTInto(nil, in, w, 1, 1024, 1024)
	if calls != 1 || len(got) != 1 || got[0] != 42 {
		t.Fatalf("device hook not used verbatim at floor (calls=%d, got=%v)", calls, got[:min(len(got), 4)])
	}
	// Device error: host fallback produces the real result.
	ProjMatMulInto = func(out, x, w []float32, M, K, N int) ([]float32, error) {
		return nil, core.NewError("synthetic device failure")
	}
	fallback := matNTInto(nil, in, w, 1, 1024, 1024)
	if len(fallback) != 1024 {
		t.Fatalf("error fallback returned %d elems, want 1024", len(fallback))
	}
	for i, v := range fallback {
		if v != 0 {
			t.Fatalf("fallback[%d]=%v, want 0 (zero inputs)", i, v)
		}
	}
}
