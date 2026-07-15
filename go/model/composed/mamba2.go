// SPDX-Licence-Identifier: EUPL-1.2

package composed

import (
	"slices"

	"dappco.re/go/inference/model/arch/mamba2"
)

// ResidualNormMLPProjMamba2InputDevice folds a predecessor's projection-fused tail together with this
// layer's input RMSNorm and single Mamba-2 InProj GEMM. A nil hook or error keeps the ordinary path.
var ResidualNormMLPProjMamba2InputDevice func(
	mixerHidden, projW, h, normW, gate, up, down []float32, L, D, mixCols, FF int, eps float32,
	nextNormW, nextInProjW []float32, nextProjDim int,
) (y, projected []float32, err error)

// mamba2.go adapts the Mamba-2 SSD block (model/mamba2) to the composed Mixer interface — Cut 3 of the
// composed-mixer roster (mixers.go's gated-delta is Cut 1, attention.go's full-attention Cut 2): a
// breadth extension, not a depth one — the proj-fused tail these cuts share already serves attention and
// gated-delta layers; this wires the SAME projMixer capability onto a Mamba-2 layer so a future hybrid
// schedule (or a pure Mamba-2 stack run through the composed session) reaches it too. Its state is the
// causal-conv ring + the SSM state, carried as mamba2State.

// mamba2Mixer wraps one layer's Mamba-2 block weights + geometry.
type mamba2Mixer struct {
	w   *mamba2.BlockWeights
	cfg mamba2.BlockConfig
}

// mamba2State threads the recurrent state (conv ring + SSM state) AND the projection scratch — same
// per-session/never-on-shared-weights ownership as gatedDeltaState (mixers.go). sc is created lazily on
// the first (nil-prior) step.
type mamba2State struct {
	conv, ssm []float32
	sc        *mamba2.BlockScratch
}

// NewMamba2Mixer builds a Mamba-2 mixer for one layer.
func NewMamba2Mixer(w *mamba2.BlockWeights, cfg mamba2.BlockConfig) Mixer {
	return &mamba2Mixer{w: w, cfg: cfg}
}

func (m *mamba2Mixer) Kind() string { return "mamba2" }

// CloneState returns a deep copy of Mamba-2 state: fresh backing arrays for the causal-conv ring and the
// SSM state, with the projection scratch sc nil'd — the same per-Forward-reusable-workspace rationale as
// attnMixer/gatedDeltaMixer (mamba2.BlockScratch reallocates lazily on a nil prior already).
func (m *mamba2Mixer) CloneState(prior any) any {
	st, ok := prior.(mamba2State)
	if !ok {
		return nil
	}
	return mamba2State{conv: slices.Clone(st.conv), ssm: slices.Clone(st.ssm)}
}

func (m *mamba2Mixer) Forward(h []float32, L, D int, prior any) ([]float32, any, error) {
	var pc, ps []float32
	var sc *mamba2.BlockScratch
	if st, ok := prior.(mamba2State); ok {
		pc, ps, sc = st.conv, st.ssm, st.sc
	}
	if sc == nil {
		sc = &mamba2.BlockScratch{}
	}
	out, nc, ns, err := mamba2.BlockForwardScratchF32(h, m.w, m.cfg, pc, ps, L, D, sc)
	if err != nil {
		return nil, nil, err
	}
	return out, mamba2State{conv: nc, ssm: ns, sc: sc}, nil
}

// forwardNoProj runs the Mamba-2 block up to but NOT including out_proj, returning the gated
// pre-projection hidden [L,dInner], the out_proj weight OutProj [D,dInner] and mixCols=dInner, plus the
// advanced state. It implements composed.projMixer so the session folds out_proj into the FFN-tail
// command buffer; the recurrent state advances identically to Forward.
func (m *mamba2Mixer) forwardNoProj(h []float32, L, D int, prior any) (mixerHidden, projW []float32, mixCols int, next any, err error) {
	var pc, ps []float32
	var sc *mamba2.BlockScratch
	if st, ok := prior.(mamba2State); ok {
		pc, ps, sc = st.conv, st.ssm, st.sc
	}
	if sc == nil {
		sc = &mamba2.BlockScratch{}
	}
	gated, dInner, nc, ns, err := mamba2.BlockForwardScratchNoProjF32(h, m.w, m.cfg, pc, ps, L, D, sc)
	if err != nil {
		return nil, nil, 0, nil, err
	}
	return gated, m.w.OutProj, dInner, mamba2State{conv: nc, ssm: ns, sc: sc}, nil
}

func (m *mamba2Mixer) forwardFromInput(projected []float32, L, D int, prior any) (mixerHidden, projW []float32, mixCols int, next any, err error) {
	var pc, ps []float32
	var sc *mamba2.BlockScratch
	if st, ok := prior.(mamba2State); ok {
		pc, ps, sc = st.conv, st.ssm, st.sc
	}
	if sc == nil {
		sc = &mamba2.BlockScratch{}
	}
	gated, dInner, nc, ns, err := mamba2.BlockForwardScratchFromInputF32(projected, m.w, m.cfg, pc, ps, L, D, sc)
	if err != nil {
		return nil, nil, 0, nil, err
	}
	return gated, m.w.OutProj, dInner, mamba2State{conv: nc, ssm: ns, sc: sc}, nil
}
