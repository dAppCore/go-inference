// SPDX-Licence-Identifier: EUPL-1.2

package composed

import (
	"slices"

	"dappco.re/go/inference/model/arch/rwkv7"
)

// ResidualNormMLPProjRWKV7InputDevice folds a predecessor's projection-fused tail together with this
// layer's input RMSNorm and six RWKV-7 projection GEMMs. A nil hook or error keeps the ordinary path.
var ResidualNormMLPProjRWKV7InputDevice func(
	mixerHidden, projW, h, normW, gate, up, down []float32, L, D, mixCols, FF int, eps float32,
	nextNormW, nextRW, nextWW, nextKW, nextVW, nextAW, nextBW []float32, nextHK, nextHV int,
) (y, r, w, k, v, a, b []float32, err error)

// rwkv7.go adapts the RWKV-7 time-mix block (model/rwkv7) to the composed Mixer interface — Cut 4 of the
// composed-mixer roster (mixers.go's gated-delta is Cut 1, attention.go's full-attention Cut 2, mamba2.go
// Cut 3): the other half of this breadth extension — the proj-fused tail already serving attention and
// gated-delta layers reaches an RWKV-7 layer too. Its state is the single [H,K,V] recurrence state,
// carried as rwkv7State. Simpler than mamba2Mixer: RWKV-7's time-mix has no gate/norm between the
// recurrence read-out and out_proj (see rwkv7.BlockForwardScratchNoProjF32), so forwardNoProj hands back
// the recurrence output directly.

// rwkv7Mixer wraps one layer's RWKV-7 block weights + geometry.
type rwkv7Mixer struct {
	w   *rwkv7.BlockWeights
	cfg rwkv7.BlockConfig
}

// rwkv7State threads the recurrent [H,K,V] state AND the projection scratch — same
// per-session/never-on-shared-weights ownership as gatedDeltaState/mamba2State. sc is created lazily on
// the first (nil-prior) step.
type rwkv7State struct {
	state []float32
	sc    *rwkv7.BlockScratch
}

// NewRWKV7Mixer builds an RWKV-7 mixer for one layer.
func NewRWKV7Mixer(w *rwkv7.BlockWeights, cfg rwkv7.BlockConfig) Mixer {
	return &rwkv7Mixer{w: w, cfg: cfg}
}

func (m *rwkv7Mixer) Kind() string { return "rwkv7" }

// CloneState returns a deep copy of RWKV-7 state: a fresh backing array for the [H,K,V] recurrence state,
// with the projection scratch sc nil'd — the same per-Forward-reusable-workspace rationale as the other
// three mixers (rwkv7.BlockScratch reallocates lazily on a nil prior already).
func (m *rwkv7Mixer) CloneState(prior any) any {
	st, ok := prior.(rwkv7State)
	if !ok {
		return nil
	}
	return rwkv7State{state: slices.Clone(st.state)}
}

func (m *rwkv7Mixer) Forward(h []float32, L, D int, prior any) ([]float32, any, error) {
	var ps []float32
	var sc *rwkv7.BlockScratch
	if st, ok := prior.(rwkv7State); ok {
		ps, sc = st.state, st.sc
	}
	if sc == nil {
		sc = &rwkv7.BlockScratch{}
	}
	out, ns, err := rwkv7.BlockForwardScratchF32(h, m.w, m.cfg, ps, L, D, sc)
	if err != nil {
		return nil, nil, err
	}
	return out, rwkv7State{state: ns, sc: sc}, nil
}

// forwardNoProj runs the RWKV-7 time-mix block up to but NOT including out_proj, returning the recurrence
// read-out [L,hv], the out_proj weight OutProj [D,hv] and mixCols=hv, plus the advanced [H,K,V] state. It
// implements composed.projMixer so the session folds out_proj into the FFN-tail command buffer; the
// recurrent state advances identically to Forward.
func (m *rwkv7Mixer) forwardNoProj(h []float32, L, D int, prior any) (mixerHidden, projW []float32, mixCols int, next any, err error) {
	var ps []float32
	var sc *rwkv7.BlockScratch
	if st, ok := prior.(rwkv7State); ok {
		ps, sc = st.state, st.sc
	}
	if sc == nil {
		sc = &rwkv7.BlockScratch{}
	}
	o, hv, ns, err := rwkv7.BlockForwardScratchNoProjF32(h, m.w, m.cfg, ps, L, D, sc)
	if err != nil {
		return nil, nil, 0, nil, err
	}
	return o, m.w.OutProj, hv, rwkv7State{state: ns, sc: sc}, nil
}

func (m *rwkv7Mixer) forwardFromInput(r, w, k, v, a, b []float32, L, D int, prior any) (mixerHidden, projW []float32, mixCols int, next any, err error) {
	var ps []float32
	var sc *rwkv7.BlockScratch
	if st, ok := prior.(rwkv7State); ok {
		ps, sc = st.state, st.sc
	}
	if sc == nil {
		sc = &rwkv7.BlockScratch{}
	}
	o, hv, ns, err := rwkv7.BlockForwardScratchFromInputF32(r, w, k, v, a, b, m.w, m.cfg, ps, L, D, sc)
	if err != nil {
		return nil, nil, 0, nil, err
	}
	return o, m.w.OutProj, hv, rwkv7State{state: ns, sc: sc}, nil
}
