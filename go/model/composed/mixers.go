// SPDX-Licence-Identifier: EUPL-1.2

package composed

import "dappco.re/go/inference/model/qwen3"

// mixers.go adapts the concrete sequence mixers to the composed Mixer interface. Each wraps a family's
// block forward and carries that family's state shape; the session threads the state opaquely. Cut 1 wires
// the gated-delta (Qwen 3.6 linear_attention) mixer; the full-attention mixer is Cut 2.

// gatedDeltaMixer adapts the Qwen 3.6 gated-delta block. Its state is the causal-conv ring + the delta
// matrix, carried as gatedDeltaState.
type gatedDeltaMixer struct {
	w   *qwen3.GatedDeltaWeights
	cfg qwen3.GatedDeltaConfig
}

type gatedDeltaState struct{ conv, delta []float32 }

// NewGatedDeltaMixer builds a gated-delta mixer for one layer.
func NewGatedDeltaMixer(w *qwen3.GatedDeltaWeights, cfg qwen3.GatedDeltaConfig) Mixer {
	return &gatedDeltaMixer{w: w, cfg: cfg}
}

func (m *gatedDeltaMixer) Kind() string { return "gated_deltanet" }

func (m *gatedDeltaMixer) Forward(h []float32, L, D int, prior any) ([]float32, any, error) {
	var pc, pd []float32
	if st, ok := prior.(gatedDeltaState); ok {
		pc, pd = st.conv, st.delta
	}
	out, nc, nd, err := qwen3.GatedDeltaForwardF32(h, m.w, m.cfg, pc, pd, L, D)
	if err != nil {
		return nil, nil, err
	}
	return out, gatedDeltaState{conv: nc, delta: nd}, nil
}
