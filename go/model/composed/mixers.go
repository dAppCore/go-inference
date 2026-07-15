// SPDX-Licence-Identifier: EUPL-1.2

package composed

import "dappco.re/go/inference/model/arch/Qwen/qwen3"

// mixers.go adapts the concrete sequence mixers to the composed Mixer interface. Each wraps a family's
// block forward and carries that family's state shape; the session threads the state opaquely. Cut 1 wires
// the gated-delta (Qwen 3.6 linear_attention) mixer; the full-attention mixer is Cut 2.

// ResidualNormMLPProjGatedDeltaInputDevice is the input-side mirror of ResidualNormMLPProjDevice for a
// gated-delta NEXT layer: given the PREVIOUS layer's already-fused proj+tail inputs (identical to
// ResidualNormMLPProjDevice) PLUS THIS layer's input RMSNorm weight and its four gated-delta input
// projection weights (in_proj_qkv/z/a/b), it computes the previous layer's tail (out_proj → residual →
// post-attn RMSNorm → SwiGLU → residual = y) AND this layer's RMSNorm(y) → in_proj_qkv/z/a/b, all in ONE
// command buffer. Returns y [L,D] plus qkv [L,nextConvDim], z [L,nextVDim], a/b [L,nextVH] — the same
// shapes qwen3.GatedDeltaInputDevice returns. nil — or an error — leaves the standard per-layer path in
// charge (see composed.forwardEmb); the session then resumes this layer via
// gatedDeltaMixer.forwardFromInput instead of recomputing its input norm + projections. Declared here
// (not in composed.go) because it needs qwen3's gated-delta shapes, which only this file already imports.
var ResidualNormMLPProjGatedDeltaInputDevice func(
	mixerHidden, projW, h, normW, gate, up, down []float32, L, D, mixCols, FF int, eps float32,
	nextNormW, nextQKVW, nextZW, nextAW, nextBW []float32, nextConvDim, nextVDim, nextVH int,
) (y, qkv, z, a, b []float32, err error)

// gatedDeltaMixer adapts the Qwen 3.6 gated-delta block. Its state is the causal-conv ring + the delta
// matrix, carried as gatedDeltaState.
type gatedDeltaMixer struct {
	w   *qwen3.GatedDeltaWeights
	cfg qwen3.GatedDeltaConfig
}

// gatedDeltaState threads the recurrent state (conv ring + delta matrix) AND the projection scratch. The
// scratch is per-session memory, not carried information, but it rides the same per-layer state slot so
// one decode stream reuses its projection buffers every token (the shared mixer weights must never hold
// it — sessions are concurrent). sc is created lazily on the first (nil-prior) step.
type gatedDeltaState struct {
	conv, delta []float32
	sc          *qwen3.GatedDeltaScratch
}

// NewGatedDeltaMixer builds a gated-delta mixer for one layer.
func NewGatedDeltaMixer(w *qwen3.GatedDeltaWeights, cfg qwen3.GatedDeltaConfig) Mixer {
	return &gatedDeltaMixer{w: w, cfg: cfg}
}

func (m *gatedDeltaMixer) Kind() string { return "gated_deltanet" }

func (m *gatedDeltaMixer) Forward(h []float32, L, D int, prior any) ([]float32, any, error) {
	var pc, pd []float32
	var sc *qwen3.GatedDeltaScratch
	if st, ok := prior.(gatedDeltaState); ok {
		pc, pd, sc = st.conv, st.delta, st.sc
	}
	if sc == nil {
		sc = &qwen3.GatedDeltaScratch{}
	}
	out, nc, nd, err := qwen3.GatedDeltaForwardScratchF32(h, m.w, m.cfg, pc, pd, L, D, sc)
	if err != nil {
		return nil, nil, err
	}
	return out, gatedDeltaState{conv: nc, delta: nd, sc: sc}, nil
}

// forwardNoProj runs the gated-delta block up to but NOT including out_proj, returning the gated
// pre-projection hidden [L,vDim], the out_proj weight OutProj [D,vDim] and mixCols=vDim, plus the advanced
// state. It implements composed.projMixer so the session folds out_proj into the FFN-tail command buffer;
// the recurrent state advances identically to Forward.
func (m *gatedDeltaMixer) forwardNoProj(h []float32, L, D int, prior any) (mixerHidden, projW []float32, mixCols int, next any, err error) {
	var pc, pd []float32
	var sc *qwen3.GatedDeltaScratch
	if st, ok := prior.(gatedDeltaState); ok {
		pc, pd, sc = st.conv, st.delta, st.sc
	}
	if sc == nil {
		sc = &qwen3.GatedDeltaScratch{}
	}
	gated, vDim, nc, nd, err := qwen3.GatedDeltaForwardScratchNoProjF32(h, m.w, m.cfg, pc, pd, L, D, sc)
	if err != nil {
		return nil, nil, 0, nil, err
	}
	return gated, m.w.OutProj, vDim, gatedDeltaState{conv: nc, delta: nd, sc: sc}, nil
}

// forwardFromInput resumes the gated-delta block from ALREADY-COMPUTED input projections (qkv, z, a, b) —
// supplied by ResidualNormMLPProjGatedDeltaInputDevice, which folds the PREVIOUS layer's tail + THIS
// layer's input RMSNorm + in_proj_qkv/z/a/b into one command buffer — returning the gated pre-projection
// hidden [L,vDim], the out_proj weight OutProj [D,vDim] and mixCols=vDim, plus the advanced state, exactly
// as forwardNoProj does; only the input projection step is skipped.
func (m *gatedDeltaMixer) forwardFromInput(qkv, z, a, b []float32, L, D int, prior any) (mixerHidden, projW []float32, mixCols int, next any, err error) {
	var pc, pd []float32
	var sc *qwen3.GatedDeltaScratch
	if st, ok := prior.(gatedDeltaState); ok {
		pc, pd, sc = st.conv, st.delta, st.sc
	}
	if sc == nil {
		sc = &qwen3.GatedDeltaScratch{}
	}
	gated, vDim, nc, nd, err := qwen3.GatedDeltaForwardScratchFromInputF32(qkv, z, a, b, m.w, m.cfg, pc, pd, L, D, sc)
	if err != nil {
		return nil, nil, 0, nil, err
	}
	return gated, m.w.OutProj, vDim, gatedDeltaState{conv: nc, delta: nd, sc: sc}, nil
}
