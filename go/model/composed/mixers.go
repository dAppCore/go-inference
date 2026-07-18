// SPDX-Licence-Identifier: EUPL-1.2

package composed

import (
	"slices"

	"dappco.re/go/inference/model"
	"dappco.re/go/inference/model/attn"
)

// mixers.go adapts the concrete sequence mixers to the composed Mixer interface. Each wraps a family's
// block forward and carries that family's state shape; the session threads the state opaquely. Cut 1 wires
// the gated-delta (Qwen 3.6 linear_attention) mixer; the full-attention mixer is Cut 2.

// ResidualNormMLPProjGatedDeltaInputDevice is the input-side mirror of ResidualNormMLPProjDevice for a
// gated-delta NEXT layer: given the PREVIOUS layer's already-fused proj+tail inputs (identical to
// ResidualNormMLPProjDevice) PLUS THIS layer's input RMSNorm weight and its four gated-delta input
// projection weights (in_proj_qkv/z/a/b), it computes the previous layer's tail (out_proj → residual →
// post-attn RMSNorm → SwiGLU → residual = y) AND this layer's RMSNorm(y) → in_proj_qkv/z/a/b, all in ONE
// command buffer. Returns y [L,D] plus qkv [L,nextConvDim], z [L,nextVDim], a/b [L,nextVH] — the same
// shapes attn.GatedDeltaInputDevice returns. nil — or an error — leaves the standard per-layer path in
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
	w   *model.GatedDeltaWeights
	cfg model.GatedDeltaConfig
}

// gatedDeltaState threads the recurrent state (conv ring + delta matrix) AND the projection scratch. The
// scratch is per-session memory, not carried information, but it rides the same per-layer state slot so
// one decode stream reuses its projection buffers every token (the shared mixer weights must never hold
// it — sessions are concurrent). sc is created lazily on the first (nil-prior) step.
type gatedDeltaState struct {
	conv, delta []float32
	sc          *attn.GatedDeltaScratch
}

// NewGatedDeltaMixer builds a gated-delta mixer for one layer.
func NewGatedDeltaMixer(w *model.GatedDeltaWeights, cfg model.GatedDeltaConfig) Mixer {
	return &gatedDeltaMixer{w: w, cfg: cfg}
}

func (m *gatedDeltaMixer) Kind() string { return "gated_deltanet" }

// CloneState returns a deep copy of gated-delta state: fresh backing arrays for the causal-conv ring and
// the delta matrix, with the projection scratch sc nil'd — sc (attn.GatedDeltaScratch) is per-Forward
// reusable workspace, not logical state, and Forward/forwardNoProj/forwardFromInput already reallocate it
// lazily whenever it is nil (the same path a fresh, nil-prior session already takes), so nil-ing it here
// is byte-identical to that path and stops the clone aliasing the live session's scratch buffers. When
// the device block path holds the state (sc.Device — the host conv/delta slices are then nil and stale),
// the clone reads it back through the engine's export seam; the clone itself carries host slices only,
// so a restored session re-primes its own device state lazily.
func (m *gatedDeltaMixer) CloneState(prior any) any {
	st, ok := prior.(gatedDeltaState)
	if !ok {
		return nil
	}
	if st.sc != nil && st.sc.Device != nil && attn.GatedDeltaDeviceStateExport != nil {
		if conv, delta, ok := attn.GatedDeltaDeviceStateExport(st.sc.Device); ok {
			return gatedDeltaState{conv: conv, delta: delta}
		}
	}
	return gatedDeltaState{conv: slices.Clone(st.conv), delta: slices.Clone(st.delta)}
}

func (m *gatedDeltaMixer) Forward(h []float32, L, D int, prior any) ([]float32, any, error) {
	var pc, pd []float32
	var sc *attn.GatedDeltaScratch
	if st, ok := prior.(gatedDeltaState); ok {
		pc, pd, sc = st.conv, st.delta, st.sc
	}
	if sc == nil {
		sc = &attn.GatedDeltaScratch{}
	}
	qkv, z, a, b, err := attn.GatedDeltaInputProjectF32(h, m.w, m.cfg, L, D, sc)
	if err != nil {
		return nil, nil, err
	}
	if gated, engaged, derr := attn.GatedDeltaBlockDeviceTry(sc, qkv, z, a, b, m.w, m.cfg, pc, pd, L); engaged {
		if derr != nil {
			return nil, nil, derr
		}
		out, oerr := attn.GatedDeltaOutProjF32(gated, m.w, m.cfg, L, D, sc)
		if oerr != nil {
			return nil, nil, oerr
		}
		return out, gatedDeltaState{sc: sc}, nil
	}
	gated, _, nc, nd, err := attn.GatedDeltaForwardScratchFromInputF32(qkv, z, a, b, m.w, m.cfg, pc, pd, L, D, sc)
	if err != nil {
		return nil, nil, err
	}
	out, err := attn.GatedDeltaOutProjF32(gated, m.w, m.cfg, L, D, sc)
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
	var sc *attn.GatedDeltaScratch
	if st, ok := prior.(gatedDeltaState); ok {
		pc, pd, sc = st.conv, st.delta, st.sc
	}
	if sc == nil {
		sc = &attn.GatedDeltaScratch{}
	}
	qkv, z, a, b, err := attn.GatedDeltaInputProjectF32(h, m.w, m.cfg, L, D, sc)
	if err != nil {
		return nil, nil, 0, nil, err
	}
	if gated, engaged, derr := attn.GatedDeltaBlockDeviceTry(sc, qkv, z, a, b, m.w, m.cfg, pc, pd, L); engaged {
		if derr != nil {
			return nil, nil, 0, nil, derr
		}
		return gated, m.w.OutProj, m.cfg.VDim(), gatedDeltaState{sc: sc}, nil
	}
	gated, vDim, nc, nd, err := attn.GatedDeltaForwardScratchFromInputF32(qkv, z, a, b, m.w, m.cfg, pc, pd, L, D, sc)
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
	var sc *attn.GatedDeltaScratch
	if st, ok := prior.(gatedDeltaState); ok {
		pc, pd, sc = st.conv, st.delta, st.sc
	}
	if sc == nil {
		sc = &attn.GatedDeltaScratch{}
	}
	if gated, engaged, derr := attn.GatedDeltaBlockDeviceTry(sc, qkv, z, a, b, m.w, m.cfg, pc, pd, L); engaged {
		if derr != nil {
			return nil, nil, 0, nil, derr
		}
		return gated, m.w.OutProj, m.cfg.VDim(), gatedDeltaState{sc: sc}, nil
	}
	gated, vDim, nc, nd, err := attn.GatedDeltaForwardScratchFromInputF32(qkv, z, a, b, m.w, m.cfg, pc, pd, L, D, sc)
	if err != nil {
		return nil, nil, 0, nil, err
	}
	return gated, m.w.OutProj, vDim, gatedDeltaState{conv: nc, delta: nd, sc: sc}, nil
}

// forwardBF16Layer runs one WHOLE dense bf16 gated-delta layer through the bf16 layer seam
// (attn.GatedDeltaBF16LayerDeviceTry) — the quant layer fold's raw-bf16 twin, same engagement and
// state contracts.
func (m *gatedDeltaMixer) forwardBF16Layer(h, inputNorm, postNorm []float32, gate, up, down *model.BF16Weight, FF, L, D int, eps float32, prior any) (y []float32, next any, engaged bool, err error) {
	var pc, pd []float32
	var sc *attn.GatedDeltaScratch
	if st, ok := prior.(gatedDeltaState); ok {
		pc, pd, sc = st.conv, st.delta, st.sc
	}
	if sc == nil {
		sc = &attn.GatedDeltaScratch{}
	}
	y, engaged, err = attn.GatedDeltaBF16LayerDeviceTry(sc, h, inputNorm, m.w, m.cfg, postNorm, gate, up, down, L, D, FF, eps, pc, pd)
	if !engaged {
		return nil, nil, false, nil
	}
	if err != nil {
		return nil, nil, true, err
	}
	return y, gatedDeltaState{sc: sc}, true, nil
}

// chainableBF16 reports whether this gated-delta layer can ride the chained device path.
func (m *gatedDeltaMixer) chainableBF16(mlp *MLP) bool {
	return attn.GatedDeltaBF16ChainLayerDevice != nil &&
		attn.GatedDeltaChainGeometryOK != nil && attn.GatedDeltaChainGeometryOK(m.cfg) &&
		m.w.InProjQKVB != nil && m.w.InProjAB != nil && m.w.InProjBB != nil && m.w.InProjZB != nil && m.w.OutProjB != nil &&
		mlp != nil && mlp.GateB != nil && mlp.UpB != nil && mlp.DownB != nil
}

// chainBF16Layer encodes this gated-delta layer onto the chain and returns the advanced state.
func (m *gatedDeltaMixer) chainBF16Layer(ctx any, inputNorm, postNorm []float32, mlp *MLP, eps float32, prior any) (next any, err error) {
	var pc, pd []float32
	var sc *attn.GatedDeltaScratch
	if st, ok := prior.(gatedDeltaState); ok {
		pc, pd, sc = st.conv, st.delta, st.sc
	}
	if sc == nil {
		sc = &attn.GatedDeltaScratch{}
	}
	if cerr := attn.GatedDeltaBF16ChainLayerDevice(ctx, sc, inputNorm, m.w, m.cfg, postNorm, mlp.GateB, mlp.UpB, mlp.DownB, pc, pd, mlp.FF, eps); cerr != nil {
		return nil, cerr
	}
	return gatedDeltaState{sc: sc}, nil
}

// chainableQuant reports whether this gated-delta layer can ride the chained device path over
// its packed weights — the quant twin of chainableBF16.
func (m *gatedDeltaMixer) chainableQuant(mlp *MLP) bool {
	return attn.GatedDeltaQuantChainLayerDevice != nil &&
		attn.GatedDeltaChainGeometryOK != nil && attn.GatedDeltaChainGeometryOK(m.cfg) &&
		m.w.InProjQKVQ != nil && m.w.InProjAQ != nil && m.w.InProjBQ != nil && m.w.InProjZQ != nil && m.w.OutProjQ != nil &&
		mlp != nil && mlp.GateQ != nil && mlp.UpQ != nil && mlp.DownQ != nil
}

// chainQuantLayer encodes this gated-delta layer onto the chain over packed weights and returns
// the advanced state — the quant twin of chainBF16Layer.
func (m *gatedDeltaMixer) chainQuantLayer(ctx any, inputNorm, postNorm []float32, mlp *MLP, eps float32, prior any) (next any, err error) {
	var pc, pd []float32
	var sc *attn.GatedDeltaScratch
	if st, ok := prior.(gatedDeltaState); ok {
		pc, pd, sc = st.conv, st.delta, st.sc
	}
	if sc == nil {
		sc = &attn.GatedDeltaScratch{}
	}
	if cerr := attn.GatedDeltaQuantChainLayerDevice(ctx, sc, inputNorm, m.w, m.cfg, postNorm, mlp.GateQ, mlp.UpQ, mlp.DownQ, pc, pd, mlp.FF, eps); cerr != nil {
		return nil, cerr
	}
	return gatedDeltaState{sc: sc}, nil
}

// chainableMoEQuant reports whether this gated-delta layer, whose FFN is a MoE, can ride the chained
// device path over packed weights — chainableQuant's gated-delta conditions plus the MoE tail's
// recordability (batched experts + lean gather present).
func (m *gatedDeltaMixer) chainableMoEQuant(moe *MoEMLP) bool {
	return GatedDeltaQuantChainMoELayerDevice != nil && MoEChainRecordable != nil && MoEChainRecordable(moe) &&
		attn.GatedDeltaChainGeometryOK != nil && attn.GatedDeltaChainGeometryOK(m.cfg) &&
		m.w.InProjQKVQ != nil && m.w.InProjAQ != nil && m.w.InProjBQ != nil && m.w.InProjZQ != nil && m.w.OutProjQ != nil
}

// chainQuantMoELayer encodes this gated-delta layer + its MoE FFN tail onto the chain over packed
// weights — the MoE twin of chainQuantLayer.
func (m *gatedDeltaMixer) chainQuantMoELayer(ctx any, inputNorm, postNorm []float32, moe *MoEMLP, eps float32, prior any) (next any, err error) {
	var pc, pd []float32
	var sc *attn.GatedDeltaScratch
	if st, ok := prior.(gatedDeltaState); ok {
		pc, pd, sc = st.conv, st.delta, st.sc
	}
	if sc == nil {
		sc = &attn.GatedDeltaScratch{}
	}
	if cerr := GatedDeltaQuantChainMoELayerDevice(ctx, sc, inputNorm, m.w, m.cfg, postNorm, moe, pc, pd, eps); cerr != nil {
		return nil, cerr
	}
	return gatedDeltaState{sc: sc}, nil
}

// forwardQuantLayer runs one WHOLE packed gated-delta layer through the device layer seam
// (attn.GatedDeltaQuantLayerDeviceTry): input norm, the five packed projections, the block and the
// packed FFN tail in one command buffer, x in / y out, state device-resident. engaged=false leaves
// the standard quant branch (per-stage seams) in charge; engaged=true with err propagates — the
// device owns this sequence's state (see the Try contract). Called from forwardEmb's quant branch
// for a dense packed SwiGLU at residual scale 1.
func (m *gatedDeltaMixer) forwardQuantLayer(h, inputNorm, postNorm []float32, gate, up, down *model.QuantWeight, FF, L, D int, eps float32, prior any) (y []float32, next any, engaged bool, err error) {
	var pc, pd []float32
	var sc *attn.GatedDeltaScratch
	if st, ok := prior.(gatedDeltaState); ok {
		pc, pd, sc = st.conv, st.delta, st.sc
	}
	if sc == nil {
		sc = &attn.GatedDeltaScratch{}
	}
	y, engaged, err = attn.GatedDeltaQuantLayerDeviceTry(sc, h, inputNorm, m.w, m.cfg, postNorm, gate, up, down, L, D, FF, eps, pc, pd)
	if !engaged {
		return nil, nil, false, nil
	}
	if err != nil {
		return nil, nil, true, err
	}
	return y, gatedDeltaState{sc: sc}, true, nil
}
