// SPDX-Licence-Identifier: EUPL-1.2

package composed

import (
	"math"
	"slices"

	core "dappco.re/go"
	"dappco.re/go/inference/model"
)

// attention.go is the full_attention mixer for the hybrid stack — qwen3-style dense attention (per-head
// QK-norm → partial rotary → GQA → causal softmax) with a growing KV cache as its state. It is the cut-2
// peer of the gated-delta mixer: same Mixer interface, but its state is a KV cache instead of a recurrent
// matrix, exactly the per-layer cache-typing metal's composed model does. Host f32; the projections use
// the package matNT (the device-GEMM path is a later optimisation, shared with the gated-delta seam).

// AttnConfig is the per-layer attention geometry. RotaryDim ≤ HeadDim (partial rotary; Qwen 3.6 uses
// 0.25·HeadDim). KVHeads ≤ Heads (GQA). OutputGate is the gated-attention flag (attn_output_gate): when
// set, q_proj emits [q ; gate] per head and the attention output is σ(gate)-gated before o_proj.
type AttnConfig struct {
	Heads, KVHeads, HeadDim, RotaryDim int
	RopeTheta, NormEps                 float32
	QKVClip                            float32
	OutputGate                         bool
	ALiBi                              bool
	QKNormalization                    model.QKNormalization
	SlidingWindow                      int
}

// AttnWeights is one layer's attention weights. QProj is [Heads*HeadDim, D] (or [2·Heads*HeadDim, D] when
// OutputGate — the [q ; gate] projection); K/VProj [KVHeads*HeadDim, D]; OProj [D, Heads*HeadDim];
// QNorm/KNorm [HeadDim] (per-head RMSNorm, plain — qwen is not gemma).
type AttnWeights struct {
	QProj, KProj, VProj, OProj []float32
	// Packed forms in a quant checkpoint (nil ⇒ the dense f32 field is used). When set, the mixer
	// dispatches q/k/v/o to the quant matvec seam instead of the f32 matNT + the f32 AttnQKVDevice fuse.
	QProjQ, KProjQ, VProjQ, OProjQ *model.QuantWeight
	QProjB, KProjB, VProjB, OProjB *model.BF16Weight
	QNorm, KNorm                   []float32
}

type attnMixer struct {
	w   *AttnWeights
	cfg AttnConfig
}

// attnState is the KV cache: n past tokens, k/v laid out [n, KVHeads, HeadDim] (rotary already applied to
// the cached keys). It also carries the projection scratch — per-session memory, not cache content, but
// threaded in the same per-layer state slot so one decode stream reuses its q/k/v/o buffers every token
// (the shared mixer weights must never hold it). sc is nil ⇒ the projections allocate fresh.
type attnState struct {
	k, v []float32
	n    int
	sc   *attnScratch
}

// attnScratch holds the reusable projection-output buffers: qRaw is the q_proj output (or the [q;gate]
// raw when OutputGate); k, v the key/value projections; o the out-proj. The de-interleaved q/gate and the
// grown KV cache are NOT scratch — they are per-token state that outlives the call or grows each step.
type attnScratch struct {
	qRaw, k, v, o []float32
	// Device carries the engine resident KV handle when the full-layer device path is engaged.
	Device any
}

// AttnQKVDevice is the fused q/k/v projection hook for the full-attention mixer: q_proj, k_proj and
// v_proj all read the same hidden h [L,D], so a backend can encode the three GEMMs into ONE command
// buffer (k/v — sub-floor standalone, their host matmul is serial — ride along inside it). qCols is
// H*HD ungated or 2·H*HD gated ([q;gate] per head); kvCols is KVH*HD. nil ⇒ the per-projection
// matNTInto path (the device hook for q, host for the sub-floor k/v). Same AX-8 shape as MLPDevice:
// composed declares the hook, engine/metal binds it; the lib never imports the backend.
var AttnQKVDevice func(h, qW, kW, vW []float32, L, D, qCols, kvCols int) (q, k, v []float32, err error)

// ResidualNormMLPProjAttnInputDevice is the input-side mirror of ResidualNormMLPProjDevice: given the
// PREVIOUS layer's already-fused proj+tail inputs (identical to ResidualNormMLPProjDevice) PLUS THIS
// layer's input RMSNorm weight and q/k/v projection weights, it computes the previous layer's tail
// (o_proj/out_proj → residual → post-attn RMSNorm → SwiGLU → residual = y) AND this layer's RMSNorm(y) →
// q/k/v projections, all in ONE command buffer — y never crosses the host floor before feeding this
// layer's projections. Returns y [L,D] (the previous layer's output, needed on the host for THIS layer's
// mixer-output residual add) plus q [L,qCols], k/v [L,kvCols] (the same shapes AttnQKVDevice returns). nil
// — or an error — leaves the standard per-layer path in charge (see composed.forwardEmb); the session then
// resumes this layer via attnMixer.forwardFromQKV instead of recomputing its input norm + projections.
var ResidualNormMLPProjAttnInputDevice func(
	mixerHidden, projW, h, normW, gate, up, down []float32, L, D, mixCols, FF int, eps float32,
	nextNormW, nextQW, nextKW, nextVW []float32, nextQCols, nextKVCols int,
) (y, q, k, v []float32, err error)

// NewAttnMixer builds a full-attention mixer for one layer.
func NewAttnMixer(w *AttnWeights, cfg AttnConfig) Mixer { return &attnMixer{w: w, cfg: cfg} }

func (m *attnMixer) Kind() string {
	if m.cfg.SlidingWindow > 0 {
		return "sliding_attention"
	}
	return "full_attention"
}

// CloneState returns a deep copy of an attention KV cache: fresh backing arrays for the cached keys/
// values and n copied by value, with the projection scratch sc nil'd — sc (attnScratch) is per-Forward
// reusable workspace, not logical state, and forwardNoProj/forwardFromQKV already reallocate it lazily
// whenever it is nil (the same path a fresh, nil-prior session already takes), so nil-ing it here is
// byte-identical to that path and stops the clone aliasing the live session's projection buffers.
func (m *attnMixer) CloneState(prior any) any {
	st, ok := prior.(attnState)
	if !ok {
		return nil
	}
	if st.sc != nil && st.sc.Device != nil && AttnKVExportDevice != nil {
		if k, v, n, ok := AttnKVExportDevice(st.sc.Device); ok {
			return attnState{k: k, v: v, n: n}
		}
	}
	return attnState{k: slices.Clone(st.k), v: slices.Clone(st.v), n: st.n}
}

// rmsNormHead RMS-normalises a single [HeadDim] vector in place by weight w.
func rmsNormHead(x, w []float32, eps float32) {
	if len(w) == 0 { // Llama/Mistral have no per-head QK norm.
		return
	}
	var ss float64
	for _, e := range x {
		ss += float64(e) * float64(e)
	}
	r := math.Sqrt(ss/float64(len(x)) + float64(eps))
	for i := range x {
		x[i] = float32(float64(x[i]) / r * float64(w[i]))
	}
}

// l2NormHead applies Llama 4's non-parametric per-head normalisation.
func l2NormHead(x []float32, eps float32) {
	var ss float64
	for _, value := range x {
		ss += float64(value) * float64(value)
	}
	inv := 1 / math.Sqrt(ss/float64(len(x))+float64(eps))
	for i := range x {
		x[i] = float32(float64(x[i]) * inv)
	}
}

// layerNormHead applies Cohere's learned per-head LayerNorm. Unlike RMSNorm it
// centres each head before variance normalisation.
func layerNormHead(x, w []float32, eps float32) {
	if len(w) == 0 {
		return
	}
	var mean float64
	for _, value := range x {
		mean += float64(value)
	}
	mean /= float64(len(x))
	var variance float64
	for _, value := range x {
		delta := float64(value) - mean
		variance += delta * delta
	}
	inv := 1 / math.Sqrt(variance/float64(len(x))+float64(eps))
	for i := range x {
		x[i] = float32((float64(x[i]) - mean) * inv * float64(w[i]))
	}
}

// applyRotaryHalf rotates the first rotaryDim dims of a [HeadDim] vector at position pos (the rotate_half
// convention: pair i with i+rotaryDim/2), leaving dims [rotaryDim:] unchanged.
func applyRotaryHalf(x []float32, pos, rotaryDim int, theta float64) {
	half := rotaryDim / 2
	for i := range half {
		freq := 1.0 / math.Pow(theta, float64(2*i)/float64(rotaryDim))
		ang := float64(pos) * freq
		c, s := math.Cos(ang), math.Sin(ang)
		a, b := float64(x[i]), float64(x[i+half])
		x[i] = float32(a*c - b*s)
		x[i+half] = float32(b*c + a*s)
	}
}

// AttnBF16FrontDevice is the attention fold's FRONT seam (#26): input RMSNorm + the three raw-bf16
// q/k/v projections in ONE command buffer, q/k/v returned for the host attention core. AX-8: the
// lib declares, the backend binds; nil ⇒ the per-stage path serves.
var AttnBF16FrontDevice func(x, inputNorm []float32, qw, kw, vw *model.BF16Weight, L, D, qCols, kvCols int, eps float32) (q, k, v []float32, err error)

// AttnBF16TailDevice is the fold's TAIL seam: o_proj + the FFN tail (residual + post-norm + bf16
// SwiGLU + residual) in ONE command buffer. Stateless — the KV cache advanced in the host core —
// so a decline here falls back to the host tail with no state hazard.
var AttnBF16TailDevice func(h, attnOut []float32, ow *model.BF16Weight, postNorm []float32, gate, up, down *model.BF16Weight, L, D, mixCols, FF int, eps float32) (y []float32, err error)

// AttnQuantFrontDevice / AttnQuantTailDevice are the fold seams' PACKED twins — the same two
// command buffers over a quant checkpoint's codes (the 27B's attention layers). Identical
// contracts to the bf16 pair.
var AttnQuantFrontDevice func(x, inputNorm []float32, qw, kw, vw *model.QuantWeight, L, D, qCols, kvCols int, eps float32) (q, k, v []float32, err error)
var AttnQuantTailDevice func(h, attnOut []float32, ow *model.QuantWeight, postNorm []float32, gate, up, down *model.QuantWeight, L, D, mixCols, FF int, eps float32) (y []float32, err error)

// forwardQuantLayer runs one WHOLE packed attention layer through the fold seams — the quant twin
// of forwardBF16Layer with the same engagement and failure semantics.
func (m *attnMixer) forwardQuantLayer(h, inputNorm, postNorm []float32, mlp *MLP, L, D int, eps float32, prior any) (y []float32, next any, engaged bool, err error) {
	if AttnQuantFrontDevice == nil || AttnQuantTailDevice == nil ||
		m.w.QProjQ == nil || m.w.KProjQ == nil || m.w.VProjQ == nil || m.w.OProjQ == nil ||
		mlp.GateQ == nil || mlp.UpQ == nil || mlp.DownQ == nil {
		return nil, nil, false, nil
	}
	cfg := m.cfg
	qCols := cfg.Heads * cfg.HeadDim
	if cfg.OutputGate {
		qCols = 2 * cfg.Heads * cfg.HeadDim
	}
	kvCols := cfg.KVHeads * cfg.HeadDim
	qRaw, k, v, ferr := AttnQuantFrontDevice(h, inputNorm, m.w.QProjQ, m.w.KProjQ, m.w.VProjQ, L, D, qCols, kvCols, eps)
	if ferr != nil {
		return nil, nil, false, nil // nothing touched — the per-stage path serves this token
	}
	var st attnState
	if p, ok := prior.(attnState); ok {
		st = p
	}
	sc := st.sc
	if sc == nil {
		sc = &attnScratch{}
	}
	attnOut, _, mixCols, nextState, cerr := m.continueFromQKV(qRaw, k, v, L, D, st, sc)
	if cerr != nil {
		return nil, nil, true, cerr
	}
	if yDev, terr := AttnQuantTailDevice(h, attnOut, m.w.OProjQ, postNorm, mlp.GateQ, mlp.UpQ, mlp.DownQ, L, D, mixCols, mlp.FF, eps); terr == nil {
		return yDev, nextState, true, nil
	}
	ns := nextState.(attnState)
	ns.sc.o = matNTQuant(ns.sc.o, attnOut, m.w.OProjQ, L, mixCols, D)
	y = tailHost(append([]float32(nil), h...), ns.sc.o, postNorm, mlp, L, D, eps, 1)
	return y, nextState, true, nil
}

// AttnBF16FullLayerDevice is the WHOLE-attention-layer device seam (#26 device-KV): norm, q/k/v,
// rope+norm prep, SDPA over the RESIDENT KV cache, sigma-gate, o_proj and the FFN tail in ONE
// command buffer — only x and y cross the host. dev threads the engine opaque cache handle (nil
// first call; primed from priorK/priorV when pos0 > 0). AX-8: declared here, bound by the backend.
var AttnBF16FullLayerDevice func(dev any, x, inputNorm []float32, qw, kw, vw, ow *model.BF16Weight, qNormW, kNormW, postNorm []float32, gate, up, down *model.BF16Weight, priorK, priorV []float32, L, D, H, KVH, HD, RD, pos0, window, gated, qkNorm, FF int, eps, theta float32) (y []float32, devOut any, err error)

// AttnKVExportDevice reads a resident KV handle back into host slices — the snapshot/clone seam.
var AttnKVExportDevice func(dev any) (k, v []float32, n int, ok bool)

// AttnBF16ChainLayerDevice encodes one dense bf16 attention layer (device-KV) onto an open chain
// context — AttnBF16FullLayerDevice without its own command buffer.
var AttnBF16ChainLayerDevice func(ctx, dev any, inputNorm []float32, qw, kw, vw, ow *model.BF16Weight, qNormW, kNormW, postNorm []float32, gate, up, down *model.BF16Weight, priorK, priorV []float32, H, KVH, HD, RD, pos0, window, gated, qkNorm, FF int, eps, theta float32) (devOut any, err error)

// chainableBF16 reports whether this attention layer can ride the chained device path — the same
// conditions the full-layer arm checks, decided without touching state.
func (m *attnMixer) chainableBF16(mlp *MLP) bool {
	cfg := m.cfg
	return AttnBF16ChainLayerDevice != nil && !cfg.ALiBi && cfg.QKVClip == 0 &&
		cfg.QKNormalization != model.QKLayerNorm && cfg.RotaryDim%2 == 0 &&
		m.w.QProjB != nil && m.w.KProjB != nil && m.w.VProjB != nil && m.w.OProjB != nil &&
		mlp != nil && mlp.GateB != nil && mlp.UpB != nil && mlp.DownB != nil
}

// chainBF16Layer encodes this attention layer onto the chain and returns the advanced state. The
// chain owns execution; an error here aborts the whole forward (the device owns the sequence).
func (m *attnMixer) chainBF16Layer(ctx any, inputNorm, postNorm []float32, mlp *MLP, L, D int, eps float32, prior any) (next any, err error) {
	cfg := m.cfg
	var st attnState
	if p, ok := prior.(attnState); ok {
		st = p
	}
	sc := st.sc
	if sc == nil {
		sc = &attnScratch{}
	}
	qkNorm := 0
	switch {
	case cfg.QKNormalization == model.QKL2Norm:
		qkNorm = 2
	case len(m.w.QNorm) > 0 && len(m.w.KNorm) > 0:
		qkNorm = 1
	}
	gatedI := 0
	if cfg.OutputGate {
		gatedI = 1
	}
	theta := cfg.RopeTheta
	if theta == 0 {
		theta = 1e6
	}
	devOut, ferr := AttnBF16ChainLayerDevice(ctx, sc.Device, inputNorm,
		m.w.QProjB, m.w.KProjB, m.w.VProjB, m.w.OProjB, m.w.QNorm, m.w.KNorm, postNorm,
		mlp.GateB, mlp.UpB, mlp.DownB, st.k, st.v,
		cfg.Heads, cfg.KVHeads, cfg.HeadDim, cfg.RotaryDim, st.n, cfg.SlidingWindow, gatedI, qkNorm, mlp.FF, eps, theta)
	if ferr != nil {
		return nil, ferr
	}
	sc.Device = devOut
	return attnState{n: st.n + L, sc: sc}, nil
}

// forwardBF16Layer runs one WHOLE dense bf16 attention layer through the fold seams — [front CB]
// → host rope/cache/SDPA → [tail CB] — engaging only when both seams are bound and every
// projection carries its bf16 form. engaged=false leaves the per-stage path in charge. A front
// failure declines cleanly (no state touched); a tail failure completes on the host path (the
// cache already advanced, and the tail is stateless).
func (m *attnMixer) forwardBF16Layer(h, inputNorm, postNorm []float32, mlp *MLP, L, D int, eps float32, prior any) (y []float32, next any, engaged bool, err error) {
	if AttnBF16FrontDevice == nil || AttnBF16TailDevice == nil ||
		m.w.QProjB == nil || m.w.KProjB == nil || m.w.VProjB == nil || m.w.OProjB == nil ||
		mlp.GateB == nil || mlp.UpB == nil || mlp.DownB == nil {
		return nil, nil, false, nil
	}
	cfg := m.cfg
	qCols := cfg.Heads * cfg.HeadDim
	if cfg.OutputGate {
		qCols = 2 * cfg.Heads * cfg.HeadDim
	}
	kvCols := cfg.KVHeads * cfg.HeadDim

	// Device-KV full-layer path: the whole layer in one CB over the resident cache. Engagement
	// mirrors the state contract: a first-call decline (dev still nil) falls to the two-CB fold;
	// once the handle exists the device owns this sequence KV — propagate, never fall back.
	if AttnBF16FullLayerDevice != nil && !cfg.ALiBi && cfg.QKVClip == 0 &&
		cfg.QKNormalization != model.QKLayerNorm && cfg.RotaryDim%2 == 0 {
		var st attnState
		if p, ok := prior.(attnState); ok {
			st = p
		}
		sc := st.sc
		if sc == nil {
			sc = &attnScratch{}
		}
		qkNorm := 0
		switch {
		case cfg.QKNormalization == model.QKL2Norm:
			qkNorm = 2
		case len(m.w.QNorm) > 0 && len(m.w.KNorm) > 0:
			qkNorm = 1
		}
		gatedI := 0
		if cfg.OutputGate {
			gatedI = 1
		}
		theta := cfg.RopeTheta
		if theta == 0 {
			theta = 1e6
		}
		y, devOut, ferr := AttnBF16FullLayerDevice(sc.Device, h, inputNorm,
			m.w.QProjB, m.w.KProjB, m.w.VProjB, m.w.OProjB, m.w.QNorm, m.w.KNorm, postNorm,
			mlp.GateB, mlp.UpB, mlp.DownB, st.k, st.v,
			L, D, cfg.Heads, cfg.KVHeads, cfg.HeadDim, cfg.RotaryDim, st.n, cfg.SlidingWindow, gatedI, qkNorm, mlp.FF, eps, theta)
		if ferr == nil {
			sc.Device = devOut
			return y, attnState{n: st.n + L, sc: sc}, true, nil
		}
		if sc.Device != nil {
			return nil, nil, true, ferr
		}
		// first-call decline: the two-CB fold below serves
	}

	qRaw, k, v, ferr := AttnBF16FrontDevice(h, inputNorm, m.w.QProjB, m.w.KProjB, m.w.VProjB, L, D, qCols, kvCols, eps)
	if ferr != nil {
		return nil, nil, false, nil // nothing touched — the per-stage path serves this token
	}
	var st attnState
	if p, ok := prior.(attnState); ok {
		st = p
	}
	sc := st.sc
	if sc == nil {
		sc = &attnScratch{}
	}
	attnOut, _, mixCols, nextState, cerr := m.continueFromQKV(qRaw, k, v, L, D, st, sc)
	if cerr != nil {
		return nil, nil, true, cerr
	}
	if yDev, terr := AttnBF16TailDevice(h, attnOut, m.w.OProjB, postNorm, mlp.GateB, mlp.UpB, mlp.DownB, L, D, mixCols, mlp.FF, eps); terr == nil {
		return yDev, nextState, true, nil
	}
	// Tail decline: complete on the host — o_proj through the bf16 seam, then the host tail.
	ns := nextState.(attnState)
	ns.sc.o = matNTBF16(ns.sc.o, attnOut, m.w.OProjB, L, mixCols, D)
	y = tailHost(append([]float32(nil), h...), ns.sc.o, postNorm, mlp, L, D, eps, 1)
	return y, nextState, true, nil
}

// Forward runs attention over hidden [L,D], appending the new K/V to the cache and attending causally over
// all cached tokens, then applies o_proj. Returns out [L,D] and the grown cache. It is forwardNoProj (which
// stops one GEMM short) plus the o_proj into the session-owned scratch — the projection is split out so the
// session can instead fold it into the FFN-tail command buffer (see projMixer / ResidualNormMLPProjDevice).
func (m *attnMixer) Forward(h []float32, L, D int, prior any) ([]float32, any, error) {
	attnOut, oProj, mixCols, next, err := m.forwardNoProj(h, L, D, prior)
	if err != nil {
		return nil, nil, err
	}
	st := next.(attnState)
	if m.w.OProjB != nil { // bf16-resident o_proj (#26)
		st.sc.o = matNTBF16(st.sc.o, attnOut, m.w.OProjB, L, mixCols, D)
		return st.sc.o, st, nil
	}
	if m.w.OProjQ != nil { // packed o_proj — oProj (m.w.OProj) is nil in a quant checkpoint
		st.sc.o = matNTQuant(st.sc.o, attnOut, m.w.OProjQ, L, mixCols, D)
	} else {
		st.sc.o = matNTInto(st.sc.o, attnOut, oProj, L, mixCols, D)
	}
	return st.sc.o, st, nil
}

// forwardNoProj runs attention over hidden [L,D] up to but NOT including o_proj, returning the pre-projection
// attention output [L,H*HD], the o_proj weight OProj [D,H*HD] and mixCols=H*HD, plus the grown cache. The
// state advances identically to Forward; only the final projection is deferred to the caller. It computes
// its own q/k/v projections then hands off to continueFromQKV, which forwardFromQKV also feeds — the two
// entry points differ only in HOW qRaw/k/v were produced.
func (m *attnMixer) forwardNoProj(h []float32, L, D int, prior any) (mixerHidden, projW []float32, mixCols int, next any, err error) {
	cfg := m.cfg
	H, KVH := cfg.Heads, cfg.KVHeads
	qCols := H * cfg.HeadDim
	if cfg.OutputGate {
		qCols = 2 * H * cfg.HeadDim // q_proj emits [q ; gate] per head
	}
	qLen := len(m.w.QProj)
	if m.w.QProjQ != nil {
		qLen = m.w.QProjQ.OutDim * D // packed q_proj: rows·D
	}
	if m.w.QProjB != nil {
		qLen = m.w.QProjB.OutDim * D // bf16-resident q_proj: rows·D
	}
	if qLen != qCols*D {
		return nil, nil, 0, nil, core.NewError(core.Sprintf("composed.attnMixer: q_proj size mismatch: qLen %d != qCols %d x D %d (OutputGate %v)", qLen, qCols, D, cfg.OutputGate))
	}
	var st attnState
	if p, ok := prior.(attnState); ok {
		st = p
	}
	sc := st.sc
	if sc == nil {
		sc = &attnScratch{} // throwaway: nil buffers ⇒ every projection allocates, the legacy path
	}

	// q/k/v all read the hidden h [L,D]. A packed checkpoint dispatches each projection to the quant matvec
	// seam (the f32 AttnQKVDevice fuse takes f32 weights — bypassed). Otherwise, when the backend supplies
	// the fused hook and the q projection crosses the device floor, q_raw/k/v are computed in ONE command
	// buffer (k/v ride free — sub-floor standalone); else the per-projection matNTInto path runs. k/v feed
	// the KV cache either way; deterministic.
	var qRaw, k, v []float32
	kvCols := KVH * cfg.HeadDim
	if m.w.QProjB != nil { // bf16-resident q/k/v (#26): same structure as the packed branch
		sc.qRaw = matNTBF16(sc.qRaw, h, m.w.QProjB, L, D, qCols)
		qRaw = sc.qRaw
		sc.k = matNTBF16(sc.k, h, m.w.KProjB, L, D, kvCols)
		k = sc.k
		sc.v = matNTBF16(sc.v, h, m.w.VProjB, L, D, kvCols)
		v = sc.v
	} else if m.w.QProjQ != nil {
		sc.qRaw = matNTQuant(sc.qRaw, h, m.w.QProjQ, L, D, qCols) // [L, qCols]
		qRaw = sc.qRaw
		sc.k = matNTQuant(sc.k, h, m.w.KProjQ, L, D, kvCols) // [L, KVH*HD]
		k = sc.k
		sc.v = matNTQuant(sc.v, h, m.w.VProjQ, L, D, kvCols) // [L, KVH*HD]
		v = sc.v
	} else {
		qkvFused := false
		if AttnQKVDevice != nil && L*D*qCols >= deviceMinWork {
			if fq, fk, fv, ferr := AttnQKVDevice(h, m.w.QProj, m.w.KProj, m.w.VProj, L, D, qCols, kvCols); ferr == nil {
				qRaw, k, v = fq, fk, fv
				qkvFused = true
			}
		}
		if !qkvFused {
			sc.qRaw = matNTInto(sc.qRaw, h, m.w.QProj, L, D, qCols) // [L, qCols]
			qRaw = sc.qRaw
			sc.k = matNTInto(sc.k, h, m.w.KProj, L, D, kvCols) // [L, KVH*HD]
			k = sc.k
			sc.v = matNTInto(sc.v, h, m.w.VProj, L, D, kvCols) // [L, KVH*HD]
			v = sc.v
		}
	}
	if cfg.QKVClip > 0 {
		for _, values := range [][]float32{qRaw, k, v} {
			for i, value := range values {
				if value > cfg.QKVClip {
					values[i] = cfg.QKVClip
				} else if value < -cfg.QKVClip {
					values[i] = -cfg.QKVClip
				}
			}
		}
	}
	return m.continueFromQKV(qRaw, k, v, L, D, st, sc)
}

// forwardFromQKV resumes attention from ALREADY-COMPUTED q/k/v projections — e.g. supplied by
// ResidualNormMLPProjAttnInputDevice, which folds the PREVIOUS layer's tail + THIS layer's input RMSNorm +
// q/k/v projections into one command buffer — skipping this mixer's own input norm and projection step
// entirely. State threading is identical to forwardNoProj (prior in, next out); the caller is responsible
// for having applied this layer's input RMSNorm before computing qRaw/k/v (the fused device op does this
// on device) and for qRaw/k/v matching m.w.QProj/KProj/VProj's shapes exactly.
func (m *attnMixer) forwardFromQKV(qRaw, k, v []float32, L, D int, prior any) (mixerHidden, projW []float32, mixCols int, next any, err error) {
	var st attnState
	if p, ok := prior.(attnState); ok {
		st = p
	}
	sc := st.sc
	if sc == nil {
		sc = &attnScratch{}
	}
	return m.continueFromQKV(qRaw, k, v, L, D, st, sc)
}

// continueFromQKV is the shared attention math from q/k/v projections onward: de-interleave the
// output-gate (if configured), per-head QK-norm + partial rotary, grow the KV cache, causal softmax
// attention and the output-gate multiply. Both forwardNoProj and forwardFromQKV feed it.
func (m *attnMixer) continueFromQKV(qRaw, k, v []float32, L, D int, st attnState, sc *attnScratch) (mixerHidden, projW []float32, mixCols int, next any, err error) {
	cfg := m.cfg
	H, KVH, HD, RD := cfg.Heads, cfg.KVHeads, cfg.HeadDim, cfg.RotaryDim
	if H <= 0 || KVH <= 0 || HD <= 0 || H%KVH != 0 {
		return nil, nil, 0, nil, core.NewError("composed.attnMixer: bad geometry")
	}
	theta := float64(cfg.RopeTheta)
	if theta == 0 {
		theta = 1e6
	}
	pos0 := st.n
	scale := 1.0 / math.Sqrt(float64(HD))
	rep := H / KVH
	var alibiSlopes []float32
	if cfg.ALiBi {
		alibiSlopes = model.ALiBiSlopes(H)
	}

	// q_proj raw → per-head q [L,H*HD] and (gated) the σ-gate [L,H*HD] ([q_h ; gate_h] within each head's
	// 2·HD block, per the transformers qwen3_5 chunk).
	var q, gate []float32
	if cfg.OutputGate {
		raw := qRaw
		q = make([]float32, L*H*HD)    // de-interleave targets: per-token state (q is written in place by
		gate = make([]float32, L*H*HD) // attention, gate outlives to the σ-gate), not scratch
		for t := range L {
			for hd := range H {
				src := raw[t*2*H*HD+hd*2*HD:]
				copy(q[(t*H+hd)*HD:(t*H+hd)*HD+HD], src[:HD])
				copy(gate[(t*H+hd)*HD:(t*H+hd)*HD+HD], src[HD:2*HD])
			}
		}
	} else {
		q = qRaw
	}

	// QK-norm (per head) + partial rotary at absolute positions pos0+t.
	for t := range L {
		for hd := range H {
			row := q[t*H*HD+hd*HD : t*H*HD+hd*HD+HD]
			if cfg.QKNormalization == model.QKL2Norm {
				l2NormHead(row, cfg.NormEps)
			} else if cfg.QKNormalization == model.QKLayerNorm {
				layerNormHead(row, m.w.QNorm, cfg.NormEps)
			} else {
				rmsNormHead(row, m.w.QNorm, cfg.NormEps)
			}
			applyRotaryHalf(row, pos0+t, RD, theta)
		}
		for hd := range KVH {
			row := k[t*KVH*HD+hd*HD : t*KVH*HD+hd*HD+HD]
			if cfg.QKNormalization == model.QKL2Norm {
				l2NormHead(row, cfg.NormEps)
			} else if cfg.QKNormalization == model.QKLayerNorm {
				layerNormHead(row, m.w.KNorm, cfg.NormEps)
			} else {
				rmsNormHead(row, m.w.KNorm, cfg.NormEps)
			}
			applyRotaryHalf(row, pos0+t, RD, theta)
		}
	}

	// grow the cache: [pos0+L, KVH*HD]. ck and cv are the returned state — one backing slab, two
	// non-overlapping capped windows. Both are copied out (read-only) at the head of the next call
	// before any write, so sharing one array between the K and V caches is bit-identical and saves
	// one alloc per token on the decode path.
	N := pos0 + L
	cacheN := N * KVH * HD
	ckcv := make([]float32, 2*cacheN)
	ck := ckcv[0:cacheN:cacheN]
	cv := ckcv[cacheN : 2*cacheN : 2*cacheN]
	copy(ck, st.k)
	copy(cv, st.v)
	copy(ck[pos0*KVH*HD:], k)
	copy(cv[pos0*KVH*HD:], v)

	// causal attention: query t (position pos0+t) attends to cached keys 0..pos0+t. The query buffer q
	// is [L,H,HD] = the output shape and is dead after this loop; each head's qrow is fully consumed by
	// its score dot-products before that head's orow is written (orow is at the same offset), so the
	// attention output is written in place over q — bit-identical, one fewer alloc per token.
	out := q
	scores := make([]float64, N)
	for t := range L {
		last := pos0 + t // inclusive
		first := 0
		if cfg.SlidingWindow > 0 && last+1 > cfg.SlidingWindow {
			first = last + 1 - cfg.SlidingWindow
		}
		for hd := range H {
			kvh := hd / rep
			qrow := q[t*H*HD+hd*HD:]
			// scores over keys 0..last
			maxS := math.Inf(-1)
			for j := first; j <= last; j++ {
				krow := ck[j*KVH*HD+kvh*HD:]
				var dot float64
				for d := range HD {
					dot += float64(qrow[d]) * float64(krow[d])
				}
				dot *= scale
				scores[j] = dot
				if dot > maxS {
					maxS = dot
				}
			}
			if cfg.ALiBi {
				model.ApplyALiBi(scores[:last+1], alibiSlopes[hd], last, 0)
				maxS = math.Inf(-1)
				for j := first; j <= last; j++ {
					if scores[j] > maxS {
						maxS = scores[j]
					}
				}
			}
			// softmax
			var sum float64
			for j := first; j <= last; j++ {
				scores[j] = math.Exp(scores[j] - maxS)
				sum += scores[j]
			}
			// weighted sum of values
			orow := out[t*H*HD+hd*HD:]
			for d := range HD {
				var acc float64
				for j := first; j <= last; j++ {
					acc += scores[j] * float64(cv[j*KVH*HD+kvh*HD+d])
				}
				orow[d] = float32(acc / sum)
			}
		}
	}
	// attn_output_gate: gate the attention output (per head, per dim) by σ(gate) before o_proj. The gate
	// is the second half of each head's q_proj block — never QK-normed or rotated. The transformers
	// qwen3_5 reference hardcodes sigmoid here (output_gate_type is not consumed by the reference forward).
	if gate != nil {
		for i := range out {
			s := 1.0 / (1.0 + math.Exp(-float64(gate[i])))
			out[i] = float32(float64(out[i]) * s)
		}
	}
	return out, m.w.OProj, H * HD, attnState{k: ck, v: cv, n: N, sc: sc}, nil
}
