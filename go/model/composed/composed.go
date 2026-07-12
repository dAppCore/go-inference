// SPDX-Licence-Identifier: EUPL-1.2

// Package composed is the native (no-cgo) config-composed hybrid transformer — a pre-norm SwiGLU stack
// whose per-layer attention slot is a config-dispatched sequence Mixer (gated-delta for the
// linear_attention layers, full attention for the rest). It is the native port of metal's
// composed.ComposedModel, the orchestration that turns the FLA mixer math into a servable model: the
// Qwen 3.6 hybrid (gemma4's peer for local inference) runs here. A ComposedSession threads each layer's
// own state — recurrent (conv + delta) for a gated-delta mixer, a KV cache for an attention mixer — so a
// streaming decode reproduces a one-pass prefill exactly. Pure Go host f32; the mixers' projections use
// their own device-GEMM seams.
package composed

import (
	"math"
	"runtime"

	core "dappco.re/go"
)

// Mixer is one layer's sequence mixer (the attention slot). A mixer owns its WEIGHTS (shared across
// sessions); its STATE is threaded by the session, passed in and returned, so one model serves many
// concurrent sessions. prior is nil for a fresh sequence.
type Mixer interface {
	// Forward mixes hidden [L,D] (L tokens) and returns out [L,D] plus the advanced state. The state is
	// opaque to the session (gated-delta carries conv+delta; attention carries a KV cache).
	Forward(hidden []float32, L, D int, prior any) (out []float32, next any, err error)
	// Kind reports the mixer family ("gated_deltanet", "full_attention") for diagnostics + cache typing.
	Kind() string
}

// FFN is a layer's feed-forward slot: a dense SwiGLU MLP or a Mixture-of-Experts (qwen3_6_moe). Both map
// hidden [L,D] → [L,D].
type FFN interface {
	forward(x []float32, L, D int) []float32
}

// MLP is a per-layer SwiGLU feed-forward: out = (SiLU(x·Gateᵀ) ⊙ x·Upᵀ)·Downᵀ. Gate/Up are [FF,D],
// Down is [D,FF].
type MLP struct {
	Gate, Up, Down []float32
	FF             int
}

// Layer is one pre-norm block: InputNorm → Mixer → residual, PostAttnNorm → MLP → residual.
type Layer struct {
	InputNorm    []float32 // [D] plain RMSNorm (qwen is not gemma)
	Mixer        Mixer
	PostAttnNorm []float32 // [D]
	MLP          FFN       // dense SwiGLU or MoE
}

// ComposedModel is the loaded hybrid stack: token embedding, the per-layer blocks, the final norm and the
// LM head (tied to Embed when Output is nil). All f32 (the loader widens the bf16 checkpoint).
type ComposedModel struct {
	Embed  []float32 // [Vocab, D]
	Layers []Layer
	NormF  []float32 // [D] final RMSNorm
	Output []float32 // [Vocab, D] (nil ⇒ tied to Embed)
	D      int
	Vocab  int
	Eps    float32
}

func silu(v float64) float64 { return v / (1 + math.Exp(-v)) }

// ProjMatMulInto is the device-GEMM seam for the composed stack's OWN projections — the attention
// mixer's q/k/v/o, the MLP/MoE matmuls, and the LM head (the largest single matmul of every decode
// step). Same AX-8 shape as qwen3/mamba2/rwkv7's hooks: the lib declares it and runs the host matNT
// by default; importing the native backend binds it to the steel GEMM. The device kernel accumulates
// in f32 (host is f64), so binding trades at the same numeric tier the gated-delta projections
// already serve at — the composed -state contract (a token-prefix snapshot, recomputed on wake) is
// deterministic per build either way.
var ProjMatMulInto func(out, x, w []float32, M, K, N int) ([]float32, error)

// MLPDevice is the fused-SwiGLU device hook: gate/up GEMMs, the silu-and-multiply glue, and the
// down GEMM encoded into ONE command buffer with device-resident intermediates — one round-trip
// where the per-projection hook pays three. Same AX-8 shape as ProjMatMulInto; nil runs the
// per-projection path.
var MLPDevice func(gate, up, down, x []float32, L, D, FF int) ([]float32, error)

// ResidualNormMLPDevice is the fused pre-norm FFN-tail hook: the mixer-output residual add, the post-attn
// RMSNorm, the SwiGLU MLP and the MLP residual add encoded into ONE command buffer with device-resident
// intermediates — where the host path pays three passes over [L,D] (add, norm, add) bracketing the MLP's
// own command buffer. Given the pre-mixer hidden h and the mixer output mixOut, it returns the new hidden
// h = (h+mixOut) + SwiGLU(RMSNorm(h+mixOut, normW)). Same AX-8 shape as MLPDevice; nil — or a MoE FFN, or
// a sub-floor shape — runs the host add/norm/MLP/add path. Arch-neutral: every pre-norm SwiGLU stack has
// exactly this tail, so the backend primitive is named for the op, not for this stack.
var ResidualNormMLPDevice func(h, mixOut, normW, gate, up, down []float32, L, D, FF int, eps float32) ([]float32, error)

// ResidualNormMLPProjDevice is ResidualNormMLPDevice with the mixer's FINAL projection (attention o_proj /
// gated-delta out_proj) folded onto the front: given the mixer's pre-projection hidden mixerHidden [L,mixCols]
// and that projection's weight projW [D,mixCols], it computes mixOut = mixerHidden@projWᵀ, then the same
// (h+mixOut) → RMSNorm → SwiGLU → (+mlpOut) tail, all in ONE command buffer. The projection's output never
// crosses the host floor — where the unfused path runs it as its own command buffer (a commit+wait per
// layer) before the tail's. The session uses this only for a mixer that implements projMixer AND a dense MLP
// above the device floor; nil — or a MoE FFN, or an error — falls back to the mixer's own projection + the
// standard tail (see forwardEmb). Same AX-8 shape as ResidualNormMLPDevice; arch-neutral (named for the op,
// not the mixer).
var ResidualNormMLPProjDevice func(mixerHidden, projW, h, normW, gate, up, down []float32, L, D, mixCols, FF int, eps float32) ([]float32, error)

// projMixer is the OPTIONAL capability of a mixer whose output is produced by a single final GEMM (the
// attention o_proj, the gated-delta out_proj): it runs the mixer up to but NOT including that projection and
// hands back the pre-projection hidden [L,mixCols] plus the projection weight projW [D,mixCols], so the
// session can fold the projection into the FFN-tail command buffer (one CB where the mixer-owned projection
// would otherwise be a second). forwardNoProj advances the mixer state exactly as Forward does — the ONLY
// difference is that the caller now owns the final projection. A mixer without a single-GEMM output does not
// implement this and the session runs the standard Forward → tail path.
type projMixer interface {
	forwardNoProj(hidden []float32, L, D int, prior any) (mixerHidden, projW []float32, mixCols int, next any, err error)
}

// deviceMinWork is the M·K·N floor below which matNTInto ignores the device hook — a tiny GEMV's
// command-buffer round-trip outweighs its compute, so sub-MMAC shapes stay on the host path.
const deviceMinWork = 1 << 20

// matNT computes out[M,N] = in[M,K] @ w[N,K]ᵀ (the Linear y = x·Wᵀ), f32 host.
func matNT(in, w []float32, M, K, N int) []float32 {
	return matNTInto(nil, in, w, M, K, N)
}

// matNTInto is matNT writing into out, reusing it when cap(out) ≥ M·N (else it allocates a fresh M·N
// slab). Identical f64 accumulation + write order to the fresh-buffer form — only WHERE the result lands
// changes, so the output is bit-identical. The attention mixer threads a caller-owned scratch through
// this for its per-token q/k/v/o projections.
//
// Large shapes shard the OUTPUT COLUMNS across CPU cores: every out[m·N+n] keeps exactly the serial
// per-element k-accumulation order (only which goroutine computes it changes), so the sharded form is
// bit-identical too — the composed -state byte-identity contract holds. Small shapes stay serial: the
// goroutine fan-out costs more than it saves below the work floor (tests' toy shapes, tiny projections).
func matNTInto(out, in, w []float32, M, K, N int) []float32 {
	if ProjMatMulInto != nil && M*K*N >= deviceMinWork {
		if res, err := ProjMatMulInto(out, in, w, M, K, N); err == nil {
			return res
		}
		// A device failure (missing kernel, no Metal device) falls through to the
		// host path — deterministic for the rest of the process either way.
	}
	if cap(out) < M*N {
		out = make([]float32, M*N)
	} else {
		out = out[:M*N]
	}
	if M*K*N < matNTParMinWork {
		matNTCols(out, in, w, M, K, N, 0, N)
		return out
	}
	workers := runtime.GOMAXPROCS(0)
	if workers > N {
		workers = N
	}
	span := (N + workers - 1) / workers
	var wg core.WaitGroup
	for lo := 0; lo < N; lo += span {
		hi := lo + span
		if hi > N {
			hi = N
		}
		wg.Add(1)
		go func(lo, hi int) {
			defer wg.Done()
			matNTCols(out, in, w, M, K, N, lo, hi)
		}(lo, hi)
	}
	wg.Wait()
	return out
}

// matNTParMinWork is the M·K·N floor below which matNTInto stays serial — under ~1 MMAC the
// fan-out/join overhead exceeds the compute it spreads.
const matNTParMinWork = 1 << 20

// matNTCols is the serial kernel over output columns [n0,n1) — the one accumulation-order-defining
// loop both the serial and sharded paths run.
func matNTCols(out, in, w []float32, M, K, N, n0, n1 int) {
	for m := range M {
		for n := n0; n < n1; n++ {
			var acc float64
			for k := range K {
				acc += float64(in[m*K+k]) * float64(w[n*K+k])
			}
			out[m*N+n] = float32(acc)
		}
	}
}

// rmsNormRowsPlain RMS-norms each of the `rows` rows of x [rows,d] by the shared plain weight w [d].
func rmsNormRowsPlain(x, w []float32, rows, d int, eps float32) []float32 {
	out := make([]float32, rows*d)
	for r := range rows {
		xr := x[r*d : (r+1)*d]
		var ss float64
		for i := range d {
			ss += float64(xr[i]) * float64(xr[i])
		}
		rms := math.Sqrt(ss/float64(d) + float64(eps))
		for i := range d {
			out[r*d+i] = float32(float64(xr[i]) / rms * float64(w[i]))
		}
	}
	return out
}

// swiglu runs the SwiGLU MLP over x [L,D] → [L,D].
func (mlp *MLP) forward(x []float32, L, D int) []float32 {
	if MLPDevice != nil && L*D*mlp.FF >= deviceMinWork {
		if out, err := MLPDevice(mlp.Gate, mlp.Up, mlp.Down, x, L, D, mlp.FF); err == nil {
			return out
		}
	}
	g := matNT(x, mlp.Gate, L, D, mlp.FF) // [L,FF]
	u := matNT(x, mlp.Up, L, D, mlp.FF)   // [L,FF]
	h := make([]float32, L*mlp.FF)
	for i := range h {
		h[i] = float32(silu(float64(g[i])) * float64(u[i]))
	}
	return matNT(h, mlp.Down, L, mlp.FF, D) // [L,D]
}

// ComposedSession is a recurrent decode session over a ComposedModel: per-layer mixer state, threaded
// across forward calls. Single-goroutine.
type ComposedSession struct {
	m      *ComposedModel
	states []any // per-layer opaque mixer state; nil ⇒ fresh
}

// NewSession builds a fresh session (each layer's mixer state starts empty).
func NewSession(m *ComposedModel) *ComposedSession {
	return &ComposedSession{m: m, states: make([]any, len(m.Layers))}
}

// pendingAttnQKV carries a NEXT layer's ALREADY-COMPUTED input RMSNorm + q/k/v projections, folded into
// the PREVIOUS layer's proj-fused tail command buffer by ResidualNormMLPProjAttnInputDevice — the
// input-side mirror of the o_proj fuse. The layer at that index resumes via attnMixer.forwardFromQKV
// instead of recomputing its input norm + projections.
type pendingAttnQKV struct{ q, k, v []float32 }

// pendingGatedDeltaInput is pendingAttnQKV's gated-delta counterpart — set by
// ResidualNormMLPProjGatedDeltaInputDevice; the layer resumes via gatedDeltaMixer.forwardFromInput.
type pendingGatedDeltaInput struct{ qkv, z, a, b []float32 }

// forwardEmb runs L input embeddings [L,D] through the stack, advancing each layer's mixer state, and
// returns the output hiddens [L,D]. Serves both prefill (L>1) and decode (L=1).
func (s *ComposedSession) forwardEmb(h []float32, L int) ([]float32, error) {
	D, eps := s.m.D, s.m.Eps

	// pendingAttn / pendingGD carry a NEXT layer's already-computed input RMSNorm + input projections —
	// folded into the PREVIOUS layer's proj-fused tail command buffer, the symmetric collapse to the
	// o_proj fuse below. At most one is ever set (a layer has exactly one mixer kind); both nil ⇒ this
	// layer computes its input norm + projections fresh (always true for layer 0 — no predecessor tail).
	var pendingAttn *pendingAttnQKV
	var pendingGD *pendingGatedDeltaInput

	for li := range s.m.Layers {
		layer := &s.m.Layers[li]

		// Resolve this layer's (mixerHidden, projW, mixCols, next-state): either resume from a pending
		// input-fuse (skipping this layer's own RMSNorm + projection step — both mixer kinds implement
		// projMixer, so a pending resume always lands here) or run the mixer's forwardNoProj off a
		// freshly-computed input RMSNorm.
		var mixerHidden, projW []float32
		var mixCols int
		var next any
		var err error
		isProj := false

		switch {
		case pendingAttn != nil:
			p := pendingAttn
			pendingAttn = nil
			am := layer.Mixer.(*attnMixer) // guaranteed: only ever set for an *attnMixer next layer
			mixerHidden, projW, mixCols, next, err = am.forwardFromQKV(p.q, p.k, p.v, L, D, s.states[li])
			isProj = true
		case pendingGD != nil:
			p := pendingGD
			pendingGD = nil
			gm := layer.Mixer.(*gatedDeltaMixer) // guaranteed: only ever set for a *gatedDeltaMixer next layer
			mixerHidden, projW, mixCols, next, err = gm.forwardFromInput(p.qkv, p.z, p.a, p.b, L, D, s.states[li])
			isProj = true
		default:
			if pm, ok := layer.Mixer.(projMixer); ok {
				normed := rmsNormRowsPlain(h, layer.InputNorm, L, D, eps)
				mixerHidden, projW, mixCols, next, err = pm.forwardNoProj(normed, L, D, s.states[li])
				isProj = true
			}
		}
		if err != nil {
			return nil, err
		}

		if isProj {
			s.states[li] = next
			if mlp, isDense := layer.MLP.(*MLP); isDense && ResidualNormMLPProjDevice != nil && L*D*mlp.FF >= deviceMinWork {
				// Input-fuse: when the NEXT layer exists and is itself a full-attention or gated-delta
				// mixer, fold its input RMSNorm + input projections onto the BACK of this layer's
				// proj+tail command buffer — the symmetric collapse to the o_proj fuse. y never crosses
				// the host floor before feeding the next layer's projections; only y and the projections
				// do. The projections ride free whatever their own size (the tail CB is already paid
				// for); a nil hook, an unmatched next-mixer kind, the last layer, or a device error all
				// fall through to the plain proj-fused tail below.
				if li+1 < len(s.m.Layers) {
					nextLayer := &s.m.Layers[li+1]
					switch nm := nextLayer.Mixer.(type) {
					case *attnMixer:
						if ResidualNormMLPProjAttnInputDevice != nil {
							qCols := nm.cfg.Heads * nm.cfg.HeadDim
							if nm.cfg.OutputGate {
								qCols *= 2
							}
							kvCols := nm.cfg.KVHeads * nm.cfg.HeadDim
							if y, q, k, v, ferr := ResidualNormMLPProjAttnInputDevice(
								mixerHidden, projW, h, layer.PostAttnNorm, mlp.Gate, mlp.Up, mlp.Down, L, D, mixCols, mlp.FF, eps,
								nextLayer.InputNorm, nm.w.QProj, nm.w.KProj, nm.w.VProj, qCols, kvCols,
							); ferr == nil {
								h = y
								pendingAttn = &pendingAttnQKV{q: q, k: k, v: v}
								continue
							}
						}
					case *gatedDeltaMixer:
						if ResidualNormMLPProjGatedDeltaInputDevice != nil {
							convDim, vDim, VH := nm.cfg.ConvDim(), nm.cfg.VDim(), nm.cfg.ValueHeads
							if y, qkv, z, a, b, ferr := ResidualNormMLPProjGatedDeltaInputDevice(
								mixerHidden, projW, h, layer.PostAttnNorm, mlp.Gate, mlp.Up, mlp.Down, L, D, mixCols, mlp.FF, eps,
								nextLayer.InputNorm, nm.w.InProjQKV, nm.w.InProjZ, nm.w.InProjA, nm.w.InProjB, convDim, vDim, VH,
							); ferr == nil {
								h = y
								pendingGD = &pendingGatedDeltaInput{qkv: qkv, z: z, a: a, b: b}
								continue
							}
						}
					}
				}
				// Plain proj-fused tail (no next-input fuse applies): o_proj/out_proj → residual →
				// post-attn RMSNorm → SwiGLU → residual, in one command buffer.
				if y, ferr := ResidualNormMLPProjDevice(mixerHidden, projW, h, layer.PostAttnNorm, mlp.Gate, mlp.Up, mlp.Down, L, D, mixCols, mlp.FF, eps); ferr == nil {
					h = y
					continue
				}
			}
			// Fallback — covers a failed proj-fused tail attempt above AND a projMixer layer whose MLP
			// isn't dense/above-floor (including one reached via a pending resume): complete the mixer's
			// own projection on the host, then the plain device tail-fuse, else host tail.
			mixOut := matNT(mixerHidden, projW, L, mixCols, D)
			if mlp, ok := layer.MLP.(*MLP); ok && ResidualNormMLPDevice != nil && L*D*mlp.FF >= deviceMinWork {
				if y, err := ResidualNormMLPDevice(h, mixOut, layer.PostAttnNorm, mlp.Gate, mlp.Up, mlp.Down, L, D, mlp.FF, eps); err == nil {
					h = y
					continue
				}
			}
			h = tailHost(h, mixOut, layer.PostAttnNorm, layer.MLP, L, D, eps)
			continue
		}

		// Standard path: the mixer doesn't implement projMixer at all — it owns its final projection
		// outright; the FFN tail fuses on its own device CB (h += mixOut → RMSNorm → SwiGLU → h += mlpOut)
		// for a dense MLP above the floor, else host.
		normed := rmsNormRowsPlain(h, layer.InputNorm, L, D, eps)
		mixOut, next, err := layer.Mixer.Forward(normed, L, D, s.states[li])
		if err != nil {
			return nil, err
		}
		s.states[li] = next
		if mlp, ok := layer.MLP.(*MLP); ok && ResidualNormMLPDevice != nil && L*D*mlp.FF >= deviceMinWork {
			if y, err := ResidualNormMLPDevice(h, mixOut, layer.PostAttnNorm, mlp.Gate, mlp.Up, mlp.Down, L, D, mlp.FF, eps); err == nil {
				h = y
				continue
			}
		}
		h = tailHost(h, mixOut, layer.PostAttnNorm, layer.MLP, L, D, eps)
	}
	return h, nil
}

// tailHost runs the FFN tail on the host — the mixer-output residual add, the post-attn RMSNorm, the FFN
// (dense SwiGLU or MoE) and the MLP residual add — returning the new hidden. It is the shared fallback for
// both the projection-fused and standard paths when the device tail hook is nil or errors; a MoE FFN always
// lands here (no fused-MLP device kernel).
func tailHost(h, mixOut, normW []float32, ffn FFN, L, D int, eps float32) []float32 {
	for i := range h {
		h[i] += mixOut[i] // mixer residual
	}
	normed := rmsNormRowsPlain(h, normW, L, D, eps)
	mlpOut := ffn.forward(normed, L, D)
	for i := range h {
		h[i] += mlpOut[i] // MLP residual
	}
	return h
}

// forward embeds tokens then runs the stack.
func (s *ComposedSession) forward(tokens []int32) ([]float32, error) {
	L, D := len(tokens), s.m.D
	h := make([]float32, L*D)
	for t, tok := range tokens {
		if int(tok) < 0 || int(tok) >= s.m.Vocab {
			return nil, core.NewError("composed.forward: token out of range")
		}
		copy(h[t*D:(t+1)*D], s.m.Embed[int(tok)*D:int(tok)*D+D])
	}
	return s.forwardEmb(h, L)
}

// Forward prefills tokens and returns the per-position hiddens [L,D] (state advanced).
func (s *ComposedSession) Forward(tokens []int32) ([]float32, error) { return s.forward(tokens) }

// headLogits maps a single hidden [D] to vocab logits via the final norm + LM head.
func (s *ComposedSession) headLogits(hidden []float32) []float32 {
	normed := rmsNormRowsPlain(hidden, s.m.NormF, 1, s.m.D, s.m.Eps)
	head := s.m.Output
	if head == nil {
		head = s.m.Embed
	}
	return matNT(normed, head, 1, s.m.D, s.m.Vocab)
}

// Generate greedily decodes up to maxNew tokens after prefilling prompt, threading every layer's mixer
// state. eosID < 0 disables early stop.
func (s *ComposedSession) Generate(prompt []int32, maxNew, eosID int) ([]int32, error) {
	if len(prompt) == 0 || maxNew <= 0 {
		return nil, core.NewError("composed.Generate: empty prompt or maxNew<=0")
	}
	h, err := s.forward(prompt)
	if err != nil {
		return nil, err
	}
	D := s.m.D
	last := h[(len(prompt)-1)*D:]
	gen := make([]int32, 0, maxNew)
	for len(gen) < maxNew {
		next := argmaxF32(s.headLogits(last))
		gen = append(gen, next)
		if eosID >= 0 && int(next) == eosID {
			break
		}
		h1, err := s.forward([]int32{next})
		if err != nil {
			return nil, err
		}
		last = h1
	}
	return gen, nil
}

func argmaxF32(v []float32) int32 {
	best, bi := v[0], int32(0)
	for i := 1; i < len(v); i++ {
		if v[i] > best {
			best, bi = v[i], int32(i)
		}
	}
	return bi
}
