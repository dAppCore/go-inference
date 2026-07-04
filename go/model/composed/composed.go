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

// matNT computes out[M,N] = in[M,K] @ w[N,K]ᵀ (the Linear y = x·Wᵀ), f32 host.
func matNT(in, w []float32, M, K, N int) []float32 {
	out := make([]float32, M*N)
	for m := 0; m < M; m++ {
		for n := 0; n < N; n++ {
			var acc float64
			for k := 0; k < K; k++ {
				acc += float64(in[m*K+k]) * float64(w[n*K+k])
			}
			out[m*N+n] = float32(acc)
		}
	}
	return out
}

// rmsNormRowsPlain RMS-norms each of the `rows` rows of x [rows,d] by the shared plain weight w [d].
func rmsNormRowsPlain(x, w []float32, rows, d int, eps float32) []float32 {
	out := make([]float32, rows*d)
	for r := 0; r < rows; r++ {
		xr := x[r*d : (r+1)*d]
		var ss float64
		for i := 0; i < d; i++ {
			ss += float64(xr[i]) * float64(xr[i])
		}
		rms := math.Sqrt(ss/float64(d) + float64(eps))
		for i := 0; i < d; i++ {
			out[r*d+i] = float32(float64(xr[i]) / rms * float64(w[i]))
		}
	}
	return out
}

// swiglu runs the SwiGLU MLP over x [L,D] → [L,D].
func (mlp *MLP) forward(x []float32, L, D int) []float32 {
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

// forwardEmb runs L input embeddings [L,D] through the stack, advancing each layer's mixer state, and
// returns the output hiddens [L,D]. Serves both prefill (L>1) and decode (L=1).
func (s *ComposedSession) forwardEmb(h []float32, L int) ([]float32, error) {
	D, eps := s.m.D, s.m.Eps
	for li := range s.m.Layers {
		layer := &s.m.Layers[li]
		normed := rmsNormRowsPlain(h, layer.InputNorm, L, D, eps)
		mixOut, next, err := layer.Mixer.Forward(normed, L, D, s.states[li])
		if err != nil {
			return nil, err
		}
		s.states[li] = next
		for i := range h {
			h[i] += mixOut[i] // mixer residual
		}
		normed2 := rmsNormRowsPlain(h, layer.PostAttnNorm, L, D, eps)
		mlpOut := layer.MLP.forward(normed2, L, D)
		for i := range h {
			h[i] += mlpOut[i] // MLP residual
		}
	}
	return h, nil
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
