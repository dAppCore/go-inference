// SPDX-Licence-Identifier: EUPL-1.2

package mamba2

import (
	"math"

	core "dappco.re/go"
)

// model.go is the Mamba-2 model + recurrent decode session: the full stack of Mamba-2 blocks with the
// per-layer pre-norm + residual, the final norm and the LM head, and a session that threads the per-layer
// recurrent state (conv-state ring + SSM state) across calls. Unlike the transformer ArchSession (a
// growing K/V cache), a Mamba session keeps a FIXED-size recurrent state per layer — so a streaming
// decode is O(1)/token and reproduces a one-pass prefill exactly (the block carry invariant, lifted to
// the whole model). Pure Go host f32, engine-neutral and testable without a checkpoint.

// MambaLayer is one decoder layer: a plain pre-mixer RMSNorm weight + the mixer's block weights. The
// layer computes x = x + block(RMSNorm(x, Norm)).
type MambaLayer struct {
	Norm []float32 // [D] pre-mixer RMSNorm (plain — mamba is not gemma)
	W    *BlockWeights
}

// MambaModel is a loaded Mamba-2 model: the token embedding, the per-layer stack, the final norm and the
// LM head (tied to Embed when LMHead is nil). All f32 (the loader widens the bf16 checkpoint).
type MambaModel struct {
	Embed  []float32 // [Vocab, D]
	NormF  []float32 // [D] final RMSNorm
	LMHead []float32 // [Vocab, D] (nil ⇒ tied to Embed)
	Layers []MambaLayer
	Cfg    BlockConfig
	D      int
	Vocab  int
}

// rmsNormRowsPlain RMS-norms each of the `rows` rows of x [rows, d] by the shared plain weight w [d].
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
			v := float64(xr[i]) / rms
			if w != nil {
				v *= float64(w[i])
			}
			out[r*d+i] = float32(v)
		}
	}
	return out
}

// MambaSession is a persistent recurrent decode session over a MambaModel: per-layer conv-state ring +
// SSM state, threaded across forward calls. Single-goroutine (the per-layer state is mutable).
type MambaSession struct {
	m         *MambaModel
	convState [][]float32   // per-layer [(K-1)*convDim]; nil ⇒ fresh
	ssmState  [][]float32   // per-layer [H*P*N];          nil ⇒ fresh
	scratch   *BlockScratch // reused in/out projection buffers, shared across layers (they run sequentially
	// and each layer's out is folded into the residual before the next runs) — per-session, so it lives here
	// beside the recurrent state and never on the shared block weights.
}

// NewSession builds a fresh recurrent session (zero state).
func NewSession(m *MambaModel) *MambaSession {
	return &MambaSession{
		m:         m,
		convState: make([][]float32, len(m.Layers)),
		ssmState:  make([][]float32, len(m.Layers)),
		scratch:   &BlockScratch{},
	}
}

// forwardEmb runs L input embeddings [L, D] through the whole stack (per-layer pre-RMSNorm → block →
// residual), advancing the per-layer recurrent state, and returns the output hiddens [L, D] (in place).
// A single call serves both prefill (L>1) and decode (L=1) — the recurrent state makes them produce
// identical hiddens for the same input sequence. This is the embedding-in/hidden-out core the serve
// bookends (Embed/Head) wrap.
func (s *MambaSession) forwardEmb(hidden []float32, L int) ([]float32, error) {
	D := s.m.D
	if len(hidden) != L*D {
		return nil, core.NewError("mamba2.forwardEmb: hidden must be [L,D]")
	}
	for li := range s.m.Layers {
		layer := s.m.Layers[li]
		normed := rmsNormRowsPlain(hidden, layer.Norm, L, D, s.m.Cfg.Eps)
		out, nc, ns, err := BlockForwardScratchF32(normed, layer.W, s.m.Cfg, s.convState[li], s.ssmState[li], L, D, s.scratch)
		if err != nil {
			return nil, err
		}
		s.convState[li], s.ssmState[li] = nc, ns
		for i := range hidden {
			hidden[i] += out[i] // residual
		}
	}
	return hidden, nil
}

// forward embeds `tokens` and runs them through the stack — the token-in/hidden-out path.
func (s *MambaSession) forward(tokens []int32) ([]float32, error) {
	L, D := len(tokens), s.m.D
	hidden := make([]float32, L*D)
	for t, tok := range tokens {
		if int(tok) < 0 || int(tok) >= s.m.Vocab {
			return nil, core.NewError("mamba2.forward: token out of range")
		}
		copy(hidden[t*D:(t+1)*D], s.m.Embed[int(tok)*D:int(tok)*D+D])
	}
	return s.forwardEmb(hidden, L)
}

// headLogits maps a single hidden [D] to vocab logits via the final norm + LM head.
func (s *MambaSession) headLogits(hidden []float32) []float32 {
	normed := rmsNormRowsPlain(hidden, s.m.NormF, 1, s.m.D, s.m.Cfg.Eps)
	head := s.m.LMHead
	if head == nil {
		head = s.m.Embed // tied
	}
	return matNT(normed, head, 1, s.m.D, s.m.Vocab)
}

// Forward prefills `tokens` and returns the per-position hiddens [L,D] (state advanced) — the building
// block for Generate and the prefill-vs-decode equivalence test.
func (s *MambaSession) Forward(tokens []int32) ([]float32, error) { return s.forward(tokens) }

// Generate greedily decodes up to maxNew tokens after prefilling prompt, threading the recurrent state
// (so each new token is O(1)). eosID < 0 disables early stop. Token-identical to a one-pass run.
func (s *MambaSession) Generate(prompt []int32, maxNew, eosID int) ([]int32, error) {
	if len(prompt) == 0 {
		return nil, core.NewError("mamba2.Generate: empty prompt")
	}
	if maxNew <= 0 {
		return nil, core.NewError("mamba2.Generate: maxNew must be > 0")
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
