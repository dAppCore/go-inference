// SPDX-Licence-Identifier: EUPL-1.2

package rwkv7

import core "dappco.re/go"

// model.go is the RWKV-7 model + recurrent decode session: the full stack of RWKV-7 blocks (each a
// token-shift + WKV7 time-mix, then a token-shift + channel-mix, both pre-normed and residual — see
// timemix.go/channelmix.go), the final norm and LM head, and a session that threads EVERY layer's carried
// state across calls: one [H,K,V] WKV7 matrix plus TWO independent one-token shift registers (time-mix's
// and channel-mix's observe different input streams, so they carry separately). Unlike the transformer
// ArchSession's growing K/V cache, this is a FIXED-size recurrent state per layer, on the same scaffold as
// mamba2/model.go — a streaming decode is O(1)/token and reproduces a one-pass prefill exactly (the
// tokenShift/WKV7F32 carry invariants, lifted to the whole model). Pure Go host f32, engine-neutral.
//
// Per-layer forward (fla.models.rwkv7.modeling_rwkv7.RWKV7Block.forward — config.norm_first=true, every
// released RWKV-7 checkpoint):
//
//	base0 = pre_norm(x)   if layer 0   else x        // ln0 — applied ONCE, becomes the running residual
//	h1    = attn_norm(base0)                          // ln1
//	x1    = base0 + timeMix(h1)
//	h2    = ffn_norm(x1)                              // ln2
//	x2    = x1 + channelMix(h2)
//
// x2 is the running residual the NEXT layer (or, after the last layer, the final norm) receives.

// RWKV7Layer is one decoder layer's real weights: PreNormW/B are non-nil ONLY for layer 0 (norm_first's
// embedding-layernorm branch); every other field is present on every layer.
type RWKV7Layer struct {
	PreNormW, PreNormB   []float32 // [D]; nil except layer 0
	AttnNormW, AttnNormB []float32 // [D]
	Attn                 *timeMixWeights
	FfnNormW, FfnNormB   []float32 // [D]
	FFN                  *channelMixWeights
}

// RWKV7Model is a loaded RWKV-7 model: the token embedding, the per-layer stack, the final LayerNorm, and
// the LM head (tied to Embed when LMHead is nil — the checkpoint this port targets does NOT tie them, but
// the fallback is cheap and matches mamba2's same convention). All f32 (the loader widens the checkpoint's
// bf16 weights). Cfg is the per-layer WKV7 geometry (asserted uniform across layers by the loader); FF is
// the channel-mix intermediate size.
type RWKV7Model struct {
	Embed        []float32 // [Vocab,D]
	NormW        []float32 // [D] final LayerNorm weight
	NormB        []float32 // [D] final LayerNorm bias (nil ⇒ norm_bias=false)
	LMHead       []float32 // [Vocab,D] (nil ⇒ tied to Embed)
	Layers       []RWKV7Layer
	Cfg          BlockConfig
	D, Vocab, FF int
	Eps          float32
}

// RWKV7Session is a persistent recurrent decode session over an RWKV7Model: per-layer WKV7 [H,K,V] state
// plus the two independent token-shift registers, threaded across forward calls. Single-goroutine (the
// per-layer state is mutable) — the same ownership rule block.go/model/composed document: never share a
// session across concurrently-stepped conversations.
type RWKV7Session struct {
	m      *RWKV7Model
	wkv    [][]float32 // per-layer [H,K,V]; nil entry ⇒ fresh
	shift1 [][]float32 // per-layer [D]; time-mix's token-shift register
	shift2 [][]float32 // per-layer [D]; channel-mix's token-shift register
}

// NewSession builds a fresh recurrent session (zero state).
func NewSession(m *RWKV7Model) *RWKV7Session {
	return &RWKV7Session{
		m:      m,
		wkv:    make([][]float32, len(m.Layers)),
		shift1: make([][]float32, len(m.Layers)),
		shift2: make([][]float32, len(m.Layers)),
	}
}

// forwardEmb runs L input embeddings [L,D] through the whole stack, advancing every layer's carried
// state, and returns the output hiddens [L,D] — a single call serves both prefill (L>1) and decode (L=1)
// identically, the recurrent-carry invariant the whole model inherits from tokenShift/WKV7F32.
func (s *RWKV7Session) forwardEmb(x []float32, L int) ([]float32, error) {
	D := s.m.D
	if len(x) != L*D {
		return nil, core.NewError("rwkv7.forwardEmb: hidden must be [L,D]")
	}
	cur := x
	var vFirst []float32
	for li := range s.m.Layers {
		layer := &s.m.Layers[li]

		base0 := cur
		if layer.PreNormW != nil {
			base0 = layerNormRows(cur, layer.PreNormW, layer.PreNormB, L, D, s.m.Eps)
		}

		h1 := layerNormRows(base0, layer.AttnNormW, layer.AttnNormB, L, D, s.m.Eps)
		attnOut, vf, newSt, err := timeMixForward(h1, layer.Attn, s.m.Cfg, li, vFirst, timeMixState{WKV: s.wkv[li], Shift: s.shift1[li]}, L, D, s.m.Eps)
		if err != nil {
			return nil, err
		}
		vFirst = vf
		s.wkv[li], s.shift1[li] = newSt.WKV, newSt.Shift

		x1 := make([]float32, L*D)
		for i := range x1 {
			x1[i] = base0[i] + attnOut[i]
		}

		h2 := layerNormRows(x1, layer.FfnNormW, layer.FfnNormB, L, D, s.m.Eps)
		ffnOut, newShift2, err := channelMixForward(h2, layer.FFN, s.shift2[li], L, D, s.m.FF)
		if err != nil {
			return nil, err
		}
		s.shift2[li] = newShift2

		x2 := make([]float32, L*D)
		for i := range x2 {
			x2[i] = x1[i] + ffnOut[i]
		}
		cur = x2
	}
	return cur, nil
}

// forward embeds `tokens` and runs them through the stack — the token-in/hidden-out path.
func (s *RWKV7Session) forward(tokens []int32) ([]float32, error) {
	L, D := len(tokens), s.m.D
	hidden := make([]float32, L*D)
	for t, tok := range tokens {
		if int(tok) < 0 || int(tok) >= s.m.Vocab {
			return nil, core.NewError("rwkv7.forward: token out of range")
		}
		copy(hidden[t*D:(t+1)*D], s.m.Embed[int(tok)*D:int(tok)*D+D])
	}
	return s.forwardEmb(hidden, L)
}

// headLogits maps a single hidden [D] to vocab logits via the final norm + LM head — plain host matNT
// (no backend hook), mirroring mamba2.headLogits: the head GEMM stays host-only in this first port.
func (s *RWKV7Session) headLogits(hidden []float32) []float32 {
	normed := layerNormRows(hidden, s.m.NormW, s.m.NormB, 1, s.m.D, s.m.Eps)
	head := s.m.LMHead
	if head == nil {
		head = s.m.Embed // tied
	}
	return matNT(normed, head, 1, s.m.D, s.m.Vocab)
}

// Forward prefills `tokens` and returns the per-position hiddens [L,D] (state advanced) — the building
// block for Generate and the prefill-vs-decode equivalence test.
func (s *RWKV7Session) Forward(tokens []int32) ([]float32, error) { return s.forward(tokens) }

// Generate greedily decodes up to maxNew tokens after prefilling prompt, threading every layer's carried
// state (so each new token is O(1)). eosID < 0 disables early stop. Token-identical to a one-pass run.
func (s *RWKV7Session) Generate(prompt []int32, maxNew, eosID int) ([]int32, error) {
	if len(prompt) == 0 {
		return nil, core.NewError("rwkv7.Generate: empty prompt")
	}
	if maxNew <= 0 {
		return nil, core.NewError("rwkv7.Generate: maxNew must be > 0")
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
