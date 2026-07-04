// SPDX-Licence-Identifier: EUPL-1.2

package composed

import (
	"math"

	core "dappco.re/go"
	"dappco.re/go/inference/model"
)

// token_model.go adapts a ComposedModel to model.TokenModel + model.SessionModel, so the shared Generate
// loop and the serve path drive the Qwen 3.6 hybrid exactly like a transformer — no generation logic
// re-rolled. The seam is bf16 []byte (Embed→embedding, Step/DecodeForward→hidden, Head→logits); the model
// runs f32, converting at the boundary. The hybrid is incremental (each layer threads its own recurrent or
// KV state), so it implements the SessionModel fast path: OpenSession returns a stepper threading every
// layer's state.

func f32ToBF16Bytes(v []float32) []byte {
	out := make([]byte, len(v)*2)
	for i, f := range v {
		bits := math.Float32bits(f)
		r := uint16((bits + 0x7fff + ((bits >> 16) & 1)) >> 16)
		out[2*i], out[2*i+1] = byte(r), byte(r>>8)
	}
	return out
}

func bf16BytesToF32(b []byte) []float32 {
	out := make([]float32, len(b)/2)
	for i := range out {
		out[i] = math.Float32frombits(uint32(uint16(b[2*i])|uint16(b[2*i+1])<<8) << 16)
	}
	return out
}

// ComposedTokenModel wraps a ComposedModel as a model.SessionModel.
type ComposedTokenModel struct{ m *ComposedModel }

// NewTokenModel adapts a loaded ComposedModel to the serve/generate contract.
func NewTokenModel(m *ComposedModel) *ComposedTokenModel { return &ComposedTokenModel{m: m} }

func (tm *ComposedTokenModel) Vocab() int { return tm.m.Vocab }

// Embed maps a token id to its input embedding (dModel bf16 bytes).
func (tm *ComposedTokenModel) Embed(id int32) ([]byte, error) {
	if int(id) < 0 || int(id) >= tm.m.Vocab {
		return nil, core.NewError("composed.Embed: id out of range")
	}
	return f32ToBF16Bytes(tm.m.Embed[int(id)*tm.m.D : int(id)*tm.m.D+tm.m.D]), nil
}

// Head maps a final hidden (dModel bf16) to vocab logits (vocab bf16).
func (tm *ComposedTokenModel) Head(hidden []byte) ([]byte, error) {
	if len(hidden) != tm.m.D*2 {
		return nil, core.NewError("composed.Head: hidden must be dModel bf16 bytes")
	}
	return f32ToBF16Bytes(NewSession(tm.m).headLogits(bf16BytesToF32(hidden))), nil
}

// DecodeForward runs the whole-sequence stack over T input embeddings (bf16) → T hiddens (bf16), fresh
// per-layer state.
func (tm *ComposedTokenModel) DecodeForward(inputs [][]byte) ([][]byte, error) {
	L, D := len(inputs), tm.m.D
	if L == 0 {
		return nil, nil
	}
	hidden := make([]float32, L*D)
	for t, e := range inputs {
		if len(e) != D*2 {
			return nil, core.NewError("composed.DecodeForward: each input must be dModel bf16 bytes")
		}
		copy(hidden[t*D:(t+1)*D], bf16BytesToF32(e))
	}
	out, err := NewSession(tm.m).forwardEmb(hidden, L)
	if err != nil {
		return nil, err
	}
	res := make([][]byte, L)
	for t := 0; t < L; t++ {
		res[t] = f32ToBF16Bytes(out[t*D : (t+1)*D])
	}
	return res, nil
}

// OpenSession opens a fresh hybrid stepper (the SessionModel fast path — O(1)/token, each layer threading
// its own recurrent or KV state).
func (tm *ComposedTokenModel) OpenSession() (model.DecodeStepper, error) {
	return &composedStepper{s: NewSession(tm.m)}, nil
}

type composedStepper struct{ s *ComposedSession }

// Step decodes one token embedding (bf16) over the resident per-layer state, returning the output hidden
// (bf16).
func (st *composedStepper) Step(emb []byte) ([]byte, error) {
	D := st.s.m.D
	if len(emb) != D*2 {
		return nil, core.NewError("composed.Step: emb must be dModel bf16 bytes")
	}
	out, err := st.s.forwardEmb(bf16BytesToF32(emb), 1)
	if err != nil {
		return nil, err
	}
	return f32ToBF16Bytes(out), nil
}

var (
	_ model.TokenModel    = (*ComposedTokenModel)(nil)
	_ model.SessionModel  = (*ComposedTokenModel)(nil)
	_ model.DecodeStepper = (*composedStepper)(nil)
)
