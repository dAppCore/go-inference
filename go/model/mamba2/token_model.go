// SPDX-Licence-Identifier: EUPL-1.2

package mamba2

import (
	"math"

	core "dappco.re/go"
	"dappco.re/go/inference/model"
)

// token_model.go adapts a MambaModel to the engine's model.TokenModel / model.SessionModel contract, so
// the shared Generate loop and the serve path drive it exactly like a transformer — no generation logic
// re-rolled. The contract seam is bf16 []byte (embeddings, hiddens, logits); the model runs f32, so this
// converts at the boundary. Mamba is naturally incremental (a fixed recurrent state, O(1)/token), so it
// implements the SessionModel FAST path: OpenSession returns a stepper that threads the recurrent state.

func f32ToBF16Bytes(v []float32) []byte {
	out := make([]byte, len(v)*2)
	for i, f := range v {
		bits := math.Float32bits(f)
		r := uint16((bits + 0x7fff + ((bits >> 16) & 1)) >> 16) // round-to-nearest-even
		out[2*i], out[2*i+1] = byte(r), byte(r>>8)
	}
	return out
}

func bf16BytesToF32(b []byte) []float32 {
	out := make([]float32, len(b)/2)
	for i := range out {
		bits := uint16(b[2*i]) | uint16(b[2*i+1])<<8
		out[i] = math.Float32frombits(uint32(bits) << 16)
	}
	return out
}

// MambaTokenModel wraps a MambaModel as a model.SessionModel.
type MambaTokenModel struct {
	m *MambaModel
}

// NewTokenModel adapts a loaded MambaModel to the serve/generate contract.
func NewTokenModel(m *MambaModel) *MambaTokenModel { return &MambaTokenModel{m: m} }

func (tm *MambaTokenModel) Vocab() int { return tm.m.Vocab }

// Embed maps a token id to its input embedding (dModel bf16 bytes).
func (tm *MambaTokenModel) Embed(id int32) ([]byte, error) {
	if int(id) < 0 || int(id) >= tm.m.Vocab {
		return nil, core.NewError("mamba2.Embed: id out of range")
	}
	row := tm.m.Embed[int(id)*tm.m.D : int(id)*tm.m.D+tm.m.D]
	return f32ToBF16Bytes(row), nil
}

// Head maps a final hidden (dModel bf16) to vocab logits (vocab bf16) via the final norm + LM head.
func (tm *MambaTokenModel) Head(hidden []byte) ([]byte, error) {
	if len(hidden) != tm.m.D*2 {
		return nil, core.NewError("mamba2.Head: hidden must be dModel bf16 bytes")
	}
	logits := NewSession(tm.m).headLogits(bf16BytesToF32(hidden))
	return f32ToBF16Bytes(logits), nil
}

// DecodeForward runs the whole-sequence stack over T input embeddings (bf16) → T hiddens (bf16), fresh
// recurrent state — the whole-sequence fallback (OpenSession is the fast incremental path).
func (tm *MambaTokenModel) DecodeForward(inputs [][]byte) ([][]byte, error) {
	L, D := len(inputs), tm.m.D
	if L == 0 {
		return nil, nil
	}
	hidden := make([]float32, L*D)
	for t, e := range inputs {
		if len(e) != D*2 {
			return nil, core.NewError("mamba2.DecodeForward: each input must be dModel bf16 bytes")
		}
		copy(hidden[t*D:(t+1)*D], bf16BytesToF32(e))
	}
	out, err := NewSession(tm.m).forwardEmb(hidden, L)
	if err != nil {
		return nil, err
	}
	res := make([][]byte, L)
	for t := range L {
		res[t] = f32ToBF16Bytes(out[t*D : (t+1)*D])
	}
	return res, nil
}

// OpenSession opens a fresh recurrent stepper — the SessionModel fast path (O(1)/token).
func (tm *MambaTokenModel) OpenSession() (model.DecodeStepper, error) {
	return &mambaStepper{s: NewSession(tm.m)}, nil
}

// mambaStepper is the per-conversation recurrent decode stepper.
type mambaStepper struct{ s *MambaSession }

// Step decodes one token embedding (bf16) over the resident recurrent state, returning the output
// hidden (bf16) and advancing the conv-state + SSM state.
func (st *mambaStepper) Step(emb []byte) ([]byte, error) {
	D := st.s.m.D
	if len(emb) != D*2 {
		return nil, core.NewError("mamba2.Step: emb must be dModel bf16 bytes")
	}
	out, err := st.s.forwardEmb(bf16BytesToF32(emb), 1)
	if err != nil {
		return nil, err
	}
	return f32ToBF16Bytes(out), nil
}

// compile-time proof the wrapper satisfies the full contract.
var (
	_ model.TokenModel    = (*MambaTokenModel)(nil)
	_ model.SessionModel  = (*MambaTokenModel)(nil)
	_ model.DecodeStepper = (*mambaStepper)(nil)
)
