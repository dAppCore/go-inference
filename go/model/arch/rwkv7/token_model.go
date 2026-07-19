// SPDX-Licence-Identifier: EUPL-1.2

package rwkv7

import (
	"math"

	core "dappco.re/go"
	"dappco.re/go/inference/model"
)

// token_model.go adapts an RWKV7Model to the engine's model.TokenModel / model.SessionModel contract, so
// the shared Generate loop and the serve path drive it exactly like a transformer — no generation logic
// re-rolled. The contract seam is bf16 []byte (embeddings, hiddens, logits); the model runs f32, so this
// converts at the boundary. RWKV-7 is naturally incremental (every layer's state is fixed-size, O(1)/
// token), so it implements the SessionModel FAST path: OpenSession returns a stepper that threads the
// carried state — the same shape as mamba2/token_model.go (bf16 helpers duplicated rather than shared:
// neither arch package imports the other, AX-8 sideways).

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

// RWKV7TokenModel wraps an RWKV7Model as a model.SessionModel.
type RWKV7TokenModel struct {
	m *RWKV7Model
}

// NewTokenModel adapts a loaded RWKV7Model to the serve/generate contract.
func NewTokenModel(m *RWKV7Model) *RWKV7TokenModel { return &RWKV7TokenModel{m: m} }

func (tm *RWKV7TokenModel) Vocab() int { return tm.m.Vocab }

// Embed maps a token id to its input embedding (dModel bf16 bytes).
func (tm *RWKV7TokenModel) Embed(id int32) ([]byte, error) {
	if int(id) < 0 || int(id) >= tm.m.Vocab {
		return nil, core.NewError("rwkv7.Embed: id out of range")
	}
	row := tm.m.Embed[int(id)*tm.m.D : int(id)*tm.m.D+tm.m.D]
	return f32ToBF16Bytes(row), nil
}

// Head maps a final hidden (dModel bf16) to vocab logits (vocab bf16) via the final norm + LM head.
func (tm *RWKV7TokenModel) Head(hidden []byte) ([]byte, error) {
	if len(hidden) != tm.m.D*2 {
		return nil, core.NewError("rwkv7.Head: hidden must be dModel bf16 bytes")
	}
	logits := NewSession(tm.m).headLogits(bf16BytesToF32(hidden))
	return f32ToBF16Bytes(logits), nil
}

// DecodeForward runs the whole-sequence stack over T input embeddings (bf16) -> T hiddens (bf16), fresh
// state — the whole-sequence fallback (OpenSession is the fast incremental path).
func (tm *RWKV7TokenModel) DecodeForward(inputs [][]byte) ([][]byte, error) {
	L, D := len(inputs), tm.m.D
	if L == 0 {
		return nil, nil
	}
	hidden := make([]float32, L*D)
	for t, e := range inputs {
		if len(e) != D*2 {
			return nil, core.NewError("rwkv7.DecodeForward: each input must be dModel bf16 bytes")
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
func (tm *RWKV7TokenModel) OpenSession() (model.DecodeStepper, error) {
	return &rwkv7Stepper{s: NewSession(tm.m)}, nil
}

// rwkv7Stepper is the per-conversation recurrent decode stepper.
type rwkv7Stepper struct{ s *RWKV7Session }

// Step decodes one token embedding (bf16) over the resident state, returning the output hidden (bf16)
// and advancing every layer's WKV7 state + both token-shift registers.
func (st *rwkv7Stepper) Step(emb []byte) ([]byte, error) {
	D := st.s.m.D
	if len(emb) != D*2 {
		return nil, core.NewError("rwkv7.Step: emb must be dModel bf16 bytes")
	}
	out, err := st.s.forwardEmb(bf16BytesToF32(emb), 1)
	if err != nil {
		return nil, err
	}
	return f32ToBF16Bytes(out), nil
}

// compile-time proof the wrapper satisfies the full contract.
var (
	_ model.TokenModel    = (*RWKV7TokenModel)(nil)
	_ model.SessionModel  = (*RWKV7TokenModel)(nil)
	_ model.DecodeStepper = (*rwkv7Stepper)(nil)
)
