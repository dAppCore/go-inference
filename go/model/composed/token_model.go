// SPDX-Licence-Identifier: EUPL-1.2

package composed

import (
	"bytes"
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

// HiddenSize is the model hidden dimension D — the width of an embedding / hidden row. The serve wrap
// reports it on inference.ModelInfo.
func (tm *ComposedTokenModel) HiddenSize() int { return tm.m.D }

// NumLayers is the composed block count (each a config-dispatched sequence mixer + feed-forward). The
// serve wrap reports it on inference.ModelInfo.
func (tm *ComposedTokenModel) NumLayers() int { return len(tm.m.Layers) }

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
	for t := range L {
		res[t] = f32ToBF16Bytes(out[t*D : (t+1)*D])
	}
	return res, nil
}

// OpenSession opens a fresh hybrid stepper (the SessionModel fast path — O(1)/token, each layer threading
// its own recurrent or KV state).
func (tm *ComposedTokenModel) OpenSession() (model.DecodeStepper, error) {
	return &composedStepper{s: NewSession(tm.m)}, nil
}

// composedStepper is a stateful hybrid decode session (SessionModel.OpenSession) — plus, when the most
// recent Step/PrefillBatch's forwardEmb call fused the model's final RMSNorm + LM head GEMM onto the LAST
// layer's tail command buffer, the logits that fuse already computed. headHidden/headLogits cache them
// keyed to the exact hidden bytes that call just returned: Head reuses the cache for THAT hidden (the
// shared generation loop's usage pattern is at most one Head call between two Step calls — see
// model.generateStepwiseWithSession) and falls through to the ordinary host-RMSNorm + device-GEMM
// recompute for any other hidden (a mismatched/stale hidden, or a session whose last call didn't fuse).
type composedStepper struct {
	s                    *ComposedSession
	headHidden, headBF16 []byte
}

// cacheHeadFuse records st.s.PendingHeadLogits() (if the tail just fused it) keyed to hidden — the
// bookkeeping Step and PrefillBatch share after their own forwardEmb call, so the Head call that follows
// either one can reuse a just-fused head GEMM instead of recomputing it. Clears the cache when this call
// didn't fuse, so a stale entry never survives past the call that didn't produce one.
func (st *composedStepper) cacheHeadFuse(hidden []byte) {
	if lg := st.s.PendingHeadLogits(); lg != nil {
		st.headHidden, st.headBF16 = hidden, f32ToBF16Bytes(lg)
	} else {
		st.headHidden, st.headBF16 = nil, nil
	}
}

// Step decodes one token embedding (bf16) over the resident per-layer state, returning the output hidden
// (bf16). When the fused device path just computed this token's logits as a side effect (the terminal
// head-fuse on the LAST layer's tail), they are cached for the Head call that follows.
func (st *composedStepper) Step(emb []byte) ([]byte, error) {
	D := st.s.m.D
	if len(emb) != D*2 {
		return nil, core.NewError("composed.Step: emb must be dModel bf16 bytes")
	}
	out, err := st.s.forwardEmb(bf16BytesToF32(emb), 1)
	if err != nil {
		return nil, err
	}
	outBytes := f32ToBF16Bytes(out)
	st.cacheHeadFuse(outBytes)
	return outBytes, nil
}

// PrefillBatch runs the WHOLE prompt through the resident per-layer state in ONE forwardEmb call (L =
// len(embs)) instead of the len(embs) Step calls model.generateStepwiseWithSession otherwise makes during
// prefill — implements model.BatchPrefillStepper. Every mixer's Forward already accepts L>1 (the attention
// mixer's causal KV-cache growth carried via pos0, the gated-delta/mamba2/rwkv7 recurrences) and the FFN
// tail is row-independent (no cross-row mixing — only the mixer itself is sequential), so one L-token
// forwardEmb call advances every layer's state identically to L single-token calls; the only observable
// difference is that a device fusion bound to the LAST layer's tail (e.g. the head-fuse,
// ResidualNormMLPProjHeadDevice) now runs ONCE for the whole batch instead of once per prompt token — see
// forwardEmb's terminal-collapse branch. Returns the LAST token's hidden state (dModel bf16), matching
// Step's contract, and caches that row's fused head logits exactly as Step does, so the Head call for the
// first sampled token (immediately following prefill) still gets the fast path.
func (st *composedStepper) PrefillBatch(embs [][]byte) ([]byte, error) {
	D := st.s.m.D
	L := len(embs)
	if L == 0 {
		return nil, core.NewError("composed.PrefillBatch: empty batch")
	}
	hidden := make([]float32, L*D)
	for i, e := range embs {
		if len(e) != D*2 {
			return nil, core.NewError("composed.PrefillBatch: each input must be dModel bf16 bytes")
		}
		copy(hidden[i*D:(i+1)*D], bf16BytesToF32(e))
	}
	out, err := st.s.forwardEmb(hidden, L)
	if err != nil {
		return nil, err
	}
	last := f32ToBF16Bytes(out[(L-1)*D : L*D])
	st.cacheHeadFuse(last)
	return last, nil
}

// Head maps a final hidden (dModel bf16) to vocab logits (vocab bf16) — the SessionModel fast path's
// optional LMHead capability (model.generateStepwiseWithSession prefers it over ComposedTokenModel.Head
// when a stepper implements it). Reuses the last Step call's fused-device logits when hidden matches
// exactly what that Step returned; recomputes via the model's own (session-independent) path otherwise.
func (st *composedStepper) Head(hidden []byte) ([]byte, error) {
	if st.headBF16 != nil && bytes.Equal(hidden, st.headHidden) {
		out := st.headBF16
		st.headHidden, st.headBF16 = nil, nil
		return out, nil
	}
	return f32ToBF16Bytes(NewSession(st.s.m).headLogits(bf16BytesToF32(hidden))), nil
}

var (
	_ model.TokenModel          = (*ComposedTokenModel)(nil)
	_ model.SessionModel        = (*ComposedTokenModel)(nil)
	_ model.DecodeStepper       = (*composedStepper)(nil)
	_ model.LMHead              = (*composedStepper)(nil)
	_ model.BatchPrefillStepper = (*composedStepper)(nil)
)
