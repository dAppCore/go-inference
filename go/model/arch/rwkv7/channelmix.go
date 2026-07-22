// SPDX-Licence-Identifier: EUPL-1.2

package rwkv7

import core "dappco.re/go"

// channelmix.go is RWKV-7's channel-mix sub-block — fla.models.rwkv7.modeling_rwkv7.RWKV7FeedForward —
// the arch's FFN equivalent, distinct from a transformer's SwiGLU/GeGLU MLP: its own token-shift register
// (separate from time-mix's, since it observes a different input stream — see model.go), a single mix
// vector (no LoRA), and a squared-ReLU activation (config hidden_act="sqrelu", every released RWKV-7
// checkpoint):
//
//	delta   = tokenShift(x, priorShift)
//	xk      = x + delta*XK
//	hidden  = relu(xk @ KeyProj^T) ^ 2          // sqrelu — fla.modules.activations.sqrelu_fwd
//	out     = hidden @ ValueProj^T

// channelMixWeights holds one layer's channel-mix parameters: the token-shift mix vector plus the two
// dense projections (no bias on either, matching nn.Linear(..., bias=False) in the reference).
type channelMixWeights struct {
	XK        []float32 // [D]
	KeyProj   []float32 // [FF,D]
	ValueProj []float32 // [D,FF]
}

// channelMixForward runs one layer's channel-mix over x [L,D] (already ffn_norm'd), threading this
// layer's own token-shift register. Returns the [L,D] output and the advanced shift register.
func channelMixForward(x []float32, w *channelMixWeights, priorShift []float32, L, D, FF int) (out, newShift []float32, err error) {
	if w == nil {
		return nil, nil, core.NewError("rwkv7.channelMixForward: nil weights")
	}
	if len(x) != L*D || len(w.XK) != D || len(w.KeyProj) != FF*D || len(w.ValueProj) != D*FF {
		return nil, nil, core.NewError("rwkv7.channelMixForward: bad geometry or x size")
	}

	delta, newShift := tokenShift(x, priorShift, L, D)
	xk := addcmulRows(x, delta, w.XK, L, D)

	hidden, err := projMatMul(xk, w.KeyProj, L, D, FF)
	if err != nil {
		return nil, nil, err
	}
	for i, v := range hidden {
		if v < 0 {
			v = 0
		}
		hidden[i] = v * v // sqrelu = relu(x)^2
	}

	out, err = projMatMul(hidden, w.ValueProj, L, FF, D)
	if err != nil {
		return nil, nil, err
	}
	return out, newShift, nil
}
