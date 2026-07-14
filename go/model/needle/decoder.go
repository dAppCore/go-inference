// SPDX-Licence-Identifier: EUPL-1.2

package needle

import core "dappco.re/go"

// decode runs the full decoder over the current decoder token ids, cross-attending
// to the encoder output. Each of the 8 layers is masked self-attention then
// cross-attention, both pre-norm with their own scalar sigmoid gate, no FFN:
//
//	normed = input_layernorm(h)
//	h      = clip(h + sigmoid(self_attn_gate)  * self_attn(normed, normed))   # causal, RoPE
//	cross  = encoder_attn(encoder_attn_layer_norm(h), encoderHidden)          # cross, NO RoPE
//	h      = clip(h + sigmoid(cross_attn_gate) * cross)
//
// The cross-attention keys/values come from the FINAL encoder output (the same
// encoderHidden for every layer, each with its own k/v projection). Returns the
// decoder hidden states [decLen, hidden] after the trailing norm.
func (m *Model) decode(decIDs []int, encoderHidden []float32, encLen int) []float32 {
	c := m.cfg
	hidden := c.HiddenSize
	decLen := len(decIDs)

	h := m.embed(decIDs) // [decLen, hidden], scaled
	rope := newRopeTable(c.HeadDim(), c.RopeTheta, decLen)

	for layer := range c.NumDecoderLayers {
		p := core.Sprintf("model.decoder.layers.%d", layer)

		normed := zcRMSNormRows(h, decLen, hidden, m.w.get(p+".input_layernorm.weight"), c.RMSNormEps)
		selfAttn := m.attention(p+".self_attn", normed, decLen, normed, decLen, rope, true)
		selfGate := sigmoid(m.w.get(p + ".self_attn_gate")[0])
		for i := range decLen * hidden {
			h[i] = clipAdd(h[i], selfGate*selfAttn[i])
		}

		crossNormed := zcRMSNormRows(h, decLen, hidden, m.w.get(p+".encoder_attn_layer_norm.weight"), c.RMSNormEps)
		cross := m.attention(p+".encoder_attn", crossNormed, decLen, encoderHidden, encLen, nil, false)
		crossGate := sigmoid(m.w.get(p + ".cross_attn_gate")[0])
		for i := range decLen * hidden {
			h[i] = clipAdd(h[i], crossGate*cross[i])
		}
	}
	return zcRMSNormRows(h, decLen, hidden, m.w.get("model.decoder.norm.weight"), c.RMSNormEps)
}
