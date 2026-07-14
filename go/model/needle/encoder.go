// SPDX-Licence-Identifier: EUPL-1.2

package needle

import core "dappco.re/go"

// encode runs the full bidirectional encoder over token ids: embed (scaled by
// sqrt(hidden)), then 12 gated self-attention layers with no FFN, then final_norm.
// Each layer is pre-norm with a scalar sigmoid gate on the attention branch:
//
//	normed = input_layernorm(h)
//	attn   = self_attn(normed, normed)           # bidirectional, RoPE
//	h      = clip(h + sigmoid(attn_gate) * attn) # no FFN
//
// Returns the encoder hidden states [seqLen, hidden] the decoder cross-attends to.
func (m *Model) encode(ids []int) []float32 {
	c := m.cfg
	hidden := c.HiddenSize
	seqLen := len(ids)

	h := m.embed(ids) // [seqLen, hidden], already scaled
	rope := newRopeTable(c.HeadDim(), c.RopeTheta, seqLen)

	for layer := range c.NumEncoderLayers {
		p := core.Sprintf("model.encoder.layers.%d", layer)
		normed := zcRMSNormRows(h, seqLen, hidden, m.w.get(p+".input_layernorm.weight"), c.RMSNormEps)
		attn := m.attention(p+".self_attn", normed, seqLen, normed, seqLen, rope, false)
		gate := sigmoid(m.w.get(p + ".attn_gate")[0])
		for i := range seqLen * hidden {
			h[i] = clipAdd(h[i], gate*attn[i])
		}
	}
	return zcRMSNormRows(h, seqLen, hidden, m.w.get("model.encoder.final_norm.weight"), c.RMSNormEps)
}
