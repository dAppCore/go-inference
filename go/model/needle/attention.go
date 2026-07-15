// SPDX-Licence-Identifier: EUPL-1.2

package needle

import "math"

// attention is the one attention block Needle reuses for encoder self-attention,
// decoder masked self-attention and decoder cross-attention (NeedleAttention).
// It is grouped-query (8 query heads share 4 key/value heads, kvHead = h/group),
// applies per-head QK-norm before RoPE, and scales scores by 1/sqrt(head_dim).
//
//   - prefix names the weight group, e.g. "model.decoder.layers.0.encoder_attn".
//   - xq/xkv are already-normalised hidden states ([qLen,hidden] / [kvLen,hidden]);
//     for self-attention they are the same slice.
//   - rope is nil for cross-attention (no positional rotation there).
//   - causal masks key j>i (decoder self-attention); false is bidirectional.
//
// Returns the out_proj'd result [qLen, hidden].
func (m *Model) attention(prefix string, xq []float32, qLen int, xkv []float32, kvLen int, rope *ropeTable, causal bool) []float32 {
	c := m.cfg
	hidden := c.HiddenSize
	headDim := c.HeadDim()
	kvDim := c.KVDim()
	group := c.NumHeads / c.NumKVHeads
	scale := float32(1.0 / math.Sqrt(float64(headDim)))

	qProj := m.w.get(prefix + ".q_proj.weight")
	kProj := m.w.get(prefix + ".k_proj.weight")
	vProj := m.w.get(prefix + ".v_proj.weight")
	oProj := m.w.get(prefix + ".out_proj.weight")
	qNorm := m.w.get(prefix + ".q_norm.weight")
	kNorm := m.w.get(prefix + ".k_norm.weight")

	// Project then reshape to [len, heads, head_dim] (flattened row-major).
	q := linearRows(xq, qLen, qProj, hidden, hidden)  // [qLen, numHeads*headDim]
	k := linearRows(xkv, kvLen, kProj, kvDim, hidden) // [kvLen, numKV*headDim]
	v := linearRows(xkv, kvLen, vProj, kvDim, hidden)

	// Per-head QK-norm, then RoPE (self-attention only).
	for i := range qLen {
		for h := range c.NumHeads {
			head := q[i*hidden+h*headDim : i*hidden+h*headDim+headDim]
			copy(head, zcRMSNorm(head, qNorm, c.RMSNormEps))
			if rope != nil {
				rope.apply(head, i)
			}
		}
	}
	for j := range kvLen {
		for h := range c.NumKVHeads {
			head := k[j*kvDim+h*headDim : j*kvDim+h*headDim+headDim]
			copy(head, zcRMSNorm(head, kNorm, c.RMSNormEps))
			if rope != nil {
				rope.apply(head, j)
			}
		}
	}

	// Attention per query head against its grouped kv head.
	attnOut := make([]float32, qLen*hidden)
	scores := make([]float32, kvLen)
	for h := range c.NumHeads {
		kvHead := h / group
		for i := range qLen {
			qv := q[i*hidden+h*headDim : i*hidden+h*headDim+headDim]
			limit := kvLen
			if causal {
				limit = i + 1 // key j > i is masked out
			}
			for j := range kvLen {
				if j >= limit {
					scores[j] = float32(math.Inf(-1))
					continue
				}
				kv := k[j*kvDim+kvHead*headDim : j*kvDim+kvHead*headDim+headDim]
				var dot float32
				for d := range headDim {
					dot += qv[d] * kv[d]
				}
				scores[j] = dot * scale
			}
			softmaxInPlace(scores[:kvLen])
			out := attnOut[i*hidden+h*headDim : i*hidden+h*headDim+headDim]
			for j := range kvLen {
				p := scores[j]
				if p == 0 {
					continue
				}
				vv := v[j*kvDim+kvHead*headDim : j*kvDim+kvHead*headDim+headDim]
				for d := range headDim {
					out[d] += p * vv[d]
				}
			}
		}
	}
	return linearRows(attnOut, qLen, oProj, hidden, hidden)
}
