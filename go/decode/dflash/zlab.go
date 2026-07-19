// SPDX-Licence-Identifier: EUPL-1.2

package dflash

import (
	"math"

	core "dappco.re/go"
)

// zlab.go is the host-f32 reference forward for the z-lab-native DFlash drafter
// architecture — the convention every checkpoint arXiv 2602.06036's own authors
// publish uses (z-lab/Qwen3-4B-DFlash-b16, z-lab/Qwen3-8B/27B/35B-A3B-DFlash,
// z-lab/gemma-4-26B-A4B-it-DFlash, ...), evidenced against a downloaded real
// checkpoint (z-lab/Qwen3-4B-DFlash-b16's actual safetensors tensor names/shapes)
// and cross-checked bit-for-bit-ish against that checkpoint's own
// modeling_dflash.py executed through transformers (trust_remote_code) — see
// docs/design-dflash-survey.md for the full receipt.
//
// It is DIFFERENT FROM — and a correction to — the architecture
// engine/metal/assistant_dflash.go was built against before any real checkpoint
// had been inspected (that file's own doc comment says as much: "no public
// gemma-4 DFlash checkpoint exists to measure accept-length against"). Concretely,
// every published z-lab checkpoint's real shape is:
//
//   - the drafter's own decoder is FIVE qwen3-style layers (plain pre-norm: just
//     input_layernorm + post_attention_layernorm, q_norm AND k_norm, SiLU-gated
//     MLP) — REGARDLESS of the target family a checkpoint drafts for (even
//     gemma-4-26B-A4B-it-DFlash's own decoder is qwen3, not gemma4); never the
//     gemma4 sandwich-norm (4 layernorms) / GELU-gated layer
//     draftLayerIntoScratch reuses;
//   - the fused verifier context is ONE linear + RMSNorm (fc, hidden_norm) over
//     the CONCATENATED target_layer_ids hidden states, producing a
//     PER-CONTEXT-TOKEN row — context length is the number of target TOKENS
//     fused in, not the count of fused LAYERS (numAux only sizes the
//     concatenated feature width fc consumes, fc.weight is [hidden,
//     numAux*hidden]);
//   - every draft layer both CROSS-attends that shared context and
//     SELF-attends its own block's noise embeddings in the SAME softmax
//     (k/v = concat(k_proj(context), k_proj(noise)), both k rows sharing one
//     k_norm) — genuine intra-block attention, not a context-only readout, and
//     not causally masked (every block position sees every other, "diffusion");
//   - the checkpoint carries no lm_head / embed_tokens / reduced-vocab head of
//     its own — no dflash.lm_head, no dflash.d2t: the drafter borrows the
//     TARGET's own tied embedding and lm_head at serve time (modeling_dflash.py
//     calls target.model.embed_tokens / target.lm_head directly), so there is no
//     vocab remap to apply for this convention.
//
// ZLabForward implements exactly DFlashDraftModel.forward(): given
// noiseEmbedding (the block's own already-embedded candidate/mask token
// embeddings) and targetHiddenRaw (the concatenated target_layer_ids hidden
// states, PRE-fusion), it returns the block's hidden state after each layer
// (for oracle-gating at depth) and the final-normed output — the input the
// (borrowed) target lm_head would consume. Position convention: the simplest
// well-defined case, sidestepping the multi-round KV-cache bookkeeping
// spec_generate does at serve time (a SERVING concern, not a per-call maths
// one) — context rows at positions [0,ctxLen), block rows at
// [ctxLen,ctxLen+blockLen), i.e. the block continues where the context left
// off. Weights are a plain map keyed EXACTLY as the checkpoint's own
// safetensors names them (fc.weight, hidden_norm.weight, layers.N.*,
// norm.weight — no "model." / "dflash." prefix), so this stays engine-free:
// no cgo, no GPU, runnable on any host.
//
//	w := dflash.ZLabWeights{"fc.weight": ..., "layers.0.self_attn.q_proj.weight": ...}
//	arch := dflash.ZLabArch{Hidden: 2560, Heads: 32, KVHeads: 8, HeadDim: 128,
//		Intermediate: 9728, NumLayers: 5, NumAux: 5, Eps: 1e-6, RopeTheta: 1e6}
//	final, perLayer, err := dflash.ZLabForward(w, arch, noiseEmbedding, targetHiddenRaw, ctxLen, blockLen)

// ZLabArch is the decoder-shape parameters a z-lab DFlash drafter's config.json
// declares flat (qwen3-family: hidden_size, num_attention_heads,
// num_key_value_heads, head_dim, intermediate_size, num_hidden_layers,
// rms_norm_eps, rope_theta — every published checkpoint sets model_type "qwen3"
// here) plus NumAux, the count of dflash_config.target_layer_ids — how many
// target-layer hidden states are concatenated into the drafter's own context
// feature (NOT the cross-attention context length, which is the number of
// target TOKENS fused in and varies per call).
type ZLabArch struct {
	Hidden       int     // hidden_size
	Heads        int     // num_attention_heads
	KVHeads      int     // num_key_value_heads
	HeadDim      int     // head_dim
	Intermediate int     // intermediate_size (MLP width)
	NumLayers    int     // num_hidden_layers (every published checkpoint: 5)
	NumAux       int     // len(dflash_config.target_layer_ids)
	Eps          float32 // rms_norm_eps
	RopeTheta    float32 // rope_theta
}

// ZLabWeights is the drafter's real tensor set, keyed exactly as the checkpoint's
// own safetensors names them, each already widened bf16->f32 (ZLabWidenBF16).
type ZLabWeights map[string][]float32

// get returns w[name] or an honest "missing tensor" error naming it — a mis-typed
// checkpoint fails loudly rather than silently zero-filling.
func (w ZLabWeights) get(name string) ([]float32, error) {
	v, ok := w[name]
	if !ok {
		return nil, core.NewError("dflash.zlab: missing required tensor " + name)
	}
	return v, nil
}

const zlabBF16Size = 2

// ZLabWidenBF16 widens a row-major bf16 tensor (2 bytes/element, the
// model/safetensors.Tensor.Data layout) to f32 — the same bit trick used
// throughout this codebase (e.g. model.bf16ToF32), duplicated here in one small
// function rather than importing a foreign package's unexported helper, so this
// package stays a leaf: no engine, no GPU, no cgo.
func ZLabWidenBF16(raw []byte) []float32 {
	n := len(raw) / zlabBF16Size
	out := make([]float32, n)
	for i := 0; i < n; i++ {
		out[i] = math.Float32frombits(uint32(uint16(raw[2*i])|uint16(raw[2*i+1])<<8) << 16)
	}
	return out
}

// ZLabForward runs the drafter's own NumLayers-layer forward. noiseEmbedding is
// [blockLen*Hidden] (the block's positions, row-major); targetHiddenRaw is
// [ctxLen*NumAux*Hidden] (the already-concatenated target_layer_ids hidden
// states, row-major per context position — concatenation is the CALLER's job,
// exactly extract_context_feature's torch.cat in the reference). It returns the
// hidden state after EVERY layer (layerOutputs[i], i in [0,NumLayers)) so a
// caller can oracle-gate at depth rather than only the end, plus the
// final-normed output.
func ZLabForward(w ZLabWeights, arch ZLabArch, noiseEmbedding, targetHiddenRaw []float32, ctxLen, blockLen int) (final []float32, layerOutputs [][]float32, err error) {
	if arch.Hidden <= 0 || arch.Heads <= 0 || arch.KVHeads <= 0 || arch.HeadDim <= 0 || arch.NumLayers <= 0 || arch.NumAux <= 0 {
		return nil, nil, core.NewError("dflash.zlab: incomplete arch dims")
	}
	if arch.Heads%arch.KVHeads != 0 {
		return nil, nil, core.NewError(core.Sprintf("dflash.zlab: heads %d not a multiple of kv_heads %d (GQA)", arch.Heads, arch.KVHeads))
	}
	if blockLen <= 0 || ctxLen < 0 {
		return nil, nil, core.NewError("dflash.zlab: blockLen must be >= 1 and ctxLen >= 0")
	}
	if len(noiseEmbedding) != blockLen*arch.Hidden {
		return nil, nil, core.NewError(core.Sprintf("dflash.zlab: noise embedding is %d floats, want %d (blockLen*hidden)", len(noiseEmbedding), blockLen*arch.Hidden))
	}
	if len(targetHiddenRaw) != ctxLen*arch.NumAux*arch.Hidden {
		return nil, nil, core.NewError(core.Sprintf("dflash.zlab: target hidden is %d floats, want %d (ctxLen*numAux*hidden)", len(targetHiddenRaw), ctxLen*arch.NumAux*arch.Hidden))
	}

	fcWeight, err := w.get("fc.weight")
	if err != nil {
		return nil, nil, err
	}
	hiddenNormWeight, err := w.get("hidden_norm.weight")
	if err != nil {
		return nil, nil, err
	}
	finalNormWeight, err := w.get("norm.weight")
	if err != nil {
		return nil, nil, err
	}

	// fc + hidden_norm ONCE — every layer cross-attends this SAME fused context
	// (it is not re-normed per layer, unlike the block's own evolving hidden).
	var targetHidden []float32
	if ctxLen > 0 {
		targetHidden = zlabLinearRows(targetHiddenRaw, ctxLen, fcWeight, arch.Hidden, arch.NumAux*arch.Hidden)
		targetHidden = zlabRMSNormRows(targetHidden, ctxLen, arch.Hidden, hiddenNormWeight, arch.Eps)
	}

	cos, sin := zlabRopeCosSin(ctxLen+blockLen, arch.HeadDim, arch.RopeTheta)

	hidden := append([]float32(nil), noiseEmbedding...)
	layerOutputs = make([][]float32, arch.NumLayers)
	for li := 0; li < arch.NumLayers; li++ {
		hidden, err = zlabDecoderLayer(w, li, hidden, targetHidden, arch, blockLen, ctxLen, cos, sin)
		if err != nil {
			return nil, nil, core.E("dflash.zlab", core.Sprintf("layer %d", li), err)
		}
		layerOutputs[li] = append([]float32(nil), hidden...)
	}
	final = zlabRMSNormRows(hidden, blockLen, arch.Hidden, finalNormWeight, arch.Eps)
	return final, layerOutputs, nil
}

// zlabDecoderLayer mirrors Qwen3DFlashDecoderLayer.forward: pre-norm residual
// attention, then pre-norm residual MLP — TWO layernorms total, never gemma4's
// sandwich four.
func zlabDecoderLayer(w ZLabWeights, li int, hidden, targetHidden []float32, arch ZLabArch, blockLen, ctxLen int, cos, sin []float32) ([]float32, error) {
	inputLN, err := w.get(core.Sprintf("layers.%d.input_layernorm.weight", li))
	if err != nil {
		return nil, err
	}
	postLN, err := w.get(core.Sprintf("layers.%d.post_attention_layernorm.weight", li))
	if err != nil {
		return nil, err
	}

	normed := zlabRMSNormRows(hidden, blockLen, arch.Hidden, inputLN, arch.Eps)
	attnOut, err := zlabAttention(w, li, normed, targetHidden, arch, blockLen, ctxLen, cos, sin)
	if err != nil {
		return nil, err
	}
	h1 := make([]float32, len(hidden))
	for i := range hidden {
		h1[i] = hidden[i] + attnOut[i]
	}

	normed2 := zlabRMSNormRows(h1, blockLen, arch.Hidden, postLN, arch.Eps)
	mlpOut, err := zlabMLP(w, li, normed2, blockLen, arch)
	if err != nil {
		return nil, err
	}
	out := make([]float32, len(h1))
	for i := range h1 {
		out[i] = h1[i] + mlpOut[i]
	}
	return out, nil
}

// zlabAttention mirrors Qwen3DFlashAttention.forward: q comes from the block's
// own (normed) hidden; k/v are the CONCATENATION of the shared target context
// and the block's own (normed) hidden — cross- and self-attention in one
// softmax, no causal mask (every block position attends every context row AND
// every other block row) — the "diffusion" in block-diffusion. q and the
// concatenated k both get their own per-head RMSNorm (q_norm / k_norm) before
// RoPE; q ropes at its own tail positions, k ropes across the full
// ctxLen+blockLen range (see zlabApplyRope).
func zlabAttention(w ZLabWeights, li int, normedHidden, targetHidden []float32, arch ZLabArch, blockLen, ctxLen int, cos, sin []float32) ([]float32, error) {
	prefix := core.Sprintf("layers.%d.self_attn.", li)
	qW, err := w.get(prefix + "q_proj.weight")
	if err != nil {
		return nil, err
	}
	kW, err := w.get(prefix + "k_proj.weight")
	if err != nil {
		return nil, err
	}
	vW, err := w.get(prefix + "v_proj.weight")
	if err != nil {
		return nil, err
	}
	oW, err := w.get(prefix + "o_proj.weight")
	if err != nil {
		return nil, err
	}
	qNorm, err := w.get(prefix + "q_norm.weight")
	if err != nil {
		return nil, err
	}
	kNorm, err := w.get(prefix + "k_norm.weight")
	if err != nil {
		return nil, err
	}

	qDim := arch.Heads * arch.HeadDim
	kvDim := arch.KVHeads * arch.HeadDim
	totalKV := ctxLen + blockLen

	q := zlabLinearRows(normedHidden, blockLen, qW, qDim, arch.Hidden)
	q = zlabPerHeadNormRows(q, blockLen, arch.Heads, arch.HeadDim, qNorm, arch.Eps)

	kNoise := zlabLinearRows(normedHidden, blockLen, kW, kvDim, arch.Hidden)
	vNoise := zlabLinearRows(normedHidden, blockLen, vW, kvDim, arch.Hidden)
	var k, v []float32
	if ctxLen > 0 {
		kCtx := zlabLinearRows(targetHidden, ctxLen, kW, kvDim, arch.Hidden)
		vCtx := zlabLinearRows(targetHidden, ctxLen, vW, kvDim, arch.Hidden)
		k = append(append(make([]float32, 0, totalKV*kvDim), kCtx...), kNoise...)
		v = append(append(make([]float32, 0, totalKV*kvDim), vCtx...), vNoise...)
	} else {
		k, v = kNoise, vNoise
	}
	k = zlabPerHeadNormRows(k, totalKV, arch.KVHeads, arch.HeadDim, kNorm, arch.Eps)

	q = zlabApplyRope(q, blockLen, arch.Heads, arch.HeadDim, cos, sin, ctxLen)
	k = zlabApplyRope(k, totalKV, arch.KVHeads, arch.HeadDim, cos, sin, 0)

	group := arch.Heads / arch.KVHeads
	scale := float32(1.0 / math.Sqrt(float64(arch.HeadDim)))
	attnOut := make([]float32, blockLen*qDim)
	scores := make([]float32, totalKV)
	for r := 0; r < blockLen; r++ {
		for h := 0; h < arch.Heads; h++ {
			kvh := h / group
			qVec := q[(r*arch.Heads+h)*arch.HeadDim : (r*arch.Heads+h+1)*arch.HeadDim]
			maxS := float32(math.Inf(-1))
			for t := 0; t < totalKV; t++ {
				kVec := k[(t*arch.KVHeads+kvh)*arch.HeadDim : (t*arch.KVHeads+kvh+1)*arch.HeadDim]
				var dot float32
				for d := 0; d < arch.HeadDim; d++ {
					dot += qVec[d] * kVec[d]
				}
				s := dot * scale
				scores[t] = s
				if s > maxS {
					maxS = s
				}
			}
			var sum float32
			for t := 0; t < totalKV; t++ {
				e := float32(math.Exp(float64(scores[t] - maxS)))
				scores[t] = e
				sum += e
			}
			outBase := (r*arch.Heads + h) * arch.HeadDim
			for d := 0; d < arch.HeadDim; d++ {
				var acc float32
				for t := 0; t < totalKV; t++ {
					acc += (scores[t] / sum) * v[(t*arch.KVHeads+kvh)*arch.HeadDim+d]
				}
				attnOut[outBase+d] = acc
			}
		}
	}
	return zlabLinearRows(attnOut, blockLen, oW, arch.Hidden, qDim), nil
}

// zlabMLP mirrors Qwen3MLP: down(silu(gate(x)) * up(x)) — SwiGLU, never gemma's
// GELU-gate.
func zlabMLP(w ZLabWeights, li int, x []float32, rows int, arch ZLabArch) ([]float32, error) {
	prefix := core.Sprintf("layers.%d.mlp.", li)
	gateW, err := w.get(prefix + "gate_proj.weight")
	if err != nil {
		return nil, err
	}
	upW, err := w.get(prefix + "up_proj.weight")
	if err != nil {
		return nil, err
	}
	downW, err := w.get(prefix + "down_proj.weight")
	if err != nil {
		return nil, err
	}
	gate := zlabLinearRows(x, rows, gateW, arch.Intermediate, arch.Hidden)
	up := zlabLinearRows(x, rows, upW, arch.Intermediate, arch.Hidden)
	gated := make([]float32, len(gate))
	for i := range gate {
		g := gate[i]
		silu := g / (1 + float32(math.Exp(float64(-g))))
		gated[i] = silu * up[i]
	}
	return zlabLinearRows(gated, rows, downW, arch.Hidden, arch.Intermediate), nil
}

// zlabLinear applies a bias-free linear layer y = x . Wt (weight stored PyTorch-
// style, row-major [outDim, inDim] — every projection in this checkpoint family
// is bias-free).
func zlabLinear(x, weight []float32, outDim, inDim int) []float32 {
	out := make([]float32, outDim)
	for o := 0; o < outDim; o++ {
		row := weight[o*inDim : (o+1)*inDim]
		var acc float32
		for i, xv := range x {
			acc += xv * row[i]
		}
		out[o] = acc
	}
	return out
}

// zlabLinearRows applies zlabLinear to each of rows rows of a [rows, inDim]
// matrix, yielding a [rows, outDim] matrix (both flattened row-major).
func zlabLinearRows(x []float32, rows int, weight []float32, outDim, inDim int) []float32 {
	out := make([]float32, rows*outDim)
	for r := 0; r < rows; r++ {
		y := zlabLinear(x[r*inDim:(r+1)*inDim], weight, outDim, inDim)
		copy(out[r*outDim:(r+1)*outDim], y)
	}
	return out
}

// zlabRMSNormRows applies the PLAIN Qwen3RMSNorm (x * rsqrt(mean(x^2)+eps) *
// weight — no gemma-style "+1" zero-centred shift) independently to each of
// rows rows of a [rows, dim] matrix.
func zlabRMSNormRows(x []float32, rows, dim int, weight []float32, eps float32) []float32 {
	out := make([]float32, rows*dim)
	for r := 0; r < rows; r++ {
		row := x[r*dim : (r+1)*dim]
		var ss float64
		for _, v := range row {
			ss += float64(v) * float64(v)
		}
		inv := float32(1.0 / math.Sqrt(ss/float64(dim)+float64(eps)))
		for i := 0; i < dim; i++ {
			out[r*dim+i] = row[i] * inv * weight[i]
		}
	}
	return out
}

// zlabPerHeadNormRows applies the same RMSNorm as zlabRMSNormRows but per HEAD:
// x is [rows, heads, headDim] flattened row-major, which is byte-identical to
// [rows*heads, headDim] flattened — so a per-head norm is exactly a row norm
// over rows*heads rows of width headDim, using the ONE [headDim] weight shared
// by every head (q_norm / k_norm are not per-head weights).
func zlabPerHeadNormRows(x []float32, rows, heads, headDim int, weight []float32, eps float32) []float32 {
	return zlabRMSNormRows(x, rows*heads, headDim, weight, eps)
}

// zlabRopeCosSin computes the standard (non-interleaved, split-half) rotary
// cos/sin table for positions [0,total), matching Qwen3RotaryEmbedding: half-dim
// inverse frequencies duplicated across both halves of each row
// (emb = cat(freqs, freqs)). Returned flattened row-major [total, headDim].
func zlabRopeCosSin(total, headDim int, theta float32) (cos, sin []float32) {
	half := headDim / 2
	cos = make([]float32, total*headDim)
	sin = make([]float32, total*headDim)
	invFreq := make([]float64, half)
	for i := 0; i < half; i++ {
		invFreq[i] = 1.0 / math.Pow(float64(theta), float64(2*i)/float64(headDim))
	}
	for t := 0; t < total; t++ {
		for i := 0; i < half; i++ {
			th := float64(t) * invFreq[i]
			c := float32(math.Cos(th))
			s := float32(math.Sin(th))
			cos[t*headDim+i] = c
			cos[t*headDim+half+i] = c
			sin[t*headDim+i] = s
			sin[t*headDim+half+i] = s
		}
	}
	return cos, sin
}

// zlabApplyRope ropes rows [rowOffset, rowOffset+rows) of x (shape [rows, heads,
// headDim]) against cos/sin rows [rowOffset, rowOffset+rows) — i.e. x's row r
// corresponds to absolute position rowOffset+r, mirroring
// apply_rotary_pos_emb's rotate_half convention: out[d] = x1*cos - x2*sin,
// out[d+half] = x1*sin + x2*cos for x1,x2 the two halves of the head.
func zlabApplyRope(x []float32, rows, heads, headDim int, cos, sin []float32, rowOffset int) []float32 {
	out := make([]float32, len(x))
	half := headDim / 2
	for r := 0; r < rows; r++ {
		cRow := cos[(rowOffset+r)*headDim : (rowOffset+r)*headDim+half]
		sRow := sin[(rowOffset+r)*headDim : (rowOffset+r)*headDim+half]
		for h := 0; h < heads; h++ {
			base := (r*heads + h) * headDim
			for d := 0; d < half; d++ {
				x1 := x[base+d]
				x2 := x[base+half+d]
				c := cRow[d]
				s := sRow[d]
				out[base+d] = x1*c - x2*s
				out[base+half+d] = x2*c + x1*s
			}
		}
	}
	return out
}
