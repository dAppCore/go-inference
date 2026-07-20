// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"math"

	core "dappco.re/go"
	zlabdflash "dappco.re/go/inference/model/arch/z-lab/dflash"
)

// dflash_zlab.go is the engine's block forward for the REAL z-lab DFlash
// drafter architecture (arXiv 2602.06036 as its own lab actually ships it) —
// the #52 rewrite the #37 survey called for. It consumes the payload
// model/arch/z-lab/dflash assembles and reproduces decode/dflash.ZLabForward
// (the host-f32 reference cross-validated against the real
// z-lab/Qwen3-4B-DFlash-b16 checkpoint's own modeling_dflash.py — see
// docs/design-dflash-survey.md §4-5 and docs/design-dflash-forward.md), on the
// qwen vision port's numeric tiers: every projection dispatches to the steel
// f32 GEMM above the 2^20 M·K·N work floor (qwenVisionMatNT, reused — it is
// the engine's generic two-tier Linear, nothing vision-specific in it) and
// runs the host f64-accumulation GEMM below it; norms, RoPE, the joint
// softmax and SiLU are host f64/f32.
//
// This file is DELIBERATELY separate from assistant_dflash*.go: those files
// carry the pre-checkpoint architecture the survey falsified (gemma4
// sandwich-norm layer, GELU gate, single shared anchor embedding,
// numAux-as-context-length, invented reduced head) and are owned by the
// assistant/MTP lane. Nothing here reuses their maths, and nothing there is
// changed by this file. The real architecture, per block:
//
//   - fused context ONCE per call: fc (Linear [hidden, numAux*hidden]) over
//     each context token's concatenated target-layer hiddens, then
//     hidden_norm (plain RMSNorm) — context length = target TOKENS fused in;
//   - NumLayers (5) plain pre-norm qwen3-style layers: input_layernorm →
//     attention → residual; post_attention_layernorm → SiLU-gated MLP →
//     residual — TWO norms, never gemma4's four;
//   - the attention IS the diffusion: q from the block's own normed rows;
//     k/v = concat(projected fused context, projected block rows) in ONE
//     non-causal softmax (every block position sees every context row and
//     every other block position); per-head q_norm, k_norm on the
//     CONCATENATED k; RoPE q at [ctxLen, ctxLen+blockLen), k over the full
//     [0, ctxLen+blockLen);
//   - final norm → [blockLen, hidden], the input the BORROWED target lm_head
//     consumes (the checkpoint carries no head/embedding of its own).
//
// Serving stays honestly declined until the glue lane re-points the proposer
// at this forward and lands a live accept-rate receipt
// (serving.DFlashEngineProbe remains false — docs/design-dflash-forward.md §7).

// DFlashZLabForward runs the drafter's block forward. noiseEmbedding is
// [blockLen*hidden] (the block's own already-embedded candidate/mask tokens,
// row-major — embedded through the TARGET's table by the caller);
// targetHiddenRaw is [ctxLen*numAux*hidden] (per context token the
// concatenated target_layer_ids hidden states — concatenation is the caller's
// job, exactly extract_context_feature's torch.cat in the reference). Returns
// the final-normed block hidden [blockLen*hidden].
//
//	final, err := native.DFlashZLabForward(m, noise, targetHidden, ctxLen, blockLen)
func DFlashZLabForward(m *zlabdflash.DraftModel, noiseEmbedding, targetHiddenRaw []float32, ctxLen, blockLen int) ([]float32, error) {
	if m == nil {
		return nil, core.NewError("native.DFlashZLabForward: nil draft model")
	}
	cfg := m.Cfg
	numAux := cfg.NumAux()
	if cfg.Hidden <= 0 || cfg.Heads <= 0 || cfg.KVHeads <= 0 || cfg.HeadDim <= 0 || cfg.NumLayers <= 0 || numAux <= 0 {
		return nil, core.NewError("native.DFlashZLabForward: incomplete draft geometry")
	}
	if cfg.Heads%cfg.KVHeads != 0 {
		return nil, core.NewError(core.Sprintf("native.DFlashZLabForward: heads %d not a multiple of kv_heads %d (GQA)", cfg.Heads, cfg.KVHeads))
	}
	if len(m.Layers) != cfg.NumLayers {
		return nil, core.NewError(core.Sprintf("native.DFlashZLabForward: payload has %d layers, config declares %d", len(m.Layers), cfg.NumLayers))
	}
	if blockLen <= 0 || ctxLen < 0 {
		return nil, core.NewError("native.DFlashZLabForward: blockLen must be >= 1 and ctxLen >= 0")
	}
	if len(noiseEmbedding) != blockLen*cfg.Hidden {
		return nil, core.NewError(core.Sprintf("native.DFlashZLabForward: noise embedding is %d floats, want %d (blockLen*hidden)", len(noiseEmbedding), blockLen*cfg.Hidden))
	}
	if len(targetHiddenRaw) != ctxLen*numAux*cfg.Hidden {
		return nil, core.NewError(core.Sprintf("native.DFlashZLabForward: target hidden is %d floats, want %d (ctxLen*numAux*hidden)", len(targetHiddenRaw), ctxLen*numAux*cfg.Hidden))
	}

	// fc + hidden_norm ONCE — every layer cross-attends this SAME fused context
	// (the reference does not re-norm it per layer, unlike the block's own
	// evolving hidden).
	var targetHidden []float32
	if ctxLen > 0 {
		targetHidden = qwenVisionMatNT(targetHiddenRaw, m.FC, ctxLen, numAux*cfg.Hidden, cfg.Hidden)
		targetHidden = dflashRMSNormRows(targetHidden, ctxLen, cfg.Hidden, m.HiddenNorm, cfg.Eps)
	}

	cos, sin := dflashRopeCosSin(ctxLen+blockLen, cfg.HeadDim, cfg.RopeTheta)

	hidden := append([]float32(nil), noiseEmbedding...)
	for li := range m.Layers {
		var err error
		hidden, err = dflashDecoderLayer(&m.Layers[li], hidden, targetHidden, cfg, blockLen, ctxLen, cos, sin)
		if err != nil {
			return nil, core.E("native.DFlashZLabForward", core.Sprintf("layer %d", li), err)
		}
	}
	return dflashRMSNormRows(hidden, blockLen, cfg.Hidden, m.FinalNorm, cfg.Eps), nil
}

// dflashDecoderLayer runs one plain pre-norm layer: input_layernorm →
// attention → residual, post_attention_layernorm → SiLU MLP → residual.
func dflashDecoderLayer(l *zlabdflash.DraftLayer, hidden, targetHidden []float32, cfg zlabdflash.Config, blockLen, ctxLen int, cos, sin []float32) ([]float32, error) {
	normed := dflashRMSNormRows(hidden, blockLen, cfg.Hidden, l.InputNorm, cfg.Eps)
	attnOut, err := dflashAttention(l, normed, targetHidden, cfg, blockLen, ctxLen, cos, sin)
	if err != nil {
		return nil, err
	}
	h1 := make([]float32, len(hidden))
	for i := range hidden {
		h1[i] = hidden[i] + attnOut[i]
	}
	normed2 := dflashRMSNormRows(h1, blockLen, cfg.Hidden, l.PostAttnNorm, cfg.Eps)
	mlpOut := dflashMLP(l, normed2, blockLen, cfg)
	out := make([]float32, len(h1))
	for i := range h1 {
		out[i] = h1[i] + mlpOut[i]
	}
	return out, nil
}

// dflashAttention is the joint cross+self attention: q from the block's own
// normed rows; k/v the concatenation of the projected fused context and the
// projected block rows, k_norm applied to the CONCATENATED k, one non-causal
// softmax per (row, head) over all ctxLen+blockLen positions. GQA broadcast,
// scale 1/sqrt(headDim); the score/softmax/weighted-sum core accumulates in
// f64 (the qwen vision attention tier).
func dflashAttention(l *zlabdflash.DraftLayer, normedHidden, targetHidden []float32, cfg zlabdflash.Config, blockLen, ctxLen int, cos, sin []float32) ([]float32, error) {
	qDim := cfg.Heads * cfg.HeadDim
	kvDim := cfg.KVHeads * cfg.HeadDim
	totalKV := ctxLen + blockLen

	q := qwenVisionMatNT(normedHidden, l.Q, blockLen, cfg.Hidden, qDim)
	q = dflashPerHeadNormRows(q, blockLen, cfg.Heads, cfg.HeadDim, l.QNorm, cfg.Eps)

	kNoise := qwenVisionMatNT(normedHidden, l.K, blockLen, cfg.Hidden, kvDim)
	vNoise := qwenVisionMatNT(normedHidden, l.V, blockLen, cfg.Hidden, kvDim)
	var k, v []float32
	if ctxLen > 0 {
		kCtx := qwenVisionMatNT(targetHidden, l.K, ctxLen, cfg.Hidden, kvDim)
		vCtx := qwenVisionMatNT(targetHidden, l.V, ctxLen, cfg.Hidden, kvDim)
		k = append(append(make([]float32, 0, totalKV*kvDim), kCtx...), kNoise...)
		v = append(append(make([]float32, 0, totalKV*kvDim), vCtx...), vNoise...)
	} else {
		k, v = kNoise, vNoise
	}
	k = dflashPerHeadNormRows(k, totalKV, cfg.KVHeads, cfg.HeadDim, l.KNorm, cfg.Eps)

	q = dflashApplyRope(q, blockLen, cfg.Heads, cfg.HeadDim, cos, sin, ctxLen)
	k = dflashApplyRope(k, totalKV, cfg.KVHeads, cfg.HeadDim, cos, sin, 0)

	group := cfg.Heads / cfg.KVHeads
	scale := 1.0 / math.Sqrt(float64(cfg.HeadDim))
	attnOut := make([]float32, blockLen*qDim)
	scores := make([]float64, totalKV)
	for r := 0; r < blockLen; r++ {
		for h := 0; h < cfg.Heads; h++ {
			kvh := h / group
			qVec := q[(r*cfg.Heads+h)*cfg.HeadDim : (r*cfg.Heads+h+1)*cfg.HeadDim]
			maxS := math.Inf(-1)
			for t := 0; t < totalKV; t++ {
				kVec := k[(t*cfg.KVHeads+kvh)*cfg.HeadDim : (t*cfg.KVHeads+kvh+1)*cfg.HeadDim]
				var dot float64
				for d := 0; d < cfg.HeadDim; d++ {
					dot += float64(qVec[d]) * float64(kVec[d])
				}
				s := dot * scale
				scores[t] = s
				if s > maxS {
					maxS = s
				}
			}
			var sum float64
			for t := 0; t < totalKV; t++ {
				e := math.Exp(scores[t] - maxS)
				scores[t] = e
				sum += e
			}
			outBase := (r*cfg.Heads + h) * cfg.HeadDim
			for d := 0; d < cfg.HeadDim; d++ {
				var acc float64
				for t := 0; t < totalKV; t++ {
					acc += scores[t] * float64(v[(t*cfg.KVHeads+kvh)*cfg.HeadDim+d])
				}
				attnOut[outBase+d] = float32(acc / sum)
			}
		}
	}
	return qwenVisionMatNT(attnOut, l.O, blockLen, qDim, cfg.Hidden), nil
}

// dflashMLP is the SwiGLU feed-forward: down(silu(gate(x)) * up(x)) — SiLU,
// never gemma's GELU gate (the survey's named activation-mismatch bug class).
func dflashMLP(l *zlabdflash.DraftLayer, x []float32, rows int, cfg zlabdflash.Config) []float32 {
	gate := qwenVisionMatNT(x, l.Gate, rows, cfg.Hidden, cfg.Intermediate)
	up := qwenVisionMatNT(x, l.Up, rows, cfg.Hidden, cfg.Intermediate)
	gated := make([]float32, len(gate))
	for i := range gate {
		gated[i] = float32(qwenVisionSilu(float64(gate[i])) * float64(up[i]))
	}
	return qwenVisionMatNT(gated, l.Down, rows, cfg.Intermediate, cfg.Hidden)
}

// dflashRMSNormRows applies the PLAIN qwen3 RMSNorm (x * rsqrt(mean(x^2)+eps) *
// weight — no gemma "+1" shift) independently to each of rows rows of [rows,
// dim], sum-of-squares in f64.
func dflashRMSNormRows(x []float32, rows, dim int, weight []float32, eps float32) []float32 {
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

// dflashPerHeadNormRows applies dflashRMSNormRows per HEAD: [rows, heads,
// headDim] flattened is byte-identical to [rows*heads, headDim], and q_norm /
// k_norm are ONE [headDim] weight shared by every head (the qwen3 convention).
func dflashPerHeadNormRows(x []float32, rows, heads, headDim int, weight []float32, eps float32) []float32 {
	return dflashRMSNormRows(x, rows*heads, headDim, weight, eps)
}

// dflashRopeCosSin computes the standard non-interleaved split-half rotary
// table for positions [0, total): half-dim inverse frequencies duplicated
// across both halves of each [headDim] row (emb = cat(freqs, freqs)).
func dflashRopeCosSin(total, headDim int, theta float32) (cos, sin []float32) {
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

// dflashApplyRope ropes rows [rowOffset, rowOffset+rows) of x ([rows, heads,
// headDim]) against the table — x's row r is absolute position rowOffset+r,
// rotate_half convention: out[d] = x1*cos - x2*sin, out[d+half] = x2*cos +
// x1*sin. q passes rowOffset=ctxLen (the block continues where the context
// left off); k passes 0 (context rows first, then the block's own).
func dflashApplyRope(x []float32, rows, heads, headDim int, cos, sin []float32, rowOffset int) []float32 {
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
