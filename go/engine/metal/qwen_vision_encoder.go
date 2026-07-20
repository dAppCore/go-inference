// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"math"

	core "dappco.re/go"
	"dappco.re/go/inference/model/arch/Qwen/qwen35"
)

// qwen_vision_encoder.go is the engine's forward for the qwen35 factory vision tower — the verbatim
// port of the retired composed engine's tower maths (model/composed/vision.go at b1f6c21a^), consuming
// the payload model/arch/Qwen/qwen35 assembles (vision_loader.go): patch embed → an optional additive
// LEARNED position embedding → N bidirectional pre-norm blocks (2-D rotary in EVERY block, additive
// with the learned table — the reference convention; optional per-head RMS QK-norm; SwiGLU or plain
// 2-linear GELU MLP) → the merger (LayerNorm → spatial merge → 2-layer GELU MLP) → text-hidden
// soft-token features. It runs ONCE per image at prefill — stateless, no KV cache.
//
// Numeric tier — the same two tiers the composed lane served with: every projection dispatches to the
// engine's steel f32 GEMM (MatMulF32NT) above the reference's 2^20 M·K·N work floor and runs the
// ported host f64-accumulation GEMM below it (or on a device error); the attention core, norms, rope
// and activations are host f64/f32 exactly as the reference computed them. The gemma4 SigLIP tower in
// vision.go is a DIFFERENT architecture (RMSNorm, no merger, pooling) — only geluTanhScalar is
// genuinely shared and reused; everything qwen-specific is ported, not adapted.

// qwenVisionDeviceMinWork is the M·K·N floor below which a projection ignores the device GEMM — a
// tiny GEMV's command-buffer round-trip outweighs its compute (the composed lane's deviceMinWork,
// ported with its value).
const qwenVisionDeviceMinWork = 1 << 20

// qwenVisionMatNT computes out[M,N] = in[M,K] @ w[N,K]ᵀ (the Linear y = x·Wᵀ): the steel f32 GEMM for
// shapes above the work floor, else the ported host reference — f64 accumulation in ascending-k
// order, the composed matNTCols tier. A device failure falls through to the host path, deterministic
// for the rest of the process either way.
func qwenVisionMatNT(in, w []float32, M, K, N int) []float32 {
	if M*K*N >= qwenVisionDeviceMinWork {
		if res, err := MatMulF32NT(in, w, M, K, N); err == nil {
			return res
		}
	}
	out := make([]float32, M*N)
	for n := range N {
		wr := w[n*K : (n+1)*K]
		for m := range M {
			xr := in[m*K : m*K+K]
			var acc float64
			for k := range K {
				acc += float64(xr[k]) * float64(wr[k])
			}
			out[m*N+n] = float32(acc)
		}
	}
	return out
}

// qwenVisionLinearForward runs one qwen35.VisionLinear over L rows of width In, adding the bias when
// present.
func qwenVisionLinearForward(x []float32, w *qwen35.VisionLinear, L int) []float32 {
	out := qwenVisionMatNT(x, w.W, L, w.In, w.Out)
	if len(w.B) > 0 {
		for t := range L {
			row := out[t*w.Out : (t+1)*w.Out]
			for i := range row {
				row[i] += w.B[i]
			}
		}
	}
	return out
}

// qwenVisionLayerNorm LayerNorm-normalises each of the `rows` rows of x [rows,d]:
// (x-mean)/√(var+eps)·w + b. The Qwen-VL vision tower's norms carry a bias term, unlike the text
// stack's RMSNorm — the reference's layerNormRowsWithBias, ported.
func qwenVisionLayerNorm(x, w, b []float32, rows, d int, eps float32) []float32 {
	out := make([]float32, rows*d)
	for r := range rows {
		xr := x[r*d : (r+1)*d]
		var mean float64
		for _, v := range xr {
			mean += float64(v)
		}
		mean /= float64(d)
		var variance float64
		for _, v := range xr {
			delta := float64(v) - mean
			variance += delta * delta
		}
		inv := 1 / math.Sqrt(variance/float64(d)+float64(eps))
		for i := range d {
			v := (float64(xr[i]) - mean) * inv * float64(w[i])
			if b != nil {
				v += float64(b[i])
			}
			out[r*d+i] = float32(v)
		}
	}
	return out
}

// qwenVisionRMSNormHead RMS-normalises a single [HeadDim] vector in place by weight w — the optional
// per-head QK-norm. No-op when w is empty (the REAL qwen layout ships no q/k norms); NOT the engine's
// rmsNormVec, which normalises even with a nil scale — a different contract.
func qwenVisionRMSNormHead(x, w []float32, eps float32) {
	if len(w) == 0 {
		return
	}
	var ss float64
	for _, e := range x {
		ss += float64(e) * float64(e)
	}
	r := math.Sqrt(ss/float64(len(x)) + float64(eps))
	for i := range x {
		x[i] = float32(float64(x[i]) / r * float64(w[i]))
	}
}

// qwenVisionSilu is the SwiGLU gate activation (the GUESSED layout's MLP), f64 as the reference
// computed it.
func qwenVisionSilu(v float64) float64 { return v / (1 + math.Exp(-v)) }

// qwenVisionRotaryTable returns the HeadDim/4 shared inverse-frequency table for the 2-D vision
// rotary embedding — Qwen-VL's VisionRotaryEmbedding(HeadDim/2, theta): inv_freq[i] =
// theta^(-2i/(HeadDim/2)), i in [0, HeadDim/4). Both spatial axes (row, col) read this SAME table at
// their own coordinate.
func qwenVisionRotaryTable(headDim int, theta float64) []float64 {
	dim := headDim / 2
	n := dim / 2
	inv := make([]float64, n)
	for i := range n {
		inv[i] = 1.0 / math.Pow(theta, float64(2*i)/float64(dim))
	}
	return inv
}

// qwenVisionRope2D rotates a [headDim] q/k vector IN PLACE for a patch at grid (row,col) — Qwen-VL's
// 2-D vision rotary position embedding: standard rotate-half pairing (index d with d+headDim/2)
// across the full vector, where the per-pair angle at pair-index d<headDim/4 is row·invFreq[d], and
// at headDim/4<=d<headDim/2 is col·invFreq[d-headDim/4] — the row coordinate drives the first quarter
// of pair-angles, the column coordinate the second, then the SAME two quarters repeat via the
// rotate-half doubling. headDim must be divisible by 4; invFreq has headDim/4 entries.
func qwenVisionRope2D(x []float32, row, col, headDim int, invFreq []float64) {
	half := headDim / 2
	quarter := half / 2
	out := make([]float32, headDim)
	for d := range half {
		var angle float64
		if d < quarter {
			angle = float64(row) * invFreq[d]
		} else {
			angle = float64(col) * invFreq[d-quarter]
		}
		c, s := float32(math.Cos(angle)), float32(math.Sin(angle))
		out[d] = x[d]*c - x[d+half]*s
		out[d+half] = x[d+half]*c + x[d]*s
	}
	copy(x, out)
}

// qwenVisionAttentionForward runs full bidirectional (non-causal) multi-head self-attention over x
// [L,Hidden] — no KV cache, no mask: every patch attends to every other patch. gridW recovers each
// row's (row,col) grid coordinate (t/gridW, t%gridW) for the 2-D rope. Standard 1/√HeadDim scaling.
// The 2-D rope applies in EVERY block, for BOTH layout families — additive with the learned position
// table, never an alternative to it (the reference convention; the composed lane's original XOR gate
// is what made the first live 27B image turn misread a giant Q as 'E').
func qwenVisionAttentionForward(x []float32, w *qwen35.VisionAttnWeights, L, gridW int, cfg qwen35.VisionTowerConfig) ([]float32, error) {
	H, KVH, HD := cfg.NumHeads, cfg.NumKVHeads, cfg.HeadDim
	if H <= 0 || KVH <= 0 || HD <= 0 || H%KVH != 0 || gridW <= 0 {
		return nil, core.NewError("native.qwenVisionAttentionForward: bad attention/grid geometry")
	}
	q := qwenVisionLinearForward(x, &w.Q, L)
	k := qwenVisionLinearForward(x, &w.K, L)
	v := qwenVisionLinearForward(x, &w.V, L)

	theta := float64(cfg.RopeTheta)
	if theta == 0 {
		theta = 10000
	}
	invFreq := qwenVisionRotaryTable(HD, theta)
	for t := range L {
		row, col := t/gridW, t%gridW
		for h := range H {
			qh := q[t*H*HD+h*HD : t*H*HD+h*HD+HD]
			qwenVisionRMSNormHead(qh, w.QNorm, cfg.Eps) // no-op when QNorm is nil
			qwenVisionRope2D(qh, row, col, HD, invFreq)
		}
		for h := range KVH {
			kh := k[t*KVH*HD+h*HD : t*KVH*HD+h*HD+HD]
			qwenVisionRMSNormHead(kh, w.KNorm, cfg.Eps)
			qwenVisionRope2D(kh, row, col, HD, invFreq)
		}
	}

	scale := 1.0 / math.Sqrt(float64(HD))
	rep := H / KVH
	out := make([]float32, L*H*HD)
	scores := make([]float64, L)
	for t := range L {
		for h := range H {
			kvh := h / rep
			qrow := q[t*H*HD+h*HD : t*H*HD+h*HD+HD]
			maxS := math.Inf(-1)
			for j := range L {
				krow := k[j*KVH*HD+kvh*HD : j*KVH*HD+kvh*HD+HD]
				var dot float64
				for d := range HD {
					dot += float64(qrow[d]) * float64(krow[d])
				}
				dot *= scale
				scores[j] = dot
				if dot > maxS {
					maxS = dot
				}
			}
			var sum float64
			for j := range L {
				scores[j] = math.Exp(scores[j] - maxS)
				sum += scores[j]
			}
			orow := out[t*H*HD+h*HD : t*H*HD+h*HD+HD]
			for d := range HD {
				var acc float64
				for j := range L {
					vrow := v[j*KVH*HD+kvh*HD : j*KVH*HD+kvh*HD+HD]
					acc += scores[j] * float64(vrow[d])
				}
				orow[d] = float32(acc / sum)
			}
		}
	}
	return qwenVisionLinearForward(out, &w.O, L), nil
}

// qwenVisionMLPForward runs the block's feed-forward over x [L,Hidden] → [L,Hidden]: SwiGLU
// (Gate/Up/Down) or plain GELU (FC1/FC2), selected by w.GELU — set at load time by which tensor names
// resolved.
func qwenVisionMLPForward(x []float32, w *qwen35.VisionMLPWeights, L int) []float32 {
	if w.GELU {
		h := qwenVisionLinearForward(x, &w.FC1, L)
		for i := range h {
			h[i] = geluTanhScalar(h[i])
		}
		return qwenVisionLinearForward(h, &w.FC2, L)
	}
	g := qwenVisionLinearForward(x, &w.Gate, L)
	u := qwenVisionLinearForward(x, &w.Up, L)
	h := make([]float32, len(g))
	for i := range h {
		h[i] = float32(qwenVisionSilu(float64(g[i])) * float64(u[i]))
	}
	return qwenVisionLinearForward(h, &w.Down, L)
}

// qwenVisionAddRows returns a+b elementwise — the residual add, kept as a fresh slice since a block's
// input x is the PREVIOUS block's return value, not caller-owned scratch.
func qwenVisionAddRows(a, b []float32) []float32 {
	out := make([]float32, len(a))
	for i := range a {
		out[i] = a[i] + b[i]
	}
	return out
}

// qwenVisionBlockForward runs one pre-norm encoder block over x [L,Hidden], returning the new
// [L,Hidden] hidden.
func qwenVisionBlockForward(b *qwen35.VisionBlock, x []float32, L, gridW int, cfg qwen35.VisionTowerConfig) ([]float32, error) {
	normed := qwenVisionLayerNorm(x, b.Norm1W, b.Norm1B, L, cfg.Hidden, cfg.Eps)
	attnOut, err := qwenVisionAttentionForward(normed, &b.Attn, L, gridW, cfg)
	if err != nil {
		return nil, err
	}
	h := qwenVisionAddRows(x, attnOut)
	normed2 := qwenVisionLayerNorm(h, b.Norm2W, b.Norm2B, L, cfg.Hidden, cfg.Eps)
	mlpOut := qwenVisionMLPForward(normed2, &b.MLP, L)
	return qwenVisionAddRows(h, mlpOut), nil
}

// qwenVisionMergeSpatial gathers x [gridH*gridW,hidden] (row-major raster order) into
// [(gridH/M)*(gridW/M), hidden*M*M]: each output row concatenates one M×M spatial block of input
// rows, block-row-major within the block — the Qwen-VL merger's "M² adjacent patches become one row"
// reshape. gridH and gridW must already be M-divisible (the caller's job).
func qwenVisionMergeSpatial(x []float32, gridH, gridW, hidden, m int) []float32 {
	outRows, outCols := (gridH/m)*(gridW/m), hidden*m*m
	out := make([]float32, outRows*outCols)
	idx := 0
	for by := 0; by < gridH; by += m {
		for bx := 0; bx < gridW; bx += m {
			dst := out[idx*outCols : (idx+1)*outCols]
			col := 0
			for dy := range m {
				for dx := range m {
					src := x[((by+dy)*gridW+(bx+dx))*hidden : ((by+dy)*gridW+(bx+dx))*hidden+hidden]
					copy(dst[col:col+hidden], src)
					col += hidden
				}
			}
			idx++
		}
	}
	return out
}

// qwenVisionMergerForward runs the merger over the encoder's final hidden x [gridH*gridW, Hidden]:
// per-row LayerNorm, then the spatial merge, then Linear1 → GELU → Linear2 into text hidden width.
// Returns the projected features [softTokens, TextHidden] and softTokens =
// (gridH/mergeSize)·(gridW/mergeSize).
func qwenVisionMergerForward(m *qwen35.VisionMerger, x []float32, gridH, gridW int, cfg qwen35.VisionTowerConfig) (features []float32, softTokens int, err error) {
	M := cfg.MergeSize
	if M <= 0 {
		M = 1
	}
	if gridH%M != 0 || gridW%M != 0 {
		return nil, 0, core.NewError(core.Sprintf("native.qwenVisionMergerForward: grid %dx%d not divisible by merge size %d", gridH, gridW, M))
	}
	normed := qwenVisionLayerNorm(x, m.NormW, m.NormB, gridH*gridW, cfg.Hidden, cfg.Eps)
	merged := qwenVisionMergeSpatial(normed, gridH, gridW, cfg.Hidden, M)
	outRows := (gridH / M) * (gridW / M)
	h1 := qwenVisionLinearForward(merged, &m.L1, outRows)
	for i := range h1 {
		h1[i] = geluTanhScalar(h1[i])
	}
	features = qwenVisionLinearForward(h1, &m.L2, outRows)
	return features, outRows, nil
}

// QwenVisionTowerForward runs the whole Qwen-VL-family vision forward on a pre-patchified grid
// (qwen35.ImageToPatchGrid's output): patch embed → an optional additive LEARNED position embedding
// (bilinearly resampled onto a non-native grid) → N bidirectional blocks → the merger, returning the
// projected soft-token features [softTokens,TextHidden] and softTokens.
func QwenVisionTowerForward(patches []float32, gridH, gridW int, tower *qwen35.VisionTower) (features []float32, softTokens int, err error) {
	cfg := tower.Cfg
	L := gridH * gridW
	if L <= 0 {
		return nil, 0, core.NewError("native.QwenVisionTowerForward: empty patch grid")
	}
	if len(patches) != L*cfg.PatchDim {
		return nil, 0, core.NewError(core.Sprintf("native.QwenVisionTowerForward: patch buffer len %d != L·PatchDim %d", len(patches), L*cfg.PatchDim))
	}
	h := qwenVisionLinearForward(patches, &tower.Patch, L)
	if len(tower.PosEmbed) > 0 {
		// The REAL layout's learned position table is a FIXED [NumPositions,Hidden] lookup trained on
		// one square grid. A grid-exact request adds the table directly; any other grid resamples it
		// bilinearly onto (gridH × gridW) — the reference's align-corners convention — so arbitrary
		// image sizes ride the same trained table.
		pos := tower.PosEmbed
		if len(pos) != L*cfg.Hidden {
			var perr error
			pos, perr = qwen35.InterpolatePosEmbed(tower.PosEmbed, cfg.Hidden, gridH, gridW)
			if perr != nil {
				return nil, 0, core.E("native.QwenVisionTowerForward", "pos_embed interpolation", perr)
			}
		}
		h = qwenVisionAddRows(h, pos)
	}
	for i := range tower.Blocks {
		if h, err = qwenVisionBlockForward(&tower.Blocks[i], h, L, gridW, cfg); err != nil {
			return nil, 0, core.E("native.QwenVisionTowerForward", core.Sprintf("block %d", i), err)
		}
	}
	return qwenVisionMergerForward(&tower.Merger, h, gridH, gridW, cfg)
}
