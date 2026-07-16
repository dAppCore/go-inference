// SPDX-Licence-Identifier: EUPL-1.2

package composed

import (
	"bytes"
	"image"
	_ "image/jpeg" // register the JPEG decoder with image.Decode
	_ "image/png"  // register the PNG decoder with image.Decode
	"math"

	core "dappco.re/go"
	"dappco.re/go/inference/model"
)

// vision.go is the composed model's own side of the repo's ONE multimodal route (go/engine/vision.go): a
// host Qwen-VL-family vision tower + merger/projector, run once per image at prefill (no recurrent/KV state
// of its own — a ComposedModel.Vision forward is stateless, mirroring how the metal engine's VisionTower
// runs once per image ahead of the token decode loop). Geometry lives on visionTowerCfg, assembled by
// vision_loader.go from the checkpoint's own tensor shapes; this file only runs the maths: patch embed →
// (an optional additive LEARNED position embedding — visionTowerCfg.LearnedPositions) → N bidirectional
// transformer blocks (2-D rotary position embedding XOR the learned table, never both; optional per-head
// QK-norm; SwiGLU MLP or a plain 2-linear GELU MLP — visionMLPWeights.GELU) → the merger (LayerNorm →
// spatial mergeSize×mergeSize concat → 2-layer GELU MLP) → text-hidden soft-token features. Every
// projection runs host f32 OR, on a quantised checkpoint, the SAME packed model.QuantWeight/matNTQuant seam
// the composed text stack's projections use (visionLinear.WQ; see vision_loader.go's visionProj) — never
// widened blind. No engine/model-root import: composed stays a leaf the engine depends on, never the
// reverse (AX-8).

// The Qwen-VL family's stable special-token spellings that wrap an image's soft-token run in the rendered
// prompt text. config.json carries only the numeric id (parsed as loaderConfig.ImageTokenID); the literal
// spellings are a family constant hardcoded here, exactly as register.go's chatMLChatTemplate hardcodes
// the ChatML turn markers ("<|im_start|>"/"<|im_end|>") for the same family.
const (
	qwenVisionBeginToken = "<|vision_start|>"
	qwenVisionToken      = "<|image_pad|>"
	qwenVisionEndToken   = "<|vision_end|>"
)

// visionLinear is one dense projection: out[L,Out] = in[L,In]·Wᵀ (+ bias). W is row-major [Out,In]; B is
// nil when the checkpoint carries no bias for this projection. WQ is the SAME projection kept PACKED — an
// mlx-affine quantised weight, model.QuantWeight, the exact type the composed text stack's AttnWeights/MLP
// carry (attention.go/composed.go) — when the checkpoint quantises it; exactly one of W/WQ is set (never
// both: vision_loader.go's visionProj resolves either the packed or the dense form per tensor, mirroring
// loader.go's own proj closure), and linearForward dispatches on WQ's presence.
type visionLinear struct {
	W       []float32
	WQ      *model.QuantWeight
	B       []float32
	Out, In int
}

// linearForward runs one visionLinear over L rows of width In, adding the bias when present. A packed
// weight (WQ set) dispatches to matNTQuant — the SAME quant matvec seam the text stack's projections use
// (composed.go) — instead of ever widening a quantised vision weight to f32.
func linearForward(x []float32, w *visionLinear, L int) []float32 {
	var out []float32
	if w.WQ != nil {
		out = matNTQuant(nil, x, w.WQ, L, w.In, w.Out)
	} else {
		out = matNT(x, w.W, L, w.In, w.Out)
	}
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

// visionAttnWeights is one block's bidirectional self-attention weights. QNorm/KNorm are the optional
// per-head RMSNorm scale [HeadDim] (nil ⇒ no QK-norm — plain projected q/k, matching Qwen2-VL's original
// vision tower; a later Qwen3-VL-family checkpoint that DOES ship q_norm/k_norm gets it for free since the
// loader probes for the tensors rather than assuming their presence).
type visionAttnWeights struct {
	Q, K, V, O   visionLinear
	QNorm, KNorm []float32
}

// visionMLPWeights is one block's feed-forward, one of TWO shapes selected by GELU (set by the loader from
// which tensor names resolved — vision_loader.go's loadBlockMLP — never both populated):
//   - SwiGLU (GELU false, the zero value): out = (SiLU(x·Gateᵀ)⊙x·Upᵀ)·Downᵀ — Gate/Up/Down set, the same
//     shape as the text stack's MLP, reusing the package's own silu() helper. The "guessed" layout's MLP.
//   - plain 2-linear GELU (GELU true): out = GELU(x·FC1ᵀ)·FC2ᵀ — FC1/FC2 set, activation geluTanh (this
//     file's own tanh-approx GELU, already used by the merger below) — the REAL Qwen-VL-family layout's
//     mlp.linear_fc1/linear_fc2 shape (vision_loader.go), config hidden_act "gelu_pytorch_tanh".
type visionMLPWeights struct {
	Gate, Up, Down visionLinear
	FC1, FC2       visionLinear
	GELU           bool
}

// visionBlock is one pre-norm encoder block: LayerNorm → bidirectional attention → residual, LayerNorm →
// SwiGLU MLP → residual. Qwen-VL vision blocks use full LayerNorm (mean-centred, weight AND bias), unlike
// the text stack's plain RMSNorm — layerNormRowsWithBias below is this file's own helper for that shape.
type visionBlock struct {
	Norm1W, Norm1B []float32
	Attn           visionAttnWeights
	Norm2W, Norm2B []float32
	MLP            visionMLPWeights
}

// visionMerger is the vision-to-text projector: LayerNorm (per pre-merge patch row) → spatial
// mergeSize×mergeSize concatenation → Linear1 → GELU(tanh approx) → Linear2 into text hidden width.
// Mirrors HF's Qwen2VLPatchMerger (ln_q, mlp.0, mlp.2) by SHAPE, not by borrowed code.
type visionMerger struct {
	NormW, NormB []float32
	L1, L2       visionLinear
}

// visionTowerCfg is the derived, engine-neutral tower geometry — every field except PatchSize/InChannels/
// RopeTheta/Eps is DERIVED from the checkpoint's tensor shapes by vision_loader.go (see its doc comment);
// nothing here is a magic number invented for this package.
type visionTowerCfg struct {
	Hidden, PatchDim              int
	NumHeads, NumKVHeads, HeadDim int
	PatchSize, InChannels         int
	TemporalPatchSize             int
	MergeSize                     int
	TextHidden                    int
	RopeTheta                     float32
	Eps                           float32
	// LearnedPositions marks a tower whose positions come from an ADDITIVE table added once after the patch
	// embed (visionTower.PosEmbed — the REAL Qwen-VL-family layout's vision_tower.pos_embed.weight; see
	// vision_loader.go) rather than the 2-D rotary embedding visionAttentionForward otherwise applies per
	// block (the "guessed" layout's convention). The two are mutually exclusive per checkpoint family.
	// False is the zero value, so every visionTowerCfg literal that predates this field (every test in
	// vision_test.go, and any future direct construction) keeps running 2-D RoPE exactly as before — only
	// the loader's real-layout path (driven by pos_embed's presence) sets this true.
	LearnedPositions bool
}

// visionTower is the whole loaded tower + merger for one checkpoint.
type visionTower struct {
	Patch visionLinear
	// PosEmbed is the learned absolute position-embedding table [NumPositions,Hidden] (flattened row-major),
	// added once to the patch-embed output before the first block — see visionTowerCfg.LearnedPositions and
	// visionTowerForward. nil for the 2-D-RoPE "guessed" layout.
	PosEmbed []float32
	Blocks   []visionBlock
	Merger   visionMerger
	Cfg      visionTowerCfg
}

// layerNormRowsWithBias LayerNorm-normalises each of the `rows` rows of x [rows,d]: (x-mean)/√(var+eps)·w
// + b. Unlike the package's rmsNormRowsPlain/layerNormRowsPlain (both weight-only, the text stack's
// shape), the Qwen-VL vision tower's norms carry a bias term too — this is that shape, kept local to
// vision.go since no text mixer needs it.
func layerNormRowsWithBias(x, w, b []float32, rows, d int, eps float32) []float32 {
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

// geluTanh is the tanh approximation of GELU (the merger MLP's activation between Linear1 and Linear2):
// 0.5x(1+tanh(√(2/π)·(x+0.044715x³))) — the standard formula, not a fitted/invented constant.
func geluTanh(x float32) float32 {
	const c = 0.7978845608028654 // √(2/π)
	return 0.5 * x * (1 + float32(math.Tanh(float64(c*(x+0.044715*x*x*x)))))
}

// visionRotaryTable returns the HeadDim/4 shared inverse-frequency table for the 2-D vision rotary
// embedding — Qwen-VL's VisionRotaryEmbedding(HeadDim/2, theta): inv_freq[i] = theta^(-2i/(HeadDim/2)), i
// in [0, HeadDim/4). Both spatial axes (row, col) read this SAME table at their own coordinate.
func visionRotaryTable(headDim int, theta float64) []float64 {
	dim := headDim / 2
	n := dim / 2
	inv := make([]float64, n)
	for i := range n {
		inv[i] = 1.0 / math.Pow(theta, float64(2*i)/float64(dim))
	}
	return inv
}

// visionRope2D rotates a [headDim] q/k vector IN PLACE for a patch at grid (row,col) — Qwen-VL's 2-D
// vision rotary position embedding: standard rotate-half pairing (index d with d+headDim/2) across the
// full vector, where the per-pair angle at pair-index d<headDim/4 is row·invFreq[d], and at
// headDim/4<=d<headDim/2 is col·invFreq[d-headDim/4] — i.e. the row coordinate drives the first quarter of
// pair-angles, the column coordinate the second quarter, then the SAME two quarters repeat via the
// rotate-half doubling (mirrors apply_rotary_pos_emb_vision over cat(rope_row,rope_col) doubled to
// headDim). headDim must be divisible by 4 (both axes need an even quarter); invFreq has headDim/4 entries
// (visionRotaryTable's output).
func visionRope2D(x []float32, row, col, headDim int, invFreq []float64) {
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

// visionAttentionForward runs full bidirectional (non-causal) multi-head self-attention over x [L,Hidden]
// — no KV cache, no mask: every patch attends to every other patch, the vision-tower convention (unlike
// the text stack's causal attnMixer). gridW recovers each row's (row,col) grid coordinate (t/gridW,
// t%gridW) for the 2-D rope. Standard 1/√HeadDim scaling (Qwen-VL vision attention is not QK-normed the
// way gemma4's SigLIP tower hardcodes scale=1 — this stack derives the scale rather than borrowing that
// quirk).
func visionAttentionForward(x []float32, w *visionAttnWeights, L, gridW int, cfg visionTowerCfg) ([]float32, error) {
	H, KVH, HD := cfg.NumHeads, cfg.NumKVHeads, cfg.HeadDim
	if H <= 0 || KVH <= 0 || HD <= 0 || H%KVH != 0 || gridW <= 0 {
		return nil, core.NewError("composed.visionAttentionForward: bad attention/grid geometry")
	}
	q := linearForward(x, &w.Q, L)
	k := linearForward(x, &w.K, L)
	v := linearForward(x, &w.V, L)

	// The 2-D rope applies in EVERY block, for BOTH layout families: the reference
	// (Qwen3_5VisionModel.forward) adds the learned table once after the patch embed AND
	// threads rotary cos/sin into every attention — the two position mechanisms are
	// additive, not alternatives. (The pre-fix XOR gate here is what made the first live
	// 27B image turn read 'E' for a giant Q: the real layout ran attention with no rotary
	// signal at all.) QNorm/KNorm application stays unconditional (rmsNormHead no-ops on
	// nil).
	theta := float64(cfg.RopeTheta)
	if theta == 0 {
		theta = 10000
	}
	invFreq := visionRotaryTable(HD, theta)
	for t := range L {
		row, col := t/gridW, t%gridW
		for h := range H {
			qh := q[t*H*HD+h*HD : t*H*HD+h*HD+HD]
			rmsNormHead(qh, w.QNorm, cfg.Eps) // no-op when QNorm is nil (rmsNormHead's own guard)
			visionRope2D(qh, row, col, HD, invFreq)
		}
		for h := range KVH {
			kh := k[t*KVH*HD+h*HD : t*KVH*HD+h*HD+HD]
			rmsNormHead(kh, w.KNorm, cfg.Eps)
			visionRope2D(kh, row, col, HD, invFreq)
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
	return linearForward(out, &w.O, L), nil
}

// visionMLPForward runs the block's feed-forward over x [L,Hidden] → [L,Hidden]: SwiGLU (Gate/Up/Down) or
// plain GELU (FC1/FC2), selected by w.GELU — set at load time by which tensor names resolved (see
// vision_loader.go's loadBlockMLP).
func visionMLPForward(x []float32, w *visionMLPWeights, L int) []float32 {
	if w.GELU {
		h := linearForward(x, &w.FC1, L)
		for i := range h {
			h[i] = geluTanh(h[i])
		}
		return linearForward(h, &w.FC2, L)
	}
	g := linearForward(x, &w.Gate, L)
	u := linearForward(x, &w.Up, L)
	h := make([]float32, len(g))
	for i := range h {
		h[i] = float32(silu(float64(g[i])) * float64(u[i]))
	}
	return linearForward(h, &w.Down, L)
}

// addRows returns a+b elementwise — the residual add, kept as a fresh slice (unlike the text stack's
// in-place forwardEmb accumulation) since a block's input x is the PREVIOUS block's return value, not
// caller-owned scratch.
func addRows(a, b []float32) []float32 {
	out := make([]float32, len(a))
	for i := range a {
		out[i] = a[i] + b[i]
	}
	return out
}

// forward runs one pre-norm encoder block over x [L,Hidden], returning the new [L,Hidden] hidden.
func (b *visionBlock) forward(x []float32, L, gridW int, cfg visionTowerCfg) ([]float32, error) {
	normed := layerNormRowsWithBias(x, b.Norm1W, b.Norm1B, L, cfg.Hidden, cfg.Eps)
	attnOut, err := visionAttentionForward(normed, &b.Attn, L, gridW, cfg)
	if err != nil {
		return nil, err
	}
	h := addRows(x, attnOut)
	normed2 := layerNormRowsWithBias(h, b.Norm2W, b.Norm2B, L, cfg.Hidden, cfg.Eps)
	mlpOut := visionMLPForward(normed2, &b.MLP, L)
	return addRows(h, mlpOut), nil
}

// mergeSpatialBlocks gathers x [gridH*gridW,hidden] (row-major raster order) into
// [(gridH/M)*(gridW/M), hidden*M*M]: each output row concatenates one M×M spatial block of input rows,
// block-row-major within the block — (0,0),(0,1),…,(0,M-1),(1,0),… — the Qwen-VL merger's "M² adjacent
// patches become one row" reshape. gridH and gridW must already be M-divisible (the caller's job — see
// visionMerger.forward and imageToPatchGrid's crop-to-multiple).
func mergeSpatialBlocks(x []float32, gridH, gridW, hidden, m int) []float32 {
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

// forward runs the merger over the encoder's final hidden x [gridH*gridW, Hidden]: per-row LayerNorm, then
// mergeSpatialBlocks, then Linear1 → GELU → Linear2 into text hidden width. Returns the projected features
// [softTokens, TextHidden] and softTokens = (gridH/mergeSize)·(gridW/mergeSize).
func (m *visionMerger) forward(x []float32, gridH, gridW int, cfg visionTowerCfg) (features []float32, softTokens int, err error) {
	M := cfg.MergeSize
	if M <= 0 {
		M = 1
	}
	if gridH%M != 0 || gridW%M != 0 {
		return nil, 0, core.NewError(core.Sprintf("composed.visionMerger.forward: grid %dx%d not divisible by merge size %d", gridH, gridW, M))
	}
	normed := layerNormRowsWithBias(x, m.NormW, m.NormB, gridH*gridW, cfg.Hidden, cfg.Eps)
	merged := mergeSpatialBlocks(normed, gridH, gridW, cfg.Hidden, M)
	outRows := (gridH / M) * (gridW / M)
	h1 := linearForward(merged, &m.L1, outRows)
	for i := range h1 {
		h1[i] = geluTanh(h1[i])
	}
	features = linearForward(h1, &m.L2, outRows)
	return features, outRows, nil
}

// visionTowerForward runs the whole Qwen-VL-family vision forward on a pre-patchified grid: patch embed →
// an optional additive LEARNED position embedding → N bidirectional blocks → the merger, returning the
// projected soft-token features [softTokens,TextHidden] and softTokens.
func visionTowerForward(patches []float32, gridH, gridW int, tower *visionTower) (features []float32, softTokens int, err error) {
	cfg := tower.Cfg
	L := gridH * gridW
	if L <= 0 {
		return nil, 0, core.NewError("composed.visionTowerForward: empty patch grid")
	}
	if len(patches) != L*cfg.PatchDim {
		return nil, 0, core.NewError(core.Sprintf("composed.visionTowerForward: patch buffer len %d != L·PatchDim %d", len(patches), L*cfg.PatchDim))
	}
	h := linearForward(patches, &tower.Patch, L)
	if len(tower.PosEmbed) > 0 {
		// The REAL layout's learned position table is a FIXED [NumPositions,Hidden] lookup trained on one
		// square grid. A grid-exact request adds the table directly; any other grid resamples it
		// bilinearly onto (gridH × gridW) — the reference towers' F.interpolate(align_corners=false)
		// behaviour — so arbitrary image sizes ride the same trained table.
		pos := tower.PosEmbed
		if len(pos) != L*cfg.Hidden {
			var perr error
			pos, perr = visionInterpolatePosEmbed(tower.PosEmbed, cfg.Hidden, gridH, gridW)
			if perr != nil {
				return nil, 0, core.E("composed.visionTowerForward", "pos_embed interpolation", perr)
			}
		}
		h = addRows(h, pos)
	}
	for i := range tower.Blocks {
		if h, err = tower.Blocks[i].forward(h, L, gridW, cfg); err != nil {
			return nil, 0, core.E("composed.visionTowerForward", core.Sprintf("block %d", i), err)
		}
	}
	return tower.Merger.forward(h, gridH, gridW, cfg)
}

// imageToPatchGrid decodes raw PNG/JPEG image bytes and slices it into a non-overlapping grid of
// PatchSize×PatchSize patches, returning the flattened per-patch pixel rows [gridH*gridW,PatchDim]
// (row-major raster order; each patch row is [PatchSize·PatchSize·InChannels] channel-last HWC, repeated
// TemporalPatchSize times — the Qwen-VL convention of feeding a still image as TemporalPatchSize identical
// frames into what a checkpoint's Conv3D patch embed reduces to for a non-overlapping kernel) ready for
// visionTowerForward. The image is TOP-LEFT CROPPED — not resampled — down to the nearest multiple of
// PatchSize·MergeSize on each axis, so every patch is exact source pixels and the grid always merges
// cleanly; a production resize/pad POLICY (aspect-preserving budget, bicubic resample, …) is a
// serving-layer preprocessing concern layered on top of this, not this function's job — mirrored, at the
// metal engine, by VisionImagePixels living in its own file separate from the tower forward.
func imageToPatchGrid(data []byte, cfg visionTowerCfg) (patches []float32, gridH, gridW int, err error) {
	if cfg.PatchSize <= 0 {
		return nil, 0, 0, core.NewError("composed.imageToPatchGrid: PatchSize must be positive")
	}
	img, _, derr := image.Decode(bytes.NewReader(data))
	if derr != nil {
		return nil, 0, 0, core.E("composed.imageToPatchGrid", "decode image", derr)
	}
	bounds := img.Bounds()
	h, w := bounds.Dy(), bounds.Dx()
	unit := cfg.PatchSize * max(cfg.MergeSize, 1)
	th, tw := (h/unit)*unit, (w/unit)*unit
	if th <= 0 || tw <= 0 {
		return nil, 0, 0, core.NewError(core.Sprintf("composed.imageToPatchGrid: %dx%d image smaller than one %d-pixel patch·merge block", w, h, unit))
	}
	gridH, gridW = th/cfg.PatchSize, tw/cfg.PatchSize

	channels := cfg.InChannels
	if channels <= 0 {
		channels = 3
	}
	frames := cfg.TemporalPatchSize
	if frames <= 0 {
		frames = 1
	}
	patchPixels := channels * cfg.PatchSize * cfg.PatchSize
	patches = make([]float32, gridH*gridW*patchPixels*frames)
	idx := 0
	for gy := range gridH {
		for gx := range gridW {
			base := idx * patchPixels * frames
			col := 0
			for py := range cfg.PatchSize {
				y := bounds.Min.Y + gy*cfg.PatchSize + py
				for px := range cfg.PatchSize {
					x := bounds.Min.X + gx*cfg.PatchSize + px
					r, g, b, _ := img.At(x, y).RGBA() // 16-bit, premultiplied per image.Color's contract
					patches[base+col] = float32(r>>8) / 255
					if channels > 1 {
						patches[base+col+1] = float32(g>>8) / 255
					}
					if channels > 2 {
						patches[base+col+2] = float32(b>>8) / 255
					}
					col += channels
				}
			}
			for f := 1; f < frames; f++ {
				copy(patches[base+f*patchPixels:base+(f+1)*patchPixels], patches[base:base+patchPixels])
			}
			idx++
		}
	}
	return patches, gridH, gridW, nil
}

// visionInterpolatePosEmbed resamples a learned square [side²,hidden] position
// table onto a (dstH × dstW) patch grid with separable bilinear interpolation
// in the REFERENCE convention (vision_utils.get_vision_bilinear_indices_and_
// weights): per-axis coordinates are linspace(0, side-1, n) — align_corners
// style, both endpoints landing exactly on the table's edge rows — with the
// ceil corner clamped to side-1. A single-point axis (n == 1) sits at 0. The
// source table must be square (side = √positions), the real layout's
// convention; anything else fails loud.
func visionInterpolatePosEmbed(table []float32, hidden, dstH, dstW int) ([]float32, error) {
	if hidden <= 0 || dstH <= 0 || dstW <= 0 {
		return nil, core.NewError("composed.visionInterpolatePosEmbed: hidden and grid dims must be positive")
	}
	if len(table)%hidden != 0 {
		return nil, core.NewError("composed.visionInterpolatePosEmbed: table length is not a multiple of hidden")
	}
	positions := len(table) / hidden
	side := isqrt(positions)
	if side <= 0 || side*side != positions {
		return nil, core.NewError(core.Sprintf("composed.visionInterpolatePosEmbed: %d positions is not a square grid", positions))
	}
	out := make([]float32, dstH*dstW*hidden)
	axis := func(n, i int) float64 { // linspace(0, side-1, n)[i]
		if n <= 1 {
			return 0
		}
		return float64(i) * float64(side-1) / float64(n-1)
	}
	clamp := func(v int) int {
		if v >= side {
			return side - 1
		}
		return v
	}
	for y := range dstH {
		fy := axis(dstH, y)
		y0 := int(math.Floor(fy))
		ty := float32(fy - float64(y0))
		y0c, y1c := y0, clamp(y0+1)
		for x := range dstW {
			fx := axis(dstW, x)
			x0 := int(math.Floor(fx))
			tx := float32(fx - float64(x0))
			x0c, x1c := x0, clamp(x0+1)
			r00 := table[(y0c*side+x0c)*hidden:]
			r01 := table[(y0c*side+x1c)*hidden:]
			r10 := table[(y1c*side+x0c)*hidden:]
			r11 := table[(y1c*side+x1c)*hidden:]
			dst := out[(y*dstW+x)*hidden:]
			for h := range hidden {
				top := r00[h] + (r01[h]-r00[h])*tx
				bot := r10[h] + (r11[h]-r10[h])*tx
				dst[h] = top + (bot-top)*ty
			}
		}
	}
	return out, nil
}
