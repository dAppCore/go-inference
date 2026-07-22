// SPDX-Licence-Identifier: EUPL-1.2

package qwen35

import (
	"bytes"
	"image"
	_ "image/jpeg" // register the JPEG decoder with image.Decode
	_ "image/png"  // register the PNG decoder with image.Decode
	"math"

	core "dappco.re/go"
)

// vision.go is the qwen35 family's vision-tower PAYLOAD: the loaded weight set + derived geometry a
// backend runs an image through (engine/metal's qwen_vision_encoder.go carries the forward), plus the
// two pieces of pure host preprocessing that belong with the payload — the patchifier
// (ImageToPatchGrid) and the learned-position resampler (InterpolatePosEmbed). It is the
// factory-native port of the retired composed engine's qwen tower (model/composed/vision.go at
// b1f6c21a^): a host Qwen-VL-family vision tower + merger, run once per image at prefill — stateless,
// no recurrent/KV state of its own. The maths ported to the engine: patch embed → an optional additive
// LEARNED position embedding (VisionTowerConfig.LearnedPositions) → N bidirectional pre-norm blocks
// (2-D rotary position embedding in EVERY block — additive with the learned table, the reference
// convention; optional per-head RMS QK-norm; SwiGLU MLP or a plain 2-linear GELU MLP —
// VisionMLPWeights.GELU) → the merger (LayerNorm → spatial MergeSize×MergeSize concat → 2-layer GELU
// MLP) → text-hidden soft-token features. AX-8: this package never imports engine; the engine imports
// these types and satisfies the vision serve contracts by shape.

// The Qwen-VL family's stable special-token spellings that wrap an image's soft-token run in the
// rendered prompt text. config.json carries only the numeric ids (Config.ImageTokenID and friends);
// the literal spellings are a family constant hardcoded here, exactly as the engine's ChatML template
// hardcodes the <|im_start|>/<|im_end|> turn markers for the same family.
const (
	VisionBeginToken = "<|vision_start|>"
	VisionPadToken   = "<|image_pad|>"
	VisionEndToken   = "<|vision_end|>"
)

// VisionLinear is one dense projection: out[L,Out] = in[L,In]·Wᵀ (+ bias). W is row-major [Out,In]
// host f32; B is nil when the checkpoint carries no bias for this projection. The payload is ALWAYS
// f32: dense checkpoint tensors widen at load, and a packed (mlx-affine) projection dequantises at
// load through mlxaffine.DequantizeTensor — the SAME primitive the retired composed tower's host quant
// matvec called per output row, so the values the forward dots against are identical to that
// reference's host path (see vision_loader.go's visionProj).
type VisionLinear struct {
	W       []float32
	B       []float32
	Out, In int
}

// VisionAttnWeights is one block's bidirectional self-attention weights. QNorm/KNorm are the optional
// per-head RMSNorm scale [HeadDim] (nil ⇒ no QK-norm — the REAL Qwen-VL layout ships none; the loader
// probes for the tensors rather than assuming their presence).
type VisionAttnWeights struct {
	Q, K, V, O   VisionLinear
	QNorm, KNorm []float32
}

// VisionMLPWeights is one block's feed-forward, one of TWO shapes selected by GELU (set by the loader
// from which tensor names resolved — never both populated):
//   - SwiGLU (GELU false, the zero value): out = (SiLU(x·Gateᵀ)⊙x·Upᵀ)·Downᵀ — Gate/Up/Down set (the
//     "guessed" layout's MLP).
//   - plain 2-linear GELU (GELU true): out = GELU(x·FC1ᵀ)·FC2ᵀ — FC1/FC2 set, activation
//     gelu_pytorch_tanh — the REAL Qwen-VL-family layout's mlp.linear_fc1/linear_fc2 shape.
type VisionMLPWeights struct {
	Gate, Up, Down VisionLinear
	FC1, FC2       VisionLinear
	GELU           bool
}

// VisionBlock is one pre-norm encoder block: LayerNorm → bidirectional attention → residual,
// LayerNorm → MLP → residual. Qwen-VL vision blocks use full LayerNorm (mean-centred, weight AND
// bias), unlike the text stack's plain RMSNorm.
type VisionBlock struct {
	Norm1W, Norm1B []float32
	Attn           VisionAttnWeights
	Norm2W, Norm2B []float32
	MLP            VisionMLPWeights
}

// VisionMerger is the vision-to-text projector: LayerNorm (per pre-merge patch row) → spatial
// MergeSize×MergeSize concatenation → Linear1 → GELU(tanh approx) → Linear2 into text hidden width.
// Mirrors HF's Qwen2VLPatchMerger (ln_q, mlp.0, mlp.2) by SHAPE, not by borrowed code.
type VisionMerger struct {
	NormW, NormB []float32
	L1, L2       VisionLinear
}

// VisionTowerConfig is the derived, engine-neutral tower geometry — every field except
// PatchSize/InChannels/RopeTheta/Eps and the token ids is DERIVED from the checkpoint's tensor shapes
// by vision_loader.go; nothing here is a magic number invented for this package.
type VisionTowerConfig struct {
	Hidden, PatchDim              int
	NumHeads, NumKVHeads, HeadDim int
	PatchSize, InChannels         int
	TemporalPatchSize             int
	MergeSize                     int
	TextHidden                    int
	RopeTheta                     float32
	Eps                           float32
	// LearnedPositions marks a tower whose positions include an ADDITIVE table added once after the
	// patch embed (VisionTower.PosEmbed — the REAL layout's vision_tower.pos_embed.weight) alongside
	// the per-block 2-D rotary embedding the forward always applies — the two mechanisms are
	// additive, not alternatives (the reference convention; the composed lane's original XOR gate is
	// the bug that made the first live 27B image turn misread).
	LearnedPositions bool
	// ImageTokenID is the vocabulary id one image soft-token occupies (config.json's top-level
	// image_token_id); 0 when the config omits it — the splice then fails loud rather than landing
	// features on the wrong rows.
	ImageTokenID int32
}

// VisionTower is the whole loaded tower + merger for one checkpoint — the payload
// engine/metal's QwenVisionTowerForward consumes.
type VisionTower struct {
	Patch VisionLinear
	// PosEmbed is the learned absolute position-embedding table [NumPositions,Hidden] (flattened
	// row-major), added once to the patch-embed output before the first block; nil for the 2-D-RoPE
	// "guessed" layout.
	PosEmbed []float32
	Blocks   []VisionBlock
	Merger   VisionMerger
	Cfg      VisionTowerConfig
	// Preprocess is the checkpoint's declared preprocessor_config.json policy (normalisation +
	// smart_resize bounds) — set by the engine load seam via LoadVisionPreprocessConfig, NOT by
	// LoadVisionTower (which only reads config.json + tensors; preprocessor_config.json is a
	// separate file with separate provenance, see vision_preprocess.go). The zero value degrades
	// safely to the pre-#59 behaviour (VisionPreprocessConfig.normalized()'s doc comment).
	Preprocess VisionPreprocessConfig
	// DeviceSeam is an opaque hook a backend may attach its own device-resident encode state to —
	// engine/metal's #59 device-tower follow-up (docs/design-qwen-vision-factory.md §6) stores its
	// bf16 buffer mirror here once built. qwen35 never reads or writes it beyond the nil zero value
	// (AX-8: this package never imports engine); its lifetime is exactly the VisionTower's, so a
	// backend's device state is released whenever the tower itself becomes unreachable — no separate
	// package-level cache to evict on model unload.
	DeviceSeam any
}

// isqrt returns the integer square root of n (⌊√n⌋), or -1 for n<0 — used to recover the merger's
// spatial merge size from its first linear's width as a multiple of Hidden (must be a perfect square)
// and the learned table's square side.
func isqrt(n int) int {
	if n < 0 {
		return -1
	}
	r := int(math.Sqrt(float64(n)))
	for r*r > n {
		r--
	}
	for (r+1)*(r+1) <= n {
		r++
	}
	return r
}

// ImageToPatchGrid decodes raw PNG/JPEG image bytes, resizes it to the HF Qwen2VLImageProcessor's
// smart_resize target (both axes rounded to the nearest PatchSize·MergeSize multiple, the resized
// pixel COUNT clamped into pp's [MinPixels,MaxPixels] budget — qwenSmartResizeTarget is the dimension
// arithmetic, resizeBicubicRGB the pixel resample; vision_preprocess.go), rescales to [0,1], and
// normalises per channel ((v-mean[c])/std[c], pp's ImageMean/ImageStd) before slicing the result into
// a non-overlapping grid of PatchSize×PatchSize patches. Returns the flattened per-patch rows
// [gridH*gridW,PatchDim] (row-major raster order; each patch row is
// [PatchSize·PatchSize·InChannels] channel-last HWC, repeated TemporalPatchSize times — the Qwen-VL
// convention of feeding a still image as TemporalPatchSize identical frames into what a checkpoint's
// Conv3D patch embed reduces to for a non-overlapping kernel).
//
// v1 of this port (pre-#59-normalisation-follow-up) top-left CROPPED to the nearest patch·merge
// multiple with no resample and no normalisation, matching the retired composed engine's own
// patchifier verbatim (the values that lane's live 27B image turns fed the tower). The real HF
// processor resizes (not crops) and normalises; this function now does too — the measured follow-up
// docs/design-qwen-vision-factory.md §4 named. Because smart_resize's pixel-count floor always
// produces a target at or above MinPixels, a genuinely tiny image is upscaled rather than rejected —
// the old "image smaller than one patch·merge block" refusal only remains reachable when pp disables
// both bounds (MinPixels<=0) AND the image rounds all the way down to zero.
func ImageToPatchGrid(data []byte, cfg VisionTowerConfig, pp VisionPreprocessConfig) (patches []float32, gridH, gridW int, err error) {
	if cfg.PatchSize <= 0 {
		return nil, 0, 0, core.NewError("qwen35.ImageToPatchGrid: PatchSize must be positive")
	}
	img, _, derr := image.Decode(bytes.NewReader(data))
	if derr != nil {
		return nil, 0, 0, core.E("qwen35.ImageToPatchGrid", "decode image", derr)
	}
	bounds := img.Bounds()
	h, w := bounds.Dy(), bounds.Dx()
	if h <= 0 || w <= 0 {
		return nil, 0, 0, core.NewError("qwen35.ImageToPatchGrid: image has empty bounds")
	}
	unit := cfg.PatchSize * max(cfg.MergeSize, 1)
	pp = pp.normalized()
	th, tw, rerr := qwenSmartResizeTarget(h, w, unit, pp.MinPixels, pp.MaxPixels)
	if rerr != nil {
		return nil, 0, 0, core.E("qwen35.ImageToPatchGrid", "smart-resize target", rerr)
	}
	if th <= 0 || tw <= 0 {
		return nil, 0, 0, core.NewError(core.Sprintf(
			"qwen35.ImageToPatchGrid: %dx%d image resolves to a %dx%d target — smaller than one %d-pixel patch·merge block",
			w, h, tw, th, unit))
	}
	gridH, gridW = th/cfg.PatchSize, tw/cfg.PatchSize

	rgb := decodeImageRGB255(img, bounds)
	if th != h || tw != w {
		rgb = resizeBicubicRGB(rgb, h, w, th, tw)
	}
	for i, v := range rgb {
		rgb[i] = roundClampByte(v)
	}

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
				y := gy*cfg.PatchSize + py
				for px := range cfg.PatchSize {
					x := gx*cfg.PatchSize + px
					src := (y*tw + x) * 3
					patches[base+col] = normalisePixel(rgb[src], pp.ImageMean[0], pp.ImageStd[0])
					if channels > 1 {
						patches[base+col+1] = normalisePixel(rgb[src+1], pp.ImageMean[1], pp.ImageStd[1])
					}
					if channels > 2 {
						patches[base+col+2] = normalisePixel(rgb[src+2], pp.ImageMean[2], pp.ImageStd[2])
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

// normalisePixel rescales a 0..255 pixel value to [0,1] and applies the reference's per-channel
// normalisation: (v/255 - mean) / std.
func normalisePixel(v255 float64, mean, std float32) float32 {
	return (float32(v255)/255 - mean) / std
}

// InterpolatePosEmbed resamples a learned square [side²,hidden] position table onto a (dstH × dstW)
// patch grid with separable bilinear interpolation in the REFERENCE convention (the composed lane's
// port of vision_utils.get_vision_bilinear_indices_and_weights): per-axis coordinates are
// linspace(0, side-1, n) — align_corners style, both endpoints landing exactly on the table's edge
// rows — with the ceil corner clamped to side-1. A single-point axis (n == 1) sits at 0. The source
// table must be square (side = √positions), the real layout's convention; anything else fails loud.
func InterpolatePosEmbed(table []float32, hidden, dstH, dstW int) ([]float32, error) {
	if hidden <= 0 || dstH <= 0 || dstW <= 0 {
		return nil, core.NewError("qwen35.InterpolatePosEmbed: hidden and grid dims must be positive")
	}
	if len(table)%hidden != 0 {
		return nil, core.NewError("qwen35.InterpolatePosEmbed: table length is not a multiple of hidden")
	}
	positions := len(table) / hidden
	side := isqrt(positions)
	if side <= 0 || side*side != positions {
		return nil, core.NewError(core.Sprintf("qwen35.InterpolatePosEmbed: %d positions is not a square grid", positions))
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
