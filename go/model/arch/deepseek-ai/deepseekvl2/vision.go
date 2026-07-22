// SPDX-Licence-Identifier: EUPL-1.2

package deepseekvl2

import (
	"bytes"
	"image"
	_ "image/jpeg" // register the JPEG decoder with image.Decode
	_ "image/png"  // register the PNG decoder with image.Decode

	core "dappco.re/go"
)

// vision.go is DeepseekOCRModel.forward's image-conditioning half ported host-side: decode the
// image → SAM tower → CLIP tower (fed SAM's raw output as ITS patch_embeds — deepencoder.py's
// actual wiring: CLIPVisionEmbeddings.forward skips its own conv whenever patch_embeds is
// supplied, and DeepseekOCRModel ALWAYS supplies it) → concat the two towers' per-patch features
// → the linear projector → assemble the image_newline/view_separator soft-token run the decoder's
// prompt embeds get scattered into (tokens.go).
//
// V1 SCOPE (host-f32 correctness-first, matching the whisper lane's phasing — device fusion is
// NOT this lane): DeepSeek-OCR's "Base" resolution mode only (base_size=image_size=1024,
// crop_mode=false in the reference's infer() — one of its four documented single-view presets,
// see the README) — the GLOBAL VIEW, always padded onto a fixed 1024×1024 canvas via PIL's
// ImageOps.pad. That pad step resizes with a bicubic filter whenever the source size differs from
// 1024×1024, and bit-exact bicubic parity with PIL's C resampler is impractical to port (see
// testdata/gen_fixture.py's doc comment for the empirical proof that a SAME-SIZE pad is an exact
// no-op, which is what makes the golden fixture and this v1 boundary consistent) — so this lane
// REQUIRES an already-1024×1024 image and refuses anything else by name, rather than silently
// resizing via a different, non-matching kernel. The Gundam mode's dynamic local-crop tiling
// (dynamic_preprocess — multiple 640×640 crops of a larger source, each run through the SAME
// towers and concatenated) is a distinct, separately-scoped v2 slice: it also needs
// get_abs_pos/get_abs_pos_sam's bicubic POSITION-TABLE interpolation (a crop's patch grid is
// smaller than the pretrained one), which the "Base" mode never exercises (SAM's pos_embed is
// already 64×64 = the checkpoint's one supported input's own patch grid; CLIP's 16×16 pretrained
// position grid always matches SAM's 1024-canvas output exactly — see weights_sam.go/
// weights_clip.go's doc comments) — implementing it is real, scoped follow-on work, not a stub.

// visionGridSize is the patch grid SAM/CLIP/the projector hand off at (16×16) — samGridSize's own
// two stride-2 downsamples (net_2/net_3, see vision_sam.go).
const visionGridSize = samGridSize / 4

// NumImageTokens is the exact count of <image>-placeholder tokens (and, correspondingly, soft-
// token embedding rows VisionForward returns) the v1 "Base" resolution mode always produces:
// visionGridSize rows of visionGridSize patches each, one image_newline row appended after every
// grid row, plus one final view_separator row — ([id]*16+[id])*16+[id] in the reference's own
// tokeniser math (modeling_deepseekocr.py's infer(), crop_mode=False branch).
const NumImageTokens = visionGridSize*(visionGridSize+1) + 1

// DecodeAndNormaliseImage decodes PNG/JPEG bytes and returns [1024,1024,3] channel-last pixels
// normalised to [-1,1] (BasicImageTransform's ToTensor()+Normalize(mean=0.5,std=0.5) —
// pixel/127.5 - 1), refusing anything not already exactly samImgSize×samImgSize (see the file
// doc comment's V1 SCOPE).
func DecodeAndNormaliseImage(data []byte) ([]float32, error) {
	img, _, err := image.Decode(bytes.NewReader(data))
	if err != nil {
		return nil, core.E("deepseekvl2.DecodeAndNormaliseImage", "decode image", err)
	}
	bounds := img.Bounds()
	w, h := bounds.Dx(), bounds.Dy()
	if w != samImgSize || h != samImgSize {
		return nil, core.NewError(core.Sprintf(
			"deepseekvl2.DecodeAndNormaliseImage: image is %dx%d, want exactly %dx%d — this v1 lane implements "+
				"DeepSeek-OCR's fixed-canvas \"Base\" resolution mode only (bicubic resize/letterbox to that "+
				"canvas is not implemented; see vision.go's doc comment); pre-resize/pad the image to %dx%d "+
				"(e.g. centred on a mid-grey background) before calling",
			w, h, samImgSize, samImgSize, samImgSize, samImgSize))
	}
	pixels := make([]float32, samImgSize*samImgSize*3)
	idx := 0
	for y := range h {
		for x := range w {
			r, g, b, _ := img.At(bounds.Min.X+x, bounds.Min.Y+y).RGBA() // 16-bit, premultiplied per image.Color's contract
			pixels[idx] = float32(r>>8)/127.5 - 1
			pixels[idx+1] = float32(g>>8)/127.5 - 1
			pixels[idx+2] = float32(b>>8)/127.5 - 1
			idx += 3
		}
	}
	return pixels, nil
}

// VisionForward runs the whole DeepEncoder (SAM feeding CLIP, concat, project, assemble) over one
// already-decoded-and-normalised [1024,1024,3] image (DecodeAndNormaliseImage's output),
// returning the [NumImageTokens,hidden] soft-token embedding run tokens.go scatters into the
// prompt at every <image>-placeholder position, in order.
func VisionForward(pixels []float32, w *Weights) ([]float32, error) {
	samOut, err := SAMForward(pixels, w.SAM) // [visionGridSize*visionGridSize, samNeckOut3]
	if err != nil {
		return nil, core.E("deepseekvl2.VisionForward", "SAM tower", err)
	}
	clipOut, err := CLIPForward(samOut, w.CLIP) // [clipNumPositions, clipHidden] (CLS-first)
	if err != nil {
		return nil, core.E("deepseekvl2.VisionForward", "CLIP tower", err)
	}

	patches := visionGridSize * visionGridSize
	concat := make([]float32, patches*projectorInputDim)
	for i := range patches {
		dst := concat[i*projectorInputDim : (i+1)*projectorInputDim]
		copy(dst[0:clipHidden], clipOut[(i+1)*clipHidden:(i+2)*clipHidden]) // clip_out[:,1:] — drop the CLS row (index 0)
		copy(dst[clipHidden:], samOut[i*samNeckOut3:(i+1)*samNeckOut3])
	}
	projected := linear(concat, w.ProjW, projectorInputDim, w.Decoder.hiddenSize(), w.ProjB) // [patches, hidden]
	hidden := w.Decoder.hiddenSize()

	// Reassemble: visionGridSize rows of visionGridSize patch-tokens each, one ImageNewline row
	// appended after every row, then one final ViewSeparator row — matches
	// modeling_deepseekocr.py's "NO PATCHES" (no local crops) global-only branch exactly:
	//   global_features = global_features.view(h,w,n_dim)
	//   global_features = cat([global_features, image_newline.expand(h,1,n_dim)], dim=1).view(-1,n_dim)
	//   global_local_features = cat([global_features, view_seperator], dim=0)
	out := make([]float32, NumImageTokens*hidden)
	row := 0
	for r := range visionGridSize {
		for c := range visionGridSize {
			copy(out[row*hidden:(row+1)*hidden], projected[(r*visionGridSize+c)*hidden:(r*visionGridSize+c+1)*hidden])
			row++
		}
		copy(out[row*hidden:(row+1)*hidden], w.ImageNewline)
		row++
	}
	copy(out[row*hidden:(row+1)*hidden], w.ViewSeparator)
	row++
	if row != NumImageTokens {
		return nil, core.NewError(core.Sprintf("deepseekvl2.VisionForward: assembled %d soft-token rows, want %d", row, NumImageTokens))
	}
	return out, nil
}

// hiddenSize reports the decoder's hidden width — VisionForward's own dimension check (the
// projector's OUTPUT width, which weights.go's LoadWeights already shape-checked against
// cfg.HiddenSize when reading model.projector.layers.{weight,bias}).
func (d DecoderWeights) hiddenSize() int { return len(d.FinalNormW) }
