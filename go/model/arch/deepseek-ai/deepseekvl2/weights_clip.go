// SPDX-Licence-Identifier: EUPL-1.2

package deepseekvl2

import core "dappco.re/go"

// weights_clip.go reads the CLIP-L tower (deepencoder.py's VitModel, built by build_clip_l() with
// the hardcoded vit_model_cfg — again NOT config.json-derived, see VisionConfig's doc comment).
// Tensor names read verbatim off the real checkpoint, prefix model.vision_model.*.
//
// model.vision_model.embeddings.patch_embedding.weight (a real tensor the checkpoint carries,
// [1024,3,14,14]) is DELIBERATELY NOT LOADED: CLIPVisionEmbeddings.forward only runs its own conv
// when patch_embeds is nil, and DeepseekOCRModel.forward NEVER calls it that way — it always
// passes SAM's raw tower output as patch_embeds (see vision.go's doc comment) — so that
// projection is provably dead weight for this checkpoint's actual forward path, not an oversight.

// CLIP's hardcoded geometry (deepencoder.py's vit_model_cfg/build_clip_l — see VisionConfig's doc
// comment). clipNumPositions is the PRETRAINED grid this checkpoint's position_embedding table
// was trained at (16x16 patches + 1 CLS, image_size 224 / patch_size 14) — SAM's own tower always
// hands CLIP exactly a 16x16 grid too (1024/16/2/2, see vision_sam.go's samGridSize/downsample
// chain), so get_abs_pos's bicubic resample branch is NEVER exercised for the v1 "Base" resolution
// mode this package implements (see ocr.go's doc comment) — VisionForward asserts the grid size
// rather than silently resampling a mismatched one.
const (
	clipHidden       = 1024
	clipNumHeads     = 16
	clipHeadDim      = clipHidden / clipNumHeads // 64
	clipFFNHidden    = 4096
	clipNumLayers    = 24
	clipNumPositions = 257 // 16*16 + 1 CLS
	clipLayerNormEps = 1e-5
)

// CLIPBlockWeights is one pre-norm transformer block: LayerNorm -> combined-QKV bidirectional
// self-attention -> residual, LayerNorm -> quick-GELU 2-layer MLP -> residual
// (deepencoder.NoTPTransformerBlock.forward).
type CLIPBlockWeights struct {
	Norm1W, Norm1B     []float32
	QKVWeight, QKVBias []float32 // [3072,1024] / [3072]
	OutWeight, OutBias []float32 // [1024,1024] / [1024]
	Norm2W, Norm2B     []float32
	FC1Weight, FC1Bias []float32 // [4096,1024] / [4096]
	FC2Weight, FC2Bias []float32 // [1024,4096] / [1024]
}

// CLIPWeights is the whole loaded CLIP-L tower: the class(CLS) embedding + learned position
// table (embeddings.forward — patch_embeds is supplied externally, see the file doc comment), one
// pre-transformer LayerNorm, then clipNumLayers blocks. There is NO final/post LayerNorm — VitModel.forward
// returns the transformer stack's raw output (confirmed: no "post_layernorm"/final-norm tensor
// exists in the checkpoint).
type CLIPWeights struct {
	ClassEmbedding         []float32 // [1024]
	PositionEmbedding      []float32 // [257,1024]
	PreLNWeight, PreLNBias []float32 // [1024]
	Blocks                 []CLIPBlockWeights
}

func loadCLIPWeights(l weightLoader) (CLIPWeights, error) {
	var w CLIPWeights
	var err error
	if w.ClassEmbedding, err = l.f32shaped("model.vision_model.embeddings.class_embedding", clipHidden); err != nil {
		return w, err
	}
	if w.PositionEmbedding, err = l.f32shaped("model.vision_model.embeddings.position_embedding.weight", clipNumPositions*clipHidden); err != nil {
		return w, err
	}
	if w.PreLNWeight, w.PreLNBias, err = l.lnBiasW("model.vision_model.pre_layrnorm", clipHidden); err != nil {
		return w, err
	}

	w.Blocks = make([]CLIPBlockWeights, clipNumLayers)
	for i := range w.Blocks {
		p := core.Sprintf("model.vision_model.transformer.layers.%d", i)
		var b CLIPBlockWeights
		if b.Norm1W, b.Norm1B, err = l.lnBiasW(p+".layer_norm1", clipHidden); err != nil {
			return w, err
		}
		if b.QKVWeight, b.QKVBias, err = l.linearW(p+".self_attn.qkv_proj", clipHidden, 3*clipHidden, true); err != nil {
			return w, err
		}
		if b.OutWeight, b.OutBias, err = l.linearW(p+".self_attn.out_proj", clipHidden, clipHidden, true); err != nil {
			return w, err
		}
		if b.Norm2W, b.Norm2B, err = l.lnBiasW(p+".layer_norm2", clipHidden); err != nil {
			return w, err
		}
		if b.FC1Weight, b.FC1Bias, err = l.linearW(p+".mlp.fc1", clipHidden, clipFFNHidden, true); err != nil {
			return w, err
		}
		if b.FC2Weight, b.FC2Bias, err = l.linearW(p+".mlp.fc2", clipFFNHidden, clipHidden, true); err != nil {
			return w, err
		}
		w.Blocks[i] = b
	}
	return w, nil
}
