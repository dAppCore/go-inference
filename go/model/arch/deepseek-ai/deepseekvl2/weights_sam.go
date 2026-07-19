// SPDX-Licence-Identifier: EUPL-1.2

package deepseekvl2

import core "dappco.re/go"

// weights_sam.go reads the SAM ViT-B tower (deepencoder.py's ImageEncoderViT, built by
// build_sam_vit_b() with NO config argument — every dimension below is hardcoded in that
// function, confirmed against the checkpoint's real model.sam_model.* tensor shapes, not read
// from config.json's vision_config — see VisionConfig's doc comment). Tensor names read verbatim
// off the real checkpoint's model.safetensors.index.json, prefix model.sam_model.*.

// SAM's hardcoded geometry (deepencoder.py's build_sam_vit_b/_build_sam/ImageEncoderViT — the
// checkpoint takes no vision_config input for these; see VisionConfig's doc comment). The neck's
// two downsample convs (net_2/net_3) are ALSO hardcoded to 256->512->1024 in ImageEncoderViT
// itself, independent of the constructor's out_chans argument — samOutChans (=256) is that
// argument's real value, which happens to equal net_2's expected input width.
const (
	samImgSize    = 1024
	samPatchSize  = 16
	samGridSize   = samImgSize / samPatchSize // 64 — both the patch grid AND pos_embed's own square side
	samEmbedDim   = 768
	samDepth      = 12
	samNumHeads   = 12
	samHeadDim    = samEmbedDim / samNumHeads // 64
	samMLPHidden  = samEmbedDim * 4           // mlp_ratio=4.0
	samOutChans   = 256                       // neck's output width == net_2's input width
	samNeckOut2   = 512                       // net_2's output width == net_3's input width
	samNeckOut3   = 1024                      // net_3's output width == CLIP's hidden size
	samWindowSize = 14
)

// samGlobalAttnIndexes are the 0-indexed block positions that run attention over the FULL 64x64
// grid rather than samWindowSize x samWindowSize windows (deepencoder.py's
// build_sam_vit_b(encoder_global_attn_indexes=[2,5,8,11])).
var samGlobalAttnIndexes = map[int]bool{2: true, 5: true, 8: true, 11: true}

// SAMAttnWeights is one SAM block's windowed-or-global multi-head attention: a single combined
// QKV projection (deepencoder.Attention.qkv, unlike CLIP's separately-named-but-still-combined
// qkv_proj — same shape, different tensor name prefix) plus the decomposed relative-position
// bias tables (deepencoder.py's add_decomposed_rel_pos). RelPosLen is 2*windowSize-1 for a
// windowed block (27, windowSize=14) or 2*samGridSize-1 for a global block (127, grid=64) — the
// two block KINDS carry DIFFERENT-SHAPED tables, read directly off each block's own tensors, not
// assumed.
type SAMAttnWeights struct {
	QKVWeight, QKVBias   []float32 // [3*768,768] / [3*768]
	ProjWeight, ProjBias []float32 // [768,768] / [768]
	RelPosH, RelPosW     []float32 // [RelPosLen,64] each
	RelPosLen            int
}

// SAMBlockWeights is one pre-norm ViTDet block: LayerNorm -> (window-partitioned or global)
// attention -> residual, LayerNorm -> GELU MLP -> residual (deepencoder.Block.forward).
// WindowSize is 0 for a global-attention block, samWindowSize otherwise (deepencoder.py's
// ImageEncoderViT.__init__: "window_size if i not in global_attn_indexes else 0").
type SAMBlockWeights struct {
	Norm1W, Norm1B     []float32
	Attn               SAMAttnWeights
	Norm2W, Norm2B     []float32
	MLPLin1W, MLPLin1B []float32 // [3072,768] / [3072]
	MLPLin2W, MLPLin2B []float32 // [768,3072] / [768]
	WindowSize         int
}

// SAMWeights is the whole loaded SAM ViT-B tower: patch embed conv, the fixed absolute position
// table, samDepth blocks, and the neck+downsample tail (neck: 1x1 conv -> LayerNorm2d -> 3x3
// conv(pad1) -> LayerNorm2d; then net_2/net_3: two more stride-2 3x3 convs — deepencoder.py's
// ImageEncoderViT.forward).
type SAMWeights struct {
	PatchEmbedW, PatchEmbedB []float32 // [768,3,16,16] / [768]
	PosEmbed                 []float32 // [64,64,768] flat (leading batch=1 dim dropped)
	Blocks                   []SAMBlockWeights

	NeckConv1W         []float32 // [256,768,1,1]
	NeckLN1W, NeckLN1B []float32 // [256]
	NeckConv2W         []float32 // [256,256,3,3]
	NeckLN2W, NeckLN2B []float32 // [256]
	Net2W              []float32 // [512,256,3,3]
	Net3W              []float32 // [1024,512,3,3]
}

func loadSAMWeights(l weightLoader) (SAMWeights, error) {
	var w SAMWeights
	var err error
	w.PatchEmbedW, err = l.f32shaped("model.sam_model.patch_embed.proj.weight", samEmbedDim*3*samPatchSize*samPatchSize)
	if err != nil {
		return w, err
	}
	w.PatchEmbedB, err = l.f32shaped("model.sam_model.patch_embed.proj.bias", samEmbedDim)
	if err != nil {
		return w, err
	}
	w.PosEmbed, err = l.f32shaped("model.sam_model.pos_embed", samGridSize*samGridSize*samEmbedDim)
	if err != nil {
		return w, err
	}

	w.Blocks = make([]SAMBlockWeights, samDepth)
	for i := range w.Blocks {
		p := core.Sprintf("model.sam_model.blocks.%d", i)
		var b SAMBlockWeights
		if b.Norm1W, b.Norm1B, err = l.lnBiasW(p+".norm1", samEmbedDim); err != nil {
			return w, err
		}
		if b.Norm2W, b.Norm2B, err = l.lnBiasW(p+".norm2", samEmbedDim); err != nil {
			return w, err
		}
		if b.Attn.QKVWeight, b.Attn.QKVBias, err = l.linearW(p+".attn.qkv", samEmbedDim, 3*samEmbedDim, true); err != nil {
			return w, err
		}
		if b.Attn.ProjWeight, b.Attn.ProjBias, err = l.linearW(p+".attn.proj", samEmbedDim, samEmbedDim, true); err != nil {
			return w, err
		}
		if samGlobalAttnIndexes[i] {
			b.WindowSize = 0
			b.Attn.RelPosLen = 2*samGridSize - 1
		} else {
			b.WindowSize = samWindowSize
			b.Attn.RelPosLen = 2*samWindowSize - 1
		}
		if b.Attn.RelPosH, err = l.f32shaped(p+".attn.rel_pos_h", b.Attn.RelPosLen*samHeadDim); err != nil {
			return w, err
		}
		if b.Attn.RelPosW, err = l.f32shaped(p+".attn.rel_pos_w", b.Attn.RelPosLen*samHeadDim); err != nil {
			return w, err
		}
		if b.MLPLin1W, b.MLPLin1B, err = l.linearW(p+".mlp.lin1", samEmbedDim, samMLPHidden, true); err != nil {
			return w, err
		}
		if b.MLPLin2W, b.MLPLin2B, err = l.linearW(p+".mlp.lin2", samMLPHidden, samEmbedDim, true); err != nil {
			return w, err
		}
		w.Blocks[i] = b
	}

	if w.NeckConv1W, err = l.f32shaped("model.sam_model.neck.0.weight", samOutChans*samEmbedDim*1*1); err != nil {
		return w, err
	}
	if w.NeckLN1W, w.NeckLN1B, err = l.lnBiasW("model.sam_model.neck.1", samOutChans); err != nil {
		return w, err
	}
	if w.NeckConv2W, err = l.f32shaped("model.sam_model.neck.2.weight", samOutChans*samOutChans*3*3); err != nil {
		return w, err
	}
	if w.NeckLN2W, w.NeckLN2B, err = l.lnBiasW("model.sam_model.neck.3", samOutChans); err != nil {
		return w, err
	}
	if w.Net2W, err = l.f32shaped("model.sam_model.net_2.weight", samNeckOut2*samOutChans*3*3); err != nil {
		return w, err
	}
	if w.Net3W, err = l.f32shaped("model.sam_model.net_3.weight", samNeckOut3*samNeckOut2*3*3); err != nil {
		return w, err
	}
	return w, nil
}
