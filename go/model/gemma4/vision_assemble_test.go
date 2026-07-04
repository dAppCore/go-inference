// SPDX-Licence-Identifier: EUPL-1.2

package gemma4

import (
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference/safetensors"
)

// TestAssembleVision builds a synthetic 2-layer SigLIP tower + projector and pins that AssembleVision
// gathers every role, infers the layer count, and validates presence.
func TestAssembleVision(t *testing.T) {
	mk := func(rows, cols int) safetensors.Tensor {
		return safetensors.Tensor{Dtype: "BF16", Shape: []int{rows, cols}, Data: make([]byte, rows*cols*2)}
	}
	vec := func(n int) safetensors.Tensor {
		return safetensors.Tensor{Dtype: "BF16", Shape: []int{n}, Data: make([]byte, n*2)}
	}
	const H, layers = 64, 2
	w := map[string]safetensors.Tensor{"patch_embedding.weight": mk(H, 588)} // hidden 64, patchDim 588 → patch 14
	for i := 0; i < layers; i++ {
		p := core.Sprintf("encoder.layers.%d", i)
		for _, n := range []string{".input_layernorm", ".post_attention_layernorm", ".pre_feedforward_layernorm", ".post_feedforward_layernorm", ".self_attn.q_norm", ".self_attn.k_norm"} {
			w[p+n+".weight"] = vec(H)
		}
		for _, n := range []string{".self_attn.q_proj", ".self_attn.k_proj", ".self_attn.v_proj", ".self_attn.o_proj"} {
			w[p+n+".weight"] = mk(H, H)
		}
		w[p+".mlp.gate_proj.weight"] = mk(H*4, H)
		w[p+".mlp.up_proj.weight"] = mk(H*4, H)
		w[p+".mlp.down_proj.weight"] = mk(H, H*4)
	}
	w["multi_modal_projector.proj.weight"] = mk(H, H)
	w["patch_embedder.position_embedding_table"] = safetensors.Tensor{
		Dtype: "BF16",
		Shape: []int{2, 7, H},
		Data:  make([]byte, 2*7*H*2),
	}

	tc := &Gemma4TextConfig{}
	tc.ModelType = "gemma4"
	tc.ImageTokenID = 262145
	tc.VideoTokenID = 258884
	tc.VisionConfig = &Gemma4VisionConfig{}
	tc.VisionConfig.NumAttentionHeads = 8

	v, err := AssembleVision(w, tc)
	if err != nil {
		t.Fatalf("AssembleVision: %v", err)
	}
	if v == nil {
		t.Fatal("expected a vision tower")
	}
	if len(v.Layers) != layers {
		t.Fatalf("layers = %d, want %d", len(v.Layers), layers)
	}
	if v.PatchEmbedding == nil {
		t.Fatal("patch embedding missing")
	}
	if len(v.PatchConvWeight) != len(v.PatchEmbedding) {
		t.Fatalf("patch conv weight bytes = %d, want %d", len(v.PatchConvWeight), len(v.PatchEmbedding))
	}
	if v.Cfg.PatchSize != 14 || v.Cfg.NumChannels != 3 {
		t.Fatalf("patch geometry = patch %d channels %d, want 14/3", v.Cfg.PatchSize, v.Cfg.NumChannels)
	}
	if v.Layers[0].Q.Weight == nil || v.Layers[0].QNorm == nil || v.Layers[0].Gate.Weight == nil {
		t.Fatal("layer 0 q/qnorm/gate missing")
	}
	if v.Projector.Projection.Weight == nil {
		t.Fatal("projector missing")
	}
	if v.Cfg.ImageTokenID != 262145 {
		t.Fatalf("image token id = %d, want 262145", v.Cfg.ImageTokenID)
	}
	if v.Cfg.ImageBeginToken != Gemma4BOIToken || v.Cfg.ImageToken != Gemma4ImageToken || v.Cfg.ImageEndToken != Gemma4EOIToken {
		t.Fatalf("image prompt tokens = %q/%q/%q", v.Cfg.ImageBeginToken, v.Cfg.ImageToken, v.Cfg.ImageEndToken)
	}
	if v.Cfg.VideoTokenID != 258884 || v.Cfg.VideoToken != Gemma4VideoToken {
		t.Fatalf("video prompt tokens = %d/%q", v.Cfg.VideoTokenID, v.Cfg.VideoToken)
	}
	if v.Cfg.PositionEmbeddingSize != 7 {
		t.Fatalf("position embedding size = %d, want 7", v.Cfg.PositionEmbeddingSize)
	}
	if len(v.PositionEmbeddings) != 2*7*H*2 {
		t.Fatalf("position embedding bytes = %d, want %d", len(v.PositionEmbeddings), 2*7*H*2)
	}
}

func TestAssembleVisionPatchConvChannelsFirstNormalisesGood(t *testing.T) {
	mk := func(shape ...int) safetensors.Tensor {
		n := 1
		for _, d := range shape {
			n *= d
		}
		data := make([]byte, n*2)
		for i := range data {
			data[i] = byte(i)
		}
		return safetensors.Tensor{Dtype: "BF16", Shape: shape, Data: data}
	}
	weights := map[string]safetensors.Tensor{
		"patch_embedding.weight": mk(1, 3, 2, 2), // [hidden, channels, patch, patch]
	}
	tc := &Gemma4TextConfig{}
	tc.ModelType = "gemma4"
	tc.VisionConfig = &Gemma4VisionConfig{}
	tc.VisionConfig.NumChannels = 3
	tc.VisionConfig.NumHiddenLayers = 0

	v, err := AssembleVision(weights, tc)
	if err != nil {
		t.Fatalf("AssembleVision: %v", err)
	}
	if v == nil {
		t.Fatal("expected a vision tower")
	}
	want := []byte{
		0, 1, 8, 9, 16, 17,
		2, 3, 10, 11, 18, 19,
		4, 5, 12, 13, 20, 21,
		6, 7, 14, 15, 22, 23,
	}
	if len(v.PatchConvWeight) != len(want) {
		t.Fatalf("channels-first conv bytes = %d, want %d", len(v.PatchConvWeight), len(want))
	}
	for i := range want {
		if v.PatchConvWeight[i] != want[i] || v.PatchEmbedding[i] != want[i] {
			t.Fatalf("normalised byte %d = conv %d linear %d, want %d", i, v.PatchConvWeight[i], v.PatchEmbedding[i], want[i])
		}
	}
}

func TestVisionLinearWithInputDimQuantMetadata(t *testing.T) {
	const outDim, inDim, groupSize, bits = 8, 64, 16, 4
	weights := map[string]safetensors.Tensor{
		"embed_vision.embedding_projection.weight": {
			Dtype: "U32",
			Shape: []int{outDim, inDim * bits / 32},
			Data:  make([]byte, outDim*(inDim*bits/32)*4),
		},
		"embed_vision.embedding_projection.scales": {
			Dtype: "BF16",
			Shape: []int{outDim, inDim / groupSize},
			Data:  make([]byte, outDim*(inDim/groupSize)*2),
		},
		"embed_vision.embedding_projection.biases": {
			Dtype: "BF16",
			Shape: []int{outDim, inDim / groupSize},
			Data:  make([]byte, outDim*(inDim/groupSize)*2),
		},
	}

	lin := visionLinearWithInputDim(weights, inDim, "embed_vision.embedding_projection")
	if len(lin.Scales) == 0 || len(lin.Biases) == 0 {
		t.Fatalf("quant vision projector scales/biases missing: %+v", lin)
	}
	if lin.OutDim != outDim || lin.InDim != inDim || lin.GroupSize != groupSize || lin.Bits != bits || lin.Kind != "affine" {
		t.Fatalf("quant vision projector geometry = out:%d in:%d group:%d bits:%d kind:%q", lin.OutDim, lin.InDim, lin.GroupSize, lin.Bits, lin.Kind)
	}
}

// TestAssembleVisionTextOnly pins that a pack with no vision tower yields (nil, nil).
func TestAssembleVisionTextOnly(t *testing.T) {
	tc := &Gemma4TextConfig{}
	tc.ModelType = "gemma4"
	v, err := AssembleVision(map[string]safetensors.Tensor{
		"model.layers.0.self_attn.q_proj.weight": {Shape: []int{4, 4}},
	}, tc)
	if err != nil || v != nil {
		t.Fatalf("text-only pack should yield (nil,nil), got (%v, %v)", v, err)
	}
}
