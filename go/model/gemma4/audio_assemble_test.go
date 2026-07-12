// SPDX-Licence-Identifier: EUPL-1.2

package gemma4

import (
	"math"
	"testing"

	"dappco.re/go/inference/model/safetensors"
)

func audioBF16Tensor(shape ...int) safetensors.Tensor {
	n := 1
	for _, d := range shape {
		n *= d
	}
	data := make([]byte, n*2)
	for i := 0; i < n; i++ {
		data[i*2] = byte(i)
		data[i*2+1] = byte(i >> 8)
	}
	return safetensors.Tensor{Dtype: "BF16", Shape: shape, Data: data}
}

func TestAssembleAudio(t *testing.T) {
	const hidden, heads, headDim, ff, outDim, mel, outC0, outC1, kernel = 8, 2, 4, 16, 6, 4, 4, 2, 5
	cfg := &Gemma4TextConfig{
		AudioTokenID: 77,
		AudioConfig: normalizeGemma4AudioConfig(&Gemma4AudioConfig{
			HiddenSize:              hidden,
			NumHiddenLayers:         1,
			NumAttentionHeads:       heads,
			AttentionChunkSize:      2,
			AttentionContextLeft:    3,
			AttentionContextRight:   1,
			AttentionLogitCap:       50,
			ConvKernelSize:          kernel,
			SubsamplingConvChannels: []int32{mel, outC1},
			OutputProjDims:          outDim,
		}),
	}
	w := map[string]safetensors.Tensor{
		"audio_tower.subsample_conv_projection.layer0.conv.weight":       audioBF16Tensor(outC0, 1, 3, 3),
		"audio_tower.subsample_conv_projection.layer0.norm.weight":       audioBF16Tensor(outC0),
		"audio_tower.subsample_conv_projection.layer1.conv.weight":       audioBF16Tensor(outC1, outC0, 3, 3),
		"audio_tower.subsample_conv_projection.layer1.norm.weight":       audioBF16Tensor(outC1),
		"audio_tower.subsample_conv_projection.input_proj_linear.weight": audioBF16Tensor(hidden, (mel/4)*outC1),
		"audio_tower.output_proj.weight":                                 audioBF16Tensor(outDim, hidden),
		"audio_tower.output_proj.bias":                                   audioBF16Tensor(outDim),
		"embed_audio.embedding_projection.weight":                        audioBF16Tensor(hidden, outDim),
		"audio_tower.layers.0.self_attn.q_proj.linear.weight":            audioBF16Tensor(hidden, hidden),
		"audio_tower.layers.0.self_attn.k_proj.linear.weight":            audioBF16Tensor(hidden, hidden),
		"audio_tower.layers.0.self_attn.v_proj.linear.weight":            audioBF16Tensor(hidden, hidden),
		"audio_tower.layers.0.self_attn.post.linear.weight":              audioBF16Tensor(hidden, hidden),
		"audio_tower.layers.0.self_attn.relative_k_proj.weight":          audioBF16Tensor(hidden, hidden),
		"audio_tower.layers.0.self_attn.per_dim_scale":                   {Dtype: "BF16", Shape: []int{headDim}, Data: make([]byte, headDim*2)},
		"audio_tower.layers.0.lconv1d.linear_start.linear.weight":        audioBF16Tensor(2*hidden, hidden),
		"audio_tower.layers.0.lconv1d.linear_end.linear.weight":          audioBF16Tensor(hidden, hidden),
		"audio_tower.layers.0.lconv1d.depthwise_conv1d.weight":           audioBF16Tensor(hidden, 1, kernel),
		"audio_tower.layers.0.lconv1d.pre_layer_norm.weight":             audioBF16Tensor(hidden),
		"audio_tower.layers.0.lconv1d.conv_norm.weight":                  audioBF16Tensor(hidden),
		"audio_tower.layers.0.norm_pre_attn.weight":                      audioBF16Tensor(hidden),
		"audio_tower.layers.0.norm_post_attn.weight":                     audioBF16Tensor(hidden),
		"audio_tower.layers.0.norm_out.weight":                           audioBF16Tensor(hidden),
	}
	for _, ffName := range []string{"feed_forward1", "feed_forward2"} {
		base := "audio_tower.layers.0." + ffName
		w[base+".ffw_layer_1.linear.weight"] = audioBF16Tensor(ff, hidden)
		w[base+".ffw_layer_2.linear.weight"] = audioBF16Tensor(hidden, ff)
		w[base+".pre_layer_norm.weight"] = audioBF16Tensor(hidden)
		w[base+".post_layer_norm.weight"] = audioBF16Tensor(hidden)
	}

	a, err := AssembleAudio(SanitizeAudioWeights(w), cfg)
	if err != nil {
		t.Fatalf("AssembleAudio: %v", err)
	}
	if a == nil || len(a.Layers) != 1 {
		t.Fatalf("audio payload = %+v, want one layer", a)
	}
	if a.Cfg.AudioTokenID != 77 || a.Cfg.AudioBeginToken != Gemma4BOAToken || a.Cfg.AudioToken != Gemma4AudioToken || a.Cfg.AudioEndToken != Gemma4EOAToken {
		t.Fatalf("audio prompt metadata = %+v", a.Cfg)
	}
	if a.Cfg.Hidden != hidden || a.Cfg.FFInter != ff || a.Cfg.Channels != hidden || a.Cfg.HeadDim != headDim || a.Cfg.OutputDim != outDim {
		t.Fatalf("audio config = %+v", a.Cfg)
	}
	if len(a.Subsample.Norm0B) != outC0*2 || len(a.Subsample.Norm1B) != outC1*2 {
		t.Fatalf("subsample synthetic norm biases len = %d/%d", len(a.Subsample.Norm0B), len(a.Subsample.Norm1B))
	}
	rawConv1 := w["audio_tower.subsample_conv_projection.layer1.conv.weight"].Data
	srcElem := ((1*outC0+3)*3+2)*3 + 1
	dstElem := ((1*3+2)*3+1)*outC0 + 3
	if a.Subsample.Conv1[dstElem*2] != rawConv1[srcElem*2] || a.Subsample.Conv1[dstElem*2+1] != rawConv1[srcElem*2+1] {
		t.Fatal("layer1 conv was not transposed from torch OIHW to native OHWI")
	}
	wantScale := float32(1 / math.Sqrt(headDim))
	if got := a.Layers[0].Attn.QScalePerDim[0]; math.Abs(float64(got-wantScale)) > 1e-6 {
		t.Fatalf("folded q scale = %v, want %v", got, wantScale)
	}
	if len(a.Projector.Weight) == 0 || len(a.OutputProj) == 0 || len(a.Layers[0].LConv.DepthwiseWeight) != hidden*kernel*2 {
		t.Fatal("audio projector/output/depthwise payload missing")
	}
	// Defect-catcher: output_proj.bias must be loaded (HF's output_proj is bias=True; dropping it
	// corrupts every clip). The bytes must be the checkpoint's [outDim] bias verbatim.
	wantBias := w["audio_tower.output_proj.bias"].Data
	if len(a.OutputProjBias) != outDim*2 {
		t.Fatalf("output_proj.bias len = %d, want %d bytes", len(a.OutputProjBias), outDim*2)
	}
	for i := range wantBias {
		if a.OutputProjBias[i] != wantBias[i] {
			t.Fatalf("output_proj.bias byte %d = %d, want %d", i, a.OutputProjBias[i], wantBias[i])
		}
	}
}

func TestAssembleAudioQuantizedProjectorMetadata(t *testing.T) {
	const outDim, inDim, groupSize, bits = 8, 64, 16, 4
	weights := map[string]safetensors.Tensor{
		"embed_audio.embedding_projection.weight": {
			Dtype: "U32",
			Shape: []int{outDim, inDim * bits / 32},
			Data:  make([]byte, outDim*(inDim*bits/32)*4),
		},
		"embed_audio.embedding_projection.scales": audioBF16Tensor(outDim, inDim/groupSize),
		"embed_audio.embedding_projection.biases": audioBF16Tensor(outDim, inDim/groupSize),
	}
	cfg := &Gemma4TextConfig{
		AudioTokenID: 77,
		AudioConfig:  &Gemma4AudioConfig{OutputProjDims: inDim},
	}

	audio, err := AssembleAudio(SanitizeAudioWeights(weights), cfg)
	if err != nil {
		t.Fatalf("AssembleAudio(quant projector): %v", err)
	}
	if audio == nil {
		t.Fatal("AssembleAudio(quant projector) returned nil payload")
	}
	p := audio.Projector
	if len(p.Scales) == 0 || len(p.Biases) == 0 {
		t.Fatalf("quant projector scales/biases missing: %+v", p)
	}
	if p.OutDim != outDim || p.InDim != inDim || p.GroupSize != groupSize || p.Bits != bits || p.Kind != "affine" {
		t.Fatalf("quant projector geometry = out:%d in:%d group:%d bits:%d kind:%q", p.OutDim, p.InDim, p.GroupSize, p.Bits, p.Kind)
	}
}

func TestAssembleAudioTextOnly(t *testing.T) {
	a, err := AssembleAudio(map[string]safetensors.Tensor{
		"model.layers.0.self_attn.q_proj.weight": audioBF16Tensor(4, 4),
	}, &Gemma4TextConfig{})
	if err != nil || a != nil {
		t.Fatalf("text-only pack should yield (nil,nil), got (%v, %v)", a, err)
	}
}
