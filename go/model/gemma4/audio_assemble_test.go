// SPDX-Licence-Identifier: EUPL-1.2

package gemma4

import (
	"bytes"
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

func TestAudioConv2dToOHWI(t *testing.T) {
	// Defect-catcher: real mlx checkpoints ship the subsample conv already in OHWI [outC,kh,kw,inC]
	// (layer0 [128,3,3,1], layer1 [32,3,3,128]). The assembler must pass them through byte-identical;
	// transposing an already-OHWI weight (the old OIHW assumption) scrambles it into garbage audio.
	for _, shape := range [][]int{{128, 3, 3, 1}, {32, 3, 3, 128}} {
		src := audioBF16Tensor(shape...)
		out, err := audioConv2dToOHWI(src)
		if err != nil {
			t.Fatalf("audioConv2dToOHWI(OHWI %v): %v", shape, err)
		}
		if !bytes.Equal(out, src.Data) {
			t.Fatalf("OHWI conv %v was transposed; want byte-identical pass-through", shape)
		}
	}

	// A synthetic torch OIHW [outC,inC,kh,kw] must still be transposed to OHWI.
	const outC, inC, kh, kw = 2, 4, 3, 3
	src := audioBF16Tensor(outC, inC, kh, kw)
	out, err := audioConv2dToOHWI(src)
	if err != nil {
		t.Fatalf("audioConv2dToOHWI(OIHW): %v", err)
	}
	// element (oc=1, ic=3, y=2, x=1): OIHW source index vs OHWI destination index.
	srcElem := ((1*inC+3)*kh+2)*kw + 1
	dstElem := ((1*kh+2)*kw+1)*inC + 3
	if out[dstElem*2] != src.Data[srcElem*2] || out[dstElem*2+1] != src.Data[srcElem*2+1] {
		t.Fatal("torch OIHW conv was not transposed to native OHWI")
	}
	if bytes.Equal(out, src.Data) {
		t.Fatal("torch OIHW conv unexpectedly passed through unchanged")
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

// audioF32Tensor builds a little-endian F32 tensor from float values — the second dtype
// audioF32Values decodes (clip bounds / per_dim_scale ship BF16 or F32).
func audioF32Tensor(vals ...float32) safetensors.Tensor {
	data := make([]byte, len(vals)*4)
	for i, v := range vals {
		bits := math.Float32bits(v)
		data[i*4] = byte(bits)
		data[i*4+1] = byte(bits >> 8)
		data[i*4+2] = byte(bits >> 16)
		data[i*4+3] = byte(bits >> 24)
	}
	return safetensors.Tensor{Dtype: "F32", Shape: []int{len(vals)}, Data: data}
}

// TestAudioF32Values covers audioF32Values across both accepted dtypes and their guards: a BF16
// tensor decodes to the folded float32s, an F32 tensor round-trips its exact values, a data slice
// shorter than the shape declares is rejected for each dtype, and an unsupported dtype returns
// (nil,false). This scalar/vector reader feeds every audio clip bound and per_dim_scale fold.
func TestAudioF32Values(t *testing.T) {
	// BF16: high byte carries the mantissa top — audioBF16Tensor(2) encodes bits 0x0000, 0x0100
	// (index i → low byte i, high byte i>>8), so element 1 is BF16 0x0100 = 2^-126 * ... just
	// assert the count + that decode succeeds rather than pin the tiny subnormal.
	bf, ok := audioF32Values(audioBF16Tensor(2, 2))
	if !ok || len(bf) != 4 {
		t.Fatalf("BF16 decode = (%v, %v), want 4 values, true", bf, ok)
	}

	// F32: exact round-trip of the little-endian bytes.
	f32, ok := audioF32Values(audioF32Tensor(1.5, -2.25, 0))
	if !ok || len(f32) != 3 || f32[0] != 1.5 || f32[1] != -2.25 || f32[2] != 0 {
		t.Fatalf("F32 decode = (%v, %v), want [1.5 -2.25 0], true", f32, ok)
	}

	// Short data: shape declares more elements than the byte slice holds → reject, both dtypes.
	shortBF := safetensors.Tensor{Dtype: "BF16", Shape: []int{4}, Data: make([]byte, 2)} // want 8 bytes
	if v, ok := audioF32Values(shortBF); ok || v != nil {
		t.Fatalf("short BF16 = (%v, %v), want (nil, false)", v, ok)
	}
	shortF32 := safetensors.Tensor{Dtype: "F32", Shape: []int{4}, Data: make([]byte, 4)} // want 16 bytes
	if v, ok := audioF32Values(shortF32); ok || v != nil {
		t.Fatalf("short F32 = (%v, %v), want (nil, false)", v, ok)
	}

	// Unsupported dtype (e.g. a quantised U32 clip tensor) → (nil, false), not a garbage decode.
	if v, ok := audioF32Values(safetensors.Tensor{Dtype: "U32", Shape: []int{2}, Data: make([]byte, 8)}); ok || v != nil {
		t.Fatalf("U32 decode = (%v, %v), want (nil, false)", v, ok)
	}
	t.Logf("audioF32Values: BF16 + F32 decode, short data rejected per dtype, unsupported dtype → (nil,false)")
}

// validAudioConfigForAssemble returns a Conformer-complete audio config: the base every branch of
// TestValidateGemma4AudioConfigForAssemble clones and mutates one field of.
func validAudioConfigForAssemble() *Gemma4AudioConfig {
	return &Gemma4AudioConfig{
		HiddenSize: 8, NumHiddenLayers: 1, NumAttentionHeads: 2,
		AttentionChunkSize: 2, AttentionContextLeft: 3, ConvKernelSize: 5,
		SubsamplingConvChannels: []int32{4, 2}, OutputProjDims: 6,
		ResidualWeight: 0.5, AttentionLogitCap: 50,
	}
}

// TestValidateGemma4AudioConfigForAssemble covers the encoder-config gate: a complete config
// passes, and each malformed shape the Conformer path cannot build is rejected — a zeroed
// dimensional field, a subsampling-channel list that isn't exactly two entries, a zero logit cap,
// and a hidden size the head count does not divide.
func TestValidateGemma4AudioConfigForAssemble(t *testing.T) {
	if err := validateGemma4AudioConfigForAssemble(validAudioConfigForAssemble()); err != nil {
		t.Fatalf("a complete audio config should validate, got %v", err)
	}

	cases := []struct {
		name   string
		mutate func(*Gemma4AudioConfig)
	}{
		{"hidden_size zero", func(c *Gemma4AudioConfig) { c.HiddenSize = 0 }},
		{"num_hidden_layers zero", func(c *Gemma4AudioConfig) { c.NumHiddenLayers = 0 }},
		{"num_attention_heads zero", func(c *Gemma4AudioConfig) { c.NumAttentionHeads = 0 }},
		{"attention_context_left zero", func(c *Gemma4AudioConfig) { c.AttentionContextLeft = 0 }},
		{"conv_kernel_size zero", func(c *Gemma4AudioConfig) { c.ConvKernelSize = 0 }},
		{"subsampling channels not two", func(c *Gemma4AudioConfig) { c.SubsamplingConvChannels = []int32{4} }},
		{"output_proj_dims zero", func(c *Gemma4AudioConfig) { c.OutputProjDims = 0 }},
		{"attention_logit_cap zero", func(c *Gemma4AudioConfig) { c.AttentionLogitCap = 0 }},
		{"hidden not divisible by heads", func(c *Gemma4AudioConfig) { c.HiddenSize = 9 }}, // 9 % 2 != 0
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			cfg := validAudioConfigForAssemble()
			tc.mutate(cfg)
			if err := validateGemma4AudioConfigForAssemble(cfg); err == nil {
				t.Fatalf("%s should be rejected", tc.name)
			}
		})
	}
	t.Logf("validateGemma4AudioConfigForAssemble: complete config passes; each malformed dim / channel-count / logit-cap / indivisible-hidden rejected")
}
