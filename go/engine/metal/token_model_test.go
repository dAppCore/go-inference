// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"bytes"
	"os"
	"slices"
	"testing"

	"dappco.re/go/inference/model"
	g4 "dappco.re/go/inference/model/gemma4"
)

func TestNativeTokenModelAcceptsImageInput_Good(t *testing.T) {
	tm := &NativeTokenModel{}
	if tm.AcceptsImageInput() {
		t.Fatal("AcceptsImageInput = true without a vision payload, want false")
	}
	tm.vision = &model.LoadedVision{}
	if !tm.AcceptsImageInput() {
		t.Fatal("AcceptsImageInput = false with a vision payload, want true")
	}
}

func TestNativeTokenModelAcceptsAudioInput_Good(t *testing.T) {
	tm := &NativeTokenModel{}
	if tm.AcceptsAudioInput() {
		t.Fatal("AcceptsAudioInput = true without an audio payload, want false")
	}
	tm.audio = &model.LoadedAudio{}
	if !tm.AcceptsAudioInput() {
		t.Fatal("AcceptsAudioInput = false with an audio payload, want true")
	}
}

func TestNativeTokenModelBlockDiffusionCapable_Good(t *testing.T) {
	tm := &NativeTokenModel{}
	bd, ok := any(tm).(interface{ BlockDiffusionCapable() bool })
	if !ok {
		t.Fatal("NativeTokenModel does not expose BlockDiffusionCapable")
	}
	if bd.BlockDiffusionCapable() {
		t.Fatal("BlockDiffusionCapable = true without a diffusion payload, want false")
	}
	tm.diffusion = &model.LoadedDiffusion{}
	if !bd.BlockDiffusionCapable() {
		t.Fatal("BlockDiffusionCapable = false with a diffusion payload, want true")
	}
}

func TestNativeVisionFromLoadedMapsPayload_Good(t *testing.T) {
	loaded := &model.LoadedVision{
		PatchEmbedding:     []byte{1, 2},
		PatchConvWeight:    []byte{31, 32},
		PositionEmbeddings: []byte{3, 4},
		PostLayernorm:      []byte{5, 6},
		StdBias:            []byte{7, 8},
		StdScale:           []byte{9, 10},
		Cfg: model.LoadedVisionConfig{
			Hidden: 64, PatchDim: 48, NumLayers: 1, NumHeads: 2, NumKVHeads: 1,
			HeadDim: 32, PatchSize: 4, NumChannels: 3, PositionEmbeddingSize: 16, RopeBase: 100, RMSNormEps: 1e-6, PoolKernel: 3,
			Standardize: true, EmbeddingScale: 8,
			ImageTokenID: 262145, ImageBeginToken: "<|image>", ImageToken: "<|image|>", ImageEndToken: "<image|>",
			VideoTokenID: 258884, VideoToken: "<|video|>",
		},
		Layers: []model.LoadedVisionLayer{{
			InputNorm:    []byte{11},
			PostAttnNorm: []byte{12},
			PreFFNorm:    []byte{13},
			PostFFNorm:   []byte{14},
			Q:            model.LoadedVisionLinear{Weight: []byte{15}, Bias: []byte{115}},
			K:            model.LoadedVisionLinear{Weight: []byte{16}, Bias: []byte{116}},
			V:            model.LoadedVisionLinear{Weight: []byte{17}, Bias: []byte{117}},
			O:            model.LoadedVisionLinear{Weight: []byte{18}, Bias: []byte{118}},
			QNorm:        []byte{19},
			KNorm:        []byte{20},
			Gate:         model.LoadedVisionLinear{Weight: []byte{21}, Bias: []byte{121}},
			Up:           model.LoadedVisionLinear{Weight: []byte{22}, Bias: []byte{122}},
			Down:         model.LoadedVisionLinear{Weight: []byte{23}, Bias: []byte{123}},
		}},
		Projector: model.LoadedVisionProjector{
			Projection: model.LoadedVisionLinear{Weight: []byte{24}, Bias: []byte{124}},
			Linear1:    model.LoadedVisionLinear{Weight: []byte{25}, Bias: []byte{125}},
			Linear2:    model.LoadedVisionLinear{Weight: []byte{26}, Bias: []byte{126}},
		},
	}

	weights, cfg, ok := nativeVisionFromLoaded(loaded)
	if !ok {
		t.Fatal("nativeVisionFromLoaded ok = false, want true")
	}
	if cfg.Hidden != 64 || cfg.PatchDim != 48 || cfg.NumLayers != 1 || cfg.NumHeads != 2 || cfg.NumKVHeads != 1 || cfg.HeadDim != 32 {
		t.Fatalf("native vision cfg = %+v, want loaded geometry", cfg)
	}
	if cfg.PatchSize != 4 || cfg.NumChannels != 3 || cfg.PositionEmbeddingSize != 16 || cfg.RopeBase != 100 || cfg.RMSNormEps != 1e-6 || cfg.PoolKernel != 3 || !cfg.Standardize || cfg.EmbeddingScale != 8 {
		t.Fatalf("native vision cfg extras = %+v, want loaded extras", cfg)
	}
	if cfg.ImageTokenID != 262145 || cfg.ImageBeginToken != "<|image>" || cfg.ImageToken != "<|image|>" || cfg.ImageEndToken != "<image|>" {
		t.Fatalf("native vision prompt metadata = %+v", cfg)
	}
	if cfg.VideoTokenID != 258884 || cfg.VideoToken != "<|video|>" {
		t.Fatalf("native vision video metadata = %+v", cfg)
	}
	if weights.PatchEmbedding[0] != 1 || weights.PatchConvWeight[0] != 31 || weights.PositionEmbeddings[0] != 3 || weights.PostLayernorm[0] != 5 ||
		weights.StdBias[0] != 7 || weights.StdScale[0] != 9 {
		t.Fatalf("native vision top-level weights = %+v", weights)
	}
	if len(weights.Layers) != 1 || weights.Layers[0].WQ[0] != 15 || weights.Layers[0].WK[0] != 16 ||
		weights.Layers[0].WV[0] != 17 || weights.Layers[0].WO[0] != 18 ||
		weights.Layers[0].WGate[0] != 21 || weights.Layers[0].WUp[0] != 22 || weights.Layers[0].WDown[0] != 23 {
		t.Fatalf("native vision layer weights = %+v", weights.Layers)
	}
	if weights.Layers[0].BQ[0] != 115 || weights.Layers[0].BK[0] != 116 || weights.Layers[0].BV[0] != 117 ||
		weights.Layers[0].BO[0] != 118 || weights.Layers[0].BGate[0] != 121 || weights.Layers[0].BUp[0] != 122 ||
		weights.Layers[0].BDown[0] != 123 {
		t.Fatalf("native vision layer biases = %+v", weights.Layers[0])
	}
	if weights.Projector.Projection.Weight[0] != 24 || weights.Projector.Linear1.Weight[0] != 25 || weights.Projector.Linear2.Weight[0] != 26 {
		t.Fatalf("native vision projector = %+v", weights.Projector)
	}
	if weights.Projector.Projection.Bias[0] != 124 || weights.Projector.Linear1.Bias[0] != 125 || weights.Projector.Linear2.Bias[0] != 126 {
		t.Fatalf("native vision projector biases = %+v", weights.Projector)
	}
	loaded.PatchEmbedding[0] = 99
	if weights.PatchEmbedding[0] != 99 {
		t.Fatal("native vision converter copied patch embedding, want no-copy alias")
	}
	loaded.PatchConvWeight[0] = 98
	if weights.PatchConvWeight[0] != 98 {
		t.Fatal("native vision converter copied patch conv weight, want no-copy alias")
	}
}

func TestNativeTokenModelImagePlaceholderBlock_Good(t *testing.T) {
	tm := &NativeTokenModel{vision: &model.LoadedVision{Cfg: model.LoadedVisionConfig{
		ImageTokenID: 262145, ImageBeginToken: "<|image>", ImageToken: "<|image|>", ImageEndToken: "<image|>",
		VideoTokenID: 258884, VideoToken: "<|video|>",
	}}}
	if got := tm.ImagePlaceholderTokenID(); got != 262145 {
		t.Fatalf("ImagePlaceholderTokenID = %d, want 262145", got)
	}
	if got := tm.ImagePlaceholderBlock(2); got != "<|image><|image|><|image|><image|>" {
		t.Fatalf("ImagePlaceholderBlock(2) = %q", got)
	}
	if got := tm.ImagePlaceholderBlock(0); got != "" {
		t.Fatalf("ImagePlaceholderBlock(0) = %q, want empty", got)
	}
	if got := tm.VideoPlaceholderTokenID(); got != 258884 {
		t.Fatalf("VideoPlaceholderTokenID = %d, want 258884", got)
	}
	if got := tm.VideoPlaceholderBlock(2); got != "<|image><|video|><|video|><image|>" {
		t.Fatalf("VideoPlaceholderBlock(2) = %q", got)
	}
	if got := tm.VideoPlaceholderBlock(0); got != "" {
		t.Fatalf("VideoPlaceholderBlock(0) = %q, want empty", got)
	}
}

func TestNativeAudioFromLoadedMapsPayload_Good(t *testing.T) {
	loaded := &model.LoadedAudio{
		Subsample: model.LoadedAudioSubsample{
			Conv0: []byte{1}, Norm0W: []byte{2, 0, 3, 0}, Norm0B: []byte{4, 0, 5, 0},
			Conv1: []byte{6}, Norm1W: []byte{7, 0}, Norm1B: []byte{8, 0},
			InputProj: model.LoadedAudioLinear{
				Weight: []byte{9},
				Clip: model.LoadedAudioClipPair{
					In: model.LoadedAudioClipBound{Min: -1, Max: 1, Present: true},
				},
			},
		},
		OutputProj: []byte{10, 0, 11, 0},
		Projector:  model.LoadedAudioLinear{Weight: []byte{12, 0}},
		Cfg: model.LoadedAudioConfig{
			Hidden: 8, FFInter: 16, Channels: 8, KernelSize: 5, Eps: 1e-6, Act: "silu",
			FFResidual: 0.5, ClipMin: -6, ClipMax: 6, NumHeads: 2, HeadDim: 4,
			ChunkSize: 3, PastHorizon: 2, FutureHorizon: 1, KScale: 0.5, LogitCap: 50,
			InvalidLogit: -1e9, OutputDim: 2, AudioTokenID: 77,
			AudioBeginToken: "<|audio>", AudioToken: "<|audio|>", AudioEndToken: "<audio|>",
		},
		Layers: []model.LoadedAudioLayer{{
			FF1: model.LoadedAudioFeedForward{
				PreNorm: []byte{13}, PostNorm: []byte{14},
				FFW1: model.LoadedAudioLinear{Weight: []byte{15}},
				FFW2: model.LoadedAudioLinear{Weight: []byte{16}},
			},
			FF2: model.LoadedAudioFeedForward{
				PreNorm: []byte{17}, PostNorm: []byte{18},
				FFW1: model.LoadedAudioLinear{Weight: []byte{19}},
				FFW2: model.LoadedAudioLinear{Weight: []byte{20}},
			},
			Attn: model.LoadedAudioAttention{
				Q: model.LoadedAudioLinear{Weight: []byte{21}},
				K: model.LoadedAudioLinear{Weight: []byte{22}},
				V: model.LoadedAudioLinear{Weight: []byte{23}},
				Post: model.LoadedAudioLinear{
					Weight: []byte{24},
					Clip: model.LoadedAudioClipPair{
						Out: model.LoadedAudioClipBound{Min: -2, Max: 2, Present: true},
					},
				},
				RelativeKProj: []byte{25},
				QScalePerDim:  []float32{0.5, 0.6, 0.7, 0.8},
				PosEmbed:      []float32{1, 2, 3, 4},
				PosCount:      1,
			},
			LConv: model.LoadedAudioLightConv{
				PreNorm: []byte{26}, ConvNorm: []byte{27},
				LinearStart:     model.LoadedAudioLinear{Weight: []byte{28}},
				LinearEnd:       model.LoadedAudioLinear{Weight: []byte{29}},
				DepthwiseWeight: []byte{30},
			},
			NormPreAttn:  []byte{31},
			NormPostAttn: []byte{32},
			NormOut:      []byte{33},
		}},
	}

	weights, cfg, projector, ok := nativeAudioFromLoaded(loaded, 24, 8)
	if !ok {
		t.Fatal("nativeAudioFromLoaded ok = false, want true")
	}
	if cfg.Hidden != 8 || cfg.FFInter != 16 || cfg.NumHeads != 2 || cfg.HeadDim != 4 || cfg.PastHorizon != 2 {
		t.Fatalf("native audio cfg = %+v, want loaded geometry", cfg)
	}
	if weights.SubsampleC.Frames != 24 || weights.SubsampleC.MelBins != 8 || weights.SubsampleC.OutC0 != 2 || weights.SubsampleC.OutC1 != 1 {
		t.Fatalf("native audio subsample cfg = %+v", weights.SubsampleC)
	}
	if weights.Subsample.InputProj[0] != 9 || !weights.Subsample.InputProjClip.In.Present {
		t.Fatalf("native audio subsample weights/clip = %+v", weights.Subsample)
	}
	if len(weights.Layers) != 1 || weights.Layers[0].Attn.QProj[0] != 21 || weights.Layers[0].Attn.PostClip.Out.Max != 2 ||
		weights.Layers[0].LConv.DepthwiseWeight[0] != 30 || weights.Layers[0].NormOut[0] != 33 {
		t.Fatalf("native audio layer weights = %+v", weights.Layers)
	}
	if projector.Weight[0] != 12 {
		t.Fatalf("projector = %+v, want loaded projector", projector)
	}
	loaded.OutputProj[0] = 99
	if weights.OutputProj[0] != 99 {
		t.Fatal("native audio converter copied output projection, want no-copy alias")
	}
}

func TestNativeTokenModelAudioPlaceholderBlock_Good(t *testing.T) {
	tm := &NativeTokenModel{audio: &model.LoadedAudio{Cfg: model.LoadedAudioConfig{
		AudioTokenID: 77, AudioBeginToken: "<|audio>", AudioToken: "<|audio|>", AudioEndToken: "<audio|>",
	}}}
	if got := tm.AudioPlaceholderTokenID(); got != 77 {
		t.Fatalf("AudioPlaceholderTokenID = %d, want 77", got)
	}
	if got := tm.AudioPlaceholderBlock(2); got != "<|audio><|audio|><|audio|><audio|>" {
		t.Fatalf("AudioPlaceholderBlock(2) = %q", got)
	}
	if got := tm.AudioSoftTokens(24); got != 6 {
		t.Fatalf("AudioSoftTokens(24) = %d, want 6", got)
	}
	if got := tm.AudioPlaceholderBlock(0); got != "" {
		t.Fatalf("AudioPlaceholderBlock(0) = %q, want empty", got)
	}
}

func TestNativeAudioProjectorNoScaleNormalisesRows_Good(t *testing.T) {
	rows := []float32{3, 4, 0, 2}
	got, err := nativeAudioProjector(rows, model.LoadedAudioLinear{}, 2, 0)
	if err != nil {
		t.Fatalf("nativeAudioProjector(no projection): %v", err)
	}
	want := append([]float32(nil), rows...)
	rmsNormVec(want[:2], nil, 0)
	rmsNormVec(want[2:], nil, 0)
	if !slices.Equal(got, f32ToBf16Slice(want)) {
		t.Fatalf("nativeAudioProjector(no projection) = %v, want no-scale RMS rows %v", bf16Floats(got), bf16Floats(f32ToBf16Slice(want)))
	}
}

func TestNativeAudioProjectorQuantizedRows_Good(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	const inDim, outDim, groupSize, bits = 64, 2, 64, 4
	projector := model.LoadedAudioLinear{
		Weight:    make([]byte, outDim*(inDim*bits/32)*4),
		Scales:    toBF16Bytes([]float32{1, 1}),
		Biases:    toBF16Bytes([]float32{0, 0}),
		OutDim:    outDim,
		InDim:     inDim,
		GroupSize: groupSize,
		Bits:      bits,
		Kind:      "affine",
	}
	got, err := nativeAudioProjector(syntheticFloat32(inDim, 5), projector, inDim, 1e-6)
	if err != nil {
		t.Fatalf("nativeAudioProjector(quant): %v", err)
	}
	if len(got) != outDim*bf16Size {
		t.Fatalf("quant projector bytes = %d, want %d", len(got), outDim*bf16Size)
	}
}

func TestNativeTokenModelInjectAudioFeatures_Good(t *testing.T) {
	const H = 8
	const audioTok = int32(77)
	tm := &NativeTokenModel{
		NativeBackend: &NativeBackend{arch: model.Arch{Hidden: H}},
		audio:         &model.LoadedAudio{Cfg: model.LoadedAudioConfig{AudioTokenID: int(audioTok)}},
	}
	tokenIDs := []int32{10, audioTok, 11, audioTok}
	emb := toBF16Bytes(syntheticFloat32(4*H, 3))
	feat := toBF16Bytes(syntheticFloat32(2*H, 7))
	got, err := tm.InjectAudioFeatures(emb, tokenIDs, feat)
	if err != nil {
		t.Fatalf("InjectAudioFeatures: %v", err)
	}
	g, e, f := bf16Floats(got), bf16Floats(emb), bf16Floats(feat)
	if !slices.Equal(g[1*H:2*H], f[0:H]) || !slices.Equal(g[3*H:4*H], f[1*H:2*H]) {
		t.Fatalf("audio rows were not spliced into placeholder slots: got=%v features=%v", g, f)
	}
	if !slices.Equal(g[0:H], e[0:H]) || !slices.Equal(g[2*H:3*H], e[2*H:3*H]) {
		t.Fatalf("ordinary token rows changed: got=%v embeddings=%v", g, e)
	}

	noAudio := &NativeTokenModel{NativeBackend: &NativeBackend{arch: model.Arch{Hidden: H}}}
	if _, err := noAudio.InjectAudioFeatures(emb, tokenIDs, feat); err == nil {
		t.Fatal("InjectAudioFeatures without audio payload error = nil")
	}
}

func TestNativeTokenModelInjectImageFeatures_Good(t *testing.T) {
	const H = 8
	const imageTok = int32(88)
	tm := &NativeTokenModel{
		NativeBackend: &NativeBackend{arch: model.Arch{Hidden: H}},
		vision:        &model.LoadedVision{Cfg: model.LoadedVisionConfig{ImageTokenID: imageTok}},
	}
	tokenIDs := []int32{10, imageTok, 11, imageTok}
	emb := toBF16Bytes(syntheticFloat32(4*H, 3))
	feat := toBF16Bytes(syntheticFloat32(2*H, 17))
	got, err := tm.InjectImageFeatures(emb, tokenIDs, feat)
	if err != nil {
		t.Fatalf("InjectImageFeatures: %v", err)
	}
	g, e, f := bf16Floats(got), bf16Floats(emb), bf16Floats(feat)
	if !slices.Equal(g[1*H:2*H], f[0:H]) || !slices.Equal(g[3*H:4*H], f[1*H:2*H]) {
		t.Fatalf("image rows were not spliced into placeholder slots: got=%v features=%v", g, f)
	}
	if !slices.Equal(g[0:H], e[0:H]) || !slices.Equal(g[2*H:3*H], e[2*H:3*H]) {
		t.Fatalf("ordinary token rows changed: got=%v embeddings=%v", g, e)
	}

	noVision := &NativeTokenModel{NativeBackend: &NativeBackend{arch: model.Arch{Hidden: H}}}
	if _, err := noVision.InjectImageFeatures(emb, tokenIDs, feat); err == nil {
		t.Fatal("InjectImageFeatures without vision payload error = nil")
	}
}

func TestNativeTokenModelInjectVideoFeatures_Good(t *testing.T) {
	const H = 8
	const videoTok = int32(99)
	tm := &NativeTokenModel{
		NativeBackend: &NativeBackend{arch: model.Arch{Hidden: H}},
		vision:        &model.LoadedVision{Cfg: model.LoadedVisionConfig{VideoTokenID: videoTok}},
	}
	tokenIDs := []int32{10, videoTok, 11, videoTok}
	emb := toBF16Bytes(syntheticFloat32(4*H, 5))
	feat := toBF16Bytes(syntheticFloat32(2*H, 23))
	got, err := tm.InjectVideoFeatures(emb, tokenIDs, feat)
	if err != nil {
		t.Fatalf("InjectVideoFeatures: %v", err)
	}
	g, e, f := bf16Floats(got), bf16Floats(emb), bf16Floats(feat)
	if !slices.Equal(g[1*H:2*H], f[0:H]) || !slices.Equal(g[3*H:4*H], f[1*H:2*H]) {
		t.Fatalf("video rows were not spliced into placeholder slots: got=%v features=%v", g, f)
	}
	if !slices.Equal(g[0:H], e[0:H]) || !slices.Equal(g[2*H:3*H], e[2*H:3*H]) {
		t.Fatalf("ordinary token rows changed: got=%v embeddings=%v", g, e)
	}

	noVision := &NativeTokenModel{NativeBackend: &NativeBackend{arch: model.Arch{Hidden: H}}}
	if _, err := noVision.InjectVideoFeatures(emb, tokenIDs, feat); err == nil {
		t.Fatal("InjectVideoFeatures without vision payload error = nil")
	}
}

func TestNativeTokenModelTokenEmbeddingsWithFeatures_Good(t *testing.T) {
	const H = 4
	const imageTok = int32(88)
	const audioTok = int32(77)
	const videoTok = int32(99)
	tokenIDs := []int32{12, imageTok, 13, audioTok, videoTok, 14}
	tm := &NativeTokenModel{
		NativeBackend: &NativeBackend{arch: model.Arch{Hidden: H}},
		vision: &model.LoadedVision{Cfg: model.LoadedVisionConfig{
			ImageTokenID: imageTok,
			VideoTokenID: videoTok,
		}},
		audio: &model.LoadedAudio{Cfg: model.LoadedAudioConfig{AudioTokenID: int(audioTok)}},
		embedInto: func(dst []byte, id int32) ([]byte, error) {
			row := make([]float32, H)
			for i := range row {
				row[i] = float32(id) + float32(i)/10
			}
			copy(dst, toBF16Bytes(row))
			return dst, nil
		},
	}
	imageFeatures := toBF16Bytes([]float32{101, 102, 103, 104})
	audioFeatures := toBF16Bytes([]float32{201, 202, 203, 204})
	videoFeatures := toBF16Bytes([]float32{301, 302, 303, 304})

	rows, err := tm.TokenEmbeddingsWithFeatures(tokenIDs, imageFeatures, audioFeatures, videoFeatures)
	if err != nil {
		t.Fatalf("TokenEmbeddingsWithFeatures: %v", err)
	}
	if len(rows) != len(tokenIDs) {
		t.Fatalf("row count = %d, want %d", len(rows), len(tokenIDs))
	}
	for i, row := range rows {
		if len(row) != H*bf16Size {
			t.Fatalf("row %d bytes = %d, want %d", i, len(row), H*bf16Size)
		}
	}
	if got, want := bf16Floats(rows[1]), bf16Floats(imageFeatures); !slices.Equal(got, want) {
		t.Fatalf("image row = %v, want %v", got, want)
	}
	if got, want := bf16Floats(rows[3]), bf16Floats(audioFeatures); !slices.Equal(got, want) {
		t.Fatalf("audio row = %v, want %v", got, want)
	}
	if got, want := bf16Floats(rows[4]), bf16Floats(videoFeatures); !slices.Equal(got, want) {
		t.Fatalf("video row = %v, want %v", got, want)
	}
	ordinaryWant := bf16Floats(toBF16Bytes([]float32{12, 12.1, 12.2, 12.3}))
	if got, want := bf16Floats(rows[0]), ordinaryWant; !slices.Equal(got, want) {
		t.Fatalf("ordinary row = %v, want %v", got, want)
	}

	noEmbedding := *tm
	noEmbedding.embedInto = nil
	if _, err := noEmbedding.TokenEmbeddingsWithFeatures(tokenIDs, nil, nil, nil); err == nil {
		t.Fatal("TokenEmbeddingsWithFeatures without embedding bookend error = nil")
	}
}

func TestNativeTokenModelProjectImageFeatures_Good(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	tm := &NativeTokenModel{vision: &model.LoadedVision{
		PatchEmbedding: toBF16Bytes([]float32{1, 0, 0, 1}),
		Cfg: model.LoadedVisionConfig{
			Hidden: 2, PatchDim: 2, NumHeads: 1, NumKVHeads: 1, HeadDim: 2,
			RMSNormEps: 1e-6, PoolKernel: 1,
		},
	}}
	got, err := tm.ProjectImageFeatures(toBF16Bytes([]float32{0.75, 0.25, 0.25, 0.75}))
	if err != nil {
		t.Fatalf("ProjectImageFeatures: %v", err)
	}
	if len(got) != 2*2*2 {
		t.Fatalf("projected feature bytes = %d, want 8", len(got))
	}
}

func TestNativeTokenModelProjectImagePixelsNHWC_Good(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	tm := &NativeTokenModel{vision: &model.LoadedVision{
		PatchConvWeight: toBF16Bytes([]float32{
			1, 1,
			1, 1,
			1, 0,
			0, 0,
		}),
		Cfg: model.LoadedVisionConfig{
			Hidden: 2, PatchDim: 4, PatchSize: 2, NumChannels: 1, NumHeads: 1, NumKVHeads: 1, HeadDim: 2,
			RMSNormEps: 1e-6, PoolKernel: 1,
		},
	}}
	got, err := tm.ProjectImagePixels([]float32{
		1.0, 0.5,
		0.0, 0.25,
	}, 2, 2)
	if err != nil {
		t.Fatalf("ProjectImagePixels: %v", err)
	}
	if len(got) != 2*bf16Size {
		t.Fatalf("projected raw feature bytes = %d, want %d", len(got), 2*bf16Size)
	}
}

// TestNativeTokenModel_ContractParity gates the token-loop CONTRACT against the
// proven native generation loop: model.Generate over a NativeTokenModel
// (whole-sequence decode through model.Backend + the embed/head bookends) must
// produce the EXACT greedy tokens GenerateBF16 produces (native's
// incremental persistent-cache loop) on the same bf16 gemma4. The two loops
// share no code — one is the contract loop in pkg/model, the other native's
// bespoke loop — so full-sequence equality proves the contract path yields real
// tokens identical to the path it generalises. The surface pkg/rocm drops into
// is proven, not asserted.
func TestNativeTokenModel_ContractParity(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	const dModel, nHeads, nKV, headDim, dFF, vocab = 128, 2, 1, 64, 256, 32
	arch, err := g4.Config{
		HiddenSize: dModel, NumHiddenLayers: 2, IntermediateSize: dFF,
		NumAttentionHeads: nHeads, NumKeyValueHeads: nKV, HeadDim: headDim,
		VocabSize: vocab, RMSNormEps: 1e-6,
	}.Arch()
	if err != nil {
		t.Fatalf("Arch: %v", err)
	}
	mk := func(n, salt int) []float32 {
		s := make([]float32, n)
		for i := range s {
			s[i] = float32((i*salt+13)%97-48) * 0.02
		}
		return s
	}
	layers := make([]DecodeLayerWeights, len(arch.Layer))
	for li := range layers {
		layers[li] = forwardLayer(dModel, nHeads, nKV, headDim, dFF, (li+1)*100)
	}
	g := &BF16Model{
		Layers:    layers,
		Embed:     toBF16Bytes(mk(vocab*dModel, 11)),
		FinalNorm: toBF16Bytes(mk(dModel, 7)),
	}
	g.LMHead, g.Tied = g.Embed, true // tied head

	prompt := []int32{1, 5, 3, 9}
	const maxNew, maxLen = 6, 16

	// reference: native's proven incremental (persistent-cache) generation loop.
	want, err := GenerateBF16(g, arch, prompt, maxNew, maxLen, -1)
	if err != nil {
		t.Fatalf("GenerateBF16: %v", err)
	}

	// the contract path: model.Generate over the NativeTokenModel (whole-seq).
	tm, err := NewBF16TokenModel(g, arch, maxLen)
	if err != nil {
		t.Fatalf("NewBF16TokenModel: %v", err)
	}
	got, err := model.Generate(tm, prompt, maxNew, -1)
	if err != nil {
		t.Fatalf("model.Generate: %v", err)
	}
	if len(got) != len(want) {
		t.Fatalf("contract generated %d tokens, want %d (%v vs %v)", len(got), len(want), got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("contract token %d = %d, native loop = %d (full: %v vs %v)", i, got[i], want[i], got, want)
		}
	}

	// whole-seq reference via the SAME model's contract pieces (tm.NativeBackend's
	// DecodeForward is the whole-sequence fallback): the incremental result must be
	// output-identical to the path it supersedes — the additive refinement changes
	// speed, not tokens.
	seq := make([][]byte, 0, len(prompt)+maxNew)
	for _, id := range prompt {
		e, eerr := tm.Embed(id)
		if eerr != nil {
			t.Fatalf("Embed: %v", eerr)
		}
		seq = append(seq, e)
	}
	var wholeSeq []int32
	for len(wholeSeq) < maxNew {
		hs, derr := tm.NativeBackend.DecodeForward(seq)
		if derr != nil {
			t.Fatalf("whole-seq DecodeForward: %v", derr)
		}
		logits, herr := tm.Head(hs[len(hs)-1])
		if herr != nil {
			t.Fatalf("Head: %v", herr)
		}
		nx, gerr := model.Greedy(logits, vocab)
		if gerr != nil {
			t.Fatalf("Greedy: %v", gerr)
		}
		wholeSeq = append(wholeSeq, nx)
		if len(wholeSeq) >= maxNew {
			break
		}
		e, eerr := tm.Embed(nx)
		if eerr != nil {
			t.Fatalf("Embed: %v", eerr)
		}
		seq = append(seq, e)
	}
	for i := range want {
		if wholeSeq[i] != want[i] {
			t.Fatalf("whole-seq token %d = %d, want %d (incremental %v vs whole-seq %v)", i, wholeSeq[i], want[i], got, wholeSeq)
		}
	}

	// the contract Vocab() reports the logit width Greedy reads.
	if tm.Vocab() != vocab {
		t.Fatalf("Vocab() = %d, want %d", tm.Vocab(), vocab)
	}

	// zero-temp sampled generation falls back to greedy → same sequence.
	sampled, err := model.GenerateSampled(tm, model.NewSampler(7), model.SampleParams{Temperature: 0}, prompt, maxNew, -1)
	if err != nil {
		t.Fatalf("GenerateSampled: %v", err)
	}
	for i := range want {
		if sampled[i] != want[i] {
			t.Fatalf("zero-temp sampled token %d = %d, want %d (%v)", i, sampled[i], want[i], sampled)
		}
	}

	t.Logf("token-loop contract (incremental session) ≡ native generation ≡ whole-seq: model.Generate(NativeTokenModel) = GenerateBF16 = %v", got)
}

func TestNativeTokenModelTopologyCapabilities(t *testing.T) {
	arch := model.Arch{
		Hidden:        32,
		Heads:         16,
		SlidingWindow: 4096,
		Layer: []model.LayerSpec{
			{Attention: model.SlidingAttention, KVShareFrom: 0, CacheIndex: 0},
			{Attention: model.GlobalAttention, KVShareFrom: 1, CacheIndex: 1},
			{Attention: model.SlidingAttention, KVShareFrom: 0, CacheIndex: -1},
		},
	}
	tm := &NativeTokenModel{NativeBackend: &NativeBackend{arch: arch}}

	if got := tm.NumLayers(); got != len(arch.Layer) {
		t.Fatalf("NumLayers() = %d, want %d", got, len(arch.Layer))
	}
	if got := tm.NumQueryHeads(); got != arch.Heads {
		t.Fatalf("NumQueryHeads() = %d, want %d", got, arch.Heads)
	}
	reporter, ok := any(tm).(interface {
		HiddenSize() int
		QuantBits() int
		QuantGroup() int
	})
	if !ok {
		t.Fatal("NativeTokenModel does not expose metadata reporter methods")
	}
	if got := reporter.HiddenSize(); got != arch.Hidden {
		t.Fatalf("HiddenSize() = %d, want %d", got, arch.Hidden)
	}
	if reporter.QuantBits() != 0 || reporter.QuantGroup() != 0 {
		t.Fatalf("bf16 quant metadata = %d/%d, want 0/0", reporter.QuantBits(), reporter.QuantGroup())
	}
	if !tm.UsesFixedSlidingCache() {
		t.Fatal("UsesFixedSlidingCache() = false, want true for sliding-window arch")
	}
	if !tm.NeedsThoughtChannelSuppressor() {
		t.Fatal("NeedsThoughtChannelSuppressor() = false, want true at 16 query heads")
	}
	if got, want := tm.AttentionCacheLayout(3, 2), []int{0, 1, 0}; !slices.Equal(got, want) {
		t.Fatalf("AttentionCacheLayout() = %v, want %v", got, want)
	}
	if got, want := tm.AttentionCacheLayout(4, 1), []int{0, -1, 0, -1}; !slices.Equal(got, want) {
		t.Fatalf("AttentionCacheLayout(capped caches) = %v, want %v", got, want)
	}

	dense := &NativeTokenModel{NativeBackend: &NativeBackend{arch: model.Arch{Heads: 8}}}
	if dense.UsesFixedSlidingCache() {
		t.Fatal("dense UsesFixedSlidingCache() = true, want false")
	}
	if dense.NeedsThoughtChannelSuppressor() {
		t.Fatal("dense NeedsThoughtChannelSuppressor() = true, want false below 16 query heads")
	}

	quant := &NativeTokenModel{NativeBackend: &NativeBackend{arch: arch}, quantBits: 4, quantGroup: 64}
	if quant.QuantBits() != 4 || quant.QuantGroup() != 64 {
		t.Fatalf("quant metadata = %d/%d, want 4/64", quant.QuantBits(), quant.QuantGroup())
	}
}

func TestNativeBF16TokenModelEmbedSingleTokenAllocationBudget(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	const dModel, nHeads, nKV, headDim, dFF, vocab = 128, 2, 1, 64, 256, 32
	arch, err := g4.Config{
		HiddenSize: dModel, NumHiddenLayers: 1, IntermediateSize: dFF,
		NumAttentionHeads: nHeads, NumKeyValueHeads: nKV, HeadDim: headDim,
		VocabSize: vocab, RMSNormEps: 1e-6,
	}.Arch()
	if err != nil {
		t.Fatalf("Arch: %v", err)
	}
	layers := []DecodeLayerWeights{forwardLayer(dModel, nHeads, nKV, headDim, dFF, 100)}
	g := &BF16Model{
		Layers:    layers,
		Embed:     toBF16Bytes(syntheticFloat32(vocab*dModel, 11)),
		FinalNorm: toBF16Bytes(syntheticFloat32(dModel, 7)),
	}
	g.LMHead, g.Tied = g.Embed, true
	tm, err := NewBF16TokenModel(g, arch, 16)
	if err != nil {
		t.Fatalf("NewBF16TokenModel: %v", err)
	}

	var embedErr error
	allocs := testing.AllocsPerRun(10, func() {
		_, embedErr = tm.Embed(3)
	})
	if embedErr != nil {
		t.Fatalf("Embed: %v", embedErr)
	}
	if allocs > 1 {
		t.Fatalf("Embed allocations = %.0f, want <= 1", allocs)
	}
}

func TestNativeBF16TokenModelEmbedIntoNoAllocation(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	const dModel, nHeads, nKV, headDim, dFF, vocab = 128, 2, 1, 64, 256, 32
	arch, err := g4.Config{
		HiddenSize: dModel, NumHiddenLayers: 1, IntermediateSize: dFF,
		NumAttentionHeads: nHeads, NumKeyValueHeads: nKV, HeadDim: headDim,
		VocabSize: vocab, RMSNormEps: 1e-6,
	}.Arch()
	if err != nil {
		t.Fatalf("Arch: %v", err)
	}
	layers := []DecodeLayerWeights{forwardLayer(dModel, nHeads, nKV, headDim, dFF, 100)}
	g := &BF16Model{
		Layers:    layers,
		Embed:     toBF16Bytes(syntheticFloat32(vocab*dModel, 11)),
		FinalNorm: toBF16Bytes(syntheticFloat32(dModel, 7)),
	}
	g.LMHead, g.Tied = g.Embed, true
	tm, err := NewBF16TokenModel(g, arch, 16)
	if err != nil {
		t.Fatalf("NewBF16TokenModel: %v", err)
	}
	embedInto, ok := any(tm).(interface {
		EmbedInto([]byte, int32) ([]byte, error)
	})
	if !ok {
		t.Fatal("NativeTokenModel does not expose EmbedInto")
	}
	if got, want := tm.EmbeddingBytes(), dModel*bf16Size; got != want {
		t.Fatalf("EmbeddingBytes = %d, want %d", got, want)
	}
	want, err := tm.Embed(3)
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}
	dst := make([]byte, len(want))
	got, err := embedInto.EmbedInto(dst, 3)
	if err != nil {
		t.Fatalf("EmbedInto: %v", err)
	}
	if len(got) == 0 || &got[0] != &dst[0] {
		t.Fatal("EmbedInto did not return caller-owned destination")
	}
	if !bytes.Equal(got, want) {
		t.Fatal("EmbedInto output differs from Embed")
	}
	var embedErr error
	allocs := testing.AllocsPerRun(10, func() {
		_, embedErr = embedInto.EmbedInto(dst, 3)
	})
	if embedErr != nil {
		t.Fatalf("EmbedInto allocation run: %v", embedErr)
	}
	if allocs > 0 {
		t.Fatalf("EmbedInto allocations = %.0f, want 0", allocs)
	}
}

func TestNativeBF16TokenModelUsesResidentHead(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	const dModel, nHeads, nKV, headDim, dFF, vocab = 128, 2, 1, 64, 256, 128
	arch, err := g4.Config{
		HiddenSize: dModel, NumHiddenLayers: 1, IntermediateSize: dFF,
		NumAttentionHeads: nHeads, NumKeyValueHeads: nKV, HeadDim: headDim,
		VocabSize: vocab, RMSNormEps: 1e-6,
	}.Arch()
	if err != nil {
		t.Fatalf("Arch: %v", err)
	}
	layers := []DecodeLayerWeights{forwardLayer(dModel, nHeads, nKV, headDim, dFF, 100)}
	g := &BF16Model{
		Layers:    layers,
		Embed:     toBF16Bytes(syntheticFloat32(vocab*dModel, 11)),
		FinalNorm: toBF16Bytes(syntheticFloat32(dModel, 7)),
	}
	g.LMHead, g.Tied = g.Embed, true
	tm, err := NewBF16TokenModel(g, arch, 16)
	if err != nil {
		t.Fatalf("NewBF16TokenModel: %v", err)
	}
	if tm.headEnc == nil {
		t.Fatal("NewBF16TokenModel did not bind a resident LM head")
	}
	hidden := toBF16Bytes(syntheticFloat32(dModel, 5))
	got, err := tm.Head(hidden)
	if err != nil {
		t.Fatalf("Head: %v", err)
	}
	want, err := LMHeadBF16(hidden, g.FinalNorm, g.LMHead, dModel, vocab, arch.Eps, arch.SoftCap)
	if err != nil {
		t.Fatalf("LMHeadBF16: %v", err)
	}
	if string(got) != string(want) {
		t.Fatal("resident token-model head differs from LMHeadBF16")
	}
	stepper, err := tm.OpenSession()
	if err != nil {
		t.Fatalf("OpenSession: %v", err)
	}
	sess, ok := stepper.(*ArchSession)
	if !ok {
		t.Fatalf("OpenSession returned %T, want *ArchSession", stepper)
	}
	if sess.headEnc != tm.headEnc {
		t.Fatal("OpenSession rebuilt the resident LM head instead of reusing the token model head")
	}
}

func TestNativeTokenModelOpenSessionHonoursPagedKVLoadOptions_Good(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	const dModel, nHeads, nKV, headDim, dFF, vocab = 128, 2, 1, 64, 256, 128
	const maxLen, pageSize = 16, 2
	arch, err := g4.Config{
		HiddenSize: dModel, NumHiddenLayers: 1, IntermediateSize: dFF,
		NumAttentionHeads: nHeads, NumKeyValueHeads: nKV, HeadDim: headDim,
		VocabSize: vocab, RMSNormEps: 1e-6,
	}.Arch()
	if err != nil {
		t.Fatalf("Arch: %v", err)
	}
	g := &BF16Model{
		Layers:    []DecodeLayerWeights{forwardLayer(dModel, nHeads, nKV, headDim, dFF, 101)},
		Embed:     toBF16Bytes(syntheticFloat32(vocab*dModel, 103)),
		FinalNorm: toBF16Bytes(syntheticFloat32(dModel, 107)),
	}
	g.LMHead, g.Tied = g.Embed, true

	tm, err := NewBF16TokenModel(g, arch, maxLen, withPagedKVPageSize(pageSize), withPagedKVPrealloc(true))
	if err != nil {
		t.Fatalf("NewBF16TokenModel: %v", err)
	}
	stepper, err := tm.OpenSession()
	if err != nil {
		t.Fatalf("OpenSession: %v", err)
	}
	sess, ok := stepper.(*ArchSession)
	if !ok {
		t.Fatalf("OpenSession returned %T, want *ArchSession", stepper)
	}
	if len(sess.state.pagedKV) != 1 || sess.state.pagedKV[0] == nil {
		t.Fatalf("paged KV caches = %d, want one owner cache", len(sess.state.pagedKV))
	}
	cache := sess.state.pagedKV[0]
	if cache.pageSize != pageSize {
		t.Fatalf("paged KV page size = %d, want %d", cache.pageSize, pageSize)
	}
	if got, want := len(cache.kPages), cache.pageForPos(maxLen-1)+1; got != want {
		t.Fatalf("preallocated K pages = %d, want %d (geometric schedule)", got, want)
	}
	if cache.length != 0 || cache.pageLens[0] != 0 {
		t.Fatalf("preallocated cache visible length/page = %d/%d, want 0/0", cache.length, cache.pageLens[0])
	}
}

func TestNativeQuantTokenModelEmbedSingleTokenAllocationBudget(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	const gs, bits = 32, 4
	cfg := g4.Config{
		HiddenSize: 128, NumHiddenLayers: 1, IntermediateSize: 256,
		NumAttentionHeads: 2, NumKeyValueHeads: 1, HeadDim: 64, VocabSize: 32, RMSNormEps: 1e-6,
		Quantization: &model.QuantConfig{GroupSize: gs, Bits: bits},
	}
	arch, err := cfg.Arch()
	if err != nil {
		t.Fatalf("Arch: %v", err)
	}
	ts := quantGemma4Tensors(t, arch, gs, bits)
	lm, err := model.Assemble(ts, arch, model.StandardWeightNames())
	if err != nil {
		t.Fatalf("model.Assemble: %v", err)
	}
	g, err := loadedToQuant(lm, gs, bits)
	if err != nil {
		t.Fatalf("loadedToQuant: %v", err)
	}
	tm, err := NewQuantTokenModel(g, arch, 16)
	if err != nil {
		t.Fatalf("NewQuantTokenModel: %v", err)
	}

	var embedErr error
	allocs := testing.AllocsPerRun(10, func() {
		_, embedErr = tm.Embed(3)
	})
	if embedErr != nil {
		t.Fatalf("Embed: %v", embedErr)
	}
	if allocs > 1 {
		t.Fatalf("quant Embed allocations = %.0f, want <= 1", allocs)
	}
}

func TestNativeQuantTokenModelEmbedIntoNoAllocation(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	const gs, bits = 32, 4
	cfg := g4.Config{
		HiddenSize: 128, NumHiddenLayers: 1, IntermediateSize: 256,
		NumAttentionHeads: 2, NumKeyValueHeads: 1, HeadDim: 64, VocabSize: 32, RMSNormEps: 1e-6,
		Quantization: &model.QuantConfig{GroupSize: gs, Bits: bits},
	}
	arch, err := cfg.Arch()
	if err != nil {
		t.Fatalf("Arch: %v", err)
	}
	ts := quantGemma4Tensors(t, arch, gs, bits)
	lm, err := model.Assemble(ts, arch, model.StandardWeightNames())
	if err != nil {
		t.Fatalf("model.Assemble: %v", err)
	}
	g, err := loadedToQuant(lm, gs, bits)
	if err != nil {
		t.Fatalf("loadedToQuant: %v", err)
	}
	tm, err := NewQuantTokenModel(g, arch, 16)
	if err != nil {
		t.Fatalf("NewQuantTokenModel: %v", err)
	}
	embedInto, ok := any(tm).(interface {
		EmbedInto([]byte, int32) ([]byte, error)
	})
	if !ok {
		t.Fatal("NativeTokenModel does not expose EmbedInto")
	}
	if got, want := tm.EmbeddingBytes(), arch.Hidden*bf16Size; got != want {
		t.Fatalf("EmbeddingBytes = %d, want %d", got, want)
	}
	want, err := tm.Embed(3)
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}
	dst := make([]byte, len(want))
	got, err := embedInto.EmbedInto(dst, 3)
	if err != nil {
		t.Fatalf("EmbedInto: %v", err)
	}
	if len(got) == 0 || &got[0] != &dst[0] {
		t.Fatal("EmbedInto did not return caller-owned destination")
	}
	if !bytes.Equal(got, want) {
		t.Fatal("EmbedInto output differs from Embed")
	}
	var embedErr error
	allocs := testing.AllocsPerRun(10, func() {
		_, embedErr = embedInto.EmbedInto(dst, 3)
	})
	if embedErr != nil {
		t.Fatalf("EmbedInto allocation run: %v", embedErr)
	}
	if allocs > 0 {
		t.Fatalf("EmbedInto allocations = %.0f, want 0", allocs)
	}
}

// TestNativeTokenModel_QuantContractParity is the 4-bit sibling: model.Generate
// over a quant NativeTokenModel (whole-sequence DecodeForwardArchQuant + the
// quant embed/head bookends) must produce the EXACT greedy tokens
// NewArchQuantSession produces (native's incremental quant loop) on the same
// synthetic 4-bit gemma4. The model is all-global, so the session's per-type
// RoPE coincides with the whole-seq one base — and the two independent loops
// agree token-for-token, proving the contract covers the serving quant too.
func TestNativeTokenModel_QuantContractParity(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	const gs, bits = 32, 4
	const maxLen, n = 16, 6
	cfg := g4.Config{
		HiddenSize: 128, NumHiddenLayers: 2, IntermediateSize: 256,
		NumAttentionHeads: 2, NumKeyValueHeads: 1, HeadDim: 64, VocabSize: 32, RMSNormEps: 1e-6,
		Quantization: &model.QuantConfig{GroupSize: gs, Bits: bits},
	}
	arch, err := cfg.Arch()
	if err != nil {
		t.Fatalf("Arch: %v", err)
	}
	ts := quantGemma4Tensors(t, arch, gs, bits)
	lm, err := model.Assemble(ts, arch, model.StandardWeightNames())
	if err != nil {
		t.Fatalf("model.Assemble: %v", err)
	}
	g, err := loadedToQuant(lm, gs, bits)
	if err != nil {
		t.Fatalf("loadedToQuant: %v", err)
	}
	prompt := []int32{1, 5, 3}

	// reference: native's proven incremental (persistent-cache) quant loop.
	sess, err := NewArchQuantSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("NewArchQuantSession: %v", err)
	}
	want, err := sess.Generate(prompt, n, -1)
	if err != nil {
		t.Fatalf("quant session Generate: %v", err)
	}

	// the contract path: model.Generate over the quant NativeTokenModel (whole-seq).
	tm, err := NewQuantTokenModel(g, arch, maxLen)
	if err != nil {
		t.Fatalf("NewQuantTokenModel: %v", err)
	}
	got, err := model.Generate(tm, prompt, n, -1)
	if err != nil {
		t.Fatalf("model.Generate (quant): %v", err)
	}
	if len(got) != len(want) {
		t.Fatalf("quant contract generated %d tokens, want %d (%v vs %v)", len(got), len(want), got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("quant contract token %d = %d, native session = %d (full: %v vs %v)", i, got[i], want[i], got, want)
		}
	}
	t.Logf("4-bit token-loop contract ≡ native quant session: model.Generate(NewQuantTokenModel) = %v", got)
}

// TestNativeTokenModel_PLEContractParity gates E2B/E4B (per-layer-input) decode
// THROUGH the contract: model.Generate over a quant NativeTokenModel must produce
// the exact tokens NewArchQuantSession produces (native's PLE generation loop) on
// the same synthetic PLE model — proving the contract drives the per-layer-input
// tower via the id-aware StepWithID (the per-layer inputs are gathered from the
// token id, which the plain embeddings-only Step can't supply). The whole-sequence
// DecodeForward fallback correctly refuses a PLE model.
func TestNativeTokenModel_PLEContractParity(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	const dModel, nHeads, nKV, headDim, dFF, vocab = 128, 2, 1, 64, 256, 32
	const numLayers, pliDim, gs, bits = 2, 64, 64, 4
	const maxLen, n = 16, 5
	cfg := g4.Config{
		HiddenSize: dModel, NumHiddenLayers: numLayers, IntermediateSize: dFF,
		NumAttentionHeads: nHeads, NumKeyValueHeads: nKV, HeadDim: headDim, VocabSize: vocab, RMSNormEps: 1e-6,
		HiddenSizePerLayerInput: pliDim, VocabSizePerLayerInput: vocab,
		Quantization: &model.QuantConfig{GroupSize: gs, Bits: bits},
	}
	arch, err := cfg.Arch()
	if err != nil {
		t.Fatalf("Arch: %v", err)
	}
	ts := quantGemma4Tensors(t, arch, gs, bits)
	addPLETensors(t, ts, arch, gs, bits)
	lm, err := model.Assemble(ts, arch, model.StandardWeightNames())
	if err != nil {
		t.Fatalf("model.Assemble: %v", err)
	}
	g, err := loadedToQuant(lm, gs, bits)
	if err != nil {
		t.Fatalf("loadedToQuant: %v", err)
	}
	if !g.HasPLE() {
		t.Fatal("assembled model should have the per-layer-input tower")
	}
	prompt := []int32{1, 5, 3}

	// reference: native's PLE generation loop.
	sess, err := NewArchQuantSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("NewArchQuantSession: %v", err)
	}
	want, err := sess.Generate(prompt, n, -1)
	if err != nil {
		t.Fatalf("quant PLE session Generate: %v", err)
	}

	// contract: model.Generate over the quant token model — the incremental session
	// + StepWithID thread the per-layer inputs each token.
	tm, err := NewQuantTokenModel(g, arch, maxLen)
	if err != nil {
		t.Fatalf("NewQuantTokenModel: %v", err)
	}
	got, err := model.Generate(tm, prompt, n, -1)
	if err != nil {
		t.Fatalf("model.Generate (PLE): %v", err)
	}
	if len(got) != len(want) {
		t.Fatalf("PLE contract generated %d tokens, want %d (%v vs %v)", len(got), len(want), got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("PLE contract token %d = %d, native PLE session = %d (%v vs %v)", i, got[i], want[i], got, want)
		}
	}

	// the whole-seq fallback correctly refuses a PLE model (no token ids to gather
	// the per-layer inputs from).
	if _, derr := tm.NativeBackend.DecodeForward([][]byte{make([]byte, dModel*2)}); derr == nil {
		t.Fatal("whole-seq DecodeForward should reject a PLE model")
	}
	t.Logf("E2B/E4B PLE through the contract: model.Generate(NewQuantTokenModel) = NewArchQuantSession = %v", got)
}
