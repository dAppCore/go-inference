// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import (
	"context"
	"iter"
	"slices"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/inference/engine"
	sharedmodel "dappco.re/go/inference/model"
)

type hipKVCaptureTokenModel struct{}

func (hipKVCaptureTokenModel) OpenEngineSession() (engine.Session, error) {
	return hipEngineSessionForTest(nil), nil
}

func (hipKVCaptureTokenModel) Close() error { return nil }

func hipKVCaptureModelForTest() *rocmModel {
	tok := hipInferenceModelFixtureTokenizer()
	shared := engine.NewTextModel(hipKVCaptureTokenModel{}, tok, "gemma4", inference.ModelInfo{Architecture: "gemma4"}, 32)
	return &rocmModel{engineModel: shared}
}

func hipInferenceModelFixtureTokenizer() *hipTokenTextDecoder {
	decoder := &hipTokenTextDecoder{
		vocab:       map[string]int32{"<bos>": 2, "<|turn>": 105, "<turn|>": 106, "hello": 7},
		pieces:      map[int32]string{2: "<bos>", 105: "<|turn>", 106: "<turn|>", 7: "hello"},
		special:     map[int32]bool{2: true, 105: true, 106: true},
		specialText: map[string]int32{"<bos>": 2, "<|turn>": 105, "<turn|>": 106},
		bosID:       2,
		hasBOS:      true,
		eosID:       106,
		hasEOS:      true,
		mergeRanks:  map[string]int{"h e": 0, "he l": 1, "hel l": 2, "hell o": 3},
	}
	decoder.mergePairRanks = hipTokenTextMergePairRanks(decoder.mergeRanks)
	decoder.precomputeDecodedPieces()
	return decoder
}

func TestHIPInferenceModel_EmbeddedTokenizer_Good(t *testing.T) {
	decoder := hipInferenceModelFixtureTokenizer()
	loaded := &hipLoadedModel{
		modelInfo:   inference.ModelInfo{Architecture: "gemma4", VocabSize: 107, NumLayers: 1, HiddenSize: 8},
		contextSize: 4096,
		tokenText:   decoder,
	}

	model, err := newHipEngineTextModel(loaded, decoder, "gemma4")
	core.RequireNoError(t, err)
	core.AssertEqual(t, []int32{2, 7}, model.Encode("hello"))
	core.AssertEqual(t,
		"<|turn>system\n<|think|>\n<turn|>\n<|turn>model\n",
		model.FormatChatPrompt([]inference.Message{}),
	)
}

func TestHIPInferenceModel_PromptReuseDeclaration_Good(t *testing.T) {
	var declaration engine.PromptReuseCapableModel = newHipTokenModel(&hipLoadedModel{}, hipInferenceModelFixtureTokenizer(), "gemma4")
	core.AssertTrue(t, declaration.SessionsReusePrompts())
}

func TestHIPInferenceModel_CacheModes_Good(t *testing.T) {
	tokenModel := newHipTokenModel(&hipLoadedModel{}, hipInferenceModelFixtureTokenizer(), "gemma4")
	var reporter engine.CacheModeReporter = tokenModel
	core.AssertEqual(t, []string{rocmKVCacheModeFP16, rocmKVCacheModeQ8, rocmKVCacheModeKQ8VQ4}, reporter.SupportedCacheModes())
}

func TestHIPInferenceModel_AudioContract_Good(t *testing.T) {
	tokenModel := newHipTokenModel(&hipLoadedModel{}, hipInferenceModelFixtureTokenizer(), "gemma4")
	var _ engine.AudioInputTokenModel = tokenModel
	var _ engine.VisionTokenModel = tokenModel
	core.AssertFalse(t, tokenModel.AcceptsAudioInput())
	core.AssertFalse(t, tokenModel.AcceptsImageInput())

	tokenModel.loaded.audio = &AudioTower{loaded: &sharedmodel.LoadedAudio{Cfg: sharedmodel.LoadedAudioConfig{
		AudioTokenID:    77,
		AudioBeginToken: "<|audio><|",
		AudioToken:      "audio|>",
		AudioEndToken:   "<audio|>",
	}}}
	core.AssertTrue(t, tokenModel.AcceptsAudioInput())
	core.AssertEqual(t, int32(77), tokenModel.AudioPlaceholderTokenID())
	core.AssertEqual(t, "<|audio><|audio|>audio|><audio|>", tokenModel.AudioPlaceholderBlock(2))
	core.AssertEqual(t, "", tokenModel.AudioPlaceholderBlock(0))
}

func TestHIPInferenceModel_UnifiedVisionContract_Good(t *testing.T) {
	tokenModel := newHipTokenModel(&hipLoadedModel{
		unifiedVision: &UnifiedVisionTower{
			loaded: &sharedmodel.LoadedUnifiedVision{Cfg: sharedmodel.LoadedUnifiedVisionConfig{
				ImageTokenID:         22,
				ImageBeginToken:      "<|image>",
				ImageToken:           "<|image|>",
				ImageEndToken:        "<image|>",
				VideoTokenID:         23,
				VideoToken:           "<|video|>",
				AudioSamplesPerToken: 2,
				AudioTokenID:         24,
				AudioBeginToken:      "<|audio>",
				AudioToken:           "<|audio|>",
				AudioEndToken:        "<audio|>",
			}},
			audioProjection: []float32{1},
		},
	}, hipInferenceModelFixtureTokenizer(), "gemma4")
	var _ engine.VideoTokenModel = tokenModel
	core.AssertTrue(t, tokenModel.AcceptsImageInput())
	core.AssertTrue(t, tokenModel.AcceptsAudioInput())
	core.AssertEqual(t, int32(22), tokenModel.ImagePlaceholderTokenID())
	core.AssertEqual(t, "<|image><|image|><|image|><image|>", tokenModel.ImagePlaceholderBlock(2))
	core.AssertEqual(t, int32(23), tokenModel.VideoPlaceholderTokenID())
	core.AssertEqual(t, "<|image><|video|><|video|><image|>", tokenModel.VideoPlaceholderBlock(2))
	core.AssertEqual(t, int32(24), tokenModel.AudioPlaceholderTokenID())
	core.AssertEqual(t, "<|audio><|audio|><|audio|><audio|>", tokenModel.AudioPlaceholderBlock(2))
}

func TestHIPInferenceModel_EncoderVisionContract_Good(t *testing.T) {
	tokenModel := newHipTokenModel(&hipLoadedModel{
		vision: &HIPVisionEncoderTower{loaded: &sharedmodel.LoadedVision{Cfg: sharedmodel.LoadedVisionConfig{
			ImageTokenID:    31,
			ImageBeginToken: "<|image>",
			ImageToken:      "<|image|>",
			ImageEndToken:   "<image|>",
			VideoTokenID:    32,
			VideoToken:      "<|video|>",
		}}},
	}, hipInferenceModelFixtureTokenizer(), "gemma4")
	core.AssertTrue(t, tokenModel.AcceptsImageInput())
	core.AssertEqual(t, int32(31), tokenModel.ImagePlaceholderTokenID())
	core.AssertEqual(t, "<|image><|image|><image|>", tokenModel.ImagePlaceholderBlock(1))
	core.AssertEqual(t, int32(32), tokenModel.VideoPlaceholderTokenID())
	core.AssertEqual(t, "<|image><|video|><image|>", tokenModel.VideoPlaceholderBlock(1))
}

func TestHIPInferenceModel_SpliceTokenFeatures_Good(t *testing.T) {
	stream := []byte{
		1, 1, 1, 1,
		2, 2, 2, 2,
		3, 3, 3, 3,
		4, 4, 4, 4,
	}
	features := []byte{
		9, 8, 7, 6,
		5, 4, 3, 2,
	}
	err := (&hipTokenModel{}).spliceTokenFeaturesInto(stream, []int32{1, 22, 2, 22}, features, 4, 22, "image")
	core.RequireNoError(t, err)
	core.AssertEqual(t, []byte{
		1, 1, 1, 1,
		9, 8, 7, 6,
		3, 3, 3, 3,
		5, 4, 3, 2,
	}, stream)
}

func TestHIPInferenceModel_DeclaredGemma4ChatTemplate_Good(t *testing.T) {
	decoder := hipInferenceModelFixtureTokenizer()
	large := &hipLoadedModel{
		modelPath:   "/models/gemma-4-12b-it",
		modelInfo:   inference.ModelInfo{Architecture: "gemma4", NumLayers: 48, HiddenSize: 3840},
		modelLabels: map[string]string{"attention_heads": "16", "gemma4_size": "12B"},
		contextSize: 4096,
	}
	largeTokenModel := newHipTokenModel(large, decoder, "gemma4")
	var _ engine.StopTokenDeclarer = largeTokenModel
	var _ engine.ThoughtSuppressorDeclarer = largeTokenModel
	var _ engine.ChatTemplateDeclarer = largeTokenModel
	core.AssertTrue(t, largeTokenModel.NeedsThoughtChannelSuppressor())
	template, ok := largeTokenModel.DeclaredChatTemplate()
	core.RequireTrue(t, ok)
	core.AssertEqual(t, "<|channel>thought\n<channel|>", template.Thinking.OffSuffix)

	largeTextModel, err := newHipEngineTextModel(large, decoder, "gemma4")
	core.RequireNoError(t, err)
	core.AssertEqual(t,
		"<|turn>system\n<|think|>\n<turn|>\n<|turn>user\nhello<turn|>\n<|turn>model\n",
		largeTextModel.FormatChatPromptWithThinking([]inference.Message{{Role: "user", Content: "hello"}}, nil),
	)
	thinkingOff := false
	core.AssertEqual(t,
		"<|turn>user\nhello<turn|>\n<|turn>model\n<|channel>thought\n<channel|>",
		largeTextModel.FormatChatPromptWithThinking([]inference.Message{{Role: "user", Content: "hello"}}, &thinkingOff),
	)

	embedded := newHipTokenModel(&hipLoadedModel{
		modelPath:   "/models/gemma-4-e2b-it",
		modelInfo:   inference.ModelInfo{Architecture: "gemma4"},
		modelLabels: map[string]string{"attention_heads": "8", "gemma4_size": "E2B"},
	}, decoder, "gemma4")
	core.AssertFalse(t, embedded.NeedsThoughtChannelSuppressor())
	embeddedTemplate, ok := embedded.DeclaredChatTemplate()
	core.RequireTrue(t, ok)
	core.AssertEqual(t, "", embeddedTemplate.Thinking.OffSuffix)
}

func TestHIPInferenceModel_DeclaredQwenChatTemplate_Good(t *testing.T) {
	tokenModel := newHipTokenModel(
		&hipLoadedModel{modelInfo: inference.ModelInfo{Architecture: "qwen3_6"}},
		hipInferenceModelFixtureTokenizer(),
		"qwen3_6",
	)
	template, ok := tokenModel.DeclaredChatTemplate()
	core.RequireTrue(t, ok)
	core.AssertEqual(t, "<|im_start|>", template.Open)
	core.AssertEqual(t, "<|im_end|>", template.Close)
	core.AssertEqual(t, "assistant", template.AssistantRole)
	core.AssertEqual(t, []string{"<|im_end|>"}, template.Stops)
	if template.Thinking == nil {
		t.Fatal("Qwen chat template missing thinking-channel declaration")
	}
	core.AssertEqual(t, "<think>\n\n</think>\n\n", template.Thinking.OffSuffix)
}

func TestHIPInferenceModel_ROCmProductionBridge_Good(t *testing.T) {
	decoder := hipInferenceModelFixtureTokenizer()
	loaded := &hipLoadedModel{
		modelInfo:   inference.ModelInfo{Architecture: "gemma4", VocabSize: 107, NumLayers: 1, HiddenSize: 8},
		contextSize: 4096,
		tokenText:   decoder,
	}
	shared, err := newHipEngineTextModel(loaded, decoder, "gemma4")
	core.RequireNoError(t, err)
	model := &rocmModel{
		native:        loaded,
		modelType:     "gemma4",
		modelInfo:     loaded.modelInfo,
		contextLength: loaded.contextSize,
		engineModel:   shared,
	}

	var _ inference.SessionFactory = model
	core.AssertFalse(t, model.ChatInterceptorInstalled())
	model.SetChatInterceptor(func(context.Context, []inference.Message, ...inference.GenerateOption) (iter.Seq[inference.Token], bool) {
		return func(yield func(inference.Token) bool) {
			yield(inference.Token{ID: 7, Text: "bridge"})
		}, true
	})
	core.AssertTrue(t, model.ChatInterceptorInstalled())
	got := slices.Collect(model.Chat(context.Background(), []inference.Message{{Role: "user", Content: "hello"}}))
	core.AssertEqual(t, []inference.Token{{ID: 7, Text: "bridge"}}, got)
	core.AssertEqual(t,
		"<|turn>system\n<|think|>\n<turn|>\n<|turn>user\nhello<turn|>\n<|turn>model\n",
		model.FormatChatPromptWithThinking([]inference.Message{{Role: "user", Content: "hello"}}, nil),
	)
	core.AssertEqual(t,
		shared.FormatChatPrompt([]inference.Message{{Role: "user", Content: "hello"}}),
		model.FormatChatPrompt([]inference.Message{{Role: "user", Content: "hello"}}),
	)
	core.AssertEqual(t,
		shared.FormatChatContinuation([]inference.Message{{Role: "user", Content: "hello"}}),
		model.FormatChatContinuation([]inference.Message{{Role: "user", Content: "hello"}}),
	)
	model.SetChatInterceptor(nil)
	core.AssertFalse(t, model.ChatInterceptorInstalled())
}

func TestHIPInferenceModel_CaptureKV_Good(t *testing.T) {
	model := hipKVCaptureModelForTest()
	var _ inference.KVSnapshotter = model
	snapshot, err := model.CaptureKV(context.Background(), "hello", inference.KVSnapshotCaptureOptions{RawKVOnly: true})
	core.RequireNoError(t, err)
	core.AssertEqual(t, []int32{2, 7}, snapshot.Tokens)
	core.AssertEqual(t, 0, snapshot.SeqLen)
	core.AssertEqual(t, 1, len(snapshot.Layers))
	core.AssertEqual(t, 0, len(snapshot.Layers[0].Heads))
}

func TestHIPInferenceModel_CaptureKV_Bad(t *testing.T) {
	model := hipKVCaptureModelForTest()
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, err := model.CaptureKV(ctx, "hello", inference.KVSnapshotCaptureOptions{})
	core.AssertErrorIs(t, err, context.Canceled)
}

func TestHIPInferenceModel_CaptureKV_Ugly(t *testing.T) {
	var model *rocmModel
	_, err := model.CaptureKV(context.Background(), "hello", inference.KVSnapshotCaptureOptions{})
	core.AssertError(t, err)
}

func TestHIPInferenceModel_CaptureKVChunks_Good(t *testing.T) {
	model := hipKVCaptureModelForTest()
	var _ inference.KVChunkSnapshotter = model
	chunks := func(yield func(string) bool) {
		yield("hello")
		yield("hello")
	}
	snapshot, err := model.CaptureKVChunks(context.Background(), chunks, inference.KVSnapshotCaptureOptions{})
	core.RequireNoError(t, err)
	core.AssertEqual(t, []int32{2, 7, 2, 7}, snapshot.Tokens)
	core.AssertEqual(t, 0, snapshot.SeqLen)
}

func TestHIPInferenceModel_CaptureKVChunks_Bad(t *testing.T) {
	model := hipKVCaptureModelForTest()
	_, err := model.CaptureKVChunks(context.Background(), nil, inference.KVSnapshotCaptureOptions{})
	core.AssertError(t, err)
}

func TestHIPInferenceModel_CaptureKVChunks_Ugly(t *testing.T) {
	var model *rocmModel
	_, err := model.CaptureKVChunks(context.Background(), func(func(string) bool) {}, inference.KVSnapshotCaptureOptions{})
	core.AssertError(t, err)
}
