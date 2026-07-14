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
	model.SetChatInterceptor(func(context.Context, []inference.Message, ...inference.GenerateOption) (iter.Seq[inference.Token], bool) {
		return func(yield func(inference.Token) bool) {
			yield(inference.Token{ID: 7, Text: "bridge"})
		}, true
	})
	got := slices.Collect(model.Chat(context.Background(), []inference.Message{{Role: "user", Content: "hello"}}))
	core.AssertEqual(t, []inference.Token{{ID: 7, Text: "bridge"}}, got)
	core.AssertEqual(t,
		"<|turn>system\n<|think|>\n<turn|>\n<|turn>user\nhello<turn|>\n<|turn>model\n",
		model.FormatChatPromptWithThinking([]inference.Message{{Role: "user", Content: "hello"}}, nil),
	)
}
