// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import (
	"os"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/inference/engine"
	internalgguf "dappco.re/go/inference/engine/hip/internal/gguf"
	modelgguf "dappco.re/go/inference/model/gguf"
)

func TestHIPTokenTextDecoder_Good_LoadEncodeDecode(t *testing.T) {
	path := core.PathJoin(t.TempDir(), "tokenizer.json")
	payload := []byte(`{
		"model": {
			"vocab": {
				"<unk>": 0,
				"<bos>": 2,
				"he": 3,
				"▁": 4,
				"<0x7A>": 5
			},
			"merges": ["h e"]
		},
		"added_tokens": [
			{"id": 2, "content": "<bos>", "special": true},
			{"id": 9, "content": "<turn>", "special": true}
		]
	}`)
	write := core.WriteFile(path, payload, 0o644)
	core.RequireTrue(t, write.OK)

	decoder, err := loadHIPTokenTextDecoder(path)
	core.RequireNoError(t, err)
	core.AssertNotNil(t, decoder)
	core.AssertEqual(t, []int32{2, 3, 9, 5}, decoder.Encode("he<turn>z"))
	core.AssertEqual(t, "he z", decoder.Decode([]int32{2, 3, 4, 5, 9}))
	core.AssertEqual(t, "he", decoder.DecodeToken(3))
	core.AssertEqual(t, "", decoder.DecodeToken(9))
	core.AssertNotNil(t, loadHIPTokenTextDecoderIfPresent(path))
	core.AssertNil(t, loadHIPTokenTextDecoderIfPresent(" "))
	core.AssertNil(t, loadHIPTokenTextDecoderIfPresent(core.PathJoin(t.TempDir(), "missing.json")))
}

func TestHIPTokenTextDecoder_EngineTextTokenizer_Good(t *testing.T) {
	decoder := &hipTokenTextDecoder{
		vocab: map[string]int32{
			"<turn|>":    106,
			"<|channel>": 105,
			"<channel|>": 107,
			"▁world":     8,
		},
		pieces: map[int32]string{
			106: "<turn|>",
			105: "<|channel>",
			107: "<channel|>",
			8:   "▁world",
		},
		special: map[int32]bool{105: true, 106: true, 107: true},
		specialText: map[string]int32{
			"<turn|>":    106,
			"<|channel>": 105,
			"<channel|>": 107,
		},
		eosID:  106,
		hasEOS: true,
	}
	decoder.precomputeDecodedPieces()

	var tok engine.TextTokenizer = decoder
	id, ok := tok.TokenID("<turn|>")
	core.AssertTrue(t, ok)
	core.AssertEqual(t, int32(106), id)
	core.AssertEqual(t, int32(106), tok.EOS())
	core.AssertEqual(t, "world", tok.DecodeOne(8))
	core.AssertEqual(t, "<|channel>", tok.DecodeToken(105))
	core.AssertEqual(t, "<channel|>", tok.DecodeToken(107))
	core.AssertEqual(t, "", tok.DecodeToken(106))
}

func TestHIPTokenTextDecoder_GGUFMetadata_Good(t *testing.T) {
	decoder, err := newHIPTokenTextDecoderFromGGUF(internalgguf.Metadata{
		TokenizerModel:          "gemma4",
		TokenizerTokens:         []string{"<pad>", "<eos>", "<bos>", "hello", "<turn|>", "<|channel>"},
		TokenizerMerges:         []string{"h e", "he l", "hel l", "hell o"},
		TokenizerTokenTypes:     []int32{3, 3, 3, 1, 3, 3},
		TokenizerBOSID:          2,
		TokenizerBOSIDSet:       true,
		TokenizerEOSID:          4,
		TokenizerEOSIDSet:       true,
		TokenizerUnknownID:      1,
		TokenizerUnknownIDSet:   true,
		TokenizerAddBOS:         true,
		TokenizerAddSpacePrefix: false,
	})
	core.RequireNoError(t, err)
	core.AssertEqual(t, []int32{2, 3}, decoder.Encode("hello"))
	core.AssertEqual(t, int32(4), decoder.EOS())
	core.AssertEqual(t, "hello", decoder.DecodeOne(3))
	core.AssertEqual(t, "<|channel>", decoder.DecodeToken(5))
	core.AssertEqual(t, "", decoder.DecodeToken(4))
}

func TestHIPTokenTextDecoder_GGUFMetadata_Bad(t *testing.T) {
	_, err := newHIPTokenTextDecoderFromGGUF(internalgguf.Metadata{
		TokenizerTokens:     []string{"<bos>", "hello"},
		TokenizerTokenTypes: []int32{3},
	})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "token type count")
}

func TestHIPTokenTextDecoder_GGUFLoadConfig_Good(t *testing.T) {
	path := core.PathJoin(t.TempDir(), "tokenizer.gguf")
	metadata := []modelgguf.MetadataEntry{
		{Key: "general.architecture", ValueType: modelgguf.ValueTypeString, Value: "gemma4"},
		{Key: "gemma4.block_count", ValueType: modelgguf.ValueTypeUint32, Value: uint32(1)},
		{Key: "tokenizer.ggml.model", ValueType: modelgguf.ValueTypeString, Value: "gemma4"},
		{Key: "tokenizer.ggml.tokens", ValueType: modelgguf.ValueTypeArray, Value: []string{"<pad>", "<eos>", "<bos>", "hello", "<turn|>"}},
		{Key: "tokenizer.ggml.merges", ValueType: modelgguf.ValueTypeArray, Value: []string{"h e", "he l", "hel l", "hell o"}},
		{Key: "tokenizer.ggml.token_type", ValueType: modelgguf.ValueTypeArray, Value: []int32{3, 3, 3, 1, 3}},
		{Key: "tokenizer.ggml.bos_token_id", ValueType: modelgguf.ValueTypeUint32, Value: uint32(2)},
		{Key: "tokenizer.ggml.eos_token_id", ValueType: modelgguf.ValueTypeUint32, Value: uint32(4)},
		{Key: "tokenizer.ggml.unknown_token_id", ValueType: modelgguf.ValueTypeUint32, Value: uint32(1)},
		{Key: "tokenizer.ggml.add_bos_token", ValueType: modelgguf.ValueTypeBool, Value: true},
	}
	core.RequireNoError(t, modelgguf.WriteFile(path, metadata, nil))
	runtime := &fakeNativeRuntime{available: true, model: &fakeNativeModel{}}
	model, err := resultValue[inference.TextModel](newROCmBackendWithRuntime(runtime).LoadModel(path))
	core.RequireNoError(t, err)
	defer model.Close()
	core.AssertNotNil(t, runtime.loadConfig.TokenText)
	core.AssertEqual(t, []int32{2, 3}, runtime.loadConfig.TokenText.Encode("hello"))
}

func TestHIPTokenTextDecoder_Gemma4TextPromptPreservesGenerationNewline(t *testing.T) {
	decoder := &hipTokenTextDecoder{
		vocab: map[string]int32{
			"<bos>":   2,
			"<0x0A>":  107,
			"<|turn>": 105,
		},
		pieces:      map[int32]string{2: "<bos>", 107: "<0x0A>", 105: "<|turn>"},
		special:     map[int32]bool{2: true, 105: true},
		specialText: map[string]int32{"<bos>": 2, "<|turn>": 105},
		bosID:       2,
		hasBOS:      true,
	}
	model := &hipLoadedModel{
		modelInfo: inference.ModelInfo{Architecture: "gemma4", QuantBits: 4},
		tokenText: decoder,
	}
	tokens, ok, err := hipGemma4Q4TextPromptIDs("text:<bos><|turn>\n", model)
	core.RequireNoError(t, err)
	core.RequireTrue(t, ok)
	core.AssertEqual(t, []int32{2, 105, 107}, tokens)
}

func TestHIPTokenTextDecoder_Gemma4LocalChatTemplateIDs_Good(t *testing.T) {
	path := os.Getenv("GO_ROCM_GEMMA4_Q4_TOKENIZER_PATH")
	if path == "" {
		t.Skip("set GO_ROCM_GEMMA4_Q4_TOKENIZER_PATH to verify local Gemma4 tokenizer chat-template IDs")
	}
	decoder, err := loadHIPTokenTextDecoder(path)
	core.RequireNoError(t, err)
	got := decoder.Encode("<bos><|turn>user\nHi<turn|>\n<|turn>model\n")
	core.AssertEqual(t, []int32{2, 105, 2364, 107, 10979, 106, 107, 105, 4368, 107}, got)
}

func TestHIPTokenTextDecoder_Gemma4DefaultSuppressTokenIDs_Good(t *testing.T) {
	decoder := &hipTokenTextDecoder{
		specialText: map[string]int32{
			"<pad>":        0,
			"<bos>":        2,
			"<|turn>":      105,
			"<turn|>":      106,
			"<|tool_call>": 200,
			"<notlisted|>": 201,
		},
	}
	model := &hipLoadedModel{
		modelInfo: inference.ModelInfo{Architecture: "gemma4", QuantBits: 4},
		tokenText: decoder,
	}
	ids := hipGemma4Q4DefaultSuppressTokenIDs(model)
	core.AssertTrue(t, hipTokenIsSuppressed(0, ids))
	core.AssertTrue(t, hipTokenIsSuppressed(2, ids))
	core.AssertTrue(t, hipTokenIsSuppressed(105, ids))
	core.AssertTrue(t, hipTokenIsSuppressed(200, ids))
	core.AssertFalse(t, hipTokenIsSuppressed(106, ids))
	core.AssertFalse(t, hipTokenIsSuppressed(201, ids))

	generationIDs := hipGemma4Q4GenerationSuppressTokenIDs(model, nil)
	core.AssertTrue(t, hipTokenIsSuppressed(106, generationIDs))
	explicitStopIDs := hipGemma4Q4GenerationSuppressTokenIDs(model, []int32{106})
	core.AssertFalse(t, hipTokenIsSuppressed(106, explicitStopIDs))
}

func BenchmarkHIPTokenTextDecoder_EncodeRepeatedMerges(b *testing.B) {
	decoder := &hipTokenTextDecoder{
		vocab: map[string]int32{
			"abc": 1,
			"▁":   2,
		},
		mergeRanks: map[string]int{
			"a b":  0,
			"ab c": 1,
		},
	}
	text := "abc abc abc abc abc abc abc abc"
	if got := decoder.Encode(text); len(got) == 0 {
		b.Fatal("empty tokenization")
	}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_ = decoder.Encode(text)
	}
}

func BenchmarkHIPTokenTextDecoder_EncodeShortText(b *testing.B) {
	decoder := &hipTokenTextDecoder{
		vocab: map[string]int32{
			"<bos>": 2,
			"H":     10,
			"i":     11,
		},
		pieces: map[int32]string{2: "<bos>"},
		bosID:  2,
		hasBOS: true,
	}
	if got := decoder.Encode("Hi"); len(got) != 3 {
		b.Fatalf("Encode(Hi) tokens = %v, want 3 tokens", got)
	}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_ = decoder.Encode("Hi")
	}
}

func BenchmarkHIPGemma4Q4GenerationSuppressTokenIDs_CachedExplicitStop(b *testing.B) {
	decoder := &hipTokenTextDecoder{
		specialText: map[string]int32{
			"<pad>":        0,
			"<bos>":        2,
			"<|turn>":      105,
			"<turn|>":      106,
			"<|tool_call>": 200,
		},
	}
	model := &hipLoadedModel{
		modelInfo: inference.ModelInfo{Architecture: "gemma4", QuantBits: 4},
		tokenText: decoder,
	}
	if ids := hipGemma4Q4GenerationSuppressTokenIDs(model, []int32{106}); len(ids) == 0 {
		b.Fatal("initial suppress IDs are empty")
	}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		ids := hipGemma4Q4GenerationSuppressTokenIDs(model, []int32{106})
		if !hipTokenIsSuppressed(200, ids) || hipTokenIsSuppressed(106, ids) {
			b.Fatalf("suppress IDs = %#v", ids)
		}
	}
}

func BenchmarkHIPGemma4Q4GenerationSuppressTokenIDs_CachedDefaultStop(b *testing.B) {
	decoder := &hipTokenTextDecoder{
		specialText: map[string]int32{
			"<pad>":        0,
			"<bos>":        2,
			"<|turn>":      105,
			"<turn|>":      106,
			"<|tool_call>": 200,
		},
	}
	model := &hipLoadedModel{
		modelInfo: inference.ModelInfo{Architecture: "gemma4", QuantBits: 4},
		tokenText: decoder,
	}
	if ids := hipGemma4Q4GenerationSuppressTokenIDs(model, nil); len(ids) == 0 {
		b.Fatal("initial suppress IDs are empty")
	}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		ids := hipGemma4Q4GenerationSuppressTokenIDs(model, nil)
		if !hipTokenIsSuppressed(106, ids) || !hipTokenIsSuppressed(200, ids) {
			b.Fatalf("suppress IDs = %#v", ids)
		}
	}
}

func TestHIPTokenTextDecoder_Bad_MergeAndFallbackEdges(t *testing.T) {
	stringRanks := hipTokenTextMergeRanks([]byte(`["a b","bad","c d"]`))
	core.AssertEqual(t, 0, stringRanks["a b"])
	core.AssertEqual(t, 2, stringRanks["c d"])
	arrayRanks := hipTokenTextMergeRanks([]byte(`[["x","y"],["bad"],["y","z"]]`))
	core.AssertEqual(t, 0, arrayRanks["x y"])
	core.AssertEqual(t, 2, arrayRanks["y z"])
	escapedRanks := hipTokenTextMergeRanks([]byte(`[["\n","x"],"y z"]`))
	core.AssertEqual(t, 0, escapedRanks["\n x"])
	core.AssertEqual(t, 1, escapedRanks["y z"])
	core.AssertEqual(t, 0, len(hipTokenTextMergeRanks(nil)))
	core.AssertEqual(t, 0, len(hipTokenTextMergeRanks([]byte(`{"not":"a merge list"}`))))

	decoder := &hipTokenTextDecoder{
		vocab:      map[string]int32{"<unk>": 7},
		pieces:     map[int32]string{7: "<unk>"},
		hasUnknown: true,
		unknownID:  7,
	}
	core.AssertEqual(t, []int32{7, 7}, decoder.Encode("é"))
	core.AssertEqual(t, "", (*hipTokenTextDecoder)(nil).Decode([]int32{1}))
	core.AssertEqual(t, "", decoder.DecodeToken(404))
}

func BenchmarkHIPTokenTextMergeRanks_ArrayPairs(b *testing.B) {
	raw := []byte(`[["a","b"],["b","c"],["c","d"],["d","e"],["e","f"],["f","g"],["g","h"],["h","i"],["i","j"],["j","k"],["k","l"],["l","m"],["m","n"],["n","o"],["o","p"],["p","q"]]`)
	if got := hipTokenTextMergeRanks(raw); got["a b"] != 0 || got["p q"] != 15 {
		b.Fatalf("merge ranks = %#v", got)
	}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		got := hipTokenTextMergeRanks(raw)
		if got["a b"] != 0 || got["p q"] != 15 {
			b.Fatalf("merge ranks = %#v", got)
		}
	}
}

func BenchmarkHIPTokenTextDecoder_LoadLocalGemma4(b *testing.B) {
	path := os.Getenv("GO_ROCM_GEMMA4_Q4_TOKENIZER_PATH")
	if path == "" {
		b.Skip("set GO_ROCM_GEMMA4_Q4_TOKENIZER_PATH to benchmark local Gemma4 tokenizer loading")
	}
	decoder, err := loadHIPTokenTextDecoder(path)
	if err != nil {
		b.Fatal(err)
	}
	if decoder == nil || len(decoder.mergeRanks) == 0 {
		b.Fatal("Gemma4 tokenizer loaded without merge ranks")
	}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		decoder, err := loadHIPTokenTextDecoder(path)
		if err != nil {
			b.Fatal(err)
		}
		if len(decoder.mergeRanks) == 0 {
			b.Fatal("Gemma4 tokenizer loaded without merge ranks")
		}
	}
}

func BenchmarkHIPTokenTextDecoder_DecodeTokenCached(b *testing.B) {
	decoder := &hipTokenTextDecoder{
		pieces:  map[int32]string{7: "hello", 8: "▁world", 9: "<0x0A>"},
		special: map[int32]bool{},
	}
	decoder.precomputeDecodedPieces()
	ids := []int32{7, 8, 9}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_ = decoder.DecodeToken(ids[i%len(ids)])
	}
}
