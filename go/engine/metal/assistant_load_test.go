// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"slices"

	g4 "dappco.re/go/inference/model/gemma4"
	"encoding/binary"
	"sort"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference/model"
	"dappco.re/go/inference/model/gguf"
	"dappco.re/go/inference/model/safetensors"
	coreio "dappco.re/go/io"
)

const (
	nativeAssistantWordedPromptText  = "native assistant sampled drafting uses words"
	nativeAssistantWordedPromptWords = 6
)

var nativeAssistantWordedPromptTokens = [...]int32{1, 5, 3, 2, 6, 4}

func nativeAssistantWordedPromptIDs() []int32 {
	return nativeAssistantWordedPromptTokens[:]
}

func nativeAssistantWordedPromptCandidates() [][]int32 {
	return [][]int32{
		nativeAssistantWordedPromptIDs(),
		{2, 4, 6, 1, 5},
		{3, 1, 7, 5, 2},
		{4, 2, 5, 6, 3},
		{5, 3, 1, 7, 4},
		{6, 7, 2, 4, 1},
	}
}

func TestNativeAssistantWordedPromptFixtureUsesAFewWords(t *testing.T) {
	prompt := nativeAssistantWordedPromptIDs()
	if nativeAssistantWordedPromptWords < 5 {
		t.Fatalf("native assistant worded prompt %q has %d words, want a few words", nativeAssistantWordedPromptText, nativeAssistantWordedPromptWords)
	}
	if len(prompt) != nativeAssistantWordedPromptWords {
		t.Fatalf("native assistant worded prompt token count = %d, want one stable token id per word", len(prompt))
	}
	for i, id := range prompt {
		if id <= 0 || id >= 8 {
			t.Fatalf("native assistant worded prompt token %d = %d outside fixture vocab", i, id)
		}
	}
}

func TestLoadAssistantDirLoadsMetadataAndTensors(t *testing.T) {
	dir := writeNativeAssistantDir(t, nativeAssistantTinyTensors(true))

	assistant, err := LoadAssistantDir(dir)
	if err != nil {
		t.Fatalf("LoadAssistantDir: %v", err)
	}
	defer assistant.Close()

	if assistant.ModelType() != "gemma4_assistant" {
		t.Fatalf("ModelType = %q, want gemma4_assistant", assistant.ModelType())
	}
	if assistant.Tokenizer() == nil {
		t.Fatal("Tokenizer = nil, want loaded assistant tokenizer")
	}
	if assistant.NumLayers() != 2 {
		t.Fatalf("NumLayers = %d, want 2", assistant.NumLayers())
	}
	if assistant.BackboneHiddenSize != 8 || assistant.NumCentroids != 2 || !assistant.UseOrderedEmbeddings {
		t.Fatalf("assistant metadata backbone=%d centroids=%d ordered=%v", assistant.BackboneHiddenSize, assistant.NumCentroids, assistant.UseOrderedEmbeddings)
	}
	if assistant.Arch.Hidden != 4 || assistant.Arch.Vocab != 8 || assistant.Arch.FF != 8 {
		t.Fatalf("assistant Arch = %+v, want hidden/vocab/ff 4/8/8", assistant.Arch)
	}
	if tok, ok := assistant.Tensor("masked_embedding.token_ordering"); !ok || tok.Dtype != "I64" || len(tok.Shape) != 1 || tok.Shape[0] != 8 {
		t.Fatalf("token_ordering tensor = %+v, ok=%v; want I64 [8]", tok, ok)
	}
}

func TestLoadAssistantDirAcceptsFlatTextConfig(t *testing.T) {
	dir := writeNativeAssistantFlatDir(t, nativeAssistantTinyTensors(true), true)

	assistant, err := LoadAssistantDir(dir)
	if err != nil {
		t.Fatalf("LoadAssistantDir(flat config): %v", err)
	}
	defer assistant.Close()

	if assistant.Arch.Hidden != 4 || assistant.Arch.Vocab != 8 || assistant.Arch.FF != 8 {
		t.Fatalf("assistant flat Arch = %+v, want hidden/vocab/ff 4/8/8", assistant.Arch)
	}
	if assistant.BackboneHiddenSize != 8 || assistant.NumCentroids != 2 || !assistant.UseOrderedEmbeddings {
		t.Fatalf("assistant flat metadata backbone=%d centroids=%d ordered=%v", assistant.BackboneHiddenSize, assistant.NumCentroids, assistant.UseOrderedEmbeddings)
	}
}

func TestLoadGemma4UnifiedAssistantDirReportsAssistantModelType(t *testing.T) {
	dir := writeNativeAssistantDirWithModelType(t, nativeAssistantTinyTensors(true), true, "gemma4_unified_assistant")

	assistant, err := LoadAssistantDir(dir)
	if err != nil {
		t.Fatalf("LoadAssistantDir(unified assistant): %v", err)
	}
	defer assistant.Close()

	if assistant.Config.ModelType != "gemma4_unified_assistant" {
		t.Fatalf("Config.ModelType = %q, want raw unified assistant model type", assistant.Config.ModelType)
	}
	if assistant.ModelType() != "gemma4_assistant" {
		t.Fatalf("ModelType = %q, want public assistant model type", assistant.ModelType())
	}
}

func TestLoadAssistantDirRejectsMissingRequiredTensor(t *testing.T) {
	tensors := nativeAssistantTinyTensors(false)
	delete(tensors, "post_projection.weight")
	dir := writeNativeAssistantDir(t, tensors)

	assistant, err := LoadAssistantDir(dir)
	if assistant != nil {
		t.Fatalf("LoadAssistantDir assistant = %v, want nil on invalid tensor set", assistant)
	}
	if err == nil {
		t.Fatal("LoadAssistantDir error = nil, want missing post_projection.weight")
	}
	if !core.Contains(err.Error(), "post_projection.weight") {
		t.Fatalf("LoadAssistantDir error = %v, want post_projection.weight", err)
	}
}

func TestLoadAssistantPairDirsValidatesTargetCompatibility(t *testing.T) {
	targetDir := writeNativeAssistantTargetDir(t, 8, []string{"sliding_attention", "full_attention"})
	assistantDir := writeNativeAssistantDir(t, nativeAssistantTinyTensors(true))

	pair, err := LoadAssistantPairDirs(targetDir, assistantDir)
	if err != nil {
		t.Fatalf("LoadAssistantPairDirs: %v", err)
	}
	defer pair.Close()

	if pair.TargetArch.Hidden != 8 || pair.TargetArch.Vocab != 8 {
		t.Fatalf("TargetArch = %+v, want hidden/vocab 8/8", pair.TargetArch)
	}
	if pair.Assistant == nil || pair.Assistant.NumLayers() != 2 {
		t.Fatalf("Assistant = %+v, want loaded two-layer assistant", pair.Assistant)
	}
}

func TestLoadAssistantPairDirsRejectsBackboneMismatch(t *testing.T) {
	targetDir := writeNativeAssistantTargetDir(t, 12, []string{"sliding_attention", "full_attention"})
	assistantDir := writeNativeAssistantDir(t, nativeAssistantTinyTensors(true))

	pair, err := LoadAssistantPairDirs(targetDir, assistantDir)
	if pair != nil {
		t.Fatalf("LoadAssistantPairDirs pair = %v, want nil on mismatch", pair)
	}
	if err == nil {
		t.Fatal("LoadAssistantPairDirs error = nil, want backbone mismatch")
	}
	if !core.Contains(err.Error(), "backbone_hidden_size") {
		t.Fatalf("LoadAssistantPairDirs error = %v, want backbone_hidden_size", err)
	}
}

func TestLoadAssistantPairDirsLoadsGGUFDrafter(t *testing.T) {
	targetDir := writeNativeAssistantTargetDir(t, 8, []string{"sliding_attention", "full_attention"})
	writeNativeAssistantTokenizer(t, targetDir)
	ggufPath := writeNativeAssistantGGUF(t, nativeAssistantTinyTensors(false))

	pair, err := LoadAssistantPairDirs(targetDir, ggufPath)
	if err != nil {
		t.Fatalf("LoadAssistantPairDirs(gguf): %v", err)
	}
	defer pair.Close()

	if pair.Assistant.Tokenizer() == nil {
		t.Fatal("GGUF assistant tokenizer = nil, want borrowed target tokenizer")
	}
	if pair.Assistant.Arch.Vocab != 8 || pair.Assistant.Arch.Hidden != 4 {
		t.Fatalf("GGUF assistant arch = %+v, want vocab/hidden 8/4", pair.Assistant.Arch)
	}
	if tensor, ok := pair.Assistant.Tensor("model.embed_tokens.weight"); !ok || tensor.Dtype != "BF16" || len(tensor.Shape) != 2 {
		t.Fatalf("GGUF mapped embed tensor = %+v ok=%v, want BF16 rank-2", tensor, ok)
	}
	if _, ok := pair.Assistant.Tensor("model.layers.0.layer_scalar.weight"); !ok {
		t.Fatal("GGUF layer_output_scale was not mapped to layer_scalar.weight")
	}
}

func TestAssistantTargetKVByLayerTypeResolvesSharedOwners(t *testing.T) {
	assistant := nativeAssistantTinyLoaded(t, true)
	defer assistant.Close()
	pair := &AssistantPair{
		TargetArch: model.Arch{Hidden: 8, Vocab: 8, Layer: []model.LayerSpec{
			{Attention: model.SlidingAttention, KVShareFrom: 0, CacheIndex: 0},
			{Attention: model.GlobalAttention, KVShareFrom: 1, CacheIndex: 1},
			{Attention: model.SlidingAttention, KVShareFrom: 0, CacheIndex: -1},
			{Attention: model.GlobalAttention, KVShareFrom: 1, CacheIndex: -1},
		}},
		Assistant: assistant,
	}

	streams, err := pair.TargetKVByLayerType([]AssistantTargetKV{
		nativeAssistantTargetKVFixture(0x11),
		nativeAssistantTargetKVFixture(0x22),
	})
	if err != nil {
		t.Fatalf("TargetKVByLayerType: %v", err)
	}

	sliding, ok := streams.Get("sliding_attention")
	if !ok || len(sliding.Key) == 0 || sliding.Key[0] != 0x11 {
		t.Fatalf("sliding stream = %+v, ok=%v; want cache 0", sliding, ok)
	}
	full, ok := streams.Get("full_attention")
	if !ok || len(full.Key) == 0 || full.Key[0] != 0x22 {
		t.Fatalf("full stream = %+v, ok=%v; want cache 1", full, ok)
	}
}

func TestAssistantTargetKVByLayerTypeRejectsMissingAssistantStream(t *testing.T) {
	assistant := nativeAssistantTinyLoaded(t, true)
	defer assistant.Close()
	pair := &AssistantPair{
		TargetArch: model.Arch{Hidden: 8, Vocab: 8, Layer: []model.LayerSpec{
			{Attention: model.SlidingAttention, KVShareFrom: 0, CacheIndex: 0},
		}},
		Assistant: assistant,
	}

	_, err := pair.TargetKVByLayerType([]AssistantTargetKV{nativeAssistantTargetKVFixture(0x11)})
	if err == nil {
		t.Fatal("TargetKVByLayerType error = nil, want missing full_attention stream")
	}
	if !core.Contains(err.Error(), "full_attention") {
		t.Fatalf("TargetKVByLayerType error = %v, want full_attention", err)
	}
}

func TestAssistantTargetKVByLayerTypeLastOwnerWins(t *testing.T) {
	assistant := nativeAssistantTinyLoaded(t, false)
	defer assistant.Close()
	pair := &AssistantPair{
		TargetArch: model.Arch{Hidden: 8, Vocab: 8, Layer: []model.LayerSpec{
			{Attention: model.SlidingAttention, KVShareFrom: 0, CacheIndex: 0},
			{Attention: model.SlidingAttention, KVShareFrom: 1, CacheIndex: 1},
		}},
		Assistant: assistant,
	}
	pair.Assistant.Config.LayerTypes = []string{"sliding_attention", "sliding_attention"}

	streams, err := pair.TargetKVByLayerType([]AssistantTargetKV{
		nativeAssistantTargetKVFixture(0x11),
		nativeAssistantTargetKVFixture(0x33),
	})
	if err != nil {
		t.Fatalf("TargetKVByLayerType: %v", err)
	}

	sliding, ok := streams.Get("sliding_attention")
	if !ok || len(sliding.Key) == 0 || sliding.Key[0] != 0x33 {
		t.Fatalf("sliding stream = %+v, ok=%v; want last owner cache 1", sliding, ok)
	}
}

func TestAssistantPairTargetKVByLayerTypeFromSessionTransposesResidentRows(t *testing.T) {
	assistant := nativeAssistantTinyLoaded(t, true)
	defer assistant.Close()

	arch := nativeAssistantSessionTargetArchForTest()
	rowBytes := 2 * 2 * bf16Size
	slidingKey := nativeAssistantSessionKVRowsForTest(4, 2, 2, 0x10)
	slidingValue := nativeAssistantSessionKVRowsForTest(4, 2, 2, 0x20)
	fullKey := nativeAssistantSessionKVRowsForTest(4, 2, 2, 0x30)
	fullValue := nativeAssistantSessionKVRowsForTest(4, 2, 2, 0x40)
	session := &ArchSession{
		arch: arch,
		state: archDecodeState{
			specs: arch.Layer,
		},
		stateBlockViews: []sessionStateLayerView{
			{
				layer: 0, kvHeads: 2, headDim: 2, rowBytes: rowBytes, cacheIndex: 0,
				cacheMode: nativeStateCacheModeFixed, cacheRows: 4, keyBytes: slidingKey, valueBytes: slidingValue,
			},
			{
				layer: 1, kvHeads: 2, headDim: 2, rowBytes: rowBytes, cacheIndex: 1,
				cacheMode: nativeStateCacheModeFixed, cacheRows: 4, keyBytes: fullKey, valueBytes: fullValue,
			},
		},
		pos:    3,
		maxLen: 4,
	}
	pair := &AssistantPair{TargetArch: arch, Assistant: assistant}

	streams, err := pair.TargetKVByLayerTypeFromSession(session)
	if err != nil {
		t.Fatalf("TargetKVByLayerTypeFromSession: %v", err)
	}
	sliding, ok := streams.Get("sliding_attention")
	if !ok {
		t.Fatal("sliding_attention stream missing")
	}
	if sliding.Offset != 0 || sliding.Length != 3 || sliding.KVHeads != 2 || sliding.HeadDim != 2 {
		t.Fatalf("sliding stream = %+v, want offset 0 length 3 2x2 geometry", sliding)
	}
	if len(sliding.Key) != 3*rowBytes || len(sliding.Value) != 3*rowBytes {
		t.Fatalf("sliding stream bytes = %d/%d, want %d", len(sliding.Key), len(sliding.Value), 3*rowBytes)
	}
	if got := sliding.Key[0]; got != 0x10 {
		t.Fatalf("sliding head0 seq0 key = %#x, want token0/head0", got)
	}
	if got := sliding.Key[1*2*bf16Size]; got != 0x20 {
		t.Fatalf("sliding head0 seq1 key = %#x, want token1/head0", got)
	}
	if got := sliding.Key[3*2*bf16Size]; got != 0x11 {
		t.Fatalf("sliding head1 seq0 key = %#x, want token0/head1", got)
	}
	full, ok := streams.Get("full_attention")
	if !ok {
		t.Fatal("full_attention stream missing")
	}
	if full.Key[0] != 0x30 || full.Value[0] != 0x40 || full.Key[3*2*bf16Size] != 0x31 {
		t.Fatalf("full stream head-major bytes = %#x/%#x/%#x, want cache-index 1 rows transposed", full.Key[0], full.Value[0], full.Key[3*2*bf16Size])
	}
}

func TestAssistantPairTargetKVByLayerTypeFromSessionScratchReusesSlabs(t *testing.T) {
	assistant := nativeAssistantTinyLoaded(t, true)
	defer assistant.Close()

	arch := nativeAssistantSessionTargetArchForTest()
	rowBytes := 2 * 2 * bf16Size
	session := &ArchSession{
		arch: arch,
		state: archDecodeState{
			specs: arch.Layer,
		},
		stateBlockViews: []sessionStateLayerView{
			{
				layer: 0, kvHeads: 2, headDim: 2, rowBytes: rowBytes, cacheIndex: 0,
				cacheMode: nativeStateCacheModeFixed, cacheRows: 4,
				keyBytes:   nativeAssistantSessionKVRowsForTest(4, 2, 2, 0x10),
				valueBytes: nativeAssistantSessionKVRowsForTest(4, 2, 2, 0x20),
			},
			{
				layer: 1, kvHeads: 2, headDim: 2, rowBytes: rowBytes, cacheIndex: 1,
				cacheMode: nativeStateCacheModeFixed, cacheRows: 4,
				keyBytes:   nativeAssistantSessionKVRowsForTest(4, 2, 2, 0x30),
				valueBytes: nativeAssistantSessionKVRowsForTest(4, 2, 2, 0x40),
			},
		},
		pos:    3,
		maxLen: 4,
	}
	pair := &AssistantPair{TargetArch: arch, Assistant: assistant}

	first, err := pair.targetKVByLayerTypeFromSessionScratch(session)
	if err != nil {
		t.Fatalf("targetKVByLayerTypeFromSessionScratch first: %v", err)
	}
	sliding1, ok := first.Get("sliding_attention")
	if !ok {
		t.Fatal("sliding_attention stream missing")
	}
	full1, ok := first.Get("full_attention")
	if !ok {
		t.Fatal("full_attention stream missing")
	}
	if len(first.entries) == 0 {
		t.Fatal("first scratch target KV mapping has no layer-type entries")
	}
	entryPtr := &first.entries[0]
	slidingKeyPtr := byteDataPointer(sliding1.Key)
	slidingValuePtr := byteDataPointer(sliding1.Value)
	fullKeyPtr := byteDataPointer(full1.Key)
	fullValuePtr := byteDataPointer(full1.Value)

	second, err := pair.targetKVByLayerTypeFromSessionScratch(session)
	if err != nil {
		t.Fatalf("targetKVByLayerTypeFromSessionScratch second: %v", err)
	}
	sliding2, _ := second.Get("sliding_attention")
	full2, _ := second.Get("full_attention")

	if len(second.entries) == 0 || &second.entries[0] != entryPtr {
		t.Fatal("scratch target KV mapping did not reuse layer-type entry backing")
	}
	if byteDataPointer(sliding2.Key) != slidingKeyPtr || byteDataPointer(sliding2.Value) != slidingValuePtr ||
		byteDataPointer(full2.Key) != fullKeyPtr || byteDataPointer(full2.Value) != fullValuePtr {
		t.Fatal("scratch target KV mapping did not reuse K/V slab backing")
	}
	if sliding2.Key[0] != 0x10 || sliding2.Value[0] != 0x20 || full2.Key[0] != 0x30 || full2.Value[0] != 0x40 {
		t.Fatalf("scratch target KV bytes changed: sliding %#x/%#x full %#x/%#x", sliding2.Key[0], sliding2.Value[0], full2.Key[0], full2.Value[0])
	}
}

func TestAssistantPairTargetKVByLayerTypeFromSessionUsesSlidingWindowOffset(t *testing.T) {
	assistant := nativeAssistantTinyLoaded(t, true)
	defer assistant.Close()

	arch := nativeAssistantSessionTargetArchForTest()
	rowBytes := 2 * 2 * bf16Size
	slidingKey := make([]byte, 4*rowBytes)
	slidingValue := make([]byte, 4*rowBytes)
	for token := 2; token < 6; token++ {
		slot := token % 4
		slidingKey[slot*rowBytes] = byte(token)
		slidingValue[slot*rowBytes] = byte(token + 0x10)
	}
	session := &ArchSession{
		arch: arch,
		state: archDecodeState{
			specs: arch.Layer,
		},
		stateBlockViews: []sessionStateLayerView{
			{
				layer: 0, kvHeads: 2, headDim: 2, rowBytes: rowBytes, cacheIndex: 0,
				cacheMode: nativeStateCacheModeFixed, maxSize: 4, cacheRows: 4, keyBytes: slidingKey, valueBytes: slidingValue,
			},
			{
				layer: 1, kvHeads: 2, headDim: 2, rowBytes: rowBytes, cacheIndex: 1,
				cacheMode: nativeStateCacheModeFixed, cacheRows: 8,
				keyBytes:   nativeAssistantSessionRowsForTest(8, rowBytes, 0x30),
				valueBytes: nativeAssistantSessionRowsForTest(8, rowBytes, 0x40),
			},
		},
		pos:    6,
		maxLen: 8,
	}
	pair := &AssistantPair{TargetArch: arch, Assistant: assistant}

	streams, err := pair.TargetKVByLayerTypeFromSession(session)
	if err != nil {
		t.Fatalf("TargetKVByLayerTypeFromSession: %v", err)
	}
	sliding, ok := streams.Get("sliding_attention")
	if !ok {
		t.Fatal("sliding_attention stream missing")
	}
	if sliding.Offset != 2 || sliding.Length != 4 {
		t.Fatalf("sliding stream offset/length = %d/%d, want 2/4", sliding.Offset, sliding.Length)
	}
	for row, want := range []byte{2, 3, 4, 5} {
		if got := sliding.Key[row*2*bf16Size]; got != want {
			t.Fatalf("sliding key head0 seq %d starts %#x, want token %#x", row, got, want)
		}
		if got := sliding.Value[row*2*bf16Size]; got != want+0x10 {
			t.Fatalf("sliding value head0 seq %d starts %#x, want token %#x", row, got, want+0x10)
		}
	}
}

func TestAssistantDraftInputProjectionMatchesReference(t *testing.T) {
	requireNativeRuntime(t)

	tensors := nativeAssistantTinyTensors(true)
	preW := nativeAssistantProjectionFixture(4, 16)
	tensors["pre_projection.weight"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{4, 16}, Data: toBF16Bytes(preW)}
	dir := writeNativeAssistantDir(t, tensors)

	assistant, err := LoadAssistantDir(dir)
	if err != nil {
		t.Fatalf("LoadAssistantDir: %v", err)
	}
	defer assistant.Close()

	tokenEmbedding := toBF16Bytes([]float32{1, 2, -1, 0.5, 0.25, -0.5, 1.5, -2})
	previousHidden := toBF16Bytes([]float32{0.5, -1.5, 2, 1, -0.25, 0.75, -1, 0.125})
	got, err := assistant.DraftInputProjection(tokenEmbedding, previousHidden)
	if err != nil {
		t.Fatalf("DraftInputProjection: %v", err)
	}

	combined := append(append([]byte{}, tokenEmbedding...), previousHidden...)
	want := nativeAssistantMatMulBF16NTReference(combined, toBF16Bytes(preW), 1, 16, 4)
	assertFloat32Near(t, "draft input projection", bf16Floats(got), want, 0.02)
}

func TestAssistantDraftInputProjectionIntoAllocationBudget(t *testing.T) {
	requireNativeRuntime(t)

	tensors := nativeAssistantTinyTensors(true)
	preW := nativeAssistantProjectionFixture(4, 16)
	tensors["pre_projection.weight"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{4, 16}, Data: toBF16Bytes(preW)}
	dir := writeNativeAssistantDir(t, tensors)

	assistant, err := LoadAssistantDir(dir)
	if err != nil {
		t.Fatalf("LoadAssistantDir: %v", err)
	}
	defer assistant.Close()

	tokenEmbedding := toBF16Bytes([]float32{1, 2, -1, 0.5, 0.25, -0.5, 1.5, -2})
	previousHidden := toBF16Bytes([]float32{0.5, -1.5, 2, 1, -0.25, 0.75, -1, 0.125})
	out := make([]byte, assistant.Arch.Hidden*bf16Size)
	if _, err := assistant.DraftInputProjectionInto(out, tokenEmbedding, previousHidden); err != nil {
		t.Fatalf("warm DraftInputProjectionInto: %v", err)
	}

	allocs := testing.AllocsPerRun(20, func() {
		if _, err := assistant.DraftInputProjectionInto(out, tokenEmbedding, previousHidden); err != nil {
			t.Fatalf("DraftInputProjectionInto: %v", err)
		}
	})
	if allocs > 10 {
		t.Fatalf("DraftInputProjectionInto allocations/run = %.0f, want <= 10 with caller output and warm scratch", allocs)
	}
}

func TestAssistantPairDraftInputProjectionForTokenUsesScaledTargetEmbedding(t *testing.T) {
	requireNativeRuntime(t)

	targetDir := writeNativeAssistantTargetDir(t, 8, []string{"sliding_attention", "full_attention"})
	tensors := nativeAssistantTinyTensors(true)
	preW := nativeAssistantProjectionFixture(4, 16)
	tensors["pre_projection.weight"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{4, 16}, Data: toBF16Bytes(preW)}
	assistantDir := writeNativeAssistantDir(t, tensors)

	pair, err := LoadAssistantPairDirs(targetDir, assistantDir)
	if err != nil {
		t.Fatalf("LoadAssistantPairDirs: %v", err)
	}
	defer pair.Close()

	targetEmbed := toBF16Bytes([]float32{
		0, 0, 0, 0, 0, 0, 0, 0,
		1, -0.5, 0.25, 2, -1, 0.75, 1.5, -2,
		0.5, 1, -1.5, 0, 0.125, -0.25, 2, -0.75,
		-1, 1.25, 0.5, -0.5, 2, 0, -2, 0.25,
		0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0,
	})
	previousHidden := toBF16Bytes([]float32{0.5, -1.5, 2, 1, -0.25, 0.75, -1, 0.125})

	got, err := pair.DraftInputProjectionForToken(targetEmbed, 1, previousHidden)
	if err != nil {
		t.Fatalf("DraftInputProjectionForToken: %v", err)
	}

	embedding, err := EmbedTokensBF16(targetEmbed, []int32{1}, pair.TargetArch.Vocab, pair.TargetArch.Hidden, embedScaleOf(pair.TargetArch))
	if err != nil {
		t.Fatalf("EmbedTokensBF16 reference: %v", err)
	}
	combined := append(append([]byte{}, embedding[0]...), previousHidden...)
	want := nativeAssistantMatMulBF16NTReference(combined, toBF16Bytes(preW), 1, 16, 4)
	assertFloat32Near(t, "pair draft input projection for token", bf16Floats(got), want, 0.02)
}

func TestAssistantPairDraftInputProjectionForTokenIntoLargeEmbeddingMatchesDirectAllocationBudget(t *testing.T) {
	requireNativeRuntime(t)

	const targetHidden, assistantHidden, vocab = 2048, 4, 8
	preWeight := toBF16Bytes(nativeAssistantProjectionFixture(assistantHidden, targetHidden*2))
	pair := &AssistantPair{
		TargetArch: model.Arch{Hidden: targetHidden, Vocab: vocab},
		Assistant: &AssistantModel{
			Arch:               model.Arch{Hidden: assistantHidden},
			BackboneHiddenSize: targetHidden,
			Tensors: map[string]safetensors.Tensor{
				"pre_projection.weight": {Dtype: "BF16", Shape: []int{assistantHidden, targetHidden * 2}, Data: preWeight},
			},
		},
	}
	targetEmbed := toBF16Bytes(syntheticFloat32(vocab*targetHidden, 811))
	previousHidden := toBF16Bytes(syntheticFloat32(targetHidden, 823))
	tokenEmbedding := make([]byte, targetHidden*bf16Size)
	if _, err := embedTokenBF16Into(tokenEmbedding, targetEmbed, 3, vocab, targetHidden, embedScaleOf(pair.TargetArch)); err != nil {
		t.Fatalf("embedTokenBF16Into: %v", err)
	}
	directOut := make([]byte, assistantHidden*bf16Size)
	tokenOut := make([]byte, assistantHidden*bf16Size)
	if _, err := pair.Assistant.DraftInputProjectionInto(directOut, tokenEmbedding, previousHidden); err != nil {
		t.Fatalf("warm DraftInputProjectionInto: %v", err)
	}
	if _, err := pair.DraftInputProjectionForTokenInto(tokenOut, targetEmbed, 3, previousHidden); err != nil {
		t.Fatalf("warm DraftInputProjectionForTokenInto: %v", err)
	}

	directAllocs := testing.AllocsPerRun(10, func() {
		if _, err := pair.Assistant.DraftInputProjectionInto(directOut, tokenEmbedding, previousHidden); err != nil {
			t.Fatalf("DraftInputProjectionInto: %v", err)
		}
	})
	tokenAllocs := testing.AllocsPerRun(10, func() {
		if _, err := pair.DraftInputProjectionForTokenInto(tokenOut, targetEmbed, 3, previousHidden); err != nil {
			t.Fatalf("DraftInputProjectionForTokenInto: %v", err)
		}
	})
	if tokenAllocs > directAllocs+0.5 {
		t.Fatalf("DraftInputProjectionForTokenInto allocations/run = %.0f, want direct budget %.0f", tokenAllocs, directAllocs)
	}
}

func TestAssistantPairDraftInputProjectionForQuantTokenUsesScaledTargetEmbedding(t *testing.T) {
	requireNativeRuntime(t)

	targetDir := writeNativeAssistantTargetDir(t, 8, []string{"sliding_attention", "full_attention"})
	tensors := nativeAssistantTinyTensors(true)
	preW := nativeAssistantProjectionFixture(4, 16)
	tensors["pre_projection.weight"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{4, 16}, Data: toBF16Bytes(preW)}
	assistantDir := writeNativeAssistantDir(t, tensors)

	pair, err := LoadAssistantPairDirs(targetDir, assistantDir)
	if err != nil {
		t.Fatalf("LoadAssistantPairDirs: %v", err)
	}
	defer pair.Close()

	const groupSize, bits = 4, 4
	packed, scales, biases := nativeAssistantQuantEmbeddingFixture(8, 8, groupSize)
	previousHidden := toBF16Bytes([]float32{0.5, -1.5, 2, 1, -0.25, 0.75, -1, 0.125})

	got, err := pair.DraftInputProjectionForTokenQuant(packed, scales, biases, groupSize, bits, 3, previousHidden)
	if err != nil {
		t.Fatalf("DraftInputProjectionForTokenQuant: %v", err)
	}

	embedding, err := EmbedTokensQuant(packed, scales, biases, []int32{3}, pair.TargetArch.Vocab, pair.TargetArch.Hidden, groupSize, bits, embedScaleOf(pair.TargetArch))
	if err != nil {
		t.Fatalf("EmbedTokensQuant reference: %v", err)
	}
	combined := append(append([]byte{}, embedding[0]...), previousHidden...)
	want := nativeAssistantMatMulBF16NTReference(combined, toBF16Bytes(preW), 1, 16, 4)
	assertFloat32Near(t, "pair draft input projection for quant token", bf16Floats(got), want, 0.02)
}

func TestAssistantPairDraftInputProjectionForQuantTokenIntoLargeEmbeddingMatchesDirectAllocationBudget(t *testing.T) {
	requireNativeRuntime(t)

	const targetHidden, assistantHidden, vocab, groupSize, bits = 2048, 4, 8, 32, 4
	preWeight := toBF16Bytes(nativeAssistantProjectionFixture(assistantHidden, targetHidden*2))
	pair := &AssistantPair{
		TargetArch: model.Arch{Hidden: targetHidden, Vocab: vocab},
		Assistant: &AssistantModel{
			Arch:               model.Arch{Hidden: assistantHidden},
			BackboneHiddenSize: targetHidden,
			Tensors: map[string]safetensors.Tensor{
				"pre_projection.weight": {Dtype: "BF16", Shape: []int{assistantHidden, targetHidden * 2}, Data: preWeight},
			},
		},
	}
	packed, scales, biases := nativeAssistantQuantEmbeddingFixture(vocab, targetHidden, groupSize)
	previousHidden := toBF16Bytes(syntheticFloat32(targetHidden, 829))
	tokenEmbedding := make([]byte, targetHidden*bf16Size)
	if _, err := embedTokenQuantInto(tokenEmbedding, packed, scales, biases, 3, vocab, targetHidden, groupSize, bits, embedScaleOf(pair.TargetArch)); err != nil {
		t.Fatalf("embedTokenQuantInto: %v", err)
	}
	directOut := make([]byte, assistantHidden*bf16Size)
	tokenOut := make([]byte, assistantHidden*bf16Size)
	if _, err := pair.Assistant.DraftInputProjectionInto(directOut, tokenEmbedding, previousHidden); err != nil {
		t.Fatalf("warm DraftInputProjectionInto: %v", err)
	}
	if _, err := pair.DraftInputProjectionForTokenQuantInto(tokenOut, packed, scales, biases, groupSize, bits, 3, previousHidden); err != nil {
		t.Fatalf("warm DraftInputProjectionForTokenQuantInto: %v", err)
	}

	directAllocs := testing.AllocsPerRun(10, func() {
		if _, err := pair.Assistant.DraftInputProjectionInto(directOut, tokenEmbedding, previousHidden); err != nil {
			t.Fatalf("DraftInputProjectionInto: %v", err)
		}
	})
	tokenAllocs := testing.AllocsPerRun(10, func() {
		if _, err := pair.DraftInputProjectionForTokenQuantInto(tokenOut, packed, scales, biases, groupSize, bits, 3, previousHidden); err != nil {
			t.Fatalf("DraftInputProjectionForTokenQuantInto: %v", err)
		}
	})
	if tokenAllocs > directAllocs+0.5 {
		t.Fatalf("DraftInputProjectionForTokenQuantInto allocations/run = %.0f, want direct budget %.0f", tokenAllocs, directAllocs)
	}
}

func TestAssistantDraftOutputProjectionMatchesReference(t *testing.T) {
	requireNativeRuntime(t)

	tensors := nativeAssistantTinyTensors(true)
	postW := nativeAssistantProjectionFixture(8, 4)
	tensors["post_projection.weight"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{8, 4}, Data: toBF16Bytes(postW)}
	dir := writeNativeAssistantDir(t, tensors)

	assistant, err := LoadAssistantDir(dir)
	if err != nil {
		t.Fatalf("LoadAssistantDir: %v", err)
	}
	defer assistant.Close()

	assistantHidden := toBF16Bytes([]float32{1, -0.5, 0.25, 2})
	got, err := assistant.DraftOutputProjection(assistantHidden)
	if err != nil {
		t.Fatalf("DraftOutputProjection: %v", err)
	}

	want := nativeAssistantMatMulBF16NTReference(assistantHidden, toBF16Bytes(postW), 1, 4, 8)
	assertFloat32Near(t, "draft output projection", bf16Floats(got), want, 0.02)
}

func TestAssistantDraftOutputProjectionIntoAllocationBudget(t *testing.T) {
	requireNativeRuntime(t)

	tensors := nativeAssistantTinyTensors(true)
	postW := nativeAssistantProjectionFixture(8, 4)
	tensors["post_projection.weight"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{8, 4}, Data: toBF16Bytes(postW)}
	dir := writeNativeAssistantDir(t, tensors)

	assistant, err := LoadAssistantDir(dir)
	if err != nil {
		t.Fatalf("LoadAssistantDir: %v", err)
	}
	defer assistant.Close()

	assistantHidden := toBF16Bytes([]float32{1, -1, 0.5, 2})
	out := make([]byte, assistant.BackboneHiddenSize*bf16Size)
	if _, err := assistant.DraftOutputProjectionInto(out, assistantHidden); err != nil {
		t.Fatalf("warm DraftOutputProjectionInto: %v", err)
	}

	allocs := testing.AllocsPerRun(20, func() {
		if _, err := assistant.DraftOutputProjectionInto(out, assistantHidden); err != nil {
			t.Fatalf("DraftOutputProjectionInto: %v", err)
		}
	})
	if allocs > 10 {
		t.Fatalf("DraftOutputProjectionInto allocations/run = %.0f, want <= 10 with caller output", allocs)
	}
}

func TestAssistantDraftFinalNormMatchesRMSNorm(t *testing.T) {
	requireNativeRuntime(t)

	tensors := nativeAssistantTinyTensors(true)
	normW := []float32{1, 0.75, 1.25, 0.5}
	tensors["model.norm.weight"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{4}, Data: toBF16Bytes(normW)}
	dir := writeNativeAssistantDir(t, tensors)

	assistant, err := LoadAssistantDir(dir)
	if err != nil {
		t.Fatalf("LoadAssistantDir: %v", err)
	}
	defer assistant.Close()

	hidden := toBF16Bytes([]float32{1, -0.5, 0.25, 2})
	got, err := assistant.DraftFinalNorm(hidden)
	if err != nil {
		t.Fatalf("DraftFinalNorm: %v", err)
	}
	want, err := RMSNormBF16(hidden, toBF16Bytes(normW), 1, 4, assistant.Arch.Eps)
	if err != nil {
		t.Fatalf("RMSNormBF16 reference: %v", err)
	}
	assertFloat32Near(t, "draft final norm", bf16Floats(got), bf16Floats(want), 0)
}

func TestAssistantDraftAttentionMatchesTargetKVPrimitivePath(t *testing.T) {
	requireNativeRuntime(t)

	const hidden, nHeads, kvHeads, headDim, kvLen = 128, 2, 2, 64, 3
	tensors := nativeAssistantAttentionTensors()
	qW := nativeAssistantProjectionFixture(nHeads*headDim, hidden)
	oW := nativeAssistantProjectionFixture(hidden, nHeads*headDim)
	qNorm := syntheticFloat32(headDim, 9)
	tensors["model.layers.0.self_attn.q_proj.weight"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{nHeads * headDim, hidden}, Data: toBF16Bytes(qW)}
	tensors["model.layers.0.self_attn.o_proj.weight"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{hidden, nHeads * headDim}, Data: toBF16Bytes(oW)}
	tensors["model.layers.0.self_attn.q_norm.weight"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{headDim}, Data: toBF16Bytes(qNorm)}
	dir := writeNativeAssistantAttentionDir(t, tensors)

	assistant, err := LoadAssistantDir(dir)
	if err != nil {
		t.Fatalf("LoadAssistantDir: %v", err)
	}
	defer assistant.Close()

	x := toBF16Bytes(syntheticFloat32(hidden, 3))
	targetKV := AssistantTargetKV{
		Key:     toBF16Bytes(syntheticFloat32(kvHeads*kvLen*headDim, 5)),
		Value:   toBF16Bytes(syntheticFloat32(kvHeads*kvLen*headDim, 7)),
		Offset:  2,
		Length:  kvLen,
		KVHeads: kvHeads,
		HeadDim: headDim,
	}

	got, err := assistant.DraftAttention(0, x, targetKV)
	if err != nil {
		t.Fatalf("DraftAttention: %v", err)
	}

	q, err := MatVecBF16(toBF16Bytes(qW), x, nHeads*headDim, hidden)
	if err != nil {
		t.Fatalf("MatVecBF16 q reference: %v", err)
	}
	q, err = RMSNormBF16(q, toBF16Bytes(qNorm), nHeads, headDim, assistant.Arch.Eps)
	if err != nil {
		t.Fatalf("RMSNormBF16 q reference: %v", err)
	}
	// the draft query ropes at the LAST SEEN token's position (Offset+Length-1), the
	// trained constant per the HF reference — see draftAttentionIntoScratch.
	q, err = RoPEDimsBF16(q, 1, nHeads, headDim, headDim, assistant.Arch.RopeLocalBase, 1, targetKV.Offset+targetKV.Length-1, false)
	if err != nil {
		t.Fatalf("RoPEDimsBF16 q reference: %v", err)
	}
	attn, err := SDPA(q, targetKV.Key, targetKV.Value, 1, nHeads, kvHeads, headDim, targetKV.Length, nativeAssistantAttentionScale(assistant))
	if err != nil {
		t.Fatalf("SDPA reference: %v", err)
	}
	want, err := MatVecBF16(toBF16Bytes(oW), attn, hidden, nHeads*headDim)
	if err != nil {
		t.Fatalf("MatVecBF16 o reference: %v", err)
	}
	assertFloat32Near(t, "draft attention target kv path", bf16Floats(got), bf16Floats(want), 0)
}

func TestAssistantDraftLayerMatchesComposedPrimitivePath(t *testing.T) {
	requireNativeRuntime(t)

	const hidden, nHeads, kvHeads, headDim, kvLen, dFF = 128, 2, 2, 64, 3, 256
	tensors := nativeAssistantAttentionTensors()
	inputNorm := syntheticFloat32(hidden, 11)
	postAttnNorm := syntheticFloat32(hidden, 13)
	preFFNorm := syntheticFloat32(hidden, 17)
	postFFNorm := syntheticFloat32(hidden, 19)
	qW := nativeAssistantProjectionFixture(nHeads*headDim, hidden)
	oW := nativeAssistantProjectionFixture(hidden, nHeads*headDim)
	qNorm := syntheticFloat32(headDim, 23)
	gateW := nativeAssistantProjectionFixture(dFF, hidden)
	upW := nativeAssistantProjectionFixture(dFF, hidden)
	downW := nativeAssistantProjectionFixture(hidden, dFF)
	scalar := []float32{0.75}
	p := "model.layers.0"
	tensors[p+".input_layernorm.weight"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{hidden}, Data: toBF16Bytes(inputNorm)}
	tensors[p+".post_attention_layernorm.weight"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{hidden}, Data: toBF16Bytes(postAttnNorm)}
	tensors[p+".pre_feedforward_layernorm.weight"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{hidden}, Data: toBF16Bytes(preFFNorm)}
	tensors[p+".post_feedforward_layernorm.weight"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{hidden}, Data: toBF16Bytes(postFFNorm)}
	tensors[p+".layer_scalar"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{1}, Data: toBF16Bytes(scalar)}
	tensors[p+".self_attn.q_proj.weight"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{nHeads * headDim, hidden}, Data: toBF16Bytes(qW)}
	tensors[p+".self_attn.o_proj.weight"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{hidden, nHeads * headDim}, Data: toBF16Bytes(oW)}
	tensors[p+".self_attn.q_norm.weight"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{headDim}, Data: toBF16Bytes(qNorm)}
	tensors[p+".mlp.gate_proj.weight"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{dFF, hidden}, Data: toBF16Bytes(gateW)}
	tensors[p+".mlp.up_proj.weight"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{dFF, hidden}, Data: toBF16Bytes(upW)}
	tensors[p+".mlp.down_proj.weight"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{hidden, dFF}, Data: toBF16Bytes(downW)}
	dir := writeNativeAssistantAttentionDir(t, tensors)

	assistant, err := LoadAssistantDir(dir)
	if err != nil {
		t.Fatalf("LoadAssistantDir: %v", err)
	}
	defer assistant.Close()

	x := toBF16Bytes(syntheticFloat32(hidden, 29))
	targetKV := AssistantTargetKV{
		Key:     toBF16Bytes(syntheticFloat32(kvHeads*kvLen*headDim, 31)),
		Value:   toBF16Bytes(syntheticFloat32(kvHeads*kvLen*headDim, 37)),
		Offset:  4,
		Length:  kvLen,
		KVHeads: kvHeads,
		HeadDim: headDim,
	}
	got, err := assistant.DraftLayer(0, x, targetKV)
	if err != nil {
		t.Fatalf("DraftLayer: %v", err)
	}

	normed, err := RMSNormBF16(x, toBF16Bytes(inputNorm), 1, hidden, assistant.Arch.Eps)
	if err != nil {
		t.Fatalf("input RMSNormBF16 reference: %v", err)
	}
	attnOut, err := assistant.DraftAttention(0, normed, targetKV)
	if err != nil {
		t.Fatalf("DraftAttention reference: %v", err)
	}
	attnResidual, err := RMSNormBF16(attnOut, toBF16Bytes(postAttnNorm), 1, hidden, assistant.Arch.Eps)
	if err != nil {
		t.Fatalf("post-attention RMSNormBF16 reference: %v", err)
	}
	h, err := AddBF16(x, attnResidual)
	if err != nil {
		t.Fatalf("attention residual AddBF16 reference: %v", err)
	}
	ffIn, err := RMSNormBF16(h, toBF16Bytes(preFFNorm), 1, hidden, assistant.Arch.Eps)
	if err != nil {
		t.Fatalf("pre-FF RMSNormBF16 reference: %v", err)
	}
	gate, err := MatVecBF16(toBF16Bytes(gateW), ffIn, dFF, hidden)
	if err != nil {
		t.Fatalf("gate MatVecBF16 reference: %v", err)
	}
	up, err := MatVecBF16(toBF16Bytes(upW), ffIn, dFF, hidden)
	if err != nil {
		t.Fatalf("up MatVecBF16 reference: %v", err)
	}
	gated, err := GeluGateMulBF16(gate, up)
	if err != nil {
		t.Fatalf("GeluGateMulBF16 reference: %v", err)
	}
	ff, err := MatVecBF16(toBF16Bytes(downW), gated, hidden, dFF)
	if err != nil {
		t.Fatalf("down MatVecBF16 reference: %v", err)
	}
	ffResidual, err := RMSNormBF16(ff, toBF16Bytes(postFFNorm), 1, hidden, assistant.Arch.Eps)
	if err != nil {
		t.Fatalf("post-FF RMSNormBF16 reference: %v", err)
	}
	want, err := AddBF16(h, ffResidual)
	if err != nil {
		t.Fatalf("FF residual AddBF16 reference: %v", err)
	}
	want, err = MulScalarBF16(want, toBF16Bytes(scalar))
	if err != nil {
		t.Fatalf("MulScalarBF16 reference: %v", err)
	}
	assertFloat32Near(t, "draft layer primitive path", bf16Floats(got), bf16Floats(want), 0)
}

func TestAssistantDraftStepActivationsRunsLayerStackAndPostProjection(t *testing.T) {
	requireNativeRuntime(t)

	const hidden, backbone, nHeads, kvHeads, headDim, kvLen, dFF = 128, 8, 2, 2, 64, 3, 256
	tensors := nativeAssistantAttentionTensors()
	p := "model.layers.0"
	tensors["model.norm.weight"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{hidden}, Data: toBF16Bytes(syntheticFloat32(hidden, 41))}
	tensors["post_projection.weight"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{backbone, hidden}, Data: toBF16Bytes(nativeAssistantProjectionFixture(backbone, hidden))}
	tensors[p+".input_layernorm.weight"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{hidden}, Data: toBF16Bytes(syntheticFloat32(hidden, 43))}
	tensors[p+".post_attention_layernorm.weight"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{hidden}, Data: toBF16Bytes(syntheticFloat32(hidden, 47))}
	tensors[p+".pre_feedforward_layernorm.weight"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{hidden}, Data: toBF16Bytes(syntheticFloat32(hidden, 53))}
	tensors[p+".post_feedforward_layernorm.weight"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{hidden}, Data: toBF16Bytes(syntheticFloat32(hidden, 59))}
	tensors[p+".layer_scalar"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{1}, Data: toBF16Bytes([]float32{0.5})}
	tensors[p+".self_attn.q_proj.weight"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{nHeads * headDim, hidden}, Data: toBF16Bytes(nativeAssistantProjectionFixture(nHeads*headDim, hidden))}
	tensors[p+".self_attn.o_proj.weight"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{hidden, nHeads * headDim}, Data: toBF16Bytes(nativeAssistantProjectionFixture(hidden, nHeads*headDim))}
	tensors[p+".self_attn.q_norm.weight"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{headDim}, Data: toBF16Bytes(syntheticFloat32(headDim, 61))}
	tensors[p+".mlp.gate_proj.weight"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{dFF, hidden}, Data: toBF16Bytes(nativeAssistantProjectionFixture(dFF, hidden))}
	tensors[p+".mlp.up_proj.weight"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{dFF, hidden}, Data: toBF16Bytes(nativeAssistantProjectionFixture(dFF, hidden))}
	tensors[p+".mlp.down_proj.weight"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{hidden, dFF}, Data: toBF16Bytes(nativeAssistantProjectionFixture(hidden, dFF))}
	dir := writeNativeAssistantAttentionDir(t, tensors)

	assistant, err := LoadAssistantDir(dir)
	if err != nil {
		t.Fatalf("LoadAssistantDir: %v", err)
	}
	defer assistant.Close()

	projectedHidden := toBF16Bytes(syntheticFloat32(hidden, 67))
	targetKV := AssistantTargetKV{
		Key:     toBF16Bytes(syntheticFloat32(kvHeads*kvLen*headDim, 71)),
		Value:   toBF16Bytes(syntheticFloat32(kvHeads*kvLen*headDim, 73)),
		Offset:  5,
		Length:  kvLen,
		KVHeads: kvHeads,
		HeadDim: headDim,
	}
	targetKVs := AssistantTargetKVByType{}
	targetKVs.set("sliding_attention", targetKV)

	gotNormed, gotHidden, err := assistant.DraftStepActivations(projectedHidden, targetKVs)
	if err != nil {
		t.Fatalf("DraftStepActivations: %v", err)
	}

	layerOut, err := assistant.DraftLayer(0, projectedHidden, targetKV)
	if err != nil {
		t.Fatalf("DraftLayer reference: %v", err)
	}
	wantNormed, err := assistant.DraftFinalNorm(layerOut)
	if err != nil {
		t.Fatalf("DraftFinalNorm reference: %v", err)
	}
	wantHidden, err := assistant.DraftOutputProjection(wantNormed)
	if err != nil {
		t.Fatalf("DraftOutputProjection reference: %v", err)
	}
	assertFloat32Near(t, "draft step normed activations", bf16Floats(gotNormed), bf16Floats(wantNormed), 0)
	assertFloat32Near(t, "draft step target hidden", bf16Floats(gotHidden), bf16Floats(wantHidden), 0)
}

func TestAssistantPairDraftStepUsesTokenAndTargetKVPath(t *testing.T) {
	requireNativeRuntime(t)

	const hidden, backbone, nHeads, kvHeads, headDim, kvLen, dFF, vocab = 128, 8, 2, 2, 64, 3, 256, 8
	targetDir := writeNativeAssistantAttentionTargetDir(t)
	tensors := nativeAssistantAttentionTensors()
	p := "model.layers.0"
	tensors["model.embed_tokens.weight"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{vocab, hidden}, Data: toBF16Bytes(nativeAssistantProjectionFixture(vocab, hidden))}
	tensors["model.norm.weight"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{hidden}, Data: toBF16Bytes(syntheticFloat32(hidden, 83))}
	tensors["pre_projection.weight"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{hidden, backbone * 2}, Data: toBF16Bytes(nativeAssistantProjectionFixture(hidden, backbone*2))}
	tensors["post_projection.weight"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{backbone, hidden}, Data: toBF16Bytes(nativeAssistantProjectionFixture(backbone, hidden))}
	tensors[p+".input_layernorm.weight"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{hidden}, Data: toBF16Bytes(syntheticFloat32(hidden, 89))}
	tensors[p+".post_attention_layernorm.weight"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{hidden}, Data: toBF16Bytes(syntheticFloat32(hidden, 97))}
	tensors[p+".pre_feedforward_layernorm.weight"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{hidden}, Data: toBF16Bytes(syntheticFloat32(hidden, 101))}
	tensors[p+".post_feedforward_layernorm.weight"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{hidden}, Data: toBF16Bytes(syntheticFloat32(hidden, 103))}
	tensors[p+".layer_scalar"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{1}, Data: toBF16Bytes([]float32{0.625})}
	tensors[p+".self_attn.q_proj.weight"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{nHeads * headDim, hidden}, Data: toBF16Bytes(nativeAssistantProjectionFixture(nHeads*headDim, hidden))}
	tensors[p+".self_attn.o_proj.weight"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{hidden, nHeads * headDim}, Data: toBF16Bytes(nativeAssistantProjectionFixture(hidden, nHeads*headDim))}
	tensors[p+".self_attn.q_norm.weight"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{headDim}, Data: toBF16Bytes(syntheticFloat32(headDim, 107))}
	tensors[p+".mlp.gate_proj.weight"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{dFF, hidden}, Data: toBF16Bytes(nativeAssistantProjectionFixture(dFF, hidden))}
	tensors[p+".mlp.up_proj.weight"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{dFF, hidden}, Data: toBF16Bytes(nativeAssistantProjectionFixture(dFF, hidden))}
	tensors[p+".mlp.down_proj.weight"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{hidden, dFF}, Data: toBF16Bytes(nativeAssistantProjectionFixture(hidden, dFF))}
	assistantDir := writeNativeAssistantAttentionDir(t, tensors)

	pair, err := LoadAssistantPairDirs(targetDir, assistantDir)
	if err != nil {
		t.Fatalf("LoadAssistantPairDirs: %v", err)
	}
	defer pair.Close()

	targetEmbed := toBF16Bytes(syntheticFloat32(vocab*backbone, 109))
	previousHidden := toBF16Bytes(syntheticFloat32(backbone, 113))
	targetKV := AssistantTargetKV{
		Key:     toBF16Bytes(syntheticFloat32(kvHeads*kvLen*headDim, 127)),
		Value:   toBF16Bytes(syntheticFloat32(kvHeads*kvLen*headDim, 131)),
		Offset:  6,
		Length:  kvLen,
		KVHeads: kvHeads,
		HeadDim: headDim,
	}
	targetKVs := AssistantTargetKVByType{}
	targetKVs.set("sliding_attention", targetKV)

	got, err := pair.DraftStep(targetEmbed, 3, previousHidden, targetKVs)
	if err != nil {
		t.Fatalf("DraftStep: %v", err)
	}

	projected, err := pair.DraftInputProjectionForToken(targetEmbed, 3, previousHidden)
	if err != nil {
		t.Fatalf("DraftInputProjectionForToken reference: %v", err)
	}
	normed, hiddenOut, err := pair.Assistant.DraftStepActivations(projected, targetKVs)
	if err != nil {
		t.Fatalf("DraftStepActivations reference: %v", err)
	}
	logits, err := pair.Assistant.DraftLogits(normed)
	if err != nil {
		t.Fatalf("DraftLogits reference: %v", err)
	}
	token, err := pair.Assistant.DraftGreedyToken(logits)
	if err != nil {
		t.Fatalf("DraftGreedyToken reference: %v", err)
	}
	if got.Token != token {
		t.Fatalf("DraftStep token = %d, want %d", got.Token, token)
	}
	assertFloat32Near(t, "draft step logits", bf16Floats(got.Logits), bf16Floats(logits), 0)
	assertFloat32Near(t, "draft step hidden", bf16Floats(got.Hidden), bf16Floats(hiddenOut), 0)
}

func TestAssistantPairDraftStepQuantUsesTokenAndTargetKVPath(t *testing.T) {
	requireNativeRuntime(t)

	const hidden, backbone, nHeads, kvHeads, headDim, kvLen, dFF, vocab = 128, 8, 2, 2, 64, 3, 256, 8
	targetDir := writeNativeAssistantAttentionTargetDir(t)
	tensors := nativeAssistantAttentionTensors()
	p := "model.layers.0"
	tensors["model.embed_tokens.weight"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{vocab, hidden}, Data: toBF16Bytes(nativeAssistantProjectionFixture(vocab, hidden))}
	tensors["model.norm.weight"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{hidden}, Data: toBF16Bytes(syntheticFloat32(hidden, 137))}
	tensors["pre_projection.weight"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{hidden, backbone * 2}, Data: toBF16Bytes(nativeAssistantProjectionFixture(hidden, backbone*2))}
	tensors["post_projection.weight"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{backbone, hidden}, Data: toBF16Bytes(nativeAssistantProjectionFixture(backbone, hidden))}
	tensors[p+".input_layernorm.weight"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{hidden}, Data: toBF16Bytes(syntheticFloat32(hidden, 139))}
	tensors[p+".post_attention_layernorm.weight"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{hidden}, Data: toBF16Bytes(syntheticFloat32(hidden, 149))}
	tensors[p+".pre_feedforward_layernorm.weight"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{hidden}, Data: toBF16Bytes(syntheticFloat32(hidden, 151))}
	tensors[p+".post_feedforward_layernorm.weight"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{hidden}, Data: toBF16Bytes(syntheticFloat32(hidden, 157))}
	tensors[p+".layer_scalar"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{1}, Data: toBF16Bytes([]float32{0.875})}
	tensors[p+".self_attn.q_proj.weight"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{nHeads * headDim, hidden}, Data: toBF16Bytes(nativeAssistantProjectionFixture(nHeads*headDim, hidden))}
	tensors[p+".self_attn.o_proj.weight"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{hidden, nHeads * headDim}, Data: toBF16Bytes(nativeAssistantProjectionFixture(hidden, nHeads*headDim))}
	tensors[p+".self_attn.q_norm.weight"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{headDim}, Data: toBF16Bytes(syntheticFloat32(headDim, 163))}
	tensors[p+".mlp.gate_proj.weight"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{dFF, hidden}, Data: toBF16Bytes(nativeAssistantProjectionFixture(dFF, hidden))}
	tensors[p+".mlp.up_proj.weight"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{dFF, hidden}, Data: toBF16Bytes(nativeAssistantProjectionFixture(dFF, hidden))}
	tensors[p+".mlp.down_proj.weight"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{hidden, dFF}, Data: toBF16Bytes(nativeAssistantProjectionFixture(hidden, dFF))}
	assistantDir := writeNativeAssistantAttentionDir(t, tensors)

	pair, err := LoadAssistantPairDirs(targetDir, assistantDir)
	if err != nil {
		t.Fatalf("LoadAssistantPairDirs: %v", err)
	}
	defer pair.Close()

	const groupSize, bits = 4, 4
	packed, scales, biases := nativeAssistantQuantEmbeddingFixture(vocab, backbone, groupSize)
	previousHidden := toBF16Bytes(syntheticFloat32(backbone, 167))
	targetKV := AssistantTargetKV{
		Key:     toBF16Bytes(syntheticFloat32(kvHeads*kvLen*headDim, 173)),
		Value:   toBF16Bytes(syntheticFloat32(kvHeads*kvLen*headDim, 179)),
		Offset:  7,
		Length:  kvLen,
		KVHeads: kvHeads,
		HeadDim: headDim,
	}
	targetKVs := AssistantTargetKVByType{}
	targetKVs.set("sliding_attention", targetKV)

	got, err := pair.DraftStepQuant(packed, scales, biases, groupSize, bits, 4, previousHidden, targetKVs)
	if err != nil {
		t.Fatalf("DraftStepQuant: %v", err)
	}

	projected, err := pair.DraftInputProjectionForTokenQuant(packed, scales, biases, groupSize, bits, 4, previousHidden)
	if err != nil {
		t.Fatalf("DraftInputProjectionForTokenQuant reference: %v", err)
	}
	normed, hiddenOut, err := pair.Assistant.DraftStepActivations(projected, targetKVs)
	if err != nil {
		t.Fatalf("DraftStepActivations reference: %v", err)
	}
	logits, err := pair.Assistant.DraftLogits(normed)
	if err != nil {
		t.Fatalf("DraftLogits reference: %v", err)
	}
	token, err := pair.Assistant.DraftGreedyToken(logits)
	if err != nil {
		t.Fatalf("DraftGreedyToken reference: %v", err)
	}
	if got.Token != token {
		t.Fatalf("DraftStepQuant token = %d, want %d", got.Token, token)
	}
	assertFloat32Near(t, "draft step quant logits", bf16Floats(got.Logits), bf16Floats(logits), 0)
	assertFloat32Near(t, "draft step quant hidden", bf16Floats(got.Hidden), bf16Floats(hiddenOut), 0)
}

func TestAssistantPairDraftStepFromSessionMatchesExplicitPath(t *testing.T) {
	requireNativeRuntime(t)

	targetDir := writeNativeAssistantAttentionTargetDir(t)
	assistantDir := writeNativeAssistantAttentionDir(t, nativeAssistantAttentionTensors())
	pair, err := LoadAssistantPairDirs(targetDir, assistantDir)
	if err != nil {
		t.Fatalf("LoadAssistantPairDirs: %v", err)
	}
	defer pair.Close()

	arch := pair.TargetArch
	kvHeads := arch.Layer[0].KVHeads
	if kvHeads <= 0 {
		kvHeads = arch.KVHeads
	}
	headDim := arch.Layer[0].HeadDim
	if headDim <= 0 {
		headDim = arch.HeadDim
	}
	rowBytes := kvHeads * headDim * bf16Size
	tokenEmbedding := toBF16Bytes(syntheticFloat32(arch.Hidden, 83))
	retainedHidden := toBF16Bytes(syntheticFloat32(arch.Hidden, 89))
	finalNorm := toBF16Bytes(syntheticFloat32(arch.Hidden, 97))
	session := &ArchSession{
		arch: arch,
		state: archDecodeState{
			specs: arch.Layer,
		},
		stateBlockViews: []sessionStateLayerView{
			{
				layer: 0, kvHeads: kvHeads, headDim: headDim, rowBytes: rowBytes, cacheIndex: 0,
				cacheMode: nativeStateCacheModeFixed, cacheRows: 4,
				keyBytes:   nativeAssistantSessionKVRowsForTest(4, kvHeads, headDim, 0x10),
				valueBytes: nativeAssistantSessionKVRowsForTest(4, kvHeads, headDim, 0x20),
			},
		},
		pos:            3,
		maxLen:         4,
		retainedHidden: retainedHidden,
		finalNorm:      finalNorm,
	}
	session.embedInto = func(dst []byte, id int32) ([]byte, error) {
		if id != 5 {
			return nil, core.NewError("unexpected token id")
		}
		if len(dst) < len(tokenEmbedding) {
			return nil, core.NewError("short embedding destination")
		}
		copy(dst, tokenEmbedding)
		return dst[:len(tokenEmbedding)], nil
	}

	targetKVs, err := pair.TargetKVByLayerTypeFromSession(session)
	if err != nil {
		t.Fatalf("TargetKVByLayerTypeFromSession: %v", err)
	}
	previousHidden, err := RMSNormBF16(retainedHidden, finalNorm, 1, arch.Hidden, arch.Eps)
	if err != nil {
		t.Fatalf("RMSNormBF16 boundary reference: %v", err)
	}
	projected, err := pair.Assistant.DraftInputProjection(tokenEmbedding, previousHidden)
	if err != nil {
		t.Fatalf("DraftInputProjection reference: %v", err)
	}
	want, err := pair.draftStepFromProjected(projected, targetKVs)
	if err != nil {
		t.Fatalf("draftStepFromProjected reference: %v", err)
	}

	got, err := pair.DraftStepFromSession(session, 5)
	if err != nil {
		t.Fatalf("DraftStepFromSession: %v", err)
	}
	if got.Token != want.Token {
		t.Fatalf("DraftStepFromSession token = %d, want %d", got.Token, want.Token)
	}
	eqBytes(t, "DraftStepFromSession logits", got.Logits, want.Logits)
	eqBytes(t, "DraftStepFromSession hidden", got.Hidden, want.Hidden)
}

func TestAssistantPairDraftStepFromSessionKeepsProjectionScratch(t *testing.T) {
	requireNativeRuntime(t)

	pair, mk := newNativeAssistantGenerateFixture(t)
	defer pair.Close()
	session := mk()
	prompt := []int32{1, 5, 3}
	if err := session.PrefillTokens(prompt); err != nil {
		t.Fatalf("PrefillTokens: %v", err)
	}

	first, err := pair.DraftStepFromSession(session, prompt[len(prompt)-1])
	if err != nil {
		t.Fatalf("DraftStepFromSession: %v", err)
	}
	if len(first.Logits) == 0 || len(first.Hidden) == 0 {
		t.Fatal("DraftStepFromSession returned empty logits or hidden state")
	}
	logitsPtr := byteDataPointer(first.Logits)
	hiddenPtr := byteDataPointer(first.Hidden)
	if len(session.mtpDraftLayerScratch.inputNorm) == 0 ||
		len(session.mtpDraftLayerScratch.attnQ) == 0 ||
		len(session.mtpDraftLayerScratch.gate) == 0 {
		t.Fatal("DraftStepFromSession did not warm layer temporary scratch")
	}
	inputNormPtr := byteDataPointer(session.mtpDraftLayerScratch.inputNorm)
	attnQPtr := byteDataPointer(session.mtpDraftLayerScratch.attnQ)
	gatePtr := byteDataPointer(session.mtpDraftLayerScratch.gate)
	if _, ok := registeredPinnedNoCopyBytes(session.mtpDraftLayerScratch.inputNorm); !ok {
		t.Fatal("DraftStepFromSession layer input norm scratch is not registered pinned backing")
	}
	if _, ok := registeredPinnedNoCopyBytes(session.mtpDraftLayerScratch.attnQ); !ok {
		t.Fatal("DraftStepFromSession layer attention query scratch is not registered pinned backing")
	}
	if _, ok := registeredPinnedNoCopyBytes(session.mtpDraftLayerScratch.gate); !ok {
		t.Fatal("DraftStepFromSession layer gate scratch is not registered pinned backing")
	}

	second, err := pair.DraftStepFromSession(session, prompt[len(prompt)-1])
	if err != nil {
		t.Fatalf("second DraftStepFromSession: %v", err)
	}
	if byteDataPointer(second.Logits) != logitsPtr || byteDataPointer(second.Hidden) != hiddenPtr {
		t.Fatal("DraftStepFromSession did not reuse session-owned output scratch")
	}
	if byteDataPointer(session.mtpDraftLayerScratch.inputNorm) != inputNormPtr ||
		byteDataPointer(session.mtpDraftLayerScratch.attnQ) != attnQPtr ||
		byteDataPointer(session.mtpDraftLayerScratch.gate) != gatePtr {
		t.Fatal("DraftStepFromSession did not reuse session-owned layer temporary scratch")
	}

	want := pair.Assistant.Arch.Hidden * bf16Size
	if cap(session.mtpProjected) < want {
		t.Fatalf("MTP projection scratch cap = %d, want at least %d", cap(session.mtpProjected), want)
	}
}

func TestAssistantPairDraftBlockFromSessionMatchesRepeatedSteps(t *testing.T) {
	requireNativeRuntime(t)

	targetDir := writeNativeAssistantAttentionTargetDir(t)
	assistantDir := writeNativeAssistantAttentionDir(t, nativeAssistantAttentionTensors())
	pair, err := LoadAssistantPairDirs(targetDir, assistantDir)
	if err != nil {
		t.Fatalf("LoadAssistantPairDirs: %v", err)
	}
	defer pair.Close()

	arch := pair.TargetArch
	kvHeads := arch.Layer[0].KVHeads
	if kvHeads <= 0 {
		kvHeads = arch.KVHeads
	}
	headDim := arch.Layer[0].HeadDim
	if headDim <= 0 {
		headDim = arch.HeadDim
	}
	rowBytes := kvHeads * headDim * bf16Size
	retainedHidden := toBF16Bytes(syntheticFloat32(arch.Hidden, 191))
	finalNorm := toBF16Bytes(syntheticFloat32(arch.Hidden, 193))
	session := &ArchSession{
		arch: arch,
		state: archDecodeState{
			specs: arch.Layer,
		},
		stateBlockViews: []sessionStateLayerView{
			{
				layer: 0, kvHeads: kvHeads, headDim: headDim, rowBytes: rowBytes, cacheIndex: 0,
				cacheMode: nativeStateCacheModeFixed, cacheRows: 4,
				keyBytes:   nativeAssistantSessionKVRowsForTest(4, kvHeads, headDim, 0x30),
				valueBytes: nativeAssistantSessionKVRowsForTest(4, kvHeads, headDim, 0x40),
			},
		},
		pos:            3,
		maxLen:         4,
		retainedHidden: retainedHidden,
		finalNorm:      finalNorm,
	}
	session.embedInto = func(dst []byte, id int32) ([]byte, error) {
		if len(dst) < arch.Hidden*bf16Size {
			return nil, core.NewError("short embedding destination")
		}
		embedding := toBF16Bytes(syntheticFloat32(arch.Hidden, int(197+id)))
		copy(dst, embedding)
		return dst[:len(embedding)], nil
	}

	got, err := pair.DraftBlockFromSession(session, 5, 2)
	if err != nil {
		t.Fatalf("DraftBlockFromSession: %v", err)
	}

	targetKVs, err := pair.TargetKVByLayerTypeFromSession(session)
	if err != nil {
		t.Fatalf("TargetKVByLayerTypeFromSession: %v", err)
	}
	currentHidden, err := session.BoundaryNormedHidden()
	if err != nil {
		t.Fatalf("BoundaryNormedHidden: %v", err)
	}
	currentToken := int32(5)
	wantTokens := make([]int32, 0, 2)
	for len(wantTokens) < 2 {
		tokenEmbedding, err := session.embedID(currentToken)
		if err != nil {
			t.Fatalf("embedID reference: %v", err)
		}
		projected, err := pair.Assistant.DraftInputProjection(tokenEmbedding, currentHidden)
		if err != nil {
			t.Fatalf("DraftInputProjection reference: %v", err)
		}
		step, err := pair.draftStepFromProjected(projected, targetKVs)
		if err != nil {
			t.Fatalf("draftStepFromProjected reference: %v", err)
		}
		wantTokens = append(wantTokens, step.Token)
		currentToken = step.Token
		currentHidden = step.Hidden
	}
	if !idsEqual(got.Tokens, wantTokens) {
		t.Fatalf("DraftBlockFromSession tokens = %v, want %v", got.Tokens, wantTokens)
	}
	eqBytes(t, "DraftBlockFromSession hidden", got.Hidden, currentHidden)
}

func TestAssistantPairDraftBlockFromSessionCanUseTokenScratch(t *testing.T) {
	requireNativeRuntime(t)

	pair, mk := newNativeAssistantGenerateFixture(t)
	defer pair.Close()
	session := mk()
	prompt := []int32{1, 5, 3}
	if err := session.PrefillTokens(prompt); err != nil {
		t.Fatalf("PrefillTokens: %v", err)
	}

	first, err := pair.draftBlockFromSession(session, prompt[len(prompt)-1], 2, false)
	if err != nil {
		t.Fatalf("draftBlockFromSession scratch first: %v", err)
	}
	if len(first.Tokens) != 2 || len(session.mtpDraftTokens) != 2 {
		t.Fatalf("scratch draft tokens len = %d/session %d, want 2", len(first.Tokens), len(session.mtpDraftTokens))
	}
	tokenPtr := &session.mtpDraftTokens[0]
	if &first.Tokens[0] != tokenPtr {
		t.Fatal("scratch draft block did not return session-owned token backing")
	}

	second, err := pair.draftBlockFromSession(session, prompt[len(prompt)-1], 2, false)
	if err != nil {
		t.Fatalf("draftBlockFromSession scratch second: %v", err)
	}
	if len(second.Tokens) != 2 || &second.Tokens[0] != tokenPtr {
		t.Fatal("scratch draft block did not reuse session-owned token backing")
	}

	public, err := pair.DraftBlockFromSession(session, prompt[len(prompt)-1], 2)
	if err != nil {
		t.Fatalf("DraftBlockFromSession public: %v", err)
	}
	if len(public.Tokens) != 2 {
		t.Fatalf("public draft tokens len = %d, want 2", len(public.Tokens))
	}
	if &public.Tokens[0] == tokenPtr {
		t.Fatal("public DraftBlockFromSession returned session-owned token scratch")
	}
}

// TestAssistantPairDraftBlockSampledFromSessionDraftsGreedily pins the corrected
// sampled-lane draft contract: DRAFTS ARE ALWAYS THE DRAFTER'S ARGMAX, at every
// request temperature — the reference (HF SinglePositionMultiTokenCandidateGenerator)
// drafts greedily and leaves ALL sampling to the target's verify side. The previous
// behaviour (sampling drafts with the request sampler) made proposals random draws
// the sampled target almost never matched; live acceptance collapsed to 0%.
func TestAssistantPairDraftBlockSampledFromSessionDraftsGreedily(t *testing.T) {
	requireNativeRuntime(t)

	pair, mk := newNativeAssistantGenerateFixture(t)
	defer pair.Close()
	prompt := nativeAssistantWordedPromptIDs()
	const draftTokens = 3
	params := model.SampleParams{
		Temperature:    1.3,
		TopK:           4,
		TopP:           0.85,
		MinP:           0.01,
		SuppressTokens: []int32{0},
	}
	greedyTarget := mk()
	if err := greedyTarget.prepareAssistantPrompt(prompt); err != nil {
		t.Fatalf("prepareAssistantPrompt(%q): %v", nativeAssistantWordedPromptText, err)
	}
	pickParams := greedyTarget.mtpSamplePickParams(params, nil, 0)
	greedy, err := pair.draftBlockFromSessionWithSuppress(greedyTarget, prompt[len(prompt)-1], draftTokens, true, pickParams.SuppressTokens)
	if err != nil {
		t.Fatalf("draftBlockFromSessionWithSuppress(%q): %v", nativeAssistantWordedPromptText, err)
	}

	target := mk()
	if err := target.prepareAssistantPrompt(prompt); err != nil {
		t.Fatalf("prepareAssistantPrompt(sampled %q): %v", nativeAssistantWordedPromptText, err)
	}
	got, err := pair.draftBlockSampledFromSessionWithSuppress(target, prompt[len(prompt)-1], draftTokens, true, pickParams, model.NewSampler(7))
	if err != nil {
		t.Fatalf("draftBlockSampledFromSessionWithSuppress(%q): %v", nativeAssistantWordedPromptText, err)
	}
	if !idsEqual(got.Tokens, greedy.Tokens) {
		t.Fatalf("sampled-lane draft tokens = %v, want the drafter's greedy argmax %v (drafts never sample; the target's verify side owns sampling)", got.Tokens, greedy.Tokens)
	}
}

func TestArchSessionAssistantCarryBlockUsesScratch(t *testing.T) {
	session := &ArchSession{}
	draft := []int32{2, 3}

	first := session.mtpDraftVerifyBlockScratch(1, draft)
	if !idsEqual(first, []int32{1, 2, 3}) {
		t.Fatalf("first carry block = %v, want [1 2 3]", first)
	}
	firstPtr := &first[0]

	second := session.mtpDraftVerifyBlockScratch(4, draft[:1])
	if !idsEqual(second, []int32{4, 2}) {
		t.Fatalf("second carry block = %v, want [4 2]", second)
	}
	if &second[0] != firstPtr {
		t.Fatal("carry block did not reuse session-owned scratch")
	}
}

func TestArchSessionAssistantSequentialVerifyHiddensUsePinnedScratch(t *testing.T) {
	requireNativeRuntime(t)
	mk := newMTPDecodeFixture(t)
	session := mtpSequentialFallbackSession(mk())
	for _, id := range []int32{1, 2, 3} {
		if _, err := session.stepID(id); err != nil {
			t.Fatalf("prefill stepID(%d): %v", id, err)
		}
	}

	ids := []int32{4, 5, 6}
	hiddens, err := session.verifyAssistantDraftHiddens(ids)
	if err != nil {
		t.Fatalf("verifyAssistantDraftHiddens: %v", err)
	}
	if len(hiddens) != len(ids) {
		t.Fatalf("hidden rows = %d, want %d", len(hiddens), len(ids))
	}
	if session.mtpVerifyHiddenPinned == nil || session.mtpVerifyHiddenPinned.buf == nil {
		t.Fatal("sequential verify did not retain pinned hidden rows")
	}
	rowBytes := session.arch.Hidden * bf16Size
	for i, hidden := range hiddens {
		if len(hidden) != rowBytes {
			t.Fatalf("hidden row %d bytes = %d, want %d", i, len(hidden), rowBytes)
		}
		if byteDataPointer(hidden) != byteDataPointer(session.mtpVerifyHiddenRows[i]) {
			t.Fatalf("hidden row %d does not reuse session hidden-row scratch", i)
		}
	}
	firstPtr := byteDataPointer(hiddens[0])

	hiddens, err = session.verifyAssistantDraftHiddens(ids)
	if err != nil {
		t.Fatalf("second verifyAssistantDraftHiddens: %v", err)
	}
	if len(hiddens) != len(ids) || byteDataPointer(hiddens[0]) != firstPtr {
		t.Fatal("sequential verify hidden rows did not reuse pinned backing")
	}
}

func TestArchSessionAssistantVerifyRowsUseScratch(t *testing.T) {
	requireNativeRuntime(t)
	mk := newMTPDecodeFixture(t)
	session := mtpSequentialFallbackSession(mk())
	for _, id := range []int32{1, 2, 3} {
		if _, err := session.stepID(id); err != nil {
			t.Fatalf("prefill stepID(%d): %v", id, err)
		}
	}

	ids := []int32{4, 5}
	rows, _, err := session.verifyAssistantDraftRows(ids, nil)
	if err != nil {
		t.Fatalf("verifyAssistantDraftRows: %v", err)
	}
	if len(rows) != len(ids) || len(session.mtpVerifyRows) != len(ids) {
		t.Fatalf("verify rows len = %d/session %d, want %d", len(rows), len(session.mtpVerifyRows), len(ids))
	}
	rowPtr := &rows[0]
	if rowPtr != &session.mtpVerifyRows[0] {
		t.Fatal("verify rows did not use session-owned scratch")
	}

	rows, _, err = session.verifyAssistantDraftRows(ids, nil)
	if err != nil {
		t.Fatalf("second verifyAssistantDraftRows: %v", err)
	}
	if len(rows) != len(ids) || &rows[0] != rowPtr {
		t.Fatal("verify rows did not reuse session-owned scratch")
	}
}

func TestAssistantPairVerifyDraftBlockFromSessionAcceptsFullBlockWithPlainBoundary(t *testing.T) {
	requireNativeRuntime(t)

	mk := newMTPDecodeFixture(t)
	prompt := []int32{1, 5, 3}
	want, err := mk().Generate(prompt, 3, -1)
	if err != nil {
		t.Fatalf("reference Generate: %v", err)
	}
	target := mk()
	if err := target.PrefillTokens(prompt); err != nil {
		t.Fatalf("PrefillTokens: %v", err)
	}
	pair := &AssistantPair{TargetArch: target.arch}

	got, err := pair.VerifyDraftBlockFromSession(target, want[:2])
	if err != nil {
		t.Fatalf("VerifyDraftBlockFromSession: %v", err)
	}

	if !got.AllAccepted || got.AcceptedCount != 2 || got.RejectedCount != 0 {
		t.Fatalf("verify counts allAccepted=%v accepted=%d rejected=%d, want true/2/0", got.AllAccepted, got.AcceptedCount, got.RejectedCount)
	}
	if !idsEqual(got.DraftedTokens, want[:2]) || !idsEqual(got.AcceptedTokens, want[:2]) || len(got.RejectedTokens) != 0 {
		t.Fatalf("verify tokens drafted=%v accepted=%v rejected=%v, want accepted %v", got.DraftedTokens, got.AcceptedTokens, got.RejectedTokens, want[:2])
	}
	if !idsEqual(got.TargetTokens, []int32{want[0]}) {
		t.Fatalf("verify target tokens = %v, want [%d]", got.TargetTokens, want[0])
	}
	if target.Pos() != len(prompt)+2 {
		t.Fatalf("target Pos after verify = %d, want %d", target.Pos(), len(prompt)+2)
	}
	if got.ReplacementToken != 0 {
		t.Fatalf("ReplacementToken = %d, want 0 when all accepted", got.ReplacementToken)
	}
	if len(got.Hidden) != target.arch.Hidden*bf16Size {
		t.Fatalf("Hidden bytes = %d, want %d", len(got.Hidden), target.arch.Hidden*bf16Size)
	}
	if len(got.Logits) != target.arch.Vocab*bf16Size {
		t.Fatalf("Logits bytes = %d, want %d", len(got.Logits), target.arch.Vocab*bf16Size)
	}
	ref := mk()
	if err := ref.PrefillTokens(prompt); err != nil {
		t.Fatalf("reference PrefillTokens: %v", err)
	}
	var wantBoundaryHidden []byte
	for i, id := range want[:2] {
		wantBoundaryHidden, err = ref.stepID(id)
		if err != nil {
			t.Fatalf("reference boundary stepID(%d): %v", i, err)
		}
	}
	wantBoundaryLogits, err := ref.BoundaryLogits()
	if err != nil {
		t.Fatalf("reference boundary logits: %v", err)
	}
	eqBytes(t, "strict greedy boundary hidden", got.Hidden, wantBoundaryHidden)
	eqBytes(t, "strict greedy boundary logits", got.Logits, wantBoundaryLogits)
}

func TestAssistantPairVerifyDraftBlockFromSessionRejectsSuffixAndRestoresAcceptedBoundary(t *testing.T) {
	requireNativeRuntime(t)

	mk := newMTPDecodeFixture(t)
	prompt := []int32{1, 5, 3}
	want, err := mk().Generate(prompt, 4, -1)
	if err != nil {
		t.Fatalf("reference Generate: %v", err)
	}
	badSecond := nativeAssistantWrongToken(want[1])
	target := mk()
	if err := target.PrefillTokens(prompt); err != nil {
		t.Fatalf("PrefillTokens: %v", err)
	}
	pair := &AssistantPair{TargetArch: target.arch}

	got, err := pair.VerifyDraftBlockFromSession(target, []int32{want[0], badSecond})
	if err != nil {
		t.Fatalf("VerifyDraftBlockFromSession: %v", err)
	}

	if got.AllAccepted || got.AcceptedCount != 1 || got.RejectedCount != 1 {
		t.Fatalf("verify counts allAccepted=%v accepted=%d rejected=%d, want false/1/1", got.AllAccepted, got.AcceptedCount, got.RejectedCount)
	}
	if !idsEqual(got.AcceptedTokens, []int32{want[0]}) || !idsEqual(got.RejectedTokens, []int32{badSecond}) {
		t.Fatalf("verify accepted=%v rejected=%v, want [%d]/[%d]", got.AcceptedTokens, got.RejectedTokens, want[0], badSecond)
	}
	if got.ReplacementToken != want[1] {
		t.Fatalf("ReplacementToken = %d, want %d", got.ReplacementToken, want[1])
	}
	if target.Pos() != len(prompt)+1 {
		t.Fatalf("target Pos after verify = %d, want %d", target.Pos(), len(prompt)+1)
	}
	if len(got.Hidden) != target.arch.Hidden*bf16Size {
		t.Fatalf("Hidden bytes = %d, want %d", len(got.Hidden), target.arch.Hidden*bf16Size)
	}
	ref := mk()
	if err := ref.PrefillTokens(prompt); err != nil {
		t.Fatalf("reference PrefillTokens: %v", err)
	}
	wantBoundaryHidden, err := ref.stepID(want[0])
	if err != nil {
		t.Fatalf("reference boundary stepID: %v", err)
	}
	wantBoundaryLogits, err := ref.BoundaryLogits()
	if err != nil {
		t.Fatalf("reference boundary logits: %v", err)
	}
	eqBytes(t, "reforged greedy boundary hidden", got.Hidden, wantBoundaryHidden)
	eqBytes(t, "reforged greedy boundary logits", got.Logits, wantBoundaryLogits)
	continued, err := target.GenerateFromCache(2, -1)
	if err != nil {
		t.Fatalf("GenerateFromCache after verify: %v", err)
	}
	wantContinued, err := mk().Generate(append(append([]int32{}, prompt...), want[0]), 2, -1)
	if err != nil {
		t.Fatalf("reference continuation: %v", err)
	}
	if !idsEqual(continued, wantContinued) {
		t.Fatalf("continuation after rollback = %v, want %v", continued, wantContinued)
	}
}

func TestAssistantPairVerifyDraftBlockFromSessionRejectsFirstTokenAndRestoresPromptBoundary(t *testing.T) {
	requireNativeRuntime(t)

	mk := newMTPDecodeFixture(t)
	prompt := []int32{1, 5, 3}
	want, err := mk().Generate(prompt, 3, -1)
	if err != nil {
		t.Fatalf("reference Generate: %v", err)
	}
	badFirst := nativeAssistantWrongToken(want[0])
	target := mk()
	if err := target.PrefillTokens(prompt); err != nil {
		t.Fatalf("PrefillTokens: %v", err)
	}
	pair := &AssistantPair{TargetArch: target.arch}

	got, err := pair.VerifyDraftBlockFromSession(target, []int32{badFirst})
	if err != nil {
		t.Fatalf("VerifyDraftBlockFromSession: %v", err)
	}

	if got.AllAccepted || got.AcceptedCount != 0 || got.RejectedCount != 1 {
		t.Fatalf("verify counts allAccepted=%v accepted=%d rejected=%d, want false/0/1", got.AllAccepted, got.AcceptedCount, got.RejectedCount)
	}
	if len(got.AcceptedTokens) != 0 || !idsEqual(got.RejectedTokens, []int32{badFirst}) {
		t.Fatalf("verify accepted=%v rejected=%v, want none/[%d]", got.AcceptedTokens, got.RejectedTokens, badFirst)
	}
	if got.ReplacementToken != want[0] {
		t.Fatalf("ReplacementToken = %d, want %d", got.ReplacementToken, want[0])
	}
	if target.Pos() != len(prompt) {
		t.Fatalf("target Pos after verify = %d, want %d", target.Pos(), len(prompt))
	}
	if len(got.Hidden) != 0 {
		t.Fatalf("Hidden bytes = %d, want 0 when no draft token is accepted", len(got.Hidden))
	}
	continued, err := target.GenerateFromCache(2, -1)
	if err != nil {
		t.Fatalf("GenerateFromCache after verify: %v", err)
	}
	wantContinued, err := mk().Generate(prompt, 2, -1)
	if err != nil {
		t.Fatalf("reference continuation: %v", err)
	}
	if !idsEqual(continued, wantContinued) {
		t.Fatalf("continuation after full rollback = %v, want %v", continued, wantContinued)
	}
}

func TestAssistantPairVerifyDraftBlockFromSessionRejectsFirstTokenWithoutDraftForward(t *testing.T) {
	arch := model.Arch{
		Hidden: 4,
		Vocab:  4,
		Layer:  []model.LayerSpec{{Attention: model.SlidingAttention, CacheIndex: 0}},
	}
	target := &ArchSession{
		arch:           arch,
		pos:            3,
		maxLen:         3,
		retainedLogits: toBF16Bytes([]float32{-1, 0, 3, 1}),
	}
	pair := &AssistantPair{TargetArch: target.arch}

	got, err := pair.VerifyDraftBlockFromSession(target, []int32{1})
	if err != nil {
		t.Fatalf("VerifyDraftBlockFromSession: %v", err)
	}

	if got.AcceptedCount != 0 || got.ReplacementToken != 2 {
		t.Fatalf("verify accepted=%d replacement=%d, want 0/2", got.AcceptedCount, got.ReplacementToken)
	}
	if target.Pos() != 3 {
		t.Fatalf("target Pos after first-token reject = %d, want 3", target.Pos())
	}
}

func TestAssistantPairVerifyDraftBlockNoCopyModeAliasesDraftSlices(t *testing.T) {
	arch := model.Arch{
		Hidden: 4,
		Vocab:  4,
		Layer:  []model.LayerSpec{{Attention: model.SlidingAttention, CacheIndex: 0}},
	}
	target := &ArchSession{
		arch:           arch,
		pos:            3,
		maxLen:         3,
		retainedLogits: toBF16Bytes([]float32{-1, 0, 3, 1}),
	}
	pair := &AssistantPair{TargetArch: target.arch}
	draft := []int32{1, 3}

	got, err := pair.verifyDraftBlockFromSession(target, draft, false)
	if err != nil {
		t.Fatalf("verifyDraftBlockFromSession(no-copy): %v", err)
	}

	if got.AcceptedCount != 0 || got.RejectedCount != len(draft) || got.ReplacementToken != 2 {
		t.Fatalf("verify accepted/rejected/replacement = %d/%d/%d, want 0/%d/2",
			got.AcceptedCount, got.RejectedCount, got.ReplacementToken, len(draft))
	}
	if len(got.Logits) != 0 || len(got.Hidden) != 0 {
		t.Fatalf("no-copy verifier returned hidden/logits bytes = %d/%d, want 0/0", len(got.Hidden), len(got.Logits))
	}
	draft[0] = 7
	if got.DraftedTokens[0] != 7 || got.RejectedTokens[0] != 7 {
		t.Fatalf("no-copy verifier did not alias draft slices: drafted=%v rejected=%v", got.DraftedTokens, got.RejectedTokens)
	}
}

func TestAssistantPairVerifyDraftBlockSampledNoCopyModeAliasesDraftSlices(t *testing.T) {
	requireNativeRuntime(t)

	pair, mk := newNativeAssistantGenerateFixture(t)
	defer pair.Close()
	params := model.SampleParams{Temperature: 1.5}
	prompt, seed, sampled, badDraft := nativeAssistantSampledVerifierRejectFixture(t, mk, params)
	target := mk()
	if err := target.PrefillTokens(prompt); err != nil {
		t.Fatalf("PrefillTokens: %v", err)
	}
	draft := []int32{badDraft, nativeAssistantWrongToken(badDraft)}

	got, err := pair.verifyDraftBlockSampledFromSession(target, draft, model.NewSampler(seed), params, false, false, nil)
	if err != nil {
		t.Fatalf("verifyDraftBlockSampledFromSession(no-copy): %v", err)
	}

	if got.AcceptedCount != 0 || got.RejectedCount != len(draft) || got.ReplacementToken != sampled {
		t.Fatalf("sampled no-copy accepted/rejected/replacement = %d/%d/%d, want 0/%d/%d",
			got.AcceptedCount, got.RejectedCount, got.ReplacementToken, len(draft), sampled)
	}
	if len(got.Logits) != 0 || len(got.Hidden) != 0 {
		t.Fatalf("sampled no-copy returned hidden/logits bytes = %d/%d, want 0/0", len(got.Hidden), len(got.Logits))
	}
	draft[0] = 7
	if got.DraftedTokens[0] != 7 || got.RejectedTokens[0] != 7 {
		t.Fatalf("sampled no-copy did not alias draft slices: drafted=%v rejected=%v", got.DraftedTokens, got.RejectedTokens)
	}
}

func TestAssistantPairVerifyDraftBlockSampledFromSessionUsesTargetSampler(t *testing.T) {
	requireNativeRuntime(t)

	pair, mk := newNativeAssistantGenerateFixture(t)
	defer pair.Close()
	params := model.SampleParams{Temperature: 1.5}
	prompt, seed, sampled, badDraft := nativeAssistantSampledVerifierRejectFixture(t, mk, params)
	target := mk()
	if err := target.PrefillTokens(prompt); err != nil {
		t.Fatalf("PrefillTokens: %v", err)
	}

	got, err := pair.VerifyDraftBlockSampledFromSession(target, []int32{badDraft}, model.NewSampler(seed), params, false)
	if err != nil {
		t.Fatalf("VerifyDraftBlockSampledFromSession: %v", err)
	}

	if got.AllAccepted || got.AcceptedCount != 0 || got.RejectedCount != 1 {
		t.Fatalf("sampled verify counts allAccepted=%v accepted=%d rejected=%d, want false/0/1", got.AllAccepted, got.AcceptedCount, got.RejectedCount)
	}
	if got.ReplacementToken != sampled {
		t.Fatalf("sampled replacement = %d, want target sampled token %d", got.ReplacementToken, sampled)
	}
	if !idsEqual(got.TargetTokens, []int32{sampled}) {
		t.Fatalf("sampled target tokens = %v, want [%d]", got.TargetTokens, sampled)
	}
	if target.Pos() != len(prompt) {
		t.Fatalf("target Pos after sampled reject = %d, want %d", target.Pos(), len(prompt))
	}
	if len(got.Hidden) != 0 {
		t.Fatalf("sampled reject hidden bytes = %d, want 0 when no draft token is accepted", len(got.Hidden))
	}
	if len(got.Logits) != target.arch.Vocab*bf16Size {
		t.Fatalf("sampled reject logits bytes = %d, want %d", len(got.Logits), target.arch.Vocab*bf16Size)
	}
}

func TestAssistantPairGenerateFromSessionMatchesTargetGenerate(t *testing.T) {
	requireNativeRuntime(t)

	pair, mk := newNativeAssistantGenerateFixture(t)
	defer pair.Close()
	prompt := []int32{1, 5, 3}
	maxNew := 4
	want, err := mk().Generate(prompt, maxNew, -1)
	if err != nil {
		t.Fatalf("reference Generate: %v", err)
	}
	target := mk()

	got, err := pair.GenerateFromSession(target, prompt, maxNew, -1, 2, nil)
	if err != nil {
		t.Fatalf("GenerateFromSession: %v", err)
	}

	if !idsEqual(got.Tokens, want) {
		t.Fatalf("GenerateFromSession tokens = %v, want %v", got.Tokens, want)
	}
	if target.Pos() != len(prompt)+len(want) {
		t.Fatalf("target Pos after GenerateFromSession = %d, want %d", target.Pos(), len(prompt)+len(want))
	}
	if got.PromptTokens != len(prompt) || got.TargetTokens != len(want) {
		t.Fatalf("generate token counts prompt=%d target=%d, want %d/%d", got.PromptTokens, got.TargetTokens, len(prompt), len(want))
	}
	if got.DraftCalls == 0 || got.TargetVerifyCalls == 0 || got.DraftTokens == 0 {
		t.Fatalf("generate counters draftCalls=%d verifyCalls=%d draftTokens=%d, want non-zero speculative path", got.DraftCalls, got.TargetVerifyCalls, got.DraftTokens)
	}
	for _, n := range got.DraftTokenSchedule {
		if n <= 0 || n > 2 {
			t.Fatalf("draft schedule entry = %d, want 1..2", n)
		}
	}
}

func TestAssistantGenerateResultPreallocatesOutputBuffers(t *testing.T) {
	got := newAssistantGenerateResult(6, 7, 3)

	if got.PromptTokens != 6 {
		t.Fatalf("PromptTokens = %d, want 6", got.PromptTokens)
	}
	if len(got.Tokens) != 0 || cap(got.Tokens) != 7 {
		t.Fatalf("Tokens len/cap = %d/%d, want 0/7", len(got.Tokens), cap(got.Tokens))
	}
	if len(got.DraftTokenSchedule) != 0 || cap(got.DraftTokenSchedule) != 3 {
		t.Fatalf("DraftTokenSchedule len/cap = %d/%d, want 0/3", len(got.DraftTokenSchedule), cap(got.DraftTokenSchedule))
	}
}

func TestAssistantPairGenerateFromSessionFallsBackAfterLowAcceptFullBlock(t *testing.T) {
	requireNativeRuntime(t)

	pair, mk := newNativeAssistantGenerateFixture(t)
	defer pair.Close()
	maxNew := 6
	draftTokens := 2
	prompt := nativeAssistantPromptWhoseTargetTokensAvoid(t, mk, 0, maxNew)
	want, err := mk().Generate(prompt, maxNew, -1)
	if err != nil {
		t.Fatalf("reference Generate: %v", err)
	}
	target := mk()

	got, err := pair.GenerateFromSession(target, prompt, maxNew, -1, draftTokens, nil)
	if err != nil {
		t.Fatalf("GenerateFromSession: %v", err)
	}

	if !idsEqual(got.Tokens, want) {
		t.Fatalf("GenerateFromSession tokens = %v, want %v", got.Tokens, want)
	}
	if target.Pos() != len(prompt)+len(want) {
		t.Fatalf("target Pos after low-accept fallback = %d, want %d", target.Pos(), len(prompt)+len(want))
	}
	if got.AcceptedTokens != 0 {
		t.Fatalf("accepted draft tokens = %d, want 0 for zero-accept fixture", got.AcceptedTokens)
	}
	// A single weak block is transient (any near-tie zeroes one block); the loop keeps
	// drafting and only falls back to plain target decode after nativeAssistantLowAcceptPatience
	// CONSECUTIVE weak blocks. So a persistently-zero-accept fixture drafts exactly that many
	// full blocks before the fallback, not just one.
	if got.DraftCalls != nativeAssistantLowAcceptPatience || got.TargetVerifyCalls != nativeAssistantLowAcceptPatience {
		t.Fatalf("draft/verify calls = %d/%d, want %d weak blocks before target-cache fallback", got.DraftCalls, got.TargetVerifyCalls, nativeAssistantLowAcceptPatience)
	}
	wantDrafted := draftTokens * nativeAssistantLowAcceptPatience
	if got.DraftTokens != wantDrafted || got.RejectedTokens != wantDrafted {
		t.Fatalf("draft/reject tokens = %d/%d, want %d rejected full blocks of %d", got.DraftTokens, got.RejectedTokens, nativeAssistantLowAcceptPatience, draftTokens)
	}
	if got.TargetTokens != len(want) {
		t.Fatalf("target tokens = %d, want %d", got.TargetTokens, len(want))
	}
	if len(got.DraftTokenSchedule) == 0 || got.DraftTokenSchedule[0] != draftTokens {
		t.Fatalf("draft schedule = %v, want first verify block to use requested draft size %d", got.DraftTokenSchedule, draftTokens)
	}
}

func TestAssistantPairGenerateFromSessionUsesFullDraftBlockWithoutProbeRamp(t *testing.T) {
	requireNativeRuntime(t)

	pair, mk := newNativeAssistantGenerateFixture(t)
	defer pair.Close()
	maxNew := 6
	draftTokens := 4
	prompt := nativeAssistantPromptWhoseTargetTokensStartThenAvoid(t, mk, 0, 0, maxNew)
	want, err := mk().Generate(prompt, maxNew, -1)
	if err != nil {
		t.Fatalf("reference Generate: %v", err)
	}
	target := mk()

	got, err := pair.GenerateFromSession(target, prompt, maxNew, -1, draftTokens, nil)
	if err != nil {
		t.Fatalf("GenerateFromSession: %v", err)
	}

	if !idsEqual(got.Tokens, want) {
		t.Fatalf("GenerateFromSession tokens = %v, want %v", got.Tokens, want)
	}
	if got.AcceptedTokens != 1 {
		t.Fatalf("accepted draft tokens = %d, want only the first probe accepted", got.AcceptedTokens)
	}
	if got.RejectedTokens == 0 {
		t.Fatalf("rejected tokens = %d, want the weak blocks' proposals rejected", got.RejectedTokens)
	}
	// The FIRST block must use the full requested draft size straight away (no probe ramp).
	if len(got.DraftTokenSchedule) == 0 || got.DraftTokenSchedule[0] != draftTokens {
		t.Fatalf("draft schedule = %v, want first verify block to use requested draft size %d", got.DraftTokenSchedule, draftTokens)
	}
	// One weak block no longer bails — the loop drafts through nativeAssistantLowAcceptPatience
	// consecutive weak blocks before falling back to plain target decode.
	if got.TargetVerifyCalls != nativeAssistantLowAcceptPatience {
		t.Fatalf("target verify calls = %d, want %d before target-cache fallback", got.TargetVerifyCalls, nativeAssistantLowAcceptPatience)
	}
	if target.Pos() != len(prompt)+len(want) {
		t.Fatalf("target Pos after continued speculative generate = %d, want %d", target.Pos(), len(prompt)+len(want))
	}
}

func TestAssistantCommitReplacementKeepsPlainBoundary(t *testing.T) {
	requireNativeRuntime(t)

	mk := newMTPDecodeFixture(t)
	prompt := []int32{1, 5, 3}
	want, err := mk().Generate(prompt, 4, -1)
	if err != nil {
		t.Fatalf("reference Generate: %v", err)
	}
	target := mk()
	if err := target.PrefillTokens(prompt); err != nil {
		t.Fatalf("PrefillTokens: %v", err)
	}
	pair := &AssistantPair{TargetArch: target.arch}
	wrongSecond := (want[1] + 1) % int32(target.arch.Vocab)
	verify, err := pair.VerifyDraftBlockFromSession(target, []int32{want[0], wrongSecond})
	if err != nil {
		t.Fatalf("VerifyDraftBlockFromSession: %v", err)
	}
	if verify.AcceptedCount != 1 || verify.ReplacementToken != want[1] {
		t.Fatalf("verify accepted=%d replacement=%d, want 1/%d", verify.AcceptedCount, verify.ReplacementToken, want[1])
	}

	if err := target.commitAssistantReplacement(verify.ReplacementToken); err != nil {
		t.Fatalf("commit replacement: %v", err)
	}

	if target.Pos() != len(prompt)+2 {
		t.Fatalf("target Pos after replacement commit = %d, want %d", target.Pos(), len(prompt)+2)
	}
	wantIDs := append(append([]int32{}, prompt...), want[:2]...)
	if !idsEqual(target.cachedIDs, wantIDs) {
		t.Fatalf("cached IDs after replacement commit = %v, want %v", target.cachedIDs, wantIDs)
	}
	ref := mk()
	if err := ref.PrefillTokens(prompt); err != nil {
		t.Fatalf("reference PrefillTokens: %v", err)
	}
	if _, err := ref.stepID(want[0]); err != nil {
		t.Fatalf("reference first stepID: %v", err)
	}
	wantHidden, err := ref.stepID(want[1])
	if err != nil {
		t.Fatalf("reference replacement stepID: %v", err)
	}
	wantLogits, err := ref.BoundaryLogits()
	if err != nil {
		t.Fatalf("reference replacement logits: %v", err)
	}
	eqBytes(t, "replacement commit hidden", target.retainedHidden, wantHidden)
	gotLogits, err := target.BoundaryLogits()
	if err != nil {
		t.Fatalf("replacement boundary logits: %v", err)
	}
	eqBytes(t, "replacement commit logits", gotLogits, wantLogits)
}

func TestAssistantPairVerifyDraftBlockCarriesReplacementIntoNextBlock(t *testing.T) {
	requireNativeRuntime(t)

	mk := newMTPDecodeFixture(t)
	prompt := []int32{1, 5, 3}
	want, err := mk().Generate(prompt, 4, -1)
	if err != nil {
		t.Fatalf("reference Generate: %v", err)
	}
	target := mk()
	if err := target.PrefillTokens(prompt); err != nil {
		t.Fatalf("PrefillTokens: %v", err)
	}
	pair := &AssistantPair{TargetArch: target.arch}
	wrongSecond := (want[1] + 1) % int32(target.arch.Vocab)
	first, err := pair.VerifyDraftBlockFromSession(target, []int32{want[0], wrongSecond})
	if err != nil {
		t.Fatalf("first VerifyDraftBlockFromSession: %v", err)
	}
	if first.AcceptedCount != 1 || first.ReplacementToken != want[1] {
		t.Fatalf("first verify accepted=%d replacement=%d, want 1/%d", first.AcceptedCount, first.ReplacementToken, want[1])
	}
	if target.Pos() != len(prompt)+1 {
		t.Fatalf("target Pos after partial verify = %d, want accepted-prefix boundary %d", target.Pos(), len(prompt)+1)
	}

	carried := []int32{first.ReplacementToken, want[2]}
	second, err := pair.VerifyDraftBlockFromSession(target, carried)
	if err != nil {
		t.Fatalf("carried VerifyDraftBlockFromSession: %v", err)
	}

	if !second.AllAccepted || second.AcceptedCount != len(carried) {
		t.Fatalf("carried verify allAccepted=%v accepted=%d, want true/%d", second.AllAccepted, second.AcceptedCount, len(carried))
	}
	if !idsEqual(second.AcceptedTokens, carried) {
		t.Fatalf("carried accepted tokens = %v, want %v", second.AcceptedTokens, carried)
	}
	if target.Pos() != len(prompt)+3 {
		t.Fatalf("target Pos after carried verify = %d, want %d", target.Pos(), len(prompt)+3)
	}
}

func TestAssistantPairGenerateFromSessionUsesExactWarmPromptCache(t *testing.T) {
	requireNativeRuntime(t)

	pair, mk := newNativeAssistantGenerateFixture(t)
	defer pair.Close()
	prompt := []int32{1, 5, 3}
	maxNew := 4
	want, err := mk().Generate(prompt, maxNew, -1)
	if err != nil {
		t.Fatalf("reference Generate: %v", err)
	}
	target := mk()
	if err := target.WarmPromptCache(prompt); err != nil {
		t.Fatalf("WarmPromptCache: %v", err)
	}
	if hit := target.CachedPrefixLen(prompt); hit != len(prompt) {
		t.Fatalf("warm CachedPrefixLen = %d, want exact prompt hit %d", hit, len(prompt))
	}

	got, err := pair.GenerateFromSession(target, prompt, maxNew, -1, 2, nil)
	if err != nil {
		t.Fatalf("GenerateFromSession after WarmPromptCache: %v", err)
	}

	if !idsEqual(got.Tokens, want) {
		t.Fatalf("GenerateFromSession warm tokens = %v, want %v", got.Tokens, want)
	}
	if hit := target.CachedPrefixLen(prompt); hit != len(prompt) {
		t.Fatalf("CachedPrefixLen after assistant generate = %d, want exact prompt hit %d retained", hit, len(prompt))
	}
}

func TestAssistantPreparePromptExactCacheHitSkipsPagedKVTruncateUnderICB(t *testing.T) {
	requireNativeRuntime(t)

	prompt := []int32{1, 5, 3}
	arch := model.Arch{Hidden: 4, Vocab: 4}
	sess := &ArchSession{
		arch:   arch,
		maxLen: 16,
		state: archDecodeState{
			icb: &archICBReplay{},
			pagedKV: []*devicePagedKVCache{
				{length: 0, maxSize: 16, pageSize: 4, pageLens: make([]int, 4)},
			},
		},
		cachedIDs: append(append([]int32{}, prompt...), 7, 8),
	}
	hidden := toBF16Bytes(syntheticFloat32(arch.Hidden, 307))
	logits := toBF16Bytes(syntheticFloat32(arch.Vocab, 311))
	sess.rememberCachedPromptEntry(prompt, hidden, logits)

	if err := sess.prepareAssistantPrompt(prompt); err != nil {
		t.Fatalf("prepareAssistantPrompt exact cache hit: %v", err)
	}

	if sess.Pos() != len(prompt) {
		t.Fatalf("prepared Pos = %d, want %d", sess.Pos(), len(prompt))
	}
	if !idsEqual(sess.cachedIDs, prompt) {
		t.Fatalf("prepared cached IDs = %v, want %v", sess.cachedIDs, prompt)
	}
	eqBytes(t, "prepared retained hidden", sess.retainedHidden, hidden)
	eqBytes(t, "prepared retained logits", sess.retainedLogits, logits)
}

func TestAssistantPairGenerateFromSessionUsesWarmPromptPrefix(t *testing.T) {
	requireNativeRuntime(t)

	pair, mk := newNativeAssistantGenerateFixture(t)
	defer pair.Close()
	shared := []int32{1, 5}
	prompt := []int32{1, 5, 3}
	maxNew := 4
	want, err := mk().Generate(prompt, maxNew, -1)
	if err != nil {
		t.Fatalf("reference Generate: %v", err)
	}
	target := mk()
	if err := target.WarmPromptCache(shared); err != nil {
		t.Fatalf("WarmPromptCache(shared): %v", err)
	}
	if hit := target.CachedPrefixLen(prompt); hit != len(shared) {
		t.Fatalf("warm CachedPrefixLen(full prompt) = %d, want shared prefix hit %d", hit, len(shared))
	}

	got, err := pair.GenerateFromSession(target, prompt, maxNew, -1, 2, nil)
	if err != nil {
		t.Fatalf("GenerateFromSession after shared WarmPromptCache: %v", err)
	}

	if !idsEqual(got.Tokens, want) {
		t.Fatalf("GenerateFromSession shared-prefix tokens = %v, want %v", got.Tokens, want)
	}
	if hit := target.CachedPrefixLen(prompt); hit != len(prompt) {
		t.Fatalf("CachedPrefixLen after shared-prefix assistant generate = %d, want exact prompt hit %d retained", hit, len(prompt))
	}
}

func TestAssistantPairGenerateFromSessionStopsWhenYieldReturnsFalse(t *testing.T) {
	requireNativeRuntime(t)

	pair, mk := newNativeAssistantGenerateFixture(t)
	defer pair.Close()
	prompt := nativeAssistantPromptWhoseFirstTargetTokenIsNot(t, mk, 0)
	target := mk()
	var yielded []int32

	got, err := pair.GenerateFromSessionEach(target, prompt, 4, -1, 2, nil, func(id int32) bool {
		yielded = append(yielded, id)
		return false
	})
	if err != nil {
		t.Fatalf("GenerateFromSessionEach: %v", err)
	}

	if len(got.Tokens) != 1 || len(yielded) != 1 || got.Tokens[0] != yielded[0] {
		t.Fatalf("yield stop tokens got=%v yielded=%v, want one matching token", got.Tokens, yielded)
	}
	if target.Pos() != len(prompt) {
		t.Fatalf("target Pos after replacement yield stop = %d, want unforwarded carry position %d", target.Pos(), len(prompt))
	}
}

func TestAssistantPairGenerateFromSessionEachFallsBackAfterLowAcceptFullBlock(t *testing.T) {
	requireNativeRuntime(t)

	pair, mk := newNativeAssistantGenerateFixture(t)
	defer pair.Close()
	maxNew := 6
	draftTokens := 2
	prompt := nativeAssistantPromptWhoseTargetTokensAvoid(t, mk, 0, maxNew)
	want, err := mk().Generate(prompt, maxNew, -1)
	if err != nil {
		t.Fatalf("reference Generate: %v", err)
	}
	target := mk()
	var yielded []int32

	got, err := pair.GenerateFromSessionEach(target, prompt, maxNew, -1, draftTokens, nil, func(id int32) bool {
		yielded = append(yielded, id)
		return true
	})
	if err != nil {
		t.Fatalf("GenerateFromSessionEach: %v", err)
	}

	if !idsEqual(got.Tokens, want) || !idsEqual(yielded, want) {
		t.Fatalf("stream fallback tokens got=%v yielded=%v, want %v", got.Tokens, yielded, want)
	}
	// Fallback only after nativeAssistantLowAcceptPatience consecutive weak blocks, not one.
	if got.DraftCalls != nativeAssistantLowAcceptPatience || got.TargetVerifyCalls != nativeAssistantLowAcceptPatience {
		t.Fatalf("stream draft/verify calls = %d/%d, want %d weak blocks before target-cache fallback", got.DraftCalls, got.TargetVerifyCalls, nativeAssistantLowAcceptPatience)
	}
	wantDrafted := draftTokens * nativeAssistantLowAcceptPatience
	if got.DraftTokens != wantDrafted || got.RejectedTokens != wantDrafted {
		t.Fatalf("stream draft/reject tokens = %d/%d, want %d rejected full blocks of %d", got.DraftTokens, got.RejectedTokens, nativeAssistantLowAcceptPatience, draftTokens)
	}
	if target.Pos() != len(prompt)+len(want) {
		t.Fatalf("target Pos after stream low-accept fallback = %d, want %d", target.Pos(), len(prompt)+len(want))
	}
}

func TestAssistantPairGenerateFromSessionCountsAcceptedYieldStop(t *testing.T) {
	requireNativeRuntime(t)

	pair, mk := newNativeAssistantGenerateFixture(t)
	defer pair.Close()
	prompt := nativeAssistantPromptWithAcceptedFirstDraft(t, pair, mk)
	target := mk()
	var yielded []int32

	got, err := pair.GenerateFromSessionEach(target, prompt, 4, -1, 2, nil, func(id int32) bool {
		yielded = append(yielded, id)
		return false
	})
	if err != nil {
		t.Fatalf("GenerateFromSessionEach: %v", err)
	}

	if len(got.Tokens) != 1 || len(yielded) != 1 || got.Tokens[0] != yielded[0] {
		t.Fatalf("accepted yield stop tokens got=%v yielded=%v, want one matching token", got.Tokens, yielded)
	}
	if got.AcceptedTokens != 1 || got.TargetTokens != 1 {
		t.Fatalf("accepted yield stop counts accepted=%d target=%d, want 1/1", got.AcceptedTokens, got.TargetTokens)
	}
	if target.Pos() != len(prompt)+1 {
		t.Fatalf("target Pos after accepted yield stop = %d, want %d", target.Pos(), len(prompt)+1)
	}
}

func TestAssistantPairGenerateSampledFromSessionMatchesTargetGenerateSampled(t *testing.T) {
	requireNativeRuntime(t)

	pair, mk := newNativeAssistantGenerateFixture(t)
	defer pair.Close()
	params := model.SampleParams{Temperature: 1.5}
	prompt, seed, _, _ := nativeAssistantSampledVerifierRejectFixture(t, mk, params)
	maxNew := 4
	want, err := mk().GenerateSampledEach(prompt, maxNew, nil, model.NewSampler(seed), params, nil, nil)
	if err != nil {
		t.Fatalf("reference GenerateSampledEach: %v", err)
	}
	target := mk()

	got, err := pair.GenerateSampledFromSession(target, prompt, maxNew, nil, model.NewSampler(seed), params, 2)
	if err != nil {
		t.Fatalf("GenerateSampledFromSession: %v", err)
	}

	if !idsEqual(got.Tokens, want) {
		t.Fatalf("GenerateSampledFromSession tokens = %v, want %v", got.Tokens, want)
	}
	if target.Pos() != len(prompt)+len(want) {
		t.Fatalf("target Pos after GenerateSampledFromSession = %d, want %d", target.Pos(), len(prompt)+len(want))
	}
	if got.DraftCalls == 0 || got.TargetVerifyCalls == 0 || got.DraftTokens == 0 {
		t.Fatalf("sampled counters draftCalls=%d verifyCalls=%d draftTokens=%d, want non-zero speculative path", got.DraftCalls, got.TargetVerifyCalls, got.DraftTokens)
	}
}

func TestAssistantPairGenerateSampledFromSessionRepeatPenaltyMatchesTarget(t *testing.T) {
	requireNativeRuntime(t)

	pair, mk := newNativeAssistantGenerateFixture(t)
	defer pair.Close()
	params := model.SampleParams{
		Temperature:   1.2,
		TopK:          4,
		TopP:          0.9,
		RepeatPenalty: 1.4,
	}
	prompt := []int32{1, 5, 3, 2, 6}
	const maxNew = 6
	for seed := uint64(1); seed <= 32; seed++ {
		want, err := mk().GenerateSampledEach(prompt, maxNew, nil, model.NewSampler(seed), params, nil, nil)
		if err != nil {
			t.Fatalf("reference GenerateSampledEach(seed=%d): %v", seed, err)
		}
		target := mk()
		got, err := pair.GenerateSampledFromSession(target, prompt, maxNew, nil, model.NewSampler(seed), params, 2)
		if err != nil {
			t.Fatalf("GenerateSampledFromSession(seed=%d): %v", seed, err)
		}
		if !idsEqual(got.Tokens, want) {
			t.Fatalf("GenerateSampledFromSession(seed=%d) tokens = %v, want %v", seed, got.Tokens, want)
		}
	}
}

func TestAssistantPairGenerateSampledFromSessionEachKeepsDraftBlockWhileStreaming(t *testing.T) {
	requireNativeRuntime(t)

	pair, mk := newNativeAssistantGenerateFixture(t)
	defer pair.Close()
	params := model.SampleParams{Temperature: 1.5}
	prompt, seed, _, _ := nativeAssistantSampledVerifierRejectFixture(t, mk, params)
	maxNew := 4
	want, err := mk().GenerateSampledEach(prompt, maxNew, nil, model.NewSampler(seed), params, nil, nil)
	if err != nil {
		t.Fatalf("reference GenerateSampledEach: %v", err)
	}
	target := mk()
	var yielded []int32

	got, err := pair.GenerateSampledFromSessionEach(target, prompt, maxNew, nil, model.NewSampler(seed), params, 2, func(id int32) bool {
		yielded = append(yielded, id)
		return true
	})
	if err != nil {
		t.Fatalf("GenerateSampledFromSessionEach: %v", err)
	}

	if !idsEqual(got.Tokens, want) {
		t.Fatalf("streaming sampled assistant tokens = %v, want %v", got.Tokens, want)
	}
	if !idsEqual(yielded, got.Tokens) {
		t.Fatalf("streaming sampled assistant yielded %v, want result tokens %v", yielded, got.Tokens)
	}
	hasBlock := false
	for _, n := range got.DraftTokenSchedule {
		if n > 1 {
			hasBlock = true
			break
		}
	}
	if !hasBlock {
		t.Fatalf("streaming sampled assistant draft schedule = %v, want a multi-token verify block", got.DraftTokenSchedule)
	}
}

func TestAssistantPairGenerateSampledFromSessionFallsBackAfterLowAcceptFullBlock(t *testing.T) {
	requireNativeRuntime(t)

	pair, mk := newNativeAssistantGenerateFixture(t)
	defer pair.Close()
	params := model.SampleParams{Temperature: 1.5}
	prompt, seed, _ := nativeAssistantSampledPromptWithRejectedFirstDraft(t, pair, mk, params)
	const maxNew = 6
	const draftTokens = 2
	want, err := mk().GenerateSampledEach(prompt, maxNew, nil, model.NewSampler(seed), params, nil, nil)
	if err != nil {
		t.Fatalf("reference GenerateSampledEach: %v", err)
	}
	target := mk()

	got, err := pair.GenerateSampledFromSession(target, prompt, maxNew, nil, model.NewSampler(seed), params, draftTokens)
	if err != nil {
		t.Fatalf("GenerateSampledFromSession: %v", err)
	}

	if !idsEqual(got.Tokens, want) {
		t.Fatalf("sampled low-accept fallback tokens = %v, want %v", got.Tokens, want)
	}
	if got.AcceptedTokens != 0 {
		t.Fatalf("sampled accepted draft tokens = %d, want 0 for rejected first block", got.AcceptedTokens)
	}
	// Fallback only after nativeAssistantLowAcceptPatience consecutive weak blocks, not one.
	if got.DraftCalls != nativeAssistantLowAcceptPatience || got.TargetVerifyCalls != nativeAssistantLowAcceptPatience {
		t.Fatalf("sampled draft/verify calls = %d/%d, want %d weak blocks before target-cache fallback", got.DraftCalls, got.TargetVerifyCalls, nativeAssistantLowAcceptPatience)
	}
	wantDrafted := draftTokens * nativeAssistantLowAcceptPatience
	if got.DraftTokens != wantDrafted || got.RejectedTokens != wantDrafted {
		t.Fatalf("sampled draft/reject tokens = %d/%d, want %d rejected full blocks of %d", got.DraftTokens, got.RejectedTokens, nativeAssistantLowAcceptPatience, draftTokens)
	}
	if got.TargetTokens != len(want) {
		t.Fatalf("sampled target tokens = %d, want %d", got.TargetTokens, len(want))
	}
	if len(got.DraftTokenSchedule) == 0 || got.DraftTokenSchedule[0] != draftTokens {
		t.Fatalf("sampled draft schedule = %v, want first verify block to use requested draft size %d", got.DraftTokenSchedule, draftTokens)
	}
	if target.Pos() != len(prompt)+len(want) {
		t.Fatalf("target Pos after sampled low-accept fallback = %d, want %d", target.Pos(), len(prompt)+len(want))
	}
}

func TestAssistantPairGenerateSampledFromSessionCountsAcceptedYieldStop(t *testing.T) {
	requireNativeRuntime(t)

	pair, mk := newNativeAssistantGenerateFixture(t)
	defer pair.Close()
	params := model.SampleParams{Temperature: 1.5, SuppressTokens: []int32{0, 1, 2, 3, 4, 5, 6}}
	prompt := nativeAssistantWordedPromptIDs()
	const seed = uint64(1)
	target := mk()
	var yielded []int32

	got, err := pair.GenerateSampledFromSessionEach(target, prompt, 4, nil, model.NewSampler(seed), params, 2, func(id int32) bool {
		yielded = append(yielded, id)
		return false
	})
	if err != nil {
		t.Fatalf("GenerateSampledFromSessionEach: %v", err)
	}

	if len(got.Tokens) != 1 || len(yielded) != 1 || got.Tokens[0] != yielded[0] {
		t.Fatalf("sampled accepted yield stop tokens got=%v yielded=%v, want one matching token", got.Tokens, yielded)
	}
	if got.AcceptedTokens != 1 || got.TargetTokens != 1 {
		t.Fatalf("sampled accepted yield stop counts accepted=%d target=%d, want 1/1", got.AcceptedTokens, got.TargetTokens)
	}
	if target.Pos() != len(prompt)+1 {
		t.Fatalf("target Pos after sampled accepted yield stop = %d, want %d", target.Pos(), len(prompt)+1)
	}
}

func TestAssistantPairGenerateSampledFromSessionCommitsReplacementStop(t *testing.T) {
	requireNativeRuntime(t)

	pair, mk := newNativeAssistantGenerateFixture(t)
	defer pair.Close()
	params := model.SampleParams{Temperature: 1.5}
	prompt, seed, stopToken := nativeAssistantSampledPromptWithRejectedFirstDraft(t, pair, mk, params)
	stopTokens := []int32{stopToken}
	const maxNew = 4
	ref := mk()
	want, err := ref.GenerateSampledEach(prompt, maxNew, stopTokens, model.NewSampler(seed), params, nil, nil)
	if err != nil {
		t.Fatalf("reference GenerateSampledEach: %v", err)
	}
	if !idsEqual(want, []int32{stopToken}) {
		t.Fatalf("reference sampled stop tokens = %v, want [%d]", want, stopToken)
	}
	if ref.Pos() != len(prompt)+len(want) {
		t.Fatalf("reference Pos after sampled stop = %d, want %d", ref.Pos(), len(prompt)+len(want))
	}

	target := mk()
	got, err := pair.GenerateSampledFromSession(target, prompt, maxNew, stopTokens, model.NewSampler(seed), params, 2)
	if err != nil {
		t.Fatalf("GenerateSampledFromSession: %v", err)
	}
	if !idsEqual(got.Tokens, want) {
		t.Fatalf("sampled assistant replacement stop tokens = %v, want %v", got.Tokens, want)
	}
	if target.Pos() != ref.Pos() {
		t.Fatalf("target Pos after sampled replacement stop = %d, want reference %d", target.Pos(), ref.Pos())
	}
}

func TestAssistantDraftInputProjectionRejectsBadHidden(t *testing.T) {
	tensors := nativeAssistantTinyTensors(true)
	dir := writeNativeAssistantDir(t, tensors)

	assistant, err := LoadAssistantDir(dir)
	if err != nil {
		t.Fatalf("LoadAssistantDir: %v", err)
	}
	defer assistant.Close()

	_, err = assistant.DraftInputProjection(make([]byte, 8*bf16Size), make([]byte, 7*bf16Size))
	if err == nil {
		t.Fatal("DraftInputProjection error = nil, want previous hidden length error")
	}
	if !core.Contains(err.Error(), "previous hidden") {
		t.Fatalf("DraftInputProjection error = %v, want previous hidden", err)
	}
}

func TestAssistantDraftOutputProjectionRejectsBadHidden(t *testing.T) {
	tensors := nativeAssistantTinyTensors(true)
	dir := writeNativeAssistantDir(t, tensors)

	assistant, err := LoadAssistantDir(dir)
	if err != nil {
		t.Fatalf("LoadAssistantDir: %v", err)
	}
	defer assistant.Close()

	_, err = assistant.DraftOutputProjection(make([]byte, 3*bf16Size))
	if err == nil {
		t.Fatal("DraftOutputProjection error = nil, want assistant hidden length error")
	}
	if !core.Contains(err.Error(), "assistant hidden") {
		t.Fatalf("DraftOutputProjection error = %v, want assistant hidden", err)
	}
}

func TestAssistantDraftLogitsDenseMatchesReference(t *testing.T) {
	requireNativeRuntime(t)

	tensors := nativeAssistantTinyTensors(false)
	embedW := nativeAssistantProjectionFixture(8, 4)
	tensors["model.embed_tokens.weight"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{8, 4}, Data: toBF16Bytes(embedW)}
	dir := writeNativeAssistantDirWithOrdered(t, tensors, false)

	assistant, err := LoadAssistantDir(dir)
	if err != nil {
		t.Fatalf("LoadAssistantDir: %v", err)
	}
	defer assistant.Close()

	hidden := toBF16Bytes([]float32{1, -0.5, 0.25, 2})
	got, err := assistant.DraftLogits(hidden)
	if err != nil {
		t.Fatalf("DraftLogits dense: %v", err)
	}

	want := nativeAssistantMatMulBF16NTReference(hidden, toBF16Bytes(embedW), 1, 4, 8)
	assertFloat32Near(t, "dense draft logits", bf16Floats(got), want, 0.02)
}

func TestAssistantDraftLogitsOrderedMasksNonCandidates(t *testing.T) {
	tensors := nativeAssistantTinyTensors(true)
	embedW := []float32{
		1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0,
		0, 0, 0, 1,
		-1, 0, 0, 0,
		0, -1, 0, 0,
		0, 0, -1, 0,
		0, 0, 0, -1,
	}
	centroids := []float32{
		1, 0, 0, 0,
		-1, 0, 0, 0,
	}
	tensors["model.embed_tokens.weight"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{8, 4}, Data: toBF16Bytes(embedW)}
	tensors["masked_embedding.centroids.weight"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{2, 4}, Data: toBF16Bytes(centroids)}
	tensors["masked_embedding.token_ordering"] = safetensors.Tensor{Dtype: "I64", Shape: []int{2, 4}, Data: nativeAssistantI64Tensor(0, 1, 2, 3, 4, 5, 6, 7)}
	dir := writeNativeAssistantDir(t, tensors)

	assistant, err := LoadAssistantDir(dir)
	if err != nil {
		t.Fatalf("LoadAssistantDir: %v", err)
	}
	defer assistant.Close()

	hidden := toBF16Bytes([]float32{1, 0.5, -0.25, 2})
	got, err := assistant.DraftLogits(hidden)
	if err != nil {
		t.Fatalf("DraftLogits ordered: %v", err)
	}

	floor := nativeAssistantBF16Float(nativeAssistantLogitsFloorForTest)
	want := []float32{1, 0.5, -0.25, 2, floor, floor, floor, floor}
	assertFloat32Near(t, "ordered draft logits", bf16Floats(got), want, 0.02)
}

func TestAssistantDraftLogitsOrderedReusesScratch(t *testing.T) {
	tensors := nativeAssistantTinyTensors(true)
	tensors["model.embed_tokens.weight"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{8, 4}, Data: toBF16Bytes(syntheticFloat32(8*4, 313))}
	tensors["masked_embedding.centroids.weight"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{2, 4}, Data: toBF16Bytes(syntheticFloat32(2*4, 317))}
	tensors["masked_embedding.token_ordering"] = safetensors.Tensor{Dtype: "I64", Shape: []int{2, 4}, Data: nativeAssistantI64Tensor(0, 1, 2, 3, 4, 5, 6, 7)}
	dir := writeNativeAssistantDir(t, tensors)

	assistant, err := LoadAssistantDir(dir)
	if err != nil {
		t.Fatalf("LoadAssistantDir: %v", err)
	}
	defer assistant.Close()

	hidden := toBF16Bytes([]float32{1, 0.5, -0.25, 2})
	out := make([]byte, assistant.Arch.Vocab*bf16Size)
	scores := make([]float32, assistant.NumCentroids)
	selected := make([]int, assistant.CentroidIntermediateTopK)
	scorePtr := &scores[0]
	selectedPtr := &selected[0]

	for i := range scores {
		scores[i] = -123
	}
	for i := range selected {
		selected[i] = -1
	}
	if _, err := assistant.draftLogitsIntoScratch(out, hidden, scores, selected); err != nil {
		t.Fatalf("draftLogitsIntoScratch: %v", err)
	}
	if &scores[0] != scorePtr || scores[0] == -123 {
		t.Fatal("ordered logits did not reuse score scratch")
	}
	if &selected[0] != selectedPtr || selected[0] == -1 {
		t.Fatal("ordered logits did not reuse selected-index scratch")
	}
}

func TestAssistantDraftGreedyTokenSelectsArgmax(t *testing.T) {
	tensors := nativeAssistantTinyTensors(false)
	dir := writeNativeAssistantDirWithOrdered(t, tensors, false)

	assistant, err := LoadAssistantDir(dir)
	if err != nil {
		t.Fatalf("LoadAssistantDir: %v", err)
	}
	defer assistant.Close()

	got, err := assistant.DraftGreedyToken(toBF16Bytes([]float32{-1, 0.5, 3, 2.75, -0.25, 1, 0, 2}))
	if err != nil {
		t.Fatalf("DraftGreedyToken: %v", err)
	}
	if got != 2 {
		t.Fatalf("DraftGreedyToken = %d, want 2", got)
	}
}

func TestAssistantDraftGreedyTokenSuppressesIDs(t *testing.T) {
	tensors := nativeAssistantTinyTensors(false)
	dir := writeNativeAssistantDirWithOrdered(t, tensors, false)

	assistant, err := LoadAssistantDir(dir)
	if err != nil {
		t.Fatalf("LoadAssistantDir: %v", err)
	}
	defer assistant.Close()

	got, err := assistant.DraftGreedyToken(toBF16Bytes([]float32{-1, 0.5, 3, 2.75, -0.25, 1, 0, 2}), []int32{2, -1})
	if err != nil {
		t.Fatalf("DraftGreedyToken suppressed: %v", err)
	}
	if got != 3 {
		t.Fatalf("DraftGreedyToken suppressed = %d, want 3", got)
	}
}

func nativeAssistantProjectionFixture(out, in int) []float32 {
	weights := make([]float32, out*in)
	palette := []float32{-0.5, -0.25, 0, 0.25, 0.5}
	for o := range out {
		for k := range in {
			weights[o*in+k] = palette[(o*3+k*2)%len(palette)]
		}
	}
	return weights
}

func nativeAssistantMatMulBF16NTReference(a, w []byte, m, k, n int) []float32 {
	af, wf := bf16Floats(a), bf16Floats(w)
	out := make([]float32, m*n)
	for row := range m {
		for col := range n {
			var sum float32
			for inner := range k {
				sum += af[row*k+inner] * wf[col*k+inner]
			}
			h := f32ToBF16(sum)
			out[row*n+col] = bf16ToF32(byte(h), byte(h>>8))
		}
	}
	return out
}

const nativeAssistantLogitsFloorForTest = -3.4028234663852886e38

func nativeAssistantBF16Float(v float32) float32 {
	h := f32ToBF16(v)
	return bf16ToF32(byte(h), byte(h>>8))
}

func nativeAssistantI64Tensor(values ...int64) []byte {
	out := make([]byte, len(values)*8)
	for i, v := range values {
		binary.LittleEndian.PutUint64(out[i*8:], uint64(v))
	}
	return out
}

func nativeAssistantWrongToken(want int32) int32 {
	return (want + 1) % int32(mtpFixtureVocab)
}

func nativeAssistantQuantEmbeddingFixture(vocab, dModel, groupSize int) ([]byte, []byte, []byte) {
	packed := make([]byte, vocab*dModel/2)
	for row := range vocab {
		for col := 0; col < dModel; col += 2 {
			lo := byte((row + col) & 0x0F)
			hi := byte((row + col + 1) & 0x0F)
			packed[row*dModel/2+col/2] = lo | hi<<4
		}
	}
	groups := dModel / groupSize
	scales := make([]float32, vocab*groups)
	biases := make([]float32, vocab*groups)
	for i := range scales {
		scales[i] = 0.25
		biases[i] = -1
	}
	return packed, toBF16Bytes(scales), toBF16Bytes(biases)
}

func nativeAssistantTinyLoaded(t *testing.T, ordered bool) *AssistantModel {
	t.Helper()
	tensors := nativeAssistantTinyTensors(ordered)
	dir := writeNativeAssistantDirWithOrdered(t, tensors, ordered)
	assistant, err := LoadAssistantDir(dir)
	if err != nil {
		t.Fatalf("LoadAssistantDir: %v", err)
	}
	return assistant
}

func nativeAssistantTargetKVFixture(seed byte) AssistantTargetKV {
	return AssistantTargetKV{
		Key:    []byte{seed, seed + 1, seed + 2, seed + 3},
		Value:  []byte{seed + 4, seed + 5, seed + 6, seed + 7},
		Offset: 1,
		Length: 2,
	}
}

func writeNativeAssistantDir(t *testing.T, tensors map[string]safetensors.Tensor) string {
	return writeNativeAssistantDirWithOrdered(t, tensors, true)
}

func writeNativeAssistantDirWithOrdered(t *testing.T, tensors map[string]safetensors.Tensor, ordered bool) string {
	return writeNativeAssistantDirWithModelType(t, tensors, ordered, "gemma4_assistant")
}

func writeNativeAssistantDirWithModelType(t *testing.T, tensors map[string]safetensors.Tensor, ordered bool, modelType string) string {
	t.Helper()
	dir := t.TempDir()
	cfg := []byte(core.Sprintf(`{
		"model_type": %q,
		"backbone_hidden_size": 8,
		"num_centroids": 2,
		"centroid_intermediate_top_k": 1,
		"use_ordered_embeddings": %v,
		"text_config": {
			"model_type": "gemma4_assistant",
			"hidden_size": 4,
			"num_hidden_layers": 2,
			"intermediate_size": 8,
			"num_attention_heads": 2,
			"num_key_value_heads": 2,
			"head_dim": 2,
			"vocab_size": 8,
			"rms_norm_eps": 0.000001,
			"max_position_embeddings": 16,
			"layer_types": ["sliding_attention", "full_attention"],
			"rope_parameters": {
				"sliding_attention": {"rope_theta": 10000, "partial_rotary_factor": 1.0},
				"full_attention": {"rope_theta": 1000000, "partial_rotary_factor": 1.0}
			}
		}
	}`, modelType, ordered))
	if err := coreio.Local.Write(core.PathJoin(dir, "config.json"), string(cfg)); err != nil {
		t.Fatalf("write config.json: %v", err)
	}
	writeNativeAssistantTokenizer(t, dir)
	blob, err := safetensors.Encode(tensors)
	if err != nil {
		t.Fatalf("Encode assistant tensors: %v", err)
	}
	if err := coreio.Local.Write(core.PathJoin(dir, "model.safetensors"), string(blob)); err != nil {
		t.Fatalf("write model.safetensors: %v", err)
	}
	return dir
}

func writeNativeAssistantFlatDir(t *testing.T, tensors map[string]safetensors.Tensor, ordered bool) string {
	t.Helper()
	dir := t.TempDir()
	cfg := []byte(core.Sprintf(`{
		"model_type": "gemma4_assistant",
		"backbone_hidden_size": 8,
		"num_centroids": 2,
		"centroid_intermediate_top_k": 1,
		"use_ordered_embeddings": %v,
		"hidden_size": 4,
		"num_hidden_layers": 2,
		"intermediate_size": 8,
		"num_attention_heads": 2,
		"num_key_value_heads": 2,
		"head_dim": 2,
		"vocab_size": 8,
		"rms_norm_eps": 0.000001,
		"max_position_embeddings": 16,
		"layer_types": ["sliding_attention", "full_attention"],
		"rope_parameters": {
			"sliding_attention": {"rope_theta": 10000, "partial_rotary_factor": 1.0},
			"full_attention": {"rope_theta": 1000000, "partial_rotary_factor": 1.0}
		}
	}`, ordered))
	if err := coreio.Local.Write(core.PathJoin(dir, "config.json"), string(cfg)); err != nil {
		t.Fatalf("write config.json: %v", err)
	}
	writeNativeAssistantTokenizer(t, dir)
	blob, err := safetensors.Encode(tensors)
	if err != nil {
		t.Fatalf("Encode assistant tensors: %v", err)
	}
	if err := coreio.Local.Write(core.PathJoin(dir, "model.safetensors"), string(blob)); err != nil {
		t.Fatalf("write model.safetensors: %v", err)
	}
	return dir
}

func writeNativeAssistantTokenizer(t testing.TB, dir string) {
	t.Helper()
	const body = `{
  "model": {
    "type": "BPE",
    "vocab": {"h": 1, "e": 2, "l": 3, "o": 4},
    "merges": []
  },
  "added_tokens": [
    {"id": 0, "content": "<bos>", "special": true},
    {"id": 5, "content": "<eos>", "special": true}
  ]
}`
	if err := coreio.Local.Write(core.PathJoin(dir, "tokenizer.json"), body); err != nil {
		t.Fatalf("write tokenizer.json: %v", err)
	}
}

func writeNativeAssistantTargetDir(t *testing.T, hidden int, layerTypes []string) string {
	t.Helper()
	dir := t.TempDir()
	layerTypesJSON := core.JSONMarshal(layerTypes)
	if !layerTypesJSON.OK {
		t.Fatalf("marshal layer types: %s", layerTypesJSON.Error())
	}
	cfg := []byte(core.Sprintf(`{
		"model_type": "gemma4_text",
		"hidden_size": %d,
		"num_hidden_layers": %d,
		"intermediate_size": 16,
		"num_attention_heads": 2,
		"num_key_value_heads": 2,
		"head_dim": 2,
		"vocab_size": 8,
		"rms_norm_eps": 0.000001,
		"sliding_window": 16,
		"max_position_embeddings": 16,
		"layer_types": %s,
		"rope_parameters": {
			"sliding_attention": {"rope_theta": 10000, "partial_rotary_factor": 1.0},
			"full_attention": {"rope_theta": 1000000, "partial_rotary_factor": 1.0}
		}
	}`, hidden, len(layerTypes), string(layerTypesJSON.Value.([]byte))))
	if err := coreio.Local.Write(core.PathJoin(dir, "config.json"), string(cfg)); err != nil {
		t.Fatalf("write target config.json: %v", err)
	}
	return dir
}

func nativeAssistantTinyTensors(includeOrdered bool) map[string]safetensors.Tensor {
	tensors := map[string]safetensors.Tensor{}
	bf := func(name string, shape ...int) {
		elems := 1
		for _, dim := range shape {
			elems *= dim
		}
		tensors[name] = safetensors.Tensor{Dtype: "BF16", Shape: shape, Data: make([]byte, elems*2)}
	}
	bf("model.embed_tokens.weight", 8, 4)
	bf("model.norm.weight", 4)
	bf("pre_projection.weight", 4, 16)
	bf("post_projection.weight", 8, 4)
	if includeOrdered {
		bf("masked_embedding.centroids.weight", 2, 4)
		tensors["masked_embedding.token_ordering"] = safetensors.Tensor{Dtype: "I64", Shape: []int{8}, Data: make([]byte, 8*8)}
	}
	for i := range 2 {
		p := core.Sprintf("model.layers.%d", i)
		bf(p+".input_layernorm.weight", 4)
		bf(p+".post_attention_layernorm.weight", 4)
		bf(p+".pre_feedforward_layernorm.weight", 4)
		bf(p+".post_feedforward_layernorm.weight", 4)
		bf(p+".layer_scalar", 4)
		bf(p+".self_attn.q_proj.weight", 4, 4)
		bf(p+".self_attn.o_proj.weight", 4, 4)
		bf(p+".self_attn.q_norm.weight", 2)
		bf(p+".mlp.gate_proj.weight", 8, 4)
		bf(p+".mlp.up_proj.weight", 8, 4)
		bf(p+".mlp.down_proj.weight", 4, 8)
	}
	return tensors
}

func nativeAssistantAttentionTensors() map[string]safetensors.Tensor {
	return nativeAssistantAttentionTensorsForBackbone(8)
}

func nativeAssistantAttentionTensorsForBackbone(backbone int) map[string]safetensors.Tensor {
	const hidden, headDim, nHeads, intermediate, vocab = 128, 64, 2, 256, 8
	tensors := map[string]safetensors.Tensor{}
	bf := func(name string, shape ...int) {
		elems := 1
		for _, dim := range shape {
			elems *= dim
		}
		tensors[name] = safetensors.Tensor{Dtype: "BF16", Shape: shape, Data: make([]byte, elems*bf16Size)}
	}
	bf("model.embed_tokens.weight", vocab, hidden)
	bf("model.norm.weight", hidden)
	bf("pre_projection.weight", hidden, backbone*2)
	bf("post_projection.weight", backbone, hidden)
	p := "model.layers.0"
	bf(p+".input_layernorm.weight", hidden)
	bf(p+".post_attention_layernorm.weight", hidden)
	bf(p+".pre_feedforward_layernorm.weight", hidden)
	bf(p+".post_feedforward_layernorm.weight", hidden)
	bf(p+".layer_scalar", hidden)
	bf(p+".self_attn.q_proj.weight", nHeads*headDim, hidden)
	bf(p+".self_attn.o_proj.weight", hidden, nHeads*headDim)
	bf(p+".self_attn.q_norm.weight", headDim)
	bf(p+".mlp.gate_proj.weight", intermediate, hidden)
	bf(p+".mlp.up_proj.weight", intermediate, hidden)
	bf(p+".mlp.down_proj.weight", hidden, intermediate)
	return tensors
}

func writeNativeAssistantAttentionDir(t testing.TB, tensors map[string]safetensors.Tensor) string {
	return writeNativeAssistantAttentionDirForBackbone(t, tensors, 8)
}

func writeNativeAssistantAttentionDirForBackbone(t testing.TB, tensors map[string]safetensors.Tensor, backbone int) string {
	t.Helper()
	dir := t.TempDir()
	cfg := []byte(core.Sprintf(`{
			"model_type": "gemma4_assistant",
			"backbone_hidden_size": %d,
			"num_centroids": 0,
			"centroid_intermediate_top_k": 0,
			"use_ordered_embeddings": false,
		"text_config": {
			"model_type": "gemma4_assistant",
			"hidden_size": 128,
			"num_hidden_layers": 1,
			"intermediate_size": 256,
			"num_attention_heads": 2,
			"num_key_value_heads": 2,
			"head_dim": 64,
			"vocab_size": 8,
			"rms_norm_eps": 0.000001,
			"max_position_embeddings": 16,
			"layer_types": ["sliding_attention"],
			"rope_parameters": {
				"sliding_attention": {"rope_theta": 10000, "partial_rotary_factor": 1.0}
				}
			}
		}`, backbone))
	if err := coreio.Local.Write(core.PathJoin(dir, "config.json"), string(cfg)); err != nil {
		t.Fatalf("write config.json: %v", err)
	}
	writeNativeAssistantTokenizer(t, dir)
	blob, err := safetensors.Encode(tensors)
	if err != nil {
		t.Fatalf("Encode assistant tensors: %v", err)
	}
	if err := coreio.Local.Write(core.PathJoin(dir, "model.safetensors"), string(blob)); err != nil {
		t.Fatalf("write model.safetensors: %v", err)
	}
	return dir
}

func writeNativeAssistantAttentionTargetDir(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	cfg := []byte(`{
		"model_type": "gemma4_text",
		"hidden_size": 8,
		"num_hidden_layers": 1,
		"intermediate_size": 256,
		"num_attention_heads": 2,
		"num_key_value_heads": 2,
		"head_dim": 64,
		"vocab_size": 8,
		"rms_norm_eps": 0.000001,
		"sliding_window": 16,
		"max_position_embeddings": 16,
		"layer_types": ["sliding_attention"],
		"rope_parameters": {
			"sliding_attention": {"rope_theta": 10000, "partial_rotary_factor": 1.0}
		}
	}`)
	if err := coreio.Local.Write(core.PathJoin(dir, "config.json"), string(cfg)); err != nil {
		t.Fatalf("write target config.json: %v", err)
	}
	return dir
}

const nativeTestGGUFTensorTypeBF16 = 30

type nativeTestGGUFTensor struct {
	Name string
	Type uint32
	Dims []uint64
	Data []byte
}

func writeNativeAssistantGGUF(t *testing.T, tensors map[string]safetensors.Tensor) string {
	t.Helper()
	path := core.PathJoin(t.TempDir(), "mtp-tiny.gguf")
	names := make([]string, 0, len(tensors))
	for name := range tensors {
		if nativeAssistantGGUFNameForTest(t, name) != "" {
			names = append(names, name)
		}
	}
	sort.Strings(names)
	payloads := make([]nativeTestGGUFTensor, 0, len(names))
	for _, name := range names {
		tensor := tensors[name]
		dims := make([]uint64, len(tensor.Shape))
		for i, dim := range tensor.Shape {
			dims[i] = uint64(dim)
		}
		payloads = append(payloads, nativeTestGGUFTensor{
			Name: nativeAssistantGGUFNameForTest(t, name),
			Type: nativeTestGGUFTensorTypeBF16,
			Dims: dims,
			Data: tensor.Data,
		})
	}
	writeNativeTestGGUF(t, path, nativeAssistantGGUFMetadata(), payloads)
	return path
}

const assistantGGUFArchName = "gemma4-assistant"

func nativeAssistantGGUFMetadata() []nativeTestGGUFMeta {
	const p = assistantGGUFArchName + "."
	return []nativeTestGGUFMeta{
		{Key: "general.architecture", ValueType: gguf.ValueTypeString, Value: assistantGGUFArchName},
		{Key: "general.alignment", ValueType: gguf.ValueTypeUint32, Value: uint32(32)},
		{Key: p + "block_count", ValueType: gguf.ValueTypeUint32, Value: uint32(2)},
		{Key: p + "embedding_length", ValueType: gguf.ValueTypeUint32, Value: uint32(4)},
		{Key: p + "embedding_length_out", ValueType: gguf.ValueTypeUint32, Value: uint32(8)},
		{Key: p + "attention.head_count", ValueType: gguf.ValueTypeUint32, Value: uint32(2)},
		{Key: p + "attention.head_count_kv", ValueType: gguf.ValueTypeUint32, Value: uint32(2)},
		{Key: p + "attention.key_length", ValueType: gguf.ValueTypeUint32, Value: uint32(2)},
		{Key: p + "attention.sliding_window_pattern", ValueType: gguf.ValueTypeUint32, Value: uint32(2)},
		{Key: p + "attention.sliding_window", ValueType: gguf.ValueTypeUint32, Value: uint32(16)},
		{Key: p + "attention.shared_kv_layers", ValueType: gguf.ValueTypeUint32, Value: uint32(0)},
		{Key: p + "feed_forward_length", ValueType: gguf.ValueTypeUint32, Value: uint32(8)},
		{Key: p + "context_length", ValueType: gguf.ValueTypeUint32, Value: uint32(16)},
	}
}

func nativeAssistantGGUFNameForTest(t *testing.T, hf string) string {
	t.Helper()
	base := []string{
		"token_embd.weight",
		"output_norm.weight",
		"nextn.pre_projection.weight",
		"nextn.post_projection.weight",
	}
	for _, name := range base {
		if g4.AssistantGGUFWeightName(name) == hf {
			return name
		}
	}
	leaves := []string{
		"attn_norm.weight",
		"post_attention_norm.weight",
		"ffn_norm.weight",
		"post_ffw_norm.weight",
		"attn_q.weight",
		"attn_q_norm.weight",
		"attn_output.weight",
		"ffn_gate.weight",
		"ffn_up.weight",
		"ffn_down.weight",
		"layer_output_scale.weight",
	}
	for layer := range 4 {
		for _, leaf := range leaves {
			name := core.Sprintf("blk.%d.%s", layer, leaf)
			mapped := g4.AssistantGGUFWeightName(name)
			if mapped == hf || (leaf == "layer_output_scale.weight" && mapped == hf+".weight") {
				return name
			}
		}
	}
	return ""
}

type nativeTestGGUFMeta struct {
	Key       string
	ValueType uint32
	Value     any
}

func writeNativeTestGGUF(t *testing.T, path string, metadata []nativeTestGGUFMeta, tensors []nativeTestGGUFTensor) {
	t.Helper()
	created := core.Create(path)
	if !created.OK {
		t.Fatalf("create gguf: %v", created.Value)
	}
	file := created.Value.(*core.OSFile)
	defer file.Close()
	writeNativeTestGGUFScalar(t, file, uint32(0x46554747))
	writeNativeTestGGUFScalar(t, file, uint32(3))
	writeNativeTestGGUFScalar(t, file, uint64(len(tensors)))
	writeNativeTestGGUFScalar(t, file, uint64(len(metadata)))
	for _, entry := range metadata {
		writeNativeTestGGUFString(t, file, entry.Key)
		writeNativeTestGGUFScalar(t, file, entry.ValueType)
		writeNativeTestGGUFValue(t, file, entry)
	}
	var offset uint64
	offsets := make([]uint64, len(tensors))
	for i, tensor := range tensors {
		offset += nativeTestGGUFAlignPadding(offset, 32)
		offsets[i] = offset
		offset += uint64(len(tensor.Data))
	}
	for i, tensor := range tensors {
		writeNativeTestGGUFString(t, file, tensor.Name)
		writeNativeTestGGUFScalar(t, file, uint32(len(tensor.Dims)))
		for _, dim := range tensor.Dims {
			writeNativeTestGGUFScalar(t, file, dim)
		}
		writeNativeTestGGUFScalar(t, file, tensor.Type)
		writeNativeTestGGUFScalar(t, file, offsets[i])
	}
	position, err := file.Seek(0, 1)
	if err != nil {
		t.Fatalf("seek gguf: %v", err)
	}
	writeNativeTestGGUFPadding(t, file, nativeTestGGUFAlignPadding(uint64(position), 32))
	var written uint64
	for i, tensor := range tensors {
		writeNativeTestGGUFPadding(t, file, offsets[i]-written)
		if _, err := file.Write(tensor.Data); err != nil {
			t.Fatalf("write gguf tensor: %v", err)
		}
		written = offsets[i] + uint64(len(tensor.Data))
	}
}

func writeNativeTestGGUFValue(t *testing.T, file *core.OSFile, entry nativeTestGGUFMeta) {
	t.Helper()
	switch entry.ValueType {
	case gguf.ValueTypeString:
		value, ok := entry.Value.(string)
		if !ok {
			t.Fatalf("metadata %s = %T, want string", entry.Key, entry.Value)
		}
		writeNativeTestGGUFString(t, file, value)
	case gguf.ValueTypeUint32:
		value, ok := entry.Value.(uint32)
		if !ok {
			t.Fatalf("metadata %s = %T, want uint32", entry.Key, entry.Value)
		}
		writeNativeTestGGUFScalar(t, file, value)
	default:
		t.Fatalf("unsupported native test gguf metadata type %d", entry.ValueType)
	}
}

func writeNativeTestGGUFString(t *testing.T, file *core.OSFile, value string) {
	t.Helper()
	writeNativeTestGGUFScalar(t, file, uint64(len(value)))
	if _, err := file.Write([]byte(value)); err != nil {
		t.Fatalf("write gguf string: %v", err)
	}
}

func writeNativeTestGGUFScalar(t *testing.T, file *core.OSFile, value any) {
	t.Helper()
	if err := binary.Write(file, binary.LittleEndian, value); err != nil {
		t.Fatalf("write gguf scalar: %v", err)
	}
}

func writeNativeTestGGUFPadding(t *testing.T, file *core.OSFile, n uint64) {
	t.Helper()
	if n == 0 {
		return
	}
	padding := make([]byte, int(n))
	if _, err := file.Write(padding); err != nil {
		t.Fatalf("write gguf padding: %v", err)
	}
}

func nativeTestGGUFAlignPadding(offset, alignment uint64) uint64 {
	if alignment == 0 {
		return 0
	}
	return (alignment - (offset % alignment)) % alignment
}

func newNativeAssistantGenerateFixture(t testing.TB) (*AssistantPair, func() *ArchSession) {
	t.Helper()
	const hidden, heads, kvHeads, headDim, ff, vocab = 128, 2, 2, 64, 256, 8
	layers := []DecodeLayerWeights{forwardLayer(hidden, heads, kvHeads, headDim, ff, 701)}
	embed := toBF16Bytes(syntheticFloat32(vocab*hidden, 703))
	g := &BF16Model{
		Layers:    layers,
		Embed:     embed,
		FinalNorm: toBF16Bytes(syntheticFloat32(hidden, 707)),
		LMHead:    embed,
		Tied:      true,
	}
	arch := model.Arch{
		Hidden: hidden, Heads: heads, KVHeads: kvHeads, HeadDim: headDim, FF: ff, Vocab: vocab,
		GlobalHeadDim: headDim, GlobalKVHeads: kvHeads,
		Eps: 1e-5, AttnScale: 0.125, RopeBase: 10000, RopeScale: 1, RopeLocalBase: 10000,
		RotaryDim: headDim, RotaryDimLocal: headDim, SlidingWindow: 16,
		Layer: model.DeriveLayers([]string{"sliding_attention"}, 0),
	}
	assistantDir := writeNativeAssistantAttentionDirForBackbone(t, nativeAssistantAttentionTensorsForBackbone(hidden), hidden)
	assistant, err := LoadAssistantDir(assistantDir)
	if err != nil {
		t.Fatalf("LoadAssistantDir: %v", err)
	}
	pair := &AssistantPair{TargetArch: arch, Assistant: assistant}
	if err := validateNativeAssistantPair(pair); err != nil {
		_ = pair.Close()
		t.Fatalf("validateNativeAssistantPair: %v", err)
	}
	mk := func() *ArchSession {
		s, err := NewArchSession(g, arch, 64)
		if err != nil {
			t.Fatalf("NewArchSession: %v", err)
		}
		head := &headEncoder{
			finalNorm: copyView(g.FinalNorm),
			weight:    copyView(g.LMHead),
			dModel:    arch.Hidden,
			vocab:     arch.Vocab,
			eps:       arch.Eps,
			softCap:   arch.SoftCap,
		}
		s.headEnc = head
		s.head = func(hidden []byte, skipSoftcap bool) ([]byte, error) {
			return head.encode(hidden, skipSoftcap)
		}
		s.greedy = func(hidden []byte, suppress []int32) (int32, bool, error) {
			return head.greedyInPool(hidden, suppress)
		}
		s.markDefaultHeadFunc()
		s.markDefaultGreedyFunc()
		return s
	}
	return pair, mk
}

func nativeAssistantPromptWhoseFirstTargetTokenIsNot(t testing.TB, mk func() *ArchSession, excluded int32) []int32 {
	t.Helper()
	candidates := [][]int32{
		{1, 5, 3},
		{2, 4, 6},
		{3, 1, 7},
		{4, 2, 5},
		{5, 3, 1},
		{6, 7, 2},
	}
	for _, prompt := range candidates {
		got, err := mk().Generate(prompt, 1, -1)
		if err != nil {
			t.Fatalf("reference Generate(%v): %v", prompt, err)
		}
		if len(got) == 1 && got[0] != excluded {
			return prompt
		}
	}
	t.Fatalf("no prompt produced a first target token outside %d", excluded)
	return nil
}

func nativeAssistantPromptWithAcceptedFirstDraft(t testing.TB, pair *AssistantPair, mk func() *ArchSession) []int32 {
	t.Helper()
	const fixtureVocab = 8
	for a := range int32(fixtureVocab) {
		for b := range int32(fixtureVocab) {
			for c := range int32(fixtureVocab) {
				prompt := []int32{a, b, c}
				target := mk()
				if err := target.prepareAssistantPrompt(prompt); err != nil {
					t.Fatalf("prepareAssistantPrompt(%v): %v", prompt, err)
				}
				logits, err := target.BoundaryLogits()
				if err != nil {
					t.Fatalf("BoundaryLogits(%v): %v", prompt, err)
				}
				first, err := greedyBF16Suppressed(logits, target.arch.Vocab, nil)
				if err != nil {
					t.Fatalf("greedyBF16Suppressed(%v): %v", prompt, err)
				}
				draft, err := pair.DraftBlockFromSession(target, prompt[len(prompt)-1], 1)
				if err != nil {
					t.Fatalf("DraftBlockFromSession(%v): %v", prompt, err)
				}
				if len(draft.Tokens) == 1 && draft.Tokens[0] == first {
					return prompt
				}
			}
		}
	}
	t.Fatal("no prompt produced an accepted first assistant draft")
	return nil
}

func nativeAssistantReferenceSampledDraftBlock(t testing.TB, pair *AssistantPair, target *ArchSession, prompt []int32, maxDraftTokens int, params model.SampleParams, sampler *model.Sampler) []int32 {
	t.Helper()
	if err := target.prepareAssistantPrompt(prompt); err != nil {
		t.Fatalf("prepareAssistantPrompt(reference %v): %v", prompt, err)
	}
	targetKVs, err := pair.targetKVByLayerTypeFromSessionScratch(target)
	if err != nil {
		t.Fatalf("targetKVByLayerTypeFromSessionScratch(reference): %v", err)
	}
	currentHidden, err := target.boundaryNormedHiddenScratch()
	if err != nil {
		t.Fatalf("boundaryNormedHiddenScratch(reference): %v", err)
	}
	currentToken := prompt[len(prompt)-1]
	tokens := make([]int32, 0, maxDraftTokens)
	for len(tokens) < maxDraftTokens {
		tokenEmbedding, err := target.embedID(currentToken)
		if err != nil {
			t.Fatalf("embedID(reference %d): %v", currentToken, err)
		}
		projectedOut := target.mtpProjectionScratch(pair.Assistant.Arch.Hidden * bf16Size)
		projected, err := pair.Assistant.DraftInputProjectionInto(projectedOut, tokenEmbedding, currentHidden)
		if err != nil {
			t.Fatalf("DraftInputProjectionInto(reference): %v", err)
		}
		normedOut := target.mtpDraftScratch(&target.mtpDraftNormed, pair.Assistant.Arch.Hidden*bf16Size)
		hiddenOut := target.mtpDraftScratch(&target.mtpDraftHidden, pair.TargetArch.Hidden*bf16Size)
		logitsOut := target.mtpDraftScratch(&target.mtpDraftLogits, pair.Assistant.Arch.Vocab*bf16Size)
		logitScores := target.mtpDraftLogitScoreScratch(pair.Assistant.NumCentroids)
		logitSelected := target.mtpDraftLogitSelectedScratch(pair.Assistant.CentroidIntermediateTopK)
		target.mtpDraftLayerScratch.usePinnedBacking()
		step, err := pair.draftStepFromProjectedIntoWithSuppress(projected, targetKVs, normedOut, hiddenOut, logitsOut, logitScores, logitSelected, &target.mtpDraftLayerScratch, params.SuppressTokens)
		if err != nil {
			t.Fatalf("draftStepFromProjectedIntoWithSuppress(reference): %v", err)
		}
		currentToken, err = sampler.Sample(step.Logits, pair.Assistant.Arch.Vocab, params)
		if err != nil {
			t.Fatalf("Sample(reference): %v", err)
		}
		tokens = append(tokens, currentToken)
		currentHidden = step.Hidden
	}
	return tokens
}

func nativeAssistantSampledPromptWithRejectedFirstDraft(t testing.TB, pair *AssistantPair, mk func() *ArchSession, params model.SampleParams) ([]int32, uint64, int32) {
	t.Helper()
	for _, prompt := range nativeAssistantWordedPromptCandidates() {
		for seed := uint64(1); seed <= 512; seed++ {
			target := mk()
			if err := target.prepareAssistantPrompt(prompt); err != nil {
				t.Fatalf("prepareAssistantPrompt(%v): %v", prompt, err)
			}
			pickParams := target.mtpSamplePickParams(params, nil, 0)
			draft, err := pair.draftBlockSampledFromSessionWithSuppress(target, prompt[len(prompt)-1], 1, false, pickParams, model.NewSampler(0))
			if err != nil {
				t.Fatalf("draftBlockSampledFromSessionWithSuppress(%v): %v", prompt, err)
			}
			sampled, err := target.sampleMTPTokenFromHidden(target.retainedHidden, model.NewSampler(seed), pickParams, nil)
			if err != nil {
				t.Fatalf("sampleMTPTokenFromHidden(%v, seed=%d): %v", prompt, seed, err)
			}
			if len(draft.Tokens) == 1 && draft.Tokens[0] != sampled {
				return prompt, seed, sampled
			}
		}
	}
	t.Fatal("no five-token prompt and sampler seed produced a rejected first assistant draft")
	return nil, 0, 0
}

func nativeAssistantPromptWhoseTargetTokensAvoid(t testing.TB, mk func() *ArchSession, excluded int32, maxNew int) []int32 {
	t.Helper()
	candidates := [][]int32{
		{1, 5, 3},
		{2, 4, 6},
		{3, 1, 7},
		{4, 2, 5},
		{5, 3, 1},
		{6, 7, 2},
		{7, 6, 4},
		{1, 2, 7, 3},
		{3, 5, 2, 6},
		{6, 4, 1, 7},
	}
	for _, prompt := range candidates {
		got, err := mk().Generate(prompt, maxNew, -1)
		if err != nil {
			t.Fatalf("reference Generate(%v): %v", prompt, err)
		}
		avoids := !slices.Contains(got, excluded)
		if avoids {
			return prompt
		}
	}
	t.Fatalf("no prompt produced %d target tokens avoiding %d", maxNew, excluded)
	return nil
}

func nativeAssistantPromptWhoseTargetTokensStartThenAvoid(t testing.TB, mk func() *ArchSession, first, excluded int32, maxNew int) []int32 {
	t.Helper()
	const fixtureVocab = 8
	for a := range int32(fixtureVocab) {
		for b := range int32(fixtureVocab) {
			for c := range int32(fixtureVocab) {
				prompt := []int32{a, b, c}
				got, err := mk().Generate(prompt, maxNew, -1)
				if err != nil {
					t.Fatalf("reference Generate(%v): %v", prompt, err)
				}
				if len(got) == 0 || got[0] != first {
					continue
				}
				avoids := !slices.Contains(got[1:], excluded)
				if avoids {
					return prompt
				}
			}
		}
	}
	t.Fatalf("no prompt produced first target token %d followed by %d tokens avoiding %d", first, maxNew-1, excluded)
	return nil
}

func nativeAssistantSampledVerifierRejectFixture(t testing.TB, mk func() *ArchSession, params model.SampleParams) ([]int32, uint64, int32, int32) {
	t.Helper()
	candidates := [][]int32{
		{1, 5, 3},
		{2, 4, 6},
		{3, 1, 7},
		{4, 2, 5},
		{5, 3, 1},
		{6, 7, 2},
	}
	for _, prompt := range candidates {
		greedy, err := mk().Generate(prompt, 1, -1)
		if err != nil {
			t.Fatalf("reference Generate(%v): %v", prompt, err)
		}
		for seed := uint64(1); seed <= 64; seed++ {
			for draft := range int32(8) {
				probe := mk()
				if err := probe.PrefillTokens(prompt); err != nil {
					t.Fatalf("PrefillTokens(%v): %v", prompt, err)
				}
				logits, err := probe.BoundaryLogits()
				if err != nil {
					t.Fatalf("BoundaryLogits(%v): %v", prompt, err)
				}
				sampled, err := model.NewSampler(seed).Sample(logits, probe.arch.Vocab, params)
				if err != nil {
					t.Fatalf("sample verifier logits(%v, seed %d, draft %d): %v", prompt, seed, draft, err)
				}
				if len(greedy) == 1 && sampled != greedy[0] && sampled != draft {
					return prompt, seed, sampled, draft
				}
			}
		}
	}
	t.Fatal("no sampled verifier fixture produced a reject token different from greedy")
	return nil, 0, 0, 0
}

func nativeAssistantSessionTargetArchForTest() model.Arch {
	return model.Arch{
		Hidden: 8, Vocab: 8, Heads: 2, KVHeads: 2, HeadDim: 2, SlidingWindow: 4,
		Layer: []model.LayerSpec{
			{Attention: model.SlidingAttention, KVShareFrom: 0, CacheIndex: 0, HeadDim: 2, KVHeads: 2},
			{Attention: model.GlobalAttention, KVShareFrom: 1, CacheIndex: 1, HeadDim: 2, KVHeads: 2},
			{Attention: model.SlidingAttention, KVShareFrom: 0, CacheIndex: -1, HeadDim: 2, KVHeads: 2},
			{Attention: model.GlobalAttention, KVShareFrom: 1, CacheIndex: -1, HeadDim: 2, KVHeads: 2},
		},
	}
}

func nativeAssistantSessionRowsForTest(rows, rowBytes int, seed byte) []byte {
	out := make([]byte, rows*rowBytes)
	for row := range rows {
		for col := range rowBytes {
			out[row*rowBytes+col] = seed + byte(row+col)
		}
	}
	return out
}

func nativeAssistantSessionKVRowsForTest(tokens, kvHeads, headDim int, seed byte) []byte {
	rowBytes := kvHeads * headDim * bf16Size
	out := make([]byte, tokens*rowBytes)
	for token := range tokens {
		for head := range kvHeads {
			out[token*rowBytes+head*headDim*bf16Size] = seed + byte(token*0x10+head)
		}
	}
	return out
}
