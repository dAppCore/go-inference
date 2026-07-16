// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import (
	"bytes"
	"context"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/inference/engine"
	"dappco.re/go/inference/kv"
	"dappco.re/go/inference/model"
)

func hipEngineSessionForTest(tokens []int32) *hipEngineSession {
	return &hipEngineSession{
		cfg:     hipGemma4Q4ForwardConfig{Layers: []hipGemma4Q4Layer0Config{{HeadDim: 1}}},
		loaded:  &hipLoadedModel{contextSize: 32},
		tokens:  append([]int32(nil), tokens...),
		pending: append([]int32(nil), tokens...),
	}
}

func TestHipEngineSession_RangeKVBlocks_Good_HonoursBlockStartToken(t *testing.T) {
	session := hipEngineSessionForTest([]int32{1, 2, 3, 4, 5, 6})
	var blocks []kv.Block
	err := session.RangeKVBlocks(2, kv.CaptureOptions{BlockStartToken: 4}, func(block kv.Block) (bool, error) {
		blocks = append(blocks, block)
		return true, nil
	})

	core.RequireNoError(t, err)
	core.AssertEqual(t, 1, len(blocks))
	core.AssertEqual(t, 2, blocks[0].Index)
	core.AssertEqual(t, 4, blocks[0].TokenStart)
	core.AssertEqual(t, 2, blocks[0].TokenCount)
	core.AssertEqual(t, 6, blocks[0].Snapshot.TokenOffset)
	core.AssertEqual(t, []int32{5, 6}, blocks[0].Snapshot.Tokens)
}

// TestHipEngineSession_RestoreFromKV_Good_PreservesRawDevicePageState proves
// that a capture preserves the resident encoded-page representation. The test
// page has noncanonical scales: reconstructing it from decoded float32 values
// changes the quantized bytes, while a raw page restore must not.
func TestHipEngineSession_RestoreFromKV_Good_PreservesRawDevicePageState(t *testing.T) {
	source := hipEngineSessionWithNonCanonicalDeviceKVForTest(t)
	defer func() { _ = source.Close() }()
	source.generated = []int32{8}

	snapshot, err := source.CaptureKVWithOptions(kv.CaptureOptions{})
	core.RequireNoError(t, err)
	if len(snapshot.Layers) != 1 || len(snapshot.Layers[0].TurboQuantPayloads) == 0 {
		t.Fatal("CaptureKVWithOptions did not retain the raw device KV page")
	}
	core.AssertEqual(t, 2, snapshot.Layers[0].CacheIndex)

	restored := &hipEngineSession{cfg: source.cfg, mode: source.mode, driver: source.driver}
	defer func() { _ = restored.Close() }()
	core.RequireNoError(t, restored.RestoreFromKV(context.Background(), snapshot))
	core.AssertEqual(t, []int32{8}, restored.generated)

	got, err := restored.CaptureKVWithOptions(kv.CaptureOptions{})
	core.RequireNoError(t, err)
	core.AssertEqual(t, snapshot.Layers[0].CacheIndex, got.Layers[0].CacheIndex)
	core.AssertEqual(t, len(snapshot.Layers[0].TurboQuantPayloads), len(got.Layers[0].TurboQuantPayloads))
	for index := range snapshot.Layers[0].TurboQuantPayloads {
		if !bytes.Equal(snapshot.Layers[0].TurboQuantPayloads[index], got.Layers[0].TurboQuantPayloads[index]) {
			t.Fatalf("CaptureKV -> RestoreFromKV -> CaptureKV changed raw device KV page %d", index)
		}
	}
	if !bytes.Equal(snapshot.Layers[0].KeyBytes, got.Layers[0].KeyBytes) ||
		!bytes.Equal(snapshot.Layers[0].ValueBytes, got.Layers[0].ValueBytes) {
		t.Fatal("CaptureKV -> RestoreFromKV -> CaptureKV changed the encoded device KV state")
	}
}

// TestHipEngineSession_GenerateFromCacheEach_Good_ReplaysMaterializedBoundary
// proves that a portable snapshot whose K/V already includes every token can
// still continue. HIP's combined entry point needs one seed token, so the
// session rolls back exactly the final row and forwards only that token.
func TestHipEngineSession_GenerateFromCacheEach_Good_ReplaysMaterializedBoundary(t *testing.T) {
	source := hipEngineSessionWithNonCanonicalDeviceKVForTest(t)
	defer func() { _ = source.Close() }()
	layer, cleanup := hipGemma4Q4FixtureConfig(t, source.driver, 0, 2, 4, 8)
	defer cleanup()
	source.cfg = hipGemma4Q4ForwardConfig{Layers: []hipGemma4Q4Layer0Config{layer}}
	source.generated = []int32{8}
	snapshot, err := source.CaptureKVWithOptions(kv.CaptureOptions{})
	core.RequireNoError(t, err)

	restored := &hipEngineSession{
		cfg:    source.cfg,
		mode:   source.mode,
		driver: source.driver,
		loaded: &hipLoadedModel{contextSize: 32},
	}
	defer func() { _ = restored.Close() }()
	core.RequireNoError(t, restored.RestoreFromKV(context.Background(), snapshot))
	core.AssertEqual(t, []int32(nil), restored.pending)

	var gotPrompt []int32
	gotInitialTokens := -1
	restored.drive = func(_ context.Context, prompt []int32, _ []byte, _ inference.GenerateConfig, _ *model.Sampler, state *hipGemma4Q4DeviceDecodeState, emit func(int32) bool) ([]int32, *hipGemma4Q4DeviceDecodeState, error) {
		gotPrompt = append([]int32(nil), prompt...)
		if state != nil {
			gotInitialTokens = state.maxLayerTokenCount()
		}
		if emit != nil {
			emit(9)
		}
		return []int32{9}, state, nil
	}

	generated, err := restored.GenerateFromCacheEach(1, -1, nil)
	core.RequireNoError(t, err)
	core.AssertEqual(t, []int32{9}, generated)
	core.AssertEqual(t, []int32{2}, gotPrompt)
	core.AssertEqual(t, 1, gotInitialTokens)
	core.AssertEqual(t, []int32{8, 9}, restored.generated)
}

func TestHipEngineSession_GenerateFromCacheEach_Good_UsesRestoredBoundaryLogits(t *testing.T) {
	source := hipEngineSessionWithNonCanonicalDeviceKVForTest(t)
	defer func() { _ = source.Close() }()
	layer, cleanup := hipGemma4Q4FixtureConfig(t, source.driver, 0, 2, 4, 8)
	defer cleanup()
	source.cfg = hipGemma4Q4ForwardConfig{Layers: []hipGemma4Q4Layer0Config{layer}}
	source.generated = []int32{8}
	snapshot, err := source.CaptureKVWithOptions(kv.CaptureOptions{})
	core.RequireNoError(t, err)
	snapshot.LogitShape = []int32{1, 2}
	snapshot.Logits = []float32{-4, 6}

	restored := &hipEngineSession{
		cfg:    source.cfg,
		mode:   source.mode,
		driver: source.driver,
		loaded: &hipLoadedModel{contextSize: 32},
	}
	defer func() { _ = restored.Close() }()
	core.RequireNoError(t, restored.RestoreFromKV(context.Background(), snapshot))

	var gotPrompt []int32
	gotInitialTokens := -1
	gotMaxTokens := -1
	restored.drive = func(_ context.Context, prompt []int32, _ []byte, generate inference.GenerateConfig, _ *model.Sampler, state *hipGemma4Q4DeviceDecodeState, emit func(int32) bool) ([]int32, *hipGemma4Q4DeviceDecodeState, error) {
		gotPrompt = append([]int32(nil), prompt...)
		gotMaxTokens = generate.MaxTokens
		if state != nil {
			gotInitialTokens = state.maxLayerTokenCount()
		}
		if emit != nil {
			emit(0)
		}
		return []int32{0}, state, nil
	}
	var yielded []int32

	generated, err := restored.GenerateFromCacheEach(2, -1, func(id int32) bool {
		yielded = append(yielded, id)
		return true
	})
	core.RequireNoError(t, err)
	core.AssertEqual(t, []int32{1, 0}, generated)
	core.AssertEqual(t, []int32{1, 0}, yielded)
	core.AssertEqual(t, []int32{1}, gotPrompt)
	core.AssertEqual(t, 2, gotInitialTokens)
	core.AssertEqual(t, 1, gotMaxTokens)
	core.AssertEqual(t, []int32{8, 1, 0}, restored.generated)
}

func TestHipEngineSession_CaptureKVWithOptions_Good_PreservesRestoredBoundaryLogits(t *testing.T) {
	source := hipEngineSessionWithNonCanonicalDeviceKVForTest(t)
	defer func() { _ = source.Close() }()
	layer, cleanup := hipGemma4Q4FixtureConfig(t, source.driver, 0, 2, 4, 8)
	defer cleanup()
	source.cfg = hipGemma4Q4ForwardConfig{Layers: []hipGemma4Q4Layer0Config{layer}}
	snapshot, err := source.CaptureKVWithOptions(kv.CaptureOptions{})
	core.RequireNoError(t, err)
	snapshot.LogitShape = []int32{1, 1, 2}
	snapshot.Logits = []float32{-4, 6}

	restored := &hipEngineSession{cfg: source.cfg, mode: source.mode, driver: source.driver}
	defer func() { _ = restored.Close() }()
	core.RequireNoError(t, restored.RestoreFromKV(context.Background(), snapshot))

	roundTrip, err := restored.CaptureKVWithOptions(kv.CaptureOptions{})
	core.RequireNoError(t, err)
	core.AssertEqual(t, []int32{1, 1, 2}, roundTrip.LogitShape)
	core.AssertEqual(t, []float32{-4, 6}, roundTrip.Logits)

	var blocks []kv.Block
	core.RequireNoError(t, restored.RangeKVBlocks(1, kv.CaptureOptions{}, func(block kv.Block) (bool, error) {
		blocks = append(blocks, block)
		return true, nil
	}))
	core.AssertEqual(t, 2, len(blocks))
	core.AssertEqual(t, []float32(nil), blocks[0].Snapshot.Logits)
	core.AssertEqual(t, []int32{1, 1, 2}, blocks[1].Snapshot.LogitShape)
	core.AssertEqual(t, []float32{-4, 6}, blocks[1].Snapshot.Logits)

	state, err := restored.StateBlockSource(1)
	core.RequireNoError(t, err)
	core.AssertEqual(t, 2, len(state.Blocks))
	core.AssertEqual(t, []float32(nil), state.Blocks[0].Snapshot.Logits)
	core.AssertEqual(t, []int32{1, 1, 2}, state.Blocks[1].Snapshot.LogitShape)
	core.AssertEqual(t, []float32{-4, 6}, state.Blocks[1].Snapshot.Logits)
}

func TestHipEngineSession_RestoreFromKV_Good_PreservesMultiHeadRawDevicePageState(t *testing.T) {
	source := hipEngineSessionWithNonCanonicalDeviceKVForTest(t)
	defer func() { _ = source.Close() }()
	source.cfg.Layers[0].HeadDim = 1
	source.cfg.Layers[0].KeyHeads = 2

	snapshot, err := source.CaptureKVWithOptions(kv.CaptureOptions{})
	core.RequireNoError(t, err)
	core.AssertEqual(t, 2, snapshot.NumHeads)
	core.AssertEqual(t, 2, snapshot.SeqLen)

	restored := &hipEngineSession{cfg: source.cfg, mode: source.mode, driver: source.driver}
	defer func() { _ = restored.Close() }()
	core.RequireNoError(t, restored.RestoreFromKV(context.Background(), snapshot))

	got, err := restored.CaptureKVWithOptions(kv.CaptureOptions{})
	core.RequireNoError(t, err)
	core.AssertEqual(t, snapshot.NumHeads, got.NumHeads)
	core.AssertEqual(t, snapshot.SeqLen, got.SeqLen)
	if !bytes.Equal(snapshot.Layers[0].TurboQuantPayloads[0], got.Layers[0].TurboQuantPayloads[0]) {
		t.Fatal("multi-head raw device KV page changed across capture and restore")
	}
}

func TestHipEngineSession_RangeKVBlocks_Good_PreservesRawDevicePageWindows(t *testing.T) {
	source := hipEngineSessionWithNonCanonicalDeviceKVForTest(t)
	defer func() { _ = source.Close() }()
	source.tokens = append(source.tokens, 3)

	var blocks []kv.Block
	err := source.RangeKVBlocks(1, kv.CaptureOptions{RawKVOnly: true}, func(block kv.Block) (bool, error) {
		blocks = append(blocks, block)
		return true, nil
	})
	core.RequireNoError(t, err)
	if len(blocks) != 3 {
		t.Fatalf("RangeKVBlocks yielded %d blocks, want 3", len(blocks))
	}

	wantKeyScales := []float32{0.1, 0.2}
	wantKeyCodes := [][]int8{{5, 7}, {-4, 6}}
	wantValueScales := []float32{0.25, 0.5}
	wantValueCodes := [][]int8{{2, 3}, {-2, 4}}
	for index, block := range blocks[:2] {
		if block.Snapshot == nil || len(block.Snapshot.Layers) != 1 {
			t.Fatalf("block %d has no layer snapshot", index)
		}
		layer := block.Snapshot.Layers[0]
		core.AssertEqual(t, hipKVSnapshotDevicePayloadMode, layer.CacheMode)
		if len(layer.TurboQuantPayloads) != 1 {
			t.Fatalf("block %d raw device payload count = %d, want 1", index, len(layer.TurboQuantPayloads))
		}
		raw, err := rocmKVCacheBlockFromRawPayload(layer.TurboQuantPayloads[0])
		core.RequireNoError(t, err)
		core.AssertEqual(t, index, raw.tokenStart)
		core.AssertEqual(t, 1, raw.tokenCount)
		core.AssertEqual(t, []float32{wantKeyScales[index]}, raw.key.scales)
		core.AssertEqual(t, wantKeyCodes[index], raw.key.q8)
		core.AssertEqual(t, []float32{wantValueScales[index]}, raw.value.scales)
		core.AssertEqual(t, wantValueCodes[index][0], unpackSignedQ4(raw.value.packedQ4[0]&0x0f))
		core.AssertEqual(t, wantValueCodes[index][1], unpackSignedQ4(raw.value.packedQ4[0]>>4))
	}
	core.AssertEqual(t, []int32{3}, blocks[2].Snapshot.Tokens)
	core.AssertEqual(t, 0, blocks[2].Snapshot.SeqLen)
	core.AssertEqual(t, 0, len(blocks[2].Snapshot.Layers[0].TurboQuantPayloads))

	assembled, err := kv.AssembleBlocks(blocks)
	core.RequireNoError(t, err)
	core.AssertEqual(t, []int32{1, 2, 3}, assembled.Tokens)
	core.AssertEqual(t, 2, assembled.SeqLen)
	restored := &hipEngineSession{cfg: source.cfg, mode: source.mode, driver: source.driver}
	defer func() { _ = restored.Close() }()
	core.RequireNoError(t, restored.RestoreFromKV(context.Background(), assembled))
	roundTrip, err := restored.CaptureKVWithOptions(kv.CaptureOptions{})
	core.RequireNoError(t, err)
	core.AssertEqual(t, []int32{1, 2, 3}, roundTrip.Tokens)
	core.AssertEqual(t, 2, len(roundTrip.Layers[0].TurboQuantPayloads))
	for index := range blocks[:2] {
		if !bytes.Equal(blocks[index].Snapshot.Layers[0].TurboQuantPayloads[0], roundTrip.Layers[0].TurboQuantPayloads[index]) {
			t.Fatalf("restored raw device KV block %d changed encoded bytes", index)
		}
	}
}

func TestHipEngineSession_Generate_Good_KeepsUnforwardedFinalTokenPending(t *testing.T) {
	session := hipEngineSessionWithNonCanonicalDeviceKVForTest(t)
	defer func() { _ = session.Close() }()
	session.tokens = []int32{1, 2, 3}

	err := session.syncPendingWithDeviceLocked()

	core.RequireNoError(t, err)
	core.AssertEqual(t, []int32{3}, session.pending)
}

func TestHipEngineSession_PrefillTokenEmbeddings_Good(t *testing.T) {
	session := &hipEngineSession{
		loaded: &hipLoadedModel{modelInfo: inference.ModelInfo{HiddenSize: 2}, contextSize: 8},
	}
	var _ engine.VisionSession = session
	rows := [][]byte{{1, 2, 3, 4, 5, 6, 7, 8}, {9, 10, 11, 12, 13, 14, 15, 16}}
	core.RequireNoError(t, session.PrefillTokenEmbeddings([]int32{4, 77}, rows))
	core.AssertEqual(t, []int32{4, 77}, session.pending)
	core.AssertEqual(t, []int32{4, 77}, session.tokens)
	core.AssertEqual(t, []byte{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}, session.pendingEmbeddings)
	rows[0][0] = 99
	core.AssertEqual(t, byte(1), session.pendingEmbeddings[0])
}

func TestHipEngineSession_SetReuseCanonicalLanding_Good(t *testing.T) {
	session := hipEngineSessionForTest(nil)
	var _ engine.PromptReuseSession = session
	var _ engine.CanonicalLandingSession = session
	var _ engine.ContextDecodeSession = session

	session.SetReuseCanonicalLanding(true)
	core.AssertTrue(t, session.engine.DisableBatchedPrefill)
	session.SetReuseCanonicalLanding(false)
	core.AssertFalse(t, session.engine.DisableBatchedPrefill)
}

func TestHipEngineSession_RestoreFromKV_Good_ArmsCanonicalLandingForQuantizedKV(t *testing.T) {
	source := hipEngineSessionForTest([]int32{1})
	snapshot, err := source.CaptureKVWithOptions(kv.CaptureOptions{})
	core.RequireNoError(t, err)

	restored := &hipEngineSession{cfg: source.cfg, mode: rocmKVCacheModeKQ8VQ4}
	defer func() { _ = restored.Close() }()
	core.RequireNoError(t, restored.RestoreFromKV(context.Background(), snapshot))
	core.AssertTrue(t, restored.engine.DisableBatchedPrefill)
}

func TestHipEngineSession_RestoreFromKV_Good_KeepsBatchedLandingForExactDeviceKV(t *testing.T) {
	source := hipEngineSessionWithNonCanonicalDeviceKVForTest(t)
	defer func() { _ = source.Close() }()
	snapshot, err := source.CaptureKVWithOptions(kv.CaptureOptions{})
	core.RequireNoError(t, err)

	restored := &hipEngineSession{cfg: source.cfg, mode: rocmKVCacheModeKQ8VQ4, driver: source.driver}
	defer func() { _ = restored.Close() }()
	core.RequireNoError(t, restored.RestoreFromKV(context.Background(), snapshot))
	core.AssertFalse(t, restored.engine.DisableBatchedPrefill)
}

func TestHipEngineSession_PrefillTokenEmbeddings_Bad(t *testing.T) {
	session := &hipEngineSession{
		loaded: &hipLoadedModel{modelInfo: inference.ModelInfo{HiddenSize: 2}, contextSize: 8},
	}
	core.AssertError(t, session.PrefillTokenEmbeddings([]int32{1, 2}, [][]byte{{1, 2, 3, 4, 5, 6, 7, 8}}))
	core.AssertError(t, session.PrefillTokenEmbeddings([]int32{1}, [][]byte{{1, 2, 3}}))
}

func TestHipEngineSession_CaptureKVWithOptions_Bad_RejectsUnforwardedEmbeddings(t *testing.T) {
	session := &hipEngineSession{
		loaded:            &hipLoadedModel{modelInfo: inference.ModelInfo{HiddenSize: 1}, contextSize: 8},
		pending:           []int32{77},
		tokens:            []int32{77},
		pendingEmbeddings: []byte{0, 0, 0, 0},
	}
	_, err := session.CaptureKVWithOptions(kv.CaptureOptions{})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "custom embeddings")
}

func TestHipEngineSession_CaptureKVWithOptions_Bad_RejectsNegativeBlockStart(t *testing.T) {
	session := hipEngineSessionForTest([]int32{1})
	_, err := session.CaptureKVWithOptions(kv.CaptureOptions{BlockStartToken: -1})
	core.AssertError(t, err)
}

func hipEngineSessionWithNonCanonicalDeviceKVForTest(t testing.TB) *hipEngineSession {
	t.Helper()
	driver := &fakeHIPDriver{available: true}
	cache, err := newROCmKVCache(rocmKVCacheModeKQ8VQ4, 2)
	core.RequireNoError(t, err)
	raw, err := cache.rawBlock(rocmKVCacheBlock{
		tokenStart: 0,
		tokenCount: 2,
		keyWidth:   2,
		valueWidth: 2,
		key: rocmKVEncodedTensor{
			encoding: rocmKVEncodingQ8RowsI,
			length:   4,
			scales:   []float32{0.1, 0.2},
			q8:       []int8{5, 7, -4, 6},
		},
		value: rocmKVEncodedTensor{
			encoding: rocmKVEncodingQ4RowsI,
			length:   4,
			scales:   []float32{0.25, 0.5},
			packedQ4: []byte{packSignedQ4(2) | packSignedQ4(3)<<4, packSignedQ4(-2) | packSignedQ4(4)<<4},
		},
	})
	core.RequireNoError(t, err)
	page, err := rocmDeviceKVPageFromRawPayload(driver, raw)
	core.RequireNoError(t, err)
	deviceCache := rocmBorrowDeviceKVCache(driver, rocmKVCacheModeKQ8VQ4, 2, page.tokenCount, []rocmDeviceKVPage{page}, false)
	table, err := deviceCache.kernelDescriptorTableLabeled("", "")
	core.RequireNoError(t, err)
	launch, err := deviceCache.KernelLaunchDescriptor(table)
	core.RequireNoError(t, err)
	device := hipNewGemma4Q4DeviceDecodeState(rocmKVCacheModeKQ8VQ4, 1)
	device.layers = append(device.layers, hipGemma4Q4DeviceLayerKVState{cache: deviceCache, descriptorTable: table, launch: launch})
	return &hipEngineSession{
		cfg:    hipGemma4Q4ForwardConfig{Layers: []hipGemma4Q4Layer0Config{{HeadDim: 2}}},
		mode:   rocmKVCacheModeKQ8VQ4,
		driver: driver,
		device: device,
		tokens: []int32{1, 2},
	}
}

func TestHipEngineSession_GenerateFromCacheEachContext_Ugly_PreservesCancellation(t *testing.T) {
	session := hipEngineSessionForTest([]int32{1})
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	_, err := session.GenerateFromCacheEachContext(ctx, 1, -1, nil)

	core.AssertErrorIs(t, err, context.Canceled)
	core.AssertEqual(t, []int32{1}, session.pending)
	core.AssertEqual(t, []int32{1}, session.tokens)
}

func TestHipEngineSession_GenerateSampledFromCacheEachContext_Bad_RejectsNilSampler(t *testing.T) {
	session := hipEngineSessionForTest([]int32{1})

	_, err := session.GenerateSampledFromCacheEachContext(context.Background(), 1, nil, nil, model.SampleParams{Temperature: 1}, nil, nil)

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "nil sampler")
	core.AssertEqual(t, []int32{1}, session.pending)
	core.AssertEqual(t, []int32{1}, session.tokens)
}

func TestHipEngineSession_PrefillTokensCached_Good_ReusesAppendPrefix(t *testing.T) {
	session := hipEngineSessionForTest([]int32{1, 2, 3})
	session.pending = nil

	reused, err := session.PrefillTokensCached([]int32{1, 2, 3, 4, 5})

	core.RequireNoError(t, err)
	core.AssertEqual(t, 3, reused)
	core.AssertEqual(t, []int32{1, 2, 3, 4, 5}, session.tokens)
	core.AssertEqual(t, []int32{4, 5}, session.pending)
}

func TestHipEngineSession_PrefillTokensCached_Good_ExactHitKeepsDecodeSeed(t *testing.T) {
	session := hipEngineSessionForTest([]int32{1, 2, 3})

	reused, err := session.PrefillTokensCached([]int32{1, 2, 3})

	core.RequireNoError(t, err)
	core.AssertEqual(t, 0, reused)
	core.AssertEqual(t, []int32{1, 2, 3}, session.tokens)
	core.AssertEqual(t, []int32{1, 2, 3}, session.pending)
}

func TestHipEngineSession_PrefillTokensCached_Good_ExactHitKeepsBoundaryLogits(t *testing.T) {
	session := hipEngineSessionWithNonCanonicalDeviceKVForTest(t)
	defer func() { _ = session.Close() }()
	layer, cleanup := hipGemma4Q4FixtureConfig(t, session.driver, 0, 2, 4, 8)
	defer cleanup()
	session.cfg = hipGemma4Q4ForwardConfig{Layers: []hipGemma4Q4Layer0Config{layer}}
	session.generated = []int32{8}
	session.boundaryLogitShape = []int32{1, 2}
	session.boundaryLogits = []float32{-4, 6}
	device := session.device

	reused, err := session.PrefillTokensCached([]int32{1, 2})

	core.RequireNoError(t, err)
	core.AssertEqual(t, 2, reused)
	core.AssertEqual(t, []int32(nil), session.pending)
	core.AssertEqual(t, []int32(nil), session.generated)
	core.AssertEqual(t, []int32{1, 2}, session.boundaryLogitShape)
	core.AssertEqual(t, []float32{-4, 6}, session.boundaryLogits)
	if session.device != device {
		t.Fatal("exact boundary hit replaced the fully resident device cache")
	}
}

func TestHipEngineSession_PrefillTokensCached_Good_DivergenceResetsCold(t *testing.T) {
	session := hipEngineSessionForTest([]int32{1, 2, 3, 4})

	reused, err := session.PrefillTokensCached([]int32{1, 2, 9, 10})

	core.RequireNoError(t, err)
	core.AssertEqual(t, 0, reused)
	core.AssertEqual(t, []int32{1, 2, 9, 10}, session.tokens)
	core.AssertEqual(t, []int32{1, 2, 9, 10}, session.pending)
}

func TestHipEngineSession_PrefillTokensCached_Bad_RejectsContextOverflow(t *testing.T) {
	session := hipEngineSessionForTest([]int32{1})
	session.loaded.contextSize = 3

	_, err := session.PrefillTokensCached([]int32{1, 2, 3, 4})

	if err == nil {
		t.Fatal("PrefillTokensCached accepted a prompt beyond the context window")
	}
	core.AssertContains(t, err.Error(), "context window")
	core.AssertEqual(t, []int32{1}, session.tokens)
}

func TestHipEngineSession_PrefillTokens_Bad_ContextOverflowPreservesRetainedState(t *testing.T) {
	session := hipEngineSessionWithNonCanonicalDeviceKVForTest(t)
	defer func() { _ = session.Close() }()
	session.loaded = &hipLoadedModel{contextSize: 3}
	session.pending = []int32{2}
	session.generated = []int32{9}
	device := session.device

	err := session.PrefillTokens([]int32{3, 4, 5, 6})

	if err == nil {
		t.Fatal("PrefillTokens accepted a prompt beyond the context window")
	}
	core.AssertContains(t, err.Error(), "context window")
	core.AssertEqual(t, []int32{1, 2}, session.tokens)
	core.AssertEqual(t, []int32{2}, session.pending)
	core.AssertEqual(t, []int32{9}, session.generated)
	if session.device != device {
		t.Fatal("PrefillTokens overflow replaced retained device state")
	}
	core.AssertFalse(t, device.closed)
}

func TestHipEngineSession_AppendTokens_Bad_ContextOverflowPreservesRetainedState(t *testing.T) {
	session := hipEngineSessionWithNonCanonicalDeviceKVForTest(t)
	defer func() { _ = session.Close() }()
	session.loaded = &hipLoadedModel{contextSize: 3}
	session.pending = []int32{2}
	session.generated = []int32{9}
	device := session.device

	err := session.AppendTokens([]int32{3, 4})

	if err == nil {
		t.Fatal("AppendTokens accepted tokens beyond the context window")
	}
	core.AssertContains(t, err.Error(), "context window")
	core.AssertEqual(t, []int32{1, 2}, session.tokens)
	core.AssertEqual(t, []int32{2}, session.pending)
	core.AssertEqual(t, []int32{9}, session.generated)
	if session.device != device {
		t.Fatal("AppendTokens overflow replaced retained device state")
	}
	core.AssertFalse(t, device.closed)
}

func TestHipEngineSession_GenerateFromCacheEach_Bad_ContextOverflowPreservesRetainedState(t *testing.T) {
	session := hipEngineSessionForTest([]int32{1, 2})
	session.loaded.contextSize = 2
	called := false
	session.drive = func(_ context.Context, _ []int32, _ []byte, _ inference.GenerateConfig, _ *model.Sampler, state *hipGemma4Q4DeviceDecodeState, _ func(int32) bool) ([]int32, *hipGemma4Q4DeviceDecodeState, error) {
		called = true
		return []int32{9}, state, nil
	}

	_, err := session.GenerateFromCacheEach(1, -1, nil)

	if err == nil {
		t.Fatal("GenerateFromCacheEach accepted a generation budget beyond the context window")
	}
	core.AssertContains(t, err.Error(), "context window")
	core.AssertFalse(t, called)
	core.AssertEqual(t, []int32{1, 2}, session.tokens)
	core.AssertEqual(t, []int32{1, 2}, session.pending)
}

func TestHipEngineSession_AppendTokens_Bad_RejectsEmptyInputWithoutMutation(t *testing.T) {
	session := hipEngineSessionForTest([]int32{1, 2})
	session.generated = []int32{9}

	err := session.AppendTokens(nil)

	if err == nil {
		t.Fatal("AppendTokens accepted empty input")
	}
	core.AssertContains(t, err.Error(), "empty prompt tokens")
	core.AssertEqual(t, []int32{1, 2}, session.tokens)
	core.AssertEqual(t, []int32{1, 2}, session.pending)
	core.AssertEqual(t, []int32{9}, session.generated)
}

func TestHipEngineSession_PrefillTokensCached_Ugly_WrappedRingResetsCold(t *testing.T) {
	session := hipEngineSessionForTest([]int32{1, 2, 3, 4})
	session.cfg.Layers[0].SlidingWindow = 3

	reused, err := session.PrefillTokensCached([]int32{1, 2, 9})

	core.RequireNoError(t, err)
	core.AssertEqual(t, 0, reused)
	core.AssertEqual(t, []int32{1, 2, 9}, session.tokens)
	core.AssertEqual(t, []int32{1, 2, 9}, session.pending)
}

func TestHipEngineSession_PrefillTokensCached_Good_WrappedRingAppendIsSafe(t *testing.T) {
	session := hipEngineSessionForTest([]int32{1, 2, 3, 4})
	session.pending = nil
	session.cfg.Layers[0].SlidingWindow = 3

	reused, err := session.PrefillTokensCached([]int32{1, 2, 3, 4, 5})

	core.RequireNoError(t, err)
	core.AssertEqual(t, 4, reused)
	core.AssertEqual(t, []int32{5}, session.pending)
}

func TestHipEngineSession_RingRollbackUnsafe_Good(t *testing.T) {
	session := hipEngineSessionForTest(nil)
	session.cfg.Layers[0].SlidingWindow = 3
	core.AssertFalse(t, session.ringRollbackUnsafeLocked(3))
	core.AssertTrue(t, session.ringRollbackUnsafeLocked(4))
}

func TestHipEngineSession_ClearPromptCache_Good(t *testing.T) {
	session := hipEngineSessionWithNonCanonicalDeviceKVForTest(t)
	device := session.device
	session.pending = []int32{3}
	session.pendingEmbeddings = []byte{1, 2, 3, 4}
	session.generated = []int32{5}
	var clearer inference.PromptCacheClearer = session

	clearer.ClearPromptCache()

	if session.device != nil {
		t.Fatal("ClearPromptCache retained the device KV state")
	}
	core.AssertTrue(t, device.closed)
	core.AssertEqual(t, []int32(nil), session.pending)
	core.AssertEqual(t, []byte(nil), session.pendingEmbeddings)
	core.AssertEqual(t, []int32(nil), session.tokens)
	core.AssertEqual(t, []int32(nil), session.generated)
	core.AssertFalse(t, session.closed)
	core.RequireNoError(t, session.PrefillTokens([]int32{7}))
}

func TestHipEngineSession_ClearPromptCache_Bad_NilReceiver(t *testing.T) {
	var session *hipEngineSession
	var clearer inference.PromptCacheClearer = session

	clearer.ClearPromptCache()
}

func TestHipEngineSession_ClearPromptCache_Ugly_IsIdempotentOnBufferedState(t *testing.T) {
	session := hipEngineSessionForTest([]int32{1, 2})
	session.pendingEmbeddings = []byte{1, 2, 3, 4}
	var clearer inference.PromptCacheClearer = session

	clearer.ClearPromptCache()
	clearer.ClearPromptCache()

	core.AssertEqual(t, []int32(nil), session.pending)
	core.AssertEqual(t, []byte(nil), session.pendingEmbeddings)
	core.AssertEqual(t, []int32(nil), session.tokens)
	core.AssertEqual(t, []int32(nil), session.generated)
	core.AssertFalse(t, session.closed)
	core.RequireNoError(t, session.PrefillTokens([]int32{8}))
}

// TestHipEngineSession_RestoreStateBlocks_Good_SuffixPreservesNextDecode proves
// that a target holding the shared raw-KV prefix can graft the source's suffix
// blocks and enter HIP's combined decode with the same state as the source.
func TestHipEngineSession_RestoreStateBlocks_Good_SuffixPreservesNextDecode(t *testing.T) {
	source := hipEngineSessionWithNonCanonicalDeviceKVForTest(t)
	defer func() { _ = source.Close() }()
	source.tokens = []int32{1, 2, 3}
	source.pending = []int32{3}

	all, err := source.StateBlockSource(1)
	core.RequireNoError(t, err)
	prefix := all
	prefix.Position = 2
	prefix.Tokens = []int32{1, 2}
	prefix.Blocks = append([]kv.Block(nil), all.Blocks[:2]...)
	target := &hipEngineSession{cfg: source.cfg, mode: source.mode, driver: source.driver}
	defer func() { _ = target.Close() }()
	core.RequireNoError(t, target.RestoreStateBlocks(prefix))

	suffix, err := source.StateBlockSourceFrom(2, 1)
	core.RequireNoError(t, err)
	core.RequireNoError(t, target.RestoreStateBlocks(suffix))

	decode := func(_ context.Context, _ []int32, _ []byte, _ inference.GenerateConfig, _ *model.Sampler, state *hipGemma4Q4DeviceDecodeState, _ func(int32) bool) ([]int32, *hipGemma4Q4DeviceDecodeState, error) {
		if state == nil || len(state.layers) != 1 {
			return nil, nil, core.NewError("test decode state is unavailable")
		}
		host, err := state.HostState()
		if err != nil {
			return nil, nil, err
		}
		if len(host.Layers) != 1 || len(host.Layers[0].Keys) == 0 || len(host.Layers[0].Values) == 0 {
			return nil, nil, core.NewError("test decode K/V state is unavailable")
		}
		return []int32{int32((host.Layers[0].Keys[0] + host.Layers[0].Values[0]) * 1000)}, state, nil
	}
	source.drive = decode
	target.drive = decode

	want, err := source.GenerateFromCacheEach(1, -1, nil)
	core.RequireNoError(t, err)
	got, err := target.GenerateFromCacheEach(1, -1, nil)
	core.RequireNoError(t, err)
	core.AssertEqual(t, want, got)
}

// TestHipEngineSession_WarmPromptCache_Good_AvoidsRepeatedNativePrefill pins
// HIP's honest warm lifecycle: warming itself performs no speculative native
// work, while each later decode forwards only the unmaterialized final seed
// token instead of the already resident prompt prefix.
func TestHipEngineSession_WarmPromptCache_Good_AvoidsRepeatedNativePrefill(t *testing.T) {
	session := hipEngineSessionWithNonCanonicalDeviceKVForTest(t)
	defer func() { _ = session.Close() }()
	var nativePrefillPrompts [][]int32
	session.drive = func(_ context.Context, prompt []int32, _ []byte, _ inference.GenerateConfig, _ *model.Sampler, state *hipGemma4Q4DeviceDecodeState, _ func(int32) bool) ([]int32, *hipGemma4Q4DeviceDecodeState, error) {
		nativePrefillPrompts = append(nativePrefillPrompts, append([]int32(nil), prompt...))
		return []int32{9}, state, nil
	}

	core.RequireNoError(t, session.WarmPromptCache([]int32{1, 2, 3}))
	core.AssertEqual(t, 0, len(nativePrefillPrompts))
	_, err := session.GenerateFromCacheEach(1, -1, nil)
	core.RequireNoError(t, err)

	core.RequireNoError(t, session.WarmPromptCache([]int32{1, 2, 3}))
	core.AssertEqual(t, 1, len(nativePrefillPrompts))
	_, err = session.GenerateFromCacheEach(1, -1, nil)
	core.RequireNoError(t, err)

	core.AssertEqual(t, [][]int32{{3}, {3}}, nativePrefillPrompts)
}
