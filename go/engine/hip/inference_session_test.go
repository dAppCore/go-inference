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

	snapshot, err := source.CaptureKVWithOptions(kv.CaptureOptions{})
	core.RequireNoError(t, err)
	if len(snapshot.Layers) != 1 || len(snapshot.Layers[0].TurboQuantPayloads) == 0 {
		t.Fatal("CaptureKVWithOptions did not retain the raw device KV page")
	}
	core.AssertEqual(t, 2, snapshot.Layers[0].CacheIndex)

	restored := &hipEngineSession{cfg: source.cfg, mode: source.mode, driver: source.driver}
	defer func() { _ = restored.Close() }()
	core.RequireNoError(t, restored.RestoreFromKV(context.Background(), snapshot))

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

	core.AssertError(t, err)
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
