// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"math"
	"reflect"
	"testing"
	"unsafe"

	core "dappco.re/go"
	"dappco.re/go/inference/kv"
	"dappco.re/go/inference/model"
	g4 "dappco.re/go/inference/model/gemma4"
)

func sessionStateFixture(t testing.TB) (*BF16Model, model.Arch, int) {
	t.Helper()
	const dModel, nHeads, nKV, headDim, dFF = 512, 8, 4, 64, 1024
	const vocab, nL, maxLen = 64, 3, 96
	layers := make([]DecodeLayerWeights, nL)
	types := make([]string, nL)
	for li := range layers {
		layers[li] = forwardLayer(dModel, nHeads, nKV, headDim, dFF, (li+1)*100)
		types[li] = "full_attention"
	}
	specs := model.DeriveLayers(types, 0)
	embed := toBF16Bytes(syntheticFloat32(vocab*dModel, 21))
	g := &BF16Model{Layers: layers, Embed: embed, FinalNorm: toBF16Bytes(syntheticFloat32(dModel, 22)), LMHead: embed, Tied: true}
	arch := model.Arch{
		Hidden: dModel, Heads: nHeads, KVHeads: nKV, HeadDim: headDim, FF: dFF, Vocab: vocab,
		GlobalHeadDim: headDim, GlobalKVHeads: nKV,
		Eps: 1e-5, AttnScale: 0.125, RopeBase: 10000, RopeScale: 1, RopeLocalBase: 10000,
		RotaryDim: headDim, RotaryDimLocal: headDim, Layer: specs,
	}
	return g, arch, maxLen
}

func newSessionStateFixture(t testing.TB) *ArchSession {
	t.Helper()
	g, arch, maxLen := sessionStateFixture(t)
	s, err := NewArchSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("NewArchSession: %v", err)
	}
	return s
}

func newSingleLayerSessionStateFixture(t testing.TB) *ArchSession {
	t.Helper()
	g, arch, maxLen := sessionStateFixture(t)
	g.Layers = g.Layers[:1]
	arch.Layer = arch.Layer[:1]
	s, err := NewArchSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("NewArchSession(single layer): %v", err)
	}
	return s
}

func restoredStateLayerView(t testing.TB, sess *ArchSession, layer int) sessionStateLayerView {
	t.Helper()
	views, err := sess.stateLayerViews()
	if err != nil {
		t.Fatalf("stateLayerViews: %v", err)
	}
	for _, view := range views {
		if view.layer == layer {
			return view
		}
	}
	t.Fatalf("stateLayerViews missing layer %d", layer)
	return sessionStateLayerView{}
}

func icbSessionStateFixture(t testing.TB) (*QuantModel, model.Arch, int) {
	t.Helper()
	const dModel, nHeads, nKV, headDim, dFF, vocab = 128, 2, 1, 64, 256, 32
	const numLayers, pliDim, gs, bits = 2, 64, 64, 4
	const maxLen = 24
	cfg := g4.Config{
		HiddenSize: dModel, NumHiddenLayers: numLayers, IntermediateSize: dFF,
		NumAttentionHeads: nHeads, NumKeyValueHeads: nKV, HeadDim: headDim,
		VocabSize: vocab, RMSNormEps: 1e-6,
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
	return g, arch, maxLen
}

func newICBSessionStateFixture(t testing.TB, g *QuantModel, arch model.Arch, maxLen int) *ArchSession {
	t.Helper()
	s, err := NewArchQuantSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("NewArchQuantSession: %v", err)
	}
	if s.state.icb == nil {
		t.Fatal("fixture must build an ICB replay session")
	}
	return s
}

func TestLoadStateBlock_Good(t *testing.T) {
	// A full block is a zero-copy range into each fixed cache's row-major slab.
	keys := []byte{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
	values := []byte{10, 11, 12, 13, 14, 15, 16, 17, 18, 19}
	block, err := loadStateBlock(1, 2, 3, 5, []sessionStateLayerView{{
		layer: 3, cacheIndex: 7, cacheMode: nativeStateCacheModeFixed,
		maxSize: 0, kvHeads: 1, headDim: 1, rowBytes: 2, cacheRows: 5,
		keyBytes: keys, valueBytes: values,
	}})
	if err != nil {
		t.Fatalf("loadStateBlock: %v", err)
	}
	if block.Index != 1 || block.TokenStart != 2 || block.TokenCount != 2 || len(block.Layers) != 1 {
		t.Fatalf("loadStateBlock metadata = %#v, want block 1 at tokens [2,4) with one layer", block)
	}
	if got, want := block.Layers[0].KeyBytes, keys[4:8]; !bytes.Equal(got, want) {
		t.Fatalf("loadStateBlock key range = %v, want %v", got, want)
	}
	if got, want := block.Layers[0].ValueBytes, values[4:8]; !bytes.Equal(got, want) {
		t.Fatalf("loadStateBlock value range = %v, want %v", got, want)
	}
}

func TestLoadStateBlock_Bad(t *testing.T) {
	if _, err := loadStateBlock(3, 2, 3, 5, nil); err == nil {
		t.Fatal("loadStateBlock accepted an index beyond the block count")
	}
}

func TestLoadStateBlock_Ugly(t *testing.T) {
	// Position 6 with a four-row ring makes block [3,6) wrap slots 3,0,1.
	// The implementation must materialise that non-contiguous range in token order.
	keys := []byte{0, 1, 2, 3, 4, 5, 6, 7}
	values := []byte{10, 11, 12, 13, 14, 15, 16, 17}
	block, err := loadStateBlock(1, 3, 2, 6, []sessionStateLayerView{{
		layer: 0, cacheIndex: 0, cacheMode: nativeStateCacheModeFixed,
		maxSize: 4, kvHeads: 1, headDim: 1, rowBytes: 2, cacheRows: 4,
		keyBytes: keys, valueBytes: values,
	}})
	if err != nil {
		t.Fatalf("loadStateBlock wrapped range: %v", err)
	}
	if got, want := block.Layers[0].KeyBytes, []byte{6, 7, 0, 1, 2, 3}; !bytes.Equal(got, want) {
		t.Fatalf("loadStateBlock wrapped key range = %v, want %v", got, want)
	}
	if got, want := block.Layers[0].ValueBytes, []byte{16, 17, 10, 11, 12, 13}; !bytes.Equal(got, want) {
		t.Fatalf("loadStateBlock wrapped value range = %v, want %v", got, want)
	}

	targetKeys := make([]byte, len(keys))
	targetValues := make([]byte, len(values))
	target := []sessionStateLayerView{{
		layer: 0, cacheIndex: 0, cacheMode: nativeStateCacheModeFixed,
		maxSize: 4, kvHeads: 1, headDim: 1, rowBytes: 2, cacheRows: 4,
		keyBytes: targetKeys, valueBytes: targetValues,
	}}
	if err := restoreStateBlock(1, 3, 6, 1, target, block); err != nil {
		t.Fatalf("restoreStateBlock wrapped range: %v", err)
	}
	if got, want := targetKeys, keys; !bytes.Equal(got, want) {
		t.Fatalf("restoreStateBlock wrapped keys = %v, want %v", got, want)
	}
	if got, want := targetValues, values; !bytes.Equal(got, want) {
		t.Fatalf("restoreStateBlock wrapped values = %v, want %v", got, want)
	}
}

func TestArchSession_refreshPagedStateLayerViews_Good(t *testing.T) {
	requireNativeRuntime(t)
	s := newSessionStateFixture(t)
	if err := s.PrefillTokens([]int32{1, 2, 3, 4}); err != nil {
		t.Fatalf("PrefillTokens: %v", err)
	}
	views, err := s.stateLayerViews()
	if err != nil {
		t.Fatalf("stateLayerViews: %v", err)
	}
	if len(views) < 2 || views[0].cacheIndex == views[1].cacheIndex {
		t.Fatalf("fixture cache indexes = %v/%v, want distinct paged owners", views[0].cacheIndex, views[1].cacheIndex)
	}
	// Keep a known current snapshot, then replace the cached view slices. The
	// requested cache index must be re-materialised while the unrequested view
	// stays untouched; this is the MTP selective-refresh contract.
	wantKeys := append([]byte(nil), views[0].keyBytes...)
	wantValues := append([]byte(nil), views[0].valueBytes...)
	views[0].keyBytes, views[0].valueBytes = []byte{0xde}, []byte{0xad}
	views[1].keyBytes, views[1].valueBytes = []byte{0xbe}, []byte{0xef}
	if err := s.refreshPagedStateLayerViews(views, map[int]bool{views[0].cacheIndex: true}); err != nil {
		t.Fatalf("refreshPagedStateLayerViews: %v", err)
	}
	if got := views[0].keyBytes; !bytes.Equal(got, wantKeys) {
		t.Fatalf("selected paged key snapshot = %v, want current cache bytes %v", got, wantKeys)
	}
	if got := views[0].valueBytes; !bytes.Equal(got, wantValues) {
		t.Fatalf("selected paged value snapshot = %v, want current cache bytes %v", got, wantValues)
	}
	if got := views[1].keyBytes; !bytes.Equal(got, []byte{0xbe}) {
		t.Fatalf("unrequested paged key snapshot changed to %v", got)
	}
	if got := views[1].valueBytes; !bytes.Equal(got, []byte{0xef}) {
		t.Fatalf("unrequested paged value snapshot changed to %v", got)
	}
}

func firstOwnedCacheLayer(t testing.TB, s *ArchSession) int {
	t.Helper()
	for li, spec := range s.state.specs {
		if spec.OwnsCache() {
			return li
		}
	}
	t.Fatal("fixture has no cache-owning layer")
	return 0
}

func emptySessionStateBlob(pos, layers, cachedIDs int) []byte {
	blob := make([]byte, 12+4+4*cachedIDs)
	binary.LittleEndian.PutUint32(blob[0:], sessionStateMagic)
	binary.LittleEndian.PutUint32(blob[4:], uint32(pos))
	binary.LittleEndian.PutUint32(blob[8:], uint32(layers))
	binary.LittleEndian.PutUint32(blob[12:], uint32(cachedIDs))
	for i := range cachedIDs {
		binary.LittleEndian.PutUint32(blob[16+4*i:], uint32(i+1))
	}
	return blob
}

func TestSessionStateSnapshotCacheViewsUseCachedContentsPointers(t *testing.T) {
	requireNativeRuntime(t)

	t.Run("layer buffers", func(t *testing.T) {
		s := newSessionStateFixture(t)
		// Pins the NON-ICB session's view aliasing (paged snapshot / lb caches); the
		// "icb replay" subtest below covers the ICB views. bf16 sessions record the
		// arch ICB now, so force the lane this subtest is about.
		s.state.icb = nil
		li := firstOwnedCacheLayer(t, s)
		k, v, kPtr, vPtr, err := s.snapshotCacheViews(li)
		if err != nil {
			t.Fatalf("snapshotCacheViews: %v", err)
		}
		// session lb KV materialises lazily (ensureLBKVCaches at the state-view seam);
		// after the first view the contents pointers must be CACHED — never re-derived
		// per call (the paged branch below never touches lb, so only assert on the lb lane).
		if s.state.layerPagedKV(li) == nil && (s.state.lb[li].kCachePtr == nil || s.state.lb[li].vCachePtr == nil) {
			t.Fatal("layer KV cache contents pointers were not cached at materialisation")
		}
		if cache := s.state.layerPagedKV(li); cache != nil {
			if k != cache.snapshotK || v != cache.snapshotV {
				t.Fatal("snapshotCacheViews returned unexpected paged snapshot buffers")
			}
			if kPtr != (*byte)(k.Contents()) || vPtr != (*byte)(v.Contents()) {
				t.Fatal("paged snapshot pointers do not reference Metal buffer contents")
			}
			return
		}
		if k != s.state.lb[li].kCache || v != s.state.lb[li].vCache {
			t.Fatal("snapshotCacheViews returned unexpected layer cache buffers")
		}
		if kPtr != s.state.lb[li].kCachePtr || vPtr != s.state.lb[li].vCachePtr {
			t.Fatal("snapshotCacheViews did not return cached layer cache pointers")
		}
		if kPtr != (*byte)(k.Contents()) || vPtr != (*byte)(v.Contents()) {
			t.Fatal("cached layer cache pointers do not reference Metal buffer contents")
		}
	})

	t.Run("icb replay", func(t *testing.T) {
		g, arch, maxLen := icbSessionStateFixture(t)
		s := newICBSessionStateFixture(t, g, arch, maxLen)
		li := firstOwnedCacheLayer(t, s)
		if len(s.state.icb.kCachePtrs) != len(s.state.icb.kCaches) || len(s.state.icb.vCachePtrs) != len(s.state.icb.vCaches) {
			t.Fatal("ICB KV cache pointer slices do not match cache slices")
		}
		if s.state.icb.kCachePtrs[li] == nil || s.state.icb.vCachePtrs[li] == nil {
			t.Fatal("ICB KV cache contents pointers were not cached at record time")
		}
		k, v, kPtr, vPtr, err := s.snapshotCacheViews(li)
		if err != nil {
			t.Fatalf("snapshotCacheViews ICB: %v", err)
		}
		if k != s.state.icb.kCaches[li] || v != s.state.icb.vCaches[li] {
			t.Fatal("snapshotCacheViews returned unexpected ICB cache buffers")
		}
		if kPtr != s.state.icb.kCachePtrs[li] || vPtr != s.state.icb.vCachePtrs[li] {
			t.Fatal("snapshotCacheViews did not return cached ICB cache pointers")
		}
		if kPtr != (*byte)(k.Contents()) || vPtr != (*byte)(v.Contents()) {
			t.Fatal("cached ICB cache pointers do not reference Metal buffer contents")
		}
	})
}

func TestSessionStateBlocksRestoreGenerateFromCacheBoundary(t *testing.T) {
	requireNativeRuntime(t)
	prompt := []int32{1, 2, 3, 4, 5}

	saved := newSessionStateFixture(t)
	if err := saved.PrefillTokens(prompt); err != nil {
		t.Fatalf("PrefillTokens: %v", err)
	}
	source, err := saved.StateBlockSource(2)
	if err != nil {
		t.Fatalf("StateBlockSource: %v", err)
	}
	if source.Position != saved.Pos() {
		t.Fatalf("block source position = %d, want %d", source.Position, saved.Pos())
	}
	if source.BlockCount != 3 {
		t.Fatalf("block source count = %d, want 3", source.BlockCount)
	}
	if !idsEqual(source.CachedIDs, prompt) {
		t.Fatalf("block source cached ids = %v, want %v", source.CachedIDs, prompt)
	}

	rangedTokens := 0
	if err := saved.RangeStateBlocks(2, func(block SessionStateBlock) (bool, error) {
		if block.TokenCount <= 0 {
			t.Fatalf("block %d token count = %d, want > 0", block.Index, block.TokenCount)
		}
		if len(block.Layers) == 0 {
			t.Fatalf("block %d has no layer payloads", block.Index)
		}
		for _, layer := range block.Layers {
			wantBytes := layer.RowBytes * block.TokenCount
			if len(layer.KeyBytes) != wantBytes || len(layer.ValueBytes) != wantBytes {
				t.Fatalf("block %d layer %d bytes = %d/%d, want %d", block.Index, layer.Layer, len(layer.KeyBytes), len(layer.ValueBytes), wantBytes)
			}
		}
		rangedTokens += block.TokenCount
		return true, nil
	}); err != nil {
		t.Fatalf("RangeStateBlocks: %v", err)
	}
	if rangedTokens != saved.Pos() {
		t.Fatalf("ranged tokens = %d, want %d", rangedTokens, saved.Pos())
	}

	restored := newSessionStateFixture(t)
	if err := restored.RestoreStateBlocks(source); err != nil {
		t.Fatalf("RestoreStateBlocks: %v", err)
	}
	if restored.Pos() != saved.Pos() {
		t.Fatalf("restored pos = %d, want %d", restored.Pos(), saved.Pos())
	}
	if !idsEqual(restored.cachedIDs, prompt) {
		t.Fatalf("restored cached ids = %v, want %v", restored.cachedIDs, prompt)
	}
	got, err := restored.GenerateFromCache(3, -1)
	if err != nil {
		t.Fatalf("GenerateFromCache after block restore: %v", err)
	}
	cold := newSessionStateFixture(t)
	want, err := cold.Generate(prompt, 3, -1)
	if err != nil {
		t.Fatalf("cold Generate: %v", err)
	}
	if !idsEqual(got, want) {
		t.Fatalf("block-restored GenerateFromCache = %v, want cold prompt continuation %v", got, want)
	}
}

func TestSessionStateBlocksRestoreGenerateFromBoundaryLogits(t *testing.T) {
	requireNativeRuntime(t)
	prompt := []int32{1, 2, 3, 4, 5}

	saved := newSessionStateFixture(t)
	if err := saved.PrefillTokens(prompt); err != nil {
		t.Fatalf("PrefillTokens: %v", err)
	}
	logits, err := saved.BoundaryLogits()
	if err != nil {
		t.Fatalf("BoundaryLogits: %v", err)
	}
	source, err := saved.StateBlockSource(2)
	if err != nil {
		t.Fatalf("StateBlockSource: %v", err)
	}

	restored := newSessionStateFixture(t)
	if err := restored.RestoreStateBlocks(source); err != nil {
		t.Fatalf("RestoreStateBlocks: %v", err)
	}
	restored.resetRetainedHidden()
	got, err := restored.GenerateFromCacheLogitsEach(logits, 3, -1, nil)
	if err != nil {
		t.Fatalf("GenerateFromCacheLogitsEach after block restore: %v", err)
	}
	cold := newSessionStateFixture(t)
	want, err := cold.Generate(prompt, 3, -1)
	if err != nil {
		t.Fatalf("cold Generate: %v", err)
	}
	if !idsEqual(got, want) {
		t.Fatalf("logit-restored GenerateFromCache = %v, want cold prompt continuation %v", got, want)
	}
	if restored.Pos() != len(prompt)+len(got) {
		t.Fatalf("restored pos after logit continuation = %d, want %d", restored.Pos(), len(prompt)+len(got))
	}
}

func TestSessionStateBlocksRestoreReloadsDevicePagedKV(t *testing.T) {
	requireNativeRuntime(t)
	prompt := []int32{1, 2, 3, 4, 5}

	saved := newSessionStateFixture(t)
	if err := saved.PrefillTokens(prompt); err != nil {
		t.Fatalf("PrefillTokens: %v", err)
	}
	source, err := saved.StateBlockSource(2)
	if err != nil {
		t.Fatalf("StateBlockSource: %v", err)
	}
	restored := newSessionStateFixture(t)
	if err := restored.RestoreStateBlocks(source); err != nil {
		t.Fatalf("RestoreStateBlocks: %v", err)
	}
	for li, spec := range saved.state.specs {
		if !spec.OwnsCache() || saved.state.layerPagedKV(li) == nil {
			continue
		}
		_, _, savedK, savedV, err := saved.snapshotCacheViews(li)
		if err != nil {
			t.Fatalf("saved snapshotCacheViews L%d: %v", li, err)
		}
		_, _, restoredK, restoredV, err := restored.snapshotCacheViews(li)
		if err != nil {
			t.Fatalf("restored snapshotCacheViews L%d: %v", li, err)
		}
		n := saved.maxLen * kvHeadsOf(spec, saved.arch.KVHeads) * headDimOf(spec, saved.arch.HeadDim) * bf16Size
		eqBytes(t, core.Sprintf("restored block paged K L%d", li), unsafe.Slice(restoredK, n), unsafe.Slice(savedK, n))
		eqBytes(t, core.Sprintf("restored block paged V L%d", li), unsafe.Slice(restoredV, n), unsafe.Slice(savedV, n))
	}
}

func TestSessionStateBlocksRestoreGenerateSampledFromBoundaryLogits(t *testing.T) {
	requireNativeRuntime(t)
	prompt := []int32{1, 2, 3, 4, 5}
	params := model.SampleParams{Temperature: 0.8, TopK: 4, TopP: 0.9, MinTokensBeforeStop: 1}
	stopTokens := []int32{63}
	const seed = 0x5eed1234

	saved := newSessionStateFixture(t)
	if err := saved.PrefillTokens(prompt); err != nil {
		t.Fatalf("PrefillTokens: %v", err)
	}
	logits, err := saved.BoundaryLogits()
	if err != nil {
		t.Fatalf("BoundaryLogits: %v", err)
	}
	source, err := saved.StateBlockSource(2)
	if err != nil {
		t.Fatalf("StateBlockSource: %v", err)
	}

	restored := newSessionStateFixture(t)
	if err := restored.RestoreStateBlocks(source); err != nil {
		t.Fatalf("RestoreStateBlocks: %v", err)
	}
	restored.resetRetainedHidden()
	got, err := restored.GenerateSampledFromCacheLogitsEach(logits, 3, stopTokens, model.NewSampler(seed), params, nil, nil)
	if err != nil {
		t.Fatalf("GenerateSampledFromCacheLogitsEach after block restore: %v", err)
	}
	cold := newSessionStateFixture(t)
	want, err := cold.GenerateSampledEach(prompt, 3, stopTokens, model.NewSampler(seed), params, nil, nil)
	if err != nil {
		t.Fatalf("cold GenerateSampledEach: %v", err)
	}
	if !idsEqual(got, want) {
		t.Fatalf("sampled logit-restored GenerateFromCache = %v, want cold prompt continuation %v", got, want)
	}
	if restored.Pos() != len(prompt)+len(got) {
		t.Fatalf("restored pos after sampled logit continuation = %d, want %d", restored.Pos(), len(prompt)+len(got))
	}
}

func TestSessionStateBlocksRestoreGenerateSampledFromRetainedHidden(t *testing.T) {
	requireNativeRuntime(t)
	prompt := []int32{1, 2, 3, 4, 5}
	params := model.SampleParams{Temperature: 0.8, TopK: 4, TopP: 0.9, MinTokensBeforeStop: 1}
	stopTokens := []int32{63}
	const seed = 0x5eed1234

	saved := newSessionStateFixture(t)
	if err := saved.PrefillTokens(prompt); err != nil {
		t.Fatalf("PrefillTokens: %v", err)
	}
	source, err := saved.StateBlockSource(2)
	if err != nil {
		t.Fatalf("StateBlockSource: %v", err)
	}
	if len(source.RetainedHidden) == 0 {
		t.Fatal("StateBlockSource retained hidden is empty")
	}

	restored := newSessionStateFixture(t)
	if err := restored.RestoreStateBlocks(source); err != nil {
		t.Fatalf("RestoreStateBlocks: %v", err)
	}
	got, err := restored.GenerateSampledFromCacheEach(3, stopTokens, model.NewSampler(seed), params, nil, nil)
	if err != nil {
		t.Fatalf("GenerateSampledFromCacheEach after block restore: %v", err)
	}
	cold := newSessionStateFixture(t)
	want, err := cold.GenerateSampledEach(prompt, 3, stopTokens, model.NewSampler(seed), params, nil, nil)
	if err != nil {
		t.Fatalf("cold GenerateSampledEach: %v", err)
	}
	if !idsEqual(got, want) {
		t.Fatalf("sampled retained-hidden GenerateFromCache = %v, want cold prompt continuation %v", got, want)
	}
	if restored.Pos() != len(prompt)+len(got) {
		t.Fatalf("restored pos after sampled retained-hidden continuation = %d, want %d", restored.Pos(), len(prompt)+len(got))
	}
}

func TestSessionStateBlocksRestoreGenerateSampledFromRetainedLogits(t *testing.T) {
	requireNativeRuntime(t)
	prompt := []int32{1, 2, 3, 4, 5}
	params := model.SampleParams{Temperature: 0.8, TopK: 4, TopP: 0.9, MinTokensBeforeStop: 1}
	stopTokens := []int32{63}
	const seed = 0x5eed1234

	saved := newSessionStateFixture(t)
	if err := saved.PrefillTokens(prompt); err != nil {
		t.Fatalf("PrefillTokens: %v", err)
	}
	logits, err := saved.BoundaryLogits()
	if err != nil {
		t.Fatalf("BoundaryLogits: %v", err)
	}
	source, err := saved.StateBlockSource(2)
	if err != nil {
		t.Fatalf("StateBlockSource: %v", err)
	}
	if !bytes.Equal(source.RetainedLogits, logits) {
		t.Fatal("StateBlockSource did not carry retained boundary logits")
	}
	source.RetainedHidden = nil

	restored := newSessionStateFixture(t)
	if err := restored.RestoreStateBlocks(source); err != nil {
		t.Fatalf("RestoreStateBlocks: %v", err)
	}
	got, err := restored.GenerateSampledFromCacheEach(3, stopTokens, model.NewSampler(seed), params, nil, nil)
	if err != nil {
		t.Fatalf("GenerateSampledFromCacheEach after logit-only block restore: %v", err)
	}
	cold := newSessionStateFixture(t)
	want, err := cold.GenerateSampledEach(prompt, 3, stopTokens, model.NewSampler(seed), params, nil, nil)
	if err != nil {
		t.Fatalf("cold GenerateSampledEach: %v", err)
	}
	if !idsEqual(got, want) {
		t.Fatalf("sampled retained-logit GenerateFromCache = %v, want cold prompt continuation %v", got, want)
	}
}

func TestSessionStateBlocksGenerateSampledFromRetainedHiddenAllocationBudget(t *testing.T) {
	requireNativeRuntime(t)
	prompt := []int32{1, 2, 3, 4, 5}
	params := model.SampleParams{Temperature: 0.8, TopK: 4, TopP: 0.9}
	stopTokens := []int32{63}

	saved := newSessionStateFixture(t)
	if err := saved.PrefillTokens(prompt); err != nil {
		t.Fatalf("PrefillTokens: %v", err)
	}
	source, err := saved.StateBlockSource(2)
	if err != nil {
		t.Fatalf("StateBlockSource: %v", err)
	}
	restored := newSessionStateFixture(t)
	if err := restored.RestoreStateBlocks(source); err != nil {
		t.Fatalf("RestoreStateBlocks warmup: %v", err)
	}
	if _, err := restored.GenerateSampledFromCacheEach(1, stopTokens, model.NewSampler(1), params, nil, nil); err != nil {
		t.Fatalf("GenerateSampledFromCacheEach warmup: %v", err)
	}

	seed := uint64(10)
	allocs := testing.AllocsPerRun(5, func() {
		if err := restored.RestoreStateBlocks(source); err != nil {
			t.Fatalf("RestoreStateBlocks: %v", err)
		}
		seed++
		if _, err := restored.GenerateSampledFromCacheEach(1, stopTokens, model.NewSampler(seed), params, nil, nil); err != nil {
			t.Fatalf("GenerateSampledFromCacheEach: %v", err)
		}
	})
	if allocs > 120 {
		t.Fatalf("restored retained-hidden sampled wake allocations = %.0f, want <= 120", allocs)
	}
}

func TestSessionStateBlocksGenerateSampledFromRetainedLogitsAllocationBudget(t *testing.T) {
	requireNativeRuntime(t)
	prompt := []int32{1, 2, 3, 4, 5}
	params := model.SampleParams{Temperature: 0.8, TopK: 4, TopP: 0.9}
	stopTokens := []int32{63}

	saved := newSessionStateFixture(t)
	if err := saved.PrefillTokens(prompt); err != nil {
		t.Fatalf("PrefillTokens: %v", err)
	}
	source, err := saved.StateBlockSource(2)
	if err != nil {
		t.Fatalf("StateBlockSource: %v", err)
	}
	if len(source.RetainedLogits) == 0 {
		t.Fatal("StateBlockSource did not retain boundary logits")
	}
	source.RetainedHidden = nil
	restored := newSessionStateFixture(t)
	if err := restored.RestoreStateBlocks(source); err != nil {
		t.Fatalf("RestoreStateBlocks warmup: %v", err)
	}
	if _, err := restored.GenerateSampledFromCacheEach(1, stopTokens, model.NewSampler(1), params, nil, nil); err != nil {
		t.Fatalf("GenerateSampledFromCacheEach warmup: %v", err)
	}

	seed := uint64(20)
	allocs := testing.AllocsPerRun(5, func() {
		if err := restored.RestoreStateBlocks(source); err != nil {
			t.Fatalf("RestoreStateBlocks: %v", err)
		}
		seed++
		if _, err := restored.GenerateSampledFromCacheEach(1, stopTokens, model.NewSampler(seed), params, nil, nil); err != nil {
			t.Fatalf("GenerateSampledFromCacheEach: %v", err)
		}
	})
	if allocs > 20 {
		t.Fatalf("restored retained-logit sampled wake allocations = %.0f, want <= 20", allocs)
	}
}

func TestSessionStateBlocksGenerateSampledFromRetainedHiddenTopPOnlyAllocationBudget(t *testing.T) {
	requireNativeRuntime(t)
	prompt := []int32{1, 2, 3, 4, 5}
	params := model.SampleParams{Temperature: 1, TopP: 0.72}
	stopTokens := []int32{63}

	saved := newSessionStateFixture(t)
	if err := saved.PrefillTokens(prompt); err != nil {
		t.Fatalf("PrefillTokens: %v", err)
	}
	source, err := saved.StateBlockSource(2)
	if err != nil {
		t.Fatalf("StateBlockSource: %v", err)
	}
	source.RetainedLogits = nil
	restored := newSessionStateFixture(t)
	if err := restored.RestoreStateBlocks(source); err != nil {
		t.Fatalf("RestoreStateBlocks warmup: %v", err)
	}
	if _, err := restored.GenerateSampledFromCacheEach(1, stopTokens, model.NewSampler(1), params, nil, nil); err != nil {
		t.Fatalf("GenerateSampledFromCacheEach warmup: %v", err)
	}

	seed := uint64(30)
	allocs := testing.AllocsPerRun(5, func() {
		if err := restored.RestoreStateBlocks(source); err != nil {
			t.Fatalf("RestoreStateBlocks: %v", err)
		}
		seed++
		if _, err := restored.GenerateSampledFromCacheEach(1, stopTokens, model.NewSampler(seed), params, nil, nil); err != nil {
			t.Fatalf("GenerateSampledFromCacheEach: %v", err)
		}
	})
	if allocs > 25 {
		t.Fatalf("restored retained-hidden TopP-only sampled wake allocations = %.0f, want <= 25", allocs)
	}
}

func TestSessionStateBlocksGenerateSampledFromRetainedLogitsTopPOnlyLargeVocabAllocationBudget(t *testing.T) {
	requireNativeRuntime(t)
	const dModel, nHeads, nKV, headDim, dFF, vocab = 128, 2, 1, 64, 256, 128
	const maxLen = 24
	g, arch := gemma4BF16Fixture(t, dModel, nHeads, nKV, headDim, dFF, vocab, 2)
	saved, err := NewArchSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("NewArchSession saved: %v", err)
	}
	prompt := []int32{1, 2, 3, 4, 5}
	params := model.SampleParams{Temperature: 1, TopP: 0.72}
	stopTokens := []int32{int32(vocab - 1)}
	if err := saved.PrefillTokens(prompt); err != nil {
		t.Fatalf("PrefillTokens: %v", err)
	}
	source, err := saved.StateBlockSource(2)
	if err != nil {
		t.Fatalf("StateBlockSource: %v", err)
	}
	if len(source.RetainedLogits) == 0 {
		t.Fatal("StateBlockSource did not retain boundary logits")
	}
	source.RetainedHidden = nil
	restored, err := NewArchSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("NewArchSession restored: %v", err)
	}
	if err := restored.RestoreStateBlocks(source); err != nil {
		t.Fatalf("RestoreStateBlocks warmup: %v", err)
	}
	if _, err := restored.GenerateSampledFromCacheEach(1, stopTokens, model.NewSampler(1), params, nil, nil); err != nil {
		t.Fatalf("GenerateSampledFromCacheEach warmup: %v", err)
	}

	const paritySeed = 99
	if err := restored.RestoreStateBlocks(source); err != nil {
		t.Fatalf("RestoreStateBlocks parity: %v", err)
	}
	got, err := restored.GenerateSampledFromCacheEach(1, stopTokens, model.NewSampler(paritySeed), params, nil, nil)
	if err != nil {
		t.Fatalf("GenerateSampledFromCacheEach parity: %v", err)
	}
	cold, err := NewArchSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("NewArchSession cold: %v", err)
	}
	want, err := cold.GenerateSampledEach(prompt, 1, stopTokens, model.NewSampler(paritySeed), params, nil, nil)
	if err != nil {
		t.Fatalf("cold GenerateSampledEach: %v", err)
	}
	if !idsEqual(got, want) {
		t.Fatalf("retained-logit large-vocab TopP-only wake = %v, want cold sampled continuation %v", got, want)
	}

	seed := uint64(40)
	allocs := testing.AllocsPerRun(5, func() {
		if err := restored.RestoreStateBlocks(source); err != nil {
			t.Fatalf("RestoreStateBlocks: %v", err)
		}
		seed++
		if _, err := restored.GenerateSampledFromCacheEach(1, stopTokens, model.NewSampler(seed), params, nil, nil); err != nil {
			t.Fatalf("GenerateSampledFromCacheEach: %v", err)
		}
	})
	if allocs > 12 {
		t.Fatalf("restored retained-logit large-vocab TopP-only sampled wake allocations = %.0f, want <= 12", allocs)
	}
}

func TestArchSessionRetainedHiddenUsesPinnedNoCopyBuffer(t *testing.T) {
	requireNativeRuntime(t)
	sess := newSessionStateFixture(t)
	n := sess.arch.Hidden * bf16Size
	first := toBF16Bytes(syntheticFloat32(sess.arch.Hidden, 31))
	second := toBF16Bytes(syntheticFloat32(sess.arch.Hidden, 32))

	sess.rememberRetainedHidden(first)
	if sess.retainedHiddenPinned == nil || sess.retainedHiddenPinned.buf == nil {
		t.Fatal("retained hidden was not stored in a pinned no-copy buffer")
	}
	if len(sess.retainedHidden) != n || !bytes.Equal(sess.retainedHidden, first) {
		t.Fatal("retained hidden did not preserve first boundary contents")
	}
	if unsafe.Pointer(&sess.retainedHidden[0]) != unsafe.Pointer(&sess.retainedHiddenPinned.bytes[0]) {
		t.Fatal("retained hidden slice does not point at pinned backing bytes")
	}
	buf := sess.retainedHiddenBuffer()
	if buf == nil || buf.GetID() != sess.retainedHiddenPinned.buf.GetID() {
		t.Fatal("retainedHiddenBuffer did not return the session-owned no-copy buffer")
	}
	backing := unsafe.Pointer(&sess.retainedHidden[0])
	bufID := sess.retainedHiddenPinned.buf.GetID()

	sess.rememberRetainedHidden(second)
	if unsafe.Pointer(&sess.retainedHidden[0]) != backing {
		t.Fatal("retained hidden backing changed across same-shape boundary updates")
	}
	if sess.retainedHiddenPinned.buf.GetID() != bufID {
		t.Fatal("retained hidden no-copy buffer changed across same-shape boundary updates")
	}
	if !bytes.Equal(sess.retainedHidden, second) {
		t.Fatal("retained hidden did not refresh second boundary contents")
	}
}

func TestBoundaryLogitsUsesRetainedHiddenNoCopyHeadPath(t *testing.T) {
	requireNativeRuntime(t)
	sess := newSessionStateFixture(t)
	if sess.headEnc == nil {
		t.Fatal("session fixture did not build resident head encoder")
	}
	hidden := toBF16Bytes(syntheticFloat32(sess.arch.Hidden, 37))
	sess.rememberRetainedHidden(hidden)
	if sess.retainedHiddenBuffer() == nil {
		t.Fatal("retained hidden did not expose its pinned no-copy buffer")
	}
	sess.sampleHeadLogits = make([]byte, sess.arch.Vocab*bf16Size)

	head := sess.head
	headCalls := 0
	sess.head = func(hidden []byte, skipSoftcap bool) ([]byte, error) {
		headCalls++
		return head(hidden, skipSoftcap)
	}
	logits, err := sess.BoundaryLogits()
	if err != nil {
		t.Fatalf("BoundaryLogits: %v", err)
	}
	if len(logits) != sess.arch.Vocab*bf16Size {
		t.Fatalf("BoundaryLogits length = %d, want %d", len(logits), sess.arch.Vocab*bf16Size)
	}
	if headCalls != 0 {
		t.Fatalf("BoundaryLogits generic head calls = %d, want retained no-copy head path", headCalls)
	}
	if buf := sess.retainedLogitsBuffer(); buf == nil {
		t.Fatal("BoundaryLogits did not retain logits in a pinned no-copy buffer")
	}
	if len(sess.retainedLogits) == 0 || unsafe.Pointer(&logits[0]) != unsafe.Pointer(&sess.retainedLogits[0]) {
		t.Fatal("BoundaryLogits did not return retained logits backing")
	}
	if cap(sess.sampleHeadLogits) != 0 {
		t.Fatalf("BoundaryLogits retained transient head logits scratch cap = %d, want 0", cap(sess.sampleHeadLogits))
	}
	allocs := testing.AllocsPerRun(10, func() {
		sess.resetRetainedLogits()
		if _, err := sess.BoundaryLogits(); err != nil {
			t.Fatalf("BoundaryLogits allocation run: %v", err)
		}
	})
	if allocs > 1 {
		t.Fatalf("BoundaryLogits retained-hidden no-copy allocations = %.0f, want <= 1", allocs)
	}
}

func TestBoundaryNormedHiddenIntoReusesCallerOutput(t *testing.T) {
	requireNativeRuntime(t)
	sess := newSessionStateFixture(t)
	hidden := toBF16Bytes(syntheticFloat32(sess.arch.Hidden, 39))
	sess.rememberRetainedHidden(hidden)
	want, err := sess.BoundaryNormedHidden()
	if err != nil {
		t.Fatalf("BoundaryNormedHidden: %v", err)
	}
	out := make([]byte, sess.arch.Hidden*bf16Size)

	got, err := sess.boundaryNormedHiddenInto(out)
	if err != nil {
		t.Fatalf("boundaryNormedHiddenInto: %v", err)
	}

	if len(got) == 0 || unsafe.Pointer(&got[0]) != unsafe.Pointer(&out[0]) {
		t.Fatal("boundaryNormedHiddenInto did not reuse caller output backing")
	}
	if !bytes.Equal(got, want) {
		t.Fatal("boundaryNormedHiddenInto output differs from BoundaryNormedHidden")
	}
	allocs := testing.AllocsPerRun(10, func() {
		if _, err := sess.boundaryNormedHiddenInto(out); err != nil {
			t.Fatalf("boundaryNormedHiddenInto allocation run: %v", err)
		}
	})
	if allocs > 1 {
		t.Fatalf("boundaryNormedHiddenInto allocations = %.0f, want <= 1", allocs)
	}
}

func TestArchSessionRetainedLogitsUsesPinnedNoCopyBuffer(t *testing.T) {
	requireNativeRuntime(t)
	sess := newSessionStateFixture(t)
	n := sess.arch.Vocab * bf16Size
	first := toBF16Bytes(syntheticFloat32(sess.arch.Vocab, 41))
	second := toBF16Bytes(syntheticFloat32(sess.arch.Vocab, 42))

	sess.rememberRetainedLogits(first)
	if sess.retainedLogitsPinned == nil || sess.retainedLogitsPinned.buf == nil {
		t.Fatal("retained logits were not stored in a pinned no-copy buffer")
	}
	if len(sess.retainedLogits) != n || !bytes.Equal(sess.retainedLogits, first) {
		t.Fatal("retained logits did not preserve first boundary contents")
	}
	if unsafe.Pointer(&sess.retainedLogits[0]) != unsafe.Pointer(&sess.retainedLogitsPinned.bytes[0]) {
		t.Fatal("retained logits slice does not point at pinned backing bytes")
	}
	buf := sess.retainedLogitsBuffer()
	if buf == nil || buf.GetID() != sess.retainedLogitsPinned.buf.GetID() {
		t.Fatal("retainedLogitsBuffer did not return the session-owned no-copy buffer")
	}
	backing := unsafe.Pointer(&sess.retainedLogits[0])
	bufID := sess.retainedLogitsPinned.buf.GetID()

	sess.rememberRetainedLogits(second)
	if unsafe.Pointer(&sess.retainedLogits[0]) != backing {
		t.Fatal("retained logits backing changed across same-shape boundary updates")
	}
	if sess.retainedLogitsPinned.buf.GetID() != bufID {
		t.Fatal("retained logits no-copy buffer changed across same-shape boundary updates")
	}
	if !bytes.Equal(sess.retainedLogits, second) {
		t.Fatal("retained logits did not refresh second boundary contents")
	}
}

func TestSessionStateBlockSourceBorrowsRetainedBoundaryNoCopy(t *testing.T) {
	requireNativeRuntime(t)
	prompt := []int32{1, 2, 3, 4, 5}

	sess := newSessionStateFixture(t)
	if err := sess.PrefillTokens(prompt); err != nil {
		t.Fatalf("PrefillTokens: %v", err)
	}
	if sess.retainedHiddenBuffer() == nil {
		t.Fatal("prefill did not retain hidden in a pinned no-copy buffer")
	}
	source, err := sess.StateBlockSource(2)
	if err != nil {
		t.Fatalf("StateBlockSource: %v", err)
	}
	if len(source.RetainedHidden) == 0 || len(source.RetainedLogits) == 0 {
		t.Fatalf("source retained boundary lengths = hidden %d logits %d, want both non-empty", len(source.RetainedHidden), len(source.RetainedLogits))
	}
	if unsafe.Pointer(&source.RetainedHidden[0]) != unsafe.Pointer(&sess.retainedHidden[0]) {
		t.Fatal("StateBlockSource copied retained hidden; want borrowed no-copy boundary")
	}
	if unsafe.Pointer(&source.RetainedLogits[0]) != unsafe.Pointer(&sess.retainedLogits[0]) {
		t.Fatal("StateBlockSource copied retained logits; want borrowed no-copy boundary")
	}
}

func TestSessionStateBlockSourceBorrowsCachedPromptBoundaryNoCopy(t *testing.T) {
	requireNativeRuntime(t)
	prompt := []int32{1, 2, 3, 4, 5}

	sess := newSessionStateFixture(t)
	if err := sess.WarmPromptCache(prompt); err != nil {
		t.Fatalf("WarmPromptCache: %v", err)
	}
	if sess.cachedPromptHiddenBuffer() == nil || sess.cachedPromptLogitsBuffer() == nil {
		t.Fatal("warm prompt cache did not retain no-copy boundary buffers")
	}
	source, err := sess.StateBlockSource(2)
	if err != nil {
		t.Fatalf("StateBlockSource: %v", err)
	}
	if len(source.CachedPromptHidden) == 0 || len(source.CachedPromptLogits) == 0 {
		t.Fatalf("source cached prompt boundary lengths = hidden %d logits %d, want both non-empty", len(source.CachedPromptHidden), len(source.CachedPromptLogits))
	}
	if unsafe.Pointer(&source.CachedPromptHidden[0]) != unsafe.Pointer(&sess.cachedPromptHidden[0]) {
		t.Fatal("StateBlockSource copied cached prompt hidden; want borrowed no-copy boundary")
	}
	if unsafe.Pointer(&source.CachedPromptLogits[0]) != unsafe.Pointer(&sess.cachedPromptLogits[0]) {
		t.Fatal("StateBlockSource copied cached prompt logits; want borrowed no-copy boundary")
	}
}

func TestCachedPromptEntryExposesNoCopyBoundaryBuffers(t *testing.T) {
	requireNativeRuntime(t)
	sess := newSessionStateFixture(t)
	prompt := []int32{1, 2, 3}
	hidden := toBF16Bytes(syntheticFloat32(sess.arch.Hidden, 43))
	logits := toBF16Bytes(syntheticFloat32(sess.arch.Vocab, 44))

	sess.rememberCachedPromptEntry(prompt, hidden, logits)
	if !bytes.Equal(sess.cachedPromptHidden, hidden) {
		t.Fatal("cached prompt hidden did not preserve boundary contents")
	}
	if !bytes.Equal(sess.cachedPromptLogits, logits) {
		t.Fatal("cached prompt logits did not preserve boundary contents")
	}
	if buf := sess.retainedHiddenBufferFor(sess.cachedPromptHidden); buf == nil {
		t.Fatal("cached prompt hidden did not expose a no-copy buffer")
	}
	if buf := sess.retainedLogitsBufferFor(sess.cachedPromptLogits); buf == nil {
		t.Fatal("cached prompt logits did not expose a no-copy buffer")
	}
}

func TestCachedPromptEntryAliasesRetainedNoCopyBoundaryBuffers(t *testing.T) {
	requireNativeRuntime(t)
	sess := newSessionStateFixture(t)
	prompt := []int32{1, 2, 3}
	hidden := toBF16Bytes(syntheticFloat32(sess.arch.Hidden, 45))
	logits := toBF16Bytes(syntheticFloat32(sess.arch.Vocab, 46))
	sess.rememberRetainedHidden(hidden)
	sess.rememberRetainedLogits(logits)
	retainedHiddenBuf := sess.retainedHiddenBuffer()
	retainedLogitsBuf := sess.retainedLogitsBuffer()
	retainedHiddenPinned := sess.retainedHiddenPinned
	retainedLogitsPinned := sess.retainedLogitsPinned
	if retainedHiddenBuf == nil || retainedLogitsBuf == nil {
		t.Fatal("retained boundary buffers are not pinned no-copy")
	}

	sess.rememberCachedPromptEntry(prompt, sess.retainedHidden, sess.retainedLogits)
	if len(sess.cachedPromptHidden) == 0 || unsafe.Pointer(&sess.cachedPromptHidden[0]) != unsafe.Pointer(&sess.retainedHidden[0]) {
		t.Fatal("cached prompt hidden did not alias retained no-copy hidden")
	}
	if len(sess.cachedPromptLogits) == 0 || unsafe.Pointer(&sess.cachedPromptLogits[0]) != unsafe.Pointer(&sess.retainedLogits[0]) {
		t.Fatal("cached prompt logits did not alias retained no-copy logits")
	}
	if sess.retainedHiddenBufferFor(sess.cachedPromptHidden) != retainedHiddenBuf {
		t.Fatal("cached prompt hidden did not reuse retained hidden no-copy buffer")
	}
	if sess.retainedLogitsBufferFor(sess.cachedPromptLogits) != retainedLogitsBuf {
		t.Fatal("cached prompt logits did not reuse retained logits no-copy buffer")
	}
	if sess.cachedPromptHiddenPinned != retainedHiddenPinned || sess.cachedPromptLogitsPinned != retainedLogitsPinned {
		t.Fatal("cached prompt did not share retained no-copy buffers")
	}
	if sess.retainedHiddenPinned != retainedHiddenPinned || sess.retainedLogitsPinned != retainedLogitsPinned {
		t.Fatal("retained no-copy owners were not preserved for current boundary reuse")
	}
	cachedHidden := append([]byte(nil), sess.cachedPromptHidden...)
	cachedLogits := append([]byte(nil), sess.cachedPromptLogits...)
	nextHidden := toBF16Bytes(syntheticFloat32(sess.arch.Hidden, 47))
	nextLogits := toBF16Bytes(syntheticFloat32(sess.arch.Vocab, 48))
	sess.rememberRetainedHidden(nextHidden)
	sess.rememberRetainedLogits(nextLogits)
	if !bytes.Equal(sess.cachedPromptHidden, cachedHidden) {
		t.Fatal("retained hidden refresh mutated cached prompt hidden")
	}
	if !bytes.Equal(sess.cachedPromptLogits, cachedLogits) {
		t.Fatal("retained logits refresh mutated cached prompt logits")
	}
	if len(sess.retainedHidden) == 0 || unsafe.Pointer(&sess.retainedHidden[0]) == unsafe.Pointer(&sess.cachedPromptHidden[0]) {
		t.Fatal("retained hidden did not detach from cached prompt hidden")
	}
	if len(sess.retainedLogits) == 0 || unsafe.Pointer(&sess.retainedLogits[0]) == unsafe.Pointer(&sess.cachedPromptLogits[0]) {
		t.Fatal("retained logits did not detach from cached prompt logits")
	}
}

func TestSessionStateRangeBlocksSkipsTrustedPrefix(t *testing.T) {
	requireNativeRuntime(t)
	prompt := []int32{1, 2, 3, 4, 5, 6, 7}

	saved := newSessionStateFixture(t)
	if err := saved.PrefillTokens(prompt); err != nil {
		t.Fatalf("PrefillTokens: %v", err)
	}
	var got []SessionStateBlock
	if err := saved.RangeStateBlocksFrom(4, 2, func(block SessionStateBlock) (bool, error) {
		got = append(got, block)
		return true, nil
	}); err != nil {
		t.Fatalf("RangeStateBlocksFrom: %v", err)
	}
	if len(got) != 2 {
		t.Fatalf("ranged block count = %d, want 2", len(got))
	}
	if got[0].Index != 2 || got[0].TokenStart != 4 || got[0].TokenCount != 2 {
		t.Fatalf("first yielded block = index %d start %d count %d, want index 2 start 4 count 2", got[0].Index, got[0].TokenStart, got[0].TokenCount)
	}
	if got[1].Index != 3 || got[1].TokenStart != 6 || got[1].TokenCount != 1 {
		t.Fatalf("second yielded block = index %d start %d count %d, want index 3 start 6 count 1", got[1].Index, got[1].TokenStart, got[1].TokenCount)
	}

	source, err := saved.StateBlockSourceFrom(4, 2)
	if err != nil {
		t.Fatalf("StateBlockSourceFrom: %v", err)
	}
	if source.BlockCount != len(got) {
		t.Fatalf("source block count = %d, want %d", source.BlockCount, len(got))
	}
	for i := 0; i < source.BlockCount; i++ {
		block, err := source.Load(i)
		if err != nil {
			t.Fatalf("source.Load(%d): %v", i, err)
		}
		if block.Index != got[i].Index || block.TokenStart != got[i].TokenStart || block.TokenCount != got[i].TokenCount {
			t.Fatalf("source block %d = index %d start %d count %d, want index %d start %d count %d", i, block.Index, block.TokenStart, block.TokenCount, got[i].Index, got[i].TokenStart, got[i].TokenCount)
		}
	}
}

func TestSessionStateBlockSourceTrustPrefixBlocks(t *testing.T) {
	source := SessionStateBlockSource{Position: 7, BlockCount: 2}
	if err := source.TrustPrefixBlocks(2, 2); err != nil {
		t.Fatalf("TrustPrefixBlocks: %v", err)
	}
	if got := source.trustedPrefixTokens(); got != 4 {
		t.Fatalf("trusted prefix = %d, want 4", got)
	}
	if source.firstBlockIndex != 2 || source.totalBlockCount != 4 {
		t.Fatalf("block grid = first %d total %d, want 2/4", source.firstBlockIndex, source.totalBlockCount)
	}
	if err := source.TrustPrefixBlocks(2, 0); err != nil {
		t.Fatalf("TrustPrefixBlocks reset: %v", err)
	}
	if got := source.trustedPrefixTokens(); got != 0 {
		t.Fatalf("trusted prefix after reset = %d, want 0", got)
	}
	if err := source.TrustPrefixBlocks(0, 1); err == nil {
		t.Fatal("TrustPrefixBlocks zero block size error = nil")
	}
	if err := source.TrustPrefixBlocks(4, 3); err == nil {
		t.Fatal("TrustPrefixBlocks oversized prefix error = nil")
	}
}

func TestSessionStateBlockSourceTrustPrefixTokens(t *testing.T) {
	source := SessionStateBlockSource{Position: 7, BlockCount: 2}
	if err := source.TrustPrefixTokens(3, 2); err != nil {
		t.Fatalf("TrustPrefixTokens: %v", err)
	}
	if got := source.trustedPrefixTokens(); got != 3 {
		t.Fatalf("trusted token prefix = %d, want 3", got)
	}
	if source.firstBlockIndex != 2 || source.totalBlockCount != 4 {
		t.Fatalf("block grid = first %d total %d, want 2/4", source.firstBlockIndex, source.totalBlockCount)
	}
	if err := source.TrustPrefixTokens(0, 0); err != nil {
		t.Fatalf("TrustPrefixTokens reset: %v", err)
	}
	if got := source.trustedPrefixTokens(); got != 0 {
		t.Fatalf("trusted token prefix after reset = %d, want 0", got)
	}
	if err := source.TrustPrefixTokens(-1, 1); err == nil {
		t.Fatal("TrustPrefixTokens negative prefix error = nil")
	}
	if err := source.TrustPrefixTokens(8, 1); err == nil {
		t.Fatal("TrustPrefixTokens oversized prefix error = nil")
	}
	if err := source.TrustPrefixTokens(3, 0); err == nil {
		t.Fatal("TrustPrefixTokens positive prefix with zero first block error = nil")
	}
}

func TestSessionStateBlockSourceCarriesSlidingCacheMetadata(t *testing.T) {
	requireNativeRuntime(t)
	g, arch, maxLen := sessionStateFixture(t)
	arch.SlidingWindow = 4
	arch.Layer[0].Attention = model.SlidingAttention
	sess, err := NewArchSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("NewArchSession: %v", err)
	}
	if err := sess.PrefillTokens([]int32{1, 2, 3}); err != nil {
		t.Fatalf("PrefillTokens: %v", err)
	}
	source, err := sess.StateBlockSource(2)
	if err != nil {
		t.Fatalf("StateBlockSource: %v", err)
	}
	block, err := source.Load(0)
	if err != nil {
		t.Fatalf("Load(0): %v", err)
	}
	if len(block.Layers) == 0 {
		t.Fatal("state block has no layers")
	}
	layer := block.Layers[0]
	if layer.Layer != 0 {
		t.Fatalf("first layer = %d, want sliding layer 0", layer.Layer)
	}
	if layer.CacheMode != "fixed" || layer.MaxSize != arch.SlidingWindow {
		t.Fatalf("sliding layer cache metadata = %q/%d, want fixed/%d", layer.CacheMode, layer.MaxSize, arch.SlidingWindow)
	}
}

func TestSessionStateBlockSourceSplitsSlidingWindowBoundary(t *testing.T) {
	requireNativeRuntime(t)
	g, arch, maxLen := sessionStateFixture(t)
	arch.SlidingWindow = 4
	arch.Layer[0].Attention = model.SlidingAttention
	sess, err := NewArchSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("NewArchSession: %v", err)
	}
	if err := sess.PrefillTokens([]int32{1, 2, 3, 4, 5, 6, 7}); err != nil {
		t.Fatalf("PrefillTokens: %v", err)
	}
	source, err := sess.StateBlockSource(2)
	if err != nil {
		t.Fatalf("StateBlockSource: %v", err)
	}
	want := []struct {
		index int
		start int
		count int
	}{
		{0, 0, 2},
		{1, 2, 1},
		{2, 3, 1},
		{3, 4, 2},
		{4, 6, 1},
	}
	if source.BlockCount != len(want) {
		t.Fatalf("block count = %d, want %d", source.BlockCount, len(want))
	}
	for i, w := range want {
		block, err := source.Load(i)
		if err != nil {
			t.Fatalf("Load(%d): %v", i, err)
		}
		if block.Index != w.index || block.TokenStart != w.start || block.TokenCount != w.count {
			t.Fatalf("block %d = index %d start %d count %d, want index %d start %d count %d", i, block.Index, block.TokenStart, block.TokenCount, w.index, w.start, w.count)
		}
	}
}

func TestSessionStateFillBlockMapsSlidingRingRows(t *testing.T) {
	keyRows := []byte{
		4, 0,
		5, 0,
		2, 0,
		3, 0,
	}
	valueRows := []byte{
		14, 0,
		15, 0,
		12, 0,
		13, 0,
	}
	layers := make([]SessionStateLayerBlock, 1)
	block, err := fillStateBlock(2, 2, 3, 6, []sessionStateLayerView{{
		layer:      0,
		cacheIndex: 0,
		cacheMode:  nativeStateCacheModeFixed,
		maxSize:    4,
		cacheRows:  4,
		kvHeads:    1,
		headDim:    1,
		rowBytes:   2,
		keyBytes:   keyRows,
		valueBytes: valueRows,
	}}, layers)
	if err != nil {
		t.Fatalf("fillStateBlock sliding ring: %v", err)
	}
	if block.TokenStart != 4 || block.TokenCount != 2 {
		t.Fatalf("block range = %d/%d, want 4/2", block.TokenStart, block.TokenCount)
	}
	if !bytes.Equal(block.Layers[0].KeyBytes, []byte{4, 0, 5, 0}) {
		t.Fatalf("sliding key rows = %v, want logical rows 4,5", block.Layers[0].KeyBytes)
	}
	if !bytes.Equal(block.Layers[0].ValueBytes, []byte{14, 0, 15, 0}) {
		t.Fatalf("sliding value rows = %v, want logical rows 4,5", block.Layers[0].ValueBytes)
	}
	if unsafe.Pointer(&block.Layers[0].KeyBytes[0]) != unsafe.Pointer(&keyRows[0]) {
		t.Fatal("contiguous sliding key rows were copied instead of returned as a resident view")
	}
	if unsafe.Pointer(&block.Layers[0].ValueBytes[0]) != unsafe.Pointer(&valueRows[0]) {
		t.Fatal("contiguous sliding value rows were copied instead of returned as a resident view")
	}
}

func TestSessionStateBlockBoundariesSplitSlidingRingWrap(t *testing.T) {
	sess := &ArchSession{}
	got := append([]int(nil), sess.stateBlockBoundaries(3, 10, []sessionStateLayerView{{
		maxSize:   4,
		cacheRows: 4,
	}})...)
	want := []int{0, 3, 6, 8, 9, 10}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("sliding boundaries = %v, want %v", got, want)
	}
}

func TestSessionStateFillBlockOmitsExpiredSlidingRows(t *testing.T) {
	layers := make([]SessionStateLayerBlock, 1)
	block, err := fillStateBlock(0, 2, 3, 6, []sessionStateLayerView{{
		layer:      0,
		cacheIndex: 0,
		cacheMode:  nativeStateCacheModeFixed,
		maxSize:    4,
		cacheRows:  4,
		kvHeads:    1,
		headDim:    1,
		rowBytes:   2,
		keyBytes:   []byte{4, 0, 5, 0, 2, 0, 3, 0},
		valueBytes: []byte{14, 0, 15, 0, 12, 0, 13, 0},
	}}, layers)
	if err != nil {
		t.Fatalf("fillStateBlock expired sliding rows: %v", err)
	}
	if block.TokenStart != 0 || block.TokenCount != 2 {
		t.Fatalf("block range = %d/%d, want 0/2", block.TokenStart, block.TokenCount)
	}
	if len(block.Layers) != 1 || block.Layers[0].Layer != 0 {
		t.Fatalf("block layers = %+v, want metadata-only sliding layer", block.Layers)
	}
	if len(block.Layers[0].KeyBytes) != 0 || len(block.Layers[0].ValueBytes) != 0 {
		t.Fatalf("expired sliding rows carried KV bytes key=%v value=%v, want metadata-only", block.Layers[0].KeyBytes, block.Layers[0].ValueBytes)
	}
	if block.Layers[0].CacheMode != nativeStateCacheModeFixed || block.Layers[0].MaxSize != 4 {
		t.Fatalf("expired sliding metadata = %q/%d, want fixed/4", block.Layers[0].CacheMode, block.Layers[0].MaxSize)
	}
}

func TestSessionStateRestoreBlockMapsSlidingRingRows(t *testing.T) {
	keyRows := make([]byte, 8)
	valueRows := make([]byte, 8)
	err := restoreStateBlock(2, 4, 6, 1, []sessionStateLayerView{{
		layer:      0,
		cacheIndex: 0,
		cacheMode:  nativeStateCacheModeFixed,
		maxSize:    4,
		cacheRows:  4,
		kvHeads:    1,
		headDim:    1,
		rowBytes:   2,
		keyBytes:   keyRows,
		valueBytes: valueRows,
	}}, SessionStateBlock{
		Index:      2,
		TokenStart: 4,
		TokenCount: 2,
		Layers: []SessionStateLayerBlock{{
			Layer:      0,
			CacheIndex: 0,
			CacheMode:  nativeStateCacheModeFixed,
			MaxSize:    4,
			KVHeads:    1,
			HeadDim:    1,
			RowBytes:   2,
			KeyBytes:   []byte{4, 0, 5, 0},
			ValueBytes: []byte{14, 0, 15, 0},
		}},
	})
	if err != nil {
		t.Fatalf("restoreStateBlock sliding ring: %v", err)
	}
	if !bytes.Equal(keyRows[:4], []byte{4, 0, 5, 0}) {
		t.Fatalf("restored key ring prefix = %v, want logical rows 4,5 in slots 0,1", keyRows)
	}
	if !bytes.Equal(valueRows[:4], []byte{14, 0, 15, 0}) {
		t.Fatalf("restored value ring prefix = %v, want logical rows 4,5 in slots 0,1", valueRows)
	}
}

func TestSessionStateRestoreBlockSkipsExpiredSlidingRows(t *testing.T) {
	keyRows := []byte{4, 0, 5, 0, 2, 0, 3, 0}
	valueRows := []byte{14, 0, 15, 0, 12, 0, 13, 0}
	origKey := append([]byte(nil), keyRows...)
	origValue := append([]byte(nil), valueRows...)
	err := restoreStateBlock(0, 0, 6, 1, []sessionStateLayerView{{
		layer:      0,
		cacheIndex: 0,
		cacheMode:  nativeStateCacheModeFixed,
		maxSize:    4,
		cacheRows:  4,
		kvHeads:    1,
		headDim:    1,
		rowBytes:   2,
		keyBytes:   keyRows,
		valueBytes: valueRows,
	}}, SessionStateBlock{
		Index:      0,
		TokenStart: 0,
		TokenCount: 2,
		Layers: []SessionStateLayerBlock{{
			Layer:      0,
			CacheIndex: 0,
			CacheMode:  nativeStateCacheModeFixed,
			MaxSize:    4,
			KVHeads:    1,
			HeadDim:    1,
			RowBytes:   2,
		}},
	})
	if err != nil {
		t.Fatalf("restoreStateBlock expired sliding rows: %v", err)
	}
	if !bytes.Equal(keyRows, origKey) || !bytes.Equal(valueRows, origValue) {
		t.Fatalf("expired sliding restore mutated rows key=%v value=%v", keyRows, valueRows)
	}
}

func TestSessionStateRestoreBlockRejectsUnlabelledMaxSizeMismatch(t *testing.T) {
	err := restoreStateBlock(0, 0, 2, 1, []sessionStateLayerView{{
		layer:      0,
		cacheIndex: 0,
		cacheMode:  nativeStateCacheModeFixed,
		maxSize:    4,
		cacheRows:  4,
		kvHeads:    1,
		headDim:    1,
		rowBytes:   2,
		keyBytes:   make([]byte, 8),
		valueBytes: make([]byte, 8),
	}}, SessionStateBlock{
		Index:      0,
		TokenStart: 0,
		TokenCount: 2,
		Layers: []SessionStateLayerBlock{{
			Layer:      0,
			CacheIndex: 0,
			MaxSize:    6,
			KVHeads:    1,
			HeadDim:    1,
			RowBytes:   2,
			KeyBytes:   []byte{1, 0, 2, 0},
			ValueBytes: []byte{11, 0, 12, 0},
		}},
	})
	if err == nil {
		t.Fatal("restoreStateBlock unlabelled max-size mismatch error = nil")
	}
}

func TestSessionStateRestoreBlockAllowsPortableSourceMetadata(t *testing.T) {
	for _, tc := range []struct {
		name    string
		mode    string
		maxSize int
	}{
		{name: "paged", mode: "paged", maxSize: 8},
		{name: "rotating", mode: "rotating", maxSize: 8},
		{name: "sliding", mode: "sliding", maxSize: 8},
		{name: "turboquant", mode: "turboquant", maxSize: 8},
	} {
		t.Run(tc.name, func(t *testing.T) {
			keyRows := make([]byte, 8)
			valueRows := make([]byte, 8)
			err := restoreStateBlock(0, 0, 2, 1, []sessionStateLayerView{{
				layer:      0,
				cacheIndex: 0,
				cacheMode:  nativeStateCacheModeFixed,
				maxSize:    4,
				cacheRows:  4,
				kvHeads:    1,
				headDim:    1,
				rowBytes:   2,
				keyBytes:   keyRows,
				valueBytes: valueRows,
			}}, SessionStateBlock{
				Index:      0,
				TokenStart: 0,
				TokenCount: 2,
				Layers: []SessionStateLayerBlock{{
					Layer:      0,
					CacheIndex: 0,
					CacheMode:  tc.mode,
					MaxSize:    tc.maxSize,
					KVHeads:    1,
					HeadDim:    1,
					RowBytes:   2,
					KeyBytes:   []byte{1, 0, 2, 0},
					ValueBytes: []byte{11, 0, 12, 0},
				}},
			})
			if err != nil {
				t.Fatalf("restoreStateBlock portable %s metadata: %v", tc.mode, err)
			}
			if !bytes.Equal(keyRows[:4], []byte{1, 0, 2, 0}) {
				t.Fatalf("restored key rows = %v, want source payload", keyRows)
			}
			if !bytes.Equal(valueRows[:4], []byte{11, 0, 12, 0}) {
				t.Fatalf("restored value rows = %v, want source payload", valueRows)
			}
		})
	}
}

func TestSessionStateRestoreBlockRejectsCacheModeMismatch(t *testing.T) {
	err := restoreStateBlock(0, 0, 2, 1, []sessionStateLayerView{{
		layer:      0,
		cacheIndex: 0,
		cacheMode:  nativeStateCacheModeFixed,
		maxSize:    4,
		cacheRows:  4,
		kvHeads:    1,
		headDim:    1,
		rowBytes:   2,
		keyBytes:   make([]byte, 8),
		valueBytes: make([]byte, 8),
	}}, SessionStateBlock{
		Index:      0,
		TokenStart: 0,
		TokenCount: 2,
		Layers: []SessionStateLayerBlock{{
			Layer:      0,
			CacheIndex: 0,
			CacheMode:  "compaction",
			MaxSize:    4,
			KVHeads:    1,
			HeadDim:    1,
			RowBytes:   2,
			KeyBytes:   []byte{1, 0, 2, 0},
			ValueBytes: []byte{11, 0, 12, 0},
		}},
	})
	if err == nil {
		t.Fatal("restoreStateBlock cache-mode mismatch error = nil")
	}
}

func TestSessionStateRestoreBlocksGraftsTrustedPrefix(t *testing.T) {
	requireNativeRuntime(t)
	prefix := []int32{1, 2, 3, 4}
	suffix := []int32{5, 6, 7}
	prompt := append(append([]int32(nil), prefix...), suffix...)

	saved := newSessionStateFixture(t)
	if err := saved.PrefillTokens(prompt); err != nil {
		t.Fatalf("PrefillTokens full prompt: %v", err)
	}
	source, err := saved.StateBlockSourceFrom(len(prefix), 2)
	if err != nil {
		t.Fatalf("StateBlockSourceFrom: %v", err)
	}

	empty := newSessionStateFixture(t)
	if err := empty.RestoreStateBlocks(source); err == nil {
		t.Fatal("RestoreStateBlocks skipped-prefix into empty session error = nil")
	}

	restored := newSessionStateFixture(t)
	if err := restored.PrefillTokens(prefix); err != nil {
		t.Fatalf("PrefillTokens prefix: %v", err)
	}
	if err := restored.RestoreStateBlocks(source); err != nil {
		t.Fatalf("RestoreStateBlocks skipped-prefix: %v", err)
	}
	if restored.Pos() != len(prompt) {
		t.Fatalf("restored pos = %d, want %d", restored.Pos(), len(prompt))
	}
	if !idsEqual(restored.cachedIDs, prompt) {
		t.Fatalf("restored cached ids = %v, want %v", restored.cachedIDs, prompt)
	}
	got, err := restored.GenerateFromCache(3, -1)
	if err != nil {
		t.Fatalf("GenerateFromCache after skipped-prefix block restore: %v", err)
	}
	cold := newSessionStateFixture(t)
	want, err := cold.Generate(prompt, 3, -1)
	if err != nil {
		t.Fatalf("cold Generate: %v", err)
	}
	if !idsEqual(got, want) {
		t.Fatalf("skipped-prefix block-restored GenerateFromCache = %v, want cold prompt continuation %v", got, want)
	}
}

func TestSessionStateRestoreBlocksGraftsExactTrustedPrefix(t *testing.T) {
	requireNativeRuntime(t)
	prefix := []int32{1, 2, 3}
	suffix := []int32{4}
	prompt := append(append([]int32(nil), prefix...), suffix...)

	saved := newSessionStateFixture(t)
	if err := saved.PrefillTokens(prompt); err != nil {
		t.Fatalf("PrefillTokens full prompt: %v", err)
	}
	sourceAll, err := saved.StateBlockSource(2)
	if err != nil {
		t.Fatalf("StateBlockSource: %v", err)
	}
	parentBlock, err := sourceAll.Load(1)
	if err != nil {
		t.Fatalf("Load parent suffix block: %v", err)
	}
	if parentBlock.TokenStart != 2 || parentBlock.TokenCount != 2 {
		t.Fatalf("parent block = start %d count %d, want 2/2", parentBlock.TokenStart, parentBlock.TokenCount)
	}
	suffixLayers := make([]SessionStateLayerBlock, len(parentBlock.Layers))
	for i, layer := range parentBlock.Layers {
		rowOff := (len(prefix) - parentBlock.TokenStart) * layer.RowBytes
		rowEnd := rowOff + layer.RowBytes
		if rowOff < 0 || rowEnd > len(layer.KeyBytes) || rowEnd > len(layer.ValueBytes) {
			t.Fatalf("suffix layer %d row slice [%d:%d] outside key/value payloads", i, rowOff, rowEnd)
		}
		suffixLayers[i] = layer
		suffixLayers[i].KeyBytes = append([]byte(nil), layer.KeyBytes[rowOff:rowEnd]...)
		suffixLayers[i].ValueBytes = append([]byte(nil), layer.ValueBytes[rowOff:rowEnd]...)
	}
	source := SessionStateBlockSource{
		Position:           len(prompt),
		CachedIDs:          append([]int32(nil), prompt...),
		CachedPromptIDs:    append([]int32(nil), prompt...),
		CachedPromptHidden: append([]byte(nil), sourceAll.CachedPromptHidden...),
		CachedPromptLogits: append([]byte(nil), sourceAll.CachedPromptLogits...),
		RetainedHidden:     append([]byte(nil), sourceAll.RetainedHidden...),
		RetainedLogits:     append([]byte(nil), sourceAll.RetainedLogits...),
		BlockCount:         1,
		Load: func(index int) (SessionStateBlock, error) {
			if index != 0 {
				return SessionStateBlock{}, core.NewError("test: block index out of range")
			}
			return SessionStateBlock{
				Index:      2,
				TokenStart: len(prefix),
				TokenCount: len(suffix),
				Layers:     suffixLayers,
			}, nil
		},
	}
	if err := source.TrustPrefixTokens(len(prefix), 2); err != nil {
		t.Fatalf("TrustPrefixTokens exact prefix: %v", err)
	}

	empty := newSessionStateFixture(t)
	if err := empty.RestoreStateBlocks(source); err == nil {
		t.Fatal("RestoreStateBlocks exact-prefix into empty session error = nil")
	}

	restored := newSessionStateFixture(t)
	if err := restored.PrefillTokens(prefix); err != nil {
		t.Fatalf("PrefillTokens prefix: %v", err)
	}
	if err := restored.RestoreStateBlocks(source); err != nil {
		t.Fatalf("RestoreStateBlocks exact-prefix: %v", err)
	}
	if restored.Pos() != len(prompt) {
		t.Fatalf("restored pos = %d, want %d", restored.Pos(), len(prompt))
	}
	if !idsEqual(restored.cachedIDs, prompt) {
		t.Fatalf("restored cached ids = %v, want %v", restored.cachedIDs, prompt)
	}
}

func TestSessionStateBlocksRestorePromptCacheEntry(t *testing.T) {
	requireNativeRuntime(t)
	prompt := []int32{1, 2, 3, 4, 5}

	saved := newSessionStateFixture(t)
	if err := saved.WarmPromptCache(prompt); err != nil {
		t.Fatalf("WarmPromptCache: %v", err)
	}
	source, err := saved.StateBlockSource(2)
	if err != nil {
		t.Fatalf("StateBlockSource: %v", err)
	}

	restored := newSessionStateFixture(t)
	// Counts HOST head-closure calls below — the host decode lane's cached-logits flow.
	// bf16 sessions record the arch ICB + GPU next-inputs seam now, which fuse the head
	// device-side (the closure never fires); force the host lane BEFORE the restore so
	// it populates the host-lane caches too.
	restored.state.icb = nil
	restored.encNextInputsGPU = nil
	if err := restored.RestoreStateBlocks(source); err != nil {
		t.Fatalf("RestoreStateBlocks: %v", err)
	}
	if hit := restored.CachedPrefixLen(prompt); hit != len(prompt) {
		t.Fatalf("restored exact prompt-cache hit = %d, want %d", hit, len(prompt))
	}
	head := restored.head
	headCalls := 0
	restored.greedy = nil
	restored.head = func(hidden []byte, skipSoftcap bool) ([]byte, error) {
		headCalls++
		return head(hidden, skipSoftcap)
	}
	got, err := restored.GenerateCached(prompt, 3, -1)
	if err != nil {
		t.Fatalf("GenerateCached after RestoreStateBlocks: %v", err)
	}
	if headCalls != len(got)-1 {
		t.Fatalf("restored exact prompt-cache head calls = %d, want %d", headCalls, len(got)-1)
	}

	cold := newSessionStateFixture(t)
	want, err := cold.Generate(prompt, 3, -1)
	if err != nil {
		t.Fatalf("cold Generate: %v", err)
	}
	if !idsEqual(got, want) {
		t.Fatalf("block-restored prompt-cache generation = %v, want %v", got, want)
	}
}

func TestArchSessionCaptureKVRootSnapshotUsesNativeLayerSlabs(t *testing.T) {
	requireNativeRuntime(t)
	prompt := []int32{1, 2, 3, 4, 5}

	sess := newSessionStateFixture(t)
	if err := sess.PrefillTokens(prompt); err != nil {
		t.Fatalf("PrefillTokens: %v", err)
	}
	snapshot, err := sess.CaptureKVWithOptions(kv.CaptureOptions{RawKVOnly: true})
	if err != nil {
		t.Fatalf("CaptureKVWithOptions: %v", err)
	}
	if snapshot.Version != kv.SnapshotVersion {
		t.Fatalf("snapshot version = %d, want %d", snapshot.Version, kv.SnapshotVersion)
	}
	if !idsEqual(snapshot.Tokens, prompt) || snapshot.TokenOffset != len(prompt) || snapshot.SeqLen != len(prompt) {
		t.Fatalf("snapshot tokens/offset/seq = %v/%d/%d, want %v/%d/%d", snapshot.Tokens, snapshot.TokenOffset, snapshot.SeqLen, prompt, len(prompt), len(prompt))
	}
	if snapshot.NumLayers != len(sess.state.specs) || len(snapshot.Layers) != len(sess.state.specs) {
		t.Fatalf("snapshot layers = %d/%d, want %d", snapshot.NumLayers, len(snapshot.Layers), len(sess.state.specs))
	}
	layer := snapshot.Layers[0]
	if layer.Layer != 0 || layer.CacheIndex != sess.state.specs[0].CacheIndex {
		t.Fatalf("layer identity = %d/%d, want layer 0 cache %d", layer.Layer, layer.CacheIndex, sess.state.specs[0].CacheIndex)
	}
	wantShape := []int32{1, int32(kvHeadsOf(sess.state.specs[0], sess.arch.KVHeads)), int32(len(prompt)), int32(headDimOf(sess.state.specs[0], sess.arch.HeadDim))}
	if !reflect.DeepEqual(layer.KeyShape, wantShape) || !reflect.DeepEqual(layer.ValueShape, wantShape) {
		t.Fatalf("layer shapes = %v/%v, want %v", layer.KeyShape, layer.ValueShape, wantShape)
	}
	if layer.KeyDType != "bfloat16" || layer.ValueDType != "bfloat16" {
		t.Fatalf("layer dtypes = %q/%q, want bfloat16", layer.KeyDType, layer.ValueDType)
	}
	wantBytes := int(wantShape[1] * wantShape[2] * wantShape[3] * bf16Size)
	if len(layer.KeyBytes) != wantBytes || len(layer.ValueBytes) != wantBytes {
		t.Fatalf("layer byte lengths = %d/%d, want %d", len(layer.KeyBytes), len(layer.ValueBytes), wantBytes)
	}
	if len(layer.Heads) != 0 {
		t.Fatalf("raw-only snapshot carried %d per-head float snapshots, want none", len(layer.Heads))
	}
}

func TestArchSessionRestoreKVRootSnapshotContinuesFromNativeLayerSlabs(t *testing.T) {
	requireNativeRuntime(t)
	prompt := []int32{1, 2, 3, 4, 5}

	saved := newSessionStateFixture(t)
	if err := saved.PrefillTokens(prompt); err != nil {
		t.Fatalf("PrefillTokens: %v", err)
	}
	snapshot, err := saved.CaptureKVWithOptions(kv.CaptureOptions{RawKVOnly: true})
	if err != nil {
		t.Fatalf("CaptureKVWithOptions: %v", err)
	}

	restored := newSessionStateFixture(t)
	if err := restored.RestoreKV(snapshot); err != nil {
		t.Fatalf("RestoreKV: %v", err)
	}
	if restored.Pos() != saved.Pos() {
		t.Fatalf("restored pos = %d, want %d", restored.Pos(), saved.Pos())
	}
	if !idsEqual(restored.cachedIDs, prompt) {
		t.Fatalf("restored cached ids = %v, want %v", restored.cachedIDs, prompt)
	}
	got, err := restored.GenerateFromCache(3, -1)
	if err != nil {
		t.Fatalf("GenerateFromCache after RestoreKV: %v", err)
	}
	cold := newSessionStateFixture(t)
	want, err := cold.Generate(prompt, 3, -1)
	if err != nil {
		t.Fatalf("cold Generate: %v", err)
	}
	if !idsEqual(got, want) {
		t.Fatalf("restored GenerateFromCache = %v, want cold continuation %v", got, want)
	}
}

func TestArchSessionRestoreKVNativeLayerSlabsAllocationBudget(t *testing.T) {
	requireNativeRuntime(t)
	prompt := []int32{1, 2, 3, 4, 5}

	saved := newSessionStateFixture(t)
	if err := saved.PrefillTokens(prompt); err != nil {
		t.Fatalf("PrefillTokens: %v", err)
	}
	snapshot, err := saved.CaptureKVWithOptions(kv.CaptureOptions{RawKVOnly: true})
	if err != nil {
		t.Fatalf("CaptureKVWithOptions: %v", err)
	}
	snapshot.Generated = nil
	snapshot.LogitShape = nil
	snapshot.Logits = nil

	restored := newSessionStateFixture(t)
	if err := restored.RestoreKV(snapshot); err != nil {
		t.Fatalf("RestoreKV warmup: %v", err)
	}
	var restoreErr error
	allocs := testing.AllocsPerRun(5, func() {
		restoreErr = restored.RestoreKV(snapshot)
	})
	if restoreErr != nil {
		t.Fatalf("RestoreKV: %v", restoreErr)
	}
	if allocs > 0 {
		t.Fatalf("RestoreKV native slab allocations = %.0f, want 0", allocs)
	}
}

func TestArchSessionRestoreKVRootSnapshotContinuesFromFloat32LayerSlabs(t *testing.T) {
	requireNativeRuntime(t)
	prompt := []int32{1, 2, 3, 4, 5}

	saved := newSessionStateFixture(t)
	if err := saved.PrefillTokens(prompt); err != nil {
		t.Fatalf("PrefillTokens: %v", err)
	}
	snapshot, err := saved.CaptureKVWithOptions(kv.CaptureOptions{RawKVOnly: true})
	if err != nil {
		t.Fatalf("CaptureKVWithOptions: %v", err)
	}
	for idx := range snapshot.Layers {
		layer := &snapshot.Layers[idx]
		layer.KeyDType = "float32"
		layer.KeyBytes = bf16RawToF32Raw(layer.KeyBytes)
		layer.ValueDType = "float32"
		layer.ValueBytes = bf16RawToF32Raw(layer.ValueBytes)
	}

	restored := newSessionStateFixture(t)
	if err := restored.RestoreKV(snapshot); err != nil {
		t.Fatalf("RestoreKV(float32 slabs): %v", err)
	}
	if restored.Pos() != saved.Pos() {
		t.Fatalf("restored pos = %d, want %d", restored.Pos(), saved.Pos())
	}
	if !idsEqual(restored.cachedIDs, prompt) {
		t.Fatalf("restored cached ids = %v, want %v", restored.cachedIDs, prompt)
	}
	got, err := restored.GenerateFromCache(3, -1)
	if err != nil {
		t.Fatalf("GenerateFromCache after RestoreKV(float32 slabs): %v", err)
	}
	cold := newSessionStateFixture(t)
	want, err := cold.Generate(prompt, 3, -1)
	if err != nil {
		t.Fatalf("cold Generate: %v", err)
	}
	if !idsEqual(got, want) {
		t.Fatalf("float32-restored GenerateFromCache = %v, want cold continuation %v", got, want)
	}
}

func bf16RawToF32Raw(src []byte) []byte {
	out := make([]byte, len(src)/bf16Size*4)
	for i := 0; i < len(src)/bf16Size; i++ {
		f := bf16ToF32(src[i*bf16Size], src[i*bf16Size+1])
		binary.LittleEndian.PutUint32(out[i*4:i*4+4], math.Float32bits(f))
	}
	return out
}

func firstTokensFromLayerSlab(t testing.TB, src []byte, seqLen, tokenCount, heads, headDim int) []byte {
	t.Helper()
	if tokenCount <= 0 || tokenCount > seqLen || heads <= 0 || headDim <= 0 {
		t.Fatalf("invalid layer slab prefix shape seq=%d tokens=%d heads=%d dim=%d", seqLen, tokenCount, heads, headDim)
	}
	rowBytes := headDim * bf16Size
	if len(src) != heads*seqLen*rowBytes {
		t.Fatalf("layer slab bytes = %d, want %d", len(src), heads*seqLen*rowBytes)
	}
	out := make([]byte, heads*tokenCount*rowBytes)
	for head := range heads {
		srcStart := head * seqLen * rowBytes
		srcEnd := srcStart + tokenCount*rowBytes
		dstStart := head * tokenCount * rowBytes
		copy(out[dstStart:dstStart+tokenCount*rowBytes], src[srcStart:srcEnd])
	}
	return out
}

func TestNativeKVRawToBF16ConvertsRawDTypes(t *testing.T) {
	rawF16 := make([]byte, 4)
	binary.LittleEndian.PutUint16(rawF16[0:2], 0x3c00) // 1.0
	binary.LittleEndian.PutUint16(rawF16[2:4], 0xc000) // -2.0
	gotF16 := make([]byte, 4)
	if err := nativeKVRawToBF16(gotF16, rawF16, "float16"); err != nil {
		t.Fatalf("nativeKVRawToBF16(float16): %v", err)
	}
	if got := []float32{bf16ToF32(gotF16[0], gotF16[1]), bf16ToF32(gotF16[2], gotF16[3])}; got[0] != 1 || got[1] != -2 {
		t.Fatalf("float16 conversion = %v, want [1 -2]", got)
	}

	rawF32 := make([]byte, 8)
	binary.LittleEndian.PutUint32(rawF32[0:4], math.Float32bits(3.5))
	binary.LittleEndian.PutUint32(rawF32[4:8], math.Float32bits(-4.25))
	gotF32 := make([]byte, 4)
	if err := nativeKVRawToBF16(gotF32, rawF32, "float32"); err != nil {
		t.Fatalf("nativeKVRawToBF16(float32): %v", err)
	}
	if got := []float32{bf16ToF32(gotF32[0], gotF32[1]), bf16ToF32(gotF32[2], gotF32[3])}; got[0] != 3.5 || got[1] != -4.25 {
		t.Fatalf("float32 conversion = %v, want [3.5 -4.25]", got)
	}
}

func TestNativeKVRawDTypeUppercaseAliasesAllocateZero(t *testing.T) {
	allocs := testing.AllocsPerRun(100, func() {
		canonical, bytesPerValue, ok := nativeKVRawDType("BF16")
		if !ok || canonical != nativeKVSnapshotDTypeBF16 || bytesPerValue != bf16Size {
			t.Fatalf("nativeKVRawDType(BF16) = %q/%d/%v, want %q/%d/true", canonical, bytesPerValue, ok, nativeKVSnapshotDTypeBF16, bf16Size)
		}
		canonical, bytesPerValue, ok = nativeKVRawDType("F32")
		if !ok || canonical != "float32" || bytesPerValue != 4 {
			t.Fatalf("nativeKVRawDType(F32) = %q/%d/%v, want float32/4/true", canonical, bytesPerValue, ok)
		}
	})
	if allocs > 0 {
		t.Fatalf("nativeKVRawDType uppercase alias allocations = %.0f, want 0", allocs)
	}
}

func TestNativeKVLayerSnapshotSlabsRestoresTurboQuantPayload(t *testing.T) {
	const heads, tokenCount, headDim = 2, 2, 8
	payload := nativeKVTestTurboQuantZeroPayload(t, heads, tokenCount, headDim)
	view := sessionStateLayerView{
		layer:      0,
		kvHeads:    heads,
		headDim:    headDim,
		rowBytes:   heads * headDim * bf16Size,
		cacheIndex: 0,
		cacheMode:  "turboquant",
		cacheRows:  8,
	}

	keySlab, valueSlab, seqLen, err := nativeKVLayerSnapshotSlabs(kv.LayerSnapshot{
		Layer:              0,
		CacheIndex:         0,
		CacheMode:          "turboquant",
		TurboQuantPayloads: [][]byte{payload},
	}, view)
	if err != nil {
		t.Fatalf("nativeKVLayerSnapshotSlabs(turboquant): %v", err)
	}
	wantBytes := heads * tokenCount * headDim * bf16Size
	if seqLen != tokenCount || len(keySlab) != wantBytes || len(valueSlab) != wantBytes {
		t.Fatalf("turboquant slabs seq/bytes = %d/%d/%d, want %d/%d/%d", seqLen, len(keySlab), len(valueSlab), tokenCount, wantBytes, wantBytes)
	}
	if !bytes.Equal(keySlab, make([]byte, wantBytes)) || !bytes.Equal(valueSlab, make([]byte, wantBytes)) {
		t.Fatalf("turboquant zero-norm slabs = key %v value %v, want all zero bf16", keySlab, valueSlab)
	}
}

func TestNativeKVLayerSnapshotSlabsDecodesTurboQuantCentroids(t *testing.T) {
	const heads, tokenCount, headDim = 1, 1, 2
	payload := nativeKVTestTurboQuantPayload(t, heads, tokenCount, headDim, 1, func(name string, section []byte) {
		switch name {
		case "key_centroids", "value_centroids":
			section[0] = 0x03 // two 1-bit centroid codes: [+1,+1]
		case "key_norms_bf16", "value_norms_bf16":
			binary.LittleEndian.PutUint16(section[:bf16Size], 0x3f80) // 1.0
		}
	})
	view := sessionStateLayerView{
		layer:      0,
		kvHeads:    heads,
		headDim:    headDim,
		rowBytes:   heads * headDim * bf16Size,
		cacheIndex: 0,
		cacheMode:  "turboquant",
		cacheRows:  8,
	}

	keySlab, valueSlab, seqLen, err := nativeKVLayerSnapshotSlabs(kv.LayerSnapshot{
		Layer:              0,
		CacheIndex:         0,
		CacheMode:          "turboquant",
		TurboQuantPayloads: [][]byte{payload},
	}, view)
	if err != nil {
		t.Fatalf("nativeKVLayerSnapshotSlabs(turboquant non-zero): %v", err)
	}
	if seqLen != tokenCount {
		t.Fatalf("turboquant seqLen = %d, want %d", seqLen, tokenCount)
	}
	want := toBF16Bytes([]float32{float32(math.Sqrt2), 0})
	if !bytes.Equal(keySlab, want) || !bytes.Equal(valueSlab, want) {
		t.Fatalf("turboquant decoded slabs = key %v value %v, want %v", bf16ToF32Slice(keySlab), bf16ToF32Slice(valueSlab), bf16ToF32Slice(want))
	}
}

func TestNativeKVLayerSnapshotSlabsAppliesTurboQuantProdQJLResidual(t *testing.T) {
	const heads, tokenCount, headDim = 1, 1, 2
	payload := nativeKVTestTurboQuantPayload(t, heads, tokenCount, headDim, 1, func(name string, section []byte) {
		switch name {
		case "key_centroids", "value_centroids":
			section[0] = 0x03 // two 1-bit centroid codes: [+1,+1]
		case "key_norms_bf16", "value_norms_bf16", "key_residual_norms_bf16":
			binary.LittleEndian.PutUint16(section[:bf16Size], 0x3f80) // 1.0
		case "key_qjl_signs":
			section[0] = 0x01 // signs: [-1,+1]
		}
	})
	view := sessionStateLayerView{
		layer:      0,
		kvHeads:    heads,
		headDim:    headDim,
		rowBytes:   heads * headDim * bf16Size,
		cacheIndex: 0,
		cacheMode:  "turboquant",
		cacheRows:  8,
	}

	keySlab, valueSlab, seqLen, err := nativeKVLayerSnapshotSlabs(kv.LayerSnapshot{
		Layer:              0,
		CacheIndex:         0,
		CacheMode:          "turboquant",
		TurboQuantPayloads: [][]byte{payload},
	}, view)
	if err != nil {
		t.Fatalf("nativeKVLayerSnapshotSlabs(turboquant qjl): %v", err)
	}
	if seqLen != tokenCount {
		t.Fatalf("turboquant seqLen = %d, want %d", seqLen, tokenCount)
	}
	base := toBF16Bytes([]float32{float32(math.Sqrt2), 0})
	if !bytes.Equal(valueSlab, base) {
		t.Fatalf("turboquant value slab = %v, want MSE base %v", bf16ToF32Slice(valueSlab), bf16ToF32Slice(base))
	}
	rotatedSigns := []float64{-1, 1}
	residual := make([]float64, headDim)
	nativeTurboQuantKVRotate(residual, rotatedSigns, 124, true)
	baseF := bf16ToF32Slice(base)
	scale := 1 / math.Sqrt(float64(headDim))
	want := toBF16Bytes([]float32{
		baseF[0] + float32(scale*residual[0]),
		baseF[1] + float32(scale*residual[1]),
	})
	if !bytes.Equal(keySlab, want) {
		t.Fatalf("turboquant key slab = %v, want QJL residual restore %v", bf16ToF32Slice(keySlab), bf16ToF32Slice(want))
	}
}

func TestNativeKVLayerSnapshotSlabsOrdersTurboQuantPagesByTokenOffset(t *testing.T) {
	const heads, tokenCount, headDim = 1, 1, 2
	first := nativeKVTestTurboQuantPayloadAt(t, heads, tokenCount, headDim, 1, 0, func(name string, section []byte) {
		switch name {
		case "key_centroids", "value_centroids":
			section[0] = 0x03
		case "key_norms_bf16", "value_norms_bf16":
			binary.LittleEndian.PutUint16(section[:bf16Size], 0x3f80)
		}
	})
	second := nativeKVTestTurboQuantPayloadAt(t, heads, tokenCount, headDim, 1, 1, nil)
	view := sessionStateLayerView{
		layer:      0,
		kvHeads:    heads,
		headDim:    headDim,
		rowBytes:   heads * headDim * bf16Size,
		cacheIndex: 0,
		cacheMode:  "turboquant",
		cacheRows:  8,
	}

	keySlab, valueSlab, seqLen, err := nativeKVLayerSnapshotSlabs(kv.LayerSnapshot{
		Layer:              0,
		CacheIndex:         0,
		CacheMode:          "turboquant",
		TurboQuantPayloads: [][]byte{second, first},
	}, view)
	if err != nil {
		t.Fatalf("nativeKVLayerSnapshotSlabs(turboquant reordered): %v", err)
	}
	if seqLen != 2 {
		t.Fatalf("turboquant seqLen = %d, want 2", seqLen)
	}
	wantFirst := toBF16Bytes([]float32{float32(math.Sqrt2), 0})
	want := append(append([]byte(nil), wantFirst...), make([]byte, headDim*bf16Size)...)
	if !bytes.Equal(keySlab, want) || !bytes.Equal(valueSlab, want) {
		t.Fatalf("turboquant reordered slabs = key %v value %v, want %v", bf16ToF32Slice(keySlab), bf16ToF32Slice(valueSlab), bf16ToF32Slice(want))
	}
}

func TestNativeTurboQuantLayerPayloadsRowsIntoMultiPageAllocationBudget(t *testing.T) {
	const heads, tokenCount, headDim = 1, 1, 2
	firstRaw := nativeKVTestTurboQuantPayloadAt(t, heads, tokenCount, headDim, 1, 0, func(name string, section []byte) {
		switch name {
		case "key_centroids", "value_centroids":
			section[0] = 0x03
		case "key_norms_bf16", "value_norms_bf16":
			binary.LittleEndian.PutUint16(section[:bf16Size], 0x3f80)
		}
	})
	secondRaw := nativeKVTestTurboQuantPayloadAt(t, heads, tokenCount, headDim, 1, 1, nil)
	first, err := nativeTurboQuantKVParsePayload(firstRaw, 0)
	if err != nil {
		t.Fatalf("parse first turboquant payload: %v", err)
	}
	second, err := nativeTurboQuantKVParsePayload(secondRaw, 1)
	if err != nil {
		t.Fatalf("parse second turboquant payload: %v", err)
	}
	payloads := []nativeTurboQuantKVPagePayload{second, first}
	view := sessionStateLayerView{
		layer:      0,
		kvHeads:    heads,
		headDim:    headDim,
		rowBytes:   heads * headDim * bf16Size,
		cacheIndex: 0,
		cacheMode:  "turboquant",
		cacheRows:  8,
	}
	keyRows := make([]byte, heads*2*headDim*bf16Size)
	valueRows := make([]byte, heads*2*headDim*bf16Size)
	rotated := make([]float64, headDim)
	normalised := make([]float64, headDim)
	if seqLen, err := nativeTurboQuantKVLayerPayloadsRowsIntoScratch(payloads, view, 0, keyRows, valueRows, rotated, normalised); err != nil {
		t.Fatalf("warm nativeTurboQuantKVLayerPayloadsRowsIntoScratch: %v", err)
	} else if seqLen != 2 {
		t.Fatalf("warm seqLen = %d, want 2", seqLen)
	}

	var decodeErr error
	allocs := testing.AllocsPerRun(10, func() {
		payloads[0], payloads[1] = payloads[1], payloads[0]
		clear(keyRows)
		clear(valueRows)
		_, decodeErr = nativeTurboQuantKVLayerPayloadsRowsIntoScratch(payloads, view, 0, keyRows, valueRows, rotated, normalised)
	})
	if decodeErr != nil {
		t.Fatalf("nativeTurboQuantKVLayerPayloadsRowsIntoScratch: %v", decodeErr)
	}
	if allocs > 0 {
		t.Fatalf("multi-page turboquant decode allocations = %.0f, want 0", allocs)
	}
}

func TestNativeTurboQuantKVPayloadEstimateCountsSectionsAndPadding(t *testing.T) {
	const heads, tokenCount, headDim, normalBits = 2, 3, 8, 5
	raw := nativeKVTestTurboQuantPayload(t, heads, tokenCount, headDim, normalBits, nil)
	payload, err := nativeTurboQuantKVParsePayload(raw, 0)
	if err != nil {
		t.Fatalf("parse turboquant payload: %v", err)
	}
	estimate, err := nativeTurboQuantKVPayloadsEstimate([]nativeTurboQuantKVPagePayload{payload})
	if err != nil {
		t.Fatalf("nativeTurboQuantKVPayloadsEstimate: %v", err)
	}

	var sectionBytes uint64
	for _, section := range payload.Sections {
		sectionBytes += section.Bytes
	}
	if estimate.Pages != 1 {
		t.Fatalf("estimate pages = %d, want 1", estimate.Pages)
	}
	if estimate.PageVectors != heads*tokenCount || estimate.PageElements != heads*tokenCount*headDim {
		t.Fatalf("estimate vectors/elements = %d/%d, want %d/%d", estimate.PageVectors, estimate.PageElements, heads*tokenCount, heads*tokenCount*headDim)
	}
	if estimate.PayloadBytes != sectionBytes {
		t.Fatalf("estimate payload bytes = %d, want section sum %d", estimate.PayloadBytes, sectionBytes)
	}
	if estimate.PaddedPayloadBytes != uint64(len(payload.Data)) {
		t.Fatalf("estimate padded payload bytes = %d, want data len %d", estimate.PaddedPayloadBytes, len(payload.Data))
	}
	if estimate.AlignmentPaddingBytes != uint64(len(payload.Data))-sectionBytes {
		t.Fatalf("estimate padding bytes = %d, want %d", estimate.AlignmentPaddingBytes, uint64(len(payload.Data))-sectionBytes)
	}
	if estimate.FP16BaselineBytes != heads*tokenCount*headDim*2*bf16Size {
		t.Fatalf("estimate fp16 baseline = %d, want %d", estimate.FP16BaselineBytes, heads*tokenCount*headDim*2*bf16Size)
	}
	if estimate.KeyQJLSignBytes == 0 || estimate.KeyResidualNormBytes == 0 || estimate.PayloadSavingsRatio <= 0 {
		t.Fatalf("estimate side channels/savings = qjl %d residual %d savings %.4f", estimate.KeyQJLSignBytes, estimate.KeyResidualNormBytes, estimate.PayloadSavingsRatio)
	}
}

func TestNativeTurboQuantKVLayerPrefixSlabs_Good(t *testing.T) {
	const heads, tokens, prefixTokens, headDim = 2, 3, 2, 4
	raw := nativeKVSeededTurboQuantPayload(t, heads, tokens, headDim)
	view := sessionStateLayerView{layer: 0, cacheIndex: 0, kvHeads: heads, headDim: headDim}

	key, value, seqLen, err := nativeTurboQuantKVLayerPrefixSlabs([][]byte{raw}, view, prefixTokens)
	if err != nil {
		t.Fatalf("nativeTurboQuantKVLayerPrefixSlabs: %v", err)
	}
	if seqLen != prefixTokens || len(key) != heads*prefixTokens*headDim*bf16Size || len(value) != len(key) {
		t.Fatalf("prefix slab geometry = key %d value %d seq %d", len(key), len(value), seqLen)
	}
	if bytes.Equal(key, make([]byte, len(key))) || bytes.Equal(value, make([]byte, len(value))) {
		t.Fatal("seeded TurboQuant prefix slabs decoded to all zero bytes")
	}
}

func TestNativeTurboQuantKVLayerRows_Good(t *testing.T) {
	const heads, tokens, headDim = 2, 3, 4
	raw := nativeKVSeededTurboQuantPayload(t, heads, tokens, headDim)
	view := sessionStateLayerView{layer: 0, cacheIndex: 0, kvHeads: heads, headDim: headDim}

	key, value, seqLen, err := nativeTurboQuantKVLayerRows([][]byte{raw}, view)
	if err != nil {
		t.Fatalf("nativeTurboQuantKVLayerRows: %v", err)
	}
	if seqLen != tokens || len(key) != heads*tokens*headDim*bf16Size || len(value) != len(key) {
		t.Fatalf("row geometry = key %d value %d seq %d", len(key), len(value), seqLen)
	}
}

func TestNativeTurboQuantKVLayerPrefixRows_Good(t *testing.T) {
	const heads, tokens, prefixTokens, headDim = 2, 3, 1, 4
	raw := nativeKVSeededTurboQuantPayload(t, heads, tokens, headDim)
	view := sessionStateLayerView{layer: 0, cacheIndex: 0, kvHeads: heads, headDim: headDim}

	key, value, seqLen, err := nativeTurboQuantKVLayerPrefixRows([][]byte{raw}, view, prefixTokens)
	if err != nil {
		t.Fatalf("nativeTurboQuantKVLayerPrefixRows: %v", err)
	}
	if seqLen != prefixTokens || len(key) != heads*prefixTokens*headDim*bf16Size || len(value) != len(key) {
		t.Fatalf("prefix row geometry = key %d value %d seq %d", len(key), len(value), seqLen)
	}
}

func TestNativeTurboQuantKVLayerDecodeLimit_Good(t *testing.T) {
	const heads, tokens, prefixTokens, headDim = 2, 3, 2, 4
	raw := nativeKVSeededTurboQuantPayload(t, heads, tokens, headDim)
	view := sessionStateLayerView{layer: 0, cacheIndex: 0, kvHeads: heads, headDim: headDim}

	key, value, seqLen, err := nativeTurboQuantKVLayerDecodeLimit([][]byte{raw}, view, prefixTokens, false)
	if err != nil {
		t.Fatalf("nativeTurboQuantKVLayerDecodeLimit: %v", err)
	}
	if seqLen != prefixTokens || len(key) != heads*prefixTokens*headDim*bf16Size || len(value) != len(key) {
		t.Fatalf("decode limit geometry = key %d value %d seq %d", len(key), len(value), seqLen)
	}
}

func TestNativeTurboQuantKVLayerRowsInto_Good(t *testing.T) {
	const heads, tokens, prefixTokens, headDim = 2, 3, 2, 4
	raw := nativeKVSeededTurboQuantPayload(t, heads, tokens, headDim)
	view := sessionStateLayerView{layer: 0, cacheIndex: 0, kvHeads: heads, headDim: headDim}
	key := make([]byte, heads*prefixTokens*headDim*bf16Size)
	value := make([]byte, len(key))

	seqLen, err := nativeTurboQuantKVLayerRowsInto([][]byte{raw}, view, prefixTokens, key, value)
	if err != nil {
		t.Fatalf("nativeTurboQuantKVLayerRowsInto: %v", err)
	}
	if seqLen != prefixTokens || bytes.Equal(key, make([]byte, len(key))) || bytes.Equal(value, make([]byte, len(value))) {
		t.Fatalf("rows-into result = seq %d key-zero %v value-zero %v", seqLen, bytes.Equal(key, make([]byte, len(key))), bytes.Equal(value, make([]byte, len(value))))
	}
}

func TestNativeTurboQuantKVLayerRowsIntoScratch_Good(t *testing.T) {
	const heads, tokens, prefixTokens, headDim = 2, 3, 2, 4
	raw := nativeKVSeededTurboQuantPayload(t, heads, tokens, headDim)
	view := sessionStateLayerView{layer: 0, cacheIndex: 0, kvHeads: heads, headDim: headDim}
	key := make([]byte, heads*prefixTokens*headDim*bf16Size)
	value := make([]byte, len(key))
	rotated := make([]float64, headDim)
	normalised := make([]float64, headDim)

	seqLen, err := nativeTurboQuantKVLayerRowsIntoScratch([][]byte{raw}, view, prefixTokens, key, value, rotated, normalised)
	if err != nil {
		t.Fatalf("nativeTurboQuantKVLayerRowsIntoScratch: %v", err)
	}
	if seqLen != prefixTokens || bytes.Equal(key, make([]byte, len(key))) || bytes.Equal(value, make([]byte, len(value))) {
		t.Fatalf("rows-into-scratch result = seq %d", seqLen)
	}
}

func TestNativeTurboQuantKVPagePayloadDecodeBaseBF16Into_Good(t *testing.T) {
	const heads, tokens, headDim = 2, 3, 4
	raw := nativeKVSeededTurboQuantPayload(t, heads, tokens, headDim)
	payload, err := nativeTurboQuantKVParsePayload(raw, 0)
	if err != nil {
		t.Fatalf("parse seeded TurboQuant payload: %v", err)
	}
	key := make([]byte, heads*tokens*headDim*bf16Size)
	value := make([]byte, len(key))
	rotated := make([]float64, headDim)
	normalised := make([]float64, headDim)
	if err := payload.decodeBaseBF16Into(key, value, tokens, 0, rotated, normalised); err != nil {
		t.Fatalf("decodeBaseBF16Into: %v", err)
	}
	if bytes.Equal(key, make([]byte, len(key))) || bytes.Equal(value, make([]byte, len(value))) {
		t.Fatal("decodeBaseBF16Into returned all zero seeded payload bytes")
	}
}

func TestNativeTurboQuantKVMaskBytes_Good(t *testing.T) {
	if got := nativeTurboQuantKVMaskBytes(9); got != 2 {
		t.Fatalf("nativeTurboQuantKVMaskBytes(9) = %d, want 2", got)
	}
	if got := nativeTurboQuantKVMaskBytes(0); got != 0 {
		t.Fatalf("nativeTurboQuantKVMaskBytes(0) = %d, want 0", got)
	}
}

func TestNativeTurboQuantKVOutlierMask_Good(t *testing.T) {
	got := nativeTurboQuantKVOutlierMask(10, 3)
	if !bytes.Equal(got, []byte{0x80, 0x03}) {
		t.Fatalf("nativeTurboQuantKVOutlierMask = %08b, want 10000000 00000011", got)
	}
	if got := nativeTurboQuantKVOutlierMask(4, 8); !bytes.Equal(got, []byte{0x0f}) {
		t.Fatalf("clamped outlier mask = %08b, want 00001111", got)
	}
}

func TestNativeTurboQuantKVBytesEqual_Good(t *testing.T) {
	if !nativeTurboQuantKVBytesEqual([]byte{1, 2, 3}, []byte{1, 2, 3}) {
		t.Fatal("identical payload bytes were not equal")
	}
	if nativeTurboQuantKVBytesEqual([]byte{1, 2, 3}, []byte{1, 2, 4}) {
		t.Fatal("different payload bytes were equal")
	}
	if nativeTurboQuantKVBytesEqual([]byte{1}, []byte{1, 0}) {
		t.Fatal("different payload lengths were equal")
	}
}

func TestNativeTurboQuantKVRotate_Good(t *testing.T) {
	source := []float64{1, -2, 3, -4}
	rotated := make([]float64, len(source))
	restored := make([]float64, len(source))
	nativeTurboQuantKVRotate(rotated, source, 124, false)
	nativeTurboQuantKVRotate(restored, rotated, 124, true)
	for idx := range source {
		if math.Abs(restored[idx]-source[idx]) > 1e-9 {
			t.Fatalf("rotation round-trip[%d] = %.12f, want %.12f", idx, restored[idx], source[idx])
		}
	}
}

func TestNativeTurboQuantKVCodecOutlierChannels_Good(t *testing.T) {
	codec := nativeTurboQuantKVCodec{OutlierMask: []byte{0x81, 0x01}}
	if got := codec.outlierChannels(9); got != 3 {
		t.Fatalf("outlierChannels = %d, want 3", got)
	}
	if got := codec.outlierChannels(0); got != 0 {
		t.Fatalf("outlierChannels(0) = %d, want 0", got)
	}
}

func TestNativeTurboQuantKVCodecBitsForChannel_Good(t *testing.T) {
	codec := nativeTurboQuantKVCodec{NormalBits: 3, OutlierBits: 7, OutlierMask: []byte{0x81}}
	want := map[int32]int{0: 7, 1: 3, 7: 7, 8: 3, -1: 3}
	for channel, expected := range want {
		if got := codec.bitsForChannel(channel); got != expected {
			t.Fatalf("bitsForChannel(%d) = %d, want %d", channel, got, expected)
		}
	}
}

func TestArchSessionTurboQuantKVPayloadEstimate_Good(t *testing.T) {
	raw := nativeKVSeededTurboQuantPayload(t, 2, 2, 4)
	payload, err := nativeTurboQuantKVParsePayload(raw, 0)
	if err != nil {
		t.Fatalf("parse seeded TurboQuant payload: %v", err)
	}
	session := &ArchSession{turboQuantPayloads: []nativeTurboQuantKVPagePayload{payload}}
	estimate, err := session.TurboQuantKVPayloadEstimate()
	if err != nil {
		t.Fatalf("TurboQuantKVPayloadEstimate: %v", err)
	}
	if estimate == nil || estimate.Pages != 1 || estimate.PayloadBytes == 0 || estimate.FP16BaselineBytes == 0 {
		t.Fatalf("TurboQuantKVPayloadEstimate = %+v, want one non-empty estimate", estimate)
	}
}

func TestArchSessionTurboQuantKVPayloadEstimate_Bad(t *testing.T) {
	session := &ArchSession{turboQuantPayloads: []nativeTurboQuantKVPagePayload{{}}}
	if estimate, err := session.TurboQuantKVPayloadEstimate(); err == nil || estimate != nil {
		t.Fatalf("TurboQuantKVPayloadEstimate malformed payload = %+v/%v, want error", estimate, err)
	}
}

func TestArchSessionTurboQuantKVPayloadEstimate_Ugly(t *testing.T) {
	var session *ArchSession
	if estimate, err := session.TurboQuantKVPayloadEstimate(); err != nil || estimate != nil {
		t.Fatalf("TurboQuantKVPayloadEstimate(nil) = %+v/%v, want nil/nil", estimate, err)
	}
	if estimate, err := (&ArchSession{}).TurboQuantKVPayloadEstimate(); err != nil || estimate != nil {
		t.Fatalf("TurboQuantKVPayloadEstimate(empty) = %+v/%v, want nil/nil", estimate, err)
	}
}

func nativeKVSeededTurboQuantPayload(t testing.TB, heads, tokenCount, headDim int) []byte {
	t.Helper()
	return nativeKVTestTurboQuantPayload(t, heads, tokenCount, headDim, 1, func(name string, section []byte) {
		switch name {
		case "key_centroids", "value_centroids":
			for vector := range section {
				if vector%2 == 0 {
					section[vector] = 0x03
				} else {
					section[vector] = 0x00
				}
			}
		case "key_norms_bf16", "value_norms_bf16":
			for offset := 0; offset < len(section); offset += bf16Size {
				binary.LittleEndian.PutUint16(section[offset:offset+bf16Size], 0x3f80)
			}
		}
	})
}

func TestNativeKVValidateLayerMetadataAllowsPortableSourceModes(t *testing.T) {
	view := sessionStateLayerView{
		layer:      0,
		cacheIndex: 3,
		cacheMode:  nativeStateCacheModeFixed,
	}
	for _, mode := range []string{"fixed", "paged", "fp16", "q8", "k-q8-v-q4", "turboquant", "rotating", "sliding"} {
		layer := kv.LayerSnapshot{
			Layer:      0,
			CacheIndex: 3,
			CacheMode:  mode,
		}
		if err := nativeKVValidateLayerMetadata("native.RestoreKV", layer, view); err != nil {
			t.Fatalf("nativeKVValidateLayerMetadata(%q source): %v", mode, err)
		}
	}
	layer := kv.LayerSnapshot{
		Layer:      0,
		CacheIndex: 3,
		CacheMode:  "compaction",
	}
	if err := nativeKVValidateLayerMetadata("native.RestoreKV", layer, view); err == nil {
		t.Fatal("nativeKVValidateLayerMetadata(compaction source) error = nil, want mismatch")
	}
}

func TestNativeKVValidateLayerMetadataAllowsPortableSourceMaxSize(t *testing.T) {
	view := sessionStateLayerView{
		layer:      0,
		cacheIndex: 3,
		cacheMode:  nativeStateCacheModeFixed,
		maxSize:    4,
	}
	for _, mode := range []string{"fixed", "paged", "rotating", "sliding", "turboquant"} {
		layer := kv.LayerSnapshot{
			Layer:      0,
			CacheIndex: 3,
			CacheMode:  mode,
			MaxSize:    8,
		}
		if err := nativeKVValidateLayerMetadata("native.RestoreKV", layer, view); err != nil {
			t.Fatalf("nativeKVValidateLayerMetadata(%q MaxSize source): %v", mode, err)
		}
	}
}

func TestArchSessionRestoreKVAllowsPortableSourceMaxSize(t *testing.T) {
	requireNativeRuntime(t)
	g, arch, maxLen := sessionStateFixture(t)
	arch.SlidingWindow = 4
	arch.Layer[0].Attention = model.SlidingAttention
	prompt := []int32{1, 2, 3, 4, 5, 6}

	saved, err := NewArchSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("NewArchSession(saved): %v", err)
	}
	if err := saved.PrefillTokens(prompt); err != nil {
		t.Fatalf("PrefillTokens: %v", err)
	}
	snapshot, err := saved.CaptureKVWithOptions(kv.CaptureOptions{RawKVOnly: true})
	if err != nil {
		t.Fatalf("CaptureKVWithOptions: %v", err)
	}
	if snapshot.Layers[0].CacheMode != nativeStateCacheModeFixed || snapshot.Layers[0].MaxSize != arch.SlidingWindow {
		t.Fatalf("captured layer 0 metadata = %q/%d, want fixed/%d", snapshot.Layers[0].CacheMode, snapshot.Layers[0].MaxSize, arch.SlidingWindow)
	}
	snapshot.Layers[0].CacheMode = "rotating"
	snapshot.Layers[0].MaxSize = 8

	restored, err := NewArchSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("NewArchSession(restored): %v", err)
	}
	if err := restored.RestoreKV(snapshot); err != nil {
		t.Fatalf("RestoreKV portable source max size: %v", err)
	}
	if restored.Pos() != saved.Pos() {
		t.Fatalf("restored pos = %d, want %d", restored.Pos(), saved.Pos())
	}
	if !idsEqual(restored.cachedIDs, prompt) {
		t.Fatalf("restored cached ids = %v, want %v", restored.cachedIDs, prompt)
	}

	got, err := restored.GenerateFromCache(2, -1)
	if err != nil {
		t.Fatalf("GenerateFromCache after RestoreKV: %v", err)
	}
	cold, err := NewArchSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("NewArchSession(cold): %v", err)
	}
	want, err := cold.Generate(prompt, 2, -1)
	if err != nil {
		t.Fatalf("cold Generate: %v", err)
	}
	if !idsEqual(got, want) {
		t.Fatalf("restored GenerateFromCache = %v, want cold continuation %v", got, want)
	}
}

func nativeKVTestTurboQuantZeroPayload(t testing.TB, heads, tokenCount, headDim int) []byte {
	t.Helper()
	return nativeKVTestTurboQuantPayload(t, heads, tokenCount, headDim, 5, nil)
}

func nativeKVTestTurboQuantPayload(t testing.TB, heads, tokenCount, headDim, normalBits int, fill func(string, []byte)) []byte {
	t.Helper()
	return nativeKVTestTurboQuantPayloadAt(t, heads, tokenCount, headDim, normalBits, 0, fill)
}

func nativeKVTestTurboQuantPayloadAt(t testing.TB, heads, tokenCount, headDim, normalBits, tokenOffset int, fill func(string, []byte)) []byte {
	t.Helper()
	const alignment = 64
	pageVectors := heads * tokenCount
	data := make([]byte, 0)
	sections := make([]map[string]any, 0, 6)
	addSection := func(name string, byteCount int) {
		if rem := len(data) % alignment; rem != 0 {
			data = append(data, make([]byte, alignment-rem)...)
		}
		offset := len(data)
		sections = append(sections, map[string]any{
			"name":      name,
			"offset":    offset,
			"bytes":     byteCount,
			"alignment": alignment,
		})
		data = append(data, make([]byte, byteCount)...)
		if fill != nil {
			fill(name, data[offset:offset+byteCount])
		}
	}
	keyCentroidBytes := (headDim*normalBits + 7) / 8
	qjlBytes := (headDim + 7) / 8
	valueCentroidBytes := keyCentroidBytes
	addSection("key_centroids", pageVectors*keyCentroidBytes)
	addSection("key_qjl_signs", pageVectors*qjlBytes)
	addSection("key_norms_bf16", pageVectors*bf16Size)
	addSection("key_residual_norms_bf16", pageVectors*bf16Size)
	addSection("value_centroids", pageVectors*valueCentroidBytes)
	addSection("value_norms_bf16", pageVectors*bf16Size)
	payload := map[string]any{
		"layout": map[string]any{
			"version":      1,
			"codec":        "turboquant-kv-v1",
			"cache_index":  0,
			"layer":        0,
			"layer_type":   "full_attention",
			"shared_owner": 0,
			"shape": map[string]any{
				"batch":    1,
				"heads":    heads,
				"seq_len":  tokenCount,
				"head_dim": headDim,
			},
			"token_offset": tokenOffset,
			"page_tokens":  tokenCount,
			"page_size":    tokenCount,
			"key": map[string]any{
				"algorithm":            "turboquantprod",
				"normal_bits":          normalBits,
				"norm_policy":          "explicit-vector-norm-bf16-v1",
				"residual_norm_policy": "explicit-vector-residual-norm-bf16-v1",
				"rotation_seed":        2,
				"qjl_seed":             124,
				"codebook_id":          "uniform-fwht",
			},
			"value": map[string]any{
				"algorithm":     "turboquantmse",
				"normal_bits":   normalBits,
				"norm_policy":   "explicit-vector-norm-bf16-v1",
				"rotation_seed": 2,
				"codebook_id":   "uniform-fwht",
			},
		},
		"endian":    "little",
		"alignment": alignment,
		"sections":  sections,
		"data":      data,
	}
	encoded, err := json.Marshal(payload)
	if err != nil {
		t.Fatalf("marshal turboquant fixture: %v", err)
	}
	return encoded
}

func TestArchSessionRangeKVBlocksRootSnapshots(t *testing.T) {
	requireNativeRuntime(t)
	prompt := []int32{1, 2, 3, 4, 5}

	sess := newSessionStateFixture(t)
	if err := sess.PrefillTokens(prompt); err != nil {
		t.Fatalf("PrefillTokens: %v", err)
	}
	var blocks []kv.Block
	err := sess.RangeKVBlocks(2, kv.CaptureOptions{RawKVOnly: true}, func(block kv.Block) (bool, error) {
		blocks = append(blocks, block)
		return true, nil
	})
	if err != nil {
		t.Fatalf("RangeKVBlocks: %v", err)
	}
	if len(blocks) != 3 {
		t.Fatalf("RangeKVBlocks yielded %d blocks, want 3", len(blocks))
	}
	for i, block := range blocks {
		if block.Index != i {
			t.Fatalf("block %d index = %d, want %d", i, block.Index, i)
		}
		if block.Snapshot == nil {
			t.Fatalf("block %d snapshot = nil", i)
		}
		if block.Snapshot.TokenOffset != block.TokenStart+block.TokenCount || block.Snapshot.SeqLen != block.TokenCount {
			t.Fatalf("block %d offset/seq = %d/%d, want %d/%d", i, block.Snapshot.TokenOffset, block.Snapshot.SeqLen, block.TokenStart+block.TokenCount, block.TokenCount)
		}
		if !idsEqual(block.Snapshot.Tokens, prompt[block.TokenStart:block.TokenStart+block.TokenCount]) {
			t.Fatalf("block %d tokens = %v, want %v", i, block.Snapshot.Tokens, prompt[block.TokenStart:block.TokenStart+block.TokenCount])
		}
	}
	layer := blocks[0].Snapshot.Layers[0]
	wantShape := []int32{1, int32(kvHeadsOf(sess.state.specs[0], sess.arch.KVHeads)), 2, int32(headDimOf(sess.state.specs[0], sess.arch.HeadDim))}
	if !reflect.DeepEqual(layer.KeyShape, wantShape) || !reflect.DeepEqual(layer.ValueShape, wantShape) {
		t.Fatalf("first block layer shapes = %v/%v, want %v", layer.KeyShape, layer.ValueShape, wantShape)
	}
	if len(blocks[0].Snapshot.Logits) != 0 {
		t.Fatalf("non-final block carried %d logits, want none", len(blocks[0].Snapshot.Logits))
	}
	if len(blocks[len(blocks)-1].Snapshot.Logits) != sess.arch.Vocab {
		t.Fatalf("final block logits = %d, want vocab %d", len(blocks[len(blocks)-1].Snapshot.Logits), sess.arch.Vocab)
	}
}

func TestArchSessionKVBlockSourceBorrowsRetainedLogitsNoCopy(t *testing.T) {
	requireNativeRuntime(t)
	prompt := []int32{1, 2, 3, 4, 5}

	sess := newSessionStateFixture(t)
	if err := sess.PrefillTokens(prompt); err != nil {
		t.Fatalf("PrefillTokens: %v", err)
	}
	source, err := sess.KVBlockSource(2, kv.CaptureOptions{RawKVOnly: true})
	if err != nil {
		t.Fatalf("KVBlockSource: %v", err)
	}
	if len(source.RetainedLogits) == 0 || len(sess.retainedLogits) == 0 {
		t.Fatalf("retained logits lengths = source %d session %d, want both non-empty", len(source.RetainedLogits), len(sess.retainedLogits))
	}
	if unsafe.Pointer(&source.RetainedLogits[0]) != unsafe.Pointer(&sess.retainedLogits[0]) {
		t.Fatal("KVBlockSource copied retained logits; want borrowed no-copy boundary")
	}
}

func TestArchSessionRangeKVBlocksHonoursBlockStartToken(t *testing.T) {
	requireNativeRuntime(t)
	prompt := []int32{1, 2, 3, 4, 5, 6}

	sess := newSessionStateFixture(t)
	if err := sess.PrefillTokens(prompt); err != nil {
		t.Fatalf("PrefillTokens: %v", err)
	}
	var blocks []kv.Block
	err := sess.RangeKVBlocks(2, kv.CaptureOptions{RawKVOnly: true, BlockStartToken: 4}, func(block kv.Block) (bool, error) {
		blocks = append(blocks, block)
		return true, nil
	})
	if err != nil {
		t.Fatalf("RangeKVBlocks: %v", err)
	}
	if len(blocks) != 1 {
		t.Fatalf("RangeKVBlocks yielded %d blocks, want 1", len(blocks))
	}
	if blocks[0].Index != 2 || blocks[0].TokenStart != 4 || blocks[0].TokenCount != 2 {
		t.Fatalf("block identity = index %d start %d count %d, want 2/4/2", blocks[0].Index, blocks[0].TokenStart, blocks[0].TokenCount)
	}
	if !idsEqual(blocks[0].Snapshot.Tokens, prompt[4:]) {
		t.Fatalf("block tokens = %v, want %v", blocks[0].Snapshot.Tokens, prompt[4:])
	}
}

func TestArchSessionRestoreKVBlocksRootSnapshotsContinues(t *testing.T) {
	requireNativeRuntime(t)
	prompt := []int32{1, 2, 3, 4, 5}

	saved := newSessionStateFixture(t)
	if err := saved.PrefillTokens(prompt); err != nil {
		t.Fatalf("PrefillTokens: %v", err)
	}
	source, err := saved.KVBlockSource(2, kv.CaptureOptions{RawKVOnly: true})
	if err != nil {
		t.Fatalf("KVBlockSource: %v", err)
	}

	restored := newSessionStateFixture(t)
	if err := restored.RestoreKVBlocks(source); err != nil {
		t.Fatalf("RestoreKVBlocks: %v", err)
	}
	if restored.Pos() != saved.Pos() {
		t.Fatalf("restored pos = %d, want %d", restored.Pos(), saved.Pos())
	}
	if !idsEqual(restored.cachedIDs, prompt) {
		t.Fatalf("restored cached ids = %v, want %v", restored.cachedIDs, prompt)
	}
	got, err := restored.GenerateFromCache(3, -1)
	if err != nil {
		t.Fatalf("GenerateFromCache after RestoreKVBlocks: %v", err)
	}
	cold := newSessionStateFixture(t)
	want, err := cold.Generate(prompt, 3, -1)
	if err != nil {
		t.Fatalf("cold Generate: %v", err)
	}
	if !idsEqual(got, want) {
		t.Fatalf("restored GenerateFromCache = %v, want cold continuation %v", got, want)
	}
}

func TestArchSessionRestoreKVBlocksNativeLayerSlabsAllocationBudget(t *testing.T) {
	requireNativeRuntime(t)
	prompt := []int32{1, 2, 3, 4, 5}

	saved := newSessionStateFixture(t)
	if err := saved.PrefillTokens(prompt); err != nil {
		t.Fatalf("PrefillTokens: %v", err)
	}
	source, err := saved.KVBlockSource(len(prompt), kv.CaptureOptions{RawKVOnly: true})
	if err != nil {
		t.Fatalf("KVBlockSource: %v", err)
	}
	block, err := source.Load(0)
	if err != nil {
		t.Fatalf("source.Load(0): %v", err)
	}
	restored := newSessionStateFixture(t)
	views, err := restored.stateLayerViews()
	if err != nil {
		t.Fatalf("stateLayerViews: %v", err)
	}
	if err := restored.restoreKVSnapshotBlockLayers(block, len(prompt), views); err != nil {
		t.Fatalf("restoreKVSnapshotBlockLayers warmup: %v", err)
	}
	var restoreErr error
	allocs := testing.AllocsPerRun(10, func() {
		restoreErr = restored.restoreKVSnapshotBlockLayers(block, len(prompt), views)
	})
	if restoreErr != nil {
		t.Fatalf("restoreKVSnapshotBlockLayers: %v", restoreErr)
	}
	if allocs > 0 {
		t.Fatalf("RestoreKVBlocks native slab allocations = %.0f, want 0", allocs)
	}
}

func TestArchSessionRestoreKVBlocksNativeSourceAllocationBudget(t *testing.T) {
	requireNativeRuntime(t)
	prompt := []int32{1, 2, 3, 4, 5}

	saved := newSessionStateFixture(t)
	if err := saved.PrefillTokens(prompt); err != nil {
		t.Fatalf("PrefillTokens: %v", err)
	}
	source, err := saved.KVBlockSource(2, kv.CaptureOptions{RawKVOnly: true})
	if err != nil {
		t.Fatalf("KVBlockSource: %v", err)
	}
	restored := newSessionStateFixture(t)
	if err := restored.RestoreKVBlocks(source); err != nil {
		t.Fatalf("RestoreKVBlocks warmup: %v", err)
	}
	var restoreErr error
	allocs := testing.AllocsPerRun(10, func() {
		restoreErr = restored.RestoreKVBlocks(source)
	})
	if restoreErr != nil {
		t.Fatalf("RestoreKVBlocks: %v", restoreErr)
	}
	if allocs > 0 {
		t.Fatalf("RestoreKVBlocks native source allocations = %.0f, want 0", allocs)
	}
}

func TestArchSessionRestoreKVBlocksPortableSourceRetainedLogitsAllocationBudget(t *testing.T) {
	requireNativeRuntime(t)
	prompt := []int32{1, 2, 3, 4, 5}

	saved := newSessionStateFixture(t)
	if err := saved.PrefillTokens(prompt); err != nil {
		t.Fatalf("PrefillTokens: %v", err)
	}
	source, err := saved.KVBlockSource(2, kv.CaptureOptions{RawKVOnly: true})
	if err != nil {
		t.Fatalf("KVBlockSource: %v", err)
	}
	blocks := make([]kv.Block, source.BlockCount)
	for i := range blocks {
		blocks[i], err = source.Load(i)
		if err != nil {
			t.Fatalf("source.Load(%d): %v", i, err)
		}
	}
	source.nativeStateSource = nil
	source.Load = func(index int) (kv.Block, error) {
		if index < 0 || index >= len(blocks) {
			return kv.Block{}, core.NewError("unexpected block index")
		}
		return blocks[index], nil
	}

	restored := newSessionStateFixture(t)
	if err := restored.RestoreKVBlocks(source); err != nil {
		t.Fatalf("RestoreKVBlocks warmup: %v", err)
	}
	var restoreErr error
	allocs := testing.AllocsPerRun(10, func() {
		restoreErr = restored.RestoreKVBlocks(source)
	})
	if restoreErr != nil {
		t.Fatalf("RestoreKVBlocks: %v", restoreErr)
	}
	if allocs > 0 {
		t.Fatalf("RestoreKVBlocks portable retained-logits allocations = %.0f, want 0", allocs)
	}
}

func TestArchSessionRestoreKVBlocksHonoursPrefixTokens(t *testing.T) {
	requireNativeRuntime(t)
	prompt := []int32{1, 2, 3, 4, 5, 6}
	prefix := prompt[:4]

	saved := newSessionStateFixture(t)
	if err := saved.PrefillTokens(prompt); err != nil {
		t.Fatalf("PrefillTokens(saved): %v", err)
	}
	source, err := saved.KVBlockSource(2, kv.CaptureOptions{RawKVOnly: true})
	if err != nil {
		t.Fatalf("KVBlockSource: %v", err)
	}
	source.PrefixTokens = len(prefix)

	restored := newSessionStateFixture(t)
	if err := restored.RestoreKVBlocks(source); err != nil {
		t.Fatalf("RestoreKVBlocks: %v", err)
	}
	if restored.Pos() != len(prefix) {
		t.Fatalf("restored pos = %d, want %d", restored.Pos(), len(prefix))
	}
	if !idsEqual(restored.cachedIDs, prefix) {
		t.Fatalf("restored cached ids = %v, want %v", restored.cachedIDs, prefix)
	}
	if _, err := restored.GenerateFromCache(1, -1); err == nil {
		t.Fatal("GenerateFromCache after prefix-only RestoreKVBlocks error = nil")
	}
}

func TestArchSessionRestoreKVBlocksSlicesPartialPrefixBlock(t *testing.T) {
	requireNativeRuntime(t)
	prompt := []int32{1, 2, 3, 4, 5, 6}
	prefixLen := 3

	saved := newSessionStateFixture(t)
	if err := saved.PrefillTokens(prompt); err != nil {
		t.Fatalf("PrefillTokens(saved): %v", err)
	}
	source, err := saved.KVBlockSource(2, kv.CaptureOptions{RawKVOnly: true})
	if err != nil {
		t.Fatalf("KVBlockSource: %v", err)
	}
	source.PrefixTokens = prefixLen

	restored := newSessionStateFixture(t)
	if err := restored.RestoreKVBlocks(source); err != nil {
		t.Fatalf("RestoreKVBlocks: %v", err)
	}
	if restored.Pos() != prefixLen {
		t.Fatalf("restored pos = %d, want %d", restored.Pos(), prefixLen)
	}
	if !idsEqual(restored.cachedIDs, prompt[:prefixLen]) {
		t.Fatalf("restored cached ids = %v, want %v", restored.cachedIDs, prompt[:prefixLen])
	}
	wantPrefix := newSessionStateFixture(t)
	if err := wantPrefix.PrefillTokens(prompt[:prefixLen]); err != nil {
		t.Fatalf("PrefillTokens(want prefix): %v", err)
	}
	gotBlocks, err := restored.StateBlockSource(2)
	if err != nil {
		t.Fatalf("StateBlockSource(restored): %v", err)
	}
	wantBlocks, err := wantPrefix.StateBlockSource(2)
	if err != nil {
		t.Fatalf("StateBlockSource(want): %v", err)
	}
	if gotBlocks.BlockCount != wantBlocks.BlockCount {
		t.Fatalf("block count = %d, want %d", gotBlocks.BlockCount, wantBlocks.BlockCount)
	}
	for i := 0; i < gotBlocks.BlockCount; i++ {
		got, err := gotBlocks.Load(i)
		if err != nil {
			t.Fatalf("Load(restored %d): %v", i, err)
		}
		want, err := wantBlocks.Load(i)
		if err != nil {
			t.Fatalf("Load(want %d): %v", i, err)
		}
		if got.TokenStart != want.TokenStart || got.TokenCount != want.TokenCount || len(got.Layers) != len(want.Layers) {
			t.Fatalf("block %d metadata = start %d count %d layers %d, want %d/%d/%d", i, got.TokenStart, got.TokenCount, len(got.Layers), want.TokenStart, want.TokenCount, len(want.Layers))
		}
		for li := range got.Layers {
			if !bytes.Equal(got.Layers[li].KeyBytes, want.Layers[li].KeyBytes) || !bytes.Equal(got.Layers[li].ValueBytes, want.Layers[li].ValueBytes) {
				t.Fatalf("block %d layer %d KV bytes mismatch", i, li)
			}
		}
	}
}

func TestArchSessionRestoreKVBlocksSlicesTurboQuantPrefixBlock(t *testing.T) {
	requireNativeRuntime(t)
	restored := newSingleLayerSessionStateFixture(t)
	source, layer, view := turboQuantPrefixKVBlockSourceFixture(t, restored)

	if err := restored.RestoreKVBlocks(source); err != nil {
		t.Fatalf("RestoreKVBlocks(turboquant prefix): %v", err)
	}
	if restored.Pos() != 1 || !idsEqual(restored.cachedIDs, []int32{11}) {
		t.Fatalf("restored metadata = pos %d ids %v, want pos 1 ids [11]", restored.Pos(), restored.cachedIDs)
	}
	snapshot, err := restored.CaptureKVWithOptions(kv.CaptureOptions{RawKVOnly: true})
	if err != nil {
		t.Fatalf("CaptureKVWithOptions(restored): %v", err)
	}
	fullKey, fullValue, seqLen, err := nativeKVLayerSnapshotSlabs(layer, view)
	if err != nil {
		t.Fatalf("nativeKVLayerSnapshotSlabs(source): %v", err)
	}
	if seqLen != 2 {
		t.Fatalf("source seqLen = %d, want 2", seqLen)
	}
	wantKey := firstTokensFromLayerSlab(t, fullKey, 2, 1, view.kvHeads, view.headDim)
	wantValue := firstTokensFromLayerSlab(t, fullValue, 2, 1, view.kvHeads, view.headDim)
	if !bytes.Equal(snapshot.Layers[0].KeyBytes, wantKey) || !bytes.Equal(snapshot.Layers[0].ValueBytes, wantValue) {
		t.Fatalf("restored turboquant prefix KV mismatch")
	}
}

func TestArchSessionRestoreKVBlocksTurboQuantPrefixAllocationBudget(t *testing.T) {
	requireNativeRuntime(t)
	restored := newSingleLayerSessionStateFixture(t)
	source, _, _ := turboQuantPrefixKVBlockSourceFixture(t, restored)
	if err := restored.RestoreKVBlocks(source); err != nil {
		t.Fatalf("RestoreKVBlocks warmup: %v", err)
	}
	var restoreErr error
	allocs := testing.AllocsPerRun(10, func() {
		restoreErr = restored.RestoreKVBlocks(source)
	})
	if restoreErr != nil {
		t.Fatalf("RestoreKVBlocks: %v", restoreErr)
	}
	if allocs > 0 {
		t.Fatalf("RestoreKVBlocks turboquant prefix allocations = %.0f, want 0", allocs)
	}
}

func TestArchSessionRestoreKVBlocksTurboQuantFullBlockAllocationBudget(t *testing.T) {
	requireNativeRuntime(t)
	restored := newSingleLayerSessionStateFixture(t)
	source, _, _ := turboQuantPrefixKVBlockSourceFixture(t, restored)
	source.PrefixTokens = source.TokenCount
	if err := restored.RestoreKVBlocks(source); err != nil {
		t.Fatalf("RestoreKVBlocks warmup: %v", err)
	}
	var restoreErr error
	allocs := testing.AllocsPerRun(10, func() {
		restoreErr = restored.RestoreKVBlocks(source)
	})
	if restoreErr != nil {
		t.Fatalf("RestoreKVBlocks: %v", restoreErr)
	}
	if allocs > 0 {
		t.Fatalf("RestoreKVBlocks turboquant full-block allocations = %.0f, want 0", allocs)
	}
}

func TestArchSessionRestoreKVBlocksTurboQuantPayloadCacheSeesSameBackingMutation(t *testing.T) {
	requireNativeRuntime(t)
	restored := newSingleLayerSessionStateFixture(t)
	view := restoredStateLayerView(t, restored, 0)
	payload := nativeKVTestTurboQuantPayloadAt(t, view.kvHeads, 2, view.headDim, 1, 0, func(name string, section []byte) {
		switch name {
		case "key_centroids", "value_centroids":
			for idx := range section {
				section[idx] = 0xff
			}
		case "key_norms_bf16", "value_norms_bf16":
			for vector := 0; vector < view.kvHeads*2; vector++ {
				binary.LittleEndian.PutUint16(section[vector*bf16Size:], 0x3f80)
			}
		}
	})
	layer := kv.LayerSnapshot{
		Layer:              view.layer,
		CacheIndex:         view.cacheIndex,
		CacheMode:          "turboquant",
		TurboQuantPayloads: [][]byte{payload},
	}
	source := KVBlockSource{
		TokenCount:      2,
		PrefixTokens:    2,
		CachedIDs:       []int32{11, 12},
		BlockCount:      1,
		FirstBlockIndex: 0,
	}
	source.Load = func(index int) (kv.Block, error) {
		if index != 0 {
			return kv.Block{}, core.NewError("unexpected block index")
		}
		return kv.Block{
			Index:      0,
			TokenStart: 0,
			TokenCount: 2,
			Snapshot: &kv.Snapshot{
				Version:       kv.SnapshotVersion,
				Tokens:        []int32{11, 12},
				TokenOffset:   2,
				NumLayers:     1,
				NumHeads:      view.kvHeads,
				SeqLen:        2,
				HeadDim:       view.headDim,
				NumQueryHeads: restored.arch.Heads,
				Layers:        []kv.LayerSnapshot{layer},
			},
		}, nil
	}
	if err := restored.RestoreKVBlocks(source); err != nil {
		t.Fatalf("RestoreKVBlocks warmup: %v", err)
	}

	replacement := nativeKVTestTurboQuantPayloadAt(t, view.kvHeads, 2, view.headDim, 1, 0, nil)
	if len(replacement) != len(payload) {
		t.Fatalf("replacement payload length = %d, want %d", len(replacement), len(payload))
	}
	copy(payload, replacement)
	if err := restored.RestoreKVBlocks(source); err != nil {
		t.Fatalf("RestoreKVBlocks mutated payload: %v", err)
	}
	snapshot, err := restored.CaptureKVWithOptions(kv.CaptureOptions{RawKVOnly: true})
	if err != nil {
		t.Fatalf("CaptureKVWithOptions: %v", err)
	}
	wantKey, wantValue, seqLen, err := nativeKVLayerSnapshotSlabs(kv.LayerSnapshot{
		Layer:              view.layer,
		CacheIndex:         view.cacheIndex,
		CacheMode:          "turboquant",
		TurboQuantPayloads: [][]byte{replacement},
	}, view)
	if err != nil {
		t.Fatalf("nativeKVLayerSnapshotSlabs(replacement): %v", err)
	}
	if seqLen != 2 {
		t.Fatalf("replacement seqLen = %d, want 2", seqLen)
	}
	if !bytes.Equal(snapshot.Layers[0].KeyBytes, wantKey) || !bytes.Equal(snapshot.Layers[0].ValueBytes, wantValue) {
		t.Fatal("RestoreKVBlocks reused stale parsed TurboQuant payload after same-backing mutation")
	}
}

func TestArchSessionTurboQuantKVPayloadEstimateReportsRestoredPayloads(t *testing.T) {
	requireNativeRuntime(t)
	restored := newSingleLayerSessionStateFixture(t)
	source, _, _ := turboQuantPrefixKVBlockSourceFixture(t, restored)
	if err := restored.RestoreKVBlocks(source); err != nil {
		t.Fatalf("RestoreKVBlocks: %v", err)
	}
	estimate, err := restored.TurboQuantKVPayloadEstimate()
	if err != nil {
		t.Fatalf("TurboQuantKVPayloadEstimate: %v", err)
	}
	if estimate == nil {
		t.Fatal("TurboQuantKVPayloadEstimate = nil, want restored payload accounting")
	}
	if estimate.Pages != 1 || estimate.PayloadBytes == 0 || estimate.FP16BaselineBytes == 0 {
		t.Fatalf("TurboQuantKVPayloadEstimate = %+v, want one non-empty payload estimate", *estimate)
	}
	if estimate.KeyQJLSignBytes == 0 || estimate.KeyResidualNormBytes == 0 {
		t.Fatalf("TurboQuantKVPayloadEstimate side channels = qjl %d residual %d, want both", estimate.KeyQJLSignBytes, estimate.KeyResidualNormBytes)
	}
}

func turboQuantPrefixKVBlockSourceFixture(t testing.TB, restored *ArchSession) (KVBlockSource, kv.LayerSnapshot, sessionStateLayerView) {
	t.Helper()
	view := restoredStateLayerView(t, restored, 0)
	payload := nativeKVTestTurboQuantPayloadAt(t, view.kvHeads, 2, view.headDim, 1, 0, func(name string, section []byte) {
		switch name {
		case "key_centroids", "value_centroids":
			for idx := range section {
				section[idx] = 0xff
			}
		case "key_norms_bf16", "value_norms_bf16":
			for vector := 0; vector < view.kvHeads*2; vector++ {
				if vector%2 == 0 {
					binary.LittleEndian.PutUint16(section[vector*bf16Size:], 0x3f80)
				}
			}
		}
	})
	layer := kv.LayerSnapshot{
		Layer:              view.layer,
		CacheIndex:         view.cacheIndex,
		CacheMode:          "turboquant",
		TurboQuantPayloads: [][]byte{payload},
	}
	source := KVBlockSource{
		TokenCount:      2,
		PrefixTokens:    1,
		CachedIDs:       []int32{11, 12},
		BlockCount:      1,
		FirstBlockIndex: 0,
	}
	block := kv.Block{
		Index:      0,
		TokenStart: 0,
		TokenCount: 2,
		Snapshot: &kv.Snapshot{
			Version:       kv.SnapshotVersion,
			Tokens:        []int32{11, 12},
			TokenOffset:   2,
			NumLayers:     1,
			NumHeads:      view.kvHeads,
			SeqLen:        2,
			HeadDim:       view.headDim,
			NumQueryHeads: restored.arch.Heads,
			Layers:        []kv.LayerSnapshot{layer},
		},
	}
	source.Load = func(index int) (kv.Block, error) {
		if index != 0 {
			return kv.Block{}, core.NewError("unexpected block index")
		}
		return block, nil
	}
	return source, layer, view
}

func TestArchSessionRestoreKVBlocksRootSnapshotsTrustedPrefix(t *testing.T) {
	requireNativeRuntime(t)
	prompt := []int32{1, 2, 3, 4, 5, 6}
	prefix := prompt[:4]

	saved := newSessionStateFixture(t)
	if err := saved.PrefillTokens(prompt); err != nil {
		t.Fatalf("PrefillTokens(saved): %v", err)
	}
	source, err := saved.KVBlockSource(2, kv.CaptureOptions{RawKVOnly: true, BlockStartToken: len(prefix)})
	if err != nil {
		t.Fatalf("KVBlockSource: %v", err)
	}
	if source.TrustedPrefixTokens != len(prefix) || source.FirstBlockIndex != 2 {
		t.Fatalf("trusted prefix/index = %d/%d, want %d/2", source.TrustedPrefixTokens, source.FirstBlockIndex, len(prefix))
	}

	restored := newSessionStateFixture(t)
	if err := restored.PrefillTokens(prefix); err != nil {
		t.Fatalf("PrefillTokens(restored prefix): %v", err)
	}
	if err := restored.RestoreKVBlocks(source); err != nil {
		t.Fatalf("RestoreKVBlocks: %v", err)
	}
	if restored.Pos() != saved.Pos() {
		t.Fatalf("restored pos = %d, want %d", restored.Pos(), saved.Pos())
	}
	if !idsEqual(restored.cachedIDs, prompt) {
		t.Fatalf("restored cached ids = %v, want %v", restored.cachedIDs, prompt)
	}
	got, err := restored.GenerateFromCache(3, -1)
	if err != nil {
		t.Fatalf("GenerateFromCache after RestoreKVBlocks trusted prefix: %v", err)
	}
	cold := newSessionStateFixture(t)
	want, err := cold.Generate(prompt, 3, -1)
	if err != nil {
		t.Fatalf("cold Generate: %v", err)
	}
	if !idsEqual(got, want) {
		t.Fatalf("trusted-prefix GenerateFromCache = %v, want cold continuation %v", got, want)
	}
}

func TestArchSessionRestoreKVBlocksPortableRootSnapshotsTrustedPrefix(t *testing.T) {
	requireNativeRuntime(t)
	prompt := []int32{1, 2, 3, 4, 5, 6}
	prefix := prompt[:4]

	saved := newSessionStateFixture(t)
	if err := saved.PrefillTokens(prompt); err != nil {
		t.Fatalf("PrefillTokens(saved): %v", err)
	}
	source, err := saved.KVBlockSource(2, kv.CaptureOptions{RawKVOnly: true, BlockStartToken: len(prefix)})
	if err != nil {
		t.Fatalf("KVBlockSource: %v", err)
	}
	source.nativeStateSource = nil

	restored := newSessionStateFixture(t)
	if err := restored.PrefillTokens(prefix); err != nil {
		t.Fatalf("PrefillTokens(restored prefix): %v", err)
	}
	if err := restored.RestoreKVBlocks(source); err != nil {
		t.Fatalf("RestoreKVBlocks: %v", err)
	}
	if restored.Pos() != saved.Pos() {
		t.Fatalf("restored pos = %d, want %d", restored.Pos(), saved.Pos())
	}
	if !idsEqual(restored.cachedIDs, prompt) {
		t.Fatalf("restored cached ids = %v, want %v", restored.cachedIDs, prompt)
	}
	got, err := restored.GenerateFromCache(3, -1)
	if err != nil {
		t.Fatalf("GenerateFromCache after portable RestoreKVBlocks trusted prefix: %v", err)
	}
	cold := newSessionStateFixture(t)
	want, err := cold.Generate(prompt, 3, -1)
	if err != nil {
		t.Fatalf("cold Generate: %v", err)
	}
	if !idsEqual(got, want) {
		t.Fatalf("portable trusted-prefix GenerateFromCache = %v, want cold continuation %v", got, want)
	}
}

func TestArchSessionRestoreKVBlocksRejectsTrustedPrefixMismatch(t *testing.T) {
	requireNativeRuntime(t)
	prompt := []int32{1, 2, 3, 4, 5, 6}
	prefix := prompt[:4]

	saved := newSessionStateFixture(t)
	if err := saved.PrefillTokens(prompt); err != nil {
		t.Fatalf("PrefillTokens(saved): %v", err)
	}
	source, err := saved.KVBlockSource(2, kv.CaptureOptions{RawKVOnly: true, BlockStartToken: len(prefix)})
	if err != nil {
		t.Fatalf("KVBlockSource: %v", err)
	}

	restored := newSessionStateFixture(t)
	if err := restored.PrefillTokens([]int32{1, 2, 3, 7}); err != nil {
		t.Fatalf("PrefillTokens(restored prefix): %v", err)
	}
	if err := restored.RestoreKVBlocks(source); err == nil {
		t.Fatal("RestoreKVBlocks mismatch error = nil")
	}
}

func TestArchSessionRestoreKVBlocksAllTrustedPrefixContinues(t *testing.T) {
	requireNativeRuntime(t)
	prompt := []int32{1, 2, 3, 4, 5, 6}

	saved := newSessionStateFixture(t)
	if err := saved.PrefillTokens(prompt); err != nil {
		t.Fatalf("PrefillTokens(saved): %v", err)
	}
	source, err := saved.KVBlockSource(2, kv.CaptureOptions{RawKVOnly: true, BlockStartToken: len(prompt)})
	if err != nil {
		t.Fatalf("KVBlockSource: %v", err)
	}
	if source.BlockCount != 0 || source.TrustedPrefixTokens != len(prompt) {
		t.Fatalf("source blocks/trusted prefix = %d/%d, want 0/%d", source.BlockCount, source.TrustedPrefixTokens, len(prompt))
	}

	restored := newSessionStateFixture(t)
	if err := restored.PrefillTokens(prompt); err != nil {
		t.Fatalf("PrefillTokens(restored): %v", err)
	}
	if err := restored.RestoreKVBlocks(source); err != nil {
		t.Fatalf("RestoreKVBlocks: %v", err)
	}
	got, err := restored.GenerateFromCache(3, -1)
	if err != nil {
		t.Fatalf("GenerateFromCache after all-trusted RestoreKVBlocks: %v", err)
	}
	cold := newSessionStateFixture(t)
	want, err := cold.Generate(prompt, 3, -1)
	if err != nil {
		t.Fatalf("cold Generate: %v", err)
	}
	if !idsEqual(got, want) {
		t.Fatalf("all-trusted GenerateFromCache = %v, want cold continuation %v", got, want)
	}
}

func TestSessionStateNoRuntimeValidation(t *testing.T) {
	icbSession := &ArchSession{state: archDecodeState{icb: &archICBReplay{}}}
	if _, err := icbSession.SerializeState(); err != nil {
		t.Fatalf("SerializeState(empty ICB) error = %v", err)
	}
	if err := icbSession.RestoreState(emptySessionStateBlob(0, 0, 0)); err != nil {
		t.Fatalf("RestoreState(empty ICB) error = %v", err)
	}

	if err := (&ArchSession{}).RestoreState(nil); err == nil {
		t.Fatal("RestoreState(nil) error = nil")
	}
	if err := (&ArchSession{}).RestoreState(emptySessionStateBlob(0, 1, 0)); err == nil {
		t.Fatal("RestoreState(layer mismatch) error = nil")
	}

	legacy := make([]byte, 12)
	binary.LittleEndian.PutUint32(legacy[0:], sessionStateMagic)
	binary.LittleEndian.PutUint32(legacy[4:], 7)
	if err := (&ArchSession{}).RestoreState(legacy); err != nil {
		t.Fatalf("RestoreState(legacy snapshot) error = %v", err)
	}

	if err := (&ArchSession{}).RestoreState(append(legacy, 0)); err == nil {
		t.Fatal("RestoreState(truncated metadata length) error = nil")
	}
	truncatedIDs := emptySessionStateBlob(0, 0, 1)[:16]
	if err := (&ArchSession{}).RestoreState(truncatedIDs); err == nil {
		t.Fatal("RestoreState(truncated metadata ids) error = nil")
	}
	trailing := append(emptySessionStateBlob(0, 0, 1), 0)
	if err := (&ArchSession{}).RestoreState(trailing); err == nil {
		t.Fatal("RestoreState(trailing metadata) error = nil")
	}
}

func TestSessionStateSerializeZeroLayerCachedIDs(t *testing.T) {
	saved := &ArchSession{pos: 3, cachedIDs: []int32{7, 8, 9}}
	blob, err := saved.SerializeState()
	if err != nil {
		t.Fatalf("SerializeState: %v", err)
	}

	restored := &ArchSession{}
	if err := restored.RestoreState(blob); err != nil {
		t.Fatalf("RestoreState: %v", err)
	}
	if restored.Pos() != saved.Pos() {
		t.Fatalf("restored pos = %d, want %d", restored.Pos(), saved.Pos())
	}
	next := []int32{7, 8, 9, 10}
	if got := restored.CachedPrefixLen(next); got != len(saved.cachedIDs) {
		t.Fatalf("restored cached prefix = %d, want %d", got, len(saved.cachedIDs))
	}
}

func TestSessionStateRestoresPromptCacheEntry(t *testing.T) {
	requireNativeRuntime(t)
	prompt := []int32{1, 2, 3, 4, 5}

	a := newSessionStateFixture(t)
	if err := a.WarmPromptCache(prompt); err != nil {
		t.Fatalf("WarmPromptCache: %v", err)
	}
	blob, err := a.SerializeState()
	if err != nil {
		t.Fatalf("SerializeState: %v", err)
	}

	b := newSessionStateFixture(t)
	// Counts HOST head-closure calls below — the host decode lane's cached-logits flow.
	// bf16 sessions record the arch ICB + GPU next-inputs seam now, which fuse the head
	// device-side (the closure never fires); force the host lane BEFORE RestoreState so
	// the restore populates the host-lane caches too.
	b.state.icb = nil
	b.encNextInputsGPU = nil
	if err := b.RestoreState(blob); err != nil {
		t.Fatalf("RestoreState: %v", err)
	}
	if hit := b.CachedPrefixLen(prompt); hit != len(prompt) {
		t.Fatalf("restored exact prompt-cache hit = %d, want %d", hit, len(prompt))
	}
	head := b.head
	headCalls := 0
	b.greedy = nil
	b.head = func(hidden []byte, skipSoftcap bool) ([]byte, error) {
		headCalls++
		return head(hidden, skipSoftcap)
	}
	got, err := b.GenerateCached(prompt, 3, -1)
	if err != nil {
		t.Fatalf("GenerateCached after RestoreState: %v", err)
	}
	if headCalls != len(got)-1 {
		t.Fatalf("restored exact prompt-cache head calls = %d, want %d", headCalls, len(got)-1)
	}

	cold := newSessionStateFixture(t)
	want, err := cold.Generate(prompt, 3, -1)
	if err != nil {
		t.Fatalf("cold Generate: %v", err)
	}
	if len(got) != len(want) {
		t.Fatalf("generated length = %d, want %d", len(got), len(want))
	}
	for i := range got {
		if got[i] != want[i] {
			t.Fatalf("token %d after restored prompt-cache entry = %d, want %d", i, got[i], want[i])
		}
	}
}

func TestSessionStateRestorePreservesPromptCacheNoCopyBuffers(t *testing.T) {
	requireNativeRuntime(t)
	prompt := []int32{1, 2, 3, 4, 5}

	saved := newSessionStateFixture(t)
	if err := saved.WarmPromptCache(prompt); err != nil {
		t.Fatalf("WarmPromptCache: %v", err)
	}
	blob, err := saved.SerializeState()
	if err != nil {
		t.Fatalf("SerializeState: %v", err)
	}

	restored := newSessionStateFixture(t)
	if err := restored.RestoreState(blob); err != nil {
		t.Fatalf("RestoreState: %v", err)
	}
	if restored.cachedPromptHiddenBuffer() == nil {
		t.Fatal("RestoreState prompt-cache hidden did not restore a pinned no-copy buffer")
	}
	if restored.cachedPromptLogitsBuffer() == nil {
		t.Fatal("RestoreState prompt-cache logits did not restore a pinned no-copy buffer")
	}
	if restored.retainedHiddenBufferFor(restored.cachedPromptHidden) == nil {
		t.Fatal("RestoreState cached hidden is not reusable by retained-hidden consumers")
	}
	if restored.retainedLogitsBufferFor(restored.cachedPromptLogits) == nil {
		t.Fatal("RestoreState cached logits are not reusable by retained-logits consumers")
	}
}

func TestSessionStateRestorePromptCacheEntryAllocationBudget(t *testing.T) {
	requireNativeRuntime(t)
	prompt := []int32{1, 2, 3, 4, 5}

	saved := newSessionStateFixture(t)
	if err := saved.WarmPromptCache(prompt); err != nil {
		t.Fatalf("WarmPromptCache: %v", err)
	}
	blob, err := saved.SerializeState()
	if err != nil {
		t.Fatalf("SerializeState: %v", err)
	}

	restored := newSessionStateFixture(t)
	if err := restored.RestoreState(blob); err != nil {
		t.Fatalf("RestoreState warmup: %v", err)
	}
	var restoreErr error
	allocs := testing.AllocsPerRun(20, func() {
		restoreErr = restored.RestoreState(blob)
	})
	if restoreErr != nil {
		t.Fatalf("RestoreState: %v", restoreErr)
	}
	if allocs > 16 {
		t.Fatalf("RestoreState prompt-cache allocations = %.0f, want <= 16", allocs)
	}
	if restored.cachedPromptHiddenBuffer() == nil || restored.cachedPromptLogitsBuffer() == nil {
		t.Fatal("RestoreState allocation run dropped prompt-cache no-copy buffers")
	}
}

func TestSessionStateRestoreBlocksPreservesPromptCacheNoCopyBuffers(t *testing.T) {
	requireNativeRuntime(t)
	prompt := []int32{1, 2, 3, 4, 5}

	saved := newSessionStateFixture(t)
	if err := saved.WarmPromptCache(prompt); err != nil {
		t.Fatalf("WarmPromptCache: %v", err)
	}
	source, err := saved.StateBlockSource(2)
	if err != nil {
		t.Fatalf("StateBlockSource: %v", err)
	}

	restored := newSessionStateFixture(t)
	if err := restored.RestoreStateBlocks(source); err != nil {
		t.Fatalf("RestoreStateBlocks: %v", err)
	}
	if restored.cachedPromptHiddenBuffer() == nil {
		t.Fatal("RestoreStateBlocks prompt-cache hidden did not restore a pinned no-copy buffer")
	}
	if restored.cachedPromptLogitsBuffer() == nil {
		t.Fatal("RestoreStateBlocks prompt-cache logits did not restore a pinned no-copy buffer")
	}
	if restored.retainedHiddenBufferFor(restored.cachedPromptHidden) == nil {
		t.Fatal("RestoreStateBlocks cached hidden is not reusable by retained-hidden consumers")
	}
	if restored.retainedLogitsBufferFor(restored.cachedPromptLogits) == nil {
		t.Fatal("RestoreStateBlocks cached logits are not reusable by retained-logits consumers")
	}
}

// TestSessionStateRoundTrip proves native conversation continuity: a session is decoded, snapshotted
// with SerializeState, and the snapshot is RestoreState'd into a FRESH session — which then continues
// the conversation TOKEN-IDENTICALLY to the original. This is save/resume across a process restart with
// no cgo, the no-cgo equivalent of metal's EnableConversationContinuity.
func TestSessionStateRoundTrip(t *testing.T) {
	requireNativeRuntime(t)

	// session A: decode a first turn, then snapshot.
	a := newSessionStateFixture(t)
	if _, err := a.Generate([]int32{1, 2, 3, 4, 5}, 6, -1); err != nil {
		t.Fatalf("A turn 1: %v", err)
	}
	blob, err := a.SerializeState()
	if err != nil {
		t.Fatalf("SerializeState: %v", err)
	}

	// session B: fresh, restore A's snapshot.
	b := newSessionStateFixture(t)
	if err := b.RestoreState(blob); err != nil {
		t.Fatalf("RestoreState: %v", err)
	}
	if b.Pos() != a.Pos() {
		t.Fatalf("restored pos %d != saved pos %d", b.Pos(), a.Pos())
	}

	// both continue the conversation with the same next turn — must produce identical tokens.
	cont := []int32{20, 21, 22}
	genA, err := a.Generate(cont, 8, -1)
	if err != nil {
		t.Fatalf("A turn 2: %v", err)
	}
	genB, err := b.Generate(cont, 8, -1)
	if err != nil {
		t.Fatalf("B turn 2: %v", err)
	}
	if len(genA) != len(genB) {
		t.Fatalf("continuation length mismatch: A=%d B=%d", len(genA), len(genB))
	}
	for i := range genA {
		if genA[i] != genB[i] {
			t.Fatalf("token %d diverged after restore: A=%d B=%d", i, genA[i], genB[i])
		}
	}
	t.Logf("native continuity: serialize→restore→continue is token-identical over %d continuation tokens (snapshot %d bytes)", len(genA), len(blob))
}

// TestSessionStateRoundTripICBReplayReserialize pins the restore-side twin of the
// ICB extraction bug: an ICB session's live K/V lives in the ICB cache buffers and
// its paged caches are dormant, but RestoreState used to take the paged branch for
// any layer with a device paged cache — so the blob landed in pages decode never
// reads, the ICB buffers stayed zeroed, and re-serialising the "restored" session
// exported an EMPTY conversation (save → restore → save silently lost the state;
// the continuation also decoded against zero history). Restore must write the same
// store SerializeState reads: byte-identical re-serialisation is the contract.
func TestSessionStateRoundTripICBReplayReserialize(t *testing.T) {
	requireNativeRuntime(t)
	g, arch, maxLen := icbSessionStateFixture(t)

	a := newICBSessionStateFixture(t, g, arch, maxLen)
	if _, err := a.Generate([]int32{1, 5, 3, 2}, 4, -1); err != nil {
		t.Fatalf("A turn 1: %v", err)
	}
	blob, err := a.SerializeState()
	if err != nil {
		t.Fatalf("SerializeState: %v", err)
	}

	b := newICBSessionStateFixture(t, g, arch, maxLen)
	if err := b.RestoreState(blob); err != nil {
		t.Fatalf("RestoreState: %v", err)
	}
	blob2, err := b.SerializeState()
	if err != nil {
		t.Fatalf("re-serialise restored session: %v", err)
	}
	if !bytes.Equal(blob, blob2) {
		nz := func(bs []byte) int {
			n := 0
			for _, x := range bs {
				if x != 0 {
					n++
				}
			}
			return n
		}
		t.Fatalf("restore → serialize is not lossless: %d vs %d bytes, %d vs %d nonzero — the restored session's live KV store does not hold the snapshot",
			len(blob), len(blob2), nz(blob), nz(blob2))
	}
}

func TestSessionStateRoundTripICBReplay(t *testing.T) {
	requireNativeRuntime(t)
	g, arch, maxLen := icbSessionStateFixture(t)

	a := newICBSessionStateFixture(t, g, arch, maxLen)
	if _, err := a.Generate([]int32{1, 5, 3, 2}, 4, -1); err != nil {
		t.Fatalf("A turn 1: %v", err)
	}
	blob, err := a.SerializeState()
	if err != nil {
		t.Fatalf("SerializeState ICB: %v", err)
	}

	b := newICBSessionStateFixture(t, g, arch, maxLen)
	if err := b.RestoreState(blob); err != nil {
		t.Fatalf("RestoreState ICB: %v", err)
	}
	if b.Pos() != a.Pos() {
		t.Fatalf("restored ICB pos %d != saved pos %d", b.Pos(), a.Pos())
	}

	cont := []int32{7, 8}
	genA, err := a.Generate(cont, 5, -1)
	if err != nil {
		t.Fatalf("A turn 2: %v", err)
	}
	genB, err := b.Generate(cont, 5, -1)
	if err != nil {
		t.Fatalf("B turn 2: %v", err)
	}
	if len(genA) != len(genB) {
		t.Fatalf("ICB continuation length mismatch: A=%d B=%d", len(genA), len(genB))
	}
	for i := range genA {
		if genA[i] != genB[i] {
			t.Fatalf("ICB token %d diverged after restore: A=%d B=%d", i, genA[i], genB[i])
		}
	}
}

// TestSessionStateRoundTripRestoresCachedPrefixMetadata proves state restore
// preserves the prompt-cache metadata that lets GenerateCached reuse resident KV
// rows. Token parity alone is insufficient here: a restored session can produce
// the same tokens by cold re-prefilling, but then the native engine has lost the
// resource-saving prefix hit that metal's prompt-cache restore path preserves.
func TestSessionStateRoundTripRestoresCachedPrefixMetadata(t *testing.T) {
	requireNativeRuntime(t)
	a := newSessionStateFixture(t)
	prompt := []int32{1, 2, 3, 4, 5}
	if _, err := a.GenerateCached(prompt, 6, -1); err != nil {
		t.Fatalf("GenerateCached warmup: %v", err)
	}
	nextPrompt := []int32{1, 2, 3, 4, 5, 6}
	wantHit := a.CachedPrefixLen(nextPrompt)
	if wantHit != len(prompt) {
		t.Fatalf("warm CachedPrefixLen = %d, want %d", wantHit, len(prompt))
	}
	blob, err := a.SerializeState()
	if err != nil {
		t.Fatalf("SerializeState: %v", err)
	}

	b := newSessionStateFixture(t)
	if err := b.RestoreState(blob); err != nil {
		t.Fatalf("RestoreState: %v", err)
	}
	if got := b.CachedPrefixLen(nextPrompt); got != wantHit {
		t.Fatalf("restored CachedPrefixLen = %d, want %d", got, wantHit)
	}
}

func TestSessionStateSerializeCachedPrefixAllocationBudget(t *testing.T) {
	requireNativeRuntime(t)
	s := newSessionStateFixture(t)
	if _, err := s.GenerateCached([]int32{1, 2, 3, 4, 5}, 6, -1); err != nil {
		t.Fatalf("GenerateCached warmup: %v", err)
	}
	allocs := testing.AllocsPerRun(20, func() {
		if _, err := s.SerializeState(); err != nil {
			t.Fatalf("SerializeState: %v", err)
		}
	})
	if allocs > 82 {
		t.Fatalf("SerializeState allocations = %.0f, want <= 82", allocs)
	}
}

func TestSessionStateRangeBlocksAllocationBudget(t *testing.T) {
	requireNativeRuntime(t)
	s := newSessionStateFixture(t)
	if _, err := s.GenerateCached([]int32{1, 2, 3, 4, 5}, 6, -1); err != nil {
		t.Fatalf("GenerateCached warmup: %v", err)
	}
	if err := s.RangeStateBlocks(2, func(block SessionStateBlock) (bool, error) {
		return true, nil
	}); err != nil {
		t.Fatalf("RangeStateBlocks warmup: %v", err)
	}
	allocs := testing.AllocsPerRun(20, func() {
		if err := s.RangeStateBlocks(2, func(block SessionStateBlock) (bool, error) {
			if block.TokenCount == 0 {
				t.Fatal("empty block")
			}
			return true, nil
		}); err != nil {
			t.Fatalf("RangeStateBlocks: %v", err)
		}
	})
	if allocs > 0 {
		t.Fatalf("RangeStateBlocks allocations = %.0f, want 0", allocs)
	}
}

func TestSessionStateLayerViewsRefreshAllocationBudget(t *testing.T) {
	requireNativeRuntime(t)
	s := newSessionStateFixture(t)
	if _, err := s.GenerateCached([]int32{1, 2, 3, 4, 5}, 1, -1); err != nil {
		t.Fatalf("GenerateCached warmup: %v", err)
	}
	if _, err := s.stateLayerViews(); err != nil {
		t.Fatalf("stateLayerViews warmup: %v", err)
	}
	icb := s.state.icb != nil
	allocs := testing.AllocsPerRun(20, func() {
		s.stateBlockViewsICB = !icb
		views, err := s.stateLayerViews()
		if err != nil {
			t.Fatalf("stateLayerViews: %v", err)
		}
		if len(views) == 0 {
			t.Fatal("stateLayerViews returned no owner views")
		}
	})
	if allocs > 0 {
		t.Fatalf("stateLayerViews refresh allocations = %.0f, want 0", allocs)
	}
}

func TestSessionStateRestoreBlocksAllocationBudget(t *testing.T) {
	requireNativeRuntime(t)
	saved := newSessionStateFixture(t)
	prompt := []int32{1, 2, 3, 4, 5}
	if err := saved.WarmPromptCache(prompt); err != nil {
		t.Fatalf("WarmPromptCache: %v", err)
	}
	source, err := saved.StateBlockSource(2)
	if err != nil {
		t.Fatalf("StateBlockSource: %v", err)
	}
	restored := newSessionStateFixture(t)
	if err := restored.RestoreStateBlocks(source); err != nil {
		t.Fatalf("RestoreStateBlocks warmup: %v", err)
	}
	allocs := testing.AllocsPerRun(20, func() {
		if err := restored.RestoreStateBlocks(source); err != nil {
			t.Fatalf("RestoreStateBlocks: %v", err)
		}
	})
	if allocs > 0 {
		t.Fatalf("RestoreStateBlocks allocations = %.0f, want 0", allocs)
	}
}
