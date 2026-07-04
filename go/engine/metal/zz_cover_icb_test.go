// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"sort"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference/model"
	"github.com/tmc/apple/metal"
)

// zz_cover_icb_test.go closes the per-pipeline error legs in the ICB recorders
// (DecodeLayerICB, AttentionBlockICB, DecodeTokenICB and the decode_forward_*ICB
// cores). Each recorder builds ~10 ICB-capable pipelines in sequence, every one
// guarded by `if err != nil { return nil, err }`. The existing guard suite nulls
// the library so the FIRST builder fails and the rest are unreachable. Here the
// whole ICB pipeline cache is warmed by a real successful call, then exactly ONE
// cache entry is evicted while the library is nulled — so the recorder reaches
// that builder's call site (all earlier kernels still cached), the lone uncached
// build hits `library == nil`, and the error leg at that specific line fires.
//
// pipelineForICB / ropePipelineICB / sdpaVectorPipelineICB / geluPipelineICB all
// share icbPSOCache and check the cache BEFORE the library, which is what makes
// the single-key eviction land on exactly one call site.

// icbCacheSnapshot copies the current ICB pipeline cache (under its mutex).
func icbCacheSnapshot() map[string]metal.MTLComputePipelineState {
	icbPSOMu.Lock()
	defer icbPSOMu.Unlock()
	out := make(map[string]metal.MTLComputePipelineState, len(icbPSOCache))
	for k, v := range icbPSOCache {
		out[k] = v
	}
	return out
}

// icbCacheKeys returns the cache keys, sorted, for diagnostics.
func icbCacheKeys() []string {
	icbPSOMu.Lock()
	defer icbPSOMu.Unlock()
	ks := make([]string, 0, len(icbPSOCache))
	for k := range icbPSOCache {
		ks = append(ks, k)
	}
	sort.Strings(ks)
	return ks
}

// withICBKeyEvicted restores the full ICB cache + library from a warmed snapshot,
// drops the single key whose builder leg we want to fail, nulls the library, runs
// invoke (which must error at that builder's call site), then restores everything.
// snap is a warmed cache snapshot; the live library is captured here and restored.
func withICBKeyEvicted(t *testing.T, snap map[string]metal.MTLComputePipelineState, key string, invoke func() error) {
	t.Helper()
	if _, ok := snap[key]; !ok {
		t.Fatalf("ICB key %q not in warmed cache; keys=%v", key, icbCacheKeys())
	}
	oldLib, oldCustom := library, customLibrary
	// install the snapshot minus the one key.
	icbPSOMu.Lock()
	icbPSOCache = make(map[string]metal.MTLComputePipelineState, len(snap))
	for k, v := range snap {
		if k == key {
			continue
		}
		icbPSOCache[k] = v
	}
	sdpaVectorICBHeadDimPSOCache = map[int]metal.MTLComputePipelineState{}
	icbPSOMu.Unlock()
	// Null BOTH libraries: the gemm/rope/sdpa/elementwise pipelines resolve from the
	// main library, the fused-gelu pipeline from customLibrary — nulling both makes
	// the single evicted key's rebuild fail regardless of which library it uses.
	library, customLibrary = nil, nil

	err := invoke()

	// restore the full cache + libraries before asserting, so a failed assertion
	// never leaves the package poisoned for later files.
	library, customLibrary = oldLib, oldCustom
	icbPSOMu.Lock()
	icbPSOCache = make(map[string]metal.MTLComputePipelineState, len(snap))
	for k, v := range snap {
		icbPSOCache[k] = v
	}
	icbPSOMu.Unlock()

	if err == nil {
		t.Fatalf("evicting ICB key %q: expected error, got nil", key)
	}
}

// findICBKey returns the single cache key (from a warmed snapshot) matching pred,
// failing if zero or more than one match — so a leg target is unambiguous.
func findICBKey(t *testing.T, snap map[string]metal.MTLComputePipelineState, what string, pred func(string) bool) string {
	t.Helper()
	var hits []string
	for k := range snap {
		if pred(k) {
			hits = append(hits, k)
		}
	}
	sort.Strings(hits)
	if len(hits) != 1 {
		t.Fatalf("ICB key match for %s: want exactly 1, got %d: %v", what, len(hits), hits)
	}
	return hits[0]
}

func hasPrefix(s, p string) bool { return len(s) >= len(p) && s[:len(p)] == p }

// TestCoverAttentionBlockICBPipelineLegs covers the per-builder error legs in
// AttentionBlockICB (gemvQ, gemvO, rope, sdpa, add). Dims are chosen so qDim
// (128) differs from dModel (64): the Q gemv (64→128) and O gemv (128→64) then
// have DISTINCT tile keys, so each can be evicted independently.
func TestCoverAttentionBlockICBPipelineLegs(t *testing.T) {
	requireNativeRuntime(t)

	// dModel=64 (k<=64 ⇒ small-k tile) and qDim=256 (k>64 ⇒ standard tile) give the
	// Q gemv (64→256) and O gemv (256→64) DISTINCT tile keys, so each gemv leg can be
	// evicted independently. headDim=64 so the sdpa_vector_bfloat16_t_64_64 kernel exists.
	const dModel, nHeads, nKV, headDim, kvLen = 64, 4, 2, 64, 4
	const eps = float32(1e-6)
	qDim := nHeads * headDim // 256, != dModel
	x := toBF16Bytes(syntheticFloat32(dModel, 1))
	normW := toBF16Bytes(syntheticFloat32(dModel, 3))
	wQ := toBF16Bytes(syntheticFloat32(qDim*dModel, 5))
	wO := toBF16Bytes(syntheticFloat32(dModel*qDim, 7))
	kCache := toBF16Bytes(syntheticFloat32(nKV*kvLen*headDim, 9))
	vCache := toBF16Bytes(syntheticFloat32(nKV*kvLen*headDim, 11))

	invoke := func() error {
		_, e := AttentionBlockICB(x, normW, wQ, wO, kCache, vCache, dModel, nHeads, nKV, headDim, kvLen, 10000, 0.125, 0, eps, 1)
		return e
	}
	// clear first so the snapshot holds exactly this call's keys (the cache is global,
	// stragglers from other tests would make findICBKey ambiguous), then warm.
	icbPSOMu.Lock()
	icbPSOCache = map[string]metal.MTLComputePipelineState{}
	sdpaVectorICBHeadDimPSOCache = map[int]metal.MTLComputePipelineState{}
	icbPSOMu.Unlock()
	if err := invoke(); err != nil {
		t.Fatalf("warm AttentionBlockICB: %v", err)
	}
	snap := icbCacheSnapshot()

	gemvQ := findICBKey(t, snap, "gemvQ (dModel->qDim)", func(k string) bool {
		return hasPrefix(k, "gemv_bfloat16_") && gemvKeyShape(k, dModel, qDim)
	})
	gemvO := findICBKey(t, snap, "gemvO (qDim->dModel)", func(k string) bool {
		return hasPrefix(k, "gemv_bfloat16_") && gemvKeyShape(k, qDim, dModel)
	})
	rope := findICBKey(t, snap, "rope", func(k string) bool { return hasPrefix(k, "rope_single_bfloat16|icb") })
	sdpa := findICBKey(t, snap, "sdpa", func(k string) bool { return hasPrefix(k, "sdpa_vector_bfloat16_t_") })
	add := findICBKey(t, snap, "add", func(k string) bool { return k == "vv_Addbfloat16" })

	for _, key := range []string{gemvQ, gemvO, rope, sdpa, add} {
		withICBKeyEvicted(t, snap, key, invoke)
	}
}

// coverICBEvictAll warms the ICB cache with a successful invoke, then evicts each
// distinct cached key one at a time (library nulled) so every unique pipeline's
// `if err != nil` leg fires at its first call site. gemv sites whose tile keys
// collide (the tile function has only a few regimes) share a key, so the first
// such site is covered and the later identical-code siblings are not — that is the
// inherent ceiling of key-eviction, accepted here.
func coverICBEvictAll(t *testing.T, invoke func() error) {
	t.Helper()
	// Clear the ICB cache first so the warmed snapshot contains EXACTLY the keys this
	// invoke builds — not stragglers warmed by earlier tests (the cache is global).
	// Evicting a key the invoke never touches would not trigger a rebuild and the
	// expected error would never fire. The cache is pure memoisation, safe to clear.
	icbPSOMu.Lock()
	icbPSOCache = map[string]metal.MTLComputePipelineState{}
	sdpaVectorICBHeadDimPSOCache = map[int]metal.MTLComputePipelineState{}
	icbPSOMu.Unlock()
	if err := invoke(); err != nil {
		t.Fatalf("warm: %v", err)
	}
	snap := icbCacheSnapshot()
	if len(snap) == 0 {
		t.Fatal("ICB cache empty after warm")
	}
	keys := make([]string, 0, len(snap))
	for k := range snap {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, key := range keys {
		withICBKeyEvicted(t, snap, key, invoke)
	}
}

// TestCoverDecodeLayerICBPipelineLegs covers the per-builder error legs in
// DecodeLayerICB by evicting each warmed ICB key in turn. Dims give THREE distinct
// gemv tile keys — small-k (Q/F: k=64), standard (O: k=qDim=256) and huge-k (D:
// k=dFF=1024 >= 16*dModel ⇒ the bm=1,bn=8 regime) — plus the unique
// rope/sdpa/add/mul/tanh/gelu keys, so the down-proj gemv leg separates from O.
func TestCoverDecodeLayerICBPipelineLegs(t *testing.T) {
	requireNativeRuntime(t)

	const dModel, nHeads, nKV, headDim, kvLen, dFF = 64, 4, 2, 64, 4, 1024
	const eps = float32(1e-6)
	qDim := nHeads * headDim
	x := toBF16Bytes(syntheticFloat32(dModel, 1))
	attnNormW := toBF16Bytes(syntheticFloat32(dModel, 3))
	mlpNormW := toBF16Bytes(syntheticFloat32(dModel, 5))
	wQ := toBF16Bytes(syntheticFloat32(qDim*dModel, 7))
	wO := toBF16Bytes(syntheticFloat32(dModel*qDim, 9))
	wGate := toBF16Bytes(syntheticFloat32(dFF*dModel, 11))
	wUp := toBF16Bytes(syntheticFloat32(dFF*dModel, 13))
	wDown := toBF16Bytes(syntheticFloat32(dModel*dFF, 15))
	kCache := toBF16Bytes(syntheticFloat32(nKV*kvLen*headDim, 17))
	vCache := toBF16Bytes(syntheticFloat32(nKV*kvLen*headDim, 19))

	coverICBEvictAll(t, func() error {
		_, e := DecodeLayerICB(x, attnNormW, wQ, wO, kCache, vCache, mlpNormW, wGate, wUp, wDown,
			dModel, nHeads, nKV, headDim, kvLen, dFF, 10000, 0.125, 0, eps, 1)
		return e
	})
}

// TestCoverDecodeTokenICBPipelineLegs is the multi-layer sibling: DecodeTokenICB
// builds the same pipeline set, so evicting each warmed key covers its leg here
// too (distinct lines from DecodeLayerICB in the coverage profile).
func TestCoverDecodeTokenICBPipelineLegs(t *testing.T) {
	requireNativeRuntime(t)

	const dModel, nHeads, nKV, headDim, kvLen, dFF, nLayers = 64, 4, 2, 64, 4, 1024, 2
	const eps = float32(1e-6)
	qDim := nHeads * headDim
	x := toBF16Bytes(syntheticFloat32(dModel, 1))
	attnNormW := toBF16Bytes(syntheticFloat32(dModel, 3))
	mlpNormW := toBF16Bytes(syntheticFloat32(dModel, 5))
	wQ := toBF16Bytes(syntheticFloat32(qDim*dModel, 7))
	wO := toBF16Bytes(syntheticFloat32(dModel*qDim, 9))
	wGate := toBF16Bytes(syntheticFloat32(dFF*dModel, 11))
	wUp := toBF16Bytes(syntheticFloat32(dFF*dModel, 13))
	wDown := toBF16Bytes(syntheticFloat32(dModel*dFF, 15))
	kCache := toBF16Bytes(syntheticFloat32(nKV*kvLen*headDim, 17))
	vCache := toBF16Bytes(syntheticFloat32(nKV*kvLen*headDim, 19))

	coverICBEvictAll(t, func() error {
		_, e := DecodeTokenICB(x, attnNormW, wQ, wO, kCache, vCache, mlpNormW, wGate, wUp, wDown,
			dModel, nHeads, nKV, headDim, kvLen, dFF, nLayers, 10000, 0.125, 0, eps, 1)
		return e
	})
}

// TestCoverDecodeForwardICBPipelineLegs covers the gemv-recorder legs in
// DecodeForwardICB and the shared-pipeline legs in decodeForwardICBCore by
// evicting each warmed ICB key. Two distinct gemv tile keys (small-k Q/KV/F vs
// standard O/D) plus the unique rope/sdpa/add/mul/tanh/gelu keys.
func TestCoverDecodeForwardICBPipelineLegs(t *testing.T) {
	requireNativeRuntime(t)

	const dModel, nHeads, nKV, headDim, dFF, maxLen = 64, 4, 2, 64, 256, 4
	const base, scale, eps = float32(10000), float32(0.125), float32(1e-5)
	inputs := decodeInputsFixture(2, dModel)
	layers := []DecodeLayerWeights{decodeLayerFixture(dModel, nHeads, nKV, headDim, dFF, 3)}

	coverICBEvictAll(t, func() error {
		_, e := DecodeForwardICB(inputs, layers, dModel, nHeads, nKV, headDim, maxLen, dFF, base, scale, eps)
		return e
	})
}

// TestCoverDecodeForwardICBQuantPipelineLegs covers the qmv-recorder legs in
// DecodeForwardICBQuant and its shared-pipeline legs by eviction. The small dims
// keep all qmv shapes on the single slow-qmv tile key, so the first qmv leg is
// covered and the identical-code siblings hit the same key.
func TestCoverDecodeForwardICBQuantPipelineLegs(t *testing.T) {
	requireNativeRuntime(t)

	// dModel=512 (inDim 512 ⇒ the _qmv_fast_ variant for Q/KV/O/gate/up) and dFF=256
	// (the down proj's inDim 256 is NOT a multiple of 512 ⇒ the slow _qmv_ variant) so
	// TWO distinct qmv tile keys warm, letting the first fast and the first slow qmv
	// leg both be covered by eviction.
	const dModel, nHeads, nKV, headDim, dFF, maxLen = 512, 4, 2, 64, 256, 4
	const gs, bits = 64, 4
	const base, scale, eps = float32(10000), float32(0.125), float32(1e-5)
	inputs := decodeInputsFixture(2, dModel)
	qlayers := []QuantizedLayerWeights{quantizedLayerFixture(t, dModel, nHeads, nKV, headDim, dFF, gs, bits, 3)}

	coverICBEvictAll(t, func() error {
		_, e := DecodeForwardICBQuant(inputs, qlayers, dModel, nHeads, nKV, headDim, maxLen, dFF, base, scale, eps)
		return e
	})
}

// TestCoverDecodeForwardArchICBPipelineLegs covers the gemvPSO-recorder legs in
// DecodeForwardArchICB and the shared-pipeline legs in its core by evicting each
// warmed ICB key. Uses the proven arch dims (all-global 2-layer arch).
func TestCoverDecodeForwardArchICBPipelineLegs(t *testing.T) {
	requireNativeRuntime(t)

	const dModel, nHeads, nKV, headDim, dFF, maxLen = 512, 8, 4, 64, 1024, 8
	const base, scale, eps = float32(10000), float32(0.125), float32(1e-5)
	specs := model.DeriveLayers([]string{"full_attention", "full_attention"}, 0)
	layers := []DecodeLayerWeights{
		forwardLayer(dModel, nHeads, nKV, headDim, dFF, 100),
		forwardLayer(dModel, nHeads, nKV, headDim, dFF, 200),
	}
	inputs := decodeInputsFixture(2, dModel)

	coverICBEvictAll(t, func() error {
		_, e := DecodeForwardArchICB(inputs, layers, specs, dModel, nHeads, nKV, headDim, maxLen, dFF, 0, base, scale, eps, false)
		return e
	})
}

// TestCoverDecodeForwardArchICBQuantPipelineLegs covers the qmvPSO-recorder legs
// in DecodeForwardArchICBQuant and its shared-pipeline legs by eviction.
func TestCoverDecodeForwardArchICBQuantPipelineLegs(t *testing.T) {
	requireNativeRuntime(t)

	// dFF=256 (not a multiple of 512) so the down proj takes the slow _qmv_ variant
	// while the dModel=512-fed projections take _qmv_fast_ — two qmv keys, so the
	// first fast and first slow qmv leg are both covered by eviction.
	const dModel, nHeads, nKV, headDim, dFF, maxLen = 512, 8, 4, 64, 256, 8
	const gs, bits = 64, 4
	const base, scale, eps = float32(10000), float32(0.125), float32(1e-5)
	specs := model.DeriveLayers([]string{"full_attention", "full_attention"}, 0)
	ql := []QuantizedLayerWeights{
		coverQuantLayer(t, dModel, nHeads, nKV, headDim, dFF, gs, bits, 100),
		coverQuantLayer(t, dModel, nHeads, nKV, headDim, dFF, gs, bits, 200),
	}
	inputs := decodeInputsFixture(2, dModel)

	coverICBEvictAll(t, func() error {
		_, e := DecodeForwardArchICBQuant(inputs, ql, specs, dModel, nHeads, nKV, headDim, maxLen, dFF, 0, base, scale, eps, false)
		return e
	})
}

// gemvKeyShape reports whether the gemv ICB cache key's tiles match gemvTiles for
// the (inDim,outDim) shape — letting a test target a specific gemv by its shape
// rather than by re-deriving the tile string.
func gemvKeyShape(key string, inDim, outDim int) bool {
	bm, bn, sm, sn, tm, tn := gemvTiles(inDim, outDim)
	want := sprintfGemvKey(bm, bn, sm, sn, tm, tn)
	return key == want
}

func sprintfGemvKey(bm, bn, sm, sn, tm, tn int) string {
	return core.Sprintf("gemv_bfloat16_bm%d_bn%d_sm%d_sn%d_tm%d_tn%d_nc0_axpby0", bm, bn, sm, sn, tm, tn)
}

func coverQuantLayer(tb testing.TB, dModel, nHeads, nKV, headDim, dFF, groupSize, bits, salt int) QuantizedLayerWeights {
	tb.Helper()
	qDim, kvDim := nHeads*headDim, nKV*headDim
	return QuantizedLayerWeights{
		AttnNormW: toBF16Bytes(syntheticFloat32(dModel, salt+13)),
		MLPNormW:  toBF16Bytes(syntheticFloat32(dModel, salt+19)),
		Q:         quantWeightFixture(tb, qDim, dModel, groupSize, bits, salt+53),
		K:         quantWeightFixture(tb, kvDim, dModel, groupSize, bits, salt+71),
		V:         quantWeightFixture(tb, kvDim, dModel, groupSize, bits, salt+83),
		O:         quantWeightFixture(tb, dModel, qDim, groupSize, bits, salt+17),
		Gate:      quantWeightFixture(tb, dFF, dModel, groupSize, bits, salt+61),
		Up:        quantWeightFixture(tb, dFF, dModel, groupSize, bits, salt+29),
		Down:      quantWeightFixture(tb, dModel, dFF, groupSize, bits, salt+47),
		GroupSize: groupSize,
		Bits:      bits,
	}
}
