// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"maps"
	"sort"
	"testing"

	"dappco.re/go/inference/model"
	"github.com/tmc/apple/metal"
)

// zz_cover_encode_test.go closes the per-op error legs INSIDE the single-command
// encoders (MLPBlockBF16, DecodeStepKV, AttentionStepKV, the MoE blocks,
// PerLayerInputs, the measure / chain / head_nocopy entries). Each builds its
// pipelines up front, opens a command encoder, and guards every encode step with
// `if encErr = encXxx(...); encErr != nil { enc.EndEncoding(); return }`. The
// per-op guard suite nulls the library, so the FIRST pipeline build fails before
// any encode runs — those legs are unreachable that way.
//
// The same single-key cache-eviction used for the ICB recorders works here on the
// NON-ICB caches: warm the whole op (a real successful call), then evict exactly
// one warmed pipeline key (across psoCache / ropePSOCache / ropePSOBF16Cache /
// ropeFreqsPSOBF16Cache / sdpaPSOCache) with the library nulled — so every
// earlier kernel is still cached and the lone evicted kernel's rebuild hits
// `library == nil`, surfacing through encXxx into the encode-step error leg at its
// call site. Evicting EVERY distinct warmed key in turn harvests exactly the
// independently-reachable legs (collision-siblings share a key and are skipped
// for free).

// the five non-ICB pipeline caches, snapshotted/cleared/restored as a set.
type psoCaches struct {
	plain    map[string]metal.MTLComputePipelineState
	rope     map[string]metal.MTLComputePipelineState
	ropeBF16 map[string]metal.MTLComputePipelineState
	freqs    map[string]metal.MTLComputePipelineState
	sdpa     map[string]metal.MTLComputePipelineState
}

func snapshotPSOCaches() psoCaches {
	cp := func(mu muLocker, m map[string]metal.MTLComputePipelineState) map[string]metal.MTLComputePipelineState {
		mu.Lock()
		defer mu.Unlock()
		out := make(map[string]metal.MTLComputePipelineState, len(m))
		maps.Copy(out, m)
		return out
	}
	return psoCaches{
		plain:    cp(&psoMu, psoCache),
		rope:     cp(&ropePSOMu, ropePSOCache),
		ropeBF16: cp(&ropePSOBF16Mu, ropePSOBF16Cache),
		freqs:    cp(&ropeFreqsPSOBF16Mu, ropeFreqsPSOBF16Cache),
		sdpa:     cp(&sdpaPSOMu, sdpaPSOCache),
	}
}

// muLocker abstracts *sync.Mutex so the snapshot/restore helpers can take a
// pointer to each cache's mutex uniformly.
type muLocker = interface {
	Lock()
	Unlock()
}

// installPSOCaches overwrites all five caches with the given snapshot (optionally
// dropping one key from whichever cache holds it).
func installPSOCaches(s psoCaches, dropKey string) {
	put := func(mu muLocker, dst *map[string]metal.MTLComputePipelineState, src map[string]metal.MTLComputePipelineState) {
		mu.Lock()
		defer mu.Unlock()
		m := make(map[string]metal.MTLComputePipelineState, len(src))
		for k, v := range src {
			if k == dropKey {
				continue
			}
			m[k] = v
		}
		*dst = m
	}
	put(&psoMu, &psoCache, s.plain)
	put(&ropePSOMu, &ropePSOCache, s.rope)
	put(&ropePSOBF16Mu, &ropePSOBF16Cache, s.ropeBF16)
	put(&ropeFreqsPSOBF16Mu, &ropeFreqsPSOBF16Cache, s.freqs)
	put(&sdpaPSOMu, &sdpaPSOCache, s.sdpa)
}

// allPSOKeys returns every key across the five caches in a snapshot, sorted.
func allPSOKeys(s psoCaches) []string {
	var ks []string
	for _, m := range []map[string]metal.MTLComputePipelineState{s.plain, s.rope, s.ropeBF16, s.freqs, s.sdpa} {
		for k := range m {
			ks = append(ks, k)
		}
	}
	sort.Strings(ks)
	return ks
}

// coverEncodeEvictAll warms invoke, then for each distinct warmed non-ICB pipeline
// key evicts it (library nulled) so the encode-step error leg at its call site
// fires. Library + caches are restored inline before each assert, so a t.Fatal
// never poisons a later test file.
func coverEncodeEvictAll(t *testing.T, invoke func() error) {
	t.Helper()
	// clear all five caches so the warmed snapshot is exactly this invoke's keys.
	clearPSOCaches()
	if err := invoke(); err != nil {
		t.Fatalf("warm: %v", err)
	}
	snap := snapshotPSOCaches()
	keys := allPSOKeys(snap)
	if len(keys) == 0 {
		t.Fatal("no non-ICB pipelines warmed")
	}
	oldLib := library
	errored := 0
	for _, key := range keys {
		installPSOCaches(snap, key)
		library = nil
		err := invoke()
		library = oldLib
		installPSOCaches(snap, "")
		// A warmed key that the invoke does NOT rebuild on its critical path (a
		// collision sibling, or a conditionally-taken branch like the composed-vs-
		// fused gelu) yields no error on eviction — that is expected, not a failure.
		// What the test asserts is that whenever a load-bearing pipeline fails, the
		// op surfaces the error instead of panicking or returning a nil-err buffer.
		if err != nil {
			errored++
		}
	}
	if errored == 0 {
		t.Fatal("no evicted pipeline produced an error — the eviction mechanism did not bite this op")
	}
}

func clearPSOCaches() {
	psoMu.Lock()
	psoCache = map[string]metal.MTLComputePipelineState{}
	psoMu.Unlock()
	ropePSOMu.Lock()
	ropePSOCache = map[string]metal.MTLComputePipelineState{}
	ropePSOMu.Unlock()
	ropePSOBF16Mu.Lock()
	ropePSOBF16Cache = map[string]metal.MTLComputePipelineState{}
	ropePSOBF16Mu.Unlock()
	ropeFreqsPSOBF16Mu.Lock()
	ropeFreqsPSOBF16Cache = map[string]metal.MTLComputePipelineState{}
	ropeFreqsPSOBF16Mu.Unlock()
	sdpaPSOMu.Lock()
	sdpaPSOCache = map[string]metal.MTLComputePipelineState{}
	sdpaPSOMu.Unlock()
}

// coverEncodeEvictAllComposed is coverEncodeEvictAll with the fused-gelu kernel
// disabled, so ops take the COMPOSED bf16 gelu chain (the tanh/add/mul primitive
// sequence) and that chain's downstream error legs become reachable by eviction.
func coverEncodeEvictAllComposed(t *testing.T, invoke func() error) {
	t.Helper()
	old := customLibraryLoaded
	customLibraryLoaded = false
	defer func() { customLibraryLoaded = old }()
	coverEncodeEvictAll(t, invoke)
}

// TestCoverGeluComposedEncodeLegs covers the composed-gelu chain downstream legs
// in GeluBF16 / Gelu (the tanh / add / mul steps after the initial loop) by
// forcing the composed path and evicting each warmed primitive key.
func TestCoverGeluComposedEncodeLegs(t *testing.T) {
	requireNativeRuntime(t)

	xb := toBF16Bytes(syntheticFloat32(32, 3))
	x32 := syntheticFloat32(32, 5)
	coverEncodeEvictAllComposed(t, func() error {
		_, e := GeluBF16(xb)
		return e
	})
	coverEncodeEvictAllComposed(t, func() error {
		_, e := Gelu(x32)
		return e
	})
	// GeluGateMulBF16's composed path: gelu(gate) then a binary multiply by up.
	up := toBF16Bytes(syntheticFloat32(32, 7))
	coverEncodeEvictAllComposed(t, func() error {
		_, e := GeluGateMulBF16(xb, up)
		return e
	})
}

// TestCoverMoEBlockComposedEncodeLegs re-covers the MoE blocks with the composed
// gelu path so the mlpTransform composed-gelu legs (skipped when the fused kernel
// is used) become reachable by eviction.
func TestCoverMoEBlockComposedEncodeLegs(t *testing.T) {
	requireNativeRuntime(t)

	const dModel, dFF, expertDFF = 64, 256, 256
	const gs, bits = 64, 4
	const eps = float32(1e-6)
	wBF := moeLayerWeightsFixture(2, 2, dModel, dFF, expertDFF, 3)
	wQ := quantMoELayerWeightsGuard(t, 2, 2, dModel, dFF, expertDFF, gs, bits)
	h := toBF16Bytes(syntheticFloat32(dModel, 1))

	coverEncodeEvictAllComposed(t, func() error {
		_, e := MoEBlockBF16(h, wBF, dModel, dFF, eps)
		return e
	})
	coverEncodeEvictAllComposed(t, func() error {
		_, e := MoEBlockQuant(h, wQ, dModel, dFF, eps)
		return e
	})
}

// TestCoverMLPBlockBF16EncodeLegs covers the encode-step error legs in
// MLPBlockBF16 (the rms / gate-gemv / down-gemv / residual-add steps, plus the
// post-gelu error check) via single-key eviction.
func TestCoverMLPBlockBF16EncodeLegs(t *testing.T) {
	requireNativeRuntime(t)

	const dModel, dFF = 64, 256
	const eps = float32(1e-6)
	x := toBF16Bytes(syntheticFloat32(dModel, 1))
	normW := toBF16Bytes(syntheticFloat32(dModel, 3))
	wGate := toBF16Bytes(syntheticFloat32(dFF*dModel, 5))
	wUp := toBF16Bytes(syntheticFloat32(dFF*dModel, 7))
	wDown := toBF16Bytes(syntheticFloat32(dModel*dFF, 9))

	coverEncodeEvictAll(t, func() error {
		_, e := MLPBlockBF16(x, normW, wGate, wUp, wDown, dModel, dFF, eps)
		return e
	})
}

// TestCoverAttentionStepKVEncodeLegs covers the encAttnHalfKV error leg in
// AttentionStepKV via single-key eviction.
func TestCoverAttentionStepKVEncodeLegs(t *testing.T) {
	requireNativeRuntime(t)

	const dModel, nHeads, nKV, headDim, maxLen, pos, dFF = 64, 4, 2, 64, 4, 0, 256
	const base, scale, eps = float32(10000), float32(0.125), float32(1e-6)
	layer := decodeLayerFixture(dModel, nHeads, nKV, headDim, dFF, 3)
	x := toBF16Bytes(syntheticFloat32(dModel, 1))
	kCache := make([]byte, nKV*maxLen*headDim*bf16Size)
	vCache := make([]byte, nKV*maxLen*headDim*bf16Size)

	coverEncodeEvictAll(t, func() error {
		_, e := AttentionStepKV(x, layer.AttnNormW, layer.WQ, layer.WK, layer.WV, layer.WO, kCache, vCache,
			dModel, nHeads, nKV, headDim, maxLen, pos, base, scale, eps)
		return e
	})
}

// TestCoverDecodeStepKVEncodeLegs covers the attention-half + MLP-half encode
// error legs in DecodeStepKV via single-key eviction.
func TestCoverDecodeStepKVEncodeLegs(t *testing.T) {
	requireNativeRuntime(t)

	const dModel, nHeads, nKV, headDim, maxLen, pos, dFF = 64, 4, 2, 64, 4, 0, 256
	const base, scale, eps = float32(10000), float32(0.125), float32(1e-6)
	layer := decodeLayerFixture(dModel, nHeads, nKV, headDim, dFF, 3)
	x := toBF16Bytes(syntheticFloat32(dModel, 1))
	kCache := make([]byte, nKV*maxLen*headDim*bf16Size)
	vCache := make([]byte, nKV*maxLen*headDim*bf16Size)

	coverEncodeEvictAll(t, func() error {
		_, e := DecodeStepKV(x, layer.AttnNormW, layer.WQ, layer.WK, layer.WV, layer.WO, kCache, vCache,
			layer.MLPNormW, layer.WGate, layer.WUp, layer.WDown,
			dModel, nHeads, nKV, headDim, maxLen, dFF, pos, base, scale, eps)
		return e
	})
}

// TestCoverMoEBlockBF16EncodeLegs covers the encode/op error legs in MoEBlockBF16
// and its mlpTransformBF16 helper via single-key eviction.
func TestCoverMoEBlockBF16EncodeLegs(t *testing.T) {
	requireNativeRuntime(t)

	const dModel, dFF, expertDFF = 64, 256, 256
	const eps = float32(1e-6)
	w := moeLayerWeightsFixture(2, 2, dModel, dFF, expertDFF, 3)
	h := toBF16Bytes(syntheticFloat32(dModel, 1))

	coverEncodeEvictAll(t, func() error {
		_, e := MoEBlockBF16(h, w, dModel, dFF, eps)
		return e
	})
}

// TestCoverMoEBlockQuantEncodeLegs covers the encode/op error legs in
// MoEBlockQuant and its mlpTransformQuant helper via single-key eviction.
func TestCoverMoEBlockQuantEncodeLegs(t *testing.T) {
	requireNativeRuntime(t)

	const dModel, dFF, expertDFF = 64, 256, 256
	const gs, bits = 64, 4
	const eps = float32(1e-6)
	w := quantMoELayerWeightsGuard(t, 2, 2, dModel, dFF, expertDFF, gs, bits)
	h := toBF16Bytes(syntheticFloat32(dModel, 1))

	coverEncodeEvictAll(t, func() error {
		_, e := MoEBlockQuant(h, w, dModel, dFF, eps)
		return e
	})
}

// TestCoverPerLayerInputsEncodeLegs covers the downstream-op error legs in
// PerLayerInputs (the bf16-projection path) via single-key eviction. Each step
// (embed gather, project matvec, scale-mul, rms, add, combine-mul) uses a distinct
// kernel sequence, so the legs flip independently.
func TestCoverPerLayerInputsEncodeLegs(t *testing.T) {
	requireNativeRuntime(t)

	const dModel, pliDim, numLayers, vocabPLI = 64, 32, 2, 8
	const eps = float32(1e-5)
	plDim := numLayers * pliDim
	embedPacked := toBF16Bytes(syntheticFloat32(vocabPLI*plDim, 3))
	projW := toBF16Bytes(syntheticFloat32(plDim*dModel, 5))
	projNormW := toBF16Bytes(syntheticFloat32(pliDim, 7))
	hidden := toBF16Bytes(syntheticFloat32(dModel, 9))

	coverEncodeEvictAll(t, func() error {
		_, e := PerLayerInputs(embedPacked, nil, nil, projW, nil, nil, projNormW, 0, hidden,
			vocabPLI, numLayers, pliDim, dModel, 0, 0, 0, 0, eps, bufView{})
		return e
	})
}

// TestCoverLMHeadEncodeLegs covers the downstream-op error leg in LMHeadBF16
// (the final matvec after the norm) via single-key eviction.
func TestCoverLMHeadEncodeLegs(t *testing.T) {
	requireNativeRuntime(t)

	const dModel, vocab = 64, 32
	const eps = float32(1e-6)
	hidden := toBF16Bytes(syntheticFloat32(dModel, 1))
	finalNormW := toBF16Bytes(syntheticFloat32(dModel, 3))
	outWeight := toBF16Bytes(syntheticFloat32(vocab*dModel, 5))

	coverEncodeEvictAll(t, func() error {
		_, e := LMHeadBF16(hidden, finalNormW, outWeight, dModel, vocab, eps, 0)
		return e
	})
}

// TestCoverChainEncodeLegs covers the float32 chain ops MLPBlock + NormProject
// (the gemv/encode legs) via single-key eviction.
func TestCoverChainEncodeLegs(t *testing.T) {
	requireNativeRuntime(t)

	const dModel, dFF = 64, 256
	const eps = float32(1e-6)
	x := syntheticFloat32(dModel, 1)
	normW := syntheticFloat32(dModel, 3)
	wGate := syntheticFloat32(dFF*dModel, 5)
	wUp := syntheticFloat32(dFF*dModel, 7)
	wDown := syntheticFloat32(dModel*dFF, 9)
	projW := syntheticFloat32(dModel*dModel, 11)

	coverEncodeEvictAll(t, func() error {
		if _, e := MLPBlock(x, normW, wGate, wUp, wDown, dModel, dFF, eps); e != nil {
			return e
		}
		_, e := NormProject(x, normW, projW, dModel, dModel, eps)
		return e
	})
}

// TestCoverDecodeLayerEncodeLegs covers the encode-step error legs in the
// composed DecodeLayer (the step-fn chain) via single-key eviction.
func TestCoverDecodeLayerEncodeLegs(t *testing.T) {
	requireNativeRuntime(t)

	const dModel, nHeads, nKV, headDim, kvLen, dFF = 64, 4, 2, 64, 4, 256
	const base, scale, eps = float32(10000), float32(0.125), float32(1e-6)
	layer := decodeLayerFixture(dModel, nHeads, nKV, headDim, dFF, 3)
	x := toBF16Bytes(syntheticFloat32(dModel, 1))
	kCache := make([]byte, nKV*kvLen*headDim*bf16Size)
	vCache := make([]byte, nKV*kvLen*headDim*bf16Size)

	coverEncodeEvictAll(t, func() error {
		_, e := DecodeLayer(x, layer.AttnNormW, layer.WQ, layer.WO, kCache, vCache, layer.MLPNormW,
			layer.WGate, layer.WUp, layer.WDown, dModel, nHeads, nKV, headDim, kvLen, dFF, base, scale, 0, eps)
		return e
	})
}

// TestCoverDecodeForwardEncodeLegs covers the per-layer encAttnHalfKV +
// encMLPHalfBF16 error legs in DecodeForward via single-key eviction.
func TestCoverDecodeForwardEncodeLegs(t *testing.T) {
	requireNativeRuntime(t)

	const dModel, nHeads, nKV, headDim, dFF, maxLen = 64, 4, 2, 64, 256, 4
	const base, scale, eps = float32(10000), float32(0.125), float32(1e-5)
	inputs := decodeInputsFixture(2, dModel)
	layer := decodeLayerFixture(dModel, nHeads, nKV, headDim, dFF, 3)
	// QK-norm set so the per-head q/k norm encode legs in encAttnHalfKV also run.
	layer.QNormW = toBF16Bytes(syntheticFloat32(headDim, 21))
	layer.KNormW = toBF16Bytes(syntheticFloat32(headDim, 23))
	layers := []DecodeLayerWeights{layer}

	coverEncodeEvictAll(t, func() error {
		_, e := DecodeForward(inputs, layers, dModel, nHeads, nKV, headDim, maxLen, dFF, base, scale, eps)
		return e
	})
}

// TestCoverDecodeForwardQuantEncodeLegs covers the per-layer encode error legs in
// DecodeForwardQuant via single-key eviction.
func TestCoverDecodeForwardQuantEncodeLegs(t *testing.T) {
	requireNativeRuntime(t)

	const dModel, nHeads, nKV, headDim, dFF, maxLen = 64, 4, 2, 64, 256, 4
	const gs, bits = 64, 4
	const base, scale, eps = float32(10000), float32(0.125), float32(1e-5)
	inputs := decodeInputsFixture(2, dModel)
	ql := quantizedLayerFixture(t, dModel, nHeads, nKV, headDim, dFF, gs, bits, 3)
	ql.QNormW = toBF16Bytes(syntheticFloat32(headDim, 21))
	ql.KNormW = toBF16Bytes(syntheticFloat32(headDim, 23))
	qlayers := []QuantizedLayerWeights{ql}

	coverEncodeEvictAll(t, func() error {
		_, e := DecodeForwardQuant(inputs, qlayers, dModel, nHeads, nKV, headDim, maxLen, dFF, base, scale, eps)
		return e
	})
}

// TestCoverDecodeForwardArchNormEncodeLegs covers the gemma4 norm-branch encode
// legs in the arch decode (the QK-norm + value-norm + layer-scalar branches in
// encAttnHalfShared / the arch step) by setting all those norm weights and
// evicting each warmed key. decodeLayerFixture leaves them nil, so a plain
// fixture skips those branches; here they are populated.
func TestCoverDecodeForwardArchNormEncodeLegs(t *testing.T) {
	requireNativeRuntime(t)

	const dModel, nHeads, nKV, headDim, dFF, maxLen = 64, 4, 2, 64, 256, 8
	const base, scale, eps = float32(10000), float32(0.125), float32(1e-5)
	specs := model.DeriveLayers([]string{"full_attention"}, 0)
	layer := decodeLayerFixture(dModel, nHeads, nKV, headDim, dFF, 3)
	// populate the gemma4 norms + layer scalar so the conditional encode legs run.
	layer.QNormW = toBF16Bytes(syntheticFloat32(headDim, 21))
	layer.KNormW = toBF16Bytes(syntheticFloat32(headDim, 23))
	layer.PostAttnNormW = toBF16Bytes(syntheticFloat32(dModel, 25))
	layer.PostFFNormW = toBF16Bytes(syntheticFloat32(dModel, 27))
	layer.LayerScalarW = toBF16Bytes(syntheticFloat32(dModel, 29))
	inputs := decodeInputsFixture(2, dModel)

	coverEncodeEvictAll(t, func() error {
		_, e := DecodeForwardArch(inputs, []DecodeLayerWeights{layer}, specs, dModel, nHeads, nKV, headDim, maxLen, dFF, 0, base, scale, eps, false)
		return e
	})
}

// TestCoverMoEExpertsEncodeLegs covers the encGeluGateMul error leg in the
// MoEExperts expert loop via single-key eviction.
func TestCoverMoEExpertsEncodeLegs(t *testing.T) {
	requireNativeRuntime(t)

	const dModel, dFF, numExperts, topK = 64, 256, 2, 2
	w := moeLayerWeightsFixture(numExperts, topK, dModel, dFF, dFF, 3)
	x := toBF16Bytes(syntheticFloat32(dModel, 1))
	idx := []int32{0, 1}
	weights := toBF16Bytes([]float32{0.6, 0.4})

	coverEncodeEvictAll(t, func() error {
		_, e := MoEExperts(x, idx, weights, w.ExpGateW, w.ExpUpW, w.ExpDownW, numExperts, topK, dModel, dFF)
		return e
	})
}

// TestCoverPerLayerInputGateEncodeLegs covers the downstream-op error legs in
// PerLayerInputGateBF16 and PerLayerInputGateQuant via single-key eviction.
func TestCoverPerLayerInputGateEncodeLegs(t *testing.T) {
	requireNativeRuntime(t)

	// dModel=128, pliDim=64 (both multiples of groupSize, so the quant gate/proj
	// quantise cleanly) AND in different gemv tile regimes: the gate gemv
	// gemvTiles(dModel=128, pliDim=64) is standard while the projection gemv
	// gemvTiles(pliDim=64, dModel=128) is small-k, so they take DISTINCT keys and the
	// projection downstream leg separates from the gate one under eviction.
	const dModel, pliDim = 128, 64
	const gs, bits = 64, 4
	const eps = float32(1e-5)
	hNext := toBF16Bytes(syntheticFloat32(dModel, 1))
	perLayerInput := toBF16Bytes(syntheticFloat32(pliDim, 3))
	postNormW := toBF16Bytes(syntheticFloat32(dModel, 5))
	gateW := toBF16Bytes(syntheticFloat32(pliDim*dModel, 7))
	projW := toBF16Bytes(syntheticFloat32(dModel*pliDim, 9))

	coverEncodeEvictAll(t, func() error {
		_, e := PerLayerInputGateBF16(hNext, gateW, perLayerInput, projW, postNormW, dModel, pliDim, eps)
		return e
	})

	qGate := quantWeightFixture(t, pliDim, dModel, gs, bits, 11)
	qProj := quantWeightFixture(t, dModel, pliDim, gs, bits, 13)
	coverEncodeEvictAll(t, func() error {
		_, e := PerLayerInputGateQuant(hNext, qGate, perLayerInput, qProj, postNormW, dModel, pliDim, gs, bits, eps)
		return e
	})
}

// TestCoverLMHeadQuantEncodeLegs covers the encQMVBF16 error leg in LMHeadQuant
// via single-key eviction.
func TestCoverLMHeadQuantEncodeLegs(t *testing.T) {
	requireNativeRuntime(t)

	const dModel, vocab = 64, 64
	const gs, bits = 64, 4
	const eps = float32(1e-6)
	hidden := toBF16Bytes(syntheticFloat32(dModel, 1))
	finalNormW := toBF16Bytes(syntheticFloat32(dModel, 3))
	q := quantWeightFixture(t, vocab, dModel, gs, bits, 5)

	coverEncodeEvictAll(t, func() error {
		_, e := LMHeadQuant(hidden, finalNormW, q.Packed, q.Scales, q.Biases, dModel, vocab, gs, bits, eps, 0)
		return e
	})
}
