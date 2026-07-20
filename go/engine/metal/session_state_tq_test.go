// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"bytes"
	"os"
	"testing"
	"unsafe"

	core "dappco.re/go"
	"dappco.re/go/inference/model"
	g4 "dappco.re/go/inference/model/gemma4"
)

// session_state_tq_test.go — the MIXED-cache-kind snapshot block codec gates
// (docs/design-tq-moe-hybrid.md): one block stream carries turboquant-codes
// layers (raw codes + γ, bytes preserved exactly) beside native bf16 layers
// (sliding ring — the proven path untouched); restore validates the KIND per
// layer, so codes can never land in a bf16 cache nor bf16 rows in a codes
// cache; and both TQ carriers (recorded-ICB and state-lane) snapshot.

// tqMixedKindsMoEModel assembles a sliding+global MoE gemma4 (head dim 128):
// the sliding layer keeps a NATIVE bf16 ring, the global layer becomes a
// turboquant-codes owner, and the MoE FFN routes the session onto the STATE
// carrier — the full mixed-kind showcase in two layers.
func tqMixedKindsMoEModel(t *testing.T) (*QuantModel, model.Arch) {
	t.Helper()
	const numLayers = 2
	quant := &model.QuantConfig{GroupSize: 64, Bits: 4, Overrides: map[string]model.ModuleQuant{}}
	for i := range numLayers {
		for _, m := range []string{"mlp.gate_proj", "mlp.up_proj", "mlp.down_proj", "router.proj"} {
			quant.Overrides[core.Sprintf("model.layers.%d.%s", i, m)] = model.ModuleQuant{GroupSize: 64, Bits: 8}
		}
	}
	cfg := g4.Config{
		HiddenSize: 64, NumHiddenLayers: numLayers, IntermediateSize: 128,
		NumAttentionHeads: 2, NumKeyValueHeads: 1, HeadDim: 128, VocabSize: 32, RMSNormEps: 1e-6,
		SlidingWindow: 8, LayerTypes: []string{"sliding_attention", "full_attention"},
		EnableMoEBlock: true, NumExperts: 4, TopKExperts: 2, MoEIntermediateSize: 64,
		Quantization: quant,
	}
	arch, err := cfg.Arch()
	if err != nil {
		t.Fatalf("Arch: %v", err)
	}
	if arch.Layer[0].Attention != model.SlidingAttention || arch.Layer[1].Attention != model.GlobalAttention {
		t.Fatalf("fixture wants [sliding, global], got %+v", arch.Layer)
	}
	ts := moeQuantTensors(t, arch, quant)
	lm, err := model.Assemble(ts, arch, model.StandardWeightNames())
	if err != nil {
		t.Fatalf("model.Assemble: %v", err)
	}
	g, err := loadedToQuant(lm, quant.GroupSize, quant.Bits)
	if err != nil {
		t.Fatalf("loadedToQuant: %v", err)
	}
	return g, arch
}

func tqBufferBytes(t *testing.T, buf metalBufferLike, n int) []byte {
	t.Helper()
	if buf == nil {
		t.Fatal("nil buffer")
	}
	return unsafe.Slice((*byte)(buf.Contents()), n)
}

// metalBufferLike narrows the buffer surface the byte receipts read.
type metalBufferLike interface{ Contents() unsafe.Pointer }

// TestStateBlocksTurboQuantMixedKinds_Good gates the mixed-kind round trip on
// the STATE carrier: a sliding+global MoE TQ session snapshots one block
// stream carrying BOTH kinds, a fresh session restores it, the TQ layer's
// codes + γ and the native layer's bf16 ring land byte-exact, and the
// restored session generates the same continuation as the live one.
func TestStateBlocksTurboQuantMixedKinds_Good(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	g, arch := tqMixedKindsMoEModel(t)
	const maxLen = 32
	prompt := []int32{1, 5, 3, 7, 2}

	saved, err := newArchQuantSessionShardsWithHeadConfig(g, arch, maxLen, nil, nil, archSessionConfig{kvCacheMode: "turboquant:4"})
	if err != nil {
		t.Fatalf("turboquant session: %v", err)
	}
	if !saved.state.tqStateArmed() {
		t.Fatal("mixed MoE session did not arm the state carrier")
	}
	if saved.state.kvTQState.on(0) || !saved.state.kvTQState.on(1) {
		t.Fatalf("kind selection wrong: sliding layer 0 TQ=%v (want false), global layer 1 TQ=%v (want true)",
			saved.state.kvTQState.on(0), saved.state.kvTQState.on(1))
	}
	if err := saved.PrefillTokens(prompt); err != nil {
		t.Fatalf("prefill: %v", err)
	}
	embScale := embedScaleOf(arch)
	for _, id := range []int32{4, 9, 6, 8, 2, 1, 3} {
		emb, err := embedTokenQuant(g.Embed, g.EmbedScales, g.EmbedBiases, id, arch.Vocab, arch.Hidden, g.GroupSize, g.Bits, embScale)
		if err != nil {
			t.Fatalf("embed: %v", err)
		}
		if _, err := saved.StepWithID(id, emb); err != nil {
			t.Fatalf("step: %v", err)
		}
	}
	pos := saved.Pos()

	source, err := saved.StateBlockSource(4)
	if err != nil {
		t.Fatalf("StateBlockSource: %v", err)
	}
	// The stream carries BOTH kinds, with the TQ layer's payload at the code
	// strides + γ planes and the native layer untouched.
	sawTQ, sawNative := false, false
	if err := saved.RangeStateBlocks(4, func(block SessionStateBlock) (bool, error) {
		for _, layer := range block.Layers {
			switch layer.CacheMode {
			case nativeStateCacheModeTurboQuantCodes:
				sawTQ = true
				if layer.Layer != 1 || layer.KBits != 4 || layer.VBits != 4 {
					t.Fatalf("TQ block layer wrong: %+v", layer)
				}
				if len(layer.KeyBytes) != block.TokenCount*layer.RowBytes ||
					len(layer.ValueBytes) != block.TokenCount*layer.ValueRowBytes ||
					len(layer.KeyGammaBytes) != block.TokenCount*layer.GammaRowBytes ||
					len(layer.ValueGammaBytes) != block.TokenCount*layer.GammaRowBytes {
					t.Fatalf("TQ block payload sizes wrong: %+v", layer)
				}
			case nativeStateCacheModeFixed:
				sawNative = true
				if layer.Layer != 0 {
					t.Fatalf("native block layer wrong: %+v", layer)
				}
				if layer.ValueRowBytes != 0 || len(layer.KeyGammaBytes) != 0 {
					t.Fatalf("native layer carries TQ fields: %+v", layer)
				}
			default:
				t.Fatalf("unexpected cache mode %q", layer.CacheMode)
			}
		}
		return true, nil
	}); err != nil {
		t.Fatalf("RangeStateBlocks: %v", err)
	}
	if !sawTQ || !sawNative {
		t.Fatalf("stream did not carry both kinds: tq=%v native=%v", sawTQ, sawNative)
	}

	restored, err := newArchQuantSessionShardsWithHeadConfig(g, arch, maxLen, nil, nil, archSessionConfig{kvCacheMode: "turboquant:4"})
	if err != nil {
		t.Fatalf("restore session: %v", err)
	}
	if err := restored.RestoreStateBlocks(source); err != nil {
		t.Fatalf("RestoreStateBlocks: %v", err)
	}
	if restored.Pos() != pos {
		t.Fatalf("restored pos = %d, want %d", restored.Pos(), pos)
	}

	// Byte receipts, per kind. TQ layer 1: codes + γ over the live rows.
	sTQ, rTQ := saved.state.kvTQState, restored.state.kvTQState
	kBytes := pos * sTQ.set.kRowBytes[1]
	vBytes := pos * sTQ.set.vRowBytes[1]
	gBytes := pos * sTQ.set.gammaRowBytes[1]
	if !bytes.Equal(tqBufferBytes(t, sTQ.kCaches[1], kBytes), tqBufferBytes(t, rTQ.kCaches[1], kBytes)) {
		t.Fatal("TQ K codes differ after restore")
	}
	if !bytes.Equal(tqBufferBytes(t, sTQ.vCaches[1], vBytes), tqBufferBytes(t, rTQ.vCaches[1], vBytes)) {
		t.Fatal("TQ V codes differ after restore")
	}
	if !bytes.Equal(tqBufferBytes(t, sTQ.set.kGammas[1], gBytes), tqBufferBytes(t, rTQ.set.kGammas[1], gBytes)) {
		t.Fatal("TQ K γ plane differs after restore")
	}
	if !bytes.Equal(tqBufferBytes(t, sTQ.set.vGammas[1], gBytes), tqBufferBytes(t, rTQ.set.vGammas[1], gBytes)) {
		t.Fatal("TQ V γ plane differs after restore")
	}
	// Native sliding layer 0: the whole ring (window 8 < pos → full ring live).
	ringBytes := int(saved.state.lb[0].kvCacheBytes)
	if ringBytes == 0 || saved.state.lb[0].kCache == nil || restored.state.lb[0].kCache == nil {
		t.Fatal("sliding layer lb ring missing")
	}
	if !bytes.Equal(tqBufferBytes(t, saved.state.lb[0].kCache, ringBytes), tqBufferBytes(t, restored.state.lb[0].kCache, ringBytes)) {
		t.Fatal("native sliding K ring differs after restore")
	}
	if !bytes.Equal(tqBufferBytes(t, saved.state.lb[0].vCache, ringBytes), tqBufferBytes(t, restored.state.lb[0].vCache, ringBytes)) {
		t.Fatal("native sliding V ring differs after restore")
	}

	// Continuation receipt: the restored session generates the same greedy
	// tokens as the live one from the same boundary.
	wantGen, err := saved.GenerateFromCache(3, -1)
	if err != nil {
		t.Fatalf("saved GenerateFromCache: %v", err)
	}
	gotGen, err := restored.GenerateFromCache(3, -1)
	if err != nil {
		t.Fatalf("restored GenerateFromCache: %v", err)
	}
	if len(wantGen) != len(gotGen) {
		t.Fatalf("continuation lengths differ: %v vs %v", wantGen, gotGen)
	}
	for i := range wantGen {
		if wantGen[i] != gotGen[i] {
			t.Fatalf("continuation diverged at %d: %v vs %v", i, wantGen, gotGen)
		}
	}
}

// TestStateBlocksTurboQuantICBCarrier_Good proves the RECORDED carrier
// snapshots through the same kind-aware codec: the dense all-global TQ
// session (arch ICB armed) round-trips its codes + γ byte-exact and
// continues identically.
func TestStateBlocksTurboQuantICBCarrier_Good(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	g, arch := tqTestQuantModel(t)
	const maxLen = 32
	prompt := []int32{1, 5, 3, 7, 2, 4}

	saved, err := newArchQuantSessionShardsWithHeadConfig(g, arch, maxLen, nil, nil, archSessionConfig{kvCacheMode: "turboquant:3.5"})
	if err != nil {
		t.Fatalf("turboquant session: %v", err)
	}
	if !saved.state.icb.hasKVTQ() {
		t.Fatal("dense session did not arm the recorded carrier")
	}
	if err := saved.PrefillTokens(prompt); err != nil {
		t.Fatalf("prefill: %v", err)
	}
	pos := saved.Pos()
	source, err := saved.StateBlockSource(4)
	if err != nil {
		t.Fatalf("StateBlockSource: %v", err)
	}
	restored, err := newArchQuantSessionShardsWithHeadConfig(g, arch, maxLen, nil, nil, archSessionConfig{kvCacheMode: "turboquant:3.5"})
	if err != nil {
		t.Fatalf("restore session: %v", err)
	}
	if err := restored.RestoreStateBlocks(source); err != nil {
		t.Fatalf("RestoreStateBlocks: %v", err)
	}
	if restored.Pos() != pos {
		t.Fatalf("restored pos = %d, want %d", restored.Pos(), pos)
	}
	sSet, sK, sV := saved.tqSnapshotCarrierBuffers()
	rSet, rK, rV := restored.tqSnapshotCarrierBuffers()
	for li := range arch.Layer {
		if !sSet.on(li) {
			continue
		}
		kBytes := pos * sSet.kRowBytes[li]
		vBytes := pos * sSet.vRowBytes[li]
		gBytes := pos * sSet.gammaRowBytes[li]
		if !bytes.Equal(tqBufferBytes(t, sK[li], kBytes), tqBufferBytes(t, rK[li], kBytes)) {
			t.Fatalf("layer %d: ICB-carrier K codes differ after restore", li)
		}
		if !bytes.Equal(tqBufferBytes(t, sV[li], vBytes), tqBufferBytes(t, rV[li], vBytes)) {
			t.Fatalf("layer %d: ICB-carrier V codes differ after restore", li)
		}
		if !bytes.Equal(tqBufferBytes(t, sSet.kGammas[li], gBytes), tqBufferBytes(t, rSet.kGammas[li], gBytes)) {
			t.Fatalf("layer %d: ICB-carrier K γ differs after restore", li)
		}
		if !bytes.Equal(tqBufferBytes(t, sSet.vGammas[li], gBytes), tqBufferBytes(t, rSet.vGammas[li], gBytes)) {
			t.Fatalf("layer %d: ICB-carrier V γ differs after restore", li)
		}
	}
	wantGen, err := saved.GenerateFromCache(3, -1)
	if err != nil {
		t.Fatalf("saved GenerateFromCache: %v", err)
	}
	gotGen, err := restored.GenerateFromCache(3, -1)
	if err != nil {
		t.Fatalf("restored GenerateFromCache: %v", err)
	}
	for i := range wantGen {
		if wantGen[i] != gotGen[i] {
			t.Fatalf("continuation diverged at %d: %v vs %v", i, wantGen, gotGen)
		}
	}
}

// tqHybridMixedKindsTestQuantModel assembles a sliding+gated-delta+global hybrid (head dim 128):
// the sliding layer keeps a NATIVE bf16 ring, the linear_attention (MixerGatedDelta) layer owns no
// cache at all (CacheIndex -1 by construction), and the global layer becomes a turboquant-codes
// owner — the mixed-kind showcase (docs/design-tq-moe-hybrid.md) PLUS the hybrid's third,
// cache-free layer kind riding alongside it in the same three-layer stack.
func tqHybridMixedKindsTestQuantModel(t *testing.T) (*QuantModel, model.Arch) {
	t.Helper()
	const gs, bits = 64, 4
	cfg := g4.Config{
		HiddenSize: 64, NumHiddenLayers: 3, IntermediateSize: 128,
		NumAttentionHeads: 2, NumKeyValueHeads: 1, HeadDim: 128, VocabSize: 32, RMSNormEps: 1e-6,
		SlidingWindow: 8, LayerTypes: []string{"sliding_attention", "linear_attention", "full_attention"},
		Quantization: &model.QuantConfig{GroupSize: gs, Bits: bits},
	}
	arch, err := cfg.Arch()
	if err != nil {
		t.Fatalf("Arch: %v", err)
	}
	if arch.Layer[0].Attention != model.SlidingAttention || arch.Layer[0].Mixer != model.MixerAttention {
		t.Fatalf("fixture wants layer 0 sliding attention, got %+v", arch.Layer[0])
	}
	if arch.Layer[1].Mixer != model.MixerGatedDelta || arch.Layer[1].CacheIndex != -1 {
		t.Fatalf("fixture wants layer 1 gated-delta with no cache, got %+v", arch.Layer[1])
	}
	if arch.Layer[2].Attention != model.GlobalAttention || arch.Layer[2].Mixer != model.MixerAttention {
		t.Fatalf("fixture wants layer 2 global attention, got %+v", arch.Layer[2])
	}
	ts := hybridQuantTensors(t, arch, gs, bits)
	lm, err := model.Assemble(ts, arch, model.StandardWeightNames())
	if err != nil {
		t.Fatalf("model.Assemble: %v", err)
	}
	g, err := loadedToQuant(lm, gs, bits)
	if err != nil {
		t.Fatalf("loadedToQuant: %v", err)
	}
	return g, arch
}

// TestStateBlocksTurboQuantHybridMixedKinds_Good gates the mixed-kind round trip on a PLAIN-
// attention + gated-delta hybrid (#48 declared boundary #1): one block stream carries a native
// bf16 ring (sliding) beside turboquant codes+γ (global), byte-exact on restore, while the
// gated-delta layer's absence from the KV block stream is itself asserted — CacheIndex -1 means
// it owns no cache and must never appear as a block layer, never mis-tagged into either kind.
func TestStateBlocksTurboQuantHybridMixedKinds_Good(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	g, arch := tqHybridMixedKindsTestQuantModel(t)
	const maxLen = 32
	prompt := []int32{1, 5, 3, 7, 2}

	saved, err := newArchQuantSessionShardsWithHeadConfig(g, arch, maxLen, nil, nil, archSessionConfig{kvCacheMode: "turboquant:4"})
	if err != nil {
		t.Fatalf("turboquant session: %v", err)
	}
	if !saved.state.tqStateArmed() {
		t.Fatal("hybrid session did not arm the state carrier")
	}
	if saved.state.kvTQState.on(0) || !saved.state.kvTQState.on(2) {
		t.Fatalf("kind selection wrong: sliding layer 0 TQ=%v (want false), global layer 2 TQ=%v (want true)",
			saved.state.kvTQState.on(0), saved.state.kvTQState.on(2))
	}
	if err := saved.PrefillTokens(prompt); err != nil {
		t.Fatalf("prefill: %v", err)
	}
	embScale := embedScaleOf(arch)
	for _, id := range []int32{4, 9, 6, 8, 2, 1, 3} {
		emb, err := embedTokenQuant(g.Embed, g.EmbedScales, g.EmbedBiases, id, arch.Vocab, arch.Hidden, g.GroupSize, g.Bits, embScale)
		if err != nil {
			t.Fatalf("embed: %v", err)
		}
		if _, err := saved.StepWithID(id, emb); err != nil {
			t.Fatalf("step: %v", err)
		}
	}
	pos := saved.Pos()

	source, err := saved.StateBlockSource(4)
	if err != nil {
		t.Fatalf("StateBlockSource: %v", err)
	}
	// The stream carries BOTH KV kinds, with the TQ layer's payload at the code strides + γ
	// planes and the native layer untouched — and NOTHING for the gated-delta layer: it owns no
	// cache (CacheIndex -1), so it must never surface as a block layer of either kind.
	sawTQ, sawNative, layerCount := false, false, 0
	if err := saved.RangeStateBlocks(4, func(block SessionStateBlock) (bool, error) {
		layerCount = len(block.Layers)
		for _, layer := range block.Layers {
			switch layer.CacheMode {
			case nativeStateCacheModeTurboQuantCodes:
				sawTQ = true
				if layer.Layer != 2 || layer.KBits != 4 || layer.VBits != 4 {
					t.Fatalf("TQ block layer wrong: %+v", layer)
				}
				if len(layer.KeyBytes) != block.TokenCount*layer.RowBytes ||
					len(layer.ValueBytes) != block.TokenCount*layer.ValueRowBytes ||
					len(layer.KeyGammaBytes) != block.TokenCount*layer.GammaRowBytes ||
					len(layer.ValueGammaBytes) != block.TokenCount*layer.GammaRowBytes {
					t.Fatalf("TQ block payload sizes wrong: %+v", layer)
				}
			case nativeStateCacheModeFixed:
				sawNative = true
				if layer.Layer != 0 {
					t.Fatalf("native block layer wrong: %+v", layer)
				}
				if layer.ValueRowBytes != 0 || len(layer.KeyGammaBytes) != 0 {
					t.Fatalf("native layer carries TQ fields: %+v", layer)
				}
			default:
				t.Fatalf("unexpected cache mode %q", layer.CacheMode)
			}
		}
		return true, nil
	}); err != nil {
		t.Fatalf("RangeStateBlocks: %v", err)
	}
	if !sawTQ || !sawNative {
		t.Fatalf("stream did not carry both KV kinds: tq=%v native=%v", sawTQ, sawNative)
	}
	if layerCount != 2 {
		t.Fatalf("block carries %d layers, want 2 — the gated-delta layer owns no cache (CacheIndex -1) and must never appear", layerCount)
	}

	restored, err := newArchQuantSessionShardsWithHeadConfig(g, arch, maxLen, nil, nil, archSessionConfig{kvCacheMode: "turboquant:4"})
	if err != nil {
		t.Fatalf("restore session: %v", err)
	}
	if err := restored.RestoreStateBlocks(source); err != nil {
		t.Fatalf("RestoreStateBlocks: %v", err)
	}
	if restored.Pos() != pos {
		t.Fatalf("restored pos = %d, want %d", restored.Pos(), pos)
	}

	// Byte receipts, per KV kind.
	sTQ, rTQ := saved.state.kvTQState, restored.state.kvTQState
	kBytes := pos * sTQ.set.kRowBytes[2]
	vBytes := pos * sTQ.set.vRowBytes[2]
	gBytes := pos * sTQ.set.gammaRowBytes[2]
	if !bytes.Equal(tqBufferBytes(t, sTQ.kCaches[2], kBytes), tqBufferBytes(t, rTQ.kCaches[2], kBytes)) {
		t.Fatal("TQ K codes differ after restore")
	}
	if !bytes.Equal(tqBufferBytes(t, sTQ.vCaches[2], vBytes), tqBufferBytes(t, rTQ.vCaches[2], vBytes)) {
		t.Fatal("TQ V codes differ after restore")
	}
	if !bytes.Equal(tqBufferBytes(t, sTQ.set.kGammas[2], gBytes), tqBufferBytes(t, rTQ.set.kGammas[2], gBytes)) {
		t.Fatal("TQ K γ plane differs after restore")
	}
	if !bytes.Equal(tqBufferBytes(t, sTQ.set.vGammas[2], gBytes), tqBufferBytes(t, rTQ.set.vGammas[2], gBytes)) {
		t.Fatal("TQ V γ plane differs after restore")
	}
	// Native sliding layer 0: the whole ring (window 8 < pos → full ring live).
	ringBytes := int(saved.state.lb[0].kvCacheBytes)
	if ringBytes == 0 || saved.state.lb[0].kCache == nil || restored.state.lb[0].kCache == nil {
		t.Fatal("sliding layer lb ring missing")
	}
	if !bytes.Equal(tqBufferBytes(t, saved.state.lb[0].kCache, ringBytes), tqBufferBytes(t, restored.state.lb[0].kCache, ringBytes)) {
		t.Fatal("native sliding K ring differs after restore")
	}
	if !bytes.Equal(tqBufferBytes(t, saved.state.lb[0].vCache, ringBytes), tqBufferBytes(t, restored.state.lb[0].vCache, ringBytes)) {
		t.Fatal("native sliding V ring differs after restore")
	}

	// Continuation receipt: the restored session generates the same greedy tokens as the live one
	// from the same boundary — see TestStateBlocksTurboQuantHybridGatedDeltaStateNotRestored_Ugly
	// for the important caveat this receipt does NOT cover (the gated-delta layer's own recurrent
	// state is not part of this codec and does not restore; it happens not to move the greedy
	// argmax on this short, small-vocab fixture, which is a property of THIS fixture, not a
	// guarantee the codec makes).
	wantGen, err := saved.GenerateFromCache(3, -1)
	if err != nil {
		t.Fatalf("saved GenerateFromCache: %v", err)
	}
	gotGen, err := restored.GenerateFromCache(3, -1)
	if err != nil {
		t.Fatalf("restored GenerateFromCache: %v", err)
	}
	if len(wantGen) != len(gotGen) {
		t.Fatalf("continuation lengths differ: %v vs %v", wantGen, gotGen)
	}
	for i := range wantGen {
		if wantGen[i] != gotGen[i] {
			t.Fatalf("continuation diverged at %d: %v vs %v", i, wantGen, gotGen)
		}
	}
}

// TestStateBlocksTurboQuantHybridGatedDeltaStateNotRestored_Ugly pins a boundary this campaign
// discovered rather than introduced: the state-block codec (stateLayerViewsRefreshingKinds,
// session_state_blocks.go) iterates only cache-OWNING layers (spec.OwnsCache()) — a MixerGatedDelta
// layer's CacheIndex is -1 by construction (docs/design-tq-moe-hybrid.md's own words: "it can never
// be handed a KV cache of any kind"), so it never appears in a block's Layers at all. Its recurrent
// state (gatedDeltaLayer.conv/delta — host Go fields threaded token to token, arch_gated_delta.go)
// is consequently invisible to StateBlockSource / RestoreStateBlocks: a restored session's
// gated-delta layer starts from a FRESH (empty) recurrence, not the saved one's accumulated history.
// This predates #48 (the recurrence itself is #18); the design doc's mixed-kind codec is scoped to
// the KV-cache axis only and never claims recurrent-state fidelity. Surprising but valid — pinned
// explicitly so a caller resuming a hybrid conversation via the block codec, or a future maintainer
// reading TestStateBlocksTurboQuantHybridMixedKinds_Good's passing continuation receipt, does not
// mistake either for a guarantee that gated-delta memory survives a save/restore round trip.
func TestStateBlocksTurboQuantHybridGatedDeltaStateNotRestored_Ugly(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	g, arch := tqHybridMixedKindsTestQuantModel(t)
	const maxLen = 32
	prompt := []int32{1, 5, 3, 7, 2}

	saved, err := newArchQuantSessionShardsWithHeadConfig(g, arch, maxLen, nil, nil, archSessionConfig{kvCacheMode: "turboquant:4"})
	if err != nil {
		t.Fatalf("turboquant session: %v", err)
	}
	if err := saved.PrefillTokens(prompt); err != nil {
		t.Fatalf("prefill: %v", err)
	}
	embScale := embedScaleOf(arch)
	for _, id := range []int32{4, 9, 6, 8, 2, 1, 3} {
		emb, err := embedTokenQuant(g.Embed, g.EmbedScales, g.EmbedBiases, id, arch.Vocab, arch.Hidden, g.GroupSize, g.Bits, embScale)
		if err != nil {
			t.Fatalf("embed: %v", err)
		}
		if _, err := saved.StepWithID(id, emb); err != nil {
			t.Fatalf("step: %v", err)
		}
	}
	if len(saved.state.gatedDelta) <= 1 || saved.state.gatedDelta[1] == nil {
		t.Fatal("fixture wants a bound gated-delta holder on layer 1")
	}
	if len(saved.state.gatedDelta[1].conv) == 0 || len(saved.state.gatedDelta[1].delta) == 0 {
		t.Fatal("prefill+decode must have advanced the gated-delta recurrence — the compare below would be vacuous")
	}

	source, err := saved.StateBlockSource(4)
	if err != nil {
		t.Fatalf("StateBlockSource: %v", err)
	}
	restored, err := newArchQuantSessionShardsWithHeadConfig(g, arch, maxLen, nil, nil, archSessionConfig{kvCacheMode: "turboquant:4"})
	if err != nil {
		t.Fatalf("restore session: %v", err)
	}
	if err := restored.RestoreStateBlocks(source); err != nil {
		t.Fatalf("RestoreStateBlocks: %v", err)
	}
	// The KV side restored correctly (proven byte-exact by the sibling _Good test); the recurrent
	// side did not move at all — it is still the fresh session's zero-value, untouched by restore.
	if len(restored.state.gatedDelta) <= 1 || restored.state.gatedDelta[1] == nil {
		t.Fatal("restored session lost its gated-delta holder entirely")
	}
	if len(restored.state.gatedDelta[1].conv) != 0 || len(restored.state.gatedDelta[1].delta) != 0 {
		t.Fatalf("restored gated-delta state is no longer empty (conv=%d delta=%d) — the codec grew recurrent-state support; update this test's premise and comment",
			len(restored.state.gatedDelta[1].conv), len(restored.state.gatedDelta[1].delta))
	}
}

// TestStateBlocksTurboQuantKindMismatch_Bad proves the kind boundary is a
// wall: TQ blocks refuse to land on a native session, native blocks refuse to
// land on a TQ layer, and a bit-width mismatch between snapshot and session
// refuses — never a reinterpret, never a silent mixed cache.
func TestStateBlocksTurboQuantKindMismatch_Bad(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	g, arch := tqMixedKindsMoEModel(t)
	const maxLen = 32
	prompt := []int32{1, 5, 3, 7, 2}

	tqSaved, err := newArchQuantSessionShardsWithHeadConfig(g, arch, maxLen, nil, nil, archSessionConfig{kvCacheMode: "turboquant:4"})
	if err != nil {
		t.Fatalf("turboquant session: %v", err)
	}
	if err := tqSaved.PrefillTokens(prompt); err != nil {
		t.Fatalf("tq prefill: %v", err)
	}
	tqSource, err := tqSaved.StateBlockSource(4)
	if err != nil {
		t.Fatalf("tq StateBlockSource: %v", err)
	}

	// TQ blocks → native session: kind mismatch, loud.
	nativeTarget, err := newArchQuantSessionShardsWithHeadConfig(g, arch, maxLen, nil, nil, archSessionConfig{})
	if err != nil {
		t.Fatalf("native session: %v", err)
	}
	if err := nativeTarget.RestoreStateBlocks(tqSource); err == nil {
		t.Fatal("turboquant-codes blocks restored onto a native session: expected the kind refusal")
	}

	// Native blocks → TQ session: kind mismatch, loud.
	nativeSaved, err := newArchQuantSessionShardsWithHeadConfig(g, arch, maxLen, nil, nil, archSessionConfig{})
	if err != nil {
		t.Fatalf("native session: %v", err)
	}
	if err := nativeSaved.PrefillTokens(prompt); err != nil {
		t.Fatalf("native prefill: %v", err)
	}
	nativeSource, err := nativeSaved.StateBlockSource(4)
	if err != nil {
		t.Fatalf("native StateBlockSource: %v", err)
	}
	tqTarget, err := newArchQuantSessionShardsWithHeadConfig(g, arch, maxLen, nil, nil, archSessionConfig{kvCacheMode: "turboquant:4"})
	if err != nil {
		t.Fatalf("turboquant target: %v", err)
	}
	if err := tqTarget.RestoreStateBlocks(nativeSource); err == nil {
		t.Fatal("native blocks restored onto a turboquant session: expected the kind refusal")
	}

	// Same kind, different bit widths: snapshot and session must run the same
	// -kv-cache mode.
	tqSource2, err := tqSaved.StateBlockSource(4)
	if err != nil {
		t.Fatalf("tq StateBlockSource: %v", err)
	}
	tq3Target, err := newArchQuantSessionShardsWithHeadConfig(g, arch, maxLen, nil, nil, archSessionConfig{kvCacheMode: "turboquant:3"})
	if err != nil {
		t.Fatalf("turboquant:3 target: %v", err)
	}
	if err := tq3Target.RestoreStateBlocks(tqSource2); err == nil {
		t.Fatal("turboquant:4 blocks restored onto a turboquant:3 session: expected the bit-width refusal")
	}
}
