// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"bytes"
	"os"
	"slices"
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
// bf16 ring (sliding) beside turboquant codes+γ (global) beside the gated-delta layer's OWN
// gated-delta-state host view (#62) — three coexisting kinds, never mis-tagged into one another.
// The gated-delta layer still owns no KV cache (CacheIndex -1) and never appears as either KV kind;
// it appears under its own kind instead, present (HostStatePresent true) only on the block whose
// span reaches the stream's position, absent on every earlier block — see
// TestStateBlocksTurboQuantHybridGatedDeltaStateRestored_Good for the byte-exact conv/delta receipt.
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
	// The stream carries ALL THREE kinds: the TQ layer's payload at the code strides + γ planes,
	// the native layer untouched, and the gated-delta layer's own host-state kind — present only
	// on the block reaching pos, explicitly absent (never omitted, never mis-tagged) elsewhere.
	sawTQ, sawNative, sawGD, layerCount := false, false, false, 0
	if err := saved.RangeStateBlocks(4, func(block SessionStateBlock) (bool, error) {
		layerCount = len(block.Layers)
		isFinal := block.TokenStart+block.TokenCount == pos
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
			case nativeStateCacheModeGatedDeltaState:
				if layer.Layer != 1 {
					t.Fatalf("gated-delta block layer wrong: %+v", layer)
				}
				if layer.HostStatePresent != isFinal {
					t.Fatalf("block [%d,%d) (pos %d): gated-delta HostStatePresent=%v, want %v",
						block.TokenStart, block.TokenStart+block.TokenCount, pos, layer.HostStatePresent, isFinal)
				}
				if isFinal {
					sawGD = true
					if len(layer.KeyBytes) == 0 || len(layer.ValueBytes) == 0 {
						t.Fatalf("final block's gated-delta payload is empty: %+v", layer)
					}
				} else if len(layer.KeyBytes) != 0 || len(layer.ValueBytes) != 0 {
					t.Fatalf("absent gated-delta block still carries bytes: %+v", layer)
				}
			default:
				t.Fatalf("unexpected cache mode %q", layer.CacheMode)
			}
		}
		return true, nil
	}); err != nil {
		t.Fatalf("RangeStateBlocks: %v", err)
	}
	if !sawTQ || !sawNative || !sawGD {
		t.Fatalf("stream did not carry all three kinds: tq=%v native=%v gated-delta=%v", sawTQ, sawNative, sawGD)
	}
	if layerCount != 3 {
		t.Fatalf("block carries %d layers, want 3 (native + TQ + gated-delta-state)", layerCount)
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

// TestStateBlocksTurboQuantHybridGatedDeltaStateRestored_Good closes #62
// (docs/design-gd-state-blocks.md): the state-block codec (stateLayerViewsRefreshingKinds,
// session_state_blocks.go) used to iterate only cache-OWNING layers (spec.OwnsCache()) — a
// MixerGatedDelta layer's CacheIndex is -1 by construction (docs/design-tq-moe-hybrid.md's own
// words: "it can never be handed a KV cache of any kind"), so it never appeared in a block's Layers
// at all, and its recurrent state (gatedDeltaLayer.conv/delta — host Go fields threaded token to
// token, arch_gated_delta.go) was invisible to StateBlockSource / RestoreStateBlocks. The layer now
// carries its own gated-delta-state host view beside the KV kinds: a restored session's gated-delta
// layer holds the SAME conv/delta the saved one held, byte-exact, and its continuation now matches
// the live session's because the recurrence — not just the KV cache — carried over. Supersedes
// TestStateBlocksTurboQuantHybridGatedDeltaStateNotRestored_Ugly, which pinned the gap this closes;
// TestStateBlocksTurboQuantHybridMixedKinds_Good's own continuation receipt note pointed here.
func TestStateBlocksTurboQuantHybridGatedDeltaStateRestored_Good(t *testing.T) {
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
	wantConv := append([]float32(nil), saved.state.gatedDelta[1].conv...)
	wantDelta := append([]float32(nil), saved.state.gatedDelta[1].delta...)
	if len(wantConv) == 0 || len(wantDelta) == 0 {
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
	// The KV side restores correctly (proven byte-exact by the sibling _Good test); the recurrent
	// side now moves too — direct equality against what the SAVED session held at capture time.
	if len(restored.state.gatedDelta) <= 1 || restored.state.gatedDelta[1] == nil {
		t.Fatal("restored session lost its gated-delta holder entirely")
	}
	gotConv := restored.state.gatedDelta[1].conv
	gotDelta := restored.state.gatedDelta[1].delta
	if !slices.Equal(wantConv, gotConv) {
		t.Fatalf("restored gated-delta conv state differs:\nwant %v\ngot  %v", wantConv, gotConv)
	}
	if !slices.Equal(wantDelta, gotDelta) {
		t.Fatalf("restored gated-delta delta state differs:\nwant %v\ngot  %v", wantDelta, gotDelta)
	}

	// Continuation receipt: now that the recurrence itself carried over (not just the KV cache), the
	// restored session's greedy continuation matches the live one at the token level — the guarantee
	// TestStateBlocksTurboQuantHybridGatedDeltaStateNotRestored_Ugly proved this codec did NOT make.
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

// TestRestoreStateBlocksGatedDeltaKindMismatch_Bad proves the gated-delta host-state kind is ALSO a
// wall, mirroring TestStateBlocksTurboQuantKindMismatch_Bad's proof for the TQ kind: a block tagged
// gated-delta-state refuses to land on a KV (native) view, and a native-tagged block refuses to land
// on a gated-delta-state view. Unlike the TQ mismatch test, there is no session-CONFIG axis that
// turns the gated-delta-state kind on/off for the SAME arch (it is unconditional whenever the arch
// declares a MixerGatedDelta layer) — two live sessions can't be made to disagree on it the way
// native-vs-turboquant kv-cache-mode disagrees. So this drives restoreStateBlock (session_state_blocks.go)
// directly, at the same dispatch layer TestStateBlocksTurboQuantKindMismatch_Bad exercises through a
// full session round trip: it isolates the exact kind check precisely, with no cross-arch fixture
// needed. See docs/design-gd-state-blocks.md.
func TestRestoreStateBlocksGatedDeltaKindMismatch_Bad(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	g, arch := tqHybridMixedKindsTestQuantModel(t)
	const maxLen = 32
	saved, err := newArchQuantSessionShardsWithHeadConfig(g, arch, maxLen, nil, nil, archSessionConfig{kvCacheMode: "turboquant:4"})
	if err != nil {
		t.Fatalf("session: %v", err)
	}
	if err := saved.PrefillTokens([]int32{1, 5, 3}); err != nil {
		t.Fatalf("prefill: %v", err)
	}
	gd := saved.state.gatedDelta[1]
	if gd == nil || len(gd.conv) == 0 || len(gd.delta) == 0 {
		t.Fatal("fixture wants a populated gated-delta holder on layer 1")
	}

	gdView := sessionStateLayerView{layer: 1, cacheMode: nativeStateCacheModeGatedDeltaState, gd: gd}
	nativeView := sessionStateLayerView{
		layer: 1, cacheMode: nativeStateCacheModeFixed, rowBytes: 8, cacheRows: maxLen,
		keyBytes: make([]byte, 8*maxLen), valueBytes: make([]byte, 8*maxLen),
	}

	// A native-shaped block lands on a gated-delta-state view: refuse.
	nativeBlock := SessionStateBlock{Index: 0, TokenStart: 0, TokenCount: 3, Layers: []SessionStateLayerBlock{{
		Layer: 1, CacheMode: nativeStateCacheModeFixed, RowBytes: 8, KeyBytes: make([]byte, 24), ValueBytes: make([]byte, 24),
	}}}
	if err := restoreStateBlock(0, 0, 3, 1, []sessionStateLayerView{gdView}, nativeBlock); err == nil {
		t.Fatal("native block restored onto a gated-delta-state view: expected the kind refusal")
	}

	// A gated-delta-state block lands on a native (KV) view: refuse.
	gdBlock := SessionStateBlock{Index: 0, TokenStart: 0, TokenCount: 3, Layers: []SessionStateLayerBlock{{
		Layer: 1, CacheMode: nativeStateCacheModeGatedDeltaState, HostStatePresent: true,
		KeyBytes: float32Bytes(gd.conv), ValueBytes: float32Bytes(gd.delta),
	}}}
	if err := restoreStateBlock(0, 0, 3, 1, []sessionStateLayerView{nativeView}, gdBlock); err == nil {
		t.Fatal("gated-delta-state block restored onto a native view: expected the kind refusal")
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
