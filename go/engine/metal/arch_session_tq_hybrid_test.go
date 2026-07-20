// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"math"
	"os"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference/model"
	g4 "dappco.re/go/inference/model/gemma4"
	"dappco.re/go/inference/model/safetensors"
)

// arch_session_tq_hybrid_test.go — the state-carrier TurboQuant gates on a PLAIN-attention +
// gated-delta hybrid (#48 follow-on, docs/design-tq-moe-hybrid.md, closing the design's declared
// boundary #1): "a future plain-attn+gated-delta arch rides the carrier unchanged" had no in-tree
// fixture. This is that fixture — a synthetic hybrid checkpoint whose full_attention layers carry
// NO attn_output_gate (the qwen3_5/next gated shape keeps refusing loudly, unchanged) beside
// linear_attention (MixerGatedDelta) layers, so the layer-kind matrix's two hybrid-relevant rows
// both fire in the SAME session: TQ codes+γ on the qualifying attention layers, native recurrent
// state on the gated-delta layers.

// hybridQuantTensors builds a synthetic PLAIN-attention + gated-delta hybrid checkpoint's tensors:
// standard-attention layers quantised exactly as quantGemma4Tensors (q/k/v/o + dense MLP, 4-bit),
// linear_attention (MixerGatedDelta) layers carry the five gated-delta projections bf16
// (assembleGatedDelta's quant-or-bf16 form, unforced — this fixture exercises the bf16 leg, exactly
// as TestAssembleGatedDelta's own fixture does) beside the small recurrence tensors (A_log/norm/
// conv1d, host f32) and a dense MLP tail — every layer, mixer or recurrence, carries an FFN (#18:
// the mixer replaces attention, not the whole block). No attn_output_gate anywhere in the caller's
// arch: this is the design's declared boundary #1, not the gated qwen3_5/next shape that still
// refuses (tq_kv_mode.go's arch-wide decline).
func hybridQuantTensors(t testing.TB, arch model.Arch, gs, bits int) map[string]safetensors.Tensor {
	t.Helper()
	ts := map[string]safetensors.Tensor{}
	salt := 1
	mkNorm := func(name string, elems int) {
		f := make([]float32, elems)
		for i := range f {
			f[i] = float32((i*salt+13)%97-48) * 0.02
		}
		ts[name] = safetensors.Tensor{Dtype: "BF16", Shape: []int{elems}, Data: toBF16Bytes(f)}
		salt++
	}
	mkQuant := func(prefix string, outDim, inDim int) {
		p, s, b := quantizeProj(t, outDim, inDim, gs, bits, salt)
		salt++
		ts[prefix+".weight"] = safetensors.Tensor{Dtype: "U32", Shape: []int{outDim, inDim * bits / 32}, Data: p}
		ts[prefix+".scales"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{outDim, inDim / gs}, Data: s}
		ts[prefix+".biases"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{outDim, inDim / gs}, Data: b}
	}
	mkBF16Lin := func(name string, outDim, inDim int) {
		ts[name] = safetensors.Tensor{Dtype: "BF16", Shape: []int{outDim, inDim}, Data: toBF16Bytes(syntheticFloat32(outDim*inDim, salt))}
		salt++
	}
	dModel, headDim, dFF, vocab := arch.Hidden, arch.HeadDim, arch.FF, arch.Vocab
	mkQuant("model.embed_tokens", vocab, dModel)
	mkNorm("model.norm.weight", dModel)
	for i := range arch.Layer {
		p := core.Sprintf("model.layers.%d", i)
		// gemma4's full norm sandwich (input + post-mixer + pre-FF + post-FF), on every layer
		// regardless of mixer kind — quantGemma4Tensors' own set, matched here so a hybrid
		// fixture built on gemma4.Config carries the same normalisation the architecture
		// declares rather than a partial subset that leaves logits under-damped.
		mkNorm(p+".input_layernorm.weight", dModel)
		mkNorm(p+".post_attention_layernorm.weight", dModel)
		mkNorm(p+".pre_feedforward_layernorm.weight", dModel)
		mkNorm(p+".post_feedforward_layernorm.weight", dModel)
		mkQuant(p+".mlp.gate_proj", dFF, dModel)
		mkQuant(p+".mlp.up_proj", dFF, dModel)
		mkQuant(p+".mlp.down_proj", dModel, dFF)
		if arch.Layer[i].Mixer == model.MixerGatedDelta {
			// linear_attn geometry: small, independent of the attention dims (the recurrence
			// math is host f32 — size here is a test-speed knob, not a correctness constraint).
			const valueHeads, keyHeads, gdHeadDim, convK = 2, 1, 8, 4
			const vDim, qDim = valueHeads * gdHeadDim, keyHeads * gdHeadDim
			const convDim = 2*qDim + vDim
			lp := p + ".linear_attn."
			mkNorm(lp+"A_log", valueHeads)
			mkNorm(lp+"norm.weight", gdHeadDim)
			// dt_bias is loader-optional (tensorFloat32Opt) but the host recurrence
			// (GatedDeltaForwardScratchFromInputF32) indexes w.DtBias[h] unconditionally —
			// every real checkpoint ships it, so this fixture does too.
			mkNorm(lp+"dt_bias", valueHeads)
			ts[lp+"conv1d.weight"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{convDim, convK}, Data: toBF16Bytes(syntheticFloat32(convDim*convK, salt))}
			salt++
			mkBF16Lin(lp+"in_proj_qkv.weight", convDim, dModel)
			mkBF16Lin(lp+"in_proj_a.weight", valueHeads, dModel)
			mkBF16Lin(lp+"in_proj_b.weight", valueHeads, dModel)
			mkBF16Lin(lp+"in_proj_z.weight", vDim, dModel)
			mkBF16Lin(lp+"out_proj.weight", dModel, vDim)
			continue
		}
		lhd := headDimOf(arch.Layer[i], headDim)
		lkv := kvHeadsOf(arch.Layer[i], arch.KVHeads)
		lqDim, lkvDim := arch.Heads*lhd, lkv*lhd
		// gemma4's Config.Arch declares AttnScale 1.0 (not 1/√headDim) because the per-head
		// QK-norm IS the scaling (metal's gemma4AttentionScale) — omitting q_norm/k_norm here
		// while inheriting AttnScale 1.0 would leave raw, unnormalised Q·K logits (unscaled by
		// EITHER mechanism) feeding softmax, saturating it and making the result hypersensitive
		// to TQ's quantisation noise. Every fixture riding gemma4.Config must supply both.
		mkNorm(p+".self_attn.q_norm.weight", lhd)
		mkNorm(p+".self_attn.k_norm.weight", lhd)
		mkQuant(p+".self_attn.q_proj", lqDim, dModel)
		mkQuant(p+".self_attn.k_proj", lkvDim, dModel)
		mkQuant(p+".self_attn.v_proj", lkvDim, dModel)
		mkQuant(p+".self_attn.o_proj", dModel, lqDim)
	}
	return ts
}

// tqHybridTestQuantModel assembles a synthetic PLAIN-attention + gated-delta hybrid (head dim 128 —
// a TQ-instantiated width): two GLOBAL standard-attention layers (attn_output_gate stays false — the
// design's declared boundary #1) sandwich one linear_attention (MixerGatedDelta, no KV cache) layer,
// so the state-carrier session must coexist BOTH kinds: TQ codes+γ on the attention layers, native
// recurrent state on the gated-delta layer, in the SAME session.
func tqHybridTestQuantModel(t *testing.T) (*QuantModel, model.Arch) {
	t.Helper()
	const gs, bits = 64, 4
	cfg := g4.Config{
		HiddenSize: 64, NumHiddenLayers: 3, IntermediateSize: 128,
		NumAttentionHeads: 2, NumKeyValueHeads: 1, HeadDim: 128, VocabSize: 32, RMSNormEps: 1e-6,
		LayerTypes:   []string{"full_attention", "linear_attention", "full_attention"},
		Quantization: &model.QuantConfig{GroupSize: gs, Bits: bits},
	}
	arch, err := cfg.Arch()
	if err != nil {
		t.Fatalf("Arch: %v", err)
	}
	if arch.AttnOutputGate {
		t.Fatal("fixture must stay ungated — the design's declared boundary is a PLAIN-attention hybrid")
	}
	if arch.Layer[0].Mixer != model.MixerAttention || arch.Layer[2].Mixer != model.MixerAttention {
		t.Fatalf("fixture wants layers 0/2 plain attention, got %+v / %+v", arch.Layer[0], arch.Layer[2])
	}
	if arch.Layer[1].Mixer != model.MixerGatedDelta || arch.Layer[1].CacheIndex != -1 {
		t.Fatalf("fixture wants layer 1 gated-delta with no cache, got %+v", arch.Layer[1])
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

// TestArchQuantSessionTurboQuantHybrid_Good closes #48's declared boundary #1: a PLAIN-attention +
// gated-delta hybrid (no attn_output_gate — the qwen3_5/next gated shape still refuses, unchanged)
// arms the state-lane TQ carrier on its attention layers, keeps the gated-delta layer on native
// recurrent state in the SAME session (the per-layer-kind coexistence the design promises), and
// stays within the existing session codec band against its native twin — the same method as
// TestArchQuantSessionTurboQuantMoE_Good, over the hybrid fixture instead of the MoE one.
func TestArchQuantSessionTurboQuantHybrid_Good(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	g, arch := tqHybridTestQuantModel(t)
	const maxLen = 32
	prompt := []int32{1, 5, 3, 7}

	native, err := newArchQuantSessionShardsWithHeadConfig(g, arch, maxLen, nil, nil, archSessionConfig{})
	if err != nil {
		t.Fatalf("native session: %v", err)
	}
	tq, err := newArchQuantSessionShardsWithHeadConfig(g, arch, maxLen, nil, nil, archSessionConfig{kvCacheMode: "turboquant:4"})
	if err != nil {
		t.Fatalf("turboquant session: %v", err)
	}
	if tq.state.icb != nil {
		t.Fatal("hybrid session recorded an arch ICB — the state carrier gates assume stepToken decode")
	}
	if !tq.state.tqStateArmed() {
		t.Fatal("turboquant hybrid session did not arm the state carrier")
	}
	if !tq.hasKVTQAny() {
		t.Fatal("hasKVTQAny must report the state carrier — every TQ decline gates on it")
	}
	if native.state.tqStateArmed() || native.hasKVTQAny() {
		t.Fatal("native hybrid session armed a TQ carrier — the off-path must be untouched")
	}
	if tq.state.hasDevicePagedKV() {
		t.Fatal("turboquant state carrier left the paged pool alive — stepToken would read pages while the TQ branch reads codes")
	}
	if !native.state.hasDevicePagedKV() {
		t.Fatal("native hybrid session lost its paged pool — the off-path must be untouched")
	}

	// The per-layer-kind coexistence the design promises: BOTH plain-attention layers (0, 2)
	// TQ-armed with code caches, the gated-delta layer (1) on native recurrent state, and no
	// cross-contamination between the two kinds.
	if !tq.state.kvTQState.on(0) || !tq.state.kvTQState.on(2) {
		t.Fatalf("both plain-attention layers must arm TQ: layer0=%v layer2=%v", tq.state.kvTQState.on(0), tq.state.kvTQState.on(2))
	}
	if tq.state.kvTQState.on(1) {
		t.Fatal("the gated-delta layer must never arm TQ — it owns no KV cache to quantise")
	}
	if tq.state.lb[1].kCache != nil || tq.state.lb[1].vCache != nil {
		t.Fatal("the gated-delta layer must hold no bf16 KV cache either — CacheIndex -1 owns nothing")
	}
	if len(tq.state.gatedDelta) <= 1 || tq.state.gatedDelta[1] == nil {
		t.Fatal("the gated-delta layer must carry its native recurrent-state holder")
	}
	if tq.state.gatedDelta[0] != nil || tq.state.gatedDelta[2] != nil {
		t.Fatal("the plain-attention layers must carry no gated-delta holder")
	}
	// Residency receipt on the armed layers (the recorded-carrier receipt, state-lane sibling):
	// every TQ owner holds code rows at the packed stride, under half the bf16 cache the native
	// lane would keep resident.
	for _, li := range []int{0, 2} {
		lkv, lhd := kvHeadsOf(arch.Layer[li], arch.KVHeads), headDimOf(arch.Layer[li], arch.HeadDim)
		codeBytes := int(bufferLengthFast(tq.state.kvTQState.kCaches[li]))
		bf16Bytes := maxLen * lkv * lhd * bf16Size
		if wantRow := maxLen * tq.state.kvTQState.set.kRowBytes[li]; codeBytes != wantRow {
			t.Fatalf("layer %d: K code cache %dB, want %dB (maxLen×kRowBytes)", li, codeBytes, wantRow)
		}
		if codeBytes*2 >= bf16Bytes {
			t.Fatalf("layer %d: TQ K cache %dB is not under half the bf16 %dB", li, codeBytes, bf16Bytes)
		}
	}

	// Stepwise agreement over the codec: prefill the same prompt (the batched pass declines the
	// state carrier, so this also exercises the sequential TQ prefill THROUGH a gated-delta layer),
	// then step the same ids through both; the hidden trajectories must stay aligned.
	if err := native.PrefillTokens(prompt); err != nil {
		t.Fatalf("native prefill: %v", err)
	}
	if err := tq.PrefillTokens(prompt); err != nil {
		t.Fatalf("tq prefill: %v", err)
	}
	minCos := 1.0
	embScale := embedScaleOf(arch)
	for step, id := range []int32{2, 9, 4, 6, 8, 1, 3, 7} {
		emb, err := embedTokenQuant(g.Embed, g.EmbedScales, g.EmbedBiases, id, arch.Vocab, arch.Hidden, g.GroupSize, g.Bits, embScale)
		if err != nil {
			t.Fatalf("embed %d: %v", step, err)
		}
		hN, err := native.StepWithID(id, emb)
		if err != nil {
			t.Fatalf("native step %d: %v", step, err)
		}
		hT, err := tq.StepWithID(id, emb)
		if err != nil {
			t.Fatalf("tq step %d: %v", step, err)
		}
		a, b := bf16ToF32Slice(hN), bf16ToF32Slice(hT)
		var dot, na, nb float64
		for i := range a {
			dot += float64(a[i]) * float64(b[i])
			na += float64(a[i]) * float64(a[i])
			nb += float64(b[i]) * float64(b[i])
		}
		cos := dot / math.Sqrt(na*nb)
		if cos < minCos {
			minCos = cos
		}
	}
	// The recorded/MoE-carrier session gates hold 0.98 — the SAME codec, the SAME band. Never widened.
	t.Logf("turboquant:4 vs native hybrid stepwise hidden cosine: min %.6f over 8 steps", minCos)
	if minCos < 0.98 {
		t.Fatalf("min stepwise hidden cosine %.6f under the 0.98 codec band", minCos)
	}

	// And the generation surface holds: greedy tokens in-vocab, decoding through both kinds.
	gen, err := tq.Generate([]int32{1, 5, 3}, 4, -1)
	if err != nil {
		t.Fatalf("tq Generate: %v", err)
	}
	for i, id := range gen {
		if id < 0 || int(id) >= arch.Vocab {
			t.Fatalf("token %d = %d out of vocab", i, id)
		}
	}
}

// TestArchQuantSessionTurboQuantHybrid_Bad proves the arch-wide gated-attention decline still fires
// on a hybrid whose full_attention layers set attn_output_gate — the design's rule 4 (docs/design-tq-
// moe-hybrid.md): "Hybrids whose attention is all gated decline loudly." The gated shape (real
// qwen3_5/qwen3_5_next) must never silently arm TQ on the gated lane's resident state.
func TestArchQuantSessionTurboQuantHybrid_Bad(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	gated := model.Arch{
		Hidden: 64, Heads: 2, KVHeads: 1, HeadDim: 128, AttnOutputGate: true,
		Layer: []model.LayerSpec{
			{Mixer: model.MixerGatedDelta, CacheIndex: -1},
			{Mixer: model.MixerAttention, Attention: model.GlobalAttention, HeadDim: 128, KVHeads: 1, CacheIndex: 0},
		},
	}
	if err := tqKVArchServable(gated, nil, &tqKVConfig{kBits: 4, vBits: 4}); err == nil {
		t.Fatal("gated-attention hybrid: expected the arch-wide refusal")
	}
}
