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
)

// arch_session_tq_moe_test.go — the STATE-carrier TurboQuant gates (#48
// follow-on, docs/design-tq-moe-hybrid.md): a gemma4-MoE-shaped session (the
// 26B-A4B pattern — host router, stepToken decode, no arch ICB) arms the
// state-lane TQ carrier, decodes within the session codec band against its
// native twin over the SAME weights, keeps the residency win, and declines
// every bf16-shaped KV consumer exactly as the recorded carrier does.

// tqMoETestQuantModel assembles the shared synthetic all-global MoE gemma4
// (head dim 128 — a TQ-instantiated width; 4-bit experts + attention, 8-bit
// local MLP + router — the 26B-A4B QAT pattern) both sessions load.
func tqMoETestQuantModel(t *testing.T) (*QuantModel, model.Arch) {
	t.Helper()
	const numLayers, numExperts, topK = 2, 4, 2
	quant := &model.QuantConfig{GroupSize: 64, Bits: 4, Overrides: map[string]model.ModuleQuant{}}
	for i := range numLayers {
		for _, m := range []string{"mlp.gate_proj", "mlp.up_proj", "mlp.down_proj", "router.proj"} {
			quant.Overrides[core.Sprintf("model.layers.%d.%s", i, m)] = model.ModuleQuant{GroupSize: 64, Bits: 8}
		}
	}
	cfg := g4.Config{
		HiddenSize: 64, NumHiddenLayers: numLayers, IntermediateSize: 128,
		NumAttentionHeads: 2, NumKeyValueHeads: 1, HeadDim: 128, VocabSize: 32, RMSNormEps: 1e-6,
		EnableMoEBlock: true, NumExperts: numExperts, TopKExperts: topK, MoEIntermediateSize: 64,
		Quantization: quant,
	}
	arch, err := cfg.Arch()
	if err != nil {
		t.Fatalf("Arch: %v", err)
	}
	if !arch.HasMoE() {
		t.Fatal("arch should be MoE")
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

// TestArchQuantSessionTurboQuantMoE_Good gates the state-carrier TQ session
// against its native MoE twin: same weights, same prompt, stepwise hidden
// agreement within the session codec band, valid greedy tokens, the STATE
// carrier armed (no ICB exists to carry TQ on a MoE stack), the paged pool
// declined, and the residency receipt (code cache ≪ the bf16 equivalent).
func TestArchQuantSessionTurboQuantMoE_Good(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	g, arch := tqMoETestQuantModel(t)
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
		t.Fatal("MoE session recorded an arch ICB — the state carrier gates assume stepToken decode")
	}
	if !tq.state.tqStateArmed() {
		t.Fatal("turboquant MoE session did not arm the state carrier")
	}
	if !tq.hasKVTQAny() {
		t.Fatal("hasKVTQAny must report the state carrier — every TQ decline gates on it")
	}
	if native.state.tqStateArmed() || native.hasKVTQAny() {
		t.Fatal("native MoE session armed a TQ carrier — the off-path must be untouched")
	}
	if tq.state.hasDevicePagedKV() {
		t.Fatal("turboquant state carrier left the paged pool alive — stepToken would read pages while the TQ branch reads codes")
	}
	if !native.state.hasDevicePagedKV() {
		t.Fatal("native MoE session lost its paged pool — the off-path must be untouched")
	}
	// Residency receipt: every TQ owner holds code rows at the packed stride,
	// under half the bf16 cache the native lane would keep resident.
	armed := 0
	for li := range arch.Layer {
		if !tq.state.kvTQState.on(li) {
			continue
		}
		armed++
		lkv, lhd := kvHeadsOf(arch.Layer[li], arch.KVHeads), headDimOf(arch.Layer[li], arch.HeadDim)
		codeBytes := int(bufferLengthFast(tq.state.kvTQState.kCaches[li]))
		bf16Bytes := maxLen * lkv * lhd * bf16Size
		if wantRow := maxLen * tq.state.kvTQState.set.kRowBytes[li]; codeBytes != wantRow {
			t.Fatalf("layer %d: K code cache %dB, want %dB (maxLen×kRowBytes)", li, codeBytes, wantRow)
		}
		if codeBytes*2 >= bf16Bytes {
			t.Fatalf("layer %d: TQ K cache %dB is not under half the bf16 %dB", li, codeBytes, bf16Bytes)
		}
		if tq.state.lb[li].kCache != nil {
			t.Fatalf("layer %d: a bf16 lb cache exists beside the code cache — the dead-duplicate/mixed-write hazard", li)
		}
	}
	if armed == 0 {
		t.Fatal("no layer armed on the state carrier")
	}

	// Stepwise agreement over the codec: prefill the same prompt (the batched
	// pass declines the state carrier, so this also exercises the sequential
	// TQ prefill), then step the same ids through both; the hidden
	// trajectories must stay aligned.
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
	// The recorded-carrier session gate holds 0.98 (measured 0.999182 on its
	// dense synthetic) — the SAME codec, the SAME band. Never widened.
	t.Logf("turboquant:4 vs native MoE stepwise hidden cosine: min %.6f over 8 steps", minCos)
	if minCos < 0.98 {
		t.Fatalf("min stepwise hidden cosine %.6f under the 0.98 codec band", minCos)
	}

	// And the generation surface holds: greedy tokens in-vocab.
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

// TestArchQuantSessionTurboQuantMoEShared_Good gates the SHARED-owner read on
// the state carrier — the gemma4 num_kv_shared_layers tail (the 26B shape): a
// global sharer attends its TQ owner's codes through encAttnHalfSharedKVTQ
// (q-only leg, no store), and the whole stack stays inside the session codec
// band against its native twin.
func TestArchQuantSessionTurboQuantMoEShared_Good(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	const numLayers = 3
	quant := &model.QuantConfig{GroupSize: 64, Bits: 4, Overrides: map[string]model.ModuleQuant{}}
	for i := range numLayers {
		for _, m := range []string{"mlp.gate_proj", "mlp.up_proj", "mlp.down_proj", "router.proj"} {
			quant.Overrides[core.Sprintf("model.layers.%d.%s", i, m)] = model.ModuleQuant{GroupSize: 64, Bits: 8}
		}
	}
	cfg := g4.Config{
		HiddenSize: 64, NumHiddenLayers: numLayers, IntermediateSize: 128,
		NumAttentionHeads: 2, NumKeyValueHeads: 1, HeadDim: 128, VocabSize: 32, RMSNormEps: 1e-6,
		SlidingWindow: 8, LayerTypes: []string{"full_attention", "sliding_attention", "full_attention"},
		NumKVSharedLayers: 1,
		EnableMoEBlock:    true, NumExperts: 4, TopKExperts: 2, MoEIntermediateSize: 64,
		Quantization: quant,
	}
	arch, err := cfg.Arch()
	if err != nil {
		t.Fatalf("Arch: %v", err)
	}
	if arch.Layer[2].OwnsCache() || arch.Layer[2].KVShareFrom != 0 {
		t.Fatalf("fixture wants layer 2 sharing layer 0's KV, got %+v", arch.Layer[2])
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
	const maxLen = 32
	native, err := newArchQuantSessionShardsWithHeadConfig(g, arch, maxLen, nil, nil, archSessionConfig{})
	if err != nil {
		t.Fatalf("native session: %v", err)
	}
	tq, err := newArchQuantSessionShardsWithHeadConfig(g, arch, maxLen, nil, nil, archSessionConfig{kvCacheMode: "turboquant:4"})
	if err != nil {
		t.Fatalf("turboquant session: %v", err)
	}
	if !tq.state.kvTQState.on(0) {
		t.Fatal("shared GLOBAL owner 0 must arm TQ (global sharers are served; only sliding sharers force native)")
	}
	if tq.state.kvTQState.on(1) || tq.state.kvTQState.on(2) {
		t.Fatal("sliding owner / sharer must never arm TQ")
	}
	prompt := []int32{1, 5, 3, 7}
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
	t.Logf("turboquant:4 vs native shared-owner MoE stepwise hidden cosine: min %.6f over 8 steps", minCos)
	if minCos < 0.98 {
		t.Fatalf("min stepwise hidden cosine %.6f under the 0.98 codec band", minCos)
	}
}

// TestArchQuantSessionTurboQuantMoE_Bad proves the state-carrier refusals: a
// MoE stack whose head dim has no TQ instantiation refuses at build (never
// silently native), and an armed state carrier declines the bf16-shaped KV
// snapshot exactly as the recorded carrier does.
func TestArchQuantSessionTurboQuantMoE_Bad(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	// head dim 64 — no TQ instantiation → the build must refuse.
	const numLayers = 2
	quant := &model.QuantConfig{GroupSize: 64, Bits: 4, Overrides: map[string]model.ModuleQuant{}}
	for i := range numLayers {
		for _, m := range []string{"mlp.gate_proj", "mlp.up_proj", "mlp.down_proj", "router.proj"} {
			quant.Overrides[core.Sprintf("model.layers.%d.%s", i, m)] = model.ModuleQuant{GroupSize: 64, Bits: 8}
		}
	}
	cfg := g4.Config{
		HiddenSize: 64, NumHiddenLayers: numLayers, IntermediateSize: 128,
		NumAttentionHeads: 2, NumKeyValueHeads: 1, HeadDim: 64, VocabSize: 32, RMSNormEps: 1e-6,
		EnableMoEBlock: true, NumExperts: 4, TopKExperts: 2, MoEIntermediateSize: 64,
		Quantization: quant,
	}
	arch64, err := cfg.Arch()
	if err != nil {
		t.Fatalf("Arch: %v", err)
	}
	ts := moeQuantTensors(t, arch64, quant)
	lm, err := model.Assemble(ts, arch64, model.StandardWeightNames())
	if err != nil {
		t.Fatalf("model.Assemble: %v", err)
	}
	g64, err := loadedToQuant(lm, quant.GroupSize, quant.Bits)
	if err != nil {
		t.Fatalf("loadedToQuant: %v", err)
	}
	if _, err := newArchQuantSessionShardsWithHeadConfig(g64, arch64, 16, nil, nil, archSessionConfig{kvCacheMode: "turboquant:4"}); err == nil {
		t.Fatal("MoE head dim 64: expected the session build to refuse (no qualifying global layer)")
	}

	// Armed state carrier: CaptureKV (bf16-shaped) declines loudly.
	g, arch := tqMoETestQuantModel(t)
	tq, err := newArchQuantSessionShardsWithHeadConfig(g, arch, 16, nil, nil, archSessionConfig{kvCacheMode: "turboquant:3.5"})
	if err != nil {
		t.Fatalf("turboquant session: %v", err)
	}
	if !tq.state.tqStateArmed() {
		t.Fatal("turboquant:3.5 MoE session did not arm the state carrier")
	}
	if err := tq.PrefillTokens([]int32{1, 5, 3}); err != nil {
		t.Fatalf("prefill: %v", err)
	}
	if _, err := tq.CaptureKV(); err == nil {
		t.Fatal("CaptureKV on a state-carrier turboquant session: expected the decline")
	}
}
