// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"path/filepath"
	"strings"
	"testing"

	"dappco.re/go/inference"
	"dappco.re/go/inference/model"
	"dappco.re/go/inference/model/safetensors"
)

// train_trainer_layers_test.go gates the stage-3 wiring of #40: per-layer projection TargetKeys are
// ACCEPTED by NewLoRATrainer exactly when the model shape is fully covered by the FD-gated
// real-arch reference, and REFUSED — naming the blocking feature — everywhere else. The mode
// classification, shape gate, and adapter resolution are host-gated; the training loop itself is
// runtime-gated end to end (engine capture + host chain), anchored to the ENGINE's own hiddens at
// B = 0.

// TestLoRATargetMode_Good: empty/lm_head keys are the head mode; canonical projection keys are the
// layers mode.
func TestLoRATargetMode_Good(t *testing.T) {
	for _, cfg := range []inference.LoRAConfig{{}, {TargetKeys: []string{"lm_head"}}, {TargetKeys: []string{"lm_head", "lm_head"}}} {
		perLayer, err := loraTargetMode(cfg)
		if err != nil || perLayer {
			t.Fatalf("%v must classify as the head mode: perLayer=%v err=%v", cfg.TargetKeys, perLayer, err)
		}
	}
	for _, keys := range [][]string{{"q_proj", "v_proj"}, {ProjQ, ProjK, ProjV, ProjO, ProjGate, ProjUp, ProjDown}} {
		perLayer, err := loraTargetMode(inference.LoRAConfig{TargetKeys: keys})
		if err != nil || !perLayer {
			t.Fatalf("%v must classify as the layers mode: perLayer=%v err=%v", keys, perLayer, err)
		}
	}
}

// TestLoRATargetMode_Bad: mixed head+layer keys and unknown keys are refused, naming the offender.
func TestLoRATargetMode_Bad(t *testing.T) {
	_, err := loraTargetMode(inference.LoRAConfig{TargetKeys: []string{"lm_head", "q_proj"}})
	if err == nil || !strings.Contains(err.Error(), "separately") {
		t.Fatalf("mixed head+layer keys must be refused with the train-separately pointer; got: %v", err)
	}
	_, err = loraTargetMode(inference.LoRAConfig{TargetKeys: []string{"qq_proj"}})
	if err == nil || !strings.Contains(err.Error(), "qq_proj") {
		t.Fatalf("an unknown key must be refused by name; got: %v", err)
	}
}

// eligibleFixtureModel builds a host-only (no session) synthetic bf16 model + arch of the shape the
// per-layer path accepts: uniform dense full-attention layers, no extras.
func eligibleFixtureModel(nL int) (*BF16Model, model.Arch) {
	// headDim 64 (nHeads 1): hd32 has no sdpa_vector pipeline in the metallib (#28).
	const dModel, nHeads, nKV, headDim, dFF, vocab = 64, 1, 1, 64, 128, 32
	layers := make([]DecodeLayerWeights, nL)
	types := make([]string, nL)
	for li := range layers {
		layers[li] = stableLayerWeights(dModel, nHeads, nKV, headDim, dFF, (li+1)*100)
		types[li] = "full_attention"
	}
	specs := model.DeriveLayers(types, 0)
	for i := range specs {
		specs[i].HeadDim, specs[i].KVHeads = headDim, nKV
	}
	embed := toBF16Bytes(scaleSlice(syntheticFloat32(vocab*dModel, 21), 0.1))
	g := &BF16Model{Layers: layers, Embed: embed, FinalNorm: toBF16Bytes(syntheticFloat32(dModel, 22)), LMHead: embed, Tied: true}
	arch := model.Arch{
		Hidden: dModel, Heads: nHeads, KVHeads: nKV, HeadDim: headDim, FF: dFF, Vocab: vocab,
		GlobalHeadDim: headDim, GlobalKVHeads: nKV,
		Eps: 1e-5, AttnScale: 0.176776695, RopeBase: 10000, RopeScale: 1, RopeLocalBase: 10000,
		RotaryDim: headDim, RotaryDimLocal: headDim, Layer: specs,
	}
	return g, arch
}

// TestValidatePerLayerLoRAShape_Good: the eligible dense shape passes the stage-3 gate, and a
// WELL-FORMED KV-sharing shape now passes it too (#42 — the consumer mirror is FD-gated; which
// adapter placements train is the separate validateSharedKVAdapterSubset gate).
func TestValidatePerLayerLoRAShape_Good(t *testing.T) {
	g, arch := eligibleFixtureModel(2)
	tm := &NativeTokenModel{NativeBackend: &NativeBackend{arch: arch}, bf16: g}
	if err := validatePerLayerLoRAShape(tm); err != nil {
		t.Fatalf("the eligible dense shape must pass: %v", err)
	}
	gS, archS := eligibleFixtureModel(2)
	archS.Layer[1].KVShareFrom = 0 // layer 1 consumes layer 0's cache — the E2B tail shape
	archS.Layer[1].CacheIndex = -1
	tmS := &NativeTokenModel{NativeBackend: &NativeBackend{arch: archS}, bf16: gS}
	if err := validatePerLayerLoRAShape(tmS); err != nil {
		t.Fatalf("a well-formed KV-sharing shape must pass the shape gate (#42): %v", err)
	}
}

// TestValidatePerLayerLoRAShape_Bad: every un-gated feature refuses BY NAME, and every refusal
// points at the head adapter that remains available.
func TestValidatePerLayerLoRAShape_Bad(t *testing.T) {
	check := func(name, wantFeature string, mut func(g *BF16Model, arch *model.Arch)) {
		t.Helper()
		g, arch := eligibleFixtureModel(2)
		mut(g, &arch)
		err := validatePerLayerLoRAShape(&NativeTokenModel{NativeBackend: &NativeBackend{arch: arch}, bf16: g})
		if err == nil {
			t.Fatalf("%s: must refuse", name)
		}
		if !strings.Contains(err.Error(), wantFeature) || !strings.Contains(err.Error(), "lm_head") {
			t.Fatalf("%s: refusal must name %q and the lm_head fallback; got: %s", name, wantFeature, err.Error())
		}
	}
	if err := validatePerLayerLoRAShape(nil); err == nil || !strings.Contains(err.Error(), "lm_head") {
		t.Fatalf("nil model must refuse with the lm_head pointer; got: %v", err)
	}
	// well-formed KV sharing passes since #42 (see _Good); only a topology the decode itself
	// could not serve refuses — a later/self "owner", an owner that is itself a consumer, and a
	// consumer whose cache geometry differs from its owner's.
	check("kv share later owner", "malformed KV-share topology", func(g *BF16Model, arch *model.Arch) {
		arch.Layer[0].KVShareFrom = 1
		arch.Layer[0].CacheIndex = -1
	})
	check("kv share geometry mismatch", "cache geometry", func(g *BF16Model, arch *model.Arch) {
		arch.Layer[1].KVShareFrom = 0
		arch.Layer[1].CacheIndex = -1
		arch.Layer[1].HeadDim = arch.Layer[0].HeadDim * 2
	})
	check("moe", "MoE", func(g *BF16Model, arch *model.Arch) { arch.Experts = 8 })
	check("moe layer", "MoE", func(g *BF16Model, arch *model.Arch) { arch.Layer[0].MoE = true })
	// SoftCap is WIRED since #42's follow-through (train_softcap.go, FD-gated) — a capped shape
	// now VALIDATES.
	{
		g, arch := eligibleFixtureModel(2)
		arch.SoftCap = 30
		if err := validatePerLayerLoRAShape(&NativeTokenModel{NativeBackend: &NativeBackend{arch: arch}, bf16: g}); err != nil {
			t.Fatalf("softcap shape must validate (wired + FD-gated): %v", err)
		}
	}
	// Per-layer head-dim switching is WIRED since the #42 last rung (the mixed-geometry chain FD
	// gate + the real-E2B host probe) — the gemma4 global-vs-sliding shape now VALIDATES.
	{
		g, arch := eligibleFixtureModel(2)
		arch.Layer[1].HeadDim = arch.HeadDim * 2
		arch.Layer[1].KVShareFrom = 1 // owns its cache; only the head dim differs
		if err := validatePerLayerLoRAShape(&NativeTokenModel{NativeBackend: &NativeBackend{arch: arch}, bf16: g}); err != nil {
			t.Fatalf("a per-layer head-dim-switching shape must validate (mixed-geometry FD-gated): %v", err)
		}
	}
	check("qkv bias", "biases", func(g *BF16Model, arch *model.Arch) {
		g.Layers[0].BQ = toBF16Bytes(make([]float32, arch.Heads*arch.HeadDim))
	})
	check("recurrent mixer", "mixer", func(g *BF16Model, arch *model.Arch) { arch.Layer[0].Mixer = model.MixerGatedDelta })
	check("alibi", "non-rotary", func(g *BF16Model, arch *model.Arch) { arch.ALiBi = true })
	check("longrope", "LongRoPE", func(g *BF16Model, arch *model.Arch) { arch.RopeShortFreqs = []float32{1} })
}

// TestBuildLayerAdapters_Good: targets resolve per layer; the ONE ecosystem absence — v_proj on a
// K==V layer — is skipped while every other layer trains it; a request resolving to nothing is
// refused.
func TestBuildLayerAdapters_Good(t *testing.T) {
	g, arch := eligibleFixtureModel(2)
	g.Layers[1].WV = nil // layer 1 is the K==V shape
	layers, err := buildRealLayerTemplates(g, arch)
	if err != nil {
		t.Fatalf("templates: %v", err)
	}
	adapters, err := buildLayerAdapters(layers, []string{ProjQ, ProjV}, 4, 0.02)
	if err != nil {
		t.Fatalf("buildLayerAdapters: %v", err)
	}
	var got []string
	for _, ad := range adapters {
		got = append(got, layerAdapterTensorName(ad.layer, ad.target))
	}
	want := []string{
		"model.layers.0.self_attn.q_proj",
		"model.layers.1.self_attn.q_proj",
		"model.layers.0.self_attn.v_proj", // layer 1's v_proj does not exist (K==V) — skipped
	}
	if len(got) != len(want) {
		t.Fatalf("adapter set: got %v want %v", got, want)
	}
	for _, w := range want {
		found := false
		for _, g2 := range got {
			if g2 == w {
				found = true
			}
		}
		if !found {
			t.Fatalf("adapter set missing %s: got %v", w, got)
		}
	}

	// v_proj-only on an all-K==V model resolves to nothing → refused.
	g2, arch2 := eligibleFixtureModel(2)
	g2.Layers[0].WV, g2.Layers[1].WV = nil, nil
	layers2, err := buildRealLayerTemplates(g2, arch2)
	if err != nil {
		t.Fatalf("templates: %v", err)
	}
	if _, err := buildLayerAdapters(layers2, []string{ProjV}, 4, 0.02); err == nil {
		t.Fatal("a request resolving to no trainable projection must be refused")
	}
}

// TestLayerAdapterTensorName_Good: the on-disk names stay canonical (mlx per-layer format —
// self_attn for the attention projections, mlp for the feed-forward ones).
func TestLayerAdapterTensorName_Good(t *testing.T) {
	if got := layerAdapterTensorName(3, ProjQ); got != "model.layers.3.self_attn.q_proj" {
		t.Fatalf("q_proj name: %s", got)
	}
	if got := layerAdapterTensorName(0, ProjDown); got != "model.layers.0.mlp.down_proj" {
		t.Fatalf("down_proj name: %s", got)
	}
}

// perLayerTrainerFixture opens the runtime fixture: the eligible synthetic model bound as a
// NativeTokenModel (real ArchSession under the trainer).
func perLayerTrainerFixture(t *testing.T, nL int) *NativeTokenModel {
	t.Helper()
	g, arch := eligibleFixtureModel(nL)
	tm, err := NewBF16TokenModel(g, arch, 16)
	if err != nil {
		t.Fatalf("NewBF16TokenModel: %v", err)
	}
	return tm
}

// TestNewLoRATrainerPerLayer_Good: the eligible shape ACCEPTS per-layer TargetKeys (the #31
// refusal lifts exactly where the FD gates cover), and at B = 0 the trainer's host layer chain
// reproduces the ENGINE's own captured hiddens layer by layer — the mirror-the-engine receipt the
// whole reference exists for.
func TestNewLoRATrainerPerLayer_Good(t *testing.T) {
	requireNativeRuntime(t)
	tm := perLayerTrainerFixture(t, 2)
	tr, err := NewLoRATrainer(tm, inference.TrainingConfig{
		LoRA:         inference.LoRAConfig{Rank: 4, Alpha: 8, TargetKeys: []string{"q_proj", "v_proj"}},
		LearningRate: 0.02,
	})
	if err != nil {
		t.Fatalf("NewLoRATrainer must accept per-layer targets on the eligible shape: %v", err)
	}
	defer func() { _ = tr.Close() }()
	if !tr.perLayer || len(tr.adapters) != 4 {
		t.Fatalf("expected the layers mode with 2 layers × {q,v}: perLayer=%v adapters=%d", tr.perLayer, len(tr.adapters))
	}

	// B = 0 parity anchor: the host chain must reproduce the engine's captured hiddens.
	ids := []int32{1, 2, 3, 4, 5, 6}
	embeds, perLayer, err := tr.sess.ForwardCaptureHiddens(ids)
	if err != nil {
		t.Fatalf("ForwardCaptureHiddens: %v", err)
	}
	sets, err := tr.effectiveWeightSets()
	if err != nil {
		t.Fatalf("effectiveWeightSets: %v", err)
	}
	_, tapes, err := tr.layerChainForward(ids, embeds, sets)
	if err != nil {
		t.Fatalf("layerChainForward: %v", err)
	}
	for li := range tapes {
		host := toBF16Bytes(tapes[li].out)
		cos := cosineBF16(host, perLayer[li])
		t.Logf("layer %d host-chain vs engine-capture cosine=%.6f", li, cos)
		if cos < 0.999 {
			t.Fatalf("layer %d: the host chain diverges from the engine's own forward (cosine=%.6f) — the reference does not mirror the engine", li, cos)
		}
	}
}

// TestLoRATrainerPerLayerSFT_Good: the per-layer training loop end to end on the real session —
// the loss falls over steps, a masked final target makes Step bit-blind to that token (the #39
// semantics on the LAYER path), and Save writes the canonical per-layer adapter package.
func TestLoRATrainerPerLayerSFT_Good(t *testing.T) {
	requireNativeRuntime(t)
	tm := perLayerTrainerFixture(t, 2)
	open := func() *LoRATrainer {
		tr, err := NewLoRATrainer(tm, inference.TrainingConfig{
			LoRA:         inference.LoRAConfig{Rank: 4, Alpha: 8, TargetKeys: []string{"q_proj", "v_proj", "down_proj"}},
			LearningRate: 0.02,
		})
		if err != nil {
			t.Fatalf("NewLoRATrainer: %v", err)
		}
		return tr
	}

	tr := open()
	defer func() { _ = tr.Close() }()
	batch := inference.Batch{TokenIDs: [][]int32{{1, 2, 3, 4, 5, 6}, {7, 8, 9, 10}}}
	loss0, err := tr.Loss(batch)
	if err != nil {
		t.Fatalf("initial loss: %v", err)
	}
	var lossLast float64
	const steps = 30
	for s := range steps {
		l, serr := tr.Step(batch)
		if serr != nil {
			t.Fatalf("step %d: %v", s, serr)
		}
		lossLast = l
		if s%10 == 0 || s == steps-1 {
			t.Logf("per-layer SFT step %d: loss %.4f", s, l)
		}
	}
	if lossLast >= loss0 {
		t.Fatalf("per-layer LoRA SFT did not reduce loss: first=%.4f last=%.4f", loss0, lossLast)
	}
	t.Logf("per-layer LoRA SFT receipt: loss %.4f -> %.4f over %d steps (q/v/down on 2 layers)", loss0, lossLast, steps)

	// LossMask on the layer path: masking the final target makes Step blind to that token.
	ids1 := []int32{1, 2, 3, 4, 5, 6}
	ids2 := []int32{1, 2, 3, 4, 5, 29}
	mask := inference.LossMask{Values: [][]float32{{1, 1, 1, 1, 1, 0}}}
	step := func(ids []int32) float64 {
		tr2 := open()
		defer func() { _ = tr2.Close() }()
		l, serr := tr2.Step(inference.Batch{TokenIDs: [][]int32{ids}, LossMask: mask})
		if serr != nil {
			t.Fatalf("masked step: %v", serr)
		}
		return l
	}
	if l1, l2 := step(ids1), step(ids2); l1 != l2 {
		t.Fatalf("a masked position's token changed the per-layer loss: %v vs %v", l1, l2)
	}

	// Save: the canonical per-layer adapter package round-trips through the safetensors reader.
	dir := filepath.Join(t.TempDir(), "adapter")
	if err := tr.Save(dir); err != nil {
		t.Fatalf("Save: %v", err)
	}
	tensors, err := safetensors.Load(filepath.Join(dir, "adapter.safetensors"))
	if err != nil {
		t.Fatalf("load saved adapter: %v", err)
	}
	ta, ok := tensors["model.layers.0.self_attn.q_proj.lora_a"]
	if !ok {
		t.Fatalf("saved adapter missing the canonical q_proj lora_a tensor; have %d tensors", len(tensors))
	}
	if len(ta.Shape) != 2 || ta.Shape[0] != 4 {
		t.Fatalf("q_proj lora_a shape: %v (want [4 dModel])", ta.Shape)
	}
	if _, ok := tensors["model.layers.1.mlp.down_proj.lora_b"]; !ok {
		t.Fatal("saved adapter missing the canonical down_proj lora_b tensor")
	}
}

// sharingFixtureTokenModel builds the runtime KV-sharing fixture: nL uniform full-attention
// layers with the LAST numShared sharing the most recent owner's cache (model.DeriveLayers — the
// gemma4 E2B/E4B tail shape), bound as a real NativeTokenModel.
func sharingFixtureTokenModel(t *testing.T, nL, numShared int) *NativeTokenModel {
	t.Helper()
	// headDim 64 (nHeads 1): hd32 has no sdpa_vector pipeline in the metallib (#28).
	const dModel, nHeads, nKV, headDim, dFF, vocab = 64, 1, 1, 64, 128, 32
	layers := make([]DecodeLayerWeights, nL)
	types := make([]string, nL)
	for li := range layers {
		layers[li] = stableLayerWeights(dModel, nHeads, nKV, headDim, dFF, (li+1)*100)
		types[li] = "full_attention"
	}
	specs := model.DeriveLayers(types, numShared)
	for i := range specs {
		specs[i].HeadDim, specs[i].KVHeads = headDim, nKV
	}
	embed := toBF16Bytes(scaleSlice(syntheticFloat32(vocab*dModel, 21), 0.1))
	g := &BF16Model{Layers: layers, Embed: embed, FinalNorm: toBF16Bytes(syntheticFloat32(dModel, 22)), LMHead: embed, Tied: true}
	arch := model.Arch{
		Hidden: dModel, Heads: nHeads, KVHeads: nKV, HeadDim: headDim, FF: dFF, Vocab: vocab,
		GlobalHeadDim: headDim, GlobalKVHeads: nKV,
		Eps: 1e-5, AttnScale: 0.176776695, RopeBase: 10000, RopeScale: 1, RopeLocalBase: 10000,
		RotaryDim: headDim, RotaryDimLocal: headDim, Layer: specs,
	}
	tm, err := NewBF16TokenModel(g, arch, 16)
	if err != nil {
		t.Fatalf("NewBF16TokenModel: %v", err)
	}
	return tm
}

// TestNewLoRATrainerSharedKV_Good: the FULL per-layer LoRA shape OPENS on a real KV-sharing
// session (#42 — owner + 2 consumers, all seven targets: k/v resolve on the owner, consumers
// carry the other five) — and at B = 0 the share-aware host chain reproduces the ENGINE's own
// captured hiddens layer by layer, consumers included: the receipt that the consumer mirror
// carries encAttnHalfShared's exact semantics (owner-position rope'd K, value-normed V, read
// as-is).
func TestNewLoRATrainerSharedKV_Good(t *testing.T) {
	requireNativeRuntime(t)
	tm := sharingFixtureTokenModel(t, 3, 2) // layer 0 owns; layers 1 and 2 attend its cache
	tr, err := NewLoRATrainer(tm, inference.TrainingConfig{
		LoRA: inference.LoRAConfig{Rank: 4, Alpha: 8, TargetKeys: []string{
			ProjQ, ProjK, ProjV, ProjO, ProjGate, ProjUp, ProjDown,
		}},
		LearningRate: 0.02,
	})
	if err != nil {
		t.Fatalf("NewLoRATrainer must accept the full per-layer shape on a sharing stack: %v", err)
	}
	defer func() { _ = tr.Close() }()
	// 7 targets on the owner + 5 on each consumer (no k/v tensors there — the documented skip).
	if !tr.perLayer || len(tr.adapters) != 17 {
		t.Fatalf("expected the layers mode with 7 owner + 2×5 consumer adapters: perLayer=%v adapters=%d", tr.perLayer, len(tr.adapters))
	}

	// B = 0 parity anchor: the share-aware host chain must reproduce the engine's captured
	// hiddens — the consumer layers are the ones under test (they attend layer 0's cache).
	ids := []int32{1, 2, 3, 4, 5, 6}
	embeds, perLayer, err := tr.sess.ForwardCaptureHiddens(ids)
	if err != nil {
		t.Fatalf("ForwardCaptureHiddens: %v", err)
	}
	sets, err := tr.effectiveWeightSets()
	if err != nil {
		t.Fatalf("effectiveWeightSets: %v", err)
	}
	_, tapes, err := tr.layerChainForward(ids, embeds, sets)
	if err != nil {
		t.Fatalf("layerChainForward: %v", err)
	}
	for li := range tapes {
		host := toBF16Bytes(tapes[li].out)
		cos := cosineBF16(host, perLayer[li])
		t.Logf("layer %d host-chain vs engine-capture cosine=%.6f (KVShareFrom=%d)", li, cos, tm.arch.Layer[li].KVShareFrom)
		if cos < 0.999 {
			t.Fatalf("layer %d: the shared-KV host chain diverges from the engine's own forward (cosine=%.6f) — the consumer mirror does not carry encAttnHalfShared's semantics", li, cos)
		}
	}

	// and the loop trains: a few steps on a tiny batch reduce the loss through the shared stack.
	batch := inference.Batch{TokenIDs: [][]int32{{1, 2, 3, 4, 5, 6}}}
	loss0, err := tr.Loss(batch)
	if err != nil {
		t.Fatalf("initial loss: %v", err)
	}
	var lossLast float64
	for s := range 20 {
		l, serr := tr.Step(batch)
		if serr != nil {
			t.Fatalf("step %d: %v", s, serr)
		}
		lossLast = l
	}
	if lossLast >= loss0 {
		t.Fatalf("shared-KV per-layer SFT did not reduce loss: first=%.4f last=%.4f", loss0, lossLast)
	}
	t.Logf("shared-KV per-layer SFT receipt: loss %.4f -> %.4f over 20 steps (all 7 targets, owner + 2 consumers)", loss0, lossLast)
}

// TestNewLoRATrainerPerLayer_Bad: the refusal boundary in both directions on a REAL model — a
// shape with a genuinely un-gated feature (the final-logit soft-cap) refuses per-layer targets
// naming the feature, while the head adapter still opens on the very same model. (KV sharing no
// longer refuses — the owner-routed backward covers it; see TestNewLoRATrainerSharedKV_Good.)
func TestNewLoRATrainerPerLayer_Bad(t *testing.T) {
	requireNativeRuntime(t)
	g, arch := eligibleFixtureModel(2)
	arch.LogitsScaling = 2 // an un-gated head feature (weight-independent, so the head adapter still opens below)
	tm, err := NewBF16TokenModel(g, arch, 16)
	if err != nil {
		t.Fatalf("NewBF16TokenModel: %v", err)
	}

	_, err = NewLoRATrainer(tm, inference.TrainingConfig{LoRA: inference.LoRAConfig{TargetKeys: []string{"q_proj"}}})
	if err == nil {
		t.Fatal("per-layer targets on a logit-scaling shape must be refused")
	}
	for _, want := range []string{"q_proj", "final-logit scaling", "lm_head"} {
		if !strings.Contains(err.Error(), want) {
			t.Fatalf("the boundary refusal must name %q; got: %s", want, err.Error())
		}
	}

	head, err := NewLoRATrainer(tm, inference.TrainingConfig{LoRA: inference.LoRAConfig{Rank: 4, Alpha: 8}})
	if err != nil {
		t.Fatalf("the head adapter must still open on the refused shape: %v", err)
	}
	_ = head.Close()
}
