// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"bytes"
	"testing"
	"unsafe"

	core "dappco.re/go"
	"dappco.re/go/inference/model"
	g4 "dappco.re/go/inference/model/gemma4"
	"dappco.re/go/inference/model/safetensors"
)

// mtp_rows_driver_test.go — the #53 WIRING acceptance pin: verifyAssistantDraftHiddens (the
// byte-exact greedy MTP verify lane, assistant_load.go) with LTHN_MTP_ROWS_MOE=1 (the layer-major
// driver, mtp_rows_driver.go) must produce EXACTLY the bytes the unchanged row-major per-row lane
// produces — hiddens, KV cache contents, and the greedy token each row implies — on a synthetic
// gemma4 26B-A4B-shaped quant MoE session (moe_session_test.go's moeQuantTensors fixture family),
// with an engagement proof so the compare cannot pass by both runs quietly taking the same
// fallback.

const (
	mtpRowsDriverTestDModel     = 512 // qmv_rows tiled envelope: outDim%8==0 && inDim%512==0 on every projection
	mtpRowsDriverTestDFF        = 512
	mtpRowsDriverTestExpertDFF  = 512
	mtpRowsDriverTestNumExperts = 2 // == TopK: every row routes to BOTH experts, deterministically
	mtpRowsDriverTestTopK       = 2 // grouping every expert's pairs to size K — no router luck needed
	mtpRowsDriverTestNHeads     = 8
	mtpRowsDriverTestNKV        = 2
	mtpRowsDriverTestHeadDim    = 64
	mtpRowsDriverTestVocab      = 64
	mtpRowsDriverTestNumLayers  = 2
	mtpRowsDriverTestMaxLen     = 24
)

// mtpRowsDriverNewSession assembles a fresh ArchSession from the given arch/quant/tensors and
// steps it through prompt — leaving it positioned exactly where a live MTP verify block would
// start (mirrors TestLoadGemma4QuantMoEConcurrentMatchesSerial's gen() closure: rebuild the
// session per run, reuse only the deterministic tensor bytes).
func mtpRowsDriverNewSession(t *testing.T, arch model.Arch, quant *model.QuantConfig, ts map[string]safetensors.Tensor, prompt []int32) *ArchSession {
	t.Helper()
	lm, err := model.Assemble(ts, arch, model.StandardWeightNames())
	if err != nil {
		t.Fatalf("model.Assemble: %v", err)
	}
	g, err := loadedToQuant(lm, quant.GroupSize, quant.Bits)
	if err != nil {
		t.Fatalf("loadedToQuant: %v", err)
	}
	sess, err := NewArchQuantSession(g, arch, mtpRowsDriverTestMaxLen)
	if err != nil {
		t.Fatalf("NewArchQuantSession: %v", err)
	}
	for _, id := range prompt {
		if _, err := sess.stepID(id); err != nil {
			t.Fatalf("stepID(prompt): %v", err)
		}
	}
	return sess
}

// mtpRowsDriverKVSnapshot copies every owning layer's live KV cache bytes — the acceptance
// criterion's "KV cache contents" half of the bytes.Equal set.
func mtpRowsDriverKVSnapshot(s *ArchSession) [][2][]byte {
	out := make([][2][]byte, len(s.state.lb))
	for li := range s.state.lb {
		lb := &s.state.lb[li]
		if lb.kCache == nil || lb.vCache == nil {
			continue
		}
		k := append([]byte(nil), unsafe.Slice((*byte)(lb.kCache.Contents()), int(lb.kvCacheBytes))...)
		v := append([]byte(nil), unsafe.Slice((*byte)(lb.vCache.Contents()), int(lb.kvCacheBytes))...)
		out[li] = [2][]byte{k, v}
	}
	return out
}

// TestMTPRowsDriverVerifyMatchesRowMajor_Good pins the #53 wiring: verifyAssistantDraftHiddens's
// byte-exact greedy lane with LTHN_MTP_ROWS_MOE=1 (mtpRowsDriverEligible engages the layer-major
// driver) produces bytes.Equal hiddens, KV cache contents, and per-row greedy token ids against
// the unchanged row-major per-row lane (the lever off) — AND the compare is proven non-vacuous:
// both the wiring-level engagement counter (mtpRowsDriverEngaged) and the primitive's own
// grouped-expert counter (mtpRowsMoEMaxGroupSize) must have moved during the lever-on run.
func TestMTPRowsDriverVerifyMatchesRowMajor_Good(t *testing.T) {
	requireNativeRuntime(t)
	quant := &model.QuantConfig{GroupSize: 64, Bits: 4, Overrides: map[string]model.ModuleQuant{}}
	for i := range mtpRowsDriverTestNumLayers {
		for _, m := range []string{"mlp.gate_proj", "mlp.up_proj", "mlp.down_proj", "router.proj"} {
			quant.Overrides[core.Sprintf("model.layers.%d.%s", i, m)] = model.ModuleQuant{GroupSize: 64, Bits: 8}
		}
	}
	cfg := g4.Config{
		HiddenSize: mtpRowsDriverTestDModel, NumHiddenLayers: mtpRowsDriverTestNumLayers,
		IntermediateSize:  mtpRowsDriverTestDFF,
		NumAttentionHeads: mtpRowsDriverTestNHeads, NumKeyValueHeads: mtpRowsDriverTestNKV,
		HeadDim: mtpRowsDriverTestHeadDim, VocabSize: mtpRowsDriverTestVocab, RMSNormEps: 1e-6,
		EnableMoEBlock: true, NumExperts: mtpRowsDriverTestNumExperts, TopKExperts: mtpRowsDriverTestTopK,
		MoEIntermediateSize: mtpRowsDriverTestExpertDFF,
		Quantization:        quant,
	}
	arch, err := cfg.Arch()
	if err != nil {
		t.Fatalf("Arch: %v", err)
	}
	if !arch.HasMoE() {
		t.Fatal("fixture arch should be MoE")
	}
	ts := moeQuantTensors(t, arch, quant)

	probe := mtpRowsDriverNewSession(t, arch, quant, ts, nil)
	if !mtpRowsDriverEligible(&probe.state) {
		t.Fatal("fixture geometry declined mtpRowsDriverEligible — the fixture must exercise the layer-major driver")
	}

	prompt := []int32{1, 5, 3}
	draft := []int32{2, 6, 4, 7} // K=4 — safely below batchedDenseICBMaxRows(16), so
	// verifyBatchedHiddens (the sampled fold) declines this exact-lane call before either run below
	// ever reaches it: both runs contend for the SAME candidate path (row-major vs layer-major).

	run := func(forced bool) (hiddens [][]byte, kv [][2][]byte, greedy []int32) {
		t.Helper()
		saved := mtpRowsMoEForced
		mtpRowsMoEForced = forced
		defer func() { mtpRowsMoEForced = saved }()

		sess := mtpRowsDriverNewSession(t, arch, quant, ts, prompt)
		hs, err := sess.verifyAssistantDraftHiddens(draft, true)
		if err != nil {
			t.Fatalf("verifyAssistantDraftHiddens(forced=%v): %v", forced, err)
		}
		if len(hs) != len(draft) {
			t.Fatalf("forced=%v: got %d hiddens, want %d", forced, len(hs), len(draft))
		}
		greedy = make([]int32, len(hs))
		for i, h := range hs {
			tok, gerr := sess.greedyOf(h)
			if gerr != nil {
				t.Fatalf("greedyOf row %d (forced=%v): %v", i, forced, gerr)
			}
			greedy[i] = tok
		}
		return hs, mtpRowsDriverKVSnapshot(sess), greedy
	}

	engagedBefore, groupBefore := mtpRowsDriverEngaged.Load(), mtpRowsMoEMaxGroupSize.Load()
	hiddensOff, kvOff, greedyOff := run(false)
	engagedAfterOff := mtpRowsDriverEngaged.Load()
	if engagedAfterOff != engagedBefore {
		t.Fatal("the layer-major driver engaged with the lever OFF — mtpRowsMoEForced routing is broken")
	}

	hiddensOn, kvOn, greedyOn := run(true)
	engagedAfterOn, groupAfterOn := mtpRowsDriverEngaged.Load(), mtpRowsMoEMaxGroupSize.Load()
	if engagedAfterOn == engagedAfterOff {
		t.Fatal("the layer-major driver never engaged with the lever ON — compare is vacuous")
	}
	if groupAfterOn == groupBefore {
		t.Fatal("mtpRowsMoEBatched never grouped >1 row onto one expert during the lever-on run — compare never engaged the grouped lane")
	}

	if len(hiddensOff) != len(hiddensOn) {
		t.Fatalf("row count mismatch: off=%d on=%d", len(hiddensOff), len(hiddensOn))
	}
	for i := range hiddensOff {
		if !bytes.Equal(hiddensOff[i], hiddensOn[i]) {
			t.Fatalf("verify hidden row %d diverged between row-major and layer-major (lever off vs on)", i)
		}
	}
	if len(kvOff) != len(kvOn) {
		t.Fatalf("KV layer count mismatch: off=%d on=%d", len(kvOff), len(kvOn))
	}
	for li := range kvOff {
		if !bytes.Equal(kvOff[li][0], kvOn[li][0]) {
			t.Fatalf("layer %d K cache diverged between row-major and layer-major", li)
		}
		if !bytes.Equal(kvOff[li][1], kvOn[li][1]) {
			t.Fatalf("layer %d V cache diverged between row-major and layer-major", li)
		}
	}
	if len(greedyOff) != len(greedyOn) {
		t.Fatalf("greedy row count mismatch: off=%d on=%d", len(greedyOff), len(greedyOn))
	}
	for i := range greedyOff {
		if greedyOff[i] != greedyOn[i] {
			t.Fatalf("row %d greedy token diverged: off=%d on=%d — the accepted-token sequence would differ", i, greedyOff[i], greedyOn[i])
		}
	}
	t.Logf("layer-major verify == row-major verify: %d rows, %d KV layers, greedy %v (engaged %d driver block(s), max expert group %d)",
		len(hiddensOff), len(kvOff), greedyOff, engagedAfterOn-engagedAfterOff, groupAfterOn)
}

// TestMTPRowsDriverEligible_Bad pins the whole-block decline: a session that is otherwise
// eligible but carries ONE dense (non-MoE) layer must decline entirely — this driver is
// uniform-MoE only (gemma4 applies MoE to every layer; a mixed model is out of scope).
func TestMTPRowsDriverEligible_Bad(t *testing.T) {
	requireNativeRuntime(t)
	quant := &model.QuantConfig{GroupSize: 64, Bits: 4, Overrides: map[string]model.ModuleQuant{}}
	for i := range mtpRowsDriverTestNumLayers {
		for _, m := range []string{"mlp.gate_proj", "mlp.up_proj", "mlp.down_proj", "router.proj"} {
			quant.Overrides[core.Sprintf("model.layers.%d.%s", i, m)] = model.ModuleQuant{GroupSize: 64, Bits: 8}
		}
	}
	cfg := g4.Config{
		HiddenSize: mtpRowsDriverTestDModel, NumHiddenLayers: mtpRowsDriverTestNumLayers,
		IntermediateSize:  mtpRowsDriverTestDFF,
		NumAttentionHeads: mtpRowsDriverTestNHeads, NumKeyValueHeads: mtpRowsDriverTestNKV,
		HeadDim: mtpRowsDriverTestHeadDim, VocabSize: mtpRowsDriverTestVocab, RMSNormEps: 1e-6,
		EnableMoEBlock: true, NumExperts: mtpRowsDriverTestNumExperts, TopKExperts: mtpRowsDriverTestTopK,
		MoEIntermediateSize: mtpRowsDriverTestExpertDFF,
		Quantization:        quant,
	}
	arch, err := cfg.Arch()
	if err != nil {
		t.Fatalf("Arch: %v", err)
	}
	ts := moeQuantTensors(t, arch, quant)
	sess := mtpRowsDriverNewSession(t, arch, quant, ts, []int32{1})
	if !mtpRowsDriverEligible(&sess.state) {
		t.Fatal("fixture session should be eligible before mutation")
	}
	saved := sess.state.moeQuant[0]
	sess.state.moeQuant[0] = nil
	defer func() { sess.state.moeQuant[0] = saved }()
	if mtpRowsDriverEligible(&sess.state) {
		t.Fatal("a dense (non-MoE) layer must decline the whole block")
	}
}

// TestMTPRowsDriverEligible_Ugly pins the defensive nil/shape guards — no metallib required,
// since both decline before any GPU-dependent check runs.
func TestMTPRowsDriverEligible_Ugly(t *testing.T) {
	if mtpRowsDriverEligible(nil) {
		t.Fatal("nil state must decline")
	}
	if mtpRowsDriverEligible(&archDecodeState{}) {
		t.Fatal("an empty state (no layers) must decline")
	}
}
