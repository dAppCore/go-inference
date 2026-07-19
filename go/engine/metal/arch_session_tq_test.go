// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"math"
	"os"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/inference/model"
	g4 "dappco.re/go/inference/model/gemma4"
)

// arch_session_tq_test.go — the session-level TurboQuant live-KV gates: a TQ
// session decodes coherently against its native twin over the SAME weights
// (band, not byte-parity — the cache holds 4-bit codes by design), the
// residency win is real (code cache ≪ bf16 cache), and every v1 decline
// refuses loudly (snapshot, submit-ahead peer, laneSet, no-qualifying-layer).

// tqTestQuantModel assembles the shared synthetic all-global 4-bit gemma4
// (head dim 128 — a TQ-instantiated width) both sessions load.
func tqTestQuantModel(t *testing.T) (*QuantModel, model.Arch) {
	t.Helper()
	const gs, bits = 32, 4
	cfg := g4.Config{
		HiddenSize: 128, NumHiddenLayers: 2, IntermediateSize: 256,
		NumAttentionHeads: 2, NumKeyValueHeads: 1, HeadDim: 128, VocabSize: 32, RMSNormEps: 1e-6,
		Quantization: &model.QuantConfig{GroupSize: gs, Bits: bits},
	}
	arch, err := cfg.Arch()
	if err != nil {
		t.Fatalf("Arch: %v", err)
	}
	ts := quantGemma4Tensors(t, arch, gs, bits)
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

// TestArchQuantSessionTurboQuant_Good gates the live TQ session against its
// native twin: same weights, same prompt, stepwise hidden agreement within the
// codec band (cosine — the cache IS lossy by design), valid greedy tokens, the
// TQ carrier armed, and the residency receipt (code cache bytes ≪ bf16 bytes).
func TestArchQuantSessionTurboQuant_Good(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	g, arch := tqTestQuantModel(t)
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
	if tq.state.icb == nil || !tq.state.icb.hasKVTQ() {
		t.Fatal("turboquant session did not arm the TQ ICB carrier")
	}
	if native.state.icb.hasKVTQ() {
		t.Fatal("native session armed the TQ carrier — the off-path must be untouched")
	}
	// The submit-ahead peer DECLINES under TQ (v1): recordPeerICB stays nil.
	if tq.recordPeerICB != nil {
		t.Fatal("turboquant session armed recordPeerICB — the submit-ahead tail must decline (v1)")
	}
	// Residency receipt: every global owner's code cache is smaller than its
	// bf16 twin (hd=128 b=4: 64 codes + 4 γ bytes vs 256 bf16 bytes per row per head).
	for li := range arch.Layer {
		if !tq.state.icb.kvTQ.on(li) {
			continue
		}
		tqBytes := bufferLengthFast(tq.state.icb.kCaches[li])
		nBytes := bufferLengthFast(native.state.icb.kCaches[li])
		if tqBytes*2 >= nBytes {
			t.Fatalf("layer %d: TQ K cache %dB is not under half the bf16 %dB", li, tqBytes, nBytes)
		}
	}

	// Stepwise agreement over the codec: prefill the same prompt, then step the
	// same token ids through both; the hidden trajectories must stay aligned.
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
	// Measured on this synthetic (2026-07-19): min stepwise cosine 0.999182 at
	// k4v4. The 0.98 floor carries margin while still catching a structural
	// break (wrong Π, wrong γ addressing, stale codes → cosine collapses).
	t.Logf("turboquant:4 vs native stepwise hidden cosine: min %.6f over 8 steps", minCos)
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

// TestArchQuantSessionTurboQuant_Bad proves the session-build refusals: an
// unknown mode string and a stack with no qualifying global head dim.
func TestArchQuantSessionTurboQuant_Bad(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	g, arch := tqTestQuantModel(t)
	if _, err := newArchQuantSessionShardsWithHeadConfig(g, arch, 16, nil, nil, archSessionConfig{kvCacheMode: "turboquant:7"}); err == nil {
		t.Fatal("unknown mode: expected the session build to refuse")
	}

	// head dim 64 — no TQ instantiation → the build must refuse, never run
	// silently native.
	const gs, bits = 32, 4
	cfg := g4.Config{
		HiddenSize: 128, NumHiddenLayers: 2, IntermediateSize: 256,
		NumAttentionHeads: 2, NumKeyValueHeads: 1, HeadDim: 64, VocabSize: 32, RMSNormEps: 1e-6,
		Quantization: &model.QuantConfig{GroupSize: gs, Bits: bits},
	}
	arch64, err := cfg.Arch()
	if err != nil {
		t.Fatalf("Arch: %v", err)
	}
	ts := quantGemma4Tensors(t, arch64, gs, bits)
	lm, err := model.Assemble(ts, arch64, model.StandardWeightNames())
	if err != nil {
		t.Fatalf("model.Assemble: %v", err)
	}
	g64, err := loadedToQuant(lm, gs, bits)
	if err != nil {
		t.Fatalf("loadedToQuant: %v", err)
	}
	if _, err := newArchQuantSessionShardsWithHeadConfig(g64, arch64, 16, nil, nil, archSessionConfig{kvCacheMode: "turboquant:4"}); err == nil {
		t.Fatal("head dim 64: expected the session build to refuse (no qualifying global layer)")
	}
}

// TestArchQuantSessionTurboQuantSnapshot_Bad proves the KV snapshot / -state
// sleep decline: CaptureKV on a TQ session errors loudly (v1 — no bf16 mirror
// exists for the code caches).
func TestArchQuantSessionTurboQuantSnapshot_Bad(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	g, arch := tqTestQuantModel(t)
	tq, err := newArchQuantSessionShardsWithHeadConfig(g, arch, 16, nil, nil, archSessionConfig{kvCacheMode: "turboquant:3.5"})
	if err != nil {
		t.Fatalf("turboquant session: %v", err)
	}
	if err := tq.PrefillTokens([]int32{1, 5, 3}); err != nil {
		t.Fatalf("prefill: %v", err)
	}
	if _, err := tq.CaptureKV(); err == nil {
		t.Fatal("CaptureKV on a turboquant session: expected the v1 decline")
	}
}

// TestOpenLaneSetTurboQuant_Bad proves the batch/interleave decline: a model
// loaded with a TQ cache mode refuses to open the continuous-batching laneSet.
func TestOpenLaneSetTurboQuant_Bad(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	g, arch := tqTestQuantModel(t)
	tm, err := NewQuantTokenModel(g, arch, 16, withKVCacheMode("turboquant"))
	if err != nil {
		t.Fatalf("NewQuantTokenModel: %v", err)
	}
	if _, err := tm.OpenLaneSet(inference.LaneSetConfig{MaxLanes: 2}); err == nil {
		t.Fatal("OpenLaneSet under turboquant: expected the v1 decline")
	}
}

// TestLoadSpeculativePairTurboQuant_Bad proves the MTP pairing decline: the
// pair loader refuses a TQ cache mode BEFORE touching any checkpoint (the
// paths here do not exist — reaching the filesystem would fail differently).
func TestLoadSpeculativePairTurboQuant_Bad(t *testing.T) {
	_, err := LoadSpeculativePair("/nonexistent/target", "/nonexistent/draft", 0, inference.WithCacheMode("turboquant:4"))
	if err == nil {
		t.Fatal("expected the TQ pairing decline")
	}
	if !core.Contains(err.Error(), "turboquant") {
		t.Fatalf("expected the turboquant decline, got: %v", err)
	}
}
