// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"bytes"
	"runtime"
	"testing"

	"dappco.re/go/inference/model"
	g4 "dappco.re/go/inference/model/gemma4"
)

func newPromptCachePLEFixture(t testing.TB) *ArchSession {
	t.Helper()
	const dModel, nHeads, nKV, headDim, dFF, vocab = 128, 2, 1, 64, 256, 32
	const numLayers, pliDim, gs, bits = 2, 64, 64, 4
	const maxLen = 16
	cfg := g4.Config{
		HiddenSize: dModel, NumHiddenLayers: numLayers, IntermediateSize: dFF,
		NumAttentionHeads: nHeads, NumKeyValueHeads: nKV, HeadDim: headDim, VocabSize: vocab, RMSNormEps: 1e-6,
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
	if !g.HasPLE() {
		t.Fatal("fixture model should have PLE")
	}
	sess, err := NewArchQuantSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("NewArchQuantSession: %v", err)
	}
	return sess
}

func TestWarmPromptCachePLESequentialAllocationBudget(t *testing.T) {
	requireNativeRuntime(t)
	sess := newPromptCachePLEFixture(t)
	prefix := []int32{1, 5, 3, 7}
	if err := sess.WarmPromptCache(prefix); err != nil {
		t.Fatalf("WarmPromptCache warmup: %v", err)
	}

	var warmErr error
	allocs := testing.AllocsPerRun(3, func() {
		sess.pos = 0
		sess.cachedIDs = sess.cachedIDs[:0]
		warmErr = sess.WarmPromptCache(prefix)
	})
	if warmErr != nil {
		t.Fatalf("WarmPromptCache: %v", warmErr)
	}
	if allocs > 5000 {
		t.Fatalf("PLE WarmPromptCache allocations = %.0f, want <= 5000", allocs)
	}
}

func TestPrefillCachedIDsPLEUsesGPUNextInputs(t *testing.T) {
	requireNativeRuntime(t)
	ids := []int32{1, 5, 3, 7}
	serial := newPromptCachePLEFixture(t)
	chained := newPromptCachePLEFixture(t)
	if chained.encNextInputsGPU == nil {
		t.Fatal("PLE fixture did not wire GPU next-inputs seam")
	}

	if err := serial.prefillCachedIDs(ids); err != nil {
		t.Fatalf("serial prefillCachedIDs: %v", err)
	}

	hostEmbeds := 0
	hostPLE := 0
	origEmbed := chained.embed
	origPLE := chained.perLayerInput
	chained.embed = func(id int32) ([]byte, error) {
		hostEmbeds++
		return origEmbed(id)
	}
	chained.perLayerInput = func(id int32, emb []byte) ([]byte, error) {
		hostPLE++
		return origPLE(id, emb)
	}
	if err := chained.prefillCachedIDs(ids); err != nil {
		t.Fatalf("chained prefillCachedIDs: %v", err)
	}
	if hostEmbeds != 0 || hostPLE != 0 {
		t.Fatalf("prefillCachedIDs used host embed/PLE: embeds=%d ple=%d", hostEmbeds, hostPLE)
	}
	if chained.nextInputTokenPtr == nil {
		t.Fatal("prefillCachedIDs did not seed the GPU token buffer")
	}
	if staged := *chained.nextInputTokenPtr; staged != ids[len(ids)-1] {
		t.Fatalf("prefillCachedIDs staged token %d, want final prefix token %d", staged, ids[len(ids)-1])
	}

	chained.embed = origEmbed
	chained.perLayerInput = origPLE
	serialHidden, err := serial.stepID(9)
	if err != nil {
		t.Fatalf("serial continuation stepID: %v", err)
	}
	chainedHidden, err := chained.stepID(9)
	if err != nil {
		t.Fatalf("chained continuation stepID: %v", err)
	}
	if !bytes.Equal(chainedHidden, serialHidden) {
		t.Fatal("GPU-input cached prefix produced different continuation hidden than host-input prefix")
	}
}

func TestStepIDRetainedGPUInputsWritesRetainedHiddenDirectly(t *testing.T) {
	requireNativeRuntime(t)
	oldChainDisabled := chainedGPUInputsDisabled
	defer func() { chainedGPUInputsDisabled = oldChainDisabled }()
	chainedGPUInputsDisabled = false

	control := newPromptCachePLEFixture(t)
	candidate := newPromptCachePLEFixture(t)
	if candidate.encNextInputsGPU == nil || candidate.state.icb == nil {
		t.Fatal("PLE fixture did not wire GPU next-inputs seam")
	}

	wantHidden, ok, err := control.stepIDRetainedGPUInputsInPool(5)
	if err != nil || !ok {
		t.Fatalf("control stepIDRetainedGPUInputsInPool ok=%v err=%v", ok, err)
	}
	if len(control.retainedHidden) == 0 {
		t.Fatal("control did not retain GPU-input hidden")
	}

	poison := bytes.Repeat([]byte{0x6d}, candidate.arch.Hidden*bf16Size)
	candidate.state.icb.lastOutPtr = &poison[0]
	gotHidden, ok, err := candidate.stepIDRetainedGPUInputsInPool(5)
	runtime.KeepAlive(poison)
	if err != nil || !ok {
		t.Fatalf("candidate stepIDRetainedGPUInputsInPool ok=%v err=%v", ok, err)
	}
	if !bytes.Equal(gotHidden, wantHidden) {
		t.Fatal("GPU-input retained step read hidden from lastOutPtr instead of direct retained output")
	}
	if !bytes.Equal(candidate.retainedHidden, wantHidden) {
		t.Fatal("GPU-input retained boundary differs from direct hidden")
	}
	if candidate.retainedHiddenBuffer() == nil {
		t.Fatal("GPU-input retained hidden is not backed by the pinned no-copy buffer")
	}
}

func TestWarmPromptCachePLEUsesGPUNextInputsThroughFinalToken(t *testing.T) {
	requireNativeRuntime(t)
	prefix := []int32{1, 5, 3, 7}
	serial := newPromptCachePLEFixture(t)
	chained := newPromptCachePLEFixture(t)
	if chained.encNextInputsGPU == nil {
		t.Fatal("PLE fixture did not wire GPU next-inputs seam")
	}
	if err := serial.WarmPromptCache(prefix); err != nil {
		t.Fatalf("serial WarmPromptCache: %v", err)
	}

	hostEmbeds := 0
	hostPLE := 0
	origEmbed := chained.embed
	origPLE := chained.perLayerInput
	chained.embed = func(id int32) ([]byte, error) {
		hostEmbeds++
		return origEmbed(id)
	}
	chained.perLayerInput = func(id int32, emb []byte) ([]byte, error) {
		hostPLE++
		return origPLE(id, emb)
	}
	if err := chained.WarmPromptCache(prefix); err != nil {
		t.Fatalf("chained WarmPromptCache: %v", err)
	}
	if hostEmbeds != 0 || hostPLE != 0 {
		t.Fatalf("WarmPromptCache used host embed/PLE: embeds=%d ple=%d", hostEmbeds, hostPLE)
	}
	if len(chained.retainedHidden) == 0 || len(chained.retainedLogits) == 0 {
		t.Fatal("WarmPromptCache did not retain prompt-boundary hidden/logits")
	}
	if !bytes.Equal(chained.retainedHidden, serial.retainedHidden) {
		t.Fatal("GPU-input warm cache retained hidden differs from host-input retained hidden")
	}
	if !bytes.Equal(chained.retainedLogits, serial.retainedLogits) {
		t.Fatal("GPU-input warm cache retained logits differ from host-input retained logits")
	}
	if hit := chained.CachedPrefixLen(prefix); hit != len(prefix) {
		t.Fatalf("GPU-input warm cache prefix hit = %d, want %d", hit, len(prefix))
	}
}

func TestPrefillPromptRetainedPLEUsesGPUNextInputs(t *testing.T) {
	requireNativeRuntime(t)
	ids := []int32{1, 5, 3, 7}
	serial := newPromptCachePLEFixture(t)
	chained := newPromptCachePLEFixture(t)
	if chained.encNextInputsGPU == nil {
		t.Fatal("PLE fixture did not wire GPU next-inputs seam")
	}

	oldChainDisabled := chainedGPUInputsDisabled
	defer func() { chainedGPUInputsDisabled = oldChainDisabled }()
	chainedGPUInputsDisabled = true
	serialHidden, err := serial.prefillPromptRetainedInPool(ids)
	if err != nil {
		t.Fatalf("serial prefillPromptRetainedInPool: %v", err)
	}
	serialHidden = append([]byte(nil), serialHidden...)

	hostEmbeds := 0
	hostPLE := 0
	origEmbed := chained.embed
	origPLE := chained.perLayerInput
	chained.embed = func(id int32) ([]byte, error) {
		hostEmbeds++
		return origEmbed(id)
	}
	chained.perLayerInput = func(id int32, emb []byte) ([]byte, error) {
		hostPLE++
		return origPLE(id, emb)
	}
	chainedGPUInputsDisabled = false
	chainedHidden, err := chained.prefillPromptRetainedInPool(ids)
	if err != nil {
		t.Fatalf("chained prefillPromptRetainedInPool: %v", err)
	}
	if hostEmbeds != 0 || hostPLE != 0 {
		t.Fatalf("prefillPromptRetainedInPool used host embed/PLE: embeds=%d ple=%d", hostEmbeds, hostPLE)
	}
	if chained.Pos() != len(ids) {
		t.Fatalf("prefillPromptRetainedInPool pos = %d, want %d", chained.Pos(), len(ids))
	}
	if len(chained.retainedHidden) == 0 {
		t.Fatal("prefillPromptRetainedInPool did not retain prompt-boundary hidden")
	}
	if !bytes.Equal(chainedHidden, serialHidden) {
		t.Fatal("GPU-input retained prompt hidden differs from host-input retained prompt hidden")
	}
	if !bytes.Equal(chained.retainedHidden, serialHidden) {
		t.Fatal("GPU-input retained prompt boundary differs from host-input retained prompt hidden")
	}
}

func TestPrefillTokensPLEUsesGPUNextInputs(t *testing.T) {
	requireNativeRuntime(t)
	ids := []int32{1, 5, 3, 7}
	serial := newPromptCachePLEFixture(t)
	chained := newPromptCachePLEFixture(t)
	if chained.encNextInputsGPU == nil {
		t.Fatal("PLE fixture did not wire GPU next-inputs seam")
	}

	oldChainDisabled := chainedGPUInputsDisabled
	defer func() { chainedGPUInputsDisabled = oldChainDisabled }()
	chainedGPUInputsDisabled = true
	if err := serial.PrefillTokens(ids); err != nil {
		t.Fatalf("serial PrefillTokens: %v", err)
	}
	serialHidden := append([]byte(nil), serial.retainedHidden...)

	hostEmbeds := 0
	hostPLE := 0
	origEmbed := chained.embed
	origPLE := chained.perLayerInput
	chained.embed = func(id int32) ([]byte, error) {
		hostEmbeds++
		return origEmbed(id)
	}
	chained.perLayerInput = func(id int32, emb []byte) ([]byte, error) {
		hostPLE++
		return origPLE(id, emb)
	}
	chainedGPUInputsDisabled = false
	if err := chained.PrefillTokens(ids); err != nil {
		t.Fatalf("chained PrefillTokens: %v", err)
	}
	if hostEmbeds != 0 || hostPLE != 0 {
		t.Fatalf("PrefillTokens used host embed/PLE: embeds=%d ple=%d", hostEmbeds, hostPLE)
	}
	if chained.Pos() != len(ids) {
		t.Fatalf("PrefillTokens pos = %d, want %d", chained.Pos(), len(ids))
	}
	if !idsEqual(chained.cachedIDs, ids) {
		t.Fatalf("PrefillTokens cached ids = %v, want %v", chained.cachedIDs, ids)
	}
	if len(chained.retainedHidden) == 0 {
		t.Fatal("PrefillTokens did not retain prompt-boundary hidden")
	}
	if !bytes.Equal(chained.retainedHidden, serialHidden) {
		t.Fatal("GPU-input PrefillTokens retained hidden differs from host-input retained hidden")
	}
}
