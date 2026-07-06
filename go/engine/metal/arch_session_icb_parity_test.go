// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"bytes"
	"os"
	"testing"
	"unsafe"

	"dappco.re/go/inference/model"
	g4 "dappco.re/go/inference/model/gemma4"
)

// TestArchQuantSessionICBParity proves the incremental ICB encode-bypass (Phase B) is
// byte-identical to the stepToken host-encode path: an eligible E2B-shaped PLE session records
// the arch ICB (state.icb != nil) and replays it per StepWithID; Generate through the ICB must
// equal Generate with the ICB force-disabled (the stepToken path), token-for-token over a
// multi-step prefill+decode. The synthetic model is uniform (no sliding, no MoE, simple rope) so
// it is ICB-eligible — the assertion that state.icb != nil pins that the ICB path is the one
// actually exercised.
func TestArchQuantSessionICBParity(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	const dModel, nHeads, nKV, headDim, dFF, vocab = 128, 2, 1, 64, 256, 32
	const numLayers, pliDim, gs, bits = 2, 64, 64, 4
	const maxLen, n = 16, 6
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
		t.Fatal("assembled model should have the per-layer-input tower")
	}
	prompt := []int32{1, 5, 3, 2}

	// ICB path: the eligible session records + replays the recorded arch ICB.
	sessICB, err := NewArchQuantSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("NewArchQuantSession (ICB): %v", err)
	}
	if sessICB.state.icb == nil {
		t.Fatal("expected the uniform E2B-shaped session to be ICB-eligible (icb recorded) — the parity check is meaningless if the ICB path is not exercised")
	}
	genICB, err := sessICB.Generate(prompt, n, -1)
	if err != nil {
		t.Fatalf("Generate (ICB): %v", err)
	}

	// stepToken path: a fresh identical session with the ICB force-disabled.
	sessHost, err := NewArchQuantSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("NewArchQuantSession (host): %v", err)
	}
	sessHost.state.icb = nil // force the stepToken host re-encode path
	genHost, err := sessHost.Generate(prompt, n, -1)
	if err != nil {
		t.Fatalf("Generate (host): %v", err)
	}

	if len(genICB) != len(genHost) || len(genICB) != n {
		t.Fatalf("token count: ICB %d, host %d, want %d", len(genICB), len(genHost), n)
	}
	for i := range genICB {
		if genICB[i] != genHost[i] {
			t.Fatalf("token %d: ICB %d != host %d — the incremental ICB replay is NOT byte-identical to stepToken", i, genICB[i], genHost[i])
		}
	}
}

func TestArchQuantSessionICBPrefillTokenEmbeddingsMatchesSerial(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	const dModel, nHeads, nKV, headDim, dFF, vocab = 128, 2, 1, 64, 256, 256
	const numLayers, gs, bits = 2, 64, 4
	const maxLen = 16
	cfg := g4.Config{
		HiddenSize: dModel, NumHiddenLayers: numLayers, IntermediateSize: dFF,
		NumAttentionHeads: nHeads, NumKeyValueHeads: nKV, HeadDim: headDim, VocabSize: vocab, RMSNormEps: 1e-6,
		Quantization: &model.QuantConfig{GroupSize: gs, Bits: bits},
	}
	arch, err := cfg.Arch()
	if err != nil {
		t.Fatalf("Arch: %v", err)
	}
	lm, err := model.Assemble(quantGemma4Tensors(t, arch, gs, bits), arch, model.StandardWeightNames())
	if err != nil {
		t.Fatalf("model.Assemble: %v", err)
	}
	g, err := loadedToQuant(lm, gs, bits)
	if err != nil {
		t.Fatalf("loadedToQuant: %v", err)
	}
	ids := []int32{1, 5, 3, 9}
	serial, err := NewArchQuantSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("NewArchQuantSession serial: %v", err)
	}
	serial.state.icb = nil
	icb, err := NewArchQuantSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("NewArchQuantSession ICB: %v", err)
	}
	if icb.state.icb == nil {
		t.Fatal("expected quant session to record an ICB replay")
	}
	embeddings := make([][]byte, len(ids))
	for i, id := range ids {
		emb, err := serial.embedID(id)
		if err != nil {
			t.Fatalf("embedID(%d): %v", id, err)
		}
		embeddings[i] = append([]byte(nil), emb...)
	}
	replacement, err := serial.embedID(17)
	if err != nil {
		t.Fatalf("replacement embedID: %v", err)
	}
	embeddings[1] = append([]byte(nil), replacement...)

	var serialHidden []byte
	for i, id := range ids {
		serialHidden, err = serial.StepWithID(id, embeddings[i])
		if err != nil {
			t.Fatalf("serial StepWithID(%d): %v", id, err)
		}
	}
	if err := icb.PrefillTokenEmbeddings(ids, embeddings); err != nil {
		t.Fatalf("ICB PrefillTokenEmbeddings: %v", err)
	}
	if icb.Pos() != len(ids) {
		t.Fatalf("ICB pos = %d, want %d", icb.Pos(), len(ids))
	}
	if !bytes.Equal(icb.retainedHidden, serialHidden) {
		t.Fatal("ICB explicit-embedding hidden differs from serial StepWithID")
	}
	if icb.retainedHiddenPinned == nil || icb.retainedHiddenPinned.buf == nil {
		t.Fatal("ICB explicit-embedding prefill did not retain a pinned hidden")
	}
	if unsafe.Pointer(&icb.retainedHidden[0]) != unsafe.Pointer(&icb.retainedHiddenPinned.bytes[0]) {
		t.Fatal("ICB explicit-embedding retained hidden does not alias pinned backing")
	}
	if icb.retainedHiddenBufferFor(icb.retainedHidden) == nil {
		t.Fatal("ICB explicit-embedding retained hidden is not exposed as a no-copy buffer")
	}
	nextSerialEmb, err := serial.embedID(4)
	if err != nil {
		t.Fatalf("serial next embedID: %v", err)
	}
	nextICBEmb, err := icb.embedID(4)
	if err != nil {
		t.Fatalf("ICB next embedID: %v", err)
	}
	serialNext, err := serial.StepWithID(4, nextSerialEmb)
	if err != nil {
		t.Fatalf("serial next StepWithID: %v", err)
	}
	icbNext, err := icb.StepWithID(4, nextICBEmb)
	if err != nil {
		t.Fatalf("ICB next StepWithID: %v", err)
	}
	if !bytes.Equal(icbNext, serialNext) {
		t.Fatal("ICB explicit-embedding cache differs from serial on next token")
	}
}

func TestArchQuantSessionICBPLEPrefillTokenEmbeddingsBatchMatchesSerial(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	const dModel, nHeads, nKV, headDim, dFF, vocab = 128, 2, 1, 64, 256, 256
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
		t.Fatal("assembled model should have the per-layer-input tower")
	}
	ids := []int32{1, 5, 3, 9}
	serial, err := NewArchQuantSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("NewArchQuantSession serial: %v", err)
	}
	serial.state.icb = nil
	icb, err := NewArchQuantSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("NewArchQuantSession ICB: %v", err)
	}
	if icb.state.icb == nil {
		t.Fatal("expected quant PLE session to record an ICB replay")
	}
	if !icb.state.icb.hasPLE {
		t.Fatal("expected recorded ICB replay to carry PLE inputs")
	}
	embeddings := make([][]byte, len(ids))
	for i, id := range ids {
		emb, err := serial.embedID(id)
		if err != nil {
			t.Fatalf("embedID(%d): %v", id, err)
		}
		embeddings[i] = append([]byte(nil), emb...)
	}
	replacement, err := serial.embedID(17)
	if err != nil {
		t.Fatalf("replacement embedID: %v", err)
	}
	embeddings[1] = append([]byte(nil), replacement...)

	var serialHidden []byte
	for i, id := range ids {
		serialHidden, err = serial.StepWithID(id, embeddings[i])
		if err != nil {
			t.Fatalf("serial StepWithID(%d): %v", id, err)
		}
	}
	hidden, ok, err := icb.prefillRetainedEmbeddingsICB(ids, embeddings, "native.test.PLEICBPrefill")
	if err != nil {
		t.Fatalf("ICB PLE prefillRetainedEmbeddingsICB: %v", err)
	}
	if !ok {
		t.Fatal("ICB PLE prefillRetainedEmbeddingsICB ok = false")
	}
	if icb.Pos() != len(ids) {
		t.Fatalf("ICB pos = %d, want %d", icb.Pos(), len(ids))
	}
	if !bytes.Equal(hidden, serialHidden) {
		t.Fatal("ICB PLE explicit-embedding hidden differs from serial StepWithID")
	}
	if icb.retainedHiddenPinned == nil || icb.retainedHiddenPinned.buf == nil {
		t.Fatal("ICB PLE explicit-embedding batch prefill did not retain a pinned hidden")
	}
	if len(icb.retainedHiddenPinned.bytes) != len(hidden) {
		t.Fatalf("ICB PLE retained hidden backing len = %d, want %d", len(icb.retainedHiddenPinned.bytes), len(hidden))
	}
	if unsafe.Pointer(&hidden[0]) != unsafe.Pointer(&icb.retainedHiddenPinned.bytes[0]) {
		t.Fatal("ICB PLE explicit-embedding hidden does not alias retained pinned backing")
	}
	if icb.retainedHiddenBufferFor(hidden) == nil {
		t.Fatal("ICB PLE explicit-embedding hidden is not exposed as a no-copy buffer")
	}
	nextSerialEmb, err := serial.embedID(4)
	if err != nil {
		t.Fatalf("serial next embedID: %v", err)
	}
	nextICBEmb, err := icb.embedID(4)
	if err != nil {
		t.Fatalf("ICB next embedID: %v", err)
	}
	serialNext, err := serial.StepWithID(4, nextSerialEmb)
	if err != nil {
		t.Fatalf("serial next StepWithID: %v", err)
	}
	icbNext, err := icb.StepWithID(4, nextICBEmb)
	if err != nil {
		t.Fatalf("ICB next StepWithID: %v", err)
	}
	if !bytes.Equal(icbNext, serialNext) {
		t.Fatal("ICB PLE explicit-embedding cache differs from serial on next token")
	}
}

// TestArchQuantSessionICBParity_KVShared exercises the KV-SHARING path that real gemma4 E2B uses
// heavily (num_kv_shared_layers: 20 of 35) but that NO other quant ICB parity fixture has: a layer
// that shares an earlier layer's KV cache carries NO own k/v projection weights (assemble.go drops
// them for non-owners). The shared recorder still emits a discarded projK/projV per layer for ICB
// op-layout uniformity — bf16 keeps that slot valid with its single shared gemv PSO, but the quant
// path has no per-geometry qmv pipeline for an absent weight, so it must reuse the owner's weight.
// Get it wrong and the ICB replay corrupts the decode while the host stepToken path stays correct —
// exactly the divergence that made real E2B-4bit emit `</story` + ``` garbage. ICB Generate must
// equal the stepToken path token-for-token.
func TestArchQuantSessionICBParity_KVShared(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	const dModel, nHeads, nKV, headDim, dFF, vocab = 128, 2, 1, 64, 256, 32
	const numLayers, pliDim, gs, bits = 3, 64, 64, 4
	const kvShared = 1 // the last layer shares an earlier layer's KV — no own k/v weights
	const maxLen, n = 16, 6
	cfg := g4.Config{
		HiddenSize: dModel, NumHiddenLayers: numLayers, IntermediateSize: dFF,
		NumAttentionHeads: nHeads, NumKeyValueHeads: nKV, HeadDim: headDim, VocabSize: vocab, RMSNormEps: 1e-6,
		HiddenSizePerLayerInput: pliDim, VocabSizePerLayerInput: vocab,
		Quantization:      &model.QuantConfig{GroupSize: gs, Bits: bits},
		NumKVSharedLayers: kvShared,
	}
	arch, err := cfg.Arch()
	if err != nil {
		t.Fatalf("Arch: %v", err)
	}
	sharer := -1 // confirm the fixture actually has a KV-shared (non-owner) layer — the whole point
	for i := range arch.Layer {
		if !arch.Layer[i].OwnsCache() {
			sharer = i
			break
		}
	}
	if sharer < 0 {
		t.Fatal("fixture must have a KV-shared (non-owner) layer to exercise the sharer ICB path")
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
	prompt := []int32{1, 5, 3, 2}

	sessICB, err := NewArchQuantSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("NewArchQuantSession (ICB): %v", err)
	}
	if sessICB.state.icb == nil {
		t.Fatal("expected the KV-shared session to be ICB-eligible (icb recorded)")
	}
	genICB, err := sessICB.Generate(prompt, n, -1)
	if err != nil {
		t.Fatalf("Generate (ICB): %v", err)
	}

	sessHost, err := NewArchQuantSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("NewArchQuantSession (host): %v", err)
	}
	sessHost.state.icb = nil // force the stepToken host re-encode path
	genHost, err := sessHost.Generate(prompt, n, -1)
	if err != nil {
		t.Fatalf("Generate (host): %v", err)
	}

	if len(genICB) != len(genHost) || len(genICB) != n {
		t.Fatalf("token count: ICB %d, host %d, want %d", len(genICB), len(genHost), n)
	}
	for i := range genICB {
		if genICB[i] != genHost[i] {
			t.Fatalf("token %d: ICB %d != host %d — KV-shared (sharer L%d) quant ICB replay NOT byte-identical to stepToken", i, genICB[i], genHost[i], sharer)
		}
	}
}

// TestArchQuantSessionICBParity_PerLayerRope exercises the NEW per-layer rope branches: a model
// with a sliding layer (rope theta 10000) + a global layer (theta 1000000) so localBase != base —
// the exact shape (sliding/global different theta) that gates real gemma4 E2B. The ICB must rope each
// layer on its own base (the recorder's ropeLocalBaseB vs ropeBaseB), matching the host stepToken
// pick token-for-token. If the per-layer rope were wrong, the bases would diverge and the tokens drift.
func TestArchQuantSessionICBParity_PerLayerRope(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	const dModel, nHeads, nKV, headDim, dFF, vocab = 128, 2, 1, 64, 256, 32
	const numLayers, pliDim, gs, bits = 2, 64, 64, 4
	const maxLen, n = 16, 6
	cfg := g4.Config{
		HiddenSize: dModel, NumHiddenLayers: numLayers, IntermediateSize: dFF,
		NumAttentionHeads: nHeads, NumKeyValueHeads: nKV, HeadDim: headDim, VocabSize: vocab, RMSNormEps: 1e-6,
		HiddenSizePerLayerInput: pliDim, VocabSizePerLayerInput: vocab,
		Quantization:  &model.QuantConfig{GroupSize: gs, Bits: bits},
		SlidingWindow: 8,
		LayerTypes:    []string{"sliding_attention", "full_attention"},
		RopeParameters: map[string]g4.RopeParam{
			"sliding_attention": {RopeTheta: 10000},
			"full_attention":    {RopeTheta: 1000000},
		},
	}
	arch, err := cfg.Arch()
	if err != nil {
		t.Fatalf("Arch: %v", err)
	}
	if arch.RopeLocalBase == arch.RopeBase {
		t.Fatalf("fixture must have localBase != base to exercise per-layer rope (both %v)", arch.RopeBase)
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
	prompt := []int32{1, 5, 3, 2}

	sessICB, err := NewArchQuantSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("NewArchQuantSession (ICB): %v", err)
	}
	if sessICB.state.icb == nil {
		t.Fatal("expected the per-layer-rope session to be ICB-eligible (icb recorded)")
	}
	genICB, err := sessICB.Generate(prompt, n, -1)
	if err != nil {
		t.Fatalf("Generate (ICB): %v", err)
	}

	sessHost, err := NewArchQuantSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("NewArchQuantSession (host): %v", err)
	}
	sessHost.state.icb = nil
	genHost, err := sessHost.Generate(prompt, n, -1)
	if err != nil {
		t.Fatalf("Generate (host): %v", err)
	}

	if len(genICB) != len(genHost) || len(genICB) != n {
		t.Fatalf("token count: ICB %d, host %d, want %d", len(genICB), len(genHost), n)
	}
	for i := range genICB {
		if genICB[i] != genHost[i] {
			t.Fatalf("token %d: ICB %d != host %d — per-layer rope (sliding localBase=%v vs base=%v) ICB replay NOT byte-identical to stepToken", i, genICB[i], genHost[i], arch.RopeLocalBase, arch.RopeBase)
		}
	}
}

// TestArchQuantSessionICBParity_PerLayerHeadDim exercises the per-layer HEAD DIM path: a sliding
// layer (head_dim 64) + a global layer (head_dim 128 via global_head_dim) — gemma4's real shape (E2B:
// 256 sliding / 512 global). The ICB sizes the KV cache + attention scratch per layer, picks the SDPA
// PSO + qmv dim buffers per hd, and must decode token-identical to stepToken.
func TestArchQuantSessionICBParity_PerLayerHeadDim(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	const dModel, nHeads, nKV, headDim, globalHeadDim, dFF, vocab = 256, 2, 1, 64, 128, 256, 32
	const numLayers, pliDim, gs, bits = 2, 64, 64, 4
	const maxLen, n = 16, 6
	cfg := g4.Config{
		HiddenSize: dModel, NumHiddenLayers: numLayers, IntermediateSize: dFF,
		NumAttentionHeads: nHeads, NumKeyValueHeads: nKV, HeadDim: headDim, GlobalHeadDim: globalHeadDim,
		VocabSize: vocab, RMSNormEps: 1e-6,
		HiddenSizePerLayerInput: pliDim, VocabSizePerLayerInput: vocab,
		Quantization:  &model.QuantConfig{GroupSize: gs, Bits: bits},
		SlidingWindow: 8,
		LayerTypes:    []string{"sliding_attention", "full_attention"},
		RopeParameters: map[string]g4.RopeParam{
			"sliding_attention": {RopeTheta: 10000},
			"full_attention":    {RopeTheta: 1000000},
		},
	}
	arch, err := cfg.Arch()
	if err != nil {
		t.Fatalf("Arch: %v", err)
	}
	if arch.GlobalHeadDim == arch.HeadDim {
		t.Fatalf("fixture must have globalHeadDim != headDim to exercise per-layer head dim (both %d)", arch.HeadDim)
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
	prompt := []int32{1, 5, 3, 2}

	sessICB, err := NewArchQuantSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("NewArchQuantSession (ICB): %v", err)
	}
	if sessICB.state.icb == nil {
		t.Fatal("expected the per-layer-head-dim session to be ICB-eligible (icb recorded)")
	}
	genICB, err := sessICB.Generate(prompt, n, -1)
	if err != nil {
		t.Fatalf("Generate (ICB): %v", err)
	}

	sessHost, err := NewArchQuantSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("NewArchQuantSession (host): %v", err)
	}
	sessHost.state.icb = nil
	genHost, err := sessHost.Generate(prompt, n, -1)
	if err != nil {
		t.Fatalf("Generate (host): %v", err)
	}

	if len(genICB) != len(genHost) || len(genICB) != n {
		t.Fatalf("token count: ICB %d, host %d, want %d", len(genICB), len(genHost), n)
	}
	for i := range genICB {
		if genICB[i] != genHost[i] {
			t.Fatalf("token %d: ICB %d != host %d — per-layer head dim (sliding %d / global %d) ICB replay NOT byte-identical to stepToken", i, genICB[i], genHost[i], arch.HeadDim, arch.GlobalHeadDim)
		}
	}
}

// TestArchQuantSessionICBParity_PerLayerHiddenCosine is the CI-runnable (no real model) guard for the
// global-layer value-norm bug. Token-identical generation does NOT catch a small per-layer numerical
// error that fails to flip an argmax on a tiny vocab — the bug shipped green for exactly that reason.
// This asserts the STRONGER property: the quant ICB replay's per-layer hidden is cosine 1.0 to the host
// re-encode at pos 0. The fixture has a global layer (head_dim 128 > sliding 64); sizing valueNormOnes
// at the base head dim makes the global value-norm read off the end of the ones vector, which surfaces
// here as cos < 1 even while the generated tokens still match. (Real-model counterpart, gated on
// E2B_Q4_DIR: q4_icb_localize_test.go.)
func TestArchQuantSessionICBParity_PerLayerHiddenCosine(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	const dModel, nHeads, nKV, headDim, globalHeadDim, dFF, vocab = 256, 2, 1, 64, 128, 256, 32
	const numLayers, pliDim, gs, bits = 2, 64, 64, 4
	const maxLen = 16
	cfg := g4.Config{
		HiddenSize: dModel, NumHiddenLayers: numLayers, IntermediateSize: dFF,
		NumAttentionHeads: nHeads, NumKeyValueHeads: nKV, HeadDim: headDim, GlobalHeadDim: globalHeadDim,
		VocabSize: vocab, RMSNormEps: 1e-6,
		HiddenSizePerLayerInput: pliDim, VocabSizePerLayerInput: vocab,
		Quantization:  &model.QuantConfig{GroupSize: gs, Bits: bits},
		SlidingWindow: 8,
		LayerTypes:    []string{"sliding_attention", "full_attention"},
		RopeParameters: map[string]g4.RopeParam{
			"sliding_attention": {RopeTheta: 10000},
			"full_attention":    {RopeTheta: 1000000},
		},
	}
	arch, err := cfg.Arch()
	if err != nil {
		t.Fatalf("Arch: %v", err)
	}
	if arch.GlobalHeadDim == arch.HeadDim {
		t.Fatalf("fixture must have globalHeadDim != headDim to exercise the wider value-norm read")
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

	s, err := NewArchQuantSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("NewArchQuantSession: %v", err)
	}
	if s.state.icb == nil {
		t.Fatal("expected an ICB-eligible session (icb recorded)")
	}
	const id = int32(5)
	emb, err := s.embed(id)
	if err != nil {
		t.Fatalf("embed: %v", err)
	}
	var pli []byte
	if s.perLayerInput != nil {
		if pli, err = s.perLayerInput(id, emb); err != nil {
			t.Fatalf("perLayerInput: %v", err)
		}
		s.state.perLayerInput = pli
	}

	capturedLayerHiddens = nil
	captureLayerHiddens = true
	_, serr := s.state.stepToken(emb, 0)
	captureLayerHiddens = false
	if serr != nil {
		t.Fatalf("stepToken: %v", serr)
	}
	reLayers := capturedLayerHiddens
	_, icbLayers := s.state.icb.stepBodyCapture(emb, 0, pli)

	if len(reLayers) != numLayers || len(icbLayers) != numLayers {
		t.Fatalf("per-layer capture count: reencode=%d icb=%d want %d", len(reLayers), len(icbLayers), numLayers)
	}
	for L := range numLayers {
		c := cosineBF16(reLayers[L], icbLayers[L])
		if c < 0.9999 {
			at := "sliding"
			if s.state.specs[L].Attention == model.GlobalAttention {
				at = "GLOBAL"
			}
			t.Fatalf("L%d (%s hd=%d): ICB-vs-host per-layer cosine=%.5f < 0.9999 — the quant ICB replay diverges from the host re-encode (valueNormOnes sized at base head dim, not maxHeadDim?)", L, at, headDimOf(s.state.specs[L], headDim), c)
		}
	}
}

// TestArchQuantSessionICBParity_PerLayerKVHeads is the FAST synthetic reproduction of the 12B/31B
// non-uniform-kvHeads ICB divergence (TestRealModelICBvsReencodeParity needs an 18GB model to see it).
// The session normally gates this geometry to the re-encode path (icbEligible rejects non-uniform
// kvHeads); icbForceEligibleForTest opens that gate so the ICB IS recorded and replayed, then the
// generated tokens must equal the stepToken host path. A divergence here is the cache-stride bug that
// keeps 12B/31B off the fast ICB path — pinned in milliseconds. The fixture mirrors the real mix: a
// sliding GQA layer (kv=2, headDim=64) + a global MQA layer (kv=1, headDim=128).
func TestArchQuantSessionICBParity_PerLayerKVHeads(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	// sliding kvDim = 4·64 = 256, global kvDim = 1·128 = 128 — DIFFERENT per-layer kv strides (the real
	// 12B/31B has sliding kvDim ≫ global kvDim); equal kvDims would hide a cache-stride mismatch. The
	// 5:1-ish sliding:global pattern + a wrapping window (maxLen 16, window 8, 10 tokens) stress the ring.
	const dModel, nHeads, nKV, globalKV, headDim, globalHeadDim, dFF, vocab = 256, 8, 4, 1, 64, 128, 256, 32
	const numLayers, pliDim, gs, bits = 4, 64, 64, 4
	const maxLen, n = 16, 6
	cfg := g4.Config{
		HiddenSize: dModel, NumHiddenLayers: numLayers, IntermediateSize: dFF,
		NumAttentionHeads: nHeads, NumKeyValueHeads: nKV, NumGlobalKeyValueHeads: globalKV,
		HeadDim: headDim, GlobalHeadDim: globalHeadDim,
		VocabSize: vocab, RMSNormEps: 1e-6,
		HiddenSizePerLayerInput: pliDim, VocabSizePerLayerInput: vocab,
		Quantization:  &model.QuantConfig{GroupSize: gs, Bits: bits},
		SlidingWindow: 8,
		LayerTypes:    []string{"sliding_attention", "sliding_attention", "sliding_attention", "full_attention"},
		RopeParameters: map[string]g4.RopeParam{
			"sliding_attention": {RopeTheta: 10000},
			"full_attention":    {RopeTheta: 1000000},
		},
	}
	arch, err := cfg.Arch()
	if err != nil {
		t.Fatalf("Arch: %v", err)
	}
	if arch.GlobalKVHeads == arch.KVHeads {
		t.Fatalf("fixture must have globalKVHeads(%d) != kvHeads(%d) to exercise the non-uniform mix", arch.GlobalKVHeads, arch.KVHeads)
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
	prompt := []int32{1, 5, 3, 2}

	sessICB, err := NewArchQuantSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("NewArchQuantSession (ICB): %v", err)
	}
	if sessICB.state.icb == nil {
		t.Fatal("expected the non-uniform-kv session to record the ICB (icbEligible now accepts the MQA-global mix)")
	}
	genICB, err := sessICB.Generate(prompt, n, -1)
	if err != nil {
		t.Fatalf("Generate (ICB): %v", err)
	}

	sessHost, err := NewArchQuantSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("NewArchQuantSession (host): %v", err)
	}
	sessHost.state.icb = nil
	genHost, err := sessHost.Generate(prompt, n, -1)
	if err != nil {
		t.Fatalf("Generate (host): %v", err)
	}

	for i := range genICB {
		if genICB[i] != genHost[i] {
			t.Fatalf("token %d: ICB %d != host %d — non-uniform kvHeads (sliding kv=%d / global kv=%d) ICB replay NOT byte-identical to stepToken", i, genICB[i], genHost[i], arch.KVHeads, arch.GlobalKVHeads)
		}
	}

	// STRONGER gate: per-layer hidden cosine at pos 0. Token-equality on a tiny vocab can miss a small
	// numerical divergence that would flip a real 256k-vocab argmax (the PerLayerHiddenCosine lesson) —
	// a non-uniform-kv cache-stride error would surface HERE as a per-layer cos < 1 even while tokens match.
	sc, err := NewArchQuantSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("NewArchQuantSession (cosine): %v", err)
	}
	if sc.state.icb == nil {
		t.Fatal("expected the cosine session to record the ICB")
	}
	const id = int32(5)
	emb, err := sc.embed(id)
	if err != nil {
		t.Fatalf("embed: %v", err)
	}
	var pli []byte
	if sc.perLayerInput != nil {
		if pli, err = sc.perLayerInput(id, emb); err != nil {
			t.Fatalf("perLayerInput: %v", err)
		}
		sc.state.perLayerInput = pli
	}
	capturedLayerHiddens = nil
	captureLayerHiddens = true
	_, serr := sc.state.stepToken(emb, 0)
	captureLayerHiddens = false
	if serr != nil {
		t.Fatalf("stepToken: %v", serr)
	}
	reLayers := capturedLayerHiddens
	_, icbLayers := sc.state.icb.stepBodyCapture(emb, 0, pli)
	if len(reLayers) != numLayers || len(icbLayers) != numLayers {
		t.Fatalf("per-layer capture count: reencode=%d icb=%d want %d", len(reLayers), len(icbLayers), numLayers)
	}
	for L := range numLayers {
		if c := cosineBF16(reLayers[L], icbLayers[L]); c < 0.9999 {
			at := "sliding"
			if sc.state.specs[L].Attention == model.GlobalAttention {
				at = "GLOBAL"
			}
			t.Fatalf("L%d (%s kv=%d hd=%d): ICB-vs-host per-layer cosine=%.5f < 0.9999 — non-uniform-kv ICB replay diverges from the host re-encode",
				L, at, kvHeadsOf(sc.state.specs[L], arch.KVHeads), headDimOf(sc.state.specs[L], headDim), c)
		}
	}
	t.Logf("non-uniform kvHeads session: ICB replay ≡ stepToken across %d tokens AND per-layer hidden cosine ≥ 0.9999 (sliding kv=%d / global kv=%d) — the recorder is byte-correct; 12B/31B can take the fast path", n, arch.KVHeads, arch.GlobalKVHeads)
}
