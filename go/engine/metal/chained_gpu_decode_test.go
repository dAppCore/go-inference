// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"bytes"
	"os"
	"runtime"
	"testing"
	"time"
	"unsafe"

	"dappco.re/go/inference/model"
	g4 "dappco.re/go/inference/model/gemma4"
	"dappco.re/go/inference/model/safetensors"
)

// pleQuantModel assembles a small e2b-shaped PLE quant model (4-bit main+PLE embedding, bf16 PLE
// projection — the shape the GPU next-inputs seam handles).
func pleQuantModel(t testing.TB, numLayers, dFF, vocab, kvShared int) (*QuantModel, model.Arch) {
	const dModel, nHeads, nKV, headDim = 128, 2, 1, 64
	const pliDim, gs, bits = 64, 64, 4
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
	ts := quantGemma4Tensors(t, arch, gs, bits)
	addPLETensors(t, ts, arch, gs, bits)
	lm, err := model.Assemble(ts, arch, model.StandardWeightNames())
	if err != nil {
		t.Fatalf("Assemble: %v", err)
	}
	g, err := loadedToQuant(lm, gs, bits)
	if err != nil {
		t.Fatalf("loadedToQuant: %v", err)
	}
	if !g.HasPLE() {
		t.Fatal("fixture should have the per-layer-input tower")
	}
	return g, arch
}

// TestChainedGPUDecodeMatchesHost gates the chained-GPU decode: with the GPU next-inputs seam ON (each
// step produces the next emb+pli on-GPU, one command buffer/token) the token sequence must equal the host
// embed/PLE chained path. A bug in the on-GPU emb/pli, the no-input stepBody, or the cache/pos bookkeeping
// diverges the tokens. Also pins that the GPU path is actually wired (not silently falling back).
func TestChainedGPUDecodeMatchesHost(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	g, arch := pleQuantModel(t, 2, 256, 32, 0)
	const maxLen, N = 16, 8
	prompt := []int32{1, 5, 3, 2}

	sessGPU, err := NewArchQuantSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("session GPU: %v", err)
	}
	if sessGPU.encNextInputsGPU == nil {
		t.Fatal("expected the GPU next-inputs seam wired (e2b-shaped PLE session)")
	}
	chainedGPUInputsDisabled = false
	gpuGen, err := sessGPU.Generate(prompt, N, -1)
	if err != nil {
		t.Fatalf("Generate (GPU): %v", err)
	}

	chainedGPUInputsDisabled = true
	defer func() { chainedGPUInputsDisabled = false }()
	sessHost, err := NewArchQuantSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("session host: %v", err)
	}
	hostGen, err := sessHost.Generate(prompt, N, -1)
	if err != nil {
		t.Fatalf("Generate (host): %v", err)
	}

	if len(gpuGen) != len(hostGen) || len(gpuGen) != N {
		t.Fatalf("token count: GPU %d, host %d, want %d", len(gpuGen), len(hostGen), N)
	}
	for i := range gpuGen {
		if gpuGen[i] != hostGen[i] {
			t.Fatalf("token %d: chained-GPU %d != host %d (GPU=%v host=%v)", i, gpuGen[i], hostGen[i], gpuGen, hostGen)
		}
	}
	t.Logf("chained-GPU decode matches host embed/PLE path: %v", gpuGen)
}

func TestChainedGPUGenerateOneShotUsesGPUTailWithoutCachingFinalToken(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	g, arch := pleQuantModel(t, 2, 256, 32, 0)
	const maxLen, N = 16, 6
	prompt := []int32{1, 5, 3, 2}
	oldPipe := pipelinedGPUDecodeEnabled
	oldChainDisabled := chainedGPUInputsDisabled
	oldTiming := pieceTimingOn
	oldSpan := chainedGPUSpanNs
	defer func() {
		pipelinedGPUDecodeEnabled = oldPipe
		chainedGPUInputsDisabled = oldChainDisabled
		pieceTimingOn = oldTiming
		chainedGPUSpanNs = oldSpan
	}()
	pipelinedGPUDecodeEnabled = false
	pieceTimingOn = false

	chainedGPUInputsDisabled = true
	host, err := NewArchQuantSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("host session: %v", err)
	}
	hostGen, err := host.GenerateOneShot(prompt, N, -1)
	if err != nil {
		t.Fatalf("GenerateOneShot host: %v", err)
	}

	chainedGPUInputsDisabled = false
	gpu, err := NewArchQuantSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("GPU session: %v", err)
	}
	if gpu.encNextInputsGPU == nil || gpu.state.icb == nil {
		t.Fatal("fixture did not wire chained GPU ICB path")
	}
	pieceTimingOn = true
	chainedGPUSpanNs = 0
	gpuGen, err := gpu.GenerateOneShot(prompt, N, -1)
	pieceTimingOn = false
	if err != nil {
		t.Fatalf("GenerateOneShot GPU: %v", err)
	}
	if !idsEqual(gpuGen, hostGen) {
		t.Fatalf("one-shot chained GPU tokens = %v, want host %v", gpuGen, hostGen)
	}
	if chainedGPUSpanNs <= 0 {
		t.Fatal("one-shot decode did not enter the chained GPU tail")
	}
	if gpu.Pos() != len(prompt)+len(gpuGen)-1 {
		t.Fatalf("one-shot pos = %d, want prompt plus cached intermediate tokens (%d)", gpu.Pos(), len(prompt)+len(gpuGen)-1)
	}
}

func TestPipelinedGPUGenerateOneShotUsesPeerICBWithoutCachingFinalToken(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	g, arch := pleQuantModel(t, 3, 256, 32, 0)
	const maxLen, N = 24, 8
	prompt := []int32{1, 5, 3, 2}
	oldPipe := pipelinedGPUDecodeEnabled
	oldChainDisabled := chainedGPUInputsDisabled
	defer func() {
		pipelinedGPUDecodeEnabled = oldPipe
		chainedGPUInputsDisabled = oldChainDisabled
	}()
	chainedGPUInputsDisabled = false

	pipelinedGPUDecodeEnabled = false
	chained, err := NewArchQuantSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("chained session: %v", err)
	}
	chainedGen, err := chained.GenerateOneShot(prompt, N, -1)
	if err != nil {
		t.Fatalf("chained GenerateOneShot: %v", err)
	}

	pipelinedGPUDecodeEnabled = true
	pipe, err := NewArchQuantSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("pipelined session: %v", err)
	}
	if pipe.recordPeerICB == nil {
		t.Skip("peer ICB recorder unavailable")
	}
	pipeGen, err := pipe.GenerateOneShot(prompt, N, -1)
	if err != nil {
		t.Fatalf("pipelined GenerateOneShot: %v", err)
	}
	if !idsEqual(pipeGen, chainedGen) {
		t.Fatalf("one-shot pipelined tokens = %v, want chained %v", pipeGen, chainedGen)
	}
	if pipe.icbPeer == nil {
		t.Fatal("one-shot pipelined decode did not record/use the peer ICB")
	}
	if pipe.gpuTailPLScratch[0] == nil || pipe.gpuTailPLScratch[1] == nil {
		t.Fatal("one-shot pipelined decode did not use both session PLE scratch slots")
	}
	if pipe.gpuTailPLScratch[0] == pipe.gpuTailPLScratch[1] {
		t.Fatal("one-shot pipelined decode scratch slots alias")
	}
	if pipe.Pos() != len(prompt)+len(pipeGen)-1 {
		t.Fatalf("one-shot pipelined pos = %d, want prompt plus cached intermediate tokens (%d)", pipe.Pos(), len(prompt)+len(pipeGen)-1)
	}
}

func TestChainedGPUDecodeFinalHiddenWritesRetainedHiddenDirectly(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	g, arch := pleQuantModel(t, 2, 256, 32, 0)
	const maxLen, maxNew = 16, 4
	prompt := []int32{1, 5, 3, 2}
	oldPipe := pipelinedGPUDecodeEnabled
	oldChainDisabled := chainedGPUInputsDisabled
	defer func() {
		pipelinedGPUDecodeEnabled = oldPipe
		chainedGPUInputsDisabled = oldChainDisabled
	}()
	pipelinedGPUDecodeEnabled = false
	chainedGPUInputsDisabled = false

	control, err := NewArchQuantSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("control session: %v", err)
	}
	candidate, err := NewArchQuantSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("candidate session: %v", err)
	}
	if candidate.encNextInputsGPU == nil || candidate.state.icb == nil {
		t.Fatal("fixture did not wire chained GPU ICB path")
	}
	prepare := func(t *testing.T, sess *ArchSession) []int32 {
		t.Helper()
		var first int32
		withAutoreleasePool(func() {
			hidden, err := sess.prefillPromptRetainedInPool(prompt)
			if err != nil {
				t.Fatalf("prefillPromptRetainedInPool: %v", err)
			}
			first, err = sess.headGreedyOrLogits(hidden, nil, nil, nil, true)
			if err != nil {
				t.Fatalf("headGreedyOrLogits: %v", err)
			}
		})
		return []int32{first}
	}
	controlSeed := prepare(t, control)
	candidateSeed := prepare(t, candidate)
	if !idsEqual(candidateSeed, controlSeed) {
		t.Fatalf("candidate first token = %v, want %v", candidateSeed, controlSeed)
	}

	wantGen, err := control.generateChainedGPUTail(controlSeed, maxNew, -1, nil, nil, false)
	if err != nil {
		t.Fatalf("control generateChainedGPUTail: %v", err)
	}
	if len(control.retainedHidden) == 0 {
		t.Fatal("control did not retain final hidden")
	}

	poison := bytes.Repeat([]byte{0x3a}, candidate.arch.Hidden*bf16Size)
	candidate.state.icb.lastOutPtr = &poison[0]
	gotGen, err := candidate.generateChainedGPUTail(candidateSeed, maxNew, -1, nil, nil, false)
	runtime.KeepAlive(poison)
	if err != nil {
		t.Fatalf("candidate generateChainedGPUTail: %v", err)
	}
	if !idsEqual(gotGen, wantGen) {
		t.Fatalf("candidate tokens = %v, want %v", gotGen, wantGen)
	}
	if !bytes.Equal(candidate.retainedHidden, control.retainedHidden) {
		t.Fatal("chained GPU final hidden read from lastOutPtr instead of direct retained output")
	}
	if candidate.retainedHiddenBuffer() == nil || unsafe.Pointer(&candidate.retainedHidden[0]) != unsafe.Pointer(&candidate.retainedHiddenPinned.bytes[0]) {
		t.Fatal("chained GPU final hidden is not retained in session no-copy backing")
	}
}

func TestPipelinedGPUDecodeFinalHiddenWritesRetainedHiddenDirectly(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	g, arch := pleQuantModel(t, 3, 256, 32, 0)
	const maxLen, maxNew = 24, 6
	prompt := []int32{1, 5, 3, 2}
	oldPipe := pipelinedGPUDecodeEnabled
	oldChainDisabled := chainedGPUInputsDisabled
	defer func() {
		pipelinedGPUDecodeEnabled = oldPipe
		chainedGPUInputsDisabled = oldChainDisabled
	}()
	pipelinedGPUDecodeEnabled = true
	chainedGPUInputsDisabled = false

	control, err := NewArchQuantSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("control session: %v", err)
	}
	candidate, err := NewArchQuantSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("candidate session: %v", err)
	}
	if candidate.encNextInputsGPU == nil || candidate.state.icb == nil || candidate.recordPeerICB == nil {
		t.Fatal("fixture did not wire pipelined GPU ICB path")
	}
	prepare := func(t *testing.T, sess *ArchSession) []int32 {
		t.Helper()
		var first int32
		withAutoreleasePool(func() {
			hidden, err := sess.prefillPromptRetainedInPool(prompt)
			if err != nil {
				t.Fatalf("prefillPromptRetainedInPool: %v", err)
			}
			first, err = sess.headGreedyOrLogits(hidden, nil, nil, nil, true)
			if err != nil {
				t.Fatalf("headGreedyOrLogits: %v", err)
			}
		})
		return []int32{first}
	}
	controlSeed := prepare(t, control)
	candidateSeed := prepare(t, candidate)
	if !idsEqual(candidateSeed, controlSeed) {
		t.Fatalf("candidate first token = %v, want %v", candidateSeed, controlSeed)
	}

	wantGen, err := control.generatePipelinedGPUTail(controlSeed, maxNew, -1, nil, nil, false)
	if err != nil {
		t.Fatalf("control generatePipelinedGPUTail: %v", err)
	}
	if len(control.retainedHidden) == 0 {
		t.Fatal("control did not retain pipelined final hidden")
	}

	peer, err := candidate.peerICB()
	if err != nil {
		t.Fatalf("candidate peerICB: %v", err)
	}
	poisonA := bytes.Repeat([]byte{0x4b}, candidate.arch.Hidden*bf16Size)
	poisonB := bytes.Repeat([]byte{0x4c}, candidate.arch.Hidden*bf16Size)
	candidate.state.icb.lastOutPtr = &poisonA[0]
	peer.lastOutPtr = &poisonB[0]
	gotGen, err := candidate.generatePipelinedGPUTail(candidateSeed, maxNew, -1, nil, nil, false)
	runtime.KeepAlive(poisonA)
	runtime.KeepAlive(poisonB)
	if err != nil {
		t.Fatalf("candidate generatePipelinedGPUTail: %v", err)
	}
	if !idsEqual(gotGen, wantGen) {
		t.Fatalf("candidate pipelined tokens = %v, want %v", gotGen, wantGen)
	}
	if !bytes.Equal(candidate.retainedHidden, control.retainedHidden) {
		t.Fatal("pipelined GPU final hidden read from lastOutPtr instead of direct retained output")
	}
	if candidate.retainedHiddenBuffer() == nil || unsafe.Pointer(&candidate.retainedHidden[0]) != unsafe.Pointer(&candidate.retainedHiddenPinned.bytes[0]) {
		t.Fatal("pipelined GPU final hidden is not retained in session no-copy backing")
	}
}

// TestChainedGPUDecodeHeadroom measures the per-token GPU-execution span vs wall across a chained-GPU
// decode — the host/sync gap a submit-ahead pipeline could overlap. Reported at 16 AND 32 layers: the
// fixed per-token sync is a smaller fraction at depth, so this is the evidence for whether the 2-ICB
// submit-ahead (piece b) is worth its build cost. Diagnostic (logs), not a pass/fail gate.
func TestChainedGPUDecodeHeadroom(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	prompt := []int32{1, 5, 3, 7, 2, 9}
	const maxLen, N = 128, 48
	for _, numLayers := range []int{16, 32} {
		g, arch := pleQuantModel(t, numLayers, 6144, 8192, 0)
		sess, err := NewArchQuantSession(g, arch, maxLen)
		if err != nil {
			t.Fatalf("%dL session: %v", numLayers, err)
		}
		if sess.encNextInputsGPU == nil {
			t.Fatalf("%dL: chained-GPU path not wired", numLayers)
		}
		if err := sess.PrefillTokens(prompt); err != nil {
			t.Fatalf("%dL prefill: %v", numLayers, err)
		}
		// warmup (untimed) then a measured run.
		if _, err := sess.GenerateFromCache(4, -1); err != nil {
			t.Fatalf("%dL warmup: %v", numLayers, err)
		}
		pieceTimingOn = true
		chainedGPUSpanNs = 0
		t0 := time.Now()
		if _, err := sess.GenerateFromCache(N, -1); err != nil {
			pieceTimingOn = false
			t.Fatalf("%dL generate: %v", numLayers, err)
		}
		wall := time.Since(t0)
		pieceTimingOn = false
		_ = sess.Close()
		gpu := time.Duration(chainedGPUSpanNs)
		headroom := float64(wall-gpu) / float64(wall) * 100
		t.Logf("%2dL: wall %.2fms  gpu-span %.2fms  per-tok wall %.3fms gpu %.3fms  host/sync headroom %.1f%% (submit-ahead ceiling)",
			numLayers, float64(wall.Microseconds())/1000, float64(gpu.Microseconds())/1000,
			float64(wall.Microseconds())/1000/float64(N), float64(gpu.Microseconds())/1000/float64(N), headroom)
	}
}

// TestPipelinedGPUDecodeMatchesChained gates the submit-ahead pipeline: with two ICBs in flight over
// shared KV (host submits t+1 before reading t, 1-ahead discard-safe), the tokens must equal the proven
// synchronous chained-GPU path — including an eos-break case that exercises the discard of the trailing
// speculative cb. A bug in the ping-pong inputs, the shared-KV hazard, or the discard/pos bookkeeping
// diverges the tokens or the cache state.
func TestPipelinedGPUDecodeMatchesChained(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	g, arch := pleQuantModel(t, 3, 256, 32, 0)
	const maxLen = 24
	prompt := []int32{1, 5, 3, 2}

	for _, tc := range []struct {
		name  string
		n     int
		eosID int
	}{
		{"full", 12, -1},
		{"short", 3, -1},
		{"single", 1, -1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			pipelinedGPUDecodeEnabled = false
			sessC, err := NewArchQuantSession(g, arch, maxLen)
			if err != nil {
				t.Fatalf("chained session: %v", err)
			}
			chainGen, err := sessC.Generate(prompt, tc.n, tc.eosID)
			if err != nil {
				t.Fatalf("chained generate: %v", err)
			}

			pipelinedGPUDecodeEnabled = true
			defer func() { pipelinedGPUDecodeEnabled = true }()
			sessP, err := NewArchQuantSession(g, arch, maxLen)
			if err != nil {
				t.Fatalf("pipelined session: %v", err)
			}
			pipeGen, err := sessP.Generate(prompt, tc.n, tc.eosID)
			if err != nil {
				t.Fatalf("pipelined generate: %v", err)
			}
			if len(pipeGen) != len(chainGen) {
				t.Fatalf("token count: pipelined %d vs chained %d", len(pipeGen), len(chainGen))
			}
			for i := range chainGen {
				if pipeGen[i] != chainGen[i] {
					t.Fatalf("token %d: pipelined %d != chained %d (pipe=%v chain=%v)", i, pipeGen[i], chainGen[i], pipeGen, chainGen)
				}
			}
			t.Logf("pipelined matches chained: %v", pipeGen)
		})
	}
}

func TestChainedGPUDecodeGenerateFromCacheAllocationBudget(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	g, arch := pleQuantModel(t, 3, 256, 32, 0)
	const maxLen = 40
	oldPipe := pipelinedGPUDecodeEnabled
	oldChainDisabled := chainedGPUInputsDisabled
	defer func() {
		pipelinedGPUDecodeEnabled = oldPipe
		chainedGPUInputsDisabled = oldChainDisabled
	}()
	pipelinedGPUDecodeEnabled = false
	chainedGPUInputsDisabled = false

	sess, err := NewArchQuantSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("NewArchQuantSession: %v", err)
	}
	if sess.encNextInputsGPU == nil || sess.state.icb == nil {
		t.Skip("fixture did not wire chained GPU ICB path")
	}
	if err := sess.PrefillTokens([]int32{1, 5, 3, 2}); err != nil {
		t.Fatalf("PrefillTokens: %v", err)
	}
	if _, err := sess.GenerateFromCache(2, -1); err != nil {
		t.Fatalf("GenerateFromCache warmup: %v", err)
	}

	allocs := testing.AllocsPerRun(5, func() {
		if _, err := sess.GenerateFromCache(2, -1); err != nil {
			t.Fatalf("GenerateFromCache: %v", err)
		}
	})
	if allocs > 800 {
		t.Fatalf("chained GPU GenerateFromCache allocations = %.0f, want <= 800", allocs)
	}
}

func TestGPUTailPLScratchReusesSessionSlots(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	g, arch := pleQuantModel(t, 3, 256, 32, 0)
	const maxLen = 24
	prompt := []int32{1, 5, 3, 2}
	oldPipe := pipelinedGPUDecodeEnabled
	oldChainDisabled := chainedGPUInputsDisabled
	defer func() {
		pipelinedGPUDecodeEnabled = oldPipe
		chainedGPUInputsDisabled = oldChainDisabled
	}()
	chainedGPUInputsDisabled = false

	pipelinedGPUDecodeEnabled = false
	sessC, err := NewArchQuantSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("chained session: %v", err)
	}
	if _, err := sessC.Generate(prompt, 5, -1); err != nil {
		t.Fatalf("chained first turn: %v", err)
	}
	if sessC.gpuTailPLScratch[0] == nil {
		t.Fatal("chained GPU tail did not use session PLE scratch slot 0")
	}
	chainScratch := sessC.gpuTailPLScratch[0]
	if sessC.gpuTailPLScratch[1] != nil {
		t.Fatal("chained GPU tail unexpectedly used pipelined PLE scratch slot 1")
	}
	if _, err := sessC.GenerateFromCache(3, -1); err != nil {
		t.Fatalf("chained second turn: %v", err)
	}
	if sessC.gpuTailPLScratch[0] != chainScratch {
		t.Fatal("chained GPU tail did not reuse session PLE scratch slot 0")
	}

	pipelinedGPUDecodeEnabled = true
	sessP, err := NewArchQuantSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("pipelined session: %v", err)
	}
	if sessP.recordPeerICB == nil {
		t.Skip("peer ICB recorder unavailable")
	}
	if _, err := sessP.Generate(prompt, 5, -1); err != nil {
		t.Fatalf("pipelined first turn: %v", err)
	}
	if sessP.gpuTailPLScratch[0] == nil || sessP.gpuTailPLScratch[1] == nil {
		t.Fatal("pipelined GPU tail did not use both session PLE scratch slots")
	}
	pipeScratch0, pipeScratch1 := sessP.gpuTailPLScratch[0], sessP.gpuTailPLScratch[1]
	if pipeScratch0 == pipeScratch1 {
		t.Fatal("pipelined GPU tail scratch slots alias")
	}
	if _, err := sessP.GenerateFromCache(3, -1); err != nil {
		t.Fatalf("pipelined second turn: %v", err)
	}
	if sessP.gpuTailPLScratch[0] != pipeScratch0 || sessP.gpuTailPLScratch[1] != pipeScratch1 {
		t.Fatal("pipelined GPU tail did not reuse both session PLE scratch slots")
	}
}

func TestPipelinedGPUDecodePrewarmsPeerICB(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	g, arch := pleQuantModel(t, 3, 256, 32, 0)
	const maxLen = 24
	oldPipe := pipelinedGPUDecodeEnabled
	defer func() { pipelinedGPUDecodeEnabled = oldPipe }()

	pipelinedGPUDecodeEnabled = false
	serial, err := NewArchQuantSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("serial session: %v", err)
	}
	if serial.icbPeer != nil {
		t.Fatal("non-pipelined session prewarmed a peer ICB")
	}

	pipelinedGPUDecodeEnabled = true
	piped, err := NewArchQuantSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("pipelined session: %v", err)
	}
	if piped.recordPeerICB == nil {
		t.Skip("peer ICB recorder unavailable")
	}
	if piped.icbPeer == nil {
		t.Fatal("pipelined session did not prewarm the peer ICB")
	}
}

// TestPipelinedGPUDecodeSecondTurn pins the cache/pos byte-identity across REUSE: two back-to-back
// GenerateFromCache turns on a session must produce the same tokens pipelined as chained-GPU. The second
// turn only matches if the first turn left the KV cache, pos, and retained hidden exactly as the serial
// loop would — the subtle risk of the speculative double-buffer.
func TestPipelinedGPUDecodeSecondTurn(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	g, arch := pleQuantModel(t, 3, 256, 32, 0)
	const maxLen = 24
	prompt := []int32{1, 5, 3, 2}

	twoTurns := func(pipelined bool) []int32 {
		pipelinedGPUDecodeEnabled = pipelined
		sess, err := NewArchQuantSession(g, arch, maxLen)
		if err != nil {
			t.Fatalf("session: %v", err)
		}
		t1, err := sess.Generate(prompt, 5, -1)
		if err != nil {
			t.Fatalf("turn 1: %v", err)
		}
		t2, err := sess.GenerateFromCache(5, -1)
		if err != nil {
			t.Fatalf("turn 2: %v", err)
		}
		return append(t1, t2...)
	}
	chain := twoTurns(false)
	pipe := twoTurns(true)
	pipelinedGPUDecodeEnabled = true

	if len(pipe) != len(chain) {
		t.Fatalf("count: pipelined %d vs chained %d", len(pipe), len(chain))
	}
	for i := range chain {
		if pipe[i] != chain[i] {
			t.Fatalf("turn-spanning token %d: pipelined %d != chained %d (pipe=%v chain=%v)", i, pipe[i], chain[i], pipe, chain)
		}
	}
	t.Logf("pipelined two-turn matches chained: %v", pipe)
}

// TestPipelinedGPUDecodeKVShared soaks the submit-ahead pipeline on the KV-SHARING shape real e2b uses
// heavily (a layer carrying no own k/v weights, sharing an earlier layer's cache). Two ICBs over a SHARED
// cache that is ALSO shared across layers is the riskiest hazard case — get the cross-cb ordering wrong
// and the decode corrupts (the divergence that once made real E2B-4bit emit garbage). Pipelined must equal
// chained-GPU token-for-token.
func TestPipelinedGPUDecodeKVShared(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	g, arch := pleQuantModel(t, 3, 256, 32, 1) // last layer shares an earlier layer's KV
	shared := false
	for i := range arch.Layer {
		if !arch.Layer[i].OwnsCache() {
			shared = true
			break
		}
	}
	if !shared {
		t.Fatal("fixture must have a KV-shared layer")
	}
	const maxLen, N = 24, 10
	prompt := []int32{1, 5, 3, 2}

	pipelinedGPUDecodeEnabled = false
	sessC, err := NewArchQuantSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("chained session: %v", err)
	}
	chainGen, err := sessC.Generate(prompt, N, -1)
	if err != nil {
		t.Fatalf("chained generate: %v", err)
	}

	pipelinedGPUDecodeEnabled = true
	defer func() { pipelinedGPUDecodeEnabled = true }()
	sessP, err := NewArchQuantSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("pipelined session: %v", err)
	}
	pipeGen, err := sessP.Generate(prompt, N, -1)
	if err != nil {
		t.Fatalf("pipelined generate: %v", err)
	}
	if len(pipeGen) != len(chainGen) {
		t.Fatalf("count: pipelined %d vs chained %d", len(pipeGen), len(chainGen))
	}
	for i := range chainGen {
		if pipeGen[i] != chainGen[i] {
			t.Fatalf("KV-shared token %d: pipelined %d != chained %d (pipe=%v chain=%v)", i, pipeGen[i], chainGen[i], pipeGen, chainGen)
		}
	}
	t.Logf("pipelined KV-shared matches chained: %v", pipeGen)
}

func TestSampledChainedGPUDecodeStagesTailFromDeviceToken(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	g, arch := pleQuantModel(t, 3, 256, 32, 0)
	const maxLen, maxNew = 24, 5
	prompt := []int32{1, 5, 3, 2}
	params := model.SampleParams{Temperature: 0.9, TopK: 4, TopP: 0.8}

	oldChainDisabled := chainedGPUInputsDisabled
	defer func() { chainedGPUInputsDisabled = oldChainDisabled }()

	chainedGPUInputsDisabled = true
	host, err := NewArchQuantSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("host session: %v", err)
	}
	hostGen, err := host.GenerateSampledEach(prompt, maxNew, nil, model.NewSampler(27), params, nil, nil)
	if err != nil {
		t.Fatalf("host GenerateSampledEach: %v", err)
	}

	chainedGPUInputsDisabled = false

	sess, err := NewArchQuantSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("session: %v", err)
	}
	if sess.encNextInputsGPU == nil {
		t.Fatal("expected sampled chained-GPU path to have the GPU next-inputs seam wired")
	}
	if !sess.sampleTopKTokenParamsEligible(params) {
		t.Skip("device TopK sampled-token path unavailable")
	}
	gen, err := sess.GenerateSampledEach(prompt, maxNew, nil, model.NewSampler(27), params, nil, nil)
	if err != nil {
		t.Fatalf("GenerateSampledEach: %v", err)
	}
	if len(gen) != maxNew {
		t.Fatalf("GenerateSampledEach returned %d tokens, want %d: %v", len(gen), maxNew, gen)
	}
	if !idsEqual(gen, hostGen) {
		t.Fatalf("sampled chained-GPU tokens = %v, want host path %v", gen, hostGen)
	}
	if gen[0] == gen[len(gen)-1] {
		t.Skipf("sampled fixture produced matching first/final tokens %d; cannot distinguish host restaging", gen[0])
	}
	if sess.nextInputTokenPtr == nil {
		t.Fatal("sampled chained-GPU path never seeded the host token buffer")
	}
	if got, want := *sess.nextInputTokenPtr, gen[0]; got != want {
		t.Fatalf("sampled chained-GPU tail restaged host token %d, want host seed to remain first sampled token %d (gen=%v)", got, want, gen)
	}
}

func TestSampledChainedGPUDecodeWritesRetainedHiddenDirectly(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	g, arch := pleQuantModel(t, 3, 256, 32, 0)
	const maxLen, maxNew = 24, 5
	prompt := []int32{1, 5, 3, 2}
	params := model.SampleParams{Temperature: 0.9, TopK: 4, TopP: 0.8}
	oldPipe := pipelinedGPUDecodeEnabled
	oldChainDisabled := chainedGPUInputsDisabled
	defer func() {
		pipelinedGPUDecodeEnabled = oldPipe
		chainedGPUInputsDisabled = oldChainDisabled
	}()
	pipelinedGPUDecodeEnabled = false
	chainedGPUInputsDisabled = false

	control, err := NewArchQuantSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("control session: %v", err)
	}
	candidate, err := NewArchQuantSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("candidate session: %v", err)
	}
	if candidate.encNextInputsGPU == nil || candidate.state.icb == nil {
		t.Fatal("fixture did not wire sampled chained GPU ICB path")
	}
	if !candidate.sampleTopKTokenParamsEligible(params) {
		t.Skip("device TopK sampled-token path unavailable")
	}
	prepare := func(t *testing.T, sess *ArchSession) ([]int32, *model.Sampler) {
		t.Helper()
		sampler := model.NewSampler(27)
		var first int32
		withAutoreleasePool(func() {
			hidden, err := sess.prefillPromptRetainedInPool(prompt)
			if err != nil {
				t.Fatalf("prefillPromptRetainedInPool: %v", err)
			}
			var ok bool
			first, ok, err = sess.sampleTopKTokenFromHiddenInPool(hidden, params, sampler.Draw(), nil)
			if err != nil || !ok {
				t.Fatalf("sampleTopKTokenFromHiddenInPool ok=%v err=%v", ok, err)
			}
		})
		return []int32{first}, sampler
	}
	controlSeed, controlSampler := prepare(t, control)
	candidateSeed, candidateSampler := prepare(t, candidate)
	if !idsEqual(candidateSeed, controlSeed) {
		t.Fatalf("candidate first token = %v, want %v", candidateSeed, controlSeed)
	}

	wantGen, _, err := control.generateSampledChainedGPUTail(controlSeed, maxNew, nil, controlSampler, params, nil, true, 0, nil)
	if err != nil {
		t.Fatalf("control generateSampledChainedGPUTail: %v", err)
	}
	if len(control.retainedHidden) == 0 {
		t.Fatal("control did not retain sampled final hidden")
	}

	poison := bytes.Repeat([]byte{0x2d}, candidate.arch.Hidden*bf16Size)
	candidate.state.icb.lastOutPtr = &poison[0]
	gotGen, _, err := candidate.generateSampledChainedGPUTail(candidateSeed, maxNew, nil, candidateSampler, params, nil, true, 0, nil)
	runtime.KeepAlive(poison)
	if err != nil {
		t.Fatalf("candidate generateSampledChainedGPUTail: %v", err)
	}
	if !idsEqual(gotGen, wantGen) {
		t.Fatalf("candidate sampled tokens = %v, want %v", gotGen, wantGen)
	}
	if !bytes.Equal(candidate.retainedHidden, control.retainedHidden) {
		t.Fatal("sampled chained GPU hidden read from lastOutPtr instead of direct retained output")
	}
	if candidate.retainedHiddenBuffer() == nil || unsafe.Pointer(&candidate.retainedHidden[0]) != unsafe.Pointer(&candidate.retainedHiddenPinned.bytes[0]) {
		t.Fatal("sampled chained GPU final hidden is not retained in session no-copy backing")
	}
}

func TestSampledPipelinedGPUDecodeFinalHiddenWritesRetainedHiddenDirectly(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	g, arch := pleQuantModel(t, 3, 256, 32, 0)
	const maxLen, maxNew = 24, 6
	prompt := []int32{1, 5, 3, 2}
	params := model.SampleParams{Temperature: 0.9, TopK: 4, TopP: 0.8}
	oldPipe := pipelinedGPUDecodeEnabled
	oldChainDisabled := chainedGPUInputsDisabled
	defer func() {
		pipelinedGPUDecodeEnabled = oldPipe
		chainedGPUInputsDisabled = oldChainDisabled
	}()
	pipelinedGPUDecodeEnabled = true
	chainedGPUInputsDisabled = false

	control, err := NewArchQuantSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("control session: %v", err)
	}
	candidate, err := NewArchQuantSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("candidate session: %v", err)
	}
	if candidate.encNextInputsGPU == nil || candidate.state.icb == nil || candidate.recordPeerICB == nil {
		t.Fatal("fixture did not wire sampled pipelined GPU ICB path")
	}
	if !candidate.sampleTopKTokenParamsEligible(params) {
		t.Skip("device TopK sampled-token path unavailable")
	}
	prepare := func(t *testing.T, sess *ArchSession) ([]int32, *model.Sampler) {
		t.Helper()
		sampler := model.NewSampler(61)
		var first int32
		withAutoreleasePool(func() {
			hidden, err := sess.prefillPromptRetainedInPool(prompt)
			if err != nil {
				t.Fatalf("prefillPromptRetainedInPool: %v", err)
			}
			var ok bool
			first, ok, err = sess.sampleTopKTokenFromHiddenInPool(hidden, params, sampler.Draw(), nil)
			if err != nil || !ok {
				t.Fatalf("sampleTopKTokenFromHiddenInPool ok=%v err=%v", ok, err)
			}
		})
		return []int32{first}, sampler
	}
	controlSeed, controlSampler := prepare(t, control)
	candidateSeed, candidateSampler := prepare(t, candidate)
	if !idsEqual(candidateSeed, controlSeed) {
		t.Fatalf("candidate first token = %v, want %v", candidateSeed, controlSeed)
	}

	wantGen, _, err := control.generateSampledPipelinedGPUTail(controlSeed, maxNew, nil, controlSampler, params, nil, 0, nil)
	if err != nil {
		t.Fatalf("control generateSampledPipelinedGPUTail: %v", err)
	}
	if len(control.retainedHidden) == 0 {
		t.Fatal("control did not retain sampled pipelined final hidden")
	}

	peer, err := candidate.peerICB()
	if err != nil {
		t.Fatalf("candidate peerICB: %v", err)
	}
	poisonA := bytes.Repeat([]byte{0x5b}, candidate.arch.Hidden*bf16Size)
	poisonB := bytes.Repeat([]byte{0x5c}, candidate.arch.Hidden*bf16Size)
	candidate.state.icb.lastOutPtr = &poisonA[0]
	peer.lastOutPtr = &poisonB[0]
	gotGen, _, err := candidate.generateSampledPipelinedGPUTail(candidateSeed, maxNew, nil, candidateSampler, params, nil, 0, nil)
	runtime.KeepAlive(poisonA)
	runtime.KeepAlive(poisonB)
	if err != nil {
		t.Fatalf("candidate generateSampledPipelinedGPUTail: %v", err)
	}
	if !idsEqual(gotGen, wantGen) {
		t.Fatalf("candidate sampled pipelined tokens = %v, want %v", gotGen, wantGen)
	}
	if !bytes.Equal(candidate.retainedHidden, control.retainedHidden) {
		t.Fatal("sampled pipelined GPU final hidden read from lastOutPtr instead of direct retained output")
	}
	if candidate.retainedHiddenBuffer() == nil || unsafe.Pointer(&candidate.retainedHidden[0]) != unsafe.Pointer(&candidate.retainedHiddenPinned.bytes[0]) {
		t.Fatal("sampled pipelined GPU final hidden is not retained in session no-copy backing")
	}
}

func TestPipelinedSampledGPUDecodeMatchesChained(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	g, arch := pleQuantModel(t, 3, 256, 32, 0)
	const maxLen, maxNew = 24, 8
	prompt := []int32{1, 5, 3, 2}
	params := model.SampleParams{Temperature: 0.9, TopK: 4, TopP: 0.8}

	oldPipe := pipelinedGPUDecodeEnabled
	oldChainDisabled := chainedGPUInputsDisabled
	defer func() {
		pipelinedGPUDecodeEnabled = oldPipe
		chainedGPUInputsDisabled = oldChainDisabled
	}()
	chainedGPUInputsDisabled = false

	pipelinedGPUDecodeEnabled = false
	chain, err := NewArchQuantSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("chained session: %v", err)
	}
	chainGen, err := chain.GenerateSampledEach(prompt, maxNew, nil, model.NewSampler(91), params, nil, nil)
	if err != nil {
		t.Fatalf("chained GenerateSampledEach: %v", err)
	}

	pipelinedGPUDecodeEnabled = true
	pipe, err := NewArchQuantSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("pipelined session: %v", err)
	}
	if pipe.recordPeerICB == nil {
		t.Skip("peer ICB recorder unavailable")
	}
	pipeGen, err := pipe.GenerateSampledEach(prompt, maxNew, nil, model.NewSampler(91), params, nil, nil)
	if err != nil {
		t.Fatalf("pipelined GenerateSampledEach: %v", err)
	}
	if !idsEqual(pipeGen, chainGen) {
		t.Fatalf("sampled pipelined tokens = %v, want chained %v", pipeGen, chainGen)
	}
	if pipe.gpuTailPLScratch[0] == nil || pipe.gpuTailPLScratch[1] == nil {
		t.Fatal("sampled pipelined GPU tail did not use both session PLE scratch slots")
	}
	if pipe.gpuTailPLScratch[0] == pipe.gpuTailPLScratch[1] {
		t.Fatal("sampled pipelined GPU tail scratch slots alias")
	}
}

func TestPipelinedSampledGPUOneShotUsesPeerICBWithoutCachingFinalToken(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	g, arch := pleQuantModel(t, 3, 256, 32, 0)
	const maxLen, maxNew = 24, 8
	prompt := []int32{1, 5, 3, 2}
	params := model.SampleParams{Temperature: 0.9, TopK: 4, TopP: 0.8}

	oldPipe := pipelinedGPUDecodeEnabled
	oldChainDisabled := chainedGPUInputsDisabled
	defer func() {
		pipelinedGPUDecodeEnabled = oldPipe
		chainedGPUInputsDisabled = oldChainDisabled
	}()
	chainedGPUInputsDisabled = false

	pipelinedGPUDecodeEnabled = false
	chained, err := NewArchQuantSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("chained session: %v", err)
	}
	chainedGen, err := chained.GenerateSampledOneShotEach(prompt, maxNew, nil, model.NewSampler(91), params, nil, nil)
	if err != nil {
		t.Fatalf("chained GenerateSampledOneShotEach: %v", err)
	}

	pipelinedGPUDecodeEnabled = true
	pipe, err := NewArchQuantSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("pipelined session: %v", err)
	}
	if pipe.recordPeerICB == nil {
		t.Skip("peer ICB recorder unavailable")
	}
	pipeGen, err := pipe.GenerateSampledOneShotEach(prompt, maxNew, nil, model.NewSampler(91), params, nil, nil)
	if err != nil {
		t.Fatalf("pipelined GenerateSampledOneShotEach: %v", err)
	}
	if !idsEqual(pipeGen, chainedGen) {
		t.Fatalf("sampled one-shot pipelined tokens = %v, want chained %v", pipeGen, chainedGen)
	}
	if pipe.icbPeer == nil {
		t.Fatal("sampled one-shot pipelined decode did not record/use the peer ICB")
	}
	if pipe.gpuTailPLScratch[0] == nil || pipe.gpuTailPLScratch[1] == nil {
		t.Fatal("sampled one-shot pipelined GPU tail did not use both session PLE scratch slots")
	}
	if pipe.gpuTailPLScratch[0] == pipe.gpuTailPLScratch[1] {
		t.Fatal("sampled one-shot pipelined GPU tail scratch slots alias")
	}
	if pipe.Pos() != len(prompt)+len(pipeGen)-1 {
		t.Fatalf("sampled one-shot pipelined pos = %d, want prompt plus cached intermediate tokens (%d)", pipe.Pos(), len(prompt)+len(pipeGen)-1)
	}
}

func TestSampledCacheLogitsGPUDecodeStagesFirstTokenFromDeviceTail(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	g, arch := pleQuantModel(t, 3, 256, 32, 0)
	const maxLen, maxNew = 24, 5
	prompt := []int32{1, 5, 3, 2}
	params := model.SampleParams{Temperature: 0.9, TopK: 4, TopP: 0.8}

	oldPipe := pipelinedGPUDecodeEnabled
	oldChainDisabled := chainedGPUInputsDisabled
	defer func() {
		pipelinedGPUDecodeEnabled = oldPipe
		chainedGPUInputsDisabled = oldChainDisabled
	}()
	chainedGPUInputsDisabled = false
	pipelinedGPUDecodeEnabled = true

	sess, err := NewArchQuantSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("session: %v", err)
	}
	if err := sess.PrefillTokens(prompt); err != nil {
		t.Fatalf("PrefillTokens: %v", err)
	}
	logits, err := sess.BoundaryLogits()
	if err != nil {
		t.Fatalf("BoundaryLogits: %v", err)
	}
	got, err := sess.GenerateSampledFromCacheLogitsEach(logits, maxNew, nil, model.NewSampler(41), params, nil, nil)
	if err != nil {
		t.Fatalf("GenerateSampledFromCacheLogitsEach: %v", err)
	}
	if len(got) != maxNew {
		t.Fatalf("GenerateSampledFromCacheLogitsEach returned %d tokens, want %d: %v", len(got), maxNew, got)
	}
	cold, err := NewArchQuantSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("cold session: %v", err)
	}
	want, err := cold.GenerateSampledEach(prompt, maxNew, nil, model.NewSampler(41), params, nil, nil)
	if err != nil {
		t.Fatalf("cold GenerateSampledEach: %v", err)
	}
	if !idsEqual(got, want) {
		t.Fatalf("sampled cache-logits tokens = %v, want cold %v", got, want)
	}
	if got[0] == got[1] {
		t.Skipf("sampled fixture produced matching first/tail tokens %d; cannot distinguish tail staging", got[0])
	}
	if sess.nextInputTokenPtr == nil {
		t.Fatal("sampled cache-logits GPU tail never seeded the token buffer")
	}
	if staged := *sess.nextInputTokenPtr; staged != got[0] {
		t.Fatalf("sampled cache-logits tail staged token %d, want first sampled token %d (gen=%v)", staged, got[0], got)
	}
}

func TestCacheLogitsGPUDecodeStagesFirstTokenFromDeviceTail(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	g, arch := pleQuantModel(t, 3, 256, 32, 0)
	const maxLen, maxNew = 24, 5
	prompt := []int32{1, 5, 3, 2}

	oldPipe := pipelinedGPUDecodeEnabled
	oldChainDisabled := chainedGPUInputsDisabled
	defer func() {
		pipelinedGPUDecodeEnabled = oldPipe
		chainedGPUInputsDisabled = oldChainDisabled
	}()
	chainedGPUInputsDisabled = false
	pipelinedGPUDecodeEnabled = true

	sess, err := NewArchQuantSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("session: %v", err)
	}
	if err := sess.PrefillTokens(prompt); err != nil {
		t.Fatalf("PrefillTokens: %v", err)
	}
	logits, err := sess.BoundaryLogits()
	if err != nil {
		t.Fatalf("BoundaryLogits: %v", err)
	}
	got, err := sess.GenerateFromCacheLogitsEach(logits, maxNew, -1, nil)
	if err != nil {
		t.Fatalf("GenerateFromCacheLogitsEach: %v", err)
	}
	if len(got) != maxNew {
		t.Fatalf("GenerateFromCacheLogitsEach returned %d tokens, want %d: %v", len(got), maxNew, got)
	}
	cold, err := NewArchQuantSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("cold session: %v", err)
	}
	want, err := cold.Generate(prompt, maxNew, -1)
	if err != nil {
		t.Fatalf("cold Generate: %v", err)
	}
	if !idsEqual(got, want) {
		t.Fatalf("cache-logits tokens = %v, want cold %v", got, want)
	}
	if sess.nextInputTokenPtr == nil {
		t.Fatal("cache-logits GPU tail never seeded the token buffer")
	}
	if staged := *sess.nextInputTokenPtr; staged != got[0] {
		t.Fatalf("cache-logits tail staged token %d, want first boundary token %d (gen=%v)", staged, got[0], got)
	}
}

func benchChainedDecodePLE(b *testing.B, gpuInputs, pipelined bool) {
	if os.Getenv(MetallibPathEnv) == "" {
		b.Skip("metallib not set")
	}
	g, arch := pleQuantModel(b, 16, 6144, 8192, 0)
	const maxLen, N = 96, 32
	prompt := []int32{1, 5, 3, 7, 2, 9}
	chainedGPUInputsDisabled = !gpuInputs
	pipelinedGPUDecodeEnabled = pipelined
	defer func() { chainedGPUInputsDisabled = false; pipelinedGPUDecodeEnabled = true }()
	b.SetBytes(int64(N))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		b.StopTimer()
		sess, err := NewArchQuantSession(g, arch, maxLen)
		if err != nil {
			b.Fatal(err)
		}
		if err := sess.PrefillTokens(prompt); err != nil {
			b.Fatal(err)
		}
		b.StartTimer()
		if _, err := sess.GenerateFromCache(N, -1); err != nil {
			b.Fatal(err)
		}
		b.StopTimer()
		_ = sess.Close()
		b.StartTimer()
	}
}

// 16-layer e2b-shaped PLE decode: host embed/PLE chained (2 buffers/token), chained-GPU (1), and the
// submit-ahead pipeline (1 + overlap).
func BenchmarkChainedDecodePLEHost(b *testing.B) { benchChainedDecodePLE(b, false, false) }
func BenchmarkChainedDecodePLEGpu(b *testing.B)  { benchChainedDecodePLE(b, true, false) }
func BenchmarkChainedDecodePLEPipe(b *testing.B) { benchChainedDecodePLE(b, true, true) }

func benchCacheLogitsPLE(b *testing.B, gpuInputs, pipelined bool) {
	if os.Getenv(MetallibPathEnv) == "" {
		b.Skip("metallib not set")
	}
	g, arch := pleQuantModel(b, 16, 6144, 8192, 0)
	const maxLen, N = 96, 32
	prompt := []int32{1, 5, 3, 7, 2, 9}
	chainedGPUInputsDisabled = !gpuInputs
	pipelinedGPUDecodeEnabled = pipelined
	defer func() { chainedGPUInputsDisabled = false; pipelinedGPUDecodeEnabled = true }()
	b.SetBytes(int64(N))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		b.StopTimer()
		sess, err := NewArchQuantSession(g, arch, maxLen)
		if err != nil {
			b.Fatal(err)
		}
		if err := sess.PrefillTokens(prompt); err != nil {
			b.Fatal(err)
		}
		logits, err := sess.BoundaryLogits()
		if err != nil {
			b.Fatal(err)
		}
		b.StartTimer()
		if _, err := sess.GenerateFromCacheLogitsEach(logits, N, -1, nil); err != nil {
			b.Fatal(err)
		}
		b.StopTimer()
		_ = sess.Close()
		b.StartTimer()
	}
}

func BenchmarkCacheLogitsPLEHost(b *testing.B) { benchCacheLogitsPLE(b, false, false) }
func BenchmarkCacheLogitsPLEGpu(b *testing.B)  { benchCacheLogitsPLE(b, true, false) }
func BenchmarkCacheLogitsPLEPipe(b *testing.B) { benchCacheLogitsPLE(b, true, true) }

func benchSampledChainedDecodePLE(b *testing.B, gpuInputs, pipelined bool) {
	if os.Getenv(MetallibPathEnv) == "" {
		b.Skip("metallib not set")
	}
	g, arch := pleQuantModel(b, 16, 6144, 8192, 0)
	const maxLen, N = 96, 32
	prompt := []int32{1, 5, 3, 7, 2, 9}
	params := model.SampleParams{Temperature: 0.9, TopK: 8, TopP: 0.85}
	chainedGPUInputsDisabled = !gpuInputs
	pipelinedGPUDecodeEnabled = pipelined
	defer func() { chainedGPUInputsDisabled = false; pipelinedGPUDecodeEnabled = true }()
	b.SetBytes(int64(N))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		b.StopTimer()
		sess, err := NewArchQuantSession(g, arch, maxLen)
		if err != nil {
			b.Fatal(err)
		}
		b.StartTimer()
		if _, err := sess.GenerateSampledEach(prompt, N, nil, model.NewSampler(27), params, nil, nil); err != nil {
			b.Fatal(err)
		}
		b.StopTimer()
		_ = sess.Close()
		b.StartTimer()
	}
}

func BenchmarkSampledChainedDecodePLEHost(b *testing.B) {
	benchSampledChainedDecodePLE(b, false, false)
}
func BenchmarkSampledChainedDecodePLEGpu(b *testing.B)  { benchSampledChainedDecodePLE(b, true, false) }
func BenchmarkSampledChainedDecodePLEPipe(b *testing.B) { benchSampledChainedDecodePLE(b, true, true) }

func benchSampledCacheLogitsPLE(b *testing.B, gpuInputs, pipelined bool) {
	if os.Getenv(MetallibPathEnv) == "" {
		b.Skip("metallib not set")
	}
	g, arch := pleQuantModel(b, 16, 6144, 8192, 0)
	const maxLen, N = 96, 32
	prompt := []int32{1, 5, 3, 7, 2, 9}
	params := model.SampleParams{Temperature: 0.9, TopK: 8, TopP: 0.85}
	chainedGPUInputsDisabled = !gpuInputs
	pipelinedGPUDecodeEnabled = pipelined
	defer func() { chainedGPUInputsDisabled = false; pipelinedGPUDecodeEnabled = true }()
	b.SetBytes(int64(N))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		b.StopTimer()
		sess, err := NewArchQuantSession(g, arch, maxLen)
		if err != nil {
			b.Fatal(err)
		}
		if err := sess.PrefillTokens(prompt); err != nil {
			b.Fatal(err)
		}
		logits, err := sess.BoundaryLogits()
		if err != nil {
			b.Fatal(err)
		}
		b.StartTimer()
		if _, err := sess.GenerateSampledFromCacheLogitsEach(logits, N, nil, model.NewSampler(27), params, nil, nil); err != nil {
			b.Fatal(err)
		}
		b.StopTimer()
		_ = sess.Close()
		b.StartTimer()
	}
}

func BenchmarkSampledCacheLogitsPLEHost(b *testing.B) { benchSampledCacheLogitsPLE(b, false, false) }
func BenchmarkSampledCacheLogitsPLEGpu(b *testing.B)  { benchSampledCacheLogitsPLE(b, true, false) }
func BenchmarkSampledCacheLogitsPLEPipe(b *testing.B) { benchSampledCacheLogitsPLE(b, true, true) }

// denseQuantModel is pleQuantModel WITHOUT the per-layer-input tower — the
// 12B/31B shape. Exercises the non-PLE GPU next-inputs seam (embed gather
// alone feeds the chain).
func denseQuantModel(t testing.TB, numLayers, dFF, vocab, kvShared int) (*QuantModel, model.Arch) {
	const dModel, nHeads, nKV, headDim = 128, 2, 1, 64
	const gs, bits = 64, 4
	cfg := g4.Config{
		HiddenSize: dModel, NumHiddenLayers: numLayers, IntermediateSize: dFF,
		NumAttentionHeads: nHeads, NumKeyValueHeads: nKV, HeadDim: headDim, VocabSize: vocab, RMSNormEps: 1e-6,
		Quantization:      &model.QuantConfig{GroupSize: gs, Bits: bits},
		NumKVSharedLayers: kvShared,
	}
	arch, err := cfg.Arch()
	if err != nil {
		t.Fatalf("Arch: %v", err)
	}
	ts := quantGemma4Tensors(t, arch, gs, bits)
	lm, err := model.Assemble(ts, arch, model.StandardWeightNames())
	if err != nil {
		t.Fatalf("Assemble: %v", err)
	}
	g, err := loadedToQuant(lm, gs, bits)
	if err != nil {
		t.Fatalf("loadedToQuant: %v", err)
	}
	if g.HasPLE() {
		t.Fatal("fixture must NOT have the per-layer-input tower")
	}
	return g, arch
}

// TestDenseGPUDecodeSeamEngages pins that a non-PLE quant session gets the GPU
// next-inputs seam (embed gather only) — the gate the 12B/31B chained/pipelined
// decode rides. Without the seam the chain silently falls back to the host
// loop and a parity test would compare host against host.
func TestDenseGPUDecodeSeamEngages(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	g, arch := denseQuantModel(t, 3, 256, 32, 0)
	sess, err := NewArchQuantSession(g, arch, 24)
	if err != nil {
		t.Fatalf("session: %v", err)
	}
	if sess.state.icb == nil {
		t.Fatal("dense fixture: ICB not recorded")
	}
	if sess.encNextInputsGPU == nil {
		t.Fatal("dense fixture: GPU next-inputs seam NOT wired (chained/pipelined inactive)")
	}
	if sess.plScratchNew == nil {
		t.Fatal("dense fixture: plScratchNew placeholder NOT set (chain gates fail)")
	}
}

// TestDensePipelinedGPUDecodeMatchesHost is the non-PLE sibling of
// TestPipelinedGPUDecodeMatchesChained: host loop, chained-GPU and pipelined
// lanes must produce identical tokens on the dense (12B/31B-shaped) fixture.
func TestDensePipelinedGPUDecodeMatchesHost(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	g, arch := denseQuantModel(t, 3, 256, 32, 0)
	const maxLen = 24
	prompt := []int32{1, 5, 3, 2}
	const n = 12

	run := func(host, pipe bool) []int32 {
		oldChain := chainedGPUInputsDisabled
		oldPipe := pipelinedGPUDecodeEnabled
		defer func() {
			chainedGPUInputsDisabled = oldChain
			pipelinedGPUDecodeEnabled = oldPipe
		}()
		chainedGPUInputsDisabled = host
		pipelinedGPUDecodeEnabled = pipe
		sess, err := NewArchQuantSession(g, arch, maxLen)
		if err != nil {
			t.Fatalf("session (host=%v pipe=%v): %v", host, pipe, err)
		}
		gen, err := sess.Generate(prompt, n, -1)
		if err != nil {
			t.Fatalf("generate (host=%v pipe=%v): %v", host, pipe, err)
		}
		return gen
	}

	hostGen := run(true, false)
	chainGen := run(false, false)
	pipeGen := run(false, true)
	if len(hostGen) != n {
		t.Fatalf("host generated %d tokens, want %d", len(hostGen), n)
	}
	for i := range hostGen {
		if chainGen[i] != hostGen[i] {
			t.Fatalf("token %d: chained %d != host %d (chain=%v host=%v)", i, chainGen[i], hostGen[i], chainGen, hostGen)
		}
		if pipeGen[i] != hostGen[i] {
			t.Fatalf("token %d: pipelined %d != host %d (pipe=%v host=%v)", i, pipeGen[i], hostGen[i], pipeGen, hostGen)
		}
	}
	t.Logf("dense host/chained/pipelined identical: %v", hostGen)
}

// qatPLEQuantModel is pleQuantModel with the QAT twist: the per-layer model
// projection ships QUANTISED (own gs/bits) instead of bf16 — the shape of the
// gemma-4-E2B/E4B-it-qat-4bit conversions.
func qatPLEQuantModel(t testing.TB, numLayers, dFF, vocab, kvShared int) (*QuantModel, model.Arch) {
	const dModel, nHeads, nKV, headDim = 128, 2, 1, 64
	const pliDim, gs, bits = 64, 64, 4
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
	ts := quantGemma4Tensors(t, arch, gs, bits)
	addPLETensors(t, ts, arch, gs, bits)
	// Replace the bf16 projection with a quantised one — the qat difference.
	plDim := numLayers * pliDim
	p, s, b := quantizeProj(t, plDim, dModel, gs, bits, 977)
	delete(ts, "model.per_layer_model_projection.weight")
	ts["model.per_layer_model_projection.weight"] = safetensors.Tensor{Dtype: "U32", Shape: []int{plDim, dModel * bits / 32}, Data: p}
	ts["model.per_layer_model_projection.scales"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{plDim, dModel / gs}, Data: s}
	ts["model.per_layer_model_projection.biases"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{plDim, dModel / gs}, Data: b}
	lm, err := model.Assemble(ts, arch, model.StandardWeightNames())
	if err != nil {
		t.Fatalf("Assemble: %v", err)
	}
	g, err := loadedToQuant(lm, gs, bits)
	if err != nil {
		t.Fatalf("loadedToQuant: %v", err)
	}
	if !g.HasPLE() {
		t.Fatal("fixture should have the per-layer-input tower")
	}
	if len(g.PerLayerModelProjScales) == 0 || g.PerLayerModelProjGS <= 0 {
		t.Fatal("fixture projection must be quantised (the qat shape)")
	}
	return g, arch
}

// TestQATGPUDecodeSeamEngages pins that the qat shape (quantised PLE
// projection) gets the GPU next-inputs seam — the gate the qat e2b/e4b
// chained/pipelined decode rides.
func TestQATGPUDecodeSeamEngages(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	g, arch := qatPLEQuantModel(t, 3, 256, 32, 0)
	sess, err := NewArchQuantSession(g, arch, 24)
	if err != nil {
		t.Fatalf("session: %v", err)
	}
	if sess.state.icb == nil {
		t.Fatal("qat fixture: ICB not recorded")
	}
	if sess.encNextInputsGPU == nil {
		t.Fatal("qat fixture: GPU next-inputs seam NOT wired (chained/pipelined inactive)")
	}
	if sess.plScratchNew == nil {
		t.Fatal("qat fixture: plScratchNew NOT set (chain gates fail)")
	}
}

// TestQATPipelinedGPUDecodeMatchesHost is the qat sibling of
// TestDensePipelinedGPUDecodeMatchesHost: host loop, chained-GPU and pipelined
// lanes must produce identical tokens on the quantised-projection PLE fixture.
func TestQATPipelinedGPUDecodeMatchesHost(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	g, arch := qatPLEQuantModel(t, 3, 256, 32, 0)
	const maxLen = 24
	prompt := []int32{1, 5, 3, 2}
	const n = 12

	run := func(host, pipe bool) []int32 {
		oldChain := chainedGPUInputsDisabled
		oldPipe := pipelinedGPUDecodeEnabled
		defer func() {
			chainedGPUInputsDisabled = oldChain
			pipelinedGPUDecodeEnabled = oldPipe
		}()
		chainedGPUInputsDisabled = host
		pipelinedGPUDecodeEnabled = pipe
		sess, err := NewArchQuantSession(g, arch, maxLen)
		if err != nil {
			t.Fatalf("session (host=%v pipe=%v): %v", host, pipe, err)
		}
		gen, err := sess.Generate(prompt, n, -1)
		if err != nil {
			t.Fatalf("generate (host=%v pipe=%v): %v", host, pipe, err)
		}
		return gen
	}

	hostGen := run(true, false)
	chainGen := run(false, false)
	pipeGen := run(false, true)
	if len(hostGen) != n {
		t.Fatalf("host generated %d tokens, want %d", len(hostGen), n)
	}
	for i := range hostGen {
		if chainGen[i] != hostGen[i] {
			t.Fatalf("token %d: chained %d != host %d (chain=%v host=%v)", i, chainGen[i], hostGen[i], chainGen, hostGen)
		}
		if pipeGen[i] != hostGen[i] {
			t.Fatalf("token %d: pipelined %d != host %d (pipe=%v host=%v)", i, pipeGen[i], hostGen[i], pipeGen, hostGen)
		}
	}
	t.Logf("qat host/chained/pipelined identical: %v", hostGen)
}
