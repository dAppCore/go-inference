// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"bytes"
	"math"
	"os"
	"testing"
	"unsafe"

	"dappco.re/go/inference/model"
	g4 "dappco.re/go/inference/model/gemma4"
	"github.com/tmc/apple/metal"
)

// TestPerLayerInputsGPUParity gates the on-GPU PLE: PerLayerInputsGPU (per-layer embed-gather + projection
// + norm + combine, all on the GPU from a token id) must reproduce the host PerLayerInputs. This is the
// gate the submit-ahead decode pipeline needs for e2b — the PLE tensor computed on-GPU so the next step
// can be submitted before the token is read back.
func TestPerLayerInputsGPUParity(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	if err := ensureInit(); err != nil {
		t.Skipf("device init: %v", err)
	}
	if !gpuHasGeluKernel() {
		t.Skip("custom kernel library not loaded")
	}
	const vocabPLI, numLayers, pliDim, dModel = 32, 4, 64, 128
	const embGS, embBits = 32, 4
	const eps = float32(1e-6)
	plDim := numLayers * pliDim

	// 4-bit per-layer embedding table [vocabPLI × plDim], bf16 projection [plDim × dModel] + projNorm [pliDim].
	embedPacked := make([]byte, vocabPLI*plDim*embBits/8)
	for i := range embedPacked {
		embedPacked[i] = byte((i*131 + 17) % 256)
	}
	embedScales := toBF16Bytes(syntheticFloat32(vocabPLI*(plDim/embGS), 11))
	embedBiases := toBF16Bytes(syntheticFloat32(vocabPLI*(plDim/embGS), 13))
	projW := toBF16Bytes(syntheticFloat32(plDim*dModel, 7))
	projNormW := toBF16Bytes(syntheticFloat32(pliDim, 19))
	emb := toBF16Bytes(syntheticFloat32(dModel, 23))

	for _, tok := range []int32{0, 5, 17, 31} {
		ref, err := PerLayerInputs(embedPacked, embedScales, embedBiases, projW, nil, nil, projNormW, tok, emb, vocabPLI, numLayers, pliDim, dModel, embGS, embBits, 0, 0, eps, bufView{})
		if err != nil {
			t.Fatalf("tok %d: host PerLayerInputs: %v", tok, err)
		}
		got, err := PerLayerInputsGPU(tok, emb, embedPacked, embedScales, embedBiases, projW, projNormW, vocabPLI, numLayers, pliDim, dModel, embGS, embBits, eps)
		if err != nil {
			t.Fatalf("tok %d: PerLayerInputsGPU: %v", tok, err)
		}
		if cos := cosineBF16(got, ref); cos < 0.9999 {
			t.Fatalf("tok %d: GPU PLE cosine=%.6f vs host PerLayerInputs", tok, cos)
		}
	}
	t.Logf("GPU PLE matches host PerLayerInputs")
}

type perLayerInputsGPUFixture struct {
	embedPacked []byte
	embedScales []byte
	embedBiases []byte
	projW       []byte
	projNormW   []byte
	emb         []byte
}

func newPerLayerInputsGPUFixture(tb testing.TB, vocabPLI, numLayers, pliDim, dModel, embGS, embBits int) perLayerInputsGPUFixture {
	tb.Helper()
	plDim := numLayers * pliDim
	embedPacked := make([]byte, vocabPLI*plDim*embBits/8)
	for i := range embedPacked {
		embedPacked[i] = byte((i*131 + 17) % 256)
	}
	return perLayerInputsGPUFixture{
		embedPacked: embedPacked,
		embedScales: toBF16Bytes(syntheticFloat32(vocabPLI*(plDim/embGS), 11)),
		embedBiases: toBF16Bytes(syntheticFloat32(vocabPLI*(plDim/embGS), 13)),
		projW:       toBF16Bytes(syntheticFloat32(plDim*dModel, 7)),
		projNormW:   toBF16Bytes(syntheticFloat32(pliDim, 19)),
		emb:         toBF16Bytes(syntheticFloat32(dModel, 23)),
	}
}

func TestPerLayerInputsGPUAllocationBudget(t *testing.T) {
	requireNativeRuntime(t)
	const vocabPLI, numLayers, pliDim, dModel = 32, 4, 64, 128
	const embGS, embBits = 32, 4
	const eps = float32(1e-6)
	fx := newPerLayerInputsGPUFixture(t, vocabPLI, numLayers, pliDim, dModel, embGS, embBits)
	if _, err := PerLayerInputsGPU(5, fx.emb, fx.embedPacked, fx.embedScales, fx.embedBiases, fx.projW, fx.projNormW, vocabPLI, numLayers, pliDim, dModel, embGS, embBits, eps); err != nil {
		t.Fatalf("PerLayerInputsGPU warmup: %v", err)
	}

	var gpuErr error
	allocs := testing.AllocsPerRun(5, func() {
		_, gpuErr = PerLayerInputsGPU(17, fx.emb, fx.embedPacked, fx.embedScales, fx.embedBiases, fx.projW, fx.projNormW, vocabPLI, numLayers, pliDim, dModel, embGS, embBits, eps)
	})
	if gpuErr != nil {
		t.Fatalf("PerLayerInputsGPU: %v", gpuErr)
	}
	if allocs > 10 {
		t.Fatalf("PerLayerInputsGPU allocations = %.0f, want <= 10", allocs)
	}
}

func TestPerLayerInputsGPUScratchBuffersUseCallerEmbeddingBacking(t *testing.T) {
	requireNativeRuntime(t)

	const numLayers, pliDim, dModel = 4, 64, 128
	const projScale = float32(0.5)
	plDim := numLayers * pliDim
	emb := toBF16Bytes(syntheticFloat32(dModel, 23))
	scratch, err := getPerLayerInputsGPUScratch(plDim, dModel, projScale)
	if err != nil {
		t.Fatalf("get per-layer GPU scratch: %v", err)
	}
	defer putPerLayerInputsGPUScratch(scratch)
	_, embBuf, _, err := scratch.buffers(17, emb)
	if err != nil {
		t.Fatalf("per-layer GPU scratch buffers: %v", err)
	}
	if got, want := uintptr(embBuf.Contents()), uintptr(unsafe.Pointer(&emb[0])); got != want {
		t.Fatalf("embedding buffer pointer = %#x, want caller backing %#x", got, want)
	}
}

func TestPerLayerInputsGPUIntoReusesOutputBacking(t *testing.T) {
	requireNativeRuntime(t)
	const vocabPLI, numLayers, pliDim, dModel = 32, 4, 64, 128
	const embGS, embBits = 32, 4
	const eps = float32(1e-6)
	fx := newPerLayerInputsGPUFixture(t, vocabPLI, numLayers, pliDim, dModel, embGS, embBits)
	plBytes := numLayers * pliDim * bf16Size
	out := make([]byte, plBytes)
	outPtr := unsafe.Pointer(&out[0])

	got, err := perLayerInputsGPUInto(out, 5, fx.emb, fx.embedPacked, fx.embedScales, fx.embedBiases, fx.projW, fx.projNormW, vocabPLI, numLayers, pliDim, dModel, embGS, embBits, eps)
	if err != nil {
		t.Fatalf("perLayerInputsGPUInto first: %v", err)
	}
	if len(got) != plBytes || unsafe.Pointer(&got[0]) != outPtr {
		t.Fatal("perLayerInputsGPUInto did not reuse caller-owned output backing")
	}
	want, err := PerLayerInputsGPU(5, fx.emb, fx.embedPacked, fx.embedScales, fx.embedBiases, fx.projW, fx.projNormW, vocabPLI, numLayers, pliDim, dModel, embGS, embBits, eps)
	if err != nil {
		t.Fatalf("PerLayerInputsGPU reference: %v", err)
	}
	if cos := cosineBF16(got, want); cos < 0.9999 {
		t.Fatalf("perLayerInputsGPUInto cosine=%.6f vs PerLayerInputsGPU", cos)
	}

	got, err = perLayerInputsGPUInto(got, 17, fx.emb, fx.embedPacked, fx.embedScales, fx.embedBiases, fx.projW, fx.projNormW, vocabPLI, numLayers, pliDim, dModel, embGS, embBits, eps)
	if err != nil {
		t.Fatalf("perLayerInputsGPUInto second: %v", err)
	}
	if len(got) != plBytes || unsafe.Pointer(&got[0]) != outPtr {
		t.Fatal("perLayerInputsGPUInto changed output backing on reuse")
	}
}

func TestPerLayerInputsGPUIntoWritesDirectlyIntoCallerOutput(t *testing.T) {
	requireNativeRuntime(t)
	const vocabPLI, numLayers, pliDim, dModel = 32, 4, 64, 128
	const embGS, embBits = 32, 4
	const eps = float32(1e-6)
	fx := newPerLayerInputsGPUFixture(t, vocabPLI, numLayers, pliDim, dModel, embGS, embBits)
	plDim := numLayers * pliDim
	projScale := float32(1.0 / math.Sqrt(float64(dModel)))
	scratch, err := getPerLayerInputsGPUScratch(plDim, dModel, projScale)
	if err != nil {
		t.Fatalf("getPerLayerInputsGPUScratch: %v", err)
	}
	if scratch.pl == nil || scratch.pl.outPinned == nil {
		t.Fatal("PLE GPU scratch output is not pinned no-copy")
	}
	sentinel := bytes.Repeat([]byte{0xa5}, len(scratch.pl.outPinned.bytes))
	copy(scratch.pl.outPinned.bytes, sentinel)
	putPerLayerInputsGPUScratch(scratch)

	out := make([]byte, plDim*bf16Size)
	if _, err := perLayerInputsGPUInto(out, 5, fx.emb, fx.embedPacked, fx.embedScales, fx.embedBiases, fx.projW, fx.projNormW, vocabPLI, numLayers, pliDim, dModel, embGS, embBits, eps); err != nil {
		t.Fatalf("perLayerInputsGPUInto: %v", err)
	}

	scratch, err = getPerLayerInputsGPUScratch(plDim, dModel, projScale)
	if err != nil {
		t.Fatalf("getPerLayerInputsGPUScratch after call: %v", err)
	}
	defer putPerLayerInputsGPUScratch(scratch)
	if !bytes.Equal(scratch.pl.outPinned.bytes, sentinel) {
		t.Fatal("perLayerInputsGPUInto wrote through pooled scratch instead of caller-owned output")
	}
}

func TestPerLayerInputsGPUScratchPoolKeepsDimensionsResident(t *testing.T) {
	requireNativeRuntime(t)

	const smallPLDim, smallDModel = 64, 128
	const largePLDim, largeDModel = 128, 256
	smallScale := float32(1.0 / math.Sqrt(float64(smallDModel)))
	largeScale := float32(1.0 / math.Sqrt(float64(largeDModel)))

	small, err := getPerLayerInputsGPUScratch(smallPLDim, smallDModel, smallScale)
	if err != nil {
		t.Fatalf("get small PLE GPU scratch: %v", err)
	}
	putPerLayerInputsGPUScratch(small)
	large, err := getPerLayerInputsGPUScratch(largePLDim, largeDModel, largeScale)
	if err != nil {
		t.Fatalf("get large PLE GPU scratch: %v", err)
	}
	putPerLayerInputsGPUScratch(large)
	forceNativeGC()
	forceNativeGC()
	gotSmall, err := getPerLayerInputsGPUScratch(smallPLDim, smallDModel, smallScale)
	if err != nil {
		t.Fatalf("get small PLE GPU scratch again: %v", err)
	}
	defer putPerLayerInputsGPUScratch(gotSmall)
	if gotSmall != small {
		t.Fatal("PLE GPU scratch pool evicted the small scratch after using a larger scratch")
	}
	gotLarge, err := getPerLayerInputsGPUScratch(largePLDim, largeDModel, largeScale)
	if err != nil {
		t.Fatalf("get large PLE GPU scratch again: %v", err)
	}
	defer putPerLayerInputsGPUScratch(gotLarge)
	if gotLarge != large {
		t.Fatal("PLE GPU scratch pool evicted the large scratch after reusing the small scratch")
	}
}

// TestSessionNextInputsGPUParity gates the session wiring (not just the math): a PLE-enabled quant
// session's encNextInputsGPU must reproduce s.embed + s.perLayerInput for the SAME token, using the
// session's real resident weights/dims/scales. This is the seam the chained decode step appends to
// produce the next step's emb+pli on-GPU — a wiring slip (wrong scale, wrong weight, wrong dim) shows
// here before it ever reaches the decode loop.
func TestSessionNextInputsGPUParity(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
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
		t.Fatalf("Assemble: %v", err)
	}
	g, err := loadedToQuant(lm, gs, bits)
	if err != nil {
		t.Fatalf("loadedToQuant: %v", err)
	}
	if !g.HasPLE() {
		t.Fatal("fixture should have the per-layer-input tower")
	}
	sess, err := NewArchQuantSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("NewArchQuantSession: %v", err)
	}
	if sess.encNextInputsGPU == nil {
		t.Fatal("expected the GPU next-inputs seam wired for an e2b-shaped PLE session")
	}

	var tokenBuf metal.MTLBuffer
	var tokenPtr *int32
	var embBuf metal.MTLBuffer
	var embPtr *byte
	var pleScratch *plGPUScratch
	var pleOut metal.MTLBuffer
	var pleOutPtr *byte
	for i, tok := range []int32{1, 5, 17, 31} {
		gotEmb, gotPli, ok, err := sess.nextInputsGPU(tok)
		if err != nil {
			t.Fatalf("tok %d: nextInputsGPU: %v", tok, err)
		}
		if !ok {
			t.Fatalf("tok %d: nextInputsGPU ok=false on a wired session", tok)
		}
		if sess.nextInputToken == nil || sess.nextInputTokenPtr == nil {
			t.Fatalf("tok %d: nextInputsGPU did not use the session token scratch", tok)
		}
		if got := *sess.nextInputTokenPtr; got != tok {
			t.Fatalf("tok %d: cached session token scratch = %d, want %d", tok, got, tok)
		}
		if sess.nextInputEmb == nil || sess.nextInputEmbPtr == nil {
			t.Fatalf("tok %d: nextInputsGPU did not use the session embedding readback scratch", tok)
		}
		if sess.nextInputPLScratch == nil || sess.nextInputPLScratch.outPtr == nil {
			t.Fatalf("tok %d: nextInputsGPU did not use the session PLE scratch", tok)
		}
		if i == 0 {
			tokenBuf, tokenPtr = sess.nextInputToken, sess.nextInputTokenPtr
			embBuf, embPtr = sess.nextInputEmb, sess.nextInputEmbPtr
			pleScratch, pleOut, pleOutPtr = sess.nextInputPLScratch, sess.nextInputPLScratch.out, sess.nextInputPLScratch.outPtr
		} else {
			if sess.nextInputToken != tokenBuf {
				t.Fatalf("tok %d: nextInputsGPU did not reuse the session token buffer", tok)
			}
			if sess.nextInputTokenPtr != tokenPtr {
				t.Fatalf("tok %d: nextInputsGPU token scratch pointer changed", tok)
			}
			if sess.nextInputEmb != embBuf {
				t.Fatalf("tok %d: nextInputsGPU did not reuse the session embedding buffer", tok)
			}
			if sess.nextInputEmbPtr != embPtr {
				t.Fatalf("tok %d: nextInputsGPU embedding scratch pointer changed", tok)
			}
			if sess.nextInputPLScratch != pleScratch {
				t.Fatalf("tok %d: nextInputsGPU did not reuse the session PLE scratch", tok)
			}
			if sess.nextInputPLScratch.out != pleOut {
				t.Fatalf("tok %d: nextInputsGPU PLE output buffer changed", tok)
			}
			if sess.nextInputPLScratch.outPtr != pleOutPtr {
				t.Fatalf("tok %d: nextInputsGPU PLE output pointer changed", tok)
			}
		}
		wantEmb, err := sess.embed(tok)
		if err != nil {
			t.Fatalf("tok %d: host embed: %v", tok, err)
		}
		wantPli, err := sess.perLayerInput(tok, wantEmb)
		if err != nil {
			t.Fatalf("tok %d: host perLayerInput: %v", tok, err)
		}
		if cos := cosineBF16(gotEmb, wantEmb); cos < 0.9999 {
			t.Fatalf("tok %d: GPU emb cosine=%.6f vs host s.embed", tok, cos)
		}
		if cos := cosineBF16(gotPli, wantPli); cos < 0.9999 {
			t.Fatalf("tok %d: GPU pli cosine=%.6f vs host s.perLayerInput", tok, cos)
		}
	}
	t.Logf("session GPU next-inputs (emb+pli) matches host s.embed + s.perLayerInput")
}

func TestSessionNextInputsGPUReusesHostReadback(t *testing.T) {
	requireNativeRuntime(t)
	g, arch, maxLen := icbSessionStateFixture(t)
	sess := newICBSessionStateFixture(t, g, arch, maxLen)
	if sess.encNextInputsGPU == nil {
		t.Fatal("expected the GPU next-inputs seam wired for an e2b-shaped PLE session")
	}

	emb0, pli0, ok, err := sess.nextInputsGPU(1)
	if err != nil || !ok {
		t.Fatalf("first nextInputsGPU ok=%v err=%v", ok, err)
	}
	emb0Ptr := uintptr(unsafe.Pointer(&emb0[0]))
	pli0Ptr := uintptr(unsafe.Pointer(&pli0[0]))

	emb1, pli1, ok, err := sess.nextInputsGPU(5)
	if err != nil || !ok {
		t.Fatalf("second nextInputsGPU ok=%v err=%v", ok, err)
	}
	if got := uintptr(unsafe.Pointer(&emb1[0])); got != emb0Ptr {
		t.Fatalf("nextInputsGPU embedding readback backing changed: %#x != %#x", got, emb0Ptr)
	}
	if got := uintptr(unsafe.Pointer(&pli1[0])); got != pli0Ptr {
		t.Fatalf("nextInputsGPU PLE readback backing changed: %#x != %#x", got, pli0Ptr)
	}

	tokens := []int32{1, 5, 17, 31}
	var gpuErr error
	i := 0
	allocs := testing.AllocsPerRun(5, func() {
		_, _, ok, gpuErr = sess.nextInputsGPU(tokens[i%len(tokens)])
		i++
	})
	if gpuErr != nil || !ok {
		t.Fatalf("warmed nextInputsGPU ok=%v err=%v", ok, gpuErr)
	}
	if allocs > 20 {
		t.Fatalf("nextInputsGPU allocations = %.0f, want <= 20", allocs)
	}
}

func TestSessionNextInputsGPUPLEReadbackUsesPinnedNoCopyBacking(t *testing.T) {
	requireNativeRuntime(t)
	g, arch, maxLen := icbSessionStateFixture(t)
	sess := newICBSessionStateFixture(t, g, arch, maxLen)
	if sess.encNextInputsGPU == nil {
		t.Fatal("expected the GPU next-inputs seam wired for an e2b-shaped PLE session")
	}

	_, pli, ok, err := sess.nextInputsGPU(1)
	if err != nil || !ok {
		t.Fatalf("nextInputsGPU ok=%v err=%v", ok, err)
	}
	if sess.nextInputPLScratch == nil || sess.nextInputPLScratch.outPinned == nil || sess.nextInputPLScratch.outPinned.pinner == nil {
		t.Fatal("next-input PLE output scratch is not pinned no-copy")
	}
	if sess.nextInputPLScratch.out == nil || sess.nextInputPLScratch.out.Contents() != unsafe.Pointer(&sess.nextInputPLScratch.outPinned.bytes[0]) {
		t.Fatal("next-input PLE Metal buffer is not backed by pinned Go bytes")
	}
	if len(pli) == 0 || unsafe.Pointer(&pli[0]) != unsafe.Pointer(&sess.nextInputPLScratch.outPinned.bytes[0]) {
		t.Fatal("next-input PLE readback does not use the pinned Go backing")
	}
}
