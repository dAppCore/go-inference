// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"bytes"
	"sort"
	"strings"
	"testing"
	"unsafe"

	"dappco.re/go/inference/model"
	g4 "dappco.re/go/inference/model/gemma4"
	"github.com/tmc/apple/metal"
)

func TestNewHeadEncoderNilShardBuffersFallsBack(t *testing.T) {
	h, err := newHeadEncoder(nil, nil, nil, nil, nil, 64, 128, 64, 4, 1e-5, 0, false)
	if err != nil {
		t.Fatalf("newHeadEncoder nil shard buffers: %v", err)
	}
	if h != nil {
		t.Fatalf("newHeadEncoder nil shard buffers = %+v, want nil fallback", h)
	}
}

func TestTokenSuppressed_Membership(t *testing.T) {
	if !tokenSuppressed(7, []int32{2, 7, 9}) {
		t.Fatal("tokenSuppressed did not find a present token")
	}
	if tokenSuppressed(8, []int32{2, 7, 9}) {
		t.Fatal("tokenSuppressed reported an absent token")
	}
}

func TestGreedyBF16Suppressed_SelectsBestAllowed(t *testing.T) {
	logits := toBF16Bytes([]float32{1, 5, 3, 4})
	got, err := greedyBF16Suppressed(logits, 4, []int32{1, 3})
	if err != nil || got != 2 {
		t.Fatalf("greedyBF16Suppressed = %d, %v; want 2, nil", got, err)
	}
}

func TestGreedyBF16Suppressed_AllSuppressed(t *testing.T) {
	if _, err := greedyBF16Suppressed(toBF16Bytes([]float32{1, 2}), 2, []int32{0, 1}); err == nil {
		t.Fatal("greedyBF16Suppressed accepted an entirely suppressed vocabulary")
	}
}

func TestBF16NaNScanBytes_Mixed(t *testing.T) {
	b := []byte{0, 0, 0xc1, 0x7f, 0, 0x7f, 1}
	count, first := bf16NaNScanBytes(b)
	if count != 1 || first != 1 {
		t.Fatalf("bf16NaNScanBytes = (%d, %d), want (1, 1)", count, first)
	}
}

func TestHeadEncoderHiddenBufferOffsetInRange_InvalidReceiver(t *testing.T) {
	var h *headEncoder
	if h.hiddenBufferOffsetInRange(nil, 0) {
		t.Fatal("hiddenBufferOffsetInRange accepted a nil receiver and buffer")
	}
}

func TestHeadScratchCachesSharedContentsPointers(t *testing.T) {
	requireNativeRuntime(t)

	topK := newHeadTopKScratch(8, 4, 16, 32, false)
	if topK.sampleParamsPtr == nil {
		t.Fatal("top-k scratch did not cache sample params contents pointer")
	}
	if topK.outTokenPtr == nil {
		t.Fatal("top-k scratch did not cache output token contents pointer")
	}
	if topK.topValuesPtr == nil || topK.topIndicesPtr == nil {
		t.Fatal("top-k scratch did not cache top-k readback contents pointers")
	}
	topKParamsPtr := topK.sampleParamsPtr
	topK.sampleParamsBuffer(model.SampleParams{Temperature: 0.7, TopK: 4, TopP: 0.9, MinP: 0.01}, 0.25, 8)
	if topK.sampleParamsPtr != topKParamsPtr {
		t.Fatal("top-k sample params contents pointer changed after reuse")
	}
	gotTopK := *topK.sampleParamsPtr
	if gotTopK.n != 8 || gotTopK.topK != 4 || gotTopK.temperature != 0.7 || gotTopK.topP != 0.9 || gotTopK.minP != 0.01 || gotTopK.draw != 0.25 {
		t.Fatalf("top-k sample params = %+v", gotTopK)
	}
	topValues := unsafe.Slice(topK.topValuesPtr, 4)
	topIDs := unsafe.Slice(topK.topIndicesPtr, 4)
	topValues[0], topValues[1], topValues[2], topValues[3] = 1.5, 0.75, -1, -2
	topIDs[0], topIDs[1], topIDs[2], topIDs[3] = 3, 4, -1, 99
	h := &headEncoder{vocab: 8}
	gotCandidateLogits, gotIDs, ok, err := h.readTopKCandidatesInto(topK, 4, nil, nil)
	if err != nil || !ok {
		t.Fatalf("readTopKCandidatesInto cached pointers ok=%v err=%v", ok, err)
	}
	if len(gotIDs) != 2 || gotIDs[0] != 3 || gotIDs[1] != 4 || len(gotCandidateLogits) != 2*bf16Size {
		t.Fatalf("cached top-k readback ids=%v logits=%d, want [3 4]/%d", gotIDs, len(gotCandidateLogits), 2*bf16Size)
	}
	topK.suppressBuffer([]int32{4, 5, 6})
	if topK.suppressPtr == nil {
		t.Fatal("top-k scratch did not cache suppress contents pointer")
	}
	suppressPtr := topK.suppressPtr
	topK.suppressBuffer([]int32{8, 9})
	if topK.suppressPtr != suppressPtr {
		t.Fatal("top-k suppress contents pointer changed without growing buffer")
	}
	gotSuppress := unsafe.Slice(topK.suppressPtr, 2)
	if gotSuppress[0] != 8 || gotSuppress[1] != 9 {
		t.Fatalf("suppress buffer = %v, want [8 9]", gotSuppress)
	}
	topK.historyBuffer([]int32{1, 2, 3})
	if topK.historyPtr == nil {
		t.Fatal("top-k scratch did not cache history contents pointer")
	}
	historyPtr := topK.historyPtr
	topK.historyBuffer([]int32{7})
	if topK.historyPtr != historyPtr {
		t.Fatal("top-k history contents pointer changed without growing buffer")
	}
	gotHistory := unsafe.Slice(topK.historyPtr, 1)
	if gotHistory[0] != 7 {
		t.Fatalf("history buffer = %v, want [7]", gotHistory)
	}

	logits := newHeadGreedyScratch(1, 16, 32, true, false)
	if logits.sampleParamsPtr == nil {
		t.Fatal("logits scratch did not cache sample params contents pointer")
	}
	if logits.outTokenPtr == nil {
		t.Fatal("logits scratch did not cache output token contents pointer")
	}
	if logits.logitsPtr == nil {
		t.Fatal("logits scratch did not cache full-logits contents pointer")
	}
	logitsBytes := unsafe.Slice(logits.logitsPtr, 4)
	logitsBytes[0], logitsBytes[1], logitsBytes[2], logitsBytes[3] = 0xaa, 0xbb, 0xcc, 0xdd
	gotLogitsBytes := unsafe.Slice((*byte)(logits.logits.Contents()), 4)
	if !bytes.Equal(gotLogitsBytes, logitsBytes) {
		t.Fatalf("cached full-logits pointer did not write through to Metal buffer: got %v want %v", gotLogitsBytes, logitsBytes)
	}
	logitsParamsPtr := logits.sampleParamsPtr
	logits.logitsSampleParamsBuffer(model.SampleParams{Temperature: 0.8, TopK: 3, TopP: 0.95, MinP: 0.02, RepeatPenalty: 1.2}, 0.5, 32, 2, 4)
	if logits.sampleParamsPtr != logitsParamsPtr {
		t.Fatal("logits sample params contents pointer changed after reuse")
	}
	gotLogits := *logits.sampleParamsPtr
	if gotLogits.vocab != 32 || gotLogits.suppressCount != 2 || gotLogits.historyCount != 4 || gotLogits.topK != 3 ||
		gotLogits.temperature != 0.8 || gotLogits.topP != 0.95 || gotLogits.minP != 0.02 || gotLogits.draw != 0.5 || gotLogits.repeatPenalty != 1.2 {
		t.Fatalf("logits sample params = %+v", gotLogits)
	}
	logits.suppressBuffer([]int32{10, 11, 12})
	if logits.suppressPtr == nil {
		t.Fatal("logits scratch did not cache suppress contents pointer")
	}
	logitsSuppressPtr := logits.suppressPtr
	logits.suppressBuffer([]int32{13, 14})
	if logits.suppressPtr != logitsSuppressPtr {
		t.Fatal("logits suppress contents pointer changed without growing buffer")
	}
	gotLogitsSuppress := unsafe.Slice(logits.suppressPtr, 2)
	if gotLogitsSuppress[0] != 13 || gotLogitsSuppress[1] != 14 {
		t.Fatalf("logits suppress buffer = %v, want [13 14]", gotLogitsSuppress)
	}
	logits.historyBuffer([]int32{20, 21, 22})
	if logits.historyPtr == nil {
		t.Fatal("logits scratch did not cache history contents pointer")
	}
	logitsHistoryPtr := logits.historyPtr
	logits.historyBuffer([]int32{23})
	if logits.historyPtr != logitsHistoryPtr {
		t.Fatal("logits history contents pointer changed without growing buffer")
	}
	gotLogitsHistory := unsafe.Slice(logits.historyPtr, 1)
	if gotLogitsHistory[0] != 23 {
		t.Fatalf("logits history buffer = %v, want [23]", gotLogitsHistory)
	}
}

func TestHeadGreedyScratchLogitsOutputViewReusesPinnedOwnerBuffer(t *testing.T) {
	requireNativeRuntime(t)

	const vocab = 8
	pinned, err := newPinnedNoCopyBytes(vocab * bf16Size)
	if err != nil {
		t.Fatalf("newPinnedNoCopyBytes: %v", err)
	}
	defer pinned.Close()

	scratch := newHeadGreedyScratch(1, 64, vocab, true, false)
	defer scratch.closeLogitsOutputView()

	buf, ok := scratch.logitsOutputView(pinned.bytes)
	if !ok {
		t.Fatal("logits output view did not accept pinned caller bytes")
	}
	if got, want := buf.GetID(), pinned.buf.GetID(); got != want {
		t.Fatalf("logits output view buffer id = %d, want pinned owner buffer %d", got, want)
	}
	if got, want := uintptr(buf.Contents()), uintptr(unsafe.Pointer(&pinned.bytes[0])); got != want {
		t.Fatalf("logits output view pointer = %#x, want pinned backing %#x", got, want)
	}
}

func TestHeadSamplerScratchUsesPinnedNoCopySuppressHistory(t *testing.T) {
	requireNativeRuntime(t)

	topK := newHeadTopKScratch(8, 4, 16, 32, false)
	defer func() {
		if topK.suppressPinned != nil {
			topK.suppressPinned.Close()
		}
		if topK.historyPinned != nil {
			topK.historyPinned.Close()
		}
	}()
	topKSuppress := topK.suppressBuffer([]int32{1, 2, 3})
	if topK.suppressPinned == nil || topK.suppressPinned.pinner == nil {
		t.Fatal("top-k suppress scratch is not pinned no-copy")
	}
	if topKSuppress == nil || topKSuppress.Contents() != unsafe.Pointer(&topK.suppressPinned.bytes[0]) {
		t.Fatal("top-k suppress Metal buffer is not backed by pinned Go bytes")
	}
	firstTopKSuppress := topK.suppressPinned
	topK.suppressBuffer([]int32{4, 5})
	if topK.suppressPinned != firstTopKSuppress {
		t.Fatal("top-k suppress pinned scratch changed without growing")
	}
	topK.suppressBuffer([]int32{6, 7, 8, 9})
	if firstTopKSuppress.bytes != nil || firstTopKSuppress.pinner != nil {
		t.Fatal("top-k suppress pinned scratch was not closed on grow")
	}

	topKHistory := topK.historyBuffer([]int32{10, 11, 12})
	if topK.historyPinned == nil || topK.historyPinned.pinner == nil {
		t.Fatal("top-k history scratch is not pinned no-copy")
	}
	if topKHistory == nil || topKHistory.Contents() != unsafe.Pointer(&topK.historyPinned.bytes[0]) {
		t.Fatal("top-k history Metal buffer is not backed by pinned Go bytes")
	}

	logits := newHeadGreedyScratch(1, 16, 32, true, false)
	defer func() {
		if logits.suppressPinned != nil {
			logits.suppressPinned.Close()
		}
		if logits.historyPinned != nil {
			logits.historyPinned.Close()
		}
	}()
	logitsSuppress := logits.suppressBuffer([]int32{13, 14})
	if logits.suppressPinned == nil || logits.suppressPinned.pinner == nil {
		t.Fatal("logits suppress scratch is not pinned no-copy")
	}
	if logitsSuppress == nil || logitsSuppress.Contents() != unsafe.Pointer(&logits.suppressPinned.bytes[0]) {
		t.Fatal("logits suppress Metal buffer is not backed by pinned Go bytes")
	}

	logitsHistory := logits.historyBuffer([]int32{15, 16})
	if logits.historyPinned == nil || logits.historyPinned.pinner == nil {
		t.Fatal("logits history scratch is not pinned no-copy")
	}
	if logitsHistory == nil || logitsHistory.Contents() != unsafe.Pointer(&logits.historyPinned.bytes[0]) {
		t.Fatal("logits history Metal buffer is not backed by pinned Go bytes")
	}
}

func TestNewHeadEncoderNilShardBuffersBuildsOwnedBF16Head(t *testing.T) {
	requireNativeRuntime(t)

	const dModel, vocab = 64, 19
	const eps = float32(1e-6)
	hidden := toBF16Bytes(syntheticFloat32(dModel, 51))
	finalNorm := toBF16Bytes(syntheticFloat32(dModel, 53))
	head := toBF16Bytes(syntheticFloat32(vocab*dModel, 57))

	h, err := newHeadEncoder(nil, finalNorm, head, nil, nil, dModel, vocab, 0, 0, eps, 0, false)
	if err != nil {
		t.Fatalf("newHeadEncoder owned bf16: %v", err)
	}
	if h == nil {
		t.Fatal("newHeadEncoder owned bf16 returned nil; in-memory sessions would miss direct greedy")
	}

	logits, err := h.encode(hidden, true)
	if err != nil {
		t.Fatalf("owned bf16 head logits: %v", err)
	}
	want, err := model.Greedy(logits, vocab)
	if err != nil {
		t.Fatalf("owned bf16 full-logits greedy: %v", err)
	}
	got, ok, err := h.greedy(hidden, nil)
	if err != nil {
		t.Fatalf("owned bf16 direct greedy: %v", err)
	}
	if !ok {
		t.Fatal("owned bf16 direct greedy declined")
	}
	if got != want {
		t.Fatalf("owned bf16 direct greedy = %d, want full-logits greedy %d", got, want)
	}
}

func TestNewHeadEncoderOwnedBF16HeadUsesResidentBacking(t *testing.T) {
	requireNativeRuntime(t)

	const dModel, vocab = 64, 19
	finalNorm := toBF16Bytes(syntheticFloat32(dModel, 53))
	head := toBF16Bytes(syntheticFloat32(vocab*dModel, 57))

	h, err := newHeadEncoder(nil, finalNorm, head, nil, nil, dModel, vocab, 0, 0, 1e-6, 0, false)
	if err != nil {
		t.Fatalf("newHeadEncoder owned bf16: %v", err)
	}
	if h == nil {
		t.Fatal("newHeadEncoder owned bf16 returned nil")
	}
	if got, want := uintptr(h.finalNorm.buf.Contents()), uintptr(unsafe.Pointer(&finalNorm[0])); got != want {
		t.Fatalf("owned bf16 final norm buffer pointer = %#x, want caller backing %#x", got, want)
	}
	if got, want := uintptr(h.weight.buf.Contents()), uintptr(unsafe.Pointer(&head[0])); got != want {
		t.Fatalf("owned bf16 head weight buffer pointer = %#x, want caller backing %#x", got, want)
	}
}

func TestHeadEncoderEncodeIntoWritesCallerOutputWithoutTouchingPooledLogits(t *testing.T) {
	requireNativeRuntime(t)

	const dModel, vocab = 64, 19
	const eps = float32(1e-6)
	hidden := toBF16Bytes(syntheticFloat32(dModel, 61))
	finalNorm := toBF16Bytes(syntheticFloat32(dModel, 67))
	head := toBF16Bytes(syntheticFloat32(vocab*dModel, 71))

	h, err := newHeadEncoder(nil, finalNorm, head, nil, nil, dModel, vocab, 0, 0, eps, 0, false)
	if err != nil {
		t.Fatalf("newHeadEncoder owned bf16: %v", err)
	}
	if h == nil {
		t.Fatal("newHeadEncoder owned bf16 returned nil")
	}

	want, err := LMHeadBF16(hidden, finalNorm, head, dModel, vocab, eps, 0)
	if err != nil {
		t.Fatalf("LMHeadBF16 reference: %v", err)
	}
	scratch := newHeadGreedyScratch(1, dModel, vocab, true, false)
	logitBytes := unsafe.Slice((*byte)(scratch.logits.Contents()), vocab*bf16Size)
	for i := range logitBytes {
		logitBytes[i] = 0x7f
	}
	h.putGreedyScratch(scratch)

	out := make([]byte, vocab*bf16Size)
	logits, err := h.encodeInto(hidden, true, out)
	if err != nil {
		t.Fatalf("encodeInto: %v", err)
	}
	if len(logits) == 0 || unsafe.Pointer(&logits[0]) != unsafe.Pointer(&out[0]) {
		t.Fatal("encodeInto did not return the caller output backing")
	}
	if !bytes.Equal(logits, want) {
		t.Fatalf("encodeInto logits diverged from LMHeadBF16")
	}
	reused := h.getGreedyScratch(1, true, false)
	defer h.putGreedyScratch(reused)
	if reused.logits != scratch.logits {
		t.Fatal("encode did not return the seeded full-logits scratch to the pool")
	}
	got := unsafe.Slice((*byte)(reused.logits.Contents()), len(logits))
	if bytes.Equal(got, logits) {
		t.Fatal("encodeInto still staged logits through the pooled scratch before copying to caller output")
	}
	for i, b := range got {
		if b != 0x7f {
			t.Fatalf("pooled logits scratch byte %d changed to %#x; want sentinel 0x7f", i, b)
		}
	}
}

func TestNewHeadEncoderNilShardBuffersBuildsOwnedQuantHead(t *testing.T) {
	requireNativeRuntime(t)

	const dModel, vocab, groupSize, bits = 64, 17, 32, 4
	const eps = float32(1e-6)
	hidden := toBF16Bytes(syntheticFloat32(dModel, 31))
	finalNorm := toBF16Bytes(syntheticFloat32(dModel, 37))

	packed := make([]byte, vocab*dModel*bits/8)
	for i := range packed {
		packed[i] = byte((i*29 + 17) & 0xff)
	}
	sidecars := vocab * (dModel / groupSize)
	scalesF, biasesF := make([]float32, sidecars), make([]float32, sidecars)
	for i := range scalesF {
		scalesF[i] = 0.015 + float32((i%7)+1)*0.002
		biasesF[i] = -0.08 + float32((i%11))*0.01
	}

	h, err := newHeadEncoder(nil, finalNorm, packed, toBF16Bytes(scalesF), toBF16Bytes(biasesF), dModel, vocab, groupSize, bits, eps, 0, true)
	if err != nil {
		t.Fatalf("newHeadEncoder owned quant: %v", err)
	}
	if h == nil {
		t.Fatal("newHeadEncoder owned quant returned nil; in-memory quant sessions would miss direct greedy")
	}

	logits, err := h.encode(hidden, true)
	if err != nil {
		t.Fatalf("owned quant head logits: %v", err)
	}
	want, err := LMHeadQuant(hidden, finalNorm, packed, toBF16Bytes(scalesF), toBF16Bytes(biasesF), dModel, vocab, groupSize, bits, eps, 0)
	if err != nil {
		t.Fatalf("LMHeadQuant reference: %v", err)
	}
	if !bytes.Equal(logits, want) {
		t.Fatalf("owned quant head logits diverged from LMHeadQuant")
	}
}

func TestOwnedQuantHeadFusedTopKMatchesFullHead(t *testing.T) {
	requireNativeRuntime(t)

	const dModel, vocab, groupSize, bits = 512, 256, 64, 4
	const topK = 7
	const eps = float32(1e-6)
	const softCap = float32(2)
	if !q4LMHeadTopKUsable(dModel, vocab, groupSize, bits, topK) {
		t.Skip("fused q4 lm-head top-k custom kernel unavailable")
	}

	hidden := toBF16Bytes(syntheticFloat32(dModel, 31))
	finalNorm := toBF16Bytes(syntheticFloat32(dModel, 37))
	packed := make([]byte, vocab*dModel*bits/8)
	for i := range packed {
		packed[i] = byte((i*29 + 17) & 0xff)
	}
	sidecars := vocab * (dModel / groupSize)
	scalesF, biasesF := make([]float32, sidecars), make([]float32, sidecars)
	for i := range scalesF {
		scalesF[i] = 0.015 + float32((i%7)+1)*0.002
		biasesF[i] = -0.08 + float32(i%11)*0.01
	}

	h, err := newHeadEncoder(nil, finalNorm, packed, toBF16Bytes(scalesF), toBF16Bytes(biasesF), dModel, vocab, groupSize, bits, eps, softCap, true)
	if err != nil {
		t.Fatalf("newHeadEncoder owned quant: %v", err)
	}
	suppress := []int32{2, 7, 19}
	full, err := h.encode(hidden, false)
	if err != nil {
		t.Fatalf("full quant head logits: %v", err)
	}
	type candidate struct {
		id int32
		v  float32
	}
	want := make([]candidate, 0, vocab)
	for i := range vocab {
		if tokenSuppressed(i, suppress) {
			continue
		}
		want = append(want, candidate{id: int32(i), v: bf16ToF32(full[i*bf16Size], full[i*bf16Size+1])})
	}
	sort.SliceStable(want, func(i, j int) bool {
		if want[i].v == want[j].v {
			return want[i].id < want[j].id
		}
		return want[i].v > want[j].v
	})

	gotLogits, gotIDs, ok, err := h.sampleTopKCandidatesFusedQ4(hidden, topK, suppress)
	if err != nil {
		t.Fatalf("sampleTopKCandidates: %v", err)
	}
	if !ok {
		t.Fatal("sampleTopKCandidates declined fused q4 top-k shape")
	}
	if len(gotIDs) != topK || len(gotLogits) != topK*bf16Size {
		t.Fatalf("candidate lengths: ids=%d logits=%d, want %d/%d", len(gotIDs), len(gotLogits), topK, topK*bf16Size)
	}
	for i := range topK {
		if gotIDs[i] != want[i].id {
			t.Fatalf("topK[%d] id=%d, want %d (got %v want top=%v)", i, gotIDs[i], want[i].id, gotIDs, want[:topK])
		}
		gotV := bf16ToF32(gotLogits[i*bf16Size], gotLogits[i*bf16Size+1])
		if gotV != want[i].v {
			t.Fatalf("topK[%d] value=%g, want %g", i, gotV, want[i].v)
		}
	}
}

func TestOwnedQuantHeadDirectGreedyMatchesContractFixture(t *testing.T) {
	requireNativeRuntime(t)

	const gs, bits = 32, 4
	const maxLen, maxNew = 16, 6
	cfg := g4.Config{
		HiddenSize: 128, NumHiddenLayers: 2, IntermediateSize: 256,
		NumAttentionHeads: 2, NumKeyValueHeads: 1, HeadDim: 64, VocabSize: 32, RMSNormEps: 1e-6,
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
	sess, err := NewArchQuantSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("NewArchQuantSession: %v", err)
	}
	head, err := newHeadEncoder(nil, g.FinalNorm, g.LMHead, g.LMHeadScales, g.LMHeadBiases, arch.Hidden, arch.Vocab, gs, bits, arch.Eps, arch.SoftCap, true)
	if err != nil {
		t.Fatalf("newHeadEncoder owned quant: %v", err)
	}
	if head == nil {
		t.Fatal("newHeadEncoder owned quant returned nil")
	}

	prompt := []int32{1, 5, 3}
	var hidden []byte
	for _, id := range prompt {
		if hidden, err = sess.stepID(id); err != nil {
			t.Fatalf("prefill stepID(%d): %v", id, err)
		}
	}
	for i := range maxNew {
		logits, err := head.encode(hidden, true)
		if err != nil {
			t.Fatalf("owned quant full logits at generated step %d: %v", i, err)
		}
		want, err := model.Greedy(logits, arch.Vocab)
		if err != nil {
			t.Fatalf("owned quant full-logits greedy at generated step %d: %v", i, err)
		}
		got, ok, err := head.greedy(hidden, nil)
		if err != nil {
			t.Fatalf("owned quant direct greedy at generated step %d: %v", i, err)
		}
		if !ok {
			t.Fatal("owned quant direct greedy declined contract fixture")
		}
		if got != want {
			t.Fatalf("owned quant direct greedy at generated step %d = %d, want resident qmv full-logits greedy %d", i, got, want)
		}
		if hidden, err = sess.stepID(want); err != nil {
			t.Fatalf("generated stepID(%d) at step %d: %v", want, i, err)
		}
	}
}

func TestHeadEncoderRejectsHiddenShapeMismatch(t *testing.T) {
	h := &headEncoder{dModel: 2, vocab: 2}
	if _, err := h.encode(toBF16Bytes([]float32{1}), false); err == nil {
		t.Fatal("expected headEncoder.encode to reject hidden shape mismatch")
	}
}

func TestHeadEncoderEncodeBufferInto_RejectsMissingHiddenBuffer(t *testing.T) {
	h := &headEncoder{vocab: 3}
	if _, err := h.encodeBufferInto(nil, false, nil); err == nil {
		t.Fatal("encodeBufferInto accepted a nil hidden buffer")
	}
}

func TestHeadEncoderGreedyBufferAtInPool_RejectsMissingHiddenBuffer(t *testing.T) {
	h := &headEncoder{dModel: 4, vocab: 8}
	if _, ok, err := h.greedyBufferAtInPool(nil, 0, nil); err == nil || !ok {
		t.Fatalf("greedyBufferAtInPool(nil) = ok=%v err=%v, want handled error", ok, err)
	}
}

func TestHeadEncoderSampleLogitsTokenBufferAtInPool_RejectsMissingHiddenBuffer(t *testing.T) {
	h := &headEncoder{dModel: 4, vocab: 8}
	if _, ok, err := h.sampleLogitsTokenBufferAtInPool(nil, 0, model.SampleParams{}, 0, nil); err == nil || !ok {
		t.Fatalf("sampleLogitsTokenBufferAtInPool(nil) = ok=%v err=%v, want handled error", ok, err)
	}
}

func TestHeadEncoderSampleTopKTokenBufferAtInPool_RejectsMissingHiddenBuffer(t *testing.T) {
	h := &headEncoder{dModel: 4, vocab: 8}
	if _, ok, err := h.sampleTopKTokenBufferAtInPool(nil, 0, model.SampleParams{TopK: 1}, 0, nil); err == nil || !ok {
		t.Fatalf("sampleTopKTokenBufferAtInPool(nil) = ok=%v err=%v, want handled error", ok, err)
	}
}

func TestHeadEncoderSampleTopKCandidatesBufferWithHistoryInto_RejectsMissingHiddenBuffer(t *testing.T) {
	h := &headEncoder{dModel: 4, vocab: 8}
	if _, _, ok, err := h.sampleTopKCandidatesBufferWithHistoryInto(nil, 1, nil, nil, 1, nil, nil, false); err == nil || !ok {
		t.Fatalf("sampleTopKCandidatesBufferWithHistoryInto(nil) = ok=%v err=%v, want handled error", ok, err)
	}
}

func TestHeadEncoderEncodeTopKCandidateRows_DeclinesUnusableHead(t *testing.T) {
	h := &headEncoder{dModel: 4, vocab: 8}
	scratch, candidates, ok, err := h.encodeTopKCandidateRows(nil, nil, 1, nil, nil, 1, false, false)
	if scratch != nil || candidates != 0 || ok || err != nil {
		t.Fatalf("encodeTopKCandidateRows unusable head = scratch=%v candidates=%d ok=%v err=%v, want nil/0/false/nil", scratch, candidates, ok, err)
	}
}

func TestHeadEncoderEncodeTopKCandidateRowsObjectAt_DeclinesInvalidTopK(t *testing.T) {
	h := &headEncoder{vocab: 8}
	var enc metal.MTLComputeCommandEncoderObject
	scratch, candidates, ok, err := h.encodeTopKCandidateRowsObjectAt(enc, nil, 0, 0, nil, nil, 1, false, false)
	if scratch != nil || candidates != 0 || ok || err != nil {
		t.Fatalf("encodeTopKCandidateRowsObjectAt invalid topK = scratch=%v candidates=%d ok=%v err=%v, want nil/0/false/nil", scratch, candidates, ok, err)
	}
}

func TestHeadEncoderReadTopKCandidatesInto_RejectsMissingScratch(t *testing.T) {
	h := &headEncoder{vocab: 8}
	if _, _, ok, err := h.readTopKCandidatesInto(nil, 1, nil, nil); err == nil || !ok {
		t.Fatalf("readTopKCandidatesInto(nil) = ok=%v err=%v, want handled error", ok, err)
	}
}

func TestHeadEncoderGreedyRowsBufferInPool_RejectsInvalidBatch(t *testing.T) {
	var h *headEncoder
	if ok, err := h.greedyRowsBufferInPool(nil, 0, 0, nil, nil); err == nil || ok {
		t.Fatalf("greedyRowsBufferInPool invalid batch = ok=%v err=%v, want false/error", ok, err)
	}
}

func TestHeadEncoderEncodeTopKCandidateRows_WrongMetallibReturnsError(t *testing.T) {
	requireNativeRuntime(t)
	h, hidden, _ := covSampleHeadFixture(t)
	withWrongMainLibrary(t, func() {
		cb := queue.CommandBuffer()
		enc := cb.ComputeCommandEncoder()
		scratch, _, ok, err := h.encodeTopKCandidateRows(enc, sharedBytes(hidden), 1, nil, nil, 1, false, false)
		enc.EndEncoding()
		if scratch != nil {
			h.putTopKScratch(scratch)
		}
		// RMSNorm is the first encode stage and lives in the main metallib. A
		// real but wrong library must surface that lookup failure, not decline.
		if !ok || err == nil {
			t.Fatalf("encodeTopKCandidateRows wrong metallib = ok=%v err=%v, want handled error", ok, err)
		}
	})
}

func TestGreedyBF16Suppressed_Good(t *testing.T) {
	logits := toBF16Bytes([]float32{-3.5, 7.25, 2.5, 6.75, -1.25})

	got, err := greedyBF16Suppressed(logits, 5, []int32{1, 4})
	if err != nil {
		t.Fatalf("greedyBF16Suppressed: %v", err)
	}
	if got != 3 {
		t.Fatalf("greedyBF16Suppressed = %d, want 3", got)
	}
}

func TestGreedyBF16Suppressed_Bad(t *testing.T) {
	logits := toBF16Bytes([]float32{1.5, -2.25, 3.75})

	if _, err := greedyBF16Suppressed(logits[:len(logits)-1], 3, []int32{0}); err == nil {
		t.Fatal("greedyBF16Suppressed accepted malformed bf16 logits")
	}
}

func TestGreedyBF16Suppressed_Ugly(t *testing.T) {
	logits := toBF16Bytes([]float32{1.5, -2.25, 3.75})

	if _, err := greedyBF16Suppressed(logits, 3, []int32{2, 0, 1, 2}); err == nil {
		t.Fatal("greedyBF16Suppressed accepted an entirely suppressed vocabulary")
	}
}

func TestBF16NaNScanBytes_Good(t *testing.T) {
	values := append(toBF16Bytes([]float32{-2.5, 4.25}), []byte{0xc1, 0x7f}...)
	values = append(values, toBF16Bytes([]float32{7.5})...)
	values = append(values, []byte{0xa1, 0xff}...)

	count, first := bf16NaNScanBytes(values)
	if count != 2 || first != 2 {
		t.Fatalf("bf16NaNScanBytes = (%d, %d), want (2, 2)", count, first)
	}
}

func TestBF16NaNScanBytes_Bad(t *testing.T) {
	count, first := bf16NaNScanBytes(toBF16Bytes([]float32{-9.5, 0.25, 11.75}))
	if count != 0 || first != -1 {
		t.Fatalf("bf16NaNScanBytes finite input = (%d, %d), want (0, -1)", count, first)
	}
}

func TestBF16NaNScanBytes_Ugly(t *testing.T) {
	count, first := bf16NaNScanBytes([]byte{0xc1, 0x7f, 0xff})
	if count != 1 || first != 0 {
		t.Fatalf("bf16NaNScanBytes odd input = (%d, %d), want (1, 0)", count, first)
	}
}

func TestBF16BufStats_Good(t *testing.T) {
	if err := ensureInit(); err != nil {
		t.Skipf("device init: %v", err)
	}
	requireNativeRuntime(t)
	values := append(toBF16Bytes([]float32{-7.5, 2.25}), []byte{0x80, 0x7f}...)
	values = append(values, []byte{0xc1, 0x7f}...)
	values = append(values, toBF16Bytes([]float32{5.5})...)

	nan, inf, minV, maxV, firstNaN := bf16BufStats(sharedBytes(values), 0, 5)
	if nan != 1 || inf != 1 || minV != -7.5 || maxV != 5.5 || firstNaN != 3 {
		t.Fatalf("bf16BufStats = (%d, %d, %g, %g, %d), want (1, 1, -7.5, 5.5, 3)", nan, inf, minV, maxV, firstNaN)
	}
}

func TestBF16BufStats_Bad(t *testing.T) {
	if err := ensureInit(); err != nil {
		t.Skipf("device init: %v", err)
	}
	requireNativeRuntime(t)

	nan, inf, minV, maxV, firstNaN := bf16BufStats(nil, 0, 4)
	if nan != 0 || inf != 0 || minV != 0 || maxV != 0 || firstNaN != -1 {
		t.Fatalf("bf16BufStats nil buffer = (%d, %d, %g, %g, %d), want zero counts/range and -1", nan, inf, minV, maxV, firstNaN)
	}
}

func TestBF16BufStats_Ugly(t *testing.T) {
	if err := ensureInit(); err != nil {
		t.Skipf("device init: %v", err)
	}
	requireNativeRuntime(t)
	prefix := toBF16Bytes([]float32{99.5, -88.25})
	values := append(prefix, toBF16Bytes([]float32{3.5, -4.75, 1.25})...)

	nan, inf, minV, maxV, firstNaN := bf16BufStats(sharedBytes(values), uint(len(prefix)), 3)
	if nan != 0 || inf != 0 || minV != -4.75 || maxV != 3.5 || firstNaN != -1 {
		t.Fatalf("bf16BufStats offset range = (%d, %d, %g, %g, %d), want (0, 0, -4.75, 3.5, -1)", nan, inf, minV, maxV, firstNaN)
	}
}

func TestHeadEncoderSoftcapUsesBF16Kernel(t *testing.T) {
	requireNativeRuntime(t)

	const dModel, vocab = 1, 8
	const eps, softCap = float32(1e-6), float32(30)
	hidden := toBF16Bytes([]float32{1})
	finalNorm := toBF16Bytes([]float32{1})
	head := toBF16Bytes([]float32{-120, -30, -3, -0.5, 0.5, 3, 30, 120})
	h := &headEncoder{
		finalNorm: copyView(finalNorm),
		weight:    copyView(head),
		dModel:    dModel,
		vocab:     vocab,
		eps:       eps,
		softCap:   softCap,
	}

	raw, err := h.encode(hidden, true)
	if err != nil {
		t.Fatalf("headEncoder raw logits: %v", err)
	}
	scaled, err := MulBF16(raw, bf16ConstBytes(vocab, 1/softCap))
	if err != nil {
		t.Fatalf("scale logits: %v", err)
	}
	capped, err := TanhBF16(scaled)
	if err != nil {
		t.Fatalf("tanh logits: %v", err)
	}
	want, err := MulBF16(capped, bf16ConstBytes(vocab, softCap))
	if err != nil {
		t.Fatalf("restore logits: %v", err)
	}

	got, err := h.encode(hidden, false)
	if err != nil {
		t.Fatalf("headEncoder softcap logits: %v", err)
	}
	if !bytes.Equal(got, want) {
		t.Fatalf("headEncoder softcap = %v, want BF16-kernel softcap %v", bf16Floats(got), bf16Floats(want))
	}
}

func TestHeadEncoderSoftcapUsesScalarScaleBuffers(t *testing.T) {
	requireNativeRuntime(t)

	h := &headEncoder{vocab: 8192, softCap: 30}
	h.initSoftcapBuffers()
	if h.invSoftCapScale.buf == nil || h.softCapScale.buf == nil {
		t.Fatalf("softcap scalar buffers missing (inv=%v cap=%v)", h.invSoftCapScale.buf != nil, h.softCapScale.buf != nil)
	}
	if got := int(h.invSoftCapScale.buf.Length()); got != bf16Size {
		t.Fatalf("inverse softcap scale buffer length = %d, want scalar bf16 length %d", got, bf16Size)
	}
	if got := int(h.softCapScale.buf.Length()); got != bf16Size {
		t.Fatalf("softcap scale buffer length = %d, want scalar bf16 length %d", got, bf16Size)
	}
}

func TestHeadEncoderHiddenBufferUsesCallerBacking(t *testing.T) {
	requireNativeRuntime(t)

	h := &headEncoder{}
	hidden := toBF16Bytes(syntheticFloat32(64, 3))
	scratch, buf, err := h.hiddenBuffer(hidden)
	if err != nil {
		t.Fatalf("hiddenBuffer: %v", err)
	}
	defer h.putHiddenScratch(scratch)
	if got, want := uintptr(buf.Contents()), uintptr(unsafe.Pointer(&hidden[0])); got != want {
		t.Fatalf("hidden buffer pointer = %#x, want caller backing %#x", got, want)
	}
}

func TestHeadEncoderSoftcapEncodeIntoAllocationBudget(t *testing.T) {
	requireNativeRuntime(t)

	const dModel, vocab = 64, 2048
	const eps, softCap = float32(1e-6), float32(30)
	h := &headEncoder{
		finalNorm: copyView(toBF16Bytes(syntheticFloat32(dModel, 5))),
		weight:    copyView(toBF16Bytes(syntheticFloat32(vocab*dModel, 7))),
		dModel:    dModel,
		vocab:     vocab,
		eps:       eps,
		softCap:   softCap,
	}
	h.initSoftcapBuffers()
	hidden := toBF16Bytes(syntheticFloat32(dModel, 3))
	out := make([]byte, vocab*bf16Size)
	if _, err := h.encodeInto(hidden, false, out); err != nil {
		t.Fatalf("headEncoder softcap warmup: %v", err)
	}
	allocs := testing.AllocsPerRun(5, func() {
		if _, err := h.encodeInto(hidden, false, out); err != nil {
			t.Fatalf("headEncoder softcap encodeInto: %v", err)
		}
	})
	if allocs > 0 {
		t.Fatalf("headEncoder softcap encodeInto allocations = %.0f, want 0", allocs)
	}
}

func TestHeadEncoderTopKTokenBufferAllocationBudget(t *testing.T) {
	requireNativeRuntime(t)

	const dModel, vocab, groupSize, bits = 512, 4096, 64, 4
	const topK = 32
	h, hidden := quantHeadEncoderBenchFixture(dModel, vocab, groupSize, bits)
	params := model.SampleParams{Temperature: 1, TopK: topK}
	hiddenScratch, hiddenBuf, err := h.hiddenBuffer(hidden)
	if err != nil {
		t.Fatalf("hiddenBuffer: %v", err)
	}
	defer h.putHiddenScratch(hiddenScratch)
	sampler := model.NewSampler(123)
	if _, ok, err := h.sampleTopKTokenBufferInPool(hiddenBuf, params, sampler.Draw(), nil); err != nil {
		t.Fatalf("sampleTopKTokenBufferInPool warmup: %v", err)
	} else if !ok {
		t.Fatal("TopK token buffer sampler declined")
	}
	allocs := testing.AllocsPerRun(5, func() {
		if _, ok, err := h.sampleTopKTokenBufferInPool(hiddenBuf, params, sampler.Draw(), nil); err != nil {
			t.Fatalf("sampleTopKTokenBufferInPool: %v", err)
		} else if !ok {
			t.Fatal("TopK token buffer sampler declined")
		}
	})
	if allocs > 0 {
		t.Fatalf("TopK token buffer allocations = %.0f, want 0", allocs)
	}
}

func TestHeadEncoderTopKTokenPrefersFusedQ4Scratch(t *testing.T) {
	requireNativeRuntime(t)

	const dModel, vocab, groupSize, bits = 512, 4096, 64, 4
	const topK = 32
	if !q4LMHeadTopKUsable(dModel, vocab, groupSize, bits, topK) {
		t.Skip("fused q4 lm-head top-k custom kernel unavailable")
	}
	if !qmvLogitsTopKUsable(dModel, vocab, groupSize, bits, topK) {
		t.Skip("qmv logits top-k custom kernel unavailable")
	}
	if topKReduceSampleUsable(topK, vocab) {
		// #23: the reduce route (qmv logits + reduction pick) outranks the fused q4 arm — the
		// fused-scratch preference this test pins only applies when reduce is unavailable.
		t.Skip("reduce pick preferred over the fused q4 arm on this build")
	}
	h, hidden := quantHeadEncoderBenchFixture(dModel, vocab, groupSize, bits)
	params := model.SampleParams{Temperature: 1, TopK: topK}
	hiddenScratch, hiddenBuf, err := h.hiddenBuffer(hidden)
	if err != nil {
		t.Fatalf("hiddenBuffer: %v", err)
	}
	defer h.putHiddenScratch(hiddenScratch)
	sampler := model.NewSampler(123)
	if _, ok, err := h.sampleTopKTokenBufferInPool(hiddenBuf, params, sampler.Draw(), nil); err != nil {
		t.Fatalf("sampleTopKTokenBufferInPool: %v", err)
	} else if !ok {
		t.Fatal("TopK token buffer sampler declined")
	}

	v := h.topKScratch.Get()
	if v == nil {
		t.Fatal("top-k sampler did not return scratch to pool")
	}
	scratch := v.(*headTopKScratch)
	defer h.putTopKScratch(scratch)
	if scratch.logits != nil {
		t.Fatal("top-k sampler kept a full-vocab logits scratch; want fused q4 candidates only")
	}
}

func TestHeadEncoderSoftcapKeepsScalarBuffersResident(t *testing.T) {
	requireNativeRuntime(t)

	first := &headEncoder{vocab: 8192, softCap: 30}
	first.initSoftcapBuffers()
	second := &headEncoder{vocab: 8192, softCap: 30}
	second.initSoftcapBuffers()

	if first.invSoftCapScale.buf.GetID() != second.invSoftCapScale.buf.GetID() {
		t.Fatalf("inverse softcap buffer was not resident: first=%d second=%d", first.invSoftCapScale.buf.GetID(), second.invSoftCapScale.buf.GetID())
	}
	if first.softCapScale.buf.GetID() != second.softCapScale.buf.GetID() {
		t.Fatalf("softcap buffer was not resident: first=%d second=%d", first.softCapScale.buf.GetID(), second.softCapScale.buf.GetID())
	}
}

func TestHeadEncoderSoftcapEncodeFallbackUsesResidentScalars(t *testing.T) {
	requireNativeRuntime(t)
	resetResidentBufsForTest()
	defer resetResidentBufsForTest()

	const dModel, vocab = 8, 16
	const eps, softCap = float32(1e-5), float32(30)
	invKey := bf16ConstKey{n: 1, v: 1 / softCap}
	capKey := bf16ConstKey{n: 1, v: softCap}
	bf16ConstMu.Lock()
	delete(bf16ConstCache, invKey)
	delete(bf16ConstCache, capKey)
	bf16ConstMu.Unlock()

	h := &headEncoder{
		finalNorm: copyView(toBF16Bytes(syntheticFloat32(dModel, 5))),
		weight:    copyView(toBF16Bytes(syntheticFloat32(vocab*dModel, 7))),
		dModel:    dModel,
		vocab:     vocab,
		eps:       eps,
		softCap:   softCap,
	}
	hidden := toBF16Bytes(syntheticFloat32(dModel, 3))
	if _, err := h.encode(hidden, false); err != nil {
		t.Fatalf("headEncoder softcap fallback encode: %v", err)
	}

	bf16ConstMu.Lock()
	_, invCached := bf16ConstCache[invKey]
	_, capCached := bf16ConstCache[capKey]
	bf16ConstMu.Unlock()
	if !invCached || !capCached {
		t.Fatalf("headEncoder softcap fallback did not use resident scalar buffers (inv=%v cap=%v)", invCached, capCached)
	}
}

func TestHeadGreedyScratchKeepsPerTokenBuffersResident(t *testing.T) {
	requireNativeRuntime(t)

	s := newHeadGreedyScratch(3, 64, 17, true, true)
	if s.normed == nil {
		t.Fatal("greedy scratch did not retain the normed activation buffer")
	}
	if s.logits == nil {
		t.Fatal("quant greedy scratch did not retain the qmv logits buffer")
	}
	if got := int(s.normed.Length()); got != 64*bf16Size {
		t.Fatalf("normed scratch length = %d, want %d", got, 64*bf16Size)
	}
	if got := int(s.logits.Length()); got != 17*bf16Size {
		t.Fatalf("logits scratch length = %d, want %d", got, 17*bf16Size)
	}

	bf16 := newHeadGreedyScratch(3, 64, 17, false, false)
	if bf16.normed == nil {
		t.Fatal("BF16 greedy scratch did not retain the normed activation buffer")
	}
	if bf16.logits != nil {
		t.Fatal("BF16 greedy scratch allocated a quant logits buffer")
	}
}

func TestHeadGreedyScratchSeparatesLogitsAndSoftcapBuffers(t *testing.T) {
	requireNativeRuntime(t)

	h := &headEncoder{dModel: 64, vocab: 17}
	greedy := h.getGreedyScratch(3, true, false)
	if greedy.logits == nil {
		t.Fatal("quant greedy scratch did not retain logits buffer")
	}
	if greedy.softcapA != nil || greedy.softcapB != nil {
		t.Fatal("quant greedy scratch allocated unused softcap buffers")
	}
	h.putGreedyScratch(greedy)

	sampled := h.getGreedyScratch(3, true, true)
	defer h.putGreedyScratch(sampled)
	if sampled.logits == nil || sampled.softcapA == nil || sampled.softcapB == nil {
		t.Fatal("sampled logits scratch did not retain logits and softcap buffers")
	}
}

func TestHeadHiddenScratchReusesPinnedNoCopyBuffer(t *testing.T) {
	requireNativeRuntime(t)
	if _, err := bf16MulScalarPipeline(); err != nil {
		t.Skipf("custom scalar kernel unavailable: %v", err)
	}

	h := &headEncoder{}
	first := toBF16Bytes([]float32{1, 2, 3, 4})
	second := toBF16Bytes([]float32{5, 6, 7, 8})

	scratch, buf, err := h.hiddenBuffer(first)
	if err != nil {
		t.Fatalf("hiddenBuffer first: %v", err)
	}
	if scratch == nil || buf == nil {
		t.Fatal("hiddenBuffer first did not return pinned scratch")
	}
	firstID := buf.GetID()
	h.putHiddenScratch(scratch)

	scratch2, buf2, err := h.hiddenBuffer(second)
	if err != nil {
		t.Fatalf("hiddenBuffer second: %v", err)
	}
	if scratch2 == nil || buf2 == nil {
		t.Fatal("hiddenBuffer second did not return pinned scratch")
	}
	defer scratch2.Close()
	if scratch2 != scratch {
		t.Fatal("hiddenBuffer did not reuse the pooled hidden scratch")
	}
	if got := buf2.GetID(); got != firstID {
		t.Fatalf("hiddenBuffer Metal buffer id = %d, want reused id %d", got, firstID)
	}

	scalar := bf16ScalarBytes(1)
	out := scratchBF16(len(second) / bf16Size)
	cb := queue.CommandBuffer()
	enc := cb.ComputeCommandEncoder()
	if err := encMulScalarBF16(enc, buf2, sharedBytes(scalar[:]), out, 0, len(second)/bf16Size); err != nil {
		enc.EndEncoding()
		t.Fatalf("encMulScalarBF16: %v", err)
	}
	enc.EndEncoding()
	cb.Commit()
	cb.WaitUntilCompleted()

	got := append([]byte(nil), unsafe.Slice((*byte)(out.Contents()), len(second))...)
	if !bytes.Equal(got, second) {
		t.Fatalf("hiddenBuffer GPU read = %v, want second hidden %v", bf16Floats(got), bf16Floats(second))
	}
}

func TestHeadEncoderQuantGreedyMatchesFullLogits(t *testing.T) {
	requireNativeRuntime(t)

	const dModel, vocab, groupSize, bits = 64, 17, 32, 4
	const eps = float32(1e-6)
	hidden := toBF16Bytes(syntheticFloat32(dModel, 31))
	finalNorm := toBF16Bytes(syntheticFloat32(dModel, 37))

	packed := make([]byte, vocab*dModel*bits/8)
	for i := range packed {
		packed[i] = byte((i*29 + 17) & 0xff)
	}
	sidecars := vocab * (dModel / groupSize)
	scalesF, biasesF := make([]float32, sidecars), make([]float32, sidecars)
	for i := range scalesF {
		scalesF[i] = 0.015 + float32((i%7)+1)*0.002
		biasesF[i] = -0.08 + float32((i%11))*0.01
	}
	h := &headEncoder{
		finalNorm: copyView(finalNorm),
		weight:    copyView(packed),
		scales:    copyView(toBF16Bytes(scalesF)),
		biases:    copyView(toBF16Bytes(biasesF)),
		quant:     true,
		groupSize: groupSize,
		bits:      bits,
		dModel:    dModel,
		vocab:     vocab,
		eps:       eps,
	}

	logits, err := h.encode(hidden, true)
	if err != nil {
		t.Fatalf("headEncoder full logits: %v", err)
	}
	want, err := model.Greedy(logits, vocab)
	if err != nil {
		t.Fatalf("full-logits greedy: %v", err)
	}
	got, ok, err := h.greedy(hidden, nil)
	if err != nil {
		t.Fatalf("headEncoder direct greedy: %v", err)
	}
	if !ok {
		t.Fatal("headEncoder direct greedy declined quant head")
	}
	if got != want {
		t.Fatalf("headEncoder direct greedy = %d, want full-logits greedy %d", got, want)
	}
}

func TestHeadEncoderQuantGreedyInPoolMatchesFullLogits(t *testing.T) {
	requireNativeRuntime(t)

	const dModel, vocab, groupSize, bits = 64, 17, 32, 4
	const eps = float32(1e-6)
	hidden := toBF16Bytes(syntheticFloat32(dModel, 131))
	finalNorm := toBF16Bytes(syntheticFloat32(dModel, 137))

	packed := make([]byte, vocab*dModel*bits/8)
	for i := range packed {
		packed[i] = byte((i*31 + 19) & 0xff)
	}
	sidecars := vocab * (dModel / groupSize)
	scalesF, biasesF := make([]float32, sidecars), make([]float32, sidecars)
	for i := range scalesF {
		scalesF[i] = 0.013 + float32((i%5)+1)*0.003
		biasesF[i] = -0.06 + float32((i%13))*0.008
	}
	h := &headEncoder{
		finalNorm: copyView(finalNorm),
		weight:    copyView(packed),
		scales:    copyView(toBF16Bytes(scalesF)),
		biases:    copyView(toBF16Bytes(biasesF)),
		quant:     true,
		groupSize: groupSize,
		bits:      bits,
		dModel:    dModel,
		vocab:     vocab,
		eps:       eps,
	}

	logits, err := h.encode(hidden, true)
	if err != nil {
		t.Fatalf("headEncoder full logits: %v", err)
	}
	want, err := model.Greedy(logits, vocab)
	if err != nil {
		t.Fatalf("full-logits greedy: %v", err)
	}
	var got int32
	var ok bool
	var gotErr error
	withAutoreleasePool(func() {
		got, ok, gotErr = h.greedyInPool(hidden, nil)
	})
	if gotErr != nil {
		t.Fatalf("headEncoder direct greedy in pool: %v", gotErr)
	}
	if !ok {
		t.Fatal("headEncoder direct greedy in pool declined quant head")
	}
	if got != want {
		t.Fatalf("headEncoder direct greedy in pool = %d, want full-logits greedy %d", got, want)
	}
}

func TestHeadEncoderQuantGreedySuppressesIDs(t *testing.T) {
	requireNativeRuntime(t)

	const dModel, vocab, groupSize, bits = 64, 17, 32, 4
	const eps = float32(1e-6)
	hidden := toBF16Bytes(syntheticFloat32(dModel, 31))
	finalNorm := toBF16Bytes(syntheticFloat32(dModel, 37))

	packed := make([]byte, vocab*dModel*bits/8)
	for i := range packed {
		packed[i] = byte((i*29 + 17) & 0xff)
	}
	sidecars := vocab * (dModel / groupSize)
	scalesF, biasesF := make([]float32, sidecars), make([]float32, sidecars)
	for i := range scalesF {
		scalesF[i] = 0.015 + float32((i%7)+1)*0.002
		biasesF[i] = -0.08 + float32((i%11))*0.01
	}
	h := &headEncoder{
		finalNorm: copyView(finalNorm),
		weight:    copyView(packed),
		scales:    copyView(toBF16Bytes(scalesF)),
		biases:    copyView(toBF16Bytes(biasesF)),
		quant:     true,
		groupSize: groupSize,
		bits:      bits,
		dModel:    dModel,
		vocab:     vocab,
		eps:       eps,
	}

	logits, err := h.encode(hidden, true)
	if err != nil {
		t.Fatalf("headEncoder full logits: %v", err)
	}
	first, err := model.Greedy(logits, vocab)
	if err != nil {
		t.Fatalf("full-logits greedy: %v", err)
	}
	want, err := greedyBF16Suppressed(logits, vocab, []int32{first})
	if err != nil {
		t.Fatalf("suppressed full-logits greedy: %v", err)
	}
	got, ok, err := h.greedy(hidden, []int32{first})
	if err != nil {
		t.Fatalf("headEncoder suppressed direct greedy: %v", err)
	}
	if !ok {
		t.Fatal("headEncoder direct greedy declined quant head with suppression")
	}
	if got != want {
		t.Fatalf("headEncoder suppressed direct greedy = %d, want full-logits suppressed greedy %d", got, want)
	}
}

func TestHeadEncoderBF16GreedyMatchesFullLogits(t *testing.T) {
	requireNativeRuntime(t)

	const dModel, vocab = 64, 19
	const eps = float32(1e-6)
	hidden := toBF16Bytes(syntheticFloat32(dModel, 51))
	h := &headEncoder{
		finalNorm: copyView(toBF16Bytes(syntheticFloat32(dModel, 53))),
		weight:    copyView(toBF16Bytes(syntheticFloat32(vocab*dModel, 57))),
		dModel:    dModel,
		vocab:     vocab,
		eps:       eps,
	}

	logits, err := h.encode(hidden, true)
	if err != nil {
		t.Fatalf("headEncoder full logits: %v", err)
	}
	want, err := model.Greedy(logits, vocab)
	if err != nil {
		t.Fatalf("full-logits greedy: %v", err)
	}
	got, ok, err := h.greedy(hidden, nil)
	if err != nil {
		t.Fatalf("headEncoder direct bf16 greedy: %v", err)
	}
	if !ok {
		t.Fatal("headEncoder direct greedy declined BF16 head")
	}
	if got != want {
		t.Fatalf("headEncoder direct bf16 greedy = %d, want full-logits greedy %d", got, want)
	}
}

func TestHeadEncoderBF16GreedyAtOffsetMatchesFullLogits(t *testing.T) {
	requireNativeRuntime(t)

	const dModel, vocab = 64, 19
	const eps = float32(1e-6)
	hidden0 := toBF16Bytes(syntheticFloat32(dModel, 51))
	hidden1 := toBF16Bytes(syntheticFloat32(dModel, 73))
	packed := append(append([]byte(nil), hidden0...), hidden1...)
	h := &headEncoder{
		finalNorm: copyView(toBF16Bytes(syntheticFloat32(dModel, 53))),
		weight:    copyView(toBF16Bytes(syntheticFloat32(vocab*dModel, 57))),
		dModel:    dModel,
		vocab:     vocab,
		eps:       eps,
	}

	logits, err := h.encode(hidden1, true)
	if err != nil {
		t.Fatalf("headEncoder full logits: %v", err)
	}
	want, err := model.Greedy(logits, vocab)
	if err != nil {
		t.Fatalf("full-logits greedy: %v", err)
	}

	var got int32
	var ok bool
	var gotErr error
	withAutoreleasePool(func() {
		packedBuf := sharedBytes(packed)
		cb := queue.CommandBuffer()
		enc := cb.ComputeCommandEncoder()
		var scratch *headGreedyScratch
		scratch, ok, gotErr = h.encodeGreedyAt(enc, packedBuf, uint(len(hidden0)), nil)
		if !ok || gotErr != nil {
			enc.EndEncoding()
			if scratch != nil {
				h.putGreedyScratch(scratch)
			}
			return
		}
		enc.EndEncoding()
		cb.Commit()
		cb.WaitUntilCompleted()
		got = scratch.token()
		h.putGreedyScratch(scratch)
	})
	if gotErr != nil {
		t.Fatalf("headEncoder direct bf16 greedy at offset: %v", gotErr)
	}
	if !ok {
		t.Fatal("headEncoder direct greedy at offset declined BF16 head")
	}
	if got != want {
		t.Fatalf("headEncoder direct bf16 greedy at offset = %d, want full-logits greedy %d", got, want)
	}
}

func TestHeadEncoderBF16TopKTokenAtOffsetMatchesStandalone(t *testing.T) {
	requireNativeRuntime(t)

	const dModel, vocab = 64, 19
	const eps = float32(1e-6)
	hidden0 := toBF16Bytes(syntheticFloat32(dModel, 51))
	hidden1 := toBF16Bytes(syntheticFloat32(dModel, 73))
	packed := append(append([]byte(nil), hidden0...), hidden1...)
	h := &headEncoder{
		finalNorm: copyView(toBF16Bytes(syntheticFloat32(dModel, 53))),
		weight:    copyView(toBF16Bytes(syntheticFloat32(vocab*dModel, 57))),
		dModel:    dModel,
		vocab:     vocab,
		eps:       eps,
	}
	params := model.SampleParams{
		Temperature:    0.8,
		TopK:           7,
		TopP:           0.75,
		MinP:           0.01,
		RepeatPenalty:  1.2,
		SuppressTokens: []int32{2, 7},
	}
	history := []int32{3, 5, 8}
	draw := model.NewSampler(123).Draw()
	want, ok, err := h.sampleTopKTokenInPool(hidden1, params, draw, history)
	if err != nil {
		t.Fatalf("headEncoder standalone top-k sample: %v", err)
	}
	if !ok {
		t.Fatal("headEncoder standalone top-k sample declined BF16 head")
	}

	var got int32
	withAutoreleasePool(func() {
		got, ok, err = h.sampleTopKTokenBufferAtInPool(sharedBytes(packed), uint(len(hidden0)), params, draw, history)
	})
	if err != nil {
		t.Fatalf("headEncoder top-k sample at offset: %v", err)
	}
	if !ok {
		t.Fatal("headEncoder top-k sample at offset declined BF16 head")
	}
	if got != want {
		t.Fatalf("headEncoder top-k sample at offset = %d, want standalone sample %d", got, want)
	}
}

func TestHeadEncoderBF16LogitsSampleAtOffsetMatchesStandalone(t *testing.T) {
	requireNativeRuntime(t)

	const dModel, vocab = 64, 19
	const eps = float32(1e-6)
	hidden0 := toBF16Bytes(syntheticFloat32(dModel, 51))
	hidden1 := toBF16Bytes(syntheticFloat32(dModel, 73))
	packed := append(append([]byte(nil), hidden0...), hidden1...)
	h := &headEncoder{
		finalNorm: copyView(toBF16Bytes(syntheticFloat32(dModel, 53))),
		weight:    copyView(toBF16Bytes(syntheticFloat32(vocab*dModel, 57))),
		dModel:    dModel,
		vocab:     vocab,
		eps:       eps,
	}
	params := model.SampleParams{
		Temperature:    0.8,
		TopP:           0.85,
		MinP:           0.01,
		RepeatPenalty:  1.2,
		SuppressTokens: []int32{2, 7},
	}
	history := []int32{3, 5, 8}
	draw := model.NewSampler(123).Draw()
	want, ok, err := h.sampleLogitsTokenInPool(hidden1, params, draw, history)
	if err != nil {
		t.Fatalf("headEncoder standalone logits sample: %v", err)
	}
	if !ok {
		t.Fatal("headEncoder standalone logits sample declined BF16 head")
	}

	var got int32
	withAutoreleasePool(func() {
		got, ok, err = h.sampleLogitsTokenBufferAtInPool(sharedBytes(packed), uint(len(hidden0)), params, draw, history)
	})
	if err != nil {
		t.Fatalf("headEncoder logits sample at offset: %v", err)
	}
	if !ok {
		t.Fatal("headEncoder logits sample at offset declined BF16 head")
	}
	if got != want {
		t.Fatalf("headEncoder logits sample at offset = %d, want standalone sample %d", got, want)
	}
}

func TestHeadEncoderBF16GreedySuppressesIDs(t *testing.T) {
	requireNativeRuntime(t)

	const dModel, vocab = 1, 8
	const eps = float32(1e-6)
	hidden := toBF16Bytes([]float32{1})
	h := &headEncoder{
		finalNorm: copyView(toBF16Bytes([]float32{1})),
		weight:    copyView(toBF16Bytes([]float32{-4, -2, -1, 0, 1, 2, 4, 8})),
		dModel:    dModel,
		vocab:     vocab,
		eps:       eps,
	}

	logits, err := h.encode(hidden, true)
	if err != nil {
		t.Fatalf("headEncoder full logits: %v", err)
	}
	first, err := model.Greedy(logits, vocab)
	if err != nil {
		t.Fatalf("full-logits greedy: %v", err)
	}
	if first != 7 {
		t.Fatalf("fixture top token = %d, want 7", first)
	}
	want, err := greedyBF16Suppressed(logits, vocab, []int32{first})
	if err != nil {
		t.Fatalf("suppressed full-logits greedy: %v", err)
	}
	if want != 6 {
		t.Fatalf("fixture suppressed token = %d, want 6", want)
	}
	got, ok, err := h.greedy(hidden, []int32{first})
	if err != nil {
		t.Fatalf("headEncoder suppressed direct bf16 greedy: %v", err)
	}
	if !ok {
		t.Fatal("headEncoder direct greedy declined BF16 head with suppression")
	}
	if got != want {
		t.Fatalf("headEncoder suppressed direct bf16 greedy = %d, want full-logits suppressed greedy %d", got, want)
	}
}

func covSampleHeadFixture(t *testing.T) (*headEncoder, []byte, []byte) {
	t.Helper()
	const dModel, vocab = 64, 19
	h := &headEncoder{
		finalNorm: copyView(toBF16Bytes(syntheticFloat32(dModel, 53))),
		weight:    copyView(toBF16Bytes(syntheticFloat32(vocab*dModel, 57))),
		dModel:    dModel,
		vocab:     vocab,
		eps:       1e-6,
	}
	hidden := toBF16Bytes(syntheticFloat32(dModel, 73))
	full, err := h.encode(hidden, true)
	if err != nil {
		t.Fatalf("head fixture encode: %v", err)
	}
	return h, hidden, full
}

func covAssertCandidateSet(t *testing.T, gotLogits []byte, gotIDs []int32, full []byte, vocab, topK int, suppress []int32) {
	t.Helper()
	vals := make([]byte, topK*bf16Size)
	ids := make([]int32, topK)
	wantLogits, wantIDs := hostTopKCandidatesBF16(full, vocab, topK, suppress, vals, ids)
	want := make(map[int32][]byte, len(wantIDs))
	for i, id := range wantIDs {
		want[id] = append([]byte(nil), wantLogits[i*bf16Size:(i+1)*bf16Size]...)
	}
	if len(gotIDs) != len(want) {
		t.Fatalf("candidate count = %d, want %d (got ids %v want %v)", len(gotIDs), len(want), gotIDs, wantIDs)
	}
	for i, id := range gotIDs {
		wantBytes, ok := want[id]
		if !ok {
			t.Fatalf("candidate id[%d] = %d, want one of %v", i, id, wantIDs)
		}
		if !bytes.Equal(gotLogits[i*bf16Size:(i+1)*bf16Size], wantBytes) {
			t.Fatalf("candidate logits for id %d = %v, want %v", id, gotLogits[i*bf16Size:(i+1)*bf16Size], wantBytes)
		}
		delete(want, id)
	}
}

func TestHeadEncoderArgmaxInvalidDiagReportsStages(t *testing.T) {
	requireNativeRuntime(t)
	h := &headEncoder{quant: true, dModel: 2, vocab: 5}
	scratch := newHeadGreedyScratch(1, h.dModel, h.vocab, true, false)
	if scratch == nil || scratch.normed == nil || scratch.logits == nil || scratch.tileValues == nil || scratch.tileIndices == nil {
		t.Fatal("newHeadGreedyScratch did not allocate diagnostic buffers")
	}
	defer h.putGreedyScratch(scratch)
	unsafe.Slice((*float32)(scratch.tileValues.Contents()), 1)[0] = 4.5
	unsafe.Slice((*int32)(scratch.tileIndices.Contents()), 1)[0] = 3
	diag := h.argmaxInvalidDiag(scratch, sharedBytes(toBF16Bytes([]float32{1, 2})), 0)
	for _, want := range []string{"hidden NaN=0", "normed NaN=0", "logits NaN=0", "tiles=1", "tile0=(4.5,3)"} {
		if !strings.Contains(diag, want) {
			t.Fatalf("argmaxInvalidDiag = %q, missing %q", diag, want)
		}
	}
}

func TestHeadEncoderSampleLogitsTokenMatchesHostReference(t *testing.T) {
	requireNativeRuntime(t)
	h, hidden, full := covSampleHeadFixture(t)
	params := model.SampleParams{Temperature: 0.8, MinP: 0.01, SuppressTokens: []int32{2, 7}}
	draw := model.NewSampler(123).Draw()
	want, err := sampleBF16VocabOrderForTest(full, h.vocab, params, draw, nil)
	if err != nil {
		t.Fatalf("host sample: %v", err)
	}
	got, ok, err := h.sampleLogitsToken(hidden, params, draw, nil)
	if err != nil {
		t.Fatalf("sampleLogitsToken: %v", err)
	}
	if !ok {
		t.Fatal("sampleLogitsToken declined the BF16 fixture")
	}
	if got != want {
		t.Fatalf("sampleLogitsToken = %d, want host reference %d", got, want)
	}
}

func TestHeadEncoderEncodeLogitsSampleAtMatchesHostReference(t *testing.T) {
	requireNativeRuntime(t)
	h, hidden, full := covSampleHeadFixture(t)
	params := model.SampleParams{Temperature: 0.8, MinP: 0.01, SuppressTokens: []int32{2, 7}}
	draw := model.NewSampler(123).Draw()
	want, err := sampleBF16VocabOrderForTest(full, h.vocab, params, draw, nil)
	if err != nil {
		t.Fatalf("host sample: %v", err)
	}
	cb := queue.CommandBuffer()
	enc := cb.ComputeCommandEncoder()
	scratch, ok, err := h.encodeLogitsSampleAt(enc, sharedBytes(hidden), 0, params, draw, nil)
	if err != nil || !ok {
		enc.EndEncoding()
		if scratch != nil {
			h.putGreedyScratch(scratch)
		}
		t.Fatalf("encodeLogitsSampleAt ok=%v err=%v", ok, err)
	}
	enc.EndEncoding()
	cb.Commit()
	cb.WaitUntilCompleted()
	got := scratch.token()
	h.putGreedyScratch(scratch)
	if got != want {
		t.Fatalf("encodeLogitsSampleAt = %d, want host reference %d", got, want)
	}
}

func TestHeadEncoderEncodeLogitsSampleObjectMatchesHostReference(t *testing.T) {
	requireNativeRuntime(t)
	h, hidden, full := covSampleHeadFixture(t)
	params := model.SampleParams{Temperature: 0.8, MinP: 0.01, SuppressTokens: []int32{2, 7}}
	draw := model.NewSampler(123).Draw()
	want, err := sampleBF16VocabOrderForTest(full, h.vocab, params, draw, nil)
	if err != nil {
		t.Fatalf("host sample: %v", err)
	}
	cb := commandBufferFast(queue)
	enc := computeCommandEncoderFast(cb)
	scratch, ok, err := h.encodeLogitsSampleObject(enc, sharedBytes(hidden), params, draw, nil)
	if err != nil || !ok {
		endEncodingFast(enc)
		if scratch != nil {
			h.putGreedyScratch(scratch)
		}
		t.Fatalf("encodeLogitsSampleObject ok=%v err=%v", ok, err)
	}
	endEncodingFast(enc)
	commitCommandBufferFast(cb)
	waitUntilCompletedFast(cb)
	got := scratch.token()
	h.putGreedyScratch(scratch)
	if got != want {
		t.Fatalf("encodeLogitsSampleObject = %d, want host reference %d", got, want)
	}
}

func TestHeadEncoderSampleTopKTokenReturnsHostArgmaxAtZeroDraw(t *testing.T) {
	requireNativeRuntime(t)
	h, hidden, full := covSampleHeadFixture(t)
	params := model.SampleParams{Temperature: 1, TopK: 5, TopP: 1, SuppressTokens: []int32{2, 7}}
	vals := make([]byte, params.TopK*bf16Size)
	ids := make([]int32, params.TopK)
	vals, ids = hostTopKCandidatesBF16(full, h.vocab, params.TopK, params.SuppressTokens, vals, ids)
	var want int32 = -1
	var best float32
	for i, id := range ids {
		v := bf16ToF32(vals[i*bf16Size], vals[i*bf16Size+1])
		if want < 0 || v > best {
			want, best = id, v
		}
	}
	got, ok, err := h.sampleTopKToken(hidden, params, 0, nil)
	if err != nil {
		t.Fatalf("sampleTopKToken: %v", err)
	}
	if !ok {
		t.Fatal("sampleTopKToken declined the BF16 fixture")
	}
	if got != want {
		t.Fatalf("sampleTopKToken = %d, want top candidate %d", got, want)
	}
}

func TestHeadEncoderSampleTopKCandidatesBufferIntoMatchesHostSort(t *testing.T) {
	requireNativeRuntime(t)
	h, hidden, full := covSampleHeadFixture(t)
	const topK = 5
	suppress := []int32{2, 7}
	outLogits := make([]byte, topK*bf16Size)
	outIDs := make([]int32, 0, topK)
	gotLogits, gotIDs, ok, err := h.sampleTopKCandidatesBufferInto(sharedBytes(hidden), topK, suppress, outLogits, outIDs, false)
	if err != nil {
		t.Fatalf("sampleTopKCandidatesBufferInto: %v", err)
	}
	if !ok {
		t.Fatal("sampleTopKCandidatesBufferInto declined the BF16 fixture")
	}
	covAssertCandidateSet(t, gotLogits, gotIDs, full, h.vocab, topK, suppress)
}

func TestHeadEncoderEncodeTopKSampleMatchesHostArgmax(t *testing.T) {
	requireNativeRuntime(t)
	h, hidden, full := covSampleHeadFixture(t)
	params := model.SampleParams{Temperature: 1, TopK: 5, TopP: 1, SuppressTokens: []int32{2, 7}}
	vals := make([]byte, params.TopK*bf16Size)
	ids := make([]int32, params.TopK)
	vals, ids = hostTopKCandidatesBF16(full, h.vocab, params.TopK, params.SuppressTokens, vals, ids)
	var want int32 = -1
	var best float32
	for i, id := range ids {
		v := bf16ToF32(vals[i*bf16Size], vals[i*bf16Size+1])
		if want < 0 || v > best {
			want, best = id, v
		}
	}
	cb := queue.CommandBuffer()
	enc := cb.ComputeCommandEncoder()
	scratch, ok, err := h.encodeTopKSample(enc, sharedBytes(hidden), params, 0, nil, false)
	if err != nil || !ok {
		enc.EndEncoding()
		if scratch != nil {
			h.putTopKScratch(scratch)
		}
		t.Fatalf("encodeTopKSample ok=%v err=%v", ok, err)
	}
	enc.EndEncoding()
	cb.Commit()
	cb.WaitUntilCompleted()
	got := scratch.token()
	h.putTopKScratch(scratch)
	if got != want {
		t.Fatalf("encodeTopKSample = %d, want top candidate %d", got, want)
	}
}

func TestHeadEncoderEncodeTopKSampleObjectMatchesHostArgmax(t *testing.T) {
	requireNativeRuntime(t)
	h, hidden, full := covSampleHeadFixture(t)
	params := model.SampleParams{Temperature: 1, TopK: 5, TopP: 1, SuppressTokens: []int32{2, 7}}
	vals := make([]byte, params.TopK*bf16Size)
	ids := make([]int32, params.TopK)
	vals, ids = hostTopKCandidatesBF16(full, h.vocab, params.TopK, params.SuppressTokens, vals, ids)
	var want int32 = -1
	var best float32
	for i, id := range ids {
		v := bf16ToF32(vals[i*bf16Size], vals[i*bf16Size+1])
		if want < 0 || v > best {
			want, best = id, v
		}
	}
	cb := commandBufferFast(queue)
	enc := computeCommandEncoderFast(cb)
	scratch, ok, err := h.encodeTopKSampleObject(enc, sharedBytes(hidden), params, 0, nil, false)
	if err != nil || !ok {
		endEncodingFast(enc)
		if scratch != nil {
			h.putTopKScratch(scratch)
		}
		t.Fatalf("encodeTopKSampleObject ok=%v err=%v", ok, err)
	}
	endEncodingFast(enc)
	commitCommandBufferFast(cb)
	waitUntilCompletedFast(cb)
	got := scratch.token()
	h.putTopKScratch(scratch)
	if got != want {
		t.Fatalf("encodeTopKSampleObject = %d, want top candidate %d", got, want)
	}
}

func TestHeadEncoderEncodeTopKCandidatesReadsHostSortedCandidates(t *testing.T) {
	requireNativeRuntime(t)
	h, hidden, full := covSampleHeadFixture(t)
	const topK = 5
	cb := queue.CommandBuffer()
	enc := cb.ComputeCommandEncoder()
	scratch, ok, err := h.encodeTopKCandidates(enc, sharedBytes(hidden), topK, []int32{2, 7}, false)
	if err != nil || !ok {
		enc.EndEncoding()
		if scratch != nil {
			h.putTopKScratch(scratch)
		}
		t.Fatalf("encodeTopKCandidates ok=%v err=%v", ok, err)
	}
	enc.EndEncoding()
	cb.Commit()
	cb.WaitUntilCompleted()
	gotLogits, gotIDs, ok, err := h.readTopKCandidates(scratch, topK)
	h.putTopKScratch(scratch)
	if err != nil || !ok {
		t.Fatalf("readTopKCandidates ok=%v err=%v", ok, err)
	}
	covAssertCandidateSet(t, gotLogits, gotIDs, full, h.vocab, topK, []int32{2, 7})
}

func TestHeadEncoderEncodeTopKCandidatesWithHistoryMatchesPenalizedHostSort(t *testing.T) {
	requireNativeRuntime(t)
	h, hidden, full := covSampleHeadFixture(t)
	const topK = 5
	history := []int32{3, 5, 8}
	penalized, err := nativeApplyRepeatPenaltyBF16(full, h.vocab, history, 1.4)
	if err != nil {
		t.Fatalf("nativeApplyRepeatPenaltyBF16: %v", err)
	}
	cb := queue.CommandBuffer()
	enc := cb.ComputeCommandEncoder()
	scratch, ok, err := h.encodeTopKCandidatesWithHistory(enc, sharedBytes(hidden), topK, []int32{2, 7}, history, 1.4, false)
	if err != nil || !ok {
		enc.EndEncoding()
		if scratch != nil {
			h.putTopKScratch(scratch)
		}
		t.Fatalf("encodeTopKCandidatesWithHistory ok=%v err=%v", ok, err)
	}
	enc.EndEncoding()
	cb.Commit()
	cb.WaitUntilCompleted()
	gotLogits, gotIDs, ok, err := h.readTopKCandidates(scratch, topK)
	h.putTopKScratch(scratch)
	if err != nil || !ok {
		t.Fatalf("readTopKCandidates ok=%v err=%v", ok, err)
	}
	covAssertCandidateSet(t, gotLogits, gotIDs, penalized, h.vocab, topK, []int32{2, 7})
}

func TestHeadEncoderEncodeTopKCandidateRowsMatchesFullLogits(t *testing.T) {
	requireNativeRuntime(t)
	h, hidden, full := covSampleHeadFixture(t)
	const topK = 5
	cb := queue.CommandBuffer()
	enc := cb.ComputeCommandEncoder()
	scratch, candidateCount, ok, err := h.encodeTopKCandidateRows(enc, sharedBytes(hidden), topK, []int32{2, 7}, nil, 1, false, false)
	if err != nil || !ok {
		enc.EndEncoding()
		if scratch != nil {
			h.putTopKScratch(scratch)
		}
		t.Fatalf("encodeTopKCandidateRows ok=%v err=%v", ok, err)
	}
	enc.EndEncoding()
	cb.Commit()
	cb.WaitUntilCompleted()
	values := unsafe.Slice((*float32)(scratch.candidateValues.Contents()), candidateCount)
	ids := unsafe.Slice((*int32)(scratch.candidateIndices.Contents()), candidateCount)
	for i := 0; i < h.vocab; i++ {
		if i == 2 || i == 7 {
			if ids[i] != -1 {
				t.Fatalf("candidate row %d id = %d, want suppressed -1", i, ids[i])
			}
			continue
		}
		if ids[i] != int32(i) {
			t.Fatalf("candidate row %d id = %d, want %d", i, ids[i], i)
		}
		want := bf16ToF32(full[i*bf16Size], full[i*bf16Size+1])
		if d := values[i] - want; d < -0.08 || d > 0.08 {
			t.Fatalf("candidate row %d value = %v, want %v", i, values[i], want)
		}
	}
	h.putTopKScratch(scratch)
}

// TestGreedyRowsQuantFusedMatchesPerRow gates the quant K-row fused greedy
// head (#359 rows-head): one qmm_t weight pass + per-row logits argmax must
// pick the same tokens as the per-row lane (K full qmv sweeps + argmax) on a
// synthetic 4-bit head. qmv and qmm_t accumulate in different orders, so the
// gate is token equality over well-separated random logits — the same
// token-identity tier the MTP verify fold already runs on.
func TestGreedyRowsQuantFusedMatchesPerRow(t *testing.T) {
	requireNativeRuntime(t)

	const dModel, vocab, gs, bits, k = 64, 512, 64, 4, 5
	groups := dModel / gs
	packed := make([]byte, vocab*dModel/2)
	rnd := syntheticFloat32(len(packed), 401)
	for i := range packed {
		packed[i] = byte(int(rnd[i]*127+128) & 0xFF)
	}
	scales := toBF16Bytes(syntheticFloat32(vocab*groups, 403))
	biases := toBF16Bytes(syntheticFloat32(vocab*groups, 407))
	normW := toBF16Bytes(syntheticFloat32(dModel, 409))

	h, err := newHeadEncoder(nil, normW, packed, scales, biases, dModel, vocab, gs, bits, 1e-6, 0, true)
	if err != nil || h == nil {
		t.Fatalf("newHeadEncoder: h=%v err=%v", h, err)
	}
	rows := toBF16Bytes(syntheticFloat32(k*dModel, 411))
	rowsBuf := sharedBytes(rows)
	if rowsBuf == nil {
		t.Fatal("rowsBuf alloc failed")
	}

	want := make([]int32, k)
	for i := range k {
		tok, ok, gerr := h.greedyBufferAtInPool(rowsBuf, uint(i*dModel*bf16Size), nil)
		if gerr != nil || !ok {
			t.Fatalf("per-row greedy %d: ok=%v err=%v", i, ok, gerr)
		}
		want[i] = tok
	}

	got := make([]int32, k)
	handled, err := h.greedyRowsQuantFusedInPool(rowsBuf, k, got)
	if err != nil {
		t.Fatalf("greedyRowsQuantFusedInPool: %v", err)
	}
	if !handled {
		t.Fatal("greedyRowsQuantFusedInPool declined — qmm pipeline or scratch unavailable")
	}
	for i := range k {
		if got[i] != want[i] {
			t.Fatalf("row %d: fused quant head token = %d, per-row lane = %d", i, got[i], want[i])
		}
	}

	// the public entry must route the quant fused path for this geometry.
	viaEntry := make([]int32, k)
	ok, err := h.greedyRowsBufferInPool(rowsBuf, uint(dModel*bf16Size), k, nil, viaEntry)
	if err != nil || !ok {
		t.Fatalf("greedyRowsBufferInPool: ok=%v err=%v", ok, err)
	}
	for i := range k {
		if viaEntry[i] != want[i] {
			t.Fatalf("row %d: greedyRowsBufferInPool token = %d, per-row lane = %d", i, viaEntry[i], want[i])
		}
	}
}
