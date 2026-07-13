// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"bytes"
	"errors"
	"fmt"
	"math"
	"os"
	"runtime"
	"slices"
	"sort"
	"testing"
	"unsafe"

	"dappco.re/go/inference/model"
	g4 "dappco.re/go/inference/model/gemma4"
	"github.com/tmc/apple/metal"
)

func idsEqual(a, b []int32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func TestArchSessionAppendKnownResidentIDs_Good(t *testing.T) {
	sess := &ArchSession{cachedIDs: []int32{3, 5, 8, 13}}

	sess.appendKnownResidentIDs(2, []int32{21, 34}, []int32{55, 89})

	want := []int32{3, 5, 21, 34, 55, 89}
	if !idsEqual(sess.cachedIDs, want) {
		t.Fatalf("cached IDs = %v, want %v", sess.cachedIDs, want)
	}
}

func TestArchSessionAppendKnownResidentIDs_Bad(t *testing.T) {
	var sess *ArchSession

	sess.appendKnownResidentIDs(0, []int32{2, 3}, []int32{5})
}

func TestArchSessionAppendKnownResidentIDs_Ugly(t *testing.T) {
	for _, startPos := range []int{-1, 4} {
		sess := &ArchSession{cachedIDs: []int32{7, 11, 13}}

		sess.appendKnownResidentIDs(startPos, []int32{17}, []int32{19})

		if sess.cachedIDs != nil {
			t.Fatalf("start position %d left cached IDs %v, want nil", startPos, sess.cachedIDs)
		}
	}
}

func TestArchSessionMTPProjectionScratch_Good(t *testing.T) {
	sess := &ArchSession{}
	first := sess.mtpProjectionScratch(5)
	copy(first, []byte{2, 3, 5, 7, 11})

	second := sess.mtpProjectionScratch(3)

	if !bytes.Equal(second, []byte{2, 3, 5}) {
		t.Fatalf("reused projection scratch = %v, want [2 3 5]", second)
	}
	if &second[0] != &first[0] {
		t.Fatal("projection scratch did not reuse its backing allocation")
	}
}

func TestArchSessionMTPProjectionScratch_Bad(t *testing.T) {
	sess := &ArchSession{}

	got := sess.mtpProjectionScratch(0)

	if got != nil {
		t.Fatalf("zero projection scratch = %#v, want nil", got)
	}
}

func TestArchSessionMTPProjectionScratch_Ugly(t *testing.T) {
	sess := &ArchSession{}
	first := sess.mtpProjectionScratch(2)
	copy(first, []byte{13, 17})

	second := sess.mtpProjectionScratch(9)

	if len(second) != 9 || cap(second) < 9 {
		t.Fatalf("grown projection scratch len/cap = %d/%d, want 9/>=9", len(second), cap(second))
	}
	if !bytes.Equal(second[:2], []byte{0, 0}) {
		t.Fatalf("grown projection scratch retained stale bytes: %v", second[:2])
	}
}

func TestArchSessionMTPDraftTokenScratch_Good(t *testing.T) {
	sess := &ArchSession{}
	first := sess.mtpDraftTokenScratch(4)
	first = append(first, 2, 3, 5, 7)

	second := sess.mtpDraftTokenScratch(3)
	second = append(second, 11, 13, 17)

	if !idsEqual(second, []int32{11, 13, 17}) {
		t.Fatalf("reused draft token scratch = %v, want [11 13 17]", second)
	}
	if &second[0] != &first[0] {
		t.Fatal("draft token scratch did not reuse its backing allocation")
	}
}

func TestArchSessionMTPDraftTokenScratch_Bad(t *testing.T) {
	sess := &ArchSession{}

	got := sess.mtpDraftTokenScratch(0)

	if got != nil {
		t.Fatalf("zero draft token scratch = %#v, want nil", got)
	}
}

func TestArchSessionMTPDraftTokenScratch_Ugly(t *testing.T) {
	sess := &ArchSession{}
	first := sess.mtpDraftTokenScratch(2)
	first = append(first, 19, 23)

	second := sess.mtpDraftTokenScratch(7)

	if len(second) != 0 || cap(second) < 7 {
		t.Fatalf("grown draft token scratch len/cap = %d/%d, want 0/>=7", len(second), cap(second))
	}
	second = append(second, 29, 31, 37)
	if !idsEqual(second, []int32{29, 31, 37}) {
		t.Fatalf("grown draft token scratch = %v, want [29 31 37]", second)
	}
}

func TestArchSessionMTPDraftScratch_Good(t *testing.T) {
	sess := &ArchSession{}
	var slot []byte
	first := sess.mtpDraftScratch(&slot, 5)
	copy(first, []byte{31, 37, 41, 43, 47})

	second := sess.mtpDraftScratch(&slot, 3)

	if !bytes.Equal(second, []byte{31, 37, 41}) {
		t.Fatalf("reused draft scratch = %v, want [31 37 41]", second)
	}
	if &second[0] != &first[0] {
		t.Fatal("draft scratch did not reuse its backing allocation")
	}
}

func TestArchSessionMTPDraftScratch_Bad(t *testing.T) {
	sess := &ArchSession{}
	var slot []byte

	got := sess.mtpDraftScratch(&slot, 0)

	if got != nil {
		t.Fatalf("zero draft scratch = %#v, want nil", got)
	}
}

func TestArchSessionMTPDraftScratch_Ugly(t *testing.T) {
	sess := &ArchSession{}
	slot := []byte{53, 59}

	got := sess.mtpDraftScratch(&slot, 7)

	if len(got) != 7 || cap(got) < 7 {
		t.Fatalf("grown draft scratch len/cap = %d/%d, want 7/>=7", len(got), cap(got))
	}
	if !bytes.Equal(got[:2], []byte{0, 0}) {
		t.Fatalf("grown draft scratch retained stale bytes: %v", got[:2])
	}
}

func TestArchSessionMTPDraftLogitScoreScratch_Good(t *testing.T) {
	sess := &ArchSession{}
	first := sess.mtpDraftLogitScoreScratch(4)
	copy(first, []float32{0.125, 0.25, 0.5, 1})

	second := sess.mtpDraftLogitScoreScratch(2)

	if !slices.Equal(second, []float32{0.125, 0.25}) {
		t.Fatalf("reused draft logit scores = %v, want [0.125 0.25]", second)
	}
	if &second[0] != &first[0] {
		t.Fatal("draft logit score scratch did not reuse its backing allocation")
	}
}

func TestArchSessionMTPDraftLogitScoreScratch_Bad(t *testing.T) {
	sess := &ArchSession{}

	got := sess.mtpDraftLogitScoreScratch(0)

	if got != nil {
		t.Fatalf("zero draft logit score scratch = %#v, want nil", got)
	}
}

func TestArchSessionMTPDraftLogitScoreScratch_Ugly(t *testing.T) {
	sess := &ArchSession{}
	first := sess.mtpDraftLogitScoreScratch(2)
	copy(first, []float32{-0.75, 1.25})

	second := sess.mtpDraftLogitScoreScratch(8)

	if len(second) != 8 || cap(second) < 8 {
		t.Fatalf("grown draft logit score scratch len/cap = %d/%d, want 8/>=8", len(second), cap(second))
	}
	if !slices.Equal(second[:2], []float32{0, 0}) {
		t.Fatalf("grown draft logit score scratch retained stale values: %v", second[:2])
	}
}

func TestArchSessionMTPDraftLogitSelectedScratch_Good(t *testing.T) {
	sess := &ArchSession{}
	first := sess.mtpDraftLogitSelectedScratch(4)
	first = append(first, 2, 5, 7, 11)

	second := sess.mtpDraftLogitSelectedScratch(3)
	second = append(second, 13, 17, 19)

	if !slices.Equal(second, []int{13, 17, 19}) {
		t.Fatalf("reused selected-index scratch = %v, want [13 17 19]", second)
	}
	if &second[0] != &first[0] {
		t.Fatal("selected-index scratch did not reuse its backing allocation")
	}
}

func TestArchSessionMTPDraftLogitSelectedScratch_Bad(t *testing.T) {
	sess := &ArchSession{}

	got := sess.mtpDraftLogitSelectedScratch(0)

	if got != nil {
		t.Fatalf("zero selected-index scratch = %#v, want nil", got)
	}
}

func TestArchSessionMTPDraftLogitSelectedScratch_Ugly(t *testing.T) {
	sess := &ArchSession{}
	first := sess.mtpDraftLogitSelectedScratch(2)
	first = append(first, 23, 29)

	second := sess.mtpDraftLogitSelectedScratch(9)

	if len(second) != 0 || cap(second) < 9 {
		t.Fatalf("grown selected-index scratch len/cap = %d/%d, want 0/>=9", len(second), cap(second))
	}
	second = append(second, 31, 37, 41)
	if !slices.Equal(second, []int{31, 37, 41}) {
		t.Fatalf("grown selected-index scratch = %v, want [31 37 41]", second)
	}
}

func TestArchSessionTruncateToRollsBackPositionAndResidentIDs_Good(t *testing.T) {
	sess := &ArchSession{
		pos:                5,
		cachedIDs:          []int32{1, 2, 3, 4, 5},
		cachedPromptIDs:    []int32{1, 2, 3, 4, 5},
		cachedPromptHidden: []byte{1, 2},
		cachedPromptLogits: []byte{3, 4},
		retainedHidden:     []byte{5, 6},
		retainedLogits:     []byte{7, 8},
	}

	if !sess.TruncateTo(3) {
		t.Fatal("TruncateTo(3) = false, want true")
	}
	if sess.Pos() != 3 {
		t.Fatalf("Pos after TruncateTo = %d, want 3", sess.Pos())
	}
	if !idsEqual(sess.cachedIDs, []int32{1, 2, 3}) {
		t.Fatalf("cachedIDs after TruncateTo = %v, want [1 2 3]", sess.cachedIDs)
	}
	if len(sess.cachedPromptIDs) != 0 || len(sess.cachedPromptHidden) != 0 || len(sess.cachedPromptLogits) != 0 {
		t.Fatalf("cached prompt entry survived rollback: ids=%v hidden=%v logits=%v", sess.cachedPromptIDs, sess.cachedPromptHidden, sess.cachedPromptLogits)
	}
	if len(sess.retainedHidden) != 0 || len(sess.retainedLogits) != 0 {
		t.Fatalf("retained boundary survived rollback: hidden=%v logits=%v", sess.retainedHidden, sess.retainedLogits)
	}
	if !sess.TruncateTo(3) {
		t.Fatal("TruncateTo(current pos) = false, want true")
	}
	if sess.TruncateTo(4) || sess.TruncateTo(-1) {
		t.Fatal("TruncateTo allowed growing or negative rollback")
	}
}

func repeatPenalizedLogitForTest(id int32, v float32, history []int32, penalty float32) float32 {
	if penalty <= 1 {
		return v
	}
	for _, hid := range history {
		if hid == id {
			if v > 0 {
				return v / penalty
			}
			return v * penalty
		}
	}
	return v
}

func newQuantICBAllocationSession(tb testing.TB, maxLen int) *ArchSession {
	tb.Helper()
	const dModel, nHeads, nKV, headDim, dFF, vocab = 128, 2, 1, 64, 256, 256
	const gs, bits = 64, 4
	arch, err := g4.Config{
		HiddenSize: dModel, NumHiddenLayers: 2, IntermediateSize: dFF,
		NumAttentionHeads: nHeads, NumKeyValueHeads: nKV, HeadDim: headDim,
		VocabSize: vocab, RMSNormEps: 1e-6,
		Quantization: &model.QuantConfig{GroupSize: gs, Bits: bits},
	}.Arch()
	if err != nil {
		tb.Fatalf("Arch: %v", err)
	}
	lm, err := model.Assemble(quantGemma4Tensors(tb, arch, gs, bits), arch, model.StandardWeightNames())
	if err != nil {
		tb.Fatalf("Assemble: %v", err)
	}
	g, err := loadedToQuant(lm, gs, bits)
	if err != nil {
		tb.Fatalf("loadedToQuant: %v", err)
	}
	sess, err := NewArchQuantSession(g, arch, maxLen)
	if err != nil {
		tb.Fatalf("NewArchQuantSession: %v", err)
	}
	if sess.state.icb == nil {
		tb.Skip("ICB replay unavailable")
	}
	for _, id := range []int32{1, 5, 3} {
		if _, err := sess.stepID(id); err != nil {
			tb.Fatalf("prefix stepID(%d): %v", id, err)
		}
	}
	return sess
}

func TestArchSessionNextInputTokenBufferCachesContentsPointer(t *testing.T) {
	requireNativeRuntime(t)

	sess := &ArchSession{}
	buf1 := sess.nextInputTokenBuffer(7)
	if buf1 == nil {
		t.Fatal("nextInputTokenBuffer returned nil")
	}
	if sess.nextInputTokenPtr == nil {
		t.Fatal("nextInputTokenBuffer did not cache token contents pointer")
	}
	ptr := sess.nextInputTokenPtr
	if got := *ptr; got != 7 {
		t.Fatalf("cached token value = %d, want 7", got)
	}
	buf2 := sess.nextInputTokenBuffer(11)
	if buf2 != buf1 {
		t.Fatal("nextInputTokenBuffer did not reuse the Metal token buffer")
	}
	if sess.nextInputTokenPtr != ptr {
		t.Fatal("nextInputTokenBuffer contents pointer changed after reuse")
	}
	if got := *ptr; got != 11 {
		t.Fatalf("cached token value after reuse = %d, want 11", got)
	}
}

func TestArchSessionNextInputBuffersUsePinnedNoCopyBacking(t *testing.T) {
	requireNativeRuntime(t)

	sess := &ArchSession{}
	tokenBuf := sess.nextInputTokenBuffer(17)
	if sess.nextInputTokenPinned == nil || sess.nextInputTokenPinned.pinner == nil {
		t.Fatal("next-input token scratch is not pinned no-copy")
	}
	if tokenBuf == nil || tokenBuf.Contents() != unsafe.Pointer(&sess.nextInputTokenPinned.bytes[0]) {
		t.Fatal("next-input token Metal buffer is not backed by pinned Go bytes")
	}
	if sess.nextInputTokenPtr != (*int32)(unsafe.Pointer(&sess.nextInputTokenPinned.bytes[0])) {
		t.Fatal("next-input token pointer is not the pinned Go backing")
	}

	embBuf := sess.nextInputEmbBuffer(4)
	if sess.nextInputEmbPinned == nil || sess.nextInputEmbPinned.pinner == nil {
		t.Fatal("next-input embedding scratch is not pinned no-copy")
	}
	if embBuf == nil || embBuf.Contents() != unsafe.Pointer(&sess.nextInputEmbPinned.bytes[0]) {
		t.Fatal("next-input embedding Metal buffer is not backed by pinned Go bytes")
	}
	readback := sess.nextInputEmbReadback(4)
	if len(readback) != 4*bf16Size || unsafe.Pointer(&readback[0]) != unsafe.Pointer(&sess.nextInputEmbPinned.bytes[0]) {
		t.Fatal("next-input embedding readback does not use the pinned Go backing")
	}

	firstEmbPinned := sess.nextInputEmbPinned
	embBuf2 := sess.nextInputEmbBuffer(4)
	if embBuf2 != embBuf || sess.nextInputEmbPinned != firstEmbPinned {
		t.Fatal("next-input embedding scratch changed without growing")
	}
	sess.nextInputEmbBuffer(8)
	if firstEmbPinned.bytes != nil || firstEmbPinned.pinner != nil {
		t.Fatal("next-input embedding pinned scratch was not closed on grow")
	}
}

func TestArchSessionNextInputEmbBufferReuseAllocationBudget(t *testing.T) {
	requireNativeRuntime(t)

	sess := &ArchSession{}
	if buf := sess.nextInputEmbBuffer(64); buf == nil {
		t.Fatal("nextInputEmbBuffer warmup returned nil")
	}
	allocs := testing.AllocsPerRun(100, func() {
		if buf := sess.nextInputEmbBuffer(64); buf == nil {
			t.Fatal("nextInputEmbBuffer reuse returned nil")
		}
	})
	if allocs > 0 {
		t.Fatalf("nextInputEmbBuffer reuse allocations = %.0f, want 0", allocs)
	}
}

func TestNewArchSessionInitialisesDevicePagedKVForGlobalOwners(t *testing.T) {
	requireNativeRuntime(t)

	const dModel, nHeads, nKV, headDim, dFF, vocab, maxLen = 64, 1, 1, 64, 128, 32, 4
	specs := []model.LayerSpec{
		{Attention: model.GlobalAttention, KVShareFrom: 0, CacheIndex: 0, HeadDim: headDim, KVHeads: nKV},
		{Attention: model.GlobalAttention, KVShareFrom: 0, CacheIndex: -1, HeadDim: headDim, KVHeads: nKV},
		{Attention: model.SlidingAttention, KVShareFrom: 2, CacheIndex: 1, HeadDim: headDim, KVHeads: nKV},
	}
	layers := make([]DecodeLayerWeights, len(specs))
	for i := range layers {
		layers[i] = decodeLayerFixture(dModel, nHeads, nKV, headDim, dFF, 800+i)
	}
	g := &BF16Model{
		Layers:    layers,
		Embed:     toBF16Bytes(syntheticFloat32(vocab*dModel, 811)),
		FinalNorm: toBF16Bytes(syntheticFloat32(dModel, 823)),
	}
	g.LMHead, g.Tied = g.Embed, true
	arch := model.Arch{
		Hidden: dModel, Heads: nHeads, KVHeads: nKV, HeadDim: headDim, FF: dFF, Vocab: vocab,
		Layer: specs, SlidingWindow: 2, RotaryDim: headDim, RotaryDimLocal: headDim,
		RopeBase: 10000, RopeLocalBase: 10000, AttnScale: 0.125, Eps: 1e-5,
	}

	sess, err := NewArchSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("NewArchSession: %v", err)
	}
	defer sess.Close()
	if len(sess.state.pagedKV) != len(specs) {
		t.Fatalf("paged KV entries = %d, want %d", len(sess.state.pagedKV), len(specs))
	}
	if sess.state.pagedKV[0] == nil {
		t.Fatal("global owner layer did not receive a device-paged KV cache")
	}
	if sess.state.pagedKV[1] != nil {
		t.Fatal("KV-sharing layer should read the owner page cache, not own one")
	}
	if sess.state.pagedKV[2] == nil {
		t.Fatal("sliding owner layer did not receive a ring device-paged KV cache")
	}
	if sess.state.pagedKV[0].maxSize != maxLen {
		t.Fatalf("global owner paged maxSize = %d, want %d", sess.state.pagedKV[0].maxSize, maxLen)
	}
	if !sess.state.pagedKV[2].ring || sess.state.pagedKV[2].maxSize != arch.SlidingWindow {
		t.Fatalf("sliding owner paged ring/maxSize = %v/%d, want true/%d", sess.state.pagedKV[2].ring, sess.state.pagedKV[2].maxSize, arch.SlidingWindow)
	}
}

func TestArchSessionPrefillTokensPopulatesDevicePagedKV(t *testing.T) {
	requireSDPAPagedKernel(t)

	g, arch, maxLen := sessionStateFixture(t)
	sess, err := NewArchSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("NewArchSession: %v", err)
	}
	defer sess.Close()
	// Pins the HOST prefill lane's paged-KV population (still the live lane for
	// MoE/trace sessions). bf16 sessions record the arch ICB now, whose prefill
	// writes the replay's own caches and leaves the paged KV empty by design.
	sess.state.icb = nil
	prompt := []int32{1, 2, 3, 4, 5}
	if err := sess.PrefillTokens(prompt); err != nil {
		t.Fatalf("PrefillTokens: %v", err)
	}
	for li, spec := range sess.state.specs {
		if !spec.OwnsCache() {
			continue
		}
		cache := sess.state.layerPagedKV(li)
		if cache == nil {
			continue
		}
		if cache.length != len(prompt) {
			t.Fatalf("paged KV layer %d length = %d, want %d", li, cache.length, len(prompt))
		}
		if len(cache.kPages) == 0 || len(cache.vPages) == 0 {
			t.Fatalf("paged KV layer %d has no allocated pages after prefill", li)
		}
	}
}

func TestArchSessionCloseClearsSessionOwnedScratch(t *testing.T) {
	requireNativeRuntime(t)

	sess := &ArchSession{
		sampleCandidateLogits: []byte{1, 2},
		sampleCandidateIDs:    []int32{3},
		sampleHeadLogits:      []byte{4, 5},
		sampleHidden:          []byte{6, 7},
		sampleHistory:         []int32{8},
		samplePenaltyIDs:      []int32{9},
		samplePenaltyLogits:   []byte{10, 11},
		sampleSuppressTokens:  []int32{12},
		embedScratch:          []byte{13, 14},
	}
	sess.nextInputTokenBuffer(13)
	sess.nextInputEmbBuffer(4)
	sess.nextInputEmbReadback(4)
	sess.nextInputPLEReadback(4)
	sess.plScratchNew = func() *plGPUScratch { return newPLGPUScratch(4, 0.5) }
	sess.nextInputPLScratchBuffer()
	sess.gpuTailPLScratchBuffer(0)
	sess.gpuTailPLScratchBuffer(1)

	if err := sess.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	if sess.nextInputToken != nil || sess.nextInputTokenPtr != nil || sess.nextInputTokenPinned != nil {
		t.Fatal("Close left next-input token staging alive")
	}
	if sess.nextInputEmb != nil || sess.nextInputEmbPtr != nil || sess.nextInputEmbPinned != nil {
		t.Fatal("Close left next-input embedding staging alive")
	}
	if sess.nextInputEmbHost != nil || sess.nextInputPLEHost != nil {
		t.Fatal("Close left next-input host readback backing alive")
	}
	if sess.nextInputPLScratch != nil || sess.gpuTailPLScratch[0] != nil || sess.gpuTailPLScratch[1] != nil {
		t.Fatal("Close left PLE GPU scratch alive")
	}
	if sess.sampleCandidateLogits != nil || sess.sampleCandidateIDs != nil || sess.sampleHeadLogits != nil ||
		sess.sampleHidden != nil || sess.sampleHistory != nil || sess.samplePenaltyIDs != nil ||
		sess.samplePenaltyLogits != nil || sess.sampleSuppressTokens != nil || sess.embedScratch != nil {
		t.Fatal("Close left sampled host scratch slices alive")
	}
}

func TestArchSessionEmbedIDScratchReusesBacking(t *testing.T) {
	sess := &ArchSession{
		arch: model.Arch{Hidden: 4},
		embedInto: func(dst []byte, id int32) ([]byte, error) {
			if len(dst) != 4*bf16Size {
				return nil, fmt.Errorf("dst length = %d", len(dst))
			}
			for i := range dst {
				dst[i] = byte(int(id) + i)
			}
			return dst, nil
		},
	}

	first, err := sess.embedID(3)
	if err != nil {
		t.Fatalf("embedID first: %v", err)
	}
	firstPtr := unsafe.Pointer(&first[0])
	if got := append([]byte(nil), first...); !bytes.Equal(got, []byte{3, 4, 5, 6, 7, 8, 9, 10}) {
		t.Fatalf("first embedding = %v", got)
	}

	second, err := sess.embedID(11)
	if err != nil {
		t.Fatalf("embedID second: %v", err)
	}
	if unsafe.Pointer(&second[0]) != firstPtr {
		t.Fatal("embedID did not reuse the embedding scratch backing")
	}
	if !bytes.Equal(second, []byte{11, 12, 13, 14, 15, 16, 17, 18}) {
		t.Fatalf("second embedding = %v", second)
	}
}

func TestArchSessionCloseClearsModelAndDecodeStateReferences(t *testing.T) {
	sess := &ArchSession{
		arch:          model.Arch{Hidden: 4, Vocab: 8},
		embed:         func(int32) ([]byte, error) { return nil, nil },
		embedInto:     func([]byte, int32) ([]byte, error) { return nil, nil },
		embedFuncPtr:  1,
		head:          func([]byte, bool) ([]byte, error) { return nil, nil },
		greedy:        func([]byte, []int32) (int32, bool, error) { return 0, false, nil },
		headEnc:       &headEncoder{},
		perLayerInput: func(int32, []byte) ([]byte, error) { return nil, nil },
		encNextInputsGPU: func(metal.MTLComputeCommandEncoderObject, metal.MTLBuffer, metal.MTLBuffer, *plGPUScratch) error {
			return nil
		},
		plScratchNew:       func() *plGPUScratch { return &plGPUScratch{} },
		recordPeerICB:      func() (*archICBReplay, error) { return nil, nil },
		icbPeer:            &archICBReplay{},
		state:              archDecodeState{specs: []model.LayerSpec{{}}, perLayerInput: []byte{1, 2}, hostScratch: []byte{3, 4}, icb: &archICBReplay{}},
		pos:                2,
		maxLen:             8,
		cachedIDs:          []int32{1, 2},
		cachedPromptIDs:    []int32{1},
		cachedPromptHidden: []byte{2, 3},
		cachedPromptLogits: []byte{4, 5},
		retainedHidden:     []byte{6, 7},
	}

	if err := sess.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	if sess.embed != nil || sess.embedInto != nil || sess.embedFuncPtr != 0 || sess.head != nil || sess.greedy != nil || sess.headEnc != nil || sess.perLayerInput != nil {
		t.Fatal("Close left model callbacks or head encoder alive")
	}
	if sess.encNextInputsGPU != nil || sess.plScratchNew != nil || sess.recordPeerICB != nil || sess.icbPeer != nil {
		t.Fatal("Close left GPU-tail callbacks or peer ICB alive")
	}
	if sess.state.specs != nil || sess.state.perLayerInput != nil || sess.state.hostScratch != nil || sess.state.icb != nil {
		t.Fatal("Close left decode state resources alive")
	}
	if sess.cachedIDs != nil || sess.cachedPromptIDs != nil || sess.cachedPromptHidden != nil || sess.cachedPromptLogits != nil || sess.retainedHidden != nil {
		t.Fatal("Close left prompt/cache boundary slices alive")
	}
	if sess.arch.Hidden != 0 || sess.arch.Vocab != 0 || sess.pos != 0 || sess.maxLen != 0 {
		t.Fatal("Close left session scalar state populated")
	}
}

func TestArchSessionRememberRetainedHiddenFromPointerReusesBacking(t *testing.T) {
	sess := &ArchSession{arch: model.Arch{Hidden: 4}}
	first := []byte{1, 2, 3, 4, 5, 6, 7, 8}
	sess.rememberRetainedHiddenFrom(&first[0])
	if !bytes.Equal(sess.retainedHidden, first) {
		t.Fatalf("retained hidden = %v, want %v", sess.retainedHidden, first)
	}
	first[0] = 99
	if sess.retainedHidden[0] == first[0] {
		t.Fatal("retained hidden aliases source pointer")
	}
	backing := unsafe.Pointer(&sess.retainedHidden[0])
	second := []byte{8, 7, 6, 5, 4, 3, 2, 1}
	sess.rememberRetainedHiddenFrom(&second[0])
	if !bytes.Equal(sess.retainedHidden, second) {
		t.Fatalf("retained hidden after reuse = %v, want %v", sess.retainedHidden, second)
	}
	if unsafe.Pointer(&sess.retainedHidden[0]) != backing {
		t.Fatal("retained hidden backing changed despite equal size")
	}
}

func TestArchSessionPrefillRetainedTokensBatchedDenseMatchesSerial(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	const dModel, nHeads, nKV, headDim, dFF, vocab = 128, 2, 1, 64, 256, 64
	const maxLen = 16
	g, arch := gemma4BF16Fixture(t, dModel, nHeads, nKV, headDim, dFF, vocab, 2)
	serial, err := NewArchSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("NewArchSession serial: %v", err)
	}
	batched, err := NewArchSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("NewArchSession batched: %v", err)
	}
	serial.state.icb = nil
	batched.state.icb = nil
	ids := []int32{1, 5, 3, 9}
	var serialHidden []byte
	withAutoreleasePool(func() {
		for _, id := range ids {
			serialHidden, err = serial.stepIDInPool(id)
			if err != nil {
				return
			}
		}
	})
	if err != nil {
		t.Fatalf("serial stepIDInPool: %v", err)
	}
	batchedHidden, ok, err := batched.prefillRetainedTokensBatchedDense(ids, "test")
	if err != nil {
		t.Fatalf("prefillRetainedTokensBatchedDense: %v", err)
	}
	if !ok {
		t.Fatal("prefillRetainedTokensBatchedDense declined dense fixture")
	}
	if batched.Pos() != len(ids) {
		t.Fatalf("batched pos = %d, want %d", batched.Pos(), len(ids))
	}
	if !bytes.Equal(batchedHidden, serialHidden) {
		t.Fatal("batched retained hidden differs from serial")
	}
	var serialNext, batchedNext []byte
	withAutoreleasePool(func() {
		serialNext, err = serial.stepIDInPool(4)
		if err != nil {
			return
		}
		batchedNext, err = batched.stepIDInPool(4)
	})
	if err != nil {
		t.Fatalf("post-prefill stepIDInPool: %v", err)
	}
	if !bytes.Equal(batchedNext, serialNext) {
		t.Fatal("batched prefill cache differs from serial on next token")
	}
}

func TestArchSessionPrefillTokenEmbeddingsBatchedDenseMatchesSerial(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	const dModel, nHeads, nKV, headDim, dFF, vocab = 128, 2, 1, 64, 256, 64
	const maxLen = 16
	g, arch := gemma4BF16Fixture(t, dModel, nHeads, nKV, headDim, dFF, vocab, 2)
	serial, err := NewArchSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("NewArchSession serial: %v", err)
	}
	batched, err := NewArchSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("NewArchSession batched: %v", err)
	}
	serial.state.icb = nil
	batched.state.icb = nil
	ids := []int32{1, 5, 3, 9}
	embeddings := make([][]byte, len(ids))
	for i, id := range ids {
		emb, err := serial.embedID(id)
		if err != nil {
			t.Fatalf("embedID(%d): %v", id, err)
		}
		embeddings[i] = append([]byte(nil), emb...)
	}
	replacement, err := serial.embedID(7)
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
	if err := batched.PrefillTokenEmbeddings(ids, embeddings); err != nil {
		t.Fatalf("PrefillTokenEmbeddings: %v", err)
	}
	if batched.Pos() != len(ids) {
		t.Fatalf("batched pos = %d, want %d", batched.Pos(), len(ids))
	}
	if !idsEqual(batched.cachedIDs, ids) {
		t.Fatalf("cached ids = %v, want %v", batched.cachedIDs, ids)
	}
	if batched.state.denseBatch.lastRows == nil {
		t.Fatal("PrefillTokenEmbeddings did not use dense batched final-row output")
	}
	if !bytes.Equal(batched.retainedHidden, serialHidden) {
		t.Fatal("batched explicit-embedding hidden differs from serial")
	}
	var serialNext, batchedNext []byte
	nextSerialEmb, err := serial.embedID(4)
	if err != nil {
		t.Fatalf("serial next embedID: %v", err)
	}
	nextBatchedEmb, err := batched.embedID(4)
	if err != nil {
		t.Fatalf("batched next embedID: %v", err)
	}
	serialNext, err = serial.StepWithID(4, nextSerialEmb)
	if err != nil {
		t.Fatalf("serial next StepWithID: %v", err)
	}
	batchedNext, err = batched.StepWithID(4, nextBatchedEmb)
	if err != nil {
		t.Fatalf("batched next StepWithID: %v", err)
	}
	if !bytes.Equal(batchedNext, serialNext) {
		t.Fatal("batched explicit-embedding cache differs from serial on next token")
	}
}

func TestArchSessionPrefillRetainedTokensBatchedDenseUsesEmbedInto(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	const dModel, nHeads, nKV, headDim, dFF, vocab = 128, 2, 1, 64, 256, 64
	const maxLen = 16
	g, arch := gemma4BF16Fixture(t, dModel, nHeads, nKV, headDim, dFF, vocab, 2)
	ids := []int32{1, 5, 3, 9}
	control, err := NewArchSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("NewArchSession control: %v", err)
	}
	candidate, err := NewArchSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("NewArchSession candidate: %v", err)
	}
	control.state.icb = nil
	candidate.state.icb = nil

	want, ok, err := control.prefillRetainedTokensBatchedDense(ids, "test")
	if err != nil {
		t.Fatalf("control prefillRetainedTokensBatchedDense: %v", err)
	}
	if !ok {
		t.Fatal("control prefillRetainedTokensBatchedDense declined dense fixture")
	}

	candidate.embed = func(int32) ([]byte, error) {
		return nil, errors.New("allocating embed path called")
	}
	candidate.embedFuncPtr = 0
	got, ok, err := candidate.prefillRetainedTokensBatchedDense(ids, "test")
	if err != nil {
		t.Fatalf("candidate prefillRetainedTokensBatchedDense: %v", err)
	}
	if !ok {
		t.Fatal("candidate prefillRetainedTokensBatchedDense declined dense fixture")
	}
	if !bytes.Equal(got, want) {
		t.Fatal("embedInto dense prefill hidden differs from allocating reference")
	}
}

func TestArchSessionPrefillRetainedTokensBatchedDenseChunksSlidingRingWrap(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	const dModel, nHeads, nKV, headDim, dFF, vocab = 128, 2, 1, 64, 256, 64
	const maxLen = 16
	g, arch := gemma4BF16Fixture(t, dModel, nHeads, nKV, headDim, dFF, vocab, 1)
	arch.SlidingWindow = 4
	arch.Layer[0].Attention = model.SlidingAttention
	serial, err := NewArchSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("NewArchSession serial: %v", err)
	}
	chunked, err := NewArchSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("NewArchSession chunked: %v", err)
	}
	serial.state.icb = nil
	chunked.state.icb = nil

	ids := []int32{1, 5, 3, 9, 4}
	var serialHidden []byte
	withAutoreleasePool(func() {
		for _, id := range ids {
			serialHidden, err = serial.stepIDInPool(id)
			if err != nil {
				return
			}
		}
	})
	if err != nil {
		t.Fatalf("serial stepIDInPool: %v", err)
	}

	hidden, ok, err := chunked.prefillRetainedTokensBatchedDense(ids, "test")
	if err != nil {
		t.Fatalf("prefillRetainedTokensBatchedDense chunked sliding wrap: %v", err)
	}
	if !ok {
		t.Fatal("prefillRetainedTokensBatchedDense chunked sliding wrap ok = false")
	}
	if chunked.Pos() != len(ids) {
		t.Fatalf("chunked pos = %d, want %d", chunked.Pos(), len(ids))
	}
	if !bytes.Equal(hidden, serialHidden) {
		t.Fatal("chunked sliding dense prefill hidden differs from serial")
	}
	var serialNext, chunkedNext []byte
	withAutoreleasePool(func() {
		serialNext, err = serial.stepIDInPool(6)
		if err != nil {
			return
		}
		chunkedNext, err = chunked.stepIDInPool(6)
	})
	if err != nil {
		t.Fatalf("post-prefill stepIDInPool: %v", err)
	}
	if !bytes.Equal(chunkedNext, serialNext) {
		t.Fatal("chunked sliding dense prefill cache differs from serial on next token")
	}
}

func TestArchSessionBatchedDensePrefillChunkLenSkip(t *testing.T) {
	t.Setenv("LTHN_PREFILL_WINDOWS", "4") // pin wide = 4×512 = 2048
	walk := func(pos, total int) []int {
		s := &ArchSession{maxLen: 1 << 20}
		s.arch.SlidingWindow = 512
		s.pos = pos
		var seq []int
		for total > 0 {
			n := s.batchedDensePrefillChunkLenSkip(total)
			if n <= 0 || n > total {
				t.Fatalf("walk(pos=%d): chunk %d of remaining %d", pos, n, total)
			}
			seq = append(seq, n)
			s.pos += n
			total -= n
		}
		return seq
	}
	cases := []struct {
		name       string
		pos, total int
		want       []int
	}{
		{"8K shape: minimal 57-row boundary chunk", 0, 7225, []int{2048, 2048, 2048, 1024, 57}},
		{"sub-floor remainder rises by a window", 0, 7180, []int{2048, 2048, 2048, 512, 524}},
		{"aligned end keeps one whole window", 0, 6144, []int{2048, 2048, 1536, 512}},
		{"below the floor stays one chunk", 0, 20, []int{20}},
		{"mid-window start realigns first", 100, 1000, []int{412, 512, 76}},
	}
	for _, tc := range cases {
		if got := walk(tc.pos, tc.total); !slices.Equal(got, tc.want) {
			t.Errorf("%s: chunks = %v, want %v", tc.name, got, tc.want)
		}
	}
}

func TestArchSessionPrefillChunksSkipSharedSuffix(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	t.Setenv("LTHN_PREFILL_WINDOWS", "1") // 4-token chunks: 14 ids → two skipped chunks + a final
	const dModel, nHeads, nKV, headDim, dFF, vocab = 64, 1, 1, 64, 128, 32
	const maxLen = 16
	specs := []model.LayerSpec{
		{Attention: model.SlidingAttention, KVShareFrom: 0, CacheIndex: 0, HeadDim: headDim, KVHeads: nKV},
		{Attention: model.SlidingAttention, KVShareFrom: 1, CacheIndex: 1, HeadDim: headDim, KVHeads: nKV},
		{Attention: model.SlidingAttention, KVShareFrom: 1, CacheIndex: -1, HeadDim: headDim, KVHeads: nKV},
		{Attention: model.SlidingAttention, KVShareFrom: 1, CacheIndex: -1, HeadDim: headDim, KVHeads: nKV},
	}
	layers := make([]DecodeLayerWeights, len(specs))
	for i := range layers {
		layers[i] = decodeLayerFixture(dModel, nHeads, nKV, headDim, dFF, 900+i)
	}
	g := &BF16Model{
		Layers:    layers,
		Embed:     toBF16Bytes(syntheticFloat32(vocab*dModel, 911)),
		FinalNorm: toBF16Bytes(syntheticFloat32(dModel, 923)),
	}
	g.LMHead, g.Tied = g.Embed, true
	arch := model.Arch{
		Hidden: dModel, Heads: nHeads, KVHeads: nKV, HeadDim: headDim, FF: dFF, Vocab: vocab,
		Layer: specs, SlidingWindow: 4, RotaryDim: headDim, RotaryDimLocal: headDim,
		RopeBase: 10000, RopeLocalBase: 10000, AttnScale: 0.125, Eps: 1e-5,
	}
	newSess := func(name string) *ArchSession {
		sess, err := NewArchSession(g, arch, maxLen)
		if err != nil {
			t.Fatalf("NewArchSession %s: %v", name, err)
		}
		t.Cleanup(func() { _ = sess.Close() })
		sess.state.icb = nil
		return sess
	}
	serial, skip := newSess("serial"), newSess("skip")
	if got := skip.state.sharedSuffix; got != 2 {
		t.Fatalf("sharedSuffix = %d, want 2 (two trailing KV-shared layers)", got)
	}

	ids := []int32{1, 5, 3, 9, 4, 7, 2, 8, 6, 11, 13, 10, 12, 14}
	var serialHidden []byte
	var err error
	withAutoreleasePool(func() {
		for _, id := range ids {
			serialHidden, err = serial.stepIDInPool(id)
			if err != nil {
				return
			}
		}
	})
	if err != nil {
		t.Fatalf("serial stepIDInPool: %v", err)
	}

	hidden, ok, err := skip.prefillRetainedTokensBatchedDense(ids, "test")
	if err != nil {
		t.Fatalf("prefillRetainedTokensBatchedDense skip: %v", err)
	}
	if !ok {
		t.Fatal("prefillRetainedTokensBatchedDense skip declined dense fixture")
	}
	if !bytes.Equal(hidden, serialHidden) {
		t.Fatal("shared-suffix skip chunked prefill hidden differs from serial")
	}
	if skip.state.prefillSkipToLayer != 0 {
		t.Fatalf("prefillSkipToLayer leaked = %d, want 0 after prefill", skip.state.prefillSkipToLayer)
	}

	// Kill switch: the full-stack chunk lane must produce the same bytes.
	prefillSkipSharedOffForTest = true
	defer func() { prefillSkipSharedOffForTest = false }()
	noskip := newSess("noskip")
	noskipHidden, ok, err := noskip.prefillRetainedTokensBatchedDense(ids, "test")
	if err != nil {
		t.Fatalf("prefillRetainedTokensBatchedDense noskip: %v", err)
	}
	if !ok {
		t.Fatal("prefillRetainedTokensBatchedDense noskip declined dense fixture")
	}
	if !bytes.Equal(noskipHidden, serialHidden) {
		t.Fatal("full-stack chunked prefill hidden differs from serial")
	}

	var serialNext, skipNext []byte
	withAutoreleasePool(func() {
		serialNext, err = serial.stepIDInPool(6)
		if err != nil {
			return
		}
		skipNext, err = skip.stepIDInPool(6)
	})
	if err != nil {
		t.Fatalf("post-prefill stepIDInPool: %v", err)
	}
	if !bytes.Equal(skipNext, serialNext) {
		t.Fatal("shared-suffix skip prefill cache differs from serial on next token")
	}
}

// TestArchSessionPrefillTokenEmbeddingsKeepsFullStackIgnoresSkip guards the
// multimodal embeddings/bidir prefill lane against inheriting the causal
// kv-shared suffix skip (#381). The fixture has a clean shared suffix
// (sharedSuffix == 2) so the skip WOULD engage if the lane honoured
// prefillSkipToLayer. With the skip pre-armed, the lane must still produce the
// full-stack serial boundary hidden AND clear the flag: unlike the causal lane
// it has no per-chunk reset, so a leaked value would bound the FINAL (read)
// chunk and corrupt the boundary hidden. The skip is byte-identical on non-final
// chunks by construction, so only the final-chunk divergence + the residual flag
// are observable — both are asserted.
func TestArchSessionPrefillTokenEmbeddingsKeepsFullStackIgnoresSkip(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	const dModel, nHeads, nKV, headDim, dFF, vocab = 64, 1, 1, 64, 128, 32
	const maxLen = 16
	specs := []model.LayerSpec{
		{Attention: model.SlidingAttention, KVShareFrom: 0, CacheIndex: 0, HeadDim: headDim, KVHeads: nKV},
		{Attention: model.SlidingAttention, KVShareFrom: 1, CacheIndex: 1, HeadDim: headDim, KVHeads: nKV},
		{Attention: model.SlidingAttention, KVShareFrom: 1, CacheIndex: -1, HeadDim: headDim, KVHeads: nKV},
		{Attention: model.SlidingAttention, KVShareFrom: 1, CacheIndex: -1, HeadDim: headDim, KVHeads: nKV},
	}
	layers := make([]DecodeLayerWeights, len(specs))
	for i := range layers {
		layers[i] = decodeLayerFixture(dModel, nHeads, nKV, headDim, dFF, 900+i)
	}
	g := &BF16Model{
		Layers:    layers,
		Embed:     toBF16Bytes(syntheticFloat32(vocab*dModel, 911)),
		FinalNorm: toBF16Bytes(syntheticFloat32(dModel, 923)),
	}
	g.LMHead, g.Tied = g.Embed, true
	arch := model.Arch{
		Hidden: dModel, Heads: nHeads, KVHeads: nKV, HeadDim: headDim, FF: dFF, Vocab: vocab,
		Layer: specs, SlidingWindow: 4, RotaryDim: headDim, RotaryDimLocal: headDim,
		RopeBase: 10000, RopeLocalBase: 10000, AttnScale: 0.125, Eps: 1e-5,
	}
	newSess := func(name string) *ArchSession {
		sess, err := NewArchSession(g, arch, maxLen)
		if err != nil {
			t.Fatalf("NewArchSession %s: %v", name, err)
		}
		t.Cleanup(func() { _ = sess.Close() })
		sess.state.icb = nil
		return sess
	}
	serial, chunked := newSess("serial"), newSess("chunked")
	if got := chunked.state.sharedSuffix; got != 2 {
		t.Fatalf("sharedSuffix = %d, want 2 (fixture would engage the skip if honoured)", got)
	}

	// SlidingWindow 4 over 14 ids forces the batched-dense chunk lane (multiple
	// chunks, so a bounded FINAL chunk would show).
	ids := []int32{1, 5, 3, 9, 4, 7, 2, 8, 6, 11, 13, 10, 12, 14}
	embeddings := make([][]byte, len(ids))
	for i, id := range ids {
		emb, err := serial.embedID(id)
		if err != nil {
			t.Fatalf("embedID(%d): %v", id, err)
		}
		embeddings[i] = append([]byte(nil), emb...)
	}
	var serialHidden []byte
	var err error
	for i, id := range ids {
		serialHidden, err = serial.StepWithID(id, embeddings[i])
		if err != nil {
			t.Fatalf("serial StepWithID(%d): %v", id, err)
		}
	}

	// Pre-arm the skip: a leaked prefillSkipToLayer must be IGNORED, not honoured.
	chunked.state.prefillSkipToLayer = chunked.state.sharedSuffix
	if err := chunked.PrefillTokenEmbeddings(ids, embeddings); err != nil {
		t.Fatalf("PrefillTokenEmbeddings: %v", err)
	}
	if !bytes.Equal(chunked.retainedHidden, serialHidden) {
		t.Fatal("embeddings prefill hidden differs from full-stack serial — a leaked skip bounded the read (final) chunk")
	}
	if chunked.state.prefillSkipToLayer != 0 {
		t.Fatalf("prefillSkipToLayer = %d after embeddings prefill, want 0 (lane must pin the full stack)", chunked.state.prefillSkipToLayer)
	}
}

func TestArchSessionPrefillTokenEmbeddingsBatchedDenseChunksSlidingRingWrap(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	const dModel, nHeads, nKV, headDim, dFF, vocab = 128, 2, 1, 64, 256, 64
	const maxLen = 16
	g, arch := gemma4BF16Fixture(t, dModel, nHeads, nKV, headDim, dFF, vocab, 1)
	arch.SlidingWindow = 4
	arch.Layer[0].Attention = model.SlidingAttention
	serial, err := NewArchSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("NewArchSession serial: %v", err)
	}
	chunked, err := NewArchSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("NewArchSession chunked: %v", err)
	}
	serial.state.icb = nil
	chunked.state.icb = nil
	ids := []int32{1, 5, 3, 9, 4}
	embeddings := make([][]byte, len(ids))
	for i, id := range ids {
		emb, err := serial.embedID(id)
		if err != nil {
			t.Fatalf("embedID(%d): %v", id, err)
		}
		embeddings[i] = append([]byte(nil), emb...)
	}
	var serialHidden []byte
	for i, id := range ids {
		serialHidden, err = serial.StepWithID(id, embeddings[i])
		if err != nil {
			t.Fatalf("serial StepWithID(%d): %v", id, err)
		}
	}
	if err := chunked.PrefillTokenEmbeddings(ids, embeddings); err != nil {
		t.Fatalf("PrefillTokenEmbeddings sliding: %v", err)
	}
	if chunked.Pos() != len(ids) {
		t.Fatalf("chunked pos = %d, want %d", chunked.Pos(), len(ids))
	}
	if !bytes.Equal(chunked.retainedHidden, serialHidden) {
		t.Fatal("chunked explicit-embedding retained hidden differs from serial")
	}
	nextSerialEmb, err := serial.embedID(2)
	if err != nil {
		t.Fatalf("serial next embedID: %v", err)
	}
	nextChunkedEmb, err := chunked.embedID(2)
	if err != nil {
		t.Fatalf("chunked next embedID: %v", err)
	}
	serialNext, err := serial.StepWithID(2, nextSerialEmb)
	if err != nil {
		t.Fatalf("serial next StepWithID: %v", err)
	}
	chunkedNext, err := chunked.StepWithID(2, nextChunkedEmb)
	if err != nil {
		t.Fatalf("chunked next StepWithID: %v", err)
	}
	if !bytes.Equal(chunkedNext, serialNext) {
		t.Fatal("chunked explicit-embedding cache differs from serial on next token")
	}
}

func TestArchSessionPrefillTokensSlidingRingWrapMatchesSerial(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	const dModel, nHeads, nKV, headDim, dFF, vocab = 128, 2, 1, 64, 256, 64
	const maxLen = 16
	g, arch := gemma4BF16Fixture(t, dModel, nHeads, nKV, headDim, dFF, vocab, 1)
	arch.SlidingWindow = 4
	arch.Layer[0].Attention = model.SlidingAttention
	serial, err := NewArchSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("NewArchSession serial: %v", err)
	}
	retained, err := NewArchSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("NewArchSession retained: %v", err)
	}
	serial.state.icb = nil
	retained.state.icb = nil
	ids := []int32{1, 5, 3, 9, 4}
	var serialHidden []byte
	withAutoreleasePool(func() {
		for _, id := range ids {
			serialHidden, err = serial.stepIDInPool(id)
			if err != nil {
				return
			}
		}
	})
	if err != nil {
		t.Fatalf("serial stepIDInPool: %v", err)
	}
	if err := retained.PrefillTokens(ids); err != nil {
		t.Fatalf("PrefillTokens: %v", err)
	}
	if retained.Pos() != len(ids) {
		t.Fatalf("retained pos = %d, want %d", retained.Pos(), len(ids))
	}
	if !bytes.Equal(retained.retainedHidden, serialHidden) {
		t.Fatal("PrefillTokens sliding wrap hidden differs from serial")
	}
	var serialNext, retainedNext []byte
	withAutoreleasePool(func() {
		serialNext, err = serial.stepIDInPool(6)
		if err != nil {
			return
		}
		retainedNext, err = retained.stepIDInPool(6)
	})
	if err != nil {
		t.Fatalf("post-prefill stepIDInPool: %v", err)
	}
	if !bytes.Equal(retainedNext, serialNext) {
		t.Fatal("PrefillTokens sliding wrap cache differs from serial on next token")
	}
}

// TestArchQuantSessionICBBoundsSlidingCacheToWindow is the sliding-window KV memory fix gate on
// the session's ICB fast path (the recorded-replay Step/StepWithID actually use — see
// newArchQuantSessionShardsWithHead's icbEligible block in arch_session.go). Before the fix, EVERY
// owning layer's kCaches/vCaches buffer — sliding or global — was allocated at the full maxLen
// context; a sliding layer only ever attends its own window, so that was O(context) memory for
// O(window) need. It proves both halves of the gate:
//
//   - the memory bound itself: the sliding owner's cacheRows (its buffer's actual row capacity,
//     computed in recordArchICB from the allocated buffer's length) is arch.SlidingWindow, not
//     maxLen — a direct, white-box measurement of the allocation, not just an inference from
//     matching output;
//   - correctness: archICBReplay.prepareStepRebind's pos%cacheRows ring write/read stays
//     byte-identical to the re-encode oracle (DecodeForwardArchQuant, reached here by forcing
//     state.icb nil) for every token — both while pos is still inside the window and once the
//     ring has slid past it several times over.
func TestArchQuantSessionICBBoundsSlidingCacheToWindow(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	const dModel, nHeads, nKV, headDim, dFF, vocab = 128, 2, 1, 64, 256, 256
	const gs, bits = 64, 4
	const maxLen, window = 20, 4

	build := func(tb testing.TB) *ArchSession {
		tb.Helper()
		arch, err := g4.Config{
			HiddenSize: dModel, NumHiddenLayers: 1, IntermediateSize: dFF,
			NumAttentionHeads: nHeads, NumKeyValueHeads: nKV, HeadDim: headDim,
			VocabSize: vocab, RMSNormEps: 1e-6,
			Quantization: &model.QuantConfig{GroupSize: gs, Bits: bits},
		}.Arch()
		if err != nil {
			tb.Fatalf("Arch: %v", err)
		}
		arch.SlidingWindow = window
		arch.Layer[0].Attention = model.SlidingAttention
		lm, err := model.Assemble(quantGemma4Tensors(tb, arch, gs, bits), arch, model.StandardWeightNames())
		if err != nil {
			tb.Fatalf("Assemble: %v", err)
		}
		g, err := loadedToQuant(lm, gs, bits)
		if err != nil {
			tb.Fatalf("loadedToQuant: %v", err)
		}
		sess, err := NewArchQuantSession(g, arch, maxLen)
		if err != nil {
			tb.Fatalf("NewArchQuantSession: %v", err)
		}
		return sess
	}

	icbSess := build(t)
	if icbSess.state.icb == nil {
		t.Skip("ICB replay unavailable for this fixture")
	}
	refSess := build(t)
	refSess.state.icb = nil // force the re-encode path as the byte-identical oracle

	if got := icbSess.state.icb.cacheRows[0]; got != window {
		t.Fatalf("sliding owner cacheRows = %d, want %d (bounded to SlidingWindow, not maxLen=%d)", got, window, maxLen)
	}

	// 20 tokens over a window of 4: the ring slides 4x over — well past a single wrap.
	ids := []int32{1, 5, 3, 9, 4, 2, 7, 6, 3, 1, 8, 2, 5, 9, 3, 6, 1, 4, 7, 2}
	for i, id := range ids {
		var icbHidden, refHidden []byte
		var icbErr, refErr error
		withAutoreleasePool(func() {
			icbHidden, icbErr = icbSess.stepIDInPool(id)
			refHidden, refErr = refSess.stepIDInPool(id)
		})
		if icbErr != nil {
			t.Fatalf("icb stepIDInPool(%d) tok%d: %v", id, i, icbErr)
		}
		if refErr != nil {
			t.Fatalf("ref stepIDInPool(%d) tok%d: %v", id, i, refErr)
		}
		if !bytes.Equal(icbHidden, refHidden) {
			t.Fatalf("tok%d (pos %d, window %d): bounded-ring ICB hidden differs from re-encode oracle", i, i, window)
		}
	}
	t.Logf("sliding owner cache bounded to %d rows (maxLen=%d, %.0fx smaller) — ICB ring replay == re-encode oracle byte-for-byte across %d tokens, inside and past the window", window, maxLen, float64(maxLen)/float64(window), len(ids))
}

func TestArchSessionPrefillRetainedTokensBatchedDenseReusesHiddenReadback(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	const dModel, nHeads, nKV, headDim, dFF, vocab = 128, 2, 1, 64, 256, 64
	const maxLen = 16
	g, arch := gemma4BF16Fixture(t, dModel, nHeads, nKV, headDim, dFF, vocab, 2)
	sess, err := NewArchSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("NewArchSession: %v", err)
	}
	sess.state.icb = nil
	firstHidden, ok, err := sess.prefillRetainedTokensBatchedDense([]int32{1, 5, 3}, "test")
	if err != nil {
		t.Fatalf("first prefillRetainedTokensBatchedDense: %v", err)
	}
	if !ok {
		t.Fatal("first prefillRetainedTokensBatchedDense declined dense fixture")
	}
	if len(firstHidden) == 0 {
		t.Fatal("first hidden is empty")
	}
	firstPtr := uintptr(unsafe.Pointer(&firstHidden[0]))
	firstCopy := append([]byte(nil), firstHidden...)
	heldHidden := [][]byte{firstHidden}

	secondHidden, ok, err := sess.prefillRetainedTokensBatchedDense([]int32{9, 4}, "test")
	if err != nil {
		t.Fatalf("second prefillRetainedTokensBatchedDense: %v", err)
	}
	if !ok {
		t.Fatal("second prefillRetainedTokensBatchedDense declined dense fixture")
	}
	if len(secondHidden) == 0 {
		t.Fatal("second hidden is empty")
	}
	secondPtr := uintptr(unsafe.Pointer(&secondHidden[0]))
	runtime.KeepAlive(heldHidden)
	if secondPtr != firstPtr {
		t.Fatalf("dense retained prefill hidden readback allocated new backing: first=%#x second=%#x", firstPtr, secondPtr)
	}
	if bytes.Equal(secondHidden, firstCopy) {
		t.Fatal("second hidden did not refresh contents")
	}
	if sess.Pos() != 5 {
		t.Fatalf("session position = %d, want 5", sess.Pos())
	}
}

func TestArchSessionPrefillRetainedTokensBatchedDenseReturnsRetainedHidden(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	const dModel, nHeads, nKV, headDim, dFF, vocab = 128, 2, 1, 64, 256, 64
	const maxLen = 16
	g, arch := gemma4BF16Fixture(t, dModel, nHeads, nKV, headDim, dFF, vocab, 2)
	sess, err := NewArchSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("NewArchSession: %v", err)
	}
	sess.state.icb = nil
	sess.sampleHidden = make([]byte, arch.Hidden*bf16Size)
	hidden, ok, err := sess.prefillRetainedTokensBatchedDense([]int32{1, 5, 3}, "test")
	if err != nil {
		t.Fatalf("prefillRetainedTokensBatchedDense: %v", err)
	}
	if !ok {
		t.Fatal("prefillRetainedTokensBatchedDense declined dense fixture")
	}
	if len(hidden) == 0 {
		t.Fatal("hidden is empty")
	}
	if len(sess.retainedHidden) == 0 || unsafe.Pointer(&hidden[0]) != unsafe.Pointer(&sess.retainedHidden[0]) {
		t.Fatal("batched retained prefill did not return retained hidden backing")
	}
	if sess.retainedHiddenBuffer() == nil {
		t.Fatal("batched retained prefill did not keep a pinned retained hidden buffer")
	}
	if cap(sess.sampleHidden) != 0 {
		t.Fatalf("sample hidden scratch cap = %d, want 0", cap(sess.sampleHidden))
	}
}

func TestArchSessionPrefillRetainedTokensBatchedDenseWritesLastHiddenDirectly(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	const dModel, nHeads, nKV, headDim, dFF, vocab = 128, 2, 1, 64, 256, 64
	const maxLen = 16
	g, arch := gemma4BF16Fixture(t, dModel, nHeads, nKV, headDim, dFF, vocab, 1)
	sess, err := NewArchSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("NewArchSession: %v", err)
	}
	defer sess.Close()
	sess.state.icb = nil

	withAutoreleasePool(func() {
		sess.state.denseBatch.rows(1, arch.Hidden)
	})
	outScratch := unsafe.Slice((*byte)(sess.state.denseBatch.outPacked.Contents()), arch.Hidden*bf16Size)
	sentinel := bytes.Repeat([]byte{0x73}, len(outScratch))
	copy(outScratch, sentinel)

	hidden, ok, err := sess.prefillRetainedTokensBatchedDense([]int32{1}, "test")
	if err != nil {
		t.Fatalf("prefillRetainedTokensBatchedDense: %v", err)
	}
	if !ok {
		t.Fatal("prefillRetainedTokensBatchedDense declined dense fixture")
	}
	if len(sess.retainedHidden) == 0 || unsafe.Pointer(&hidden[0]) != unsafe.Pointer(&sess.retainedHidden[0]) {
		t.Fatal("batched retained prefill did not return retained hidden backing")
	}
	retainedBuf := sess.retainedHiddenBuffer()
	if retainedBuf == nil {
		t.Fatal("batched retained prefill did not keep a pinned retained hidden buffer")
	}
	if sess.state.denseBatch.lastRows == nil {
		t.Fatal("batched retained prefill did not record a final row buffer")
	}
	if sess.state.denseBatch.lastRows.GetID() != retainedBuf.GetID() {
		t.Fatalf("batched retained prefill final row buffer id = %d, want retained buffer %d", sess.state.denseBatch.lastRows.GetID(), retainedBuf.GetID())
	}
	if !bytes.Equal(outScratch, sentinel) {
		t.Fatal("batched retained prefill wrote last hidden through dense output scratch")
	}
}

func TestArchSessionPrefillRetainedTokensBatchedDenseReusesRowScratch(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	const dModel, nHeads, nKV, headDim, dFF, vocab = 128, 2, 1, 64, 256, 64
	const maxLen = 16
	g, arch := gemma4BF16Fixture(t, dModel, nHeads, nKV, headDim, dFF, vocab, 2)
	sess, err := NewArchSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("NewArchSession: %v", err)
	}
	sess.state.icb = nil
	if _, ok, err := sess.prefillRetainedTokensBatchedDense([]int32{1, 5, 3}, "test"); err != nil {
		t.Fatalf("first prefillRetainedTokensBatchedDense: %v", err)
	} else if !ok {
		t.Fatal("first prefillRetainedTokensBatchedDense declined dense fixture")
	}
	if len(sess.state.denseBatch.inRows) < 3 || len(sess.state.denseBatch.outRows) < 3 || len(sess.state.denseBatch.offBuf) < 3 {
		t.Fatalf("dense batch scratch lengths = in:%d out:%d off:%d, want at least 3",
			len(sess.state.denseBatch.inRows), len(sess.state.denseBatch.outRows), len(sess.state.denseBatch.offBuf))
	}
	firstIn0, firstOut0, firstOff0 := sess.state.denseBatch.inRows[0], sess.state.denseBatch.outRows[0], sess.state.denseBatch.offBuf[0]
	firstOffPtr0 := sess.state.denseBatch.offPtr[0]
	if firstOffPtr0 == nil {
		t.Fatal("first offset pointer is nil")
	}
	if got := *firstOffPtr0; got != 0 {
		t.Fatalf("first offset value = %d, want 0", got)
	}

	if _, ok, err := sess.prefillRetainedTokensBatchedDense([]int32{9, 4}, "test"); err != nil {
		t.Fatalf("second prefillRetainedTokensBatchedDense: %v", err)
	} else if !ok {
		t.Fatal("second prefillRetainedTokensBatchedDense declined dense fixture")
	}
	if sess.state.denseBatch.inRows[0] != firstIn0 {
		t.Fatal("dense batch input row scratch was replaced")
	}
	if sess.state.denseBatch.outRows[0] != firstOut0 {
		t.Fatal("dense batch output row scratch was replaced")
	}
	if sess.state.denseBatch.offBuf[0] != firstOff0 {
		t.Fatal("dense batch offset buffer was replaced")
	}
	if sess.state.denseBatch.offPtr[0] != firstOffPtr0 {
		t.Fatal("dense batch offset pointer changed")
	}
	if got := *sess.state.denseBatch.offPtr[0]; got != 3 {
		t.Fatalf("reused first offset value = %d, want 3", got)
	}
	if got := *sess.state.denseBatch.offPtr[1]; got != 4 {
		t.Fatalf("reused second offset value = %d, want 4", got)
	}
}

func TestArchSessionPrefillRetainedTokensBatchedDensePacksOffsetScratch(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	const dModel, nHeads, nKV, headDim, dFF, vocab = 128, 2, 1, 64, 256, 64
	const maxLen = 16
	g, arch := gemma4BF16Fixture(t, dModel, nHeads, nKV, headDim, dFF, vocab, 2)
	sess, err := NewArchSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("NewArchSession: %v", err)
	}
	sess.state.icb = nil
	if _, ok, err := sess.prefillRetainedTokensBatchedDense([]int32{1, 5, 3}, "test"); err != nil {
		t.Fatalf("prefillRetainedTokensBatchedDense: %v", err)
	} else if !ok {
		t.Fatal("prefillRetainedTokensBatchedDense declined dense fixture")
	}
	if len(sess.state.denseBatch.offBuf) < 3 || len(sess.state.denseBatch.offPtr) < 3 {
		t.Fatalf("dense batch offset scratch lengths = off:%d ptr:%d, want at least 3",
			len(sess.state.denseBatch.offBuf), len(sess.state.denseBatch.offPtr))
	}
	if sess.state.denseBatch.offBuf[1] != sess.state.denseBatch.offBuf[0] || sess.state.denseBatch.offBuf[2] != sess.state.denseBatch.offBuf[0] {
		t.Fatal("dense batch offsets use multiple Metal buffers instead of one packed buffer")
	}
	if got := *sess.state.denseBatch.offPtr[0]; got != 0 {
		t.Fatalf("packed offset[0] = %d, want 0", got)
	}
	if got := *sess.state.denseBatch.offPtr[1]; got != 1 {
		t.Fatalf("packed offset[1] = %d, want 1", got)
	}
	if got := *sess.state.denseBatch.offPtr[2]; got != 2 {
		t.Fatalf("packed offset[2] = %d, want 2", got)
	}
}

func TestArchSessionPrefillRetainedTokensBatchedDensePacksRowScratch(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	const dModel, nHeads, nKV, headDim, dFF, vocab = 128, 2, 1, 64, 256, 64
	const maxLen = 16
	g, arch := gemma4BF16Fixture(t, dModel, nHeads, nKV, headDim, dFF, vocab, 2)
	sess, err := NewArchSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("NewArchSession: %v", err)
	}
	sess.state.icb = nil
	if _, ok, err := sess.prefillRetainedTokensBatchedDense([]int32{1, 5, 3}, "test"); err != nil {
		t.Fatalf("prefillRetainedTokensBatchedDense: %v", err)
	} else if !ok {
		t.Fatal("prefillRetainedTokensBatchedDense declined dense fixture")
	}
	if len(sess.state.denseBatch.inRows) < 3 || len(sess.state.denseBatch.outRows) < 3 {
		t.Fatalf("dense batch row scratch lengths = in:%d out:%d, want at least 3",
			len(sess.state.denseBatch.inRows), len(sess.state.denseBatch.outRows))
	}
	if sess.state.denseBatch.inRows[1] != sess.state.denseBatch.inRows[0] || sess.state.denseBatch.inRows[2] != sess.state.denseBatch.inRows[0] {
		t.Fatal("dense batch input rows use multiple Metal buffers instead of one packed buffer")
	}
	if sess.state.denseBatch.outRows[1] != sess.state.denseBatch.outRows[0] || sess.state.denseBatch.outRows[2] != sess.state.denseBatch.outRows[0] {
		t.Fatal("dense batch output rows use multiple Metal buffers instead of one packed buffer")
	}
}

func sampleBF16VocabOrderForTest(logits []byte, vocab int, params model.SampleParams, draw float32, history []int32) (int32, error) {
	if len(logits) != vocab*bf16Size {
		return 0, fmt.Errorf("logits length %d, want %d", len(logits), vocab*bf16Size)
	}
	if params.Temperature <= 0 {
		return greedyBF16Suppressed(logits, vocab, params.SuppressTokens)
	}
	maxV := float32(math.Inf(-1))
	allowed := 0
	for i := range vocab {
		if tokenSuppressed(i, params.SuppressTokens) {
			continue
		}
		v := repeatPenalizedLogitForTest(int32(i), bf16ToF32(logits[i*bf16Size], logits[i*bf16Size+1]), history, params.RepeatPenalty) / params.Temperature
		if v > maxV {
			maxV = v
		}
		allowed++
	}
	if allowed == 0 {
		return 0, fmt.Errorf("all tokens suppressed")
	}
	var sum float32
	for i := range vocab {
		if tokenSuppressed(i, params.SuppressTokens) {
			continue
		}
		v := repeatPenalizedLogitForTest(int32(i), bf16ToF32(logits[i*bf16Size], logits[i*bf16Size+1]), history, params.RepeatPenalty)
		p := float32(math.Exp(float64(v/params.Temperature - maxV)))
		if params.MinP > 0 && p < params.MinP {
			continue
		}
		sum += p
	}
	target := draw * sum
	var acc float32
	fallback := int32(-1)
	for i := range vocab {
		if tokenSuppressed(i, params.SuppressTokens) {
			continue
		}
		v := repeatPenalizedLogitForTest(int32(i), bf16ToF32(logits[i*bf16Size], logits[i*bf16Size+1]), history, params.RepeatPenalty)
		p := float32(math.Exp(float64(v/params.Temperature - maxV)))
		if params.MinP > 0 && p < params.MinP {
			continue
		}
		fallback = int32(i)
		acc += p
		if acc >= target {
			return int32(i), nil
		}
	}
	if fallback >= 0 {
		return fallback, nil
	}
	return 0, fmt.Errorf("empty sampled distribution")
}

func TestArchSessionSampleTopKCandidatesFromLogitsMatchesFullSampler(t *testing.T) {
	vals := []float32{-3, 4, 0.5, 2, 4, -1, 3, 1, 2.5}
	logits := toBF16Bytes(vals)
	params := model.SampleParams{
		Temperature:    0.8,
		TopK:           4,
		TopP:           0.75,
		SuppressTokens: []int32{4},
	}
	sess := &ArchSession{arch: model.Arch{Vocab: len(vals)}}
	candidateLogits, candidateIDs, ok, err := sess.sampleTopKCandidatesFromLogits(logits, params, nil)
	if err != nil {
		t.Fatalf("sampleTopKCandidatesFromLogits: %v", err)
	}
	if !ok {
		t.Fatal("sampleTopKCandidatesFromLogits declined TopK params")
	}
	wantIDs := []int32{1, 6, 8, 3}
	if !idsEqual(candidateIDs, wantIDs) {
		t.Fatalf("candidate ids = %v, want %v", candidateIDs, wantIDs)
	}
	fullSampler := model.NewSampler(123)
	candidateSampler := model.NewSampler(123)
	for i := range 32 {
		want, err := fullSampler.Sample(logits, len(vals), params)
		if err != nil {
			t.Fatalf("full sample %d: %v", i, err)
		}
		got, err := sampleSortedBF16Candidates(candidateLogits, candidateIDs, candidateSampler, params)
		if err != nil {
			t.Fatalf("candidate sample %d: %v", i, err)
		}
		if got != want {
			t.Fatalf("draw %d: candidate sample = %d, want %d", i, got, want)
		}
	}
}

// sampleSortedCandidatesFixture returns a small candidate set ALREADY sorted descending
// by value — sampleSortedBF16Candidates (unlike model.Sampler.SampleCandidates, which
// re-ranks defensively) trusts that ordering outright, exactly as the real GPU top-k
// merge kernel hands it candidates. Feeding an already-descending set to both sides
// keeps a stable re-sort a no-op, so the parity tests below never pin a byte computed
// by hand — they compare against the independent, already-trusted model.Sampler.
func sampleSortedCandidatesFixture() ([]int32, []byte) {
	ids := []int32{7, 1, 9, 3, 5}
	vals := []float32{5, 4.5, 3, 1, 0.2} // strictly descending, no ties
	return ids, toBF16Bytes(vals)
}

// TestSampleSortedBF16CandidatesGuardErrors pins the four input-validation guards ahead
// of any GPU/session state — sampleSortedBF16Candidates is pure host arithmetic over
// caller-supplied candidate bytes (#30 r5, the pipelined-GPU sampling-tail cluster).
func TestSampleSortedBF16CandidatesGuardErrors(t *testing.T) {
	oneID := []int32{5}
	oneLogit := toBF16Bytes([]float32{1})
	t.Run("nil sampler", func(t *testing.T) {
		if _, err := sampleSortedBF16Candidates(oneLogit, oneID, nil, model.SampleParams{Temperature: 1}); err == nil {
			t.Fatal("nil sampler: want error, got nil")
		}
	})
	t.Run("empty candidates", func(t *testing.T) {
		if _, err := sampleSortedBF16Candidates(nil, nil, model.NewSampler(1), model.SampleParams{Temperature: 1}); err == nil {
			t.Fatal("empty candidates: want error, got nil")
		}
	})
	t.Run("too many candidates", func(t *testing.T) {
		ids := make([]int32, headSampleTopKMaxK+1)
		vals := make([]float32, len(ids))
		for i := range ids {
			ids[i] = int32(i)
			vals[i] = float32(-i)
		}
		if _, err := sampleSortedBF16Candidates(toBF16Bytes(vals), ids, model.NewSampler(1), model.SampleParams{Temperature: 1}); err == nil {
			t.Fatal("too many candidates: want error, got nil")
		}
	})
	t.Run("logits ids length mismatch", func(t *testing.T) {
		if _, err := sampleSortedBF16Candidates(oneLogit, []int32{1, 2}, model.NewSampler(1), model.SampleParams{Temperature: 1}); err == nil {
			t.Fatal("length mismatch: want error, got nil")
		}
	})
}

// TestSampleSortedBF16CandidatesGreedyBranch pins the sampledGreedyParamsEligible arm
// (Temperature<=0, MinP<=0, RepeatPenalty<=1): a best-of-unsuppressed scan across the
// candidates, RNG-free, agreeing with model.Sampler.SampleCandidates run over the same
// already-sorted candidates.
func TestSampleSortedBF16CandidatesGreedyBranch(t *testing.T) {
	t.Run("zero temperature", func(t *testing.T) {
		ids, logits := sampleSortedCandidatesFixture()
		params := model.SampleParams{Temperature: 0}
		want, err := model.NewSampler(1).SampleCandidates(logits, ids, params)
		if err != nil {
			t.Fatalf("SampleCandidates: %v", err)
		}
		sampler := model.NewSampler(1)
		got, err := sampleSortedBF16Candidates(logits, ids, sampler, params)
		if err != nil {
			t.Fatalf("sampleSortedBF16Candidates: %v", err)
		}
		if got != want {
			t.Fatalf("greedy candidate = %d, want %d (SampleCandidates parity)", got, want)
		}
		if got != ids[0] {
			t.Fatalf("greedy candidate = %d, want dominant candidate %d", got, ids[0])
		}
		if sampler.Draw() != model.NewSampler(1).Draw() {
			t.Fatal("greedy branch consumed a draw — greedy must be RNG-free")
		}
	})
	t.Run("negative temperature", func(t *testing.T) {
		ids, logits := sampleSortedCandidatesFixture()
		params := model.SampleParams{Temperature: -1}
		want, err := model.NewSampler(2).SampleCandidates(logits, ids, params)
		if err != nil {
			t.Fatalf("SampleCandidates: %v", err)
		}
		got, err := sampleSortedBF16Candidates(logits, ids, model.NewSampler(2), params)
		if err != nil {
			t.Fatalf("sampleSortedBF16Candidates: %v", err)
		}
		if got != want {
			t.Fatalf("greedy candidate = %d, want %d (SampleCandidates parity)", got, want)
		}
	})
	t.Run("skips suppressed", func(t *testing.T) {
		ids, logits := sampleSortedCandidatesFixture()
		params := model.SampleParams{Temperature: 0, SuppressTokens: []int32{ids[0]}}
		got, err := sampleSortedBF16Candidates(logits, ids, model.NewSampler(1), params)
		if err != nil {
			t.Fatalf("sampleSortedBF16Candidates: %v", err)
		}
		if got != ids[1] {
			t.Fatalf("greedy suppressed candidate = %d, want next-highest %d", got, ids[1])
		}
	})
	t.Run("all suppressed errors", func(t *testing.T) {
		ids := []int32{7, 1}
		logits := toBF16Bytes([]float32{5, 2})
		params := model.SampleParams{Temperature: 0, SuppressTokens: []int32{7, 1}}
		if _, err := sampleSortedBF16Candidates(logits, ids, model.NewSampler(1), params); err == nil {
			t.Fatal("all candidates suppressed (greedy): want error, got nil")
		}
	})
}

// TestSampleSortedBF16CandidatesTopKOneBranch pins the params.TopK==1 arm: candidates
// arrive pre-sorted descending, so TopK==1 takes the first unsuppressed entry outright
// (no scan), but still consumes exactly one Draw so the sampler's RNG stream stays the
// same length as the general branch would have used.
func TestSampleSortedBF16CandidatesTopKOneBranch(t *testing.T) {
	t.Run("takes first candidate", func(t *testing.T) {
		ids, logits := sampleSortedCandidatesFixture()
		params := model.SampleParams{Temperature: 1, TopK: 1}
		sampler := model.NewSampler(1)
		got, err := sampleSortedBF16Candidates(logits, ids, sampler, params)
		if err != nil {
			t.Fatalf("sampleSortedBF16Candidates: %v", err)
		}
		if got != ids[0] {
			t.Fatalf("TopK==1 candidate = %d, want first candidate %d", got, ids[0])
		}
		control := model.NewSampler(1)
		control.Draw() // the one draw sampleSortedBF16Candidates should have consumed
		if sampler.Draw() != control.Draw() {
			t.Fatal("TopK==1 branch did not consume exactly one draw")
		}
	})
	t.Run("skips suppressed", func(t *testing.T) {
		ids, logits := sampleSortedCandidatesFixture()
		params := model.SampleParams{Temperature: 1, TopK: 1, SuppressTokens: []int32{ids[0]}}
		got, err := sampleSortedBF16Candidates(logits, ids, model.NewSampler(1), params)
		if err != nil {
			t.Fatalf("sampleSortedBF16Candidates: %v", err)
		}
		if got != ids[1] {
			t.Fatalf("TopK==1 suppressed candidate = %d, want next %d", got, ids[1])
		}
	})
	t.Run("all suppressed errors", func(t *testing.T) {
		ids := []int32{7, 1}
		logits := toBF16Bytes([]float32{5, 2})
		params := model.SampleParams{Temperature: 1, TopK: 1, SuppressTokens: []int32{7, 1}}
		if _, err := sampleSortedBF16Candidates(logits, ids, model.NewSampler(1), params); err == nil {
			t.Fatal("all candidates suppressed (TopK==1): want error, got nil")
		}
	})
}

// TestSampleSortedBF16CandidatesGeneralBranch pins the general temperature/top-k/top-p/
// min-p arm against model.Sampler.SampleCandidates over the SAME already-sorted
// candidates across several seeds — including the Temperature<=0-with-MinP>0 edge (NOT
// greedy-eligible, since sampledGreedyParamsEligible also requires MinP<=0) that
// exercises the temp-defaults-to-1 line.
func TestSampleSortedBF16CandidatesGeneralBranch(t *testing.T) {
	ids, _ := sampleSortedCandidatesFixture()
	for _, tc := range []struct {
		name   string
		params model.SampleParams
	}{
		{"temperature only", model.SampleParams{Temperature: 0.7}},
		{"topK keep window", model.SampleParams{Temperature: 0.7, TopK: 3}},
		{"topP nucleus", model.SampleParams{Temperature: 0.9, TopP: 0.6}},
		{"minP keep", model.SampleParams{Temperature: 0.9, MinP: 0.2}},
		{"topP and minP", model.SampleParams{Temperature: 0.9, TopP: 0.8, MinP: 0.05}},
		{"suppression zeroes a slot", model.SampleParams{Temperature: 0.7, SuppressTokens: []int32{ids[0]}}},
		{"non-positive temperature defaults via MinP", model.SampleParams{Temperature: 0, MinP: 0.1}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, logits := sampleSortedCandidatesFixture()
			for _, seed := range []uint64{1, 7, 42} {
				want, err := model.NewSampler(seed).SampleCandidates(logits, ids, tc.params)
				if err != nil {
					t.Fatalf("seed %d: SampleCandidates: %v", seed, err)
				}
				got, err := sampleSortedBF16Candidates(logits, ids, model.NewSampler(seed), tc.params)
				if err != nil {
					t.Fatalf("seed %d: sampleSortedBF16Candidates: %v", seed, err)
				}
				if got != want {
					t.Fatalf("seed %d: candidate = %d, want %d (SampleCandidates parity)", seed, got, want)
				}
			}
		})
	}
}

// TestSampleSortedBF16CandidatesTopKWindowFullySuppressedErrors pins the ksum==0 guard
// in the general branch: TopK truncates the kept window to candidates that are ALL
// suppressed, while a higher-value candidate OUTSIDE that window stays unsuppressed —
// so the earlier, whole-candidate-set "all suppressed" guard (computed over allowed
// counted across every candidate, not just the kept window) does not fire, and only the
// kept window's own probability mass is (correctly) recognised as empty.
func TestSampleSortedBF16CandidatesTopKWindowFullySuppressedErrors(t *testing.T) {
	ids := []int32{10, 11, 12, 99}
	logits := toBF16Bytes([]float32{9, 8, 7, 1})
	params := model.SampleParams{Temperature: 0.8, TopK: 3, SuppressTokens: []int32{10, 11, 12}}
	if _, err := sampleSortedBF16Candidates(logits, ids, model.NewSampler(1), params); err == nil {
		t.Fatal("fully-suppressed TopK window: want empty sampled distribution error, got nil")
	}
}

// TestSampleSortedBF16CandidatesGeneralAllSuppressedErrors pins the general branch's OWN
// whole-candidate-set "all suppressed" guard (allowed==0) — distinct from the greedy and
// TopK==1 branches' own copies of the same guard, since Temperature>0 with TopK!=1 takes
// neither of those and reaches this one instead.
func TestSampleSortedBF16CandidatesGeneralAllSuppressedErrors(t *testing.T) {
	ids := []int32{7, 1}
	logits := toBF16Bytes([]float32{5, 2})
	params := model.SampleParams{Temperature: 0.8, SuppressTokens: []int32{7, 1}}
	if _, err := sampleSortedBF16Candidates(logits, ids, model.NewSampler(1), params); err == nil {
		t.Fatal("all candidates suppressed (general branch): want error, got nil")
	}
}

func TestArchSessionSampleTokenFromLogitsTopKRepeatPenaltyUsesCompactCandidates(t *testing.T) {
	vals := []float32{-2.5, 3.25, 0.5, 2.75, 3.1, -1.5, 2.9, 0.25, 1.8, -0.75, 2.4}
	logits := toBF16Bytes(vals)
	params := model.SampleParams{
		Temperature:    0.9,
		TopK:           5,
		TopP:           0.82,
		SuppressTokens: []int32{4},
		RepeatPenalty:  1.4,
	}
	history := []int32{1, 1, 6, 9}
	sess := &ArchSession{arch: model.Arch{Vocab: len(vals)}}

	penalized, err := nativeApplyRepeatPenaltyBF16(logits, len(vals), history, params.RepeatPenalty)
	if err != nil {
		t.Fatalf("nativeApplyRepeatPenaltyBF16: %v", err)
	}
	want, err := model.NewSampler(77).Sample(penalized, len(vals), params)
	if err != nil {
		t.Fatalf("full penalized sample: %v", err)
	}
	got, err := sess.sampleTokenFromLogits(logits, model.NewSampler(77), params, history)
	if err != nil {
		t.Fatalf("sampleTokenFromLogits: %v", err)
	}
	if got != want {
		t.Fatalf("compact candidate sample = %d, want full penalized sample %d", got, want)
	}
	if len(sess.sampleCandidateIDs) != params.TopK {
		t.Fatalf("candidate ids len = %d, want %d", len(sess.sampleCandidateIDs), params.TopK)
	}
	if sess.samplePenaltyLogits != nil {
		t.Fatal("TopK repeat-penalty logits path used vocab-sized repeat-penalty scratch")
	}
}

func TestRankSampleOrderPrefixTopPOnlyStopsBeforeFullVocab(t *testing.T) {
	probs := []float32{0.42, 0.24, 0.17, 0.08, 0.05, 0.04}
	order := []int32{0, 1, 2, 3, 4, 5}

	keep := rankSampleOrderPrefix(order, probs, 1, model.SampleParams{TopP: 0.7})
	if keep != 3 {
		t.Fatalf("keep = %d, want 3", keep)
	}
	if !slices.Equal(order[:keep], []int32{0, 1, 2}) {
		t.Fatalf("ranked prefix = %v, want [0 1 2]", order[:keep])
	}
}

func TestRankSampleOrderPrefixTopKTopPUsesTopKMass(t *testing.T) {
	probs := []float32{0.4, 0.25, 0.2, 0.1, 0.05}
	order := []int32{0, 1, 2, 3, 4}

	keep := rankSampleOrderPrefix(order, probs, 1, model.SampleParams{TopK: 4, TopP: 0.7})
	if keep != 3 {
		t.Fatalf("keep = %d, want 3", keep)
	}
	if !slices.Equal(order[:keep], []int32{0, 1, 2}) {
		t.Fatalf("ranked prefix = %v, want [0 1 2]", order[:keep])
	}
}

func TestRankSampleOrderPrefixFallbackSortsFullOrder(t *testing.T) {
	probs := []float32{0.2, 0.4, 0.4, 0.1}
	order := []int32{0, 1, 2, 3}

	keep := rankSampleOrderPrefix(order, probs, 1, model.SampleParams{})
	if keep != len(order) {
		t.Fatalf("keep = %d, want %d", keep, len(order))
	}
	if !slices.Equal(order, []int32{1, 2, 0, 3}) {
		t.Fatalf("ranked order = %v, want [1 2 0 3]", order)
	}
}

func TestSampleRankPrefixPreferred(t *testing.T) {
	if sampleRankPrefixPreferred(model.SampleParams{}, 8) {
		t.Fatal("plain sampling should keep full-order path")
	}
	if !sampleRankPrefixPreferred(model.SampleParams{TopK: 4}, 8) {
		t.Fatal("TopK below vocab should use prefix ranking")
	}
	if sampleRankPrefixPreferred(model.SampleParams{TopK: 8}, 8) {
		t.Fatal("TopK covering vocab should keep full-order path without another rank filter")
	}
	if !sampleRankPrefixPreferred(model.SampleParams{TopP: 0.9}, 8) {
		t.Fatal("TopP should use prefix ranking")
	}
	if !sampleRankPrefixPreferred(model.SampleParams{MinP: 0.05}, 8) {
		t.Fatal("MinP should use prefix ranking")
	}
}

func TestArchSessionSampleVocabLargeRankedSamplerMatchesModel(t *testing.T) {
	vals := []float32{
		-2.0, 1.5, 0.25, 3.0, 3.0, -0.5, 2.25, 0.75,
		1.0, -1.0, 2.0, 1.25, -3.0, 0.5, 1.75, 2.5,
		0.0, 1.5, -0.25, 2.75, 0.33, 0.66, 1.99, -1.5,
		2.1, 0.9, 1.1, -0.75, 0.45, 2.35, 1.65, -2.5,
		0.12, 2.6, 1.4, -0.1, 0.8, 2.8, 1.9, 0.3,
		-1.2, 2.2, 1.8, 0.6, 2.4, -0.6, 1.2, 0.2,
		2.7, 1.7, -0.3, 0.4, 2.05, 1.05, -2.2, 0.55,
		2.15, 1.35, -0.45, 0.95, 2.45, 1.55, -1.8, 0.15,
		2.65, 1.85, -0.15, 0.85, 2.95, 1.95, -1.1, 0.05,
	}
	logits := toBF16Bytes(vals)
	params := model.SampleParams{
		Temperature:    0.85,
		TopP:           0.76,
		MinP:           0.02,
		SuppressTokens: []int32{4, 19, 68},
	}
	sess := &ArchSession{}
	for seed := uint64(1); seed <= 32; seed++ {
		want, err := model.NewSampler(seed).Sample(logits, len(vals), params)
		if err != nil {
			t.Fatalf("model sample seed %d: %v", seed, err)
		}
		got, err := sess.sampleVocabBF16(logits, len(vals), model.NewSampler(seed), params)
		if err != nil {
			t.Fatalf("native sample seed %d: %v", seed, err)
		}
		if got != want {
			t.Fatalf("seed %d native sample = %d, want model sample %d", seed, got, want)
		}
	}
}

func TestArchSessionSampleVocabLargeRankedSamplerAvoidsProbabilityScratch(t *testing.T) {
	const vocab = 72
	vals := make([]float32, vocab)
	for i := range vals {
		vals[i] = float32(i%9) * 0.125
	}
	logits := toBF16Bytes(vals)
	params := model.SampleParams{Temperature: 1, TopP: 0.7, MinP: 0.05}
	sess := &ArchSession{sampleProbs: make([]float32, vocab)}

	want, err := model.NewSampler(7).Sample(logits, vocab, params)
	if err != nil {
		t.Fatalf("model sample: %v", err)
	}
	got, err := sess.sampleVocabBF16(logits, vocab, model.NewSampler(7), params)
	if err != nil {
		t.Fatalf("native sample: %v", err)
	}
	if got != want {
		t.Fatalf("native sample = %d, want model sample %d", got, want)
	}
	orderScratchBytes := cap(sess.sampleOrder) * int(unsafe.Sizeof(sess.sampleOrder[0]))
	if orderScratchBytes > vocab*4 {
		t.Fatalf("native ranked order scratch bytes = %d, want <= %d", orderScratchBytes, vocab*4)
	}
	if cap(sess.sampleScaled) != 0 {
		t.Fatalf("native ranked sampler retained scaled scratch cap = %d, want 0", cap(sess.sampleScaled))
	}
	if cap(sess.sampleProbs) != 0 {
		t.Fatalf("native ranked sampler probability scratch cap = %d, want 0", cap(sess.sampleProbs))
	}
}

func TestArchSessionSampleVocabLargeTopKTopPAvoidsProbabilityScratch(t *testing.T) {
	const vocab = 96
	vals := make([]float32, vocab)
	for i := range vals {
		vals[i] = float32((i*23)%41-11) * 0.07
	}
	logits := toBF16Bytes(vals)
	params := model.SampleParams{Temperature: 0.9, TopK: 17, TopP: 0.64, MinP: 0.03, SuppressTokens: []int32{5, 17, 91}}
	for seed := uint64(1); seed <= 16; seed++ {
		sess := &ArchSession{sampleProbs: make([]float32, vocab)}
		want, err := model.NewSampler(seed).Sample(logits, vocab, params)
		if err != nil {
			t.Fatalf("model sample seed %d: %v", seed, err)
		}
		got, err := sess.sampleVocabBF16(logits, vocab, model.NewSampler(seed), params)
		if err != nil {
			t.Fatalf("native sample seed %d: %v", seed, err)
		}
		if got != want {
			t.Fatalf("seed %d native sample = %d, want model sample %d", seed, got, want)
		}
		if cap(sess.sampleProbs) != 0 {
			t.Fatalf("seed %d native TopK+TopP probability scratch cap = %d, want 0", seed, cap(sess.sampleProbs))
		}
	}
}

func TestArchSessionSampleVocabLargeMinPOnlyAvoidsProbabilityScratch(t *testing.T) {
	const vocab = 80
	vals := make([]float32, vocab)
	for i := range vals {
		vals[i] = float32((i*19)%37-8) * 0.06
	}
	logits := toBF16Bytes(vals)
	params := model.SampleParams{Temperature: 1.1, MinP: 0.08, SuppressTokens: []int32{3, 9}}
	for seed := uint64(1); seed <= 16; seed++ {
		sess := &ArchSession{sampleProbs: make([]float32, vocab)}
		want, err := model.NewSampler(seed).Sample(logits, vocab, params)
		if err != nil {
			t.Fatalf("model sample seed %d: %v", seed, err)
		}
		got, err := sess.sampleVocabBF16(logits, vocab, model.NewSampler(seed), params)
		if err != nil {
			t.Fatalf("native sample seed %d: %v", seed, err)
		}
		if got != want {
			t.Fatalf("seed %d native sample = %d, want model sample %d", seed, got, want)
		}
		if cap(sess.sampleProbs) != 0 {
			t.Fatalf("seed %d native MinP probability scratch cap = %d, want 0", seed, cap(sess.sampleProbs))
		}
	}
}

func TestArchSessionSampleVocabLargeTempOnlyAvoidsProbabilityScratch(t *testing.T) {
	const vocab = 72
	vals := make([]float32, vocab)
	for i := range vals {
		vals[i] = float32((i*17)%31) * 0.05
	}
	logits := toBF16Bytes(vals)
	params := model.SampleParams{Temperature: 1}
	sess := &ArchSession{sampleProbs: make([]float32, vocab)}

	want, err := model.NewSampler(13).Sample(logits, vocab, params)
	if err != nil {
		t.Fatalf("model sample: %v", err)
	}
	got, err := sess.sampleVocabBF16(logits, vocab, model.NewSampler(13), params)
	if err != nil {
		t.Fatalf("native sample: %v", err)
	}
	if got != want {
		t.Fatalf("native sample = %d, want model sample %d", got, want)
	}
	if cap(sess.sampleOrder) != 0 {
		t.Fatalf("native temp-only rank scratch cap = %d, want 0", cap(sess.sampleOrder))
	}
	if cap(sess.sampleProbs) != 0 {
		t.Fatalf("native temp-only probability scratch cap = %d, want 0", cap(sess.sampleProbs))
	}
}

func TestLogitsSampleTopPOnlyKernelTopK(t *testing.T) {
	params := model.SampleParams{Temperature: 1, TopP: 0.9}
	if !logitsSampleTopPOnlyFullVocab(params, headSampleTopKMaxK) {
		t.Fatal("TopP-only sampler did not accept exact ranked-window vocab")
	}
	if got := logitsSampleKernelTopK(params, headSampleTopKMaxK); got != headSampleTopKMaxK {
		t.Fatalf("TopP-only ranked-window topK = %d, want %d", got, headSampleTopKMaxK)
	}
	if !logitsSampleTopPOnlyFullVocab(params, headSampleTopKMaxK+1) {
		t.Fatal("TopP-only sampler rejected full-vocab ranked-prefix sampling above the old fixed window")
	}
	if got := logitsSampleKernelTopK(params, headSampleTopKMaxK+1); got != headSampleTopKMaxK+1 {
		t.Fatalf("large-vocab TopP-only topK = %d, want full vocab %d", got, headSampleTopKMaxK+1)
	}
}

// TestArchSession gates the persistent serving session: a second Generate continues the
// running sequence from the carried-over cache, and its output is byte-identical to a fresh
// whole-sequence generate on the concatenated history — which proves the resident caches
// SURVIVED across the constructor + per-call autorelease pools and that the continuation is
// correct. Plus: Pos tracks the sequence length, a fresh session reproduces it, and a third
// turn runs (the buffer lifetime holds across many calls).
func TestArchSession(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	const dModel, nHeads, nKV, headDim, dFF, vocab = 128, 2, 1, 64, 256, 32
	const maxLen = 32
	arch, err := g4.Config{
		HiddenSize: dModel, NumHiddenLayers: 2, IntermediateSize: dFF,
		NumAttentionHeads: nHeads, NumKeyValueHeads: nKV, HeadDim: headDim,
		VocabSize: vocab, RMSNormEps: 1e-6,
	}.Arch()
	if err != nil {
		t.Fatalf("Arch: %v", err)
	}
	mk := func(n, salt int) []float32 {
		s := make([]float32, n)
		for i := range s {
			s[i] = float32((i*salt+13)%97-48) * 0.02
		}
		return s
	}
	layers := make([]DecodeLayerWeights, len(arch.Layer))
	for li := range layers {
		layers[li] = forwardLayer(dModel, nHeads, nKV, headDim, dFF, (li+1)*100)
	}
	g := &BF16Model{Layers: layers, Embed: toBF16Bytes(mk(vocab*dModel, 11)), FinalNorm: toBF16Bytes(mk(dModel, 7))}
	g.LMHead, g.Tied = g.Embed, true

	sess, err := NewArchSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("NewArchSession: %v", err)
	}
	promptA := []int32{1, 5, 3}
	gA, err := sess.Generate(promptA, 3, -1)
	if err != nil {
		t.Fatalf("Generate A: %v", err)
	}
	if sess.Pos() != len(promptA)+len(gA) {
		t.Fatalf("Pos after turn 1 = %d, want %d", sess.Pos(), len(promptA)+len(gA))
	}
	promptB := []int32{7, 2}
	gB, err := sess.Generate(promptB, 4, -1)
	if err != nil {
		t.Fatalf("Generate B: %v", err)
	}
	if sess.Pos() != len(promptA)+len(gA)+len(promptB)+len(gB) {
		t.Fatalf("Pos after turn 2 = %d, want %d", sess.Pos(), len(promptA)+len(gA)+len(promptB)+len(gB))
	}

	// the continuation must equal a fresh whole-sequence generate on the full history.
	concat := append(append(append([]int32{}, promptA...), gA...), promptB...)
	ref, err := GenerateBF16(g, arch, concat, 4, maxLen, -1)
	if err != nil {
		t.Fatalf("reference GenerateBF16: %v", err)
	}
	if !idsEqual(gB, ref) {
		t.Fatalf("session continuation %v != fresh whole-sequence %v (cache did not carry over correctly)", gB, ref)
	}

	// a fresh session reproduces both turns (deterministic).
	sess2, err := NewArchSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("NewArchSession 2: %v", err)
	}
	gA2, _ := sess2.Generate(promptA, 3, -1)
	gB2, _ := sess2.Generate(promptB, 4, -1)
	if !idsEqual(gA2, gA) || !idsEqual(gB2, gB) {
		t.Fatalf("non-deterministic across sessions: A %v vs %v, B %v vs %v", gA2, gA, gB2, gB)
	}

	// a third turn runs (buffer lifetime holds across many calls).
	gC, err := sess.Generate([]int32{9}, 3, -1)
	if err != nil {
		t.Fatalf("Generate C: %v", err)
	}
	if len(gC) != 3 || sess.Pos() != 16 {
		t.Fatalf("turn 3: got %d tokens, Pos %d (want 3, 16)", len(gC), sess.Pos())
	}

	t.Logf("session: turn1 %v → turn2 %v continues the cache (≡ fresh whole-sequence on the 8-token history), turn3 %v; Pos %d; deterministic — persistent KV cache survives across calls", gA, gB, gC, sess.Pos())
}

func TestArchSessionGenerateWithSuppression(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	const dModel, nHeads, nKV, headDim, dFF, vocab = 128, 2, 1, 64, 256, 32
	const maxLen = 16
	arch, err := g4.Config{
		HiddenSize: dModel, NumHiddenLayers: 1, IntermediateSize: dFF,
		NumAttentionHeads: nHeads, NumKeyValueHeads: nKV, HeadDim: headDim,
		VocabSize: vocab, RMSNormEps: 1e-6,
	}.Arch()
	if err != nil {
		t.Fatalf("Arch: %v", err)
	}
	mk := func(n, salt int) []float32 {
		s := make([]float32, n)
		for i := range s {
			s[i] = float32((i*salt+13)%97-48) * 0.02
		}
		return s
	}
	layers := []DecodeLayerWeights{forwardLayer(dModel, nHeads, nKV, headDim, dFF, 100)}
	g := &BF16Model{Layers: layers, Embed: toBF16Bytes(mk(vocab*dModel, 11)), FinalNorm: toBF16Bytes(mk(dModel, 7))}
	g.LMHead, g.Tied = g.Embed, true

	sess, err := NewArchSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("NewArchSession: %v", err)
	}
	const survivor int32 = 7
	suppress := make([]int32, 0, vocab-1)
	for id := range int32(vocab) {
		if id != survivor {
			suppress = append(suppress, id)
		}
	}
	got, err := sess.GenerateWithSuppression([]int32{1, 5, 3}, 1, -1, suppress)
	if err != nil {
		t.Fatalf("GenerateWithSuppression: %v", err)
	}
	if !idsEqual(got, []int32{survivor}) {
		t.Fatalf("GenerateWithSuppression = %v, want lone unsuppressed token %d", got, survivor)
	}
}

func TestArchSessionGenerateEachStopsAfterFirstYield(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	const dModel, nHeads, nKV, headDim, dFF, vocab = 128, 2, 1, 64, 256, 32
	const maxLen = 16
	arch, err := g4.Config{
		HiddenSize: dModel, NumHiddenLayers: 1, IntermediateSize: dFF,
		NumAttentionHeads: nHeads, NumKeyValueHeads: nKV, HeadDim: headDim,
		VocabSize: vocab, RMSNormEps: 1e-6,
	}.Arch()
	if err != nil {
		t.Fatalf("Arch: %v", err)
	}
	mk := func(n, salt int) []float32 {
		s := make([]float32, n)
		for i := range s {
			s[i] = float32((i*salt+13)%97-48) * 0.02
		}
		return s
	}
	layers := []DecodeLayerWeights{forwardLayer(dModel, nHeads, nKV, headDim, dFF, 100)}
	g := &BF16Model{Layers: layers, Embed: toBF16Bytes(mk(vocab*dModel, 11)), FinalNorm: toBF16Bytes(mk(dModel, 7))}
	g.LMHead, g.Tied = g.Embed, true

	sess, err := NewArchSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("NewArchSession: %v", err)
	}
	prompt := []int32{1, 5, 3}
	var yielded []int32
	gen, err := sess.GenerateEach(prompt, 4, -1, func(id int32) bool {
		yielded = append(yielded, id)
		return false
	})
	if err != nil {
		t.Fatalf("GenerateEach: %v", err)
	}
	if len(gen) != 1 || !idsEqual(gen, yielded) {
		t.Fatalf("GenerateEach gen/yielded = %v/%v, want one matching streamed token", gen, yielded)
	}
	if sess.Pos() != len(prompt)+1 {
		t.Fatalf("Pos after stopped stream = %d, want prompt plus one generated token (%d)", sess.Pos(), len(prompt)+1)
	}
}

func TestArchSessionGenerateSampledEachStopsAndCachesFinalToken(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	const dModel, nHeads, nKV, headDim, dFF, vocab = 128, 2, 1, 64, 256, 32
	const maxLen = 16
	g, arch := gemma4BF16Fixture(t, dModel, nHeads, nKV, headDim, dFF, vocab, 1)
	sess, err := NewArchSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("NewArchSession: %v", err)
	}
	const survivor int32 = 7
	suppress := make([]int32, 0, vocab-1)
	for id := range int32(vocab) {
		if id != survivor {
			suppress = append(suppress, id)
		}
	}
	var yielded []int32
	got, err := sess.GenerateSampledEach([]int32{1, 5, 3}, 4, []int32{survivor}, model.NewSampler(1), model.SampleParams{Temperature: 0, SuppressTokens: suppress}, nil, func(id int32) bool {
		yielded = append(yielded, id)
		return true
	})
	if err != nil {
		t.Fatalf("GenerateSampledEach: %v", err)
	}
	if !idsEqual(got, []int32{survivor}) || !idsEqual(yielded, got) {
		t.Fatalf("GenerateSampledEach got/yielded = %v/%v, want [%d]", got, yielded, survivor)
	}
	if sess.Pos() != 4 {
		t.Fatalf("Pos after sampled stop = %d, want prompt plus cached final token (4)", sess.Pos())
	}
}

func TestArchSessionGenerateSampledOneShotEachStopsWithoutCachingFinalToken(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	const dModel, nHeads, nKV, headDim, dFF, vocab = 128, 2, 1, 64, 256, 32
	const maxLen = 16
	g, arch := gemma4BF16Fixture(t, dModel, nHeads, nKV, headDim, dFF, vocab, 1)
	sess, err := NewArchSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("NewArchSession: %v", err)
	}
	const survivor int32 = 7
	suppress := make([]int32, 0, vocab-1)
	for id := range int32(vocab) {
		if id != survivor {
			suppress = append(suppress, id)
		}
	}
	var yielded []int32
	got, err := sess.GenerateSampledOneShotEach([]int32{1, 5, 3}, 4, []int32{survivor}, model.NewSampler(1), model.SampleParams{Temperature: 0, SuppressTokens: suppress}, nil, func(id int32) bool {
		yielded = append(yielded, id)
		return true
	})
	if err != nil {
		t.Fatalf("GenerateSampledOneShotEach: %v", err)
	}
	if !idsEqual(got, []int32{survivor}) || !idsEqual(yielded, got) {
		t.Fatalf("GenerateSampledOneShotEach got/yielded = %v/%v, want [%d]", got, yielded, survivor)
	}
	if sess.Pos() != 3 {
		t.Fatalf("Pos after sampled one-shot stop = %d, want prompt only (3)", sess.Pos())
	}
}

func TestArchSessionGenerateSampledReusesHistoryScratch(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	const dModel, nHeads, nKV, headDim, dFF, vocab = 128, 2, 1, 64, 256, 64
	const maxLen, maxNew = 32, 3
	g, arch := gemma4BF16Fixture(t, dModel, nHeads, nKV, headDim, dFF, vocab, 2)
	sess, err := NewArchSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("NewArchSession: %v", err)
	}
	params := model.SampleParams{Temperature: 0.8, RepeatPenalty: 1.2}
	if _, err := sess.GenerateSampledEach([]int32{1}, maxNew, nil, model.NewSampler(1), params, nil, nil); err != nil {
		t.Fatalf("first GenerateSampledEach: %v", err)
	}
	if len(sess.sampleHistory) != maxNew {
		t.Fatalf("first sampled history length = %d, want %d", len(sess.sampleHistory), maxNew)
	}
	firstPtr := unsafe.Pointer(&sess.sampleHistory[0])
	sess.sampleHistory[0] = -12345

	if _, err := sess.GenerateSampledEach([]int32{5}, maxNew, nil, model.NewSampler(2), params, nil, nil); err != nil {
		t.Fatalf("second GenerateSampledEach: %v", err)
	}
	if len(sess.sampleHistory) != maxNew {
		t.Fatalf("second sampled history length = %d, want %d", len(sess.sampleHistory), maxNew)
	}
	if unsafe.Pointer(&sess.sampleHistory[0]) != firstPtr {
		t.Fatal("sampled repeat-penalty history allocated a new backing buffer")
	}
	if sess.sampleHistory[0] == -12345 {
		t.Fatal("sampled repeat-penalty history was not refreshed for the second generation")
	}
}

func TestArchSessionGenerateSampledSkipsHistoryScratchWithoutRepeatPenalty(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	const dModel, nHeads, nKV, headDim, dFF, vocab = 128, 2, 1, 64, 256, 64
	const maxLen, maxNew = 32, 3
	g, arch := gemma4BF16Fixture(t, dModel, nHeads, nKV, headDim, dFF, vocab, 2)
	sess, err := NewArchSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("NewArchSession: %v", err)
	}
	params := model.SampleParams{Temperature: 0.8, TopK: 7}
	if _, err := sess.GenerateSampledEach([]int32{1}, maxNew, nil, model.NewSampler(1), params, nil, nil); err != nil {
		t.Fatalf("GenerateSampledEach: %v", err)
	}
	if len(sess.sampleHistory) != 0 || cap(sess.sampleHistory) != 0 {
		t.Fatalf("sampled history scratch allocated without repeat penalty: len=%d cap=%d", len(sess.sampleHistory), cap(sess.sampleHistory))
	}
}

func TestArchSessionRepeatPenaltyScratchReusesBacking(t *testing.T) {
	const vocab = 8
	logits := toBF16Bytes([]float32{1, -2, 3, -4, 5, -6, 7, -8})
	original := append([]byte(nil), logits...)
	history := []int32{6, 1, 6, -1, 99, 3}
	sess := &ArchSession{}

	want, err := nativeApplyRepeatPenaltyBF16(logits, vocab, history, 1.5)
	if err != nil {
		t.Fatalf("nativeApplyRepeatPenaltyBF16: %v", err)
	}
	got, err := sess.repeatPenaltyLogitsScratch(logits, vocab, history, 1.5)
	if err != nil {
		t.Fatalf("repeatPenaltyLogitsScratch: %v", err)
	}
	if !bytes.Equal(got, want) {
		t.Fatalf("scratch repeat penalty = %v, want %v", got, want)
	}
	if !bytes.Equal(logits, original) {
		t.Fatal("repeat penalty scratch mutated source logits")
	}
	if !idsEqual(sess.samplePenaltyIDs, []int32{1, 3, 6}) {
		t.Fatalf("repeat penalty id scratch = %v, want unique sorted [1 3 6]", sess.samplePenaltyIDs)
	}
	firstOutPtr := unsafe.Pointer(&got[0])
	firstIDsPtr := unsafe.Pointer(&sess.samplePenaltyIDs[0])
	got[0] = 0
	sess.samplePenaltyIDs[0] = -12345

	got, err = sess.repeatPenaltyLogitsScratch(logits, vocab, history, 1.5)
	if err != nil {
		t.Fatalf("second repeatPenaltyLogitsScratch: %v", err)
	}
	if unsafe.Pointer(&got[0]) != firstOutPtr {
		t.Fatal("repeat penalty logits scratch allocated a new backing buffer")
	}
	if unsafe.Pointer(&sess.samplePenaltyIDs[0]) != firstIDsPtr {
		t.Fatal("repeat penalty id scratch allocated a new backing buffer")
	}
	if got[0] != want[0] {
		t.Fatal("repeat penalty logits scratch did not refresh mutated contents")
	}
	if sess.samplePenaltyIDs[0] == -12345 {
		t.Fatal("repeat penalty id scratch did not refresh mutated contents")
	}
	allocs := testing.AllocsPerRun(100, func() {
		out, err := sess.repeatPenaltyLogitsScratch(logits, vocab, history, 1.5)
		if err != nil {
			t.Fatalf("repeatPenaltyLogitsScratch during alloc check: %v", err)
		}
		if len(out) != len(logits) {
			t.Fatalf("repeatPenaltyLogitsScratch length = %d, want %d", len(out), len(logits))
		}
	})
	if allocs != 0 {
		t.Fatalf("warmed repeatPenaltyLogitsScratch allocs/run = %.1f, want 0", allocs)
	}
}

func TestArchSessionHeadLogitsScratchReusesBacking(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	const dModel, nHeads, nKV, headDim, dFF, vocab = 128, 2, 1, 64, 256, 64
	const maxLen = 16
	g, arch := gemma4BF16Fixture(t, dModel, nHeads, nKV, headDim, dFF, vocab, 2)
	sess, err := NewArchSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("NewArchSession: %v", err)
	}
	if sess.headEnc == nil {
		t.Fatal("test requires resident head encoder")
	}
	firstHidden := toBF16Bytes(syntheticFloat32(dModel, 41))
	wantFirst, err := sess.head(firstHidden, false)
	if err != nil {
		t.Fatalf("fresh first head: %v", err)
	}
	gotFirst, err := sess.headLogitsScratch(firstHidden, false)
	if err != nil {
		t.Fatalf("scratch first head: %v", err)
	}
	if !bytes.Equal(gotFirst, wantFirst) {
		t.Fatal("scratch first head logits differ from fresh head")
	}
	if len(gotFirst) == 0 {
		t.Fatal("scratch first head logits are empty")
	}
	firstPtr := uintptr(unsafe.Pointer(&gotFirst[0]))
	held := [][]byte{gotFirst}

	secondHidden := toBF16Bytes(syntheticFloat32(dModel, 43))
	wantSecond, err := sess.head(secondHidden, false)
	if err != nil {
		t.Fatalf("fresh second head: %v", err)
	}
	gotSecond, err := sess.headLogitsScratch(secondHidden, false)
	if err != nil {
		t.Fatalf("scratch second head: %v", err)
	}
	if !bytes.Equal(gotSecond, wantSecond) {
		t.Fatal("scratch second head logits differ from fresh head")
	}
	secondPtr := uintptr(unsafe.Pointer(&gotSecond[0]))
	runtime.KeepAlive(held)
	if secondPtr != firstPtr {
		t.Fatalf("head logits scratch allocated a new backing buffer: first=%#x second=%#x", firstPtr, secondPtr)
	}
}

func TestArchSessionHeadGreedyFallbackUsesLogitsScratch(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	const dModel, nHeads, nKV, headDim, dFF, vocab = 128, 2, 1, 64, 256, 64
	const maxLen = 16
	g, arch := gemma4BF16Fixture(t, dModel, nHeads, nKV, headDim, dFF, vocab, 2)
	sess, err := NewArchSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("NewArchSession: %v", err)
	}
	if sess.headEnc == nil {
		t.Fatal("test requires resident head encoder")
	}
	sess.greedy = nil

	hidden := toBF16Bytes(syntheticFloat32(dModel, 47))
	wantLogits, err := sess.head(hidden, true)
	if err != nil {
		t.Fatalf("fresh head: %v", err)
	}
	want, err := greedyBF16Suppressed(wantLogits, vocab, nil)
	if err != nil {
		t.Fatalf("fresh greedy: %v", err)
	}

	got, err := sess.headGreedyOrLogits(hidden, nil, nil, nil, false)
	if err != nil {
		t.Fatalf("headGreedyOrLogits: %v", err)
	}
	if got != want {
		t.Fatalf("fallback greedy token = %d, want %d", got, want)
	}
	if len(sess.sampleHeadLogits) != vocab*bf16Size {
		t.Fatalf("fallback did not populate reusable logits scratch, len=%d", len(sess.sampleHeadLogits))
	}
	firstPtr := uintptr(unsafe.Pointer(&sess.sampleHeadLogits[0]))

	hidden = toBF16Bytes(syntheticFloat32(dModel, 49))
	if _, err := sess.headGreedyOrLogits(hidden, nil, nil, nil, false); err != nil {
		t.Fatalf("second headGreedyOrLogits: %v", err)
	}
	if got := uintptr(unsafe.Pointer(&sess.sampleHeadLogits[0])); got != firstPtr {
		t.Fatalf("fallback logits scratch backing changed: %#x != %#x", got, firstPtr)
	}
}

func TestArchSessionHeadGreedyFallbackHonoursHeadOverride(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	const dModel, nHeads, nKV, headDim, dFF, vocab = 128, 2, 1, 64, 256, 64
	const maxLen = 16
	g, arch := gemma4BF16Fixture(t, dModel, nHeads, nKV, headDim, dFF, vocab, 2)
	sess, err := NewArchSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("NewArchSession: %v", err)
	}
	sess.greedy = nil
	sess.head = func([]byte, bool) ([]byte, error) {
		return nil, errors.New("head override called")
	}

	_, err = sess.headGreedyOrLogits(toBF16Bytes(syntheticFloat32(dModel, 51)), nil, nil, nil, false)
	if err == nil || err.Error() != "head override called" {
		t.Fatalf("headGreedyOrLogits override error = %v, want head override called", err)
	}
	if sess.sampleHeadLogits != nil {
		t.Fatal("head override path populated head logits scratch")
	}
}

func TestArchSessionHeadGreedyUsesRetainedHiddenNoCopyBuffer(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	const dModel, nHeads, nKV, headDim, dFF, vocab = 128, 2, 1, 64, 256, 64
	g, arch := gemma4BF16Fixture(t, dModel, nHeads, nKV, headDim, dFF, vocab, 2)
	sess, err := NewArchSession(g, arch, 16)
	if err != nil {
		t.Fatalf("NewArchSession: %v", err)
	}
	if !sess.canUseDirectHeadGreedy() {
		t.Fatal("session fixture cannot use default resident direct greedy")
	}
	if err := sess.PrefillTokens([]int32{1, 5, 3}); err != nil {
		t.Fatalf("PrefillTokens: %v", err)
	}
	if sess.retainedHiddenBuffer() == nil {
		t.Fatal("retained hidden did not expose no-copy buffer")
	}
	sentinel, err := newHeadHiddenScratch(len(sess.retainedHidden))
	if err != nil {
		t.Fatalf("newHeadHiddenScratch: %v", err)
	}
	defer sentinel.Close()
	for i := range sentinel.pinned.bytes {
		sentinel.pinned.bytes[i] = 0xa5
	}
	sess.headEnc.hiddenScratch.Put(sentinel)

	if _, err := sess.headGreedyOrLogits(sess.retainedHidden, nil, nil, nil, false); err != nil {
		t.Fatalf("headGreedyOrLogits: %v", err)
	}
	gotScratch, _ := sess.headEnc.hiddenScratch.Get().(*headHiddenScratch)
	if gotScratch != sentinel {
		t.Fatalf("retained-hidden greedy path consumed unexpected hidden scratch %p, want sentinel %p", gotScratch, sentinel)
	}
	if bytes.Equal(gotScratch.pinned.bytes, sess.retainedHidden) {
		t.Fatal("retained-hidden greedy path copied hidden into head scratch; want direct no-copy buffer")
	}
}

func TestArchSessionGenerateFirstHeadUsesRetainedPromptHiddenNoCopy(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	const dModel, nHeads, nKV, headDim, dFF, vocab = 128, 2, 1, 64, 256, 64
	g, arch := gemma4BF16Fixture(t, dModel, nHeads, nKV, headDim, dFF, vocab, 2)
	sess, err := NewArchSession(g, arch, 16)
	if err != nil {
		t.Fatalf("NewArchSession: %v", err)
	}
	if !sess.canUseDirectHeadGreedy() {
		t.Fatal("session fixture cannot use default resident direct greedy")
	}
	sentinel, err := newHeadHiddenScratch(dModel * bf16Size)
	if err != nil {
		t.Fatalf("newHeadHiddenScratch: %v", err)
	}
	defer sentinel.Close()
	for i := range sentinel.pinned.bytes {
		sentinel.pinned.bytes[i] = 0xa5
	}
	sess.headEnc.hiddenScratch.Put(sentinel)
	forceNativeGC()
	forceNativeGC()

	if _, err := sess.Generate([]int32{1, 5, 3}, 1, -1); err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if sess.retainedHiddenBuffer() == nil {
		t.Fatal("Generate did not retain prompt/generated boundary hidden in a no-copy buffer")
	}
	gotScratch, _ := sess.headEnc.hiddenScratch.Get().(*headHiddenScratch)
	if gotScratch != sentinel {
		t.Fatalf("Generate first head consumed unexpected hidden scratch %p, want sentinel %p", gotScratch, sentinel)
	}
	for i, b := range gotScratch.pinned.bytes {
		if b != 0xa5 {
			t.Fatalf("Generate first head copied prompt hidden into head scratch at byte %d", i)
		}
	}
}

func TestArchSessionHeadGreedyFreshAllocationBudget(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	const dModel, nHeads, nKV, headDim, dFF, vocab = 128, 2, 1, 64, 256, 64
	g, arch := gemma4BF16Fixture(t, dModel, nHeads, nKV, headDim, dFF, vocab, 2)
	sess, err := NewArchSession(g, arch, 16)
	if err != nil {
		t.Fatalf("NewArchSession: %v", err)
	}
	hidden := toBF16Bytes(syntheticFloat32(dModel, 52))
	if _, err := sess.headGreedyOrLogits(hidden, nil, nil, nil, false); err != nil {
		t.Fatalf("headGreedyOrLogits warmup: %v", err)
	}
	allocs := testing.AllocsPerRun(5, func() {
		if _, err := sess.headGreedyOrLogits(hidden, nil, nil, nil, false); err != nil {
			t.Fatalf("headGreedyOrLogits: %v", err)
		}
	})
	if allocs > 170 {
		t.Fatalf("fresh hidden greedy allocations = %.0f, want <= 170", allocs)
	}
}

func TestArchSessionSuppressionScratchReusesBacking(t *testing.T) {
	base := []int32{2, 7}
	extra := []int32{7, 9, 11}
	want := nativeAppendSuppressionTokens(base, extra)
	sess := &ArchSession{}

	got := sess.suppressionTokensScratch(base, extra)
	if !idsEqual(got, want) {
		t.Fatalf("suppression scratch = %v, want %v", got, want)
	}
	if !idsEqual(base, []int32{2, 7}) {
		t.Fatalf("suppression scratch mutated base tokens: %v", base)
	}
	firstPtr := unsafe.Pointer(&got[0])
	got[0] = -12345

	got = sess.suppressionTokensScratch(base, extra)
	if unsafe.Pointer(&got[0]) != firstPtr {
		t.Fatal("suppression scratch allocated a new backing buffer")
	}
	if !idsEqual(got, want) {
		t.Fatalf("suppression scratch after reuse = %v, want %v", got, want)
	}
	if got[0] == -12345 {
		t.Fatal("suppression scratch did not refresh mutated contents")
	}
	allocs := testing.AllocsPerRun(100, func() {
		got := sess.suppressionTokensScratch(base, extra)
		if len(got) != len(want) {
			t.Fatalf("suppression scratch length = %d, want %d", len(got), len(want))
		}
	})
	if allocs != 0 {
		t.Fatalf("warmed suppressionTokensScratch allocs/run = %.1f, want 0", allocs)
	}
}

func TestArchSessionSuppressionScratchReusesExtraWhenBaseEmpty(t *testing.T) {
	extra := []int32{9, 11, 13}
	sess := &ArchSession{}

	got := sess.suppressionTokensScratch(nil, extra)
	if !idsEqual(got, extra) {
		t.Fatalf("suppression scratch = %v, want %v", got, extra)
	}
	if len(got) == 0 {
		t.Fatal("suppression scratch unexpectedly empty")
	}
	if unsafe.Pointer(&got[0]) != unsafe.Pointer(&extra[0]) {
		t.Fatal("suppression scratch copied stop tokens when base list was empty")
	}
	if cap(sess.sampleSuppressTokens) != 0 {
		t.Fatalf("session suppression scratch allocated with empty base: cap=%d", cap(sess.sampleSuppressTokens))
	}
	allocs := testing.AllocsPerRun(100, func() {
		got := sess.suppressionTokensScratch(nil, extra)
		if len(got) != len(extra) {
			t.Fatalf("suppression scratch length = %d, want %d", len(got), len(extra))
		}
	})
	if allocs != 0 {
		t.Fatalf("base-empty suppressionTokensScratch allocs/run = %.1f, want 0", allocs)
	}
}

func TestArchSessionSuppressionScratchReusesBaseWhenExtraAlreadySuppressed(t *testing.T) {
	base := []int32{2, 7, 11}
	extra := []int32{7, 2}
	sess := &ArchSession{}

	got := sess.suppressionTokensScratch(base, extra)
	if !idsEqual(got, base) {
		t.Fatalf("suppression scratch = %v, want %v", got, base)
	}
	if len(got) == 0 {
		t.Fatal("suppression scratch unexpectedly empty")
	}
	if unsafe.Pointer(&got[0]) != unsafe.Pointer(&base[0]) {
		t.Fatal("suppression scratch copied base tokens when extras were already suppressed")
	}
	if cap(sess.sampleSuppressTokens) != 0 {
		t.Fatalf("session suppression scratch allocated with covered extras: cap=%d", cap(sess.sampleSuppressTokens))
	}
	allocs := testing.AllocsPerRun(100, func() {
		got := sess.suppressionTokensScratch(base, extra)
		if len(got) != len(base) {
			t.Fatalf("suppression scratch length = %d, want %d", len(got), len(base))
		}
	})
	if allocs != 0 {
		t.Fatalf("covered-extra suppressionTokensScratch allocs/run = %.1f, want 0", allocs)
	}
}

func TestArchSessionGenerateSampledZeroTempMatchesSuppressedGreedy(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	const dModel, nHeads, nKV, headDim, dFF, vocab = 128, 2, 1, 64, 256, 64
	const maxLen, maxNew = 16, 5
	g, arch := gemma4BF16Fixture(t, dModel, nHeads, nKV, headDim, dFF, vocab, 2)
	sampled, err := NewArchSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("NewArchSession sampled: %v", err)
	}
	greedy, err := NewArchSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("NewArchSession greedy: %v", err)
	}
	prompt := []int32{1, 5, 3}
	suppress := []int32{2, 7}
	got, err := sampled.GenerateSampledEach(prompt, maxNew, nil, model.NewSampler(1), model.SampleParams{Temperature: 0, SuppressTokens: suppress}, nil, nil)
	if err != nil {
		t.Fatalf("GenerateSampledEach zero-temp: %v", err)
	}
	want, err := greedy.GenerateWithSuppression(prompt, maxNew, -1, suppress)
	if err != nil {
		t.Fatalf("GenerateWithSuppression: %v", err)
	}
	if !idsEqual(got, want) {
		t.Fatalf("zero-temp sampled = %v, want suppressed greedy %v", got, want)
	}
	if sampled.Pos() != greedy.Pos() {
		t.Fatalf("positions diverged: sampled=%d greedy=%d", sampled.Pos(), greedy.Pos())
	}
}

func TestArchSessionGenerateSampledTopKOneAvoidsTopKScratch(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	const dModel, nHeads, nKV, headDim, dFF, vocab = 128, 2, 1, 64, 256, 64
	const maxLen = 16
	g, arch := gemma4BF16Fixture(t, dModel, nHeads, nKV, headDim, dFF, vocab, 2)
	sess, err := NewArchSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("NewArchSession: %v", err)
	}
	// Pins the HOST sampled-token lane's TopK=1 contract (exactly one RNG draw, no
	// TopK scratch). bf16 sessions record the arch ICB now and sample through the
	// candidates lane, whose draw accounting is its own contract — force the host lane.
	sess.state.icb = nil
	params := model.SampleParams{Temperature: 1, TopK: 1, TopP: 0.75, MinP: 0.05, SuppressTokens: []int32{2, 7}}
	if !sess.sampleTopKTokenParamsEligible(params) {
		t.Skip("device TopK sampled-token path unavailable")
	}

	sampler := model.NewSampler(123)
	wantSampler := model.NewSampler(123)
	wantSampler.Draw()
	got, err := sess.GenerateSampledEach([]int32{1, 5, 3}, 1, nil, sampler, params, nil, nil)
	if err != nil {
		t.Fatalf("GenerateSampledEach TopK=1: %v", err)
	}
	if len(got) != 1 {
		t.Fatalf("GenerateSampledEach TopK=1 returned %d tokens, want 1: %v", len(got), got)
	}
	if nativeTokenInSet(got[0], params.SuppressTokens) {
		t.Fatalf("GenerateSampledEach TopK=1 returned suppressed token %d", got[0])
	}
	if next, want := sampler.Draw(), wantSampler.Draw(); next != want {
		t.Fatalf("TopK=1 sampled session consumed wrong RNG count: next draw %.8f, want %.8f", next, want)
	}
	if scratch := sess.headEnc.topKScratch.Get(); scratch != nil {
		t.Fatalf("TopK=1 sampled session used TopK scratch: %T", scratch)
	}
}

func TestArchSessionGenerateSampledTopKOneRepeatPenaltyEmptyHistoryAvoidsTopKScratch(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	const dModel, nHeads, nKV, headDim, dFF, vocab = 128, 2, 1, 64, 256, 64
	const maxLen = 16
	g, arch := gemma4BF16Fixture(t, dModel, nHeads, nKV, headDim, dFF, vocab, 2)
	sess, err := NewArchSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("NewArchSession: %v", err)
	}
	params := model.SampleParams{Temperature: 1, TopK: 1, RepeatPenalty: 1.2, SuppressTokens: []int32{2, 7}}
	if !sess.sampleTopKTokenParamsEligible(params) {
		t.Skip("device TopK sampled-token path unavailable")
	}

	sampler := model.NewSampler(456)
	wantSampler := model.NewSampler(456)
	wantSampler.Draw()
	got, err := sess.GenerateSampledOneShotEach([]int32{1, 5, 3}, 1, nil, sampler, params, nil, nil)
	if err != nil {
		t.Fatalf("GenerateSampledOneShotEach TopK=1 repeat-penalty empty history: %v", err)
	}
	if len(got) != 1 {
		t.Fatalf("GenerateSampledOneShotEach TopK=1 returned %d tokens, want 1: %v", len(got), got)
	}
	if nativeTokenInSet(got[0], params.SuppressTokens) {
		t.Fatalf("GenerateSampledOneShotEach TopK=1 returned suppressed token %d", got[0])
	}
	if next, want := sampler.Draw(), wantSampler.Draw(); next != want {
		t.Fatalf("TopK=1 repeat-penalty empty-history session consumed wrong RNG count: next draw %.8f, want %.8f", next, want)
	}
	if scratch := sess.headEnc.topKScratch.Get(); scratch != nil {
		t.Fatalf("TopK=1 repeat-penalty empty-history session used TopK scratch: %T", scratch)
	}
}

func TestArchSessionSampleTopKTopPMatchesFullHead(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	const dModel, nHeads, nKV, headDim, dFF, vocab = 128, 2, 1, 64, 256, 64
	const maxLen = 16
	g, arch := gemma4BF16Fixture(t, dModel, nHeads, nKV, headDim, dFF, vocab, 2)
	sess, err := NewArchSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("NewArchSession: %v", err)
	}
	if err := sess.PrefillTokens([]int32{1, 5, 3}); err != nil {
		t.Fatalf("PrefillTokens: %v", err)
	}
	hidden := append([]byte(nil), sess.retainedHidden...)
	params := model.SampleParams{Temperature: 1, TopK: 5, TopP: 0.5, SuppressTokens: []int32{2, 7}}
	full, err := sess.head(hidden, false)
	if err != nil {
		t.Fatalf("head: %v", err)
	}
	want, err := model.NewSampler(123).Sample(full, arch.Vocab, params)
	if err != nil {
		t.Fatalf("full Sample: %v", err)
	}
	candidateLogits, candidateIDs, ok, err := sess.sampleTopKCandidatesFromHiddenInPool(hidden, params)
	if err != nil {
		t.Fatalf("sampleTopKCandidatesFromHiddenInPool: %v", err)
	}
	if !ok {
		t.Skip("head top-k custom kernel unavailable")
	}
	got, err := model.NewSampler(123).SampleCandidates(candidateLogits, candidateIDs, params)
	if err != nil {
		t.Fatalf("candidate SampleCandidates: %v", err)
	}
	if got != want {
		t.Fatalf("TopK+TopP candidate sample = %d, want full-head sample %d (ids %v)", got, want, candidateIDs)
	}

	draw := model.NewSampler(123).Draw()
	deviceGot, ok, err := sess.sampleTopKTokenFromHiddenInPool(hidden, params, draw, nil)
	if err != nil {
		t.Fatalf("sampleTopKTokenFromHiddenInPool: %v", err)
	}
	if !ok {
		t.Skip("device TopK sampler unavailable")
	}
	if deviceGot != want {
		t.Fatalf("device TopK+TopP sample = %d, want candidate/full-head sample %d (ids %v)", deviceGot, want, candidateIDs)
	}
}

func TestArchSessionSampleTopKCandidatesRepeatPenaltyEmptyHistoryDoesNotDecline(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	const dModel, nHeads, nKV, headDim, dFF, vocab = 128, 2, 1, 64, 256, 64
	const maxLen = 16
	g, arch := gemma4BF16Fixture(t, dModel, nHeads, nKV, headDim, dFF, vocab, 2)
	sess, err := NewArchSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("NewArchSession: %v", err)
	}
	if err := sess.PrefillTokens([]int32{1, 5, 3}); err != nil {
		t.Fatalf("PrefillTokens: %v", err)
	}
	hidden := append([]byte(nil), sess.retainedHidden...)
	params := model.SampleParams{Temperature: 1, TopK: 5, TopP: 0.5, SuppressTokens: []int32{2, 7}, RepeatPenalty: 1.2}
	full, err := sess.head(hidden, false)
	if err != nil {
		t.Fatalf("head: %v", err)
	}
	want, err := model.NewSampler(123).Sample(full, arch.Vocab, params)
	if err != nil {
		t.Fatalf("full Sample: %v", err)
	}
	candidateLogits, candidateIDs, ok, err := sess.sampleTopKCandidatesFromHiddenInPool(hidden, params)
	if err != nil {
		t.Fatalf("sampleTopKCandidatesFromHiddenInPool: %v", err)
	}
	if !ok {
		t.Fatal("TopK candidate path declined repeat-penalty params with empty history")
	}
	got, err := model.NewSampler(123).SampleCandidates(candidateLogits, candidateIDs, params)
	if err != nil {
		t.Fatalf("candidate SampleCandidates: %v", err)
	}
	if got != want {
		t.Fatalf("TopK+TopP repeat-penalty empty-history candidate sample = %d, want full-head sample %d (ids %v)", got, want, candidateIDs)
	}
}

func TestArchSessionSampleTopKCandidatesRepeatPenaltyHistoryMatchesFullHead(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	const dModel, nHeads, nKV, headDim, dFF, vocab = 128, 2, 1, 64, 256, 64
	const maxLen = 16
	g, arch := gemma4BF16Fixture(t, dModel, nHeads, nKV, headDim, dFF, vocab, 2)
	sess, err := NewArchSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("NewArchSession: %v", err)
	}
	if err := sess.PrefillTokens([]int32{1, 5, 3}); err != nil {
		t.Fatalf("PrefillTokens: %v", err)
	}
	hidden := append([]byte(nil), sess.retainedHidden...)
	params := model.SampleParams{Temperature: 1, TopK: 7, TopP: 0.75, SuppressTokens: []int32{2, 7}, RepeatPenalty: 1.2}
	history := []int32{4, 5, 5, 31}
	full, err := sess.head(hidden, false)
	if err != nil {
		t.Fatalf("head: %v", err)
	}
	penalized, err := nativeApplyRepeatPenaltyBF16(full, arch.Vocab, history, params.RepeatPenalty)
	if err != nil {
		t.Fatalf("nativeApplyRepeatPenaltyBF16: %v", err)
	}
	want, err := model.NewSampler(123).Sample(penalized, arch.Vocab, params)
	if err != nil {
		t.Fatalf("penalized full Sample: %v", err)
	}
	candidateLogits, candidateIDs, ok, err := sess.sampleTopKCandidatesFromHiddenWithHistoryInPool(hidden, params, history)
	if err != nil {
		t.Fatalf("sampleTopKCandidatesFromHiddenWithHistoryInPool: %v", err)
	}
	if !ok {
		t.Fatal("TopK candidate path declined repeat-penalty params with history")
	}
	got, err := model.NewSampler(123).SampleCandidates(candidateLogits, candidateIDs, params)
	if err != nil {
		t.Fatalf("candidate SampleCandidates: %v", err)
	}
	if got != want {
		t.Fatalf("TopK+TopP repeat-penalty history candidate sample = %d, want full-head sample %d (ids %v)", got, want, candidateIDs)
	}
	if sess.samplePenaltyLogits != nil {
		t.Fatal("TopK candidate repeat-penalty path used vocab-sized repeat-penalty scratch")
	}
}

func TestArchSessionSampleTopKCandidatesUsesRetainedHiddenNoCopyBuffer(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	const dModel, nHeads, nKV, headDim, dFF, vocab = 128, 2, 1, 64, 256, 64
	g, arch := gemma4BF16Fixture(t, dModel, nHeads, nKV, headDim, dFF, vocab, 2)
	sess, err := NewArchSession(g, arch, 16)
	if err != nil {
		t.Fatalf("NewArchSession: %v", err)
	}
	if sess.headEnc == nil {
		t.Fatal("session fixture did not build resident head encoder")
	}
	if err := sess.PrefillTokens([]int32{1, 5, 3}); err != nil {
		t.Fatalf("PrefillTokens: %v", err)
	}
	if sess.retainedHiddenBuffer() == nil {
		t.Fatal("retained hidden did not expose no-copy buffer")
	}
	sentinel, err := newHeadHiddenScratch(len(sess.retainedHidden))
	if err != nil {
		t.Fatalf("newHeadHiddenScratch: %v", err)
	}
	defer sentinel.Close()
	for i := range sentinel.pinned.bytes {
		sentinel.pinned.bytes[i] = 0xa5
	}
	sess.headEnc.hiddenScratch.Put(sentinel)

	params := model.SampleParams{Temperature: 1, TopK: 5, TopP: 0.5, SuppressTokens: []int32{2, 7}}
	_, _, ok, err := sess.sampleTopKCandidatesFromHiddenInPool(sess.retainedHidden, params)
	if err != nil {
		t.Fatalf("sampleTopKCandidatesFromHiddenInPool: %v", err)
	}
	if !ok {
		t.Skip("head top-k custom kernel unavailable")
	}
	if bytes.Equal(sentinel.pinned.bytes, sess.retainedHidden) {
		t.Fatal("retained-hidden candidate path copied hidden into head scratch; want direct no-copy buffer")
	}
	for i, b := range sentinel.pinned.bytes {
		if b != 0xa5 {
			t.Fatalf("retained-hidden candidate path mutated hidden scratch at byte %d: got %#x, want 0xa5", i, b)
		}
	}
}

func TestArchSessionSampleTopKCandidatesRetainedHiddenAllocationBudget(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	const dModel, nHeads, nKV, headDim, dFF, vocab = 128, 2, 1, 64, 256, 64
	g, arch := gemma4BF16Fixture(t, dModel, nHeads, nKV, headDim, dFF, vocab, 2)
	sess, err := NewArchSession(g, arch, 16)
	if err != nil {
		t.Fatalf("NewArchSession: %v", err)
	}
	sess.rememberRetainedHidden(toBF16Bytes(syntheticFloat32(dModel, 50)))
	if sess.retainedHiddenBuffer() == nil {
		t.Fatal("retained hidden did not expose no-copy buffer")
	}
	params := model.SampleParams{Temperature: 1, TopK: 5, TopP: 0.5, SuppressTokens: []int32{2, 7}}
	if _, _, ok, err := sess.sampleTopKCandidatesFromHiddenInPool(sess.retainedHidden, params); err != nil {
		t.Fatalf("sampleTopKCandidates warmup: %v", err)
	} else if !ok {
		t.Skip("head top-k custom kernel unavailable")
	}
	allocs := testing.AllocsPerRun(5, func() {
		if _, _, ok, err := sess.sampleTopKCandidatesFromHiddenInPool(sess.retainedHidden, params); err != nil {
			t.Fatalf("sampleTopKCandidates: %v", err)
		} else if !ok {
			t.Fatal("sampleTopKCandidates declined after warmup")
		}
	})
	if allocs > 90 {
		t.Fatalf("retained-hidden TopK candidate allocations = %.0f, want <= 90", allocs)
	}
}

func TestArchSessionSampleTopKRepeatPenaltyMatchesFullHead(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	const dModel, nHeads, nKV, headDim, dFF, vocab = 128, 2, 1, 64, 256, 64
	const maxLen = 16
	g, arch := gemma4BF16Fixture(t, dModel, nHeads, nKV, headDim, dFF, vocab, 2)
	sess, err := NewArchSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("NewArchSession: %v", err)
	}
	if err := sess.PrefillTokens([]int32{1, 5, 3}); err != nil {
		t.Fatalf("PrefillTokens: %v", err)
	}
	hidden := append([]byte(nil), sess.retainedHidden...)
	params := model.SampleParams{Temperature: 1, TopK: 7, TopP: 0.75, SuppressTokens: []int32{2, 7}, RepeatPenalty: 1.2}
	history := []int32{4, 5, 5, 31}
	full, err := sess.head(hidden, false)
	if err != nil {
		t.Fatalf("head: %v", err)
	}
	penalized, err := nativeApplyRepeatPenaltyBF16(full, arch.Vocab, history, params.RepeatPenalty)
	if err != nil {
		t.Fatalf("nativeApplyRepeatPenaltyBF16: %v", err)
	}
	draw := model.NewSampler(123).Draw()
	want, err := model.NewSampler(123).Sample(penalized, arch.Vocab, params)
	if err != nil {
		t.Fatalf("penalized full Sample: %v", err)
	}
	deviceGot, ok, err := sess.sampleTopKTokenFromHiddenInPool(hidden, params, draw, history)
	if err != nil {
		t.Fatalf("sampleTopKTokenFromHiddenInPool: %v", err)
	}
	if !ok {
		t.Fatal("device TopK repeat-penalty sampler declined")
	}
	if deviceGot != want {
		t.Fatalf("device TopK repeat-penalty sample = %d, want penalized full-head sample %d", deviceGot, want)
	}
}

func TestArchSessionSampleLogitsTokenMatchesVocabOrderHead(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	const dModel, nHeads, nKV, headDim, dFF, vocab = 128, 2, 1, 64, 256, 64
	const maxLen = 16
	g, arch := gemma4BF16Fixture(t, dModel, nHeads, nKV, headDim, dFF, vocab, 2)
	sess, err := NewArchSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("NewArchSession: %v", err)
	}
	if err := sess.PrefillTokens([]int32{1, 5, 3}); err != nil {
		t.Fatalf("PrefillTokens: %v", err)
	}
	hidden := append([]byte(nil), sess.retainedHidden...)
	params := model.SampleParams{Temperature: 0.8, MinP: 0.02, SuppressTokens: []int32{2, 7}, RepeatPenalty: 1.2}
	history := []int32{4, 5, 5, 31}
	full, err := sess.head(hidden, false)
	if err != nil {
		t.Fatalf("head: %v", err)
	}
	draw := float32(0.37)
	want, err := sampleBF16VocabOrderForTest(full, arch.Vocab, params, draw, history)
	if err != nil {
		t.Fatalf("sampleBF16VocabOrderForTest: %v", err)
	}
	got, ok, err := sess.sampleLogitsTokenFromHiddenInPool(hidden, params, draw, history)
	if err != nil {
		t.Fatalf("sampleLogitsTokenFromHiddenInPool: %v", err)
	}
	if !ok {
		t.Skip("device logits sampler unavailable")
	}
	if got != want {
		t.Fatalf("device logits sample = %d, want vocab-order sample %d", got, want)
	}
}

func TestArchSessionSampleLogitsTopKRepeatPenaltyMatchesFullHead(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	const dModel, nHeads, nKV, headDim, dFF, vocab = 128, 2, 1, 64, 256, 64
	const maxLen = 16
	g, arch := gemma4BF16Fixture(t, dModel, nHeads, nKV, headDim, dFF, vocab, 2)
	sess, err := NewArchSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("NewArchSession: %v", err)
	}
	if err := sess.PrefillTokens([]int32{1, 5, 3}); err != nil {
		t.Fatalf("PrefillTokens: %v", err)
	}
	hidden := append([]byte(nil), sess.retainedHidden...)
	params := model.SampleParams{Temperature: 1, TopK: 7, TopP: 0.75, SuppressTokens: []int32{2, 7}, RepeatPenalty: 1.2}
	history := []int32{4, 5, 5, 31}
	full, err := sess.head(hidden, false)
	if err != nil {
		t.Fatalf("head: %v", err)
	}
	penalized, err := nativeApplyRepeatPenaltyBF16(full, arch.Vocab, history, params.RepeatPenalty)
	if err != nil {
		t.Fatalf("nativeApplyRepeatPenaltyBF16: %v", err)
	}
	draw := model.NewSampler(123).Draw()
	want, err := model.NewSampler(123).Sample(penalized, arch.Vocab, params)
	if err != nil {
		t.Fatalf("penalized full Sample: %v", err)
	}
	got, ok, err := sess.sampleLogitsTokenFromHiddenInPool(hidden, params, draw, history)
	if err != nil {
		t.Fatalf("sampleLogitsTokenFromHiddenInPool: %v", err)
	}
	if !ok {
		t.Fatal("device logits TopK repeat-penalty sampler declined")
	}
	if got != want {
		t.Fatalf("device logits TopK repeat-penalty sample = %d, want penalized full-head sample %d", got, want)
	}
}

func TestArchSessionSampleRetainedLogitsBufferMatchesFullSampler(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	const dModel, nHeads, nKV, headDim, dFF, vocab = 128, 2, 1, 64, 256, 64
	const maxLen = 16
	g, arch := gemma4BF16Fixture(t, dModel, nHeads, nKV, headDim, dFF, vocab, 2)
	sess, err := NewArchSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("NewArchSession: %v", err)
	}
	if err := sess.PrefillTokens([]int32{1, 5, 3}); err != nil {
		t.Fatalf("PrefillTokens: %v", err)
	}
	logits, err := sess.BoundaryLogits()
	if err != nil {
		t.Fatalf("BoundaryLogits: %v", err)
	}
	params := model.SampleParams{Temperature: 1, TopK: 7, TopP: 0.75, SuppressTokens: []int32{2, 7}}
	draw := model.NewSampler(123).Draw()
	want, err := model.NewSampler(123).Sample(logits, arch.Vocab, params)
	if err != nil {
		t.Fatalf("full Sample: %v", err)
	}
	got, ok, err := sess.sampleTokenFromRetainedLogitsInPool(params, draw, nil)
	if err != nil {
		t.Fatalf("sampleTokenFromRetainedLogitsInPool: %v", err)
	}
	if !ok {
		t.Fatal("retained-logits device sampler declined")
	}
	if got != want {
		t.Fatalf("retained-logits device sample = %d, want full retained-logits sample %d", got, want)
	}
}

func TestArchSessionSampleLogitsTopPOnlySmallVocabMatchesFullHead(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	const dModel, nHeads, nKV, headDim, dFF, vocab = 128, 2, 1, 64, 256, 64
	const maxLen = 16
	g, arch := gemma4BF16Fixture(t, dModel, nHeads, nKV, headDim, dFF, vocab, 2)
	sess, err := NewArchSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("NewArchSession: %v", err)
	}
	if err := sess.PrefillTokens([]int32{1, 5, 3}); err != nil {
		t.Fatalf("PrefillTokens: %v", err)
	}
	hidden := append([]byte(nil), sess.retainedHidden...)
	params := model.SampleParams{Temperature: 1, TopP: 0.72, SuppressTokens: []int32{2, 7}}
	full, err := sess.head(hidden, false)
	if err != nil {
		t.Fatalf("head: %v", err)
	}
	draw := model.NewSampler(123).Draw()
	want, err := model.NewSampler(123).Sample(full, arch.Vocab, params)
	if err != nil {
		t.Fatalf("full Sample: %v", err)
	}
	got, ok, err := sess.sampleLogitsTokenFromHiddenInPool(hidden, params, draw, nil)
	if err != nil {
		t.Fatalf("sampleLogitsTokenFromHiddenInPool: %v", err)
	}
	if !ok {
		t.Fatal("device logits TopP-only sampler declined")
	}
	if got != want {
		t.Fatalf("device logits TopP-only sample = %d, want full-head sample %d", got, want)
	}
}

func TestArchSessionSampleLogitsTopPOnlyLargeVocabMatchesFullHead(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	const dModel, nHeads, nKV, headDim, dFF, vocab = 128, 2, 1, 64, 256, headSampleTopKMaxK + 8
	const maxLen = 16
	g, arch := gemma4BF16Fixture(t, dModel, nHeads, nKV, headDim, dFF, vocab, 2)
	sess, err := NewArchSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("NewArchSession: %v", err)
	}
	if err := sess.PrefillTokens([]int32{1, 5, 3}); err != nil {
		t.Fatalf("PrefillTokens: %v", err)
	}
	hidden := append([]byte(nil), sess.retainedHidden...)
	params := model.SampleParams{Temperature: 1, TopP: 0.72, SuppressTokens: []int32{2, 7}}
	full, err := sess.head(hidden, false)
	if err != nil {
		t.Fatalf("head: %v", err)
	}
	draw := model.NewSampler(123).Draw()
	want, err := model.NewSampler(123).Sample(full, arch.Vocab, params)
	if err != nil {
		t.Fatalf("full Sample: %v", err)
	}
	got, ok, err := sess.sampleLogitsTokenFromHiddenInPool(hidden, params, draw, nil)
	if err != nil {
		t.Fatalf("sampleLogitsTokenFromHiddenInPool: %v", err)
	}
	if !ok {
		t.Fatal("device logits TopP-only large-vocab sampler declined")
	}
	if got != want {
		t.Fatalf("device logits TopP-only large-vocab sample = %d, want full-head sample %d", got, want)
	}
}

func TestArchSessionSampleLogitsTopPOnlyLargeVocabRepeatPenaltyMatchesFullHead(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	const dModel, nHeads, nKV, headDim, dFF, vocab = 128, 2, 1, 64, 256, headSampleTopKMaxK + 8
	const maxLen = 16
	g, arch := gemma4BF16Fixture(t, dModel, nHeads, nKV, headDim, dFF, vocab, 2)
	sess, err := NewArchSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("NewArchSession: %v", err)
	}
	if err := sess.PrefillTokens([]int32{1, 5, 3}); err != nil {
		t.Fatalf("PrefillTokens: %v", err)
	}
	hidden := append([]byte(nil), sess.retainedHidden...)
	params := model.SampleParams{Temperature: 1, TopP: 0.72, SuppressTokens: []int32{2, 7}, RepeatPenalty: 1.2}
	history := []int32{4, 5, 5, 31}
	full, err := sess.head(hidden, false)
	if err != nil {
		t.Fatalf("head: %v", err)
	}
	penalized, err := nativeApplyRepeatPenaltyBF16(full, arch.Vocab, history, params.RepeatPenalty)
	if err != nil {
		t.Fatalf("nativeApplyRepeatPenaltyBF16: %v", err)
	}
	draw := model.NewSampler(123).Draw()
	want, err := model.NewSampler(123).Sample(penalized, arch.Vocab, params)
	if err != nil {
		t.Fatalf("penalized full Sample: %v", err)
	}
	got, ok, err := sess.sampleLogitsTokenFromHiddenInPool(hidden, params, draw, history)
	if err != nil {
		t.Fatalf("sampleLogitsTokenFromHiddenInPool: %v", err)
	}
	if !ok {
		t.Fatal("device logits TopP-only large-vocab repeat-penalty sampler declined")
	}
	if got != want {
		t.Fatalf("device logits TopP-only large-vocab repeat-penalty sample = %d, want penalized full-head sample %d", got, want)
	}
}

func TestArchSessionSampleRetainedHiddenLogitsBufferMatchesFullHead(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	const dModel, nHeads, nKV, headDim, dFF, vocab = 128, 2, 1, 64, 256, 64
	const maxLen = 16
	g, arch := gemma4BF16Fixture(t, dModel, nHeads, nKV, headDim, dFF, vocab, 2)
	sess, err := NewArchSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("NewArchSession: %v", err)
	}
	if err := sess.PrefillTokens([]int32{1, 5, 3}); err != nil {
		t.Fatalf("PrefillTokens: %v", err)
	}
	hiddenBuf := sess.retainedHiddenBuffer()
	if hiddenBuf == nil {
		t.Fatal("retained hidden did not expose pinned no-copy buffer")
	}
	params := model.SampleParams{Temperature: 1, TopP: 0.72, SuppressTokens: []int32{2, 7}}
	full, err := sess.head(sess.retainedHidden, false)
	if err != nil {
		t.Fatalf("head: %v", err)
	}
	draw := model.NewSampler(123).Draw()
	want, err := model.NewSampler(123).Sample(full, arch.Vocab, params)
	if err != nil {
		t.Fatalf("full Sample: %v", err)
	}
	got, ok, err := sess.headEnc.sampleLogitsTokenBufferInPool(hiddenBuf, params, draw, nil)
	if err != nil {
		t.Fatalf("sampleLogitsTokenBufferInPool: %v", err)
	}
	if !ok {
		t.Fatal("retained-hidden logits-buffer sampler declined")
	}
	if got != want {
		t.Fatalf("retained-hidden logits-buffer sample = %d, want full-head sample %d", got, want)
	}
}

func TestArchSessionSampleRetainedHiddenLogitsBufferAllocationBudget(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	const dModel, nHeads, nKV, headDim, dFF, vocab = 128, 2, 1, 64, 256, 64
	const maxLen = 16
	g, arch := gemma4BF16Fixture(t, dModel, nHeads, nKV, headDim, dFF, vocab, 2)
	sess, err := NewArchSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("NewArchSession: %v", err)
	}
	if err := sess.PrefillTokens([]int32{1, 5, 3}); err != nil {
		t.Fatalf("PrefillTokens: %v", err)
	}
	hiddenBuf := sess.retainedHiddenBuffer()
	if hiddenBuf == nil {
		t.Fatal("retained hidden did not expose pinned no-copy buffer")
	}
	params := model.SampleParams{Temperature: 1, TopP: 0.72}
	sampler := model.NewSampler(123)
	if _, ok, err := sess.headEnc.sampleLogitsTokenBufferInPool(hiddenBuf, params, sampler.Draw(), nil); err != nil {
		t.Fatalf("sampleLogitsTokenBufferInPool warmup: %v", err)
	} else if !ok {
		t.Fatal("retained-hidden logits-buffer sampler declined")
	}
	allocs := testing.AllocsPerRun(5, func() {
		if _, ok, err := sess.headEnc.sampleLogitsTokenBufferInPool(hiddenBuf, params, sampler.Draw(), nil); err != nil {
			t.Fatalf("sampleLogitsTokenBufferInPool: %v", err)
		} else if !ok {
			t.Fatal("retained-hidden logits-buffer sampler declined")
		}
	})
	if allocs > 0 {
		t.Fatalf("retained-hidden logits-buffer TopP allocations = %.0f, want 0", allocs)
	}
}

func TestArchSessionStepSampleTopKCandidatesICBMatchesSerial(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	const dModel, nHeads, nKV, headDim, dFF, vocab = 128, 2, 1, 64, 256, 64
	const maxLen = 16
	g, arch := gemma4BF16Fixture(t, dModel, nHeads, nKV, headDim, dFF, vocab, 2)
	serial, err := NewArchSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("NewArchSession serial: %v", err)
	}
	chained, err := NewArchSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("NewArchSession chained: %v", err)
	}
	if chained.state.icb == nil {
		t.Skip("ICB replay unavailable for sampled chain")
	}
	for _, id := range []int32{1, 5, 3} {
		if _, err := serial.stepID(id); err != nil {
			t.Fatalf("serial prefix stepID(%d): %v", id, err)
		}
		if _, err := chained.stepID(id); err != nil {
			t.Fatalf("chained prefix stepID(%d): %v", id, err)
		}
	}
	params := model.SampleParams{Temperature: 1, TopK: 5, SuppressTokens: []int32{2, 7}}
	serialHidden, err := serial.stepID(9)
	if err != nil {
		t.Fatalf("serial stepID: %v", err)
	}
	wantLogits, wantIDs, ok, err := serial.sampleTopKCandidatesFromHiddenInPool(serialHidden, params)
	if err != nil {
		t.Fatalf("serial sampleTopKCandidates: %v", err)
	}
	if !ok {
		t.Fatal("serial sampleTopKCandidates declined")
	}
	gotHidden, gotLogits, gotIDs, ok, err := chained.stepSampleTopKCandidatesInPool(9, params)
	if err != nil {
		t.Fatalf("chained stepSampleTopKCandidatesInPool: %v", err)
	}
	if !ok {
		t.Fatal("chained stepSampleTopKCandidatesInPool declined")
	}
	if !bytes.Equal(gotHidden, serialHidden) {
		t.Fatal("chained sampled hidden differs from serial stepID hidden")
	}
	if !bytes.Equal(gotLogits, wantLogits) || !idsEqual(gotIDs, wantIDs) {
		t.Fatalf("chained candidates logits/ids differ from serial: ids got %v want %v", gotIDs, wantIDs)
	}
	if chained.Pos() != serial.Pos() {
		t.Fatalf("positions diverged: chained=%d serial=%d", chained.Pos(), serial.Pos())
	}

	serial, err = NewArchSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("NewArchSession serial repeat penalty: %v", err)
	}
	chained, err = NewArchSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("NewArchSession chained repeat penalty: %v", err)
	}
	for _, id := range []int32{1, 5, 3} {
		if _, err := serial.stepID(id); err != nil {
			t.Fatalf("serial repeat-penalty prefix stepID(%d): %v", id, err)
		}
		if _, err := chained.stepID(id); err != nil {
			t.Fatalf("chained repeat-penalty prefix stepID(%d): %v", id, err)
		}
	}
	params = model.SampleParams{Temperature: 1, TopK: 7, TopP: 0.75, SuppressTokens: []int32{2, 7}, RepeatPenalty: 8}
	serialHidden, err = serial.stepID(9)
	if err != nil {
		t.Fatalf("serial repeat-penalty stepID: %v", err)
	}
	unpenalizedLogits, unpenalizedIDs, ok, err := serial.sampleTopKCandidatesFromHiddenInPool(serialHidden, params)
	if err != nil || !ok {
		t.Fatalf("serial unpenalized sampleTopKCandidatesFromHiddenInPool ok=%v err=%v", ok, err)
	}
	// snapshot: the returned slices alias the session's reusable candidate scratch, which the
	// next sample call overwrites — comparing without copying compares a buffer with itself.
	unpenalizedLogits = append([]byte(nil), unpenalizedLogits...)
	unpenalizedIDs = append([]int32(nil), unpenalizedIDs...)
	history := append([]int32(nil), unpenalizedIDs...)
	wantLogits, wantIDs, ok = nil, nil, false
	wantLogits, wantIDs, ok, err = serial.sampleTopKCandidatesFromHiddenWithHistoryInPool(serialHidden, params, history)
	if err != nil || !ok {
		t.Fatalf("serial sampleTopKCandidatesFromHiddenWithHistoryInPool ok=%v err=%v", ok, err)
	}
	if bytes.Equal(unpenalizedLogits, wantLogits) && idsEqual(unpenalizedIDs, wantIDs) {
		t.Fatal("BF16 fixture does not exercise repeat-penalty candidate differences")
	}
	gotHidden, gotLogits, gotIDs, ok = nil, nil, nil, false
	gotHidden, gotLogits, gotIDs, ok, err = chained.stepSampleTopKCandidatesWithHistoryInPool(9, params, history)
	if err != nil || !ok {
		t.Fatalf("chained stepSampleTopKCandidatesWithHistoryInPool ok=%v err=%v", ok, err)
	}
	if !bytes.Equal(gotHidden, serialHidden) {
		t.Fatal("chained sampled-candidate repeat-penalty hidden differs from serial stepID hidden")
	}
	if !bytes.Equal(gotLogits, wantLogits) || !idsEqual(gotIDs, wantIDs) {
		t.Fatalf("chained repeat-penalty candidates differ from serial: ids got %v want %v", gotIDs, wantIDs)
	}
}

func TestArchSessionStepSampleTopKCandidatesICBAllocationBudget(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	sess := newQuantICBAllocationSession(t, 32)
	params := model.SampleParams{Temperature: 1, TopK: 7, TopP: 0.75, SuppressTokens: []int32{2, 7}}
	if _, _, _, ok, err := sess.stepSampleTopKCandidatesInPool(9, params); err != nil {
		t.Fatalf("stepSampleTopKCandidatesInPool warmup: %v", err)
	} else if !ok {
		t.Skip("device TopK candidate sampler unavailable")
	}
	allocs := testing.AllocsPerRun(5, func() {
		if _, _, _, ok, err := sess.stepSampleTopKCandidatesInPool(9, params); err != nil {
			t.Fatalf("stepSampleTopKCandidatesInPool: %v", err)
		} else if !ok {
			t.Fatal("stepSampleTopKCandidatesInPool declined after warmup")
		}
	})
	if allocs > 40 {
		t.Fatalf("ICB sampled-TopK candidate allocations = %.0f, want <= 40", allocs)
	}
}

func TestArchSessionStepSampleQuantICBMatchesSerial(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	t.Run("logits-token", func(t *testing.T) {
		serial := newQuantICBAllocationSession(t, 32)
		chained := newQuantICBAllocationSession(t, 32)
		params := model.SampleParams{Temperature: 0.8, MinP: 0.02, SuppressTokens: []int32{2, 7}, RepeatPenalty: 1.2}
		history := []int32{4, 5, 5, 31}
		draw := float32(0.37)
		serialHidden, err := serial.stepID(9)
		if err != nil {
			t.Fatalf("serial stepID: %v", err)
		}
		wantToken, ok, err := serial.sampleLogitsTokenFromHiddenInPool(serialHidden, params, draw, history)
		if err != nil || !ok {
			t.Fatalf("serial sampleLogitsTokenFromHiddenInPool ok=%v err=%v", ok, err)
		}
		gotHidden, gotToken, ok, err := chained.stepSampleLogitsTokenInPool(9, params, draw, history)
		if err != nil || !ok {
			t.Fatalf("chained stepSampleLogitsTokenInPool ok=%v err=%v", ok, err)
		}
		if !bytes.Equal(gotHidden, serialHidden) {
			t.Fatal("chained sampled-logits hidden differs from serial stepID hidden")
		}
		if gotToken != wantToken {
			t.Fatalf("chained sampled-logits token = %d, want %d", gotToken, wantToken)
		}
	})
	t.Run("topk-token", func(t *testing.T) {
		serial := newQuantICBAllocationSession(t, 32)
		chained := newQuantICBAllocationSession(t, 32)
		params := model.SampleParams{Temperature: 1, TopK: 7, TopP: 0.75, SuppressTokens: []int32{2, 7}, RepeatPenalty: 1.2}
		history := []int32{4, 5, 5, 31}
		draw := float32(0.42)
		serialHidden, err := serial.stepID(9)
		if err != nil {
			t.Fatalf("serial stepID: %v", err)
		}
		wantToken, ok, err := serial.sampleTopKTokenFromHiddenInPool(serialHidden, params, draw, history)
		if err != nil || !ok {
			t.Fatalf("serial sampleTopKTokenFromHiddenInPool ok=%v err=%v", ok, err)
		}
		gotHidden, gotToken, ok, err := chained.stepSampleTopKTokenInPool(9, params, draw, history)
		if err != nil || !ok {
			t.Fatalf("chained stepSampleTopKTokenInPool ok=%v err=%v", ok, err)
		}
		if !bytes.Equal(gotHidden, serialHidden) {
			t.Fatal("chained sampled-TopK hidden differs from serial stepID hidden")
		}
		if gotToken != wantToken {
			t.Fatalf("chained sampled-TopK token = %d, want %d", gotToken, wantToken)
		}
	})
	t.Run("topk-candidates", func(t *testing.T) {
		serial := newQuantICBAllocationSession(t, 32)
		chained := newQuantICBAllocationSession(t, 32)
		params := model.SampleParams{Temperature: 1, TopK: 7, TopP: 0.75, SuppressTokens: []int32{2, 7}}
		serialHidden, err := serial.stepID(9)
		if err != nil {
			t.Fatalf("serial stepID: %v", err)
		}
		wantLogits, wantIDs, ok, err := serial.sampleTopKCandidatesFromHiddenInPool(serialHidden, params)
		if err != nil || !ok {
			t.Fatalf("serial sampleTopKCandidatesFromHiddenInPool ok=%v err=%v", ok, err)
		}
		gotHidden, gotLogits, gotIDs, ok, err := chained.stepSampleTopKCandidatesInPool(9, params)
		if err != nil || !ok {
			t.Fatalf("chained stepSampleTopKCandidatesInPool ok=%v err=%v", ok, err)
		}
		if !bytes.Equal(gotHidden, serialHidden) {
			t.Fatal("chained sampled-candidate hidden differs from serial stepID hidden")
		}
		if !bytes.Equal(gotLogits, wantLogits) || !idsEqual(gotIDs, wantIDs) {
			t.Fatalf("chained candidates differ from serial: ids got %v want %v", gotIDs, wantIDs)
		}
	})
}

func TestArchSessionStepSampleQuantICBWritesRetainedHiddenDirectly(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	t.Run("logits-token", func(t *testing.T) {
		control := newQuantICBAllocationSession(t, 32)
		candidate := newQuantICBAllocationSession(t, 32)
		params := model.SampleParams{Temperature: 0.8, MinP: 0.02, SuppressTokens: []int32{2, 7}, RepeatPenalty: 1.2}
		history := []int32{4, 5, 5, 31}
		draw := float32(0.37)
		wantHidden, wantToken, ok, err := control.stepSampleLogitsTokenInPool(9, params, draw, history)
		if err != nil || !ok {
			t.Fatalf("control stepSampleLogitsTokenInPool ok=%v err=%v", ok, err)
		}
		poison := bytes.Repeat([]byte{0x7e}, candidate.arch.Hidden*bf16Size)
		candidate.state.icb.lastOutPtr = &poison[0]
		gotHidden, gotToken, ok, err := candidate.stepSampleLogitsTokenInPool(9, params, draw, history)
		runtime.KeepAlive(poison)
		if err != nil || !ok {
			t.Fatalf("candidate stepSampleLogitsTokenInPool ok=%v err=%v", ok, err)
		}
		if !bytes.Equal(gotHidden, wantHidden) || gotToken != wantToken {
			t.Fatal("sampled logits-token path read retained hidden from lastOutPtr instead of direct output")
		}
		if len(candidate.retainedHidden) == 0 || unsafe.Pointer(&gotHidden[0]) != unsafe.Pointer(&candidate.retainedHidden[0]) {
			t.Fatal("sampled logits-token path returned transient hidden instead of retained backing")
		}
	})
	t.Run("topk-token", func(t *testing.T) {
		control := newQuantICBAllocationSession(t, 32)
		candidate := newQuantICBAllocationSession(t, 32)
		params := model.SampleParams{Temperature: 1, TopK: 7, TopP: 0.75, SuppressTokens: []int32{2, 7}, RepeatPenalty: 1.2}
		history := []int32{4, 5, 5, 31}
		draw := float32(0.42)
		wantHidden, wantToken, ok, err := control.stepSampleTopKTokenInPool(9, params, draw, history)
		if err != nil || !ok {
			t.Fatalf("control stepSampleTopKTokenInPool ok=%v err=%v", ok, err)
		}
		poison := bytes.Repeat([]byte{0x6d}, candidate.arch.Hidden*bf16Size)
		candidate.state.icb.lastOutPtr = &poison[0]
		gotHidden, gotToken, ok, err := candidate.stepSampleTopKTokenInPool(9, params, draw, history)
		runtime.KeepAlive(poison)
		if err != nil || !ok {
			t.Fatalf("candidate stepSampleTopKTokenInPool ok=%v err=%v", ok, err)
		}
		if !bytes.Equal(gotHidden, wantHidden) || gotToken != wantToken {
			t.Fatal("sampled TopK-token path read retained hidden from lastOutPtr instead of direct output")
		}
		if len(candidate.retainedHidden) == 0 || unsafe.Pointer(&gotHidden[0]) != unsafe.Pointer(&candidate.retainedHidden[0]) {
			t.Fatal("sampled TopK-token path returned transient hidden instead of retained backing")
		}
	})
	t.Run("topk-candidates", func(t *testing.T) {
		control := newQuantICBAllocationSession(t, 32)
		candidate := newQuantICBAllocationSession(t, 32)
		params := model.SampleParams{Temperature: 1, TopK: 7, TopP: 0.75, SuppressTokens: []int32{2, 7}}
		wantHidden, wantLogits, wantIDs, ok, err := control.stepSampleTopKCandidatesInPool(9, params)
		if err != nil || !ok {
			t.Fatalf("control stepSampleTopKCandidatesInPool ok=%v err=%v", ok, err)
		}
		poison := bytes.Repeat([]byte{0x5c}, candidate.arch.Hidden*bf16Size)
		candidate.state.icb.lastOutPtr = &poison[0]
		gotHidden, gotLogits, gotIDs, ok, err := candidate.stepSampleTopKCandidatesInPool(9, params)
		runtime.KeepAlive(poison)
		if err != nil || !ok {
			t.Fatalf("candidate stepSampleTopKCandidatesInPool ok=%v err=%v", ok, err)
		}
		if !bytes.Equal(gotHidden, wantHidden) || !bytes.Equal(gotLogits, wantLogits) || !idsEqual(gotIDs, wantIDs) {
			t.Fatal("sampled TopK-candidate path read retained hidden from lastOutPtr instead of direct output")
		}
		if len(candidate.retainedHidden) == 0 || unsafe.Pointer(&gotHidden[0]) != unsafe.Pointer(&candidate.retainedHidden[0]) {
			t.Fatal("sampled TopK-candidate path returned transient hidden instead of retained backing")
		}
	})
}

// TestArchSessionStepSampleHostChainMatchesGPUInputs pins the host embed/PLE fallback
// body of stepSampleTopKCandidatesWithHistoryInPool and stepSampleLogitsTokenInPool (#30
// r5, the pipelined-GPU sampling-tail cluster): with chainedGPUInputsDisabled forced,
// both step onto their OWN encodeStepBody/perLayerInput path (host embedID readback, PLE
// tensor compute, pinned-hidden direct-out encode) instead of delegating to the
// GPU-next-inputs sibling (stepSampleTopKCandidatesGPUInputsWithHistoryInPool /
// stepSampleLogitsTokenGPUInputsInPool) — every existing test on these two symbols
// leaves this body dark, because chainedGPUInputsDisabled is always left at its zero
// value (false) elsewhere, and the ICB/PLE fixture always wires the GPU next-inputs seam.
// A PLE-bearing fixture (pleQuantModel) also exercises the `s.perLayerInput != nil`
// branch that the no-PLE newQuantICBAllocationSession fixture cannot reach.
func TestArchSessionStepSampleHostChainMatchesGPUInputs(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	g, arch := pleQuantModel(t, 2, 256, 32, 0)
	const maxLen = 24
	prefix := []int32{1, 5, 3}

	buildPrimed := func(t *testing.T) *ArchSession {
		t.Helper()
		sess, err := NewArchQuantSession(g, arch, maxLen)
		if err != nil {
			t.Fatalf("NewArchQuantSession: %v", err)
		}
		if sess.state.icb == nil {
			t.Skip("ICB replay unavailable")
		}
		if sess.encNextInputsGPU == nil {
			t.Fatal("fixture did not wire GPU next-inputs seam")
		}
		if sess.perLayerInput == nil {
			t.Fatal("fixture did not wire the PLE per-layer-input tower")
		}
		for _, id := range prefix {
			if _, err := sess.stepID(id); err != nil {
				t.Fatalf("prefix stepID(%d): %v", id, err)
			}
		}
		return sess
	}

	t.Run("topk-candidates", func(t *testing.T) {
		params := model.SampleParams{Temperature: 1, TopK: 7, SuppressTokens: []int32{2, 7}}
		history := []int32{4, 5, 5, 31}

		gpu := buildPrimed(t)
		gpuHidden, gpuLogits, gpuIDs, ok, err := gpu.stepSampleTopKCandidatesWithHistoryInPool(9, params, history)
		if err != nil || !ok {
			t.Fatalf("GPU-inputs stepSampleTopKCandidatesWithHistoryInPool ok=%v err=%v", ok, err)
		}

		oldChainDisabled := chainedGPUInputsDisabled
		defer func() { chainedGPUInputsDisabled = oldChainDisabled }()
		chainedGPUInputsDisabled = true

		host := buildPrimed(t)
		hostHidden, hostLogits, hostIDs, ok, err := host.stepSampleTopKCandidatesWithHistoryInPool(9, params, history)
		if err != nil || !ok {
			t.Fatalf("host-chain stepSampleTopKCandidatesWithHistoryInPool ok=%v err=%v", ok, err)
		}
		if !bytes.Equal(hostHidden, gpuHidden) {
			t.Fatal("host-chain hidden differs from GPU-inputs hidden")
		}
		if !bytes.Equal(hostLogits, gpuLogits) || !idsEqual(hostIDs, gpuIDs) {
			t.Fatalf("host-chain candidates differ from GPU-inputs: ids got %v want %v", hostIDs, gpuIDs)
		}
	})

	t.Run("logits-token", func(t *testing.T) {
		params := model.SampleParams{Temperature: 0.8, MinP: 0.02, SuppressTokens: []int32{2, 7}}
		history := []int32{4, 5, 5, 31}
		draw := float32(0.37)

		gpu := buildPrimed(t)
		gpuHidden, gpuToken, ok, err := gpu.stepSampleLogitsTokenInPool(9, params, draw, history)
		if err != nil || !ok {
			t.Fatalf("GPU-inputs stepSampleLogitsTokenInPool ok=%v err=%v", ok, err)
		}

		oldChainDisabled := chainedGPUInputsDisabled
		defer func() { chainedGPUInputsDisabled = oldChainDisabled }()
		chainedGPUInputsDisabled = true

		host := buildPrimed(t)
		hostHidden, hostToken, ok, err := host.stepSampleLogitsTokenInPool(9, params, draw, history)
		if err != nil || !ok {
			t.Fatalf("host-chain stepSampleLogitsTokenInPool ok=%v err=%v", ok, err)
		}
		if !bytes.Equal(hostHidden, gpuHidden) {
			t.Fatal("host-chain hidden differs from GPU-inputs hidden")
		}
		if hostToken != gpuToken {
			t.Fatalf("host-chain token = %d, want GPU-inputs token %d", hostToken, gpuToken)
		}
	})
}

func TestArchSessionStepSampleLogitsTokenICBUsesGPUNextInputs(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	g, arch := pleQuantModel(t, 2, 256, 32, 0)
	serial, err := NewArchQuantSession(g, arch, 16)
	if err != nil {
		t.Fatalf("serial session: %v", err)
	}
	chained, err := NewArchQuantSession(g, arch, 16)
	if err != nil {
		t.Fatalf("chained session: %v", err)
	}
	if chained.encNextInputsGPU == nil {
		t.Fatal("fixture did not wire GPU next-inputs seam")
	}
	for _, id := range []int32{1, 5, 3} {
		if _, err := serial.stepID(id); err != nil {
			t.Fatalf("serial prefix stepID(%d): %v", id, err)
		}
		if _, err := chained.stepID(id); err != nil {
			t.Fatalf("chained prefix stepID(%d): %v", id, err)
		}
	}
	params := model.SampleParams{Temperature: 0.8, MinP: 0.02, SuppressTokens: []int32{2, 7}, RepeatPenalty: 1.2}
	history := []int32{4, 5, 5, 31}
	draw := float32(0.37)
	serialHidden, err := serial.stepID(9)
	if err != nil {
		t.Fatalf("serial stepID: %v", err)
	}
	wantToken, ok, err := serial.sampleLogitsTokenFromHiddenInPool(serialHidden, params, draw, history)
	if err != nil || !ok {
		t.Fatalf("serial sampleLogitsTokenFromHiddenInPool ok=%v err=%v", ok, err)
	}

	chained.embed = func(int32) ([]byte, error) {
		return nil, errors.New("host embed should not be called")
	}
	chained.embedInto = nil
	chained.perLayerInput = func(int32, []byte) ([]byte, error) {
		return nil, errors.New("host PLE should not be called")
	}
	gotHidden, gotToken, ok, err := chained.stepSampleLogitsTokenInPool(9, params, draw, history)
	if err != nil || !ok {
		t.Fatalf("chained stepSampleLogitsTokenInPool ok=%v err=%v", ok, err)
	}
	if !bytes.Equal(gotHidden, serialHidden) {
		t.Fatal("GPU-input sampled-logits hidden differs from serial host-input hidden")
	}
	if gotToken != wantToken {
		t.Fatalf("GPU-input sampled-logits token = %d, want %d", gotToken, wantToken)
	}
}

func TestArchSessionStepSampleTopKTokenICBUsesGPUNextInputs(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	g, arch := pleQuantModel(t, 2, 256, 32, 0)
	serial, err := NewArchQuantSession(g, arch, 16)
	if err != nil {
		t.Fatalf("serial session: %v", err)
	}
	chained, err := NewArchQuantSession(g, arch, 16)
	if err != nil {
		t.Fatalf("chained session: %v", err)
	}
	if chained.encNextInputsGPU == nil {
		t.Fatal("fixture did not wire GPU next-inputs seam")
	}
	for _, id := range []int32{1, 5, 3} {
		if _, err := serial.stepID(id); err != nil {
			t.Fatalf("serial prefix stepID(%d): %v", id, err)
		}
		if _, err := chained.stepID(id); err != nil {
			t.Fatalf("chained prefix stepID(%d): %v", id, err)
		}
	}
	params := model.SampleParams{Temperature: 1, TopK: 7, TopP: 0.75, SuppressTokens: []int32{2, 7}, RepeatPenalty: 1.2}
	history := []int32{4, 5, 5, 31}
	draw := float32(0.42)
	serialHidden, err := serial.stepID(9)
	if err != nil {
		t.Fatalf("serial stepID: %v", err)
	}
	wantToken, ok, err := serial.sampleTopKTokenFromHiddenInPool(serialHidden, params, draw, history)
	if err != nil || !ok {
		t.Fatalf("serial sampleTopKTokenFromHiddenInPool ok=%v err=%v", ok, err)
	}

	chained.embed = func(int32) ([]byte, error) {
		return nil, errors.New("host embed should not be called")
	}
	chained.embedInto = nil
	chained.perLayerInput = func(int32, []byte) ([]byte, error) {
		return nil, errors.New("host PLE should not be called")
	}
	gotHidden, gotToken, ok, err := chained.stepSampleTopKTokenInPool(9, params, draw, history)
	if err != nil || !ok {
		t.Fatalf("chained stepSampleTopKTokenInPool ok=%v err=%v", ok, err)
	}
	if !bytes.Equal(gotHidden, serialHidden) {
		t.Fatal("GPU-input sampled-TopK hidden differs from serial host-input hidden")
	}
	if gotToken != wantToken {
		t.Fatalf("GPU-input sampled-TopK token = %d, want %d", gotToken, wantToken)
	}
}

func TestArchSessionStepSampleTopKCandidatesICBUsesGPUNextInputs(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	g, arch := pleQuantModel(t, 2, 256, 32, 0)
	serial, err := NewArchQuantSession(g, arch, 16)
	if err != nil {
		t.Fatalf("serial session: %v", err)
	}
	chained, err := NewArchQuantSession(g, arch, 16)
	if err != nil {
		t.Fatalf("chained session: %v", err)
	}
	if chained.encNextInputsGPU == nil {
		t.Fatal("fixture did not wire GPU next-inputs seam")
	}
	for _, id := range []int32{1, 5, 3} {
		if _, err := serial.stepID(id); err != nil {
			t.Fatalf("serial prefix stepID(%d): %v", id, err)
		}
		if _, err := chained.stepID(id); err != nil {
			t.Fatalf("chained prefix stepID(%d): %v", id, err)
		}
	}
	params := model.SampleParams{Temperature: 1, TopK: 7, TopP: 0.75, SuppressTokens: []int32{2, 7}}
	serialHidden, err := serial.stepID(9)
	if err != nil {
		t.Fatalf("serial stepID: %v", err)
	}
	wantLogits, wantIDs, ok, err := serial.sampleTopKCandidatesFromHiddenInPool(serialHidden, params)
	if err != nil || !ok {
		t.Fatalf("serial sampleTopKCandidatesFromHiddenInPool ok=%v err=%v", ok, err)
	}

	chained.embed = func(int32) ([]byte, error) {
		return nil, errors.New("host embed should not be called")
	}
	chained.embedInto = nil
	chained.perLayerInput = func(int32, []byte) ([]byte, error) {
		return nil, errors.New("host PLE should not be called")
	}
	gotHidden, gotLogits, gotIDs, ok, err := chained.stepSampleTopKCandidatesInPool(9, params)
	if err != nil || !ok {
		t.Fatalf("chained stepSampleTopKCandidatesInPool ok=%v err=%v", ok, err)
	}
	if !bytes.Equal(gotHidden, serialHidden) {
		t.Fatal("GPU-input sampled-candidate hidden differs from serial host-input hidden")
	}
	if !bytes.Equal(gotLogits, wantLogits) || !idsEqual(gotIDs, wantIDs) {
		t.Fatalf("GPU-input candidates differ from serial: ids got %v want %v", gotIDs, wantIDs)
	}
}

func TestArchSessionStepSampleGPUInputsICBWritesRetainedHiddenDirectly(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	g, arch := pleQuantModel(t, 2, 256, 32, 0)
	prepare := func(t *testing.T) *ArchSession {
		t.Helper()
		sess, err := NewArchQuantSession(g, arch, 16)
		if err != nil {
			t.Fatalf("NewArchQuantSession: %v", err)
		}
		if sess.encNextInputsGPU == nil {
			t.Fatal("fixture did not wire GPU next-inputs seam")
		}
		for _, id := range []int32{1, 5, 3} {
			if _, err := sess.stepID(id); err != nil {
				t.Fatalf("prefix stepID(%d): %v", id, err)
			}
		}
		sess.embed = func(int32) ([]byte, error) {
			return nil, errors.New("host embed should not be called")
		}
		sess.embedInto = nil
		sess.perLayerInput = func(int32, []byte) ([]byte, error) {
			return nil, errors.New("host PLE should not be called")
		}
		return sess
	}

	t.Run("logits-token", func(t *testing.T) {
		control := prepare(t)
		candidate := prepare(t)
		params := model.SampleParams{Temperature: 0.8, MinP: 0.02, SuppressTokens: []int32{2, 7}, RepeatPenalty: 1.2}
		history := []int32{4, 5, 5, 31}
		draw := float32(0.37)
		wantHidden, wantToken, ok, err := control.stepSampleLogitsTokenInPool(9, params, draw, history)
		if err != nil || !ok {
			t.Fatalf("control stepSampleLogitsTokenInPool ok=%v err=%v", ok, err)
		}
		poison := bytes.Repeat([]byte{0x7e}, candidate.arch.Hidden*bf16Size)
		candidate.state.icb.lastOutPtr = &poison[0]
		gotHidden, gotToken, ok, err := candidate.stepSampleLogitsTokenInPool(9, params, draw, history)
		runtime.KeepAlive(poison)
		if err != nil || !ok {
			t.Fatalf("candidate stepSampleLogitsTokenInPool ok=%v err=%v", ok, err)
		}
		if !bytes.Equal(gotHidden, wantHidden) || gotToken != wantToken {
			t.Fatal("GPU-input sampled logits-token path read retained hidden from lastOutPtr")
		}
		if len(candidate.retainedHidden) == 0 || unsafe.Pointer(&gotHidden[0]) != unsafe.Pointer(&candidate.retainedHidden[0]) {
			t.Fatal("GPU-input sampled logits-token path returned transient hidden")
		}
	})
	t.Run("topk-token", func(t *testing.T) {
		control := prepare(t)
		candidate := prepare(t)
		params := model.SampleParams{Temperature: 1, TopK: 7, TopP: 0.75, SuppressTokens: []int32{2, 7}, RepeatPenalty: 1.2}
		history := []int32{4, 5, 5, 31}
		draw := float32(0.42)
		wantHidden, wantToken, ok, err := control.stepSampleTopKTokenInPool(9, params, draw, history)
		if err != nil || !ok {
			t.Fatalf("control stepSampleTopKTokenInPool ok=%v err=%v", ok, err)
		}
		poison := bytes.Repeat([]byte{0x6d}, candidate.arch.Hidden*bf16Size)
		candidate.state.icb.lastOutPtr = &poison[0]
		gotHidden, gotToken, ok, err := candidate.stepSampleTopKTokenInPool(9, params, draw, history)
		runtime.KeepAlive(poison)
		if err != nil || !ok {
			t.Fatalf("candidate stepSampleTopKTokenInPool ok=%v err=%v", ok, err)
		}
		if !bytes.Equal(gotHidden, wantHidden) || gotToken != wantToken {
			t.Fatal("GPU-input sampled TopK-token path read retained hidden from lastOutPtr")
		}
		if len(candidate.retainedHidden) == 0 || unsafe.Pointer(&gotHidden[0]) != unsafe.Pointer(&candidate.retainedHidden[0]) {
			t.Fatal("GPU-input sampled TopK-token path returned transient hidden")
		}
	})
	t.Run("topk-candidates", func(t *testing.T) {
		control := prepare(t)
		candidate := prepare(t)
		params := model.SampleParams{Temperature: 1, TopK: 7, TopP: 0.75, SuppressTokens: []int32{2, 7}}
		wantHidden, wantLogits, wantIDs, ok, err := control.stepSampleTopKCandidatesInPool(9, params)
		if err != nil || !ok {
			t.Fatalf("control stepSampleTopKCandidatesInPool ok=%v err=%v", ok, err)
		}
		poison := bytes.Repeat([]byte{0x5c}, candidate.arch.Hidden*bf16Size)
		candidate.state.icb.lastOutPtr = &poison[0]
		gotHidden, gotLogits, gotIDs, ok, err := candidate.stepSampleTopKCandidatesInPool(9, params)
		runtime.KeepAlive(poison)
		if err != nil || !ok {
			t.Fatalf("candidate stepSampleTopKCandidatesInPool ok=%v err=%v", ok, err)
		}
		if !bytes.Equal(gotHidden, wantHidden) || !bytes.Equal(gotLogits, wantLogits) || !idsEqual(gotIDs, wantIDs) {
			t.Fatal("GPU-input sampled TopK-candidate path read retained hidden from lastOutPtr")
		}
		if len(candidate.retainedHidden) == 0 || unsafe.Pointer(&gotHidden[0]) != unsafe.Pointer(&candidate.retainedHidden[0]) {
			t.Fatal("GPU-input sampled TopK-candidate path returned transient hidden")
		}
	})
}

func TestArchSessionStepGreedyICBWritesRetainedHiddenDirectly(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	control := newQuantICBAllocationSession(t, 32)
	candidate := newQuantICBAllocationSession(t, 32)
	wantToken, wantHidden, ok, err := control.stepGreedyInPool(9, nil, nil)
	if err != nil || !ok {
		t.Fatalf("control stepGreedyInPool ok=%v err=%v", ok, err)
	}
	poison := bytes.Repeat([]byte{0x4b}, candidate.arch.Hidden*bf16Size)
	candidate.state.icb.lastOutPtr = &poison[0]
	gotToken, gotHidden, ok, err := candidate.stepGreedyInPool(9, nil, nil)
	runtime.KeepAlive(poison)
	if err != nil || !ok {
		t.Fatalf("candidate stepGreedyInPool ok=%v err=%v", ok, err)
	}
	if gotToken != wantToken || !bytes.Equal(gotHidden, wantHidden) {
		t.Fatal("greedy ICB path read retained hidden from lastOutPtr")
	}
	if len(candidate.retainedHidden) == 0 || unsafe.Pointer(&gotHidden[0]) != unsafe.Pointer(&candidate.retainedHidden[0]) {
		t.Fatal("greedy ICB path returned transient hidden")
	}
}

func TestArchSessionStepSampleTopKTokenICBMatchesSerial(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	const dModel, nHeads, nKV, headDim, dFF, vocab = 128, 2, 1, 64, 256, 64
	const maxLen = 16
	g, arch := gemma4BF16Fixture(t, dModel, nHeads, nKV, headDim, dFF, vocab, 2)
	serial, err := NewArchSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("NewArchSession serial: %v", err)
	}
	chained, err := NewArchSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("NewArchSession chained: %v", err)
	}
	if chained.state.icb == nil {
		t.Skip("ICB replay unavailable for sampled token chain")
	}
	for _, id := range []int32{1, 5, 3} {
		if _, err := serial.stepID(id); err != nil {
			t.Fatalf("serial prefix stepID(%d): %v", id, err)
		}
		if _, err := chained.stepID(id); err != nil {
			t.Fatalf("chained prefix stepID(%d): %v", id, err)
		}
	}
	params := model.SampleParams{Temperature: 1, TopK: 5, TopP: 0.5, SuppressTokens: []int32{2, 7}}
	draw := model.NewSampler(123).Draw()
	serialHidden, err := serial.stepID(9)
	if err != nil {
		t.Fatalf("serial stepID: %v", err)
	}
	wantToken, ok, err := serial.sampleTopKTokenFromHiddenInPool(serialHidden, params, draw, nil)
	if err != nil {
		t.Fatalf("serial sampleTopKTokenFromHiddenInPool: %v", err)
	}
	if !ok {
		t.Skip("device TopK sampler unavailable")
	}
	gotHidden, gotToken, ok, err := chained.stepSampleTopKTokenInPool(9, params, draw, nil)
	if err != nil {
		t.Fatalf("chained stepSampleTopKTokenInPool: %v", err)
	}
	if !ok {
		t.Fatal("chained stepSampleTopKTokenInPool declined")
	}
	if !bytes.Equal(gotHidden, serialHidden) {
		t.Fatal("chained sampled-token hidden differs from serial stepID hidden")
	}
	if gotToken != wantToken {
		t.Fatalf("chained sampled token = %d, want serial %d", gotToken, wantToken)
	}
	if chained.Pos() != serial.Pos() {
		t.Fatalf("positions diverged: chained=%d serial=%d", chained.Pos(), serial.Pos())
	}
}

func TestArchSessionStepSampleTopKTokenICBReusesHiddenReadback(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	const dModel, nHeads, nKV, headDim, dFF, vocab = 128, 2, 1, 64, 256, 64
	const maxLen = 16
	g, arch := gemma4BF16Fixture(t, dModel, nHeads, nKV, headDim, dFF, vocab, 2)
	serial, err := NewArchSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("NewArchSession serial: %v", err)
	}
	chained, err := NewArchSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("NewArchSession chained: %v", err)
	}
	if chained.state.icb == nil {
		t.Skip("ICB replay unavailable for sampled token chain")
	}
	for _, id := range []int32{1, 5, 3} {
		if _, err := serial.stepID(id); err != nil {
			t.Fatalf("serial prefix stepID(%d): %v", id, err)
		}
		if _, err := chained.stepID(id); err != nil {
			t.Fatalf("chained prefix stepID(%d): %v", id, err)
		}
	}
	params := model.SampleParams{Temperature: 1, TopK: 5, TopP: 0.5, SuppressTokens: []int32{2, 7}}
	sampler := model.NewSampler(123)
	draw1 := sampler.Draw()
	draw2 := sampler.Draw()

	serialHidden1, err := serial.stepID(9)
	if err != nil {
		t.Fatalf("serial first stepID: %v", err)
	}
	wantToken1, ok, err := serial.sampleTopKTokenFromHiddenInPool(serialHidden1, params, draw1, nil)
	if err != nil {
		t.Fatalf("serial first sampleTopKTokenFromHiddenInPool: %v", err)
	}
	if !ok {
		t.Skip("device TopK sampler unavailable")
	}
	gotHidden1, gotToken1, ok, err := chained.stepSampleTopKTokenInPool(9, params, draw1, nil)
	if err != nil {
		t.Fatalf("chained first stepSampleTopKTokenInPool: %v", err)
	}
	if !ok {
		t.Fatal("chained first stepSampleTopKTokenInPool declined")
	}
	if !bytes.Equal(gotHidden1, serialHidden1) {
		t.Fatal("first chained hidden differs from serial stepID hidden")
	}
	if gotToken1 != wantToken1 {
		t.Fatalf("first chained token = %d, want serial %d", gotToken1, wantToken1)
	}
	if len(gotHidden1) == 0 {
		t.Fatal("first chained hidden is empty")
	}
	if len(chained.retainedHidden) == 0 || unsafe.Pointer(&gotHidden1[0]) != unsafe.Pointer(&chained.retainedHidden[0]) {
		t.Fatal("first chained hidden is not returned from retained hidden backing")
	}
	firstPtr := uintptr(unsafe.Pointer(&gotHidden1[0]))
	heldHidden := [][]byte{gotHidden1}

	serialHidden2, err := serial.stepID(gotToken1)
	if err != nil {
		t.Fatalf("serial second stepID: %v", err)
	}
	wantToken2, ok, err := serial.sampleTopKTokenFromHiddenInPool(serialHidden2, params, draw2, nil)
	if err != nil {
		t.Fatalf("serial second sampleTopKTokenFromHiddenInPool: %v", err)
	}
	if !ok {
		t.Skip("device TopK sampler unavailable on second step")
	}
	gotHidden2, gotToken2, ok, err := chained.stepSampleTopKTokenInPool(gotToken1, params, draw2, nil)
	if err != nil {
		t.Fatalf("chained second stepSampleTopKTokenInPool: %v", err)
	}
	if !ok {
		t.Fatal("chained second stepSampleTopKTokenInPool declined")
	}
	if !bytes.Equal(gotHidden2, serialHidden2) {
		t.Fatal("second chained hidden differs from serial stepID hidden")
	}
	if gotToken2 != wantToken2 {
		t.Fatalf("second chained token = %d, want serial %d", gotToken2, wantToken2)
	}
	if len(chained.retainedHidden) == 0 || unsafe.Pointer(&gotHidden2[0]) != unsafe.Pointer(&chained.retainedHidden[0]) {
		t.Fatal("second chained hidden is not returned from retained hidden backing")
	}
	secondPtr := uintptr(unsafe.Pointer(&gotHidden2[0]))
	runtime.KeepAlive(heldHidden)
	if secondPtr != firstPtr {
		t.Fatalf("sampled hidden readback allocated a new backing buffer: first=%#x second=%#x", firstPtr, secondPtr)
	}
}

func TestArchSessionHiddenReadbackScratchReusesBackingBuffer(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	if err := ensureInit(); err != nil {
		t.Skipf("native init unavailable: %v", err)
	}
	sess := &ArchSession{arch: model.Arch{Hidden: 4}}
	first := toBF16Bytes([]float32{1, 2, 3, 4})
	firstBuf := scratchBF16(sess.arch.Hidden)
	copy(unsafe.Slice((*byte)(firstBuf.Contents()), len(first)), first)
	firstHidden := sess.copyHiddenReadback(firstBuf)
	if !bytes.Equal(firstHidden, first) {
		t.Fatalf("first hidden readback = %v, want %v", firstHidden, first)
	}
	if len(firstHidden) == 0 {
		t.Fatal("first hidden readback is empty")
	}
	firstPtr := uintptr(unsafe.Pointer(&firstHidden[0]))
	heldHidden := [][]byte{firstHidden}

	second := toBF16Bytes([]float32{5, 6, 7, 8})
	secondBuf := scratchBF16(sess.arch.Hidden)
	copy(unsafe.Slice((*byte)(secondBuf.Contents()), len(second)), second)
	secondHidden := sess.copyHiddenReadback(secondBuf)
	if !bytes.Equal(secondHidden, second) {
		t.Fatalf("second hidden readback = %v, want %v", secondHidden, second)
	}
	secondPtr := uintptr(unsafe.Pointer(&secondHidden[0]))
	runtime.KeepAlive(heldHidden)
	if secondPtr != firstPtr {
		t.Fatalf("hidden readback allocated a new backing buffer: first=%#x second=%#x", firstPtr, secondPtr)
	}
}

func TestArchSessionStepIDInPoolICBReusesHiddenReadback(t *testing.T) {
	requireNativeRuntime(t)
	g, arch, maxLen := icbSessionStateFixture(t)
	sess := newICBSessionStateFixture(t, g, arch, maxLen)
	if sess.state.icb == nil {
		t.Fatal("fixture must exercise ICB replay")
	}

	var first, firstCopy, second []byte
	var err error
	withAutoreleasePool(func() {
		first, err = sess.stepIDInPool(1)
		if err != nil {
			return
		}
		firstCopy = append([]byte(nil), first...)
		second, err = sess.stepIDInPool(5)
	})
	if err != nil {
		t.Fatalf("stepIDInPool: %v", err)
	}
	if len(first) == 0 || len(second) == 0 {
		t.Fatal("stepIDInPool returned empty hidden")
	}
	if uintptr(unsafe.Pointer(&second[0])) != uintptr(unsafe.Pointer(&first[0])) {
		t.Fatal("ICB stepIDInPool did not reuse session hidden readback backing")
	}
	if bytes.Equal(second, firstCopy) {
		t.Fatal("ICB stepIDInPool reused backing but did not refresh hidden contents")
	}
}

func TestArchSessionStepIDRetainedInPoolNonICBReturnsRetainedHidden(t *testing.T) {
	requireNativeRuntime(t)
	g, arch, maxLen := icbSessionStateFixture(t)
	sess := newICBSessionStateFixture(t, g, arch, maxLen)
	oldICBDisabled := icbDisabledForTest
	icbDisabledForTest = true
	defer func() { icbDisabledForTest = oldICBDisabled }()

	first, err := sess.stepIDRetainedInPool(1)
	if err != nil {
		t.Fatalf("first stepIDRetainedInPool: %v", err)
	}
	if len(first) == 0 {
		t.Fatal("first retained step returned empty hidden")
	}
	if len(sess.retainedHidden) == 0 || unsafe.Pointer(&first[0]) != unsafe.Pointer(&sess.retainedHidden[0]) {
		t.Fatal("non-ICB retained step returned a transient hidden copy instead of retained backing")
	}
	if sess.retainedHiddenBuffer() == nil {
		t.Fatal("non-ICB retained step did not keep a pinned retained hidden buffer")
	}
	firstCopy := append([]byte(nil), first...)
	firstPtr := unsafe.Pointer(&first[0])

	second, err := sess.stepIDRetainedInPool(5)
	if err != nil {
		t.Fatalf("second stepIDRetainedInPool: %v", err)
	}
	if unsafe.Pointer(&second[0]) != firstPtr {
		t.Fatal("non-ICB retained step changed retained hidden backing across same-shape steps")
	}
	if bytes.Equal(second, firstCopy) {
		t.Fatal("non-ICB retained step reused backing but did not refresh hidden contents")
	}
}

func TestArchSessionStepIDRetainedInPoolICBWritesRetainedHiddenDirectly(t *testing.T) {
	requireNativeRuntime(t)
	g, arch, maxLen := icbSessionStateFixture(t)
	control := newICBSessionStateFixture(t, g, arch, maxLen)
	candidate := newICBSessionStateFixture(t, g, arch, maxLen)
	if candidate.state.icb == nil {
		t.Fatal("fixture must exercise ICB replay")
	}

	var want, got []byte
	var err error
	withAutoreleasePool(func() {
		want, err = control.stepIDRetainedInPool(1)
		if err != nil {
			return
		}
		poison := bytes.Repeat([]byte{0x7e}, arch.Hidden*bf16Size)
		candidate.state.icb.lastOutPtr = &poison[0]
		got, err = candidate.stepIDRetainedInPool(1)
		runtime.KeepAlive(poison)
	})
	if err != nil {
		t.Fatalf("stepIDRetainedInPool: %v", err)
	}
	if len(got) == 0 {
		t.Fatal("ICB retained step returned empty hidden")
	}
	if !bytes.Equal(got, want) {
		t.Fatal("ICB retained step read from lastOutPtr instead of writing into retained hidden directly")
	}
	if len(candidate.retainedHidden) == 0 || unsafe.Pointer(&got[0]) != unsafe.Pointer(&candidate.retainedHidden[0]) {
		t.Fatal("ICB retained step returned a transient hidden copy instead of retained hidden backing")
	}
	if candidate.retainedHiddenBuffer() == nil {
		t.Fatal("ICB retained step did not keep a pinned retained hidden buffer")
	}
}

func TestArchSessionStepSampleLogitsTokenICBMatchesSerial(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	const dModel, nHeads, nKV, headDim, dFF, vocab = 128, 2, 1, 64, 256, 64
	const maxLen = 16
	g, arch := gemma4BF16Fixture(t, dModel, nHeads, nKV, headDim, dFF, vocab, 2)
	serial, err := NewArchSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("NewArchSession serial: %v", err)
	}
	chained, err := NewArchSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("NewArchSession chained: %v", err)
	}
	if chained.state.icb == nil {
		t.Skip("ICB replay unavailable for sampled logits token chain")
	}
	for _, id := range []int32{1, 5, 3} {
		if _, err := serial.stepID(id); err != nil {
			t.Fatalf("serial prefix stepID(%d): %v", id, err)
		}
		if _, err := chained.stepID(id); err != nil {
			t.Fatalf("chained prefix stepID(%d): %v", id, err)
		}
	}
	params := model.SampleParams{Temperature: 0.8, MinP: 0.02, SuppressTokens: []int32{2, 7}, RepeatPenalty: 1.2}
	history := []int32{4, 5, 5, 31}
	draw := float32(0.37)
	serialHidden, err := serial.stepID(9)
	if err != nil {
		t.Fatalf("serial stepID: %v", err)
	}
	wantToken, ok, err := serial.sampleLogitsTokenFromHiddenInPool(serialHidden, params, draw, history)
	if err != nil {
		t.Fatalf("serial sampleLogitsTokenFromHiddenInPool: %v", err)
	}
	if !ok {
		t.Skip("device logits sampler unavailable")
	}
	gotHidden, gotToken, ok, err := chained.stepSampleLogitsTokenInPool(9, params, draw, history)
	if err != nil {
		t.Fatalf("chained stepSampleLogitsTokenInPool: %v", err)
	}
	if !ok {
		t.Fatal("chained stepSampleLogitsTokenInPool declined")
	}
	if !bytes.Equal(gotHidden, serialHidden) {
		t.Fatal("chained sampled-logits hidden differs from serial stepID hidden")
	}
	if gotToken != wantToken {
		t.Fatalf("chained sampled logits token = %d, want serial %d", gotToken, wantToken)
	}
	if chained.Pos() != serial.Pos() {
		t.Fatalf("positions diverged: chained=%d serial=%d", chained.Pos(), serial.Pos())
	}
}

func TestArchSessionStepSampleLogitsTokenICBAllocationBudget(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	sess := newQuantICBAllocationSession(t, 32)
	params := model.SampleParams{Temperature: 0.8, MinP: 0.02, SuppressTokens: []int32{2, 7}, RepeatPenalty: 1.2}
	history := []int32{4, 5, 5, 31}
	if _, _, ok, err := sess.stepSampleLogitsTokenInPool(9, params, 0.37, history); err != nil {
		t.Fatalf("stepSampleLogitsTokenInPool warmup: %v", err)
	} else if !ok {
		t.Skip("device logits sampler unavailable")
	}
	allocs := testing.AllocsPerRun(5, func() {
		if _, _, ok, err := sess.stepSampleLogitsTokenInPool(9, params, 0.37, history); err != nil {
			t.Fatalf("stepSampleLogitsTokenInPool: %v", err)
		} else if !ok {
			t.Fatal("stepSampleLogitsTokenInPool declined after warmup")
		}
	})
	if allocs > 40 {
		t.Fatalf("ICB sampled-logits token allocations = %.0f, want <= 40", allocs)
	}
}

func TestArchSessionStepSampleTopKRepeatPenaltyICBMatchesSerial(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	const dModel, nHeads, nKV, headDim, dFF, vocab = 128, 2, 1, 64, 256, 64
	const maxLen = 16
	g, arch := gemma4BF16Fixture(t, dModel, nHeads, nKV, headDim, dFF, vocab, 2)
	serial, err := NewArchSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("NewArchSession serial: %v", err)
	}
	chained, err := NewArchSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("NewArchSession chained: %v", err)
	}
	if chained.state.icb == nil {
		t.Skip("ICB replay unavailable for sampled token chain")
	}
	for _, id := range []int32{1, 5, 3} {
		if _, err := serial.stepID(id); err != nil {
			t.Fatalf("serial prefix stepID(%d): %v", id, err)
		}
		if _, err := chained.stepID(id); err != nil {
			t.Fatalf("chained prefix stepID(%d): %v", id, err)
		}
	}
	params := model.SampleParams{Temperature: 1, TopK: 7, TopP: 0.75, SuppressTokens: []int32{2, 7}, RepeatPenalty: 1.2}
	history := []int32{4, 5, 5, 31}
	draw := model.NewSampler(123).Draw()
	serialHidden, err := serial.stepID(9)
	if err != nil {
		t.Fatalf("serial stepID: %v", err)
	}
	wantToken, ok, err := serial.sampleTopKTokenFromHiddenInPool(serialHidden, params, draw, history)
	if err != nil {
		t.Fatalf("serial sampleTopKTokenFromHiddenInPool: %v", err)
	}
	if !ok {
		t.Skip("device TopK repeat-penalty sampler unavailable")
	}
	gotHidden, gotToken, ok, err := chained.stepSampleTopKTokenInPool(9, params, draw, history)
	if err != nil {
		t.Fatalf("chained stepSampleTopKTokenInPool: %v", err)
	}
	if !ok {
		t.Fatal("chained stepSampleTopKTokenInPool declined")
	}
	if !bytes.Equal(gotHidden, serialHidden) {
		t.Fatal("chained sampled-token hidden differs from serial stepID hidden")
	}
	if gotToken != wantToken {
		t.Fatalf("chained sampled token = %d, want serial %d", gotToken, wantToken)
	}
	if chained.Pos() != serial.Pos() {
		t.Fatalf("positions diverged: chained=%d serial=%d", chained.Pos(), serial.Pos())
	}
}

func TestArchSessionStepSampleTopKTokenICBAllocationBudget(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	sess := newQuantICBAllocationSession(t, 32)
	params := model.SampleParams{Temperature: 1, TopK: 7, TopP: 0.75, SuppressTokens: []int32{2, 7}, RepeatPenalty: 1.2}
	history := []int32{4, 5, 5, 31}
	draw := float32(0.42)
	if _, _, ok, err := sess.stepSampleTopKTokenInPool(9, params, draw, history); err != nil {
		t.Fatalf("stepSampleTopKTokenInPool warmup: %v", err)
	} else if !ok {
		t.Skip("device TopK sampler unavailable")
	}
	allocs := testing.AllocsPerRun(5, func() {
		if _, _, ok, err := sess.stepSampleTopKTokenInPool(9, params, draw, history); err != nil {
			t.Fatalf("stepSampleTopKTokenInPool: %v", err)
		} else if !ok {
			t.Fatal("stepSampleTopKTokenInPool declined after warmup")
		}
	})
	if allocs > 40 {
		t.Fatalf("ICB sampled-TopK token allocations = %.0f, want <= 40", allocs)
	}
}

func TestArchSessionStepSampleTopKTokenICBReturnsRetainedHidden(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	sess := newQuantICBAllocationSession(t, 32)
	params := model.SampleParams{Temperature: 1, TopK: 7, TopP: 0.75, SuppressTokens: []int32{2, 7}, RepeatPenalty: 1.2}
	hidden, _, ok, err := sess.stepSampleTopKTokenInPool(9, params, 0.42, []int32{4, 5, 5, 31})
	if err != nil {
		t.Fatalf("stepSampleTopKTokenInPool: %v", err)
	}
	if !ok {
		t.Skip("device TopK sampler unavailable")
	}
	if len(hidden) == 0 {
		t.Fatal("stepSampleTopKTokenInPool returned empty hidden")
	}
	if len(sess.retainedHidden) == 0 || unsafe.Pointer(&hidden[0]) != unsafe.Pointer(&sess.retainedHidden[0]) {
		t.Fatal("stepSampleTopKTokenInPool returned a transient hidden copy instead of retained hidden backing")
	}
	if sess.retainedHiddenBuffer() == nil {
		t.Fatal("retained hidden backing is not pinned for no-copy head reuse")
	}
}

func TestHeadEncoderSampleTopKCandidatesMatchesFullHead(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	for _, softCap := range []float32{0, 2} {
		t.Run(fmt.Sprintf("softcap_%g", softCap), func(t *testing.T) {
			const dModel, nHeads, nKV, headDim, dFF, vocab = 128, 2, 1, 64, 256, 32
			const maxLen = 16
			g, arch := gemma4BF16Fixture(t, dModel, nHeads, nKV, headDim, dFF, vocab, 2)
			arch.SoftCap = softCap
			sess, err := NewArchSession(g, arch, maxLen)
			if err != nil {
				t.Fatalf("NewArchSession: %v", err)
			}
			if sess.headEnc == nil || !bf16LMHeadTopKUsable(dModel, vocab, 3) {
				t.Skip("head top-k custom kernel unavailable")
			}
			if err := sess.PrefillTokens([]int32{1, 5, 3}); err != nil {
				t.Fatalf("PrefillTokens: %v", err)
			}
			hidden := append([]byte(nil), sess.retainedHidden...)
			const topK = 3
			suppress := []int32{2}
			full, err := sess.head(hidden, false)
			if err != nil {
				t.Fatalf("head: %v", err)
			}
			type candidate struct {
				id int32
				v  float32
			}
			want := make([]candidate, 0, vocab)
			for i := range vocab {
				if int32(i) == suppress[0] {
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
			gotLogits, gotIDs, ok, err := sess.headEnc.sampleTopKCandidates(hidden, topK, suppress)
			if err != nil {
				t.Fatalf("sampleTopKCandidates: %v", err)
			}
			if !ok {
				t.Fatal("sampleTopKCandidates returned ok=false")
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
		})
	}
}

func TestHeadEncoderQuantSampleTopKCandidatesMatchesFullHead(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	const dModel, nHeads, nKV, headDim, dFF, vocab = 128, 2, 1, 64, 256, 256
	const maxLen = 16
	const gs, bits = 64, 4
	arch, err := g4.Config{
		HiddenSize: dModel, NumHiddenLayers: 2, IntermediateSize: dFF,
		NumAttentionHeads: nHeads, NumKeyValueHeads: nKV, HeadDim: headDim,
		VocabSize: vocab, RMSNormEps: 1e-6,
		Quantization: &model.QuantConfig{GroupSize: gs, Bits: bits},
	}.Arch()
	if err != nil {
		t.Fatalf("Arch: %v", err)
	}
	arch.SoftCap = 2
	lm, err := model.Assemble(quantGemma4Tensors(t, arch, gs, bits), arch, model.StandardWeightNames())
	if err != nil {
		t.Fatalf("Assemble: %v", err)
	}
	g, err := loadedToQuant(lm, gs, bits)
	if err != nil {
		t.Fatalf("loadedToQuant: %v", err)
	}
	sess, err := NewArchQuantSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("NewArchQuantSession: %v", err)
	}
	if sess.headEnc == nil || !qmvLogitsTopKUsable(dModel, vocab, gs, bits, 5) {
		t.Skip("quant head top-k custom kernel unavailable")
	}
	if err := sess.PrefillTokens([]int32{1, 5, 3}); err != nil {
		t.Fatalf("PrefillTokens: %v", err)
	}
	hidden := append([]byte(nil), sess.retainedHidden...)
	const topK = 5
	suppress := []int32{2, 7}
	full, err := sess.head(hidden, false)
	if err != nil {
		t.Fatalf("head: %v", err)
	}
	type candidate struct {
		id int32
		v  float32
	}
	want := make([]candidate, 0, vocab)
	for i := range vocab {
		if int32(i) == suppress[0] || int32(i) == suppress[1] {
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
	gotLogits, gotIDs, ok, err := sess.headEnc.sampleTopKCandidates(hidden, topK, suppress)
	if err != nil {
		t.Fatalf("sampleTopKCandidates: %v", err)
	}
	if !ok {
		t.Fatal("sampleTopKCandidates returned ok=false")
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

// TestBidirTokenSpans is a direct table test of the pure span-extraction function bidirTokenSpans
// (#30 r4): no session or fixture needed. Cases cover every branch — the toks={0,0} disabled
// guard, a run that never matches either span token, a single contiguous span ending mid-slice,
// a span left open at the end of ids, two SEPARATE runs of the same span token (a non-span gap
// resets start), and two DIFFERENT span tokens immediately adjacent with no gap (the transition
// arm that both closes the old span and opens a new one in the same iteration).
func TestBidirTokenSpans(t *testing.T) {
	cases := []struct {
		name string
		ids  []int32
		toks [2]int32
		want [][2]int
	}{
		{"disabled when both tokens are zero", []int32{1, 5, 5, 2}, [2]int32{0, 0}, nil},
		{"no id matches either span token", []int32{1, 2, 3}, [2]int32{5, 6}, nil},
		{"single contiguous span ending mid-slice", []int32{1, 2, 5, 5, 5, 3}, [2]int32{5, 0}, [][2]int{{2, 5}}},
		{"span left open at the end", []int32{1, 5, 5}, [2]int32{5, 0}, [][2]int{{1, 3}}},
		{"two separate runs of the same span token", []int32{5, 5, 1, 1, 5}, [2]int32{5, 0}, [][2]int{{0, 2}, {4, 5}}},
		{"two different span tokens adjacent with no gap", []int32{7, 8, 1}, [2]int32{7, 8}, [][2]int{{0, 1}, {1, 2}}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := bidirTokenSpans(tc.ids, tc.toks)
			if !slices.EqualFunc(got, tc.want, func(a, b [2]int) bool { return a == b }) {
				t.Fatalf("bidirTokenSpans(%v, %v) = %v, want %v", tc.ids, tc.toks, got, tc.want)
			}
		})
	}
}

// TestArchSessionRetainHiddenReadbackFrom pins the retained-hidden readback wrapper the ICB step
// paths hand back as `hidden`: it copies arch.Hidden*bf16Size bytes from ptr into the session's
// retained-hidden backing (via rememberRetainedHiddenFrom) and RETURNS that same backing — not the
// source. Pure: no metal device needed. With a device the pinned no-copy path may back the bytes;
// without one it declines to a plain make/copy — either way the return is the retained backing and
// a copy of the source, so the invariants below hold in both lanes.
func TestArchSessionRetainHiddenReadbackFrom(t *testing.T) {
	sess := &ArchSession{arch: model.Arch{Hidden: 4}}
	src := []byte{10, 20, 30, 40, 50, 60, 70, 80} // four bf16 elements
	got := sess.retainHiddenReadbackFrom(&src[0])
	if !bytes.Equal(got, src) {
		t.Fatalf("retainHiddenReadbackFrom = %v, want a copy of %v", got, src)
	}
	if len(sess.retainedHidden) != len(src) || unsafe.Pointer(&got[0]) != unsafe.Pointer(&sess.retainedHidden[0]) {
		t.Fatal("retainHiddenReadbackFrom did not return the session retained-hidden backing")
	}
	src[0] = 99 // mutating the source must not touch the readback — it is a copy, not an alias
	if got[0] == 99 {
		t.Fatal("retainHiddenReadbackFrom aliases the source pointer instead of copying")
	}
}

// TestPrefillChunkWindowsFor pins the pure chunk-width resolver (no env override in play): the width
// is batchedDensePrefillTargetRows (2048) / slidingWindow WHOLE windows, forced to a single window
// below four and capped at eight above. A zero sliding window resolves against the 512 the family
// ships. LTHN_PREFILL_WINDOWS is neutralised so an ambient value cannot skew the default resolution.
func TestPrefillChunkWindowsFor(t *testing.T) {
	t.Setenv("LTHN_PREFILL_WINDOWS", "") // empty reads as unset — force the default resolution
	cases := []struct {
		name    string
		sliding int
		want    int
	}{
		{"zero sliding uses the 512 default: four windows", 0, 4},
		{"512 window divides into exactly four", 512, 4},
		{"256 window: eight windows (2048/256)", 256, 8},
		{"128 window capped at eight (2048/128=16)", 128, 8},
		{"1024 window divides into two: below four keeps a single window", 1024, 1},
		{"4096 window wider than the target: single window", 4096, 1},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := prefillChunkWindowsFor(tc.sliding); got != tc.want {
				t.Fatalf("prefillChunkWindowsFor(%d) = %d, want %d", tc.sliding, got, tc.want)
			}
		})
	}
}

// TestPrefillChunkWindowsForEnvOverride pins the LTHN_PREFILL_WINDOWS override arm: an in-range
// integer (1..8) wins outright, while an out-of-range integer or a non-integer is ignored and the
// sliding-window default resolves instead.
func TestPrefillChunkWindowsForEnvOverride(t *testing.T) {
	t.Setenv("LTHN_PREFILL_WINDOWS", "3")
	if got := prefillChunkWindowsFor(512); got != 3 {
		t.Fatalf("in-range override = %d, want 3", got)
	}
	t.Setenv("LTHN_PREFILL_WINDOWS", "9") // above the 1..8 clamp — ignored
	if got := prefillChunkWindowsFor(512); got != 4 {
		t.Fatalf("out-of-range override = %d, want 4 (default resolution)", got)
	}
	t.Setenv("LTHN_PREFILL_WINDOWS", "wide") // not an integer — ignored
	if got := prefillChunkWindowsFor(256); got != 8 {
		t.Fatalf("non-integer override = %d, want 8 (default resolution)", got)
	}
}

// TestPrefillChunkWindows pins the zero-arg helper: it is prefillChunkWindowsFor(0), i.e. the 512
// default resolving to four windows when no override is set.
func TestPrefillChunkWindows(t *testing.T) {
	t.Setenv("LTHN_PREFILL_WINDOWS", "") // empty reads as unset
	if got := prefillChunkWindows(); got != 4 {
		t.Fatalf("prefillChunkWindows() = %d, want 4", got)
	}
}

// TestSampleOrderMinPKeep pins the pure min-p prefix cut. order indexes into probs and is walked in
// its given (already ranked) sequence; the kept prefix is every leading entry at or above
// probs[order[0]]*minP. Cases cover the two early-return guards (min-p disabled, non-positive keep),
// the trimming path, and the n==0 fallback (a min-p above 1 pushes the threshold past even the head,
// so nothing clears it and the full keep is returned unchanged).
func TestSampleOrderMinPKeep(t *testing.T) {
	cases := []struct {
		name  string
		order []int32
		probs []float32
		keep  int
		minP  float32
		want  int
	}{
		{"min-p disabled returns keep unchanged", []int32{0, 1, 2}, []float32{1, 0.5, 0.1}, 3, 0, 3},
		{"non-positive keep returns keep unchanged", []int32{0, 1}, []float32{1, 0.5}, 0, 0.5, 0},
		{"threshold trims the low tail", []int32{0, 1, 2, 3}, []float32{1.0, 0.5, 0.1, 0.05}, 4, 0.4, 2},
		{"threshold above the head keeps everything", []int32{0, 1, 2}, []float32{1.0, 0.5, 0.1}, 3, 2.0, 3},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := sampleOrderMinPKeep(tc.order, tc.probs, tc.keep, tc.minP); got != tc.want {
				t.Fatalf("sampleOrderMinPKeep(%v, keep=%d, minP=%v) = %d, want %d", tc.order, tc.keep, tc.minP, got, tc.want)
			}
		})
	}
}

// TestSortSampleOrderByProbShortInputIsNoOp pins the len(order) < 2 guard: a single element (and an
// empty slice) sorts to itself without touching the range partitioner.
func TestSortSampleOrderByProbShortInputIsNoOp(t *testing.T) {
	one := []int32{7}
	sortSampleOrderByProb(one, make([]float32, 8))
	if !slices.Equal(one, []int32{7}) {
		t.Fatalf("single-element sort mutated order: %v", one)
	}
	sortSampleOrderByProb(nil, nil) // empty must not panic
}

// TestSortSampleOrderByProbRangeQuicksortPath drives the introsort partition arm: 32 entries clear
// the hi-lo > 12 threshold, exercising the median-of-three pivot, the two recursion-size branches
// and the insertion-sort tail. sampleOrderLess is a TOTAL order (probability descending, ties broken
// by ascending id), so the output is uniquely determined and must equal a stable reference sort by
// the same key.
func TestSortSampleOrderByProbRangeQuicksortPath(t *testing.T) {
	const n = 32
	order := make([]int32, n)
	probs := make([]float32, n)
	for i := range order {
		order[i] = int32(i)
		probs[i] = float32((i*7+3)%11) * 0.1 // deliberate ties across ids exercise the id tiebreak
	}
	want := append([]int32(nil), order...)
	sort.SliceStable(want, func(a, b int) bool {
		ia, ib := want[a], want[b]
		if probs[ia] != probs[ib] {
			return probs[ia] > probs[ib]
		}
		return ia < ib
	})
	sortSampleOrderByProb(order, probs)
	if !slices.Equal(order, want) {
		t.Fatalf("sortSampleOrderByProb = %v, want %v", order, want)
	}
}

// TestArchSessionGenerateFromHiddenMatchesGenerateFromCache pins the greedy generate-from-hidden
// entry point (the unexported wrapper the assistant/drafter seams reach through). Both this path and
// GenerateFromCache feed the SAME boundary hidden into generateFromHiddenInPool — PrefillTokens
// remembers the boundary hidden and clears the retained logits, so GenerateFromCache takes its
// hidden arm — so on two identically prefilled sessions the two must decode the same tokens.
func TestArchSessionGenerateFromHiddenMatchesGenerateFromCache(t *testing.T) {
	requireNativeRuntime(t)
	prompt := []int32{1, 2, 3, 4}
	const maxNew = 4

	fromHidden := newSessionStateFixture(t)
	if err := fromHidden.PrefillTokens(prompt); err != nil {
		t.Fatalf("PrefillTokens: %v", err)
	}
	if len(fromHidden.retainedHidden) != fromHidden.arch.Hidden*bf16Size {
		t.Fatalf("retained hidden len = %d, want %d", len(fromHidden.retainedHidden), fromHidden.arch.Hidden*bf16Size)
	}
	// copy the boundary hidden before generation overwrites the retained backing.
	hidden := append([]byte(nil), fromHidden.retainedHidden...)
	got, err := fromHidden.generateFromHidden(hidden, maxNew, -1, nil)
	if err != nil {
		t.Fatalf("generateFromHidden: %v", err)
	}

	fromCache := newSessionStateFixture(t)
	if err := fromCache.PrefillTokens(prompt); err != nil {
		t.Fatalf("PrefillTokens (cache): %v", err)
	}
	want, err := fromCache.GenerateFromCache(maxNew, -1)
	if err != nil {
		t.Fatalf("GenerateFromCache: %v", err)
	}
	if len(got) != maxNew || !idsEqual(got, want) {
		t.Fatalf("generateFromHidden = %v, want GenerateFromCache continuation %v", got, want)
	}
}

// TestArchSessionGenerateFromHiddenSuppressedMatchesCacheSuppression pins the suppressed sibling:
// masking a handful of ids before argmax must match GenerateFromCacheEachWithSuppression on the same
// boundary hidden and suppression set.
func TestArchSessionGenerateFromHiddenSuppressedMatchesCacheSuppression(t *testing.T) {
	requireNativeRuntime(t)
	prompt := []int32{1, 2, 3, 4}
	const maxNew = 3
	suppress := []int32{0, 1, 2, 3, 4, 5}

	fromHidden := newSessionStateFixture(t)
	if err := fromHidden.PrefillTokens(prompt); err != nil {
		t.Fatalf("PrefillTokens: %v", err)
	}
	hidden := append([]byte(nil), fromHidden.retainedHidden...)
	got, err := fromHidden.generateFromHiddenSuppressed(hidden, maxNew, -1, nil, suppress)
	if err != nil {
		t.Fatalf("generateFromHiddenSuppressed: %v", err)
	}

	fromCache := newSessionStateFixture(t)
	if err := fromCache.PrefillTokens(prompt); err != nil {
		t.Fatalf("PrefillTokens (cache): %v", err)
	}
	want, err := fromCache.GenerateFromCacheEachWithSuppression(maxNew, -1, suppress, nil)
	if err != nil {
		t.Fatalf("GenerateFromCacheEachWithSuppression: %v", err)
	}
	if !idsEqual(got, want) {
		t.Fatalf("generateFromHiddenSuppressed = %v, want suppressed cache continuation %v", got, want)
	}
	for _, id := range got {
		if slices.Contains(suppress, id) {
			t.Fatalf("generateFromHiddenSuppressed emitted a suppressed token %d in %v", id, got)
		}
	}
}

// TestArchSessionGenerateFromHiddenRejectsInvalidInputs pins the four validation arms shared by
// generateFromHidden / generateFromHiddenSuppressed (via generateFromHiddenSuppressedEach): a
// non-positive maxNew, a hidden that is not exactly arch.Hidden*bf16Size, a firstLogits that is not
// exactly arch.Vocab*bf16Size, and a request that would overflow the maxLen cache rows.
func TestArchSessionGenerateFromHiddenRejectsInvalidInputs(t *testing.T) {
	requireNativeRuntime(t)
	sess := newSessionStateFixture(t)
	if err := sess.PrefillTokens([]int32{1, 2}); err != nil {
		t.Fatalf("PrefillTokens: %v", err)
	}
	hidden := append([]byte(nil), sess.retainedHidden...)
	if _, err := sess.generateFromHidden(hidden, 0, -1, nil); err == nil {
		t.Fatal("generateFromHidden accepted maxNew=0")
	}
	if _, err := sess.generateFromHidden(hidden[:len(hidden)-2], 2, -1, nil); err == nil {
		t.Fatal("generateFromHidden accepted a short hidden vector")
	}
	if _, err := sess.generateFromHidden(hidden, 2, -1, []byte{0, 0}); err == nil {
		t.Fatal("generateFromHidden accepted a wrong-size firstLogits")
	}
	if _, err := sess.generateFromHidden(hidden, sess.maxLen+1, -1, nil); err == nil {
		t.Fatal("generateFromHidden accepted a maxNew that overflows the cache rows")
	}
}

// TestArchSessionGenerateEachWithSuppressionStreamsSurvivor pins the streamed suppressed entry
// point: masking every id but one lone survivor makes greedy decode deterministic, so every yielded
// and returned token must be that survivor.
func TestArchSessionGenerateEachWithSuppressionStreamsSurvivor(t *testing.T) {
	requireNativeRuntime(t)
	const dModel, nHeads, nKV, headDim, dFF, vocab = 128, 2, 1, 64, 256, 32
	const maxLen = 16
	g, arch := gemma4BF16Fixture(t, dModel, nHeads, nKV, headDim, dFF, vocab, 1)
	sess, err := NewArchSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("NewArchSession: %v", err)
	}
	const survivor int32 = 7
	suppress := make([]int32, 0, vocab-1)
	for id := range int32(vocab) {
		if id != survivor {
			suppress = append(suppress, id)
		}
	}
	var yielded []int32
	got, err := sess.GenerateEachWithSuppression([]int32{1, 5, 3}, 3, -1, suppress, func(id int32) bool {
		yielded = append(yielded, id)
		return true
	})
	if err != nil {
		t.Fatalf("GenerateEachWithSuppression: %v", err)
	}
	if len(got) != 3 || !idsEqual(yielded, got) {
		t.Fatalf("GenerateEachWithSuppression got/yielded = %v/%v, want three matching streamed tokens", got, yielded)
	}
	for _, id := range got {
		if id != survivor {
			t.Fatalf("GenerateEachWithSuppression emitted %d, want the lone unsuppressed token %d (%v)", id, survivor, got)
		}
	}
}

// TestArchSessionGenerateEachTransformedMatchesGenerateEach pins the committed-token transform
// entry point: an identity transform is a no-op, so the decoded tokens must equal a plain
// GenerateEach on an identical session, AND the transform must be invoked exactly once per generated
// token (proving the wrapper actually wired it through, not silently dropped it).
func TestArchSessionGenerateEachTransformedMatchesGenerateEach(t *testing.T) {
	requireNativeRuntime(t)
	const dModel, nHeads, nKV, headDim, dFF, vocab = 128, 2, 1, 64, 256, 32
	const maxLen = 16
	g, arch := gemma4BF16Fixture(t, dModel, nHeads, nKV, headDim, dFF, vocab, 1)
	prompt := []int32{1, 5, 3}
	const maxNew = 4

	transformed, err := NewArchSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("NewArchSession (transformed): %v", err)
	}
	calls := 0
	var yielded []int32
	got, err := transformed.GenerateEachTransformed(prompt, maxNew, -1, func(id int32) int32 {
		calls++
		return id
	}, func(id int32) bool {
		yielded = append(yielded, id)
		return true
	})
	if err != nil {
		t.Fatalf("GenerateEachTransformed: %v", err)
	}

	plain, err := NewArchSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("NewArchSession (plain): %v", err)
	}
	want, err := plain.GenerateEach(prompt, maxNew, -1, nil)
	if err != nil {
		t.Fatalf("GenerateEach: %v", err)
	}
	if !idsEqual(got, want) {
		t.Fatalf("GenerateEachTransformed (identity) = %v, want GenerateEach %v", got, want)
	}
	if calls != len(got) {
		t.Fatalf("transform invoked %d times, want once per generated token (%d)", calls, len(got))
	}
	if !idsEqual(yielded, got) {
		t.Fatalf("streamed tokens %v differ from returned %v", yielded, got)
	}
}

// TestArchSessionStepSampleTopKCandidatesGPUInputsInPoolMatchesSerial pins the no-history GPU-inputs
// top-k candidate step (the wrapper the sampled decode reaches when history is empty): with the GPU
// next-inputs seam wired and the host embed/PLE closures fenced off (they must NOT be called), it
// must return byte-identical hidden and candidate (logits, ids) to the serial host-input reference
// on the same step.
func TestArchSessionStepSampleTopKCandidatesGPUInputsInPoolMatchesSerial(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	g, arch := pleQuantModel(t, 2, 256, 32, 0)
	serial, err := NewArchQuantSession(g, arch, 16)
	if err != nil {
		t.Fatalf("serial session: %v", err)
	}
	chained, err := NewArchQuantSession(g, arch, 16)
	if err != nil {
		t.Fatalf("chained session: %v", err)
	}
	if chained.encNextInputsGPU == nil {
		t.Fatal("fixture did not wire GPU next-inputs seam")
	}
	for _, id := range []int32{1, 5, 3} {
		if _, err := serial.stepID(id); err != nil {
			t.Fatalf("serial prefix stepID(%d): %v", id, err)
		}
		if _, err := chained.stepID(id); err != nil {
			t.Fatalf("chained prefix stepID(%d): %v", id, err)
		}
	}
	params := model.SampleParams{Temperature: 1, TopK: 7, TopP: 0.75, SuppressTokens: []int32{2, 7}}
	serialHidden, err := serial.stepID(9)
	if err != nil {
		t.Fatalf("serial stepID: %v", err)
	}
	wantLogits, wantIDs, ok, err := serial.sampleTopKCandidatesFromHiddenInPool(serialHidden, params)
	if err != nil || !ok {
		t.Fatalf("serial sampleTopKCandidatesFromHiddenInPool ok=%v err=%v", ok, err)
	}

	chained.embed = func(int32) ([]byte, error) {
		return nil, errors.New("host embed should not be called")
	}
	chained.embedInto = nil
	chained.perLayerInput = func(int32, []byte) ([]byte, error) {
		return nil, errors.New("host PLE should not be called")
	}
	gotHidden, gotLogits, gotIDs, ok, err := chained.stepSampleTopKCandidatesGPUInputsInPool(9, params)
	if err != nil || !ok {
		t.Fatalf("chained stepSampleTopKCandidatesGPUInputsInPool ok=%v err=%v", ok, err)
	}
	if !bytes.Equal(gotHidden, serialHidden) {
		t.Fatal("GPU-input no-history candidate hidden differs from serial host-input hidden")
	}
	if !bytes.Equal(gotLogits, wantLogits) || !idsEqual(gotIDs, wantIDs) {
		t.Fatalf("GPU-input no-history candidates differ from serial: ids got %v want %v", gotIDs, wantIDs)
	}
}
