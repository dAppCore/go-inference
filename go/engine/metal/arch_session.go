// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"math"
	"os"
	"reflect"
	"slices"
	"sync/atomic"
	"time"
	"unsafe"

	core "dappco.re/go"
	"dappco.re/go/inference/decode/tokenizer"
	"dappco.re/go/inference/model"
	"github.com/tmc/apple/metal"
)

// ArchSession is a PERSISTENT decode session: it holds the KV caches across calls, so a
// multi-turn conversation continues without re-prefilling the whole history — each Generate
// only prefills its new prompt and decodes, attending the cache built by previous turns.
//
// The resident buffers (caches + scratch, built once in NewArchSession over the
// archDecodeState) survive across the per-call autorelease pools because device.NewBuffer*
// returns a retained buffer (objc "new" = +1, not autoreleased); the Go session holds the
// reference, so they live until the session is dropped. Single-goroutine: the buffers and
// position are mutable session state with no synchronisation — drive one session from one
// goroutine (one session per conversation).
// ArchSession decodes against resident weights+caches; embed/head are the only
// representation-specific pieces (bf16 or 4-bit), so the prefill+decode loop is shared — set
// by NewArchSession (bf16) or NewArchQuantSession (4-bit).
type ArchSession struct {
	arch          model.Arch
	embed         func(id int32) ([]byte, error)             // token id → its embedded bf16 vector (dModel bytes)
	embedInto     func(dst []byte, id int32) ([]byte, error) // token id → caller-owned embedded bf16 vector
	embedFuncPtr  uintptr
	head          func(hidden []byte, skipSoftcap bool) ([]byte, error)      // hidden bf16 → vocab bf16 logits; skipSoftcap for argmax callers
	greedy        func(hidden []byte, suppress []int32) (int32, bool, error) // optional direct greedy token path; ok=false falls back to head+Greedy
	headEnc       *headEncoder
	headFuncPtr   uintptr
	greedyFuncPtr uintptr
	finalNorm     []byte
	// perLayerInput, when set (gemma4 E2B/E4B), computes the per-token PerLayerInputs tensor
	// from the token id + its embedding; Generate sets it on the state before stepToken. nil
	// for models without the PLE tower.
	perLayerInput func(id int32, emb []byte) ([]byte, error)
	// perLayerInputBatch fills a layer-major PLE slab for a whole token batch in one command
	// buffer (steel GEMM + batched chain) — the K-per-token CB round-trips were the prefill's
	// largest host cost. false = not applicable (small batch, quant tower) → per-token loop.
	perLayerInputBatch func(ids []int32, embs [][]byte, slab []byte) (bool, error)
	// perLayerInputBatchDevice hands a prefill chunk's inputs back DEVICE-RESIDENT,
	// committed but NOT waited (#381): the K main-embed rows (the batched pass's
	// input rows) and the layer-major PLE tensor at exactly outLayers width, built
	// in one command buffer from just the token ids — the pass reads both
	// GPU-ordered on the shared queue, so the host never dequants an embedding,
	// blocks on the build, or copies either tensor. false = not available at that
	// shape → the host embed loop + host-slab path.
	perLayerInputBatchDevice func(ids []int32, outLayers int) (metal.MTLBuffer, metal.MTLBuffer, bool, error)
	// pleHostScratch reuses pinned host staging and intermediate Metal buffers for the host-side
	// resident BF16 PLE projection path. nil when the model has no PLE tower or uses quant PLE projection.
	pleHostScratch *plHostScratch
	// encNextInputsGPU, when set (e2b: 4-bit main+PLE embedding, bf16 PLE projection), encodes the GPU
	// embed-gather (token → embOut, dModel) + the GPU PLE (token, embOut → sc.out, numLayers·pliDim) for
	// one token read from tokenBuf into a shared encoder — the NEXT decode step's emb+pli produced on-GPU
	// with no host round-trip (the submit-ahead pipeline seam). nil → the host embed/PLE path stays.
	encNextInputsGPU func(enc metal.MTLComputeCommandEncoderObject, tokenBuf, embOut metal.MTLBuffer, sc *plGPUScratch) error
	plScratchNew     func() *plGPUScratch
	// recordPeerICB records a SECOND ICB sharing this session's KV caches (its own ping0/pleInput) — the
	// submit-ahead decode keeps two ICBs in flight over the same KV so the host can submit token t+1
	// before reading t. Recorded lazily via peerICB() (most sessions never pipeline). nil when not ICB.
	recordPeerICB      func() (*archICBReplay, error)
	icbPeer            *archICBReplay
	state              archDecodeState
	stateBlockViews    []sessionStateLayerView
	stateBlockViewsICB bool
	stateBlockLayers   []SessionStateLayerBlock
	stateBlockBounds   []int
	turboQuantRotated  []float64
	turboQuantNormed   []float64
	turboQuantPayloads []nativeTurboQuantKVPagePayload
	turboQuantCache    map[nativeTurboQuantKVPayloadCacheKey]nativeTurboQuantKVPagePayload
	kvBlockCachedIDs   []int32
	pos                int // tokens already in the cache (the next token decodes at this position)
	maxLen             int
	// cachedIDs are the token ids currently resident in the KV cache (prompt + generated), tracked so
	// GenerateCached can reuse the longest shared prefix of a new prompt and re-prefill only the suffix.
	cachedIDs []int32
	// cachedPromptIDs/cachedPromptHidden/cachedPromptLogits capture the exact prompt boundary. This
	// mirrors metal's prompt-cache entry hidden/logits replay: an exact prompt hit can decode
	// immediately from saved state instead of re-prefilling the last prompt token or re-running the
	// first head projection just to recreate it.
	cachedPromptIDs          []int32
	cachedPromptHidden       []byte
	cachedPromptLogits       []byte
	cachedPromptHiddenPinned *pinnedNoCopyBytes
	cachedPromptLogitsPinned *pinnedNoCopyBytes
	// retainedHidden is the hidden state at the current session boundary. It is
	// the native equivalent of metal's retained logits boundary for token-only
	// session operation: PrefillTokens/AppendTokens populate it, and
	// GenerateFromCache can continue without requiring a new prompt token.
	retainedHidden       []byte
	retainedLogits       []byte
	retainedHiddenPinned *pinnedNoCopyBytes
	retainedLogitsPinned *pinnedNoCopyBytes
	// restoredKV marks a session whose K/V state came from RestoreState /
	// RestoreStateBlocks rather than live decode. The batched dense prefill's
	// paged→linear sync assumptions do not hold for restored state (the
	// decode-parity carve-out): restored sessions append on the token path.
	restoredKV bool
	// bidirSpanTokens, when non-zero, mark the placeholder token ids whose
	// runs attend BIDIRECTIONALLY during embedding prefill (gemma4_unified
	// image + video spans; use_bidirectional_attention == "vision"). Set at
	// session open by the token model; PrefillTokenEmbeddings detects the
	// runs itself.
	bidirSpanTokens [2]int32
	// verifyBatchedDisabledForTest forces the MTP batched verify to decline
	// (verifyBatchedHiddens / verifyBatchedInto return ok=false) so the caller
	// takes the byte-identical sequential verify lane. Test-only — the honest
	// way to exercise the sequential fallback now that every resident arch
	// (dense + PLE) batches; production never sets it and the guard is a single
	// bool test at the top of each verify entry point (zero decode cost).
	verifyBatchedDisabledForTest bool
	// shards holds the memory-mapped checkpoint + its per-shard no-copy Metal buffers when the
	// session was loaded from a directory zero-copy (LoadGemma4*Dir). The weight []byte fields the
	// embed/head closures and the decode buffers reference are VIEWS into these mmaps, so shards
	// MUST stay alive for the session's life; Close unmaps them. nil for a session built from
	// in-memory weight bytes (NewArchSession over an already-parsed BF16Model) — those weights
	// are heap-owned, nothing to unmap.
	shards *shardBuffers
	// sampled candidate readback scratch. Generation is single-goroutine per
	// session, so the TopK path can reuse these K-sized host buffers instead of
	// allocating logits/ids every sampled token.
	sampleCandidateLogits []byte
	sampleCandidateIDs    []int32
	sampleHeadLogits      []byte
	sampleHidden          []byte
	sampleHistory         []int32
	samplePenaltyIDs      []int32
	samplePenaltyLogits   []byte
	sampleScaled          []float32
	sampleProbs           []float32
	sampleOrder           []int32
	sampleSuppressTokens  []int32
	embedScratch          []byte
	mtpBoundaryNormed     []byte
	mtpProjected          []byte
	mtpDraftNormed        []byte
	mtpDraftHidden        []byte
	mtpDraftLogits        []byte
	mtpDraftTokens        []int32
	mtpDraftVerifyBlock   []int32
	mtpDraftLogitScores   []float32
	mtpDraftLogitSelected []int
	mtpDraftLayerScratch  assistantDraftLayerScratch
	mtpTargetKVScratch    []AssistantTargetKV
	mtpTargetKVByType     []AssistantKVEntry
	mtpTargetKVKeySlabs   [][]byte
	mtpTargetKVValueSlabs [][]byte
	mtpVerifyHiddenPinned *pinnedNoCopyBytes
	mtpVerifyHiddenRows   [][]byte
	mtpVerifyRows         []int32
	// mtpVerifyGreedyRowsBuf holds the verify hiddens for the batched K-row
	// greedy head (one command buffer for all rows) — grown on demand,
	// explicit-copy each verify (pooled Go rows must never back a cached
	// resident buffer).
	mtpVerifyGreedyRowsBuf metal.MTLBuffer
	nextInputToken         metal.MTLBuffer
	nextInputTokenPtr      *int32
	nextInputTokenPinned   *pinnedNoCopyBytes
	nextInputEmb           metal.MTLBuffer
	nextInputEmbPtr        *byte
	nextInputEmbPinned     *pinnedNoCopyBytes
	nextInputEmbHost       []byte
	nextInputPLEHost       []byte
	nextInputPLScratch     *plGPUScratch
	gpuTailPLScratch       [2]*plGPUScratch
}

// Close releases a directory-loaded session's memory-mapped checkpoint. It is safe on a session
// built from in-memory bytes (shards nil ⇒ no-op) and idempotent. Call it once decoding is done;
// the no-copy weight buffers reference the mmap, so do not Close while a Generate/Step is in
// flight (single-goroutine sessions make that the caller's natural discipline).
func (s *ArchSession) Close() error {
	if s == nil {
		return nil
	}
	if s.pleHostScratch != nil {
		s.pleHostScratch.Close()
		s.pleHostScratch = nil
	}
	s.closeSessionOwnedScratch()
	s.closeModelAndDecodeStateReferences()
	if s.shards == nil {
		return nil
	}
	err := s.shards.Close()
	s.shards = nil
	return err
}

func (s *ArchSession) closeSessionOwnedScratch() {
	s.sampleCandidateLogits = nil
	s.sampleCandidateIDs = nil
	s.sampleHeadLogits = nil
	s.sampleHidden = nil
	s.sampleHistory = nil
	s.samplePenaltyIDs = nil
	s.samplePenaltyLogits = nil
	s.sampleScaled = nil
	s.sampleProbs = nil
	s.sampleOrder = nil
	s.sampleSuppressTokens = nil
	s.embedScratch = nil
	s.mtpBoundaryNormed = nil
	s.mtpProjected = nil
	s.mtpDraftNormed = nil
	s.mtpDraftHidden = nil
	s.mtpDraftLogits = nil
	s.mtpDraftTokens = nil
	s.mtpDraftVerifyBlock = nil
	s.mtpDraftLogitScores = nil
	s.mtpDraftLogitSelected = nil
	s.mtpDraftLayerScratch.close()
	s.mtpDraftLayerScratch = assistantDraftLayerScratch{}
	s.mtpTargetKVScratch = nil
	s.mtpTargetKVByType = nil
	s.mtpTargetKVKeySlabs = nil
	s.mtpTargetKVValueSlabs = nil
	if s.mtpVerifyHiddenPinned != nil {
		s.mtpVerifyHiddenPinned.Close()
		s.mtpVerifyHiddenPinned = nil
	}
	s.mtpVerifyHiddenRows = nil
	s.mtpVerifyRows = nil
	s.mtpVerifyGreedyRowsBuf = nil

	s.nextInputToken = nil
	s.nextInputTokenPtr = nil
	if s.nextInputTokenPinned != nil {
		s.nextInputTokenPinned.Close()
		s.nextInputTokenPinned = nil
	}
	s.nextInputEmb = nil
	s.nextInputEmbPtr = nil
	if s.nextInputEmbPinned != nil {
		s.nextInputEmbPinned.Close()
		s.nextInputEmbPinned = nil
	}
	s.nextInputEmbHost = nil
	s.nextInputPLEHost = nil

	if s.nextInputPLScratch != nil {
		s.nextInputPLScratch.Close()
		s.nextInputPLScratch = nil
	}
	for i := range s.gpuTailPLScratch {
		if s.gpuTailPLScratch[i] != nil {
			s.gpuTailPLScratch[i].Close()
			s.gpuTailPLScratch[i] = nil
		}
	}
}

func (s *ArchSession) closeModelAndDecodeStateReferences() {
	s.embed = nil
	s.embedInto = nil
	s.embedFuncPtr = 0
	s.head = nil
	s.greedy = nil
	s.headEnc = nil
	s.headFuncPtr = 0
	s.greedyFuncPtr = 0
	s.finalNorm = nil
	s.perLayerInput = nil
	s.encNextInputsGPU = nil
	s.plScratchNew = nil
	s.recordPeerICB = nil
	s.icbPeer = nil
	if s.state.icb != nil { // the peer shares the KV set; release once, via the primary
		s.state.icb.releaseKVCaches()
	}

	s.state.Close()
	s.state = archDecodeState{}
	s.stateBlockViews = nil
	s.stateBlockViewsICB = false
	s.stateBlockLayers = nil
	s.stateBlockBounds = nil
	s.turboQuantRotated = nil
	s.turboQuantNormed = nil
	s.turboQuantPayloads = nil
	s.turboQuantCache = nil
	s.kvBlockCachedIDs = nil
	s.cachedIDs = nil
	s.cachedPromptIDs = nil
	s.cachedPromptHidden = nil
	s.cachedPromptLogits = nil
	if s.cachedPromptHiddenPinned != nil {
		s.cachedPromptHiddenPinned.Close()
		s.cachedPromptHiddenPinned = nil
	}
	if s.cachedPromptLogitsPinned != nil {
		s.cachedPromptLogitsPinned.Close()
		s.cachedPromptLogitsPinned = nil
	}
	if s.retainedHiddenPinned != nil {
		s.retainedHiddenPinned.Close()
		s.retainedHiddenPinned = nil
	}
	if s.retainedLogitsPinned != nil {
		s.retainedLogitsPinned.Close()
		s.retainedLogitsPinned = nil
	}
	s.retainedHidden = nil
	s.retainedLogits = nil

	s.arch = model.Arch{}
	s.pos = 0
	s.maxLen = 0
}

func (s *ArchSession) embedID(id int32) ([]byte, error) {
	if !s.canUseEmbedScratch() {
		return s.embed(id)
	}
	n := s.arch.Hidden * bf16Size
	if cap(s.embedScratch) < n {
		s.embedScratch = make([]byte, n)
	}
	return s.embedInto(s.embedScratch[:n], id)
}

func (s *ArchSession) markDefaultEmbedFunc() {
	if s == nil || s.embed == nil {
		return
	}
	s.embedFuncPtr = reflect.ValueOf(s.embed).Pointer()
}

func (s *ArchSession) canUseEmbedScratch() bool {
	if s == nil || s.embedInto == nil {
		return false
	}
	if s.embed == nil || s.embedFuncPtr == 0 {
		return true
	}
	return reflect.ValueOf(s.embed).Pointer() == s.embedFuncPtr
}

func (s *ArchSession) copyHiddenReadback(buf metal.MTLBuffer) []byte {
	if buf == nil {
		return nil
	}
	return s.copyHiddenReadbackFrom((*byte)(buf.Contents()))
}

func (s *ArchSession) copyHiddenReadbackFrom(ptr *byte) []byte {
	n := s.arch.Hidden * bf16Size
	if n <= 0 || ptr == nil {
		return nil
	}
	if cap(s.sampleHidden) < n {
		s.sampleHidden = make([]byte, n)
	} else {
		s.sampleHidden = s.sampleHidden[:n]
	}
	copy(s.sampleHidden, unsafe.Slice(ptr, n))
	return s.sampleHidden
}

func (s *ArchSession) retainHiddenReadbackFrom(ptr *byte) []byte {
	s.rememberRetainedHiddenFrom(ptr)
	return s.retainedHidden
}

func (s *ArchSession) retainHiddenDirectFromICB(icb *archICBReplay, emb []byte, pos int, pli []byte) ([]byte, bool) {
	if s == nil || icb == nil {
		return nil, false
	}
	n := s.arch.Hidden * bf16Size
	pinned, ok := s.ensureRetainedHiddenPinned(n)
	if !ok || pinned.buf == nil {
		return nil, false
	}
	s.resetRetainedLogits()
	h := pinned.bytes[:n]
	if !icb.stepBodyIntoBuffer(emb, pos, pli, pinned.buf) {
		return nil, false
	}
	s.retainedHidden = h
	return h, true
}

func (s *ArchSession) headLogitsScratch(hidden []byte, skipSoftcap bool) ([]byte, error) {
	if s.headEnc == nil {
		return s.head(hidden, skipSoftcap)
	}
	var logits []byte
	var err error
	if hiddenBuf := s.retainedHiddenBufferFor(hidden); hiddenBuf != nil {
		if cap(s.sampleHeadLogits) < s.arch.Vocab*bf16Size {
			s.sampleHeadLogits = make([]byte, s.arch.Vocab*bf16Size)
		} else {
			s.sampleHeadLogits = s.sampleHeadLogits[:s.arch.Vocab*bf16Size]
		}
		err = s.headEnc.encodeBufferIntoPool(hiddenBuf, skipSoftcap, s.sampleHeadLogits)
		logits = s.sampleHeadLogits
	} else {
		logits, err = s.headEnc.encodeInto(hidden, skipSoftcap, s.sampleHeadLogits)
	}
	if err != nil {
		return nil, err
	}
	s.sampleHeadLogits = logits
	return logits, nil
}

func (s *ArchSession) markDefaultHeadFunc() {
	if s == nil || s.head == nil {
		return
	}
	s.headFuncPtr = reflect.ValueOf(s.head).Pointer()
}

func (s *ArchSession) markDefaultGreedyFunc() {
	if s == nil || s.greedy == nil {
		return
	}
	s.greedyFuncPtr = reflect.ValueOf(s.greedy).Pointer()
}

func (s *ArchSession) canUseHeadLogitsScratch() bool {
	return s != nil && s.headEnc != nil && s.head != nil && s.headFuncPtr != 0 && reflect.ValueOf(s.head).Pointer() == s.headFuncPtr
}

func (s *ArchSession) canUseDirectHeadGreedy() bool {
	return s != nil && s.canUseHeadLogitsScratch() && s.greedy != nil && s.greedyFuncPtr != 0 &&
		reflect.ValueOf(s.greedy).Pointer() == s.greedyFuncPtr && s.headEnc.directGreedyUsable()
}

func (s *ArchSession) directGreedyFromHiddenInPool(hidden []byte, suppress []int32) (int32, bool, error) {
	if s.canUseDirectHeadGreedy() {
		if hiddenBuf := s.retainedHiddenBufferFor(hidden); hiddenBuf != nil {
			return s.headEnc.greedyBufferInPool(hiddenBuf, suppress)
		}
	}
	return s.greedy(hidden, suppress)
}

func (s *ArchSession) sampleHistoryScratch(maxNew int) []int32 {
	if maxNew <= 0 {
		s.sampleHistory = s.sampleHistory[:0]
		return s.sampleHistory
	}
	if cap(s.sampleHistory) < maxNew {
		s.sampleHistory = make([]int32, 0, maxNew)
	} else {
		s.sampleHistory = s.sampleHistory[:0]
	}
	return s.sampleHistory
}

func (s *ArchSession) sampleHistoryScratchFor(params model.SampleParams, maxNew int) []int32 {
	if params.RepeatPenalty <= 1 {
		return s.sampleHistory[:0]
	}
	return s.sampleHistoryScratch(maxNew)
}

func (s *ArchSession) repeatPenaltyLogitsScratch(logits []byte, vocab int, history []int32, penalty float32) ([]byte, error) {
	if len(logits) != vocab*bf16Size {
		return nil, core.NewError("native.applyRepeatPenalty: logits must be vocab bf16 bytes")
	}
	if penalty <= 1 || len(history) == 0 {
		return logits, nil
	}
	ids := s.repeatPenaltyIDsScratch(vocab, history)
	if len(ids) == 0 {
		return logits, nil
	}
	if cap(s.samplePenaltyLogits) < len(logits) {
		s.samplePenaltyLogits = make([]byte, len(logits))
	} else {
		s.samplePenaltyLogits = s.samplePenaltyLogits[:len(logits)]
	}
	copy(s.samplePenaltyLogits, logits)
	applyRepeatPenaltySortedIDsBF16(s.samplePenaltyLogits, ids, penalty)
	return s.samplePenaltyLogits, nil
}

func (s *ArchSession) repeatPenaltyIDsScratch(vocab int, history []int32) []int32 {
	if cap(s.samplePenaltyIDs) < len(history) {
		s.samplePenaltyIDs = make([]int32, 0, len(history))
	} else {
		s.samplePenaltyIDs = s.samplePenaltyIDs[:0]
	}
	for _, id := range history {
		if id >= 0 && int(id) < vocab {
			s.samplePenaltyIDs = append(s.samplePenaltyIDs, id)
		}
	}
	if len(s.samplePenaltyIDs) == 0 {
		return nil
	}
	slices.Sort(s.samplePenaltyIDs)
	s.samplePenaltyIDs = slices.Compact(s.samplePenaltyIDs)
	return s.samplePenaltyIDs
}

func (s *ArchSession) suppressionTokensScratch(base, extra []int32) []int32 {
	if len(extra) == 0 {
		return base
	}
	if len(base) == 0 {
		return extra
	}
	allExtraSuppressed := true
	for _, token := range extra {
		if !nativeTokenInSet(token, base) {
			allExtraSuppressed = false
			break
		}
	}
	if allExtraSuppressed {
		return base
	}
	wantCap := len(base) + len(extra)
	if cap(s.sampleSuppressTokens) < wantCap {
		s.sampleSuppressTokens = make([]int32, 0, wantCap)
	} else {
		s.sampleSuppressTokens = s.sampleSuppressTokens[:0]
	}
	s.sampleSuppressTokens = append(s.sampleSuppressTokens, base...)
	for _, token := range extra {
		if nativeTokenInSet(token, s.sampleSuppressTokens) {
			continue
		}
		s.sampleSuppressTokens = append(s.sampleSuppressTokens, token)
	}
	return s.sampleSuppressTokens
}

func (s *ArchSession) nextInputTokenBuffer(id int32) metal.MTLBuffer {
	if s.nextInputToken == nil {
		if pinned, err := newPinnedNoCopyBytes(4); err == nil {
			s.nextInputTokenPinned = pinned
			s.nextInputToken = pinned.buf
			s.nextInputTokenPtr = (*int32)(unsafe.Pointer(&pinned.bytes[0]))
		} else {
			s.nextInputToken = device.NewBufferWithLengthOptions(4, metal.MTLResourceStorageModeShared)
			s.nextInputTokenPtr = (*int32)(s.nextInputToken.Contents())
		}
	}
	*s.nextInputTokenPtr = id
	return s.nextInputToken
}

func (s *ArchSession) nextInputEmbBuffer(dModel int) metal.MTLBuffer {
	n := dModel * bf16Size
	if n <= 0 {
		return nil
	}
	if s.nextInputEmb == nil || int(bufferLengthFast(s.nextInputEmb)) != n {
		if s.nextInputEmbPinned != nil {
			s.nextInputEmbPinned.Close()
			s.nextInputEmbPinned = nil
		}
		if pinned, err := newPinnedNoCopyBytes(n); err == nil {
			s.nextInputEmbPinned = pinned
			s.nextInputEmb = pinned.buf
			s.nextInputEmbPtr = (*byte)(unsafe.Pointer(&pinned.bytes[0]))
		} else {
			s.nextInputEmb = device.NewBufferWithLengthOptions(uint(n), metal.MTLResourceStorageModeShared)
			s.nextInputEmbPtr = (*byte)(s.nextInputEmb.Contents())
		}
	}
	return s.nextInputEmb
}

func (s *ArchSession) nextInputEmbReadback(dModel int) []byte {
	n := dModel * bf16Size
	if n <= 0 {
		return nil
	}
	if s.nextInputEmbPinned != nil && len(s.nextInputEmbPinned.bytes) == n {
		return s.nextInputEmbPinned.bytes[:n]
	}
	if cap(s.nextInputEmbHost) < n {
		s.nextInputEmbHost = make([]byte, n)
	}
	return s.nextInputEmbHost[:n]
}

func (s *ArchSession) nextInputPLEReadback(plDim int) []byte {
	n := plDim * bf16Size
	if n <= 0 {
		return nil
	}
	if s.nextInputPLScratch != nil && s.nextInputPLScratch.outPinned != nil && len(s.nextInputPLScratch.outPinned.bytes) == n {
		return s.nextInputPLScratch.outPinned.bytes[:n]
	}
	if cap(s.nextInputPLEHost) < n {
		s.nextInputPLEHost = make([]byte, n)
	}
	return s.nextInputPLEHost[:n]
}

func (s *ArchSession) nextInputPLScratchBuffer() *plGPUScratch {
	if s.nextInputPLScratch == nil {
		s.nextInputPLScratch = s.plScratchNew()
	}
	return s.nextInputPLScratch
}

func (s *ArchSession) gpuTailPLScratchBuffer(slot int) *plGPUScratch {
	if s.gpuTailPLScratch[slot] == nil {
		s.gpuTailPLScratch[slot] = s.plScratchNew()
	}
	return s.gpuTailPLScratch[slot]
}

// NewArchSession builds a session over assembled bf16 weights: it allocates the resident
// per-layer buffers + caches once (empty), ready for Generate to fill incrementally. The weights
// are uploaded into owned Metal buffers (the in-memory path). The directory loader uses
// newArchSessionShards to bind them zero-copy from the shard mmaps instead.
func NewArchSession(g *BF16Model, arch model.Arch, maxLen int) (*ArchSession, error) {
	return newArchSessionShards(g, arch, maxLen, nil)
}

// newArchSessionShards is NewArchSession with an optional zero-copy weight source: when sb is
// non-nil, every per-layer + bookend weight is bound as a no-copy view into the shard mmaps (no
// upload, no second resident copy); when nil, the weights are uploaded into owned buffers (the
// in-memory path). The decode is byte-identical either way — only the weight binding differs.
func newArchSessionShards(g *BF16Model, arch model.Arch, maxLen int, sb *shardBuffers) (*ArchSession, error) {
	return newArchSessionShardsWithHead(g, arch, maxLen, sb, nil)
}

func newArchSessionShardsWithHead(g *BF16Model, arch model.Arch, maxLen int, sb *shardBuffers, sharedHead *headEncoder) (*ArchSession, error) {
	return newArchSessionShardsWithHeadConfig(g, arch, maxLen, sb, sharedHead, archSessionConfig{})
}

func newArchSessionShardsWithHeadConfig(g *BF16Model, arch model.Arch, maxLen int, sb *shardBuffers, sharedHead *headEncoder, cfg archSessionConfig) (*ArchSession, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	if g == nil || len(g.Layers) != len(arch.Layer) {
		return nil, core.NewError("native.NewArchSession: weights/arch layer count mismatch")
	}
	if maxLen <= 0 {
		return nil, core.NewError("native.NewArchSession: maxLen must be > 0")
	}
	attnScale := attnScaleOf(arch)
	embedScale := embedScaleOf(arch)
	var sess *ArchSession
	var buildErr error
	withAutoreleasePool(func() {
		lb, moeWeights, berr := buildBF16ArchLayerBufs(g.Layers, arch.Layer, arch.Hidden, arch.Heads, arch.KVHeads, arch.HeadDim, arch.FF, maxLen, arch.SlidingWindow, sb)
		if berr != nil {
			buildErr = berr
			return
		}
		state := newArchDecodeState(arch.Layer, lb, moeWeights, arch.Hidden, arch.Heads, arch.KVHeads, arch.HeadDim, arch.FF, arch.SlidingWindow, arch.RotaryDim, arch.RotaryDimLocal, arch.RopeBase, arch.RopeLocalBase, attnScale, arch.Eps, arch.ValueNorm, maxLen)
		state.ropeFreqs = uploadRopePeriods(arch.RopeFreqs) // YaRN long-context spectrum (nil ⇒ base rope)
		if err := state.initDevicePagedKVWithPrealloc(cfg.pagedKVPageSize, cfg.pagedKVPrealloc); err != nil {
			buildErr = err
			return
		}
		// gemma4 per-layer-input tower (E2B/E4B), bf16 sibling of the quant session: the per-layer
		// gates carry bf16 bytes (bits 0 ⇒ the decode applies PerLayerInputGateBF16, not the qmv).
		if g.HasPLE() {
			state.pliDim = arch.PerLayerInputHidden
			state.ple = make([]pleLayer, len(g.Layers))
			for i := range g.Layers {
				if len(g.Layers[i].PostPerLayerInputNormW) > 0 {
					state.ple[i] = pleLayer{
						gate:     QuantWeight{Packed: g.Layers[i].PerLayerGate},
						proj:     QuantWeight{Packed: g.Layers[i].PerLayerProjection},
						postNorm: g.Layers[i].PostPerLayerInputNormW,
					}
				}
			}
		}
		// zero-copy head: bind the [vocab×dModel] head weight no-copy, resolved once, reused every
		// token (kills the per-token re-upload balloon). nil ⇒ no shards / unresolved ⇒ upload head.
		head := sharedHead
		if head == nil {
			var herr error
			head, herr = newHeadEncoder(sb, g.FinalNorm, g.LMHead, nil, nil, arch.Hidden, arch.Vocab, 0, 0, arch.Eps, arch.SoftCap, false)
			if herr != nil {
				buildErr = herr
				return
			}
		}
		sess = &ArchSession{
			arch: arch, state: state, maxLen: maxLen, headEnc: head, finalNorm: g.FinalNorm,
			embed: func(id int32) ([]byte, error) {
				return embedTokenBF16(g.Embed, id, arch.Vocab, arch.Hidden, embedScale)
			},
			embedInto: func(dst []byte, id int32) ([]byte, error) {
				return embedTokenBF16Into(dst, g.Embed, id, arch.Vocab, arch.Hidden, embedScale)
			},
			head: func(hidden []byte, skipSoftcap bool) ([]byte, error) {
				if head != nil {
					return head.encode(hidden, skipSoftcap)
				}
				sc := arch.SoftCap
				if skipSoftcap {
					sc = 0 // LMHeadBF16 skips the softcap when softCap<=0
				}
				return LMHeadBF16(hidden, g.FinalNorm, g.LMHead, arch.Hidden, arch.Vocab, arch.Eps, sc)
			},
			greedy: func(hidden []byte, suppress []int32) (int32, bool, error) {
				if head == nil {
					return 0, false, nil
				}
				return head.greedyInPool(hidden, suppress)
			},
		}
		sess.markDefaultEmbedFunc()
		sess.markDefaultHeadFunc()
		sess.markDefaultGreedyFunc()
		if g.HasPLE() {
			var pleProjView bufView // resident no-copy bf16 PLE projection — bound once at its shard offset, not re-uploaded per token
			if sb != nil {
				pleProjView, _ = sb.bufFor(g.PerLayerModelProjW)
			}
			var pleScratch *plHostScratch
			if pleProjView.buf != nil {
				plDim := len(arch.Layer) * arch.PerLayerInputHidden
				projScale := float32(1.0 / math.Sqrt(float64(arch.Hidden)))
				pleScratch, buildErr = newPLHostScratch(plDim, arch.Hidden, projScale)
				if buildErr != nil {
					return
				}
				sess.pleHostScratch = pleScratch
			}
			sess.perLayerInput = func(id int32, emb []byte) ([]byte, error) {
				pv := pleProjView
				scratch := pleScratch
				if pleResidentDisabled { // call-time host-path toggle (byte-identity test hook; always false in production)
					pv = bufView{}
					scratch = nil
				}
				return PerLayerInputs(g.EmbedPerLayer, nil, nil, g.PerLayerModelProjW, nil, nil, g.PerLayerProjNormW, id, emb, arch.PerLayerInputVocab, len(arch.Layer), arch.PerLayerInputHidden, arch.Hidden, 0, 0, 0, 0, arch.Eps, pv, scratch)
			}
			if pleProjView.buf != nil {
				// the K-token slab builder: one steel GEMM + batched chain in ONE command buffer
				// instead of K per-token CB round-trips (the 183ms/512-token host wall).
				batchScratch := &pleBatchScratch{}
				sess.perLayerInputBatch = func(ids []int32, embs [][]byte, slab []byte) (bool, error) {
					if pleResidentDisabled {
						return false, nil
					}
					return perLayerInputsBatchIntoSlab(batchScratch, g.EmbedPerLayer, pleProjView, g.PerLayerProjNormW, ids, embs, slab, arch.PerLayerInputVocab, len(arch.Layer), arch.PerLayerInputHidden, arch.Hidden, arch.Eps)
				}
			}
			// GPU next-inputs seam (full-bf16 E-family): produce the next step's emb+pli on-GPU
			// from a token-id buffer (no host round-trip) — the chained/pipelined decode's gate.
			// Both tables are dense bf16, so the gather is the bf16 row kernel; the projection is
			// the same resident bf16 gemv the host closure dispatches (byte-identity preserved).
			embedTableBuf := residentBytes(g.Embed)
			pleTableBuf := residentBytes(g.EmbedPerLayer)
			pleNormBuf := residentBytes(g.PerLayerProjNormW)
			projWBuf, projWOff := pleProjView.buf, pleProjView.off
			if projWBuf == nil {
				projWBuf = residentBytes(g.PerLayerModelProjW)
				projWOff = 0
			}
			if embedTableBuf != nil && pleTableBuf != nil && pleNormBuf != nil && projWBuf != nil {
				numLayers, pliDim, dModel := len(arch.Layer), arch.PerLayerInputHidden, arch.Hidden
				plDim := numLayers * pliDim
				embScalePLE := float32(math.Sqrt(float64(pliDim)))
				projScale := float32(1.0 / math.Sqrt(float64(dModel)))
				sess.plScratchNew = func() *plGPUScratch { return newPLGPUScratch(plDim, projScale) }
				sess.encNextInputsGPU = func(enc metal.MTLComputeCommandEncoderObject, tokenBuf, embOut metal.MTLBuffer, sc *plGPUScratch) error {
					gpso, gerr := embedGatherRowBF16Pipeline()
					if gerr != nil {
						return gerr
					}
					encEmbedGatherRowBF16Object(enc, gpso, tokenBuf, embedTableBuf, embOut, 0, 0, dModel, embedScale)
					return encPerLayerInputsGPUBF16Object(enc, gpso, tokenBuf, embOut, pleTableBuf, 0, projWBuf, projWOff, pleNormBuf, sc, numLayers, pliDim, dModel, embScalePLE, arch.Eps)
				}
			}
		} else {
			// GPU next-inputs seam, non-PLE dense bf16 (12B/31B): the only per-step input is the
			// token's embedding, so the seam is the bf16 row-gather alone — the zero-value
			// plGPUScratch placeholder satisfies the chained/pipelined gates, exactly as on the
			// quant constructor's dense seam.
			dModel := arch.Hidden
			embedTableBuf := residentBytes(g.Embed)
			if embedTableBuf != nil {
				sess.plScratchNew = func() *plGPUScratch { return &plGPUScratch{} }
				sess.encNextInputsGPU = func(enc metal.MTLComputeCommandEncoderObject, tokenBuf, embOut metal.MTLBuffer, _ *plGPUScratch) error {
					gpso, gerr := embedGatherRowBF16Pipeline()
					if gerr != nil {
						return gerr
					}
					encEmbedGatherRowBF16Object(enc, gpso, tokenBuf, embedTableBuf, embOut, 0, 0, dModel, embedScale)
					return nil
				}
			}
		}
		// bf16 incremental ICB encode-bypass — the quant constructor's block with the bf16
		// recorder (recordArchICBBF16): record the decode stack once + replay it per Step/
		// StepWithID instead of re-encoding every layer every token. The replay holds its OWN
		// linear/ring caches; the PLE runtime wraps the session's perLayerInput closure.
		if sess.icbEligible() {
			var pleRuntime *archDecodePLEInputs
			if g.HasPLE() {
				pleRuntime = &archDecodePLEInputs{compute: sess.perLayerInput}
			}
			// per-owner caches (sliding = bounded ring; global = maxLen rows), with the
			// q8 opt-in allocating int8+scale caches on qualifying global owners (#367).
			kCaches, vCaches, icbKVQ8 := allocArchICBCaches(arch.Layer, arch.KVHeads, arch.HeadDim, maxLen, arch.SlidingWindow)
			rope := icbRope{
				base: arch.RopeBase, localBase: arch.RopeLocalBase,
				rotaryDim: arch.RotaryDim, rotaryDimLocal: arch.RotaryDimLocal,
				globalHeadDim: state.globalHeadDim,
				globalFreqs:   state.globalRopeFreqs, freqs: state.ropeFreqs,
			}
			rep, rerr := recordArchICBBF16(g.Layers, arch.Layer, kCaches, vCaches, pleRuntime, arch.PerLayerInputHidden, arch.Hidden, arch.Heads, arch.KVHeads, arch.HeadDim, maxLen, arch.FF, arch.SlidingWindow, rope, attnScale, arch.Eps, arch.ValueNorm, icbKVQ8)
			if rerr != nil {
				buildErr = rerr
				return
			}
			sess.state.icb = rep
			// Recorder for a PEER ICB sharing these KV caches (own ping0/pleInput) — the submit-ahead
			// decode keeps two in flight over the same KV. Lazily invoked; most sessions never pipeline.
			sess.recordPeerICB = func() (*archICBReplay, error) {
				return recordArchICBBF16(g.Layers, arch.Layer, kCaches, vCaches, pleRuntime, arch.PerLayerInputHidden, arch.Hidden, arch.Heads, arch.KVHeads, arch.HeadDim, maxLen, arch.FF, arch.SlidingWindow, rope, attnScale, arch.Eps, arch.ValueNorm, icbKVQ8)
			}
			if pipelinedGPUDecodeEnabled {
				peer, perr := sess.recordPeerICB()
				if perr != nil {
					buildErr = perr
					return
				}
				sess.icbPeer = peer
			}
		}
	})
	if buildErr != nil {
		return nil, buildErr
	}
	return sess, nil
}

// NewArchQuantSession builds a persistent session over assembled 4-bit weights — the quant
// sibling of NewArchSession. Same resident caches + shared prefill/decode loop; only the
// embed/head closures differ (EmbedTokensQuant / LMHeadQuant over the packed embedding) and
// the layer buffers carry qmv projectors (buildQuantArchLayerBufs). Per-attention-type RoPE
// applies here too (the state is built with both bases).
func NewArchQuantSession(g *QuantModel, arch model.Arch, maxLen int) (*ArchSession, error) {
	return newArchQuantSessionShards(g, arch, maxLen, nil)
}

// newArchQuantSessionShards is NewArchQuantSession with an optional zero-copy weight source.
// sb is kept alive on the session and the per-layer quant weights DO bind as no-copy shard views
// (buildQuantArchLayerBufs receives sb; the live loaders pass a real one).
//
// History, because a stale version of this comment mis-guided sessions for months: an earlier
// build quarantined the quant layers onto the copy path over a "cross-layer NaN" that looked like
// an aliasing hazard. The actual root cause was FILE OFFSET ALIGNMENT — safetensors gives no
// alignment guarantee (e2b-4bit's data section starts at 8+header = ≡2 mod 4, so roughly half the
// U32 packed tensors land 4-byte-misaligned), and binding a misaligned offset as `device const
// uint*` reads garbage. "One layer fine, many layers NaN" was just per-tensor offset parity.
// The cure lives in bufForAligned: an aligned tensor binds as a no-copy view, a misaligned one
// falls back to a private copy of JUST that tensor (mustBufFor4 enforces the packed-U32 rule).
// There is no aliasing hazard; do not re-quarantine.
func newArchQuantSessionShards(g *QuantModel, arch model.Arch, maxLen int, sb *shardBuffers) (*ArchSession, error) {
	return newArchQuantSessionShardsWithHead(g, arch, maxLen, sb, nil)
}

func newArchQuantSessionShardsWithHead(g *QuantModel, arch model.Arch, maxLen int, sb *shardBuffers, sharedHead *headEncoder) (*ArchSession, error) {
	return newArchQuantSessionShardsWithHeadConfig(g, arch, maxLen, sb, sharedHead, archSessionConfig{})
}

func newArchQuantSessionShardsWithHeadConfig(g *QuantModel, arch model.Arch, maxLen int, sb *shardBuffers, sharedHead *headEncoder, cfg archSessionConfig) (*ArchSession, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	if g == nil || len(g.Layers) != len(arch.Layer) {
		return nil, core.NewError("native.NewArchQuantSession: weights/arch layer count mismatch")
	}
	if maxLen <= 0 {
		return nil, core.NewError("native.NewArchQuantSession: maxLen must be > 0")
	}
	attnScale := attnScaleOf(arch)
	embedScale := embedScaleOf(arch)
	gs, bits := g.GroupSize, g.Bits
	var sess *ArchSession
	var buildErr error
	withAutoreleasePool(func() {
		// sb (no-copy) for the per-layer quant weights. The documented "cross-layer multi-bind NaN"
		// hypothesis = the packed uint32 weights bound at non-4-aligned offsets (Metal can't do a
		// misaligned uint32 read); bufFor now copies only those (mustBufFor4), aligned stay zero-copy.
		// If the smoke is coherent this reclaims the 4-bit 2× resident; if not, revert to nil.
		lb, moeQuant, berr := buildQuantArchLayerBufs(g.Layers, arch.Layer, arch.Hidden, arch.Heads, arch.KVHeads, arch.HeadDim, arch.FF, maxLen, arch.SlidingWindow, sb)
		if berr != nil {
			buildErr = berr
			return
		}
		moeWeights := make([]*MoELayerWeights, len(arch.Layer)) // bf16 MoE unused on the quant path
		state := newArchDecodeState(arch.Layer, lb, moeWeights, arch.Hidden, arch.Heads, arch.KVHeads, arch.HeadDim, arch.FF, arch.SlidingWindow, arch.RotaryDim, arch.RotaryDimLocal, arch.RopeBase, arch.RopeLocalBase, attnScale, arch.Eps, arch.ValueNorm, maxLen)
		state.moeQuant = moeQuant
		state.moeScratchOwnable = true // long-lived session: own the MoE scratch, decode wait-free
		if err := state.initDevicePagedKVWithPrealloc(cfg.pagedKVPageSize, cfg.pagedKVPrealloc); err != nil {
			buildErr = err
			return
		}
		// gemma4 per-layer-input tower (E2B/E4B): the per-layer gates + the per-token tensor.
		if g.HasPLE() {
			state.pliDim = arch.PerLayerInputHidden
			state.ple = make([]pleLayer, len(g.Layers))
			for i := range g.Layers {
				if len(g.Layers[i].PostPerLayerInputNormW) > 0 {
					state.ple[i] = pleLayer{
						gate: g.Layers[i].PerLayerGate, proj: g.Layers[i].PerLayerProjection,
						postNorm: g.Layers[i].PostPerLayerInputNormW, groupSize: gs, bits: bits,
					}
				}
			}
		}
		// zero-copy 4-bit head: bind the tied [vocab×dModel] packed embedding + scales/biases no-copy,
		// resolved once, reused every token — this is the projection the per-token balloon lived on
		// (the ~503 MB tied embedding re-uploaded per token at 12B). A single qmv dispatch over the
		// shard buffer is byte-identical (the cross-layer hazard that gates the quant LAYER weights
		// does not apply to a one-shot head). nil ⇒ no shards / unresolved ⇒ the upload head.
		head := sharedHead
		if head == nil {
			var herr error
			head, herr = newHeadEncoder(sb, g.FinalNorm, g.LMHead, g.LMHeadScales, g.LMHeadBiases, arch.Hidden, arch.Vocab, gs, bits, arch.Eps, arch.SoftCap, true)
			if herr != nil {
				buildErr = herr
				return
			}
		}
		sess = &ArchSession{
			arch: arch, state: state, maxLen: maxLen, headEnc: head, finalNorm: g.FinalNorm,
			embed: func(id int32) ([]byte, error) {
				return embedTokenQuant(g.Embed, g.EmbedScales, g.EmbedBiases, id, arch.Vocab, arch.Hidden, gs, bits, embedScale)
			},
			embedInto: func(dst []byte, id int32) ([]byte, error) {
				return embedTokenQuantInto(dst, g.Embed, g.EmbedScales, g.EmbedBiases, id, arch.Vocab, arch.Hidden, gs, bits, embedScale)
			},
			head: func(hidden []byte, skipSoftcap bool) ([]byte, error) {
				if head != nil {
					return head.encode(hidden, skipSoftcap)
				}
				sc := arch.SoftCap
				if skipSoftcap {
					sc = 0 // LMHeadQuant skips the softcap when softCap<=0
				}
				return LMHeadQuant(hidden, g.FinalNorm, g.LMHead, g.LMHeadScales, g.LMHeadBiases, arch.Hidden, arch.Vocab, gs, bits, arch.Eps, sc)
			},
			greedy: func(hidden []byte, suppress []int32) (int32, bool, error) {
				if head == nil {
					return 0, false, nil
				}
				return head.greedyInPool(hidden, suppress)
			},
		}
		sess.markDefaultEmbedFunc()
		sess.markDefaultHeadFunc()
		sess.markDefaultGreedyFunc()
		if g.HasPLE() {
			var pleProjView bufView // resident no-copy PLE projection when it's bf16 (e2b: no proj scales) — bound once, not re-uploaded per token
			if sb != nil && len(g.PerLayerModelProjScales) == 0 {
				pleProjView, _ = sb.bufFor(g.PerLayerModelProjW)
			}
			var pleScratch *plHostScratch
			if pleProjView.buf != nil {
				plDim := len(arch.Layer) * arch.PerLayerInputHidden
				projScale := float32(1.0 / math.Sqrt(float64(arch.Hidden)))
				pleScratch, buildErr = newPLHostScratch(plDim, arch.Hidden, projScale)
				if buildErr != nil {
					return
				}
				sess.pleHostScratch = pleScratch
			}
			sess.perLayerInput = func(id int32, emb []byte) ([]byte, error) {
				pv := pleProjView
				scratch := pleScratch
				if pleResidentDisabled { // call-time host-path toggle (byte-identity test hook; always false in production)
					pv = bufView{}
					scratch = nil
				}
				return PerLayerInputs(g.EmbedPerLayer, g.EmbedPerLayerScales, g.EmbedPerLayerBiases, g.PerLayerModelProjW, g.PerLayerModelProjScales, g.PerLayerModelProjBiases, g.PerLayerProjNormW, id, emb, arch.PerLayerInputVocab, len(arch.Layer), arch.PerLayerInputHidden, arch.Hidden, gs, bits, g.PerLayerModelProjGS, g.PerLayerModelProjBits, arch.Eps, pv, scratch)
			}
			// GPU next-inputs seam: produce the next step's emb+pli on-GPU from a token-id buffer (no host
			// round-trip), the submit-ahead pipeline's gate. Handles e2b's shape only — 4-bit main + PLE
			// embedding, bf16 PLE projection; other shapes leave it nil and keep the host path.
			if affineBitsSupported(bits) && len(g.EmbedPerLayerScales) > 0 && len(g.PerLayerModelProjScales) == 0 {
				numLayers, pliDim, dModel := len(arch.Layer), arch.PerLayerInputHidden, arch.Hidden
				plDim := numLayers * pliDim
				embScalePLE := float32(math.Sqrt(float64(pliDim)))
				projScale := float32(1.0 / math.Sqrt(float64(dModel)))
				projWBuf, projWOff := pleProjView.buf, pleProjView.off
				sess.plScratchNew = func() *plGPUScratch { return newPLGPUScratch(plDim, projScale) }
				embedPackedBuf, embedScalesBuf, embedBiasesBuf := residentBytes(g.Embed), residentBytes(g.EmbedScales), residentBytes(g.EmbedBiases)
				plePackedBuf, pleScalesBuf, pleBiasesBuf := residentBytes(g.EmbedPerLayer), residentBytes(g.EmbedPerLayerScales), residentBytes(g.EmbedPerLayerBiases)
				pleNormBuf := residentBytes(g.PerLayerProjNormW)
				if projWBuf == nil {
					projWBuf = residentBytes(g.PerLayerModelProjW)
					projWOff = 0
				}
				sess.encNextInputsGPU = func(enc metal.MTLComputeCommandEncoderObject, tokenBuf, embOut metal.MTLBuffer, sc *plGPUScratch) error {
					gpso, gerr := embedGatherPipeline()
					if gerr != nil {
						return gerr
					}
					encEmbedGatherQuantObject(enc, gpso, tokenBuf, embedPackedBuf, embedScalesBuf, embedBiasesBuf, embOut, 0, 0, 0, dModel, gs, bits, embedScale)
					return encPerLayerInputsGPUObject(enc, gpso, tokenBuf, embOut, plePackedBuf, pleScalesBuf, pleBiasesBuf, 0, 0, 0, projWBuf, projWOff, pleNormBuf, sc, numLayers, pliDim, dModel, gs, bits, embScalePLE, arch.Eps)
				}
				// the K-token slab builder (quant table + bf16 steel projection): the prompt
				// prefill's PLE inputs on-GPU instead of the per-token host loop that idled
				// the GPU ~a third of every chunk.
				plainBatchScratch := &pleBatchScratch{}
				sess.perLayerInputBatch = func(ids []int32, embs [][]byte, slab []byte) (bool, error) {
					return perLayerInputsBatchQuantIntoSlab(plainBatchScratch,
						plePackedBuf, pleScalesBuf, pleBiasesBuf, gs, bits,
						projWBuf, projWOff, nil, nil, nil, 0, 0,
						g.PerLayerProjNormW, ids, embs, slab, numLayers, pliDim, dModel, arch.Eps)
				}
				plainMainEmb := &mainEmbedGather{packed: embedPackedBuf, scales: embedScalesBuf, biases: embedBiasesBuf, gs: gs, bits: bits, scale: embedScale}
				sess.perLayerInputBatchDevice = func(ids []int32, outLayers int) (metal.MTLBuffer, metal.MTLBuffer, bool, error) {
					return perLayerInputsBatchQuantDevice(plainBatchScratch,
						plePackedBuf, pleScalesBuf, pleBiasesBuf, gs, bits,
						projWBuf, projWOff, nil, nil, nil, 0, 0,
						g.PerLayerProjNormW, ids, plainMainEmb, outLayers, numLayers, pliDim, dModel, arch.Eps)
				}
			}
			// GPU next-inputs seam, QAT shape: same PLE tower but the per-layer model
			// projection ships QUANTISED (own gs/bits from the shapes) — the projection
			// matvec runs as the standard steel qmv instead of the bf16 gemv. This is
			// what lets qat e2b/e4b ride the chained/pipelined decode like the plain
			// 4-bit conversions do.
			if affineBitsSupported(bits) && len(g.EmbedPerLayerScales) > 0 && len(g.PerLayerModelProjScales) > 0 &&
				g.PerLayerModelProjGS > 0 && g.PerLayerModelProjBits > 0 {
				numLayers, pliDim, dModel := len(arch.Layer), arch.PerLayerInputHidden, arch.Hidden
				plDim := numLayers * pliDim
				embScalePLE := float32(math.Sqrt(float64(pliDim)))
				projScale := float32(1.0 / math.Sqrt(float64(dModel)))
				projGS, projBits := g.PerLayerModelProjGS, g.PerLayerModelProjBits
				sess.plScratchNew = func() *plGPUScratch { return newPLGPUScratch(plDim, projScale) }
				embedPackedBuf, embedScalesBuf, embedBiasesBuf := residentBytes(g.Embed), residentBytes(g.EmbedScales), residentBytes(g.EmbedBiases)
				plePackedBuf, pleScalesBuf, pleBiasesBuf := residentBytes(g.EmbedPerLayer), residentBytes(g.EmbedPerLayerScales), residentBytes(g.EmbedPerLayerBiases)
				pleNormBuf := residentBytes(g.PerLayerProjNormW)
				projPackedBuf, projScalesBuf, projBiasesBuf := residentBytes(g.PerLayerModelProjW), residentBytes(g.PerLayerModelProjScales), residentBytes(g.PerLayerModelProjBiases)
				sess.encNextInputsGPU = func(enc metal.MTLComputeCommandEncoderObject, tokenBuf, embOut metal.MTLBuffer, sc *plGPUScratch) error {
					gpso, gerr := embedGatherPipeline()
					if gerr != nil {
						return gerr
					}
					encEmbedGatherQuantObject(enc, gpso, tokenBuf, embedPackedBuf, embedScalesBuf, embedBiasesBuf, embOut, 0, 0, 0, dModel, gs, bits, embedScale)
					return encPerLayerInputsGPUQuantProjObject(enc, gpso, tokenBuf, embOut, plePackedBuf, pleScalesBuf, pleBiasesBuf, 0, 0, 0, projPackedBuf, projScalesBuf, projBiasesBuf, projGS, projBits, pleNormBuf, sc, numLayers, pliDim, dModel, gs, bits, embScalePLE, arch.Eps)
				}
				// the K-token slab builder (quant table + quant qmm_t projection — QAT shape)
				qatBatchScratch := &pleBatchScratch{}
				sess.perLayerInputBatch = func(ids []int32, embs [][]byte, slab []byte) (bool, error) {
					return perLayerInputsBatchQuantIntoSlab(qatBatchScratch,
						plePackedBuf, pleScalesBuf, pleBiasesBuf, gs, bits,
						nil, 0, projPackedBuf, projScalesBuf, projBiasesBuf, projGS, projBits,
						g.PerLayerProjNormW, ids, embs, slab, numLayers, pliDim, dModel, arch.Eps)
				}
				qatMainEmb := &mainEmbedGather{packed: embedPackedBuf, scales: embedScalesBuf, biases: embedBiasesBuf, gs: gs, bits: bits, scale: embedScale}
				sess.perLayerInputBatchDevice = func(ids []int32, outLayers int) (metal.MTLBuffer, metal.MTLBuffer, bool, error) {
					return perLayerInputsBatchQuantDevice(qatBatchScratch,
						plePackedBuf, pleScalesBuf, pleBiasesBuf, gs, bits,
						nil, 0, projPackedBuf, projScalesBuf, projBiasesBuf, projGS, projBits,
						g.PerLayerProjNormW, ids, qatMainEmb, outLayers, numLayers, pliDim, dModel, arch.Eps)
				}
			}
		} else if affineBitsSupported(bits) {
			// GPU next-inputs seam, non-PLE dense (12B/31B): the only per-step input
			// is the token's embedding, so the seam is the embed gather alone — no
			// PLE stage, and the plGPUScratch the chain hands through is a zero-value
			// placeholder the closure never reads (it exists to satisfy the chained/
			// pipelined gates, which key on encNextInputsGPU + plScratchNew). This is
			// what lets the submit-ahead decode engage beyond the E-family.
			dModel := arch.Hidden
			embedPackedBuf, embedScalesBuf, embedBiasesBuf := residentBytes(g.Embed), residentBytes(g.EmbedScales), residentBytes(g.EmbedBiases)
			if embedPackedBuf != nil && embedScalesBuf != nil && embedBiasesBuf != nil {
				sess.plScratchNew = func() *plGPUScratch { return &plGPUScratch{} }
				sess.encNextInputsGPU = func(enc metal.MTLComputeCommandEncoderObject, tokenBuf, embOut metal.MTLBuffer, _ *plGPUScratch) error {
					gpso, gerr := embedGatherPipeline()
					if gerr != nil {
						return gerr
					}
					encEmbedGatherQuantObject(enc, gpso, tokenBuf, embedPackedBuf, embedScalesBuf, embedBiasesBuf, embOut, 0, 0, 0, dModel, gs, bits, embedScale)
					return nil
				}
				// prefill chunk inputs, device-first (#381 dense port): the rows kernel
				// gathers a whole chunk's embed rows in one committed-not-waited command
				// buffer — no PLE tensor here, so the device closure hands back
				// (embBuf, nil) and the batched pass binds embBuf as its input rows.
				denseEmbScratch := &embedRowsScratch{}
				denseMainEmb := &mainEmbedGather{packed: embedPackedBuf, scales: embedScalesBuf, biases: embedBiasesBuf, gs: gs, bits: bits, scale: embedScale}
				sess.perLayerInputBatchDevice = func(ids []int32, _ int) (metal.MTLBuffer, metal.MTLBuffer, bool, error) {
					embBuf, ok, err := embedRowsBatchQuantDevice(denseEmbScratch, ids, denseMainEmb, dModel)
					return embBuf, nil, ok, err
				}
			}
		}
		// gemma4 incremental ICB encode-bypass (E2B/E4B dense): record the decode stack once + replay
		// it per Step/StepWithID instead of re-encoding every layer. The replay holds its OWN linear
		// maxLen caches (the session's lb sliding caches are RING-sized + unused on this path); the PLE
		// runtime wraps the session's own perLayerInput closure (the per-token tensor stays host-side).
		if sess.icbEligible() {
			var pleRuntime *archDecodePLEInputs
			if g.HasPLE() {
				pleRuntime = &archDecodePLEInputs{compute: sess.perLayerInput}
			}
			// per-owner caches (sliding = bounded ring; global = maxLen rows), with the
			// q8 opt-in allocating int8+scale caches on qualifying global owners (#367).
			kCaches, vCaches, icbKVQ8 := allocArchICBCaches(arch.Layer, arch.KVHeads, arch.HeadDim, maxLen, arch.SlidingWindow)
			rope := icbRope{
				base: arch.RopeBase, localBase: arch.RopeLocalBase,
				rotaryDim: arch.RotaryDim, rotaryDimLocal: arch.RotaryDimLocal,
				globalHeadDim: state.globalHeadDim,
				globalFreqs:   state.globalRopeFreqs, freqs: state.ropeFreqs,
			}
			rep, rerr := recordArchICBQuant(g.Layers, arch.Layer, kCaches, vCaches, pleRuntime, arch.PerLayerInputHidden, gs, bits, arch.Hidden, arch.Heads, arch.KVHeads, arch.HeadDim, maxLen, arch.FF, arch.SlidingWindow, rope, attnScale, arch.Eps, arch.ValueNorm, icbKVQ8)
			if rerr != nil {
				buildErr = rerr
				return
			}
			sess.state.icb = rep
			// Recorder for a PEER ICB sharing these KV caches (own ping0/pleInput) — the submit-ahead
			// decode keeps two in flight over the same KV. Lazily invoked; most sessions never pipeline.
			sess.recordPeerICB = func() (*archICBReplay, error) {
				return recordArchICBQuant(g.Layers, arch.Layer, kCaches, vCaches, pleRuntime, arch.PerLayerInputHidden, gs, bits, arch.Hidden, arch.Heads, arch.KVHeads, arch.HeadDim, maxLen, arch.FF, arch.SlidingWindow, rope, attnScale, arch.Eps, arch.ValueNorm, icbKVQ8)
			}
			if pipelinedGPUDecodeEnabled {
				peer, perr := sess.recordPeerICB()
				if perr != nil {
					buildErr = perr
					return
				}
				sess.icbPeer = peer
			}
		}
	})
	if buildErr != nil {
		return nil, buildErr
	}
	return sess, nil
}

// icbEligible reports whether this session can replay a recorded arch ICB instead of re-encoding
// per token. The ICB core (decodeForwardArchICBCore) assumes the SIMPLE uniform decode: no MoE
// (host router), no trace (per-layer host reads), uniform head geometry, and simple uniform rope
// (single base, no YaRN spectrum, no proportional-global). A model that varies any of those falls
// back to stepToken — byte-identical, just not encode-bypassed.
func (s *ArchSession) icbEligible() bool {
	if s.state.trace {
		return false
	}
	for li := range s.state.specs {
		sp := s.state.specs[li]
		// Per-layer head dim AND per-layer kvHeads are both recorded byte-identically: the forward-level
		// gate TestDecodeForwardArchICBQuantPerLayerKVHeads (DecodeForwardArchICBQuant ≡ DecodeForwardArchQuant
		// on a sliding-GQA/global-MQA mix) and the session-level TestArchQuantSessionICBParity_PerLayerKVHeads
		// (per-layer hidden cosine ≥ 0.9999) both pass. The old "14/24 divergence" came from a CONFOUNDED
		// session-level real-model test (PLE/head/chained paths differ from host re-encode even when the
		// recorder is byte-identical — it fails on uniform e2b too), not a recorder bug. So the 12B/31B
		// MQA-global mix now takes the fast ICB path. Only MoE (host router) and trace stay re-encode.
		if sp.MoE {
			return false
		}
	}
	return true
}

// Pos reports the number of tokens currently in the cache (the running sequence length).
func (s *ArchSession) Pos() int { return s.pos }

func (s *ArchSession) truncateSpeculativeKV(position int) error {
	if s == nil {
		return nil
	}
	if s.state.icb != nil && !icbDisabledForTest {
		return nil
	}
	return s.state.truncateDevicePagedKV(position)
}

// TruncateTo rolls the session boundary back so the next step overwrites any
// speculative cache rows beyond pos. The cache buffers do not carry a separate
// length; s.pos is the authoritative boundary used by every decode step.
func (s *ArchSession) TruncateTo(pos int) bool {
	if s == nil || pos < 0 || pos > s.pos {
		return false
	}
	if pos == s.pos {
		return true
	}
	s.pos = pos
	if len(s.cachedIDs) >= pos {
		s.cachedIDs = s.cachedIDs[:pos]
	} else {
		s.cachedIDs = nil
	}
	s.resetCachedPromptEntry()
	s.resetRetainedHidden()
	return true
}

var _ model.DecodeStepper = (*ArchSession)(nil)

// TokenTransform observes the selected token ID and returns the ID that should
// actually be committed into the resident decode cache. It is used for engine
// features such as thinking-budget close forcing, where changing only the
// streamed text would leave the cache conditioned on the wrong token.
type TokenTransform func(int32) int32

// PrefillTokens resets the retained decode state and prefills already-tokenised
// prompt ids into the resident KV cache. It is the token-native sibling of
// pkg/metal's ModelSession.PrefillTokens.
func (s *ArchSession) PrefillTokens(ids []int32) error {
	if len(ids) == 0 {
		return core.NewError("native.ArchSession.PrefillTokens: empty prompt tokens")
	}
	if len(ids) > s.maxLen {
		return core.NewError("native.ArchSession.PrefillTokens: sequence would exceed maxLen cache rows")
	}
	s.pos = 0
	s.resetCachedPromptEntry()
	s.resetRetainedHidden()
	resident := s.cachedIDs[:0]
	s.cachedIDs = resident
	hidden, err := s.prefillRetainedTokens(ids, "native.ArchSession.PrefillTokens")
	if err != nil {
		s.pos = 0
		s.cachedIDs = resident[:0]
		s.resetRetainedHidden()
		return err
	}
	s.cachedIDs = append(resident, ids...)
	s.rememberRetainedHidden(hidden)
	s.releaseQ8PrefillMirrors()
	return nil
}

// releaseQ8PrefillMirrors returns the q8 GEMM prefix's mirror memory at the
// prefill→decode seam and drops the cached state views that pointed into it
// (they rebuild — and re-materialise only the layers they need — on the next
// -state sleep or drafter export).
func (s *ArchSession) releaseQ8PrefillMirrors() {
	if s.state.icb == nil || !s.state.icb.releaseQ8Mirrors() {
		return
	}
	s.stateBlockViews = nil
}

// PrefillTokenEmbeddings resets the retained decode state and prefills already
// tokenised ids with caller-supplied embeddings. It is the multimodal sibling
// of PrefillTokens: image placeholder ids still drive PLE/cache metadata, while
// their embedding rows can be replaced by projected vision features.
func (s *ArchSession) PrefillTokenEmbeddings(ids []int32, embeddings [][]byte) error {
	if len(ids) == 0 {
		return core.NewError("native.ArchSession.PrefillTokenEmbeddings: empty prompt tokens")
	}
	if len(ids) != len(embeddings) {
		return core.NewError("native.ArchSession.PrefillTokenEmbeddings: token and embedding counts differ")
	}
	if len(ids) > s.maxLen {
		return core.NewError("native.ArchSession.PrefillTokenEmbeddings: sequence would exceed maxLen cache rows")
	}
	s.pos = 0
	s.resetCachedPromptEntry()
	s.resetRetainedHidden()
	resident := s.cachedIDs[:0]
	s.cachedIDs = resident
	hidden, err := s.prefillRetainedTokenEmbeddings(ids, embeddings, "native.ArchSession.PrefillTokenEmbeddings")
	if err != nil {
		s.pos = 0
		s.cachedIDs = resident[:0]
		s.resetRetainedHidden()
		return err
	}
	s.cachedIDs = append(resident, ids...)
	s.rememberRetainedHidden(hidden)
	s.releaseQ8PrefillMirrors()
	return nil
}

// AppendTokens appends already-tokenised prompt ids to the retained session
// state without replaying the existing prefix.
func (s *ArchSession) AppendTokens(ids []int32) error {
	if len(ids) == 0 {
		return core.NewError("native.ArchSession.AppendTokens: empty prompt tokens")
	}
	if s.pos == 0 {
		return core.NewError("native.ArchSession.AppendTokens: no retained prefill state")
	}
	if s.pos+len(ids) > s.maxLen {
		return core.NewError("native.ArchSession.AppendTokens: sequence would exceed maxLen cache rows")
	}
	s.resetRetainedLogits()
	hidden, err := s.prefillRetainedTokens(ids, "native.ArchSession.AppendTokens")
	if err != nil {
		s.cachedIDs = nil
		s.resetRetainedHidden()
		return err
	}
	s.cachedIDs = append(s.cachedIDs, ids...)
	s.clearCachedPromptHidden()
	s.rememberRetainedHidden(hidden)
	return nil
}

// GenerateFromCache greedily generates from the retained session boundary
// populated by PrefillTokens, AppendTokens, WarmPromptCache, Generate, or
// GenerateCached. No new prompt token is required.
func (s *ArchSession) GenerateFromCache(maxNew, eosID int) ([]int32, error) {
	return s.GenerateFromCacheEach(maxNew, eosID, nil)
}

// GenerateFromCacheEach is GenerateFromCache with per-token streaming.
func (s *ArchSession) GenerateFromCacheEach(maxNew, eosID int, yield func(int32) bool) ([]int32, error) {
	return s.GenerateFromCacheEachTransformed(maxNew, eosID, nil, yield)
}

// GenerateSampledFromCacheEach samples from the retained session boundary
// without replaying prompt tokens or requiring captured boundary logits.
func (s *ArchSession) GenerateSampledFromCacheEach(maxNew int, stopTokens []int32, sampler *model.Sampler, params model.SampleParams, transform model.TokenTransform, yield func(int32) bool) ([]int32, error) {
	if sampler == nil {
		return nil, core.NewError("native.ArchSession.GenerateSampledFromCache: nil sampler")
	}
	if maxNew <= 0 {
		return nil, core.NewError("native.ArchSession.GenerateSampledFromCache: maxNew must be > 0")
	}
	if len(s.retainedLogits) == s.arch.Vocab*bf16Size {
		return s.GenerateSampledFromCacheLogitsEach(s.retainedLogits, maxNew, stopTokens, sampler, params, transform, yield)
	}
	if len(s.retainedHidden) != s.arch.Hidden*bf16Size {
		return nil, core.NewError("native.ArchSession.GenerateSampledFromCache: no retained prefill state")
	}
	if s.pos+maxNew > s.maxLen {
		return nil, core.NewError("native.ArchSession.GenerateSampledFromCache: sequence would exceed maxLen cache rows")
	}
	hidden := s.retainedHidden
	var gen []int32
	var err error
	withAutoreleasePool(func() {
		gen, err = s.generateSampledFromHiddenInPool(hidden, maxNew, stopTokens, sampler, params, transform, yield, true)
	})
	if err != nil {
		s.cachedIDs = nil
		s.resetRetainedHidden()
		return nil, err
	}
	s.cachedIDs = append(s.cachedIDs, gen...)
	return gen, nil
}

// BoundaryNormedHidden returns the post-final-RMSNorm hidden vector at the
// retained session boundary. Gemma 4 assistant drafting seeds from this target
// feature, matching the vector the target LM head consumes.
func (s *ArchSession) BoundaryNormedHidden() ([]byte, error) {
	return s.boundaryNormedHiddenInto(nil)
}

func (s *ArchSession) boundaryNormedHiddenScratch() ([]byte, error) {
	n := s.arch.Hidden * bf16Size
	if cap(s.mtpBoundaryNormed) < n {
		s.mtpBoundaryNormed = make([]byte, n)
	}
	return s.boundaryNormedHiddenInto(s.mtpBoundaryNormed[:n])
}

func (s *ArchSession) mtpProjectionScratch(byteLen int) []byte {
	if cap(s.mtpProjected) < byteLen {
		s.mtpProjected = make([]byte, byteLen)
	}
	return s.mtpProjected[:byteLen]
}

func (s *ArchSession) mtpDraftScratch(slot *[]byte, byteLen int) []byte {
	if cap(*slot) < byteLen {
		*slot = make([]byte, byteLen)
	}
	return (*slot)[:byteLen]
}

func (s *ArchSession) mtpDraftTokenScratch(n int) []int32 {
	if cap(s.mtpDraftTokens) < n {
		s.mtpDraftTokens = make([]int32, 0, n)
	} else {
		s.mtpDraftTokens = s.mtpDraftTokens[:0]
	}
	return s.mtpDraftTokens
}

func (s *ArchSession) mtpDraftVerifyBlockScratch(carry int32, draft []int32) []int32 {
	n := len(draft) + 1
	if cap(s.mtpDraftVerifyBlock) < n {
		s.mtpDraftVerifyBlock = make([]int32, n)
	} else {
		s.mtpDraftVerifyBlock = s.mtpDraftVerifyBlock[:n]
	}
	s.mtpDraftVerifyBlock[0] = carry
	copy(s.mtpDraftVerifyBlock[1:], draft)
	return s.mtpDraftVerifyBlock
}

func (s *ArchSession) mtpDraftLogitScoreScratch(n int) []float32 {
	if cap(s.mtpDraftLogitScores) < n {
		s.mtpDraftLogitScores = make([]float32, n)
	} else {
		s.mtpDraftLogitScores = s.mtpDraftLogitScores[:n]
	}
	return s.mtpDraftLogitScores
}

func (s *ArchSession) mtpDraftLogitSelectedScratch(n int) []int {
	if cap(s.mtpDraftLogitSelected) < n {
		s.mtpDraftLogitSelected = make([]int, 0, n)
	} else {
		s.mtpDraftLogitSelected = s.mtpDraftLogitSelected[:0]
	}
	return s.mtpDraftLogitSelected
}

func (s *ArchSession) mtpTargetKVScratchEntries(n int) []AssistantTargetKV {
	if cap(s.mtpTargetKVScratch) < n {
		s.mtpTargetKVScratch = make([]AssistantTargetKV, n)
	} else {
		s.mtpTargetKVScratch = s.mtpTargetKVScratch[:n]
		for i := range s.mtpTargetKVScratch {
			s.mtpTargetKVScratch[i] = AssistantTargetKV{}
		}
	}
	return s.mtpTargetKVScratch
}

func (s *ArchSession) mtpTargetKVByTypeEntries(capacity int) []AssistantKVEntry {
	if cap(s.mtpTargetKVByType) < capacity {
		s.mtpTargetKVByType = make([]AssistantKVEntry, 0, capacity)
	} else {
		s.mtpTargetKVByType = s.mtpTargetKVByType[:cap(s.mtpTargetKVByType)]
		for i := range s.mtpTargetKVByType {
			s.mtpTargetKVByType[i] = AssistantKVEntry{}
		}
		s.mtpTargetKVByType = s.mtpTargetKVByType[:0]
	}
	return s.mtpTargetKVByType
}

func (s *ArchSession) mtpTargetKVSlabs(cacheIndex, keyBytes, valueBytes int) ([]byte, []byte) {
	for len(s.mtpTargetKVKeySlabs) <= cacheIndex {
		s.mtpTargetKVKeySlabs = append(s.mtpTargetKVKeySlabs, nil)
		s.mtpTargetKVValueSlabs = append(s.mtpTargetKVValueSlabs, nil)
	}
	key := s.mtpTargetKVKeySlabs[cacheIndex]
	if cap(key) < keyBytes {
		key = make([]byte, keyBytes)
	}
	key = key[:keyBytes]
	s.mtpTargetKVKeySlabs[cacheIndex] = key

	value := s.mtpTargetKVValueSlabs[cacheIndex]
	if cap(value) < valueBytes {
		value = make([]byte, valueBytes)
	}
	value = value[:valueBytes]
	s.mtpTargetKVValueSlabs[cacheIndex] = value
	return key, value
}

func (s *ArchSession) boundaryNormedHiddenInto(out []byte) ([]byte, error) {
	if s == nil {
		return nil, core.NewError("native.ArchSession.BoundaryNormedHidden: nil session")
	}
	if len(s.retainedHidden) != s.arch.Hidden*bf16Size {
		return nil, core.NewError("native.ArchSession.BoundaryNormedHidden: no retained prefill state")
	}
	// retainedHidden is the decode step's head-input vector: on the arch paths a
	// unit-RMS hidden WITHOUT the final-norm gain (the head consumes it gain-folded;
	// greedy argmax is scale-invariant, so logits stay correct either way). The MTP
	// drafter, however, was trained on the reference boundary hidden — HF
	// hidden_states[-1] = x̂ ⊙ (1+norm_w) — so export the FULL final RMSNorm: the
	// rms() divides out whatever scalar the step retained (exact regardless of that
	// path's internal scaling) and the gain restores the trained per-dim magnitudes.
	// Without the gain the drafter's hidden half sat ~37× low against the trained
	// pre_projection weights and every draft went target-blind (live MTP acceptance
	// ~5%, HF-parity showed sum|ours| 214 vs reference 7958 with per-dim ratio
	// tracking (1+norm_w)).
	n := s.arch.Hidden * bf16Size
	if len(s.finalNorm) == n {
		return RMSNormBF16Into(out, s.retainedHidden, s.finalNorm, 1, s.arch.Hidden, s.arch.Eps)
	}
	if cap(out) < n {
		out = make([]byte, n)
	}
	out = out[:n]
	copy(out, s.retainedHidden)
	return out, nil
}

// BoundaryLogits returns the bf16 logits at the retained session boundary.
// Restore paths can use these logits to select the first continuation token
// without recomputing the restored prompt prefix.
func (s *ArchSession) BoundaryLogits() ([]byte, error) {
	if len(s.retainedLogits) == s.arch.Vocab*bf16Size {
		return s.retainedLogits, nil
	}
	if len(s.retainedHidden) != s.arch.Hidden*bf16Size {
		return nil, core.NewError("native.ArchSession.BoundaryLogits: no retained prefill state")
	}
	var logits []byte
	var err error
	if hiddenBuf := s.retainedHiddenBufferFor(s.retainedHidden); hiddenBuf != nil && s.headEnc != nil {
		if pinned, ok := s.ensureRetainedLogitsPinned(s.arch.Vocab * bf16Size); ok {
			logits, err = s.headEnc.encodeBufferInto(hiddenBuf, false, pinned.bytes)
			if err != nil {
				return nil, err
			}
			s.retainedLogits = logits
			s.sampleHeadLogits = nil
			return s.retainedLogits, nil
		}
		logits, err = s.headEnc.encodeBufferInto(hiddenBuf, false, s.sampleHeadLogits)
		if err == nil {
			s.sampleHeadLogits = logits
		}
	} else {
		logits, err = s.head(s.retainedHidden, false)
	}
	if err != nil {
		return nil, err
	}
	s.rememberRetainedLogits(logits)
	return s.retainedLogits, nil
}

// GenerateFromCacheLogitsEach greedily continues a restored cache from already
// captured boundary logits. The first token is selected directly from
// firstLogits; subsequent tokens use the resident K/V cache and normal native
// step path, so the prompt prefix is not replayed.
func (s *ArchSession) GenerateFromCacheLogitsEach(firstLogits []byte, maxNew, eosID int, yield func(int32) bool) ([]int32, error) {
	return s.generateFromCacheLogitsEach(firstLogits, maxNew, eosID, nil, nil, yield)
}

func (s *ArchSession) generateFromCacheLogitsEach(firstLogits []byte, maxNew, eosID int, suppress []int32, transform TokenTransform, yield func(int32) bool) ([]int32, error) {
	if maxNew <= 0 {
		return nil, core.NewError("native.ArchSession.GenerateFromCacheLogits: maxNew must be > 0")
	}
	if len(firstLogits) != s.arch.Vocab*bf16Size {
		return nil, core.NewError("native.ArchSession.GenerateFromCacheLogits: logits must be vocab bf16 bytes")
	}
	if s.pos+maxNew > s.maxLen {
		return nil, core.NewError("native.ArchSession.GenerateFromCacheLogits: sequence would exceed maxLen cache rows")
	}
	var gen []int32
	var err error
	withAutoreleasePool(func() {
		gen, err = s.generateFromLogitsInPool(firstLogits, maxNew, eosID, suppress, transform, yield)
	})
	if err != nil {
		s.cachedIDs = nil
		s.resetRetainedHidden()
		return nil, err
	}
	s.cachedIDs = append(s.cachedIDs, gen...)
	return gen, nil
}

// GenerateSampledFromCacheLogitsEach samples a restored-cache continuation from
// already captured boundary logits. The first token is sampled from firstLogits;
// subsequent tokens reuse the resident K/V cache and sampled native step loop.
func (s *ArchSession) GenerateSampledFromCacheLogitsEach(firstLogits []byte, maxNew int, stopTokens []int32, sampler *model.Sampler, params model.SampleParams, transform model.TokenTransform, yield func(int32) bool) ([]int32, error) {
	if sampler == nil {
		return nil, core.NewError("native.ArchSession.GenerateSampledFromCacheLogits: nil sampler")
	}
	if maxNew <= 0 {
		return nil, core.NewError("native.ArchSession.GenerateSampledFromCacheLogits: maxNew must be > 0")
	}
	if len(firstLogits) != s.arch.Vocab*bf16Size {
		return nil, core.NewError("native.ArchSession.GenerateSampledFromCacheLogits: logits must be vocab bf16 bytes")
	}
	if s.pos+maxNew > s.maxLen {
		return nil, core.NewError("native.ArchSession.GenerateSampledFromCacheLogits: sequence would exceed maxLen cache rows")
	}
	var gen []int32
	var err error
	withAutoreleasePool(func() {
		gen, err = s.generateSampledFromLogitsInPool(firstLogits, maxNew, stopTokens, sampler, params, transform, yield, true)
	})
	if err != nil {
		s.cachedIDs = nil
		s.resetRetainedHidden()
		return nil, err
	}
	s.cachedIDs = append(s.cachedIDs, gen...)
	return gen, nil
}

// GenerateFromCacheEachTransformed is GenerateFromCacheEach with a committed-token
// transform applied before each generated token is written to the cache.
func (s *ArchSession) GenerateFromCacheEachTransformed(maxNew, eosID int, transform TokenTransform, yield func(int32) bool) ([]int32, error) {
	return s.GenerateFromCacheEachWithSuppressionAndTransform(maxNew, eosID, nil, transform, yield)
}

// GenerateFromCacheEachWithSuppression is GenerateFromCacheEach with suppressed
// token ids masked before greedy argmax.
func (s *ArchSession) GenerateFromCacheEachWithSuppression(maxNew, eosID int, suppress []int32, yield func(int32) bool) ([]int32, error) {
	return s.GenerateFromCacheEachWithSuppressionAndTransform(maxNew, eosID, suppress, nil, yield)
}

// GenerateFromCacheEachWithSuppressionAndTransform combines restored-cache
// greedy token suppression with a committed-token transform.
func (s *ArchSession) GenerateFromCacheEachWithSuppressionAndTransform(maxNew, eosID int, suppress []int32, transform TokenTransform, yield func(int32) bool) ([]int32, error) {
	if maxNew <= 0 {
		return nil, core.NewError("native.ArchSession.GenerateFromCache: maxNew must be > 0")
	}
	if len(s.retainedLogits) == s.arch.Vocab*bf16Size {
		return s.generateFromCacheLogitsEach(s.retainedLogits, maxNew, eosID, suppress, transform, yield)
	}
	if len(s.retainedHidden) != s.arch.Hidden*bf16Size {
		return nil, core.NewError("native.ArchSession.GenerateFromCache: no retained prefill state")
	}
	if s.pos+maxNew > s.maxLen {
		return nil, core.NewError("native.ArchSession.GenerateFromCache: sequence would exceed maxLen cache rows")
	}
	hidden := s.retainedHidden
	var gen []int32
	var err error
	withAutoreleasePool(func() {
		gen, err = s.generateFromHiddenInPool(hidden, maxNew, eosID, nil, nil, suppress, transform, yield)
	})
	if err != nil {
		s.cachedIDs = nil
		s.resetRetainedHidden()
		return nil, err
	}
	s.cachedIDs = append(s.cachedIDs, gen...)
	return gen, nil
}

func (s *ArchSession) prefillRetainedTokens(ids []int32, scope string) ([]byte, error) {
	if len(ids) == 0 {
		return nil, nil
	}
	if s.pos+len(ids) > s.maxLen {
		return nil, core.NewError(scope + ": sequence would exceed maxLen cache rows")
	}
	// Persisted block restores can resume from K/V plus boundary logits only.
	// In that shape, the token step path matches decode parity while batched
	// prompt append needs a live retained hidden boundary.
	if len(s.retainedHidden) != s.arch.Hidden*bf16Size {
		return s.prefillPromptRetainedInPool(ids)
	}
	if !s.restoredKV {
		if hidden, ok, err := s.prefillRetainedTokensBatchedDense(ids, scope); ok || err != nil {
			return hidden, err
		}
	}
	if hidden, ok, err := s.prefillPromptRetainedGPUInputsInPool(ids); ok || err != nil {
		return hidden, err
	}
	if len(ids) > 1 {
		if err := s.prefillCachedIDs(ids[:len(ids)-1]); err != nil {
			return nil, err
		}
	}
	var hidden []byte
	var err error
	withAutoreleasePool(func() {
		hidden, err = s.stepIDInPool(ids[len(ids)-1])
	})
	return hidden, err
}

func (s *ArchSession) prefillRetainedTokenEmbeddings(ids []int32, embeddings [][]byte, scope string) ([]byte, error) {
	if len(ids) == 0 {
		return nil, nil
	}
	if len(ids) != len(embeddings) {
		return nil, core.NewError(scope + ": token and embedding counts differ")
	}
	if s.pos+len(ids) > s.maxLen {
		return nil, core.NewError(scope + ": sequence would exceed maxLen cache rows")
	}
	// Multimodal embeddings/bidir prefill keeps the FULL stack on every chunk:
	// pin prefillSkipToLayer to 0 so this lane never inherits the causal kv-shared
	// suffix skip (#381). Its consumer (engine/vision.go PrefillTokenEmbeddings ->
	// decodeFromPrefilled) reads only the FINAL chunk's boundary hidden for
	// autoregressive generation, so the skip is STRUCTURALLY valid here too — but
	// it stays UNARMED: arming it on the correctness-critical bidirectional
	// image/video span path is a perf change owing the same real-model
	// byte-identity receipt the causal lane earned (pp8K/pp62K), and the skip is
	// byte-identical by construction so no fixture A/B can stand in for that. The
	// pin is also load-bearing: unlike the causal lane this lane has no per-chunk
	// reset, so a leaked non-zero value would bound the FINAL (read) chunk and
	// corrupt the boundary hidden. Both sub-lanes (bidir spans + batched-dense
	// embeddings) enter here. Guarded by
	// TestArchSessionPrefillTokenEmbeddingsKeepsFullStackIgnoresSkip.
	s.state.prefillSkipToLayer = 0
	if spans := bidirTokenSpans(ids, s.bidirSpanTokens); len(spans) > 0 {
		if unifiedVisionDiag {
			nativeTraceLog(core.Sprintf("vision-diag bidir: %d spans over %d ids (first %v)\n", len(spans), len(ids), spans[0]))
		}
		// Bidirectional spans NEVER fall back to a sequential lane: stepping
		// writes each token's K/V after its own attention, so a span row can
		// never see the rows after it — a silent causal evaluation misreads
		// the image. Batched-or-error.
		return s.prefillRetainedEmbeddingsBidir(ids, embeddings, spans, scope)
	}
	if hidden, ok, err := s.prefillRetainedEmbeddingsBatchedDense(ids, embeddings, scope); ok || err != nil {
		return hidden, err
	}
	var hidden []byte
	var err error
	for i, id := range ids {
		hidden, err = s.StepWithID(id, embeddings[i])
		if err != nil {
			return nil, err
		}
	}
	return hidden, nil
}

// bidirTokenSpans returns the [start,end) index runs of the span tokens in
// ids — the bidirectional image/video spans. Runs are per-token (an image
// block and an adjacent video block stay separate spans); zero ids find
// nothing.
func bidirTokenSpans(ids []int32, toks [2]int32) [][2]int {
	if toks[0] == 0 && toks[1] == 0 {
		return nil
	}
	isSpan := func(id int32) bool { return id != 0 && (id == toks[0] || id == toks[1]) }
	var spans [][2]int
	start := -1
	var startTok int32
	for i, id := range ids {
		switch {
		case isSpan(id) && start < 0:
			start, startTok = i, id
		case start >= 0 && id != startTok:
			spans = append(spans, [2]int{start, i})
			if isSpan(id) {
				start, startTok = i, id
			} else {
				start = -1
			}
		}
	}
	if start >= 0 {
		spans = append(spans, [2]int{start, len(ids)})
	}
	return spans
}

// prefillRetainedEmbeddingsBidir prefills embedding rows whose span rows
// attend bidirectionally: chunks are cut so no span straddles a chunk, and
// each chunk runs the batched dense pass with per-row attention caps (a span
// row sees through to its span end — legal because the fold lands the WHOLE
// chunk's K/V before any SDPA reads). There is deliberately no sequential
// fallback.
func (s *ArchSession) prefillRetainedEmbeddingsBidir(ids []int32, embeddings [][]byte, spans [][2]int, scope string) ([]byte, error) {
	if s.arch.SlidingWindow > 0 && s.arch.SlidingWindow < s.maxLen && s.pos+len(ids) > s.arch.SlidingWindow {
		return nil, core.NewError(scope + ": bidirectional image spans require the prompt to fit the sliding window (staged ring landings are per-row)")
	}
	var hidden []byte
	base := 0
	for base < len(ids) {
		n := s.batchedDensePrefillChunkLen(len(ids) - base)
		if n <= 0 {
			return nil, core.NewError(scope + ": invalid bidirectional prefill chunk")
		}
		// never cut inside a span: shrink the chunk to the span start, or grow
		// it to the span end when the span starts the chunk. Growing may reach
		// a later span, so iterate to a stable cut.
		for adjusted := true; adjusted; {
			adjusted = false
			for _, sp := range spans {
				st, en := sp[0]-base, sp[1]-base
				if st < n && en > n {
					if st > 0 {
						n = st
					} else {
						n = en
					}
					adjusted = true
				}
			}
		}
		if base+n > len(ids) {
			n = len(ids) - base
		}
		caps := make([]int32, n)
		for i := range n {
			caps[i] = int32(s.pos + i + 1) // causal default
		}
		for _, sp := range spans {
			for i := max(sp[0], base); i < min(sp[1], base+n); i++ {
				caps[i-base] = int32(s.pos + (min(sp[1], base+n) - base)) // see through to span end
			}
		}
		next, err := s.prefillRetainedEmbeddingsBidirChunk(embeddings[base:base+n], caps, scope)
		if err != nil {
			return nil, err
		}
		hidden = next
		base += n
	}
	return hidden, nil
}

func (s *ArchSession) prefillRetainedEmbeddingsBidirChunk(embeddings [][]byte, caps []int32, scope string) ([]byte, error) {
	if s.verifyBatchedCrossesSlidingRingWrap(len(embeddings)) {
		return nil, core.NewError(scope + ": bidirectional span chunk crosses the sliding ring wrap")
	}
	rowBytes := s.arch.Hidden * bf16Size
	for i := range embeddings {
		if len(embeddings[i]) != rowBytes {
			return nil, core.NewError(scope + ": emb must be hidden bf16 bytes")
		}
	}
	dst := s.sampleHidden
	retained := false
	if pinned, pinnedOK := s.ensureRetainedHiddenPinned(rowBytes); pinnedOK {
		s.resetRetainedLogits()
		dst = pinned.bytes[:rowBytes]
		retained = true
	}
	var (
		hidden []byte
		ok     bool
		err    error
	)
	s.state.rowAttnCaps = caps
	withAutoreleasePool(func() {
		hidden, ok, err = s.state.stepTokensBatchedDenseLastIntoCopyInputs(embeddings, s.pos, dst)
	})
	s.state.rowAttnCaps = nil
	if err != nil {
		return nil, err
	}
	if !ok {
		return nil, core.NewError(scope + ": the batched dense pass declined a bidirectional span chunk (no causal fallback exists for image spans)")
	}
	if retained {
		s.sampleHidden = nil
		s.retainedHidden = hidden
	} else {
		s.sampleHidden = hidden
	}
	s.pos += len(embeddings)
	return hidden, nil
}

func (s *ArchSession) prefillPromptRetainedInPool(ids []int32) ([]byte, error) {
	if len(ids) == 0 {
		return nil, nil
	}
	// A FRESH session (pos 0 — the first prompt of every generate/serve request) and a
	// LIVE session appending a turn (pos > 0 with a retained boundary — the prompt
	// cache's suffix path, every multi-turn serve request) both batch byte-identically
	// to stepping (proven by the batched parity + append tests). Only a restored
	// session without a live retained-hidden boundary stays on the token path — the
	// decode-parity carve-out the prefillRetainedTokens guard exists for.
	if s.pos == 0 || (len(s.retainedHidden) == s.arch.Hidden*bf16Size && !s.restoredKV) {
		if hidden, ok, err := s.prefillRetainedTokensBatchedDense(ids, "native.prefillPromptRetained"); ok || err != nil {
			return hidden, err
		}
	}
	if hidden, ok, err := s.prefillPromptRetainedGPUInputsInPool(ids); ok || err != nil {
		return hidden, err
	}
	var err error
	for _, id := range ids[:len(ids)-1] {
		if _, err = s.stepIDInPool(id); err != nil {
			return nil, err
		}
	}
	return s.stepIDRetainedInPool(ids[len(ids)-1])
}

func (s *ArchSession) prefillRetainedEmbeddingsBatchedDense(ids []int32, embeddings [][]byte, scope string) ([]byte, bool, error) {
	if len(ids) == 0 {
		return nil, false, nil
	}
	if len(ids) != len(embeddings) {
		return nil, false, core.NewError(scope + ": token and embedding counts differ")
	}
	if s.pos+len(ids) > s.maxLen {
		return nil, false, core.NewError(scope + ": sequence would exceed maxLen cache rows")
	}
	if s.verifyBatchedCrossesSlidingRingWrap(len(ids)) {
		return s.prefillRetainedEmbeddingsBatchedDenseChunks(ids, embeddings, scope)
	}
	return s.prefillRetainedEmbeddingsBatchedDenseOne(ids, embeddings, scope)
}

func (s *ArchSession) prefillRetainedEmbeddingsBatchedDenseChunks(ids []int32, embeddings [][]byte, scope string) ([]byte, bool, error) {
	var hidden []byte
	for len(ids) > 0 {
		n := s.batchedDensePrefillChunkLen(len(ids))
		if n <= 0 {
			return nil, false, core.NewError("native.prefillRetainedEmbeddingsBatchedDense: invalid sliding chunk")
		}
		nextHidden, ok, err := s.prefillRetainedEmbeddingsBatchedDenseOne(ids[:n], embeddings[:n], scope)
		if err != nil || !ok {
			return nil, ok, err
		}
		hidden = nextHidden
		ids = ids[n:]
		embeddings = embeddings[n:]
	}
	return hidden, true, nil
}

func (s *ArchSession) prefillRetainedEmbeddingsBatchedDenseOne(ids []int32, embeddings [][]byte, scope string) ([]byte, bool, error) {
	if len(embeddings) == 0 {
		return nil, false, nil
	}
	if len(ids) != len(embeddings) {
		return nil, false, core.NewError(scope + ": token and embedding counts differ")
	}
	if s.pos+len(embeddings) > s.maxLen {
		return nil, false, core.NewError(scope + ": sequence would exceed maxLen cache rows")
	}
	if s.verifyBatchedCrossesSlidingRingWrap(len(embeddings)) {
		return nil, false, nil
	}
	if hidden, ok, err := s.prefillRetainedEmbeddingsICB(ids, embeddings, scope); ok || err != nil {
		return hidden, ok, err
	}
	if s.perLayerInput != nil || s.state.icb != nil {
		return nil, false, nil
	}
	var (
		hidden []byte
		ok     bool
		err    error
	)
	dst := s.sampleHidden
	retained := false
	if pinned, pinnedOK := s.ensureRetainedHiddenPinned(s.arch.Hidden * bf16Size); pinnedOK {
		s.resetRetainedLogits()
		dst = pinned.bytes[:s.arch.Hidden*bf16Size]
		retained = true
	}
	withAutoreleasePool(func() {
		hidden, ok, err = s.state.stepTokensBatchedDenseLastIntoCopyInputs(embeddings, s.pos, dst)
	})
	if err != nil || !ok {
		return nil, ok, err
	}
	if retained {
		s.sampleHidden = nil
		s.retainedHidden = hidden
	} else {
		s.sampleHidden = hidden
	}
	s.pos += len(embeddings)
	return hidden, true, nil
}

func (s *ArchSession) prefillRetainedEmbeddingsICB(ids []int32, embeddings [][]byte, scope string) ([]byte, bool, error) {
	if len(embeddings) == 0 {
		return nil, false, nil
	}
	if len(ids) != len(embeddings) {
		return nil, false, core.NewError(scope + ": token and embedding counts differ")
	}
	icb := s.state.icb
	if icb == nil || icbDisabledForTest || s.pos != 0 {
		return nil, false, nil
	}
	if icb.hasPLE {
		if icb.pleRuntime == nil || icb.pleRuntime.compute == nil {
			return nil, true, core.NewError(scope + ": ICB PLE runtime is unavailable")
		}
		prevTokenIDs := icb.pleRuntime.tokenIDs
		icb.pleRuntime.tokenIDs = ids
		defer func() {
			icb.pleRuntime.tokenIDs = prevTokenIDs
		}()
	} else if s.perLayerInput != nil {
		return nil, false, nil
	}
	rowBytes := s.arch.Hidden * bf16Size
	for i := range embeddings {
		if len(embeddings[i]) != rowBytes {
			return nil, false, core.NewError(scope + ": emb must be hidden bf16 bytes")
		}
	}
	var dst []byte
	if pinned, pinnedOK := s.ensureRetainedHiddenPinned(rowBytes); pinnedOK {
		s.resetRetainedLogits()
		dst = pinned.bytes[:rowBytes]
	}
	if dst == nil {
		if cap(s.sampleHidden) < rowBytes {
			s.sampleHidden = make([]byte, rowBytes)
		}
		dst = s.sampleHidden[:rowBytes]
	}
	hidden, err := icb.runBatchLastInto(dst, embeddings)
	if err != nil {
		return nil, true, err
	}
	if len(hidden) != rowBytes {
		return nil, true, core.NewError(scope + ": ICB hidden result width mismatch")
	}
	if s.retainedHiddenPinned != nil && len(s.retainedHiddenPinned.bytes) == len(hidden) && len(hidden) != 0 &&
		unsafe.Pointer(&hidden[0]) == unsafe.Pointer(&s.retainedHiddenPinned.bytes[0]) {
		s.sampleHidden = nil
		s.retainedHidden = hidden
	} else {
		s.sampleHidden = hidden
	}
	s.pos += len(embeddings)
	return hidden, true, nil
}

func (s *ArchSession) prefillPromptRetainedGPUInputsInPool(ids []int32) ([]byte, bool, error) {
	if s.state.icb == nil || icbDisabledForTest || s.encNextInputsGPU == nil || s.plScratchNew == nil || chainedGPUInputsDisabled {
		return nil, false, nil
	}
	if len(ids) > 1 {
		if err := s.prefillCachedIDsGPUInputs(ids[:len(ids)-1]); err != nil {
			return nil, true, err
		}
	}
	return s.stepIDRetainedGPUInputsInPool(ids[len(ids)-1])
}

func (s *ArchSession) prefillRetainedTokensBatchedDense(ids []int32, scope string) ([]byte, bool, error) {
	if len(ids) == 0 {
		return nil, false, nil
	}
	if s.pos+len(ids) > s.maxLen {
		return nil, false, core.NewError(scope + ": sequence would exceed maxLen cache rows")
	}
	if s.verifyBatchedCrossesSlidingRingWrap(len(ids)) {
		return s.prefillRetainedTokensBatchedDenseChunks(ids, scope)
	}
	return s.prefillRetainedTokensBatchedDenseOne(ids, scope)
}

func (s *ArchSession) prefillRetainedTokensBatchedDenseChunks(ids []int32, scope string) ([]byte, bool, error) {
	var hidden []byte
	chunk := 0
	// Non-final chunks bound the layer loop at the shared suffix (#381): the
	// skipped layers own no cache rows and only the FINAL chunk's boundary
	// hidden is ever read, so the pruned compute feeds nothing downstream.
	skipTo := 0
	if prefillSkipSharedEnabled && !prefillSkipSharedOffForTest && s.state.sharedSuffix > 0 {
		skipTo = s.state.sharedSuffix
	}
	defer func() { s.state.prefillSkipToLayer = 0 }()
	for len(ids) > 0 {
		var n int
		if skipTo > 0 {
			n = s.batchedDensePrefillChunkLenSkip(len(ids))
		} else {
			n = s.batchedDensePrefillChunkLen(len(ids))
		}
		if n <= 0 {
			return nil, false, core.NewError("native.prefillRetainedTokensBatchedDense: invalid sliding chunk")
		}
		s.state.prefillSkipToLayer = 0
		if n < len(ids) {
			s.state.prefillSkipToLayer = skipTo
		}
		chunkStart := time.Now()
		nextHidden, ok, err := s.prefillRetainedTokensBatchedDenseOne(ids[:n], scope)
		if err != nil || !ok {
			return nil, ok, err
		}
		hostSpan("chunk", chunkStart, n)
		if argmaxDebugEnabled() {
			if nan, first := bf16NaNScanBytes(nextHidden); nan > 0 {
				nativeTraceLog(core.Sprintf("argmax-diag: batched prefill chunk %d (rows %d, pos now %d): boundary hidden NaN=%d first=%d\n",
					chunk, n, s.pos, nan, first))
				if views, verr := s.stateLayerViews(); verr == nil {
					for _, v := range views {
						kc, kf := bf16NaNScanBytes(v.keyBytes)
						vc, vf := bf16NaNScanBytes(v.valueBytes)
						if kc > 0 || vc > 0 {
							sp := s.state.specs[v.layer]
							at := "sliding"
							if sp.Attention == model.GlobalAttention {
								at = "GLOBAL "
							}
							rowB := kvHeadsOf(sp, s.arch.KVHeads) * headDimOf(sp, s.arch.HeadDim)
							nativeTraceLog(core.Sprintf("argmax-diag:   L%2d %s owns=%v shareFrom=%d  K-NaN=%d(first row %d)  V-NaN=%d(first row %d)\n",
								v.layer, at, sp.OwnsCache(), sp.KVShareFrom, kc, kf/rowB, vc, vf/rowB))
						}
					}
				}
			}
		}
		chunk++
		hidden = nextHidden
		ids = ids[n:]
	}
	return hidden, true, nil
}

// batchedDensePrefillTargetRows is the prefill chunk's target row count once the
// position is window-aligned; the per-model window divides it into whole windows.
// Wider chunks raise every per-chunk GEMM's M (the projections, the qmm fold, the
// prompt SDPA) and amortise the per-chunk seams — 2048 is the receipted sweet spot
// (#367 2026-07-13, e2b depth ladder: 512-row chunks 2907/2594/2122 tok/s at
// 8K/32K/62K vs 2048-row 3074/2669/2179; 4096 gave the gain back) and matches
// mlx-lm's prefill_step_size. The deferred-ring lane handles wrap-crossing batches
// at any basePos, so the width is an engine tuning choice, not a model contract.
// LTHN_PREFILL_WINDOWS overrides with an explicit window count (the A/B lever).
const batchedDensePrefillTargetRows = 2048

func prefillChunkWindows() int {
	return prefillChunkWindowsFor(0)
}

// prefillChunkWindowsFor resolves the chunk width in windows for a model's
// sliding window (0 ⇒ the 512 the family ships): the explicit env override
// wins, else enough whole windows to reach the target rows, capped at 8.
// The wider default applies only when the target divides into ≥4 windows
// (the 512-window E-family, receipted +3-6%); the 1024-window models
// measured slightly WORSE at 2 windows (31B@8K: 261 vs 269 tok/s), so they
// keep single-window chunks until a receipt says otherwise.
func prefillChunkWindowsFor(slidingWindow int) int {
	if v := os.Getenv("LTHN_PREFILL_WINDOWS"); v != "" {
		if r := core.Atoi(v); r.OK {
			if n, ok := r.Value.(int); ok && n >= 1 && n <= 8 {
				return n
			}
		}
	}
	if slidingWindow <= 0 {
		slidingWindow = 512
	}
	n := batchedDensePrefillTargetRows / slidingWindow
	if n < 4 {
		return 1
	}
	return min(8, n)
}

func (s *ArchSession) batchedDensePrefillChunkLen(limit int) int {
	if limit <= 1 || s == nil || s.arch.SlidingWindow <= 0 || s.arch.SlidingWindow >= s.maxLen {
		return limit
	}
	w := s.arch.SlidingWindow
	remain := w - s.pos%w
	if remain <= 0 {
		remain = w
	}
	if remain < w {
		// realign to a window boundary first (a mid-window start after a partial append);
		// wide chunks only ever begin window-aligned so the ring slot math stays exact.
		if remain > limit {
			return limit
		}
		if limit <= remain+w/2 {
			return limit
		}
		return remain
	}
	wide := w * prefillChunkWindowsFor(w)
	if wide > limit {
		return limit
	}
	// absorb a small tail into ONE wrap-crossing chunk: the deferred-ring lane handles a batch
	// wider than the window (and the per-row staged fallback always did), while a skinny
	// follow-up chunk pays a full weight sweep for a handful of rows.
	if limit <= wide+w/2 {
		return limit
	}
	return wide
}

// batchedDenseSkipFinalFloor is the minimum FINAL-chunk width when the
// shared-suffix skip is armed: above batchedDenseICBMaxRows so a recorded-ICB
// session's DenseOne never declines the boundary chunk (a decline there falls
// the WHOLE prompt back to the per-token lane), and at steelGEMMMinRows so the
// q8 fold gate stays open.
const batchedDenseSkipFinalFloor = 32

// batchedDensePrefillChunkLenSkip is the chunk policy with the shared-suffix
// skip armed (#381): only the FINAL chunk runs the full layer stack, so
// instead of absorbing the tail into a wide full-stack chunk (up to wide+w/2
// rows paying the 20 skipped fat layers for one read row), it splits a
// MINIMAL window-aligned boundary chunk off the end — the last partial
// window, raised to batchedDenseSkipFinalFloor. e2b pp8K: the full-stack span
// falls 1081 -> 57 rows.
func (s *ArchSession) batchedDensePrefillChunkLenSkip(limit int) int {
	n := s.batchedDensePrefillChunkLen(limit)
	if n < limit {
		return n // a non-final chunk — the base policy's width stands
	}
	w := s.arch.SlidingWindow
	if w <= 0 || limit <= batchedDenseSkipFinalFloor {
		return n
	}
	rem := (s.pos + limit) % w
	if rem == 0 {
		rem = w
	}
	for rem < batchedDenseSkipFinalFloor {
		rem += w
	}
	if rem >= limit {
		return n // already minimal
	}
	// emit the skipped span first (window-aligned end); the loop calls back
	// for the boundary chunk.
	span := limit - rem
	if wide := w * prefillChunkWindowsFor(w); span > wide {
		span = wide
	}
	return span
}

func (s *ArchSession) prefillRetainedTokensBatchedDenseOne(ids []int32, scope string) ([]byte, bool, error) {
	if len(ids) == 0 {
		return nil, false, nil
	}
	if s.pos+len(ids) > s.maxLen {
		return nil, false, core.NewError(scope + ": sequence would exceed maxLen cache rows")
	}
	// A PLE arch (gemma4 E2B/E4B) batches here: the per-token PLE tensors are gathered into
	// one slab below and the gate is encoded per row inside the same command buffer — without
	// this, E-family prompts fell to n host-synced single-token forwards (O(n²) prefill).
	// Recorded-ICB (quant) sessions batch PROMPT-SCALE runs (the dense body's qmm fold over
	// the replay's own caches); short appends decline HERE — before any host embed/PLE work —
	// and keep the replay's GPU-chained lane (the dense body would decline them anyway: its
	// small-K carve-out preserves the save/restore byte contract).
	if s.state.icb != nil && (len(ids) <= batchedDenseICBMaxRows || batchedMLPFoldDisabledForTest || !gpuHasGeluKernel()) {
		return nil, false, nil
	}
	// Device-first inputs (#381): on the quant PLE lane one committed-not-waited
	// command buffer gathers the K main-embed rows AND builds the PLE tensor from
	// just the token ids — the host never dequants an embedding row.
	embBuf, pleBuf, devErr := s.prefillInputsDevice(ids)
	if devErr != nil {
		return nil, false, devErr
	}
	var embStack [16][]byte
	var embs [][]byte
	if len(ids) <= len(embStack) {
		embs = embStack[:len(ids)]
	} else {
		embs = make([][]byte, len(ids))
	}
	var pleSlab []byte
	if embBuf == nil {
		embedStart := time.Now()
		if s.canUseEmbedScratch() {
			rowBytes := s.arch.Hidden * bf16Size
			need := len(ids) * rowBytes
			if cap(s.embedScratch) < need {
				s.embedScratch = make([]byte, need)
			} else {
				s.embedScratch = s.embedScratch[:need]
			}
			for i, id := range ids {
				dst := s.embedScratch[i*rowBytes : (i+1)*rowBytes]
				emb, err := s.embedInto(dst, id)
				if err != nil {
					return nil, false, err
				}
				if len(emb) != rowBytes {
					return nil, false, core.NewError("native.prefillRetainedTokensBatchedDense: embedInto returned wrong hidden size")
				}
				embs[i] = emb
			}
		} else {
			for i, id := range ids {
				emb, err := s.embed(id)
				if err != nil {
					return nil, false, err
				}
				embs[i] = emb
			}
		}
		hostSpan("embed", embedStart, len(ids))
		pleStart := time.Now()
		var slabErr error
		pleSlab, slabErr = s.pleSlabFor(ids, embs)
		if slabErr != nil {
			return nil, false, slabErr
		}
		hostSpan("pleSlab", pleStart, len(ids))
	}
	var (
		hidden []byte
		ok     bool
		err    error
	)
	dst := s.sampleHidden
	retained := false
	if pinned, pinnedOK := s.ensureRetainedHiddenPinned(s.arch.Hidden * bf16Size); pinnedOK {
		s.resetRetainedLogits()
		dst = pinned.bytes[:s.arch.Hidden*bf16Size]
		retained = true
	}
	stepStart := time.Now()
	withAutoreleasePool(func() {
		if pleBuf != nil {
			s.state.prefillPLESlabDevice = pleBuf
			s.state.prefillEmbedDevice = embBuf
			defer func() {
				s.state.prefillPLESlabDevice = nil
				s.state.prefillEmbedDevice = nil
			}()
			hidden, ok, err = s.state.stepTokensBatchedDenseLastIntoPLE(embs, nil, s.pos, dst)
		} else if embBuf != nil {
			// dense device inputs (#381): no PLE tensor — the builder's embed rows
			// are the pass's input rows; embs stays a nil-filled K-carrier.
			s.state.prefillEmbedDevice = embBuf
			defer func() { s.state.prefillEmbedDevice = nil }()
			hidden, ok, err = s.state.stepTokensBatchedDenseLastInto(embs, s.pos, dst)
		} else if pleSlab != nil {
			hidden, ok, err = s.state.stepTokensBatchedDenseLastIntoPLE(embs, pleSlab, s.pos, dst)
		} else {
			hidden, ok, err = s.state.stepTokensBatchedDenseLastInto(embs, s.pos, dst)
		}
	})
	hostSpan("chunk.step", stepStart, len(ids))
	if err != nil || !ok {
		return nil, ok, err
	}
	if retained {
		s.sampleHidden = nil
		s.retainedHidden = hidden
	} else {
		s.sampleHidden = hidden
	}
	s.pos += len(ids)
	return hidden, true, nil
}

// prefillEmbedDeviceEnabled gates prefillInputsDevice's device-first prefill
// build (#381: the E-series PLE gather and its dense/MoE embed-rows port
// alike) — the K main-embed rows (and, on a PLE arch, the layer-major PLE
// tensor) built in one committed-not-waited command buffer from just the
// token ids. Default ON; LTHN_PREFILL_EMBED_DEVICE=0 restores the per-token
// host embed-dequant loop on every arch this seam touches (same shape as
// LTHN_PREFILL_SKIP_SHARED in flash_prompt.go).
var prefillEmbedDeviceEnabled = os.Getenv("LTHN_PREFILL_EMBED_DEVICE") != "0"

// prefillEmbedDeviceOffForTest pins the host embed loop in-process — the
// byte-identity A/B flips it instead of restarting the process with the env
// var unset (mirrors prefillSkipSharedOffForTest).
var prefillEmbedDeviceOffForTest bool

// prefillInputsDevice builds a prefill chunk's inputs DEVICE-first (#381): one
// committed-not-waited command buffer gathers the K main-embed rows AND — on a
// PLE arch — the PLE tensor at the chunk's exact layer bound, from just the
// token ids; a dense (non-PLE) arch builds the embed rows alone (pleBuf nil).
// The batched pass reads the buffers GPU-ordered on the shared queue, so the
// host skips the per-token embed dequant, the builder wait, the slab copy-out
// and the re-uploads. (nil, nil, nil) when the lane is not available (no
// device closure, stale metallib, small K) or the kill switch is off — the
// host paths take over.
func (s *ArchSession) prefillInputsDevice(ids []int32) (metal.MTLBuffer, metal.MTLBuffer, error) {
	if !prefillEmbedDeviceEnabled || prefillEmbedDeviceOffForTest || s.perLayerInputBatchDevice == nil {
		return nil, nil, nil
	}
	plePresent := s.perLayerInput != nil && len(s.state.ple) > 0
	if !plePresent && (s.perLayerInput != nil || len(s.state.ple) > 0) {
		// mixed shape — a session closure without state PLE layers or the
		// reverse (test fakes): the host paths stay authoritative.
		return nil, nil, nil
	}
	// a dense (non-PLE) session's closure ignores the layer bound: the build is
	// the embed gather alone, so outLayers stays 0.
	outLayers := 0
	if plePresent {
		outLayers = len(s.state.specs)
		if b := s.state.prefillSkipToLayer; b > 0 && b < outLayers {
			outLayers = b
		}
	}
	devStart := time.Now()
	embBuf, pleBuf, ok, err := s.perLayerInputBatchDevice(ids, outLayers)
	if err != nil || !ok {
		return nil, nil, err
	}
	hostSpan("inputsDev", devStart, len(ids))
	return embBuf, pleBuf, nil
}

// pleSlabFor gathers the per-token PLE tensors for a token batch into one
// LAYER-major slab ([numLayers × len(ids) × pliDim] bf16) — layer li's K
// per-token slices are contiguous, so the batched dense forward's PLE gate can
// run the whole layer's gelu(gate)·pli in one dispatch (and the per-row gate
// reads its slice at (li·K + i)·pliDim). nil (no error) for models without the
// per-layer-input tower.
func (s *ArchSession) pleSlabFor(ids []int32, embs [][]byte) ([]byte, error) {
	// key on the STATE's tower, not just the session closure — a session can carry
	// the closure while its decode state has no PLE layers (test fakes; the forward
	// applies the gate from state.ple, so that is the authority).
	if s.perLayerInput == nil || len(s.state.ple) == 0 {
		return nil, nil
	}
	if len(ids) != len(embs) {
		return nil, core.NewError("native.pleSlabFor: token and embedding counts differ")
	}
	numLayers, pliBytes := len(s.state.specs), s.state.pliDim*bf16Size
	tokenPLE := numLayers * pliBytes
	// A skipped prefill chunk (#381) reads only the owner layers' gate slices —
	// the slab carries just that layer-major prefix (the batch builders bound
	// their compute to it, or compute full width and copy the prefix).
	outLayers := numLayers
	if b := s.state.prefillSkipToLayer; b > 0 && b < numLayers {
		outLayers = b
	}
	pleSlab := make([]byte, len(ids)*outLayers*pliBytes)
	if s.perLayerInputBatch != nil {
		if ok, err := s.perLayerInputBatch(ids, embs, pleSlab); err != nil {
			return nil, err
		} else if ok {
			return pleSlab, nil
		}
	}
	for i, id := range ids {
		pli, err := s.perLayerInput(id, embs[i])
		if err != nil {
			return nil, err
		}
		if len(pli) != tokenPLE {
			return nil, core.NewError("native.pleSlabFor: PLE tensor size mismatch")
		}
		// the closure returns token i's [numLayers × pliDim] tensor (and may reuse its
		// scratch across calls) — scatter each layer's slice to its layer-major home.
		for li := range outLayers {
			copy(pleSlab[(li*len(ids)+i)*pliBytes:(li*len(ids)+i+1)*pliBytes], pli[li*pliBytes:(li+1)*pliBytes])
		}
	}
	return pleSlab, nil
}

func (s *ArchSession) rememberRetainedHidden(hidden []byte) {
	if s == nil || len(hidden) != s.arch.Hidden*bf16Size {
		s.resetRetainedHidden()
		return
	}
	s.resetRetainedLogits()
	if len(s.retainedHidden) == len(hidden) && len(hidden) != 0 && unsafe.Pointer(&hidden[0]) == unsafe.Pointer(&s.retainedHidden[0]) {
		return
	}
	if pinned, ok := s.ensureRetainedHiddenPinned(len(hidden)); ok {
		copy(pinned.bytes, hidden)
		s.retainedHidden = pinned.bytes[:len(hidden)]
		return
	}
	retained := s.retainedHidden[:0]
	s.retainedHidden = append(retained, hidden...)
}

func (s *ArchSession) rememberRetainedHiddenFrom(ptr *byte) {
	if s == nil || ptr == nil || s.arch.Hidden <= 0 {
		s.resetRetainedHidden()
		return
	}
	s.resetRetainedLogits()
	n := s.arch.Hidden * bf16Size
	if pinned, ok := s.ensureRetainedHiddenPinned(n); ok {
		s.retainedHidden = pinned.bytes[:n]
		copy(s.retainedHidden, unsafe.Slice(ptr, n))
		return
	}
	if cap(s.retainedHidden) < n {
		s.closeRetainedHiddenPinned()
		s.retainedHidden = make([]byte, n)
	} else {
		s.retainedHidden = s.retainedHidden[:n]
	}
	copy(s.retainedHidden, unsafe.Slice(ptr, n))
}

func (s *ArchSession) resetRetainedHidden() {
	if s == nil {
		return
	}
	s.resetRetainedLogits()
	if s.retainedHiddenPinned != nil && s.retainedHiddenPinned.bytes != nil {
		if s.retainedHiddenPinned == s.cachedPromptHiddenPinned {
			s.retainedHiddenPinned = nil
			s.retainedHidden = nil
			return
		}
		s.retainedHidden = s.retainedHiddenPinned.bytes[:0]
		return
	}
	s.retainedHidden = s.retainedHidden[:0]
}

func (s *ArchSession) rememberRetainedLogits(logits []byte) {
	if s == nil || len(logits) != s.arch.Vocab*bf16Size {
		s.resetRetainedLogits()
		return
	}
	if len(s.retainedLogits) == len(logits) && len(logits) != 0 && unsafe.Pointer(&logits[0]) == unsafe.Pointer(&s.retainedLogits[0]) {
		return
	}
	if pinned, ok := s.ensureRetainedLogitsPinned(len(logits)); ok {
		copy(pinned.bytes, logits)
		s.retainedLogits = pinned.bytes
		return
	}
	retained := s.retainedLogits[:0]
	s.retainedLogits = append(retained, logits...)
}

func (s *ArchSession) resetRetainedLogits() {
	if s == nil {
		return
	}
	if s.retainedLogitsPinned != nil && s.retainedLogitsPinned.bytes != nil {
		if s.retainedLogitsPinned == s.cachedPromptLogitsPinned {
			s.retainedLogitsPinned = nil
			s.retainedLogits = nil
			return
		}
		s.retainedLogits = s.retainedLogitsPinned.bytes[:0]
		return
	}
	s.retainedLogits = s.retainedLogits[:0]
}

func (s *ArchSession) ensureRetainedHiddenPinned(n int) (*pinnedNoCopyBytes, bool) {
	if s == nil || n <= 0 {
		return nil, false
	}
	if s.retainedHiddenPinned != nil {
		if s.retainedHiddenPinned == s.cachedPromptHiddenPinned &&
			len(s.retainedHidden) == len(s.cachedPromptHidden) &&
			len(s.retainedHidden) != 0 &&
			unsafe.Pointer(&s.retainedHidden[0]) == unsafe.Pointer(&s.cachedPromptHidden[0]) {
			s.retainedHiddenPinned = nil
			s.retainedHidden = nil
		} else if len(s.retainedHiddenPinned.bytes) == n && s.retainedHiddenPinned.buf != nil {
			return s.retainedHiddenPinned, true
		} else {
			s.closeRetainedHiddenPinned()
		}
	}
	pinned, err := newPinnedNoCopyBytes(n)
	if err != nil {
		return nil, false
	}
	s.retainedHiddenPinned = pinned
	return pinned, true
}

func (s *ArchSession) closeRetainedHiddenPinned() {
	if s == nil || s.retainedHiddenPinned == nil {
		return
	}
	if s.retainedHiddenPinned == s.cachedPromptHiddenPinned {
		s.retainedHiddenPinned = nil
		s.retainedHidden = nil
		return
	}
	s.retainedHiddenPinned.Close()
	s.retainedHiddenPinned = nil
	s.retainedHidden = nil
}

func (s *ArchSession) ensureRetainedLogitsPinned(n int) (*pinnedNoCopyBytes, bool) {
	if s == nil || n <= 0 {
		return nil, false
	}
	if s.retainedLogitsPinned != nil {
		if s.retainedLogitsPinned == s.cachedPromptLogitsPinned &&
			len(s.retainedLogits) == len(s.cachedPromptLogits) &&
			len(s.retainedLogits) != 0 &&
			unsafe.Pointer(&s.retainedLogits[0]) == unsafe.Pointer(&s.cachedPromptLogits[0]) {
			s.retainedLogitsPinned = nil
			s.retainedLogits = nil
		} else if len(s.retainedLogitsPinned.bytes) == n && s.retainedLogitsPinned.buf != nil {
			return s.retainedLogitsPinned, true
		} else {
			s.closeRetainedLogitsPinned()
		}
	}
	pinned, err := newPinnedNoCopyBytes(n)
	if err != nil {
		return nil, false
	}
	s.retainedLogitsPinned = pinned
	return pinned, true
}

func (s *ArchSession) closeRetainedLogitsPinned() {
	if s == nil || s.retainedLogitsPinned == nil {
		return
	}
	if s.retainedLogitsPinned == s.cachedPromptLogitsPinned {
		s.retainedLogitsPinned = nil
		s.retainedLogits = nil
		return
	}
	s.retainedLogitsPinned.Close()
	s.retainedLogitsPinned = nil
	s.retainedLogits = nil
}

func (s *ArchSession) retainedHiddenBuffer() metal.MTLBuffer {
	if s == nil || len(s.retainedHidden) == 0 || s.retainedHiddenPinned == nil || s.retainedHiddenPinned.buf == nil || len(s.retainedHiddenPinned.bytes) != len(s.retainedHidden) {
		return nil
	}
	if unsafe.Pointer(&s.retainedHidden[0]) != unsafe.Pointer(&s.retainedHiddenPinned.bytes[0]) {
		return nil
	}
	return s.retainedHiddenPinned.buf
}

func (s *ArchSession) retainedHiddenBufferFor(hidden []byte) metal.MTLBuffer {
	if s == nil || len(hidden) == 0 {
		return nil
	}
	if len(hidden) == len(s.retainedHidden) && len(s.retainedHidden) != 0 && unsafe.Pointer(&hidden[0]) == unsafe.Pointer(&s.retainedHidden[0]) {
		if buf := s.retainedHiddenBuffer(); buf != nil {
			return buf
		}
	}
	if len(hidden) == len(s.cachedPromptHidden) && len(s.cachedPromptHidden) != 0 && unsafe.Pointer(&hidden[0]) == unsafe.Pointer(&s.cachedPromptHidden[0]) {
		return s.cachedPromptHiddenBuffer()
	}
	return nil
}

func (s *ArchSession) retainedLogitsBuffer() metal.MTLBuffer {
	if s == nil || len(s.retainedLogits) == 0 || s.retainedLogitsPinned == nil || s.retainedLogitsPinned.buf == nil || len(s.retainedLogitsPinned.bytes) != len(s.retainedLogits) {
		return nil
	}
	if unsafe.Pointer(&s.retainedLogits[0]) != unsafe.Pointer(&s.retainedLogitsPinned.bytes[0]) {
		return nil
	}
	return s.retainedLogitsPinned.buf
}

func (s *ArchSession) retainedLogitsBufferFor(logits []byte) metal.MTLBuffer {
	if s == nil || len(logits) == 0 {
		return nil
	}
	if len(logits) == len(s.retainedLogits) && len(s.retainedLogits) != 0 && unsafe.Pointer(&logits[0]) == unsafe.Pointer(&s.retainedLogits[0]) {
		if buf := s.retainedLogitsBuffer(); buf != nil {
			return buf
		}
	}
	if len(logits) == len(s.cachedPromptLogits) && len(s.cachedPromptLogits) != 0 && unsafe.Pointer(&logits[0]) == unsafe.Pointer(&s.cachedPromptLogits[0]) {
		return s.cachedPromptLogitsBuffer()
	}
	return nil
}

func (s *ArchSession) mtpVerifyHiddenRowsScratch(k, rowBytes int) ([][]byte, bool) {
	if s == nil || k <= 0 || rowBytes <= 0 {
		return nil, false
	}
	need := k * rowBytes
	if s.mtpVerifyHiddenPinned != nil {
		if len(s.mtpVerifyHiddenPinned.bytes) != need || s.mtpVerifyHiddenPinned.buf == nil {
			s.mtpVerifyHiddenPinned.Close()
			s.mtpVerifyHiddenPinned = nil
			s.mtpVerifyHiddenRows = nil
		}
	}
	if s.mtpVerifyHiddenPinned == nil {
		pinned, err := newPinnedNoCopyBytes(need)
		if err != nil {
			return nil, false
		}
		s.mtpVerifyHiddenPinned = pinned
	}
	if cap(s.mtpVerifyHiddenRows) < k {
		s.mtpVerifyHiddenRows = make([][]byte, k)
	} else {
		s.mtpVerifyHiddenRows = s.mtpVerifyHiddenRows[:k]
	}
	for i := range k {
		s.mtpVerifyHiddenRows[i] = s.mtpVerifyHiddenPinned.bytes[i*rowBytes : (i+1)*rowBytes]
	}
	return s.mtpVerifyHiddenRows, true
}

func (s *ArchSession) mtpVerifyRowScratch(k int) []int32 {
	if s == nil || k <= 0 {
		return nil
	}
	if cap(s.mtpVerifyRows) < k {
		s.mtpVerifyRows = make([]int32, k)
	} else {
		s.mtpVerifyRows = s.mtpVerifyRows[:k]
	}
	return s.mtpVerifyRows
}

// Step decodes one token's embedding at the current cache position over the
// persistent KV cache, returning its output hidden state (dModel bf16 bytes) and
// advancing the position — the contract-native incremental decode
// (model.DecodeStepper), so model.Generate drives this session O(1)/token. The
// returned hidden is a fresh Go copy (stepToken copies out of the device
// buffer), so it survives the per-step autorelease pool. PLE models (E2B/E4B)
// derive a per-layer-input tensor from each token id, which Step (embedding
// only) can't supply — they must generate via Generate, so Step rejects a PLE
// session.
func (s *ArchSession) Step(emb []byte) ([]byte, error) {
	if s.perLayerInput != nil {
		return nil, core.NewError("native.ArchSession.Step: per-layer-input models must use Generate, not Step")
	}
	if len(emb) != s.arch.Hidden*bf16Size {
		return nil, core.NewError("native.ArchSession.Step: emb must be hidden bf16 bytes")
	}
	if s.pos >= s.maxLen {
		return nil, core.NewError("native.ArchSession.Step: sequence would exceed maxLen cache rows")
	}
	var res []byte
	var err error
	withAutoreleasePool(func() {
		if s.state.icb != nil && !icbDisabledForTest { // recorded encode-bypass: replay one token over the ICB's caches
			res = s.state.icb.stepBody(emb, s.pos, nil)
		} else {
			res, err = s.state.stepToken(emb, s.pos)
		}
	})
	if err != nil {
		return nil, err
	}
	s.pos++
	return res, nil
}

// StepWithID is Step with the token id available — the contract's id-aware
// incremental step (model.Generate calls it in preference to Step when present).
// gemma4 E2B/E4B per-layer-input models need the id: the per-layer input is gathered
// from embed_tokens_per_layer[id] (not derivable from the token embedding), so
// StepWithID computes the per-layer-input tensor from (id, emb) and threads it into
// the step, exactly as Generate does. For a model without the PLE tower it is just
// Step (perLayerInput is nil), so it carries no PLE guard.
func (s *ArchSession) StepWithID(id int32, emb []byte) ([]byte, error) {
	if len(emb) != s.arch.Hidden*bf16Size {
		return nil, core.NewError("native.ArchSession.StepWithID: emb must be hidden bf16 bytes")
	}
	if s.pos >= s.maxLen {
		return nil, core.NewError("native.ArchSession.StepWithID: sequence would exceed maxLen cache rows")
	}
	var res []byte
	var err error
	withAutoreleasePool(func() {
		var pli []byte
		if s.perLayerInput != nil { // PLE: per-layer inputs from this token's id + embedding
			if pli, err = s.perLayerInput(id, emb); err != nil {
				return
			}
			s.state.perLayerInput = pli
		}
		if s.state.icb != nil && !icbDisabledForTest { // recorded encode-bypass: replay one token over the ICB's caches
			res = s.state.icb.stepBody(emb, s.pos, pli)
		} else {
			res, err = s.state.stepToken(emb, s.pos)
		}
	})
	if err != nil {
		return nil, err
	}
	s.pos++
	return res, nil
}

// sharedStepEligible reports whether this session's one-token forward can
// encode into a caller-owned shared submission (see sharedEncodeEligible for
// the state-level conditions; PLE models additionally decline — their
// per-token prologue is host-produced and stays on the per-lane path).
func (s *ArchSession) sharedStepEligible() bool {
	return s != nil && s.perLayerInput == nil && s.state.sharedEncodeEligible()
}

// stepIDEncodeShared encodes this session's next token's forward into the
// owner's shared submission — no commit, no wait, no position bump. The final
// hidden's buffer lands in sink.finalOut, readable only after the owner's
// wait; the owner bumps s.pos itself once the shared submission completes (a
// failed shared attempt must leave the session untouched for the per-lane
// retry).
func (s *ArchSession) stepIDEncodeShared(id int32, sink *sharedStepSink) error {
	if s.pos >= s.maxLen {
		return core.NewError("native.ArchSession.stepIDEncodeShared: sequence would exceed maxLen cache rows")
	}
	if s.perLayerInput != nil {
		return core.NewError("native.ArchSession.stepIDEncodeShared: PLE models keep the per-lane step")
	}
	emb, err := s.embedID(id)
	if err != nil {
		return err
	}
	_, err = s.state.stepTokenEncode(emb, s.pos, false, true, nil, sink)
	return err
}

func (s *ArchSession) stepIDInPool(id int32) ([]byte, error) {
	emb, err := s.embedID(id)
	if err != nil {
		return nil, err
	}
	var pli []byte
	if s.perLayerInput != nil { // gemma4 PLE: per-token per-layer-input tensor, from this token's embedding
		_ptPLE := ptStart()
		pli, err = s.perLayerInput(id, emb)
		ptEnd(0, _ptPLE)
		if err != nil {
			return nil, err
		}
		s.state.perLayerInput = pli
	}
	var h []byte
	_ptICB := ptStart()
	if s.state.icb != nil && !icbDisabledForTest { // recorded encode-bypass: replay one token over the ICB (as Step/StepWithID do)
		icb := s.state.icb
		if direct, ok := s.retainHiddenDirectFromICB(icb, emb, s.pos, pli); ok {
			h = direct
		} else {
			if icb.lastOutPtr == nil {
				icb.cacheLastOutContents()
			}
			icb.stepBodyNoResult(emb, s.pos, pli)
			h = s.retainHiddenReadbackFrom(icb.lastOutPtr)
		}
		if h == nil {
			h = make([]byte, s.arch.Hidden*bf16Size)
			icb.copyLastOutInto(h)
		}
	} else if h, err = s.state.stepToken(emb, s.pos); err != nil {
		return nil, err
	}
	ptEnd(1, _ptICB)
	s.pos++
	return h, nil
}

func (s *ArchSession) stepIDRetainedInPool(id int32) ([]byte, error) {
	emb, err := s.embedID(id)
	if err != nil {
		return nil, err
	}
	var pli []byte
	if s.perLayerInput != nil {
		_ptPLE := ptStart()
		pli, err = s.perLayerInput(id, emb)
		ptEnd(0, _ptPLE)
		if err != nil {
			return nil, err
		}
		s.state.perLayerInput = pli
	}
	var h []byte
	_ptICB := ptStart()
	if s.state.icb != nil && !icbDisabledForTest {
		icb := s.state.icb
		if direct, ok := s.retainHiddenDirectFromICB(icb, emb, s.pos, pli); ok {
			h = direct
		} else {
			if icb.lastOutPtr == nil {
				icb.cacheLastOutContents()
			}
			icb.stepBodyNoResult(emb, s.pos, pli)
			h = s.retainHiddenReadbackFrom(icb.lastOutPtr)
		}
		if h == nil {
			h = make([]byte, s.arch.Hidden*bf16Size)
			icb.copyLastOutInto(h)
		}
	} else if pinned, ok := s.ensureRetainedHiddenPinned(s.arch.Hidden * bf16Size); ok {
		s.resetRetainedLogits()
		h, err = s.state.stepTokenInto(emb, s.pos, pinned.bytes[:s.arch.Hidden*bf16Size])
		if err != nil {
			return nil, err
		}
		s.retainedHidden = h
	} else if h, err = s.state.stepToken(emb, s.pos); err != nil {
		return nil, err
	}
	ptEnd(1, _ptICB)
	s.pos++
	return h, nil
}

func (s *ArchSession) generateFromHidden(hidden []byte, maxNew, eosID int, firstLogits []byte) ([]int32, error) {
	return s.generateFromHiddenSuppressed(hidden, maxNew, eosID, firstLogits, nil)
}

func (s *ArchSession) generateFromHiddenSuppressed(hidden []byte, maxNew, eosID int, firstLogits []byte, suppress []int32) ([]int32, error) {
	return s.generateFromHiddenSuppressedEach(hidden, maxNew, eosID, firstLogits, suppress, nil, nil)
}

func (s *ArchSession) generateFromHiddenSuppressedEach(hidden []byte, maxNew, eosID int, firstLogits []byte, suppress []int32, transform TokenTransform, yield func(int32) bool) ([]int32, error) {
	if maxNew <= 0 {
		return nil, core.NewError("native.ArchSession.generateFromHidden: maxNew must be > 0")
	}
	if len(hidden) != s.arch.Hidden*bf16Size {
		return nil, core.NewError("native.ArchSession.generateFromHidden: hidden must be hidden bf16 bytes")
	}
	if firstLogits != nil && len(firstLogits) != s.arch.Vocab*bf16Size {
		return nil, core.NewError("native.ArchSession.generateFromHidden: logits must be vocab bf16 bytes")
	}
	if s.pos+maxNew > s.maxLen {
		return nil, core.NewError("native.ArchSession.generateFromHidden: sequence would exceed maxLen cache rows")
	}
	var gen []int32
	var err error
	withAutoreleasePool(func() {
		gen, err = s.generateFromHiddenInPool(hidden, maxNew, eosID, firstLogits, nil, suppress, transform, yield)
	})
	return gen, err
}

func (s *ArchSession) generateFromLogitsInPool(firstLogits []byte, maxNew, eosID int, suppress []int32, transform TokenTransform, yield func(int32) bool) ([]int32, error) {
	next, err := greedyBF16Suppressed(firstLogits, s.arch.Vocab, suppress)
	if err != nil {
		return nil, err
	}
	if transform != nil {
		next = transform(next)
	}
	gen := make([]int32, 0, maxNew)
	gen = append(gen, next)
	stop := (yield != nil && !yield(next)) || (eosID >= 0 && int(next) == eosID)
	// The chained/pipelined tails consume the GPU argmax head MID-CHAIN (the next step's input
	// binds the head's token buffer with no host round-trip) — engaging without it fails at the
	// first link. The head requirement is the head's own capability, not the input seam's: the
	// old bits==4 seam gate masked this until the gather widened to every affine width.
	if s.encNextInputsGPU != nil && s.plScratchNew != nil && s.state.icb != nil && s.headEnc != nil && s.greedy != nil &&
		s.canUseDirectHeadGreedy() &&
		!stepGreedyChainDisabled && !chainedGPUInputsDisabled && !icbDisabledForTest && transform == nil {
		if pipelinedGPUDecodeEnabled && s.recordPeerICB != nil {
			return s.generatePipelinedGPUTail(gen, maxNew, eosID, suppress, yield, stop)
		}
		return s.generateChainedGPUTail(gen, maxNew, eosID, suppress, yield, stop)
	}
	var hidden []byte
	for !stop && len(gen) < maxNew {
		prev := gen[len(gen)-1]
		if hidden, err = s.stepIDRetainedInPool(prev); err != nil {
			return nil, err
		}
		if next, err = s.headGreedyOrLogits(hidden, suppress, nil, nil, false); err != nil {
			return nil, err
		}
		if transform != nil {
			next = transform(next)
		}
		gen = append(gen, next)
		s.rememberRetainedHidden(hidden)
		stop = (yield != nil && !yield(next)) || (eosID >= 0 && int(next) == eosID)
	}
	if hidden, err = s.stepIDRetainedInPool(gen[len(gen)-1]); err != nil {
		return nil, err
	}
	s.rememberRetainedHidden(hidden)
	return gen, nil
}

func (s *ArchSession) generateSampledFromLogitsInPool(firstLogits []byte, maxNew int, stopTokens []int32, sampler *model.Sampler, params model.SampleParams, transform model.TokenTransform, yield func(int32) bool, cacheFinal bool) ([]int32, error) {
	gen := make([]int32, 0, maxNew)
	history := s.sampleHistoryScratchFor(params, maxNew)
	finalHistory := history
	defer func() { s.sampleHistory = finalHistory }()

	pickParams := params
	if params.MinTokensBeforeStop > 0 {
		pickParams.SuppressTokens = s.suppressionTokensScratch(params.SuppressTokens, stopTokens)
	}
	next, err := s.sampleTokenFromLogits(firstLogits, sampler, pickParams, history)
	if err != nil {
		return nil, err
	}
	if transform != nil {
		next = transform(next)
	}
	gen = append(gen, next)
	if params.RepeatPenalty > 1 {
		history = append(history, next)
		finalHistory = history
	}
	stop := (yield != nil && !yield(next)) || nativeTokenInSet(next, stopTokens)
	if !cacheFinal && (stop || len(gen) >= maxNew) {
		return gen, nil
	}
	if !stop && len(gen) < maxNew && s.sampledChainedGPUTailCanContinue(params, history, transform) {
		var tail []int32
		tail, finalHistory, err = s.generateSampledChainedGPUTail(gen, maxNew, stopTokens, sampler, params, yield, cacheFinal, 0, history)
		if err != nil {
			return nil, err
		}
		return tail, nil
	}
	hidden, err := s.stepIDRetainedInPool(next)
	if err != nil {
		return nil, err
	}
	s.rememberRetainedHidden(hidden)
	if stop || len(gen) >= maxNew {
		return gen, nil
	}
	var tail []int32
	tail, finalHistory, err = s.generateSampledFromHiddenInPoolWithHistory(hidden, maxNew-len(gen), stopTokens, sampler, params, transform, yield, cacheFinal, len(gen), history)
	if err != nil {
		return nil, err
	}
	gen = append(gen, tail...)
	return gen, nil
}

func (s *ArchSession) sampleTokenFromLogits(logits []byte, sampler *model.Sampler, params model.SampleParams, history []int32) (int32, error) {
	if sampledGreedyParamsEligible(params) {
		return greedyBF16Suppressed(logits, s.arch.Vocab, params.SuppressTokens)
	}
	if sampledTopOneGreedyParamsEligible(params, history) {
		sampler.Draw()
		return greedyBF16Suppressed(logits, s.arch.Vocab, params.SuppressTokens)
	}
	if sampleLogitsTokenCPUPreferred(params, s.arch.Vocab) {
		return sampleSmallVocabBF16(logits, s.arch.Vocab, sampler, params)
	}
	if !s.retainedLogitsCompactSampleEligible(params) {
		logitsBuf := s.retainedLogitsBufferFor(logits)
		if logitsBuf != nil && s.retainedLogitsSampleParamsEligible(params) {
			token, ok, err := s.headEnc.sampleLogitsBufferInPool(logitsBuf, params, sampler.Draw(), history)
			if err != nil {
				return 0, err
			}
			if ok {
				return token, nil
			}
		}
	}
	if s.retainedLogitsCompactSampleEligible(params) {
		candidateLogits, candidateIDs, ok, err := s.sampleTopKCandidatesFromLogits(logits, params, history)
		if err != nil {
			return 0, err
		}
		if ok {
			candidateParams := params
			candidateParams.RepeatPenalty = 1
			return sampleSortedBF16Candidates(candidateLogits, candidateIDs, sampler, candidateParams)
		}
	}
	pickLogits := logits
	var err error
	if params.RepeatPenalty > 1 {
		pickLogits, err = s.repeatPenaltyLogitsScratch(logits, s.arch.Vocab, history, params.RepeatPenalty)
		if err != nil {
			return 0, err
		}
	}
	return s.sampleVocabBF16(pickLogits, s.arch.Vocab, sampler, params)
}

func (s *ArchSession) retainedLogitsCompactSampleEligible(params model.SampleParams) bool {
	return s != nil && params.TopK > 0 && params.TopK <= headSampleTopKMaxK && params.TopK <= s.arch.Vocab
}

func (s *ArchSession) sampleTopKCandidatesFromLogits(logits []byte, params model.SampleParams, history []int32) ([]byte, []int32, bool, error) {
	vocab := s.arch.Vocab
	if len(logits) != vocab*bf16Size {
		return nil, nil, true, core.NewError("native.ArchSession.sampleTopKCandidatesFromLogits: logits must be vocab bf16 bytes")
	}
	topK := params.TopK
	if topK <= 0 || topK > headSampleTopKMaxK || topK > vocab {
		return nil, nil, false, nil
	}
	if cap(s.sampleCandidateLogits) < topK*bf16Size {
		s.sampleCandidateLogits = make([]byte, topK*bf16Size)
	} else {
		s.sampleCandidateLogits = s.sampleCandidateLogits[:topK*bf16Size]
	}
	if cap(s.sampleCandidateIDs) < topK {
		s.sampleCandidateIDs = make([]int32, topK)
	} else {
		s.sampleCandidateIDs = s.sampleCandidateIDs[:topK]
	}
	var scores [headSampleTopKMaxK]float32
	var penaltyIDs []int32
	if params.RepeatPenalty > 1 && len(history) > 0 {
		penaltyIDs = s.repeatPenaltyIDsScratch(vocab, history)
	}
	penaltyPos := 0
	count := 0
	for id := range vocab {
		if tokenSuppressed(id, params.SuppressTokens) {
			continue
		}
		off := id * bf16Size
		lo, hi := logits[off], logits[off+1]
		for penaltyPos < len(penaltyIDs) && penaltyIDs[penaltyPos] < int32(id) {
			penaltyPos++
		}
		if penaltyPos < len(penaltyIDs) && penaltyIDs[penaltyPos] == int32(id) {
			v := bf16ToF32(lo, hi)
			if v > 0 {
				v /= params.RepeatPenalty
			} else {
				v *= params.RepeatPenalty
			}
			h := f32ToBF16(v)
			lo, hi = byte(h), byte(h>>8)
		}
		v := bf16ToF32(lo, hi)
		insert := count
		for insert > 0 && (v > scores[insert-1] || (v == scores[insert-1] && int32(id) < s.sampleCandidateIDs[insert-1])) {
			insert--
		}
		if insert >= topK {
			continue
		}
		if count < topK {
			count++
		}
		for j := count - 1; j > insert; j-- {
			scores[j] = scores[j-1]
			s.sampleCandidateIDs[j] = s.sampleCandidateIDs[j-1]
			prev := (j - 1) * bf16Size
			dst := j * bf16Size
			s.sampleCandidateLogits[dst] = s.sampleCandidateLogits[prev]
			s.sampleCandidateLogits[dst+1] = s.sampleCandidateLogits[prev+1]
		}
		scores[insert] = v
		s.sampleCandidateIDs[insert] = int32(id)
		dst := insert * bf16Size
		s.sampleCandidateLogits[dst] = lo
		s.sampleCandidateLogits[dst+1] = hi
	}
	if count == 0 {
		return nil, nil, true, core.NewError("native.ArchSession.sampleTopKCandidatesFromLogits: all vocab ids are suppressed")
	}
	return s.sampleCandidateLogits[:count*bf16Size], s.sampleCandidateIDs[:count], true, nil
}

func sampleSortedBF16Candidates(logits []byte, ids []int32, sampler *model.Sampler, params model.SampleParams) (int32, error) {
	if sampler == nil {
		return 0, core.NewError("native.sampleSortedBF16Candidates: nil sampler")
	}
	if len(ids) == 0 {
		return 0, core.NewError("native.sampleSortedBF16Candidates: empty candidates")
	}
	if len(ids) > headSampleTopKMaxK {
		return 0, core.NewError("native.sampleSortedBF16Candidates: too many candidates")
	}
	if len(logits) != len(ids)*bf16Size {
		return 0, core.NewError("native.sampleSortedBF16Candidates: logits must be candidate bf16 bytes")
	}
	if sampledGreedyParamsEligible(params) {
		best := -1
		var bestV float32
		for i, id := range ids {
			if nativeTokenInSet(id, params.SuppressTokens) {
				continue
			}
			v := bf16ToF32(logits[i*bf16Size], logits[i*bf16Size+1])
			if best < 0 || v > bestV {
				best, bestV = i, v
			}
		}
		if best < 0 {
			return 0, core.NewError("native.sampleSortedBF16Candidates: all candidates are suppressed")
		}
		return ids[best], nil
	}
	if params.TopK == 1 {
		for _, id := range ids {
			if nativeTokenInSet(id, params.SuppressTokens) {
				continue
			}
			sampler.Draw()
			return id, nil
		}
		return 0, core.NewError("native.sampleSortedBF16Candidates: all candidates are suppressed")
	}
	temp := params.Temperature
	if temp <= 0 {
		temp = 1
	}
	var weights [headSampleTopKMaxK]float32
	maxL := float32(math.Inf(-1))
	allowed := 0
	for i, id := range ids {
		if nativeTokenInSet(id, params.SuppressTokens) {
			weights[i] = float32(math.Inf(-1))
			continue
		}
		v := bf16ToF32(logits[i*bf16Size], logits[i*bf16Size+1]) / temp
		weights[i] = v
		allowed++
		if v > maxL {
			maxL = v
		}
	}
	if allowed == 0 {
		return 0, core.NewError("native.sampleSortedBF16Candidates: all candidates are suppressed")
	}
	for i := range ids {
		if weights[i] == float32(math.Inf(-1)) {
			weights[i] = 0
			continue
		}
		weights[i] = float32(math.Exp(float64(weights[i] - maxL)))
	}
	keep := len(ids)
	if params.TopK > 0 && params.TopK < keep {
		keep = params.TopK
	}
	if params.TopP > 0 && params.TopP < 1 {
		var keptMass float32
		for i := 0; i < keep; i++ {
			keptMass += weights[i]
		}
		var cum float32
		n := 0
		for n < keep {
			cum += weights[n]
			n++
			if cum >= params.TopP*keptMass {
				break
			}
		}
		keep = n
	}
	if params.MinP > 0 && keep > 0 {
		threshold := weights[0] * params.MinP
		n := 0
		for n < keep && weights[n] >= threshold {
			n++
		}
		if n > 0 {
			keep = n
		}
	}
	var ksum float32
	for i := 0; i < keep; i++ {
		ksum += weights[i]
	}
	if ksum == 0 {
		return 0, core.NewError("native.sampleSortedBF16Candidates: empty sampled distribution")
	}
	target := sampler.Draw() * ksum
	var acc float32
	for i := 0; i < keep; i++ {
		acc += weights[i]
		if acc >= target {
			return ids[i], nil
		}
	}
	return ids[keep-1], nil
}

func sampleSmallVocabBF16(logits []byte, vocab int, sampler *model.Sampler, params model.SampleParams) (int32, error) {
	if sampler == nil {
		return 0, core.NewError("native.sampleSmallVocabBF16: nil sampler")
	}
	if vocab <= 0 || vocab > headSampleTopKMaxK || len(logits) != vocab*bf16Size {
		return 0, core.NewError("native.sampleSmallVocabBF16: logits must be small-vocab bf16 bytes")
	}
	if sampledGreedyParamsEligible(params) {
		return greedyBF16Suppressed(logits, vocab, params.SuppressTokens)
	}
	if params.TopK == 1 {
		next, err := greedyBF16Suppressed(logits, vocab, params.SuppressTokens)
		if err != nil {
			return 0, err
		}
		sampler.Draw()
		return next, nil
	}
	temp := params.Temperature
	if temp <= 0 {
		temp = 1
	}
	var scaled [headSampleTopKMaxK]float32
	var probs [headSampleTopKMaxK]float32
	var order [headSampleTopKMaxK]int
	maxL := float32(math.Inf(-1))
	allowed := 0
	for i := range vocab {
		if tokenSuppressed(i, params.SuppressTokens) {
			scaled[i] = float32(math.Inf(-1))
			continue
		}
		v := bf16ToF32(logits[i*bf16Size], logits[i*bf16Size+1]) / temp
		scaled[i] = v
		allowed++
		if v > maxL {
			maxL = v
		}
	}
	if allowed == 0 {
		return 0, core.NewError("native.sampleSmallVocabBF16: all tokens are suppressed")
	}
	var sum float32
	for i := range vocab {
		e := float32(math.Exp(float64(scaled[i] - maxL)))
		probs[i] = e
		sum += e
		order[i] = i
	}
	for i := range vocab {
		probs[i] /= sum
	}
	for i := 1; i < vocab; i++ {
		key := order[i]
		j := i - 1
		for j >= 0 && probs[order[j]] < probs[key] {
			order[j+1] = order[j]
			j--
		}
		order[j+1] = key
	}
	keep := vocab
	if params.TopK > 0 && params.TopK < keep {
		keep = params.TopK
	}
	if params.TopP > 0 && params.TopP < 1 {
		var keptMass float32
		for i := 0; i < keep; i++ {
			keptMass += probs[order[i]]
		}
		var cum float32
		n := 0
		for n < keep {
			cum += probs[order[n]]
			n++
			if cum >= params.TopP*keptMass {
				break
			}
		}
		keep = n
	}
	if params.MinP > 0 && keep > 0 {
		threshold := probs[order[0]] * params.MinP
		n := 0
		for n < keep && probs[order[n]] >= threshold {
			n++
		}
		if n > 0 {
			keep = n
		}
	}
	var ksum float32
	for i := 0; i < keep; i++ {
		ksum += probs[order[i]]
	}
	if ksum == 0 {
		return 0, core.NewError("native.sampleSmallVocabBF16: empty sampled distribution")
	}
	target := sampler.Draw() * ksum
	var acc float32
	for i := 0; i < keep; i++ {
		acc += probs[order[i]]
		if acc >= target {
			return int32(order[i]), nil
		}
	}
	return int32(order[keep-1]), nil
}

func (s *ArchSession) sampleVocabBF16(logits []byte, vocab int, sampler *model.Sampler, params model.SampleParams) (int32, error) {
	if vocab <= headSampleTopKMaxK {
		return sampleSmallVocabBF16(logits, vocab, sampler, params)
	}
	if sampler == nil {
		return 0, core.NewError("native.ArchSession.sampleVocabBF16: nil sampler")
	}
	if vocab <= 0 || len(logits) != vocab*bf16Size {
		return 0, core.NewError("native.ArchSession.sampleVocabBF16: logits must be vocab bf16 bytes")
	}
	if sampledGreedyParamsEligible(params) {
		return greedyBF16Suppressed(logits, vocab, params.SuppressTokens)
	}
	if params.TopK == 1 {
		next, err := greedyBF16Suppressed(logits, vocab, params.SuppressTokens)
		if err != nil {
			return 0, err
		}
		sampler.Draw()
		return next, nil
	}
	if hostTopKSamplePreferred(params, vocab) {
		// TopK>1 at real vocab: one-pass host candidate select + the shared
		// candidate sampler — the fast lane the GPU selection rungs decline to
		// (head_host_topk.go carries the measurements).
		return sampleHostTopKBF16(logits, vocab, sampler, params)
	}
	rankFilter := sampleRankPrefixPreferred(params, vocab)
	s.sampleScaled = nil
	temp := params.Temperature
	if temp <= 0 {
		temp = 1
	}
	noSuppress := len(params.SuppressTokens) == 0
	maxL := float32(math.Inf(-1))
	allowed := 0
	if noSuppress {
		allowed = vocab
		for i := range vocab {
			v := bf16ToF32(logits[i*bf16Size], logits[i*bf16Size+1]) / temp
			if v > maxL {
				maxL = v
			}
		}
	} else {
		for i := range vocab {
			if tokenSuppressed(i, params.SuppressTokens) {
				continue
			}
			v := bf16ToF32(logits[i*bf16Size], logits[i*bf16Size+1]) / temp
			allowed++
			if v > maxL {
				maxL = v
			}
		}
	}
	if allowed == 0 {
		return 0, core.NewError("native.ArchSession.sampleVocabBF16: all tokens are suppressed")
	}
	if !rankFilter {
		s.sampleProbs = nil
		s.sampleOrder = nil
		if noSuppress {
			return sampleVocabBF16InVocabOrderStreamingNoSuppress(logits, vocab, sampler, temp, maxL)
		}
		return sampleVocabBF16InVocabOrderStreaming(logits, vocab, sampler, params, temp, maxL)
	}
	s.sampleProbs = nil
	if cap(s.sampleOrder) < vocab {
		s.sampleOrder = make([]int32, vocab)
	} else {
		s.sampleOrder = s.sampleOrder[:vocab]
	}
	for i := range vocab {
		s.sampleOrder[i] = int32(i)
	}
	if noSuppress {
		probTotal := sampleVocabBF16WeightTotalNoSuppress(logits, vocab, temp, maxL)
		keep := rankSampleOrderPrefixLogitsNoSuppress(s.sampleOrder, logits, probTotal, params, temp, maxL)
		var ksum float32
		for i := range keep {
			ksum += sampleVocabBF16IDWeightNoSuppress(logits, s.sampleOrder[i], temp, maxL)
		}
		if ksum == 0 {
			return 0, core.NewError("native.ArchSession.sampleVocabBF16: empty sampled distribution")
		}
		target := sampler.Draw() * ksum
		var acc float32
		for i := range keep {
			acc += sampleVocabBF16IDWeightNoSuppress(logits, s.sampleOrder[i], temp, maxL)
			if acc >= target {
				return s.sampleOrder[i], nil
			}
		}
		return s.sampleOrder[keep-1], nil
	}
	probTotal := sampleVocabBF16WeightTotal(logits, vocab, params, temp, maxL)
	keep := rankSampleOrderPrefixLogits(s.sampleOrder, logits, probTotal, params, temp, maxL)
	var ksum float32
	for i := range keep {
		ksum += sampleVocabBF16IDWeight(logits, s.sampleOrder[i], params, temp, maxL)
	}
	if ksum == 0 {
		return 0, core.NewError("native.ArchSession.sampleVocabBF16: empty sampled distribution")
	}
	target := sampler.Draw() * ksum
	var acc float32
	for i := range keep {
		acc += sampleVocabBF16IDWeight(logits, s.sampleOrder[i], params, temp, maxL)
		if acc >= target {
			return s.sampleOrder[i], nil
		}
	}
	return s.sampleOrder[keep-1], nil
}

func sampleVocabBF16InVocabOrderStreamingNoSuppress(logits []byte, vocab int, sampler *model.Sampler, temp, maxL float32) (int32, error) {
	var sum float32
	for i := range vocab {
		v := bf16ToF32(logits[i*bf16Size], logits[i*bf16Size+1]) / temp
		sum += float32(math.Exp(float64(v - maxL)))
	}
	if sum == 0 {
		return 0, core.NewError("native.ArchSession.sampleVocabBF16: empty sampled distribution")
	}
	target := sampler.Draw() * sum
	var acc float32
	for i := range vocab {
		v := bf16ToF32(logits[i*bf16Size], logits[i*bf16Size+1]) / temp
		acc += float32(math.Exp(float64(v - maxL)))
		if acc >= target {
			return int32(i), nil
		}
	}
	return int32(vocab - 1), nil
}

func sampleVocabBF16InVocabOrderStreaming(logits []byte, vocab int, sampler *model.Sampler, params model.SampleParams, temp, maxL float32) (int32, error) {
	var sum float32
	for i := range vocab {
		if tokenSuppressed(i, params.SuppressTokens) {
			continue
		}
		v := bf16ToF32(logits[i*bf16Size], logits[i*bf16Size+1]) / temp
		sum += float32(math.Exp(float64(v - maxL)))
	}
	if sum == 0 {
		return 0, core.NewError("native.ArchSession.sampleVocabBF16: empty sampled distribution")
	}
	target := sampler.Draw() * sum
	var acc float32
	for i := range vocab {
		e := float32(0)
		if !tokenSuppressed(i, params.SuppressTokens) {
			v := bf16ToF32(logits[i*bf16Size], logits[i*bf16Size+1]) / temp
			e = float32(math.Exp(float64(v - maxL)))
		}
		acc += e
		if acc >= target {
			return int32(i), nil
		}
	}
	return int32(vocab - 1), nil
}

func sampleVocabBF16WeightTotal(logits []byte, vocab int, params model.SampleParams, temp, maxL float32) float32 {
	var sum float32
	for i := range vocab {
		sum += sampleVocabBF16IDWeight(logits, int32(i), params, temp, maxL)
	}
	return sum
}

func sampleVocabBF16WeightTotalNoSuppress(logits []byte, vocab int, temp, maxL float32) float32 {
	var sum float32
	for i := range vocab {
		sum += sampleVocabBF16IDWeightNoSuppress(logits, int32(i), temp, maxL)
	}
	return sum
}

func sampleVocabBF16IDWeight(logits []byte, id int32, params model.SampleParams, temp, maxL float32) float32 {
	if id < 0 || int(id) >= len(logits)/bf16Size || nativeTokenInSet(id, params.SuppressTokens) {
		return 0
	}
	v := bf16ToF32(logits[int(id)*bf16Size], logits[int(id)*bf16Size+1]) / temp
	return float32(math.Exp(float64(v - maxL)))
}

func sampleVocabBF16IDWeightNoSuppress(logits []byte, id int32, temp, maxL float32) float32 {
	v := bf16ToF32(logits[int(id)*bf16Size], logits[int(id)*bf16Size+1]) / temp
	return float32(math.Exp(float64(v - maxL)))
}

func rankSampleOrderPrefixLogits(order []int32, logits []byte, probTotal float32, params model.SampleParams, temp, maxL float32) int {
	if len(order) == 0 {
		return 0
	}
	if probTotal <= 0 {
		probTotal = 1
	}
	heapifySampleOrderLogits(order, logits, params)
	heapLen := len(order)
	popped := 0
	keptMass := float32(0)
	if params.TopK > 0 && params.TopK < heapLen {
		for popped < params.TopK {
			id := popSampleOrderHeapLogits(order, logits, params, heapLen)
			heapLen--
			popped++
			keptMass += sampleVocabBF16IDWeight(logits, id, params, temp, maxL)
		}
		reverseSampleOrderTailToPrefix(order, popped)
		keep := popped
		if params.TopP > 0 && params.TopP < 1 {
			keep = sampleOrderTopPKeepLogits(order, logits, params, temp, maxL, keep, params.TopP*keptMass)
		}
		return sampleOrderMinPKeepLogits(order, logits, params, temp, maxL, keep)
	}
	if params.TopP > 0 && params.TopP < 1 {
		target := params.TopP * probTotal
		for heapLen > 0 {
			id := popSampleOrderHeapLogits(order, logits, params, heapLen)
			heapLen--
			popped++
			keptMass += sampleVocabBF16IDWeight(logits, id, params, temp, maxL)
			if keptMass >= target {
				break
			}
		}
		reverseSampleOrderTailToPrefix(order, popped)
		return sampleOrderMinPKeepLogits(order, logits, params, temp, maxL, popped)
	}
	if params.MinP > 0 {
		id := popSampleOrderHeapLogits(order, logits, params, heapLen)
		heapLen--
		popped++
		threshold := sampleVocabBF16IDWeight(logits, id, params, temp, maxL) * params.MinP
		for heapLen > 0 && sampleVocabBF16IDWeight(logits, order[0], params, temp, maxL) >= threshold {
			popSampleOrderHeapLogits(order, logits, params, heapLen)
			heapLen--
			popped++
		}
		reverseSampleOrderTailToPrefix(order, popped)
		return popped
	}
	return len(order)
}

func sampleOrderTopPKeepLogits(order []int32, logits []byte, params model.SampleParams, temp, maxL float32, keep int, targetMass float32) int {
	var cum float32
	n := 0
	for n < keep {
		cum += sampleVocabBF16IDWeight(logits, order[n], params, temp, maxL)
		n++
		if cum >= targetMass {
			break
		}
	}
	return n
}

func sampleOrderMinPKeepLogits(order []int32, logits []byte, params model.SampleParams, temp, maxL float32, keep int) int {
	if params.MinP <= 0 || keep <= 0 {
		return keep
	}
	threshold := sampleVocabBF16IDWeight(logits, order[0], params, temp, maxL) * params.MinP
	n := 0
	for n < keep && sampleVocabBF16IDWeight(logits, order[n], params, temp, maxL) >= threshold {
		n++
	}
	if n > 0 {
		return n
	}
	return keep
}

func rankSampleOrderPrefixLogitsNoSuppress(order []int32, logits []byte, probTotal float32, params model.SampleParams, temp, maxL float32) int {
	if len(order) == 0 {
		return 0
	}
	if probTotal <= 0 {
		probTotal = 1
	}
	heapifySampleOrderLogitsNoSuppress(order, logits)
	heapLen := len(order)
	popped := 0
	keptMass := float32(0)
	if params.TopK > 0 && params.TopK < heapLen {
		for popped < params.TopK {
			id := popSampleOrderHeapLogitsNoSuppress(order, logits, heapLen)
			heapLen--
			popped++
			keptMass += sampleVocabBF16IDWeightNoSuppress(logits, id, temp, maxL)
		}
		reverseSampleOrderTailToPrefix(order, popped)
		keep := popped
		if params.TopP > 0 && params.TopP < 1 {
			keep = sampleOrderTopPKeepLogitsNoSuppress(order, logits, temp, maxL, keep, params.TopP*keptMass)
		}
		return sampleOrderMinPKeepLogitsNoSuppress(order, logits, temp, maxL, keep, params.MinP)
	}
	if params.TopP > 0 && params.TopP < 1 {
		target := params.TopP * probTotal
		for heapLen > 0 {
			id := popSampleOrderHeapLogitsNoSuppress(order, logits, heapLen)
			heapLen--
			popped++
			keptMass += sampleVocabBF16IDWeightNoSuppress(logits, id, temp, maxL)
			if keptMass >= target {
				break
			}
		}
		reverseSampleOrderTailToPrefix(order, popped)
		return sampleOrderMinPKeepLogitsNoSuppress(order, logits, temp, maxL, popped, params.MinP)
	}
	if params.MinP > 0 {
		id := popSampleOrderHeapLogitsNoSuppress(order, logits, heapLen)
		heapLen--
		popped++
		threshold := sampleVocabBF16IDWeightNoSuppress(logits, id, temp, maxL) * params.MinP
		for heapLen > 0 && sampleVocabBF16IDWeightNoSuppress(logits, order[0], temp, maxL) >= threshold {
			popSampleOrderHeapLogitsNoSuppress(order, logits, heapLen)
			heapLen--
			popped++
		}
		reverseSampleOrderTailToPrefix(order, popped)
		return popped
	}
	return len(order)
}

func sampleOrderTopPKeepLogitsNoSuppress(order []int32, logits []byte, temp, maxL float32, keep int, targetMass float32) int {
	var cum float32
	n := 0
	for n < keep {
		cum += sampleVocabBF16IDWeightNoSuppress(logits, order[n], temp, maxL)
		n++
		if cum >= targetMass {
			break
		}
	}
	return n
}

func sampleOrderMinPKeepLogitsNoSuppress(order []int32, logits []byte, temp, maxL float32, keep int, minP float32) int {
	if minP <= 0 || keep <= 0 {
		return keep
	}
	threshold := sampleVocabBF16IDWeightNoSuppress(logits, order[0], temp, maxL) * minP
	n := 0
	for n < keep && sampleVocabBF16IDWeightNoSuppress(logits, order[n], temp, maxL) >= threshold {
		n++
	}
	if n > 0 {
		return n
	}
	return keep
}

func heapifySampleOrderLogits(order []int32, logits []byte, params model.SampleParams) {
	for i := len(order)/2 - 1; i >= 0; i-- {
		siftSampleOrderHeapLogits(order, logits, params, i, len(order))
	}
}

func popSampleOrderHeapLogits(order []int32, logits []byte, params model.SampleParams, heapLen int) int32 {
	top := order[0]
	last := heapLen - 1
	order[0] = order[last]
	order[last] = top
	siftSampleOrderHeapLogits(order, logits, params, 0, last)
	return top
}

func siftSampleOrderHeapLogits(order []int32, logits []byte, params model.SampleParams, root, heapLen int) {
	for {
		child := root*2 + 1
		if child >= heapLen {
			return
		}
		if right := child + 1; right < heapLen && sampleOrderLogitsLess(order[right], order[child], logits, params) {
			child = right
		}
		if !sampleOrderLogitsLess(order[child], order[root], logits, params) {
			return
		}
		order[root], order[child] = order[child], order[root]
		root = child
	}
}

func sampleOrderLogitsLess(a, b int32, logits []byte, params model.SampleParams) bool {
	aSuppressed, bSuppressed := nativeTokenInSet(a, params.SuppressTokens), nativeTokenInSet(b, params.SuppressTokens)
	if aSuppressed || bSuppressed {
		if aSuppressed != bSuppressed {
			return !aSuppressed
		}
		return a < b
	}
	ai, bi := int(a)*bf16Size, int(b)*bf16Size
	av, bv := bf16ToF32(logits[ai], logits[ai+1]), bf16ToF32(logits[bi], logits[bi+1])
	return av > bv || (av == bv && a < b)
}

func heapifySampleOrderLogitsNoSuppress(order []int32, logits []byte) {
	for i := len(order)/2 - 1; i >= 0; i-- {
		siftSampleOrderHeapLogitsNoSuppress(order, logits, i, len(order))
	}
}

func popSampleOrderHeapLogitsNoSuppress(order []int32, logits []byte, heapLen int) int32 {
	top := order[0]
	last := heapLen - 1
	order[0] = order[last]
	order[last] = top
	siftSampleOrderHeapLogitsNoSuppress(order, logits, 0, last)
	return top
}

func siftSampleOrderHeapLogitsNoSuppress(order []int32, logits []byte, root, heapLen int) {
	for {
		child := root*2 + 1
		if child >= heapLen {
			return
		}
		if right := child + 1; right < heapLen && sampleOrderLogitsLessNoSuppress(order[right], order[child], logits) {
			child = right
		}
		if !sampleOrderLogitsLessNoSuppress(order[child], order[root], logits) {
			return
		}
		order[root], order[child] = order[child], order[root]
		root = child
	}
}

func sampleOrderLogitsLessNoSuppress(a, b int32, logits []byte) bool {
	ai, bi := int(a)*bf16Size, int(b)*bf16Size
	av, bv := bf16ToF32(logits[ai], logits[ai+1]), bf16ToF32(logits[bi], logits[bi+1])
	return av > bv || (av == bv && a < b)
}

func sampleRankPrefixPreferred(params model.SampleParams, vocab int) bool {
	if params.TopK > 0 && params.TopK < vocab {
		return true
	}
	if params.TopP > 0 && params.TopP < 1 {
		return true
	}
	return params.MinP > 0
}

func rankSampleOrderPrefix(order []int32, probs []float32, probTotal float32, params model.SampleParams) int {
	if len(order) == 0 {
		return 0
	}
	if probTotal <= 0 {
		probTotal = 1
	}
	heapifySampleOrder(order, probs)
	heapLen := len(order)
	popped := 0
	keptMass := float32(0)
	if params.TopK > 0 && params.TopK < heapLen {
		for popped < params.TopK {
			id := popSampleOrderHeap(order, probs, heapLen)
			heapLen--
			popped++
			keptMass += probs[id]
		}
		reverseSampleOrderTailToPrefix(order, popped)
		keep := popped
		if params.TopP > 0 && params.TopP < 1 {
			keep = sampleOrderTopPKeep(order, probs, keep, params.TopP*keptMass)
		}
		return sampleOrderMinPKeep(order, probs, keep, params.MinP)
	}
	if params.TopP > 0 && params.TopP < 1 {
		target := params.TopP * probTotal
		for heapLen > 0 {
			id := popSampleOrderHeap(order, probs, heapLen)
			heapLen--
			popped++
			keptMass += probs[id]
			if keptMass >= target {
				break
			}
		}
		reverseSampleOrderTailToPrefix(order, popped)
		return sampleOrderMinPKeep(order, probs, popped, params.MinP)
	}
	if params.MinP > 0 {
		id := popSampleOrderHeap(order, probs, heapLen)
		heapLen--
		popped++
		threshold := probs[id] * params.MinP
		for heapLen > 0 && probs[order[0]] >= threshold {
			popSampleOrderHeap(order, probs, heapLen)
			heapLen--
			popped++
		}
		reverseSampleOrderTailToPrefix(order, popped)
		return popped
	}
	sortSampleOrderByProb(order, probs)
	return len(order)
}

func sampleOrderTopPKeep(order []int32, probs []float32, keep int, targetMass float32) int {
	var cum float32
	n := 0
	for n < keep {
		cum += probs[int(order[n])]
		n++
		if cum >= targetMass {
			break
		}
	}
	return n
}

func sampleOrderMinPKeep(order []int32, probs []float32, keep int, minP float32) int {
	if minP <= 0 || keep <= 0 {
		return keep
	}
	threshold := probs[int(order[0])] * minP
	n := 0
	for n < keep && probs[int(order[n])] >= threshold {
		n++
	}
	if n > 0 {
		return n
	}
	return keep
}

func heapifySampleOrder(order []int32, probs []float32) {
	for i := len(order)/2 - 1; i >= 0; i-- {
		siftSampleOrderHeap(order, probs, i, len(order))
	}
}

func popSampleOrderHeap(order []int32, probs []float32, heapLen int) int32 {
	top := order[0]
	last := heapLen - 1
	order[0] = order[last]
	order[last] = top
	siftSampleOrderHeap(order, probs, 0, last)
	return top
}

func siftSampleOrderHeap(order []int32, probs []float32, root, heapLen int) {
	for {
		child := root*2 + 1
		if child >= heapLen {
			return
		}
		if right := child + 1; right < heapLen && sampleOrderLess(order[right], order[child], probs) {
			child = right
		}
		if !sampleOrderLess(order[child], order[root], probs) {
			return
		}
		order[root], order[child] = order[child], order[root]
		root = child
	}
}

func reverseSampleOrderTailToPrefix(order []int32, n int) {
	start := len(order) - n
	for i, j := start, len(order)-1; i < j; i, j = i+1, j-1 {
		order[i], order[j] = order[j], order[i]
	}
	if start > 0 {
		copy(order[:n], order[start:])
	}
}

func sortSampleOrderByProb(order []int32, probs []float32) {
	if len(order) < 2 {
		return
	}
	sortSampleOrderByProbRange(order, probs, 0, len(order)-1)
}

func sortSampleOrderByProbRange(order []int32, probs []float32, lo, hi int) {
	for hi-lo > 12 {
		mid := lo + (hi-lo)/2
		if sampleOrderLess(order[mid], order[lo], probs) {
			order[mid], order[lo] = order[lo], order[mid]
		}
		if sampleOrderLess(order[hi], order[mid], probs) {
			order[hi], order[mid] = order[mid], order[hi]
			if sampleOrderLess(order[mid], order[lo], probs) {
				order[mid], order[lo] = order[lo], order[mid]
			}
		}
		pivot := order[mid]
		i, j := lo, hi
		for {
			for sampleOrderLess(order[i], pivot, probs) {
				i++
			}
			for sampleOrderLess(pivot, order[j], probs) {
				j--
			}
			if i >= j {
				break
			}
			order[i], order[j] = order[j], order[i]
			i++
			j--
		}
		if j-lo < hi-i {
			sortSampleOrderByProbRange(order, probs, lo, j)
			lo = i
		} else {
			sortSampleOrderByProbRange(order, probs, i, hi)
			hi = j
		}
	}
	for i := lo + 1; i <= hi; i++ {
		v := order[i]
		j := i - 1
		for j >= lo && sampleOrderLess(v, order[j], probs) {
			order[j+1] = order[j]
			j--
		}
		order[j+1] = v
	}
}

func sampleOrderLess(a, b int32, probs []float32) bool {
	pa, pb := probs[int(a)], probs[int(b)]
	return pa > pb || (pa == pb && a < b)
}

func (s *ArchSession) sampleTopKParamsEligible(params model.SampleParams) bool {
	if s.headEnc == nil {
		return false
	}
	if params.TopK <= 0 || params.TopK > headSampleTopKMaxK {
		return false
	}
	// Large-vocab TopK selection is faster on the HOST (see head_host_topk.go):
	// the GPU selection kernels are value-dependent and lose by 4-20x at real
	// vocab sizes, so this lane declines and the ladder falls through to the
	// logits+host-select fallback.
	return !hostTopKSamplePreferred(params, s.arch.Vocab)
}

func (s *ArchSession) sampleTopKTokenParamsEligible(params model.SampleParams) bool {
	if s.headEnc == nil || params.Temperature <= 0 {
		return false
	}
	if params.TopK <= 0 || params.TopK > headSampleTopKMaxK {
		return false
	}
	if hostTopKSamplePreferred(params, s.arch.Vocab) { // see head_host_topk.go
		return false
	}
	return s.headEnc.topKSampleUsable(params.TopK)
}

func (s *ArchSession) sampleLogitsTokenParamsEligible(params model.SampleParams) bool {
	if s.headEnc == nil || params.Temperature <= 0 {
		return false
	}
	if params.TopK < 0 || params.TopK > headSampleTopKMaxK {
		return false
	}
	if params.TopK == 0 && params.TopP > 0 && params.TopP < 1 && !logitsSampleTopPOnlyFullVocab(params, s.arch.Vocab) {
		return false
	}
	if hostTopKSamplePreferred(params, s.arch.Vocab) { // TopK>1: the kernel's k-selection is the 33ms case; TopK==0 full-vocab stays (fast)
		return false
	}
	return s.headEnc.logitsSampleUsable()
}

func (s *ArchSession) retainedLogitsSampleParamsEligible(params model.SampleParams) bool {
	if s.headEnc == nil || params.Temperature <= 0 {
		return false
	}
	if params.TopK < 0 || params.TopK > headSampleTopKMaxK {
		return false
	}
	if params.TopK == 0 && params.TopP > 0 && params.TopP < 1 && !logitsSampleTopPOnlyFullVocab(params, s.arch.Vocab) {
		return false
	}
	if hostTopKSamplePreferred(params, s.arch.Vocab) { // TopK>1: kernel k-selection is the slow case (head_host_topk.go)
		return false
	}
	return s.headEnc.logitsBufferSampleUsable()
}

func sampleLogitsTokenCPUPreferred(params model.SampleParams, vocab int) bool {
	return params.TopK == 0 && params.TopP > 0 && params.TopP < 1 && params.RepeatPenalty <= 1 && vocab > 0 && vocab <= headSampleTopKMaxK
}

func logitsSampleTopPOnlyFullVocab(params model.SampleParams, vocab int) bool {
	return params.TopK == 0 && params.TopP > 0 && params.TopP < 1 && vocab > 0
}

func logitsSampleKernelTopK(params model.SampleParams, vocab int) int {
	if logitsSampleTopPOnlyFullVocab(params, vocab) {
		return vocab
	}
	return params.TopK
}

func sampledGreedyParamsEligible(params model.SampleParams) bool {
	return params.Temperature <= 0 && params.MinP <= 0 && params.RepeatPenalty <= 1
}

func sampledTopOneGreedyParamsEligible(params model.SampleParams, history []int32) bool {
	return params.TopK == 1 && !sampledGreedyParamsEligible(params) && (params.RepeatPenalty <= 1 || len(history) == 0)
}

// stepSampleTopKCandidatesInPool is the sampled sibling of stepGreedyInPool.
// For ICB sessions it decodes token id at the current cache row and runs the
// resident TopK head over the resulting hidden in the same command buffer. The
// host waits once, then reads this step's hidden plus only K candidate logits.
func (s *ArchSession) stepSampleTopKCandidatesInPool(id int32, params model.SampleParams) (hidden, logits []byte, ids []int32, ok bool, err error) {
	return s.stepSampleTopKCandidatesWithHistoryInPool(id, params, nil)
}

func (s *ArchSession) stepSampleTopKCandidatesWithHistoryInPool(id int32, params model.SampleParams, history []int32) (hidden, logits []byte, ids []int32, ok bool, err error) {
	if s.state.icb == nil || icbDisabledForTest || !s.sampleTopKParamsEligible(params) {
		return nil, nil, nil, false, nil
	}
	if s.encNextInputsGPU != nil && s.plScratchNew != nil && !chainedGPUInputsDisabled {
		return s.stepSampleTopKCandidatesGPUInputsWithHistoryInPool(id, params, history)
	}
	emb, err := s.embedID(id)
	if err != nil {
		return nil, nil, nil, false, err
	}
	var pli []byte
	if s.perLayerInput != nil {
		pli, err = s.perLayerInput(id, emb)
		if err != nil {
			return nil, nil, nil, false, err
		}
		s.state.perLayerInput = pli
	}
	icb := s.state.icb
	var scratch *headTopKScratch
	withAutoreleasePool(func() {
		cb := commandBufferFast(queue)
		enc := computeCommandEncoderFast(cb)
		var (
			lastOut      metal.MTLBuffer
			directHidden []byte
			directOut    bool
		)
		if pinned, pinnedOK := s.ensureRetainedHiddenPinned(s.arch.Hidden * bf16Size); pinnedOK && pinned.buf != nil {
			s.resetRetainedLogits()
			if out, ok := icb.encodeStepBodyIntoBuffer(enc, emb, s.pos, pli, pinned.buf); ok {
				lastOut = out
				directHidden = pinned.bytes[:s.arch.Hidden*bf16Size]
				directOut = true
			}
		}
		if !directOut {
			lastOut = icb.encodeStepBody(enc, emb, s.pos, pli)
		}
		scratch, ok, err = s.headEnc.encodeTopKCandidatesWithHistoryFast(enc, lastOut, params.TopK, params.SuppressTokens, history, params.RepeatPenalty)
		// Keep the encoder decline guard: sampleTopKParamsEligible does not prove
		// the backend-specific Q4/QMV/BF16 kernel shape used by this call.
		if !ok || err != nil {
			endEncodingFast(enc)
			if scratch != nil {
				s.headEnc.putTopKScratch(scratch)
				scratch = nil
			}
			return
		}
		endEncodingFast(enc)
		commitCommandBufferFast(cb)
		waitUntilCompletedFast(cb)
		if directOut {
			s.retainedHidden = directHidden
			hidden = directHidden
		} else {
			hidden = s.retainHiddenReadbackFrom(icb.lastOutPtr)
		}
		var readOK bool
		logits, ids, readOK, err = s.headEnc.readTopKCandidatesInto(scratch, params.TopK, s.sampleCandidateLogits, s.sampleCandidateIDs)
		s.sampleCandidateLogits, s.sampleCandidateIDs = logits, ids
		s.headEnc.putTopKScratch(scratch)
		scratch = nil
		ok = readOK
	})
	if err != nil || !ok {
		return nil, nil, nil, ok, err
	}
	s.pos++
	return hidden, logits, ids, true, nil
}

func (s *ArchSession) stepSampleTopKCandidatesGPUInputsInPool(id int32, params model.SampleParams) (hidden, logits []byte, ids []int32, ok bool, err error) {
	return s.stepSampleTopKCandidatesGPUInputsWithHistoryInPool(id, params, nil)
}

func (s *ArchSession) stepSampleTopKCandidatesGPUInputsWithHistoryInPool(id int32, params model.SampleParams, history []int32) (hidden, logits []byte, ids []int32, ok bool, err error) {
	icb := s.state.icb
	if icb == nil || s.encNextInputsGPU == nil || s.plScratchNew == nil {
		return nil, nil, nil, false, nil
	}
	var scratch *headTopKScratch
	withAutoreleasePool(func() {
		cb := commandBufferFast(queue)
		enc := computeCommandEncoderFast(cb)
		var (
			lastOut      metal.MTLBuffer
			directHidden []byte
			directOut    bool
		)
		if pinned, pinnedOK := s.ensureRetainedHiddenPinned(s.arch.Hidden * bf16Size); pinnedOK && pinned.buf != nil {
			s.resetRetainedLogits()
			var directOK bool
			lastOut, directOK, err = s.encodeStepBodyFromGPUInputsIntoBufferInPool(enc, id, pinned.buf)
			if err != nil {
				endEncodingFast(enc)
				return
			}
			if directOK {
				directHidden = pinned.bytes[:s.arch.Hidden*bf16Size]
				directOut = true
			}
		}
		if !directOut {
			lastOut, err = s.encodeStepBodyFromGPUInputsInPool(enc, id)
			if err != nil {
				endEncodingFast(enc)
				return
			}
		}
		scratch, ok, err = s.headEnc.encodeTopKCandidatesWithHistoryFast(enc, lastOut, params.TopK, params.SuppressTokens, history, params.RepeatPenalty)
		// Keep the encoder decline guard: this GPU-input helper is also reached by
		// a wrapper that does not establish the backend-specific kernel shape.
		if !ok || err != nil {
			endEncodingFast(enc)
			if scratch != nil {
				s.headEnc.putTopKScratch(scratch)
				scratch = nil
			}
			return
		}
		endEncodingFast(enc)
		commitCommandBufferFast(cb)
		waitUntilCompletedFast(cb)
		if directOut {
			s.retainedHidden = directHidden
			hidden = directHidden
		} else {
			hidden = s.retainHiddenReadbackFrom(icb.lastOutPtr)
		}
		var readOK bool
		logits, ids, readOK, err = s.headEnc.readTopKCandidatesInto(scratch, params.TopK, s.sampleCandidateLogits, s.sampleCandidateIDs)
		s.sampleCandidateLogits, s.sampleCandidateIDs = logits, ids
		s.headEnc.putTopKScratch(scratch)
		scratch = nil
		ok = readOK
	})
	if err != nil || !ok {
		return nil, nil, nil, ok, err
	}
	s.pos++
	return hidden, logits, ids, true, nil
}

func (s *ArchSession) stepSampleTopKTokenInPool(id int32, params model.SampleParams, draw float32, history []int32) (hidden []byte, token int32, ok bool, err error) {
	if s.state.icb == nil || icbDisabledForTest || !s.sampleTopKTokenParamsEligible(params) {
		return nil, 0, false, nil
	}
	if s.encNextInputsGPU != nil && s.plScratchNew != nil && !chainedGPUInputsDisabled {
		return s.stepSampleTopKTokenGPUInputsInPool(id, params, draw, history)
	}
	emb, err := s.embedID(id)
	if err != nil {
		return nil, 0, false, err
	}
	var pli []byte
	if s.perLayerInput != nil {
		pli, err = s.perLayerInput(id, emb)
		if err != nil {
			return nil, 0, false, err
		}
		s.state.perLayerInput = pli
	}
	icb := s.state.icb
	var scratch *headTopKScratch
	withAutoreleasePool(func() {
		cb := commandBufferFast(queue)
		enc := computeCommandEncoderFast(cb)
		var (
			lastOut      metal.MTLBuffer
			directHidden []byte
			directOut    bool
		)
		if pinned, pinnedOK := s.ensureRetainedHiddenPinned(s.arch.Hidden * bf16Size); pinnedOK && pinned.buf != nil {
			s.resetRetainedLogits()
			if out, ok := icb.encodeStepBodyIntoBuffer(enc, emb, s.pos, pli, pinned.buf); ok {
				lastOut = out
				directHidden = pinned.bytes[:s.arch.Hidden*bf16Size]
				directOut = true
			}
		}
		if !directOut {
			lastOut = icb.encodeStepBody(enc, emb, s.pos, pli)
		}
		scratch, ok, err = s.headEnc.encodeTopKSampleFast(enc, lastOut, params, draw, history)
		if !ok || err != nil {
			endEncodingFast(enc)
			if scratch != nil {
				s.headEnc.putTopKScratch(scratch)
				scratch = nil
			}
			return
		}
		endEncodingFast(enc)
		commitCommandBufferFast(cb)
		waitUntilCompletedFast(cb)
		if directOut {
			s.retainedHidden = directHidden
			hidden = directHidden
		} else {
			hidden = s.retainHiddenReadbackFrom(icb.lastOutPtr)
		}
		token = scratch.token()
		s.headEnc.putTopKScratch(scratch)
		scratch = nil
	})
	if err != nil || !ok {
		return nil, 0, ok, err
	}
	if token < 0 || int(token) >= s.arch.Vocab {
		return nil, 0, true, core.NewError(core.Sprintf("native.ArchSession.stepSampleTopKTokenInPool: sampled invalid token %d for vocab %d", token, s.arch.Vocab))
	}
	s.pos++
	return hidden, token, true, nil
}

func (s *ArchSession) encodeStepBodyFromGPUInputsInPool(enc metal.MTLComputeCommandEncoderObject, id int32) (metal.MTLBuffer, error) {
	icb := s.state.icb
	if icb == nil || s.encNextInputsGPU == nil || s.plScratchNew == nil {
		return nil, core.NewError("native.ArchSession.encodeStepBodyFromGPUInputsInPool: GPU inputs unavailable")
	}
	sc := s.gpuTailPLScratchBuffer(0)
	sc.out = icb.pleInput
	tokBuf := s.nextInputTokenBuffer(id)
	if err := s.encNextInputsGPU(enc, tokBuf, icb.ping0, sc); err != nil {
		return nil, err
	}
	memoryBarrierObject(enc, metal.MTLBarrierScopeBuffers)
	return icb.encodeStepBodyNoInput(enc, s.pos), nil
}

func (s *ArchSession) encodeStepBodyFromGPUInputsIntoBufferInPool(enc metal.MTLComputeCommandEncoderObject, id int32, out metal.MTLBuffer) (metal.MTLBuffer, bool, error) {
	icb := s.state.icb
	if icb == nil || s.encNextInputsGPU == nil || s.plScratchNew == nil {
		return nil, false, core.NewError("native.ArchSession.encodeStepBodyFromGPUInputsIntoBufferInPool: GPU inputs unavailable")
	}
	sc := s.gpuTailPLScratchBuffer(0)
	sc.out = icb.pleInput
	tokBuf := s.nextInputTokenBuffer(id)
	if err := s.encNextInputsGPU(enc, tokBuf, icb.ping0, sc); err != nil {
		return nil, false, err
	}
	memoryBarrierObject(enc, metal.MTLBarrierScopeBuffers)
	lastOut, ok := icb.encodeStepBodyNoInputIntoBuffer(enc, s.pos, out)
	return lastOut, ok, nil
}

func (s *ArchSession) encodeStepBodyNoInputRetained(enc metal.MTLComputeCommandEncoderObject, icb *archICBReplay, pos int) (metal.MTLBuffer, []byte) {
	if pinned, ok := s.ensureRetainedHiddenPinned(s.arch.Hidden * bf16Size); ok && pinned.buf != nil {
		s.resetRetainedLogits()
		if out, ok := icb.encodeStepBodyNoInputIntoBuffer(enc, pos, pinned.buf); ok {
			return out, pinned.bytes[:s.arch.Hidden*bf16Size]
		}
	}
	return icb.encodeStepBodyNoInput(enc, pos), nil
}

func (s *ArchSession) stepSampleTopKTokenGPUInputsInPool(id int32, params model.SampleParams, draw float32, history []int32) (hidden []byte, token int32, ok bool, err error) {
	icb := s.state.icb
	if icb == nil || s.encNextInputsGPU == nil || s.plScratchNew == nil {
		return nil, 0, false, nil
	}
	var scratch *headTopKScratch
	token = -1
	withAutoreleasePool(func() {
		cb := commandBufferFast(queue)
		enc := computeCommandEncoderFast(cb)
		var (
			lastOut      metal.MTLBuffer
			directHidden []byte
			directOut    bool
		)
		if pinned, pinnedOK := s.ensureRetainedHiddenPinned(s.arch.Hidden * bf16Size); pinnedOK && pinned.buf != nil {
			s.resetRetainedLogits()
			var directOK bool
			lastOut, directOK, err = s.encodeStepBodyFromGPUInputsIntoBufferInPool(enc, id, pinned.buf)
			if err != nil {
				endEncodingFast(enc)
				return
			}
			if directOK {
				directHidden = pinned.bytes[:s.arch.Hidden*bf16Size]
				directOut = true
			}
		}
		if !directOut {
			lastOut, err = s.encodeStepBodyFromGPUInputsInPool(enc, id)
			if err != nil {
				endEncodingFast(enc)
				return
			}
		}
		scratch, ok, err = s.headEnc.encodeTopKSampleFast(enc, lastOut, params, draw, history)
		if !ok || err != nil {
			endEncodingFast(enc)
			if scratch != nil {
				s.headEnc.putTopKScratch(scratch)
				scratch = nil
			}
			return
		}
		endEncodingFast(enc)
		commitCommandBufferFast(cb)
		waitUntilCompletedFast(cb)
		if directOut {
			s.retainedHidden = directHidden
			hidden = directHidden
		} else {
			hidden = s.retainHiddenReadbackFrom(icb.lastOutPtr)
		}
		token = scratch.token()
		s.headEnc.putTopKScratch(scratch)
		scratch = nil
	})
	if err != nil || !ok {
		return nil, 0, ok, err
	}
	if token < 0 || int(token) >= s.arch.Vocab {
		return nil, 0, true, core.NewError(core.Sprintf("native.ArchSession.stepSampleTopKTokenGPUInputsInPool: sampled invalid token %d for vocab %d", token, s.arch.Vocab))
	}
	s.pos++
	return hidden, token, true, nil
}

func (s *ArchSession) stepSampleLogitsTokenInPool(id int32, params model.SampleParams, draw float32, history []int32) (hidden []byte, token int32, ok bool, err error) {
	if s.state.icb == nil || icbDisabledForTest || !s.sampleLogitsTokenParamsEligible(params) {
		return nil, 0, false, nil
	}
	if s.encNextInputsGPU != nil && s.plScratchNew != nil && !chainedGPUInputsDisabled {
		return s.stepSampleLogitsTokenGPUInputsInPool(id, params, draw, history)
	}
	emb, err := s.embedID(id)
	if err != nil {
		return nil, 0, false, err
	}
	var pli []byte
	if s.perLayerInput != nil {
		pli, err = s.perLayerInput(id, emb)
		if err != nil {
			return nil, 0, false, err
		}
		s.state.perLayerInput = pli
	}
	icb := s.state.icb
	var scratch *headGreedyScratch
	withAutoreleasePool(func() {
		cb := commandBufferFast(queue)
		enc := computeCommandEncoderFast(cb)
		var (
			lastOut      metal.MTLBuffer
			directHidden []byte
			directOut    bool
		)
		if pinned, pinnedOK := s.ensureRetainedHiddenPinned(s.arch.Hidden * bf16Size); pinnedOK && pinned.buf != nil {
			s.resetRetainedLogits()
			if out, ok := icb.encodeStepBodyIntoBuffer(enc, emb, s.pos, pli, pinned.buf); ok {
				lastOut = out
				directHidden = pinned.bytes[:s.arch.Hidden*bf16Size]
				directOut = true
			}
		}
		if !directOut {
			lastOut = icb.encodeStepBody(enc, emb, s.pos, pli)
		}
		scratch, ok, err = s.headEnc.encodeLogitsSample(enc, lastOut, params, draw, history)
		if err != nil {
			endEncodingFast(enc)
			if scratch != nil {
				s.headEnc.putGreedyScratch(scratch)
				scratch = nil
			}
			return
		}
		endEncodingFast(enc)
		commitCommandBufferFast(cb)
		waitUntilCompletedFast(cb)
		if directOut {
			s.retainedHidden = directHidden
			hidden = directHidden
		} else {
			hidden = s.retainHiddenReadbackFrom(icb.lastOutPtr)
		}
		token = scratch.token()
		s.headEnc.putGreedyScratch(scratch)
		scratch = nil
	})
	if err != nil {
		return nil, 0, ok, err
	}
	if token < 0 || int(token) >= s.arch.Vocab {
		return nil, 0, true, core.NewError(core.Sprintf("native.ArchSession.stepSampleLogitsTokenInPool: sampled invalid token %d for vocab %d", token, s.arch.Vocab))
	}
	s.pos++
	return hidden, token, true, nil
}

func (s *ArchSession) stepSampleLogitsTokenGPUInputsInPool(id int32, params model.SampleParams, draw float32, history []int32) (hidden []byte, token int32, ok bool, err error) {
	icb := s.state.icb
	if icb == nil || s.encNextInputsGPU == nil || s.plScratchNew == nil {
		return nil, 0, false, nil
	}
	var scratch *headGreedyScratch
	token = -1
	withAutoreleasePool(func() {
		cb := commandBufferFast(queue)
		enc := computeCommandEncoderFast(cb)
		var (
			lastOut      metal.MTLBuffer
			directHidden []byte
			directOut    bool
		)
		if pinned, pinnedOK := s.ensureRetainedHiddenPinned(s.arch.Hidden * bf16Size); pinnedOK && pinned.buf != nil {
			s.resetRetainedLogits()
			var directOK bool
			lastOut, directOK, err = s.encodeStepBodyFromGPUInputsIntoBufferInPool(enc, id, pinned.buf)
			if err != nil {
				endEncodingFast(enc)
				return
			}
			if directOK {
				directHidden = pinned.bytes[:s.arch.Hidden*bf16Size]
				directOut = true
			}
		}
		if !directOut {
			lastOut, err = s.encodeStepBodyFromGPUInputsInPool(enc, id)
			if err != nil {
				endEncodingFast(enc)
				return
			}
		}
		scratch, ok, err = s.headEnc.encodeLogitsSample(enc, lastOut, params, draw, history)
		if err != nil {
			endEncodingFast(enc)
			if scratch != nil {
				s.headEnc.putGreedyScratch(scratch)
				scratch = nil
			}
			return
		}
		endEncodingFast(enc)
		commitCommandBufferFast(cb)
		waitUntilCompletedFast(cb)
		if directOut {
			s.retainedHidden = directHidden
			hidden = directHidden
		} else {
			hidden = s.retainHiddenReadbackFrom(icb.lastOutPtr)
		}
		token = scratch.token()
		s.headEnc.putGreedyScratch(scratch)
		scratch = nil
	})
	if err != nil {
		return nil, 0, ok, err
	}
	if token < 0 || int(token) >= s.arch.Vocab {
		return nil, 0, true, core.NewError(core.Sprintf("native.ArchSession.stepSampleLogitsTokenGPUInputsInPool: sampled invalid token %d for vocab %d", token, s.arch.Vocab))
	}
	s.pos++
	return hidden, token, true, nil
}

func (s *ArchSession) stepGreedyInPool(id int32, emb []byte, suppress []int32) (token int32, hidden []byte, ok bool, err error) {
	if s.state.icb == nil || icbDisabledForTest || s.headEnc == nil {
		return 0, nil, false, nil
	}
	if emb == nil {
		emb, err = s.embedID(id)
		if err != nil {
			return 0, nil, false, err
		}
	}
	icb := s.state.icb
	var pli []byte
	if s.perLayerInput != nil {
		pli, err = s.perLayerInput(id, emb)
		if err != nil {
			return 0, nil, false, err
		}
		s.state.perLayerInput = pli
	}
	token = -1
	withAutoreleasePool(func() {
		cb := commandBufferFast(queue)
		enc := computeCommandEncoderFast(cb)
		var (
			lastOut      metal.MTLBuffer
			directHidden []byte
			directOut    bool
		)
		if pinned, pinnedOK := s.ensureRetainedHiddenPinned(s.arch.Hidden * bf16Size); pinnedOK && pinned.buf != nil {
			s.resetRetainedLogits()
			if out, ok := icb.encodeStepBodyIntoBuffer(enc, emb, s.pos, pli, pinned.buf); ok {
				lastOut = out
				directHidden = pinned.bytes[:s.arch.Hidden*bf16Size]
				directOut = true
			}
		}
		if !directOut {
			lastOut = icb.encodeStepBody(enc, emb, s.pos, pli)
		}
		scratch, gok, gerr := s.headEnc.encodeGreedy(enc, lastOut, suppress)
		if !gok || gerr != nil {
			endEncodingFast(enc)
			if scratch != nil {
				s.headEnc.putGreedyScratch(scratch)
			}
			ok, err = gok, gerr
			return
		}
		endEncodingFast(enc)
		commitCommandBufferFast(cb)
		waitUntilCompletedFast(cb)
		token = scratch.token()
		if directOut {
			s.retainedHidden = directHidden
			hidden = directHidden
		} else {
			hidden = s.retainHiddenReadbackFrom(icb.lastOutPtr)
		}
		s.headEnc.putGreedyScratch(scratch)
		ok = true
	})
	if err != nil || !ok {
		return 0, nil, ok, err
	}
	s.pos++
	if token < 0 || int(token) >= s.arch.Vocab {
		return 0, nil, true, core.NewError("native.ArchSession.stepGreedyInPool: invalid token")
	}
	return token, hidden, true, nil
}

// stepGreedyLiveInPool is stepGreedyInPool for the LIVE-ENCODER (non-ICB) decode: stepToken's
// chainTail hook encodes the head argmax + the next token's input production (encNextInputsGPU
// into xA) into the step's OWN command buffer — one cb and one wait per token, no host embed or
// argmax between. ok=false with hidden non-nil means the chain declined this token (cb break /
// head decline): the step still ran and cached the token, and hidden carries the valid host
// readback for a serial head fallback. When chained, the host readback is CLOBBERED by the
// next-input write (see the chainTail hook) and is not returned.
func (s *ArchSession) stepGreedyLiveInPool(id int32, emb []byte, copyInput bool, suppress []int32, sc *plGPUScratch) (token int32, hidden []byte, ok bool, err error) {
	if copyInput && emb == nil {
		if emb, err = s.embedID(id); err != nil {
			return 0, nil, false, err
		}
	}
	var scratch *headGreedyScratch
	chained := false
	s.state.chainTail = func(enc metal.MTLComputeCommandEncoderObject, hiddenBuf metal.MTLBuffer) error {
		sc2, gok, gerr := s.headEnc.encodeGreedy(enc, hiddenBuf, suppress)
		if gerr != nil {
			return gerr
		}
		if !gok {
			return nil // head declined: the step completes normally, the caller falls back
		}
		if nerr := s.encNextInputsGPU(enc, sc2.outToken, s.state.xA, sc); nerr != nil {
			s.headEnc.putGreedyScratch(sc2)
			return nerr
		}
		scratch = sc2
		chained = true
		return nil
	}
	var res []byte
	withAutoreleasePool(func() {
		res, err = s.state.stepTokenResultWithInputInto(emb, s.pos, true, copyInput, nil)
	})
	s.state.chainTail = nil
	if err != nil {
		if scratch != nil {
			s.headEnc.putGreedyScratch(scratch)
		}
		return 0, nil, false, err
	}
	s.pos++
	if !chained {
		return 0, res, false, nil
	}
	token = scratch.token()
	s.headEnc.putGreedyScratch(scratch)
	if token < 0 || int(token) >= s.arch.Vocab {
		return 0, nil, true, core.NewError("native.ArchSession.stepGreedyLiveInPool: invalid token")
	}
	chainedLiveLinks.Add(1)
	return token, nil, true, nil
}

// chainedLiveLinks counts successfully chained live-decode links — the engagement receipt
// (a gate regression that silently drops the lane shows up as zero here, not as a perf blur).
var chainedLiveLinks atomic.Int64

// generateChainedLiveTail runs the chained live decode for archs the ICB cannot record (MoE):
// each token is ONE command buffer — stepToken + head argmax + the next token's embed gather —
// with the wait its only sync and 4 bytes its only readback. Once a probe link chains, the
// SUBMIT-AHEAD inner loop takes over (one speculative command buffer in flight, host encode
// overlapping the GPU). A declined link finishes that token with the serial head and re-primes
// the chain with a host embed.
func (s *ArchSession) generateChainedLiveTail(gen []int32, maxNew, eosID int, suppress []int32, yield func(int32) bool, stop bool) ([]int32, error) {
	sc := s.plScratchNew()
	emb, err := s.embedID(gen[len(gen)-1])
	if err != nil {
		return nil, err
	}
	copyInput := true
	// Submit-ahead unwinds cleanly on EVERY paged shape. Linear caches truncate by rewind;
	// ring caches are safe for the ONE-token speculation too: the speculative write lands in
	// exactly the slot the position's redo (or the next real token) rewrites BEFORE its own
	// SDPA reads it, the row it clobbered sits outside every future window, and the ring
	// truncate is a pure counter rewind (devicePagedKVCache.truncate).
	submitAhead := !liveSubmitAheadDisabled
	for !stop && len(gen) < maxNew {
		prev := gen[len(gen)-1]
		tok, hidden, ok, serr := s.stepGreedyLiveInPool(prev, emb, copyInput, suppress, sc)
		if serr != nil {
			return nil, serr
		}
		if !ok {
			// The step ran (prev is cached, position advanced) but the head never encoded:
			// argmax the returned hidden serially and hand the next link a host embed.
			if tok, serr = s.headGreedyOrLogits(hidden, suppress, nil, nil, false); serr != nil {
				return nil, serr
			}
			s.rememberRetainedHidden(hidden)
			if emb, serr = s.embedID(tok); serr != nil {
				return nil, serr
			}
			copyInput = true
			gen = append(gen, tok)
			stop = (yield != nil && !yield(tok)) || (eosID >= 0 && int(tok) == eosID)
			continue
		}
		emb, copyInput = nil, false
		gen = append(gen, tok)
		stop = (yield != nil && !yield(tok)) || (eosID >= 0 && int(tok) == eosID)
		if submitAhead && !stop && len(gen) < maxNew {
			// The probe link chained, so xA holds the next input on-GPU — enter the
			// pipelined steady state. It exits only at stop or the token budget.
			if gen, stop, serr = s.generateSubmitAheadLinks(gen, maxNew, eosID, suppress, sc, yield); serr != nil {
				return nil, serr
			}
		}
	}
	// Cache the last produced token (each chain link steps prev, not the freshly produced
	// token) and retain its hidden as the session boundary — the serial loop's semantics.
	hidden, err := s.stepIDRetainedInPool(gen[len(gen)-1])
	if err != nil {
		return nil, err
	}
	s.rememberRetainedHidden(hidden)
	return gen, nil
}

// pendingLiveLink is one submit-ahead chained link in flight: the committed command buffer
// and the head scratch whose token it will produce.
type pendingLiveLink struct {
	cb      metal.MTLCommandBufferObject
	scratch *headGreedyScratch
}

func (s *ArchSession) discardLiveLink(link pendingLiveLink) {
	waitUntilCompletedFast(link.cb)
	s.headEnc.putGreedyScratch(link.scratch)
}

// liveSubmitAheadDepth is how many speculative chained links stay committed while the
// oldest executes. Depth 1 by receipt (#341 phase 4): with the correct encode-before-
// wait ordering, depth 2 measured 154.2 tok/s vs depth 1's 152.9 on the real 26B —
// inside the run-to-run wobble band — with the host/sync gap pinned at 0.18 ms/token
// either way. One queued link already lets the GPU start the next command buffer the
// instant the current completes; the residual gap is cb-to-cb scheduling latency that
// no queue depth removes, while every extra depth discards one more speculative
// token's GPU work at a stop. Greedy chaining keeps any depth correct-by-construction
// (a stop is a pure position/counter rewind, ring-safe — the identity tests cover the
// wrapped-ring rollback shape), so this stays a documented tunable.
const liveSubmitAheadDepth = 1

// generateSubmitAheadLinks runs the chained live decode with up to liveSubmitAheadDepth
// speculative links in flight: each link's command buffer commits while older links
// execute — its input is the previous link's on-GPU embed, so no host value is needed —
// and the host encode overlaps the GPU. A stop with speculative links in flight unwinds
// them (wait, position back by the queue length, one paged-KV truncate). A declined link
// here is a structural fault, not a fallback case: its tail never wrote xA, so a further
// link would compute on stale input.
func (s *ArchSession) generateSubmitAheadLinks(gen []int32, maxNew, eosID int, suppress []int32, sc *plGPUScratch, yield func(int32) bool) ([]int32, bool, error) {
	queue := make([]pendingLiveLink, 0, liveSubmitAheadDepth)
	discardQueue := func() {
		for _, link := range queue {
			s.discardLiveLink(link)
		}
		queue = queue[:0]
	}
	push := func() error {
		link, ok, err := s.stepGreedyLiveCommit(suppress, sc)
		if err != nil {
			discardQueue()
			return err
		}
		if !ok {
			discardQueue()
			return core.NewError("native.ArchSession.generateSubmitAheadLinks: chain declined mid-stream")
		}
		queue = append(queue, link)
		return nil
	}
	for len(queue) < liveSubmitAheadDepth && len(gen)+len(queue) < maxNew {
		if err := push(); err != nil {
			return gen, false, err
		}
	}
	for len(queue) > 0 {
		// Commit the NEXT link before waiting the oldest: the host encode overlaps
		// the queued links' GPU execution (waiting first would idle the GPU through
		// every encode — measured 132.5 vs 155.0 tok/s on the real 26B).
		if len(gen)+len(queue) < maxNew {
			if err := push(); err != nil {
				return gen, false, err
			}
		}
		pending := queue[0]
		queue = queue[1:]
		tok, werr := s.waitLiveLink(pending)
		if werr != nil {
			discardQueue()
			return gen, false, werr
		}
		gen = append(gen, tok)
		stop := (yield != nil && !yield(tok)) || (eosID >= 0 && int(tok) == eosID)
		if stop {
			if n := len(queue); n > 0 {
				// unwind the speculative links: their steps cached tokens past the stop.
				discardQueue()
				s.pos -= n
				if terr := s.truncateSpeculativeKV(s.pos); terr != nil {
					return gen, true, terr
				}
			}
			return gen, true, nil
		}
	}
	return gen, false, nil // token budget reached
}

// stepGreedyLiveCommit encodes + COMMITS one chained live link without waiting (the
// submit-ahead pipeline's producer): input comes from the previous link's on-GPU embed
// (copyInput=false), so no host token value is needed. ok=false means the chain declined —
// the step still ran and cached its token, and the command buffer has been waited.
func (s *ArchSession) stepGreedyLiveCommit(suppress []int32, sc *plGPUScratch) (pendingLiveLink, bool, error) {
	var scratch *headGreedyScratch
	chained := false
	s.state.chainTail = func(enc metal.MTLComputeCommandEncoderObject, hiddenBuf metal.MTLBuffer) error {
		sc2, gok, gerr := s.headEnc.encodeGreedy(enc, hiddenBuf, suppress)
		if gerr != nil {
			return gerr
		}
		if !gok {
			return nil
		}
		if nerr := s.encNextInputsGPU(enc, sc2.outToken, s.state.xA, sc); nerr != nil {
			s.headEnc.putGreedyScratch(sc2)
			return nerr
		}
		scratch = sc2
		chained = true
		return nil
	}
	s.state.chainSkipWait = true
	s.state.chainPendingCB = metal.MTLCommandBufferObject{}
	// NO autorelease pool here: the command buffer is autoreleased, and a pool scoped to this
	// call would drain and FREE it before the caller's wait (the committed cb then hangs the
	// wait forever). The generate loop's outer pool owns the lifetime; the wait happens well
	// inside it.
	_, err := s.state.stepTokenResultWithInputInto(nil, s.pos, false, false, nil)
	s.state.chainTail = nil
	s.state.chainSkipWait = false
	cb := s.state.chainPendingCB
	s.state.chainPendingCB = metal.MTLCommandBufferObject{}
	if err != nil {
		if cb.GetID() != 0 {
			waitUntilCompletedFast(cb) // never leave a committed cb dangling
		}
		if scratch != nil {
			s.headEnc.putGreedyScratch(scratch)
		}
		return pendingLiveLink{}, false, err
	}
	s.pos++
	if !chained || cb.GetID() == 0 {
		if cb.GetID() != 0 {
			waitUntilCompletedFast(cb)
		}
		if scratch != nil {
			s.headEnc.putGreedyScratch(scratch)
		}
		return pendingLiveLink{}, false, nil
	}
	chainedLiveLinks.Add(1)
	return pendingLiveLink{cb: cb, scratch: scratch}, true, nil
}

// waitLiveLink completes one submit-ahead link: wait, read the 4-byte token, recycle the
// head scratch.
func (s *ArchSession) waitLiveLink(link pendingLiveLink) (int32, error) {
	waitUntilCompletedFast(link.cb)
	if pieceTimingOn {
		chainedGPUSpanNs += int64(float64(link.cb.GPUEndTime()-link.cb.GPUStartTime()) * 1e9)
	}
	token := link.scratch.token()
	s.headEnc.putGreedyScratch(link.scratch)
	if token < 0 || int(token) >= s.arch.Vocab {
		return 0, core.NewError("native.ArchSession.waitLiveLink: invalid token")
	}
	return token, nil
}

// headGreedyOrLogits argmaxes the next token from `hidden`: the GPU direct-argmax head when available,
// else the logits path (with the first-token firstLogits/cacheFirstLogits boundary honoured when isFirst).
func (s *ArchSession) headGreedyOrLogits(hidden []byte, suppress []int32, firstLogits []byte, cacheFirstLogits func([]byte), isFirst bool) (int32, error) {
	if !(isFirst && (firstLogits != nil || cacheFirstLogits != nil)) && s.greedy != nil {
		_ptHead := ptStart()
		next, ok, err := s.directGreedyFromHiddenInPool(hidden, suppress)
		ptEnd(2, _ptHead)
		if err != nil {
			return 0, err
		}
		if ok {
			return next, nil
		}
	}
	var logits []byte
	var err error
	if isFirst && firstLogits != nil {
		logits = firstLogits
	} else {
		_ptHead := ptStart()
		// cacheFirstLogits retains this slice for prompt replay, so keep that path on
		// the owned logits backing. Other greedy fallback calls consume logits
		// immediately and can reuse the session scratch.
		if isFirst && cacheFirstLogits != nil {
			logits, err = s.head(hidden, true) // greedy: argmax — skip the monotonic softcap (token-identical)
		} else if s.canUseHeadLogitsScratch() {
			logits, err = s.headLogitsScratch(hidden, true)
		} else {
			logits, err = s.head(hidden, true)
		}
		ptEnd(2, _ptHead)
		if err != nil {
			return 0, err
		}
	}
	if isFirst && cacheFirstLogits != nil {
		cacheFirstLogits(logits)
	}
	return greedyBF16Suppressed(logits, s.arch.Vocab, suppress)
}

func (s *ArchSession) generateFromHiddenInPool(hidden []byte, maxNew, eosID int, firstLogits []byte, cacheFirstLogits func([]byte), suppress []int32, transform TokenTransform, yield func(int32) bool) ([]int32, error) {
	gen := make([]int32, 0, maxNew)
	// First token: head+argmax on the prefill/retained hidden (no step yet — the chain caches each token
	// via the NEXT step, and a final step caches the last one).
	next, err := s.headGreedyOrLogits(hidden, suppress, firstLogits, cacheFirstLogits, true)
	if err != nil {
		return nil, err
	}
	if transform != nil {
		next = transform(next)
	}
	gen = append(gen, next)
	stop := (yield != nil && !yield(next)) || (eosID >= 0 && int(next) == eosID)

	// Chained-GPU decode (e2b): the prior step produces the next step's emb+pli on-GPU (encNextInputsGPU
	// appended to the step's command buffer), so each token is ONE command buffer with no host embed/PLE.
	// transform would change the token after the GPU already embedded it, so only when transform == nil.
	// The chained/pipelined tails consume the GPU argmax head MID-CHAIN (the next step's input
	// binds the head's token buffer with no host round-trip) — engaging without it fails at the
	// first link. The head requirement is the head's own capability, not the input seam's: the
	// old bits==4 seam gate masked this until the gather widened to every affine width.
	if s.encNextInputsGPU != nil && s.plScratchNew != nil && s.state.icb != nil && s.headEnc != nil && s.greedy != nil &&
		s.canUseDirectHeadGreedy() &&
		!stepGreedyChainDisabled && !chainedGPUInputsDisabled && !icbDisabledForTest && transform == nil {
		if pipelinedGPUDecodeEnabled && s.recordPeerICB != nil {
			return s.generatePipelinedGPUTail(gen, maxNew, eosID, suppress, yield, stop)
		}
		return s.generateChainedGPUTail(gen, maxNew, eosID, suppress, yield, stop)
	}

	// Chained live decode: the archs the ICB cannot record (MoE) chain step + head argmax +
	// next-embed through stepToken's chainTail hook instead — the same one-cb-per-token shape,
	// re-encoded live each token. PLE archs need the per-layer tensor produced per token too
	// (the ICB chain's plumbing), so they stay on their existing lanes.
	if s.encNextInputsGPU != nil && s.plScratchNew != nil && s.state.icb == nil && s.headEnc != nil &&
		s.canUseDirectHeadGreedy() && len(s.state.ple) == 0 &&
		!stepGreedyChainDisabled && !chainedGPUInputsDisabled && transform == nil &&
		s.state.gpuProf == nil && !s.state.trace && layerSpanProbeForTest == nil && !captureLayerHiddens {
		return s.generateChainedLiveTail(gen, maxNew, eosID, suppress, yield, stop)
	}

	for !stop && len(gen) < maxNew {
		prev := gen[len(gen)-1]
		emb, eerr := s.embedID(prev)
		if eerr != nil {
			return nil, eerr
		}
		var n2 int32
		// Chain prev's stepBody with this token's head+argmax in ONE command buffer (one sync/token).
		if !stepGreedyChainDisabled {
			_ptH := ptStart()
			tok, h, ok, serr := s.stepGreedyInPool(prev, emb, suppress)
			ptEnd(2, _ptH)
			if serr != nil {
				return nil, serr
			}
			if ok {
				n2, hidden = tok, h
				goto produced
			}
		}
		// Serial fallback: step prev (cache it), then head on the new hidden.
		if hidden, err = s.stepIDRetainedInPool(prev); err != nil {
			return nil, err
		}
		if n2, err = s.headGreedyOrLogits(hidden, suppress, nil, nil, false); err != nil {
			return nil, err
		}
	produced:
		if transform != nil {
			n2 = transform(n2)
		}
		gen = append(gen, n2)
		s.rememberRetainedHidden(hidden)
		stop = (yield != nil && !yield(n2)) || (eosID >= 0 && int(n2) == eosID)
	}
	// Cache the last produced token (the chain steps prev, not the freshly produced token), so the session
	// state matches the serial loop (every generated token cached) for reuse / a second turn.
	if hidden, err = s.stepIDRetainedInPool(gen[len(gen)-1]); err != nil {
		return nil, err
	}
	s.rememberRetainedHidden(hidden)
	return gen, nil
}

// generateChainedGPUTail decodes from the first token `gen[0]` with the GPU next-inputs seam: each token's
// command buffer replays the layer stack (reading the prior step's GPU-produced emb+pli from the ICB's
// ping0/pleInput), argmaxes the head, then runs encNextInputsGPU on the GPU head output to seed THIS step's
// emb+pli for the next — no host embed/PLE round-trip. Cache/pos bookkeeping matches the serial loop: each
// step caches the token whose emb is in ping0; a final no-input step caches the last produced token (so
// session reuse / second turn is byte-identical). `stop` is the first token's stop verdict from the caller.
func (s *ArchSession) generateChainedGPUTail(gen []int32, maxNew, eosID int, suppress []int32, yield func(int32) bool, stop bool) ([]int32, error) {
	icb := s.state.icb
	sc := s.gpuTailPLScratchBuffer(0)
	sc.out = icb.pleInput // the PLE result lands directly in the ICB's pli input for the next step
	var rerr error
	withAutoreleasePool(func() {
		// Seed: produce emb(gen[last])/pli(gen[last]) into ping0/pleInput from the first token.
		tokBuf := s.nextInputTokenBuffer(gen[len(gen)-1])
		seedCB := commandBufferFast(queue)
		seedEnc := computeCommandEncoderFast(seedCB)
		if e := s.encNextInputsGPU(seedEnc, tokBuf, icb.ping0, sc); e != nil {
			endEncodingFast(seedEnc)
			rerr = e
			return
		}
		endEncodingFast(seedEnc)
		commitCommandBufferFast(seedCB)
		waitUntilCompletedFast(seedCB)

		for !stop && len(gen) < maxNew {
			cb := commandBufferFast(queue)
			enc := computeCommandEncoderFast(cb)
			lastOut := icb.encodeStepBodyNoInput(enc, s.pos) // caches the token in ping0 (gen[last]) at s.pos
			scratch, gok, gerr := s.headEnc.encodeGreedy(enc, lastOut, suppress)
			if !gok || gerr != nil {
				endEncodingFast(enc)
				if scratch != nil {
					s.headEnc.putGreedyScratch(scratch)
				}
				if rerr = gerr; rerr == nil {
					rerr = core.NewError("native.ArchSession.generateChainedGPUTail: GPU head argmax unavailable mid-chain")
				}
				return
			}
			// Produce THIS token's emb+pli on-GPU (into ping0/pleInput) for the NEXT step. Within the
			// encoder the stepBody read of ping0/pleInput is ordered before this write (serial dispatch).
			s.encNextInputsGPU(enc, scratch.outToken, icb.ping0, sc)
			endEncodingFast(enc)
			commitCommandBufferFast(cb)
			waitUntilCompletedFast(cb)
			if pieceTimingOn {
				chainedGPUSpanNs += int64(float64(cb.GPUEndTime()-cb.GPUStartTime()) * 1e9)
			}
			tk := scratch.token()
			s.headEnc.putGreedyScratch(scratch)
			s.pos++
			if tk < 0 || int(tk) >= s.arch.Vocab {
				rerr = core.NewError("native.ArchSession.generateChainedGPUTail: invalid token")
				return
			}
			gen = append(gen, tk)
			stop = (yield != nil && !yield(tk)) || (eosID >= 0 && int(tk) == eosID)
		}

		// Cache the last produced token (its emb is in ping0 but stepBody hasn't run), matching the serial
		// loop's final stepID, and retain that hidden as the session boundary.
		cb := commandBufferFast(queue)
		enc := computeCommandEncoderFast(cb)
		_, directHidden := s.encodeStepBodyNoInputRetained(enc, icb, s.pos)
		endEncodingFast(enc)
		commitCommandBufferFast(cb)
		waitUntilCompletedFast(cb)
		s.pos++
		if directHidden != nil {
			s.retainedHidden = directHidden
		} else {
			s.rememberRetainedHiddenFrom(icb.lastOutPtr)
		}
	})
	if rerr != nil {
		return nil, rerr
	}
	return gen, nil
}

// generateChainedGPUOneShotTail is the one-shot sibling of generateChainedGPUTail. It uses the
// same GPU next-input seam for generated tokens after the first, but intentionally skips the final
// no-input cache step because GenerateOneShot closes the session boundary after returning tokens.
func (s *ArchSession) generateChainedGPUOneShotTail(gen []int32, maxNew, eosID int, stop bool) ([]int32, error) {
	if len(gen) == 0 {
		return gen, core.NewError("native.ArchSession.generateChainedGPUOneShotTail: empty generation seed")
	}
	if !stop && eosID < 0 && pipelinedGPUDecodeEnabled && s.recordPeerICB != nil {
		return s.generatePipelinedGPUOneShotTail(gen, maxNew)
	}
	icb := s.state.icb
	sc := s.gpuTailPLScratchBuffer(0)
	sc.out = icb.pleInput
	var rerr error
	withAutoreleasePool(func() {
		tokBuf := s.nextInputTokenBuffer(gen[len(gen)-1])
		seedCB := commandBufferFast(queue)
		seedEnc := computeCommandEncoderFast(seedCB)
		if e := s.encNextInputsGPU(seedEnc, tokBuf, icb.ping0, sc); e != nil {
			endEncodingFast(seedEnc)
			rerr = e
			return
		}
		endEncodingFast(seedEnc)
		commitCommandBufferFast(seedCB)
		waitUntilCompletedFast(seedCB)

		for !stop && len(gen) < maxNew {
			cb := commandBufferFast(queue)
			enc := computeCommandEncoderFast(cb)
			lastOut := icb.encodeStepBodyNoInput(enc, s.pos)
			scratch, gok, gerr := s.headEnc.encodeGreedy(enc, lastOut, nil)
			if !gok || gerr != nil {
				endEncodingFast(enc)
				if scratch != nil {
					s.headEnc.putGreedyScratch(scratch)
				}
				if rerr = gerr; rerr == nil {
					rerr = core.NewError("native.ArchSession.generateChainedGPUOneShotTail: GPU head argmax unavailable mid-chain")
				}
				return
			}
			if e := s.encNextInputsGPU(enc, scratch.outToken, icb.ping0, sc); e != nil {
				endEncodingFast(enc)
				s.headEnc.putGreedyScratch(scratch)
				rerr = e
				return
			}
			endEncodingFast(enc)
			commitCommandBufferFast(cb)
			waitUntilCompletedFast(cb)
			if pieceTimingOn {
				chainedGPUSpanNs += int64(float64(cb.GPUEndTime()-cb.GPUStartTime()) * 1e9)
			}
			tk := scratch.token()
			s.headEnc.putGreedyScratch(scratch)
			s.pos++
			if tk < 0 || int(tk) >= s.arch.Vocab {
				rerr = core.NewError("native.ArchSession.generateChainedGPUOneShotTail: invalid token")
				return
			}
			gen = append(gen, tk)
			stop = eosID >= 0 && int(tk) == eosID
		}
	})
	if rerr != nil {
		return nil, rerr
	}
	return gen, nil
}

// generatePipelinedGPUOneShotTail is the submit-ahead one-shot decode path. It keeps one command
// buffer in flight ahead while the generated token is known not to be final by budget, then drains
// the last needed step without submitting a final cache step. EOS-aware calls stay on the synchronous
// one-shot tail so a stop token is not speculatively cached before the host can observe it.
func (s *ArchSession) generatePipelinedGPUOneShotTail(gen []int32, maxNew int) ([]int32, error) {
	if len(gen) == 0 {
		return gen, core.NewError("native.ArchSession.generatePipelinedGPUOneShotTail: empty generation seed")
	}
	if len(gen) >= maxNew {
		return gen, nil
	}
	icbB, err := s.peerICB()
	if err != nil {
		return nil, err
	}
	icbs := [2]*archICBReplay{s.state.icb, icbB}
	sc := [2]*plGPUScratch{s.gpuTailPLScratchBuffer(0), s.gpuTailPLScratchBuffer(1)}
	type infl struct {
		cb      metal.MTLCommandBufferObject
		scratch *headGreedyScratch
	}
	var rerr error
	withAutoreleasePool(func() {
		tokBuf := s.nextInputTokenBuffer(gen[len(gen)-1])
		sc[0].out = icbs[0].pleInput
		seedCB := commandBufferFast(queue)
		seedEnc := computeCommandEncoderFast(seedCB)
		if e := s.encNextInputsGPU(seedEnc, tokBuf, icbs[0].ping0, sc[0]); e != nil {
			endEncodingFast(seedEnc)
			rerr = e
			return
		}
		endEncodingFast(seedEnc)
		commitCommandBufferFast(seedCB)
		waitUntilCompletedFast(seedCB)

		submit := func(i int) (infl, bool) {
			icb, tgt := icbs[i], icbs[1-i]
			sc[i].out = tgt.pleInput
			cb := commandBufferFast(queue)
			enc := computeCommandEncoderFast(cb)
			lastOut := icb.encodeStepBodyNoInput(enc, s.pos)
			scratch, gok, gerr := s.headEnc.encodeGreedy(enc, lastOut, nil)
			if !gok || gerr != nil {
				endEncodingFast(enc)
				if scratch != nil {
					s.headEnc.putGreedyScratch(scratch)
				}
				if rerr = gerr; rerr == nil {
					rerr = core.NewError("native.ArchSession.generatePipelinedGPUOneShotTail: GPU head argmax unavailable mid-chain")
				}
				return infl{}, false
			}
			if e := s.encNextInputsGPU(enc, scratch.outToken, tgt.ping0, sc[i]); e != nil {
				endEncodingFast(enc)
				s.headEnc.putGreedyScratch(scratch)
				rerr = e
				return infl{}, false
			}
			endEncodingFast(enc)
			commitCommandBufferFast(cb)
			s.pos++
			return infl{cb: cb, scratch: scratch}, true
		}

		read := func(p infl) (int32, bool) {
			waitUntilCompletedFast(p.cb)
			if pieceTimingOn {
				chainedGPUSpanNs += int64(float64(p.cb.GPUEndTime()-p.cb.GPUStartTime()) * 1e9)
			}
			tk := p.scratch.token()
			s.headEnc.putGreedyScratch(p.scratch)
			if tk < 0 || int(tk) >= s.arch.Vocab {
				rerr = core.NewError("native.ArchSession.generatePipelinedGPUOneShotTail: invalid token")
				return 0, false
			}
			return tk, true
		}

		prev, ok := submit(0)
		if !ok {
			return
		}
		i := 1
		for len(gen) < maxNew {
			if len(gen)+1 < maxNew {
				nxt, ok := submit(i)
				if !ok {
					waitUntilCompletedFast(prev.cb)
					s.headEnc.putGreedyScratch(prev.scratch)
					return
				}
				i = 1 - i
				tk, valid := read(prev)
				if !valid {
					waitUntilCompletedFast(nxt.cb)
					s.headEnc.putGreedyScratch(nxt.scratch)
					return
				}
				gen = append(gen, tk)
				prev = nxt
				continue
			}
			tk, valid := read(prev)
			if valid {
				gen = append(gen, tk)
			}
			return
		}
		waitUntilCompletedFast(prev.cb)
		s.headEnc.putGreedyScratch(prev.scratch)
	})
	if rerr != nil {
		return nil, rerr
	}
	return gen, nil
}

// peerICB lazily records (once) the second ICB sharing this session's KV caches — its own ping0/pleInput,
// the same KV — for the submit-ahead decode's double buffer.
func (s *ArchSession) peerICB() (*archICBReplay, error) {
	if s.icbPeer != nil {
		return s.icbPeer, nil
	}
	if s.recordPeerICB == nil {
		return nil, core.NewError("native.ArchSession.peerICB: no peer recorder")
	}
	rep, err := s.recordPeerICB()
	if err != nil {
		return nil, err
	}
	s.icbPeer = rep
	return rep, nil
}

// generatePipelinedGPUTail is the submit-ahead form of generateChainedGPUTail: two ICBs (A/B) over the
// SAME KV caches, each with its own ping0/pleInput. Each step's cb writes the NEXT step's emb+pli into the
// OTHER ICB, so the host submits step t+1 before reading t's token — one command buffer always in flight
// ahead, the GPU serialising them through the shared KV. 1-ahead is discard-safe for greedy: each cb
// caches the token it reads (advancing pos by one per submit, so cached-count == pos), and the trailing
// speculative cb's produced token is dropped past eos/maxNew. Cache/pos byte-identical to the serial loop.
func (s *ArchSession) generatePipelinedGPUTail(gen []int32, maxNew, eosID int, suppress []int32, yield func(int32) bool, stop bool) ([]int32, error) {
	icbB, err := s.peerICB()
	if err != nil {
		return nil, err
	}
	icbs := [2]*archICBReplay{s.state.icb, icbB}
	sc := [2]*plGPUScratch{s.gpuTailPLScratchBuffer(0), s.gpuTailPLScratchBuffer(1)}
	type infl struct {
		cb           metal.MTLCommandBufferObject
		lastOut      *byte
		directHidden []byte
		scratch      *headGreedyScratch
	}
	var rerr error
	withAutoreleasePool(func() {
		// Seed icbA's inputs from the first token.
		tokBuf := s.nextInputTokenBuffer(gen[len(gen)-1])
		sc[0].out = icbs[0].pleInput
		seedCB := commandBufferFast(queue)
		seedEnc := computeCommandEncoderFast(seedCB)
		if e := s.encNextInputsGPU(seedEnc, tokBuf, icbs[0].ping0, sc[0]); e != nil {
			endEncodingFast(seedEnc)
			rerr = e
			return
		}
		endEncodingFast(seedEnc)
		commitCommandBufferFast(seedCB)
		waitUntilCompletedFast(seedCB)

		// submit encodes+commits one step on ICB i, writing the next step's emb+pli into ICB 1-i (no wait).
		submit := func(i int) (infl, bool) {
			icb, tgt := icbs[i], icbs[1-i]
			sc[i].out = tgt.pleInput
			cb := commandBufferFast(queue)
			enc := computeCommandEncoderFast(cb)
			lastOut, directHidden := s.encodeStepBodyNoInputRetained(enc, icb, s.pos)
			scratch, gok, gerr := s.headEnc.encodeGreedy(enc, lastOut, suppress)
			if !gok || gerr != nil {
				endEncodingFast(enc)
				if scratch != nil {
					s.headEnc.putGreedyScratch(scratch)
				}
				if rerr = gerr; rerr == nil {
					rerr = core.NewError("native.ArchSession.generatePipelinedGPUTail: GPU head argmax unavailable mid-chain")
				}
				return infl{}, false
			}
			s.encNextInputsGPU(enc, scratch.outToken, tgt.ping0, sc[i])
			endEncodingFast(enc)
			commitCommandBufferFast(cb)
			s.pos++
			return infl{cb: cb, lastOut: icb.lastOutPtr, directHidden: directHidden, scratch: scratch}, true
		}

		read := func(p infl) (int32, bool) {
			waitUntilCompletedFast(p.cb)
			if pieceTimingOn {
				chainedGPUSpanNs += int64(float64(p.cb.GPUEndTime()-p.cb.GPUStartTime()) * 1e9)
			}
			tk := p.scratch.token()
			s.headEnc.putGreedyScratch(p.scratch)
			if tk < 0 || int(tk) >= s.arch.Vocab {
				rerr = core.NewError("native.ArchSession.generatePipelinedGPUTail: invalid token")
				return 0, false
			}
			return tk, true
		}

		prev, ok := submit(0)
		if !ok {
			return
		}
		i := 1
		for len(gen) < maxNew && !stop {
			nxt, ok := submit(i)
			if !ok {
				waitUntilCompletedFast(prev.cb)
				s.headEnc.putGreedyScratch(prev.scratch)
				return
			}
			i = 1 - i
			tk, valid := read(prev)
			if !valid {
				waitUntilCompletedFast(nxt.cb)
				s.headEnc.putGreedyScratch(nxt.scratch)
				return
			}
			gen = append(gen, tk)
			stop = (yield != nil && !yield(tk)) || (eosID >= 0 && int(tk) == eosID)
			prev = nxt
		}
		// Drain the trailing in-flight cb. Its produced token is appended only if still within budget
		// (it was a needed token), else dropped (speculation past eos/maxNew). Either way its stepBody
		// cached the last appended token — so retain its hidden as the session boundary.
		tk, valid := read(prev)
		if valid && !stop && len(gen) < maxNew {
			gen = append(gen, tk)
		}
		if prev.directHidden != nil {
			s.retainedHidden = prev.directHidden
		} else {
			s.rememberRetainedHiddenFrom(prev.lastOut)
		}
	})
	if rerr != nil {
		return nil, rerr
	}
	return gen, nil
}

func (s *ArchSession) greedyFromHiddenInPool(hidden []byte, suppress []int32) (int32, error) {
	if s.greedy != nil {
		_ptHead := ptStart()
		next, ok, err := s.directGreedyFromHiddenInPool(hidden, suppress)
		ptEnd(2, _ptHead)
		if err != nil {
			return 0, err
		}
		if ok {
			return next, nil
		}
	}
	_ptHead := ptStart()
	var logits []byte
	var err error
	if s.canUseHeadLogitsScratch() {
		logits, err = s.headLogitsScratch(hidden, true)
	} else {
		logits, err = s.head(hidden, true)
	}
	ptEnd(2, _ptHead)
	if err != nil {
		return 0, err
	}
	return greedyBF16Suppressed(logits, s.arch.Vocab, suppress)
}

// GenerateText is the text-in/text-out wrapper over Generate, now that the tokenizer is a
// shared no-cgo package: it encodes prompt with tok, generates up to maxNew tokens (stopping
// at the tokenizer's EOS when it has one), and decodes the result back to a string. The
// session's cache carries over across calls, so successive GenerateText turns continue the
// conversation. The whole text → tokens → decode → text path runs with no cgo and no Python.
func (s *ArchSession) GenerateText(tok *tokenizer.Tokenizer, prompt string, maxNew int) (string, error) {
	if tok == nil {
		return "", core.NewError("native.ArchSession.GenerateText: nil tokenizer")
	}
	ids := tok.Encode(prompt)
	if len(ids) == 0 {
		return "", core.NewError("native.ArchSession.GenerateText: prompt encoded to no tokens")
	}
	eos := -1
	if tok.HasEOSToken() {
		eos = int(tok.EOSToken())
	}
	gen, err := s.Generate(ids, maxNew, eos)
	if err != nil {
		return "", err
	}
	return tok.Decode(gen), nil
}

// Generate appends promptIDs to the running sequence and greedily decodes up to maxNew
// tokens (or until eosID; eosID < 0 disables early stop), returning the generated ids.
// EVERY token — prompt and generated — is written to the persistent cache (the generated
// tokens too, so the sequence is complete), so a following Generate continues this exact
// sequence. The cache carries over until the session is dropped.
func (s *ArchSession) Generate(promptIDs []int32, maxNew, eosID int) ([]int32, error) {
	return s.generate(promptIDs, maxNew, eosID, nil, nil)
}

// GenerateEach is Generate with per-token streaming: each token is yielded after it is
// selected and written into the session cache. If yield returns false, decoding stops
// without treating consumer stop as an error; the returned slice contains the tokens
// emitted before the stop.
func (s *ArchSession) GenerateEach(promptIDs []int32, maxNew, eosID int, yield func(int32) bool) ([]int32, error) {
	return s.GenerateEachWithSuppressionAndTransform(promptIDs, maxNew, eosID, nil, nil, yield)
}

// GenerateEachTransformed is GenerateEach with a committed-token transform
// applied before each generated token is written to the session cache.
func (s *ArchSession) GenerateEachTransformed(promptIDs []int32, maxNew, eosID int, transform TokenTransform, yield func(int32) bool) ([]int32, error) {
	return s.GenerateEachWithSuppressionAndTransform(promptIDs, maxNew, eosID, nil, transform, yield)
}

// GenerateEachWithSuppression is GenerateEach with suppressed token ids masked
// before greedy argmax.
func (s *ArchSession) GenerateEachWithSuppression(promptIDs []int32, maxNew, eosID int, suppress []int32, yield func(int32) bool) ([]int32, error) {
	return s.GenerateEachWithSuppressionAndTransform(promptIDs, maxNew, eosID, suppress, nil, yield)
}

// GenerateEachWithSuppressionAndTransform combines greedy token suppression
// with a committed-token transform.
func (s *ArchSession) GenerateEachWithSuppressionAndTransform(promptIDs []int32, maxNew, eosID int, suppress []int32, transform TokenTransform, yield func(int32) bool) ([]int32, error) {
	return s.generateWithYield(promptIDs, maxNew, eosID, nil, suppress, transform, yield)
}

// GenerateSampledEach is native's sampled retained-session path: it keeps the
// transformer stack on the ArchSession replay path, materialises full vocab
// logits for the host sampler, then commits every sampled token into the
// resident cache. This is the sampled sibling of GenerateEach for serve paths
// that cannot use direct on-GPU greedy argmax.
func (s *ArchSession) GenerateSampledEach(promptIDs []int32, maxNew int, stopTokens []int32, sampler *model.Sampler, params model.SampleParams, transform model.TokenTransform, yield func(int32) bool) ([]int32, error) {
	if sampler == nil {
		return nil, core.NewError("native.ArchSession.GenerateSampledEach: nil sampler")
	}
	if len(promptIDs) == 0 {
		return nil, core.NewError("native.ArchSession.GenerateSampledEach: empty prompt")
	}
	if maxNew <= 0 {
		return nil, core.NewError("native.ArchSession.GenerateSampledEach: maxNew must be > 0")
	}
	if s.pos+len(promptIDs)+maxNew > s.maxLen {
		return nil, core.NewError("native.ArchSession.GenerateSampledEach: sequence would exceed maxLen cache rows")
	}
	startPos := s.pos
	var gen []int32
	var genErr error
	withAutoreleasePool(func() {
		hidden, err := s.prefillPromptRetainedInPool(promptIDs)
		if err != nil {
			genErr = err
			return
		}
		gen, genErr = s.generateSampledFromHiddenInPool(hidden, maxNew, stopTokens, sampler, params, transform, yield, true)
	})
	if genErr != nil {
		return nil, genErr
	}
	s.appendKnownResidentIDs(startPos, promptIDs, gen)
	return gen, genErr
}

// GenerateSampledOneShotEach is the serve/request sibling of GenerateSampledEach:
// it streams sampled tokens through the native session but does not cache the
// final generated token because the fresh request session is about to be
// dropped. That mirrors GenerateOneShot's greedy final-step saving.
func (s *ArchSession) GenerateSampledOneShotEach(promptIDs []int32, maxNew int, stopTokens []int32, sampler *model.Sampler, params model.SampleParams, transform model.TokenTransform, yield func(int32) bool) ([]int32, error) {
	if sampler == nil {
		return nil, core.NewError("native.ArchSession.GenerateSampledOneShotEach: nil sampler")
	}
	if len(promptIDs) == 0 {
		return nil, core.NewError("native.ArchSession.GenerateSampledOneShotEach: empty prompt")
	}
	if maxNew <= 0 {
		return nil, core.NewError("native.ArchSession.GenerateSampledOneShotEach: maxNew must be > 0")
	}
	if s.pos+len(promptIDs)+maxNew > s.maxLen {
		return nil, core.NewError("native.ArchSession.GenerateSampledOneShotEach: sequence would exceed maxLen cache rows")
	}
	var gen []int32
	var genErr error
	withAutoreleasePool(func() {
		hidden, err := s.prefillPromptRetainedInPool(promptIDs)
		if err != nil {
			genErr = err
			return
		}
		gen, genErr = s.generateSampledFromHiddenInPool(hidden, maxNew, stopTokens, sampler, params, transform, yield, false)
	})
	return gen, genErr
}

// GenerateWithSuppression is the native sibling of pkg/metal's suppressed
// direct-greedy path: suppressed token ids are masked before argmax, including
// when the resident head can return the token directly without materialising
// full vocab logits.
func (s *ArchSession) GenerateWithSuppression(promptIDs []int32, maxNew, eosID int, suppress []int32) ([]int32, error) {
	return s.generate(promptIDs, maxNew, eosID, nil, suppress)
}

// GenerateOneShot is the contract-level greedy path used by model.Generate
// when it opens and closes a fresh session for one request. It uses the same
// direct greedy engine as retained Generate, but does not step the final
// generated token because no caller can reuse that closed session's final cache
// row. Retained callers should use Generate / GenerateEach instead.
func (s *ArchSession) GenerateOneShot(promptIDs []int32, maxNew, eosID int) ([]int32, error) {
	if len(promptIDs) == 0 {
		return nil, core.NewError("native.ArchSession.GenerateOneShot: empty prompt")
	}
	if maxNew <= 0 {
		return nil, core.NewError("native.ArchSession.GenerateOneShot: maxNew must be > 0")
	}
	if s.pos+len(promptIDs)+maxNew > s.maxLen {
		return nil, core.NewError("native.ArchSession.GenerateOneShot: sequence would exceed maxLen cache rows")
	}
	var gen []int32
	var genErr error
	withAutoreleasePool(func() {
		hidden, err := s.prefillPromptRetainedInPool(promptIDs)
		if err != nil {
			genErr = err
			return
		}
		gen, genErr = s.generateOneShotFromHiddenInPool(hidden, maxNew, eosID)
	})
	return gen, genErr
}

func (s *ArchSession) generateOneShotFromHiddenInPool(hidden []byte, maxNew, eosID int) ([]int32, error) {
	gen := make([]int32, 0, maxNew)
	next, err := s.greedyFromHiddenInPool(hidden, nil)
	if err != nil {
		return nil, err
	}
	gen = append(gen, next)
	stop := eosID >= 0 && int(next) == eosID

	if !stop && len(gen) < maxNew &&
		s.encNextInputsGPU != nil && s.plScratchNew != nil && s.state.icb != nil && s.headEnc != nil && s.greedy != nil &&
		!stepGreedyChainDisabled && !chainedGPUInputsDisabled && !icbDisabledForTest {
		return s.generateChainedGPUOneShotTail(gen, maxNew, eosID, stop)
	}

	for !stop && len(gen) < maxNew {
		if hidden, err = s.stepIDInPool(next); err != nil {
			return nil, err
		}
		next, err = s.greedyFromHiddenInPool(hidden, nil)
		if err != nil {
			return nil, err
		}
		gen = append(gen, next)
		stop = eosID >= 0 && int(next) == eosID
	}
	return gen, nil
}

func (s *ArchSession) generate(promptIDs []int32, maxNew, eosID int, rememberPromptIDs []int32, suppress []int32) ([]int32, error) {
	return s.generateWithYield(promptIDs, maxNew, eosID, rememberPromptIDs, suppress, nil, nil)
}

func (s *ArchSession) generateSampledFromHiddenInPool(hidden []byte, maxNew int, stopTokens []int32, sampler *model.Sampler, params model.SampleParams, transform model.TokenTransform, yield func(int32) bool, cacheFinal bool) ([]int32, error) {
	history := s.sampleHistoryScratchFor(params, maxNew)
	finalHistory := history
	defer func() { s.sampleHistory = finalHistory }()
	gen, finalHistory, err := s.generateSampledFromHiddenInPoolWithHistory(hidden, maxNew, stopTokens, sampler, params, transform, yield, cacheFinal, 0, history)
	return gen, err
}

func (s *ArchSession) generateSampledFromHiddenInPoolWithHistory(hidden []byte, maxNew int, stopTokens []int32, sampler *model.Sampler, params model.SampleParams, transform model.TokenTransform, yield func(int32) bool, cacheFinal bool, initialGenerated int, history []int32) ([]int32, []int32, error) {
	gen := make([]int32, 0, maxNew)
	var readyLogits []byte
	var readyIDs []int32
	var readyToken int32
	readyTokenOK := false
	for len(gen) < maxNew {
		pickParams := params
		if params.MinTokensBeforeStop > 0 && initialGenerated+len(gen) < params.MinTokensBeforeStop {
			pickParams.SuppressTokens = s.suppressionTokensScratch(params.SuppressTokens, stopTokens)
		}
		var next int32
		var err error
		if sampledGreedyParamsEligible(pickParams) {
			next, err = s.headGreedyOrLogits(hidden, pickParams.SuppressTokens, nil, nil, false)
			readyLogits, readyIDs = nil, nil
			readyTokenOK = false
		} else if readyTokenOK {
			next = readyToken
			readyTokenOK = false
		} else if readyIDs != nil {
			next, err = sampler.SampleCandidates(readyLogits, readyIDs, pickParams)
			readyLogits, readyIDs = nil, nil
		} else if sampledTopOneGreedyParamsEligible(pickParams, history) {
			sampler.Draw()
			next, err = s.headGreedyOrLogits(hidden, pickParams.SuppressTokens, nil, nil, false)
			readyLogits, readyIDs = nil, nil
			readyTokenOK = false
		} else if s.sampleTopKTokenParamsEligible(pickParams) {
			draw := sampler.Draw()
			var ok bool
			next, ok, err = s.sampleTopKTokenFromHiddenInPool(hidden, pickParams, draw, history)
			if !ok && err == nil {
				err = core.NewError("native.ArchSession.generateSampledFromHiddenInPool: TopK token path declined after eligibility check")
			}
		} else if s.sampleLogitsTokenParamsEligible(pickParams) && !sampleLogitsTokenCPUPreferred(pickParams, s.arch.Vocab) {
			draw := sampler.Draw()
			var ok bool
			next, ok, err = s.sampleLogitsTokenFromHiddenInPool(hidden, pickParams, draw, history)
			if !ok && err == nil {
				err = core.NewError("native.ArchSession.generateSampledFromHiddenInPool: logits token path declined after eligibility check")
			}
		} else if candidateLogits, candidateIDs, ok, topKErr := s.sampleTopKCandidatesFromHiddenWithHistoryInPool(hidden, pickParams, history); topKErr != nil {
			return nil, history, topKErr
		} else if ok {
			next, err = sampler.SampleCandidates(candidateLogits, candidateIDs, pickParams)
		} else {
			logits, headErr := s.headLogitsScratch(hidden, false)
			if headErr != nil {
				return nil, history, headErr
			}
			pickLogits := logits
			if params.RepeatPenalty > 1 {
				pickLogits, err = s.repeatPenaltyLogitsScratch(logits, s.arch.Vocab, history, params.RepeatPenalty)
				if err != nil {
					return nil, history, err
				}
			}
			if sampleLogitsTokenCPUPreferred(pickParams, s.arch.Vocab) {
				next, err = sampleSmallVocabBF16(pickLogits, s.arch.Vocab, sampler, pickParams)
			} else {
				next, err = s.sampleVocabBF16(pickLogits, s.arch.Vocab, sampler, pickParams)
			}
		}
		if err != nil {
			return nil, history, err
		}
		if transform != nil {
			next = transform(next)
		}
		gen = append(gen, next)
		if params.RepeatPenalty > 1 {
			history = append(history, next)
		}
		stop := (yield != nil && !yield(next)) || nativeTokenInSet(next, stopTokens)
		if !cacheFinal && (stop || len(gen) >= maxNew) {
			break
		}
		nextPickParams := params
		if params.MinTokensBeforeStop > 0 && initialGenerated+len(gen) < params.MinTokensBeforeStop {
			nextPickParams.SuppressTokens = s.suppressionTokensScratch(params.SuppressTokens, stopTokens)
		}
		if !stop && len(gen) < maxNew && s.sampledChainedGPUTailCanContinue(nextPickParams, history, transform) {
			return s.generateSampledChainedGPUTail(gen, maxNew, stopTokens, sampler, params, yield, cacheFinal, initialGenerated, history)
		}
		stepped := false
		if !sampledGreedyParamsEligible(nextPickParams) {
			if sampledTopOneGreedyParamsEligible(nextPickParams, history) && s.state.icb != nil && !icbDisabledForTest && s.headEnc != nil && s.greedy != nil {
				sampler.Draw()
				if chainedToken, chainedHidden, ok, chainErr := s.stepGreedyInPool(next, nil, nextPickParams.SuppressTokens); chainErr != nil {
					return nil, history, chainErr
				} else if ok {
					hidden, readyToken, readyTokenOK = chainedHidden, chainedToken, true
					readyLogits, readyIDs = nil, nil
					stepped = true
				}
			} else if s.state.icb != nil && !icbDisabledForTest && s.sampleTopKTokenParamsEligible(nextPickParams) {
				draw := sampler.Draw()
				if chainedHidden, chainedToken, ok, chainErr := s.stepSampleTopKTokenInPool(next, nextPickParams, draw, history); chainErr != nil {
					return nil, history, chainErr
				} else if ok {
					hidden, readyToken, readyTokenOK = chainedHidden, chainedToken, true
					readyLogits, readyIDs = nil, nil
					stepped = true
				}
			} else if s.state.icb != nil && !icbDisabledForTest && s.sampleLogitsTokenParamsEligible(nextPickParams) {
				draw := sampler.Draw()
				if chainedHidden, chainedToken, ok, chainErr := s.stepSampleLogitsTokenInPool(next, nextPickParams, draw, history); chainErr != nil {
					return nil, history, chainErr
				} else if ok {
					hidden, readyToken, readyTokenOK = chainedHidden, chainedToken, true
					readyLogits, readyIDs = nil, nil
					stepped = true
				}
			}
		}
		if !stepped && !sampledGreedyParamsEligible(nextPickParams) {
			if chainedHidden, chainedLogits, chainedIDs, ok, chainErr := s.stepSampleTopKCandidatesWithHistoryInPool(next, nextPickParams, history); chainErr != nil {
				return nil, history, chainErr
			} else if ok {
				hidden, readyLogits, readyIDs = chainedHidden, chainedLogits, chainedIDs
				readyTokenOK = false
				stepped = true
			}
		}
		if !stepped {
			hidden, err = s.stepIDRetainedInPool(next)
			if err != nil {
				return nil, history, err
			}
		}
		s.rememberRetainedHidden(hidden)
		if stop {
			break
		}
	}
	return gen, history, nil
}

func (s *ArchSession) sampledChainedGPUTailCanContinue(params model.SampleParams, history []int32, transform model.TokenTransform) bool {
	if transform != nil || chainedGPUInputsDisabled || icbDisabledForTest {
		return false
	}
	if s == nil || s.state.icb == nil || s.encNextInputsGPU == nil || s.plScratchNew == nil || s.headEnc == nil {
		return false
	}
	if sampledGreedyParamsEligible(params) || sampledTopOneGreedyParamsEligible(params, history) {
		return false
	}
	if s.sampleTopKTokenParamsEligible(params) {
		return true
	}
	return s.sampleLogitsTokenParamsEligible(params) && !sampleLogitsTokenCPUPreferred(params, s.arch.Vocab)
}

func (s *ArchSession) sampledPipelinedGPUTailCanContinue(params model.SampleParams, history []int32, transform model.TokenTransform) bool {
	return pipelinedGPUDecodeEnabled &&
		params.RepeatPenalty <= 1 &&
		s != nil &&
		s.recordPeerICB != nil &&
		s.sampledChainedGPUTailCanContinue(params, history, transform)
}

func (s *ArchSession) generateSampledChainedGPUTail(gen []int32, maxNew int, stopTokens []int32, sampler *model.Sampler, params model.SampleParams, yield func(int32) bool, cacheFinal bool, initialGenerated int, history []int32) ([]int32, []int32, error) {
	if s.sampledPipelinedGPUTailCanContinue(params, history, nil) {
		if cacheFinal {
			return s.generateSampledPipelinedGPUTail(gen, maxNew, stopTokens, sampler, params, yield, initialGenerated, history)
		}
		if yield == nil && len(stopTokens) == 0 {
			return s.generateSampledPipelinedGPUOneShotTail(gen, maxNew, sampler, params, initialGenerated, history)
		}
	}
	icb := s.state.icb
	sc := s.gpuTailPLScratchBuffer(0)
	sc.out = icb.pleInput
	if len(gen) == 0 {
		return gen, history, core.NewError("native.ArchSession.generateSampledChainedGPUTail: empty generation seed")
	}
	tokBuf := s.nextInputTokenBuffer(gen[len(gen)-1])
	seedCB := commandBufferFast(queue)
	seedEnc := computeCommandEncoderFast(seedCB)
	if err := s.encNextInputsGPU(seedEnc, tokBuf, icb.ping0, sc); err != nil {
		endEncodingFast(seedEnc)
		return gen, history, err
	}
	endEncodingFast(seedEnc)
	commitCommandBufferFast(seedCB)
	waitUntilCompletedFast(seedCB)

	for len(gen) < maxNew {
		pickParams := params
		if params.MinTokensBeforeStop > 0 && initialGenerated+len(gen) < params.MinTokensBeforeStop {
			pickParams.SuppressTokens = s.suppressionTokensScratch(params.SuppressTokens, stopTokens)
		}
		if !s.sampledChainedGPUTailCanContinue(pickParams, history, nil) {
			break
		}
		draw := sampler.Draw()
		var token int32
		var ok bool
		var stepErr error
		cb := commandBufferFast(queue)
		enc := computeCommandEncoderFast(cb)
		lastOut, directHidden := s.encodeStepBodyNoInputRetained(enc, icb, s.pos)
		if s.sampleTopKTokenParamsEligible(pickParams) {
			var scratch *headTopKScratch
			scratch, ok, stepErr = s.headEnc.encodeTopKSampleFast(enc, lastOut, pickParams, draw, history)
			if !ok || stepErr != nil {
				endEncodingFast(enc)
				if scratch != nil {
					s.headEnc.putTopKScratch(scratch)
				}
				if stepErr == nil {
					stepErr = core.NewError("native.ArchSession.generateSampledChainedGPUTail: TopK token path declined mid-chain")
				}
				return gen, history, stepErr
			}
			stepErr = s.encNextInputsGPU(enc, scratch.outToken, icb.ping0, sc)
			endEncodingFast(enc)
			if stepErr != nil {
				s.headEnc.putTopKScratch(scratch)
				return gen, history, stepErr
			}
			commitCommandBufferFast(cb)
			waitUntilCompletedFast(cb)
			token = scratch.token()
			s.headEnc.putTopKScratch(scratch)
		} else {
			var scratch *headGreedyScratch
			scratch, ok, stepErr = s.headEnc.encodeLogitsSample(enc, lastOut, pickParams, draw, history)
			if !ok || stepErr != nil {
				endEncodingFast(enc)
				if scratch != nil {
					s.headEnc.putGreedyScratch(scratch)
				}
				if stepErr == nil {
					stepErr = core.NewError("native.ArchSession.generateSampledChainedGPUTail: logits token path declined mid-chain")
				}
				return gen, history, stepErr
			}
			stepErr = s.encNextInputsGPU(enc, scratch.outToken, icb.ping0, sc)
			endEncodingFast(enc)
			if stepErr != nil {
				s.headEnc.putGreedyScratch(scratch)
				return gen, history, stepErr
			}
			commitCommandBufferFast(cb)
			waitUntilCompletedFast(cb)
			token = scratch.token()
			s.headEnc.putGreedyScratch(scratch)
		}
		s.pos++
		if token < 0 || int(token) >= s.arch.Vocab {
			return gen, history, core.NewError("native.ArchSession.generateSampledChainedGPUTail: sampled invalid token")
		}
		if directHidden != nil {
			s.retainedHidden = directHidden
		} else {
			s.rememberRetainedHiddenFrom(icb.lastOutPtr)
		}
		gen = append(gen, token)
		if params.RepeatPenalty > 1 {
			history = append(history, token)
		}
		stop := (yield != nil && !yield(token)) || nativeTokenInSet(token, stopTokens)
		if !cacheFinal && (stop || len(gen) >= maxNew) {
			return gen, history, nil
		}
		if stop {
			break
		}
	}
	if cacheFinal && len(gen) > 0 {
		cb := commandBufferFast(queue)
		enc := computeCommandEncoderFast(cb)
		_, directHidden := s.encodeStepBodyNoInputRetained(enc, icb, s.pos)
		endEncodingFast(enc)
		commitCommandBufferFast(cb)
		waitUntilCompletedFast(cb)
		s.pos++
		if directHidden != nil {
			s.retainedHidden = directHidden
		} else {
			s.rememberRetainedHiddenFrom(icb.lastOutPtr)
		}
	}
	return gen, history, nil
}

func (s *ArchSession) generateSampledPipelinedGPUTail(gen []int32, maxNew int, stopTokens []int32, sampler *model.Sampler, params model.SampleParams, yield func(int32) bool, initialGenerated int, history []int32) ([]int32, []int32, error) {
	if len(gen) == 0 {
		return gen, history, core.NewError("native.ArchSession.generateSampledPipelinedGPUTail: empty generation seed")
	}
	icbB, err := s.peerICB()
	if err != nil {
		return gen, history, err
	}
	icbs := [2]*archICBReplay{s.state.icb, icbB}
	sc := [2]*plGPUScratch{s.gpuTailPLScratchBuffer(0), s.gpuTailPLScratchBuffer(1)}

	type inflightSampledStep struct {
		cb           metal.MTLCommandBufferObject
		lastOut      *byte
		directHidden []byte
		topK         *headTopKScratch
		logits       *headGreedyScratch
	}
	var rerr error

	release := func(p inflightSampledStep) {
		if p.topK != nil {
			s.headEnc.putTopKScratch(p.topK)
		}
		if p.logits != nil {
			s.headEnc.putGreedyScratch(p.logits)
		}
	}

	read := func(p inflightSampledStep) (int32, bool) {
		waitUntilCompletedFast(p.cb)
		if pieceTimingOn {
			chainedGPUSpanNs += int64(float64(p.cb.GPUEndTime()-p.cb.GPUStartTime()) * 1e9)
		}
		var token int32
		switch {
		case p.topK != nil:
			token = p.topK.token()
		case p.logits != nil:
			token = p.logits.token()
		default:
			rerr = core.NewError("native.ArchSession.generateSampledPipelinedGPUTail: missing sampled scratch")
			return 0, false
		}
		release(p)
		if token < 0 || int(token) >= s.arch.Vocab {
			rerr = core.NewError("native.ArchSession.generateSampledPipelinedGPUTail: sampled invalid token")
			return 0, false
		}
		return token, true
	}

	submit := func(i, generatedBefore int) (inflightSampledStep, bool) {
		pickParams := params
		if params.MinTokensBeforeStop > 0 && initialGenerated+generatedBefore < params.MinTokensBeforeStop {
			pickParams.SuppressTokens = s.suppressionTokensScratch(params.SuppressTokens, stopTokens)
		}
		if !s.sampledPipelinedGPUTailCanContinue(pickParams, history, nil) {
			rerr = core.NewError("native.ArchSession.generateSampledPipelinedGPUTail: sampled parameters changed to a non-pipeline shape")
			return inflightSampledStep{}, false
		}
		draw := sampler.Draw()
		icb, tgt := icbs[i], icbs[1-i]
		sc[i].out = tgt.pleInput
		cb := commandBufferFast(queue)
		enc := computeCommandEncoderFast(cb)
		lastOut, directHidden := s.encodeStepBodyNoInputRetained(enc, icb, s.pos)
		if s.sampleTopKTokenParamsEligible(pickParams) {
			scratch, _, stepErr := s.headEnc.encodeTopKSampleFast(enc, lastOut, pickParams, draw, history)
			if stepErr != nil {
				endEncodingFast(enc)
				if scratch != nil {
					s.headEnc.putTopKScratch(scratch)
				}
				if stepErr == nil {
					stepErr = core.NewError("native.ArchSession.generateSampledPipelinedGPUTail: TopK token path declined mid-pipeline")
				}
				rerr = stepErr
				return inflightSampledStep{}, false
			}
			if stepErr = s.encNextInputsGPU(enc, scratch.outToken, tgt.ping0, sc[i]); stepErr != nil {
				endEncodingFast(enc)
				s.headEnc.putTopKScratch(scratch)
				rerr = stepErr
				return inflightSampledStep{}, false
			}
			endEncodingFast(enc)
			commitCommandBufferFast(cb)
			s.pos++
			return inflightSampledStep{cb: cb, lastOut: icb.lastOutPtr, directHidden: directHidden, topK: scratch}, true
		}
		scratch, _, stepErr := s.headEnc.encodeLogitsSample(enc, lastOut, pickParams, draw, history)
		if stepErr != nil {
			endEncodingFast(enc)
			if scratch != nil {
				s.headEnc.putGreedyScratch(scratch)
			}
			if stepErr == nil {
				stepErr = core.NewError("native.ArchSession.generateSampledPipelinedGPUTail: logits token path declined mid-pipeline")
			}
			rerr = stepErr
			return inflightSampledStep{}, false
		}
		if stepErr = s.encNextInputsGPU(enc, scratch.outToken, tgt.ping0, sc[i]); stepErr != nil {
			endEncodingFast(enc)
			s.headEnc.putGreedyScratch(scratch)
			rerr = stepErr
			return inflightSampledStep{}, false
		}
		endEncodingFast(enc)
		commitCommandBufferFast(cb)
		s.pos++
		return inflightSampledStep{cb: cb, lastOut: icb.lastOutPtr, directHidden: directHidden, logits: scratch}, true
	}

	tokBuf := s.nextInputTokenBuffer(gen[len(gen)-1])
	sc[0].out = icbs[0].pleInput
	seedCB := commandBufferFast(queue)
	seedEnc := computeCommandEncoderFast(seedCB)
	if err := s.encNextInputsGPU(seedEnc, tokBuf, icbs[0].ping0, sc[0]); err != nil {
		endEncodingFast(seedEnc)
		return gen, history, err
	}
	endEncodingFast(seedEnc)
	commitCommandBufferFast(seedCB)
	waitUntilCompletedFast(seedCB)

	prev, ok := submit(0, len(gen))
	if !ok {
		return gen, history, rerr
	}
	i := 1
	stop := false
	for len(gen) < maxNew && !stop {
		nxt, ok := submit(i, len(gen)+1)
		if !ok {
			waitUntilCompletedFast(prev.cb)
			release(prev)
			return gen, history, rerr
		}
		i = 1 - i
		token, valid := read(prev)
		if !valid {
			waitUntilCompletedFast(nxt.cb)
			release(nxt)
			return gen, history, rerr
		}
		gen = append(gen, token)
		stop = (yield != nil && !yield(token)) || nativeTokenInSet(token, stopTokens)
		prev = nxt
	}
	read(prev)
	if rerr != nil {
		return gen, history, rerr
	}
	if prev.directHidden != nil {
		s.retainedHidden = prev.directHidden
	} else {
		s.rememberRetainedHiddenFrom(prev.lastOut)
	}
	return gen, history, nil
}

func (s *ArchSession) generateSampledPipelinedGPUOneShotTail(gen []int32, maxNew int, sampler *model.Sampler, params model.SampleParams, initialGenerated int, history []int32) ([]int32, []int32, error) {
	if len(gen) == 0 {
		return gen, history, core.NewError("native.ArchSession.generateSampledPipelinedGPUOneShotTail: empty generation seed")
	}
	if len(gen) >= maxNew {
		return gen, history, nil
	}
	icbB, err := s.peerICB()
	if err != nil {
		return gen, history, err
	}
	icbs := [2]*archICBReplay{s.state.icb, icbB}
	sc := [2]*plGPUScratch{s.gpuTailPLScratchBuffer(0), s.gpuTailPLScratchBuffer(1)}

	type inflightSampledStep struct {
		cb           metal.MTLCommandBufferObject
		lastOut      *byte
		directHidden []byte
		topK         *headTopKScratch
		logits       *headGreedyScratch
	}
	var rerr error

	release := func(p inflightSampledStep) {
		if p.topK != nil {
			s.headEnc.putTopKScratch(p.topK)
		}
		if p.logits != nil {
			s.headEnc.putGreedyScratch(p.logits)
		}
	}

	read := func(p inflightSampledStep) (int32, bool) {
		waitUntilCompletedFast(p.cb)
		if pieceTimingOn {
			chainedGPUSpanNs += int64(float64(p.cb.GPUEndTime()-p.cb.GPUStartTime()) * 1e9)
		}
		var token int32
		switch {
		case p.topK != nil:
			token = p.topK.token()
		case p.logits != nil:
			token = p.logits.token()
		default:
			rerr = core.NewError("native.ArchSession.generateSampledPipelinedGPUOneShotTail: missing sampled scratch")
			return 0, false
		}
		release(p)
		if token < 0 || int(token) >= s.arch.Vocab {
			rerr = core.NewError("native.ArchSession.generateSampledPipelinedGPUOneShotTail: sampled invalid token")
			return 0, false
		}
		return token, true
	}

	submit := func(i, generatedBefore int) (inflightSampledStep, bool) {
		pickParams := params
		if params.MinTokensBeforeStop > 0 && initialGenerated+generatedBefore < params.MinTokensBeforeStop {
			pickParams.SuppressTokens = s.suppressionTokensScratch(params.SuppressTokens, nil)
		}
		if !s.sampledPipelinedGPUTailCanContinue(pickParams, history, nil) {
			rerr = core.NewError("native.ArchSession.generateSampledPipelinedGPUOneShotTail: sampled parameters changed to a non-pipeline shape")
			return inflightSampledStep{}, false
		}
		draw := sampler.Draw()
		icb, tgt := icbs[i], icbs[1-i]
		sc[i].out = tgt.pleInput
		cb := commandBufferFast(queue)
		enc := computeCommandEncoderFast(cb)
		lastOut, directHidden := s.encodeStepBodyNoInputRetained(enc, icb, s.pos)
		if s.sampleTopKTokenParamsEligible(pickParams) {
			scratch, _, stepErr := s.headEnc.encodeTopKSampleFast(enc, lastOut, pickParams, draw, history)
			if stepErr != nil {
				endEncodingFast(enc)
				if scratch != nil {
					s.headEnc.putTopKScratch(scratch)
				}
				if stepErr == nil {
					stepErr = core.NewError("native.ArchSession.generateSampledPipelinedGPUOneShotTail: TopK token path declined mid-pipeline")
				}
				rerr = stepErr
				return inflightSampledStep{}, false
			}
			if stepErr = s.encNextInputsGPU(enc, scratch.outToken, tgt.ping0, sc[i]); stepErr != nil {
				endEncodingFast(enc)
				s.headEnc.putTopKScratch(scratch)
				rerr = stepErr
				return inflightSampledStep{}, false
			}
			endEncodingFast(enc)
			commitCommandBufferFast(cb)
			s.pos++
			return inflightSampledStep{cb: cb, lastOut: icb.lastOutPtr, directHidden: directHidden, topK: scratch}, true
		}
		scratch, _, stepErr := s.headEnc.encodeLogitsSample(enc, lastOut, pickParams, draw, history)
		if stepErr != nil {
			endEncodingFast(enc)
			if scratch != nil {
				s.headEnc.putGreedyScratch(scratch)
			}
			if stepErr == nil {
				stepErr = core.NewError("native.ArchSession.generateSampledPipelinedGPUOneShotTail: logits token path declined mid-pipeline")
			}
			rerr = stepErr
			return inflightSampledStep{}, false
		}
		if stepErr = s.encNextInputsGPU(enc, scratch.outToken, tgt.ping0, sc[i]); stepErr != nil {
			endEncodingFast(enc)
			s.headEnc.putGreedyScratch(scratch)
			rerr = stepErr
			return inflightSampledStep{}, false
		}
		endEncodingFast(enc)
		commitCommandBufferFast(cb)
		s.pos++
		return inflightSampledStep{cb: cb, lastOut: icb.lastOutPtr, directHidden: directHidden, logits: scratch}, true
	}

	tokBuf := s.nextInputTokenBuffer(gen[len(gen)-1])
	sc[0].out = icbs[0].pleInput
	seedCB := commandBufferFast(queue)
	seedEnc := computeCommandEncoderFast(seedCB)
	if err := s.encNextInputsGPU(seedEnc, tokBuf, icbs[0].ping0, sc[0]); err != nil {
		endEncodingFast(seedEnc)
		return gen, history, err
	}
	endEncodingFast(seedEnc)
	commitCommandBufferFast(seedCB)
	waitUntilCompletedFast(seedCB)

	prev, ok := submit(0, len(gen))
	if !ok {
		return gen, history, rerr
	}
	i := 1
	for {
		if len(gen)+1 < maxNew {
			nxt, ok := submit(i, len(gen)+1)
			if !ok {
				waitUntilCompletedFast(prev.cb)
				release(prev)
				return gen, history, rerr
			}
			i = 1 - i
			token, valid := read(prev)
			if !valid {
				waitUntilCompletedFast(nxt.cb)
				release(nxt)
				return gen, history, rerr
			}
			gen = append(gen, token)
			prev = nxt
			continue
		}
		token, valid := read(prev)
		if valid {
			gen = append(gen, token)
		}
		if prev.directHidden != nil {
			s.retainedHidden = prev.directHidden
		} else {
			s.rememberRetainedHiddenFrom(prev.lastOut)
		}
		return gen, history, rerr
	}
}

func (s *ArchSession) sampleTopKCandidatesFromHiddenInPool(hidden []byte, params model.SampleParams) ([]byte, []int32, bool, error) {
	return s.sampleTopKCandidatesFromHiddenWithHistoryInPool(hidden, params, nil)
}

func (s *ArchSession) sampleTopKCandidatesFromHiddenWithHistoryInPool(hidden []byte, params model.SampleParams, history []int32) ([]byte, []int32, bool, error) {
	if !s.sampleTopKParamsEligible(params) {
		return nil, nil, false, nil
	}
	var logits []byte
	var ids []int32
	var ok bool
	var err error
	if hiddenBuf := s.retainedHiddenBufferFor(hidden); hiddenBuf != nil {
		logits, ids, ok, err = s.headEnc.sampleTopKCandidatesBufferWithHistoryInto(hiddenBuf, params.TopK, params.SuppressTokens, history, params.RepeatPenalty, s.sampleCandidateLogits, s.sampleCandidateIDs, false)
	} else {
		logits, ids, ok, err = s.headEnc.sampleTopKCandidatesWithHistoryInto(hidden, params.TopK, params.SuppressTokens, history, params.RepeatPenalty, s.sampleCandidateLogits, s.sampleCandidateIDs, false)
	}
	if ok {
		s.sampleCandidateLogits, s.sampleCandidateIDs = logits, ids
	}
	return logits, ids, ok, err
}

func (s *ArchSession) sampleTopKTokenFromHiddenInPool(hidden []byte, params model.SampleParams, draw float32, history []int32) (int32, bool, error) {
	if !s.sampleTopKTokenParamsEligible(params) {
		return 0, false, nil
	}
	if hiddenBuf := s.retainedHiddenBufferFor(hidden); hiddenBuf != nil {
		return s.headEnc.sampleTopKTokenBufferInPool(hiddenBuf, params, draw, history)
	}
	return s.headEnc.sampleTopKTokenInPool(hidden, params, draw, history)
}

func (s *ArchSession) sampleLogitsTokenFromHiddenInPool(hidden []byte, params model.SampleParams, draw float32, history []int32) (int32, bool, error) {
	if !s.sampleLogitsTokenParamsEligible(params) {
		return 0, false, nil
	}
	if hiddenBuf := s.retainedHiddenBufferFor(hidden); hiddenBuf != nil {
		return s.headEnc.sampleLogitsTokenBufferInPool(hiddenBuf, params, draw, history)
	}
	return s.headEnc.sampleLogitsTokenInPool(hidden, params, draw, history)
}

func (s *ArchSession) sampleTokenFromRetainedLogitsInPool(params model.SampleParams, draw float32, history []int32) (int32, bool, error) {
	logitsBuf := s.retainedLogitsBuffer()
	if logitsBuf == nil || !s.retainedLogitsSampleParamsEligible(params) {
		return 0, false, nil
	}
	return s.headEnc.sampleLogitsBufferInPool(logitsBuf, params, draw, history)
}

func (s *ArchSession) generateWithYield(promptIDs []int32, maxNew, eosID int, rememberPromptIDs []int32, suppress []int32, transform TokenTransform, yield func(int32) bool) ([]int32, error) {
	if len(promptIDs) == 0 {
		return nil, core.NewError("native.ArchSession.Generate: empty prompt")
	}
	if maxNew <= 0 {
		return nil, core.NewError("native.ArchSession.Generate: maxNew must be > 0")
	}
	if s.pos+len(promptIDs)+maxNew > s.maxLen {
		return nil, core.NewError("native.ArchSession.Generate: sequence would exceed maxLen cache rows")
	}
	startPos := s.pos
	var gen []int32
	var genErr error
	withAutoreleasePool(func() {
		// prefill the new prompt over the carried-over cache; keep the last hidden state.
		hidden, err := s.prefillPromptRetainedInPool(promptIDs)
		if err != nil {
			genErr = err
			return
		}
		if len(rememberPromptIDs) > 0 {
			cacheFirstLogits := func(logits []byte) {
				s.rememberCachedPromptEntry(rememberPromptIDs, hidden, logits)
			}
			gen, genErr = s.generateFromHiddenInPool(hidden, maxNew, eosID, nil, cacheFirstLogits, suppress, transform, yield)
			return
		}
		// decode: head → greedy → append → step the new token (caching it for the next turn).
		gen, genErr = s.generateFromHiddenInPool(hidden, maxNew, eosID, nil, nil, suppress, transform, yield)
	})
	if genErr != nil {
		return nil, genErr
	}
	s.appendKnownResidentIDs(startPos, promptIDs, gen)
	return gen, genErr
}

func (s *ArchSession) appendKnownResidentIDs(startPos int, promptIDs, gen []int32) {
	if s == nil {
		return
	}
	if startPos < 0 || len(s.cachedIDs) < startPos {
		s.cachedIDs = nil
		return
	}
	s.cachedIDs = s.cachedIDs[:startPos]
	s.cachedIDs = append(s.cachedIDs, promptIDs...)
	s.cachedIDs = append(s.cachedIDs, gen...)
}

func nativeTokenInSet(id int32, tokens []int32) bool {
	return slices.Contains(tokens, id)
}

func nativeAppendSuppressionTokens(base, extra []int32) []int32 {
	if len(extra) == 0 {
		return base
	}
	out := make([]int32, 0, len(base)+len(extra))
	out = append(out, base...)
	for _, token := range extra {
		if nativeTokenInSet(token, out) {
			continue
		}
		out = append(out, token)
	}
	return out
}

func nativeApplyRepeatPenaltyBF16(logits []byte, vocab int, history []int32, penalty float32) ([]byte, error) {
	if len(logits) != vocab*bf16Size {
		return nil, core.NewError("native.applyRepeatPenalty: logits must be vocab bf16 bytes")
	}
	if penalty <= 1 || len(history) == 0 {
		return logits, nil
	}
	ids := make([]int32, 0, len(history))
	for _, id := range history {
		if id >= 0 && int(id) < vocab {
			ids = append(ids, id)
		}
	}
	if len(ids) == 0 {
		return logits, nil
	}
	slices.Sort(ids)
	out := make([]byte, len(logits))
	copy(out, logits)
	applyRepeatPenaltySortedIDsBF16(out, ids, penalty)
	return out, nil
}

func applyRepeatPenaltySortedIDsBF16(out []byte, ids []int32, penalty float32) {
	var prev int32
	for i, id := range ids {
		if i > 0 && id == prev {
			continue
		}
		prev = id
		off := int(id) * bf16Size
		v := bf16ToF32(out[off], out[off+1])
		if v > 0 {
			v /= penalty
		} else {
			v *= penalty
		}
		h := f32ToBF16(v)
		out[off] = byte(h)
		out[off+1] = byte(h >> 8)
	}
}
