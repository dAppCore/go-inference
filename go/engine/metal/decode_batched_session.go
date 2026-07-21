// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"time"
	"unsafe"

	core "dappco.re/go"
	"dappco.re/go/inference/model"
	"github.com/tmc/apple/metal"
)

type denseBatchScratch struct {
	inRowsStack        [16]metal.MTLBuffer
	outRowsStack       [16]metal.MTLBuffer
	readRowsStack      [16]metal.MTLBuffer
	directOutRowsStack [16]metal.MTLBuffer
	lastRowBufStack    [16]metal.MTLBuffer
	offBufStack        [16]metal.MTLBuffer
	offPtrStack        [16]*int32
	offOffStack        [16]uint
	rowOffStack        [16]uint
	readOffStack       [16]uint
	directOutOffStack  [16]uint
	inputViewStack     [16]cachedNoCopyBytesView
	outputViewStack    [16]cachedNoCopyBytesView
	inRows             []metal.MTLBuffer
	outRows            []metal.MTLBuffer
	readRows           []metal.MTLBuffer
	directOutRows      []metal.MTLBuffer
	lastRowBuf         []metal.MTLBuffer
	offBuf             []metal.MTLBuffer
	offPtr             []*int32
	offOff             []uint
	rowOff             []uint
	readOff            []uint
	directOutOff       []uint
	inputViews         []cachedNoCopyBytesView
	outputViews        []cachedNoCopyBytesView
	lastOutView        cachedNoCopyBytesView
	offPacked          metal.MTLBuffer
	offPackedCap       int
	inPacked           metal.MTLBuffer
	outPacked          metal.MTLBuffer
	rowPackedCap       int
	rowBytes           int
	lastRows           metal.MTLBuffer
	lastRowOff         []uint
	lastK              int
	lastResult         [1][]byte
	// MLP-fold slabs (K-row): the attn halves write their outputs into hPacked so all K rows are
	// alive at once, then ONE rms-rows + three batched gemvs + one fused gelu run the whole layer's
	// MLP — each layer's gate/up/down weights swept once instead of K times.
	hPacked       metal.MTLBuffer // K × dModel attn-half outputs (the fold's h)
	mlpNormPacked metal.MTLBuffer // K × dModel rms(h) feeding gate/up
	gatePacked    metal.MTLBuffer // K × dFFMax
	upPacked      metal.MTLBuffer // K × dFFMax
	gatedPacked   metal.MTLBuffer // K × dFFMax gelu(gate)·up
	downPacked    metal.MTLBuffer // K × dModel down-projection outputs
	foldRowCap    int
	foldDModel    int
	foldDFFCap    int
	// attention-fold slabs: the Q/K/V/O projections batch across rows the same way, with the
	// ordered per-row tail (norm+rope, value norm, SDPA) keeping exact sequential cache semantics.
	attnNormPacked metal.MTLBuffer // K × dModel rms(x) feeding Q/K/V
	qPacked        metal.MTLBuffer // K × qDimMax roped queries
	attnPacked     metal.MTLBuffer // K × qDimMax SDPA outputs
	attnOutPacked  metal.MTLBuffer // K × dModel O-projection outputs
	kStagePacked   metal.MTLBuffer // K × kvDimMax staged K rows (ring-wrap landing)
	vStagePacked   metal.MTLBuffer // K × kvDimMax staged V rows
	attnRowCap     int             // attnFold's OWN row capacity — NOT foldRowCap (mlpFold updates that first, which masked row growth and left these slabs short: the ~52K wide-tail-chunk corruption)
	attnDModel     int
	foldQDimCap    int
	foldKVDimCap   int
	// per-layer staging for the deferred-landing lane (the big-K staged sliding tail): each
	// staged owner's K/V stay alive across the whole layer loop for its sharers, landing in bulk
	// at the end of the chunk.
	layerKStage      []metal.MTLBuffer
	layerVStage      []metal.MTLBuffer
	layerStageRowCap int
	layerStageKVCap  int
	// prompt-attention GEMM score slabs (sdpa_prompt_gemm.go): two K × nCap buffers
	// double-buffered across heads so head h+1's wide QKᵀ GEMM overlaps head h's skinny
	// P@V GEMM instead of draining the GPU behind its 32 output tiles.
	sdpaS0      metal.MTLBuffer
	sdpaS1      metal.MTLBuffer
	sdpaSRowCap int // sdpaPromptS's OWN capacities — never another fold's (the attnRowCap lesson)
	sdpaSNCap   int
	// q8 prompt staging (#375/#367): the per-chunk dequant of a q8 owner's
	// attended prefix lands here instead of materialising per-layer
	// full-cacheRows mirror planes (31B@256K: ~19GB of planes vs one pair per
	// parity). Ping-pong by owner-index parity so layer li+1's dequant write
	// does not serialise against layer li's flash read under Metal's
	// in-encoder hazard tracking.
	q8StageK   [2]metal.MTLBuffer
	q8StageV   [2]metal.MTLBuffer
	q8StageCap [2]int // bf16 element capacity per parity — q8Stage's OWN (the attnRowCap lesson)
	// moeBatch holds the K-row MoE fold slabs (moe_batch.go) — nil until an MoE chunk runs.
	moeBatch *moeBatchScratch
	// moeGrouped holds the grouped-GEMM expert lane's sorted-order slabs (moe_grouped.go).
	moeGrouped *moeGroupedScratch
}

// mlpFold returns the K-row MLP-fold slabs, (re)allocating when the batch width, model width or
// the widest per-layer FFN grows. dFFMax is the max dFF across the foldable layers (gemma4 E2B/E4B
// vary it per layer); each layer's gate/up rows still land contiguously at z·itsOwnDFF in the slab.
func (s *denseBatchScratch) mlpFold(k, dModel, dFFMax int) (h, normed, gate, up, gated, down metal.MTLBuffer) {
	if s.hPacked == nil || s.foldRowCap < k || s.foldDModel != dModel || s.foldDFFCap < dFFMax {
		// growth re-allocates: release the outgrown slabs first (newBuffer returns +1
		// retained; the old handles leaked on every widening chunk). GPU-only slabs —
		// no cached host pointers — and in-flight command buffers retain their own refs.
		releaseDeviceBuffers(s.hPacked, s.mlpNormPacked, s.downPacked, s.gatePacked, s.upPacked, s.gatedPacked)
		s.hPacked = scratchBF16(k * dModel)
		s.mlpNormPacked = scratchBF16(k * dModel)
		s.downPacked = scratchBF16(k * dModel)
		s.gatePacked = scratchBF16(k * dFFMax)
		s.upPacked = scratchBF16(k * dFFMax)
		s.gatedPacked = scratchBF16(k * dFFMax)
		s.foldRowCap, s.foldDModel, s.foldDFFCap = k, dModel, dFFMax
	}
	return s.hPacked, s.mlpNormPacked, s.gatePacked, s.upPacked, s.gatedPacked, s.downPacked
}

// attnFold returns the attention-fold slabs, (re)allocating when the batch width, model width or
// the widest per-layer head geometry grows. Growth is tracked by attnFold's OWN attnRowCap /
// attnDModel: the old code keyed on mlpFold's foldRowCap, which mlpFold (always called first)
// had ALREADY raised for a wider chunk — attnFold then skipped its realloc and every attention
// slab stayed at the old row count. At ~52K-token prompts the sliding-tail absorption produces
// one chunk wider than all before it (window + tail), so rows past the stale capacity read and
// wrote out of bounds: undefined bytes → per-process NaN/garbage → the long-context corruption.
func (s *denseBatchScratch) attnFold(k, dModel, qDimMax, kvDimMax int) (normed, q, attn, attnOut, kStage, vStage metal.MTLBuffer) {
	if s.attnNormPacked == nil || s.attnRowCap < k || s.attnDModel != dModel || s.foldQDimCap < qDimMax || s.foldKVDimCap < kvDimMax {
		releaseDeviceBuffers(s.attnNormPacked, s.attnOutPacked, s.qPacked, s.attnPacked, s.kStagePacked, s.vStagePacked)
		s.attnNormPacked = scratchBF16(k * dModel)
		s.attnOutPacked = scratchBF16(k * dModel)
		s.qPacked = scratchBF16(k * qDimMax)
		s.attnPacked = scratchBF16(k * qDimMax)
		s.kStagePacked = scratchBF16(k * kvDimMax)
		s.vStagePacked = scratchBF16(k * kvDimMax)
		s.attnRowCap, s.attnDModel = k, dModel
		s.foldQDimCap, s.foldKVDimCap = qDimMax, kvDimMax
	}
	return s.attnNormPacked, s.qPacked, s.attnPacked, s.attnOutPacked, s.kStagePacked, s.vStagePacked
}

// layerStage returns layer li's PRIVATE K/V staging slabs for the deferred-landing lane — every
// staged owner keeps its batch K/V alive until the end-of-chunk landing, so shared-KV layers can
// read the owner's true pre-batch ring + stage. Sized by the attnFold caps; call after attnFold.
func (s *denseBatchScratch) layerStage(li, layers, k, kvDimMax int) (kSt, vSt metal.MTLBuffer) {
	if len(s.layerKStage) != layers || s.layerStageRowCap < k || s.layerStageKVCap < kvDimMax {
		releaseDeviceBuffers(s.layerKStage...)
		releaseDeviceBuffers(s.layerVStage...)
		s.layerKStage = make([]metal.MTLBuffer, layers)
		s.layerVStage = make([]metal.MTLBuffer, layers)
		s.layerStageRowCap, s.layerStageKVCap = k, kvDimMax
	}
	if s.layerKStage[li] == nil {
		s.layerKStage[li] = scratchBF16(s.layerStageRowCap * s.layerStageKVCap)
		s.layerVStage[li] = scratchBF16(s.layerStageRowCap * s.layerStageKVCap)
	}
	return s.layerKStage[li], s.layerVStage[li]
}

// sdpaPromptS returns the two prompt-attention score slabs (kRows × nCap bf16 each) for the
// GEMM SDPA composition, (re)allocating when the chunk row count grows or the attended-length
// cap rises. nCap is the session's maxLen so the allocation happens ONCE per session instead
// of every deepening chunk. Growth is tracked by sdpaPromptS's OWN capacity fields — never
// another fold's (the attnRowCap lesson: capacity consumed outside its owner left slabs short
// at the wide tail-absorbed chunk).
func (s *denseBatchScratch) sdpaPromptS(kRows, nCap int) (s0, s1 metal.MTLBuffer) {
	if s.sdpaS0 == nil || s.sdpaSRowCap < kRows || s.sdpaSNCap < nCap {
		// the census's GB-scale leak: these two slabs run to sdpaPromptSBudgetBytes/
		// sdpaPromptSMaxBytes each, and every growth dropped the old pair unreleased.
		releaseDeviceBuffers(s.sdpaS0, s.sdpaS1)
		rows := max(kRows, s.sdpaSRowCap)
		cols := max(nCap, s.sdpaSNCap)
		s.sdpaS0 = scratchBF16(rows * cols)
		s.sdpaS1 = scratchBF16(rows * cols)
		s.sdpaSRowCap, s.sdpaSNCap = rows, cols
	}
	return s.sdpaS0, s.sdpaS1
}

// q8Stage returns the ping-pong q8 prompt-staging pair for parity, sized to at
// least rows×kvd bf16 values, releasing an outgrown pair before reallocating.
// The prefill dequant is per-chunk work either way (the mirror lane also
// re-dequantised the attended prefix every chunk) — staging only changes WHERE
// it lands: one shared pair per parity instead of a full-cacheRows plane per
// layer, which is the 31B@256K ingest-peak cut.
func (s *denseBatchScratch) q8Stage(parity, rows, kvd int) (k, v metal.MTLBuffer) {
	need := rows * kvd
	if s.q8StageK[parity] == nil || s.q8StageCap[parity] < need {
		releaseDeviceBuffers(s.q8StageK[parity], s.q8StageV[parity])
		grown := max(need, s.q8StageCap[parity])
		s.q8StageK[parity] = scratchBF16(grown)
		s.q8StageV[parity] = scratchBF16(grown)
		s.q8StageCap[parity] = grown
	}
	return s.q8StageK[parity], s.q8StageV[parity]
}

func (s *denseBatchScratch) Close() {
	if s == nil {
		return
	}
	for i := range s.inputViewStack {
		s.inputViewStack[i].Close()
	}
	for i := range s.outputViewStack {
		s.outputViewStack[i].Close()
	}
	for i := range s.inputViews {
		s.inputViews[i].Close()
	}
	for i := range s.outputViews {
		s.outputViews[i].Close()
	}
	s.lastOutView.Close()
	// the fold/stage/score slabs are +1 retained and session-lifetime — zeroing the
	// struct without releasing them leaked the whole grown set on every session close.
	// (The row-plumbing buffers — offBuf/inPacked/outPacked/lastRows — cache host
	// pointers and keep their own lifecycle; they are NOT released here.)
	releaseDeviceBuffers(s.hPacked, s.mlpNormPacked, s.downPacked, s.gatePacked, s.upPacked, s.gatedPacked)
	releaseDeviceBuffers(s.attnNormPacked, s.attnOutPacked, s.qPacked, s.attnPacked, s.kStagePacked, s.vStagePacked)
	releaseDeviceBuffers(s.layerKStage...)
	releaseDeviceBuffers(s.layerVStage...)
	releaseDeviceBuffers(s.sdpaS0, s.sdpaS1)
	releaseDeviceBuffers(s.q8StageK[0], s.q8StageV[0], s.q8StageK[1], s.q8StageV[1])
	*s = denseBatchScratch{}
}

func (s *denseBatchScratch) inputViewsFor(k int) []cachedNoCopyBytesView {
	if k <= len(s.inputViewStack) {
		return s.inputViewStack[:k]
	}
	if cap(s.inputViews) < k {
		for i := range s.inputViews {
			s.inputViews[i].Close()
		}
		s.inputViews = make([]cachedNoCopyBytesView, k)
	} else {
		s.inputViews = s.inputViews[:k]
	}
	return s.inputViews
}

func (s *denseBatchScratch) outputViewsFor(k int) []cachedNoCopyBytesView {
	if k <= len(s.outputViewStack) {
		return s.outputViewStack[:k]
	}
	if cap(s.outputViews) < k {
		for i := range s.outputViews {
			s.outputViews[i].Close()
		}
		s.outputViews = make([]cachedNoCopyBytesView, k)
	} else {
		s.outputViews = s.outputViews[:k]
	}
	return s.outputViews
}

func (s *denseBatchScratch) rows(k, dModel int) (inRows, outRows, offBuf []metal.MTLBuffer, offPtr []*int32, offOff, rowOff []uint) {
	if k <= len(s.inRowsStack) {
		s.inRows = s.inRowsStack[:k]
		s.outRows = s.outRowsStack[:k]
		s.offBuf = s.offBufStack[:k]
		s.offPtr = s.offPtrStack[:k]
		s.offOff = s.offOffStack[:k]
		s.rowOff = s.rowOffStack[:k]
	} else if cap(s.inRows) < k || cap(s.outRows) < k || cap(s.offBuf) < k || cap(s.offPtr) < k || cap(s.offOff) < k || cap(s.rowOff) < k {
		s.inRows = make([]metal.MTLBuffer, k)
		s.outRows = make([]metal.MTLBuffer, k)
		s.offBuf = make([]metal.MTLBuffer, k)
		s.offPtr = make([]*int32, k)
		s.offOff = make([]uint, k)
		s.rowOff = make([]uint, k)
	} else {
		s.inRows = s.inRows[:k]
		s.outRows = s.outRows[:k]
		s.offBuf = s.offBuf[:k]
		s.offPtr = s.offPtr[:k]
		s.offOff = s.offOff[:k]
		s.rowOff = s.rowOff[:k]
	}
	if s.offPacked == nil || s.offPackedCap < k {
		s.offPacked = device.NewBufferWithLengthOptions(uint(k*4), metal.MTLResourceStorageModeShared)
		s.offPackedCap = k
	}
	rowBytes := dModel * bf16Size
	if s.inPacked == nil || s.outPacked == nil || s.rowPackedCap < k || s.rowBytes != rowBytes {
		s.inPacked = scratchBF16(k * dModel)
		s.outPacked = scratchBF16(k * dModel)
		s.rowPackedCap = k
		s.rowBytes = rowBytes
	}
	offsets := unsafe.Slice((*int32)(s.offPacked.Contents()), k)
	for i := range k {
		s.inRows[i] = s.inPacked
		s.outRows[i] = s.outPacked
		s.offBuf[i] = s.offPacked
		s.offPtr[i] = &offsets[i]
		s.offOff[i] = uint(i * 4)
		s.rowOff[i] = uint(i * rowBytes)
	}
	return s.inRows, s.outRows, s.offBuf, s.offPtr, s.offOff, s.rowOff
}

func (s *denseBatchScratch) readRowsFor(k int) ([]metal.MTLBuffer, []uint) {
	if k <= len(s.readRowsStack) {
		s.readRows = s.readRowsStack[:k]
		s.readOff = s.readOffStack[:k]
	} else if cap(s.readRows) < k || cap(s.readOff) < k {
		s.readRows = make([]metal.MTLBuffer, k)
		s.readOff = make([]uint, k)
	} else {
		s.readRows = s.readRows[:k]
		s.readOff = s.readOff[:k]
	}
	return s.readRows, s.readOff
}

func (s *denseBatchScratch) directOutputRowsFor(k int) ([]metal.MTLBuffer, []uint) {
	if k <= len(s.directOutRowsStack) {
		s.directOutRows = s.directOutRowsStack[:k]
		s.directOutOff = s.directOutOffStack[:k]
	} else if cap(s.directOutRows) < k || cap(s.directOutOff) < k {
		s.directOutRows = make([]metal.MTLBuffer, k)
		s.directOutOff = make([]uint, k)
	} else {
		s.directOutRows = s.directOutRows[:k]
		s.directOutOff = s.directOutOff[:k]
	}
	return s.directOutRows, s.directOutOff
}

func (s *denseBatchScratch) setLastRows(rows []metal.MTLBuffer, rowOff []uint, k int) {
	if k <= 0 || len(rows) < k || len(rowOff) < k {
		s.lastRows = nil
		s.lastRowBuf = nil
		s.lastRowOff = nil
		s.lastK = 0
		return
	}
	if k <= len(s.lastRowBufStack) {
		s.lastRowBuf = s.lastRowBufStack[:k]
	} else if cap(s.lastRowBuf) < k {
		s.lastRowBuf = make([]metal.MTLBuffer, k)
	} else {
		s.lastRowBuf = s.lastRowBuf[:k]
	}
	copy(s.lastRowBuf, rows[:k])
	s.lastRows = rows[0]
	s.lastRowOff = rowOff[:k]
	s.lastK = k
}

func (s *denseBatchScratch) lastOutputView(out []byte) (metal.MTLBuffer, bool) {
	if s == nil || len(out) == 0 {
		return nil, false
	}
	return s.lastOutView.buffer(out)
}

// decode_batched_session.go — the session-level MTP batched verify: K query tokens through the WHOLE
// resident decode stack in as few command buffers as possible, reusing the resident layer weights and
// caches (no re-upload). Each row i decodes at position basePos+i, writes its K/V into every layer's
// cache at row basePos+i, and attends [0..basePos+i] with the SAME single-query kernels stepToken
// uses — so the K returned hiddens are BYTE-IDENTICAL to calling stepToken K times at basePos..
// basePos+K-1 (proven in decode_batched_session_test.go). This is what lets MTPDecode verify a whole
// K-token draft block against the resident cache in one batched pass instead of K stepGreedy rounds.
//
// v1 covers the dense uniform path (every layer owns its cache; per-layer output scalar handled
// on-device). Layers needing a host flush per row — MoE FFN, the PLE input gate, shared-KV, the trace
// hooks — are out of scope here; stepTokensBatchedDense reports !ok so MTPDecode falls back to the
// byte-identical sequential verify for those models. Folding the K per-row projections into one steel
// GEMM (weight reuse) is the further speedup that trades byte- for token-identity (metal-MTP parity).

// stepTokensBatchedDense runs K tokens at positions [basePos, basePos+K) through the resident layer
// stack and returns their K output hiddens ([]([]byte), each dModel bf16). It writes each token's K/V
// into the per-layer caches at row basePos+i. ok is false (no work done, no cache mutation) when the
// model is outside the dense uniform path — the caller then steps sequentially. Single-goroutine, like
// every ArchSession decode. Must run inside a withAutoreleasePool.
func (s *archDecodeState) stepTokensBatchedDense(embs [][]byte, basePos int) (out [][]byte, ok bool, err error) {
	return s.stepTokensBatchedDenseResult(embs, basePos, true, false, nil, nil)
}

// stepTokensBatchedDensePLE / ...NoResultPLE / ...IntoPLE are the PLE-arch twins
// (gemma4 E2B/E4B): pleSlab carries the K tokens' per-layer-input tensors and each
// row's gate encodes in the same command buffer — the MTP verify's batched fast
// path for the E-family.
func (s *archDecodeState) stepTokensBatchedDensePLE(embs [][]byte, pleSlab []byte, basePos int) (out [][]byte, ok bool, err error) {
	return s.stepTokensBatchedDenseResultWithInputViewsPLE(embs, pleSlab, basePos, true, false, nil, nil, true)
}

func (s *archDecodeState) stepTokensBatchedDenseNoResultPLE(embs [][]byte, pleSlab []byte, basePos int) (ok bool, err error) {
	_, ok, err = s.stepTokensBatchedDenseResultWithInputViewsPLE(embs, pleSlab, basePos, false, false, nil, nil, true)
	return ok, err
}

func (s *archDecodeState) stepTokensBatchedDenseIntoPLE(embs [][]byte, pleSlab []byte, basePos int, dstRows [][]byte) (out [][]byte, ok bool, err error) {
	return s.stepTokensBatchedDenseResultWithInputViewsPLE(embs, pleSlab, basePos, true, false, nil, dstRows, true)
}

func (s *archDecodeState) stepTokensBatchedDenseNoResult(embs [][]byte, basePos int) (ok bool, err error) {
	_, ok, err = s.stepTokensBatchedDenseResult(embs, basePos, false, false, nil, nil)
	return ok, err
}

func (s *archDecodeState) stepTokensBatchedDenseLastInto(embs [][]byte, basePos int, dst []byte) (last []byte, ok bool, err error) {
	out, ok, err := s.stepTokensBatchedDenseResult(embs, basePos, true, true, dst, nil)
	if err != nil || !ok {
		return nil, ok, err
	}
	if len(out) != 1 {
		return nil, true, core.NewError("native.stepTokensBatchedDenseLast: hidden result count mismatch")
	}
	return out[0], true, nil
}

// stepTokensBatchedDenseLastIntoPLE is stepTokensBatchedDenseLastInto for a PLE (gemma4 E2B/E4B)
// arch: pleSlab carries the K tokens' per-layer-input tensors (token-major, K × numLayers·pliDim
// bf16) and each row's gate is encoded in the same command buffer as its attention + MLP halves.
// Without a slab a PLE arch still declines — the bail keeps the MTP verify wrappers (which pass
// no slab) on their proven sequential fallback.
func (s *archDecodeState) stepTokensBatchedDenseLastIntoPLE(embs [][]byte, pleSlab []byte, basePos int, dst []byte) (last []byte, ok bool, err error) {
	out, ok, err := s.stepTokensBatchedDenseResultWithInputViewsPLE(embs, pleSlab, basePos, true, true, dst, nil, true)
	if err != nil || !ok {
		return nil, ok, err
	}
	if len(out) != 1 {
		return nil, true, core.NewError("native.stepTokensBatchedDenseLast: hidden result count mismatch")
	}
	return out[0], true, nil
}

func (s *archDecodeState) stepTokensBatchedDenseLastIntoCopyInputs(embs [][]byte, basePos int, dst []byte) (last []byte, ok bool, err error) {
	out, ok, err := s.stepTokensBatchedDenseResultWithInputViews(embs, basePos, true, true, dst, nil, false)
	if err != nil || !ok {
		return nil, ok, err
	}
	if len(out) != 1 {
		return nil, true, core.NewError("native.stepTokensBatchedDenseLast: hidden result count mismatch")
	}
	return out[0], true, nil
}

func (s *archDecodeState) stepTokensBatchedDenseInto(embs [][]byte, basePos int, dstRows [][]byte) (out [][]byte, ok bool, err error) {
	return s.stepTokensBatchedDenseResult(embs, basePos, true, false, nil, dstRows)
}

func (s *archDecodeState) stepTokensBatchedDenseResult(embs [][]byte, basePos int, readResult, readLastOnly bool, lastDst []byte, dstRows [][]byte) (out [][]byte, ok bool, err error) {
	return s.stepTokensBatchedDenseResultWithInputViews(embs, basePos, readResult, readLastOnly, lastDst, dstRows, true)
}

func (s *archDecodeState) stepTokensBatchedDenseResultWithInputViews(embs [][]byte, basePos int, readResult, readLastOnly bool, lastDst []byte, dstRows [][]byte, directInputs bool) (out [][]byte, ok bool, err error) {
	return s.stepTokensBatchedDenseResultWithInputViewsPLE(embs, nil, basePos, readResult, readLastOnly, lastDst, dstRows, directInputs)
}

// batchedMLPFoldDisabledForTest forces the batched dense pass onto the per-row MLP interleave —
// the A/B lever for the fold's parity tests and profiling. Production never sets it; the fold and
// the per-row path produce byte-identical rows either way.
var batchedMLPFoldDisabledForTest bool

// batchedRopeDisabledForTest forces the attention fold back onto per-row fused norm+rope
// dispatches — the A/B lever for the batched-rows rope's parity/engagement tests.
var batchedRopeDisabledForTest bool

// batchedEpilogueDisabledForTest forces the fold back onto the per-row entry-rms, residual and
// layer-tail (PLE gate + scalar) dispatches — the A/B lever for the rows-batched epilogue.
var batchedEpilogueDisabledForTest bool

// batchedDenseICBMaxRows caps the batch width the dense body accepts on a recorded-ICB session
// when the FOLD cannot engage (per-row interleave only — right for MTP verify and accept-commit
// blocks, ≤ draft block+1, and hopeless at prompt scale). With the fold available the cap lifts:
// prompt prefill runs the batched projections over the replay's caches.
const batchedDenseICBMaxRows = 16

// projectRowsRequired encodes a fold projection through the projector's batched dispatch and
// hard-errors on a mid-fold decline — the fold pre-checked rowsCapable, so a decline here would
// leave the layer half-encoded (a silent wrong-bytes hazard, never a fallback point).
func projectRowsRequired(proj projector, enc metal.MTLComputeCommandEncoder, in, out metal.MTLBuffer, inOff, outOff uint, rows int, p projIndex) error {
	handled, err := proj.projectRows(enc, in, out, inOff, outOff, rows, p)
	if err != nil {
		return err
	}
	if !handled {
		return core.NewError("native.stepTokensBatchedDense: fold projection declined mid-encode")
	}
	return nil
}

// encBatchedRowEpilogue encodes row i's gemma4 tail for layer li — the per-layer-input gate (PLE,
// when the arch has one and the caller supplied the K-token slab) and the per-layer output scalar —
// reading and writing the row's layer output in place. Shared by the per-row MLP path and the
// MLP-fold's last-layer fallback; the shared gate scratch hazard-orders the rows. rows is the batch
// width K (the layer-major PLE slab strides by it).
func (s *archDecodeState) encBatchedRowEpilogue(enc metal.MTLComputeCommandEncoder, pleSlabBuf metal.MTLBuffer, li, i, rows int, outBuf metal.MTLBuffer, outOff uint) error {
	if pleSlabBuf != nil && len(s.ple) > li && len(s.ple[li].postNorm) > 0 {
		pl := s.ple[li]
		if len(pl.postNorm) != s.dModel*bf16Size {
			return core.NewError("native.stepTokensBatchedDense: PLE post norm size mismatch")
		}
		pliOff := uint((li*rows + i) * s.pliDim * bf16Size)
		if pl.bits == 0 { // bf16 PLE gate (the quant path sets bits 4/8 ⇒ the qmv)
			if len(pl.gate.Packed) != s.pliDim*s.dModel*bf16Size || len(pl.proj.Packed) != s.dModel*s.pliDim*bf16Size {
				return core.NewError("native.stepTokensBatchedDense: PLE bf16 weight size mismatch")
			}
			if err := encPerLayerInputGateBF16ScratchAt(enc, s.perLayerInputGateScratch(), outBuf, outOff, residentBytes(pl.gate.Packed), pleSlabBuf, residentBytes(pl.proj.Packed), residentBytes(pl.postNorm), outBuf, outOff, pliOff, s.dModel, s.pliDim, s.eps); err != nil {
				return err
			}
		} else {
			gateGroupSize, gateBits, err := validatePerLayerInputGateQuantWeight("gate", pl.gate, s.pliDim, s.dModel, pl.groupSize, pl.bits)
			if err != nil {
				return err
			}
			projGroupSize, projBits, err := validatePerLayerInputGateQuantWeight("projection", pl.proj, s.dModel, s.pliDim, pl.groupSize, pl.bits)
			if err != nil {
				return err
			}
			gatePacked, gateScales, gateBiases := quantWeightViews(pl.gate)
			projPacked, projScales, projBiases := quantWeightViews(pl.proj)
			if err := encPerLayerInputGateQuantScratchAt(enc, s.perLayerInputGateScratch(), outBuf, outOff, gatePacked, gateScales, gateBiases, pleSlabBuf, projPacked, projScales, projBiases, residentBytes(pl.postNorm), outBuf, outOff, pliOff, s.dModel, s.pliDim, gateGroupSize, gateBits, projGroupSize, projBits, s.eps); err != nil {
				return err
			}
		}
	}
	if s.lb[li].layerScalar != nil { // gemma4 per-layer output scalar (on-device)
		return encMulBF16To(enc, outBuf, s.lb[li].layerScalar, outBuf, outOff, 0, outOff, s.dModel)
	}
	return nil
}

// encBatchedEpilogueRows encodes the WHOLE layer tail for K contiguous output rows in a handful of
// dispatches: the PLE gate chain (gate gemv → gelu·pli → proj gemv → post-norm rows → add, each
// batched across the rows via grid Z / the layer-major slab) and the per-layer output scalar (the
// broadcast rows-multiply). Byte-identical per row to K encBatchedRowEpilogue calls — same kernels
// per element, the weight matrices swept once instead of K times, and none of the shared-scratch
// hazard serialisation. The caller guarantees outBuf rows are contiguous at outBase + r·dModel and
// supplies the free fold slabs as scratch (gate/mult K×pliDim-capable, proj/norm K×dModel).
func (s *archDecodeState) encBatchedEpilogueRows(enc metal.MTLComputeCommandEncoder, pleSlabBuf metal.MTLBuffer, li, rows int, outBuf metal.MTLBuffer, outBase uint, gateSlab, multSlab, projSlab, normSlab metal.MTLBuffer) error {
	if pleSlabBuf != nil && len(s.ple) > li && len(s.ple[li].postNorm) > 0 {
		pl := s.ple[li]
		if len(pl.postNorm) != s.dModel*bf16Size {
			return core.NewError("native.stepTokensBatchedDense: PLE post norm size mismatch")
		}
		if pl.bits == 0 { // bf16 PLE gate (the quant path sets bits 4/8 ⇒ the qmm)
			if len(pl.gate.Packed) != s.pliDim*s.dModel*bf16Size || len(pl.proj.Packed) != s.dModel*s.pliDim*bf16Size {
				return core.NewError("native.stepTokensBatchedDense: PLE bf16 weight size mismatch")
			}
			if err := encGemvBF16BatchedAt(enc, residentBytes(pl.gate.Packed), outBuf, gateSlab, 0, outBase, 0, s.pliDim, s.dModel, rows); err != nil {
				return err
			}
			// the layer's K per-token PLE slices are contiguous in the layer-major slab
			pliBase := uint(li * rows * s.pliDim * bf16Size)
			if err := encGeluGateMulFusedTo(enc, gateSlab, pleSlabBuf, multSlab, 0, pliBase, 0, rows*s.pliDim); err != nil {
				return err
			}
			if err := encGemvBF16BatchedAt(enc, residentBytes(pl.proj.Packed), multSlab, projSlab, 0, 0, 0, s.dModel, s.pliDim, rows); err != nil {
				return err
			}
			if err := encRMSNormRowsBF16(enc, projSlab, residentBytes(pl.postNorm), normSlab, 0, 0, 0, rows, s.dModel, s.eps); err != nil {
				return err
			}
			if err := encAddBF16To(enc, outBuf, normSlab, outBuf, outBase, 0, outBase, rows*s.dModel); err != nil {
				return err
			}
			if rec := s.verifyTailRec; rec.recording(li) {
				rec.recordGemvBatched(residentBytes(pl.gate.Packed), 0, outBuf, outBase, gateSlab, 0, s.pliDim, s.dModel, rows)
				rec.recordGeluGateMul(gateSlab, pleSlabBuf, multSlab, 0, pliBase, 0, rows*s.pliDim)
				rec.recordGemvBatched(residentBytes(pl.proj.Packed), 0, multSlab, 0, projSlab, 0, s.dModel, s.pliDim, rows)
				rec.recordRMSRows(projSlab, residentBytes(pl.postNorm), normSlab, 0, 0, 0, rows, s.dModel, s.eps)
				rec.recordAdd(outBuf, outBase, normSlab, 0, outBuf, outBase, rows*s.dModel)
			}
		} else {
			gateGroupSize, gateBits, err := validatePerLayerInputGateQuantWeight("gate", pl.gate, s.pliDim, s.dModel, pl.groupSize, pl.bits)
			if err != nil {
				return err
			}
			projGroupSize, projBits, err := validatePerLayerInputGateQuantWeight("projection", pl.proj, s.dModel, s.pliDim, pl.groupSize, pl.bits)
			if err != nil {
				return err
			}
			gatePacked, gateScales, gateBiases := quantWeightViews(pl.gate)
			projPacked, projScales, projBiases := quantWeightViews(pl.proj)
			pliBase := uint(li * rows * s.pliDim * bf16Size)
			// MTP verify blocks (small K) collapse the chain's five dispatches to
			// three: gate+gelu fused, the proj qmm_t, rms+add fused — the stage is
			// launch-bound there (#372: the gpu-trace bucket held 3.8ms at K=5 AND
			// K=33). Same rounding stations; the gate matvec moves from qmm_t's
			// MMA order to the qgemv order (the fold's token-identity tier).
			// handled=false (older metallib) keeps the composed chain below.
			// PROMPT scale keeps the composed chain: extending the fused chain to
			// K=512 was tried and FALSIFIED (#367: 2578 vs 2901 tok/s
			// prefill, −11% — the qgemv-order gate that wins launch-bound small-K
			// loses to qmm_t's MMA order at prompt K). A prompt-scale fusion needs
			// an MMA-order fused gate kernel, not this one.
			fusedChain := false
			if s.verifyFoldSmallK {
				handled, gerr := encPLEGateGeluRows(enc, gatePacked, gateScales, gateBiases,
					outBuf, outBase, pleSlabBuf, pliBase, multSlab, 0, rows, s.dModel, s.pliDim, gateGroupSize, gateBits)
				if gerr != nil {
					return gerr
				}
				if handled {
					if err := encQMMTBF16At(enc, projPacked.buf, projScales.buf, projBiases.buf, multSlab, projSlab, projPacked.off, projScales.off, projBiases.off, 0, 0, rows, s.dModel, s.pliDim, projGroupSize, projBits); err != nil {
						return err
					}
					if err := encRMSNormResidualRowsBF16At(enc, projSlab, residentBytes(pl.postNorm), outBuf, outBuf, 0, 0, outBase, outBase, rows, s.dModel, s.eps); err != nil {
						return err
					}
					if rec := s.verifyTailRec; rec.recording(li) {
						rec.recordPLEGateGeluRows(gatePacked, gateScales, gateBiases, outBuf, outBase, pleSlabBuf, pliBase, multSlab, 0, rows, s.dModel, s.pliDim, gateGroupSize, gateBits)
						rec.recordQMMT(projPacked.buf, projScales.buf, projBiases.buf, multSlab, projSlab, projPacked.off, projScales.off, projBiases.off, 0, 0, rows, s.dModel, s.pliDim, projGroupSize, projBits)
						rec.recordRMSNormResidualRows(projSlab, residentBytes(pl.postNorm), outBuf, outBuf, 0, 0, outBase, outBase, rows, s.dModel, s.eps)
					}
					fusedChain = true
				}
			}
			if !fusedChain {
				if err := encQMMTBF16At(enc, gatePacked.buf, gateScales.buf, gateBiases.buf, outBuf, gateSlab, gatePacked.off, gateScales.off, gateBiases.off, outBase, 0, rows, s.pliDim, s.dModel, gateGroupSize, gateBits); err != nil {
					return err
				}
				if err := encGeluGateMulFusedTo(enc, gateSlab, pleSlabBuf, multSlab, 0, pliBase, 0, rows*s.pliDim); err != nil {
					return err
				}
				if err := encQMMTBF16At(enc, projPacked.buf, projScales.buf, projBiases.buf, multSlab, projSlab, projPacked.off, projScales.off, projBiases.off, 0, 0, rows, s.dModel, s.pliDim, projGroupSize, projBits); err != nil {
					return err
				}
				if err := encRMSNormRowsBF16(enc, projSlab, residentBytes(pl.postNorm), normSlab, 0, 0, 0, rows, s.dModel, s.eps); err != nil {
					return err
				}
				if err := encAddBF16To(enc, outBuf, normSlab, outBuf, outBase, 0, outBase, rows*s.dModel); err != nil {
					return err
				}
				if rec := s.verifyTailRec; rec.recording(li) {
					rec.recordQMMT(gatePacked.buf, gateScales.buf, gateBiases.buf, outBuf, gateSlab, gatePacked.off, gateScales.off, gateBiases.off, outBase, 0, rows, s.pliDim, s.dModel, gateGroupSize, gateBits)
					rec.recordGeluGateMul(gateSlab, pleSlabBuf, multSlab, 0, pliBase, 0, rows*s.pliDim)
					rec.recordQMMT(projPacked.buf, projScales.buf, projBiases.buf, multSlab, projSlab, projPacked.off, projScales.off, projBiases.off, 0, 0, rows, s.dModel, s.pliDim, projGroupSize, projBits)
					rec.recordRMSRows(projSlab, residentBytes(pl.postNorm), normSlab, 0, 0, 0, rows, s.dModel, s.eps)
					rec.recordAdd(outBuf, outBase, normSlab, 0, outBuf, outBase, rows*s.dModel)
				}
			}
		}
	}
	if s.lb[li].layerScalar != nil { // gemma4 per-layer output scalar, all rows in one dispatch
		if err := encMulRowsBF16(enc, outBuf, s.lb[li].layerScalar, outBuf, outBase, 0, outBase, rows, s.dModel); err != nil {
			return err
		}
		if rec := s.verifyTailRec; rec.recording(li) {
			rec.recordMulRows(outBuf, s.lb[li].layerScalar, outBuf, outBase, 0, outBase, rows, s.dModel)
		}
	}
	return nil
}

// archQ8ICBBlocks mirrors decode_forward_arch_icb.go's sdpa2PassICBBlocks
// EXACTLY: the recorded ICB replay bakes ONE 2-pass-or-not decision (and block
// count) for EVERY q8 GLOBAL layer from the session's maxLen at construction —
// unconditionally, from position 0, NEVER re-derived from the live n — fanned
// to the worst-served global layer (fewest KV heads, #365). #53/#54's own bug:
// encSDPADecodeQ8At (and this whole file's other q8 SDPA call sites) instead
// pick 2-pass from the LIVE n, which only agrees with the replay once n itself
// crosses sdpa2PassMinKV — below that knee the replay is ALREADY on the 2-pass
// ladder (safe at any smaller live n, per the replay's own comment: a block
// whose strided walk starts past n writes finite_min/0 partials the pass-2
// merge zeroes) while an n-keyed re-encode is still single-pass — two
// different kernels computing the "same" softmax, not bit-identical. A fresh
// per-row re-encode standing in for the replay (this file's interleave) must
// therefore key off maxLen, not n, or its bytes drift from the replay by the
// same token-identity-tier margin the #55 qmm_t fold already carries — just
// from the ATTENTION side instead of the projections.
func (s *archDecodeState) archQ8ICBBlocks() int {
	if s.maxLen < sdpa2PassMinKV {
		return 0
	}
	minKV := 0
	for li := range s.specs {
		if s.specs[li].Attention != model.GlobalAttention {
			continue
		}
		if kv := kvHeadsOf(s.specs[li], s.nKVHeads); minKV == 0 || kv < minKV {
			minKV = kv
		}
	}
	if minKV == 0 {
		return 0
	}
	return int(sdpa2PassBlocks(s.maxLen, minKV))
}

// interleaveKVQ8Usable reports whether the per-row interleave (foldAttn=false,
// the small-K lane the byte-exact MTP verify takes — #53/#54) can serve this
// K-row batch under q8 ICB caches. The interleave never lands a whole chunk
// before any row reads (the fold's land-before-read contract, #15), so a
// bidirectional row cap is never servable here — that keeps declining to the
// existing fold-or-sequential routing unchanged. Otherwise the interleave's q8
// needs are exactly the per-row q8 kernels: encKVQ8StoreRows at rows=1 (the
// fold's own K-row landing, at N=1 — see encAttnHalfKVQ8InputAt below) and
// encSDPADecodeQ8ForcedBlocksAt keyed on archQ8ICBBlocks (matching the
// replay's OWN maxLen-keyed choice, not the live n — see archQ8ICBBlocks) for
// every layer that touches a q8 cache, owner or sharer. None of the fold's
// rows-batched projection/rope/multiQ/gelu kernels are needed here.
func (s *archDecodeState) interleaveKVQ8Usable(basePos, K int) bool {
	if s.icb == nil || !s.icb.hasKVQ8() || s.rowAttnCaps != nil || !gpuHasKVQ8StoreRows() {
		return false
	}
	blocks := s.archQ8ICBBlocks()
	for li := range s.specs {
		ownIdx := li
		if !s.specs[li].OwnsCache() {
			ownIdx = s.specs[li].KVShareFrom
		}
		if ownIdx < 0 || ownIdx >= len(s.specs) || !s.icb.kvQ8.on(ownIdx) {
			continue // this layer never touches a q8 cache
		}
		lhd := headDimOf(s.specs[li], s.headDim)
		if blocks > 0 {
			if _, perr := sdpaVector2Pass1Q8Pipeline(lhd, int32(blocks)); perr != nil {
				return false
			}
			if _, perr := sdpaVector2Pass2PipelineForHeadDim(lhd); perr != nil {
				return false
			}
		} else if _, perr := sdpaVectorQ8Pipeline(lhd); perr != nil {
			return false
		}
		if s.specs[li].OwnsCache() && s.lb[li].kNorm.buf == nil {
			return false // per-row landing needs the owner's kNorm, exactly as the fold requires
		}
	}
	return s.asc.kProj != nil && s.asc.vProj != nil
}

// encSDPADecodeQ8ForcedBlocksAt is encSDPADecodeQ8At but takes the 2-pass
// block count as an EXTERNAL decision (archQ8ICBBlocks) instead of deriving it
// from the live n — see archQ8ICBBlocks for why: the recorded ICB replay's
// choice is keyed on the session's maxLen, fixed at construction, not on n.
// blocks<=0 selects the single-pass kernel; blocks>0 selects the 2-pass pair
// at exactly that block count — the pass-2 merge zeroes any block whose
// strided walk starts past n, so a live n smaller than the blocks ladder
// expects is always safe (the replay's own safety argument).
func encSDPADecodeQ8ForcedBlocksAt(enc metal.MTLComputeCommandEncoder, sc attnScratch, q metal.MTLBuffer, qOff uint, k, v metal.MTLBuffer, kScales, vScales metal.MTLBuffer, out metal.MTLBuffer, outOff uint, nHeads, nKVHeads, headDim, n, blocks int, kHeadStride, kSeqStride, vHeadStride, vSeqStride int64, scale float32) error {
	if blocks > 0 && sc.p2Partials != nil {
		pso1, err := sdpaVector2Pass1Q8Pipeline(headDim, int32(blocks))
		if err != nil {
			return err
		}
		pso2, err := sdpaVector2Pass2PipelineForHeadDim(headDim)
		if err != nil {
			return err
		}
		sink := encSink{enc}
		emitSDPAVector2Pass1Q8At(sink, pso1, q, qOff, k, v, sc.p2Partials, sc.p2Sums, sc.p2Maxs, kScales, vScales, 0, 0, nil, nHeads, nKVHeads, n, blocks, kHeadStride, kSeqStride, vHeadStride, vSeqStride, scale)
		emitSDPA2Pass2At(sink, pso2, sc.p2Partials, sc.p2Sums, sc.p2Maxs, out, outOff, 1, nHeads, blocks)
		return nil
	}
	pso, err := sdpaVectorQ8Pipeline(headDim)
	if err != nil {
		return err
	}
	emitSDPAVectorQ8At(encSink{enc}, pso, q, qOff, k, v, out, outOff, kScales, vScales, 0, 0, nil, nHeads, nKVHeads, n, kHeadStride, kSeqStride, vHeadStride, vSeqStride, scale)
	return nil
}

// encAttnHalfKVQ8InputAt is encAttnHalfKVInputAt (decode_step.go) for a q8 ICB
// OWNER row: K/V project into the session's shared single-row scratch
// (sc.kProj/sc.vProj — always allocated, see newAttnScratch) instead of
// straight into the cache, norm+rope/value-norm run there exactly as the bf16
// path runs them on the cache row in place, then ONE encKVQ8StoreRows
// (rows=1) quantises the staged row into the int8 cache + f32 group scale at
// this row's own offset — the SAME store math (lthn_kv_q8_store_rows_bf16)
// the K-row fold's landing already calls, just at N=1. The SDPA read is
// encSDPADecodeQ8At — the SAME q8 kernel pair the fold's own per-row corner
// already calls (line ~1434) — so a q8-owner interleave row is byte-identical
// to sequential stepID: every OTHER op (Q projection/rope, O projection,
// residual) is the UNCHANGED proj.project/encQKNormRopeAt/encResidualMaybeNormAt
// single-row call encAttnHalfKVInputAt itself makes. q8 only arms on GLOBAL
// owner layers (TestKVQ8ICBDecodeTracksBF16 pins this), so there is no ring
// slot / sliding-window arithmetic to carry here.
func encAttnHalfKVQ8InputAt(
	enc metal.MTLComputeCommandEncoder,
	x metal.MTLBuffer, xOff uint, kCacheBuf, vCacheBuf, kScales, vScales metal.MTLBuffer, offBuf, h metal.MTLBuffer, hOff, offOff uint,
	attnNormW, postAttnNorm, qNorm, kNorm bufView, valueNorm metal.MTLBuffer,
	sc attnScratch, proj projector,
	dModel, nHeads, nKVHeads, headDim, pos, rotaryDim, q8Blocks int, base, scale, ropeScale, eps float32,
	ropeFreqs metal.MTLBuffer,
) error {
	if sc.kProj == nil || sc.vProj == nil {
		return core.NewError("native.encAttnHalfKVQ8InputAt: q8 landing needs kProj/vProj scratch")
	}
	kvDim := nKVHeads * headDim
	if err := encRMSNormBF16At(enc, x, attnNormW.buf, sc.normed, xOff, attnNormW.off, 0, dModel, eps); err != nil {
		return err
	}
	// query: unchanged from encAttnHalfKVInputAt — the head projection never touches the KV cache.
	if err := proj.project(enc, sc.normed, sc.q, 0, projQ); err != nil {
		return err
	}
	if gpuHasGeluKernel() && qNorm.buf != nil {
		if err := encQKNormRopeAt(enc, sc.q, qNorm.buf, sc.q, 0, qNorm.off, 0, offBuf, offOff, ropeFreqs, nHeads, headDim, rotaryDim, base, ropeScale, eps); err != nil {
			return err
		}
	} else {
		if qNorm.buf != nil {
			if err := encRMSNormRowsBF16(enc, sc.q, qNorm.buf, sc.q, 0, qNorm.off, 0, nHeads, headDim, eps); err != nil {
				return err
			}
		}
		if err := encRopeDecodeAt(enc, sc.q, sc.q, 0, 0, offBuf, offOff, ropeFreqs, nHeads, headDim, rotaryDim, base, ropeScale); err != nil {
			return err
		}
	}
	// key: project into the STAGING row (not the int8 cache), norm+rope there —
	// identical maths to the bf16 path norming/roping the cache row in place.
	if err := proj.project(enc, sc.normed, sc.kProj, 0, projK); err != nil {
		return err
	}
	if gpuHasGeluKernel() && kNorm.buf != nil {
		if err := encQKNormRopeAt(enc, sc.kProj, kNorm.buf, sc.kProj, 0, kNorm.off, 0, offBuf, offOff, ropeFreqs, nKVHeads, headDim, rotaryDim, base, ropeScale, eps); err != nil {
			return err
		}
	} else {
		if kNorm.buf != nil {
			if err := encRMSNormRowsBF16(enc, sc.kProj, kNorm.buf, sc.kProj, 0, kNorm.off, 0, nKVHeads, headDim, eps); err != nil {
				return err
			}
		}
		if err := encRopeDecodeAt(enc, sc.kProj, sc.kProj, 0, 0, offBuf, offOff, ropeFreqs, nKVHeads, headDim, rotaryDim, base, ropeScale); err != nil {
			return err
		}
	}
	// value: project STRAIGHT into the staging row (no rotation) — gemma4 K==V
	// layers carry no v_proj, so project via wK exactly as encAttnHalfKVInputAt does.
	vIdx := projV
	if !proj.hasV() {
		vIdx = projK
	}
	if err := proj.project(enc, sc.normed, sc.vProj, 0, vIdx); err != nil {
		return err
	}
	if valueNorm != nil {
		if err := encRMSNormRowsBF16(enc, sc.vProj, valueNorm, sc.vProj, 0, 0, 0, nKVHeads, headDim, eps); err != nil {
			return err
		}
	}
	// the quantise-store hop: rows=1, the fold's own K-row store at N=1 — same
	// kernel, same per-group scale formula, only the row count differs.
	rowOff := uint(pos * kvDim)
	scaleOff := uint(pos * (kvDim / kvQ8GroupSize) * 4)
	if err := encKVQ8StoreRows(enc, sc.kProj, kCacheBuf, rowOff, kScales, scaleOff, 1, kvDim); err != nil {
		return err
	}
	if err := encKVQ8StoreRows(enc, sc.vProj, vCacheBuf, rowOff, vScales, scaleOff, 1, kvDim); err != nil {
		return err
	}
	if err := encSDPADecodeQ8ForcedBlocksAt(enc, sc, sc.q, 0, kCacheBuf, vCacheBuf, kScales, vScales, sc.attn, 0,
		nHeads, nKVHeads, headDim, pos+1, q8Blocks,
		int64(headDim), int64(kvDim), int64(headDim), int64(kvDim), scale); err != nil {
		return err
	}
	if err := proj.project(enc, sc.attn, sc.attnOut, 0, projO); err != nil {
		return err
	}
	return encResidualMaybeNormAt(enc, x, xOff, sc.attnOut, 0, sc.normed, h, hOff, postAttnNorm, dModel, eps)
}

// encAttnHalfSharedKVQ8At is encAttnHalfSharedInputAt (decode_forward_arch.go)
// attending a q8-armed OWNER's cache: the sharer never lands its own K/V (it
// has none), so the only change from the bf16 twin is the SDPA read —
// encSDPADecodeQ8At against the owner's int8 K/V + group scales, the SAME
// kernel pair a q8 owner's own row reads (encAttnHalfKVQ8InputAt above) and
// the fold's per-row corner already calls. gemma3n E2B/E4B's trailing
// shared-KV layers are exactly why this twin exists — without it a sharer of
// a q8 owner would read int8 bytes through the plain bf16 SDPA kernel.
func encAttnHalfSharedKVQ8At(
	enc metal.MTLComputeCommandEncoder,
	x metal.MTLBuffer, xOff uint, attendK, attendV, kScales, vScales, offBuf, h metal.MTLBuffer, hOff, offOff uint,
	attnNormW, postAttnNorm, qNorm bufView,
	sc attnScratch, proj projector,
	dModel, nHeads, nKVHeads, headDim, pos, rotaryDim, q8Blocks int, base, scale, ropeScale, eps float32,
	ropeFreqs metal.MTLBuffer,
) error {
	kvDim := nKVHeads * headDim
	if err := encRMSNormBF16At(enc, x, attnNormW.buf, sc.normed, xOff, attnNormW.off, 0, dModel, eps); err != nil {
		return err
	}
	if err := proj.project(enc, sc.normed, sc.q, 0, projQ); err != nil {
		return err
	}
	if gpuHasGeluKernel() && qNorm.buf != nil {
		if err := encQKNormRopeAt(enc, sc.q, qNorm.buf, sc.q, 0, qNorm.off, 0, offBuf, offOff, ropeFreqs, nHeads, headDim, rotaryDim, base, ropeScale, eps); err != nil {
			return err
		}
	} else {
		if qNorm.buf != nil {
			if err := encRMSNormRowsBF16(enc, sc.q, qNorm.buf, sc.q, 0, qNorm.off, 0, nHeads, headDim, eps); err != nil {
				return err
			}
		}
		if err := encRopeDecodeAt(enc, sc.q, sc.q, 0, 0, offBuf, offOff, ropeFreqs, nHeads, headDim, rotaryDim, base, ropeScale); err != nil {
			return err
		}
	}
	if err := encSDPADecodeQ8ForcedBlocksAt(enc, sc, sc.q, 0, attendK, attendV, kScales, vScales, sc.attn, 0,
		nHeads, nKVHeads, headDim, pos+1, q8Blocks,
		int64(headDim), int64(kvDim), int64(headDim), int64(kvDim), scale); err != nil {
		return err
	}
	if err := proj.project(enc, sc.attn, sc.attnOut, 0, projO); err != nil {
		return err
	}
	return encResidualMaybeNormAt(enc, x, xOff, sc.attnOut, 0, sc.normed, h, hOff, postAttnNorm, dModel, eps)
}

func (s *archDecodeState) stepTokensBatchedDenseResultWithInputViewsPLE(embs [][]byte, pleSlab []byte, basePos int, readResult, readLastOnly bool, lastDst []byte, dstRows [][]byte, directInputs bool) (out [][]byte, ok bool, err error) {
	K := len(embs)
	if K == 0 {
		return nil, false, core.NewError("native.stepTokensBatchedDense: empty batch")
	}
	decline := func(why string) ([][]byte, bool, error) {
		if mtpDiagForTest {
			nativeTraceLog("mtp-diag batched-dense decline: " + why + "\n")
		}
		return nil, false, nil
	}
	// dense uniform guard: every layer owns its cache + is non-MoE; no trace (per-layer host reads).
	// The PLE gate is NOT a host flush (it is an encoded kernel chain reading a per-token input
	// buffer — bf16 gemv or quant qmv), so a PLE arch batches when the caller supplies the K-token
	// slab; without one it declines to the proven sequential fallback.
	if s.trace {
		return decline("trace")
	}
	// State-lane TurboQuant (tq_kv_state.go): the batched pass is not TQ-aware
	// for this carrier — its landings would write bf16 rows where the decode
	// reads packed codes. Decline wholesale; the per-token stepToken path is
	// TQ-correct by construction (correct-but-sequential prefill, v1).
	if s.tqStateArmed() {
		return decline("state turboquant caches: batched pass not TQ-aware for the state carrier — per-token prefill")
	}
	// recorded-ICB sessions (the quant decode lane): the replay owns the LIVE per-layer caches —
	// s.lb is ring-sized and UNUSED there, so the batch must read and write the replay's own
	// buffers or its rows would be invisible to every later replayed step. The per-row interleave's
	// slot math (pos%slideW ring / linear global) matches prepareStepRebind's pos%cacheRows exactly
	// when the capacities line up (checked per owner below); the folds stay off in this mode so no
	// staged/deferred lane ever touches the other cache set. Small batches only — the MTP verify
	// and accept-commit blocks — the prompt prefill keeps the replay's own pipelined lane.
	var icbK, icbV []metal.MTLBuffer
	if s.icb != nil {
		// Small batches (MTP verify / accept-commit / short turn-appends) stay on the per-row
		// interleave — byte-identical to the replay's chained lane, which keeps the save/restore
		// contract exact (a RESTORED session's appends run chained; the live session must write
		// the same bytes). Prompt-scale batches take the qmm fold (token-identity tier) — both
		// the live and the restored session route those identically by size.
		foldable := !batchedMLPFoldDisabledForTest && K > batchedDenseICBMaxRows && gpuHasGeluKernel()
		if s.icb.hasKVTQ() {
			// TurboQuant ICB caches (#48): the batched pass lands the chunk into
			// the code caches (encTQKVStoreRows) and scores the ordinary bf16
			// SDPA against a per-layer scratch reconstructed from the prior codes
			// (tqBatchedLanding, decode_batched_tq.go). It needs the whole-chunk
			// batched-rope landing (the fold shape); anything below the fold gate,
			// a bidirectional span, or a missing kernel declines to the per-token
			// replay, which is TQ-aware by recording.
			if batchedTQPrefillDisabledForTest || tqBatchedPrefillEnvOff || batchedMLPFoldDisabledForTest ||
				batchedRopeDisabledForTest || batchedEpilogueDisabledForTest || sdpaMultiQDisabledForTest ||
				!gpuHasQKNormRopeRows() || !gpuHasMulRowsKernel() {
				return decline("turboquant icb caches: batched prefill disabled or fold prerequisites")
			}
			if !(K > batchedDenseICBMaxRows) || K <= 1 || !gpuHasGeluKernel() {
				return decline("turboquant icb caches: batch below the fold gate — per-token replay")
			}
			if s.rowAttnCaps != nil {
				return decline("turboquant icb caches: bidirectional span — per-token replay")
			}
			if !s.tqBatchedPrefillUsable() {
				return decline("turboquant icb caches: store/dequant kernel or fold geometry unservable")
			}
		}
		if s.icb.hasKVQ8() {
			// q8 ICB caches (#367 slice C): the pass is q8-aware ONLY on the
			// rows-batched fold — landings stage + quantise (encKVQ8StoreRows)
			// and the reads ride the multi-query causal q8 kernel, or the
			// per-row q8 ladder when bidirectional row caps disable multiQ
			// (image/video spans — the chunk-wide q8 landing satisfies their
			// land-before-read rule, #15). Anything that would route a q8 layer
			// through a bf16 write or an un-instantiated read declines to the
			// per-token replay (q8-aware by recording): the per-row lanes, the
			// deep+small 2-pass corner, and the test levers that force per-row
			// shapes.
			if batchedMLPFoldDisabledForTest || batchedRopeDisabledForTest || batchedEpilogueDisabledForTest ||
				sdpaMultiQDisabledForTest ||
				!gpuHasQKNormRopeRows() || !gpuHasKVQ8StoreRows() || !gpuHasMulRowsKernel() {
				return decline("q8 icb caches: fold prerequisites")
			}
			if !(K > batchedDenseICBMaxRows || s.verifyFoldSmallK) || K <= 1 || !gpuHasGeluKernel() {
				// #53/#54: small batches used to decline WHOLESALE here — the per-row
				// interleave (below, foldAttn=false) wrote bf16 rows straight into the
				// int8 caches, so a per-row MTP verify (LTHN_MTP_VERIFY_FOLD=0, the
				// forensics lane — both verify tiers fold by default) paid K
				// sequential stepID rounds instead of one command buffer. The
				// interleave now
				// lands q8 rows itself (encAttnHalfKVQ8InputAt / the sharer twin
				// encAttnHalfSharedKVQ8At, below) with encKVQ8StoreRows at rows=1 —
				// the SAME store math the fold's own K-row landing calls — and reads
				// them back through encSDPADecodeQ8At, the SAME per-row q8 SDPA the
				// fold's own per-row corner already uses (line ~1434). Every OTHER
				// projection stays on proj.project (the per-row qmv kernel), so this
				// path is byte-identical to sequential decode, unlike the fold's qmm_t
				// token-identity tier (#55) — that is exactly why it may serve the
				// byte-exact lane where the fold must not (TestMtpVerifyFoldArmed_Good).
				// interleaveKVQ8Usable is the narrower prerequisite check for THIS path
				// (per-row q8 kernels only — none of the fold's rows-batched/multiQ/gelu
				// requirements above); anything it can't serve keeps declining exactly
				// as before, unchanged for the cached-prefix prefill paths this gate
				// protects.
				if !s.interleaveKVQ8Usable(basePos, K) {
					return decline("q8 icb caches: batch below the fold gate")
				}
			}
			for li := range s.specs {
				ownIdx := li
				if !s.specs[li].OwnsCache() {
					ownIdx = s.specs[li].KVShareFrom
				}
				if ownIdx < 0 || ownIdx >= len(s.specs) || !s.icb.kvQ8.on(ownIdx) {
					continue // this layer never touches a q8 cache
				}
				lhd := headDimOf(s.specs[li], s.headDim)
				if s.rowAttnCaps != nil {
					// bidirectional row caps: multiQ is structurally off (useMultiQ
					// requires nil caps), so every row reads the per-row q8 ladder
					// (encSDPADecodeQ8At) against the chunk-wide q8 landing. Require
					// both rungs up front — a row's cap reaches at most the chunk
					// end, so the 2-pass rung matters exactly when basePos+K passes
					// the knee.
					if _, perr := sdpaVectorQ8Pipeline(lhd); perr != nil {
						return decline("q8 icb caches: no per-row q8 kernel for head dim")
					}
					if basePos+K >= sdpa2PassMinKV {
						if _, perr := sdpaVector2Pass1Q8Pipeline(lhd, sdpa2PassBlocks(basePos+K, kvHeadsOf(s.specs[li], s.nKVHeads))); perr != nil {
							return decline("q8 icb caches: no 2-pass q8 kernel for head dim")
						}
						if _, perr := sdpaVector2Pass2PipelineForHeadDim(lhd); perr != nil {
							return decline("q8 icb caches: no 2-pass merge kernel for head dim")
						}
					}
				} else if !gpuHasSDPAMultiQQ8(lhd) || !gpuHasSDPAMultiQ(lhd) {
					return decline("q8 icb caches: no multiQ q8 kernel for head dim")
				}
				if li >= len(s.lb) || s.lb[li].proj == nil || !s.lb[li].proj.rowsCapable() ||
					s.lb[li].qNorm.buf == nil {
					return decline("q8 icb caches: layer not fold-capable")
				}
				if s.specs[li].OwnsCache() && s.lb[li].kNorm.buf == nil {
					return decline("q8 icb caches: owner missing kNorm (per-row landing shape)")
				}
				if !(basePos+K < sdpa2PassMinKV || K >= steelGEMMMinRows) {
					// the deep+small corner (MTP verify past the knee) rides the
					// per-row 2-pass q8 — require the pipeline pair up front so a
					// mid-encode resolve failure can never half-encode a layer.
					if _, perr := sdpaVector2Pass1Q8Pipeline(lhd, sdpa2PassBlocks(basePos+K, kvHeadsOf(s.specs[li], s.nKVHeads))); perr != nil {
						return decline("q8 icb caches: no 2-pass q8 kernel for head dim")
					}
					if _, perr := sdpaVector2Pass2PipelineForHeadDim(lhd); perr != nil {
						return decline("q8 icb caches: no 2-pass merge kernel for head dim")
					}
				}
			}
		}
		if (K > batchedDenseICBMaxRows && !foldable) ||
			len(s.icb.kCaches) < len(s.specs) || len(s.icb.vCaches) < len(s.specs) || len(s.icb.cacheRows) < len(s.specs) {
			return decline(core.Sprintf("icb caches: K=%d kCaches=%d vCaches=%d cacheRows=%d specs=%d",
				K, len(s.icb.kCaches), len(s.icb.vCaches), len(s.icb.cacheRows), len(s.specs)))
		}
		icbK, icbV = s.icb.kCaches, s.icb.vCaches
	}
	if icbK == nil { // non-ICB batches read+write the linear lb caches (deferred on sessions)
		if err := s.ensureLBKVCaches(); err != nil {
			return nil, false, err
		}
	}
	if len(s.ple) > 0 {
		if pleSlab == nil && s.prefillPLESlabDevice == nil {
			return decline("PLE arch without slab")
		}
		// a skipped chunk's slab carries only the owner layers' slices (#381) —
		// the bounded layer loop below never reads past prefillSkipToLayer. The
		// device tensor's geometry is the builder's outLayers contract.
		if pleSlab != nil {
			wantLayers := len(s.specs)
			if s.prefillSkipToLayer > 0 && s.prefillSkipToLayer < wantLayers {
				wantLayers = s.prefillSkipToLayer
			}
			if want := K * wantLayers * s.pliDim * bf16Size; len(pleSlab) != want {
				return nil, false, core.NewError("native.stepTokensBatchedDense: PLE slab size mismatch")
			}
		}
	} else if pleSlab != nil {
		return nil, false, core.NewError("native.stepTokensBatchedDense: PLE slab supplied for a non-PLE arch")
	}
	// Quant MoE layers batch through the K-row MoE block (moe_batch.go) at prompt scale —
	// the fold lane only (small batches keep the per-row interleave's byte contract, and MoE
	// has no per-row interleave, so they fall to per-token stepping as before). The MTP
	// verify (verifyFoldSmallK, K>1 so the fold slabs exist) takes the block at small K too:
	// without it the 26B verify ran K sequential full steps — 35.7ms for K=5 against a 7ms
	// plain step — and the pair HALVED throughput at 81% acceptance (#354). bf16 MoE
	// still declines.
	batchMoE := (K > batchedDenseICBMaxRows || (s.verifyFoldSmallK && K > 1)) && !batchedMLPFoldDisabledForTest && gpuHasGeluKernel()
	for li := range s.specs {
		if s.specs[li].MoE {
			if !batchMoE || li >= len(s.moeQuant) || s.moeQuant[li] == nil || !s.batchedMoEUsable(s.moeQuant[li]) ||
				li >= len(s.lb) || s.lb[li].proj == nil || !s.lb[li].proj.rowsCapable() {
				return decline(core.Sprintf("layer %d: quant MoE not batchable", li))
			}
		} else if li < len(s.moeQuant) && s.moeQuant[li] != nil {
			return decline(core.Sprintf("layer %d: moeQuant on non-MoE spec", li))
		}
		if li < len(s.moeWeights) && s.moeWeights[li] != nil {
			return decline(core.Sprintf("layer %d: bf16 MoE", li))
		}
		if s.specs[li].OwnsCache() {
			if icbK == nil {
				continue
			}
			if icbK[li] == nil || icbV[li] == nil {
				return decline(core.Sprintf("layer %d: icb owner cache nil (k=%v v=%v)", li, icbK[li] == nil, icbV[li] == nil))
			}
			rows := s.icb.cacheRows[li]
			if s.specs[li].Attention == model.SlidingAttention && s.slidingWindow > 0 {
				// the interleave's slot math is pos%slidingWindow: identical to the replay's
				// pos%cacheRows ring when rows==slidingWindow, and to its linear write when
				// the allocation is un-bounded (rows>=maxLen ⇒ every pos writes linearly).
				if rows != s.slidingWindow && rows < s.maxLen {
					return decline(core.Sprintf("layer %d: sliding rows=%d window=%d maxLen=%d", li, rows, s.slidingWindow, s.maxLen))
				}
			} else if rows < basePos+K {
				return decline(core.Sprintf("layer %d: cache rows=%d < basePos+K=%d", li, rows, basePos+K))
			}
			continue
		}
		// shared-KV layers (gemma4 E2B/E4B tails) attend an OWNER's cache: batchable —
		// the owner's rows for this batch are encoded at a lower layer index in the same
		// command buffer — provided the owner's caches (live set) are resident.
		own := s.specs[li].KVShareFrom
		if own < 0 || own >= len(s.specs) {
			return decline(core.Sprintf("layer %d: KVShareFrom=%d out of range", li, own))
		}
		if icbK != nil {
			if icbK[own] == nil || icbV[own] == nil {
				return decline(core.Sprintf("layer %d: icb shared owner %d cache nil", li, own))
			}
		} else if own >= len(s.lb) || s.lb[own].kCache == nil || s.lb[own].vCache == nil {
			return decline(core.Sprintf("layer %d: lb shared owner %d cache nil", li, own))
		}
	}
	if s.prefillEmbedDevice == nil {
		for i := range embs {
			if len(embs[i]) != s.dModel*bf16Size {
				return nil, false, core.NewError("native.stepTokensBatchedDense: emb must be dModel bf16 bytes")
			}
		}
	}
	syncStart := time.Now()
	if icbK == nil { // ICB sessions decode over the replay's caches; the paged pool is bypassed
		if err := s.syncLinearKVFromDevicePaged(basePos); err != nil {
			return nil, false, err
		}
	}
	hostSpan("syncKV", syncStart, K)

	rowBytes := s.dModel * bf16Size
	var (
		lastOutBuf    metal.MTLBuffer
		directLastOut bool
	)
	if readResult && readLastOnly {
		if cap(lastDst) < rowBytes {
			lastDst = make([]byte, rowBytes)
		} else {
			lastDst = lastDst[:rowBytes]
		}
		if tmp, ok := s.denseBatch.lastOutputView(lastDst); ok {
			lastOutBuf = tmp
			directLastOut = true
		}
	}
	var (
		directOutputRows      []metal.MTLBuffer
		directOutputOff       []uint
		usingDirectOutputRows bool
	)
	// K-wide working rows (ping-ponged across layers) + per-row position buffers, retained on the state.
	inRows, outRows, offBuf, offPtr, offOff, rowOff := s.denseBatch.rows(K, s.dModel)
	readRows, readOff := inRows, rowOff
	directInputRows, directInputOff := s.denseBatch.readRowsFor(K)
	var inputViews []cachedNoCopyBytesView
	if directInputs {
		inputViews = s.denseBatch.inputViewsFor(K)
	}
	usingDirectInputRows := false
	for i := range K {
		*offPtr[i] = int32(basePos + i)
		if s.prefillEmbedDevice != nil {
			// device-gathered embed rows (#381): the builder's buffer, row i at
			// i·rowBytes — GPU-ordered behind the gather on the shared queue.
			directInputRows[i] = s.prefillEmbedDevice
			directInputOff[i] = uint(i * rowBytes)
			usingDirectInputRows = true
			continue
		}
		if directInputs {
			if buf, direct := inputViews[i].buffer(embs[i]); direct {
				directInputRows[i] = buf
				directInputOff[i] = 0
				usingDirectInputRows = true
				continue
			}
		}
		directInputRows[i] = inRows[i]
		directInputOff[i] = rowOff[i]
		off := int(rowOff[i])
		copy(unsafe.Slice((*byte)(inRows[i].Contents()), off+rowBytes)[off:], embs[i])
	}
	if usingDirectInputRows {
		readRows, readOff = directInputRows, directInputOff
	}
	if readResult && !readLastOnly && len(dstRows) >= K {
		directOutputRows, directOutputOff = s.denseBatch.directOutputRowsFor(K)
		outputViews := s.denseBatch.outputViewsFor(K)
		usingDirectOutputRows = true
		for i := range K {
			if cap(dstRows[i]) < rowBytes {
				usingDirectOutputRows = false
				break
			}
			dstRows[i] = dstRows[i][:rowBytes]
			buf, direct := outputViews[i].buffer(dstRows[i])
			if !direct {
				usingDirectOutputRows = false
				break
			}
			directOutputRows[i] = buf
			directOutputOff[i] = 0
		}
	}

	var pleSlabBuf metal.MTLBuffer
	if s.prefillPLESlabDevice != nil {
		// device-resident build (#381): committed on the shared queue ahead of this
		// pass — GPU-ordered, no host wait, no copy-out, no re-upload.
		pleSlabBuf = s.prefillPLESlabDevice
	} else if len(pleSlab) > 0 {
		if pleSlabBuf, err = s.pleSlabBuffer(pleSlab); err != nil {
			return nil, false, err
		}
	}
	// MLP fold (rows-capable layers, K>1): the attn halves write hPacked so all K rows are alive
	// at once, then the layer's MLP runs as ONE rms-rows + three batched projections + one fused
	// gelu — the layer's gate/up/down weights swept once instead of K times. The projection is the
	// projector's OWN batched dispatch (projectRows): bf16 → batched gemv (byte-identical per row
	// — its z-slices run the single-row tile loop unchanged), quant → MLX qmm_t (token-identity
	// tier: simdgroup-MMA accumulation order, one weight pass — the prompt-prefill reclaim).
	// Metallib-less runs / unfoldable geometry keep the proven per-row interleave.
	foldDFFMax, foldQDimMax, foldKVDimMax := 0, 0, 0
	if (icbK == nil || K > batchedDenseICBMaxRows || s.verifyFoldSmallK) && !batchedMLPFoldDisabledForTest && K > 1 && gpuHasGeluKernel() {
		for li := range s.specs {
			if !s.lb[li].proj.rowsCapable() {
				continue
			}
			lff := s.dFF
			if s.lb[li].dFF > 0 {
				lff = s.lb[li].dFF
			}
			if lff > foldDFFMax {
				foldDFFMax = lff
			}
			lhd := headDimOf(s.specs[li], s.headDim)
			if q := s.nHeads * lhd; q > foldQDimMax {
				foldQDimMax = q
			}
			if kv := kvHeadsOf(s.specs[li], s.nKVHeads) * lhd; kv > foldKVDimMax {
				foldKVDimMax = kv
			}
		}
	}
	var hSlab, mlpNormSlab, gateSlab, upSlab, gatedSlab, downSlab metal.MTLBuffer
	var attnNormSlab, qSlab, attnSlab, attnOutSlab, kStage, vStage metal.MTLBuffer
	if foldDFFMax > 0 {
		hSlab, mlpNormSlab, gateSlab, upSlab, gatedSlab, downSlab = s.denseBatch.mlpFold(K, s.dModel, foldDFFMax)
		attnNormSlab, qSlab, attnSlab, attnOutSlab, kStage, vStage = s.denseBatch.attnFold(K, s.dModel, foldQDimMax, foldKVDimMax)
	}
	// verify-tail ICB (#372): the fold's pos-independent per-layer tail replays
	// recorded on repeat verify blocks at the SAME width — recordings are per-K
	// (the adaptive draft cap wobbles K block-to-block; a single-K recording
	// never replayed). A width's first eligible block records alongside its own
	// live encodes; a reallocated slab (K growth reallocs the fold slabs)
	// mismatches the key and re-records that width once under the new buffers.
	var vtKey verifyTailKey
	var vtReplay *verifyTailICB
	if s.verifyFoldSmallK && foldDFFMax > 0 && !verifyTailICBDisabled && !verifyTailICBDisabledForTest {
		vtKey = verifyTailKey{
			k: K, dModel: s.dModel,
			inPacked: bufID(inRows[0]), outPacked: bufID(outRows[0]),
			hSlab: bufID(hSlab), mlpNormSlab: bufID(mlpNormSlab),
			gateSlab: bufID(gateSlab), upSlab: bufID(upSlab),
			gatedSlab: bufID(gatedSlab), downSlab: bufID(downSlab),
			attnNormSlab: bufID(attnNormSlab), attnSlab: bufID(attnSlab), attnOut: bufID(attnOutSlab),
			pleSlab: bufID(pleSlabBuf),
		}
		if vt := s.verifyTail[K]; vt != nil && vt.key != vtKey {
			delete(s.verifyTail, K) // the buffer world changed under this K: re-record once
			delete(s.verifyTailTried, K)
		}
		vtReplay = s.verifyTail[K]
		if vtReplay != nil {
			// fresh pass, fresh encoder: the resident set must be re-declared
			// (a new encoder can reallocate at a previous pass's address).
			vtReplay.declaredEnc = 0
		}
		if vtReplay == nil && !s.verifyTailTried[K] {
			if s.verifyTailTried == nil {
				s.verifyTailTried = map[int]bool{}
			}
			s.verifyTailTried[K] = true
			s.verifyTailRec = newVerifyTailRecorder(len(s.specs), vtKey)
		}
	}
	if s.verifyTailRec != nil {
		defer func() {
			rec := s.verifyTailRec
			s.verifyTailRec = nil
			if vt := rec.finish(); vt != nil {
				if s.verifyTail == nil {
					s.verifyTail = map[int]*verifyTailICB{}
				}
				s.verifyTail[vt.key.k] = vt
			}
		}()
	}
	// deferred-landing bookkeeping (the big-K staged sliding tail): which owners deferred their
	// ring landing (their sharers then ride the owner's stage), and the landings to encode after
	// every layer has read the pre-batch ring state.
	type ringLanding struct{ li, kvDim, slideW int }
	var pendingLandings []ringLanding
	var stagedDeferred []bool
	if foldDFFMax > 0 && K >= steelGEMMMinRows && !stagedRingDisabledForTest && s.rowAttnCaps == nil {
		stagedDeferred = make([]bool, len(s.specs))
	}
	cb := commandBufferFast(queue)
	enc := computeCommandEncoderFast(cb)
	trace := newBatchedGPUTrace(cb, "prologue") // LTHN_GPU_TRACE: per-stage GPU attribution
	// Non-final prefill chunks stop at the shared suffix (#381): the bounded-out
	// layers own no cache rows, and readRows past the loop is only consumed for
	// the FINAL chunk, which always runs the full stack. The last-layer output
	// redirects (directLastOut / usingDirectOutputRows) key off len(s.specs)-1
	// and simply never fire on a bounded pass.
	layerEnd := len(s.specs)
	if s.prefillSkipToLayer > 0 && s.prefillSkipToLayer < layerEnd {
		layerEnd = s.prefillSkipToLayer
	}
	// q8Blocks: the interleave's q8 SDPA reads (encAttnHalfKVQ8InputAt /
	// encAttnHalfSharedKVQ8At, #53/#54) must key their 2-pass-or-not choice off
	// the SAME session-wide value the recorded ICB replay bakes at construction
	// (archQ8ICBBlocks), never off the live basePos+i — see archQ8ICBBlocks.
	// Computed once per call (session-invariant), not per row.
	var q8Blocks int
	if icbK != nil && s.icb.hasKVQ8() {
		q8Blocks = s.archQ8ICBBlocks()
	}
	for li := 0; li < layerEnd; li++ {
		lhd, lkv := headDimOf(s.specs[li], s.headDim), kvHeadsOf(s.specs[li], s.nKVHeads)
		slideW, rbase, rotDim := 0, s.base, s.rotaryDim
		layerRopeFreqs := s.ropeFreqs
		if s.specs[li].Attention == model.SlidingAttention {
			slideW, rbase, rotDim = s.slidingWindow, s.localBase, s.rotaryDimLocal
		} else if s.globalRopeFreqs != nil {
			layerRopeFreqs, rotDim = s.globalRopeFreqs, lhd
		}
		lff := s.dFF
		if s.lb[li].dFF > 0 {
			lff = s.lb[li].dFF
		}
		proj := s.lb[li].proj
		foldMLP := hSlab != nil && proj.rowsCapable()
		// recorded-ICB sessions: the replay's caches are the live set for every read AND write
		layerK, layerV := s.lb[li].kCache, s.lb[li].vCache
		if icbK != nil {
			layerK, layerV = icbK[li], icbV[li]
		}
		// attention fold: the Q/K/V/O projections batch across the K rows too (grid Z, each weight
		// read once), while the ordered per-row tail — fused per-head norm+rope, value norm, SDPA
		// capped at the row's own live length — keeps the exact sequential cache semantics: only
		// the projections were hoisted; the cache MUTATIONS still land row by row. A ring layer
		// whose window would evict during this batch projects K/V into staging rows and the fused
		// norm+rope (a full-row write) lands each row into its slot in order. Needs the fused
		// qknorm-rope kernel + the gemma4 norms; anything else keeps the proven per-row halves.
		foldAttn := foldMLP && s.lb[li].qNorm.buf != nil
		// Attention sinks (gpt_oss) are wired through the per-row interleave's encSDPADecodeSinksAt
		// ONLY — the fold's multi-query SDPA kernels have no has_sinks lane, so a sinks layer folding
		// would silently drop its sinks. (Today this is doubly-guarded: gpt_oss carries no qNorm, so
		// foldAttn is already false above — this line keeps the guard explicit, not incidental.)
		foldAttn = foldAttn && s.lb[li].sinks.buf == nil
		ownsCache := s.specs[li].OwnsCache()
		kvDim := lkv * lhd
		qDim := s.nHeads * lhd
		staged := false
		if ownsCache {
			foldAttn = foldAttn && s.lb[li].kNorm.buf != nil
			staged = slideW > 0 && basePos+K > slideW
			if staged && s.valueNormOnes == nil {
				foldAttn = false // staged V lands via the value norm's full-row write
			}
		}
		// rows-batched epilogues: entry rms, residuals and the layer tail run once over the K
		// contiguous rows instead of once per row — valid whenever the rows live in the shared
		// ping-pong slabs (layer 0 may read direct input views; the LAST layer may scatter to
		// direct output rows — those keep the per-row path).
		batchedRows := foldMLP && !batchedEpilogueDisabledForTest && gpuHasMulRowsKernel() &&
			(len(s.ple) == 0 || s.pliDim <= foldDFFMax)
		xContig := !(li == 0 && usingDirectInputRows)
		// tailReplayed: this layer's whole pos-independent tail (o-proj → layer
		// scalar) ran as one recorded-ICB range — the live tail encodes below
		// (o-proj/resid AND the foldMLP section) are skipped for it. GPU-trace
		// note: the replayed tail's span lands in the attn.o+resid bucket.
		tailReplayed := false
		if foldAttn {
			enc = trace.checkpoint(enc, "attn.norm+qkv")
			anw := s.lb[li].anw
			if batchedRows && xContig {
				// all K layer-input rows are the contiguous ping-pong slab: one rms-rows dispatch
				if err = encRMSNormRowsBF16(enc, readRows[0], anw.buf, attnNormSlab, readOff[0], anw.off, 0, K, s.dModel, s.eps); err != nil {
					endEncodingFast(enc)
					return nil, false, err
				}
			} else {
				// per-row rms into the norm slab (layer inputs may be non-contiguous direct views)
				for i := range K {
					if err = encRMSNormRowsBF16(enc, readRows[i], anw.buf, attnNormSlab, readOff[i], anw.off, uint(i*rowBytes), 1, s.dModel, s.eps); err != nil {
						endEncodingFast(enc)
						return nil, false, err
					}
				}
			}
			if err = projectRowsRequired(proj, enc, attnNormSlab, qSlab, 0, 0, K, projQ); err != nil {
				endEncodingFast(enc)
				return nil, false, err
			}
			ownIdx := li
			if !ownsCache {
				ownIdx = s.specs[li].KVShareFrom
			}
			ownerK, ownerV := s.lb[ownIdx].kCache, s.lb[ownIdx].vCache
			if icbK != nil {
				ownerK, ownerV = icbK[ownIdx], icbV[ownIdx]
			}
			// batched rope: the K per-row fused norm+rope dispatches fold into one (grid Y carries
			// the row, positions from the packed offsets buffer). Q always; the K landing + value
			// norm only on the direct/no-evict path (a staged ring lands slot-wrapped, per row).
			batchedRope := !batchedRopeDisabledForTest && gpuHasQKNormRopeRows()
			// deferred-landing lane (the big-K staged sliding tail): K/V project into this layer's
			// PRIVATE stage (roped/normed in place there), ONE two-segment ring SDPA reads the
			// pre-batch ring minus each query's evicted run plus the staged causal rows, and the
			// ring lands in bulk after every layer has read the pre-batch state. Sharers ride the
			// owner's stage — the true sequential window. Token-identity lane (fp accumulation
			// order differs from the ring-order oracle), engaged only at steelGEMMMinRows with a
			// FULL ring; the byte-identical per-row interleave stays below.
			deferredRing := false
			if stagedDeferred != nil {
				if ownsCache {
					// any basePos: the ring kernel handles a partial/fresh pre-batch ring and a
					// batch wider than the window (a chunk may cross the ring wrap).
					deferredRing = staged && batchedRope &&
						gpuHasSDPAMultiQRing(lhd) && gpuHasCopyKernel()
				} else {
					deferredRing = stagedDeferred[ownIdx] && slideW > 0
				}
			}
			kvQ8Layer := s.icb != nil && s.icb.kvQ8.on(li)
			kvTQLayer := s.icb != nil && s.icb.kvTQ.on(li)
			if ownsCache {
				kDst, vDst, dstOff := layerK, layerV, uint(basePos*kvDim*bf16Size)
				if kvQ8Layer {
					// q8 global owner (#367): project into the shared stage slabs,
					// rope/norm there, then ONE rows-store quantises the K rows
					// into the int8 cache + scale rows — the batch attends its own
					// rows post-round-trip, exactly as the sequential replay does.
					kDst, vDst, dstOff = kStage, vStage, 0
				} else if kvTQLayer {
					// TQ global owner (#48): project into the shared stage slabs,
					// rope/norm there, then encTQKVStoreRows quantises the whole
					// chunk into the code cache + γ planes and encTQKVDequantRows
					// reconstructs the history into the bf16 scratch the SDPA reads.
					kDst, vDst, dstOff = kStage, vStage, 0
				} else if deferredRing {
					kDst, vDst = s.denseBatch.layerStage(li, len(s.specs), K, foldKVDimMax)
					dstOff = 0
				} else if staged {
					kDst, vDst, dstOff = kStage, vStage, 0
				}
				if err = projectRowsRequired(proj, enc, attnNormSlab, kDst, 0, dstOff, K, projK); err != nil {
					endEncodingFast(enc)
					return nil, false, err
				}
				vIdx := projV
				if !proj.hasV() {
					vIdx = projK // gemma4 K==V layers: V is the k-proj output, value-normed
				}
				if err = projectRowsRequired(proj, enc, attnNormSlab, vDst, 0, dstOff, K, vIdx); err != nil {
					endEncodingFast(enc)
					return nil, false, err
				}
			}
			enc = trace.checkpoint(enc, "attn.rope+vnorm")
			// multi-query SDPA: all K rows' attention in ONE dispatch (grid Y carries the rows,
			// the per-query causal cap computed in-kernel) — needs the direct/no-evict landing
			// AND every row below the 2-pass knee, so each row's bytes match the per-row
			// single-query kernel exactly (the same routing the sequential oracle takes).
			// The knee guards the SMALL-K case only: with few threadgroups a long kv serialises
			// inside one TG and the per-row 2-pass re-parallelises it. At prompt scale K×heads
			// threadgroups saturate the GPU regardless, so the single-pass multiQ stays fastest
			// at any kv — the per-row 2-pass loop it replaces ran K dispatch pairs per global
			// layer (~173ms per 512-row chunk at basePos 512). Token-identity tier past the knee
			// (the per-row oracle would have routed 2-pass there), same tier as the fold's qmm.
			useMultiQ := !sdpaMultiQDisabledForTest && (slideW == 0 || basePos+K <= slideW) &&
				(basePos+K < sdpa2PassMinKV || K >= steelGEMMMinRows) && gpuHasSDPAMultiQ(lhd) &&
				s.rowAttnCaps == nil // per-row caps: the multiQ/GEMM kernels compute causal caps in-kernel
			// Deep-prompt global layers route the SDPA to the steel GEMM composition
			// (sdpa_prompt_gemm.go): K/V read once per HEAD instead of once per query row,
			// so the traffic no longer scales with the row count and the multiQ kernel's
			// deep-context SLC-decay ramp never engages. Same emission seam as multiQ —
			// the per-row loop still runs its staged/rope tail, only the SDPA dispatches
			// differ. Token-identity tier (S stores bf16 between the GEMMs), the same
			// boundary the fold's qmm and ≥32-row steel projections already trade at.
			ownerQ8 := s.icb != nil && s.icb.kvQ8.on(ownIdx) // sharers of a q8 owner read q8 too
			ownerTQ := s.icb != nil && s.icb.kvTQ.on(ownIdx) // sharers of a TQ owner read its bf16 scratch too
			// q8 GEMM prefix (#367): the steel GEMM composition reads bf16, so a
			// q8 owner dequantises its attended prefix into the layer's snapshot
			// mirrors first (in this encoder, after the landing) and the GEMM
			// reads the mirrors. Without this the deep-prompt lane fell back to
			// multiQ q8: attn.sdpa 323.6ms vs 71.3ms per chunk at basePos 25600
			// (the whole 2-3.6x q8 prefill tax). Mirror ensure failing (no
			// dequant kernel, alloc failure) keeps the multiQ fallback.
			// q8 flash (#375 phase 3): the flash lane reads the int8 codes + scales
			// directly, so a q8 owner needs NO mirrors and NO dequant dispatches on
			// that route — the ensure below runs only for the composition/bf16-flash
			// fallbacks.
			flashQ8 := ownerQ8 && flashPromptEnabled && flashQ8Enabled &&
				!flashQ8OffForTest && flashQ8Usable(lhd, basePos+K)
			// q8 staging (#375): the per-chunk prefix dequant lands in the shared
			// ping-pong staging pair instead of a per-layer full-cacheRows mirror
			// plane — same dequant kernel, same flash/GEMM consumers, ~one plane
			// pair of memory instead of one per global owner (the 31B@256K
			// ingest-peak cut). LTHN_Q8_STAGE=0 restores the mirror planes.
			q8Stage := ownerQ8 && !flashQ8 && q8StageEnabled && !q8StageOffForTest &&
				gpuHasKVQ8DequantRows()
			var q8GEMMK, q8GEMMV metal.MTLBuffer
			if ownerQ8 && !flashQ8 && !q8Stage && gpuHasKVQ8DequantRows() {
				q8GEMMK, q8GEMMV, _ = s.icb.ensureQ8Mirrors(ownIdx)
			}
			useGEMMSDPA := useMultiQ && slideW == 0 && basePos+K >= sdpaPromptGEMMKnee() &&
				K <= sdpaPromptGEMMMaxRows && sdpaPromptGEMMFeasible(K, s.maxLen) &&
				!sdpaPromptGEMMDisabledForTest &&
				!sdpaPromptGEMMEnvDisabled() && gpuHasPromptSDPAGEMM() &&
				(!ownerQ8 || q8Stage || q8GEMMK != nil || flashQ8)
			if batchedRope {
				if err = encQKNormRopeRows(enc, qSlab, s.lb[li].qNorm.buf, qSlab, 0, s.lb[li].qNorm.off, 0, qDim, qDim, offBuf[0], layerRopeFreqs, K, s.nHeads, lhd, rotDim, rbase, s.ropeScale, s.eps); err != nil {
					endEncodingFast(enc)
					return nil, false, err
				}
				if ownsCache && kvQ8Layer {
					// q8 global owner (#367): rope/norm the STAGED rows in place, then
					// ONE rows-store each quantises them into the int8 cache rows +
					// scale rows at basePos — the same bytes the sequential replay's
					// per-token store lands, chunk-wide.
					if err = encQKNormRopeRows(enc, kStage, s.lb[li].kNorm.buf, kStage, 0, s.lb[li].kNorm.off, 0, kvDim, kvDim, offBuf[0], layerRopeFreqs, K, lkv, lhd, rotDim, rbase, s.ropeScale, s.eps); err != nil {
						endEncodingFast(enc)
						return nil, false, err
					}
					if s.valueNormOnes != nil {
						if err = encRMSNormRowsBF16(enc, vStage, s.valueNormOnes, vStage, 0, 0, 0, K*lkv, lhd, s.eps); err != nil {
							endEncodingFast(enc)
							return nil, false, err
						}
					}
					q8RowOff := uint(basePos * kvDim)
					q8ScaleOff := uint(basePos * (kvDim / kvQ8GroupSize) * 4)
					if err = encKVQ8StoreRows(enc, kStage, layerK, q8RowOff, s.icb.kvQ8.kScales[li], q8ScaleOff, K, kvDim); err != nil {
						endEncodingFast(enc)
						return nil, false, err
					}
					if err = encKVQ8StoreRows(enc, vStage, layerV, q8RowOff, s.icb.kvQ8.vScales[li], q8ScaleOff, K, kvDim); err != nil {
						endEncodingFast(enc)
						return nil, false, err
					}
				} else if ownsCache && kvTQLayer {
					// TQ global owner (#48): rope/norm the STAGED rows in place, then
					// tqBatchedLanding quantises the whole chunk into the code cache
					// and reconstructs the code history into the bf16 scratch the SDPA
					// scores against. V is value-normed but NOT roped, exactly as the
					// q8 twin above and the recorded per-token store.
					if err = encQKNormRopeRows(enc, kStage, s.lb[li].kNorm.buf, kStage, 0, s.lb[li].kNorm.off, 0, kvDim, kvDim, offBuf[0], layerRopeFreqs, K, lkv, lhd, rotDim, rbase, s.ropeScale, s.eps); err != nil {
						endEncodingFast(enc)
						return nil, false, err
					}
					if s.valueNormOnes != nil {
						if err = encRMSNormRowsBF16(enc, vStage, s.valueNormOnes, vStage, 0, 0, 0, K*lkv, lhd, s.eps); err != nil {
							endEncodingFast(enc)
							return nil, false, err
						}
					}
					if _, _, err = s.tqBatchedLanding(enc, li, basePos, K, kStage, vStage); err != nil {
						endEncodingFast(enc)
						return nil, false, err
					}
				} else if ownsCache && deferredRing {
					// rope/norm the staged rows IN PLACE — the deferred landing copies the
					// finished bytes into the ring slots, so the landed rows are identical to
					// what the per-row landing would have written.
					kSt, vSt := s.denseBatch.layerStage(li, len(s.specs), K, foldKVDimMax)
					if err = encQKNormRopeRows(enc, kSt, s.lb[li].kNorm.buf, kSt, 0, s.lb[li].kNorm.off, 0, kvDim, kvDim, offBuf[0], layerRopeFreqs, K, lkv, lhd, rotDim, rbase, s.ropeScale, s.eps); err != nil {
						endEncodingFast(enc)
						return nil, false, err
					}
					if s.valueNormOnes != nil {
						if err = encRMSNormRowsBF16(enc, vSt, s.valueNormOnes, vSt, 0, 0, 0, K*lkv, lhd, s.eps); err != nil {
							endEncodingFast(enc)
							return nil, false, err
						}
					}
				} else if ownsCache && !staged {
					kvBase := uint(basePos * kvDim * bf16Size)
					if err = encQKNormRopeRows(enc, layerK, s.lb[li].kNorm.buf, layerK, kvBase, s.lb[li].kNorm.off, kvBase, kvDim, kvDim, offBuf[0], layerRopeFreqs, K, lkv, lhd, rotDim, rbase, s.ropeScale, s.eps); err != nil {
						endEncodingFast(enc)
						return nil, false, err
					}
					if s.valueNormOnes != nil {
						if err = encRMSNormRowsBF16(enc, layerV, s.valueNormOnes, layerV, kvBase, 0, kvBase, K*lkv, lhd, s.eps); err != nil {
							endEncodingFast(enc)
							return nil, false, err
						}
					}
				}
			}
			if ownerTQ {
				// TQ (#48): score the ordinary bf16 SDPA against the reconstructed
				// code history — tqBatchedLanding filled the owner's scratch this
				// chunk (a GQA sharer reads the same owner scratch). ownerQ8 is
				// false for a TQ layer, so every SDPA route below stays bf16.
				ownerK, ownerV = s.tqPrefill.k[ownIdx], s.tqPrefill.v[ownIdx]
			}
			// the sdpa span, labelled by ROUTE (#375): the lump hid which lane the
			// time lived in — window flash (sliding layers), the steel GEMM
			// composition, or the multiQ vector kernel (shallow globals).
			sdpaLbl := "attn.sdpa.mq"
			if slideW > 0 {
				sdpaLbl = "attn.sdpa.win"
			} else if useGEMMSDPA {
				sdpaLbl = "attn.sdpa.gemm"
			}
			enc = trace.checkpoint(enc, sdpaLbl)
			// Bidirectional row caps demand the batched-rope fold: only there
			// does the WHOLE chunk's K/V land before any SDPA reads. Anything
			// else would evaluate the span causally — hard-error, never fall
			// through silently.
			if s.rowAttnCaps != nil && ownsCache && (!foldAttn || !batchedRope || staged) {
				endEncodingFast(enc)
				return nil, false, core.NewError("native.stepTokensBatchedDense: bidirectional row caps need the batched-rope attention fold")
			}
			for i := 0; !deferredRing && i < K; i++ { // skipped whole on the deferred-ring lane
				pos := basePos + i
				slot, n := pos, pos+1
				if slideW > 0 {
					slot = pos % slideW
					if n > slideW {
						n = slideW
					}
				}
				if caps := s.rowAttnCaps; caps != nil && i < len(caps) {
					if c := int(caps[i]); c > n {
						if slideW > 0 && c > slideW {
							c = slideW
						}
						n = c
					}
					if unifiedVisionDiag && li == 0 && (i == 5 || i == 128 || i == 260) {
						nativeTraceLog(core.Sprintf("vision-diag cap: li=0 row=%d pos=%d n=%d cap=%d slideW=%d\n", i, pos, n, caps[i], slideW))
					}
				}
				qRow := uint(i * qDim * bf16Size)
				if !batchedRope {
					if err = encQKNormRopeAt(enc, qSlab, s.lb[li].qNorm.buf, qSlab, qRow, s.lb[li].qNorm.off, qRow, offBuf[i], offOff[i], layerRopeFreqs, s.nHeads, lhd, rotDim, rbase, s.ropeScale, s.eps); err != nil {
						endEncodingFast(enc)
						return nil, false, err
					}
				}
				if ownsCache && (staged || !batchedRope) {
					kvRow := uint(slot * kvDim * bf16Size)
					kSrc, vSrc, srcOff := layerK, layerV, kvRow
					if staged {
						kSrc, vSrc, srcOff = kStage, vStage, uint(i*kvDim*bf16Size)
					}
					if err = encQKNormRopeAt(enc, kSrc, s.lb[li].kNorm.buf, layerK, srcOff, s.lb[li].kNorm.off, kvRow, offBuf[i], offOff[i], layerRopeFreqs, lkv, lhd, rotDim, rbase, s.ropeScale, s.eps); err != nil {
						endEncodingFast(enc)
						return nil, false, err
					}
					if s.valueNormOnes != nil {
						if err = encRMSNormRowsBF16(enc, vSrc, s.valueNormOnes, layerV, srcOff, 0, kvRow, lkv, lhd, s.eps); err != nil {
							endEncodingFast(enc)
							return nil, false, err
						}
					}
				}
				if useMultiQ {
					continue // the K SDPAs run as one multi-query dispatch after every landing
				}
				if ownerQ8 {
					// deep-verify q8 (#367): the same knee + blocks ladder as the
					// bf16 row, reading the owner's int8 cache + scale planes.
					if err = encSDPADecodeQ8At(enc, s.asc, qSlab, qRow, ownerK, ownerV,
						s.icb.kvQ8.kScales[ownIdx], s.icb.kvQ8.vScales[ownIdx], attnSlab, qRow, s.nHeads, lkv, lhd, n,
						int64(lhd), int64(kvDim), int64(lhd), int64(kvDim), s.scale); err != nil {
						endEncodingFast(enc)
						return nil, false, err
					}
				} else if err = encSDPADecodeAt(enc, s.asc, qSlab, qRow, ownerK, ownerV, attnSlab, qRow, s.nHeads, lkv, lhd, n,
					int64(lhd), int64(kvDim), int64(lhd), int64(kvDim), s.scale, 0); err != nil {
					endEncodingFast(enc)
					return nil, false, err
				}
			}
			if deferredRing {
				kSt, vSt := s.denseBatch.layerStage(ownIdx, len(s.specs), K, foldKVDimMax)
				ringLive := min(basePos, slideW)
				if flashPromptEnabled && flashWinEnabled && !flashWinOffForTest && K >= flashWinMinRows && flashWinUsable(lhd) {
					// window flash (#375 phase 4): the query tile streams its own
					// ≤ W+BQ key span once, shared by 32 queries — the multiQ ring
					// kernel it replaces re-read the window per query row.
					if err = encFlashWindowSDPA(enc, qSlab, ownerK, ownerV, kSt, vSt, attnSlab,
						s.nHeads, lkv, lhd, K, slideW, basePos, ringLive, qDim, kvDim, s.scale); err != nil {
						endEncodingFast(enc)
						return nil, false, err
					}
				} else if err = encSDPAMultiQRing(enc, qSlab, ownerK, ownerV, kSt, vSt, attnSlab,
					s.nHeads, lkv, lhd, K, slideW, basePos%slideW, ringLive,
					int64(lhd), int64(kvDim), int64(lhd), int64(kvDim), s.scale); err != nil {
					endEncodingFast(enc)
					return nil, false, err
				}
				if ownsCache {
					stagedDeferred[li] = true
					pendingLandings = append(pendingLandings, ringLanding{li: li, kvDim: kvDim, slideW: slideW})
				}
			} else if useGEMMSDPA && flashQ8 {
				// q8 codes + scales straight into the flash tiles — no mirrors on
				// this route (the ensure above was skipped), half the K/V bytes.
				if err = encFlashPromptQ8(enc, qSlab, ownerK, ownerV,
					s.icb.kvQ8.kScales[ownIdx], s.icb.kvQ8.vScales[ownIdx], attnSlab,
					s.nHeads, lkv, lhd, K, basePos+K, qDim, kvDim, s.scale); err != nil {
					endEncodingFast(enc)
					return nil, false, err
				}
			} else if useGEMMSDPA {
				kGEMM, vGEMM := ownerK, ownerV
				if ownerQ8 {
					kDst, vDst := q8GEMMK, q8GEMMV
					if q8Stage {
						kDst, vDst = s.denseBatch.q8Stage(ownIdx&1, basePos+K, kvDim)
						if kDst == nil || vDst == nil {
							endEncodingFast(enc)
							return nil, false, core.NewError("native.batched prefill: q8 staging allocation failed")
						}
					}
					// dequantise the attended prefix [0, basePos+K) into the staging
					// pair (or the legacy mirror planes) — Metal's in-encoder hazard
					// tracking orders this after the landing above and before the
					// GEMM below.
					if err = encKVQ8DequantRows(enc, ownerK, s.icb.kvQ8.kScales[ownIdx], kDst, basePos+K, kvDim); err != nil {
						endEncodingFast(enc)
						return nil, false, err
					}
					if err = encKVQ8DequantRows(enc, ownerV, s.icb.kvQ8.vScales[ownIdx], vDst, basePos+K, kvDim); err != nil {
						endEncodingFast(enc)
						return nil, false, err
					}
					kGEMM, vGEMM = kDst, vDst
				}
				if flashPromptEnabled && flashPromptUsable(lhd, basePos+K) {
					// flash lane (#375): one dispatch, no S materialisation — the
					// composition's S round-trip taxed every neighbouring GEMM ~6x
					// its own bandwidth (the #367 fold-context conviction). Same
					// inputs (q8 owners read the dequanted mirrors), same causal
					// rule, token-identity tier either way.
					if err = encFlashPromptSDPA(enc, qSlab, kGEMM, vGEMM, attnSlab,
						s.nHeads, lkv, lhd, K, basePos+K, qDim, kvDim, s.scale); err != nil {
						endEncodingFast(enc)
						return nil, false, err
					}
				} else {
					headBatch := sdpaPromptHeadBatch(s.nHeads/lkv, K, s.maxLen)
					sScore0, sScore1 := s.denseBatch.sdpaPromptS(headBatch*K, s.maxLen)
					if err = encSDPAPromptGEMM(enc, qSlab, kGEMM, vGEMM, attnSlab, sScore0, sScore1,
						s.nHeads, lkv, lhd, K, basePos+K, qDim, kvDim, headBatch, s.scale); err != nil {
						endEncodingFast(enc)
						return nil, false, err
					}
				}
			} else if useMultiQ {
				if ownerQ8 {
					if err = encSDPAMultiQCausalQ8(enc, qSlab, ownerK, ownerV, attnSlab,
						s.icb.kvQ8.kScales[ownIdx], s.icb.kvQ8.vScales[ownIdx], s.nHeads, lkv, lhd, K, basePos+K,
						int64(lhd), int64(kvDim), int64(lhd), int64(kvDim), s.scale); err != nil {
						endEncodingFast(enc)
						return nil, false, err
					}
				} else if err = encSDPAMultiQCausal(enc, qSlab, ownerK, ownerV, attnSlab, s.nHeads, lkv, lhd, K, basePos+K,
					int64(lhd), int64(kvDim), int64(lhd), int64(kvDim), s.scale); err != nil {
					endEncodingFast(enc)
					return nil, false, err
				}
			}
			enc = trace.checkpoint(enc, "attn.o+resid")
			// verify-tail replay (#372): the recorded range covers this layer's
			// whole tail — o-proj, both residuals, the MLP fold, the PLE chain
			// and the layer scalar — so the live encodes below AND the foldMLP
			// section are skipped for it.
			if vtReplay != nil && batchedRows && !s.specs[li].MoE &&
				vtReplay.replayable(li, vtKey) {
				vtReplay.execute(enc, li)
				tailReplayed = true
			}
			if !tailReplayed {
				if err = projectRowsRequired(proj, enc, attnSlab, attnOutSlab, 0, 0, K, projO); err != nil {
					endEncodingFast(enc)
					return nil, false, err
				}
				if rec := s.verifyTailRec; rec != nil && batchedRows && !s.specs[li].MoE {
					rec.beginLayer(li) // no-op outside [1, nLayers-2]
					if rec.recording(li) {
						rec.recordProjectRows(proj, attnSlab, attnOutSlab, 0, 0, K, projO)
					}
				}
				if batchedRows && xContig {
					// h = x + postAttnNorm(Wo·attn) for all K rows — attnNormSlab is free as scratch
					if err = encResidualRowsMaybeNorm(enc, readRows[0], readOff[0], attnOutSlab, 0, attnNormSlab, hSlab, 0, s.lb[li].postAttnNorm, K, s.dModel, s.eps); err != nil {
						endEncodingFast(enc)
						return nil, false, err
					}
					if rec := s.verifyTailRec; rec.recording(li) {
						rec.recordResidualRowsMaybeNorm(readRows[0], readOff[0], attnOutSlab, 0, attnNormSlab, hSlab, 0, s.lb[li].postAttnNorm, K, s.dModel, s.eps)
					}
				} else {
					for i := range K {
						// h row i = x row i + postAttnNorm(Wo·attn row i) — attnNormSlab is free as scratch
						if err = encResidualMaybeNormAt(enc, readRows[i], readOff[i], attnOutSlab, uint(i*rowBytes), attnNormSlab, hSlab, uint(i*rowBytes), s.lb[li].postAttnNorm, s.dModel, s.eps); err != nil {
							endEncodingFast(enc)
							return nil, false, err
						}
					}
				}
			}
		}
		// each row in turn: attention half (writes its K/V row, attends [0..basePos+i]), then the
		// MLP half — folded across the K rows for bf16 layers, per-row otherwise. Metal's buffer
		// hazard tracking orders the cross-row cache write→read, so row i+1 attends row i's freshly
		// written K/V — exactly the sequential per-token causal structure.
		if s.rowAttnCaps != nil && !foldAttn && ownsCache {
			// per-row K/V landings can never satisfy a forward-looking cap
			endEncodingFast(enc)
			return nil, false, core.NewError("native.stepTokensBatchedDense: bidirectional row caps need the batched-rope attention fold")
		}
		// #53/#54: the interleave's q8 twins (see encAttnHalfKVQ8InputAt /
		// encAttnHalfSharedKVQ8At above) — interleaveKVQ8Usable already proved every
		// q8-touching layer has what these need before the function ever reached
		// this loop, so a plain kvQ8.on lookup here is enough.
		kvQ8Layer := !foldAttn && ownsCache && s.icb != nil && s.icb.kvQ8.on(li)
		var ownerQ8 bool
		var q8Own int
		if !foldAttn && !ownsCache && s.icb != nil {
			q8Own = s.specs[li].KVShareFrom
			ownerQ8 = s.icb.kvQ8.on(q8Own)
		}
		for i := 0; !foldAttn && i < K; i++ { // skipped whole when the attention fold ran above
			hTarget, hOff := s.hBuf, uint(0)
			if foldMLP {
				hTarget, hOff = hSlab, uint(i*rowBytes)
			}
			if ownsCache {
				if kvQ8Layer {
					if err = encAttnHalfKVQ8InputAt(enc, readRows[i], readOff[i], layerK, layerV,
						s.icb.kvQ8.kScales[li], s.icb.kvQ8.vScales[li], offBuf[i], hTarget, hOff, offOff[i],
						s.lb[li].anw, s.lb[li].postAttnNorm, s.lb[li].qNorm, s.lb[li].kNorm, s.valueNormOnes, s.asc, s.lb[li].proj,
						s.dModel, s.nHeads, lkv, lhd, basePos+i, rotDim, q8Blocks, rbase, s.scale, s.ropeScale, s.eps, layerRopeFreqs); err != nil {
						endEncodingFast(enc)
						return nil, false, err
					}
				} else if err = encAttnHalfKVInputAt(enc, readRows[i], readOff[i], layerK, layerV, offBuf[i], hTarget, hOff, offOff[i],
					s.lb[li].anw, s.lb[li].postAttnNorm, s.lb[li].qNorm, s.lb[li].kNorm, s.valueNormOnes, s.asc, s.lb[li].proj,
					s.dModel, s.nHeads, lkv, lhd, basePos+i, slideW, rotDim, rbase, s.scale, s.ropeScale, s.eps, layerRopeFreqs, s.lb[li].sinks); err != nil {
					endEncodingFast(enc)
					return nil, false, err
				}
			} else {
				own := s.specs[li].KVShareFrom
				var kC, vC metal.MTLBuffer
				if icbK != nil {
					kC, vC = icbK[own], icbV[own]
				} else {
					kC, vC = s.lb[own].kCache, s.lb[own].vCache
				}
				if ownerQ8 {
					if err = encAttnHalfSharedKVQ8At(enc, readRows[i], readOff[i], kC, vC,
						s.icb.kvQ8.kScales[q8Own], s.icb.kvQ8.vScales[q8Own], offBuf[i], hTarget, hOff, offOff[i],
						s.lb[li].anw, s.lb[li].postAttnNorm, s.lb[li].qNorm, s.asc, s.lb[li].proj,
						s.dModel, s.nHeads, lkv, lhd, basePos+i, rotDim, q8Blocks, rbase, s.scale, s.ropeScale, s.eps, layerRopeFreqs); err != nil {
						endEncodingFast(enc)
						return nil, false, err
					}
				} else if err = encAttnHalfSharedInputAt(enc, readRows[i], readOff[i], kC, vC, offBuf[i], hTarget, hOff, offOff[i],
					s.lb[li].anw, s.lb[li].postAttnNorm, s.lb[li].qNorm, s.asc, s.lb[li].proj,
					s.dModel, s.nHeads, lkv, lhd, basePos+i, slideW, rotDim, rbase, s.scale, s.ropeScale, s.eps, layerRopeFreqs, s.lb[li].sinks); err != nil {
					endEncodingFast(enc)
					return nil, false, err
				}
			}
			if foldMLP {
				continue // the MLP runs folded once every row's attention half is encoded
			}
			outBuf, outOff := outRows[i], rowOff[i]
			if directLastOut && li == len(s.specs)-1 && i == K-1 {
				outBuf, outOff = lastOutBuf, 0
			} else if usingDirectOutputRows && li == len(s.specs)-1 {
				outBuf, outOff = directOutputRows[i], directOutputOff[i]
			}
			if err = encMLPHalfBF16At(enc, s.hBuf, outBuf, outOff, s.lb[li].mnw, s.lb[li].postFFNorm, s.msc, s.lb[li].proj, s.dModel, lff, s.eps); err != nil {
				endEncodingFast(enc)
				return nil, false, err
			}
			// gemma4 per-layer-input gate (E2B/E4B) + per-layer scalar: same encoded chain the
			// sequential stepToken runs, reading row i's pliDim slice from the K-token slab.
			if err = s.encBatchedRowEpilogue(enc, pleSlabBuf, li, i, K, outBuf, outOff); err != nil {
				endEncodingFast(enc)
				return nil, false, err
			}
		}
		if moeQ := moeQuantAt(s.moeQuant, li); moeQ != nil && s.specs[li].MoE {
			enc = trace.checkpoint(enc, "moe")
			// the batched MoE block: router + local MLP + all-pairs expert gathers + combine,
			// every stage swept once over the K rows (moe_batch.go).
			outContig := li != len(s.specs)-1 || (!directLastOut && !usingDirectOutputRows)
			if batchedRows && outContig {
				if err = s.encMoEBlockQuantBatched(enc, *moeQ, hSlab, outRows, rowOff, true, K); err != nil {
					endEncodingFast(enc)
					return nil, false, err
				}
				if err = s.encBatchedEpilogueRows(enc, pleSlabBuf, li, K, outRows[0], rowOff[0], gateSlab, gatedSlab, downSlab, mlpNormSlab); err != nil {
					endEncodingFast(enc)
					return nil, false, err
				}
			} else {
				rowBufs := make([]metal.MTLBuffer, K)
				rowOffs := make([]uint, K)
				for i := range K {
					rowBufs[i], rowOffs[i] = outRows[i], rowOff[i]
					if directLastOut && li == len(s.specs)-1 && i == K-1 {
						rowBufs[i], rowOffs[i] = lastOutBuf, 0
					} else if usingDirectOutputRows && li == len(s.specs)-1 {
						rowBufs[i], rowOffs[i] = directOutputRows[i], directOutputOff[i]
					}
				}
				if err = s.encMoEBlockQuantBatched(enc, *moeQ, hSlab, rowBufs, rowOffs, false, K); err != nil {
					endEncodingFast(enc)
					return nil, false, err
				}
				for i := range K {
					if err = s.encBatchedRowEpilogue(enc, pleSlabBuf, li, i, K, rowBufs[i], rowOffs[i]); err != nil {
						endEncodingFast(enc)
						return nil, false, err
					}
				}
			}
		} else if foldMLP && !tailReplayed {
			enc = trace.checkpoint(enc, "mlp")
			// the folded MLP: one rms across the K rows, gate/up/down as batched gemvs (grid Z=K,
			// the layer's weight matrix shared across rows), gelu(gate)·up fused over K·lff.
			mnw := s.lb[li].mnw
			if err = encRMSNormRowsBF16(enc, hSlab, mnw.buf, mlpNormSlab, 0, mnw.off, 0, K, s.dModel, s.eps); err != nil {
				endEncodingFast(enc)
				return nil, false, err
			}
			enc = trace.checkpoint(enc, "mlp.gate")
			if err = projectRowsRequired(proj, enc, mlpNormSlab, gateSlab, 0, 0, K, projGate); err != nil {
				endEncodingFast(enc)
				return nil, false, err
			}
			enc = trace.checkpoint(enc, "mlp.up")
			if err = projectRowsRequired(proj, enc, mlpNormSlab, upSlab, 0, 0, K, projUp); err != nil {
				endEncodingFast(enc)
				return nil, false, err
			}
			enc = trace.checkpoint(enc, "mlp.gelu")
			if proj.usesSiLU() { // SwiGLU (llama/mistral/qwen): silu(gate)·up over the K·lff slab (sigmoid + two in-place muls)
				if err = encSiLUGateMulBF16(enc, gateSlab, upSlab, gatedSlab, K*lff); err != nil {
					endEncodingFast(enc)
					return nil, false, err
				}
			} else if err = encGeluGateMulFused(enc, gateSlab, upSlab, gatedSlab, K*lff); err != nil {
				endEncodingFast(enc)
				return nil, false, err
			}
			enc = trace.checkpoint(enc, "mlp.down")
			if err = projectRowsRequired(proj, enc, gatedSlab, downSlab, 0, 0, K, projDown); err != nil {
				endEncodingFast(enc)
				return nil, false, err
			}
			if rec := s.verifyTailRec; rec.recording(li) {
				rec.recordRMSRows(hSlab, mnw.buf, mlpNormSlab, 0, mnw.off, 0, K, s.dModel, s.eps)
				rec.recordProjectRows(proj, mlpNormSlab, gateSlab, 0, 0, K, projGate)
				rec.recordProjectRows(proj, mlpNormSlab, upSlab, 0, 0, K, projUp)
				rec.recordGeluGateMul(gateSlab, upSlab, gatedSlab, 0, 0, 0, K*lff)
				rec.recordProjectRows(proj, gatedSlab, downSlab, 0, 0, K, projDown)
			}
			enc = trace.checkpoint(enc, "resid+epilogue")
			outContig := li != len(s.specs)-1 || (!directLastOut && !usingDirectOutputRows)
			if batchedRows && outContig {
				// out = h + rms(down) for all K rows, then the whole layer tail (PLE gate chain +
				// scalar) rows-batched — mlpNormSlab/downSlab are free as scratch after this
				// (the hazards order the reuse).
				if err = encResidualRowsMaybeNorm(enc, hSlab, 0, downSlab, 0, mlpNormSlab, outRows[0], rowOff[0], s.lb[li].postFFNorm, K, s.dModel, s.eps); err != nil {
					endEncodingFast(enc)
					return nil, false, err
				}
				if rec := s.verifyTailRec; rec.recording(li) {
					rec.recordResidualRowsMaybeNorm(hSlab, 0, downSlab, 0, mlpNormSlab, outRows[0], rowOff[0], s.lb[li].postFFNorm, K, s.dModel, s.eps)
				}
				if err = s.encBatchedEpilogueRows(enc, pleSlabBuf, li, K, outRows[0], rowOff[0], gateSlab, gatedSlab, downSlab, mlpNormSlab); err != nil {
					endEncodingFast(enc)
					return nil, false, err
				}
				if rec := s.verifyTailRec; rec != nil {
					rec.endLayer(li)
				}
			} else {
				for i := range K {
					outBuf, outOff := outRows[i], rowOff[i]
					if directLastOut && li == len(s.specs)-1 && i == K-1 {
						outBuf, outOff = lastOutBuf, 0
					} else if usingDirectOutputRows && li == len(s.specs)-1 {
						outBuf, outOff = directOutputRows[i], directOutputOff[i]
					}
					// out row i = h row i + rms(down row i) — mlpNormSlab is free as the norm scratch
					// (the gate/up gemvs already consumed it; the hazard orders the reuse).
					if err = encResidualMaybeNormAt(enc, hSlab, uint(i*rowBytes), downSlab, uint(i*rowBytes), mlpNormSlab, outBuf, outOff, s.lb[li].postFFNorm, s.dModel, s.eps); err != nil {
						endEncodingFast(enc)
						return nil, false, err
					}
					if err = s.encBatchedRowEpilogue(enc, pleSlabBuf, li, i, K, outBuf, outOff); err != nil {
						endEncodingFast(enc)
						return nil, false, err
					}
				}
			}
		}
		readRows, outRows = outRows, inRows // this layer's outputs feed the next layer
		readOff = rowOff
	}
	// deferred ring landings: every layer (owners AND their sharers) has read the pre-batch ring
	// state, so the staged rows now land in their slots — at most two contiguous runs per owner
	// (the wrap split). Only the LAST slideW rows land (a batch wider than the window evicted its
	// own head rows during the batch); the landed bytes are exactly the staged roped/normed rows.
	enc = trace.checkpoint(enc, "landings")
	for _, p := range pendingLandings {
		kSt, vSt := s.denseBatch.layerKStage[p.li], s.denseBatch.layerVStage[p.li]
		landK, landV := s.lb[p.li].kCache, s.lb[p.li].vCache
		if icbK != nil { // recorded-ICB sessions: land into the replay's ring
			landK, landV = icbK[p.li], icbV[p.li]
		}
		r0 := 0
		if K > p.slideW {
			r0 = K - p.slideW
		}
		landRows := K - r0
		slotBase := (basePos + r0) % p.slideW
		run1 := min(p.slideW-slotBase, landRows)
		if err = encCopyBF16Contig(enc, kSt, landK, uint(r0*p.kvDim*bf16Size), uint(slotBase*p.kvDim*bf16Size), run1*p.kvDim); err != nil {
			endEncodingFast(enc)
			return nil, false, err
		}
		if err = encCopyBF16Contig(enc, vSt, landV, uint(r0*p.kvDim*bf16Size), uint(slotBase*p.kvDim*bf16Size), run1*p.kvDim); err != nil {
			endEncodingFast(enc)
			return nil, false, err
		}
		if landRows > run1 {
			if err = encCopyBF16Contig(enc, kSt, landK, uint((r0+run1)*p.kvDim*bf16Size), 0, (landRows-run1)*p.kvDim); err != nil {
				endEncodingFast(enc)
				return nil, false, err
			}
			if err = encCopyBF16Contig(enc, vSt, landV, uint((r0+run1)*p.kvDim*bf16Size), 0, (landRows-run1)*p.kvDim); err != nil {
				endEncodingFast(enc)
				return nil, false, err
			}
		}
	}
	cb = trace.commandBuffer(cb) // checkpoints rotate the CB — commit the live one
	endEncodingFast(enc)
	commitCommandBufferFast(cb)
	waitUntilCompletedFast(cb)
	trace.finish(K, basePos)
	if K > 0 {
		if usingDirectOutputRows {
			s.denseBatch.setLastRows(directOutputRows, directOutputOff, K)
		} else if directLastOut && readLastOnly {
			s.denseBatch.readOffStack[0] = 0
			s.denseBatch.lastRowBufStack[0] = lastOutBuf
			s.denseBatch.setLastRows(s.denseBatch.lastRowBufStack[:1], s.denseBatch.readOffStack[:1], 1)
		} else {
			s.denseBatch.setLastRows(readRows, readOff, K)
		}
	}
	reloadStart := time.Now()
	if icbK == nil { // ICB sessions decode over the replay's caches; the paged pool stays bypassed
		if err := s.reloadDevicePagedKVFromLinear(basePos + K); err != nil {
			return nil, false, err
		}
	}
	hostSpan("reloadKV", reloadStart, K)

	if readResult {
		if readLastOnly {
			out = s.denseBatch.lastResult[:1]
			out[0] = lastDst
			if !directLastOut {
				off := int(readOff[K-1])
				copy(out[0], unsafe.Slice((*byte)(readRows[K-1].Contents()), off+rowBytes)[off:]) // readRows = final layer out
			}
			return out, true, nil
		}
		if len(dstRows) >= K {
			out = dstRows[:K]
		} else {
			out = make([][]byte, K)
		}
		for i := range K {
			if usingDirectOutputRows {
				out[i] = out[i][:rowBytes]
				continue
			}
			if cap(out[i]) < rowBytes {
				out[i] = make([]byte, rowBytes)
			} else {
				out[i] = out[i][:rowBytes]
			}
			off := int(readOff[i])
			copy(out[i], unsafe.Slice((*byte)(readRows[i].Contents()), off+rowBytes)[off:]) // readRows = final layer out
		}
	}
	return out, true, nil
}

// verifyBatched is the MTP verify's batched fast path: it embeds the K ids, runs them through the
// resident stack in ONE pass at positions [pos, pos+K), writes their K/V into the caches, and returns
// each id's NEXT-token greedy (greedys[i] = the target's greedy of the hidden after ids[i]). It does
// NOT advance s.pos — MTPDecode sets the position to the committed length after accept/reject, exactly
// as the sequential verify leaves it. ok is false (no work, no cache mutation) for models outside the
// dense path (PLE / MoE / recorded-ICB / shared-KV), where MTPDecode steps sequentially instead — both
// paths produce the identical greedys, so the token stream is unchanged either way.
func (s *ArchSession) verifyBatched(ids []int32) (greedys []int32, ok bool, err error) {
	return s.verifyBatchedInto(ids, nil)
}

func (s *ArchSession) verifyBatchedHiddens(ids []int32) ([][]byte, bool, error) {
	if s.verifyBatchedDisabledForTest { // test-only: force the sequential verify lane
		return nil, false, nil
	}
	if len(ids) == 0 {
		return nil, false, core.NewError("native.verifyBatchedHiddens: empty batch")
	}
	// PLE archs batch via the per-token slab (the batched pass ring-writes each
	// row at its own slot, so wrap-crossing blocks are handled).
	if s.pos+len(ids) > s.maxLen {
		if mtpDiagForTest {
			nativeTraceLog(core.Sprintf("mtp-diag verifyBatchedHiddens: pos+K=%d > maxLen=%d\n", s.pos+len(ids), s.maxLen))
		}
		return nil, false, nil
	}
	var embStack [16][]byte
	var embs [][]byte
	if len(ids) <= len(embStack) {
		embs = embStack[:len(ids)]
	} else {
		embs = make([][]byte, len(ids))
	}
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
			emb, eerr := s.embedInto(dst, id)
			if eerr != nil {
				return nil, false, eerr
			}
			if len(emb) != rowBytes {
				return nil, false, core.NewError("native.verifyBatchedHiddens: embedInto returned wrong hidden size")
			}
			embs[i] = emb
		}
	} else {
		for i, id := range ids {
			emb, eerr := s.embed(id)
			if eerr != nil {
				return nil, false, eerr
			}
			embs[i] = emb
		}
	}
	pleSlab, slabErr := s.pleSlabFor(ids, embs)
	if slabErr != nil {
		return nil, false, slabErr
	}
	var (
		hiddens [][]byte
		ok      bool
		err     error
	)
	withAutoreleasePool(func() {
		rows, rowsOK := s.mtpVerifyHiddenRowsScratch(len(ids), s.arch.Hidden*bf16Size)
		switch {
		case pleSlab != nil && rowsOK:
			hiddens, ok, err = s.state.stepTokensBatchedDenseIntoPLE(embs, pleSlab, s.pos, rows)
		case pleSlab != nil:
			hiddens, ok, err = s.state.stepTokensBatchedDensePLE(embs, pleSlab, s.pos)
		case rowsOK:
			hiddens, ok, err = s.state.stepTokensBatchedDenseInto(embs, s.pos, rows)
		default:
			hiddens, ok, err = s.state.stepTokensBatchedDense(embs, s.pos)
		}
	})
	if err != nil || !ok {
		if mtpDiagForTest && err == nil {
			nativeTraceLog("mtp-diag verifyBatchedHiddens: state declined the batched dense lane\n")
		}
		return nil, ok, err
	}
	return hiddens, true, nil
}

func (s *ArchSession) verifyBatchedInto(ids []int32, greedys []int32) ([]int32, bool, error) {
	if s.verifyBatchedDisabledForTest { // test-only: force the sequential verify lane
		return nil, false, nil
	}
	if len(ids) == 0 {
		return nil, false, core.NewError("native.verifyBatched: empty batch")
	}
	// PLE archs batch via the per-token slab (ring wraps are handled per row); no
	// cache headroom → sequential fallback.
	if s.pos+len(ids) > s.maxLen {
		return nil, false, nil
	}
	var embStack [16][]byte
	var embs [][]byte
	if len(ids) <= len(embStack) {
		embs = embStack[:len(ids)]
	} else {
		embs = make([][]byte, len(ids))
	}
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
			emb, eerr := s.embedInto(dst, id)
			if eerr != nil {
				return nil, false, eerr
			}
			if len(emb) != rowBytes {
				return nil, false, core.NewError("native.verifyBatched: embedInto returned wrong hidden size")
			}
			embs[i] = emb
		}
	} else {
		for i, id := range ids {
			e, eerr := s.embed(id)
			if eerr != nil {
				return nil, false, eerr
			}
			embs[i] = e
		}
	}
	pleSlab, slabErr := s.pleSlabFor(ids, embs)
	if slabErr != nil {
		return nil, false, slabErr
	}
	if s.canUseDirectHeadGreedy() {
		if len(greedys) < len(ids) {
			greedys = make([]int32, len(ids))
		} else {
			greedys = greedys[:len(ids)]
		}
		var (
			ok  bool
			err error
		)
		withAutoreleasePool(func() {
			if pleSlab != nil {
				ok, err = s.state.stepTokensBatchedDenseNoResultPLE(embs, pleSlab, s.pos)
			} else {
				ok, err = s.state.stepTokensBatchedDenseNoResult(embs, s.pos)
			}
			if err != nil || !ok {
				return
			}
			err = s.encodePackedGreedyRowsInto(s.state.denseBatch.lastRows, s.state.denseBatch.lastRowOff, len(ids), greedys)
		})
		if err != nil || !ok {
			return nil, ok, err
		}
		return greedys, true, nil
	}
	var (
		hiddens [][]byte
		ok      bool
		err     error
	)
	withAutoreleasePool(func() {
		rows, rowsOK := s.mtpVerifyHiddenRowsScratch(len(ids), s.arch.Hidden*bf16Size)
		switch {
		case pleSlab != nil && rowsOK:
			hiddens, ok, err = s.state.stepTokensBatchedDenseIntoPLE(embs, pleSlab, s.pos, rows)
		case pleSlab != nil:
			hiddens, ok, err = s.state.stepTokensBatchedDensePLE(embs, pleSlab, s.pos)
		case rowsOK:
			hiddens, ok, err = s.state.stepTokensBatchedDenseInto(embs, s.pos, rows)
		default:
			hiddens, ok, err = s.state.stepTokensBatchedDense(embs, s.pos)
		}
	})
	if err != nil || !ok {
		return nil, ok, err
	}
	if len(greedys) < len(hiddens) {
		greedys = make([]int32, len(hiddens))
	} else {
		greedys = greedys[:len(hiddens)]
	}
	for i, h := range hiddens {
		g, gerr := s.greedyOf(h)
		if gerr != nil {
			return nil, false, gerr
		}
		greedys[i] = g
	}
	return greedys, true, nil
}

func (s *ArchSession) verifyBatchedCrossesSlidingRingWrap(n int) bool {
	if s == nil || n <= 0 || s.arch.SlidingWindow <= 0 || s.arch.SlidingWindow >= s.maxLen {
		return false
	}
	window := s.arch.SlidingWindow
	if s.pos%window+n <= window {
		return false
	}
	for _, spec := range s.state.specs {
		if spec.OwnsCache() && spec.Attention != model.GlobalAttention {
			return true
		}
	}
	return false
}

func (s *ArchSession) rememberDenseBatchRetainedHidden(row int) error {
	rowBuf, off, ok, err := s.denseBatchHiddenRowBuffer(row)
	if err != nil {
		return err
	}
	if !ok {
		return core.NewError("native.verifyBatched: retained hidden row is unavailable")
	}
	base := unsafe.Pointer((*byte)(rowBuf.Contents()))
	s.rememberRetainedHiddenFrom((*byte)(unsafe.Add(base, int(off))))
	return nil
}

func (s *ArchSession) denseBatchHiddenRowBuffer(row int) (metal.MTLBuffer, uint, bool, error) {
	if s == nil || row < 0 || row >= s.state.denseBatch.lastK || row >= len(s.state.denseBatch.lastRowOff) {
		return nil, 0, false, nil
	}
	rowBuf := s.state.denseBatch.lastRows
	if row < len(s.state.denseBatch.lastRowBuf) && s.state.denseBatch.lastRowBuf[row] != nil {
		rowBuf = s.state.denseBatch.lastRowBuf[row]
	}
	if rowBuf == nil {
		return nil, 0, false, nil
	}
	off := s.state.denseBatch.lastRowOff[row]
	rowBytes := uint(s.arch.Hidden * bf16Size)
	n := bufferLengthFast(rowBuf)
	if off > n || rowBytes > n-off {
		return nil, 0, true, core.NewError("native.verifyBatched: hidden row is out of range")
	}
	return rowBuf, off, true, nil
}

func (s *ArchSession) encodePackedGreedyRowsInto(rows metal.MTLBuffer, rowOff []uint, n int, greedys []int32) error {
	if rows == nil || len(rowOff) < n || len(greedys) < n {
		return core.NewError("native.verifyBatched: missing packed dense rows")
	}
	var scratchStack [16]*headGreedyScratch
	scratches := scratchStack[:0]
	if n > len(scratchStack) {
		scratches = make([]*headGreedyScratch, 0, n)
	}
	cb := commandBufferFast(queue)
	enc := computeCommandEncoderFast(cb)
	for i := range n {
		scratch, ok, err := s.headEnc.encodeGreedyAt(enc, rows, rowOff[i], nil)
		if err != nil || !ok {
			endEncodingFast(enc)
			for _, sc := range scratches {
				s.headEnc.putGreedyScratch(sc)
			}
			if err != nil {
				return err
			}
			return core.NewError("native.verifyBatched: direct head greedy unavailable")
		}
		scratches = append(scratches, scratch)
	}
	endEncodingFast(enc)
	commitCommandBufferFast(cb)
	waitUntilCompletedFast(cb)
	for i, scratch := range scratches {
		greedys[i] = scratch.token()
		s.headEnc.putGreedyScratch(scratch)
	}
	for i, token := range greedys[:n] {
		if token < 0 || int(token) >= s.arch.Vocab {
			return core.NewError(core.Sprintf("native.verifyBatched: greedy row %d returned invalid token %d for vocab %d", i, token, s.arch.Vocab))
		}
	}
	return nil
}
