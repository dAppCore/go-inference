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
	foldQDimCap    int
	foldKVDimCap   int
	// per-layer staging for the deferred-landing lane (the big-K staged sliding tail): each
	// staged owner's K/V stay alive across the whole layer loop for its sharers, landing in bulk
	// at the end of the chunk.
	layerKStage      []metal.MTLBuffer
	layerVStage      []metal.MTLBuffer
	layerStageRowCap int
	layerStageKVCap  int
}

// mlpFold returns the K-row MLP-fold slabs, (re)allocating when the batch width, model width or
// the widest per-layer FFN grows. dFFMax is the max dFF across the foldable layers (gemma4 E2B/E4B
// vary it per layer); each layer's gate/up rows still land contiguously at z·itsOwnDFF in the slab.
func (s *denseBatchScratch) mlpFold(k, dModel, dFFMax int) (h, normed, gate, up, gated, down metal.MTLBuffer) {
	if s.hPacked == nil || s.foldRowCap < k || s.foldDModel != dModel || s.foldDFFCap < dFFMax {
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

// attnFold returns the attention-fold slabs, (re)allocating alongside mlpFold's sizing. Call after
// mlpFold (it owns foldRowCap/foldDModel); qDimMax/kvDimMax are the widest per-layer head geometry.
func (s *denseBatchScratch) attnFold(k, dModel, qDimMax, kvDimMax int) (normed, q, attn, attnOut, kStage, vStage metal.MTLBuffer) {
	if s.attnNormPacked == nil || s.foldRowCap < k || s.foldDModel != dModel || s.foldQDimCap < qDimMax || s.foldKVDimCap < kvDimMax {
		s.attnNormPacked = scratchBF16(k * dModel)
		s.attnOutPacked = scratchBF16(k * dModel)
		s.qPacked = scratchBF16(k * qDimMax)
		s.attnPacked = scratchBF16(k * qDimMax)
		s.kStagePacked = scratchBF16(k * kvDimMax)
		s.vStagePacked = scratchBF16(k * kvDimMax)
		s.foldQDimCap, s.foldKVDimCap = qDimMax, kvDimMax
	}
	return s.attnNormPacked, s.qPacked, s.attnPacked, s.attnOutPacked, s.kStagePacked, s.vStagePacked
}

// layerStage returns layer li's PRIVATE K/V staging slabs for the deferred-landing lane — every
// staged owner keeps its batch K/V alive until the end-of-chunk landing, so shared-KV layers can
// read the owner's true pre-batch ring + stage. Sized by the attnFold caps; call after attnFold.
func (s *denseBatchScratch) layerStage(li, layers, k, kvDimMax int) (kSt, vSt metal.MTLBuffer) {
	if len(s.layerKStage) != layers || s.layerStageRowCap < k || s.layerStageKVCap < kvDimMax {
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
			if err := encQMMTBF16At(enc, gatePacked.buf, gateScales.buf, gateBiases.buf, outBuf, gateSlab, gatePacked.off, gateScales.off, gateBiases.off, outBase, 0, rows, s.pliDim, s.dModel, gateGroupSize, gateBits); err != nil {
				return err
			}
			pliBase := uint(li * rows * s.pliDim * bf16Size)
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
		}
	}
	if s.lb[li].layerScalar != nil { // gemma4 per-layer output scalar, all rows in one dispatch
		return encMulRowsBF16(enc, outBuf, s.lb[li].layerScalar, outBuf, outBase, 0, outBase, rows, s.dModel)
	}
	return nil
}

func (s *archDecodeState) stepTokensBatchedDenseResultWithInputViewsPLE(embs [][]byte, pleSlab []byte, basePos int, readResult, readLastOnly bool, lastDst []byte, dstRows [][]byte, directInputs bool) (out [][]byte, ok bool, err error) {
	K := len(embs)
	if K == 0 {
		return nil, false, core.NewError("native.stepTokensBatchedDense: empty batch")
	}
	// dense uniform guard: every layer owns its cache + is non-MoE; no trace (per-layer host reads).
	// The PLE gate is NOT a host flush (it is an encoded kernel chain reading a per-token input
	// buffer — bf16 gemv or quant qmv), so a PLE arch batches when the caller supplies the K-token
	// slab; without one it declines to the proven sequential fallback.
	if s.trace {
		return nil, false, nil
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
		if (K > batchedDenseICBMaxRows && !foldable) ||
			len(s.icb.kCaches) < len(s.specs) || len(s.icb.vCaches) < len(s.specs) || len(s.icb.cacheRows) < len(s.specs) {
			return nil, false, nil
		}
		icbK, icbV = s.icb.kCaches, s.icb.vCaches
	}
	if len(s.ple) > 0 {
		if pleSlab == nil {
			return nil, false, nil
		}
		if want := K * len(s.specs) * s.pliDim * bf16Size; len(pleSlab) != want {
			return nil, false, core.NewError("native.stepTokensBatchedDense: PLE slab size mismatch")
		}
	} else if pleSlab != nil {
		return nil, false, core.NewError("native.stepTokensBatchedDense: PLE slab supplied for a non-PLE arch")
	}
	for li := range s.specs {
		if s.specs[li].MoE {
			return nil, false, nil
		}
		if li < len(s.moeWeights) && s.moeWeights[li] != nil {
			return nil, false, nil
		}
		if li < len(s.moeQuant) && s.moeQuant[li] != nil {
			return nil, false, nil
		}
		if s.specs[li].OwnsCache() {
			if icbK == nil {
				continue
			}
			if icbK[li] == nil || icbV[li] == nil {
				return nil, false, nil
			}
			rows := s.icb.cacheRows[li]
			if s.specs[li].Attention == model.SlidingAttention && s.slidingWindow > 0 {
				// the interleave's slot math is pos%slidingWindow: identical to the replay's
				// pos%cacheRows ring when rows==slidingWindow, and to its linear write when
				// the allocation is un-bounded (rows>=maxLen ⇒ every pos writes linearly).
				if rows != s.slidingWindow && rows < s.maxLen {
					return nil, false, nil
				}
			} else if rows < basePos+K {
				return nil, false, nil
			}
			continue
		}
		// shared-KV layers (gemma4 E2B/E4B tails) attend an OWNER's cache: batchable —
		// the owner's rows for this batch are encoded at a lower layer index in the same
		// command buffer — provided the owner's caches (live set) are resident.
		own := s.specs[li].KVShareFrom
		if own < 0 || own >= len(s.specs) {
			return nil, false, nil
		}
		if icbK != nil {
			if icbK[own] == nil || icbV[own] == nil {
				return nil, false, nil
			}
		} else if own >= len(s.lb) || s.lb[own].kCache == nil || s.lb[own].vCache == nil {
			return nil, false, nil
		}
	}
	for i := range embs {
		if len(embs[i]) != s.dModel*bf16Size {
			return nil, false, core.NewError("native.stepTokensBatchedDense: emb must be dModel bf16 bytes")
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
	if len(pleSlab) > 0 {
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
	if (icbK == nil || K > batchedDenseICBMaxRows) && !batchedMLPFoldDisabledForTest && K > 1 && gpuHasGeluKernel() {
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
	// deferred-landing bookkeeping (the big-K staged sliding tail): which owners deferred their
	// ring landing (their sharers then ride the owner's stage), and the landings to encode after
	// every layer has read the pre-batch ring state.
	type ringLanding struct{ li, kvDim, slideW int }
	var pendingLandings []ringLanding
	var stagedDeferred []bool
	if foldDFFMax > 0 && K >= steelGEMMMinRows && !stagedRingDisabledForTest {
		stagedDeferred = make([]bool, len(s.specs))
	}
	cb := commandBufferFast(queue)
	enc := computeCommandEncoderFast(cb)
	trace := newBatchedGPUTrace(cb, "prologue") // LTHN_GPU_TRACE: per-stage GPU attribution
	for li := 0; li < len(s.specs); li++ {
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
			if ownsCache {
				kDst, vDst, dstOff := layerK, layerV, uint(basePos*kvDim*bf16Size)
				if deferredRing {
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
			useMultiQ := !sdpaMultiQDisabledForTest && (slideW == 0 || basePos+K <= slideW) &&
				basePos+K < sdpa2PassMinKV && gpuHasSDPAMultiQ(lhd)
			if batchedRope {
				if err = encQKNormRopeRows(enc, qSlab, s.lb[li].qNorm.buf, qSlab, 0, s.lb[li].qNorm.off, 0, qDim, qDim, offBuf[0], layerRopeFreqs, K, s.nHeads, lhd, rotDim, rbase, s.scale, s.eps); err != nil {
					endEncodingFast(enc)
					return nil, false, err
				}
				if ownsCache && deferredRing {
					// rope/norm the staged rows IN PLACE — the deferred landing copies the
					// finished bytes into the ring slots, so the landed rows are identical to
					// what the per-row landing would have written.
					kSt, vSt := s.denseBatch.layerStage(li, len(s.specs), K, foldKVDimMax)
					if err = encQKNormRopeRows(enc, kSt, s.lb[li].kNorm.buf, kSt, 0, s.lb[li].kNorm.off, 0, kvDim, kvDim, offBuf[0], layerRopeFreqs, K, lkv, lhd, rotDim, rbase, s.scale, s.eps); err != nil {
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
					if err = encQKNormRopeRows(enc, layerK, s.lb[li].kNorm.buf, layerK, kvBase, s.lb[li].kNorm.off, kvBase, kvDim, kvDim, offBuf[0], layerRopeFreqs, K, lkv, lhd, rotDim, rbase, s.scale, s.eps); err != nil {
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
			enc = trace.checkpoint(enc, "attn.sdpa")
			for i := 0; !deferredRing && i < K; i++ { // skipped whole on the deferred-ring lane
				pos := basePos + i
				slot, n := pos, pos+1
				if slideW > 0 {
					slot = pos % slideW
					if n > slideW {
						n = slideW
					}
				}
				qRow := uint(i * qDim * bf16Size)
				if !batchedRope {
					if err = encQKNormRopeAt(enc, qSlab, s.lb[li].qNorm.buf, qSlab, qRow, s.lb[li].qNorm.off, qRow, offBuf[i], offOff[i], layerRopeFreqs, s.nHeads, lhd, rotDim, rbase, s.scale, s.eps); err != nil {
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
					if err = encQKNormRopeAt(enc, kSrc, s.lb[li].kNorm.buf, layerK, srcOff, s.lb[li].kNorm.off, kvRow, offBuf[i], offOff[i], layerRopeFreqs, lkv, lhd, rotDim, rbase, s.scale, s.eps); err != nil {
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
				if err = encSDPADecodeAt(enc, s.asc, qSlab, qRow, ownerK, ownerV, attnSlab, qRow, s.nHeads, lkv, lhd, n,
					int64(lhd), int64(kvDim), int64(lhd), int64(kvDim), s.scale, 0); err != nil {
					endEncodingFast(enc)
					return nil, false, err
				}
			}
			if deferredRing {
				kSt, vSt := s.denseBatch.layerStage(ownIdx, len(s.specs), K, foldKVDimMax)
				ringLive := min(basePos, slideW)
				if err = encSDPAMultiQRing(enc, qSlab, ownerK, ownerV, kSt, vSt, attnSlab,
					s.nHeads, lkv, lhd, K, slideW, basePos%slideW, ringLive,
					int64(lhd), int64(kvDim), int64(lhd), int64(kvDim), s.scale); err != nil {
					endEncodingFast(enc)
					return nil, false, err
				}
				if ownsCache {
					stagedDeferred[li] = true
					pendingLandings = append(pendingLandings, ringLanding{li: li, kvDim: kvDim, slideW: slideW})
				}
			} else if useMultiQ {
				if err = encSDPAMultiQCausal(enc, qSlab, ownerK, ownerV, attnSlab, s.nHeads, lkv, lhd, K, basePos+K,
					int64(lhd), int64(kvDim), int64(lhd), int64(kvDim), s.scale); err != nil {
					endEncodingFast(enc)
					return nil, false, err
				}
			}
			enc = trace.checkpoint(enc, "attn.o+resid")
			if err = projectRowsRequired(proj, enc, attnSlab, attnOutSlab, 0, 0, K, projO); err != nil {
				endEncodingFast(enc)
				return nil, false, err
			}
			if batchedRows && xContig {
				// h = x + postAttnNorm(Wo·attn) for all K rows — attnNormSlab is free as scratch
				if err = encResidualRowsMaybeNorm(enc, readRows[0], readOff[0], attnOutSlab, 0, attnNormSlab, hSlab, 0, s.lb[li].postAttnNorm, K, s.dModel, s.eps); err != nil {
					endEncodingFast(enc)
					return nil, false, err
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
		// each row in turn: attention half (writes its K/V row, attends [0..basePos+i]), then the
		// MLP half — folded across the K rows for bf16 layers, per-row otherwise. Metal's buffer
		// hazard tracking orders the cross-row cache write→read, so row i+1 attends row i's freshly
		// written K/V — exactly the sequential per-token causal structure.
		for i := 0; !foldAttn && i < K; i++ { // skipped whole when the attention fold ran above
			hTarget, hOff := s.hBuf, uint(0)
			if foldMLP {
				hTarget, hOff = hSlab, uint(i*rowBytes)
			}
			if ownsCache {
				if err = encAttnHalfKVInputAt(enc, readRows[i], readOff[i], layerK, layerV, offBuf[i], hTarget, hOff, offOff[i],
					s.lb[li].anw, s.lb[li].postAttnNorm, s.lb[li].qNorm, s.lb[li].kNorm, s.valueNormOnes, s.asc, s.lb[li].proj,
					s.dModel, s.nHeads, lkv, lhd, basePos+i, slideW, rotDim, rbase, s.scale, s.eps, layerRopeFreqs); err != nil {
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
				if err = encAttnHalfSharedInputAt(enc, readRows[i], readOff[i], kC, vC, offBuf[i], hTarget, hOff, offOff[i],
					s.lb[li].anw, s.lb[li].postAttnNorm, s.lb[li].qNorm, s.asc, s.lb[li].proj,
					s.dModel, s.nHeads, lkv, lhd, basePos+i, slideW, rotDim, rbase, s.scale, s.eps, layerRopeFreqs); err != nil {
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
		if foldMLP {
			enc = trace.checkpoint(enc, "mlp")
			// the folded MLP: one rms across the K rows, gate/up/down as batched gemvs (grid Z=K,
			// the layer's weight matrix shared across rows), gelu(gate)·up fused over K·lff.
			mnw := s.lb[li].mnw
			if err = encRMSNormRowsBF16(enc, hSlab, mnw.buf, mlpNormSlab, 0, mnw.off, 0, K, s.dModel, s.eps); err != nil {
				endEncodingFast(enc)
				return nil, false, err
			}
			if err = projectRowsRequired(proj, enc, mlpNormSlab, gateSlab, 0, 0, K, projGate); err != nil {
				endEncodingFast(enc)
				return nil, false, err
			}
			if err = projectRowsRequired(proj, enc, mlpNormSlab, upSlab, 0, 0, K, projUp); err != nil {
				endEncodingFast(enc)
				return nil, false, err
			}
			if err = encGeluGateMulFused(enc, gateSlab, upSlab, gatedSlab, K*lff); err != nil {
				endEncodingFast(enc)
				return nil, false, err
			}
			if err = projectRowsRequired(proj, enc, gatedSlab, downSlab, 0, 0, K, projDown); err != nil {
				endEncodingFast(enc)
				return nil, false, err
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
				if err = s.encBatchedEpilogueRows(enc, pleSlabBuf, li, K, outRows[0], rowOff[0], gateSlab, gatedSlab, downSlab, mlpNormSlab); err != nil {
					endEncodingFast(enc)
					return nil, false, err
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
