// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"math"
	"sync"
	"time"
	"unsafe"

	core "dappco.re/go"
	"github.com/tmc/apple/metal"
)

// pleBatchScratch holds the K-token PLE builder's reusable buffers (grow-once on the session):
// the token ids, the contiguous hidden slab feeding the steel GEMM, the gather/projection/normed
// slabs, and the two broadcast scale buffers.
type pleBatchScratch struct {
	hidden          []byte          // K × dModel bf16, host staging for the GEMM input
	idsBuf          metal.MTLBuffer // K int32 token ids (the GPU gather's input)
	hiddenBuf       metal.MTLBuffer
	perLayerBuf     metal.MTLBuffer // K × plDim gathered per-layer embeddings (GPU gather output)
	projectedBuf    metal.MTLBuffer // K × plDim projection; free after the rms — reused as the relayout dst
	normedBuf       metal.MTLBuffer // K × plDim, the rms output and final combined tensor
	projScaleBuf    metal.MTLBuffer // 1-element bf16 broadcast scales (mul-rows rowLen=1)
	combineScaleBuf metal.MTLBuffer
	rowCap          int
	plDim, dModel   int
}

func (s *pleBatchScratch) ensure(k, plDim, dModel int, projScale, combineScale float32) {
	if s.rowCap >= k && s.plDim == plDim && s.dModel == dModel && s.hiddenBuf != nil {
		return
	}
	s.hidden = make([]byte, k*dModel*bf16Size)
	s.idsBuf = device.NewBufferWithLengthOptions(uint(k*4), metal.MTLResourceStorageModeShared)
	s.hiddenBuf = device.NewBufferWithLengthOptions(uint(k*dModel*bf16Size), metal.MTLResourceStorageModeShared)
	s.perLayerBuf = device.NewBufferWithLengthOptions(uint(k*plDim*bf16Size), metal.MTLResourceStorageModeShared)
	s.projectedBuf = device.NewBufferWithLengthOptions(uint(k*plDim*bf16Size), metal.MTLResourceStorageModeShared)
	s.normedBuf = device.NewBufferWithLengthOptions(uint(k*plDim*bf16Size), metal.MTLResourceStorageModeShared)
	ps, cs := bf16ScalarBytes(projScale), bf16ScalarBytes(combineScale)
	s.projScaleBuf = device.NewBufferWithBytesLengthOptions(unsafe.Pointer(&ps[0]), 2, metal.MTLResourceStorageModeShared)
	s.combineScaleBuf = device.NewBufferWithBytesLengthOptions(unsafe.Pointer(&cs[0]), 2, metal.MTLResourceStorageModeShared)
	s.rowCap, s.plDim, s.dModel = k, plDim, dModel
}

var (
	pleGatherPSOOnce sync.Once
	pleGatherPSO     metal.MTLComputePipelineState
	pleGatherPSOErr  error

	pleRelayoutPSOOnce sync.Once
	pleRelayoutPSO     metal.MTLComputePipelineState
	pleRelayoutPSOErr  error
)

func pleGatherRowsPipeline() (metal.MTLComputePipelineState, error) {
	pleGatherPSOOnce.Do(func() {
		if customLibrary == nil || customLibrary.GetID() == 0 {
			pleGatherPSOErr = core.NewError("native.pleGatherRowsPipeline: custom library unavailable")
			return
		}
		fn := customLibrary.NewFunctionWithName("lthn_ple_gather_rows_bf16")
		if fn == nil || fn.GetID() == 0 {
			pleGatherPSOErr = core.NewError("native.pleGatherRowsPipeline: kernel lthn_ple_gather_rows_bf16 not found")
			return
		}
		pleGatherPSO, pleGatherPSOErr = device.NewComputePipelineStateWithFunctionError(fn)
	})
	return pleGatherPSO, pleGatherPSOErr
}

func pleRelayoutPipeline() (metal.MTLComputePipelineState, error) {
	pleRelayoutPSOOnce.Do(func() {
		if customLibrary == nil || customLibrary.GetID() == 0 {
			pleRelayoutPSOErr = core.NewError("native.pleRelayoutPipeline: custom library unavailable")
			return
		}
		fn := customLibrary.NewFunctionWithName("lthn_ple_relayout_bf16")
		if fn == nil || fn.GetID() == 0 {
			pleRelayoutPSOErr = core.NewError("native.pleRelayoutPipeline: kernel lthn_ple_relayout_bf16 not found")
			return
		}
		pleRelayoutPSO, pleRelayoutPSOErr = device.NewComputePipelineStateWithFunctionError(fn)
	})
	return pleRelayoutPSO, pleRelayoutPSOErr
}

var (
	pleGatherRowsQuantPSOOnce sync.Once
	pleGatherRowsQuantPSO     metal.MTLComputePipelineState
	pleGatherRowsQuantPSOErr  error

	pleGatherRowsBF16PfxPSOOnce sync.Once
	pleGatherRowsBF16PfxPSO     metal.MTLComputePipelineState
	pleGatherRowsBF16PfxPSOErr  error
)

// pleGatherRowsQuantPipeline resolves the ONE-dispatch quant PLE gather
// (lthn_ple_gather_rows_quant) — K rows per dispatch instead of the K-loop of
// per-token gathers whose encode+launch overhead was ~100ms of an 8K e2b
// prefill. Absent from an older metallib the builders keep the loop.
func pleGatherRowsQuantPipeline() (metal.MTLComputePipelineState, error) {
	pleGatherRowsQuantPSOOnce.Do(func() {
		if customLibrary == nil || customLibrary.GetID() == 0 {
			pleGatherRowsQuantPSOErr = core.NewError("native.pleGatherRowsQuantPipeline: custom library unavailable")
			return
		}
		fn := customLibrary.NewFunctionWithName("lthn_ple_gather_rows_quant")
		if fn == nil || fn.GetID() == 0 {
			pleGatherRowsQuantPSOErr = core.NewError("native.pleGatherRowsQuantPipeline: kernel lthn_ple_gather_rows_quant not found")
			return
		}
		pleGatherRowsQuantPSO, pleGatherRowsQuantPSOErr = device.NewComputePipelineStateWithFunctionError(fn)
	})
	return pleGatherRowsQuantPSO, pleGatherRowsQuantPSOErr
}

// pleGatherRowsBF16PfxPipeline resolves the width/stride-split bf16 rows
// gather (lthn_ple_gather_rows_bf16_pfx) — the bounded-slab lane's gather for
// bf16 PLE tables (#381). Absent, the builders compute full width and copy
// the layer-major prefix out.
func pleGatherRowsBF16PfxPipeline() (metal.MTLComputePipelineState, error) {
	pleGatherRowsBF16PfxPSOOnce.Do(func() {
		if customLibrary == nil || customLibrary.GetID() == 0 {
			pleGatherRowsBF16PfxPSOErr = core.NewError("native.pleGatherRowsBF16PfxPipeline: custom library unavailable")
			return
		}
		fn := customLibrary.NewFunctionWithName("lthn_ple_gather_rows_bf16_pfx")
		if fn == nil || fn.GetID() == 0 {
			pleGatherRowsBF16PfxPSOErr = core.NewError("native.pleGatherRowsBF16PfxPipeline: kernel lthn_ple_gather_rows_bf16_pfx not found")
			return
		}
		pleGatherRowsBF16PfxPSO, pleGatherRowsBF16PfxPSOErr = device.NewComputePipelineStateWithFunctionError(fn)
	})
	return pleGatherRowsBF16PfxPSO, pleGatherRowsBF16PfxPSOErr
}

// pleSlabOutLayers derives the slab's LAYER COUNT from its length — the
// bounded-slab contract (#381): a skipped prefill chunk's slab carries only
// the owner layers' slices ([nOut × K × pliDim] layer-major), a full slab all
// of them. nOut ≤ numLayers because layer-major truncation is a prefix.
func pleSlabOutLayers(slabLen, k, pliDim, numLayers int) (int, error) {
	rowBytes := k * pliDim * bf16Size
	if k <= 0 || rowBytes <= 0 || slabLen%rowBytes != 0 {
		return 0, core.NewError("native.perLayerInputsBatch: slab size mismatch")
	}
	nOut := slabLen / rowBytes
	if nOut < 1 || nOut > numLayers {
		return 0, core.NewError("native.perLayerInputsBatch: slab layer count out of range")
	}
	return nOut, nil
}

// perLayerInputsBatchIntoSlab builds the K-token PLE tensor set in ONE command buffer and
// scatters it layer-major into slab — the batched twin of K PerLayerInputs calls, which each
// paid their own CB round-trip (the 183ms/512-token host wall the GPU trace exposed). The
// projection runs as one steel GEMM (token-identity at large K, the pass's standing policy);
// the scale/rms/combine steps are the same per-element math batched over K·plDim. bf16
// projection weights only (the resident view path); anything else reports false and the caller
// keeps the per-token closure loop.
func perLayerInputsBatchIntoSlab(sc *pleBatchScratch, embedPerLayer []byte, projView bufView, projNormW []byte, ids []int32, embs [][]byte, slab []byte, vocabPLI, numLayers, pliDim, dModel int, eps float32) (bool, error) {
	k := len(ids)
	plDim := numLayers * pliDim
	if projView.buf == nil || k < steelGEMMMinRows || len(projNormW) != pliDim*bf16Size {
		return false, nil
	}
	nOut, serr := pleSlabOutLayers(len(slab), k, pliDim, numLayers)
	if serr != nil {
		return false, serr
	}
	gatherPSO, gerr := pleGatherRowsPipeline()
	relayoutPSO, rerr := pleRelayoutPipeline()
	if gerr != nil || rerr != nil {
		return false, nil // kernels unavailable — the per-token loop still works
	}
	// A bounded slab (#381 shared-suffix chunk) runs the pipeline at the
	// owner-layer prefix when the width/stride-split gather resolves; otherwise
	// full width + layer-major prefix copy — identical slab bytes either way.
	pfxPSO, pfxErr := pleGatherRowsBF16PfxPipeline()
	plDimC, nC := plDim, numLayers
	if pfxErr == nil && nOut < numLayers {
		plDimC, nC = nOut*pliDim, nOut
	}
	embScale := float32(math.Sqrt(float64(pliDim)))
	projScale := float32(1.0 / math.Sqrt(float64(dModel)))
	sc.ensure(k, plDim, dModel, projScale, gemma4PerLayerCombineScale)

	// host: the token ids and the contiguous hidden rows — everything else stays on the GPU.
	rowBytes := dModel * bf16Size
	for i, emb := range embs {
		if len(emb) != rowBytes {
			return false, core.NewError("native.perLayerInputsBatch: hidden row size mismatch")
		}
		copy(sc.hidden[i*rowBytes:(i+1)*rowBytes], emb)
	}
	copy(unsafe.Slice((*int32)(sc.idsBuf.Contents()), k), ids)
	copy(unsafe.Slice((*byte)(sc.hiddenBuf.Contents()), len(sc.hidden)), sc.hidden)

	var encErr error
	engaged := false
	withAutoreleasePool(func() {
		cb := commandBufferFast(queue)
		enc := computeCommandEncoderFast(cb)
		// gather + scale the K per-layer embedding rows on-device
		if plDimC < plDim {
			sink := encSink{enc}
			sink.setPSO(pfxPSO)
			sink.setBuf(sc.idsBuf, 0, 0)
			sink.setBuf(residentBytes(embedPerLayer), 0, 1)
			sink.setBuf(sc.perLayerBuf, 0, 2)
			sink.setI32(int32(plDimC), 3)
			sink.setI32(int32(plDim), 4)
			sink.setF32(embScale, 5)
			sink.dispatchThreads(
				metal.MTLSize{Width: uint(plDimC), Height: uint(k), Depth: 1},
				metal.MTLSize{Width: uint(elemGroupTG(plDimC)), Height: 1, Depth: 1},
			)
		} else {
			sink := encSink{enc}
			sink.setPSO(gatherPSO)
			sink.setBuf(sc.idsBuf, 0, 0)
			sink.setBuf(residentBytes(embedPerLayer), 0, 1)
			sink.setBuf(sc.perLayerBuf, 0, 2)
			sink.setI32(int32(plDim), 3)
			sink.setF32(embScale, 4)
			sink.dispatchThreads(
				metal.MTLSize{Width: uint(plDim), Height: uint(k), Depth: 1},
				metal.MTLSize{Width: uint(elemGroupTG(plDim)), Height: 1, Depth: 1},
			)
		}
		// projected = hidden @ projWᵀ (ONE steel GEMM for all K tokens; a bounded
		// pass sweeps only the owner layers' projection rows — a prefix of projW)
		if !encGemmBF16NT(enc, projView.buf, sc.hiddenBuf, sc.projectedBuf, projView.off, 0, 0, plDimC, dModel, k) {
			endEncodingFast(enc)
			commitCommandBufferFast(cb)
			waitUntilCompletedFast(cb)
			return // steel unavailable — fall back to the per-token loop
		}
		// ×1/√dModel → rms per (token,layer) row → +perLayer → ×1/√2, all batched
		if encErr = encMulRowsBF16(enc, sc.projectedBuf, sc.projScaleBuf, sc.projectedBuf, 0, 0, 0, k*plDimC, 1); encErr == nil {
			if encErr = encRMSNormRowsBF16(enc, sc.projectedBuf, residentBytes(projNormW), sc.normedBuf, 0, 0, 0, k*nC, pliDim, eps); encErr == nil {
				if encErr = encAddBF16To(enc, sc.normedBuf, sc.perLayerBuf, sc.normedBuf, 0, 0, 0, k*plDimC); encErr == nil {
					encErr = encMulRowsBF16(enc, sc.normedBuf, sc.combineScaleBuf, sc.normedBuf, 0, 0, 0, k*plDimC, 1)
				}
			}
		}
		if encErr == nil {
			// token-major → layer-major on-device (projectedBuf is free after the rms consumed it)
			sink := encSink{enc}
			sink.setPSO(relayoutPSO)
			sink.setBuf(sc.normedBuf, 0, 0)
			sink.setBuf(sc.projectedBuf, 0, 1)
			sink.setI32(int32(k), 2)
			sink.setI32(int32(nC), 3)
			sink.setI32(int32(pliDim), 4)
			n := k * plDimC
			sink.dispatchThreads(
				metal.MTLSize{Width: uint(n), Height: 1, Depth: 1},
				metal.MTLSize{Width: uint(elemGroupTG(n)), Height: 1, Depth: 1},
			)
		}
		endEncodingFast(enc)
		commitCommandBufferFast(cb)
		waitUntilCompletedFast(cb)
		engaged = encErr == nil
	})
	if encErr != nil || !engaged {
		return false, encErr
	}
	// layer-major: the slab (nOut ≤ nC layers) is a prefix of the computed tensor
	copy(slab, unsafe.Slice((*byte)(sc.projectedBuf.Contents()), k*plDimC*bf16Size)[:len(slab)])
	return true, nil
}

// perLayerInputsBatchQuantIntoSlab is perLayerInputsBatchIntoSlab for the QUANT PLE shapes —
// the -4bit conversions carry a 4-bit per-layer embedding table (gathered per token with the
// quant dequant-gather kernel) and either a resident bf16 model projection (plain 4-bit, steel
// GEMM) or a quantised one (QAT, qmm_t). Same scratch, same batched tail, same relayout; the
// slab lands host-side exactly like the bf16 builder so every consumer is unchanged.
func perLayerInputsBatchQuantIntoSlab(
	sc *pleBatchScratch,
	tablePacked, tableScales, tableBiases metal.MTLBuffer, tableGS, tableBits int,
	projW metal.MTLBuffer, projWOff uint,
	projPacked, projScales, projBiases metal.MTLBuffer, projGS, projBits int,
	projNormW []byte,
	ids []int32, embs [][]byte, slab []byte,
	numLayers, pliDim, dModel int, eps float32,
) (bool, error) {
	nOut, serr := pleSlabOutLayers(len(slab), len(ids), pliDim, numLayers)
	if serr != nil {
		return false, serr
	}
	plDimC, ok, err := perLayerInputsBatchQuantEncode(sc, tablePacked, tableScales, tableBiases, tableGS, tableBits,
		projW, projWOff, projPacked, projScales, projBiases, projGS, projBits,
		projNormW, ids, embs, nOut, numLayers, pliDim, dModel, eps, true)
	if err != nil || !ok {
		return false, err
	}
	outStart := time.Now()
	// layer-major: the slab (nOut ≤ computed layers) is a prefix of the computed tensor
	copy(slab, unsafe.Slice((*byte)(sc.projectedBuf.Contents()), len(ids)*plDimC*bf16Size)[:len(slab)])
	hostSpan("pleSlab.out", outStart, len(ids))
	return true, nil
}

// perLayerInputsBatchQuantDevice is the DEVICE-RESIDENT quant slab build (#381):
// encode + commit WITHOUT waiting and hand the layer-major tensor's buffer
// straight to the batched pass — the pass's own command buffer follows on the
// same queue, so the GPU orders the read after the build and the host never
// blocks on the builder, copies the slab out, or uploads it again. The scratch
// is safe single-buffered: the caller stages the NEXT chunk only after the
// pass's wait, which covers this command buffer too. ok=false (no work) when
// the encode cannot run at exactly outLayers — the bounded-width contract the
// pass's slab geometry check enforces.
func perLayerInputsBatchQuantDevice(
	sc *pleBatchScratch,
	tablePacked, tableScales, tableBiases metal.MTLBuffer, tableGS, tableBits int,
	projW metal.MTLBuffer, projWOff uint,
	projPacked, projScales, projBiases metal.MTLBuffer, projGS, projBits int,
	projNormW []byte,
	ids []int32, embs [][]byte,
	outLayers, numLayers, pliDim, dModel int, eps float32,
) (metal.MTLBuffer, bool, error) {
	if outLayers < 1 || outLayers > numLayers {
		return nil, false, core.NewError("native.perLayerInputsBatchQuantDevice: layer bound out of range")
	}
	plDimC, ok, err := perLayerInputsBatchQuantEncode(sc, tablePacked, tableScales, tableBiases, tableGS, tableBits,
		projW, projWOff, projPacked, projScales, projBiases, projGS, projBits,
		projNormW, ids, embs, outLayers, numLayers, pliDim, dModel, eps, false)
	if err != nil || !ok || plDimC != outLayers*pliDim {
		return nil, ok && plDimC == outLayers*pliDim, err
	}
	return sc.projectedBuf, true, nil
}

func perLayerInputsBatchQuantEncode(
	sc *pleBatchScratch,
	tablePacked, tableScales, tableBiases metal.MTLBuffer, tableGS, tableBits int,
	projW metal.MTLBuffer, projWOff uint,
	projPacked, projScales, projBiases metal.MTLBuffer, projGS, projBits int,
	projNormW []byte,
	ids []int32, embs [][]byte,
	nOut, numLayers, pliDim, dModel int, eps float32,
	wait bool,
) (int, bool, error) {
	k := len(ids)
	plDim := numLayers * pliDim
	if tablePacked == nil || tableScales == nil || tableBiases == nil || k < steelGEMMMinRows || len(projNormW) != pliDim*bf16Size {
		return 0, false, nil
	}
	if projPacked == nil && projW == nil {
		return 0, false, nil
	}
	gatherPSO, gerr := embedGatherPipeline()
	relayoutPSO, rerr := pleRelayoutPipeline()
	if gerr != nil || rerr != nil {
		return 0, false, nil // kernels unavailable — the per-token loop still works
	}
	if projPacked != nil {
		if dModel%projGS != 0 {
			return 0, false, nil
		}
		if _, perr := pipelineFor(qmmTKernelName(plDim, projGS, projBits)); perr != nil {
			return 0, false, nil
		}
	}
	// The compute width: a bounded build (#381 shared-suffix chunk) runs the whole
	// pipeline at the owner-layer prefix when the rows gather + (quant path) the
	// bounded qmm_t instantiation resolve; otherwise compute full width — the
	// IntoSlab wrapper copies the layer-major prefix out either way (identical
	// bytes), and the Device wrapper declines a full-width result.
	rowsPSO, rowsErr := pleGatherRowsQuantPipeline()
	plDimC, nC := plDim, numLayers
	if rowsErr == nil && nOut < numLayers {
		bounded := true
		if projPacked != nil {
			if _, perr := pipelineFor(qmmTKernelName(nOut*pliDim, projGS, projBits)); perr != nil {
				bounded = false
			}
		}
		if bounded {
			plDimC, nC = nOut*pliDim, nOut
		}
	}
	embScale := float32(math.Sqrt(float64(pliDim)))
	projScale := float32(1.0 / math.Sqrt(float64(dModel)))
	ensureStart := time.Now()
	sc.ensure(k, plDim, dModel, projScale, gemma4PerLayerCombineScale)
	hostSpan("pleSlab.ensure", ensureStart, k)

	stageStart := time.Now()
	rowBytes := dModel * bf16Size
	for i, emb := range embs {
		if len(emb) != rowBytes {
			return 0, false, core.NewError("native.perLayerInputsBatchQuant: hidden row size mismatch")
		}
		copy(sc.hidden[i*rowBytes:(i+1)*rowBytes], emb)
	}
	copy(unsafe.Slice((*int32)(sc.idsBuf.Contents()), k), ids)
	copy(unsafe.Slice((*byte)(sc.hiddenBuf.Contents()), len(sc.hidden)), sc.hidden)
	hostSpan("pleSlab.stage", stageStart, k)

	gpuStart := time.Now()
	var encErr error
	engaged := false
	withAutoreleasePool(func() {
		cb := commandBufferFast(queue)
		enc := computeCommandEncoderFast(cb)
		if rowsErr == nil {
			// ONE dispatch dequant-gathers all K rows (prefix width, full-row strides)
			sink := encSink{enc}
			sink.setPSO(rowsPSO)
			sink.setBuf(sc.idsBuf, 0, 0)
			sink.setBuf(tablePacked, 0, 1)
			sink.setBuf(tableScales, 0, 2)
			sink.setBuf(tableBiases, 0, 3)
			sink.setBuf(sc.perLayerBuf, 0, 4)
			sink.setI32(int32(plDimC), 5)
			sink.setI32(int32(tableGS), 6)
			sink.setF32(embScale, 7)
			sink.setI32(int32(plDim*tableBits/8), 8)
			sink.setI32(int32(plDim/tableGS), 9)
			sink.setI32(int32(tableBits), 10)
			sink.dispatchThreads(
				metal.MTLSize{Width: uint(plDimC), Height: uint(k), Depth: 1},
				metal.MTLSize{Width: uint(elemGroupTG(plDimC)), Height: 1, Depth: 1},
			)
		} else {
			// legacy metallib: K per-token dequant-gathers (full width — plDimC == plDim here)
			for i := range k {
				emitEmbedGatherQuantAt(encSink{enc}, gatherPSO, sc.idsBuf, uint(i*4), tablePacked, tableScales, tableBiases, sc.perLayerBuf, uint(i*plDim*bf16Size), 0, 0, 0, plDim, tableGS, tableBits, embScale)
			}
		}
		// projected[K, plDimC] = hidden[K, dModel] @ projWᵀ — one weight sweep for the chunk
		// (a bounded pass sweeps only the owner layers' projection rows: a prefix of projW)
		if projPacked != nil {
			encErr = encQMMTBF16At(enc, projPacked, projScales, projBiases, sc.hiddenBuf, sc.projectedBuf, 0, 0, 0, 0, 0, k, plDimC, dModel, projGS, projBits)
		} else if !encGemmBF16NT(enc, projW, sc.hiddenBuf, sc.projectedBuf, projWOff, 0, 0, plDimC, dModel, k) {
			endEncodingFast(enc)
			commitCommandBufferFast(cb)
			waitUntilCompletedFast(cb)
			return // steel unavailable — fall back to the per-token loop
		}
		if encErr == nil {
			// ×1/√dModel → rms per (token,layer) row → +perLayer → ×1/√2, all batched
			if encErr = encMulRowsBF16(enc, sc.projectedBuf, sc.projScaleBuf, sc.projectedBuf, 0, 0, 0, k*plDimC, 1); encErr == nil {
				if encErr = encRMSNormRowsBF16(enc, sc.projectedBuf, residentBytes(projNormW), sc.normedBuf, 0, 0, 0, k*nC, pliDim, eps); encErr == nil {
					if encErr = encAddBF16To(enc, sc.normedBuf, sc.perLayerBuf, sc.normedBuf, 0, 0, 0, k*plDimC); encErr == nil {
						encErr = encMulRowsBF16(enc, sc.normedBuf, sc.combineScaleBuf, sc.normedBuf, 0, 0, 0, k*plDimC, 1)
					}
				}
			}
		}
		if encErr == nil {
			// token-major → layer-major on-device (projectedBuf is free after the rms consumed it)
			sink := encSink{enc}
			sink.setPSO(relayoutPSO)
			sink.setBuf(sc.normedBuf, 0, 0)
			sink.setBuf(sc.projectedBuf, 0, 1)
			sink.setI32(int32(k), 2)
			sink.setI32(int32(nC), 3)
			sink.setI32(int32(pliDim), 4)
			n := k * plDimC
			sink.dispatchThreads(
				metal.MTLSize{Width: uint(n), Height: 1, Depth: 1},
				metal.MTLSize{Width: uint(elemGroupTG(n)), Height: 1, Depth: 1},
			)
		}
		endEncodingFast(enc)
		commitCommandBufferFast(cb)
		if wait {
			waitUntilCompletedFast(cb)
		}
		engaged = encErr == nil
	})
	if encErr != nil || !engaged {
		return 0, false, encErr
	}
	hostSpan("pleSlab.gpu", gpuStart, k)
	return plDimC, true, nil
}

type plHostScratchKey struct {
	plDim, dModel int
	projScale     [2]byte
}

var plHostScratchPools sync.Map

func plHostScratchPoolForKey(key plHostScratchKey) *scratchLIFOPool[*plHostScratch] {
	if v, ok := plHostScratchPools.Load(key); ok {
		return v.(*scratchLIFOPool[*plHostScratch])
	}
	pool := &scratchLIFOPool[*plHostScratch]{}
	if v, loaded := plHostScratchPools.LoadOrStore(key, pool); loaded {
		return v.(*scratchLIFOPool[*plHostScratch])
	}
	return pool
}

type plHostScratch struct {
	hidden, perLayer                        *pinnedNoCopyBytes
	hiddenView, perLayerView                cachedNoCopyBytesView
	projected, scaled, projNormed, combined metal.MTLBuffer
	out                                     metal.MTLBuffer
	projScaleBuf, combineScaleBuf           metal.MTLBuffer
	projScaleBytes, combineScaleBytes       [2]byte
	outHost                                 []byte
	outHostPinned                           *pinnedNoCopyBytes
	plDim, dModel                           int
}

func newPLHostScratch(plDim, dModel int, projScale float32) (*plHostScratch, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	if plDim <= 0 || dModel <= 0 {
		return nil, core.NewError("native.newPLHostScratch: invalid dimensions")
	}
	hidden, err := newPinnedNoCopyBytes(dModel * bf16Size)
	if err != nil {
		return nil, err
	}
	perLayer, err := newPinnedNoCopyBytes(plDim * bf16Size)
	if err != nil {
		hidden.Close()
		return nil, err
	}
	nb := func() metal.MTLBuffer {
		return device.NewBufferWithLengthOptions(uint(plDim*bf16Size), metal.MTLResourceStorageModeShared)
	}
	s := &plHostScratch{
		hidden: hidden, perLayer: perLayer,
		projected: nb(), scaled: nb(), projNormed: nb(), combined: nb(), out: nb(),
		plDim: plDim, dModel: dModel,
	}
	s.projScaleBytes = bf16ScalarBytes(projScale)
	s.combineScaleBytes = bf16ScalarBytes(gemma4PerLayerCombineScale)
	s.projScaleBuf = bf16ConstBuffer(1, projScale)
	s.combineScaleBuf = bf16ConstBuffer(1, gemma4PerLayerCombineScale)
	return s, nil
}

func plHostScratchPoolFor(plDim, dModel int, projScale float32) *scratchLIFOPool[*plHostScratch] {
	return plHostScratchPoolForKey(plHostScratchKey{plDim: plDim, dModel: dModel, projScale: bf16ScalarBytes(projScale)})
}

func getPLHostScratch(plDim, dModel int, projScale float32) (*plHostScratch, error) {
	pool := plHostScratchPoolFor(plDim, dModel, projScale)
	if s := pool.Get(); s != nil {
		if s != nil &&
			s.plDim == plDim &&
			s.dModel == dModel &&
			s.hidden != nil &&
			s.perLayer != nil &&
			s.projected != nil &&
			s.scaled != nil &&
			s.projNormed != nil &&
			s.combined != nil &&
			s.out != nil &&
			s.projScaleBytes == bf16ScalarBytes(projScale) &&
			s.combineScaleBytes == bf16ScalarBytes(gemma4PerLayerCombineScale) {
			return s, nil
		}
		s.Close()
	}
	return newPLHostScratch(plDim, dModel, projScale)
}

func putPLHostScratch(s *plHostScratch) {
	if s != nil && s.plDim > 0 && s.dModel > 0 && s.hidden != nil && s.perLayer != nil && s.out != nil {
		plHostScratchPoolForKey(plHostScratchKey{plDim: s.plDim, dModel: s.dModel, projScale: s.projScaleBytes}).Put(s)
	}
}

func (s *plHostScratch) Close() {
	if s == nil {
		return
	}
	if s.hidden != nil {
		s.hidden.Close()
		s.hidden = nil
	}
	if s.perLayer != nil {
		s.perLayer.Close()
		s.perLayer = nil
	}
	s.hiddenView.Close()
	s.perLayerView.Close()
	s.projected, s.scaled, s.projNormed, s.combined, s.out = nil, nil, nil, nil, nil
	s.projScaleBuf, s.combineScaleBuf = nil, nil
	s.closeHostReadback()
}

func (s *plHostScratch) closeHostReadback() {
	if s == nil {
		return
	}
	if s.outHostPinned != nil {
		s.outHostPinned.Close()
		s.outHostPinned = nil
	}
	s.outHost = nil
}

func (s *plHostScratch) hostReadbackBuffer(n int) ([]byte, metal.MTLBuffer, error) {
	if s == nil {
		return nil, nil, core.NewError("native.plHostScratch.hostReadbackBuffer: scratch is nil")
	}
	if n <= 0 {
		return nil, nil, core.NewError("native.plHostScratch.hostReadbackBuffer: size must be > 0")
	}
	if s.outHostPinned == nil || len(s.outHostPinned.bytes) != n {
		s.closeHostReadback()
		pinned, err := newPinnedNoCopyBytes(n)
		if err != nil {
			return nil, nil, err
		}
		s.outHostPinned = pinned
	}
	s.outHost = s.outHostPinned.bytes[:n]
	return s.outHost, s.outHostPinned.buf, nil
}

// perLayerProjBatched runs the gemma4 PLE projection chain — steps 2-6 of PerLayerInputs: resident-weight
// matvec → ×projScale → RMSNorm(rows) → +perLayer → ×combineScale — as ONE command buffer: a single
// Commit()+WaitUntilCompleted() instead of five. That collapses five per-token GPU round-trips (~5×199µs ≈
// 1ms/token of host stall, GPU idle between) to one. The ops chain via device buffers (no per-op host
// download), driving the SAME kernels as the host path, so the result is byte-identical to the unbatched
// steps 2-6. Intermediate buffers are autoreleased (pool-freed); the projection weight is the resident
// no-copy shard view (projView). scratch, when supplied by a session, keeps the dynamic hidden/per-layer inputs
// in reusable pinned no-copy staging buffers and reuses intermediates plus the host readback across tokens.
func perLayerProjBatched(projView bufView, hidden, perLayer []byte, projScale float32, projNormW []byte, plDim, numLayers, pliDim, dModel int, eps float32, scratchArg ...*plHostScratch) ([]byte, error) {
	out, _, err := perLayerProjBatchedCore(projView, hidden, nil, perLayer, projScale, projNormW, plDim, numLayers, pliDim, dModel, eps, true, scratchArg...)
	return out, err
}

func perLayerProjBatchedResident(projView bufView, hidden, perLayer []byte, projScale float32, projNormW []byte, plDim, numLayers, pliDim, dModel int, eps float32, scratch *plHostScratch) (metal.MTLBuffer, error) {
	if scratch == nil {
		return nil, core.NewError("native.perLayerProjBatchedResident: scratch is required")
	}
	_, buf, err := perLayerProjBatchedCore(projView, hidden, nil, perLayer, projScale, projNormW, plDim, numLayers, pliDim, dModel, eps, false, scratch)
	return buf, err
}

func perLayerProjBatchedResidentBuffer(projView bufView, hiddenBuf metal.MTLBuffer, perLayer []byte, projScale float32, projNormW []byte, plDim, numLayers, pliDim, dModel int, eps float32, scratch *plHostScratch) (metal.MTLBuffer, error) {
	if scratch == nil {
		return nil, core.NewError("native.perLayerProjBatchedResidentBuffer: scratch is required")
	}
	if hiddenBuf == nil {
		return nil, core.NewError("native.perLayerProjBatchedResidentBuffer: hidden buffer is nil")
	}
	_, buf, err := perLayerProjBatchedCore(projView, nil, hiddenBuf, perLayer, projScale, projNormW, plDim, numLayers, pliDim, dModel, eps, false, scratch)
	return buf, err
}

func perLayerProjBatchedCore(projView bufView, hidden []byte, hiddenBufArg metal.MTLBuffer, perLayer []byte, projScale float32, projNormW []byte, plDim, numLayers, pliDim, dModel int, eps float32, readback bool, scratchArg ...*plHostScratch) ([]byte, metal.MTLBuffer, error) {
	if numLayers <= 0 || pliDim <= 0 || dModel <= 0 || plDim != numLayers*pliDim {
		return nil, nil, core.NewError("native.perLayerProjBatched: invalid dimensions")
	}
	if hiddenBufArg == nil && len(hidden) != dModel*bf16Size {
		return nil, nil, core.NewError("native.perLayerProjBatched: hidden must be dModel bf16 bytes")
	}
	if len(perLayer) != plDim*bf16Size {
		return nil, nil, core.NewError("native.perLayerProjBatched: perLayer must be numLayers*pliDim bf16 bytes")
	}
	if len(projNormW) != pliDim*bf16Size {
		return nil, nil, core.NewError("native.perLayerProjBatched: projNormW must be pliDim bf16 bytes")
	}
	if projView.buf == nil {
		return nil, nil, core.NewError("native.perLayerProjBatched: resident projection buffer is nil")
	}
	var scratch *plHostScratch
	if len(scratchArg) > 0 {
		scratch = scratchArg[0]
	}
	outLen := plDim * bf16Size
	var out []byte
	var residentOut metal.MTLBuffer
	directReadback := false
	var ferr error
	withAutoreleasePool(func() {
		projScaleBytes := bf16ScalarBytes(projScale)
		combineScaleBytes := bf16ScalarBytes(gemma4PerLayerCombineScale)
		var hiddenBuf, perLayerBuf, projNormWBuf, projScaleBuf, combineScaleBuf metal.MTLBuffer
		var projectedBuf, scaledBuf, projNormedBuf, combinedBuf, outBuf metal.MTLBuffer
		var ok bool
		if scratch != nil {
			if scratch.plDim != plDim || scratch.dModel != dModel {
				ferr = core.NewError("native.perLayerProjBatched: scratch dimension mismatch")
				return
			}
			if scratch.projScaleBytes != projScaleBytes || scratch.combineScaleBytes != combineScaleBytes {
				ferr = core.NewError("native.perLayerProjBatched: scratch scale mismatch")
				return
			}
			if hiddenBufArg != nil {
				hiddenBuf = hiddenBufArg
			} else {
				if hiddenBuf, ok = scratch.hiddenView.buffer(hidden); !ok {
					if hiddenBuf, ferr = scratch.hidden.copyBuffer(hidden); ferr != nil {
						return
					}
				}
			}
			if len(perLayer) == len(scratch.perLayer.bytes) && len(perLayer) > 0 && unsafe.Pointer(&perLayer[0]) == unsafe.Pointer(&scratch.perLayer.bytes[0]) {
				perLayerBuf = scratch.perLayer.buf
			} else {
				if perLayerBuf, ok = scratch.perLayerView.buffer(perLayer); !ok {
					if perLayerBuf, ferr = scratch.perLayer.copyBuffer(perLayer); ferr != nil {
						return
					}
				}
			}
			projNormWBuf = residentBytes(projNormW)
			projScaleBuf, combineScaleBuf = scratch.projScaleBuf, scratch.combineScaleBuf
			projectedBuf, scaledBuf, projNormedBuf, combinedBuf, outBuf = scratch.projected, scratch.scaled, scratch.projNormed, scratch.combined, scratch.out
			if readback {
				if out, outBuf, ferr = scratch.hostReadbackBuffer(outLen); ferr != nil {
					return
				}
				directReadback = true
			} else {
				scratch.closeHostReadback()
			}
		} else {
			mk := func(b []byte) metal.MTLBuffer {
				return device.NewBufferWithBytesLengthOptions(unsafe.Pointer(&b[0]), uint(len(b)), metal.MTLResourceStorageModeShared)
			}
			nb := func() metal.MTLBuffer {
				return device.NewBufferWithLengthOptions(uint(plDim*bf16Size), metal.MTLResourceStorageModeShared)
			}
			if hiddenBufArg != nil {
				hiddenBuf = hiddenBufArg
			} else {
				hiddenBuf = mk(hidden)
			}
			perLayerBuf = mk(perLayer)
			projNormWBuf = residentBytes(projNormW)
			projScaleBuf = mk(projScaleBytes[:])
			combineScaleBuf = mk(combineScaleBytes[:])
			projectedBuf, scaledBuf, projNormedBuf, combinedBuf, outBuf = nb(), nb(), nb(), nb(), nb()
			if readback {
				out = make([]byte, outLen)
			}
		}
		residentOut = outBuf

		cb := commandBufferFast(queue)
		enc := computeCommandEncoderFast(cb)
		encode := func() error {
			if err := encGemvBF16To(enc, projView.buf, hiddenBuf, projectedBuf, projView.off, 0, plDim, dModel); err != nil {
				return err
			}
			if err := encScaleBF16(enc, projectedBuf, projScaleBuf, scaledBuf, 0, projScaleBytes[:], plDim); err != nil {
				return err
			}
			if err := encRMSNormRowsBF16(enc, scaledBuf, projNormWBuf, projNormedBuf, 0, 0, 0, numLayers, pliDim, eps); err != nil {
				return err
			}
			if err := encAddBF16(enc, projNormedBuf, perLayerBuf, combinedBuf, plDim); err != nil {
				return err
			}
			return encScaleBF16(enc, combinedBuf, combineScaleBuf, outBuf, 0, combineScaleBytes[:], plDim)
		}
		ferr = encode()
		endEncodingFast(enc)
		if ferr != nil {
			return
		}
		commitCommandBufferFast(cb)
		waitUntilCompletedFast(cb)
		if readback && !directReadback {
			copy(out, unsafe.Slice((*byte)(outBuf.Contents()), outLen))
		}
	})
	return out, residentOut, ferr
}

func perLayerProjQuantBatched(q QuantWeight, hidden, perLayer []byte, projScale float32, projNormW []byte, plDim, numLayers, pliDim, dModel, groupSize, bits int, eps float32, scratchArg ...*plHostScratch) ([]byte, error) {
	out, _, err := perLayerProjQuantBatchedCore(q, hidden, nil, perLayer, projScale, projNormW, plDim, numLayers, pliDim, dModel, groupSize, bits, eps, true, scratchArg...)
	return out, err
}

func perLayerProjQuantBatchedResident(q QuantWeight, hidden, perLayer []byte, projScale float32, projNormW []byte, plDim, numLayers, pliDim, dModel, groupSize, bits int, eps float32, scratch *plHostScratch) (metal.MTLBuffer, error) {
	if scratch == nil {
		return nil, core.NewError("native.perLayerProjQuantBatchedResident: scratch is required")
	}
	_, buf, err := perLayerProjQuantBatchedCore(q, hidden, nil, perLayer, projScale, projNormW, plDim, numLayers, pliDim, dModel, groupSize, bits, eps, false, scratch)
	return buf, err
}

func perLayerProjQuantBatchedResidentBuffer(q QuantWeight, hiddenBuf metal.MTLBuffer, perLayer []byte, projScale float32, projNormW []byte, plDim, numLayers, pliDim, dModel, groupSize, bits int, eps float32, scratch *plHostScratch) (metal.MTLBuffer, error) {
	if scratch == nil {
		return nil, core.NewError("native.perLayerProjQuantBatchedResidentBuffer: scratch is required")
	}
	if hiddenBuf == nil {
		return nil, core.NewError("native.perLayerProjQuantBatchedResidentBuffer: hidden buffer is nil")
	}
	_, buf, err := perLayerProjQuantBatchedCore(q, nil, hiddenBuf, perLayer, projScale, projNormW, plDim, numLayers, pliDim, dModel, groupSize, bits, eps, false, scratch)
	return buf, err
}

func perLayerProjQuantBatchedCore(q QuantWeight, hidden []byte, hiddenBufArg metal.MTLBuffer, perLayer []byte, projScale float32, projNormW []byte, plDim, numLayers, pliDim, dModel, groupSize, bits int, eps float32, readback bool, scratchArg ...*plHostScratch) ([]byte, metal.MTLBuffer, error) {
	if numLayers <= 0 || pliDim <= 0 || dModel <= 0 || plDim != numLayers*pliDim {
		return nil, nil, core.NewError("native.perLayerProjQuantBatched: invalid dimensions")
	}
	if hiddenBufArg == nil && len(hidden) != dModel*bf16Size {
		return nil, nil, core.NewError("native.perLayerProjQuantBatched: hidden must be dModel bf16 bytes")
	}
	if len(perLayer) != plDim*bf16Size {
		return nil, nil, core.NewError("native.perLayerProjQuantBatched: perLayer must be numLayers*pliDim bf16 bytes")
	}
	if len(projNormW) != pliDim*bf16Size {
		return nil, nil, core.NewError("native.perLayerProjQuantBatched: projNormW must be pliDim bf16 bytes")
	}
	groupSize, bits = quantWeightGeometryForShape(q, plDim, dModel, groupSize, bits)
	if groupSize <= 0 || bits <= 0 || dModel%groupSize != 0 {
		return nil, nil, core.NewError("native.perLayerProjQuantBatched: invalid quant geometry")
	}
	wantPacked := plDim * dModel * bits / 8
	wantSB := plDim * (dModel / groupSize) * bf16Size
	if len(q.Packed) != wantPacked || len(q.Scales) != wantSB || len(q.Biases) != wantSB {
		return nil, nil, core.NewError("native.perLayerProjQuantBatched: quant projection size mismatch")
	}
	var scratch *plHostScratch
	if len(scratchArg) > 0 {
		scratch = scratchArg[0]
	}
	outLen := plDim * bf16Size
	var out []byte
	var residentOut metal.MTLBuffer
	directReadback := false
	var ferr error
	withAutoreleasePool(func() {
		projScaleBytes := bf16ScalarBytes(projScale)
		combineScaleBytes := bf16ScalarBytes(gemma4PerLayerCombineScale)
		var hiddenBuf, perLayerBuf, projNormWBuf, projScaleBuf, combineScaleBuf metal.MTLBuffer
		var projectedBuf, scaledBuf, projNormedBuf, combinedBuf, outBuf metal.MTLBuffer
		var ok bool
		if scratch != nil {
			if scratch.plDim != plDim || scratch.dModel != dModel {
				ferr = core.NewError("native.perLayerProjQuantBatched: scratch dimension mismatch")
				return
			}
			if scratch.projScaleBytes != projScaleBytes || scratch.combineScaleBytes != combineScaleBytes {
				ferr = core.NewError("native.perLayerProjQuantBatched: scratch scale mismatch")
				return
			}
			if hiddenBufArg != nil {
				hiddenBuf = hiddenBufArg
			} else {
				if hiddenBuf, ok = scratch.hiddenView.buffer(hidden); !ok {
					if hiddenBuf, ferr = scratch.hidden.copyBuffer(hidden); ferr != nil {
						return
					}
				}
			}
			if len(perLayer) == len(scratch.perLayer.bytes) && len(perLayer) > 0 && unsafe.Pointer(&perLayer[0]) == unsafe.Pointer(&scratch.perLayer.bytes[0]) {
				perLayerBuf = scratch.perLayer.buf
			} else {
				if perLayerBuf, ok = scratch.perLayerView.buffer(perLayer); !ok {
					if perLayerBuf, ferr = scratch.perLayer.copyBuffer(perLayer); ferr != nil {
						return
					}
				}
			}
			projNormWBuf = residentBytes(projNormW)
			projScaleBuf, combineScaleBuf = scratch.projScaleBuf, scratch.combineScaleBuf
			projectedBuf, scaledBuf, projNormedBuf, combinedBuf, outBuf = scratch.projected, scratch.scaled, scratch.projNormed, scratch.combined, scratch.out
			if readback {
				if out, outBuf, ferr = scratch.hostReadbackBuffer(outLen); ferr != nil {
					return
				}
				directReadback = true
			} else {
				scratch.closeHostReadback()
			}
		} else {
			mk := func(b []byte) metal.MTLBuffer {
				return device.NewBufferWithBytesLengthOptions(unsafe.Pointer(&b[0]), uint(len(b)), metal.MTLResourceStorageModeShared)
			}
			nb := func() metal.MTLBuffer {
				return device.NewBufferWithLengthOptions(uint(plDim*bf16Size), metal.MTLResourceStorageModeShared)
			}
			if hiddenBufArg != nil {
				hiddenBuf = hiddenBufArg
			} else {
				hiddenBuf = mk(hidden)
			}
			perLayerBuf = mk(perLayer)
			projNormWBuf = residentBytes(projNormW)
			projScaleBuf = mk(projScaleBytes[:])
			combineScaleBuf = mk(combineScaleBytes[:])
			projectedBuf, scaledBuf, projNormedBuf, combinedBuf, outBuf = nb(), nb(), nb(), nb(), nb()
			if readback {
				out = make([]byte, outLen)
			}
		}
		residentOut = outBuf

		wBuf, scalesBuf, biasesBuf := quantWeightViews(q)
		cb := commandBufferFast(queue)
		enc := computeCommandEncoderFast(cb)
		encode := func() error {
			if err := encQMVBF16(enc, wBuf.buf, scalesBuf.buf, biasesBuf.buf, hiddenBuf, projectedBuf, wBuf.off, scalesBuf.off, biasesBuf.off, 0, plDim, dModel, groupSize, bits); err != nil {
				return err
			}
			if err := encScaleBF16(enc, projectedBuf, projScaleBuf, scaledBuf, 0, projScaleBytes[:], plDim); err != nil {
				return err
			}
			if err := encRMSNormRowsBF16(enc, scaledBuf, projNormWBuf, projNormedBuf, 0, 0, 0, numLayers, pliDim, eps); err != nil {
				return err
			}
			if err := encAddBF16(enc, projNormedBuf, perLayerBuf, combinedBuf, plDim); err != nil {
				return err
			}
			return encScaleBF16(enc, combinedBuf, combineScaleBuf, outBuf, 0, combineScaleBytes[:], plDim)
		}
		ferr = encode()
		endEncodingFast(enc)
		if ferr != nil {
			return
		}
		commitCommandBufferFast(cb)
		waitUntilCompletedFast(cb)
		if readback && !directReadback {
			copy(out, unsafe.Slice((*byte)(outBuf.Contents()), outLen))
		}
	})
	return out, residentOut, ferr
}
