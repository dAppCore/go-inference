// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"os"
	"sync"
	"unsafe"

	core "dappco.re/go"
	"dappco.re/go/inference/model"
	"dappco.re/go/inference/model/arch/Qwen/qwen3"
	"dappco.re/go/inference/model/composed"
	"github.com/tmc/apple/metal"
)

// composed_bf16_backend.go — the dense bf16 matvec seam (#26): the exact quant-seam shape
// (composed_quant_backend.go) for a checkpoint's UNQUANTISED bf16 projections. The weight bytes
// stay the mmap view (residentBytes caches the device buffer per slice); activations cross the
// seam f32 exactly like the quant seam, cast to bf16 for the kernel. This is what stops the dense
// composed lane widening every projection to f32 — the ×2 on bytes streamed per token AND on the
// resident set that put the official bf16 Qwen exports at ×4-6 behind mlx-lm.
// bf16SeamEnabled gates the device bf16 matvec seam. Default on; LTHN_BF16_SEAM=0 leaves the hooks
// unbound so the row-widen host floor serves — the same-binary A/B arm (the LTHN_GD_BLOCK shape).
var bf16SeamEnabled = os.Getenv("LTHN_BF16_SEAM") != "0"

func init() {
	if bf16SeamEnabled {
		composed.ProjBF16MatMulInto = MatMulBF16WeightF32NTInto
		qwen3.ProjBF16MatMulInto = MatMulBF16WeightF32NTInto
		qwen3.GatedDeltaBF16LayerDevice = gatedDeltaBF16LayerDeviceHook // the WHOLE dense bf16 layer in one CB (#26)
		composed.AttnBF16FrontDevice = AttnBF16FrontDevice              // attention front: norm + q/k/v in one CB (#26/#18 S6a)
		composed.AttnBF16TailDevice = AttnBF16TailDevice                // attention tail: o_proj + FFN tail in one CB
		composed.AttnQuantFrontDevice = AttnQuantFrontDevice            // the packed twins — the 27B's attention layers
		composed.AttnQuantTailDevice = AttnQuantTailDevice
		if os.Getenv("LTHN_ATTN_DEVKV") != "0" {
			// Device-KV full attention layer (#26): the whole layer in one CB over the resident
			// cache. Default on since the sigmoid-gate fix (the transformers qwen3_5 reference
			// hardcodes sigmoid; an earlier silu reading diverged on gated models);
			// LTHN_ATTN_DEVKV=0 is the same-binary A/B arm.
			composed.AttnBF16FullLayerDevice = AttnBF16FullLayerDevice
			composed.AttnKVExportDevice = attnKVExportHook
			composed.ComposedChainBeginDevice = ComposedChainBeginDevice   // whole-token chain (#26): one upload, one wait
			composed.ComposedChainEndDevice = ComposedChainEndDevice
			composed.AttnBF16ChainLayerDevice = attnBF16ChainLayerDevice
			qwen3.GatedDeltaBF16ChainLayerDevice = gatedDeltaBF16ChainLayerDevice
		}
	}
}

// MatMulBF16WF32NTInto computes out[M,N] = x[M,K] @ wᵀ for a dense bf16 weight (w = raw row-major
// bf16 bytes [N,K]) with f32 activations at the seam. M=1 (decode) rides the MLX bf16 gemv over the
// resident weight bytes; M>1 currently loops rows through the same gemv (correct; the batched
// prefill slab is the follow-up). out is reused when cap(out) >= M*N.
func MatMulBF16WF32NTInto(out, x []float32, w []byte, M, K, N int) ([]float32, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	if M <= 0 || N <= 0 || K <= 0 {
		return nil, core.NewError("native.MatMulBF16WF32NTInto: M, N, K must be positive")
	}
	if len(x) != M*K {
		return nil, core.NewError("native.MatMulBF16WF32NTInto: len(x) must equal M*K")
	}
	if len(w) != N*K*bf16Size {
		return nil, core.NewError("native.MatMulBF16WF32NTInto: len(w) must equal N*K*2 bytes")
	}
	if cap(out) < M*N {
		out = make([]float32, M*N)
	} else {
		out = out[:M*N]
	}
	xb := f32sToBF16Bytes(x)
	var ob []byte
	for m := 0; m < M; m++ {
		row, err := MatVecBF16Into(ob, w, xb[m*K*bf16Size:(m+1)*K*bf16Size], N, K)
		if err != nil {
			return nil, err
		}
		ob = row
		for n := 0; n < N; n++ {
			out[m*N+n] = bf16ToF32(row[n*bf16Size], row[n*bf16Size+1])
		}
	}
	return out, nil
}

// MatMulBF16WeightF32NTInto is MatMulBF16WF32NTInto over the lib's BF16Weight form — the signature
// the composed/qwen3 hooks bind.
func MatMulBF16WeightF32NTInto(out, x []float32, w *model.BF16Weight, M, K, N int) ([]float32, error) {
	if w == nil || w.OutDim != N || w.InDim != K {
		return nil, core.NewError("native.MatMulBF16WeightF32NTInto: weight geometry mismatch")
	}
	return MatMulBF16WF32NTInto(out, x, w.Data, M, K, N)
}

// --- the dense bf16 whole-layer CB (#26 speed half): the S3 fold with gemv emitters in the qmv slots ---

// bf16GeometryOK verifies a raw bf16 projection against the fold's expected [outDim, inDim] shape.
func bf16GeometryOK(w *model.BF16Weight, outDim, inDim int) bool {
	return w != nil && w.OutDim == outDim && w.InDim == inDim && len(w.Data) == outDim*inDim*bf16Size
}

// encProjBF16From encodes one raw-bf16 projection from ALREADY-CAST bf16 rows: the MLX bf16 gemv
// (L=1) or the batched grid-Z gemv / steel GEMM (L>1) over the resident weight bytes, widened back
// to f32 — the gemv twin of encProjQuantBF16In.
func encProjBF16From(enc metal.MTLComputeCommandEncoder, w *model.BF16Weight, xBF, dstBF, dst metal.MTLBuffer, L, outDim, inDim int) error {
	mat := residentBytes(w.Data)
	var perr error
	if L == 1 {
		perr = encGemvBF16VecAt(enc, mat, xBF, dstBF, 0, 0, 0, outDim, inDim)
	} else {
		perr = encGemvBF16BatchedAt(enc, mat, xBF, dstBF, 0, 0, 0, outDim, inDim, L)
	}
	if perr != nil {
		return perr
	}
	memoryBarrier(enc, metal.MTLBarrierScopeBuffers)
	return encWidenBF16ToF32(enc, dstBF, dst, L*outDim)
}

// encProjBF16F32 is encProjBF16From with the f32→bf16 input cast included — the gemv twin of
// encProjQuantF32.
func encProjBF16F32(enc metal.MTLComputeCommandEncoder, w *model.BF16Weight, x, xBF, dstBF, dst metal.MTLBuffer, L, outDim, inDim int) error {
	if err := encNarrowF32ToBF16(enc, x, xBF, L*inDim); err != nil {
		return err
	}
	memoryBarrier(enc, metal.MTLBarrierScopeBuffers)
	return encProjBF16From(enc, w, xBF, dstBF, dst, L, outDim, inDim)
}

// encResidualNormMLPBF16Tail is the packed tail's raw-bf16 twin: the shared tail core with gemv
// projections over the checkpoint's own gate/up/down bytes.
func encResidualNormMLPBF16Tail(enc metal.MTLComputeCommandEncoder, tb quantTailBufs, normW metal.MTLBuffer, gate, up, down *model.BF16Weight, L, D, FF int, eps float32) error {
	return encResidualNormMLPTailCore(enc, tb, normW,
		func(enc metal.MTLComputeCommandEncoder, xBF, dstBF, dst metal.MTLBuffer, L, outDim, inDim int) error {
			return encProjBF16From(enc, gate, xBF, dstBF, dst, L, outDim, inDim)
		},
		func(enc metal.MTLComputeCommandEncoder, xBF, dstBF, dst metal.MTLBuffer, L, outDim, inDim int) error {
			return encProjBF16From(enc, up, xBF, dstBF, dst, L, outDim, inDim)
		},
		func(enc metal.MTLComputeCommandEncoder, x, xBF, dstBF, dst metal.MTLBuffer, L, outDim, inDim int) error {
			return encProjBF16F32(enc, down, x, xBF, dstBF, dst, L, outDim, inDim)
		}, L, D, FF, eps)
}

// gatedDeltaBF16LayerRun runs one WHOLE dense bf16 gated-delta layer in a single command buffer —
// gatedDeltaQuantLayerRun with the checkpoint's raw bf16 bytes riding the gemv where the packed
// codes ride the affine qmv: input RMSNorm → the five bf16 projections (cast-once/project-many) →
// the gated-delta block stages (state resident) → out_proj → the bf16 FFN tail. x [L,D] is the
// only upload, y [L,D] the only readback; the per-stage bf16 path pays seven CB round-trips.
func gatedDeltaBF16LayerRun(
	h *gatedDeltaDeviceState,
	x, inputNorm []float32,
	w *qwen3.GatedDeltaWeights,
	postNorm []float32,
	gate, up, down *model.BF16Weight,
	L, D, FF int, eps float32,
	priorConv, priorDelta []float32,
	y []float32,
) error {
	if err := ensureInit(); err != nil {
		return err
	}
	if h == nil || !gatedDeltaBlockUsable(h.Dk, h.Dv, h.Hk, h.Hv, h.K) {
		return core.NewError("native.gatedDeltaBF16LayerRun: geometry not servable")
	}
	convDim, vDim := h.convDim, h.Hv*h.Dv
	if L <= 0 || len(x) != L*D || len(y) != L*D || len(inputNorm) != D || len(postNorm) != D ||
		len(w.ALog) != h.Hv || len(w.DtBias) != h.Hv || len(w.Norm) != h.Dv ||
		(w.ConvBias != nil && len(w.ConvBias) != convDim) || len(w.ConvWeight) != convDim*h.K {
		return core.NewError("native.gatedDeltaBF16LayerRun: size mismatch")
	}
	if !bf16GeometryOK(w.InProjQKVB, convDim, D) || !bf16GeometryOK(w.InProjZB, vDim, D) ||
		!bf16GeometryOK(w.InProjAB, h.Hv, D) || !bf16GeometryOK(w.InProjBB, h.Hv, D) ||
		!bf16GeometryOK(w.OutProjB, D, vDim) ||
		!bf16GeometryOK(gate, FF, D) || !bf16GeometryOK(up, FF, D) || !bf16GeometryOK(down, D, FF) {
		return core.NewError("native.gatedDeltaBF16LayerRun: unsupported bf16 geometry")
	}
	if !h.valid {
		h.prime(priorConv, priorDelta)
	}
	rmsName := "rmsfloat32"
	if D > rmsLoopedLimit {
		rmsName = "rms_loopedfloat32"
	}
	psoRMS, err := pipelineFor(rmsName)
	if err != nil {
		return err
	}
	key := gatedDeltaQuantLayerKey{L: L, D: D, FF: FF, Hk: h.Hk, Hv: h.Hv, Dk: h.Dk, K: h.K}
	var encErr error
	withAutoreleasePool(func() {
		sc, gerr := getGatedDeltaQuantLayerScratch(key)
		if gerr != nil {
			encErr = gerr
			return
		}
		defer putGatedDeltaQuantLayerScratch(key, sc)
		xBuf, cerr := sc.x.copyBuffer(float32Bytes(x))
		if cerr != nil {
			encErr = cerr
			return
		}
		wConv := residentFloat32(w.ConvWeight)
		wBias := wConv
		hasBias := 0
		if w.ConvBias != nil {
			wBias = residentFloat32(w.ConvBias)
			hasBias = 1
		}
		wALog := residentFloat32(w.ALog)
		wDt := residentFloat32(w.DtBias)
		wNorm := residentFloat32(w.Norm)
		inNormBuf := residentFloat32(inputNorm)
		postNormBuf := residentFloat32(postNorm)

		cb := commandBufferFast(queue)
		enc := computeCommandEncoderFast(cb)
		fail := func(err error) {
			encErr = err
			endEncodingFast(enc)
		}

		emitRMSNormRows(encSink{enc}, psoRMS, xBuf, inNormBuf, sc.normed1.buf, 0, 0, 0, D, eps, L, rmsThreadgroup(D, psoRMS))
		memoryBarrier(enc, metal.MTLBarrierScopeBuffers)
		if err := encNarrowF32ToBF16(enc, sc.normed1.buf, sc.n1BF.buf, L*D); err != nil {
			fail(err)
			return
		}
		memoryBarrier(enc, metal.MTLBarrierScopeBuffers)
		if err := encProjBF16From(enc, w.InProjQKVB, sc.n1BF.buf, sc.qkvBF.buf, sc.qkv.buf, L, convDim, D); err != nil {
			fail(err)
			return
		}
		if err := encProjBF16From(enc, w.InProjZB, sc.n1BF.buf, sc.zBF.buf, sc.z.buf, L, vDim, D); err != nil {
			fail(err)
			return
		}
		if err := encProjBF16From(enc, w.InProjAB, sc.n1BF.buf, sc.aBF.buf, sc.a.buf, L, h.Hv, D); err != nil {
			fail(err)
			return
		}
		if err := encProjBF16From(enc, w.InProjBB, sc.n1BF.buf, sc.bBF.buf, sc.b.buf, L, h.Hv, D); err != nil {
			fail(err)
			return
		}
		memoryBarrier(enc, metal.MTLBarrierScopeBuffers)

		if err := encGatedDeltaBlockStages(enc, h, gdBlockStageBufs{
			qkv: sc.qkv.buf, z: sc.z.buf, a: sc.a.buf, b: sc.b.buf,
			qN: sc.qN.buf, kN: sc.kN.buf, vN: sc.vN.buf, g: sc.g.buf, beta: sc.beta.buf,
			gated: sc.gated.buf,
		}, wConv, wBias, hasBias, wALog, wDt, wNorm, L); err != nil {
			fail(err)
			return
		}
		memoryBarrier(enc, metal.MTLBarrierScopeBuffers)

		if err := encProjBF16F32(enc, w.OutProjB, sc.gated.buf, sc.gatedBF.buf, sc.mixBF.buf, sc.mix.buf, L, D, vDim); err != nil {
			fail(err)
			return
		}
		memoryBarrier(enc, metal.MTLBarrierScopeBuffers)

		if err := encResidualNormMLPBF16Tail(enc, quantTailBufs{
			h: xBuf, mix: sc.mix.buf, normed: sc.normed2.buf, nBF: sc.n2BF.buf,
			g: sc.gFF.buf, gBF: sc.gFFBF.buf, u: sc.uFF.buf, uBF: sc.uFFBF.buf, s: sc.sFF.buf, out: sc.out.buf,
		}, postNormBuf, gate, up, down, L, D, FF, eps); err != nil {
			fail(err)
			return
		}

		endEncodingFast(enc)
		commitCommandBufferFast(cb)
		waitUntilCompletedFast(cb)
		copy(y, unsafe.Slice((*float32)(unsafe.Pointer(&sc.out.bytes[0])), L*D))
	})
	return encErr
}

// gatedDeltaBF16LayerDeviceHook adapts gatedDeltaBF16LayerRun to qwen3's declared bf16 layer seam —
// the same handle discipline as the quant layer hook.
func gatedDeltaBF16LayerDeviceHook(sc *qwen3.GatedDeltaScratch, x, inputNorm []float32, w *qwen3.GatedDeltaWeights, cfg qwen3.GatedDeltaConfig, postNorm []float32, gate, up, down *model.BF16Weight, L, D, FF int, eps float32, priorConv, priorDelta []float32) ([]float32, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	h, _ := sc.Device.(*gatedDeltaDeviceState)
	if h == nil {
		if !gatedDeltaBlockUsable(cfg.HeadDim, cfg.HeadDim, cfg.KeyHeads, cfg.ValueHeads, cfg.ConvKernel) {
			return nil, core.NewError("native.gatedDeltaBF16LayerDeviceHook: geometry not servable")
		}
		nh, err := newGatedDeltaDeviceState(cfg.KeyHeads, cfg.ValueHeads, cfg.HeadDim, cfg.HeadDim, cfg.ConvKernel, 1)
		if err != nil {
			return nil, err
		}
		h = nh
	}
	y := make([]float32, L*D)
	if err := gatedDeltaBF16LayerRun(h, x, inputNorm, w, postNorm, gate, up, down, L, D, FF, eps, priorConv, priorDelta, y); err != nil {
		return nil, err
	}
	sc.Device = h
	return y, nil
}

// --- the attention fold (#26 / #18 S6a step 1): two CBs around the host attention core ---

// attnBF16FrontScratch stages one [norm → q/k/v] front CB: the x upload, the normed rows + their
// bf16 cast, and the three projection outputs (bf16 + widened f32 readback).
type attnBF16FrontScratch struct {
	x, normed, q, k, v *pinnedNoCopyBytes
	nBF, qBF, kBF, vBF *pinnedNoCopyBytes
}

type attnBF16FrontKey struct{ L, D, qCols, kvCols int }

var attnBF16FrontPools sync.Map // attnBF16FrontKey -> *sync.Pool

func getAttnBF16FrontScratch(key attnBF16FrontKey) (*attnBF16FrontScratch, error) {
	poolAny, ok := attnBF16FrontPools.Load(key)
	if !ok {
		poolAny, _ = attnBF16FrontPools.LoadOrStore(key, &sync.Pool{})
	}
	pool := poolAny.(*sync.Pool)
	if v := pool.Get(); v != nil {
		return v.(*attnBF16FrontScratch), nil
	}
	sc := &attnBF16FrontScratch{}
	var err error
	alloc := func(n int) *pinnedNoCopyBytes {
		if err != nil {
			return nil
		}
		var buf *pinnedNoCopyBytes
		buf, err = newPinnedNoCopyBytes(n)
		return buf
	}
	sc.x = alloc(key.L * key.D * 4)
	sc.normed = alloc(key.L * key.D * 4)
	sc.nBF = alloc(key.L * key.D * bf16Size)
	sc.q = alloc(key.L * key.qCols * 4)
	sc.qBF = alloc(key.L * key.qCols * bf16Size)
	sc.k = alloc(key.L * key.kvCols * 4)
	sc.kBF = alloc(key.L * key.kvCols * bf16Size)
	sc.v = alloc(key.L * key.kvCols * 4)
	sc.vBF = alloc(key.L * key.kvCols * bf16Size)
	if err != nil {
		return nil, err
	}
	return sc, nil
}

func putAttnBF16FrontScratch(key attnBF16FrontKey, sc *attnBF16FrontScratch) {
	if v, ok := attnBF16FrontPools.Load(key); ok {
		v.(*sync.Pool).Put(sc)
	}
}

// AttnBF16FrontDevice runs the attention layer's FRONT in one command buffer — input RMSNorm and
// the three raw-bf16 projections (cast-once/project-many gemv over resident views) — returning
// q/k/v for the host attention core (rope + cache + SDPA stay host until the device-KV slice).
// The per-stage path pays a host norm + three seam round-trips.
func AttnBF16FrontDevice(x, inputNorm []float32, qw, kw, vw *model.BF16Weight, L, D, qCols, kvCols int, eps float32) (q, k, v []float32, err error) {
	if !bf16GeometryOK(qw, qCols, D) || !bf16GeometryOK(kw, kvCols, D) || !bf16GeometryOK(vw, kvCols, D) {
		return nil, nil, nil, core.NewError("native.AttnBF16FrontDevice: bf16 geometry mismatch")
	}
	return attnFrontRunCore(x, inputNorm,
		func(enc metal.MTLComputeCommandEncoder, xBF, dstBF, dst metal.MTLBuffer, L, outDim, inDim int) error {
			return encProjBF16From(enc, qw, xBF, dstBF, dst, L, outDim, inDim)
		},
		func(enc metal.MTLComputeCommandEncoder, xBF, dstBF, dst metal.MTLBuffer, L, outDim, inDim int) error {
			return encProjBF16From(enc, kw, xBF, dstBF, dst, L, outDim, inDim)
		},
		func(enc metal.MTLComputeCommandEncoder, xBF, dstBF, dst metal.MTLBuffer, L, outDim, inDim int) error {
			return encProjBF16From(enc, vw, xBF, dstBF, dst, L, outDim, inDim)
		}, L, D, qCols, kvCols, eps)
}

// AttnQuantFrontDevice is the front's PACKED twin: the same one-CB norm + q/k/v, the affine qmv
// over checkpoint codes in the gemv's slot — the 27B's 16 attention layers' per-projection quant
// seams collapse into it.
func AttnQuantFrontDevice(x, inputNorm []float32, qw, kw, vw *model.QuantWeight, L, D, qCols, kvCols int, eps float32) (q, k, v []float32, err error) {
	if !quantGeometryOK(qw, qCols, D) || !quantGeometryOK(kw, kvCols, D) || !quantGeometryOK(vw, kvCols, D) {
		return nil, nil, nil, core.NewError("native.AttnQuantFrontDevice: quant geometry mismatch")
	}
	return attnFrontRunCore(x, inputNorm,
		func(enc metal.MTLComputeCommandEncoder, xBF, dstBF, dst metal.MTLBuffer, L, outDim, inDim int) error {
			return encProjQuantBF16In(enc, qw, xBF, dstBF, dst, L, outDim, inDim)
		},
		func(enc metal.MTLComputeCommandEncoder, xBF, dstBF, dst metal.MTLBuffer, L, outDim, inDim int) error {
			return encProjQuantBF16In(enc, kw, xBF, dstBF, dst, L, outDim, inDim)
		},
		func(enc metal.MTLComputeCommandEncoder, xBF, dstBF, dst metal.MTLBuffer, L, outDim, inDim int) error {
			return encProjQuantBF16In(enc, vw, xBF, dstBF, dst, L, outDim, inDim)
		}, L, D, qCols, kvCols, eps)
}

// attnFrontRunCore is the weight-form-agnostic front body.
func attnFrontRunCore(x, inputNorm []float32, projQ, projK, projV tailProjFromBF16, L, D, qCols, kvCols int, eps float32) (q, k, v []float32, err error) {
	if err := ensureInit(); err != nil {
		return nil, nil, nil, err
	}
	if L <= 0 || len(x) != L*D || len(inputNorm) != D {
		return nil, nil, nil, core.NewError("native.attnFrontRunCore: size mismatch")
	}
	rmsName := "rmsfloat32"
	if D > rmsLoopedLimit {
		rmsName = "rms_loopedfloat32"
	}
	psoRMS, perr := pipelineFor(rmsName)
	if perr != nil {
		return nil, nil, nil, perr
	}
	q = make([]float32, L*qCols)
	k = make([]float32, L*kvCols)
	v = make([]float32, L*kvCols)
	key := attnBF16FrontKey{L: L, D: D, qCols: qCols, kvCols: kvCols}
	var encErr error
	withAutoreleasePool(func() {
		sc, gerr := getAttnBF16FrontScratch(key)
		if gerr != nil {
			encErr = gerr
			return
		}
		defer putAttnBF16FrontScratch(key, sc)
		xBuf, cerr := sc.x.copyBuffer(float32Bytes(x))
		if cerr != nil {
			encErr = cerr
			return
		}
		normBuf := residentFloat32(inputNorm)
		cb := commandBufferFast(queue)
		enc := computeCommandEncoderFast(cb)
		fail := func(err error) {
			encErr = err
			endEncodingFast(enc)
		}
		emitRMSNormRows(encSink{enc}, psoRMS, xBuf, normBuf, sc.normed.buf, 0, 0, 0, D, eps, L, rmsThreadgroup(D, psoRMS))
		memoryBarrier(enc, metal.MTLBarrierScopeBuffers)
		if err := encNarrowF32ToBF16(enc, sc.normed.buf, sc.nBF.buf, L*D); err != nil {
			fail(err)
			return
		}
		memoryBarrier(enc, metal.MTLBarrierScopeBuffers)
		if err := projQ(enc, sc.nBF.buf, sc.qBF.buf, sc.q.buf, L, qCols, D); err != nil {
			fail(err)
			return
		}
		if err := projK(enc, sc.nBF.buf, sc.kBF.buf, sc.k.buf, L, kvCols, D); err != nil {
			fail(err)
			return
		}
		if err := projV(enc, sc.nBF.buf, sc.vBF.buf, sc.v.buf, L, kvCols, D); err != nil {
			fail(err)
			return
		}
		endEncodingFast(enc)
		commitCommandBufferFast(cb)
		waitUntilCompletedFast(cb)
		copy(q, unsafe.Slice((*float32)(unsafe.Pointer(&sc.q.bytes[0])), L*qCols))
		copy(k, unsafe.Slice((*float32)(unsafe.Pointer(&sc.k.bytes[0])), L*kvCols))
		copy(v, unsafe.Slice((*float32)(unsafe.Pointer(&sc.v.bytes[0])), L*kvCols))
	})
	if encErr != nil {
		return nil, nil, nil, encErr
	}
	return q, k, v, nil
}

// attnBF16TailScratch stages one [o_proj → FFN tail] CB: the attention-output upload + its bf16
// cast, the o_proj output pair, the residual upload, and the shared tail stage buffers.
type attnBF16TailScratch struct {
	h, attn, mix               *pinnedNoCopyBytes
	attnBF, mixBF              *pinnedNoCopyBytes
	normed, nBF, g, gBF        *pinnedNoCopyBytes
	u, uBF, s, out             *pinnedNoCopyBytes
}

type attnBF16TailKey struct{ L, D, mixCols, FF int }

var attnBF16TailPools sync.Map // attnBF16TailKey -> *sync.Pool

func getAttnBF16TailScratch(key attnBF16TailKey) (*attnBF16TailScratch, error) {
	poolAny, ok := attnBF16TailPools.Load(key)
	if !ok {
		poolAny, _ = attnBF16TailPools.LoadOrStore(key, &sync.Pool{})
	}
	pool := poolAny.(*sync.Pool)
	if v := pool.Get(); v != nil {
		return v.(*attnBF16TailScratch), nil
	}
	sc := &attnBF16TailScratch{}
	var err error
	alloc := func(n int) *pinnedNoCopyBytes {
		if err != nil {
			return nil
		}
		var buf *pinnedNoCopyBytes
		buf, err = newPinnedNoCopyBytes(n)
		return buf
	}
	L, D, FF := key.L, key.D, key.FF
	sc.h = alloc(L * D * 4)
	sc.attn = alloc(L * key.mixCols * 4)
	sc.attnBF = alloc(L * key.mixCols * bf16Size)
	sc.mix = alloc(L * D * 4)
	sc.mixBF = alloc(L * D * bf16Size)
	sc.normed = alloc(L * D * 4)
	sc.nBF = alloc(L * max(D, FF) * bf16Size)
	sc.g = alloc(L * FF * 4)
	sc.gBF = alloc(L * FF * bf16Size)
	sc.u = alloc(L * FF * 4)
	sc.uBF = alloc(L * FF * bf16Size)
	sc.s = alloc(L * FF * 4)
	sc.out = alloc(L * D * 4)
	if err != nil {
		return nil, err
	}
	return sc, nil
}

func putAttnBF16TailScratch(key attnBF16TailKey, sc *attnBF16TailScratch) {
	if v, ok := attnBF16TailPools.Load(key); ok {
		v.(*sync.Pool).Put(sc)
	}
}

// AttnBF16TailDevice runs the attention layer's TAIL in one command buffer — o_proj over the raw
// bf16 weight, then the shared FFN tail (residual + post-norm + bf16 SwiGLU + residual). Stateless
// (the KV cache advanced in the host core), so a device decline falls back to the host tail with
// no state hazard.
func AttnBF16TailDevice(h, attnOut []float32, ow *model.BF16Weight, postNorm []float32, gate, up, down *model.BF16Weight, L, D, mixCols, FF int, eps float32) ([]float32, error) {
	if !bf16GeometryOK(ow, D, mixCols) ||
		!bf16GeometryOK(gate, FF, D) || !bf16GeometryOK(up, FF, D) || !bf16GeometryOK(down, D, FF) {
		return nil, core.NewError("native.AttnBF16TailDevice: bf16 geometry mismatch")
	}
	return attnTailRunCore(h, attnOut,
		func(enc metal.MTLComputeCommandEncoder, xBF, dstBF, dst metal.MTLBuffer, L, outDim, inDim int) error {
			return encProjBF16From(enc, ow, xBF, dstBF, dst, L, outDim, inDim)
		},
		postNorm,
		func(enc metal.MTLComputeCommandEncoder, tb quantTailBufs, normW metal.MTLBuffer, L, D, FF int, eps float32) error {
			return encResidualNormMLPBF16Tail(enc, tb, normW, gate, up, down, L, D, FF, eps)
		}, L, D, mixCols, FF, eps)
}

// AttnQuantTailDevice is the tail's PACKED twin: o_proj + the packed FFN tail over checkpoint
// codes, one command buffer.
func AttnQuantTailDevice(h, attnOut []float32, ow *model.QuantWeight, postNorm []float32, gate, up, down *model.QuantWeight, L, D, mixCols, FF int, eps float32) ([]float32, error) {
	if !quantGeometryOK(ow, D, mixCols) ||
		!quantGeometryOK(gate, FF, D) || !quantGeometryOK(up, FF, D) || !quantGeometryOK(down, D, FF) {
		return nil, core.NewError("native.AttnQuantTailDevice: quant geometry mismatch")
	}
	return attnTailRunCore(h, attnOut,
		func(enc metal.MTLComputeCommandEncoder, xBF, dstBF, dst metal.MTLBuffer, L, outDim, inDim int) error {
			return encProjQuantBF16In(enc, ow, xBF, dstBF, dst, L, outDim, inDim)
		},
		postNorm,
		func(enc metal.MTLComputeCommandEncoder, tb quantTailBufs, normW metal.MTLBuffer, L, D, FF int, eps float32) error {
			return encResidualNormMLPQuantTail(enc, tb, normW, gate, up, down, L, D, FF, eps)
		}, L, D, mixCols, FF, eps)
}

// attnTailRunCore is the weight-form-agnostic tail body: o_proj closure + tail-emitter closure.
func attnTailRunCore(h, attnOut []float32, projO tailProjFromBF16, postNorm []float32, encTail func(metal.MTLComputeCommandEncoder, quantTailBufs, metal.MTLBuffer, int, int, int, float32) error, L, D, mixCols, FF int, eps float32) ([]float32, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	if L <= 0 || len(h) != L*D || len(attnOut) != L*mixCols || len(postNorm) != D {
		return nil, core.NewError("native.attnTailRunCore: size mismatch")
	}
	y := make([]float32, L*D)
	key := attnBF16TailKey{L: L, D: D, mixCols: mixCols, FF: FF}
	var encErr error
	withAutoreleasePool(func() {
		sc, gerr := getAttnBF16TailScratch(key)
		if gerr != nil {
			encErr = gerr
			return
		}
		defer putAttnBF16TailScratch(key, sc)
		hBuf, cerr := sc.h.copyBuffer(float32Bytes(h))
		if cerr != nil {
			encErr = cerr
			return
		}
		attnBuf, cerr := sc.attn.copyBuffer(float32Bytes(attnOut))
		if cerr != nil {
			encErr = cerr
			return
		}
		postNormBuf := residentFloat32(postNorm)
		cb := commandBufferFast(queue)
		enc := computeCommandEncoderFast(cb)
		fail := func(err error) {
			encErr = err
			endEncodingFast(enc)
		}
		// o_proj: attnOut [L,mixCols] → mix [L,D].
		if err := encNarrowF32ToBF16(enc, attnBuf, sc.attnBF.buf, L*mixCols); err != nil {
			fail(err)
			return
		}
		memoryBarrier(enc, metal.MTLBarrierScopeBuffers)
		if err := projO(enc, sc.attnBF.buf, sc.mixBF.buf, sc.mix.buf, L, D, mixCols); err != nil {
			fail(err)
			return
		}
		memoryBarrier(enc, metal.MTLBarrierScopeBuffers)
		if err := encTail(enc, quantTailBufs{
			h: hBuf, mix: sc.mix.buf, normed: sc.normed.buf, nBF: sc.nBF.buf,
			g: sc.g.buf, gBF: sc.gBF.buf, u: sc.u.buf, uBF: sc.uBF.buf, s: sc.s.buf, out: sc.out.buf,
		}, postNormBuf, L, D, FF, eps); err != nil {
			fail(err)
			return
		}
		endEncodingFast(enc)
		commitCommandBufferFast(cb)
		waitUntilCompletedFast(cb)
		copy(y, unsafe.Slice((*float32)(unsafe.Pointer(&sc.out.bytes[0])), L*D))
	})
	if encErr != nil {
		return nil, encErr
	}
	return y, nil
}
