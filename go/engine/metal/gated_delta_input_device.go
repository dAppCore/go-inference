// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"sync"
	"unsafe"

	core "dappco.re/go"
)

// gated_delta_input_device.go — the fused input projection for the Qwen 3.6 gated-delta block: the four
// x-reading GEMMs (in_proj_qkv, in_proj_z, in_proj_a, in_proj_b) encoded into ONE command buffer. They
// all read the same x [L,D], so the GEMMs are independent — no barrier between them — and pay ONE
// command-buffer round-trip where the per-projection ProjMatMulInto hook pays four. in_proj_a/b are
// sub-floor standalone (a few KMACs each), yet ride the fused CB for free. Mirrors mlp_device.go:
// pooled pinned scratch per shape key, resident f32 weights, one emitSteelGemm per matmul (steel nt,
// fused kernel — the same device-f32 tier the standalone projection GEMM already serves), one commit+wait.

// gatedDeltaInputScratch holds the pinned staging for one (L,D,convDim,vDim,VH) shape: the x upload, the
// four projection outputs, and the four steel params blocks (one per GEMM — each has its own N).
type gatedDeltaInputScratch struct {
	x, qkv, z, a, b  *pinnedNoCopyBytes
	pQKV, pZ, pA, pB *pinnedNoCopyBytes
	paramsFilled     bool
}

type gatedDeltaInputKey struct{ L, D, convDim, vDim, VH int }

var gatedDeltaInputPools sync.Map // gatedDeltaInputKey -> *sync.Pool

func getGatedDeltaInputScratch(L, D, convDim, vDim, VH int) (*gatedDeltaInputScratch, error) {
	key := gatedDeltaInputKey{L, D, convDim, vDim, VH}
	poolAny, ok := gatedDeltaInputPools.Load(key)
	if !ok {
		poolAny, _ = gatedDeltaInputPools.LoadOrStore(key, &sync.Pool{})
	}
	pool := poolAny.(*sync.Pool)
	if v := pool.Get(); v != nil {
		return v.(*gatedDeltaInputScratch), nil
	}
	sc := &gatedDeltaInputScratch{}
	var err error
	alloc := func(n int) *pinnedNoCopyBytes {
		if err != nil {
			return nil
		}
		var b *pinnedNoCopyBytes
		b, err = newPinnedNoCopyBytes(n)
		return b
	}
	sc.x = alloc(L * D * 4)
	sc.qkv = alloc(L * convDim * 4)
	sc.z = alloc(L * vDim * 4)
	sc.a = alloc(L * VH * 4)
	sc.b = alloc(L * VH * 4)
	sc.pQKV = alloc(72)
	sc.pZ = alloc(72)
	sc.pA = alloc(72)
	sc.pB = alloc(72)
	if err != nil {
		return nil, err
	}
	return sc, nil
}

func putGatedDeltaInputScratch(L, D, convDim, vDim, VH int, sc *gatedDeltaInputScratch) {
	if v, ok := gatedDeltaInputPools.Load(gatedDeltaInputKey{L, D, convDim, vDim, VH}); ok {
		v.(*sync.Pool).Put(sc)
	}
}

// GatedDeltaInputDevice computes the four gated-delta input projections in one command buffer:
// qkv = x@qkvWᵀ [L,convDim], z = x@zWᵀ [L,vDim], a = x@aWᵀ [L,VH], b = x@bWᵀ [L,VH]. x is [L,D]; qkvW is
// [convDim,D], zW [vDim,D], aW/bW [VH,D] (the steel nt kernel reads each weight transposed). The four
// GEMMs read the same x and write disjoint outputs, so no barrier separates them — one commit drains all.
func GatedDeltaInputDevice(x, qkvW, zW, aW, bW []float32, L, D, convDim, vDim, VH int) (qkv, z, a, b []float32, err error) {
	if err := ensureInit(); err != nil {
		return nil, nil, nil, nil, err
	}
	if len(x) != L*D || len(qkvW) != convDim*D || len(zW) != vDim*D || len(aW) != VH*D || len(bW) != VH*D {
		return nil, nil, nil, nil, core.NewError("native.GatedDeltaInputDevice: size mismatch")
	}
	t := steelNT
	psoQKV, err := steelGemmPipeline(t.name, false, false, false, L%t.bm == 0, convDim%t.bn == 0, D%t.bk == 0)
	if err != nil {
		return nil, nil, nil, nil, err
	}
	psoZ, err := steelGemmPipeline(t.name, false, false, false, L%t.bm == 0, vDim%t.bn == 0, D%t.bk == 0)
	if err != nil {
		return nil, nil, nil, nil, err
	}
	psoA, err := steelGemmPipeline(t.name, false, false, false, L%t.bm == 0, VH%t.bn == 0, D%t.bk == 0)
	if err != nil {
		return nil, nil, nil, nil, err
	}
	psoB := psoA // a and b share N=VH ⇒ identical alignment ⇒ same pipeline.

	qkv = make([]float32, L*convDim)
	z = make([]float32, L*vDim)
	a = make([]float32, L*VH)
	b = make([]float32, L*VH)
	tnQKV, tnZ, tnVH := (convDim+t.bn-1)/t.bn, (vDim+t.bn-1)/t.bn, (VH+t.bn-1)/t.bn
	tmL := (L + t.bm - 1) / t.bm
	var encErr error
	withAutoreleasePool(func() {
		sc, gerr := getGatedDeltaInputScratch(L, D, convDim, vDim, VH)
		if gerr != nil {
			encErr = gerr
			return
		}
		defer putGatedDeltaInputScratch(L, D, convDim, vDim, VH, sc)
		xBuf, cerr := sc.x.copyBuffer(float32Bytes(x))
		if cerr != nil {
			encErr = cerr
			return
		}
		if !sc.paramsFilled {
			fillMatMulF32SteelParams(sc.pQKV.bytes, L, D, convDim, D, tnQKV, tmL, D/t.bk)
			fillMatMulF32SteelParams(sc.pZ.bytes, L, D, vDim, D, tnZ, tmL, D/t.bk)
			fillMatMulF32SteelParams(sc.pA.bytes, L, D, VH, D, tnVH, tmL, D/t.bk)
			fillMatMulF32SteelParams(sc.pB.bytes, L, D, VH, D, tnVH, tmL, D/t.bk)
			sc.paramsFilled = true
		}
		wQKV := residentFloat32(qkvW)
		wZ := residentFloat32(zW)
		wA := residentFloat32(aW)
		wB := residentFloat32(bW)

		cb := commandBufferFast(queue)
		enc := computeCommandEncoderFast(cb)
		// The four projections read the same x and write disjoint outputs — independent, no barrier.
		emitSteelGemm(encSink{enc}, psoQKV, xBuf, wQKV, sc.qkv.buf, sc.pQKV.buf, tnQKV, tmL, uint(t.wn), uint(t.wm))
		emitSteelGemm(encSink{enc}, psoZ, xBuf, wZ, sc.z.buf, sc.pZ.buf, tnZ, tmL, uint(t.wn), uint(t.wm))
		emitSteelGemm(encSink{enc}, psoA, xBuf, wA, sc.a.buf, sc.pA.buf, tnVH, tmL, uint(t.wn), uint(t.wm))
		emitSteelGemm(encSink{enc}, psoB, xBuf, wB, sc.b.buf, sc.pB.buf, tnVH, tmL, uint(t.wn), uint(t.wm))
		endEncodingFast(enc)
		commitCommandBufferFast(cb)
		waitUntilCompletedFast(cb)
		copy(qkv, unsafe.Slice((*float32)(unsafe.Pointer(&sc.qkv.bytes[0])), L*convDim))
		copy(z, unsafe.Slice((*float32)(unsafe.Pointer(&sc.z.bytes[0])), L*vDim))
		copy(a, unsafe.Slice((*float32)(unsafe.Pointer(&sc.a.bytes[0])), L*VH))
		copy(b, unsafe.Slice((*float32)(unsafe.Pointer(&sc.b.bytes[0])), L*VH))
	})
	if encErr != nil {
		return nil, nil, nil, nil, encErr
	}
	return qkv, z, a, b, nil
}
