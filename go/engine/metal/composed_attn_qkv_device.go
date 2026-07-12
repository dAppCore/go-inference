// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"sync"
	"unsafe"

	core "dappco.re/go"
)

// composed_attn_qkv_device.go — the fused q/k/v projection for the composed stack's full-attention mixer:
// q_proj, k_proj and v_proj encoded into ONE command buffer. All three read the same hidden h [L,D], so
// the GEMMs are independent — no barrier between them — and share one command-buffer round-trip. Only
// q crosses the device floor standalone (k/v are sub-floor, otherwise a serial host matmul); inside the
// fused CB k/v ride along as free riders. Mirrors composed_mlp_device.go / gated_delta_input_device.go:
// pooled pinned scratch per shape key, resident f32 weights, one emitSteelGemm per matmul (steel nt,
// fused kernel — device-f32 tier), one commit+wait. k/v share N=kvCols ⇒ one pipeline + one params block.

// composedAttnQKVScratch holds the pinned staging for one (L,D,qCols,kvCols) shape: the h upload, the
// q/k/v projection outputs, and two steel params blocks (q has its own N; k and v share kvCols).
type composedAttnQKVScratch struct {
	h, q, k, v   *pinnedNoCopyBytes
	pQ, pKV      *pinnedNoCopyBytes
	paramsFilled bool
}

type composedAttnQKVKey struct{ L, D, qCols, kvCols int }

var composedAttnQKVPools sync.Map // composedAttnQKVKey -> *sync.Pool

func getComposedAttnQKVScratch(L, D, qCols, kvCols int) (*composedAttnQKVScratch, error) {
	key := composedAttnQKVKey{L, D, qCols, kvCols}
	poolAny, ok := composedAttnQKVPools.Load(key)
	if !ok {
		poolAny, _ = composedAttnQKVPools.LoadOrStore(key, &sync.Pool{})
	}
	pool := poolAny.(*sync.Pool)
	if v := pool.Get(); v != nil {
		return v.(*composedAttnQKVScratch), nil
	}
	sc := &composedAttnQKVScratch{}
	var err error
	alloc := func(n int) *pinnedNoCopyBytes {
		if err != nil {
			return nil
		}
		var b *pinnedNoCopyBytes
		b, err = newPinnedNoCopyBytes(n)
		return b
	}
	sc.h = alloc(L * D * 4)
	sc.q = alloc(L * qCols * 4)
	sc.k = alloc(L * kvCols * 4)
	sc.v = alloc(L * kvCols * 4)
	sc.pQ = alloc(72)
	sc.pKV = alloc(72)
	if err != nil {
		return nil, err
	}
	return sc, nil
}

func putComposedAttnQKVScratch(L, D, qCols, kvCols int, sc *composedAttnQKVScratch) {
	if v, ok := composedAttnQKVPools.Load(composedAttnQKVKey{L, D, qCols, kvCols}); ok {
		v.(*sync.Pool).Put(sc)
	}
}

// ComposedAttnQKVDevice computes the attention mixer's three projections in one command buffer:
// q = h@qWᵀ [L,qCols], k = h@kWᵀ [L,kvCols], v = h@vWᵀ [L,kvCols]. h is [L,D]; qW is [qCols,D], kW/vW
// [kvCols,D] (the steel nt kernel reads each weight transposed). The three GEMMs read the same h and
// write disjoint outputs, so no barrier separates them — one commit drains all three.
func ComposedAttnQKVDevice(h, qW, kW, vW []float32, L, D, qCols, kvCols int) (q, k, v []float32, err error) {
	if err := ensureInit(); err != nil {
		return nil, nil, nil, err
	}
	if len(h) != L*D || len(qW) != qCols*D || len(kW) != kvCols*D || len(vW) != kvCols*D {
		return nil, nil, nil, core.NewError("native.ComposedAttnQKVDevice: size mismatch")
	}
	t := steelNT
	psoQ, err := steelGemmPipeline(t.name, false, false, false, L%t.bm == 0, qCols%t.bn == 0, D%t.bk == 0)
	if err != nil {
		return nil, nil, nil, err
	}
	psoKV, err := steelGemmPipeline(t.name, false, false, false, L%t.bm == 0, kvCols%t.bn == 0, D%t.bk == 0)
	if err != nil {
		return nil, nil, nil, err
	}

	q = make([]float32, L*qCols)
	k = make([]float32, L*kvCols)
	v = make([]float32, L*kvCols)
	tnQ, tnKV := (qCols+t.bn-1)/t.bn, (kvCols+t.bn-1)/t.bn
	tmL := (L + t.bm - 1) / t.bm
	var encErr error
	withAutoreleasePool(func() {
		sc, gerr := getComposedAttnQKVScratch(L, D, qCols, kvCols)
		if gerr != nil {
			encErr = gerr
			return
		}
		defer putComposedAttnQKVScratch(L, D, qCols, kvCols, sc)
		hBuf, cerr := sc.h.copyBuffer(float32Bytes(h))
		if cerr != nil {
			encErr = cerr
			return
		}
		if !sc.paramsFilled {
			fillMatMulF32SteelParams(sc.pQ.bytes, L, D, qCols, D, tnQ, tmL, D/t.bk)
			fillMatMulF32SteelParams(sc.pKV.bytes, L, D, kvCols, D, tnKV, tmL, D/t.bk)
			sc.paramsFilled = true
		}
		wQ := residentFloat32(qW)
		wK := residentFloat32(kW)
		wV := residentFloat32(vW)

		cb := commandBufferFast(queue)
		enc := computeCommandEncoderFast(cb)
		// q/k/v read the same h and write disjoint outputs — independent, no barrier. k and v share
		// N=kvCols, so they ride the same pipeline and params block (params is read-only per dispatch).
		emitSteelGemm(encSink{enc}, psoQ, hBuf, wQ, sc.q.buf, sc.pQ.buf, tnQ, tmL, uint(t.wn), uint(t.wm))
		emitSteelGemm(encSink{enc}, psoKV, hBuf, wK, sc.k.buf, sc.pKV.buf, tnKV, tmL, uint(t.wn), uint(t.wm))
		emitSteelGemm(encSink{enc}, psoKV, hBuf, wV, sc.v.buf, sc.pKV.buf, tnKV, tmL, uint(t.wn), uint(t.wm))
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
