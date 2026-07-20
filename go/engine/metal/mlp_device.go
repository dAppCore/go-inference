// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"sync"
	"unsafe"

	core "dappco.re/go"
	"github.com/tmc/apple/metal"
)

// mlp_device.go — the fused SwiGLU MLP: gate and up GEMMs, the
// silu-and-multiply glue (sigmoid + two element-wise multiplies), and the down GEMM, all encoded
// into ONE command buffer with the [L,FF] intermediates staying device-resident. The per-projection
// hook pays a ~330µs command-buffer round-trip per matmul (three per MLP); this pays one.

// composedMLPScratch holds the pinned staging for one (L,D,FF) shape: the x upload, the g/u/s
// intermediates, the out readback, and the two steel params blocks (gate/up share a shape).
type composedMLPScratch struct {
	x, g, u, s, out   *pinnedNoCopyBytes
	paramsGU, paramsD *pinnedNoCopyBytes
	paramsFilled      bool
}

type composedMLPKey struct{ L, D, FF int }

var composedMLPPools sync.Map // composedMLPKey -> *sync.Pool

func getComposedMLPScratch(L, D, FF int) (*composedMLPScratch, error) {
	key := composedMLPKey{L, D, FF}
	poolAny, ok := composedMLPPools.Load(key)
	if !ok {
		poolAny, _ = composedMLPPools.LoadOrStore(key, &sync.Pool{})
	}
	pool := poolAny.(*sync.Pool)
	if v := pool.Get(); v != nil {
		return v.(*composedMLPScratch), nil
	}
	sc := &composedMLPScratch{}
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
	sc.g = alloc(L * FF * 4)
	sc.u = alloc(L * FF * 4)
	sc.s = alloc(L * FF * 4)
	sc.out = alloc(L * D * 4)
	sc.paramsGU = alloc(72)
	sc.paramsD = alloc(72)
	if err != nil {
		return nil, err
	}
	return sc, nil
}

func putComposedMLPScratch(L, D, FF int, sc *composedMLPScratch) {
	if v, ok := composedMLPPools.Load(composedMLPKey{L, D, FF}); ok {
		v.(*sync.Pool).Put(sc)
	}
}

// ComposedMLPDevice computes SwiGLU: (silu(x@gateᵀ) ⊙ x@upᵀ) @ downᵀ in one command buffer.
// gate/up are [FF,D], down is [D,FF], x is [L,D]; returns [L,D].
func ComposedMLPDevice(gate, up, down, x []float32, L, D, FF int) ([]float32, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	if len(x) != L*D || len(gate) != FF*D || len(up) != FF*D || len(down) != D*FF {
		return nil, core.NewError("native.ComposedMLPDevice: size mismatch")
	}
	t := steelNT
	psoGU, err := steelGemmPipeline(t.name, false, false, false, L%t.bm == 0, FF%t.bn == 0, D%t.bk == 0)
	if err != nil {
		return nil, err
	}
	psoD, err := steelGemmPipeline(t.name, false, false, false, L%t.bm == 0, D%t.bn == 0, FF%t.bk == 0)
	if err != nil {
		return nil, err
	}
	psoSig, err := pipelineFor("v_Sigmoidfloat32float32")
	if err != nil {
		return nil, err
	}
	psoMul, err := pipelineFor("vv_Multiplyfloat32")
	if err != nil {
		return nil, err
	}

	out := make([]float32, L*D)
	var encErr error
	withAutoreleasePool(func() {
		sc, err := getComposedMLPScratch(L, D, FF)
		if err != nil {
			encErr = err
			return
		}
		defer putComposedMLPScratch(L, D, FF, sc)
		xBuf, err := sc.x.copyBuffer(float32Bytes(x))
		if err != nil {
			encErr = err
			return
		}
		if !sc.paramsFilled {
			tnGU, tmGU := (FF+t.bn-1)/t.bn, (L+t.bm-1)/t.bm
			fillMatMulF32SteelParams(sc.paramsGU.bytes, L, D, FF, D, tnGU, tmGU, D/t.bk)
			tnD, tmD := (D+t.bn-1)/t.bn, (L+t.bm-1)/t.bm
			fillMatMulF32SteelParams(sc.paramsD.bytes, L, FF, D, FF, tnD, tmD, FF/t.bk)
			sc.paramsFilled = true
		}
		gateBuf := residentFloat32(gate)
		upBuf := residentFloat32(up)
		downBuf := residentFloat32(down)

		tnGU, tmGU := (FF+t.bn-1)/t.bn, (L+t.bm-1)/t.bm
		tnD, tmD := (D+t.bn-1)/t.bn, (L+t.bm-1)/t.bm
		n := L * FF

		cb := commandBufferFast(queue)
		enc := computeCommandEncoderFast(cb)
		// gate and up are independent — no barrier between them.
		emitSteelGemm(encSink{enc}, psoGU, xBuf, gateBuf, sc.g.buf, sc.paramsGU.buf, tnGU, tmGU, uint(t.wn), uint(t.wm))
		emitSteelGemm(encSink{enc}, psoGU, xBuf, upBuf, sc.u.buf, sc.paramsGU.buf, tnGU, tmGU, uint(t.wn), uint(t.wm))
		memoryBarrierObject(enc, metal.MTLBarrierScopeBuffers)
		emitUnary(encSink{enc}, psoSig, sc.g.buf, sc.s.buf, n) // s = sigmoid(g)
		memoryBarrierObject(enc, metal.MTLBarrierScopeBuffers)
		emitBinary(encSink{enc}, psoMul, sc.s.buf, 0, sc.g.buf, 0, sc.s.buf, 0, n) // s = silu(g)
		memoryBarrierObject(enc, metal.MTLBarrierScopeBuffers)
		emitBinary(encSink{enc}, psoMul, sc.s.buf, 0, sc.u.buf, 0, sc.s.buf, 0, n) // s = silu(g)·u
		memoryBarrierObject(enc, metal.MTLBarrierScopeBuffers)
		emitSteelGemm(encSink{enc}, psoD, sc.s.buf, downBuf, sc.out.buf, sc.paramsD.buf, tnD, tmD, uint(t.wn), uint(t.wm))
		endEncodingFast(enc)
		commitCommandBufferFast(cb)
		waitUntilCompletedFast(cb)
		copy(out, unsafe.Slice((*float32)(unsafe.Pointer(&sc.out.bytes[0])), L*D))
	})
	if encErr != nil {
		return nil, encErr
	}
	return out, nil
}
