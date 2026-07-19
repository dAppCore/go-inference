// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"sync"
	"unsafe"

	core "dappco.re/go"
	"github.com/tmc/apple/metal"
)

// decode_step_batched.go — the MTP batched verify forward: K query tokens through one decode layer
// in ONE command buffer over the resident KV cache, with the layer weights uploaded ONCE. This is
// the speculative-decode speedup the sequential mtp.go verify (K separate steps = K command buffers
// + K weight uploads) leaves on the table: the target verifies the whole K-token draft block in a
// single submit. Each query row i decodes at position basePos+i, writes its K/V into the cache at
// row basePos+i, and attends [0..basePos+i] with the SAME single-query kernels the per-token step
// uses — so the layer output is BYTE-IDENTICAL to K sequential DecodeStepKV calls (proven in
// decode_step_batched_test.go), only without the per-token command-buffer + weight-upload overhead.
//
// Why byte-identical (not merely close): the heavy projections still run as per-row gemv (the exact
// kernel a single-token step uses), and the attention runs per-row single-query encSDPAStrided over
// the cache window [0..basePos+i] — identical dispatches to the sequential path, just encoded into
// one command buffer with the weights resident once. The cache write→read ordering across rows is
// Metal's automatic buffer hazard tracking (row i+1's attention reads the cache after row i's K/V
// write), so the K-position causal structure is exact. The remaining speedup — folding the K per-row
// projections into one steel GEMM (weight reuse across rows) — trades this byte-identity for
// token-identity (a GEMM reduces over the contraction in a different order than K gemvs); that is the
// metal-MTP-parity follow-up. This v1 keeps byte-identity and still wins the submit + upload overhead.

// decodeLayerBatchedScratchPool keeps the reusable pinned row staging and GPU intermediates warm for
// the public batched helper. A command buffer is waited before the scratch is returned.
var decodeLayerBatchedScratchPools sync.Map

type decodeLayerBatchedScratchKey struct {
	dModel, qDim, kvDim, nHeads, dFF, K int
}

type decodeLayerBatchedScratchPool struct {
	core.Pool[*decodeLayerBatchedScratch]
}

type decodeLayerBatchedScratch struct {
	xs, out                     *pinnedNoCopyBytes
	xsView                      cachedNoCopyBytesView
	outView                     cachedNoCopyBytesView
	asc                         attnScratch
	msc                         mlpScratch
	hBuf                        metal.MTLBuffer
	offBuf                      []metal.MTLBuffer
	dModel, qDim, kvDim, nHeads int
	dFF, K                      int
}

func newDecodeLayerBatchedScratch(dModel, qDim, kvDim, nHeads, dFF, K int) (*decodeLayerBatchedScratch, error) {
	rowBytes := dModel * bf16Size
	xs, err := newPinnedNoCopyBytes(K * rowBytes)
	if err != nil {
		return nil, err
	}
	out, err := newPinnedNoCopyBytes(K * rowBytes)
	if err != nil {
		xs.Close()
		return nil, err
	}
	return &decodeLayerBatchedScratch{
		xs: xs, out: out,
		asc:  newAttnScratch(dModel, qDim, kvDim, nHeads, 0),
		msc:  newMLPScratch(dModel, dFF),
		hBuf: scratchBF16(dModel), offBuf: make([]metal.MTLBuffer, K),
		dModel: dModel, qDim: qDim, kvDim: kvDim, nHeads: nHeads,
		dFF: dFF, K: K,
	}, nil
}

func (s *decodeLayerBatchedScratch) matches(dModel, qDim, kvDim, nHeads, dFF, K int) bool {
	return s != nil && s.xs != nil && s.out != nil && s.xs.buf != nil && s.out.buf != nil &&
		s.dModel == dModel && s.qDim == qDim && s.kvDim == kvDim && s.nHeads == nHeads && s.dFF == dFF && s.K == K
}

func (s *decodeLayerBatchedScratch) Close() {
	if s == nil {
		return
	}
	if s.xs != nil {
		s.xs.Close()
		s.xs = nil
	}
	if s.out != nil {
		s.out.Close()
		s.out = nil
	}
	s.xsView.Close()
	s.outView.Close()
	s.asc = attnScratch{}
	s.msc = mlpScratch{}
	s.hBuf = nil
	s.offBuf = nil
}

func (s *decodeLayerBatchedScratch) outputView(out []byte) (metal.MTLBuffer, bool) {
	if s == nil || len(out) == 0 {
		return nil, false
	}
	return s.outView.buffer(out)
}

func decodeLayerBatchedScratchPoolFor(dModel, qDim, kvDim, nHeads, dFF, K int) *decodeLayerBatchedScratchPool {
	key := decodeLayerBatchedScratchKey{dModel: dModel, qDim: qDim, kvDim: kvDim, nHeads: nHeads, dFF: dFF, K: K}
	if v, ok := decodeLayerBatchedScratchPools.Load(key); ok {
		return v.(*decodeLayerBatchedScratchPool)
	}
	pool := &decodeLayerBatchedScratchPool{}
	actual, _ := decodeLayerBatchedScratchPools.LoadOrStore(key, pool)
	return actual.(*decodeLayerBatchedScratchPool)
}

func getDecodeLayerBatchedScratch(dModel, qDim, kvDim, nHeads, dFF, K int) (*decodeLayerBatchedScratch, error) {
	if s := decodeLayerBatchedScratchPoolFor(dModel, qDim, kvDim, nHeads, dFF, K).Get(); s != nil {
		if s.matches(dModel, qDim, kvDim, nHeads, dFF, K) {
			return s, nil
		}
		s.Close()
	}
	return newDecodeLayerBatchedScratch(dModel, qDim, kvDim, nHeads, dFF, K)
}

func putDecodeLayerBatchedScratch(s *decodeLayerBatchedScratch) {
	if s != nil && s.dModel > 0 && s.qDim > 0 && s.kvDim > 0 && s.nHeads > 0 && s.dFF > 0 && s.K > 0 {
		decodeLayerBatchedScratchPoolFor(s.dModel, s.qDim, s.kvDim, s.nHeads, s.dFF, s.K).Put(s)
	}
}

// DecodeLayerBatchedKV runs one full decode layer (attention half + gemma MLP half, both residuals)
// for K query tokens at positions [basePos, basePos+K) in one command buffer, growing the seq-major
// KV caches at rows basePos..basePos+K-1. xs is the K input hiddens [K, dModel] bf16; the result is
// the K output hiddens [K, dModel] bf16. kCache/vCache are updated in place. Byte-identical to
// stepping the same K rows one at a time with DecodeStepKV (same kernels, same cache evolution).
func DecodeLayerBatchedKV(
	xs, attnNormW, wQ, wK, wV, wO, kCache, vCache, mlpNormW, wGate, wUp, wDown []byte,
	dModel, nHeads, nKVHeads, headDim, maxLen, dFF, basePos, K int,
	base, scale, eps float32,
) ([]byte, error) {
	return DecodeLayerBatchedKVInto(nil, xs, attnNormW, wQ, wK, wV, wO, kCache, vCache, mlpNormW, wGate, wUp, wDown, dModel, nHeads, nKVHeads, headDim, maxLen, dFF, basePos, K, base, scale, eps)
}

func DecodeLayerBatchedKVInto(
	out []byte,
	xs, attnNormW, wQ, wK, wV, wO, kCache, vCache, mlpNormW, wGate, wUp, wDown []byte,
	dModel, nHeads, nKVHeads, headDim, maxLen, dFF, basePos, K int,
	base, scale, eps float32,
) ([]byte, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	if K <= 0 {
		return nil, core.NewError("native.DecodeLayerBatchedKV: K must be > 0")
	}
	if basePos < 0 || basePos+K > maxLen {
		return nil, core.NewError("native.DecodeLayerBatchedKV: [basePos, basePos+K) out of [0,maxLen)")
	}
	rowBytes := dModel * bf16Size
	if len(xs) != K*rowBytes {
		return nil, core.NewError("native.DecodeLayerBatchedKV: xs must be K*dModel bf16 bytes")
	}
	// the per-row shape contract for the weights + caches (validated at the first row's position).
	if err := validateStepKV(xs[:rowBytes], attnNormW, wQ, wK, wV, wO, kCache, vCache, dModel, nHeads, nKVHeads, headDim, maxLen, basePos); err != nil {
		return nil, err
	}
	if len(mlpNormW) != dModel*bf16Size {
		return nil, core.NewError("native.DecodeLayerBatchedKV: mlpNormW must be dModel bf16 bytes")
	}
	if len(wGate) != dFF*dModel*bf16Size || len(wUp) != dFF*dModel*bf16Size || len(wDown) != dModel*dFF*bf16Size {
		return nil, core.NewError("native.DecodeLayerBatchedKV: MLP weight size mismatch")
	}
	qDim, kvDim := nHeads*headDim, nKVHeads*headDim
	outLen := K * rowBytes
	callerOut := cap(out) >= outLen
	if callerOut {
		out = out[:outLen]
	} else {
		out = make([]byte, outLen)
	}
	var encErr error
	withAutoreleasePool(func() {
		// the layer weights are uploaded ONCE and reused across all K rows — the win over K separate
		// DecodeStepKV calls, each of which re-uploads every weight.
		proj := bf16Projector{
			wQ: bufView{buf: residentBytes(wQ)}, wK: bufView{buf: residentBytes(wK)}, wV: bufView{buf: residentBytes(wV)}, wO: bufView{buf: residentBytes(wO)},
			wGate: bufView{buf: residentBytes(wGate)}, wUp: bufView{buf: residentBytes(wUp)}, wDown: bufView{buf: residentBytes(wDown)},
			dModel: dModel, qDim: qDim, kvDim: kvDim, dFF: dFF,
		}
		nwBuf, mnwBuf := residentBytes(attnNormW), residentBytes(mlpNormW)
		kvScratch, err := getAttentionBlockKVScratch(len(kCache), len(vCache))
		if err != nil {
			encErr = err
			return
		}
		defer putAttentionBlockKVScratch(kvScratch)
		kBuf, vBuf, err := kvScratch.buffers(kCache, vCache)
		if err != nil {
			encErr = err
			return
		}
		sc, err := getDecodeLayerBatchedScratch(dModel, qDim, kvDim, nHeads, dFF, K)
		if err != nil {
			encErr = err
			return
		}
		defer putDecodeLayerBatchedScratch(sc)
		xsBuf, ok := sc.xsView.buffer(xs)
		if !ok {
			xsBuf, err = sc.xs.copyBuffer(xs)
			if err != nil {
				encErr = err
				return
			}
		}
		outBuf := sc.out.buf
		directOut := false
		if callerOut {
			if tmp, ok := sc.outputView(out); ok {
				outBuf = tmp
				directOut = true
			}
		}
		for i := range K {
			sc.offBuf[i] = scalarI32(int32(basePos + i))
		}

		cb := commandBufferFast(queue)
		enc := computeCommandEncoderFast(cb)
		for i := range K {
			xOff := uint(i * rowBytes)
			// attention half: project q/k/v from row i, write k/v into the cache at row basePos+i,
			// attend [0..basePos+i] (single-query, the exact per-token kernel) → h.
			if err := encAttnHalfKVInputAt(enc, xsBuf, xOff, kBuf, vBuf, sc.offBuf[i], sc.hBuf, 0, 0,
				bufView{buf: nwBuf}, bufView{}, bufView{}, bufView{}, nil, sc.asc, proj,
				dModel, nHeads, nKVHeads, headDim, basePos+i, 0, headDim, base, scale, scale, eps, nil, bufView{}); err != nil {
				endEncodingFast(enc)
				encErr = err
				return
			}
			// MLP half on h → row i's output inside the reusable pinned output backing.
			if err := encMLPHalfBF16At(enc, sc.hBuf, outBuf, uint(i*rowBytes), bufView{buf: mnwBuf}, bufView{}, sc.msc, proj, dModel, dFF, eps); err != nil {
				endEncodingFast(enc)
				encErr = err
				return
			}
		}
		endEncodingFast(enc)
		commitCommandBufferFast(cb)
		waitUntilCompletedFast(cb)
		if !directOut {
			copy(out, sc.out.bytes[:outLen])
		}
		if encErr != nil {
			return
		}
		copy(kCache, unsafe.Slice((*byte)(kBuf.Contents()), len(kCache)))
		copy(vCache, unsafe.Slice((*byte)(vBuf.Contents()), len(vCache)))
	})
	return out, encErr
}
