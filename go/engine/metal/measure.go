// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"unsafe"

	"github.com/tmc/apple/metal"
)

// attentionReEncode runs the bf16 attention block `reps` times the REGULAR way —
// persistent buffers, but the 6 ops re-encoded into a fresh command buffer every
// rep (the host re-encode the ICB path replaces). Buffers are created once so the
// measurement isolates per-rep host ENCODE cost, not buffer churn; the A/B
// against AttentionBlockICB(reps) is the encode-bypass number. Returns after the
// last rep completes.
func attentionReEncode(x, normWeight, wQ, wO, kCache, vCache []byte, dModel, nHeads, nKVHeads, headDim, kvLen int, base, scale float32, offset int, eps float32, reps int) error {
	if err := ensureInit(); err != nil {
		return err
	}
	qDim := nHeads * headDim
	var encErr error
	withAutoreleasePool(func() {
		xBuf, nwBuf := sharedBytes(x), sharedBytes(normWeight)
		wqBuf, woBuf := sharedBytes(wQ), sharedBytes(wO)
		kBuf, vBuf := sharedBytes(kCache), sharedBytes(vCache)
		off := int32(offset)
		offBuf := device.NewBufferWithBytesLengthOptions(unsafe.Pointer(&off), 4, metal.MTLResourceStorageModeShared)
		normed := scratchBF16(dModel)
		q, qr, attn := scratchBF16(qDim), scratchBF16(qDim), scratchBF16(qDim)
		attnOut, outBuf := scratchBF16(dModel), scratchBF16(dModel)
		_ = outBuf

		for range reps {
			cb := queue.CommandBuffer()
			enc := cb.ComputeCommandEncoder()
			if encErr = encRMSNormBF16(enc, xBuf, nwBuf, normed, 0, dModel, eps); encErr != nil {
				enc.EndEncoding()
				return
			}
			_ = encGemvBF16(enc, wqBuf, normed, q, qDim, dModel)
			_ = encRoPEBF16(enc, q, qr, offBuf, nHeads, headDim, headDim, base, scale)
			_ = encSDPA(enc, qr, kBuf, vBuf, attn, nHeads, nKVHeads, headDim, kvLen, scale)
			_ = encGemvBF16(enc, woBuf, attn, attnOut, dModel, qDim)
			_ = encAddBF16(enc, xBuf, attnOut, outBuf, dModel)
			enc.EndEncoding()
			cb.Commit()
			cb.WaitUntilCompleted()
		}
	})
	return encErr
}

// layerReEncode runs the full 21-op bf16 DecodeLayer `reps` times the REGULAR way
// — persistent buffers, but all 21 ops re-encoded into a fresh command buffer
// every rep (the host re-encode the ICB path replaces). It is the full-layer
// analogue of attentionReEncode: buffers are created once so the measurement
// isolates per-rep host ENCODE cost, not buffer churn, and the A/B against
// DecodeLayerICB(reps) is the per-layer encode-bypass number. The op sequence
// mirrors DecodeLayer exactly. Returns after the last rep completes.
func layerReEncode(
	x, attnNormW, wQ, wO, kCache, vCache, mlpNormW, wGate, wUp, wDown []byte,
	dModel, nHeads, nKVHeads, headDim, kvLen, dFF int,
	base, scale float32, offset int, eps float32,
	reps int,
) error {
	if err := ensureInit(); err != nil {
		return err
	}
	qDim := nHeads * headDim
	var encErr error
	withAutoreleasePool(func() {
		xBuf := sharedBytes(x)
		anwBuf, mnwBuf := sharedBytes(attnNormW), sharedBytes(mlpNormW)
		wqBuf, woBuf := sharedBytes(wQ), sharedBytes(wO)
		kBuf, vBuf := sharedBytes(kCache), sharedBytes(vCache)
		wgBuf, wuBuf, wdBuf := sharedBytes(wGate), sharedBytes(wUp), sharedBytes(wDown)
		off := int32(offset)
		offBuf := device.NewBufferWithBytesLengthOptions(unsafe.Pointer(&off), 4, metal.MTLResourceStorageModeShared)
		c044 := sharedBytes(bf16ConstBytes(dFF, 0.044715))
		c079 := sharedBytes(bf16ConstBytes(dFF, 0.7978845608028654))
		c1c := sharedBytes(bf16ConstBytes(dFF, 1.0))
		c05 := sharedBytes(bf16ConstBytes(dFF, 0.5))

		attnNormed := scratchBF16(dModel)
		q, qr, attn := scratchBF16(qDim), scratchBF16(qDim), scratchBF16(qDim)
		attnOut, h := scratchBF16(dModel), scratchBF16(dModel)
		mlpNormed := scratchBF16(dModel)
		gate, up := scratchBF16(dFF), scratchBF16(dFF)
		x2, x3, x3s, inner := scratchBF16(dFF), scratchBF16(dFF), scratchBF16(dFF), scratchBF16(dFF)
		scaled, tnh, onePlus, halfG := scratchBF16(dFF), scratchBF16(dFF), scratchBF16(dFF), scratchBF16(dFF)
		gelu, gated := scratchBF16(dFF), scratchBF16(dFF)
		down, outBuf := scratchBF16(dModel), scratchBF16(dModel)
		_ = outBuf

		for range reps {
			cb := queue.CommandBuffer()
			enc := cb.ComputeCommandEncoder()
			if encErr = encRMSNormBF16(enc, xBuf, anwBuf, attnNormed, 0, dModel, eps); encErr != nil {
				enc.EndEncoding()
				return
			}
			// attention half
			_ = encGemvBF16(enc, wqBuf, attnNormed, q, qDim, dModel)
			_ = encRoPEBF16(enc, q, qr, offBuf, nHeads, headDim, headDim, base, scale)
			_ = encSDPA(enc, qr, kBuf, vBuf, attn, nHeads, nKVHeads, headDim, kvLen, scale)
			_ = encGemvBF16(enc, woBuf, attn, attnOut, dModel, qDim)
			_ = encAddBF16(enc, xBuf, attnOut, h, dModel)
			// MLP half
			_ = encRMSNormBF16(enc, h, mnwBuf, mlpNormed, 0, dModel, eps)
			_ = encGemvBF16(enc, wgBuf, mlpNormed, gate, dFF, dModel)
			_ = encGemvBF16(enc, wuBuf, mlpNormed, up, dFF, dModel)
			_ = encMulBF16(enc, gate, gate, x2, dFF)
			_ = encMulBF16(enc, x2, gate, x3, dFF)
			_ = encMulBF16(enc, x3, c044, x3s, dFF)
			_ = encAddBF16(enc, gate, x3s, inner, dFF)
			_ = encMulBF16(enc, inner, c079, scaled, dFF)
			_ = encTanhBF16(enc, scaled, tnh, dFF)
			_ = encAddBF16(enc, tnh, c1c, onePlus, dFF)
			_ = encMulBF16(enc, gate, c05, halfG, dFF)
			_ = encMulBF16(enc, halfG, onePlus, gelu, dFF)
			_ = encMulBF16(enc, gelu, up, gated, dFF)
			_ = encGemvBF16(enc, wdBuf, gated, down, dModel, dFF)
			_ = encAddBF16(enc, h, down, outBuf, dModel)
			enc.EndEncoding()
			cb.Commit()
			cb.WaitUntilCompleted()
		}
	})
	return encErr
}

// tokenReEncode runs a full nLayers-deep decode TOKEN the regular way `reps`
// times — persistent buffers, but all nLayers*21 ops re-encoded into ONE command
// buffer per token with a SINGLE commit+wait per token (exactly as a real decode
// step submits its whole layer stack at once). This is the per-token analogue of
// layerReEncode, and the point of it: layerReEncode pays a commit+wait PER LAYER,
// so GPU + submission time (identical on both A/B sides) dominates each rep and
// dilutes the encode-bypass ratio toward 1; here that fixed cost is paid once per
// token and amortised across the whole stack, so the A/B against DecodeTokenICB
// reps is the UN-DILUTED per-token encode-bypass headline.
//
// Layers share one set of weights, scratch and KV (a host-cost timing harness —
// the cost to encode a command is independent of which buffer it binds, and the
// shared buffers keep it AX-11-light), ping-ponging the residual stream between
// two model-dim buffers (out of layer L is in of layer L+1). The op sequence per
// layer mirrors DecodeLayer exactly, so tokenReEncode(nLayers=1) equals DecodeLayer
// and the chained form equals nLayers applications of it (gated in the tests).
// Returns the final token output for that parity gate; with reps>1 the buffers
// chain (irrelevant to the per-token encode cost being measured).
func tokenReEncode(
	x, attnNormW, wQ, wO, kCache, vCache, mlpNormW, wGate, wUp, wDown []byte,
	dModel, nHeads, nKVHeads, headDim, kvLen, dFF, nLayers int,
	base, scale float32, offset int, eps float32,
	reps int,
) ([]byte, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	if nLayers < 1 {
		nLayers = 1
	}
	if reps < 1 {
		reps = 1
	}
	qDim := nHeads * headDim
	out := make([]byte, dModel*bf16Size)
	var encErr error
	withAutoreleasePool(func() {
		anwBuf, mnwBuf := sharedBytes(attnNormW), sharedBytes(mlpNormW)
		wqBuf, woBuf := sharedBytes(wQ), sharedBytes(wO)
		kBuf, vBuf := sharedBytes(kCache), sharedBytes(vCache)
		wgBuf, wuBuf, wdBuf := sharedBytes(wGate), sharedBytes(wUp), sharedBytes(wDown)
		off := int32(offset)
		offBuf := device.NewBufferWithBytesLengthOptions(unsafe.Pointer(&off), 4, metal.MTLResourceStorageModeShared)
		c044 := sharedBytes(bf16ConstBytes(dFF, 0.044715))
		c079 := sharedBytes(bf16ConstBytes(dFF, 0.7978845608028654))
		c1c := sharedBytes(bf16ConstBytes(dFF, 1.0))
		c05 := sharedBytes(bf16ConstBytes(dFF, 0.5))

		// residual-stream ping-pong: xA seeded with the token input, xB scratch.
		xA, xB := sharedBytes(x), scratchBF16(dModel)

		// shared per-layer scratch (reused every layer; serial dispatch orders it)
		attnNormed := scratchBF16(dModel)
		q, qr, attn := scratchBF16(qDim), scratchBF16(qDim), scratchBF16(qDim)
		attnOut, h := scratchBF16(dModel), scratchBF16(dModel)
		mlpNormed := scratchBF16(dModel)
		gate, up := scratchBF16(dFF), scratchBF16(dFF)
		x2, x3, x3s, inner := scratchBF16(dFF), scratchBF16(dFF), scratchBF16(dFF), scratchBF16(dFF)
		scaled, tnh, onePlus, halfG := scratchBF16(dFF), scratchBF16(dFF), scratchBF16(dFF), scratchBF16(dFF)
		gelu, gated := scratchBF16(dFF), scratchBF16(dFF)
		down := scratchBF16(dModel)

		// encodeLayer emits the 21-op layer reading inBuf, writing outBuf — the
		// exact DecodeLayer sequence (in is read at the rms and the attn residual).
		encodeLayer := func(enc metal.MTLComputeCommandEncoder, inBuf, outBuf metal.MTLBuffer) error {
			if err := encRMSNormBF16(enc, inBuf, anwBuf, attnNormed, 0, dModel, eps); err != nil {
				return err
			}
			_ = encGemvBF16(enc, wqBuf, attnNormed, q, qDim, dModel)
			_ = encRoPEBF16(enc, q, qr, offBuf, nHeads, headDim, headDim, base, scale)
			_ = encSDPA(enc, qr, kBuf, vBuf, attn, nHeads, nKVHeads, headDim, kvLen, scale)
			_ = encGemvBF16(enc, woBuf, attn, attnOut, dModel, qDim)
			_ = encAddBF16(enc, inBuf, attnOut, h, dModel)
			_ = encRMSNormBF16(enc, h, mnwBuf, mlpNormed, 0, dModel, eps)
			_ = encGemvBF16(enc, wgBuf, mlpNormed, gate, dFF, dModel)
			_ = encGemvBF16(enc, wuBuf, mlpNormed, up, dFF, dModel)
			if gpuHasGeluKernel() {
				_ = encGeluGateMulFused(enc, gate, up, gated, dFF)
			} else {
				_ = encMulBF16(enc, gate, gate, x2, dFF)
				_ = encMulBF16(enc, x2, gate, x3, dFF)
				_ = encMulBF16(enc, x3, c044, x3s, dFF)
				_ = encAddBF16(enc, gate, x3s, inner, dFF)
				_ = encMulBF16(enc, inner, c079, scaled, dFF)
				_ = encTanhBF16(enc, scaled, tnh, dFF)
				_ = encAddBF16(enc, tnh, c1c, onePlus, dFF)
				_ = encMulBF16(enc, gate, c05, halfG, dFF)
				_ = encMulBF16(enc, halfG, onePlus, gelu, dFF)
				_ = encMulBF16(enc, gelu, up, gated, dFF)
			}
			_ = encGemvBF16(enc, wdBuf, gated, down, dModel, dFF)
			_ = encAddBF16(enc, h, down, outBuf, dModel)
			return nil
		}

		var lastOut metal.MTLBuffer
		for r := 0; r < reps; r++ {
			cb := queue.CommandBuffer()
			enc := cb.ComputeCommandEncoder()
			in, outB := xA, xB
			for L := 0; L < nLayers; L++ {
				if encErr = encodeLayer(enc, in, outB); encErr != nil {
					enc.EndEncoding()
					return
				}
				in, outB = outB, in
			}
			lastOut = in // after the final swap, `in` is the last layer's output
			enc.EndEncoding()
			cb.Commit()
			cb.WaitUntilCompleted()
		}
		copy(out, unsafe.Slice((*byte)(lastOut.Contents()), len(out)))
	})
	return out, encErr
}
