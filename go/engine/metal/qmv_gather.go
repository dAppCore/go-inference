// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"runtime"
	"sync"
	"unsafe"

	core "dappco.re/go"
	"github.com/tmc/apple/metal"
)

type gatherQMVBF16MetaKey struct {
	numExperts, outDim, inDim, groupSize, bits, expertRows int
}

type gatherQMVBF16Meta struct {
	xShape, xStrides, wShape, wStrides, sbStrides, batchShape, zeroStride metal.MTLBuffer
}

var gatherQMVBF16Metas sync.Map
var gatherQMVBF16KernelNames sync.Map

func gatherQMVBF16SteelKernelName(outDim, inDim, groupSize, bits int) string {
	fast := outDim%8 == 0 && inDim%512 == 0
	key := qmvBF16KernelKey{groupSize: groupSize, bits: bits, fast: fast}
	if v, ok := gatherQMVBF16KernelNames.Load(key); ok {
		return v.(string)
	}
	variant := "_gather_qmv_"
	if fast {
		variant = "_gather_qmv_fast_"
	}
	name := core.Sprintf("affine%sbfloat16_t_gs_%d_b_%d", variant, groupSize, bits)
	if v, loaded := gatherQMVBF16KernelNames.LoadOrStore(key, name); loaded {
		return v.(string)
	}
	return name
}

func gatherQMVBF16SteelPipeline(outDim, inDim, groupSize, bits int) (metal.MTLComputePipelineState, error) {
	return pipelineFor(gatherQMVBF16SteelKernelName(outDim, inDim, groupSize, bits))
}

func gatherQMVBF16Metadata(numExperts, outDim, inDim, groupSize, bits, expertRows int) (*gatherQMVBF16Meta, error) {
	rowPackedBytes := inDim * bits / 8
	if rowPackedBytes%4 != 0 {
		return nil, core.NewError("native.gatherQMVBF16Metadata: packed row must be uint32 aligned")
	}
	rowPackedU32 := inDim * bits / 32
	groups := inDim / groupSize
	key := gatherQMVBF16MetaKey{numExperts: numExperts, outDim: outDim, inDim: inDim, groupSize: groupSize, bits: bits, expertRows: expertRows}
	if v, ok := gatherQMVBF16Metas.Load(key); ok {
		return v.(*gatherQMVBF16Meta), nil
	}

	xShape := [...]int32{1, int32(inDim)}
	xStrides := [...]int64{int64(inDim), 1}
	wShape := [...]int32{int32(numExperts), int32(expertRows), int32(rowPackedU32)}
	wStrides := [...]int64{int64(expertRows * rowPackedU32), int64(rowPackedU32), 1}
	sbStrides := [...]int64{int64(expertRows * groups), int64(groups), 1}
	batchShape := [...]int32{1}
	zeroStride := [...]int64{0}
	meta := &gatherQMVBF16Meta{
		xShape:     device.NewBufferWithBytesLengthOptions(unsafe.Pointer(&xShape[0]), uint(len(xShape)*4), metal.MTLResourceStorageModeShared),
		xStrides:   device.NewBufferWithBytesLengthOptions(unsafe.Pointer(&xStrides[0]), uint(len(xStrides)*8), metal.MTLResourceStorageModeShared),
		wShape:     device.NewBufferWithBytesLengthOptions(unsafe.Pointer(&wShape[0]), uint(len(wShape)*4), metal.MTLResourceStorageModeShared),
		wStrides:   device.NewBufferWithBytesLengthOptions(unsafe.Pointer(&wStrides[0]), uint(len(wStrides)*8), metal.MTLResourceStorageModeShared),
		sbStrides:  device.NewBufferWithBytesLengthOptions(unsafe.Pointer(&sbStrides[0]), uint(len(sbStrides)*8), metal.MTLResourceStorageModeShared),
		batchShape: device.NewBufferWithBytesLengthOptions(unsafe.Pointer(&batchShape[0]), uint(len(batchShape)*4), metal.MTLResourceStorageModeShared),
		zeroStride: device.NewBufferWithBytesLengthOptions(unsafe.Pointer(&zeroStride[0]), uint(len(zeroStride)*8), metal.MTLResourceStorageModeShared),
	}
	if v, loaded := gatherQMVBF16Metas.LoadOrStore(key, meta); loaded {
		return v.(*gatherQMVBF16Meta), nil
	}
	return meta, nil
}

func emitGatherQMVBF16Steel(sink encSink, pso metal.MTLComputePipelineState, meta *gatherQMVBF16Meta, x, wq metal.MTLBuffer, wqOff uint, scales metal.MTLBuffer, scalesOff uint, biases metal.MTLBuffer, biasesOff uint, indices metal.MTLBuffer, indicesOff uint, out metal.MTLBuffer, outOff uint, outDim, inDim, groupSize, bits, rowBase int) {
	rowPackedBytes := inDim * bits / 8
	groups := inDim / groupSize

	sink.setPSO(pso)
	sink.setBuf(wq, wqOff+uint(rowBase*rowPackedBytes), 0)
	sink.setBuf(scales, scalesOff+uint(rowBase*groups*bf16Size), 1)
	sink.setBuf(biases, biasesOff+uint(rowBase*groups*bf16Size), 2)
	sink.setBuf(x, 0, 3)
	sink.setBuf(scalarI32(0), 0, 4)
	sink.setBuf(indices, indicesOff, 5)
	sink.setBuf(out, outOff, 6)
	sink.setI32(int32(inDim), 7)
	sink.setI32(int32(outDim), 8)
	sink.setI32(0, 9)
	sink.setBuf(meta.xShape, 0, 10)
	sink.setBuf(meta.xStrides, 0, 11)
	sink.setI32(1, 12)
	sink.setBuf(meta.wShape, 0, 13)
	sink.setBuf(meta.wStrides, 0, 14)
	sink.setBuf(meta.sbStrides, 0, 15)
	sink.setBuf(meta.sbStrides, 0, 16)
	sink.setI32(1, 17)
	sink.setBuf(meta.batchShape, 0, 18)
	sink.setBuf(meta.zeroStride, 0, 19)
	sink.setBuf(meta.zeroStride, 0, 20)

	const bn, bk = 8, 32
	sink.dispatchThreadgroups(
		metal.MTLSize{Width: 1, Height: uint((outDim + bn - 1) / bn), Depth: 1},
		metal.MTLSize{Width: bk, Height: 2, Depth: 1},
	)
}

func gatherQMVBF16ByExpertIndex(x []byte, idx []int32, w QuantWeight, numExperts, topK, outDim, inDim, groupSize, bits int) ([]byte, error) {
	return gatherQMVBF16ByExpertIndexInto(nil, x, idx, w, numExperts, topK, outDim, inDim, groupSize, bits)
}

func gatherQMVBF16ByExpertIndexInto(out []byte, x []byte, idx []int32, w QuantWeight, numExperts, topK, outDim, inDim, groupSize, bits int) ([]byte, error) {
	if len(idx) != topK {
		return nil, core.NewError("native.gatherQMVBF16ByExpertIndex: idx length must equal topK")
	}
	for _, expert := range idx {
		if expert < 0 || int(expert) >= numExperts {
			return nil, core.NewError("native.gatherQMVBF16ByExpertIndex: expert index out of range")
		}
	}
	if len(x) != inDim*bf16Size {
		return nil, core.NewError("native.gatherQMVBF16ByExpertIndex: x must be inDim bf16 bytes")
	}
	outLen := topK * outDim * bf16Size
	callerOut := cap(out) >= outLen
	if !callerOut {
		out = make([]byte, outLen)
	} else {
		out = out[:outLen]
	}
	if topK == 0 || outDim == 0 || inDim == 0 {
		return out, nil
	}
	if bits != 4 {
		return nil, core.NewError("native.gatherQMVBF16ByExpertIndex: only 4-bit affine weights are supported")
	}
	if err := ensureInit(); err != nil {
		return nil, err
	}
	groupSize, bits = quantWeightGeometryForShape(w, numExperts*outDim, inDim, groupSize, bits)
	if groupSize <= 0 || bits != 4 || inDim%groupSize != 0 {
		return nil, core.NewError("native.gatherQMVBF16ByExpertIndex: invalid quant geometry")
	}
	wantPacked := numExperts * outDim * inDim * bits / 8
	wantSB := numExperts * outDim * (inDim / groupSize) * bf16Size
	if len(w.Packed) != wantPacked || len(w.Scales) != wantSB || len(w.Biases) != wantSB {
		return nil, core.NewError("native.gatherQMVBF16ByExpertIndex: quant weight size mismatch")
	}
	pso, err := gatherQMVBF16SteelPipeline(outDim, inDim, groupSize, bits)
	if err != nil {
		return nil, err
	}
	meta, err := gatherQMVBF16Metadata(numExperts, outDim, inDim, groupSize, bits, outDim)
	if err != nil {
		return nil, err
	}

	var encErr error
	withAutoreleasePool(func() {
		xBuf := sharedBytes(x)
		idxBuf := device.NewBufferWithBytesLengthOptions(unsafe.Pointer(&idx[0]), uint(len(idx)*4), metal.MTLResourceStorageModeShared)
		outBuf := device.NewBufferWithLengthOptions(uint(outLen), metal.MTLResourceStorageModeShared)
		directOut := false
		if callerOut && outLen > 0 {
			if buf, ok := registeredPinnedNoCopyBytes(out); ok {
				outBuf = buf
				directOut = true
			}
		}
		wBuf, sBuf, bBuf := quantWeightViews(w)

		cb := commandBufferFast(queue)
		enc := computeCommandEncoderFast(cb)
		sink := encSink{enc}
		for i := 0; i < topK; i++ {
			emitGatherQMVBF16Steel(sink, pso, meta, xBuf, wBuf.buf, wBuf.off, sBuf.buf, sBuf.off, bBuf.buf, bBuf.off, idxBuf, uint(i*4), outBuf, uint(i*outDim*bf16Size), outDim, inDim, groupSize, bits, 0)
		}
		endEncodingFast(enc)
		commitCommandBufferFast(cb)
		waitUntilCompletedFast(cb)
		if !directOut {
			copy(out, unsafe.Slice((*byte)(outBuf.Contents()), outLen))
		}
		runtime.KeepAlive(out)
	})
	if encErr != nil {
		return nil, encErr
	}
	return out, nil
}
