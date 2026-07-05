// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"runtime"
	"sync"
	"unsafe"

	core "dappco.re/go"
	"github.com/tmc/apple/metal"
	"github.com/tmc/apple/objc"
)

type qmvKernelNameKey struct {
	groupSize, bits int
	fast            bool
}

type qmvScratchKey struct {
	outDim, inDim int
}

var (
	qmvKernelNames        sync.Map
	qmvFloatScratchPools  sync.Map
	qmvBF16ScratchPools   sync.Map
	errQMVFloatScratchDim = core.NewError("native.qmvFloatScratch: dimension mismatch")
	errQMVBF16ScratchDim  = core.NewError("native.qmvBF16Scratch: dimension mismatch")
)

type qmvScratchPool struct {
	core.Pool[any]
}

func qmvScratchPoolFor(pools *sync.Map, outDim, inDim int) *qmvScratchPool {
	key := qmvScratchKey{outDim: outDim, inDim: inDim}
	if v, ok := pools.Load(key); ok {
		return v.(*qmvScratchPool)
	}
	pool := &qmvScratchPool{}
	if v, loaded := pools.LoadOrStore(key, pool); loaded {
		return v.(*qmvScratchPool)
	}
	return pool
}

type qmvFloatScratch struct {
	inDim, outDim int
	x, out        *pinnedNoCopyBytes
	xView         cachedNoCopyBytesView
	outViews      [2]cachedNoCopyBytesView
	outViewNext   int
}

func newQMVFloatScratch(outDim, inDim int) (*qmvFloatScratch, error) {
	if outDim <= 0 || inDim <= 0 {
		return nil, core.NewError("native.newQMVFloatScratch: invalid dimensions")
	}
	x, err := newPinnedNoCopyBytes(inDim * 4)
	if err != nil {
		return nil, err
	}
	out, err := newPinnedNoCopyBytes(outDim * 4)
	if err != nil {
		x.Close()
		return nil, err
	}
	return &qmvFloatScratch{inDim: inDim, outDim: outDim, x: x, out: out}, nil
}

func getQMVFloatScratch(outDim, inDim int) (*qmvFloatScratch, error) {
	pool := qmvScratchPoolFor(&qmvFloatScratchPools, outDim, inDim)
	if v := pool.Get(); v != nil {
		s := v.(*qmvFloatScratch)
		if s.inDim == inDim && s.outDim == outDim && s.x != nil && s.out != nil {
			return s, nil
		}
		s.Close()
	}
	return newQMVFloatScratch(outDim, inDim)
}

func putQMVFloatScratch(s *qmvFloatScratch) {
	if s != nil && s.inDim > 0 && s.outDim > 0 && s.x != nil && s.out != nil {
		qmvScratchPoolFor(&qmvFloatScratchPools, s.outDim, s.inDim).Put(s)
	}
}

func (s *qmvFloatScratch) Close() {
	if s == nil {
		return
	}
	if s.x != nil {
		s.x.Close()
		s.x = nil
	}
	s.xView.Close()
	if s.out != nil {
		s.out.Close()
		s.out = nil
	}
	s.closeOutputView()
	s.inDim, s.outDim = 0, 0
}

func (s *qmvFloatScratch) closeOutputView() {
	if s == nil {
		return
	}
	for i := range s.outViews {
		s.outViews[i].Close()
	}
	s.outViewNext = 0
}

func (s *qmvFloatScratch) outputView(out []float32) (metal.MTLBuffer, bool) {
	if s == nil || len(out) == 0 {
		return nil, false
	}
	ptr := uintptr(unsafe.Pointer(&out[0]))
	byteLen := len(out) * 4
	for i := range s.outViews {
		if s.outViews[i].buf != nil && s.outViews[i].ptr == ptr && s.outViews[i].len == byteLen {
			return s.outViews[i].buf, true
		}
	}
	slot := -1
	for i := range s.outViews {
		if s.outViews[i].buf == nil {
			slot = i
			break
		}
	}
	if slot < 0 {
		slot = s.outViewNext % len(s.outViews)
		s.outViews[slot].Close()
	}
	outBytes := float32Bytes(out)
	buf, ok := s.outViews[slot].buffer(outBytes)
	if ok {
		s.outViewNext = (slot + 1) % len(s.outViews)
	}
	return buf, ok
}

func (s *qmvFloatScratch) buffers(x []float32) (metal.MTLBuffer, metal.MTLBuffer, error) {
	if s == nil || s.x == nil || s.out == nil {
		return nil, nil, core.NewError("native.qmvFloatScratch.buffers: scratch is nil")
	}
	if len(x) != s.inDim || len(s.out.bytes) != s.outDim*4 {
		return nil, nil, errQMVFloatScratchDim
	}
	var err error
	xBytes := float32Bytes(x)
	xBuf, xNoCopy := s.xView.bufferAfterStable(xBytes, 3)
	if !xNoCopy {
		xBuf, err = s.x.copyBuffer(xBytes)
		if err != nil {
			return nil, nil, err
		}
	}
	return xBuf, s.out.buf, nil
}

func float32Bytes(x []float32) []byte {
	if len(x) == 0 {
		return nil
	}
	return unsafe.Slice((*byte)(unsafe.Pointer(&x[0])), len(x)*4)
}

type qmvBF16Scratch struct {
	inDim, outDim int
	x, out        *pinnedNoCopyBytes
	xView         cachedNoCopyBytesView
	outViewPtr    uintptr
	outViewLen    int
	outView       metal.MTLBuffer
	outViewPinned *pinnedNoCopyBytes
	residentIDs   []objc.ID
}

func newQMVBF16Scratch(outDim, inDim int) (*qmvBF16Scratch, error) {
	if outDim <= 0 || inDim <= 0 {
		return nil, core.NewError("native.newQMVBF16Scratch: invalid dimensions")
	}
	x, err := newPinnedNoCopyBytes(inDim * bf16Size)
	if err != nil {
		return nil, err
	}
	out, err := newPinnedNoCopyBytes(outDim * bf16Size)
	if err != nil {
		x.Close()
		return nil, err
	}
	return &qmvBF16Scratch{inDim: inDim, outDim: outDim, x: x, out: out}, nil
}

func getQMVBF16Scratch(outDim, inDim int) (*qmvBF16Scratch, error) {
	pool := qmvScratchPoolFor(&qmvBF16ScratchPools, outDim, inDim)
	if v := pool.Get(); v != nil {
		s := v.(*qmvBF16Scratch)
		if s.inDim == inDim && s.outDim == outDim && s.x != nil && s.out != nil {
			return s, nil
		}
		s.Close()
	}
	return newQMVBF16Scratch(outDim, inDim)
}

func putQMVBF16Scratch(s *qmvBF16Scratch) {
	if s != nil && s.inDim > 0 && s.outDim > 0 && s.x != nil && s.out != nil {
		qmvScratchPoolFor(&qmvBF16ScratchPools, s.outDim, s.inDim).Put(s)
	}
}

func (s *qmvBF16Scratch) Close() {
	if s == nil {
		return
	}
	if s.x != nil {
		s.x.Close()
		s.x = nil
	}
	s.xView.Close()
	if s.out != nil {
		s.out.Close()
		s.out = nil
	}
	s.closeOutputView()
	s.inDim, s.outDim = 0, 0
}

func (s *qmvBF16Scratch) closeOutputView() {
	if s == nil {
		return
	}
	if s.outViewPinned != nil {
		s.outViewPinned.Close()
	}
	s.outViewPtr = 0
	s.outViewLen = 0
	s.outView = nil
	s.outViewPinned = nil
}

func (s *qmvBF16Scratch) outputView(out []byte) (metal.MTLBuffer, bool) {
	if s == nil || len(out) == 0 {
		return nil, false
	}
	ptr := uintptr(unsafe.Pointer(&out[0]))
	if s.outView != nil && s.outViewPtr == ptr && s.outViewLen == len(out) {
		return s.outView, true
	}
	s.closeOutputView()
	if buf, ok := registeredPinnedNoCopyBytes(out); ok {
		s.outViewPtr = ptr
		s.outViewLen = len(out)
		s.outView = buf
		s.outViewPinned = nil
		return buf, true
	}
	buf, pinner, noCopy := residentNoCopyBytes(out)
	if !noCopy {
		if pinner != nil {
			pinner.Unpin()
		}
		return nil, false
	}
	pinned := &pinnedNoCopyBytes{bytes: out, buf: buf, pinner: pinner}
	runtime.SetFinalizer(pinned, (*pinnedNoCopyBytes).Close)
	s.outViewPtr = ptr
	s.outViewLen = len(out)
	s.outView = buf
	s.outViewPinned = pinned
	return buf, true
}

func (s *qmvBF16Scratch) buffers(x []byte) (metal.MTLBuffer, metal.MTLBuffer, error) {
	if s == nil || s.x == nil || s.out == nil {
		return nil, nil, core.NewError("native.qmvBF16Scratch.buffers: scratch is nil")
	}
	if len(x) != s.inDim*bf16Size || len(s.out.bytes) != s.outDim*bf16Size {
		return nil, nil, errQMVBF16ScratchDim
	}
	var err error
	if xBuf, ok := registeredPinnedNoCopyBytes(x); ok {
		return xBuf, s.out.buf, nil
	}
	xBuf, xNoCopy := s.xView.bufferAfterStable(x, 3)
	if !xNoCopy {
		xBuf, err = s.x.copyBuffer(x)
		if err != nil {
			return nil, nil, err
		}
	}
	return xBuf, s.out.buf, nil
}

func qmvKernelName(outDim, inDim, groupSize, bits int) string {
	fast := outDim%8 == 0 && inDim%512 == 0
	key := qmvKernelNameKey{groupSize: groupSize, bits: bits, fast: fast}
	if v, ok := qmvKernelNames.Load(key); ok {
		return v.(string)
	}
	variant := "_qmv_"
	if fast {
		variant = "_qmv_fast_"
	}
	name := core.Sprintf("affine%sfloat_gs_%d_b_%d_batch_0", variant, groupSize, bits)
	if v, loaded := qmvKernelNames.LoadOrStore(key, name); loaded {
		return v.(string)
	}
	return name
}

// QMV computes out = x @ Wᵀ for a 4-bit (affine) quantised weight matrix — the
// 4-bit decode hot path. wq/scales/biases are the raw packed bytes MLX's
// quantiser produces for a logically (outDim x inDim) weight; x is a length-inDim
// float32 activation vector; the result is length outDim. It drives MLX's
// affine_qmv kernel directly through the no-cgo path: w(0) scales(1) biases(2)
// x(3) out(4) K(5) N(6) — and because this is a single (B<=1) matvec, MLX's
// add_strides_and_shapes early-returns, so there are no batch params to set.
// group_size and bits are baked into the kernel name. float32 activations only.
//
// Byte-for-byte parity with pkg/metal.QuantizedMatmul (transpose=true) on the
// same packed bytes is gated in parity_test.go.
func QMV(x []float32, wq, scales, biases []byte, outDim, inDim, groupSize, bits int) ([]float32, error) {
	return QMVInto(nil, x, wq, scales, biases, outDim, inDim, groupSize, bits)
}

func QMVInto(out []float32, x []float32, wq, scales, biases []byte, outDim, inDim, groupSize, bits int) ([]float32, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	if len(x) != inDim {
		return nil, core.NewError("native.QMV: len(x) must equal inDim")
	}
	callerOut := cap(out) >= outDim
	if !callerOut {
		out = make([]float32, outDim)
	} else {
		out = out[:outDim]
	}
	if outDim == 0 || inDim == 0 {
		return out, nil
	}

	name := qmvKernelName(outDim, inDim, groupSize, bits)
	pso, err := pipelineFor(name)
	if err != nil {
		return nil, err
	}

	var encErr error
	withAutoreleasePool(func() {
		wBuf := residentBytes(wq)
		sBuf := residentBytes(scales)
		bBuf := residentBytes(biases)
		scratch, err := getQMVFloatScratch(outDim, inDim)
		if err != nil {
			encErr = err
			return
		}
		defer putQMVFloatScratch(scratch)
		xBuf, outBuf, err := scratch.buffers(x)
		if err != nil {
			encErr = err
			return
		}
		clear(scratch.out.bytes[:outDim*4])

		cb := commandBufferFast(queue)
		enc := computeCommandEncoderFast(cb)
		emitQMV(encSink{enc}, pso, wBuf, 0, sBuf, 0, bBuf, 0, xBuf, outBuf, 0, inDim, outDim)
		endEncodingFast(enc)
		commitCommandBufferFast(cb)
		waitUntilCompletedFast(cb)

		copy(float32Bytes(out), scratch.out.bytes[:outDim*4])
	})
	if encErr != nil {
		return nil, encErr
	}
	return out, nil
}

// QMVBF16 is the bfloat16-activation sibling of QMV: out = x @ Wᵀ for a 4-bit
// (affine) quantised weight matrix, with bf16 activations, scales, biases and
// output — the quantised decode projection. x is inDim bf16 bytes; wq/scales/
// biases are the packed bytes MLX's quantiser produces for a bf16 (outDim x inDim)
// weight (scales and biases bf16, one per group per row); the result is outDim
// bf16 bytes. It drives affine_qmv[_fast]_bfloat16_t_gs_G_b_B_batch_0 — the same
// kernel template and host ABI as QMV (w0 s1 b2 x3 out4 K5 N6; single B<=1 matvec,
// so MLX's add_strides_and_shapes early-returns and there are no batch params),
// only the activation dtype differs. Because the decode path is already bf16, this
// needs NO precision conversion around the projections (unlike float QMV). The bf16
// type token is bfloat16_t. Byte-for-byte parity with pkg/metal.QuantizedMatmul
// (transpose=true) on bf16 inputs + the same packed bytes is gated in parity_test.go.
func QMVBF16(x, wq, scales, biases []byte, outDim, inDim, groupSize, bits int) ([]byte, error) {
	return QMVBF16Into(nil, x, wq, scales, biases, outDim, inDim, groupSize, bits)
}

func QMVBF16Into(out []byte, x, wq, scales, biases []byte, outDim, inDim, groupSize, bits int) ([]byte, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	if len(x) != inDim*bf16Size {
		return nil, core.NewError("native.QMVBF16: len(x) must equal inDim bf16 bytes")
	}
	if outDim == 0 || inDim == 0 {
		outLen := outDim * bf16Size
		if cap(out) < outLen {
			return make([]byte, outLen), nil
		}
		return out[:outLen], nil
	}
	pso, err := pipelineFor(qmvBF16KernelName(outDim, inDim, groupSize, bits))
	if err != nil {
		return nil, err
	}

	outLen := outDim * bf16Size
	callerOut := cap(out) >= outLen
	if !callerOut {
		out = make([]byte, outLen)
	} else {
		out = out[:outLen]
	}
	var encErr error
	withAutoreleasePool(func() {
		wBuf, sBuf, bBuf := residentBytes(wq), residentBytes(scales), residentBytes(biases)
		scratch, err := getQMVBF16Scratch(outDim, inDim)
		if err != nil {
			encErr = err
			return
		}
		defer putQMVBF16Scratch(scratch)
		xBuf, outBuf, err := scratch.buffers(x)
		if err != nil {
			encErr = err
			return
		}
		directOut := false
		if callerOut {
			if tmp, ok := scratch.outputView(out); ok {
				outBuf = tmp
				directOut = true
			}
		}

		cb := commandBufferFast(queue)
		enc := computeCommandEncoderFast(cb)
		emitQMV(encSink{enc}, pso, wBuf, 0, sBuf, 0, bBuf, 0, xBuf, outBuf, 0, inDim, outDim)
		endEncodingFast(enc)
		commitCommandBufferFast(cb)
		waitUntilCompletedFast(cb)

		if !directOut {
			copy(out, scratch.out.bytes[:outLen])
		}
	})
	if encErr != nil {
		return nil, encErr
	}
	return out, nil
}

func qmvBF16Resident(x []byte, w QuantWeight, outDim, inDim, groupSize, bits int) ([]byte, error) {
	return qmvBF16ResidentInto(nil, x, w, outDim, inDim, groupSize, bits)
}

func qmvBF16ResidentInto(out []byte, x []byte, w QuantWeight, outDim, inDim, groupSize, bits int) ([]byte, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	if len(x) != inDim*bf16Size {
		return nil, core.NewError("native.qmvBF16Resident: len(x) must equal inDim bf16 bytes")
	}
	if outDim == 0 || inDim == 0 {
		outLen := outDim * bf16Size
		if cap(out) < outLen {
			return make([]byte, outLen), nil
		}
		return out[:outLen], nil
	}
	groupSize, bits = quantWeightGeometryForShape(w, outDim, inDim, groupSize, bits)
	if groupSize <= 0 || bits <= 0 || inDim%groupSize != 0 {
		return nil, core.NewError("native.qmvBF16Resident: invalid quant geometry")
	}
	wantPacked := outDim * inDim * bits / 8
	wantSB := outDim * (inDim / groupSize) * bf16Size
	if len(w.Packed) != wantPacked || len(w.Scales) != wantSB || len(w.Biases) != wantSB {
		return nil, core.NewError("native.qmvBF16Resident: quant weight size mismatch")
	}
	pso, err := pipelineFor(qmvBF16KernelName(outDim, inDim, groupSize, bits))
	if err != nil {
		return nil, err
	}

	outLen := outDim * bf16Size
	callerOut := cap(out) >= outLen
	if !callerOut {
		out = make([]byte, outLen)
	} else {
		out = out[:outLen]
	}
	var encErr error
	withAutoreleasePool(func() {
		wBuf, sBuf, bBuf := quantWeightViews(w)
		scratch, err := getQMVBF16Scratch(outDim, inDim)
		if err != nil {
			encErr = err
			return
		}
		defer putQMVBF16Scratch(scratch)
		xBuf, outBuf, err := scratch.buffers(x)
		if err != nil {
			encErr = err
			return
		}
		directOut := false
		if callerOut {
			if tmp, ok := scratch.outputView(out); ok {
				outBuf = tmp
				directOut = true
			}
		}

		cb := commandBufferFast(queue)
		enc := computeCommandEncoderFast(cb)
		emitQMV(encSink{enc}, pso, wBuf.buf, wBuf.off, sBuf.buf, sBuf.off, bBuf.buf, bBuf.off, xBuf, outBuf, 0, inDim, outDim)
		endEncodingFast(enc)
		commitCommandBufferFast(cb)
		waitUntilCompletedFast(cb)

		if !directOut {
			copy(out, scratch.out.bytes[:outLen])
		}
	})
	if encErr != nil {
		return nil, encErr
	}
	return out, nil
}

func quantWeightViews(w QuantWeight) (bufView, bufView, bufView) {
	if w.packedView.buf != nil && w.scalesView.buf != nil && w.biasesView.buf != nil {
		return w.packedView, w.scalesView, w.biasesView
	}
	return bufView{buf: residentBytes(w.Packed)}, bufView{buf: residentBytes(w.Scales)}, bufView{buf: residentBytes(w.Biases)}
}

func bf16WeightView(weight []byte, view bufView) bufView {
	if view.buf != nil {
		return view
	}
	return bufView{buf: residentBytes(weight)}
}

func rmsNormBF16View(x, weight []byte, weightView bufView, rows, axisSize int, eps float32) ([]byte, error) {
	return rmsNormBF16ViewInto(nil, x, weight, weightView, rows, axisSize, eps)
}

func rmsNormBF16ViewInto(out []byte, x, weight []byte, weightView bufView, rows, axisSize int, eps float32) ([]byte, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	if len(x) != rows*axisSize*bf16Size {
		return nil, core.NewError("native.rmsNormBF16View: len(x) must equal rows*axisSize*2 bytes")
	}
	if len(weight) != axisSize*bf16Size {
		return nil, core.NewError("native.rmsNormBF16View: len(weight) must equal axisSize*2 bytes")
	}
	if rows == 0 || axisSize == 0 {
		if cap(out) < len(x) {
			return make([]byte, len(x)), nil
		}
		return out[:len(x)], nil
	}

	outLen := len(x)
	callerOut := cap(out) >= outLen
	if !callerOut {
		out = make([]byte, outLen)
	} else {
		out = out[:outLen]
	}
	var encErr error
	withAutoreleasePool(func() {
		w := bf16WeightView(weight, weightView)
		scratch, err := getQMVBF16Scratch(rows*axisSize, rows*axisSize)
		if err != nil {
			encErr = err
			return
		}
		defer putQMVBF16Scratch(scratch)
		xBuf, outBuf, err := scratch.buffers(x)
		if err != nil {
			encErr = err
			return
		}
		directOut := false
		if callerOut {
			if tmp, ok := scratch.outputView(out); ok {
				outBuf = tmp
				directOut = true
			}
		}

		cb := commandBufferFast(queue)
		enc := computeCommandEncoderFast(cb)
		if encErr = encRMSNormRowsBF16(enc, xBuf, w.buf, outBuf, 0, w.off, 0, rows, axisSize, eps); encErr != nil {
			endEncodingFast(enc)
			return
		}
		endEncodingFast(enc)
		commitCommandBufferFast(cb)
		waitUntilCompletedFast(cb)
		if !directOut {
			copy(out, scratch.out.bytes[:outLen])
		}
	})
	if encErr != nil {
		return nil, encErr
	}
	return out, nil
}
