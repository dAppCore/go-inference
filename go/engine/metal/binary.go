// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"sync"

	core "dappco.re/go"
	"github.com/tmc/apple/metal"
)

var (
	binaryByteScratchPools  sync.Map
	errBinaryByteScratchDim = core.NewError("native.binaryByteScratch: dimension mismatch")
)

type binaryByteScratch struct {
	byteLen      int
	a, b         *pinnedNoCopyBytes
	aView, bView cachedNoCopyBytesView
	out          *pinnedNoCopyBytes
	noCopyOutputView
}

func binaryByteScratchPoolFor(byteLen int) *sync.Pool {
	if v, ok := binaryByteScratchPools.Load(byteLen); ok {
		return v.(*sync.Pool)
	}
	pool := new(sync.Pool)
	if v, loaded := binaryByteScratchPools.LoadOrStore(byteLen, pool); loaded {
		return v.(*sync.Pool)
	}
	return pool
}

func binaryByteScratchReady(s *binaryByteScratch, byteLen int) bool {
	return s != nil &&
		s.byteLen == byteLen &&
		s.a != nil &&
		s.a.buf != nil &&
		len(s.a.bytes) == byteLen &&
		s.b != nil &&
		s.b.buf != nil &&
		len(s.b.bytes) == byteLen &&
		s.out != nil &&
		s.out.buf != nil &&
		len(s.out.bytes) == byteLen
}

func newBinaryByteScratch(byteLen int) (*binaryByteScratch, error) {
	if byteLen <= 0 {
		return nil, core.NewError("native.newBinaryByteScratch: invalid byte length")
	}
	a, err := newPinnedNoCopyBytes(byteLen)
	if err != nil {
		return nil, err
	}
	b, err := newPinnedNoCopyBytes(byteLen)
	if err != nil {
		a.Close()
		return nil, err
	}
	out, err := newPinnedNoCopyBytes(byteLen)
	if err != nil {
		a.Close()
		b.Close()
		return nil, err
	}
	return &binaryByteScratch{byteLen: byteLen, a: a, b: b, out: out}, nil
}

func getBinaryByteScratch(byteLen int) (*binaryByteScratch, error) {
	pool := binaryByteScratchPoolFor(byteLen)
	if v := pool.Get(); v != nil {
		s := v.(*binaryByteScratch)
		if binaryByteScratchReady(s, byteLen) {
			return s, nil
		}
		s.Close()
	}
	return newBinaryByteScratch(byteLen)
}

func putBinaryByteScratch(s *binaryByteScratch) {
	if s == nil {
		return
	}
	if binaryByteScratchReady(s, s.byteLen) {
		binaryByteScratchPoolFor(s.byteLen).Put(s)
	}
}

func (s *binaryByteScratch) Close() {
	if s == nil {
		return
	}
	if s.a != nil {
		s.a.Close()
		s.a = nil
	}
	if s.b != nil {
		s.b.Close()
		s.b = nil
	}
	s.aView.Close()
	s.bView.Close()
	if s.out != nil {
		s.out.Close()
		s.out = nil
	}
	s.closeOutputView()
	s.byteLen = 0
}

func (s *binaryByteScratch) buffers(a, b []byte) (metal.MTLBuffer, metal.MTLBuffer, metal.MTLBuffer, error) {
	if s == nil || s.a == nil || s.b == nil || s.out == nil {
		return nil, nil, nil, core.NewError("native.binaryByteScratch.buffers: scratch is nil")
	}
	if len(a) != s.byteLen || len(b) != s.byteLen || len(s.out.bytes) != s.byteLen {
		return nil, nil, nil, errBinaryByteScratchDim
	}
	var err error
	aBuf, aNoCopy := s.aView.buffer(a)
	if !aNoCopy {
		aBuf, err = s.a.copyBuffer(a)
		if err != nil {
			return nil, nil, nil, err
		}
	}
	bBuf, bNoCopy := s.bView.buffer(b)
	if !bNoCopy {
		bBuf, err = s.b.copyBuffer(b)
		if err != nil {
			return nil, nil, nil, err
		}
	}
	return aBuf, bBuf, s.out.buf, nil
}

// RunBinary drives a contiguous binary MLX kernel over two equal-length inputs
// and returns a fresh result slice. It targets the vv_<Op>float32 family, whose
// host ABI (from mlx/backend/metal/binary.cpp) is: a → buffer(0), b → buffer(1),
// out → buffer(2), element count → buffer(3), one GPU thread per element. name is
// e.g. "vv_Addfloat32". The byte-for-byte equivalent of the mlx-c contiguous
// binary path — parity is gated in the tests.
func RunBinary(name string, a, b []float32) ([]float32, error) {
	out := make([]float32, len(a))
	if err := runBinaryInto(name, a, b, out, false); err != nil {
		return nil, err
	}
	return out, nil
}

// RunBinaryInto is RunBinary writing the result into the caller-supplied out
// (len(out) must equal len(a)) instead of allocating a fresh slice. It exists so
// a composed op (e.g. Gelu) can ping-pong a couple of reusable scratch buffers
// across its chain rather than allocating one result slice per primitive — the
// dominant B/op of the float32 compose path. The GPU work, kernel, and inputs
// are identical to RunBinary, so the bytes written are identical; only the Go
// destination differs.
func RunBinaryInto(name string, a, b, out []float32) error {
	return runBinaryInto(name, a, b, out, true)
}

func runBinaryInto(name string, a, b, out []float32, directOutput bool) error {
	if err := ensureInit(); err != nil {
		return err
	}
	if len(a) != len(b) {
		return core.NewError("native.RunBinaryInto: a and b must be the same length")
	}
	if len(out) != len(a) {
		return core.NewError("native.RunBinaryInto: out must be the same length as a")
	}
	pso, err := pipelineFor(name)
	if err != nil {
		return err
	}
	n := len(a)
	if n == 0 {
		return nil
	}

	var encErr error
	withAutoreleasePool(func() {
		ioScratch, err := getBinaryByteScratch(n * 4)
		if err != nil {
			encErr = err
			return
		}
		defer putBinaryByteScratch(ioScratch)
		aBuf, bBuf, outBuf, err := ioScratch.buffers(float32Bytes(a), float32Bytes(b))
		if err != nil {
			encErr = err
			return
		}
		directOut := false
		if directOutput {
			if tmp, ok := ioScratch.outputView(float32Bytes(out)); ok {
				outBuf = tmp
				directOut = true
			}
		}

		cb := commandBufferFast(queue)
		enc := computeCommandEncoderFast(cb)
		emitBinary(encSink{enc}, pso, aBuf, 0, bBuf, 0, outBuf, 0, n)
		endEncodingFast(enc)
		commitCommandBufferFast(cb)
		waitUntilCompletedFast(cb)

		if !directOut {
			copy(float32Bytes(out), ioScratch.out.bytes[:n*4])
		}
	})
	if encErr != nil {
		return encErr
	}
	return nil
}

// Add returns the element-wise sum a[i]+b[i] on the GPU via the shared
// mlx.metallib (kernel vv_Addfloat32). This is the residual add used twice per
// decode block. Parity with pkg/metal.Add is gated in parity_test.go.
//
//	out, err := native.Add([]float32{1, 2}, []float32{3, 4}) // out = [4 6]
func Add(a, b []float32) ([]float32, error) {
	return RunBinary("vv_Addfloat32", a, b)
}

// Mul returns the element-wise product a[i]*b[i] on the GPU via the shared
// mlx.metallib (kernel vv_Multiplyfloat32) — the gate·up step of the MLP. Parity
// with pkg/metal.Mul is gated in parity_test.go.
//
//	out, err := native.Mul([]float32{2, 3}, []float32{4, 5}) // out = [8 15]
func Mul(a, b []float32) ([]float32, error) {
	return RunBinary("vv_Multiplyfloat32", a, b)
}
