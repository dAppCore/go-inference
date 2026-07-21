// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"math"
	"sync"

	core "dappco.re/go"
	"github.com/tmc/apple/metal"
)

// bfloat16 MLP pieces: the bf16 elementwise ops and the composed bf16 GELU gate
// that a bf16 feed-forward block needs. Like the other bf16 ops these take/return
// raw bf16 []byte. GELU is composed (no metallib kernel) — in bf16 the question
// is whether the separately-run, each-step-rounded composition matches mlx's
// mlx_compile-fused gelu; the parity test answers it with data.

// f32ToBF16 converts a float32 to bfloat16 bits with round-to-nearest-even —
// matching mlx's AsType(..., bfloat16) so a constant produced here equals the
// same constant mlx would broadcast.
func f32ToBF16(v float32) uint16 {
	bits := math.Float32bits(v)
	if bits&0x7fffffff > 0x7f800000 { // NaN: keep it quiet, non-zero mantissa
		return uint16(bits>>16) | 0x0040
	}
	rounding := (bits>>16)&1 + 0x7fff
	return uint16((bits + rounding) >> 16)
}

// bf16ConstBytes returns n copies of v as bf16 bytes — a broadcast scalar operand
// for the contiguous bf16 binary kernels.
// bf16ConstKey identifies a materialised bf16 broadcast-scalar operand by length
// and value, so identical (n, v) requests share one immutable backing buffer.
type bf16ConstKey struct {
	n int
	v float32
}

// bf16ConstCache memoises the dense bf16 scalar operands bf16ConstBytes builds.
// The composed GeluBF16 fires the same four compile-time constants at a fixed
// decode width every call; caching collapses the per-call make([]byte, n*2) to a
// one-time fill. Entries are read-only kernel operands, so the shared buffer is
// byte-identical to a freshly filled one.
var (
	bf16ConstMu    sync.Mutex
	bf16ConstCache = map[bf16ConstKey][]byte{}
)

func bf16ConstBytes(n int, v float32) []byte {
	if n == 0 {
		return nil
	}
	key := bf16ConstKey{n: n, v: v}
	bf16ConstMu.Lock()
	defer bf16ConstMu.Unlock()
	if s, ok := bf16ConstCache[key]; ok {
		return s
	}
	h := f32ToBF16(v)
	out := make([]byte, n*bf16Size)
	for i := range n {
		out[i*2] = byte(h)
		out[i*2+1] = byte(h >> 8)
	}
	bf16ConstCache[key] = out
	return out
}

func bf16ConstBuffer(n int, v float32) metal.MTLBuffer {
	b := bf16ConstBytes(n, v)
	if len(b) == 0 {
		return nil
	}
	return residentBytes(b)
}

// runBinaryBF16 drives a contiguous bf16 binary kernel (vv_<Op>bfloat16) over two
// equal-length bf16 byte buffers. name e.g. "vv_Multiplybfloat16".
func runBinaryBF16(name string, a, b []byte) ([]byte, error) {
	out := make([]byte, len(a))
	if err := runBinaryBF16IntoDirect(name, a, b, out, false); err != nil {
		return nil, err
	}
	return out, nil
}

func mulBF16Const(a []byte, n int, v float32) ([]byte, error) {
	out := make([]byte, len(a))
	if err := binaryBF16ConstIntoDirect("mulBF16ConstInto", encMulBF16, a, n, v, out, false); err != nil {
		return nil, err
	}
	return out, nil
}

func mulBF16ConstInto(a []byte, n int, v float32, out []byte) error {
	return binaryBF16ConstInto("mulBF16ConstInto", encMulBF16, a, n, v, out)
}

func addBF16ConstInto(a []byte, n int, v float32, out []byte) error {
	return binaryBF16ConstInto("addBF16ConstInto", encAddBF16, a, n, v, out)
}

func binaryBF16ConstInto(name string, encFn func(metal.MTLComputeCommandEncoder, metal.MTLBuffer, metal.MTLBuffer, metal.MTLBuffer, int) error, a []byte, n int, v float32, out []byte) error {
	return binaryBF16ConstIntoDirect(name, encFn, a, n, v, out, true)
}

func binaryBF16ConstIntoDirect(name string, encFn func(metal.MTLComputeCommandEncoder, metal.MTLBuffer, metal.MTLBuffer, metal.MTLBuffer, int) error, a []byte, n int, v float32, out []byte, directOutput bool) error {
	if err := ensureInit(); err != nil {
		return err
	}
	if n < 0 {
		return core.NewError("native." + name + ": n must be >= 0")
	}
	if len(a) != n*bf16Size {
		return core.NewError("native." + name + ": a must be n bf16 values")
	}
	if len(out) != len(a) {
		return core.NewError("native." + name + ": out must be the same length as a")
	}
	if n == 0 {
		return nil
	}
	var encErr error
	withAutoreleasePool(func() {
		scratch, err := getQMVBF16Scratch(n, n)
		if err != nil {
			encErr = err
			return
		}
		defer putQMVBF16Scratch(scratch)
		aBuf, outBuf, err := scratch.buffers(a)
		if err != nil {
			encErr = err
			return
		}
		directOut := false
		if directOutput {
			tmp, ok := scratch.outputView(out)
			if ok {
				outBuf = tmp
				directOut = true
			}
		}
		cBuf := bf16ConstBuffer(n, v)
		cb := commandBufferFast(queue)
		enc := computeCommandEncoderFast(cb)
		if encErr = encFn(enc, aBuf, cBuf, outBuf, n); encErr != nil {
			endEncodingFast(enc)
			return
		}
		endEncodingFast(enc)
		commitCommandBufferFast(cb)
		waitUntilCompletedFast(cb)
		if !directOut {
			copy(out, scratch.out.bytes[:len(out)])
		}
	})
	return encErr
}

// runBinaryBF16Into is runBinaryBF16 writing into the caller-supplied out
// (len(out) must equal len(a)) instead of allocating a fresh buffer, so a
// composed bf16 op (GeluBF16) can ping-pong reusable scratch across its chain.
// Same kernel and inputs as runBinaryBF16 — the bytes written are identical.
func runBinaryBF16Into(name string, a, b, out []byte) error {
	return runBinaryBF16IntoDirect(name, a, b, out, true)
}

func runBinaryBF16IntoDirect(name string, a, b, out []byte, directOutput bool) error {
	if err := ensureInit(); err != nil {
		return err
	}
	if len(a) != len(b) {
		return core.NewError("native.runBinaryBF16Into: a and b must be the same length")
	}
	if len(a)%bf16Size != 0 {
		return core.NewError("native.runBinaryBF16Into: byte length must be a multiple of 2")
	}
	if len(out) != len(a) {
		return core.NewError("native.runBinaryBF16Into: out must be the same length as a")
	}
	pso, err := pipelineFor(name)
	if err != nil {
		return err
	}
	n := len(a) / bf16Size
	if n == 0 {
		return nil
	}
	var encErr error
	withAutoreleasePool(func() {
		ioScratch, err := getBinaryByteScratch(len(a))
		if err != nil {
			encErr = err
			return
		}
		defer putBinaryByteScratch(ioScratch)
		aBuf, bBuf, outBuf, err := ioScratch.buffers(a, b)
		if err != nil {
			encErr = err
			return
		}
		directOut := false
		if directOutput {
			tmp, ok := ioScratch.outputView(out)
			if ok {
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
			copy(out, ioScratch.out.bytes[:len(a)])
		}
	})
	return encErr
}

// MulBF16 is the bf16 sibling of Mul: element-wise a[i]*b[i] over bf16 bytes
// (kernel vv_Multiplybfloat16) — the MLP gate·up step in the decode dtype.
func MulBF16(a, b []byte) ([]byte, error) { return runBinaryBF16("vv_Multiplybfloat16", a, b) }

func MulBF16Into(out, a, b []byte) error { return runBinaryBF16Into("vv_Multiplybfloat16", a, b, out) }

// TanhBF16 is the bf16 sibling of Tanh: element-wise tanh over bf16 bytes (kernel
// v_Tanhbfloat16bfloat16) — the nonlinearity inside the gelu approximation.
func TanhBF16(x []byte) ([]byte, error) {
	out := make([]byte, len(x))
	if err := tanhBF16IntoDirect(x, out, false); err != nil {
		return nil, err
	}
	return out, nil
}

func TanhBF16Into(out, x []byte) error {
	return tanhBF16IntoDirect(x, out, true)
}

// tanhBF16Into is TanhBF16 writing into the caller-supplied out (len(out) must
// equal len(x)) instead of allocating, so GeluBF16 can keep the tanh step on its
// ping-pong scratch. Same kernel and input as TanhBF16 — bytes are identical.
func tanhBF16Into(x, out []byte) error {
	return tanhBF16IntoDirect(x, out, false)
}

func tanhBF16IntoDirect(x, out []byte, directOutput bool) error {
	if err := ensureInit(); err != nil {
		return err
	}
	if len(x)%bf16Size != 0 {
		return core.NewError("native.tanhBF16Into: byte length must be a multiple of 2")
	}
	if len(out) != len(x) {
		return core.NewError("native.tanhBF16Into: out must be the same length as x")
	}
	pso, err := pipelineFor("v_Tanhbfloat16bfloat16")
	if err != nil {
		return err
	}
	n := len(x) / bf16Size
	if n == 0 {
		return nil
	}
	var encErr error
	withAutoreleasePool(func() {
		scratch, err := getQMVBF16Scratch(n, n)
		if err != nil {
			encErr = err
			return
		}
		defer putQMVBF16Scratch(scratch)
		inBuf, outBuf, err := scratch.buffers(x)
		if err != nil {
			encErr = err
			return
		}
		directOut := false
		if directOutput {
			tmp, ok := scratch.outputView(out)
			if ok {
				outBuf = tmp
				directOut = true
			}
		}
		cb := commandBufferFast(queue)
		enc := computeCommandEncoderFast(cb)
		emitUnary(encSink{enc}, pso, inBuf, outBuf, n)
		endEncodingFast(enc)
		commitCommandBufferFast(cb)
		waitUntilCompletedFast(cb)
		if !directOut {
			copy(out, scratch.out.bytes[:len(x)])
		}
	})
	return encErr
}

func encGeluBF16Composed(enc metal.MTLComputeCommandEncoder, x, out metal.MTLBuffer, n int) error {
	c044 := bf16ConstBuffer(n, 0.044715)
	c079 := bf16ConstBuffer(n, 0.7978845608028654)
	c1 := bf16ConstBuffer(n, 1.0)
	c05 := bf16ConstBuffer(n, 0.5)
	a, b := scratchBF16(n), scratchBF16(n)

	// x2=x*x->a; x3=a*x->b; x3s=b*c044->a; inner=x+a->b;
	// scaled=b*c079->a; t=tanh(a)->b; onePlus=b+c1->a;
	// halfX=x*c05->b; gelu=b*a->out.
	if err := encMulBF16(enc, x, x, a, n); err != nil {
		return err
	}
	if err := encMulBF16(enc, a, x, b, n); err != nil {
		return err
	}
	if err := encMulBF16(enc, b, c044, a, n); err != nil {
		return err
	}
	if err := encAddBF16(enc, x, a, b, n); err != nil {
		return err
	}
	if err := encMulBF16(enc, b, c079, a, n); err != nil {
		return err
	}
	if err := encTanhBF16(enc, a, b, n); err != nil {
		return err
	}
	if err := encAddBF16(enc, b, c1, a, n); err != nil {
		return err
	}
	if err := encMulBF16(enc, x, c05, b, n); err != nil {
		return err
	}
	return encMulBF16(enc, b, a, out, n)
}

// GeluBF16 is the bf16 sibling of Gelu: the tanh-approximation GELU composed from
// the bf16 primitives, each intermediate rounded to bf16 (the gelu_approx graph
// MLX would run un-fused in bf16). Input/output raw bf16 bytes.
func GeluBF16(x []byte) ([]byte, error) {
	out := make([]byte, len(x))
	if err := geluBF16Into(out, x, false); err != nil {
		return nil, err
	}
	return out, nil
}

func GeluBF16Into(out, x []byte) error {
	return geluBF16Into(out, x, true)
}

func geluBF16Into(out, x []byte, directOutput bool) error {
	if err := ensureInit(); err != nil {
		return err
	}
	// Validate byte alignment before the empty short-circuit, matching the
	// per-primitive path (an odd length reached MulBF16's check in the old
	// composition, so an odd 1-byte input must still error rather than return).
	if len(x)%bf16Size != 0 {
		return core.NewError("native.GeluBF16: byte length must be a multiple of 2")
	}
	if len(out) != len(x) {
		return core.NewError("native.GeluBF16Into: out must be the same byte length as x")
	}
	n := len(x) / bf16Size
	if n == 0 {
		return nil
	}
	var encErr error
	withAutoreleasePool(func() {
		scratch, err := getQMVBF16Scratch(n, n)
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
		if directOutput {
			tmp, ok := scratch.outputView(out)
			if ok {
				outBuf = tmp
				directOut = true
			}
		}

		cb := commandBufferFast(queue)
		enc := computeCommandEncoderFast(cb)
		if encErr = encGeluBF16Composed(enc, xBuf, outBuf, n); encErr != nil {
			endEncodingFast(enc)
			return
		}
		endEncodingFast(enc)
		commitCommandBufferFast(cb)
		waitUntilCompletedFast(cb)
		if !directOut {
			copy(out, scratch.out.bytes[:len(out)])
		}
	})
	return encErr
}

func geluGateMulComposed(gate, up []byte, n int) ([]byte, error) {
	out := make([]byte, n*bf16Size)
	if err := geluGateMulComposedInto(out, gate, up, n, false); err != nil {
		return nil, err
	}
	return out, nil
}

func geluGateMulComposedInto(out, gate, up []byte, n int, directOutput bool) error {
	var encErr error
	withAutoreleasePool(func() {
		ioScratch, err := getBinaryByteScratch(n * bf16Size)
		if err != nil {
			encErr = err
			return
		}
		defer putBinaryByteScratch(ioScratch)
		gBuf, uBuf, outBuf, err := ioScratch.buffers(gate, up)
		if err != nil {
			encErr = err
			return
		}
		directOut := false
		if directOutput {
			tmp, ok := ioScratch.outputView(out)
			if ok {
				outBuf = tmp
				directOut = true
			}
		}
		gelu := scratchBF16(n)

		cb := commandBufferFast(queue)
		enc := computeCommandEncoderFast(cb)
		if encErr = encGeluBF16Composed(enc, gBuf, gelu, n); encErr != nil {
			endEncodingFast(enc)
			return
		}
		if encErr = encMulBF16(enc, gelu, uBuf, outBuf, n); encErr != nil {
			endEncodingFast(enc)
			return
		}
		endEncodingFast(enc)
		commitCommandBufferFast(cb)
		waitUntilCompletedFast(cb)
		if !directOut {
			copy(out, ioScratch.out.bytes[:len(out)])
		}
	})
	return encErr
}

// SiLUGateMulBF16 computes silu(gate)·up in bf16 — the SwiGLU gate in the decode
// dtype, through the same encoder choke point the decode uses (fused fp32-internal
// kernel when loaded, composed bf16 chain otherwise) so host references stay
// byte-identical to the device path.
func SiLUGateMulBF16(gate, up []byte) ([]byte, error) {
	if len(up) != len(gate) {
		return nil, core.NewError("native.SiLUGateMulBF16: gate/up length mismatch")
	}
	if len(gate)%bf16Size != 0 {
		return nil, core.NewError("native.SiLUGateMulBF16: byte length must be a multiple of 2")
	}
	if err := ensureInit(); err != nil {
		return nil, err
	}
	out := make([]byte, len(gate))
	if len(gate) == 0 {
		return out, nil
	}
	if err := siluGateMulFusedInto(out, gate, up, len(gate)/bf16Size, false); err != nil {
		return nil, err
	}
	return out, nil
}

// GeluGateMulBF16 computes gelu(gate)·up in bf16 — gemma's MLP gate in the decode
// dtype. Uses the fused kernel (fp32-internal, one dispatch) when the custom kernels
// metallib is loaded, else the composed bf16 primitive chain. Parity in parity_test.go.
func GeluGateMulBF16(gate, up []byte) ([]byte, error) {
	out := make([]byte, len(gate))
	if err := geluGateMulBF16Into(out, gate, up, false); err != nil {
		return nil, err
	}
	return out, nil
}

func GeluGateMulBF16Into(out, gate, up []byte) error {
	return geluGateMulBF16Into(out, gate, up, true)
}

func geluGateMulBF16Into(out, gate, up []byte, directOutput bool) error {
	if err := ensureInit(); err != nil {
		return err
	}
	if len(up) != len(gate) {
		return core.NewError("native.GeluGateMulBF16: gate/up length mismatch")
	}
	if len(gate)%bf16Size != 0 {
		return core.NewError("native.GeluGateMulBF16: byte length must be a multiple of 2")
	}
	if len(out) != len(gate) {
		return core.NewError("native.GeluGateMulBF16Into: out must be the same byte length as gate")
	}
	if len(gate) == 0 {
		return nil
	}
	n := len(gate) / bf16Size
	if gpuHasGeluKernel() {
		return geluGateMulFusedInto(out, gate, up, n, directOutput)
	}
	return geluGateMulComposedInto(out, gate, up, n, directOutput)
}
