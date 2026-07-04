// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"sync"

	core "dappco.re/go"
	"github.com/tmc/apple/metal"
)

// psoCache memoises one compute pipeline state per kernel name — building a PSO
// compiles the kernel's pipeline, so it is paid once and reused across calls.
var (
	psoMu    sync.Mutex
	psoCache = map[string]metal.MTLComputePipelineState{}
)

// pipelineFor returns the cached compute pipeline state for a metallib kernel,
// building it on first use.
func pipelineFor(name string) (metal.MTLComputePipelineState, error) {
	psoMu.Lock()
	defer psoMu.Unlock()
	if pso, ok := psoCache[name]; ok {
		return pso, nil
	}
	if library == nil || library.GetID() == 0 {
		return nil, core.NewError("native.pipelineFor: library unavailable for " + name)
	}
	fn := library.NewFunctionWithName(name)
	if fn == nil || fn.GetID() == 0 {
		return nil, core.NewError("native.pipelineFor: kernel " + name + " not found")
	}
	pso, err := device.NewComputePipelineStateWithFunctionError(fn)
	if err != nil {
		return nil, core.E("native.pipelineFor", name, err)
	}
	psoCache[name] = pso
	return pso, nil
}

// RunUnary drives a contiguous unary MLX kernel over in and returns a fresh
// result slice. It targets the v_<Op><itype><otype> kernel family, whose host
// ABI (read from mlx/backend/metal/unary.cpp) is: input → buffer(0), output →
// buffer(1), element count as a uint → buffer(2), and one GPU thread per
// element. name is e.g. "v_Squarefloat32float32" (the float32→float32 square).
//
// Shared-storage buffers make the result host-visible with no blit; the call
// blocks until the GPU completes (commit + wait). It is the byte-for-byte
// equivalent of the mlx-c contiguous unary path — parity is gated in the tests.
func RunUnary(name string, in []float32) ([]float32, error) {
	out := make([]float32, len(in))
	if err := runUnaryInto(name, in, out, false); err != nil {
		return nil, err
	}
	return out, nil
}

// RunUnaryInto is RunUnary writing the result into the caller-supplied out
// (len(out) must equal len(in)) instead of allocating a fresh slice. Same GPU
// kernel and input as RunUnary — only the Go destination differs, so the bytes
// are identical. It lets a composed op reuse scratch buffers across its chain
// (e.g. the Tanh step inside Gelu) rather than allocating per primitive.
func RunUnaryInto(name string, in, out []float32) error {
	return runUnaryInto(name, in, out, true)
}

func runUnaryInto(name string, in, out []float32, directOutput bool) error {
	if err := ensureInit(); err != nil {
		return err
	}
	if len(out) != len(in) {
		return core.NewError("native.RunUnaryInto: out must be the same length as in")
	}
	pso, err := pipelineFor(name)
	if err != nil {
		return err
	}
	n := len(in)
	if n == 0 {
		return nil
	}

	var encErr error
	withAutoreleasePool(func() {
		ioScratch, err := getQMVFloatScratch(n, n)
		if err != nil {
			encErr = err
			return
		}
		defer putQMVFloatScratch(ioScratch)
		inBuf, outBuf, err := ioScratch.buffers(in)
		if err != nil {
			encErr = err
			return
		}
		directOut := false
		if directOutput {
			if tmp, ok := ioScratch.outputView(out); ok {
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
			copy(float32Bytes(out), ioScratch.out.bytes[:n*4])
		}
	})
	if encErr != nil {
		return encErr
	}
	return nil
}

// RunUnaryBF16 is the bfloat16 sibling of RunUnary: it drives a contiguous unary MLX kernel
// (v_<Op>bfloat16bfloat16) over raw bf16 bytes and returns the bf16 result — same host ABI as the
// float32 path (input→0, output→1, element-count→2, one thread/element), only the kernel-name dtype
// token and the 2-byte element width differ. Byte-for-byte parity with the matching pkg/metal unary
// op on the same bf16 array is the point — it is how the vision/audio towers stay byte-identical.
func RunUnaryBF16(name string, in []byte) ([]byte, error) {
	out := make([]byte, len(in))
	if err := runUnaryBF16Into(name, in, out, false); err != nil {
		return nil, err
	}
	return out, nil
}

func RunUnaryBF16Into(name string, in, out []byte) error {
	return runUnaryBF16Into(name, in, out, true)
}

func runUnaryBF16Into(name string, in, out []byte, directOutput bool) error {
	if err := ensureInit(); err != nil {
		return err
	}
	if len(in)%bf16Size != 0 {
		return core.NewError("native.RunUnaryBF16Into: byte length must be a multiple of 2 (bf16 elements)")
	}
	if len(out) != len(in) {
		return core.NewError("native.RunUnaryBF16Into: out must be the same byte length as in")
	}
	pso, err := pipelineFor(name)
	if err != nil {
		return err
	}
	n := len(in) / bf16Size
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
		inBuf, outBuf, err := scratch.buffers(in)
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
			copy(out, scratch.out.bytes[:len(in)])
		}
	})
	if encErr != nil {
		return encErr
	}
	return nil
}

// SigmoidBF16 is the byte-parity bf16 sigmoid (kernel v_Sigmoidbfloat16bfloat16) — equals
// pkg/metal.Sigmoid on the same bf16 array.
func SigmoidBF16(in []byte) ([]byte, error) { return RunUnaryBF16("v_Sigmoidbfloat16bfloat16", in) }

// SiLUBF16 is the byte-parity bf16 SiLU/swish: x·sigmoid(x), composed EXACTLY as pkg/metal.SiLU does
// (Mul(a, Sigmoid(a))) from byte-parity primitives, so the bytes match metal.SiLU.
func SiLUBF16(in []byte) ([]byte, error) {
	s, err := SigmoidBF16(in)
	if err != nil {
		return nil, err
	}
	return MulBF16(in, s)
}

// Square returns in[i]*in[i] for every element, computed on the GPU through the
// shared mlx.metallib (kernel v_Squarefloat32float32). Byte-for-byte parity
// with pkg/metal.Square is asserted in parity_test.go — this is the first op
// proving the no-cgo path drives a real MLX kernel identically to mlx-c.
func Square(in []float32) ([]float32, error) {
	return RunUnary("v_Squarefloat32float32", in)
}

// Abs returns the element-wise absolute value, computed on the GPU through the
// shared mlx.metallib (kernel v_Absfloat32float32). Parity with pkg/metal.Abs
// is gated in parity_test.go.
//
//	out, err := native.Abs([]float32{-2, 0, 3}) // out = [2 0 3]
func Abs(in []float32) ([]float32, error) {
	return RunUnary("v_Absfloat32float32", in)
}

// Negative returns the element-wise negation -in[i], computed on the GPU
// through the shared mlx.metallib (kernel v_Negativefloat32float32). Parity
// with pkg/metal.Negative is gated in parity_test.go.
//
//	out, err := native.Negative([]float32{-2, 0, 3}) // out = [2 0 -3]
func Negative(in []float32) ([]float32, error) {
	return RunUnary("v_Negativefloat32float32", in)
}

// Exp returns the element-wise exponential exp(in[i]), computed on the GPU
// through the shared mlx.metallib (kernel v_Expfloat32float32). Parity with
// pkg/metal.Exp is gated in parity_test.go.
//
//	out, err := native.Exp([]float32{0, 1}) // out = [1 2.7182817]
func Exp(in []float32) ([]float32, error) {
	return RunUnary("v_Expfloat32float32", in)
}

// Sigmoid returns the element-wise logistic 1/(1+exp(-in[i])), computed on the
// GPU through the shared mlx.metallib (kernel v_Sigmoidfloat32float32). Parity
// with pkg/metal.Sigmoid is gated in parity_test.go.
//
//	out, err := native.Sigmoid([]float32{0}) // out = [0.5]
func Sigmoid(in []float32) ([]float32, error) {
	return RunUnary("v_Sigmoidfloat32float32", in)
}

// Tanh returns the element-wise hyperbolic tangent tanh(in[i]), computed on the
// GPU through the shared mlx.metallib (kernel v_Tanhfloat32float32). Parity
// with pkg/metal.Tanh is gated in parity_test.go.
//
//	out, err := native.Tanh([]float32{0}) // out = [0]
func Tanh(in []float32) ([]float32, error) {
	return RunUnary("v_Tanhfloat32float32", in)
}

// Sqrt returns the element-wise square root sqrt(in[i]), computed on the GPU
// through the shared mlx.metallib (kernel v_Sqrtfloat32float32). Inputs must be
// non-negative; parity with pkg/metal.Sqrt is gated in parity_test.go.
//
//	out, err := native.Sqrt([]float32{4, 9}) // out = [2 3]
func Sqrt(in []float32) ([]float32, error) {
	return RunUnary("v_Sqrtfloat32float32", in)
}

// Rsqrt returns the element-wise reciprocal square root 1/sqrt(in[i]), computed
// on the GPU through the shared mlx.metallib (kernel v_Rsqrtfloat32float32).
// Inputs must be strictly positive; parity with pkg/metal.Rsqrt is gated in
// parity_test.go.
//
//	out, err := native.Rsqrt([]float32{4}) // out = [0.5]
func Rsqrt(in []float32) ([]float32, error) {
	return RunUnary("v_Rsqrtfloat32float32", in)
}

// Log returns the element-wise natural logarithm ln(in[i]), computed on the GPU
// through the shared mlx.metallib (kernel v_Logfloat32float32). Inputs must be
// strictly positive; parity with pkg/metal.Log is gated in parity_test.go.
//
//	out, err := native.Log([]float32{1}) // out = [0]
func Log(in []float32) ([]float32, error) {
	return RunUnary("v_Logfloat32float32", in)
}

// Round returns the element-wise round-to-nearest (ties to even) of in[i],
// computed on the GPU through the shared mlx.metallib (kernel
// v_Roundfloat32float32). Parity with pkg/metal.Round is gated in
// parity_test.go.
//
//	out, err := native.Round([]float32{0.5, 1.5, 2.4}) // out = [0 2 2]
func Round(in []float32) ([]float32, error) {
	return RunUnary("v_Roundfloat32float32", in)
}
