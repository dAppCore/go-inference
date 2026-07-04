// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"math"
	"sync"
	"unsafe"

	core "dappco.re/go"
	"github.com/tmc/apple/metal"
)

// ropePSOCache memoises rope pipelines keyed by name + the function-constant
// combination (forward/traditional/transpose), since those specialise the
// kernel at build time and a name alone doesn't identify the variant.
var (
	ropePSOMu    sync.Mutex
	ropePSOCache = map[string]metal.MTLComputePipelineState{}
)

const (
	ropeSingleFloat32Key            = "rope_single_float32|trad=false"
	ropeSingleFloat32TraditionalKey = "rope_single_float32|trad=true"
)

func ropePipelineKey(name string, traditional bool) string {
	if name == "rope_single_float32" {
		if traditional {
			return ropeSingleFloat32TraditionalKey
		}
		return ropeSingleFloat32Key
	}
	if traditional {
		return name + "|trad=true"
	}
	return name + "|trad=false"
}

// ropePipeline builds (and caches) a rope kernel specialised by MLX's function
// constants: forward (id 1), traditional (id 2), head_seq_transpose (id 3) —
// set at pipeline-build time via MTLFunctionConstantValues, not as buffers. This
// is the first native kernel to use function constants; the plumbing is reusable.
func ropePipeline(name string, traditional bool) (metal.MTLComputePipelineState, error) {
	key := ropePipelineKey(name, traditional)
	ropePSOMu.Lock()
	defer ropePSOMu.Unlock()
	if pso, ok := ropePSOCache[key]; ok {
		return pso, nil
	}
	if library == nil || library.GetID() == 0 {
		return nil, core.NewError("native.ropePipeline: library unavailable for " + name)
	}
	fc := metal.NewMTLFunctionConstantValues()
	fwd, trad, transpose := uint8(1), uint8(0), uint8(0) // forward, !traditional, !transpose
	if traditional {
		trad = 1
	}
	fc.SetConstantValueTypeAtIndex(unsafe.Pointer(&fwd), metal.MTLDataTypeBool, 1)
	fc.SetConstantValueTypeAtIndex(unsafe.Pointer(&trad), metal.MTLDataTypeBool, 2)
	fc.SetConstantValueTypeAtIndex(unsafe.Pointer(&transpose), metal.MTLDataTypeBool, 3)

	fn, err := library.NewFunctionWithNameConstantValuesError(name, fc)
	if err != nil {
		return nil, core.E("native.ropePipeline", name, err)
	}
	if fn == nil || fn.GetID() == 0 {
		return nil, core.NewError("native.ropePipeline: kernel " + name + " not found")
	}
	pso, err := device.NewComputePipelineStateWithFunctionError(fn)
	if err != nil {
		return nil, core.E("native.ropePipeline", "pipeline "+name, err)
	}
	ropePSOCache[key] = pso
	return pso, nil
}

// RoPE applies rotary position embedding for the single-token (decode) case: x
// is row-major (b, nHeads, 1, headDim), offset is the absolute position, and the
// full headDim is rotated. It drives MLX's rope_single kernel directly (no cgo):
// in(0) out(1) offset(2) scale(3) out_strides[0](4) base(10), with
// forward/traditional/transpose supplied as function constants and base passed
// pre-logged (log2) exactly as MLX does. float32. Byte-for-byte parity with
// pkg/metal.RoPE is gated in parity_test.go.
func RoPE(x []float32, b, nHeads, headDim int, base, scale float32, offset int, traditional bool) ([]float32, error) {
	out := make([]float32, len(x))
	if err := ropeInto(out, x, b, nHeads, headDim, base, scale, offset, traditional, false); err != nil {
		return nil, err
	}
	return out, nil
}

// RoPEInto is RoPE with caller-owned output storage when cap(out) is large enough.
func RoPEInto(out, x []float32, b, nHeads, headDim int, base, scale float32, offset int, traditional bool) ([]float32, error) {
	outLen := len(x)
	callerOut := cap(out) >= outLen
	if !callerOut {
		out = make([]float32, outLen)
	} else {
		out = out[:outLen]
	}
	if err := ropeInto(out, x, b, nHeads, headDim, base, scale, offset, traditional, callerOut); err != nil {
		return nil, err
	}
	return out, nil
}

func ropeInto(out, x []float32, b, nHeads, headDim int, base, scale float32, offset int, traditional, directOutput bool) error {
	if err := ensureInit(); err != nil {
		return err
	}
	if len(x) != b*nHeads*headDim {
		return core.NewError("native.RoPE: len(x) must equal b*nHeads*headDim (T=1)")
	}
	if len(out) != len(x) {
		return core.NewError("native.RoPE: len(out) must equal len(x)")
	}
	if headDim == 0 || nHeads == 0 || b == 0 {
		return nil
	}

	pso, err := ropePipeline("rope_single_float32", traditional)
	if err != nil {
		return err
	}

	var encErr error
	withAutoreleasePool(func() {
		ioScratch, err := getQMVFloatScratch(len(x), len(x))
		if err != nil {
			encErr = err
			return
		}
		defer putQMVFloatScratch(ioScratch)
		xBuf, outBuf, err := ioScratch.buffers(x)
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
		offBuf := scalarI32(int32(offset))
		logBase := float32(math.Log2(float64(base)))

		cb := commandBufferFast(queue)
		enc := computeCommandEncoderFast(cb)
		emitRopeAt(encSink{enc}, pso, xBuf, outBuf, 0, 0, offBuf, 0, nil, nHeads, headDim, headDim, scale, logBase)
		endEncodingFast(enc)
		commitCommandBufferFast(cb)
		waitUntilCompletedFast(cb)

		if !directOut {
			copy(float32Bytes(out), ioScratch.out.bytes[:len(x)*4])
		}
	})
	if encErr != nil {
		return encErr
	}
	return nil
}
