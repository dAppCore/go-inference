// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"os"
	"sync"
	"unsafe"

	core "dappco.re/go"
	"github.com/tmc/apple/metal"
)

// train_xent_device.go moves the SFT step's softmax cross-entropy FORWARD + BACKWARD onto the GPU —
// the [rows × vocab] pass that on E2B bf16 (vocab 262144) was ~91% of the step wall once the batched
// capture forward (#386) took the frozen forward off the serial walk. The host reference
// (CrossEntropyBackwardF32, train_optim.go) stays the oracle: it is a serial per-row walk of ~2·vocab
// transcendentals, exact in f64 but single-goroutine; the device kernel (lthn_softmax_xent_rows_f32)
// runs one threadgroup per row over the same maths in f32, tree-reducing the vocab sum. The two agree
// to ~1e-6 relative on the loss and ~1e-6 absolute on the gradient (train_xent_device_test.go), inside
// the 1e-5 loss-trajectory bar #390 sets. CrossEntropyBackwardF32Auto is the drop-in the trainer calls:
// device when the kernel is loadable and the gate allows it, host otherwise.

// trainGPUCrossEntropyDisabled reports the kill-switch: LTHN_TRAIN_GPU_CE=0 pins the SFT loss/grad back
// onto the host CrossEntropyBackwardF32 without a rebuild — the host path is the reference oracle, so
// this is the live A/B lever the orchestrator flips to separate a numeric shift from a real fault.
func trainGPUCrossEntropyDisabled() bool { return os.Getenv("LTHN_TRAIN_GPU_CE") == "0" }

var (
	softmaxXentPSOMu   sync.Mutex
	softmaxXentPSO     metal.MTLComputePipelineState
	softmaxXentErr     error
	softmaxXentOnce    sync.Once
	softmaxXentThreads = 1024 // one threadgroup per row, striding the vocab
)

// softmaxXentPipeline loads lthn_softmax_xent_rows_f32 from the sibling custom metallib. Cached
// forever; an absent kernel reports an error and the caller falls back to the host reference.
func softmaxXentPipeline() (metal.MTLComputePipelineState, error) {
	softmaxXentOnce.Do(func() {
		if customLibrary == nil || customLibrary.GetID() == 0 {
			softmaxXentErr = core.NewError("native.softmaxXentPipeline: custom library unavailable")
			return
		}
		fn := customLibrary.NewFunctionWithName("lthn_softmax_xent_rows_f32")
		if fn == nil || fn.GetID() == 0 {
			softmaxXentErr = core.NewError("native.softmaxXentPipeline: kernel lthn_softmax_xent_rows_f32 not found — rebuild lthn_kernels.metallib")
			return
		}
		softmaxXentPSO, softmaxXentErr = device.NewComputePipelineStateWithFunctionError(fn)
	})
	softmaxXentPSOMu.Lock()
	defer softmaxXentPSOMu.Unlock()
	return softmaxXentPSO, softmaxXentErr
}

// gpuHasSoftmaxXent reports whether the fused softmax-xent kernel is buildable on this device.
func gpuHasSoftmaxXent() bool {
	_, err := softmaxXentPipeline()
	return err == nil
}

// softmaxXentScratch holds the pinned no-copy device buffers for one (rows, vocab) shape: logits in,
// targets in, dLogits out, rowLoss out. Pooled per shape so a retained trainer reuses them each step.
type softmaxXentScratch struct {
	logits, targets, dLogits, rowLoss *pinnedNoCopyBytes
}

type softmaxXentKey struct{ rows, vocab int }

var softmaxXentPools sync.Map // softmaxXentKey -> *sync.Pool

func getSoftmaxXentScratch(rows, vocab int) (*softmaxXentScratch, error) {
	key := softmaxXentKey{rows, vocab}
	poolAny, ok := softmaxXentPools.Load(key)
	if !ok {
		poolAny, _ = softmaxXentPools.LoadOrStore(key, &sync.Pool{})
	}
	pool := poolAny.(*sync.Pool)
	if v := pool.Get(); v != nil {
		return v.(*softmaxXentScratch), nil
	}
	sc := &softmaxXentScratch{}
	var err error
	alloc := func(n int) *pinnedNoCopyBytes {
		if err != nil {
			return nil
		}
		var b *pinnedNoCopyBytes
		b, err = newPinnedNoCopyBytes(n)
		return b
	}
	sc.logits = alloc(rows * vocab * 4)
	sc.targets = alloc(rows * 4)
	sc.dLogits = alloc(rows * vocab * 4)
	sc.rowLoss = alloc(rows * 4)
	if err != nil {
		return nil, err
	}
	return sc, nil
}

func putSoftmaxXentScratch(rows, vocab int, sc *softmaxXentScratch) {
	if v, ok := softmaxXentPools.Load(softmaxXentKey{rows, vocab}); ok {
		v.(*sync.Pool).Put(sc)
	}
}

// crossEntropyBackwardF32Device computes the mean softmax cross-entropy and its gradient on the GPU —
// the device twin of CrossEntropyBackwardF32, returning the SAME (meanLoss, dLogits[rows,vocab]). The
// per-row NLL comes back from the kernel raw; the mean is taken here in a float64 accumulator, matching
// the host reference's reduction shape exactly (lossSum f64, then ×inv).
func crossEntropyBackwardF32Device(logits []float32, targets []int32, rows, vocab int) (float32, []float32, error) {
	if len(logits) != rows*vocab || len(targets) != rows {
		return 0, nil, core.NewError("native.crossEntropyBackwardF32Device: logits must be [rows,vocab] and targets [rows]")
	}
	if rows <= 0 || vocab <= 0 {
		return 0, nil, core.NewError("native.crossEntropyBackwardF32Device: rows and vocab must be positive")
	}
	for _, t := range targets {
		if t < 0 || int(t) >= vocab {
			return 0, nil, core.NewError("native.crossEntropyBackwardF32Device: target out of range")
		}
	}
	pso, err := softmaxXentPipeline()
	if err != nil {
		return 0, nil, err
	}

	dLogits := make([]float32, rows*vocab)
	rowLoss := make([]float32, rows)
	invRows := float32(1.0 / float64(rows))
	var encErr error
	withAutoreleasePool(func() {
		sc, gerr := getSoftmaxXentScratch(rows, vocab)
		if gerr != nil {
			encErr = gerr
			return
		}
		defer putSoftmaxXentScratch(rows, vocab, sc)
		logitsBuf, cerr := sc.logits.copyBuffer(float32Bytes(logits))
		if cerr != nil {
			encErr = cerr
			return
		}
		targetsBuf, cerr := sc.targets.copyBuffer(unsafe.Slice((*byte)(unsafe.Pointer(&targets[0])), rows*4))
		if cerr != nil {
			encErr = cerr
			return
		}

		cb := commandBufferFast(queue)
		enc := computeCommandEncoderFast(cb)
		sink := encSink{enc}
		sink.setPSO(pso)
		sink.setBuf(logitsBuf, 0, 0)
		sink.setBuf(targetsBuf, 0, 1)
		sink.setBuf(sc.dLogits.buf, 0, 2)
		sink.setBuf(sc.rowLoss.buf, 0, 3)
		sink.setI32(int32(vocab), 4)
		sink.setF32(invRows, 5)
		sink.dispatchThreadgroups(
			metal.MTLSize{Width: uint(rows), Height: 1, Depth: 1},
			metal.MTLSize{Width: uint(softmaxXentThreads), Height: 1, Depth: 1},
		)
		endEncodingFast(enc)
		commitCommandBufferFast(cb)
		waitUntilCompletedFast(cb)

		copy(dLogits, unsafe.Slice((*float32)(unsafe.Pointer(&sc.dLogits.bytes[0])), rows*vocab))
		copy(rowLoss, unsafe.Slice((*float32)(unsafe.Pointer(&sc.rowLoss.bytes[0])), rows))
	})
	if encErr != nil {
		return 0, nil, encErr
	}

	var lossSum float64
	for _, l := range rowLoss {
		lossSum += float64(l)
	}
	return float32(lossSum / float64(rows)), dLogits, nil
}

// CrossEntropyBackwardF32Auto returns the mean softmax cross-entropy loss and its gradient, using the
// GPU kernel when it is loadable and the LTHN_TRAIN_GPU_CE gate allows it, and the host reference
// (CrossEntropyBackwardF32) otherwise. The signature and outputs match the host oracle within the
// 1e-5 loss tolerance #390 sets; a checkout without the custom metallib transparently gets the host
// path, so the trainer is correct with or without the kernel built.
func CrossEntropyBackwardF32Auto(logits []float32, targets []int32, rows, vocab int) (float32, []float32, error) {
	if !trainGPUCrossEntropyDisabled() && gpuHasSoftmaxXent() {
		return crossEntropyBackwardF32Device(logits, targets, rows, vocab)
	}
	return CrossEntropyBackwardF32(logits, targets, rows, vocab)
}
