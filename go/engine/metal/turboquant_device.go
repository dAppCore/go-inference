// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"sync"
	"unsafe"

	core "dappco.re/go"
	"github.com/tmc/apple/metal"
)

// turboquant_device.go — S2 device kernels for TurboQuant KV-cache quantisation (RFC #41,
// kernels/lthn_turboquant.metal): pipeline resolvers for the three instantiated bit widths, the
// encoder emitters a future S3 block/layer wiring reuses, and host round-trip wrappers the parity
// tests (turboquant_device_test.go) and any pre-integration benches drive. This slice proves the
// KERNELS only — it does not touch the decode loop, the session, or the paged-KV machinery (that is
// S3). This file is deliberately engine-neutral about WHERE Π/centroids come from: it takes plain
// []float32 buffers and never imports kv/turboquant (S1) itself — matching lthn_gated_delta.go's own
// layering, where the production file never imports its host oracle ("deltanet"); only the _test.go
// does. It is also unrelated to this package's existing turboquant_kv_payload.go, which decodes a
// DIFFERENT, pre-existing FWHT+uniform-quantiser KV snapshot format under the same "TurboQuant" name
// — that file is session/paged-KV machinery and out of scope here.

// tqThreadgroupCap is the fixed threadgroup width kernels/lthn_turboquant.metal's LTHN_TQ_CAP
// dispatches, independent of the row's actual dimension d.
const tqThreadgroupCap = 256

// tqGeometryOK is the shared bits/row-dimension bound both TurboQuant device kernels enforce: bits
// must be one of the instantiated widths (2, 3, or 4) and d must fit the kernels' fixed
// tqThreadgroupCap-wide threadgroup span.
func tqGeometryOK(bits, d int) bool {
	return (bits == 2 || bits == 3 || bits == 4) && d > 0 && d <= tqThreadgroupCap
}

// tqBytesPerRow returns the packed-index byte stride per row for (bits, d) — ceil(d*bits/8),
// matching kv/turboquant's packBits length exactly (see kv/turboquant.UnpackIndices, which the parity
// tests use to unpack this same layout host-side).
func tqBytesPerRow(bits, d int) int {
	return (d*bits + 7) / 8
}

// --- pipeline resolvers ---------------------------------------------------------------------------

var (
	tqRotateQuantB2Once sync.Once
	tqRotateQuantB2PSO  metal.MTLComputePipelineState
	tqRotateQuantB2Err  error
	tqRotateQuantB3Once sync.Once
	tqRotateQuantB3PSO  metal.MTLComputePipelineState
	tqRotateQuantB3Err  error
	tqRotateQuantB4Once sync.Once
	tqRotateQuantB4PSO  metal.MTLComputePipelineState
	tqRotateQuantB4Err  error

	tqDequantB2Once sync.Once
	tqDequantB2PSO  metal.MTLComputePipelineState
	tqDequantB2Err  error
	tqDequantB3Once sync.Once
	tqDequantB3PSO  metal.MTLComputePipelineState
	tqDequantB3Err  error
	tqDequantB4Once sync.Once
	tqDequantB4PSO  metal.MTLComputePipelineState
	tqDequantB4Err  error
)

// tqResolvePipeline is the shared NewFunctionWithName/NewComputePipelineStateWithFunctionError
// closure body every (bits, kernel) case below runs once via its own sync.Once.
func tqResolvePipeline(name string, pso *metal.MTLComputePipelineState, errOut *error) {
	if customLibrary == nil || customLibrary.GetID() == 0 {
		*errOut = core.NewError("native.tqResolvePipeline: custom library unavailable")
		return
	}
	fn := customLibrary.NewFunctionWithName(name)
	if fn == nil || fn.GetID() == 0 {
		*errOut = core.NewError("native.tqResolvePipeline: kernel " + name + " not found")
		return
	}
	*pso, *errOut = device.NewComputePipelineStateWithFunctionError(fn)
}

// tqRotateQuantPipeline resolves lthn_tq_rotate_quant_bN for bits ∈ {2,3,4} — the only instantiated
// widths (kernels/lthn_turboquant.metal).
func tqRotateQuantPipeline(bits int) (metal.MTLComputePipelineState, error) {
	switch bits {
	case 2:
		tqRotateQuantB2Once.Do(func() { tqResolvePipeline("lthn_tq_rotate_quant_b2", &tqRotateQuantB2PSO, &tqRotateQuantB2Err) })
		return tqRotateQuantB2PSO, tqRotateQuantB2Err
	case 3:
		tqRotateQuantB3Once.Do(func() { tqResolvePipeline("lthn_tq_rotate_quant_b3", &tqRotateQuantB3PSO, &tqRotateQuantB3Err) })
		return tqRotateQuantB3PSO, tqRotateQuantB3Err
	case 4:
		tqRotateQuantB4Once.Do(func() { tqResolvePipeline("lthn_tq_rotate_quant_b4", &tqRotateQuantB4PSO, &tqRotateQuantB4Err) })
		return tqRotateQuantB4PSO, tqRotateQuantB4Err
	default:
		return nil, core.NewError("native.tqRotateQuantPipeline: unsupported bit width (want 2, 3, or 4)")
	}
}

// tqDequantUnrotatePipeline resolves lthn_tq_dequant_unrotate_bN for bits ∈ {2,3,4}.
func tqDequantUnrotatePipeline(bits int) (metal.MTLComputePipelineState, error) {
	switch bits {
	case 2:
		tqDequantB2Once.Do(func() { tqResolvePipeline("lthn_tq_dequant_unrotate_b2", &tqDequantB2PSO, &tqDequantB2Err) })
		return tqDequantB2PSO, tqDequantB2Err
	case 3:
		tqDequantB3Once.Do(func() { tqResolvePipeline("lthn_tq_dequant_unrotate_b3", &tqDequantB3PSO, &tqDequantB3Err) })
		return tqDequantB3PSO, tqDequantB3Err
	case 4:
		tqDequantB4Once.Do(func() { tqResolvePipeline("lthn_tq_dequant_unrotate_b4", &tqDequantB4PSO, &tqDequantB4Err) })
		return tqDequantB4PSO, tqDequantB4Err
	default:
		return nil, core.NewError("native.tqDequantUnrotatePipeline: unsupported bit width (want 2, 3, or 4)")
	}
}

// tqRotateQuantUsable reports whether the device encoder serves this (bits, d). The customLibrary
// check runs FIRST so a pre-init caller cannot latch the sync.Once into a permanent nil-library
// failure (the #23 lesson — see gatedDeltaStepUsable in lthn_gated_delta.go).
func tqRotateQuantUsable(bits, d int) bool {
	if customLibrary == nil || customLibrary.GetID() == 0 {
		return false
	}
	if !tqGeometryOK(bits, d) {
		return false
	}
	pso, err := tqRotateQuantPipeline(bits)
	return err == nil && pso != nil && pso.GetID() != 0
}

// tqDequantUnrotateUsable is tqRotateQuantUsable's sibling for the decode kernel.
func tqDequantUnrotateUsable(bits, d int) bool {
	if customLibrary == nil || customLibrary.GetID() == 0 {
		return false
	}
	if !tqGeometryOK(bits, d) {
		return false
	}
	pso, err := tqDequantUnrotatePipeline(bits)
	return err == nil && pso != nil && pso.GetID() != 0
}

// --- encoder emitters ------------------------------------------------------------------------------

// encTQRotateQuant encodes lthn_tq_rotate_quant_bN over numRows rows of dimension d: x -> (gammaOut,
// packedOut). pi is [d,d] row-major (Π[i][j]=pi[i*d+j]), centroids is [1<<bits] ascending — both
// resident, caller-owned. The caller owns buffer residency and the surrounding command buffer; this
// is the S3-reusable core (mirrors encGatedDeltaStepF32's shape in lthn_gated_delta.go).
func encTQRotateQuant(
	enc metal.MTLComputeCommandEncoder,
	bits int,
	x, pi, centroids, gammaOut, packedOut metal.MTLBuffer,
	xOff, piOff, centroidsOff, gammaOff, packedOff uint,
	numRows, d int,
) error {
	if numRows <= 0 || !tqGeometryOK(bits, d) {
		return core.NewError("native.encTQRotateQuant: invalid geometry")
	}
	pso, err := tqRotateQuantPipeline(bits)
	if err != nil {
		return err
	}
	setPSO(enc, pso)
	setBuf(enc, x, xOff, 0)
	setBuf(enc, pi, piOff, 1)
	setBuf(enc, centroids, centroidsOff, 2)
	setBuf(enc, gammaOut, gammaOff, 3)
	setBuf(enc, packedOut, packedOff, 4)
	setEncInt32(enc, int32(d), 5)
	dispatchThreadgroups(enc,
		metal.MTLSize{Width: uint(numRows), Height: 1, Depth: 1},
		metal.MTLSize{Width: tqThreadgroupCap, Height: 1, Depth: 1},
	)
	return nil
}

// encTQDequantUnrotate encodes lthn_tq_dequant_unrotate_bN over numRows rows of dimension d:
// (packed, gamma) -> out. pi and centroids are the SAME resident buffers encTQRotateQuant used (pi
// read transposed in-kernel — no separate transpose buffer, exactly mirroring kv/turboquant's
// mulVec/mulVecT pair over one matrix store).
func encTQDequantUnrotate(
	enc metal.MTLComputeCommandEncoder,
	bits int,
	packed, pi, centroids, gamma, out metal.MTLBuffer,
	packedOff, piOff, centroidsOff, gammaOff, outOff uint,
	numRows, d int,
) error {
	if numRows <= 0 || !tqGeometryOK(bits, d) {
		return core.NewError("native.encTQDequantUnrotate: invalid geometry")
	}
	pso, err := tqDequantUnrotatePipeline(bits)
	if err != nil {
		return err
	}
	setPSO(enc, pso)
	setBuf(enc, packed, packedOff, 0)
	setBuf(enc, pi, piOff, 1)
	setBuf(enc, centroids, centroidsOff, 2)
	setBuf(enc, gamma, gammaOff, 3)
	setBuf(enc, out, outOff, 4)
	setEncInt32(enc, int32(d), 5)
	dispatchThreadgroups(enc,
		metal.MTLSize{Width: uint(numRows), Height: 1, Depth: 1},
		metal.MTLSize{Width: tqThreadgroupCap, Height: 1, Depth: 1},
	)
	return nil
}

// --- host round-trip wrappers -----------------------------------------------------------------------

// tqRotateQuantScratch is the pooled pinned staging for one (bits,numRows,d) rotate_quant shape —
// the test/bench round-trip only; a future S3 integration keeps its buffers resident instead
// (mirrors gatedDeltaStepScratch in lthn_gated_delta.go).
type tqRotateQuantScratch struct {
	x, gamma *pinnedNoCopyBytes
	packed   *pinnedNoCopyBytes
}

type tqRotateQuantKey struct{ bits, numRows, d int }

var tqRotateQuantPools sync.Map // tqRotateQuantKey -> *sync.Pool

func getTQRotateQuantScratch(key tqRotateQuantKey) (*tqRotateQuantScratch, error) {
	poolAny, ok := tqRotateQuantPools.Load(key)
	if !ok {
		poolAny, _ = tqRotateQuantPools.LoadOrStore(key, &sync.Pool{})
	}
	pool := poolAny.(*sync.Pool)
	if v := pool.Get(); v != nil {
		return v.(*tqRotateQuantScratch), nil
	}
	sc := &tqRotateQuantScratch{}
	var err error
	alloc := func(n int) *pinnedNoCopyBytes {
		if err != nil {
			return nil
		}
		var b *pinnedNoCopyBytes
		b, err = newPinnedNoCopyBytes(n)
		return b
	}
	sc.x = alloc(key.numRows * key.d * 4)
	sc.gamma = alloc(key.numRows * 4)
	sc.packed = alloc(key.numRows * tqBytesPerRow(key.bits, key.d))
	if err != nil {
		return nil, err
	}
	return sc, nil
}

func putTQRotateQuantScratch(key tqRotateQuantKey, sc *tqRotateQuantScratch) {
	if v, ok := tqRotateQuantPools.Load(key); ok {
		v.(*sync.Pool).Put(sc)
	}
}

// TurboQuantRotateQuantDevice runs the encode kernel over host slices — upload, one dispatch, wait,
// read back gamma and the packed indices. rows is [numRows*d] row-major. pi is [d,d] row-major and
// centroids is [1<<bits] ascending — the CALLER supplies both (e.g. narrowed from
// kv/turboquant.RotationMatrix/Centroids to f32); using the SAME values the S1 host reference used
// internally is what makes byte/index parity against it meaningful. This wrapper exists for the
// parity tests and pre-integration benches — a future S3 device block binds encTQRotateQuant directly
// with resident buffers and never round-trips.
func TurboQuantRotateQuantDevice(rows, pi, centroids []float32, bits, numRows, d int) ([]float32, []byte, error) {
	if err := ensureInit(); err != nil {
		return nil, nil, err
	}
	if !tqRotateQuantUsable(bits, d) {
		return nil, nil, core.NewError("native.TurboQuantRotateQuantDevice: geometry not servable by the device kernel")
	}
	if len(rows) != numRows*d || len(pi) != d*d || len(centroids) != 1<<uint(bits) {
		return nil, nil, core.NewError("native.TurboQuantRotateQuantDevice: size mismatch")
	}
	bytesPerRow := tqBytesPerRow(bits, d)
	key := tqRotateQuantKey{bits: bits, numRows: numRows, d: d}
	gamma := make([]float32, numRows)
	packed := make([]byte, numRows*bytesPerRow)
	var encErr error
	withAutoreleasePool(func() {
		sc, gerr := getTQRotateQuantScratch(key)
		if gerr != nil {
			encErr = gerr
			return
		}
		defer putTQRotateQuantScratch(key, sc)
		xBuf, cerr := sc.x.copyBuffer(float32Bytes(rows))
		if cerr != nil {
			encErr = cerr
			return
		}
		piBuf := residentFloat32(pi)
		centroidsBuf := residentFloat32(centroids)

		cb := commandBufferFast(queue)
		enc := computeCommandEncoderFast(cb)
		if err := encTQRotateQuant(enc, bits, xBuf, piBuf, centroidsBuf, sc.gamma.buf, sc.packed.buf,
			0, 0, 0, 0, 0, numRows, d); err != nil {
			endEncodingFast(enc)
			encErr = err
			return
		}
		endEncodingFast(enc)
		commitCommandBufferFast(cb)
		waitUntilCompletedFast(cb)
		copy(gamma, unsafe.Slice((*float32)(unsafe.Pointer(&sc.gamma.bytes[0])), numRows))
		copy(packed, sc.packed.bytes[:numRows*bytesPerRow])
	})
	if encErr != nil {
		return nil, nil, encErr
	}
	return gamma, packed, nil
}

// tqDequantScratch is the pooled pinned staging for one (bits,numRows,d) dequant_unrotate shape.
type tqDequantScratch struct {
	packed, gamma, out *pinnedNoCopyBytes
}

type tqDequantKey struct{ bits, numRows, d int }

var tqDequantPools sync.Map // tqDequantKey -> *sync.Pool

func getTQDequantScratch(key tqDequantKey) (*tqDequantScratch, error) {
	poolAny, ok := tqDequantPools.Load(key)
	if !ok {
		poolAny, _ = tqDequantPools.LoadOrStore(key, &sync.Pool{})
	}
	pool := poolAny.(*sync.Pool)
	if v := pool.Get(); v != nil {
		return v.(*tqDequantScratch), nil
	}
	sc := &tqDequantScratch{}
	var err error
	alloc := func(n int) *pinnedNoCopyBytes {
		if err != nil {
			return nil
		}
		var b *pinnedNoCopyBytes
		b, err = newPinnedNoCopyBytes(n)
		return b
	}
	sc.packed = alloc(key.numRows * tqBytesPerRow(key.bits, key.d))
	sc.gamma = alloc(key.numRows * 4)
	sc.out = alloc(key.numRows * key.d * 4)
	if err != nil {
		return nil, err
	}
	return sc, nil
}

func putTQDequantScratch(key tqDequantKey, sc *tqDequantScratch) {
	if v, ok := tqDequantPools.Load(key); ok {
		v.(*sync.Pool).Put(sc)
	}
}

// TurboQuantDequantUnrotateDevice runs the decode kernel over host slices — upload, one dispatch,
// wait, read back the reconstructed rows [numRows*d]. packed is [numRows*bytesPerRow]
// (tqBytesPerRow(bits,d)), gamma is [numRows]; pi/centroids follow the same caller-supplies-the-
// values contract as TurboQuantRotateQuantDevice (and are typically the exact arrays that call used).
func TurboQuantDequantUnrotateDevice(packed []byte, pi, centroids, gamma []float32, bits, numRows, d int) ([]float32, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	if !tqDequantUnrotateUsable(bits, d) {
		return nil, core.NewError("native.TurboQuantDequantUnrotateDevice: geometry not servable by the device kernel")
	}
	bytesPerRow := tqBytesPerRow(bits, d)
	if len(packed) != numRows*bytesPerRow || len(pi) != d*d || len(centroids) != 1<<uint(bits) || len(gamma) != numRows {
		return nil, core.NewError("native.TurboQuantDequantUnrotateDevice: size mismatch")
	}
	key := tqDequantKey{bits: bits, numRows: numRows, d: d}
	rows := make([]float32, numRows*d)
	var encErr error
	withAutoreleasePool(func() {
		sc, gerr := getTQDequantScratch(key)
		if gerr != nil {
			encErr = gerr
			return
		}
		defer putTQDequantScratch(key, sc)
		up := func(dst *pinnedNoCopyBytes, src []byte) metal.MTLBuffer {
			if encErr != nil {
				return nil
			}
			buf, cerr := dst.copyBuffer(src)
			if cerr != nil {
				encErr = cerr
			}
			return buf
		}
		packedBuf := up(sc.packed, packed)
		gammaBuf := up(sc.gamma, float32Bytes(gamma))
		if encErr != nil {
			return
		}
		piBuf := residentFloat32(pi)
		centroidsBuf := residentFloat32(centroids)

		cb := commandBufferFast(queue)
		enc := computeCommandEncoderFast(cb)
		if err := encTQDequantUnrotate(enc, bits, packedBuf, piBuf, centroidsBuf, gammaBuf, sc.out.buf,
			0, 0, 0, 0, 0, numRows, d); err != nil {
			endEncodingFast(enc)
			encErr = err
			return
		}
		endEncodingFast(enc)
		commitCommandBufferFast(cb)
		waitUntilCompletedFast(cb)
		copy(rows, unsafe.Slice((*float32)(unsafe.Pointer(&sc.out.bytes[0])), numRows*d))
	})
	if encErr != nil {
		return nil, encErr
	}
	return rows, nil
}
