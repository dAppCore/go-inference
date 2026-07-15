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

// rope_freqs.go applies rotary position embedding with EXPLICIT per-dimension
// inverse frequencies — the YaRN long-context spectrum the arch carries in
// RopeFreqs — instead of a single uniform base. It drives MLX's
// rope_single_freqs_bfloat16 kernel, the freqs sibling of rope_single_bfloat16:
// identical buffer ABI except buffer(10) is a per-dim frequency array (not the
// log2 base) and buffer(11) its stride. The kernel reads inv_freq = 1/freqs[d],
// so the buffer holds the reciprocal of the inverse frequencies (the periods);
// RoPEFreqsBF16 inverts the caller's inv-freqs into that form.

var (
	ropeFreqsPSOBF16Cache = map[string]metal.MTLComputePipelineState{}
	ropeFreqsPSOBF16Mu    sync.Mutex

	ropePeriodsBufCache = map[ropePeriodsKey][]ropePeriodsCacheEntry{}
	ropePeriodsBufMu    sync.Mutex

	rawRopePeriodsBufCache = map[ropePeriodsKey][]ropePeriodsCacheEntry{}
	rawRopePeriodsBufMu    sync.Mutex
)

const (
	ropeFreqsBF16Key            = "rope_single_freqs_bfloat16|trad=false"
	ropeFreqsBF16TraditionalKey = "rope_single_freqs_bfloat16|trad=true"
)

type ropePeriodsKey struct {
	n    int
	hash uint64
}

type ropePeriodsCacheEntry struct {
	bits []uint32
	buf  metal.MTLBuffer
}

// ropeFreqsPipelineBF16 builds (and caches) the rope_single_freqs_bfloat16 kernel,
// specialised by the same forward/traditional/transpose function constants as the
// base rope_single_bfloat16 pipeline (both call rope_single_impl).
func ropeFreqsPipelineBF16(traditional bool) (metal.MTLComputePipelineState, error) {
	key := ropeFreqsPipelineBF16Key(traditional)
	ropeFreqsPSOBF16Mu.Lock()
	defer ropeFreqsPSOBF16Mu.Unlock()
	if pso, ok := ropeFreqsPSOBF16Cache[key]; ok {
		return pso, nil
	}
	if library == nil || library.GetID() == 0 {
		return nil, core.NewError("native.ropeFreqsPipelineBF16: library unavailable")
	}
	fc := metal.NewMTLFunctionConstantValues()
	fwd, trad, transpose := uint8(1), uint8(0), uint8(0)
	if traditional {
		trad = 1
	}
	fc.SetConstantValueTypeAtIndex(unsafe.Pointer(&fwd), metal.MTLDataTypeBool, 1)
	fc.SetConstantValueTypeAtIndex(unsafe.Pointer(&trad), metal.MTLDataTypeBool, 2)
	fc.SetConstantValueTypeAtIndex(unsafe.Pointer(&transpose), metal.MTLDataTypeBool, 3)
	fn, err := library.NewFunctionWithNameConstantValuesError("rope_single_freqs_bfloat16", fc)
	if err != nil {
		return nil, core.E("native.ropeFreqsPipelineBF16", "rope_single_freqs_bfloat16", err)
	}
	if fn == nil || fn.GetID() == 0 {
		return nil, core.NewError("native.ropeFreqsPipelineBF16: kernel rope_single_freqs_bfloat16 not found")
	}
	pso, err := device.NewComputePipelineStateWithFunctionError(fn)
	if err != nil {
		return nil, core.E("native.ropeFreqsPipelineBF16", "pipeline rope_single_freqs_bfloat16", err)
	}
	ropeFreqsPSOBF16Cache[key] = pso
	return pso, nil
}

func ropeFreqsPipelineBF16Key(traditional bool) string {
	if traditional {
		return ropeFreqsBF16TraditionalKey
	}
	return ropeFreqsBF16Key
}

func ropePeriodsKeyFor(invFreqs []float32) ropePeriodsKey {
	const (
		offset64 = 1469598103934665603
		prime64  = 1099511628211
	)
	h := uint64(offset64)
	for _, f := range invFreqs {
		h ^= uint64(math.Float32bits(f))
		h *= prime64
	}
	return ropePeriodsKey{n: len(invFreqs), hash: h}
}

func sameFloat32Bits(invFreqs []float32, bits []uint32) bool {
	if len(invFreqs) != len(bits) {
		return false
	}
	for i, f := range invFreqs {
		if math.Float32bits(f) != bits[i] {
			return false
		}
	}
	return true
}

func cachedRopePeriodsBuffer(invFreqs []float32) metal.MTLBuffer {
	if len(invFreqs) == 0 {
		return nil
	}
	key := ropePeriodsKeyFor(invFreqs)
	ropePeriodsBufMu.Lock()
	for _, entry := range ropePeriodsBufCache[key] {
		if sameFloat32Bits(invFreqs, entry.bits) {
			buf := entry.buf
			ropePeriodsBufMu.Unlock()
			return buf
		}
	}
	ropePeriodsBufMu.Unlock()

	periods := make([]float32, len(invFreqs))
	bits := make([]uint32, len(invFreqs))
	for i, f := range invFreqs {
		bits[i] = math.Float32bits(f)
		periods[i] = 1.0 / f
	}

	ropePeriodsBufMu.Lock()
	for _, entry := range ropePeriodsBufCache[key] {
		if sameFloat32Bits(invFreqs, entry.bits) {
			existing := entry.buf
			ropePeriodsBufMu.Unlock()
			return existing
		}
	}
	buf := device.NewBufferWithBytesLengthOptions(unsafe.Pointer(&periods[0]), uint(len(periods)*4), metal.MTLResourceStorageModeShared)
	ropePeriodsBufCache[key] = append(ropePeriodsBufCache[key], ropePeriodsCacheEntry{bits: bits, buf: buf})
	ropePeriodsBufMu.Unlock()
	return buf
}

func cachedRawRopePeriodsBuffer(periods []float32) metal.MTLBuffer {
	if len(periods) == 0 {
		return nil
	}
	key := ropePeriodsKeyFor(periods)
	rawRopePeriodsBufMu.Lock()
	for _, entry := range rawRopePeriodsBufCache[key] {
		if sameFloat32Bits(periods, entry.bits) {
			buf := entry.buf
			rawRopePeriodsBufMu.Unlock()
			return buf
		}
	}
	rawRopePeriodsBufMu.Unlock()

	bits := make([]uint32, len(periods))
	for i, f := range periods {
		bits[i] = math.Float32bits(f)
	}

	rawRopePeriodsBufMu.Lock()
	for _, entry := range rawRopePeriodsBufCache[key] {
		if sameFloat32Bits(periods, entry.bits) {
			existing := entry.buf
			rawRopePeriodsBufMu.Unlock()
			return existing
		}
	}
	buf := device.NewBufferWithBytesLengthOptions(unsafe.Pointer(&periods[0]), uint(len(periods)*4), metal.MTLResourceStorageModeShared)
	rawRopePeriodsBufCache[key] = append(rawRopePeriodsBufCache[key], ropePeriodsCacheEntry{bits: bits, buf: buf})
	rawRopePeriodsBufMu.Unlock()
	return buf
}

// RoPEFreqsBF16 is the explicit-frequency sibling of RoPEDimsBF16: it applies
// rotary embedding to x (bf16 bytes, row-major (b, nHeads, 1, headDim)) at
// absolute position offset, rotating the first rotaryDim of each head with the
// per-dim inverse frequencies invFreqs (len rotaryDim/2 — the arch's RopeFreqs),
// the tail [rotaryDim:headDim] passing through. The kernel uses inv_freq =
// 1/freqs[d], so invFreqs is inverted into the periods it expects. When invFreqs
// is the plain-rope spectrum (base^(-2d/rotaryDim)) the result is identical to
// RoPEDimsBF16 with that base — gated in rope_freqs_test.go.
//
//	out, err := native.RoPEFreqsBF16(xBytes, 1, 8, 128, 128, yarnInvFreqs, 1, pos, false)
func RoPEFreqsBF16(x []byte, b, nHeads, headDim, rotaryDim int, invFreqs []float32, scale float32, offset int, traditional bool) ([]byte, error) {
	return RoPEFreqsBF16Into(nil, x, b, nHeads, headDim, rotaryDim, invFreqs, scale, offset, traditional)
}

func RoPEFreqsBF16Into(out []byte, x []byte, b, nHeads, headDim, rotaryDim int, invFreqs []float32, scale float32, offset int, traditional bool) ([]byte, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	if len(x) != b*nHeads*headDim*bf16Size {
		return nil, core.NewError("native.RoPEFreqsBF16: len(x) must equal b*nHeads*headDim*2 bytes (T=1)")
	}
	outLen := len(x)
	if headDim == 0 || nHeads == 0 || b == 0 {
		if cap(out) < outLen {
			return make([]byte, outLen), nil
		}
		return out[:outLen], nil
	}
	if rotaryDim <= 0 || rotaryDim > headDim || rotaryDim%2 != 0 {
		return nil, core.NewError("native.RoPEFreqsBF16: rotaryDim must be even and in (0, headDim]")
	}
	if len(invFreqs) != rotaryDim/2 {
		return nil, core.NewError("native.RoPEFreqsBF16: len(invFreqs) must equal rotaryDim/2")
	}

	pso, err := ropeFreqsPipelineBF16(traditional)
	if err != nil {
		return nil, err
	}

	callerOut := cap(out) >= outLen
	if !callerOut {
		out = make([]byte, outLen)
	} else {
		out = out[:outLen]
	}
	var encErr error
	withAutoreleasePool(func() {
		scratch, err := getQMVBF16Scratch(len(x)/bf16Size, len(x)/bf16Size)
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
		if rotaryDim < headDim {
			// partial: seed out with x so the non-rotated tail passes through.
			if directOut {
				copy(out, x)
			} else {
				copy(scratch.out.bytes[:outLen], x)
			}
		}
		offBuf := scalarI32(int32(offset))
		freqsBuf := cachedRopePeriodsBuffer(invFreqs)

		cb := commandBufferFast(queue)
		enc := computeCommandEncoderFast(cb)
		emitRopeAt(encSink{enc}, pso, xBuf, outBuf, 0, 0, offBuf, 0, freqsBuf, nHeads, rotaryDim, headDim, scale, 0)
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

// encRoPEFreqsBF16 encodes freqs-aware rotary embedding into an existing encoder —
// the explicit-frequency sibling of encRoPEBF16, for the decode executor's hot
// path. periods is the resident GPU buffer of 1/inv_freq values (the executor
// uploads it once from the arch's RopeFreqs); freqStride is its element stride.
func encRoPEFreqsBF16(enc metal.MTLComputeCommandEncoder, x, out, offBuf, periods metal.MTLBuffer, nHeads, headDim, rotaryDim int, scale float32) error {
	return encRoPEFreqsBF16To(enc, x, out, 0, 0, offBuf, periods, nHeads, headDim, rotaryDim, scale)
}

// encRoPEFreqsBF16To is encRoPEFreqsBF16 reading from inOff and writing at outOff
// BYTES — the freqs sibling of encRoPEBF16To, used to rope the new token's K in
// place within the KV cache row. Same buffer ABI as encRoPEBF16To except buffer(10)
// is the periods array (not the log2 base) and buffer(11) its stride.
func encRoPEFreqsBF16To(enc metal.MTLComputeCommandEncoder, x, out metal.MTLBuffer, inOff, outOff uint, offBuf, periods metal.MTLBuffer, nHeads, headDim, rotaryDim int, scale float32) error {
	return encRoPEFreqsBF16ToAt(enc, x, out, inOff, outOff, offBuf, 0, periods, nHeads, headDim, rotaryDim, scale)
}

func encRoPEFreqsBF16ToAt(enc metal.MTLComputeCommandEncoder, x, out metal.MTLBuffer, inOff, outOff uint, offBuf metal.MTLBuffer, offOff uint, periods metal.MTLBuffer, nHeads, headDim, rotaryDim int, scale float32) error {
	pso, err := ropeFreqsPipelineBF16(false)
	if err != nil {
		return err
	}
	rd := headDim
	if rotaryDim > 0 && rotaryDim < headDim {
		rd = rotaryDim
	}
	// freqs partial-rotary RoPE through the SHARED emitRope body (with encRoPEBF16To + the ICB setRope);
	// periods != nil selects the freqs form (periods@10 + stride@11). log2base unused here.
	emitRopeAt(encSink{enc}, pso, x, out, inOff, outOff, offBuf, offOff, periods, nHeads, rd, headDim, scale, 0)
	return nil
}

// encRopeDecode is the decode hot-path rope dispatch: explicit-frequency rope when
// the layer carries a resident periods buffer (YaRN), else the base-derived rope.
// One branch point so encAttnHalfKV/encAttnHalfShared rope Q and K uniformly.
func encRopeDecode(enc metal.MTLComputeCommandEncoder, x, out metal.MTLBuffer, inOff, outOff uint, offBuf, ropeFreqs metal.MTLBuffer, nHeads, headDim, rotaryDim int, base, scale float32) error {
	return encRopeDecodeAt(enc, x, out, inOff, outOff, offBuf, 0, ropeFreqs, nHeads, headDim, rotaryDim, base, scale)
}

func encRopeDecodeAt(enc metal.MTLComputeCommandEncoder, x, out metal.MTLBuffer, inOff, outOff uint, offBuf metal.MTLBuffer, offOff uint, ropeFreqs metal.MTLBuffer, nHeads, headDim, rotaryDim int, base, scale float32) error {
	if ropeFreqs != nil {
		return encRoPEFreqsBF16ToAt(enc, x, out, inOff, outOff, offBuf, offOff, ropeFreqs, nHeads, headDim, rotaryDim, scale)
	}
	return encRoPEBF16ToAt(enc, x, out, inOff, outOff, offBuf, offOff, nHeads, headDim, rotaryDim, base, scale)
}

// uploadRopePeriods builds the resident periods buffer (1/inv_freq) for the
// freqs-rope hot path from the arch's RopeFreqs (inverse frequencies), or returns
// nil when there are none (the base-rope path). Retained for the session lifetime.
func uploadRopePeriods(invFreqs []float32) metal.MTLBuffer {
	return cachedRopePeriodsBuffer(invFreqs)
}

// proportionalRopePeriods builds the rope periods for a gemma4 proportional + partial-rotary
// layer (the global / full_attention layers), MATCHING metal's gemma4ProportionalFreqs: the first
// rotaryDim/2 entries are base^(2i/headDim) — the rope_type "proportional" scaling divides the
// exponent by the FULL head dim, NOT the rotated subset; the rest are +Inf (period → inv_freq 0 →
// no rotation); length headDim/2. base MUST be the RAW global rope_theta (1e6 on gemma4) — an
// arch-derived base is pre-folded to raw^(rotaryDim/headDim) for the base-derived kernel path and
// goes through globalRopePeriodsFromFolded instead. Feeding the folded base here lands every
// period at the 4th root of metal's (at 0.25 partial-rotary): exact at position 0, then an angle
// error growing linearly with position — the 12B cross-engine drift signature.
func proportionalRopePeriods(headDim, rotaryDim int, base float32) []float32 {
	half, rot := headDim/2, rotaryDim/2
	p := make([]float32, half)
	for i := range half {
		if i < rot {
			p[i] = float32(math.Pow(float64(base), float64(2*i)/float64(headDim)))
		} else {
			p[i] = float32(math.Inf(1))
		}
	}
	return p
}

// globalRopePeriodsFromFolded is proportionalRopePeriods for callers holding the ARCH-DERIVED
// base: arch.RopeBase is pre-folded to raw^(rotaryDim/headDim) (config.go folds it so the
// base-derived ÷rotaryDim rope kernels reproduce proportional rope), so the raw global theta is
// recovered by the inverse power before building the spectrum. The two bases coincide only at
// full rotary, where the fold is the identity.
func globalRopePeriodsFromFolded(headDim, rotaryDim int, foldedBase float32) []float32 {
	rawBase := float32(math.Pow(float64(foldedBase), float64(headDim)/float64(rotaryDim)))
	return proportionalRopePeriods(headDim, rotaryDim, rawBase)
}
