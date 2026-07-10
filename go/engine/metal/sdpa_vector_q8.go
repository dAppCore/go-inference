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

// sdpa_vector_q8.go — the linear-cache q8 decode SDPA pair (#367): int8 K/V +
// f32 group scales (kvQ8GroupSize=64) read by lthn ports of MLX's sdpa_vector
// and sdpa_vector_2pass_1 (pass 2 is the unchanged MLX merge — it never
// touches K/V). The dense family's GLOBAL layers (12B kv=1 hd=512, 31B kv=4
// hd=512, e2b/e4b kv=1 hd=512) carry the unbounded deep-context scan — the
// measured depth slope — and int8 halves its bytes. gqa is a runtime scalar,
// so one instantiation per head dim covers every dense model.

var (
	sdpaVectorQ8PSOMu    sync.Mutex
	sdpaVectorQ8PSOCache = map[int]metal.MTLComputePipelineState{}
	sdpaV2P1Q8PSOCache   = map[[2]int]metal.MTLComputePipelineState{}
)

func sdpaVectorQ8KernelName(headDim int) string {
	return core.Sprintf("lthn_sdpa_vector_q8_bf16_%d", headDim)
}

// sdpaVectorQ8Pipeline resolves (and caches) the single-pass q8 decode SDPA
// for a head dim (256/512 instantiated).
func sdpaVectorQ8Pipeline(headDim int) (metal.MTLComputePipelineState, error) {
	sdpaVectorQ8PSOMu.Lock()
	defer sdpaVectorQ8PSOMu.Unlock()
	if pso, ok := sdpaVectorQ8PSOCache[headDim]; ok {
		if pso == nil {
			return nil, core.NewError("native.sdpaVectorQ8Pipeline: kernel unavailable")
		}
		return pso, nil
	}
	if customLibrary == nil || customLibrary.GetID() == 0 {
		sdpaVectorQ8PSOCache[headDim] = nil
		return nil, core.NewError("native.sdpaVectorQ8Pipeline: custom library unavailable")
	}
	fn := customLibrary.NewFunctionWithName(sdpaVectorQ8KernelName(headDim))
	if fn == nil || fn.GetID() == 0 {
		sdpaVectorQ8PSOCache[headDim] = nil
		return nil, core.NewError("native.sdpaVectorQ8Pipeline: kernel " + sdpaVectorQ8KernelName(headDim) + " not found")
	}
	pso, err := device.NewComputePipelineStateWithFunctionError(fn)
	if err != nil {
		sdpaVectorQ8PSOCache[headDim] = nil
		return nil, err
	}
	sdpaVectorQ8PSOCache[headDim] = pso
	return pso, nil
}

// sdpaVector2Pass1Q8Pipeline resolves (and caches) the q8 pass-1 with the
// block count baked as function constant 26 (the MLX 2-pass convention — the
// kernel strides its key walk and indexes the partials by it).
func sdpaVector2Pass1Q8Pipeline(headDim int, blocks int32) (metal.MTLComputePipelineState, error) {
	key := [2]int{headDim, int(blocks)}
	sdpaVectorQ8PSOMu.Lock()
	defer sdpaVectorQ8PSOMu.Unlock()
	if pso, ok := sdpaV2P1Q8PSOCache[key]; ok {
		if pso == nil {
			return nil, core.NewError("native.sdpaVector2Pass1Q8Pipeline: kernel unavailable")
		}
		return pso, nil
	}
	if customLibrary == nil || customLibrary.GetID() == 0 {
		sdpaV2P1Q8PSOCache[key] = nil
		return nil, core.NewError("native.sdpaVector2Pass1Q8Pipeline: custom library unavailable")
	}
	fc := metal.NewMTLFunctionConstantValues()
	blk := blocks
	fc.SetConstantValueTypeAtIndex(unsafe.Pointer(&blk), metal.MTLDataTypeInt, 26)
	name := core.Sprintf("lthn_sdpa_vector_2pass_1_q8_bf16_%d", headDim)
	fn, err := customLibrary.NewFunctionWithNameConstantValuesError(name, fc)
	if err != nil || fn == nil || fn.GetID() == 0 {
		sdpaV2P1Q8PSOCache[key] = nil
		return nil, core.E("native.sdpaVector2Pass1Q8Pipeline", name, err)
	}
	pso, perr := device.NewComputePipelineStateWithFunctionError(fn)
	if perr != nil {
		sdpaV2P1Q8PSOCache[key] = nil
		return nil, perr
	}
	sdpaV2P1Q8PSOCache[key] = pso
	return pso, nil
}

// emitSDPAVectorQ8 records the single-pass q8 decode SDPA through any sink:
// q=0(bf16), k=1(int8), v=2(int8), out=3, gqa=4, N=5 (nBuf when non-nil — the
// ICB's rebindable length), strides(elements)=6..9, scale=10, kScales=11,
// vScales=12. One threadgroup per q head, 32×32 threads — the MLX
// sdpa_vector dispatch. Strides must be multiples of kvQ8GroupSize (the
// kernel derives the scale-plane strides as stride/64).
func emitSDPAVectorQ8[S dispatchSink](sink S, pso metal.MTLComputePipelineState, q, k, v, out metal.MTLBuffer, kScales, vScales metal.MTLBuffer, kvByteOff, scaleByteOff uint, nBuf metal.MTLBuffer, nHeads, nKVHeads, n int, kHeadStride, kSeqStride, vHeadStride, vSeqStride int64, scale float32) {
	sink.setPSO(pso)
	sink.setBuf(q, 0, 0)
	sink.setBuf(k, kvByteOff, 1)
	sink.setBuf(v, kvByteOff, 2)
	sink.setBuf(out, 0, 3)
	sink.setI32(int32(nHeads/nKVHeads), 4)
	if nBuf != nil {
		sink.setBuf(nBuf, 0, 5)
	} else {
		sink.setI32(int32(n), 5)
	}
	sink.setI64(kHeadStride, 6)
	sink.setI64(kSeqStride, 7)
	sink.setI64(vHeadStride, 8)
	sink.setI64(vSeqStride, 9)
	sink.setF32(scale, 10)
	sink.setBuf(kScales, scaleByteOff, 11)
	sink.setBuf(vScales, scaleByteOff, 12)
	sink.dispatchThreadgroups(metal.MTLSize{Width: uint(nHeads), Height: 1, Depth: 1}, metal.MTLSize{Width: 1024, Height: 1, Depth: 1})
}

// emitSDPAVector2Pass1Q8 records the q8 pass-1 through any sink — the MLX
// 2-pass ABI with the scale planes appended (N at 7 as a buffer or inline;
// strides 8..11; scale 12; kScales 13; vScales 14) and the decode grid
// (nKVHeads, 1, blocks) of (32, gqa, 1). Pass 2 is MLX's sdpa_vector_2pass_2
// unchanged.
func emitSDPAVector2Pass1Q8[S dispatchSink](sink S, pso metal.MTLComputePipelineState, q, k, v, partials metal.MTLBuffer, sums, maxs metal.MTLBuffer, kScales, vScales metal.MTLBuffer, kvByteOff, scaleByteOff uint, nBuf metal.MTLBuffer, nHeads, nKVHeads, n, blocks int, kHeadStride, kSeqStride, vHeadStride, vSeqStride int64, scale float32) {
	sink.setPSO(pso)
	sink.setBuf(q, 0, 0)
	sink.setBuf(k, kvByteOff, 1)
	sink.setBuf(v, kvByteOff, 2)
	sink.setBuf(partials, 0, 3)
	sink.setBuf(sums, 0, 4)
	sink.setBuf(maxs, 0, 5)
	if nBuf != nil {
		sink.setBuf(nBuf, 0, 7)
	} else {
		sink.setI32(int32(n), 7)
	}
	sink.setI64(kHeadStride, 8)
	sink.setI64(kSeqStride, 9)
	sink.setI64(vHeadStride, 10)
	sink.setI64(vSeqStride, 11)
	sink.setF32(scale, 12)
	sink.setBuf(kScales, scaleByteOff, 13)
	sink.setBuf(vScales, scaleByteOff, 14)
	sink.dispatchThreadgroups(
		metal.MTLSize{Width: uint(nKVHeads), Height: 1, Depth: uint(blocks)},
		metal.MTLSize{Width: 32, Height: uint(nHeads / nKVHeads), Depth: 1},
	)
}

// kvQ8QuantiseRows quantises `rows` contiguous bf16 KV rows (kvDim elements
// each) into int8 + f32 group scales HOST-SIDE — the test/reference twin of
// the lthn_kv_q8_store_bf16 kernel's math (symmetric per-64 group,
// scale = max|group|/127, round-to-nearest-even like rint).
func kvQ8QuantiseRows(rows []byte, kvDim int) (codes []byte, scales []float32) {
	n := len(rows) / 2
	codes = make([]byte, n)
	scales = make([]float32, n/kvQ8GroupSize)
	for g := 0; g < n/kvQ8GroupSize; g++ {
		base := g * kvQ8GroupSize
		var m float32
		for i := 0; i < kvQ8GroupSize; i++ {
			f := bf16ToF32(rows[(base+i)*2], rows[(base+i)*2+1])
			if a := absF32(f); a > m {
				m = a
			}
		}
		// Metal fast-math compiles /127.0f as a reciprocal multiply — match it
		// exactly so the kernel and this host twin agree bit-for-bit.
		scale := m * (1.0 / 127.0)
		inv := float32(0)
		if scale > 0 {
			inv = 1 / scale
		}
		scales[g] = scale
		for i := 0; i < kvQ8GroupSize; i++ {
			f := bf16ToF32(rows[(base+i)*2], rows[(base+i)*2+1])
			q := rintF32(f * inv)
			if q > 127 {
				q = 127
			} else if q < -127 {
				q = -127
			}
			codes[base+i] = byte(int8(q))
		}
	}
	return codes, scales
}

// rintF32 rounds half-to-even, matching Metal's rint.
func rintF32(f float32) float32 {
	fl := float32(int32(f))
	if f < 0 && fl > f {
		fl--
	}
	d := f - fl
	switch {
	case d > 0.5:
		return fl + 1
	case d < 0.5:
		return fl
	default: // exactly .5 — to even
		if int32(fl)%2 == 0 {
			return fl
		}
		return fl + 1
	}
}

// kvQ8DequantiseRows expands int8 codes + scales back to bf16 rows — the
// reference-cache side of the parity method (both kernels then see the same
// VALUES; only the arithmetic path differs).
func kvQ8DequantiseRows(codes []byte, scales []float32) []byte {
	out := make([]byte, len(codes)*2)
	for i, c := range codes {
		f := float32(int8(c)) * scales[i/kvQ8GroupSize]
		lo, hi := bf16BytesOfF32(f)
		out[i*2], out[i*2+1] = lo, hi
	}
	return out
}

// bf16BytesOfF32 rounds an f32 to bf16 (round-to-nearest-even) and returns its
// little-endian bytes — the production twin of the test helpers' bf16FromF32.
func bf16BytesOfF32(f float32) (lo, hi byte) {
	bits := math.Float32bits(f)
	r := (bits + 0x7FFF + ((bits >> 16) & 1)) >> 16
	return byte(r), byte(r >> 8)
}
