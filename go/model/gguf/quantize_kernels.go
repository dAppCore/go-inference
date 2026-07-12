// SPDX-Licence-Identifier: EUPL-1.2

package gguf

import (
	"encoding/binary"
	"math"
	"sync"
)

func quantizeQ8_0(values []float32) []byte {
	return appendQuantizeQ8_0(make([]byte, 0, len(values)/32*34), values)
}

// appendQuantizeQ8_0 appends the Q8_0-quantised blocks of values to out and
// returns the grown slice. quantizeQ8_0 is the make-a-fresh-buffer wrapper;
// the streaming writer hands a reused buffer (out[:0]) so the per-chunk
// output allocation is amortised. Output bytes are identical either way.
func appendQuantizeQ8_0(out []byte, values []float32) []byte {
	for blockStart := 0; blockStart < len(values); blockStart += 32 {
		block := values[blockStart : blockStart+32]
		maxAbs := maxAbsFloat32(block)
		scale := float32(0)
		if maxAbs > 0 {
			scale = maxAbs / 127
		}
		// Inline AppendUint16: skip the appendUint16LE func-call + its
		// [2]byte temp. binary.LittleEndian.AppendUint16 lowers to a
		// direct two-byte append.
		out = binary.LittleEndian.AppendUint16(out, float32ToFloat16(scale))
		// Stack-allocated pack buffer + single append at end of block —
		// replaces 32 individual `out = append(out, byte)` calls (each
		// with its own bounds check + length update) with one bulk
		// memcpy. Matches the pattern Q4_0 already uses.
		var packed [32]byte
		if scale == 0 {
			// Zero-block fast path: invScale would be zero so every q
			// is 0; skip the per-element work. `packed` already zeroed
			// by the var declaration.
			out = append(out, packed[:]...)
			continue
		}
		invScale := 1 / scale
		// Hoist the invScale==0 branch out of the inner loop — saves
		// 32 branch evaluations per block.
		for i, value := range block {
			// Multiply by 1/scale instead of dividing — single FMUL
			// vs FDIV per element (32x per block, millions per tensor).
			// Round-half-away-from-zero in float32 directly; skips the
			// float32→float64→math.Round→int round-trip and the call
			// overhead of math.Round (which handles edge cases
			// irrelevant to a clamped-to-127 quantiser).
			scaled := value * invScale
			var q int
			if scaled >= 0 {
				q = int(scaled + 0.5)
			} else {
				q = int(scaled - 0.5)
			}
			// Inline clampInt — avoids the func-call boundary on a
			// 2-branch primitive. The compiler will most likely inline
			// already, but doing it explicitly keeps the hot path
			// dependency-light.
			if q < -127 {
				q = -127
			} else if q > 127 {
				q = 127
			}
			packed[i] = byte(int8(q))
		}
		out = append(out, packed[:]...)
	}
	return out
}

func quantizeQ4_0(values []float32) []byte {
	return appendQuantizeQ4_0(make([]byte, 0, len(values)/32*18), values)
}

// appendQuantizeQ4_0 appends ggml block_q4_0 blocks (18 B/block) for values to
// out. Faithful port of ggml's quantize_row_q4_0_ref: the scale is d =
// max/-8 where max is the SIGNED value at the largest-magnitude position (not
// maxAbs/7) — this is what lets the extremal element hit nibble 0 exactly and
// use the full [0,15] nibble range, vs a naive maxAbs/N scheme that only ever
// reaches nibbles [1,15]. Levels are truncated (not rounded) after a +8.5
// bias: q = trunc(x*id + 8.5), clamped only on the top (MIN 15) — ggml has no
// bottom clamp either, relying on id being derived from the block's own max.
func appendQuantizeQ4_0(out []byte, values []float32) []byte {
	for blockStart := 0; blockStart < len(values); blockStart += 32 {
		block := values[blockStart : blockStart+32]
		amax, signedMax := maxAbsSignedFloat32(block)
		scale := float32(0)
		if amax > 0 {
			scale = signedMax / -8
		}
		out = binary.LittleEndian.AppendUint16(out, float32ToFloat16(scale))
		invScale := float32(0)
		if scale != 0 {
			invScale = 1 / scale
		}
		// Stack-allocated pack buffer instead of make([]byte, 16) per
		// block — saves one heap alloc per 32 input floats. invScale==0
		// (all-zero block) naturally quantises every element to nibble 8
		// (trunc(0+8.5)=8), matching ggml's id=0 fast path without a
		// separate branch.
		var packed [16]byte
		for i := range 16 {
			x0 := block[i] * invScale
			x1 := block[i+16] * invScale
			q0 := int(x0 + 8.5)
			if q0 > 15 {
				q0 = 15
			}
			q1 := int(x1 + 8.5)
			if q1 > 15 {
				q1 = 15
			}
			packed[i] = byte(q0) | byte(q1)<<4
		}
		out = append(out, packed[:]...)
	}
	return out
}

func quantizeQ5_0(values []float32) []byte {
	return appendQuantizeQ5_0(make([]byte, 0, len(values)/32*22), values)
}

// appendQuantizeQ5_0 appends ggml block_q5_0 blocks (22 B/block: 2 d + 4 qh +
// 16 qs — NOT an affine min/scale scheme) for values to out. Faithful port of
// ggml's quantize_row_q5_0_ref: like Q4_0, d = max/-16 from the SIGNED
// extremal value; each element's low 4 bits pack into qs the same
// lower/upper-half-interleaved way as Q4_0, and the 5th (high) bit of every
// element packs into a separate 32-bit qh field (bit j for element j).
func appendQuantizeQ5_0(out []byte, values []float32) []byte {
	for blockStart := 0; blockStart < len(values); blockStart += 32 {
		block := values[blockStart : blockStart+32]
		amax, signedMax := maxAbsSignedFloat32(block)
		scale := float32(0)
		if amax > 0 {
			scale = signedMax / -16
		}
		out = binary.LittleEndian.AppendUint16(out, float32ToFloat16(scale))
		invScale := float32(0)
		if scale != 0 {
			invScale = 1 / scale
		}
		var qh uint32
		var qs [16]byte
		for j := range 16 {
			x0 := block[j] * invScale
			x1 := block[j+16] * invScale
			q0 := int(x0 + 16.5)
			if q0 > 31 {
				q0 = 31
			}
			q1 := int(x1 + 16.5)
			if q1 > 31 {
				q1 = 31
			}
			qs[j] = byte(q0&0xF) | byte(q1&0xF)<<4
			qh |= uint32((q0>>4)&1) << uint(j)
			qh |= uint32((q1>>4)&1) << uint(j+16)
		}
		var qhBytes [4]byte
		binary.LittleEndian.PutUint32(qhBytes[:], qh)
		out = append(out, qhBytes[:]...)
		out = append(out, qs[:]...)
	}
	return out
}

const qkBlockSize = 256
const qkSubBlocks = 16
const qkSubBlockSize = qkBlockSize / qkSubBlocks

type qkScratch struct {
	minBlock     float32
	maxBlock     float32
	subMin       [qkSubBlocks]float32
	subMax       [qkSubBlocks]float32
	scales       [qkSubBlocks]float32
	scalesPacked [12]byte
}

var qkScratchPool = sync.Pool{New: func() any { return &qkScratch{} }}

func quantizeQ4_K(values []float32) []byte {
	nBlocks := len(values) / qkBlockSize
	return appendQuantizeQ4_K(make([]byte, 0, nBlocks*144), values)
}

// kQuantSubBlocks / kQuantSubBlockSize are the Q4_K / Q5_K super-block geometry:
// 8 sub-blocks of 32 elements each (distinct from Q6_K's 16 sub-blocks of 16).
const (
	kQuantSubBlocks    = 8
	kQuantSubBlockSize = qkBlockSize / kQuantSubBlocks // 32
)

// appendQuantizeQ4_K appends ggml block_q4_K super-blocks (144 B/block) for
// values to out. It is a faithful port of ggml's quantize_row_q4_K_ref: each of
// the 8 sub-blocks gets an optimal (scale, min) from makeQKX2Quants, the 8
// scales/mins are quantised to 6 bits against the super-block d/dmin and packed
// in the get_scale_min_k4 layout, and every element is requantised to a 4-bit
// level against its reconstructed sub-scale. Dequant is d*sc*q - dmin*m.
func appendQuantizeQ4_K(out []byte, values []float32) []byte {
	var (
		levels  [qkBlockSize]uint8
		weights [kQuantSubBlockSize]float32
		scales  [kQuantSubBlocks]float32
		mins    [kQuantSubBlocks]float32
		ls, lm  [kQuantSubBlocks]uint8
		packed  [12]byte
		qs      [qkBlockSize / 2]byte
	)
	for blockStart := 0; blockStart < len(values); blockStart += qkBlockSize {
		block := values[blockStart : blockStart+qkBlockSize]
		maxScale, maxMin := kQuantSubBlockScales(block, 15, weights[:], levels[:], scales[:], mins[:], -1.0, 0.1, 20)

		invScale, invMin := kQuantSuperInverses(maxScale, maxMin)
		for j := range kQuantSubBlocks {
			ls[j] = uint8(clampInt(nearestIntGGML(invScale*scales[j]), 0, 63))
			lm[j] = uint8(clampInt(nearestIntGGML(invMin*mins[j]), 0, 63))
		}
		packQ4Q5Scales(ls, lm, &packed)
		d := maxScale / 63
		dmin := maxMin / 63
		kQuantRequantLevels(block, &packed, d, dmin, 15, levels[:])

		out = binary.LittleEndian.AppendUint16(out, float32ToFloat16(d))
		out = binary.LittleEndian.AppendUint16(out, float32ToFloat16(dmin))
		out = append(out, packed[:]...)
		qi := 0
		for j := 0; j < qkBlockSize; j += 64 {
			for l := range 32 {
				qs[qi] = levels[j+l] | (levels[j+l+32] << 4)
				qi++
			}
		}
		out = append(out, qs[:]...)
	}
	return out
}

// kQuantSubBlockScales runs makeQKX2Quants over the 8 sub-blocks of block,
// filling levels/scales/mins and returning the largest scale and min (the
// super-block d/dmin numerators). The ggml importance weights (av_x + |x|) are
// computed per sub-block.
func kQuantSubBlockScales(block []float32, nmax int, weights []float32, levels []uint8, scales, mins []float32, rmin, rdelta float32, nstep int) (maxScale, maxMin float32) {
	for j := range kQuantSubBlocks {
		start := j * kQuantSubBlockSize
		sub := block[start : start+kQuantSubBlockSize]
		var sumX2 float32
		for _, v := range sub {
			sumX2 += v * v
		}
		avX := float32(math.Sqrt(float64(sumX2 / kQuantSubBlockSize)))
		for l := range sub {
			weights[l] = avX + absFloat32(sub[l])
		}
		sc, mn := makeQKX2Quants(nmax, sub, weights, levels[start:start+kQuantSubBlockSize], rmin, rdelta, nstep)
		scales[j] = sc
		mins[j] = mn
		if sc > maxScale {
			maxScale = sc
		}
		if mn > maxMin {
			maxMin = mn
		}
	}
	return maxScale, maxMin
}

// nearestIntGGML rounds like ggml's nearest_int — round half to even via the
// 12582912.0 magic-number bit trick. |fval| must be <= 4194303.
func nearestIntGGML(fval float32) int {
	val := fval + 12582912.0
	i := int32(math.Float32bits(val))
	return int((i & 0x007fffff) - 0x00400000)
}

// kQuantSuperInverses returns the 63/max reciprocals used to quantise the 8
// sub-block scales and mins into 6-bit fields (0 when the numerator is 0).
func kQuantSuperInverses(maxScale, maxMin float32) (invScale, invMin float32) {
	if maxScale > 0 {
		invScale = 63 / maxScale
	}
	if maxMin > 0 {
		invMin = 63 / maxMin
	}
	return invScale, invMin
}

// packQ4Q5Scales packs the 8 6-bit sub-block scales (ls) and mins (lm) into the
// 12-byte block_q4_K / block_q5_K scales field, the exact inverse of ggml's
// get_scale_min_k4: sub-blocks 0..3 keep their 6 bits in bytes 0..3 (scale) and
// 4..7 (min); sub-blocks 4..7 split into a low nibble in bytes 8..11 and a high
// 2 bits folded into the top of bytes 0..3 / 4..7.
func packQ4Q5Scales(ls, lm [kQuantSubBlocks]uint8, packed *[12]byte) {
	for i := range packed {
		packed[i] = 0
	}
	for j := range kQuantSubBlocks {
		if j < 4 {
			packed[j] = ls[j]
			packed[j+4] = lm[j]
		} else {
			packed[j+4] = (ls[j] & 0xF) | ((lm[j] & 0xF) << 4)
			packed[j-4] |= (ls[j] >> 4) << 6
			packed[j] |= (lm[j] >> 4) << 6
		}
	}
}

// getScaleMinK4 reads sub-block j's 6-bit scale and min back out of a packed
// block_q4_K / block_q5_K scales field (ggml's get_scale_min_k4).
func getScaleMinK4(j int, packed *[12]byte) (scale, min uint8) {
	if j < 4 {
		return packed[j] & 63, packed[j+4] & 63
	}
	scale = (packed[j+4] & 0xF) | ((packed[j-4] >> 6) << 4)
	min = (packed[j+4] >> 4) | ((packed[j] >> 6) << 4)
	return scale, min
}

// kQuantRequantLevels requantises every element of block to an [0,nmax] level
// against its reconstructed sub-scale, matching ggml's second pass:
// L = clamp(round((x + dmin*m) / (d*sc))). d/dmin are rounded through f16 first,
// exactly as ggml reads them back from the stored block.
func kQuantRequantLevels(block []float32, packed *[12]byte, d, dmin float32, nmax int, levels []uint8) {
	dF16 := ggufFloat16ToFloat32(float32ToFloat16(d))
	dminF16 := ggufFloat16ToFloat32(float32ToFloat16(dmin))
	for j := range kQuantSubBlocks {
		sc, m := getScaleMinK4(j, packed)
		dsub := dF16 * float32(sc)
		subStart := j * kQuantSubBlockSize
		if dsub == 0 {
			for ii := range kQuantSubBlockSize {
				levels[subStart+ii] = 0
			}
			continue
		}
		dm := dminF16 * float32(m)
		for ii := range kQuantSubBlockSize {
			l := nearestIntGGML((block[subStart+ii] + dm) / dsub)
			levels[subStart+ii] = uint8(clampInt(l, 0, nmax))
		}
	}
}

// makeQKX2Quants finds the sub-block scale and non-negative min that best fit x
// under the importance weights, filling levels with [0,nmax] quant levels.
// Faithful port of ggml's make_qkx2_quants (use_mad=false). Returns (scale,
// min) such that the dequant is x ≈ scale*level - min.
func makeQKX2Quants(nmax int, x, weights []float32, levels []uint8, rmin, rdelta float32, nstep int) (scale, theMin float32) {
	n := len(x)
	minv, maxv := x[0], x[0]
	sumW := weights[0]
	sumX := sumW * x[0]
	for i := 1; i < n; i++ {
		if x[i] < minv {
			minv = x[i]
		}
		if x[i] > maxv {
			maxv = x[i]
		}
		w := weights[i]
		sumW += w
		sumX += w * x[i]
	}
	if minv > 0 {
		minv = 0
	}
	if maxv == minv {
		for i := range levels[:n] {
			levels[i] = 0
		}
		return 0, -minv
	}
	iscale := float32(nmax) / (maxv - minv)
	scale = 1 / iscale
	var bestMad float32
	for i := 0; i < n; i++ {
		l := clampInt(nearestIntGGML(iscale*(x[i]-minv)), 0, nmax)
		levels[i] = uint8(l)
		diff := scale*float32(l) + minv - x[i]
		bestMad += weights[i] * diff * diff
	}
	if nstep < 1 {
		return scale, -minv
	}
	var laux [kQuantSubBlockSize]uint8
	for is := 0; is <= nstep; is++ {
		iscale = (rmin + rdelta*float32(is) + float32(nmax)) / (maxv - minv)
		var sumL, sumL2, sumXL float32
		for i := 0; i < n; i++ {
			l := clampInt(nearestIntGGML(iscale*(x[i]-minv)), 0, nmax)
			laux[i] = uint8(l)
			w := weights[i]
			fl := float32(l)
			sumL += w * fl
			sumL2 += w * fl * fl
			sumXL += w * fl * x[i]
		}
		det := sumW*sumL2 - sumL*sumL
		if det > 0 {
			thisScale := (sumW*sumXL - sumX*sumL) / det
			thisMin := (sumL2*sumX - sumL*sumXL) / det
			if thisMin > 0 {
				thisMin = 0
				thisScale = sumXL / sumL2
			}
			var mad float32
			for i := 0; i < n; i++ {
				diff := thisScale*float32(laux[i]) + thisMin - x[i]
				mad += weights[i] * diff * diff
			}
			if mad < bestMad {
				for i := 0; i < n; i++ {
					levels[i] = laux[i]
				}
				bestMad = mad
				scale = thisScale
				minv = thisMin
			}
		}
	}
	return scale, -minv
}

// groupMaxEPS is ggml's GROUP_MAX_EPS: below this magnitude a (super-)block
// scale is treated as all-zero, guarding the -128/maxScale and 63/maxScale
// style reciprocals from a divide-by-zero.
const groupMaxEPS = 1e-15

// makeQxQuantsRMSE1 returns the optimal SIGNED scale for a symmetric
// (no-min) quantiser fitting x into the levels [-nmax,nmax-1], found by
// least-squares refinement over a grid of candidate scales. Faithful port of
// ggml's make_qx_quants specialised to the one call-site shape Q6_K actually
// uses: rmse_type=1 (weight w=x[i]^2) and no importance-weight override. The
// per-element level output (L in the C source) is discarded by every ggml
// caller of this specific shape — Q6_K re-derives levels in a second pass
// against the coarsened (f16 d * int8 scale) reconstructed scale — so this
// port only returns the float scale.
func makeQxQuantsRMSE1(x []float32, nmax int) float32 {
	var amax, max float32
	for _, v := range x {
		if a := absFloat32(v); a > amax {
			amax = a
			max = v
		}
	}
	if amax < groupMaxEPS {
		return 0
	}
	iscale := float32(-nmax) / max
	var sumlx, suml2 float32
	for _, v := range x {
		l := clampInt(nearestIntGGML(iscale*v), -nmax, nmax-1)
		w := v * v
		fl := float32(l)
		sumlx += w * v * fl
		suml2 += w * fl * fl
	}
	scale := float32(0)
	if suml2 != 0 {
		scale = sumlx / suml2
	}
	best := scale * sumlx
	for is := -9; is <= 9; is++ {
		if is == 0 {
			continue
		}
		iscale = -(float32(nmax) + 0.1*float32(is)) / max
		var sumlxTrial, suml2Trial float32
		for _, v := range x {
			l := clampInt(nearestIntGGML(iscale*v), -nmax, nmax-1)
			w := v * v
			fl := float32(l)
			sumlxTrial += w * v * fl
			suml2Trial += w * fl * fl
		}
		if suml2Trial > 0 && sumlxTrial*sumlxTrial > best*suml2Trial {
			scale = sumlxTrial / suml2Trial
			best = scale * sumlxTrial
		}
	}
	return scale
}

func quantizeQ5_K(values []float32) []byte {
	nBlocks := len(values) / qkBlockSize
	return appendQuantizeQ5_K(make([]byte, 0, nBlocks*176), values)
}

// appendQuantizeQ5_K appends ggml block_q5_K super-blocks (176 B/block) for
// values to out — the same super-block/scale machinery as Q4_K with 5-bit
// levels split into a 4-bit qs field (128 B) and a 1-bit qh field (32 B). A
// faithful port of ggml's quantize_row_q5_K_ref; dequant is d*sc*q - dmin*m.
func appendQuantizeQ5_K(out []byte, values []float32) []byte {
	var (
		levels  [qkBlockSize]uint8
		weights [kQuantSubBlockSize]float32
		scales  [kQuantSubBlocks]float32
		mins    [kQuantSubBlocks]float32
		ls, lm  [kQuantSubBlocks]uint8
		packed  [12]byte
		qh      [qkBlockSize / 8]byte
		qs      [qkBlockSize / 2]byte
	)
	for blockStart := 0; blockStart < len(values); blockStart += qkBlockSize {
		block := values[blockStart : blockStart+qkBlockSize]
		maxScale, maxMin := kQuantSubBlockScales(block, 31, weights[:], levels[:], scales[:], mins[:], -0.5, 0.1, 15)

		invScale, invMin := kQuantSuperInverses(maxScale, maxMin)
		for j := range kQuantSubBlocks {
			ls[j] = uint8(clampInt(nearestIntGGML(invScale*scales[j]), 0, 63))
			lm[j] = uint8(clampInt(nearestIntGGML(invMin*mins[j]), 0, 63))
		}
		packQ4Q5Scales(ls, lm, &packed)
		d := maxScale / 63
		dmin := maxMin / 63
		kQuantRequantLevels(block, &packed, d, dmin, 31, levels[:])

		out = binary.LittleEndian.AppendUint16(out, float32ToFloat16(d))
		out = binary.LittleEndian.AppendUint16(out, float32ToFloat16(dmin))
		out = append(out, packed[:]...)
		for i := range qh {
			qh[i] = 0
		}
		qi := 0
		m1, m2 := uint8(1), uint8(2)
		for n := 0; n < qkBlockSize; n += 64 {
			for j := range 32 {
				l1 := int(levels[n+j])
				if l1 > 15 {
					l1 -= 16
					qh[j] |= m1
				}
				l2 := int(levels[n+j+32])
				if l2 > 15 {
					l2 -= 16
					qh[j] |= m2
				}
				qs[qi+j] = uint8(l1) | (uint8(l2) << 4)
			}
			m1 <<= 2
			m2 <<= 2
			qi += 32
		}
		out = append(out, qh[:]...)
		out = append(out, qs[:]...)
	}
	return out
}

// quantizeQ6_K emits the canonical ggml block_q6_K layout (210 B/block,
// lib/gguflib/gguflib.c + upstream ggml-common.h):
//
//	[  0..128)  ql      — lower 4 bits of each 6-bit quant (2 per byte)
//	[128..192)  qh      — upper 2 bits of each 6-bit quant (4 per byte)
//	[192..208)  scales  — 16 signed int8 sub-block scales
//	[208..210)  d       — f16 super-block scale
//
// Q6_K is symmetric (no dmin): the dequantised value is
// d * scales[sub] * (q - 32) where q ∈ [0,63] and sub = element/16.
// The lower-4/upper-2 split is packed in 128-element groups exactly as
// upstream quantize_row_q6_K_ref does, so a canonical decoder reads it
// back bit-for-bit.
func quantizeQ6_K(values []float32) []byte {
	nBlocks := len(values) / qkBlockSize
	return appendQuantizeQ6_K(make([]byte, 0, nBlocks*210), values)
}

func appendQuantizeQ6_K(out []byte, values []float32) []byte {
	scratch := qkScratchPool.Get().(*qkScratch)
	defer qkScratchPool.Put(scratch)
	var ql [qkBlockSize / 2]byte
	var qh [qkBlockSize / 4]byte
	var scales [qkSubBlocks]int8
	var levels [qkBlockSize]byte // requantised q ∈ [0,63] per element
	for blockStart := 0; blockStart < len(values); blockStart += qkBlockSize {
		block := values[blockStart : blockStart+qkBlockSize]

		// Per-sub-block optimal SIGNED scale (least-squares fit via
		// makeQxQuantsRMSE1, not a naive maxAbs/32) and the global
		// scale-of-scales, picked from the sub-block scale with the
		// largest MAGNITUDE so it maps to exactly int8 -128 — using the
		// full signed scale-field range, the same edge-mapping pattern
		// as Q4_0/Q5_0's d = max/-N.
		var maxScale, maxAbsScale float32
		for sb := range qkSubBlocks {
			subStart := sb * qkSubBlockSize
			scale := makeQxQuantsRMSE1(block[subStart:subStart+qkSubBlockSize], 32)
			scratch.scales[sb] = scale
			if a := absFloat32(scale); a > maxAbsScale {
				maxAbsScale = a
				maxScale = scale
			}
		}
		d := float32(0)
		if maxAbsScale >= groupMaxEPS {
			iscale := float32(-128) / maxScale
			d = 1 / iscale
			for sb := range qkSubBlocks {
				l := nearestIntGGML(iscale * scratch.scales[sb])
				if l > 127 {
					l = 127
				}
				scales[sb] = int8(l)
			}
		} else {
			for sb := range qkSubBlocks {
				scales[sb] = 0
			}
		}

		// Requantise every element against its reconstructed sub-scale,
		// to q ∈ [0,63] (signed -32..31 re-centred by +32). A sub-block
		// whose rounded scale lands on exactly 0 leaves its levels at 0
		// (Go's zero value) rather than ggml's carried-over previous-block
		// state — numerically identical, since dequant is dsub*(level-32)
		// which is 0 either way when dsub is 0.
		for i := range levels {
			levels[i] = 0
		}
		for sb := range qkSubBlocks {
			dsub := d * float32(scales[sb])
			if dsub == 0 {
				continue
			}
			subStart := sb * qkSubBlockSize
			for j := range qkSubBlockSize {
				l := clampInt(nearestIntGGML(block[subStart+j]/dsub), -32, 31)
				levels[subStart+j] = byte(l + 32)
			}
		}

		// Pack ql/qh in 128-element groups, matching
		// quantize_row_q6_K_ref: for each half j∈{0,128}, l∈[0,32),
		// ql holds 4-bit lows of L[j+l], L[j+l+32], L[j+l+64], L[j+l+96];
		// qh holds their 2-bit highs.
		for i := range ql {
			ql[i] = 0
		}
		for i := range qh {
			qh[i] = 0
		}
		for j := 0; j < qkBlockSize; j += 128 {
			for l := range 32 {
				q1 := levels[j+l] & 0xF
				q2 := levels[j+l+32] & 0xF
				q3 := levels[j+l+64] & 0xF
				q4 := levels[j+l+96] & 0xF
				ql[j/2+l] = q1 | (q3 << 4)
				ql[j/2+l+32] = q2 | (q4 << 4)
				qh[j/4+l] = (levels[j+l] >> 4) |
					((levels[j+l+32] >> 4) << 2) |
					((levels[j+l+64] >> 4) << 4) |
					((levels[j+l+96] >> 4) << 6)
			}
		}

		out = append(out, ql[:]...)
		out = append(out, qh[:]...)
		for _, s := range scales {
			out = append(out, byte(s))
		}
		out = binary.LittleEndian.AppendUint16(out, float32ToFloat16(d))
	}
	return out
}

// packQ3KScales packs 16 unsigned 6-bit scale values (signed scale + 32)
// into the 12-byte form that dequantize_row_q3_K's kmask unpack reverses:
// each value's low nibble lands in bytes [0,8), its high 2 bits in bytes
// [8,12). It is the exact arithmetic inverse of that unpack (asserted by
// TestQuantizeQ3KScalePack_RoundTrips).
func packQ3KScales(scales [qkSubBlocks]uint8, out *[12]byte) {
	for i := range out {
		out[i] = 0
	}
	// Low nibbles → bytes 0..7 (positions 0..7) and 0..7 (positions 8..15).
	for j := range qkSubBlocks {
		lo := scales[j] & 0xF
		if j < 8 {
			out[j] |= lo
		} else {
			out[j-8] |= lo << 4
		}
	}
	// High 2 bits of each scale → bytes 8..11, two bits per (j mod 4),
	// grouped so the decoder's tmp>>{0,2,4,6} & kmask1 recovers them.
	for j := range qkSubBlocks {
		hi := (scales[j] >> 4) & 3
		out[8+(j%4)] |= hi << (2 * (j / 4))
	}
}

// quantizeQ3_K emits the canonical ggml block_q3_K layout (110 B/block):
//
//	[  0.. 32)  hmask   — high bit of each 3-bit quant (1 per element)
//	[ 32.. 96)  qs      — low 2 bits of each quant (4 per byte)
//	[ 96..108)  scales  — 16 six-bit scales packed into 12 bytes
//	[108..110)  d       — f16 super-block scale
//
// Q3_K is symmetric (no dmin). The dequantised value is
// d * (scale[sub]-32) * ((qs&3) - (hmask_set ? 0 : 4)), reproducing
// dequantize_row_q3_K. qs uses the same 128-element-group interleave as
// Q2_K; the hmask walk mirrors the decoder's m/shift/is loop exactly.
func quantizeQ3_K(values []float32) []byte {
	nBlocks := len(values) / qkBlockSize
	return appendQuantizeQ3_K(make([]byte, 0, nBlocks*110), values)
}

func appendQuantizeQ3_K(out []byte, values []float32) []byte {
	scratch := qkScratchPool.Get().(*qkScratch)
	defer qkScratchPool.Put(scratch)
	var hmask [qkBlockSize / 8]byte
	var qs [qkBlockSize / 4]byte
	var packedScales [12]byte
	var rawScales [qkSubBlocks]uint8 // signed sub-scale + 32, ∈ [0,63]
	var levels [qkBlockSize]uint8    // unsigned Lq ∈ [0,7] per element
	for blockStart := 0; blockStart < len(values); blockStart += qkBlockSize {
		block := values[blockStart : blockStart+qkBlockSize]

		// Per-sub-block signed scale (max |value| / 4 covers [-4,3]) and the
		// scale-of-scales mapping into the 6-bit signed scale field.
		maxScale := float32(0)
		for sb := range qkSubBlocks {
			subStart := sb * qkSubBlockSize
			maxAbs := float32(0)
			for j := range qkSubBlockSize {
				if a := absFloat32(block[subStart+j]); a > maxAbs {
					maxAbs = a
				}
			}
			scratch.scales[sb] = maxAbs / 4
			if scratch.scales[sb] > maxScale {
				maxScale = scratch.scales[sb]
			}
		}
		d := float32(0)
		var iscale float32
		if maxScale > 0 {
			iscale = 31 / maxScale // signed scale range is [-32,31]
			d = maxScale / 31
		}
		for sb := range qkSubBlocks {
			s := clampInt(int(roundFloat32(iscale*scratch.scales[sb])), -32, 31)
			rawScales[sb] = uint8(s + 32)
		}

		// Requantise to signed L ∈ [-4,3]; store as unsigned Lq = L+4.
		for sb := range qkSubBlocks {
			subStart := sb * qkSubBlockSize
			subScale := d * float32(int(rawScales[sb])-32)
			inv := float32(0)
			if subScale != 0 {
				inv = 1 / subScale
			}
			for j := range qkSubBlockSize {
				l := 0
				if inv != 0 {
					l = clampInt(int(roundFloat32(block[subStart+j]*inv)), -4, 3)
				}
				levels[subStart+j] = uint8(l + 4)
			}
		}

		for i := range hmask {
			hmask[i] = 0
		}
		for i := range qs {
			qs[i] = 0
		}
		// hmask: high bit (Lq>3 → set) following the decoder's m/is walk.
		// m = 1<<g, g advances per (n-half, j) group; hm byte index = l or
		// l+16 within each 32-element pair. is selects the sub-block.
		m := uint8(1)
		is := 0
		for n := 0; n < qkBlockSize; n += 128 {
			for range 4 {
				base := is * qkSubBlockSize
				for l := range 16 {
					if levels[base+l] > 3 {
						hmask[l] |= m
					}
				}
				is++
				base = is * qkSubBlockSize
				for l := range 16 {
					if levels[base+l] > 3 {
						hmask[16+l] |= m
					}
				}
				is++
				m <<= 1
			}
			_ = n
		}
		// qs: low 2 bits (Lq&3). dequantize_row_q3_K reads, per 128-element
		// half, q[l] at shift 2j (j=0..3, l=0..15) then q[l+16] at the same
		// shift — i.e. output position p within the half uses qs byte p%32
		// and shift 2*(p/32). Pack the exact inverse.
		for n := 0; n < qkBlockSize; n += 128 {
			byteBase := n / 4
			for p := range 128 {
				qs[byteBase+(p%32)] |= (levels[n+p] & 3) << (2 * (p / 32))
			}
		}

		packQ3KScales(rawScales, &packedScales)
		out = append(out, hmask[:]...)
		out = append(out, qs[:]...)
		out = append(out, packedScales[:]...)
		out = binary.LittleEndian.AppendUint16(out, float32ToFloat16(d))
	}
	return out
}

// quantizeQ2_K emits the canonical ggml block_q2_K layout (84 B/block —
// the upstream static_assert is 84, not 82: the gguflib type-size table's
// 82 drops dmin, and its own decoder advances 16+64+4=84):
//
//	[ 0..16)  scales  — 16 bytes, each (scale_lo4 | min_hi4)
//	[16..80)  qs      — 64 bytes, 2-bit quants (4 per byte)
//	[80..82)  d       — f16 super-block scale-of-scales
//	[82..84)  dmin    — f16 super-block scale-of-mins
//
// Q2_K is affine: the dequantised value is d*scale*q - dmin*min with
// q ∈ [0,3], reproducing dequantize_row_q2_K. qs uses the same
// sequential-within-shift layout as Q3_K (byte p%32, shift 2*(p/32) per
// 128-element half).
func quantizeQ2_K(values []float32) []byte {
	nBlocks := len(values) / qkBlockSize
	return appendQuantizeQ2_K(make([]byte, 0, nBlocks*84), values)
}

func appendQuantizeQ2_K(out []byte, values []float32) []byte {
	scratch := qkScratchPool.Get().(*qkScratch)
	defer qkScratchPool.Put(scratch)
	var scales [qkSubBlocks]byte
	var qs [qkBlockSize / 4]byte
	var levels [qkBlockSize]uint8 // q ∈ [0,3] per element
	for blockStart := 0; blockStart < len(values); blockStart += qkBlockSize {
		block := values[blockStart : blockStart+qkBlockSize]

		// Per-sub-block affine fit: scale = (max-min)/3, min = -minValue
		// (the decoder subtracts dmin*min, so min is stored as a positive
		// magnitude of the most-negative offset). Then the block-global d
		// and dmin map each sub scale/min into a 4-bit field.
		maxScale := float32(0)
		maxMin := float32(0)
		for sb := range qkSubBlocks {
			subStart := sb * qkSubBlockSize
			lo, hi := block[subStart], block[subStart]
			for j := 1; j < qkSubBlockSize; j++ {
				v := block[subStart+j]
				if v < lo {
					lo = v
				}
				if v > hi {
					hi = v
				}
			}
			sc := (hi - lo) / 3
			mn := -lo // y = scale*q - min ⇒ min = -lo so q=0 → lo
			scratch.subMax[sb] = sc
			scratch.subMin[sb] = mn
			if sc > maxScale {
				maxScale = sc
			}
			if mn > maxMin {
				maxMin = mn
			}
		}
		d := float32(0)
		dmin := float32(0)
		var iscale, imin float32
		if maxScale > 0 {
			d = maxScale / 15
			iscale = 15 / maxScale
		}
		if maxMin > 0 {
			dmin = maxMin / 15
			imin = 15 / maxMin
		}
		for sb := range qkSubBlocks {
			sc := clampInt(int(roundFloat32(iscale*scratch.subMax[sb])), 0, 15)
			mn := clampInt(int(roundFloat32(imin*scratch.subMin[sb])), 0, 15)
			scales[sb] = byte(sc) | byte(mn<<4)
		}

		// Requantise each element to q ∈ [0,3] against the reconstructed
		// sub-scale/sub-min (exactly what the decoder reconstructs).
		for sb := range qkSubBlocks {
			subStart := sb * qkSubBlockSize
			sc := d * float32(scales[sb]&0xF)
			ml := dmin * float32(scales[sb]>>4)
			inv := float32(0)
			if sc != 0 {
				inv = 1 / sc
			}
			for j := range qkSubBlockSize {
				q := 0
				if inv != 0 {
					q = clampInt(int(roundFloat32((block[subStart+j]+ml)*inv)), 0, 3)
				}
				levels[subStart+j] = uint8(q)
			}
		}

		for i := range qs {
			qs[i] = 0
		}
		for n := 0; n < qkBlockSize; n += 128 {
			byteBase := n / 4
			for p := range 128 {
				qs[byteBase+(p%32)] |= (levels[n+p] & 3) << (2 * (p / 32))
			}
		}

		out = append(out, scales[:]...)
		out = append(out, qs[:]...)
		out = binary.LittleEndian.AppendUint16(out, float32ToFloat16(d))
		out = binary.LittleEndian.AppendUint16(out, float32ToFloat16(dmin))
	}
	return out
}

// quantizeQ8_K emits the canonical ggml block_q8_K layout (292 B/block):
//
//	[  0..  4)  d      — float32 super-block scale (NOT f16, unlike the
//	                     other K-quants)
//	[  4..260)  qs     — 256 signed int8 quants
//	[260..292)  bsums  — 16 int16 sums of qs over each 16-element group
//
// Q8_K is a symmetric int8 quantiser (no dmin): d = max|x|/127,
// q = round(x/d) ∈ [-127,127], reproducing quantize_row_q8_K_ref. The
// bsums let consumers skip a re-sum during dot products.
func quantizeQ8_K(values []float32) []byte {
	nBlocks := len(values) / qkBlockSize
	return appendQuantizeQ8_K(make([]byte, 0, nBlocks*292), values)
}

func appendQuantizeQ8_K(out []byte, values []float32) []byte {
	var qs [qkBlockSize]int8
	for blockStart := 0; blockStart < len(values); blockStart += qkBlockSize {
		block := values[blockStart : blockStart+qkBlockSize]
		maxAbs := maxAbsFloat32(block)
		d := float32(0)
		var inv float32
		if maxAbs > 0 {
			d = maxAbs / 127
			inv = 127 / maxAbs
		}
		for i, value := range block {
			q := 0
			if inv != 0 {
				q = clampInt(int(roundFloat32(value*inv)), -127, 127)
			}
			qs[i] = int8(q)
		}
		out = binary.LittleEndian.AppendUint32(out, math.Float32bits(d))
		for _, q := range qs {
			out = append(out, byte(q))
		}
		// 16 int16 group sums, little-endian.
		for sb := range qkSubBlocks {
			sum := int16(0)
			base := sb * qkSubBlockSize
			for j := range qkSubBlockSize {
				sum += int16(qs[base+j])
			}
			out = binary.LittleEndian.AppendUint16(out, uint16(sum))
		}
	}
	return out
}

// maxAbsFloat32 returns max(|v|) over values. The inner loop avoids
// math.Abs (which round-trips float32→float64→float32 per element); a
// direct bit-clear of the float32 sign bit lowers to ARM64 FABS in one
// instruction. The 4-way unroll (W8-A2 lever) lets the M-series pipeline
// keep four FABS+FCMP chains independent so per-iteration latency hides
// behind instruction-level parallelism. Block-sized inputs (32 / 256
// elements) hit the unrolled path; the scalar tail handles the
// remainder.
// absFloat32 returns |value| via a sign-bit clear — matches the
// branchless style maxAbsFloat32 already uses, no math.Abs call.
func absFloat32(value float32) float32 {
	return math.Float32frombits(math.Float32bits(value) & 0x7fffffff)
}

// roundFloat32 rounds half away from zero in float32 directly, the same
// quantiser-friendly rounding quantizeQ8_0 inlines (skips the
// float32→float64→math.Round round-trip).
func roundFloat32(value float32) float32 {
	if value >= 0 {
		return float32(int(value + 0.5))
	}
	return float32(int(value - 0.5))
}

func maxAbsFloat32(values []float32) float32 {
	const mask = 0x7fffffff
	var m0, m1, m2, m3 float32
	i := 0
	n := len(values)
	for ; i+4 <= n; i += 4 {
		a0 := math.Float32frombits(math.Float32bits(values[i]) & mask)
		a1 := math.Float32frombits(math.Float32bits(values[i+1]) & mask)
		a2 := math.Float32frombits(math.Float32bits(values[i+2]) & mask)
		a3 := math.Float32frombits(math.Float32bits(values[i+3]) & mask)
		if a0 > m0 {
			m0 = a0
		}
		if a1 > m1 {
			m1 = a1
		}
		if a2 > m2 {
			m2 = a2
		}
		if a3 > m3 {
			m3 = a3
		}
	}
	maxAbs := m0
	if m1 > maxAbs {
		maxAbs = m1
	}
	if m2 > maxAbs {
		maxAbs = m2
	}
	if m3 > maxAbs {
		maxAbs = m3
	}
	for ; i < n; i++ {
		abs := math.Float32frombits(math.Float32bits(values[i]) & mask)
		if abs > maxAbs {
			maxAbs = abs
		}
	}
	return maxAbs
}

// maxAbsSignedFloat32 returns the largest |v| over values (amax) together
// with the SIGNED v at that position (not just its magnitude) — the exact
// amax/max pair ggml's quantize_row_q4_0_ref / q5_0_ref scan for. The sign
// of the extremal element, not just its magnitude, drives those formats'
// scale (d = max/-8 or max/-16), which is what lets the extremum map onto
// the very edge of the nibble range instead of wasting one level.
func maxAbsSignedFloat32(values []float32) (amax, signedMax float32) {
	for _, v := range values {
		a := absFloat32(v)
		if a > amax {
			amax = a
			signedMax = v
		}
	}
	return amax, signedMax
}

func appendUint16LE(out []byte, value uint16) []byte {
	var buf [2]byte
	binary.LittleEndian.PutUint16(buf[:], value)
	return append(out, buf[:]...)
}

func clampInt(value, minValue, maxValue int) int {
	if value < minValue {
		return minValue
	}
	if value > maxValue {
		return maxValue
	}
	return value
}

func float32ToFloat16(value float32) uint16 {
	bits := math.Float32bits(value)
	sign := uint16((bits >> 16) & 0x8000)
	exp := int((bits >> 23) & 0xff)
	frac := bits & 0x7fffff
	if exp == 255 {
		if frac == 0 {
			return sign | 0x7c00
		}
		return sign | 0x7e00
	}
	exp = exp - 127 + 15
	if exp >= 31 {
		return sign | 0x7c00
	}
	if exp <= 0 {
		if exp < -10 {
			return sign
		}
		frac |= 0x800000
		shift := uint32(14 - exp)
		half := uint16(frac >> shift)
		if (frac>>(shift-1))&1 != 0 {
			half++
		}
		return sign | half
	}
	half := sign | uint16(exp<<10) | uint16(frac>>13)
	if frac&0x00001000 != 0 {
		half++
	}
	return half
}
