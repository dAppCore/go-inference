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

func appendQuantizeQ4_0(out []byte, values []float32) []byte {
	for blockStart := 0; blockStart < len(values); blockStart += 32 {
		block := values[blockStart : blockStart+32]
		maxAbs := maxAbsFloat32(block)
		scale := float32(0)
		if maxAbs > 0 {
			scale = maxAbs / 7
		}
		out = binary.LittleEndian.AppendUint16(out, float32ToFloat16(scale))
		// Stack-allocated pack buffer instead of make([]byte, 16) per
		// block — saves one heap alloc per 32 input floats.
		var packed [16]byte
		if scale == 0 {
			// Zero-block fast path: q=0 → q+8=8 (Q4_0 stores
			// (q+8) ∈ [0,15] unsigned). Both nibbles of each packed
			// byte are 8, so the byte value is 0x88. Skips the
			// per-element multiply + round + branch work.
			for i := range packed {
				packed[i] = 0x88
			}
			out = append(out, packed[:]...)
			continue
		}
		invScale := 1 / scale
		// Split the i<16 branch out of the inner loop — two clean
		// 16-iter loops let the back-end keep the lower-nibble writes
		// (packed[i] = q) and upper-nibble OR-writes (packed[i-16] |=
		// q<<4) on independent memory dependencies. Same total work,
		// less branch overhead and a cleaner dep chain.
		for i := range 16 {
			value := block[i]
			scaled := value * invScale
			var q int
			// Round-half-away-from-zero in float32 — same optimisation
			// as quantizeQ8_0. The +8 bias re-centres the signed
			// quantised range into the [0,15] unsigned range Q4_0
			// stores.
			if scaled >= 0 {
				q = int(scaled+0.5) + 8
			} else {
				q = int(scaled-0.5) + 8
			}
			if q < 0 {
				q = 0
			} else if q > 15 {
				q = 15
			}
			packed[i] = byte(q)
		}
		for i := 16; i < 32; i++ {
			value := block[i]
			scaled := value * invScale
			var q int
			if scaled >= 0 {
				q = int(scaled+0.5) + 8
			} else {
				q = int(scaled-0.5) + 8
			}
			if q < 0 {
				q = 0
			} else if q > 15 {
				q = 15
			}
			packed[i-16] |= byte(q << 4)
		}
		out = append(out, packed[:]...)
	}
	return out
}

func quantizeQ5_0(values []float32) []byte {
	return appendQuantizeQ5_0(make([]byte, 0, len(values)/32*24), values)
}

func appendQuantizeQ5_0(out []byte, values []float32) []byte {
	for blockStart := 0; blockStart < len(values); blockStart += 32 {
		block := values[blockStart : blockStart+32]
		maxAbs := maxAbsFloat32(block)
		minVal := minFloat32(block)
		scale := float32(0)
		if maxAbs > 0 {
			scale = (maxAbs - minVal) / 31
		}
		out = binary.LittleEndian.AppendUint16(out, float32ToFloat16(scale))
		out = binary.LittleEndian.AppendUint16(out, float32ToFloat16(minVal))

		var packed [20]byte
		if scale == 0 {
			for i := range packed {
				packed[i] = 0x44 // 0b01000100 → each 5-bit nibble is 4 (midpoint)
			}
		} else {
			invScale := 1 / scale
			bitBuf := uint64(0)
			bitCount := 0
			byteIdx := 0
			for _, value := range block {
				scaled := (value - minVal) * invScale
				var q int
				if scaled >= 0 {
					q = int(scaled + 0.5)
				} else {
					q = int(scaled - 0.5)
				}
				if q < 0 {
					q = 0
				} else if q > 31 {
					q = 31
				}
				bitBuf |= uint64(q) << bitCount
				bitCount += 5
				for bitCount >= 8 {
					packed[byteIdx] = byte(bitBuf & 0xFF)
					bitBuf >>= 8
					bitCount -= 8
					byteIdx++
				}
			}
		}
		out = append(out, packed[:]...)
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

func appendQuantizeQ4_K(out []byte, values []float32) []byte {
	scratch := qkScratchPool.Get().(*qkScratch)
	defer qkScratchPool.Put(scratch)

	for blockStart := 0; blockStart < len(values); blockStart += qkBlockSize {
		block := values[blockStart : blockStart+qkBlockSize]
		scratch.minBlock, scratch.maxBlock = block[0], block[0]
		for _, v := range block[1:] {
			if v < scratch.minBlock {
				scratch.minBlock = v
			}
			if v > scratch.maxBlock {
				scratch.maxBlock = v
			}
		}
		d := float32(0)
		if scratch.maxBlock > scratch.minBlock {
			d = (scratch.maxBlock - scratch.minBlock) / 15
		}
		dmin := scratch.minBlock
		out = binary.LittleEndian.AppendUint16(out, float32ToFloat16(d))
		out = binary.LittleEndian.AppendUint16(out, float32ToFloat16(dmin))

		var quants [qkBlockSize / 2]byte
		if d == 0 {
			for i := range quants {
				quants[i] = 0x88
			}
		} else {
			invD := 1 / d
			for sb := range qkSubBlocks {
				subStart := sb * qkSubBlockSize
				scratch.subMin[sb] = block[subStart]
				scratch.subMax[sb] = block[subStart]
				for j := 1; j < qkSubBlockSize; j++ {
					v := block[subStart+j]
					if v < scratch.subMin[sb] {
						scratch.subMin[sb] = v
					}
					if v > scratch.subMax[sb] {
						scratch.subMax[sb] = v
					}
				}
				if scratch.subMax[sb] > scratch.subMin[sb] {
					scratch.scales[sb] = (scratch.subMax[sb] - scratch.subMin[sb]) / 63
				} else {
					scratch.scales[sb] = 0
				}
			}
			for sb := range qkSubBlocks {
				subStart := sb * qkSubBlockSize
				for j := range qkSubBlockSize {
					scaled := (block[subStart+j] - dmin) * invD
					q := clampInt(int(scaled+0.5), 0, 15)
					if j%2 == 0 {
						quants[(subStart+j)/2] = byte(q)
					} else {
						quants[(subStart+j)/2] |= byte(q << 4)
					}
				}
			}
		}
		packKScales(scratch.scales[:], &scratch.scalesPacked)
		out = append(out, scratch.scalesPacked[:]...)
		out = append(out, quants[:]...)
	}
	return out
}

func packKScales(scales []float32, packed *[12]byte) {
	var scMin, scMax float32 = scales[0], scales[0]
	for _, s := range scales[1:] {
		if s < scMin {
			scMin = s
		}
		if s > scMax {
			scMax = s
		}
	}
	if scMax <= scMin {
		return
	}
	dScale := (scMax - scMin) / 63
	invDScale := 1 / dScale
	bitBuf := uint64(0)
	bitCount := 0
	byteIdx := 0
	for _, s := range scales {
		scaled := (s - scMin) * invDScale
		q := clampInt(int(scaled+0.5), 0, 63)
		bitBuf |= uint64(q) << bitCount
		bitCount += 6
		for bitCount >= 8 && byteIdx < 12 {
			packed[byteIdx] = byte(bitBuf & 0xFF)
			bitBuf >>= 8
			bitCount -= 8
			byteIdx++
		}
	}
}

func quantizeKBlock(values []float32, quants []byte, bits int, d, dmin float32, scratch *qkScratch) {
	if d == 0 {
		return
	}
	invD := 1 / d
	bitBuf := uint64(0)
	bitCount := 0
	byteIdx := 0
	for idx, value := range values {
		if idx%qkSubBlockSize == 0 {
			sb := idx / qkSubBlockSize
			scratch.subMin[sb] = value
			scratch.subMax[sb] = value
			for j := 1; j < qkSubBlockSize && idx+j < len(values); j++ {
				v := values[idx+j]
				if v < scratch.subMin[sb] {
					scratch.subMin[sb] = v
				}
				if v > scratch.subMax[sb] {
					scratch.subMax[sb] = v
				}
			}
			if scratch.subMax[sb] > scratch.subMin[sb] {
				scratch.scales[sb] = (scratch.subMax[sb] - scratch.subMin[sb]) / 63
			} else {
				scratch.scales[sb] = 0
			}
		}
		scaled := (value - dmin) * invD
		q := clampInt(int(scaled+0.5), 0, (1<<bits)-1)
		bitBuf |= uint64(q) << bitCount
		bitCount += bits
		for bitCount >= 8 && byteIdx < len(quants) {
			quants[byteIdx] = byte(bitBuf & 0xFF)
			bitBuf >>= 8
			bitCount -= 8
			byteIdx++
		}
	}
}

func quantizeQ5_K(values []float32) []byte {
	nBlocks := len(values) / qkBlockSize
	return appendQuantizeQ5_K(make([]byte, 0, nBlocks*176), values)
}

func appendQuantizeQ5_K(out []byte, values []float32) []byte {
	scratch := qkScratchPool.Get().(*qkScratch)
	defer qkScratchPool.Put(scratch)
	for blockStart := 0; blockStart < len(values); blockStart += qkBlockSize {
		block := values[blockStart : blockStart+qkBlockSize]
		scratch.minBlock, scratch.maxBlock = block[0], block[0]
		for _, v := range block[1:] {
			if v < scratch.minBlock {
				scratch.minBlock = v
			}
			if v > scratch.maxBlock {
				scratch.maxBlock = v
			}
		}
		d := float32(0)
		if scratch.maxBlock > scratch.minBlock {
			d = (scratch.maxBlock - scratch.minBlock) / 31
		}
		dmin := scratch.minBlock
		out = binary.LittleEndian.AppendUint16(out, float32ToFloat16(d))
		out = binary.LittleEndian.AppendUint16(out, float32ToFloat16(dmin))
		var quants [qkBlockSize * 5 / 8]byte
		quantizeKBlock(block, quants[:], 5, d, dmin, scratch)
		packKScales(scratch.scales[:], &scratch.scalesPacked)
		out = append(out, scratch.scalesPacked[:]...)
		out = append(out, quants[:]...)
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

		// Per-sub-block signed scale (max |value| / 32) and the global
		// scale-of-scales that maps each into the int8 scale field.
		maxScale := float32(0)
		for sb := range qkSubBlocks {
			subStart := sb * qkSubBlockSize
			maxAbs := float32(0)
			for j := range qkSubBlockSize {
				if a := absFloat32(block[subStart+j]); a > maxAbs {
					maxAbs = a
				}
			}
			scratch.scales[sb] = maxAbs / 32 // sub-block scale candidate
			if scratch.scales[sb] > maxScale {
				maxScale = scratch.scales[sb]
			}
		}
		d := float32(0)
		var iscale float32
		if maxScale > 0 {
			iscale = 127 / maxScale
			d = maxScale / 127
		}
		for sb := range qkSubBlocks {
			scales[sb] = int8(clampInt(int(roundFloat32(iscale*scratch.scales[sb])), -127, 127))
		}

		// Requantise every element against its reconstructed sub-scale,
		// to q ∈ [0,63] (signed -32..31 re-centred by +32).
		for sb := range qkSubBlocks {
			subStart := sb * qkSubBlockSize
			subScale := d * float32(scales[sb])
			inv := float32(0)
			if subScale != 0 {
				inv = 1 / subScale
			}
			for j := range qkSubBlockSize {
				q := 0
				if inv != 0 {
					q = clampInt(int(roundFloat32(block[subStart+j]*inv)), -32, 31)
				}
				levels[subStart+j] = byte(q + 32)
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

func minFloat32(values []float32) float32 {
	minVal := values[0]
	for i := 1; i < len(values); i++ {
		if values[i] < minVal {
			minVal = values[i]
		}
	}
	return minVal
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
