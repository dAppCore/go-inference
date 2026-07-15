// SPDX-Licence-Identifier: EUPL-1.2

package enginetest

import (
	"math"

	core "dappco.re/go"
)

// quant_reference.go is the pure-Go, engine-independent reference for the
// group-affine weight-quant decode projection every backend registers against
// model.BackendQuant(backend, "affine") (model/quant.go): dequantise a packed
// row — LSB-first bit-packed codes, one bf16 scale+bias per group — to
// float32, then dot with the bf16 activation vector. It imports no engine
// package: the group-affine format (MLX's packing scheme) is reimplemented
// independently here rather than borrowed from any backend's own dequantizer,
// so agreement between a backend's MatVec and ReferenceAffineMatVec is
// evidence the backend's arithmetic is correct, not merely self-consistent
// with itself.

// bf16Decode reads one little-endian bfloat16 (2 bytes: lo, hi) as float32.
func bf16Decode(lo, hi byte) float32 {
	return math.Float32frombits(uint32(uint16(lo)|uint16(hi)<<8) << 16)
}

// bf16Encode converts a float32 to bfloat16 bits with round-to-nearest-even.
func bf16Encode(v float32) uint16 {
	bits := math.Float32bits(v)
	if bits&0x7fffffff > 0x7f800000 { // NaN: keep it quiet, non-zero mantissa
		return uint16(bits>>16) | 0x0040
	}
	rounding := (bits>>16)&1 + 0x7fff
	return uint16((bits + rounding) >> 16)
}

// affineExtractCode reads the bits-wide affine code at bit offset bitOff from a
// packed row, LSB-first contiguous — MLX's group-affine packing (for 4-bit this
// is the familiar low-nibble-then-high-nibble layout; other widths span byte
// boundaries the same way).
func affineExtractCode(p []byte, bitOff, bits int) uint32 {
	var v uint32
	for got := 0; got < bits; {
		bi := (bitOff + got) / 8
		off := (bitOff + got) % 8
		take := min(8-off, bits-got)
		chunk := (uint32(p[bi]) >> uint(off)) & ((1 << uint(take)) - 1)
		v |= chunk << uint(got)
		got += take
	}
	return v
}

// affineSetCode writes a bits-wide code at bit offset bitOff within p,
// LSB-first across byte boundaries — the exact inverse of affineExtractCode.
// Used only to build this package's own deterministic fixtures (quant_parity.go).
func affineSetCode(p []byte, bitOff, bits int, code uint32) {
	for got := 0; got < bits; {
		bi := (bitOff + got) / 8
		off := (bitOff + got) % 8
		take := min(8-off, bits-got)
		mask := byte((1<<uint(take))-1) << uint(off)
		shifted := byte((code >> uint(got)) << uint(off))
		p[bi] = (p[bi] &^ mask) | (shifted & mask)
		got += take
	}
}

// ReferenceAffineMatVec is the pure-Go reference for the group-affine quant
// decode projection: out = x @ Wᵀ for a group-affine quantised (outDim x inDim)
// weight — bf16 activations in, bf16 result out — matching model.QuantMatVec's
// MatVec contract (model/quant.go) exactly, so a backend's registered
// implementation and this function are directly comparable on the same
// fixture. packed/scales/biases follow MLX's group-affine layout: packed is
// outDim*inDim*bits/8 LSB-first bit-packed codes; scales and biases are each
// outDim*(inDim/groupSize) bf16 values, one pair per group per row; the
// dequantised weight element is scale*code+bias. The row/activation dot
// product accumulates in float64 for a stable, order-independent reference sum.
func ReferenceAffineMatVec(x, packed, scales, biases []byte, outDim, inDim, groupSize, bits int) ([]byte, error) {
	if outDim < 0 || inDim < 0 {
		return nil, core.NewError("enginetest.ReferenceAffineMatVec: outDim/inDim must be non-negative")
	}
	if outDim == 0 || inDim == 0 {
		return make([]byte, outDim*2), nil
	}
	if bits <= 0 || bits > 8 {
		return nil, core.NewError("enginetest.ReferenceAffineMatVec: bits must be in 1..8")
	}
	if groupSize <= 0 || inDim%groupSize != 0 {
		return nil, core.NewError("enginetest.ReferenceAffineMatVec: groupSize must be > 0 and divide inDim")
	}
	if inDim*bits%8 != 0 {
		return nil, core.NewError("enginetest.ReferenceAffineMatVec: inDim*bits must be byte-aligned")
	}
	if len(x) != inDim*2 {
		return nil, core.NewError("enginetest.ReferenceAffineMatVec: len(x) must equal inDim bf16 bytes")
	}
	rowPacked := inDim * bits / 8
	rowSB := (inDim / groupSize) * 2
	if len(packed) != outDim*rowPacked || len(scales) != outDim*rowSB || len(biases) != outDim*rowSB {
		return nil, core.NewError("enginetest.ReferenceAffineMatVec: packed/scales/biases size mismatch")
	}

	xf := make([]float64, inDim)
	for i := range xf {
		xf[i] = float64(bf16Decode(x[i*2], x[i*2+1]))
	}

	out := make([]byte, outDim*2)
	for r := 0; r < outDim; r++ {
		pRow := packed[r*rowPacked : (r+1)*rowPacked]
		sRow := scales[r*rowSB : (r+1)*rowSB]
		bRow := biases[r*rowSB : (r+1)*rowSB]
		var acc float64
		for c := 0; c < inDim; c++ {
			g := c / groupSize
			scale := bf16Decode(sRow[g*2], sRow[g*2+1])
			bias := bf16Decode(bRow[g*2], bRow[g*2+1])
			code := affineExtractCode(pRow, c*bits, bits)
			w := float64(scale)*float64(code) + float64(bias)
			acc += w * xf[c]
		}
		h := bf16Encode(float32(acc))
		out[r*2], out[r*2+1] = byte(h), byte(h>>8)
	}
	return out, nil
}
