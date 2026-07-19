// SPDX-Licence-Identifier: EUPL-1.2

package turboquant

// packBits packs values (each in [0, 1<<bits)) into a little-endian bit
// stream: value i's bits occupy stream positions [i*bits, i*bits+bits), LSB
// first, spanning byte boundaries as needed. bits == 0 packs every value to
// nothing (used by Q_prod's degenerate b=1 stage, whose stage-1 bit-width is
// b-1 == 0 — a single Lloyd-Max level, so there is no index to store).
//
//	packBits([]int{5, 2}, 3) // []byte{0b010_101} = {0x15}
func packBits(values []int, bits int) []byte {
	if bits == 0 || len(values) == 0 {
		return nil
	}
	total := len(values) * bits
	out := make([]byte, (total+7)/8)
	pos := 0
	for _, v := range values {
		for b := 0; b < bits; b++ {
			if v&(1<<uint(b)) != 0 {
				out[pos/8] |= 1 << uint(pos%8)
			}
			pos++
		}
	}
	return out
}

// unpackBits reverses packBits, reading n values of bits width each. bits ==
// 0 returns n zeros without touching data. A data slice shorter than
// required is treated as zero-padded (out-of-range reads return 0 bits)
// rather than panicking — the packed payload's own length is the source of
// truth for how many bytes exist, and callers already know n from the row
// dimension.
//
//	unpackBits([]byte{0x15}, 2, 3) // []int{5, 2}
func unpackBits(data []byte, n, bits int) []int {
	out := make([]int, n)
	if bits == 0 {
		return out
	}
	pos := 0
	for i := 0; i < n; i++ {
		var v int
		for b := 0; b < bits; b++ {
			byteIdx := pos / 8
			if byteIdx < len(data) && data[byteIdx]&(1<<uint(pos%8)) != 0 {
				v |= 1 << uint(b)
			}
			pos++
		}
		out[i] = v
	}
	return out
}

// packedByteLen returns the byte length packBits(values, bits) produces for
// n values of the given bit width, without materialising the values —
// exactly the arithmetic Codec.BytesPerRow implementations need.
//
//	packedByteLen(128, 3) // 48
func packedByteLen(n, bits int) int {
	if bits == 0 || n == 0 {
		return 0
	}
	total := n * bits
	return (total + 7) / 8
}

// packSigns packs a slice of signs (true = positive/set bit) 1 bit per
// value, LSB first — the QJL sign-bit encoding (RFC #41 Q_prod stage 2).
//
//	packSigns([]bool{true, false, true}) // []byte{0b101} = {0x05}
func packSigns(signs []bool) []byte {
	values := make([]int, len(signs))
	for i, s := range signs {
		if s {
			values[i] = 1
		}
	}
	return packBits(values, 1)
}

// unpackSigns reverses packSigns into ±1 float64 values (the form the QJL
// reconstruction sum consumes directly) — a set bit decodes to +1, a clear
// bit to -1.
//
//	unpackSigns([]byte{0x05}, 3) // []float64{1, -1, 1}
func unpackSigns(data []byte, n int) []float64 {
	bits := unpackBits(data, n, 1)
	out := make([]float64, n)
	for i, b := range bits {
		if b != 0 {
			out[i] = 1
		} else {
			out[i] = -1
		}
	}
	return out
}
