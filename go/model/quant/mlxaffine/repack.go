// SPDX-Licence-Identifier: EUPL-1.2

package mlxaffine

import (
	"encoding/binary"

	core "dappco.re/go"
)

// repack.go bridges MLX affine 1-bit packs to the stock 2-bit device kernels. The shipped
// metallib carries affine_qmv/qmm_t kernels for bits {2,3,4,5,6,8} but NO b_1 kernel, so a
// 1-bit checkpoint (prism-ml Bonsai) cannot dispatch directly. RepackB1ToB2 rewrites the
// packed codes as 2-bit fields with the SAME scales/biases: a b1 code q ∈ {0,1} widened to
// a b2 code q ∈ {0,1} dequantises to the identical value (w = scale·q + bias is unchanged),
// so the repack is EXACT — a zero-quality-change format bridge, not a re-quantisation. The
// cost is 2× packed bytes (still ~1/16 of the f32 weight). A native b_1 kernel in
// lthn_kernels.metallib is a named follow-up; this bridge is what serves Bonsai today.

// RepackB1ToB2 widens a 1-bit MLX affine tensor to the 2-bit layout the stock kernels
// dispatch, returning owned buffers. packed is the b1 codes ([outDim, inDim/32] uint32,
// 32 LSB-first codes per word); scales/biases are the bf16 group parameters ([outDim,
// inDim/groupSize]), carried through UNCHANGED. The returned packed2 is [outDim, inDim/16]
// uint32 (16 codes per word, 2 bits each); scales2/biases2 are copies (owned buffers the
// caller keeps after the source mmap is closed). Dequantising the b2 result byte-for-byte
// reproduces the b1 dequant — gated in repack_test.go.
func RepackB1ToB2(packed, scales, biases []byte, outDim, inDim, groupSize int) (packed2, scales2, biases2 []byte, err error) {
	if outDim <= 0 || inDim <= 0 || groupSize <= 0 || inDim%groupSize != 0 {
		return nil, nil, nil, core.NewError("mlxaffine.RepackB1ToB2: invalid dimensions")
	}
	if inDim%32 != 0 {
		return nil, nil, nil, core.Errorf("mlxaffine.RepackB1ToB2: inDim %d must be a multiple of 32 (b1 packs 32 codes per word)", inDim)
	}
	wordsPerRow1 := PackedWords(inDim, 1) // inDim/32
	wordsPerRow2 := PackedWords(inDim, 2) // inDim/16
	if len(packed) != outDim*wordsPerRow1*4 {
		return nil, nil, nil, core.Errorf("mlxaffine.RepackB1ToB2: packed length %d != %d", len(packed), outDim*wordsPerRow1*4)
	}
	wantSB := outDim * (inDim / groupSize) * 2
	if len(scales) != wantSB || len(biases) != wantSB {
		return nil, nil, nil, core.NewError("mlxaffine.RepackB1ToB2: scales/biases length does not match [outDim, inDim/groupSize]")
	}

	packed2 = make([]byte, outDim*wordsPerRow2*4)
	for r := 0; r < outDim; r++ {
		for j := 0; j < inDim; j++ {
			w1 := binary.LittleEndian.Uint32(packed[(r*wordsPerRow1+j/32)*4:])
			code := (w1 >> uint(j%32)) & 1
			if code == 0 {
				continue // packed2 starts zeroed — a 0 code needs no write
			}
			off := (r*wordsPerRow2 + j/16) * 4
			word := binary.LittleEndian.Uint32(packed2[off:]) | (code << (uint(j%16) * 2))
			binary.LittleEndian.PutUint32(packed2[off:], word)
		}
	}
	// The group parameters are invariant under the code widening; copy so the caller owns
	// buffers that outlive the source mmap.
	scales2 = append([]byte(nil), scales...)
	biases2 = append([]byte(nil), biases...)
	return packed2, scales2, biases2, nil
}
