// SPDX-Licence-Identifier: EUPL-1.2

package gguf

import (
	"encoding/binary"
	"encoding/hex"
	"math"
	"testing"
)

// --- shared test-only helpers -----------------------------------------
//
// None of these formats have a production dequantiser in this package
// (go-mlx's gguf/ reference package only ever WRITES GGUF — see the
// package doc comment) so correctness of the quantise kernels is
// verified here against hand-derived decoders mirroring the bit layouts
// documented on each quantizeQ*_K function.

// testFloat16Decode converts one IEEE-754 binary16 bit pattern to
// float32 — the inverse of the production float32ToFloat16.
func testFloat16Decode(bits uint16) float32 {
	sign := uint32(bits>>15) & 0x1
	exp := int((bits >> 10) & 0x1f)
	frac := uint32(bits & 0x03ff)
	switch {
	case exp == 0 && frac == 0:
		return math.Float32frombits(sign << 31)
	case exp == 0:
		for frac&0x0400 == 0 {
			frac <<= 1
			exp--
		}
		exp++
		frac &= 0x03ff
	case exp == 31:
		return math.Float32frombits((sign << 31) | 0x7f800000 | (frac << 13))
	}
	exp += 127 - 15
	return math.Float32frombits((sign << 31) | (uint32(exp) << 23) | (frac << 13))
}

// testUnpackQ3KScale reverses packQ3KScales for sub-block index j (0..15).
func testUnpackQ3KScale(packed [12]byte, j int) uint8 {
	var lo uint8
	if j < 8 {
		lo = packed[j] & 0xF
	} else {
		lo = packed[j-8] >> 4
	}
	hi := (packed[8+j%4] >> (2 * (j / 4))) & 3
	return lo | (hi << 4)
}

func absFloat32Diff(a, b float32) float32 {
	d := a - b
	if d < 0 {
		return -d
	}
	return d
}

func maxAbsFloat32Slice(values []float32) float32 {
	var m float32
	for _, v := range values {
		if v < 0 {
			v = -v
		}
		if v > m {
			m = v
		}
	}
	return m
}

// rampBlock returns count deterministic, non-trivial values spanning a
// wide dynamic range — enough to exercise both the positive and
// negative quantiser branches.
func rampBlock(count int) []float32 {
	values := make([]float32, count)
	for i := range values {
		values[i] = float32(i-count/2) / float32(count) * 4
	}
	return values
}

// noisyBlock returns count deterministic, zero-centred values with high
// local (within any 16-element window) sign variance — representative of
// real per-tensor weight distributions. Some K-quant formats (Q2_K) pair
// a per-sub-block affine min with a block-global scale-of-mins that is
// only ever non-negative; a sub-block whose local window happens to sit
// entirely on one side of zero (as rampBlock's monotonic run does for
// several of its 16-element windows) is a pathological input for that
// scheme, not a porting bug — noisyBlock avoids it the way a real weight
// tensor would.
func noisyBlock(count int) []float32 {
	values := make([]float32, count)
	for i := range values {
		values[i] = float32(math.Sin(float64(i)*0.9)+0.3*math.Sin(float64(i)*3.1)) * 0.8
	}
	return values
}

// --- Q8_0 ---------------------------------------------------------------

func testDequantQ8_0(data []byte) []float32 {
	const blockBytes = 34
	nBlocks := len(data) / blockBytes
	out := make([]float32, 0, nBlocks*32)
	for b := range nBlocks {
		block := data[b*blockBytes:]
		scale := testFloat16Decode(binary.LittleEndian.Uint16(block[0:2]))
		for i := range 32 {
			out = append(out, scale*float32(int8(block[2+i])))
		}
	}
	return out
}

func TestQuantizeKernels_QuantizeQ8_0_Good(t *testing.T) {
	values := rampBlock(32)
	data := quantizeQ8_0(values)
	if len(data) != 34 {
		t.Fatalf("len(data) = %d, want 34", len(data))
	}
	decoded := testDequantQ8_0(data)
	step := maxAbsFloat32Slice(values) / 127
	for i, want := range values {
		if got := absFloat32Diff(decoded[i], want); got > step*1.5 {
			t.Errorf("Q8_0[%d]: decoded %v want ~%v (err %v > tol %v)", i, decoded[i], want, got, step*1.5)
		}
	}
}

func TestQuantizeKernels_QuantizeQ8_0_Bad(t *testing.T) {
	data := quantizeQ8_0(make([]float32, 32)) // all-zero block
	if len(data) != 34 {
		t.Fatalf("len(data) = %d, want 34", len(data))
	}
	for _, b := range data[2:] {
		if b != 0 {
			t.Fatalf("zero-block Q8_0 quant bytes = %v, want all zero", data[2:])
		}
	}
}

// TestQuantizeKernels_QuantizeQ8_0_Ugly pins quantizeQ8_0(rampBlock(32))
// against the byte-exact output of ggml's own quantize_row_q8_0_ref (called
// directly, via cgo-free C harness, against homebrew ggml 0.15.3 —
// libggml-base.dylib's compiled reference implementation, not a
// reimplementation). This is the load-bearing conformance gate: a
// self-consistent round-trip through testDequantQ8_0 above can't catch a bug
// shared between encoder and test decoder, but a real llama.cpp/ggml would
// read these bytes wrong. Q8_0 needed no fix; this pin is the receipt.
func TestQuantizeKernels_QuantizeQ8_0_Ugly(t *testing.T) {
	const wantHex = "082481899199a1a9b1b9c0c8d0d8e0e8f0f8000810182028303840474f575f676f77"
	got := hex.EncodeToString(quantizeQ8_0(rampBlock(32)))
	if got != wantHex {
		t.Errorf("quantizeQ8_0(rampBlock(32)) = %s, want ggml-ref %s", got, wantHex)
	}
}

// --- Q4_0 ---------------------------------------------------------------

func testDequantQ4_0(data []byte) []float32 {
	const blockBytes = 18
	nBlocks := len(data) / blockBytes
	out := make([]float32, 0, nBlocks*32)
	for b := range nBlocks {
		block := data[b*blockBytes:]
		scale := testFloat16Decode(binary.LittleEndian.Uint16(block[0:2]))
		packed := block[2:18]
		for i := range 16 {
			out = append(out, scale*(float32(packed[i]&0xF)-8))
		}
		for i := range 16 {
			out = append(out, scale*(float32(packed[i]>>4)-8))
		}
	}
	return out
}

func TestQuantizeKernels_QuantizeQ4_0_Good(t *testing.T) {
	values := rampBlock(32)
	data := quantizeQ4_0(values)
	if len(data) != 18 {
		t.Fatalf("len(data) = %d, want 18", len(data))
	}
	decoded := testDequantQ4_0(data)
	step := maxAbsFloat32Slice(values) / 7
	for i, want := range values {
		if got := absFloat32Diff(decoded[i], want); got > step*1.5 {
			t.Errorf("Q4_0[%d]: decoded %v want ~%v (err %v > tol %v)", i, decoded[i], want, got, step*1.5)
		}
	}
}

func TestQuantizeKernels_QuantizeQ4_0_Bad(t *testing.T) {
	data := quantizeQ4_0(make([]float32, 32))
	for _, b := range data[2:] {
		if b != 0x88 {
			t.Fatalf("zero-block Q4_0 quant bytes = %x, want all 0x88", data[2:])
		}
	}
}

// TestQuantizeKernels_QuantizeQ4_0_Ugly pins quantizeQ4_0(rampBlock(32))
// against ggml's compiled quantize_row_q4_0_ref (see the Q8_0 pin above for
// methodology). Was broken before this fix: the scale used maxAbs/7 (always
// positive, wastes nibble 0) instead of ggml's max/-8 (signed extremal
// value, uses the full [0,15] nibble range) — every non-edge nibble differed.
func TestQuantizeKernels_QuantizeQ4_0_Ugly(t *testing.T) {
	const wantHex = "0034809191a2a2b3b3c4c4d5d5e6e6f7f7f8"
	got := hex.EncodeToString(quantizeQ4_0(rampBlock(32)))
	if got != wantHex {
		t.Errorf("quantizeQ4_0(rampBlock(32)) = %s, want ggml-ref %s", got, wantHex)
	}
}

// --- Q5_0 ---------------------------------------------------------------
//
// block_q5_0 is a SYMMETRIC quantiser like Q4_0 (single scale d, no min
// field) — 22 B/block: 2 d + 4 qh (5th/high bit of each of the 32 elements,
// bit j) + 16 qs (low 4 bits, lower/upper-half interleaved exactly as Q4_0).
// testDequantQ5_0 mirrors ggml's dequantize_row_q5_0.

func testDequantQ5_0(data []byte) []float32 {
	const blockBytes = 22
	nBlocks := len(data) / blockBytes
	out := make([]float32, 0, nBlocks*32)
	for b := range nBlocks {
		block := data[b*blockBytes:]
		scale := testFloat16Decode(binary.LittleEndian.Uint16(block[0:2]))
		qh := binary.LittleEndian.Uint32(block[2:6])
		qs := block[6:22]
		for i := range 16 {
			hi := byte((qh >> uint(i)) & 1)
			q := (qs[i] & 0xF) | hi<<4
			out = append(out, scale*(float32(q)-16))
		}
		for i := range 16 {
			hi := byte((qh >> uint(i+16)) & 1)
			q := (qs[i] >> 4) | hi<<4
			out = append(out, scale*(float32(q)-16))
		}
	}
	return out
}

func TestQuantizeKernels_QuantizeQ5_0_Good(t *testing.T) {
	values := rampBlock(32)
	data := quantizeQ5_0(values)
	if len(data) != 22 {
		t.Fatalf("len(data) = %d, want 22", len(data))
	}
	decoded := testDequantQ5_0(data)
	step := maxAbsFloat32Slice(values) / 16
	for i, want := range values {
		if got := absFloat32Diff(decoded[i], want); got > step*1.5 {
			t.Errorf("Q5_0[%d]: decoded %v want ~%v (err %v > tol %v)", i, decoded[i], want, got, step*1.5)
		}
	}
}

func TestQuantizeKernels_QuantizeQ5_0_Bad(t *testing.T) {
	data := quantizeQ5_0(make([]float32, 32))
	if len(data) != 22 {
		t.Fatalf("len(data) = %d, want 22", len(data))
	}
	decoded := testDequantQ5_0(data)
	for i, v := range decoded {
		if v != 0 {
			t.Fatalf("zero-block Q5_0 decoded[%d] = %v, want 0", i, v)
		}
	}
}

// TestQuantizeKernels_QuantizeQ5_0_Ugly pins quantizeQ5_0(rampBlock(32))
// against ggml's compiled quantize_row_q5_0_ref (see the Q8_0 pin for
// methodology). Was severely broken before this fix: the encoder wrote a
// 24-byte affine (scale+min) block with a custom LSB-first 5-bit bitstream —
// wrong size, wrong scheme, wrong bit layout entirely. Any real GGUF file
// with Q5_0 tensors was byte-corrupt (block size alone didn't match), not
// just numerically off.
func TestQuantizeKernels_QuantizeQ5_0_Ugly(t *testing.T) {
	const wantHex = "00300000ffff00112233445566778899aabbccddeeff"
	got := hex.EncodeToString(quantizeQ5_0(rampBlock(32)))
	if got != wantHex {
		t.Errorf("quantizeQ5_0(rampBlock(32)) = %s, want ggml-ref %s", got, wantHex)
	}
}

// --- Q4_K -----------------------------------------------------------------
//
// testDequantQ4_K decodes a block_q4_K the ggml way: unpack the 8 per-sub-block
// 6-bit scale/min pairs (get_scale_min_k4), unpack the 4-bit levels from the
// 64-element-interleaved qs field, and reconstruct y = d*sc*q - dmin*m. The
// definitive correctness check for the encoder is llama.cpp loading the file;
// this round-trip guards against regressions in Go.

func testDequantQ4_K(data []byte) []float32 {
	const blockBytes = 144
	nBlocks := len(data) / blockBytes
	out := make([]float32, 0, nBlocks*qkBlockSize)
	for b := range nBlocks {
		block := data[b*blockBytes:]
		d := testFloat16Decode(binary.LittleEndian.Uint16(block[0:2]))
		dmin := testFloat16Decode(binary.LittleEndian.Uint16(block[2:4]))
		var packed [12]byte
		copy(packed[:], block[4:16])
		qs := block[16:144]
		var levels [qkBlockSize]uint8
		li := 0
		for j := 0; j < qkBlockSize; j += 64 {
			for l := 0; l < 32; l++ {
				levels[j+l] = qs[li] & 0xF
				levels[j+l+32] = qs[li] >> 4
				li++
			}
		}
		for j := 0; j < kQuantSubBlocks; j++ {
			sc, m := getScaleMinK4(j, &packed)
			dl := d * float32(sc)
			ml := dmin * float32(m)
			for ii := 0; ii < kQuantSubBlockSize; ii++ {
				out = append(out, dl*float32(levels[j*kQuantSubBlockSize+ii])-ml)
			}
		}
	}
	return out
}

func TestQuantizeKernels_QuantizeQ4_K_Good(t *testing.T) {
	values := rampBlock(qkBlockSize)
	data := quantizeQ4_K(values)
	if len(data) != 144 {
		t.Fatalf("len(data) = %d, want 144", len(data))
	}
	decoded := testDequantQ4_K(data)
	lo, hi := values[0], values[0]
	for _, v := range values {
		if v < lo {
			lo = v
		}
		if v > hi {
			hi = v
		}
	}
	step := (hi - lo) / 15
	for i, want := range values {
		if got := absFloat32Diff(decoded[i], want); got > step*1.5+1e-6 {
			t.Errorf("Q4_K[%d]: decoded %v want ~%v (err %v > tol %v)", i, decoded[i], want, got, step*1.5)
		}
	}
}

func TestQuantizeKernels_QuantizeQ4_K_Bad(t *testing.T) {
	// An all-zero block quantises to all-zero levels (scale 0, min 0), so both
	// the packed scales and the qs field are zero.
	data := quantizeQ4_K(make([]float32, qkBlockSize))
	for _, b := range data[4:144] {
		if b != 0x00 {
			t.Fatalf("zero-block Q4_K scale+quant bytes = %x, want all zero", data[4:144])
		}
	}
}

// TestQuantizeKernels_QuantizeQ4_K_Ugly pins quantizeQ4_K(rampBlock(256))
// against ggml's compiled quantize_row_q4_K_ref (see the Q8_0 pin for
// methodology). This format was already fixed to the ggml reference
// algorithm (makeQKX2Quants + get_scale_min_k4) before this pin existed;
// the pin turns that self-consistent round-trip test into a real
// conformance receipt against the compiled reference, not just proof the
// encoder and the in-repo test decoder agree with each other.
func TestQuantizeKernels_QuantizeQ4_K_Ugly(t *testing.T) {
	const wantHex = "32180c284f4fcfcf3f2f1f10000f000f0000111122223333444455556666777788889999aaaabbbbccccddddeeeeffff00101021213232434354546565767687879899a9aababbcbccdcddedeefeffff80808191929293a3a4a4a5b5b6b6b7c7c8c8c9d9dadadaebebececfdfdfefeffcacacacbcbcbcbcbdbdcdcdcdcdcdcddedededededeeeeeefefefefeffffffff"
	got := hex.EncodeToString(quantizeQ4_K(rampBlock(qkBlockSize)))
	if got != wantHex {
		t.Errorf("quantizeQ4_K(rampBlock(256)) = %s, want ggml-ref %s", got, wantHex)
	}
}

// --- Q5_K -----------------------------------------------------------------

func testDequantQ5_K(data []byte) []float32 {
	const blockBytes = 176
	nBlocks := len(data) / blockBytes
	out := make([]float32, 0, nBlocks*qkBlockSize)
	for b := range nBlocks {
		block := data[b*blockBytes:]
		d := testFloat16Decode(binary.LittleEndian.Uint16(block[0:2]))
		dmin := testFloat16Decode(binary.LittleEndian.Uint16(block[2:4]))
		var packed [12]byte
		copy(packed[:], block[4:16])
		qh := block[16:48]
		qs := block[48:176]
		var levels [qkBlockSize]uint8
		qi := 0
		u1, u2 := uint8(1), uint8(2)
		for n := 0; n < qkBlockSize; n += 64 {
			for j := 0; j < 32; j++ {
				v1 := qs[qi+j] & 0xF
				v2 := qs[qi+j] >> 4
				if qh[j]&u1 != 0 {
					v1 += 16
				}
				if qh[j]&u2 != 0 {
					v2 += 16
				}
				levels[n+j] = v1
				levels[n+j+32] = v2
			}
			u1 <<= 2
			u2 <<= 2
			qi += 32
		}
		for j := 0; j < kQuantSubBlocks; j++ {
			sc, m := getScaleMinK4(j, &packed)
			dl := d * float32(sc)
			ml := dmin * float32(m)
			for ii := 0; ii < kQuantSubBlockSize; ii++ {
				out = append(out, dl*float32(levels[j*kQuantSubBlockSize+ii])-ml)
			}
		}
	}
	return out
}

func TestQuantizeKernels_QuantizeQ5_K_Good(t *testing.T) {
	values := rampBlock(qkBlockSize)
	data := quantizeQ5_K(values)
	if len(data) != 176 {
		t.Fatalf("len(data) = %d, want 176", len(data))
	}
	decoded := testDequantQ5_K(data)
	lo, hi := values[0], values[0]
	for _, v := range values {
		if v < lo {
			lo = v
		}
		if v > hi {
			hi = v
		}
	}
	step := (hi - lo) / 31
	for i, want := range values {
		if got := absFloat32Diff(decoded[i], want); got > step*1.5+1e-6 {
			t.Errorf("Q5_K[%d]: decoded %v want ~%v (err %v > tol %v)", i, decoded[i], want, got, step*1.5)
		}
	}
}

// TestQuantizeKernels_QuantizeQ5_K_Ugly pins quantizeQ5_K(rampBlock(256))
// against ggml's compiled quantize_row_q5_K_ref (see the Q8_0 pin for
// methodology). This format was already fixed to the ggml reference
// algorithm before this pin existed; see the Q4_K pin for why the pin still
// matters on top of the existing round-trip test.
func TestQuantizeKernels_QuantizeQ5_K_Ugly(t *testing.T) {
	const wantHex = "1e141028505090d03f2f2010000f0f0fe0e0e0e0e0e0e0e0e0e0e0e0e0e0e0e4fdffffffffffffffffffffffffffffff000112233445566778899aabbccddeeff000112233445566778899aabbccddee0112233445566778899aabbccddeeff001112233445566778899aabbccddeeff001112232435364748595a6b6c7d7e8f809191a2a3b4b5c6c7d8d9eaebfcfdfe8585868696979797a8a8a8a9b9b9babacacbcbcbdcdcdcddededeeeefeffffff"
	got := hex.EncodeToString(quantizeQ5_K(rampBlock(qkBlockSize)))
	if got != wantHex {
		t.Errorf("quantizeQ5_K(rampBlock(256)) = %s, want ggml-ref %s", got, wantHex)
	}
}

// --- Q6_K -----------------------------------------------------------------

func testDequantQ6_K(data []byte) []float32 {
	const blockBytes = 210
	nBlocks := len(data) / blockBytes
	out := make([]float32, 0, nBlocks*qkBlockSize)
	for b := range nBlocks {
		block := data[b*blockBytes:]
		ql := block[0:128]
		qh := block[128:192]
		scales := block[192:208]
		d := testFloat16Decode(binary.LittleEndian.Uint16(block[208:210]))
		var levels [qkBlockSize]byte
		for n := 0; n < qkBlockSize; n += 128 {
			for l := range 32 {
				q1 := ql[n/2+l] & 0xF
				q3 := ql[n/2+l] >> 4
				q2 := ql[n/2+l+32] & 0xF
				q4 := ql[n/2+l+32] >> 4
				qhByte := qh[n/4+l]
				levels[n+l] = q1 | ((qhByte & 0x3) << 4)
				levels[n+l+32] = q2 | (((qhByte >> 2) & 0x3) << 4)
				levels[n+l+64] = q3 | (((qhByte >> 4) & 0x3) << 4)
				levels[n+l+96] = q4 | (((qhByte >> 6) & 0x3) << 4)
			}
		}
		for i := range qkBlockSize {
			sub := i / 16
			scale := int8(scales[sub])
			out = append(out, d*float32(scale)*(float32(levels[i])-32))
		}
	}
	return out
}

func TestQuantizeKernels_QuantizeQ6_K_Good(t *testing.T) {
	values := rampBlock(qkBlockSize)
	data := quantizeQ6_K(values)
	if len(data) != 210 {
		t.Fatalf("len(data) = %d, want 210", len(data))
	}
	decoded := testDequantQ6_K(data)
	maxAbs := maxAbsFloat32Slice(values)
	tol := maxAbs/32/32 + maxAbs/32 // sub-scale step + scale-of-scales quantisation slack
	for i, want := range values {
		if got := absFloat32Diff(decoded[i], want); got > tol {
			t.Errorf("Q6_K[%d]: decoded %v want ~%v (err %v > tol %v)", i, decoded[i], want, got, tol)
		}
	}
}

func TestQuantizeKernels_QuantizeQ6_K_Bad(t *testing.T) {
	data := quantizeQ6_K(make([]float32, qkBlockSize))
	decoded := testDequantQ6_K(data)
	for i, v := range decoded {
		if v != 0 {
			t.Fatalf("zero-block Q6_K decoded[%d] = %v, want 0", i, v)
		}
	}
}

// TestQuantizeKernels_QuantizeQ6_K_Ugly pins quantizeQ6_K(rampBlock(256))
// against ggml's compiled quantize_row_q6_K_ref (see the Q8_0 pin for
// methodology). Was broken before this fix: each sub-block's scale was a
// naive maxAbs/32 (always non-negative) instead of ggml's make_qx_quants
// least-squares fit, and the scale-of-scales used 127/maxScale (positive
// only) instead of -128/maxScale (signed, hits the int8 range's edge
// exactly) — every scale byte and every derived quant level differed.
func TestQuantizeKernels_QuantizeQ6_K_Ugly(t *testing.T) {
	const wantHex = "00001011212131324242525363637374001010213131425252627373839494a400102131415262728393a3b4c4d4e5f50020416181a2c2e30323446485a5c5e6606e5c5a474543313f2d2b1816140200505f4e4d4c3b3a3928272615141302014b4a4a3938382726262514141302020148373736362525242413131212010100000000000000000000000000000000000000000000000000404040404040404002010101010101010000000000000000010000000000000000000000000000008090a0b0c0d0e0f00f2031404f5f6f7ff78f"
	got := hex.EncodeToString(quantizeQ6_K(rampBlock(qkBlockSize)))
	if got != wantHex {
		t.Errorf("quantizeQ6_K(rampBlock(256)) = %s, want ggml-ref %s", got, wantHex)
	}
}

// --- Q3_K -----------------------------------------------------------------

func testDequantQ3_K(data []byte) []float32 {
	const blockBytes = 110
	nBlocks := len(data) / blockBytes
	out := make([]float32, 0, nBlocks*qkBlockSize)
	for b := range nBlocks {
		block := data[b*blockBytes:]
		hmask := block[0:32]
		qs := block[32:96]
		var packedScales [12]byte
		copy(packedScales[:], block[96:108])
		d := testFloat16Decode(binary.LittleEndian.Uint16(block[108:110]))
		for i := range qkBlockSize {
			is := i / 16
			l := i % 16
			bitIndex := is / 2
			byteOffset := 16 * (is % 2)
			highBit := (hmask[byteOffset+l] >> bitIndex) & 1

			n := (i / 128) * 128
			p := i - n
			byteBase := n / 4
			bIdx := p % 32
			shift := 2 * (p / 32)
			low2 := (qs[byteBase+bIdx] >> shift) & 3

			level := low2 | (highBit << 2) // 0..7
			scaleCode := testUnpackQ3KScale(packedScales, is)
			subScale := d * float32(int(scaleCode)-32)
			out = append(out, subScale*(float32(level)-4))
		}
	}
	return out
}

func TestQuantizeKernels_QuantizeQ3_K_Good(t *testing.T) {
	values := rampBlock(qkBlockSize)
	data := quantizeQ3_K(values)
	if len(data) != 110 {
		t.Fatalf("len(data) = %d, want 110", len(data))
	}
	decoded := testDequantQ3_K(data)
	maxAbs := maxAbsFloat32Slice(values)
	tol := maxAbs/4/4 + maxAbs/4
	for i, want := range values {
		if got := absFloat32Diff(decoded[i], want); got > tol {
			t.Errorf("Q3_K[%d]: decoded %v want ~%v (err %v > tol %v)", i, decoded[i], want, got, tol)
		}
	}
}

func TestQuantizeKernels_QuantizeQ3_K_Bad(t *testing.T) {
	data := quantizeQ3_K(make([]float32, qkBlockSize))
	decoded := testDequantQ3_K(data)
	for i, v := range decoded {
		if v != 0 {
			t.Fatalf("zero-block Q3_K decoded[%d] = %v, want 0", i, v)
		}
	}
}

// TestQuantizeKernels_QuantizeQ3_K_Ugly pins quantizeQ3_K(rampBlock(256))
// against ggml's compiled quantize_row_q3_K_ref (see the Q8_0 pin for
// methodology). Was broken before this fix on two independent axes: the
// per-sub-block scale was a naive maxAbs/4 instead of ggml's make_q3_quants
// coordinate-descent fit, and hmask packing used a convoluted per-sub-block
// scheme instead of ggml's simple sequential walk (hmask[j%32] |=
// 1<<(j/32) over all 256 elements) — the hmask bytes differed from the very
// first byte.
func TestQuantizeKernels_QuantizeQ3_K_Ugly(t *testing.T) {
	const wantHex = "1010000000000000000000000000000000000000000000000000000000000008000000000000404040404040505090900000004040404080809090d0d0d0d01014141717171706060202010101010000060606050505050505010100000000004084c81d5084c8fce4e4e4f48ea3"
	got := hex.EncodeToString(quantizeQ3_K(rampBlock(qkBlockSize)))
	if got != wantHex {
		t.Errorf("quantizeQ3_K(rampBlock(256)) = %s, want ggml-ref %s", got, wantHex)
	}
}

// --- Q2_K -----------------------------------------------------------------

func testDequantQ2_K(data []byte) []float32 {
	const blockBytes = 84
	nBlocks := len(data) / blockBytes
	out := make([]float32, 0, nBlocks*qkBlockSize)
	for b := range nBlocks {
		block := data[b*blockBytes:]
		scales := block[0:16]
		qs := block[16:80]
		d := testFloat16Decode(binary.LittleEndian.Uint16(block[80:82]))
		dmin := testFloat16Decode(binary.LittleEndian.Uint16(block[82:84]))
		for i := range qkBlockSize {
			sub := i / 16
			scEnc := scales[sub] & 0xF
			mnEnc := scales[sub] >> 4
			sc := d * float32(scEnc)
			ml := dmin * float32(mnEnc)

			n := (i / 128) * 128
			p := i - n
			byteBase := n / 4
			bIdx := p % 32
			shift := 2 * (p / 32)
			q := (qs[byteBase+bIdx] >> shift) & 3

			out = append(out, sc*float32(q)-ml)
		}
	}
	return out
}

func TestQuantizeKernels_QuantizeQ2_K_Good(t *testing.T) {
	values := noisyBlock(qkBlockSize)
	data := quantizeQ2_K(values)
	if len(data) != 84 {
		t.Fatalf("len(data) = %d, want 84", len(data))
	}
	decoded := testDequantQ2_K(data)

	// Q2_K is a coarse 2-bit-per-element quantiser (only 4 levels per
	// 16-element sub-block), so a per-element bound is too strict to be
	// meaningful — judge it the way a lossy quantiser is actually
	// evaluated: relative RMS error across the block. A generous 35%
	// bound still catches genuine encode/decode bugs (the pre-fix
	// monotonic-ramp regression measured error many multiples of the
	// signal itself) without demanding better precision than 2 bits
	// can deliver.
	var sumSq, errSq float64
	for i, want := range values {
		sumSq += float64(want) * float64(want)
		diff := float64(decoded[i]) - float64(want)
		errSq += diff * diff
	}
	rmsSignal := math.Sqrt(sumSq / float64(len(values)))
	rmsError := math.Sqrt(errSq / float64(len(values)))
	if relative := rmsError / rmsSignal; relative > 0.35 {
		t.Errorf("Q2_K relative RMS error = %.3f, want <= 0.35 (rmsSignal=%v rmsError=%v)", relative, rmsSignal, rmsError)
	}
}

func TestQuantizeKernels_QuantizeQ2_K_Bad(t *testing.T) {
	data := quantizeQ2_K(make([]float32, qkBlockSize))
	decoded := testDequantQ2_K(data)
	for i, v := range decoded {
		if v != 0 {
			t.Fatalf("zero-block Q2_K decoded[%d] = %v, want 0", i, v)
		}
	}
}

// TestQuantizeKernels_QuantizeQ2_K_Ugly pins quantizeQ2_K(rampBlock(256))
// against ggml's compiled quantize_row_q2_K_ref (see the Q8_0 pin for
// methodology). Was broken before this fix: the per-sub-block (scale, min)
// was a naive (max-min)/3 affine fit instead of ggml's make_qkx2_quants —
// the same optimal-fit search already ported for Q4_K/Q5_K, called here
// with Q2_K's own parameters (nmax=3, plain |x| weights, useMad=true).
func TestQuantizeKernels_QuantizeQ2_K_Ugly(t *testing.T) {
	const wantHex = "f1d1b1917161412102030507090b0e0f404040818185c6d6dadbebefffffffff10506060a1b1f5f6f6fafbfffffffffff8f8f8fdfdfdfdfdfdfefefefefefffffefefefefeffffffffffffffffffffff7b293830"
	got := hex.EncodeToString(quantizeQ2_K(rampBlock(qkBlockSize)))
	if got != wantHex {
		t.Errorf("quantizeQ2_K(rampBlock(256)) = %s, want ggml-ref %s", got, wantHex)
	}
}

// --- Q8_K -----------------------------------------------------------------

func testDequantQ8_K(data []byte) ([]float32, []int16) {
	const blockBytes = 292
	nBlocks := len(data) / blockBytes
	out := make([]float32, 0, nBlocks*qkBlockSize)
	var bsums []int16
	for b := range nBlocks {
		block := data[b*blockBytes:]
		d := math.Float32frombits(binary.LittleEndian.Uint32(block[0:4]))
		qs := block[4:260]
		for i := range qkBlockSize {
			out = append(out, d*float32(int8(qs[i])))
		}
		for sb := range qkSubBlocks {
			bsums = append(bsums, int16(binary.LittleEndian.Uint16(block[260+sb*2:262+sb*2])))
		}
	}
	return out, bsums
}

func TestQuantizeKernels_QuantizeQ8_K_Good(t *testing.T) {
	values := rampBlock(qkBlockSize)
	data := quantizeQ8_K(values)
	if len(data) != 292 {
		t.Fatalf("len(data) = %d, want 292", len(data))
	}
	decoded, bsums := testDequantQ8_K(data)
	step := maxAbsFloat32Slice(values) / 127
	for i, want := range values {
		if got := absFloat32Diff(decoded[i], want); got > step*1.5 {
			t.Errorf("Q8_K[%d]: decoded %v want ~%v (err %v > tol %v)", i, decoded[i], want, got, step*1.5)
		}
	}
	// bsums[sb] must equal the sum of the 16 signed quants in that group.
	for sb := range qkSubBlocks {
		var want int16
		for j := range 16 {
			idx := sb*16 + j
			d := math.Float32frombits(binary.LittleEndian.Uint32(data[0:4]))
			q := int16(math.Round(float64(decoded[idx] / d)))
			want += q
		}
		if bsums[sb] != want {
			t.Errorf("Q8_K bsums[%d] = %d, want %d", sb, bsums[sb], want)
		}
	}
}

func TestQuantizeKernels_QuantizeQ8_K_Bad(t *testing.T) {
	data := quantizeQ8_K(make([]float32, qkBlockSize))
	decoded, bsums := testDequantQ8_K(data)
	for i, v := range decoded {
		if v != 0 {
			t.Fatalf("zero-block Q8_K decoded[%d] = %v, want 0", i, v)
		}
	}
	for sb, sum := range bsums {
		if sum != 0 {
			t.Fatalf("zero-block Q8_K bsums[%d] = %d, want 0", sb, sum)
		}
	}
}

// TestQuantizeKernels_QuantizeQ8_K_Ugly pins quantizeQ8_K(rampBlock(256))
// against ggml's compiled quantize_row_q8_K_ref (see the Q8_0 pin for
// methodology). Q8_K is never itself a GGUF tensor storage type — llama.cpp
// only produces it as a matmul-intermediate quantisation of activations, not
// a file format (llama-quantize's allowed-type list has no Q8_K entry) — but
// the reference function is real and exported, so the same byte pin applies.
func TestQuantizeKernels_QuantizeQ8_K_Ugly(t *testing.T) {
	const wantHex = "0402813c8182838485868788898a8b8c8d8e8f909192939495969798999a9b9c9d9e9fa0a1a2a3a4a5a6a7a8a9aaabacadaeafb0b1b2b3b4b5b6b7b8b9babbbcbdbebfc0c0c1c2c3c4c5c6c7c8c9cacbcccdcecfd0d1d2d3d4d5d6d7d8d9dadbdcdddedfe0e1e2e3e4e5e6e7e8e9eaebecedeeeff0f1f2f3f4f5f6f7f8f9fafbfcfdfeff000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f202122232425262728292a2b2c2d2e2f303132333435363738393a3b3c3d3e3f40404142434445464748494a4b4c4d4e4f505152535455565758595a5b5c5d5e5f606162636465666768696a6b6c6d6e6f707172737475767778797a7b7c7d7e88f888f988fa88fb78fc78fd78fe78ff78007801780278036904680568066807"
	got := hex.EncodeToString(quantizeQ8_K(rampBlock(qkBlockSize)))
	if got != wantHex {
		t.Errorf("quantizeQ8_K(rampBlock(256)) = %s, want ggml-ref %s", got, wantHex)
	}
}

// --- shared numeric helpers ------------------------------------------------

func TestQuantizeKernels_Float32ToFloat16_Good(t *testing.T) {
	cases := []float32{0, 1, -1, 0.5, -0.5, 1.5, 100, -100, 3.25}
	for _, want := range cases {
		bits := float32ToFloat16(want)
		got := testFloat16Decode(bits)
		if absFloat32Diff(got, want) > 0.01 {
			t.Errorf("float32ToFloat16(%v) round-trip = %v, want ~%v", want, got, want)
		}
	}
}

func TestQuantizeKernels_Float32ToFloat16_Ugly(t *testing.T) {
	if got := testFloat16Decode(float32ToFloat16(0)); got != 0 {
		t.Errorf("float32ToFloat16(0) round-trip = %v, want 0", got)
	}
}

func TestQuantizeKernels_MaxAbsFloat32_Good(t *testing.T) {
	if got := maxAbsFloat32([]float32{1, -5, 3, -2}); got != 5 {
		t.Errorf("maxAbsFloat32 = %v, want 5", got)
	}
	// Exercise the scalar tail (len not a multiple of 4).
	if got := maxAbsFloat32([]float32{1, -2, 3, -4, 9}); got != 9 {
		t.Errorf("maxAbsFloat32(tail) = %v, want 9", got)
	}
}

func TestQuantizeKernels_ClampInt_Good(t *testing.T) {
	if got := clampInt(5, 0, 10); got != 5 {
		t.Errorf("clampInt(5,0,10) = %d, want 5", got)
	}
	if got := clampInt(-5, 0, 10); got != 0 {
		t.Errorf("clampInt(-5,0,10) = %d, want 0", got)
	}
	if got := clampInt(50, 0, 10); got != 10 {
		t.Errorf("clampInt(50,0,10) = %d, want 10", got)
	}
}
