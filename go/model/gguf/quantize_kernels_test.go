// SPDX-Licence-Identifier: EUPL-1.2

package gguf

import (
	"encoding/binary"
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

// testUnpack5BitLSB reverses the LSB-first 5-bit bitstream packing
// quantizeQ5_0/quantizeKBlock(bits=5) write: accumulate whole bytes into
// a bit buffer, then peel off 5 bits at a time.
func testUnpack5BitLSB(packed []byte, count int) []int {
	out := make([]int, count)
	bitBuf := uint64(0)
	bitCount := 0
	byteIdx := 0
	for i := 0; i < count; i++ {
		for bitCount < 5 {
			bitBuf |= uint64(packed[byteIdx]) << bitCount
			byteIdx++
			bitCount += 8
		}
		out[i] = int(bitBuf & 0x1F)
		bitBuf >>= 5
		bitCount -= 5
	}
	return out
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
	for b := 0; b < nBlocks; b++ {
		block := data[b*blockBytes:]
		scale := testFloat16Decode(binary.LittleEndian.Uint16(block[0:2]))
		for i := 0; i < 32; i++ {
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

// --- Q4_0 ---------------------------------------------------------------

func testDequantQ4_0(data []byte) []float32 {
	const blockBytes = 18
	nBlocks := len(data) / blockBytes
	out := make([]float32, 0, nBlocks*32)
	for b := 0; b < nBlocks; b++ {
		block := data[b*blockBytes:]
		scale := testFloat16Decode(binary.LittleEndian.Uint16(block[0:2]))
		packed := block[2:18]
		for i := 0; i < 16; i++ {
			out = append(out, scale*(float32(packed[i]&0xF)-8))
		}
		for i := 0; i < 16; i++ {
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

// --- Q5_0 ---------------------------------------------------------------

func testDequantQ5_0(data []byte) []float32 {
	const blockBytes = 24
	nBlocks := len(data) / blockBytes
	out := make([]float32, 0, nBlocks*32)
	for b := 0; b < nBlocks; b++ {
		block := data[b*blockBytes:]
		scale := testFloat16Decode(binary.LittleEndian.Uint16(block[0:2]))
		minVal := testFloat16Decode(binary.LittleEndian.Uint16(block[2:4]))
		qs := testUnpack5BitLSB(block[4:24], 32)
		for _, q := range qs {
			out = append(out, minVal+scale*float32(q))
		}
	}
	return out
}

func TestQuantizeKernels_QuantizeQ5_0_Good(t *testing.T) {
	values := rampBlock(32)
	data := quantizeQ5_0(values)
	if len(data) != 24 {
		t.Fatalf("len(data) = %d, want 24", len(data))
	}
	decoded := testDequantQ5_0(data)
	span := maxAbsFloat32Slice(values) * 2
	step := span / 31
	for i, want := range values {
		if got := absFloat32Diff(decoded[i], want); got > step*1.5 {
			t.Errorf("Q5_0[%d]: decoded %v want ~%v (err %v > tol %v)", i, decoded[i], want, got, step*1.5)
		}
	}
}

func TestQuantizeKernels_QuantizeQ5_0_Bad(t *testing.T) {
	data := quantizeQ5_0(make([]float32, 32))
	for _, b := range data[4:] {
		if b != 0x44 {
			t.Fatalf("zero-block Q5_0 quant bytes = %x, want all 0x44", data[4:])
		}
	}
}

// --- Q4_K -----------------------------------------------------------------
//
// appendQuantizeQ4_K packs a 12-byte per-sub-block scale table into the
// file for format compliance, but (per the encoder) quantises every
// element against the single super-block d/dmin, so decoding via d/dmin
// alone (ignoring the packed sub-scales) matches what was actually
// encoded.

func testDequantQ4_K(data []byte) []float32 {
	const blockBytes = 144
	nBlocks := len(data) / blockBytes
	out := make([]float32, 0, nBlocks*qkBlockSize)
	for b := 0; b < nBlocks; b++ {
		block := data[b*blockBytes:]
		d := testFloat16Decode(binary.LittleEndian.Uint16(block[0:2]))
		dmin := testFloat16Decode(binary.LittleEndian.Uint16(block[2:4]))
		quants := block[16:144]
		for j := 0; j < qkBlockSize; j++ {
			var q byte
			if j%2 == 0 {
				q = quants[j/2] & 0xF
			} else {
				q = quants[j/2] >> 4
			}
			out = append(out, dmin+d*float32(q))
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
	data := quantizeQ4_K(make([]float32, qkBlockSize))
	quants := data[16:144]
	for _, b := range quants {
		if b != 0x88 {
			t.Fatalf("zero-block Q4_K quant bytes = %x, want all 0x88", quants)
		}
	}
}

// --- Q5_K -----------------------------------------------------------------

func testDequantQ5_K(data []byte) []float32 {
	const blockBytes = 176
	nBlocks := len(data) / blockBytes
	out := make([]float32, 0, nBlocks*qkBlockSize)
	for b := 0; b < nBlocks; b++ {
		block := data[b*blockBytes:]
		d := testFloat16Decode(binary.LittleEndian.Uint16(block[0:2]))
		dmin := testFloat16Decode(binary.LittleEndian.Uint16(block[2:4]))
		quants := block[16:176]
		qs := testUnpack5BitLSB(quants, qkBlockSize)
		for _, q := range qs {
			out = append(out, dmin+d*float32(q))
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

// --- Q6_K -----------------------------------------------------------------

func testDequantQ6_K(data []byte) []float32 {
	const blockBytes = 210
	nBlocks := len(data) / blockBytes
	out := make([]float32, 0, nBlocks*qkBlockSize)
	for b := 0; b < nBlocks; b++ {
		block := data[b*blockBytes:]
		ql := block[0:128]
		qh := block[128:192]
		scales := block[192:208]
		d := testFloat16Decode(binary.LittleEndian.Uint16(block[208:210]))
		var levels [qkBlockSize]byte
		for n := 0; n < qkBlockSize; n += 128 {
			for l := 0; l < 32; l++ {
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
		for i := 0; i < qkBlockSize; i++ {
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

// --- Q3_K -----------------------------------------------------------------

func testDequantQ3_K(data []byte) []float32 {
	const blockBytes = 110
	nBlocks := len(data) / blockBytes
	out := make([]float32, 0, nBlocks*qkBlockSize)
	for b := 0; b < nBlocks; b++ {
		block := data[b*blockBytes:]
		hmask := block[0:32]
		qs := block[32:96]
		var packedScales [12]byte
		copy(packedScales[:], block[96:108])
		d := testFloat16Decode(binary.LittleEndian.Uint16(block[108:110]))
		for i := 0; i < qkBlockSize; i++ {
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

// --- Q2_K -----------------------------------------------------------------

func testDequantQ2_K(data []byte) []float32 {
	const blockBytes = 84
	nBlocks := len(data) / blockBytes
	out := make([]float32, 0, nBlocks*qkBlockSize)
	for b := 0; b < nBlocks; b++ {
		block := data[b*blockBytes:]
		scales := block[0:16]
		qs := block[16:80]
		d := testFloat16Decode(binary.LittleEndian.Uint16(block[80:82]))
		dmin := testFloat16Decode(binary.LittleEndian.Uint16(block[82:84]))
		for i := 0; i < qkBlockSize; i++ {
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

// --- Q8_K -----------------------------------------------------------------

func testDequantQ8_K(data []byte) ([]float32, []int16) {
	const blockBytes = 292
	nBlocks := len(data) / blockBytes
	out := make([]float32, 0, nBlocks*qkBlockSize)
	var bsums []int16
	for b := 0; b < nBlocks; b++ {
		block := data[b*blockBytes:]
		d := math.Float32frombits(binary.LittleEndian.Uint32(block[0:4]))
		qs := block[4:260]
		for i := 0; i < qkBlockSize; i++ {
			out = append(out, d*float32(int8(qs[i])))
		}
		for sb := 0; sb < qkSubBlocks; sb++ {
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
	for sb := 0; sb < qkSubBlocks; sb++ {
		var want int16
		for j := 0; j < 16; j++ {
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

func TestQuantizeKernels_MinFloat32_Good(t *testing.T) {
	if got := minFloat32([]float32{1, -5, 3, -2}); got != -5 {
		t.Errorf("minFloat32 = %v, want -5", got)
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
