// SPDX-Licence-Identifier: EUPL-1.2

package enginetest

import "testing"

// TestReferenceAffineMatVec_Good pins the reference's arithmetic on a
// hand-checkable single-group case: 1 row, 2 columns, one group of 2,
// scale=2, bias=1, codes [3,5] → dequantised weight [7,11]; x=[1.0,0.5] →
// dot=12.5, which bf16 represents exactly (its mantissa bits beyond bf16's 7
// are all zero), so the expected output bytes are pinned exactly rather than
// tolerance-compared.
func TestReferenceAffineMatVec_Good(t *testing.T) {
	const outDim, inDim, groupSize, bits = 1, 2, 2, 4
	packed := make([]byte, outDim*inDim*bits/8)
	affineSetCode(packed, 0*bits, bits, 3)
	affineSetCode(packed, 1*bits, bits, 5)
	if packed[0] != 0x53 {
		t.Fatalf("fixture packing: packed[0] = %#x, want 0x53", packed[0])
	}
	sh := bf16Encode(2.0)
	bh := bf16Encode(1.0)
	scales := []byte{byte(sh), byte(sh >> 8)}
	biases := []byte{byte(bh), byte(bh >> 8)}
	xh0, xh1 := bf16Encode(1.0), bf16Encode(0.5)
	x := []byte{byte(xh0), byte(xh0 >> 8), byte(xh1), byte(xh1 >> 8)}

	got, err := ReferenceAffineMatVec(x, packed, scales, biases, outDim, inDim, groupSize, bits)
	if err != nil {
		t.Fatalf("ReferenceAffineMatVec: %v", err)
	}
	want := bf16Encode(12.5)
	if got[0] != byte(want) || got[1] != byte(want>>8) {
		gv := bf16Decode(got[0], got[1])
		t.Fatalf("ReferenceAffineMatVec = %v (bytes %#x %#x), want 12.5 (bytes %#x %#x)",
			gv, got[0], got[1], byte(want), byte(want>>8))
	}
}

// TestReferenceAffineMatVec_Bad pins the validation surface: a malformed
// fixture is rejected with an error, never a panic or a silently wrong answer.
func TestReferenceAffineMatVec_Bad(t *testing.T) {
	x, packed, scales, biases, outDim, inDim, groupSize, bits := quantAffineFixture()

	if _, err := ReferenceAffineMatVec(x[:len(x)-1], packed, scales, biases, outDim, inDim, groupSize, bits); err == nil {
		t.Error("short x must be rejected")
	}
	if _, err := ReferenceAffineMatVec(x, packed[:len(packed)-1], scales, biases, outDim, inDim, groupSize, bits); err == nil {
		t.Error("short packed must be rejected")
	}
	if _, err := ReferenceAffineMatVec(x, packed, scales, biases, outDim, inDim, 3, bits); err == nil {
		t.Error("groupSize not dividing inDim must be rejected")
	}
	if _, err := ReferenceAffineMatVec(x, packed, scales, biases, outDim, inDim, groupSize, 0); err == nil {
		t.Error("bits <= 0 must be rejected")
	}
	if _, err := ReferenceAffineMatVec(x, packed, scales, biases, outDim, inDim, groupSize, 9); err == nil {
		t.Error("bits > 8 must be rejected")
	}
}

// TestReferenceAffineMatVec_Ugly pins the zero-sized surprising-but-valid
// case: outDim or inDim of 0 returns a clean empty result, mirroring the
// backends' own zero-sized MatVec fast path (e.g. engine/metal's
// QMVBF16Into), never an error.
func TestReferenceAffineMatVec_Ugly(t *testing.T) {
	got, err := ReferenceAffineMatVec(nil, nil, nil, nil, 0, 0, 64, 4)
	if err != nil {
		t.Fatalf("zero-sized ReferenceAffineMatVec: %v", err)
	}
	if len(got) != 0 {
		t.Fatalf("zero-sized ReferenceAffineMatVec length = %d, want 0", len(got))
	}
}

// TestQuantAffineFixtureRoundTrips_Good checks quant_parity.go's own fixture
// builder against affineExtractCode — the packer and the reference's own
// unpacker must agree on every code, or QuantParity would be comparing two
// backends against a fixture that doesn't mean what it claims to.
func TestQuantAffineFixtureRoundTrips_Good(t *testing.T) {
	_, packed, _, _, outDim, inDim, _, bits := quantAffineFixture()
	rowPacked := inDim * bits / 8
	maxCode := uint32(1)<<uint(bits) - 1
	for r := 0; r < outDim; r++ {
		pRow := packed[r*rowPacked : (r+1)*rowPacked]
		for c := 0; c < inDim; c++ {
			want := uint32(c*5+r*3+1) % (maxCode + 1)
			if got := affineExtractCode(pRow, c*bits, bits); got != want {
				t.Fatalf("row %d col %d: affineExtractCode = %d, want %d", r, c, got, want)
			}
		}
	}
}

// TestBF16RoundTrip_Good pins the codec pair this whole file leans on: every
// value bf16Encode produces, bf16Decode must read back losslessly (bf16
// encoding is already the lossy step; decoding a valid bf16 value is exact).
func TestBF16RoundTrip_Good(t *testing.T) {
	for _, v := range []float32{0, 1, -1, 12.5, 0.25, -3.75, 100, -0.03125} {
		h := bf16Encode(v)
		got := bf16Decode(byte(h), byte(h>>8))
		// bf16 keeps the top 8 bits of mantissa context (7 explicit + implicit
		// leading 1); values here are chosen exactly representable in bf16, so
		// the round trip must be exact.
		if got != v {
			t.Errorf("bf16 round trip: encode/decode(%v) = %v, want exact", v, got)
		}
	}
}
