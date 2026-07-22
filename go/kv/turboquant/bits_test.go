// SPDX-Licence-Identifier: EUPL-1.2

package turboquant

import "testing"

// TestPackBits_Good round-trips values at every bit width this package
// uses (1-4) through packBits/unpackBits.
func TestPackBits_Good(t *testing.T) {
	for bits := 1; bits <= 4; bits++ {
		max := (1 << uint(bits)) - 1
		values := make([]int, max+1)
		for i := range values {
			values[i] = i
		}
		packed := packBits(values, bits)
		got := unpackBits(packed, len(values), bits)
		for i := range values {
			if got[i] != values[i] {
				t.Errorf("bits=%d: unpackBits(packBits(...))[%d] = %d, want %d", bits, i, got[i], values[i])
			}
		}
	}
}

// TestPackBits_Ugly exercises the degenerate bits == 0 case (Q_prod's b=1
// stage-1, which has zero index bits) — packBits must return nil/empty
// without panicking, and unpackBits must return all zeros.
func TestPackBits_Ugly(t *testing.T) {
	packed := packBits([]int{1, 2, 3}, 0)
	if len(packed) != 0 {
		t.Fatalf("packBits(_, 0) = %v, want empty", packed)
	}
	got := unpackBits(packed, 3, 0)
	for i, v := range got {
		if v != 0 {
			t.Errorf("unpackBits(nil, 3, 0)[%d] = %d, want 0", i, v)
		}
	}
}

// TestUnpackBits_Bad reads more values than the packed payload can hold —
// out-of-range reads must decode as 0 bits rather than panicking (the
// payload length, not a caller-supplied count, is the source of truth for
// what was actually packed).
func TestUnpackBits_Bad(t *testing.T) {
	packed := packBits([]int{3}, 2) // one byte can hold four 2-bit values, only one is meaningful
	got := unpackBits(packed[:0], 1, 2)
	if got[0] != 0 {
		t.Errorf("unpackBits with truncated data = %d, want 0", got[0])
	}
}

// TestPackedByteLen_Good checks packedByteLen agrees with len(packBits(...))
// across a spread of (n, bits) pairs.
func TestPackedByteLen_Good(t *testing.T) {
	for _, tc := range []struct{ n, bits int }{
		{0, 3}, {1, 3}, {8, 3}, {9, 3}, {128, 2}, {128, 4}, {96, 1},
	} {
		values := make([]int, tc.n)
		got := packedByteLen(tc.n, tc.bits)
		want := len(packBits(values, tc.bits))
		if got != want {
			t.Errorf("packedByteLen(%d,%d) = %d, want %d", tc.n, tc.bits, got, want)
		}
	}
}

// TestPackedByteLen_Ugly checks the bits == 0 degenerate case reports 0
// regardless of n.
func TestPackedByteLen_Ugly(t *testing.T) {
	if got := packedByteLen(128, 0); got != 0 {
		t.Errorf("packedByteLen(128, 0) = %d, want 0", got)
	}
}

// TestPackSigns_Good round-trips a sign pattern through packSigns/unpackSigns.
func TestPackSigns_Good(t *testing.T) {
	signs := []bool{true, false, true, true, false, false, false, true, true}
	packed := packSigns(signs)
	got := unpackSigns(packed, len(signs))
	for i, s := range signs {
		want := -1.0
		if s {
			want = 1.0
		}
		if got[i] != want {
			t.Errorf("unpackSigns(packSigns(...))[%d] = %v, want %v", i, got[i], want)
		}
	}
}
