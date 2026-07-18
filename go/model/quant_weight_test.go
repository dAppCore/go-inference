// SPDX-Licence-Identifier: EUPL-1.2

package model

import (
	"bytes"
	"testing"
)

// mkQuantRows builds a QuantWeight with recognisable per-row byte fills — Packed rows of `fill`,
// Scales rows of `fill+1`, Biases rows of `fill+2`, sized to the (outDim, inDim, bits, groupSize)
// layout ConcatQuantRows preserves (Packed stride inDim·bits/8, Scales/Biases stride
// (inDim/groupSize)·2). The distinct fills let a concat assert row ORDER, not just total length.
func mkQuantRows(outDim, inDim, bits, groupSize int, fill byte) *QuantWeight {
	packedRow := inDim * bits / 8
	groupRow := (inDim / groupSize) * 2
	fillN := func(n int, b byte) []byte {
		s := make([]byte, n)
		for i := range s {
			s[i] = b
		}
		return s
	}
	return &QuantWeight{
		Packed:    fillN(outDim*packedRow, fill),
		Scales:    fillN(outDim*groupRow, fill+1),
		Biases:    fillN(outDim*groupRow, fill+2),
		Bits:      bits,
		GroupSize: groupSize,
		OutDim:    outDim,
		InDim:     inDim,
	}
}

// TestConcatQuantRows_Good pins the happy path: two matched-geometry weights concatenate into one
// whose Packed/Scales/Biases are EXACTLY a's bytes followed by b's (row order a-then-b, the [gate‖up]
// layout), OutDim is summed, and InDim/Bits/GroupSize carry through. The result must own its buffers
// (mutating an input afterwards leaves it untouched) so it outlives an mmap the inputs viewed.
func TestConcatQuantRows_Good(t *testing.T) {
	a := mkQuantRows(3, 8, 4, 8, 0x10) // FF=3 gate rows
	b := mkQuantRows(3, 8, 4, 8, 0x40) // FF=3 up rows

	got := ConcatQuantRows(a, b)
	if got == nil {
		t.Fatal("ConcatQuantRows returned nil for matched geometry")
	}
	if !bytes.Equal(got.Packed, append(append([]byte{}, a.Packed...), b.Packed...)) {
		t.Error("Packed is not a.Packed followed by b.Packed")
	}
	if !bytes.Equal(got.Scales, append(append([]byte{}, a.Scales...), b.Scales...)) {
		t.Error("Scales is not a.Scales followed by b.Scales")
	}
	if !bytes.Equal(got.Biases, append(append([]byte{}, a.Biases...), b.Biases...)) {
		t.Error("Biases is not a.Biases followed by b.Biases")
	}
	if got.OutDim != a.OutDim+b.OutDim {
		t.Errorf("OutDim = %d, want %d (summed)", got.OutDim, a.OutDim+b.OutDim)
	}
	if got.InDim != a.InDim || got.Bits != a.Bits || got.GroupSize != a.GroupSize {
		t.Errorf("geometry = (InDim %d, Bits %d, GroupSize %d), want (%d, %d, %d)",
			got.InDim, got.Bits, got.GroupSize, a.InDim, a.Bits, a.GroupSize)
	}
	// Owned, not aliased: mutate a's first Packed byte; got must not change.
	a.Packed[0] = 0xFF
	if got.Packed[0] == 0xFF {
		t.Error("result Packed aliases the input — a mutation leaked through (not materialised)")
	}
}

// TestConcatQuantRows_Bad pins the geometry guard: a mismatch on ANY of the three shared shape
// fields (InDim, Bits, GroupSize) — the byte-append would silently corrupt the row layout — returns
// nil rather than a wrong-but-plausible weight.
func TestConcatQuantRows_Bad(t *testing.T) {
	base := mkQuantRows(3, 8, 4, 8, 0x10)
	for _, tc := range []struct {
		name string
		b    *QuantWeight
	}{
		{"InDim", mkQuantRows(3, 16, 4, 8, 0x40)},
		{"Bits", mkQuantRows(3, 8, 8, 8, 0x40)},
		{"GroupSize", mkQuantRows(3, 8, 4, 4, 0x40)},
	} {
		if got := ConcatQuantRows(base, tc.b); got != nil {
			t.Errorf("%s mismatch: ConcatQuantRows = %+v, want nil", tc.name, got)
		}
	}
}

// TestConcatQuantRows_Ugly pins the boundary cases: a nil operand returns nil (never panics), and a
// zero-row operand is the identity — concatenating an empty weight leaves the other's bytes and rows
// exactly, materialised.
func TestConcatQuantRows_Ugly(t *testing.T) {
	a := mkQuantRows(3, 8, 4, 8, 0x10)
	if ConcatQuantRows(nil, a) != nil || ConcatQuantRows(a, nil) != nil {
		t.Error("a nil operand must return nil")
	}
	empty := &QuantWeight{Bits: 4, GroupSize: 8, InDim: 8} // OutDim 0, no bytes
	got := ConcatQuantRows(a, empty)
	if got == nil {
		t.Fatal("concat with a zero-row weight returned nil")
	}
	if got.OutDim != a.OutDim || !bytes.Equal(got.Packed, a.Packed) {
		t.Errorf("concat with empty is not the identity: OutDim %d Packed %d bytes, want %d / %d",
			got.OutDim, len(got.Packed), a.OutDim, len(a.Packed))
	}
}
