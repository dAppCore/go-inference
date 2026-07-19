// SPDX-Licence-Identifier: EUPL-1.2

package turboquant

import (
	"testing"

	core "dappco.re/go"
)

// TestNewMixedSplitDescriptor_Good checks the mask bits land exactly at the
// requested channels.
func TestNewMixedSplitDescriptor_Good(t *testing.T) {
	split := NewMixedSplitDescriptor(16, []int{0, 5, 15}, 2, 3)
	for ch := 0; ch < 16; ch++ {
		want := ch == 0 || ch == 5 || ch == 15
		if got := split.isOutlier(ch); got != want {
			t.Errorf("isOutlier(%d) = %v, want %v", ch, got, want)
		}
	}
	if got := split.outlierCount(); got != 3 {
		t.Errorf("outlierCount() = %d, want 3", got)
	}
}

// TestNewMixedSplitDescriptor_Ugly checks out-of-range channel indices are
// dropped rather than corrupting adjacent mask bytes.
func TestNewMixedSplitDescriptor_Ugly(t *testing.T) {
	split := NewMixedSplitDescriptor(8, []int{-1, 8, 100, 3}, 2, 3)
	if got := split.outlierCount(); got != 1 {
		t.Errorf("outlierCount() = %d, want 1 (only channel 3 is in range)", got)
	}
	if !split.isOutlier(3) {
		t.Error("isOutlier(3) = false, want true")
	}
}

// TestCalibrateMixedSplit_Good checks the k highest-amplitude channels are
// selected as outliers.
func TestCalibrateMixedSplit_Good(t *testing.T) {
	const d = 8
	samples := [][]float32{
		{10, 1, 1, 1, 1, 1, 1, 1},
		{10, 1, 1, 1, 1, 1, 1, 9},
		{10, 1, 1, 1, 1, 1, 1, 9},
	}
	split := CalibrateMixedSplit(samples, d, 2, 2, 3)
	if !split.isOutlier(0) {
		t.Error("channel 0 (consistently the largest amplitude) should be an outlier")
	}
	if !split.isOutlier(7) {
		t.Error("channel 7 (second largest amplitude) should be an outlier")
	}
	if split.outlierCount() != 2 {
		t.Errorf("outlierCount() = %d, want 2", split.outlierCount())
	}
}

// TestCalibrateMixedSplit_Ugly checks k larger than d clamps to d (every
// channel becomes an outlier) rather than panicking on an out-of-range
// slice.
func TestCalibrateMixedSplit_Ugly(t *testing.T) {
	samples := [][]float32{{1, 2, 3}}
	split := CalibrateMixedSplit(samples, 3, 100, 2, 3)
	if got := split.outlierCount(); got != 3 {
		t.Errorf("outlierCount() = %d, want 3 (k clamped to d)", got)
	}
}

// TestCalibrateMixedSplit_Bad checks k < 0 clamps to 0 (no outliers) rather
// than panicking.
func TestCalibrateMixedSplit_Bad(t *testing.T) {
	samples := [][]float32{{1, 2, 3}}
	split := CalibrateMixedSplit(samples, 3, -5, 2, 3)
	if got := split.outlierCount(); got != 0 {
		t.Errorf("outlierCount() = %d, want 0 (negative k clamped to 0)", got)
	}
}

// TestMixedSplitDescriptorSplit_Good checks split partitions preserve
// ascending channel order within each sub-row.
func TestMixedSplitDescriptorSplit_Good(t *testing.T) {
	split := NewMixedSplitDescriptor(6, []int{1, 4}, 2, 3)
	outlier, base := split.split([]float32{0, 1, 2, 3, 4, 5})
	if len(outlier) != 2 || outlier[0] != 1 || outlier[1] != 4 {
		t.Errorf("split outlier = %v, want {1,4}", outlier)
	}
	if len(base) != 4 || base[0] != 0 || base[1] != 2 || base[2] != 3 || base[3] != 5 {
		t.Errorf("split base = %v, want {0,2,3,5}", base)
	}
}

// TestEncodeMixed_Good checks a full round trip with a non-trivial split
// stays close for the outlier channel especially (it gets the extra bit).
func TestEncodeMixed_Good(t *testing.T) {
	split := NewMixedSplitDescriptor(4, []int{0}, 1, 4)
	x := []float32{9, 1, 1, 1}
	e := EncodeMixed(x, split, 42)
	got := DecodeMixed(e, 42)
	if len(got) != len(x) {
		t.Fatalf("DecodeMixed returned %d elements, want %d", len(got), len(x))
	}
}

// TestEncodeMixed_Ugly checks an empty outlier set (every channel in the
// base group) still round-trips — the outlier sub-vector is length 0.
func TestEncodeMixed_Ugly(t *testing.T) {
	split := NewMixedSplitDescriptor(4, nil, 2, 3)
	x := []float32{1, 2, 3, 4}
	e := EncodeMixed(x, split, 42)
	if e.Outlier.D != 0 {
		t.Errorf("EncodeMixed with no outliers: Outlier.D = %d, want 0", e.Outlier.D)
	}
	got := DecodeMixed(e, 42)
	if len(got) != len(x) {
		t.Fatalf("DecodeMixed returned %d elements, want %d", len(got), len(x))
	}
}

// TestEncodeMixed_Bad checks every channel marked as an outlier (empty base
// group) also round-trips.
func TestEncodeMixed_Bad(t *testing.T) {
	split := NewMixedSplitDescriptor(4, []int{0, 1, 2, 3}, 2, 3)
	x := []float32{1, 2, 3, 4}
	e := EncodeMixed(x, split, 42)
	if e.Base.D != 0 {
		t.Errorf("EncodeMixed with every channel an outlier: Base.D = %d, want 0", e.Base.D)
	}
	got := DecodeMixed(e, 42)
	if len(got) != len(x) {
		t.Fatalf("DecodeMixed returned %d elements, want %d", len(got), len(x))
	}
}

// TestMarshalMixed_Good round-trips through the wire format.
func TestMarshalMixed_Good(t *testing.T) {
	split := NewMixedSplitDescriptor(8, []int{2, 6}, 2, 3)
	x := []float32{1, 2, 3, 4, 5, 6, 7, 8}
	e := EncodeMixed(x, split, 7)
	data := MarshalMixed(e)
	back := UnmarshalMixed(data, len(x), split.BaseBits, split.OutlierBits)
	if back.Split.outlierCount() != split.outlierCount() {
		t.Errorf("round-tripped outlierCount() = %d, want %d", back.Split.outlierCount(), split.outlierCount())
	}
	got := DecodeMixed(back, 7)
	want := DecodeMixed(e, 7)
	for i := range got {
		if got[i] != want[i] {
			t.Errorf("round-tripped decode[%d] = %v, want %v", i, got[i], want[i])
		}
	}
}

// TestUnmarshalMixed_Bad checks a too-short payload (shorter than even the
// mask) decodes to the safe zero-value encoding rather than panicking.
func TestUnmarshalMixed_Bad(t *testing.T) {
	got := UnmarshalMixed(nil, 128, 2, 3)
	if got.Split.D != 128 || got.Outlier.Indices != nil {
		t.Errorf("UnmarshalMixed(nil) = %+v, want the zero-value split with D=128", got)
	}
}

// ExampleEncodeMixed demonstrates the practical mixed-bit KV mode: a
// calibrated outlier channel gets one extra bit over the rest.
func ExampleEncodeMixed() {
	split := NewMixedSplitDescriptor(4, []int{0}, 2, 3) // channel 0 at 3 bits, rest at 2
	e := EncodeMixed([]float32{9, 1, 1, 1}, split, 42)
	x := DecodeMixed(e, 42)
	core.Println("row length:", len(x))
	// Output:
	// row length: 4
}
