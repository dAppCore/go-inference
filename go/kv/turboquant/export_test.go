// SPDX-Licence-Identifier: EUPL-1.2

package turboquant

import "testing"

// TestRotationMatrix_Good checks the export delegates to rotationFor's cached matrix verbatim (same
// length, same values) and returns a defensive copy — mutating the result must never corrupt the
// cache a later Encode/Decode call (or a second RotationMatrix call) reads.
func TestRotationMatrix_Good(t *testing.T) {
	const seed, d = 42, 8
	want := rotationFor(seed, d)
	got := RotationMatrix(seed, d)
	if len(got) != d*d {
		t.Fatalf("RotationMatrix(%d,%d) len = %d, want %d", seed, d, len(got), d*d)
	}
	for i, v := range got {
		if v != want.data[i] {
			t.Fatalf("RotationMatrix(%d,%d)[%d] = %v, want %v (rotationFor's cached value)", seed, d, i, v, want.data[i])
		}
	}
	got[0] = 999
	again := RotationMatrix(seed, d)
	if again[0] == 999 {
		t.Fatal("RotationMatrix must return a defensive copy — mutating the result corrupted the cache")
	}
}

// TestCentroids_Good checks the export delegates to centroidsFor verbatim (same length, ascending,
// same values) and returns a defensive copy.
func TestCentroids_Good(t *testing.T) {
	const d, bits = 128, 2
	want := centroidsFor(d, bits)
	got := Centroids(d, bits)
	if len(got) != 1<<bits {
		t.Fatalf("Centroids(%d,%d) len = %d, want %d", d, bits, len(got), 1<<bits)
	}
	for i, v := range got {
		if v != want[i] {
			t.Fatalf("Centroids(%d,%d)[%d] = %v, want %v (centroidsFor's cached value)", d, bits, i, v, want[i])
		}
		if i > 0 && v < got[i-1] {
			t.Fatalf("Centroids(%d,%d) not ascending at %d: %v < %v", d, bits, i, v, got[i-1])
		}
	}
	got[0] = 999
	again := Centroids(d, bits)
	if again[0] == 999 {
		t.Fatal("Centroids must return a defensive copy — mutating the result corrupted the cache")
	}
}

// TestUnpackIndices_Good round-trips packBits/UnpackIndices at a representative bit width — the
// export must agree with the package's own packer, since a cross-target parity test relies on both
// sides of that pair.
func TestUnpackIndices_Good(t *testing.T) {
	values := []int{5, 2, 7, 0, 3}
	packed := packBits(values, 3)
	got := UnpackIndices(packed, len(values), 3)
	if len(got) != len(values) {
		t.Fatalf("UnpackIndices len = %d, want %d", len(got), len(values))
	}
	for i, v := range values {
		if got[i] != v {
			t.Fatalf("UnpackIndices[%d] = %d, want %d", i, got[i], v)
		}
	}
}
