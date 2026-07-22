// SPDX-Licence-Identifier: EUPL-1.2

package hip

import (
	"math"
	"testing"
)

// assertFloat32SlicesNear pins got to want element-wise within an absolute
// tolerance. It lives UNTAGGED so the portable vision/encoder tests use it on
// every platform (the "Mac untagged vet 0" contract); the tagged driver tests
// share the same definition.
func assertFloat32SlicesNear(t *testing.T, want, got []float32, tolerance float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("slice len = %d, want %d: %+v", len(got), len(want), got)
	}
	for i := range want {
		if math.Abs(float64(want[i]-got[i])) > float64(tolerance) {
			t.Fatalf("slice[%d] = %f, want %f within %f; got %+v", i, got[i], want[i], tolerance, got)
		}
	}
}
