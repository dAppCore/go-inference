// SPDX-Licence-Identifier: EUPL-1.2

package score

import (
	"testing"
)

// TestHelpers_ToInt_Good pins the numeric coercions toInt performs on the
// any-typed DuckDB row values the coverage report reads: int64/int32 pass
// through and float64 truncates toward zero.
func TestHelpers_ToInt_Good(t *testing.T) {
	cases := []struct {
		name string
		in   any
		want int
	}{
		{"int64", int64(5), 5},
		{"int32", int32(7), 7},
		{"float64 truncates", float64(3.9), 3},
		{"negative float64 truncates toward zero", float64(-2.9), -2},
	}
	for _, c := range cases {
		if got := toInt(c.in); got != c.want {
			t.Fatalf("toInt(%s=%v) = %d, want %d", c.name, c.in, got, c.want)
		}
	}
}

// TestHelpers_ToInt_Bad pins the fallback: any non-numeric (or untyped-nil)
// value coerces to 0 rather than panicking on the type assertion.
func TestHelpers_ToInt_Bad(t *testing.T) {
	for _, in := range []any{"7", nil, true, int(9), []int{1}} {
		if got := toInt(in); got != 0 {
			t.Fatalf("toInt(%v) = %d, want 0", in, got)
		}
	}
}
