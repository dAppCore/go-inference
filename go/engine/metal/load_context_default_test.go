// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import "testing"

// TestResolveDefaultContext gates the unset-context mapping: checkpoint
// window capped at defaultContextCap, 4096 floor when no window is declared.
func TestResolveDefaultContext(t *testing.T) {
	cases := []struct{ window, want int }{
		{0, 4096},
		{-1, 4096},
		{8192, 8192},
		{32768, 32768},
		{131072, 32768},
		{262144, 32768},
	}
	for _, tc := range cases {
		if got := resolveDefaultContext(tc.window); got != tc.want {
			t.Errorf("resolveDefaultContext(%d) = %d, want %d", tc.window, got, tc.want)
		}
	}
}
