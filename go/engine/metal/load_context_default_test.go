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
		{131072, 131072}, // e2b/e4b run their full 128K window by default (#367)
		{262144, 262144}, // 26B/31B run their full 256K window — the stance cap
		{524288, 262144}, // a future >256K checkpoint still caps at the stance
	}
	for _, tc := range cases {
		if got := resolveDefaultContext(tc.window); got != tc.want {
			t.Errorf("resolveDefaultContext(%d) = %d, want %d", tc.window, got, tc.want)
		}
	}
}
