// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import "testing"

func TestWithAutoreleasePoolRunsCallback(t *testing.T) {
	called := false
	withAutoreleasePool(func() {
		called = true
	})
	if !called {
		t.Fatal("withAutoreleasePool did not run callback")
	}
}
