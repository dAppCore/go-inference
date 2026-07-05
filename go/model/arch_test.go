// SPDX-Licence-Identifier: EUPL-1.2

package model

import "testing"

// TestResolveMoEGating covers the gating default: an unset gating resolves to
// MoEGatingSoftmax (the only router variant the metal engine ships, and gemma4's
// method), while an explicitly-declared gating passes through unchanged — the path
// gemma4's always-softmax config never exercises.
func TestResolveMoEGating(t *testing.T) {
	if got := resolveMoEGating(""); got != MoEGatingSoftmax {
		t.Fatalf(`resolveMoEGating("") = %q, want the default %q`, got, MoEGatingSoftmax)
	}
	if got := resolveMoEGating(MoEGatingSoftmax); got != MoEGatingSoftmax {
		t.Fatalf("resolveMoEGating(MoEGatingSoftmax) = %q, want it unchanged", got)
	}
	if got := resolveMoEGating(MoEGating("sigmoid")); got != MoEGating("sigmoid") {
		t.Fatalf(`resolveMoEGating("sigmoid") = %q, want a declared gating to pass through`, got)
	}
}
