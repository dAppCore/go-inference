// SPDX-Licence-Identifier: EUPL-1.2

package inference

import (
	"testing"
	"time"
)

// TestDecodePhaseBudget_HostPerToken_Good pins the host-serial split: the wall
// the GPU sat idle is TotalPerToken - GPUPerToken, the pipeline headroom.
func TestDecodePhaseBudget_HostPerToken_Good(t *testing.T) {
	b := DecodePhaseBudget{TotalPerToken: 6 * time.Millisecond, GPUPerToken: 2 * time.Millisecond}
	if got := b.HostPerToken(); got != 4*time.Millisecond {
		t.Fatalf("HostPerToken = %v, want 4ms", got)
	}
}

// TestDecodePhaseBudget_HostPerToken_Bad pins the clamp: when GPU time meets or
// exceeds the measured wall (overlapping spans) the host share is zero, never
// negative.
func TestDecodePhaseBudget_HostPerToken_Bad(t *testing.T) {
	b := DecodePhaseBudget{TotalPerToken: 2 * time.Millisecond, GPUPerToken: 3 * time.Millisecond}
	if got := b.HostPerToken(); got != 0 {
		t.Fatalf("HostPerToken with GPU>=total = %v, want 0", got)
	}
}

// TestDecodePhaseBudget_GPUFraction_Ugly pins the two edge cases: an untimed
// budget is 0, and a GPU span exceeding the wall clamps to 1 rather than >1.
func TestDecodePhaseBudget_GPUFraction_Ugly(t *testing.T) {
	if got := (DecodePhaseBudget{}).GPUFraction(); got != 0 {
		t.Fatalf("GPUFraction of zero budget = %v, want 0", got)
	}
	over := DecodePhaseBudget{TotalPerToken: time.Millisecond, GPUPerToken: 2 * time.Millisecond}
	if got := over.GPUFraction(); got != 1 {
		t.Fatalf("GPUFraction with GPU>total = %v, want clamp to 1", got)
	}
	half := DecodePhaseBudget{TotalPerToken: 4 * time.Millisecond, GPUPerToken: 2 * time.Millisecond}
	if got := half.GPUFraction(); got != 0.5 {
		t.Fatalf("GPUFraction 2/4 = %v, want 0.5", got)
	}
}
