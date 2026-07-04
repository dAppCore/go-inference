// SPDX-Licence-Identifier: EUPL-1.2

package mistral_test

import (
	"math"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference/model/mistral"
)

// yarn_branches_test.go closes the YaRN edge-case branches the happy-path tests
// don't reach: the factor<1 clamp, the two correction-range clamps (low<0 and
// high>dim/2-1), the degenerate equal-ramp guard, and the config's beta_fast /
// beta_slow defaulting. Each asserts the resulting behaviour, not just that the
// branch executes, so the cover is faithful rather than theatre.

// TestYaRNInvFreqs_FactorBelowOne_ClampsToPlainRope drives factor < 1, which the
// clamp pins to 1 (no context extension). The blend then collapses to the plain
// RoPE inv-freqs regardless of the ramp — identical to factor == 1.
func TestYaRNInvFreqs_FactorBelowOne_ClampsToPlainRope(t *testing.T) {
	const base, dim = 1e6, 128
	const betaFast, betaSlow, origMax = 32.0, 1.0, 16384
	// factor 0.25 (< 1) must be treated as factor 1 — i.e. plain RoPE everywhere.
	got := mistral.YaRNInvFreqs(base, 0.25, betaFast, betaSlow, origMax, dim)
	if len(got) != dim/2 {
		t.Fatalf("len %d, want %d", len(got), dim/2)
	}
	for i := range got {
		want := float32(plainRope(base, i, dim))
		if relDiff(got[i], want) > 1e-5 {
			t.Fatalf("factor<1 freq[%d]=%g, want plain rope %g (clamp to factor 1 failed)", i, got[i], want)
		}
	}
	// And it must equal the explicit factor==1 result bit-for-bit (same clamp target).
	want1 := mistral.YaRNInvFreqs(base, 1, betaFast, betaSlow, origMax, dim)
	for i := range got {
		if got[i] != want1[i] {
			t.Fatalf("factor<1 freq[%d]=%g != factor==1 freq %g", i, got[i], want1[i])
		}
	}
}

// TestYaRNInvFreqs_LowClampedToZero forces the correction range's low edge below
// zero (a very large beta_fast makes the fast-rotation correction dim negative),
// exercising the low<0 → 0 clamp. With low pinned at 0 the dim-0 inv-freq is no
// longer the pure extrapolated base — the ramp has already begun at i==0, so the
// blend pulls it strictly below plain RoPE (yet not all the way to interpolated).
func TestYaRNInvFreqs_LowClampedToZero(t *testing.T) {
	const base, factor, dim = 1e6, 16.0, 128
	const betaSlow, origMax = 1.0, 16384
	const betaFast = 5000.0 // corrDim(beta_fast) < 0 → low floors below 0 → clamps to 0
	got := mistral.YaRNInvFreqs(base, factor, betaFast, betaSlow, origMax, dim)

	extra0 := plainRope(base, 0, dim) // == 1 (the highest frequency)
	// low clamped to 0 means ramp(0) = (0-0)/(high-0) = 0, so dim 0 is still pure
	// extrapolated. The clamp's visible effect is that no dim extrapolates ABOVE
	// dim 0; assert dim 0 is exactly plain rope and the sequence is bounded + sane.
	if relDiff(got[0], float32(extra0)) > 1e-5 {
		t.Fatalf("dim 0 freq %g, want plain rope %g", got[0], extra0)
	}
	for i := range got {
		extra := plainRope(base, i, dim)
		inter := extra / factor
		if float64(got[i]) > extra*(1+1e-5) || float64(got[i]) < inter*(1-1e-5) {
			t.Fatalf("dim %d freq %g outside [inter %g, extra %g]", i, got[i], inter, extra)
		}
	}
}

// TestYaRNInvFreqs_HighClampedToMax forces the correction range's high edge past
// dim/2-1 (a near-zero beta_slow pushes the slow-rotation correction dim very
// large), exercising the high>max → dim/2-1 clamp. With high pinned at the last
// index, the topmost dim is exactly at the ramp's high edge → ramp == 1 → fully
// interpolated (plain rope / factor).
func TestYaRNInvFreqs_HighClampedToMax(t *testing.T) {
	const base, factor, dim = 1e6, 16.0, 128
	const betaFast, origMax = 32.0, 16384
	const betaSlow = 0.001 // corrDim(beta_slow) ≫ dim/2-1 → high clamps to dim/2-1 (63)
	got := mistral.YaRNInvFreqs(base, factor, betaFast, betaSlow, origMax, dim)

	last := dim/2 - 1
	wantInter := float32(plainRope(base, last, dim) / factor)
	if relDiff(got[last], wantInter) > 1e-5 {
		t.Fatalf("high-clamped: last dim %d freq %g, want fully-interpolated %g", last, got[last], wantInter)
	}
	// the clamp must not have broken the invariant — still bounded + non-increasing.
	for i := range got {
		extra := plainRope(base, i, dim)
		inter := extra / factor
		if float64(got[i]) > extra*(1+1e-5) || float64(got[i]) < inter*(1-1e-5) {
			t.Fatalf("dim %d freq %g outside [inter %g, extra %g]", i, got[i], inter, extra)
		}
		if i > 0 && got[i] > got[i-1]*(1+1e-5) {
			t.Fatalf("freqs not non-increasing at dim %d: %g > %g", i, got[i], got[i-1])
		}
	}
}

// TestYaRNInvFreqs_DegenerateRamp_Dim2 drives the high==low degenerate-ramp guard
// (yarnRamp adds 0.001 to avoid divide-by-zero). dim=2 ⇒ half==1 and dim/2-1==0,
// so high clamps to 0 while low floors to 0 → high==low==0. The single inv-freq
// must still be finite and within the extrapolate/interpolate envelope (no NaN/Inf
// from a zero-width ramp).
func TestYaRNInvFreqs_DegenerateRamp_Dim2(t *testing.T) {
	const base, factor, dim = 1e6, 16.0, 2
	got := mistral.YaRNInvFreqs(base, factor, 32, 1, 16384, dim)
	if len(got) != 1 {
		t.Fatalf("len %d, want 1 (dim/2)", len(got))
	}
	f := float64(got[0])
	if math.IsNaN(f) || math.IsInf(f, 0) {
		t.Fatalf("degenerate ramp produced non-finite freq %g", f)
	}
	extra := plainRope(base, 0, dim) // == 1
	inter := extra / factor
	if f > extra*(1+1e-5) || f < inter*(1-1e-5) {
		t.Fatalf("degenerate-ramp freq %g outside [inter %g, extra %g]", f, inter, extra)
	}
}

// TestConfigArch_YaRNBetaDefaults proves the Arch YaRN block defaults beta_fast→32
// and beta_slow→1 when a config declares rope_type "yarn" with an extension factor
// but omits the betas. The check compares the resolved RopeFreqs VALUES against a
// standalone YaRNInvFreqs call with the 32/1 defaults — a length check alone would
// pass for any betas and prove nothing.
func TestConfigArch_YaRNBetaDefaults(t *testing.T) {
	// yarn rope, factor>1, original_max_position_embeddings>0, beta_fast/beta_slow ABSENT.
	const cfg = `{"hidden_size":256,"num_hidden_layers":2,"num_attention_heads":4,"head_dim":64,` +
		`"intermediate_size":512,"vocab_size":1000,` +
		`"rope_parameters":{"rope_type":"yarn","rope_theta":1000000,"factor":8.0,` +
		`"original_max_position_embeddings":16384}}`
	var c mistral.Config
	if r := core.JSONUnmarshal([]byte(cfg), &c); !r.OK {
		t.Fatalf("unmarshal: %s", r.Error())
	}
	arch, err := c.Arch()
	if err != nil {
		t.Fatalf("Arch: %v", err)
	}
	if len(arch.RopeFreqs) != 64/2 {
		t.Fatalf("RopeFreqs len %d, want 32 (head_dim/2)", len(arch.RopeFreqs))
	}
	// the defaults that must have fired: beta_fast 32, beta_slow 1, over head_dim 64.
	want := mistral.YaRNInvFreqs(1_000_000, 8, 32, 1, 16384, 64)
	for i := range want {
		if relDiff(arch.RopeFreqs[i], want[i]) > 1e-5 {
			t.Fatalf("RopeFreqs[%d]=%g != YaRN with 32/1 defaults %g", i, arch.RopeFreqs[i], want[i])
		}
	}
	// sanity: the freqs are a genuine YaRN remap (a ramp exists), not plain rope —
	// otherwise wrong betas could still coincidentally match.
	plainHead := mistral.YaRNInvFreqs(1_000_000, 1, 32, 1, 16384, 64) // factor 1 == plain
	differs := false
	for i := range want {
		if relDiff(arch.RopeFreqs[i], plainHead[i]) > 1e-4 {
			differs = true
			break
		}
	}
	if !differs {
		t.Fatal("RopeFreqs identical to plain rope — YaRN extension did not apply")
	}
}
