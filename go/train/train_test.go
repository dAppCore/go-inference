// SPDX-Licence-Identifier: EUPL-1.2

package train

import (
	"testing"

	core "dappco.re/go"
)

// --- NormalizeConfig ---

// Good: every default fills in on a zero-value Config.
func TestNormalizeConfig_Good(t *testing.T) {
	cfg := NormalizeConfig(Config{})
	if cfg.BatchSize != 1 || cfg.GradientAccumulationSteps != 1 || cfg.Epochs != 1 {
		t.Fatalf("scalar defaults = %+v, want BatchSize/GradientAccumulationSteps/Epochs all 1", cfg)
	}
	if cfg.LearningRate != 1e-5 {
		t.Fatalf("LearningRate = %v, want default 1e-5", cfg.LearningRate)
	}
	if cfg.EvalMaxTokens != 96 {
		t.Fatalf("EvalMaxTokens = %d, want default 96", cfg.EvalMaxTokens)
	}
}

// Bad: negative scalars are floored exactly like zero — NormalizeConfig
// never lets a malformed caller value (e.g. BatchSize: -4) survive into
// the batch/loop maths.
func TestNormalizeConfig_Bad(t *testing.T) {
	cfg := NormalizeConfig(Config{BatchSize: -4, GradientAccumulationSteps: -1, Epochs: -2, EvalMaxTokens: -10})
	if cfg.BatchSize != 1 || cfg.GradientAccumulationSteps != 1 || cfg.Epochs != 1 || cfg.EvalMaxTokens != 96 {
		t.Fatalf("negative-input defaults = %+v, want every field floored to its default", cfg)
	}
}

// Ugly: explicit non-zero values pass through unchanged — NormalizeConfig
// only fills gaps, it never overrides a caller's real setting.
func TestNormalizeConfig_Ugly(t *testing.T) {
	cfg := NormalizeConfig(Config{
		BatchSize:                 8,
		GradientAccumulationSteps: 4,
		Epochs:                    3,
		LearningRate:              2e-4,
		EvalMaxTokens:             256,
	})
	if cfg.BatchSize != 8 || cfg.GradientAccumulationSteps != 4 || cfg.Epochs != 3 || cfg.LearningRate != 2e-4 || cfg.EvalMaxTokens != 256 {
		t.Fatalf("explicit values overwritten: %+v", cfg)
	}
}

// --- EffectiveBatchSize ---

// Good: batch size times gradient accumulation steps.
func TestEffectiveBatchSize_Good(t *testing.T) {
	if got := EffectiveBatchSize(Config{BatchSize: 4, GradientAccumulationSteps: 2}); got != 8 {
		t.Fatalf("EffectiveBatchSize() = %d, want 8", got)
	}
}

// Bad: a zero-value Config still floors both factors to 1 rather than
// returning zero.
func TestEffectiveBatchSize_Bad(t *testing.T) {
	if got := EffectiveBatchSize(Config{}); got != 1 {
		t.Fatalf("EffectiveBatchSize(zero-value) = %d, want 1", got)
	}
}

// Ugly: a negative GradientAccumulationSteps is floored to 1 rather than
// producing a negative or zero effective batch size.
func TestEffectiveBatchSize_Ugly(t *testing.T) {
	if got := EffectiveBatchSize(Config{BatchSize: 4, GradientAccumulationSteps: -3}); got != 4 {
		t.Fatalf("EffectiveBatchSize(negative accum) = %d, want 4 (accum floored to 1)", got)
	}
}

// --- resultError ---

// Good: an OK result reports a nil error.
func TestResultError_Good(t *testing.T) {
	if err := resultError(core.Result{OK: true}); err != nil {
		t.Fatalf("resultError(OK) = %v, want nil", err)
	}
}

// Bad: a failed result carrying a real error value returns it unchanged.
func TestResultError_Bad(t *testing.T) {
	want := core.NewError("boom")
	if err := resultError(core.Result{OK: false, Value: want}); err != want {
		t.Fatalf("resultError(failed) = %v, want %v", err, want)
	}
}

// Ugly: a failed result whose Value is not an error falls back to the
// generic sentinel rather than panicking on the type assertion.
func TestResultError_Ugly(t *testing.T) {
	if err := resultError(core.Result{OK: false, Value: "not an error"}); err != errCoreResultFailed {
		t.Fatalf("resultError(non-error value) = %v, want errCoreResultFailed", err)
	}
}
