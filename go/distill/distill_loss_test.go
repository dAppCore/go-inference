// SPDX-Licence-Identifier: EUPL-1.2

package distill

import (
	"math"
	"testing"
)

// --- NormalizeConfig ---

// Good: a zero-value Config gets the documented defaults — batch size 1,
// 1 epoch, temperature 1, KL loss.
func TestNormalizeConfig_Good(t *testing.T) {
	got := NormalizeConfig(Config{})
	if got.Batch.BatchSize != 1 {
		t.Errorf("Batch.BatchSize = %d, want 1", got.Batch.BatchSize)
	}
	if got.Epochs != 1 {
		t.Errorf("Epochs = %d, want 1", got.Epochs)
	}
	if got.Temperature != 1 {
		t.Errorf("Temperature = %v, want 1", got.Temperature)
	}
	if got.Loss != LossKL {
		t.Errorf("Loss = %q, want %q", got.Loss, LossKL)
	}
}

// Bad: a negative, NaN, or infinite temperature is poisoned to NaN so
// BatchLoss rejects it explicitly instead of silently using a
// nonsensical scale.
func TestNormalizeConfig_Bad(t *testing.T) {
	for _, temp := range []float64{-1, math.NaN(), math.Inf(1), math.Inf(-1)} {
		got := NormalizeConfig(Config{Temperature: temp})
		if !math.IsNaN(got.Temperature) {
			t.Errorf("NormalizeConfig(Temperature=%v).Temperature = %v, want NaN", temp, got.Temperature)
		}
	}
}

// Ugly: already-populated fields survive normalisation unchanged —
// NormalizeConfig only fills in defaults, it never overwrites explicit
// caller values.
func TestNormalizeConfig_Ugly(t *testing.T) {
	cfg := Config{
		Batch:       BatchConfig{BatchSize: 8},
		Epochs:      5,
		Temperature: 2.5,
		Loss:        LossSoftCrossEntropy,
	}
	got := NormalizeConfig(cfg)
	if got.Batch.BatchSize != 8 || got.Epochs != 5 || got.Temperature != 2.5 || got.Loss != LossSoftCrossEntropy {
		t.Fatalf("NormalizeConfig() mutated explicit fields: got %+v", got)
	}
}

// --- BatchLoss ---

// Good: identical teacher/student distributions yield ~zero KL, and
// Value tracks the requested loss kind (KL vs soft cross-entropy).
func TestBatchLoss_Good(t *testing.T) {
	teacher := Logits{{{2, 0, 0}}}
	student := Logits{{{2, 0, 0}}}

	loss, err := BatchLoss(teacher, student, nil, Config{Loss: LossKL, Temperature: 1})
	if err != nil {
		t.Fatalf("BatchLoss() error = %v", err)
	}
	if loss.Tokens != 1 {
		t.Fatalf("Tokens = %d, want 1", loss.Tokens)
	}
	if math.Abs(loss.KL) > 1e-9 {
		t.Fatalf("KL = %v, want ~0 for identical distributions", loss.KL)
	}
	if loss.Value != loss.KL {
		t.Fatalf("Value = %v, want it to equal KL for LossKL", loss.Value)
	}

	softLoss, err := BatchLoss(teacher, student, nil, Config{Loss: LossSoftCrossEntropy, Temperature: 1})
	if err != nil {
		t.Fatalf("BatchLoss() error = %v", err)
	}
	if softLoss.Value != softLoss.SoftCrossEntropy {
		t.Fatalf("Value = %v, want it to equal SoftCrossEntropy for LossSoftCrossEntropy", softLoss.Value)
	}
}

// Bad: shape mismatches and an invalid temperature are rejected with the
// documented sentinel errors rather than a panic or a silent NaN.
func TestBatchLoss_Bad(t *testing.T) {
	if _, err := BatchLoss(nil, nil, nil, Config{}); err != errTeacherLogitsEmpty {
		t.Errorf("empty teacher logits error = %v, want errTeacherLogitsEmpty", err)
	}

	teacher := Logits{{{1, 2}}}
	studentBadBatch := Logits{{{1, 2}}, {{1, 2}}}
	if _, err := BatchLoss(teacher, studentBadBatch, nil, Config{}); err != errLogitBatch {
		t.Errorf("batch mismatch error = %v, want errLogitBatch", err)
	}

	studentBadVocab := Logits{{{1, 2, 3}}}
	if _, err := BatchLoss(teacher, studentBadVocab, nil, Config{}); err != errLogitVocab {
		t.Errorf("vocab mismatch error = %v, want errLogitVocab", err)
	}

	if _, err := BatchLoss(teacher, teacher, nil, Config{Temperature: -1}); err != errTempInvalid {
		t.Errorf("negative temperature error = %v, want errTempInvalid", err)
	}

	if _, err := BatchLoss(teacher, teacher, nil, Config{Loss: "bogus"}); err == nil {
		t.Error("unsupported loss kind: expected error, got nil")
	}
}

// Ugly: a mask that excludes every position reports errNoMaskedTokens; a
// mask that excludes some positions counts only the unmasked ones.
func TestBatchLoss_Ugly(t *testing.T) {
	teacher := Logits{{{1, 0}, {0, 1}}}
	student := Logits{{{1, 0}, {0, 1}}}

	if _, err := BatchLoss(teacher, student, [][]float32{{0, 0}}, Config{}); err != errNoMaskedTokens {
		t.Fatalf("fully-masked-out error = %v, want errNoMaskedTokens", err)
	}

	loss, err := BatchLoss(teacher, student, [][]float32{{1, 0}}, Config{})
	if err != nil {
		t.Fatalf("BatchLoss() error = %v", err)
	}
	if loss.Tokens != 1 {
		t.Fatalf("Tokens = %d, want 1 (only the unmasked position counted)", loss.Tokens)
	}
}

// --- BatchCacheKey ---

// Good: the same inputs hash to the same key every time, and different
// inputs hash to different keys.
func TestBatchCacheKey_Good(t *testing.T) {
	a := BatchCacheKey([][]int{{1, 2}}, [][]int{{2, 3}}, [][]float32{{1, 1}})
	b := BatchCacheKey([][]int{{1, 2}}, [][]int{{2, 3}}, [][]float32{{1, 1}})
	if a == "" {
		t.Fatal("BatchCacheKey() = empty string, want a stable hash")
	}
	if a != b {
		t.Fatalf("BatchCacheKey() not deterministic: %q vs %q", a, b)
	}
	c := BatchCacheKey([][]int{{9, 9}}, [][]int{{2, 3}}, [][]float32{{1, 1}})
	if a == c {
		t.Fatalf("BatchCacheKey() produced the same key for different tokens: %q", a)
	}
}

// Bad: a NaN in the mask can't be JSON-encoded, so BatchCacheKey falls
// back to a Sprintf-based key rather than panicking or returning empty.
func TestBatchCacheKey_Bad(t *testing.T) {
	key := BatchCacheKey([][]int{{1}}, [][]int{{2}}, [][]float32{{float32(math.NaN())}})
	if key == "" {
		t.Fatal("BatchCacheKey() with NaN mask = empty string, want the Sprintf fallback hash")
	}
	again := BatchCacheKey([][]int{{1}}, [][]int{{2}}, [][]float32{{float32(math.NaN())}})
	if key != again {
		t.Fatalf("BatchCacheKey() fallback not deterministic: %q vs %q", key, again)
	}
}

// Ugly: nil and empty (non-nil, zero-length) slices hash differently —
// the JSON emitter preserves the null vs [] distinction encoding/json
// would produce.
func TestBatchCacheKey_Ugly(t *testing.T) {
	nilKey := BatchCacheKey(nil, nil, nil)
	emptyKey := BatchCacheKey([][]int{}, [][]int{}, [][]float32{})
	if nilKey == "" || emptyKey == "" {
		t.Fatal("BatchCacheKey() with nil/empty input = empty string, want a stable hash")
	}
	if nilKey == emptyKey {
		t.Fatal("BatchCacheKey(nil) and BatchCacheKey(empty) produced the same key, want them distinct")
	}
}
