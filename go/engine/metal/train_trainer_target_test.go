// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"strings"
	"testing"

	"dappco.re/go/inference"
)

// train_trainer_target_test.go gates the #31 honesty seam host-side (no GPU, no model): a LoRA config
// requesting per-layer projection targets must be REFUSED loudly by the head-only trainer — never
// silently trained as head-only — and the refusal must name both the request and the supported target.

// TestValidateHeadLoRATargets_Good: no explicit targets means the engine's default (the head) — the
// config every current caller sends (cmd/lem sft passes rank/alpha only) passes untouched.
func TestValidateHeadLoRATargets_Good(t *testing.T) {
	if err := validateHeadLoRATargets(inference.LoRAConfig{Rank: 8, Alpha: 16}); err != nil {
		t.Fatalf("empty TargetKeys must pass (the head is the default target): %v", err)
	}
	if err := validateHeadLoRATargets(inference.LoRAConfig{TargetKeys: []string{}}); err != nil {
		t.Fatalf("zero-length TargetKeys must pass: %v", err)
	}
}

// TestValidateHeadLoRATargets_Bad: per-layer projection keys are refused, and the error text names the
// requested projections AND the supported target — the loud half of #31. DefaultLoRAConfig is the
// regression anchor: its q_proj/v_proj targets were silently dropped by the trainer before this gate.
func TestValidateHeadLoRATargets_Bad(t *testing.T) {
	err := validateHeadLoRATargets(inference.LoRAConfig{Rank: 8, Alpha: 16, TargetKeys: []string{"q_proj", "v_proj"}})
	if err == nil {
		t.Fatal("a q_proj/v_proj TargetKeys request must be refused by the head-only trainer")
	}
	msg := err.Error()
	for _, want := range []string{"q_proj", "v_proj", "lm_head", "per-layer projection"} {
		if !strings.Contains(msg, want) {
			t.Fatalf("refusal must name %q (request + supported target); got: %s", want, msg)
		}
	}

	// The default config IS a per-layer request — the exact shape that used to be silently unhonoured.
	if err := validateHeadLoRATargets(inference.DefaultLoRAConfig()); err == nil {
		t.Fatal("DefaultLoRAConfig targets q_proj/v_proj and must be refused, not silently trained as head-only")
	}

	// A mixed request is still a per-layer request: the head entry does not launder the layer entry.
	err = validateHeadLoRATargets(inference.LoRAConfig{TargetKeys: []string{"lm_head", "down_proj"}})
	if err == nil {
		t.Fatal("a mixed lm_head+down_proj request must be refused")
	}
	if !strings.Contains(err.Error(), "down_proj") {
		t.Fatalf("mixed-request refusal must name the unsupported key; got: %s", err.Error())
	}
}

// TestValidateHeadLoRATargets_Ugly: explicitly requesting the head (even repeated) is surprising but
// valid — it names exactly what the trainer trains, so it passes.
func TestValidateHeadLoRATargets_Ugly(t *testing.T) {
	if err := validateHeadLoRATargets(inference.LoRAConfig{TargetKeys: []string{"lm_head"}}); err != nil {
		t.Fatalf("an explicit lm_head target must pass: %v", err)
	}
	if err := validateHeadLoRATargets(inference.LoRAConfig{TargetKeys: []string{"lm_head", "lm_head"}}); err != nil {
		t.Fatalf("a repeated lm_head target must pass: %v", err)
	}
	// An empty-string key is not the head — a malformed request is refused, not guessed at.
	if err := validateHeadLoRATargets(inference.LoRAConfig{TargetKeys: []string{""}}); err == nil {
		t.Fatal("an empty-string target key must be refused")
	}
}

// TestNewLoRATrainer_Bad: the refusal sits at the trainer's open seam BEFORE any resource is touched —
// with no model at all, a per-layer config still gets the #31 refusal (not the nil-model error), and a
// head-shaped config still gets the nil-model error (existing behaviour intact).
func TestNewLoRATrainer_Bad(t *testing.T) {
	_, err := NewLoRATrainer(nil, inference.TrainingConfig{LoRA: inference.LoRAConfig{TargetKeys: []string{"q_proj"}}})
	if err == nil {
		t.Fatal("NewLoRATrainer must refuse a per-layer projection config")
	}
	if !strings.Contains(err.Error(), "q_proj") || !strings.Contains(err.Error(), "lm_head") {
		t.Fatalf("the open-seam refusal must name the request and the supported target; got: %s", err.Error())
	}

	_, err = NewLoRATrainer(nil, inference.TrainingConfig{})
	if err == nil {
		t.Fatal("NewLoRATrainer(nil, head-shaped cfg) must still fail on the missing model")
	}
	if !strings.Contains(err.Error(), "loaded bf16 model") {
		t.Fatalf("head-shaped config on a nil model keeps the model error; got: %s", err.Error())
	}
}
