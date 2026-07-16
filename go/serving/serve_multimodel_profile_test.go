// SPDX-Licence-Identifier: EUPL-1.2

package serving

import (
	"context"
	"iter"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference"
)

// ptrFloat32 / ptrInt / ptrUint64 build the pointer fields ProfileConfig uses to
// distinguish "set" from "unset".
func ptrFloat32(v float32) *float32 { return &v }
func ptrInt(v int) *int             { return &v }
func ptrUint64(v uint64) *uint64    { return &v }

// applyOpts folds a slice of GenerateOptions onto a default config so a test can
// read back the resolved knobs a preset+caller pair produced.
func applyOpts(opts []inference.GenerateOption) inference.GenerateConfig {
	cfg := inference.DefaultGenerateConfig()
	for _, o := range opts {
		o(&cfg)
	}
	return cfg
}

// presetSpyModel records the resolved GenerateConfig of the last generation call
// so the profile decorator's option ordering is observable. It embeds a base
// mockTextModel for the non-option surface and overrides the four option-taking
// methods.
type presetSpyModel struct {
	inference.TextModel
	lastCfg  inference.GenerateConfig
	lastCall string
}

func newPresetSpy() *presetSpyModel {
	return &presetSpyModel{TextModel: &mockTextModel{}}
}

func (m *presetSpyModel) record(call string, opts []inference.GenerateOption) {
	m.lastCall = call
	m.lastCfg = applyOpts(opts)
}

func (m *presetSpyModel) Generate(_ context.Context, _ string, opts ...inference.GenerateOption) iter.Seq[inference.Token] {
	m.record("generate", opts)
	return func(func(inference.Token) bool) {}
}

func (m *presetSpyModel) Chat(_ context.Context, _ []inference.Message, opts ...inference.GenerateOption) iter.Seq[inference.Token] {
	m.record("chat", opts)
	return func(func(inference.Token) bool) {}
}

func (m *presetSpyModel) Classify(_ context.Context, _ []string, opts ...inference.GenerateOption) core.Result {
	m.record("classify", opts)
	return core.Ok(nil)
}

func (m *presetSpyModel) BatchGenerate(_ context.Context, _ []string, opts ...inference.GenerateOption) core.Result {
	m.record("batch", opts)
	return core.Ok(nil)
}

// drain runs an iterator to completion so a lazy Generate/Chat body executes.
func drain(seq iter.Seq[inference.Token]) {
	for range seq {
	}
}

// TestProfileConfig_Options_Good proves each set field emits its option and the
// resolved config carries the intended values.
func TestProfileConfig_Options_Good(t *testing.T) {
	p := ProfileConfig{
		Temperature:   ptrFloat32(0.9),
		TopP:          ptrFloat32(0.95),
		TopK:          ptrInt(40),
		MinP:          ptrFloat32(0.05),
		MaxTokens:     ptrInt(512),
		RepeatPenalty: ptrFloat32(1.1),
		Seed:          ptrUint64(7),
	}
	cfg := applyOpts(p.Options())
	if cfg.Temperature != 0.9 || cfg.TopP != 0.95 || cfg.TopK != 40 || cfg.MinP != 0.05 || cfg.MaxTokens != 512 || cfg.RepeatPenalty != 1.1 || cfg.Seed != 7 {
		t.Fatalf("ProfileConfig.Options resolved to %+v, want the seven set knobs", cfg)
	}
}

// TestProfileConfig_Options_Unset_Good proves an unset field emits no option, so
// the model/engine default is left in place (a preset overrides only what it
// names — the pointer-field contract).
func TestProfileConfig_Options_Unset_Good(t *testing.T) {
	p := ProfileConfig{MaxTokens: ptrInt(64)} // only max tokens set
	opts := p.Options()
	if len(opts) != 1 {
		t.Fatalf("ProfileConfig{MaxTokens} emitted %d options, want 1 (only the set field)", len(opts))
	}
	cfg := applyOpts(opts)
	if cfg.MaxTokens != 64 {
		t.Fatalf("MaxTokens = %d, want 64", cfg.MaxTokens)
	}
	// Temperature stays at the greedy default, untouched by the preset.
	if cfg.Temperature != 0.0 {
		t.Fatalf("Temperature = %v, want 0.0 (unset by the preset)", cfg.Temperature)
	}
}

// TestWrapProfile_EmptyPreset_ReturnsBase_Good proves an empty preset skips the
// decorator hop entirely — no reason to wrap a profile that sets nothing.
func TestWrapProfile_EmptyPreset_ReturnsBase_Good(t *testing.T) {
	base := newPresetSpy()
	got := wrapProfile(base, nil)
	if got != inference.TextModel(base) {
		t.Fatal("wrapProfile(model, nil) must return the base model unwrapped")
	}
}

// TestProfileModel_Chat_AppliesPreset_Good proves the preset options reach the
// wrapped model on a Chat call.
func TestProfileModel_Chat_AppliesPreset_Good(t *testing.T) {
	spy := newPresetSpy()
	preset := ProfileConfig{MaxTokens: ptrInt(7), Temperature: ptrFloat32(0.3)}.Options()
	wrapped := wrapProfile(spy, preset)

	drain(wrapped.Chat(context.Background(), []inference.Message{{Role: "user", Content: "hi"}}))
	if spy.lastCall != "chat" {
		t.Fatalf("last call = %q, want chat", spy.lastCall)
	}
	if spy.lastCfg.MaxTokens != 7 || spy.lastCfg.Temperature != 0.3 {
		t.Fatalf("preset not applied: got MaxTokens=%d Temperature=%v, want 7 / 0.3", spy.lastCfg.MaxTokens, spy.lastCfg.Temperature)
	}
}

// TestProfileModel_CallerWins_Ugly pins the ordering contract: when the preset
// AND the caller both set the same knob, the caller wins (Go options are
// last-set-wins and the preset is prepended). A profile establishes defaults a
// request can still override — the subtle correctness the whole design rests on.
func TestProfileModel_CallerWins_WrapProfile_Ugly(t *testing.T) {
	spy := newPresetSpy()
	preset := ProfileConfig{Temperature: ptrFloat32(0.2), MaxTokens: ptrInt(100)}.Options()
	wrapped := wrapProfile(spy, preset)

	// Caller overrides Temperature; leaves MaxTokens to the preset.
	drain(wrapped.Generate(context.Background(), "prompt", inference.WithTemperature(0.9)))
	if spy.lastCfg.Temperature != 0.9 {
		t.Fatalf("Temperature = %v, want 0.9 (caller must win over preset 0.2)", spy.lastCfg.Temperature)
	}
	if spy.lastCfg.MaxTokens != 100 {
		t.Fatalf("MaxTokens = %d, want 100 (preset applies where the caller is silent)", spy.lastCfg.MaxTokens)
	}
}

// TestProfileModel_ClassifyBatch_ApplyPreset_Good covers the remaining two
// option-taking methods so the decorator is proven complete, not just on the two
// streaming paths.
func TestProfileModel_ClassifyBatch_ApplyPreset_WrapProfile_Good(t *testing.T) {
	spy := newPresetSpy()
	wrapped := wrapProfile(spy, ProfileConfig{MaxTokens: ptrInt(11)}.Options())

	wrapped.Classify(context.Background(), []string{"a"})
	if spy.lastCall != "classify" || spy.lastCfg.MaxTokens != 11 {
		t.Fatalf("Classify: call=%q MaxTokens=%d, want classify / 11", spy.lastCall, spy.lastCfg.MaxTokens)
	}
	wrapped.BatchGenerate(context.Background(), []string{"a"})
	if spy.lastCall != "batch" || spy.lastCfg.MaxTokens != 11 {
		t.Fatalf("BatchGenerate: call=%q MaxTokens=%d, want batch / 11", spy.lastCall, spy.lastCfg.MaxTokens)
	}
}

// mediaCapableFake is a TextModel that also carries both media capabilities, so
// the profile decorator's gate forwarding is observable.
type mediaCapableFake struct{ inference.TextModel }

func (mediaCapableFake) AcceptsImages() bool { return true }
func (mediaCapableFake) AcceptsAudio() bool  { return true }

// TestProfileModel_ForwardsCapabilityGates_Good: the profile wrap must not hide
// the wrapped checkpoint's media capabilities — the serve handler gates image /
// audio input on these assertions and the embedded interface does not widen the
// decorator's method set by itself.
func TestProfileModel_ForwardsCapabilityGates_WrapProfile_Good(t *testing.T) {
	inner := mediaCapableFake{TextModel: &mockTextModel{}}
	wrapped := wrapProfile(inner, ProfileConfig{MaxTokens: ptrInt(4)}.Options())
	v, ok := wrapped.(inference.VisionModel)
	if !ok || !v.AcceptsImages() {
		t.Fatalf("profile wrap hides AcceptsImages (ok=%v) — image serve gate 400s", ok)
	}
	a, ok := wrapped.(inference.AudioModel)
	if !ok || !a.AcceptsAudio() {
		t.Fatalf("profile wrap hides AcceptsAudio (ok=%v) — audio serve gate 400s", ok)
	}
}

// embedCapableFake is a TextModel that also implements EmbeddingModel, so the
// Unwrap seam is observable at the capability gate.
type embedCapableFake struct {
	inference.TextModel
	embedCalled bool
}

func (e *embedCapableFake) Embed(context.Context, inference.EmbeddingRequest) (*inference.EmbeddingResult, error) {
	e.embedCalled = true
	return &inference.EmbeddingResult{}, nil
}

// TestProfileModel_Unwrap_ReachesEmbedding_Ugly is the capability-stripping
// guard: a profile decorator does not re-expose EmbeddingModel, so the
// /v1/embeddings gate reaches the base model via inference.BaseTextModel. Without
// Unwrap the embedder is invisible and the embeddings route 404s.
func TestProfileModel_Unwrap_ReachesEmbedding_Ugly(t *testing.T) {
	base := &embedCapableFake{TextModel: &mockTextModel{}}
	wrapped := wrapProfile(base, ProfileConfig{Temperature: ptrFloat32(0.1)}.Options())

	// Direct assertion fails (the decorator does not carry EmbeddingModel)...
	if _, ok := wrapped.(inference.EmbeddingModel); ok {
		t.Fatal("profile decorator should NOT itself satisfy EmbeddingModel — the gate must unwrap")
	}
	// ...but BaseTextModel walks past it to the base embedder.
	embedder, ok := inference.BaseTextModel(wrapped).(inference.EmbeddingModel)
	if !ok {
		t.Fatal("BaseTextModel(profiled embedder) must reach the base EmbeddingModel")
	}
	if _, err := embedder.Embed(context.Background(), inference.EmbeddingRequest{}); err != nil {
		t.Fatalf("Embed through unwrapped model: %v", err)
	}
	if !base.embedCalled {
		t.Fatal("Embed did not reach the base model")
	}
}
