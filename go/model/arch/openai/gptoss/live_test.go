// SPDX-Licence-Identifier: EUPL-1.2

package gptoss

import (
	"os"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference/model"
)

// live_test.go is the #37 tracker's Stage 3 gate — RUNTIME-gated on a real, on-disk checkpoint rather
// than the committed testdata fixture, reading the checkpoint dir from GPTOSS_MLX_DIR (skips cleanly
// when unset, exactly as briefed) and defaulting to the standard Hugging Face cache location for
// InferenceIllusionist/gpt-oss-20b-MLX-4bit (mirrors engine/metal's resolveMoE26BDir convention: env
// override, else auto-resolve the hash-named snapshot subdirectory).
//
// It deliberately does NOT additionally gate on MLX_METALLIB_PATH and does NOT load the model or assert
// a generation — Config.Arch still refuses (see config.go's doc: attention sinks + o_proj/router/expert
// biases + the YaRN mscale postscale have no engine/metal consumer yet), so model.Load cannot reach
// Assemble for gpt_oss. Writing a "loads the model, greedy-generates, asserts 'Paris'" test now would
// either assert a failure dressed up as success, or sit dead until a future pass lands the missing
// pieces — exactly the "fake whole" this package's rules forbid. What CAN be verified honestly today,
// against the REAL checkpoint rather than a committed fixture, is that Parse resolves the on-disk
// config.json into the correct geometry AND that Arch's refusal is the PRECISE named boundary, not a
// generic failure — and unlike the full generation test, this one needs no GPU, so it is verified HERE,
// not deferred to the orchestrator. The generation assertion is the natural next test once sinks/biases
// land; it belongs beside whatever change lands them, not as inert dead code here.
func resolveGptOssDir(t *testing.T) string {
	t.Helper()
	if dir := os.Getenv("GPTOSS_MLX_DIR"); dir != "" {
		return dir
	}
	home := os.Getenv("HOME")
	base := home + "/.cache/huggingface/hub/models--InferenceIllusionist--gpt-oss-20b-MLX-4bit/snapshots"
	entries, err := os.ReadDir(base)
	if err != nil {
		t.Skipf("gpt-oss-20b-MLX-4bit snapshot dir not found (%v) — set GPTOSS_MLX_DIR to override", err)
	}
	for _, e := range entries {
		if e.IsDir() {
			return base + "/" + e.Name()
		}
	}
	t.Skip("gpt-oss-20b-MLX-4bit snapshots dir has no snapshot subdirectory")
	return ""
}

// TestLive_RealCheckpoint_ParseAndBoundary is the Stage 3 gate: against the REAL, locally-cached
// InferenceIllusionist/gpt-oss-20b-MLX-4bit checkpoint (not the committed testdata fixture), Parse
// resolves the actual geometry and Arch's refusal names the actual boundary — proving this pass's work
// against the checkpoint it was verified against throughout, not just a frozen JSON snapshot of it.
func TestLive_RealCheckpoint_ParseAndBoundary(t *testing.T) {
	dir := resolveGptOssDir(t)
	data, err := os.ReadFile(dir + "/config.json")
	if err != nil {
		t.Skipf("read %s/config.json: %v", dir, err)
	}

	spec, ok := model.LookupArch("gpt_oss")
	if !ok {
		t.Fatal("model_type \"gpt_oss\" not registered")
	}
	ac, err := spec.Parse(data)
	if err != nil {
		t.Fatalf("Parse(real checkpoint config.json): %v", err)
	}
	cfg, ok := ac.(*Config)
	if !ok {
		t.Fatalf("Parse returned %T, want *Config", ac)
	}

	// The geometry this pass was verified against throughout (config.go/weights.go/yarn.go doc comments
	// cite these exact numbers, read from the checkpoint's own config.json and safetensors index).
	if cfg.HiddenSize != 2880 || cfg.NumHiddenLayers != 24 || cfg.NumAttentionHeads != 64 || cfg.NumKeyValueHeads != 8 || cfg.HeadDim != 64 {
		t.Fatalf("real checkpoint attention geometry = %+v, want hidden 2880/layers 24/heads 64/kv 8/headDim 64", cfg)
	}
	if cfg.NumLocalExperts != 32 || cfg.resolvedExpertsPerTok() != 4 {
		t.Fatalf("real checkpoint MoE geometry = experts %d topK %d, want 32/4", cfg.NumLocalExperts, cfg.resolvedExpertsPerTok())
	}
	if cfg.SlidingWindow != 128 || len(cfg.LayerTypes) != 24 {
		t.Fatalf("real checkpoint layer geometry = sliding_window %d, %d layer_types, want 128/24", cfg.SlidingWindow, len(cfg.LayerTypes))
	}
	if cfg.RopeScaling.RopeType != "yarn" || cfg.RopeScaling.Factor != 32 || cfg.RopeScaling.OriginalMaxPositionEmbeddings != 4096 || cfg.RopeScaling.Truncate {
		t.Fatalf("real checkpoint YaRN config = %+v, want yarn/factor 32/orig 4096/truncate false", cfg.RopeScaling)
	}
	if cfg.VocabSize != 201088 {
		t.Fatalf("real checkpoint vocab_size = %d, want 201088", cfg.VocabSize)
	}

	// The geometry resolves cleanly against this REAL config (proving buildArch works end-to-end on the
	// actual checkpoint, not just the hand-built fixtures in arch_test.go/config_test.go) — and Arch
	// still refuses, precisely, naming the checkpoint-real gap.
	a, err := cfg.buildArch()
	if err != nil {
		t.Fatalf("buildArch(real checkpoint config): %v", err)
	}
	if len(a.RopeFreqs) != a.RotaryDim/2 {
		t.Fatalf("real checkpoint YaRN RopeFreqs length = %d, want %d (rotaryDim/2)", len(a.RopeFreqs), a.RotaryDim/2)
	}
	_, err = cfg.Arch()
	if err == nil {
		t.Fatal("Arch(real checkpoint config) unexpectedly succeeded — gpt_oss must not claim to serve yet")
	}
	if !core.Contains(err.Error(), "self_attn.sinks") || !core.Contains(err.Error(), "biases") {
		t.Fatalf("Arch(real checkpoint config) refusal %q must name the sinks/bias boundary precisely", err.Error())
	}
	t.Logf("real gpt-oss-20b-MLX-4bit checkpoint at %s: geometry + YaRN table resolve cleanly, boundary refusal precise: %v", dir, err)
}
