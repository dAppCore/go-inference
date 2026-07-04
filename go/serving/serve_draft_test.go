// SPDX-Licence-Identifier: EUPL-1.2

package serving

import (
	"testing"

	core "dappco.re/go"
)

// writeGemma4Pack writes a minimal gemma4 model pack (config.json declaring a
// gemma4 model_type + a dummy weights shard) into dir.
func writeGemma4Pack(t *testing.T, dir string) {
	t.Helper()
	if r := core.WriteFile(core.PathJoin(dir, "config.json"), []byte(`{"model_type":"gemma4_text"}`), 0o644); !r.OK {
		t.Fatalf("write config: %v", r.Value)
	}
	if r := core.WriteFile(core.PathJoin(dir, "model.safetensors"), []byte("weights"), 0o644); !r.OK {
		t.Fatalf("write weights: %v", r.Value)
	}
}

// TestDetectGemma4DraftPath_Explicit_Good proves an explicit --draft path wins
// the ladder outright, regardless of the target architecture.
func TestDetectGemma4DraftPath_Explicit_Good(t *testing.T) {
	det := DetectGemma4DraftPath("/models/target", "/models/drafter", DraftDetectOptions{})
	if det.Source != DraftSourceFlag || det.DraftPath != "/models/drafter" {
		t.Fatalf("explicit --draft should win: %+v", det)
	}
	if !det.Active() {
		t.Fatal("explicit drafter should be Active")
	}
}

// TestDetectGemma4DraftPath_Disabled_Bad proves -draft-detect=false stands the
// reactive ladder down (only an explicit path could still engage a drafter).
func TestDetectGemma4DraftPath_Disabled_Bad(t *testing.T) {
	dir := t.TempDir()
	writeGemma4Pack(t, dir)
	if r := core.MkdirAll(core.PathJoin(dir, "assistant"), 0o755); !r.OK {
		t.Fatalf("mkdir assistant: %v", r.Value)
	}
	writeGemma4Pack(t, core.PathJoin(dir, "assistant"))
	det := DetectGemma4DraftPath(dir, "", DraftDetectOptions{Disabled: true})
	if det.Active() {
		t.Fatalf("disabled detection should stand down even with an assistant/ present: %+v", det)
	}
}

// TestDetectGemma4DraftPath_NonGemma4_Bad proves a non-Gemma4 target never
// auto-detects a drafter (detection is Gemma4-family only).
func TestDetectGemma4DraftPath_NonGemma4_Bad(t *testing.T) {
	dir := t.TempDir()
	if r := core.WriteFile(core.PathJoin(dir, "config.json"), []byte(`{"model_type":"llama"}`), 0o644); !r.OK {
		t.Fatalf("write config: %v", r.Value)
	}
	det := DetectGemma4DraftPath(dir, "", DraftDetectOptions{})
	if det.Active() {
		t.Fatalf("non-gemma4 target should get no auto drafter: %+v", det)
	}
}

// TestDetectGemma4DraftPath_AssistantDir_Ugly proves the assistant/ rung fires
// when a loadable safetensors assistant pack sits beside a gemma4 target.
func TestDetectGemma4DraftPath_AssistantDir_Ugly(t *testing.T) {
	dir := t.TempDir()
	writeGemma4Pack(t, dir)
	assistant := core.PathJoin(dir, "assistant")
	if r := core.MkdirAll(assistant, 0o755); !r.OK {
		t.Fatalf("mkdir assistant: %v", r.Value)
	}
	writeGemma4Pack(t, assistant)
	det := DetectGemma4DraftPath(dir, "", DraftDetectOptions{})
	if det.Source != DraftSourceAssistantDir || det.DraftPath != assistant {
		t.Fatalf("assistant/ beside gemma4 weights should be detected: %+v", det)
	}
	if !det.Active() {
		t.Fatal("detected assistant/ drafter should be Active")
	}
}

// TestResolveServeDraft_AutoDetects_Good proves the "auto" flag value routes
// into the reactive ladder (an explicit path still forces the drafter).
func TestResolveServeDraft_AutoDetects_Good(t *testing.T) {
	dir := t.TempDir()
	writeGemma4Pack(t, dir)
	assistant := core.PathJoin(dir, "assistant")
	if r := core.MkdirAll(assistant, 0o755); !r.OK {
		t.Fatalf("mkdir assistant: %v", r.Value)
	}
	writeGemma4Pack(t, assistant)
	det := ResolveServeDraft(dir, "auto", true)
	if det.Source != DraftSourceAssistantDir {
		t.Fatalf("auto should run the ladder and find assistant/: %+v", det)
	}
}
