// SPDX-Licence-Identifier: EUPL-1.2

package serving

import (
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference/model"
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

// ---------------------------------------------------------------------------
// v0.9.0 shape triplets — Test<File>_<Symbol>_{Good,Bad,Ugly}
// ---------------------------------------------------------------------------

// TestServeDraft_DraftDetection_Active_Good proves a resolved source plus a
// non-empty path reports Active.
func TestServeDraft_DraftDetection_Active_Good(t *testing.T) {
	det := DraftDetection{Source: DraftSourceFlag, DraftPath: "/models/drafter"}
	if !det.Active() {
		t.Fatal("a resolved source with a draft path should be Active")
	}
}

// TestServeDraft_DraftDetection_Active_Bad proves the zero value (no ladder
// rung fired) is never Active.
func TestServeDraft_DraftDetection_Active_Bad(t *testing.T) {
	var det DraftDetection
	if det.Active() {
		t.Fatal("the zero-value DraftDetection should never be Active")
	}
}

// TestServeDraft_DraftDetection_Active_Ugly proves a source without a path
// (an internal inconsistency no ladder rung actually produces) still reports
// inactive — Active requires BOTH a resolved source AND a non-empty path.
func TestServeDraft_DraftDetection_Active_Ugly(t *testing.T) {
	det := DraftDetection{Source: DraftSourceMTPDir, DraftPath: ""}
	if det.Active() {
		t.Fatal("a source with no draft path should not be Active")
	}
}

// TestServeDraft_SpeculativeServeNotice_Good proves an active detection renders
// a notice naming the drafter path, the operator note, and the resolved block.
func TestServeDraft_SpeculativeServeNotice_Good(t *testing.T) {
	det := DraftDetection{Source: DraftSourceFlag, DraftPath: "/models/drafter", Note: "flag override"}
	notice := speculativeServeNotice(det, 8)
	for _, want := range []string{"/models/drafter", "flag override", "block 8"} {
		if !core.Contains(notice, want) {
			t.Errorf("notice %q missing %q", notice, want)
		}
	}
}

// TestServeDraft_SpeculativeServeNotice_Bad proves a stood-down detector serves
// no notice — a drafterless serve prints nothing.
func TestServeDraft_SpeculativeServeNotice_Bad(t *testing.T) {
	if notice := speculativeServeNotice(DraftDetection{}, 8); notice != "" {
		t.Fatalf("inactive detection notice = %q, want empty", notice)
	}
}

// TestServeDraft_SpeculativeServeNotice_Ugly proves a non-positive draft block
// falls back to MTPDefaultDraftBlock in the rendered notice rather than
// printing "block 0".
func TestServeDraft_SpeculativeServeNotice_Ugly(t *testing.T) {
	det := DraftDetection{Source: DraftSourceFlag, DraftPath: "/models/drafter", Note: "flag override"}
	notice := speculativeServeNotice(det, 0)
	if !core.Contains(notice, core.Sprintf("block %d", MTPDefaultDraftBlock)) {
		t.Fatalf("notice %q should fall back to the default block %d", notice, MTPDefaultDraftBlock)
	}
}

// TestServeDraft_DetectGemma4DraftPath_Good proves the MTP/ subdirectory rung
// (rung 3a, the unsloth GGUF convention) fires when it carries exactly one
// gguf.
func TestServeDraft_DetectGemma4DraftPath_Good(t *testing.T) {
	dir := t.TempDir()
	writeGemma4Pack(t, dir)
	mtpDir := core.PathJoin(dir, "MTP")
	if r := core.MkdirAll(mtpDir, 0o755); !r.OK {
		t.Fatalf("mkdir MTP: %v", r.Value)
	}
	gguf := core.PathJoin(mtpDir, "drafter.gguf")
	if r := core.WriteFile(gguf, []byte("gguf"), 0o644); !r.OK {
		t.Fatalf("write gguf: %v", r.Value)
	}
	det := DetectGemma4DraftPath(dir, "", DraftDetectOptions{})
	if det.Source != DraftSourceMTPDir || det.DraftPath != gguf {
		t.Fatalf("MTP/ single-gguf rung should fire: %+v", det)
	}
}

// TestServeDraft_DetectGemma4DraftPath_Bad proves an ambiguous MTP/ directory
// (more than one gguf — the rung can't pick a winner) falls through to no
// detection rather than guessing.
func TestServeDraft_DetectGemma4DraftPath_Bad(t *testing.T) {
	dir := t.TempDir()
	writeGemma4Pack(t, dir)
	mtpDir := core.PathJoin(dir, "MTP")
	if r := core.MkdirAll(mtpDir, 0o755); !r.OK {
		t.Fatalf("mkdir MTP: %v", r.Value)
	}
	for _, name := range []string{"a.gguf", "b.gguf"} {
		if r := core.WriteFile(core.PathJoin(mtpDir, name), []byte("gguf"), 0o644); !r.OK {
			t.Fatalf("write %s: %v", name, r.Value)
		}
	}
	det := DetectGemma4DraftPath(dir, "", DraftDetectOptions{})
	if det.Active() {
		t.Fatalf("an ambiguous MTP/ directory should not auto-detect a drafter: %+v", det)
	}
}

// TestServeDraft_DetectGemma4DraftPath_Ugly proves the sibling mtp-*.gguf
// rung (rung 3b) fires when no assistant/ or MTP/ rung matched first.
func TestServeDraft_DetectGemma4DraftPath_Ugly(t *testing.T) {
	dir := t.TempDir()
	writeGemma4Pack(t, dir)
	gguf := core.PathJoin(dir, "mtp-draft.gguf")
	if r := core.WriteFile(gguf, []byte("gguf"), 0o644); !r.OK {
		t.Fatalf("write sibling gguf: %v", r.Value)
	}
	det := DetectGemma4DraftPath(dir, "", DraftDetectOptions{})
	if det.Source != DraftSourceMTPSibling || det.DraftPath != gguf {
		t.Fatalf("sibling mtp-*.gguf rung should fire: %+v", det)
	}
}

// TestServeDraft_ResolveServeDraft_Good proves "auto" runs the reactive
// ladder and finds a qualifying drafter.
func TestServeDraft_ResolveServeDraft_Good(t *testing.T) {
	dir := t.TempDir()
	writeGemma4Pack(t, dir)
	assistant := core.PathJoin(dir, "assistant")
	if r := core.MkdirAll(assistant, 0o755); !r.OK {
		t.Fatalf("mkdir assistant: %v", r.Value)
	}
	writeGemma4Pack(t, assistant)
	det := ResolveServeDraft(dir, "auto", true)
	if det.Source != DraftSourceAssistantDir {
		t.Fatalf("ResolveServeDraft(auto) should find assistant/: %+v", det)
	}
}

// TestServeDraft_ResolveServeDraft_Bad proves an explicit "" flag value
// stands the ladder down even when a qualifying assistant/ sits right there
// — "" always means disabled, regardless of the --draft-detect value.
func TestServeDraft_ResolveServeDraft_Bad(t *testing.T) {
	dir := t.TempDir()
	writeGemma4Pack(t, dir)
	assistant := core.PathJoin(dir, "assistant")
	if r := core.MkdirAll(assistant, 0o755); !r.OK {
		t.Fatalf("mkdir assistant: %v", r.Value)
	}
	writeGemma4Pack(t, assistant)
	det := ResolveServeDraft(dir, "", true)
	if det.Active() {
		t.Fatalf("ResolveServeDraft(\"\") should disable the ladder: %+v", det)
	}
}

// TestServeDraft_ResolveServeDraft_Ugly proves an explicit non-auto,
// non-empty flag value forces that exact path — bypassing the ladder outright
// even with detect=false.
func TestServeDraft_ResolveServeDraft_Ugly(t *testing.T) {
	det := ResolveServeDraft("/models/target", "/models/forced-drafter", false)
	if det.Source != DraftSourceFlag || det.DraftPath != "/models/forced-drafter" {
		t.Fatalf("ResolveServeDraft(explicit path) should force that drafter: %+v", det)
	}
}

// writeDFlashDraft writes a minimal DFlash speculator checkpoint dir (config.json
// carrying the speculators_model_type marker + the drafter contract fields).
func writeDFlashDraft(t *testing.T, dir string) {
	t.Helper()
	cfg := `{"speculators_model_type":"dflash","block_size":8,` +
		`"aux_hidden_state_layer_ids":[3,13,23,32,42],` +
		`"speculators_config":{"verifier":{"name":"deepseek-ai/DeepSeek-V4-Flash"}}}`
	if r := core.WriteFile(core.PathJoin(dir, "config.json"), []byte(cfg), 0o644); !r.OK {
		t.Fatalf("write dflash config: %v", r.Value)
	}
}

// TestDetectDFlashDraft_Good proves a DFlash checkpoint is recognised by its
// config marker and its drafter contract (block, fused layers, verifier) read.
func TestDetectDFlashDraft_Good(t *testing.T) {
	dir := t.TempDir()
	writeDFlashDraft(t, dir)
	cfg, ok := DetectDFlashDraft(dir)
	if !ok {
		t.Fatal("a speculators_model_type=dflash checkpoint must be recognised")
	}
	if cfg.BlockSize != 8 || len(cfg.AuxHiddenLayerIDs) != 5 || cfg.Verifier != "deepseek-ai/DeepSeek-V4-Flash" {
		t.Fatalf("drafter contract not read: %+v", cfg)
	}
}

// TestDetectDFlashDraft_Bad proves a non-DFlash pack (a plain gemma4 target) and a
// path with no config.json are both declined — recognition is marker-driven.
func TestDetectDFlashDraft_Bad(t *testing.T) {
	dir := t.TempDir()
	writeGemma4Pack(t, dir)
	if _, ok := DetectDFlashDraft(dir); ok {
		t.Fatal("a gemma4 pack is not a DFlash drafter")
	}
	if _, ok := DetectDFlashDraft(t.TempDir()); ok {
		t.Fatal("a dir with no config.json is not a DFlash drafter")
	}
	if _, ok := DetectDFlashDraft(""); ok {
		t.Fatal("an empty path is not a DFlash drafter")
	}
}

// TestResolveServeDraft_DFlash_StampsMethod proves an explicit --draft path to a
// DFlash checkpoint resolves to an Active detection stamped model.MTPDFlash, so
// the boot path recognises it (IsDFlash) and declines the MTP lane.
func TestResolveServeDraft_DFlash_StampsMethod(t *testing.T) {
	dir := t.TempDir()
	writeDFlashDraft(t, dir)
	det := ResolveServeDraft("/models/target", dir, true)
	if !det.Active() {
		t.Fatalf("an explicit DFlash --draft path should be Active: %+v", det)
	}
	if det.Method != model.MTPDFlash || !det.IsDFlash() {
		t.Fatalf("a DFlash drafter must be stamped MTPDFlash: %+v", det)
	}
}

// TestResolveServeDraft_NonDFlash_NoStamp proves an explicit non-DFlash drafter
// path is left unstamped (empty method, IsDFlash false) so the MTP lane still
// arms for the ordinary draft-model case.
func TestResolveServeDraft_NonDFlash_NoStamp(t *testing.T) {
	dir := t.TempDir()
	writeGemma4Pack(t, dir) // a plain pack, not a DFlash checkpoint
	det := ResolveServeDraft("/models/target", dir, true)
	if det.IsDFlash() {
		t.Fatalf("a non-DFlash drafter must not be stamped MTPDFlash: %+v", det)
	}
}

// TestDFlashDraftNotice_Good proves the honest notice names the block, the fused
// layer count, and the verifier, and states the plain-autoregressive degrade with
// a pointer to the design memo — the declared-not-linked posture, not a faked lane.
func TestDFlashDraftNotice_Good(t *testing.T) {
	dir := t.TempDir()
	writeDFlashDraft(t, dir)
	det := ResolveServeDraft("/models/target", dir, true)
	notice := DFlashDraftNotice(det)
	for _, want := range []string{"DFlash", "block 8", "5 fused", "deepseek-ai/DeepSeek-V4-Flash", "plain autoregressive", "docs/design-dflash.md"} {
		if !core.Contains(notice, want) {
			t.Fatalf("notice missing %q: %s", want, notice)
		}
	}
}

// TestDFlashDraftNotice_Bad proves the notice is empty for a detection that is not
// a DFlash drafter (nothing to announce).
func TestDFlashDraftNotice_Bad(t *testing.T) {
	det := DraftDetection{Source: DraftSourceFlag, DraftPath: "/models/plain-drafter"}
	if notice := DFlashDraftNotice(det); notice != "" {
		t.Fatalf("a non-DFlash detection yields no DFlash notice: %q", notice)
	}
}

// TestArmDFlash_EngineSupported_Good proves that when the engine declares it can run
// the block-diffusion lane (DFlashEngineProbe true), a detected DFlash drafter ARMS
// and the active notice states the lossless block-verify posture.
func TestArmDFlash_EngineSupported_Good(t *testing.T) {
	prev := DFlashEngineProbe
	DFlashEngineProbe = func() bool { return true }
	t.Cleanup(func() { DFlashEngineProbe = prev })

	dir := t.TempDir()
	writeDFlashDraft(t, dir)
	det := ResolveServeDraft("/models/target", dir, true)
	if !ArmDFlash(det) {
		t.Fatal("a DFlash drafter must arm when the engine declares support")
	}
	notice := DFlashActiveNotice(det)
	for _, want := range []string{"ACTIVE", "block 8", "5 fused", "lossless", "docs/design-dflash.md"} {
		if !core.Contains(notice, want) {
			t.Fatalf("active notice missing %q: %s", want, notice)
		}
	}
}

// TestArmDFlash_EngineUnsupported_Bad proves the shipped default declines: with no
// engine support (the default probe), a DFlash drafter does NOT arm — serve keeps the
// honest plain-autoregressive decline, never a fake or losslessness-breaking lane.
func TestArmDFlash_EngineUnsupported_Bad(t *testing.T) {
	dir := t.TempDir()
	writeDFlashDraft(t, dir)
	det := ResolveServeDraft("/models/target", dir, true)
	if ArmDFlash(det) {
		t.Fatal("a DFlash drafter must NOT arm without engine support (default probe)")
	}
}

// TestArmDFlash_NonDFlash_Ugly proves a non-DFlash detection never arms the DFlash
// lane even with the probe up, and yields no active notice.
func TestArmDFlash_NonDFlash_Ugly(t *testing.T) {
	prev := DFlashEngineProbe
	DFlashEngineProbe = func() bool { return true }
	t.Cleanup(func() { DFlashEngineProbe = prev })

	det := DraftDetection{Source: DraftSourceFlag, DraftPath: "/models/plain-drafter"}
	if ArmDFlash(det) {
		t.Fatal("a non-DFlash drafter must never arm the DFlash lane")
	}
	if DFlashActiveNotice(det) != "" {
		t.Fatal("a non-DFlash detection yields no DFlash active notice")
	}
}
