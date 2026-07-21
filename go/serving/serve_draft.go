// SPDX-Licence-Identifier: EUPL-1.2

// serve_draft.go is the reactive Gemma 4 drafter detection ported out of
// lthn-mlx (go-mlx draft_detect.go + the serve command's draft helpers) so the
// serve business logic lives in a go-inference library rather than dying with
// go-mlx's cmd/. The model declares (by the files sitting beside it), the serve
// reacts. serve resolves a drafter through one ladder instead of requiring the
// operator to know the --draft flag exists:
//
//  1. an explicit --draft path always wins (and --draft="" disables);
//  2. an assistant/ subdirectory carrying a safetensors model — the
//     MTPLX/Google pair layout (a bundle root's target/ + assistant/, or an
//     assistant/ dropped inside the model directory) — or, for a target served
//     straight from an HF hub cache snapshot, the family's own cached
//     assistant repo (models--*--gemma-4-<size>-it-assistant*, bf16 first);
//  3. an MTP/ subdirectory or sibling mtp-*.gguf — the unsloth GGUF convention.
//
// Detection is path-shape only — no weights are opened. Whether the found
// drafter actually LOADS stays the resolver's business; serve reports the
// failure honestly on first load rather than refusing to boot.

package serving

import (
	core "dappco.re/go"
	"dappco.re/go/inference/decode/dflash"
	"dappco.re/go/inference/model/mtp"
)

// MTPDefaultDraftBlock is the dense-arch engine-default MTP draft block
// (verify forward = carried lead + block-1 proposals) applied when neither the
// --draft-block flag nor a tuned profile pins one. The LOADER owns the final
// resolution and is arch-aware — engine/metal resolves quant-MoE targets to a
// longer block (LoadSpeculativePair), so a 0 block must reach the loader
// rather than being pre-substituted with this constant.
const MTPDefaultDraftBlock = 5

// DraftDetectOptions is the typed settings surface for reactive drafter
// detection. The zero value means "detect".
type DraftDetectOptions struct {
	// Disabled turns the reactive ladder off: only an explicit --draft path
	// engages a drafter. The CLI binds --draft-detect=false onto it.
	Disabled bool
}

// DraftDetectionSource names which ladder rung produced a drafter path.
type DraftDetectionSource string

const (
	DraftSourceNone             DraftDetectionSource = ""                 // no drafter
	DraftSourceFlag             DraftDetectionSource = "flag"             // explicit --draft path
	DraftSourceAssistantDir     DraftDetectionSource = "assistant-dir"    // <model>/assistant/
	DraftSourceSiblingAssistant DraftDetectionSource = "assistant-pair"   // <bundle>/target + <bundle>/assistant
	DraftSourceCacheAssistant   DraftDetectionSource = "assistant-cache"  // hub cache models--*-assistant* repo
	DraftSourceMTPDir           DraftDetectionSource = "mtp-dir"          // <model>/MTP/*.gguf
	DraftSourceMTPSibling       DraftDetectionSource = "mtp-sibling-gguf" // <model>/mtp-*.gguf
)

// DraftDetection is the resolved drafter decision for a model path.
type DraftDetection struct {
	Source    DraftDetectionSource
	DraftPath string
	// Method is the speculative method the detected drafter uses. It is left
	// empty for the ordinary MTP draft-model path (mtp.MTPDraftModel, the
	// default the loader assumes); it is stamped mtp.MTPDFlash when the drafter
	// is a DFlash block-diffusion checkpoint — a method this engine cannot yet
	// run, so serve declines to arm it (see IsDFlash).
	Method mtp.MTPMethod
	// Note carries the operator-facing wording for the serve boot notice
	// (why this drafter engaged, or why detection stood down).
	Note string
}

// Active reports whether a drafter should be engaged.
func (d DraftDetection) Active() bool {
	return d.Source != DraftSourceNone && d.DraftPath != ""
}

// IsDFlash reports whether the detected drafter is a DFlash block-diffusion
// checkpoint. This engine has no block-diffusion draft forward yet (the fused
// multi-layer hidden extraction + parallel diffusion draft are an evidenced gap,
// docs/design-dflash.md), so serve recognises such a drafter and degrades to
// plain autoregressive with an honest notice rather than misloading it on the
// autoregressive MTP lane.
func (d DraftDetection) IsDFlash() bool {
	return d.Method == mtp.MTPDFlash
}

// DetectGemma4DraftPath resolves the drafter for modelPath through the reactive
// ladder. explicit is the --draft flag value after the CLI resolved its
// sentinel: pass the path to force a drafter, "" to mean "no explicit choice".
// Detection only ever fires for Gemma 4 family targets — other architectures
// always return DraftSourceNone unless explicitly flagged.
//
//	det := serving.DetectGemma4DraftPath(modelPath, "", serving.DraftDetectOptions{})
//	if det.Active() { /* load the target + det.DraftPath speculative pair */ }
func DetectGemma4DraftPath(modelPath, explicit string, opts DraftDetectOptions) DraftDetection {
	if explicit = core.Trim(explicit); explicit != "" {
		return DraftDetection{Source: DraftSourceFlag, DraftPath: explicit, Note: "explicit --draft"}
	}
	if opts.Disabled {
		return DraftDetection{Note: "drafter detection disabled"}
	}
	modelPath = core.Trim(modelPath)
	if modelPath == "" || !isGemma4FamilyConfig(modelPath) {
		return DraftDetection{}
	}

	// Rung 2a — assistant/ inside the model directory.
	if assistant := core.PathJoin(modelPath, "assistant"); isSafetensorsModelDir(assistant) {
		return DraftDetection{Source: DraftSourceAssistantDir, DraftPath: assistant, Note: "auto-detected assistant/ beside the weights"}
	}
	// Rung 2b — the MTPLX/Google bundle layout: --model points at
	// <bundle>/target and the drafter sits at <bundle>/assistant.
	if core.PathBase(modelPath) == "target" {
		if sibling := core.PathJoin(core.PathDir(modelPath), "assistant"); isSafetensorsModelDir(sibling) {
			return DraftDetection{Source: DraftSourceSiblingAssistant, DraftPath: sibling, Note: "auto-detected the target/ + assistant/ pair bundle"}
		}
	}
	// Rung 2c — the HF hub cache: community snapshots carry no assistant/
	// beside the weights — the family drafter is a SEPARATE cached repo
	// (models--*--gemma-4-<size>-it-assistant*). Path-shape + config reads
	// only; bf16 preferred (quant drafters dequantise to bf16 at load).
	if cached := hubCacheAssistantFor(modelPath); cached != "" {
		return DraftDetection{Source: DraftSourceCacheAssistant, DraftPath: cached, Note: "auto-detected the family assistant in the model cache"}
	}
	// Rung 3a — MTP/ subdirectory carrying a single GGUF (unsloth layout).
	if mtpDir := core.PathJoin(modelPath, "MTP"); core.Stat(mtpDir).OK {
		if ggufs := core.PathGlob(core.JoinPath(mtpDir, "*.gguf")); len(ggufs) == 1 {
			return DraftDetection{Source: DraftSourceMTPDir, DraftPath: ggufs[0], Note: "auto-detected MTP/ drafter (unsloth GGUF convention)"}
		}
	}
	// Rung 3b — sibling mtp-*.gguf beside the weights.
	if ggufs := core.PathGlob(core.JoinPath(modelPath, "mtp-*.gguf")); len(ggufs) == 1 {
		return DraftDetection{Source: DraftSourceMTPSibling, DraftPath: ggufs[0], Note: "auto-detected sibling mtp-*.gguf drafter"}
	}
	return DraftDetection{}
}

// ResolveServeDraft turns the serve/generate flags into a drafter decision:
// "auto" (the default) runs the reactive detection ladder, an explicit path
// forces the drafter, and "" (or -draft-detect=false) stands the ladder down.
// Shared by the serve and generate commands (both run the same ladder).
func ResolveServeDraft(modelPath, draftFlag string, detect bool) DraftDetection {
	explicit := ""
	opts := DraftDetectOptions{Disabled: !detect}
	switch trimmed := core.Trim(draftFlag); trimmed {
	case "auto":
		// ladder decides
	case "":
		opts.Disabled = true
	default:
		explicit = trimmed
	}
	return stampDFlashMethod(DetectGemma4DraftPath(modelPath, explicit, opts))
}

// stampDFlashMethod tags a resolved detection with mtp.MTPDFlash when its
// drafter path is a DFlash block-diffusion checkpoint, so a serve/generate boot
// tells the truth (the engine cannot yet run it) instead of announcing an MTP
// lane over a drafter that would misload. A non-DFlash (or drafterless)
// detection passes through unchanged.
func stampDFlashMethod(det DraftDetection) DraftDetection {
	if det.DraftPath == "" {
		return det
	}
	if _, ok := DetectDFlashDraft(det.DraftPath); ok {
		det.Method = mtp.MTPDFlash
	}
	return det
}

// DetectDFlashDraft recognises a DFlash speculator checkpoint at draftPath by its
// config.json marker (speculators_model_type "dflash") and returns the parsed
// drafter contract (block size, fused verifier layers, target). It reads the
// config only — never weights — the same posture as the reactive assistant
// detection. A path that is not a directory carrying a DFlash config.json returns
// ok=false.
//
//	if cfg, ok := serving.DetectDFlashDraft(draftPath); ok { /* a DFlash drafter */ }
func DetectDFlashDraft(draftPath string) (dflash.Config, bool) {
	draftPath = core.Trim(draftPath)
	if draftPath == "" {
		return dflash.Config{}, false
	}
	data := core.ReadFile(core.PathJoin(draftPath, "config.json"))
	if !data.OK {
		return dflash.Config{}, false
	}
	return dflash.ParseConfig(data.Bytes())
}

// DFlashDraftNotice is the honest boot notice for a detected-but-unrunnable
// DFlash drafter: it names the block and fused-verifier-layer count from the
// checkpoint and states plainly that the engine serves plain autoregressive,
// pointing at the design memo — the same posture as the hip diffusion sampler
// route (declared, kernel not linked), not a faked lane. Shared by the serve and
// generate boot paths. Returns "" when det is not a DFlash drafter.
func DFlashDraftNotice(det DraftDetection) string {
	cfg, ok := DetectDFlashDraft(det.DraftPath)
	if !ok {
		return ""
	}
	verifier := ""
	if cfg.Verifier != "" {
		verifier = ", verifier " + cfg.Verifier
	}
	return core.Sprintf("DFlash block-diffusion drafter detected — %s (block %d, %d fused verifier layers%s); this engine has no block-diffusion draft forward yet, so serving plain autoregressive. See docs/design-dflash.md",
		det.DraftPath, cfg.BlockSize, len(cfg.AuxHiddenLayerIDs), verifier)
}

// DFlashEngineProbe reports whether the compiled-in engine can RUN a DFlash
// block-diffusion drafter end to end over a serving session — the block-parallel
// proposal AND the fused verifier-hidden extraction the proposal conditions on. serve
// consults it to decide whether a detected DFlash drafter ARMS the block-diffusion
// lane or degrades to plain autoregressive with an honest notice. It defaults to
// "unsupported" so a build without the forward — or one whose live-session aux tap is
// not yet non-corrupting (docs/design-dflash.md) — declines truthfully rather than
// arming a lane it cannot run losslessly. The engine flips it, in one place, once it
// drives the lane correctly over a live session; the block forward itself is already
// built and proven (engine/metal/assistant_dflash*.go). This is the kill switch: with
// the probe down, an explicit --draft to a DFlash pack still declines honestly.
var DFlashEngineProbe = func() bool { return false }

// ArmDFlash reports whether a detected DFlash drafter should engage the live
// block-diffusion lane: it is a DFlash checkpoint AND the engine declares it can run
// one (DFlashEngineProbe). For a DFlash drafter the engine cannot yet run, this is
// false and serve degrades to plain autoregressive with DFlashDraftNotice — the honest
// decline, never a misload onto the autoregressive MTP lane.
func ArmDFlash(det DraftDetection) bool {
	return det.IsDFlash() && DFlashEngineProbe()
}

// DFlashActiveNotice is the boot notice for an ARMED DFlash drafter: the engine's
// block-diffusion draft forward proposes a whole block per readout and the target
// verifies it with the ordinary greedy prefix-accept, so the emitted sequence stays
// byte-identical to plain decode (lossless) — the drafter changes only speed, never
// which tokens. Returns "" when det is not a DFlash drafter. Shared by the serve and
// generate boot paths, the armed twin of DFlashDraftNotice.
func DFlashActiveNotice(det DraftDetection) string {
	cfg, ok := DetectDFlashDraft(det.DraftPath)
	if !ok {
		return ""
	}
	verifier := ""
	if cfg.Verifier != "" {
		verifier = ", verifier " + cfg.Verifier
	}
	return core.Sprintf("DFlash block-diffusion lane ACTIVE — drafter %s (block %d, %d fused verifier layers%s); each proposed block is greedy-prefix verified against the target, so output stays byte-identical to plain decode (lossless). See docs/design-dflash.md",
		det.DraftPath, cfg.BlockSize, len(cfg.AuxHiddenLayerIDs), verifier)
}

// speculativeServeNotice reports the ACTIVE MTP pair at boot: which drafter
// engaged, which ladder rung chose it, and the draft block the verify forwards
// will run. Drafterless serves return "". The pair loads lazily with the model —
// a drafter that fails to load surfaces on the first request, not here.
func speculativeServeNotice(detection DraftDetection, draftBlock int) string {
	if !detection.Active() {
		return ""
	}
	blockLabel := "engine-default block"
	if draftBlock > 0 {
		blockLabel = core.Sprintf("block %d", draftBlock)
	}
	return core.Sprintf("MTP speculative decode ACTIVE — drafter %s (%s), %s; greedy and sampled requests ride the verified lane, repetition-penalty/probe requests fall back to plain decode",
		detection.DraftPath, detection.Note, blockLabel)
}

// hubCacheAssistantFor resolves the family MTP assistant for a target served
// from an HF hub cache snapshot (…/hub/models--ORG--NAME/snapshots/HASH). The
// match key is the size segment between "gemma-4-" and "-it" of the target's
// repo dirname ("e2b", "26b-a4b", …), case-insensitive; bf16 assistant repos
// win over quant ones; the repo resolves through refs/main, falling back to
// the lexically-last snapshot that carries a gemma4 config + weights.
// Non-cache paths and assistant targets return "".
func hubCacheAssistantFor(modelPath string) string {
	snapshots := core.PathDir(core.CleanPath(modelPath, "/"))
	if core.PathBase(snapshots) != "snapshots" {
		return ""
	}
	repo := core.PathDir(snapshots)
	repoName := core.Lower(core.PathBase(repo))
	if !core.HasPrefix(repoName, "models--") || core.Contains(repoName, "assistant") {
		return ""
	}
	size := gemma4SizeSegment(repoName)
	if size == "" {
		return ""
	}
	want := "gemma-4-" + size + "-it-assistant"
	var quant []string
	for _, dir := range core.PathGlob(core.JoinPath(core.PathDir(repo), "models--*")) {
		base := core.Lower(core.PathBase(dir))
		if dir == repo || !core.Contains(base, want) {
			continue
		}
		if core.Contains(base, "bf16") {
			if snap := hubSnapshotDir(dir); snap != "" {
				return snap
			}
			continue
		}
		quant = append(quant, dir)
	}
	for _, dir := range quant {
		if snap := hubSnapshotDir(dir); snap != "" {
			return snap
		}
	}
	return ""
}

// gemma4SizeSegment cuts the size segment of a lowered hub repo dirname
// between "gemma-4-" and "-it" ("…gemma-4-26b-a4b-it-4bit" → "26b-a4b");
// "" when the name has no such shape.
func gemma4SizeSegment(lowerName string) string {
	const fam = "gemma-4-"
	i := core.Index(lowerName, fam)
	if i < 0 {
		return ""
	}
	rest := lowerName[i+len(fam):]
	j := core.Index(rest, "-it")
	if j <= 0 {
		return ""
	}
	return rest[:j]
}

// hubSnapshotDir resolves a hub repo dir to a loadable gemma4 snapshot:
// refs/main's hash when it points at one, else the lexically-last snapshot
// carrying a gemma4 config + weights (PathGlob returns sorted matches).
func hubSnapshotDir(repo string) string {
	snaps := core.PathJoin(repo, "snapshots")
	if ref := core.ReadFile(core.PathJoin(repo, "refs", "main")); ref.OK {
		if hash := core.Trim(string(ref.Bytes())); hash != "" {
			if dir := core.PathJoin(snaps, hash); isSafetensorsModelDir(dir) && isGemma4FamilyConfig(dir) {
				return dir
			}
		}
	}
	dirs := core.PathGlob(core.JoinPath(snaps, "*"))
	for i := len(dirs) - 1; i >= 0; i-- {
		if isSafetensorsModelDir(dirs[i]) && isGemma4FamilyConfig(dirs[i]) {
			return dirs[i]
		}
	}
	return ""
}

// isGemma4FamilyConfig reports whether modelPath/config.json declares a Gemma 4
// family model_type (gemma4, gemma4_text, gemma4_unified_text, …). Reads ONLY
// the config — never weights.
func isGemma4FamilyConfig(modelPath string) bool {
	data := core.ReadFile(core.PathJoin(modelPath, "config.json"))
	if !data.OK {
		return false
	}
	raw := data.Bytes()
	var probe struct {
		ModelType string `json:"model_type"`
	}
	if result := core.JSONUnmarshal(raw, &probe); !result.OK {
		return false
	}
	return core.HasPrefix(probe.ModelType, "gemma4")
}

// isSafetensorsModelDir reports whether dir looks like a loadable safetensors
// model pack: a config.json plus at least one *.safetensors file.
func isSafetensorsModelDir(dir string) bool {
	if !core.Stat(core.PathJoin(dir, "config.json")).OK {
		return false
	}
	return len(core.PathGlob(core.JoinPath(dir, "*.safetensors"))) > 0
}
