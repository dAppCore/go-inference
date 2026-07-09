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
//     assistant/ dropped inside the model directory);
//  3. an MTP/ subdirectory or sibling mtp-*.gguf — the unsloth GGUF convention.
//
// Detection is path-shape only — no weights are opened. Whether the found
// drafter actually LOADS stays the resolver's business; serve reports the
// failure honestly on first load rather than refusing to boot.

package serving

import (
	core "dappco.re/go"
)

// MTPDefaultDraftBlock is the engine-default MTP draft block (verify forward =
// carried lead + block-1 proposals) applied when neither the --draft-block flag
// nor a tuned profile pins one.
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
	DraftSourceMTPDir           DraftDetectionSource = "mtp-dir"          // <model>/MTP/*.gguf
	DraftSourceMTPSibling       DraftDetectionSource = "mtp-sibling-gguf" // <model>/mtp-*.gguf
)

// DraftDetection is the resolved drafter decision for a model path.
type DraftDetection struct {
	Source    DraftDetectionSource
	DraftPath string
	// Note carries the operator-facing wording for the serve boot notice
	// (why this drafter engaged, or why detection stood down).
	Note string
}

// Active reports whether a drafter should be engaged.
func (d DraftDetection) Active() bool {
	return d.Source != DraftSourceNone && d.DraftPath != ""
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
	return DetectGemma4DraftPath(modelPath, explicit, opts)
}

// speculativeServeNotice reports the ACTIVE MTP pair at boot: which drafter
// engaged, which ladder rung chose it, and the draft block the verify forwards
// will run. Drafterless serves return "". The pair loads lazily with the model —
// a drafter that fails to load surfaces on the first request, not here.
func speculativeServeNotice(detection DraftDetection, draftBlock int) string {
	if !detection.Active() {
		return ""
	}
	if draftBlock <= 0 {
		draftBlock = MTPDefaultDraftBlock
	}
	return core.Sprintf("MTP speculative decode ACTIVE — drafter %s (%s), block %d; greedy and sampled requests ride the verified lane, repetition-penalty/probe requests fall back to plain decode",
		detection.DraftPath, detection.Note, draftBlock)
}

// isGemma4FamilyConfig reports whether modelPath/config.json declares a Gemma 4
// family model_type (gemma4, gemma4_text, gemma4_unified_text, …). Reads ONLY
// the config — never weights.
func isGemma4FamilyConfig(modelPath string) bool {
	data := core.ReadFile(core.PathJoin(modelPath, "config.json"))
	if !data.OK {
		return false
	}
	raw, ok := data.Value.([]byte)
	if !ok {
		return false
	}
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
