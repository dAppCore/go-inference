// SPDX-Licence-Identifier: EUPL-1.2

package hip

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
)

// DraftDetectOptions configures reactive Gemma4 drafter detection. The zero
// value means "detect"; Disabled stands the auto ladder down while still
// allowing callers to pass an explicit drafter path.
type DraftDetectOptions struct {
	Disabled bool
}

// DraftDetectionSource names the path-shape rung that resolved a drafter.
type DraftDetectionSource string

const (
	DraftSourceNone             DraftDetectionSource = ""
	DraftSourceFlag             DraftDetectionSource = "flag"
	DraftSourceAssistantDir     DraftDetectionSource = "assistant-dir"
	DraftSourceSiblingAssistant DraftDetectionSource = "assistant-pair"
	DraftSourceMTPDir           DraftDetectionSource = "mtp-dir"
	DraftSourceMTPSibling       DraftDetectionSource = "mtp-sibling-gguf"
)

// DraftDetection is the resolved drafter decision for a model path.
type DraftDetection struct {
	Source    DraftDetectionSource `json:"source,omitempty"`
	DraftPath string               `json:"draft_path,omitempty"`
	Note      string               `json:"note,omitempty"`
}

// Active reports whether the detected drafter should be engaged.
func (d DraftDetection) Active() bool {
	return d.Source != DraftSourceNone && strings.TrimSpace(d.DraftPath) != ""
}

// DetectGemma4DraftPath mirrors the go-mlx reactive Gemma4 drafter ladder
// without importing go-mlx or opening weights:
//  1. explicit drafter path wins;
//  2. <model>/assistant safetensors pack;
//  3. <bundle>/target + <bundle>/assistant safetensors pack;
//  4. <model>/MTP/*.gguf;
//  5. <model>/mtp-*.gguf.
func DetectGemma4DraftPath(modelPath, explicit string, opts DraftDetectOptions) DraftDetection {
	if explicit = strings.TrimSpace(explicit); explicit != "" {
		return DraftDetection{Source: DraftSourceFlag, DraftPath: explicit, Note: "explicit --draft"}
	}
	if opts.Disabled {
		return DraftDetection{Note: "drafter detection disabled"}
	}
	modelPath = strings.TrimSpace(modelPath)
	if modelPath == "" || !isROCmGemma4FamilyConfig(modelPath) {
		return DraftDetection{}
	}
	if assistant := filepath.Join(modelPath, "assistant"); isROCmSafetensorsModelDir(assistant) {
		return DraftDetection{Source: DraftSourceAssistantDir, DraftPath: assistant, Note: "auto-detected assistant/ beside the weights"}
	}
	if filepath.Base(modelPath) == "target" {
		if sibling := filepath.Join(filepath.Dir(modelPath), "assistant"); isROCmSafetensorsModelDir(sibling) {
			return DraftDetection{Source: DraftSourceSiblingAssistant, DraftPath: sibling, Note: "auto-detected the target/ + assistant/ pair bundle"}
		}
	}
	if mtpDir := filepath.Join(modelPath, "MTP"); pathExists(mtpDir) {
		if ggufs, _ := filepath.Glob(filepath.Join(mtpDir, "*.gguf")); len(ggufs) == 1 {
			return DraftDetection{Source: DraftSourceMTPDir, DraftPath: ggufs[0], Note: "auto-detected MTP/ drafter (unsloth GGUF convention)"}
		}
	}
	if ggufs, _ := filepath.Glob(filepath.Join(modelPath, "mtp-*.gguf")); len(ggufs) == 1 {
		return DraftDetection{Source: DraftSourceMTPSibling, DraftPath: ggufs[0], Note: "auto-detected sibling mtp-*.gguf drafter"}
	}
	return DraftDetection{}
}

func isROCmGemma4FamilyConfig(modelPath string) bool {
	data, err := os.ReadFile(filepath.Join(modelPath, "config.json"))
	if err != nil {
		return false
	}
	var probe struct {
		ModelType     string   `json:"model_type"`
		Architectures []string `json:"architectures"`
		TextConfig    struct {
			ModelType     string   `json:"model_type"`
			Architectures []string `json:"architectures"`
		} `json:"text_config"`
	}
	if err := json.Unmarshal(data, &probe); err != nil {
		return false
	}
	return isROCmGemma4DraftDetectArchitecture(probe.ModelType) ||
		anyROCmGemma4DraftDetectArchitecture(probe.Architectures) ||
		isROCmGemma4DraftDetectArchitecture(probe.TextConfig.ModelType) ||
		anyROCmGemma4DraftDetectArchitecture(probe.TextConfig.Architectures)
}

func anyROCmGemma4DraftDetectArchitecture(values []string) bool {
	for _, value := range values {
		if isROCmGemma4DraftDetectArchitecture(value) {
			return true
		}
	}
	return false
}

func isROCmGemma4DraftDetectArchitecture(value string) bool {
	value = strings.ToLower(strings.TrimSpace(value))
	value = strings.ReplaceAll(value, "-", "_")
	value = strings.ReplaceAll(value, ".", "_")
	value = strings.ReplaceAll(value, " ", "_")
	return strings.Contains(value, "gemma4")
}

func isROCmSafetensorsModelDir(dir string) bool {
	if !pathExists(filepath.Join(dir, "config.json")) {
		return false
	}
	matches, _ := filepath.Glob(filepath.Join(dir, "*.safetensors"))
	return len(matches) > 0
}

func pathExists(path string) bool {
	_, err := os.Stat(path)
	return err == nil
}
