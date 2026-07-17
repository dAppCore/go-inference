// SPDX-License-Identifier: EUPL-1.2

package tui

import (
	"strings"
	"testing"

	coreio "dappco.re/go/io"
)

func TestPreferences_Good(t *testing.T) {
	medium := coreio.NewMockMedium()
	opened := openPreferences(medium, appConfigPath)
	if !opened.OK {
		t.Fatalf("openPreferences failed: %v", opened.Value)
	}
	preferences, ok := opened.Value.(preferenceStore)
	if !ok {
		t.Fatalf("openPreferences value = %T, want preferenceStore", opened.Value)
	}
	defaults := preferences.Values()
	if defaults.ContextLength != 0 || defaults.MaxTokens != 4096 || defaults.Thinking != "model" {
		t.Fatalf("generation defaults = %#v", defaults)
	}
	if defaults.Theme != "midnight" || !defaults.ShowThinking || defaults.RecentSessionLimit != 12 {
		t.Fatalf("workspace defaults = %#v", defaults)
	}
	if len(defaults.KnowledgePaths) != 1 || defaults.KnowledgePaths[0] != appPacksPath || defaults.KnowledgeMaxBytes != 65536 {
		t.Fatalf("knowledge defaults = %#v", defaults)
	}
	if defaults.PreferredRuntime != "auto" || !defaults.ConfirmExecution {
		t.Fatalf("runtime defaults = %#v", defaults)
	}

	if result := preferences.Set("appearance.theme", "aurora"); !result.OK {
		t.Fatalf("set theme: %v", result.Value)
	}
	if result := preferences.Set("generation.max_tokens", 8192); !result.OK {
		t.Fatalf("set max tokens: %v", result.Value)
	}
	if medium.Exists(appConfigPath) {
		t.Fatalf("%q exists before Commit", appConfigPath)
	}
	if result := preferences.Commit(); !result.OK {
		t.Fatalf("commit preferences: %v", result.Value)
	}
	if !medium.Exists(appConfigPath) {
		t.Fatalf("%q does not exist after Commit", appConfigPath)
	}

	reopened := openPreferences(medium, appConfigPath)
	if !reopened.OK {
		t.Fatalf("reopen preferences: %v", reopened.Value)
	}
	preferences, ok = reopened.Value.(preferenceStore)
	if !ok {
		t.Fatalf("reopened value = %T, want preferenceStore", reopened.Value)
	}
	values := preferences.Values()
	if values.Theme != "aurora" || values.MaxTokens != 8192 {
		t.Fatalf("reopened values = %#v, want aurora and 8192", values)
	}
	if preferences.Warning() != nil {
		t.Fatalf("healthy preferences warning = %v, want nil", preferences.Warning())
	}
}

func TestPreferences_Bad(t *testing.T) {
	medium := coreio.NewMockMedium()
	const malformed = "generation: [unterminated\nappearance:\n  theme: should-not-load\n"
	if err := medium.Write(appConfigPath, malformed); err != nil {
		t.Fatalf("write malformed config: %v", err)
	}

	opened := openPreferences(medium, appConfigPath)
	if !opened.OK {
		t.Fatalf("openPreferences malformed config should degrade: %v", opened.Value)
	}
	preferences, ok := opened.Value.(preferenceStore)
	if !ok {
		t.Fatalf("openPreferences value = %T, want preferenceStore", opened.Value)
	}
	values := preferences.Values()
	if values.Theme != "midnight" || values.MaxTokens != 4096 {
		t.Fatalf("malformed fallback values = %#v, want defaults", values)
	}
	if preferences.Warning() == nil {
		t.Fatal("malformed preferences warning = nil, want parse warning")
	}
	if result := preferences.Set("appearance.theme", "unsafe-overwrite"); !result.OK {
		t.Fatalf("stage fallback preference: %v", result.Value)
	}
	if result := preferences.Commit(); result.OK {
		t.Fatalf("Commit malformed fallback = %#v, want failure", result.Value)
	}
	content, err := medium.Read(appConfigPath)
	if err != nil {
		t.Fatalf("read malformed config after blocked commit: %v", err)
	}
	if content != malformed {
		t.Fatalf("malformed config changed to %q, want byte preservation", content)
	}

	if err := medium.Write(appConfigPath, "appearance:\n  theme: repaired\n"); err != nil {
		t.Fatalf("repair config fixture: %v", err)
	}
	if result := preferences.Reload(); !result.OK {
		t.Fatalf("Reload repaired config: %v", result.Value)
	}
	if preferences.Warning() != nil || preferences.Values().Theme != "repaired" {
		t.Fatalf("reloaded preferences = %#v, warning %v", preferences.Values(), preferences.Warning())
	}
}

func TestPreferences_Ugly(t *testing.T) {
	t.Setenv("LEM_GENERATION_MAX_TOKENS", "12288")
	medium := coreio.NewMockMedium()
	opened := openPreferences(medium, appConfigPath)
	if !opened.OK {
		t.Fatalf("openPreferences with environment: %v", opened.Value)
	}
	preferences, ok := opened.Value.(preferenceStore)
	if !ok {
		t.Fatalf("openPreferences value = %T, want preferenceStore", opened.Value)
	}
	if got := preferences.Values().MaxTokens; got != 12288 {
		t.Fatalf("environment max tokens = %d, want 12288", got)
	}
	if result := preferences.Set("appearance.theme", "daylight"); !result.OK {
		t.Fatalf("set unrelated theme: %v", result.Value)
	}
	if result := preferences.Commit(); !result.OK {
		t.Fatalf("commit unrelated theme: %v", result.Value)
	}
	content, err := medium.Read(appConfigPath)
	if err != nil {
		t.Fatalf("read committed config: %v", err)
	}
	if !strings.Contains(content, "daylight") {
		t.Fatalf("committed config = %q, want explicit theme", content)
	}
	if strings.Contains(content, "max_tokens") || strings.Contains(content, "12288") {
		t.Fatalf("environment-derived max tokens leaked into config: %q", content)
	}
}
