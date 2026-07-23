// SPDX-License-Identifier: EUPL-1.2

package tui

import (
	"strings"
	"testing"

	"github.com/charmbracelet/x/ansi"
)

func TestServiceState_ViewMatchesCanonicalCopy(t *testing.T) {
	service := newService()
	styles := newUIStyles(midnightTheme())

	stopped := ansi.Strip(service.view("", 180, styles))
	for _, want := range []string{
		"SERVICE  OpenAI · Anthropic · Ollama HTTP API for the loaded model",
		"○ stopped",
		"address  ‹ :36911 ›",
		"Lethean's own port — the default the client hints below use.",
		"point a client here  the request's model name is cosmetic — the loaded model answers",
		"opencode / codex / OpenAI SDKs",
		"http://localhost:36911/v1",
		"Claude Code / Anthropic SDKs",
		"Ollama clients",
		"smoke",
		"curl -s http://localhost:36911/v1/chat/completions",
		"TUI chat and API requests share the model through one serial lane — turns queue behind each other, nothing races the engine.",
		"enter start/stop · ‹/› address (while stopped)",
	} {
		if !strings.Contains(stopped, want) {
			t.Fatalf("stopped Service panel missing %q:\n%s", want, stopped)
		}
	}
	service.note = "stopped"
	if stopped = ansi.Strip(service.view("", 180, styles)); strings.Contains(stopped, "\n  stopped\n") {
		t.Fatalf("routine stop note duplicated the canonical stopped toggle:\n%s", stopped)
	}

	service.running = true
	running := ansi.Strip(service.view("fixture", 180, styles))
	if !strings.Contains(running, "● listening") || strings.Contains(running, "● serving") || strings.Contains(running, "\nrequests") {
		t.Fatalf("running Service state drifted from the canonical toggle:\n%s", running)
	}
}
