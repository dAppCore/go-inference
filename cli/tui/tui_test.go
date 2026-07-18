// SPDX-License-Identifier: EUPL-1.2

package tui

import (
	"bytes"
	"context"
	"strings"
	"testing"

	tea "github.com/charmbracelet/bubbletea"

	core "dappco.re/go"
)

func TestRunWithWorkspace_Good(t *testing.T) {
	resources := openAppTestWorkspace(t)
	var stdout bytes.Buffer
	var stderr bytes.Buffer

	code := runWithWorkspace(
		context.Background(),
		[]string{"--check"},
		&stdout,
		&stderr,
		workspaceLoaders{Check: func() core.Result { return core.Ok(resources) }},
	)

	if code != 0 {
		t.Fatalf("runWithWorkspace code = %d, stderr = %q", code, stderr.String())
	}
	if !strings.Contains(stdout.String(), "LEM") || !strings.Contains(stdout.String(), "Chat") {
		t.Fatalf("check frame did not render the chat workspace:\n%s", stdout.String())
	}
	if resources.Repository != nil || resources.State != nil {
		t.Fatal("check mode did not close workspace resources")
	}
}

func TestRunWithWorkspace_Bad(t *testing.T) {
	var stdout bytes.Buffer
	var stderr bytes.Buffer

	code := runWithWorkspace(
		context.Background(),
		[]string{"--check"},
		&stdout,
		&stderr,
		workspaceLoaders{Check: func() core.Result {
			return core.Fail(core.E("test.run", "storage offline", nil))
		}},
	)

	if code != 1 {
		t.Fatalf("runWithWorkspace code = %d, want 1", code)
	}
	if !strings.Contains(stdout.String(), "Workspace could not open") || !strings.Contains(stdout.String(), "storage offline") {
		t.Fatalf("blocking check frame missing failure:\n%s", stdout.String())
	}
}

func TestRunWithWorkspace_Ugly(t *testing.T) {
	var stdout bytes.Buffer
	var stderr bytes.Buffer

	code := runWithWorkspace(context.Background(), []string{"--check"}, &stdout, &stderr, workspaceLoaders{})

	if code != 1 || !strings.Contains(stdout.String(), "workspace loader is unavailable") {
		t.Fatalf("nil loader = code %d frame %q", code, stdout.String())
	}
}

func TestAgentBootstrap_CheckModeDoesNotConstructNormalAgent(t *testing.T) {
	resources := openAppTestWorkspace(t)
	normalCalls := 0
	checkCalls := 0
	var stdout bytes.Buffer
	var stderr bytes.Buffer
	code := runWithWorkspace(context.Background(), []string{"--check"}, &stdout, &stderr, workspaceLoaders{
		Normal: func() core.Result {
			normalCalls++
			return core.Fail(core.E("test.normal", "normal composition must not run", nil))
		},
		Check: func() core.Result {
			checkCalls++
			resources.Agent = newUnavailableAgentProvider("check mode is read-only")
			return core.Ok(resources)
		},
	})
	if code != 0 || normalCalls != 0 || checkCalls != 1 {
		t.Fatalf("check composition = code %d normal %d check %d stderr %q", code, normalCalls, checkCalls, stderr.String())
	}
}

func TestShutdownProgramModel_Good(t *testing.T) {
	initial := newApp("", 0, 64)
	final := initial
	resources := openAppTestWorkspace(t)
	if result := final.connectWorkspace(resources); !result.OK {
		t.Fatalf("connect final workspace: %v", result.Value)
	}

	result := shutdownProgramModel(initial, tea.Model(final))

	if !result.OK {
		t.Fatalf("shutdown final model: %v", result.Value)
	}
	if resources.Repository != nil || resources.State != nil {
		t.Fatal("shutdown used the stale initial model and leaked final resources")
	}
}
