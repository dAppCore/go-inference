// SPDX-License-Identifier: EUPL-1.2

package tui

import (
	"bytes"
	"context"
	"os"
	"strings"
	"testing"
	"time"

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

func TestRunWithWorkspace_CheckIsReadOnlyAndClosesResources(t *testing.T) {
	resources := openAppTestWorkspace(t)
	databasePath := resources.Paths.Database
	gitRoot := t.TempDir()
	gitDir := core.PathJoin(gitRoot, ".git")
	if err := os.MkdirAll(gitDir, 0o755); err != nil {
		t.Fatalf("create Git sentinel: %v", err)
	}
	headPath := core.PathJoin(gitDir, "HEAD")
	const head = "ref: refs/heads/check-sentinel\n"
	if err := os.WriteFile(headPath, []byte(head), 0o644); err != nil {
		t.Fatalf("write Git sentinel: %v", err)
	}
	sentinel := testWorkRecord("check-work", "Check sentinel", "queued", time.Date(2026, time.July, 18, 14, 0, 0, 0, time.UTC))
	sentinel.Task, sentinel.Repo = "must not be admitted", gitRoot
	if result := resources.Repository.SaveWorkItem(sentinel); !result.OK {
		t.Fatalf("SaveWorkItem: %v", result.Value)
	}
	spy := &correctiveAgentProvider{}
	resources.Agent = spy
	normalCalls, checkCalls := 0, 0
	var stdout bytes.Buffer
	var stderr bytes.Buffer

	code := runWithWorkspace(context.Background(), []string{"--check"}, &stdout, &stderr, workspaceLoaders{
		Normal: func() core.Result {
			normalCalls++
			return core.Fail(core.E("test.normal", "normal/native composition boundary ran", nil))
		},
		Check: func() core.Result {
			checkCalls++
			return core.Ok(resources)
		},
	})

	snapshots, reviews, runs, closes := spy.CallCounts()
	if code != 0 || normalCalls != 0 || checkCalls != 1 || snapshots != 0 || reviews != 0 || runs != 0 || closes != 1 {
		t.Fatalf("check boundary = code %d normal %d check %d snapshot/review/run/close %d/%d/%d/%d stderr %q", code, normalCalls, checkCalls, snapshots, reviews, runs, closes, stderr.String())
	}
	if resources.Repository != nil || resources.State != nil || resources.Agent != nil {
		t.Fatalf("check resources remained open: repository=%v state=%v agent=%v", resources.Repository != nil, resources.State != nil, resources.Agent != nil)
	}
	gotHead, err := os.ReadFile(headPath)
	if err != nil || string(gotHead) != head {
		t.Fatalf("Git sentinel changed: head %q error %v", gotHead, err)
	}
	reopenedResult := openDuckRepository(databasePath)
	if !reopenedResult.OK {
		t.Fatalf("reopen repository: %s", reopenedResult.Error())
	}
	reopened := reopenedResult.Value.(workspaceRepository)
	defer reopened.Close()
	itemsResult := reopened.ListWorkItems(false)
	if !itemsResult.OK {
		t.Fatalf("ListWorkItems: %s", itemsResult.Error())
	}
	items := itemsResult.Value.([]workItemRecord)
	if len(items) != 1 || core.JSONMarshalString(items[0]) != core.JSONMarshalString(sentinel) {
		t.Fatalf("check mutated repository or admitted queue work: %#v", items)
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
