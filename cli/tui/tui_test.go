// SPDX-License-Identifier: EUPL-1.2

package tui

import (
	"bytes"
	"context"
	"os"
	"strings"
	"testing"
	"time"

	core "dappco.re/go"
	tea "dappco.re/go/render/display/tui"
	"github.com/charmbracelet/x/ansi"
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
	if plain := ansi.Strip(stdout.String()); !strings.Contains(plain, "LEM") || !strings.Contains(plain, "Chat") {
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

func TestLoadDefaultWorkspaceContext_UnhardenedRuntimeFailsBeforeStateMutation(t *testing.T) {
	if contract := linkedAgentRuntimeContract(context.Background()); contract.OK {
		t.Skip("linked inference module exposes the hardened runtime contract")
	}
	home := t.TempDir()
	t.Setenv("HOME", home)
	result := loadDefaultWorkspaceContext(context.Background())
	core.AssertFalse(t, result.OK)
	core.AssertContains(t, result.Error(), "release/update")
	if _, err := os.Stat(core.PathJoin(home, ".lem")); !os.IsNotExist(err) {
		t.Fatalf("normal loader mutated ~/.lem before rejecting the linked runtime: %v", err)
	}
}

func TestLoadDefaultCheckWorkspaceContext_BypassesNativeRuntimeContract(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	result := loadDefaultCheckWorkspaceContext(context.Background())
	core.AssertTrue(t, result.OK, result.Error())
	resources, ok := result.Value.(*workspaceResources)
	core.AssertTrue(t, ok)
	core.AssertTrue(t, resources.Close().OK)
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

// TestNewDataReviewApp_Good proves RunDataReview's state setup (Task 8,
// `lem data review [slug]`) — activePanel focused on Data, the slug
// trimmed and stashed for attachData to consume once the workspace
// connects — without ever invoking the stored loader (newWorkspaceApp
// only stores it; nothing here triggers workspaceBootstrap), so this
// never touches $HOME despite newDataReviewApp's real production loader
// closing over loadDefaultWorkspaceContext.
func TestNewDataReviewApp_Good(t *testing.T) {
	a := newDataReviewApp(context.Background(), "  evening-vents  ")
	if a.activePanel != panelData {
		t.Fatalf("activePanel = %d, want panelData", a.activePanel)
	}
	if a.dataInitialSlug != "evening-vents" {
		t.Fatalf("dataInitialSlug = %q, want the trimmed slug", a.dataInitialSlug)
	}
	if a.boot.phase != bootLoading {
		t.Fatalf("boot phase = %d, want bootLoading", a.boot.phase)
	}
}

func TestNewDataReviewApp_EmptySlug(t *testing.T) {
	a := newDataReviewApp(context.Background(), "")
	if a.activePanel != panelData || a.dataInitialSlug != "" {
		t.Fatalf("activePanel=%d dataInitialSlug=%q", a.activePanel, a.dataInitialSlug)
	}
}

// TestDataReviewApp_ConnectRendersDataPanelWithInitialSlugFilter drives
// the real connectWorkspace path headlessly (runWorkspaceCheck, the same
// seam every other TUI bootstrap test in this file uses) to prove
// newDataReviewApp's state actually lands: once the workspace connects,
// the Data panel is what's on screen, and dataInitialSlug narrowed the
// list to that one dataset.
func TestDataReviewApp_ConnectRendersDataPanelWithInitialSlugFilter(t *testing.T) {
	resources := openAppTestWorkspace(t)
	if resources.DatasetStore == nil {
		t.Fatal("test workspace resources have no DatasetStore — cannot prove the initial slug filter")
	}
	at := time.Now()
	dsA := seedDataDataset(t, resources.DatasetStore, "alpha", at)
	dsB := seedDataDataset(t, resources.DatasetStore, "beta", at)
	seedDataItem(t, resources.DatasetStore, dsA.ID, "a", "a", at)
	seedDataItem(t, resources.DatasetStore, dsB.ID, "b", "b", at)

	a := newWorkspaceApp("", 0, 4096, func() core.Result { return core.Ok(resources) })
	a.activePanel = panelData
	a.dataInitialSlug = "alpha"

	var stdout, stderr bytes.Buffer
	code := runWorkspaceCheck(a, &stdout, &stderr)
	if code != 0 {
		t.Fatalf("runWorkspaceCheck code = %d, stderr=%s", code, stderr.String())
	}
	view := ansi.Strip(stdout.String())
	if !strings.Contains(view, "Data") {
		t.Fatalf("check frame did not render the Data panel:\n%s", view)
	}
	if !strings.Contains(view, "alpha") || strings.Contains(view, "beta") {
		t.Fatalf("check frame did not honour the initial dataset slug filter:\n%s", view)
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
