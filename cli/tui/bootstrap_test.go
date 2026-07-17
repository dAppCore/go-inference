// SPDX-License-Identifier: EUPL-1.2

package tui

import (
	"os"
	"strings"
	"testing"
	"time"

	core "dappco.re/go"
	coreio "dappco.re/go/io"
)

func TestOpenWorkspace_Good(t *testing.T) {
	files := testWorkspaceFiles(t)
	closeOrder := make([]string, 0, 2)
	openers := workspaceOpeners{
		Repository: func(path string) core.Result {
			result := openDuckRepository(path)
			if !result.OK {
				return result
			}
			repository, ok := result.Value.(workspaceRepository)
			if !ok {
				return core.Fail(core.E("test.repository", "unexpected repository", nil))
			}
			return core.Ok(&trackingWorkspaceRepository{workspaceRepository: repository, closeOrder: &closeOrder})
		},
		State: func(paths appPaths) core.Result {
			result := openReactiveState(paths)
			if !result.OK {
				return result
			}
			state, ok := result.Value.(reactiveState)
			if !ok {
				return core.Fail(core.E("test.state", "unexpected state", nil))
			}
			return core.Ok(&trackingReactiveState{reactiveState: state, closeOrder: &closeOrder})
		},
		Now: func() time.Time {
			return time.Date(2026, time.July, 17, 14, 0, 0, 0, time.UTC)
		},
	}

	opened := openWorkspaceWith(files, openers)
	if !opened.OK {
		t.Fatalf("openWorkspaceWith failed: %v", opened.Value)
	}
	resources, ok := opened.Value.(*workspaceResources)
	if !ok {
		t.Fatalf("openWorkspaceWith value = %T, want *workspaceResources", opened.Value)
	}
	for _, directory := range []string{files.Paths.Workspaces, files.Paths.Packs, files.Paths.Exports} {
		if !files.Medium.IsDir(directory) {
			t.Errorf("workspace medium directory %q was not created", directory)
		}
	}
	if _, err := os.Stat(files.Paths.Database); err != nil {
		t.Errorf("migrated DuckDB %q: %v", files.Paths.Database, err)
	}
	if len(resources.Warnings) != 0 {
		t.Errorf("healthy workspace warnings = %#v, want none", resources.Warnings)
	}

	createdAt := time.Date(2026, time.July, 17, 14, 1, 0, 0, time.UTC)
	session := testSessionRecord("bootstrap-session", "Bootstrapped", createdAt)
	if result := resources.Repository.SaveSession(session); !result.OK {
		t.Fatalf("save through bootstrapped repository: %v", result.Value)
	}
	if result := resources.Repository.Session(session.ID); !result.OK {
		t.Fatalf("read through bootstrapped repository: %v", result.Value)
	}
	if result := resources.State.Set(reactiveGroupWorkspace, "active_panel", "chat"); !result.OK {
		t.Fatalf("write through bootstrapped state: %v", result.Value)
	}
	if value, result := resources.State.Get(reactiveGroupWorkspace, "active_panel"); !result.OK || value != "chat" {
		t.Fatalf("read through bootstrapped state = %q, %#v", value, result.Value)
	}
	if values := resources.Preferences.Values(); values.Theme != "midnight" || values.MaxTokens != 4096 {
		t.Fatalf("bootstrapped preference defaults = %#v", values)
	}

	if result := resources.Close(); !result.OK {
		t.Fatalf("close workspace resources: %v", result.Value)
	}
	if len(closeOrder) != 2 || closeOrder[0] != "state" || closeOrder[1] != "repository" {
		t.Fatalf("close order = %#v, want [state repository]", closeOrder)
	}
}

func TestOpenWorkspace_Bad(t *testing.T) {
	files := testWorkspaceFiles(t)
	openers := workspaceOpeners{
		Repository: func(string) core.Result {
			return core.Fail(core.E("test.repository", "database unavailable", nil))
		},
	}

	result := openWorkspaceWith(files, openers)
	if result.OK {
		t.Fatalf("openWorkspaceWith repository failure = %#v, want failure", result.Value)
	}
	err, ok := result.Value.(error)
	if !ok || !strings.Contains(err.Error(), files.Paths.Database) {
		t.Fatalf("repository failure = %v, want exact DuckDB path %q", result.Value, files.Paths.Database)
	}
}

func TestOpenWorkspace_Ugly(t *testing.T) {
	t.Run("state degrades", func(t *testing.T) {
		files := testWorkspaceFiles(t)
		result := openWorkspaceWith(files, workspaceOpeners{
			State: func(appPaths) core.Result {
				return core.Fail(core.E("test.state", "state offline", nil))
			},
		})
		if !result.OK {
			t.Fatalf("state failure blocked startup: %v", result.Value)
		}
		resources := result.Value.(*workspaceResources)
		defer func() { _ = resources.Close() }()
		if _, read := resources.State.Get(reactiveGroupDrafts, "session-a"); read.OK {
			t.Fatal("disabled state Get succeeded")
		}
		if write := resources.State.Set(reactiveGroupDrafts, "session-a", "draft"); write.OK {
			t.Fatal("disabled state Set succeeded")
		}
		assertWorkspaceWarning(t, resources.Warnings, "state offline")
	})

	t.Run("preferences degrade", func(t *testing.T) {
		files := testWorkspaceFiles(t)
		result := openWorkspaceWith(files, workspaceOpeners{
			Preferences: func(coreio.Medium, string) core.Result {
				return core.Fail(core.E("test.preferences", "config offline", nil))
			},
		})
		if !result.OK {
			t.Fatalf("preference failure blocked startup: %v", result.Value)
		}
		resources := result.Value.(*workspaceResources)
		defer func() { _ = resources.Close() }()
		if values := resources.Preferences.Values(); values.Theme != "midnight" || values.MaxTokens != 4096 {
			t.Fatalf("degraded preference defaults = %#v", values)
		}
		if resources.Preferences.Warning() == nil {
			t.Fatal("degraded preferences warning = nil")
		}
		if commit := resources.Preferences.Commit(); commit.OK {
			t.Fatal("degraded preferences Commit succeeded")
		}
		assertWorkspaceWarning(t, resources.Warnings, "config offline")
	})
}

type trackingWorkspaceRepository struct {
	workspaceRepository
	closeOrder *[]string
}

func (repository *trackingWorkspaceRepository) Close() core.Result {
	*repository.closeOrder = append(*repository.closeOrder, "repository")
	return repository.workspaceRepository.Close()
}

type trackingReactiveState struct {
	reactiveState
	closeOrder *[]string
}

func (state *trackingReactiveState) Close() core.Result {
	*state.closeOrder = append(*state.closeOrder, "state")
	return state.reactiveState.Close()
}

func testWorkspaceFiles(t *testing.T) appFiles {
	t.Helper()
	root := t.TempDir()
	pathsResult := appPathsAt(root)
	if !pathsResult.OK {
		t.Fatalf("appPathsAt(%q): %v", root, pathsResult.Value)
	}
	paths := pathsResult.Value.(appPaths)
	if err := os.MkdirAll(root+"/"+paths.Workspaces, 0700); err != nil {
		t.Fatalf("create local workspace state directory: %v", err)
	}
	return appFiles{Paths: paths, Medium: coreio.NewMockMedium()}
}

func assertWorkspaceWarning(t *testing.T, warnings []string, want string) {
	t.Helper()
	for _, warning := range warnings {
		if strings.Contains(warning, want) {
			return
		}
	}
	t.Fatalf("workspace warnings = %#v, want text %q", warnings, want)
}
