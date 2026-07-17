// SPDX-License-Identifier: EUPL-1.2

package tui

import (
	"os"
	"testing"
)

func TestReactiveState_Good(t *testing.T) {
	paths := testReactivePaths(t)
	opened := openReactiveState(paths)
	if !opened.OK {
		t.Fatalf("openReactiveState failed: %v", opened.Value)
	}
	state, ok := opened.Value.(reactiveState)
	if !ok {
		t.Fatalf("openReactiveState value = %T, want reactiveState", opened.Value)
	}
	values := []struct {
		group string
		key   string
		value string
	}{
		{group: reactiveGroupWorkspace, key: "active_session", value: "session-a"},
		{group: reactiveGroupDrafts, key: "session-a", value: "unfinished prompt"},
		{group: reactiveGroupViewport, key: "session-a", value: "41"},
		{group: reactiveGroupCollapsed, key: "turn-tool-1", value: "true"},
	}
	for _, item := range values {
		if result := state.Set(item.group, item.key, item.value); !result.OK {
			t.Fatalf("Set(%q, %q): %v", item.group, item.key, result.Value)
		}
	}
	if result := state.Close(); !result.OK {
		t.Fatalf("close first state: %v", result.Value)
	}

	reopened := openReactiveState(paths)
	if !reopened.OK {
		t.Fatalf("reopen reactive state: %v", reopened.Value)
	}
	state, ok = reopened.Value.(reactiveState)
	if !ok {
		t.Fatalf("reopened value = %T, want reactiveState", reopened.Value)
	}
	defer func() {
		if result := state.Close(); !result.OK {
			t.Errorf("close reopened state: %v", result.Value)
		}
	}()
	for _, item := range values {
		got, result := state.Get(item.group, item.key)
		if !result.OK {
			t.Errorf("Get(%q, %q): %v", item.group, item.key, result.Value)
			continue
		}
		if got != item.value {
			t.Errorf("Get(%q, %q) = %q, want %q", item.group, item.key, got, item.value)
		}
	}
}

func TestReactiveState_Bad(t *testing.T) {
	paths := testReactivePaths(t)
	if err := os.Remove(paths.State); err != nil && !os.IsNotExist(err) {
		t.Fatalf("remove unused state path: %v", err)
	}
	if err := os.Mkdir(paths.State, 0700); err != nil {
		t.Fatalf("create directory at database path: %v", err)
	}

	result := openReactiveState(paths)
	if result.OK {
		state, _ := result.Value.(reactiveState)
		if state != nil {
			_ = state.Close()
		}
		t.Fatalf("openReactiveState(directory path) = %#v, want failure", result.Value)
	}
}

func TestReactiveState_Ugly(t *testing.T) {
	paths := testReactivePaths(t)
	opened := openReactiveState(paths)
	if !opened.OK {
		t.Fatalf("openReactiveState failed: %v", opened.Value)
	}
	state, ok := opened.Value.(reactiveState)
	if !ok {
		t.Fatalf("openReactiveState value = %T, want reactiveState", opened.Value)
	}
	defer func() {
		if result := state.Close(); !result.OK {
			t.Errorf("close state: %v", result.Value)
		}
	}()

	if result := state.Set(reactiveGroupDrafts, "session-empty", ""); !result.OK {
		t.Fatalf("store empty draft: %v", result.Value)
	}
	value, result := state.Get(reactiveGroupDrafts, "session-empty")
	if !result.OK || value != "" {
		t.Fatalf("stored empty draft = %q, %#v, want empty successful value", value, result.Value)
	}
	if result := state.Delete(reactiveGroupDrafts, "session-empty"); !result.OK {
		t.Fatalf("delete draft: %v", result.Value)
	}
	value, result = state.Get(reactiveGroupDrafts, "session-empty")
	if result.OK || value != "" {
		t.Fatalf("deleted draft = %q, %#v, want distinguishable miss", value, result.Value)
	}
}

func testReactivePaths(t *testing.T) appPaths {
	t.Helper()
	result := appPathsAt(t.TempDir())
	if !result.OK {
		t.Fatalf("appPathsAt: %v", result.Value)
	}
	paths, ok := result.Value.(appPaths)
	if !ok {
		t.Fatalf("appPathsAt value = %T, want appPaths", result.Value)
	}
	if err := os.MkdirAll(paths.Root+"/"+paths.Workspaces, 0700); err != nil {
		t.Fatalf("create workspace state directory: %v", err)
	}
	return paths
}
