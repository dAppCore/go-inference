// SPDX-License-Identifier: EUPL-1.2

package tui

import (
	"errors"
	"os"
	"testing"

	core "dappco.re/go"
)

func TestAppPaths_Good(t *testing.T) {
	root := t.TempDir()
	result := appPathsAt(root)
	if !result.OK {
		t.Fatalf("appPathsAt(%q) failed: %v", root, result.Value)
	}

	paths, ok := result.Value.(appPaths)
	if !ok {
		t.Fatalf("appPathsAt value = %T, want appPaths", result.Value)
	}
	want := appPaths{
		Root:       root,
		Database:   core.Path(root, "lem.duckdb"),
		State:      core.Path(root, "state.db"),
		Config:     "config.yaml",
		Workspaces: "workspaces",
		Packs:      "packs",
		Exports:    "exports",
	}
	if paths != want {
		t.Fatalf("appPathsAt(%q) = %#v, want %#v", root, paths, want)
	}
}

func TestAppPaths_Bad(t *testing.T) {
	result := appPathsAt("")
	if result.OK {
		t.Fatalf("appPathsAt empty root = %#v, want failure", result.Value)
	}
}

func TestAppFiles_Good(t *testing.T) {
	parent := t.TempDir()
	root := core.Path(parent, "lem")
	result := openAppFilesAt(root)
	if !result.OK {
		t.Fatalf("openAppFilesAt(%q) failed: %v", root, result.Value)
	}

	files, ok := result.Value.(appFiles)
	if !ok {
		t.Fatalf("openAppFilesAt value = %T, want appFiles", result.Value)
	}
	for _, directory := range []string{files.Paths.Workspaces, files.Paths.Packs, files.Paths.Exports} {
		if !files.Medium.IsDir(directory) {
			t.Errorf("medium directory %q was not created", directory)
		}
	}
	if err := files.Medium.Write(files.Paths.Config, "theme: midnight\n"); err != nil {
		t.Fatalf("write config through medium: %v", err)
	}
	content, err := files.Medium.Read(files.Paths.Config)
	if err != nil || content != "theme: midnight\n" {
		t.Fatalf("read config = %q, %v", content, err)
	}

	if err := files.Medium.Write("../escape.txt", "contained"); err != nil {
		t.Fatalf("write normalised traversal path: %v", err)
	}
	if _, err := os.Stat(core.Path(parent, "escape.txt")); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("traversal escaped sandbox: %v", err)
	}
}

func TestAppFiles_Ugly(t *testing.T) {
	root := core.Path(t.TempDir(), "lem")
	const original = "do not replace"
	if err := os.WriteFile(root, []byte(original), 0600); err != nil {
		t.Fatalf("write root fixture: %v", err)
	}

	result := openAppFilesAt(root)
	if result.OK {
		t.Fatalf("openAppFilesAt regular file = %#v, want failure", result.Value)
	}
	content, err := os.ReadFile(root)
	if err != nil {
		t.Fatalf("read preserved root fixture: %v", err)
	}
	if string(content) != original {
		t.Fatalf("root fixture changed to %q", content)
	}
}
