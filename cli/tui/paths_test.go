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
		Datasets:   core.Path(root, "datasets.duckdb"),
		State:      core.Path(root, "state.db"),
		Config:     "config.yaml",
		Agents:     "agents.yaml",
		SoftServe:  core.Path(root, "soft-serve"),
		Workspaces: core.Path(root, "workspaces"),
		Packs:      "packs",
		Exports:    "exports",
		Judges:     core.Path(root, "judges"),
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
	for _, directory := range []string{appWorkspacesPath, files.Paths.Packs, files.Paths.Exports, appJudgesPath} {
		if !files.Medium.IsDir(directory) {
			t.Errorf("medium directory %q was not created", directory)
		}
	}
	for _, directory := range []string{files.Paths.SoftServe, files.Paths.Workspaces, files.Paths.Judges} {
		if !core.PathIsAbs(directory) {
			t.Errorf("host directory %q is not absolute", directory)
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

// ---- OpenJudgesDir: the CLI-facing entry point (cli/judgetemplate.go's
// override resolution, Task 9) ----

// TestOpenJudgesDir_Good proves the exported entry point resolves + creates
// $HOME/.lem/judges (no --home flag surface — HOME is the only seam, same
// as TestOpenDatasetStore_Good one file over), and that it is idempotent on
// a second call against the same HOME.
func TestOpenJudgesDir_Good(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)

	want := core.Path(home, ".lem", "judges")

	first := OpenJudgesDir()
	core.RequireTrue(t, first.OK, "OpenJudgesDir first open")
	dir, ok := first.Value.(string)
	if !ok || dir != want {
		t.Fatalf("OpenJudgesDir value = %#v, want %q", first.Value, want)
	}
	if info, err := os.Stat(dir); err != nil || !info.IsDir() {
		t.Fatalf("judges dir missing at %s: %v", dir, err)
	}

	second := OpenJudgesDir()
	core.RequireTrue(t, second.OK, "OpenJudgesDir second open")
	if second.Value.(string) != want {
		t.Fatalf("OpenJudgesDir second value = %#v, want %q", second.Value, want)
	}
}

// TestOpenJudgesDir_Bad proves a root that cannot be prepared (HOME points
// at a regular file, so $HOME/.lem cannot be created) fails closed rather
// than returning a bogus path — mirrors TestOpenDatasetStore_Bad.
func TestOpenJudgesDir_Bad(t *testing.T) {
	blocker := core.Path(t.TempDir(), "home-is-a-file")
	if err := os.WriteFile(blocker, []byte("not a directory"), 0o600); err != nil {
		t.Fatalf("write blocker fixture: %v", err)
	}
	t.Setenv("HOME", blocker)

	result := OpenJudgesDir()
	core.AssertFalse(t, result.OK, "OpenJudgesDir over a blocked root must fail")
}
