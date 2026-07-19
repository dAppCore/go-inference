// SPDX-License-Identifier: EUPL-1.2

package tui

import (
	core "dappco.re/go"
	coreio "dappco.re/go/io"
)

const (
	appConfigPath     = "config.yaml"
	appAgentsPath     = "agents.yaml"
	appSoftServePath  = "soft-serve"
	appWorkspacesPath = "workspaces"
	appPacksPath      = "packs"
	appExportsPath    = "exports"
	appJudgesPath     = "judges"
)

// appPaths keeps host-only database paths separate from medium-relative files.
type appPaths struct {
	Root     string
	Database string
	// Datasets is the host-only path to datasets.duckdb — a separate
	// file from Database (lem.duckdb), per the dataset loop design's
	// bulk/lifecycle/blast-radius decision. Only the DuckDB adapter
	// (newDuckDatasetStore) receives this resolved filename.
	Datasets   string
	State      string
	Config     string
	Agents     string
	SoftServe  string
	Workspaces string
	Packs      string
	Exports    string
	// Judges is the host-only path to ~/.lem/judges — user overrides for
	// judge-tier score templates (the dataset loop design's "in-repo
	// defaults + ~/.lem/judges/ overrides" decision, Task 9). A CLI-side
	// judge driver reads <name>.md from here directly; a present override
	// wins over the in-repo judges/<name>.md default of the same name.
	Judges string
}

// appFiles is the application's replaceable file boundary.
type appFiles struct {
	Paths  appPaths
	Medium coreio.Medium
}

func defaultAppPaths() core.Result {
	homeResult := core.UserHomeDir()
	if !homeResult.OK {
		return core.Fail(core.E("tui.defaultAppPaths", "resolve user home", resultError(homeResult)))
	}
	home := homeResult.String()
	if core.Trim(home) == "" {
		return core.Fail(core.E("tui.defaultAppPaths", "user home is empty", nil))
	}
	return appPathsAt(core.Path(home, ".lem"))
}

func appPathsAt(root string) core.Result {
	root = core.Trim(root)
	if root == "" {
		return core.Fail(core.E("tui.appPathsAt", "application root is required", nil))
	}
	root = core.Path(root)
	return core.Ok(appPaths{
		Root:       root,
		Database:   core.Path(root, "lem.duckdb"),
		Datasets:   core.Path(root, "datasets.duckdb"),
		State:      core.Path(root, "state.db"),
		Config:     appConfigPath,
		Agents:     appAgentsPath,
		SoftServe:  core.Path(root, appSoftServePath),
		Workspaces: core.Path(root, appWorkspacesPath),
		Packs:      appPacksPath,
		Exports:    appExportsPath,
		Judges:     core.Path(root, appJudgesPath),
	})
}

func openAppFilesAt(root string) core.Result {
	pathsResult := appPathsAt(root)
	if !pathsResult.OK {
		return pathsResult
	}
	paths, ok := pathsResult.Value.(appPaths)
	if !ok {
		return core.Fail(core.E("tui.openAppFilesAt", "invalid path result", nil))
	}

	medium, err := coreio.NewSandboxed(paths.Root)
	if err != nil {
		return core.Fail(core.E("tui.openAppFilesAt", core.Concat("open application root: ", paths.Root), err))
	}
	if ensureResult := ensureAppFiles(medium, paths); !ensureResult.OK {
		return ensureResult
	}
	return core.Ok(appFiles{Paths: paths, Medium: medium})
}

func ensureAppFiles(medium coreio.Medium, paths appPaths) core.Result {
	if medium == nil {
		return core.Fail(core.E("tui.ensureAppFiles", "file medium is required", nil))
	}
	for _, directory := range []string{"", appWorkspacesPath, paths.Packs, paths.Exports, appJudgesPath} {
		if err := medium.EnsureDir(directory); err != nil {
			label := directory
			if label == "" {
				label = paths.Root
			}
			return core.Fail(core.E("tui.ensureAppFiles", core.Concat("ensure directory: ", label), err))
		}
	}
	return core.Ok(nil)
}

func resultError(result core.Result) error {
	err, _ := result.Value.(error)
	return err
}

// OpenJudgesDir resolves and ensures ~/.lem/judges — the medium-backed
// override directory a CLI-side judge driver reads <name>.md templates
// from (Task 9) — returning its host-absolute path. Mirrors
// OpenDatasetStore's resolve-then-ensure sequence without opening a
// database; there is no --home override, HOME is the only seam.
//
//	dirResult := tui.OpenJudgesDir()
//	dir := dirResult.String() // e.g. "/home/snider/.lem/judges"
func OpenJudgesDir() core.Result {
	pathsResult := defaultAppPaths()
	if !pathsResult.OK {
		return pathsResult
	}
	paths, ok := pathsResult.Value.(appPaths)
	if !ok {
		return core.Fail(core.E("tui.OpenJudgesDir", "invalid application paths result", nil))
	}
	if ensured := openAppFilesAt(paths.Root); !ensured.OK {
		return core.Fail(core.E("tui.OpenJudgesDir", "ensure application layout", resultError(ensured)))
	}
	return core.Ok(paths.Judges)
}
