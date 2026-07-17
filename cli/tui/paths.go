// SPDX-License-Identifier: EUPL-1.2

package tui

import (
	core "dappco.re/go"
	coreio "dappco.re/go/io"
)

const (
	appConfigPath     = "config.yaml"
	appWorkspacesPath = "workspaces"
	appPacksPath      = "packs"
	appExportsPath    = "exports"
)

// appPaths keeps host-only database paths separate from medium-relative files.
type appPaths struct {
	Root       string
	Database   string
	State      string
	Config     string
	Workspaces string
	Packs      string
	Exports    string
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
	home, ok := homeResult.Value.(string)
	if !ok || core.Trim(home) == "" {
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
		State:      core.Path(root, "state.db"),
		Config:     appConfigPath,
		Workspaces: appWorkspacesPath,
		Packs:      appPacksPath,
		Exports:    appExportsPath,
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
	for _, directory := range []string{"", paths.Workspaces, paths.Packs, paths.Exports} {
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
