// SPDX-License-Identifier: EUPL-1.2

package tui

import (
	"time"

	core "dappco.re/go"
	coreio "dappco.re/go/io"
)

type workspaceResources struct {
	Paths       appPaths
	Files       coreio.Medium
	Repository  workspaceRepository
	State       reactiveState
	Preferences preferenceStore
	Warnings    []string
}

type workspaceOpeners struct {
	Repository  func(path string) core.Result
	State       func(paths appPaths) core.Result
	Preferences func(files coreio.Medium, path string) core.Result
	Now         func() time.Time
}

func openWorkspace(root string, openers workspaceOpeners) core.Result {
	opened := openAppFilesAt(root)
	if !opened.OK {
		return core.Fail(core.E("tui.openWorkspace", "open application files", resultError(opened)))
	}
	files, ok := opened.Value.(appFiles)
	if !ok {
		return core.Fail(core.E("tui.openWorkspace", "invalid application files result", nil))
	}
	return openWorkspaceWith(files, openers)
}

func openWorkspaceWith(files appFiles, openers workspaceOpeners) core.Result {
	if files.Medium == nil {
		return core.Fail(core.E("tui.openWorkspaceWith", "application file medium is required", nil))
	}
	if result := ensureAppFiles(files.Medium, files.Paths); !result.OK {
		return core.Fail(core.E("tui.openWorkspaceWith", "ensure application files", resultError(result)))
	}
	openers = openers.withDefaults()
	warnings := make([]string, 0)

	repositoryResult := openers.Repository(files.Paths.Database)
	if !repositoryResult.OK {
		return core.Fail(core.E(
			"tui.openWorkspaceWith",
			core.Concat("open repository: ", files.Paths.Database),
			workspaceOpenError(repositoryResult, "open repository"),
		))
	}
	repository, ok := repositoryResult.Value.(workspaceRepository)
	if !ok {
		return core.Fail(core.E(
			"tui.openWorkspaceWith",
			core.Concat("open repository: ", files.Paths.Database),
			core.E("tui.workspace.repository", "invalid repository result", nil),
		))
	}
	if result := repository.InterruptActiveJobs(openers.Now()); !result.OK {
		if closeResult := repository.Close(); !closeResult.OK {
			core.Warn("tui.workspace.repository_close_after_recovery_failure", "error", closeResult.Value)
		}
		return core.Fail(core.E(
			"tui.openWorkspaceWith",
			core.Concat("recover active jobs: ", files.Paths.Database),
			workspaceOpenError(result, "recover active jobs"),
		))
	}

	stateResult := openers.State(files.Paths)
	state, stateOK := stateResult.Value.(reactiveState)
	if !stateResult.OK || !stateOK {
		reason := workspaceOpenError(stateResult, "open reactive state")
		if stateResult.OK && !stateOK {
			reason = core.E("tui.workspace.state", "invalid reactive state result", nil)
		}
		warnings = append(warnings, core.Concat("reactive state: ", reason.Error()))
		state = newDisabledReactiveState(reason)
	}

	preferencesResult := openers.Preferences(files.Medium, files.Paths.Config)
	preferences, preferencesOK := preferencesResult.Value.(preferenceStore)
	if !preferencesResult.OK || !preferencesOK {
		reason := workspaceOpenError(preferencesResult, "open preferences")
		if preferencesResult.OK && !preferencesOK {
			reason = core.E("tui.workspace.preferences", "invalid preference result", nil)
		}
		warnings = append(warnings, core.Concat("preferences: ", reason.Error()))
		fallback := openDegradedPreferences(files.Medium, files.Paths.Config, reason)
		if !fallback.OK {
			_ = state.Close()
			_ = repository.Close()
			return core.Fail(core.E("tui.openWorkspaceWith", "create preference fallback", resultError(fallback)))
		}
		preferences = fallback.Value.(preferenceStore)
	} else if warning := preferences.Warning(); warning != nil {
		warnings = append(warnings, core.Concat("preferences: ", warning.Error()))
	}

	return core.Ok(&workspaceResources{
		Paths:       files.Paths,
		Files:       files.Medium,
		Repository:  repository,
		State:       state,
		Preferences: preferences,
		Warnings:    warnings,
	})
}

func (resources *workspaceResources) Close() core.Result {
	if resources == nil {
		return core.Ok(nil)
	}
	result := core.Ok(nil)
	if resources.State != nil {
		if closeResult := resources.State.Close(); !closeResult.OK {
			result = closeResult
		}
		resources.State = nil
	}
	if resources.Repository != nil {
		if closeResult := resources.Repository.Close(); !closeResult.OK && result.OK {
			result = closeResult
		}
		resources.Repository = nil
	}
	return result
}

func (openers workspaceOpeners) withDefaults() workspaceOpeners {
	if openers.Repository == nil {
		openers.Repository = openDuckRepository
	}
	if openers.State == nil {
		openers.State = openReactiveState
	}
	if openers.Preferences == nil {
		openers.Preferences = openPreferences
	}
	if openers.Now == nil {
		openers.Now = time.Now
	}
	return openers
}

func openDegradedPreferences(medium coreio.Medium, path string, warning error) core.Result {
	fallback := openPreferences(coreio.NewMemoryMedium(), path)
	if !fallback.OK {
		return fallback
	}
	preferences, ok := fallback.Value.(*configPreferenceStore)
	if !ok {
		return core.Fail(core.E("tui.openDegradedPreferences", "invalid preference fallback", nil))
	}
	preferences.mu.Lock()
	preferences.medium = medium
	preferences.path = path
	preferences.warning = warning
	preferences.commitDisabled = true
	preferences.mu.Unlock()
	return core.Ok(preferenceStore(preferences))
}

func workspaceOpenError(result core.Result, fallback string) error {
	if err := resultError(result); err != nil {
		return err
	}
	return core.E("tui.workspace", fallback, nil)
}
