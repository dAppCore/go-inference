// SPDX-License-Identifier: EUPL-1.2

package tui

import (
	"context"
	"sync"
	"time"

	core "dappco.re/go"
	"dappco.re/go/inference/agent/gitserver"
	"dappco.re/go/inference/agent/orchestrator"
	"dappco.re/go/inference/agent/provider"
	"dappco.re/go/inference/agent/queue"
	"dappco.re/go/inference/agent/work"
	"dappco.re/go/inference/agent/workspace"
	coreio "dappco.re/go/io"
	coreprocess "dappco.re/go/process"
)

type workspaceResources struct {
	Paths       appPaths
	Files       coreio.Medium
	Repository  workspaceRepository
	State       reactiveState
	Preferences preferenceStore
	Agent       agentProvider
	Warnings    []string
}

type workspaceOpeners struct {
	Repository  func(path string) core.Result
	State       func(paths appPaths) core.Result
	Preferences func(files coreio.Medium, path string) core.Result
	Agent       func(context.Context, appFiles, workspaceRepository) core.Result
	Now         func() time.Time
}

type agentBootstrapInput struct {
	Files      appFiles
	Repository workspaceRepository
}

type agentBootstrapResult struct {
	Provider agentProvider
	Warnings []string
}

type nativeProviderDetection struct {
	Available int
	Warnings  []string
}

type agentBootstrapFactories struct {
	LoadPolicy          func(coreio.Medium, string) core.Result
	OpenWorkspaceMedium func(string) core.Result
	NewStore            func(workspaceRepository) core.Result
	GitAvailable        func() core.Result
	GitOptions          func(string) core.Result
	NewGitServer        func(gitserver.Options) core.Result
	NewWorkspaces       func(workspace.ManagerOptions) core.Result
	NewProviders        func(provider.Finder, map[string]provider.Config) core.Result
	DetectProviders     func(context.Context, *provider.Registry) core.Result
	NewQueue            func(queue.Policy, work.QueueState, []work.ProviderState) core.Result
	NewLauncher         func() core.Result
	NewOrchestrator     func(orchestrator.Options) core.Result
	Now                 func() time.Time
	IDs                 orchestrator.Identifier
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

	agentResult := openers.Agent(context.Background(), files, repository)
	agent := agentProvider(nil)
	if !agentResult.OK {
		reason := workspaceOpenError(agentResult, defaultAgentUnavailableReason)
		warnings = append(warnings, core.Concat("agent: ", reason.Error()))
		agent = newUnavailableAgentProvider(reason.Error())
	} else {
		switch value := agentResult.Value.(type) {
		case agentBootstrapResult:
			agent = value.Provider
			warnings = append(warnings, value.Warnings...)
		case agentProvider:
			agent = value
		default:
			reason := core.E("tui.workspace.agent", "invalid agent provider result", nil)
			warnings = append(warnings, core.Concat("agent: ", reason.Error()))
			agent = newUnavailableAgentProvider(reason.Error())
		}
	}
	if agent == nil {
		agent = newUnavailableAgentProvider(defaultAgentUnavailableReason)
	}

	return core.Ok(&workspaceResources{
		Paths:       files.Paths,
		Files:       files.Medium,
		Repository:  repository,
		State:       state,
		Preferences: preferences,
		Agent:       agent,
		Warnings:    warnings,
	})
}

func (resources *workspaceResources) Close() core.Result {
	if resources == nil {
		return core.Ok(nil)
	}
	failures := make([]string, 0, 3)
	if closeResult := resources.closeAgent(); !closeResult.OK {
		failures = append(failures, closeResult.Error())
	}
	if resources.State != nil {
		if closeResult := resources.State.Close(); !closeResult.OK {
			failures = append(failures, closeResult.Error())
		}
		resources.State = nil
	}
	if resources.Repository != nil {
		if closeResult := resources.Repository.Close(); !closeResult.OK {
			failures = append(failures, closeResult.Error())
		}
		resources.Repository = nil
	}
	if len(failures) > 0 {
		return core.Fail(core.E("tui.workspaceResources.Close", core.Join("; ", failures...), nil))
	}
	return core.Ok(nil)
}

func (resources *workspaceResources) closeAgent() core.Result {
	if resources == nil || resources.Agent == nil {
		return core.Ok(nil)
	}
	agent := resources.Agent
	resources.Agent = nil
	return agent.Close()
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
	if openers.Agent == nil {
		openers.Agent = openReadOnlyWorkspaceAgent
	}
	if openers.Now == nil {
		openers.Now = time.Now
	}
	return openers
}

func openReadOnlyWorkspaceAgent(context.Context, appFiles, workspaceRepository) core.Result {
	return core.Ok(newUnavailableAgentProvider("native agent execution is disabled for this read-only workspace"))
}

func openNativeWorkspaceAgent(ctx context.Context, files appFiles, repository workspaceRepository) core.Result {
	return composeNativeAgent(ctx, agentBootstrapInput{Files: files, Repository: repository}, agentBootstrapFactories{})
}

func composeNativeAgent(ctx context.Context, input agentBootstrapInput, factories agentBootstrapFactories) core.Result {
	if ctx == nil {
		return core.Fail(core.E("tui.composeNativeAgent", "agent bootstrap context is required", nil))
	}
	if input.Files.Medium == nil || input.Repository == nil {
		return core.Fail(core.E("tui.composeNativeAgent", "agent bootstrap files and repository are required", nil))
	}
	factories = factories.withDefaults()

	policyResult := factories.LoadPolicy(input.Files.Medium, input.Files.Paths.Agents)
	if !policyResult.OK {
		return core.Fail(core.E("tui.composeNativeAgent", core.Concat("agent policy is invalid; queue is frozen: ", policyResult.Error()), resultError(policyResult)))
	}
	policy, ok := policyResult.Value.(queue.Policy)
	if !ok {
		return core.Fail(core.E("tui.composeNativeAgent", "agent policy loader returned an invalid policy; queue is frozen", nil))
	}

	workspaceMediumResult := factories.OpenWorkspaceMedium(input.Files.Paths.Workspaces)
	if !workspaceMediumResult.OK {
		return agentCompositionFailure("open workspace medium", workspaceMediumResult, nil, nil)
	}
	workspaceMedium, ok := workspaceMediumResult.Value.(coreio.Medium)
	if !ok || workspaceMedium == nil {
		return core.Fail(core.E("tui.composeNativeAgent", "workspace medium constructor returned an invalid medium", nil))
	}

	storeResult := factories.NewStore(input.Repository)
	if !storeResult.OK {
		return agentCompositionFailure("open durable agent store", storeResult, nil, nil)
	}
	store, ok := storeResult.Value.(orchestrator.Store)
	if !ok || store == nil {
		return core.Fail(core.E("tui.composeNativeAgent", "agent store constructor returned an invalid store", nil))
	}

	if gitResult := factories.GitAvailable(); !gitResult.OK {
		return agentCompositionFailure("detect Git", gitResult, nil, nil)
	}
	optionsResult := factories.GitOptions(input.Files.Paths.SoftServe)
	if !optionsResult.OK {
		return agentCompositionFailure("configure private Git", optionsResult, nil, nil)
	}
	gitOptions, ok := optionsResult.Value.(gitserver.Options)
	if !ok {
		return core.Fail(core.E("tui.composeNativeAgent", "private Git options have an invalid type", nil))
	}
	serverResult := factories.NewGitServer(gitOptions)
	if !serverResult.OK {
		return agentCompositionFailure("construct lazy private Git", serverResult, nil, nil)
	}
	server, ok := serverResult.Value.(gitserver.Service)
	if !ok || server == nil {
		return core.Fail(core.E("tui.composeNativeAgent", "private Git constructor returned an invalid service", nil))
	}

	workspacesResult := factories.NewWorkspaces(workspace.ManagerOptions{
		Root: input.Files.Paths.Workspaces, Files: workspaceMedium, Git: workspace.ProcessRunner{}, Server: server,
		IDs: factories.IDs.New, Now: factories.Now,
	})
	if !workspacesResult.OK {
		return agentCompositionFailure("construct workspace manager", workspacesResult, server, nil)
	}
	workspaces, ok := workspacesResult.Value.(*workspace.Manager)
	if !ok || workspaces == nil {
		return agentCompositionFailure("construct workspace manager", core.Fail(core.E("tui.composeNativeAgent", "workspace constructor returned an invalid manager", nil)), server, nil)
	}

	providerConfigs := make(map[string]provider.Config, len(policy.Providers))
	for name, configured := range policy.Providers {
		providerConfigs[name] = provider.Config{
			Executable: configured.Executable, DefaultModel: configured.DefaultModel,
			CredentialEnv: append([]string(nil), configured.CredentialEnv...),
			Flags:         append([]string(nil), configured.Flags...),
		}
	}
	providersResult := factories.NewProviders(nil, providerConfigs)
	if !providersResult.OK {
		return agentCompositionFailure("construct provider registry", providersResult, server, nil)
	}
	providers, ok := providersResult.Value.(*provider.Registry)
	if !ok || providers == nil {
		return agentCompositionFailure("construct provider registry", core.Fail(core.E("tui.composeNativeAgent", "provider constructor returned an invalid registry", nil)), server, nil)
	}
	detectionResult := factories.DetectProviders(ctx, providers)
	if !detectionResult.OK {
		return agentCompositionFailure("detect native providers", detectionResult, server, nil)
	}
	detection, ok := detectionResult.Value.(nativeProviderDetection)
	if !ok {
		return agentCompositionFailure("detect native providers", core.Fail(core.E("tui.composeNativeAgent", "provider detection returned an invalid result", nil)), server, nil)
	}

	at := factories.Now()
	queueResult := factories.NewQueue(policy, work.QueueState{
		ID: "default", Status: work.QueueFrozen, Reason: "agent queue is frozen until explicitly started", UpdatedAt: at,
	}, nil)
	if !queueResult.OK {
		return agentCompositionFailure("construct queue controller", queueResult, server, nil)
	}
	controller, ok := queueResult.Value.(*queue.Controller)
	if !ok || controller == nil {
		return agentCompositionFailure("construct queue controller", core.Fail(core.E("tui.composeNativeAgent", "queue constructor returned an invalid controller", nil)), server, nil)
	}

	launcherResult := factories.NewLauncher()
	if !launcherResult.OK {
		return agentCompositionFailure("construct native launcher", launcherResult, server, nil)
	}
	launcher, ok := launcherResult.Value.(orchestrator.Launcher)
	if !ok || launcher == nil {
		return agentCompositionFailure("construct native launcher", core.Fail(core.E("tui.composeNativeAgent", "launcher constructor returned an invalid launcher", nil)), server, nil)
	}

	engineResult := factories.NewOrchestrator(orchestrator.Options{
		Store: store, GitServer: server, Workspaces: workspaces, Providers: providers,
		Queue: controller, Launcher: launcher, Clock: agentClock{now: factories.Now}, IDs: factories.IDs,
	})
	if !engineResult.OK {
		return agentCompositionFailure("construct native orchestrator", engineResult, server, launcher)
	}
	engine, ok := engineResult.Value.(nativeAgentEngine)
	if !ok || engine == nil {
		return agentCompositionFailure("construct native orchestrator", core.Fail(core.E("tui.composeNativeAgent", "orchestrator constructor returned an invalid engine", nil)), server, launcher)
	}
	adapterResult := newAgentAdapter(engine)
	if !adapterResult.OK {
		if closed := engine.Close(); !closed.OK {
			return core.Fail(core.E("tui.composeNativeAgent", adapterResult.Error(), resultError(closed)))
		}
		return adapterResult
	}
	return core.Ok(agentBootstrapResult{Provider: adapterResult.Value.(agentProvider), Warnings: detection.Warnings})
}

func (factories agentBootstrapFactories) withDefaults() agentBootstrapFactories {
	if factories.LoadPolicy == nil {
		factories.LoadPolicy = queue.LoadPolicy
	}
	if factories.OpenWorkspaceMedium == nil {
		factories.OpenWorkspaceMedium = func(root string) core.Result {
			medium, err := coreio.NewSandboxed(root)
			if err != nil {
				return core.Fail(core.E("tui.agentWorkspaceMedium", core.Concat("open ", root), err))
			}
			return core.Ok(coreio.Medium(medium))
		}
	}
	if factories.NewStore == nil {
		factories.NewStore = newDuckAgentStore
	}
	if factories.GitAvailable == nil {
		factories.GitAvailable = func() core.Result {
			program := &coreprocess.Program{Name: "git"}
			if found := program.Find(); !found.OK {
				return core.Fail(core.E("tui.agentGit", "Git executable is unavailable", resultError(found)))
			}
			return core.Ok(program.Path)
		}
	}
	if factories.GitOptions == nil {
		factories.GitOptions = gitserver.DefaultOptions
	}
	if factories.NewGitServer == nil {
		factories.NewGitServer = func(options gitserver.Options) core.Result { return gitserver.NewSoftServe(options) }
	}
	if factories.NewWorkspaces == nil {
		factories.NewWorkspaces = workspace.NewManager
	}
	if factories.NewProviders == nil {
		factories.NewProviders = provider.DefaultRegistry
	}
	if factories.DetectProviders == nil {
		factories.DetectProviders = detectNativeProviders
	}
	if factories.NewQueue == nil {
		factories.NewQueue = queue.NewController
	}
	if factories.NewLauncher == nil {
		factories.NewLauncher = newOwnedNativeLauncher
	}
	if factories.NewOrchestrator == nil {
		factories.NewOrchestrator = func(options orchestrator.Options) core.Result { return orchestrator.New(options) }
	}
	if factories.Now == nil {
		factories.Now = time.Now
	}
	if factories.IDs == nil {
		factories.IDs = &agentIdentifiers{}
	}
	return factories
}

func detectNativeProviders(ctx context.Context, registry *provider.Registry) core.Result {
	if registry == nil {
		return core.Fail(core.E("tui.detectNativeProviders", "provider registry is required", nil))
	}
	detection := nativeProviderDetection{Warnings: make([]string, 0)}
	for _, name := range registry.Names() {
		adapterResult := registry.Adapter(name)
		if !adapterResult.OK {
			detection.Warnings = append(detection.Warnings, core.Concat("agent provider ", name, ": ", adapterResult.Error()))
			continue
		}
		adapter, ok := adapterResult.Value.(provider.Adapter)
		if !ok {
			detection.Warnings = append(detection.Warnings, core.Concat("agent provider ", name, ": invalid adapter"))
			continue
		}
		result := adapter.Detect(ctx)
		if !result.OK {
			detection.Warnings = append(detection.Warnings, core.Concat("agent provider ", name, ": ", result.Error()))
			continue
		}
		value, ok := result.Value.(provider.Detection)
		if !ok {
			detection.Warnings = append(detection.Warnings, core.Concat("agent provider ", name, ": invalid detection result"))
			continue
		}
		if value.Available {
			detection.Available++
			continue
		}
		reason := core.Trim(value.Reason)
		if reason == "" {
			reason = core.Concat(name, " executable is unavailable")
		}
		detection.Warnings = append(detection.Warnings, core.Concat("agent provider ", name, ": ", reason))
	}
	return core.Ok(detection)
}

func agentCompositionFailure(stage string, failure core.Result, server gitserver.Service, launcher orchestrator.Launcher) core.Result {
	failures := []string{core.Concat(stage, ": ", failure.Error())}
	if launcher != nil {
		if result := launcher.Close(); !result.OK {
			failures = append(failures, core.Concat("launcher cleanup: ", result.Error()))
		}
	}
	if server != nil {
		if result := server.Close(); !result.OK {
			failures = append(failures, core.Concat("private Git cleanup: ", result.Error()))
		}
	}
	return core.Fail(core.E("tui.composeNativeAgent", core.Join("; ", failures...), resultError(failure)))
}

type agentClock struct{ now func() time.Time }

func (clock agentClock) Now() time.Time {
	if clock.now == nil {
		return time.Time{}
	}
	return clock.now()
}

type agentIdentifiers struct{}

func (*agentIdentifiers) New() string { return newRecordID() }

type ownedNativeLauncher struct {
	launcher orchestrator.Launcher
	service  *coreprocess.Service
	once     sync.Once
	result   core.Result
}

func newOwnedNativeLauncher() core.Result {
	app := core.New()
	serviceResult := coreprocess.NewService(coreprocess.Options{})(app)
	if !serviceResult.OK {
		return core.Fail(core.E("tui.newOwnedNativeLauncher", "construct process service", resultError(serviceResult)))
	}
	service, ok := serviceResult.Value.(*coreprocess.Service)
	if !ok || service == nil {
		return core.Fail(core.E("tui.newOwnedNativeLauncher", "process service constructor returned an invalid service", nil))
	}
	if started := service.OnStartup(context.Background()); !started.OK {
		return core.Fail(core.E("tui.newOwnedNativeLauncher", "start process service", resultError(started)))
	}
	launcherResult := orchestrator.NewNativeLauncher(service, []string{"PATH", "HOME", "TMPDIR", "LANG", "LC_ALL", "TERM"})
	if !launcherResult.OK {
		shutdown := service.OnShutdown(context.Background())
		if !shutdown.OK {
			return core.Fail(core.E("tui.newOwnedNativeLauncher", launcherResult.Error(), resultError(shutdown)))
		}
		return launcherResult
	}
	launcher, ok := launcherResult.Value.(orchestrator.Launcher)
	if !ok {
		if shutdown := service.OnShutdown(context.Background()); !shutdown.OK {
			return core.Fail(core.E("tui.newOwnedNativeLauncher", "native launcher constructor returned an invalid launcher", resultError(shutdown)))
		}
		return core.Fail(core.E("tui.newOwnedNativeLauncher", "native launcher constructor returned an invalid launcher", nil))
	}
	return core.Ok(orchestrator.Launcher(&ownedNativeLauncher{launcher: launcher, service: service, result: core.Ok(nil)}))
}

func (launcher *ownedNativeLauncher) DetectEnvironment(keys []string) core.Result {
	return launcher.launcher.DetectEnvironment(keys)
}

func (launcher *ownedNativeLauncher) Start(ctx context.Context, command provider.Command, output func(string, string)) core.Result {
	return launcher.launcher.Start(ctx, command, output)
}

func (launcher *ownedNativeLauncher) Close() core.Result {
	if launcher == nil {
		return core.Ok(nil)
	}
	launcher.once.Do(func() {
		failures := make([]string, 0, 2)
		if launcher.launcher != nil {
			if result := launcher.launcher.Close(); !result.OK {
				failures = append(failures, result.Error())
			}
		}
		if launcher.service != nil {
			if result := launcher.service.OnShutdown(context.Background()); !result.OK {
				failures = append(failures, result.Error())
			}
		}
		if len(failures) > 0 {
			launcher.result = core.Fail(core.E("tui.ownedNativeLauncher.Close", core.Join("; ", failures...), nil))
		}
	})
	return launcher.result
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
