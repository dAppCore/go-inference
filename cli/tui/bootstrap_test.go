// SPDX-License-Identifier: EUPL-1.2

package tui

import (
	"context"
	"os"
	"strings"
	"testing"
	"time"

	core "dappco.re/go"
	"dappco.re/go/inference/agent/gitserver"
	"dappco.re/go/inference/agent/orchestrator"
	"dappco.re/go/inference/agent/provider"
	"dappco.re/go/inference/agent/queue"
	"dappco.re/go/inference/agent/work"
	"dappco.re/go/inference/agent/workspace"
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
	for _, directory := range []string{appWorkspacesPath, files.Paths.Packs, files.Paths.Exports} {
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

func TestAgentBootstrap_ValidPolicy_Good(t *testing.T) {
	files := testWorkspaceFiles(t)
	if err := files.Medium.Write(files.Paths.Agents, "version: 1\ndispatch:\n  default_agent: codex\n  global_concurrency: 2\nproviders:\n  codex:\n    executable: custom-codex\n    default_model: gpt-5\n    credential_env: [OPENAI_API_KEY]\n    flags: [--search]\n"); err != nil {
		t.Fatalf("write policy: %v", err)
	}
	repository := openTestDuckRepository(t)
	defer closeTestDuckRepository(t, repository)
	engine := &fixtureNativeAgentEngine{capabilities: []work.Capability{{Name: "dispatch", Available: true}}}
	server := &fixtureGitServer{}
	var configured map[string]provider.Config
	factories := fixtureAgentBootstrapFactories(t, repository, engine, server)
	factories.NewProviders = func(_ provider.Finder, configurations map[string]provider.Config) core.Result {
		configured = configurations
		return provider.NewRegistry(&fixtureNativeProvider{name: "codex", available: true})
	}

	result := composeNativeAgent(context.Background(), agentBootstrapInput{Files: files, Repository: repository}, factories)
	if !result.OK {
		t.Fatalf("composeNativeAgent: %s", result.Error())
	}
	opened := result.Value.(agentBootstrapResult)
	if configured["codex"].Executable != "custom-codex" || configured["codex"].DefaultModel != "gpt-5" || len(configured["codex"].CredentialEnv) != 1 || len(configured["codex"].Flags) != 1 {
		t.Fatalf("mapped provider config = %#v", configured)
	}
	if server.startCalls != 0 {
		t.Fatalf("composition started lazy Git service %d times", server.startCalls)
	}
	if snapshot := opened.Provider.Snapshot(context.Background()); !snapshot.OK {
		t.Fatalf("adapter Snapshot: %s", snapshot.Error())
	}
	if server.startCalls != 0 {
		t.Fatalf("snapshot started lazy Git service %d times", server.startCalls)
	}
	if closed := opened.Provider.Close(); !closed.OK {
		t.Fatalf("provider Close: %s", closed.Error())
	}
}

func TestAgentBootstrap_MalformedPolicy_Bad(t *testing.T) {
	files := testWorkspaceFiles(t)
	if err := files.Medium.Write(files.Paths.Agents, "version: 99\ndispatch:\n  global_concurrency: 1\n"); err != nil {
		t.Fatalf("write malformed policy: %v", err)
	}
	repository := openTestDuckRepository(t)
	defer closeTestDuckRepository(t, repository)
	factories := fixtureAgentBootstrapFactories(t, repository, &fixtureNativeAgentEngine{}, &fixtureGitServer{})

	result := composeNativeAgent(context.Background(), agentBootstrapInput{Files: files, Repository: repository}, factories)
	if result.OK || !strings.Contains(result.Error(), "queue is frozen") || !strings.Contains(result.Error(), "version must be 1") {
		t.Fatalf("malformed policy result = %#v", result)
	}
}

func TestAgentBootstrap_OwnerContention_Bad(t *testing.T) {
	files := testWorkspaceFiles(t)
	repository := openTestDuckRepository(t)
	defer closeTestDuckRepository(t, repository)
	exact := "private Git owner lock is held by live PID 4242"
	openers := workspaceOpeners{Agent: func(context.Context, appFiles, workspaceRepository) core.Result {
		return core.Fail(core.E("test.gitserver", exact, nil))
	}}

	result := openWorkspaceWith(files, openers)
	if !result.OK {
		t.Fatalf("owner contention blocked the rest of workspace: %s", result.Error())
	}
	resources := result.Value.(*workspaceResources)
	defer resources.Close()
	assertWorkspaceWarning(t, resources.Warnings, exact)
	for _, capability := range resources.Agent.Capabilities() {
		if capability.Available || !strings.Contains(capability.Reason, exact) {
			t.Fatalf("degraded capability = %#v", capability)
		}
	}
}

func TestAgentBootstrap_MissingGitAndProviders_Ugly(t *testing.T) {
	files := testWorkspaceFiles(t)
	repository := openTestDuckRepository(t)
	defer closeTestDuckRepository(t, repository)
	t.Run("Git reason is exact", func(t *testing.T) {
		factories := fixtureAgentBootstrapFactories(t, repository, &fixtureNativeAgentEngine{}, &fixtureGitServer{})
		factories.GitAvailable = func() core.Result { return core.Fail(core.E("test.git", "git executable missing from PATH", nil)) }
		result := composeNativeAgent(context.Background(), agentBootstrapInput{Files: files, Repository: repository}, factories)
		if result.OK || !strings.Contains(result.Error(), "git executable missing from PATH") {
			t.Fatalf("missing Git result = %#v", result)
		}
	})
	t.Run("provider reasons are preserved", func(t *testing.T) {
		server := &fixtureGitServer{}
		factories := fixtureAgentBootstrapFactories(t, repository, &fixtureNativeAgentEngine{}, server)
		factories.NewProviders = func(provider.Finder, map[string]provider.Config) core.Result {
			return provider.NewRegistry(
				&fixtureNativeProvider{name: "codex", reason: "codex executable not found"},
				&fixtureNativeProvider{name: "claude", reason: "claude executable not found"},
			)
		}
		result := composeNativeAgent(context.Background(), agentBootstrapInput{Files: files, Repository: repository}, factories)
		if !result.OK {
			t.Fatalf("missing providers should leave durable controls usable: %s", result.Error())
		}
		warnings := result.Value.(agentBootstrapResult).Warnings
		assertWorkspaceWarning(t, warnings, "codex executable not found")
		assertWorkspaceWarning(t, warnings, "claude executable not found")
		_ = result.Value.(agentBootstrapResult).Provider.Close()
	})
}

func TestAgentBootstrap_PartialFailureCleanup_Ugly(t *testing.T) {
	files := testWorkspaceFiles(t)
	repository := openTestDuckRepository(t)
	defer closeTestDuckRepository(t, repository)
	stages := []string{"workspace-medium", "store", "git", "git-options", "git-server", "workspaces", "providers", "provider-detection", "queue", "launcher", "orchestrator"}
	for _, stage := range stages {
		t.Run(stage, func(t *testing.T) {
			server := &fixtureGitServer{}
			launcher := &fixtureAgentLauncher{}
			factories := fixtureAgentBootstrapFactories(t, repository, &fixtureNativeAgentEngine{}, server)
			factories.NewLauncher = func() core.Result { return core.Ok(orchestrator.Launcher(launcher)) }
			fail := func() core.Result { return core.Fail(core.E("test.bootstrap", "fail at "+stage, nil)) }
			switch stage {
			case "workspace-medium":
				factories.OpenWorkspaceMedium = func(string) core.Result { return fail() }
			case "store":
				factories.NewStore = func(workspaceRepository) core.Result { return fail() }
			case "git":
				factories.GitAvailable = fail
			case "git-options":
				factories.GitOptions = func(string) core.Result { return fail() }
			case "git-server":
				factories.NewGitServer = func(gitserver.Options) core.Result { return fail() }
			case "workspaces":
				factories.NewWorkspaces = func(workspace.ManagerOptions) core.Result { return fail() }
			case "providers":
				factories.NewProviders = func(provider.Finder, map[string]provider.Config) core.Result { return fail() }
			case "provider-detection":
				factories.DetectProviders = func(context.Context, *provider.Registry) core.Result { return fail() }
			case "queue":
				factories.NewQueue = func(queue.Policy, work.QueueState, []work.ProviderState) core.Result { return fail() }
			case "launcher":
				factories.NewLauncher = fail
			case "orchestrator":
				factories.NewOrchestrator = func(orchestrator.Options) core.Result { return fail() }
			}
			result := composeNativeAgent(context.Background(), agentBootstrapInput{Files: files, Repository: repository}, factories)
			if result.OK || !strings.Contains(result.Error(), "fail at "+stage) {
				t.Fatalf("failure result = %#v", result)
			}
			serverConstructed := stage == "workspaces" || stage == "providers" || stage == "provider-detection" || stage == "queue" || stage == "launcher" || stage == "orchestrator"
			if serverConstructed != (server.closeCalls == 1) {
				t.Fatalf("server Close calls = %d, constructed=%v", server.closeCalls, serverConstructed)
			}
			launcherConstructed := stage == "orchestrator"
			if launcherConstructed != (launcher.closeCalls == 1) {
				t.Fatalf("launcher Close calls = %d, constructed=%v", launcher.closeCalls, launcherConstructed)
			}
		})
	}
}

func TestAgentBootstrap_ResourceClosePreservesErrors_Ugly(t *testing.T) {
	order := make([]string, 0, 3)
	resources := &workspaceResources{
		Agent:      &failingCloseAgent{order: &order},
		State:      &failingCloseReactiveState{order: &order},
		Repository: &failingCloseWorkspaceRepository{order: &order},
	}
	result := resources.Close()
	if result.OK || strings.Join(order, ",") != "agent,state,repository" {
		t.Fatalf("Close = %#v, order=%#v", result, order)
	}
	for _, reason := range []string{"agent close failed", "state close failed", "repository close failed"} {
		if !strings.Contains(result.Error(), reason) {
			t.Fatalf("Close error %q does not preserve %q", result.Error(), reason)
		}
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
	if err := os.MkdirAll(paths.Workspaces, 0700); err != nil {
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

func fixtureAgentBootstrapFactories(t *testing.T, repository workspaceRepository, engine nativeAgentEngine, server *fixtureGitServer) agentBootstrapFactories {
	t.Helper()
	return agentBootstrapFactories{
		OpenWorkspaceMedium: func(string) core.Result { return core.Ok(coreio.Medium(coreio.NewMockMedium())) },
		NewStore:            func(workspaceRepository) core.Result { return newDuckAgentStore(repository) },
		GitAvailable:        func() core.Result { return core.Ok("/usr/bin/git") },
		GitOptions:          gitserver.DefaultOptions,
		NewGitServer:        func(gitserver.Options) core.Result { return core.Ok(gitserver.Service(server)) },
		NewWorkspaces:       func(workspace.ManagerOptions) core.Result { return core.Ok(&workspace.Manager{}) },
		NewProviders: func(provider.Finder, map[string]provider.Config) core.Result {
			return provider.NewRegistry(&fixtureNativeProvider{name: "codex", available: true})
		},
		DetectProviders: detectNativeProviders,
		NewQueue: func(queue.Policy, work.QueueState, []work.ProviderState) core.Result {
			return core.Ok(&queue.Controller{})
		},
		NewLauncher: func() core.Result { return core.Ok(orchestrator.Launcher(&fixtureAgentLauncher{})) },
		NewOrchestrator: func(orchestrator.Options) core.Result {
			return core.Ok(engine)
		},
		Now: func() time.Time { return time.Date(2026, time.July, 18, 12, 0, 0, 0, time.UTC) },
		IDs: &fixtureAgentIdentifiers{},
	}
}

type fixtureGitServer struct {
	startCalls int
	closeCalls int
}

func (server *fixtureGitServer) Start(context.Context) core.Result {
	server.startCalls++
	return core.Ok(gitserver.Health{Running: true, Address: "127.0.0.1:0"})
}

func (*fixtureGitServer) EnsureRepository(context.Context, string) core.Result {
	return core.Ok(gitserver.Repository{Name: "fixture", CloneURL: "ssh://127.0.0.1/fixture"})
}

func (*fixtureGitServer) Health(context.Context) core.Result {
	return core.Ok(gitserver.Health{Reason: "private Git service is not started"})
}

func (server *fixtureGitServer) Close() core.Result {
	server.closeCalls++
	return core.Ok(nil)
}

type fixtureNativeProvider struct {
	name      string
	available bool
	reason    string
}

func (adapter *fixtureNativeProvider) Name() string { return adapter.name }

func (adapter *fixtureNativeProvider) Detect(context.Context) core.Result {
	return core.Ok(provider.Detection{Provider: adapter.name, Available: adapter.available, Reason: adapter.reason})
}

func (adapter *fixtureNativeProvider) Build(provider.Launch) core.Result {
	return core.Ok(provider.Command{Provider: adapter.name, Executable: adapter.name, Receipt: adapter.name})
}

func (*fixtureNativeProvider) ParseLine(string, string) []provider.Output { return nil }

type fixtureAgentLauncher struct{ closeCalls int }

func (*fixtureAgentLauncher) DetectEnvironment([]string) core.Result { return core.Ok([]string{}) }

func (*fixtureAgentLauncher) Start(context.Context, provider.Command, func(string, string)) core.Result {
	return core.Fail(core.E("test.launcher", "not used", nil))
}

func (launcher *fixtureAgentLauncher) Close() core.Result {
	launcher.closeCalls++
	return core.Ok(nil)
}

type fixtureAgentIdentifiers struct{ next int }

func (ids *fixtureAgentIdentifiers) New() string {
	ids.next++
	return core.Sprintf("fixture-%d", ids.next)
}

type failingCloseAgent struct{ order *[]string }

func (*failingCloseAgent) Capabilities() []agentCapability { return nil }
func (*failingCloseAgent) Snapshot(context.Context) core.Result {
	return core.Ok(agentSnapshot{})
}
func (*failingCloseAgent) Review(context.Context, agentReviewRequest) core.Result {
	return core.Fail(core.E("test.agent", "not used", nil))
}
func (*failingCloseAgent) Run(context.Context, agentRequest) core.Result {
	return core.Fail(core.E("test.agent", "not used", nil))
}
func (agent *failingCloseAgent) Close() core.Result {
	*agent.order = append(*agent.order, "agent")
	return core.Fail(core.E("test.agent", "agent close failed", nil))
}

type failingCloseReactiveState struct{ order *[]string }

func (*failingCloseReactiveState) Get(string, string) (string, core.Result) {
	return "", core.Fail(core.E("test.state", "not used", nil))
}
func (*failingCloseReactiveState) Set(string, string, string) core.Result {
	return core.Fail(core.E("test.state", "not used", nil))
}
func (*failingCloseReactiveState) Delete(string, string) core.Result {
	return core.Fail(core.E("test.state", "not used", nil))
}
func (state *failingCloseReactiveState) Close() core.Result {
	*state.order = append(*state.order, "state")
	return core.Fail(core.E("test.state", "state close failed", nil))
}

type failingCloseWorkspaceRepository struct {
	workspaceRepository
	order *[]string
}

func (repository *failingCloseWorkspaceRepository) Close() core.Result {
	*repository.order = append(*repository.order, "repository")
	return core.Fail(core.E("test.repository", "repository close failed", nil))
}
