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
	"dappco.re/go/inference/dataset"
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

// TestOpenWorkspace_DatasetStoreOpensAndCloses proves the Data panel's
// store attaches through openWorkspaceWithContext exactly like Repository/
// State already do (Task 8) — its own file, datasets.duckdb, separate
// from lem.duckdb per the design's bulk/lifecycle/blast-radius decision.
func TestOpenWorkspace_DatasetStoreOpensAndCloses(t *testing.T) {
	files := testWorkspaceFiles(t)
	opened := openWorkspaceWith(files, workspaceOpeners{})
	if !opened.OK {
		t.Fatalf("openWorkspaceWith: %v", opened.Value)
	}
	resources := opened.Value.(*workspaceResources)
	if resources.DatasetStore == nil {
		t.Fatal("openWorkspaceWith did not attach a DatasetStore")
	}
	if _, err := os.Stat(files.Paths.Datasets); err != nil {
		t.Errorf("migrated datasets.duckdb %q: %v", files.Paths.Datasets, err)
	}
	if len(resources.Warnings) != 0 {
		t.Errorf("healthy dataset store warnings = %#v, want none", resources.Warnings)
	}
	if files.Paths.Datasets == files.Paths.Database {
		t.Fatal("datasets.duckdb and lem.duckdb resolved to the same path")
	}

	ds := dataset.Dataset{ID: dataset.NewID(), Slug: "bootstrap-vents", Title: "bootstrap", CreatedAt: time.Now()}
	if result := resources.DatasetStore.DatasetCreate(ds); !result.OK {
		t.Fatalf("write through the bootstrapped dataset store: %v", result.Value)
	}

	if result := resources.Close(); !result.OK {
		t.Fatalf("close workspace resources: %v", result.Value)
	}
	if resources.DatasetStore != nil {
		t.Fatal("Close did not clear resources.DatasetStore")
	}
}

// TestOpenWorkspace_DatasetStoreFailureDegradesWithWarningNotFailure proves
// the design's own rationale for a separate file: "a bloated or damaged
// dataset file must never take the agent/TUI state down with it" — a
// dataset store that cannot open degrades to nil + a warning, exactly
// like State/Preferences already do, never a hard openWorkspaceWith
// failure the way a broken Repository is.
func TestOpenWorkspace_DatasetStoreFailureDegradesWithWarningNotFailure(t *testing.T) {
	files := testWorkspaceFiles(t)
	if err := os.MkdirAll(files.Paths.Datasets, 0o755); err != nil {
		t.Fatalf("seed a directory at the dataset path: %v", err)
	}
	opened := openWorkspaceWith(files, workspaceOpeners{})
	if !opened.OK {
		t.Fatalf("openWorkspaceWith failed instead of degrading: %v", opened.Value)
	}
	resources := opened.Value.(*workspaceResources)
	if resources.DatasetStore != nil {
		t.Fatal("resources.DatasetStore is non-nil despite the open failure")
	}
	assertWorkspaceWarning(t, resources.Warnings, "dataset store")
	if result := resources.Close(); !result.OK {
		t.Fatalf("close workspace resources: %v", result.Value)
	}
}

func TestOpenWorkspaceNativePreflightRunsBeforeStateMutation(t *testing.T) {
	files := testWorkspaceFiles(t)
	calls := 0
	result := openWorkspaceWith(files, workspaceOpeners{
		Preflight: func(context.Context) core.Result {
			return core.Fail(core.NewError("linked native runtime requires release/update"))
		},
		Repository: func(string) core.Result {
			calls++
			return core.Fail(core.NewError("repository must not open"))
		},
		State: func(appPaths) core.Result {
			calls++
			return core.Fail(core.NewError("state must not open"))
		},
		Agent: func(context.Context, appFiles, workspaceRepository) core.Result {
			calls++
			return core.Fail(core.NewError("agent must not open"))
		},
	})
	core.AssertFalse(t, result.OK)
	core.AssertContains(t, result.Error(), "release/update")
	core.AssertEqual(t, 0, calls)
	core.AssertFalse(t, files.Medium.IsDir(appWorkspacesPath))
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

func TestAgentBootstrap_LauncherEnvironmentAllowlist_Good(t *testing.T) {
	want := []string{"PATH", "HOME", "USER", "TMPDIR", "TMP", "TEMP", "LANG", "LC_ALL", "SHELL"}
	got := nativeAgentEssentialEnvironment()
	if core.JSONMarshalString(got) != core.JSONMarshalString(want) {
		t.Fatalf("launcher environment allowlist = %#v, want %#v", got, want)
	}
	for _, key := range got {
		if key == "TERM" {
			t.Fatal("launcher environment allowlist includes TERM")
		}
	}
}

func TestAgentBootstrap_DurableQueueState_Good(t *testing.T) {
	files := testWorkspaceFiles(t)
	repository := openTestDuckRepository(t)
	defer closeTestDuckRepository(t, repository)
	at := time.Date(2026, time.July, 18, 10, 30, 0, 0, time.UTC)
	durable := work.Snapshot{
		Queue: work.QueueState{ID: "default", Status: work.QueueDraining, Reason: "recovering active work", UpdatedAt: at},
		Providers: []work.ProviderState{{
			Provider: "codex", BackoffReason: "quota", LastRunID: "run-9",
			BackoffUntil: at.Add(45 * time.Minute), LastStartedAt: at.Add(-time.Minute),
			WindowStartedAt: at.Add(-3 * time.Hour), WindowAdmissions: 17, UpdatedAt: at,
		}},
	}
	engine := &fixtureNativeAgentEngine{}
	server := &fixtureGitServer{}
	factories := fixtureAgentBootstrapFactories(t, repository, engine, server)
	baseStore := newDuckAgentStore(repository).Value.(orchestrator.Store)
	factories.NewStore = func(workspaceRepository) core.Result {
		return core.Ok(orchestrator.Store(&fixtureSnapshotStore{Store: baseStore, result: core.Ok(durable)}))
	}
	var gotQueue work.QueueState
	var gotProviders []work.ProviderState
	factories.NewQueue = func(_ queue.Policy, initial work.QueueState, providers []work.ProviderState) core.Result {
		gotQueue = initial
		gotProviders = append([]work.ProviderState(nil), providers...)
		return core.Ok(&queue.Controller{})
	}

	result := composeNativeAgent(context.Background(), agentBootstrapInput{Files: files, Repository: repository}, factories)
	if !result.OK {
		t.Fatalf("composeNativeAgent: %s", result.Error())
	}
	defer result.Value.(agentBootstrapResult).Provider.Close()
	if core.JSONMarshalString(gotQueue) != core.JSONMarshalString(durable.Queue) || core.JSONMarshalString(gotProviders) != core.JSONMarshalString(durable.Providers) {
		t.Fatalf("NewQueue state = queue %#v providers %#v, want queue %#v providers %#v", gotQueue, gotProviders, durable.Queue, durable.Providers)
	}
}

func TestAgentBootstrapPropagatesDispatchAttemptTimeout(t *testing.T) {
	files := testWorkspaceFiles(t)
	repository := openTestDuckRepository(t)
	defer closeTestDuckRepository(t, repository)
	engine := &fixtureNativeAgentEngine{}
	factories := fixtureAgentBootstrapFactories(t, repository, engine, &fixtureGitServer{})
	policy := queue.Policy{
		Version:     1,
		Dispatch:    queue.DispatchConfig{DefaultAgent: "codex", GlobalConcurrency: 1, TimeoutMinutes: 7},
		Concurrency: map[string]queue.ConcurrencyLimit{"codex": {Total: 1}},
		Rates:       map[string]queue.RateConfig{}, Providers: map[string]queue.NativeConfig{},
	}
	factories.LoadPolicy = func(coreio.Medium, string) core.Result { return core.Ok(policy) }
	var captured orchestrator.Options
	var capturedWorkspace workspace.ManagerOptions
	factories.NewWorkspaces = func(options workspace.ManagerOptions) core.Result {
		capturedWorkspace = options
		return core.Ok(&workspace.Manager{})
	}
	factories.NewOrchestrator = func(options orchestrator.Options) core.Result {
		captured = options
		return core.Ok(nativeAgentEngine(engine))
	}
	result := composeNativeAgent(context.Background(), agentBootstrapInput{Files: files, Repository: repository}, factories)
	core.AssertTrue(t, result.OK, result.Error())
	defer result.Value.(agentBootstrapResult).Provider.Close()
	duration, available := agentOrchestratorDurationOption(captured, "AttemptTimeout")
	core.AssertTrue(t, available)
	core.AssertEqual(t, 7*time.Minute, duration)
	cleanup, cleanupAvailable := agentOrchestratorDurationOption(captured, "CleanupTimeout")
	core.AssertTrue(t, cleanupAvailable)
	core.AssertEqual(t, defaultAgentCleanupTimeout, cleanup)
	managerCleanup, managerCleanupAvailable := agentWorkspaceDurationOption(capturedWorkspace, "CleanupTimeout")
	core.AssertTrue(t, managerCleanupAvailable)
	core.AssertEqual(t, defaultAgentCleanupTimeout, managerCleanup)
}

func TestAgentBootstrapRejectsUnhardenedRuntimeBeforeCompositionMutation(t *testing.T) {
	files := testWorkspaceFiles(t)
	repository := openTestDuckRepository(t)
	defer closeTestDuckRepository(t, repository)
	factories := agentBootstrapFactories{}
	mutations := 0
	factories.RuntimeContract = func(context.Context) core.Result {
		return core.Fail(core.NewError("linked inference runtime has no hardened contract"))
	}
	factories.LoadPolicy = func(coreio.Medium, string) core.Result {
		mutations++
		return core.Fail(core.NewError("policy loader must not run"))
	}

	result := composeNativeAgent(context.Background(), agentBootstrapInput{Files: files, Repository: repository}, factories)
	core.AssertFalse(t, result.OK)
	core.AssertContains(t, result.Error(), "release/update")
	core.AssertEqual(t, 0, mutations)
}

func TestAgentBootstrapContractCancellationDoesNotMasqueradeAsReleaseMismatch(t *testing.T) {
	files := testWorkspaceFiles(t)
	repository := openTestDuckRepository(t)
	defer closeTestDuckRepository(t, repository)
	ctx, cancel := context.WithCancel(context.Background())
	mutations := 0
	factories := agentBootstrapFactories{
		RuntimeContract: func(context.Context) core.Result {
			cancel()
			return core.Fail(core.NewError("contract interrupted"))
		},
		LoadPolicy: func(coreio.Medium, string) core.Result {
			mutations++
			return core.Fail(core.NewError("policy loader must not run"))
		},
	}

	result := composeNativeAgent(ctx, agentBootstrapInput{Files: files, Repository: repository}, factories)
	core.AssertFalse(t, result.OK)
	core.AssertContains(t, result.Error(), "context is done")
	core.AssertFalse(t, strings.Contains(result.Error(), "release/update"))
	core.AssertEqual(t, 0, mutations)
}

func TestAgentBootstrapLinkedRuntimeContractBoundary(t *testing.T) {
	result := linkedAgentRuntimeContract(context.Background())
	_, hardened := any(orchestrator.Options{}).(agentHardenedRuntime)
	if hardened {
		core.AssertTrue(t, result.OK, result.Error())
		core.AssertEqual(t, agentHardenedRuntimeContract, result.Value.(string))
		return
	}
	core.AssertFalse(t, result.OK)
	core.AssertContains(t, result.Error(), "does not expose the hardened native runtime contract")
}

func TestAgentBootstrap_DurableSnapshotFailure_Bad(t *testing.T) {
	files := testWorkspaceFiles(t)
	repository := openTestDuckRepository(t)
	defer closeTestDuckRepository(t, repository)
	for _, test := range []struct {
		name   string
		result core.Result
		want   string
	}{
		{name: "read failure", result: core.Fail(core.E("test.snapshot", "durable queue offline", nil)), want: "durable queue offline"},
		{name: "type failure", result: core.Ok("not a snapshot"), want: "invalid durable snapshot"},
	} {
		t.Run(test.name, func(t *testing.T) {
			server := &fixtureGitServer{}
			launcher := &fixtureAgentLauncher{}
			factories := fixtureAgentBootstrapFactories(t, repository, &fixtureNativeAgentEngine{}, server)
			baseStore := newDuckAgentStore(repository).Value.(orchestrator.Store)
			factories.NewStore = func(workspaceRepository) core.Result {
				return core.Ok(orchestrator.Store(&fixtureSnapshotStore{Store: baseStore, result: test.result}))
			}
			factories.NewLauncher = func() core.Result { return core.Ok(orchestrator.Launcher(launcher)) }
			result := composeNativeAgent(context.Background(), agentBootstrapInput{Files: files, Repository: repository}, factories)
			if result.OK || !strings.Contains(result.Error(), "load durable queue state") || !strings.Contains(result.Error(), test.want) {
				t.Fatalf("snapshot failure = %#v, want %q", result, test.want)
			}
			if server.startCalls != 0 || server.closeCalls != 0 || launcher.closeCalls != 0 {
				t.Fatalf("snapshot failure constructed later resources: server start=%d close=%d launcher close=%d", server.startCalls, server.closeCalls, launcher.closeCalls)
			}
		})
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
	startFailure := core.Fail(core.E("test.gitserver", "private Git owner lock is held by live PID 4242", nil))
	exact := startFailure.Error()
	server := &fixtureGitServer{startResult: startFailure}
	projectReview := orchestrator.ProjectReview{
		Work:           work.Item{ID: "work-owner", Title: "Owner contention", Task: "review lazily", Repository: "/src/owner"},
		Source:         workspace.SourceReview{Path: "/src/owner", Root: "/src/owner", Branch: "main", Revision: "abc", IncludedHash: "hash"},
		RepositoryName: "work-owner",
	}
	engine := &fixtureNativeAgentEngine{capabilities: nativeFixtureCapabilities(), projectReview: projectReview}
	factories := fixtureAgentBootstrapFactories(t, repository, engine, server)
	factories.NewOrchestrator = func(options orchestrator.Options) core.Result {
		engine.registerProject = func(ctx context.Context, _ orchestrator.ProjectReview, _ bool) core.Result {
			registered := options.GitServer.EnsureRepository(ctx, projectReview.RepositoryName)
			if !registered.OK {
				return registered
			}
			return core.Ok(engine.registeredProject)
		}
		return core.Ok(nativeAgentEngine(engine))
	}
	openers := workspaceOpeners{Agent: func(ctx context.Context, files appFiles, repository workspaceRepository) core.Result {
		return composeNativeAgent(ctx, agentBootstrapInput{Files: files, Repository: repository}, factories)
	}}

	result := openWorkspaceWith(files, openers)
	if !result.OK {
		t.Fatalf("owner contention blocked the rest of workspace: %s", result.Error())
	}
	resources := result.Value.(*workspaceResources)
	defer resources.Close()
	if server.startCalls != 0 {
		t.Fatalf("composition started lazy Git service %d times", server.startCalls)
	}
	if snapshot := resources.Agent.Snapshot(context.Background()); !snapshot.OK {
		t.Fatalf("Snapshot: %s", snapshot.Error())
	}
	if server.startCalls != 0 {
		t.Fatalf("snapshot started lazy Git service %d times", server.startCalls)
	}
	reviewResult := resources.Agent.Review(context.Background(), agentReviewRequest{
		Feature: agentFeatureDispatch, WorkID: projectReview.Work.ID,
		Work: agentWorkRequest{ID: projectReview.Work.ID, Title: projectReview.Work.Title, Task: projectReview.Work.Task, Repository: projectReview.Work.Repository},
	})
	if !reviewResult.OK || server.startCalls != 0 {
		t.Fatalf("project Review = %#v, start calls=%d", reviewResult, server.startCalls)
	}
	action := resources.Agent.Run(context.Background(), agentRequest{Feature: agentFeatureDispatch, Review: reviewResult.Value.(agentReview), Confirmed: true})
	if action.OK || action.Error() != exact || server.startCalls != 1 {
		t.Fatalf("owner action = %#v, start calls=%d, want exact %q", action, server.startCalls, exact)
	}
	available := make(map[agentFeature]agentCapability)
	for _, capability := range resources.Agent.Capabilities() {
		available[capability.Feature] = capability
	}
	for _, feature := range []agentFeature{agentFeatureDispatch, agentFeatureRetry, agentFeatureResume, agentFeatureChangesReview, agentFeatureAccept, agentFeatureReject} {
		if capability := available[feature]; capability.Available || capability.Reason != exact {
			t.Fatalf("Git-dependent capability %q = %#v", feature, capability)
		}
	}
	for _, feature := range []agentFeature{agentFeatureCancel, agentFeatureAnswer, agentFeatureQueueStart, agentFeatureQueueStop} {
		if capability := available[feature]; !capability.Available {
			t.Fatalf("unrelated capability %q degraded: %#v", feature, capability)
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
	if result.OK || strings.Join(order, ",") != "agent" {
		t.Fatalf("Close = %#v, order=%#v", result, order)
	}
	if !strings.Contains(result.Error(), "agent close failed") {
		t.Fatalf("Close error %q does not preserve agent failure", result.Error())
	}
	if resources.State == nil || resources.Repository == nil {
		t.Fatalf("Close destroyed retry dependencies after agent failure: %#v", resources)
	}
}

func TestAgentBootstrapResourceCloseRetriesAgentOwnershipCleanup(t *testing.T) {
	repository := openTestDuckRepository(t)
	state := &retryCloseReactiveState{}
	agent := &retryCloseAgent{repository: repository}
	resources := &workspaceResources{Agent: agent, State: state, Repository: repository}
	first := resources.Close()
	core.AssertFalse(t, first.OK)
	core.AssertTrue(t, resources.Agent != nil)
	core.AssertTrue(t, resources.State != nil)
	core.AssertTrue(t, resources.Repository != nil)
	core.AssertFalse(t, state.closed)
	core.AssertTrue(t, repository.ListSessions(false).OK)
	second := resources.Close()
	core.AssertTrue(t, second.OK, second.Error())
	core.AssertEqual(t, 2, agent.calls)
	core.AssertTrue(t, resources.Agent == nil)
	core.AssertTrue(t, resources.State == nil)
	core.AssertTrue(t, resources.Repository == nil)
	core.AssertTrue(t, state.closed)
}

func TestAgentBootstrapOwnedNativeLauncherCloseRetriesFailure(t *testing.T) {
	inner := &fixtureAgentLauncher{closeFailures: 1}
	launcher := &ownedNativeLauncher{launcher: inner, result: core.Ok(nil)}
	first := launcher.Close()
	core.AssertFalse(t, first.OK)
	core.AssertEqual(t, 1, inner.closeCalls)
	second := launcher.Close()
	core.AssertTrue(t, second.OK, second.Error())
	core.AssertEqual(t, 2, inner.closeCalls)
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

func TestOpenWorkspace_Context_Good(t *testing.T) {
	type contextKey string
	const key contextKey = "agent-bootstrap"
	ctx := context.WithValue(context.Background(), key, "caller-context")
	files := testWorkspaceFiles(t)
	seen := ""
	result := openWorkspaceWithContext(ctx, files, workspaceOpeners{
		Agent: func(agentContext context.Context, workspaceFiles appFiles, repository workspaceRepository) core.Result {
			factories := fixtureAgentBootstrapFactories(t, repository, &fixtureNativeAgentEngine{}, &fixtureGitServer{})
			factories.DetectProviders = func(providerContext context.Context, registry *provider.Registry) core.Result {
				seen, _ = providerContext.Value(key).(string)
				return detectNativeProviders(providerContext, registry)
			}
			return composeNativeAgent(agentContext, agentBootstrapInput{Files: workspaceFiles, Repository: repository}, factories)
		},
	})
	if !result.OK {
		t.Fatalf("openWorkspaceWithContext: %s", result.Error())
	}
	defer result.Value.(*workspaceResources).Close()
	if seen != "caller-context" {
		t.Fatalf("provider detection context value = %q, want caller-context", seen)
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
	closeOrder   *[]string
	closeFailure string
}

func (repository *trackingWorkspaceRepository) Close() core.Result {
	*repository.closeOrder = append(*repository.closeOrder, "repository")
	result := repository.workspaceRepository.Close()
	if !result.OK {
		return result
	}
	if repository.closeFailure != "" {
		return core.Fail(core.E("test.repository.Close", repository.closeFailure, nil))
	}
	return core.Ok(nil)
}

type trackingReactiveState struct {
	reactiveState
	closeOrder   *[]string
	closeFailure string
}

func (state *trackingReactiveState) Close() core.Result {
	*state.closeOrder = append(*state.closeOrder, "state")
	result := state.reactiveState.Close()
	if !result.OK {
		return result
	}
	if state.closeFailure != "" {
		return core.Fail(core.E("test.state.Close", state.closeFailure, nil))
	}
	return core.Ok(nil)
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
	if contract := linkedAgentRuntimeContract(context.Background()); !contract.OK {
		t.Skipf("native bootstrap is unavailable with the linked inference module: %s", contract.Error())
	}
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
	startCalls  int
	closeCalls  int
	startResult core.Result
}

func (server *fixtureGitServer) Start(context.Context) core.Result {
	server.startCalls++
	if server.startResult.OK || server.startResult.Value != nil {
		return server.startResult
	}
	return core.Ok(gitserver.Health{Running: true, Address: "127.0.0.1:0"})
}

func (server *fixtureGitServer) EnsureRepository(ctx context.Context, _ string) core.Result {
	if started := server.Start(ctx); !started.OK {
		return started
	}
	return core.Ok(gitserver.Repository{Name: "fixture", CloneURL: "ssh://127.0.0.1/fixture"})
}

func (*fixtureGitServer) Health(context.Context) core.Result {
	return core.Ok(gitserver.Health{Reason: "private Git service is not started"})
}

func (server *fixtureGitServer) Close() core.Result {
	server.closeCalls++
	return core.Ok(nil)
}

type fixtureSnapshotStore struct {
	orchestrator.Store
	result core.Result
}

func (store *fixtureSnapshotStore) Snapshot(string) core.Result { return store.result }

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

type fixtureAgentLauncher struct {
	closeCalls    int
	closeFailures int
}

func (*fixtureAgentLauncher) DetectEnvironment([]string) core.Result { return core.Ok([]string{}) }

func (*fixtureAgentLauncher) Start(context.Context, provider.Command, func(string, string)) core.Result {
	return core.Fail(core.E("test.launcher", "not used", nil))
}

func (launcher *fixtureAgentLauncher) Close() core.Result {
	launcher.closeCalls++
	if launcher.closeFailures > 0 {
		launcher.closeFailures--
		return core.Fail(core.E("test.launcher", "ownership cleanup pending", nil))
	}
	return core.Ok(nil)
}

type fixtureAgentIdentifiers struct{ next int }

func (ids *fixtureAgentIdentifiers) New() string {
	ids.next++
	return core.Sprintf("fixture-%d", ids.next)
}

type failingCloseAgent struct{ order *[]string }

type retryCloseAgent struct {
	calls      int
	repository workspaceRepository
}

func (*retryCloseAgent) Capabilities() []agentCapability      { return nil }
func (*retryCloseAgent) Snapshot(context.Context) core.Result { return core.Ok(agentSnapshot{}) }
func (*retryCloseAgent) Review(context.Context, agentReviewRequest) core.Result {
	return core.Fail(core.E("test.retryAgent", "not used", nil))
}
func (*retryCloseAgent) Run(context.Context, agentRequest) core.Result {
	return core.Fail(core.E("test.retryAgent", "not used", nil))
}
func (agent *retryCloseAgent) Close() core.Result {
	agent.calls++
	if agent.calls == 1 {
		return core.Fail(core.E("test.retryAgent", "ownership cleanup pending", nil))
	}
	if agent.repository != nil {
		if result := agent.repository.ListSessions(false); !result.OK {
			return core.Fail(core.E("test.retryAgent", "repository unavailable during retry", result.Err()))
		}
	}
	return core.Ok(nil)
}

type retryCloseReactiveState struct{ closed bool }

func (*retryCloseReactiveState) Get(string, string) (string, core.Result) {
	return "", core.Ok(nil)
}
func (*retryCloseReactiveState) Set(string, string, string) core.Result { return core.Ok(nil) }
func (*retryCloseReactiveState) Delete(string, string) core.Result      { return core.Ok(nil) }
func (state *retryCloseReactiveState) Close() core.Result {
	state.closed = true
	return core.Ok(nil)
}

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
