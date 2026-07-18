// SPDX-License-Identifier: EUPL-1.2

package orchestrator

import (
	"context"
	"sync"
	"testing"
	"time"

	core "dappco.re/go"
	"dappco.re/go/inference/agent/gitserver"
	"dappco.re/go/inference/agent/provider"
	"dappco.re/go/inference/agent/queue"
	"dappco.re/go/inference/agent/work"
	"dappco.re/go/inference/agent/workspace"
	coreio "dappco.re/go/io"
	coreprocess "dappco.re/go/process"
)

type orchestratorTestGitRunner struct {
	mu        sync.Mutex
	afterPush func(string)
}

func (runner *orchestratorTestGitRunner) Run(ctx context.Context, command workspace.Command) core.Result {
	program := &coreprocess.Program{Name: command.Executable}
	if found := program.Find(); !found.OK {
		return found
	}
	result := program.RunDir(ctx, command.Dir, command.Args...)
	if !result.OK {
		return result
	}
	push := false
	for _, argument := range command.Args {
		if argument == "push" {
			push = true
			break
		}
	}
	if push {
		runner.mu.Lock()
		hook := runner.afterPush
		runner.afterPush = nil
		runner.mu.Unlock()
		if hook != nil {
			hook(command.Dir)
		}
	}
	return result
}

func (runner *orchestratorTestGitRunner) setAfterPush(hook func(string)) {
	runner.mu.Lock()
	runner.afterPush = hook
	runner.mu.Unlock()
}

type orchestratorTestStore struct {
	mu              sync.Mutex
	projects        map[string]work.Project
	runs            map[string]work.Run
	events          []work.Event
	logs            []work.LogChunk
	questions       []work.Question
	answers         []work.Answer
	acceptances     []work.Acceptance
	queue           work.QueueState
	providers       map[string]work.ProviderState
	commits         []Commit
	recoveredAt     time.Time
	recoverFail     bool
	snapshotFail    bool
	projectFail     bool
	beforeCommit    func(Commit)
	failCommitOnce  func(Commit) bool
	failCommit      func(Commit) bool
	projectValue    any
	projectSet      bool
	projectIDValue  any
	projectIDSet    bool
	runValue        any
	runSet          bool
	snapshotValue   any
	snapshotSet     bool
	nextResult      *core.Result
	nextResultAfter int
	continuation    *core.Result
}

func newOrchestratorTestStore(at time.Time) *orchestratorTestStore {
	return &orchestratorTestStore{
		projects: make(map[string]work.Project), runs: make(map[string]work.Run),
		providers: make(map[string]work.ProviderState),
		queue:     work.QueueState{ID: "default", Status: work.QueueAccepting, UpdatedAt: at},
	}
}

func (store *orchestratorTestStore) failNext(predicate func(Commit) bool) {
	store.mu.Lock()
	store.failCommitOnce = predicate
	store.mu.Unlock()
}

func (store *orchestratorTestStore) setRecoverFailure(fail bool) {
	store.mu.Lock()
	store.recoverFail = fail
	store.mu.Unlock()
}

func (store *orchestratorTestStore) setSnapshotFailure(fail bool) {
	store.mu.Lock()
	store.snapshotFail = fail
	store.mu.Unlock()
}

func (store *orchestratorTestStore) setProjectFailure(fail bool) {
	store.mu.Lock()
	store.projectFail = fail
	store.mu.Unlock()
}

func (store *orchestratorTestStore) overrideProject(value any) {
	store.mu.Lock()
	store.projectValue = value
	store.projectSet = true
	store.mu.Unlock()
}

func (store *orchestratorTestStore) clearProjectOverride() {
	store.mu.Lock()
	store.projectSet = false
	store.projectValue = nil
	store.mu.Unlock()
}

func (store *orchestratorTestStore) overrideProjectID(value any) {
	store.mu.Lock()
	store.projectIDValue = value
	store.projectIDSet = true
	store.mu.Unlock()
}

func (store *orchestratorTestStore) clearProjectIDOverride() {
	store.mu.Lock()
	store.projectIDSet = false
	store.projectIDValue = nil
	store.mu.Unlock()
}

func (store *orchestratorTestStore) overrideContinuation(result core.Result) {
	store.mu.Lock()
	store.continuation = &result
	store.mu.Unlock()
}

func (store *orchestratorTestStore) clearContinuationOverride() {
	store.mu.Lock()
	store.continuation = nil
	store.mu.Unlock()
}

func (store *orchestratorTestStore) overrideRun(value any) {
	store.mu.Lock()
	store.runValue = value
	store.runSet = true
	store.mu.Unlock()
}

func (store *orchestratorTestStore) clearRunOverride() {
	store.mu.Lock()
	store.runSet = false
	store.runValue = nil
	store.mu.Unlock()
}

func (store *orchestratorTestStore) overrideSnapshot(value any) {
	store.mu.Lock()
	store.snapshotValue = value
	store.snapshotSet = true
	store.mu.Unlock()
}

func (store *orchestratorTestStore) clearSnapshotOverride() {
	store.mu.Lock()
	store.snapshotSet = false
	store.snapshotValue = nil
	store.mu.Unlock()
}

func (store *orchestratorTestStore) overrideNext(result core.Result) {
	store.mu.Lock()
	store.nextResult = &result
	store.nextResultAfter = 0
	store.mu.Unlock()
}

func (store *orchestratorTestStore) overrideNextAfter(calls int, result core.Result) {
	store.mu.Lock()
	store.nextResult = &result
	store.nextResultAfter = calls
	store.mu.Unlock()
}

func (store *orchestratorTestStore) clearNextOverride() {
	store.mu.Lock()
	store.nextResult = nil
	store.nextResultAfter = 0
	store.mu.Unlock()
}

func (store *orchestratorTestStore) Recover(at time.Time) core.Result {
	store.mu.Lock()
	defer store.mu.Unlock()
	if store.recoverFail {
		return core.Fail(core.NewError("injected recovery failure"))
	}
	store.recoveredAt = at
	count := 0
	for id, run := range store.runs {
		switch run.Status {
		case work.RunQueued, work.RunPreparing, work.RunRunning, work.RunCancelling:
			run.Status = work.RunInterrupted
			run.FinishedAt = at
			run.UpdatedAt = at
			store.runs[id] = run
			count++
		}
	}
	return core.Ok(count)
}

func (store *orchestratorTestStore) Commit(commit Commit) core.Result {
	store.mu.Lock()
	before := store.beforeCommit
	store.mu.Unlock()
	if before != nil {
		before(commit)
	}
	store.mu.Lock()
	defer store.mu.Unlock()
	if store.failCommitOnce != nil && store.failCommitOnce(commit) {
		store.failCommitOnce = nil
		return core.Fail(core.NewError("injected commit failure"))
	}
	if store.failCommit != nil && store.failCommit(commit) {
		return core.Fail(core.NewError("injected persistent commit failure"))
	}
	if commit.Run != nil {
		existing, exists := store.runs[commit.Run.ID]
		if commit.CreateRun {
			if exists {
				return core.Fail(core.NewError("duplicate run"))
			}
		} else if !exists || commit.ExpectedStatus == nil || existing.Status != *commit.ExpectedStatus {
			return core.Fail(core.NewError("stale expected status"))
		}
	}
	if commit.Answer != nil {
		for _, answer := range store.answers {
			if answer.ID == commit.Answer.ID || answer.QuestionID == commit.Answer.QuestionID || answer.ResumeRunID == commit.Answer.ResumeRunID {
				return core.Fail(core.NewError("duplicate answer"))
			}
		}
	}
	if commit.Project != nil {
		store.projects[commit.Project.ID] = *commit.Project
	}
	if commit.Run != nil {
		store.runs[commit.Run.ID] = *commit.Run
	}
	if commit.Event != nil {
		store.events = append(store.events, *commit.Event)
	}
	store.logs = append(store.logs, commit.Logs...)
	if commit.Question != nil {
		store.questions = append(store.questions, *commit.Question)
	}
	if commit.Answer != nil {
		store.answers = append(store.answers, *commit.Answer)
	}
	if commit.Acceptance != nil {
		store.acceptances = append(store.acceptances, *commit.Acceptance)
	}
	if commit.Queue != nil {
		store.queue = *commit.Queue
	}
	if commit.Provider != nil {
		store.providers[commit.Provider.Provider] = *commit.Provider
	}
	store.commits = append(store.commits, commit)
	return core.Ok(nil)
}

func (store *orchestratorTestStore) Project(id string) core.Result {
	store.mu.Lock()
	defer store.mu.Unlock()
	if store.projectIDSet {
		return core.Ok(store.projectIDValue)
	}
	project, exists := store.projects[id]
	if !exists {
		return core.Fail(core.NewError("project not found"))
	}
	return core.Ok(project)
}

func (store *orchestratorTestStore) ProjectBySource(source string) core.Result {
	store.mu.Lock()
	defer store.mu.Unlock()
	if store.projectFail {
		return core.Fail(core.NewError("injected project lookup failure"))
	}
	if store.projectSet {
		return core.Ok(store.projectValue)
	}
	for _, project := range store.projects {
		if project.SourcePath == source {
			return core.Ok(project)
		}
	}
	return core.Ok(nil)
}

func (store *orchestratorTestStore) Run(id string) core.Result {
	store.mu.Lock()
	defer store.mu.Unlock()
	if store.runSet {
		return core.Ok(store.runValue)
	}
	run, exists := store.runs[id]
	if !exists {
		return core.Fail(core.NewError("run not found"))
	}
	return core.Ok(run)
}

func (store *orchestratorTestStore) NextRunNumber(workID string) core.Result {
	store.mu.Lock()
	defer store.mu.Unlock()
	if store.nextResult != nil {
		if store.nextResultAfter > 0 {
			store.nextResultAfter--
		} else {
			return *store.nextResult
		}
	}
	next := 1
	for _, run := range store.runs {
		if run.WorkID == workID && run.Number >= next {
			next = run.Number + 1
		}
	}
	return core.Ok(next)
}

func (store *orchestratorTestStore) Continuation(runID string) core.Result {
	store.mu.Lock()
	defer store.mu.Unlock()
	if store.continuation != nil {
		return *store.continuation
	}
	run, exists := store.runs[runID]
	if !exists {
		return core.Fail(core.NewError("run not found"))
	}
	continuation := work.Continuation{Run: run}
	for _, event := range store.events {
		if event.RunID == runID && event.Kind == "queued" {
			continuation.Task = event.Detail
			break
		}
	}
	for _, log := range store.logs {
		if log.RunID == runID {
			continuation.Logs = append(continuation.Logs, log)
		}
	}
	core.SliceSortFunc(continuation.Logs, func(left, right work.LogChunk) bool {
		return left.Sequence < right.Sequence
	})
	for _, question := range store.questions {
		if question.RunID == runID {
			continuation.Question = question
			break
		}
	}
	for _, answer := range store.answers {
		if answer.QuestionID == continuation.Question.ID {
			continuation.Answer = answer
			break
		}
	}
	return core.Ok(continuation)
}

func (store *orchestratorTestStore) Snapshot(workID string) core.Result {
	store.mu.Lock()
	defer store.mu.Unlock()
	if store.snapshotFail {
		return core.Fail(core.NewError("injected snapshot failure"))
	}
	if store.snapshotSet {
		return core.Ok(store.snapshotValue)
	}
	snapshot := work.Snapshot{Queue: store.queue}
	for _, project := range store.projects {
		snapshot.Projects = append(snapshot.Projects, project)
	}
	for _, run := range store.runs {
		if workID == "" || run.WorkID == workID {
			snapshot.Runs = append(snapshot.Runs, run)
		}
	}
	for _, event := range store.events {
		if workID == "" || event.WorkID == workID {
			snapshot.Events = append(snapshot.Events, event)
		}
	}
	for _, log := range store.logs {
		if workID == "" || store.runs[log.RunID].WorkID == workID {
			snapshot.Logs = append(snapshot.Logs, log)
		}
	}
	for _, question := range store.questions {
		if workID == "" || store.runs[question.RunID].WorkID == workID {
			snapshot.Questions = append(snapshot.Questions, question)
		}
	}
	snapshot.Acceptances = append(snapshot.Acceptances, store.acceptances...)
	for _, state := range store.providers {
		snapshot.Providers = append(snapshot.Providers, state)
	}
	core.SliceSortFunc(snapshot.Runs, func(left, right work.Run) bool { return left.ID < right.ID })
	core.SliceSortFunc(snapshot.Events, func(left, right work.Event) bool {
		return left.CreatedAt.Before(right.CreatedAt) || left.CreatedAt.Equal(right.CreatedAt) && left.ID < right.ID
	})
	core.SliceSortFunc(snapshot.Logs, func(left, right work.LogChunk) bool { return left.Sequence < right.Sequence })
	return core.Ok(snapshot)
}

type orchestratorTestClock struct {
	mu        sync.Mutex
	at        time.Time
	zeroAfter int
}

func (clock *orchestratorTestClock) Now() time.Time {
	clock.mu.Lock()
	defer clock.mu.Unlock()
	if clock.zeroAfter > 0 {
		clock.zeroAfter--
		if clock.zeroAfter == 0 {
			return time.Time{}
		}
	}
	return clock.at
}

func (clock *orchestratorTestClock) ZeroAfter(calls int) {
	clock.mu.Lock()
	clock.zeroAfter = calls
	clock.mu.Unlock()
}

func (clock *orchestratorTestClock) Advance(duration time.Duration) {
	clock.mu.Lock()
	clock.at = clock.at.Add(duration)
	clock.mu.Unlock()
}

func (clock *orchestratorTestClock) Set(at time.Time) {
	clock.mu.Lock()
	clock.at = at
	clock.mu.Unlock()
}

type orchestratorTestIDs struct {
	mu       sync.Mutex
	number   int
	empty    bool
	failNext int
}

func (ids *orchestratorTestIDs) New() string {
	ids.mu.Lock()
	defer ids.mu.Unlock()
	if ids.empty || ids.failNext > 0 {
		if ids.failNext > 0 {
			ids.failNext--
		}
		return ""
	}
	ids.number++
	return core.Sprintf("agent-id-%04d", ids.number)
}

type orchestratorTestGitServer struct {
	mu         sync.Mutex
	root       string
	started    bool
	closed     bool
	failEnsure bool
	failClose  bool
}

func (server *orchestratorTestGitServer) Start(context.Context) core.Result {
	server.mu.Lock()
	server.started = true
	server.mu.Unlock()
	return core.Ok(nil)
}

func (server *orchestratorTestGitServer) EnsureRepository(ctx context.Context, name string) core.Result {
	server.mu.Lock()
	fail := server.failEnsure
	server.mu.Unlock()
	if fail {
		return core.Fail(core.NewError("injected private Git failure"))
	}
	path := core.PathJoin(server.root, core.Concat(name, ".git"))
	if !core.Stat(path).OK {
		if created := core.MkdirAll(server.root, 0o700); !created.OK {
			return created
		}
		result := (&orchestratorTestGitRunner{}).Run(ctx, workspace.Command{Executable: "git", Args: []string{"init", "--bare", "--initial-branch=main", path}})
		if !result.OK {
			return result
		}
	}
	return core.Ok(gitserver.Repository{Name: name, CloneURL: path})
}

func (server *orchestratorTestGitServer) Health(context.Context) core.Result {
	server.mu.Lock()
	defer server.mu.Unlock()
	return core.Ok(gitserver.Health{Running: server.started && !server.closed, Address: server.root})
}

func (server *orchestratorTestGitServer) Close() core.Result {
	server.mu.Lock()
	server.closed = true
	fail := server.failClose
	server.mu.Unlock()
	if fail {
		return core.Fail(core.NewError("injected private Git close failure"))
	}
	return core.Ok(nil)
}

type orchestratorTestAdapter struct {
	mu          sync.Mutex
	name        string
	available   bool
	failBuild   bool
	failDetect  bool
	detectSet   bool
	detectValue any
	buildSet    bool
	buildValue  any
	builds      []provider.Launch
	afterBuild  func()
}

func (adapter *orchestratorTestAdapter) Name() string { return adapter.name }

func (adapter *orchestratorTestAdapter) Detect(context.Context) core.Result {
	adapter.mu.Lock()
	defer adapter.mu.Unlock()
	if adapter.failDetect {
		return core.Fail(core.NewError("injected provider detection failure"))
	}
	if adapter.detectSet {
		return core.Ok(adapter.detectValue)
	}
	return core.Ok(provider.Detection{
		Provider: adapter.name, Executable: core.Concat("/fake/", adapter.name),
		Version: "test-1", Available: adapter.available, Reason: "provider executable is unavailable",
	})
}

func (adapter *orchestratorTestAdapter) Build(launch provider.Launch) core.Result {
	adapter.mu.Lock()
	defer adapter.mu.Unlock()
	if adapter.failBuild {
		return core.Fail(core.NewError("injected provider build failure"))
	}
	if adapter.buildSet {
		return core.Ok(adapter.buildValue)
	}
	adapter.builds = append(adapter.builds, launch)
	if adapter.afterBuild != nil {
		adapter.afterBuild()
	}
	args := []string{"run", launch.WorkID, launch.RunID}
	args = append(args, launch.UnsafeFlags...)
	return core.Ok(provider.Command{
		Provider: adapter.name, Executable: core.Concat("/fake/", adapter.name), Dir: launch.Worktree,
		Args: args, Receipt: core.Concat(adapter.name, " run <redacted>"),
	})
}

func (adapter *orchestratorTestAdapter) ParseLine(stream, line string) []provider.Output {
	if core.HasPrefix(line, "RATE:") {
		return []provider.Output{{Kind: "rate_limit", Text: core.TrimPrefix(line, "RATE:"), RetryAfter: "1h"}}
	}
	if core.HasPrefix(line, "PROGRESS:") {
		return []provider.Output{{Kind: "progress", Text: core.TrimPrefix(line, "PROGRESS:")}}
	}
	if line == "MALFORMED" {
		return []provider.Output{{Kind: "raw", Text: line, DetailJSON: "{"}}
	}
	if line == "" {
		return nil
	}
	return []provider.Output{{Kind: "text", Text: line}}
}

type orchestratorTestProcess struct {
	mu           sync.Mutex
	id           string
	pid          int
	done         chan struct{}
	exitCode     int
	finishOnce   sync.Once
	shutdownOnce sync.Once
	shutdown     bool
	waitFail     bool
	waitValue    any
	shutdownFail bool
}

func newOrchestratorTestProcess(number int) *orchestratorTestProcess {
	return &orchestratorTestProcess{id: core.Sprintf("process-%d", number), pid: 4000 + number, done: make(chan struct{})}
}

func (process *orchestratorTestProcess) ID() string { return process.id }
func (process *orchestratorTestProcess) PID() int   { return process.pid }

func (process *orchestratorTestProcess) Wait() core.Result {
	<-process.done
	process.mu.Lock()
	code := process.exitCode
	process.mu.Unlock()
	if process.waitFail {
		return core.Fail(core.NewError("injected process wait failure"))
	}
	if process.waitValue != nil {
		return core.Ok(process.waitValue)
	}
	return core.Ok(code)
}

func (process *orchestratorTestProcess) Shutdown() core.Result {
	process.mu.Lock()
	fail := process.shutdownFail
	process.mu.Unlock()
	if fail {
		return core.Fail(core.NewError("injected process shutdown failure"))
	}
	process.shutdownOnce.Do(func() {
		process.mu.Lock()
		process.shutdown = true
		process.mu.Unlock()
		process.Finish(-1)
	})
	return core.Ok(nil)
}

func (process *orchestratorTestProcess) Finish(code int) {
	process.finishOnce.Do(func() {
		process.mu.Lock()
		process.exitCode = code
		process.mu.Unlock()
		close(process.done)
	})
}

type orchestratorTestLaunch struct {
	command  provider.Command
	callback func(string, string)
	process  *orchestratorTestProcess
}

type orchestratorTestLauncher struct {
	mu          sync.Mutex
	starts      []*orchestratorTestLaunch
	startSignal chan struct{}
	beforeStart func(provider.Command)
	failStart   bool
	failClose   bool
	startSet    bool
	startValue  any
	configure   func(*orchestratorTestProcess)
	closed      bool
}

func newOrchestratorTestLauncher() *orchestratorTestLauncher {
	return &orchestratorTestLauncher{startSignal: make(chan struct{}, 32)}
}

func (launcher *orchestratorTestLauncher) DetectEnvironment([]string) core.Result {
	return core.Ok([]string{"PATH=/usr/bin"})
}

func (launcher *orchestratorTestLauncher) Start(_ context.Context, command provider.Command, callback func(string, string)) core.Result {
	launcher.mu.Lock()
	if launcher.closed || launcher.failStart {
		launcher.mu.Unlock()
		return core.Fail(core.NewError("injected launcher start failure"))
	}
	before := launcher.beforeStart
	launcher.mu.Unlock()
	if before != nil {
		before(command)
	}
	launcher.mu.Lock()
	if launcher.startSet {
		value := launcher.startValue
		launcher.mu.Unlock()
		return core.Ok(value)
	}
	launcher.mu.Unlock()
	launcher.mu.Lock()
	process := newOrchestratorTestProcess(len(launcher.starts) + 1)
	if launcher.configure != nil {
		launcher.configure(process)
	}
	launch := &orchestratorTestLaunch{command: command, callback: callback, process: process}
	launcher.starts = append(launcher.starts, launch)
	launcher.mu.Unlock()
	launcher.startSignal <- struct{}{}
	return core.Ok(Process(process))
}

func (launcher *orchestratorTestLauncher) Close() core.Result {
	launcher.mu.Lock()
	launcher.closed = true
	fail := launcher.failClose
	launches := append([]*orchestratorTestLaunch(nil), launcher.starts...)
	launcher.mu.Unlock()
	if fail {
		return core.Fail(core.NewError("injected launcher close failure"))
	}
	for _, launch := range launches {
		if shutdown := launch.process.Shutdown(); !shutdown.OK {
			return shutdown
		}
		if waited := launch.process.Wait(); !waited.OK {
			return waited
		}
	}
	return core.Ok(nil)
}

func (launcher *orchestratorTestLauncher) WaitStart(t *testing.T) *orchestratorTestLaunch {
	t.Helper()
	select {
	case <-launcher.startSignal:
	case <-time.After(8 * time.Second):
		t.Fatal("provider did not start")
	}
	launcher.mu.Lock()
	defer launcher.mu.Unlock()
	return launcher.starts[len(launcher.starts)-1]
}

func (launcher *orchestratorTestLauncher) Count() int {
	launcher.mu.Lock()
	defer launcher.mu.Unlock()
	return len(launcher.starts)
}

type orchestratorFixture struct {
	t            *testing.T
	at           time.Time
	store        *orchestratorTestStore
	clock        *orchestratorTestClock
	ids          *orchestratorTestIDs
	server       *orchestratorTestGitServer
	git          *orchestratorTestGitRunner
	manager      *workspace.Manager
	adapter      *orchestratorTestAdapter
	launcher     *orchestratorTestLauncher
	queue        *queue.Controller
	orchestrator *Orchestrator
}

func newOrchestratorFixture(t *testing.T) *orchestratorFixture {
	t.Helper()
	at := time.Date(2026, 7, 18, 12, 0, 0, 0, time.UTC)
	store := newOrchestratorTestStore(at)
	clock := &orchestratorTestClock{at: at}
	ids := &orchestratorTestIDs{}
	server := &orchestratorTestGitServer{root: core.PathJoin(t.TempDir(), "private Git repositories")}
	root := core.PathJoin(t.TempDir(), "LEM internal workspaces")
	core.AssertTrue(t, core.MkdirAll(root, 0o700).OK)
	medium, mediumErr := coreio.NewSandboxed(root)
	if mediumErr != nil {
		t.Fatalf("NewSandboxed failed: %s", mediumErr)
	}
	git := &orchestratorTestGitRunner{}
	managerResult := workspace.NewManager(workspace.ManagerOptions{
		Root: root, Files: medium, Git: git, Server: server,
		IDs: ids.New, Now: clock.Now,
	})
	core.AssertTrue(t, managerResult.OK, managerResult.Error())
	adapter := &orchestratorTestAdapter{name: "fake", available: true}
	registryResult := provider.NewRegistry(adapter)
	core.AssertTrue(t, registryResult.OK, registryResult.Error())
	queueResult := queue.NewController(queue.Policy{
		Version: 1, Dispatch: queue.DispatchConfig{DefaultAgent: "fake", GlobalConcurrency: 1, TimeoutMinutes: 60},
		Concurrency: map[string]queue.ConcurrencyLimit{"fake": {Total: 1, Models: map[string]int{}}},
		Rates:       map[string]queue.RateConfig{}, Providers: map[string]queue.NativeConfig{"fake": {Executable: "fake"}},
	}, store.queue, nil)
	core.AssertTrue(t, queueResult.OK, queueResult.Error())
	launcher := newOrchestratorTestLauncher()
	result := New(Options{
		Store: store, GitServer: server, Workspaces: managerResult.Value.(*workspace.Manager),
		Providers: registryResult.Value.(*provider.Registry), Queue: queueResult.Value.(*queue.Controller),
		Launcher: launcher, Clock: clock, IDs: ids, LogBatchBytes: 32, LogBatchDelay: 20 * time.Millisecond,
	})
	core.AssertTrue(t, result.OK, result.Error())
	fixture := &orchestratorFixture{
		t: t, at: at, store: store, clock: clock, ids: ids, server: server, git: git,
		manager: managerResult.Value.(*workspace.Manager), adapter: adapter, launcher: launcher,
		queue: queueResult.Value.(*queue.Controller), orchestrator: result.Value.(*Orchestrator),
	}
	t.Cleanup(func() {
		if closed := fixture.orchestrator.Close(); !closed.OK {
			t.Logf("orchestrator Close retained an expected test failure: %s", closed.Error())
		}
	})
	return fixture
}

func orchestratorRunGit(t *testing.T, directory string, args ...string) string {
	t.Helper()
	result := (&orchestratorTestGitRunner{}).Run(context.Background(), workspace.Command{Dir: directory, Executable: "git", Args: args})
	if !result.OK {
		t.Fatalf("git %v failed: %s", args, result.Error())
	}
	return core.Trim(result.Value.(string))
}

func orchestratorCreateRepository(t *testing.T, root string) string {
	t.Helper()
	core.AssertTrue(t, core.MkdirAll(root, 0o700).OK)
	orchestratorRunGit(t, root, "init", "--initial-branch=main")
	orchestratorRunGit(t, root, "config", "user.name", "LEM Test")
	orchestratorRunGit(t, root, "config", "user.email", "lem-test@localhost")
	core.AssertTrue(t, core.WriteFile(core.PathJoin(root, "README.md"), []byte("seed\n"), 0o600).OK)
	orchestratorRunGit(t, root, "add", "--all")
	orchestratorRunGit(t, root, "commit", "-m", "seed")
	return orchestratorRunGit(t, root, "rev-parse", "HEAD")
}

func (fixture *orchestratorFixture) registerRepository() (work.Item, work.Project, string) {
	fixture.t.Helper()
	source := core.PathJoin(fixture.t.TempDir(), "source repository")
	revision := orchestratorCreateRepository(fixture.t, source)
	item := work.Item{ID: "work-1", Title: "Improve it", Task: "Make a tested change", Repository: source}
	reviewResult := fixture.orchestrator.ReviewProject(context.Background(), item)
	core.AssertTrue(fixture.t, reviewResult.OK, reviewResult.Error())
	registered := fixture.orchestrator.RegisterProject(context.Background(), reviewResult.Value.(ProjectReview), true)
	core.AssertTrue(fixture.t, registered.OK, registered.Error())
	return item, registered.Value.(work.Project), revision
}

func TestOrchestrator_New_Good(t *testing.T) {
	fixture := newOrchestratorFixture(t)
	core.AssertTrue(t, fixture.orchestrator.Close().OK)
	launcher := newOrchestratorTestLauncher()
	result := New(Options{
		Store: fixture.store, GitServer: fixture.server, Workspaces: fixture.manager,
		Providers: fixture.orchestrator.providers, Queue: fixture.queue, Launcher: launcher,
		Clock: fixture.clock, IDs: fixture.ids,
	})
	core.AssertTrue(t, result.OK, result.Error())
	recovered := result.Value.(*Orchestrator)
	t.Cleanup(func() { core.AssertTrue(t, recovered.Close().OK) })
	core.AssertEqual(t, fixture.at, fixture.store.recoveredAt)
	core.AssertEqual(t, work.QueueFrozen, fixture.store.queue.Status)
}

func TestOrchestrator_New_Bad(t *testing.T) {
	result := New(Options{})
	core.AssertFalse(t, result.OK)
	core.AssertContains(t, result.Error(), "requires store")
}

func TestOrchestrator_New_Ugly(t *testing.T) {
	fixture := newOrchestratorFixture(t)
	options := Options{
		Store: fixture.store, GitServer: fixture.server, Workspaces: fixture.manager,
		Providers: fixture.orchestrator.providers, Queue: fixture.queue, Launcher: fixture.launcher,
		Clock: fixture.clock, IDs: fixture.ids, LogBatchBytes: -1,
	}
	core.AssertFalse(t, New(options).OK)
	fixture.store.setRecoverFailure(true)
	options.LogBatchBytes = 1
	core.AssertFalse(t, New(options).OK)
}

func TestOrchestratorRecoveryInterruptsUnsafeRuns(t *testing.T) {
	fixture := newOrchestratorFixture(t)
	core.AssertTrue(t, fixture.orchestrator.Close().OK)
	fixture.store.mu.Lock()
	for index, status := range []work.RunStatus{work.RunQueued, work.RunPreparing, work.RunRunning, work.RunCancelling} {
		run := storeTestRun(status)
		run.ID = core.Sprintf("recovery-%d", index)
		fixture.store.runs[run.ID] = run
	}
	fixture.store.mu.Unlock()
	launcher := newOrchestratorTestLauncher()
	result := New(Options{
		Store: fixture.store, GitServer: fixture.server, Workspaces: fixture.manager,
		Providers: fixture.orchestrator.providers, Queue: fixture.queue, Launcher: launcher,
		Clock: fixture.clock, IDs: fixture.ids,
	})
	core.AssertTrue(t, result.OK, result.Error())
	recovered := result.Value.(*Orchestrator)
	for index := 0; index < 4; index++ {
		run := fixture.store.Run(core.Sprintf("recovery-%d", index))
		core.AssertTrue(t, run.OK, run.Error())
		core.AssertEqual(t, work.RunInterrupted, run.Value.(work.Run).Status)
	}
	core.AssertEqual(t, work.QueueFrozen, fixture.store.queue.Status)
	core.AssertEqual(t, 0, launcher.Count())
	core.AssertTrue(t, recovered.Close().OK)
}

func TestOrchestrator_Orchestrator_Capabilities_Good(t *testing.T) {
	fixture := newOrchestratorFixture(t)
	capabilities := fixture.orchestrator.Capabilities()
	core.AssertEqual(t, 10, len(capabilities))
	core.AssertEqual(t, work.Capability{Name: "dispatch", Available: true}, capabilities[0])
	core.AssertEqual(t, work.Capability{Name: "queue.start", Available: true}, capabilities[2])
	core.AssertEqual(t, work.Capability{Name: "queue.stop", Available: true}, capabilities[3])
	core.AssertEqual(t, work.Capability{Name: "answer", Available: true}, capabilities[4])
	core.AssertEqual(t, work.Capability{Name: "retry", Available: true}, capabilities[5])
	core.AssertEqual(t, work.Capability{Name: "resume", Available: true}, capabilities[6])
}

func TestOrchestrator_Orchestrator_Capabilities_Bad(t *testing.T) {
	var orchestrator *Orchestrator
	core.AssertEqual(t, 10, len(orchestrator.Capabilities()))
	core.AssertFalse(t, orchestrator.Capabilities()[0].Available)
}

func TestOrchestrator_Orchestrator_Capabilities_Ugly(t *testing.T) {
	fixture := newOrchestratorFixture(t)
	core.AssertTrue(t, fixture.orchestrator.Close().OK)
	core.AssertFalse(t, fixture.orchestrator.Capabilities()[0].Available)
}

func TestOrchestrator_Orchestrator_Snapshot_Good(t *testing.T) {
	fixture := newOrchestratorFixture(t)
	item, project, _ := fixture.registerRepository()
	result := fixture.orchestrator.Snapshot(context.Background(), item.ID)
	core.AssertTrue(t, result.OK, result.Error())
	snapshot := result.Value.(work.Snapshot)
	core.AssertEqual(t, project.ID, snapshot.Projects[0].ID)
	snapshot.Projects[0].ID = "mutated"
	core.AssertEqual(t, project.ID, fixture.store.projects[project.ID].ID)
}

func TestOrchestrator_Orchestrator_Snapshot_Bad(t *testing.T) {
	fixture := newOrchestratorFixture(t)
	core.AssertFalse(t, fixture.orchestrator.Snapshot(nil, "").OK)
	fixture.store.setSnapshotFailure(true)
	core.AssertFalse(t, fixture.orchestrator.Snapshot(context.Background(), "").OK)
}

func TestOrchestrator_Orchestrator_Snapshot_Ugly(t *testing.T) {
	var orchestrator *Orchestrator
	core.AssertFalse(t, orchestrator.Snapshot(context.Background(), "").OK)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	fixture := newOrchestratorFixture(t)
	core.AssertFalse(t, fixture.orchestrator.Snapshot(ctx, "").OK)
}

func TestOrchestrator_Orchestrator_ReviewProject_Good(t *testing.T) {
	fixture := newOrchestratorFixture(t)
	source := core.PathJoin(t.TempDir(), "review source")
	revision := orchestratorCreateRepository(t, source)
	item := work.Item{ID: "work-review", Repository: source}
	headBefore := orchestratorRunGit(t, source, "rev-parse", "HEAD")
	statusBefore := orchestratorRunGit(t, source, "status", "--porcelain=v1")
	result := fixture.orchestrator.ReviewProject(context.Background(), item)
	core.AssertTrue(t, result.OK, result.Error())
	review := result.Value.(ProjectReview)
	core.AssertEqual(t, revision, review.Source.Revision)
	core.AssertFalse(t, review.RequiresGitEnable)
	core.AssertEqual(t, headBefore, orchestratorRunGit(t, source, "rev-parse", "HEAD"))
	core.AssertEqual(t, statusBefore, orchestratorRunGit(t, source, "status", "--porcelain=v1"))
}

func TestOrchestrator_Orchestrator_ReviewProject_Bad(t *testing.T) {
	fixture := newOrchestratorFixture(t)
	core.AssertFalse(t, fixture.orchestrator.ReviewProject(nil, work.Item{}).OK)
	core.AssertFalse(t, fixture.orchestrator.ReviewProject(context.Background(), work.Item{}).OK)
}

func TestOrchestrator_Orchestrator_ReviewProject_Ugly(t *testing.T) {
	fixture := newOrchestratorFixture(t)
	fixture.store.setProjectFailure(true)
	source := core.PathJoin(t.TempDir(), "lookup source")
	orchestratorCreateRepository(t, source)
	core.AssertFalse(t, fixture.orchestrator.ReviewProject(context.Background(), work.Item{ID: "work", Repository: source}).OK)
}

func TestOrchestrator_Orchestrator_RegisterProject_Good(t *testing.T) {
	fixture := newOrchestratorFixture(t)
	source := core.PathJoin(t.TempDir(), "registered source")
	revision := orchestratorCreateRepository(t, source)
	item := work.Item{ID: "registered-work", Repository: source}
	reviewed := fixture.orchestrator.ReviewProject(context.Background(), item)
	core.AssertTrue(t, reviewed.OK, reviewed.Error())
	registered := fixture.orchestrator.RegisterProject(context.Background(), reviewed.Value.(ProjectReview), true)
	core.AssertTrue(t, registered.OK, registered.Error())
	project := registered.Value.(work.Project)
	core.AssertEqual(t, revision, project.SourceRevision)
	canonical := core.PathEvalSymlinks(item.Repository)
	core.AssertTrue(t, canonical.OK, canonical.Error())
	core.AssertEqual(t, canonical.Value.(string), project.SourcePath)
	core.AssertTrue(t, core.Stat(project.ClonePath).OK)
	reviewedAgain := fixture.orchestrator.ReviewProject(context.Background(), item)
	core.AssertTrue(t, reviewedAgain.OK, reviewedAgain.Error())
	core.AssertEqual(t, project.RepositoryName, reviewedAgain.Value.(ProjectReview).RepositoryName)

	shared := item
	shared.ID = "shared-source-work"
	sharedReview := fixture.orchestrator.ReviewProject(context.Background(), shared)
	core.AssertTrue(t, sharedReview.OK, sharedReview.Error())
	fixture.server.mu.Lock()
	fixture.server.failEnsure = true
	fixture.server.mu.Unlock()
	reused := fixture.orchestrator.RegisterProject(context.Background(), sharedReview.Value.(ProjectReview), true)
	if !reused.OK {
		t.Fatalf("RegisterProject shared source failed: %s", reused.Error())
	}
	core.AssertEqual(t, project.ID, reused.Value.(work.Project).ID)
	fixture.server.mu.Lock()
	fixture.server.failEnsure = false
	fixture.server.mu.Unlock()

	core.AssertTrue(t, core.WriteFile(core.PathJoin(source, "next.txt"), []byte("next\n"), 0o600).OK)
	orchestratorRunGit(t, source, "add", "--all")
	orchestratorRunGit(t, source, "commit", "-m", "next")
	fixture.clock.Advance(time.Hour)
	refreshedReview := fixture.orchestrator.ReviewProject(context.Background(), shared)
	core.AssertTrue(t, refreshedReview.OK, refreshedReview.Error())
	refreshed := fixture.orchestrator.RegisterProject(context.Background(), refreshedReview.Value.(ProjectReview), true)
	if !refreshed.OK {
		t.Fatalf("RegisterProject source refresh failed: %s", refreshed.Error())
	}
	core.AssertEqual(t, project.ID, refreshed.Value.(work.Project).ID)
	core.AssertEqual(t, project.CreatedAt, refreshed.Value.(work.Project).CreatedAt)
	core.AssertTrue(t, refreshed.Value.(work.Project).UpdatedAt.After(project.UpdatedAt))
	core.AssertEqual(t, orchestratorRunGit(t, source, "rev-parse", "HEAD"), refreshed.Value.(work.Project).SourceRevision)
	fixture.store.mu.Lock()
	projectCount := len(fixture.store.projects)
	fixture.store.mu.Unlock()
	core.AssertEqual(t, 1, projectCount)
}

func TestOrchestrator_Orchestrator_RegisterProject_Bad(t *testing.T) {
	fixture := newOrchestratorFixture(t)
	core.AssertFalse(t, fixture.orchestrator.RegisterProject(nil, ProjectReview{}, false).OK)
	core.AssertFalse(t, fixture.orchestrator.RegisterProject(context.Background(), ProjectReview{}, false).OK)
	core.AssertFalse(t, fixture.orchestrator.RegisterProject(context.Background(), ProjectReview{}, true).OK)
}

func TestOrchestrator_Orchestrator_RegisterProject_Ugly(t *testing.T) {
	fixture := newOrchestratorFixture(t)
	source := core.PathJoin(t.TempDir(), "ad hoc source")
	core.AssertTrue(t, core.MkdirAll(source, 0o700).OK)
	core.AssertTrue(t, core.WriteFile(core.PathJoin(source, "notes.txt"), []byte("notes\n"), 0o600).OK)
	item := work.Item{ID: "ad-hoc", Repository: source}
	reviewResult := fixture.orchestrator.ReviewProject(context.Background(), item)
	core.AssertTrue(t, reviewResult.OK, reviewResult.Error())
	review := reviewResult.Value.(ProjectReview)
	core.AssertTrue(t, review.RequiresGitEnable)
	core.AssertFalse(t, fixture.orchestrator.RegisterProject(context.Background(), review, false).OK)
	core.AssertFalse(t, core.Stat(core.PathJoin(source, ".git")).OK)
	core.AssertTrue(t, core.WriteFile(core.PathJoin(source, "moved.txt"), []byte("changed\n"), 0o600).OK)
	core.AssertFalse(t, fixture.orchestrator.RegisterProject(context.Background(), review, true).OK)
	core.AssertFalse(t, core.Stat(core.PathJoin(source, ".git")).OK)
	refreshed := fixture.orchestrator.ReviewProject(context.Background(), item)
	core.AssertTrue(t, refreshed.OK, refreshed.Error())
	registered := fixture.orchestrator.RegisterProject(context.Background(), refreshed.Value.(ProjectReview), true)
	core.AssertTrue(t, registered.OK, registered.Error())
	core.AssertTrue(t, core.Stat(core.PathJoin(source, ".git")).OK)
	core.AssertFalse(t, core.Stat(core.PathJoin(source, ".lem")).OK)
}

func TestOrchestrator_Orchestrator_Close_Good(t *testing.T) {
	fixture := newOrchestratorFixture(t)
	core.AssertTrue(t, fixture.orchestrator.Close().OK)
	core.AssertTrue(t, fixture.server.closed)
	core.AssertTrue(t, fixture.orchestrator.Close().OK)
}

func TestOrchestrator_Orchestrator_Close_Bad(t *testing.T) {
	var orchestrator *Orchestrator
	result := orchestrator.Close()
	core.AssertFalse(t, result.OK)
	core.AssertContains(t, result.Error(), "orchestrator is required")
}

func TestOrchestrator_Orchestrator_Close_Ugly(t *testing.T) {
	fixture := newOrchestratorFixture(t)
	fixture.store.setSnapshotFailure(true)
	core.AssertFalse(t, fixture.orchestrator.Close().OK)
	core.AssertTrue(t, fixture.server.closed)
}

func TestOrchestratorBoundaryFailures(t *testing.T) {
	fixture := newOrchestratorFixture(t)
	fixture.store.overrideSnapshot("wrong snapshot")
	core.AssertFalse(t, fixture.orchestrator.Snapshot(context.Background(), "").OK)
	fixture.store.clearSnapshotOverride()

	item, _, _ := fixture.registerRepository()
	fixture.store.overrideProject("wrong project")
	core.AssertFalse(t, fixture.orchestrator.ReviewProject(context.Background(), item).OK)
	fixture.store.clearProjectOverride()

	clockTime := fixture.clock.Now()
	fixture.clock.Set(time.Time{})
	core.AssertFalse(t, fixture.orchestrator.now().OK)
	fixture.clock.Set(clockTime)
	fixture.ids.mu.Lock()
	fixture.ids.empty = true
	fixture.ids.mu.Unlock()
	core.AssertFalse(t, fixture.orchestrator.nextID("test").OK)
	fixture.ids.mu.Lock()
	fixture.ids.empty = false
	fixture.ids.mu.Unlock()
	core.AssertTrue(t, (&Orchestrator{}).isClosed() == false)
	var missing *Orchestrator
	core.AssertTrue(t, missing.isClosed())
}

func TestOrchestratorConstructionAndProjectFailureBoundaries(t *testing.T) {
	t.Run("zero recovery clock", func(t *testing.T) {
		fixture := newOrchestratorFixture(t)
		core.AssertTrue(t, fixture.orchestrator.Close().OK)
		fixture.clock.Set(time.Time{})
		result := New(Options{
			Store: fixture.store, GitServer: fixture.server, Workspaces: fixture.manager,
			Providers: fixture.orchestrator.providers, Queue: fixture.queue, Launcher: newOrchestratorTestLauncher(),
			Clock: fixture.clock, IDs: fixture.ids,
		})
		core.AssertFalse(t, result.OK)
	})

	t.Run("recovered queue persistence", func(t *testing.T) {
		fixture := newOrchestratorFixture(t)
		core.AssertTrue(t, fixture.orchestrator.Close().OK)
		fixture.store.failNext(func(commit Commit) bool { return commit.Queue != nil })
		result := New(Options{
			Store: fixture.store, GitServer: fixture.server, Workspaces: fixture.manager,
			Providers: fixture.orchestrator.providers, Queue: fixture.queue, Launcher: newOrchestratorTestLauncher(),
			Clock: fixture.clock, IDs: fixture.ids,
		})
		core.AssertFalse(t, result.OK)
		core.AssertContains(t, result.Error(), "persist recovered queue")
	})

	fixture := newOrchestratorFixture(t)
	missingSource := work.Item{ID: "missing-source", Repository: core.PathJoin(t.TempDir(), "does-not-exist")}
	core.AssertFalse(t, fixture.orchestrator.ReviewProject(context.Background(), missingSource).OK)
	source := core.PathJoin(t.TempDir(), "project commit failure")
	orchestratorCreateRepository(t, source)
	item := work.Item{ID: "project-commit-failure", Repository: source}
	reviewed := fixture.orchestrator.ReviewProject(context.Background(), item)
	core.AssertTrue(t, reviewed.OK, reviewed.Error())
	fixture.store.failNext(func(commit Commit) bool { return commit.Project != nil })
	core.AssertFalse(t, fixture.orchestrator.RegisterProject(context.Background(), reviewed.Value.(ProjectReview), true).OK)
	core.AssertTrue(t, fixture.orchestrator.Close().OK)
	core.AssertFalse(t, fixture.orchestrator.ReviewProject(context.Background(), item).OK)
	core.AssertFalse(t, fixture.orchestrator.RegisterProject(context.Background(), reviewed.Value.(ProjectReview), true).OK)
}

func TestOrchestratorWithdrawQueuedFailureBoundaries(t *testing.T) {
	fixture := newOrchestratorFixture(t)
	item, _, revision := fixture.registerRepository()
	run := fixture.queueDispatch(item, revision)

	fixture.store.overrideSnapshot("not a snapshot")
	core.AssertFalse(t, fixture.orchestrator.withdrawQueued().OK)
	fixture.store.clearSnapshotOverride()
	at := fixture.clock.Now()
	fixture.clock.Set(time.Time{})
	core.AssertFalse(t, fixture.orchestrator.withdrawQueued().OK)
	fixture.clock.Set(at)
	fixture.ids.mu.Lock()
	fixture.ids.failNext = 1
	fixture.ids.mu.Unlock()
	core.AssertFalse(t, fixture.orchestrator.withdrawQueued().OK)
	fixture.store.failNext(func(commit Commit) bool {
		return commit.Run != nil && commit.Run.ID == run.ID
	})
	core.AssertFalse(t, fixture.orchestrator.withdrawQueued().OK)
	core.AssertTrue(t, fixture.orchestrator.withdrawQueued().OK)
	stored := fixture.store.Run(run.ID)
	core.AssertTrue(t, stored.OK, stored.Error())
	core.AssertEqual(t, work.RunCancelled, stored.Value.(work.Run).Status)
}

func TestOrchestratorClosePersistsQueueFailure(t *testing.T) {
	fixture := newOrchestratorFixture(t)
	fixture.store.failNext(func(commit Commit) bool { return commit.Queue != nil })
	closed := fixture.orchestrator.Close()
	core.AssertFalse(t, closed.OK)
	core.AssertContains(t, closed.Error(), "injected commit failure")
}

func TestOrchestratorCloseWaitsForAdmittedDispatch(t *testing.T) {
	fixture := newOrchestratorFixture(t)
	item, _, revision := fixture.registerRepository()
	review := fixture.reviewDispatch(item, revision)
	commitReached := make(chan struct{}, 1)
	releaseCommit := make(chan struct{})
	fixture.store.mu.Lock()
	fixture.store.beforeCommit = func(commit Commit) {
		if commit.CreateRun {
			commitReached <- struct{}{}
			<-releaseCommit
		}
	}
	fixture.store.mu.Unlock()
	dispatched := make(chan core.Result, 1)
	go func() { dispatched <- fixture.orchestrator.Dispatch(context.Background(), review) }()
	select {
	case <-commitReached:
	case <-time.After(5 * time.Second):
		t.Fatal("dispatch did not reach its durable commit")
	}
	closed := make(chan core.Result, 1)
	go func() { closed <- fixture.orchestrator.Close() }()
	select {
	case result := <-closed:
		t.Fatalf("Close returned before admitted dispatch completed: %s", result.Error())
	case <-time.After(100 * time.Millisecond):
	}
	close(releaseCommit)
	dispatchResult := <-dispatched
	core.AssertTrue(t, dispatchResult.OK, dispatchResult.Error())
	closeResult := <-closed
	core.AssertTrue(t, closeResult.OK, closeResult.Error())
	run := dispatchResult.Value.(work.Run)
	stored := fixture.store.Run(run.ID)
	core.AssertTrue(t, stored.OK, stored.Error())
	core.AssertEqual(t, work.RunCancelled, stored.Value.(work.Run).Status)
}

func TestOrchestratorCloseWaitsForAdmittedQueueStart(t *testing.T) {
	fixture := newOrchestratorFixture(t)
	commitReached := make(chan struct{}, 1)
	releaseCommit := make(chan struct{})
	fixture.store.mu.Lock()
	fixture.store.beforeCommit = func(commit Commit) {
		if commit.Queue != nil && commit.Queue.Status == work.QueueAccepting {
			commitReached <- struct{}{}
			<-releaseCommit
		}
	}
	fixture.store.mu.Unlock()
	started := make(chan core.Result, 1)
	go func() { started <- fixture.orchestrator.StartQueue(context.Background()) }()
	select {
	case <-commitReached:
	case <-time.After(5 * time.Second):
		t.Fatal("queue start did not reach its durable commit")
	}
	closed := make(chan core.Result, 1)
	go func() { closed <- fixture.orchestrator.Close() }()
	select {
	case result := <-closed:
		t.Fatalf("Close returned before admitted queue start completed: %s", result.Error())
	case <-time.After(100 * time.Millisecond):
	}
	close(releaseCommit)
	startResult := <-started
	core.AssertTrue(t, startResult.OK, startResult.Error())
	closeResult := <-closed
	core.AssertTrue(t, closeResult.OK, closeResult.Error())
	fixture.store.mu.Lock()
	status := fixture.store.queue.Status
	fixture.store.mu.Unlock()
	core.AssertEqual(t, work.QueueFrozen, status)
}

func TestOrchestratorCloseWithdrawsQueuedAndPreservesFailures(t *testing.T) {
	fixture := newOrchestratorFixture(t)
	item, _, revision := fixture.registerRepository()
	run := fixture.queueDispatch(item, revision)
	fixture.launcher.mu.Lock()
	fixture.launcher.failClose = true
	fixture.launcher.mu.Unlock()
	fixture.server.mu.Lock()
	fixture.server.failClose = true
	fixture.server.mu.Unlock()
	closed := fixture.orchestrator.Close()
	core.AssertFalse(t, closed.OK)
	core.AssertContains(t, closed.Error(), "launcher close")
	core.AssertContains(t, closed.Error(), "Git close")
	stored := fixture.store.Run(run.ID)
	core.AssertTrue(t, stored.OK, stored.Error())
	core.AssertEqual(t, work.RunCancelled, stored.Value.(work.Run).Status)
}

func TestOrchestratorCloseCollectsProcessAndClockFailures(t *testing.T) {
	fixture := newOrchestratorFixture(t)
	process := newOrchestratorTestProcess(99)
	process.shutdownFail = true
	fixture.orchestrator.mu.Lock()
	fixture.orchestrator.runs["broken"] = process
	fixture.orchestrator.mu.Unlock()
	closed := fixture.orchestrator.Close()
	core.AssertFalse(t, closed.OK)
	core.AssertContains(t, closed.Error(), "shutdown failure")

	zeroFixture := newOrchestratorFixture(t)
	zeroFixture.clock.Set(time.Time{})
	zeroClosed := zeroFixture.orchestrator.Close()
	core.AssertFalse(t, zeroClosed.OK)
	core.AssertContains(t, zeroClosed.Error(), "clock returned zero")
}
