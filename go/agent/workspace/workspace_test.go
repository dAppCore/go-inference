// SPDX-License-Identifier: EUPL-1.2

package workspace

import (
	"context"
	"sync"
	"testing"
	"time"

	core "dappco.re/go"
	"dappco.re/go/inference/agent/gitserver"
	"dappco.re/go/inference/agent/work"
	coreio "dappco.re/go/io"
	commandexec "dappco.re/go/process/exec"
)

const (
	workspaceTestName  = "Workspace Test"
	workspaceTestEmail = "workspace@example.invalid"
)

type workspaceTestRunner struct {
	mu        sync.Mutex
	commands  []Command
	failPush  bool
	failWhen  func(Command) bool
	valueWhen func(Command) (any, bool)
}

type workspaceRunOnly struct {
	result core.Result
}

func (runner workspaceRunOnly) Run(context.Context, Command) core.Result {
	return runner.result
}

type workspaceDetailedProbeRunner struct {
	result core.Result
}

func (runner workspaceDetailedProbeRunner) Run(context.Context, Command) core.Result {
	return runner.result
}

func (runner workspaceDetailedProbeRunner) RunDetailed(context.Context, Command) core.Result {
	return runner.result
}

func (runner *workspaceTestRunner) Run(ctx context.Context, command Command) core.Result {
	runner.mu.Lock()
	copyCommand := Command{
		Dir:         command.Dir,
		Executable:  command.Executable,
		Args:        append([]string(nil), command.Args...),
		Environment: append([]string(nil), command.Environment...),
	}
	runner.commands = append(runner.commands, copyCommand)
	failPush := runner.failPush && workspaceContainsArgument(command.Args, "push")
	failWhen := runner.failWhen
	valueWhen := runner.valueWhen
	runner.mu.Unlock()
	if failPush || failWhen != nil && failWhen(copyCommand) {
		return core.Fail(core.NewError("injected push failure"))
	}
	if valueWhen != nil {
		if value, matched := valueWhen(copyCommand); matched {
			return core.Ok(value)
		}
	}
	environment := append([]string{
		"GIT_CONFIG_GLOBAL=/dev/null",
		"GIT_CONFIG_NOSYSTEM=1",
		"LC_ALL=C",
	}, command.Environment...)
	result := commandexec.Command(ctx, command.Executable, command.Args...).
		WithDir(command.Dir).
		WithEnv(environment).
		CombinedOutput()
	if !result.OK {
		return result
	}
	return core.Ok(string(result.Value.([]byte)))
}

func (runner *workspaceTestRunner) RunDetailed(ctx context.Context, command Command) core.Result {
	runner.mu.Lock()
	copyCommand := Command{
		Dir:         command.Dir,
		Executable:  command.Executable,
		Args:        append([]string(nil), command.Args...),
		Environment: append([]string(nil), command.Environment...),
	}
	runner.commands = append(runner.commands, copyCommand)
	failWhen := runner.failWhen
	valueWhen := runner.valueWhen
	runner.mu.Unlock()
	if failWhen != nil && failWhen(copyCommand) {
		return core.Ok(CommandOutcome{ExitCode: -1, Failure: core.NewError("injected Git root probe failure")})
	}
	if valueWhen != nil {
		if value, matched := valueWhen(copyCommand); matched {
			text, ok := value.(string)
			if !ok {
				return core.Ok(CommandOutcome{ExitCode: 0, Output: core.Sprintf("%v", value)})
			}
			return core.Ok(CommandOutcome{ExitCode: 0, Output: text})
		}
	}
	copyCommand.Environment = append([]string{"GIT_CONFIG_GLOBAL=/dev/null", "GIT_CONFIG_NOSYSTEM=1", "LC_ALL=C"}, copyCommand.Environment...)
	return (ProcessRunner{}).RunDetailed(ctx, copyCommand)
}

func (runner *workspaceTestRunner) setFailPush(fail bool) {
	runner.mu.Lock()
	runner.failPush = fail
	runner.mu.Unlock()
}

func (runner *workspaceTestRunner) setFailure(predicate func(Command) bool) {
	runner.mu.Lock()
	runner.failWhen = predicate
	runner.mu.Unlock()
}

func (runner *workspaceTestRunner) setValue(provider func(Command) (any, bool)) {
	runner.mu.Lock()
	runner.valueWhen = provider
	runner.mu.Unlock()
}

type workspaceFaultMedium struct {
	coreio.Medium
	failEnsure string
	failDelete string
}

func (medium *workspaceFaultMedium) EnsureDir(path string) error {
	if path == medium.failEnsure {
		return core.NewError("injected ensure directory failure")
	}
	return medium.Medium.EnsureDir(path)
}

func (medium *workspaceFaultMedium) DeleteAll(path string) error {
	if path == medium.failDelete {
		return core.NewError("injected delete directory failure")
	}
	return medium.Medium.DeleteAll(path)
}

func workspaceContainsArgument(arguments []string, expected string) bool {
	for _, argument := range arguments {
		if argument == expected {
			return true
		}
	}
	return false
}

func workspaceCommandContains(command Command, fragment string) bool {
	return core.Contains(core.Join("\x00", command.Args...), fragment)
}

func workspaceLastPushArguments(t *testing.T, runner *workspaceTestRunner) []string {
	t.Helper()
	runner.mu.Lock()
	defer runner.mu.Unlock()
	for index := len(runner.commands) - 1; index >= 0; index-- {
		if len(runner.commands[index].Args) > 0 && runner.commands[index].Args[0] == "push" {
			return append([]string(nil), runner.commands[index].Args...)
		}
	}
	t.Fatal("no git push command was recorded")
	return nil
}

type workspaceTestServer struct {
	root        string
	runner      Runner
	mu          sync.Mutex
	starts      int
	closed      bool
	failStart   bool
	failEnsure  bool
	ensureValue any
	repository  *gitserver.Repository
}

func (server *workspaceTestServer) Start(ctx context.Context) core.Result {
	if ctx == nil {
		return core.Fail(core.NewError("context required"))
	}
	server.mu.Lock()
	if server.failStart {
		server.mu.Unlock()
		return core.Fail(core.NewError("injected server start failure"))
	}
	server.starts++
	server.mu.Unlock()
	return core.Ok(gitserver.Health{Running: true, Address: "local-test"})
}

func (server *workspaceTestServer) EnsureRepository(ctx context.Context, name string) core.Result {
	server.mu.Lock()
	failEnsure := server.failEnsure
	ensureValue := server.ensureValue
	repository := server.repository
	server.mu.Unlock()
	if failEnsure {
		return core.Fail(core.NewError("injected ensure repository failure"))
	}
	if ensureValue != nil {
		return core.Ok(ensureValue)
	}
	if repository != nil {
		copyRepository := *repository
		return core.Ok(copyRepository)
	}
	if started := server.Start(ctx); !started.OK {
		return started
	}
	path := core.PathJoin(server.root, core.Concat(name, ".git"))
	if !core.Stat(path).OK {
		created := server.runner.Run(ctx, Command{
			Dir:        server.root,
			Executable: "git",
			Args:       []string{"init", "--bare", "--initial-branch=main", path},
		})
		if !created.OK {
			return created
		}
	}
	return core.Ok(gitserver.Repository{Name: name, CloneURL: path})
}

func (server *workspaceTestServer) setFailure(start, ensure bool) {
	server.mu.Lock()
	server.failStart = start
	server.failEnsure = ensure
	server.mu.Unlock()
}

func (server *workspaceTestServer) setEnsureValue(value any) {
	server.mu.Lock()
	server.ensureValue = value
	server.mu.Unlock()
}

func (server *workspaceTestServer) setRepository(repository gitserver.Repository) {
	server.mu.Lock()
	server.repository = &repository
	server.mu.Unlock()
}

func (server *workspaceTestServer) Health(context.Context) core.Result {
	server.mu.Lock()
	defer server.mu.Unlock()
	return core.Ok(gitserver.Health{Running: !server.closed, Address: "local-test"})
}

func (server *workspaceTestServer) Close() core.Result {
	server.mu.Lock()
	server.closed = true
	server.mu.Unlock()
	return core.Ok(nil)
}

type workspaceFixture struct {
	manager *Manager
	runner  *workspaceTestRunner
	server  *workspaceTestServer
	root    string
	files   coreio.Medium
}

type workspaceRunFixture struct {
	workspaceFixture
	project  work.Project
	run      work.Run
	prepared RunWorkspace
}

func workspaceNewFixture(t *testing.T) workspaceFixture {
	t.Helper()
	root := core.PathJoin(t.TempDir(), "LEM workspaces with spaces")
	core.AssertTrue(t, core.MkdirAll(root, 0o700).OK)
	files, filesErr := coreio.NewSandboxed(root)
	if filesErr != nil {
		t.Fatalf("NewSandboxed failed: %s", filesErr)
	}
	runner := &workspaceTestRunner{}
	serverRoot := core.PathJoin(t.TempDir(), "private repos")
	core.AssertTrue(t, core.MkdirAll(serverRoot, 0o700).OK)
	server := &workspaceTestServer{root: serverRoot, runner: runner}
	result := NewManager(ManagerOptions{
		Root:   root,
		Files:  files,
		Git:    runner,
		Server: server,
		IDs:    func() string { return "review-id" },
		Now:    func() time.Time { return time.Date(2026, 7, 18, 12, 0, 0, 0, time.UTC) },
	})
	if !result.OK {
		t.Fatalf("NewManager failed: %s", result.Error())
	}
	manager := result.Value.(*Manager)
	return workspaceFixture{
		manager: manager,
		runner:  runner,
		server:  server,
		root:    manager.root,
		files:   files,
	}
}

func workspaceRunGit(t *testing.T, runner Runner, directory string, arguments ...string) string {
	t.Helper()
	result := runner.Run(context.Background(), Command{Dir: directory, Executable: "git", Args: arguments})
	if !result.OK {
		t.Fatalf("git %v failed: %s", arguments, result.Error())
	}
	return core.Trim(result.Value.(string))
}

func workspaceWriteFile(t *testing.T, path, content string) {
	t.Helper()
	core.AssertTrue(t, core.MkdirAll(core.PathDir(path), 0o700).OK)
	result := core.WriteFile(path, []byte(content), 0o600)
	core.AssertTrue(t, result.OK, result.Error())
}

func workspaceCanonical(t *testing.T, path string) string {
	t.Helper()
	result := canonicalDirectory(path)
	if !result.OK {
		t.Fatalf("canonicalDirectory failed: %s", result.Error())
	}
	return result.Value.(string)
}

func workspaceCreateRepository(t *testing.T, runner Runner, path string) string {
	t.Helper()
	core.AssertTrue(t, core.MkdirAll(path, 0o700).OK)
	workspaceRunGit(t, runner, path, "init", "--initial-branch=main")
	workspaceRunGit(t, runner, path, "config", "user.name", workspaceTestName)
	workspaceRunGit(t, runner, path, "config", "user.email", workspaceTestEmail)
	workspaceWriteFile(t, core.PathJoin(path, ".gitignore"), "*.ignored\n")
	workspaceWriteFile(t, core.PathJoin(path, "README.md"), "seed\n")
	workspaceWriteFile(t, core.PathJoin(path, "nested", "tracked.txt"), "nested\n")
	workspaceRunGit(t, runner, path, "add", "--all")
	workspaceRunGit(t, runner, path, "commit", "-m", "initial")
	return workspaceRunGit(t, runner, path, "rev-parse", "HEAD")
}

func workspaceRegisterRepository(t *testing.T, fixture workspaceFixture, source, projectID, repositoryName string) work.Project {
	t.Helper()
	reviewResult := fixture.manager.ReviewSource(context.Background(), source)
	if !reviewResult.OK {
		t.Fatalf("ReviewSource failed: %s", reviewResult.Error())
	}
	review := reviewResult.Value.(SourceReview)
	result := fixture.manager.Register(context.Background(), RegisterRequest{
		ProjectID:            projectID,
		SourcePath:           source,
		RepositoryName:       repositoryName,
		Confirmed:            true,
		ExpectedIncludedHash: review.IncludedHash,
	})
	if !result.OK {
		t.Fatalf("Register failed: %s", result.Error())
	}
	return result.Value.(work.Project)
}

func workspacePrepareRun(t *testing.T, fixture workspaceFixture, project work.Project, run work.Run) RunWorkspace {
	t.Helper()
	result := fixture.manager.PrepareRun(context.Background(), project, run)
	if !result.OK {
		t.Fatalf("PrepareRun failed: %s", result.Error())
	}
	return result.Value.(RunWorkspace)
}

func workspaceNewRunFixture(t *testing.T) workspaceRunFixture {
	t.Helper()
	fixture := workspaceNewRegisteredRunFixture(t)
	fixture.prepared = workspacePrepareRun(t, fixture.workspaceFixture, fixture.project, fixture.run)
	return fixture
}

func workspaceNewRegisteredRunFixture(t *testing.T) workspaceRunFixture {
	t.Helper()
	fixture := workspaceNewFixture(t)
	source := core.PathJoin(t.TempDir(), "run fixture source")
	workspaceCreateRepository(t, fixture.runner, source)
	project := workspaceRegisterRepository(t, fixture, source, "project", "project")
	run := work.Run{ID: "run", WorkID: "work", ProjectID: project.ID, Number: 1, SourceRevision: project.SourceRevision}
	return workspaceRunFixture{workspaceFixture: fixture, project: project, run: run}
}

func workspaceNewDurableRunFixture(t *testing.T) workspaceRunFixture {
	t.Helper()
	fixture := workspaceNewRunFixture(t)
	workspaceWriteFile(t, core.PathJoin(fixture.prepared.Path, "durable.txt"), "durable\n")
	captured := fixture.manager.CaptureRun(context.Background(), fixture.prepared)
	if !captured.OK || !captured.Value.(Capture).Pushed {
		t.Fatalf("CaptureRun failed: %s", captured.Error())
	}
	fixture.run.DurableRevision = captured.Value.(Capture).DurableRevision
	if released := fixture.manager.ReleaseRun(context.Background(), fixture.prepared); !released.OK {
		t.Fatalf("ReleaseRun failed: %s", released.Error())
	}
	fixture.run.Branch = fixture.prepared.Branch
	fixture.run.Worktree = fixture.prepared.Path
	return fixture
}

func TestWorkspace_NewManager_Good(t *testing.T) {
	fixture := workspaceNewFixture(t)
	result := NewManager(ManagerOptions{
		Root: fixture.root, Files: fixture.files, Git: fixture.runner, Server: fixture.server,
		IDs: func() string { return "manager-good" }, Now: time.Now,
	})
	core.AssertTrue(t, result.OK, result.Error())
	manager := result.Value.(*Manager)
	core.AssertEqual(t, fixture.root, manager.root)
	core.AssertTrue(t, manager.files == fixture.files)
	core.AssertEqual(t, 0, len(manager.leases))
}

func TestWorkspace_NewManager_Bad(t *testing.T) {
	valid := workspaceNewFixture(t)
	tests := []ManagerOptions{
		{},
		{Root: "relative", Files: valid.files, Git: valid.runner, Server: valid.server, IDs: func() string { return "id" }, Now: time.Now},
		{Root: valid.root, Git: valid.runner, Server: valid.server, IDs: func() string { return "id" }, Now: time.Now},
		{Root: valid.root, Files: valid.files, Server: valid.server, IDs: func() string { return "id" }, Now: time.Now},
		{Root: valid.root, Files: valid.files, Git: valid.runner, IDs: func() string { return "id" }, Now: time.Now},
		{Root: valid.root, Files: valid.files, Git: valid.runner, Server: valid.server, Now: time.Now},
		{Root: valid.root, Files: valid.files, Git: valid.runner, Server: valid.server, IDs: func() string { return "id" }},
	}
	for _, options := range tests {
		core.AssertFalse(t, NewManager(options).OK)
	}
	fault := &workspaceFaultMedium{Medium: valid.files, failEnsure: ""}
	core.AssertFalse(t, NewManager(ManagerOptions{
		Root: valid.root, Files: fault, Git: valid.runner, Server: valid.server,
		IDs: func() string { return "id" }, Now: time.Now,
	}).OK)
}

func TestWorkspace_NewManager_Ugly(t *testing.T) {
	fixture := workspaceNewFixture(t)
	rootWithDots := core.PathJoin(fixture.root, "child", "..")
	result := NewManager(ManagerOptions{
		Root: rootWithDots, Files: fixture.files, Git: fixture.runner, Server: fixture.server,
		IDs: func() string { return "id" }, Now: time.Now,
	})
	core.AssertTrue(t, result.OK, result.Error())
	core.AssertEqual(t, fixture.root, result.Value.(*Manager).root)
}

func TestWorkspace_Manager_ReviewSource_Good(t *testing.T) {
	fixture := workspaceNewFixture(t)
	source := core.PathJoin(t.TempDir(), "source repository")
	revision := workspaceCreateRepository(t, fixture.runner, source)
	workspaceWriteFile(t, core.PathJoin(source, "generated.ignored"), "ignored\n")

	result := fixture.manager.ReviewSource(context.Background(), core.PathJoin(source, "nested"))
	core.AssertTrue(t, result.OK, result.Error())
	review := result.Value.(SourceReview)
	core.AssertEqual(t, workspaceCanonical(t, core.PathJoin(source, "nested")), review.Path)
	core.AssertEqual(t, workspaceCanonical(t, source), review.Root)
	core.AssertEqual(t, "main", review.Branch)
	core.AssertEqual(t, revision, review.Revision)
	core.AssertEqual(t, core.Concat(workspaceTestName, " <", workspaceTestEmail, ">"), review.CommitIdentity)
	core.AssertTrue(t, review.Git)
	core.AssertTrue(t, review.Clean)
	core.AssertFalse(t, review.Detached)
	core.AssertTrue(t, workspaceContainsArgument(review.Included, "README.md"))
	core.AssertTrue(t, workspaceContainsArgument(review.Included, "nested/tracked.txt"))
	core.AssertFalse(t, workspaceContainsArgument(review.Included, "generated.ignored"))
	core.AssertTrue(t, review.IncludedHash != "")
}

func TestWorkspace_Manager_ReviewSource_Bad(t *testing.T) {
	var manager *Manager
	core.AssertFalse(t, manager.ReviewSource(context.Background(), "/tmp").OK)
	fixture := workspaceNewFixture(t)
	core.AssertFalse(t, fixture.manager.ReviewSource(nil, "/tmp").OK)
	core.AssertFalse(t, fixture.manager.ReviewSource(context.Background(), "").OK)
	core.AssertFalse(t, fixture.manager.ReviewSource(context.Background(), core.PathJoin(t.TempDir(), "missing")).OK)
}

func TestWorkspace_Manager_ReviewSource_Ugly(t *testing.T) {
	fixture := workspaceNewFixture(t)
	source := core.PathJoin(t.TempDir(), "dirty repository")
	workspaceCreateRepository(t, fixture.runner, source)
	workspaceWriteFile(t, core.PathJoin(source, "README.md"), "dirty\n")
	review := fixture.manager.ReviewSource(context.Background(), source)
	core.AssertTrue(t, review.OK, review.Error())
	core.AssertFalse(t, review.Value.(SourceReview).Clean)

	workspaceRunGit(t, fixture.runner, source, "reset", "--hard", "HEAD")
	workspaceRunGit(t, fixture.runner, source, "checkout", "--detach", "HEAD")
	review = fixture.manager.ReviewSource(context.Background(), source)
	core.AssertTrue(t, review.OK, review.Error())
	core.AssertTrue(t, review.Value.(SourceReview).Detached)

	adhoc := core.PathJoin(t.TempDir(), "ad hoc folder")
	workspaceWriteFile(t, core.PathJoin(adhoc, ".gitignore"), "secret.txt\n")
	workspaceWriteFile(t, core.PathJoin(adhoc, "keep.txt"), "keep\n")
	workspaceWriteFile(t, core.PathJoin(adhoc, "secret.txt"), "secret\n")
	review = fixture.manager.ReviewSource(context.Background(), adhoc)
	core.AssertTrue(t, review.OK, review.Error())
	adhocReview := review.Value.(SourceReview)
	core.AssertFalse(t, adhocReview.Git)
	core.AssertEqual(t, "main", adhocReview.ProposedBranch)
	core.AssertEqual(t, "LEM Agent <lem@localhost>", adhocReview.CommitIdentity)
	core.AssertTrue(t, workspaceContainsArgument(adhocReview.Included, "keep.txt"))
	core.AssertFalse(t, workspaceContainsArgument(adhocReview.Included, "secret.txt"))
}

func TestWorkspaceReviewGitFailures(t *testing.T) {
	stages := []string{"root", "revision", "status", "list", "hash"}
	for _, stage := range stages {
		t.Run(stage, func(t *testing.T) {
			fixture := workspaceNewFixture(t)
			source := core.PathJoin(t.TempDir(), "review failure source")
			workspaceCreateRepository(t, fixture.runner, source)
			if stage == "root" {
				fixture.runner.setValue(func(command Command) (any, bool) {
					if workspaceCommandContains(command, "rev-parse\x00--show-toplevel") {
						return core.PathJoin(t.TempDir(), "missing-root"), true
					}
					return nil, false
				})
			} else if stage == "hash" {
				fixture.runner.setValue(func(command Command) (any, bool) {
					if workspaceContainsArgument(command.Args, "ls-files") {
						return "missing.txt\x00", true
					}
					return nil, false
				})
			} else {
				fixture.runner.setFailure(func(command Command) bool {
					switch stage {
					case "revision":
						return workspaceCommandContains(command, "rev-parse\x00HEAD")
					case "status":
						return workspaceCommandContains(command, "status\x00--porcelain=v1")
					case "list":
						return workspaceContainsArgument(command.Args, "ls-files")
					}
					return false
				})
			}
			core.AssertFalse(t, fixture.manager.ReviewSource(context.Background(), source).OK)
		})
	}
}

func TestWorkspaceManagerReviewSourceRejectsAmbiguousRepositoryRootFailures(t *testing.T) {
	for _, existingRepository := range []bool{false, true} {
		t.Run(core.Sprintf("existing=%t", existingRepository), func(t *testing.T) {
			fixture := workspaceNewFixture(t)
			source := core.PathJoin(t.TempDir(), "ambiguous repository root")
			workspaceWriteFile(t, core.PathJoin(source, "file.txt"), "content\n")
			if existingRepository {
				workspaceCreateRepository(t, fixture.runner, source)
			}
			fixture.runner.setFailure(func(command Command) bool {
				return workspaceCommandContains(command, "rev-parse\x00--show-toplevel")
			})

			result := fixture.manager.ReviewSource(context.Background(), source)
			core.AssertFalse(t, result.OK)
		})
	}
}

func TestWorkspace_probeRepositoryRoot_Bad(t *testing.T) {
	selectedPath := t.TempDir()
	manager := &Manager{git: workspaceRunOnly{result: core.Ok(" /reviewed/root\n")}}
	result := manager.probeRepositoryRoot(context.Background(), selectedPath)
	core.AssertTrue(t, result.OK, result.Error())
	core.AssertEqual(t, "/reviewed/root", result.Value.(string))

	manager.git = workspaceRunOnly{result: core.Fail(core.NewError("injected root failure"))}
	result = manager.probeRepositoryRoot(context.Background(), selectedPath)
	core.AssertFalse(t, result.OK)
	core.AssertContains(t, result.Error(), "cannot classify")

	manager.git = workspaceDetailedProbeRunner{result: core.Fail(core.NewError("injected detailed failure"))}
	result = manager.probeRepositoryRoot(context.Background(), selectedPath)
	core.AssertFalse(t, result.OK)
	core.AssertContains(t, result.Error(), "failed to inspect")

	manager.git = workspaceDetailedProbeRunner{result: core.Ok("wrong outcome")}
	result = manager.probeRepositoryRoot(context.Background(), selectedPath)
	core.AssertFalse(t, result.OK)
	core.AssertContains(t, result.Error(), "instead of command outcome")

	manager.git = workspaceDetailedProbeRunner{result: core.Ok(CommandOutcome{ExitCode: 0, Output: "  \n"})}
	result = manager.probeRepositoryRoot(context.Background(), selectedPath)
	core.AssertFalse(t, result.OK)
	core.AssertContains(t, result.Error(), "empty repository root")

	loopRoot := t.TempDir()
	core.AssertTrue(t, core.Symlink(".git", core.PathJoin(loopRoot, ".git")).OK)
	manager.git = workspaceDetailedProbeRunner{result: core.Ok(CommandOutcome{
		ExitCode: 128, Output: "fatal: not a git repository",
	})}
	result = manager.probeRepositoryRoot(context.Background(), loopRoot)
	core.AssertFalse(t, result.OK)
	core.AssertContains(t, result.Error(), "repository marker")
}

func TestWorkspaceAdHocReviewFailures(t *testing.T) {
	stages := []string{"id", "stale-delete", "ensure", "init", "list", "list-cleanup", "cleanup", "hash"}
	for _, stage := range stages {
		t.Run(stage, func(t *testing.T) {
			fixture := workspaceNewFixture(t)
			source := core.PathJoin(t.TempDir(), "ad hoc review failure")
			workspaceWriteFile(t, core.PathJoin(source, "file.txt"), "content\n")
			reviewRelative := core.PathJoin(".reviews", "review-id.git")
			switch stage {
			case "id":
				fixture.manager.ids = func() string { return "../bad" }
			case "stale-delete":
				core.AssertTrue(t, fixture.files.EnsureDir(reviewRelative) == nil)
				fixture.manager.files = &workspaceFaultMedium{Medium: fixture.files, failDelete: reviewRelative}
			case "ensure":
				fixture.manager.files = &workspaceFaultMedium{Medium: fixture.files, failEnsure: ".reviews"}
			case "init":
				fixture.runner.setFailure(func(command Command) bool { return workspaceCommandContains(command, "init\x00--bare") })
			case "list":
				fixture.runner.setFailure(func(command Command) bool { return workspaceContainsArgument(command.Args, "ls-files") })
			case "list-cleanup":
				fixture.runner.setFailure(func(command Command) bool { return workspaceContainsArgument(command.Args, "ls-files") })
				fixture.manager.files = &workspaceFaultMedium{Medium: fixture.files, failDelete: reviewRelative}
			case "cleanup":
				fixture.manager.files = &workspaceFaultMedium{Medium: fixture.files, failDelete: reviewRelative}
			case "hash":
				fixture.runner.setValue(func(command Command) (any, bool) {
					if workspaceContainsArgument(command.Args, "ls-files") {
						return "missing.txt\x00", true
					}
					return nil, false
				})
			}
			core.AssertFalse(t, fixture.manager.ReviewSource(context.Background(), source).OK)
		})
	}
}

func TestWorkspace_Manager_Register_Good(t *testing.T) {
	fixture := workspaceNewFixture(t)
	source := core.PathJoin(t.TempDir(), "registered source")
	revision := workspaceCreateRepository(t, fixture.runner, source)
	workspaceRunGit(t, fixture.runner, source, "remote", "add", "upstream", "ssh://example.invalid/team/repo")
	remotesBefore := workspaceRunGit(t, fixture.runner, source, "remote", "-v")
	filesBefore := workspaceRunGit(t, fixture.runner, source, "ls-files", "--cached", "--others", "--exclude-standard", "-z")
	review := fixture.manager.ReviewSource(context.Background(), source).Value.(SourceReview)

	result := fixture.manager.Register(context.Background(), RegisterRequest{
		ProjectID: "project-1", SourcePath: source, RepositoryName: "project-1",
		Confirmed: true, ExpectedIncludedHash: review.IncludedHash,
	})
	core.AssertTrue(t, result.OK, result.Error())
	project := result.Value.(work.Project)
	core.AssertEqual(t, "project-1", project.ID)
	core.AssertEqual(t, workspaceCanonical(t, source), project.SourcePath)
	core.AssertEqual(t, workspaceCanonical(t, source), project.RepositoryRoot)
	core.AssertEqual(t, "main", project.SourceBranch)
	core.AssertEqual(t, revision, project.SourceRevision)
	core.AssertEqual(t, core.PathJoin(fixture.root, "project-1", "repo.git"), project.ClonePath)
	core.AssertEqual(t, remotesBefore, workspaceRunGit(t, fixture.runner, source, "remote", "-v"))
	core.AssertEqual(t, filesBefore, workspaceRunGit(t, fixture.runner, source, "ls-files", "--cached", "--others", "--exclude-standard", "-z"))
	core.AssertFalse(t, core.Stat(core.PathJoin(source, ".lem")).OK)
	remote := fixture.server.EnsureRepository(context.Background(), "project-1").Value.(gitserver.Repository)
	core.AssertEqual(t, revision, workspaceRunGit(t, fixture.runner, remote.CloneURL, "rev-parse", "refs/heads/main"))
	repeated := fixture.manager.Register(context.Background(), RegisterRequest{
		ProjectID: "project-1", SourcePath: source, RepositoryName: "project-1",
		Confirmed: true, ExpectedIncludedHash: review.IncludedHash,
	})
	core.AssertTrue(t, repeated.OK, repeated.Error())
}

func TestWorkspace_Manager_Register_Bad(t *testing.T) {
	var manager *Manager
	core.AssertFalse(t, manager.Register(context.Background(), RegisterRequest{}).OK)
	fixture := workspaceNewFixture(t)
	source := core.PathJoin(t.TempDir(), "registration failures")
	workspaceCreateRepository(t, fixture.runner, source)
	review := fixture.manager.ReviewSource(context.Background(), source).Value.(SourceReview)
	core.AssertFalse(t, fixture.manager.Register(nil, RegisterRequest{}).OK)
	cancelled, cancel := context.WithCancel(context.Background())
	cancel()
	core.AssertFalse(t, fixture.manager.Register(cancelled, RegisterRequest{}).OK)
	core.AssertFalse(t, fixture.manager.Register(context.Background(), RegisterRequest{ProjectID: "bad/id", SourcePath: source, RepositoryName: "repo", Confirmed: true, ExpectedIncludedHash: review.IncludedHash}).OK)
	core.AssertFalse(t, fixture.manager.Register(context.Background(), RegisterRequest{ProjectID: "project", SourcePath: source, Confirmed: true, ExpectedIncludedHash: review.IncludedHash}).OK)
	core.AssertFalse(t, fixture.manager.Register(context.Background(), RegisterRequest{ProjectID: "project", SourcePath: source, RepositoryName: "repo", ExpectedIncludedHash: review.IncludedHash}).OK)
	core.AssertFalse(t, fixture.manager.Register(context.Background(), RegisterRequest{ProjectID: "project", SourcePath: source, RepositoryName: "repo", Confirmed: true}).OK)
	core.AssertFalse(t, fixture.manager.Register(context.Background(), RegisterRequest{ProjectID: "project", SourcePath: source, RepositoryName: "repo", Confirmed: true, ExpectedIncludedHash: "wrong"}).OK)
	core.AssertFalse(t, fixture.manager.Register(context.Background(), RegisterRequest{ProjectID: "project", SourcePath: source, RepositoryName: "repo", EnableGit: true, Confirmed: true, ExpectedIncludedHash: review.IncludedHash}).OK)
	core.AssertFalse(t, fixture.manager.Register(context.Background(), RegisterRequest{ProjectID: "project", SourcePath: core.PathJoin(t.TempDir(), "missing"), RepositoryName: "repo", Confirmed: true, ExpectedIncludedHash: review.IncludedHash}).OK)
	workspaceWriteFile(t, core.PathJoin(source, "dirty.txt"), "dirty\n")
	fresh := fixture.manager.ReviewSource(context.Background(), source).Value.(SourceReview)
	core.AssertFalse(t, fresh.Clean)
	core.AssertFalse(t, fixture.manager.Register(context.Background(), RegisterRequest{ProjectID: "project", SourcePath: source, RepositoryName: "repo", Confirmed: true, ExpectedIncludedHash: fresh.IncludedHash}).OK)
	workspaceRunGit(t, fixture.runner, source, "reset", "--hard", "HEAD")
	workspaceRunGit(t, fixture.runner, source, "clean", "-fd")
	workspaceRunGit(t, fixture.runner, source, "checkout", "--detach", "HEAD")
	detached := fixture.manager.ReviewSource(context.Background(), source).Value.(SourceReview)
	core.AssertFalse(t, fixture.manager.Register(context.Background(), RegisterRequest{ProjectID: "project", SourcePath: source, RepositoryName: "repo", Confirmed: true, ExpectedIncludedHash: detached.IncludedHash}).OK)

	adhoc := core.PathJoin(t.TempDir(), "adhoc without enablement")
	workspaceWriteFile(t, core.PathJoin(adhoc, "file.txt"), "file\n")
	adhocReview := fixture.manager.ReviewSource(context.Background(), adhoc).Value.(SourceReview)
	core.AssertFalse(t, fixture.manager.Register(context.Background(), RegisterRequest{ProjectID: "adhoc", SourcePath: adhoc, RepositoryName: "adhoc", Confirmed: true, ExpectedIncludedHash: adhocReview.IncludedHash}).OK)
}

func TestWorkspaceRegisterInfrastructureFailures(t *testing.T) {
	stages := []string{
		"server-start", "server-ensure", "server-shape", "credentials", "seed", "ensure-project",
		"invalid-cache", "clone", "fetch", "identity-name", "identity-email", "verify", "clock",
	}
	for _, stage := range stages {
		t.Run(stage, func(t *testing.T) {
			fixture := workspaceNewFixture(t)
			source := core.PathJoin(t.TempDir(), "registration infrastructure source")
			workspaceCreateRepository(t, fixture.runner, source)
			review := fixture.manager.ReviewSource(context.Background(), source).Value.(SourceReview)
			request := RegisterRequest{
				ProjectID: "project", SourcePath: source, RepositoryName: "project",
				Confirmed: true, ExpectedIncludedHash: review.IncludedHash,
			}
			switch stage {
			case "server-start":
				fixture.server.setFailure(true, false)
			case "server-ensure":
				fixture.server.setFailure(false, true)
			case "server-shape":
				fixture.server.setEnsureValue("not a repository")
			case "credentials":
				fixture.server.setRepository(gitserver.Repository{Name: "project", CloneURL: "/tmp/project.git", IdentityFile: "/tmp/id"})
			case "seed":
				fixture.runner.setFailure(func(command Command) bool {
					return workspaceCommandContains(command, "push\x00--force\x00")
				})
			case "ensure-project":
				fixture.manager.files = &workspaceFaultMedium{Medium: fixture.files, failEnsure: "project"}
			case "invalid-cache":
				core.AssertTrue(t, fixture.files.EnsureDir(core.PathJoin("project", "repo.git")) == nil)
			case "clone":
				fixture.runner.setFailure(func(command Command) bool { return len(command.Args) > 0 && command.Args[0] == "clone" })
			case "fetch":
				fixture.runner.setFailure(func(command Command) bool { return workspaceContainsArgument(command.Args, "fetch") })
			case "identity-name":
				fixture.runner.setFailure(func(command Command) bool { return workspaceCommandContains(command, "config\x00user.name") })
			case "identity-email":
				fixture.runner.setFailure(func(command Command) bool { return workspaceCommandContains(command, "config\x00user.email") })
			case "verify":
				fixture.runner.setFailure(func(command Command) bool { return workspaceCommandContains(command, "^{commit}") })
			case "clock":
				fixture.manager.now = func() time.Time { return time.Time{} }
			}
			result := fixture.manager.Register(context.Background(), request)
			core.AssertFalse(t, result.OK)
		})
	}
}

func TestWorkspaceAdHocInitializationFailures(t *testing.T) {
	stages := []string{"init", "identity-name", "identity-email", "add", "commit", "second-review", "baseline-drift"}
	for _, stage := range stages {
		t.Run(stage, func(t *testing.T) {
			fixture := workspaceNewFixture(t)
			source := core.PathJoin(t.TempDir(), "ad hoc initialization source")
			workspaceWriteFile(t, core.PathJoin(source, "file.txt"), "content\n")
			review := fixture.manager.ReviewSource(context.Background(), source).Value.(SourceReview)
			afterCommit := false
			if stage == "baseline-drift" {
				fixture.runner.setValue(func(command Command) (any, bool) {
					if len(command.Args) > 0 && command.Args[0] == "commit" {
						afterCommit = true
						return nil, false
					}
					if afterCommit && workspaceCommandContains(command, "status\x00--porcelain=v1") {
						return " M file.txt\n", true
					}
					return nil, false
				})
			} else {
				fixture.runner.setFailure(func(command Command) bool {
					switch stage {
					case "init":
						return workspaceCommandContains(command, "init\x00--initial-branch")
					case "identity-name":
						return workspaceCommandContains(command, "config\x00--local\x00user.name")
					case "identity-email":
						return workspaceCommandContains(command, "config\x00--local\x00user.email")
					case "add":
						return len(command.Args) > 0 && command.Args[0] == "add"
					case "commit":
						return len(command.Args) > 0 && command.Args[0] == "commit"
					case "second-review":
						if len(command.Args) > 0 && command.Args[0] == "commit" {
							afterCommit = true
							return false
						}
						return afterCommit && (workspaceCommandContains(command, "rev-parse\x00--show-toplevel") || workspaceCommandContains(command, "init\x00--bare"))
					}
					return false
				})
			}
			result := fixture.manager.Register(context.Background(), RegisterRequest{
				ProjectID: "adhoc", SourcePath: source, RepositoryName: "adhoc", EnableGit: true,
				Confirmed: true, ExpectedIncludedHash: review.IncludedHash,
			})
			core.AssertFalse(t, result.OK)
		})
	}
}

func TestWorkspace_Manager_Register_Ugly(t *testing.T) {
	fixture := workspaceNewFixture(t)
	source := core.PathJoin(t.TempDir(), "ad hoc registration")
	workspaceWriteFile(t, core.PathJoin(source, ".gitignore"), "secret.txt\n")
	workspaceWriteFile(t, core.PathJoin(source, "keep.txt"), "before\n")
	workspaceWriteFile(t, core.PathJoin(source, "secret.txt"), "ignored\n")
	review := fixture.manager.ReviewSource(context.Background(), source).Value.(SourceReview)

	workspaceWriteFile(t, core.PathJoin(source, "keep.txt"), "drift\n")
	drifted := fixture.manager.Register(context.Background(), RegisterRequest{
		ProjectID: "adhoc", SourcePath: source, RepositoryName: "adhoc", EnableGit: true,
		Confirmed: true, ExpectedIncludedHash: review.IncludedHash,
	})
	core.AssertFalse(t, drifted.OK)
	core.AssertFalse(t, core.Stat(core.PathJoin(source, ".git")).OK)

	review = fixture.manager.ReviewSource(context.Background(), source).Value.(SourceReview)
	result := fixture.manager.Register(context.Background(), RegisterRequest{
		ProjectID: "adhoc", SourcePath: source, RepositoryName: "adhoc", EnableGit: true,
		Confirmed: true, ExpectedIncludedHash: review.IncludedHash,
	})
	core.AssertTrue(t, result.OK, result.Error())
	core.AssertTrue(t, core.Stat(core.PathJoin(source, ".git")).OK)
	core.AssertEqual(t, "LEM Agent", workspaceRunGit(t, fixture.runner, source, "config", "--local", "user.name"))
	core.AssertEqual(t, "lem@localhost", workspaceRunGit(t, fixture.runner, source, "config", "--local", "user.email"))
	core.AssertEqual(t, "", workspaceRunGit(t, fixture.runner, source, "ls-files", "secret.txt"))
	core.AssertFalse(t, core.Stat(core.PathJoin(source, ".lem")).OK)
}

func TestWorkspace_Manager_PrepareRun_Good(t *testing.T) {
	fixture := workspaceNewFixture(t)
	source := core.PathJoin(t.TempDir(), "prepare source")
	workspaceCreateRepository(t, fixture.runner, source)
	project := workspaceRegisterRepository(t, fixture, source, "project-1", "project-1")
	run := work.Run{ID: "run-1", WorkID: "Work / Alpha", ProjectID: project.ID, Number: 2, SourceRevision: project.SourceRevision}

	result := fixture.manager.PrepareRun(context.Background(), project, run)
	core.AssertTrue(t, result.OK, result.Error())
	prepared := result.Value.(RunWorkspace)
	core.AssertEqual(t, "lem/work/Work-Alpha/run-2", prepared.Branch)
	core.AssertEqual(t, core.PathJoin(fixture.root, "project-1", "runs", "run-1", "worktree"), prepared.Path)
	core.AssertEqual(t, project.SourceRevision, prepared.BaseRevision)
	core.AssertEqual(t, "", prepared.DurableRevision)
	core.AssertTrue(t, core.Stat(core.PathJoin(prepared.Path, "README.md")).OK)
	core.AssertEqual(t, "seed", core.Trim(string(core.ReadFile(core.PathJoin(source, "README.md")).Value.([]byte))))
}

func TestWorkspace_Manager_PrepareRun_Bad(t *testing.T) {
	var manager *Manager
	core.AssertFalse(t, manager.PrepareRun(context.Background(), work.Project{}, work.Run{}).OK)
	fixture := workspaceNewFixture(t)
	core.AssertFalse(t, fixture.manager.PrepareRun(nil, work.Project{}, work.Run{}).OK)
	core.AssertFalse(t, fixture.manager.PrepareRun(context.Background(), work.Project{}, work.Run{}).OK)
}

func TestWorkspace_Manager_PrepareRun_Ugly(t *testing.T) {
	fixture := workspaceNewFixture(t)
	source := core.PathJoin(t.TempDir(), "prepare edge source")
	workspaceCreateRepository(t, fixture.runner, source)
	project := workspaceRegisterRepository(t, fixture, source, "project", "project")
	run := work.Run{ID: "run", WorkID: "work", ProjectID: project.ID, Number: 1, SourceRevision: project.SourceRevision}
	core.AssertTrue(t, fixture.manager.PrepareRun(context.Background(), project, run).OK)
	core.AssertFalse(t, fixture.manager.PrepareRun(context.Background(), project, run).OK)

	project.ClonePath = core.PathJoin(t.TempDir(), "outside.git")
	run.ID = "run-2"
	core.AssertFalse(t, fixture.manager.PrepareRun(context.Background(), project, run).OK)
}

func TestWorkspacePrepareRunFailures(t *testing.T) {
	stages := []string{"cancelled", "branch-lease", "existing-path", "server", "credentials", "fetch", "ensure", "add", "add-cleanup", "base", "base-mismatch"}
	for _, stage := range stages {
		t.Run(stage, func(t *testing.T) {
			fixture := workspaceNewRegisteredRunFixture(t)
			ctx := context.Background()
			branch := runBranch(fixture.run.WorkID, fixture.run.Number).Value.(string)
			path := core.PathJoin(fixture.root, fixture.project.ID, "runs", fixture.run.ID, "worktree")
			switch stage {
			case "cancelled":
				cancelled, cancel := context.WithCancel(context.Background())
				cancel()
				ctx = cancelled
			case "branch-lease":
				fixture.manager.leases["other"] = RunWorkspace{RunID: "other", Branch: branch, Path: core.PathJoin(fixture.root, "other")}
			case "existing-path":
				core.AssertTrue(t, fixture.files.EnsureDir(core.PathJoin("project", "runs", "run", "worktree")) == nil)
			case "server":
				fixture.server.setFailure(false, true)
			case "credentials":
				fixture.server.setRepository(gitserver.Repository{Name: "project", CloneURL: "/tmp/project.git", IdentityFile: "/tmp/id"})
			case "fetch":
				fixture.runner.setFailure(func(command Command) bool { return workspaceContainsArgument(command.Args, "fetch") })
			case "ensure":
				fixture.manager.files = &workspaceFaultMedium{Medium: fixture.files, failEnsure: core.PathJoin("project", "runs", "run")}
			case "add", "add-cleanup":
				fixture.runner.setFailure(func(command Command) bool { return workspaceCommandContains(command, "worktree\x00add") })
				if stage == "add-cleanup" {
					fixture.manager.files = &workspaceFaultMedium{Medium: fixture.files, failDelete: core.PathJoin("project", "runs", "run")}
				}
			case "base":
				fixture.runner.setFailure(func(command Command) bool {
					return command.Dir == path && workspaceCommandContains(command, "rev-parse\x00HEAD")
				})
			case "base-mismatch":
				fixture.runner.setValue(func(command Command) (any, bool) {
					if command.Dir == path && workspaceCommandContains(command, "rev-parse\x00HEAD") {
						return "0000000000000000000000000000000000000000\n", true
					}
					return nil, false
				})
			}
			result := fixture.manager.PrepareRun(ctx, fixture.project, fixture.run)
			core.AssertFalse(t, result.OK)
		})
	}
}

func TestWorkspacePrepareRunPostAddFailureCleansProvisionalWorktreeAndBranch(t *testing.T) {
	fixture := workspaceNewRegisteredRunFixture(t)
	branch := runBranch(fixture.run.WorkID, fixture.run.Number).Value.(string)
	path := core.PathJoin(fixture.root, fixture.project.ID, "runs", fixture.run.ID, "worktree")
	fixture.runner.setFailure(func(command Command) bool {
		return command.Dir == path && workspaceCommandContains(command, "rev-parse\x00HEAD")
	})

	result := fixture.manager.PrepareRun(context.Background(), fixture.project, fixture.run)
	core.AssertFalse(t, result.OK)
	fixture.runner.setFailure(nil)
	core.AssertFalse(t, core.Stat(path).OK)
	core.AssertEqual(t, "", workspaceRunGit(t, fixture.runner, fixture.root, "--git-dir", fixture.project.ClonePath, "branch", "--list", branch))
	_, leased := fixture.manager.leases[fixture.run.ID]
	core.AssertFalse(t, leased)
}

func TestWorkspaceRunValidationEdges(t *testing.T) {
	fixture := workspaceNewRegisteredRunFixture(t)
	project := fixture.project
	run := fixture.run

	badProject := project
	badProject.ID = "bad/id"
	core.AssertFalse(t, fixture.manager.validateRun(badProject, run, false).OK)
	badRun := run
	badRun.ID = "bad/id"
	core.AssertFalse(t, fixture.manager.validateRun(project, badRun, false).OK)
	badRun = run
	badRun.ProjectID = "another"
	core.AssertFalse(t, fixture.manager.validateRun(project, badRun, false).OK)
	for _, field := range []string{"repository", "branch", "revision"} {
		badProject = project
		switch field {
		case "repository":
			badProject.RepositoryName = ""
		case "branch":
			badProject.SourceBranch = ""
		case "revision":
			badProject.SourceRevision = ""
		}
		core.AssertFalse(t, fixture.manager.validateRun(badProject, run, false).OK)
	}
	badProject = project
	badProject.ClonePath = core.PathJoin(t.TempDir(), "outside.git")
	core.AssertFalse(t, fixture.manager.validateRun(badProject, run, false).OK)
	badRun = run
	badRun.Number = 0
	core.AssertFalse(t, fixture.manager.validateRun(project, badRun, false).OK)
	badRun = run
	badRun.Branch = "lem/work/wrong/run-1"
	core.AssertFalse(t, fixture.manager.validateRun(project, badRun, false).OK)
	for _, branch := range []string{"bad", "lem/work/../bad", "lem/work/bad branch/run-1"} {
		badRun = run
		badRun.Branch = branch
		core.AssertFalse(t, fixture.manager.validateRun(project, badRun, true).OK)
	}
	badRun = run
	badRun.Worktree = core.PathJoin(t.TempDir(), "outside")
	core.AssertFalse(t, fixture.manager.validateRun(project, badRun, true).OK)
	badRun = run
	badRun.Worktree = core.PathJoin(fixture.root, "wrong")
	core.AssertFalse(t, fixture.manager.validateRun(project, badRun, false).OK)
}

func TestWorkspace_Manager_CaptureRun_Good(t *testing.T) {
	fixture := workspaceNewFixture(t)
	source := core.PathJoin(t.TempDir(), "capture source")
	workspaceCreateRepository(t, fixture.runner, source)
	project := workspaceRegisterRepository(t, fixture, source, "project", "project")
	run := work.Run{ID: "run", WorkID: "work", ProjectID: project.ID, Number: 1, SourceRevision: project.SourceRevision}
	prepared := workspacePrepareRun(t, fixture, project, run)
	workspaceWriteFile(t, core.PathJoin(prepared.Path, "README.md"), "agent change\n")
	workspaceWriteFile(t, core.PathJoin(prepared.Path, "new.txt"), "new\n")
	workspaceWriteFile(t, core.PathJoin(prepared.Path, "cache.ignored"), "ignored\n")
	result := fixture.manager.CaptureRun(context.Background(), prepared)
	core.AssertTrue(t, result.OK, result.Error())
	capture := result.Value.(Capture)
	core.AssertTrue(t, capture.Changed)
	core.AssertTrue(t, capture.Pushed)
	core.AssertFalse(t, capture.Retained)
	core.AssertTrue(t, capture.Revision != project.SourceRevision)
	core.AssertEqual(t, "", workspaceRunGit(t, fixture.runner, prepared.Path, "ls-files", "cache.ignored"))
	remote := fixture.server.EnsureRepository(context.Background(), project.RepositoryName).Value.(gitserver.Repository)
	core.AssertEqual(t, capture.Revision, workspaceRunGit(t, fixture.runner, remote.CloneURL, "rev-parse", core.Concat("refs/heads/", prepared.Branch)))
}

func TestWorkspaceCaptureRunCancelledContextRetainsLease(t *testing.T) {
	fixture := workspaceNewRunFixture(t)
	cancelled, cancel := context.WithCancel(context.Background())
	cancel()
	result := fixture.manager.CaptureRun(cancelled, fixture.prepared)
	core.AssertTrue(t, result.OK, result.Error())
	capture := result.Value.(Capture)
	core.AssertTrue(t, capture.Retained)
	core.AssertFalse(t, capture.Pushed)
	core.AssertContains(t, capture.Summary, "context canceled")
	core.AssertTrue(t, core.Stat(fixture.prepared.Path).OK)
	core.AssertEqual(t, fixture.prepared.Path, fixture.manager.leases[fixture.prepared.RunID].Path)
}

func TestWorkspace_Manager_CaptureRun_Bad(t *testing.T) {
	var manager *Manager
	core.AssertFalse(t, manager.CaptureRun(context.Background(), RunWorkspace{}).OK)
	fixture := workspaceNewFixture(t)
	core.AssertFalse(t, fixture.manager.CaptureRun(nil, RunWorkspace{}).OK)
	core.AssertFalse(t, fixture.manager.CaptureRun(context.Background(), RunWorkspace{RunID: "unknown", Path: fixture.root}).OK)
}

func TestWorkspace_Manager_CaptureRun_Ugly(t *testing.T) {
	fixture := workspaceNewFixture(t)
	source := core.PathJoin(t.TempDir(), "capture failure source")
	workspaceCreateRepository(t, fixture.runner, source)
	project := workspaceRegisterRepository(t, fixture, source, "project", "project")
	run := work.Run{ID: "run", WorkID: "work", ProjectID: project.ID, Number: 1, SourceRevision: project.SourceRevision}
	prepared := workspacePrepareRun(t, fixture, project, run)
	workspaceWriteFile(t, core.PathJoin(prepared.Path, "README.md"), "retained change\n")
	fixture.runner.setFailPush(true)

	result := fixture.manager.CaptureRun(context.Background(), prepared)
	core.AssertTrue(t, result.OK, result.Error())
	capture := result.Value.(Capture)
	core.AssertTrue(t, capture.Changed)
	core.AssertFalse(t, capture.Pushed)
	core.AssertTrue(t, capture.Retained)
	core.AssertContains(t, capture.Summary, "push")
	core.AssertTrue(t, core.Stat(prepared.Path).OK)
	core.AssertFalse(t, fixture.manager.ReleaseRun(context.Background(), prepared).OK)
}

func TestWorkspaceCaptureRunFailures(t *testing.T) {
	stages := []string{"server", "server-shape", "credentials", "status", "add", "identity-name", "identity-email", "commit", "revision"}
	for _, stage := range stages {
		t.Run(stage, func(t *testing.T) {
			fixture := workspaceNewRunFixture(t)
			workspaceWriteFile(t, core.PathJoin(fixture.prepared.Path, "README.md"), "capture fault\n")
			switch stage {
			case "server":
				fixture.server.setFailure(false, true)
			case "server-shape":
				fixture.server.setEnsureValue("not a repository")
			case "credentials":
				fixture.server.setRepository(gitserver.Repository{Name: "project", CloneURL: "/tmp/project.git", IdentityFile: "/tmp/id"})
			case "status":
				fixture.runner.setFailure(func(command Command) bool { return workspaceCommandContains(command, "status\x00--porcelain=v1") })
			case "add":
				fixture.runner.setFailure(func(command Command) bool { return len(command.Args) > 0 && command.Args[0] == "add" })
			case "identity-name", "identity-email":
				workspaceRunGit(t, fixture.runner, fixture.prepared.Path, "config", "--local", "--unset-all", "user.name")
				workspaceRunGit(t, fixture.runner, fixture.prepared.Path, "config", "--local", "--unset-all", "user.email")
				fixture.runner.setFailure(func(command Command) bool {
					if stage == "identity-name" {
						return workspaceCommandContains(command, "config\x00--local\x00user.name")
					}
					return workspaceCommandContains(command, "config\x00--local\x00user.email")
				})
			case "commit":
				fixture.runner.setFailure(func(command Command) bool { return len(command.Args) > 0 && command.Args[0] == "commit" })
			case "revision":
				fixture.runner.setFailure(func(command Command) bool {
					return command.Dir == fixture.prepared.Path && workspaceCommandContains(command, "rev-parse\x00HEAD")
				})
			}
			result := fixture.manager.CaptureRun(context.Background(), fixture.prepared)
			core.AssertTrue(t, result.OK, result.Error())
			capture := result.Value.(Capture)
			core.AssertTrue(t, capture.Retained)
			core.AssertFalse(t, capture.Pushed)
			core.AssertTrue(t, capture.Summary != "")
			core.AssertTrue(t, core.Stat(fixture.prepared.Path).OK)
		})
	}
}

func TestWorkspace_Manager_ReconstructRun_Good(t *testing.T) {
	fixture := workspaceNewFixture(t)
	source := core.PathJoin(t.TempDir(), "reconstruct source")
	workspaceCreateRepository(t, fixture.runner, source)
	project := workspaceRegisterRepository(t, fixture, source, "project", "project")
	run := work.Run{ID: "run", WorkID: "work", ProjectID: project.ID, Number: 1, SourceRevision: project.SourceRevision}
	prepared := workspacePrepareRun(t, fixture, project, run)
	workspaceWriteFile(t, core.PathJoin(prepared.Path, "restored.txt"), "durable\n")
	capture := fixture.manager.CaptureRun(context.Background(), prepared).Value.(Capture)
	core.AssertTrue(t, fixture.manager.ReleaseRun(context.Background(), prepared).OK)
	core.AssertFalse(t, core.Stat(prepared.Path).OK)

	reopened := NewManager(ManagerOptions{
		Root: fixture.root, Files: fixture.files, Git: fixture.runner, Server: fixture.server,
		IDs: func() string { return "reopen" }, Now: time.Now,
	}).Value.(*Manager)
	run.Branch = prepared.Branch
	run.Worktree = prepared.Path
	run.DurableRevision = capture.DurableRevision
	result := reopened.ReconstructRun(context.Background(), project, run)
	core.AssertTrue(t, result.OK, result.Error())
	reconstructed := result.Value.(RunWorkspace)
	core.AssertEqual(t, prepared.Path, reconstructed.Path)
	core.AssertEqual(t, capture.Revision, reconstructed.BaseRevision)
	core.AssertTrue(t, core.Stat(core.PathJoin(reconstructed.Path, "restored.txt")).OK)
}

func TestWorkspaceReconstructRetainedCaptureRefreshesLease(t *testing.T) {
	fixture := workspaceNewRunFixture(t)
	workspaceWriteFile(t, core.PathJoin(fixture.prepared.Path, "first.txt"), "first\n")
	firstResult := fixture.manager.CaptureRun(context.Background(), fixture.prepared)
	core.AssertTrue(t, firstResult.OK, firstResult.Error())
	first := firstResult.Value.(Capture)
	core.AssertTrue(t, first.Pushed)
	core.AssertEqual(t, first.Revision, first.DurableRevision)
	core.AssertTrue(t, core.Stat(fixture.prepared.Path).OK)

	reopened := NewManager(ManagerOptions{
		Root: fixture.root, Files: fixture.files, Git: fixture.runner, Server: fixture.server,
		IDs: func() string { return "reopen" }, Now: time.Now,
	}).Value.(*Manager)
	child := fixture.run
	child.ID = "child"
	child.ParentRunID = fixture.run.ID
	child.Branch = fixture.prepared.Branch
	child.Worktree = fixture.prepared.Path
	child.DurableRevision = first.DurableRevision
	reconstructedResult := reopened.ReconstructRun(context.Background(), fixture.project, child)
	core.AssertTrue(t, reconstructedResult.OK, reconstructedResult.Error())
	reconstructed := reconstructedResult.Value.(RunWorkspace)
	core.AssertEqual(t, first.Revision, reconstructed.BaseRevision)
	core.AssertEqual(t, first.DurableRevision, reconstructed.DurableRevision)

	workspaceWriteFile(t, core.PathJoin(reconstructed.Path, "second.txt"), "second\n")
	secondResult := reopened.CaptureRun(context.Background(), reconstructed)
	core.AssertTrue(t, secondResult.OK, secondResult.Error())
	second := secondResult.Value.(Capture)
	core.AssertTrue(t, second.Pushed, second.Summary)
	core.AssertTrue(t, second.Revision != first.Revision)
	core.AssertEqual(t, second.Revision, second.DurableRevision)
	remote := fixture.server.EnsureRepository(context.Background(), fixture.project.RepositoryName).Value.(gitserver.Repository)
	core.AssertEqual(t, second.Revision, workspaceRunGit(t, fixture.runner, remote.CloneURL, "rev-parse", core.Concat("refs/heads/", reconstructed.Branch)))
}

func TestWorkspaceReconstructRetainedRecoversTrackingRefreshFailure(t *testing.T) {
	fixture := workspaceNewRunFixture(t)
	workspaceWriteFile(t, core.PathJoin(fixture.prepared.Path, "first.txt"), "first\n")
	fixture.runner.setFailure(func(command Command) bool {
		return workspaceContainsArgument(command.Args, "update-ref") || workspaceContainsArgument(command.Args, "fetch")
	})
	firstResult := fixture.manager.CaptureRun(context.Background(), fixture.prepared)
	core.AssertTrue(t, firstResult.OK, firstResult.Error())
	first := firstResult.Value.(Capture)
	core.AssertFalse(t, first.Pushed)
	core.AssertTrue(t, first.Retained)
	core.AssertEqual(t, first.Revision, first.DurableRevision)
	core.AssertContains(t, first.Summary, "tracking")
	core.AssertFalse(t, fixture.manager.ReleaseRun(context.Background(), fixture.prepared).OK)
	fixture.runner.setFailure(nil)

	reopened := NewManager(ManagerOptions{
		Root: fixture.root, Files: fixture.files, Git: fixture.runner, Server: fixture.server,
		IDs: func() string { return "reopen" }, Now: time.Now,
	}).Value.(*Manager)
	child := fixture.run
	child.ID = "child"
	child.ParentRunID = fixture.run.ID
	child.Branch = fixture.prepared.Branch
	child.Worktree = fixture.prepared.Path
	child.DurableRevision = first.DurableRevision
	reconstructedResult := reopened.ReconstructRun(context.Background(), fixture.project, child)
	core.AssertTrue(t, reconstructedResult.OK, reconstructedResult.Error())
	reconstructed := reconstructedResult.Value.(RunWorkspace)
	core.AssertEqual(t, first.Revision, reconstructed.BaseRevision)
	core.AssertEqual(t, first.DurableRevision, reconstructed.DurableRevision)

	workspaceWriteFile(t, core.PathJoin(reconstructed.Path, "second.txt"), "second\n")
	secondResult := reopened.CaptureRun(context.Background(), reconstructed)
	core.AssertTrue(t, secondResult.OK, secondResult.Error())
	second := secondResult.Value.(Capture)
	core.AssertTrue(t, second.Pushed, second.Summary)
	core.AssertEqual(t, second.Revision, second.DurableRevision)
}

func TestWorkspaceReconstructRetainedRejectsAcknowledgedPushLoss(t *testing.T) {
	t.Run("deleted after initial acknowledged push", func(t *testing.T) {
		fixture := workspaceNewRunFixture(t)
		workspaceWriteFile(t, core.PathJoin(fixture.prepared.Path, "first.txt"), "first\n")
		fixture.runner.setFailure(func(command Command) bool {
			return workspaceContainsArgument(command.Args, "update-ref") || workspaceContainsArgument(command.Args, "fetch")
		})
		captureResult := fixture.manager.CaptureRun(context.Background(), fixture.prepared)
		core.AssertTrue(t, captureResult.OK, captureResult.Error())
		capture := captureResult.Value.(Capture)
		core.AssertFalse(t, capture.Pushed)
		core.AssertEqual(t, capture.Revision, capture.DurableRevision)
		core.AssertContains(t, capture.Summary, "tracking")
		fixture.runner.setFailure(nil)
		remote := fixture.server.EnsureRepository(context.Background(), fixture.project.RepositoryName).Value.(gitserver.Repository)
		remoteRef := core.Concat("refs/heads/", fixture.prepared.Branch)
		core.AssertEqual(t, capture.Revision, workspaceRunGit(t, fixture.runner, remote.CloneURL, "rev-parse", remoteRef))
		workspaceRunGit(t, fixture.runner, remote.CloneURL, "update-ref", "-d", remoteRef)

		reopened := NewManager(ManagerOptions{
			Root: fixture.root, Files: fixture.files, Git: fixture.runner, Server: fixture.server,
			IDs: func() string { return "reopen" }, Now: time.Now,
		}).Value.(*Manager)
		child := fixture.run
		child.ID = "child"
		child.ParentRunID = fixture.run.ID
		child.Branch = fixture.prepared.Branch
		child.Worktree = fixture.prepared.Path
		child.DurableRevision = capture.DurableRevision
		result := reopened.ReconstructRun(context.Background(), fixture.project, child)
		core.AssertFalse(t, result.OK)
		core.AssertContains(t, result.Error(), "differs")
	})

	t.Run("reset after later acknowledged push", func(t *testing.T) {
		fixture := workspaceNewRunFixture(t)
		workspaceWriteFile(t, core.PathJoin(fixture.prepared.Path, "first.txt"), "first\n")
		firstResult := fixture.manager.CaptureRun(context.Background(), fixture.prepared)
		core.AssertTrue(t, firstResult.OK, firstResult.Error())
		first := firstResult.Value.(Capture)
		core.AssertTrue(t, first.Pushed, first.Summary)
		core.AssertEqual(t, first.Revision, first.DurableRevision)

		workspaceWriteFile(t, core.PathJoin(fixture.prepared.Path, "second.txt"), "second\n")
		fixture.runner.setFailure(func(command Command) bool {
			return workspaceContainsArgument(command.Args, "update-ref") || workspaceContainsArgument(command.Args, "fetch")
		})
		captureResult := fixture.manager.CaptureRun(context.Background(), fixture.prepared)
		core.AssertTrue(t, captureResult.OK, captureResult.Error())
		capture := captureResult.Value.(Capture)
		core.AssertFalse(t, capture.Pushed)
		core.AssertTrue(t, capture.Revision != first.Revision)
		core.AssertEqual(t, capture.Revision, capture.DurableRevision)
		core.AssertContains(t, capture.Summary, "tracking")
		fixture.runner.setFailure(nil)
		remote := fixture.server.EnsureRepository(context.Background(), fixture.project.RepositoryName).Value.(gitserver.Repository)
		remoteRef := core.Concat("refs/heads/", fixture.prepared.Branch)
		core.AssertEqual(t, capture.Revision, workspaceRunGit(t, fixture.runner, remote.CloneURL, "rev-parse", remoteRef))
		workspaceRunGit(t, fixture.runner, remote.CloneURL, "update-ref", remoteRef, first.Revision)

		reopened := NewManager(ManagerOptions{
			Root: fixture.root, Files: fixture.files, Git: fixture.runner, Server: fixture.server,
			IDs: func() string { return "reopen" }, Now: time.Now,
		}).Value.(*Manager)
		child := fixture.run
		child.ID = "child"
		child.ParentRunID = fixture.run.ID
		child.Branch = fixture.prepared.Branch
		child.Worktree = fixture.prepared.Path
		child.DurableRevision = capture.DurableRevision
		result := reopened.ReconstructRun(context.Background(), fixture.project, child)
		core.AssertFalse(t, result.OK)
		core.AssertContains(t, result.Error(), "differs")
	})
}

func TestWorkspaceCaptureAdvancesAcknowledgedLeaseBeforeTrackingRecovery(t *testing.T) {
	fixture := workspaceNewRunFixture(t)
	workspaceWriteFile(t, core.PathJoin(fixture.prepared.Path, "first.txt"), "first\n")
	fixture.runner.setFailure(func(command Command) bool {
		return workspaceContainsArgument(command.Args, "update-ref") || workspaceContainsArgument(command.Args, "fetch")
	})
	firstResult := fixture.manager.CaptureRun(context.Background(), fixture.prepared)
	core.AssertTrue(t, firstResult.OK, firstResult.Error())
	first := firstResult.Value.(Capture)
	core.AssertFalse(t, first.Pushed)
	core.AssertEqual(t, first.Revision, first.DurableRevision)
	core.AssertContains(t, first.Summary, "tracking")
	fixture.runner.setFailure(nil)

	workspaceWriteFile(t, core.PathJoin(fixture.prepared.Path, "second.txt"), "second\n")
	secondResult := fixture.manager.CaptureRun(context.Background(), fixture.prepared)
	core.AssertTrue(t, secondResult.OK, secondResult.Error())
	second := secondResult.Value.(Capture)
	core.AssertTrue(t, second.Pushed, second.Summary)
	core.AssertTrue(t, second.Revision != first.Revision)
	lease := core.Concat("--force-with-lease=refs/heads/", fixture.prepared.Branch, ":", first.Revision)
	core.AssertTrue(t, workspaceContainsArgument(workspaceLastPushArguments(t, fixture.runner), lease))
}

func TestWorkspaceReconstructRetainedPublishesFailedInitialPush(t *testing.T) {
	fixture := workspaceNewRunFixture(t)
	workspaceWriteFile(t, core.PathJoin(fixture.prepared.Path, "first.txt"), "first\n")
	fixture.runner.setFailPush(true)
	firstResult := fixture.manager.CaptureRun(context.Background(), fixture.prepared)
	core.AssertTrue(t, firstResult.OK, firstResult.Error())
	first := firstResult.Value.(Capture)
	core.AssertFalse(t, first.Pushed)
	core.AssertTrue(t, first.Retained)
	core.AssertEqual(t, "", first.DurableRevision)
	fixture.runner.setFailPush(false)
	remote := fixture.server.EnsureRepository(context.Background(), fixture.project.RepositoryName).Value.(gitserver.Repository)
	remoteRef := core.Concat("refs/heads/", fixture.prepared.Branch)
	core.AssertEqual(t, "", workspaceRunGit(t, fixture.runner, fixture.root, "ls-remote", "--heads", remote.CloneURL, remoteRef))

	reopened := NewManager(ManagerOptions{
		Root: fixture.root, Files: fixture.files, Git: fixture.runner, Server: fixture.server,
		IDs: func() string { return "reopen" }, Now: time.Now,
	}).Value.(*Manager)
	child := fixture.run
	child.ID = "child"
	child.ParentRunID = fixture.run.ID
	child.Branch = fixture.prepared.Branch
	child.Worktree = fixture.prepared.Path
	child.DurableRevision = first.DurableRevision
	reconstructedResult := reopened.ReconstructRun(context.Background(), fixture.project, child)
	core.AssertTrue(t, reconstructedResult.OK, reconstructedResult.Error())
	reconstructed := reconstructedResult.Value.(RunWorkspace)
	core.AssertEqual(t, first.Revision, reconstructed.BaseRevision)
	core.AssertEqual(t, "", reconstructed.DurableRevision)

	secondResult := reopened.CaptureRun(context.Background(), reconstructed)
	core.AssertTrue(t, secondResult.OK, secondResult.Error())
	second := secondResult.Value.(Capture)
	core.AssertTrue(t, second.Pushed, second.Summary)
	core.AssertEqual(t, first.Revision, second.Revision)
	core.AssertEqual(t, second.Revision, second.DurableRevision)
	core.AssertEqual(t, first.Revision, workspaceRunGit(t, fixture.runner, remote.CloneURL, "rev-parse", remoteRef))
	lease := core.Concat("--force-with-lease=", remoteRef, ":")
	core.AssertTrue(t, workspaceContainsArgument(workspaceLastPushArguments(t, fixture.runner), lease))
}

func TestWorkspaceReconstructRetainedPublishesFailedSubsequentPush(t *testing.T) {
	fixture := workspaceNewRunFixture(t)
	workspaceWriteFile(t, core.PathJoin(fixture.prepared.Path, "first.txt"), "first\n")
	firstResult := fixture.manager.CaptureRun(context.Background(), fixture.prepared)
	core.AssertTrue(t, firstResult.OK, firstResult.Error())
	first := firstResult.Value.(Capture)
	core.AssertTrue(t, first.Pushed, first.Summary)
	core.AssertEqual(t, first.Revision, first.DurableRevision)

	workspaceWriteFile(t, core.PathJoin(fixture.prepared.Path, "second.txt"), "second\n")
	fixture.runner.setFailPush(true)
	failedResult := fixture.manager.CaptureRun(context.Background(), fixture.prepared)
	core.AssertTrue(t, failedResult.OK, failedResult.Error())
	failed := failedResult.Value.(Capture)
	core.AssertFalse(t, failed.Pushed)
	core.AssertTrue(t, failed.Retained)
	core.AssertTrue(t, failed.Revision != first.Revision)
	core.AssertEqual(t, first.DurableRevision, failed.DurableRevision)
	fixture.runner.setFailPush(false)

	reopened := NewManager(ManagerOptions{
		Root: fixture.root, Files: fixture.files, Git: fixture.runner, Server: fixture.server,
		IDs: func() string { return "reopen" }, Now: time.Now,
	}).Value.(*Manager)
	child := fixture.run
	child.ID = "child"
	child.ParentRunID = fixture.run.ID
	child.Branch = fixture.prepared.Branch
	child.Worktree = fixture.prepared.Path
	child.DurableRevision = failed.DurableRevision
	reconstructedResult := reopened.ReconstructRun(context.Background(), fixture.project, child)
	core.AssertTrue(t, reconstructedResult.OK, reconstructedResult.Error())
	reconstructed := reconstructedResult.Value.(RunWorkspace)
	core.AssertEqual(t, failed.Revision, reconstructed.BaseRevision)
	core.AssertEqual(t, first.DurableRevision, reconstructed.DurableRevision)

	secondResult := reopened.CaptureRun(context.Background(), reconstructed)
	core.AssertTrue(t, secondResult.OK, secondResult.Error())
	second := secondResult.Value.(Capture)
	core.AssertTrue(t, second.Pushed, second.Summary)
	core.AssertEqual(t, failed.Revision, second.Revision)
	core.AssertEqual(t, second.Revision, second.DurableRevision)
	remoteRef := core.Concat("refs/heads/", reconstructed.Branch)
	lease := core.Concat("--force-with-lease=", remoteRef, ":", first.Revision)
	core.AssertTrue(t, workspaceContainsArgument(workspaceLastPushArguments(t, fixture.runner), lease))
}

func TestWorkspaceReconstructRetainedRejectsRemoteDivergence(t *testing.T) {
	for _, test := range []struct {
		name   string
		mutate func(*testing.T, workspaceRunFixture, gitserver.Repository)
	}{
		{
			name: "changed",
			mutate: func(t *testing.T, fixture workspaceRunFixture, remote gitserver.Repository) {
				workspaceRunGit(t, fixture.runner, remote.CloneURL, "update-ref", core.Concat("refs/heads/", fixture.prepared.Branch), fixture.project.SourceRevision)
			},
		},
		{
			name: "deleted",
			mutate: func(t *testing.T, fixture workspaceRunFixture, remote gitserver.Repository) {
				workspaceRunGit(t, fixture.runner, remote.CloneURL, "update-ref", "-d", core.Concat("refs/heads/", fixture.prepared.Branch))
			},
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			fixture := workspaceNewRunFixture(t)
			workspaceWriteFile(t, core.PathJoin(fixture.prepared.Path, "first.txt"), "first\n")
			captureResult := fixture.manager.CaptureRun(context.Background(), fixture.prepared)
			core.AssertTrue(t, captureResult.OK, captureResult.Error())
			capture := captureResult.Value.(Capture)
			core.AssertTrue(t, capture.Pushed, capture.Summary)
			core.AssertEqual(t, capture.Revision, capture.DurableRevision)
			remote := fixture.server.EnsureRepository(context.Background(), fixture.project.RepositoryName).Value.(gitserver.Repository)
			test.mutate(t, fixture, remote)

			reopened := NewManager(ManagerOptions{
				Root: fixture.root, Files: fixture.files, Git: fixture.runner, Server: fixture.server,
				IDs: func() string { return "reopen" }, Now: time.Now,
			}).Value.(*Manager)
			child := fixture.run
			child.ID = "child"
			child.ParentRunID = fixture.run.ID
			child.Branch = fixture.prepared.Branch
			child.Worktree = fixture.prepared.Path
			child.DurableRevision = capture.DurableRevision
			result := reopened.ReconstructRun(context.Background(), fixture.project, child)
			core.AssertFalse(t, result.OK)
			core.AssertContains(t, result.Error(), "differs")
			core.AssertTrue(t, core.Stat(fixture.prepared.Path).OK)
		})
	}
}

func TestWorkspace_Manager_ReconstructRun_Bad(t *testing.T) {
	var manager *Manager
	core.AssertFalse(t, manager.ReconstructRun(context.Background(), work.Project{}, work.Run{}).OK)
	fixture := workspaceNewFixture(t)
	core.AssertFalse(t, fixture.manager.ReconstructRun(nil, work.Project{}, work.Run{}).OK)
	core.AssertFalse(t, fixture.manager.ReconstructRun(context.Background(), work.Project{}, work.Run{}).OK)
}

func TestWorkspace_Manager_ReconstructRun_Ugly(t *testing.T) {
	fixture := workspaceNewFixture(t)
	source := core.PathJoin(t.TempDir(), "wrong branch source")
	workspaceCreateRepository(t, fixture.runner, source)
	project := workspaceRegisterRepository(t, fixture, source, "project", "project")
	run := work.Run{ID: "run", WorkID: "work", ProjectID: project.ID, Number: 1, SourceRevision: project.SourceRevision}
	prepared := workspacePrepareRun(t, fixture, project, run)
	workspaceRunGit(t, fixture.runner, prepared.Path, "checkout", "-b", "wrong-branch")

	reopened := NewManager(ManagerOptions{
		Root: fixture.root, Files: fixture.files, Git: fixture.runner, Server: fixture.server,
		IDs: func() string { return "reopen" }, Now: time.Now,
	}).Value.(*Manager)
	run.Branch = prepared.Branch
	run.Worktree = prepared.Path
	core.AssertFalse(t, reopened.ReconstructRun(context.Background(), project, run).OK)
	core.AssertTrue(t, core.Stat(prepared.Path).OK)
}

func TestWorkspaceReconstructRetainedAndLeaseEdges(t *testing.T) {
	retained := workspaceNewRunFixture(t)
	reopened := NewManager(ManagerOptions{
		Root: retained.root, Files: retained.files, Git: retained.runner, Server: retained.server,
		IDs: func() string { return "reopen" }, Now: time.Now,
	}).Value.(*Manager)
	retained.run.Branch = retained.prepared.Branch
	retained.run.Worktree = retained.prepared.Path
	result := reopened.ReconstructRun(context.Background(), retained.project, retained.run)
	core.AssertTrue(t, result.OK, result.Error())
	core.AssertEqual(t, retained.prepared.Path, result.Value.(RunWorkspace).Path)
	core.AssertTrue(t, reopened.ReconstructRun(context.Background(), retained.project, retained.run).OK)

	reopened.leases[retained.run.ID] = RunWorkspace{
		Project: retained.project, RunID: retained.run.ID, Branch: retained.prepared.Branch,
		Path: core.PathJoin(retained.root, "wrong"),
	}
	core.AssertFalse(t, reopened.ReconstructRun(context.Background(), retained.project, retained.run).OK)

	attached := workspaceNewRunFixture(t)
	attachedManager := NewManager(ManagerOptions{
		Root: attached.root, Files: attached.files, Git: attached.runner, Server: attached.server,
		IDs: func() string { return "reopen" }, Now: time.Now,
	}).Value.(*Manager)
	child := attached.run
	child.ID = "child"
	child.Branch = attached.prepared.Branch
	child.Worktree = ""
	core.AssertFalse(t, attachedManager.ReconstructRun(context.Background(), attached.project, child).OK)
}

func TestWorkspaceReconstructRetainedRevisionFailure(t *testing.T) {
	fixture := workspaceNewRunFixture(t)
	reopened := NewManager(ManagerOptions{
		Root: fixture.root, Files: fixture.files, Git: fixture.runner, Server: fixture.server,
		IDs: func() string { return "reopen" }, Now: time.Now,
	}).Value.(*Manager)
	fixture.run.Branch = fixture.prepared.Branch
	fixture.run.Worktree = fixture.prepared.Path
	fixture.runner.setFailure(func(command Command) bool {
		return command.Dir == fixture.prepared.Path && workspaceCommandContains(command, "rev-parse\x00HEAD")
	})
	core.AssertFalse(t, reopened.ReconstructRun(context.Background(), fixture.project, fixture.run).OK)
}

func TestWorkspaceReconstructRunFailures(t *testing.T) {
	stages := []string{"cancelled", "list", "server", "server-shape", "credentials", "remote", "branch", "ensure", "add", "add-cleanup", "revision"}
	for _, stage := range stages {
		t.Run(stage, func(t *testing.T) {
			fixture := workspaceNewDurableRunFixture(t)
			ctx := context.Background()
			switch stage {
			case "cancelled":
				cancelled, cancel := context.WithCancel(context.Background())
				cancel()
				ctx = cancelled
			case "list":
				fixture.runner.setFailure(func(command Command) bool { return workspaceCommandContains(command, "worktree\x00list") })
			case "server":
				fixture.server.setFailure(false, true)
			case "server-shape":
				fixture.server.setEnsureValue("not a repository")
			case "credentials":
				fixture.server.setRepository(gitserver.Repository{Name: "project", CloneURL: "/tmp/project.git", IdentityFile: "/tmp/id"})
			case "remote":
				fixture.runner.setFailure(func(command Command) bool { return workspaceContainsArgument(command.Args, "ls-remote") })
			case "branch":
				fixture.runner.setFailure(func(command Command) bool { return workspaceCommandContains(command, "branch\x00--force") })
			case "ensure":
				fixture.manager.files = &workspaceFaultMedium{Medium: fixture.files, failEnsure: core.PathJoin("project", "runs", "run")}
			case "add", "add-cleanup":
				fixture.runner.setFailure(func(command Command) bool { return workspaceCommandContains(command, "worktree\x00add") })
				if stage == "add-cleanup" {
					fixture.manager.files = &workspaceFaultMedium{Medium: fixture.files, failDelete: core.PathJoin("project", "runs", "run")}
				}
			case "revision":
				fixture.runner.setFailure(func(command Command) bool {
					return command.Dir == fixture.prepared.Path && workspaceCommandContains(command, "rev-parse\x00HEAD")
				})
			}
			result := fixture.manager.ReconstructRun(ctx, fixture.project, fixture.run)
			core.AssertFalse(t, result.OK)
		})
	}
}

func TestWorkspace_Manager_ReleaseRun_Good(t *testing.T) {
	fixture := workspaceNewFixture(t)
	source := core.PathJoin(t.TempDir(), "release source")
	workspaceCreateRepository(t, fixture.runner, source)
	project := workspaceRegisterRepository(t, fixture, source, "project", "project")
	run := work.Run{ID: "run", WorkID: "work", ProjectID: project.ID, Number: 1, SourceRevision: project.SourceRevision}
	prepared := workspacePrepareRun(t, fixture, project, run)
	capture := fixture.manager.CaptureRun(context.Background(), prepared)
	core.AssertTrue(t, capture.OK, capture.Error())

	result := fixture.manager.ReleaseRun(context.Background(), prepared)
	core.AssertTrue(t, result.OK, result.Error())
	core.AssertFalse(t, core.Stat(prepared.Path).OK)
	core.AssertEqual(t, 0, len(fixture.manager.leases))
}

func TestWorkspace_Manager_ReleaseRun_Bad(t *testing.T) {
	var manager *Manager
	core.AssertFalse(t, manager.ReleaseRun(context.Background(), RunWorkspace{}).OK)
	fixture := workspaceNewFixture(t)
	core.AssertFalse(t, fixture.manager.ReleaseRun(nil, RunWorkspace{}).OK)
	core.AssertFalse(t, fixture.manager.ReleaseRun(context.Background(), RunWorkspace{RunID: "unknown", Path: fixture.root}).OK)
}

func TestWorkspace_Manager_ReleaseRun_Ugly(t *testing.T) {
	fixture := workspaceNewFixture(t)
	source := core.PathJoin(t.TempDir(), "retained source")
	workspaceCreateRepository(t, fixture.runner, source)
	project := workspaceRegisterRepository(t, fixture, source, "project", "project")
	run := work.Run{ID: "run", WorkID: "work", ProjectID: project.ID, Number: 1, SourceRevision: project.SourceRevision}
	prepared := workspacePrepareRun(t, fixture, project, run)
	result := fixture.manager.ReleaseRun(context.Background(), prepared)
	core.AssertFalse(t, result.OK)
	core.AssertTrue(t, core.Stat(prepared.Path).OK)
}

func TestWorkspaceReleaseRunFailures(t *testing.T) {
	stages := []string{"cancelled", "dirty", "status", "remove", "delete"}
	for _, stage := range stages {
		t.Run(stage, func(t *testing.T) {
			fixture := workspaceNewRunFixture(t)
			captured := fixture.manager.CaptureRun(context.Background(), fixture.prepared)
			core.AssertTrue(t, captured.OK, captured.Error())
			ctx := context.Background()
			switch stage {
			case "cancelled":
				cancelled, cancel := context.WithCancel(context.Background())
				cancel()
				ctx = cancelled
			case "dirty":
				workspaceWriteFile(t, core.PathJoin(fixture.prepared.Path, "dirty.txt"), "dirty\n")
			case "status":
				fixture.runner.setFailure(func(command Command) bool { return workspaceCommandContains(command, "status\x00--porcelain=v1") })
			case "remove":
				fixture.runner.setFailure(func(command Command) bool { return workspaceCommandContains(command, "worktree\x00remove") })
			case "delete":
				fixture.manager.files = &workspaceFaultMedium{Medium: fixture.files, failDelete: core.PathJoin("project", "runs", "run")}
			}
			result := fixture.manager.ReleaseRun(ctx, fixture.prepared)
			core.AssertFalse(t, result.OK)
		})
	}
}

func TestWorkspaceReleaseRunClearsAliases(t *testing.T) {
	fixture := workspaceNewRunFixture(t)
	captured := fixture.manager.CaptureRun(context.Background(), fixture.prepared)
	core.AssertTrue(t, captured.OK, captured.Error())
	alias := fixture.prepared
	alias.RunID = "alias"
	fixture.manager.leases[alias.RunID] = alias
	fixture.manager.durable[alias.RunID] = true
	result := fixture.manager.ReleaseRun(context.Background(), fixture.prepared)
	core.AssertTrue(t, result.OK, result.Error())
	core.AssertEqual(t, 0, len(fixture.manager.leases))
	core.AssertEqual(t, 0, len(fixture.manager.durable))
}

func TestWorkspacePathAndCredentialSafety(t *testing.T) {
	fixture := workspaceNewFixture(t)
	core.AssertFalse(t, fixture.manager.internalAbsolute("relative").OK)
	core.AssertFalse(t, fixture.manager.internalAbsolute(core.PathJoin(t.TempDir(), "outside")).OK)
	core.AssertFalse(t, fixture.manager.internalRelative("relative").OK)

	filePath := core.PathJoin(t.TempDir(), "not-a-directory")
	workspaceWriteFile(t, filePath, "file\n")
	core.AssertFalse(t, canonicalDirectory("").OK)
	core.AssertFalse(t, canonicalDirectory(core.PathJoin(t.TempDir(), "missing")).OK)
	core.AssertFalse(t, canonicalDirectory(filePath).OK)

	core.AssertFalse(t, runBranch("work", 0).OK)
	core.AssertFalse(t, runBranch("...", 1).OK)
	core.AssertEqual(t, "alpha-beta", branchComponent(" ../alpha///beta.. "))

	emptyEnvironment := repositoryEnvironment(gitserver.Repository{})
	core.AssertTrue(t, emptyEnvironment.OK, emptyEnvironment.Error())
	core.AssertEqual(t, 0, len(emptyEnvironment.Value.([]string)))
	core.AssertFalse(t, repositoryEnvironment(gitserver.Repository{IdentityFile: "/tmp/id"}).OK)
	core.AssertFalse(t, repositoryEnvironment(gitserver.Repository{IdentityFile: "relative", KnownHostsFile: "/tmp/hosts"}).OK)
	credentials := repositoryEnvironment(gitserver.Repository{
		IdentityFile:   "/tmp/key with ' quote",
		KnownHostsFile: "/tmp/known hosts",
	})
	core.AssertTrue(t, credentials.OK, credentials.Error())
	command := credentials.Value.([]string)[0]
	core.AssertContains(t, command, "IdentitiesOnly=yes")
	core.AssertContains(t, command, `'"'"'`)
	core.AssertContains(t, command, "StrictHostKeyChecking=yes")

	core.AssertEqual(t, "/tmp/one", attachedBranchPath("worktree /tmp/one\nbranch refs/heads/lem/work/a/run-1\n", "lem/work/a/run-1"))
	core.AssertEqual(t, "", attachedBranchPath("branch refs/heads/lem/work/a/run-1\n", "lem/work/a/run-1"))
}

func TestWorkspaceIncludedHashSafety(t *testing.T) {
	root := t.TempDir()
	workspaceWriteFile(t, core.PathJoin(root, "one.txt"), "one\n")
	first := includedHash(root, []string{"one.txt"})
	core.AssertTrue(t, first.OK, first.Error())
	workspaceWriteFile(t, core.PathJoin(root, "one.txt"), "two\n")
	second := includedHash(root, []string{"one.txt"})
	core.AssertTrue(t, second.OK, second.Error())
	core.AssertTrue(t, first.Value.(string) != second.Value.(string))
	core.AssertFalse(t, includedHash(root, []string{"../escape"}).OK)
	core.AssertFalse(t, includedHash(root, []string{"missing.txt"}).OK)
}

func TestWorkspaceLeaseValidationAndCleanup(t *testing.T) {
	fixture := workspaceNewFixture(t)
	lease := RunWorkspace{
		Project: work.Project{ID: "project", ClonePath: core.PathJoin(fixture.root, "project", "repo.git")},
		RunID:   "run", Branch: "lem/work/work/run-1",
		Path: core.PathJoin(fixture.root, "project", "runs", "run", "worktree"),
	}
	fixture.manager.leases[lease.RunID] = lease
	core.AssertFalse(t, fixture.manager.validLease(RunWorkspace{RunID: "missing"}).OK)
	wrong := lease
	wrong.Path = core.PathJoin(fixture.root, "another")
	core.AssertFalse(t, fixture.manager.validLease(wrong).OK)
	fixture.manager.leases[lease.RunID] = RunWorkspace{RunID: lease.RunID, Path: core.PathJoin(t.TempDir(), "outside"), Branch: lease.Branch, Project: lease.Project}
	core.AssertFalse(t, fixture.manager.validLease(fixture.manager.leases[lease.RunID]).OK)

	worktreeRelative := core.PathJoin("project", "runs", "failed", "worktree")
	core.AssertTrue(t, fixture.files.EnsureDir(worktreeRelative) == nil)
	cleaned := fixture.manager.cleanupFailedWorktree(context.Background(), lease.Project.ClonePath, core.PathJoin(fixture.root, worktreeRelative), worktreeRelative, "", false)
	core.AssertFalse(t, cleaned.OK)
	core.AssertFalse(t, fixture.files.Exists(core.PathDir(worktreeRelative)))

	fault := &workspaceFaultMedium{Medium: fixture.files, failDelete: core.PathDir(worktreeRelative)}
	fixture.manager.files = fault
	core.AssertFalse(t, fixture.manager.cleanupFailedWorktree(context.Background(), lease.Project.ClonePath, core.PathJoin(fixture.root, worktreeRelative), worktreeRelative, "", false).OK)

	fixture.manager.files = fixture.files
	branchRelative := core.PathJoin("project", "runs", "branch-failure", "worktree")
	core.AssertTrue(t, fixture.files.EnsureDir(branchRelative) == nil)
	fixture.runner.setValue(func(Command) (any, bool) { return "", true })
	fixture.runner.setFailure(func(command Command) bool {
		return workspaceCommandContains(command, "branch\x00--list")
	})
	branchCleanup := fixture.manager.cleanupFailedWorktree(
		context.Background(), lease.Project.ClonePath, core.PathJoin(fixture.root, branchRelative), branchRelative,
		"lem/work/work/run-1", true,
	)
	core.AssertFalse(t, branchCleanup.OK)
	core.AssertContains(t, branchCleanup.Error(), "branch inspect")
	core.AssertFalse(t, fixture.files.Exists(core.PathDir(branchRelative)))
}

func TestWorkspaceGitOutputShape(t *testing.T) {
	fixture := workspaceNewFixture(t)
	fixture.runner.setValue(func(command Command) (any, bool) {
		if workspaceContainsArgument(command.Args, "shape") {
			return 17, true
		}
		return nil, false
	})
	core.AssertFalse(t, fixture.manager.gitOutput(context.Background(), fixture.root, nil, "shape").OK)
}
