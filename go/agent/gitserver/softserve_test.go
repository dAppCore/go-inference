// SPDX-License-Identifier: EUPL-1.2

package gitserver

import (
	"context"
	"testing"
	"time"

	core "dappco.re/go"
	command "dappco.re/go/process/exec"
)

func gitserverTestOptions(t *testing.T) Options {
	t.Helper()
	result := DefaultOptions(t.TempDir())
	if !result.OK {
		t.Fatalf("DefaultOptions failed: %s", result.Error())
	}
	options := result.Value.(Options)
	options.ListenAddress = "127.0.0.1:0"
	options.PublicURL = "ssh://127.0.0.1:0"
	return options
}

func gitserverTestService(t *testing.T) (*softServe, Repository) {
	t.Helper()
	options := gitserverTestOptions(t)
	result := NewSoftServe(options)
	if !result.OK {
		t.Fatalf("NewSoftServe failed: %s", result.Error())
	}
	service := result.Value.(*softServe)
	started := service.Start(context.Background())
	if !started.OK {
		t.Fatalf("Start failed: %s", started.Error())
	}
	t.Cleanup(func() {
		if closed := service.Close(); !closed.OK {
			t.Errorf("Close failed: %s", closed.Error())
		}
	})
	repositoryResult := service.EnsureRepository(context.Background(), "fixture")
	if !repositoryResult.OK {
		t.Fatalf("EnsureRepository failed: %s", repositoryResult.Error())
	}
	return service, repositoryResult.Value.(Repository)
}

func gitserverRunGit(t *testing.T, directory string, environment []string, args ...string) string {
	t.Helper()
	result := command.Command(context.Background(), "git", args...).
		WithDir(directory).
		WithEnv(environment).
		CombinedOutput()
	if !result.OK {
		t.Fatalf("git %v failed: %s", args, result.Error())
	}
	return core.Trim(string(result.Value.([]byte)))
}

func gitserverSSHEnvironment(repository Repository) []string {
	sshCommand := core.Concat(
		"ssh -i ", gitserverQuote(repository.IdentityFile),
		" -o IdentitiesOnly=yes",
		" -o UserKnownHostsFile=", gitserverQuote(repository.KnownHostsFile),
		" -o StrictHostKeyChecking=yes",
	)
	return []string{core.Concat("GIT_SSH_COMMAND=", sshCommand)}
}

func gitserverQuote(value string) string {
	return core.Concat("'", core.Replace(value, "'", `'"'"'`), "'")
}

func TestSoftserve_NewSoftServe_Good(t *testing.T) {
	options := gitserverTestOptions(t)
	result := NewSoftServe(options)
	core.AssertTrue(t, result.OK, result.Error())
	service := result.Value.(*softServe)
	core.AssertEqual(t, options.DataPath, service.options.DataPath)
	core.AssertEqual(t, "stopped", service.state)
	closed := service.Close()
	core.AssertTrue(t, closed.OK, closed.Error())
}

func TestSoftserve_NewSoftServe_Bad(t *testing.T) {
	options := gitserverTestOptions(t)
	options.ListenAddress = "0.0.0.0:23231"
	core.AssertFalse(t, NewSoftServe(options).OK)

	options = gitserverTestOptions(t)
	options.PublicURL = "ssh://example.com:23231"
	core.AssertFalse(t, NewSoftServe(options).OK)

	options = gitserverTestOptions(t)
	options.ShutdownTimeout = -time.Second
	core.AssertFalse(t, NewSoftServe(options).OK)
}

func TestSoftserve_NewSoftServe_Ugly(t *testing.T) {
	core.AssertFalse(t, NewSoftServe(Options{}).OK)
	options := gitserverTestOptions(t)
	options.PID = nil
	core.AssertFalse(t, NewSoftServe(options).OK)
	options = gitserverTestOptions(t)
	options.ProcessAlive = nil
	core.AssertFalse(t, NewSoftServe(options).OK)
}

func TestSoftserveLifecycle(t *testing.T) {
	options := gitserverTestOptions(t)
	service := NewSoftServe(options).Value.(*softServe)

	health := service.Health(context.Background())
	core.AssertTrue(t, health.OK, health.Error())
	core.AssertFalse(t, health.Value.(Health).Running)

	started := service.Start(context.Background())
	core.AssertTrue(t, started.OK, started.Error())
	core.AssertTrue(t, service.Start(context.Background()).OK)

	health = service.Health(context.Background())
	core.AssertTrue(t, health.OK, health.Error())
	status := health.Value.(Health)
	core.AssertTrue(t, status.Running)
	core.AssertTrue(t, core.HasPrefix(status.Address, "127.0.0.1:"))
	core.AssertFalse(t, core.Contains(status.Address, "0.0.0.0"))

	closed := service.Close()
	core.AssertTrue(t, closed.OK, closed.Error())
	core.AssertTrue(t, service.Close().OK)
	core.AssertFalse(t, service.Health(context.Background()).Value.(Health).Running)
	core.AssertFalse(t, service.Start(nil).OK)
	core.AssertFalse(t, service.Health(nil).OK)
}

func TestSoftserveEnsureRepository(t *testing.T) {
	options := gitserverTestOptions(t)
	service := NewSoftServe(options).Value.(*softServe)
	t.Cleanup(func() { core.AssertTrue(t, service.Close().OK) })

	result := service.EnsureRepository(context.Background(), " Team/App.git ")
	core.AssertTrue(t, result.OK, result.Error())
	repository := result.Value.(Repository)
	core.AssertEqual(t, "Team/App", repository.Name)
	core.AssertTrue(t, core.HasPrefix(repository.CloneURL, "ssh://127.0.0.1:"))
	core.AssertFalse(t, core.Contains(repository.CloneURL, "@"))
	core.AssertTrue(t, core.HasSuffix(repository.CloneURL, "/Team/App"))
	core.AssertTrue(t, core.HasSuffix(repository.IdentityFile, "soft_serve_client_ed25519"))
	core.AssertTrue(t, core.HasSuffix(repository.KnownHostsFile, "known_hosts"))

	again := service.EnsureRepository(context.Background(), "Team/App")
	core.AssertTrue(t, again.OK, again.Error())
	core.AssertEqual(t, repository, again.Value.(Repository))

	core.AssertFalse(t, service.EnsureRepository(context.Background(), "../../escape").OK)
	core.AssertFalse(t, service.EnsureRepository(context.Background(), " ").OK)
	core.AssertFalse(t, service.EnsureRepository(nil, "another").OK)
}

func TestSoftserveOwnerContention(t *testing.T) {
	options := gitserverTestOptions(t)
	options.PID = func() int { return 101 }
	options.ProcessAlive = func(pid int) bool { return pid == 101 }
	first := NewSoftServe(options).Value.(*softServe)
	core.AssertTrue(t, first.Start(context.Background()).OK)
	t.Cleanup(func() { core.AssertTrue(t, first.Close().OK) })

	secondOptions := options
	secondOptions.PID = func() int { return 202 }
	second := NewSoftServe(secondOptions).Value.(*softServe)
	result := second.Start(context.Background())
	core.AssertFalse(t, result.OK)
	core.AssertContains(t, result.Error(), "101")
	core.AssertContains(t, result.Error(), "owned")
	core.AssertTrue(t, second.Close().OK)
}

func TestSoftserveStaleOwnerRecovery(t *testing.T) {
	options := gitserverTestOptions(t)
	options.PID = func() int { return 303 }
	options.ProcessAlive = func(int) bool { return false }
	ownerPath := core.PathJoin(options.DataPath, "owner.lock")
	core.AssertTrue(t, core.WriteFile(ownerPath, []byte(`{"pid":999,"started_at":"2026-07-18T12:00:00Z"}`), 0o600).OK)

	service := NewSoftServe(options).Value.(*softServe)
	core.AssertTrue(t, service.Start(context.Background()).OK)
	owner := core.ReadFile(ownerPath)
	core.AssertTrue(t, owner.OK, owner.Error())
	core.AssertContains(t, string(owner.Value.([]byte)), `"pid":303`)
	core.AssertTrue(t, service.Close().OK)
	core.AssertFalse(t, core.Stat(ownerPath).OK)
}

func TestSoftservePortCollision(t *testing.T) {
	listenerResult := core.NetListen("tcp", "127.0.0.1:0")
	core.AssertTrue(t, listenerResult.OK, listenerResult.Error())
	listener := listenerResult.Value.(core.Listener)
	t.Cleanup(func() { core.AssertTrue(t, listener.Close() == nil) })

	options := gitserverTestOptions(t)
	options.ListenAddress = listener.Addr().String()
	options.PublicURL = core.Concat("ssh://", listener.Addr().String())
	service := NewSoftServe(options).Value.(*softServe)
	result := service.Start(context.Background())
	core.AssertFalse(t, result.OK)
	core.AssertContains(t, result.Error(), "listen")
	core.AssertFalse(t, core.Stat(core.PathJoin(options.DataPath, "owner.lock")).OK)
	core.AssertTrue(t, service.Close().OK)
}

func TestSoftserveGitRoundTrip(t *testing.T) {
	service, repository := gitserverTestService(t)
	core.AssertTrue(t, service.Health(context.Background()).Value.(Health).Running)

	source := core.PathJoin(t.TempDir(), "source with spaces")
	core.AssertTrue(t, core.MkdirAll(source, 0o700).OK)
	gitserverRunGit(t, source, nil, "init", "-b", "main")
	gitserverRunGit(t, source, nil, "config", "user.name", "LEM Test")
	gitserverRunGit(t, source, nil, "config", "user.email", "lem@example.invalid")
	core.AssertTrue(t, core.WriteFile(core.PathJoin(source, "README.txt"), []byte("private fixture\n"), 0o600).OK)
	gitserverRunGit(t, source, nil, "add", "README.txt")
	gitserverRunGit(t, source, nil, "commit", "-m", "fixture")
	gitserverRunGit(t, source, nil, "remote", "add", "lem", repository.CloneURL)
	environment := gitserverSSHEnvironment(repository)
	gitserverRunGit(t, source, environment, "push", "lem", "HEAD:refs/heads/main")

	cloneParent := t.TempDir()
	gitserverRunGit(t, cloneParent, environment, "clone", repository.CloneURL, "clone with spaces")
	cloned := core.ReadFile(core.PathJoin(cloneParent, "clone with spaces", "README.txt"))
	core.AssertTrue(t, cloned.OK, cloned.Error())
	core.AssertEqual(t, "private fixture\n", string(cloned.Value.([]byte)))

	knownHosts := core.ReadFile(repository.KnownHostsFile)
	core.AssertTrue(t, knownHosts.OK, knownHosts.Error())
	core.AssertContains(t, string(knownHosts.Value.([]byte)), "ssh-ed25519")
	core.AssertFalse(t, core.Contains(string(knownHosts.Value.([]byte)), "0.0.0.0"))
}

func TestSoftservePermissions(t *testing.T) {
	service, repository := gitserverTestService(t)
	core.AssertTrue(t, service.Health(context.Background()).Value.(Health).Running)

	dataInfo := core.Stat(service.options.DataPath)
	core.AssertTrue(t, dataInfo.OK, dataInfo.Error())
	core.AssertEqual(t, core.FileMode(0o700), dataInfo.Value.(core.FsFileInfo).Mode().Perm())
	identityInfo := core.Stat(repository.IdentityFile)
	core.AssertTrue(t, identityInfo.OK, identityInfo.Error())
	core.AssertEqual(t, core.FileMode(0o600), identityInfo.Value.(core.FsFileInfo).Mode().Perm())
	knownInfo := core.Stat(repository.KnownHostsFile)
	core.AssertTrue(t, knownInfo.OK, knownInfo.Error())
	core.AssertEqual(t, core.FileMode(0o600), knownInfo.Value.(core.FsFileInfo).Mode().Perm())
}

type gitserverTestAddress string

func (address gitserverTestAddress) Network() string { return "tcp" }
func (address gitserverTestAddress) String() string  { return string(address) }

type gitserverFailingListener struct {
	closed bool
}

func (listener *gitserverFailingListener) Accept() (core.Conn, error) {
	return nil, core.NewError("accept failed")
}

func (listener *gitserverFailingListener) Close() error {
	listener.closed = true
	return core.NewError("listener close failed")
}

func (listener *gitserverFailingListener) Addr() core.Addr {
	return gitserverTestAddress("127.0.0.1:1")
}

func TestSoftserveNilAndContextEdges(t *testing.T) {
	var service *softServe
	core.AssertFalse(t, service.Start(context.Background()).OK)
	core.AssertFalse(t, service.EnsureRepository(context.Background(), "repo").OK)
	core.AssertFalse(t, service.Health(context.Background()).OK)
	core.AssertFalse(t, service.Close().OK)

	configured := NewSoftServe(gitserverTestOptions(t)).Value.(*softServe)
	cancelled, cancel := context.WithCancel(context.Background())
	cancel()
	core.AssertFalse(t, configured.Start(cancelled).OK)
	core.AssertFalse(t, configured.Health(cancelled).OK)
}

func TestSoftserveStateEdges(t *testing.T) {
	service := NewSoftServe(gitserverTestOptions(t)).Value.(*softServe)
	for _, state := range []string{"starting", "closing", "failed"} {
		service.state = state
		result := service.Start(context.Background())
		core.AssertFalse(t, result.OK)
		core.AssertContains(t, result.Error(), state)
		health := service.Health(context.Background())
		core.AssertTrue(t, health.OK, health.Error())
		core.AssertFalse(t, health.Value.(Health).Running)
		core.AssertTrue(t, health.Value.(Health).Reason != "")
	}
	service.setServeResult(core.Fail(core.NewError("serve exploded")))
	service.state = "failed"
	core.AssertContains(t, service.failureReason(), "exploded")
	service.setServeResult(core.Ok(nil))
}

func TestSoftserveValidationEdges(t *testing.T) {
	for _, authority := range []string{"", "127.0.0.1", ":23231", "example.com:1", "127.0.0.1:nope", "127.0.0.1:-1", "127.0.0.1:65536"} {
		core.AssertFalse(t, loopbackAuthority(authority))
	}
	core.AssertTrue(t, loopbackAuthority("[::1]:0"))

	for _, publicURL := range []string{
		"", "://bad", "http://127.0.0.1:1", "ssh://user@127.0.0.1:1",
		"ssh://127.0.0.1:1/repo", "ssh://127.0.0.1:1?query=1", "ssh://127.0.0.1:1#fragment",
	} {
		core.AssertFalse(t, loopbackPublicURL(publicURL))
	}
	core.AssertTrue(t, loopbackPublicURL("ssh://[::1]:23231"))

	for _, name := range []string{".git", "bad name", "team//app", "team/../app", "team/./app"} {
		core.AssertFalse(t, normalizeRepositoryName(name).OK)
	}
}

func TestSoftserveAcquireOwnerEdges(t *testing.T) {
	filePath := core.PathJoin(t.TempDir(), "data-file")
	core.AssertTrue(t, core.WriteFile(filePath, []byte("not a directory"), 0o600).OK)
	fileOptions := gitserverTestOptions(t)
	fileOptions.DataPath = filePath
	fileService := &softServe{options: fileOptions}
	core.AssertFalse(t, fileService.acquireOwner().OK)

	pidOptions := gitserverTestOptions(t)
	pidOptions.PID = func() int { return 0 }
	pidService := &softServe{options: pidOptions}
	core.AssertFalse(t, pidService.acquireOwner().OK)

	malformedOptions := gitserverTestOptions(t)
	malformedPath := core.PathJoin(malformedOptions.DataPath, ownerFilename)
	core.AssertTrue(t, core.WriteFile(malformedPath, []byte("not-json"), 0o600).OK)
	malformedService := &softServe{options: malformedOptions}
	core.AssertFalse(t, malformedService.acquireOwner().OK)

	directoryOptions := gitserverTestOptions(t)
	directoryPath := core.PathJoin(directoryOptions.DataPath, ownerFilename)
	core.AssertTrue(t, core.MkdirAll(directoryPath, 0o700).OK)
	directoryService := &softServe{options: directoryOptions}
	core.AssertFalse(t, directoryService.acquireOwner().OK)

	staleOptions := gitserverTestOptions(t)
	stalePath := core.PathJoin(staleOptions.DataPath, ownerFilename)
	core.AssertTrue(t, core.WriteFile(stalePath, []byte(`{"pid":909,"started_at":"2026-07-18T12:00:00Z"}`), 0o600).OK)
	staleOptions.ProcessAlive = func(int) bool {
		core.AssertTrue(t, core.Remove(stalePath).OK)
		core.AssertTrue(t, core.MkdirAll(stalePath, 0o700).OK)
		core.AssertTrue(t, core.WriteFile(core.PathJoin(stalePath, "child"), []byte("busy"), 0o600).OK)
		return false
	}
	staleService := &softServe{options: staleOptions}
	core.AssertFalse(t, staleService.acquireOwner().OK)
}

func TestSoftserveReleaseOwnerEdges(t *testing.T) {
	options := gitserverTestOptions(t)
	service := &softServe{options: options}
	core.AssertTrue(t, service.releaseOwner().OK)

	service.owned = true
	service.ownerReceipt = "missing"
	core.AssertTrue(t, service.releaseOwner().OK)

	ownerPath := core.PathJoin(options.DataPath, ownerFilename)
	service.owned = true
	service.ownerReceipt = "ours"
	core.AssertTrue(t, core.WriteFile(ownerPath, []byte("theirs"), 0o600).OK)
	core.AssertFalse(t, service.releaseOwner().OK)

	core.AssertTrue(t, core.Remove(ownerPath).OK)
	core.AssertTrue(t, core.MkdirAll(ownerPath, 0o700).OK)
	core.AssertFalse(t, service.releaseOwner().OK)

	core.AssertTrue(t, core.RemoveAll(ownerPath).OK)
	core.AssertTrue(t, core.WriteFile(ownerPath, []byte("ours"), 0o600).OK)
	service.operations.before = func(operation string) core.Result {
		if operation == "owner-release-remove" {
			return core.Fail(core.NewError("injected owner release removal failure"))
		}
		return core.Ok(nil)
	}
	core.AssertFalse(t, service.releaseOwner().OK)
	service.operations.before = nil
	core.AssertTrue(t, service.releaseOwner().OK)
}

func TestSoftserveRuntimeFilesystemEdges(t *testing.T) {
	sshFileOptions := gitserverTestOptions(t)
	sshFileService := &softServe{options: sshFileOptions}
	core.AssertTrue(t, sshFileService.acquireOwner().OK)
	core.AssertTrue(t, core.WriteFile(core.PathJoin(sshFileOptions.DataPath, "ssh"), []byte("blocked"), 0o600).OK)
	core.AssertFalse(t, sshFileService.startRuntime(context.Background()).OK)
	core.AssertTrue(t, sshFileService.cleanupStart().OK)

	keyOptions := gitserverTestOptions(t)
	keyService := &softServe{options: keyOptions}
	core.AssertTrue(t, keyService.acquireOwner().OK)
	sshPath := core.PathJoin(keyOptions.DataPath, "ssh")
	clientPath := core.PathJoin(sshPath, clientKeyName)
	core.AssertTrue(t, core.MkdirAll(clientPath, 0o700).OK)
	core.AssertTrue(t, core.MkdirAll(core.Concat(clientPath, ".pub"), 0o700).OK)
	core.AssertFalse(t, keyService.startRuntime(context.Background()).OK)
	core.AssertTrue(t, keyService.cleanupStart().OK)
}

func TestSoftserveCloseFailureEdges(t *testing.T) {
	options := gitserverTestOptions(t)
	options.ShutdownTimeout = time.Millisecond
	logResult := core.Create(core.PathJoin(t.TempDir(), "closed.log"))
	core.AssertTrue(t, logResult.OK, logResult.Error())
	logFile := logResult.Value.(*core.OSFile)
	core.AssertTrue(t, logFile.Close() == nil)
	listener := &gitserverFailingListener{}
	service := &softServe{
		options:   options,
		state:     "failed",
		listener:  listener,
		logFile:   logFile,
		serveDone: make(chan struct{}),
	}
	service.setServeResult(core.Fail(core.NewError("serve failed")))
	result := service.Close()
	core.AssertFalse(t, result.OK)
	core.AssertTrue(t, listener.closed)
	core.AssertContains(t, result.Error(), "listener close failed")
	core.AssertContains(t, result.Error(), "serve loop")
	core.AssertContains(t, result.Error(), "log close")
	core.AssertContains(t, result.Error(), "serve failed")
}

func TestSoftserveCloseOwnerMismatch(t *testing.T) {
	options := gitserverTestOptions(t)
	service := NewSoftServe(options).Value.(*softServe)
	core.AssertTrue(t, service.Start(context.Background()).OK)
	receipt := service.ownerReceipt
	ownerPath := core.PathJoin(options.DataPath, ownerFilename)
	core.AssertTrue(t, core.WriteFile(ownerPath, []byte("another owner"), 0o600).OK)
	result := service.Close()
	core.AssertFalse(t, result.OK)
	core.AssertContains(t, result.Error(), "another owner")
	core.AssertTrue(t, core.WriteFile(ownerPath, []byte(receipt), 0o600).OK)
	core.AssertTrue(t, service.Close().OK)
}

func TestSoftserveRepositoryFailureEdges(t *testing.T) {
	options := gitserverTestOptions(t)
	service := NewSoftServe(options).Value.(*softServe)
	core.AssertTrue(t, service.Start(context.Background()).OK)
	core.AssertTrue(t, service.database.Close() == nil)
	result := service.EnsureRepository(context.Background(), "database-closed")
	core.AssertFalse(t, result.OK)
	core.AssertContains(t, result.Error(), "owner")
	service.database = nil
	core.AssertTrue(t, service.Close().OK)

	options = gitserverTestOptions(t)
	service = NewSoftServe(options).Value.(*softServe)
	core.AssertTrue(t, service.Start(context.Background()).OK)
	core.AssertTrue(t, service.EnsureRepository(context.Background(), "hook-failure").OK)
	hookPath := core.PathJoin(options.DataPath, "repos", "hook-failure.git", "hooks", "pre-receive.d", "soft-serve")
	core.AssertTrue(t, core.MkdirAll(hookPath, 0o700).OK)
	core.AssertTrue(t, core.WriteFile(core.PathJoin(hookPath, "child"), []byte("busy"), 0o600).OK)
	result = service.EnsureRepository(context.Background(), "hook-failure")
	core.AssertFalse(t, result.OK)
	core.AssertContains(t, result.Error(), "command hook")
	core.AssertTrue(t, core.RemoveAll(hookPath).OK)
	core.AssertTrue(t, service.Close().OK)
}

func TestSoftserveInjectedRuntimeFailures(t *testing.T) {
	stages := []string{
		"listen",
		"ssh-mkdir",
		"ssh-chmod",
		"client-key",
		"client-key-chmod",
		"config-validate",
		"logger",
		"database-open",
		"database-migrate",
		"disable-keyless",
		"disable-anonymous",
		"ssh-server",
		"host-key",
		"known-hosts-write",
		"known-hosts-chmod",
		"ssh-ready",
	}
	for _, stage := range stages {
		t.Run(stage, func(t *testing.T) {
			options := gitserverTestOptions(t)
			service := NewSoftServe(options).Value.(*softServe)
			service.operations.before = func(operation string) core.Result {
				if operation == stage {
					return core.Fail(core.Errorf("injected %s failure", stage))
				}
				return core.Ok(nil)
			}
			result := service.Start(context.Background())
			core.AssertFalse(t, result.OK)
			core.AssertContains(t, result.Error(), "injected")
			core.AssertFalse(t, core.Stat(core.PathJoin(options.DataPath, ownerFilename)).OK)
			service.operations.before = nil
			core.AssertTrue(t, service.Close().OK)
		})
	}
}

func TestSoftserveInjectedOwnerFailures(t *testing.T) {
	for _, stage := range []string{"owner-mkdir", "owner-chmod", "owner-open", "owner-write", "owner-sync", "owner-close"} {
		t.Run(stage, func(t *testing.T) {
			options := gitserverTestOptions(t)
			service := NewSoftServe(options).Value.(*softServe)
			service.operations.before = func(operation string) core.Result {
				if operation == stage {
					return core.Fail(core.Errorf("injected %s failure", stage))
				}
				return core.Ok(nil)
			}
			result := service.acquireOwner()
			core.AssertFalse(t, result.OK)
			core.AssertContains(t, result.Error(), "injected")
			core.AssertFalse(t, core.Stat(core.PathJoin(options.DataPath, ownerFilename)).OK)
		})
	}
}

func TestSoftserveInjectedOwnerReadFailures(t *testing.T) {
	options := gitserverTestOptions(t)
	ownerPath := core.PathJoin(options.DataPath, ownerFilename)
	core.AssertTrue(t, core.WriteFile(ownerPath, []byte(`{"pid":777,"started_at":"2026-07-18T12:00:00Z"}`), 0o600).OK)
	service := NewSoftServe(options).Value.(*softServe)
	service.operations.before = func(operation string) core.Result {
		if operation == "owner-read" {
			return core.Fail(core.NewError("injected owner read failure"))
		}
		return core.Ok(nil)
	}
	core.AssertFalse(t, service.acquireOwner().OK)

	service.operations.before = func(operation string) core.Result {
		if operation == "owner-remove-stale" {
			return core.Fail(core.NewError("injected stale removal failure"))
		}
		return core.Ok(nil)
	}
	core.AssertFalse(t, service.acquireOwner().OK)
}

func TestSoftserveStartCleanupFailure(t *testing.T) {
	options := gitserverTestOptions(t)
	service := NewSoftServe(options).Value.(*softServe)
	service.operations.before = func(operation string) core.Result {
		if operation == "listen" || operation == "owner-release-read" {
			return core.Fail(core.Errorf("injected %s failure", operation))
		}
		return core.Ok(nil)
	}
	result := service.Start(context.Background())
	core.AssertFalse(t, result.OK)
	core.AssertContains(t, result.Error(), "owner-release-read")
	service.operations.before = nil
	core.AssertTrue(t, service.Close().OK)
}

func TestSoftserveEnsureStartAndStateFailures(t *testing.T) {
	options := gitserverTestOptions(t)
	options.PID = func() int { return 707 }
	options.ProcessAlive = func(pid int) bool { return pid == 707 }
	owner := NewSoftServe(options).Value.(*softServe)
	core.AssertTrue(t, owner.Start(context.Background()).OK)
	t.Cleanup(func() { core.AssertTrue(t, owner.Close().OK) })
	contender := NewSoftServe(options).Value.(*softServe)
	core.AssertFalse(t, contender.EnsureRepository(context.Background(), "repo").OK)

	stateOnly := NewSoftServe(gitserverTestOptions(t)).Value.(*softServe)
	stateOnly.state = "running"
	core.AssertFalse(t, stateOnly.EnsureRepository(context.Background(), "repo").OK)
}

func TestSoftserveCloseDatabaseFailure(t *testing.T) {
	service := NewSoftServe(gitserverTestOptions(t)).Value.(*softServe)
	core.AssertTrue(t, service.Start(context.Background()).OK)
	service.operations.before = func(operation string) core.Result {
		if operation == "database-close" {
			return core.Fail(core.NewError("injected database close failure"))
		}
		return core.Ok(nil)
	}
	result := service.Close()
	core.AssertFalse(t, result.OK)
	core.AssertContains(t, result.Error(), "database close")
}
