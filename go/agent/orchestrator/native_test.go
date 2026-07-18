// SPDX-License-Identifier: EUPL-1.2

package orchestrator

import (
	"context"
	"slices"
	"sync"
	"testing"
	"time"

	core "dappco.re/go"
	"dappco.re/go/inference/agent/gitserver"
	"dappco.re/go/inference/agent/provider"
	"dappco.re/go/inference/agent/queue"
	"dappco.re/go/inference/agent/work"
	"dappco.re/go/inference/agent/workspace"
	coreprocess "dappco.re/go/process"
	processapi "dappco.re/go/process/pkg/api"
)

type nativeTestLine struct {
	stream string
	line   string
}

func nativeTestService(t *testing.T, withCore bool) *coreprocess.Service {
	t.Helper()
	var app *core.Core
	if withCore {
		app = core.New()
	}
	result := coreprocess.NewService(coreprocess.Options{BufferSize: 4096})(app)
	if !result.OK {
		t.Fatalf("process NewService failed: %s", result.Error())
	}
	service := result.Value.(*coreprocess.Service)
	if started := service.OnStartup(context.Background()); !started.OK {
		t.Fatalf("process OnStartup failed: %s", started.Error())
	}
	t.Cleanup(func() {
		if closed := service.OnShutdown(context.Background()); !closed.OK {
			t.Errorf("process OnShutdown failed: %s", closed.Error())
		}
	})
	return service
}

func nativeTestLauncher(t *testing.T, essentials ...string) Launcher {
	t.Helper()
	result := NewNativeLauncher(nativeTestService(t, true), essentials)
	if !result.OK {
		t.Fatalf("NewNativeLauncher failed: %s", result.Error())
	}
	launcher := result.Value.(Launcher)
	t.Cleanup(func() {
		if closed := launcher.Close(); !closed.OK {
			t.Errorf("launcher Close failed: %s", closed.Error())
		}
	})
	return launcher
}

func nativeTestEnvironment(t *testing.T, name, value string) {
	t.Helper()
	t.Setenv(name, value)
}

func nativeTestScript(t *testing.T, body string) string {
	t.Helper()
	directory := core.PathJoin(t.TempDir(), "native scripts with spaces")
	core.AssertTrue(t, core.MkdirAll(directory, 0o700).OK)
	path := core.PathJoin(directory, "fake provider")
	result := core.WriteFile(path, []byte(core.Concat("#!/bin/sh\n", body)), 0o700)
	core.AssertTrue(t, result.OK, result.Error())
	return path
}

func nativeWaitDead(pid int, timeout time.Duration) bool {
	deadline := time.Now().Add(timeout)
	for processapi.PIDAlive(pid) && time.Now().Before(deadline) {
		time.Sleep(20 * time.Millisecond)
	}
	return !processapi.PIDAlive(pid)
}

func nativeReceiptScript(t *testing.T, providerName string) string {
	t.Helper()
	return nativeTestScript(t, core.Concat(
		"if [ \"${1-}\" = \"--version\" ]; then printf '", providerName, " receipt 1.0\\n'; exit 0; fi\n",
		"printf 'ordered-one:", providerName, "\\n'\n",
		"printf '", providerName, " native receipt\\n' > native-receipt.txt\n",
		"git add native-receipt.txt\n",
		"git -c user.name='LEM Receipt' -c user.email='lem-receipt@localhost' commit -m '", providerName, " native receipt' >/dev/null\n",
		"printf 'ordered-two:", providerName, "\\n'\n",
		"printf '<<<LEM_STATUS>>>{\"status\":\"completed\",\"summary\":\"", providerName, " complete\"}<<<END_LEM_STATUS>>>\\n'\n",
	))
}

func nativeInterruptedReceiptScript(t *testing.T) string {
	t.Helper()
	return nativeTestScript(t, "if [ \"${1-}\" = \"--version\" ]; then printf 'codex receipt 1.0\\n'; exit 0; fi\nif [ -f interrupted-receipt.txt ]; then\n  printf 'resume-one\\n'\n  printf 'resumed\\n' > resumed-receipt.txt\n  git add resumed-receipt.txt\n  git -c user.name='LEM Receipt' -c user.email='lem-receipt@localhost' commit -m 'resumed native receipt' >/dev/null\n  printf 'resume-two\\n'\n  printf '<<<LEM_STATUS>>>{\"status\":\"completed\",\"summary\":\"resume complete\"}<<<END_LEM_STATUS>>>\\n'\n  exit 0\nfi\nprintf 'interrupt-one\\n'\nprintf 'interrupted\\n' > interrupted-receipt.txt\ngit add interrupted-receipt.txt\ngit -c user.name='LEM Receipt' -c user.email='lem-receipt@localhost' commit -m 'interrupted native receipt' >/dev/null\n/bin/sh -c 'while :; do /bin/sleep 1; done' &\nchild=$!\nprintf 'child:%s\\n' \"$child\"\nwhile :; do /bin/sleep 1; done\n")
}

func nativeReceiptRuntime(t *testing.T, fixture *orchestratorFixture, providerName, executable string) {
	t.Helper()
	core.AssertTrue(t, fixture.orchestrator.Close().OK)
	fixture.server.mu.Lock()
	fixture.server.closed = false
	fixture.server.started = false
	fixture.server.mu.Unlock()

	registryResult := provider.DefaultRegistry(nil, map[string]provider.Config{
		providerName: {Executable: executable},
	})
	core.AssertTrue(t, registryResult.OK, registryResult.Error())
	policy := queue.Policy{
		Version: 1,
		Dispatch: queue.DispatchConfig{
			DefaultAgent: providerName, GlobalConcurrency: 1, TimeoutMinutes: 1,
			Validation: []queue.Command{{Command: "git", Args: []string{"diff", "--check", "HEAD^", "HEAD"}}},
		},
		Concurrency: map[string]queue.ConcurrencyLimit{
			"codex": {Total: 1}, "claude": {Total: 1}, "opencode": {Total: 1},
		},
		Rates: map[string]queue.RateConfig{},
		Providers: map[string]queue.NativeConfig{
			providerName: {Executable: executable},
		},
	}
	queueResult := queue.NewController(policy, fixture.store.queue, nil)
	core.AssertTrue(t, queueResult.OK, queueResult.Error())
	launcher := nativeTestLauncher(t, "HOME", "PATH")
	result := New(Options{
		Store: fixture.store, GitServer: fixture.server, Workspaces: fixture.manager,
		Providers: registryResult.Value.(*provider.Registry), Queue: queueResult.Value.(*queue.Controller),
		Launcher: launcher, Clock: fixture.clock, IDs: fixture.ids,
		LogBatchBytes: 1, LogBatchDelay: 10 * time.Millisecond,
	})
	core.AssertTrue(t, result.OK, result.Error())
	fixture.queue = queueResult.Value.(*queue.Controller)
	fixture.orchestrator = result.Value.(*Orchestrator)
}

func nativeReceiptRegister(t *testing.T, fixture *orchestratorFixture, providerName string) (work.Item, work.Project, string) {
	t.Helper()
	source := core.PathJoin(t.TempDir(), core.Concat(providerName, " clean source"))
	revision := orchestratorCreateRepository(t, source)
	item := work.Item{
		ID: core.Concat("work-", providerName), Title: core.Concat("Verify ", providerName),
		Task: core.Concat("Create the ", providerName, " native lifecycle receipt"), Repository: source,
	}
	reviewed := fixture.orchestrator.ReviewProject(context.Background(), item)
	core.AssertTrue(t, reviewed.OK, reviewed.Error())
	registered := fixture.orchestrator.RegisterProject(context.Background(), reviewed.Value.(ProjectReview), true)
	core.AssertTrue(t, registered.OK, registered.Error())
	return item, registered.Value.(work.Project), revision
}

func nativeReceiptDispatch(t *testing.T, fixture *orchestratorFixture, item work.Item, providerName, revision string) (DispatchReview, work.Run) {
	t.Helper()
	reviewed := fixture.orchestrator.ReviewDispatch(context.Background(), work.DispatchRequest{
		Work: item, Provider: providerName, Model: "receipt-model", ConfirmedSourceRevision: revision,
	})
	core.AssertTrue(t, reviewed.OK, reviewed.Error())
	review := reviewed.Value.(DispatchReview)
	core.AssertTrue(t, review.Detection.Available)
	core.AssertContains(t, review.Warning, "host")
	dispatched := fixture.orchestrator.Dispatch(context.Background(), review)
	core.AssertTrue(t, dispatched.OK, dispatched.Error())
	return review, dispatched.Value.(work.Run)
}

func nativeReceiptWaitLog(t *testing.T, fixture *orchestratorFixture, runID, prefix string) work.LogChunk {
	t.Helper()
	deadline := time.Now().Add(10 * time.Second)
	for time.Now().Before(deadline) {
		snapshotResult := fixture.store.Snapshot("")
		if snapshotResult.OK {
			for _, log := range snapshotResult.Value.(work.Snapshot).Logs {
				if log.RunID == runID && core.HasPrefix(log.Text, prefix) {
					return log
				}
			}
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatalf("run %s did not persist log prefix %q", runID, prefix)
	return work.LogChunk{}
}

func TestNative_NewNativeLauncher_Good(t *testing.T) {
	service := nativeTestService(t, true)
	result := NewNativeLauncher(service, []string{"PATH", "HOME"})
	core.AssertTrue(t, result.OK, result.Error())
	launcher := result.Value.(Launcher)
	core.AssertTrue(t, launcher.Close().OK)
}

func TestNative_NewNativeLauncher_Bad(t *testing.T) {
	core.AssertFalse(t, NewNativeLauncher(nil, nil).OK)
	service := nativeTestService(t, true)
	core.AssertFalse(t, NewNativeLauncher(service, []string{"BAD-NAME"}).OK)
	core.AssertFalse(t, NewNativeLauncher(service, []string{"PATH", "PATH"}).OK)
	core.AssertFalse(t, NewNativeLauncher(service, []string{""}).OK)
}

func TestNative_NewNativeLauncher_Ugly(t *testing.T) {
	service := nativeTestService(t, false)
	result := NewNativeLauncher(service, []string{"PATH"})
	core.AssertFalse(t, result.OK)
	nativeTestEnvironment(t, "PATH", "")
	core.AssertFalse(t, NewNativeLauncher(nativeTestService(t, true), nil).OK)
}

func TestNativeEnvironmentIsolationAndRedaction(t *testing.T) {
	nativeTestEnvironment(t, "LEM_NATIVE_KEEP", "kept")
	nativeTestEnvironment(t, "LEM_NATIVE_SECRET", "secret-value")
	nativeTestEnvironment(t, "LEM_NATIVE_DROP", "discarded")
	launcher := nativeTestLauncher(t, "PATH", "LEM_NATIVE_KEEP")

	detected := launcher.DetectEnvironment([]string{"LEM_NATIVE_SECRET"})
	core.AssertTrue(t, detected.OK, detected.Error())
	environment := detected.Value.([]string)
	core.AssertTrue(t, slices.Contains(environment, "LEM_NATIVE_KEEP=kept"))
	core.AssertTrue(t, slices.Contains(environment, "LEM_NATIVE_SECRET=secret-value"))
	core.AssertFalse(t, slices.Contains(environment, "LEM_NATIVE_DROP="))

	script := nativeTestScript(t, "printf 'out-one:%s:%s:%s\\n' \"$LEM_NATIVE_KEEP\" \"$LEM_NATIVE_SECRET\" \"${LEM_NATIVE_DROP-unset}\"\n/bin/sleep 0.05\nprintf 'err-one:%s\\n' \"$LEM_NATIVE_SECRET\" >&2\n/bin/sleep 0.05\nprintf 'out-two\\n'\n")
	var mu sync.Mutex
	lines := make([]nativeTestLine, 0, 3)
	started := launcher.Start(context.Background(), provider.Command{
		Provider: "fake", Executable: script, Dir: core.PathDir(script),
		CredentialKeys: []string{"LEM_NATIVE_SECRET"}, Receipt: "fake provider <redacted>",
	}, func(stream, line string) {
		mu.Lock()
		lines = append(lines, nativeTestLine{stream: stream, line: line})
		mu.Unlock()
	})
	core.AssertTrue(t, started.OK, started.Error())
	process := started.Value.(Process)
	core.AssertTrue(t, process.ID() != "")
	core.AssertTrue(t, process.PID() > 0)
	waited := process.Wait()
	core.AssertTrue(t, waited.OK, waited.Error())
	core.AssertEqual(t, 0, waited.Value.(int))

	mu.Lock()
	actual := append([]nativeTestLine(nil), lines...)
	mu.Unlock()
	expected := []nativeTestLine{
		{stream: "stdout", line: "out-one:kept:[REDACTED]:unset"},
		{stream: "stderr", line: "err-one:[REDACTED]"},
		{stream: "stdout", line: "out-two"},
	}
	core.AssertTrue(t, slices.Equal(actual, expected), core.Sprintf("unexpected output: %#v", actual))
	for _, output := range actual {
		core.AssertFalse(t, core.Contains(output.line, "secret-value"))
	}
}

func TestNativeFastOutputIsBuffered(t *testing.T) {
	launcher := nativeTestLauncher(t, "PATH")
	script := nativeTestScript(t, "printf 'early-one\\n'\nprintf 'early-two\\n'\n")
	for attempt := 0; attempt < 10; attempt++ {
		var mu sync.Mutex
		lines := make([]string, 0, 2)
		started := launcher.Start(context.Background(), provider.Command{
			Provider: "fake", Executable: script, Dir: core.PathDir(script),
		}, func(_ string, line string) {
			mu.Lock()
			lines = append(lines, line)
			mu.Unlock()
		})
		core.AssertTrue(t, started.OK, started.Error())
		core.AssertTrue(t, started.Value.(Process).Wait().OK)
		mu.Lock()
		actual := append([]string(nil), lines...)
		mu.Unlock()
		core.AssertEqual(t, []string{"early-one", "early-two"}, actual)
	}
}

func TestNativeStartFlushesSynchronousOutput(t *testing.T) {
	service := nativeTestService(t, true)
	result := NewNativeLauncher(service, []string{"PATH"})
	core.AssertTrue(t, result.OK, result.Error())
	launcher := result.Value.(Launcher)
	t.Cleanup(func() { core.AssertTrue(t, launcher.Close().OK) })
	service.Core().RegisterAction(func(app *core.Core, message core.Message) core.Result {
		started, ok := message.(coreprocess.ActionProcessStarted)
		if !ok {
			return core.Ok(nil)
		}
		return app.ACTION(coreprocess.ActionProcessOutput{ID: started.ID, Stream: coreprocess.StreamStdout, Line: "before-route"})
	})

	script := nativeTestScript(t, "exit 0\n")
	var lines []string
	var mu sync.Mutex
	started := launcher.Start(context.Background(), provider.Command{
		Provider: "fake", Executable: script, Dir: core.PathDir(script),
	}, func(_ string, line string) {
		mu.Lock()
		lines = append(lines, line)
		mu.Unlock()
	})
	core.AssertTrue(t, started.OK, started.Error())
	core.AssertTrue(t, started.Value.(Process).Wait().OK)
	mu.Lock()
	actual := append([]string(nil), lines...)
	mu.Unlock()
	core.AssertEqual(t, []string{"before-route"}, actual)
}

func TestNativeNonZeroExitAndOutputRedaction(t *testing.T) {
	nativeTestEnvironment(t, "LEM_NATIVE_SECRET", "nonzero-secret")
	launcher := nativeTestLauncher(t, "PATH")
	script := nativeTestScript(t, "printf 'failed:%s\\n' \"$LEM_NATIVE_SECRET\" >&2\nexit 7\n")
	var lines []string
	var mu sync.Mutex
	started := launcher.Start(context.Background(), provider.Command{
		Provider: "fake", Executable: script, Dir: core.PathDir(script), CredentialKeys: []string{"LEM_NATIVE_SECRET"},
	}, func(_ string, line string) {
		mu.Lock()
		lines = append(lines, line)
		mu.Unlock()
	})
	core.AssertTrue(t, started.OK, started.Error())
	waited := started.Value.(Process).Wait()
	core.AssertTrue(t, waited.OK, waited.Error())
	core.AssertEqual(t, 7, waited.Value.(int))
	mu.Lock()
	actual := append([]string(nil), lines...)
	mu.Unlock()
	core.AssertEqual(t, []string{"failed:[REDACTED]"}, actual)
}

func TestNativeEnvironmentValidation(t *testing.T) {
	launcher := nativeTestLauncher(t, "PATH", "LEM_NATIVE_KEEP")
	core.AssertFalse(t, launcher.DetectEnvironment([]string{"BAD-NAME"}).OK)
	nativeTestEnvironment(t, "LEM_NATIVE_MULTILINE", "secret\nvalue")
	core.AssertFalse(t, launcher.DetectEnvironment([]string{"LEM_NATIVE_MULTILINE"}).OK)
	script := nativeTestScript(t, "exit 0\n")
	callback := func(string, string) {}
	commands := []provider.Command{
		{},
		{Provider: "fake", Executable: script, Dir: "relative"},
		{Provider: "fake", Executable: script, Dir: core.PathDir(script), CredentialKeys: []string{"BAD-NAME"}},
		{Provider: "fake", Executable: script, Dir: core.PathDir(script), Environment: []string{"UNLISTED=value"}},
		{Provider: "fake", Executable: script, Dir: core.PathDir(script), Environment: []string{"LEM_NATIVE_KEEP"}},
		{Provider: "fake", Executable: script, Dir: core.PathDir(script), Environment: []string{"LEM_NATIVE_KEEP=line\nvalue"}},
		{Provider: "fake", Executable: script, Dir: core.PathDir(script), Args: []string{"bad\x00argument"}},
		{Provider: "fake", Executable: "bad\x00executable", Dir: core.PathDir(script)},
	}
	for _, command := range commands {
		core.AssertFalse(t, launcher.Start(context.Background(), command, callback).OK)
	}
	core.AssertFalse(t, launcher.Start(nil, provider.Command{Provider: "fake", Executable: script, Dir: core.PathDir(script)}, callback).OK)
	core.AssertFalse(t, launcher.Start(context.Background(), provider.Command{Provider: "fake", Executable: script, Dir: core.PathDir(script)}, nil).OK)
	core.AssertFalse(t, uniqueEnvironmentNames([]string{"1INVALID"}).OK)
}

func TestNativeBoundaryGuards(t *testing.T) {
	var missingLauncher *nativeLauncher
	core.AssertFalse(t, missingLauncher.DetectEnvironment(nil).OK)
	core.AssertFalse(t, missingLauncher.Start(context.Background(), provider.Command{}, func(string, string) {}).OK)
	core.AssertFalse(t, missingLauncher.Close().OK)

	var missingProcess *nativeProcess
	core.AssertEqual(t, "", missingProcess.ID())
	core.AssertEqual(t, 0, missingProcess.PID())
	core.AssertFalse(t, missingProcess.Wait().OK)
	core.AssertFalse(t, missingProcess.Shutdown().OK)
	core.AssertEqual(t, "", (&nativeProcess{}).ID())
	core.AssertEqual(t, 0, (&nativeProcess{}).PID())
	core.AssertFalse(t, (&nativeProcess{}).Wait().OK)
	core.AssertFalse(t, (&nativeProcess{}).Shutdown().OK)

	service := nativeTestService(t, true)
	broken := &nativeLauncher{
		service: service, routes: make(map[string]*nativeRoute), pending: make(map[string][]nativeOutput),
		processes: map[string]*nativeProcess{"broken": {}},
	}
	closed := broken.Close()
	core.AssertFalse(t, closed.OK)
	core.AssertContains(t, closed.Error(), "process is required")

	closedLauncher := nativeTestLauncher(t, "PATH")
	core.AssertTrue(t, closedLauncher.Close().OK)
	script := nativeTestScript(t, "exit 0\n")
	start := closedLauncher.Start(context.Background(), provider.Command{
		Provider: "fake", Executable: script, Dir: core.PathDir(script),
	}, func(string, string) {})
	core.AssertFalse(t, start.OK)
	core.AssertContains(t, start.Error(), "launcher is closed")
}

func TestNativeActionRoutingAndFailedStart(t *testing.T) {
	launcher := &nativeLauncher{
		routes: make(map[string]*nativeRoute), pending: make(map[string][]nativeOutput),
		processes: make(map[string]*nativeProcess),
	}
	core.AssertTrue(t, launcher.handleAction(storeTestEvent("ignored")).OK)
	core.AssertTrue(t, launcher.handleAction(coreprocess.ActionProcessOutput{ID: "unowned", Stream: coreprocess.StreamStdout, Line: "lost"}).OK)
	core.AssertEqual(t, 0, len(launcher.pending))

	launcher.starting = 2
	core.AssertTrue(t, launcher.handleAction(coreprocess.ActionProcessOutput{ID: "early", Stream: coreprocess.StreamStdout, Line: "buffered"}).OK)
	core.AssertEqual(t, 1, len(launcher.pending["early"]))
	launcher.finishStart("early")
	core.AssertEqual(t, 1, launcher.starting)
	core.AssertEqual(t, 0, len(launcher.pending))
	launcher.finishStart("")
	core.AssertEqual(t, 0, launcher.starting)

	var delivered string
	launcher.routes["owned"] = &nativeRoute{callback: func(_ string, line string) { delivered = line }, secrets: []string{"secret"}}
	core.AssertTrue(t, launcher.handleAction(coreprocess.ActionProcessOutput{ID: "owned", Stream: coreprocess.StreamStderr, Line: "a secret"}).OK)
	core.AssertEqual(t, "a [REDACTED]", delivered)
	launcher.routes["panic"] = &nativeRoute{callback: func(string, string) { panic("callback") }}
	core.AssertTrue(t, launcher.handleAction(coreprocess.ActionProcessOutput{ID: "panic", Stream: coreprocess.StreamStdout, Line: "safe"}).OK)
	launcher.closed = true
	core.AssertTrue(t, launcher.handleAction(coreprocess.ActionProcessOutput{ID: "owned", Stream: coreprocess.StreamStdout, Line: "ignored"}).OK)
	core.AssertEqual(t, "a [REDACTED]", delivered)

	service := nativeTestService(t, true)
	result := NewNativeLauncher(service, []string{"PATH"})
	core.AssertTrue(t, result.OK, result.Error())
	native := result.Value.(*nativeLauncher)
	native.envPath = core.PathJoin(t.TempDir(), "missing-env")
	script := nativeTestScript(t, "exit 0\n")
	failed := native.Start(context.Background(), provider.Command{Provider: "fake", Executable: script, Dir: core.PathDir(script)}, func(string, string) {})
	core.AssertFalse(t, failed.OK)
	core.AssertEqual(t, 0, native.starting)
	core.AssertTrue(t, native.Close().OK)
}

func TestNativeCredentialReceiptAndOrdering(t *testing.T) {
	nativeTestEnvironment(t, "LEM_NATIVE_SHORT", "small")
	nativeTestEnvironment(t, "LEM_NATIVE_LONG", "a-longer-secret")
	launcher := nativeTestLauncher(t, "PATH")
	script := nativeTestScript(t, "exit 0\n")
	command := provider.Command{
		Provider: "fake", Executable: script, Dir: core.PathDir(script),
		CredentialKeys: []string{"LEM_NATIVE_SHORT", "LEM_NATIVE_LONG"},
	}
	environment := launcher.(*nativeLauncher).commandEnvironment(command)
	core.AssertTrue(t, environment.OK, environment.Error())
	secrets := environment.Value.(nativeEnvironment).secrets
	core.AssertEqual(t, []string{"a-longer-secret", "small"}, secrets)
	command.Receipt = "unsafe a-longer-secret receipt"
	core.AssertFalse(t, launcher.Start(context.Background(), command, func(string, string) {}).OK)

	overlap := launcher.DetectEnvironment([]string{"PATH"})
	core.AssertTrue(t, overlap.OK, overlap.Error())
	core.AssertEqual(t, 1, len(overlap.Value.([]string)))
	overlapCommand := launcher.(*nativeLauncher).commandEnvironment(provider.Command{CredentialKeys: []string{"PATH"}})
	core.AssertTrue(t, overlapCommand.OK, overlapCommand.Error())
	core.AssertEqual(t, 1, len(overlapCommand.Value.(nativeEnvironment).assignments))

	missingName := "LEM_NATIVE_TEST_MISSING_019F7061"
	_, missingExists := core.LookupEnv(missingName)
	core.AssertFalse(t, missingExists)
	missing := launcher.(*nativeLauncher).commandEnvironment(provider.Command{CredentialKeys: []string{missingName}})
	core.AssertTrue(t, missing.OK, missing.Error())
	core.AssertEqual(t, 1, len(missing.Value.(nativeEnvironment).assignments))
}

func TestNativeCloseDuringStartAndReleaseFailure(t *testing.T) {
	service := nativeTestService(t, true)
	result := NewNativeLauncher(service, []string{"PATH"})
	core.AssertTrue(t, result.OK, result.Error())
	launcher := result.Value.(Launcher)
	service.Core().RegisterAction(func(_ *core.Core, message core.Message) core.Result {
		if _, started := message.(coreprocess.ActionProcessStarted); started {
			time.Sleep(100 * time.Millisecond)
			return launcher.Close()
		}
		return core.Ok(nil)
	})
	script := nativeTestScript(t, "trap 'exit 0' TERM\nwhile :; do /bin/sleep 1; done\n")
	started := launcher.Start(context.Background(), provider.Command{Provider: "fake", Executable: script, Dir: core.PathDir(script)}, func(string, string) {})
	core.AssertFalse(t, started.OK)

	secondService := nativeTestService(t, true)
	secondResult := NewNativeLauncher(secondService, []string{"PATH"})
	core.AssertTrue(t, secondResult.OK, secondResult.Error())
	second := secondResult.Value.(*nativeLauncher)
	quick := nativeTestScript(t, "exit 0\n")
	quickResult := second.Start(context.Background(), provider.Command{Provider: "fake", Executable: quick, Dir: core.PathDir(quick)}, func(string, string) {})
	core.AssertTrue(t, quickResult.OK, quickResult.Error())
	process := quickResult.Value.(*nativeProcess)
	<-process.process.Done()
	core.AssertTrue(t, secondService.Remove(process.ID()).OK)
	core.AssertFalse(t, process.Wait().OK)
	core.AssertTrue(t, second.Close().OK)
}

func TestNativeExplicitEnvironmentOverride(t *testing.T) {
	nativeTestEnvironment(t, "LEM_NATIVE_KEEP", "host-value")
	nativeTestEnvironment(t, "LEM_NATIVE_SECRET", "host-secret")
	launcher := nativeTestLauncher(t, "PATH", "LEM_NATIVE_KEEP")
	script := nativeTestScript(t, "printf '%s:%s\\n' \"$LEM_NATIVE_KEEP\" \"$LEM_NATIVE_SECRET\"\n")
	var mu sync.Mutex
	var line string
	started := launcher.Start(context.Background(), provider.Command{
		Provider: "fake", Executable: script, Dir: core.PathDir(script),
		CredentialKeys: []string{"LEM_NATIVE_SECRET"},
		Environment:    []string{"LEM_NATIVE_KEEP=override", "LEM_NATIVE_SECRET=override-secret"},
	}, func(_ string, output string) {
		mu.Lock()
		line = output
		mu.Unlock()
	})
	core.AssertTrue(t, started.OK, started.Error())
	core.AssertTrue(t, started.Value.(Process).Wait().OK)
	mu.Lock()
	defer mu.Unlock()
	core.AssertEqual(t, "override:[REDACTED]", line)
}

func TestNativeShutdownKillsProcessGroup(t *testing.T) {
	launcher := nativeTestLauncher(t, "PATH")
	script := nativeTestScript(t, "trap '' TERM\n/bin/sh -c 'trap \"\" TERM; while :; do /bin/sleep 1; done' &\nchild=$!\nprintf 'child:%s\\n' \"$child\"\nwhile :; do /bin/sleep 1; done\n")
	childChannel := make(chan int, 1)
	started := launcher.Start(context.Background(), provider.Command{
		Provider: "fake", Executable: script, Dir: core.PathDir(script),
	}, func(_ string, line string) {
		if core.HasPrefix(line, "child:") {
			parsed := core.Atoi(core.TrimPrefix(line, "child:"))
			if parsed.OK {
				childChannel <- parsed.Value.(int)
			}
		}
	})
	core.AssertTrue(t, started.OK, started.Error())
	process := started.Value.(Process)
	var childPID int
	select {
	case childPID = <-childChannel:
	case <-time.After(5 * time.Second):
		t.Fatal("child PID was not streamed")
	}
	parentPID := process.PID()
	core.AssertTrue(t, processapi.PIDAlive(parentPID))
	core.AssertTrue(t, processapi.PIDAlive(childPID))
	shutdown := process.Shutdown()
	core.AssertTrue(t, shutdown.OK, shutdown.Error())
	core.AssertTrue(t, process.Wait().OK)
	core.AssertTrue(t, nativeWaitDead(parentPID, 3*time.Second))
	core.AssertTrue(t, nativeWaitDead(childPID, 3*time.Second))
}

func TestNativeLauncherCloseJoinsProcesses(t *testing.T) {
	service := nativeTestService(t, true)
	result := NewNativeLauncher(service, []string{"PATH"})
	core.AssertTrue(t, result.OK, result.Error())
	launcher := result.Value.(Launcher)
	script := nativeTestScript(t, "trap 'exit 0' TERM\nwhile :; do /bin/sleep 1; done\n")
	started := launcher.Start(context.Background(), provider.Command{
		Provider: "fake", Executable: script, Dir: core.PathDir(script),
	}, func(string, string) {})
	core.AssertTrue(t, started.OK, started.Error())
	process := started.Value.(Process)
	pid := process.PID()
	core.AssertTrue(t, launcher.Close().OK)
	core.AssertTrue(t, nativeWaitDead(pid, 2*time.Second))
	core.AssertTrue(t, launcher.Close().OK)
	core.AssertFalse(t, launcher.Start(context.Background(), provider.Command{Provider: "fake", Executable: script, Dir: core.PathDir(script)}, func(string, string) {}).OK)
}

func TestNativeAdaptersEndToEndLifecycleReceipt(t *testing.T) {
	for _, providerName := range []string{"codex", "claude", "opencode"} {
		t.Run(providerName, func(t *testing.T) {
			fixture := newOrchestratorFixture(t)
			executable := nativeReceiptScript(t, providerName)
			nativeReceiptRuntime(t, fixture, providerName, executable)
			item, project, sourceRevision := nativeReceiptRegister(t, fixture, providerName)
			before := orchestratorAcceptanceSourceSnapshot(t, project.RepositoryRoot)
			review, run := nativeReceiptDispatch(t, fixture, item, providerName, sourceRevision)
			core.AssertEqual(t, executable, review.Command.Executable)
			core.AssertEqual(t, providerName, review.Command.Provider)

			core.AssertTrue(t, fixture.orchestrator.StartQueue(context.Background()).OK)
			running := orchestratorWaitRunStatus(t, fixture.store, run.ID, work.RunRunning, work.RunCompleted)
			pid := running.ProcessID
			finished := orchestratorWaitRunStatus(t, fixture.store, run.ID, work.RunCompleted)
			core.AssertEqual(t, run.ID, finished.ID)
			core.AssertEqual(t, 0, finished.ExitCode)
			core.AssertTrue(t, finished.DurableRevision != "")
			core.AssertTrue(t, finished.DurableRevision != sourceRevision)
			if pid == 0 {
				pid = finished.ProcessID
			}
			core.AssertTrue(t, pid > 0)
			core.AssertTrue(t, nativeWaitDead(pid, 3*time.Second))

			snapshot := fixture.store.Snapshot(item.ID).Value.(work.Snapshot)
			ordered := make([]string, 0, 3)
			for _, log := range snapshot.Logs {
				if log.RunID == run.ID {
					ordered = append(ordered, log.Text)
				}
			}
			core.AssertEqual(t, []string{
				core.Concat("ordered-one:", providerName),
				core.Concat("ordered-two:", providerName),
				core.Concat(`<<<LEM_STATUS>>>{"status":"completed","summary":"`, providerName, ` complete"}<<<END_LEM_STATUS>>>`),
			}, ordered)
			core.AssertEqual(t, before, orchestratorAcceptanceSourceSnapshot(t, project.RepositoryRoot))
			core.AssertFalse(t, core.Stat(core.PathJoin(project.RepositoryRoot, "native-receipt.txt")).OK)
			orchestratorAssertNoStateMarkdown(t, project.RepositoryRoot)
			core.AssertFalse(t, core.Stat(core.PathJoin(project.RepositoryRoot, ".lem")).OK)

			reviewed := fixture.orchestrator.ReviewChanges(context.Background(), run.ID)
			core.AssertTrue(t, reviewed.OK, reviewed.Error())
			changes := reviewed.Value.(workspace.ChangeReview)
			core.AssertEqual(t, 1, len(changes.Validation))
			core.AssertTrue(t, changes.Validation[0].Passed, changes.Validation[0].Output)
			core.AssertContains(t, changes.CommitLog, core.Concat(providerName, " native receipt"))
			core.AssertEqual(t, before, orchestratorAcceptanceSourceSnapshot(t, project.RepositoryRoot))

			accepted := fixture.orchestrator.Accept(context.Background(), workspace.AcceptRequest{Review: changes, Confirmed: true})
			core.AssertTrue(t, accepted.OK, accepted.Error())
			core.AssertEqual(t, "accepted", accepted.Value.(work.Acceptance).Status)
			core.AssertEqual(t, changes.ResultRevision, orchestratorRunGit(t, project.RepositoryRoot, "rev-parse", "HEAD"))
			content := core.ReadFile(core.PathJoin(project.RepositoryRoot, "native-receipt.txt"))
			core.AssertTrue(t, content.OK, content.Error())
			core.AssertEqual(t, core.Concat(providerName, " native receipt\n"), string(content.Value.([]byte)))
			core.AssertEqual(t, "", orchestratorRunGit(t, project.RepositoryRoot, "status", "--porcelain=v1", "--untracked-files=all"))
			core.AssertTrue(t, fixture.orchestrator.Close().OK)
			health := fixture.server.Health(context.Background())
			core.AssertTrue(t, health.OK, health.Error())
			core.AssertFalse(t, health.Value.(gitserver.Health).Running)
		})
	}
}

func TestNativeInterruptionRecoveryAndResumeReceipt(t *testing.T) {
	fixture := newOrchestratorFixture(t)
	executable := nativeInterruptedReceiptScript(t)
	nativeReceiptRuntime(t, fixture, "codex", executable)
	item, project, sourceRevision := nativeReceiptRegister(t, fixture, "codex")
	_, run := nativeReceiptDispatch(t, fixture, item, "codex", sourceRevision)
	core.AssertTrue(t, fixture.orchestrator.StartQueue(context.Background()).OK)
	orchestratorWaitRunStatus(t, fixture.store, run.ID, work.RunRunning)
	childLog := nativeReceiptWaitLog(t, fixture, run.ID, "child:")
	childResult := core.Atoi(core.TrimPrefix(childLog.Text, "child:"))
	core.AssertTrue(t, childResult.OK, childResult.Error())
	childPID := childResult.Value.(int)
	runningResult := fixture.store.Run(run.ID)
	core.AssertTrue(t, runningResult.OK, runningResult.Error())
	parentPID := runningResult.Value.(work.Run).ProcessID
	core.AssertTrue(t, processapi.PIDAlive(parentPID))
	core.AssertTrue(t, processapi.PIDAlive(childPID))

	core.AssertTrue(t, fixture.orchestrator.Close().OK)
	interrupted := orchestratorWaitRunStatus(t, fixture.store, run.ID, work.RunInterrupted)
	core.AssertEqual(t, run.ID, interrupted.ID)
	core.AssertTrue(t, interrupted.DurableRevision != "")
	core.AssertTrue(t, nativeWaitDead(parentPID, 3*time.Second))
	core.AssertTrue(t, nativeWaitDead(childPID, 3*time.Second))
	logs := fixture.store.Snapshot(item.ID).Value.(work.Snapshot).Logs
	core.AssertTrue(t, len(logs) >= 2)
	core.AssertEqual(t, "interrupt-one", logs[0].Text)
	core.AssertEqual(t, childLog.Text, logs[1].Text)
	privateRepository := core.PathJoin(fixture.server.root, core.Concat(project.RepositoryName, ".git"))
	core.AssertEqual(t, interrupted.DurableRevision, orchestratorRunGit(t, privateRepository, "rev-parse", core.Concat("refs/heads/", interrupted.Branch)))
	core.AssertFalse(t, core.Stat(core.PathJoin(project.RepositoryRoot, "interrupted-receipt.txt")).OK)
	orchestratorAssertNoStateMarkdown(t, project.RepositoryRoot)
	core.AssertFalse(t, core.Stat(core.PathJoin(project.RepositoryRoot, ".lem")).OK)

	nativeReceiptRuntime(t, fixture, "codex", executable)
	retried := fixture.orchestrator.Retry(context.Background(), item, interrupted.ID)
	core.AssertTrue(t, retried.OK, retried.Error())
	childRun := retried.Value.(work.Run)
	core.AssertTrue(t, childRun.ID != interrupted.ID)
	core.AssertEqual(t, interrupted.ID, childRun.ParentRunID)
	core.AssertEqual(t, interrupted.Branch, childRun.Branch)
	core.AssertTrue(t, fixture.orchestrator.StartQueue(context.Background()).OK)
	completed := orchestratorWaitRunStatus(t, fixture.store, childRun.ID, work.RunCompleted)
	core.AssertEqual(t, childRun.ID, completed.ID)
	core.AssertTrue(t, completed.DurableRevision != interrupted.DurableRevision)
	resumeOne := nativeReceiptWaitLog(t, fixture, childRun.ID, "resume-one")
	resumeTwo := nativeReceiptWaitLog(t, fixture, childRun.ID, "resume-two")
	core.AssertTrue(t, resumeOne.Sequence < resumeTwo.Sequence)

	reviewed := fixture.orchestrator.ReviewChanges(context.Background(), childRun.ID)
	core.AssertTrue(t, reviewed.OK, reviewed.Error())
	changes := reviewed.Value.(workspace.ChangeReview)
	core.AssertTrue(t, changes.Validation[0].Passed, changes.Validation[0].Output)
	accepted := fixture.orchestrator.Accept(context.Background(), workspace.AcceptRequest{Review: changes, Confirmed: true})
	core.AssertTrue(t, accepted.OK, accepted.Error())
	for path, want := range map[string]string{
		"interrupted-receipt.txt": "interrupted\n",
		"resumed-receipt.txt":     "resumed\n",
	} {
		content := core.ReadFile(core.PathJoin(project.RepositoryRoot, path))
		core.AssertTrue(t, content.OK, content.Error())
		core.AssertEqual(t, want, string(content.Value.([]byte)))
	}
	core.AssertEqual(t, changes.ResultRevision, orchestratorRunGit(t, project.RepositoryRoot, "rev-parse", "HEAD"))
	core.AssertEqual(t, "", orchestratorRunGit(t, project.RepositoryRoot, "status", "--porcelain=v1", "--untracked-files=all"))
	core.AssertTrue(t, fixture.orchestrator.Close().OK)
	health := fixture.server.Health(context.Background())
	core.AssertTrue(t, health.OK, health.Error())
	core.AssertFalse(t, health.Value.(gitserver.Health).Running)
}
