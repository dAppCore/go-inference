// SPDX-License-Identifier: EUPL-1.2

package orchestrator

import (
	"context"
	"slices"
	"sync"
	"testing"
	"time"

	core "dappco.re/go"
	"dappco.re/go/inference/agent/provider"
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
