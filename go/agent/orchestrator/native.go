// SPDX-License-Identifier: EUPL-1.2

package orchestrator

import (
	"context"
	"sync"
	"time"

	core "dappco.re/go"
	"dappco.re/go/inference/agent/provider"
	coreprocess "dappco.re/go/process"
)

// Process is one owned native provider process group.
type Process interface {
	ID() string
	PID() int
	Wait() core.Result
	Shutdown() core.Result
}

// Launcher isolates native process creation, output routing, and shutdown.
type Launcher interface {
	DetectEnvironment([]string) core.Result
	Start(context.Context, provider.Command, func(stream, line string)) core.Result
	Close() core.Result
}

// Clock supplies deterministic orchestration time.
type Clock interface {
	Now() time.Time
}

// Identifier supplies durable unique identifiers.
type Identifier interface {
	New() string
}

type nativeOutput struct {
	stream string
	line   string
}

type nativeRoute struct {
	mu       sync.Mutex
	callback func(string, string)
	secrets  []string
}

type nativeLauncher struct {
	service    *coreprocess.Service
	envPath    string
	essentials []string
	mu         sync.Mutex
	routes     map[string]*nativeRoute
	pending    map[string][]nativeOutput
	processes  map[string]*nativeProcess
	starting   int
	closed     bool
}

type nativeProcess struct {
	process  *coreprocess.ManagedProcess
	launcher *nativeLauncher
	waitOnce sync.Once
	waited   core.Result
}

type nativeEnvironment struct {
	assignments []string
	secrets     []string
}

// NewNativeLauncher binds a process service to an isolated environment runner.
func NewNativeLauncher(service *coreprocess.Service, essentials []string) core.Result {
	if service == nil {
		return core.Fail(core.NewError("agent native launcher process service is required"))
	}
	if service.Core() == nil {
		return core.Fail(core.NewError("agent native launcher process service requires a Core action bus"))
	}
	validated := uniqueEnvironmentNames(essentials)
	if !validated.OK {
		return core.Fail(core.E("orchestrator.NewNativeLauncher", "invalid essential environment", validated.Err()))
	}
	envProgram := &coreprocess.Program{Name: "env"}
	if found := envProgram.Find(); !found.OK {
		return core.Fail(core.E("orchestrator.NewNativeLauncher", "platform env program is unavailable", found.Err()))
	}
	launcher := &nativeLauncher{
		service:    service,
		envPath:    envProgram.Path,
		essentials: validated.Value.([]string),
		routes:     make(map[string]*nativeRoute),
		pending:    make(map[string][]nativeOutput),
		processes:  make(map[string]*nativeProcess),
	}
	service.Core().RegisterAction(func(_ *core.Core, message core.Message) core.Result {
		return launcher.handleAction(message)
	})
	return core.Ok(Launcher(launcher))
}

func (launcher *nativeLauncher) DetectEnvironment(credentials []string) core.Result {
	if launcher == nil {
		return core.Fail(core.NewError("agent native launcher is required"))
	}
	launcher.mu.Lock()
	closed := launcher.closed
	essentials := append([]string(nil), launcher.essentials...)
	launcher.mu.Unlock()
	if closed {
		return core.Fail(core.NewError("agent native launcher is closed"))
	}
	credentialResult := uniqueEnvironmentNames(credentials)
	if !credentialResult.OK {
		return credentialResult
	}
	keys := append(essentials, credentialResult.Value.([]string)...)
	seen := make(map[string]struct{}, len(keys))
	assignments := make([]string, 0, len(keys))
	for _, key := range keys {
		if _, exists := seen[key]; exists {
			continue
		}
		seen[key] = struct{}{}
		value, exists := core.LookupEnv(key)
		if !exists {
			continue
		}
		if !safeEnvironmentValue(value) {
			return core.Fail(core.Errorf("agent native environment %s contains a forbidden control character", key))
		}
		assignments = append(assignments, core.Concat(key, "=", value))
	}
	return core.Ok(assignments)
}

func (launcher *nativeLauncher) Start(ctx context.Context, command provider.Command, callback func(stream, line string)) core.Result {
	if launcher == nil {
		return core.Fail(core.NewError("agent native launcher is required"))
	}
	if ctx == nil {
		return core.Fail(core.NewError("agent native launch context is required"))
	}
	if callback == nil {
		return core.Fail(core.NewError("agent native launch output callback is required"))
	}
	validated := validateNativeCommand(command)
	if !validated.OK {
		return validated
	}
	command = validated.Value.(provider.Command)
	environmentResult := launcher.commandEnvironment(command)
	if !environmentResult.OK {
		return environmentResult
	}
	environment := environmentResult.Value.(nativeEnvironment)
	for _, secret := range environment.secrets {
		if secret != "" && core.Contains(command.Receipt, secret) {
			return core.Fail(core.NewError("agent native command receipt contains a credential value"))
		}
	}

	launcher.mu.Lock()
	if launcher.closed {
		launcher.mu.Unlock()
		return core.Fail(core.NewError("agent native launcher is closed"))
	}
	launcher.starting++
	launcher.mu.Unlock()

	arguments := make([]string, 0, len(environment.assignments)+len(command.Args)+2)
	arguments = append(arguments, "-i")
	arguments = append(arguments, environment.assignments...)
	arguments = append(arguments, command.Executable)
	arguments = append(arguments, command.Args...)
	started := launcher.service.StartWithOptions(ctx, coreprocess.RunOptions{
		Command:        launcher.envPath,
		Args:           arguments,
		Dir:            command.Dir,
		DisableCapture: true,
		Detach:         true,
		KillGroup:      true,
		GracePeriod:    3 * time.Second,
	})
	if !started.OK {
		launcher.finishStart("")
		return core.Fail(core.Errorf("agent native provider failed to start: %s", redactText(started.Error(), environment.secrets)))
	}
	managed, ok := started.Value.(*coreprocess.ManagedProcess)
	if !ok {
		launcher.finishStart("")
		return core.Fail(core.Errorf("agent native process service returned %T instead of process", started.Value))
	}
	handle := &nativeProcess{process: managed, launcher: launcher}
	route := &nativeRoute{callback: callback, secrets: append([]string(nil), environment.secrets...)}
	route.mu.Lock()
	launcher.mu.Lock()
	launcher.starting--
	if launcher.closed {
		if launcher.starting == 0 {
			launcher.pending = make(map[string][]nativeOutput)
		}
		launcher.mu.Unlock()
		route.mu.Unlock()
		shutdown := managed.Shutdown()
		waited := managed.Wait()
		if !shutdown.OK {
			return shutdown
		}
		if !waited.OK {
			return waited
		}
		return core.Fail(core.NewError("agent native launcher closed during process start"))
	}
	pending := append([]nativeOutput(nil), launcher.pending[managed.ID]...)
	delete(launcher.pending, managed.ID)
	launcher.routes[managed.ID] = route
	launcher.processes[managed.ID] = handle
	if launcher.starting == 0 {
		launcher.pending = make(map[string][]nativeOutput)
	}
	launcher.mu.Unlock()
	for _, output := range pending {
		route.deliverLocked(output)
	}
	route.mu.Unlock()
	return core.Ok(Process(handle))
}

func (launcher *nativeLauncher) Close() core.Result {
	if launcher == nil {
		return core.Fail(core.NewError("agent native launcher is required"))
	}
	launcher.mu.Lock()
	launcher.closed = true
	processes := make([]*nativeProcess, 0, len(launcher.processes))
	for _, process := range launcher.processes {
		processes = append(processes, process)
	}
	launcher.mu.Unlock()

	failures := make([]string, 0, len(processes)*2)
	for _, process := range processes {
		if shutdown := process.Shutdown(); !shutdown.OK {
			failures = append(failures, shutdown.Error())
		}
	}
	for _, process := range processes {
		if waited := process.Wait(); !waited.OK {
			failures = append(failures, waited.Error())
		}
	}
	launcher.mu.Lock()
	launcher.routes = make(map[string]*nativeRoute)
	launcher.pending = make(map[string][]nativeOutput)
	launcher.mu.Unlock()
	if len(failures) > 0 {
		return core.Fail(core.NewError(core.Join("; ", failures...)))
	}
	return core.Ok(nil)
}

func (launcher *nativeLauncher) commandEnvironment(command provider.Command) core.Result {
	detected := launcher.DetectEnvironment(command.CredentialKeys)
	if !detected.OK {
		return detected
	}
	credentialResult := uniqueEnvironmentNames(command.CredentialKeys)
	if !credentialResult.OK {
		return credentialResult
	}
	credentials := credentialResult.Value.([]string)
	allowed := make(map[string]bool, len(launcher.essentials)+len(credentials))
	order := make([]string, 0, len(launcher.essentials)+len(credentials))
	for _, key := range append(append([]string(nil), launcher.essentials...), credentials...) {
		if _, exists := allowed[key]; exists {
			continue
		}
		allowed[key] = false
		order = append(order, key)
	}
	for _, key := range credentials {
		allowed[key] = true
	}
	values := make(map[string]string, len(order))
	for _, assignment := range detected.Value.([]string) {
		key, value, found := core.Cut(assignment, "=")
		if found {
			values[key] = value
		}
	}
	for _, assignment := range command.Environment {
		key, value, found := core.Cut(assignment, "=")
		if !found || !validEnvironmentName(key) {
			return core.Fail(core.NewError("agent native explicit environment assignment is malformed"))
		}
		secret, accepted := allowed[key]
		if !accepted {
			return core.Fail(core.Errorf("agent native environment %s is not allowlisted", key))
		}
		if !safeEnvironmentValue(value) {
			return core.Fail(core.Errorf("agent native environment %s contains a forbidden control character", key))
		}
		allowed[key] = secret
		values[key] = value
	}
	assignments := make([]string, 0, len(values))
	secrets := make([]string, 0, len(credentials))
	for _, key := range order {
		value, exists := values[key]
		if !exists {
			continue
		}
		assignments = append(assignments, core.Concat(key, "=", value))
		if allowed[key] && value != "" {
			secrets = append(secrets, value)
		}
	}
	core.SliceSortFunc(secrets, func(left, right string) bool { return len(left) > len(right) })
	return core.Ok(nativeEnvironment{assignments: assignments, secrets: secrets})
}

func (launcher *nativeLauncher) handleAction(message core.Message) core.Result {
	output, ok := message.(coreprocess.ActionProcessOutput)
	if !ok {
		return core.Ok(nil)
	}
	event := nativeOutput{stream: string(output.Stream), line: output.Line}
	launcher.mu.Lock()
	if launcher.closed {
		launcher.mu.Unlock()
		return core.Ok(nil)
	}
	route := launcher.routes[output.ID]
	if route == nil {
		if launcher.starting > 0 {
			launcher.pending[output.ID] = append(launcher.pending[output.ID], event)
		}
		launcher.mu.Unlock()
		return core.Ok(nil)
	}
	launcher.mu.Unlock()
	route.deliver(event)
	return core.Ok(nil)
}

func (launcher *nativeLauncher) finishStart(processID string) {
	launcher.mu.Lock()
	if launcher.starting > 0 {
		launcher.starting--
	}
	if processID != "" {
		delete(launcher.pending, processID)
	}
	if launcher.starting == 0 {
		launcher.pending = make(map[string][]nativeOutput)
	}
	launcher.mu.Unlock()
}

func (launcher *nativeLauncher) release(processID string) core.Result {
	launcher.mu.Lock()
	delete(launcher.routes, processID)
	delete(launcher.pending, processID)
	delete(launcher.processes, processID)
	launcher.mu.Unlock()
	return launcher.service.Remove(processID)
}

func (route *nativeRoute) deliver(output nativeOutput) {
	route.mu.Lock()
	defer route.mu.Unlock()
	route.deliverLocked(output)
}

func (route *nativeRoute) deliverLocked(output nativeOutput) {
	delivered := core.Try(func() any {
		route.callback(output.stream, redactText(output.line, route.secrets))
		return nil
	})
	if !delivered.OK {
		return
	}
}

func (process *nativeProcess) ID() string {
	if process == nil || process.process == nil {
		return ""
	}
	return process.process.ID
}

func (process *nativeProcess) PID() int {
	if process == nil || process.process == nil {
		return 0
	}
	return process.process.Info().PID
}

func (process *nativeProcess) Wait() core.Result {
	if process == nil || process.process == nil || process.launcher == nil {
		return core.Fail(core.NewError("agent native process is required"))
	}
	process.waitOnce.Do(func() {
		<-process.process.Done()
		exitCode := process.process.Info().ExitCode
		released := process.launcher.release(process.process.ID)
		if !released.OK {
			process.waited = released
			return
		}
		process.waited = core.Ok(exitCode)
	})
	return process.waited
}

func (process *nativeProcess) Shutdown() core.Result {
	if process == nil || process.process == nil {
		return core.Fail(core.NewError("agent native process is required"))
	}
	return process.process.Shutdown()
}

func validateNativeCommand(command provider.Command) core.Result {
	command.Provider = core.Trim(command.Provider)
	command.Executable = core.Trim(command.Executable)
	command.Dir = core.Trim(command.Dir)
	if command.Provider == "" || command.Executable == "" || command.Dir == "" {
		return core.Fail(core.NewError("agent native command requires provider, executable, and working directory"))
	}
	if !core.PathIsAbs(command.Dir) {
		return core.Fail(core.NewError("agent native command working directory must be absolute"))
	}
	if core.Contains(command.Executable, "\x00") {
		return core.Fail(core.NewError("agent native command executable contains NUL"))
	}
	for _, argument := range command.Args {
		if core.Contains(argument, "\x00") {
			return core.Fail(core.NewError("agent native command argument contains NUL"))
		}
	}
	command.Args = append([]string(nil), command.Args...)
	command.Environment = append([]string(nil), command.Environment...)
	command.CredentialKeys = append([]string(nil), command.CredentialKeys...)
	return core.Ok(command)
}

func uniqueEnvironmentNames(configured []string) core.Result {
	seen := make(map[string]struct{}, len(configured))
	names := make([]string, 0, len(configured))
	for _, name := range configured {
		name = core.Trim(name)
		if !validEnvironmentName(name) {
			return core.Fail(core.Errorf("agent native environment name %q is invalid", name))
		}
		if _, exists := seen[name]; exists {
			return core.Fail(core.Errorf("agent native environment name %s is duplicated", name))
		}
		seen[name] = struct{}{}
		names = append(names, name)
	}
	return core.Ok(names)
}

func validEnvironmentName(name string) bool {
	if name == "" {
		return false
	}
	for index := 0; index < len(name); index++ {
		character := name[index]
		if index == 0 {
			if character != '_' && (character < 'A' || character > 'Z') && (character < 'a' || character > 'z') {
				return false
			}
			continue
		}
		if character != '_' && (character < 'A' || character > 'Z') && (character < 'a' || character > 'z') && (character < '0' || character > '9') {
			return false
		}
	}
	return true
}

func safeEnvironmentValue(value string) bool {
	return !core.Contains(value, "\x00") && !core.Contains(value, "\n") && !core.Contains(value, "\r")
}

func redactText(text string, secrets []string) string {
	for _, secret := range secrets {
		if secret != "" {
			text = core.Replace(text, secret, "[REDACTED]")
		}
	}
	return text
}
