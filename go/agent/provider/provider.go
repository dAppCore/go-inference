// SPDX-License-Identifier: EUPL-1.2

// Package provider builds safe native agent commands and normalizes their output.
package provider

import (
	"context"

	core "dappco.re/go"
	coreprocess "dappco.re/go/process"
)

// Launch is the complete durable context passed to one native provider attempt.
type Launch struct {
	WorkID       string
	RunID        string
	Title        string
	Task         string
	Worktree     string
	Branch       string
	Model        string
	Continuation string
	UnsafeFlags  []string
}

// Command is an explicit executable and argument vector. It is never a shell command.
type Command struct {
	Provider       string
	Executable     string
	Dir            string
	Args           []string
	Environment    []string
	CredentialKeys []string
	Receipt        string
}

// Detection describes whether one native provider executable is usable.
type Detection struct {
	Provider   string
	Executable string
	Version    string
	Available  bool
	Reason     string
}

// Output is one normalized provider event or log line.
type Output struct {
	Kind       string
	Text       string
	DetailJSON string
	RetryAfter string
	UsageJSON  string
}

// FinalStatus is the validated machine-readable outcome from a provider response.
type FinalStatus struct {
	Status   string `json:"status"`
	Summary  string `json:"summary,omitempty"`
	Question string `json:"question,omitempty"`
	Reason   string `json:"reason,omitempty"`
}

// Finder isolates executable discovery and version probing for native adapters.
type Finder interface {
	Find(name string) core.Result
	Version(context.Context, string) core.Result
}

// Adapter is the native-provider boundary consumed by the orchestrator.
type Adapter interface {
	Name() string
	Detect(context.Context) core.Result
	Build(Launch) core.Result
	ParseLine(stream, line string) []Output
}

// Config customizes one native provider without changing its safe defaults.
type Config struct {
	Executable    string
	DefaultModel  string
	CredentialEnv []string
	Flags         []string
}

// Registry is an immutable ordered set of named native adapters.
type Registry struct {
	adapters map[string]Adapter
	names    []string
}

type commandBuilder func(Config, Launch, string) []string
type lineParser func(string, string) []Output

// NativeAdapter implements Adapter for a non-interactive native CLI.
type NativeAdapter struct {
	name      string
	config    Config
	finder    Finder
	buildArgs commandBuilder
	parse     lineParser
}

type systemFinder struct{}

// NewRegistry validates and isolates an ordered adapter set.
func NewRegistry(adapters ...Adapter) core.Result {
	if len(adapters) == 0 {
		return core.Fail(core.NewError("agent provider registry requires at least one adapter"))
	}
	registry := &Registry{
		adapters: make(map[string]Adapter, len(adapters)),
		names:    make([]string, 0, len(adapters)),
	}
	for _, adapter := range adapters {
		if adapter == nil {
			return core.Fail(core.NewError("agent provider registry adapter is required"))
		}
		name := core.Lower(core.Trim(adapter.Name()))
		if name == "" {
			return core.Fail(core.NewError("agent provider registry adapter name is required"))
		}
		if _, exists := registry.adapters[name]; exists {
			return core.Fail(core.Errorf("agent provider registry contains duplicate adapter %q", name))
		}
		registry.adapters[name] = adapter
		registry.names = append(registry.names, name)
	}
	return core.Ok(registry)
}

// DefaultRegistry constructs the supported Codex, Claude, and OpenCode adapters.
func DefaultRegistry(finder Finder, configurations map[string]Config) core.Result {
	if finder == nil {
		finder = systemFinder{}
	}
	normalized := make(map[string]Config, len(configurations))
	for configuredName, configuration := range configurations {
		name := core.Lower(core.Trim(configuredName))
		if name != "codex" && name != "claude" && name != "opencode" {
			return core.Fail(core.Errorf("agent provider %q is not supported", configuredName))
		}
		if _, exists := normalized[name]; exists {
			return core.Fail(core.Errorf("agent provider configuration %q is duplicated", name))
		}
		validated := normalizeConfig(name, configuration)
		if !validated.OK {
			return core.Fail(core.E("provider.DefaultRegistry", core.Concat("invalid ", name, " configuration"), validated.Err()))
		}
		normalized[name] = validated.Value.(Config)
	}

	definitions := []struct {
		name    string
		builder commandBuilder
		parser  lineParser
	}{
		{name: "codex", builder: buildCodexArgs, parser: parseCodexLine},
		{name: "claude", builder: buildClaudeArgs, parser: parseClaudeLine},
		{name: "opencode", builder: buildOpenCodeArgs, parser: parseOpenCodeLine},
	}
	adapters := make([]Adapter, 0, len(definitions))
	for _, definition := range definitions {
		configuration, configured := normalized[definition.name]
		if !configured {
			configuration = Config{Executable: definition.name}
		}
		adapters = append(adapters, &NativeAdapter{
			name:      definition.name,
			config:    cloneConfig(configuration),
			finder:    finder,
			buildArgs: definition.builder,
			parse:     definition.parser,
		})
	}
	return NewRegistry(adapters...)
}

// Adapter returns a named adapter without exposing the registry's map.
func (registry *Registry) Adapter(name string) core.Result {
	if registry == nil {
		return core.Fail(core.NewError("agent provider registry is required"))
	}
	name = core.Lower(core.Trim(name))
	if name == "" {
		return core.Fail(core.NewError("agent provider name is required"))
	}
	adapter, exists := registry.adapters[name]
	if !exists {
		return core.Fail(core.Errorf("agent provider %q is not registered", name))
	}
	return core.Ok(adapter)
}

// Names returns the registry's stable provider order as a detached slice.
func (registry *Registry) Names() []string {
	if registry == nil {
		return nil
	}
	return append([]string(nil), registry.names...)
}

// Name returns the adapter's canonical provider name.
func (adapter *NativeAdapter) Name() string {
	if adapter == nil {
		return ""
	}
	return core.Lower(core.Trim(adapter.name))
}

// Detect locates the configured executable and probes its version.
func (adapter *NativeAdapter) Detect(ctx context.Context) core.Result {
	if adapter == nil || adapter.finder == nil || adapter.Name() == "" {
		return core.Fail(core.NewError("native agent provider adapter is not configured"))
	}
	if ctx == nil {
		return core.Fail(core.NewError("native agent provider detection context is required"))
	}
	executable := core.Trim(adapter.config.Executable)
	if executable == "" {
		return core.Fail(core.NewError("native agent provider executable is required"))
	}
	found := adapter.finder.Find(executable)
	if !found.OK {
		return core.Ok(Detection{
			Provider: adapter.Name(),
			Reason:   found.Error(),
		})
	}
	path, ok := found.Value.(string)
	if !ok || core.Trim(path) == "" {
		return core.Fail(core.NewError("native agent provider finder returned an invalid executable path"))
	}
	versionResult := adapter.finder.Version(ctx, path)
	if !versionResult.OK {
		return core.Ok(Detection{
			Provider:   adapter.Name(),
			Executable: path,
			Reason:     versionResult.Error(),
		})
	}
	version, ok := versionResult.Value.(string)
	if !ok || core.Trim(version) == "" {
		return core.Fail(core.NewError("native agent provider finder returned an invalid version"))
	}
	return core.Ok(Detection{
		Provider:   adapter.Name(),
		Executable: path,
		Version:    core.Trim(version),
		Available:  true,
	})
}

// Build validates one launch and returns a shell-free native command.
func (adapter *NativeAdapter) Build(launch Launch) core.Result {
	if adapter == nil || adapter.Name() == "" || adapter.buildArgs == nil {
		return core.Fail(core.NewError("native agent provider adapter is not configured"))
	}
	normalized := normalizeLaunch(launch)
	if !normalized.OK {
		return normalized
	}
	launch = normalized.Value.(Launch)
	prompt := buildPrompt(launch)
	args := adapter.buildArgs(adapter.config, launch, prompt)
	command := Command{
		Provider:       adapter.Name(),
		Executable:     adapter.config.Executable,
		Dir:            launch.Worktree,
		Args:           append([]string(nil), args...),
		CredentialKeys: append([]string(nil), adapter.config.CredentialEnv...),
	}
	command.Receipt = renderReceipt(command.Executable, command.Args)
	return core.Ok(command)
}

// ParseLine normalizes one stdout or stderr line without rejecting raw output.
func (adapter *NativeAdapter) ParseLine(stream, line string) []Output {
	if adapter == nil || adapter.parse == nil {
		return nil
	}
	return adapter.parse(core.Lower(core.Trim(stream)), core.Trim(line))
}

func (systemFinder) Find(name string) core.Result {
	program := &coreprocess.Program{Name: name}
	result := program.Find()
	if !result.OK {
		return result
	}
	return core.Ok(program.Path)
}

func (systemFinder) Version(ctx context.Context, executable string) core.Result {
	program := &coreprocess.Program{Path: executable}
	return program.Run(ctx, "--version")
}

func normalizeConfig(name string, configuration Config) core.Result {
	configuration.Executable = core.Trim(configuration.Executable)
	if configuration.Executable == "" {
		configuration.Executable = name
	}
	configuration.DefaultModel = core.Trim(configuration.DefaultModel)
	credentials := make([]string, len(configuration.CredentialEnv))
	for index, configured := range configuration.CredentialEnv {
		credential := core.Trim(configured)
		if !validEnvironmentName(credential) {
			return core.Fail(core.Errorf("credential environment name %q is invalid", credential))
		}
		credentials[index] = credential
	}
	flags := make([]string, len(configuration.Flags))
	for index, configured := range configuration.Flags {
		flag := core.Trim(configured)
		if flag == "" {
			return core.Fail(core.NewError("native provider flag cannot be empty"))
		}
		flags[index] = flag
	}
	configuration.CredentialEnv = credentials
	configuration.Flags = flags
	return core.Ok(configuration)
}

func cloneConfig(configuration Config) Config {
	configuration.CredentialEnv = append([]string(nil), configuration.CredentialEnv...)
	configuration.Flags = append([]string(nil), configuration.Flags...)
	return configuration
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

func normalizeLaunch(launch Launch) core.Result {
	launch.WorkID = core.Trim(launch.WorkID)
	launch.RunID = core.Trim(launch.RunID)
	launch.Title = core.Trim(launch.Title)
	launch.Task = core.Trim(launch.Task)
	launch.Worktree = core.Trim(launch.Worktree)
	launch.Branch = core.Trim(launch.Branch)
	launch.Model = core.Trim(launch.Model)
	launch.Continuation = core.Trim(launch.Continuation)
	required := []struct {
		name  string
		value string
	}{
		{name: "work ID", value: launch.WorkID},
		{name: "run ID", value: launch.RunID},
		{name: "title", value: launch.Title},
		{name: "task", value: launch.Task},
		{name: "worktree", value: launch.Worktree},
		{name: "branch", value: launch.Branch},
	}
	for _, field := range required {
		if field.value == "" {
			return core.Fail(core.Errorf("native agent provider launch %s is required", field.name))
		}
	}
	flags := make([]string, len(launch.UnsafeFlags))
	for index, configured := range launch.UnsafeFlags {
		flag := core.Trim(configured)
		if flag == "" {
			return core.Fail(core.NewError("native agent provider unsafe flag cannot be empty"))
		}
		flags[index] = flag
	}
	launch.UnsafeFlags = flags
	return core.Ok(launch)
}

func buildPrompt(launch Launch) string {
	prompt := core.Concat(
		"You are working on LEM work item ", launch.WorkID, " (run ", launch.RunID, ").\n",
		"Title: ", launch.Title, "\n",
		"Task:\n", launch.Task, "\n\n",
		"Work only in this isolated Git worktree: ", launch.Worktree, "\n",
		"Branch: ", launch.Branch, "\n",
		"Do not modify the source checkout or any path outside this worktree.\n",
		"Commit coherent changes to this branch before reporting completion.",
	)
	if launch.Continuation != "" {
		prompt = core.Concat(prompt, "\n\nContinuation context from the immutable parent attempt:\n", launch.Continuation)
	}
	return core.Concat(
		prompt,
		"\n\nEnd your final response with exactly one machine-readable status envelope:\n",
		`<<<LEM_STATUS>>>{"status":"completed","summary":"short result"}<<<END_LEM_STATUS>>>`, "\n",
		"or\n",
		`<<<LEM_STATUS>>>{"status":"waiting","question":"one concrete question"}<<<END_LEM_STATUS>>>`, "\n",
		"or\n",
		`<<<LEM_STATUS>>>{"status":"failed","reason":"short reason"}<<<END_LEM_STATUS>>>`, "\n",
		"Do not emit more than one status envelope.",
	)
}

func renderReceipt(executable string, args []string) string {
	parts := make([]string, 0, len(args)+1)
	parts = append(parts, quoteReceiptArgument(executable))
	redactNext := false
	for _, arg := range args {
		shown := arg
		if redactNext {
			shown = "<redacted>"
			redactNext = false
		}
		lower := core.Lower(arg)
		if isSensitiveFlag(lower) {
			redactNext = true
		}
		if before, _, found := core.Cut(arg, "="); found && isSensitiveFlag(core.Lower(before)) {
			shown = core.Concat(before, "=<redacted>")
			redactNext = false
		}
		parts = append(parts, quoteReceiptArgument(shown))
	}
	return core.Join(" ", parts...)
}

func isSensitiveFlag(value string) bool {
	return core.Contains(value, "token") ||
		core.Contains(value, "secret") ||
		core.Contains(value, "password") ||
		core.Contains(value, "credential") ||
		core.Contains(value, "api-key") ||
		core.Contains(value, "api_key")
}

func quoteReceiptArgument(argument string) string {
	if argument != "" && !core.ContainsAny(argument, " \t\r\n'\"") {
		return argument
	}
	return core.Concat("'", core.Replace(argument, "'", `'"'"'`), "'")
}
