// SPDX-License-Identifier: EUPL-1.2

package queue

import (
	core "dappco.re/go"
	coreconfig "dappco.re/go/config"
	coreio "dappco.re/go/io"
)

const (
	policyVersion                 = 1
	defaultDispatchTimeoutMinutes = 60
)

// Command is an explicit validation command and argument vector.
type Command struct {
	Command string   `yaml:"command" mapstructure:"command"`
	Args    []string `yaml:"args" mapstructure:"args"`
}

// DispatchConfig controls default selection and global admission.
type DispatchConfig struct {
	DefaultAgent      string    `yaml:"default_agent" mapstructure:"default_agent"`
	GlobalConcurrency int       `yaml:"global_concurrency" mapstructure:"global_concurrency"`
	TimeoutMinutes    int       `yaml:"timeout_minutes" mapstructure:"timeout_minutes"`
	Validation        []Command `yaml:"validation" mapstructure:"validation"`
}

// ConcurrencyLimit sets a provider total and optional per-model limits.
type ConcurrencyLimit struct {
	Total  int
	Models map[string]int
}

// RateConfig controls provider pacing and a UTC quota window.
type RateConfig struct {
	ResetUTC       string `yaml:"reset_utc" mapstructure:"reset_utc"`
	DailyLimit     int    `yaml:"daily_limit" mapstructure:"daily_limit"`
	MinDelay       int    `yaml:"min_delay" mapstructure:"min_delay"`
	SustainedDelay int    `yaml:"sustained_delay" mapstructure:"sustained_delay"`
	BurstWindow    int    `yaml:"burst_window" mapstructure:"burst_window"`
	BurstDelay     int    `yaml:"burst_delay" mapstructure:"burst_delay"`
}

// NativeConfig is additive policy for one native provider executable.
type NativeConfig struct {
	Executable    string   `yaml:"executable" mapstructure:"executable"`
	DefaultModel  string   `yaml:"default_model" mapstructure:"default_model"`
	CredentialEnv []string `yaml:"credential_env" mapstructure:"credential_env"`
	Flags         []string `yaml:"flags" mapstructure:"flags"`
}

// Policy is the validated CoreAgent-compatible queue configuration.
type Policy struct {
	Version     int                         `yaml:"version" mapstructure:"version"`
	Dispatch    DispatchConfig              `yaml:"dispatch" mapstructure:"dispatch"`
	Concurrency map[string]ConcurrencyLimit `yaml:"-" mapstructure:"-"`
	Rates       map[string]RateConfig       `yaml:"rates" mapstructure:"rates"`
	Providers   map[string]NativeConfig     `yaml:"providers" mapstructure:"providers"`
}

type policyWire struct {
	Version     int                     `mapstructure:"version"`
	Dispatch    DispatchConfig          `mapstructure:"dispatch"`
	Concurrency map[string]any          `mapstructure:"concurrency"`
	Rates       map[string]RateConfig   `mapstructure:"rates"`
	Providers   map[string]NativeConfig `mapstructure:"providers"`
}

// LoadPolicy reads, validates, and isolates an agents.yaml policy.
func LoadPolicy(medium coreio.Medium, path string) core.Result {
	if medium == nil {
		return core.Fail(core.NewError("agent queue policy medium is required"))
	}
	path = core.Trim(path)
	if path == "" {
		return core.Fail(core.NewError("agent queue policy path is required"))
	}
	if !medium.Exists(path) {
		return core.Ok(defaultPolicy())
	}

	configResult := coreconfig.New(coreconfig.WithMedium(medium), coreconfig.WithPath(path))
	if !configResult.OK {
		return core.Fail(core.E("queue.LoadPolicy", core.Concat("failed to load ", path), configResult.Err()))
	}
	configuration, ok := configResult.Value.(*coreconfig.Config)
	if !ok || configuration == nil {
		return core.Fail(core.NewError("agent queue policy loader returned an invalid configuration"))
	}

	var wire policyWire
	decodeResult := configuration.Get("", &wire)
	if !decodeResult.OK {
		return core.Fail(core.E("queue.LoadPolicy", core.Concat("failed to decode ", path), decodeResult.Err()))
	}
	concurrencyResult := decodeConcurrency(wire.Concurrency)
	if !concurrencyResult.OK {
		return core.Fail(core.E("queue.LoadPolicy", core.Concat("invalid concurrency in ", path), concurrencyResult.Err()))
	}

	policy := Policy{
		Version:     wire.Version,
		Dispatch:    wire.Dispatch,
		Concurrency: concurrencyResult.Value.(map[string]ConcurrencyLimit),
		Rates:       wire.Rates,
		Providers:   wire.Providers,
	}
	validated := validatePolicy(policy)
	if !validated.OK {
		return core.Fail(core.E("queue.LoadPolicy", core.Concat("invalid policy in ", path), validated.Err()))
	}
	return validated
}

func defaultPolicy() Policy {
	return Policy{
		Version: policyVersion,
		Dispatch: DispatchConfig{
			DefaultAgent:      "claude",
			GlobalConcurrency: 1,
			TimeoutMinutes:    defaultDispatchTimeoutMinutes,
		},
		Concurrency: map[string]ConcurrencyLimit{
			"claude":   {Total: 1},
			"codex":    {Total: 1},
			"opencode": {Total: 1},
		},
		Rates:     map[string]RateConfig{},
		Providers: map[string]NativeConfig{},
	}
}

func validatePolicy(policy Policy) core.Result {
	if policy.Version != policyVersion {
		return core.Fail(core.Errorf("agent queue policy version must be %d", policyVersion))
	}
	if policy.Dispatch.GlobalConcurrency < 0 {
		return core.Fail(core.NewError("agent queue global concurrency cannot be negative"))
	}
	if policy.Dispatch.TimeoutMinutes < 0 {
		return core.Fail(core.NewError("agent queue timeout cannot be negative"))
	}

	policy.Dispatch.DefaultAgent = core.Trim(policy.Dispatch.DefaultAgent)
	if policy.Dispatch.DefaultAgent == "" {
		policy.Dispatch.DefaultAgent = "claude"
	}
	if policy.Dispatch.TimeoutMinutes == 0 {
		policy.Dispatch.TimeoutMinutes = defaultDispatchTimeoutMinutes
	}
	for index := range policy.Dispatch.Validation {
		policy.Dispatch.Validation[index].Command = core.Trim(policy.Dispatch.Validation[index].Command)
		if policy.Dispatch.Validation[index].Command == "" {
			return core.Fail(core.NewError("agent queue validation command is required"))
		}
		policy.Dispatch.Validation[index].Args = append([]string(nil), policy.Dispatch.Validation[index].Args...)
	}

	concurrency := make(map[string]ConcurrencyLimit, len(policy.Concurrency))
	for configuredProvider, limit := range policy.Concurrency {
		provider := core.Trim(configuredProvider)
		if provider == "" {
			return core.Fail(core.NewError("agent queue concurrency provider is required"))
		}
		if limit.Total < 0 {
			return core.Fail(core.Errorf("agent queue concurrency for %s cannot be negative", provider))
		}
		models := make(map[string]int, len(limit.Models))
		for configuredModel, modelLimit := range limit.Models {
			model := core.Trim(configuredModel)
			if model == "" {
				return core.Fail(core.Errorf("agent queue model name for %s is required", provider))
			}
			if modelLimit < 0 {
				return core.Fail(core.Errorf("agent queue model concurrency for %s/%s cannot be negative", provider, model))
			}
			models[model] = modelLimit
		}
		concurrency[provider] = ConcurrencyLimit{Total: limit.Total, Models: models}
	}
	policy.Concurrency = concurrency

	rates := make(map[string]RateConfig, len(policy.Rates))
	for configuredProvider, rate := range policy.Rates {
		provider := core.Trim(configuredProvider)
		if provider == "" {
			return core.Fail(core.NewError("agent queue rate provider is required"))
		}
		rate.ResetUTC = core.Trim(rate.ResetUTC)
		if rate.ResetUTC == "" {
			rate.ResetUTC = "00:00"
		}
		if parsed := core.TimeParse("15:04", rate.ResetUTC); !parsed.OK {
			return core.Fail(core.Errorf("agent queue reset time for %s must be HH:MM UTC", provider))
		}
		if rate.DailyLimit < 0 || rate.MinDelay < 0 || rate.SustainedDelay < 0 || rate.BurstWindow < 0 || rate.BurstDelay < 0 {
			return core.Fail(core.Errorf("agent queue rate values for %s cannot be negative", provider))
		}
		rates[provider] = rate
	}
	policy.Rates = rates

	providers := make(map[string]NativeConfig, len(policy.Providers))
	for configuredProvider, native := range policy.Providers {
		provider := core.Trim(configuredProvider)
		if provider == "" {
			return core.Fail(core.NewError("agent queue native provider is required"))
		}
		native.Executable = core.Trim(native.Executable)
		if native.Executable == "" {
			native.Executable = provider
		}
		native.DefaultModel = core.Trim(native.DefaultModel)
		credentials := make([]string, len(native.CredentialEnv))
		for index, configuredName := range native.CredentialEnv {
			name := core.Trim(configuredName)
			if !validEnvironmentName(name) {
				return core.Fail(core.Errorf("agent queue credential environment name %q is invalid", name))
			}
			credentials[index] = name
		}
		native.CredentialEnv = credentials
		native.Flags = append([]string(nil), native.Flags...)
		providers[provider] = native
	}
	policy.Providers = providers

	return core.Ok(clonePolicy(policy))
}

func clonePolicy(policy Policy) Policy {
	cloned := policy
	cloned.Dispatch.Validation = make([]Command, len(policy.Dispatch.Validation))
	for index, command := range policy.Dispatch.Validation {
		cloned.Dispatch.Validation[index] = Command{
			Command: command.Command,
			Args:    append([]string(nil), command.Args...),
		}
	}
	cloned.Concurrency = make(map[string]ConcurrencyLimit, len(policy.Concurrency))
	for provider, limit := range policy.Concurrency {
		models := make(map[string]int, len(limit.Models))
		for model, modelLimit := range limit.Models {
			models[model] = modelLimit
		}
		cloned.Concurrency[provider] = ConcurrencyLimit{Total: limit.Total, Models: models}
	}
	cloned.Rates = make(map[string]RateConfig, len(policy.Rates))
	for provider, rate := range policy.Rates {
		cloned.Rates[provider] = rate
	}
	cloned.Providers = make(map[string]NativeConfig, len(policy.Providers))
	for provider, native := range policy.Providers {
		native.CredentialEnv = append([]string(nil), native.CredentialEnv...)
		native.Flags = append([]string(nil), native.Flags...)
		cloned.Providers[provider] = native
	}
	return cloned
}

func decodeConcurrency(values map[string]any) core.Result {
	limits := make(map[string]ConcurrencyLimit, len(values))
	for provider, value := range values {
		switch typed := value.(type) {
		case map[string]any:
			limitResult := decodeConcurrencyMap(typed)
			if !limitResult.OK {
				return limitResult
			}
			limits[provider] = limitResult.Value.(ConcurrencyLimit)
		default:
			integerResult := decodeInteger(value)
			if !integerResult.OK {
				return core.Fail(core.Errorf("agent queue concurrency for %s must be an integer or map", provider))
			}
			limits[provider] = ConcurrencyLimit{Total: integerResult.Int()}
		}
	}
	return core.Ok(limits)
}

func decodeConcurrencyMap(values map[string]any) core.Result {
	limit := ConcurrencyLimit{Models: map[string]int{}}
	for name, value := range values {
		integerResult := decodeInteger(value)
		if !integerResult.OK {
			return integerResult
		}
		integer := integerResult.Int()
		if name == "total" {
			limit.Total = integer
			continue
		}
		limit.Models[name] = integer
	}
	return core.Ok(limit)
}

func decodeInteger(value any) core.Result {
	switch typed := value.(type) {
	case int:
		return core.Ok(typed)
	}
	return core.Fail(core.NewError("agent queue value must be an integer"))
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
