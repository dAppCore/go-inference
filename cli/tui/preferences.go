// SPDX-License-Identifier: EUPL-1.2

package tui

import (
	core "dappco.re/go"
	config "dappco.re/go/config"
	coreio "dappco.re/go/io"
)

const (
	preferenceContextLength     = "generation.context_length"
	preferenceMaxTokens         = "generation.max_tokens"
	preferenceThinking          = "generation.thinking"
	preferenceTheme             = "appearance.theme"
	preferenceShowThinking      = "appearance.show_thinking"
	preferenceRecentSessions    = "sessions.recent_limit"
	preferenceKnowledgePaths    = "knowledge.paths"
	preferenceKnowledgeMaxBytes = "knowledge.max_bytes"
	preferencePreferredRuntime  = "runtime.preferred"
	preferenceConfirmExecution  = "execution.confirm"
	preferenceEnvironmentPrefix = "LEM"
)

type preferenceValues struct {
	ContextLength      int
	MaxTokens          int
	Thinking           string
	Theme              string
	ShowThinking       bool
	RecentSessionLimit int
	KnowledgePaths     []string
	KnowledgeMaxBytes  int64
	PreferredRuntime   string
	ConfirmExecution   bool
}

type preferenceStore interface {
	Values() preferenceValues
	Set(key string, value any) core.Result
	Commit() core.Result
	Reload() core.Result
	Warning() error
}

type configPreferenceStore struct {
	mu             core.Mutex
	configuration  *config.Config
	medium         coreio.Medium
	path           string
	warning        error
	commitDisabled bool
}

func openPreferences(medium coreio.Medium, path string) core.Result {
	if medium == nil {
		return core.Fail(core.E("tui.openPreferences", "configuration medium is required", nil))
	}
	path = core.Trim(path)
	if path == "" {
		return core.Fail(core.E("tui.openPreferences", "configuration path is required", nil))
	}

	opened := newPreferenceConfig(medium, path)
	if opened.OK {
		configuration, ok := opened.Value.(*config.Config)
		if !ok {
			return core.Fail(core.E("tui.openPreferences", "invalid config result", nil))
		}
		return core.Ok(&configPreferenceStore{
			configuration: configuration,
			medium:        medium,
			path:          path,
		})
	}

	warning := preferenceResultError(opened, "load configuration")
	fallback := newPreferenceConfig(coreio.NewMemoryMedium(), path)
	if !fallback.OK {
		return core.Fail(core.E(
			"tui.openPreferences",
			"create default configuration fallback",
			preferenceResultError(fallback, "create fallback"),
		))
	}
	configuration, ok := fallback.Value.(*config.Config)
	if !ok {
		return core.Fail(core.E("tui.openPreferences", "invalid fallback config result", nil))
	}
	return core.Ok(&configPreferenceStore{
		configuration:  configuration,
		medium:         medium,
		path:           path,
		warning:        warning,
		commitDisabled: true,
	})
}

func newPreferenceConfig(medium coreio.Medium, path string) core.Result {
	return config.New(
		config.WithMedium(medium),
		config.WithPath(path),
		config.WithEnvPrefix(preferenceEnvironmentPrefix),
		config.WithDefaults(preferenceDefaults()),
	)
}

func (preferences *configPreferenceStore) Values() preferenceValues {
	values := defaultPreferenceValues()
	if preferences == nil {
		return values
	}
	preferences.mu.Lock()
	configuration := preferences.configuration
	preferences.mu.Unlock()
	if configuration == nil {
		return values
	}

	preferenceGet(configuration, preferenceContextLength, &values.ContextLength)
	preferenceGet(configuration, preferenceMaxTokens, &values.MaxTokens)
	preferenceGet(configuration, preferenceThinking, &values.Thinking)
	preferenceGet(configuration, preferenceTheme, &values.Theme)
	preferenceGet(configuration, preferenceShowThinking, &values.ShowThinking)
	preferenceGet(configuration, preferenceRecentSessions, &values.RecentSessionLimit)
	preferenceGet(configuration, preferenceKnowledgePaths, &values.KnowledgePaths)
	preferenceGet(configuration, preferenceKnowledgeMaxBytes, &values.KnowledgeMaxBytes)
	preferenceGet(configuration, preferencePreferredRuntime, &values.PreferredRuntime)
	preferenceGet(configuration, preferenceConfirmExecution, &values.ConfirmExecution)
	return values
}

func (preferences *configPreferenceStore) Set(key string, value any) core.Result {
	if preferences == nil {
		return core.Fail(core.E("tui.preferences.Set", "preferences are unavailable", nil))
	}
	key = core.Trim(key)
	if key == "" {
		return core.Fail(core.E("tui.preferences.Set", "preference key is required", nil))
	}
	preferences.mu.Lock()
	configuration := preferences.configuration
	preferences.mu.Unlock()
	if configuration == nil {
		return core.Fail(core.E("tui.preferences.Set", "configuration is unavailable", nil))
	}
	return configuration.Set(key, value)
}

func (preferences *configPreferenceStore) Commit() core.Result {
	if preferences == nil {
		return core.Fail(core.E("tui.preferences.Commit", "preferences are unavailable", nil))
	}
	preferences.mu.Lock()
	disabled := preferences.commitDisabled
	warning := preferences.warning
	configuration := preferences.configuration
	preferences.mu.Unlock()
	if disabled {
		return core.Fail(core.E(
			"tui.preferences.Commit",
			"commit disabled until the configuration reloads successfully",
			warning,
		))
	}
	if configuration == nil {
		return core.Fail(core.E("tui.preferences.Commit", "configuration is unavailable", nil))
	}
	return configuration.Commit()
}

func (preferences *configPreferenceStore) Reload() core.Result {
	if preferences == nil {
		return core.Fail(core.E("tui.preferences.Reload", "preferences are unavailable", nil))
	}
	preferences.mu.Lock()
	medium := preferences.medium
	path := preferences.path
	preferences.mu.Unlock()

	opened := newPreferenceConfig(medium, path)
	if !opened.OK {
		warning := preferenceResultError(opened, "reload configuration")
		preferences.mu.Lock()
		preferences.warning = warning
		preferences.commitDisabled = true
		preferences.mu.Unlock()
		return core.Fail(core.E("tui.preferences.Reload", "reload configuration", warning))
	}
	configuration, ok := opened.Value.(*config.Config)
	if !ok {
		return core.Fail(core.E("tui.preferences.Reload", "invalid config result", nil))
	}

	preferences.mu.Lock()
	preferences.configuration = configuration
	preferences.warning = nil
	preferences.commitDisabled = false
	preferences.mu.Unlock()
	return core.Ok(nil)
}

func (preferences *configPreferenceStore) Warning() error {
	if preferences == nil {
		return nil
	}
	preferences.mu.Lock()
	defer preferences.mu.Unlock()
	return preferences.warning
}

func defaultPreferenceValues() preferenceValues {
	return preferenceValues{
		ContextLength:      0,
		MaxTokens:          4096,
		Thinking:           "model",
		Theme:              "midnight",
		ShowThinking:       true,
		RecentSessionLimit: 12,
		KnowledgePaths:     []string{appPacksPath},
		KnowledgeMaxBytes:  65536,
		PreferredRuntime:   "auto",
		ConfirmExecution:   true,
	}
}

func preferenceDefaults() map[string]any {
	values := defaultPreferenceValues()
	return map[string]any{
		preferenceContextLength:     values.ContextLength,
		preferenceMaxTokens:         values.MaxTokens,
		preferenceThinking:          values.Thinking,
		preferenceTheme:             values.Theme,
		preferenceShowThinking:      values.ShowThinking,
		preferenceRecentSessions:    values.RecentSessionLimit,
		preferenceKnowledgePaths:    values.KnowledgePaths,
		preferenceKnowledgeMaxBytes: values.KnowledgeMaxBytes,
		preferencePreferredRuntime:  values.PreferredRuntime,
		preferenceConfirmExecution:  values.ConfirmExecution,
	}
}

func preferenceGet(configuration *config.Config, key string, target any) {
	if result := configuration.Get(key, target); !result.OK {
		core.Warn("tui.preferences.get_default", "key", key, "error", result.Value)
	}
}

func preferenceResultError(result core.Result, message string) error {
	if err := resultError(result); err != nil {
		return err
	}
	return core.E("tui.preferences", message, nil)
}
