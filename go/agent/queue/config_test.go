// SPDX-License-Identifier: EUPL-1.2

package queue

import (
	"testing"

	core "dappco.re/go"
	coreio "dappco.re/go/io"
)

const completePolicyFixture = `version: 1
dispatch:
  default_agent: claude
  global_concurrency: 4
  timeout_minutes: 90
  validation:
    - command: go
      args: [test, ./...]
concurrency:
  codex: 1
  opencode:
    total: 3
    opencode-go/deepseek-v4-pro: 1
rates:
  codex:
    reset_utc: "06:30"
    daily_limit: 20
    min_delay: 60
    sustained_delay: 300
    burst_window: 2
    burst_delay: 30
providers:
  codex:
    executable: /opt/codex
    default_model: gpt-5.6
    credential_env: [OPENAI_API_KEY]
    flags: [--search, allow-network]
  opencode:
    default_model: opencode-go/deepseek-v4-pro
`

func TestConfig_LoadPolicy_Good(t *testing.T) {
	medium := coreio.NewMemoryMedium()
	core.RequireNoError(t, medium.Write("agents.yaml", completePolicyFixture))

	result := LoadPolicy(medium, "agents.yaml")
	core.AssertTrue(t, result.OK, result.Error())
	policy, ok := result.Value.(Policy)
	core.AssertTrue(t, ok)
	core.AssertEqual(t, 1, policy.Version)
	core.AssertEqual(t, "claude", policy.Dispatch.DefaultAgent)
	core.AssertEqual(t, 4, policy.Dispatch.GlobalConcurrency)
	core.AssertEqual(t, 90, policy.Dispatch.TimeoutMinutes)
	core.AssertEqual(t, "go", policy.Dispatch.Validation[0].Command)
	core.AssertEqual(t, []string{"test", "./..."}, policy.Dispatch.Validation[0].Args)
	core.AssertEqual(t, 1, policy.Concurrency["codex"].Total)
	core.AssertEqual(t, 3, policy.Concurrency["opencode"].Total)
	core.AssertEqual(t, 1, policy.Concurrency["opencode"].Models["opencode-go/deepseek-v4-pro"])
	core.AssertEqual(t, "06:30", policy.Rates["codex"].ResetUTC)
	core.AssertEqual(t, 20, policy.Rates["codex"].DailyLimit)
	core.AssertEqual(t, "/opt/codex", policy.Providers["codex"].Executable)
	core.AssertEqual(t, []string{"OPENAI_API_KEY"}, policy.Providers["codex"].CredentialEnv)
	core.AssertEqual(t, []string{"--search", "allow-network"}, policy.Providers["codex"].Flags)
	core.AssertEqual(t, "opencode", policy.Providers["opencode"].Executable)
}

func TestConfig_LoadPolicy_Bad(t *testing.T) {
	medium := coreio.NewMemoryMedium()
	core.RequireNoError(t, medium.Write("agents.yaml", "version: [\n"))

	result := LoadPolicy(medium, "agents.yaml")
	core.AssertFalse(t, result.OK)
	core.AssertContains(t, result.Error(), "agents.yaml")
}

func TestConfig_LoadPolicy_Ugly(t *testing.T) {
	t.Run("missing policy returns conservative defaults without writing", func(t *testing.T) {
		medium := coreio.NewMemoryMedium()
		result := LoadPolicy(medium, "agents.yaml")
		core.AssertTrue(t, result.OK, result.Error())
		policy := result.Value.(Policy)
		core.AssertEqual(t, 1, policy.Version)
		core.AssertEqual(t, "claude", policy.Dispatch.DefaultAgent)
		core.AssertEqual(t, 1, policy.Dispatch.GlobalConcurrency)
		core.AssertEqual(t, 60, policy.Dispatch.TimeoutMinutes)
		core.AssertEqual(t, 1, policy.Concurrency["claude"].Total)
		core.AssertEqual(t, 1, policy.Concurrency["codex"].Total)
		core.AssertEqual(t, 1, policy.Concurrency["opencode"].Total)
		core.AssertFalse(t, medium.Exists("agents.yaml"))
	})

	tests := []struct {
		name    string
		fixture string
	}{
		{name: "unsupported version", fixture: "version: 2\n"},
		{name: "negative timeout", fixture: "version: 1\ndispatch:\n  timeout_minutes: -1\n"},
		{name: "empty validation command", fixture: "version: 1\ndispatch:\n  validation:\n    - command: '  '\n"},
		{name: "negative concurrency", fixture: "version: 1\nconcurrency:\n  codex: -1\n"},
		{name: "invalid scalar concurrency", fixture: "version: 1\nconcurrency:\n  codex: many\n"},
		{name: "invalid nested concurrency", fixture: "version: 1\nconcurrency:\n  codex:\n    total: many\n"},
		{name: "negative model concurrency", fixture: "version: 1\nconcurrency:\n  codex:\n    total: 2\n    gpt-5.6: -1\n"},
		{name: "invalid reset", fixture: "version: 1\nrates:\n  codex:\n    reset_utc: 25:00\n"},
		{name: "negative delay", fixture: "version: 1\nrates:\n  codex:\n    min_delay: -1\n"},
		{name: "invalid credential name", fixture: "version: 1\nproviders:\n  codex:\n    executable: codex\n    credential_env: [OPENAI-KEY]\n"},
		{name: "invalid credential first byte", fixture: "version: 1\nproviders:\n  codex:\n    executable: codex\n    credential_env: [1_OPENAI_KEY]\n"},
		{name: "empty credential name", fixture: "version: 1\nproviders:\n  codex:\n    executable: codex\n    credential_env: ['']\n"},
		{name: "wrong dispatch shape", fixture: "version: 1\ndispatch: [not, a, map]\n"},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			medium := coreio.NewMemoryMedium()
			core.RequireNoError(t, medium.Write("agents.yaml", test.fixture))
			result := LoadPolicy(medium, "agents.yaml")
			core.AssertFalse(t, result.OK)
		})
	}

	core.AssertFalse(t, LoadPolicy(nil, "agents.yaml").OK)
	core.AssertFalse(t, LoadPolicy(coreio.NewMemoryMedium(), " ").OK)
}
