// SPDX-License-Identifier: EUPL-1.2

package provider

import (
	"slices"
	"testing"

	core "dappco.re/go"
)

func TestClaudeCommandExactArgs(t *testing.T) {
	registry := providerTestRegistry(t, nil)
	adapter := registry.Adapter("claude").Value.(Adapter)
	result := adapter.Build(providerTestLaunch())
	core.AssertTrue(t, result.OK, result.Error())
	command := result.Value.(Command)
	want := []string{
		"--print",
		"--output-format", "stream-json",
		"--permission-mode", "acceptEdits",
		"--no-session-persistence",
		"--model", "openai/gpt-5.6:test",
		providerTestPrompt(),
	}
	if !slices.Equal(want, command.Args) {
		t.Fatalf("claude args = %#v, want %#v", command.Args, want)
	}
	core.AssertEqual(t, "claude", command.Executable)
	core.AssertFalse(t, slices.Contains(command.Args, "--dangerously-skip-permissions"))
	core.AssertFalse(t, slices.Contains(command.Args, "--add-dir"))
}

func TestClaudeCommandDefaultModel(t *testing.T) {
	registry := providerTestRegistry(t, map[string]Config{"claude": {DefaultModel: "claude-sonnet-4-5"}})
	adapter := registry.Adapter("claude").Value.(Adapter)
	launch := providerTestLaunch()
	launch.Model = ""
	result := adapter.Build(launch)
	core.AssertTrue(t, result.OK, result.Error())
	command := result.Value.(Command)
	modelIndex := slices.Index(command.Args, "--model")
	core.AssertTrue(t, modelIndex >= 0)
	core.AssertEqual(t, "claude-sonnet-4-5", command.Args[modelIndex+1])
}
