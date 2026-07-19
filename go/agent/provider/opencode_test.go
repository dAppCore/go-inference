// SPDX-License-Identifier: EUPL-1.2

package provider

import (
	"slices"
	"testing"

	core "dappco.re/go"
)

func TestOpenCodeCommandExactArgs(t *testing.T) {
	registry := providerTestRegistry(t, nil)
	adapter := registry.Adapter("opencode").Value.(Adapter)
	result := adapter.Build(providerTestLaunch())
	core.AssertTrue(t, result.OK, result.Error())
	command := result.Value.(Command)
	want := []string{
		"run",
		"--format", "json",
		"--pure",
		"--dir", "/tmp/LEM workspace",
		"--model", "openai/gpt-5.6:test",
		providerTestPrompt(),
	}
	if !slices.Equal(want, command.Args) {
		t.Fatalf("opencode args = %#v, want %#v", command.Args, want)
	}
	core.AssertEqual(t, "opencode", command.Executable)
	core.AssertFalse(t, slices.Contains(command.Args, "--add-dir"))
}

func TestOpenCodeCommandCustomExecutable(t *testing.T) {
	registry := providerTestRegistry(t, map[string]Config{"opencode": {Executable: "/Applications/OpenCode CLI"}})
	adapter := registry.Adapter("opencode").Value.(Adapter)
	result := adapter.Build(providerTestLaunch())
	core.AssertTrue(t, result.OK, result.Error())
	command := result.Value.(Command)
	core.AssertEqual(t, "/Applications/OpenCode CLI", command.Executable)
	core.AssertContains(t, command.Receipt, "'/Applications/OpenCode CLI'")
}

func TestOpenCodeCommandDefaultModel(t *testing.T) {
	registry := providerTestRegistry(t, map[string]Config{"opencode": {DefaultModel: "anthropic/claude-sonnet"}})
	adapter := registry.Adapter("opencode").Value.(Adapter)
	launch := providerTestLaunch()
	launch.Model = ""
	result := adapter.Build(launch)
	core.AssertTrue(t, result.OK, result.Error())
	command := result.Value.(Command)
	modelIndex := slices.Index(command.Args, "--model")
	core.AssertTrue(t, modelIndex >= 0)
	core.AssertEqual(t, "anthropic/claude-sonnet", command.Args[modelIndex+1])
}
