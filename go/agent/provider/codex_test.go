// SPDX-License-Identifier: EUPL-1.2

package provider

import (
	"slices"
	"testing"

	core "dappco.re/go"
)

func TestCodexCommandExactArgs(t *testing.T) {
	registry := providerTestRegistry(t, nil)
	adapter := registry.Adapter("codex").Value.(Adapter)
	result := adapter.Build(providerTestLaunch())
	core.AssertTrue(t, result.OK, result.Error())
	command := result.Value.(Command)
	want := []string{
		"--ask-for-approval", "never",
		"--sandbox", "workspace-write",
		"--cd", "/tmp/LEM workspace",
		"exec", "--json", "--color", "never",
		"--model", "openai/gpt-5.6:test",
		providerTestPrompt(),
	}
	if !slices.Equal(want, command.Args) {
		t.Fatalf("codex args = %#v, want %#v", command.Args, want)
	}
	core.AssertEqual(t, "codex", command.Executable)
	core.AssertEqual(t, 1, len(command.Args[len(command.Args)-1:]))
	core.AssertFalse(t, slices.Contains(command.Args, "--dangerously-bypass-approvals-and-sandbox"))
	core.AssertFalse(t, slices.Contains(command.Args, "--add-dir"))
}

func TestCodexCommandContinuation(t *testing.T) {
	registry := providerTestRegistry(t, nil)
	adapter := registry.Adapter("codex").Value.(Adapter)
	launch := providerTestLaunch()
	launch.Model = ""
	launch.Continuation = "Earlier run asked: which API?\nAnswer: keep Adapter."
	result := adapter.Build(launch)
	core.AssertTrue(t, result.OK, result.Error())
	command := result.Value.(Command)
	core.AssertFalse(t, slices.Contains(command.Args, "--model"))
	core.AssertContains(t, command.Args[len(command.Args)-1], "Continuation context")
	core.AssertContains(t, command.Args[len(command.Args)-1], "Answer: keep Adapter.")
}
