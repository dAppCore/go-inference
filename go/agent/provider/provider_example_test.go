// SPDX-License-Identifier: EUPL-1.2

package provider

import (
	"context"

	core "dappco.re/go"
)

type providerExampleFinder struct{}

func (providerExampleFinder) Find(name string) core.Result {
	return core.Ok(core.Concat("/usr/local/bin/", name))
}

func (providerExampleFinder) Version(_ context.Context, executable string) core.Result {
	return core.Ok(core.Concat(executable, " 1.0.0"))
}

type providerExampleAdapter struct{}

func (providerExampleAdapter) Name() string {
	return "example"
}

func (providerExampleAdapter) Detect(context.Context) core.Result {
	return core.Ok(Detection{Provider: "example", Available: true})
}

func (providerExampleAdapter) Build(Launch) core.Result {
	return core.Ok(Command{Provider: "example", Executable: "example"})
}

func (providerExampleAdapter) ParseLine(stream, line string) []Output {
	return []Output{{Kind: stream, Text: line}}
}

func providerExampleRegistry(configuration map[string]Config) *Registry {
	result := DefaultRegistry(providerExampleFinder{}, configuration)
	if !result.OK {
		return nil
	}
	return result.Value.(*Registry)
}

func providerExampleLaunch() Launch {
	return Launch{
		WorkID:   "work-7",
		RunID:    "run-9",
		Title:    "Improve the TUI",
		Task:     "Render agent output as Markdown.",
		Worktree: "/tmp/lem/workspaces/work-7",
		Branch:   "feat/work-7",
		Model:    "provider/model",
	}
}

func ExampleNewRegistry() {
	result := NewRegistry(providerExampleAdapter{})
	registry := result.Value.(*Registry)
	core.Println(registry.Names()[0])
	// Output: example
}

func ExampleDefaultRegistry() {
	result := DefaultRegistry(providerExampleFinder{}, nil)
	registry := result.Value.(*Registry)
	core.Println(core.Join(", ", registry.Names()...))
	// Output: codex, claude, opencode
}

func ExampleRegistry_Adapter() {
	registry := providerExampleRegistry(nil)
	result := registry.Adapter("claude")
	core.Println(result.Value.(Adapter).Name())
	// Output: claude
}

func ExampleRegistry_Names() {
	registry := providerExampleRegistry(nil)
	core.Println(registry.Names())
	// Output: [codex claude opencode]
}

func ExampleNativeAdapter_Name() {
	registry := providerExampleRegistry(nil)
	adapter := registry.Adapter("opencode").Value.(Adapter)
	core.Println(adapter.Name())
	// Output: opencode
}

func ExampleNativeAdapter_Detect() {
	registry := providerExampleRegistry(nil)
	adapter := registry.Adapter("codex").Value.(Adapter)
	result := adapter.Detect(context.Background())
	detection := result.Value.(Detection)
	core.Println(detection.Available, detection.Executable)
	// Output: true /usr/local/bin/codex
}

func ExampleNativeAdapter_Build() {
	registry := providerExampleRegistry(nil)
	adapter := registry.Adapter("claude").Value.(Adapter)
	result := adapter.Build(providerExampleLaunch())
	command := result.Value.(Command)
	core.Println(command.Provider, command.Executable, command.Args[0])
	// Output: claude claude --print
}

func ExampleNativeAdapter_ParseLine() {
	registry := providerExampleRegistry(nil)
	adapter := registry.Adapter("codex").Value.(Adapter)
	outputs := adapter.ParseLine("stdout", `{"type":"item.completed","item":{"type":"agent_message","text":"Done"}}`)
	core.Println(outputs[0].Kind, outputs[0].Text)
	// Output: text Done
}
