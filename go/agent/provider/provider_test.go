// SPDX-License-Identifier: EUPL-1.2

package provider

import (
	"context"
	"slices"
	"testing"

	core "dappco.re/go"
)

type providerTestFinder struct {
	paths          map[string]string
	versions       map[string]string
	findFailure    map[string]bool
	versionFail    map[string]bool
	invalidPath    bool
	invalidVersion bool
}

func (finder *providerTestFinder) Find(name string) core.Result {
	if finder == nil || finder.findFailure[name] {
		return core.Fail(core.Errorf("%s is unavailable", name))
	}
	if finder.invalidPath {
		return core.Ok(42)
	}
	path := finder.paths[name]
	if path == "" {
		return core.Fail(core.Errorf("%s is unavailable", name))
	}
	return core.Ok(path)
}

func (finder *providerTestFinder) Version(ctx context.Context, executable string) core.Result {
	if finder == nil || ctx == nil || finder.versionFail[executable] {
		return core.Fail(core.Errorf("%s version failed", executable))
	}
	if finder.invalidVersion {
		return core.Ok(42)
	}
	version := finder.versions[executable]
	if version == "" {
		return core.Fail(core.Errorf("%s version is unavailable", executable))
	}
	return core.Ok(version)
}

type providerTestAdapter struct {
	name string
}

func (adapter providerTestAdapter) Name() string {
	return adapter.name
}

func (adapter providerTestAdapter) Detect(context.Context) core.Result {
	return core.Ok(Detection{Provider: adapter.name, Available: true})
}

func (adapter providerTestAdapter) Build(Launch) core.Result {
	return core.Ok(Command{Provider: adapter.name})
}

func (adapter providerTestAdapter) ParseLine(stream, line string) []Output {
	return []Output{{Kind: stream, Text: line}}
}

func providerTestRegistry(t *testing.T, configs map[string]Config) *Registry {
	t.Helper()
	finder := &providerTestFinder{
		paths: map[string]string{
			"codex":    "/opt/bin/codex",
			"claude":   "/opt/bin/claude",
			"opencode": "/opt/bin/opencode",
		},
		versions: map[string]string{
			"/opt/bin/codex":    "codex-cli 0.144.3",
			"/opt/bin/claude":   "2.1.211",
			"/opt/bin/opencode": "1.16.2",
		},
	}
	result := DefaultRegistry(finder, configs)
	if !result.OK {
		t.Fatalf("DefaultRegistry failed: %s", result.Error())
	}
	registry, ok := result.Value.(*Registry)
	if !ok {
		t.Fatalf("DefaultRegistry returned %T", result.Value)
	}
	return registry
}

func providerTestLaunch() Launch {
	return Launch{
		WorkID:   "work-7",
		RunID:    "run-9",
		Title:    "Make the TUI lovely",
		Task:     "Polish tabs and markdown.",
		Worktree: "/tmp/LEM workspace",
		Branch:   "feat/lem work",
		Model:    "openai/gpt-5.6:test",
	}
}

func providerTestPrompt() string {
	return "You are working on LEM work item work-7 (run run-9).\n" +
		"Title: Make the TUI lovely\n" +
		"Task:\nPolish tabs and markdown.\n\n" +
		"Work only in this isolated Git worktree: /tmp/LEM workspace\n" +
		"Branch: feat/lem work\n" +
		"Do not modify the source checkout or any path outside this worktree.\n" +
		"Commit coherent changes to this branch before reporting completion.\n\n" +
		"End your final response with exactly one machine-readable status envelope:\n" +
		"<<<LEM_STATUS>>>{\"status\":\"completed\",\"summary\":\"short result\"}<<<END_LEM_STATUS>>>\n" +
		"or\n" +
		"<<<LEM_STATUS>>>{\"status\":\"waiting\",\"question\":\"one concrete question\"}<<<END_LEM_STATUS>>>\n" +
		"or\n" +
		"<<<LEM_STATUS>>>{\"status\":\"failed\",\"reason\":\"short reason\"}<<<END_LEM_STATUS>>>\n" +
		"Do not emit more than one status envelope."
}

func TestProvider_NewRegistry_Good(t *testing.T) {
	result := NewRegistry(providerTestAdapter{name: "codex"}, providerTestAdapter{name: "claude"})
	core.AssertTrue(t, result.OK, result.Error())
	registry := result.Value.(*Registry)
	core.AssertTrue(t, slices.Equal([]string{"codex", "claude"}, registry.Names()))
}

func TestProvider_NewRegistry_Bad(t *testing.T) {
	result := NewRegistry(providerTestAdapter{name: "codex"}, providerTestAdapter{name: "codex"})
	core.AssertFalse(t, result.OK)
	core.AssertContains(t, result.Error(), "duplicate")
}

func TestProvider_NewRegistry_Ugly(t *testing.T) {
	core.AssertFalse(t, NewRegistry().OK)
	var adapter Adapter
	core.AssertFalse(t, NewRegistry(adapter).OK)
	core.AssertFalse(t, NewRegistry(providerTestAdapter{name: " \t "}).OK)
}

func TestProvider_DefaultRegistry_Good(t *testing.T) {
	credentials := []string{"OPENAI_API_KEY"}
	flags := []string{"--search"}
	result := DefaultRegistry(&providerTestFinder{
		paths: map[string]string{
			"codex":    "/opt/bin/codex",
			"claude":   "/opt/bin/claude",
			"opencode": "/opt/bin/opencode",
		},
	}, map[string]Config{
		"codex": {CredentialEnv: credentials, Flags: flags},
	})
	core.AssertTrue(t, result.OK, result.Error())
	registry := result.Value.(*Registry)
	credentials[0] = "MUTATED"
	flags[0] = "--dangerously-bypass-approvals-and-sandbox"

	core.AssertTrue(t, slices.Equal([]string{"codex", "claude", "opencode"}, registry.Names()))
	adapterResult := registry.Adapter("codex")
	core.AssertTrue(t, adapterResult.OK, adapterResult.Error())
	command := adapterResult.Value.(Adapter).Build(providerTestLaunch())
	core.AssertTrue(t, command.OK, command.Error())
	built := command.Value.(Command)
	core.AssertEqual(t, "OPENAI_API_KEY", built.CredentialKeys[0])
	core.AssertTrue(t, slices.Contains(built.Args, "--search"))
	core.AssertFalse(t, slices.Contains(built.Args, "--dangerously-bypass-approvals-and-sandbox"))
}

func TestProvider_DefaultRegistry_Bad(t *testing.T) {
	finder := &providerTestFinder{}
	core.AssertFalse(t, DefaultRegistry(finder, map[string]Config{"unknown": {}}).OK)
	core.AssertFalse(t, DefaultRegistry(finder, map[string]Config{"codex": {Flags: []string{" "}}}).OK)
	core.AssertFalse(t, DefaultRegistry(finder, map[string]Config{"codex": {CredentialEnv: []string{"NOT-VALID"}}}).OK)
	core.AssertFalse(t, DefaultRegistry(finder, map[string]Config{"codex": {}, " codex ": {}}).OK)
	core.AssertFalse(t, DefaultRegistry(finder, map[string]Config{"codex": {CredentialEnv: []string{"9INVALID"}}}).OK)
}

func TestProvider_DefaultRegistry_Ugly(t *testing.T) {
	result := DefaultRegistry(nil, nil)
	core.AssertTrue(t, result.OK, result.Error())
	registry := result.Value.(*Registry)
	core.AssertEqual(t, 3, len(registry.Names()))
}

func TestProvider_Registry_Adapter_Good(t *testing.T) {
	registry := providerTestRegistry(t, nil)
	result := registry.Adapter(" codex ")
	core.AssertTrue(t, result.OK, result.Error())
	core.AssertEqual(t, "codex", result.Value.(Adapter).Name())
}

func TestProvider_Registry_Adapter_Bad(t *testing.T) {
	registry := providerTestRegistry(t, nil)
	result := registry.Adapter("missing")
	core.AssertFalse(t, result.OK)
	core.AssertContains(t, result.Error(), "missing")
}

func TestProvider_Registry_Adapter_Ugly(t *testing.T) {
	var registry *Registry
	core.AssertFalse(t, registry.Adapter("codex").OK)
	registry = providerTestRegistry(t, nil)
	core.AssertFalse(t, registry.Adapter(" \t ").OK)
}

func TestProvider_Registry_Names_Good(t *testing.T) {
	registry := providerTestRegistry(t, nil)
	names := registry.Names()
	names[0] = "mutated"
	core.AssertEqual(t, "codex", registry.Names()[0])
}

func TestProvider_Registry_Names_Bad(t *testing.T) {
	result := NewRegistry(providerTestAdapter{name: "codex"})
	core.AssertTrue(t, result.OK, result.Error())
	registry := result.Value.(*Registry)
	core.AssertFalse(t, slices.Contains(registry.Names(), "claude"))
}

func TestProvider_Registry_Names_Ugly(t *testing.T) {
	var registry *Registry
	names := registry.Names()
	core.AssertEqual(t, 0, len(names))
	core.AssertTrue(t, names == nil)
}

func TestProvider_NativeAdapter_Name_Good(t *testing.T) {
	registry := providerTestRegistry(t, nil)
	adapter := registry.Adapter("codex").Value.(Adapter)
	core.AssertEqual(t, "codex", adapter.Name())
}

func TestProvider_NativeAdapter_Name_Bad(t *testing.T) {
	adapter := &NativeAdapter{name: ""}
	name := adapter.Name()
	core.AssertEqual(t, "", name)
	core.AssertEqual(t, 0, len(name))
}

func TestProvider_NativeAdapter_Name_Ugly(t *testing.T) {
	var adapter *NativeAdapter
	name := adapter.Name()
	core.AssertEqual(t, "", name)
	core.AssertEqual(t, 0, len(name))
}

func TestProvider_NativeAdapter_Detect_Good(t *testing.T) {
	registry := providerTestRegistry(t, nil)
	adapter := registry.Adapter("claude").Value.(Adapter)
	result := adapter.Detect(context.Background())
	core.AssertTrue(t, result.OK, result.Error())
	detection := result.Value.(Detection)
	core.AssertTrue(t, detection.Available)
	core.AssertEqual(t, "/opt/bin/claude", detection.Executable)
	core.AssertEqual(t, "2.1.211", detection.Version)
}

func TestProvider_NativeAdapter_Detect_Bad(t *testing.T) {
	finder := &providerTestFinder{findFailure: map[string]bool{"codex": true}}
	result := DefaultRegistry(finder, nil)
	core.AssertTrue(t, result.OK, result.Error())
	adapter := result.Value.(*Registry).Adapter("codex").Value.(Adapter)
	detected := adapter.Detect(context.Background())
	core.AssertTrue(t, detected.OK, detected.Error())
	detection := detected.Value.(Detection)
	core.AssertFalse(t, detection.Available)
	core.AssertContains(t, detection.Reason, "unavailable")

	versionFinder := &providerTestFinder{
		paths:       map[string]string{"codex": "/opt/bin/codex"},
		versionFail: map[string]bool{"/opt/bin/codex": true},
	}
	versionRegistry := DefaultRegistry(versionFinder, nil).Value.(*Registry)
	versionAdapter := versionRegistry.Adapter("codex").Value.(Adapter)
	versionDetection := versionAdapter.Detect(context.Background())
	core.AssertTrue(t, versionDetection.OK, versionDetection.Error())
	core.AssertFalse(t, versionDetection.Value.(Detection).Available)
	core.AssertEqual(t, "/opt/bin/codex", versionDetection.Value.(Detection).Executable)
}

func TestProvider_NativeAdapter_Detect_Ugly(t *testing.T) {
	registry := providerTestRegistry(t, nil)
	adapter := registry.Adapter("codex").Value.(Adapter)
	core.AssertFalse(t, adapter.Detect(nil).OK)
	var empty *NativeAdapter
	core.AssertFalse(t, empty.Detect(context.Background()).OK)

	invalidPathRegistry := DefaultRegistry(&providerTestFinder{invalidPath: true}, nil).Value.(*Registry)
	invalidPathAdapter := invalidPathRegistry.Adapter("codex").Value.(Adapter)
	core.AssertFalse(t, invalidPathAdapter.Detect(context.Background()).OK)

	invalidVersionFinder := &providerTestFinder{
		paths:          map[string]string{"codex": "/opt/bin/codex"},
		invalidVersion: true,
	}
	invalidVersionRegistry := DefaultRegistry(invalidVersionFinder, nil).Value.(*Registry)
	invalidVersionAdapter := invalidVersionRegistry.Adapter("codex").Value.(Adapter)
	core.AssertFalse(t, invalidVersionAdapter.Detect(context.Background()).OK)
}

func TestProvider_NativeAdapter_Build_Good(t *testing.T) {
	registry := providerTestRegistry(t, map[string]Config{"codex": {Flags: []string{"--search"}}})
	adapter := registry.Adapter("codex").Value.(Adapter)
	launch := providerTestLaunch()
	launch.UnsafeFlags = []string{"--enable", "network"}
	result := adapter.Build(launch)
	core.AssertTrue(t, result.OK, result.Error())
	command := result.Value.(Command)
	core.AssertTrue(t, slices.Contains(command.Args, "--search"))
	core.AssertTrue(t, slices.Contains(command.Args, "network"))
	core.AssertContains(t, command.Receipt, "codex")

	launch = providerTestLaunch()
	launch.UnsafeFlags = []string{"--api-key", "super-secret", "--password=hunter2", "--token", "value"}
	redacted := adapter.Build(launch).Value.(Command)
	core.AssertFalse(t, core.Contains(redacted.Receipt, "super-secret"))
	core.AssertFalse(t, core.Contains(redacted.Receipt, "hunter2"))
	core.AssertFalse(t, core.Contains(redacted.Receipt, " value "))
	core.AssertContains(t, redacted.Receipt, "<redacted>")
}

func TestProvider_NativeAdapter_Build_Bad(t *testing.T) {
	registry := providerTestRegistry(t, nil)
	adapter := registry.Adapter("codex").Value.(Adapter)
	launch := providerTestLaunch()
	launch.Worktree = ""
	core.AssertFalse(t, adapter.Build(launch).OK)
	launch = providerTestLaunch()
	launch.UnsafeFlags = []string{""}
	core.AssertFalse(t, adapter.Build(launch).OK)
}

func TestProvider_NativeAdapter_Build_Ugly(t *testing.T) {
	var adapter *NativeAdapter
	core.AssertFalse(t, adapter.Build(Launch{}).OK)
	registry := providerTestRegistry(t, nil)
	configured := registry.Adapter("codex").Value.(Adapter)
	core.AssertFalse(t, configured.Build(Launch{}).OK)
}

func TestProvider_NativeAdapter_ParseLine_Good(t *testing.T) {
	registry := providerTestRegistry(t, nil)
	adapter := registry.Adapter("codex").Value.(Adapter)
	outputs := adapter.ParseLine("stdout", `{"type":"item.completed","item":{"type":"agent_message","text":"Done"}}`)
	core.AssertEqual(t, 1, len(outputs))
	core.AssertEqual(t, "text", outputs[0].Kind)
	core.AssertEqual(t, "Done", outputs[0].Text)
}

func TestProvider_NativeAdapter_ParseLine_Bad(t *testing.T) {
	registry := providerTestRegistry(t, nil)
	adapter := registry.Adapter("codex").Value.(Adapter)
	outputs := adapter.ParseLine("stderr", "authentication failed")
	core.AssertEqual(t, 1, len(outputs))
	core.AssertEqual(t, "stderr", outputs[0].Kind)
	core.AssertContains(t, outputs[0].Text, "failed")
}

func TestProvider_NativeAdapter_ParseLine_Ugly(t *testing.T) {
	var adapter *NativeAdapter
	core.AssertEqual(t, 0, len(adapter.ParseLine("stdout", "anything")))
	registry := providerTestRegistry(t, nil)
	configured := registry.Adapter("codex").Value.(Adapter)
	core.AssertEqual(t, 0, len(configured.ParseLine("stdout", " \t ")))
}

func TestProviderSystemFinder(t *testing.T) {
	finder := systemFinder{}
	found := finder.Find("git")
	core.AssertTrue(t, found.OK, found.Error())
	path := found.Value.(string)
	core.AssertTrue(t, path != "")

	version := finder.Version(context.Background(), path)
	core.AssertTrue(t, version.OK, version.Error())
	core.AssertContains(t, version.Value.(string), "git version")

	core.AssertFalse(t, finder.Find("lem-provider-definitely-missing").OK)
	core.AssertFalse(t, finder.Version(nil, path).OK)
}

func TestProviderReceiptEdges(t *testing.T) {
	receipt := renderReceipt("", []string{"--secret=hidden", "--credential", "hidden", "--api_key", "hidden", "--api-key", "hidden", "--token", "hidden", "--password", "hidden"})
	core.AssertTrue(t, core.HasPrefix(receipt, "''"))
	core.AssertFalse(t, core.Contains(receipt, "hidden"))
	core.AssertTrue(t, core.Count(receipt, "<redacted>") >= 6)
}
