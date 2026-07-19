// SPDX-License-Identifier: EUPL-1.2

package tui

import (
	"strings"
	"testing"

	core "dappco.re/go"
)

func TestRuntimeAdapter_Good(t *testing.T) {
	source := fixtureRuntimeSource{result: core.Ok([]runtimeProbe{
		fixtureRuntime{name: "apple", version: "1.2.0", path: "/usr/local/bin/container", network: true, volumes: true, hardware: true, subSecond: true},
		fixtureRuntime{name: "vz", version: "native", path: "Virtualization.framework", gpu: true, network: true, volumes: true, encryption: true, hardware: true},
		fixtureRuntime{name: "docker", version: "28.1", path: "/usr/local/bin/docker", gpu: true, network: true, volumes: true},
		fixtureRuntime{name: "podman", version: "5.5", path: "/opt/homebrew/bin/podman", network: true, volumes: true},
	})}
	result := newRuntimeAdapter(source).Detect()
	if !result.OK {
		t.Fatalf("Detect: %v", result.Value)
	}
	capabilities, ok := result.Value.([]runtimeCapability)
	if !ok || len(capabilities) != 4 {
		t.Fatalf("Detect = %#v (%T)", result.Value, result.Value)
	}
	if capabilities[0].Name != "apple" || capabilities[1].Name != "vz" || capabilities[2].Name != "docker" || capabilities[3].Name != "podman" {
		t.Fatalf("priority order = %#v", capabilities)
	}
	apple := capabilities[0]
	if apple.GPU || !apple.NetworkIsolation || !apple.VolumeMounts || apple.Encryption || !apple.HardwareIsolation || !apple.SubSecondStart {
		t.Fatalf("Apple projection = %#v", apple)
	}
	vz := capabilities[1]
	if !vz.GPU || !vz.Encryption || !vz.HardwareIsolation || vz.SubSecondStart {
		t.Fatalf("VZ projection = %#v", vz)
	}
}

func TestRuntimeAdapter_Bad(t *testing.T) {
	reason := "runtime probe permission denied"
	result := newRuntimeAdapter(fixtureRuntimeSource{result: core.Fail(core.E("test.runtime", reason, nil))}).Detect()
	if result.OK {
		t.Fatal("Detect unexpectedly succeeded")
	}
	a := newApp("", 0, 64)
	a.activePanel = panelWork
	a.inspector.ApplyRuntime(result)
	view := a.inspector.View(a, 72, 24)
	if !strings.Contains(view, "RUNTIME") || !strings.Contains(view, "unavailable") || !strings.Contains(view, reason) {
		t.Fatalf("disabled runtime inspector:\n%s", view)
	}
}

func TestRuntimeAdapter_Ugly(t *testing.T) {
	result := newRuntimeAdapter(fixtureRuntimeSource{result: core.Ok([]runtimeProbe{})}).Detect()
	if !result.OK {
		t.Fatalf("Detect empty: %v", result.Value)
	}
	a := newApp("", 0, 64)
	a.activePanel = panelWork
	a.inspector.ApplyRuntime(result)
	view := a.inspector.View(a, 48, 20)
	if !strings.Contains(view, "RUNTIME") || !strings.Contains(view, "none available") {
		t.Fatalf("empty runtime inspector:\n%s", view)
	}
}

type fixtureRuntimeSource struct{ result core.Result }

func (source fixtureRuntimeSource) All() core.Result { return source.result }

type fixtureRuntime struct {
	name       string
	version    string
	path       string
	gpu        bool
	network    bool
	volumes    bool
	encryption bool
	hardware   bool
	subSecond  bool
}

func (runtime fixtureRuntime) RuntimeName() string       { return runtime.name }
func (runtime fixtureRuntime) RuntimeVersion() string    { return runtime.version }
func (runtime fixtureRuntime) RuntimePath() string       { return runtime.path }
func (runtime fixtureRuntime) HasGPU() bool              { return runtime.gpu }
func (runtime fixtureRuntime) HasNetworkIsolation() bool { return runtime.network }
func (runtime fixtureRuntime) HasVolumeMounts() bool     { return runtime.volumes }
func (runtime fixtureRuntime) HasEncryption() bool       { return runtime.encryption }
func (runtime fixtureRuntime) IsHardwareIsolated() bool  { return runtime.hardware }
func (runtime fixtureRuntime) HasSubSecondStart() bool   { return runtime.subSecond }
