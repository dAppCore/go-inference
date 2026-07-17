// SPDX-License-Identifier: EUPL-1.2

package tui

import (
	container "dappco.re/go/container"

	core "dappco.re/go"
)

type runtimeCapability struct {
	Name              string
	Version           string
	Path              string
	GPU               bool
	NetworkIsolation  bool
	VolumeMounts      bool
	Encryption        bool
	HardwareIsolation bool
	SubSecondStart    bool
}

type runtimeDetector interface {
	Detect() core.Result
}

type runtimeProbe interface {
	RuntimeName() string
	RuntimeVersion() string
	RuntimePath() string
	HasGPU() bool
	HasNetworkIsolation() bool
	HasVolumeMounts() bool
	HasEncryption() bool
	IsHardwareIsolated() bool
	HasSubSecondStart() bool
}

type runtimeSource interface {
	All() core.Result
}

type runtimeAdapter struct{ source runtimeSource }

func newRuntimeAdapter(source runtimeSource) runtimeDetector {
	return &runtimeAdapter{source: source}
}

func newContainerRuntimeDetector() runtimeDetector {
	return newRuntimeAdapter(containerRuntimeSource{})
}

func (adapter *runtimeAdapter) Detect() core.Result {
	if adapter == nil || adapter.source == nil {
		return core.Fail(core.E("tui.runtimeAdapter.Detect", "runtime source is unavailable", nil))
	}
	result := adapter.source.All()
	if !result.OK {
		return result
	}
	probes, ok := result.Value.([]runtimeProbe)
	if !ok {
		return core.Fail(core.E("tui.runtimeAdapter.Detect", "invalid runtime probe result", nil))
	}
	capabilities := make([]runtimeCapability, 0, len(probes))
	for _, probe := range probes {
		if probe == nil {
			continue
		}
		capabilities = append(capabilities, runtimeCapability{
			Name:              probe.RuntimeName(),
			Version:           probe.RuntimeVersion(),
			Path:              probe.RuntimePath(),
			GPU:               probe.HasGPU(),
			NetworkIsolation:  probe.HasNetworkIsolation(),
			VolumeMounts:      probe.HasVolumeMounts(),
			Encryption:        probe.HasEncryption(),
			HardwareIsolation: probe.IsHardwareIsolated(),
			SubSecondStart:    probe.HasSubSecondStart(),
		})
	}
	return core.Ok(capabilities)
}

type containerRuntimeSource struct{}

func (containerRuntimeSource) All() core.Result {
	detected := container.DetectAll()
	probes := make([]runtimeProbe, 0, len(detected))
	for _, runtime := range detected {
		probes = append(probes, containerRuntimeProbe{runtime: runtime})
	}
	return core.Ok(probes)
}

type containerRuntimeProbe struct{ runtime container.ContainerRuntime }

func (probe containerRuntimeProbe) RuntimeName() string {
	return string(probe.runtime.Type)
}

func (probe containerRuntimeProbe) RuntimeVersion() string {
	return probe.runtime.Version
}

func (probe containerRuntimeProbe) RuntimePath() string {
	return probe.runtime.Path
}

func (probe containerRuntimeProbe) HasGPU() bool {
	return probe.runtime.HasGPU()
}

func (probe containerRuntimeProbe) HasNetworkIsolation() bool {
	return probe.runtime.HasNetworkIsolation()
}

func (probe containerRuntimeProbe) HasVolumeMounts() bool {
	return probe.runtime.HasVolumeMounts()
}

func (probe containerRuntimeProbe) HasEncryption() bool {
	return probe.runtime.HasEncryption()
}

func (probe containerRuntimeProbe) IsHardwareIsolated() bool {
	return probe.runtime.IsHardwareIsolated()
}

func (probe containerRuntimeProbe) HasSubSecondStart() bool {
	return probe.runtime.HasSubSecondStart()
}

type runtimeInspection struct {
	ready        bool
	reason       string
	capabilities []runtimeCapability
}

func runtimeInspectionFrom(result core.Result) runtimeInspection {
	inspection := runtimeInspection{ready: true, capabilities: []runtimeCapability{}}
	if !result.OK {
		inspection.reason = result.Error()
		return inspection
	}
	capabilities, ok := result.Value.([]runtimeCapability)
	if !ok {
		inspection.reason = "invalid runtime capability result"
		return inspection
	}
	inspection.capabilities = append(inspection.capabilities, capabilities...)
	return inspection
}
