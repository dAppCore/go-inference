// SPDX-Licence-Identifier: EUPL-1.2

package hip

import (
	"path/filepath"
	"runtime"
	"strings"
)

const (
	RuntimeDispatchStatusActive                    = "active"
	RuntimeDispatchStatusDevROCm                   = "dev_rocm"
	RuntimeDispatchStatusCompileReadyPending       = "compile_ready_runtime_dispatch_pending"
	RuntimeLaneAMD                                 = "amd"
	RuntimeLaneCUDA                                = "cuda"
	RuntimeLaneCPUX86                              = "cpu-x86"
	RuntimeLaneCPUAArch64                          = "cpu-aarch64"
	RuntimeLaneArtifactAMD                         = "lthn-amd"
	RuntimeLaneArtifactCUDA                        = "lthn-cuda"
	RuntimeLaneArtifactCPUX86                      = "lthn-cpu-x86"
	RuntimeLaneArtifactCPUAArch64                  = "lthn-cpu-aarch64"
	RuntimeLaneSidecarAMD                          = "rocm_kernels_gfx1100.hsaco"
	RuntimeLaneSidecarCUDA                         = "rocm_kernels_nvidia_sm_75.o"
	RuntimeLaneSidecarCPUX86                       = "rocm_kernels_hip_cpu_x86_64.o"
	RuntimeLaneSidecarCPUAArch64                   = "rocm_kernels_hip_cpu_aarch64.o"
	RuntimeLaneDispatchNextWorkCUDA                = "register_cuda_runtime_dispatch"
	RuntimeLaneDispatchNextWorkCPU                 = "register_hip_cpu_runtime_dispatch"
	RuntimeLaneDispatchNextWorkStatefulGenerate    = "stateful_generate_smoke"
	RuntimeLaneDispatchNextWorkOpenAIServer        = "openai_server_smoke"
	RuntimeLaneDispatchNextWorkThroughputBenchmark = "qat_mtp_throughput_benchmark"
)

// RuntimeLaneStatus describes the release artifact lane a binary represents.
// It is intentionally backend-neutral so API consumers can reason about AMD,
// CUDA, and CPU artifacts before every runtime has native dispatch wired.
type RuntimeLaneStatus struct {
	Name                  string            `json:"name"`
	Artifact              string            `json:"artifact,omitempty"`
	Kind                  string            `json:"kind"`
	Backend               string            `json:"backend"`
	RuntimeLane           string            `json:"runtime_lane"`
	RuntimeDispatchStatus string            `json:"runtime_dispatch_status"`
	OS                    string            `json:"os,omitempty"`
	Arch                  string            `json:"arch,omitempty"`
	Native                bool              `json:"native"`
	StubRuntime           bool              `json:"stub_runtime,omitempty"`
	ProductionArtifact    bool              `json:"production_artifact"`
	HIPPlatform           string            `json:"hip_platform,omitempty"`
	KernelOutput          string            `json:"kernel_output,omitempty"`
	Sidecars              []string          `json:"sidecars,omitempty"`
	NextWork              []string          `json:"next_work,omitempty"`
	Labels                map[string]string `json:"labels,omitempty"`
}

func (lane RuntimeLaneStatus) Active() bool {
	return strings.TrimSpace(lane.RuntimeDispatchStatus) == RuntimeDispatchStatusActive
}

func (lane RuntimeLaneStatus) Pending() bool {
	status := strings.TrimSpace(lane.RuntimeDispatchStatus)
	return status != "" && status != RuntimeDispatchStatusActive && status != RuntimeDispatchStatusDevROCm
}

func (lane RuntimeLaneStatus) Clone() RuntimeLaneStatus {
	lane.Sidecars = append([]string(nil), lane.Sidecars...)
	lane.NextWork = append([]string(nil), lane.NextWork...)
	lane.Labels = cloneStringMap(lane.Labels)
	return lane
}

// DefaultRuntimeLanes returns the release lanes shipped by the ROCm CLI family.
func DefaultRuntimeLanes() []RuntimeLaneStatus {
	lanes := []RuntimeLaneStatus{
		{
			Name:                  RuntimeLaneArtifactAMD,
			Artifact:              RuntimeLaneArtifactAMD,
			Kind:                  "release-binary",
			Backend:               "rocm",
			RuntimeLane:           RuntimeLaneAMD,
			RuntimeDispatchStatus: RuntimeDispatchStatusActive,
			OS:                    "linux",
			Arch:                  "amd64",
			Native:                true,
			ProductionArtifact:    true,
			HIPPlatform:           "amd",
			KernelOutput:          "hsaco",
			Sidecars:              []string{RuntimeLaneSidecarAMD},
			NextWork: []string{
				RuntimeLaneDispatchNextWorkStatefulGenerate,
				RuntimeLaneDispatchNextWorkOpenAIServer,
				RuntimeLaneDispatchNextWorkThroughputBenchmark,
			},
			Labels: map[string]string{
				"active_backend":          "rocm",
				"default_backend":         "rocm",
				"hip_platform":            "amd",
				"kernel_output":           "hsaco",
				"production_artifact":     "true",
				"runtime_dispatch_status": RuntimeDispatchStatusActive,
				"runtime_lane":            RuntimeLaneAMD,
				"static_hip":              "true",
			},
		},
		{
			Name:                  RuntimeLaneArtifactCUDA,
			Artifact:              RuntimeLaneArtifactCUDA,
			Kind:                  "release-binary",
			Backend:               "cuda",
			RuntimeLane:           RuntimeLaneCUDA,
			RuntimeDispatchStatus: RuntimeDispatchStatusCompileReadyPending,
			OS:                    "linux",
			Arch:                  "amd64",
			Native:                true,
			ProductionArtifact:    true,
			HIPPlatform:           "nvidia",
			KernelOutput:          "object",
			Sidecars:              []string{RuntimeLaneSidecarCUDA},
			NextWork: []string{
				RuntimeLaneDispatchNextWorkCUDA,
				RuntimeLaneDispatchNextWorkStatefulGenerate,
				RuntimeLaneDispatchNextWorkOpenAIServer,
				RuntimeLaneDispatchNextWorkThroughputBenchmark,
			},
			Labels: map[string]string{
				"active_backend":          "rocm",
				"default_backend":         "rocm",
				"hip_platform":            "nvidia",
				"kernel_output":           "object",
				"production_artifact":     "true",
				"runtime_dispatch_status": RuntimeDispatchStatusCompileReadyPending,
				"runtime_lane":            RuntimeLaneCUDA,
				"static_hip":              "true",
			},
		},
		{
			Name:                  RuntimeLaneArtifactCPUX86,
			Artifact:              RuntimeLaneArtifactCPUX86,
			Kind:                  "release-binary",
			Backend:               "cpu",
			RuntimeLane:           RuntimeLaneCPUX86,
			RuntimeDispatchStatus: RuntimeDispatchStatusCompileReadyPending,
			OS:                    "linux",
			Arch:                  "amd64",
			Native:                true,
			StubRuntime:           true,
			ProductionArtifact:    true,
			HIPPlatform:           "cpu",
			KernelOutput:          "object",
			Sidecars:              []string{RuntimeLaneSidecarCPUX86},
			NextWork: []string{
				RuntimeLaneDispatchNextWorkCPU,
				RuntimeLaneDispatchNextWorkStatefulGenerate,
				RuntimeLaneDispatchNextWorkOpenAIServer,
				RuntimeLaneDispatchNextWorkThroughputBenchmark,
			},
			Labels: map[string]string{
				"active_backend":          "rocm",
				"default_backend":         "rocm",
				"hip_platform":            "cpu",
				"kernel_output":           "object",
				"production_artifact":     "true",
				"runtime_dispatch_status": RuntimeDispatchStatusCompileReadyPending,
				"runtime_lane":            RuntimeLaneCPUX86,
				"static_go":               "true",
			},
		},
		{
			Name:                  RuntimeLaneArtifactCPUAArch64,
			Artifact:              RuntimeLaneArtifactCPUAArch64,
			Kind:                  "release-binary",
			Backend:               "cpu",
			RuntimeLane:           RuntimeLaneCPUAArch64,
			RuntimeDispatchStatus: RuntimeDispatchStatusCompileReadyPending,
			OS:                    "linux",
			Arch:                  "arm64",
			Native:                true,
			StubRuntime:           true,
			ProductionArtifact:    true,
			HIPPlatform:           "cpu",
			KernelOutput:          "object",
			Sidecars:              []string{RuntimeLaneSidecarCPUAArch64},
			NextWork: []string{
				RuntimeLaneDispatchNextWorkCPU,
				RuntimeLaneDispatchNextWorkStatefulGenerate,
				RuntimeLaneDispatchNextWorkOpenAIServer,
				RuntimeLaneDispatchNextWorkThroughputBenchmark,
			},
			Labels: map[string]string{
				"active_backend":          "rocm",
				"default_backend":         "rocm",
				"hip_platform":            "cpu",
				"kernel_output":           "object",
				"production_artifact":     "true",
				"runtime_dispatch_status": RuntimeDispatchStatusCompileReadyPending,
				"runtime_lane":            RuntimeLaneCPUAArch64,
				"static_go":               "true",
			},
		},
	}
	out := make([]RuntimeLaneStatus, 0, len(lanes))
	for _, lane := range lanes {
		out = append(out, lane.Clone())
	}
	return out
}

func RuntimeLaneForArtifact(name string) (RuntimeLaneStatus, bool) {
	key := normalizeRuntimeLaneToken(name)
	if key == "" {
		return RuntimeLaneStatus{}, false
	}
	for _, lane := range DefaultRuntimeLanes() {
		if normalizeRuntimeLaneToken(lane.Name) == key || normalizeRuntimeLaneToken(lane.Artifact) == key {
			return lane.Clone(), true
		}
	}
	return RuntimeLaneStatus{}, false
}

func RuntimeLanesForBackend(backend string) []RuntimeLaneStatus {
	key := normalizeRuntimeLaneToken(backend)
	out := []RuntimeLaneStatus{}
	for _, lane := range DefaultRuntimeLanes() {
		if normalizeRuntimeLaneToken(lane.Backend) == key {
			out = append(out, lane.Clone())
		}
	}
	return out
}

func CurrentProcessRuntimeLane(name string) RuntimeLaneStatus {
	if lane, ok := RuntimeLaneForArtifact(name); ok {
		return AnnotateRuntimeLaneForCurrentProcess(lane)
	}
	return AnnotateRuntimeLaneForCurrentProcess(RuntimeLaneStatus{
		Name:                  runtime.GOOS + "/" + runtime.GOARCH,
		Artifact:              name,
		Kind:                  "go-dev-binary",
		Backend:               "rocm",
		RuntimeLane:           RuntimeLaneAMD,
		RuntimeDispatchStatus: RuntimeDispatchStatusDevROCm,
		OS:                    runtime.GOOS,
		Arch:                  runtime.GOARCH,
		Native:                runtime.GOOS == "linux" && runtime.GOARCH == "amd64",
		StubRuntime:           !(runtime.GOOS == "linux" && runtime.GOARCH == "amd64"),
		Sidecars:              []string{RuntimeLaneSidecarAMD},
		Labels: map[string]string{
			"active_backend":          "rocm",
			"default_backend":         "rocm",
			"module":                  "dappco.re/go/rocm",
			"production_artifact":     "false",
			"runtime_dispatch_status": RuntimeDispatchStatusDevROCm,
			"runtime_lane":            RuntimeLaneAMD,
		},
	})
}

func AnnotateRuntimeLaneForCurrentProcess(lane RuntimeLaneStatus) RuntimeLaneStatus {
	lane = lane.Clone()
	if lane.Labels == nil {
		lane.Labels = map[string]string{}
	}
	lane.Labels["current_goos"] = runtime.GOOS
	lane.Labels["current_goarch"] = runtime.GOARCH
	lane.Labels["module"] = "dappco.re/go/rocm"
	lane.Labels["process_matches_artifact"] = boolRuntimeLaneLabel(lane.OS == runtime.GOOS && lane.Arch == runtime.GOARCH)
	return lane
}

func normalizeRuntimeLaneToken(value string) string {
	value = strings.TrimSpace(value)
	if value == "" {
		return ""
	}
	return strings.ToLower(filepath.Base(value))
}

func boolRuntimeLaneLabel(value bool) string {
	if value {
		return "true"
	}
	return "false"
}
