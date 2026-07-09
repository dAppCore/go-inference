// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import (
	"os"
	"path/filepath"
	"sort"
	"strings"
)

const hipKernelModuleEnv = "GO_ROCM_KERNEL_HSACO"

var hipKernelModuleExecutable = os.Executable

type hipKernelModuleResolution struct {
	Path   string
	Source string
}

func resolveHIPKernelModule() hipKernelModuleResolution {
	if explicit := strings.TrimSpace(os.Getenv(hipKernelModuleEnv)); explicit != "" {
		return hipKernelModuleResolution{Path: explicit, Source: "env"}
	}
	for _, candidate := range hipKernelModuleCandidates() {
		info, err := os.Stat(candidate)
		if err != nil || info.IsDir() {
			continue
		}
		return hipKernelModuleResolution{Path: candidate, Source: "sidecar"}
	}
	return hipKernelModuleResolution{}
}

func hipKernelModuleCandidates() []string {
	exe, err := hipKernelModuleExecutable()
	if err != nil {
		return nil
	}
	exe = strings.TrimSpace(exe)
	if exe == "" {
		return nil
	}
	dir := filepath.Dir(exe)
	patterns := []string{
		filepath.Join(dir, "rocm_kernels_*.hsaco"),
		filepath.Join(dir, "kernels", "rocm_kernels_*.hsaco"),
		filepath.Join(dir, "..", "kernels", "rocm_kernels_*.hsaco"),
	}
	seen := map[string]struct{}{}
	candidates := make([]string, 0, len(patterns))
	for _, pattern := range patterns {
		matches, err := filepath.Glob(pattern)
		if err != nil {
			continue
		}
		sort.Strings(matches)
		for _, match := range matches {
			clean := filepath.Clean(match)
			if _, ok := seen[clean]; ok {
				continue
			}
			seen[clean] = struct{}{}
			candidates = append(candidates, clean)
		}
	}
	return candidates
}

func hipKernelModulePath() string {
	return resolveHIPKernelModule().Path
}

func hipKernelModuleSourceLabel(source string) string {
	switch source {
	case "env":
		return hipKernelModuleEnv
	case "sidecar":
		return "packaged HSACO sidecar"
	default:
		return "packaged HSACO sidecar or " + hipKernelModuleEnv
	}
}
