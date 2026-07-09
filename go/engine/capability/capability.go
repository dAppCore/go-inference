// SPDX-Licence-Identifier: EUPL-1.2

package capability

import (
	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/inference/serving"
)

// CapabilityReportForBackend returns the shared inference capability report for
// a serving backend without requiring callers to import a concrete runtime.
func CapabilityReportForBackend(name string, backend serving.Backend) inference.CapabilityReport {
	if backend == nil {
		return inference.CapabilityReport{Runtime: inference.RuntimeIdentity{Backend: name}}
	}
	if reporter, ok := backend.(inference.CapabilityReporter); ok {
		report := reporter.Capabilities()
		if report.Runtime.Backend == "" && !report.Available && len(report.Capabilities) == 0 {
			return fallbackCapabilityReport(name, backend)
		}
		if report.Runtime.Backend == "" {
			report.Runtime.Backend = core.Coalesce(name, backend.Name())
		}
		return report
	}
	return fallbackCapabilityReport(name, backend)
}

func fallbackCapabilityReport(name string, backend serving.Backend) inference.CapabilityReport {
	backendName := core.Coalesce(name, backend.Name())
	return inference.CapabilityReport{
		Runtime:   inference.RuntimeIdentity{Backend: backendName},
		Available: backend.Available(),
		Capabilities: []inference.Capability{
			inference.SupportedCapability(inference.CapabilityGenerate, inference.CapabilityGroupModel),
			inference.SupportedCapability(inference.CapabilityChat, inference.CapabilityGroupModel),
		},
	}
}
