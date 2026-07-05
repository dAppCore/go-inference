// SPDX-Licence-Identifier: EUPL-1.2

package capability

import (
	"testing"

	"dappco.re/go/inference"
)

// CapabilityReportForBackend is the bridge every serving backend goes through
// to surface its capability report. Two shapes matter: the reporter path
// (backend implements CapabilityReporter) and the fallback path (it does not).

func BenchmarkCapabilityReportForBackend_Reporter(b *testing.B) {
	backend := &capabilityTestBackend{
		name:      "mlx",
		available: true,
		report: inference.CapabilityReport{
			Runtime:   inference.RuntimeIdentity{Backend: "metal", NativeRuntime: true},
			Available: true,
			Capabilities: []inference.Capability{
				inference.SupportedCapability(inference.CapabilityGenerate, inference.CapabilityGroupModel),
				inference.SupportedCapability(inference.CapabilityProbeEvents, inference.CapabilityGroupProbe),
			},
		},
	}
	b.ReportAllocs()
	for b.Loop() {
		reportSink = CapabilityReportForBackend("mlx", backend)
	}
}

func BenchmarkCapabilityReportForBackend_Fallback(b *testing.B) {
	backend := &capabilityTestBackend{name: "http", available: true}
	b.ReportAllocs()
	for b.Loop() {
		reportSink = CapabilityReportForBackend("http", backend)
	}
}

var reportSink inference.CapabilityReport
