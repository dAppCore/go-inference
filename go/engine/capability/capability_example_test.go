package capability

import (
	core "dappco.re/go"
	"dappco.re/go/inference"
)

func ExampleCapabilityReportForBackend() {
	backend := &capabilityTestBackend{
		name:      "local",
		available: true,
		report: inference.CapabilityReport{
			Runtime:      inference.RuntimeIdentity{Backend: "local"},
			Available:    true,
			Capabilities: []inference.Capability{inference.SupportedCapability(inference.CapabilityGenerate, inference.CapabilityGroupModel)},
		},
	}
	report := CapabilityReportForBackend("local", backend)
	core.Println(report.Available, report.Supports(inference.CapabilityGenerate))
	// Output:
	// true true
}
