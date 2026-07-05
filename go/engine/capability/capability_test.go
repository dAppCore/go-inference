// SPDX-Licence-Identifier: EUPL-1.2

package capability

import (
	"context"

	"dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/inference/serving"
)

type capabilityTestBackend struct {
	name      string
	available bool
	report    inference.CapabilityReport
}

func (backend *capabilityTestBackend) Capabilities() inference.CapabilityReport {
	return backend.report
}

func (backend *capabilityTestBackend) Generate(_ context.Context, prompt string, _ serving.GenOpts) core.Result {
	return core.Ok(serving.Result{Text: prompt, Content: prompt})
}

func (backend *capabilityTestBackend) Chat(_ context.Context, messages []serving.Message, _ serving.GenOpts) core.Result {
	if len(messages) == 0 {
		return core.Ok(serving.Result{})
	}
	last := messages[len(messages)-1].Content
	return core.Ok(serving.Result{Text: last, Content: last})
}

func (backend *capabilityTestBackend) Name() string { return backend.name }

func (backend *capabilityTestBackend) Available() bool { return backend.available }

func TestCapability_CapabilityReportForBackend_Good(t *core.T) {
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

	report := CapabilityReportForBackend("mlx", backend)

	core.AssertEqual(t, "metal", report.Runtime.Backend)
	core.AssertTrue(t, report.Supports(inference.CapabilityGenerate))
	core.AssertTrue(t, report.Supports(inference.CapabilityProbeEvents))
}

func TestCapability_CapabilityReportForBackend_Bad(t *core.T) {
	report := CapabilityReportForBackend("missing", nil)

	core.AssertEqual(t, "missing", report.Runtime.Backend)
	core.AssertFalse(t, report.Available)
	core.AssertLen(t, report.Capabilities, 0)
}

func TestCapability_CapabilityReportForBackend_Ugly(t *core.T) {
	backend := &capabilityTestBackend{name: "http", available: true}

	report := CapabilityReportForBackend("", backend)

	core.AssertEqual(t, "http", report.Runtime.Backend)
	core.AssertTrue(t, report.Available)
	core.AssertTrue(t, report.Supports(inference.CapabilityGenerate))
	core.AssertTrue(t, report.Supports(inference.CapabilityChat))
}
