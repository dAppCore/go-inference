// SPDX-Licence-Identifier: EUPL-1.2

// Capability honesty: what this serving package actually delivers — tool
// declarations rendered/parsed via decode/parser's Gemma 4 syntax, and
// structured output via serving/structured's text-level validate-and-repair —
// layered onto a model/backend's own inference.CapabilityReport so
// /v1/models/capabilities (services.go) and the tool_choice gate (handler.go)
// never drift apart from what the request path actually does.
package openai

import (
	"dappco.re/go/inference"
	"dappco.re/go/inference/decode/parser"
)

// supportsToolSyntax reports whether serving has a matching native declaration
// renderer and response parser for info's architecture.
func supportsToolSyntax(info inference.ModelInfo) bool {
	return parser.SupportsToolSyntax(info.Architecture)
}

// withServingCapabilities layers the capabilities this serving package itself
// provides on top of a model/backend's own report:
//
//   - tool-calling (inference.CapabilityToolParse) — gated on architecture,
//     since decode/parser's Gemma 4 render/parse pair is what actually powers
//     it here, not a model-level ToolParser implementation;
//   - structured output (inference.CapabilityStructuredOutput) — always
//     reported, since serving/structured's validate-and-repair loop works from
//     the model's plain visible text alone and needs no architecture support.
//
// A capability the underlying report already marks supported/experimental is
// left untouched — this only upgrades an absent or unsupported entry, never
// downgrades a model's own richer claim (e.g. a future model that implements
// inference.ToolParser directly for a non-Gemma architecture keeps its own
// answer).
func withServingCapabilities(report inference.CapabilityReport, info inference.ModelInfo) inference.CapabilityReport {
	if supportsToolSyntax(info) {
		report = upsertCapability(report, inference.SupportedCapability(inference.CapabilityToolParse, inference.CapabilityGroupModel))
	}
	report = upsertCapability(report, inference.SupportedCapability(inference.CapabilityStructuredOutput, inference.CapabilityGroupModel))
	return report
}

// upsertCapability replaces report's existing entry for capability.ID with
// capability, or appends it when absent. An existing entry already Usable()
// (supported or experimental) is left as-is — this never downgrades a
// model/backend's own richer claim.
func upsertCapability(report inference.CapabilityReport, capability inference.Capability) inference.CapabilityReport {
	for i, existing := range report.Capabilities {
		if existing.ID == capability.ID {
			if existing.Usable() {
				return report
			}
			report.Capabilities[i] = capability
			return report
		}
	}
	report.Capabilities = append(report.Capabilities, capability)
	return report
}
