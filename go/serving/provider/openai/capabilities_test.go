// SPDX-Licence-Identifier: EUPL-1.2

package openai

import (
	"testing"

	"dappco.re/go/inference"
)

// TestSupportsGemmaToolSyntax_Good pins the accepted Gemma 4 architecture
// strings — the same set decode/parser.SupportsToolSyntax accepts.
func TestSupportsGemmaToolSyntax_Good(t *testing.T) {
	for _, arch := range []string{"gemma4", "gemma4_text", "GEMMA4_UNIFIED"} {
		if !supportsGemmaToolSyntax(inference.ModelInfo{Architecture: arch}) {
			t.Fatalf("supportsGemmaToolSyntax(%q) = false, want true", arch)
		}
	}
}

// TestSupportsGemmaToolSyntax_Bad pins the rejected architectures, including
// the coarser "gemma3" bucket and the empty ModelInfo zero value.
func TestSupportsGemmaToolSyntax_Bad(t *testing.T) {
	for _, arch := range []string{"qwen3", "gemma3", "mistral", ""} {
		if supportsGemmaToolSyntax(inference.ModelInfo{Architecture: arch}) {
			t.Fatalf("supportsGemmaToolSyntax(%q) = true, want false", arch)
		}
	}
}

// --- withServingCapabilities / upsertCapability -----------------------------

// TestWithServingCapabilities_Good pins the gemma4 path: both tool-calling and
// structured-output land as supported, appended to whatever the base report
// already carried.
func TestWithServingCapabilities_Good(t *testing.T) {
	base := inference.CapabilityReport{
		Capabilities: []inference.Capability{inference.SupportedCapability(inference.CapabilityChat, inference.CapabilityGroupModel)},
	}
	report := withServingCapabilities(base, inference.ModelInfo{Architecture: "gemma4_text"})
	if !report.Supports(inference.CapabilityChat) {
		t.Fatal("withServingCapabilities dropped the base report's own capability")
	}
	if !report.Supports(inference.CapabilityToolParse) {
		t.Fatal("withServingCapabilities: gemma4 architecture should report tool.parse supported")
	}
	if !report.Supports(inference.CapabilityStructuredOutput) {
		t.Fatal("withServingCapabilities: structured.output should always report supported")
	}
}

// TestWithServingCapabilities_Bad pins the non-gemma path: structured output is
// still supported (it works from plain text alone, no architecture needed),
// but tool.parse is never added — a client can trust its absence as an honest
// "tools will not work here" signal, not silence.
func TestWithServingCapabilities_Bad(t *testing.T) {
	report := withServingCapabilities(inference.CapabilityReport{}, inference.ModelInfo{Architecture: "qwen3"})
	if report.Supports(inference.CapabilityToolParse) {
		t.Fatal("withServingCapabilities: non-gemma architecture must not report tool.parse supported")
	}
	if !report.Supports(inference.CapabilityStructuredOutput) {
		t.Fatal("withServingCapabilities: structured.output should always report supported regardless of architecture")
	}
}

// TestWithServingCapabilities_Ugly pins that an existing Usable() entry (e.g. a
// model/backend that already reports tool.parse as experimental) is left
// untouched rather than downgraded or duplicated — the upsert only fills gaps.
func TestWithServingCapabilities_Ugly(t *testing.T) {
	base := inference.CapabilityReport{
		Capabilities: []inference.Capability{inference.ExperimentalCapability(inference.CapabilityToolParse, inference.CapabilityGroupModel, "native")},
	}
	report := withServingCapabilities(base, inference.ModelInfo{Architecture: "gemma4"})
	count := 0
	for _, c := range report.Capabilities {
		if c.ID == inference.CapabilityToolParse {
			count++
			if c.Status != inference.CapabilityStatusExperimental {
				t.Fatalf("tool.parse status = %s, want the pre-existing experimental status preserved", c.Status)
			}
		}
	}
	if count != 1 {
		t.Fatalf("tool.parse entries = %d, want exactly 1 (no duplicate)", count)
	}
}

// TestUpsertCapability_Good pins the append-when-absent path.
func TestUpsertCapability_Good(t *testing.T) {
	report := upsertCapability(inference.CapabilityReport{}, inference.SupportedCapability(inference.CapabilityChat, inference.CapabilityGroupModel))
	if len(report.Capabilities) != 1 || report.Capabilities[0].ID != inference.CapabilityChat {
		t.Fatalf("upsertCapability append = %+v, want one Chat capability", report.Capabilities)
	}
}

// TestUpsertCapability_Bad pins the replace-when-unsupported path: a
// previously Unsupported entry is upgraded in place, not duplicated.
func TestUpsertCapability_Bad(t *testing.T) {
	base := inference.CapabilityReport{
		Capabilities: []inference.Capability{inference.UnsupportedCapability(inference.CapabilityToolParse, inference.CapabilityGroupModel, "no native parser")},
	}
	report := upsertCapability(base, inference.SupportedCapability(inference.CapabilityToolParse, inference.CapabilityGroupModel))
	if len(report.Capabilities) != 1 {
		t.Fatalf("upsertCapability replace = %+v, want exactly 1 entry", report.Capabilities)
	}
	if report.Capabilities[0].Status != inference.CapabilityStatusSupported {
		t.Fatalf("upsertCapability replace status = %s, want supported", report.Capabilities[0].Status)
	}
}

// TestUpsertCapability_Ugly pins the no-op path: an entry already Usable() is
// left byte-for-byte, even if the caller offers a "better" one of the same ID.
func TestUpsertCapability_Ugly(t *testing.T) {
	existing := inference.ExperimentalCapability(inference.CapabilityToolParse, inference.CapabilityGroupModel, "beta")
	base := inference.CapabilityReport{Capabilities: []inference.Capability{existing}}
	report := upsertCapability(base, inference.SupportedCapability(inference.CapabilityToolParse, inference.CapabilityGroupModel))
	if len(report.Capabilities) != 1 || report.Capabilities[0].Detail != "beta" {
		t.Fatalf("upsertCapability on a Usable() entry = %+v, want the original left untouched", report.Capabilities)
	}
}
