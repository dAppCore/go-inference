// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import (
	core "dappco.re/go"
	"dappco.re/go/inference"
)

func Example_evaluationLossCapabilityReport() {
	defaultReport := (&rocmBackend{}).Capabilities()
	linkedReport := rocmCapabilityReport(nativeDeviceInfo{}, inference.ModelIdentity{}, inference.AdapterIdentity{}, true, hipKernelStatus{
		CrossEntropy: hipKernelStatusLinked,
	})

	for _, report := range []inference.CapabilityReport{defaultReport, linkedReport} {
		capability, _ := report.Capability(inference.CapabilityEvaluation)
		core.Println(
			capability.ID,
			capability.Status,
			capability.Labels["loss_kernel"],
			capability.Labels["loss_kernel_name"],
			capability.Labels["loss_scope"],
		)
	}
	// Output:
	// evaluation experimental not_linked rocm_cross_entropy_loss toy_cross_entropy
	// evaluation experimental linked rocm_cross_entropy_loss toy_cross_entropy
}

func Example_trainingCapabilityReport() {
	report := (&rocmBackend{}).Capabilities()
	for _, id := range []inference.CapabilityID{
		inference.CapabilityLoRATraining,
		inference.CapabilityDistillation,
		inference.CapabilityGRPO,
	} {
		capability, _ := report.Capability(id)
		core.Println(
			capability.ID,
			capability.Status,
			capability.Labels["runtime_status"],
			capability.Labels["training_kernel"],
			capability.Labels["training_interface"],
			capability.Labels["required_kernel"],
			capability.Labels["optimizer_status"],
			capability.Labels["optimizer_helper"],
		)
	}
	// Output:
	// lora.training planned planned not_linked not_implemented lora_backward update_only RunNativeAdamWUpdatePass
	// distillation planned planned not_linked not_implemented distillation_forward_loss update_only RunNativeAdamWUpdatePass
	// grpo planned planned not_linked not_implemented grpo_rollout_policy update_only RunNativeAdamWUpdatePass
}

func Example_metadataOnlyFixtureCapabilities() {
	report := (&rocmBackend{}).Capabilities()
	for _, id := range []inference.CapabilityID{
		inference.CapabilityMoERouting,
		inference.CapabilityMoELazyExperts,
		inference.CapabilityJANGTQ,
		inference.CapabilityCodebookVQ,
	} {
		capability, _ := report.Capability(id)
		core.Println(
			capability.ID,
			capability.Labels["runtime_status"],
			firstNonEmptyString(capability.Labels["fixture_kernel"], capability.Labels["kernel_status"]),
			capability.Labels["fixture_kernel_name"],
			capability.Labels["production_integration"],
			capability.Labels["required_integration"],
		)
	}
	// Output:
	// moe.routing experimental linked rocm_moe_router pending model_router_forward
	// moe.lazy_experts experimental linked rocm_moe_lazy_experts pending expert_paging
	// jangtq experimental linked rocm_jangtq_projection pending packed_weight_model_integration
	// codebook.vq experimental linked rocm_codebook_lookup pending codebook_weight_model_integration
}
