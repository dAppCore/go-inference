// SPDX-Licence-Identifier: EUPL-1.2

package hip

import (
	"time"

	inferdecode "dappco.re/go/inference/decode"
)

const (
	attachedDrafterNativeHandoffPendingTargetDecode   = "pending_target_retained_decode"
	attachedDrafterNativeHandoffTargetDecodeOnly      = "target_retained_decode_only"
	attachedDrafterNativeHandoffRetainedStateVerifier = "retained_state_attached_drafter"

	attachedDrafterAssistantVerifierPreflightNotReady     = "not_ready"
	attachedDrafterAssistantVerifierPreflightMetadataOnly = "metadata_only"
	attachedDrafterAssistantVerifierPreflightTensorReady  = "tensor_ready"

	attachedDrafterAssistantVerifierLayoutOfficial = "official"
	attachedDrafterAssistantVerifierLayoutInferred = "inferred"
	attachedDrafterAssistantVerifierLayoutInvalid  = "invalid"

	attachedDrafterAssistantVerifierTensorsEmpty    = "empty"
	attachedDrafterAssistantVerifierTensorsMissing  = "missing"
	attachedDrafterAssistantVerifierTensorsComplete = "complete"

	attachedDrafterAssistantVerifierPlanNotReady    = "not_ready"
	attachedDrafterAssistantVerifierPlanTensorBound = "tensor_bound"
	attachedDrafterAssistantVerifierPlanUnsupported = "unsupported"
)

// AttachedDrafterMetrics exposes ROCm-native MTP counters without expanding the
// shared go-inference GenerateMetrics contract.
type AttachedDrafterMetrics struct {
	DraftTokens    int
	AcceptedTokens int
	RejectedTokens int
	EmittedTokens  int
	ProposedTokens int
	VerifyCalls    int
	TargetCalls    int
	DraftCalls     int
	AcceptanceRate float64
	Duration       time.Duration
	TargetDuration time.Duration
	DraftDuration  time.Duration
}

func attachedDrafterMetricsFromDecode(metrics inferdecode.Metrics) *AttachedDrafterMetrics {
	if metrics.DraftTokens == 0 &&
		metrics.AcceptedTokens == 0 &&
		metrics.RejectedTokens == 0 &&
		metrics.DraftCalls == 0 {
		return nil
	}
	proposed := metrics.AcceptedTokens + metrics.RejectedTokens
	if proposed == 0 {
		proposed = metrics.DraftTokens
	}
	acceptance := metrics.AcceptanceRate
	if acceptance == 0 && proposed > 0 {
		acceptance = float64(metrics.AcceptedTokens) / float64(proposed)
	}
	return &AttachedDrafterMetrics{
		DraftTokens:    metrics.DraftTokens,
		AcceptedTokens: metrics.AcceptedTokens,
		RejectedTokens: metrics.RejectedTokens,
		EmittedTokens:  metrics.EmittedTokens,
		ProposedTokens: proposed,
		VerifyCalls:    metrics.DraftCalls,
		TargetCalls:    metrics.TargetCalls,
		DraftCalls:     metrics.DraftCalls,
		AcceptanceRate: acceptance,
		Duration:       metrics.Duration,
		TargetDuration: metrics.TargetDuration,
		DraftDuration:  metrics.DraftDuration,
	}
}
