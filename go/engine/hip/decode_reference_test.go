// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import (
	"context"
	"iter"
	"strings"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference"
	inferdecode "dappco.re/go/inference/decode"
)

func TestDecodeReferencePromptLookup_Good_ReturnsDraftAfterRepeatedSuffix(t *testing.T) {
	draft, err := rocmReferencePromptLookupDraft([]int32{1, 2, 3, 4, 1, 2}, 2, 2)

	core.RequireNoError(t, err)
	core.AssertEqual(t, []int32{3, 4}, draft)
}

func TestDecodeReferencePromptLookup_Good_NoMatchReturnsNil(t *testing.T) {
	draft, err := rocmReferencePromptLookupDraft([]int32{1, 2, 3, 4}, 2, 2)

	core.RequireNoError(t, err)
	core.AssertEqual(t, 0, len(draft))
}

func TestDecodeReferencePromptLookup_Good_TruncatesDraft(t *testing.T) {
	draft, err := rocmReferencePromptLookupDraft([]int32{1, 2, 3, 4, 1, 2}, 2, 1)

	core.RequireNoError(t, err)
	core.AssertEqual(t, []int32{3}, draft)
}

func TestDecodeReferencePromptLookup_Good_UsesLongestRepeatedSuffix(t *testing.T) {
	draft, err := rocmReferencePromptLookupDraft([]int32{1, 2, 3, 4, 1, 2, 3}, 2, 4)

	core.RequireNoError(t, err)
	core.AssertEqual(t, []int32{4}, draft)
}

func TestDecodeReferencePromptLookup_Good_TooShortReturnsNil(t *testing.T) {
	draft, err := rocmReferencePromptLookupDraft([]int32{1, 2, 1}, 2, 2)

	core.RequireNoError(t, err)
	core.AssertEqual(t, 0, len(draft))
}

func TestDecodeReferencePromptLookup_Bad_RejectsInvalidConfig(t *testing.T) {
	_, err := rocmReferencePromptLookupDraft([]int32{1, 2}, 0, 1)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "min match")

	_, err = rocmReferencePromptLookupDraft([]int32{1, 2}, 1, 0)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "max draft")
}

func TestDecodeReferenceSpeculativeAccept_Good_AcceptsMatchingPrefix(t *testing.T) {
	accepted, rejectedAt := rocmReferenceSpeculativeAccept([]int32{4, 5, 6}, []int32{4, 5, 7})

	core.AssertEqual(t, []int32{4, 5}, accepted)
	core.AssertEqual(t, 2, rejectedAt)
}

func TestDecodeReferenceSpeculativeAccept_Good_AcceptsAllDraftTokens(t *testing.T) {
	accepted, rejectedAt := rocmReferenceSpeculativeAccept([]int32{4, 5}, []int32{4, 5, 6})

	core.AssertEqual(t, []int32{4, 5}, accepted)
	core.AssertEqual(t, -1, rejectedAt)
}

func TestDecodeReferenceSpeculativeAccept_Good_RejectsFirstMismatch(t *testing.T) {
	accepted, rejectedAt := rocmReferenceSpeculativeAccept([]int32{4, 5}, []int32{9, 5})

	core.AssertEqual(t, 0, len(accepted))
	core.AssertEqual(t, 0, rejectedAt)
}

func TestDecodeReferenceSpeculativeAccept_Good_DraftLongerThanTargetRejectsAtTargetEnd(t *testing.T) {
	accepted, rejectedAt := rocmReferenceSpeculativeAccept([]int32{4, 5, 6}, []int32{4, 5})

	core.AssertEqual(t, []int32{4, 5}, accepted)
	core.AssertEqual(t, 2, rejectedAt)
}

func TestDecodeReferenceSpeculativeAccept_Good_EmptyDraftAcceptsAll(t *testing.T) {
	accepted, rejectedAt := rocmReferenceSpeculativeAccept(nil, []int32{4, 5})

	core.AssertEqual(t, 0, len(accepted))
	core.AssertEqual(t, -1, rejectedAt)
}

func TestDecodeHelpers_Good_SpeculativeDecodeUsesSharedHarness(t *testing.T) {
	target := &rocmModel{native: &fakeNativeModel{tokens: []inference.Token{{ID: 4, Text: "a"}, {ID: 5, Text: "b"}, {ID: 7, Text: "c"}}}}
	draft := &rocmModel{native: &fakeNativeModel{tokens: []inference.Token{{ID: 4, Text: "a"}, {ID: 5, Text: "b"}, {ID: 6, Text: "x"}}}}

	result, err := SpeculativeDecode(context.Background(), target, draft, SpeculativeDecodeConfig{Prompt: "p", MaxTokens: 3, DraftTokens: 3})

	core.RequireNoError(t, err)
	core.AssertEqual(t, inferdecode.ModeSpeculative, result.Mode)
	core.AssertEqual(t, 2, result.Metrics.AcceptedTokens)
	core.AssertEqual(t, 1, result.Metrics.RejectedTokens)
	core.AssertEqual(t, 3, result.Metrics.EmittedTokens)
}

func TestDecodeHelpers_Good_AttachedDrafterDecodeUsesSharedHarness(t *testing.T) {
	targetNative := &fakeNativeModel{tokens: []inference.Token{{ID: 4, Text: "a"}, {ID: 5, Text: "b"}, {ID: 7, Text: "c"}}}
	draftNative := &fakeNativeModel{tokens: []inference.Token{{ID: 4, Text: "a"}, {ID: 9, Text: "x"}, {ID: 8, Text: "y"}}}
	target := newDecodeGemma4E2BQ6Target(targetNative)
	draft := newDecodeGemma4E2BBF16Assistant(draftNative)

	result, err := AttachedDrafterDecode(context.Background(), target, draft, AttachedDrafterDecodeConfig{Prompt: "p", MaxTokens: 3, DraftTokens: 3})

	core.RequireNoError(t, err)
	core.AssertEqual(t, inferdecode.ModeSpeculative, result.Mode)
	core.AssertEqual(t, 1, result.Metrics.AcceptedTokens)
	core.AssertEqual(t, 2, result.Metrics.RejectedTokens)
	core.AssertEqual(t, 3, result.Metrics.EmittedTokens)
	core.AssertEqual(t, 1, result.Metrics.TargetCalls)
	core.AssertEqual(t, 1, result.Metrics.DraftCalls)
	core.AssertEqual(t, []string{"p"}, targetNative.generatePrompts)
	core.AssertEqual(t, []string{"p"}, draftNative.generatePrompts)
}

func TestDecodeHelpers_Good_SpeculativeDecodeUsesGemma4RemainingWindow(t *testing.T) {
	targetNative := &fakeNativeModel{
		tokens:       []inference.Token{{ID: 4, Text: "a"}, {ID: 5, Text: "b"}},
		encodeResult: []int32{1, 2, 3},
	}
	draftNative := &fakeNativeModel{
		tokens:       []inference.Token{{ID: 4, Text: "a"}, {ID: 9, Text: "x"}},
		encodeResult: []int32{1, 2, 3},
	}
	target := &rocmModel{
		modelInfo: inference.ModelInfo{Architecture: "gemma4_text"},
		native:    targetNative,
	}
	draft := &rocmModel{
		modelInfo: inference.ModelInfo{Architecture: "gemma4_assistant"},
		native:    draftNative,
	}

	result, err := SpeculativeDecode(context.Background(), target, draft, SpeculativeDecodeConfig{Prompt: "ignored"})

	core.RequireNoError(t, err)
	core.AssertEqual(t, inferdecode.ModeSpeculative, result.Mode)
	core.AssertEqual(t, defaultContextLengthCap-3, targetNative.generateConfigs[0].MaxTokens)
	core.AssertEqual(t, defaultContextLengthCap-3, draftNative.generateConfigs[0].MaxTokens)

	result, err = SpeculativeDecode(context.Background(), target, draft, SpeculativeDecodeConfig{Prompt: "ignored", MaxTokens: -1})

	core.RequireNoError(t, err)
	core.AssertEqual(t, inferdecode.ModeSpeculative, result.Mode)
	core.AssertEqual(t, defaultContextLengthCap-3, targetNative.generateConfigs[1].MaxTokens)
	core.AssertEqual(t, defaultContextLengthCap-3, draftNative.generateConfigs[1].MaxTokens)
}

func TestDecodeHelpers_Good_PlanAttachedDrafterReportsNativeGap(t *testing.T) {
	targetNative := &fakeNativeModel{}
	draftNative := &fakeNativeModel{}
	target := newDecodeGemma4E2BQ6Target(targetNative)
	draft := newDecodeGemma4E2BBF16Assistant(draftNative)

	plan, err := PlanAttachedDrafter(target, draft)

	core.RequireNoError(t, err)
	core.AssertEqual(t, "mtp_attached_drafter", plan.Mode)
	core.AssertEqual(t, "gemma4_text", plan.Target.Architecture)
	core.AssertEqual(t, "gemma4_assistant", plan.Draft.Architecture)
	core.AssertEqual(t, ProductionMTPDefaultDraftTokens, plan.DraftTokens)
	core.AssertEqual(t, hipKernelStatusLinked, plan.HelperStatus)
	core.AssertEqual(t, hipKernelStatusNotLinked, plan.NativeAttachment)
	core.AssertEqual(t, hipKernelStatusLinked, plan.Labels["attached_drafter_helper"])
	core.AssertEqual(t, hipKernelStatusNotLinked, plan.Labels["attached_drafter_native_attachment"])
	core.AssertEqual(t, hipKernelStatusLinked, plan.Labels["attached_drafter_retained_state_entrypoint"])
	core.AssertEqual(t, "true", plan.Labels["attached_drafter_retained_state_required"])
	core.AssertEqual(t, "rocm_state_session_runtime_kv", plan.Labels["attached_drafter_state_source"])
	core.AssertEqual(t, "forbidden", plan.Labels["attached_drafter_prompt_replay_fallback"])
	core.AssertEqual(t, officialGemma4E2BAssistantArchitecture, plan.Labels["attached_drafter_assistant_architecture"])
	core.AssertEqual(t, "true", plan.Labels["attached_drafter_assistant_ordered_embeddings"])
	core.AssertEqual(t, productionMTPAssistantOrderedEmbeddingCentroidsLabel, plan.Labels["attached_drafter_assistant_centroids"])
	core.AssertEqual(t, productionMTPAssistantCentroidIntermediateTopKLabel, plan.Labels["attached_drafter_assistant_centroid_intermediate_top_k"])
	core.AssertEqual(t, "true", plan.Labels["attached_drafter_assistant_four_layer_drafter"])
	core.AssertEqual(t, "int64", plan.Labels["attached_drafter_assistant_token_ordering_dtype"])
	core.AssertEqual(t, productionMTPAssistantTokenOrderingShapeLabel, plan.Labels["attached_drafter_assistant_token_ordering_shape"])
	core.AssertEqual(t, officialGemma4E2BAssistantModelID, plan.Labels["attached_drafter_official_assistant_model_id"])
	core.AssertEqual(t, officialGemma4E2BAssistantRevision, plan.Labels["attached_drafter_official_assistant_revision"])
	core.AssertEqual(t, officialGemma4E2BTargetModelID, plan.Labels["attached_drafter_official_target_model_id"])
	core.AssertEqual(t, officialGemma4E2BTargetRevision, plan.Labels["attached_drafter_official_target_revision"])
	core.AssertEqual(t, "gemma4", plan.Labels["attached_drafter_target_engine_profile"])
	core.AssertEqual(t, "gemma4_text", plan.Labels["attached_drafter_target_engine_architecture_profile"])
	core.AssertEqual(t, string(inference.FeatureRuntimeNative), plan.Labels["attached_drafter_target_engine_architecture_runtime_status"])
	core.AssertEqual(t, "gemma", plan.Labels["attached_drafter_target_engine_architecture_reasoning_parser"])
	core.AssertEqual(t, "q8,paged,k-q8-v-q4,retained-state", plan.Labels["attached_drafter_target_engine_architecture_cache_hints"])
	core.AssertEqual(t, "gemma4_hf_turn", plan.Labels["attached_drafter_target_engine_chat_template"])
	core.AssertEqual(t, "q_proj,v_proj,o_proj", plan.Labels["attached_drafter_target_gemma4_lora_default_targets"])
	core.AssertEqual(t, "model_registry", plan.Labels["attached_drafter_target_gemma4_weight_policy"])
	core.AssertEqual(t, "gemma4", plan.Labels["attached_drafter_assistant_engine_profile"])
	core.AssertEqual(t, "gemma4_assistant", plan.Labels["attached_drafter_assistant_engine_architecture_profile"])
	core.AssertEqual(t, string(inference.FeatureRuntimeNative), plan.Labels["attached_drafter_assistant_engine_architecture_runtime_status"])
	core.AssertEqual(t, "retained-state,attached-drafter", plan.Labels["attached_drafter_assistant_engine_architecture_cache_hints"])
	core.AssertEqual(t, "true", plan.Labels["attached_drafter_assistant_engine_architecture_attached_only"])
	core.AssertEqual(t, "false", plan.Labels["attached_drafter_assistant_engine_architecture_generation"])
	core.AssertEqual(t, "", plan.Labels["attached_drafter_assistant_gemma4_lora_default_targets"])
	core.AssertEqual(t, "", plan.Labels["attached_drafter_assistant_gemma4_weight_policy"])
	core.AssertEqual(t, "true", plan.Labels["attached_drafter_official_pair_verified"])
	core.AssertEqual(t, productionMTPDefaultDraftTokensLabel, plan.Labels["attached_drafter_speculative_draft_tokens"])
	core.AssertEqual(t, "true", plan.Labels["production_default_candidate"])
	var evidence ProductionMTPPromotionEvidence
	core.RequireNoError(t, ApplyProductionMTPAttachedDrafterLabelEvidence(&evidence, plan.Labels))
	core.AssertEqual(t, true, evidence.AttachedDrafterRetainedStateEntrypoint)
	core.AssertEqual(t, true, evidence.AssistantOrderedEmbeddings)
	core.AssertEqual(t, true, evidence.AssistantFourLayerDrafter)
	core.AssertEqual(t, true, evidence.OfficialPairVerified)
	core.AssertEqual(t, ProductionMTPDefaultDraftTokens, evidence.SpeculativeDraftTokens)
	core.AssertEqual(t, 0, len(targetNative.generatePrompts))
	core.AssertEqual(t, 0, len(draftNative.generatePrompts))
}

func TestDecodeHelpers_Bad_AttachNativeDrafterRejectsNonNativeWithoutGenerate(t *testing.T) {
	target := newDecodeGemma4E2BQ6Target(nil)
	draft := newDecodeGemma4E2BBF16Assistant(nil)

	_, err := AttachNativeDrafter(target, draft)

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "native ROCm target and draft models")
}

func TestDecodeHelpers_Bad_AttachNativeDrafterReportsHIPNotLinkedWithoutGenerate(t *testing.T) {
	targetNative := &hipLoadedModel{modelInfo: gemma4DecodeE2BQ6Info()}
	draftNative := &hipLoadedModel{modelInfo: gemma4DecodeE2BBF16AssistantInfo()}
	target := newDecodeGemma4E2BQ6Target(targetNative)
	draft := newDecodeGemma4E2BBF16Assistant(draftNative)

	attachment, err := AttachNativeDrafter(target, draft)

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "native HIP drafter attachment is not linked yet")
	core.AssertContains(t, err.Error(), "target retained decode not_linked")
	core.AssertContains(t, err.Error(), "assistant verify not_linked")
	core.AssertEqual(t, hipKernelStatusNotLinked, attachment.NativeAttachment)
	core.AssertEqual(t, "gemma4_text", attachment.Target.Architecture)
	core.AssertEqual(t, "gemma4_assistant", attachment.Draft.Architecture)
	core.AssertEqual(t, "hip", attachment.Labels["attached_drafter_runtime"])
	core.AssertEqual(t, hipKernelStatusLinked, attachment.Labels["attached_drafter_retained_state_entrypoint"])
	core.AssertEqual(t, "true", attachment.Labels["attached_drafter_retained_state_required"])
	core.AssertEqual(t, "rocm_state_session_runtime_kv", attachment.Labels["attached_drafter_state_source"])
	core.AssertEqual(t, "forbidden", attachment.Labels["attached_drafter_prompt_replay_fallback"])
	core.AssertEqual(t, hipKernelStatusNotLinked, attachment.Labels["attached_drafter_target_retained_decode"])
	core.AssertEqual(t, hipKernelStatusNotLinked, attachment.Labels["attached_drafter_target_retained_state_decode"])
	core.AssertEqual(t, hipKernelStatusNotLinked, attachment.Labels["attached_drafter_assistant_verify"])
	core.AssertEqual(t, hipKernelStatusNotLinked, attachment.Labels["attached_drafter_assistant_state_verify"])
	core.AssertEqual(t, attachedDrafterNativeHandoffPendingTargetDecode, attachment.Labels["attached_drafter_native_handoff"])
	core.AssertEqual(t, attachedDrafterAssistantVerifierPreflightMetadataOnly, attachment.Labels["attached_drafter_assistant_verifier_preflight"])
	core.AssertEqual(t, attachedDrafterAssistantVerifierLayoutOfficial, attachment.Labels["attached_drafter_assistant_verifier_layout"])
	core.AssertEqual(t, attachedDrafterAssistantVerifierTensorsEmpty, attachment.Labels["attached_drafter_assistant_verifier_tensors"])
}

func TestDecodeHelpers_Good_NewAttachedDrafterPairRecordsHIPNotReady(t *testing.T) {
	target := &rocmModel{
		modelPath: "/models/lmstudio-community-gemma-4-e2b-it-6bit",
		modelInfo: inference.ModelInfo{Architecture: "gemma4_text", NumLayers: productionLaneGemma4E2BLayers, HiddenSize: productionLaneGemma4E2BHiddenSize, VocabSize: 262144},
		native:    &hipLoadedModel{modelInfo: inference.ModelInfo{Architecture: "gemma4_text", NumLayers: productionLaneGemma4E2BLayers, HiddenSize: productionLaneGemma4E2BHiddenSize, VocabSize: 262144}},
	}
	draft := &rocmModel{
		modelPath: rocmGemma4MTPAssistantPath("E2B", "bf16"),
		modelInfo: inference.ModelInfo{Architecture: "gemma4_assistant", NumLayers: 4, HiddenSize: productionLaneGemma4E2BHiddenSize, VocabSize: 262144, QuantBits: 16},
		native:    &hipLoadedModel{modelInfo: inference.ModelInfo{Architecture: "gemma4_assistant", NumLayers: 4, HiddenSize: productionLaneGemma4E2BHiddenSize, VocabSize: 262144, QuantBits: 16}},
	}

	pair, err := NewAttachedDrafterPair(target, draft)

	core.RequireNoError(t, err)
	core.AssertEqual(t, false, pair.NativeReady())
	core.AssertEqual(t, hipKernelStatusNotLinked, pair.Attachment.NativeAttachment)
	core.AssertEqual(t, "gemma4_text", pair.Plan.Target.Architecture)
	core.AssertEqual(t, "gemma4_assistant", pair.Plan.Draft.Architecture)
	core.AssertEqual(t, hipKernelStatusLinked, pair.Attachment.Labels["attached_drafter_retained_state_entrypoint"])
	core.AssertEqual(t, "true", pair.Attachment.Labels["attached_drafter_retained_state_required"])
	core.AssertEqual(t, "rocm_state_session_runtime_kv", pair.Attachment.Labels["attached_drafter_state_source"])
	core.AssertEqual(t, "forbidden", pair.Attachment.Labels["attached_drafter_prompt_replay_fallback"])
	core.AssertEqual(t, hipKernelStatusNotLinked, pair.Attachment.Labels["attached_drafter_target_retained_decode"])
	core.AssertEqual(t, hipKernelStatusNotLinked, pair.Attachment.Labels["attached_drafter_target_retained_state_decode"])
	core.AssertEqual(t, hipKernelStatusNotLinked, pair.Attachment.Labels["attached_drafter_assistant_verify"])
	core.AssertEqual(t, hipKernelStatusNotLinked, pair.Attachment.Labels["attached_drafter_assistant_state_verify"])
	core.AssertEqual(t, attachedDrafterNativeHandoffPendingTargetDecode, pair.Attachment.Labels["attached_drafter_native_handoff"])
	core.AssertEqual(t, attachedDrafterAssistantVerifierPreflightMetadataOnly, pair.Attachment.Labels["attached_drafter_assistant_verifier_preflight"])
	core.AssertEqual(t, attachedDrafterAssistantVerifierLayoutOfficial, pair.Attachment.Labels["attached_drafter_assistant_verifier_layout"])
	core.AssertEqual(t, attachedDrafterAssistantVerifierTensorsEmpty, pair.Attachment.Labels["attached_drafter_assistant_verifier_tensors"])
	core.AssertEqual(t, "E2B", pair.Attachment.Labels["attached_drafter_target_gemma4_size"])
	core.AssertEqual(t, "q6", pair.Attachment.Labels["attached_drafter_target_gemma4_quant_mode"])
	core.AssertEqual(t, "64", pair.Attachment.Labels["attached_drafter_target_gemma4_quant_group"])
	core.AssertEqual(t, "true", pair.Attachment.Labels["attached_drafter_target_gemma4_pack_supported"])
	core.AssertEqual(t, "true", pair.Attachment.Labels["attached_drafter_target_gemma4_runnable_on_card"])
	core.AssertEqual(t, "E2B", pair.Attachment.Labels["attached_drafter_assistant_gemma4_size"])
	core.AssertEqual(t, "bf16", pair.Attachment.Labels["attached_drafter_assistant_gemma4_quant_mode"])
	core.AssertEqual(t, "true", pair.Attachment.Labels["attached_drafter_assistant_gemma4_pack_supported"])
	core.AssertEqual(t, "true", pair.Attachment.Labels["attached_drafter_assistant_gemma4_runnable_on_card"])
	core.AssertContains(t, pair.NativeError, "native HIP drafter attachment is not linked yet")
}

func TestDecodeHelpers_Good_AttachNativeDrafterReportsAssistantVerifierTensorReady(t *testing.T) {
	targetNative := &hipLoadedModel{modelInfo: gemma4DecodeE2BQ6Info()}
	draftNative := &hipLoadedModel{
		modelInfo: gemma4DecodeE2BBF16AssistantInfo(),
		tensors:   gemma4DecodeE2BAssistantVerifierTensors(),
	}
	target := newDecodeGemma4E2BQ6Target(targetNative)
	draft := newDecodeGemma4E2BBF16Assistant(draftNative)

	attachment, err := AttachNativeDrafter(target, draft)

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "native HIP drafter attachment is not linked yet")
	core.AssertContains(t, err.Error(), "assistant preflight tensor_ready")
	core.AssertContains(t, err.Error(), "assistant plan tensor_bound")
	core.AssertEqual(t, attachedDrafterAssistantVerifierPreflightTensorReady, attachment.Labels["attached_drafter_assistant_verifier_preflight"])
	core.AssertEqual(t, attachedDrafterAssistantVerifierLayoutOfficial, attachment.Labels["attached_drafter_assistant_verifier_layout"])
	core.AssertEqual(t, attachedDrafterAssistantVerifierTensorsComplete, attachment.Labels["attached_drafter_assistant_verifier_tensors"])
	core.AssertEqual(t, "", attachment.Labels["attached_drafter_assistant_verifier_missing"])
	core.AssertEqual(t, attachedDrafterAssistantVerifierPlanTensorBound, attachment.Labels["attached_drafter_assistant_verifier_plan"])
	core.AssertEqual(t, "not_linked", attachment.Labels["attached_drafter_assistant_verifier_kernel"])
	core.AssertEqual(t, "bf16", attachment.Labels["attached_drafter_assistant_verifier_projection_encoding"])
	core.AssertEqual(t, "4", attachment.Labels["attached_drafter_assistant_verifier_layers"])
	core.AssertContains(t, attachment.Labels["attached_drafter_assistant_verifier_kernel_families"], hipKernelNameEmbedLookup)
	core.AssertContains(t, attachment.Labels["attached_drafter_assistant_verifier_kernel_families"], hipKernelNameProjection)
	core.AssertContains(t, attachment.Labels["attached_drafter_assistant_verifier_kernel_families"], hipKernelNameRMSNorm)
	core.AssertContains(t, attachment.Labels["attached_drafter_assistant_verifier_kernel_families"], hipKernelNameAttentionHeads)
	core.AssertContains(t, attachment.Labels["attached_drafter_assistant_verifier_kernel_families"], hipKernelNamePackedTopK)
	core.AssertEqual(t, hipKernelStatusNotLinked, attachment.Labels["attached_drafter_assistant_verify"])
	core.AssertEqual(t, hipKernelStatusNotLinked, attachment.Labels["attached_drafter_assistant_state_verify"])
}

func TestDecodeHelpers_Bad_AttachNativeDrafterReportsAssistantVerifierBadShape(t *testing.T) {
	targetNative := &hipLoadedModel{modelInfo: gemma4DecodeE2BQ6Info()}
	draftInfo := gemma4DecodeE2BBF16AssistantInfo()
	draftInfo.NumLayers = 2
	draftNative := &hipLoadedModel{modelInfo: draftInfo}
	target := newDecodeGemma4E2BQ6Target(targetNative)
	draft := newDecodeGemma4E2BBF16Assistant(draftNative)

	attachment, err := AttachNativeDrafter(target, draft)

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "assistant preflight not_ready")
	core.AssertEqual(t, attachedDrafterAssistantVerifierPreflightNotReady, attachment.Labels["attached_drafter_assistant_verifier_preflight"])
	core.AssertEqual(t, attachedDrafterAssistantVerifierLayoutInvalid, attachment.Labels["attached_drafter_assistant_verifier_layout"])
	core.AssertContains(t, attachment.Labels["attached_drafter_assistant_verifier_missing"], "assistant_layer_count")
	core.AssertContains(t, attachment.Labels["attached_drafter_assistant_verifier_reason"], "assistant_layer_count=2")
}

func TestDecodeHelpers_Good_AttachedDrafterPairCloseOwnedModels(t *testing.T) {
	targetNative := &fakeNativeModel{}
	draftNative := &fakeNativeModel{}
	pair := &AttachedDrafterPair{
		Target:     &rocmModel{modelInfo: inference.ModelInfo{Architecture: "gemma4_text"}, native: targetNative},
		Draft:      &rocmModel{modelInfo: inference.ModelInfo{Architecture: "gemma4_assistant"}, native: draftNative},
		ownsTarget: true,
		ownsDraft:  true,
	}

	err := pair.Close()

	core.RequireNoError(t, err)
	core.AssertEqual(t, 1, targetNative.closeCalls)
	core.AssertEqual(t, 1, draftNative.closeCalls)
	core.AssertNil(t, pair.Target)
	core.AssertNil(t, pair.Draft)
}

func TestDecodeHelpers_Good_AttachedDrafterPairGenerateNativeUsesNativeGenerator(t *testing.T) {
	targetNative := &readyAttachedDrafterNativeModel{
		fakeNativeModel: &fakeNativeModel{},
	}
	draftNative := &fakeNativeModel{}
	target := newDecodeGemma4E2BQ6Target(targetNative)
	draft := newDecodeGemma4E2BBF16Assistant(draftNative)
	pair, err := NewAttachedDrafterPair(target, draft)
	core.RequireNoError(t, err)

	result, err := pair.GenerateNative(context.Background(), "prompt", AttachedDrafterGenerateConfig{
		MaxTokens:     4,
		Temperature:   0.7,
		TopK:          32,
		TopP:          0.9,
		MinP:          0.05,
		StopTokens:    []int32{2, 3},
		RepeatPenalty: 1,
	})

	core.RequireNoError(t, err)
	core.AssertEqual(t, true, pair.NativeReady())
	core.AssertEqual(t, inferdecode.ModeSpeculative, result.Mode)
	core.AssertEqual(t, ProductionMTPDefaultDraftTokens, result.Metrics.DraftTokens)
	core.AssertEqual(t, []string{"prompt"}, targetNative.attachedPrompts)
	core.AssertEqual(t, ProductionMTPDefaultDraftTokens, targetNative.attachedConfigs[0].DraftTokens)
	core.AssertEqual(t, true, targetNative.attachedConfigs[0].AdaptiveDraftTokens)
	core.AssertEqual(t, 4, targetNative.attachedConfigs[0].MaxTokens)
	core.AssertEqual(t, float32(0.7), targetNative.attachedConfigs[0].Temperature)
	core.AssertEqual(t, 32, targetNative.attachedConfigs[0].TopK)
	core.AssertEqual(t, float32(0.9), targetNative.attachedConfigs[0].TopP)
	core.AssertEqual(t, float32(0.05), targetNative.attachedConfigs[0].MinP)
	core.AssertEqual(t, []int32{2, 3}, targetNative.attachedConfigs[0].StopTokens)
	core.AssertEqual(t, float32(1), targetNative.attachedConfigs[0].RepeatPenalty)
	core.AssertEqual(t, 0, len(targetNative.generatePrompts))
	core.AssertEqual(t, 0, len(draftNative.generatePrompts))
}

func TestDecodeHelpers_Good_AttachedDrafterTextModelGenerateUsesNativeAttachedRoute(t *testing.T) {
	targetNative := &readyAttachedDrafterNativeModel{
		fakeNativeModel: &fakeNativeModel{
			tokens: []inference.Token{{ID: 9, Text: "target-only"}},
		},
	}
	draftNative := &fakeNativeModel{}
	target := newDecodeGemma4E2BQ6Target(targetNative)
	draft := newDecodeGemma4E2BBF16Assistant(draftNative)
	pair, err := NewAttachedDrafterPair(target, draft)
	core.RequireNoError(t, err)
	model := &attachedDrafterTextModel{pair: pair, draftTokens: 3}

	tokens := collectTokenText(model.Generate(context.Background(), "prompt",
		inference.WithMaxTokens(4),
		inference.WithTemperature(0.8),
		inference.WithTopK(64),
		inference.WithTopP(0.95),
		inference.WithMinP(0.04),
		inference.WithStopTokens(2, 3),
		inference.WithRepeatPenalty(1),
	))

	core.AssertEqual(t, []string{"ok"}, tokens)
	core.RequireNoError(t, resultError(model.Err()))
	core.AssertEqual(t, []string{"prompt"}, targetNative.attachedPrompts)
	core.AssertEqual(t, 0, len(targetNative.generatePrompts))
	core.AssertEqual(t, 4, targetNative.attachedConfigs[0].MaxTokens)
	core.AssertEqual(t, false, targetNative.attachedConfigs[0].AdaptiveDraftTokens)
	core.AssertEqual(t, float32(0.8), targetNative.attachedConfigs[0].Temperature)
	core.AssertEqual(t, 64, targetNative.attachedConfigs[0].TopK)
	core.AssertEqual(t, float32(0.95), targetNative.attachedConfigs[0].TopP)
	core.AssertEqual(t, float32(0.04), targetNative.attachedConfigs[0].MinP)
	core.AssertEqual(t, []int32{2, 3}, targetNative.attachedConfigs[0].StopTokens)
	core.AssertEqual(t, float32(1), targetNative.attachedConfigs[0].RepeatPenalty)
	core.AssertEqual(t, 1, model.Metrics().GeneratedTokens)
	mtp := model.AttachedDrafterMetrics()
	if mtp == nil {
		t.Fatal("AttachedDrafterMetrics() = nil, want speculative counters")
	}
	core.AssertEqual(t, 2, mtp.AcceptedTokens)
	core.AssertEqual(t, 1, mtp.RejectedTokens)
	core.AssertEqual(t, 3, mtp.ProposedTokens)
	core.AssertEqual(t, 1, mtp.VerifyCalls)
	core.AssertEqual(t, true, IsAttachedDrafterTextModel(model))
}

func TestDecodeHelpers_Good_AttachedDrafterTextModelOpenAIGreedyDefaultsUseNativeGreedy(t *testing.T) {
	targetNative := &readyAttachedDrafterNativeModel{
		fakeNativeModel: &fakeNativeModel{
			tokens: []inference.Token{{ID: 9, Text: "target-only"}},
		},
	}
	draftNative := &fakeNativeModel{}
	target := newDecodeGemma4E2BQ6Target(targetNative)
	draft := newDecodeGemma4E2BBF16Assistant(draftNative)
	pair, err := NewAttachedDrafterPair(target, draft)
	core.RequireNoError(t, err)
	model := &attachedDrafterTextModel{pair: pair, draftTokens: 3}

	tokens := collectTokenText(model.Generate(context.Background(), "prompt",
		inference.WithMaxTokens(4),
		inference.WithTemperature(0),
		inference.WithTopK(40),
		inference.WithTopP(1),
		inference.WithMinP(0.05),
		inference.WithRepeatPenalty(1),
	))

	core.AssertEqual(t, []string{"ok"}, tokens)
	core.RequireNoError(t, resultError(model.Err()))
	core.AssertEqual(t, []string{"prompt"}, targetNative.attachedPrompts)
	core.AssertEqual(t, 0, len(targetNative.generatePrompts))
	core.AssertEqual(t, float32(0), targetNative.attachedConfigs[0].Temperature)
	core.AssertEqual(t, 0, targetNative.attachedConfigs[0].TopK)
	core.AssertEqual(t, float32(0), targetNative.attachedConfigs[0].TopP)
	core.AssertEqual(t, float32(0), targetNative.attachedConfigs[0].MinP)
	core.AssertEqual(t, float32(1), targetNative.attachedConfigs[0].RepeatPenalty)
}

func TestDecodeHelpers_Good_AttachedDrafterTextModelGenerateKeepsNativePairVisibleWhenNativeSelected(t *testing.T) {
	targetNative := &readyAttachedDrafterNativeModel{
		fakeNativeModel: &fakeNativeModel{
			tokens: []inference.Token{{ID: 9, Text: "target-only"}},
		},
	}
	draftNative := &fakeNativeModel{}
	target := newDecodeGemma4E2BQ6Target(targetNative)
	draft := newDecodeGemma4E2BBF16Assistant(draftNative)
	pair, err := NewAttachedDrafterPair(target, draft)
	core.RequireNoError(t, err)
	pair.Attachment.Labels["attached_drafter_native_handoff"] = attachedDrafterNativeHandoffRetainedStateVerifier
	pair.Attachment.Labels["attached_drafter_prompt_replay_fallback"] = "forbidden"
	pair.Attachment.Labels["attached_drafter_retained_state_entrypoint"] = hipKernelStatusLinked
	pair.Attachment.Labels["attached_drafter_retained_state_required"] = "true"
	pair.Attachment.Labels["attached_drafter_state_source"] = "rocm_state_session_runtime_kv"
	pair.Attachment.Labels["attached_drafter_target_retained_decode"] = hipKernelStatusLinked
	pair.Attachment.Labels["attached_drafter_target_retained_state_decode"] = hipKernelStatusLinked
	pair.Attachment.Labels["attached_drafter_assistant_verify"] = hipKernelStatusLinked
	pair.Attachment.Labels["attached_drafter_assistant_state_verify"] = hipKernelStatusLinked
	model := &attachedDrafterTextModel{pair: pair, draftTokens: 3}

	tokens := collectTokenText(model.Generate(context.Background(), "prompt", inference.WithMaxTokens(4)))

	core.AssertEqual(t, []string{"ok"}, tokens)
	core.RequireNoError(t, resultError(model.Err()))
	core.AssertEqual(t, []string{"prompt"}, targetNative.attachedPrompts)
	core.AssertEqual(t, 0, len(targetNative.generatePrompts))
	core.AssertEqual(t, 0, len(draftNative.generatePrompts))
	core.AssertEqual(t, 1, len(targetNative.attachedConfigs))
	identity := model.ModelIdentity()
	core.AssertEqual(t, hipKernelStatusLinked, identity.Labels["attached_drafter_native_attachment"])
	core.AssertEqual(t, "native_attached_retained_state", identity.Labels["attached_drafter_generation_route"])
	core.AssertEqual(t, "target_equivalent_batched_prefill", identity.Labels["attached_drafter_generation_route_reason"])
}

func TestDecodeHelpers_Good_AttachedDrafterTextModelGenerateUsesNativeAttachedWithTargetStatePresent(t *testing.T) {
	var _ inference.AgentMemorySession = (*attachedDrafterTextModel)(nil)

	targetNative := &readyAttachedDrafterNativeModel{
		fakeNativeModel: &fakeNativeModel{
			tokens: []inference.Token{{ID: 1, Text: "ok"}},
		},
	}
	draftNative := &fakeNativeModel{}
	cache, err := newROCmKVCache(rocmKVCacheModeQ8, 2)
	core.RequireNoError(t, err)
	core.RequireNoError(t, cache.AppendVectors(0, 2, 2, []float32{1, 2}, []float32{3, 4}))
	target := newDecodeGemma4E2BQ6Target(targetNative)
	target.state = newStateSessionWithRuntime(target.modelIdentity(), inference.TokenizerIdentity{}, nil, cache)
	draft := newDecodeGemma4E2BBF16Assistant(draftNative)
	pair, err := NewAttachedDrafterPair(target, draft)
	core.RequireNoError(t, err)
	model := &attachedDrafterTextModel{pair: pair, draftTokens: 3}

	tokens := collectTokenText(model.Generate(context.Background(), "new turn only",
		inference.WithMaxTokens(4),
		inference.WithTemperature(0.8),
		inference.WithTopK(64),
		inference.WithTopP(0.95),
		inference.WithMinP(0.04),
		inference.WithStopTokens(2, 3),
		inference.WithRepeatPenalty(1),
	))

	core.AssertEqual(t, []string{"ok"}, tokens)
	core.RequireNoError(t, resultError(model.Err()))
	core.AssertEqual(t, 0, len(targetNative.attachedStateInputs))
	core.AssertEqual(t, []string{"new turn only"}, targetNative.attachedPrompts)
	core.AssertEqual(t, 0, len(targetNative.generatePrompts))
	core.AssertEqual(t, 4, targetNative.attachedConfigs[0].MaxTokens)
	core.AssertEqual(t, float32(0.8), targetNative.attachedConfigs[0].Temperature)
	core.AssertEqual(t, 64, targetNative.attachedConfigs[0].TopK)
	core.AssertEqual(t, float32(0.95), targetNative.attachedConfigs[0].TopP)
	core.AssertEqual(t, float32(0.04), targetNative.attachedConfigs[0].MinP)
	core.AssertEqual(t, []int32{2, 3}, targetNative.attachedConfigs[0].StopTokens)
	core.AssertEqual(t, float32(1), targetNative.attachedConfigs[0].RepeatPenalty)
	core.AssertEqual(t, 0, len(draftNative.generatePrompts))
	core.AssertEqual(t, 1, model.Metrics().GeneratedTokens)
}

func TestDecodeHelpers_Good_AttachedDrafterTextModelChatPromptUsesContinuationTemplateWithRuntimeState(t *testing.T) {
	loaded := &hipLoadedModel{modelInfo: gemma4DecodeE2BQ6Info()}
	target := newDecodeGemma4E2BQ6Target(loaded)
	cache, err := newROCmKVCache(rocmKVCacheModeQ8, 2)
	core.RequireNoError(t, err)
	core.RequireNoError(t, cache.AppendVectors(0, 2, 2, []float32{1, 2}, []float32{3, 4}))
	target.state = newStateSessionWithRuntime(target.modelIdentity(), inference.TokenizerIdentity{}, nil, cache)
	model := &attachedDrafterTextModel{
		pair: &AttachedDrafterPair{Target: target},
	}

	prompt, err := model.chatPromptWithStatePreference(target, []inference.Message{{Role: "user", Content: "second turn"}}, inference.GenerateConfig{}, true)

	core.RequireNoError(t, err)
	core.AssertEqual(t, true, strings.HasPrefix(prompt, "<turn|>\n<|turn>user\nsecond turn<turn|>\n<|turn>model\n"))
	core.AssertEqual(t, false, strings.HasPrefix(prompt, "<bos>"))

	fresh, err := model.chatPromptWithStatePreference(target, []inference.Message{{Role: "user", Content: "first turn"}}, inference.GenerateConfig{}, false)
	core.RequireNoError(t, err)
	core.AssertEqual(t, true, strings.HasPrefix(fresh, "<bos>"))
}

func TestDecodeHelpers_Good_AttachedDrafterTextModelSeedsStateBeforeNativeAttachedFromState(t *testing.T) {
	targetNative := &readyAttachedDrafterNativeModel{
		fakeNativeModel: &fakeNativeModel{
			tokens: []inference.Token{{ID: 12, Text: "seed"}},
		},
	}
	draftNative := &fakeNativeModel{}
	target := newDecodeGemma4E2BQ6Target(targetNative)
	defer func() { core.RequireNoError(t, target.ResetState()) }()
	draft := newDecodeGemma4E2BBF16Assistant(draftNative)
	cache, err := newROCmKVCache(rocmKVCacheModeQ8, 2)
	core.RequireNoError(t, err)
	core.RequireNoError(t, cache.AppendVectors(0, 2, 2, []float32{1, 2}, []float32{3, 4}))
	targetNative.afterStream = func() {
		target.stateMutex.Lock()
		defer target.stateMutex.Unlock()
		if target.state == nil {
			target.state = newStateSessionWithRuntime(target.modelIdentity(), inference.TokenizerIdentity{}, nil, cache)
		}
	}
	plan, err := PlanAttachedDrafter(target, draft)
	core.RequireNoError(t, err)
	labels := cloneStringMap(plan.Labels)
	labels["attached_drafter_native_attachment"] = hipKernelStatusLinked
	labels["attached_drafter_native_handoff"] = attachedDrafterNativeHandoffRetainedStateVerifier
	labels["attached_drafter_prompt_replay_fallback"] = "forbidden"
	labels["attached_drafter_retained_state_entrypoint"] = hipKernelStatusLinked
	labels["attached_drafter_retained_state_required"] = "true"
	labels["attached_drafter_state_source"] = "rocm_state_session_runtime_kv"
	labels["attached_drafter_target_retained_decode"] = hipKernelStatusLinked
	labels["attached_drafter_target_retained_state_decode"] = hipKernelStatusLinked
	labels["attached_drafter_assistant_verify"] = hipKernelStatusLinked
	labels["attached_drafter_assistant_state_verify"] = hipKernelStatusLinked
	model := &attachedDrafterTextModel{
		pair: &AttachedDrafterPair{
			Target: target,
			Draft:  draft,
			Plan:   plan,
			Attachment: AttachedDrafterAttachment{
				Plan:             plan,
				Target:           plan.Target,
				Draft:            plan.Draft,
				NativeAttachment: hipKernelStatusLinked,
				Labels:           labels,
			},
		},
		draftTokens: 3,
	}

	first := collectTokenText(model.Chat(context.Background(), []inference.Message{{Role: "user", Content: "first turn"}}, inference.WithMaxTokens(4)))
	second := collectTokenText(model.Chat(context.Background(), []inference.Message{{Role: "user", Content: "second turn"}}, inference.WithMaxTokens(5)))

	core.AssertEqual(t, []string{"ok"}, first)
	core.AssertEqual(t, []string{"ok"}, second)
	core.RequireNoError(t, resultError(model.Err()))
	core.AssertEqual(t, []string{"user:first turn\n"}, targetNative.attachedRetainedPrompts)
	core.AssertEqual(t, 4, targetNative.attachedRetainedConfigs[0].MaxTokens)
	core.AssertEqual(t, 1, len(targetNative.attachedRetainedStates))
	core.AssertEqual(t, []string{"user:second turn\n"}, targetNative.attachedStateInputs)
	core.AssertEqual(t, 5, targetNative.attachedStateRequests[0].MaxTokens)
	core.AssertEqual(t, 0, len(targetNative.generatePrompts))
	core.AssertEqual(t, 0, len(targetNative.attachedPrompts))
	core.AssertEqual(t, 0, len(draftNative.generatePrompts))
}

func TestDecodeHelpers_Good_AttachedDrafterTextModelOpenAIGreedyDefaultsUseRetainedNativeGreedy(t *testing.T) {
	targetNative := &readyAttachedDrafterNativeModel{
		fakeNativeModel: &fakeNativeModel{
			tokens: []inference.Token{{ID: 12, Text: "seed"}},
		},
	}
	draftNative := &fakeNativeModel{}
	target := newDecodeGemma4E2BQ6Target(targetNative)
	defer func() { core.RequireNoError(t, target.ResetState()) }()
	draft := newDecodeGemma4E2BBF16Assistant(draftNative)
	cache, err := newROCmKVCache(rocmKVCacheModeQ8, 2)
	core.RequireNoError(t, err)
	core.RequireNoError(t, cache.AppendVectors(0, 2, 2, []float32{1, 2}, []float32{3, 4}))
	targetNative.afterStream = func() {
		target.stateMutex.Lock()
		defer target.stateMutex.Unlock()
		if target.state == nil {
			target.state = newStateSessionWithRuntime(target.modelIdentity(), inference.TokenizerIdentity{}, nil, cache)
		}
	}
	plan, err := PlanAttachedDrafter(target, draft)
	core.RequireNoError(t, err)
	labels := cloneStringMap(plan.Labels)
	labels["attached_drafter_native_attachment"] = hipKernelStatusLinked
	labels["attached_drafter_native_handoff"] = attachedDrafterNativeHandoffRetainedStateVerifier
	labels["attached_drafter_prompt_replay_fallback"] = "forbidden"
	labels["attached_drafter_retained_state_entrypoint"] = hipKernelStatusLinked
	labels["attached_drafter_retained_state_required"] = "true"
	labels["attached_drafter_state_source"] = "rocm_state_session_runtime_kv"
	labels["attached_drafter_target_retained_decode"] = hipKernelStatusLinked
	labels["attached_drafter_target_retained_state_decode"] = hipKernelStatusLinked
	labels["attached_drafter_assistant_verify"] = hipKernelStatusLinked
	labels["attached_drafter_assistant_state_verify"] = hipKernelStatusLinked
	model := &attachedDrafterTextModel{
		pair: &AttachedDrafterPair{
			Target: target,
			Draft:  draft,
			Plan:   plan,
			Attachment: AttachedDrafterAttachment{
				Plan:             plan,
				Target:           plan.Target,
				Draft:            plan.Draft,
				NativeAttachment: hipKernelStatusLinked,
				Labels:           labels,
			},
		},
		draftTokens: 3,
	}

	first := collectTokenText(model.Chat(context.Background(), []inference.Message{{Role: "user", Content: "first turn"}}, inference.WithMaxTokens(4)))
	second := collectTokenText(model.Chat(context.Background(), []inference.Message{{Role: "user", Content: "second turn"}},
		inference.WithMaxTokens(5),
		inference.WithTemperature(0),
		inference.WithTopK(40),
		inference.WithTopP(1),
		inference.WithMinP(0.05),
	))

	core.AssertEqual(t, []string{"ok"}, first)
	core.AssertEqual(t, []string{"ok"}, second)
	core.RequireNoError(t, resultError(model.Err()))
	core.AssertEqual(t, []string{"user:first turn\n"}, targetNative.attachedRetainedPrompts)
	core.AssertEqual(t, 4, targetNative.attachedRetainedConfigs[0].MaxTokens)
	core.AssertEqual(t, []string{"user:second turn\n"}, targetNative.attachedStateInputs)
	core.AssertEqual(t, 5, targetNative.attachedStateRequests[0].MaxTokens)
	core.AssertEqual(t, float32(0), targetNative.attachedStateRequests[0].Temperature)
	core.AssertEqual(t, 0, targetNative.attachedStateRequests[0].TopK)
	core.AssertEqual(t, float32(0), targetNative.attachedStateRequests[0].TopP)
	core.AssertEqual(t, float32(0), targetNative.attachedStateRequests[0].MinP)
	core.AssertEqual(t, 0, len(targetNative.generatePrompts))
	core.AssertEqual(t, 0, len(draftNative.generatePrompts))
}

func TestDecodeHelpers_Good_AttachedDrafterTextModelUsesTargetRetainedStateWhenMTPPending(t *testing.T) {
	targetNative := &fakeNativeModel{
		tokens: []inference.Token{{ID: 9, Text: "target-only"}},
	}
	draftNative := &fakeNativeModel{}
	cache, err := newROCmKVCache(rocmKVCacheModeQ8, 2)
	core.RequireNoError(t, err)
	core.RequireNoError(t, cache.AppendVectors(0, 2, 2, []float32{1, 2}, []float32{3, 4}))
	target := newDecodeGemma4E2BQ6Target(targetNative)
	target.state = newStateSessionWithRuntime(target.modelIdentity(), inference.TokenizerIdentity{}, nil, cache)
	draft := newDecodeGemma4E2BBF16Assistant(draftNative)
	model := newPendingTargetRetainedAttachedDrafterTextModel(t, target, draft, 3)

	tokens := collectTokenText(model.Generate(context.Background(), "new turn only",
		inference.WithMaxTokens(4),
		inference.WithTemperature(0.7),
		inference.WithTopK(32),
		inference.WithTopP(0.9),
		inference.WithMinP(0.02),
		inference.WithStopTokens(5, 6),
	))

	core.AssertEqual(t, []string{"target-only"}, tokens)
	core.RequireNoError(t, resultError(model.Err()))
	core.AssertEqual(t, []string{"new turn only"}, targetNative.generatePrompts)
	core.AssertEqual(t, 4, targetNative.generateConfigs[0].MaxTokens)
	core.AssertEqual(t, float32(0.7), targetNative.generateConfigs[0].Temperature)
	core.AssertEqual(t, 32, targetNative.generateConfigs[0].TopK)
	core.AssertEqual(t, float32(0.9), targetNative.generateConfigs[0].TopP)
	core.AssertEqual(t, float32(0.02), targetNative.generateConfigs[0].MinP)
	core.AssertEqual(t, []int32{5, 6}, targetNative.generateConfigs[0].StopTokens)
	core.AssertEqual(t, 0, len(draftNative.generatePrompts))
	core.AssertEqual(t, 1, model.Metrics().GeneratedTokens)
}

func TestDecodeHelpers_Good_AttachedDrafterTextModelUsesTargetRetainedDecodeWhenMTPPendingFreshTurn(t *testing.T) {
	targetNative := &fakeNativeModel{
		tokens: []inference.Token{{ID: 11, Text: "fresh-target"}},
	}
	draftNative := &fakeNativeModel{}
	target := newDecodeGemma4E2BQ6Target(targetNative)
	draft := newDecodeGemma4E2BBF16Assistant(draftNative)
	model := newPendingTargetRetainedAttachedDrafterTextModel(t, target, draft, 3)

	tokens := collectTokenText(model.Generate(context.Background(), "first turn",
		inference.WithMaxTokens(5),
		inference.WithTemperature(0.6),
		inference.WithTopK(24),
		inference.WithTopP(0.88),
		inference.WithMinP(0.01),
		inference.WithStopTokens(7, 8),
	))

	core.AssertEqual(t, []string{"fresh-target"}, tokens)
	core.RequireNoError(t, resultError(model.Err()))
	core.AssertEqual(t, []string{"first turn"}, targetNative.generatePrompts)
	core.AssertEqual(t, 5, targetNative.generateConfigs[0].MaxTokens)
	core.AssertEqual(t, float32(0.6), targetNative.generateConfigs[0].Temperature)
	core.AssertEqual(t, 24, targetNative.generateConfigs[0].TopK)
	core.AssertEqual(t, float32(0.88), targetNative.generateConfigs[0].TopP)
	core.AssertEqual(t, float32(0.01), targetNative.generateConfigs[0].MinP)
	core.AssertEqual(t, []int32{7, 8}, targetNative.generateConfigs[0].StopTokens)
	core.AssertEqual(t, 0, len(draftNative.generatePrompts))
	core.AssertEqual(t, 1, model.Metrics().GeneratedTokens)
}

func TestDecodeHelpers_Good_AttachedDrafterTextModelRepeatPenaltyUsesReadyTargetFallback(t *testing.T) {
	targetNative := &readyAttachedDrafterNativeModel{
		fakeNativeModel: &fakeNativeModel{
			tokens: []inference.Token{{ID: 7, Text: "plain"}},
		},
	}
	draftNative := &fakeNativeModel{}
	target := newDecodeGemma4E2BQ6Target(targetNative)
	draft := newDecodeGemma4E2BBF16Assistant(draftNative)
	pair, err := NewAttachedDrafterPair(target, draft)
	core.RequireNoError(t, err)
	model := &attachedDrafterTextModel{pair: pair, draftTokens: 3}

	tokens := collectTokenText(model.Generate(context.Background(), "prompt",
		inference.WithMaxTokens(4),
		inference.WithRepeatPenalty(1.2),
	))

	core.AssertEqual(t, []string{"plain"}, tokens)
	core.RequireNoError(t, resultError(model.Err()))
	core.AssertEqual(t, 0, len(targetNative.attachedPrompts))
	core.AssertEqual(t, []string{"prompt"}, targetNative.generatePrompts)
	core.AssertEqual(t, 4, targetNative.generateConfigs[0].MaxTokens)
	core.AssertEqual(t, float32(1.2), targetNative.generateConfigs[0].RepeatPenalty)

	tokens = collectTokenText(model.Generate(context.Background(), "prompt",
		inference.WithMaxTokens(5),
		inference.WithMinP(0.03),
		inference.WithRepeatPenalty(1.2),
	))

	core.AssertEqual(t, []string{"plain"}, tokens)
	core.RequireNoError(t, resultError(model.Err()))
	core.AssertEqual(t, 5, targetNative.generateConfigs[1].MaxTokens)
	core.AssertEqual(t, float32(0.03), targetNative.generateConfigs[1].MinP)
	core.AssertEqual(t, float32(1.2), targetNative.generateConfigs[1].RepeatPenalty)
}

func TestDecodeHelpers_Bad_AttachedDrafterTextModelRejectsNotReadyNoFallback(t *testing.T) {
	targetNative := &hipLoadedModel{modelInfo: gemma4DecodeE2BQ6Info()}
	draftNative := &hipLoadedModel{modelInfo: gemma4DecodeE2BBF16AssistantInfo()}
	target := newDecodeGemma4E2BQ6Target(targetNative)
	draft := newDecodeGemma4E2BBF16Assistant(draftNative)
	pair, err := NewAttachedDrafterPair(target, draft)
	core.RequireNoError(t, err)
	model := &attachedDrafterTextModel{pair: pair, draftTokens: 4}

	tokens := collectTokenText(model.Generate(context.Background(), "prompt", inference.WithMaxTokens(4)))

	core.AssertEqual(t, []string{}, tokens)
	core.AssertError(t, resultError(model.Err()))
	core.AssertContains(t, model.Err().Error(), "native HIP drafter generation is not linked yet")
	core.AssertEqual(t, false, pair.NativeReady())
}

func TestDecodeHelpers_Good_AttachedDrafterPairGenerateNativeUsesGemma4RemainingWindow(t *testing.T) {
	targetNative := &readyAttachedDrafterNativeModel{
		fakeNativeModel: &fakeNativeModel{encodeResult: []int32{1, 2, 3}},
	}
	draftNative := &fakeNativeModel{}
	target := newDecodeGemma4E2BQ6Target(targetNative)
	draft := newDecodeGemma4E2BBF16Assistant(draftNative)
	pair, err := NewAttachedDrafterPair(target, draft)
	core.RequireNoError(t, err)

	result, err := pair.GenerateNative(context.Background(), "ignored", AttachedDrafterGenerateConfig{})

	core.RequireNoError(t, err)
	core.AssertEqual(t, inferdecode.ModeSpeculative, result.Mode)
	core.AssertEqual(t, defaultContextLengthCap-3, targetNative.attachedConfigs[0].MaxTokens)
	core.AssertEqual(t, ProductionMTPDefaultDraftTokens, targetNative.attachedConfigs[0].DraftTokens)

	result, err = pair.GenerateNative(context.Background(), "ignored", AttachedDrafterGenerateConfig{MaxTokens: -1})

	core.RequireNoError(t, err)
	core.AssertEqual(t, inferdecode.ModeSpeculative, result.Mode)
	core.AssertEqual(t, defaultContextLengthCap-3, targetNative.attachedConfigs[1].MaxTokens)
	core.AssertEqual(t, ProductionMTPDefaultDraftTokens, targetNative.attachedConfigs[1].DraftTokens)
}

func TestDecodeHelpers_Good_AttachedDrafterPairGenerateNativeFromStateUsesStateGenerator(t *testing.T) {
	targetNative := &readyAttachedDrafterNativeModel{
		fakeNativeModel: &fakeNativeModel{},
	}
	draftNative := &fakeNativeModel{}
	target := newDecodeGemma4E2BQ6Target(targetNative)
	draft := newDecodeGemma4E2BBF16Assistant(draftNative)
	pair, err := NewAttachedDrafterPair(target, draft)
	core.RequireNoError(t, err)
	cache, err := newROCmKVCache(rocmKVCacheModeQ8, 2)
	core.RequireNoError(t, err)
	core.RequireNoError(t, cache.AppendVectors(0, 2, 2, []float32{1, 2, 3, 4}, []float32{5, 6, 7, 8}))
	state := newStateSessionWithRuntime(inference.ModelIdentity{}, inference.TokenizerIdentity{}, nil, cache)

	result, err := pair.GenerateNativeFromState(context.Background(), AttachedDrafterStateGenerateRequest{
		State:       state,
		Input:       "new turn only",
		MaxTokens:   4,
		Temperature: 0.7,
		TopK:        32,
		TopP:        0.9,
		MinP:        0.05,
	})

	core.RequireNoError(t, err)
	core.AssertEqual(t, inferdecode.ModeSpeculative, result.Mode)
	core.AssertEqual(t, ProductionMTPDefaultDraftTokens, result.Metrics.DraftTokens)
	core.AssertEqual(t, 4, result.Metrics.EmittedTokens)
	core.AssertEqual(t, []string{"new turn only"}, targetNative.attachedStateInputs)
	core.AssertEqual(t, ProductionMTPDefaultDraftTokens, targetNative.attachedStateRequests[0].DraftTokens)
	core.AssertEqual(t, true, targetNative.attachedStateRequests[0].AdaptiveDraftTokens)
	core.AssertEqual(t, float32(0.7), targetNative.attachedStateRequests[0].Temperature)
	core.AssertEqual(t, 32, targetNative.attachedStateRequests[0].TopK)
	core.AssertEqual(t, float32(0.9), targetNative.attachedStateRequests[0].TopP)
	core.AssertEqual(t, float32(0.05), targetNative.attachedStateRequests[0].MinP)
	core.AssertEqual(t, hipKernelStatusLinked, targetNative.attachedStateAttachmentLabels[0])
	core.AssertEqual(t, 0, len(targetNative.attachedPrompts))
	core.AssertEqual(t, 0, len(targetNative.generatePrompts))
	core.AssertEqual(t, 0, len(draftNative.generatePrompts))
}

func TestDecodeHelpers_Good_AttachedDrafterPairGenerateNativeFromStateUsesGemma4Q4DeviceState(t *testing.T) {
	targetNative := &readyAttachedDrafterNativeModel{
		fakeNativeModel: &fakeNativeModel{encodeResult: []int32{1, 2}},
	}
	draftNative := &fakeNativeModel{}
	target := newDecodeGemma4E2BQ6Target(targetNative)
	draft := newDecodeGemma4E2BBF16Assistant(draftNative)
	pair, err := NewAttachedDrafterPair(target, draft)
	core.RequireNoError(t, err)
	deviceState := &hipGemma4Q4DeviceDecodeState{layers: []hipGemma4Q4DeviceLayerKVState{
		{cache: &rocmDeviceKVCache{tokenCount: 5}},
		{cache: &rocmDeviceKVCache{tokenCount: 3}},
	}}
	state := newStateSessionWithRuntime(inference.ModelIdentity{}, inference.TokenizerIdentity{}, nil, deviceState)

	result, err := pair.GenerateNativeFromState(context.Background(), AttachedDrafterStateGenerateRequest{
		State:     state,
		Input:     "new turn only",
		MaxTokens: 4,
	})

	core.RequireNoError(t, err)
	core.AssertEqual(t, inferdecode.ModeSpeculative, result.Mode)
	core.AssertEqual(t, 4, result.Metrics.EmittedTokens)
	core.AssertEqual(t, []string{"new turn only"}, targetNative.attachedStateInputs)
	core.AssertEqual(t, state, targetNative.attachedStateRequests[0].State)
	core.AssertEqual(t, 4, targetNative.attachedStateRequests[0].MaxTokens)
	core.AssertEqual(t, ProductionMTPDefaultDraftTokens, targetNative.attachedStateRequests[0].DraftTokens)
	core.AssertEqual(t, 0, len(targetNative.attachedPrompts))
	core.AssertEqual(t, 0, len(targetNative.generatePrompts))
	core.AssertEqual(t, 0, len(draftNative.generatePrompts))
}

func TestDecodeHelpers_Good_AttachedDrafterPairGenerateNativeFromStateUsesGemma4Q4HostRuntimeState(t *testing.T) {
	targetNative := &readyAttachedDrafterNativeModel{
		fakeNativeModel: &fakeNativeModel{encodeResult: []int32{1, 2, 3}},
	}
	draftNative := &fakeNativeModel{}
	target := newDecodeGemma4E2BQ6Target(targetNative)
	draft := newDecodeGemma4E2BBF16Assistant(draftNative)
	pair, err := NewAttachedDrafterPair(target, draft)
	core.RequireNoError(t, err)
	state := newStateSessionWithRuntime(inference.ModelIdentity{}, inference.TokenizerIdentity{}, nil, &hipGemma4Q4HostDecodeStateRuntime{
		tokenCount: 7,
	})

	result, err := pair.GenerateNativeFromState(context.Background(), AttachedDrafterStateGenerateRequest{
		State: state,
		Input: "new turn only",
	})

	core.RequireNoError(t, err)
	core.AssertEqual(t, inferdecode.ModeSpeculative, result.Mode)
	core.AssertEqual(t, defaultContextLengthCap-7-3, targetNative.attachedStateRequests[0].MaxTokens)
	core.AssertEqual(t, state, targetNative.attachedStateRequests[0].State)
	core.RequireTrue(t, state.hasRuntimeOwnedKV())
	core.AssertEqual(t, 0, len(targetNative.attachedPrompts))
	core.AssertEqual(t, 0, len(targetNative.generatePrompts))
	core.AssertEqual(t, 0, len(draftNative.generatePrompts))
}

func TestDecodeHelpers_Good_AttachedDrafterPairGenerateNativeFromStateUsesRemainingWindow(t *testing.T) {
	targetNative := &readyAttachedDrafterNativeModel{
		fakeNativeModel: &fakeNativeModel{encodeResult: []int32{1, 2, 3}},
	}
	draftNative := &fakeNativeModel{}
	target := newDecodeGemma4E2BQ6Target(targetNative)
	draft := newDecodeGemma4E2BBF16Assistant(draftNative)
	pair, err := NewAttachedDrafterPair(target, draft)
	core.RequireNoError(t, err)
	cache, err := newROCmKVCache(rocmKVCacheModeQ8, 2)
	core.RequireNoError(t, err)
	core.RequireNoError(t, cache.AppendVectors(0, 2, 2, []float32{1, 2, 3, 4}, []float32{5, 6, 7, 8}))
	state := newStateSessionWithRuntime(inference.ModelIdentity{}, inference.TokenizerIdentity{}, nil, cache)

	result, err := pair.GenerateNativeFromState(context.Background(), AttachedDrafterStateGenerateRequest{
		State: state,
		Input: "new turn only",
	})

	core.RequireNoError(t, err)
	core.AssertEqual(t, inferdecode.ModeSpeculative, result.Mode)
	core.AssertEqual(t, defaultContextLengthCap-2-3, targetNative.attachedStateRequests[0].MaxTokens)
	core.AssertEqual(t, ProductionMTPDefaultDraftTokens, targetNative.attachedStateRequests[0].DraftTokens)

	result, err = pair.GenerateNativeFromState(context.Background(), AttachedDrafterStateGenerateRequest{
		State:     state,
		Input:     "new turn only",
		MaxTokens: -1,
	})

	core.RequireNoError(t, err)
	core.AssertEqual(t, inferdecode.ModeSpeculative, result.Mode)
	core.AssertEqual(t, defaultContextLengthCap-2-3, targetNative.attachedStateRequests[1].MaxTokens)
	core.AssertEqual(t, ProductionMTPDefaultDraftTokens, targetNative.attachedStateRequests[1].DraftTokens)
}

func TestDecodeHelpers_Good_AttachedDrafterPairGenerateNativeRetainedUsesTargetState(t *testing.T) {
	targetNative := &readyAttachedDrafterNativeModel{
		fakeNativeModel: &fakeNativeModel{},
	}
	draftNative := &fakeNativeModel{}
	cache, err := newROCmKVCache(rocmKVCacheModeQ8, 2)
	core.RequireNoError(t, err)
	core.RequireNoError(t, cache.AppendVectors(0, 2, 2, []float32{1, 2}, []float32{3, 4}))
	target := newDecodeGemma4E2BQ6Target(targetNative)
	target.state = newStateSessionWithRuntime(inference.ModelIdentity{}, inference.TokenizerIdentity{}, nil, cache)
	draft := newDecodeGemma4E2BBF16Assistant(draftNative)
	pair, err := NewAttachedDrafterPair(target, draft)
	core.RequireNoError(t, err)

	result, err := pair.GenerateNativeRetained(context.Background(), "new turn only", AttachedDrafterGenerateConfig{MaxTokens: 4})

	core.RequireNoError(t, err)
	core.AssertEqual(t, inferdecode.ModeSpeculative, result.Mode)
	core.AssertEqual(t, []string{"new turn only"}, targetNative.attachedStateInputs)
	core.AssertEqual(t, ProductionMTPDefaultDraftTokens, targetNative.attachedStateRequests[0].DraftTokens)
	core.AssertEqual(t, 4, targetNative.attachedStateRequests[0].MaxTokens)
	core.AssertEqual(t, hipKernelStatusLinked, targetNative.attachedStateAttachmentLabels[0])
	core.AssertEqual(t, 0, len(targetNative.attachedPrompts))
	core.AssertEqual(t, 0, len(targetNative.generatePrompts))
	core.AssertEqual(t, 0, len(draftNative.generatePrompts))
}

func TestDecodeHelpers_Good_PromptLookupDecodeUsesLookupDraft(t *testing.T) {
	target := &rocmModel{native: &fakeNativeModel{tokens: []inference.Token{{ID: 3}, {ID: 4}, {ID: 9}}}}

	result, err := PromptLookupDecode(context.Background(), target, PromptLookupDecodeConfig{
		Prompt:       "p",
		MaxTokens:    3,
		LookupTokens: []int32{3, 4, 8},
	})

	core.RequireNoError(t, err)
	core.AssertEqual(t, inferdecode.ModePromptLookup, result.Mode)
	core.AssertEqual(t, 2, result.Metrics.AcceptedTokens)
	core.AssertEqual(t, 1, result.Metrics.RejectedTokens)
	core.AssertEqual(t, 3, result.Metrics.LookupTokens)
}

func TestDecodeHelpers_Good_PromptLookupDecodeDerivesLookupTokensFromEncoder(t *testing.T) {
	model := &rocmModel{native: &fakeNativeModel{
		tokens:       []inference.Token{{ID: 3}, {ID: 4}, {ID: 9}},
		encodeResult: []int32{1, 2, 3, 4, 1, 2},
	}}

	result, err := PromptLookupDecode(context.Background(), model, PromptLookupDecodeConfig{
		Prompt:    "ignored",
		MaxTokens: 3,
		MaxDraft:  2,
	})

	core.RequireNoError(t, err)
	core.AssertEqual(t, inferdecode.ModePromptLookup, result.Mode)
	core.AssertEqual(t, 2, result.Metrics.AcceptedTokens)
	core.AssertEqual(t, 0, result.Metrics.RejectedTokens)
	core.AssertEqual(t, 3, result.Metrics.EmittedTokens)
	core.AssertEqual(t, 2, result.Metrics.LookupTokens)
}

func TestDecodeHelpers_Good_PromptLookupDecodeUsesGemma4RemainingWindow(t *testing.T) {
	native := &fakeNativeModel{
		tokens:       []inference.Token{{ID: 3}, {ID: 4}, {ID: 9}},
		encodeResult: []int32{1, 2, 3, 4, 1, 2},
	}
	model := &rocmModel{
		modelInfo: inference.ModelInfo{Architecture: "gemma4_text"},
		native:    native,
	}

	result, err := PromptLookupDecode(context.Background(), model, PromptLookupDecodeConfig{
		Prompt:   "ignored",
		MaxDraft: 2,
	})

	core.RequireNoError(t, err)
	core.AssertEqual(t, inferdecode.ModePromptLookup, result.Mode)
	core.AssertEqual(t, defaultContextLengthCap-6, native.generateConfigs[0].MaxTokens)
	core.AssertEqual(t, 2, result.Metrics.LookupTokens)

	result, err = PromptLookupDecode(context.Background(), model, PromptLookupDecodeConfig{
		Prompt:    "ignored",
		MaxTokens: -1,
		MaxDraft:  2,
	})

	core.RequireNoError(t, err)
	core.AssertEqual(t, inferdecode.ModePromptLookup, result.Mode)
	core.AssertEqual(t, defaultContextLengthCap-6, native.generateConfigs[1].MaxTokens)
	core.AssertEqual(t, 2, result.Metrics.LookupTokens)
}

func TestDecodeHelpers_Good_PromptLookupMaxDraftUsesGemma4RemainingWindow(t *testing.T) {
	model := &rocmModel{
		modelInfo: inference.ModelInfo{Architecture: "gemma4_text"},
		native: &hipLoadedModel{
			modelInfo:   inference.ModelInfo{Architecture: "gemma4_text", QuantBits: 6},
			contextSize: 15,
		},
	}

	got, err := rocmPromptLookupMaxDraft(model, PromptLookupDecodeConfig{}, []int32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10})

	core.RequireNoError(t, err)
	core.AssertEqual(t, 5, got)

	got, err = rocmPromptLookupMaxDraft(model, PromptLookupDecodeConfig{MaxTokens: -1}, []int32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10})

	core.RequireNoError(t, err)
	core.AssertEqual(t, 5, got)
}

func TestDecodeHelpers_Good_PromptLookupMaxDraftKeepsExplicitLimits(t *testing.T) {
	model := &rocmModel{
		modelInfo: inference.ModelInfo{Architecture: "gemma4_text"},
		native: &hipLoadedModel{
			modelInfo:   inference.ModelInfo{Architecture: "gemma4_text", QuantBits: 6},
			contextSize: 15,
		},
	}

	got, err := rocmPromptLookupMaxDraft(model, PromptLookupDecodeConfig{MaxDraft: 3, MaxTokens: 4}, []int32{1, 2})
	core.RequireNoError(t, err)
	core.AssertEqual(t, 3, got)

	got, err = rocmPromptLookupMaxDraft(model, PromptLookupDecodeConfig{MaxTokens: 4}, []int32{1, 2})
	core.RequireNoError(t, err)
	core.AssertEqual(t, 4, got)
}

func TestDecodeHelpers_Good_PromptLookupMaxDraftKeepsNonGemmaExplicitLimit(t *testing.T) {
	model := &rocmModel{modelInfo: inference.ModelInfo{Architecture: "qwen3"}}

	got, err := rocmPromptLookupMaxDraft(model, PromptLookupDecodeConfig{MaxDraft: defaultROCmPromptLookupMaxDraft + 1}, []int32{1, 2, 3})

	core.RequireNoError(t, err)
	core.AssertEqual(t, defaultROCmPromptLookupMaxDraft+1, got)
}

func TestDecodeHelpers_Good_PromptLookupMaxDraftKeepsNonGemmaDefault(t *testing.T) {
	model := &rocmModel{modelInfo: inference.ModelInfo{Architecture: "qwen3"}}

	got, err := rocmPromptLookupMaxDraft(model, PromptLookupDecodeConfig{}, []int32{1, 2, 3})

	core.RequireNoError(t, err)
	core.AssertEqual(t, defaultROCmPromptLookupMaxDraft, got)
}

func TestDecodeHelpers_Bad_PromptLookupMaxDraftRejectsExplicitGemma4LimitPastWindow(t *testing.T) {
	model := &rocmModel{
		modelInfo: inference.ModelInfo{Architecture: "gemma4_text"},
		native: &hipLoadedModel{
			modelInfo:   inference.ModelInfo{Architecture: "gemma4_text", QuantBits: 6},
			contextSize: 8,
		},
	}

	_, err := rocmPromptLookupMaxDraft(model, PromptLookupDecodeConfig{MaxDraft: 6}, []int32{1, 2, 3})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "remaining model context window")

	_, err = rocmPromptLookupMaxDraft(model, PromptLookupDecodeConfig{MaxTokens: 6}, []int32{1, 2, 3})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "remaining model context window")
}

func TestDecodeHelpers_Good_DecodeMaxTokensUsesLoadedGemma4ContextWindow(t *testing.T) {
	model := &rocmModel{
		modelInfo: inference.ModelInfo{Architecture: "gemma4_text"},
		native: &hipLoadedModel{
			modelInfo:   inference.ModelInfo{Architecture: "gemma4_text", QuantBits: 6},
			contextSize: 8,
		},
	}

	got, err := rocmDecodeMaxTokens(model, "tokens:1,2,3", 0, "test")
	core.RequireNoError(t, err)
	core.AssertEqual(t, 5, got)

	got, err = rocmDecodeMaxTokens(model, "tokens:1,2,3", 2, "test")
	core.RequireNoError(t, err)
	core.AssertEqual(t, 2, got)
}

func TestDecodeHelpers_Bad_DecodeMaxTokensRejectsGemma4PastContextWindow(t *testing.T) {
	model := &rocmModel{
		modelInfo: inference.ModelInfo{Architecture: "gemma4_text"},
		native: &hipLoadedModel{
			modelInfo:   inference.ModelInfo{Architecture: "gemma4_text", QuantBits: 6},
			contextSize: 8,
		},
	}

	_, err := rocmDecodeMaxTokens(model, "tokens:1,2,3", 6, "test")

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "remaining model context window")
}

func TestDecodeHelpers_Bad_PromptLookupMaxDraftRejectsFullGemma4Prompt(t *testing.T) {
	model := &rocmModel{
		modelInfo: inference.ModelInfo{Architecture: "gemma4_text"},
		native: &hipLoadedModel{
			modelInfo:   inference.ModelInfo{Architecture: "gemma4_text", QuantBits: 6},
			contextSize: 3,
		},
	}

	_, err := rocmPromptLookupMaxDraft(model, PromptLookupDecodeConfig{}, []int32{1, 2, 3})

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "model context window")
}

func TestDecodeHelpers_Good_PromptLookupTokensUsesDecoderText(t *testing.T) {
	model := &rocmModel{native: &fakeNativeModel{}}

	tokens, err := rocmPromptLookupTokens(model, PromptLookupDecodeConfig{LookupTokens: []int32{7, 8}})

	core.RequireNoError(t, err)
	core.AssertEqual(t, int32(7), tokens[0].ID)
	core.AssertEqual(t, "1 tokens", tokens[0].Text)
	core.AssertEqual(t, int32(8), tokens[1].ID)
	core.AssertEqual(t, "1 tokens", tokens[1].Text)
}

func TestDecodeHelpers_Bad_RejectsMissingModels(t *testing.T) {
	target := &rocmModel{native: &fakeNativeModel{}}

	_, err := SpeculativeDecode(context.Background(), nil, target, SpeculativeDecodeConfig{})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "target model")

	_, err = SpeculativeDecode(context.Background(), target, nil, SpeculativeDecodeConfig{})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "draft model")

	_, err = AttachedDrafterDecode(context.Background(), nil, target, AttachedDrafterDecodeConfig{})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "target model")

	_, err = AttachedDrafterDecode(context.Background(), target, nil, AttachedDrafterDecodeConfig{})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "draft model")

	_, err = PromptLookupDecode(context.Background(), nil, PromptLookupDecodeConfig{})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "target model")
}

func TestDecodeHelpers_Bad_AttachedDrafterDecodeRejectsWrongArchitecture(t *testing.T) {
	gemma4 := &rocmModel{
		modelInfo: inference.ModelInfo{Architecture: "gemma4_text"},
		native:    &fakeNativeModel{},
	}
	assistant := &rocmModel{
		modelInfo: inference.ModelInfo{Architecture: "gemma4_assistant"},
		native:    &fakeNativeModel{},
	}
	qwen := &rocmModel{
		modelInfo: inference.ModelInfo{Architecture: "qwen3"},
		native:    &fakeNativeModel{},
	}

	_, err := AttachedDrafterDecode(context.Background(), qwen, assistant, AttachedDrafterDecodeConfig{})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "target model must be a Gemma4 text model")

	_, err = AttachedDrafterDecode(context.Background(), gemma4, qwen, AttachedDrafterDecodeConfig{})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "draft model must be a Gemma4 assistant attached MTP drafter")
}

func TestDecodeHelpers_Bad_PlanAttachedDrafterRejectsWrongArchitecture(t *testing.T) {
	gemma4 := &rocmModel{
		modelInfo: inference.ModelInfo{Architecture: "gemma4_text"},
		native:    &fakeNativeModel{},
	}
	assistant := &rocmModel{
		modelInfo: inference.ModelInfo{Architecture: "gemma4_assistant"},
		native:    &fakeNativeModel{},
	}
	qwen := &rocmModel{
		modelInfo: inference.ModelInfo{Architecture: "qwen3"},
		native:    &fakeNativeModel{},
	}

	_, err := PlanAttachedDrafter(nil, assistant)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "target model")

	_, err = PlanAttachedDrafter(gemma4, nil)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "draft model")

	_, err = PlanAttachedDrafter(qwen, assistant)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "target model must be a Gemma4 text model")

	_, err = PlanAttachedDrafter(gemma4, qwen)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "draft model must be a Gemma4 assistant attached MTP drafter")
}

func TestDecodeHelpers_Bad_PlanAttachedDrafterRejectsGemma4NonLinkedTargetPack(t *testing.T) {
	tests := []struct {
		name   string
		target *rocmModel
		want   string
	}{
		{
			name: "e4b_mxfp8_planned_only",
			target: &rocmModel{
				modelPath: "/models/lmstudio-community-gemma-4-e4b-it-mxfp8",
				modelInfo: inference.ModelInfo{
					Architecture: "gemma4_text",
					NumLayers:    26,
					HiddenSize:   2304,
					VocabSize:    ProductionMTPAssistantTokenOrderingVocabSize,
				},
			},
			want: "target Gemma4 pack is not linked for generation",
		},
		{
			name: "31b_status_only",
			target: &rocmModel{
				modelPath: "/models/lmstudio-community-gemma-4-31b-it-6bit",
				modelInfo: inference.ModelInfo{
					Architecture: "gemma4_text",
					NumLayers:    64,
					HiddenSize:   4096,
					VocabSize:    ProductionMTPAssistantTokenOrderingVocabSize,
				},
			},
			want: "target Gemma4 pack is not runnable on this card",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := PlanAttachedDrafter(tt.target, productionMTPE2BBF16AssistantModel())

			core.AssertError(t, err)
			core.AssertContains(t, err.Error(), tt.want)
		})
	}
}

func TestDecodeHelpers_Bad_PlanAttachedDrafterRejectsIncompleteGemma4PackIdentity(t *testing.T) {
	_, err := PlanAttachedDrafter(
		&rocmModel{modelInfo: inference.ModelInfo{Architecture: "gemma4_text", QuantBits: 6}},
		productionMTPE2BBF16AssistantModel(),
	)

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "target Gemma4 pack identity is incomplete")

	_, err = PlanAttachedDrafter(
		productionMTPE2BQ6TargetModel(),
		&rocmModel{modelInfo: inference.ModelInfo{Architecture: officialGemma4E2BAssistantArchitecture, QuantBits: 16}},
	)

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "draft Gemma4 assistant pack identity is incomplete")
}

func TestDecodeHelpers_Bad_PlanAttachedDrafterRejectsUnsupportedGemma4AssistantPack(t *testing.T) {
	draft := &rocmModel{
		modelPath: "/models/google-gemma-4-e2b-it-assistant-q3",
		modelInfo: inference.ModelInfo{
			Architecture: officialGemma4E2BAssistantArchitecture,
			NumLayers:    4,
			HiddenSize:   productionLaneGemma4E2BHiddenSize,
			QuantBits:    3,
			VocabSize:    ProductionMTPAssistantTokenOrderingVocabSize,
		},
		modelLabels: map[string]string{
			"gemma4_size":             "E2B",
			"gemma4_quant_mode":       "q3",
			"gemma4_generate_status":  Gemma4GenerateLoadOnly,
			"gemma4_pack_supported":   "true",
			"gemma4_runnable_on_card": "true",
		},
	}

	_, err := PlanAttachedDrafter(productionMTPE2BQ6TargetModel(), draft)

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "draft Gemma4 assistant pack is unsupported")
}

func TestDecodeHelpers_Bad_PlanAttachedDrafterRejectsMismatchedGemma4FamilyPair(t *testing.T) {
	_, err := PlanAttachedDrafter(
		productionMTP12BQ6TargetModel(),
		productionMTPE2BBF16AssistantModel(),
	)

	if err == nil {
		t.Fatal("PlanAttachedDrafter succeeded with mismatched Gemma4 target/assistant sizes")
	}
	core.AssertContains(t, err.Error(), "Gemma4 target and assistant sizes must match")
}

func TestDecodeHelpers_Bad_AttachNativeDrafterRejectsHIPMetadataMismatch(t *testing.T) {
	targetInfo := gemma4DecodeE2BQ6Info()
	draftInfo := gemma4DecodeE2BBF16AssistantInfo()
	draftInfo.HiddenSize = 2304
	target := &rocmModel{
		modelPath: "/models/lmstudio-community-gemma-4-e2b-it-6bit",
		modelInfo: targetInfo,
		native:    &hipLoadedModel{modelInfo: targetInfo},
	}
	draft := &rocmModel{
		modelPath: rocmGemma4MTPAssistantPath("E2B", "bf16"),
		modelInfo: draftInfo,
		native:    &hipLoadedModel{modelInfo: draftInfo},
	}

	_, err := AttachNativeDrafter(target, draft)

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "hidden size")
}

func TestDecodeHelpers_Bad_NewAttachedDrafterPairRejectsNonNativeNoFallback(t *testing.T) {
	target := newDecodeGemma4E2BQ6Target(nil)
	draft := newDecodeGemma4E2BBF16Assistant(nil)

	pair, err := NewAttachedDrafterPair(target, draft)

	core.AssertError(t, err)
	core.AssertNil(t, pair)
	core.AssertContains(t, err.Error(), "native ROCm target and draft models")
}

func TestDecodeHelpers_Bad_AttachedDrafterPairGenerateNativeRejectsNotReadyNoFallback(t *testing.T) {
	targetNative := &hipLoadedModel{modelInfo: gemma4DecodeE2BQ6Info()}
	draftNative := &hipLoadedModel{modelInfo: gemma4DecodeE2BBF16AssistantInfo()}
	target := newDecodeGemma4E2BQ6Target(targetNative)
	draft := newDecodeGemma4E2BBF16Assistant(draftNative)
	pair, err := NewAttachedDrafterPair(target, draft)
	core.RequireNoError(t, err)

	_, err = pair.GenerateNative(context.Background(), "prompt", AttachedDrafterGenerateConfig{MaxTokens: 4})

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "native HIP drafter generation is not linked yet")
}

func TestDecodeHelpers_Bad_AttachedDrafterPairGenerateNativeFromStateRejectsMissingStateNoFallback(t *testing.T) {
	targetNative := &readyAttachedDrafterNativeModel{
		fakeNativeModel: &fakeNativeModel{},
	}
	draftNative := &fakeNativeModel{}
	target := newDecodeGemma4E2BQ6Target(targetNative)
	draft := newDecodeGemma4E2BBF16Assistant(draftNative)
	pair, err := NewAttachedDrafterPair(target, draft)
	core.RequireNoError(t, err)

	_, err = pair.GenerateNativeFromState(context.Background(), AttachedDrafterStateGenerateRequest{Input: "new turn only"})

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "runtime-owned KV state is required")
	core.AssertEqual(t, 0, len(targetNative.attachedStateInputs))
	core.AssertEqual(t, 0, len(targetNative.attachedPrompts))
	core.AssertEqual(t, 0, len(targetNative.generatePrompts))
	core.AssertEqual(t, 0, len(draftNative.generatePrompts))
}

func TestDecodeHelpers_Bad_AttachedDrafterPairGenerateNativeRetainedRejectsMissingTargetStateNoFallback(t *testing.T) {
	targetNative := &readyAttachedDrafterNativeModel{
		fakeNativeModel: &fakeNativeModel{},
	}
	draftNative := &fakeNativeModel{}
	target := newDecodeGemma4E2BQ6Target(targetNative)
	draft := newDecodeGemma4E2BBF16Assistant(draftNative)
	pair, err := NewAttachedDrafterPair(target, draft)
	core.RequireNoError(t, err)

	_, err = pair.GenerateNativeRetained(context.Background(), "new turn only", AttachedDrafterGenerateConfig{MaxTokens: 4})

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "runtime-owned KV state is required")
	core.AssertEqual(t, 0, len(targetNative.attachedStateInputs))
	core.AssertEqual(t, 0, len(targetNative.attachedPrompts))
	core.AssertEqual(t, 0, len(targetNative.generatePrompts))
	core.AssertEqual(t, 0, len(draftNative.generatePrompts))
}

func TestDecodeHelpers_Bad_AttachedDrafterPairGenerateNativeFromStateRejectsMetadataStateNoFallback(t *testing.T) {
	targetNative := &readyAttachedDrafterNativeModel{
		fakeNativeModel: &fakeNativeModel{},
	}
	draftNative := &fakeNativeModel{}
	target := newDecodeGemma4E2BQ6Target(targetNative)
	draft := newDecodeGemma4E2BBF16Assistant(draftNative)
	pair, err := NewAttachedDrafterPair(target, draft)
	core.RequireNoError(t, err)
	state := newStateSessionWithRuntime(inference.ModelIdentity{}, inference.TokenizerIdentity{}, nil, "metadata_only")

	_, err = pair.GenerateNativeFromState(context.Background(), AttachedDrafterStateGenerateRequest{State: state, Input: "new turn only"})

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "refusing prompt replay")
	core.AssertEqual(t, 0, len(targetNative.attachedStateInputs))
	core.AssertEqual(t, 0, len(targetNative.attachedPrompts))
	core.AssertEqual(t, 0, len(targetNative.generatePrompts))
	core.AssertEqual(t, 0, len(draftNative.generatePrompts))
}

func TestDecodeHelpers_Bad_AttachedDrafterPairGenerateNativeFromStateRejectsEmptyGemma4Q4StateNoFallback(t *testing.T) {
	targetNative := &readyAttachedDrafterNativeModel{
		fakeNativeModel: &fakeNativeModel{},
	}
	draftNative := &fakeNativeModel{}
	target := newDecodeGemma4E2BQ6Target(targetNative)
	draft := newDecodeGemma4E2BBF16Assistant(draftNative)
	pair, err := NewAttachedDrafterPair(target, draft)
	core.RequireNoError(t, err)
	state := newStateSessionWithRuntime(inference.ModelIdentity{}, inference.TokenizerIdentity{}, nil, &hipGemma4Q4DeviceDecodeState{
		layers: []hipGemma4Q4DeviceLayerKVState{{cache: &rocmDeviceKVCache{}}},
	})

	_, err = pair.GenerateNativeFromState(context.Background(), AttachedDrafterStateGenerateRequest{State: state, Input: "new turn only"})

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "refusing prompt replay")
	core.AssertEqual(t, 0, len(targetNative.attachedStateInputs))
	core.AssertEqual(t, 0, len(targetNative.attachedPrompts))
	core.AssertEqual(t, 0, len(targetNative.generatePrompts))
	core.AssertEqual(t, 0, len(draftNative.generatePrompts))
}

func TestDecodeHelpers_Bad_AttachedDrafterPairGenerateNativeFromStateRejectsPastContextWindow(t *testing.T) {
	targetNative := &readyAttachedDrafterNativeModel{
		fakeNativeModel: &fakeNativeModel{encodeResult: []int32{1, 2, 3}},
	}
	draftNative := &fakeNativeModel{}
	target := newDecodeGemma4E2BQ6Target(targetNative)
	draft := newDecodeGemma4E2BBF16Assistant(draftNative)
	pair, err := NewAttachedDrafterPair(target, draft)
	core.RequireNoError(t, err)
	cache, err := newROCmKVCache(rocmKVCacheModeQ8, 2)
	core.RequireNoError(t, err)
	core.RequireNoError(t, cache.AppendVectors(0, 2, 2, make([]float32, (defaultContextLengthCap-4)*2), make([]float32, (defaultContextLengthCap-4)*2)))
	state := newStateSessionWithRuntime(inference.ModelIdentity{}, inference.TokenizerIdentity{}, nil, cache)

	_, err = pair.GenerateNativeFromState(context.Background(), AttachedDrafterStateGenerateRequest{
		State:     state,
		Input:     "new turn only",
		MaxTokens: 2,
	})

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "remaining model context window")
	core.AssertEqual(t, 0, len(targetNative.attachedStateInputs))
}

func TestDecodeHelpers_Bad_AttachedDrafterPairGenerateNativeFromStateRejectsWrongModelStateNoFallback(t *testing.T) {
	targetNative := &readyAttachedDrafterNativeModel{
		fakeNativeModel: &fakeNativeModel{},
	}
	draftNative := &fakeNativeModel{}
	target := newDecodeGemma4E2BQ6Target(targetNative)
	draft := newDecodeGemma4E2BBF16Assistant(draftNative)
	pair, err := NewAttachedDrafterPair(target, draft)
	core.RequireNoError(t, err)
	cache, err := newROCmKVCache(rocmKVCacheModeQ8, 2)
	core.RequireNoError(t, err)
	core.RequireNoError(t, cache.AppendVectors(0, 2, 2, []float32{1, 2}, []float32{3, 4}))
	state := newStateSessionWithRuntime(inference.ModelIdentity{Architecture: "qwen3"}, inference.TokenizerIdentity{}, nil, cache)

	_, err = pair.GenerateNativeFromState(context.Background(), AttachedDrafterStateGenerateRequest{State: state, Input: "new turn only"})

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "model architecture mismatch")
	core.AssertEqual(t, 0, len(targetNative.attachedStateInputs))
	core.AssertEqual(t, 0, len(targetNative.attachedPrompts))
	core.AssertEqual(t, 0, len(targetNative.generatePrompts))
	core.AssertEqual(t, 0, len(draftNative.generatePrompts))
}

func TestDecodeHelpers_Bad_AttachedDrafterPairGenerateNativeRejectsNilPair(t *testing.T) {
	var pair *AttachedDrafterPair

	_, err := pair.GenerateNative(context.Background(), "prompt", AttachedDrafterGenerateConfig{})

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "pair is required")
}

func TestDecodeHelpers_Bad_LoadAttachedDrafterPairRejectsEmptyPaths(t *testing.T) {
	_, err := LoadAttachedDrafterPair("", "assistant", AttachedDrafterPairConfig{})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "target path")

	_, err = LoadAttachedDrafterPair("target", " ", AttachedDrafterPairConfig{})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "draft path")
}

func TestDecodeHelpers_Good_LoadAttachedDrafterPairForwardsROCmLoadConfig(t *testing.T) {
	runtime := &fakeNativeRuntime{available: true}
	_, err := newROCmBackendWithRuntime(runtime).LoadAttachedDrafterPair(
		writeGemma4ModelPackGGUF(t),
		writeGemma4ModelPackGGUF(t),
		AttachedDrafterPairConfig{
			TargetROCmConfig: ROCmLoadConfig{CacheMode: "q8"},
			DraftROCmConfig:  ROCmLoadConfig{DeviceKVMode: "k-q8-v-q4"},
		},
	)

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "validate pair")
	core.AssertEqual(t, 2, len(runtime.loadConfigs))
	core.AssertEqual(t, rocmKVCacheModeQ8, runtime.loadConfigs[0].DeviceKVMode)
	core.AssertEqual(t, rocmKVCacheModeQ8, runtime.loadConfigs[0].ModelLabels["kv_cache_mode"])
	core.AssertEqual(t, rocmKVCacheModeKQ8VQ4, runtime.loadConfigs[1].DeviceKVMode)
	core.AssertEqual(t, rocmKVCacheModeKQ8VQ4, runtime.loadConfigs[1].ModelLabels["kv_cache_mode"])
}

func TestDecodeHelpers_Bad_PromptLookupRequiresTokensWithoutEncoder(t *testing.T) {
	model := &minimalDecodeTextModel{}

	_, err := PromptLookupDecode(context.Background(), model, PromptLookupDecodeConfig{Prompt: "1 2 1"})

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "lookup tokens")
}

func TestDecodeHelpers_Ugly_PropagatesModelStreamError(t *testing.T) {
	target := &rocmModel{native: &decodeErrorNativeModel{fakeNativeModel: &fakeNativeModel{}, err: core.NewError("decode failed")}}
	draft := &rocmModel{native: &fakeNativeModel{tokens: []inference.Token{{ID: 1, Text: "a"}}}}

	_, err := SpeculativeDecode(context.Background(), target, draft, SpeculativeDecodeConfig{Prompt: "p", MaxTokens: 1})

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "model generation failed")
	core.AssertContains(t, err.Error(), "decode failed")
}

func TestDecodeHelpers_Good_ModelIdentityReporterDrivesDecodeIdentity(t *testing.T) {
	targetIdentity := officialGemma4E2BQ6TargetIdentity()
	targetIdentity.ContextLength = 8192
	targetIdentity.Labels = map[string]string{
		"gemma4_size":       "E2B",
		"gemma4_quant_mode": "q6",
	}
	model := &decodeIdentityReporterModel{identity: targetIdentity}

	identity := rocmDecodeModelIdentity(model)
	if identity.Path != targetIdentity.Path ||
		identity.Architecture != "gemma4_text" ||
		identity.ContextLength != 8192 ||
		identity.Labels["gemma4_size"] != "E2B" ||
		!rocmDecodeIsGemma4Target(model) {
		t.Fatalf("rocmDecodeModelIdentity(identity reporter) = %+v, want loaded Gemma4 target identity", identity)
	}
	identity.Labels["gemma4_size"] = "mutated"
	if next := rocmDecodeModelIdentity(model); next.Labels["gemma4_size"] == "mutated" {
		t.Fatalf("rocmDecodeModelIdentity returned aliased reporter labels: %+v", next.Labels)
	}
}

func TestDecodeHelpers_Good_AttachedDrafterPlanUsesModelReporters(t *testing.T) {
	targetIdentity := officialGemma4E2BQ6TargetIdentity()
	targetIdentity.ContextLength = 8192
	assistantIdentity := officialGemma4E2BBF16AssistantIdentity()
	assistantIdentity.Architecture = ""
	draft := &decodeProfileReporterModel{profile: ROCmModelProfile{
		Name:         "gemma4",
		Family:       "gemma4",
		Architecture: officialGemma4E2BAssistantArchitecture,
		Model:        assistantIdentity,
	}}

	plan, err := PlanAttachedDrafter(&decodeIdentityReporterModel{identity: targetIdentity}, draft)

	if err != nil {
		t.Fatalf("PlanAttachedDrafter with model reporters: %v", err)
	}
	if plan.Target.Architecture != "gemma4_text" ||
		plan.Target.QuantBits != 6 ||
		plan.Draft.Architecture != officialGemma4E2BAssistantArchitecture ||
		plan.Draft.QuantBits != 16 ||
		plan.Labels["attached_drafter_target_gemma4_size"] != "E2B" ||
		plan.Labels["attached_drafter_target_gemma4_quant_mode"] != "q6" ||
		plan.Labels["attached_drafter_assistant_gemma4_size"] != "E2B" ||
		plan.Labels["attached_drafter_assistant_gemma4_quant_mode"] != "bf16" ||
		plan.Labels["attached_drafter_gemma4_family_pair_verified"] != "true" {
		t.Fatalf("PlanAttachedDrafter = %+v labels=%+v, want reporter-declared Gemma4 MTP pair", plan, plan.Labels)
	}
	if !rocmDecodeIsGemma4AssistantDrafter(draft) {
		t.Fatalf("profile-reporter draft was not recognised as Gemma4 assistant")
	}
}

func BenchmarkDecodeHelpers_AttachedDrafterDecode_Gemma4Assistant(b *testing.B) {
	target := &benchmarkDecodeTextModel{
		architecture: "gemma4_text",
		info:         gemma4DecodeE2BQ6Info(),
		tokens:       []inference.Token{{ID: 4, Text: "a"}, {ID: 5, Text: "b"}, {ID: 7, Text: "c"}},
	}
	draft := &benchmarkDecodeTextModel{
		architecture: "gemma4_assistant",
		info:         gemma4DecodeE2BBF16AssistantInfo(),
		tokens:       []inference.Token{{ID: 4, Text: "a"}, {ID: 9, Text: "x"}, {ID: 8, Text: "y"}},
	}
	cfg := AttachedDrafterDecodeConfig{Prompt: "p", MaxTokens: 3, DraftTokens: 3}
	ctx := context.Background()

	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		result, err := AttachedDrafterDecode(ctx, target, draft, cfg)
		if err != nil {
			b.Fatal(err)
		}
		if result.Metrics.EmittedTokens != 3 {
			b.Fatalf("emitted tokens = %d, want 3", result.Metrics.EmittedTokens)
		}
	}
}

func BenchmarkDecodeHelpers_PlanAttachedDrafter_Gemma4Assistant(b *testing.B) {
	target := &benchmarkDecodeTextModel{architecture: "gemma4_text", info: gemma4DecodeE2BQ6Info()}
	draft := &benchmarkDecodeTextModel{architecture: "gemma4_assistant", info: gemma4DecodeE2BBF16AssistantInfo()}

	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		plan, err := PlanAttachedDrafter(target, draft)
		if err != nil {
			b.Fatal(err)
		}
		if plan.NativeAttachment != hipKernelStatusNotLinked {
			b.Fatalf("native attachment = %q, want %q", plan.NativeAttachment, hipKernelStatusNotLinked)
		}
	}
}

func BenchmarkDecodeHelpers_AttachNativeDrafter_HIPNotLinked(b *testing.B) {
	target := newDecodeGemma4E2BQ6Target(&hipLoadedModel{modelInfo: gemma4DecodeE2BQ6Info()})
	draft := newDecodeGemma4E2BBF16Assistant(&hipLoadedModel{modelInfo: gemma4DecodeE2BBF16AssistantInfo()})

	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		attachment, err := AttachNativeDrafter(target, draft)
		if err == nil {
			b.Fatal("AttachNativeDrafter succeeded before native HIP attachment was linked")
		}
		if attachment.NativeAttachment != hipKernelStatusNotLinked {
			b.Fatalf("native attachment = %q, want %q", attachment.NativeAttachment, hipKernelStatusNotLinked)
		}
	}
}

func BenchmarkDecodeHelpers_NewAttachedDrafterPair_HIPNotReady(b *testing.B) {
	target := newDecodeGemma4E2BQ6Target(&hipLoadedModel{modelInfo: gemma4DecodeE2BQ6Info()})
	draft := newDecodeGemma4E2BBF16Assistant(&hipLoadedModel{modelInfo: gemma4DecodeE2BBF16AssistantInfo()})

	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		pair, err := NewAttachedDrafterPair(target, draft)
		if err != nil {
			b.Fatal(err)
		}
		if pair.NativeReady() {
			b.Fatal("pair reported native ready before HIP attachment was linked")
		}
	}
}

func BenchmarkDecodeHelpers_AttachedDrafterPairGenerateNative_NotReady(b *testing.B) {
	target := newDecodeGemma4E2BQ6Target(&hipLoadedModel{modelInfo: gemma4DecodeE2BQ6Info()})
	draft := newDecodeGemma4E2BBF16Assistant(&hipLoadedModel{modelInfo: gemma4DecodeE2BBF16AssistantInfo()})
	pair, err := NewAttachedDrafterPair(target, draft)
	if err != nil {
		b.Fatal(err)
	}

	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_, err := pair.GenerateNative(context.Background(), "prompt", AttachedDrafterGenerateConfig{MaxTokens: 4})
		if err == nil {
			b.Fatal("GenerateNative succeeded before native HIP generation was linked")
		}
	}
}

func BenchmarkDecodeHelpers_AttachedDrafterPairGenerateNativeFromState_MissingState(b *testing.B) {
	targetNative := &readyAttachedDrafterNativeModel{
		fakeNativeModel: &fakeNativeModel{},
	}
	draft := newDecodeGemma4E2BBF16Assistant(&fakeNativeModel{})
	target := newDecodeGemma4E2BQ6Target(targetNative)
	pair, err := NewAttachedDrafterPair(target, draft)
	if err != nil {
		b.Fatal(err)
	}
	req := AttachedDrafterStateGenerateRequest{Input: "new turn only", MaxTokens: 4}

	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_, err := pair.GenerateNativeFromState(context.Background(), req)
		if err == nil {
			b.Fatal("GenerateNativeFromState succeeded without runtime-owned KV state")
		}
	}
}

func BenchmarkDecodeHelpers_AttachedDrafterPairGenerateNativeRetained_MissingTargetState(b *testing.B) {
	targetNative := &readyAttachedDrafterNativeModel{
		fakeNativeModel: &fakeNativeModel{},
	}
	draft := newDecodeGemma4E2BBF16Assistant(&fakeNativeModel{})
	target := newDecodeGemma4E2BQ6Target(targetNative)
	pair, err := NewAttachedDrafterPair(target, draft)
	if err != nil {
		b.Fatal(err)
	}
	cfg := AttachedDrafterGenerateConfig{MaxTokens: 4}

	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_, err := pair.GenerateNativeRetained(context.Background(), "new turn only", cfg)
		if err == nil {
			b.Fatal("GenerateNativeRetained succeeded without target runtime-owned KV state")
		}
	}
}

func BenchmarkDecodeHelpers_AttachedDrafterPairGenerateNativeRetained_ReadyState(b *testing.B) {
	targetNative := &readyAttachedDrafterNativeModel{
		fakeNativeModel: &fakeNativeModel{},
	}
	draftNative := &fakeNativeModel{}
	cache, err := newROCmKVCache(rocmKVCacheModeQ8, 2)
	if err != nil {
		b.Fatal(err)
	}
	if err := cache.AppendVectors(0, 2, 2, []float32{1, 2}, []float32{3, 4}); err != nil {
		b.Fatal(err)
	}
	target := newDecodeGemma4E2BQ6Target(targetNative)
	target.state = newStateSessionWithRuntime(inference.ModelIdentity{}, inference.TokenizerIdentity{}, nil, cache)
	draft := newDecodeGemma4E2BBF16Assistant(draftNative)
	pair, err := NewAttachedDrafterPair(target, draft)
	if err != nil {
		b.Fatal(err)
	}
	cfg := AttachedDrafterGenerateConfig{MaxTokens: 4}

	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		result, err := pair.GenerateNativeRetained(context.Background(), "new turn only", cfg)
		if err != nil {
			b.Fatal(err)
		}
		if result.Metrics.EmittedTokens != cfg.MaxTokens {
			b.Fatalf("emitted tokens = %d, want %d", result.Metrics.EmittedTokens, cfg.MaxTokens)
		}
	}
}

func BenchmarkDecodeHelpers_AttachedDrafterPairGenerateNativeRetained_ProductionHandoff(b *testing.B) {
	targetNative := &benchmarkAttachedDrafterStateNativeModel{
		fakeNativeModel: &fakeNativeModel{},
		result: inferdecode.Result{
			Mode:   inferdecode.ModeSpeculative,
			Tokens: []inferdecode.Token{{ID: 1, Text: "ok"}},
			Text:   "ok",
		},
	}
	draftNative := &fakeNativeModel{}
	cache, err := newROCmKVCache(rocmKVCacheModeQ8, 2)
	if err != nil {
		b.Fatal(err)
	}
	if err := cache.AppendVectors(0, 2, 2, []float32{1, 2}, []float32{3, 4}); err != nil {
		b.Fatal(err)
	}
	target := newDecodeGemma4E2BQ6Target(targetNative)
	target.state = newStateSessionWithRuntime(inference.ModelIdentity{}, inference.TokenizerIdentity{}, nil, cache)
	draft := newDecodeGemma4E2BBF16Assistant(draftNative)
	pair, err := NewAttachedDrafterPair(target, draft)
	if err != nil {
		b.Fatal(err)
	}
	cfg := AttachedDrafterGenerateConfig{MaxTokens: 4}

	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		result, err := pair.GenerateNativeRetained(context.Background(), "new turn only", cfg)
		if err != nil {
			b.Fatal(err)
		}
		if result.Metrics.EmittedTokens != cfg.MaxTokens {
			b.Fatalf("emitted tokens = %d, want %d", result.Metrics.EmittedTokens, cfg.MaxTokens)
		}
	}
}

type decodeErrorNativeModel struct {
	*fakeNativeModel
	err error
}

func (model *decodeErrorNativeModel) Generate(context.Context, string, inference.GenerateConfig) (iter.Seq[inference.Token], func() error) {
	return func(yield func(inference.Token) bool) {
		yield(inference.Token{ID: 1, Text: "a"})
	}, func() error { return model.err }
}

type minimalDecodeTextModel struct{}

func (*minimalDecodeTextModel) Generate(context.Context, string, ...inference.GenerateOption) iter.Seq[inference.Token] {
	return func(func(inference.Token) bool) {}
}

func (*minimalDecodeTextModel) Chat(context.Context, []inference.Message, ...inference.GenerateOption) iter.Seq[inference.Token] {
	return func(func(inference.Token) bool) {}
}

func (*minimalDecodeTextModel) Classify(context.Context, []string, ...inference.GenerateOption) core.Result {
	return core.Ok([]inference.ClassifyResult(nil))
}

func (*minimalDecodeTextModel) BatchGenerate(context.Context, []string, ...inference.GenerateOption) core.Result {
	return core.Ok([]inference.BatchResult(nil))
}

func (*minimalDecodeTextModel) Metrics() inference.GenerateMetrics {
	return inference.GenerateMetrics{}
}

func (*minimalDecodeTextModel) Err() core.Result { return core.Ok(nil) }

func (*minimalDecodeTextModel) ModelType() string { return "minimal" }

func (*minimalDecodeTextModel) Info() inference.ModelInfo { return inference.ModelInfo{} }

func (*minimalDecodeTextModel) Close() core.Result { return core.Ok(nil) }

type decodeIdentityReporterModel struct {
	minimalDecodeTextModel
	identity inference.ModelIdentity
}

func (model *decodeIdentityReporterModel) ModelIdentity() inference.ModelIdentity {
	return model.identity
}

type decodeProfileReporterModel struct {
	minimalDecodeTextModel
	profile ROCmModelProfile
}

func (model *decodeProfileReporterModel) ModelProfile() ROCmModelProfile {
	return model.profile
}

type benchmarkDecodeTextModel struct {
	architecture  string
	info          inference.ModelInfo
	tokens        []inference.Token
	err           error
	generateCalls int
}

func (model *benchmarkDecodeTextModel) Generate(context.Context, string, ...inference.GenerateOption) iter.Seq[inference.Token] {
	model.generateCalls++
	return func(yield func(inference.Token) bool) {
		for _, token := range model.tokens {
			if !yield(token) {
				return
			}
		}
	}
}

func (*benchmarkDecodeTextModel) Chat(context.Context, []inference.Message, ...inference.GenerateOption) iter.Seq[inference.Token] {
	return func(func(inference.Token) bool) {}
}

func (*benchmarkDecodeTextModel) Classify(context.Context, []string, ...inference.GenerateOption) core.Result {
	return core.Ok([]inference.ClassifyResult(nil))
}

func (*benchmarkDecodeTextModel) BatchGenerate(context.Context, []string, ...inference.GenerateOption) core.Result {
	return core.Ok([]inference.BatchResult(nil))
}

func (*benchmarkDecodeTextModel) Metrics() inference.GenerateMetrics {
	return inference.GenerateMetrics{}
}

func (model *benchmarkDecodeTextModel) Err() core.Result { return core.ResultOf(nil, model.err) }

func (model *benchmarkDecodeTextModel) ModelType() string { return model.architecture }

func (model *benchmarkDecodeTextModel) Info() inference.ModelInfo {
	info := model.info
	if info.Architecture == "" {
		info.Architecture = model.architecture
	}
	return info
}

func (*benchmarkDecodeTextModel) Close() core.Result { return core.Ok(nil) }

func gemma4DecodeE2BQ6Info() inference.ModelInfo {
	return inference.ModelInfo{
		Architecture: "gemma4_text",
		NumLayers:    productionLaneGemma4E2BLayers,
		HiddenSize:   productionLaneGemma4E2BHiddenSize,
		VocabSize:    ProductionMTPAssistantTokenOrderingVocabSize,
		QuantBits:    6,
	}
}

func gemma4DecodeE2BBF16AssistantInfo() inference.ModelInfo {
	return inference.ModelInfo{
		Architecture: officialGemma4E2BAssistantArchitecture,
		NumLayers:    4,
		HiddenSize:   productionLaneGemma4E2BHiddenSize,
		VocabSize:    ProductionMTPAssistantTokenOrderingVocabSize,
		QuantBits:    16,
	}
}

func gemma4DecodeE2BAssistantVerifierTensors() map[string]hipTensor {
	return gemma4DecodeE2BAssistantVerifierTensorsForQuant("bf16")
}

func gemma4DecodeE2BAssistantVerifierTensorsForQuant(mode string) map[string]hipTensor {
	tensors := map[string]hipTensor{}
	quantized := hipAttachedDrafterAssistantQuantModeRequiresAffine(mode)
	quantBits := 16
	if quantized {
		bits, ok := hipAttachedDrafterAssistantVerifierQuantBits(mode)
		if !ok {
			panic("unsupported assistant verifier fixture quant mode: " + mode)
		}
		quantBits = bits
	}
	tensorBytes := func(elementBytes uint64, dims ...uint64) uint64 {
		count := uint64(1)
		for _, dim := range dims {
			count *= dim
		}
		return count * elementBytes
	}
	add := func(name string, tensorType uint32, typeName string, byteSize uint64, dims ...uint64) {
		tensors[name] = hipTensor{
			info: nativeTensorInfo{
				Name:       name,
				Type:       tensorType,
				TypeName:   typeName,
				Dimensions: append([]uint64(nil), dims...),
				ByteSize:   byteSize,
			},
			pointer: 1,
		}
	}
	addBF16 := func(name string, dims ...uint64) {
		add(name, 30, "BF16", tensorBytes(2, dims...), dims...)
	}
	addU32 := func(name string, dims ...uint64) {
		add(name, 26, "U32", tensorBytes(4, dims...), dims...)
	}
	addLinear := func(baseName string, rows, cols uint64) {
		if !quantized {
			addBF16(baseName+".weight", rows, cols)
			return
		}
		packedCols, err := hipMLXAffinePackedCols(int(cols), quantBits)
		if err != nil {
			panic(err)
		}
		groups := cols / 64
		addU32(baseName+".weight", rows, uint64(packedCols))
		addBF16(baseName+".scales", rows, groups)
		addBF16(baseName+".biases", rows, groups)
	}
	hidden := uint64(productionLaneGemma4E2BHiddenSize)
	vocab := uint64(ProductionMTPAssistantTokenOrderingVocabSize)
	addLinear("model.embed_tokens", vocab, hidden)
	addBF16("model.norm.weight", hidden)
	addLinear("pre_projection", hidden, hidden*2)
	addLinear("post_projection", hidden, hidden)
	addLinear("masked_embedding.centroids", 2048, hidden)
	add("masked_embedding.token_ordering", 27, "I64", 2048*128*8, 2048, 128)
	for layer := 0; layer < 4; layer++ {
		prefix := core.Sprintf("model.layers.%d", layer)
		addBF16(prefix+".input_layernorm.weight", hidden)
		addBF16(prefix+".post_attention_layernorm.weight", hidden)
		addBF16(prefix+".pre_feedforward_layernorm.weight", hidden)
		addBF16(prefix+".post_feedforward_layernorm.weight", hidden)
		addBF16(prefix+".layer_scalar", 1)
		addLinear(prefix+".self_attn.q_proj", hidden, hidden)
		addLinear(prefix+".self_attn.o_proj", hidden, hidden)
		addBF16(prefix+".self_attn.q_norm.weight", hidden)
		addLinear(prefix+".mlp.gate_proj", hidden, hidden)
		addLinear(prefix+".mlp.up_proj", hidden, hidden)
		addLinear(prefix+".mlp.down_proj", hidden, hidden)
	}
	return tensors
}

func newDecodeGemma4E2BQ6Target(native nativeModel) *rocmModel {
	return &rocmModel{
		modelPath: "/models/lmstudio-community-gemma-4-e2b-it-6bit",
		modelInfo: gemma4DecodeE2BQ6Info(),
		native:    native,
	}
}

func newDecodeGemma4E2BBF16Assistant(native nativeModel) *rocmModel {
	return &rocmModel{
		modelPath: rocmGemma4MTPAssistantPath("E2B", "bf16"),
		modelInfo: gemma4DecodeE2BBF16AssistantInfo(),
		native:    native,
	}
}

func newPendingTargetRetainedAttachedDrafterTextModel(t *testing.T, target, draft *rocmModel, draftTokens int) *attachedDrafterTextModel {
	t.Helper()
	plan, err := PlanAttachedDrafter(target, draft)
	core.RequireNoError(t, err)
	labels := cloneStringMap(plan.Labels)
	labels["attached_drafter_native_attachment"] = hipKernelStatusNotLinked
	labels["attached_drafter_native_handoff"] = attachedDrafterNativeHandoffTargetDecodeOnly
	labels["attached_drafter_prompt_replay_fallback"] = "forbidden"
	labels["attached_drafter_retained_state_entrypoint"] = hipKernelStatusLinked
	labels["attached_drafter_retained_state_required"] = "true"
	labels["attached_drafter_state_source"] = "rocm_state_session_runtime_kv"
	labels["attached_drafter_target_retained_decode"] = hipKernelStatusLinked
	labels["attached_drafter_target_retained_state_decode"] = hipKernelStatusLinked
	labels["attached_drafter_assistant_verify"] = hipKernelStatusNotLinked
	labels["attached_drafter_assistant_state_verify"] = hipKernelStatusNotLinked
	return &attachedDrafterTextModel{
		pair: &AttachedDrafterPair{
			Target: target,
			Draft:  draft,
			Plan:   plan,
			Attachment: AttachedDrafterAttachment{
				Plan:             plan,
				Target:           plan.Target,
				Draft:            plan.Draft,
				NativeAttachment: hipKernelStatusNotLinked,
				Labels:           labels,
			},
			NativeError: "native HIP drafter attachment is not linked yet",
		},
		draftTokens: draftTokens,
	}
}

type readyAttachedDrafterNativeModel struct {
	*fakeNativeModel
	attachedPrompts               []string
	attachedConfigs               []AttachedDrafterGenerateConfig
	attachedRetainedPrompts       []string
	attachedRetainedConfigs       []AttachedDrafterGenerateConfig
	attachedRetainedStates        []*StateSession
	attachedStateInputs           []string
	attachedStateAttachmentLabels []string
	attachedStateRequests         []AttachedDrafterStateGenerateRequest
}

func (model *readyAttachedDrafterNativeModel) AttachAttachedDrafter(draft nativeModel, plan AttachedDrafterPlan) (AttachedDrafterAttachment, error) {
	return AttachedDrafterAttachment{
		Plan:             plan,
		Target:           inference.ModelInfo{Architecture: "gemma4_text", HiddenSize: 8, VocabSize: 16},
		Draft:            inference.ModelInfo{Architecture: "gemma4_assistant", HiddenSize: 8, VocabSize: 16},
		NativeAttachment: hipKernelStatusLinked,
		Labels: map[string]string{
			"attached_drafter_native_attachment": hipKernelStatusLinked,
			"attached_drafter_runtime":           "hip",
		},
	}, nil
}

func (model *readyAttachedDrafterNativeModel) GenerateAttachedDrafter(_ context.Context, attachment AttachedDrafterAttachment, prompt string, cfg AttachedDrafterGenerateConfig) (inferdecode.Result, error) {
	model.attachedPrompts = append(model.attachedPrompts, prompt)
	model.attachedConfigs = append(model.attachedConfigs, cfg)
	return inferdecode.Result{
		Mode:   inferdecode.ModeSpeculative,
		Prompt: prompt,
		Metrics: inferdecode.Metrics{
			DraftTokens:    cfg.DraftTokens,
			AcceptedTokens: 2,
			RejectedTokens: 1,
			EmittedTokens:  cfg.MaxTokens,
			TargetCalls:    3,
			DraftCalls:     1,
			AcceptanceRate: float64(2) / 3,
		},
		Tokens: []inferdecode.Token{{ID: 1, Text: "ok"}},
		Text:   "ok",
	}, nil
}

func (model *readyAttachedDrafterNativeModel) GenerateAttachedDrafterWithStateRetention(_ context.Context, attachment AttachedDrafterAttachment, prompt string, cfg AttachedDrafterGenerateConfig, state *StateSession) (inferdecode.Result, error) {
	model.attachedRetainedPrompts = append(model.attachedRetainedPrompts, prompt)
	model.attachedRetainedConfigs = append(model.attachedRetainedConfigs, cfg)
	model.attachedRetainedStates = append(model.attachedRetainedStates, state)
	if state != nil && !state.hasRuntimeOwnedKV() {
		_ = state.replaceRuntime(&hipGemma4Q4HostDecodeStateRuntime{tokenCount: 3})
	}
	return inferdecode.Result{
		Mode:   inferdecode.ModeSpeculative,
		Prompt: prompt,
		Metrics: inferdecode.Metrics{
			DraftTokens:    cfg.DraftTokens,
			AcceptedTokens: 2,
			RejectedTokens: 1,
			EmittedTokens:  cfg.MaxTokens,
			TargetCalls:    3,
			DraftCalls:     1,
			AcceptanceRate: float64(2) / 3,
		},
		Tokens: []inferdecode.Token{{ID: 1, Text: "ok"}},
		Text:   "ok",
	}, nil
}

func (model *readyAttachedDrafterNativeModel) GenerateAttachedDrafterFromState(_ context.Context, attachment AttachedDrafterAttachment, req AttachedDrafterStateGenerateRequest) (inferdecode.Result, error) {
	model.attachedStateInputs = append(model.attachedStateInputs, req.Input)
	model.attachedStateAttachmentLabels = append(model.attachedStateAttachmentLabels, attachment.Labels["attached_drafter_native_attachment"])
	model.attachedStateRequests = append(model.attachedStateRequests, req)
	return inferdecode.Result{
		Mode:   inferdecode.ModeSpeculative,
		Prompt: req.Input,
		Metrics: inferdecode.Metrics{
			DraftTokens:    req.DraftTokens,
			AcceptedTokens: 2,
			RejectedTokens: 1,
			EmittedTokens:  req.MaxTokens,
			TargetCalls:    3,
			DraftCalls:     1,
			AcceptanceRate: float64(2) / 3,
		},
		Tokens: []inferdecode.Token{{ID: 1, Text: "ok"}},
		Text:   "ok",
	}, nil
}

type benchmarkAttachedDrafterStateNativeModel struct {
	*fakeNativeModel
	result inferdecode.Result
}

func (model *benchmarkAttachedDrafterStateNativeModel) AttachAttachedDrafter(draft nativeModel, plan AttachedDrafterPlan) (AttachedDrafterAttachment, error) {
	return AttachedDrafterAttachment{
		Plan:             plan,
		Target:           inference.ModelInfo{Architecture: "gemma4_text", HiddenSize: 8, VocabSize: 16},
		Draft:            inference.ModelInfo{Architecture: "gemma4_assistant", HiddenSize: 8, VocabSize: 16},
		NativeAttachment: hipKernelStatusLinked,
		Labels: map[string]string{
			"attached_drafter_native_attachment": hipKernelStatusLinked,
			"attached_drafter_runtime":           "hip",
		},
	}, nil
}

func (model *benchmarkAttachedDrafterStateNativeModel) GenerateAttachedDrafterFromState(_ context.Context, _ AttachedDrafterAttachment, req AttachedDrafterStateGenerateRequest) (inferdecode.Result, error) {
	result := model.result
	result.Prompt = req.Input
	result.Metrics.DraftTokens = req.DraftTokens
	result.Metrics.EmittedTokens = req.MaxTokens
	return result, nil
}
