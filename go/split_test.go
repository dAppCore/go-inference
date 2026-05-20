// SPDX-Licence-Identifier: EUPL-1.2

package inference

import "testing"

func TestPlanModelSlice_ClientPreset_Good(t *testing.T) {
	plan, err := PlanModelSlice(ModelSliceRequest{
		Preset:     ModelSlicePresetClient,
		Model:      ModelIdentity{Path: "/models/gemma4", Architecture: "gemma4", NumLayers: 34, QuantBits: 4},
		OutputPath: "/tmp/gemma4-client",
	})

	checkNoError(t, err)
	checkEqual(t, ModelSlicePresetClient, plan.Preset)
	checkEqual(t, ModelExtractLevelAttention, plan.ExtractLevel)
	checkTrue(t, plan.HasComponent(ModelComponentEmbeddings))
	checkTrue(t, plan.HasComponent(ModelComponentAttention))
	checkTrue(t, plan.HasComponent(ModelComponentTokenizer))
	checkFalse(t, plan.HasComponent(ModelComponentFFN))
	checkTrue(t, plan.AttentionLocal)
	checkTrue(t, plan.FFNRemoteCandidate)
	checkEqual(t, "/models/gemma4", plan.SourcePath)
	checkEqual(t, "/tmp/gemma4-client", plan.OutputPath)
}

func TestPlanModelSlice_AttentionPreset_Good(t *testing.T) {
	plan, err := PlanModelSlice(ModelSliceRequest{Preset: ModelSlicePresetAttention})

	checkNoError(t, err)
	checkEqual(t, ModelExtractLevelAttention, plan.ExtractLevel)
	checkElementsMatch(t, []ModelComponent{
		ModelComponentManifest,
		ModelComponentNorms,
		ModelComponentAttention,
		ModelComponentLabels,
	}, plan.Components)
}

func TestPlanModelSlice_ServerPreset_Good(t *testing.T) {
	plan, err := PlanModelSlice(ModelSliceRequest{Preset: ModelSlicePresetServer})

	checkNoError(t, err)
	checkEqual(t, ModelExtractLevelInference, plan.ExtractLevel)
	checkTrue(t, plan.HasComponent(ModelComponentFFN))
	checkTrue(t, plan.HasComponent(ModelComponentEmbeddings))
	checkFalse(t, plan.HasComponent(ModelComponentAttention))
	checkFalse(t, plan.AttentionLocal)
}

func TestPlanModelSlice_CustomPreset_UglyCopiesInput(t *testing.T) {
	components := []ModelComponent{ModelComponentTokenizer, ModelComponentAttention}
	labels := map[string]string{"origin": "larql"}
	plan, err := PlanModelSlice(ModelSliceRequest{
		Components: components,
		Labels:     labels,
	})
	checkNoError(t, err)

	components[0] = ModelComponentFFN
	labels["origin"] = "mutated"

	checkEqual(t, ModelSlicePresetCustom, plan.Preset)
	checkEqual(t, ModelComponentTokenizer, plan.Components[0])
	checkEqual(t, "larql", plan.Labels["origin"])
}

func TestPlanModelSlice_UnknownPreset_Bad(t *testing.T) {
	_, err := PlanModelSlice(ModelSliceRequest{Preset: ModelSlicePreset("sideways")})

	checkError(t, err)
	checkContains(t, err.Error(), "unknown slice preset")
}

func TestValidateSplitInferencePlan_RemoteFFN_Good(t *testing.T) {
	local, err := PlanModelSlice(ModelSliceRequest{Preset: ModelSlicePresetClient})
	checkNoError(t, err)

	err = ValidateSplitInferencePlan(SplitInferencePlan{
		Mode:       SplitInferenceModeRemoteFFN,
		LocalSlice: local,
		Endpoints: []SplitEndpoint{{
			ID:   "ffn-0",
			Role: SplitEndpointRoleFFN,
			URL:  "http://127.0.0.1:8765",
		}},
	})

	checkNoError(t, err)
}

func TestValidateSplitInferencePlan_RemoteFFNMissingEndpoint_Bad(t *testing.T) {
	local, err := PlanModelSlice(ModelSliceRequest{Preset: ModelSlicePresetClient})
	checkNoError(t, err)

	err = ValidateSplitInferencePlan(SplitInferencePlan{
		Mode:       SplitInferenceModeRemoteFFN,
		LocalSlice: local,
	})

	checkError(t, err)
	checkContains(t, err.Error(), "requires an ffn endpoint")
}
