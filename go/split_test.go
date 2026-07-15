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

// localAttentionSlice is the minimal local slice a remote split mode needs: it
// holds attention, so the mode's "requires local attention" guard passes and
// the endpoint checks are reached.
func localAttentionSlice() ModelSlicePlan {
	return ModelSlicePlan{Components: []ModelComponent{ModelComponentAttention}}
}

// TestValidateSplitInferencePlan_Modes pins every split-mode arm of the
// validator, backend-neutral: the default/local pass, each remote mode's
// local-attention + endpoint-role requirements, and the unknown-mode rejection.
func TestValidateSplitInferencePlan_Modes(t *testing.T) {
	local := localAttentionSlice()
	cases := []struct {
		name    string
		plan    SplitInferencePlan
		wantErr string
	}{
		{"DefaultModeLocal", SplitInferencePlan{}, ""},
		{"ExplicitLocal", SplitInferencePlan{Mode: SplitInferenceModeLocal}, ""},
		{
			"RemoteFFNNoLocalAttention",
			SplitInferencePlan{Mode: SplitInferenceModeRemoteFFN},
			"requires local attention",
		},
		{
			"RemoteEmbedFFNNoLocalAttention",
			SplitInferencePlan{Mode: SplitInferenceModeRemoteEmbedFFN},
			"requires local attention",
		},
		{
			"RemoteEmbedFFNMissingEmbeddings",
			SplitInferencePlan{Mode: SplitInferenceModeRemoteEmbedFFN, LocalSlice: local,
				Endpoints: []SplitEndpoint{{ID: "ffn", Role: SplitEndpointRoleFFN}}},
			"requires an embeddings endpoint",
		},
		{
			"RemoteEmbedFFNMissingFFN",
			SplitInferencePlan{Mode: SplitInferenceModeRemoteEmbedFFN, LocalSlice: local,
				Endpoints: []SplitEndpoint{{ID: "emb", Role: SplitEndpointRoleEmbeddings}}},
			"requires an ffn endpoint",
		},
		{
			"RemoteEmbedFFNGood",
			SplitInferencePlan{Mode: SplitInferenceModeRemoteEmbedFFN, LocalSlice: local,
				Endpoints: []SplitEndpoint{
					{ID: "emb", Role: SplitEndpointRoleEmbeddings},
					{ID: "ffn", Role: SplitEndpointRoleFFN},
				}},
			"",
		},
		{
			"RemoteExpertsNoLocalAttention",
			SplitInferencePlan{Mode: SplitInferenceModeRemoteExperts},
			"requires local attention",
		},
		{
			"RemoteExpertsMissingEndpoint",
			SplitInferencePlan{Mode: SplitInferenceModeRemoteExperts, LocalSlice: local},
			"requires an expert endpoint",
		},
		{
			"RemoteExpertsGood",
			SplitInferencePlan{Mode: SplitInferenceModeRemoteExperts, LocalSlice: local,
				Endpoints: []SplitEndpoint{{ID: "moe", Role: SplitEndpointRoleExpert}}},
			"",
		},
		{
			"UnknownMode",
			SplitInferencePlan{Mode: SplitInferenceMode("sideways")},
			"unknown split inference mode",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			err := ValidateSplitInferencePlan(tc.plan)
			if tc.wantErr == "" {
				checkNoError(t, err)
				return
			}
			checkError(t, err)
			checkContains(t, err.Error(), tc.wantErr)
		})
	}
}

// TestValidateSplitInferencePlan_EndpointArms pins the shared endpoint validator
// reached after a mode's own checks pass: each malformed endpoint fails with its
// own message rather than being loaded into a broken topology.
func TestValidateSplitInferencePlan_EndpointArms(t *testing.T) {
	local := localAttentionSlice()
	good := SplitEndpoint{ID: "ffn", Role: SplitEndpointRoleFFN}
	cases := []struct {
		name    string
		bad     SplitEndpoint
		wantErr string
	}{
		{"NoRole", SplitEndpoint{ID: "x"}, "requires a role"},
		{"NoIDOrURL", SplitEndpoint{Role: SplitEndpointRoleAttention}, "requires an id or url"},
		{"InvalidLayerRange", SplitEndpoint{ID: "a", Role: SplitEndpointRoleAttention, LayerStart: 20, LayerEnd: 10}, "layer range is invalid"},
		{"InvalidExpertRange", SplitEndpoint{ID: "e", Role: SplitEndpointRoleExpert, ExpertStart: 8, ExpertEnd: 2}, "expert range is invalid"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			err := ValidateSplitInferencePlan(SplitInferencePlan{
				Mode:       SplitInferenceModeRemoteFFN,
				LocalSlice: local,
				Endpoints:  []SplitEndpoint{good, tc.bad},
			})
			checkError(t, err)
			checkContains(t, err.Error(), tc.wantErr)
		})
	}
}

// TestPlanModelSlice_RemainingPresets_Good pins the preset-expansion arms not
// covered elsewhere (full, embed, browse, router, expert_server), so every named
// topology maps to a concrete, non-empty component set.
func TestPlanModelSlice_RemainingPresets_Good(t *testing.T) {
	cases := []struct {
		preset   ModelSlicePreset
		contains ModelComponent
	}{
		{ModelSlicePresetFull, ModelComponentExperts},
		{ModelSlicePresetEmbed, ModelComponentEmbeddings},
		{ModelSlicePresetBrowse, ModelComponentRouter},
		{ModelSlicePresetRouter, ModelComponentRouter},
		{ModelSlicePresetExpertServer, ModelComponentExperts},
	}
	for _, tc := range cases {
		t.Run(string(tc.preset), func(t *testing.T) {
			plan, err := PlanModelSlice(ModelSliceRequest{Preset: tc.preset})
			checkNoError(t, err)
			if len(plan.Components) == 0 {
				t.Fatalf("preset %q expanded to no components", tc.preset)
			}
			if !plan.HasComponent(tc.contains) {
				t.Fatalf("preset %q components %v missing %q", tc.preset, plan.Components, tc.contains)
			}
		})
	}
}
