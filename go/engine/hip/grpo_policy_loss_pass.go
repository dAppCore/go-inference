// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import (
	"context"
	"math"
	"strconv"
	"strings"

	core "dappco.re/go"
	"dappco.re/go/inference"
)

var (
	grpoPolicyLogprobSingleKeys          = []string{"logprob", "policy_logprob", "current_logprob", "current_policy_logprob"}
	grpoPolicyLogprobMultiKeys           = []string{"logprobs", "policy_logprobs", "current_logprobs", "current_policy_logprobs"}
	grpoPolicyOldLogprobSingleKeys       = []string{"old_logprob", "old_policy_logprob"}
	grpoPolicyOldLogprobMultiKeys        = []string{"old_logprobs", "old_policy_logprobs"}
	grpoPolicyReferenceLogprobSingleKeys = []string{"reference_logprob", "ref_logprob"}
	grpoPolicyReferenceLogprobMultiKeys  = []string{"reference_logprobs", "ref_logprobs"}
	grpoPolicyAdvantageSingleKeys        = []string{"advantage"}
	grpoPolicyAdvantageMultiKeys         = []string{"advantages"}
	grpoPolicyClipRangeKeys              = []string{"policy_clip_range", "clip_range", "clip_epsilon", "grpo_clip_range"}
	grpoPolicyWeightSingleKeys           = []string{"policy_weight", "loss_weight", "policy_mask", "loss_mask", "response_mask", "action_mask", "token_mask"}
	grpoPolicyWeightMultiKeys            = []string{"policy_weights", "loss_weights", "policy_masks", "loss_masks", "response_masks", "action_masks", "token_masks"}
)

// RunNativeGRPOPolicyLossPass consumes labelled GRPO rows with rewards,
// current logprobs, old policy logprobs, and optional reference logprobs. It
// computes the scalar policy loss while keeping rollout, KL scheduling, and
// public GRPOTrainer semantics outside this helper.
func RunNativeGRPOPolicyLossPass(ctx context.Context, model inference.TextModel, dataset inference.DatasetStream, cfg inference.GRPOConfig) (*inference.TrainingResult, bool, error) {
	if model == nil {
		return nil, false, core.NewError("rocm: native GRPO policy loss pass model is nil")
	}
	rocm, ok := model.(*rocmModel)
	if !ok {
		return nil, false, core.NewError("rocm: native GRPO policy loss pass requires a ROCm model")
	}
	if dataset == nil {
		return nil, false, core.NewError("rocm: native GRPO policy loss pass dataset is nil")
	}
	if ctx == nil {
		ctx = context.Background()
	}
	rows, err := collectGRPOPolicyRows(ctx, dataset)
	if err != nil {
		return nil, false, err
	}
	if len(rows.rewards) == 0 {
		if len(rows.advantages) == 0 {
			return nil, false, core.NewError("rocm: native GRPO policy loss pass dataset produced no policy rows")
		}
	}
	advantages := rows.advantages
	nativeAdvantage := false
	if len(advantages) == 0 {
		advantages, nativeAdvantage, err = grpoPolicyAdvantages(ctx, model, rows.rewards)
		if err != nil {
			return nil, nativeAdvantage, err
		}
	}
	clipRange, clipRangeSet, err := grpoPolicyClipRangeFromLabels(cfg.Labels)
	if err != nil {
		return nil, nativeAdvantage, err
	}
	if rows.clipRangeSet {
		if clipRangeSet && clipRange != rows.clipRange {
			return nil, nativeAdvantage, core.NewError("rocm: GRPO policy clip range labels conflict")
		}
		if !clipRangeSet {
			clipRange = rows.clipRange
		}
	}
	policyTerms := len(advantages)
	if cfg.GroupSize > 0 && policyTerms%cfg.GroupSize != 0 {
		return nil, nativeAdvantage, core.NewError("rocm: GRPO policy terms must be divisible by group size")
	}
	loss, ratioMean, ratioMin, ratioMax, klMean, klMax, objectiveMean, clippedObjectiveMean, clipFraction, clipLowFraction, clipHighFraction, weightSum, activeTerms, err := rocmReferenceGRPOPolicyLoss(advantages, rows.logprobs, rows.oldLogprobs, rows.referenceLogprobs, rows.weights, cfg.KLWeight, clipRange)
	if err != nil {
		return nil, nativeAdvantage, err
	}

	labels := cloneGRPOPolicyResultLabels(cfg.Labels, clipRange > 0)
	labels["training_stage"] = "grpo_policy_loss_pass"
	labels["training_interface"] = "policy_loss_only"
	labels["training_update_status"] = "not_applied"
	labels["trainer_interface"] = "not_implemented"
	labels["policy_loss_backend"] = "reference"
	labels["policy_loss_kernel"] = hipKernelStatusNotLinked
	policyLossLabel := formatFloat64Label(loss)
	labels["policy_loss"] = policyLossLabel
	labels["policy_ratio_mean"] = formatFloat64Label(ratioMean)
	labels["policy_ratio_min"] = formatFloat64Label(ratioMin)
	labels["policy_ratio_max"] = formatFloat64Label(ratioMax)
	labels["policy_kl_mean"] = formatFloat64Label(klMean)
	labels["policy_kl_max"] = formatFloat64Label(klMax)
	labels["policy_reference_source"] = rows.referenceSource()
	labels["policy_objective_mean"] = formatFloat64Label(objectiveMean)
	klLoss := cfg.KLWeight * klMean
	if klLoss == 0 {
		labels["policy_objective_loss"] = policyLossLabel
		labels["policy_kl_loss"] = "0"
	} else {
		labels["policy_objective_loss"] = formatFloat64Label(-clippedObjectiveMean)
		labels["policy_kl_loss"] = formatFloat64Label(klLoss)
	}
	labels["advantage_native_ready"] = boolLabel(nativeAdvantage)
	if clipRange > 0 {
		labels["policy_clip_range"] = formatFloat64Label(clipRange)
		labels["policy_clipped_objective_mean"] = formatFloat64Label(clippedObjectiveMean)
		labels["policy_clip_fraction"] = formatFloat64Label(clipFraction)
		labels["policy_clip_low_fraction"] = formatFloat64Label(clipLowFraction)
		labels["policy_clip_high_fraction"] = formatFloat64Label(clipHighFraction)
	}
	if rows.weightsSet {
		labels["policy_weight_source"] = "dataset"
		labels["policy_weight_sum"] = formatFloat64Label(weightSum)
	}
	if len(rows.advantages) > 0 {
		labels["advantage_source"] = "dataset"
	} else {
		labels["advantage_source"] = "reward_normalization"
	}
	labels["advantages"] = formatFloat64CSVLabel(advantages)
	labels["grpo_policy_rows"] = strconv.Itoa(rows.samples)
	policyTermsLabel := strconv.Itoa(policyTerms)
	labels["grpo_policy_terms"] = policyTermsLabel
	referenceTermsLabel := "0"
	fallbackTermsLabel := policyTermsLabel
	if rows.referenceTerms == policyTerms {
		referenceTermsLabel = policyTermsLabel
		fallbackTermsLabel = "0"
	} else if rows.referenceTerms > 0 {
		referenceTermsLabel = strconv.Itoa(rows.referenceTerms)
		fallbackTermsLabel = strconv.Itoa(policyTerms - rows.referenceTerms)
	}
	labels["grpo_policy_reference_terms"] = referenceTermsLabel
	labels["grpo_policy_reference_fallback_terms"] = fallbackTermsLabel
	if activeTerms == policyTerms {
		labels["grpo_policy_active_terms"] = policyTermsLabel
	} else {
		labels["grpo_policy_active_terms"] = strconv.Itoa(activeTerms)
	}
	if cfg.GroupSize > 0 {
		labels["grpo_group_size"] = strconv.Itoa(cfg.GroupSize)
		labels["grpo_policy_groups"] = strconv.Itoa(policyTerms / cfg.GroupSize)
	}
	if len(rows.rolloutGroupIDs) > 0 {
		labels["grpo_rollout_group_source"] = "dataset"
		labels["grpo_rollout_groups"] = strconv.Itoa(len(rows.rolloutGroupIDs))
	}
	if len(rows.rolloutPromptIDs) > 0 {
		labels["grpo_rollout_prompt_source"] = "dataset"
		labels["grpo_rollout_prompts"] = strconv.Itoa(len(rows.rolloutPromptIDs))
	}
	if len(rows.rolloutIDs) > 0 {
		labels["grpo_rollout_source"] = "dataset"
		labels["grpo_rollouts"] = strconv.Itoa(len(rows.rolloutIDs))
	}
	if len(rows.rolloutSampleIDs) > 0 {
		labels["grpo_rollout_sample_source"] = "dataset"
		labels["grpo_rollout_samples"] = strconv.Itoa(len(rows.rolloutSampleIDs))
	}
	if len(rows.rolloutTrajectoryIDs) > 0 {
		labels["grpo_rollout_trajectory_source"] = "dataset"
		labels["grpo_rollout_trajectories"] = strconv.Itoa(len(rows.rolloutTrajectoryIDs))
	}
	if len(rows.rolloutTurnIDs) > 0 {
		labels["grpo_rollout_turn_source"] = "dataset"
		labels["grpo_rollout_turns"] = strconv.Itoa(len(rows.rolloutTurnIDs))
	}
	if len(rows.rolloutCompletionIDs) > 0 {
		labels["grpo_rollout_completion_source"] = "dataset"
		labels["grpo_rollout_completions"] = strconv.Itoa(len(rows.rolloutCompletionIDs))
	}
	if len(rows.rolloutEpisodeIDs) > 0 {
		labels["grpo_rollout_episode_source"] = "dataset"
		labels["grpo_rollout_episodes"] = strconv.Itoa(len(rows.rolloutEpisodeIDs))
	}
	if cfg.KLWeight != 0 {
		labels["grpo_kl_weight"] = formatFloat64Label(cfg.KLWeight)
	}
	return &inference.TrainingResult{
		Model:   rocm.modelIdentity(),
		Adapter: rocm.ActiveAdapter(),
		Metrics: inference.TrainingMetrics{
			Samples: len(advantages),
			Step:    1,
			Loss:    loss,
		},
		Labels: labels,
	}, nativeAdvantage, nil
}

// RunNativeGRPOPolicyAdamWUpdatePass composes GRPO policy loss with the packed
// AdamW update primitive using caller-provided gradients.
func RunNativeGRPOPolicyAdamWUpdatePass(ctx context.Context, model inference.TextModel, dataset inference.DatasetStream, state *NativeAdamWState, gradients [][]float32, cfg inference.GRPOConfig) (*inference.TrainingResult, bool, error) {
	if state == nil {
		return nil, false, core.NewError("rocm: native GRPO policy AdamW update pass state is nil")
	}
	loss, nativeAdvantage, err := RunNativeGRPOPolicyLossPass(ctx, model, dataset, cfg)
	if err != nil {
		return nil, false, err
	}
	update, err := RunNativeAdamWUpdatePass(ctx, model, state, gradients, cfg.TrainingConfig)
	if err != nil {
		return loss, nativeAdvantage, err
	}
	labels := rocmCloneLabels(loss.Labels)
	if labels == nil {
		labels = make(map[string]string, 28)
	}
	mergeNativeAdamWUpdateLabels(labels, update)
	labels["training_stage"] = "grpo_policy_loss_adamw_update_pass"
	labels["training_interface"] = "policy_loss_plus_optimizer_update"
	labels["training_update_status"] = "applied"
	labels["trainer_interface"] = "not_implemented"

	result := *loss
	result.Metrics.Step = update.Metrics.Step
	result.Metrics.LearningRate = update.Metrics.LearningRate
	result.Labels = labels
	return &result, nativeAdvantage, nil
}

func cloneGRPOPolicyResultLabels(labels map[string]string, clipped bool) map[string]string {
	capacity := 24
	if len(labels) > 0 || clipped {
		capacity = len(labels) + 36
	}
	out := make(map[string]string, capacity)
	for key, value := range labels {
		out[key] = value
	}
	return out
}

// RunNativeGRPOPolicyAdamWUpdateTrackPass applies one GRPO policy-loss +
// AdamW update step, then appends the updated optimizer state to an append-only
// track.
func RunNativeGRPOPolicyAdamWUpdateTrackPass(ctx context.Context, model inference.TextModel, dataset inference.DatasetStream, state *NativeAdamWState, gradients [][]float32, trackPath string, cfg inference.GRPOConfig) (*inference.TrainingResult, NativeAdamWTrackRecord, bool, error) {
	if trackPath == "" {
		return nil, NativeAdamWTrackRecord{}, false, core.NewError("rocm: native GRPO policy AdamW update track path is required")
	}
	result, nativeAdvantage, err := RunNativeGRPOPolicyAdamWUpdatePass(ctx, model, dataset, state, gradients, cfg)
	if err != nil {
		return result, NativeAdamWTrackRecord{}, nativeAdvantage, err
	}
	record, err := AppendNativeAdamWStateTrack(trackPath, state)
	if err != nil {
		return result, NativeAdamWTrackRecord{}, nativeAdvantage, err
	}
	labels := rocmCloneLabels(result.Labels)
	if labels == nil {
		labels = make(map[string]string, 32)
	}
	if err := addNativeAdamWTrackLabels(labels, trackPath, record); err != nil {
		return result, NativeAdamWTrackRecord{}, nativeAdvantage, err
	}
	labels["training_stage"] = "grpo_policy_loss_adamw_update_track_pass"

	out := *result
	out.Labels = labels
	return &out, record, nativeAdvantage, nil
}

type grpoPolicyRows struct {
	rewards              []float64
	advantages           []float64
	logprobs             []float64
	oldLogprobs          []float64
	referenceLogprobs    []float64
	weights              []float64
	samples              int
	referenceTerms       int
	rolloutGroupIDs      map[string]struct{}
	rolloutPromptIDs     map[string]struct{}
	rolloutIDs           map[string]struct{}
	rolloutSampleIDs     map[string]struct{}
	rolloutTrajectoryIDs map[string]struct{}
	rolloutTurnIDs       map[string]struct{}
	rolloutCompletionIDs map[string]struct{}
	rolloutEpisodeIDs    map[string]struct{}
	clipRange            float64
	clipRangeSet         bool
	weightsSet           bool
}

func collectGRPOPolicyRows(ctx context.Context, dataset inference.DatasetStream) (grpoPolicyRows, error) {
	var rows grpoPolicyRows
	sampleHint := grpoDatasetRemainingHint(dataset)
	for {
		if err := ctx.Err(); err != nil {
			return grpoPolicyRows{}, err
		}
		sample, ok, err := dataset.Next()
		if err != nil {
			return grpoPolicyRows{}, err
		}
		if !ok {
			break
		}
		advantageStart := len(rows.advantages)
		rows.advantages, err = grpoAppendOptionalPolicyValuesFromLabels(rows.advantages, sample.Labels, grpoPolicyAdvantageSingleKeys, grpoPolicyAdvantageMultiKeys, 0)
		if err != nil {
			return grpoPolicyRows{}, core.E("rocm.GRPOPolicyLossPass", "parse advantages", err)
		}
		hasAdvantages := len(rows.advantages) > advantageStart
		rewardStart := len(rows.rewards)
		rows.rewards, err = grpoAppendRewardsFromLabels(rows.rewards, sample.Labels)
		if err != nil {
			return grpoPolicyRows{}, err
		}
		hasRewards := len(rows.rewards) > rewardStart
		if !hasRewards && !hasAdvantages {
			continue
		}
		rows.addRolloutMetadata(sample.Labels)
		clipRange, clipRangeSet, err := grpoPolicyClipRangeFromLabels(sample.Labels)
		if err != nil {
			rows.rewards = rows.rewards[:rewardStart]
			rows.advantages = rows.advantages[:advantageStart]
			return grpoPolicyRows{}, err
		}
		if clipRangeSet {
			if rows.clipRangeSet && rows.clipRange != clipRange {
				rows.rewards = rows.rewards[:rewardStart]
				rows.advantages = rows.advantages[:advantageStart]
				return grpoPolicyRows{}, core.NewError("rocm: GRPO policy clip range labels conflict")
			}
			rows.clipRange = clipRange
			rows.clipRangeSet = true
		}
		if hasAdvantages && rewardStart > 0 {
			return grpoPolicyRows{}, core.NewError("rocm: GRPO policy rows cannot mix dataset advantages with reward-normalized rows")
		}
		if !hasAdvantages && len(rows.advantages) > 0 {
			rows.rewards = rows.rewards[:rewardStart]
			return grpoPolicyRows{}, core.NewError("rocm: GRPO policy rows cannot mix dataset advantages with reward-normalized rows")
		}
		terms := len(rows.rewards) - rewardStart
		if hasAdvantages {
			terms = len(rows.advantages) - advantageStart
			if hasRewards && len(rows.rewards)-rewardStart != terms {
				rows.rewards = rows.rewards[:rewardStart]
				return grpoPolicyRows{}, core.NewError("rocm: GRPO policy advantage count does not match rewards")
			}
			rows.rewards = rows.rewards[:rewardStart]
		}
		if rows.samples == 0 && sampleHint > 0 {
			reserveGRPOPolicyRows(&rows, terms*sampleHint, hasAdvantages)
		}
		weightStart := len(rows.weights)
		if rows.samples == 0 && sampleHint > 0 && grpoHasPolicyValuesFromLabels(sample.Labels, grpoPolicyWeightSingleKeys, grpoPolicyWeightMultiKeys) {
			rows.weights = reserveFloat64Capacity(rows.weights, terms*sampleHint)
		}
		rows.weights, err = grpoAppendOptionalPolicyValuesFromLabels(rows.weights, sample.Labels, grpoPolicyWeightSingleKeys, grpoPolicyWeightMultiKeys, terms)
		if err != nil {
			return grpoPolicyRows{}, core.E("rocm.GRPOPolicyLossPass", "parse policy weights", err)
		}
		hasWeights := len(rows.weights) > weightStart
		if hasWeights {
			if rows.samples > 0 && !rows.weightsSet {
				rows.weights = rows.weights[:weightStart]
				return grpoPolicyRows{}, core.NewError("rocm: GRPO policy rows cannot mix weighted and unweighted rows")
			}
			if err := validateGRPOPolicyWeights(rows.weights[weightStart:]); err != nil {
				rows.weights = rows.weights[:weightStart]
				return grpoPolicyRows{}, err
			}
			rows.weightsSet = true
		} else if rows.weightsSet {
			return grpoPolicyRows{}, core.NewError("rocm: GRPO policy rows cannot mix weighted and unweighted rows")
		}
		rows.logprobs, err = grpoAppendPolicyValuesFromLabels(rows.logprobs, sample.Labels, grpoPolicyLogprobSingleKeys, grpoPolicyLogprobMultiKeys, terms)
		if err != nil {
			return grpoPolicyRows{}, core.E("rocm.GRPOPolicyLossPass", "parse logprobs", err)
		}
		oldStart := len(rows.oldLogprobs)
		rows.oldLogprobs, err = grpoAppendPolicyValuesFromLabels(rows.oldLogprobs, sample.Labels, grpoPolicyOldLogprobSingleKeys, grpoPolicyOldLogprobMultiKeys, terms)
		if err != nil {
			return grpoPolicyRows{}, core.E("rocm.GRPOPolicyLossPass", "parse old logprobs", err)
		}
		refStart := len(rows.referenceLogprobs)
		rows.referenceLogprobs, err = grpoAppendOptionalPolicyValuesFromLabels(rows.referenceLogprobs, sample.Labels, grpoPolicyReferenceLogprobSingleKeys, grpoPolicyReferenceLogprobMultiKeys, terms)
		if err != nil {
			return grpoPolicyRows{}, core.E("rocm.GRPOPolicyLossPass", "parse reference logprobs", err)
		}
		if len(rows.referenceLogprobs) == refStart {
			rows.referenceLogprobs = append(rows.referenceLogprobs, rows.oldLogprobs[oldStart:]...)
		} else {
			rows.referenceTerms += len(rows.referenceLogprobs) - refStart
		}
		rows.samples++
	}
	return rows, nil
}

func (rows *grpoPolicyRows) addRolloutMetadata(labels map[string]string) {
	groupID := core.Trim(labels["group_id"])
	if groupID != "" {
		if rows.rolloutGroupIDs == nil {
			rows.rolloutGroupIDs = make(map[string]struct{}, 4)
		}
		rows.rolloutGroupIDs[groupID] = struct{}{}
	}
	promptID := core.Trim(labels["prompt_id"])
	if promptID == "" {
		promptID = core.Trim(labels["query_id"])
	}
	if promptID != "" {
		if rows.rolloutPromptIDs == nil {
			rows.rolloutPromptIDs = make(map[string]struct{}, 4)
		}
		rows.rolloutPromptIDs[promptID] = struct{}{}
	}
	rows.addRolloutLabelID(labels, "rollout_id", &rows.rolloutIDs)
	rows.addRolloutLabelID(labels, "sample_id", &rows.rolloutSampleIDs)
	rows.addRolloutLabelID(labels, "trajectory_id", &rows.rolloutTrajectoryIDs)
	rows.addRolloutLabelID(labels, "turn_id", &rows.rolloutTurnIDs)
	rows.addRolloutLabelID(labels, "completion_id", &rows.rolloutCompletionIDs)
	rows.addRolloutLabelID(labels, "episode_id", &rows.rolloutEpisodeIDs)
}

func (rows *grpoPolicyRows) addRolloutLabelID(labels map[string]string, key string, ids *map[string]struct{}) {
	value := core.Trim(labels[key])
	if value == "" {
		return
	}
	if *ids == nil {
		*ids = make(map[string]struct{}, 4)
	}
	(*ids)[value] = struct{}{}
}

func (rows grpoPolicyRows) referenceSource() string {
	if len(rows.referenceLogprobs) == 0 || rows.referenceTerms == 0 {
		return "old_policy_fallback"
	}
	if rows.referenceTerms == len(rows.referenceLogprobs) {
		return "dataset"
	}
	return "mixed_dataset_old_policy_fallback"
}

type grpoRemainingDataset interface {
	Remaining() int
}

func grpoDatasetRemainingHint(dataset inference.DatasetStream) int {
	if hinted, ok := dataset.(grpoRemainingDataset); ok && hinted != nil {
		return hinted.Remaining()
	}
	return 0
}

func reserveGRPOPolicyRows(rows *grpoPolicyRows, terms int, advantages bool) {
	if rows == nil || terms <= 0 {
		return
	}
	if advantages {
		rows.advantages = reserveFloat64Capacity(rows.advantages, terms)
	} else {
		rows.rewards = reserveFloat64Capacity(rows.rewards, terms)
	}
	rows.logprobs = reserveFloat64Capacity(rows.logprobs, terms)
	rows.oldLogprobs = reserveFloat64Capacity(rows.oldLogprobs, terms)
	rows.referenceLogprobs = reserveFloat64Capacity(rows.referenceLogprobs, terms)
}

func reserveFloat64Capacity(values []float64, capacity int) []float64 {
	if cap(values) >= capacity {
		return values
	}
	out := make([]float64, len(values), capacity)
	copy(out, values)
	return out
}

func grpoAppendRewardsFromLabels(out []float64, labels map[string]string) ([]float64, error) {
	start := len(out)
	rewardRaw := core.Trim(labels["reward"])
	if rewardRaw != "" {
		var err error
		out, err = parseFloat64CSVLabelAppend(out, rewardRaw)
		if err != nil {
			return out[:start], core.E("rocm.GRPOAdvantagePass", "parse reward", err)
		}
	}
	rewardsRaw := core.Trim(labels["rewards"])
	if rewardsRaw != "" {
		var err error
		out, err = parseFloat64CSVLabelAppend(out, rewardsRaw)
		if err != nil {
			return out[:start], core.E("rocm.GRPOAdvantagePass", "parse rewards", err)
		}
	}
	if len(out) == start && (rewardRaw != "" || rewardsRaw != "") {
		return out[:start], core.NewError("rocm: GRPO rewards must be non-empty")
	}
	return out, nil
}

func grpoAppendPolicyValuesFromLabels(out []float64, labels map[string]string, singleKeys, multiKeys []string, want int) ([]float64, error) {
	raw := grpoPrimaryLabelValue(labels, singleKeys)
	if raw == "" {
		raw = grpoPrimaryLabelValue(labels, multiKeys)
	}
	if raw == "" {
		return out, core.Errorf("missing %s", singleKeys[0])
	}
	start := len(out)
	out, err := parseFloat64CSVLabelAppend(out, raw)
	if err != nil {
		return out[:start], err
	}
	if len(out)-start != want {
		return out[:start], core.Errorf("%s count %d does not match rewards %d", singleKeys[0], len(out)-start, want)
	}
	return out, nil
}

func grpoAppendOptionalPolicyValuesFromLabels(out []float64, labels map[string]string, singleKeys, multiKeys []string, want int) ([]float64, error) {
	raw := grpoPrimaryLabelValue(labels, singleKeys)
	if raw == "" {
		raw = grpoPrimaryLabelValue(labels, multiKeys)
	}
	if raw == "" {
		return out, nil
	}
	start := len(out)
	out, err := parseFloat64CSVLabelAppend(out, raw)
	if err != nil {
		return out[:start], err
	}
	if want > 0 && len(out)-start != want {
		return out[:start], core.Errorf("%s count %d does not match rewards %d", singleKeys[0], len(out)-start, want)
	}
	return out, nil
}

func grpoHasPolicyValuesFromLabels(labels map[string]string, singleKeys, multiKeys []string) bool {
	if grpoPrimaryLabelValue(labels, singleKeys) != "" {
		return true
	}
	return grpoPrimaryLabelValue(labels, multiKeys) != ""
}

func grpoPrimaryLabelValue(labels map[string]string, keys []string) string {
	if len(keys) == 0 {
		return ""
	}
	if value := core.Trim(labels[keys[0]]); value != "" {
		return value
	}
	for i := 1; i < len(keys); i++ {
		if value := core.Trim(labels[keys[i]]); value != "" {
			return value
		}
	}
	return ""
}

func validateGRPOPolicyWeights(weights []float64) error {
	for _, weight := range weights {
		if weight < 0 || math.IsNaN(weight) || math.IsInf(weight, 0) {
			return core.NewError("rocm: GRPO policy weights must be finite and non-negative")
		}
	}
	return nil
}

func grpoPolicyAdvantages(ctx context.Context, model inference.TextModel, rewards []float64) ([]float64, bool, error) {
	if advantages, ok, err := RunNativeGRPOAdvantage(ctx, model, rewards); ok {
		if err != nil {
			return nil, true, err
		}
		return advantages, true, nil
	}
	advantages, err := rocmReferenceNormalizeAdvantages(rewards)
	return advantages, false, err
}

func grpoPolicyClipRangeFromLabels(labels map[string]string) (float64, bool, error) {
	raw := grpoPrimaryLabelValue(labels, grpoPolicyClipRangeKeys)
	if raw == "" {
		return 0, false, nil
	}
	raw = core.Trim(raw)
	if strings.IndexByte(raw, ',') >= 0 {
		return 0, true, core.NewError("rocm: GRPO policy clip range must be one finite non-negative value")
	}
	value, err := strconv.ParseFloat(raw, 64)
	if err != nil {
		return 0, true, core.E("rocm.GRPOPolicyLossPass", "parse clip range", err)
	}
	if value < 0 || math.IsNaN(value) || math.IsInf(value, 0) {
		return 0, true, core.NewError("rocm: GRPO policy clip range must be one finite non-negative value")
	}
	return value, true, nil
}

func rocmReferenceGRPOPolicyLoss(advantages, logprobs, oldLogprobs, referenceLogprobs, weights []float64, klWeight, clipRange float64) (float64, float64, float64, float64, float64, float64, float64, float64, float64, float64, float64, float64, int, error) {
	if len(advantages) == 0 || len(logprobs) != len(advantages) || len(oldLogprobs) != len(advantages) || len(referenceLogprobs) != len(advantages) {
		return 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, core.NewError("rocm: GRPO policy loss inputs must have matching non-empty lengths")
	}
	if len(weights) > 0 && len(weights) != len(advantages) {
		return 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, core.NewError("rocm: GRPO policy loss weights must match policy terms")
	}
	if clipRange < 0 || math.IsNaN(clipRange) || math.IsInf(clipRange, 0) {
		return 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, core.NewError("rocm: GRPO policy clip range must be finite and non-negative")
	}
	var objectiveSum, clippedObjectiveSum, ratioSum, ratioMin, ratioMax, klSum, klMax, clippedTerms, lowClippedTerms, highClippedTerms, weightSum float64
	activeTerms := 0
	for i := range advantages {
		termWeight := 1.0
		if len(weights) > 0 {
			termWeight = weights[i]
			if termWeight < 0 || math.IsNaN(termWeight) || math.IsInf(termWeight, 0) {
				return 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, core.NewError("rocm: GRPO policy weights must be finite and non-negative")
			}
		}
		if termWeight == 0 {
			continue
		}
		ratio := math.Exp(logprobs[i] - oldLogprobs[i])
		klDelta := referenceLogprobs[i] - logprobs[i]
		kl := math.Exp(klDelta) - klDelta - 1
		if math.IsNaN(ratio) || math.IsInf(ratio, 0) || math.IsNaN(kl) || math.IsInf(kl, 0) {
			return 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, core.NewError("rocm: GRPO policy loss produced non-finite term")
		}
		if activeTerms == 0 {
			ratioMin = ratio
			ratioMax = ratio
			klMax = kl
		} else {
			ratioMin = math.Min(ratioMin, ratio)
			ratioMax = math.Max(ratioMax, ratio)
			klMax = math.Max(klMax, kl)
		}
		objective := advantages[i] * ratio
		clippedObjective := objective
		if clipRange > 0 {
			lowRatio := 1 - clipRange
			highRatio := 1 + clipRange
			if ratio < lowRatio {
				lowClippedTerms += termWeight
			} else if ratio > highRatio {
				highClippedTerms += termWeight
			}
			clippedRatio := math.Min(math.Max(ratio, lowRatio), highRatio)
			clippedObjective = advantages[i] * clippedRatio
			if advantages[i] >= 0 {
				clippedObjective = math.Min(objective, clippedObjective)
			} else {
				clippedObjective = math.Max(objective, clippedObjective)
			}
			if clippedObjective != objective {
				clippedTerms += termWeight
			}
		}
		objectiveSum += objective * termWeight
		clippedObjectiveSum += clippedObjective * termWeight
		ratioSum += ratio * termWeight
		klSum += kl * termWeight
		weightSum += termWeight
		activeTerms++
	}
	if weightSum <= 0 {
		return 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, core.NewError("rocm: GRPO policy weight sum must be positive")
	}
	scale := 1 / weightSum
	objectiveMean := objectiveSum * scale
	clippedObjectiveMean := clippedObjectiveSum * scale
	loss := -clippedObjectiveMean + klWeight*klSum*scale
	return loss, ratioSum * scale, ratioMin, ratioMax, klSum * scale, klMax, objectiveMean, clippedObjectiveMean, clippedTerms * scale, lowClippedTerms * scale, highClippedTerms * scale, weightSum, activeTerms, nil
}
