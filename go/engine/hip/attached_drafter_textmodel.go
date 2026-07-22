// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import (
	"context"
	"iter"
	"sync"
	"time"

	core "dappco.re/go"
	"dappco.re/go/inference"
	inferdecode "dappco.re/go/inference/decode"
)

// LoadAttachedDrafterPairAsTextModel loads a Gemma4 target beside an attached
// assistant drafter and returns the pair behind inference.TextModel.
func LoadAttachedDrafterPairAsTextModel(targetPath, draftPath string, opts ...inference.LoadOption) (inference.TextModel, error) {
	return LoadAttachedDrafterPairAsTextModelBlock(targetPath, draftPath, 0, opts...)
}

// LoadAttachedDrafterPairAsTextModelWithConfig is LoadAttachedDrafterPairAsTextModel
// with ROCm-specific native load settings applied to both target and assistant.
func LoadAttachedDrafterPairAsTextModelWithConfig(targetPath, draftPath string, cfg ROCmLoadConfig, opts ...inference.LoadOption) (inference.TextModel, error) {
	return LoadAttachedDrafterPairAsTextModelBlockWithConfig(targetPath, draftPath, 0, cfg, opts...)
}

// LoadAttachedDrafterPairAsTextModelBlock is LoadAttachedDrafterPairAsTextModel
// with MTPLX block semantics: block N verifies the carried target lead plus
// N-1 assistant proposals. A non-positive block uses the production default.
func LoadAttachedDrafterPairAsTextModelBlock(targetPath, draftPath string, draftBlock int, opts ...inference.LoadOption) (inference.TextModel, error) {
	return LoadAttachedDrafterPairAsTextModelBlockWithConfig(targetPath, draftPath, draftBlock, ROCmLoadConfig{}, opts...)
}

// LoadAttachedDrafterPairAsTextModelBlockWithConfig is
// LoadAttachedDrafterPairAsTextModelBlock with ROCm-specific native load
// settings applied to both target and assistant.
func LoadAttachedDrafterPairAsTextModelBlockWithConfig(targetPath, draftPath string, draftBlock int, cfg ROCmLoadConfig, opts ...inference.LoadOption) (inference.TextModel, error) {
	pair, err := LoadAttachedDrafterPair(targetPath, draftPath, AttachedDrafterPairConfig{
		TargetOptions:    opts,
		TargetROCmConfig: cfg,
		DraftROCmConfig:  cfg,
	})
	if err != nil {
		return nil, err
	}
	adaptiveDraftTokens := false
	if draftBlock <= 0 {
		draftBlock = ProductionMTPDefaultDraftTokens + 1
		adaptiveDraftTokens = true
	}
	return &attachedDrafterTextModel{
		pair:                pair,
		draftTokens:         max(1, draftBlock-1),
		adaptiveDraftTokens: adaptiveDraftTokens,
	}, nil
}

// IsAttachedDrafterTextModel reports whether model is the native attached-MTP
// pair lane returned by LoadAttachedDrafterPairAsTextModelBlock.
func IsAttachedDrafterTextModel(model inference.TextModel) bool {
	_, ok := model.(*attachedDrafterTextModel)
	return ok
}

type attachedDrafterTextModel struct {
	pair                *AttachedDrafterPair
	draftTokens         int
	adaptiveDraftTokens bool

	mu      sync.Mutex
	err     error
	metrics inference.GenerateMetrics
	mtp     *AttachedDrafterMetrics
}

var (
	_ inference.SerialModel                = (*attachedDrafterTextModel)(nil)
	_ inference.SpeculativeMetricsProvider = (*attachedDrafterTextModel)(nil)
)

// SerialGeneration reports the pair-level target, drafter, KV, and scratch
// ownership shared by every request through this wrapper.
func (*attachedDrafterTextModel) SerialGeneration() bool { return true }

func (model *attachedDrafterTextModel) Generate(ctx context.Context, prompt string, opts ...inference.GenerateOption) iter.Seq[inference.Token] {
	model.clearLastGenerationState()
	cfg := inference.ApplyGenerateOpts(opts)
	return model.generatePrompt(ctx, prompt, cfg, false)
}

func (model *attachedDrafterTextModel) Chat(ctx context.Context, messages []inference.Message, opts ...inference.GenerateOption) iter.Seq[inference.Token] {
	return model.chatWithStatePreference(ctx, messages, true, opts)
}

// ChatStateless applies the target chat template but does not take the
// first-turn retained-state seeding path. It is used by one-shot CLI generate
// when the user explicitly disables retained state.
func (model *attachedDrafterTextModel) ChatStateless(ctx context.Context, messages []inference.Message, opts ...inference.GenerateOption) iter.Seq[inference.Token] {
	return model.chatWithStatePreference(ctx, messages, false, opts)
}

func (model *attachedDrafterTextModel) chatWithStatePreference(ctx context.Context, messages []inference.Message, statePreferred bool, opts []inference.GenerateOption) iter.Seq[inference.Token] {
	model.clearLastGenerationState()
	if err := validateROCmChatMessages("rocm.AttachedDrafterTextModel.Chat", messages); err != nil {
		model.setLastFailure(err)
		return emptyTokenSeq
	}
	target := model.targetROCmModel()
	if target == nil {
		err := core.E("rocm.AttachedDrafterTextModel.Chat", "target model is required", nil)
		model.setLastFailure(err)
		return emptyTokenSeq
	}
	cfg := inference.ApplyGenerateOpts(opts)
	prompt, err := model.chatPromptWithStatePreference(target, messages, cfg, statePreferred)
	if err != nil {
		err = core.E("rocm.AttachedDrafterTextModel.Chat", "apply chat template", err)
		model.setLastFailure(err)
		return emptyTokenSeq
	}
	return model.generatePrompt(ctx, prompt, cfg, statePreferred)
}

func (model *attachedDrafterTextModel) chatPromptWithStatePreference(target *rocmModel, messages []inference.Message, cfg inference.GenerateConfig, statePreferred bool) (string, error) {
	if target == nil {
		return "", core.E("rocm.AttachedDrafterTextModel.Chat", "target model is required", nil)
	}
	if loaded, ok := target.native.(*hipLoadedModel); ok && loaded != nil && isROCmGemma4Architecture(loaded.modelInfo.Architecture) {
		continuation := statePreferred && model.targetRuntimeStateSession() != nil
		templateConfig := loaded.gemma4ChatTemplateConfig(cfg, continuation)
		return formatGemma4ChatTemplateWithConfig(messages, templateConfig), nil
	}
	return target.ApplyChatTemplate(messages)
}

func (model *attachedDrafterTextModel) generatePrompt(ctx context.Context, prompt string, cfg inference.GenerateConfig, statePreferred bool) iter.Seq[inference.Token] {
	if model == nil || model.pair == nil {
		if model != nil {
			model.setLastFailure(core.E("rocm.AttachedDrafterTextModel.Generate", "pair is required", nil))
		}
		return emptyTokenSeq
	}
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		model.setLastFailure(err)
		return emptyTokenSeq
	}
	if model.pair.NativeReady() && !attachedDrafterMTPRequestEligible(cfg) {
		target := model.targetModel()
		if target == nil {
			model.setLastFailure(core.E("rocm.AttachedDrafterTextModel.Generate", "target model is required", nil))
			return emptyTokenSeq
		}
		return target.Generate(ctx, prompt, attachedDrafterGenerateOptions(cfg)...)
	}
	start := time.Now()
	result, err := model.generateNativeResult(ctx, prompt, cfg, statePreferred)
	if err != nil {
		model.setLastFailure(err)
		return emptyTokenSeq
	}
	tokens := make([]inference.Token, len(result.Tokens))
	for i, token := range result.Tokens {
		tokens[i] = inference.Token{ID: token.ID, Text: token.Text}
	}
	model.recordResultMetrics(prompt, len(tokens), result.Metrics, start)
	return func(yield func(inference.Token) bool) {
		for _, token := range tokens {
			if !yield(token) {
				return
			}
		}
	}
}

func (model *attachedDrafterTextModel) generateNativeResult(ctx context.Context, prompt string, cfg inference.GenerateConfig, statePreferred bool) (inferdecode.Result, error) {
	generateCfg := attachedDrafterGenerateConfigFromInference(cfg, model.draftTokens, model.adaptiveDraftTokens)
	if model.pair.NativeReady() {
		if statePreferred {
			if state := model.targetRuntimeStateSession(); state != nil {
				return model.generateNativeFromRuntimeState(ctx, state, prompt, generateCfg)
			}
			if state := model.targetStateSessionForRetention(); state != nil {
				return model.generateNativeWithStateRetention(ctx, state, prompt, generateCfg)
			}
			if model.targetRetainedStateReady() {
				return model.generateTargetRetainedResult(ctx, prompt, cfg)
			}
		}
		return model.pair.GenerateNative(ctx, prompt, generateCfg)
	}
	if model.targetRetainedDecodeOnlyReady() {
		return model.generateTargetRetainedResult(ctx, prompt, cfg)
	}
	return model.pair.GenerateNative(ctx, prompt, generateCfg)
}

func (model *attachedDrafterTextModel) generateNativeFromRuntimeState(ctx context.Context, state *StateSession, prompt string, cfg AttachedDrafterGenerateConfig) (inferdecode.Result, error) {
	if model == nil || model.pair == nil {
		return inferdecode.Result{}, core.E("rocm.AttachedDrafterTextModel.Generate", "pair is required", nil)
	}
	if state == nil {
		return inferdecode.Result{}, core.E("rocm.AttachedDrafterTextModel.Generate", "runtime-owned KV state is required", nil)
	}
	return model.pair.GenerateNativeFromState(ctx, AttachedDrafterStateGenerateRequest{
		State:               state,
		Input:               prompt,
		MaxTokens:           cfg.MaxTokens,
		DraftTokens:         cfg.DraftTokens,
		AdaptiveDraftTokens: cfg.AdaptiveDraftTokens,
		Temperature:         cfg.Temperature,
		TopK:                cfg.TopK,
		TopP:                cfg.TopP,
		MinP:                cfg.MinP,
		StopTokens:          append([]int32(nil), cfg.StopTokens...),
		RepeatPenalty:       cfg.RepeatPenalty,
	})
}

func (model *attachedDrafterTextModel) generateNativeWithStateRetention(ctx context.Context, state *StateSession, prompt string, cfg AttachedDrafterGenerateConfig) (inferdecode.Result, error) {
	if model == nil || model.pair == nil {
		return inferdecode.Result{}, core.E("rocm.AttachedDrafterTextModel.Generate", "pair is required", nil)
	}
	if state == nil {
		return inferdecode.Result{}, core.E("rocm.AttachedDrafterTextModel.Generate", "state session is required", nil)
	}
	return model.pair.GenerateNativeWithStateRetention(ctx, state, prompt, cfg)
}

func (model *attachedDrafterTextModel) generateTargetRetainedResult(ctx context.Context, prompt string, cfg inference.GenerateConfig) (inferdecode.Result, error) {
	target := model.targetModel()
	if target == nil {
		return inferdecode.Result{}, core.E("rocm.AttachedDrafterTextModel.Generate", "target model is required", nil)
	}
	start := time.Now()
	tokens := []inferdecode.Token{}
	for token := range target.Generate(ctx, prompt, attachedDrafterGenerateOptions(cfg)...) {
		tokens = append(tokens, rocmDecodeToken(token))
	}
	if r := target.Err(); !r.OK {
		return inferdecode.Result{}, core.E("rocm.AttachedDrafterTextModel.Generate", "target retained-state generation failed", r.Value.(error))
	}
	duration := time.Since(start)
	return inferdecode.Result{
		Mode:   "target_retained_state",
		Prompt: prompt,
		Tokens: tokens,
		Text:   inferdecode.TokensText(tokens),
		Metrics: inferdecode.Metrics{
			TargetTokens:   len(tokens),
			EmittedTokens:  len(tokens),
			TargetCalls:    1,
			Duration:       duration,
			TargetDuration: duration,
		},
	}, nil
}

func (model *attachedDrafterTextModel) targetRetainedDecodeOnlyReady() bool {
	if model == nil || model.pair == nil {
		return false
	}
	labels := model.pair.Attachment.Labels
	return labels["attached_drafter_native_handoff"] == attachedDrafterNativeHandoffTargetDecodeOnly &&
		model.targetRetainedStateReady()
}

func (model *attachedDrafterTextModel) targetRetainedStateReady() bool {
	if model == nil || model.pair == nil {
		return false
	}
	return attachedDrafterLabelsDeclareRetainedStateReady(model.pair.Attachment.Labels)
}

func (model *attachedDrafterTextModel) targetRuntimeStateSession() *StateSession {
	target := model.targetROCmModel()
	if target == nil {
		return nil
	}
	state := target.currentStateSession()
	if !rocmStateSessionHasRuntimeKV(state) {
		return nil
	}
	return state
}

func (model *attachedDrafterTextModel) targetStateSessionForRetention() *StateSession {
	target := model.targetROCmModel()
	if target == nil || !model.targetRetainedStateReady() {
		return nil
	}
	return target.stateSession()
}

func (model *attachedDrafterTextModel) Classify(ctx context.Context, prompts []string, opts ...inference.GenerateOption) core.Result {
	target := model.targetModel()
	if target == nil {
		err := core.E("rocm.AttachedDrafterTextModel.Classify", "target model is required", nil)
		model.setLastFailure(err)
		return core.Fail(err)
	}
	return target.Classify(ctx, prompts, opts...)
}

func (model *attachedDrafterTextModel) BatchGenerate(ctx context.Context, prompts []string, opts ...inference.GenerateOption) core.Result {
	target := model.targetModel()
	if target == nil {
		err := core.E("rocm.AttachedDrafterTextModel.BatchGenerate", "target model is required", nil)
		model.setLastFailure(err)
		return core.Fail(err)
	}
	return target.BatchGenerate(ctx, prompts, opts...)
}

func (model *attachedDrafterTextModel) ModelType() string {
	target := model.targetModel()
	if target == nil {
		return ""
	}
	return target.ModelType()
}

func (model *attachedDrafterTextModel) Info() inference.ModelInfo {
	target := model.targetModel()
	if target == nil {
		return inference.ModelInfo{}
	}
	return target.Info()
}

func (model *attachedDrafterTextModel) ModelIdentity() inference.ModelIdentity {
	target := model.targetModel()
	if target == nil {
		return inference.ModelIdentity{}
	}
	identity := rocmDecodeModelIdentity(target)
	if rocmModelIdentityIsZero(identity) {
		return inference.ModelIdentity{}
	}
	identity = rocmCloneModelIdentity(identity)
	identity.Labels = model.attachedDrafterLabels(identity.Labels)
	return identity
}

func (model *attachedDrafterTextModel) ModelProfile() ROCmModelProfile {
	target := model.targetModel()
	if target == nil {
		return ROCmModelProfile{}
	}
	profile := ROCmModelProfile{}
	if reporter, ok := target.(ROCmModelProfileReporter); ok {
		profile = reporter.ModelProfile()
	}
	if !profile.Matched() {
		var ok bool
		profile, ok = ResolveROCmModelProfileForModel(target)
		if !ok {
			return ROCmModelProfile{}
		}
	}
	profile = profile.clone()
	profile.Model = model.ModelIdentity()
	profile.Labels = model.attachedDrafterLabels(profile.Labels)
	return profile
}

func (model *attachedDrafterTextModel) ModelRoutePlan() ROCmModelRoutePlan {
	target := model.targetModel()
	if target == nil {
		return ROCmModelRoutePlan{}
	}
	profile := model.ModelProfile()
	if !profile.Matched() {
		return ROCmModelRoutePlan{}
	}
	return ROCmModelRoutePlanForProfileAndModel(profile, target)
}

func (model *attachedDrafterTextModel) Capabilities() inference.CapabilityReport {
	target := model.targetModel()
	if target == nil {
		return inference.CapabilityReport{Runtime: inference.RuntimeIdentity{Backend: "rocm"}}
	}
	report := rocmCapabilityReportForWrappedModel(target)
	report.Model = model.ModelIdentity()
	labels := model.attachedDrafterLabels(map[string]string{
		"wrapper": "attached_drafter",
	})
	rocmCapabilityReportApplyLabels(&report, labels)
	speculativeCapability := inference.ExperimentalCapability(
		inference.CapabilitySpeculativeDecode,
		inference.CapabilityGroupModel,
		"native attached-drafter pair is loaded; speculative decode routes through the attached drafter helper",
	)
	speculativeCapability.Labels = cloneStringMap(labels)
	rocmCapabilityReportSetCapability(&report, speculativeCapability)
	return report
}

func (model *attachedDrafterTextModel) Metrics() inference.GenerateMetrics {
	if model == nil {
		return inference.GenerateMetrics{}
	}
	model.mu.Lock()
	metrics := model.metrics
	model.mu.Unlock()
	if metrics.GeneratedTokens > 0 || metrics.TotalDuration > 0 {
		return metrics
	}
	target := model.targetModel()
	if target == nil {
		return inference.GenerateMetrics{}
	}
	return target.Metrics()
}

func (model *attachedDrafterTextModel) AttachedDrafterMetrics() *AttachedDrafterMetrics {
	if model == nil {
		return nil
	}
	model.mu.Lock()
	defer model.mu.Unlock()
	if model.mtp == nil {
		return nil
	}
	metrics := *model.mtp
	return &metrics
}

// SpeculativeMetrics exposes the last native attached-drafter operation through
// the engine-neutral metrics contract used by serving and CLI callers.
func (model *attachedDrafterTextModel) SpeculativeMetrics() inference.SpeculativeMetrics {
	if model == nil {
		return inference.SpeculativeMetrics{}
	}
	model.mu.Lock()
	if model.mtp == nil {
		model.mu.Unlock()
		return inference.SpeculativeMetrics{}
	}
	mtp := *model.mtp
	peakMemory := model.metrics.PeakMemoryBytes
	model.mu.Unlock()

	metrics := inference.SpeculativeMetrics{
		ProposedTokens:    mtp.ProposedTokens,
		AcceptedTokens:    mtp.AcceptedTokens,
		RejectedTokens:    mtp.RejectedTokens,
		TargetVerifyCalls: mtp.VerifyCalls,
		TargetCalls:       mtp.TargetCalls,
		DraftCalls:        mtp.DraftCalls,
		AcceptanceRate:    mtp.AcceptanceRate,
		WallDuration:      mtp.Duration,
		TargetDuration:    mtp.TargetDuration,
		DraftDuration:     mtp.DraftDuration,
		PeakMemoryBytes:   peakMemory,
	}
	if metrics.AcceptanceRate == 0 && metrics.ProposedTokens > 0 {
		metrics.AcceptanceRate = float64(metrics.AcceptedTokens) / float64(metrics.ProposedTokens)
	}
	if metrics.WallDuration > 0 {
		metrics.VisibleTokensPerSec = float64(mtp.EmittedTokens) / metrics.WallDuration.Seconds()
	}
	return metrics
}

func (model *attachedDrafterTextModel) Err() core.Result {
	if model == nil {
		return core.Ok(nil)
	}
	model.mu.Lock()
	err := model.err
	model.mu.Unlock()
	if err != nil {
		return core.Fail(err)
	}
	target := model.targetModel()
	if target == nil {
		return core.Ok(nil)
	}
	return target.Err()
}

func (model *attachedDrafterTextModel) Close() core.Result {
	if model == nil || model.pair == nil {
		return core.Ok(nil)
	}
	return core.ResultOf(nil, model.pair.Close())
}

func (model *attachedDrafterTextModel) WakeState(ctx context.Context, req inference.AgentMemoryWakeRequest) (*inference.AgentMemoryWakeResult, error) {
	session, err := model.targetStateSession("rocm.AttachedDrafterTextModel.WakeState")
	if err != nil {
		model.setLastFailure(err)
		return nil, err
	}
	return session.WakeState(ctx, req)
}

func (model *attachedDrafterTextModel) SleepState(ctx context.Context, req inference.AgentMemorySleepRequest) (*inference.AgentMemorySleepResult, error) {
	session, err := model.targetStateSession("rocm.AttachedDrafterTextModel.SleepState")
	if err != nil {
		model.setLastFailure(err)
		return nil, err
	}
	return session.SleepState(ctx, req)
}

func (model *attachedDrafterTextModel) targetStateSession(operation string) (inference.AgentMemorySession, error) {
	target := model.targetModel()
	if target == nil {
		return nil, core.E(operation, "target model is required", nil)
	}
	session, ok := target.(inference.AgentMemorySession)
	if !ok || session == nil {
		return nil, core.E(operation, "target model does not implement AgentMemorySession", nil)
	}
	return session, nil
}

func (model *attachedDrafterTextModel) targetModel() inference.TextModel {
	if model == nil || model.pair == nil {
		return nil
	}
	return model.pair.Target
}

func (model *attachedDrafterTextModel) targetROCmModel() *rocmModel {
	base := model.targetModel()
	if base == nil {
		return nil
	}
	target, _ := base.(*rocmModel)
	return target
}

func (model *attachedDrafterTextModel) attachedDrafterLabels(labels map[string]string) map[string]string {
	out := cloneStringMap(labels)
	if out == nil {
		out = map[string]string{}
	}
	if model == nil || model.pair == nil {
		return out
	}
	for key, value := range model.pair.Plan.Labels {
		if value != "" {
			out[key] = value
		}
	}
	for key, value := range model.pair.Attachment.Labels {
		if value != "" {
			out[key] = value
		}
	}
	if model.pair.NativeReady() {
		route := "native_attached"
		if attachedDrafterLabelsDeclareRetainedStateReady(out) {
			route = "native_attached_retained_state"
		}
		out["attached_drafter_generation_route"] = route
		out["attached_drafter_generation_route_reason"] = "target_equivalent_batched_prefill"
	}
	return out
}

func attachedDrafterLabelsDeclareRetainedStateReady(labels map[string]string) bool {
	return labels["attached_drafter_prompt_replay_fallback"] == "forbidden" &&
		labels["attached_drafter_target_retained_decode"] == hipKernelStatusLinked &&
		labels["attached_drafter_target_retained_state_decode"] == hipKernelStatusLinked
}

func (model *attachedDrafterTextModel) clearLastGenerationState() {
	if model == nil {
		return
	}
	model.mu.Lock()
	model.err = nil
	model.metrics = inference.GenerateMetrics{}
	model.mtp = nil
	model.mu.Unlock()
}

func (model *attachedDrafterTextModel) setLastFailure(err error) {
	if model == nil || err == nil {
		return
	}
	model.mu.Lock()
	model.err = err
	model.mu.Unlock()
}

func (model *attachedDrafterTextModel) recordResultMetrics(prompt string, generatedTokens int, decodeMetrics inferdecode.Metrics, start time.Time) {
	if model == nil {
		return
	}
	duration := decodeMetrics.Duration
	if duration <= 0 {
		duration = time.Since(start)
	}
	promptTokens := 0
	if target := model.targetROCmModel(); target != nil {
		promptTokens = target.promptTokenCount(prompt)
	}
	metrics := inference.GenerateMetrics{
		PromptTokens:    promptTokens,
		GeneratedTokens: generatedTokens,
		DecodeDuration:  duration,
		TotalDuration:   duration,
	}
	if duration > 0 {
		metrics.DecodeTokensPerSec = float64(generatedTokens) / duration.Seconds()
	}
	model.mu.Lock()
	model.metrics = metrics
	model.mtp = attachedDrafterMetricsFromDecode(decodeMetrics)
	model.mu.Unlock()
}

func attachedDrafterGenerateConfigFromInference(cfg inference.GenerateConfig, draftTokens int, adaptiveDraftTokens bool) AttachedDrafterGenerateConfig {
	cfg = normalizeAttachedDrafterGreedyConfig(cfg)
	return AttachedDrafterGenerateConfig{
		MaxTokens:           cfg.MaxTokens,
		DraftTokens:         draftTokens,
		AdaptiveDraftTokens: adaptiveDraftTokens,
		Temperature:         cfg.Temperature,
		TopK:                cfg.TopK,
		TopP:                cfg.TopP,
		MinP:                cfg.MinP,
		StopTokens:          append([]int32(nil), cfg.StopTokens...),
		RepeatPenalty:       cfg.RepeatPenalty,
	}
}

func attachedDrafterMTPRequestEligible(cfg inference.GenerateConfig) bool {
	return cfg.RepeatPenalty <= 1
}

func attachedDrafterGenerateOptions(cfg inference.GenerateConfig) []inference.GenerateOption {
	opts := []inference.GenerateOption{
		inference.WithMaxTokens(cfg.MaxTokens),
		inference.WithTemperature(cfg.Temperature),
		inference.WithTopK(cfg.TopK),
		inference.WithTopP(cfg.TopP),
		inference.WithMinP(cfg.MinP),
		inference.WithRepeatPenalty(cfg.RepeatPenalty),
	}
	if len(cfg.StopTokens) > 0 {
		opts = append(opts, inference.WithStopTokens(cfg.StopTokens...))
	}
	if cfg.ReturnLogits {
		opts = append(opts, inference.WithLogits())
	}
	return opts
}

func normalizeAttachedDrafterGreedyConfig(cfg inference.GenerateConfig) inference.GenerateConfig {
	if cfg.Temperature <= 0 {
		cfg.Temperature = 0
		cfg.TopK = 0
		cfg.TopP = 0
		cfg.MinP = 0
	}
	return cfg
}
