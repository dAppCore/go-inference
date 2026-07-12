// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import (
	"context"

	core "dappco.re/go"
	"dappco.re/go/inference"
	inferdecode "dappco.re/go/inference/decode"
)

const (
	defaultROCmPromptLookupMinMatch = 2
	defaultROCmPromptLookupMaxDraft = 16
)

// SpeculativeDecodeConfig configures the ROCm package helper over the shared
// backend-neutral speculative decode harness.
type SpeculativeDecodeConfig struct {
	Prompt      string
	MaxTokens   int
	DraftTokens int
}

// AttachedDrafterDecodeConfig configures the Gemma4 attached-MTP helper over
// the shared backend-neutral speculative decode harness.
type AttachedDrafterDecodeConfig struct {
	Prompt      string
	MaxTokens   int
	DraftTokens int
}

// AttachedDrafterGenerateConfig configures native attached-drafter generation.
// This is intentionally separate from AttachedDrafterDecodeConfig because it
// must not route through the portable prompt-replay speculative helper.
type AttachedDrafterGenerateConfig struct {
	MaxTokens           int
	DraftTokens         int
	AdaptiveDraftTokens bool
	Temperature         float32
	TopK                int
	TopP                float32
	MinP                float32
	StopTokens          []int32
	RepeatPenalty       float32
}

// AttachedDrafterStateGenerateRequest configures native retained-state
// attached-drafter generation. Input is only the new turn text; prior context
// must already be present in State.
type AttachedDrafterStateGenerateRequest struct {
	State               *StateSession
	Input               string
	MaxTokens           int
	DraftTokens         int
	AdaptiveDraftTokens bool
	Temperature         float32
	TopK                int
	TopP                float32
	MinP                float32
	StopTokens          []int32
	RepeatPenalty       float32
}

// AttachedDrafterPlan records the validated Gemma4 target plus assistant
// pairing ROCm can use for attached-MTP benchmark setup.
type AttachedDrafterPlan struct {
	Mode             string
	Target           inference.ModelInfo
	Draft            inference.ModelInfo
	DraftTokens      int
	HelperStatus     string
	NativeAttachment string
	Labels           map[string]string
}

// AttachedDrafterAttachment records a native target+assistant attachment.
// Current ROCm HIP builds validate the pair but report native attachment as
// not_linked until packed assistant kernels are available.
type AttachedDrafterAttachment struct {
	Plan             AttachedDrafterPlan
	Target           inference.ModelInfo
	Draft            inference.ModelInfo
	NativeAttachment string
	Labels           map[string]string
}

// AttachedDrafterPairConfig configures loading a target plus assistant pair.
type AttachedDrafterPairConfig struct {
	TargetOptions    []inference.LoadOption
	DraftOptions     []inference.LoadOption
	TargetROCmConfig ROCmLoadConfig
	DraftROCmConfig  ROCmLoadConfig
}

// AttachedDrafterPair is a validated Gemma4 target plus attached assistant.
// The pair may exist before native HIP attachment is linked; callers must check
// NativeReady before treating it as a production MTP generation path.
type AttachedDrafterPair struct {
	Target      inference.TextModel
	Draft       inference.TextModel
	Plan        AttachedDrafterPlan
	Attachment  AttachedDrafterAttachment
	NativeError string

	ownsTarget bool
	ownsDraft  bool
}

// PromptLookupDecodeConfig configures the ROCm package helper over the shared
// backend-neutral prompt-lookup decode harness.
type PromptLookupDecodeConfig struct {
	Prompt       string
	MaxTokens    int
	LookupTokens []int32
	MinMatch     int
	MaxDraft     int
}

// SpeculativeDecode compares draft model output against target model output
// using the shared go-inference/decode acceptance algorithm. It is a package
// helper; it does not imply production ROCm decode kernels are linked.
func SpeculativeDecode(ctx context.Context, target, draft inference.TextModel, cfg SpeculativeDecodeConfig) (inferdecode.Result, error) {
	if target == nil {
		return inferdecode.Result{}, core.E("rocm.SpeculativeDecode", "target model is required", nil)
	}
	if draft == nil {
		return inferdecode.Result{}, core.E("rocm.SpeculativeDecode", "draft model is required", nil)
	}
	maxTokens, err := rocmDecodeMaxTokens(target, cfg.Prompt, cfg.MaxTokens, "rocm.SpeculativeDecode")
	if err != nil {
		return inferdecode.Result{}, err
	}
	return inferdecode.Speculative(ctx, inferdecode.SpeculativeConfig{
		Prompt:         cfg.Prompt,
		MaxTokens:      maxTokens,
		DraftTokens:    cfg.DraftTokens,
		TargetGenerate: rocmDecodeGenerator{model: target},
		DraftGenerate:  rocmDecodeGenerator{model: draft},
	})
}

// AttachedDrafterDecode runs speculative decoding for a Gemma4 target plus a
// Gemma4 assistant pack. The architecture checks keep the attached-MTP path
// explicit while reusing the shared acceptance harness for metrics.
func AttachedDrafterDecode(ctx context.Context, target, draft inference.TextModel, cfg AttachedDrafterDecodeConfig) (inferdecode.Result, error) {
	if _, err := PlanAttachedDrafter(target, draft); err != nil {
		return inferdecode.Result{}, core.E("rocm.AttachedDrafterDecode", "attached drafter pair is invalid", err)
	}
	return SpeculativeDecode(ctx, target, draft, SpeculativeDecodeConfig{
		Prompt:      cfg.Prompt,
		MaxTokens:   cfg.MaxTokens,
		DraftTokens: cfg.DraftTokens,
	})
}

// PlanAttachedDrafter validates a Gemma4 target plus Gemma4 assistant MTP
// drafter pair without generating, replaying prompts, or attaching native HIP
// state. Native attachment remains explicit until those kernels are linked.
func PlanAttachedDrafter(target, draft inference.TextModel) (AttachedDrafterPlan, error) {
	if target == nil {
		return AttachedDrafterPlan{}, core.E("rocm.PlanAttachedDrafter", "target model is required", nil)
	}
	if draft == nil {
		return AttachedDrafterPlan{}, core.E("rocm.PlanAttachedDrafter", "draft model is required", nil)
	}
	targetIdentity := rocmDecodeModelIdentity(target)
	draftIdentity := rocmDecodeModelIdentity(draft)
	targetInfo := rocmModelInfoFromIdentity(targetIdentity)
	draftInfo := rocmModelInfoFromIdentity(draftIdentity)
	if !isROCmGemma4Architecture(targetInfo.Architecture) {
		return AttachedDrafterPlan{}, core.E("rocm.PlanAttachedDrafter", "target model must be a Gemma4 text model", nil)
	}
	if !isROCmGemma4AssistantArchitecture(draftInfo.Architecture) {
		return AttachedDrafterPlan{}, core.E("rocm.PlanAttachedDrafter", "draft model must be a Gemma4 assistant attached MTP drafter", nil)
	}
	if err := checkROCmGemma4AttachedDrafterTargetIdentity("rocm.PlanAttachedDrafter", targetIdentity); err != nil {
		return AttachedDrafterPlan{}, err
	}
	if err := checkROCmGemma4AttachedDrafterAssistantIdentity("rocm.PlanAttachedDrafter", draftIdentity); err != nil {
		return AttachedDrafterPlan{}, err
	}
	if err := checkROCmGemma4AttachedDrafterFamilyPair("rocm.PlanAttachedDrafter", targetIdentity, draftIdentity); err != nil {
		return AttachedDrafterPlan{}, err
	}
	policy := DefaultProductionMTPPolicy()
	labels := map[string]string{
		"mode":                         policy.Mode,
		"production_default_candidate": boolLabel(policy.EnabledByDefault),
	}
	rocmAddGemma4AttachedDrafterCapabilityLabels(labels, targetIdentity, draftIdentity)
	return AttachedDrafterPlan{
		Mode:             policy.Mode,
		Target:           targetInfo,
		Draft:            draftInfo,
		DraftTokens:      policy.DefaultDraftTokens,
		HelperStatus:     hipKernelStatusLinked,
		NativeAttachment: hipKernelStatusNotLinked,
		Labels:           labels,
	}, nil
}

// AttachNativeDrafter validates a Gemma4 target plus Gemma4 assistant pair and
// attempts the native HIP attachment path. It never falls back to prompt replay
// or package-level speculative decoding.
func AttachNativeDrafter(target, draft inference.TextModel) (AttachedDrafterAttachment, error) {
	plan, err := PlanAttachedDrafter(target, draft)
	if err != nil {
		return AttachedDrafterAttachment{}, core.E("rocm.AttachNativeDrafter", "attached drafter pair is invalid", err)
	}
	targetModel, targetOK := target.(*rocmModel)
	draftModel, draftOK := draft.(*rocmModel)
	if !targetOK || targetModel == nil || targetModel.native == nil || !draftOK || draftModel == nil || draftModel.native == nil {
		return AttachedDrafterAttachment{}, core.E("rocm.AttachNativeDrafter", "native ROCm target and draft models are required", nil)
	}
	attacher, ok := targetModel.native.(nativeAttachedDrafterTarget)
	if !ok {
		return AttachedDrafterAttachment{}, core.E("rocm.AttachNativeDrafter", "native HIP drafter attachment is not linked for this target runtime", nil)
	}
	attachment, err := attacher.AttachAttachedDrafter(draftModel.native, plan)
	if err != nil {
		return attachment, core.E("rocm.AttachNativeDrafter", "native HIP drafter attachment", err)
	}
	return attachment, nil
}

// NewAttachedDrafterPair validates an already-loaded Gemma4 target plus
// assistant. It records the native attachment status but does not fall back to
// prompt replay when HIP attachment is not linked.
func NewAttachedDrafterPair(target, draft inference.TextModel) (*AttachedDrafterPair, error) {
	plan, err := PlanAttachedDrafter(target, draft)
	if err != nil {
		return nil, core.E("rocm.NewAttachedDrafterPair", "plan attached drafter", err)
	}
	pair := &AttachedDrafterPair{
		Target: target,
		Draft:  draft,
		Plan:   plan,
		Attachment: AttachedDrafterAttachment{
			Plan:             plan,
			Target:           plan.Target,
			Draft:            plan.Draft,
			NativeAttachment: plan.NativeAttachment,
			Labels:           cloneStringMap(plan.Labels),
		},
	}
	attachment, attachErr := AttachNativeDrafter(target, draft)
	if attachErr == nil {
		pair.Attachment = cloneAttachedDrafterAttachment(attachment)
		return pair, nil
	}
	if attachment.NativeAttachment != hipKernelStatusNotLinked || !rocmIsNativeDrafterNotLinkedError(attachErr) {
		return nil, core.E("rocm.NewAttachedDrafterPair", "attach native drafter", attachErr)
	}
	pair.Attachment = cloneAttachedDrafterAttachment(attachment)
	pair.NativeError = attachErr.Error()
	return pair, nil
}

// LoadAttachedDrafterPair loads and validates a Gemma4 target plus assistant
// pair. On validation failure it closes any model it loaded.
func LoadAttachedDrafterPair(targetPath, draftPath string, cfg AttachedDrafterPairConfig) (*AttachedDrafterPair, error) {
	return (&rocmBackend{}).LoadAttachedDrafterPair(targetPath, draftPath, cfg)
}

func (b *rocmBackend) LoadAttachedDrafterPair(targetPath, draftPath string, cfg AttachedDrafterPairConfig) (*AttachedDrafterPair, error) {
	targetPath = core.Trim(targetPath)
	if targetPath == "" {
		return nil, core.E("rocm.LoadAttachedDrafterPair", "target path is required", nil)
	}
	draftPath = core.Trim(draftPath)
	if draftPath == "" {
		return nil, core.E("rocm.LoadAttachedDrafterPair", "draft path is required", nil)
	}
	target, err := b.loadAttachedDrafterModel(targetPath, cfg.TargetROCmConfig, cfg.TargetOptions, false)
	if err != nil {
		return nil, core.E("rocm.LoadAttachedDrafterPair", "load target", err)
	}
	draft, err := b.loadAttachedDrafterModel(draftPath, cfg.DraftROCmConfig, cfg.DraftOptions, true)
	if err != nil {
		if closeErr := target.Close(); !closeErr.OK {
			err = core.ErrorJoin(err, closeErr.Value.(error))
		}
		return nil, core.E("rocm.LoadAttachedDrafterPair", "load draft", err)
	}
	pair, err := NewAttachedDrafterPair(target, draft)
	if err != nil {
		if closeErr := target.Close(); !closeErr.OK {
			err = core.ErrorJoin(err, closeErr.Value.(error))
		}
		if closeErr := draft.Close(); !closeErr.OK {
			err = core.ErrorJoin(err, closeErr.Value.(error))
		}
		return nil, core.E("rocm.LoadAttachedDrafterPair", "validate pair", err)
	}
	pair.ownsTarget = true
	pair.ownsDraft = true
	return pair, nil
}

func (b *rocmBackend) loadAttachedDrafterModel(path string, cfg ROCmLoadConfig, opts []inference.LoadOption, allowAttachedOnly bool) (inference.TextModel, error) {
	return b.loadModelWithROCmConfigMode(path, inference.ApplyLoadOpts(opts), cfg, allowAttachedOnly)
}

func (pair *AttachedDrafterPair) NativeReady() bool {
	return pair != nil && pair.Attachment.NativeAttachment == hipKernelStatusLinked && pair.NativeError == ""
}

// GenerateNative runs the native attached-drafter generation path. It refuses
// to use target/draft Generate fallback paths when native HIP attachment is not
// linked.
func (pair *AttachedDrafterPair) GenerateNative(ctx context.Context, prompt string, cfg AttachedDrafterGenerateConfig) (inferdecode.Result, error) {
	if pair == nil {
		return inferdecode.Result{}, core.E("rocm.AttachedDrafterPair.GenerateNative", "pair is required", nil)
	}
	if !pair.NativeReady() {
		message := "native HIP drafter generation is not linked yet"
		if pair.NativeError != "" {
			message += ": " + pair.NativeError
		}
		return inferdecode.Result{}, core.E("rocm.AttachedDrafterPair.GenerateNative", message, nil)
	}
	target, ok := pair.Target.(*rocmModel)
	if !ok || target == nil || target.native == nil {
		return inferdecode.Result{}, core.E("rocm.AttachedDrafterPair.GenerateNative", "native ROCm target model is required", nil)
	}
	generator, ok := target.native.(nativeAttachedDrafterGenerator)
	if !ok {
		return inferdecode.Result{}, core.E("rocm.AttachedDrafterPair.GenerateNative", "native HIP drafter generation is not linked for this target runtime", nil)
	}
	maxTokens, err := rocmDecodeMaxTokens(target, prompt, cfg.MaxTokens, "rocm.AttachedDrafterPair.GenerateNative")
	if err != nil {
		return inferdecode.Result{}, err
	}
	cfg.MaxTokens = maxTokens
	if cfg.DraftTokens <= 0 {
		cfg.DraftTokens = pair.Plan.DraftTokens
		cfg.AdaptiveDraftTokens = true
	}
	cfg.StopTokens = append([]int32(nil), cfg.StopTokens...)
	return generator.GenerateAttachedDrafter(ctx, cloneAttachedDrafterAttachment(pair.Attachment), prompt, cfg)
}

// GenerateNativeWithStateRetention runs full-prompt native attached-drafter
// generation and retains the resulting target KV state for a later continuation.
func (pair *AttachedDrafterPair) GenerateNativeWithStateRetention(ctx context.Context, state *StateSession, prompt string, cfg AttachedDrafterGenerateConfig) (inferdecode.Result, error) {
	if pair == nil {
		return inferdecode.Result{}, core.E("rocm.AttachedDrafterPair.GenerateNativeWithStateRetention", "pair is required", nil)
	}
	if state == nil {
		return inferdecode.Result{}, core.E("rocm.AttachedDrafterPair.GenerateNativeWithStateRetention", "state session is required", nil)
	}
	target, ok := pair.Target.(*rocmModel)
	if !ok || target == nil || target.native == nil {
		return inferdecode.Result{}, core.E("rocm.AttachedDrafterPair.GenerateNativeWithStateRetention", "native ROCm target model is required", nil)
	}
	if !pair.NativeReady() {
		message := "native HIP drafter generation is not linked yet"
		if pair.NativeError != "" {
			message += ": " + pair.NativeError
		}
		return inferdecode.Result{}, core.E("rocm.AttachedDrafterPair.GenerateNativeWithStateRetention", message, nil)
	}
	generator, ok := target.native.(nativeAttachedDrafterStateRetainingGenerator)
	if !ok {
		return inferdecode.Result{}, core.E("rocm.AttachedDrafterPair.GenerateNativeWithStateRetention", "native HIP drafter state retention is not linked for this target runtime", nil)
	}
	maxTokens, err := rocmDecodeMaxTokens(target, prompt, cfg.MaxTokens, "rocm.AttachedDrafterPair.GenerateNativeWithStateRetention")
	if err != nil {
		return inferdecode.Result{}, err
	}
	cfg.MaxTokens = maxTokens
	if cfg.DraftTokens <= 0 {
		cfg.DraftTokens = pair.Plan.DraftTokens
		cfg.AdaptiveDraftTokens = true
	}
	cfg.StopTokens = append([]int32(nil), cfg.StopTokens...)
	return generator.GenerateAttachedDrafterWithStateRetention(ctx, cloneAttachedDrafterAttachment(pair.Attachment), prompt, cfg, state)
}

// GenerateNativeRetained runs native attached-drafter generation against the
// target model's restored ROCm KV state. The input is only the new turn text.
func (pair *AttachedDrafterPair) GenerateNativeRetained(ctx context.Context, input string, cfg AttachedDrafterGenerateConfig) (inferdecode.Result, error) {
	if pair == nil {
		return inferdecode.Result{}, core.E("rocm.AttachedDrafterPair.GenerateNativeRetained", "pair is required", nil)
	}
	target, ok := pair.Target.(*rocmModel)
	if !ok || target == nil {
		return inferdecode.Result{}, core.E("rocm.AttachedDrafterPair.GenerateNativeRetained", "native ROCm target model is required", nil)
	}
	state := target.currentStateSession()
	return pair.GenerateNativeFromState(ctx, AttachedDrafterStateGenerateRequest{
		State:               state,
		Input:               input,
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

// GenerateNativeFromState runs native attached-drafter generation against a
// restored ROCm KV state. It refuses missing or metadata-only state so callers
// cannot replay historical prompt text as a fallback.
func (pair *AttachedDrafterPair) GenerateNativeFromState(ctx context.Context, req AttachedDrafterStateGenerateRequest) (inferdecode.Result, error) {
	if pair == nil {
		return inferdecode.Result{}, core.E("rocm.AttachedDrafterPair.GenerateNativeFromState", "pair is required", nil)
	}
	target, ok := pair.Target.(*rocmModel)
	if !ok || target == nil || target.native == nil {
		return inferdecode.Result{}, core.E("rocm.AttachedDrafterPair.GenerateNativeFromState", "native ROCm target model is required", nil)
	}
	if req.State == nil {
		return inferdecode.Result{}, core.E("rocm.AttachedDrafterPair.GenerateNativeFromState", "runtime-owned KV state is required", nil)
	}
	if !rocmStateSessionHasRuntimeKV(req.State) {
		return inferdecode.Result{}, core.E("rocm.AttachedDrafterPair.GenerateNativeFromState", "runtime-owned KV state is required; refusing prompt replay", nil)
	}
	if err := checkROCmStateModelCompatibility("rocm.AttachedDrafterPair.GenerateNativeFromState", target.modelIdentity(), req.State.model); err != nil {
		return inferdecode.Result{}, err
	}
	if !pair.NativeReady() {
		message := "native HIP drafter generation is not linked yet"
		if pair.NativeError != "" {
			message += ": " + pair.NativeError
		}
		return inferdecode.Result{}, core.E("rocm.AttachedDrafterPair.GenerateNativeFromState", message, nil)
	}
	generator, ok := target.native.(nativeAttachedDrafterStateGenerator)
	if !ok {
		return inferdecode.Result{}, core.E("rocm.AttachedDrafterPair.GenerateNativeFromState", "native HIP retained-state drafter generation is not linked for this target runtime", nil)
	}
	maxTokens, err := rocmAttachedDrafterStateMaxTokens(target, req.State, req.Input, req.MaxTokens, "rocm.AttachedDrafterPair.GenerateNativeFromState")
	if err != nil {
		return inferdecode.Result{}, err
	}
	req.MaxTokens = maxTokens
	if req.DraftTokens <= 0 {
		req.DraftTokens = pair.Plan.DraftTokens
		req.AdaptiveDraftTokens = true
	}
	req.StopTokens = append([]int32(nil), req.StopTokens...)
	return generator.GenerateAttachedDrafterFromState(ctx, pair.Attachment, req)
}

func (pair *AttachedDrafterPair) Close() error {
	if pair == nil {
		return nil
	}
	var err error
	if pair.ownsDraft && pair.Draft != nil && pair.Draft != pair.Target {
		err = core.ErrorJoin(err, resultError(pair.Draft.Close()))
	}
	if pair.ownsTarget && pair.Target != nil {
		err = core.ErrorJoin(err, resultError(pair.Target.Close()))
	}
	pair.Target = nil
	pair.Draft = nil
	return err
}

type nativeAttachedDrafterTarget interface {
	AttachAttachedDrafter(draft nativeModel, plan AttachedDrafterPlan) (AttachedDrafterAttachment, error)
}

type nativeAttachedDrafterGenerator interface {
	GenerateAttachedDrafter(ctx context.Context, attachment AttachedDrafterAttachment, prompt string, cfg AttachedDrafterGenerateConfig) (inferdecode.Result, error)
}

type nativeAttachedDrafterStateRetainingGenerator interface {
	GenerateAttachedDrafterWithStateRetention(ctx context.Context, attachment AttachedDrafterAttachment, prompt string, cfg AttachedDrafterGenerateConfig, state *StateSession) (inferdecode.Result, error)
}

type nativeAttachedDrafterStateGenerator interface {
	// GenerateAttachedDrafterFromState receives immutable attachment metadata.
	// Retained generation must not mutate the attachment or replay prompt text.
	GenerateAttachedDrafterFromState(ctx context.Context, attachment AttachedDrafterAttachment, req AttachedDrafterStateGenerateRequest) (inferdecode.Result, error)
}

func rocmStateSessionHasRuntimeKV(session *StateSession) bool {
	if session == nil || session.runtime == nil {
		return false
	}
	tokens, ok := rocmStateSessionRuntimeTokenCount(session)
	return ok && tokens > 0
}

func rocmStateSessionRuntimeTokenCount(session *StateSession) (int, bool) {
	if session == nil || session.runtime == nil {
		return 0, false
	}
	switch runtime := session.runtime.(type) {
	case *rocmKVCache:
		if runtime == nil {
			return 0, false
		}
		return runtime.TokenCount(), true
	case *rocmDeviceKVCache:
		if runtime == nil || runtime.closed {
			return 0, false
		}
		return runtime.TokenCount(), true
	case *hipGemma4Q4DeviceDecodeState:
		if runtime == nil || runtime.closed {
			return 0, false
		}
		return runtime.maxLayerTokenCount(), true
	case *hipGemma4Q4HostDecodeStateRuntime:
		if runtime == nil {
			return 0, false
		}
		return runtime.tokenCount, true
	default:
		return 0, false
	}
}

func (m *rocmModel) currentStateSession() *StateSession {
	if m == nil {
		return nil
	}
	m.stateMutex.Lock()
	defer m.stateMutex.Unlock()
	return m.state
}

func rocmIsNativeDrafterNotLinkedError(err error) bool {
	return err != nil && core.Contains(err.Error(), "native HIP drafter attachment is not linked")
}

func cloneAttachedDrafterAttachment(attachment AttachedDrafterAttachment) AttachedDrafterAttachment {
	attachment.Target = rocmNormalizeModelInfo(attachment.Target)
	attachment.Draft = rocmNormalizeModelInfo(attachment.Draft)
	attachment.Labels = cloneStringMap(attachment.Labels)
	attachment.Plan.Labels = cloneStringMap(attachment.Plan.Labels)
	return attachment
}

// PromptLookupDecode derives or accepts prompt-lookup candidates and compares
// them against target model output using the shared go-inference/decode
// acceptance algorithm.
func PromptLookupDecode(ctx context.Context, target inference.TextModel, cfg PromptLookupDecodeConfig) (inferdecode.Result, error) {
	if target == nil {
		return inferdecode.Result{}, core.E("rocm.PromptLookupDecode", "target model is required", nil)
	}
	lookupTokens, err := rocmPromptLookupTokens(target, cfg)
	if err != nil {
		return inferdecode.Result{}, err
	}
	maxTokens, err := rocmDecodeMaxTokens(target, cfg.Prompt, cfg.MaxTokens, "rocm.PromptLookupDecode")
	if err != nil {
		return inferdecode.Result{}, err
	}
	return inferdecode.PromptLookup(ctx, inferdecode.PromptLookupConfig{
		Prompt:         cfg.Prompt,
		MaxTokens:      maxTokens,
		LookupTokens:   lookupTokens,
		TargetGenerate: rocmDecodeGenerator{model: target},
	})
}

type rocmDecodeGenerator struct {
	model inference.TextModel
}

func (generator rocmDecodeGenerator) Generate(ctx context.Context, prompt string, cfg inferdecode.GenerateConfig) (inferdecode.Generation, error) {
	if generator.model == nil {
		return inferdecode.Generation{}, core.E("rocm.Decode.Generate", "model is required", nil)
	}
	var opts []inference.GenerateOption
	if cfg.MaxTokens > 0 {
		opts = append(opts, inference.WithMaxTokens(cfg.MaxTokens))
	}
	tokens := []inferdecode.Token{}
	for token := range generator.model.Generate(ctx, prompt, opts...) {
		tokens = append(tokens, rocmDecodeToken(token))
	}
	if r := generator.model.Err(); !r.OK {
		return inferdecode.Generation{}, core.E("rocm.Decode.Generate", "model generation failed", r.Value.(error))
	}
	return inferdecode.Generation{Tokens: tokens, Text: inferdecode.TokensText(tokens)}, nil
}

func rocmPromptLookupTokens(model inference.TextModel, cfg PromptLookupDecodeConfig) ([]inferdecode.Token, error) {
	tokenIDs := append([]int32(nil), cfg.LookupTokens...)
	if len(tokenIDs) == 0 {
		encoder, ok := model.(interface {
			Encode(string) []int32
		})
		if !ok {
			return nil, core.E("rocm.PromptLookupDecode", "lookup tokens are required when model does not expose Encode", nil)
		}
		minMatch := cfg.MinMatch
		if minMatch <= 0 {
			minMatch = defaultROCmPromptLookupMinMatch
		}
		promptTokens := encoder.Encode(cfg.Prompt)
		maxDraft, err := rocmPromptLookupMaxDraft(model, cfg, promptTokens)
		if err != nil {
			return nil, err
		}
		tokenIDs, err = rocmReferencePromptLookupDraft(promptTokens, minMatch, maxDraft)
		if err != nil {
			return nil, err
		}
	}
	return rocmDecodeTokens(model, tokenIDs), nil
}

func rocmPromptLookupMaxDraft(model inference.TextModel, cfg PromptLookupDecodeConfig, promptTokens []int32) (int, error) {
	requested := 0
	if cfg.MaxDraft > 0 {
		requested = cfg.MaxDraft
	} else if cfg.MaxTokens > 0 {
		requested = cfg.MaxTokens
	}
	rocmModel, ok := model.(*rocmModel)
	if !ok || rocmModel == nil || !isROCmGemma4Architecture(rocmModel.modelIdentity().Architecture) {
		if requested > 0 {
			return requested, nil
		}
		return defaultROCmPromptLookupMaxDraft, nil
	}
	contextLength := rocmModel.modelIdentity().ContextLength
	if contextLength <= 0 {
		contextLength = defaultContextLengthCap
	}
	remaining := contextLength - len(promptTokens)
	if remaining <= 0 {
		return 0, core.E("rocm.PromptLookupDecode", "prompt reaches model context window", nil)
	}
	if requested > 0 {
		if requested > remaining {
			return 0, core.E("rocm.PromptLookupDecode", "max tokens exceed remaining model context window", nil)
		}
		return requested, nil
	}
	return remaining, nil
}

func rocmDecodeMaxTokens(model inference.TextModel, prompt string, requested int, operation string) (int, error) {
	if !rocmDecodeIsGemma4Target(model) {
		return requested, nil
	}
	contextLength := defaultContextLengthCap
	if rocmModel, ok := model.(*rocmModel); ok && rocmModel != nil {
		if identityContext := rocmModel.modelIdentity().ContextLength; identityContext > 0 {
			contextLength = identityContext
		}
	}
	promptTokens := rocmDecodePromptTokenCount(model, prompt)
	remaining := contextLength - promptTokens
	if remaining <= 0 {
		return 0, core.E(operation, "prompt reaches model context window", nil)
	}
	if requested > 0 {
		if requested > remaining {
			return 0, core.E(operation, "max tokens exceed remaining model context window", nil)
		}
		return requested, nil
	}
	return remaining, nil
}

func rocmAttachedDrafterStateMaxTokens(model *rocmModel, state *StateSession, input string, requested int, operation string) (int, error) {
	if model == nil || !isROCmGemma4Architecture(model.modelIdentity().Architecture) {
		return requested, nil
	}
	contextLength := model.modelIdentity().ContextLength
	if contextLength <= 0 {
		contextLength = defaultContextLengthCap
	}
	stateTokens, ok := rocmStateSessionRuntimeTokenCount(state)
	if !ok {
		return 0, core.E(operation, "runtime-owned KV state is required", nil)
	}
	inputTokens := rocmDecodePromptTokenCount(model, input)
	remaining := contextLength - stateTokens - inputTokens
	if remaining <= 0 {
		return 0, core.E(operation, "state and input reach model context window", nil)
	}
	if requested > 0 {
		if requested > remaining {
			return 0, core.E(operation, "max tokens exceed remaining model context window", nil)
		}
		return requested, nil
	}
	return remaining, nil
}

func rocmDecodePromptTokenCount(model inference.TextModel, prompt string) int {
	if rocmModel, ok := model.(*rocmModel); ok && rocmModel != nil {
		return rocmModel.promptTokenCount(prompt)
	}
	encoder, ok := model.(interface {
		Encode(string) []int32
	})
	if ok {
		return len(encoder.Encode(prompt))
	}
	return len(approximateTokenIDs(prompt))
}

func rocmDecodeTokens(model inference.TextModel, ids []int32) []inferdecode.Token {
	out := make([]inferdecode.Token, len(ids))
	decoder, _ := model.(interface {
		Decode([]int32) string
	})
	for i, id := range ids {
		text := core.Sprintf("%d", id)
		if decoder != nil {
			if decoded := decoder.Decode([]int32{id}); decoded != "" {
				text = decoded
			}
		}
		out[i] = inferdecode.Token{ID: id, Text: text}
	}
	return out
}

func rocmDecodeToken(token inference.Token) inferdecode.Token {
	return inferdecode.Token{ID: token.ID, Text: token.Text}
}

func rocmDecodeIsGemma4Target(model inference.TextModel) bool {
	return isROCmGemma4Architecture(rocmDecodeModelIdentity(model).Architecture)
}

func rocmDecodeIsGemma4AssistantDrafter(model inference.TextModel) bool {
	return isROCmGemma4AssistantArchitecture(rocmDecodeModelIdentity(model).Architecture)
}

func rocmDecodeModelInfo(model inference.TextModel) inference.ModelInfo {
	if model == nil {
		return inference.ModelInfo{}
	}
	info := model.Info()
	if info.Architecture == "" {
		info.Architecture = model.ModelType()
	}
	return rocmNormalizeModelInfo(info)
}

func rocmDecodeModelIdentity(model inference.TextModel) inference.ModelIdentity {
	if model == nil {
		return inference.ModelIdentity{}
	}
	if reporter, ok := model.(ROCmModelProfileReporter); ok {
		profile := reporter.ModelProfile()
		identity := profile.Model
		if identity.Architecture == "" {
			identity.Architecture = profile.Architecture
		}
		if !rocmModelIdentityIsZero(identity) {
			identity = rocmCloneModelIdentity(identity)
			identity.Architecture = normalizeROCmArchitecture(identity.Architecture)
			return rocmGemma4ModelWithInferredPathQuant(identity)
		}
	}
	if reporter, ok := model.(ROCmModelIdentityReporter); ok {
		identity := reporter.ModelIdentity()
		if !rocmModelIdentityIsZero(identity) {
			identity = rocmCloneModelIdentity(identity)
			identity.Architecture = normalizeROCmArchitecture(identity.Architecture)
			return rocmGemma4ModelWithInferredPathQuant(identity)
		}
	}
	info := rocmDecodeModelInfo(model)
	identity := inference.ModelIdentity{
		Architecture: info.Architecture,
		QuantBits:    info.QuantBits,
		QuantGroup:   info.QuantGroup,
		NumLayers:    info.NumLayers,
		HiddenSize:   info.HiddenSize,
		VocabSize:    info.VocabSize,
	}
	identity.Architecture = normalizeROCmArchitecture(identity.Architecture)
	return rocmGemma4ModelWithInferredPathQuant(identity)
}

func rocmModelInfoFromIdentity(identity inference.ModelIdentity) inference.ModelInfo {
	return inference.ModelInfo{
		Architecture: identity.Architecture,
		VocabSize:    identity.VocabSize,
		NumLayers:    identity.NumLayers,
		HiddenSize:   identity.HiddenSize,
		QuantBits:    identity.QuantBits,
		QuantGroup:   identity.QuantGroup,
	}
}

func rocmNormalizeModelInfo(info inference.ModelInfo) inference.ModelInfo {
	info.Architecture = normalizeROCmArchitecture(info.Architecture)
	return info
}

func boolLabel(value bool) string {
	if value {
		return "true"
	}
	return "false"
}
