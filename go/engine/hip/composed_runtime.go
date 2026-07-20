// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import (
	"context"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/inference/engine"
	"dappco.re/go/inference/kv"
	sharedmodel "dappco.re/go/inference/model"
)

// rocmRetiredComposedArchDecline reports whether a model_type is one the retired composed engine's
// fallback used to catch for HIP — an architecture with NO native ROCm execution path (#50). HIP's
// native runtime only ever consumes nativeLoadConfig (its own GGUF/safetensors-tensor-shaped
// struct); nothing converts a *model.LoadedModel (the shared factory route's reactive Assemble
// output) into one, and HIP has no gated-delta / MoE / MLA forward kernels. Letting one of these
// checkpoints fall through unchecked risks the native pipeline silently misreading a hybrid/MoE/MLA
// tensor layout as a plain dense transformer — a coherent-but-wrong decode, not a clean failure. So
// they decline HERE, named. The set is HIP's own capability statement (engine/hip/profile's table is
// proven insufficient as a stand-in — it lists several of these as "supported" for INSPECTION only):
// the qwen gated-delta hybrid family + its MTP drafter ids, the MoE arch zoo (each with a factory
// port on metal — HIP has none), the MLA/OCR refusal families, and the retired generic ids. Building
// a native HIP route for any of these (the staged, unwired IsQwen36Hybrid guard in dense_config.go
// was written for exactly that) removes it from this set.
func rocmRetiredComposedArchDecline(modelType string) bool {
	switch modelType {
	case "qwen3_5", "qwen3_5_text", "qwen3_5_moe", "qwen3_5_moe_text",
		"qwen3_6", "qwen3_6_moe", "qwen3_next",
		"qwen3_5_mtp", "qwen3_5_mtp_text", "qwen3_6_mtp",
		"mixtral", "dbrx", "olmoe", "granitemoe", "qwen2_moe", "qwen3_moe", "jetmoe",
		"llama4", "llama4_text",
		"deepseek_v2", "deepseek_v3", "deepseek_vl_v2",
		"glm_ocr", "glm_ocr_text", "dots_ocr", "dots_ocr_1_5",
		"composed", "hybrid":
		return true
	}
	return false
}

// loadHIPComposedTextModel used to detour any config-composed architecture through the retired
// composed engine's LoadComposedDir, ahead of HIP's own GGUF/safetensors native loader. #50 retired
// that engine; this function no longer imports, or calls into, it. It still probes config.json —
// through the shared model.ProbeDirArch/ProbeModelTypes — so a checkpoint that WOULD have matched
// gets a clean, NAMED decline (rocmRetiredComposedArchDecline above) instead of silently falling
// into HIP's regular native pipeline. The multimodal-wrapper fallback (top-level model_type unknown,
// nested text_config.model_type carries the family) resolves exactly as the retired registry
// resolution did.
func loadHIPComposedTextModel(path string, _ inference.LoadConfig) (inference.TextModel, bool, error) {
	stat := core.Stat(path)
	if !stat.OK {
		return nil, false, nil
	}
	info, ok := stat.Value.(core.FsFileInfo)
	if !ok || !info.IsDir() {
		return nil, false, nil
	}
	modelType, configJSON, err := sharedmodel.ProbeDirArch(path)
	if err != nil {
		// A directory whose config.json can't be read is not this function's concern to report —
		// mirrors loadHIPMamba2TextModel's identical sibling check (mamba2_runtime.go), which defers
		// the same failure to whichever loader in native.go's pipeline runs next.
		return nil, false, nil
	}
	declared := modelType
	if !rocmRetiredComposedArchDecline(declared) {
		// Multimodal wrapper fallback: the top-level model_type is not a declined family, but a
		// nested text_config.model_type may carry one.
		if _, textModelType := sharedmodel.ProbeModelTypes(configJSON); textModelType != "" && rocmRetiredComposedArchDecline(textModelType) {
			declared = textModelType
		} else {
			return nil, false, nil
		}
	}
	return nil, true, core.NewError("rocm.LoadModel: " + declared + " is a config-composed/hybrid architecture with no native ROCm execution path — the composed-engine fallback that used to serve it is retired (#50)")
}

// hipComposedTextModel bridges a backend-neutral, token-prefix sharedmodel.SessionModel into HIP's
// shared engine serving surface (OpenEngineSession → hipComposedEngineSession). Despite the name, it is
// no longer composed-specific: loadHIPComposedTextModel (above) retired its use of the composed engine
// (#50), so the only remaining constructor is mamba2_runtime.go's loadHIPMamba2TextModel, wrapping a
// model/arch/mamba2 recurrent SessionModel. The type keeps its historical name because
// mamba2_runtime.go (outside this sever's scope) already references it by that name — a rename is a
// follow-up, not part of this change. The model owns host f32 weights; its projection hooks remain
// available for portable HIP++ acceleration.
type hipComposedTextModel struct {
	model            sharedmodel.SessionModel
	tokenizer        engine.TextTokenizer
	modelType        string
	numLayers        int
	declaredStops    []int32
	declaredSampling engine.SamplingDefaults
}

var (
	_ engine.TokenModel               = (*hipComposedTextModel)(nil)
	_ engine.ChatTemplateDeclarer     = (*hipComposedTextModel)(nil)
	_ engine.StopTokenDeclarer        = (*hipComposedTextModel)(nil)
	_ engine.SamplingDefaultsDeclarer = (*hipComposedTextModel)(nil)
	_ engine.MemoryReporter           = (*hipComposedTextModel)(nil)
)

func (model *hipComposedTextModel) OpenEngineSession() (engine.Session, error) {
	if model == nil || model.model == nil {
		return nil, core.NewError("hip.composed: model is not initialised")
	}
	return &hipComposedEngineSession{
		model:        model.model,
		architecture: model.modelType,
		numLayers:    model.numLayers,
	}, nil
}

func (*hipComposedTextModel) Close() error { return nil }

// DeclaredChatTemplate reports the ChatML template for a qwen-family architecture and the Gemma
// template otherwise, resolved through HIP's own architecture-profile registry
// (hipArchitectureChatTemplate/ROCmChatTemplateID, inference_model.go + profile/architecture.go) rather
// than the retired composed engine's ChatMLDialect helper (#50).
func (model *hipComposedTextModel) DeclaredChatTemplate() (engine.ChatTemplate, bool) {
	if model == nil {
		return engine.ChatTemplate{}, false
	}
	if template, ok := hipArchitectureChatTemplate(model.modelType); ok {
		return template, true
	}
	return engine.GemmaChatTemplate(engine.DetectTurnTokens(model.tokenizer), false), true
}

func (model *hipComposedTextModel) DeclaredStopTokens() []int32 {
	if model == nil {
		return nil
	}
	return append([]int32(nil), model.declaredStops...)
}

func (model *hipComposedTextModel) DeclaredSamplingDefaults() engine.SamplingDefaults {
	if model == nil {
		return engine.SamplingDefaults{}
	}
	return model.declaredSampling
}

// The portable composed path owns host f32 slices and no persistent HIP device
// allocation. Report zero until a HIP++ projection hook acquires device memory.
func (*hipComposedTextModel) ActiveMemoryBytes() uint64 { return 0 }

func (*hipComposedTextModel) PeakMemoryBytes() uint64 { return 0 }

// hipComposedEngineSession retains a token-prefix-shaped model (today: Mamba2, via
// mamba2_runtime.go's loadHIPMamba2TextModel — see hipComposedTextModel's doc comment) as its complete
// token prefix. Each decode deterministically rebuilds recurrent/KV state from that prefix; generated
// tokens are committed so later turns and snapshots are complete.
type hipComposedEngineSession struct {
	model        sharedmodel.SessionModel
	prompt       []int32
	architecture string
	numLayers    int
	closed       bool
}

var _ engine.Session = (*hipComposedEngineSession)(nil)

func (session *hipComposedEngineSession) PrefillTokens(ids []int32) error {
	if err := session.ready("hip.composed.PrefillTokens"); err != nil {
		return err
	}
	if len(ids) == 0 {
		return core.NewError("hip.composed.PrefillTokens: empty token prefix")
	}
	session.prompt = append(session.prompt[:0], ids...)
	return nil
}

func (session *hipComposedEngineSession) AppendTokens(ids []int32) error {
	if err := session.ready("hip.composed.AppendTokens"); err != nil {
		return err
	}
	session.prompt = append(session.prompt, ids...)
	return nil
}

func (session *hipComposedEngineSession) Pos() int {
	if session == nil {
		return 0
	}
	return len(session.prompt)
}

func (session *hipComposedEngineSession) GenerateFromCacheEach(maxNew, eosID int, yield func(int32) bool) ([]int32, error) {
	var stops []int32
	if eosID >= 0 {
		stops = []int32{int32(eosID)}
	}
	return session.generate(maxNew, stops, sharedmodel.NewSampler(0), sharedmodel.SampleParams{}, nil, yield)
}

func (session *hipComposedEngineSession) GenerateSampledFromCacheEach(maxNew int, stopTokens []int32, sampler *sharedmodel.Sampler, params sharedmodel.SampleParams, transform sharedmodel.TokenTransform, yield func(int32) bool) ([]int32, error) {
	return session.generate(maxNew, stopTokens, sampler, params, transform, yield)
}

func (session *hipComposedEngineSession) generate(maxNew int, stopTokens []int32, sampler *sharedmodel.Sampler, params sharedmodel.SampleParams, transform sharedmodel.TokenTransform, yield func(int32) bool) ([]int32, error) {
	if err := session.ready("hip.composed.Generate"); err != nil {
		return nil, err
	}
	generated, err := sharedmodel.GenerateSampledWithStopTokensTransformEach(
		session.model,
		sampler,
		params,
		session.prompt,
		maxNew,
		stopTokens,
		transform,
		yield,
	)
	if err != nil {
		return nil, err
	}
	session.prompt = append(session.prompt, generated...)
	return generated, nil
}

func (session *hipComposedEngineSession) CaptureKVWithOptions(kv.CaptureOptions) (*kv.Snapshot, error) {
	if err := session.ready("hip.composed.CaptureKV"); err != nil {
		return nil, err
	}
	if len(session.prompt) == 0 {
		return nil, core.NewError("hip.composed.CaptureKV: empty session")
	}
	return &kv.Snapshot{
		Version:      kv.SnapshotVersion,
		Architecture: session.architecture,
		Tokens:       append([]int32(nil), session.prompt...),
		TokenOffset:  len(session.prompt),
		NumLayers:    session.numLayers,
	}, nil
}

func (session *hipComposedEngineSession) RangeKVBlocks(blockSize int, opts kv.CaptureOptions, yield func(kv.Block) (bool, error)) error {
	if err := session.ready("hip.composed.RangeKVBlocks"); err != nil {
		return err
	}
	if blockSize <= 0 {
		return core.NewError("hip.composed.RangeKVBlocks: block size must be > 0")
	}
	if opts.BlockStartToken < 0 {
		return core.NewError("hip.composed.RangeKVBlocks: block start token must be >= 0")
	}
	if yield == nil {
		return core.NewError("hip.composed.RangeKVBlocks: nil yield")
	}
	if len(session.prompt) == 0 {
		return core.NewError("hip.composed.RangeKVBlocks: empty session")
	}
	totalBlocks := (len(session.prompt) + blockSize - 1) / blockSize
	firstBlock := 0
	for firstBlock < totalBlocks && min((firstBlock+1)*blockSize, len(session.prompt)) <= opts.BlockStartToken {
		firstBlock++
	}
	for index := firstBlock; index < totalBlocks; index++ {
		start := index * blockSize
		end := min(start+blockSize, len(session.prompt))
		block := kv.Block{
			Index:      index,
			TokenStart: start,
			TokenCount: end - start,
			Snapshot: &kv.Snapshot{
				Version:      kv.SnapshotVersion,
				Architecture: session.architecture,
				Tokens:       append([]int32(nil), session.prompt[start:end]...),
				TokenOffset:  end,
				SeqLen:       end - start,
				NumLayers:    session.numLayers,
			},
		}
		keepGoing, err := yield(block)
		if err != nil || !keepGoing {
			return err
		}
	}
	return nil
}

func (session *hipComposedEngineSession) RestoreFromKV(ctx context.Context, snapshot *kv.Snapshot) error {
	if err := session.ready("hip.composed.RestoreFromKV"); err != nil {
		return err
	}
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if snapshot == nil || len(snapshot.Tokens) == 0 {
		return core.NewError("hip.composed.RestoreFromKV: snapshot carries no token prefix")
	}
	session.prompt = append(session.prompt[:0], snapshot.Tokens...)
	return nil
}

func (session *hipComposedEngineSession) Close() error {
	if session == nil || session.closed {
		return nil
	}
	session.closed = true
	session.prompt = nil
	return nil
}

func (session *hipComposedEngineSession) ready(operation string) error {
	if session == nil || session.model == nil {
		return core.NewError(operation + ": model is not initialised")
	}
	if session.closed {
		return core.NewError(operation + ": session is closed")
	}
	return nil
}
