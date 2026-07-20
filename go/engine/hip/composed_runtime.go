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

// loadHIPComposedTextModel used to detour any config-composed architecture (an ArchSpec registered
// with a Composed hook — the Qwen 3.6 gated-delta hybrid family, plus the arch-zoo's other hybrid/MoE
// registrations) through model/composed's LoadComposedDir, ahead of HIP's own GGUF/safetensors native
// loader. #50 retires model/composed; HIP was one of its last two consumers in go-inference (the
// other: cmd/mtp-probe) — this function no longer imports, or calls into, model/composed.
//
// It still probes config.json — through the shared, composed-free model.ProbeDirArch/ProbeModelTypes/
// LookupArch (the root "dappco.re/go/inference/model" package, not model/composed) — to recognise a
// checkpoint that WOULD have matched a Composed hook, so a retired checkpoint gets a clean, NAMED
// decline instead of silently falling into HIP's regular native pipeline. That matters: HIP's own
// architecture-profile metadata (engine/hip/profile) lists several composed-only archs as "supported"
// for INSPECTION/labelling purposes (qwen3_6, qwen3_6_moe, qwen3_next, deepseek, deepseek_r1, mixtral,
// composed, hybrid, …), but that table is proven insufficient as a stand-in for "HIP has a working
// forward kernel" — qwen3_6 sits in that very table and STILL needed the composed detour, because HIP's
// native runtime has no gated-delta linear-attention kernel (see dense_config.go's staged, unwired
// IsQwen36Hybrid/Qwen36NativeGuardMessage — written for a native guard that was never wired in because
// the composed fallback made it unnecessary). Letting a composed-registered checkpoint fall through
// unchecked risks the native pipeline silently misreading a hybrid/MoE/MLA tensor layout as a plain
// dense transformer — a coherent-but-wrong decode, not a clean failure. So every architecture that
// carries a Composed hook declines HERE, named, rather than falling through unexamined. The full set
// affected today (queried live off the registry, not hardcoded): qwen3_5/qwen3_5_moe(+text aliases)/
// qwen3_6/qwen3_6_moe/qwen3_next (dual-route, model/arch/Qwen/qwen35), mixtral/dbrx/olmoe/granitemoe/
// qwen2_moe/qwen3_moe (dual-route, each with its own factory port elsewhere — HIP has none), jetmoe/
// deepseek_v2/deepseek_v3/deepseek_vl_v2/glm_ocr(+text)/dots_ocr(+1_5)/llama4(+text) (composed-only,
// several already refuse unconditionally inside their own Composed hook), the generic "composed"/
// "hybrid" ids, and the qwen3_5_mtp/qwen3_6_mtp/qwen3_5_mtp_text drafter ids.
//
// Route (a) — routing through HIP's native runtime via the shared factory route (model.Load) — is not
// available for any of these architectures: HIP's nativeRuntime interface only ever consumes
// nativeLoadConfig (HIP's own GGUF/safetensors-tensor-shaped struct); nothing converts a
// *model.LoadedModel (model.Load's reactive Assemble output) into one. The only caller of model.Load in
// this package loads Gemma4's vision/audio TOWER as a sub-component (gemma4_vision_encoder.go,
// gemma4_unified_vision.go, gemma4_audio_tower.go) — there is no glue for a full text decode. Building
// that glue — plus a HIP-side hybrid/MoE/MLA forward to actually execute what Assemble describes —
// mirrors engine/metal's arch_qwen_fused.go, itself a substantial GPU-kernel investment; it is a new
// engine seam, not a same-file sever. Route (b) — a clean typed decline — applies uniformly to every
// architecture this loader used to reach.
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
	spec, matched := sharedmodel.LookupArch(modelType)
	declared := modelType
	if !matched {
		// Multimodal wrapper fallback, mirroring LoadComposedDir's own resolution: the top-level
		// model_type is unregistered, but a nested text_config.model_type carries the Composed hook.
		if _, textModelType := sharedmodel.ProbeModelTypes(configJSON); textModelType != "" {
			if textSpec, textMatched := sharedmodel.LookupArch(textModelType); textMatched {
				spec, matched, declared = textSpec, true, textModelType
			}
		}
	}
	if !matched || spec.Composed == nil {
		return nil, false, nil
	}
	return nil, true, core.NewError("rocm.LoadModel: " + declared + " is a config-composed/hybrid architecture with no native ROCm execution path — the model/composed fallback that used to serve it is retired (#50)")
}

// hipComposedTextModel bridges a backend-neutral, token-prefix sharedmodel.SessionModel into HIP's
// shared engine serving surface (OpenEngineSession → hipComposedEngineSession). Despite the name, it is
// no longer composed-specific: loadHIPComposedTextModel (above) retired its use of model/composed
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
// than model/composed's ChatMLDialect helper — the last direct reference this file held to model/
// composed, removed as part of #50's sever.
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
