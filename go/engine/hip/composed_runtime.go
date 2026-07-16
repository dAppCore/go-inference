// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import (
	"context"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/inference/decode/tokenizer"
	"dappco.re/go/inference/engine"
	"dappco.re/go/inference/kv"
	sharedmodel "dappco.re/go/inference/model"
	"dappco.re/go/inference/model/composed"
)

type hipComposedModelShape interface {
	HiddenSize() int
	NumLayers() int
}

// loadHIPComposedTextModel lets a registered config-composed architecture enter
// the shared session model before the standard transformer tensor loader.
func loadHIPComposedTextModel(path string, cfg inference.LoadConfig) (inference.TextModel, bool, error) {
	stat := core.Stat(path)
	if !stat.OK {
		return nil, false, nil
	}
	info, ok := stat.Value.(core.FsFileInfo)
	if !ok || !info.IsDir() {
		return nil, false, nil
	}

	tokenModel, matched, err := sharedmodel.LoadComposedDir(path)
	if err != nil || !matched {
		return nil, matched, err
	}
	if cfg.AdapterPath != "" {
		return nil, true, core.NewError("rocm.LoadModel: adapters are not supported by composed models")
	}
	sessionModel, ok := tokenModel.(sharedmodel.SessionModel)
	if !ok {
		return nil, true, core.NewError("rocm.LoadModel: composed loader returned a model without incremental sessions")
	}
	shape, ok := tokenModel.(hipComposedModelShape)
	if !ok || shape.HiddenSize() <= 0 || shape.NumLayers() <= 0 {
		return nil, true, core.NewError("rocm.LoadModel: composed loader returned incomplete model geometry")
	}
	modelType, _, err := sharedmodel.ProbeDirArch(path)
	if err != nil {
		return nil, true, err
	}
	tok, err := tokenizer.LoadTokenizer(core.PathJoin(path, "tokenizer.json"))
	if err != nil {
		return nil, true, core.E("rocm.LoadModel", "load composed tokenizer", err)
	}
	maxLen := cfg.ContextLen
	if maxLen <= 0 {
		maxLen = sharedmodel.ProbeDirContextWindow(path)
		if maxLen <= 0 {
			maxLen = defaultContextLengthCap
		}
	}
	source := &hipComposedTextModel{
		model:            sessionModel,
		tokenizer:        tok,
		modelType:        modelType,
		numLayers:        shape.NumLayers(),
		declaredStops:    loadGenerationConfigStops(path),
		declaredSampling: loadGenerationConfigSamplingDefaults(path),
	}
	modelInfo := inference.ModelInfo{
		Architecture: modelType,
		VocabSize:    tokenModel.Vocab(),
		NumLayers:    shape.NumLayers(),
		HiddenSize:   shape.HiddenSize(),
	}
	return engine.NewTextModel(source, tok, modelType, modelInfo, maxLen), true, nil
}

// hipComposedTextModel bridges the backend-neutral composed SessionModel into
// HIP's shared engine serving surface. The model owns host f32 weights; its
// projection hooks remain available for portable HIP++ acceleration.
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

func (model *hipComposedTextModel) DeclaredChatTemplate() (engine.ChatTemplate, bool) {
	if model == nil {
		return engine.ChatTemplate{}, false
	}
	if composed.ChatMLDialect(model.modelType) {
		return hipQwenChatTemplate(), true
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

// hipComposedEngineSession retains a composed architecture as its complete
// token prefix. Each decode deterministically rebuilds recurrent/KV state from
// that prefix; generated tokens are committed so later turns and snapshots are
// complete.
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
