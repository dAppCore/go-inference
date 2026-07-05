// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"encoding/binary"
	"sync"

	core "dappco.re/go"
	"dappco.re/go/inference/decode/tokenizer"
	"dappco.re/go/inference/model"
	"dappco.re/go/inference/model/gguf"
	"dappco.re/go/inference/model/safetensors"
	coreio "dappco.re/go/io"
)

const nativeAssistantLogitsFloor = -3.4028234663852886e38
const nativeAssistantDefaultDraftTokens = 4

// nativeAssistantLowAcceptPatience is how many CONSECUTIVE sub-50%-accept blocks the
// speculative loop tolerates before it gives up and finishes the request with plain
// target decode. A single weak block is expected — greedy decode of a quant target
// forks from the drafter's proposal at any near-tie (e.g. "The" vs "Here" as the very
// first token), which zeroes that one block; the drafter re-syncs on the next block once
// it re-seeds from the target's committed token. Bailing after just one such block (the
// previous behaviour) collapsed live acceptance to 0% on any prompt whose first token is
// a near-tie, even though the same drafter goes on to accept 40-80% of the rest.
const nativeAssistantLowAcceptPatience = 4

var nativeAssistantByteScratchPools sync.Map

// AssistantModel is the native, CGO-free assistant-only checkpoint
// handle. The decode integration uses the mmap-backed tensors directly in a
// later slice; this loader owns the mmap and validates the attached-drafter
// tensor layout up front. The config arrives as the NEUTRAL model.AssistantConfig
// a registered model package parsed (model.RegisterAssistant) — the engine never
// keys on which model family the drafter belongs to.
type AssistantModel struct {
	Config                   model.AssistantConfig
	Arch                     model.Arch
	Tensors                  map[string]safetensors.Tensor
	BackboneHiddenSize       int
	NumCentroids             int
	CentroidIntermediateTopK int
	UseOrderedEmbeddings     bool
	Tok                      *tokenizer.Tokenizer

	mapping *safetensors.DirMapping
	gguf    *gguf.TensorMapping
}

// AssistantPair is a native target-architecture plus assistant drafter
// compatibility record. Runtime decode attachment is layered on top of this:
// this type proves the two checkpoint configs can share target K/V streams.
type AssistantPair struct {
	TargetArch model.Arch
	Assistant  *AssistantModel
}

// Method reports the speculative-decode method inferred from the drafter (see
// model.MTPMethod), so the decode driver dispatches on the method rather than
// assuming the separate draft-model path. An unstamped config (e.g. a GGUF load
// that has not carried the field) defaults to model.MTPDraftModel — the only
// method shipped today.
func (pair *AssistantPair) Method() model.MTPMethod {
	if pair == nil || pair.Assistant == nil {
		return model.MTPDraftModel
	}
	if m := pair.Assistant.Config.Method; m != "" {
		return m
	}
	return model.MTPDraftModel
}

// AssistantDraftStepResult is one native assistant proposal from a target
// token, previous target hidden state, and target K/V streams.
type AssistantDraftStepResult struct {
	Logits []byte
	Token  int32
	Hidden []byte
}

// AssistantDraftBlockResult is a chained native assistant proposal block.
type AssistantDraftBlockResult struct {
	Tokens []int32
	Hidden []byte
}

// AssistantVerifyResult reports target-side verification of a proposed
// assistant draft block against a native target session. Logits and Hidden are
// caller-owned CPU byte copies.
type AssistantVerifyResult struct {
	DraftedTokens    []int32
	TargetTokens     []int32
	AcceptedTokens   []int32
	RejectedTokens   []int32
	AcceptedCount    int
	RejectedCount    int
	ReplacementToken int32
	AllAccepted      bool
	Logits           []byte
	Hidden           []byte
}

// AssistantGenerateResult records one native greedy assistant generation
// run over an ArchSession target.
type AssistantGenerateResult struct {
	Tokens             []int32
	PromptTokens       int
	TargetTokens       int
	DraftTokens        int
	AcceptedTokens     int
	RejectedTokens     int
	TargetVerifyCalls  int
	TargetCalls        int
	DraftCalls         int
	DraftTokenSchedule []int
}

// AssistantTokenSink receives each verified token as the native assistant
// generation loop emits it. Returning false stops generation without error.
type AssistantTokenSink func(int32) bool

func newAssistantGenerateResult(promptTokens, maxNew, draftTokens int) AssistantGenerateResult {
	scheduleCap := 0
	if maxNew > 0 && draftTokens > 0 {
		scheduleCap = (maxNew + draftTokens - 1) / draftTokens
	}
	return AssistantGenerateResult{
		Tokens:             make([]int32, 0, maxNew),
		PromptTokens:       promptTokens,
		DraftTokenSchedule: make([]int, 0, scheduleCap),
	}
}

func nativeAssistantSuppressArg(suppressTokens [][]int32) []int32 {
	if len(suppressTokens) == 0 {
		return nil
	}
	return suppressTokens[0]
}

// AssistantTargetKV is a native byte-view of a target K/V stream that the
// assistant can attend to by target layer type.
type AssistantTargetKV struct {
	Key     []byte
	Value   []byte
	Offset  int
	Length  int
	KVHeads int
	HeadDim int
}

func (kv AssistantTargetKV) HasState() bool {
	return len(kv.Key) > 0 && len(kv.Value) > 0 && kv.Length > 0
}

// AssistantKVEntry binds a Gemma 4 layer type to a target K/V byte stream.
type AssistantKVEntry struct {
	LayerType string
	KV        AssistantTargetKV
}

// AssistantTargetKVByType is the native equivalent of pkg/metal's tiny
// layer-type lookup for assistant draft steps. The key set is normally just
// "sliding_attention" and "full_attention", so a slice scan is enough.
type AssistantTargetKVByType struct {
	entries []AssistantKVEntry
}

type assistantDraftLayerScratchSlot int

const (
	assistantDraftScratchInputNorm assistantDraftLayerScratchSlot = iota
	assistantDraftScratchAttnQ
	assistantDraftScratchAttnQNorm
	assistantDraftScratchAttnQRope
	assistantDraftScratchAttn
	assistantDraftScratchAttnOut
	assistantDraftScratchAttnResidual
	assistantDraftScratchResidual
	assistantDraftScratchFFIn
	assistantDraftScratchGate
	assistantDraftScratchUp
	assistantDraftScratchGated
	assistantDraftScratchFF
	assistantDraftScratchFFResidual
	assistantDraftScratchNext
	assistantDraftScratchLayerOut
	assistantDraftScratchSlotCount
)

type assistantDraftLayerScratch struct {
	usePinned bool
	pinned    [assistantDraftScratchSlotCount]*pinnedNoCopyBytes

	inputNorm    []byte
	attnQ        []byte
	attnQNorm    []byte
	attnQRope    []byte
	attn         []byte
	attnOut      []byte
	attnResidual []byte
	residual     []byte
	ffIn         []byte
	gate         []byte
	up           []byte
	gated        []byte
	ff           []byte
	ffResidual   []byte
	next         []byte
	layerOut     []byte
}

func (s *assistantDraftLayerScratch) usePinnedBacking() {
	if s != nil {
		s.usePinned = true
	}
}

func (s *assistantDraftLayerScratch) close() {
	if s == nil {
		return
	}
	for i := range s.pinned {
		if s.pinned[i] != nil {
			s.pinned[i].Close()
			s.pinned[i] = nil
		}
	}
}

func (s *assistantDraftLayerScratch) slot(slot assistantDraftLayerScratchSlot) *[]byte {
	switch slot {
	case assistantDraftScratchInputNorm:
		return &s.inputNorm
	case assistantDraftScratchAttnQ:
		return &s.attnQ
	case assistantDraftScratchAttnQNorm:
		return &s.attnQNorm
	case assistantDraftScratchAttnQRope:
		return &s.attnQRope
	case assistantDraftScratchAttn:
		return &s.attn
	case assistantDraftScratchAttnOut:
		return &s.attnOut
	case assistantDraftScratchAttnResidual:
		return &s.attnResidual
	case assistantDraftScratchResidual:
		return &s.residual
	case assistantDraftScratchFFIn:
		return &s.ffIn
	case assistantDraftScratchGate:
		return &s.gate
	case assistantDraftScratchUp:
		return &s.up
	case assistantDraftScratchGated:
		return &s.gated
	case assistantDraftScratchFF:
		return &s.ff
	case assistantDraftScratchFFResidual:
		return &s.ffResidual
	case assistantDraftScratchNext:
		return &s.next
	case assistantDraftScratchLayerOut:
		return &s.layerOut
	default:
		return nil
	}
}

func (s *assistantDraftLayerScratch) bytes(slot assistantDraftLayerScratchSlot, n int) []byte {
	if s == nil {
		return make([]byte, n)
	}
	dst := s.slot(slot)
	if dst == nil {
		return make([]byte, n)
	}
	if s.usePinned {
		pinned := s.pinned[slot]
		if pinned != nil && len(pinned.bytes) == n && pinned.buf != nil {
			*dst = pinned.bytes[:n]
			return *dst
		}
		if pinned != nil {
			pinned.Close()
			s.pinned[slot] = nil
		}
		if pinned, err := newPinnedNoCopyBytes(n); err == nil {
			s.pinned[slot] = pinned
			*dst = pinned.bytes[:n]
			return *dst
		}
	}
	if cap(*dst) < n {
		*dst = make([]byte, n)
	}
	*dst = (*dst)[:n]
	return *dst
}

func (m *AssistantTargetKVByType) set(layerType string, targetKV AssistantTargetKV) {
	for i := range m.entries {
		if m.entries[i].LayerType == layerType {
			m.entries[i].KV = targetKV
			return
		}
	}
	if m.entries == nil {
		m.entries = make([]AssistantKVEntry, 0, 2)
	}
	m.entries = append(m.entries, AssistantKVEntry{LayerType: layerType, KV: targetKV})
}

func (m AssistantTargetKVByType) Get(layerType string) (AssistantTargetKV, bool) {
	for i := range m.entries {
		if m.entries[i].LayerType == layerType {
			return m.entries[i].KV, true
		}
	}
	return AssistantTargetKV{}, false
}

// LoadAssistantDir loads a Gemma 4 assistant-only drafter checkpoint
// without pkg/metal. The returned tensors are mmap-backed; call Close when the
// assistant runtime no longer needs them.
func LoadAssistantDir(dir string) (*AssistantModel, error) {
	cfgStr, err := coreio.Local.Read(core.PathJoin(dir, "config.json"))
	if err != nil {
		return nil, core.E("native.assistant.Load", "read config.json", err)
	}
	// the reactive parse: probe model_type → the registered model package's parser
	// (model.RegisterAssistant) → the neutral, already-validated config + derived arch.
	cfg, err := model.ParseAssistantConfig([]byte(cfgStr))
	if err != nil {
		return nil, core.E("native.assistant.Load", "parse config", err)
	}
	tok, err := tokenizer.LoadTokenizer(core.PathJoin(dir, "tokenizer.json"))
	if err != nil {
		return nil, core.E("native.assistant.Load", "load tokenizer", err)
	}
	dm, err := safetensors.LoadDirMmap(dir)
	if err != nil {
		return nil, core.E("native.assistant.Load", "load weights", err)
	}
	m := &AssistantModel{
		Config:                   cfg,
		Arch:                     cfg.Arch,
		Tensors:                  dm.Tensors,
		BackboneHiddenSize:       cfg.BackboneHidden,
		NumCentroids:             cfg.NumCentroids,
		CentroidIntermediateTopK: cfg.CentroidTopK,
		UseOrderedEmbeddings:     cfg.OrderedEmbeddings,
		Tok:                      tok,
		mapping:                  dm,
	}
	if err := validateNativeAssistantModel(m); err != nil {
		_ = m.Close()
		return nil, core.E("native.assistant.Load", "validate tensors", err)
	}
	return m, nil
}

// LoadAssistantPairDirs loads assistant metadata/tensors and validates
// them against the target checkpoint config without loading the target weights.
func LoadAssistantPairDirs(targetDir, assistantDir string) (*AssistantPair, error) {
	if core.Trim(targetDir) == "" {
		return nil, core.NewError("native.assistant pair target path is required")
	}
	if core.Trim(assistantDir) == "" {
		return nil, core.NewError("native.assistant pair assistant path is required")
	}
	targetArch, err := loadAssistantTargetArch(targetDir)
	if err != nil {
		return nil, core.E("native.assistant.Pair", "load target config", err)
	}
	assistant, err := loadNativeAssistantForTarget(targetDir, assistantDir)
	if err != nil {
		return nil, core.E("native.assistant.Pair", "load assistant", err)
	}
	pair := &AssistantPair{TargetArch: targetArch, Assistant: assistant}
	if err := validateNativeAssistantPair(pair); err != nil {
		_ = pair.Close()
		return nil, core.E("native.assistant.Pair", "validate attachment", err)
	}
	return pair, nil
}

func loadAssistantTargetArch(dir string) (model.Arch, error) {
	mt, cfg, err := model.ProbeDirArch(dir)
	if err != nil {
		return model.Arch{}, err
	}
	textMT, nestedTextMT := model.ProbeModelTypes(cfg)
	if textMT != "" {
		mt = textMT
	}
	spec, ok := model.LookupArch(mt)
	if !ok && nestedTextMT != "" {
		spec, ok = model.LookupArch(nestedTextMT)
	}
	if !ok {
		return model.Arch{}, core.NewError("native.assistant target has no registered architecture: " + mt)
	}
	ac, err := spec.Parse(cfg)
	if err != nil {
		return model.Arch{}, err
	}
	arch, err := ac.Arch()
	if err != nil {
		return model.Arch{}, err
	}
	if arch.Hidden <= 0 || len(arch.Layer) == 0 {
		return model.Arch{}, core.NewError("native.assistant target arch is incomplete")
	}
	return arch, nil
}

func loadNativeAssistantForTarget(targetDir, assistantPath string) (*AssistantModel, error) {
	if file, ok := ResolveAssistantGGUFDrafterFile(assistantPath); ok {
		tok, err := tokenizer.LoadTokenizer(core.PathJoin(targetDir, "tokenizer.json"))
		if err != nil {
			return nil, core.E("native.assistant.gguf", "load target tokenizer", err)
		}
		return loadNativeAssistantFromGGUF(file, tok)
	}
	return LoadAssistantDir(assistantPath)
}

func validateNativeAssistantPair(pair *AssistantPair) error {
	if pair == nil || pair.TargetArch.Hidden <= 0 {
		return core.NewError("native.assistant pair target is nil")
	}
	assistant := pair.Assistant
	if assistant == nil {
		return core.NewError("native.assistant pair assistant is nil")
	}
	target := pair.TargetArch
	if assistant.BackboneHiddenSize != target.Hidden {
		return core.NewError(core.Sprintf("native.assistant backbone_hidden_size = %d, want target hidden_size %d", assistant.BackboneHiddenSize, target.Hidden))
	}
	if target.Vocab > 0 && assistant.Arch.Vocab > 0 && target.Vocab != assistant.Arch.Vocab {
		return core.NewError(core.Sprintf("native.assistant vocab_size = %d, want target vocab_size %d", assistant.Arch.Vocab, target.Vocab))
	}
	return validateNativeAssistantTargetTypes(target, assistant)
}

func validateNativeAssistantTargetTypes(target model.Arch, assistant *AssistantModel) error {
	targetTypes := map[string]int{}
	for _, layer := range target.Layer {
		layerType := layer.TypeName()
		if layerType != "" {
			if _, ok := targetTypes[layerType]; !ok {
				targetTypes[layerType] = layer.HeadDim
			}
		}
	}
	if len(targetTypes) == 0 {
		return core.NewError("native.assistant pair target layer types are unavailable")
	}
	for idx, layer := range assistant.Arch.Layer {
		layerType := assistant.Config.LayerType(idx)
		if _, ok := targetTypes[layerType]; !ok {
			return core.NewError(core.Sprintf("native.assistant layer %d type %q has no target K/V stream", idx, layerType))
		}
		wantHeadDim := targetTypes[layerType]
		if wantHeadDim > 0 && layer.HeadDim != wantHeadDim {
			return core.NewError(core.Sprintf("native.assistant layer %d head_dim = %d, want target %s head_dim %d", idx, layer.HeadDim, layerType, wantHeadDim))
		}
	}
	return nil
}

func validateNativeAssistantModel(m *AssistantModel) error {
	if m == nil {
		return core.NewError("native.assistant model is nil")
	}
	var missing []string
	addMissing := func(name string) {
		t, ok := m.Tensors[name]
		if !ok || t.Dtype == "" || len(t.Data) == 0 {
			missing = append(missing, name)
		}
	}
	addAnyMissing := func(label string, names ...string) {
		for _, name := range names {
			t, ok := m.Tensors[name]
			if ok && t.Dtype != "" && len(t.Data) > 0 {
				return
			}
		}
		missing = append(missing, label)
	}
	addLinearMissing := func(name string) { addMissing(name + ".weight") }
	addNormMissing := func(name string) { addMissing(name + ".weight") }

	addMissing("model.embed_tokens.weight")
	addNormMissing("model.norm")
	addLinearMissing("pre_projection")
	addLinearMissing("post_projection")
	if m.UseOrderedEmbeddings {
		addLinearMissing("masked_embedding.centroids")
		addMissing("masked_embedding.token_ordering")
	}
	for i := range m.Arch.Layer {
		prefix := core.Sprintf("model.layers.%d", i)
		addNormMissing(prefix + ".input_layernorm")
		addNormMissing(prefix + ".post_attention_layernorm")
		addNormMissing(prefix + ".pre_feedforward_layernorm")
		addNormMissing(prefix + ".post_feedforward_layernorm")
		addAnyMissing(prefix+".layer_scalar", prefix+".layer_scalar", prefix+".layer_scalar.weight")
		addLinearMissing(prefix + ".self_attn.q_proj")
		addLinearMissing(prefix + ".self_attn.o_proj")
		addNormMissing(prefix + ".self_attn.q_norm")
		addLinearMissing(prefix + ".mlp.gate_proj")
		addLinearMissing(prefix + ".mlp.up_proj")
		addLinearMissing(prefix + ".mlp.down_proj")
	}
	if len(missing) > 0 {
		return core.NewError("missing required tensors: " + core.Join(", ", missing...))
	}
	if err := validateNativeAssistantProjectionShapes(m); err != nil {
		return err
	}
	if err := validateNativeAssistantOrderedEmbeddingShape(m); err != nil {
		return err
	}
	return nil
}

func validateNativeAssistantProjectionShapes(m *AssistantModel) error {
	if err := validateNativeAssistantLinearShape(m, "pre_projection", m.Arch.Hidden, m.BackboneHiddenSize*2); err != nil {
		return err
	}
	if err := validateNativeAssistantLinearShape(m, "post_projection", m.BackboneHiddenSize, m.Arch.Hidden); err != nil {
		return err
	}
	if m.UseOrderedEmbeddings {
		if err := validateNativeAssistantLinearShape(m, "masked_embedding.centroids", m.NumCentroids, m.Arch.Hidden); err != nil {
			return err
		}
	}
	return nil
}

func validateNativeAssistantLinearShape(m *AssistantModel, name string, out, in int) error {
	t, ok := m.Tensors[name+".weight"]
	if !ok {
		return nil
	}
	if len(t.Shape) < 2 {
		return core.NewError(name + ".weight has invalid rank")
	}
	gotOut := t.Shape[len(t.Shape)-2]
	gotIn := t.Shape[len(t.Shape)-1]
	if out > 0 && gotOut != out {
		return core.NewError(core.Sprintf("%s.weight output dim = %d, want %d", name, gotOut, out))
	}
	if in > 0 && !nativeAssistantLinearInputMatches(m, name, gotIn, in) {
		return core.NewError(core.Sprintf("%s.weight input dim = %d, want %d", name, gotIn, in))
	}
	return nil
}

func nativeAssistantLinearInputMatches(m *AssistantModel, name string, gotIn, wantIn int) bool {
	if gotIn == wantIn {
		return true
	}
	quant := m.Config.Quant
	if quant == nil {
		return false
	}
	_, bits := quant.For(name)
	if bits <= 0 {
		return false
	}
	if _, ok := m.Tensors[name+".scales"]; !ok {
		return false
	}
	packFactor := 32 / bits
	if packFactor > 0 && wantIn%packFactor == 0 && gotIn == wantIn/packFactor {
		return true
	}
	return gotIn == (wantIn*bits+31)/32
}

func validateNativeAssistantOrderedEmbeddingShape(m *AssistantModel) error {
	if !m.UseOrderedEmbeddings {
		return nil
	}
	t, ok := m.Tensors["masked_embedding.token_ordering"]
	if !ok {
		return nil
	}
	switch t.Dtype {
	case "I32", "I64":
	default:
		return core.NewError("masked_embedding.token_ordering dtype = " + t.Dtype + ", want int32 or int64")
	}
	vocabSize := m.Arch.Vocab
	numCentroids := m.NumCentroids
	if vocabSize <= 0 || numCentroids <= 0 || vocabSize%numCentroids != 0 {
		return core.NewError("masked_embedding.token_ordering requires vocab_size divisible by num_centroids")
	}
	tokensPerCentroid := vocabSize / numCentroids
	if len(t.Shape) == 1 && t.Shape[0] == vocabSize {
		return nil
	}
	if len(t.Shape) == 2 && t.Shape[0] == numCentroids && t.Shape[1] == tokensPerCentroid {
		return nil
	}
	return core.NewError(core.Sprintf("masked_embedding.token_ordering shape = %v, want [%d] or [%d %d]", t.Shape, vocabSize, numCentroids, tokensPerCentroid))
}

func (m *AssistantModel) Close() error {
	if m == nil {
		return nil
	}
	var err error
	if m.mapping != nil {
		err = core.ErrorJoin(err, m.mapping.Close())
		m.mapping = nil
	}
	if m.gguf != nil {
		err = core.ErrorJoin(err, m.gguf.Close())
		m.gguf = nil
	}
	m.Tensors = nil
	return err
}

func (m *AssistantModel) ModelType() string {
	if m == nil {
		return ""
	}
	// report the claiming spec's CANONICAL id (its first ModelTypes entry) so checkpoint
	// variants (e.g. a unified assistant) normalise to the public id their model package
	// declares — the registry is the normalisation table, never a hardcoded model list.
	if spec, ok := model.LookupAssistant(m.Config.ModelType); ok && len(spec.ModelTypes) > 0 && spec.ModelTypes[0] != "" {
		return spec.ModelTypes[0]
	}
	return m.Config.ModelType
}

func (m *AssistantModel) Tokenizer() *tokenizer.Tokenizer {
	if m == nil {
		return nil
	}
	return m.Tok
}

func (m *AssistantModel) NumLayers() int {
	if m == nil {
		return 0
	}
	return len(m.Arch.Layer)
}

func (m *AssistantModel) Tensor(name string) (safetensors.Tensor, bool) {
	if m == nil {
		return safetensors.Tensor{}, false
	}
	t, ok := m.Tensors[name]
	return t, ok
}

func (pair *AssistantPair) TargetKVByLayerType(targetKVs []AssistantTargetKV) (AssistantTargetKVByType, error) {
	return pair.targetKVByLayerType(targetKVs, nil)
}

func (pair *AssistantPair) targetKVByLayerType(targetKVs []AssistantTargetKV, entries []AssistantKVEntry) (AssistantTargetKVByType, error) {
	if pair == nil || pair.Assistant == nil {
		return AssistantTargetKVByType{}, core.NewError("native.assistant draft step requires a validated pair")
	}
	out := AssistantTargetKVByType{entries: entries[:0]}
	for layerIdx, layer := range pair.TargetArch.Layer {
		layerType := layer.TypeName()
		if layerType == "" {
			continue
		}
		ownerIdx := layerIdx
		if layer.KVShareFrom >= 0 {
			ownerIdx = layer.KVShareFrom
		}
		if ownerIdx < 0 || ownerIdx >= len(pair.TargetArch.Layer) {
			continue
		}
		cacheIdx := pair.TargetArch.Layer[ownerIdx].CacheIndex
		if cacheIdx < 0 || cacheIdx >= len(targetKVs) {
			continue
		}
		targetKV := targetKVs[cacheIdx]
		if !targetKV.HasState() {
			return AssistantTargetKVByType{}, core.NewError(core.Sprintf("native.assistant draft step target layer %d has empty K/V stream", layerIdx))
		}
		out.set(layerType, targetKV)
	}
	for idx := range pair.Assistant.Arch.Layer {
		layerType := pair.Assistant.Config.LayerType(idx)
		targetKV, ok := out.Get(layerType)
		if !ok || !targetKV.HasState() {
			return AssistantTargetKVByType{}, core.NewError("native.assistant draft step missing populated target K/V stream for " + layerType)
		}
	}
	return out, nil
}

// TargetKVByLayerTypeFromSession maps the target session's resident K/V cache
// rows to the assistant's layer-type streams. ArchSession stores K/V rows
// token-major; the assistant attention primitive consumes head-major slabs, so
// this materialises the visible cache window in assistant-ready order.
func (pair *AssistantPair) TargetKVByLayerTypeFromSession(target *ArchSession) (AssistantTargetKVByType, error) {
	return pair.targetKVByLayerTypeFromSession(target, false)
}

func (pair *AssistantPair) targetKVByLayerTypeFromSessionScratch(target *ArchSession) (AssistantTargetKVByType, error) {
	return pair.targetKVByLayerTypeFromSession(target, true)
}

func (pair *AssistantPair) targetKVByLayerTypeFromSession(target *ArchSession, useScratch bool) (AssistantTargetKVByType, error) {
	if pair == nil || pair.Assistant == nil {
		return AssistantTargetKVByType{}, core.NewError("native.assistant draft step requires a validated pair")
	}
	if target == nil {
		return AssistantTargetKVByType{}, core.NewError("native.assistant draft step target session is nil")
	}
	if target.pos <= 0 {
		return AssistantTargetKVByType{}, core.NewError("native.assistant draft step target session cache is empty")
	}
	if err := pair.validateTargetSessionArch(target.arch); err != nil {
		return AssistantTargetKVByType{}, err
	}
	views, err := target.stateLayerViews()
	if err != nil {
		return AssistantTargetKVByType{}, err
	}
	maxCacheIndex := -1
	for _, view := range views {
		if view.cacheIndex > maxCacheIndex {
			maxCacheIndex = view.cacheIndex
		}
	}
	if maxCacheIndex < 0 {
		return AssistantTargetKVByType{}, core.NewError("native.assistant draft step target session has no K/V cache owners")
	}
	var targetKVs []AssistantTargetKV
	if useScratch {
		targetKVs = target.mtpTargetKVScratchEntries(maxCacheIndex + 1)
	} else {
		targetKVs = make([]AssistantTargetKV, maxCacheIndex+1)
	}
	for _, view := range views {
		if view.cacheIndex < 0 {
			continue
		}
		start, tokenCount, err := nativeKVLayerCaptureWindow(view, target.pos)
		if err != nil {
			return AssistantTargetKVByType{}, err
		}
		keyRows, valueRows, err := stateBlockLayerBytes(view, start, tokenCount, target.pos)
		if err != nil {
			return AssistantTargetKVByType{}, err
		}
		if len(keyRows) == 0 || len(valueRows) == 0 {
			return AssistantTargetKVByType{}, core.NewError(core.Sprintf("native.assistant draft step target layer %d has empty K/V stream", view.layer))
		}
		var keySlab, valueSlab []byte
		if useScratch {
			keySlab, valueSlab = target.mtpTargetKVSlabs(view.cacheIndex, len(keyRows), len(valueRows))
		} else {
			keySlab = make([]byte, len(keyRows))
			valueSlab = make([]byte, len(valueRows))
		}
		nativeKVTokenRowsToLayerSlab(keySlab, keyRows, tokenCount, view.kvHeads, view.headDim)
		nativeKVTokenRowsToLayerSlab(valueSlab, valueRows, tokenCount, view.kvHeads, view.headDim)
		targetKVs[view.cacheIndex] = AssistantTargetKV{
			Key:     keySlab,
			Value:   valueSlab,
			Offset:  start,
			Length:  tokenCount,
			KVHeads: view.kvHeads,
			HeadDim: view.headDim,
		}
	}
	if useScratch {
		return pair.targetKVByLayerType(targetKVs, target.mtpTargetKVByTypeEntries(len(pair.Assistant.Arch.Layer)))
	}
	return pair.TargetKVByLayerType(targetKVs)
}

func (pair *AssistantPair) validateTargetSessionArch(arch model.Arch) error {
	target := pair.TargetArch
	if target.Hidden <= 0 || arch.Hidden <= 0 || target.Hidden != arch.Hidden {
		return core.NewError(core.Sprintf("native.assistant target session hidden_size = %d, want %d", arch.Hidden, target.Hidden))
	}
	if target.Vocab > 0 && arch.Vocab > 0 && target.Vocab != arch.Vocab {
		return core.NewError(core.Sprintf("native.assistant target session vocab_size = %d, want %d", arch.Vocab, target.Vocab))
	}
	if len(target.Layer) == 0 || len(arch.Layer) != len(target.Layer) {
		return core.NewError(core.Sprintf("native.assistant target session layer count = %d, want %d", len(arch.Layer), len(target.Layer)))
	}
	for idx := range target.Layer {
		want := target.Layer[idx]
		got := arch.Layer[idx]
		if got.Attention != want.Attention || got.KVShareFrom != want.KVShareFrom || got.CacheIndex != want.CacheIndex {
			return core.NewError(core.Sprintf("native.assistant target session layer %d cache topology mismatch", idx))
		}
		if want.HeadDim > 0 && got.HeadDim > 0 && got.HeadDim != want.HeadDim {
			return core.NewError(core.Sprintf("native.assistant target session layer %d head_dim = %d, want %d", idx, got.HeadDim, want.HeadDim))
		}
		if want.KVHeads > 0 && got.KVHeads > 0 && got.KVHeads != want.KVHeads {
			return core.NewError(core.Sprintf("native.assistant target session layer %d kv_heads = %d, want %d", idx, got.KVHeads, want.KVHeads))
		}
	}
	return nil
}

func (m *AssistantModel) DraftInputProjection(tokenEmbedding, previousHidden []byte) ([]byte, error) {
	return m.DraftInputProjectionInto(nil, tokenEmbedding, previousHidden)
}

func (m *AssistantModel) DraftInputProjectionInto(out []byte, tokenEmbedding, previousHidden []byte) ([]byte, error) {
	backbone, hidden, input, weight, err := m.draftInputProjectionShape()
	if err != nil {
		return nil, err
	}
	backboneBytes := backbone * bf16Size
	if len(tokenEmbedding) != backboneBytes {
		return nil, core.NewError(core.Sprintf("native.assistant draft input token embedding bytes = %d, want %d", len(tokenEmbedding), backboneBytes))
	}
	if len(previousHidden) != backboneBytes {
		return nil, core.NewError(core.Sprintf("native.assistant draft input previous hidden bytes = %d, want %d", len(previousHidden), backboneBytes))
	}
	combined := getNativeAssistantByteScratch(input * bf16Size)
	defer putNativeAssistantByteScratch(combined)
	copy(combined, tokenEmbedding)
	copy(combined[backboneBytes:], previousHidden)
	return MatMulBF16NTInto(out, combined, weight, 1, input, hidden)
}

func (m *AssistantModel) draftInputProjectionShape() (backbone, hidden, input int, weight []byte, err error) {
	if m == nil {
		err = core.NewError("native.assistant draft input model is nil")
		return
	}
	backbone = m.BackboneHiddenSize
	hidden = m.Arch.Hidden
	if backbone <= 0 || hidden <= 0 {
		err = core.NewError("native.assistant draft input has incomplete dimensions")
		return
	}
	tensor, ok := m.Tensors["pre_projection.weight"]
	if !ok {
		err = core.NewError("native.assistant draft input missing pre_projection.weight")
		return
	}
	if tensor.Dtype != "BF16" {
		err = core.NewError("native.assistant draft input pre_projection.weight dtype = " + tensor.Dtype + ", want BF16")
		return
	}
	input = backbone * 2
	if len(tensor.Shape) < 2 || tensor.Shape[len(tensor.Shape)-2] != hidden || tensor.Shape[len(tensor.Shape)-1] != input {
		err = core.NewError(core.Sprintf("native.assistant draft input pre_projection.weight shape = %v, want [%d %d]", tensor.Shape, hidden, input))
		return
	}
	if len(tensor.Data) != hidden*input*bf16Size {
		err = core.NewError(core.Sprintf("native.assistant draft input pre_projection.weight bytes = %d, want %d", len(tensor.Data), hidden*input*bf16Size))
		return
	}
	return backbone, hidden, input, tensor.Data, nil
}

func nativeAssistantByteScratchPoolFor(byteLen int) *sync.Pool {
	if v, ok := nativeAssistantByteScratchPools.Load(byteLen); ok {
		return v.(*sync.Pool)
	}
	pool := new(sync.Pool)
	if v, loaded := nativeAssistantByteScratchPools.LoadOrStore(byteLen, pool); loaded {
		return v.(*sync.Pool)
	}
	return pool
}

func getNativeAssistantByteScratch(byteLen int) []byte {
	pool := nativeAssistantByteScratchPoolFor(byteLen)
	if v := pool.Get(); v != nil {
		if b, ok := v.([]byte); ok && cap(b) >= byteLen {
			return b[:byteLen]
		}
	}
	return make([]byte, byteLen)
}

func putNativeAssistantByteScratch(buf []byte) {
	if len(buf) == 0 {
		return
	}
	nativeAssistantByteScratchPoolFor(len(buf)).Put(buf)
}

func (pair *AssistantPair) DraftInputProjectionForToken(targetEmbed []byte, lastToken int32, previousHidden []byte) ([]byte, error) {
	return pair.DraftInputProjectionForTokenInto(nil, targetEmbed, lastToken, previousHidden)
}

func (pair *AssistantPair) DraftInputProjectionForTokenInto(out []byte, targetEmbed []byte, lastToken int32, previousHidden []byte) ([]byte, error) {
	target, err := pair.validateDraftInputTarget()
	if err != nil {
		return nil, err
	}
	backbone, hidden, input, weight, err := pair.Assistant.draftInputProjectionShape()
	if err != nil {
		return nil, err
	}
	if len(previousHidden) != backbone*bf16Size {
		return nil, core.NewError(core.Sprintf("native.assistant draft input previous hidden bytes = %d, want %d", len(previousHidden), backbone*bf16Size))
	}
	combined := getNativeAssistantByteScratch(input * bf16Size)
	defer putNativeAssistantByteScratch(combined)
	backboneBytes := backbone * bf16Size
	if _, err := embedTokenBF16Into(combined[:backboneBytes], targetEmbed, lastToken, target.Vocab, target.Hidden, embedScaleOf(target)); err != nil {
		return nil, core.E("native.assistant draft input", "target token embedding", err)
	}
	copy(combined[backboneBytes:], previousHidden)
	return MatMulBF16NTInto(out, combined, weight, 1, input, hidden)
}

func (pair *AssistantPair) DraftInputProjectionForTokenQuant(packed, scales, biases []byte, groupSize, bits int, lastToken int32, previousHidden []byte) ([]byte, error) {
	return pair.DraftInputProjectionForTokenQuantInto(nil, packed, scales, biases, groupSize, bits, lastToken, previousHidden)
}

func (pair *AssistantPair) DraftInputProjectionForTokenQuantInto(out []byte, packed, scales, biases []byte, groupSize, bits int, lastToken int32, previousHidden []byte) ([]byte, error) {
	target, err := pair.validateDraftInputTarget()
	if err != nil {
		return nil, err
	}
	backbone, hidden, input, weight, err := pair.Assistant.draftInputProjectionShape()
	if err != nil {
		return nil, err
	}
	if len(previousHidden) != backbone*bf16Size {
		return nil, core.NewError(core.Sprintf("native.assistant draft input previous hidden bytes = %d, want %d", len(previousHidden), backbone*bf16Size))
	}
	combined := getNativeAssistantByteScratch(input * bf16Size)
	defer putNativeAssistantByteScratch(combined)
	backboneBytes := backbone * bf16Size
	if _, err := embedTokenQuantInto(combined[:backboneBytes], packed, scales, biases, lastToken, target.Vocab, target.Hidden, groupSize, bits, embedScaleOf(target)); err != nil {
		return nil, core.E("native.assistant draft input", "target quant token embedding", err)
	}
	copy(combined[backboneBytes:], previousHidden)
	return MatMulBF16NTInto(out, combined, weight, 1, input, hidden)
}

func (pair *AssistantPair) DraftStep(targetEmbed []byte, lastToken int32, previousHidden []byte, targetKVs AssistantTargetKVByType, suppressTokens ...[]int32) (AssistantDraftStepResult, error) {
	if lastToken < 0 {
		return AssistantDraftStepResult{}, core.NewError("native.assistant draft step token is invalid")
	}
	projected, err := pair.DraftInputProjectionForToken(targetEmbed, lastToken, previousHidden)
	if err != nil {
		return AssistantDraftStepResult{}, err
	}
	return pair.draftStepFromProjectedWithSuppress(projected, targetKVs, nativeAssistantSuppressArg(suppressTokens))
}

func (pair *AssistantPair) DraftStepQuant(packed, scales, biases []byte, groupSize, bits int, lastToken int32, previousHidden []byte, targetKVs AssistantTargetKVByType, suppressTokens ...[]int32) (AssistantDraftStepResult, error) {
	if lastToken < 0 {
		return AssistantDraftStepResult{}, core.NewError("native.assistant draft step token is invalid")
	}
	projected, err := pair.DraftInputProjectionForTokenQuant(packed, scales, biases, groupSize, bits, lastToken, previousHidden)
	if err != nil {
		return AssistantDraftStepResult{}, err
	}
	return pair.draftStepFromProjectedWithSuppress(projected, targetKVs, nativeAssistantSuppressArg(suppressTokens))
}

// DraftStepFromSession drafts one assistant token from a target ArchSession
// boundary. The target session must already hold the accepted prefix in its
// resident cache and retainedHidden boundary. Logits and Hidden are
// session-owned scratch slices and are overwritten by the next MTP draft call.
func (pair *AssistantPair) DraftStepFromSession(target *ArchSession, lastToken int32, suppressTokens ...[]int32) (AssistantDraftStepResult, error) {
	if pair == nil || pair.Assistant == nil {
		return AssistantDraftStepResult{}, core.NewError("native.assistant draft step requires a validated pair")
	}
	if lastToken < 0 {
		return AssistantDraftStepResult{}, core.NewError("native.assistant draft step token is invalid")
	}
	if target == nil {
		return AssistantDraftStepResult{}, core.NewError("native.assistant draft step target session is nil")
	}
	if target.embed == nil && target.embedInto == nil {
		return AssistantDraftStepResult{}, core.NewError("native.assistant draft step target session has no embedder")
	}
	targetKVs, err := pair.targetKVByLayerTypeFromSessionScratch(target)
	if err != nil {
		return AssistantDraftStepResult{}, err
	}
	previousHidden, err := target.boundaryNormedHiddenScratch()
	if err != nil {
		return AssistantDraftStepResult{}, core.E("native.assistant draft step", "target boundary hidden", err)
	}
	tokenEmbedding, err := target.embedID(lastToken)
	if err != nil {
		return AssistantDraftStepResult{}, core.E("native.assistant draft step", "target token embedding", err)
	}
	if len(tokenEmbedding) != pair.TargetArch.Hidden*bf16Size {
		return AssistantDraftStepResult{}, core.NewError(core.Sprintf("native.assistant draft step target token embedding bytes = %d, want %d", len(tokenEmbedding), pair.TargetArch.Hidden*bf16Size))
	}
	projectedOut := target.mtpProjectionScratch(pair.Assistant.Arch.Hidden * bf16Size)
	projected, err := pair.Assistant.DraftInputProjectionInto(projectedOut, tokenEmbedding, previousHidden)
	if err != nil {
		return AssistantDraftStepResult{}, err
	}
	normedOut := target.mtpDraftScratch(&target.mtpDraftNormed, pair.Assistant.Arch.Hidden*bf16Size)
	hiddenOut := target.mtpDraftScratch(&target.mtpDraftHidden, pair.TargetArch.Hidden*bf16Size)
	logitsOut := target.mtpDraftScratch(&target.mtpDraftLogits, pair.Assistant.Arch.Vocab*bf16Size)
	logitScores := target.mtpDraftLogitScoreScratch(pair.Assistant.NumCentroids)
	logitSelected := target.mtpDraftLogitSelectedScratch(pair.Assistant.CentroidIntermediateTopK)
	target.mtpDraftLayerScratch.usePinnedBacking()
	return pair.draftStepFromProjectedIntoWithSuppress(projected, targetKVs, normedOut, hiddenOut, logitsOut, logitScores, logitSelected, &target.mtpDraftLayerScratch, nativeAssistantSuppressArg(suppressTokens))
}

// DraftBlockFromSession chains assistant draft steps from a target ArchSession
// boundary and returns CPU-visible proposed token ids. Verification is a
// separate target-session concern. Hidden is session-owned scratch and is
// overwritten by the next MTP draft call.
func (pair *AssistantPair) DraftBlockFromSession(target *ArchSession, lastToken int32, maxDraftTokens int, suppressTokens ...[]int32) (AssistantDraftBlockResult, error) {
	return pair.draftBlockFromSessionWithSuppress(target, lastToken, maxDraftTokens, true, nativeAssistantSuppressArg(suppressTokens))
}

// PrepareAssistantPrompt prefills promptIDs into the session and retains the boundary
// hidden the draft path seeds from — the exported seam the cross-engine MTP parity
// instrument drives (pkg/metal/model/gemma4's parity test); GenerateFromSessionEach
// runs the same preparation internally. BoundaryNormedHidden (arch_session.go) reads
// the retained seed back.
func (s *ArchSession) PrepareAssistantPrompt(promptIDs []int32) error {
	return s.prepareAssistantPrompt(promptIDs)
}

func (pair *AssistantPair) draftBlockFromSession(target *ArchSession, lastToken int32, maxDraftTokens int, copyTokens bool, suppressTokens ...[]int32) (AssistantDraftBlockResult, error) {
	return pair.draftBlockFromSessionWithSuppress(target, lastToken, maxDraftTokens, copyTokens, nativeAssistantSuppressArg(suppressTokens))
}

func (pair *AssistantPair) draftBlockFromSessionWithSuppress(target *ArchSession, lastToken int32, maxDraftTokens int, copyTokens bool, suppress []int32) (AssistantDraftBlockResult, error) {
	if pair == nil || pair.Assistant == nil {
		return AssistantDraftBlockResult{}, core.NewError("native.assistant draft block requires a validated pair")
	}
	if maxDraftTokens <= 0 {
		return AssistantDraftBlockResult{}, core.NewError("native.assistant draft block maxDraftTokens must be > 0")
	}
	if lastToken < 0 {
		return AssistantDraftBlockResult{}, core.NewError("native.assistant draft step token is invalid")
	}
	if target == nil {
		return AssistantDraftBlockResult{}, core.NewError("native.assistant draft step target session is nil")
	}
	if target.embed == nil && target.embedInto == nil {
		return AssistantDraftBlockResult{}, core.NewError("native.assistant draft step target session has no embedder")
	}
	targetKVs, err := pair.targetKVByLayerTypeFromSessionScratch(target)
	if err != nil {
		return AssistantDraftBlockResult{}, err
	}
	currentHidden, err := target.boundaryNormedHiddenScratch()
	if err != nil {
		return AssistantDraftBlockResult{}, core.E("native.assistant draft block", "target boundary hidden", err)
	}
	currentToken := lastToken
	var tokens []int32
	if copyTokens {
		tokens = make([]int32, 0, maxDraftTokens)
	} else {
		tokens = target.mtpDraftTokenScratch(maxDraftTokens)
	}
	for len(tokens) < maxDraftTokens {
		tokenEmbedding, err := target.embedID(currentToken)
		if err != nil {
			return AssistantDraftBlockResult{}, core.E("native.assistant draft block", "target token embedding", err)
		}
		if len(tokenEmbedding) != pair.TargetArch.Hidden*bf16Size {
			return AssistantDraftBlockResult{}, core.NewError(core.Sprintf("native.assistant draft block target token embedding bytes = %d, want %d", len(tokenEmbedding), pair.TargetArch.Hidden*bf16Size))
		}
		projectedOut := target.mtpProjectionScratch(pair.Assistant.Arch.Hidden * bf16Size)
		projected, err := pair.Assistant.DraftInputProjectionInto(projectedOut, tokenEmbedding, currentHidden)
		if err != nil {
			return AssistantDraftBlockResult{}, err
		}
		normedOut := target.mtpDraftScratch(&target.mtpDraftNormed, pair.Assistant.Arch.Hidden*bf16Size)
		hiddenOut := target.mtpDraftScratch(&target.mtpDraftHidden, pair.TargetArch.Hidden*bf16Size)
		logitsOut := target.mtpDraftScratch(&target.mtpDraftLogits, pair.Assistant.Arch.Vocab*bf16Size)
		logitScores := target.mtpDraftLogitScoreScratch(pair.Assistant.NumCentroids)
		logitSelected := target.mtpDraftLogitSelectedScratch(pair.Assistant.CentroidIntermediateTopK)
		target.mtpDraftLayerScratch.usePinnedBacking()
		step, err := pair.draftStepFromProjectedIntoWithSuppress(projected, targetKVs, normedOut, hiddenOut, logitsOut, logitScores, logitSelected, &target.mtpDraftLayerScratch, suppress)
		if err != nil {
			return AssistantDraftBlockResult{}, err
		}
		tokens = append(tokens, step.Token)
		currentToken = step.Token
		currentHidden = step.Hidden
	}
	if !copyTokens {
		target.mtpDraftTokens = tokens
	}
	return AssistantDraftBlockResult{Tokens: tokens, Hidden: currentHidden}, nil
}

func (pair *AssistantPair) draftBlockSampledFromSessionWithSuppress(target *ArchSession, lastToken int32, maxDraftTokens int, copyTokens bool, params model.SampleParams, sampler *model.Sampler) (AssistantDraftBlockResult, error) {
	if pair == nil || pair.Assistant == nil {
		return AssistantDraftBlockResult{}, core.NewError("native.assistant sampled draft block requires a validated pair")
	}
	if sampler == nil {
		return AssistantDraftBlockResult{}, core.NewError("native.assistant sampled draft block sampler is nil")
	}
	if maxDraftTokens <= 0 {
		return AssistantDraftBlockResult{}, core.NewError("native.assistant sampled draft block maxDraftTokens must be > 0")
	}
	if lastToken < 0 {
		return AssistantDraftBlockResult{}, core.NewError("native.assistant sampled draft step token is invalid")
	}
	if target == nil {
		return AssistantDraftBlockResult{}, core.NewError("native.assistant sampled draft step target session is nil")
	}
	if target.embed == nil && target.embedInto == nil {
		return AssistantDraftBlockResult{}, core.NewError("native.assistant sampled draft step target session has no embedder")
	}
	targetKVs, err := pair.targetKVByLayerTypeFromSessionScratch(target)
	if err != nil {
		return AssistantDraftBlockResult{}, err
	}
	currentHidden, err := target.boundaryNormedHiddenScratch()
	if err != nil {
		return AssistantDraftBlockResult{}, core.E("native.assistant sampled draft block", "target boundary hidden", err)
	}
	currentToken := lastToken
	var tokens []int32
	if copyTokens {
		tokens = make([]int32, 0, maxDraftTokens)
	} else {
		tokens = target.mtpDraftTokenScratch(maxDraftTokens)
	}
	for len(tokens) < maxDraftTokens {
		tokenEmbedding, err := target.embedID(currentToken)
		if err != nil {
			return AssistantDraftBlockResult{}, core.E("native.assistant sampled draft block", "target token embedding", err)
		}
		if len(tokenEmbedding) != pair.TargetArch.Hidden*bf16Size {
			return AssistantDraftBlockResult{}, core.NewError(core.Sprintf("native.assistant sampled draft block target token embedding bytes = %d, want %d", len(tokenEmbedding), pair.TargetArch.Hidden*bf16Size))
		}
		projectedOut := target.mtpProjectionScratch(pair.Assistant.Arch.Hidden * bf16Size)
		projected, err := pair.Assistant.DraftInputProjectionInto(projectedOut, tokenEmbedding, currentHidden)
		if err != nil {
			return AssistantDraftBlockResult{}, err
		}
		normedOut := target.mtpDraftScratch(&target.mtpDraftNormed, pair.Assistant.Arch.Hidden*bf16Size)
		hiddenOut := target.mtpDraftScratch(&target.mtpDraftHidden, pair.TargetArch.Hidden*bf16Size)
		logitsOut := target.mtpDraftScratch(&target.mtpDraftLogits, pair.Assistant.Arch.Vocab*bf16Size)
		logitScores := target.mtpDraftLogitScoreScratch(pair.Assistant.NumCentroids)
		logitSelected := target.mtpDraftLogitSelectedScratch(pair.Assistant.CentroidIntermediateTopK)
		target.mtpDraftLayerScratch.usePinnedBacking()
		step, err := pair.draftStepFromProjectedIntoWithSuppress(projected, targetKVs, normedOut, hiddenOut, logitsOut, logitScores, logitSelected, &target.mtpDraftLayerScratch, params.SuppressTokens)
		if err != nil {
			return AssistantDraftBlockResult{}, err
		}
		// drafts are ALWAYS the drafter's argmax — the reference
		// (SinglePositionMultiTokenCandidateGenerator) drafts greedily at every
		// temperature and leaves sampling entirely to the TARGET's verify side.
		// Sampling the drafter at the request temperature (the previous behaviour)
		// makes proposals random draws the sampled target almost never matches —
		// acceptance collapsed to 0% live. step.Token is that argmax (suppression
		// already applied).
		currentToken = step.Token
		tokens = append(tokens, currentToken)
		currentHidden = step.Hidden
	}
	if !copyTokens {
		target.mtpDraftTokens = tokens
	}
	return AssistantDraftBlockResult{Tokens: tokens, Hidden: currentHidden}, nil
}

// VerifyDraftBlockFromSession compares assistant draft tokens against the
// target session's greedy continuation, keeps the accepted prefix resident, and
// rolls back any rejected suffix. The caller commits ReplacementToken separately
// on reject, matching pkg/metal's assistant verifier contract.
func (pair *AssistantPair) VerifyDraftBlockFromSession(target *ArchSession, draftTokens []int32, suppressTokens ...[]int32) (AssistantVerifyResult, error) {
	return pair.verifyDraftBlockFromSessionWithSuppress(target, draftTokens, true, nativeAssistantSuppressArg(suppressTokens))
}

func (pair *AssistantPair) verifyDraftBlockFromSession(target *ArchSession, draftTokens []int32, copyOutputs bool, suppressTokens ...[]int32) (AssistantVerifyResult, error) {
	return pair.verifyDraftBlockFromSessionWithSuppress(target, draftTokens, copyOutputs, nativeAssistantSuppressArg(suppressTokens))
}

func (pair *AssistantPair) verifyDraftBlockFromSessionWithSuppress(target *ArchSession, draftTokens []int32, copyOutputs bool, suppress []int32) (AssistantVerifyResult, error) {
	if pair == nil {
		return AssistantVerifyResult{}, core.NewError("native.assistant verify requires a target pair")
	}
	if target == nil {
		return AssistantVerifyResult{}, core.NewError("native.assistant verify target session is nil")
	}
	if len(draftTokens) == 0 {
		return AssistantVerifyResult{}, core.NewError("native.assistant verify draft tokens are required")
	}
	if err := pair.validateTargetSessionArch(target.arch); err != nil {
		return AssistantVerifyResult{}, err
	}
	boundaryHidden := target.retainedHidden
	if copyOutputs {
		boundaryHidden = append([]byte(nil), target.retainedHidden...)
	}
	boundaryLogits, err := target.BoundaryLogits()
	if err != nil {
		return AssistantVerifyResult{}, core.E("native.assistant verify", "target boundary logits", err)
	}
	if copyOutputs {
		boundaryLogits = append([]byte(nil), boundaryLogits...)
	}
	first, err := greedyBF16Suppressed(boundaryLogits, target.arch.Vocab, suppress)
	if err != nil {
		return AssistantVerifyResult{}, core.E("native.assistant verify", "target boundary token", err)
	}

	posBefore := target.pos
	result := AssistantVerifyResult{}
	if copyOutputs {
		result.DraftedTokens = append([]int32(nil), draftTokens...)
	} else {
		result.DraftedTokens = draftTokens
	}
	if draftTokens[0] != first {
		if copyOutputs {
			result.TargetTokens = append(result.TargetTokens, first)
		}
		result.RejectedCount = len(draftTokens)
		if copyOutputs {
			result.RejectedTokens = append([]int32(nil), draftTokens...)
		} else {
			result.RejectedTokens = draftTokens
		}
		result.ReplacementToken = first
		target.pos = posBefore
		if err := target.truncateSpeculativeKV(target.pos); err != nil {
			return AssistantVerifyResult{}, err
		}
		target.rememberAssistantAcceptedIDs(posBefore, result.AcceptedTokens)
		if copyOutputs {
			target.rememberRetainedHidden(boundaryHidden)
			target.rememberRetainedLogits(boundaryLogits)
			result.Logits = append([]byte(nil), boundaryLogits...)
		}
		return result, nil
	}
	rows, hiddens, err := target.verifyAssistantDraftRows(draftTokens, suppress)
	if err != nil {
		return AssistantVerifyResult{}, err
	}
	if len(rows) < len(draftTokens) || len(hiddens) < len(draftTokens) {
		return AssistantVerifyResult{}, core.NewError("native.assistant verify target rows are incomplete")
	}

	accepted := 0
	for i, draft := range draftTokens {
		targetToken := first
		if i > 0 {
			targetToken = rows[i-1]
		}
		if copyOutputs && i == 0 {
			result.TargetTokens = append(result.TargetTokens, targetToken)
		}
		if targetToken != draft {
			break
		}
		accepted++
	}
	if copyOutputs {
		result.AcceptedTokens = append(result.AcceptedTokens, draftTokens[:accepted]...)
	} else {
		result.AcceptedTokens = draftTokens[:accepted]
	}
	result.AcceptedCount = accepted
	result.RejectedCount = len(draftTokens) - accepted
	result.AllAccepted = accepted == len(draftTokens)
	if !result.AllAccepted {
		if copyOutputs {
			result.RejectedTokens = append([]int32(nil), draftTokens[accepted:]...)
		} else {
			result.RejectedTokens = draftTokens[accepted:]
		}
		result.ReplacementToken = first
		if accepted > 0 {
			result.ReplacementToken = rows[accepted-1]
		}
	}

	if accepted == 0 {
		target.pos = posBefore
		if err := target.truncateSpeculativeKV(target.pos); err != nil {
			return AssistantVerifyResult{}, err
		}
		target.rememberAssistantAcceptedIDs(posBefore, result.AcceptedTokens)
		target.rememberRetainedHidden(boundaryHidden)
		target.rememberRetainedLogits(boundaryLogits)
		if copyOutputs {
			result.Logits = append([]byte(nil), boundaryLogits...)
		}
		return result, nil
	}

	// Adopt the boundary from the verify pass — the sampled lane's exact shape
	// (verifyDraftBlockSampledFromSession): the accepted prefix's KV rows are
	// already correct (batched/sequential verify parity), hiddens[accepted-1] IS
	// the hidden at the last accepted token, and rows[accepted-1] already set the
	// replacement above. Re-forwarding the accepted tokens (the old reforge) paid
	// `accepted` extra target forwards per accepting round — more target work per
	// committed token than plain decode, which kept MTP slower than plain even at
	// 67% acceptance.
	target.pos = posBefore + accepted
	if err := target.truncateSpeculativeKV(target.pos); err != nil {
		return AssistantVerifyResult{}, err
	}
	hidden := hiddens[accepted-1]
	if len(hidden) != target.arch.Hidden*bf16Size {
		return AssistantVerifyResult{}, core.NewError("native.assistant verify accepted hidden has wrong size")
	}
	logits, err := target.headLogitsScratch(hidden, false)
	if err != nil {
		return AssistantVerifyResult{}, core.E("native.assistant verify", "accepted boundary logits", err)
	}
	if copyOutputs {
		result.Hidden = append([]byte(nil), hidden...)
		result.Logits = append([]byte(nil), logits...)
	}
	target.rememberRetainedHidden(hidden)
	target.rememberRetainedLogits(logits)
	target.rememberAssistantAcceptedIDs(posBefore, result.AcceptedTokens)
	return result, nil
}

// VerifyDraftBlockSampledFromSession compares assistant draft tokens against
// target-sampled decisions from the target session. When carry is true, block[0]
// is an already-emitted replacement token from the previous round and is
// accepted without consuming a sampler draw.
func (pair *AssistantPair) VerifyDraftBlockSampledFromSession(target *ArchSession, draftTokens []int32, sampler *model.Sampler, params model.SampleParams, carry bool) (AssistantVerifyResult, error) {
	return pair.verifyDraftBlockSampledFromSession(target, draftTokens, sampler, params, carry, true, nil)
}

func (pair *AssistantPair) verifyDraftBlockSampledFromSession(target *ArchSession, draftTokens []int32, sampler *model.Sampler, params model.SampleParams, carry, copyOutputs bool, history []int32) (AssistantVerifyResult, error) {
	if pair == nil {
		return AssistantVerifyResult{}, core.NewError("native.assistant sampled verify requires a target pair")
	}
	if target == nil {
		return AssistantVerifyResult{}, core.NewError("native.assistant sampled verify target session is nil")
	}
	if len(draftTokens) == 0 {
		return AssistantVerifyResult{}, core.NewError("native.assistant sampled verify draft tokens are required")
	}
	if sampler == nil {
		return AssistantVerifyResult{}, core.NewError("native.assistant sampled verify sampler is nil")
	}
	if err := pair.validateTargetSessionArch(target.arch); err != nil {
		return AssistantVerifyResult{}, err
	}
	boundaryHidden := append([]byte(nil), target.retainedHidden...)
	boundaryLogits, err := target.BoundaryLogits()
	if err != nil {
		return AssistantVerifyResult{}, core.E("native.assistant sampled verify", "target boundary logits", err)
	}
	boundaryLogits = append([]byte(nil), boundaryLogits...)

	posBefore := target.pos
	result := AssistantVerifyResult{}
	if copyOutputs {
		result.DraftedTokens = append([]int32(nil), draftTokens...)
	} else {
		result.DraftedTokens = draftTokens
	}
	hiddens, err := target.verifyAssistantDraftHiddens(draftTokens)
	if err != nil {
		return AssistantVerifyResult{}, err
	}
	if len(hiddens) < len(draftTokens) {
		return AssistantVerifyResult{}, core.NewError("native.assistant sampled verify target rows are incomplete")
	}

	accepted := 0
	verifyHistory := history
	for i, draft := range draftTokens {
		if i == 0 && carry {
			accepted++
			continue
		}
		var targetToken int32
		if i == 0 {
			targetToken, err = target.sampleMTPTokenFromHidden(boundaryHidden, sampler, params, verifyHistory)
			if err != nil {
				return AssistantVerifyResult{}, core.E("native.assistant sampled verify", "sample verifier boundary", err)
			}
		} else {
			targetToken, err = target.sampleMTPTokenFromDenseBatchRowOrHidden(i-1, hiddens[i-1], sampler, params, verifyHistory)
			if err != nil {
				return AssistantVerifyResult{}, core.E("native.assistant sampled verify", "sample verifier row", err)
			}
		}
		if len(result.TargetTokens) == 0 {
			result.TargetTokens = append(result.TargetTokens, targetToken)
		}
		if targetToken != draft {
			result.ReplacementToken = targetToken
			break
		}
		accepted++
		if params.RepeatPenalty > 1 {
			verifyHistory = append(verifyHistory, targetToken)
		}
	}
	if copyOutputs {
		result.AcceptedTokens = append(result.AcceptedTokens, draftTokens[:accepted]...)
	} else {
		result.AcceptedTokens = draftTokens[:accepted]
	}
	result.AcceptedCount = accepted
	result.RejectedCount = len(draftTokens) - accepted
	result.AllAccepted = accepted == len(draftTokens)
	if !result.AllAccepted {
		if copyOutputs {
			result.RejectedTokens = append([]int32(nil), draftTokens[accepted:]...)
		} else {
			result.RejectedTokens = draftTokens[accepted:]
		}
	}

	target.pos = posBefore + accepted
	if err := target.truncateSpeculativeKV(target.pos); err != nil {
		return AssistantVerifyResult{}, err
	}
	target.rememberAssistantAcceptedIDs(posBefore, result.AcceptedTokens)

	if accepted == 0 {
		target.rememberRetainedHidden(boundaryHidden)
		target.rememberRetainedLogits(boundaryLogits)
		if copyOutputs {
			result.Logits = append([]byte(nil), boundaryLogits...)
		}
		return result, nil
	}

	hidden := hiddens[accepted-1]
	if len(hidden) != target.arch.Hidden*bf16Size {
		return AssistantVerifyResult{}, core.NewError("native.assistant sampled verify accepted hidden has wrong size")
	}
	logits, err := target.headLogitsScratch(hidden, false)
	if err != nil {
		return AssistantVerifyResult{}, core.E("native.assistant sampled verify", "accepted logits", err)
	}
	if copyOutputs {
		result.Hidden = append([]byte(nil), hidden...)
		result.Logits = append([]byte(nil), logits...)
	}
	target.rememberRetainedHidden(hidden)
	target.rememberRetainedLogits(logits)
	return result, nil
}

// GenerateFromSession greedily generates token ids from a native target session
// using this assistant pair for speculative proposals.
func (pair *AssistantPair) GenerateFromSession(target *ArchSession, promptIDs []int32, maxNew, eosID, draftTokens int, suppress []int32) (AssistantGenerateResult, error) {
	return pair.GenerateFromSessionEach(target, promptIDs, maxNew, eosID, draftTokens, suppress, nil)
}

// GenerateFromSessionEach is GenerateFromSession with per-token streaming.
func (pair *AssistantPair) GenerateFromSessionEach(target *ArchSession, promptIDs []int32, maxNew, eosID, draftTokens int, suppress []int32, yield AssistantTokenSink) (AssistantGenerateResult, error) {
	if pair == nil || pair.Assistant == nil {
		return AssistantGenerateResult{}, core.NewError("native.assistant generation requires a validated pair")
	}
	if target == nil {
		return AssistantGenerateResult{}, core.NewError("native.assistant generation target session is nil")
	}
	if len(promptIDs) == 0 {
		return AssistantGenerateResult{}, core.NewError("native.assistant generation prompt tokens are required")
	}
	if maxNew <= 0 {
		return AssistantGenerateResult{}, core.NewError("native.assistant generation maxNew must be > 0")
	}
	draftTokens = nativeAssistantResolveDraftTokens(draftTokens)
	if err := pair.validateTargetSessionArch(target.arch); err != nil {
		return AssistantGenerateResult{}, err
	}
	if err := target.prepareAssistantPrompt(promptIDs); err != nil {
		return AssistantGenerateResult{}, err
	}

	result := newAssistantGenerateResult(len(promptIDs), maxNew, draftTokens)
	lastToken := promptIDs[len(promptIDs)-1]
	carryLead := int32(-1)
	stopped := false
	lowAcceptStreak := 0
	for len(result.Tokens) < maxNew && !stopped {
		remaining := maxNew - len(result.Tokens)
		blockSize := draftTokens
		if blockSize > remaining {
			blockSize = remaining
		}
		draft, err := pair.draftBlockFromSessionWithSuppress(target, lastToken, blockSize, false, suppress)
		if err != nil {
			return result, err
		}
		result.DraftCalls++
		result.DraftTokens += len(draft.Tokens)
		result.DraftTokenSchedule = append(result.DraftTokenSchedule, blockSize)

		block := draft.Tokens
		carryPresent := carryLead >= 0
		if carryPresent {
			block = target.mtpDraftVerifyBlockScratch(carryLead, draft.Tokens)
		}
		posBeforeVerify := target.pos
		verify, err := pair.verifyDraftBlockFromSessionWithSuppress(target, block, false, suppress)
		if err != nil {
			return result, err
		}
		result.TargetVerifyCalls++
		result.TargetCalls++
		emitStart := 0
		if carryPresent && len(verify.AcceptedTokens) > 0 && verify.AcceptedTokens[0] == carryLead {
			emitStart = 1
			carryLead = -1
		}
		newDrafts := 0
		keptAccepted := emitStart
		result.RejectedTokens += verify.RejectedCount
		for _, id := range verify.AcceptedTokens[emitStart:] {
			keptAccepted++
			beforeTokens := len(result.Tokens)
			if nativeAssistantEmitToken(&result, id, eosID, yield) {
				stopped = true
			}
			if len(result.Tokens) > beforeTokens {
				lastToken = id
				newDrafts++
			}
			if stopped {
				break
			}
		}
		result.AcceptedTokens += newDrafts
		result.TargetTokens += newDrafts
		if stopped {
			if err := nativeAssistantRollbackAccepted(target, posBeforeVerify, verify.AcceptedTokens, keptAccepted); err != nil {
				return result, err
			}
			break
		}
		if len(result.Tokens) >= maxNew {
			break
		}
		if verify.AllAccepted {
			lowAcceptStreak = 0
			carryLead = -1
			continue
		}

		replacement := verify.ReplacementToken
		if nativeAssistantEmitToken(&result, replacement, eosID, yield) {
			stopped = true
		}
		result.TargetTokens++
		lastToken = replacement
		if nativeAssistantLowAcceptBlock(len(draft.Tokens), newDrafts) {
			lowAcceptStreak++
		} else {
			lowAcceptStreak = 0
		}
		// Give up on drafting only after the drafter has stayed weak for several
		// consecutive blocks — one near-tie block is transient, not a mismatched pair.
		if !stopped && len(result.Tokens) < maxNew && lowAcceptStreak >= nativeAssistantLowAcceptPatience {
			if err := nativeAssistantFinishLowAcceptFromTargetCache(target, &result, replacement, maxNew, eosID, suppress, yield); err != nil {
				return result, err
			}
			break
		}
		carryLead = replacement
	}
	if carryLead >= 0 && !stopped && yield == nil {
		if _, err := target.stepID(carryLead); err != nil {
			return result, err
		}
		result.TargetCalls++
	}
	return result, nil
}

// GenerateSampledFromSession samples token ids from a native target session
// while using this assistant pair for speculative proposals. The target sampler
// decides every committed token; assistant proposals only affect acceptance.
func (pair *AssistantPair) GenerateSampledFromSession(target *ArchSession, promptIDs []int32, maxNew int, stopTokens []int32, sampler *model.Sampler, params model.SampleParams, draftTokens int) (AssistantGenerateResult, error) {
	return pair.GenerateSampledFromSessionEach(target, promptIDs, maxNew, stopTokens, sampler, params, draftTokens, nil)
}

// GenerateSampledFromSessionEach is GenerateSampledFromSession with per-token
// streaming.
func (pair *AssistantPair) GenerateSampledFromSessionEach(target *ArchSession, promptIDs []int32, maxNew int, stopTokens []int32, sampler *model.Sampler, params model.SampleParams, draftTokens int, yield AssistantTokenSink) (AssistantGenerateResult, error) {
	if pair == nil || pair.Assistant == nil {
		return AssistantGenerateResult{}, core.NewError("native.assistant sampled generation requires a validated pair")
	}
	if target == nil {
		return AssistantGenerateResult{}, core.NewError("native.assistant sampled generation target session is nil")
	}
	if sampler == nil {
		return AssistantGenerateResult{}, core.NewError("native.assistant sampled generation sampler is nil")
	}
	if len(promptIDs) == 0 {
		return AssistantGenerateResult{}, core.NewError("native.assistant sampled generation prompt tokens are required")
	}
	if maxNew <= 0 {
		return AssistantGenerateResult{}, core.NewError("native.assistant sampled generation maxNew must be > 0")
	}
	draftTokens = nativeAssistantResolveDraftTokens(draftTokens)
	if err := pair.validateTargetSessionArch(target.arch); err != nil {
		return AssistantGenerateResult{}, err
	}
	if err := target.prepareAssistantPrompt(promptIDs); err != nil {
		return AssistantGenerateResult{}, err
	}

	result := newAssistantGenerateResult(len(promptIDs), maxNew, draftTokens)
	lastToken := promptIDs[len(promptIDs)-1]
	carryLead := int32(-1)
	stopped := false
	history := target.sampleHistoryScratchFor(params, maxNew)
	finalHistory := history
	draftSampler := model.NewSampler(0)
	lowAcceptStreak := 0
	defer func() { target.sampleHistory = finalHistory }()
	for len(result.Tokens) < maxNew && !stopped {
		remaining := maxNew - len(result.Tokens)
		blockSize := draftTokens
		if blockSize > remaining {
			blockSize = remaining
		}
		pickParams := target.mtpSamplePickParams(params, stopTokens, len(result.Tokens))
		draft, err := pair.draftBlockSampledFromSessionWithSuppress(target, lastToken, blockSize, false, pickParams, draftSampler)
		if err != nil {
			return result, err
		}
		result.DraftCalls++
		result.DraftTokens += len(draft.Tokens)
		result.DraftTokenSchedule = append(result.DraftTokenSchedule, blockSize)

		block := draft.Tokens
		carryPresent := carryLead >= 0
		if carryPresent {
			block = target.mtpDraftVerifyBlockScratch(carryLead, draft.Tokens)
		}
		posBeforeVerify := target.pos
		verify, err := pair.verifyDraftBlockSampledFromSession(target, block, sampler, pickParams, carryPresent, false, history)
		if err != nil {
			return result, err
		}
		result.TargetVerifyCalls++
		result.TargetCalls++
		emitStart := 0
		if carryPresent && len(verify.AcceptedTokens) > 0 && verify.AcceptedTokens[0] == carryLead {
			emitStart = 1
			carryLead = -1
		}
		newDrafts := 0
		keptAccepted := emitStart
		result.RejectedTokens += verify.RejectedCount
		for _, id := range verify.AcceptedTokens[emitStart:] {
			keptAccepted++
			beforeTokens := len(result.Tokens)
			if nativeAssistantEmitSampledToken(&result, id, stopTokens, yield) {
				stopped = true
			}
			if len(result.Tokens) > beforeTokens {
				lastToken = id
				newDrafts++
				if params.RepeatPenalty > 1 {
					history = append(history, id)
					finalHistory = history
				}
			}
			if stopped {
				break
			}
		}
		result.AcceptedTokens += newDrafts
		result.TargetTokens += newDrafts
		if stopped {
			if err := nativeAssistantRollbackAccepted(target, posBeforeVerify, verify.AcceptedTokens, keptAccepted); err != nil {
				return result, err
			}
			break
		}
		if len(result.Tokens) >= maxNew {
			break
		}
		if verify.AllAccepted {
			lowAcceptStreak = 0
			carryLead = -1
			continue
		}

		replacement := verify.ReplacementToken
		result.Tokens = append(result.Tokens, replacement)
		yieldStopped := yield != nil && !yield(replacement)
		stopToken := nativeTokenInSet(replacement, stopTokens)
		if yieldStopped || stopToken {
			stopped = true
		}
		if params.RepeatPenalty > 1 {
			history = append(history, replacement)
			finalHistory = history
		}
		result.TargetTokens++
		lastToken = replacement
		if stopToken && !yieldStopped {
			if err := target.commitAssistantReplacement(replacement); err != nil {
				return result, err
			}
			result.TargetCalls++
			carryLead = -1
			continue
		}
		if nativeAssistantLowAcceptBlock(len(draft.Tokens), newDrafts) {
			lowAcceptStreak++
		} else {
			lowAcceptStreak = 0
		}
		// One weak block is a transient near-tie, not a mismatched pair — only fall
		// back to plain target decode after several consecutive weak blocks.
		if !stopped && len(result.Tokens) < maxNew && lowAcceptStreak >= nativeAssistantLowAcceptPatience {
			var err error
			history, err = nativeAssistantFinishLowAcceptSampledFromTargetCache(target, &result, replacement, maxNew, stopTokens, sampler, params, history, yield)
			if err != nil {
				return result, err
			}
			finalHistory = history
			break
		}
		carryLead = replacement
	}
	if carryLead >= 0 && !stopped && yield == nil {
		if _, err := target.stepID(carryLead); err != nil {
			return result, err
		}
		result.TargetCalls++
	}
	return result, nil
}

func nativeAssistantResolveDraftTokens(draftTokens int) int {
	if draftTokens <= 0 {
		return nativeAssistantDefaultDraftTokens
	}
	return draftTokens
}

func (s *ArchSession) prepareAssistantPrompt(promptIDs []int32) error {
	if len(promptIDs) == 0 {
		return core.NewError("native.assistant generation prompt tokens are required")
	}
	if len(promptIDs) > s.maxLen {
		return core.NewError("native.assistant generation prompt would exceed maxLen cache rows")
	}
	if hidden := s.cachedPromptHiddenFor(promptIDs); hidden != nil {
		s.pos = len(promptIDs)
		if err := s.truncateSpeculativeKV(s.pos); err != nil {
			return err
		}
		resident := s.cachedIDs[:0]
		s.cachedIDs = append(resident, promptIDs...)
		s.rememberRetainedHidden(hidden)
		if logits := s.cachedPromptLogitsFor(promptIDs); logits != nil {
			s.rememberRetainedLogits(logits)
		}
		return nil
	}
	lcp := 0
	for lcp < len(promptIDs) && lcp < len(s.cachedIDs) && promptIDs[lcp] == s.cachedIDs[lcp] {
		lcp++
	}
	if lcp == len(promptIDs) {
		lcp = len(promptIDs) - 1
	}
	s.pos = lcp
	if err := s.truncateSpeculativeKV(s.pos); err != nil {
		return err
	}
	hidden, logits, err := s.prefillPromptCacheEntry(promptIDs[lcp:])
	if err != nil {
		s.cachedIDs = nil
		s.clearCachedPromptHidden()
		s.resetRetainedHidden()
		return err
	}
	resident := s.cachedIDs[:0]
	s.cachedIDs = append(resident, promptIDs...)
	s.rememberCachedPromptEntry(promptIDs, hidden, logits)
	s.rememberRetainedHidden(hidden)
	s.rememberRetainedLogits(logits)
	return nil
}

func nativeAssistantEmitToken(result *AssistantGenerateResult, id int32, eosID int, yield AssistantTokenSink) bool {
	if eosID >= 0 && int(id) == eosID {
		return true
	}
	result.Tokens = append(result.Tokens, id)
	if yield != nil && !yield(id) {
		return true
	}
	return false
}

func nativeAssistantLowAcceptBlock(drafted, accepted int) bool {
	return drafted > 0 && accepted*2 < drafted
}

func nativeAssistantFinishLowAcceptFromTargetCache(target *ArchSession, result *AssistantGenerateResult, replacement int32, maxNew, eosID int, suppress []int32, yield AssistantTokenSink) error {
	if err := target.commitAssistantReplacement(replacement); err != nil {
		return err
	}
	result.TargetCalls++
	remaining := maxNew - len(result.Tokens)
	if remaining <= 0 {
		return nil
	}
	tail, err := target.GenerateFromCacheEachWithSuppression(remaining, eosID, suppress, func(id int32) bool {
		return !nativeAssistantEmitToken(result, id, eosID, yield)
	})
	if err != nil {
		return err
	}
	result.TargetCalls++
	result.TargetTokens += len(tail)
	return nil
}

func nativeAssistantFinishLowAcceptSampledFromTargetCache(target *ArchSession, result *AssistantGenerateResult, replacement int32, maxNew int, stopTokens []int32, sampler *model.Sampler, params model.SampleParams, history []int32, yield AssistantTokenSink) ([]int32, error) {
	if err := target.commitAssistantReplacement(replacement); err != nil {
		return history, err
	}
	result.TargetCalls++
	remaining := maxNew - len(result.Tokens)
	if remaining <= 0 {
		return history, nil
	}
	if len(target.retainedHidden) != target.arch.Hidden*bf16Size {
		return history, core.NewError("native.assistant sampled low-accept fallback has no retained target hidden")
	}
	if target.pos+remaining > target.maxLen {
		return history, core.NewError("native.assistant sampled low-accept fallback would exceed maxLen cache rows")
	}
	var tail []int32
	finalHistory := history
	var err error
	withAutoreleasePool(func() {
		tail, finalHistory, err = target.generateSampledFromHiddenInPoolWithHistory(target.retainedHidden, remaining, stopTokens, sampler, params, nil, func(id int32) bool {
			return !nativeAssistantEmitSampledToken(result, id, stopTokens, yield)
		}, true, len(result.Tokens), history)
	})
	if err != nil {
		target.cachedIDs = nil
		target.resetRetainedHidden()
		return history, err
	}
	target.cachedIDs = append(target.cachedIDs, tail...)
	result.TargetCalls++
	result.TargetTokens += len(tail)
	return finalHistory, nil
}

func nativeAssistantEmitSampledToken(result *AssistantGenerateResult, id int32, stopTokens []int32, yield AssistantTokenSink) bool {
	result.Tokens = append(result.Tokens, id)
	return (yield != nil && !yield(id)) || nativeTokenInSet(id, stopTokens)
}

func nativeAssistantRollbackAccepted(target *ArchSession, posBefore int, accepted []int32, keep int) error {
	if target == nil || keep >= len(accepted) {
		return nil
	}
	if keep < 0 {
		keep = 0
	}
	if keep == 0 {
		target.pos = posBefore
		return target.truncateSpeculativeKV(target.pos)
	}
	return target.retainMTPCommittedBoundary(posBefore, accepted[:keep])
}

func (s *ArchSession) commitAssistantReplacement(id int32) error {
	if s == nil {
		return core.NewError("native.assistant replacement commit target session is nil")
	}
	posBefore := s.pos
	hidden, err := s.stepID(id)
	if err != nil {
		return err
	}
	s.rememberRetainedHidden(hidden)
	s.rememberAssistantAcceptedIDs(posBefore, []int32{id})
	return nil
}

func (s *ArchSession) verifyAssistantDraftRows(draftTokens, suppress []int32) ([]int32, [][]byte, error) {
	hiddens, err := s.verifyAssistantDraftHiddens(draftTokens)
	if err != nil {
		return nil, nil, err
	}
	rows := s.mtpVerifyRowScratch(len(draftTokens))
	if len(hiddens) != len(draftTokens) {
		return nil, nil, core.NewError("native.assistant verify target rows are incomplete")
	}
	for i, hidden := range hiddens {
		token, err := s.greedyFromHiddenInPool(hidden, suppress)
		if err != nil {
			return nil, nil, err
		}
		rows[i] = token
	}
	return rows, hiddens, nil
}

func (s *ArchSession) verifyAssistantDraftHiddens(draftTokens []int32) ([][]byte, error) {
	hiddens, batched, err := s.verifyBatchedHiddens(draftTokens)
	if err != nil {
		return nil, err
	}
	if batched {
		if len(hiddens) != len(draftTokens) {
			return nil, core.NewError("native.assistant verify batched target rows are incomplete")
		}
		return hiddens, nil
	}

	rowBytes := s.arch.Hidden * bf16Size
	if rows, ok := s.mtpVerifyHiddenRowsScratch(len(draftTokens), rowBytes); ok {
		for i, draft := range draftTokens {
			hidden, err := s.stepID(draft)
			if err != nil {
				return nil, err
			}
			if len(hidden) != rowBytes {
				return nil, core.NewError("native.assistant verify sequential hidden has wrong size")
			}
			copy(rows[i], hidden)
		}
		return rows, nil
	}

	hiddens = make([][]byte, 0, len(draftTokens))
	for _, draft := range draftTokens {
		hidden, err := s.stepID(draft)
		if err != nil {
			return nil, err
		}
		hiddens = append(hiddens, append([]byte(nil), hidden...))
	}
	return hiddens, nil
}

func (s *ArchSession) rememberAssistantAcceptedIDs(posBefore int, accepted []int32) {
	if s == nil {
		return
	}
	if posBefore < 0 || len(s.cachedIDs) < posBefore {
		s.cachedIDs = nil
		return
	}
	s.cachedIDs = s.cachedIDs[:posBefore]
	s.cachedIDs = append(s.cachedIDs, accepted...)
}

func (pair *AssistantPair) draftStepFromProjected(projected []byte, targetKVs AssistantTargetKVByType, suppressTokens ...[]int32) (AssistantDraftStepResult, error) {
	return pair.draftStepFromProjectedWithSuppress(projected, targetKVs, nativeAssistantSuppressArg(suppressTokens))
}

func (pair *AssistantPair) draftStepFromProjectedInto(projected []byte, targetKVs AssistantTargetKVByType, normedOut, hiddenOut, logitsOut []byte, logitScores []float32, logitSelected []int, layerScratch *assistantDraftLayerScratch, suppressTokens ...[]int32) (AssistantDraftStepResult, error) {
	return pair.draftStepFromProjectedIntoWithSuppress(projected, targetKVs, normedOut, hiddenOut, logitsOut, logitScores, logitSelected, layerScratch, nativeAssistantSuppressArg(suppressTokens))
}

func (pair *AssistantPair) draftStepFromProjectedWithSuppress(projected []byte, targetKVs AssistantTargetKVByType, suppress []int32) (AssistantDraftStepResult, error) {
	return pair.draftStepFromProjectedIntoWithSuppress(projected, targetKVs, nil, nil, nil, nil, nil, nil, suppress)
}

func (pair *AssistantPair) draftStepFromProjectedIntoWithSuppress(projected []byte, targetKVs AssistantTargetKVByType, normedOut, hiddenOut, logitsOut []byte, logitScores []float32, logitSelected []int, layerScratch *assistantDraftLayerScratch, suppress []int32) (AssistantDraftStepResult, error) {
	if pair == nil || pair.Assistant == nil {
		return AssistantDraftStepResult{}, core.NewError("native.assistant draft step requires a validated pair")
	}
	normed, hidden, err := pair.Assistant.draftStepActivationsIntoScratch(normedOut, hiddenOut, projected, targetKVs, layerScratch)
	if err != nil {
		return AssistantDraftStepResult{}, err
	}
	logits, err := pair.Assistant.draftLogitsIntoScratch(logitsOut, normed, logitScores, logitSelected)
	if err != nil {
		return AssistantDraftStepResult{}, err
	}
	token, err := pair.Assistant.draftGreedyTokenWithSuppress(logits, suppress)
	if err != nil {
		return AssistantDraftStepResult{}, err
	}
	return AssistantDraftStepResult{Logits: logits, Token: token, Hidden: hidden}, nil
}

func (pair *AssistantPair) validateDraftInputTarget() (model.Arch, error) {
	if pair == nil || pair.Assistant == nil {
		return model.Arch{}, core.NewError("native.assistant draft input requires a validated pair")
	}
	target := pair.TargetArch
	if target.Hidden <= 0 || target.Vocab <= 0 {
		return model.Arch{}, core.NewError("native.assistant draft input target arch is incomplete")
	}
	if pair.Assistant.BackboneHiddenSize != target.Hidden {
		return model.Arch{}, core.NewError(core.Sprintf("native.assistant backbone_hidden_size = %d, want target hidden_size %d", pair.Assistant.BackboneHiddenSize, target.Hidden))
	}
	return target, nil
}

func (m *AssistantModel) DraftOutputProjection(assistantHidden []byte) ([]byte, error) {
	return m.DraftOutputProjectionInto(nil, assistantHidden)
}

func (m *AssistantModel) DraftFinalNorm(hiddenStates []byte) ([]byte, error) {
	return m.DraftFinalNormInto(nil, hiddenStates)
}

func (m *AssistantModel) DraftAttention(layerIdx int, hiddenStates []byte, targetKV AssistantTargetKV) ([]byte, error) {
	return m.DraftAttentionInto(nil, layerIdx, hiddenStates, targetKV)
}

func (m *AssistantModel) DraftAttentionInto(out []byte, layerIdx int, hiddenStates []byte, targetKV AssistantTargetKV) ([]byte, error) {
	return m.draftAttentionIntoScratch(out, layerIdx, hiddenStates, targetKV, nil)
}

func (m *AssistantModel) draftAttentionIntoScratch(out []byte, layerIdx int, hiddenStates []byte, targetKV AssistantTargetKV, scratch *assistantDraftLayerScratch) ([]byte, error) {
	if scratch == nil {
		scratch = &assistantDraftLayerScratch{}
	}
	layer, nHeads, headDim, err := m.validateDraftAttentionInput(layerIdx, hiddenStates, targetKV)
	if err != nil {
		return nil, err
	}
	kvHeads, err := nativeAssistantTargetKVHeads(targetKV, headDim)
	if err != nil {
		return nil, err
	}
	if nHeads%kvHeads != 0 {
		return nil, core.NewError(core.Sprintf("native.assistant draft attention heads = %d, want multiple of target kv heads %d", nHeads, kvHeads))
	}

	prefix := core.Sprintf("model.layers.%d.self_attn.", layerIdx)
	qProj, err := nativeAssistantBF16Matrix(m, prefix+"q_proj.weight", nHeads*headDim, m.Arch.Hidden)
	if err != nil {
		return nil, err
	}
	qNorm, err := nativeAssistantBF16Vector(m, prefix+"q_norm.weight", headDim)
	if err != nil {
		return nil, err
	}
	oProj, err := nativeAssistantBF16Matrix(m, prefix+"o_proj.weight", m.Arch.Hidden, nHeads*headDim)
	if err != nil {
		return nil, err
	}

	qBytes := nHeads * headDim * bf16Size
	q, err := MatVecBF16Into(scratch.bytes(assistantDraftScratchAttnQ, qBytes), qProj.Data, hiddenStates, nHeads*headDim, m.Arch.Hidden)
	if err != nil {
		return nil, core.E("native.assistant draft attention", "q_proj", err)
	}
	q, err = RMSNormBF16Into(scratch.bytes(assistantDraftScratchAttnQNorm, qBytes), q, qNorm.Data, nHeads, headDim, m.Arch.Eps)
	if err != nil {
		return nil, core.E("native.assistant draft attention", "q_norm", err)
	}
	// the draft query ropes at the LAST SEEN token's position (target pos-1), the
	// constant the drafter was trained with (HF SinglePositionMultiTokenCandidateGenerator:
	// position_ids = input_ids.shape[1]-1, never advanced across draft steps) — NOT the
	// KV capture-window start. Offset+Length-1 equals it for both stream types (full:
	// 0+pos-1; sliding: windowStart+count-1).
	qPos := targetKV.Offset + targetKV.Length - 1
	if qPos < 0 {
		qPos = 0
	}
	q, err = nativeAssistantRoPEInto(scratch.bytes(assistantDraftScratchAttnQRope, qBytes), q, m, layer, nHeads, headDim, qPos)
	if err != nil {
		return nil, err
	}
	attn, err := SDPAInto(scratch.bytes(assistantDraftScratchAttn, qBytes), q, targetKV.Key, targetKV.Value, 1, nHeads, kvHeads, headDim, targetKV.Length, nativeAssistantAttentionScale(m))
	if err != nil {
		return nil, core.E("native.assistant draft attention", "target kv sdpa", err)
	}
	return MatVecBF16Into(out, oProj.Data, attn, m.Arch.Hidden, nHeads*headDim)
}

func (m *AssistantModel) DraftLayer(layerIdx int, hiddenStates []byte, targetKV AssistantTargetKV) ([]byte, error) {
	return m.DraftLayerInto(nil, layerIdx, hiddenStates, targetKV)
}

func (m *AssistantModel) DraftStepActivations(projectedHidden []byte, targetKVs AssistantTargetKVByType) (normed []byte, targetHidden []byte, err error) {
	return m.DraftStepActivationsInto(nil, nil, projectedHidden, targetKVs)
}

func (m *AssistantModel) DraftStepActivationsInto(normedOut, targetHiddenOut []byte, projectedHidden []byte, targetKVs AssistantTargetKVByType) (normed []byte, targetHidden []byte, err error) {
	return m.draftStepActivationsIntoScratch(normedOut, targetHiddenOut, projectedHidden, targetKVs, nil)
}

func (m *AssistantModel) draftStepActivationsIntoScratch(normedOut, targetHiddenOut []byte, projectedHidden []byte, targetKVs AssistantTargetKVByType, scratch *assistantDraftLayerScratch) (normed []byte, targetHidden []byte, err error) {
	if m == nil {
		return nil, nil, core.NewError("native.assistant draft step model is nil")
	}
	hidden := m.Arch.Hidden
	if hidden <= 0 || len(m.Arch.Layer) == 0 {
		return nil, nil, core.NewError("native.assistant draft step has incomplete dimensions")
	}
	if len(projectedHidden) != hidden*bf16Size {
		return nil, nil, core.NewError(core.Sprintf("native.assistant draft step projected hidden bytes = %d, want %d", len(projectedHidden), hidden*bf16Size))
	}
	h := projectedHidden
	for idx := range m.Arch.Layer {
		layerType := m.Config.LayerType(idx)
		targetKV, ok := targetKVs.Get(layerType)
		if !ok || !targetKV.HasState() {
			return nil, nil, core.NewError("native.assistant draft step missing target K/V stream for " + layerType)
		}
		if scratch == nil {
			h, err = m.DraftLayer(idx, h, targetKV)
		} else {
			layerOut := scratch.bytes(assistantDraftScratchLayerOut, hidden*bf16Size)
			h, err = m.draftLayerIntoScratch(layerOut, idx, h, targetKV, scratch)
		}
		if err != nil {
			return nil, nil, err
		}
	}
	normed, err = m.DraftFinalNormInto(normedOut, h)
	if err != nil {
		return nil, nil, err
	}
	targetHidden, err = m.DraftOutputProjectionInto(targetHiddenOut, normed)
	if err != nil {
		return nil, nil, err
	}
	return normed, targetHidden, nil
}

func (m *AssistantModel) DraftLayerInto(out []byte, layerIdx int, hiddenStates []byte, targetKV AssistantTargetKV) ([]byte, error) {
	return m.draftLayerIntoScratch(out, layerIdx, hiddenStates, targetKV, nil)
}

func (m *AssistantModel) draftLayerIntoScratch(out []byte, layerIdx int, hiddenStates []byte, targetKV AssistantTargetKV, scratch *assistantDraftLayerScratch) ([]byte, error) {
	if scratch == nil {
		scratch = &assistantDraftLayerScratch{}
	}
	hidden, dFF, err := m.validateDraftLayerInput(layerIdx, hiddenStates)
	if err != nil {
		return nil, err
	}
	prefix := core.Sprintf("model.layers.%d", layerIdx)
	inputNorm, err := nativeAssistantBF16Vector(m, prefix+".input_layernorm.weight", hidden)
	if err != nil {
		return nil, err
	}
	postAttnNorm, err := nativeAssistantBF16Vector(m, prefix+".post_attention_layernorm.weight", hidden)
	if err != nil {
		return nil, err
	}
	preFFNorm, err := nativeAssistantBF16Vector(m, prefix+".pre_feedforward_layernorm.weight", hidden)
	if err != nil {
		return nil, err
	}
	postFFNorm, err := nativeAssistantBF16Vector(m, prefix+".post_feedforward_layernorm.weight", hidden)
	if err != nil {
		return nil, err
	}
	gateProj, err := nativeAssistantBF16Matrix(m, prefix+".mlp.gate_proj.weight", dFF, hidden)
	if err != nil {
		return nil, err
	}
	upProj, err := nativeAssistantBF16Matrix(m, prefix+".mlp.up_proj.weight", dFF, hidden)
	if err != nil {
		return nil, err
	}
	downProj, err := nativeAssistantBF16Matrix(m, prefix+".mlp.down_proj.weight", hidden, dFF)
	if err != nil {
		return nil, err
	}
	layerScalar, err := nativeAssistantLayerScalar(m, prefix, hidden)
	if err != nil {
		return nil, err
	}

	hiddenBytes := hidden * bf16Size
	ffBytes := dFF * bf16Size
	normed, err := RMSNormBF16Into(scratch.bytes(assistantDraftScratchInputNorm, hiddenBytes), hiddenStates, inputNorm.Data, 1, hidden, m.Arch.Eps)
	if err != nil {
		return nil, core.E("native.assistant draft layer", "input norm", err)
	}
	attnOut, err := m.draftAttentionIntoScratch(scratch.bytes(assistantDraftScratchAttnOut, hiddenBytes), layerIdx, normed, targetKV, scratch)
	if err != nil {
		return nil, err
	}
	attnResidual, err := RMSNormBF16Into(scratch.bytes(assistantDraftScratchAttnResidual, hiddenBytes), attnOut, postAttnNorm.Data, 1, hidden, m.Arch.Eps)
	if err != nil {
		return nil, core.E("native.assistant draft layer", "post attention norm", err)
	}
	h := scratch.bytes(assistantDraftScratchResidual, hiddenBytes)
	if err := AddBF16Into(h, hiddenStates, attnResidual); err != nil {
		return nil, core.E("native.assistant draft layer", "attention residual", err)
	}

	ffIn, err := RMSNormBF16Into(scratch.bytes(assistantDraftScratchFFIn, hiddenBytes), h, preFFNorm.Data, 1, hidden, m.Arch.Eps)
	if err != nil {
		return nil, core.E("native.assistant draft layer", "pre feed-forward norm", err)
	}
	gate, err := MatVecBF16Into(scratch.bytes(assistantDraftScratchGate, ffBytes), gateProj.Data, ffIn, dFF, hidden)
	if err != nil {
		return nil, core.E("native.assistant draft layer", "mlp gate projection", err)
	}
	up, err := MatVecBF16Into(scratch.bytes(assistantDraftScratchUp, ffBytes), upProj.Data, ffIn, dFF, hidden)
	if err != nil {
		return nil, core.E("native.assistant draft layer", "mlp up projection", err)
	}
	gated := scratch.bytes(assistantDraftScratchGated, ffBytes)
	if err := GeluGateMulBF16Into(gated, gate, up); err != nil {
		return nil, core.E("native.assistant draft layer", "mlp gate activation", err)
	}
	ff, err := MatVecBF16Into(scratch.bytes(assistantDraftScratchFF, hiddenBytes), downProj.Data, gated, hidden, dFF)
	if err != nil {
		return nil, core.E("native.assistant draft layer", "mlp down projection", err)
	}
	ffResidual, err := RMSNormBF16Into(scratch.bytes(assistantDraftScratchFFResidual, hiddenBytes), ff, postFFNorm.Data, 1, hidden, m.Arch.Eps)
	if err != nil {
		return nil, core.E("native.assistant draft layer", "post feed-forward norm", err)
	}
	hNext := scratch.bytes(assistantDraftScratchNext, hiddenBytes)
	if err := AddBF16Into(hNext, h, ffResidual); err != nil {
		return nil, core.E("native.assistant draft layer", "feed-forward residual", err)
	}
	if len(layerScalar) == bf16Size {
		return nativeAssistantMulScalarInto(out, hNext, layerScalar)
	}
	if len(layerScalar) == len(hNext) {
		return nativeAssistantMulVectorInto(out, hNext, layerScalar)
	}
	return nativeAssistantCopyInto(out, hNext), nil
}

func (m *AssistantModel) validateDraftLayerInput(layerIdx int, hiddenStates []byte) (int, int, error) {
	if m == nil {
		return 0, 0, core.NewError("native.assistant draft layer model is nil")
	}
	if layerIdx < 0 || layerIdx >= len(m.Arch.Layer) {
		return 0, 0, core.NewError(core.Sprintf("native.assistant draft layer index = %d, want [0,%d)", layerIdx, len(m.Arch.Layer)))
	}
	hidden := m.Arch.Hidden
	dFF := m.Arch.FF
	if hidden <= 0 || dFF <= 0 {
		return 0, 0, core.NewError("native.assistant draft layer has incomplete dimensions")
	}
	if len(hiddenStates) != hidden*bf16Size {
		return 0, 0, core.NewError(core.Sprintf("native.assistant draft layer hidden bytes = %d, want %d", len(hiddenStates), hidden*bf16Size))
	}
	return hidden, dFF, nil
}

func (m *AssistantModel) validateDraftAttentionInput(layerIdx int, hiddenStates []byte, targetKV AssistantTargetKV) (model.LayerSpec, int, int, error) {
	if m == nil {
		return model.LayerSpec{}, 0, 0, core.NewError("native.assistant draft attention model is nil")
	}
	if layerIdx < 0 || layerIdx >= len(m.Arch.Layer) {
		return model.LayerSpec{}, 0, 0, core.NewError(core.Sprintf("native.assistant draft attention layer index = %d, want [0,%d)", layerIdx, len(m.Arch.Layer)))
	}
	hidden := m.Arch.Hidden
	nHeads := m.Arch.Heads
	layer := m.Arch.Layer[layerIdx]
	headDim := layer.HeadDim
	if headDim <= 0 {
		headDim = m.Arch.HeadDim
	}
	if hidden <= 0 || nHeads <= 0 || headDim <= 0 {
		return model.LayerSpec{}, 0, 0, core.NewError("native.assistant draft attention has incomplete dimensions")
	}
	if len(hiddenStates) != hidden*bf16Size {
		return model.LayerSpec{}, 0, 0, core.NewError(core.Sprintf("native.assistant draft attention hidden bytes = %d, want %d", len(hiddenStates), hidden*bf16Size))
	}
	if !targetKV.HasState() {
		return model.LayerSpec{}, 0, 0, core.NewError("native.assistant draft attention target K/V stream is empty")
	}
	if targetKV.HeadDim > 0 && targetKV.HeadDim != headDim {
		return model.LayerSpec{}, 0, 0, core.NewError(core.Sprintf("native.assistant draft attention target head_dim = %d, want %d", targetKV.HeadDim, headDim))
	}
	wantBytes := nativeAssistantTargetKVByteLen(targetKV, headDim)
	if wantBytes <= 0 {
		return model.LayerSpec{}, 0, 0, core.NewError("native.assistant draft attention target K/V geometry is incomplete")
	}
	if len(targetKV.Key) != wantBytes {
		return model.LayerSpec{}, 0, 0, core.NewError(core.Sprintf("native.assistant draft attention target key bytes = %d, want %d", len(targetKV.Key), wantBytes))
	}
	if len(targetKV.Value) != wantBytes {
		return model.LayerSpec{}, 0, 0, core.NewError(core.Sprintf("native.assistant draft attention target value bytes = %d, want %d", len(targetKV.Value), wantBytes))
	}
	return layer, nHeads, headDim, nil
}

func nativeAssistantTargetKVHeads(kv AssistantTargetKV, headDim int) (int, error) {
	if kv.KVHeads > 0 {
		return kv.KVHeads, nil
	}
	if kv.Length <= 0 || headDim <= 0 {
		return 0, core.NewError("native.assistant draft attention target K/V geometry is incomplete")
	}
	denom := kv.Length * headDim * bf16Size
	if denom <= 0 || len(kv.Key)%denom != 0 {
		return 0, core.NewError("native.assistant draft attention cannot infer target kv heads")
	}
	return len(kv.Key) / denom, nil
}

func nativeAssistantTargetKVByteLen(kv AssistantTargetKV, headDim int) int {
	kvHeads := kv.KVHeads
	if kvHeads <= 0 && kv.Length > 0 && headDim > 0 {
		denom := kv.Length * headDim * bf16Size
		if denom > 0 && len(kv.Key)%denom == 0 {
			kvHeads = len(kv.Key) / denom
		}
	}
	if kvHeads <= 0 || kv.Length <= 0 || headDim <= 0 {
		return 0
	}
	return kvHeads * kv.Length * headDim * bf16Size
}

func nativeAssistantRoPE(q []byte, m *AssistantModel, layer model.LayerSpec, nHeads, headDim, offset int) ([]byte, error) {
	return nativeAssistantRoPEInto(nil, q, m, layer, nHeads, headDim, offset)
}

func nativeAssistantRoPEInto(out []byte, q []byte, m *AssistantModel, layer model.LayerSpec, nHeads, headDim, offset int) ([]byte, error) {
	rotaryDim := nativeAssistantLayerRotaryDim(m, layer, headDim)
	scale := m.Arch.RopeScale
	if scale == 0 {
		scale = 1
	}
	if len(m.Arch.RopeFreqs) > 0 {
		out, err := RoPEFreqsBF16Into(out, q, 1, nHeads, headDim, rotaryDim, m.Arch.RopeFreqs, scale, offset, false)
		if err != nil {
			return nil, core.E("native.assistant draft attention", "q_rope", err)
		}
		return out, nil
	}
	base := nativeAssistantLayerRopeBase(m, layer)
	out, err := RoPEDimsBF16Into(out, q, 1, nHeads, headDim, rotaryDim, base, scale, offset, false)
	if err != nil {
		return nil, core.E("native.assistant draft attention", "q_rope", err)
	}
	return out, nil
}

func nativeAssistantLayerRotaryDim(m *AssistantModel, layer model.LayerSpec, headDim int) int {
	rotaryDim := m.Arch.RotaryDim
	if layer.Attention == model.SlidingAttention && m.Arch.RotaryDimLocal > 0 {
		rotaryDim = m.Arch.RotaryDimLocal
	}
	if rotaryDim <= 0 || rotaryDim > headDim {
		rotaryDim = headDim
	}
	return rotaryDim
}

func nativeAssistantLayerRopeBase(m *AssistantModel, layer model.LayerSpec) float32 {
	if layer.Attention == model.SlidingAttention && m.Arch.RopeLocalBase > 0 {
		return m.Arch.RopeLocalBase
	}
	if m.Arch.RopeBase > 0 {
		return m.Arch.RopeBase
	}
	return 10000
}

func nativeAssistantAttentionScale(m *AssistantModel) float32 {
	if m == nil || m.Arch.AttnScale == 0 {
		return 1
	}
	return m.Arch.AttnScale
}

func (m *AssistantModel) DraftFinalNormInto(out []byte, hiddenStates []byte) ([]byte, error) {
	if m == nil {
		return nil, core.NewError("native.assistant draft final norm model is nil")
	}
	hidden := m.Arch.Hidden
	if hidden <= 0 {
		return nil, core.NewError("native.assistant draft final norm hidden_size is invalid")
	}
	if len(hiddenStates) != hidden*bf16Size {
		return nil, core.NewError(core.Sprintf("native.assistant draft final norm hidden bytes = %d, want %d", len(hiddenStates), hidden*bf16Size))
	}
	weight, ok := m.Tensors["model.norm.weight"]
	if !ok {
		return nil, core.NewError("native.assistant draft final norm missing model.norm.weight")
	}
	if weight.Dtype != "BF16" {
		return nil, core.NewError("native.assistant draft final norm model.norm.weight dtype = " + weight.Dtype + ", want BF16")
	}
	if len(weight.Shape) != 1 || weight.Shape[0] != hidden {
		return nil, core.NewError(core.Sprintf("native.assistant draft final norm model.norm.weight shape = %v, want [%d]", weight.Shape, hidden))
	}
	if len(weight.Data) != hidden*bf16Size {
		return nil, core.NewError(core.Sprintf("native.assistant draft final norm model.norm.weight bytes = %d, want %d", len(weight.Data), hidden*bf16Size))
	}
	return RMSNormBF16Into(out, hiddenStates, weight.Data, 1, hidden, m.Arch.Eps)
}

func (m *AssistantModel) DraftOutputProjectionInto(out []byte, assistantHidden []byte) ([]byte, error) {
	if m == nil {
		return nil, core.NewError("native.assistant draft output model is nil")
	}
	hidden := m.Arch.Hidden
	backbone := m.BackboneHiddenSize
	if hidden <= 0 || backbone <= 0 {
		return nil, core.NewError("native.assistant draft output has incomplete dimensions")
	}
	hiddenBytes := hidden * bf16Size
	if len(assistantHidden) != hiddenBytes {
		return nil, core.NewError(core.Sprintf("native.assistant draft output assistant hidden bytes = %d, want %d", len(assistantHidden), hiddenBytes))
	}
	weight, ok := m.Tensors["post_projection.weight"]
	if !ok {
		return nil, core.NewError("native.assistant draft output missing post_projection.weight")
	}
	if weight.Dtype != "BF16" {
		return nil, core.NewError("native.assistant draft output post_projection.weight dtype = " + weight.Dtype + ", want BF16")
	}
	if len(weight.Shape) < 2 || weight.Shape[len(weight.Shape)-2] != backbone || weight.Shape[len(weight.Shape)-1] != hidden {
		return nil, core.NewError(core.Sprintf("native.assistant draft output post_projection.weight shape = %v, want [%d %d]", weight.Shape, backbone, hidden))
	}
	if len(weight.Data) != backbone*hidden*bf16Size {
		return nil, core.NewError(core.Sprintf("native.assistant draft output post_projection.weight bytes = %d, want %d", len(weight.Data), backbone*hidden*bf16Size))
	}
	return MatMulBF16NTInto(out, assistantHidden, weight.Data, 1, hidden, backbone)
}

func (m *AssistantModel) DraftLogits(hiddenStates []byte) ([]byte, error) {
	return m.DraftLogitsInto(nil, hiddenStates)
}

func (m *AssistantModel) DraftLogitsInto(out []byte, hiddenStates []byte) ([]byte, error) {
	return m.draftLogitsIntoScratch(out, hiddenStates, nil, nil)
}

func (m *AssistantModel) draftLogitsIntoScratch(out []byte, hiddenStates []byte, scores []float32, selected []int) ([]byte, error) {
	if m == nil {
		return nil, core.NewError("native.assistant logits model is nil")
	}
	hidden := m.Arch.Hidden
	vocab := m.Arch.Vocab
	if hidden <= 0 || vocab <= 0 {
		return nil, core.NewError("native.assistant logits have incomplete dimensions")
	}
	if len(hiddenStates) != hidden*bf16Size {
		return nil, core.NewError(core.Sprintf("native.assistant logits hidden bytes = %d, want %d", len(hiddenStates), hidden*bf16Size))
	}
	if m.UseOrderedEmbeddings {
		return m.draftOrderedLogitsIntoScratch(out, hiddenStates, scores, selected)
	}
	embed, err := nativeAssistantBF16Matrix(m, "model.embed_tokens.weight", vocab, hidden)
	if err != nil {
		return nil, err
	}
	outLen := vocab * bf16Size
	if cap(out) < outLen {
		out = make([]byte, outLen)
	} else {
		out = out[:outLen]
	}
	for tokenID := 0; tokenID < vocab; tokenID++ {
		sum := nativeAssistantDotBF16Row(hiddenStates, embed.Data, tokenID, hidden)
		h := f32ToBF16(sum)
		off := tokenID * bf16Size
		out[off] = byte(h)
		out[off+1] = byte(h >> 8)
	}
	return out, nil
}

func (m *AssistantModel) draftOrderedLogitsInto(out []byte, hiddenStates []byte) ([]byte, error) {
	return m.draftOrderedLogitsIntoScratch(out, hiddenStates, nil, nil)
}

func (m *AssistantModel) draftOrderedLogitsIntoScratch(out []byte, hiddenStates []byte, scores []float32, selected []int) ([]byte, error) {
	hidden := m.Arch.Hidden
	vocab := m.Arch.Vocab
	numCentroids := m.NumCentroids
	topK := m.CentroidIntermediateTopK
	if numCentroids <= 0 || topK <= 0 || topK > numCentroids {
		return nil, core.NewError("native.assistant ordered embeddings centroid_intermediate_top_k is invalid")
	}
	if vocab%numCentroids != 0 {
		return nil, core.NewError("native.assistant token_ordering requires vocab_size divisible by num_centroids")
	}
	embed, err := nativeAssistantBF16Matrix(m, "model.embed_tokens.weight", vocab, hidden)
	if err != nil {
		return nil, err
	}
	centroids, err := nativeAssistantBF16Matrix(m, "masked_embedding.centroids.weight", numCentroids, hidden)
	if err != nil {
		return nil, err
	}
	ordering, ok := m.Tensors["masked_embedding.token_ordering"]
	if !ok {
		return nil, core.NewError("native.assistant ordered embeddings require masked_embedding.token_ordering")
	}
	vocabPerCentroid := vocab / numCentroids
	if err := nativeAssistantValidateOrdering(ordering, vocab, numCentroids, vocabPerCentroid); err != nil {
		return nil, err
	}

	if cap(scores) < numCentroids {
		scores = make([]float32, numCentroids)
	} else {
		scores = scores[:numCentroids]
	}
	for c := 0; c < numCentroids; c++ {
		scores[c] = nativeAssistantDotBF16Row(hiddenStates, centroids.Data, c, hidden)
	}
	selected = nativeAssistantTopKInto(selected, scores, topK)

	outLen := vocab * bf16Size
	if cap(out) < outLen {
		out = make([]byte, outLen)
	} else {
		out = out[:outLen]
	}
	floor := f32ToBF16(nativeAssistantLogitsFloor)
	for i := 0; i < vocab; i++ {
		out[i*bf16Size] = byte(floor)
		out[i*bf16Size+1] = byte(floor >> 8)
	}
	for _, centroid := range selected {
		for pos := 0; pos < vocabPerCentroid; pos++ {
			tokenID, err := nativeAssistantOrderingToken(ordering, centroid, pos, vocabPerCentroid)
			if err != nil {
				return nil, err
			}
			if tokenID < 0 || int(tokenID) >= vocab {
				return nil, core.NewError(core.Sprintf("native.assistant token_ordering token id = %d, want [0,%d)", tokenID, vocab))
			}
			sum := nativeAssistantDotBF16Row(hiddenStates, embed.Data, int(tokenID), hidden)
			h := f32ToBF16(sum)
			off := int(tokenID) * bf16Size
			out[off] = byte(h)
			out[off+1] = byte(h >> 8)
		}
	}
	return out, nil
}

func nativeAssistantBF16Matrix(m *AssistantModel, name string, rows, cols int) (safetensors.Tensor, error) {
	t, ok := m.Tensors[name]
	if !ok {
		return safetensors.Tensor{}, core.NewError("native.assistant missing " + name)
	}
	if t.Dtype != "BF16" {
		return safetensors.Tensor{}, core.NewError("native.assistant " + name + " dtype = " + t.Dtype + ", want BF16")
	}
	if len(t.Shape) < 2 || t.Shape[len(t.Shape)-2] != rows || t.Shape[len(t.Shape)-1] != cols {
		return safetensors.Tensor{}, core.NewError(core.Sprintf("native.assistant %s shape = %v, want [%d %d]", name, t.Shape, rows, cols))
	}
	if len(t.Data) != rows*cols*bf16Size {
		return safetensors.Tensor{}, core.NewError(core.Sprintf("native.assistant %s bytes = %d, want %d", name, len(t.Data), rows*cols*bf16Size))
	}
	return t, nil
}

func nativeAssistantBF16Vector(m *AssistantModel, name string, elems int) (safetensors.Tensor, error) {
	t, ok := m.Tensors[name]
	if !ok {
		return safetensors.Tensor{}, core.NewError("native.assistant missing " + name)
	}
	if t.Dtype != "BF16" {
		return safetensors.Tensor{}, core.NewError("native.assistant " + name + " dtype = " + t.Dtype + ", want BF16")
	}
	if len(t.Shape) != 1 || t.Shape[0] != elems {
		return safetensors.Tensor{}, core.NewError(core.Sprintf("native.assistant %s shape = %v, want [%d]", name, t.Shape, elems))
	}
	if len(t.Data) != elems*bf16Size {
		return safetensors.Tensor{}, core.NewError(core.Sprintf("native.assistant %s bytes = %d, want %d", name, len(t.Data), elems*bf16Size))
	}
	return t, nil
}

func nativeAssistantLayerScalar(m *AssistantModel, prefix string, hidden int) ([]byte, error) {
	for _, name := range []string{prefix + ".layer_scalar", prefix + ".layer_scalar.weight"} {
		t, ok := m.Tensors[name]
		if !ok || len(t.Data) == 0 {
			continue
		}
		if t.Dtype != "BF16" {
			return nil, core.NewError("native.assistant " + name + " dtype = " + t.Dtype + ", want BF16")
		}
		if len(t.Shape) == 1 && t.Shape[0] == 1 && len(t.Data) == bf16Size {
			return t.Data, nil
		}
		if len(t.Shape) == 1 && t.Shape[0] == hidden && len(t.Data) == hidden*bf16Size {
			return t.Data, nil
		}
		return nil, core.NewError(core.Sprintf("native.assistant %s shape = %v, want [1] or [%d]", name, t.Shape, hidden))
	}
	return nil, nil
}

func nativeAssistantMulScalarInto(out []byte, in, scalar []byte) ([]byte, error) {
	if cap(out) >= len(in) {
		out = out[:len(in)]
		if err := MulScalarBF16Into(out, in, scalar); err != nil {
			return nil, err
		}
		return out, nil
	}
	return MulScalarBF16(in, scalar)
}

func nativeAssistantMulVectorInto(out []byte, in, vec []byte) ([]byte, error) {
	if cap(out) >= len(in) {
		out = out[:len(in)]
		if err := MulBF16Into(out, in, vec); err != nil {
			return nil, err
		}
		return out, nil
	}
	return MulBF16(in, vec)
}

func nativeAssistantCopyInto(out []byte, in []byte) []byte {
	if cap(out) < len(in) {
		return in
	}
	out = out[:len(in)]
	copy(out, in)
	return out
}

func nativeAssistantValidateOrdering(t safetensors.Tensor, vocab, numCentroids, vocabPerCentroid int) error {
	switch t.Dtype {
	case "I32":
		if len(t.Data) != vocab*4 {
			return core.NewError(core.Sprintf("native.assistant token_ordering bytes = %d, want %d", len(t.Data), vocab*4))
		}
	case "I64":
		if len(t.Data) != vocab*8 {
			return core.NewError(core.Sprintf("native.assistant token_ordering bytes = %d, want %d", len(t.Data), vocab*8))
		}
	default:
		return core.NewError("native.assistant token_ordering dtype = " + t.Dtype + ", want int32 or int64")
	}
	if len(t.Shape) == 1 && t.Shape[0] == vocab {
		return nil
	}
	if len(t.Shape) == 2 && t.Shape[0] == numCentroids && t.Shape[1] == vocabPerCentroid {
		return nil
	}
	return core.NewError(core.Sprintf("native.assistant token_ordering shape = %v, want [%d] or [%d %d]", t.Shape, vocab, numCentroids, vocabPerCentroid))
}

func nativeAssistantOrderingToken(t safetensors.Tensor, centroid, pos, vocabPerCentroid int) (int32, error) {
	idx := centroid*vocabPerCentroid + pos
	switch t.Dtype {
	case "I32":
		off := idx * 4
		return int32(binary.LittleEndian.Uint32(t.Data[off:])), nil
	case "I64":
		off := idx * 8
		v := int64(binary.LittleEndian.Uint64(t.Data[off:]))
		if v < -2147483648 || v > 2147483647 {
			return 0, core.NewError(core.Sprintf("native.assistant token_ordering token id = %d, want int32 range", v))
		}
		return int32(v), nil
	default:
		return 0, core.NewError("native.assistant token_ordering dtype = " + t.Dtype + ", want int32 or int64")
	}
}

func nativeAssistantDotBF16Row(vec, rows []byte, row, cols int) float32 {
	base := row * cols * bf16Size
	var sum float32
	for i := 0; i < cols; i++ {
		vo := i * bf16Size
		wo := base + i*bf16Size
		sum += bf16ToF32(vec[vo], vec[vo+1]) * bf16ToF32(rows[wo], rows[wo+1])
	}
	return sum
}

func nativeAssistantTopK(scores []float32, k int) []int {
	return nativeAssistantTopKInto(nil, scores, k)
}

func nativeAssistantTopKInto(selected []int, scores []float32, k int) []int {
	if cap(selected) < k {
		selected = make([]int, 0, k)
	} else {
		selected = selected[:0]
	}
	for idx, score := range scores {
		pos := len(selected)
		for pos > 0 && score > scores[selected[pos-1]] {
			pos--
		}
		if pos >= k {
			continue
		}
		selected = append(selected, 0)
		copy(selected[pos+1:], selected[pos:len(selected)-1])
		selected[pos] = idx
		if len(selected) > k {
			selected = selected[:k]
		}
	}
	return selected
}

func (m *AssistantModel) DraftGreedyToken(logits []byte, suppressTokens ...[]int32) (int32, error) {
	return m.draftGreedyTokenWithSuppress(logits, nativeAssistantSuppressArg(suppressTokens))
}

func (m *AssistantModel) draftGreedyTokenWithSuppress(logits []byte, suppressed []int32) (int32, error) {
	if m == nil {
		return 0, core.NewError("native.assistant greedy token model is nil")
	}
	vocab := m.Arch.Vocab
	if vocab <= 0 {
		return 0, core.NewError("native.assistant greedy token vocab_size is invalid")
	}
	if len(logits) != vocab*bf16Size {
		return 0, core.NewError(core.Sprintf("native.assistant greedy token logits bytes = %d, want %d", len(logits), vocab*bf16Size))
	}
	var bestID int32 = -1
	var best float32
	for id := 0; id < vocab; id++ {
		if nativeAssistantSuppressed(int32(id), suppressed) {
			continue
		}
		v := bf16ToF32(logits[id*bf16Size], logits[id*bf16Size+1])
		if bestID < 0 || v > best {
			bestID = int32(id)
			best = v
		}
	}
	if bestID < 0 {
		return 0, core.NewError("native.assistant greedy token produced no token")
	}
	return bestID, nil
}

func nativeAssistantSuppressed(id int32, suppressTokens []int32) bool {
	for _, suppressed := range suppressTokens {
		if suppressed >= 0 && suppressed == id {
			return true
		}
	}
	return false
}

func (pair *AssistantPair) Close() error {
	if pair == nil || pair.Assistant == nil {
		return nil
	}
	err := pair.Assistant.Close()
	pair.Assistant = nil
	return err
}
