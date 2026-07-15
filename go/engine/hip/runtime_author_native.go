// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import (
	"context"
	"strconv"
	"strings"

	core "dappco.re/go"
	"dappco.re/go/inference"
	rocmmodel "dappco.re/go/inference/engine/hip/model"
)

// ROCmPromptCacheEntry is ROCm's portable runtime-author prompt-cache entry.
// It carries token prefixes plus cache/state refs, not backend tensor handles.
type ROCmPromptCacheEntry struct {
	Tokens        []int32                   `json:"tokens,omitempty"`
	CacheBlocks   []inference.CacheBlockRef `json:"cache_blocks,omitempty"`
	HiddenRefs    []inference.StateRef      `json:"hidden_refs,omitempty"`
	LogitRefs     []inference.StateRef      `json:"logit_refs,omitempty"`
	ModelHash     string                    `json:"model_hash,omitempty"`
	AdapterHash   string                    `json:"adapter_hash,omitempty"`
	TokenizerHash string                    `json:"tokenizer_hash,omitempty"`
	Labels        map[string]string         `json:"labels,omitempty"`
}

// NewROCmPromptCacheEntry builds a portable prompt-cache entry for runtime
// authors that already produced cache/state refs.
func NewROCmPromptCacheEntry(tokens []int32, blocks []inference.CacheBlockRef, hiddenRefs, logitRefs []inference.StateRef, labels map[string]string) *ROCmPromptCacheEntry {
	entry := &ROCmPromptCacheEntry{
		Tokens:      append([]int32(nil), tokens...),
		CacheBlocks: cloneCacheBlockRefs(blocks),
		HiddenRefs:  cloneStateRefs(hiddenRefs),
		LogitRefs:   cloneStateRefs(logitRefs),
		Labels:      cloneStringMap(labels),
	}
	for _, block := range entry.CacheBlocks {
		if entry.ModelHash == "" {
			entry.ModelHash = block.ModelHash
		}
		if entry.AdapterHash == "" {
			entry.AdapterHash = block.AdapterHash
		}
		if entry.TokenizerHash == "" {
			entry.TokenizerHash = block.TokenizerHash
		}
	}
	return entry
}

func (entry *ROCmPromptCacheEntry) Clone() *ROCmPromptCacheEntry {
	if entry == nil {
		return nil
	}
	return &ROCmPromptCacheEntry{
		Tokens:        append([]int32(nil), entry.Tokens...),
		CacheBlocks:   cloneCacheBlockRefs(entry.CacheBlocks),
		HiddenRefs:    cloneStateRefs(entry.HiddenRefs),
		LogitRefs:     cloneStateRefs(entry.LogitRefs),
		ModelHash:     entry.ModelHash,
		AdapterHash:   entry.AdapterHash,
		TokenizerHash: entry.TokenizerHash,
		Labels:        cloneStringMap(entry.Labels),
	}
}

// Hidden returns portable hidden-state refs carried by this prompt-cache entry.
func (entry *ROCmPromptCacheEntry) Hidden() []inference.StateRef {
	if entry == nil {
		return nil
	}
	return cloneStateRefs(entry.HiddenRefs)
}

// Logits returns portable last-logit refs carried by this prompt-cache entry.
func (entry *ROCmPromptCacheEntry) Logits() []inference.StateRef {
	if entry == nil {
		return nil
	}
	return cloneStateRefs(entry.LogitRefs)
}

// RestoreCaches rebuilds a standalone ROCm block-cache service from the entry's
// token prefix. Runtime tensor refs stay behind the portable refs; callers that
// need device state should wake the StateSession from those refs.
func (entry *ROCmPromptCacheEntry) RestoreCaches(ctx context.Context, prefixLen, requestFixedSize int) (*BlockCacheService, error) {
	if entry == nil {
		return nil, core.NewError("rocm: prompt cache entry is nil")
	}
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if prefixLen <= 0 || prefixLen > len(entry.Tokens) {
		prefixLen = len(entry.Tokens)
	}
	if prefixLen == 0 {
		return nil, core.NewError("rocm: prompt cache entry has no tokens")
	}
	labels := cloneStringMap(entry.Labels)
	if labels == nil {
		labels = map[string]string{}
	}
	if requestFixedSize > 0 {
		labels["request_fixed_size"] = strconv.Itoa(requestFixedSize)
	}
	mode := firstNonEmptyString(entry.cacheMode(), "block-prefix")
	service := NewBlockCacheService(BlockCacheConfig{
		ModelHash:     entry.ModelHash,
		AdapterHash:   entry.AdapterHash,
		TokenizerHash: entry.TokenizerHash,
		CacheMode:     mode,
		Labels:        labels,
	})
	_, err := service.WarmCache(ctx, inference.CacheWarmRequest{
		Model:   inference.ModelIdentity{Hash: entry.ModelHash},
		Adapter: inference.AdapterIdentity{Hash: entry.AdapterHash},
		Tokens:  append([]int32(nil), entry.Tokens[:prefixLen]...),
		Mode:    mode,
		Labels:  labels,
	})
	if err != nil {
		return nil, err
	}
	return service, nil
}

func (entry *ROCmPromptCacheEntry) cacheMode() string {
	if entry == nil {
		return ""
	}
	for _, block := range entry.CacheBlocks {
		if block.Encoding != "" {
			return block.Encoding
		}
	}
	return entry.Labels["cache_mode"]
}

// ROCmRuntimeAuthorModel is the concrete ROCm runtime-author surface for a
// loaded model. It is the ROCm-owned analogue of go-mlx's runtime_author.go:
// callers can drive safe runtime operations through the loaded model without
// depending on package-private fields or architecture-specific structs.
type ROCmRuntimeAuthorModel interface {
	UnderlyingModel() any
	RuntimeTokenizer() inference.TokenizerModel
	RequireTextRuntime(operation string) error
	AcquireSlot(ctx context.Context) (func(), error)
	AcquirePromptCache() func()
	WithDevice(fn func()) error
	NewCachesWithRequestFixedSize(requestFixedSize int) *BlockCacheService
	GenerationFixedSlidingCacheSize(promptTokens, maxTokens int) int
	RuntimeCacheService() *BlockCacheService
	RuntimeStateSession() *StateSession
	RuntimeCachesSnapshotSafe() bool
	PromptCacheEnabled() bool
	PrefillChunkSize() int
	PromptCacheMinimum() int
	SetLastErr(error)
	SetLastMetrics(inference.GenerateMetrics)
	AdapterCacheKey() string
	PromptCacheMatchWithHidden(tokens []int32) (*ROCmPromptCacheEntry, int)
	StorePromptCacheEntry(entry *ROCmPromptCacheEntry)
	RuntimeCacheProfile(ctx context.Context) (rocmmodel.CacheProfile, error)
	RuntimeModelProfile() ROCmModelProfile
	RuntimeModelRoutePlan() ROCmModelRoutePlan
	RuntimeAuthorPlan() rocmmodel.RuntimeAuthorPlan
}

// RuntimeAuthorPlanForModel returns the reactive runtime-author plan for a
// loaded model. A model-owned implementation wins; otherwise the route-plan
// reporter/registry path is used.
func RuntimeAuthorPlanForModel(model inference.TextModel) (rocmmodel.RuntimeAuthorPlan, bool) {
	if model == nil {
		return rocmmodel.RuntimeAuthorPlan{}, false
	}
	if author, ok := model.(interface {
		RuntimeAuthorPlan() rocmmodel.RuntimeAuthorPlan
	}); ok {
		plan := author.RuntimeAuthorPlan()
		if plan.Matched() {
			return plan.Clone(), true
		}
	}
	plan, ok := ROCmModelRoutePlanForModel(model)
	if !ok || !plan.RuntimeAuthorPlan.Matched() {
		return rocmmodel.RuntimeAuthorPlan{}, false
	}
	return plan.RuntimeAuthorPlan.Clone(), true
}

// UnderlyingModel exposes the runtime-owned native model handle. Runtime
// authors may type-assert it to a ROCm concrete type when they intentionally
// need a HIP-specific path.
func (m *rocmModel) UnderlyingModel() any {
	if m == nil {
		return nil
	}
	m.stateMutex.Lock()
	defer m.stateMutex.Unlock()
	return m.native
}

// RuntimeTokenizer returns the model's token codec and chat-template surface.
func (m *rocmModel) RuntimeTokenizer() inference.TokenizerModel {
	if m == nil {
		return nil
	}
	return m
}

// RequireTextRuntime verifies that a loaded native text runtime is present.
func (m *rocmModel) RequireTextRuntime(operation string) error {
	if strings.TrimSpace(operation) == "" {
		operation = "rocm.RequireTextRuntime"
	}
	if m == nil {
		return core.E(operation, "model is nil", nil)
	}
	m.stateMutex.Lock()
	native := m.native
	m.stateMutex.Unlock()
	if native == nil {
		return core.E(operation, "native model is nil", nil)
	}
	return nil
}

// AcquireSlot reserves a generation slot. Native ROCm currently serializes at
// the model/runtime layer, so the reservation is a no-op after context/runtime
// validation; the method keeps runtime authors on the same contract as go-mlx.
func (m *rocmModel) AcquireSlot(ctx context.Context) (func(), error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if err := m.RequireTextRuntime("rocm.AcquireSlot"); err != nil {
		return nil, err
	}
	return func() {}, nil
}

// AcquirePromptCache returns a scoped prompt-cache release function. ROCm's
// cache service guards its own state internally, so this is a no-op lock.
func (m *rocmModel) AcquirePromptCache() func() {
	return func() {}
}

// WithDevice runs fn against the loaded ROCm runtime. HIP context selection is
// owned below nativeModel today, so this validates the loaded runtime and then
// executes fn.
func (m *rocmModel) WithDevice(fn func()) error {
	if fn == nil {
		return nil
	}
	if err := m.RequireTextRuntime("rocm.WithDevice"); err != nil {
		return err
	}
	fn()
	return nil
}

// NewCachesWithRequestFixedSize creates a request-scoped ROCm cache service.
func (m *rocmModel) NewCachesWithRequestFixedSize(requestFixedSize int) *BlockCacheService {
	if m == nil {
		return NewBlockCacheService(BlockCacheConfig{CacheMode: "block-prefix"})
	}
	identity := m.modelIdentity()
	adapter := m.ActiveAdapter()
	labels := map[string]string{
		"backend":        "rocm",
		"runtime_author": "true",
	}
	if requestFixedSize > 0 {
		labels["request_fixed_size"] = strconv.Itoa(requestFixedSize)
	}
	mode := rocmRuntimeAuthorCacheMode(m.RuntimeModelRoutePlan())
	return NewBlockCacheService(BlockCacheConfig{
		ModelHash:    identity.Hash,
		AdapterHash:  adapter.Hash,
		CacheMode:    mode,
		Labels:       labels,
		deviceDriver: m.blockCacheDeviceDriver(),
	})
}

// GenerationFixedSlidingCacheSize returns the request fixed-cache length. A
// zero result means the model should use its normal grow-as-needed cache path.
func (m *rocmModel) GenerationFixedSlidingCacheSize(promptTokens, maxTokens int) int {
	if promptTokens < 0 {
		promptTokens = 0
	}
	if maxTokens <= 0 {
		return 0
	}
	plan := m.RuntimeModelRoutePlan()
	if !plan.RuntimeAuthorPlan.FixedSlidingCacheSize && !plan.EngineFeatures.FixedSlidingCache {
		return 0
	}
	size := promptTokens + maxTokens
	if contextLength := m.modelIdentity().ContextLength; contextLength > 0 && size > contextLength {
		return contextLength
	}
	return size
}

// RuntimeCacheService returns the model-owned block cache service.
func (m *rocmModel) RuntimeCacheService() *BlockCacheService {
	if m == nil {
		return nil
	}
	return m.blockCacheService()
}

// RuntimeStateSession returns the model-owned state session.
func (m *rocmModel) RuntimeStateSession() *StateSession {
	if m == nil {
		return nil
	}
	return m.stateSession()
}

// RuntimeCachesSnapshotSafe reports whether the current model route can expose
// portable cache/state snapshots without handing out private runtime handles.
func (m *rocmModel) RuntimeCachesSnapshotSafe() bool {
	plan := m.RuntimeModelRoutePlan()
	return plan.CacheRoute.SupportsKV ||
		plan.StateContextRoute.SleepState ||
		plan.StateContextRoute.WakeState ||
		plan.StateContextRoute.PackageLocalKV ||
		plan.StateContextRoute.BlockBundleRefs ||
		plan.StateContextRoute.PortableRefs
}

// PromptCacheEnabled reports whether the model can construct the ROCm prompt
// cache service.
func (m *rocmModel) PromptCacheEnabled() bool {
	return m != nil
}

// PrefillChunkSize returns the configured prompt prefill chunk size. The native
// ROCm wrapper does not expose a separate chunk knob yet.
func (m *rocmModel) PrefillChunkSize() int {
	return 0
}

// PromptCacheMinimum returns the minimum prompt length for cache population.
// ROCm's block cache accepts any non-empty prompt/tokens today.
func (m *rocmModel) PromptCacheMinimum() int {
	return 1
}

// SetLastErr records the most recent runtime-author failure.
func (m *rocmModel) SetLastErr(err error) {
	m.setLastFailure(err)
}

// SetLastMetrics records the most recent runtime-author metrics.
func (m *rocmModel) SetLastMetrics(metrics inference.GenerateMetrics) {
	m.setLastMetrics(metrics)
}

// AdapterCacheKey returns the active adapter's stable cache-key fragment.
func (m *rocmModel) AdapterCacheKey() string {
	adapter := m.ActiveAdapter()
	return firstNonEmptyString(adapter.Hash, adapter.Path)
}

// PromptCacheMatchWithHidden finds the longest matching prompt-cache entry. The
// ROCm entry carries portable cache/state refs; hidden/logit refs are present
// only when a runtime author stored them.
func (m *rocmModel) PromptCacheMatchWithHidden(tokens []int32) (*ROCmPromptCacheEntry, int) {
	if m == nil || len(tokens) == 0 {
		return nil, 0
	}
	adapterKey := m.AdapterCacheKey()
	m.stateMutex.Lock()
	stored := m.promptCache.Clone()
	m.stateMutex.Unlock()
	if prefixLen := stored.matchPrefix(tokens, adapterKey); prefixLen > 0 {
		return stored, prefixLen
	}
	service := m.RuntimeCacheService()
	if service == nil {
		return nil, 0
	}
	entry, prefixLen := rocmPromptCacheEntryFromServicePrefix(service, tokens)
	if entry == nil {
		return nil, 0
	}
	if entry.AdapterHash == "" {
		entry.AdapterHash = adapterKey
	}
	return entry, prefixLen
}

// StorePromptCacheEntry installs a metadata prompt-cache entry for runtime
// authors. The active adapter key is stamped so adapter swaps cannot match it.
func (m *rocmModel) StorePromptCacheEntry(entry *ROCmPromptCacheEntry) {
	if m == nil {
		return
	}
	cloned := entry.Clone()
	if cloned != nil {
		cloned.AdapterHash = m.AdapterCacheKey()
	}
	m.stateMutex.Lock()
	m.promptCache = cloned
	m.stateMutex.Unlock()
}

// RuntimeCacheProfile returns the live cache profile for runtime authors.
func (m *rocmModel) RuntimeCacheProfile(ctx context.Context) (rocmmodel.CacheProfile, error) {
	if m == nil {
		return rocmmodel.CacheProfile{}, nil
	}
	return m.CacheProfile(ctx)
}

// RuntimeModelProfile returns the loaded model's resolved reactive profile.
func (m *rocmModel) RuntimeModelProfile() ROCmModelProfile {
	if m == nil {
		return ROCmModelProfile{}
	}
	return m.ModelProfile()
}

// RuntimeModelRoutePlan returns the loaded model's resolved reactive route plan.
func (m *rocmModel) RuntimeModelRoutePlan() ROCmModelRoutePlan {
	if m == nil {
		return ROCmModelRoutePlan{}
	}
	return m.ModelRoutePlan()
}

// RuntimeAuthorPlan returns the runtime-author plan carried by the model route
// plan.
func (m *rocmModel) RuntimeAuthorPlan() rocmmodel.RuntimeAuthorPlan {
	plan := m.RuntimeModelRoutePlan()
	if !plan.RuntimeAuthorPlan.Matched() {
		return rocmmodel.RuntimeAuthorPlan{}
	}
	return plan.RuntimeAuthorPlan.Clone()
}

func (entry *ROCmPromptCacheEntry) matchPrefix(tokens []int32, adapterKey string) int {
	if entry == nil || len(entry.Tokens) == 0 || len(entry.Tokens) > len(tokens) {
		return 0
	}
	if entry.AdapterHash != "" && adapterKey != "" && entry.AdapterHash != adapterKey {
		return 0
	}
	for index, token := range entry.Tokens {
		if tokens[index] != token {
			return 0
		}
	}
	return len(entry.Tokens)
}

func rocmPromptCacheEntryFromBlock(block cacheBlock) *ROCmPromptCacheEntry {
	entry := NewROCmPromptCacheEntry(block.tokens, []inference.CacheBlockRef{block.ref}, nil, nil, block.labels)
	if entry != nil {
		entry.ModelHash = firstNonEmptyString(entry.ModelHash, block.ref.ModelHash)
		entry.AdapterHash = firstNonEmptyString(entry.AdapterHash, block.ref.AdapterHash)
		entry.TokenizerHash = firstNonEmptyString(entry.TokenizerHash, block.ref.TokenizerHash)
	}
	return entry
}

func rocmPromptCacheEntryFromServicePrefix(service *BlockCacheService, tokens []int32) (*ROCmPromptCacheEntry, int) {
	if service == nil || len(tokens) == 0 {
		return nil, 0
	}
	service.mu.Lock()
	defer service.mu.Unlock()
	var best cacheBlock
	var bestLen int
	for _, block := range service.blocks {
		if block.ref.Encoding != service.cacheMode || len(block.tokens) == 0 || len(block.tokens) > len(tokens) {
			continue
		}
		if block.ref.ModelHash != service.modelHash || block.ref.AdapterHash != service.adapterHash || block.ref.TokenizerHash != service.tokenizerHash {
			continue
		}
		matches := true
		for index, token := range block.tokens {
			if tokens[index] != token {
				matches = false
				break
			}
		}
		if matches && len(block.tokens) > bestLen {
			best = block
			bestLen = len(block.tokens)
		}
	}
	if bestLen == 0 {
		return nil, 0
	}
	return rocmPromptCacheEntryFromBlock(best), bestLen
}

func rocmRuntimeAuthorCacheMode(plan ROCmModelRoutePlan) string {
	for _, mode := range []string{plan.CacheRoute.DeviceMode, plan.CacheRoute.RecommendedMode, plan.CacheRoute.DefaultMode} {
		if mode == "block-prefix" || isROCmKVCacheMode(mode) {
			return mode
		}
	}
	return "block-prefix"
}

func cloneCacheBlockRefs(refs []inference.CacheBlockRef) []inference.CacheBlockRef {
	if len(refs) == 0 {
		return nil
	}
	out := make([]inference.CacheBlockRef, 0, len(refs))
	for _, ref := range refs {
		out = append(out, cloneCacheBlockRef(ref))
	}
	return out
}
