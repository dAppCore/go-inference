// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

// inference_session.go adapts hip's retained Gemma4-Q4 decode to the shared
// engine.Session contract (the 9 primitives engine.SessionHandle drives). It is
// engine/hip's analogue of engine/metal's *ArchSession — except hip's native
// driver is shaped differently, and that difference drives the design here.
//
// # hip's driver is one-shot; engine.Session is incremental
//
// hipGemma4Q4GenerateTokenSeqWithState is a COMBINED prefill+decode call:
// prefill processes the prompt tokens and produces the first token's logits
// (its `current`, which it errors without), then the decode loop continues from
// there. There is no "decode from a bare retained cache" entry point — hip
// always needs a token to forward to seed the next step. engine.Session, by
// contrast, splits PrefillTokens (store prompt KV) from GenerateFromCacheEach
// (decode from the cache).
//
// The bridge: this session BUFFERS the unforwarded tokens in `pending` and runs
// the combined driver at generate time. The invariant is:
//
//	device  == retained KV for tokens[: len(tokens)-len(pending)]
//	pending == the suffix tokens whose KV is not yet on the device
//
// PrefillTokens/AppendTokens only buffer (no device work). GenerateFromCacheEach
// forwards `pending` (seeding decode from it) and decodes maxNew. CaptureKV
// serialises whatever KV the device holds PLUS the full token list (Snapshot.
// Tokens); the forwarded count (Snapshot.SeqLen) tells RestoreFromKV where
// `pending` resumes — so a capture taken before any forward is a valid,
// KV-empty checkpoint that restores to a replayable prompt, and a capture taken
// after decode carries the real device KV. No prefill-only driver call is
// needed (and none is possible: hip defaults MaxTokens<=0 to the full remaining
// context, so "generate zero" is not available).
//
// HONESTY NOTE: only the pure host<->kv.Snapshot converter (inference_kv_
// snapshot.go) is proven hardware-free. Everything in THIS file is
// HIP-hardware-behavioural — the driver semantics (seeding, KV append order,
// device retention, HostState/mirror round-trip) are validated only by the
// HIP-gated parity + conformance tests in inference_conformance_test.go, which
// run on Snider's linux+AMD box. This file lands COMPILE-VERIFIED, not
// behaviourally proven.
package hip

import (
	"context"
	"sync"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/inference/engine"
	"dappco.re/go/inference/kv"
	"dappco.re/go/inference/model"
)

var (
	_ engine.Session       = (*hipEngineSession)(nil)
	_ engine.VisionSession = (*hipEngineSession)(nil)
)

// hipKVSnapshotDevicePayloadMode marks the opaque per-layer payloads that
// retain HIP's device-page encoding alongside the portable float32 snapshot.
// kv's compressed-payload wire lane preserves these bytes verbatim.
const hipKVSnapshotDevicePayloadMode = "turboquant"

// hipEngineSession is the retained Gemma4-Q4 decode session behind engine.
// Session. It is single-goroutine-guarded by mu (engine.SessionHandle already
// serialises calls; the lock guards the device/pending/tokens invariant).
type hipEngineSession struct {
	mu      sync.Mutex
	loaded  *hipLoadedModel
	cfg     hipGemma4Q4ForwardConfig
	engine  hipGemma4Q4EngineConfig
	mode    string
	driver  nativeHIPDriver
	device  *hipGemma4Q4DeviceDecodeState
	pending []int32
	// pendingEmbeddings contains one float32 hidden row per pending token when
	// the prompt entered through the multimodal prefill contract. Nil means the
	// ordinary token-embedding lookup path.
	pendingEmbeddings []byte
	tokens            []int32
	generated         []int32
	closed            bool
}

// newHipEngineSession opens a retained Gemma4-Q4 session over a loaded model.
// It requires a Gemma4-Q4-linked model — the retained-KV fast path is q4
// specific; other architectures serve through prompt replay (rocmModel.Generate)
// and have no runtime-owned KV to capture.
func newHipEngineSession(loaded *hipLoadedModel) (*hipEngineSession, error) {
	if loaded == nil {
		return nil, core.NewError("hip.EngineSession: loaded model is nil")
	}
	if !hipLoadedGemma4Q4GenerateLinked(loaded) {
		return nil, core.NewError("hip.EngineSession: model is not a Gemma4-Q4 linked runtime (no runtime-owned KV to retain)")
	}
	if loaded.modelInfo.NumLayers <= 0 {
		return nil, core.NewError("hip.EngineSession: loaded model layer count is required")
	}
	cfg, err := loaded.cachedGemma4Q4ForwardConfig(loaded.modelInfo.NumLayers)
	if err != nil {
		return nil, err
	}
	engineConfig := loaded.gemma4Q4EngineConfig()
	mode, err := engineConfig.deviceKVMode()
	if err != nil {
		return nil, err
	}
	return &hipEngineSession{
		loaded: loaded,
		cfg:    cfg,
		engine: engineConfig,
		mode:   mode,
		driver: loaded.driver,
	}, nil
}

// PrefillTokens replaces any retained state with a fresh buffered prompt. The
// prompt's KV is materialised lazily at the next generate/capture (hip forwards
// it then), so this only records the tokens.
func (s *hipEngineSession) PrefillTokens(ids []int32) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return core.NewError("hip.EngineSession.PrefillTokens: session is closed")
	}
	if len(ids) == 0 {
		return core.NewError("hip.EngineSession.PrefillTokens: empty prompt tokens")
	}
	if len(ids) > s.contextLimitLocked() {
		return core.NewError("hip.EngineSession.PrefillTokens: prompt exceeds model context window")
	}
	if err := s.closeDeviceLocked(); err != nil {
		return err
	}
	s.pending = append([]int32(nil), ids...)
	s.pendingEmbeddings = nil
	s.tokens = append([]int32(nil), ids...)
	s.generated = nil
	return nil
}

// PrefillTokenEmbeddings replaces retained state with a multimodal prompt whose
// already-spliced float32 rows must enter layer zero directly.
func (s *hipEngineSession) PrefillTokenEmbeddings(ids []int32, embeddings [][]byte) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return core.NewError("hip.EngineSession.PrefillTokenEmbeddings: session is closed")
	}
	if len(ids) == 0 {
		return core.NewError("hip.EngineSession.PrefillTokenEmbeddings: empty prompt tokens")
	}
	if len(ids) != len(embeddings) {
		return core.NewError("hip.EngineSession.PrefillTokenEmbeddings: token and embedding counts differ")
	}
	if len(ids) > s.contextLimitLocked() {
		return core.NewError("hip.EngineSession.PrefillTokenEmbeddings: prompt exceeds model context window")
	}
	rowBytes := s.embeddingRowBytesLocked()
	if rowBytes <= 0 {
		return core.NewError("hip.EngineSession.PrefillTokenEmbeddings: invalid embedding width")
	}
	stream := make([]byte, len(ids)*rowBytes)
	for index, row := range embeddings {
		if len(row) != rowBytes {
			return core.NewError("hip.EngineSession.PrefillTokenEmbeddings: embedding row width mismatch")
		}
		copy(stream[index*rowBytes:(index+1)*rowBytes], row)
	}
	if err := s.closeDeviceLocked(); err != nil {
		return err
	}
	s.pending = append([]int32(nil), ids...)
	s.pendingEmbeddings = stream
	s.tokens = append([]int32(nil), ids...)
	s.generated = nil
	return nil
}

// AppendTokens extends the buffered suffix without replaying the prefix — the
// next generate forwards these onto the retained device KV.
func (s *hipEngineSession) AppendTokens(ids []int32) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return core.NewError("hip.EngineSession.AppendTokens: session is closed")
	}
	if len(ids) == 0 {
		return core.NewError("hip.EngineSession.AppendTokens: empty prompt tokens")
	}
	if len(s.pendingEmbeddings) > 0 {
		return core.NewError("hip.EngineSession.AppendTokens: cannot append token ids before custom embeddings are forwarded")
	}
	if limit := s.contextLimitLocked(); len(s.tokens) > limit-len(ids) {
		return core.NewError("hip.EngineSession.AppendTokens: sequence exceeds model context window")
	}
	s.pending = append(s.pending, ids...)
	s.tokens = append(s.tokens, ids...)
	return nil
}

// Pos is the number of tokens in the session (forwarded + buffered).
func (s *hipEngineSession) Pos() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return len(s.tokens)
}

// GenerateFromCacheEach greedily decodes up to maxNew tokens, forwarding the
// buffered prompt to seed decode. eosID < 0 lets the caller own the stop
// decision (yield returns false to stop).
func (s *hipEngineSession) GenerateFromCacheEach(maxNew, eosID int, yield func(int32) bool) ([]int32, error) {
	return s.GenerateFromCacheEachContext(context.Background(), maxNew, eosID, yield)
}

// GenerateFromCacheEachContext is GenerateFromCacheEach with the caller's
// cancellation context passed all the way to the HIP decode driver.
func (s *hipEngineSession) GenerateFromCacheEachContext(ctx context.Context, maxNew, eosID int, yield func(int32) bool) ([]int32, error) {
	generate := inference.GenerateConfig{MaxTokens: maxNew}
	return s.generate(ctx, generate, eosID, nil, nil, yield)
}

// GenerateSampledFromCacheEach decodes with the request-owned shared sampler.
// transform remaps each selected id before it is yielded.
func (s *hipEngineSession) GenerateSampledFromCacheEach(maxNew int, stopTokens []int32, sampler *model.Sampler, params model.SampleParams, transform model.TokenTransform, yield func(int32) bool) ([]int32, error) {
	return s.GenerateSampledFromCacheEachContext(context.Background(), maxNew, stopTokens, sampler, params, transform, yield)
}

// GenerateSampledFromCacheEachContext is GenerateSampledFromCacheEach with
// the caller's cancellation context passed all the way to HIP decode.
func (s *hipEngineSession) GenerateSampledFromCacheEachContext(ctx context.Context, maxNew int, stopTokens []int32, sampler *model.Sampler, params model.SampleParams, transform model.TokenTransform, yield func(int32) bool) ([]int32, error) {
	if sampler == nil {
		return nil, core.NewError("hip.EngineSession.GenerateSampledFromCache: nil sampler")
	}
	generate := inference.GenerateConfig{
		MaxTokens:           maxNew,
		StopTokens:          append([]int32(nil), stopTokens...),
		Temperature:         params.Temperature,
		TopK:                params.TopK,
		TopP:                params.TopP,
		MinP:                params.MinP,
		RepeatPenalty:       params.RepeatPenalty,
		SuppressTokens:      append([]int32(nil), params.SuppressTokens...),
		MinTokensBeforeStop: params.MinTokensBeforeStop,
	}
	return s.generate(ctx, generate, -1, sampler, transform, yield)
}

// generate is the shared decode body: forward the buffered prompt through hip's
// combined driver and stream tokens. It requires buffered tokens — hip cannot
// decode from a bare cache (see the file header).
func (s *hipEngineSession) generate(ctx context.Context, generate inference.GenerateConfig, eosID int, sampler *model.Sampler, transform model.TokenTransform, yield func(int32) bool) ([]int32, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return nil, core.NewError("hip.EngineSession.Generate: session is closed")
	}
	if len(s.pending) == 0 {
		return nil, core.NewError("hip.EngineSession.Generate: no buffered tokens to seed decode (hip decodes from a forwarded prompt, not a bare cache — append a prompt first)")
	}
	prompt := s.pending
	promptEmbeddings := s.pendingEmbeddings
	s.pending = nil
	s.pendingEmbeddings = nil
	emit := func(id int32) bool {
		out := id
		if transform != nil {
			out = transform(id)
		}
		keep := true
		if yield != nil {
			keep = yield(out)
		}
		if eosID >= 0 && id == int32(eosID) {
			return false
		}
		return keep
	}
	out, err := s.driveLocked(ctx, prompt, promptEmbeddings, generate, sampler, emit)
	s.tokens = append(s.tokens, out...)
	s.generated = append(s.generated, out...)
	if err == nil {
		err = s.syncPendingWithDeviceLocked()
	}
	return out, err
}

// syncPendingWithDeviceLocked restores the session invariant after HIP's
// combined driver: its final emitted token has logits but is not yet a KV row.
func (s *hipEngineSession) syncPendingWithDeviceLocked() error {
	forwarded := 0
	if s.device != nil {
		forwarded = s.device.maxLayerTokenCount()
	}
	if forwarded > len(s.tokens) {
		return core.NewError("hip.EngineSession.Generate: retained device KV exceeds session tokens")
	}
	s.pending = append([]int32(nil), s.tokens[forwarded:]...)
	return nil
}

// PrefillTokensCached replaces the retained prompt while preserving a shared
// complete prefix. HIP cannot safely rewind a divergent retained run, so every
// divergence takes the cold path. Exact hits retain all but their final token
// as KV so that final token can seed HIP's combined prefill/decode entry.
func (s *hipEngineSession) PrefillTokensCached(ids []int32) (int, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return 0, core.NewError("hip.EngineSession.PrefillTokensCached: session is closed")
	}
	if len(ids) == 0 {
		return 0, core.NewError("hip.EngineSession.PrefillTokensCached: empty prompt tokens")
	}
	if limit := s.contextLimitLocked(); len(ids) > limit {
		return 0, core.NewError("hip.EngineSession.PrefillTokensCached: prompt exceeds model context window")
	}
	if len(s.pendingEmbeddings) > 0 {
		return 0, s.prefillTokensLocked(ids)
	}

	lcp := hipTokenPrefixLen(s.tokens, ids)
	if lcp == len(ids) {
		// Keep a seed token pending even for an exact hit: HIP has no decode
		// entry point that can start from a completely bare retained cache.
		lcp--
	}
	forwarded := len(s.tokens) - len(s.pending)
	if forwarded < 0 {
		forwarded = 0
	}
	if lcp <= 0 {
		return 0, s.prefillTokensLocked(ids)
	}
	if lcp < forwarded {
		if s.ringRollbackUnsafeLocked(forwarded) {
			return 0, s.prefillTokensLocked(ids)
		}
		if err := s.truncateRetainedPrefixLocked(lcp); err != nil {
			return 0, err
		}
		forwarded = lcp
	}
	s.tokens = append([]int32(nil), ids...)
	s.pending = append([]int32(nil), ids[forwarded:]...)
	s.generated = nil
	return forwarded, nil
}

func (s *hipEngineSession) contextLimitLocked() int {
	if s.loaded != nil && s.loaded.contextSize > 0 {
		return s.loaded.contextSize
	}
	return defaultContextLengthCap
}

func (s *hipEngineSession) embeddingRowBytesLocked() int {
	if s.loaded != nil && s.loaded.modelInfo.HiddenSize > 0 {
		return s.loaded.modelInfo.HiddenSize * 4
	}
	if len(s.cfg.Layers) > 0 && s.cfg.Layers[0].HiddenSize > 0 {
		return s.cfg.Layers[0].HiddenSize * 4
	}
	return 0
}

func (s *hipEngineSession) ringRollbackUnsafeLocked(forwarded int) bool {
	limit := s.contextLimitLocked()
	for _, layer := range s.cfg.Layers {
		window := layer.SlidingWindow
		if window > 0 && window < limit && forwarded > window {
			return true
		}
	}
	return false
}

func (s *hipEngineSession) prefillTokensLocked(ids []int32) error {
	if err := s.closeDeviceLocked(); err != nil {
		return err
	}
	s.pending = append([]int32(nil), ids...)
	s.pendingEmbeddings = nil
	s.tokens = append([]int32(nil), ids...)
	s.generated = nil
	return nil
}

func (s *hipEngineSession) truncateRetainedPrefixLocked(tokens int) error {
	forwarded := len(s.tokens) - len(s.pending)
	if tokens >= forwarded {
		return nil
	}
	if tokens <= 0 {
		return s.closeDeviceLocked()
	}
	host, err := s.hostStateLocked()
	if err != nil {
		return err
	}
	next, err := hipMirrorGemma4Q4DecodeState(s.driver, s.cfg, hipSliceDecodeStateTokens(host, s.cfg, 0, tokens), s.mode)
	if err != nil {
		return err
	}
	if err := s.closeDeviceLocked(); err != nil {
		_ = next.Close()
		return err
	}
	s.device = next
	return nil
}

func hipTokenPrefixLen(a, b []int32) int {
	limit := len(a)
	if len(b) < limit {
		limit = len(b)
	}
	for index := 0; index < limit; index++ {
		if a[index] != b[index] {
			return index
		}
	}
	return limit
}

// driveLocked runs hip's combined prefill+decode driver over promptTokens,
// continuing from the retained device state. Ownership of s.device moves into
// the driver (which transfers/closes it) and the retain callback re-installs the
// final state. Must be called with mu held.
func (s *hipEngineSession) driveLocked(ctx context.Context, promptTokens []int32, promptEmbeddings []byte, generate inference.GenerateConfig, sampler *model.Sampler, emit func(int32) bool) ([]int32, error) {
	initial := s.device
	s.device = nil
	var out []int32
	stopped := false
	seq, errFn := hipGemma4Q4GenerateTokenSeqWithStateSamplerEmbeddings(ctx, s.loaded, s.cfg, promptTokens, promptEmbeddings, generate, s.engine, initial, func(state *hipGemma4Q4DeviceDecodeState) error {
		s.device = state
		return nil
	}, sampler)
	seq(func(token inference.Token) bool {
		if stopped {
			return false
		}
		out = append(out, token.ID)
		if emit != nil && !emit(token.ID) {
			stopped = true
			return false
		}
		return true
	})
	if err := errFn(); err != nil {
		return out, err
	}
	return out, nil
}

// CaptureKVWithOptions copies the retained device KV to a portable kv.Snapshot
// via the host<->snapshot converter. When no KV has been forwarded yet the
// snapshot carries zero-token layers plus the buffered prompt in Snapshot.Tokens
// (a replayable checkpoint).
func (s *hipEngineSession) CaptureKVWithOptions(opts kv.CaptureOptions) (*kv.Snapshot, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return nil, core.NewError("hip.EngineSession.CaptureKVWithOptions: session is closed")
	}
	if len(s.pendingEmbeddings) > 0 {
		return nil, core.NewError("hip.EngineSession.CaptureKVWithOptions: custom embeddings must be forwarded before state capture")
	}
	host, err := s.hostStateLocked()
	if err != nil {
		return nil, err
	}
	snapshot, err := hipDecodeStateToSnapshot(host, s.cfg, s.tokens, s.generated, opts)
	if err != nil {
		return nil, err
	}
	if err := hipAttachDeviceKVPayloads(snapshot, s.device); err != nil {
		return nil, err
	}
	return snapshot, nil
}

// hostStateLocked reads the retained device KV to host float32, or an all-empty
// host state (one empty layer per config layer) when nothing is forwarded yet.
func (s *hipEngineSession) hostStateLocked() (hipGemma4Q4DecodeState, error) {
	if s.device == nil {
		return hipGemma4Q4DecodeState{Layers: make([]hipGemma4Q4LayerKVState, len(s.cfg.Layers))}, nil
	}
	return s.device.HostState()
}

// RangeKVBlocks streams the retained KV state as contiguous token blocks of
// blockSize. Each block carries a sub-snapshot sliced to its token window; a
// KV-empty session yields one token-only block so callers still see the
// sequence.
func (s *hipEngineSession) RangeKVBlocks(blockSize int, opts kv.CaptureOptions, yield func(kv.Block) (bool, error)) error {
	if yield == nil {
		return core.NewError("hip.EngineSession.RangeKVBlocks: nil yield")
	}
	if blockSize <= 0 {
		return core.NewError("hip.EngineSession.RangeKVBlocks: blockSize must be positive")
	}
	if opts.BlockStartToken < 0 {
		return core.NewError("hip.EngineSession.RangeKVBlocks: block start token must be non-negative")
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return core.NewError("hip.EngineSession.RangeKVBlocks: session is closed")
	}
	if len(s.pendingEmbeddings) > 0 {
		return core.NewError("hip.EngineSession.RangeKVBlocks: custom embeddings must be forwarded before state capture")
	}
	host, err := s.hostStateLocked()
	if err != nil {
		return err
	}
	total := host.tokenCountForConfig(s.cfg)
	if total <= 0 {
		total = len(s.tokens)
	}
	firstIndex := opts.BlockStartToken / blockSize
	for index, start := firstIndex, firstIndex*blockSize; start < total; index, start = index+1, start+blockSize {
		count := blockSize
		if start+count > total {
			count = total - start
		}
		blockHost := hipSliceDecodeStateTokens(host, s.cfg, start, count)
		blockTokens := hipTokenWindow(s.tokens, start, count)
		snapshot, err := hipDecodeStateToSnapshot(blockHost, s.cfg, blockTokens, nil, opts)
		if err != nil {
			return err
		}
		snapshot.TokenOffset = start + count
		cont, yieldErr := yield(kv.Block{Index: index, TokenStart: start, TokenCount: count, Snapshot: snapshot})
		if yieldErr != nil {
			return yieldErr
		}
		if !cont {
			return nil
		}
	}
	return nil
}

// RestoreFromKV rebuilds the retained device KV from a snapshot and resumes any
// tokens beyond the forwarded KV as the buffered prompt. A KV-empty snapshot
// restores to a replayable prompt (device nil, all tokens buffered).
func (s *hipEngineSession) RestoreFromKV(ctx context.Context, snapshot *kv.Snapshot) error {
	if ctx == nil {
		ctx = context.Background()
	}
	if snapshot == nil {
		return core.NewError("hip.EngineSession.RestoreFromKV: nil snapshot")
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return core.NewError("hip.EngineSession.RestoreFromKV: session is closed")
	}
	host, err := hipSnapshotToDecodeState(snapshot, s.cfg)
	if err != nil {
		return err
	}
	forwarded := host.tokenCountForConfig(s.cfg)
	var device *hipGemma4Q4DeviceDecodeState
	if forwarded > 0 {
		device, err = hipRestoreGemma4Q4DeviceDecodeState(snapshot, s.driver, s.cfg, s.engine, s.mode, host)
		if err != nil {
			return err
		}
	}
	if err := s.closeDeviceLocked(); err != nil {
		if device != nil {
			_ = device.Close()
		}
		return err
	}
	s.device = device
	s.tokens = append([]int32(nil), snapshot.Tokens...)
	s.pendingEmbeddings = nil
	if forwarded < len(s.tokens) {
		s.pending = append([]int32(nil), s.tokens[forwarded:]...)
	} else {
		s.pending = nil
	}
	s.generated = nil
	return ctx.Err()
}

// Close releases the retained device KV state.
func (s *hipEngineSession) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return nil
	}
	s.closed = true
	s.pending = nil
	s.pendingEmbeddings = nil
	return s.closeDeviceLocked()
}

func (s *hipEngineSession) closeDeviceLocked() error {
	if s.device == nil {
		return nil
	}
	device := s.device
	s.device = nil
	return device.Close()
}

// hipAttachDeviceKVPayloads records the exact encoded page images that back a
// retained device state. The float32 tensors remain the portable snapshot
// view; these payloads avoid a lossy decode/re-quantize cycle during restore.
func hipAttachDeviceKVPayloads(snapshot *kv.Snapshot, device *hipGemma4Q4DeviceDecodeState) error {
	if snapshot == nil || device == nil {
		return nil
	}
	if len(snapshot.Layers) != len(device.layers) {
		return core.E("rocm.hip.KVSnapshot.Capture", "device state layer count must match snapshot", nil)
	}
	for index, layer := range device.layers {
		if layer.cache == nil {
			return core.E("rocm.hip.KVSnapshot.Capture", "device layer KV cache is nil", nil)
		}
		host, err := layer.cache.hostCache()
		if err != nil {
			return core.E("rocm.hip.KVSnapshot.Capture", core.Sprintf("copy raw device KV layer %d", index), err)
		}
		payloads := make([][]byte, 0, len(host.blocks))
		for _, block := range host.blocks {
			payload, err := host.rawBlock(block)
			if err != nil {
				return core.E("rocm.hip.KVSnapshot.Capture", core.Sprintf("encode raw device KV layer %d", index), err)
			}
			payloads = append(payloads, payload)
		}
		if len(payloads) == 0 {
			return core.E("rocm.hip.KVSnapshot.Capture", "device layer KV cache has no pages", nil)
		}
		snapshot.Layers[index].CacheIndex = layer.cache.blockSize
		snapshot.Layers[index].CacheMode = hipKVSnapshotDevicePayloadMode
		snapshot.Layers[index].TurboQuantPayloads = payloads
	}
	return nil
}

// hipRestoreGemma4Q4DeviceDecodeState restores exact device pages when the
// capture carries them, falling back to the portable host mirror for older
// snapshots that predate the opaque device payloads.
func hipRestoreGemma4Q4DeviceDecodeState(snapshot *kv.Snapshot, driver nativeHIPDriver, cfg hipGemma4Q4ForwardConfig, engineConfig hipGemma4Q4EngineConfig, mode string, host hipGemma4Q4DecodeState) (*hipGemma4Q4DeviceDecodeState, error) {
	if !hipSnapshotHasDeviceKVPayloads(snapshot) {
		return hipMirrorGemma4Q4DecodeState(driver, cfg, host, mode)
	}
	if driver == nil || !driver.Available() {
		return nil, core.E("rocm.hip.KVSnapshot.Restore", "HIP driver is not available", nil)
	}
	state := hipNewGemma4Q4DeviceDecodeState(mode, len(snapshot.Layers))
	state.remirrorLayers = len(snapshot.Layers)
	success := false
	defer func() {
		if !success {
			_ = state.Close()
		}
	}()
	for index, layerSnapshot := range snapshot.Layers {
		layer, err := hipRestoreGemma4Q4DeviceLayer(snapshot, layerSnapshot, driver, cfg.Layers[index], engineConfig, mode, index)
		if err != nil {
			return nil, err
		}
		state.layers = append(state.layers, layer)
	}
	restoredHost, err := state.HostState()
	if err != nil {
		return nil, err
	}
	if !hipDecodeStatesEqual(restoredHost, host) {
		return nil, core.E("rocm.hip.KVSnapshot.Restore", "raw device KV payload does not match portable snapshot", nil)
	}
	success = true
	return state, nil
}

func hipSnapshotHasDeviceKVPayloads(snapshot *kv.Snapshot) bool {
	if snapshot == nil || len(snapshot.Layers) == 0 {
		return false
	}
	for _, layer := range snapshot.Layers {
		if layer.CacheMode != hipKVSnapshotDevicePayloadMode || len(layer.TurboQuantPayloads) == 0 {
			return false
		}
	}
	return true
}

func hipRestoreGemma4Q4DeviceLayer(snapshot *kv.Snapshot, layerSnapshot kv.LayerSnapshot, driver nativeHIPDriver, cfg hipGemma4Q4Layer0Config, engineConfig hipGemma4Q4EngineConfig, mode string, index int) (hipGemma4Q4DeviceLayerKVState, error) {
	kvWidth := cfg.keyValueDim()
	if index >= len(snapshot.Layers) || kvWidth <= 0 {
		return hipGemma4Q4DeviceLayerKVState{}, core.E("rocm.hip.KVSnapshot.Restore", "raw device KV layer configuration is invalid", nil)
	}
	pages := make([]rocmDeviceKVPage, 0, len(layerSnapshot.TurboQuantPayloads))
	pagesTransferred := false
	defer func() {
		if pagesTransferred {
			return
		}
		for _, page := range pages {
			_ = rocmDeviceKVTensorFreePair(driver, page.key, page.value)
		}
	}()
	nextStart := 0
	for _, payload := range layerSnapshot.TurboQuantPayloads {
		page, err := rocmDeviceKVPageFromRawPayload(driver, payload)
		if err != nil {
			return hipGemma4Q4DeviceLayerKVState{}, core.E("rocm.hip.KVSnapshot.Restore", core.Sprintf("restore raw device KV layer %d", index), err)
		}
		if page.tokenStart != nextStart || page.tokenCount <= 0 || page.keyWidth != kvWidth || page.valueWidth != kvWidth {
			_ = rocmDeviceKVTensorFreePair(driver, page.key, page.value)
			return hipGemma4Q4DeviceLayerKVState{}, core.E("rocm.hip.KVSnapshot.Restore", "raw device KV page geometry is invalid", nil)
		}
		nextStart += page.tokenCount
		pages = append(pages, page)
	}
	if nextStart <= 0 {
		return hipGemma4Q4DeviceLayerKVState{}, core.E("rocm.hip.KVSnapshot.Restore", "raw device KV layer has no pages", nil)
	}
	blockSize := layerSnapshot.CacheIndex
	if blockSize <= 0 {
		blockSize = engineConfig.deviceKVBlockSizeForSlidingWindow(cfg.SlidingWindow)
	}
	cache := rocmBorrowDeviceKVCache(driver, mode, blockSize, nextStart, pages, false)
	pagesTransferred = true
	table, err := cache.kernelDescriptorTableLabeled("rocm.hip.KVSnapshot.Restore", "restore_exact_device_kv")
	if err != nil {
		_ = cache.Close()
		return hipGemma4Q4DeviceLayerKVState{}, err
	}
	launch, err := cache.KernelLaunchDescriptor(table)
	if err != nil {
		_ = table.Close()
		_ = cache.Close()
		return hipGemma4Q4DeviceLayerKVState{}, err
	}
	return hipGemma4Q4DeviceLayerKVState{cache: cache, descriptorTable: table, launch: launch}, nil
}

func hipDecodeStatesEqual(left, right hipGemma4Q4DecodeState) bool {
	if len(left.Layers) != len(right.Layers) {
		return false
	}
	for index := range left.Layers {
		if !hipFloat32SlicesEqual(left.Layers[index].Keys, right.Layers[index].Keys) ||
			!hipFloat32SlicesEqual(left.Layers[index].Values, right.Layers[index].Values) {
			return false
		}
	}
	return true
}

// hipSliceDecodeStateTokens returns a host decode state holding only the
// [start, start+count) token window of each layer (float32 KV rows).
func hipSliceDecodeStateTokens(host hipGemma4Q4DecodeState, cfg hipGemma4Q4ForwardConfig, start, count int) hipGemma4Q4DecodeState {
	sliced := hipGemma4Q4DecodeState{Layers: make([]hipGemma4Q4LayerKVState, len(host.Layers))}
	for index, layer := range host.Layers {
		rowWidth := 0
		if index < len(cfg.Layers) {
			rowWidth = cfg.Layers[index].keyValueDim()
		}
		if rowWidth <= 0 {
			continue
		}
		from := start * rowWidth
		to := (start + count) * rowWidth
		if from < 0 {
			from = 0
		}
		if to > len(layer.Keys) {
			to = len(layer.Keys)
		}
		if from >= to {
			continue
		}
		sliced.Layers[index] = hipGemma4Q4LayerKVState{
			Keys:   append([]float32(nil), layer.Keys[from:to]...),
			Values: append([]float32(nil), layer.Values[from:to]...),
		}
	}
	return sliced
}

// hipTokenWindow returns a copy of tokens[start:start+count], clamped to bounds.
func hipTokenWindow(tokens []int32, start, count int) []int32 {
	if start < 0 || start >= len(tokens) {
		return nil
	}
	end := start + count
	if end > len(tokens) {
		end = len(tokens)
	}
	return append([]int32(nil), tokens[start:end]...)
}
