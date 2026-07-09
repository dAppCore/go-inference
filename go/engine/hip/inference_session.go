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

var _ engine.Session = (*hipEngineSession)(nil)

// hipEngineSession is the retained Gemma4-Q4 decode session behind engine.
// Session. It is single-goroutine-guarded by mu (engine.SessionHandle already
// serialises calls; the lock guards the device/pending/tokens invariant).
type hipEngineSession struct {
	mu        sync.Mutex
	loaded    *hipLoadedModel
	cfg       hipGemma4Q4ForwardConfig
	engine    hipGemma4Q4EngineConfig
	mode      string
	driver    nativeHIPDriver
	device    *hipGemma4Q4DeviceDecodeState
	pending   []int32
	tokens    []int32
	generated []int32
	closed    bool
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
	if err := s.closeDeviceLocked(); err != nil {
		return err
	}
	s.pending = append([]int32(nil), ids...)
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
		return nil
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
	generate := inference.GenerateConfig{MaxTokens: maxNew}
	return s.generate(generate, eosID, nil, yield)
}

// GenerateSampledFromCacheEach decodes with the sampler params. hip owns its
// own device/host sampler (driven by the GenerateConfig fields), so params map
// onto the GenerateConfig; the shared *model.Sampler exposes no seed accessor
// and hip's RNG is internal, so the sampler argument is not threaded through.
// transform remaps each selected id before it is yielded.
func (s *hipEngineSession) GenerateSampledFromCacheEach(maxNew int, stopTokens []int32, sampler *model.Sampler, params model.SampleParams, transform model.TokenTransform, yield func(int32) bool) ([]int32, error) {
	_ = sampler
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
	return s.generate(generate, -1, transform, yield)
}

// generate is the shared decode body: forward the buffered prompt through hip's
// combined driver and stream tokens. It requires buffered tokens — hip cannot
// decode from a bare cache (see the file header).
func (s *hipEngineSession) generate(generate inference.GenerateConfig, eosID int, transform model.TokenTransform, yield func(int32) bool) ([]int32, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return nil, core.NewError("hip.EngineSession.Generate: session is closed")
	}
	if len(s.pending) == 0 {
		return nil, core.NewError("hip.EngineSession.Generate: no buffered tokens to seed decode (hip decodes from a forwarded prompt, not a bare cache — append a prompt first)")
	}
	prompt := s.pending
	s.pending = nil
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
	out, err := s.driveLocked(context.Background(), prompt, generate, emit)
	s.tokens = append(s.tokens, out...)
	s.generated = append(s.generated, out...)
	return out, err
}

// driveLocked runs hip's combined prefill+decode driver over promptTokens,
// continuing from the retained device state. Ownership of s.device moves into
// the driver (which transfers/closes it) and the retain callback re-installs the
// final state. Must be called with mu held.
func (s *hipEngineSession) driveLocked(ctx context.Context, promptTokens []int32, generate inference.GenerateConfig, emit func(int32) bool) ([]int32, error) {
	initial := s.device
	s.device = nil
	var out []int32
	stopped := false
	seq, errFn := hipGemma4Q4GenerateTokenSeqWithState(ctx, s.loaded, s.cfg, promptTokens, generate, s.engine, initial, func(state *hipGemma4Q4DeviceDecodeState) error {
		s.device = state
		return nil
	})
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
	host, err := s.hostStateLocked()
	if err != nil {
		return nil, err
	}
	return hipDecodeStateToSnapshot(host, s.cfg, s.tokens, s.generated, opts)
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
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return core.NewError("hip.EngineSession.RangeKVBlocks: session is closed")
	}
	host, err := s.hostStateLocked()
	if err != nil {
		return err
	}
	total := host.tokenCountForConfig(s.cfg)
	if total <= 0 {
		snapshot, err := hipDecodeStateToSnapshot(host, s.cfg, s.tokens, s.generated, opts)
		if err != nil {
			return err
		}
		_, yieldErr := yield(kv.Block{Index: 0, TokenStart: 0, TokenCount: len(s.tokens), Snapshot: snapshot})
		return yieldErr
	}
	index := 0
	for start := 0; start < total; start += blockSize {
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
		cont, yieldErr := yield(kv.Block{Index: index, TokenStart: start, TokenCount: count, Snapshot: snapshot})
		if yieldErr != nil {
			return yieldErr
		}
		if !cont {
			return nil
		}
		index++
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
		device, err = hipMirrorGemma4Q4DecodeState(s.driver, s.cfg, host, s.mode)
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

// hipSliceDecodeStateTokens returns a host decode state holding only the
// [start, start+count) token window of each layer (float32 rows, HeadDim wide).
func hipSliceDecodeStateTokens(host hipGemma4Q4DecodeState, cfg hipGemma4Q4ForwardConfig, start, count int) hipGemma4Q4DecodeState {
	sliced := hipGemma4Q4DecodeState{Layers: make([]hipGemma4Q4LayerKVState, len(host.Layers))}
	for index, layer := range host.Layers {
		headDim := 0
		if index < len(cfg.Layers) {
			headDim = cfg.Layers[index].HeadDim
		}
		if headDim <= 0 {
			continue
		}
		from := start * headDim
		to := (start + count) * headDim
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
