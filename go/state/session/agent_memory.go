// SPDX-Licence-Identifier: EUPL-1.2

package session

import (
	"context"
	"maps"
	"strconv"

	core "dappco.re/go"
	"dappco.re/go/inference"
	statebundle "dappco.re/go/inference/bundle"
	"dappco.re/go/inference/kv"
	"dappco.re/go/inference/spine"
	state "dappco.re/go/inference/state"
	"dappco.re/go/inference/state/agent"
)

// agent_memory.go: the session-side agent-memory lifecycle — wake from a
// durable indexed KV prefix, sleep the retained state back to blocks, and
// the go-inference state.Session contract (WakeState/SleepState). The
// Model-side entries (Wake/ForkFromBundle/ForkState/FoldAgentMemory) stay
// in the root mlx package, which the go-inference Forker contract pins.

const foldedAgentMemoryPrefillWakeMaxTokens = 16 * 1024

// Hoisted sentinel errors — each returned multiple times from the
// agent-memory lifecycle entry points; package vars avoid per-call
// allocation in the validation hot path.
var (
	errAgentMemorySessionNil       = core.NewError("mlx: model session is nil")
	errAgentMemoryStoreNil         = core.NewError("mlx: state store is nil")
	errAgentMemoryFoldPlanNil      = core.NewError("mlx: folded State wake plan is nil")
	errAgentMemoryFoldNoTokens     = core.NewError("mlx: folded State prefill wake loaded no tokens")
	errAgentMemoryWakeNeedsStore   = core.NewError("mlx: inference agent memory wake requires state.Store")
	errAgentMemorySleepNeedsStore  = core.NewError("mlx: inference State sleep requires state.Writer")
	errAgentMemoryReuseNeedsReader = core.NewError("mlx: State parent-prefix reuse requires a readable state store")
)

// cloneStringMap returns a defensive copy of values, or nil if empty.
func cloneStringMap(values map[string]string) map[string]string {
	if len(values) == 0 {
		return nil
	}
	return core.MapClone(values)
}

// WakeAgentMemory restores this session from a durable indexed KV prefix.
func (s *Session) WakeAgentMemory(ctx context.Context, store state.Store, opts agent.WakeOptions) (*agent.WakeReport, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if s == nil || s.session == nil {
		return nil, errAgentMemorySessionNil
	}
	plan, err := agent.PlanWake(ctx, store, opts, spine.ModelInfoToMemory(s.info))
	if err != nil {
		return nil, err
	}
	// Cache the prefix length — consumed by kv.StateBlockSource and
	// LoadPrefixFromStateBlocksWithOptions on the two non-folded paths, and
	// re-read inside shouldPrefillFoldedAgentMemory's bounds check.
	prefixTokens := plan.Entry.PrefixTokens()
	if shouldPrefillFoldedAgentMemory(plan.Entry) {
		if err := s.prefillFoldedAgentMemory(ctx, store, plan, opts); err != nil {
			return nil, err
		}
		plan.Report.RestoreStrategy = "folded-prefill"
		s.agentMemory = agent.CloneWakeReport(plan.Report)
		return plan.Report, nil
	}
	if restorer, ok := s.session.(nativeSessionKVBlockRestorer); ok {
		source, err := kv.StateBlockSource(ctx, store, plan.Bundle, prefixTokens)
		if err != nil {
			return nil, err
		}
		if err := restorer.RestoreKVBlocks(ctx, source); err != nil {
			return nil, err
		}
		plan.Report.RestoreStrategy = "kv-blocks"
		s.agentMemory = agent.CloneWakeReport(plan.Report)
		return plan.Report, nil
	}
	snapshot, err := kv.LoadPrefixFromStateBlocksWithOptions(ctx, store, plan.Bundle, prefixTokens, opts.LoadOptions)
	if err != nil {
		return nil, err
	}
	if err := s.RestoreKV(snapshot); err != nil {
		return nil, err
	}
	plan.Report.RestoreStrategy = "snapshot"
	s.agentMemory = agent.CloneWakeReport(plan.Report)
	return plan.Report, nil
}

// Wake is a lifecycle alias for WakeAgentMemory.
func (s *Session) Wake(ctx context.Context, store state.Store, opts agent.WakeOptions) (*agent.WakeReport, error) {
	return s.WakeAgentMemory(ctx, store, opts)
}

func shouldPrefillFoldedAgentMemory(entry agent.StateIndexEntry) bool {
	prefix := entry.PrefixTokens()
	if prefix <= 0 || prefix > foldedAgentMemoryPrefillWakeMaxTokens {
		return false
	}
	if meta := entry.Meta["folded_state"]; meta != "" {
		// Canonical-form fast path. foldedAgentMemorySleepOptions writes
		// "true" verbatim — the round-trip producer / consumer pairing
		// hits the byte-equal branch and skips Lower + Trim work.
		if meta == "true" || core.Lower(core.Trim(meta)) == "true" {
			return true
		}
	}
	for _, label := range entry.Labels {
		if label == "" {
			continue
		}
		// Canonical-form fast path. foldedAgentMemorySleepOptions appends
		// "folded-state" verbatim — same round-trip pairing argument.
		if label == "folded-state" || core.Lower(core.Trim(label)) == "folded-state" {
			return true
		}
	}
	return false
}

func (s *Session) prefillFoldedAgentMemory(ctx context.Context, store state.Store, plan *agent.WakePlan, opts agent.WakeOptions) error {
	if s == nil || s.session == nil {
		return errAgentMemorySessionNil
	}
	if plan == nil || plan.Bundle == nil {
		return errAgentMemoryFoldPlanNil
	}
	loadOpts := opts.LoadOptions
	if plan.Bundle.KVEncoding == kv.EncodingNative {
		loadOpts.RawKVOnly = true
	}
	tokens, err := kv.LoadPrefixTokensFromStateBlocksWithOptions(ctx, store, plan.Bundle, plan.Entry.PrefixTokens(), loadOpts)
	if err != nil {
		return core.E("mlx: folded State prefill wake", "load tokens", err)
	}
	if len(tokens) == 0 {
		return errAgentMemoryFoldNoTokens
	}
	if err := s.PrefillTokens(ctx, tokens); err != nil {
		return core.E("mlx: folded State prefill wake", "prefill", err)
	}
	return nil
}

// WakeState implements the backend-neutral go-inference agent-memory contract.
func (s *Session) WakeState(ctx context.Context, req inference.AgentMemoryWakeRequest) (*inference.AgentMemoryWakeResult, error) {
	store, ok := req.Store.(state.Store)
	if !ok {
		return nil, errAgentMemoryWakeNeedsStore
	}
	report, err := s.WakeAgentMemory(ctx, store, WakeOptionsFromInference(req))
	if err != nil {
		return nil, err
	}
	return ToInferenceWakeResult(report), nil
}

// SleepAgentMemory streams this session's current KV state to State blocks,
// then writes a bundle manifest and one-entry wake index.
func (s *Session) SleepAgentMemory(ctx context.Context, store state.Writer, opts agent.SleepOptions) (*agent.SleepReport, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if s == nil || s.session == nil {
		return nil, errAgentMemorySessionNil
	}
	if store == nil {
		return nil, errAgentMemoryStoreNil
	}
	entryURI, bundleURI, indexURI, err := agent.SleepURIs(opts)
	if err != nil {
		return nil, err
	}
	if opts.ModelInfo.Architecture == "" {
		opts.ModelInfo = spine.ModelInfoToMemory(s.info)
	}
	// Hoist the s.agentMemory nil check — was repeated three times in
	// independent branch predicates. Single load + reused alias lets the
	// three assignments share one pointer dereference each.
	if parent := s.agentMemory; parent != nil {
		if opts.ParentEntryURI == "" {
			opts.ParentEntryURI = parent.EntryURI
		}
		if opts.ParentBundleURI == "" {
			opts.ParentBundleURI = parent.BundleURI
		}
		if opts.ParentIndexURI == "" {
			opts.ParentIndexURI = parent.IndexURI
		}
	}
	blockOpts := agent.SleepBlockOptions(opts, bundleURI)
	if opts.ReuseParentPrefix && blockOpts.ReusePrefix == nil {
		readStore, ok := store.(state.Store)
		if !ok {
			return nil, errAgentMemoryReuseNeedsReader
		}
		parentBundle, err := kv.LoadStateBlockBundle(ctx, readStore, opts.ParentBundleURI)
		if err != nil {
			return nil, err
		}
		blockOpts.ReusePrefix = parentBundle
		if blockOpts.ReusePrefixTokens <= 0 {
			blockOpts.ReusePrefixTokens = parentBundle.TokenCount
		}
	}
	bundle, err := s.SaveKVBlocksToState(ctx, store, blockOpts)
	if err != nil {
		return nil, err
	}
	bundleRef, err := kv.SaveStateBlockBundle(ctx, store, bundle, bundleURI)
	if err != nil {
		return nil, err
	}
	index, err := agent.NewSleepIndex(bundle, opts, entryURI, bundleURI)
	if err != nil {
		return nil, err
	}
	indexRef, err := agent.SaveStateIndex(ctx, store, index, indexURI)
	if err != nil {
		return nil, err
	}
	report := agent.NewSleepReport(index, bundle, opts, entryURI, bundleURI, indexURI, bundleRef, indexRef)
	s.agentMemory = agent.WakeReportFromSleep(report)
	return report, nil
}

// Sleep is a lifecycle alias for SleepAgentMemory.
func (s *Session) Sleep(ctx context.Context, store state.Writer, opts agent.SleepOptions) (*agent.SleepReport, error) {
	return s.SleepAgentMemory(ctx, store, opts)
}

// SleepState implements the backend-neutral go-inference agent-memory contract.
func (s *Session) SleepState(ctx context.Context, req inference.AgentMemorySleepRequest) (*inference.AgentMemorySleepResult, error) {
	store, ok := req.Store.(state.Writer)
	if !ok {
		return nil, errAgentMemorySleepNeedsStore
	}
	report, err := s.SleepAgentMemory(ctx, store, agentMemorySleepOptionsFromInference(req))
	if err != nil {
		return nil, err
	}
	return toInferenceAgentMemorySleepResult(report), nil
}

// AppendAndSleepAgentMemory appends new prompt material and then streams the
// resulting state to durable storage without forcing a generation/reply step.
func (s *Session) AppendAndSleepAgentMemory(ctx context.Context, prompt string, store state.Writer, opts agent.SleepOptions) (*agent.SleepReport, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if err := s.AppendPrompt(prompt); err != nil {
		return nil, err
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return s.SleepAgentMemory(ctx, store, opts)
}

// AppendAndSleep is a lifecycle alias for AppendAndSleepAgentMemory.
func (s *Session) AppendAndSleep(ctx context.Context, prompt string, store state.Writer, opts agent.SleepOptions) (*agent.SleepReport, error) {
	return s.AppendAndSleepAgentMemory(ctx, prompt, store, opts)
}

// GenerateAndSleepAgentMemory generates an answer from the current retained
// state and streams the post-answer KV state to durable storage.
func (s *Session) GenerateAndSleepAgentMemory(ctx context.Context, store state.Writer, opts agent.SleepOptions, generateOpts ...inference.GenerateOption) (string, *agent.SleepReport, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return "", nil, err
	}
	if s == nil || s.session == nil {
		return "", nil, errAgentMemorySessionNil
	}
	builder := core.NewBuilder()
	// Generations typically produce hundreds of tokens of text. Pre-grow
	// the backing slice to skip the early 64 -> 128 -> 256 -> 512 -> 1024
	// reallocations during token streaming.
	builder.Grow(1024)
	cfg := inference.ApplyGenerateOpts(generateOpts)
	for tok := range s.session.Generate(ctx, cfg) {
		builder.WriteString(tok.Text)
	}
	if err := s.session.Err(); err != nil {
		return builder.String(), nil, err
	}
	if err := ctx.Err(); err != nil {
		return builder.String(), nil, err
	}
	report, err := s.SleepAgentMemory(ctx, store, opts)
	if err != nil {
		return builder.String(), nil, err
	}
	return builder.String(), report, nil
}

// GenerateAndSleep is a lifecycle alias for GenerateAndSleepAgentMemory.
func (s *Session) GenerateAndSleep(ctx context.Context, store state.Writer, opts agent.SleepOptions, generateOpts ...inference.GenerateOption) (string, *agent.SleepReport, error) {
	return s.GenerateAndSleepAgentMemory(ctx, store, opts, generateOpts...)
}

// WakeOptionsFromInference maps the go-inference wake request onto agent
// wake options. Exported for the root Model.ForkState entry, which shares
// the same request shape.
func WakeOptionsFromInference(req inference.AgentMemoryWakeRequest) agent.WakeOptions {
	return agent.WakeOptions{
		IndexURI:               req.IndexURI,
		EntryURI:               req.EntryURI,
		Tokenizer:              stateBundleTokenizerFromInference(req.Tokenizer),
		SkipCompatibilityCheck: req.SkipCompatibilityCheck,
	}
}

func agentMemorySleepOptionsFromInference(req inference.AgentMemorySleepRequest) agent.SleepOptions {
	return agent.SleepOptions{
		EntryURI:          req.EntryURI,
		BundleURI:         req.BundleURI,
		IndexURI:          req.IndexURI,
		ParentEntryURI:    req.ParentEntryURI,
		ParentBundleURI:   req.ParentBundleURI,
		ParentIndexURI:    req.ParentIndexURI,
		Title:             req.Title,
		Model:             req.Model.ID,
		ModelPath:         req.Model.Path,
		ModelInfo:         spine.ModelInfoToMemory(modelInfoFromInferenceIdentity(req.Model)),
		Tokenizer:         stateBundleTokenizerFromInference(req.Tokenizer),
		ReuseParentPrefix: req.ReuseParentPrefix,
		BlockOptions: kv.StateBlockOptions{
			BlockSize:  req.BlockSize,
			KVEncoding: kv.Encoding(req.Encoding),
		},
		Labels: agentMemoryLabelsFromInference(req.Labels),
		Meta:   agentMemoryMetadataFromInference(req),
	}
}

func stateBundleTokenizerFromInference(tokenizer inference.TokenizerIdentity) statebundle.Tokenizer {
	return statebundle.NormaliseTokenizer(statebundle.Tokenizer{
		Kind:         tokenizer.Kind,
		Path:         tokenizer.Path,
		Hash:         tokenizer.Hash,
		BOS:          tokenizer.BOSID,
		EOS:          tokenizer.EOSID,
		ChatTemplate: tokenizer.ChatTemplate,
	})
}

func modelInfoFromInferenceIdentity(model inference.ModelIdentity) spine.ModelInfo {
	return spine.ModelInfo{
		Architecture:  model.Architecture,
		VocabSize:     model.VocabSize,
		NumLayers:     model.NumLayers,
		HiddenSize:    model.HiddenSize,
		QuantBits:     model.QuantBits,
		QuantGroup:    model.QuantGroup,
		ContextLength: model.ContextLength,
	}
}

// ToInferenceWakeResult maps a wake report onto the go-inference result
// shape. Exported for the root Model.ForkState entry.
func ToInferenceWakeResult(report *agent.WakeReport) *inference.AgentMemoryWakeResult {
	if report == nil {
		return nil
	}
	return &inference.AgentMemoryWakeResult{
		Entry: inference.AgentMemoryRef{
			URI:        report.EntryURI,
			BundleURI:  report.BundleURI,
			IndexURI:   report.IndexURI,
			Title:      report.Title,
			Hash:       report.SnapshotHash,
			TokenStart: 0,
			TokenCount: report.PrefixTokens,
		},
		Bundle:       agentMemoryStateRef(report.BundleURI, kv.StateBlockBundleKind, report.SnapshotHash, ""),
		Index:        agentMemoryStateRef(report.IndexURI, agent.StateIndexKind, report.IndexHash, ""),
		PrefixTokens: report.PrefixTokens,
		BundleTokens: report.BundleTokens,
		BlockSize:    report.BlockSize,
		BlocksRead:   report.BlocksRead,
	}
}

func toInferenceAgentMemorySleepResult(report *agent.SleepReport) *inference.AgentMemorySleepResult {
	if report == nil {
		return nil
	}
	// Hoist the KVEncoding string conversion — same value is consumed by
	// both the Bundle ref and the top-level Encoding field.
	encoding := string(report.KVEncoding)
	return &inference.AgentMemorySleepResult{
		Entry: inference.AgentMemoryRef{
			URI:        report.EntryURI,
			BundleURI:  report.BundleURI,
			IndexURI:   report.IndexURI,
			Title:      report.Title,
			Hash:       report.SnapshotHash,
			TokenStart: 0,
			TokenCount: report.TokenCount,
		},
		Parent: inference.AgentMemoryRef{
			URI:       report.ParentEntryURI,
			BundleURI: report.ParentBundleURI,
			IndexURI:  report.ParentIndexURI,
		},
		Bundle:        agentMemoryStateRef(report.BundleURI, kv.StateBlockBundleKind, report.SnapshotHash, encoding),
		Index:         agentMemoryStateRef(report.IndexURI, agent.StateIndexKind, report.IndexHash, ""),
		TokenCount:    report.TokenCount,
		BlockSize:     report.BlockSize,
		BlocksWritten: report.BlocksWritten,
		BlocksReused:  report.BlocksReused,
		Encoding:      encoding,
	}
}

func agentMemoryStateRef(uri, kind, hash, encoding string) inference.StateRef {
	return inference.StateRef{
		Kind:     kind,
		URI:      uri,
		Hash:     hash,
		Encoding: encoding,
	}
}

func agentMemoryLabelsFromInference(labels map[string]string) []string {
	if len(labels) == 0 {
		return nil
	}
	out := make([]string, 0, len(labels))
	// Tiny-N fast path: a single label avoids the size-pass + Builder
	// scaffolding (which only pays off when we have >=2 non-empty values
	// to share a backing buffer). Direct `key + "=" + value` allocates
	// once for the result string — same shape as the previous code,
	// without the per-iteration count overhead.
	if len(labels) == 1 {
		for key, value := range labels {
			if value == "" {
				out = append(out, key)
			} else {
				out = append(out, key+"="+value)
			}
		}
		return out
	}
	// Multi-entry path: build all "key=value" strings into a single
	// backing buffer, then slice that buffer into the []string output.
	// Saves one allocation per non-empty value vs the previous shape
	// (which alloced a fresh string per concat). Two-pass: size first
	// so the Builder buffer lands at the exact right capacity and the
	// growth ladder (8 -> 16 -> 32 ...) never kicks in.
	size := 0
	for key, value := range labels {
		if value == "" {
			continue
		}
		size += len(key) + 1 + len(value)
	}
	if size == 0 {
		// All-empty fast path — every entry aliases the map key.
		for key := range labels {
			out = append(out, key)
		}
		core.SliceSort(out)
		return out
	}
	var builder core.Builder
	builder.Grow(size)
	for key, value := range labels {
		if value == "" {
			out = append(out, key)
			continue
		}
		start := builder.Len()
		builder.WriteString(key)
		builder.WriteByte('=')
		builder.WriteString(value)
		// builder.String() returns the underlying buffer via unsafe —
		// every Grow-bounded write leaves earlier slices pinned to the
		// same backing memory, so it is safe to take a sub-slice here.
		out = append(out, builder.String()[start:])
	}
	core.SliceSort(out)
	return out
}

func agentMemoryMetadataFromInference(req inference.AgentMemorySleepRequest) map[string]string {
	// Pre-size the destination map. The 9 optional adapter/runtime fields
	// dominate the entry count — counting empties first lets us hand
	// runtime.makemap_small the exact capacity, replacing the addAgent
	// loop's incremental zero-cap growth.
	extras := 0
	if req.Adapter.Hash != "" {
		extras++
	}
	if req.Adapter.Path != "" {
		extras++
	}
	if req.Adapter.Format != "" {
		extras++
	}
	if req.Adapter.Rank != 0 {
		extras++
	}
	if req.Adapter.Alpha != 0 {
		extras++
	}
	if req.Runtime.Backend != "" {
		extras++
	}
	if req.Runtime.Device != "" {
		extras++
	}
	if req.Runtime.CacheMode != "" {
		extras++
	}
	if req.Runtime.Version != "" {
		extras++
	}
	if extras == 0 {
		// Nothing to fold in — defer to the existing clone, which
		// returns nil if req.Metadata is also empty (the common
		// idle-keepalive request shape).
		return cloneStringMap(req.Metadata)
	}
	// Fast path: no user-supplied metadata. Every adapter/runtime key is
	// fresh, so the addAgentMemoryMetadata 'meta[key] == ""' idempotence
	// read is wasted work — direct writes shave one map-probe per non-
	// empty field. Whitespace-only values still need to be filtered
	// (preserving addAgentMemoryMetadata's Trim safety check) — fields
	// like Adapter.Path can legitimately arrive as '   ' from upstream.
	if req.Metadata == nil {
		meta := make(map[string]string, extras)
		if v := req.Adapter.Hash; v != "" && core.Trim(v) != "" {
			meta["adapter_hash"] = v
		}
		if v := req.Adapter.Path; v != "" && core.Trim(v) != "" {
			meta["adapter_path"] = v
		}
		if v := req.Adapter.Format; v != "" && core.Trim(v) != "" {
			meta["adapter_format"] = v
		}
		if req.Adapter.Rank != 0 {
			meta["adapter_rank"] = strconv.Itoa(req.Adapter.Rank)
		}
		if req.Adapter.Alpha != 0 {
			meta["adapter_alpha"] = strconv.FormatFloat(float64(req.Adapter.Alpha), 'g', -1, 32)
		}
		if v := req.Runtime.Backend; v != "" && core.Trim(v) != "" {
			meta["runtime_backend"] = v
		}
		if v := req.Runtime.Device; v != "" && core.Trim(v) != "" {
			meta["runtime_device"] = v
		}
		if v := req.Runtime.CacheMode; v != "" && core.Trim(v) != "" {
			meta["runtime_cache_mode"] = v
		}
		if v := req.Runtime.Version; v != "" && core.Trim(v) != "" {
			meta["runtime_version"] = v
		}
		return meta
	}
	dst := make(map[string]string, len(req.Metadata)+extras)
	maps.Copy(dst, req.Metadata)
	// addAgentMemoryMetadata-equivalent inline writes — same idempotence
	// rule (don't overwrite caller-supplied keys) but skip the function
	// call. The Trim guard runs only for non-empty values (the counting
	// loop above already filtered v=="" out of extras, so the && short-
	// circuit makes Trim a one-time check per field).
	if v := req.Adapter.Hash; v != "" && dst["adapter_hash"] == "" && core.Trim(v) != "" {
		dst["adapter_hash"] = v
	}
	if v := req.Adapter.Path; v != "" && dst["adapter_path"] == "" && core.Trim(v) != "" {
		dst["adapter_path"] = v
	}
	if v := req.Adapter.Format; v != "" && dst["adapter_format"] == "" && core.Trim(v) != "" {
		dst["adapter_format"] = v
	}
	if req.Adapter.Rank != 0 && dst["adapter_rank"] == "" {
		dst["adapter_rank"] = strconv.Itoa(req.Adapter.Rank)
	}
	if req.Adapter.Alpha != 0 && dst["adapter_alpha"] == "" {
		dst["adapter_alpha"] = strconv.FormatFloat(float64(req.Adapter.Alpha), 'g', -1, 32)
	}
	if v := req.Runtime.Backend; v != "" && dst["runtime_backend"] == "" && core.Trim(v) != "" {
		dst["runtime_backend"] = v
	}
	if v := req.Runtime.Device; v != "" && dst["runtime_device"] == "" && core.Trim(v) != "" {
		dst["runtime_device"] = v
	}
	if v := req.Runtime.CacheMode; v != "" && dst["runtime_cache_mode"] == "" && core.Trim(v) != "" {
		dst["runtime_cache_mode"] = v
	}
	if v := req.Runtime.Version; v != "" && dst["runtime_version"] == "" && core.Trim(v) != "" {
		dst["runtime_version"] = v
	}
	return dst
}

func addAgentMemoryMetadata(meta map[string]string, key, value string) map[string]string {
	// Fast path: empty input is the dominant case for optional adapter
	// + runtime fields. Skip the core.Trim allocation entirely.
	if value == "" {
		return meta
	}
	if core.Trim(value) == "" {
		return meta
	}
	if meta == nil {
		meta = map[string]string{}
	}
	if meta[key] == "" {
		meta[key] = value
	}
	return meta
}
