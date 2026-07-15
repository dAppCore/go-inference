// SPDX-Licence-Identifier: EUPL-1.2

// Branch-coverage tests for the session package. These exercise the
// fallback / error-propagation / nil-context arms the happy-path tests in
// session_test.go and agent_memory_test.go leave untouched: the capability
// probes that fall back when a native handle lacks an optional interface,
// the per-method `ctx == nil` defaulting, the error returns from the
// underlying inference/state layer, and the agent-memory folded-prefill +
// reuse-parent-prefix lifecycle arms. White-box (package session) so the
// unexported helpers (sessionParserControlToken, shouldPrefillFoldedAgentMemory,
// prefillFoldedAgentMemory, toInference* mappers) are reachable directly.

package session

import (
	"context"
	"iter"
	"testing"
	"time"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/inference/kv"
	mlxbundle "dappco.re/go/inference/model/bundle"
	"dappco.re/go/inference/model/spine"
	state "dappco.re/go/inference/model/state"
	"dappco.re/go/inference/model/state/agent"
	"dappco.re/go/inference/model/state/session/internal/sessionfake"
)

// ---------------------------------------------------------------------------
// Narrow handles — each deliberately omits one or more optional capability
// interfaces so the session machinery's fallback arm (rather than the
// all-capable sessionfake.Handle path) is exercised.
// ---------------------------------------------------------------------------

// baseHandle implements only the mandatory inference.SessionHandle interface —
// none of the optional chunk/token prefill/append, KV-block, or
// CaptureKVWithOptions capabilities, and not even RestoreKV. It is the floor
// every other narrow handle in this file embeds.
type baseHandle struct {
	prefillPrompt   string
	appendPrompt    string
	captureSnapshot *kv.Snapshot
	captureErr      error
	forkResult      inference.SessionHandle
	forkErr         error
	errValue        error
}

func (h *baseHandle) Prefill(_ context.Context, p string) error      { h.prefillPrompt = p; return nil }
func (h *baseHandle) AppendPrompt(_ context.Context, p string) error { h.appendPrompt = p; return nil }
func (h *baseHandle) Generate(context.Context, inference.GenerateConfig) iter.Seq[inference.Token] {
	return func(func(inference.Token) bool) {}
}
func (h *baseHandle) CaptureKV(context.Context) (*kv.Snapshot, error) {
	return h.captureSnapshot, h.captureErr
}
func (h *baseHandle) RangeKVBlocks(context.Context, int, kv.CaptureOptions, func(kv.Block) (bool, error)) error {
	return nil
}
func (h *baseHandle) Fork(context.Context) (inference.SessionHandle, error) {
	return h.forkResult, h.forkErr
}
func (h *baseHandle) Reset()       {}
func (h *baseHandle) Close() error { return nil }
func (h *baseHandle) Err() error   { return h.errValue }

// optionSnapshotHandle implements CaptureKVWithOptions
// (nativeSessionKVSnapshotterWithOptions), the arm the all-capable
// sessionfake.Handle cannot reach.
type optionSnapshotHandle struct {
	baseHandle
	optionSnapshot *kv.Snapshot
	optionErr      error
	sawOptions     kv.CaptureOptions
}

func (h *optionSnapshotHandle) CaptureKVWithOptions(_ context.Context, opts kv.CaptureOptions) (*kv.Snapshot, error) {
	h.sawOptions = opts
	return h.optionSnapshot, h.optionErr
}

// snapshotRestoreErrHandle is a narrow handle implementing RestoreKV (so it is
// a nativeSessionRestorer) but NOT RestoreKVBlocks — the snapshot-fallback
// wake arm reaches RestoreKV, which here always fails.
type snapshotRestoreErrHandle struct {
	snapshotRestoreHandle
	err error
}

func (h *snapshotRestoreErrHandle) RestoreKV(_ context.Context, snapshot *kv.Snapshot) error {
	h.restored = snapshot
	return h.err
}

// rangeErrHandle is an all-base handle whose RangeKVBlocks always errors,
// driving the SaveKVBlocksToState (and thus SleepAgentMemory) failure arm.
type rangeErrHandle struct {
	baseHandle
	err error
}

func (h *rangeErrHandle) RangeKVBlocks(_ context.Context, _ int, _ kv.CaptureOptions, yield func(kv.Block) (bool, error)) error {
	return h.err
}

// cancelOnAppendHandle cancels a captured context the instant AppendPrompt is
// called, so the caller's post-append ctx.Err() guard trips. AppendPrompt
// itself returns nil — the cancellation, not an append error, is the trigger.
type cancelOnAppendHandle struct {
	baseHandle
	cancel context.CancelFunc
}

func (h *cancelOnAppendHandle) AppendPrompt(_ context.Context, p string) error {
	h.appendPrompt = p
	h.cancel()
	return nil
}

// ---------------------------------------------------------------------------
// session.go — capability-fallback + ctx==nil + error arms
// ---------------------------------------------------------------------------

// TestSession_PrefillChunks_FallbackAndNilCtx_Good drives the two uncovered
// PrefillChunks arms: a handle without PrefillChunks falls back to Prefill of
// the joined string, and a nil context is defaulted to Background.
func TestSession_PrefillChunks_FallbackAndNilCtx_Good(t *testing.T) {
	native := &baseHandle{}
	session := &Session{session: native}

	//nolint:staticcheck // SA1012: nil ctx is the branch under test.
	if err := session.PrefillChunks(nil, seqStrings("stable ", "context")); err != nil {
		t.Fatalf("PrefillChunks() error = %v", err)
	}
	if native.prefillPrompt != "stable context" {
		t.Fatalf("fallback prefill prompt = %q, want joined chunks", native.prefillPrompt)
	}
}

// TestSession_PrefillTokens_NoNativeSupport_Bad — a handle without
// PrefillTokens returns the sentinel, and nil ctx is defaulted first.
func TestSession_PrefillTokens_NoNativeSupport_Bad(t *testing.T) {
	session := &Session{session: &baseHandle{}}

	//nolint:staticcheck // SA1012: nil ctx is part of the branch under test.
	if err := session.PrefillTokens(nil, []int32{1, 2}); !core.Is(err, errNativeNoTokenPrefill) {
		t.Fatalf("PrefillTokens(no support) error = %v, want %v", err, errNativeNoTokenPrefill)
	}
}

// TestSession_AppendPromptChunks_FallbackAndNilCtx_Good — a handle without
// AppendPromptChunks falls back to AppendPrompt of the joined string.
func TestSession_AppendPromptChunks_FallbackAndNilCtx_Good(t *testing.T) {
	native := &baseHandle{}
	session := &Session{session: native}

	//nolint:staticcheck // SA1012: nil ctx is the branch under test.
	if err := session.AppendPromptChunks(nil, seqStrings("\n\nQ: ", "who?")); err != nil {
		t.Fatalf("AppendPromptChunks() error = %v", err)
	}
	if native.appendPrompt != "\n\nQ: who?" {
		t.Fatalf("fallback append prompt = %q, want joined chunks", native.appendPrompt)
	}
}

// TestSession_AppendTokens_NoNativeSupport_Bad — a handle without AppendTokens
// returns the sentinel; nil ctx is defaulted first.
func TestSession_AppendTokens_NoNativeSupport_Bad(t *testing.T) {
	session := &Session{session: &baseHandle{}}

	//nolint:staticcheck // SA1012: nil ctx is part of the branch under test.
	if err := session.AppendTokens(nil, []int32{1}); !core.Is(err, errNativeNoTokenAppend) {
		t.Fatalf("AppendTokens(no support) error = %v, want %v", err, errNativeNoTokenAppend)
	}
}

// TestSession_GenerateStream_NilCtxAndFlush_Good drives GenerateStream's
// ctx==nil defaulting and the final filter.Flush() emission. The Gemma4
// thinking close tag forces the parser to buffer reasoning, then flush
// trailing visible text after the loop — covering the post-loop flush arm.
func TestSession_GenerateStream_NilCtxAndFlush_Good(t *testing.T) {
	session := &Session{session: &sessionfake.Handle{
		Tokens: []inference.Token{{ID: 7, Text: "hello"}, {ID: 8, Text: " world"}},
	}}

	//nolint:staticcheck // SA1012: nil ctx is part of the branch under test.
	ch := session.GenerateStream(nil)
	got := core.NewBuilder()
	timeout := time.After(2 * time.Second)
	for {
		select {
		case tok, ok := <-ch:
			if !ok {
				if got.String() != "hello world" {
					t.Fatalf("stream text = %q, want hello world", got.String())
				}
				return
			}
			got.WriteString(tok.Text)
		case <-timeout:
			t.Fatal("timed out waiting for stream")
		}
	}
}

// TestSession_GenerateStream_EmptyTextContinue_Good — a token decoding to the
// empty string is skipped (the `if text == ""` continue), and only the
// non-empty token reaches the channel.
func TestSession_GenerateStream_EmptyTextContinue_Good(t *testing.T) {
	session := &Session{session: &sessionfake.Handle{
		Tokens: []inference.Token{{ID: 7, Text: ""}, {ID: 8, Text: "kept"}},
	}}

	ch := session.GenerateStream(context.Background())
	var got []spine.Token
	timeout := time.After(2 * time.Second)
	for {
		select {
		case tok, ok := <-ch:
			if !ok {
				if len(got) != 1 || got[0].Text != "kept" {
					t.Fatalf("stream tokens = %+v, want only [kept]", got)
				}
				return
			}
			got = append(got, tok)
		case <-timeout:
			t.Fatal("timed out waiting for stream")
		}
	}
}

// TestSessionParserControlToken_Empty_Good — the empty-string fast path
// returns false without scanning the marker set.
func TestSessionParserControlToken_Empty_Good(t *testing.T) {
	if sessionParserControlToken("") {
		t.Fatal("sessionParserControlToken(\"\") = true, want false")
	}
	if sessionParserControlToken("plain text no marker") {
		t.Fatal("sessionParserControlToken(plain) = true, want false")
	}
	if !sessionParserControlToken("a <start_of_turn> b") {
		t.Fatal("sessionParserControlToken(start_of_turn) = false, want true")
	}
}

// TestSession_CaptureKVWithOptions_OptionsHandle_Good drives the
// CaptureKVWithOptions arm (handle implements the with-options snapshotter)
// plus the RawKVOnly DropFloat32 post-step.
func TestSession_CaptureKVWithOptions_OptionsHandle_Good(t *testing.T) {
	native := &optionSnapshotHandle{optionSnapshot: sessionfake.TestKVSnapshot()}
	session := &Session{session: native}

	snapshot, err := session.CaptureKVWithOptions(kv.CaptureOptions{RawKVOnly: true})
	if err != nil {
		t.Fatalf("CaptureKVWithOptions() error = %v", err)
	}
	if !native.sawOptions.RawKVOnly {
		t.Fatalf("native options = %+v, want RawKVOnly forwarded", native.sawOptions)
	}
	// RawKVOnly dropped the decoded float32 view, keeping only the raw bytes.
	if snapshot == nil || len(snapshot.Layers[0].Heads[0].Key) != 0 {
		t.Fatalf("RawKVOnly snapshot kept float32 view = %+v", snapshot)
	}
}

// TestSession_CaptureKVWithOptions_OptionsHandleErr_Ugly — the with-options
// snapshotter's error is propagated verbatim.
func TestSession_CaptureKVWithOptions_OptionsHandleErr_Ugly(t *testing.T) {
	wantErr := core.NewError("option capture failed")
	session := &Session{session: &optionSnapshotHandle{optionErr: wantErr}}

	if _, err := session.CaptureKVWithOptions(kv.CaptureOptions{}); !core.Is(err, wantErr) {
		t.Fatalf("CaptureKVWithOptions() error = %v, want %v", err, wantErr)
	}
}

// TestSessionAnalyzeKV_CaptureErr_Ugly — AnalyzeKV surfaces the capture error
// without producing an analysis.
func TestSessionAnalyzeKV_CaptureErr_Ugly(t *testing.T) {
	wantErr := core.NewError("analyze capture failed")
	session := &Session{session: &sessionfake.Handle{CaptureErr: wantErr}}

	analysis, err := session.AnalyzeKV()
	if !core.Is(err, wantErr) {
		t.Fatalf("AnalyzeKV() error = %v, want %v", err, wantErr)
	}
	if analysis != nil {
		t.Fatalf("AnalyzeKV() = %+v, want nil on capture error", analysis)
	}
}

// TestSessionSaveKV_CaptureErr_Ugly — SaveKV surfaces the capture error
// before any path write.
func TestSessionSaveKV_CaptureErr_Ugly(t *testing.T) {
	wantErr := core.NewError("save capture failed")
	session := &Session{session: &sessionfake.Handle{CaptureErr: wantErr}}

	if err := session.SaveKV(core.PathJoin(t.TempDir(), "x.kvbin")); !core.Is(err, wantErr) {
		t.Fatalf("SaveKV() error = %v, want %v", err, wantErr)
	}
}

// TestSession_RestoreKV_NoRestorer_Bad — a handle that does not implement
// RestoreKV (nativeSessionRestorer) returns the no-restore sentinel.
func TestSession_RestoreKV_NoRestorer_Bad(t *testing.T) {
	session := &Session{session: &baseHandle{}}

	if err := session.RestoreKV(sessionTestRootSnapshot()); !core.Is(err, errNativeNoKVRestore) {
		t.Fatalf("RestoreKV(no restorer) error = %v, want %v", err, errNativeNoKVRestore)
	}
}

// TestSession_RestoreKV_RestorerErr_Ugly — the restorer's error is propagated
// and the agent-memory cache is left untouched.
func TestSession_RestoreKV_RestorerErr_Ugly(t *testing.T) {
	wantErr := core.NewError("restore failed")
	session := &Session{session: &sessionfake.Handle{RestoreErr: wantErr}}

	if err := session.RestoreKV(sessionTestRootSnapshot()); !core.Is(err, wantErr) {
		t.Fatalf("RestoreKV() error = %v, want %v", err, wantErr)
	}
}

// TestSession_LoadKV_LoadErr_Ugly — a missing KV file surfaces the load error
// before any restore.
func TestSession_LoadKV_LoadErr_Ugly(t *testing.T) {
	session := &Session{session: &sessionfake.Handle{}}

	if err := session.LoadKV(core.PathJoin(t.TempDir(), "missing.kvbin")); err == nil {
		t.Fatal("LoadKV(missing) error = nil, want load failure")
	}
}

// TestSession_SaveKVToState_NilCtxAndNativeEncoding_Good — nil ctx is defaulted
// and EncodingNative sets RawKVOnly on the capture before SaveState.
func TestSession_SaveKVToState_NilCtxAndNativeEncoding_Good(t *testing.T) {
	store := state.NewInMemoryStore(nil)
	session := &Session{session: &sessionfake.Handle{KV: sessionfake.TestKVSnapshot()}}

	//nolint:staticcheck // SA1012: nil ctx is part of the branch under test.
	ref, err := session.SaveKVToState(nil, store, kv.StateOptions{
		URI:        "mlx://session/native",
		KVEncoding: kv.EncodingNative,
	})
	if err != nil {
		t.Fatalf("SaveKVToState() error = %v", err)
	}
	if ref.ChunkID == 0 {
		t.Fatalf("SaveKVToState() ref = %+v, want a stored chunk", ref)
	}
}

// TestSession_LoadKVFromState_NilCtxAndLoadErr_Ugly — a bogus chunk ref with a
// nil ctx defaults the context then surfaces the load failure.
func TestSession_LoadKVFromState_NilCtxAndLoadErr_Ugly(t *testing.T) {
	store := state.NewInMemoryStore(nil)
	session := &Session{session: &sessionfake.Handle{}}

	//nolint:staticcheck // SA1012: nil ctx is part of the branch under test.
	if err := session.LoadKVFromState(nil, store, state.ChunkRef{ChunkID: 999}); err == nil {
		t.Fatal("LoadKVFromState(bad ref) error = nil, want load failure")
	}
}

// TestSession_LoadKVPrefixBlocksFromState_NilBundle_Bad — a nil bundle returns the
// block-bundle-nil sentinel.
func TestSession_LoadKVPrefixBlocksFromState_NilBundle_Bad(t *testing.T) {
	session := &Session{session: &sessionfake.Handle{}}

	if err := session.LoadKVPrefixBlocksFromState(context.Background(), state.NewInMemoryStore(nil), nil, 0); !core.Is(err, errStateKVBlockBundleNil) {
		t.Fatalf("LoadKVPrefixBlocksFromState(nil bundle) error = %v, want %v", err, errStateKVBlockBundleNil)
	}
}

// TestSession_LoadKVPrefixBlocksFromState_BlockRestorerErr_Ugly — the native block
// restorer's RestoreKVBlocks error is propagated.
func TestSession_LoadKVPrefixBlocksFromState_BlockRestorerErr_Ugly(t *testing.T) {
	store := state.NewInMemoryStore(nil)
	writer := &Session{session: &sessionfake.Handle{
		KVBlocks: []kv.Block{
			{Index: 0, TokenStart: 0, TokenCount: 2, Snapshot: testNativeKVBlock([]int32{10, 20}, 2, []float32{1, 2, 3, 4}, []float32{9, 10, 11, 12}, nil, nil)},
		},
	}}
	bundle, err := writer.SaveKVBlocksToState(context.Background(), store, kv.StateBlockOptions{BlockSize: 2})
	if err != nil {
		t.Fatalf("SaveKVBlocksToState() error = %v", err)
	}

	wantErr := core.NewError("block restore failed")
	reader := &Session{session: &sessionfake.Handle{RestoreBlocksErr: wantErr}}
	if err := reader.LoadKVPrefixBlocksFromState(context.Background(), store, bundle, 2); !core.Is(err, wantErr) {
		t.Fatalf("LoadKVPrefixBlocksFromState() error = %v, want %v", err, wantErr)
	}
}

// TestSession_LoadKVPrefixBlocksFromState_SnapshotFallbackNativeEncoding_Good — with a
// native-encoded bundle and a handle lacking RestoreKVBlocks, the fallback
// sets RawKVOnly on the load and restores via RestoreKV.
func TestSession_LoadKVPrefixBlocksFromState_SnapshotFallbackNativeEncoding_Good(t *testing.T) {
	store := state.NewInMemoryStore(nil)
	writer := &Session{session: &sessionfake.Handle{
		KVBlocks: []kv.Block{
			{Index: 0, TokenStart: 0, TokenCount: 2, Snapshot: testNativeKVBlock([]int32{10, 20}, 2, []float32{1, 2, 3, 4}, []float32{9, 10, 11, 12}, nil, nil)},
		},
	}}
	bundle, err := writer.SaveKVBlocksToState(context.Background(), store, kv.StateBlockOptions{BlockSize: 2, KVEncoding: kv.EncodingNative})
	if err != nil {
		t.Fatalf("SaveKVBlocksToState() error = %v", err)
	}
	if bundle.KVEncoding != kv.EncodingNative {
		t.Fatalf("bundle encoding = %q, want native", bundle.KVEncoding)
	}

	native := &snapshotRestoreHandle{}
	reader := &Session{session: native}
	if err := reader.LoadKVPrefixBlocksFromState(context.Background(), store, bundle, 2); err != nil {
		t.Fatalf("LoadKVPrefixBlocksFromState() error = %v", err)
	}
	if native.restored == nil || len(native.restored.Tokens) != 2 {
		t.Fatalf("snapshot fallback restored = %+v, want two-token state", native.restored)
	}
}

// TestSession_LoadKVPrefixBlocksFromState_SnapshotFallbackLoadErr_Ugly — the snapshot
// fallback path surfaces a load failure (corrupt block ref) before restore.
func TestSession_LoadKVPrefixBlocksFromState_SnapshotFallbackLoadErr_Ugly(t *testing.T) {
	bundle := &kv.StateBlockBundle{
		Version:      kv.StateBlockVersion,
		Kind:         kv.StateBlockBundleKind,
		SnapshotHash: "snapshot",
		KVEncoding:   kv.EncodingNative,
		Architecture: "gemma4_text",
		TokenCount:   2,
		TokenOffset:  2,
		BlockSize:    2,
		NumLayers:    1,
		NumHeads:     1,
		SeqLen:       2,
		HeadDim:      2,
		Blocks: []kv.StateBlockRef{{
			Index:      0,
			TokenStart: 0,
			TokenCount: 2,
			State:      state.ChunkRef{ChunkID: 4242},
		}},
	}

	native := &snapshotRestoreHandle{}
	reader := &Session{session: native}
	if err := reader.LoadKVPrefixBlocksFromState(context.Background(), state.NewInMemoryStore(nil), bundle, 2); err == nil {
		t.Fatal("LoadKVPrefixBlocksFromState(corrupt ref) error = nil, want load failure")
	}
	if native.restored != nil {
		t.Fatalf("restored despite load error = %+v", native.restored)
	}
}

// TestSessionRestoreBundle_CompatErr_Ugly — RestoreBundle rejects a bundle
// whose snapshot is itself invalid (b.Snapshot() error) after compatibility
// passes. Here the model identity matches but the bundle carries no KV ref,
// so Snapshot() fails.
func TestSession_RestoreBundle_SnapshotErr_Ugly(t *testing.T) {
	session := &Session{
		session: &sessionfake.Handle{},
		info:    spine.ModelInfo{Architecture: "gemma4_text", NumLayers: 1},
	}
	b := &mlxbundle.Bundle{
		Version: mlxbundle.Version,
		Kind:    mlxbundle.Kind,
		Model:   mlxbundle.Model{Architecture: "gemma4_text", NumLayers: 1},
		KVHash:  "deadbeef",
		// No KV, no refs -> Snapshot() cannot materialise a snapshot.
	}

	if err := session.RestoreBundle(b); err == nil {
		t.Fatal("RestoreBundle(no snapshot source) error = nil, want snapshot failure")
	}
}

// TestSession_RestoreBundleFromState_NilCtxAndCompatErr_Ugly — a nil ctx is
// defaulted, then a mismatched model identity is rejected by
// CheckCompatibility before any State read.
func TestSession_RestoreBundleFromState_NilCtxAndCompatErr_Ugly(t *testing.T) {
	store := state.NewInMemoryStore(nil)
	session := &Session{
		session: &sessionfake.Handle{},
		info:    spine.ModelInfo{Architecture: "llama", NumLayers: 99},
	}
	b := &mlxbundle.Bundle{
		Version: mlxbundle.Version,
		Kind:    mlxbundle.Kind,
		Model:   mlxbundle.Model{Architecture: "gemma4_text", NumLayers: 1},
		KVHash:  "deadbeef",
	}

	//nolint:staticcheck // SA1012: nil ctx is part of the branch under test.
	if err := session.RestoreBundleFromState(nil, b, store); err == nil {
		t.Fatal("RestoreBundleFromState(mismatch) error = nil, want incompatibility")
	}
}

// TestSessionFork_ForkErr_Ugly — the native Fork error is propagated.
func TestSessionFork_ForkErr_Ugly(t *testing.T) {
	wantErr := core.NewError("fork failed")
	session := &Session{session: &sessionfake.Handle{ForkErr: wantErr}}

	forked, err := session.Fork()
	if !core.Is(err, wantErr) {
		t.Fatalf("Fork() error = %v, want %v", err, wantErr)
	}
	if forked != nil {
		t.Fatalf("Fork() = %+v, want nil on error", forked)
	}
}

// TestSession_Fork_NilForkResult_Bad — a native fork that returns a nil handle
// surfaces the nil-fork sentinel.
func TestSession_Fork_NilForkResult_Bad(t *testing.T) {
	// sessionfake.Handle.Forked defaults to nil, so Fork returns (nil, nil)
	// from the native layer, tripping the nil-fork guard.
	session := &Session{session: &sessionfake.Handle{}}

	forked, err := session.Fork()
	if !core.Is(err, errNativeNilSessionFork) {
		t.Fatalf("Fork(nil result) error = %v, want %v", err, errNativeNilSessionFork)
	}
	if forked != nil {
		t.Fatalf("Fork() = %+v, want nil", forked)
	}
}

// TestSession_Err_LiveHandle_Good — a live handle's Err is forwarded (the
// non-nil-guard return arm).
func TestSession_Err_LiveHandle_Good(t *testing.T) {
	wantErr := core.NewError("session err")
	session := &Session{session: &sessionfake.Handle{ErrValue: wantErr}}

	if err := session.Err(); !core.Is(err, wantErr) {
		t.Fatalf("Err() = %v, want %v", err, wantErr)
	}
}

// ---------------------------------------------------------------------------
// agent_memory.go — folded-prefill, wake/sleep error arms, reuse-parent-prefix
// ---------------------------------------------------------------------------

// TestShouldPrefillFoldedAgentMemory_AllArms_Good drives the predicate
// directly across its branches: out-of-range prefix, the folded_state meta
// fast path + the Lower/Trim normalisation form, the folded-state label fast
// path + normalisation form, an empty-label skip, and the no-match fall-off.
func TestShouldPrefillFoldedAgentMemory_AllArms_Good(t *testing.T) {
	// Prefix <= 0 -> false.
	if shouldPrefillFoldedAgentMemory(agent.StateIndexEntry{TokenStart: 0, TokenCount: 0}) {
		t.Fatal("zero-prefix entry = true, want false")
	}
	// Prefix beyond the fold cap -> false.
	if shouldPrefillFoldedAgentMemory(agent.StateIndexEntry{TokenStart: 0, TokenCount: foldedAgentMemoryPrefillWakeMaxTokens + 1}) {
		t.Fatal("over-cap entry = true, want false")
	}
	// Canonical meta fast path.
	if !shouldPrefillFoldedAgentMemory(agent.StateIndexEntry{
		TokenCount: 4,
		Meta:       map[string]string{"folded_state": "true"},
	}) {
		t.Fatal("meta folded_state=true = false, want true")
	}
	// Non-canonical meta -> Lower/Trim normalisation arm.
	if !shouldPrefillFoldedAgentMemory(agent.StateIndexEntry{
		TokenCount: 4,
		Meta:       map[string]string{"folded_state": "  TRUE  "},
	}) {
		t.Fatal("meta folded_state=' TRUE ' = false, want true")
	}
	// Non-true meta value with no matching label -> false.
	if shouldPrefillFoldedAgentMemory(agent.StateIndexEntry{
		TokenCount: 4,
		Meta:       map[string]string{"folded_state": "no"},
	}) {
		t.Fatal("meta folded_state=no = true, want false")
	}
	// Canonical label fast path (with a leading empty label that is skipped).
	if !shouldPrefillFoldedAgentMemory(agent.StateIndexEntry{
		TokenCount: 4,
		Labels:     []string{"", "folded-state"},
	}) {
		t.Fatal("label folded-state = false, want true")
	}
	// Non-canonical label -> Lower/Trim normalisation arm.
	if !shouldPrefillFoldedAgentMemory(agent.StateIndexEntry{
		TokenCount: 4,
		Labels:     []string{"  FOLDED-STATE  "},
	}) {
		t.Fatal("label ' FOLDED-STATE ' = false, want true")
	}
	// No meta, no matching label -> false (label loop fall-off).
	if shouldPrefillFoldedAgentMemory(agent.StateIndexEntry{
		TokenCount: 4,
		Labels:     []string{"state", "other"},
	}) {
		t.Fatal("unmatched labels = true, want false")
	}
}

// TestPrefillFoldedAgentMemory_GuardArms_Bad covers the helper's two early
// guards: a nil session and a nil/empty plan bundle.
func TestPrefillFoldedAgentMemory_GuardArms_Bad(t *testing.T) {
	ctx := context.Background()
	store := state.NewInMemoryStore(nil)

	var nilSession *Session
	if err := nilSession.prefillFoldedAgentMemory(ctx, store, &agent.WakePlan{Bundle: &kv.StateBlockBundle{}}, agent.WakeOptions{}); !core.Is(err, errAgentMemorySessionNil) {
		t.Fatalf("prefillFoldedAgentMemory(nil session) = %v, want %v", err, errAgentMemorySessionNil)
	}

	live := &Session{session: &sessionfake.Handle{}}
	if err := live.prefillFoldedAgentMemory(ctx, store, nil, agent.WakeOptions{}); !core.Is(err, errAgentMemoryFoldPlanNil) {
		t.Fatalf("prefillFoldedAgentMemory(nil plan) = %v, want %v", err, errAgentMemoryFoldPlanNil)
	}
	if err := live.prefillFoldedAgentMemory(ctx, store, &agent.WakePlan{Bundle: nil}, agent.WakeOptions{}); !core.Is(err, errAgentMemoryFoldPlanNil) {
		t.Fatalf("prefillFoldedAgentMemory(nil bundle) = %v, want %v", err, errAgentMemoryFoldPlanNil)
	}
}

// TestWakeAgentMemory_FoldedPrefill_Good drives the full folded-prefill wake
// arm: sleep a session to produce a real bundle, build an index whose entry
// carries the "folded-state" label, and wake from it. The wake takes the
// folded-prefill branch (prefillFoldedAgentMemory -> PrefillTokens) and
// reports the "folded-prefill" strategy.
func TestWakeAgentMemory_FoldedPrefill_Good(t *testing.T) {
	ctx := context.Background()
	store := state.NewInMemoryStore(nil)
	info := spine.ModelInfo{Architecture: "gemma4_text", NumLayers: 1, QuantBits: 4, ContextLength: 8}

	source := &Session{session: &sessionfake.Handle{KV: sessionfake.TestKVSnapshot()}, info: info}
	sleep, err := source.SleepAgentMemory(ctx, store, agent.SleepOptions{
		EntryURI:     "mlx://agent/folded",
		BlockOptions: kv.StateBlockOptions{BlockSize: 1},
	})
	if err != nil {
		t.Fatalf("SleepAgentMemory() error = %v", err)
	}

	bundle, err := kv.LoadStateBlockBundle(ctx, store, sleep.BundleURI)
	if err != nil {
		t.Fatalf("kv.LoadStateBlockBundle() error = %v", err)
	}
	foldedIndex, err := agent.NewStateIndex(bundle, agent.StateIndexOptions{
		BundleURI: sleep.BundleURI,
		ModelInfo: spine.ModelInfoToMemory(info),
		Entries: []agent.StateIndexEntry{{
			URI:        "mlx://agent/folded/entry",
			BundleURI:  sleep.BundleURI,
			TokenStart: 0,
			TokenCount: bundle.TokenCount,
			Labels:     []string{"folded-state"},
		}},
	})
	if err != nil {
		t.Fatalf("agent.NewStateIndex() error = %v", err)
	}

	awakeNative := &sessionfake.Handle{}
	awake := &Session{session: awakeNative, info: info}
	report, err := awake.WakeAgentMemory(ctx, store, agent.WakeOptions{
		Index:                  foldedIndex,
		EntryURI:               "mlx://agent/folded/entry",
		SkipCompatibilityCheck: true,
	})
	if err != nil {
		t.Fatalf("WakeAgentMemory(folded) error = %v", err)
	}
	if report.RestoreStrategy != "folded-prefill" {
		t.Fatalf("RestoreStrategy = %q, want folded-prefill", report.RestoreStrategy)
	}
	if len(awakeNative.PrefillTokensSeen) != 2 || awakeNative.PrefillTokensSeen[0] != 1 {
		t.Fatalf("folded prefill tokens = %+v, want [1 2]", awakeNative.PrefillTokensSeen)
	}
	// agentMemory cache is populated from the wake report.
	if awake.agentMemory == nil {
		t.Fatal("folded wake did not populate agentMemory cache")
	}
}

// TestWakeAgentMemory_NilCtxAndNilSession_Bad — the WakeAgentMemory nil ctx
// default arm and the nil-session guard.
func TestWakeAgentMemory_NilCtxAndNilSession_Bad(t *testing.T) {
	store := state.NewInMemoryStore(nil)
	var nilSession *Session

	//nolint:staticcheck // SA1012: nil ctx is part of the branch under test.
	if _, err := nilSession.WakeAgentMemory(nil, store, agent.WakeOptions{}); !core.Is(err, errAgentMemorySessionNil) {
		t.Fatalf("WakeAgentMemory(nil session) = %v, want %v", err, errAgentMemorySessionNil)
	}
}

// TestWakeAgentMemory_KVBlocksRestorerErr_Ugly — the native KV-block restorer
// error is propagated from the "kv-blocks" wake arm.
func TestWakeAgentMemory_KVBlocksRestorerErr_Ugly(t *testing.T) {
	ctx := context.Background()
	store := state.NewInMemoryStore(nil)
	info := spine.ModelInfo{Architecture: "gemma4_text", NumLayers: 1, QuantBits: 4, ContextLength: 8}

	source := &Session{session: &sessionfake.Handle{KV: sessionfake.TestKVSnapshot()}, info: info}
	sleep, err := source.SleepAgentMemory(ctx, store, agent.SleepOptions{
		EntryURI:     "mlx://agent/blockerr",
		BlockOptions: kv.StateBlockOptions{BlockSize: 1},
	})
	if err != nil {
		t.Fatalf("SleepAgentMemory() error = %v", err)
	}

	wantErr := core.NewError("kv-block restore failed")
	awake := &Session{session: &sessionfake.Handle{RestoreBlocksErr: wantErr}, info: info}
	if _, err := awake.WakeAgentMemory(ctx, store, agent.WakeOptions{
		IndexURI: sleep.IndexURI,
		EntryURI: sleep.EntryURI,
	}); !core.Is(err, wantErr) {
		t.Fatalf("WakeAgentMemory() error = %v, want %v", err, wantErr)
	}
}

// TestSleepAgentMemory_NilCtx_Good — SleepAgentMemory defaults a nil ctx and
// still writes durable state.
func TestSleepAgentMemory_NilCtx_Good(t *testing.T) {
	store := state.NewInMemoryStore(nil)
	info := spine.ModelInfo{Architecture: "gemma4_text", NumLayers: 1, QuantBits: 4, ContextLength: 8}
	session := &Session{session: &sessionfake.Handle{KV: sessionfake.TestKVSnapshot()}, info: info}

	//nolint:staticcheck // SA1012: nil ctx is part of the branch under test.
	report, err := session.SleepAgentMemory(nil, store, agent.SleepOptions{EntryURI: "mlx://agent/nilctx"})
	if err != nil {
		t.Fatalf("SleepAgentMemory(nil ctx) error = %v", err)
	}
	if report.TokenCount != 2 {
		t.Fatalf("report = %+v, want two-token state", report)
	}
}

// TestSleepAgentMemory_ReuseParentPrefix_Good drives the ReuseParentPrefix
// arm: a first sleep produces a parent bundle, then a second sleep with
// ReuseParentPrefix=true and the parent bundle URI loads that bundle as the
// reuse prefix (blockOpts.ReusePrefix / ReusePrefixTokens assignment).
func TestSleepAgentMemory_ReuseParentPrefix_Good(t *testing.T) {
	ctx := context.Background()
	store := state.NewInMemoryStore(nil)
	info := spine.ModelInfo{Architecture: "gemma4_text", NumLayers: 1, QuantBits: 4, ContextLength: 8}
	session := &Session{session: &sessionfake.Handle{KV: sessionfake.TestKVSnapshot()}, info: info}

	parent, err := session.SleepAgentMemory(ctx, store, agent.SleepOptions{
		EntryURI:     "mlx://agent/parent",
		BlockOptions: kv.StateBlockOptions{BlockSize: 1},
	})
	if err != nil {
		t.Fatalf("SleepAgentMemory(parent) error = %v", err)
	}

	// Clear the session's auto-tracked parent so we exercise the explicit
	// ParentBundleURI + reuse-load arm rather than the auto-graft.
	session.agentMemory = nil
	child, err := session.SleepAgentMemory(ctx, store, agent.SleepOptions{
		EntryURI:          "mlx://agent/child",
		ParentBundleURI:   parent.BundleURI,
		ReuseParentPrefix: true,
		BlockOptions:      kv.StateBlockOptions{BlockSize: 1},
	})
	if err != nil {
		t.Fatalf("SleepAgentMemory(child reuse) error = %v", err)
	}
	if child.ParentBundleURI != parent.BundleURI {
		t.Fatalf("child parent bundle = %q, want %q", child.ParentBundleURI, parent.BundleURI)
	}
}

// TestSleepAgentMemory_ReuseParentPrefixNeedsReader_Bad — ReuseParentPrefix
// against a write-only store returns the reader-required sentinel.
func TestSleepAgentMemory_ReuseParentPrefixNeedsReader_Bad(t *testing.T) {
	ctx := context.Background()
	info := spine.ModelInfo{Architecture: "gemma4_text", NumLayers: 1, QuantBits: 4, ContextLength: 8}
	session := &Session{session: &sessionfake.Handle{KV: sessionfake.TestKVSnapshot()}, info: info}

	if _, err := session.SleepAgentMemory(ctx, writeOnlyStore{}, agent.SleepOptions{
		EntryURI:          "mlx://agent/child",
		ParentBundleURI:   "mlx://agent/parent/bundle",
		ReuseParentPrefix: true,
		BlockOptions:      kv.StateBlockOptions{BlockSize: 1},
	}); !core.Is(err, errAgentMemoryReuseNeedsReader) {
		t.Fatalf("SleepAgentMemory(reuse, write-only) = %v, want %v", err, errAgentMemoryReuseNeedsReader)
	}
}

// TestSleepAgentMemory_ReuseParentPrefixLoadErr_Ugly — ReuseParentPrefix with
// an unresolvable parent bundle URI surfaces the load error.
func TestSleepAgentMemory_ReuseParentPrefixLoadErr_Ugly(t *testing.T) {
	ctx := context.Background()
	store := state.NewInMemoryStore(nil)
	info := spine.ModelInfo{Architecture: "gemma4_text", NumLayers: 1, QuantBits: 4, ContextLength: 8}
	session := &Session{session: &sessionfake.Handle{KV: sessionfake.TestKVSnapshot()}, info: info}

	if _, err := session.SleepAgentMemory(ctx, store, agent.SleepOptions{
		EntryURI:          "mlx://agent/child",
		ParentBundleURI:   "mlx://agent/missing/bundle",
		ReuseParentPrefix: true,
		BlockOptions:      kv.StateBlockOptions{BlockSize: 1},
	}); err == nil {
		t.Fatal("SleepAgentMemory(reuse, missing parent) error = nil, want load failure")
	}
}

// TestWakeState_NotAStore_Bad — WakeState with a request store that is not a
// state.Store returns the wake-needs-store sentinel.
func TestWakeState_NotAStore_Bad(t *testing.T) {
	session := &Session{session: &sessionfake.Handle{}}

	if _, err := session.WakeState(context.Background(), inference.AgentMemoryWakeRequest{Store: notAStore{}}); !core.Is(err, errAgentMemoryWakeNeedsStore) {
		t.Fatalf("WakeState(non-store) = %v, want %v", err, errAgentMemoryWakeNeedsStore)
	}
}

// TestWakeState_WakeErr_Ugly — a wake failure (missing index) is propagated
// through the contract wrapper.
func TestWakeState_WakeErr_Ugly(t *testing.T) {
	store := state.NewInMemoryStore(nil)
	session := &Session{session: &sessionfake.Handle{}}

	if _, err := session.WakeState(context.Background(), inference.AgentMemoryWakeRequest{Store: store}); err == nil {
		t.Fatal("WakeState(missing index) error = nil, want wake failure")
	}
}

// TestSleepState_NotAWriter_Bad — SleepState with a request store that is not
// a state.Writer returns the sleep-needs-store sentinel.
func TestSleepState_NotAWriter_Bad(t *testing.T) {
	session := &Session{session: &sessionfake.Handle{}}

	if _, err := session.SleepState(context.Background(), inference.AgentMemorySleepRequest{Store: notAStore{}}); !core.Is(err, errAgentMemorySleepNeedsStore) {
		t.Fatalf("SleepState(non-writer) = %v, want %v", err, errAgentMemorySleepNeedsStore)
	}
}

// TestSleepState_SleepErr_Ugly — a sleep failure (nil session handle)
// propagates through the contract wrapper.
func TestSleepState_SleepErr_Ugly(t *testing.T) {
	store := state.NewInMemoryStore(nil)
	session := &Session{} // nil handle -> SleepAgentMemory errors

	if _, err := session.SleepState(context.Background(), inference.AgentMemorySleepRequest{Store: store}); err == nil {
		t.Fatal("SleepState(nil handle) error = nil, want sleep failure")
	}
}

// TestAppendAndSleepAgentMemory_NilCtxAndAppendErr_Ugly — a nil ctx is
// defaulted, then a native append failure is surfaced before any sleep.
func TestAppendAndSleepAgentMemory_NilCtxAndAppendErr_Ugly(t *testing.T) {
	wantErr := core.NewError("append failed")
	store := state.NewInMemoryStore(nil)
	session := &Session{session: &sessionfake.Handle{AppendErr: wantErr}}

	//nolint:staticcheck // SA1012: nil ctx is part of the branch under test.
	if _, err := session.AppendAndSleepAgentMemory(nil, "obs", store, agent.SleepOptions{EntryURI: "mlx://x"}); !core.Is(err, wantErr) {
		t.Fatalf("AppendAndSleepAgentMemory() error = %v, want %v", err, wantErr)
	}
}

// TestGenerateAndSleepAgentMemory_NilCtxAndSleepErr_Ugly — a nil ctx is
// defaulted, generation runs, and a sleep failure (write-only store + nil
// session would already trip; here use a nil-handle child) is surfaced. We
// drive the post-generation sleep-error arm by sleeping into a write-only
// store with ReuseParentPrefix to force a sleep-side failure.
func TestGenerateAndSleepAgentMemory_NilCtx_Good(t *testing.T) {
	store := state.NewInMemoryStore(nil)
	info := spine.ModelInfo{Architecture: "gemma4_text", NumLayers: 1, QuantBits: 4, ContextLength: 8}
	session := &Session{
		session: &sessionfake.Handle{KV: sessionfake.TestKVSnapshot(), Tokens: []inference.Token{{ID: 1, Text: "answer"}}},
		info:    info,
	}

	//nolint:staticcheck // SA1012: nil ctx is part of the branch under test.
	text, report, err := session.GenerateAndSleepAgentMemory(nil, store, agent.SleepOptions{EntryURI: "mlx://gen"})
	if err != nil {
		t.Fatalf("GenerateAndSleepAgentMemory(nil ctx) error = %v", err)
	}
	if text != "answer" {
		t.Fatalf("generated text = %q, want answer", text)
	}
	if report == nil || report.TokenCount != 2 {
		t.Fatalf("report = %+v, want durable two-token state", report)
	}
}

// TestGenerateAndSleepAgentMemory_SleepErr_Ugly — generation succeeds but the
// sleep fails (reuse-parent-prefix against a missing parent), returning the
// generated text plus the sleep error and a nil report.
func TestGenerateAndSleepAgentMemory_SleepErr_Ugly(t *testing.T) {
	info := spine.ModelInfo{Architecture: "gemma4_text", NumLayers: 1, QuantBits: 4, ContextLength: 8}
	session := &Session{
		session: &sessionfake.Handle{KV: sessionfake.TestKVSnapshot(), Tokens: []inference.Token{{ID: 1, Text: "answer"}}},
		info:    info,
	}

	text, report, err := session.GenerateAndSleepAgentMemory(context.Background(), writeOnlyStore{}, agent.SleepOptions{
		EntryURI:          "mlx://gen",
		ParentBundleURI:   "mlx://agent/parent/bundle",
		ReuseParentPrefix: true,
		BlockOptions:      kv.StateBlockOptions{BlockSize: 1},
	})
	if err == nil {
		t.Fatal("GenerateAndSleepAgentMemory(sleep err) error = nil, want sleep failure")
	}
	if text != "answer" {
		t.Fatalf("generated text = %q, want answer (returned even on sleep error)", text)
	}
	if report != nil {
		t.Fatalf("report = %+v, want nil on sleep error", report)
	}
}

// TestGenerateAndSleepAgentMemory_PostGenCancel_Ugly — a context cancelled
// mid-stream (via AfterGenerate) trips the post-generation ctx.Err() guard,
// surfacing the partial text and the cancellation error.
func TestGenerateAndSleepAgentMemory_PostGenCancel_Ugly(t *testing.T) {
	store := state.NewInMemoryStore(nil)
	ctx, cancel := context.WithCancel(context.Background())
	native := &sessionfake.Handle{
		KV:            sessionfake.TestKVSnapshot(),
		Tokens:        []inference.Token{{ID: 1, Text: "partial"}},
		AfterGenerate: func(*sessionfake.Handle) { cancel() },
	}
	session := &Session{session: native}

	text, report, err := session.GenerateAndSleepAgentMemory(ctx, store, agent.SleepOptions{EntryURI: "mlx://gen"})
	if err == nil {
		t.Fatal("GenerateAndSleepAgentMemory(post-gen cancel) error = nil, want cancellation")
	}
	if text != "partial" {
		t.Fatalf("text = %q, want partial", text)
	}
	if report != nil {
		t.Fatalf("report = %+v, want nil on cancellation", report)
	}
}

// TestToInferenceWakeResult_NilReport_Bad — a nil wake report maps to nil.
func TestToInferenceWakeResult_NilReport_Bad(t *testing.T) {
	if got := ToInferenceWakeResult(nil); got != nil {
		t.Fatalf("ToInferenceWakeResult(nil) = %+v, want nil", got)
	}
}

// TestToInferenceAgentMemorySleepResult_NilReport_Bad — a nil sleep report
// maps to nil.
func TestToInferenceAgentMemorySleepResult_NilReport_Bad(t *testing.T) {
	if got := toInferenceAgentMemorySleepResult(nil); got != nil {
		t.Fatalf("toInferenceAgentMemorySleepResult(nil) = %+v, want nil", got)
	}
}

// TestAgentMemoryMetadataFromInference_AllAdapterRuntimeFields_Good exercises
// the remaining fresh-map field arms (adapter_path non-blank) and the merge
// arms that fold every adapter/runtime field into caller metadata that does
// not already define those keys (adapter_path, adapter_rank, adapter_alpha,
// runtime_device, runtime_version).
func TestAgentMemoryMetadataFromInference_AllAdapterRuntimeFields_Good(t *testing.T) {
	// Fresh map with a real adapter_path (the previously-uncovered non-blank
	// arm at the fresh-map adapter_path write).
	fresh := agentMemoryMetadataFromInference(inference.AgentMemorySleepRequest{
		Adapter: inference.AdapterIdentity{Path: "/models/adapter.safetensors"},
	})
	if fresh["adapter_path"] != "/models/adapter.safetensors" {
		t.Fatalf("fresh adapter_path = %q, want the supplied path", fresh["adapter_path"])
	}

	// Merge path: caller metadata present (so the dst-merge branch runs) but
	// none of the adapter/runtime keys defined, so every fold-in fires.
	merged := agentMemoryMetadataFromInference(inference.AgentMemorySleepRequest{
		Metadata: map[string]string{"suite": "inference"},
		Adapter: inference.AdapterIdentity{
			Hash:   "h",
			Path:   "/p",
			Format: "lora",
			Rank:   8,
			Alpha:  16,
		},
		Runtime: inference.RuntimeIdentity{
			Backend:   "metal",
			Device:    "gpu",
			CacheMode: "paged-q8",
			Version:   "v2",
		},
	})
	for k, want := range map[string]string{
		"suite":              "inference",
		"adapter_hash":       "h",
		"adapter_path":       "/p",
		"adapter_format":     "lora",
		"adapter_rank":       "8",
		"adapter_alpha":      "16",
		"runtime_backend":    "metal",
		"runtime_device":     "gpu",
		"runtime_cache_mode": "paged-q8",
		"runtime_version":    "v2",
	} {
		if merged[k] != want {
			t.Fatalf("merged[%q] = %q, want %q (full %+v)", k, merged[k], want, merged)
		}
	}
}

// ---------------------------------------------------------------------------
// Batch 2 — the deeper error-propagation arms reachable only by forcing an
// inner load/restore/capture call to fail.
// ---------------------------------------------------------------------------

// TestPrefillFoldedAgentMemory_LoadTokensErr_Ugly — a bundle whose block ref
// points at a missing chunk fails the token load inside the folded helper.
func TestPrefillFoldedAgentMemory_LoadTokensErr_Ugly(t *testing.T) {
	store := state.NewInMemoryStore(nil)
	plan := &agent.WakePlan{
		Bundle: &kv.StateBlockBundle{
			Version:      kv.StateBlockVersion,
			Kind:         kv.StateBlockBundleKind,
			SnapshotHash: "snapshot",
			KVEncoding:   kv.EncodingNative,
			Architecture: "gemma4_text",
			TokenCount:   2,
			TokenOffset:  2,
			BlockSize:    2,
			NumLayers:    1,
			NumHeads:     1,
			SeqLen:       2,
			HeadDim:      2,
			Blocks: []kv.StateBlockRef{{
				Index:      0,
				TokenStart: 0,
				TokenCount: 2,
				State:      state.ChunkRef{ChunkID: 9999},
			}},
		},
		Entry: agent.StateIndexEntry{URI: "mlx://x", TokenStart: 0, TokenCount: 2},
	}
	session := &Session{session: &sessionfake.Handle{}}

	if err := session.prefillFoldedAgentMemory(context.Background(), store, plan, agent.WakeOptions{}); err == nil {
		t.Fatal("prefillFoldedAgentMemory(missing chunk) error = nil, want load failure")
	}
}

// foldedPlanFromSleep sleeps a real two-token state and returns a WakePlan
// (bundle + entry) suitable for driving prefillFoldedAgentMemory directly.
func foldedPlanFromSleep(t *testing.T, store *state.InMemoryStore) *agent.WakePlan {
	t.Helper()
	info := spine.ModelInfo{Architecture: "gemma4_text", NumLayers: 1, QuantBits: 4, ContextLength: 8}
	source := &Session{session: &sessionfake.Handle{KV: sessionfake.TestKVSnapshot()}, info: info}
	sleep, err := source.SleepAgentMemory(context.Background(), store, agent.SleepOptions{
		EntryURI:     "mlx://agent/folded-src",
		BlockOptions: kv.StateBlockOptions{BlockSize: 1},
	})
	if err != nil {
		t.Fatalf("SleepAgentMemory() error = %v", err)
	}
	bundle, err := kv.LoadStateBlockBundle(context.Background(), store, sleep.BundleURI)
	if err != nil {
		t.Fatalf("kv.LoadStateBlockBundle() error = %v", err)
	}
	return &agent.WakePlan{
		Bundle: bundle,
		Entry: agent.StateIndexEntry{
			URI:        "mlx://agent/folded-src/entry",
			TokenStart: 0,
			TokenCount: bundle.TokenCount,
		},
	}
}

// TestPrefillFoldedAgentMemory_PrefillErr_Ugly — tokens load fine, but the
// native PrefillTokens call fails and the helper wraps it.
func TestPrefillFoldedAgentMemory_PrefillErr_Ugly(t *testing.T) {
	store := state.NewInMemoryStore(nil)
	plan := foldedPlanFromSleep(t, store)

	wantErr := core.NewError("prefill tokens failed")
	session := &Session{session: &sessionfake.Handle{PrefillErr: wantErr}}
	if err := session.prefillFoldedAgentMemory(context.Background(), store, plan, agent.WakeOptions{}); !core.Is(err, wantErr) {
		t.Fatalf("prefillFoldedAgentMemory(prefill err) = %v, want %v", err, wantErr)
	}
}

// TestWakeAgentMemory_FoldedPrefillErr_Ugly — the full WakeAgentMemory folded
// branch surfacing a prefill failure (covers the folded-prefill error return).
func TestWakeAgentMemory_FoldedPrefillErr_Ugly(t *testing.T) {
	ctx := context.Background()
	store := state.NewInMemoryStore(nil)
	info := spine.ModelInfo{Architecture: "gemma4_text", NumLayers: 1, QuantBits: 4, ContextLength: 8}

	source := &Session{session: &sessionfake.Handle{KV: sessionfake.TestKVSnapshot()}, info: info}
	sleep, err := source.SleepAgentMemory(ctx, store, agent.SleepOptions{
		EntryURI:     "mlx://agent/folded-err",
		BlockOptions: kv.StateBlockOptions{BlockSize: 1},
	})
	if err != nil {
		t.Fatalf("SleepAgentMemory() error = %v", err)
	}
	bundle, err := kv.LoadStateBlockBundle(ctx, store, sleep.BundleURI)
	if err != nil {
		t.Fatalf("kv.LoadStateBlockBundle() error = %v", err)
	}
	foldedIndex, err := agent.NewStateIndex(bundle, agent.StateIndexOptions{
		BundleURI: sleep.BundleURI,
		ModelInfo: spine.ModelInfoToMemory(info),
		Entries: []agent.StateIndexEntry{{
			URI:        "mlx://agent/folded-err/entry",
			BundleURI:  sleep.BundleURI,
			TokenStart: 0,
			TokenCount: bundle.TokenCount,
			Labels:     []string{"folded-state"},
		}},
	})
	if err != nil {
		t.Fatalf("agent.NewStateIndex() error = %v", err)
	}

	wantErr := core.NewError("folded prefill failed")
	awake := &Session{session: &sessionfake.Handle{PrefillErr: wantErr}, info: info}
	if _, err := awake.WakeAgentMemory(ctx, store, agent.WakeOptions{
		Index:                  foldedIndex,
		EntryURI:               "mlx://agent/folded-err/entry",
		SkipCompatibilityCheck: true,
	}); !core.Is(err, wantErr) {
		t.Fatalf("WakeAgentMemory(folded prefill err) = %v, want %v", err, wantErr)
	}
}

// TestWakeAgentMemory_SnapshotFallbackRestoreErr_Ugly — the snapshot-fallback
// wake arm (handle without RestoreKVBlocks) surfacing a RestoreKV failure.
func TestWakeAgentMemory_SnapshotFallbackRestoreErr_Ugly(t *testing.T) {
	ctx := context.Background()
	store := state.NewInMemoryStore(nil)
	info := spine.ModelInfo{Architecture: "gemma4_text", NumLayers: 1, QuantBits: 4, ContextLength: 8}

	source := &Session{session: &sessionfake.Handle{KV: sessionfake.TestKVSnapshot()}, info: info}
	sleep, err := source.SleepAgentMemory(ctx, store, agent.SleepOptions{
		EntryURI:     "mlx://agent/snap-restore-err",
		BlockOptions: kv.StateBlockOptions{BlockSize: 1},
	})
	if err != nil {
		t.Fatalf("SleepAgentMemory() error = %v", err)
	}

	wantErr := core.NewError("snapshot restore failed")
	awake := &Session{session: &snapshotRestoreErrHandle{err: wantErr}, info: info}
	if _, err := awake.WakeAgentMemory(ctx, store, agent.WakeOptions{
		IndexURI: sleep.IndexURI,
		EntryURI: sleep.EntryURI,
	}); !core.Is(err, wantErr) {
		t.Fatalf("WakeAgentMemory(snapshot restore err) = %v, want %v", err, wantErr)
	}
}

// TestSleepAgentMemory_SaveBlocksErr_Ugly — a capture/range failure inside
// SaveKVBlocksToState surfaces from the sleep block-write arm.
func TestSleepAgentMemory_SaveBlocksErr_Ugly(t *testing.T) {
	store := state.NewInMemoryStore(nil)
	info := spine.ModelInfo{Architecture: "gemma4_text", NumLayers: 1, QuantBits: 4, ContextLength: 8}
	wantErr := core.NewError("range blocks failed")
	// rangeErrHandle yields a RangeKVBlocks error; the all-capable fake's
	// SaveKVBlocksToState path streams via RangeKVBlocks, so the error bubbles.
	session := &Session{session: &rangeErrHandle{err: wantErr}, info: info}

	if _, err := session.SleepAgentMemory(context.Background(), store, agent.SleepOptions{
		EntryURI:     "mlx://agent/saveblocks-err",
		BlockOptions: kv.StateBlockOptions{BlockSize: 1},
	}); !core.Is(err, wantErr) {
		t.Fatalf("SleepAgentMemory(save blocks err) = %v, want %v", err, wantErr)
	}
}

// TestAppendAndSleepAgentMemory_PostAppendCancel_Ugly — a context cancelled
// during AppendPrompt (via the recording fake's hook) trips the post-append
// ctx.Err() guard before any sleep.
func TestAppendAndSleepAgentMemory_PostAppendCancel_Ugly(t *testing.T) {
	store := state.NewInMemoryStore(nil)
	ctx, cancel := context.WithCancel(context.Background())
	session := &Session{session: &cancelOnAppendHandle{cancel: cancel}}

	if _, err := session.AppendAndSleepAgentMemory(ctx, "obs", store, agent.SleepOptions{EntryURI: "mlx://x"}); err == nil {
		t.Fatal("AppendAndSleepAgentMemory(post-append cancel) error = nil, want cancellation")
	}
}

// TestSession_LoadKVPrefixBlocksFromState_SourceErr_Ugly — a prefixTokens larger than
// the bundle's TokenCount makes the native block-source builder fail before
// any restore.
func TestSession_LoadKVPrefixBlocksFromState_SourceErr_Ugly(t *testing.T) {
	store := state.NewInMemoryStore(nil)
	writer := &Session{session: &sessionfake.Handle{
		KVBlocks: []kv.Block{
			{Index: 0, TokenStart: 0, TokenCount: 2, Snapshot: testNativeKVBlock([]int32{10, 20}, 2, []float32{1, 2, 3, 4}, []float32{9, 10, 11, 12}, nil, nil)},
		},
	}}
	bundle, err := writer.SaveKVBlocksToState(context.Background(), store, kv.StateBlockOptions{BlockSize: 2})
	if err != nil {
		t.Fatalf("SaveKVBlocksToState() error = %v", err)
	}

	// sessionfake.Handle IS a block-restorer, so the native arm is taken and
	// the source builder rejects the over-large prefix.
	reader := &Session{session: &sessionfake.Handle{}}
	if err := reader.LoadKVPrefixBlocksFromState(context.Background(), store, bundle, bundle.TokenCount+100); err == nil {
		t.Fatal("LoadKVPrefixBlocksFromState(prefix > tokens) error = nil, want source failure")
	}
}

// TestSession_RestoreBundle_SnapshotMaterialiseErr_Ugly — a bundle that passes
// Validate + CheckCompatibility (it carries a KVPath ref) but whose KVPath
// file is missing, so b.Snapshot() fails.
func TestSession_RestoreBundle_SnapshotMaterialiseErr_Ugly(t *testing.T) {
	session := &Session{
		session: &sessionfake.Handle{},
		info:    spine.ModelInfo{Architecture: "gemma4_text", NumLayers: 1},
	}
	b := &mlxbundle.Bundle{
		Version: mlxbundle.Version,
		Kind:    mlxbundle.Kind,
		Model:   mlxbundle.Model{Architecture: "gemma4_text", NumLayers: 1},
		KVPath:  core.PathJoin(t.TempDir(), "missing.kvbin"),
	}

	if err := session.RestoreBundle(b); err == nil {
		t.Fatal("RestoreBundle(missing KVPath) error = nil, want snapshot load failure")
	}
}

// TestSession_RestoreBundleFromState_SnapshotErr_Ugly — a bundle that passes
// Validate + CheckCompatibility (it carries a State ref) but whose State chunk
// is missing, so b.SnapshotFromState() fails.
func TestSession_RestoreBundleFromState_SnapshotErr_Ugly(t *testing.T) {
	store := state.NewInMemoryStore(nil)
	session := &Session{
		session: &sessionfake.Handle{},
		info:    spine.ModelInfo{Architecture: "gemma4_text", NumLayers: 1},
	}
	b := &mlxbundle.Bundle{
		Version: mlxbundle.Version,
		Kind:    mlxbundle.Kind,
		Model:   mlxbundle.Model{Architecture: "gemma4_text", NumLayers: 1},
		KVHash:  "deadbeef",
		Refs: []mlxbundle.Ref{{
			Kind:  mlxbundle.RefState,
			URI:   "state://missing",
			State: state.ChunkRef{ChunkID: 7777},
		}},
	}

	if err := session.RestoreBundleFromState(context.Background(), b, store); err == nil {
		t.Fatal("RestoreBundleFromState(missing chunk) error = nil, want snapshot load failure")
	}
}

// TestSleepAgentMemory_BundleManifestSaveErr_Ugly — block payloads write fine
// (via PutBytes) but the bundle-manifest Put fails, surfacing from the sleep
// bundle-save arm.
func TestSleepAgentMemory_BundleManifestSaveErr_Ugly(t *testing.T) {
	wantErr := core.NewError("bundle manifest put failed")
	store := &putFailStore{InMemoryStore: state.NewInMemoryStore(nil), failOn: 1, err: wantErr}
	info := spine.ModelInfo{Architecture: "gemma4_text", NumLayers: 1, QuantBits: 4, ContextLength: 8}
	session := &Session{session: &sessionfake.Handle{KV: sessionfake.TestKVSnapshot()}, info: info}

	if _, err := session.SleepAgentMemory(context.Background(), store, agent.SleepOptions{
		EntryURI:     "mlx://agent/bundle-put-err",
		BlockOptions: kv.StateBlockOptions{BlockSize: 1},
	}); !core.Is(err, wantErr) {
		t.Fatalf("SleepAgentMemory(bundle put err) = %v, want %v", err, wantErr)
	}
}

// TestSleepAgentMemory_IndexSaveErr_Ugly — the bundle manifest writes, but the
// index-manifest Put (the second text Put) fails, surfacing from the sleep
// index-save arm.
func TestSleepAgentMemory_IndexSaveErr_Ugly(t *testing.T) {
	wantErr := core.NewError("index put failed")
	store := &putFailStore{InMemoryStore: state.NewInMemoryStore(nil), failOn: 2, err: wantErr}
	info := spine.ModelInfo{Architecture: "gemma4_text", NumLayers: 1, QuantBits: 4, ContextLength: 8}
	session := &Session{session: &sessionfake.Handle{KV: sessionfake.TestKVSnapshot()}, info: info}

	if _, err := session.SleepAgentMemory(context.Background(), store, agent.SleepOptions{
		EntryURI:     "mlx://agent/index-put-err",
		BlockOptions: kv.StateBlockOptions{BlockSize: 1},
	}); !core.Is(err, wantErr) {
		t.Fatalf("SleepAgentMemory(index put err) = %v, want %v", err, wantErr)
	}
}

// ---------------------------------------------------------------------------
// Test doubles for the store-type-assertion arms.
// ---------------------------------------------------------------------------

// notAStore satisfies neither state.Store nor state.Writer, so the
// WakeState / SleepState request-store assertions fail.
type notAStore struct{}

// writeOnlyStore is a state.Writer (Put only) that is NOT a state.Store (no
// Get) — it drives the "reuse requires a readable store" arm, where the
// parent-prefix reuse path must read the parent bundle back but cannot.
type writeOnlyStore struct{}

func (writeOnlyStore) Put(_ context.Context, _ string, _ state.PutOptions) (state.ChunkRef, error) {
	return state.ChunkRef{ChunkID: 1}, nil
}

// Compile-time guards: writeOnlyStore is a Writer but not a Store.
var (
	_ state.Writer = writeOnlyStore{}
)

// putFailStore wraps an InMemoryStore and fails the Nth text Put call. Block
// payloads go through the embedded PutBytes (always succeed), so only the
// bundle-manifest and index-manifest text Puts reach this override — letting a
// test fail the bundle save (failOn=1) or the index save (failOn=2)
// independently while every earlier write succeeds.
type putFailStore struct {
	*state.InMemoryStore
	failOn int
	calls  int
	err    error
}

func (s *putFailStore) Put(ctx context.Context, text string, opts state.PutOptions) (state.ChunkRef, error) {
	s.calls++
	if s.calls == s.failOn {
		return state.ChunkRef{}, s.err
	}
	return s.InMemoryStore.Put(ctx, text, opts)
}
