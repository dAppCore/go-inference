// SPDX-Licence-Identifier: EUPL-1.2

package session

import (
	"context"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/inference/kv"
	"dappco.re/go/inference/model/spine"
	memvid "dappco.re/go/inference/model/state"
	"dappco.re/go/inference/model/state/agent"
	"dappco.re/go/inference/model/state/session/internal/sessionfake"
)

func TestAgentMemoryInferenceContract_Good(t *testing.T) {
	ctx := context.Background()
	store := memvid.NewInMemoryStore(nil)
	tokenizer := inference.TokenizerIdentity{Hash: "tok-contract", ChatTemplate: "chat"}
	info := spine.ModelInfo{Architecture: "gemma4_text", NumLayers: 1, QuantBits: 4, ContextLength: 8}
	source := &Session{session: &sessionfake.Handle{KV: sessionfake.TestKVSnapshot()}, info: info}

	sleep, err := any(source).(inference.AgentMemorySession).SleepState(ctx, inference.AgentMemorySleepRequest{
		Store:     store,
		EntryURI:  "mlx://agent/contract",
		Title:     "contract state",
		Tokenizer: tokenizer,
		Adapter:   inference.AdapterIdentity{Hash: "adapter-contract", Format: "lora"},
		Runtime:   inference.RuntimeIdentity{Backend: "metal", CacheMode: "paged-q8"},
		BlockSize: 1,
		Encoding:  string(kv.EncodingNative),
		Metadata:  map[string]string{"suite": "inference"},
	})

	if err != nil {
		t.Fatalf("SleepState() error = %v", err)
	}
	if sleep.Entry.URI != "mlx://agent/contract" || sleep.TokenCount != 2 || sleep.BlocksWritten != 1 {
		t.Fatalf("SleepState() = %+v, want contract state with one block", sleep)
	}
	if sleep.Index.URI == "" || sleep.Bundle.URI == "" {
		t.Fatalf("SleepState refs = %+v/%+v, want index and bundle refs", sleep.Index, sleep.Bundle)
	}
	index, err := agent.LoadMemvidIndex(ctx, store, sleep.Index.URI)
	if err != nil {
		t.Fatalf("agent.LoadMemvidIndex(contract) error = %v", err)
	}
	if index.Entries[0].Meta["adapter_hash"] != "adapter-contract" || index.Entries[0].Meta["runtime_backend"] != "metal" || index.Entries[0].Meta["runtime_cache_mode"] != "paged-q8" {
		t.Fatalf("contract metadata = %+v, want adapter/runtime identity", index.Entries[0].Meta)
	}

	awakeNative := &sessionfake.Handle{}
	awake := &Session{session: awakeNative, info: info}
	wake, err := any(awake).(inference.AgentMemorySession).WakeState(ctx, inference.AgentMemoryWakeRequest{
		Store:     store,
		IndexURI:  sleep.Index.URI,
		EntryURI:  sleep.Entry.URI,
		Tokenizer: tokenizer,
	})

	if err != nil {
		t.Fatalf("WakeState() error = %v", err)
	}
	if wake.Entry.URI != sleep.Entry.URI || wake.PrefixTokens != 2 || awakeNative.RestoredKV == nil {
		t.Fatalf("WakeState() = %+v restored=%+v, want restored contract state", wake, awakeNative.RestoredKV)
	}
}

func TestAppendAndSleepAgentMemory_NoReply_Good(t *testing.T) {
	ctx := context.Background()
	store := memvid.NewInMemoryStore(nil)
	native := &sessionfake.Handle{KV: sessionfake.TestKVSnapshot()}
	session := &Session{
		session: native,
		info:    spine.ModelInfo{Architecture: "gemma4_text", NumLayers: 1, QuantBits: 4, ContextLength: 8},
	}

	report, err := session.AppendAndSleepAgentMemory(ctx, "repo observation: tests pass", store, agent.SleepOptions{
		EntryURI: "mlx://agent/no-reply",
		Title:    "No reply observation",
	})

	if err != nil {
		t.Fatalf("AppendAndSleepAgentMemory() error = %v", err)
	}
	if native.AppendPromptSeen != "repo observation: tests pass" {
		t.Fatalf("append prompt = %q, want observation", native.AppendPromptSeen)
	}
	if native.GenerateCalls != 0 {
		t.Fatalf("Generate calls = %d, want no-reply append/sleep path", native.GenerateCalls)
	}
	if report.EntryURI != "mlx://agent/no-reply" || report.TokenCount != 2 {
		t.Fatalf("report = %+v, want durable two-token state", report)
	}
}

func TestAgentMemoryWakeSleep_Bad(t *testing.T) {
	ctx := context.Background()
	store := memvid.NewInMemoryStore(nil)
	var session *Session
	if _, err := session.SleepAgentMemory(ctx, store, agent.SleepOptions{}); err == nil {
		t.Fatal("SleepAgentMemory(nil session) error = nil")
	}
	session = &Session{session: &sessionfake.Handle{}}
	if _, err := session.SleepAgentMemory(ctx, nil, agent.SleepOptions{}); err == nil {
		t.Fatal("SleepAgentMemory(nil store) error = nil")
	}
	if _, err := session.WakeAgentMemory(ctx, store, agent.WakeOptions{}); err == nil {
		t.Fatal("WakeAgentMemory(missing index) error = nil")
	}

	bundle := kvSnapshotIndexTestBundle()
	index, err := agent.NewMemvidIndex(bundle, agent.MemvidIndexOptions{
		BundleURI: "mlx://bundle",
		ModelInfo: spine.ModelInfoToMemory(spine.ModelInfo{Architecture: "gemma4_text", NumLayers: 1}),
		Entries: []agent.MemvidIndexEntry{{
			URI:        "mlx://chapter",
			TokenStart: 0,
			TokenCount: 1,
		}},
	})
	if err != nil {
		t.Fatalf("agent.NewMemvidIndex() error = %v", err)
	}
	_, err = session.WakeAgentMemory(ctx, store, agent.WakeOptions{
		Index:    index,
		EntryURI: "mlx://chapter",
	})
	if err == nil {
		t.Fatal("WakeAgentMemory(missing bundle) error = nil")
	}
}

// TestAppendAndSleepAgentMemory_Cancelled_Ugly — a context cancelled before
// the call returns the context error and never appends or sleeps. Covers the
// pre-append ctx.Err() guard.
func TestAppendAndSleepAgentMemory_Cancelled_Ugly(t *testing.T) {
	store := memvid.NewInMemoryStore(nil)
	native := &sessionfake.Handle{KV: sessionfake.TestKVSnapshot()}
	session := &Session{session: native}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	if _, err := session.AppendAndSleepAgentMemory(ctx, "obs", store, agent.SleepOptions{EntryURI: "mlx://x"}); err == nil {
		t.Fatal("AppendAndSleepAgentMemory(cancelled) error = nil")
	}
	if native.AppendPromptSeen != "" {
		t.Fatalf("append ran despite cancellation: %q", native.AppendPromptSeen)
	}
}

// TestGenerateAndSleepAgentMemory_Cancelled_Ugly — a context cancelled before
// the call returns the context error before generation begins.
func TestGenerateAndSleepAgentMemory_Cancelled_Ugly(t *testing.T) {
	store := memvid.NewInMemoryStore(nil)
	native := &sessionfake.Handle{KV: sessionfake.TestKVSnapshot()}
	session := &Session{session: native}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	_, _, err := session.GenerateAndSleepAgentMemory(ctx, store, agent.SleepOptions{EntryURI: "mlx://x"})
	if err == nil {
		t.Fatal("GenerateAndSleepAgentMemory(cancelled) error = nil")
	}
	if native.GenerateCalls != 0 {
		t.Fatalf("generate ran despite pre-cancellation: %d calls", native.GenerateCalls)
	}
}

// TestGenerateAndSleepAgentMemory_GenerateError_Ugly — a native generation
// failure surfaces the partial text plus the error, and the sleep never runs.
func TestGenerateAndSleepAgentMemory_GenerateError_Ugly(t *testing.T) {
	wantErr := core.NewError("decode failed")
	store := memvid.NewInMemoryStore(nil)
	native := &sessionfake.Handle{
		KV:       sessionfake.TestKVSnapshot(),
		Tokens:   []inference.Token{{ID: 1, Text: "partial"}},
		ErrValue: wantErr,
	}
	session := &Session{session: native}

	text, report, err := session.GenerateAndSleepAgentMemory(context.Background(), store, agent.SleepOptions{EntryURI: "mlx://x"})
	if !core.Is(err, wantErr) {
		t.Fatalf("GenerateAndSleepAgentMemory() error = %v, want %v", err, wantErr)
	}
	if text != "partial" {
		t.Fatalf("partial text = %q, want partial", text)
	}
	if report != nil {
		t.Fatalf("report = %+v, want nil on generate error", report)
	}
}

// TestGenerateAndSleepAgentMemory_NilSession_Bad — a nil-handle session
// reports the nil-session error before generating.
func TestGenerateAndSleepAgentMemory_NilSession_Bad(t *testing.T) {
	store := memvid.NewInMemoryStore(nil)
	if _, _, err := (&Session{}).GenerateAndSleepAgentMemory(context.Background(), store, agent.SleepOptions{}); err == nil {
		t.Fatal("GenerateAndSleepAgentMemory(nil handle) error = nil")
	}
}

// TestCloneStringMap_Good — a populated map is defensively copied; mutating
// the source afterwards must not reach the returned copy.
func TestCloneStringMap_Good(t *testing.T) {
	src := map[string]string{"suite": "inference", "lane": "a"}

	clone := cloneStringMap(src)
	src["suite"] = "mutated"

	if clone["suite"] != "inference" || clone["lane"] != "a" {
		t.Fatalf("cloneStringMap() = %+v, want independent copy", clone)
	}
}

// TestCloneStringMap_Bad — empty and nil inputs both collapse to nil so the
// caller can store the result without a separate emptiness check.
func TestCloneStringMap_Bad(t *testing.T) {
	if got := cloneStringMap(nil); got != nil {
		t.Fatalf("cloneStringMap(nil) = %+v, want nil", got)
	}
	if got := cloneStringMap(map[string]string{}); got != nil {
		t.Fatalf("cloneStringMap(empty) = %+v, want nil", got)
	}
}

// TestAgentMemoryLabelsFromInference_Good exercises the three size classes
// the builder branches on — single entry, all-empty values (key-only), and
// the multi-entry shared-buffer path — and asserts the sorted "key=value"
// output shape.
func TestAgentMemoryLabelsFromInference_Good(t *testing.T) {
	if got := agentMemoryLabelsFromInference(nil); got != nil {
		t.Fatalf("labels(nil) = %+v, want nil", got)
	}

	single := agentMemoryLabelsFromInference(map[string]string{"role": "planner"})
	if len(single) != 1 || single[0] != "role=planner" {
		t.Fatalf("labels(single) = %+v, want [role=planner]", single)
	}

	singleEmpty := agentMemoryLabelsFromInference(map[string]string{"folded-state": ""})
	if len(singleEmpty) != 1 || singleEmpty[0] != "folded-state" {
		t.Fatalf("labels(single empty) = %+v, want [folded-state]", singleEmpty)
	}

	allEmpty := agentMemoryLabelsFromInference(map[string]string{"b": "", "a": ""})
	if len(allEmpty) != 2 || allEmpty[0] != "a" || allEmpty[1] != "b" {
		t.Fatalf("labels(all empty) = %+v, want sorted [a b]", allEmpty)
	}

	multi := agentMemoryLabelsFromInference(map[string]string{
		"role":  "planner",
		"flag":  "",
		"lane":  "a",
		"stage": "draft",
	})
	if len(multi) != 4 {
		t.Fatalf("labels(multi) = %+v, want four entries", multi)
	}
	// SliceSort puts the bare "flag" key first, then the "key=value" forms.
	want := []string{"flag", "lane=a", "role=planner", "stage=draft"}
	for i, w := range want {
		if multi[i] != w {
			t.Fatalf("labels(multi)[%d] = %q, want %q (full %+v)", i, multi[i], w, multi)
		}
	}
}

// TestAgentMemoryMetadataFromInference_Good walks the three return shapes:
// the no-extras clone (defers to cloneStringMap), the no-user-metadata fast
// path (fresh map from adapter/runtime identity), and the merge path that
// folds adapter/runtime keys into caller-supplied metadata without clobbering
// existing keys. Whitespace-only adapter fields are filtered on every path.
func TestAgentMemoryMetadataFromInference_Good(t *testing.T) {
	// No extras → clone of caller metadata (nil when that is empty too).
	if got := agentMemoryMetadataFromInference(inference.AgentMemorySleepRequest{}); got != nil {
		t.Fatalf("metadata(empty) = %+v, want nil", got)
	}
	cloned := agentMemoryMetadataFromInference(inference.AgentMemorySleepRequest{
		Metadata: map[string]string{"suite": "inference"},
	})
	if cloned["suite"] != "inference" || len(cloned) != 1 {
		t.Fatalf("metadata(clone only) = %+v, want {suite}", cloned)
	}

	// No user metadata → fresh map built from adapter/runtime identity. The
	// whitespace-only Adapter.Path must be dropped by the Trim guard.
	fresh := agentMemoryMetadataFromInference(inference.AgentMemorySleepRequest{
		Adapter: inference.AdapterIdentity{
			Hash:   "adapter-hash",
			Path:   "   ",
			Format: "lora",
			Rank:   8,
			Alpha:  16,
		},
		Runtime: inference.RuntimeIdentity{Backend: "metal", Device: "gpu", CacheMode: "paged-q8", Version: "v1"},
	})
	if fresh["adapter_hash"] != "adapter-hash" || fresh["adapter_format"] != "lora" {
		t.Fatalf("metadata(fresh) adapter = %+v", fresh)
	}
	if _, ok := fresh["adapter_path"]; ok {
		t.Fatalf("metadata(fresh) kept whitespace adapter_path = %+v", fresh)
	}
	if fresh["adapter_rank"] != "8" || fresh["adapter_alpha"] != "16" {
		t.Fatalf("metadata(fresh) rank/alpha = %+v", fresh)
	}
	if fresh["runtime_backend"] != "metal" || fresh["runtime_device"] != "gpu" ||
		fresh["runtime_cache_mode"] != "paged-q8" || fresh["runtime_version"] != "v1" {
		t.Fatalf("metadata(fresh) runtime = %+v", fresh)
	}

	// Merge path → caller metadata wins, adapter/runtime fold in around it.
	merged := agentMemoryMetadataFromInference(inference.AgentMemorySleepRequest{
		Metadata: map[string]string{"suite": "inference", "adapter_hash": "caller-wins"},
		Adapter:  inference.AdapterIdentity{Hash: "ignored", Format: "lora"},
		Runtime:  inference.RuntimeIdentity{Backend: "metal"},
	})
	if merged["suite"] != "inference" {
		t.Fatalf("metadata(merge) dropped caller suite = %+v", merged)
	}
	if merged["adapter_hash"] != "caller-wins" {
		t.Fatalf("metadata(merge) clobbered caller adapter_hash = %+v", merged)
	}
	if merged["adapter_format"] != "lora" || merged["runtime_backend"] != "metal" {
		t.Fatalf("metadata(merge) failed to fold identity = %+v", merged)
	}
}

// TestAddAgentMemoryMetadata_Good covers the standalone helper directly.
// Production inlined this logic into agentMemoryMetadataFromInference (the
// helper has no production caller — see the inline-equivalent comments
// there); the test pins its idempotence + Trim contract so a future
// re-use of the helper keeps the same shape.
func TestAddAgentMemoryMetadata_Good(t *testing.T) {
	// Empty + whitespace-only values are no-ops that return the input map.
	if got := addAgentMemoryMetadata(nil, "k", ""); got != nil {
		t.Fatalf("addAgentMemoryMetadata(nil,empty) = %+v, want nil", got)
	}
	if got := addAgentMemoryMetadata(nil, "k", "   "); got != nil {
		t.Fatalf("addAgentMemoryMetadata(nil,blank) = %+v, want nil", got)
	}

	// First real value allocates the map; a second key adds; an existing key
	// is left untouched (idempotent).
	meta := addAgentMemoryMetadata(nil, "adapter_hash", "h1")
	meta = addAgentMemoryMetadata(meta, "runtime_backend", "metal")
	meta = addAgentMemoryMetadata(meta, "adapter_hash", "h2")
	if meta["adapter_hash"] != "h1" || meta["runtime_backend"] != "metal" {
		t.Fatalf("addAgentMemoryMetadata() = %+v, want {adapter_hash:h1, runtime_backend:metal}", meta)
	}
}

func kvSnapshotIndexTestBundle() *kv.MemvidBlockBundle {
	return &kv.MemvidBlockBundle{
		Version:      kv.MemvidBlockVersion,
		Kind:         kv.MemvidBlockBundleKind,
		SnapshotHash: "snapshot",
		KVEncoding:   kv.EncodingNative,
		Architecture: "gemma4_text",
		TokenCount:   4,
		TokenOffset:  4,
		BlockSize:    2,
		NumLayers:    1,
		NumHeads:     1,
		SeqLen:       4,
		HeadDim:      2,
		Blocks: []kv.MemvidBlockRef{{
			Index:      0,
			TokenStart: 0,
			TokenCount: 2,
			Memvid:     memvid.ChunkRef{ChunkID: 1},
		}},
	}
}

// ---------------------------------------------------------------------------
// House test-standard triplets — one clean
// Test<File>_<Receiver>_<Symbol>_{Good,Bad,Ugly} per agent_memory.go public
// symbol, added alongside the richer scenario-named tests above.
// ---------------------------------------------------------------------------

// TestAgentMemory_Session_WakeAgentMemory_Good round-trips a sleep then a
// wake through the kv-blocks restore strategy.
func TestAgentMemory_Session_WakeAgentMemory_Good(t *testing.T) {
	ctx := context.Background()
	store := memvid.NewInMemoryStore(nil)
	info := spine.ModelInfo{Architecture: "gemma4_text", NumLayers: 1}
	asleep := &Session{session: &sessionfake.Handle{KV: sessionfake.TestKVSnapshot()}, info: info}

	sleep, err := asleep.SleepAgentMemory(ctx, store, agent.SleepOptions{EntryURI: "mlx://agent/wake-good"})
	if err != nil {
		t.Fatalf("SleepAgentMemory() error = %v", err)
	}

	awakeNative := &sessionfake.Handle{}
	awake := &Session{session: awakeNative, info: info}
	report, err := awake.WakeAgentMemory(ctx, store, agent.WakeOptions{IndexURI: sleep.IndexURI, EntryURI: sleep.EntryURI})

	if err != nil {
		t.Fatalf("WakeAgentMemory() error = %v", err)
	}
	if report.RestoreStrategy != "kv-blocks" || awakeNative.RestoredBlocks == nil {
		t.Fatalf("WakeAgentMemory() = %+v, want kv-blocks restore", report)
	}
}

// TestAgentMemory_Session_WakeAgentMemory_Bad — a nil session returns the
// sentinel before any State read.
func TestAgentMemory_Session_WakeAgentMemory_Bad(t *testing.T) {
	var session *Session
	store := memvid.NewInMemoryStore(nil)

	if _, err := session.WakeAgentMemory(context.Background(), store, agent.WakeOptions{}); !core.Is(err, errAgentMemorySessionNil) {
		t.Fatalf("WakeAgentMemory(nil session) error = %v, want %v", err, errAgentMemorySessionNil)
	}
}

// TestAgentMemory_Session_WakeAgentMemory_Ugly — an unresolvable index/entry
// surfaces the PlanWake failure before any restore is attempted.
func TestAgentMemory_Session_WakeAgentMemory_Ugly(t *testing.T) {
	session := &Session{session: &sessionfake.Handle{}}
	store := memvid.NewInMemoryStore(nil)

	if _, err := session.WakeAgentMemory(context.Background(), store, agent.WakeOptions{EntryURI: "mlx://missing"}); err == nil {
		t.Fatal("WakeAgentMemory(missing index) error = nil, want PlanWake failure")
	}
}

// TestAgentMemory_Session_Wake_Good — the lifecycle alias forwards to
// WakeAgentMemory unchanged.
func TestAgentMemory_Session_Wake_Good(t *testing.T) {
	ctx := context.Background()
	store := memvid.NewInMemoryStore(nil)
	info := spine.ModelInfo{Architecture: "gemma4_text", NumLayers: 1}
	asleep := &Session{session: &sessionfake.Handle{KV: sessionfake.TestKVSnapshot()}, info: info}
	sleep, err := asleep.Sleep(ctx, store, agent.SleepOptions{EntryURI: "mlx://agent/wake-alias-good"})
	if err != nil {
		t.Fatalf("Sleep() error = %v", err)
	}

	awake := &Session{session: &sessionfake.Handle{}, info: info}
	report, err := awake.Wake(ctx, store, agent.WakeOptions{IndexURI: sleep.IndexURI, EntryURI: sleep.EntryURI})

	if err != nil {
		t.Fatalf("Wake() error = %v", err)
	}
	if report.EntryURI != sleep.EntryURI {
		t.Fatalf("Wake() = %+v, want entry %q", report, sleep.EntryURI)
	}
}

// TestAgentMemory_Session_Wake_Bad — a nil session returns the same
// sentinel as the WakeAgentMemory path it forwards to.
func TestAgentMemory_Session_Wake_Bad(t *testing.T) {
	var session *Session
	store := memvid.NewInMemoryStore(nil)

	if _, err := session.Wake(context.Background(), store, agent.WakeOptions{}); !core.Is(err, errAgentMemorySessionNil) {
		t.Fatalf("Wake(nil session) error = %v, want %v", err, errAgentMemorySessionNil)
	}
}

// TestAgentMemory_Session_Wake_Ugly — an unresolvable index/entry surfaces
// the PlanWake failure through the forwarded call.
func TestAgentMemory_Session_Wake_Ugly(t *testing.T) {
	session := &Session{session: &sessionfake.Handle{}}
	store := memvid.NewInMemoryStore(nil)

	if _, err := session.Wake(context.Background(), store, agent.WakeOptions{EntryURI: "mlx://missing"}); err == nil {
		t.Fatal("Wake(missing index) error = nil, want PlanWake failure")
	}
}

// TestAgentMemory_Session_WakeState_Good implements the backend-neutral
// go-inference contract over an ordinary state.Store.
func TestAgentMemory_Session_WakeState_Good(t *testing.T) {
	ctx := context.Background()
	store := memvid.NewInMemoryStore(nil)
	info := spine.ModelInfo{Architecture: "gemma4_text", NumLayers: 1}
	asleep := &Session{session: &sessionfake.Handle{KV: sessionfake.TestKVSnapshot()}, info: info}
	sleep, err := asleep.SleepAgentMemory(ctx, store, agent.SleepOptions{EntryURI: "mlx://agent/wakestate-good"})
	if err != nil {
		t.Fatalf("SleepAgentMemory() error = %v", err)
	}

	awake := &Session{session: &sessionfake.Handle{}, info: info}
	result, err := awake.WakeState(ctx, inference.AgentMemoryWakeRequest{Store: store, IndexURI: sleep.IndexURI, EntryURI: sleep.EntryURI})

	if err != nil {
		t.Fatalf("WakeState() error = %v", err)
	}
	if result.Entry.URI != sleep.EntryURI {
		t.Fatalf("WakeState() = %+v, want entry %q", result, sleep.EntryURI)
	}
}

// TestAgentMemory_Session_WakeState_Bad — a Store value that does not
// implement state.Store is rejected before any wake is attempted.
func TestAgentMemory_Session_WakeState_Bad(t *testing.T) {
	session := &Session{session: &sessionfake.Handle{}}

	if _, err := session.WakeState(context.Background(), inference.AgentMemoryWakeRequest{Store: "not a store"}); !core.Is(err, errAgentMemoryWakeNeedsStore) {
		t.Fatalf("WakeState(bad store) error = %v, want %v", err, errAgentMemoryWakeNeedsStore)
	}
}

// TestAgentMemory_Session_WakeState_Ugly — a valid store type but an
// unresolvable entry surfaces the underlying WakeAgentMemory failure.
func TestAgentMemory_Session_WakeState_Ugly(t *testing.T) {
	session := &Session{session: &sessionfake.Handle{}}
	store := memvid.NewInMemoryStore(nil)

	if _, err := session.WakeState(context.Background(), inference.AgentMemoryWakeRequest{Store: store, EntryURI: "mlx://missing"}); err == nil {
		t.Fatal("WakeState(missing entry) error = nil, want PlanWake failure")
	}
}

// TestAgentMemory_Session_SleepAgentMemory_Good streams the retained KV
// state and writes a bundle manifest plus wake index.
func TestAgentMemory_Session_SleepAgentMemory_Good(t *testing.T) {
	session := &Session{session: &sessionfake.Handle{KV: sessionfake.TestKVSnapshot()}}
	store := memvid.NewInMemoryStore(nil)

	report, err := session.SleepAgentMemory(context.Background(), store, agent.SleepOptions{EntryURI: "mlx://agent/sleep-good"})

	if err != nil {
		t.Fatalf("SleepAgentMemory() error = %v", err)
	}
	if report.EntryURI != "mlx://agent/sleep-good" || report.IndexURI == "" || report.BundleURI == "" {
		t.Fatalf("SleepAgentMemory() = %+v, want populated refs", report)
	}
}

// TestAgentMemory_Session_SleepAgentMemory_Bad — a nil session returns the
// sentinel before any store write.
func TestAgentMemory_Session_SleepAgentMemory_Bad(t *testing.T) {
	var session *Session
	store := memvid.NewInMemoryStore(nil)

	if _, err := session.SleepAgentMemory(context.Background(), store, agent.SleepOptions{}); !core.Is(err, errAgentMemorySessionNil) {
		t.Fatalf("SleepAgentMemory(nil session) error = %v, want %v", err, errAgentMemorySessionNil)
	}
}

// TestAgentMemory_Session_SleepAgentMemory_Ugly — a nil store returns the
// sentinel, distinct from the nil-session Bad case.
func TestAgentMemory_Session_SleepAgentMemory_Ugly(t *testing.T) {
	session := &Session{session: &sessionfake.Handle{}}

	if _, err := session.SleepAgentMemory(context.Background(), nil, agent.SleepOptions{}); !core.Is(err, errAgentMemoryStoreNil) {
		t.Fatalf("SleepAgentMemory(nil store) error = %v, want %v", err, errAgentMemoryStoreNil)
	}
}

// TestAgentMemory_Session_Sleep_Good — the lifecycle alias forwards to
// SleepAgentMemory unchanged.
func TestAgentMemory_Session_Sleep_Good(t *testing.T) {
	session := &Session{session: &sessionfake.Handle{KV: sessionfake.TestKVSnapshot()}}
	store := memvid.NewInMemoryStore(nil)

	report, err := session.Sleep(context.Background(), store, agent.SleepOptions{EntryURI: "mlx://agent/sleep-alias-good"})

	if err != nil {
		t.Fatalf("Sleep() error = %v", err)
	}
	if report.EntryURI != "mlx://agent/sleep-alias-good" {
		t.Fatalf("Sleep() = %+v, want the requested entry", report)
	}
}

// TestAgentMemory_Session_Sleep_Bad — a nil session returns the same
// sentinel as the SleepAgentMemory path it forwards to.
func TestAgentMemory_Session_Sleep_Bad(t *testing.T) {
	var session *Session
	store := memvid.NewInMemoryStore(nil)

	if _, err := session.Sleep(context.Background(), store, agent.SleepOptions{}); !core.Is(err, errAgentMemorySessionNil) {
		t.Fatalf("Sleep(nil session) error = %v, want %v", err, errAgentMemorySessionNil)
	}
}

// TestAgentMemory_Session_Sleep_Ugly — a nil store surfaces the same
// sentinel through the forwarded call.
func TestAgentMemory_Session_Sleep_Ugly(t *testing.T) {
	session := &Session{session: &sessionfake.Handle{}}

	if _, err := session.Sleep(context.Background(), nil, agent.SleepOptions{}); !core.Is(err, errAgentMemoryStoreNil) {
		t.Fatalf("Sleep(nil store) error = %v, want %v", err, errAgentMemoryStoreNil)
	}
}

// TestAgentMemory_Session_SleepState_Good implements the backend-neutral
// go-inference contract over an ordinary state.Writer.
func TestAgentMemory_Session_SleepState_Good(t *testing.T) {
	session := &Session{session: &sessionfake.Handle{KV: sessionfake.TestKVSnapshot()}}
	store := memvid.NewInMemoryStore(nil)

	result, err := session.SleepState(context.Background(), inference.AgentMemorySleepRequest{Store: store, EntryURI: "mlx://agent/sleepstate-good"})

	if err != nil {
		t.Fatalf("SleepState() error = %v", err)
	}
	if result.Entry.URI != "mlx://agent/sleepstate-good" {
		t.Fatalf("SleepState() = %+v, want the requested entry", result)
	}
}

// TestAgentMemory_Session_SleepState_Bad — a Store value that does not
// implement state.Writer is rejected before any sleep is attempted.
func TestAgentMemory_Session_SleepState_Bad(t *testing.T) {
	session := &Session{session: &sessionfake.Handle{}}

	if _, err := session.SleepState(context.Background(), inference.AgentMemorySleepRequest{Store: "not a writer"}); !core.Is(err, errAgentMemorySleepNeedsStore) {
		t.Fatalf("SleepState(bad store) error = %v, want %v", err, errAgentMemorySleepNeedsStore)
	}
}

// TestAgentMemory_Session_SleepState_Ugly — a valid writer but a nil session
// surfaces the underlying SleepAgentMemory failure.
func TestAgentMemory_Session_SleepState_Ugly(t *testing.T) {
	var session *Session
	store := memvid.NewInMemoryStore(nil)

	if _, err := session.SleepState(context.Background(), inference.AgentMemorySleepRequest{Store: store}); !core.Is(err, errAgentMemorySessionNil) {
		t.Fatalf("SleepState(nil session) error = %v, want %v", err, errAgentMemorySessionNil)
	}
}

// TestAgentMemory_Session_AppendAndSleepAgentMemory_Good appends new prompt
// material then streams the resulting state to durable storage.
func TestAgentMemory_Session_AppendAndSleepAgentMemory_Good(t *testing.T) {
	native := &sessionfake.Handle{KV: sessionfake.TestKVSnapshot()}
	session := &Session{session: native}
	store := memvid.NewInMemoryStore(nil)

	report, err := session.AppendAndSleepAgentMemory(context.Background(), "observation", store, agent.SleepOptions{EntryURI: "mlx://agent/append-good"})

	if err != nil {
		t.Fatalf("AppendAndSleepAgentMemory() error = %v", err)
	}
	if native.AppendPromptSeen != "observation" {
		t.Fatalf("append prompt = %q, want observation", native.AppendPromptSeen)
	}
	if report.EntryURI != "mlx://agent/append-good" {
		t.Fatalf("AppendAndSleepAgentMemory() = %+v, want the requested entry", report)
	}
}

// TestAgentMemory_Session_AppendAndSleepAgentMemory_Bad — a context
// cancelled before the call returns the context error before any append.
func TestAgentMemory_Session_AppendAndSleepAgentMemory_Bad(t *testing.T) {
	native := &sessionfake.Handle{KV: sessionfake.TestKVSnapshot()}
	session := &Session{session: native}
	store := memvid.NewInMemoryStore(nil)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	if _, err := session.AppendAndSleepAgentMemory(ctx, "obs", store, agent.SleepOptions{EntryURI: "mlx://x"}); err == nil {
		t.Fatal("AppendAndSleepAgentMemory(cancelled) error = nil")
	}
	if native.AppendPromptSeen != "" {
		t.Fatalf("append ran despite pre-cancellation: %q", native.AppendPromptSeen)
	}
}

// TestAgentMemory_Session_AppendAndSleepAgentMemory_Ugly — the native
// append's error is propagated, distinct from the ctx-cancellation Bad case.
func TestAgentMemory_Session_AppendAndSleepAgentMemory_Ugly(t *testing.T) {
	wantErr := core.NewError("native append failed")
	session := &Session{session: &sessionfake.Handle{AppendErr: wantErr}}
	store := memvid.NewInMemoryStore(nil)

	if _, err := session.AppendAndSleepAgentMemory(context.Background(), "obs", store, agent.SleepOptions{EntryURI: "mlx://x"}); !core.Is(err, wantErr) {
		t.Fatalf("AppendAndSleepAgentMemory() error = %v, want %v", err, wantErr)
	}
}

// TestAgentMemory_Session_AppendAndSleep_Good — the lifecycle alias forwards
// to AppendAndSleepAgentMemory unchanged.
func TestAgentMemory_Session_AppendAndSleep_Good(t *testing.T) {
	native := &sessionfake.Handle{KV: sessionfake.TestKVSnapshot()}
	session := &Session{session: native}
	store := memvid.NewInMemoryStore(nil)

	report, err := session.AppendAndSleep(context.Background(), "observation", store, agent.SleepOptions{EntryURI: "mlx://agent/append-alias-good"})

	if err != nil {
		t.Fatalf("AppendAndSleep() error = %v", err)
	}
	if native.AppendPromptSeen != "observation" || report.EntryURI != "mlx://agent/append-alias-good" {
		t.Fatalf("AppendAndSleep() = %+v, append = %q", report, native.AppendPromptSeen)
	}
}

// TestAgentMemory_Session_AppendAndSleep_Bad — a cancelled context surfaces
// the same pre-append guard as the AppendAndSleepAgentMemory path.
func TestAgentMemory_Session_AppendAndSleep_Bad(t *testing.T) {
	session := &Session{session: &sessionfake.Handle{KV: sessionfake.TestKVSnapshot()}}
	store := memvid.NewInMemoryStore(nil)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	if _, err := session.AppendAndSleep(ctx, "obs", store, agent.SleepOptions{EntryURI: "mlx://x"}); err == nil {
		t.Fatal("AppendAndSleep(cancelled) error = nil")
	}
}

// TestAgentMemory_Session_AppendAndSleep_Ugly — the native append's error is
// propagated through the forwarded call.
func TestAgentMemory_Session_AppendAndSleep_Ugly(t *testing.T) {
	wantErr := core.NewError("native append failed")
	session := &Session{session: &sessionfake.Handle{AppendErr: wantErr}}
	store := memvid.NewInMemoryStore(nil)

	if _, err := session.AppendAndSleep(context.Background(), "obs", store, agent.SleepOptions{EntryURI: "mlx://x"}); !core.Is(err, wantErr) {
		t.Fatalf("AppendAndSleep() error = %v, want %v", err, wantErr)
	}
}

// TestAgentMemory_Session_GenerateAndSleepAgentMemory_Good generates a reply
// then streams the post-answer KV state to durable storage.
func TestAgentMemory_Session_GenerateAndSleepAgentMemory_Good(t *testing.T) {
	session := &Session{session: &sessionfake.Handle{
		KV:     sessionfake.TestKVSnapshot(),
		Tokens: []inference.Token{{ID: 1, Text: "hello"}},
	}}
	store := memvid.NewInMemoryStore(nil)

	text, report, err := session.GenerateAndSleepAgentMemory(context.Background(), store, agent.SleepOptions{EntryURI: "mlx://agent/gen-good"})

	if err != nil {
		t.Fatalf("GenerateAndSleepAgentMemory() error = %v", err)
	}
	if text != "hello" || report.EntryURI != "mlx://agent/gen-good" {
		t.Fatalf("GenerateAndSleepAgentMemory() = %q/%+v", text, report)
	}
}

// TestAgentMemory_Session_GenerateAndSleepAgentMemory_Bad — a nil session
// returns the sentinel before any generation is attempted.
func TestAgentMemory_Session_GenerateAndSleepAgentMemory_Bad(t *testing.T) {
	session := &Session{}
	store := memvid.NewInMemoryStore(nil)

	if _, _, err := session.GenerateAndSleepAgentMemory(context.Background(), store, agent.SleepOptions{}); !core.Is(err, errAgentMemorySessionNil) {
		t.Fatalf("GenerateAndSleepAgentMemory(nil handle) error = %v, want %v", err, errAgentMemorySessionNil)
	}
}

// TestAgentMemory_Session_GenerateAndSleepAgentMemory_Ugly — a context
// cancelled before the call returns the context error before the nil-session
// guard is even reached.
func TestAgentMemory_Session_GenerateAndSleepAgentMemory_Ugly(t *testing.T) {
	session := &Session{session: &sessionfake.Handle{KV: sessionfake.TestKVSnapshot()}}
	store := memvid.NewInMemoryStore(nil)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	if _, _, err := session.GenerateAndSleepAgentMemory(ctx, store, agent.SleepOptions{EntryURI: "mlx://x"}); err == nil {
		t.Fatal("GenerateAndSleepAgentMemory(cancelled) error = nil")
	}
}

// TestAgentMemory_Session_GenerateAndSleep_Good — the lifecycle alias
// forwards to GenerateAndSleepAgentMemory unchanged.
func TestAgentMemory_Session_GenerateAndSleep_Good(t *testing.T) {
	session := &Session{session: &sessionfake.Handle{
		KV:     sessionfake.TestKVSnapshot(),
		Tokens: []inference.Token{{ID: 1, Text: "hello"}},
	}}
	store := memvid.NewInMemoryStore(nil)

	text, report, err := session.GenerateAndSleep(context.Background(), store, agent.SleepOptions{EntryURI: "mlx://agent/gen-alias-good"})

	if err != nil {
		t.Fatalf("GenerateAndSleep() error = %v", err)
	}
	if text != "hello" || report.EntryURI != "mlx://agent/gen-alias-good" {
		t.Fatalf("GenerateAndSleep() = %q/%+v", text, report)
	}
}

// TestAgentMemory_Session_GenerateAndSleep_Bad — a nil session returns the
// same sentinel as the GenerateAndSleepAgentMemory path it forwards to.
func TestAgentMemory_Session_GenerateAndSleep_Bad(t *testing.T) {
	session := &Session{}
	store := memvid.NewInMemoryStore(nil)

	if _, _, err := session.GenerateAndSleep(context.Background(), store, agent.SleepOptions{}); !core.Is(err, errAgentMemorySessionNil) {
		t.Fatalf("GenerateAndSleep(nil handle) error = %v, want %v", err, errAgentMemorySessionNil)
	}
}

// TestAgentMemory_Session_GenerateAndSleep_Ugly — a cancelled context
// surfaces the same pre-flight guard through the forwarded call.
func TestAgentMemory_Session_GenerateAndSleep_Ugly(t *testing.T) {
	session := &Session{session: &sessionfake.Handle{KV: sessionfake.TestKVSnapshot()}}
	store := memvid.NewInMemoryStore(nil)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	if _, _, err := session.GenerateAndSleep(ctx, store, agent.SleepOptions{EntryURI: "mlx://x"}); err == nil {
		t.Fatal("GenerateAndSleep(cancelled) error = nil")
	}
}

// TestAgentMemory_WakeOptionsFromInference_Good maps a populated wake
// request onto agent.WakeOptions field-for-field.
func TestAgentMemory_WakeOptionsFromInference_Good(t *testing.T) {
	req := inference.AgentMemoryWakeRequest{
		IndexURI:               "mlx://index",
		EntryURI:               "mlx://entry",
		Tokenizer:              inference.TokenizerIdentity{Hash: "tok-hash"},
		SkipCompatibilityCheck: false,
	}

	got := WakeOptionsFromInference(req)

	if got.IndexURI != "mlx://index" || got.EntryURI != "mlx://entry" {
		t.Fatalf("WakeOptionsFromInference() = %+v, want URIs preserved", got)
	}
	if got.Tokenizer.Hash != "tok-hash" {
		t.Fatalf("WakeOptionsFromInference() tokenizer = %+v, want hash preserved", got.Tokenizer)
	}
}

// TestAgentMemory_WakeOptionsFromInference_Bad — a zero-value request maps
// to a zero-value-shaped WakeOptions rather than panicking.
func TestAgentMemory_WakeOptionsFromInference_Bad(t *testing.T) {
	got := WakeOptionsFromInference(inference.AgentMemoryWakeRequest{})

	if got.IndexURI != "" || got.EntryURI != "" || got.SkipCompatibilityCheck {
		t.Fatalf("WakeOptionsFromInference(zero) = %+v, want zero value", got)
	}
}

// TestAgentMemory_WakeOptionsFromInference_Ugly — SkipCompatibilityCheck is
// carried through untouched, the one boolean flag on the mapping.
func TestAgentMemory_WakeOptionsFromInference_Ugly(t *testing.T) {
	got := WakeOptionsFromInference(inference.AgentMemoryWakeRequest{SkipCompatibilityCheck: true})

	if !got.SkipCompatibilityCheck {
		t.Fatal("WakeOptionsFromInference() dropped SkipCompatibilityCheck = true")
	}
}

// TestAgentMemory_ToInferenceWakeResult_Good maps a populated wake report
// onto the go-inference result shape.
func TestAgentMemory_ToInferenceWakeResult_Good(t *testing.T) {
	report := &agent.WakeReport{
		EntryURI:     "mlx://entry",
		BundleURI:    "mlx://bundle",
		IndexURI:     "mlx://index",
		Title:        "chapter",
		PrefixTokens: 4,
		BundleTokens: 8,
		BlockSize:    2,
		BlocksRead:   2,
	}

	got := ToInferenceWakeResult(report)

	if got.Entry.URI != "mlx://entry" || got.PrefixTokens != 4 || got.BundleTokens != 8 {
		t.Fatalf("ToInferenceWakeResult() = %+v, want mapped report fields", got)
	}
	if got.Bundle.URI != "mlx://bundle" || got.Index.URI != "mlx://index" {
		t.Fatalf("ToInferenceWakeResult() refs = %+v, want bundle/index URIs", got)
	}
}

// TestAgentMemory_ToInferenceWakeResult_Bad — a nil report maps to a nil
// result rather than panicking.
func TestAgentMemory_ToInferenceWakeResult_Bad(t *testing.T) {
	if got := ToInferenceWakeResult(nil); got != nil {
		t.Fatalf("ToInferenceWakeResult(nil) = %+v, want nil", got)
	}
}

// TestAgentMemory_ToInferenceWakeResult_Ugly — a report with zero token
// counts still maps its URIs through untouched.
func TestAgentMemory_ToInferenceWakeResult_Ugly(t *testing.T) {
	report := &agent.WakeReport{EntryURI: "mlx://entry-only"}

	got := ToInferenceWakeResult(report)

	if got.Entry.URI != "mlx://entry-only" || got.PrefixTokens != 0 || got.BlocksRead != 0 {
		t.Fatalf("ToInferenceWakeResult(zero counts) = %+v, want zero counts preserved", got)
	}
}
