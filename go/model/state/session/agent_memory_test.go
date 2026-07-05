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
