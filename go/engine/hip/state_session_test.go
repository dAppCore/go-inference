// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import (
	"context"
	"encoding/json"
	"path/filepath"
	"strings"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/inference/model/state"
	"dappco.re/go/inference/model/state/filestore"
)

func TestStateSession_Good_WakeStateReturnsRefs(t *testing.T) {
	store := state.NewInMemoryStore(nil)
	session := NewStateSession(inference.ModelIdentity{Hash: "model-a"}, inference.TokenizerIdentity{Hash: "tok-a"}, nil)
	sleep := seedStateSessionKV(t, store, "state://entry", inference.ModelIdentity{Hash: "model-a"}, inference.TokenizerIdentity{Hash: "tok-a"})

	wake, err := session.WakeState(context.Background(), inference.AgentMemoryWakeRequest{
		Store:     store,
		IndexURI:  sleep.Entry.IndexURI,
		Model:     inference.ModelIdentity{Hash: "model-a"},
		Tokenizer: inference.TokenizerIdentity{Hash: "tok-a"},
	})

	core.RequireNoError(t, err)
	core.AssertEqual(t, "state://entry", wake.Entry.URI)
	core.AssertEqual(t, 3, wake.PrefixTokens)
	core.AssertEqual(t, defaultROCmStateBlockSize, wake.BlockSize)
	core.AssertEqual(t, 1, wake.BlocksRead)
	core.AssertEqual(t, "runtime_owned", wake.Labels["kv_restore"])
	core.AssertEqual(t, "block_stream", wake.Labels["kv_restore_path"])
	core.AssertEqual(t, "runtime_owned", wake.Bundle.Labels["kv_restore"])
	core.AssertEqual(t, "rocm", wake.Bundle.Labels["backend"])
}

func TestStateSession_Bad_WakeStateRejectsPromptTextState(t *testing.T) {
	store := state.NewInMemoryStore(nil)
	_, err := store.Put(context.Background(), "one two three", state.PutOptions{URI: "state://entry/text"})
	core.RequireNoError(t, err)
	session := NewStateSession(inference.ModelIdentity{}, inference.TokenizerIdentity{}, nil)

	wake, err := session.WakeState(context.Background(), inference.AgentMemoryWakeRequest{Store: store, EntryURI: "state://entry/text"})

	core.AssertNil(t, wake)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "KV state is required")
}

func TestStateSession_Bad_CloseFailureKeepsRuntime(t *testing.T) {
	runtime := &failingStateRuntime{err: core.NewError("close failed")}
	session := newStateSessionWithRuntime(inference.ModelIdentity{}, inference.TokenizerIdentity{}, nil, runtime)

	err := session.Close()

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "close failed")
	core.AssertEqual(t, 1, runtime.closeCalls)
	if session.runtime != runtime {
		t.Fatal("StateSession.Close cleared runtime after close failure")
	}
}

func TestStateSession_Bad_WakeRejectsModelHashMismatch(t *testing.T) {
	session := NewStateSession(inference.ModelIdentity{Hash: "model-a"}, inference.TokenizerIdentity{}, nil)

	_, err := session.WakeState(context.Background(), inference.AgentMemoryWakeRequest{
		Store:    state.NewInMemoryStore(nil),
		EntryURI: "state://entry",
		Model:    inference.ModelIdentity{Hash: "model-b"},
	})

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "model hash mismatch")
}

func TestStateSession_Bad_WakeRejectsModelArchitectureMismatch(t *testing.T) {
	session := NewStateSession(inference.ModelIdentity{Architecture: "qwen3"}, inference.TokenizerIdentity{}, nil)

	_, err := session.WakeState(context.Background(), inference.AgentMemoryWakeRequest{
		Store:    state.NewInMemoryStore(nil),
		EntryURI: "state://entry",
		Model:    inference.ModelIdentity{Architecture: "gemma"},
	})

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "model architecture mismatch")
}

func TestStateSession_Bad_WakeRejectsGemma4ModelSizeMismatch(t *testing.T) {
	sessionModel := gemma4StateModelIdentityForTest("/models/lmstudio-community-gemma-4-e4b-it-6bit", 26, 2304)
	reqModel := gemma4StateModelIdentityForTest("/models/lmstudio-community-gemma-4-e2b-it-6bit", 35, 1536)
	session := NewStateSession(sessionModel, inference.TokenizerIdentity{}, nil)

	_, err := session.WakeState(context.Background(), inference.AgentMemoryWakeRequest{
		Store:    state.NewInMemoryStore(nil),
		EntryURI: "state://entry",
		Model:    reqModel,
	})

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "model Gemma4 size mismatch")
}

func TestStateSession_Good_WakeAllowsMismatchWithSkip(t *testing.T) {
	store := state.NewInMemoryStore(nil)
	seedStateSessionKV(t, store, "state://entry", inference.ModelIdentity{Hash: "model-a"}, inference.TokenizerIdentity{})
	session := NewStateSession(inference.ModelIdentity{Hash: "model-a"}, inference.TokenizerIdentity{}, nil)

	wake, err := session.WakeState(context.Background(), inference.AgentMemoryWakeRequest{
		Store:                  store,
		EntryURI:               "state://entry",
		Model:                  inference.ModelIdentity{Hash: "model-b"},
		SkipCompatibilityCheck: true,
	})

	core.RequireNoError(t, err)
	core.AssertEqual(t, 3, wake.PrefixTokens)
}

func TestStateSession_Good_WakeStateReturnsClonedLabels(t *testing.T) {
	store := state.NewInMemoryStore(nil)
	seedStateSessionKV(t, store, "state://entry", inference.ModelIdentity{}, inference.TokenizerIdentity{})
	sessionLabels := map[string]string{"tenant": "a"}
	requestLabels := map[string]string{"request": "wake"}
	session := NewStateSession(inference.ModelIdentity{}, inference.TokenizerIdentity{}, sessionLabels)
	sessionLabels["tenant"] = "mutated"

	wake, err := session.WakeState(context.Background(), inference.AgentMemoryWakeRequest{
		Store:    store,
		EntryURI: "state://entry",
		Labels:   requestLabels,
	})
	core.RequireNoError(t, err)
	requestLabels["request"] = "mutated"

	wake.Labels["tenant"] = "mutated"
	wake.Entry.Labels["request"] = "entry-mutated"
	wake.Bundle.Labels["backend"] = "bundle-mutated"
	second, err := session.WakeState(context.Background(), inference.AgentMemoryWakeRequest{
		Store:    store,
		EntryURI: "state://entry",
		Labels:   map[string]string{"request": "wake"},
	})
	core.RequireNoError(t, err)

	core.AssertEqual(t, "a", second.Labels["tenant"])
	core.AssertEqual(t, "wake", second.Labels["request"])
	core.AssertEqual(t, "rocm", second.Bundle.Labels["backend"])
	core.AssertEqual(t, "wake", second.Entry.Labels["request"])
}

func TestStateSession_Good_IdentityLabelsCloned(t *testing.T) {
	store := state.NewInMemoryStore(nil)
	modelLabels := map[string]string{"model": "source"}
	tokenizerLabels := map[string]string{"tokenizer": "source"}
	session := NewStateSession(
		inference.ModelIdentity{Hash: "model-a", Labels: modelLabels},
		inference.TokenizerIdentity{Hash: "tok-a", Labels: tokenizerLabels},
		nil,
	)
	seedStateSessionKV(t, store, "state://entry", inference.ModelIdentity{Hash: "model-a"}, inference.TokenizerIdentity{Hash: "tok-a"})
	modelLabels["model"] = "mutated"
	tokenizerLabels["tokenizer"] = "mutated"

	core.AssertEqual(t, "source", session.model.Labels["model"])
	core.AssertEqual(t, "source", session.tokenizer.Labels["tokenizer"])

	forked, _, err := session.ForkState(context.Background(), inference.AgentMemoryWakeRequest{
		Store:     store,
		EntryURI:  "state://entry",
		Model:     inference.ModelIdentity{Hash: "model-a"},
		Tokenizer: inference.TokenizerIdentity{Hash: "tok-a"},
	})
	core.RequireNoError(t, err)
	forkedSession, ok := forked.(*StateSession)
	if !ok {
		t.Fatalf("forked session = %T, want *StateSession", forked)
	}
	session.model.Labels["model"] = "parent-mutated"
	session.tokenizer.Labels["tokenizer"] = "parent-mutated"
	forkedSession.model.Labels["model"] = "fork-mutated"
	forkedSession.tokenizer.Labels["tokenizer"] = "fork-mutated"

	core.AssertEqual(t, "parent-mutated", session.model.Labels["model"])
	core.AssertEqual(t, "parent-mutated", session.tokenizer.Labels["tokenizer"])
	core.AssertEqual(t, "fork-mutated", forkedSession.model.Labels["model"])
	core.AssertEqual(t, "fork-mutated", forkedSession.tokenizer.Labels["tokenizer"])
}

func TestStateSession_Good_SleepStateURIFirstJSON(t *testing.T) {
	store := state.NewInMemoryStore(nil)
	session := NewStateSession(inference.ModelIdentity{Hash: "model-a", ContextLength: 256}, inference.TokenizerIdentity{}, nil)

	sleep, err := session.SleepState(context.Background(), inference.AgentMemorySleepRequest{
		Store:    store,
		EntryURI: "state://entry/new",
		Title:    "after",
		Encoding: state.CodecMemory,
		Metadata: map[string]string{"scene": "test"},
	})

	core.AssertNil(t, sleep)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "KV runtime is required")
}

func TestStateSession_Good_SleepStateWritesMergedPlaceholderTags(t *testing.T) {
	store := &recordingStateWriter{}
	session := NewStateSession(inference.ModelIdentity{ContextLength: 128}, inference.TokenizerIdentity{}, map[string]string{"tenant": "a"})

	sleep, err := session.SleepState(context.Background(), inference.AgentMemorySleepRequest{
		Store:    store,
		EntryURI: "state://entry/tags",
		Metadata: map[string]string{"scene": "test"},
		Labels:   map[string]string{"request": "one"},
	})

	core.AssertNil(t, sleep)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "KV runtime is required")
	core.AssertEqual(t, 0, store.putCalls)
}

func TestStateSession_Bad_SleepStateRequiresStore(t *testing.T) {
	session := NewStateSession(inference.ModelIdentity{ContextLength: 128}, inference.TokenizerIdentity{}, nil)

	sleep, err := session.SleepState(context.Background(), inference.AgentMemorySleepRequest{EntryURI: "state://entry/missing-store"})

	core.AssertNil(t, sleep)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "rocm.SleepState")
	core.AssertContains(t, err.Error(), "state store is missing")
}

func TestStateSession_Bad_SleepStatePlaceholderRequiresWriter(t *testing.T) {
	session := NewStateSession(inference.ModelIdentity{ContextLength: 128}, inference.TokenizerIdentity{}, nil)

	sleep, err := session.SleepState(context.Background(), inference.AgentMemorySleepRequest{
		Store:    struct{}{},
		EntryURI: "state://entry/not-writer",
	})

	core.AssertNil(t, sleep)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "KV runtime is required")
}

func TestStateSession_Bad_SleepStatePlaceholderWriteFailure(t *testing.T) {
	store := &recordingStateWriter{err: core.NewError("write failed")}
	session := NewStateSession(inference.ModelIdentity{ContextLength: 128}, inference.TokenizerIdentity{}, map[string]string{"tenant": "a"})

	sleep, err := session.SleepState(context.Background(), inference.AgentMemorySleepRequest{
		Store:    store,
		EntryURI: "state://entry/write-failed",
		Metadata: map[string]string{"scene": "test"},
	})

	core.AssertNil(t, sleep)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "KV runtime is required")
	core.AssertEqual(t, 0, store.putCalls)
}

func TestStateSession_Good_SleepStateReturnsClonedLabels(t *testing.T) {
	store := state.NewInMemoryStore(nil)
	sessionLabels := map[string]string{"tenant": "a"}
	requestLabels := map[string]string{"request": "sleep"}
	cache, err := newROCmKVCache(rocmKVCacheModeQ8, defaultROCmStateBlockSize)
	core.RequireNoError(t, err)
	core.RequireNoError(t, cache.AppendVectors(0, 1, 1, []float32{1, 2}, []float32{2, 1}))
	session := newStateSessionWithRuntime(inference.ModelIdentity{ContextLength: 128}, inference.TokenizerIdentity{}, sessionLabels, cache)
	sessionLabels["tenant"] = "mutated"

	sleep, err := session.SleepState(context.Background(), inference.AgentMemorySleepRequest{
		Store:    store,
		EntryURI: "state://entry/one",
		Labels:   requestLabels,
	})
	core.RequireNoError(t, err)
	requestLabels["request"] = "mutated"

	sleep.Labels["tenant"] = "mutated"
	sleep.Entry.Labels["request"] = "entry-mutated"
	sleep.Entry.StateRefs[0].Labels["kv_serialize"] = "ref-mutated"
	sleep.Bundle.Labels["backend"] = "bundle-mutated"
	second, err := session.SleepState(context.Background(), inference.AgentMemorySleepRequest{
		Store:    store,
		EntryURI: "state://entry/two",
		Labels:   map[string]string{"request": "sleep"},
	})
	core.RequireNoError(t, err)

	core.AssertEqual(t, "a", second.Labels["tenant"])
	core.AssertEqual(t, "sleep", second.Labels["request"])
	core.AssertEqual(t, "rocm", second.Bundle.Labels["backend"])
	core.AssertEqual(t, "runtime_owned_blocks", second.Entry.StateRefs[0].Labels["kv_serialize"])
	core.AssertEqual(t, "sleep", second.Entry.Labels["request"])
}

func TestStateSession_Good_SleepStateBundleRefUsesWrittenURI(t *testing.T) {
	store := state.NewInMemoryStore(nil)
	cache, err := newROCmKVCache(rocmKVCacheModeQ8, defaultROCmStateBlockSize)
	core.RequireNoError(t, err)
	core.RequireNoError(t, cache.AppendVectors(0, 1, 1, []float32{1}, []float32{2}))
	session := newStateSessionWithRuntime(inference.ModelIdentity{ContextLength: 128}, inference.TokenizerIdentity{}, nil, cache)

	sleep, err := session.SleepState(context.Background(), inference.AgentMemorySleepRequest{
		Store:     store,
		EntryURI:  "state://entry/written",
		BundleURI: "state://bundle/requested",
	})

	core.RequireNoError(t, err)
	core.AssertEqual(t, "state://entry/written", sleep.Entry.URI)
	core.AssertEqual(t, "state://bundle/requested", sleep.Entry.BundleURI)
	core.AssertEqual(t, "state://bundle/requested", sleep.Bundle.URI)
	_, err = store.ResolveURI(context.Background(), sleep.Bundle.URI)
	core.RequireNoError(t, err)
}

func TestStateSession_Good_SleepStateSerializesRuntimeOwnedKVSnapshot(t *testing.T) {
	store := state.NewInMemoryStore(nil)
	cache, err := newROCmKVCache(rocmKVCacheModeQ8, 2)
	core.RequireNoError(t, err)
	core.RequireNoError(t, cache.Append(0, []float32{1, 2, 3}, []float32{3, 2, 1}))
	session := newStateSessionWithRuntime(inference.ModelIdentity{Hash: "model-a"}, inference.TokenizerIdentity{}, nil, cache)

	sleep, err := session.SleepState(context.Background(), inference.AgentMemorySleepRequest{
		Store:    store,
		EntryURI: "state://entry/kv",
		Model:    inference.ModelIdentity{Hash: "model-a"},
		Encoding: rocmKVSnapshotEncoding,
	})

	core.RequireNoError(t, err)
	core.AssertEqual(t, rocmKVSnapshotEncoding, sleep.Encoding)
	core.AssertEqual(t, "runtime_owned", sleep.Labels["kv_serialize"])
	core.AssertEqual(t, rocmKVCacheModeQ8, sleep.Labels["cache_mode"])
	core.AssertEqual(t, "2", sleep.Labels["kv_cache_block_size"])
	core.AssertEqual(t, "1", sleep.Labels["kv_key_width"])
	core.AssertEqual(t, "1", sleep.Labels["kv_value_width"])
	core.AssertEqual(t, "2", sleep.Labels["kv_pages"])
	core.AssertEqual(t, "3", sleep.Labels["kv_tokens"])
	core.RequireTrue(t, len(sleep.Entry.StateRefs) == 1)
	core.AssertEqual(t, "runtime_owned", sleep.Entry.StateRefs[0].Labels["kv_serialize"])
	core.AssertEqual(t, "2", sleep.Entry.StateRefs[0].Labels["kv_cache_block_size"])
	core.AssertEqual(t, "1", sleep.Bundle.Labels["kv_key_width"])
	core.AssertEqual(t, "1", sleep.Bundle.Labels["kv_value_width"])
	core.AssertNotEmpty(t, sleep.Bundle.Labels["chunk_id"])
	core.AssertEqual(t, 3, sleep.TokenCount)
	core.AssertEqual(t, 2, sleep.BlocksWritten)
	core.AssertGreater(t, sleep.Bundle.SizeBytes, uint64(0))
	chunk, err := store.ResolveURI(context.Background(), sleep.Bundle.URI)
	core.RequireNoError(t, err)
	core.AssertContains(t, string(chunk.Data), rocmKVCacheModeQ8)
}

func TestStateSession_Good_SleepWakeRuntimeOwnedKVBlockBundle(t *testing.T) {
	store := state.NewInMemoryStore(nil)
	cache, err := newROCmKVCache(rocmKVCacheModeQ8, 2)
	core.RequireNoError(t, err)
	core.RequireNoError(t, cache.AppendVectors(0, 2, 2, []float32{1, 0, 0, 1, 2, 3}, []float32{3, 2, 1, 0, -1, -2}))
	session := newStateSessionWithRuntime(inference.ModelIdentity{Hash: "model-a"}, inference.TokenizerIdentity{}, nil, cache)

	sleep, err := session.SleepState(context.Background(), inference.AgentMemorySleepRequest{
		Store:    store,
		EntryURI: "state://entry/kv-blocks",
		Model:    inference.ModelIdentity{Hash: "model-a"},
		Encoding: rocmKVBlockBundleEncoding,
	})

	core.RequireNoError(t, err)
	core.AssertEqual(t, rocmKVBlockBundleEncoding, sleep.Encoding)
	core.AssertEqual(t, "runtime_owned_blocks", sleep.Labels["kv_serialize"])
	core.AssertEqual(t, "state_refs", sleep.Labels["kv_block_bundle"])
	core.AssertEqual(t, "2", sleep.Labels["kv_block_bundle_blocks"])
	core.AssertEqual(t, 3, sleep.TokenCount)
	core.AssertEqual(t, 2, sleep.BlocksWritten)
	core.RequireTrue(t, len(sleep.Entry.StateRefs) == 2)
	core.AssertEqual(t, "kv-block", sleep.Entry.StateRefs[0].Kind)
	core.AssertEqual(t, rocmKVBlockRawEncoding, sleep.Entry.StateRefs[0].Encoding)
	core.AssertEqual(t, "0", sleep.Entry.StateRefs[0].Labels["kv_block_token_start"])
	core.AssertEqual(t, "2", sleep.Entry.StateRefs[1].Labels["kv_block_token_start"])
	chunk, err := store.ResolveURI(context.Background(), sleep.Bundle.URI)
	core.RequireNoError(t, err)
	var manifest rocmKVBlockBundleSnapshot
	core.RequireNoError(t, json.Unmarshal(chunk.Data, &manifest))
	core.AssertEqual(t, rocmKVBlockBundleKind, manifest.Kind)
	core.AssertEqual(t, 2, len(manifest.Blocks))
	core.AssertEqual(t, rocmKVBlockRawEncoding, manifest.Blocks[0].Encoding)
	core.AssertEqual(t, true, manifest.Blocks[0].State.HasFrameOffset)
	_, err = store.ResolveURI(context.Background(), manifest.Blocks[0].URI)
	core.RequireNoError(t, err)

	woken := NewStateSession(inference.ModelIdentity{Hash: "model-a"}, inference.TokenizerIdentity{}, nil)
	wake, err := woken.WakeState(context.Background(), inference.AgentMemoryWakeRequest{
		Store:    store,
		EntryURI: "state://entry/kv-blocks",
		Model:    inference.ModelIdentity{Hash: "model-a"},
	})

	core.RequireNoError(t, err)
	core.AssertEqual(t, rocmKVBlockBundleEncoding, wake.Bundle.Encoding)
	core.AssertEqual(t, "runtime_owned", wake.Labels["kv_restore"])
	core.AssertEqual(t, "block_stream", wake.Labels["kv_restore_path"])
	core.AssertEqual(t, 3, wake.PrefixTokens)
	core.AssertEqual(t, 2, wake.BlocksRead)
	restored, ok := woken.runtime.(*rocmKVCache)
	core.RequireTrue(t, ok)
	keys, values, err := restored.Restore(0, 3)
	core.RequireNoError(t, err)
	assertFloat32SlicesNear(t, []float32{1, 0, 0, 1, 2, 3}, keys, 0.02)
	assertFloat32SlicesNear(t, []float32{3, 2, 1, 0, -1, -2}, values, 0.02)
}

func TestStateSession_Good_SleepStateReusesParentPrefixBlocks(t *testing.T) {
	store := state.NewInMemoryStore(nil)
	model := inference.ModelIdentity{Hash: "model-a"}
	parentCache, err := newROCmKVCache(rocmKVCacheModeQ8, 2)
	core.RequireNoError(t, err)
	core.RequireNoError(t, parentCache.AppendVectors(
		0,
		1,
		1,
		[]float32{1, 2, 3},
		[]float32{3, 2, 1},
	))
	parentSession := newStateSessionWithRuntime(model, inference.TokenizerIdentity{}, nil, parentCache)
	parent, err := parentSession.SleepState(context.Background(), inference.AgentMemorySleepRequest{
		Store:    store,
		EntryURI: "state://entry/reuse-parent",
		Model:    model,
		Encoding: rocmKVBlockBundleEncoding,
	})
	core.RequireNoError(t, err)

	childCache, err := newROCmKVCache(rocmKVCacheModeQ8, 2)
	core.RequireNoError(t, err)
	core.RequireNoError(t, childCache.AppendVectors(
		0,
		1,
		1,
		[]float32{1, 2, 3, 4},
		[]float32{3, 2, 1, 0},
	))
	childSession := newStateSessionWithRuntime(model, inference.TokenizerIdentity{}, nil, childCache)
	child, err := childSession.SleepState(context.Background(), inference.AgentMemorySleepRequest{
		Store:             store,
		EntryURI:          "state://entry/reuse-child",
		ParentEntryURI:    parent.Entry.URI,
		ParentBundleURI:   parent.Bundle.URI,
		ParentIndexURI:    parent.Index.URI,
		ReuseParentPrefix: true,
		Model:             model,
		Encoding:          rocmKVBlockBundleEncoding,
	})
	core.RequireNoError(t, err)
	core.AssertEqual(t, 2, child.BlocksWritten)
	core.AssertEqual(t, 1, child.BlocksReused)
	core.AssertEqual(t, parent.Bundle.URI, child.Parent.BundleURI)

	parentChunk, err := store.ResolveURI(context.Background(), parent.Bundle.URI)
	core.RequireNoError(t, err)
	childChunk, err := store.ResolveURI(context.Background(), child.Bundle.URI)
	core.RequireNoError(t, err)
	var parentManifest rocmKVBlockBundleSnapshot
	var childManifest rocmKVBlockBundleSnapshot
	core.RequireNoError(t, json.Unmarshal(parentChunk.Data, &parentManifest))
	core.RequireNoError(t, json.Unmarshal(childChunk.Data, &childManifest))
	core.AssertEqual(t, parentManifest.Blocks[0].State.ChunkID, childManifest.Blocks[0].State.ChunkID)
	core.RequireTrue(t, parentManifest.Blocks[1].State.ChunkID != childManifest.Blocks[1].State.ChunkID)

	woken := NewStateSession(model, inference.TokenizerIdentity{}, nil)
	wake, err := woken.WakeState(context.Background(), inference.AgentMemoryWakeRequest{
		Store:    store,
		EntryURI: child.Entry.URI,
		Model:    model,
	})
	core.RequireNoError(t, err)
	core.AssertEqual(t, 4, wake.PrefixTokens)
	cache, ok := woken.runtime.(*rocmKVCache)
	core.RequireTrue(t, ok)
	core.RequireNoError(t, cache.AppendVectors(4, 1, 1, []float32{5}, []float32{-1}))
	keys, values, err := cache.Restore(0, 5)
	core.RequireNoError(t, err)
	assertFloat32SlicesNear(t, []float32{1, 2, 3, 4, 5}, keys, 0.02)
	assertFloat32SlicesNear(t, []float32{3, 2, 1, 0, -1}, values, 0.02)
}

func TestStateSession_Ugly_SleepStateReuseParentPrefixSkipsChangedBlocks(t *testing.T) {
	store := state.NewInMemoryStore(nil)
	model := inference.ModelIdentity{Hash: "model-a"}
	parent := seedStateSessionKV(t, store, "state://entry/reuse-changed-parent", model, inference.TokenizerIdentity{})
	childCache, err := newROCmKVCache(rocmKVCacheModeQ8, defaultROCmStateBlockSize)
	core.RequireNoError(t, err)
	core.RequireNoError(t, childCache.AppendVectors(0, 1, 1, []float32{4, 5, 6}, []float32{6, 5, 4}))
	child, err := newStateSessionWithRuntime(model, inference.TokenizerIdentity{}, nil, childCache).SleepState(context.Background(), inference.AgentMemorySleepRequest{
		Store:             store,
		EntryURI:          "state://entry/reuse-changed-child",
		ParentBundleURI:   parent.Bundle.URI,
		ReuseParentPrefix: true,
		Model:             model,
		Encoding:          rocmKVBlockBundleEncoding,
	})
	core.RequireNoError(t, err)
	core.AssertEqual(t, 0, child.BlocksReused)

	parentChunk, err := store.ResolveURI(context.Background(), parent.Bundle.URI)
	core.RequireNoError(t, err)
	childChunk, err := store.ResolveURI(context.Background(), child.Bundle.URI)
	core.RequireNoError(t, err)
	var parentManifest rocmKVBlockBundleSnapshot
	var childManifest rocmKVBlockBundleSnapshot
	core.RequireNoError(t, json.Unmarshal(parentChunk.Data, &parentManifest))
	core.RequireNoError(t, json.Unmarshal(childChunk.Data, &childManifest))
	core.RequireTrue(t, parentManifest.Blocks[0].State.ChunkID != childManifest.Blocks[0].State.ChunkID)

	woken := NewStateSession(model, inference.TokenizerIdentity{}, nil)
	_, err = woken.WakeState(context.Background(), inference.AgentMemoryWakeRequest{Store: store, EntryURI: child.Entry.URI, Model: model})
	core.RequireNoError(t, err)
	cache, ok := woken.runtime.(*rocmKVCache)
	core.RequireTrue(t, ok)
	keys, values, err := cache.Restore(0, 3)
	core.RequireNoError(t, err)
	assertFloat32SlicesNear(t, []float32{4, 5, 6}, keys, 0.02)
	assertFloat32SlicesNear(t, []float32{6, 5, 4}, values, 0.02)
}

func TestStateSession_Good_Gemma4Q6ProductionLabelsSurviveSleepWake(t *testing.T) {
	store := state.NewInMemoryStore(nil)
	model := inference.ModelIdentity{
		Architecture: "gemma4_text",
		Path:         ProductionLaneCurrentModelID,
		NumLayers:    productionLaneGemma4E2BLayers,
		HiddenSize:   productionLaneGemma4E2BHiddenSize,
		VocabSize:    productionLaneGemma4E2BVocabSize,
		QuantBits:    ProductionLaneProductDefaultQuantBits,
		QuantGroup:   64,
	}
	cache, err := newROCmKVCache(rocmKVCacheModeKQ8VQ4, 2)
	core.RequireNoError(t, err)
	core.RequireNoError(t, cache.AppendVectors(0, 1, 1, []float32{1, 0, 0}, []float32{0, 1, 0}))
	sleeping := newStateSessionWithRuntime(model, inference.TokenizerIdentity{}, nil, cache)

	sleep, err := sleeping.SleepState(context.Background(), inference.AgentMemorySleepRequest{
		Store:    store,
		EntryURI: "state://entry/gemma4-q6",
		Model:    model,
		Encoding: rocmKVBlockBundleEncoding,
	})

	core.RequireNoError(t, err)
	core.AssertEqual(t, "gemma4_mlx_affine", sleep.Labels["production_quant_policy"])
	core.AssertEqual(t, "default", sleep.Labels["production_quant_tier"])
	core.AssertEqual(t, ProductionLaneCurrentModelID, sleep.Labels["production_quant_model"])
	core.AssertEqual(t, "100", sleep.Entry.StateRefs[0].Labels["production_quant_min_visible_tokens_per_sec"])

	woken := NewStateSession(model, inference.TokenizerIdentity{}, nil)
	wake, err := woken.WakeState(context.Background(), inference.AgentMemoryWakeRequest{
		Store:    store,
		EntryURI: "state://entry/gemma4-q6",
		Model:    model,
	})

	core.RequireNoError(t, err)
	core.AssertEqual(t, "runtime_owned", wake.Labels["kv_restore"])
	core.AssertEqual(t, "gemma4_mlx_affine", wake.Labels["production_quant_policy"])
	core.AssertEqual(t, "default", wake.Labels["production_quant_tier"])
	core.AssertEqual(t, ProductionLaneCurrentModelID, wake.Entry.Labels["production_quant_model"])
	core.AssertEqual(t, "100", wake.Bundle.Labels["production_quant_min_visible_tokens_per_sec"])
}

func TestStateSession_Good_Gemma4AdapterLabelsSurviveSleepWakeFork(t *testing.T) {
	store := state.NewInMemoryStore(nil)
	model := inference.ModelIdentity{
		Path:         "/models/lmstudio-community-gemma-4-e4b-it-6bit",
		Architecture: "gemma4_text",
		NumLayers:    26,
		HiddenSize:   2304,
		VocabSize:    262144,
	}
	adapter := rocmAdapterIdentityForModel(inference.AdapterIdentity{
		Path:   "domain.safetensors",
		Format: "lora",
		Hash:   "adapter-hash",
	}, model)
	cache, err := newROCmKVCache(rocmKVCacheModeKQ8VQ4, 2)
	core.RequireNoError(t, err)
	core.RequireNoError(t, cache.AppendVectors(0, 1, 1, []float32{1, 0, 0}, []float32{0, 1, 0}))
	sleeping := newStateSessionWithRuntime(model, inference.TokenizerIdentity{}, nil, cache)

	sleep, err := sleeping.SleepState(context.Background(), inference.AgentMemorySleepRequest{
		Store:    store,
		EntryURI: "state://entry/gemma4-lora",
		Model:    model,
		Adapter:  adapter,
		Encoding: rocmKVBlockBundleEncoding,
	})

	core.RequireNoError(t, err)
	core.AssertEqual(t, "metadata_only", sleep.Labels["state_adapter"])
	core.AssertEqual(t, "E4B", sleep.Labels["adapter_base_gemma4_size"])
	core.AssertEqual(t, "q6", sleep.Labels["adapter_base_gemma4_quant_mode"])
	core.AssertEqual(t, "64", sleep.Labels["adapter_base_gemma4_quant_group"])
	core.AssertEqual(t, Gemma4RuntimeMLXAffine, sleep.Entry.StateRefs[0].Labels["adapter_base_gemma4_runtime"])
	core.AssertEqual(t, Gemma4GenerateLinked, sleep.Bundle.Labels["adapter_base_gemma4_generate_status"])

	woken := NewStateSession(model, inference.TokenizerIdentity{}, nil)
	wake, err := woken.WakeState(context.Background(), inference.AgentMemoryWakeRequest{
		Store:    store,
		EntryURI: sleep.Bundle.URI,
		Model:    model,
	})

	core.RequireNoError(t, err)
	core.AssertEqual(t, "runtime_owned", wake.Labels["kv_restore"])
	core.AssertEqual(t, "E4B", wake.Labels["adapter_base_gemma4_size"])
	core.AssertEqual(t, "q6", wake.Entry.Labels["adapter_base_gemma4_quant_mode"])
	core.AssertEqual(t, "64", wake.Entry.Labels["adapter_base_gemma4_quant_group"])
	core.AssertEqual(t, Gemma4GenerateLinked, wake.Bundle.Labels["adapter_base_gemma4_generate_status"])

	forked, forkWake, err := NewStateSession(model, inference.TokenizerIdentity{}, nil).ForkState(context.Background(), inference.AgentMemoryWakeRequest{
		Store:    store,
		IndexURI: sleep.Index.URI,
		EntryURI: sleep.Entry.URI,
		Model:    model,
	})
	core.RequireNoError(t, err)
	core.AssertEqual(t, "true", forkWake.Labels["fork"])
	core.AssertEqual(t, "E4B", forkWake.Labels["adapter_base_gemma4_size"])
	core.AssertEqual(t, "q6", forkWake.Bundle.Labels["adapter_base_gemma4_quant_mode"])
	core.AssertEqual(t, "64", forkWake.Bundle.Labels["adapter_base_gemma4_quant_group"])
	forkedSession, ok := forked.(*StateSession)
	core.RequireTrue(t, ok)
	_, ok = forkedSession.runtime.(*rocmKVCache)
	core.RequireTrue(t, ok)
}

func TestStateSession_Good_WakeKVBlockBundleBorrowsChunkRefs(t *testing.T) {
	store := &borrowRecordingStateStore{InMemoryStore: state.NewInMemoryStore(nil)}
	cache, err := newROCmKVCache(rocmKVCacheModeKQ8VQ4, 2)
	core.RequireNoError(t, err)
	core.RequireNoError(t, cache.AppendVectors(
		0,
		2,
		3,
		[]float32{1, 0, 0, 1},
		[]float32{3, 2, 1, 0, -1, -2},
	))
	session := newStateSessionWithRuntime(inference.ModelIdentity{Hash: "model-a"}, inference.TokenizerIdentity{}, nil, cache)
	sleep, err := session.SleepState(context.Background(), inference.AgentMemorySleepRequest{
		Store:    store,
		EntryURI: "state://entry/kv-borrow",
		Model:    inference.ModelIdentity{Hash: "model-a"},
		Encoding: rocmKVBlockBundleEncoding,
	})
	core.RequireNoError(t, err)
	chunk, err := store.ResolveURI(context.Background(), sleep.Bundle.URI)
	core.RequireNoError(t, err)
	var manifest rocmKVBlockBundleSnapshot
	core.RequireNoError(t, json.Unmarshal(chunk.Data, &manifest))

	woken := NewStateSession(inference.ModelIdentity{Hash: "model-a"}, inference.TokenizerIdentity{}, nil)
	wake, err := woken.WakeState(context.Background(), inference.AgentMemoryWakeRequest{
		Store:    store,
		EntryURI: "state://entry/kv-borrow",
		Model:    inference.ModelIdentity{Hash: "model-a"},
	})

	core.RequireNoError(t, err)
	core.AssertEqual(t, "block_stream", wake.Labels["kv_restore_path"])
	core.AssertEqual(t, len(manifest.Blocks), len(store.borrowRefs))
	core.AssertEqual(t, manifest.Blocks[0].State.ChunkID, store.borrowRefs[0].ChunkID)
	core.AssertEqual(t, true, store.borrowRefs[0].HasFrameOffset)
}

func TestStateSession_Good_WakeKVBlockBundleRetainsReleasedRawBytes(t *testing.T) {
	store := &releasingBorrowStateStore{InMemoryStore: state.NewInMemoryStore(nil)}
	cache, err := newROCmKVCache(rocmKVCacheModeKQ8VQ4, 2)
	core.RequireNoError(t, err)
	core.RequireNoError(t, cache.AppendVectors(
		0,
		2,
		2,
		[]float32{1, 0, 0, 1},
		[]float32{0.75, -0.5, 0.25, 1},
	))
	session := newStateSessionWithRuntime(inference.ModelIdentity{Hash: "model-a"}, inference.TokenizerIdentity{}, nil, cache)
	_, err = session.SleepState(context.Background(), inference.AgentMemorySleepRequest{
		Store:    store,
		EntryURI: "state://entry/kv-release",
		Model:    inference.ModelIdentity{Hash: "model-a"},
		Encoding: rocmKVBlockBundleEncoding,
	})
	core.RequireNoError(t, err)

	woken := NewStateSession(inference.ModelIdentity{Hash: "model-a"}, inference.TokenizerIdentity{}, nil)
	wake, err := woken.WakeState(context.Background(), inference.AgentMemoryWakeRequest{
		Store:    store,
		EntryURI: "state://entry/kv-release",
		Model:    inference.ModelIdentity{Hash: "model-a"},
	})

	core.RequireNoError(t, err)
	core.AssertEqual(t, "block_stream", wake.Labels["kv_restore_path"])
	core.AssertEqual(t, 1, store.releaseCalls)
	restored, ok := woken.runtime.(*rocmKVCache)
	core.RequireTrue(t, ok)
	keys, values, err := restored.Restore(0, 2)
	core.RequireNoError(t, err)
	assertFloat32SlicesNear(t, []float32{1, 0, 0, 1}, keys, 0.02)
	assertFloat32SlicesNear(t, []float32{0.75, -0.5, 0.25, 1}, values, 0.15)
}

func BenchmarkStateSessionWakeKVBlockBundlePrefixTrim_KQ8VQ4Page(b *testing.B) {
	store := state.NewInMemoryStore(nil)
	keys, values := benchmarkROCmKVVectors(512, 128, 128)
	cache, err := newROCmKVCache(rocmKVCacheModeKQ8VQ4, 512)
	if err != nil {
		b.Fatalf("create KV cache: %v", err)
	}
	if err := cache.AppendVectors(0, 128, 128, keys, values); err != nil {
		b.Fatalf("append KV cache vectors: %v", err)
	}
	session := newStateSessionWithRuntime(inference.ModelIdentity{Hash: "model-a"}, inference.TokenizerIdentity{}, nil, cache)
	sleep, err := session.SleepState(context.Background(), inference.AgentMemorySleepRequest{
		Store:    store,
		EntryURI: "state://entry/kv-prefix-bench",
		Model:    inference.ModelIdentity{Hash: "model-a"},
		Encoding: rocmKVBlockBundleEncoding,
	})
	if err != nil {
		b.Fatalf("sleep KV block bundle: %v", err)
	}
	chunk, err := store.ResolveURI(context.Background(), sleep.Bundle.URI)
	if err != nil {
		b.Fatalf("resolve KV block bundle: %v", err)
	}

	b.SetBytes(int64(384 * 128 * 2 * 4))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		woken, ok, err := wakeKVCacheBlockBundleFromChunk(context.Background(), store, chunk, 384)
		if err != nil {
			b.Fatalf("wake KV block bundle prefix: %v", err)
		}
		if !ok || woken.TokenCount() != 384 || woken.PageCount() != 1 {
			b.Fatalf("woken prefix ok=%v tokens=%d pages=%d, want true/384/1", ok, woken.TokenCount(), woken.PageCount())
		}
	}
}

func BenchmarkStateSessionWakeKVJSONBlockBundlePrefixTrim_KQ8VQ4Page(b *testing.B) {
	store := state.NewInMemoryStore(nil)
	keys, values := benchmarkROCmKVVectors(512, 128, 128)
	cache, err := newROCmKVCache(rocmKVCacheModeKQ8VQ4, 512)
	if err != nil {
		b.Fatalf("create KV cache: %v", err)
	}
	if err := cache.AppendVectors(0, 128, 128, keys, values); err != nil {
		b.Fatalf("append KV cache vectors: %v", err)
	}
	blockPayload, err := cache.snapshotBlock(cache.blocks[0])
	if err != nil {
		b.Fatalf("snapshot KV block: %v", err)
	}
	blockURI := "state://entry/kv-json-prefix-bench/block/0"
	blockRef, err := store.PutBytes(context.Background(), blockPayload, state.PutOptions{
		URI:   blockURI,
		Kind:  "kv-block",
		Track: rocmKVSnapshotEncoding,
	})
	if err != nil {
		b.Fatalf("write KV block: %v", err)
	}
	manifest := rocmKVBlockBundleSnapshot{
		Version:    1,
		Kind:       rocmKVBlockBundleKind,
		Mode:       rocmKVCacheModeKQ8VQ4,
		BlockSize:  512,
		TokenCount: 512,
		Blocks: []rocmKVBlockBundleRef{{
			Index:      0,
			URI:        blockURI,
			ChunkID:    blockRef.ChunkID,
			State:      blockRef,
			TokenStart: 0,
			TokenCount: 512,
			KeyWidth:   128,
			ValueWidth: 128,
			SizeBytes:  uint64(len(blockPayload)),
			Encoding:   rocmKVSnapshotEncoding,
		}},
	}
	manifestPayload, err := json.Marshal(manifest)
	if err != nil {
		b.Fatalf("marshal KV block bundle: %v", err)
	}
	chunk := state.Chunk{Data: manifestPayload}

	b.SetBytes(int64(384 * 128 * 2 * 4))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		woken, ok, err := wakeKVCacheBlockBundleFromChunk(context.Background(), store, chunk, 384)
		if err != nil {
			b.Fatalf("wake JSON KV block bundle prefix: %v", err)
		}
		if !ok || woken.TokenCount() != 384 || woken.PageCount() != 1 {
			b.Fatalf("woken prefix ok=%v tokens=%d pages=%d, want true/384/1", ok, woken.TokenCount(), woken.PageCount())
		}
	}
}

func TestStateSession_Bad_SleepStateRuntimeOwnedKVWriteFailureKeepsRuntime(t *testing.T) {
	store := &failingStateBinaryWriter{err: core.NewError("write failed")}
	cache, err := newROCmKVCache(rocmKVCacheModeQ8, 2)
	core.RequireNoError(t, err)
	core.RequireNoError(t, cache.AppendVectors(0, 2, 2, []float32{1, 0, 0, 1}, []float32{2, 0, 0, 2}))
	session := newStateSessionWithRuntime(inference.ModelIdentity{Hash: "model-a"}, inference.TokenizerIdentity{}, nil, cache)

	sleep, err := session.SleepState(context.Background(), inference.AgentMemorySleepRequest{
		Store:    store,
		EntryURI: "state://entry/kv-write-failed",
		Model:    inference.ModelIdentity{Hash: "model-a"},
		Encoding: rocmKVSnapshotEncoding,
	})

	core.AssertNil(t, sleep)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "write KV state ref")
	core.AssertContains(t, err.Error(), "write failed")
	core.AssertEqual(t, 1, store.putBytesCalls)
	core.AssertEqual(t, "rocm-kv-state", store.options.Kind)
	core.AssertEqual(t, rocmKVCacheModeQ8, store.options.Track)
	if session.runtime != cache {
		t.Fatal("SleepState replaced package-local KV runtime after write failure")
	}
}

func TestStateSession_Bad_SleepStateRuntimeOwnedKVRequiresBinaryWriter(t *testing.T) {
	store := &recordingStateWriter{}
	cache, err := newROCmKVCache(rocmKVCacheModeQ8, 2)
	core.RequireNoError(t, err)
	core.RequireNoError(t, cache.AppendVectors(0, 2, 2, []float32{1, 0, 0, 1}, []float32{2, 0, 0, 2}))
	session := newStateSessionWithRuntime(inference.ModelIdentity{Hash: "model-a"}, inference.TokenizerIdentity{}, nil, cache)

	sleep, err := session.SleepState(context.Background(), inference.AgentMemorySleepRequest{
		Store:    store,
		EntryURI: "state://entry/kv-binary-missing",
		Model:    inference.ModelIdentity{Hash: "model-a"},
	})

	core.AssertNil(t, sleep)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "binary state store is missing")
	core.AssertEqual(t, "", store.text)
	if session.runtime != cache {
		t.Fatal("SleepState replaced package-local KV runtime after missing binary writer")
	}
}

func TestStateSession_Good_SleepStateSerializesHIPDeviceKVSnapshot(t *testing.T) {
	store := state.NewInMemoryStore(nil)
	cache, err := newROCmKVCache(rocmKVCacheModeKQ8VQ4, 2)
	core.RequireNoError(t, err)
	core.RequireNoError(t, cache.AppendVectors(
		0,
		2,
		3,
		[]float32{1, 0.5, -1, 0},
		[]float32{0.75, -0.5, 0.25, 1, -1, 0.5},
	))
	device, err := cache.MirrorToDevice(&fakeHIPDriver{available: true})
	core.RequireNoError(t, err)
	defer device.Close()
	session := newStateSessionWithRuntime(inference.ModelIdentity{Hash: "model-a"}, inference.TokenizerIdentity{}, nil, device)

	sleep, err := session.SleepState(context.Background(), inference.AgentMemorySleepRequest{
		Store:    store,
		EntryURI: "state://entry/device-kv",
		Model:    inference.ModelIdentity{Hash: "model-a"},
		Encoding: rocmKVSnapshotEncoding,
	})

	core.RequireNoError(t, err)
	core.AssertEqual(t, rocmKVSnapshotEncoding, sleep.Encoding)
	core.AssertEqual(t, "device_mirror", sleep.Labels["kv_serialize"])
	core.AssertEqual(t, "hip_device_mirror", sleep.Labels["kv_backing"])
	core.AssertEqual(t, "mirrored", sleep.Labels["kv_device_backing"])
	core.AssertEqual(t, rocmKVCacheModeKQ8VQ4, sleep.Labels["cache_mode"])
	core.AssertEqual(t, "2", sleep.Labels["kv_key_width"])
	core.AssertEqual(t, "3", sleep.Labels["kv_value_width"])
	core.AssertEqual(t, "1", sleep.Labels["kv_pages"])
	core.AssertEqual(t, "2", sleep.Labels["kv_tokens"])
	core.AssertEqual(t, 2, sleep.TokenCount)
	core.AssertEqual(t, 1, sleep.BlocksWritten)
	core.AssertGreater(t, sleep.Bundle.SizeBytes, uint64(0))
	chunk, err := store.ResolveURI(context.Background(), sleep.Bundle.URI)
	core.RequireNoError(t, err)
	restored, err := newROCmKVCacheFromSnapshot(chunk.Data)
	core.RequireNoError(t, err)
	keys, values, err := restored.Restore(0, 2)
	core.RequireNoError(t, err)
	assertFloat32SlicesNear(t, []float32{1, 0.5, -1, 0}, keys, 0.01)
	assertFloat32SlicesNear(t, []float32{0.75, -0.5, 0.25, 1, -1, 0.5}, values, 0.15)
}

func TestStateSession_Good_SleepWakeGemma4Q4DeviceStateBundle(t *testing.T) {
	store := state.NewInMemoryStore(nil)
	driver := &fakeHIPDriver{available: true}
	runtime := hipNewGemma4Q4DeviceDecodeState(rocmKVCacheModeKQ8VQ4, 2)
	for layerIndex := 0; layerIndex < 2; layerIndex++ {
		cache, err := newROCmKVCache(rocmKVCacheModeKQ8VQ4, 2)
		core.RequireNoError(t, err)
		offset := float32(layerIndex)
		core.RequireNoError(t, cache.AppendVectors(
			0,
			2,
			2,
			[]float32{1 + offset, 0, 0, 1 + offset},
			[]float32{0.75 + offset, -0.5, 0.25, 1 + offset},
		))
		device, err := cache.MirrorToDevice(driver)
		core.RequireNoError(t, err)
		table, err := device.kernelDescriptorTableLabeled("rocm.StateSession.Gemma4Q4", "test_roundtrip")
		core.RequireNoError(t, err)
		launch, err := device.KernelLaunchDescriptor(table)
		core.RequireNoError(t, err)
		runtime.layers = append(runtime.layers, hipGemma4Q4DeviceLayerKVState{cache: device, descriptorTable: table, launch: launch})
	}
	model := inference.ModelIdentity{Architecture: "gemma4_text", QuantBits: 4, Labels: map[string]string{"gemma4_size": "E2B"}}
	session := newStateSessionWithRuntime(model, inference.TokenizerIdentity{}, nil, runtime)
	defer session.Close()

	sleep, err := session.SleepState(context.Background(), inference.AgentMemorySleepRequest{
		Store:    store,
		EntryURI: "state://entry/gemma4-q4",
	})
	core.RequireNoError(t, err)
	core.AssertEqual(t, rocmGemma4Q4StateBundleEncoding, sleep.Encoding)
	core.AssertEqual(t, "layer_block_bundles", sleep.Labels["gemma4_q4_state_bundle"])
	core.AssertEqual(t, 2, sleep.TokenCount)
	core.AssertEqual(t, 2, sleep.BlocksWritten)

	woken := NewStateSession(model, inference.TokenizerIdentity{}, nil)
	defer woken.Close()
	wake, err := woken.WakeState(context.Background(), inference.AgentMemoryWakeRequest{
		Store:    store,
		EntryURI: "state://entry/gemma4-q4",
	})
	core.RequireNoError(t, err)
	core.AssertEqual(t, rocmGemma4Q4StateBundleEncoding, wake.Bundle.Encoding)
	core.AssertEqual(t, "gemma4_q4_layer_block_stream", wake.Labels["kv_restore_path"])
	restored, ok := woken.runtime.(*hipGemma4Q4HostDecodeStateRuntime)
	core.RequireTrue(t, ok)
	core.AssertEqual(t, 2, restored.tokenCount)
	core.AssertEqual(t, 2, len(restored.state.Layers))
	assertFloat32SlicesNear(t, []float32{1, 0, 0, 1}, restored.state.Layers[0].Keys, 0.02)
	assertFloat32SlicesNear(t, []float32{0.75, -0.5, 0.25, 1}, restored.state.Layers[0].Values, 0.15)
	assertFloat32SlicesNear(t, []float32{2, 0, 0, 2}, restored.state.Layers[1].Keys, 0.02)
	assertFloat32SlicesNear(t, []float32{1.75, -0.5, 0.25, 2}, restored.state.Layers[1].Values, 0.15)
}

func TestStateSession_Bad_SleepStateDeviceKVWriteFailureKeepsRuntime(t *testing.T) {
	store := &failingStateBinaryWriter{err: core.NewError("write failed")}
	cache, err := newROCmKVCache(rocmKVCacheModeQ8, 2)
	core.RequireNoError(t, err)
	core.RequireNoError(t, cache.AppendVectors(0, 2, 2, []float32{1, 0, 0, 1}, []float32{2, 0, 0, 2}))
	device, err := cache.MirrorToDevice(&fakeHIPDriver{available: true})
	core.RequireNoError(t, err)
	defer device.Close()
	session := newStateSessionWithRuntime(inference.ModelIdentity{Hash: "model-a"}, inference.TokenizerIdentity{}, nil, device)

	sleep, err := session.SleepState(context.Background(), inference.AgentMemorySleepRequest{
		Store:    store,
		EntryURI: "state://entry/device-kv-write-failed",
		Model:    inference.ModelIdentity{Hash: "model-a"},
		Encoding: rocmKVSnapshotEncoding,
	})

	core.AssertNil(t, sleep)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "write HIP device KV state ref")
	core.AssertContains(t, err.Error(), "write failed")
	core.AssertEqual(t, 1, store.putBytesCalls)
	core.AssertEqual(t, "rocm-hip-kv-state", store.options.Kind)
	core.AssertEqual(t, rocmKVCacheModeQ8, store.options.Track)
	if session.runtime != device {
		t.Fatal("SleepState replaced HIP device KV runtime after write failure")
	}
}

func TestStateSession_Bad_SleepStateDeviceKVRequiresBinaryWriter(t *testing.T) {
	store := &recordingStateWriter{}
	cache, err := newROCmKVCache(rocmKVCacheModeQ8, 2)
	core.RequireNoError(t, err)
	core.RequireNoError(t, cache.AppendVectors(0, 2, 2, []float32{1, 0, 0, 1}, []float32{2, 0, 0, 2}))
	device, err := cache.MirrorToDevice(&fakeHIPDriver{available: true})
	core.RequireNoError(t, err)
	defer device.Close()
	session := newStateSessionWithRuntime(inference.ModelIdentity{Hash: "model-a"}, inference.TokenizerIdentity{}, nil, device)

	sleep, err := session.SleepState(context.Background(), inference.AgentMemorySleepRequest{
		Store:    store,
		EntryURI: "state://entry/device-kv-binary-missing",
		Model:    inference.ModelIdentity{Hash: "model-a"},
	})

	core.AssertNil(t, sleep)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "binary state store is missing")
	core.AssertEqual(t, "", store.text)
	if session.runtime != device {
		t.Fatal("SleepState replaced HIP device KV runtime after missing binary writer")
	}
}

func TestStateSession_Bad_SleepStateDeviceKVSnapshotFailureDoesNotWriteStateRef(t *testing.T) {
	store := state.NewInMemoryStore(nil)
	cache, err := newROCmKVCache(rocmKVCacheModeQ8, 2)
	core.RequireNoError(t, err)
	core.RequireNoError(t, cache.AppendVectors(0, 2, 2, []float32{1, 0, 0, 1}, []float32{2, 0, 0, 2}))
	driver := &fakeHIPDriver{available: true}
	device, err := cache.MirrorToDevice(driver)
	core.RequireNoError(t, err)
	defer device.Close()
	driver.copyErr = core.NewError("device read failed")
	driver.copyErrAt = len(driver.copies) + 1
	session := newStateSessionWithRuntime(inference.ModelIdentity{Hash: "model-a"}, inference.TokenizerIdentity{}, nil, device)

	sleep, err := session.SleepState(context.Background(), inference.AgentMemorySleepRequest{
		Store:    store,
		EntryURI: "state://entry/device-kv-failed",
		Model:    inference.ModelIdentity{Hash: "model-a"},
	})

	core.AssertNil(t, sleep)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "snapshot HIP device KV cache")
	core.AssertContains(t, err.Error(), "copy KV key page")
	core.AssertContains(t, err.Error(), "device read failed")
	if session.runtime != device {
		t.Fatal("SleepState replaced device runtime after snapshot failure")
	}
	_, resolveErr := store.ResolveURI(context.Background(), "state://entry/device-kv-failed")
	core.AssertError(t, resolveErr)
}

func TestStateSession_Good_WakeStateRestoresHIPDeviceKVSnapshotAsPackageLocal(t *testing.T) {
	store := state.NewInMemoryStore(nil)
	cache, err := newROCmKVCache(rocmKVCacheModeKQ8VQ4, 2)
	core.RequireNoError(t, err)
	core.RequireNoError(t, cache.AppendVectors(
		0,
		2,
		3,
		[]float32{1, 0.5, -1, 0},
		[]float32{0.75, -0.5, 0.25, 1, -1, 0.5},
	))
	device, err := cache.MirrorToDevice(&fakeHIPDriver{available: true})
	core.RequireNoError(t, err)
	defer device.Close()
	sleeping := newStateSessionWithRuntime(inference.ModelIdentity{Hash: "model-a"}, inference.TokenizerIdentity{}, nil, device)
	_, err = sleeping.SleepState(context.Background(), inference.AgentMemorySleepRequest{
		Store:    store,
		EntryURI: "state://entry/device-kv",
		Model:    inference.ModelIdentity{Hash: "model-a"},
		Encoding: rocmKVSnapshotEncoding,
	})
	core.RequireNoError(t, err)
	waking := NewStateSession(inference.ModelIdentity{Hash: "model-a"}, inference.TokenizerIdentity{}, nil)

	wake, err := waking.WakeState(context.Background(), inference.AgentMemoryWakeRequest{
		Store:    store,
		EntryURI: "state://entry/device-kv",
		Model:    inference.ModelIdentity{Hash: "model-a"},
	})

	core.RequireNoError(t, err)
	core.AssertEqual(t, rocmKVSnapshotEncoding, wake.Bundle.Encoding)
	core.AssertEqual(t, "runtime_owned", wake.Labels["kv_restore"])
	core.AssertEqual(t, "package_local", wake.Labels["kv_backing"])
	core.AssertEqual(t, "planned", wake.Labels["kv_device_backing"])
	core.AssertEqual(t, rocmKVCacheModeKQ8VQ4, wake.Labels["cache_mode"])
	core.AssertEqual(t, "2", wake.Labels["kv_key_width"])
	core.AssertEqual(t, "3", wake.Labels["kv_value_width"])
	restored, ok := waking.runtime.(*rocmKVCache)
	core.RequireTrue(t, ok)
	keys, values, err := restored.Restore(0, 2)
	core.RequireNoError(t, err)
	assertFloat32SlicesNear(t, []float32{1, 0.5, -1, 0}, keys, 0.01)
	assertFloat32SlicesNear(t, []float32{0.75, -0.5, 0.25, 1, -1, 0.5}, values, 0.15)
}

func TestStateSession_Good_WakeStateRestoresRuntimeOwnedKVSnapshot(t *testing.T) {
	store := state.NewInMemoryStore(nil)
	cache, err := newROCmKVCache(rocmKVCacheModeKQ8VQ4, 2)
	core.RequireNoError(t, err)
	core.RequireNoError(t, cache.AppendVectors(
		0,
		2,
		3,
		[]float32{1, 0.5, -1, 0},
		[]float32{0.75, -0.5, 0.25, 1, -1, 0.5},
	))
	sleeping := newStateSessionWithRuntime(inference.ModelIdentity{Hash: "model-a"}, inference.TokenizerIdentity{}, nil, cache)
	_, err = sleeping.SleepState(context.Background(), inference.AgentMemorySleepRequest{
		Store:    store,
		EntryURI: "state://entry/kv",
		Model:    inference.ModelIdentity{Hash: "model-a"},
		Encoding: rocmKVSnapshotEncoding,
	})
	core.RequireNoError(t, err)
	waking := NewStateSession(inference.ModelIdentity{Hash: "model-a"}, inference.TokenizerIdentity{}, nil)

	wake, err := waking.WakeState(context.Background(), inference.AgentMemoryWakeRequest{
		Store:    store,
		EntryURI: "state://entry/kv",
		Model:    inference.ModelIdentity{Hash: "model-a"},
	})

	core.RequireNoError(t, err)
	core.AssertEqual(t, rocmKVSnapshotEncoding, wake.Bundle.Encoding)
	core.AssertEqual(t, "runtime_owned", wake.Labels["kv_restore"])
	core.AssertEqual(t, rocmKVCacheModeKQ8VQ4, wake.Labels["cache_mode"])
	core.AssertEqual(t, "2", wake.Labels["kv_cache_block_size"])
	core.AssertEqual(t, "2", wake.Labels["kv_key_width"])
	core.AssertEqual(t, "3", wake.Labels["kv_value_width"])
	core.AssertEqual(t, "1", wake.Labels["kv_pages"])
	core.AssertEqual(t, "2", wake.Labels["kv_tokens"])
	core.AssertEqual(t, 2, wake.PrefixTokens)
	core.AssertEqual(t, 1, wake.BlocksRead)
	restored, ok := waking.runtime.(*rocmKVCache)
	core.RequireTrue(t, ok)
	keys, values, err := restored.Restore(0, 2)
	core.RequireNoError(t, err)
	assertFloat32SlicesNear(t, []float32{1, 0.5, -1, 0}, keys, 0.01)
	assertFloat32SlicesNear(t, []float32{0.75, -0.5, 0.25, 1, -1, 0.5}, values, 0.15)
}

func TestStateSession_Good_WakeStateClosesPreviousRuntime(t *testing.T) {
	store := state.NewInMemoryStore(nil)
	nextCache, err := newROCmKVCache(rocmKVCacheModeQ8, 2)
	core.RequireNoError(t, err)
	core.RequireNoError(t, nextCache.AppendVectors(0, 2, 2, []float32{1, 0, 0, 1}, []float32{2, 0, 0, 2}))
	nextPayload, err := nextCache.Snapshot()
	core.RequireNoError(t, err)
	_, err = store.PutBytes(context.Background(), nextPayload, state.PutOptions{URI: "state://entry/next-kv"})
	core.RequireNoError(t, err)
	previousCache, err := newROCmKVCache(rocmKVCacheModeQ8, 2)
	core.RequireNoError(t, err)
	core.RequireNoError(t, previousCache.AppendVectors(0, 2, 2, []float32{3, 0, 0, 3}, []float32{4, 0, 0, 4}))
	driver := &fakeHIPDriver{available: true}
	previousDevice, err := previousCache.MirrorToDevice(driver)
	core.RequireNoError(t, err)
	session := newStateSessionWithRuntime(inference.ModelIdentity{}, inference.TokenizerIdentity{}, nil, previousDevice)

	wake, err := session.WakeState(context.Background(), inference.AgentMemoryWakeRequest{Store: store, EntryURI: "state://entry/next-kv"})

	core.RequireNoError(t, err)
	core.AssertEqual(t, "runtime_owned", wake.Labels["kv_restore"])
	core.AssertEqual(t, true, previousDevice.closed)
	if len(driver.frees) == 0 {
		t.Fatal("previous HIP device KV runtime was not freed")
	}
	restored, ok := session.runtime.(*rocmKVCache)
	core.RequireTrue(t, ok)
	core.AssertEqual(t, 2, restored.TokenCount())
}

func TestStateSession_Bad_WakeStateClosePreviousDeviceRuntimeFailureDoesNotInstallSnapshot(t *testing.T) {
	store := state.NewInMemoryStore(nil)
	nextCache, err := newROCmKVCache(rocmKVCacheModeQ8, 2)
	core.RequireNoError(t, err)
	core.RequireNoError(t, nextCache.AppendVectors(0, 2, 2, []float32{1, 0, 0, 1}, []float32{2, 0, 0, 2}))
	nextPayload, err := nextCache.Snapshot()
	core.RequireNoError(t, err)
	_, err = store.PutBytes(context.Background(), nextPayload, state.PutOptions{URI: "state://entry/next-kv"})
	core.RequireNoError(t, err)
	previousCache, err := newROCmKVCache(rocmKVCacheModeQ8, 2)
	core.RequireNoError(t, err)
	core.RequireNoError(t, previousCache.AppendVectors(0, 2, 2, []float32{3, 0, 0, 3}, []float32{4, 0, 0, 4}))
	driver := &failingHIPDriver{available: true, freeErr: core.NewError("free failed")}
	previousDevice, err := previousCache.MirrorToDevice(driver)
	core.RequireNoError(t, err)
	session := newStateSessionWithRuntime(inference.ModelIdentity{}, inference.TokenizerIdentity{}, nil, previousDevice)

	wake, err := session.WakeState(context.Background(), inference.AgentMemoryWakeRequest{Store: store, EntryURI: "state://entry/next-kv"})

	core.AssertError(t, err)
	core.AssertNil(t, wake)
	core.AssertContains(t, err.Error(), "close previous state runtime")
	core.AssertContains(t, err.Error(), "free failed")
	if session.runtime != previousDevice {
		t.Fatal("WakeState installed restored snapshot after previous device runtime close failure")
	}
	core.AssertEqual(t, len(driver.allocations), len(driver.frees))
}

func TestStateSession_Bad_SleepRejectsTokenizerHashMismatch(t *testing.T) {
	store := state.NewInMemoryStore(nil)
	session := NewStateSession(inference.ModelIdentity{}, inference.TokenizerIdentity{Hash: "tok-a"}, nil)

	_, err := session.SleepState(context.Background(), inference.AgentMemorySleepRequest{
		Store:     store,
		EntryURI:  "state://entry/new",
		Tokenizer: inference.TokenizerIdentity{Hash: "tok-b"},
	})

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "tokenizer hash mismatch")
}

func TestStateSession_Bad_SleepRejectsTokenizerKindMismatch(t *testing.T) {
	store := state.NewInMemoryStore(nil)
	session := NewStateSession(inference.ModelIdentity{}, inference.TokenizerIdentity{Kind: "Qwen2Tokenizer"}, nil)

	_, err := session.SleepState(context.Background(), inference.AgentMemorySleepRequest{
		Store:     store,
		EntryURI:  "state://entry/new",
		Tokenizer: inference.TokenizerIdentity{Kind: "GemmaTokenizer"},
	})

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "tokenizer kind mismatch")
}

func TestStateSession_Bad_SleepRejectsModelHashMismatch(t *testing.T) {
	store := state.NewInMemoryStore(nil)
	session := NewStateSession(inference.ModelIdentity{Hash: "model-a"}, inference.TokenizerIdentity{}, nil)

	_, err := session.SleepState(context.Background(), inference.AgentMemorySleepRequest{
		Store:    store,
		EntryURI: "state://entry/new",
		Model:    inference.ModelIdentity{Hash: "model-b"},
	})

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "model hash mismatch")
}

func TestStateSession_Bad_SleepRejectsGemma4ModelQuantMismatch(t *testing.T) {
	store := state.NewInMemoryStore(nil)
	sessionModel := gemma4StateModelIdentityForTest("/models/lmstudio-community-gemma-4-e2b-it-8bit", 35, 1536)
	reqModel := gemma4StateModelIdentityForTest("/models/lmstudio-community-gemma-4-e2b-it-6bit", 35, 1536)
	session := NewStateSession(sessionModel, inference.TokenizerIdentity{}, nil)

	_, err := session.SleepState(context.Background(), inference.AgentMemorySleepRequest{
		Store:    store,
		EntryURI: "state://entry/new",
		Model:    reqModel,
	})

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "model Gemma4 quant mismatch")
}

func TestStateSession_Bad_SleepRejectsGemma4AdapterBaseMismatch(t *testing.T) {
	store := state.NewInMemoryStore(nil)
	model := inference.ModelIdentity{
		Path:         "/models/lmstudio-community-gemma-4-e4b-it-6bit",
		Architecture: "gemma4_text",
		NumLayers:    26,
		HiddenSize:   2304,
		VocabSize:    262144,
	}
	session := NewStateSession(model, inference.TokenizerIdentity{}, nil)

	_, err := session.SleepState(context.Background(), inference.AgentMemorySleepRequest{
		Store:    store,
		EntryURI: "state://entry/new",
		Model:    model,
		Adapter: inference.AdapterIdentity{
			Path:   "domain.safetensors",
			Format: "lora",
			Labels: map[string]string{
				"adapter_base_architecture":      "gemma4_text",
				"adapter_base_gemma4_size":       "E2B",
				"adapter_base_gemma4_quant_mode": "q6",
			},
		},
	})

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "adapter base Gemma4 size mismatch")
}

func TestStateSession_Bad_WakeRejectsGemma4AdapterBaseMismatch(t *testing.T) {
	model := inference.ModelIdentity{
		Path:         "/models/lmstudio-community-gemma-4-e4b-it-6bit",
		Architecture: "gemma4_text",
		NumLayers:    26,
		HiddenSize:   2304,
		VocabSize:    262144,
	}
	session := NewStateSession(model, inference.TokenizerIdentity{}, nil)

	_, err := session.WakeState(context.Background(), inference.AgentMemoryWakeRequest{
		EntryURI: "state://entry/new",
		Model:    model,
		Adapter: inference.AdapterIdentity{
			Path:   "domain.safetensors",
			Format: "lora",
			Labels: map[string]string{
				"adapter_base_architecture":      "gemma4_text",
				"adapter_base_gemma4_size":       "E2B",
				"adapter_base_gemma4_quant_mode": "q6",
			},
		},
	})

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "adapter base Gemma4 size mismatch")
}

func TestStateSession_Bad_WakeRejectsGemma4AdapterBaseQuantGroupMismatch(t *testing.T) {
	model := inference.ModelIdentity{
		Path:         "/models/lmstudio-community-gemma-4-e4b-it-6bit",
		Architecture: "gemma4_text",
		NumLayers:    26,
		HiddenSize:   2304,
		VocabSize:    262144,
	}
	session := NewStateSession(model, inference.TokenizerIdentity{}, nil)

	_, err := session.WakeState(context.Background(), inference.AgentMemoryWakeRequest{
		EntryURI: "state://entry/new",
		Model:    model,
		Adapter: inference.AdapterIdentity{
			Path:   "domain.safetensors",
			Format: "lora",
			Labels: map[string]string{
				"adapter_base_architecture":       "gemma4_text",
				"adapter_base_gemma4_size":        "E4B",
				"adapter_base_gemma4_quant_mode":  "q6",
				"adapter_base_gemma4_quant_group": "32",
			},
		},
	})

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "adapter base Gemma4 quant group mismatch")
}

func TestStateSession_Bad_WakeRejectsIncompleteGemma4AdapterBaseIdentity(t *testing.T) {
	model := inference.ModelIdentity{
		Path:         "/models/lmstudio-community-gemma-4-e4b-it-6bit",
		Architecture: "gemma4_text",
		NumLayers:    26,
		HiddenSize:   2304,
		VocabSize:    262144,
	}
	session := NewStateSession(model, inference.TokenizerIdentity{}, nil)

	_, err := session.WakeState(context.Background(), inference.AgentMemoryWakeRequest{
		EntryURI: "state://entry/new",
		Model:    model,
		Adapter: inference.AdapterIdentity{
			Path:   "domain.safetensors",
			Format: "lora",
			Labels: map[string]string{
				"adapter_base_gemma4_generate_status": Gemma4GenerateLinked,
			},
		},
	})

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "adapter base Gemma4 identity is incomplete")
}

func TestStateSession_Bad_WakeRejectsMalformedKVSnapshot(t *testing.T) {
	store := state.NewInMemoryStore(nil)
	_, err := store.PutBytes(context.Background(), []byte(`{"version":1,"mode":"q8","block_size":2,"blocks":[{"token_start":0,"token_count":1,"key":{"encoding":"q8","length":1,"scale":0,"q8":[1]},"value":{"encoding":"q8","length":1,"scale":1,"q8":[1]}}]}`), state.PutOptions{URI: "state://entry/bad-kv"})
	core.RequireNoError(t, err)
	session := NewStateSession(inference.ModelIdentity{}, inference.TokenizerIdentity{}, nil)

	_, err = session.WakeState(context.Background(), inference.AgentMemoryWakeRequest{Store: store, EntryURI: "state://entry/bad-kv"})

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "restore KV cache snapshot")
	core.AssertContains(t, err.Error(), "q8 scale")
}

func TestStateSession_Good_ForkStateCreatesIndependentSession(t *testing.T) {
	store := state.NewInMemoryStore(nil)
	seedStateSessionKV(t, store, "state://entry", inference.ModelIdentity{}, inference.TokenizerIdentity{})
	session := NewStateSession(inference.ModelIdentity{}, inference.TokenizerIdentity{}, nil)

	forked, wake, err := session.ForkState(context.Background(), inference.AgentMemoryWakeRequest{Store: store, EntryURI: "state://entry"})

	core.RequireNoError(t, err)
	core.AssertNotNil(t, forked)
	core.AssertEqual(t, 3, wake.PrefixTokens)
	if forked == session {
		t.Fatal("forked session aliases parent")
	}
}

func TestStateSession_Good_ForkStateRestoresIndependentRuntimeOwnedKVSnapshot(t *testing.T) {
	store := state.NewInMemoryStore(nil)
	cache, err := newROCmKVCache(rocmKVCacheModeQ8, 2)
	core.RequireNoError(t, err)
	core.RequireNoError(t, cache.AppendVectors(0, 2, 2, []float32{1, 0, 0, 1}, []float32{2, 0, 0, 2}))
	sleeping := newStateSessionWithRuntime(inference.ModelIdentity{Hash: "model-a"}, inference.TokenizerIdentity{}, nil, cache)
	_, err = sleeping.SleepState(context.Background(), inference.AgentMemorySleepRequest{
		Store:    store,
		EntryURI: "state://entry/kv-fork",
		Model:    inference.ModelIdentity{Hash: "model-a"},
	})
	core.RequireNoError(t, err)
	session := NewStateSession(inference.ModelIdentity{Hash: "model-a"}, inference.TokenizerIdentity{}, map[string]string{"tenant": "a"})

	forked, wake, err := session.ForkState(context.Background(), inference.AgentMemoryWakeRequest{
		Store:    store,
		EntryURI: "state://entry/kv-fork",
		Model:    inference.ModelIdentity{Hash: "model-a"},
	})

	core.RequireNoError(t, err)
	core.AssertEqual(t, "true", wake.Labels["fork"])
	core.AssertEqual(t, "a", wake.Labels["tenant"])
	core.AssertEqual(t, "runtime_owned", wake.Labels["kv_restore"])
	core.AssertEqual(t, "2", wake.Labels["kv_key_width"])
	core.AssertEqual(t, "2", wake.Labels["kv_value_width"])
	forkedSession, ok := forked.(*StateSession)
	core.RequireTrue(t, ok)
	forkedCache, ok := forkedSession.runtime.(*rocmKVCache)
	core.RequireTrue(t, ok)
	if forkedCache == cache {
		t.Fatal("forked KV cache aliases source runtime cache")
	}
	core.RequireNoError(t, forkedCache.AppendToken(forkedCache.TokenCount(), []float32{3, 3}, []float32{4, 4}))
	core.AssertEqual(t, 3, forkedCache.TokenCount())
	core.AssertEqual(t, 2, cache.TokenCount())
}

func TestStateSession_Bad_ForkStateRejectsNilSession(t *testing.T) {
	var session *StateSession

	forked, wake, err := session.ForkState(context.Background(), inference.AgentMemoryWakeRequest{})

	core.AssertNil(t, forked)
	core.AssertNil(t, wake)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "rocm.ForkState")
	core.AssertContains(t, err.Error(), "state session is nil")
}

func TestStateSession_Bad_ForkStateWrapsWakeFailure(t *testing.T) {
	session := NewStateSession(inference.ModelIdentity{Hash: "model-a"}, inference.TokenizerIdentity{}, nil)

	forked, wake, err := session.ForkState(context.Background(), inference.AgentMemoryWakeRequest{
		Model: inference.ModelIdentity{Hash: "model-b"},
	})

	core.AssertNil(t, forked)
	core.AssertNil(t, wake)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "rocm.ForkState")
	core.AssertContains(t, err.Error(), "wake forked state")
	core.AssertContains(t, err.Error(), "model hash mismatch")
}

func TestStateSession_Good_RocmModelForkStateRemirrorsKVSnapshotToHIPDevice(t *testing.T) {
	store := state.NewInMemoryStore(nil)
	cache, err := newROCmKVCache(rocmKVCacheModeKQ8VQ4, 2)
	core.RequireNoError(t, err)
	core.RequireNoError(t, cache.AppendVectors(
		0,
		2,
		3,
		[]float32{1, 0.5, -1, 0},
		[]float32{0.75, -0.5, 0.25, 1, -1, 0.5},
	))
	sleeping := newStateSessionWithRuntime(inference.ModelIdentity{Hash: "model-a"}, inference.TokenizerIdentity{}, nil, cache)
	_, err = sleeping.SleepState(context.Background(), inference.AgentMemorySleepRequest{
		Store:    store,
		EntryURI: "state://entry/fork-kv",
		Model:    inference.ModelIdentity{Hash: "model-a"},
	})
	core.RequireNoError(t, err)
	model := &rocmModel{
		modelInfo: inference.ModelInfo{Architecture: "qwen3"},
		native:    &hipLoadedModel{driver: &fakeHIPDriver{available: true}},
	}

	forked, wake, err := model.ForkState(context.Background(), inference.AgentMemoryWakeRequest{
		Store:    store,
		EntryURI: "state://entry/fork-kv",
		Model:    inference.ModelIdentity{Hash: "model-a"},
	})

	core.RequireNoError(t, err)
	core.AssertEqual(t, "device_mirror", wake.Labels["kv_restore"])
	core.AssertEqual(t, "hip_device_mirror", wake.Labels["kv_backing"])
	core.AssertEqual(t, "mirrored", wake.Labels["kv_device_backing"])
	core.AssertEqual(t, "mirrored", wake.Labels["kv_device_restore"])
	forkedSession, ok := forked.(*StateSession)
	core.RequireTrue(t, ok)
	device, ok := forkedSession.runtime.(*rocmDeviceKVCache)
	core.RequireTrue(t, ok)
	defer device.Close()
	core.AssertEqual(t, 2, device.TokenCount())
	if model.state == forkedSession {
		t.Fatal("forked session aliases model state session")
	}
}

func TestStateSession_Good_RocmModelForkStateKeepsPackageLocalKVOnDeviceMirrorFailure(t *testing.T) {
	store := state.NewInMemoryStore(nil)
	cache, err := newROCmKVCache(rocmKVCacheModeQ8, 2)
	core.RequireNoError(t, err)
	core.RequireNoError(t, cache.AppendVectors(0, 2, 2, []float32{1, 0, 0, 1}, []float32{2, 0, 0, 2}))
	sleeping := newStateSessionWithRuntime(inference.ModelIdentity{Hash: "model-a"}, inference.TokenizerIdentity{}, nil, cache)
	_, err = sleeping.SleepState(context.Background(), inference.AgentMemorySleepRequest{
		Store:    store,
		EntryURI: "state://entry/fork-kv",
		Model:    inference.ModelIdentity{Hash: "model-a"},
	})
	core.RequireNoError(t, err)
	driver := &fakeHIPDriver{available: true, copyErr: core.NewError("copy failed"), copyErrAt: 1}
	model := &rocmModel{
		modelInfo: inference.ModelInfo{Architecture: "qwen3"},
		native:    &hipLoadedModel{driver: driver},
	}

	forked, wake, err := model.ForkState(context.Background(), inference.AgentMemoryWakeRequest{
		Store:    store,
		EntryURI: "state://entry/fork-kv",
		Model:    inference.ModelIdentity{Hash: "model-a"},
	})

	core.RequireNoError(t, err)
	core.AssertEqual(t, "runtime_owned", wake.Labels["kv_restore"])
	core.AssertEqual(t, "package_local", wake.Labels["kv_backing"])
	core.AssertEqual(t, "failed", wake.Labels["kv_device_restore"])
	core.AssertContains(t, wake.Labels["kv_device_restore_error"], "copy KV key page")
	forkedSession, ok := forked.(*StateSession)
	core.RequireTrue(t, ok)
	restored, ok := forkedSession.runtime.(*rocmKVCache)
	core.RequireTrue(t, ok)
	core.AssertEqual(t, 2, restored.TokenCount())
}

func TestStateSession_Bad_MissingStoreHasOperationContext(t *testing.T) {
	session := NewStateSession(inference.ModelIdentity{}, inference.TokenizerIdentity{}, nil)

	_, err := session.WakeState(context.Background(), inference.AgentMemoryWakeRequest{EntryURI: "state://missing"})

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "rocm.WakeState")
	core.AssertContains(t, err.Error(), "state store is missing")
}

func TestStateSession_Bad_WakeRequiresEntryOrIndexURI(t *testing.T) {
	session := NewStateSession(inference.ModelIdentity{}, inference.TokenizerIdentity{}, nil)

	_, err := session.WakeState(context.Background(), inference.AgentMemoryWakeRequest{Store: state.NewInMemoryStore(nil)})

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "rocm.WakeState")
	core.AssertContains(t, err.Error(), "entry or index URI is required")
}

func TestStateSession_Good_RocmModelImplementsStateContracts(t *testing.T) {
	var _ inference.AgentMemorySession = (*rocmModel)(nil)
	var _ inference.AgentMemoryForker = (*rocmModel)(nil)
	var _ inference.StatefulModel = (*rocmModel)(nil)
}

func TestStateSession_Good_RocmModelCapturesMetadataStateBundle(t *testing.T) {
	model := &rocmModel{
		modelType:   "qwen3",
		modelInfo:   inference.ModelInfo{Architecture: "qwen3", VocabSize: 32000},
		lastMetrics: inference.GenerateMetrics{GeneratedTokens: 2},
		native: &fakeNativeModel{
			tokens: []inference.Token{{ID: 1, Text: "a"}, {ID: 2, Text: "b"}},
		},
	}

	bundle, err := model.CaptureState(context.Background(), "hello world", inference.WithMaxTokens(8), inference.WithTemperature(0.25), inference.WithStopTokens(2))

	core.RequireNoError(t, err)
	core.AssertEqual(t, "rocm-state-bundle-v1", bundle.Version)
	core.AssertEqual(t, "qwen3", bundle.Model.Architecture)
	core.AssertEqual(t, 8, bundle.Sampler.MaxTokens)
	core.AssertEqual(t, []int32{2}, bundle.Sampler.StopTokens)
	core.AssertEqual(t, 2, bundle.PromptTokens)
	core.AssertEqual(t, 2, bundle.GeneratedTokens)
	core.AssertContains(t, bundle.PromptHash, "sha256:")
	core.AssertEqual(t, "metadata_only", bundle.Labels["state_bundle"])
	core.AssertEqual(t, "use_sleep_state", bundle.Labels["state_bundle_kv_refs"])
}

func TestStateSession_Good_Gemma4CaptureStateUsesRemainingMaxTokens(t *testing.T) {
	model := &rocmModel{
		modelPath: "/models/lmstudio-community-gemma-4-e4b-it-6bit",
		modelType: "gemma4_text",
		modelInfo: inference.ModelInfo{
			Architecture: "gemma4_text",
			NumLayers:    26,
			HiddenSize:   2304,
			VocabSize:    262144,
		},
		native: &fakeNativeModel{},
	}

	bundle, err := model.CaptureState(context.Background(), "one two three", inference.WithTemperature(0.25))

	core.RequireNoError(t, err)
	core.AssertEqual(t, "gemma4_text", bundle.Model.Architecture)
	core.AssertEqual(t, "q6", bundle.Model.QuantType)
	core.AssertEqual(t, 6, bundle.Model.QuantBits)
	core.AssertEqual(t, "E4B", bundle.Model.Labels["gemma4_size"])
	core.AssertEqual(t, "q6", bundle.Model.Labels["gemma4_quant_mode"])
	core.AssertEqual(t, "E4B", bundle.Labels["gemma4_size"])
	core.AssertEqual(t, "q6", bundle.Labels["gemma4_quant_mode"])
	core.AssertEqual(t, Gemma4RuntimeMLXAffine, bundle.Labels["gemma4_runtime"])
	core.AssertEqual(t, Gemma4GenerateLinked, bundle.Labels["gemma4_generate_status"])
	core.AssertEqual(t, "gemma4_mlx_affine", bundle.Labels["production_quant_policy"])
	core.AssertEqual(t, ROCmStateContextRegistryContract, bundle.Labels["engine_state_context_route_contract"])
	core.AssertEqual(t, "true", bundle.Labels["engine_state_context_prompt_replay_refused"])
	core.AssertEqual(t, ROCmLoRAAdapterRegistryContract, bundle.Labels["engine_lora_route_contract"])
	core.AssertEqual(t, "gemma4", bundle.Labels["engine_lora_target_policy"])
	core.AssertEqual(t, ROCmAttachedDrafterRegistryContract, bundle.Labels["engine_attached_drafter_route_contract"])
	core.AssertEqual(t, "target", bundle.Labels["engine_attached_drafter_role"])
	core.AssertEqual(t, defaultContextLengthCap-3, bundle.Sampler.MaxTokens)
	core.AssertEqual(t, float32(0.25), bundle.Sampler.Temperature)

	negativeBundle, err := model.CaptureState(context.Background(), "one two three", inference.WithMaxTokens(-1))
	core.RequireNoError(t, err)
	core.AssertEqual(t, defaultContextLengthCap-3, negativeBundle.Sampler.MaxTokens)
}

func TestStateSession_Bad_Gemma4CaptureStateRejectsMaxTokensPastWindow(t *testing.T) {
	model := &rocmModel{
		modelPath: "/models/lmstudio-community-gemma-4-e4b-it-6bit",
		modelType: "gemma4_text",
		modelInfo: inference.ModelInfo{
			Architecture: "gemma4_text",
			NumLayers:    26,
			HiddenSize:   2304,
			VocabSize:    262144,
		},
		native: &fakeNativeModel{},
	}

	_, err := model.CaptureState(context.Background(), strings.Repeat("x ", defaultContextLengthCap-1), inference.WithMaxTokens(2))

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "remaining model context window")
}

func TestStateSession_Bad_RocmModelCaptureStateRejectsNilModel(t *testing.T) {
	var model *rocmModel

	_, err := model.CaptureState(context.Background(), "hello")

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "model is nil")
}

func TestStateSession_Bad_RocmModelCaptureStateRejectsCancelledContext(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	model := &rocmModel{}

	_, err := model.CaptureState(ctx, "hello")

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "context canceled")
}

func TestStateSession_Good_RocmModelRestoresMetadataStateBundle(t *testing.T) {
	model := &rocmModel{modelInfo: inference.ModelInfo{Architecture: "qwen3"}}
	bundle := &inference.StateBundle{
		Model:     inference.ModelIdentity{Architecture: "qwen3"},
		Tokenizer: inference.TokenizerIdentity{Kind: "Qwen2Tokenizer"},
		Labels:    map[string]string{"tenant": "a"},
		KVRefs:    []inference.StateRef{{Kind: "kv", URI: "state://kv"}},
	}

	err := model.RestoreState(context.Background(), bundle)

	core.RequireNoError(t, err)
	if model.state == nil {
		t.Fatal("model.state is nil after RestoreState")
	}
	core.AssertEqual(t, "metadata_only", model.state.labels["kv_restore"])
	core.AssertEqual(t, "a", model.state.labels["tenant"])
	core.AssertEqual(t, "1", model.state.labels["state_bundle_ref"])
}

func TestStateSession_Good_RocmModelRestoreStateClosesPreviousRuntime(t *testing.T) {
	cache, err := newROCmKVCache(rocmKVCacheModeQ8, 2)
	core.RequireNoError(t, err)
	core.RequireNoError(t, cache.AppendVectors(0, 2, 2, []float32{1, 0, 0, 1}, []float32{2, 0, 0, 2}))
	device, err := cache.MirrorToDevice(&fakeHIPDriver{available: true})
	core.RequireNoError(t, err)
	model := &rocmModel{
		modelInfo: inference.ModelInfo{Architecture: "qwen3"},
		state:     newStateSessionWithRuntime(inference.ModelIdentity{}, inference.TokenizerIdentity{}, nil, device),
	}

	err = model.RestoreState(context.Background(), &inference.StateBundle{
		Model:  inference.ModelIdentity{Architecture: "qwen3"},
		Labels: map[string]string{"tenant": "b"},
	})

	core.RequireNoError(t, err)
	core.AssertEqual(t, true, device.closed)
	if model.state == nil {
		t.Fatal("model.state is nil after RestoreState")
	}
	core.AssertEqual(t, "metadata_only", model.state.labels["kv_restore"])
	core.AssertEqual(t, "b", model.state.labels["tenant"])
}

func TestStateSession_Bad_RocmModelRestoreStateCloseFailureKeepsPreviousState(t *testing.T) {
	runtime := &failingStateRuntime{err: core.NewError("close failed")}
	previous := newStateSessionWithRuntime(
		inference.ModelIdentity{Architecture: "qwen3"},
		inference.TokenizerIdentity{},
		map[string]string{"previous": "true"},
		runtime,
	)
	model := &rocmModel{
		modelInfo: inference.ModelInfo{Architecture: "qwen3"},
		state:     previous,
	}

	err := model.RestoreState(context.Background(), &inference.StateBundle{
		Model:  inference.ModelIdentity{Architecture: "qwen3"},
		Labels: map[string]string{"tenant": "new"},
	})

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "close previous state runtime")
	if model.state != previous {
		t.Fatal("RestoreState replaced previous state after close failure")
	}
	core.AssertEqual(t, runtime, previous.runtime)
	core.AssertEqual(t, 1, runtime.closeCalls)
	core.AssertEqual(t, "true", model.state.labels["previous"])
}

func TestStateSession_Bad_RocmModelRestoreStateRejectsIncompatibleModel(t *testing.T) {
	model := &rocmModel{modelInfo: inference.ModelInfo{Architecture: "qwen3"}}

	err := model.RestoreState(context.Background(), &inference.StateBundle{Model: inference.ModelIdentity{Architecture: "gemma"}})

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "model architecture mismatch")
}

func TestStateSession_Bad_RocmModelRestoreStateRejectsGemma4RunnableMismatch(t *testing.T) {
	model := &rocmModel{
		modelPath: "/models/lmstudio-community-gemma-4-31b-it-6bit",
		modelType: "gemma4_text",
		modelInfo: inference.ModelInfo{
			Architecture: "gemma4_text",
			NumLayers:    64,
			HiddenSize:   4096,
			VocabSize:    262144,
		},
	}
	bundleModel := gemma4StateModelIdentityForTest("/models/lmstudio-community-gemma-4-31b-it-6bit", 64, 4096)
	bundleModel.Labels = map[string]string{
		"gemma4_size":             "31B",
		"gemma4_quant_mode":       "q6-status",
		"gemma4_runtime":          Gemma4RuntimePlanned,
		"gemma4_generate_status":  Gemma4GeneratePlannedOnly,
		"gemma4_pack_supported":   "true",
		"gemma4_runnable_on_card": "true",
	}

	err := model.RestoreState(context.Background(), &inference.StateBundle{Model: bundleModel})

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "model Gemma4 runnable status mismatch")
	core.AssertNil(t, model.state)
}

func TestStateSession_Bad_RocmModelRestoreStateRejectsNilBundle(t *testing.T) {
	model := &rocmModel{}

	err := model.RestoreState(context.Background(), nil)

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "state bundle is nil")
}

func TestStateSession_Bad_RocmModelRestoreStateRecordsErr(t *testing.T) {
	model := &rocmModel{modelInfo: inference.ModelInfo{Architecture: "qwen3"}}

	err := model.RestoreState(context.Background(), nil)

	core.AssertError(t, err)
	if resultError(model.Err()) == nil {
		t.Fatal("RestoreState failure Err() = nil")
	}
	core.AssertContains(t, resultError(model.Err()).Error(), "state bundle is nil")

	err = model.RestoreState(context.Background(), &inference.StateBundle{Model: inference.ModelIdentity{Architecture: "qwen3"}})

	core.RequireNoError(t, err)
	if resultError(model.Err()) != nil {
		t.Fatalf("RestoreState success Err() = %v, want nil", resultError(model.Err()))
	}
}

func TestStateSession_Good_RocmModelPreservesWakeRuntimeForSleep(t *testing.T) {
	store := state.NewInMemoryStore(nil)
	cache, err := newROCmKVCache(rocmKVCacheModeQ8, 2)
	core.RequireNoError(t, err)
	core.RequireNoError(t, cache.AppendVectors(0, 2, 2, []float32{1, 0, 0, 1}, []float32{2, 0, 0, 2}))
	sleeping := newStateSessionWithRuntime(inference.ModelIdentity{Hash: "model-a"}, inference.TokenizerIdentity{}, nil, cache)
	_, err = sleeping.SleepState(context.Background(), inference.AgentMemorySleepRequest{
		Store:    store,
		EntryURI: "state://entry/source",
		Model:    inference.ModelIdentity{Hash: "model-a"},
		Encoding: rocmKVSnapshotEncoding,
	})
	core.RequireNoError(t, err)
	model := &rocmModel{modelInfo: inference.ModelInfo{Architecture: "qwen3"}}

	wake, err := model.WakeState(context.Background(), inference.AgentMemoryWakeRequest{
		Store:    store,
		EntryURI: "state://entry/source",
		Model:    inference.ModelIdentity{Hash: "model-a"},
	})
	core.RequireNoError(t, err)
	core.AssertEqual(t, "runtime_owned", wake.Labels["kv_restore"])
	sleep, err := model.SleepState(context.Background(), inference.AgentMemorySleepRequest{
		Store:    store,
		EntryURI: "state://entry/roundtrip",
		Model:    inference.ModelIdentity{Hash: "model-a"},
		Encoding: rocmKVSnapshotEncoding,
	})

	core.RequireNoError(t, err)
	core.AssertEqual(t, "runtime_owned", sleep.Labels["kv_serialize"])
	core.AssertEqual(t, 2, sleep.TokenCount)
	chunk, err := store.ResolveURI(context.Background(), sleep.Bundle.URI)
	core.RequireNoError(t, err)
	restored, err := newROCmKVCacheFromSnapshot(chunk.Data)
	core.RequireNoError(t, err)
	core.AssertEqual(t, 2, restored.TokenCount())
}

func TestStateSession_Good_RocmModelWakeStateRemirrorsKVSnapshotToHIPDevice(t *testing.T) {
	store := state.NewInMemoryStore(nil)
	cache, err := newROCmKVCache(rocmKVCacheModeKQ8VQ4, 2)
	core.RequireNoError(t, err)
	core.RequireNoError(t, cache.AppendVectors(
		0,
		2,
		3,
		[]float32{1, 0.5, -1, 0},
		[]float32{0.75, -0.5, 0.25, 1, -1, 0.5},
	))
	sleeping := newStateSessionWithRuntime(inference.ModelIdentity{Hash: "model-a"}, inference.TokenizerIdentity{}, nil, cache)
	_, err = sleeping.SleepState(context.Background(), inference.AgentMemorySleepRequest{
		Store:    store,
		EntryURI: "state://entry/kv",
		Model:    inference.ModelIdentity{Hash: "model-a"},
		Encoding: rocmKVSnapshotEncoding,
	})
	core.RequireNoError(t, err)
	driver := &fakeHIPDriver{available: true}
	model := &rocmModel{
		modelInfo: inference.ModelInfo{Architecture: "qwen3"},
		native:    &hipLoadedModel{driver: driver},
	}

	wake, err := model.WakeState(context.Background(), inference.AgentMemoryWakeRequest{
		Store:    store,
		EntryURI: "state://entry/kv",
		Model:    inference.ModelIdentity{Hash: "model-a"},
	})

	core.RequireNoError(t, err)
	core.AssertEqual(t, "device_mirror", wake.Labels["kv_restore"])
	core.AssertEqual(t, "hip_device_mirror", wake.Labels["kv_backing"])
	core.AssertEqual(t, "mirrored", wake.Labels["kv_device_backing"])
	core.AssertEqual(t, "mirrored", wake.Labels["kv_device_restore"])
	core.AssertEqual(t, rocmKVCacheModeKQ8VQ4, wake.Labels["cache_mode"])
	device, ok := model.state.runtime.(*rocmDeviceKVCache)
	core.RequireTrue(t, ok)
	core.AssertEqual(t, 2, device.TokenCount())
	core.AssertEqual(t, rocmKVCacheModeKQ8VQ4, device.Stats().CacheMode)
	sleep, err := model.SleepState(context.Background(), inference.AgentMemorySleepRequest{
		Store:    store,
		EntryURI: "state://entry/remirrored",
		Model:    inference.ModelIdentity{Hash: "model-a"},
		Encoding: rocmKVSnapshotEncoding,
	})
	core.RequireNoError(t, err)
	core.AssertEqual(t, "device_mirror", sleep.Labels["kv_serialize"])
	core.AssertEqual(t, "hip_device_mirror", sleep.Labels["kv_backing"])

	core.RequireNoError(t, resultError(model.Close()))
	core.AssertEqual(t, true, device.closed)
}

func TestStateSession_Good_RocmModelWakeStateRestoresKVBlockBundleDirectToHIPDevice(t *testing.T) {
	store := state.NewInMemoryStore(nil)
	cache, err := newROCmKVCache(rocmKVCacheModeKQ8VQ4, 2)
	core.RequireNoError(t, err)
	core.RequireNoError(t, cache.AppendVectors(
		0,
		2,
		3,
		[]float32{1, 0.5, -1, 0, 2, -2},
		[]float32{0.75, -0.5, 0.25, 1, -1, 0.5, 2, -2, 3},
	))
	sleeping := newStateSessionWithRuntime(inference.ModelIdentity{Hash: "model-a"}, inference.TokenizerIdentity{}, nil, cache)
	_, err = sleeping.SleepState(context.Background(), inference.AgentMemorySleepRequest{
		Store:    store,
		EntryURI: "state://entry/kv-blocks-direct",
		Model:    inference.ModelIdentity{Hash: "model-a"},
		Encoding: rocmKVBlockBundleEncoding,
	})
	core.RequireNoError(t, err)
	driver := &fakeHIPDriver{available: true}
	model := &rocmModel{
		modelInfo: inference.ModelInfo{Architecture: "qwen3"},
		native:    &hipLoadedModel{driver: driver},
	}

	wake, err := model.WakeState(context.Background(), inference.AgentMemoryWakeRequest{
		Store:    store,
		EntryURI: "state://entry/kv-blocks-direct",
		Model:    inference.ModelIdentity{Hash: "model-a"},
	})

	core.RequireNoError(t, err)
	core.AssertEqual(t, "hip_device_block_stream", wake.Labels["kv_restore"])
	core.AssertEqual(t, "block_stream", wake.Labels["kv_device_restore"])
	core.AssertEqual(t, "borrow_ref_pinned", wake.Labels["kv_device_restore_path"])
	core.AssertEqual(t, "hip_device_mirror", wake.Labels["kv_backing"])
	device, ok := model.state.runtime.(*rocmDeviceKVCache)
	core.RequireTrue(t, ok)
	core.AssertEqual(t, 3, device.TokenCount())
	core.AssertEqual(t, 2, device.PageCount())
	if rocmHIPPinnedHostCopySupported {
		core.AssertEqual(t, true, driver.pinnedCopies >= 4)
	}

	host, err := device.hostCache()
	core.RequireNoError(t, err)
	keys, values, err := host.Restore(0, 3)
	core.RequireNoError(t, err)
	assertFloat32SlicesNear(t, []float32{1, 0.5, -1, 0, 2, -2}, keys, 0.15)
	assertFloat32SlicesNear(t, []float32{0.75, -0.5, 0.25, 1, -1, 0.5, 2, -2, 3}, values, 0.25)
}

func TestStateSession_Good_RocmModelWakeStateDirectHIPDeviceFromFileStoreBorrowedBlocks(t *testing.T) {
	path := filepath.Join(t.TempDir(), "state.mvlog")
	writer, err := filestore.Create(context.Background(), path)
	core.RequireNoError(t, err)
	cache, err := newROCmKVCache(rocmKVCacheModeQ8, 2)
	core.RequireNoError(t, err)
	core.RequireNoError(t, cache.AppendVectors(0, 2, 2, []float32{1, 0, 0, 1}, []float32{2, 0, 0, 2}))
	sleeping := newStateSessionWithRuntime(inference.ModelIdentity{Hash: "model-a"}, inference.TokenizerIdentity{}, nil, cache)
	_, err = sleeping.SleepState(context.Background(), inference.AgentMemorySleepRequest{
		Store:    writer,
		EntryURI: "state://entry/kv-file-blocks",
		Model:    inference.ModelIdentity{Hash: "model-a"},
		Encoding: rocmKVBlockBundleEncoding,
	})
	core.RequireNoError(t, err)
	core.RequireNoError(t, writer.Close())
	reader, err := filestore.Open(context.Background(), path)
	core.RequireNoError(t, err)
	defer reader.Close()
	driver := &fakeHIPDriver{available: true}
	model := &rocmModel{
		modelInfo: inference.ModelInfo{Architecture: "qwen3"},
		native:    &hipLoadedModel{driver: driver},
	}

	wake, err := model.WakeState(context.Background(), inference.AgentMemoryWakeRequest{
		Store:    reader,
		EntryURI: "state://entry/kv-file-blocks",
		Model:    inference.ModelIdentity{Hash: "model-a"},
	})

	core.RequireNoError(t, err)
	core.AssertEqual(t, "hip_device_block_stream", wake.Labels["kv_restore"])
	core.AssertEqual(t, "borrow_ref_pinned", wake.Labels["kv_device_restore_path"])
	device, ok := model.state.runtime.(*rocmDeviceKVCache)
	core.RequireTrue(t, ok)
	core.AssertEqual(t, 2, device.TokenCount())
	if rocmHIPPinnedHostCopySupported {
		core.AssertEqual(t, true, driver.pinnedCopies >= 2)
	}
}

func TestStateSession_Good_RocmModelWakeStateKeepsPackageLocalKVOnDeviceMirrorFailure(t *testing.T) {
	store := state.NewInMemoryStore(nil)
	cache, err := newROCmKVCache(rocmKVCacheModeQ8, 2)
	core.RequireNoError(t, err)
	core.RequireNoError(t, cache.AppendVectors(0, 2, 2, []float32{1, 0, 0, 1}, []float32{2, 0, 0, 2}))
	sleeping := newStateSessionWithRuntime(inference.ModelIdentity{Hash: "model-a"}, inference.TokenizerIdentity{}, nil, cache)
	_, err = sleeping.SleepState(context.Background(), inference.AgentMemorySleepRequest{
		Store:    store,
		EntryURI: "state://entry/kv",
		Model:    inference.ModelIdentity{Hash: "model-a"},
		Encoding: rocmKVSnapshotEncoding,
	})
	core.RequireNoError(t, err)
	driver := &fakeHIPDriver{available: true, copyErr: core.NewError("copy failed"), copyErrAt: 1}
	model := &rocmModel{
		modelInfo: inference.ModelInfo{Architecture: "qwen3"},
		native:    &hipLoadedModel{driver: driver},
	}

	wake, err := model.WakeState(context.Background(), inference.AgentMemoryWakeRequest{
		Store:    store,
		EntryURI: "state://entry/kv",
		Model:    inference.ModelIdentity{Hash: "model-a"},
	})

	core.RequireNoError(t, err)
	core.AssertEqual(t, "runtime_owned", wake.Labels["kv_restore"])
	core.AssertEqual(t, "package_local", wake.Labels["kv_backing"])
	core.AssertEqual(t, "failed", wake.Labels["kv_device_restore"])
	core.AssertContains(t, wake.Labels["kv_device_restore_error"], "copy KV key page")
	restored, ok := model.state.runtime.(*rocmKVCache)
	core.RequireTrue(t, ok)
	core.AssertEqual(t, 2, restored.TokenCount())
}

func TestStateSession_Bad_RocmModelWakeStateClosePreviousDeviceRuntimeFailureKeepsPreviousState(t *testing.T) {
	store := state.NewInMemoryStore(nil)
	nextCache, err := newROCmKVCache(rocmKVCacheModeQ8, 2)
	core.RequireNoError(t, err)
	core.RequireNoError(t, nextCache.AppendVectors(0, 2, 2, []float32{1, 0, 0, 1}, []float32{2, 0, 0, 2}))
	nextPayload, err := nextCache.Snapshot()
	core.RequireNoError(t, err)
	_, err = store.PutBytes(context.Background(), nextPayload, state.PutOptions{URI: "state://entry/next-kv"})
	core.RequireNoError(t, err)
	previousCache, err := newROCmKVCache(rocmKVCacheModeQ8, 2)
	core.RequireNoError(t, err)
	core.RequireNoError(t, previousCache.AppendVectors(0, 2, 2, []float32{3, 0, 0, 3}, []float32{4, 0, 0, 4}))
	driver := &failingHIPDriver{available: true, freeErr: core.NewError("free failed")}
	previousDevice, err := previousCache.MirrorToDevice(driver)
	core.RequireNoError(t, err)
	previous := newStateSessionWithRuntime(inference.ModelIdentity{}, inference.TokenizerIdentity{}, nil, previousDevice)
	model := &rocmModel{
		modelInfo: inference.ModelInfo{Architecture: "qwen3"},
		state:     previous,
	}

	wake, err := model.WakeState(context.Background(), inference.AgentMemoryWakeRequest{Store: store, EntryURI: "state://entry/next-kv"})

	core.AssertError(t, err)
	core.AssertNil(t, wake)
	core.AssertContains(t, err.Error(), "close previous state runtime")
	core.AssertContains(t, err.Error(), "free failed")
	if model.state != previous || model.state.runtime != previousDevice {
		t.Fatal("rocmModel WakeState replaced previous state after device runtime close failure")
	}
	core.AssertEqual(t, len(driver.allocations), len(driver.frees))
}

func TestStateSession_Good_RocmModelCloseClosesStateWithoutNative(t *testing.T) {
	cache, err := newROCmKVCache(rocmKVCacheModeQ8, 2)
	core.RequireNoError(t, err)
	core.RequireNoError(t, cache.AppendVectors(0, 2, 2, []float32{1, 0, 0, 1}, []float32{2, 0, 0, 2}))
	device, err := cache.MirrorToDevice(&fakeHIPDriver{available: true})
	core.RequireNoError(t, err)
	model := &rocmModel{state: newStateSessionWithRuntime(inference.ModelIdentity{}, inference.TokenizerIdentity{}, nil, device)}

	err = resultError(model.Close())

	core.RequireNoError(t, err)
	core.AssertEqual(t, true, device.closed)
	if model.state != nil {
		t.Fatal("model.state should be nil after Close")
	}
}

func TestStateSession_Good_RocmModelAdapterChangeResetsState(t *testing.T) {
	store := state.NewInMemoryStore(nil)
	cache, err := newROCmKVCache(rocmKVCacheModeQ8, 2)
	core.RequireNoError(t, err)
	core.RequireNoError(t, cache.AppendVectors(0, 1, 1, []float32{1, 0}, []float32{2, 0}))
	sleeping := newStateSessionWithRuntime(inference.ModelIdentity{}, inference.TokenizerIdentity{}, nil, cache)
	_, err = sleeping.SleepState(context.Background(), inference.AgentMemorySleepRequest{Store: store, EntryURI: "state://entry/source"})
	core.RequireNoError(t, err)
	model := &rocmModel{native: &fakeNativeModel{}, modelInfo: inference.ModelInfo{Architecture: "qwen3"}}
	_, err = model.WakeState(context.Background(), inference.AgentMemoryWakeRequest{Store: store, EntryURI: "state://entry/source"})
	core.RequireNoError(t, err)

	_, err = model.LoadAdapter("domain.safetensors")
	core.RequireNoError(t, err)
	sleep, err := model.SleepState(context.Background(), inference.AgentMemorySleepRequest{Store: store, EntryURI: "state://entry/after-adapter"})

	core.AssertNil(t, sleep)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "KV runtime is required")
}

func TestStateSession_Good_RocmModelSleepStateDefaultsActiveGemma4Adapter(t *testing.T) {
	for _, tc := range []struct {
		name    string
		path    string
		size    string
		mode    string
		group   string
		runtime string
		status  string
	}{
		{name: "linked_q4", path: "/models/lmstudio-community-gemma-4-e2b-it-4bit", size: "E2B", mode: "q4", group: "64", runtime: Gemma4RuntimeMLXAffine, status: Gemma4GenerateLinked},
		{name: "planned_mxfp8", path: "/models/lmstudio-community-gemma-4-e4b-it-mxfp8", size: "E4B", mode: "mxfp8", group: "32", runtime: Gemma4RuntimePlanned, status: Gemma4GeneratePlannedOnly},
	} {
		t.Run(tc.name, func(t *testing.T) {
			store := state.NewInMemoryStore(nil)
			cache, err := newROCmKVCache(rocmKVCacheModeKQ8VQ4, 2)
			core.RequireNoError(t, err)
			core.RequireNoError(t, cache.AppendVectors(0, 1, 1, []float32{1, 0, 0}, []float32{0, 1, 0}))
			model := &rocmModel{
				modelPath: tc.path,
				modelType: "gemma4_text",
				modelInfo: inference.ModelInfo{
					Architecture: "gemma4_text",
					VocabSize:    262144,
				},
				native: &fakeNativeModel{
					adapter: inference.AdapterIdentity{
						Path:   "domain.safetensors",
						Format: "lora",
					},
				},
			}
			model.state = newStateSessionWithRuntime(model.modelIdentity(), inference.TokenizerIdentity{}, nil, cache)

			sleep, err := model.SleepState(context.Background(), inference.AgentMemorySleepRequest{
				Store:    store,
				EntryURI: "state://entry/model-active-adapter-" + tc.name,
				Encoding: rocmKVBlockBundleEncoding,
			})

			core.RequireNoError(t, err)
			core.AssertEqual(t, "metadata_only", sleep.Labels["state_adapter"])
			core.AssertEqual(t, tc.size, sleep.Labels["adapter_base_gemma4_size"])
			core.AssertEqual(t, tc.mode, sleep.Labels["adapter_base_gemma4_quant_mode"])
			core.AssertEqual(t, tc.group, sleep.Labels["adapter_base_gemma4_quant_group"])
			core.AssertEqual(t, tc.runtime, sleep.Entry.StateRefs[0].Labels["adapter_base_gemma4_runtime"])
			core.AssertEqual(t, tc.status, sleep.Entry.StateRefs[0].Labels["adapter_base_gemma4_generate_status"])
		})
	}
}

func TestStateSession_Good_Gemma4RetainedKVRefsCarryIdentityLabels(t *testing.T) {
	store := state.NewInMemoryStore(nil)
	cache, err := newROCmKVCache(rocmKVCacheModeKQ8VQ4, 2)
	core.RequireNoError(t, err)
	core.RequireNoError(t, cache.AppendVectors(0, 3, 3, []float32{1, 0, 0, 1, 2, 0}, []float32{0, 1, 0, 1, 0, 2}))
	model := &rocmModel{
		modelPath: "/models/lmstudio-community-gemma-4-e4b-it-6bit",
		modelType: "gemma4_text",
		modelInfo: inference.ModelInfo{
			Architecture: "gemma4_text",
			NumLayers:    26,
			HiddenSize:   2304,
			VocabSize:    262144,
		},
	}
	identity := model.modelIdentity()
	sleeping := newStateSessionWithRuntime(identity, inference.TokenizerIdentity{}, nil, cache)
	source, err := sleeping.SleepState(context.Background(), inference.AgentMemorySleepRequest{
		Store:    store,
		EntryURI: "state://entry/gemma4-source",
		Model:    identity,
		Encoding: rocmKVBlockBundleEncoding,
	})
	core.RequireNoError(t, err)

	wake, err := model.WakeState(context.Background(), inference.AgentMemoryWakeRequest{
		Store:    store,
		IndexURI: source.Entry.IndexURI,
		Model:    identity,
	})
	core.RequireNoError(t, err)
	assertGemma4RetainedKVLabels(t, wake.Labels, rocmKVCacheModeKQ8VQ4)
	assertGemma4RetainedKVLabels(t, wake.Entry.Labels, rocmKVCacheModeKQ8VQ4)
	assertGemma4RetainedKVLabels(t, wake.Bundle.Labels, rocmKVCacheModeKQ8VQ4)
	assertGemma4RetainedKVLabels(t, wake.Index.Labels, rocmKVCacheModeKQ8VQ4)

	roundtrip, err := model.SleepState(context.Background(), inference.AgentMemorySleepRequest{
		Store:    store,
		EntryURI: "state://entry/gemma4-roundtrip",
		Model:    identity,
		Encoding: rocmKVBlockBundleEncoding,
	})
	core.RequireNoError(t, err)
	assertGemma4RetainedKVLabels(t, roundtrip.Labels, rocmKVCacheModeKQ8VQ4)
	assertGemma4RetainedKVLabels(t, roundtrip.Entry.Labels, rocmKVCacheModeKQ8VQ4)
	assertGemma4RetainedKVLabels(t, roundtrip.Bundle.Labels, rocmKVCacheModeKQ8VQ4)
	assertGemma4RetainedKVLabels(t, roundtrip.Index.Labels, rocmKVCacheModeKQ8VQ4)
	core.RequireTrue(t, len(roundtrip.Entry.StateRefs) == 1)
	assertGemma4RetainedKVLabels(t, roundtrip.Entry.StateRefs[0].Labels, rocmKVCacheModeKQ8VQ4)

	forked, forkWake, err := model.ForkState(context.Background(), inference.AgentMemoryWakeRequest{
		Store:    store,
		IndexURI: source.Entry.IndexURI,
		Model:    identity,
	})
	core.RequireNoError(t, err)
	core.RequireTrue(t, forked != nil)
	assertGemma4RetainedKVLabels(t, forkWake.Labels, rocmKVCacheModeKQ8VQ4)
	assertGemma4RetainedKVLabels(t, forkWake.Index.Labels, rocmKVCacheModeKQ8VQ4)
}

func assertGemma4RetainedKVLabels(t *testing.T, labels map[string]string, cacheMode string) {
	t.Helper()
	core.AssertEqual(t, "E4B", labels["gemma4_size"])
	core.AssertEqual(t, "q6", labels["gemma4_quant_mode"])
	core.AssertEqual(t, Gemma4RuntimeMLXAffine, labels["gemma4_runtime"])
	core.AssertEqual(t, Gemma4GenerateLinked, labels["gemma4_generate_status"])
	core.AssertEqual(t, "gemma4_mlx_affine", labels["production_quant_policy"])
	core.AssertEqual(t, cacheMode, labels["cache_mode"])
	core.AssertEqual(t, ROCmStateContextRegistryContract, labels["engine_state_context_route_contract"])
	core.AssertEqual(t, "true", labels["engine_state_context_prompt_replay_refused"])
	core.AssertEqual(t, "true", labels["engine_state_context_runtime_owned_kv"])
	core.AssertEqual(t, ROCmLoRAAdapterRegistryContract, labels["engine_lora_route_contract"])
	core.AssertEqual(t, "gemma4", labels["engine_lora_target_policy"])
	core.AssertEqual(t, ROCmAttachedDrafterRegistryContract, labels["engine_attached_drafter_route_contract"])
	core.AssertEqual(t, "target", labels["engine_attached_drafter_role"])
	core.AssertEqual(t, hipKernelStatusNotLinked, labels["engine_attached_drafter_native_attachment"])
	core.AssertEqual(t, "forbidden", labels["engine_attached_drafter_prompt_replay_fallback"])
}

func gemma4StateModelIdentityForTest(path string, layers, hiddenSize int) inference.ModelIdentity {
	return inference.ModelIdentity{
		Path:         path,
		Architecture: "gemma4_text",
		NumLayers:    layers,
		HiddenSize:   hiddenSize,
		VocabSize:    262144,
	}
}

func seedStateSessionKV(t *testing.T, store *state.InMemoryStore, entryURI string, model inference.ModelIdentity, tokenizer inference.TokenizerIdentity) *inference.AgentMemorySleepResult {
	t.Helper()
	cache, err := newROCmKVCache(rocmKVCacheModeQ8, defaultROCmStateBlockSize)
	core.RequireNoError(t, err)
	core.RequireNoError(t, cache.AppendVectors(0, 1, 1, []float32{1, 2, 3}, []float32{3, 2, 1}))
	session := newStateSessionWithRuntime(model, tokenizer, nil, cache)
	sleep, err := session.SleepState(context.Background(), inference.AgentMemorySleepRequest{
		Store:     store,
		EntryURI:  entryURI,
		Model:     model,
		Tokenizer: tokenizer,
		Encoding:  rocmKVBlockBundleEncoding,
	})
	core.RequireNoError(t, err)
	return sleep
}

type recordingStateWriter struct {
	text     string
	options  state.PutOptions
	err      error
	putCalls int
}

type failingStateBinaryWriter struct {
	err           error
	putBytesCalls int
	options       state.PutOptions
	payload       []byte
}

type borrowRecordingStateStore struct {
	*state.InMemoryStore
	borrowRefs []state.ChunkRef
}

type releasingBorrowStateStore struct {
	*state.InMemoryStore
	releaseCalls int
}

type failingStateRuntime struct {
	err        error
	closeCalls int
}

func (runtime *failingStateRuntime) Close() error {
	runtime.closeCalls++
	return runtime.err
}

func (writer *recordingStateWriter) Put(_ context.Context, text string, opts state.PutOptions) (state.ChunkRef, error) {
	writer.putCalls++
	writer.text = text
	writer.options = opts
	if writer.err != nil {
		return state.ChunkRef{}, writer.err
	}
	return state.ChunkRef{ChunkID: 7, Codec: state.CodecMemory}, nil
}

func (writer *failingStateBinaryWriter) PutBytes(_ context.Context, data []byte, opts state.PutOptions) (state.ChunkRef, error) {
	writer.putBytesCalls++
	writer.options = opts
	writer.payload = append([]byte(nil), data...)
	return state.ChunkRef{}, writer.err
}

func (store *borrowRecordingStateStore) BorrowRefBytes(ctx context.Context, ref state.ChunkRef) (state.BorrowedChunk, error) {
	store.borrowRefs = append(store.borrowRefs, ref)
	chunk, err := state.ResolveRefBytes(ctx, store.InMemoryStore, ref)
	if err != nil {
		return state.BorrowedChunk{}, err
	}
	return state.BorrowedChunk{Ref: chunk.Ref, Data: chunk.Data}, nil
}

func (store *releasingBorrowStateStore) BorrowRefBytes(ctx context.Context, ref state.ChunkRef) (state.BorrowedChunk, error) {
	chunk, err := state.ResolveRefBytes(ctx, store.InMemoryStore, ref)
	if err != nil {
		return state.BorrowedChunk{}, err
	}
	return state.BorrowedChunk{
		Ref:  chunk.Ref,
		Data: chunk.Data,
		Release: func() {
			store.releaseCalls++
			for i := range chunk.Data {
				chunk.Data[i] = 0xff
			}
		},
	}, nil
}
