// SPDX-Licence-Identifier: EUPL-1.2

package session

import (
	"context"
	"fmt"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/inference/kv"
	mlxbundle "dappco.re/go/inference/model/bundle"
	"dappco.re/go/inference/model/spine"
	memvid "dappco.re/go/inference/model/state"
	"dappco.re/go/inference/model/state/agent"
	"dappco.re/go/inference/model/state/session/internal/sessionfake"
)

// ExampleNew wraps an already-created native session handle — the
// construction seam the root mlx package builds on. Tests and callers
// hand New a handle, the model info, and a tokenizer.
func ExampleNew() {
	handle := &sessionfake.Handle{
		Tokens: []inference.Token{{ID: 1, Text: "Hi"}, {ID: 2, Text: " there"}},
	}
	sess := New(handle, spine.ModelInfo{Architecture: "gemma4_text"}, nil)

	_ = sess.Prefill("stable context")
	reply, _ := sess.Generate(optMaxTokens(8))

	fmt.Println(reply)
	// Output: Hi there
}

// ExampleSession_Valid reports whether a session still holds a live native
// handle — the exported form of the internal nil/closed guard.
func ExampleSession_Valid() {
	sess := New(&sessionfake.Handle{}, spine.ModelInfo{}, nil)
	fmt.Println("after new:", sess.Valid())

	_ = sess.Close()
	fmt.Println("after close:", sess.Valid())

	var nilSession *Session
	fmt.Println("nil session:", nilSession.Valid())
	// Output:
	// after new: true
	// after close: false
	// nil session: false
}

// ExampleSession_Native returns the underlying native session handle — the
// accessor callers outside the package use instead of reaching the
// unexported field. A nil *Session yields a nil handle.
func ExampleSession_Native() {
	handle := &sessionfake.Handle{}
	sess := New(handle, spine.ModelInfo{}, nil)

	fmt.Println("same handle:", sess.Native() == handle)

	var nilSession *Session
	fmt.Println("nil handle:", nilSession.Native() == nil)
	// Output:
	// same handle: true
	// nil handle: true
}

// ExampleSession_Prefill loads a prompt into the retained session KV state.
func ExampleSession_Prefill() {
	sess := New(&sessionfake.Handle{}, spine.ModelInfo{}, nil)

	err := sess.Prefill("stable context")

	fmt.Println("error:", err)
	// Output:
	// error: <nil>
}

// ExampleSession_PrefillChunks loads bounded prompt chunks into the
// retained session KV state.
func ExampleSession_PrefillChunks() {
	sess := New(&sessionfake.Handle{}, spine.ModelInfo{}, nil)

	err := sess.PrefillChunks(context.Background(), func(yield func(string) bool) {
		yield("stable ")
		yield("context")
	})

	fmt.Println("error:", err)
	// Output:
	// error: <nil>
}

// ExampleSession_PrefillTokens loads model-native token IDs into the
// retained session KV state.
func ExampleSession_PrefillTokens() {
	sess := New(&sessionfake.Handle{}, spine.ModelInfo{}, nil)

	err := sess.PrefillTokens(context.Background(), []int32{11, 12})

	fmt.Println("error:", err)
	// Output:
	// error: <nil>
}

// ExampleSession_AppendPrompt appends prompt text to the retained session KV
// state without replaying the existing prefix.
func ExampleSession_AppendPrompt() {
	sess := New(&sessionfake.Handle{}, spine.ModelInfo{}, nil)

	err := sess.AppendPrompt("\n\nQuestion: who?")

	fmt.Println("error:", err)
	// Output:
	// error: <nil>
}

// ExampleSession_AppendPromptChunks appends bounded prompt chunks to the
// retained session KV state without replaying the existing prefix.
func ExampleSession_AppendPromptChunks() {
	sess := New(&sessionfake.Handle{}, spine.ModelInfo{}, nil)

	err := sess.AppendPromptChunks(context.Background(), func(yield func(string) bool) {
		yield("\n\nQuestion: ")
		yield("who?")
	})

	fmt.Println("error:", err)
	// Output:
	// error: <nil>
}

// ExampleSession_AppendTokens appends model-native token IDs to the
// retained session KV state without replaying the existing prefix.
func ExampleSession_AppendTokens() {
	sess := New(&sessionfake.Handle{}, spine.ModelInfo{}, nil)

	err := sess.AppendTokens(context.Background(), []int32{21, 22})

	fmt.Println("error:", err)
	// Output:
	// error: <nil>
}

// ExampleSession_Generate produces a buffered reply from the retained
// session state.
func ExampleSession_Generate() {
	handle := &sessionfake.Handle{
		Tokens: []inference.Token{{ID: 1, Text: "Hi"}, {ID: 2, Text: " there"}},
	}
	sess := New(handle, spine.ModelInfo{Architecture: "gemma4_text"}, nil)

	reply, err := sess.Generate()
	if err != nil {
		fmt.Println("error:", err)
		return
	}

	fmt.Println(reply)
	// Output:
	// Hi there
}

// ExampleSession_GenerateStream streams tokens from the retained session
// state as they are produced.
func ExampleSession_GenerateStream() {
	handle := &sessionfake.Handle{
		Tokens: []inference.Token{{ID: 1, Text: "Hi"}, {ID: 2, Text: " there"}},
	}
	sess := New(handle, spine.ModelInfo{Architecture: "gemma4_text"}, nil)

	var got string
	for tok := range sess.GenerateStream(context.Background()) {
		got += tok.Text
	}

	fmt.Println(got)
	// Output:
	// Hi there
}

// ExampleSession_CaptureKV copies the current retained KV cache tensors to
// CPU memory.
func ExampleSession_CaptureKV() {
	sess := New(&sessionfake.Handle{KV: sessionfake.TestKVSnapshot()}, spine.ModelInfo{}, nil)

	snapshot, err := sess.CaptureKV()
	if err != nil {
		fmt.Println("error:", err)
		return
	}

	fmt.Println("tokens:", len(snapshot.Tokens))
	// Output:
	// tokens: 2
}

// ExampleSession_CaptureKVWithOptions copies the current retained KV cache
// tensors with explicit capture options — RawKVOnly drops the decoded
// float32 view, keeping only the raw bytes.
func ExampleSession_CaptureKVWithOptions() {
	sess := New(&sessionfake.Handle{KV: sessionfake.TestKVSnapshot()}, spine.ModelInfo{}, nil)

	snapshot, err := sess.CaptureKVWithOptions(kv.CaptureOptions{RawKVOnly: true})
	if err != nil {
		fmt.Println("error:", err)
		return
	}

	fmt.Println("float32 view kept:", len(snapshot.Layers[0].Heads[0].Key) != 0)
	// Output:
	// float32 view kept: false
}

// ExampleSession_AnalyzeKV captures and analyses the current retained KV
// state.
func ExampleSession_AnalyzeKV() {
	sess := New(&sessionfake.Handle{KV: sessionfake.TestKVSnapshot()}, spine.ModelInfo{}, nil)

	analysis, err := sess.AnalyzeKV()
	if err != nil {
		fmt.Println("error:", err)
		return
	}

	fmt.Println("analysis produced:", analysis != nil)
	// Output:
	// analysis produced: true
}

// ExampleSession_SaveKV captures and writes the current retained KV state to
// a path on disk.
func ExampleSession_SaveKV() {
	sess := New(&sessionfake.Handle{KV: sessionfake.TestKVSnapshot()}, spine.ModelInfo{}, nil)
	dir := core.MkdirTemp("", "session-savekv-example-*").Value.(string)
	path := core.PathJoin(dir, "session_example.kvbin")

	err := sess.SaveKV(path)

	fmt.Println("error:", err)
	// Output:
	// error: <nil>
}

// ExampleSession_RestoreKV replaces the retained session state with a
// restorable KV snapshot.
func ExampleSession_RestoreKV() {
	native := &sessionfake.Handle{}
	sess := New(native, spine.ModelInfo{}, nil)

	err := sess.RestoreKV(sessionfake.TestKVSnapshot())

	fmt.Println("error:", err)
	fmt.Println("restored:", native.RestoredKV != nil)
	// Output:
	// error: <nil>
	// restored: true
}

// ExampleSession_LoadKV reads a KV snapshot from disk and restores it into
// the session.
func ExampleSession_LoadKV() {
	dir := core.MkdirTemp("", "session-loadkv-example-*").Value.(string)
	path := core.PathJoin(dir, "session_example_load.kvbin")
	if err := sessionfake.TestKVSnapshot().Save(path); err != nil {
		fmt.Println("seed error:", err)
		return
	}
	native := &sessionfake.Handle{}
	sess := New(native, spine.ModelInfo{}, nil)

	err := sess.LoadKV(path)

	fmt.Println("error:", err)
	fmt.Println("restored:", native.RestoredKV != nil)
	// Output:
	// error: <nil>
	// restored: true
}

// ExampleSession_SaveKVToState captures and writes the current retained KV
// state to a State store, returning a chunk reference.
func ExampleSession_SaveKVToState() {
	store := memvid.NewInMemoryStore(nil)
	sess := New(&sessionfake.Handle{KV: sessionfake.TestKVSnapshot()}, spine.ModelInfo{}, nil)

	ref, err := sess.SaveKVToState(context.Background(), store, kv.StateOptions{URI: "mlx://session/example"})
	if err != nil {
		fmt.Println("error:", err)
		return
	}

	fmt.Println("stored:", ref.ChunkID != 0)
	// Output:
	// stored: true
}

// ExampleSession_SaveKVToMemvid is the deprecated alias for SaveKVToState —
// kv.MemvidOptions is a type alias for kv.StateOptions, so the same call
// shape works unchanged.
func ExampleSession_SaveKVToMemvid() {
	store := memvid.NewInMemoryStore(nil)
	sess := New(&sessionfake.Handle{KV: sessionfake.TestKVSnapshot()}, spine.ModelInfo{}, nil)

	ref, err := sess.SaveKVToMemvid(context.Background(), store, kv.MemvidOptions{URI: "mlx://session/example-memvid"})
	if err != nil {
		fmt.Println("error:", err)
		return
	}

	fmt.Println("stored:", ref.ChunkID != 0)
	// Output:
	// stored: true
}

// ExampleSession_LoadKVFromState restores retained session state from a
// State KV chunk reference.
func ExampleSession_LoadKVFromState() {
	store := memvid.NewInMemoryStore(nil)
	writer := New(&sessionfake.Handle{KV: sessionfake.TestKVSnapshot()}, spine.ModelInfo{}, nil)
	ref, err := writer.SaveKVToState(context.Background(), store, kv.StateOptions{URI: "mlx://session/load-example"})
	if err != nil {
		fmt.Println("seed error:", err)
		return
	}

	native := &sessionfake.Handle{}
	reader := New(native, spine.ModelInfo{}, nil)
	if err := reader.LoadKVFromState(context.Background(), store, ref); err != nil {
		fmt.Println("error:", err)
		return
	}

	fmt.Println("restored:", native.RestoredKV != nil)
	// Output:
	// restored: true
}

// ExampleSession_SaveKVBlocksToState streams the retained KV state into
// per-block State chunks.
func ExampleSession_SaveKVBlocksToState() {
	store := memvid.NewInMemoryStore(nil)
	sess := New(&sessionfake.Handle{KV: sessionfake.TestKVSnapshot()}, spine.ModelInfo{}, nil)

	bundle, err := sess.SaveKVBlocksToState(context.Background(), store, kv.StateBlockOptions{BlockSize: 2})
	if err != nil {
		fmt.Println("error:", err)
		return
	}

	fmt.Println("blocks:", len(bundle.Blocks))
	// Output:
	// blocks: 1
}

// ExampleSession_SaveKVBlocksToMemvid is the deprecated alias for
// SaveKVBlocksToState — kv.MemvidBlockOptions is a type alias for
// kv.StateBlockOptions, so the same call shape works unchanged.
func ExampleSession_SaveKVBlocksToMemvid() {
	store := memvid.NewInMemoryStore(nil)
	sess := New(&sessionfake.Handle{KV: sessionfake.TestKVSnapshot()}, spine.ModelInfo{}, nil)

	bundle, err := sess.SaveKVBlocksToMemvid(context.Background(), store, kv.MemvidBlockOptions{BlockSize: 2})
	if err != nil {
		fmt.Println("error:", err)
		return
	}

	fmt.Println("blocks:", len(bundle.Blocks))
	// Output:
	// blocks: 1
}

// ExampleSession_LoadKVBlocksFromState restores retained session state from
// every per-block State chunk in the bundle.
func ExampleSession_LoadKVBlocksFromState() {
	store := memvid.NewInMemoryStore(nil)
	writer := New(&sessionfake.Handle{KV: sessionfake.TestKVSnapshot()}, spine.ModelInfo{}, nil)
	bundle, err := writer.SaveKVBlocksToState(context.Background(), store, kv.StateBlockOptions{BlockSize: 2})
	if err != nil {
		fmt.Println("seed error:", err)
		return
	}

	native := &sessionfake.Handle{}
	reader := New(native, spine.ModelInfo{}, nil)
	if err := reader.LoadKVBlocksFromState(context.Background(), store, bundle); err != nil {
		fmt.Println("error:", err)
		return
	}

	fmt.Println("blocks restored:", len(native.RestoredBlocks))
	// Output:
	// blocks restored: 1
}

// ExampleSession_LoadKVBlocksFromMemvid is the deprecated alias for
// LoadKVBlocksFromState — kv.MemvidBlockBundle is a type alias for
// kv.StateBlockBundle.
func ExampleSession_LoadKVBlocksFromMemvid() {
	store := memvid.NewInMemoryStore(nil)
	writer := New(&sessionfake.Handle{KV: sessionfake.TestKVSnapshot()}, spine.ModelInfo{}, nil)
	bundle, err := writer.SaveKVBlocksToState(context.Background(), store, kv.StateBlockOptions{BlockSize: 2})
	if err != nil {
		fmt.Println("seed error:", err)
		return
	}

	native := &sessionfake.Handle{}
	reader := New(native, spine.ModelInfo{}, nil)
	if err := reader.LoadKVBlocksFromMemvid(context.Background(), store, bundle); err != nil {
		fmt.Println("error:", err)
		return
	}

	fmt.Println("blocks restored:", len(native.RestoredBlocks))
	// Output:
	// blocks restored: 1
}

// ExampleSession_LoadKVPrefixBlocksFromState restores only the blocks
// needed to cover prefixTokens, streaming rather than assembling a full
// snapshot.
func ExampleSession_LoadKVPrefixBlocksFromState() {
	store := memvid.NewInMemoryStore(nil)
	writer := New(&sessionfake.Handle{
		KVBlocks: []kv.Block{
			{Index: 0, TokenStart: 0, TokenCount: 2, Snapshot: sessionfake.TestKVSnapshot()},
		},
	}, spine.ModelInfo{}, nil)
	bundle, err := writer.SaveKVBlocksToState(context.Background(), store, kv.StateBlockOptions{BlockSize: 2})
	if err != nil {
		fmt.Println("seed error:", err)
		return
	}

	native := &sessionfake.Handle{}
	reader := New(native, spine.ModelInfo{}, nil)
	if err := reader.LoadKVPrefixBlocksFromState(context.Background(), store, bundle, 2); err != nil {
		fmt.Println("error:", err)
		return
	}

	fmt.Println("blocks restored:", len(native.RestoredBlocks))
	// Output:
	// blocks restored: 1
}

// ExampleSession_LoadKVPrefixBlocksFromMemvid is the deprecated alias for
// LoadKVPrefixBlocksFromState.
func ExampleSession_LoadKVPrefixBlocksFromMemvid() {
	store := memvid.NewInMemoryStore(nil)
	writer := New(&sessionfake.Handle{
		KVBlocks: []kv.Block{
			{Index: 0, TokenStart: 0, TokenCount: 2, Snapshot: sessionfake.TestKVSnapshot()},
		},
	}, spine.ModelInfo{}, nil)
	bundle, err := writer.SaveKVBlocksToState(context.Background(), store, kv.StateBlockOptions{BlockSize: 2})
	if err != nil {
		fmt.Println("seed error:", err)
		return
	}

	native := &sessionfake.Handle{}
	reader := New(native, spine.ModelInfo{}, nil)
	if err := reader.LoadKVPrefixBlocksFromMemvid(context.Background(), store, bundle, 2); err != nil {
		fmt.Println("error:", err)
		return
	}

	fmt.Println("blocks restored:", len(native.RestoredBlocks))
	// Output:
	// blocks restored: 1
}

// ExampleSession_LoadKVFromMemvid is the deprecated alias for
// LoadKVFromState.
func ExampleSession_LoadKVFromMemvid() {
	store := memvid.NewInMemoryStore(nil)
	writer := New(&sessionfake.Handle{KV: sessionfake.TestKVSnapshot()}, spine.ModelInfo{}, nil)
	ref, err := writer.SaveKVToState(context.Background(), store, kv.StateOptions{URI: "mlx://session/load-example-memvid"})
	if err != nil {
		fmt.Println("seed error:", err)
		return
	}

	native := &sessionfake.Handle{}
	reader := New(native, spine.ModelInfo{}, nil)
	if err := reader.LoadKVFromMemvid(context.Background(), store, ref); err != nil {
		fmt.Println("error:", err)
		return
	}

	fmt.Println("restored:", native.RestoredKV != nil)
	// Output:
	// restored: true
}

// ExampleSession_RestoreBundle restores the session from an in-memory
// state bundle whose model identity matches.
func ExampleSession_RestoreBundle() {
	info := spine.ModelInfo{Architecture: "gemma4_text", NumLayers: 1}
	source := New(&sessionfake.Handle{KV: sessionfake.TestKVSnapshot()}, info, nil)
	snapshot, err := source.CaptureKV()
	if err != nil {
		fmt.Println("capture error:", err)
		return
	}
	b, err := mlxbundle.New(snapshot, mlxbundle.Options{Model: "gemma4-e4b"})
	if err != nil {
		fmt.Println("bundle error:", err)
		return
	}

	native := &sessionfake.Handle{}
	target := New(native, info, nil)
	if err := target.RestoreBundle(b); err != nil {
		fmt.Println("error:", err)
		return
	}

	fmt.Println("restored:", native.RestoredKV != nil)
	// Output:
	// restored: true
}

// ExampleSession_RestoreBundleFromState restores the session from a state
// bundle whose KV is held in a State store.
func ExampleSession_RestoreBundleFromState() {
	store := memvid.NewInMemoryStore(nil)
	snapshot := sessionfake.TestKVSnapshot()
	ref, err := snapshot.SaveState(context.Background(), store, kv.StateOptions{})
	if err != nil {
		fmt.Println("save error:", err)
		return
	}
	hash, err := kv.HashSnapshot(snapshot)
	if err != nil {
		fmt.Println("hash error:", err)
		return
	}
	info := spine.ModelInfo{Architecture: "gemma4_text", NumLayers: 1}
	b := &mlxbundle.Bundle{
		Version: mlxbundle.Version,
		Kind:    mlxbundle.Kind,
		Model:   mlxbundle.Model{Architecture: "gemma4_text", NumLayers: 1},
		KVHash:  hash,
		Refs:    []mlxbundle.Ref{{Kind: mlxbundle.RefState, URI: mlxbundle.StateURI(ref), State: ref}},
	}

	native := &sessionfake.Handle{}
	target := New(native, info, nil)
	if err := target.RestoreBundleFromState(context.Background(), b, store); err != nil {
		fmt.Println("error:", err)
		return
	}

	fmt.Println("restored:", native.RestoredKV != nil)
	// Output:
	// restored: true
}

// ExampleSession_RestoreBundleFromMemvid is the deprecated alias for
// RestoreBundleFromState.
func ExampleSession_RestoreBundleFromMemvid() {
	store := memvid.NewInMemoryStore(nil)
	snapshot := sessionfake.TestKVSnapshot()
	ref, err := snapshot.SaveMemvid(context.Background(), store, kv.MemvidOptions{})
	if err != nil {
		fmt.Println("save error:", err)
		return
	}
	hash, err := kv.HashSnapshot(snapshot)
	if err != nil {
		fmt.Println("hash error:", err)
		return
	}
	info := spine.ModelInfo{Architecture: "gemma4_text", NumLayers: 1}
	b := &mlxbundle.Bundle{
		Version: mlxbundle.Version,
		Kind:    mlxbundle.Kind,
		Model:   mlxbundle.Model{Architecture: "gemma4_text", NumLayers: 1},
		KVHash:  hash,
		Refs:    []mlxbundle.Ref{{Kind: mlxbundle.RefMemvid, URI: mlxbundle.MemvidURI(ref), Memvid: ref}},
	}

	native := &sessionfake.Handle{}
	target := New(native, info, nil)
	if err := target.RestoreBundleFromMemvid(context.Background(), b, store); err != nil {
		fmt.Println("error:", err)
		return
	}

	fmt.Println("restored:", native.RestoredKV != nil)
	// Output:
	// restored: true
}

// ExampleSession_LoadBundle reads a state bundle from disk and restores it
// into the session.
func ExampleSession_LoadBundle() {
	info := spine.ModelInfo{Architecture: "gemma4_text", NumLayers: 1}
	source := New(&sessionfake.Handle{KV: sessionfake.TestKVSnapshot()}, info, nil)
	snapshot, err := source.CaptureKV()
	if err != nil {
		fmt.Println("capture error:", err)
		return
	}
	b, err := mlxbundle.New(snapshot, mlxbundle.Options{Model: "gemma4-e4b"})
	if err != nil {
		fmt.Println("bundle error:", err)
		return
	}
	dir := core.MkdirTemp("", "session-loadbundle-example-*").Value.(string)
	path := core.PathJoin(dir, "session.bundle.json")
	if err := b.Save(path); err != nil {
		fmt.Println("save error:", err)
		return
	}

	native := &sessionfake.Handle{}
	target := New(native, info, nil)
	if err := target.LoadBundle(path); err != nil {
		fmt.Println("error:", err)
		return
	}

	fmt.Println("restored:", native.RestoredKV != nil)
	// Output:
	// restored: true
}

// ExampleSession_Fork forks a slept session — the fork starts from the same
// retained state and carries the parent's agent-memory linkage, so its next
// sleep records the parent as its lineage.
func ExampleSession_Fork() {
	ctx := context.Background()
	store := memvid.NewInMemoryStore(nil)
	info := spine.ModelInfo{Architecture: "gemma4_text", NumLayers: 1, QuantBits: 4, ContextLength: 8}

	parentNative := &sessionfake.Handle{
		KV:     sessionfake.TestKVSnapshot(),
		Forked: &sessionfake.Handle{KV: sessionfake.TestKVSnapshot()},
	}
	parent := New(parentNative, info, nil)
	parentSleep, err := parent.Sleep(ctx, store, agent.SleepOptions{
		EntryURI:     "mlx://agent/parent",
		BlockOptions: kv.StateBlockOptions{BlockSize: 1},
	})
	if err != nil {
		fmt.Println("parent sleep error:", err)
		return
	}

	forked, err := parent.Fork()
	if err != nil {
		fmt.Println("fork error:", err)
		return
	}
	childSleep, err := forked.Sleep(ctx, store, agent.SleepOptions{
		EntryURI:     "mlx://agent/child",
		BlockOptions: kv.StateBlockOptions{BlockSize: 1},
	})
	if err != nil {
		fmt.Println("child sleep error:", err)
		return
	}

	fmt.Println("child entry:", childSleep.EntryURI)
	fmt.Println("inherited parent:", childSleep.ParentEntryURI == parentSleep.EntryURI)
	// Output:
	// child entry: mlx://agent/child
	// inherited parent: true
}

// ExampleSession_Reset releases retained state and leaves the session ready
// for another prefill.
func ExampleSession_Reset() {
	native := &sessionfake.Handle{}
	sess := New(native, spine.ModelInfo{}, nil)

	sess.Reset()

	fmt.Println("reset calls:", native.ResetCalls)
	// Output:
	// reset calls: 1
}

// ExampleSession_Close releases retained session state.
func ExampleSession_Close() {
	native := &sessionfake.Handle{}
	sess := New(native, spine.ModelInfo{}, nil)

	err := sess.Close()

	fmt.Println("error:", err)
	fmt.Println("valid after close:", sess.Valid())
	// Output:
	// error: <nil>
	// valid after close: false
}

// ExampleSession_Err returns the last session error.
func ExampleSession_Err() {
	wantErr := core.NewError("native err value")
	sess := New(&sessionfake.Handle{ErrValue: wantErr}, spine.ModelInfo{}, nil)

	fmt.Println("error:", sess.Err())
	// Output:
	// error: native err value
}
