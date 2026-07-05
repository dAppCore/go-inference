// SPDX-Licence-Identifier: EUPL-1.2

package bundle

import (
	"context"

	core "dappco.re/go"
	"dappco.re/go/inference/kv"
	state "dappco.re/go/inference/model/state"
	"dappco.re/go/inference/train/lora"
)

func ExampleNew() {
	b, err := New(exampleBundleSnapshot(), Options{
		Model:  "gemma4-e2b",
		Source: ModelInfo{Architecture: "gemma4_text", NumLayers: 1, ContextLength: 262144},
		Prompt: "draft the next section",
		Adapter: Adapter{Name: "outline-lora", Rank: 2, Alpha: 4, TargetKeys: []string{
			"q_proj",
			"v_proj",
		}},
	})
	if err != nil {
		core.Println(err)
		return
	}

	core.Println(b.Kind, b.Model.Architecture, b.Prompt.TokenCount, b.Adapter.TargetKeys)
	// Output: go-mlx/state-bundle gemma4_text 3 [q_proj v_proj]
}

func ExampleLoad() {
	bundlePath, cleanup, ok := exampleBundlePath()
	if !ok {
		return
	}
	defer cleanup()

	loaded, err := Load(bundlePath)
	core.Println(err == nil, loaded.Model.Name, loaded.KVHash != "")
	// Output: true gemma4-e2b true
}

func ExampleBundle_Save() {
	b, err := New(exampleBundleSnapshot(), Options{Model: "gemma4-e2b", Source: ModelInfo{Architecture: "gemma4_text"}})
	if err != nil {
		core.Println(err)
		return
	}
	dir, cleanup, ok := exampleBundleTempDir()
	if !ok {
		return
	}
	defer cleanup()

	path := core.PathJoin(dir, "state.bundle.json")
	err = b.Save(path)
	read := core.ReadFile(path)
	data := ""
	if read.OK {
		data = string(read.Value.([]byte))
	}

	core.Println(err == nil, core.Contains(data, "\"kind\": \"go-mlx/state-bundle\""))
	// Output: true true
}

func ExampleBundle_SaveCompact() {
	b, err := New(exampleBundleSnapshot(), Options{Model: "gemma4-e2b", Source: ModelInfo{Architecture: "gemma4_text"}})
	if err != nil {
		core.Println(err)
		return
	}
	dir, cleanup, ok := exampleBundleTempDir()
	if !ok {
		return
	}
	defer cleanup()

	path := core.PathJoin(dir, "compact.bundle.json")
	err = b.SaveCompact(path)
	read := core.ReadFile(path)
	data := ""
	if read.OK {
		data = string(read.Value.([]byte))
	}
	// Compact JSON has no indenting newline before "kind", and round-trips
	// through Load to the same bundle.
	loaded, loadErr := Load(path)

	core.Println(err == nil, core.Contains(data, "\n  \"kind\""), loadErr == nil, loaded.Model.Architecture)
	// Output: true false true gemma4_text
}

func ExampleBundle_Snapshot() {
	b, err := New(exampleBundleSnapshot(), Options{Model: "gemma4-e2b"})
	if err != nil {
		core.Println(err)
		return
	}
	snapshot, err := b.Snapshot()
	if err != nil {
		core.Println(err)
		return
	}
	snapshot.Tokens[0] = 99
	again, _ := b.Snapshot()

	core.Println(again.Architecture, again.Tokens[0], again.TokenOffset)
	// Output: gemma4_text 10 3
}

func ExampleBundle_SnapshotFromState() {
	store := state.NewInMemoryStore(nil)
	snapshot := exampleBundleSnapshot()
	ref, err := snapshot.SaveState(context.Background(), store, kv.StateOptions{})
	if err != nil {
		core.Println(err)
		return
	}
	hash, err := kv.HashSnapshot(snapshot)
	if err != nil {
		core.Println(err)
		return
	}
	b := &Bundle{
		Version: Version, Kind: Kind, KVHash: hash,
		Refs: []Ref{{Kind: RefState, URI: StateURI(ref), State: ref}},
	}
	loaded, err := b.SnapshotFromState(context.Background(), store)
	if err != nil {
		core.Println(err)
		return
	}

	core.Println(loaded.Architecture, loaded.TokenOffset)
	// Output: gemma4_text 3
}

func ExampleBundle_SnapshotFromMemvid() {
	b, err := New(exampleBundleSnapshot(), Options{Model: "gemma4-e2b"})
	if err != nil {
		core.Println(err)
		return
	}
	snapshot, err := b.SnapshotFromMemvid(context.Background(), nil)
	if err != nil {
		core.Println(err)
		return
	}

	core.Println(snapshot.Architecture, len(snapshot.Tokens))
	// Output: gemma4_text 3
}

func ExampleBundle_Validate() {
	b, err := New(exampleBundleSnapshot(), Options{Model: "gemma4-e2b"})
	if err != nil {
		core.Println(err)
		return
	}
	core.Println(b.Validate() == nil)
	b.Kind = "other"
	core.Println(b.Validate() != nil)
	// Output:
	// true
	// true
}

func ExampleCheckCompatibility() {
	b, err := New(exampleBundleSnapshot(), Options{
		Model:   "gemma4-e2b",
		Source:  ModelInfo{Architecture: "gemma4_text", NumLayers: 1},
		Adapter: Adapter{Name: "outline-lora", Path: "/adapters/outline", Rank: 2, Alpha: 4},
	})
	if err != nil {
		core.Println(err)
		return
	}
	active := ModelInfo{Architecture: "gemma4_text", NumLayers: 1, Adapter: AdapterToInfo(b.Adapter)}
	missingAdapter := ModelInfo{Architecture: "gemma4_text", NumLayers: 1}

	core.Println(CheckCompatibility(active, b) == nil, CheckCompatibility(missingAdapter, b) != nil)
	// Output: true true
}

func ExampleFileHash() {
	dir, cleanup, ok := exampleBundleTempDir()
	if !ok {
		return
	}
	defer cleanup()
	path := core.PathJoin(dir, "tokenizer.json")
	if result := core.WriteFile(path, []byte(`{"model":"bpe"}`), 0o600); !result.OK {
		return
	}

	hash, err := FileHash(path)
	core.Println(err == nil, len(hash), hash == HashString(`{"model":"bpe"}`))
	// Output: true 64 true
}

func ExampleNormaliseTokenizer() {
	tokenizer := NormaliseTokenizer(Tokenizer{
		Path:         "/models/gemma4/tokenizer.json",
		ChatTemplate: "<|turn>user\n{{content}}<turn|>",
	})
	core.Println(tokenizer.Hash != "", tokenizer.ChatTemplateHash != "")
	// Output: true true
}

func ExampleAdapterEmpty() {
	core.Println(
		AdapterEmpty(Adapter{}),
		AdapterEmpty(Adapter{Name: "domain-lora"}),
		AdapterEmpty(Adapter{TargetKeys: []string{"q_proj"}}),
	)
	// Output: true false false
}

func ExampleAdapterFromInfo() {
	info := lora.AdapterInfo{
		Name:       "domain-lora",
		Path:       "/adapters/domain",
		Hash:       "abc123",
		Rank:       8,
		Alpha:      16,
		Scale:      2,
		TargetKeys: []string{"q_proj", "v_proj"},
	}
	adapter := AdapterFromInfo(info)

	core.Println(adapter.Name, adapter.Path, adapter.Rank, adapter.Alpha, adapter.Scale, adapter.TargetKeys)
	// Output: domain-lora /adapters/domain 8 16 2 [q_proj v_proj]
}

func ExampleAdapterToInfo() {
	adapter := Adapter{
		Name:       "domain-lora",
		Path:       "/adapters/domain",
		Hash:       "abc123",
		Rank:       8,
		Alpha:      16,
		Scale:      2,
		TargetKeys: []string{"q_proj", "v_proj"},
	}
	info := AdapterToInfo(adapter)
	adapter.TargetKeys[0] = "mutated"

	core.Println(info.Name, info.Path, info.Rank, info.Alpha, info.Scale, info.TargetKeys)
	// Output: domain-lora /adapters/domain 8 16 2 [q_proj v_proj]
}

func ExampleHashString() {
	core.Println(len(HashString("gemma4")), HashString("") == "")
	// Output: 64 true
}

func ExampleStateURI() {
	core.Println(StateURI(state.ChunkRef{Segment: "/tmp/trace.mp4", ChunkID: 42}))
	core.Println(StateURI(state.ChunkRef{ChunkID: 7}))
	// Output:
	// state:///tmp/trace.mp4#chunk=42
	// state://chunk/7
}

func ExampleMemvidURI() {
	core.Println(MemvidURI(state.ChunkRef{Segment: "session.mp4", ChunkID: 7}))
	// Output: memvid://session.mp4#chunk=7
}

func exampleBundleSnapshot() *kv.Snapshot {
	return &kv.Snapshot{
		Version:       kv.SnapshotVersion,
		Architecture:  "gemma4_text",
		Tokens:        []int32{10, 11, 12},
		Generated:     []int32{12},
		TokenOffset:   3,
		NumLayers:     1,
		NumHeads:      1,
		SeqLen:        3,
		HeadDim:       2,
		NumQueryHeads: 8,
		LogitShape:    []int32{1, 1, 4},
		Logits:        []float32{0.1, 0.2, 0.3, 0.4},
		Layers: []kv.LayerSnapshot{{
			Layer:      0,
			CacheIndex: 0,
			Heads: []kv.HeadSnapshot{{
				Key:   []float32{1, 0, 0, 1, 1, 1},
				Value: []float32{0, 1, 1, 0, 1, 1},
			}},
		}},
	}
}

func exampleBundlePath() (string, func(), bool) {
	dir, cleanup, ok := exampleBundleTempDir()
	if !ok {
		return "", cleanup, false
	}
	b, err := New(exampleBundleSnapshot(), Options{
		Model:  "gemma4-e2b",
		Source: ModelInfo{Architecture: "gemma4_text", NumLayers: 1},
	})
	if err != nil {
		cleanup()
		return "", func() {}, false
	}
	path := core.PathJoin(dir, "state.bundle.json")
	if err := b.Save(path); err != nil {
		cleanup()
		return "", func() {}, false
	}
	return path, cleanup, true
}

func exampleBundleTempDir() (string, func(), bool) {
	dirResult := core.MkdirTemp("", "go-mlx-bundle-example-*")
	if !dirResult.OK {
		return "", func() {}, false
	}
	dir := dirResult.Value.(string)
	return dir, func() { core.RemoveAll(dir) }, true
}
