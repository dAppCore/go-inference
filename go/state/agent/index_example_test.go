// SPDX-Licence-Identifier: EUPL-1.2

package agent

import (
	"context"
	"fmt"

	pkgbundle "dappco.re/go/inference/bundle"
	"dappco.re/go/inference/kv"
	"dappco.re/go/inference/kv/memory"
	state "dappco.re/go/inference/state"
)

// exampleIndexBundle builds a small two-block durable bundle used by the
// index examples below. Synthetic, no model load: four tokens split into
// two equal blocks, enough to carve named chapter spans over.
func exampleIndexBundle() *kv.StateBlockBundle {
	return &kv.StateBlockBundle{
		Version:      kv.MemvidBlockVersion,
		Kind:         kv.MemvidBlockBundleKind,
		SnapshotHash: "snapshot",
		KVEncoding:   kv.EncodingNative,
		Architecture: "gemma4_text",
		TokenCount:   4,
		BlockSize:    2,
		NumLayers:    1,
		NumHeads:     1,
		SeqLen:       4,
		HeadDim:      2,
		Blocks: []kv.StateBlockRef{
			{Index: 0, TokenStart: 0, TokenCount: 2},
			{Index: 1, TokenStart: 2, TokenCount: 2},
		},
	}
}

// ExampleNewStateIndex builds an index over a durable bundle with two named
// chapter spans and reports the prefix length needed to wake the longest.
func ExampleNewStateIndex() {
	index, err := NewStateIndex(exampleIndexBundle(), StateIndexOptions{
		BundleURI: "mlx://book/bundle",
		Title:     "full book",
		Entries: []StateIndexEntry{
			{URI: "mlx://book/chapter-1", Title: "Chapter 1", TokenStart: 0, TokenCount: 2},
			{URI: "mlx://book/chapter-2", Title: "Chapter 2", TokenStart: 2, TokenCount: 2},
		},
	})
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	fmt.Println(index.Kind)
	fmt.Println(len(index.Entries))
	fmt.Println(index.RequiredContextLength())
	// Output:
	// go-mlx/kv-snapshot-bundle-index
	// 2
	// 4
}

// ExampleNewStateIndex_defaultEntry shows that an index built with no
// explicit entries gets a single full-bundle entry covering every token.
func ExampleNewStateIndex_defaultEntry() {
	index, err := NewStateIndex(exampleIndexBundle(), StateIndexOptions{BundleURI: "mlx://book/bundle"})
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	fmt.Println(len(index.Entries))
	fmt.Println(index.Entries[0].URI)
	fmt.Println(index.Entries[0].TokenCount)
	// Output:
	// 1
	// mlx://book/bundle
	// 4
}

// ExampleStateIndex_Entry shows that Entry returns a defensive copy keyed by
// URI: mutating the returned entry leaves the index untouched.
func ExampleStateIndex_Entry() {
	index, _ := NewStateIndex(exampleIndexBundle(), StateIndexOptions{
		BundleURI: "mlx://book/bundle",
		Entries: []StateIndexEntry{
			{URI: "mlx://book/chapter-1", Title: "Chapter 1", TokenStart: 0, TokenCount: 2, Labels: []string{"chapter"}},
		},
	})
	entry, ok := index.Entry("mlx://book/chapter-1")
	fmt.Println(ok)
	entry.Labels[0] = "mutated"
	again, _ := index.Entry("mlx://book/chapter-1")
	fmt.Println(again.Labels[0])
	_, missing := index.Entry("mlx://book/nope")
	fmt.Println(missing)
	// Output:
	// true
	// chapter
	// false
}

// ExampleStateIndexEntry_PrefixTokens shows the prefix length an entry needs
// restored: every token up to and including the entry's own span.
func ExampleStateIndexEntry_PrefixTokens() {
	entry := StateIndexEntry{TokenStart: 2, TokenCount: 2}
	fmt.Println(entry.PrefixTokens())
	// Output:
	// 4
}

// ExampleCheckStateIndexCompatibility verifies an index against the model and
// tokenizer identity it was built for; matching identity returns no error.
func ExampleCheckStateIndexCompatibility() {
	info := memory.ModelInfo{Architecture: "gemma4_text", NumLayers: 1, QuantBits: 4, ContextLength: 8}
	tok := pkgbundle.Tokenizer{Hash: "tok-a", ChatTemplateHash: "chat-a"}
	index, _ := NewStateIndex(exampleIndexBundle(), StateIndexOptions{
		BundleURI: "mlx://book/bundle",
		ModelInfo: info,
		Tokenizer: tok,
		Entries:   []StateIndexEntry{{URI: "mlx://book/chapter-1", TokenStart: 0, TokenCount: 2}},
	})
	err := CheckStateIndexCompatibility(info, tok, index)
	fmt.Println(err)
	// Output:
	// <nil>
}

// ExampleSaveStateIndex_roundTrip stores an index then reloads it by URI from
// the same in-memory State store, recovering the same span.
func ExampleSaveStateIndex_roundTrip() {
	ctx := context.Background()
	store := state.NewInMemoryStore(nil)
	index, _ := NewStateIndex(exampleIndexBundle(), StateIndexOptions{
		BundleURI: "mlx://book/bundle",
		Entries:   []StateIndexEntry{{URI: "mlx://book/chapter-1", Title: "Chapter 1", TokenStart: 0, TokenCount: 2}},
	})
	if _, err := SaveStateIndex(ctx, store, index, "mlx://book/index"); err != nil {
		fmt.Println("save:", err)
		return
	}
	loaded, err := LoadStateIndex(ctx, store, "mlx://book/index")
	if err != nil {
		fmt.Println("load:", err)
		return
	}
	fmt.Println(loaded.Entries[0].URI)
	fmt.Println(loaded.Hash == index.Hash)
	// Output:
	// mlx://book/chapter-1
	// true
}

// ExampleLoadPrefixFromStateIndex resolves a named chapter through a saved
// index and restores only the KV prefix that chapter needs — here the first
// two tokens of a four-token bundle.
func ExampleLoadPrefixFromStateIndex() {
	ctx := context.Background()
	store := state.NewInMemoryStore(nil)
	snapshot := kvSnapshotBlocksTestSnapshot()
	blk, err := snapshot.SaveStateBlocks(ctx, store, kv.StateBlockOptions{BlockSize: 2, KVEncoding: kv.EncodingNative})
	if err != nil {
		fmt.Println("blocks:", err)
		return
	}
	if _, err := kv.SaveStateBlockBundle(ctx, store, blk, "mlx://book/bundle"); err != nil {
		fmt.Println("bundle:", err)
		return
	}
	index, err := NewStateIndex(blk, StateIndexOptions{
		BundleURI: "mlx://book/bundle",
		Entries:   []StateIndexEntry{{URI: "mlx://book/chapter-1", TokenStart: 0, TokenCount: 2}},
	})
	if err != nil {
		fmt.Println("index:", err)
		return
	}
	prefix, entry, err := LoadPrefixFromStateIndex(ctx, store, index, "mlx://book/chapter-1", kv.LoadOptions{RawKVOnly: true})
	if err != nil {
		fmt.Println("prefix:", err)
		return
	}
	fmt.Println(entry.URI)
	fmt.Println(len(prefix.Tokens))
	// Output:
	// mlx://book/chapter-1
	// 2
}

// ExampleNewMemvidIndex shows the deprecated constructor forwarding to
// NewStateIndex: the same full-bundle entry and canonical kind result.
func ExampleNewMemvidIndex() {
	index, err := NewMemvidIndex(exampleIndexBundle(), MemvidIndexOptions{BundleURI: "mlx://book/bundle"})
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	fmt.Println(index.Kind)
	fmt.Println(len(index.Entries))
	fmt.Println(index.Entries[0].TokenCount)
	// Output:
	// go-mlx/kv-snapshot-bundle-index
	// 1
	// 4
}

// ExampleStateIndex_Validate shows that a freshly built index validates
// clean, while one whose kind is then tampered with is rejected.
func ExampleStateIndex_Validate() {
	index, _ := NewStateIndex(exampleIndexBundle(), StateIndexOptions{
		BundleURI: "mlx://book/bundle",
		Entries:   []StateIndexEntry{{URI: "mlx://book/chapter-1", TokenStart: 0, TokenCount: 2}},
	})
	fmt.Println(index.Validate() == nil)
	index.Kind = "tampered"
	fmt.Println(index.Validate() == nil)
	// Output:
	// true
	// false
}

// ExampleStateIndex_RequiredContextLength shows that the required context is
// the longest prefix any entry needs — here the second chapter ending at
// token four.
func ExampleStateIndex_RequiredContextLength() {
	index, _ := NewStateIndex(exampleIndexBundle(), StateIndexOptions{
		BundleURI: "mlx://book/bundle",
		Entries: []StateIndexEntry{
			{URI: "mlx://book/chapter-1", TokenStart: 0, TokenCount: 2},
			{URI: "mlx://book/chapter-2", TokenStart: 2, TokenCount: 2},
		},
	})
	fmt.Println(index.RequiredContextLength())
	// Output:
	// 4
}

// ExampleSaveMemvidIndex stores an index through the deprecated wrapper and
// reloads it, recovering the same span URI.
func ExampleSaveMemvidIndex() {
	ctx := context.Background()
	store := state.NewInMemoryStore(nil)
	index, _ := NewMemvidIndex(exampleIndexBundle(), MemvidIndexOptions{
		BundleURI: "mlx://book/bundle",
		Entries:   []MemvidIndexEntry{{URI: "mlx://book/chapter-1", TokenStart: 0, TokenCount: 2}},
	})
	if _, err := SaveMemvidIndex(ctx, store, index, "mlx://book/index"); err != nil {
		fmt.Println("save:", err)
		return
	}
	loaded, err := LoadMemvidIndex(ctx, store, "mlx://book/index")
	if err != nil {
		fmt.Println("load:", err)
		return
	}
	fmt.Println(loaded.Entries[0].URI)
	// Output:
	// mlx://book/chapter-1
}

// ExampleLoadStateIndex restores a previously saved index by URI and shows
// the reloaded copy carries the same hash as the original.
func ExampleLoadStateIndex() {
	ctx := context.Background()
	store := state.NewInMemoryStore(nil)
	index, _ := NewStateIndex(exampleIndexBundle(), StateIndexOptions{
		BundleURI: "mlx://book/bundle",
		Entries:   []StateIndexEntry{{URI: "mlx://book/chapter-1", TokenStart: 0, TokenCount: 2}},
	})
	if _, err := SaveStateIndex(ctx, store, index, "mlx://book/index"); err != nil {
		fmt.Println("save:", err)
		return
	}
	loaded, err := LoadStateIndex(ctx, store, "mlx://book/index")
	if err != nil {
		fmt.Println("load:", err)
		return
	}
	fmt.Println(loaded.Kind)
	fmt.Println(loaded.Hash == index.Hash)
	// Output:
	// go-mlx/kv-snapshot-bundle-index
	// true
}

// ExampleLoadMemvidIndex restores an index saved through the deprecated
// wrapper, recovering the same span.
func ExampleLoadMemvidIndex() {
	ctx := context.Background()
	store := state.NewInMemoryStore(nil)
	index, _ := NewMemvidIndex(exampleIndexBundle(), MemvidIndexOptions{
		BundleURI: "mlx://book/bundle",
		Entries:   []MemvidIndexEntry{{URI: "mlx://book/chapter-1", TokenStart: 0, TokenCount: 2}},
	})
	if _, err := SaveMemvidIndex(ctx, store, index, "mlx://book/index"); err != nil {
		fmt.Println("save:", err)
		return
	}
	loaded, err := LoadMemvidIndex(ctx, store, "mlx://book/index")
	if err != nil {
		fmt.Println("load:", err)
		return
	}
	fmt.Println(loaded.Entries[0].URI)
	// Output:
	// mlx://book/chapter-1
}

// ExampleLoadPrefixFromMemvidIndex resolves a named chapter through a saved
// index via the deprecated wrapper and restores just that chapter's prefix.
func ExampleLoadPrefixFromMemvidIndex() {
	ctx := context.Background()
	store := state.NewInMemoryStore(nil)
	snapshot := kvSnapshotBlocksTestSnapshot()
	blk, err := snapshot.SaveStateBlocks(ctx, store, kv.StateBlockOptions{BlockSize: 2, KVEncoding: kv.EncodingNative})
	if err != nil {
		fmt.Println("blocks:", err)
		return
	}
	if _, err := kv.SaveStateBlockBundle(ctx, store, blk, "mlx://book/bundle"); err != nil {
		fmt.Println("bundle:", err)
		return
	}
	index, err := NewMemvidIndex(blk, MemvidIndexOptions{
		BundleURI: "mlx://book/bundle",
		Entries:   []MemvidIndexEntry{{URI: "mlx://book/chapter-1", TokenStart: 0, TokenCount: 2}},
	})
	if err != nil {
		fmt.Println("index:", err)
		return
	}
	prefix, entry, err := LoadPrefixFromMemvidIndex(ctx, store, index, "mlx://book/chapter-1", kv.LoadOptions{RawKVOnly: true})
	if err != nil {
		fmt.Println("prefix:", err)
		return
	}
	fmt.Println(entry.URI)
	fmt.Println(len(prefix.Tokens))
	// Output:
	// mlx://book/chapter-1
	// 2
}

// ExampleCheckMemvidIndexCompatibility verifies an index against the model
// and tokenizer identity it was built for through the deprecated wrapper;
// matching identity returns no error.
func ExampleCheckMemvidIndexCompatibility() {
	info := memory.ModelInfo{Architecture: "gemma4_text", NumLayers: 1, QuantBits: 4, ContextLength: 8}
	tok := pkgbundle.Tokenizer{Hash: "tok-a"}
	index, _ := NewMemvidIndex(exampleIndexBundle(), MemvidIndexOptions{
		BundleURI: "mlx://book/bundle",
		ModelInfo: info,
		Tokenizer: tok,
		Entries:   []MemvidIndexEntry{{URI: "mlx://book/chapter-1", TokenStart: 0, TokenCount: 2}},
	})
	fmt.Println(CheckMemvidIndexCompatibility(info, tok, index))
	// Output:
	// <nil>
}
