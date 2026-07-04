// SPDX-Licence-Identifier: EUPL-1.2

package inference

import (
	"context"
	"iter"

	"dappco.re/go/inference/kv"
)

// KV-state capabilities — the engine-neutral contracts for capturing and
// restoring conversation state as [kv.Snapshot] (the portable wire shape;
// docs/engine-merge.md Tier 1). Engines implement these directly in
// kv.Snapshot terms — no per-engine snapshot type, no converter layer.
// Probe them off a loaded model like [AttentionInspector]:
//
//	if s, ok := model.(inference.KVSnapshotter); ok {
//	    snap, err := s.CaptureKV(ctx, prompt, inference.KVSnapshotCaptureOptions{})
//	    _ = snap; _ = err
//	}

// KVSnapshotCaptureOptions tunes a KV capture. The zero value is the default
// capture (full float32 retention, all blocks).
type KVSnapshotCaptureOptions struct {
	// RawKVOnly captures native K/V dtype bytes without retaining float32
	// key/value slices — smaller snapshots when the consumer only grafts.
	RawKVOnly bool
	// BlockStartToken skips capture of KV blocks that end at or before this
	// token — the trusted-prefix lane: blocks the parent bundle already
	// holds are grafted by reference downstream, so re-capturing and
	// re-hashing them per turn scales with the conversation, not the turn.
	BlockStartToken int
}

// KVSnapshotter captures the model's KV state after prefilling a prompt.
type KVSnapshotter interface {
	// CaptureKV prefills the prompt and returns the resulting KV state.
	// A zero-value opts is the default capture.
	CaptureKV(ctx context.Context, prompt string, opts KVSnapshotCaptureOptions) (*kv.Snapshot, error)
}

// KVChunkSnapshotter captures KV state from a prompt supplied as ordered
// chunks — the streaming/prefill-reuse lane.
type KVChunkSnapshotter interface {
	// CaptureKVChunks prefills the chunks in order and returns the KV state.
	CaptureKVChunks(ctx context.Context, chunks iter.Seq[string], opts KVSnapshotCaptureOptions) (*kv.Snapshot, error)
}

// KVRestorer restores previously captured KV state into the model's prompt
// cache, so the next generation continues from the snapshot instead of
// re-prefilling the conversation.
type KVRestorer interface {
	// RestoreFromKV loads the snapshot into the model's cache.
	RestoreFromKV(ctx context.Context, snapshot *kv.Snapshot) error
}

// PromptCacheWarmer prefills the prompt cache without generating.
type PromptCacheWarmer interface {
	// WarmPromptCache runs prefill for the prompt and retains the cache.
	WarmPromptCache(ctx context.Context, prompt string) error
}

// PromptCacheChunkWarmer prefills the prompt cache from ordered chunks.
type PromptCacheChunkWarmer interface {
	// WarmPromptCacheChunks runs prefill over the chunks in order.
	WarmPromptCacheChunks(ctx context.Context, chunks iter.Seq[string]) error
}

// PromptCacheClearer drops any retained prompt cache.
type PromptCacheClearer interface {
	// ClearPromptCache releases the retained cache state.
	ClearPromptCache()
}
