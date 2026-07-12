// SPDX-Licence-Identifier: EUPL-1.2

// Package prefixindex is the global token-prefix map behind cross-conversation
// KV sharing. Conversation identity in the serve loop is a whole-message hash
// (serving/continuity's conversationKey), so two conversations that open with
// the same system prompt can never cross-match by construction — each hashes to
// a different key. This index closes that gap on the token axis: it maps a
// conversation's framed-prompt TOKEN sequence to the durable state bundle that
// holds its KV, so a fresh conversation can find the longest leading run it
// shares with any conversation already served and wake that span instead of
// re-prefilling it.
//
// It is a thin, concurrency-safe skin over kv/radix: publish inserts a
// conversation's prompt tokens, match resolves the longest shared prefix via
// radix.LongestPrefix (which counts a partially-matched edge, so the SECOND
// conversation shares the first's prompt before any split node exists), and
// evict drops an entry whose backing state has been reclaimed. The index holds
// only URIs and sizes — never KV bytes and never a store handle — so staleness
// is the consumer's call: a wake that fails against a matched bundle evicts the
// entry and falls back to a fresh prefill, never to a wrong graft.
//
//	ix := prefixindex.New(prefixindex.Config{MaxEntries: 4096})
//	ix.Publish(promptTokensA, prefixindex.Entry{BundleURI: uriA, BlockSize: 512, TokenCount: len(promptTokensA)})
//	if e, shared, ok := ix.Match(promptTokensB); ok {
//	    // wake e.BundleURI's first (shared, block-aligned) tokens into B
//	}
package prefixindex

import (
	"sync"

	"dappco.re/go/inference/kv/radix"
)

// Entry locates a shared token prefix's KV: the durable state bundle that holds
// it, the bundle's KV block size (the granularity a graft must align to), and
// the shareable prefix length in tokens (the bundle covers at least this many,
// so a match never asks for more than the bundle has).
type Entry struct {
	BundleURI  string
	BlockSize  int
	TokenCount int
}

// Config tunes one index. MaxEntries bounds the radix node count; older entries
// are evicted LRU past it. A value <= 0 leaves the index unbounded.
type Config struct {
	MaxEntries int
}

// Index is a cross-conversation token-prefix → durable-state map. The zero
// value is not usable — construct with New. Safe for concurrent use.
type Index struct {
	mu   sync.Mutex
	tree *radix.Tree
}

// New builds an empty index with the given capacity bound.
//
//	ix := prefixindex.New(prefixindex.Config{MaxEntries: 4096})
func New(cfg Config) *Index {
	return &Index{tree: radix.New(radix.Config{MaxNodes: cfg.MaxEntries})}
}

// Publish records tokens as a shareable prefix backed by e, then trims the
// index to capacity. An empty token run or an incomplete entry (no bundle URI,
// non-positive block size, or non-positive token count) is ignored — a graft
// must have all three to wake correctly, so a partial entry is never indexed.
// Re-publishing the same tokens updates the backing entry in place.
//
//	ix.Publish(promptTokens, prefixindex.Entry{BundleURI: uri, BlockSize: 512, TokenCount: len(promptTokens)})
func (ix *Index) Publish(tokens []int32, e Entry) {
	if len(tokens) == 0 || e.BundleURI == "" || e.BlockSize <= 0 || e.TokenCount <= 0 {
		return
	}
	ix.mu.Lock()
	defer ix.mu.Unlock()
	ix.tree.Insert(toInts(tokens), e)
	ix.tree.EvictToCapacity()
}

// Match returns the entry backing the longest shared leading run of tokens and
// that run's length, or ok=false when nothing shares even a first token. The
// returned length is the raw shared token count — the caller aligns it down to
// the entry's block size before waking, so a graft only ever adopts whole KV
// blocks that lie strictly inside the verified shared run.
//
//	if e, shared, ok := ix.Match(promptTokens); ok { … }
func (ix *Index) Match(tokens []int32) (Entry, int, bool) {
	if len(tokens) == 0 {
		return Entry{}, 0, false
	}
	ix.mu.Lock()
	defer ix.mu.Unlock()
	node, matched, ok := ix.tree.LongestPrefix(toInts(tokens))
	if !ok {
		return Entry{}, 0, false
	}
	entry, ok := node.Value.(Entry)
	if !ok {
		return Entry{}, 0, false
	}
	return entry, matched, true
}

// Evict drops the entry a Match of tokens would resolve to, but only when it
// still names bundleURI — so a stale wake removes exactly the dead entry and
// never a healthy sibling that happens to share the prefix. Reports whether an
// entry was removed. A no-op (false) when the resolved entry names a different
// bundle or is an internal node the radix keeps; either way the next match
// re-detects staleness and the consumer stays on the safe fresh-prefill path.
//
//	if _, err := wake(e.BundleURI); err != nil { ix.Evict(promptTokens, e.BundleURI) }
func (ix *Index) Evict(tokens []int32, bundleURI string) bool {
	if len(tokens) == 0 || bundleURI == "" {
		return false
	}
	ix.mu.Lock()
	defer ix.mu.Unlock()
	node, _, ok := ix.tree.LongestPrefix(toInts(tokens))
	if !ok {
		return false
	}
	if entry, ok := node.Value.(Entry); !ok || entry.BundleURI != bundleURI {
		return false
	}
	return ix.tree.EvictNode(node)
}

// Len reports how many prefixes the index holds (radix node count) — for tests
// and diagnostics.
func (ix *Index) Len() int {
	ix.mu.Lock()
	defer ix.mu.Unlock()
	return ix.tree.Len()
}

// toInts widens the model-native int32 token IDs to the []int the radix keys on.
func toInts(tokens []int32) []int {
	out := make([]int, len(tokens))
	for i, t := range tokens {
		out[i] = int(t)
	}
	return out
}
