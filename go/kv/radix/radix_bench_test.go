// SPDX-Licence-Identifier: EUPL-1.2

package radix_test

import (
	"testing"

	"dappco.re/go/inference/kv/radix"
)

// --- corpus ----------------------------------------------------------------

// lcg is a tiny deterministic generator so token corpora are reproducible
// without importing math/rand or touching the timed region.
type lcg uint64

func (s *lcg) next() int {
	*s = *s*6364136223846793005 + 1442695040888963407
	return int(*s >> 33)
}

// corpus builds groups of keys that share a leading prefix and diverge in the
// tail — the prompt-cache shape (many requests share a long system prompt, then
// fork per user turn). tail[0] is a unique counter so every key is distinct and
// each group's shared prefix fans out into perGroup leaves. Returned outside any
// timed region, so its own allocations never colour a benchmark.
func corpus(groups, perGroup, prefixLen, tailLen int) [][]int {
	var s lcg = 0x9e3779b97f4a7c15
	keys := make([][]int, 0, groups*perGroup)
	uniq := 0
	for g := 0; g < groups; g++ {
		prefix := make([]int, prefixLen)
		for i := range prefix {
			prefix[i] = s.next() & 0xffff
		}
		for p := 0; p < perGroup; p++ {
			k := make([]int, 0, prefixLen+tailLen)
			k = append(k, prefix...)
			uniq++
			k = append(k, uniq) // unique divergence token → distinct keys
			for i := 1; i < tailLen; i++ {
				k = append(k, s.next()&0xffff)
			}
			keys = append(keys, k)
		}
	}
	return keys
}

// payload is pre-boxed once so passing it to Insert's any parameter never boxes
// inside a timed loop — the benchmarks measure the package, not the harness.
var payload any = "kv"

// package-level sinks defeat dead-code elimination.
var (
	sinkNode *radix.Node
	sinkInt  int
	sinkBool bool
)

// prefill loads every key into tr. Used outside the timed region.
func prefill(tr *radix.Tree, keys [][]int) {
	for _, k := range keys {
		tr.Insert(k, payload)
	}
}

// --- Match -----------------------------------------------------------------

// BenchmarkMatch_Hit walks full keys to their leaves over a shared-prefix tree —
// the cache-hit path. Lookup must not allocate.
func BenchmarkMatch_Hit(b *testing.B) {
	keys := corpus(64, 32, 48, 16)
	tr := radix.New(radix.Config{MaxNodes: 0})
	prefill(tr, keys)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		n, m := tr.Match(keys[i%len(keys)])
		sinkNode, sinkInt = n, m
	}
}

// BenchmarkMatch_Partial queries prefixes that descend the shared head and then
// diverge mid-edge — the longest-partial-prefix path. Must not allocate.
func BenchmarkMatch_Partial(b *testing.B) {
	keys := corpus(64, 32, 48, 16)
	tr := radix.New(radix.Config{MaxNodes: 0})
	prefill(tr, keys)

	// Queries that share the 48-token prefix but break at the divergence token.
	q := make([][]int, len(keys))
	for i, k := range keys {
		c := make([]int, 49)
		copy(c, k[:48])
		c[48] = -1 // token that no stored key carries → mid-region miss
		q[i] = c
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		n, m := tr.Match(q[i%len(q)])
		sinkNode, sinkInt = n, m
	}
}

// --- Insert ----------------------------------------------------------------

// BenchmarkInsert measures steady-state insertion of distinct keys into a tree
// that is rebuilt each pass over the corpus, so every op is a genuine new-key
// insert (a mix of leaf-hangs and splits), not an update. The rebuild is timer-
// excluded so only insertion allocates into the count.
func BenchmarkInsert(b *testing.B) {
	keys := corpus(64, 32, 48, 16)
	tr := radix.New(radix.Config{MaxNodes: 0})

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if i%len(keys) == 0 {
			b.StopTimer()
			tr = radix.New(radix.Config{MaxNodes: 0})
			b.StartTimer()
		}
		sinkNode = tr.Insert(keys[i%len(keys)], payload)
	}
}

// BenchmarkInsert_Update re-inserts keys that already exist — the value-update
// path. It walks and overwrites Value in place; it must not allocate.
func BenchmarkInsert_Update(b *testing.B) {
	keys := corpus(64, 32, 48, 16)
	tr := radix.New(radix.Config{MaxNodes: 0})
	prefill(tr, keys)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sinkNode = tr.Insert(keys[i%len(keys)], payload)
	}
}

// BenchmarkInsert_Split isolates the mid-edge split: a fresh tree gets one base
// key (timer-excluded), then the timed insert diverges inside that edge, forcing
// splitChild plus a new leaf.
func BenchmarkInsert_Split(b *testing.B) {
	prefix := make([]int, 48)
	var s lcg = 1
	for i := range prefix {
		prefix[i] = s.next() & 0xffff
	}
	base := append(append([]int{}, prefix...), 1, 2, 3, 4)
	diverge := append(append([]int{}, prefix...), 9, 8, 7, 6) // breaks at index 48

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		b.StopTimer()
		tr := radix.New(radix.Config{MaxNodes: 0})
		tr.Insert(base, payload)
		b.StartTimer()
		sinkNode = tr.Insert(diverge, payload)
	}
}

// --- Acquire / Release -----------------------------------------------------

// BenchmarkAcquireRelease pins and unpins a deep leaf's path. Walking the parent
// chain must not allocate.
func BenchmarkAcquireRelease(b *testing.B) {
	keys := corpus(64, 32, 48, 16)
	tr := radix.New(radix.Config{MaxNodes: 0})
	prefill(tr, keys)
	leaf, _ := tr.Match(keys[len(keys)/2])

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tr.Acquire(leaf)
		tr.Release(leaf)
	}
}

// --- Evict -----------------------------------------------------------------

// BenchmarkEvict isolates the LRU scan + leaf removal. The tree is refilled
// (timer-excluded) whenever it empties, so only Evict allocates into the count.
func BenchmarkEvict(b *testing.B) {
	keys := corpus(64, 32, 48, 16)
	tr := radix.New(radix.Config{MaxNodes: 0})
	prefill(tr, keys)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if tr.Evict() == nil {
			b.StopTimer()
			tr = radix.New(radix.Config{MaxNodes: 0})
			prefill(tr, keys)
			b.StartTimer()
		}
	}
}

// BenchmarkEvictToCapacity models a bounded cache under churn: insert a fresh
// key, then drain back to the node bound. Exercises the LRU scan, leaf removal,
// and parent merges together.
func BenchmarkEvictToCapacity(b *testing.B) {
	keys := corpus(64, 32, 48, 16)
	const bound = 1500
	tr := radix.New(radix.Config{MaxNodes: bound})
	prefill(tr, keys)
	tr.EvictToCapacity()

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		b.StopTimer()
		// Reinsert a key that prior eviction may have dropped, to keep churning.
		tr.Insert(keys[i%len(keys)], payload)
		b.StartTimer()
		sinkInt = tr.EvictToCapacity()
	}
}

// --- Snapshot --------------------------------------------------------------

// BenchmarkSnapshot exercises the Result convention. Any allocation here is the
// core.Result/Stats boxing mandated by the public signature.
func BenchmarkSnapshot(b *testing.B) {
	tr := radix.New(radix.Config{MaxNodes: 0})
	prefill(tr, corpus(8, 8, 16, 8))

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		r := tr.Snapshot()
		sinkBool = r.OK
	}
}
