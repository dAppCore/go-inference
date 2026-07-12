// SPDX-Licence-Identifier: EUPL-1.2

package prefixindex_test

import (
	"sync"
	"testing"

	"dappco.re/go/inference/kv/prefixindex"
)

// sysPrefix is a synthetic shared system-prompt token run; userA/userB are the
// divergent tails two conversations append after it.
var (
	sysPrefix = []int32{10, 11, 12, 13, 14, 15, 16, 17}
	convA     = append(append([]int32{}, sysPrefix...), 90, 91)
	convB     = append(append([]int32{}, sysPrefix...), 80, 81, 82)
)

// TestIndex_Match_Good is the headline: conversation A publishes its framed
// prompt, and conversation B — sharing only the system prefix — resolves that
// shared run and A's backing bundle, even though B is the SECOND conversation
// and no split node yet exists (radix.Match would return 0 here).
func TestIndex_Match_Good(t *testing.T) {
	ix := prefixindex.New(prefixindex.Config{MaxEntries: 64})
	ix.Publish(convA, prefixindex.Entry{BundleURI: "state://a", BlockSize: 4, TokenCount: len(convA)})

	entry, shared, ok := ix.Match(convB)
	if !ok {
		t.Fatal("Match(convB) ok=false — B must find A's shared system prefix")
	}
	if shared != len(sysPrefix) {
		t.Fatalf("shared length = %d, want %d (the system prefix)", shared, len(sysPrefix))
	}
	if entry.BundleURI != "state://a" || entry.BlockSize != 4 {
		t.Fatalf("matched entry = %+v, want A's bundle at block size 4", entry)
	}

	// An exact re-lookup of A returns A's own full length.
	if _, shared, ok = ix.Match(convA); !ok || shared != len(convA) {
		t.Fatalf("Match(convA) = (shared %d, ok %v), want (%d, true)", shared, ok, len(convA))
	}
}

// TestIndex_Match_Bad declines cleanly when nothing shares a leading token, and
// when the query is empty — the consumer stays on the fresh-prefill path.
func TestIndex_Match_Bad(t *testing.T) {
	ix := prefixindex.New(prefixindex.Config{MaxEntries: 64})
	ix.Publish(convA, prefixindex.Entry{BundleURI: "state://a", BlockSize: 4, TokenCount: len(convA)})

	if _, _, ok := ix.Match([]int32{99, 98, 97}); ok {
		t.Fatal("Match on a disjoint token run must not share")
	}
	if _, _, ok := ix.Match(nil); ok {
		t.Fatal("Match(nil) must decline")
	}
}

// TestIndex_Publish_Ugly refuses to index a run the graft could not wake: an
// empty token run, or an entry missing its bundle URI, block size, or token
// count. A partial entry indexed would resolve a match that then fails to wake.
func TestIndex_Publish_Ugly(t *testing.T) {
	ix := prefixindex.New(prefixindex.Config{MaxEntries: 64})
	ix.Publish(nil, prefixindex.Entry{BundleURI: "state://a", BlockSize: 4, TokenCount: 2})
	ix.Publish(convA, prefixindex.Entry{BlockSize: 4, TokenCount: len(convA)})          // no URI
	ix.Publish(convA, prefixindex.Entry{BundleURI: "state://a", TokenCount: len(convA)}) // no block size
	ix.Publish(convA, prefixindex.Entry{BundleURI: "state://a", BlockSize: 4})           // no token count
	if n := ix.Len(); n != 0 {
		t.Fatalf("index holds %d entries after four invalid publishes, want 0", n)
	}
}

// TestIndex_Evict_Ugly proves stale invalidation is precise: eviction removes
// the entry only when the caller names the dead bundle, never a healthy sibling,
// and a removed entry no longer matches.
func TestIndex_Evict_Ugly(t *testing.T) {
	ix := prefixindex.New(prefixindex.Config{MaxEntries: 64})
	ix.Publish(convA, prefixindex.Entry{BundleURI: "state://a", BlockSize: 4, TokenCount: len(convA)})

	// Wrong bundle URI — a graft against a DIFFERENT bundle failing must not
	// drop A's healthy entry.
	if ix.Evict(convA, "state://other") {
		t.Fatal("Evict removed an entry naming a different bundle")
	}
	if ix.Len() != 1 {
		t.Fatalf("healthy entry dropped by a mismatched evict; Len = %d, want 1", ix.Len())
	}

	// Correct URI — the stale entry goes, and B no longer finds a share.
	if !ix.Evict(convA, "state://a") {
		t.Fatal("Evict(correct URI) reported no removal")
	}
	if _, _, ok := ix.Match(convB); ok {
		t.Fatal("evicted entry still matches — stale state would be grafted")
	}
	if ix.Evict(convA, "state://a") {
		t.Fatal("second evict of an absent entry reported a removal")
	}
}

// TestIndex_Concurrent runs publish, match, and evict from many goroutines — the
// index is cross-request shared state and must be race-clean (go test -race).
func TestIndex_Concurrent(t *testing.T) {
	ix := prefixindex.New(prefixindex.Config{MaxEntries: 256})
	var wg sync.WaitGroup
	for g := 0; g < 16; g++ {
		wg.Add(1)
		go func(g int) {
			defer wg.Done()
			base := []int32{int32(g), int32(g + 1), int32(g + 2), int32(g + 3)}
			for i := 0; i < 200; i++ {
				toks := append(append([]int32{}, base...), int32(i))
				ix.Publish(toks, prefixindex.Entry{BundleURI: "state://x", BlockSize: 2, TokenCount: len(toks)})
				ix.Match(base)
				ix.Evict(toks, "state://x")
			}
		}(g)
	}
	wg.Wait()
}
