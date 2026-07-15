// SPDX-Licence-Identifier: EUPL-1.2

package radix_test

import (
	"testing"

	"dappco.re/go/inference/kv/radix"
)

// --- helpers ---------------------------------------------------------------

// toks is a terse literal for token sequences in tests.
//
//	toks(1, 2, 3) // []int{1, 2, 3}
func toks(v ...int) []int { return v }

// --- Match ----------------------------------------------------------------

// TestRadix_Match_Good covers the happy path: an exact insert is found whole,
// and a longer query over a stored prefix returns the stored prefix length.
func TestRadix_Match_Good(t *testing.T) {
	tr := radix.New(radix.Config{MaxNodes: 16})
	n := tr.Insert(toks(1, 2, 3, 4), "kv-a")
	if n == nil {
		t.Fatal("Insert returned nil node")
	}

	// Exact hit — every token matched, value present on the node.
	got, matched := tr.Match(toks(1, 2, 3, 4))
	if matched != 4 {
		t.Fatalf("exact match length = %d, want 4", matched)
	}
	if got == nil || got.Value != "kv-a" {
		t.Fatalf("exact match node value = %v, want kv-a", nodeValue(got))
	}

	// Longest-prefix hit — query extends past the stored sequence; only the
	// stored 4 tokens are a cache hit.
	_, matched = tr.Match(toks(1, 2, 3, 4, 5, 6))
	if matched != 4 {
		t.Fatalf("over-length match = %d, want 4 (stored prefix only)", matched)
	}
}

// TestRadix_Match_Bad covers the longest *partial* prefix: two sequences that
// share a head diverge, and a query down the shared head returns only the
// shared length, landing on the split (internal) node.
func TestRadix_Match_Bad(t *testing.T) {
	tr := radix.New(radix.Config{MaxNodes: 16})
	tr.Insert(toks(1, 2, 3, 4), "a")
	tr.Insert(toks(1, 2, 9, 9), "b") // diverges at index 2 → splits [1,2]

	// Query shares [1,2] then diverges at the 7 — partial hit of length 2.
	node, matched := tr.Match(toks(1, 2, 7))
	if matched != 2 {
		t.Fatalf("partial match = %d, want 2 (shared [1,2])", matched)
	}
	if node == nil {
		t.Fatal("partial match returned nil node")
	}
	// The landing node is the split point — it has no value of its own.
	if node.Value != nil {
		t.Fatalf("split node carries value %v, want nil", node.Value)
	}

	// Query that diverges inside the very first edge from root: token 1 starts
	// the [1,2] edge but token 5 breaks it mid-edge. A prefix hit must align to
	// a stored node boundary (the KV block covers [1,2] as a unit), so a
	// partial-edge match does not count — the deepest fully consumed node is
	// the root and matched is 0.
	landing, matched := tr.Match(toks(1, 5))
	if matched != 0 {
		t.Fatalf("mid-edge divergence match = %d, want 0 (no node boundary)", matched)
	}
	if landing == nil {
		t.Fatal("mid-edge divergence returned nil node, want root")
	}
}

// TestRadix_Match_Ugly covers degenerate inputs: empty query, empty tree, and a
// query whose very first token is absent — all must report zero match.
func TestRadix_Match_Ugly(t *testing.T) {
	empty := radix.New(radix.Config{MaxNodes: 4})

	// Empty tree, real query → root, zero match.
	node, matched := empty.Match(toks(1, 2, 3))
	if matched != 0 {
		t.Fatalf("empty-tree match = %d, want 0", matched)
	}
	if node == nil {
		t.Fatal("Match must return the root even on miss, got nil")
	}

	// Empty query on a populated tree → zero match at root.
	empty.Insert(toks(1, 2), "x")
	_, matched = empty.Match(nil)
	if matched != 0 {
		t.Fatalf("empty-query match = %d, want 0", matched)
	}

	// First token absent → no descent, zero match.
	_, matched = empty.Match(toks(9))
	if matched != 0 {
		t.Fatalf("absent-root-token match = %d, want 0", matched)
	}
}

// --- Insert ---------------------------------------------------------------

// TestRadix_Insert_Good covers a shared-prefix insert that reuses the existing
// edge: the second sequence extends the first, so no split occurs and both are
// retrievable.
func TestRadix_Insert_Good(t *testing.T) {
	tr := radix.New(radix.Config{MaxNodes: 16})
	tr.Insert(toks(1, 2), "ab")
	tr.Insert(toks(1, 2, 3, 4), "abcd") // pure extension of [1,2]

	_, m1 := tr.Match(toks(1, 2))
	if m1 != 2 {
		t.Fatalf("prefix match = %d, want 2", m1)
	}
	n2, m2 := tr.Match(toks(1, 2, 3, 4))
	if m2 != 4 || n2.Value != "abcd" {
		t.Fatalf("extension match = %d/%v, want 4/abcd", m2, nodeValue(n2))
	}
	// Reusing a shared prefix must not duplicate it — [1,2] is one node still.
	if got := tr.Len(); got != 2 {
		t.Fatalf("node count after extension = %d, want 2 (prefix + tail)", got)
	}
}

// TestRadix_Insert_Bad covers the classic radix split: a new key diverges in
// the middle of an existing edge, forcing that edge to break into a shared
// parent and two child branches.
func TestRadix_Insert_Bad(t *testing.T) {
	tr := radix.New(radix.Config{MaxNodes: 16})
	tr.Insert(toks(1, 2, 3, 4), "first")
	leaf := tr.Insert(toks(1, 2, 9), "second") // diverges at index 2

	// The returned node is the full-key leaf carrying the new value.
	if leaf == nil || leaf.Value != "second" {
		t.Fatalf("split insert node = %v, want second", nodeValue(leaf))
	}

	// Both original and new keys remain exactly findable post-split.
	na, ma := tr.Match(toks(1, 2, 3, 4))
	if ma != 4 || na.Value != "first" {
		t.Fatalf("post-split original = %d/%v, want 4/first", ma, nodeValue(na))
	}
	nb, mb := tr.Match(toks(1, 2, 9))
	if mb != 3 || nb.Value != "second" {
		t.Fatalf("post-split new = %d/%v, want 3/second", mb, nodeValue(nb))
	}

	// Split produced: shared [1,2] (no value) + [3,4] + [9] = 3 nodes.
	if got := tr.Len(); got != 3 {
		t.Fatalf("node count after split = %d, want 3", got)
	}
}

// TestRadix_Insert_Ugly covers duplicate inserts and the empty-sequence insert:
// re-inserting the same key updates the value in place (no new node), and
// inserting nil/empty is a no-op returning the root.
func TestRadix_Insert_Ugly(t *testing.T) {
	tr := radix.New(radix.Config{MaxNodes: 16})
	first := tr.Insert(toks(5, 6, 7), "v1")
	again := tr.Insert(toks(5, 6, 7), "v2") // duplicate key → update in place

	if first != again {
		t.Fatal("duplicate insert returned a different node, want same node")
	}
	if again.Value != "v2" {
		t.Fatalf("duplicate insert value = %v, want v2 (updated)", again.Value)
	}
	if got := tr.Len(); got != 1 {
		t.Fatalf("node count after duplicate = %d, want 1", got)
	}

	// Empty insert is a no-op → returns root, adds nothing.
	root := tr.Insert(nil, "ignored")
	if root == nil {
		t.Fatal("empty Insert returned nil, want root")
	}
	if got := tr.Len(); got != 1 {
		t.Fatalf("node count after empty insert = %d, want 1", got)
	}

	// Insert a key that ends exactly at a NEW split point: [1,2,3] then [1,2]
	// splits [1,2,3] into shared [1,2] (which the second key terminates at) and
	// tail [3]. The shared node must carry the second key's value.
	st := radix.New(radix.Config{MaxNodes: 16})
	st.Insert(toks(1, 2, 3), "long")
	mid := st.Insert(toks(1, 2), "short")
	if mid.Value != "short" {
		t.Fatalf("split-point insert value = %v, want short", mid.Value)
	}
	if n, m := st.Match(toks(1, 2)); m != 2 || n.Value != "short" {
		t.Fatalf("split-point match = %d/%v, want 2/short", m, nodeValue(n))
	}
	if n, m := st.Match(toks(1, 2, 3)); m != 3 || n.Value != "long" {
		t.Fatalf("tail still findable = %d/%v, want 3/long", m, nodeValue(n))
	}
}

// --- Evict ----------------------------------------------------------------

// TestRadix_Evict_Good covers LRU ordering: the least-recently-used leaf is the
// one evicted, and a later Match on a different leaf protects it from being the
// victim.
func TestRadix_Evict_Good(t *testing.T) {
	tr := radix.New(radix.Config{MaxNodes: 16})
	tr.Insert(toks(1), "a")
	tr.Insert(toks(2), "b")
	tr.Insert(toks(3), "c")

	// Touch [1] and [3] so [2] is the least-recently-used leaf.
	tr.Match(toks(1))
	tr.Match(toks(3))

	victim := tr.Evict()
	if victim == nil {
		t.Fatal("Evict returned nil, want the LRU leaf")
	}
	if victim.Value != "b" {
		t.Fatalf("evicted value = %v, want b (the LRU leaf)", victim.Value)
	}
	// [2] is gone; [1] and [3] survive.
	if _, m := tr.Match(toks(2)); m != 0 {
		t.Fatalf("evicted key still matches (len %d), want 0", m)
	}
	if _, m := tr.Match(toks(1)); m != 1 {
		t.Fatal("non-victim [1] was lost")
	}
}

// TestRadix_Evict_Bad covers ref-counting: an Acquired path is spared and the
// next-LRU unreferenced leaf is evicted instead; after Release the protected
// leaf becomes eligible again.
func TestRadix_Evict_Bad(t *testing.T) {
	tr := radix.New(radix.Config{MaxNodes: 16})
	a := tr.Insert(toks(1), "a")
	tr.Insert(toks(2), "b")

	// [1] is least-recently-used, but we pin it. Eviction must skip it and
	// take [2] instead.
	tr.Acquire(a)
	victim := tr.Evict()
	if victim == nil || victim.Value != "b" {
		t.Fatalf("evicted %v with [1] referenced, want b", nodeValue(victim))
	}

	// With [1] still referenced and the only remaining leaf, Evict finds no
	// eligible victim → nil.
	if got := tr.Evict(); got != nil {
		t.Fatalf("Evict returned %v while only leaf is referenced, want nil", nodeValue(got))
	}

	// Release [1] — it becomes evictable again.
	tr.Release(a)
	if got := tr.Evict(); got == nil || got.Value != "a" {
		t.Fatalf("post-release Evict = %v, want a", nodeValue(got))
	}
}

// TestRadix_Evict_Ugly covers capacity enforcement and merge-on-evict: filling
// past MaxNodes reports over capacity, EvictToCapacity drains it back, and
// evicting a leaf whose parent becomes a single-child internal node merges that
// parent back into its surviving child.
func TestRadix_Evict_Ugly(t *testing.T) {
	tr := radix.New(radix.Config{MaxNodes: 2})

	tr.Insert(toks(1), "a")
	tr.Insert(toks(2), "b")
	if tr.OverCapacity() {
		t.Fatal("tree reports over capacity at exactly MaxNodes")
	}
	tr.Insert(toks(3), "c") // 3 leaves > MaxNodes(2)
	if !tr.OverCapacity() {
		t.Fatal("tree does not report over capacity above MaxNodes")
	}

	// Drain back to capacity — evicts LRU leaves until Len <= MaxNodes.
	freed := tr.EvictToCapacity()
	if freed < 1 {
		t.Fatalf("EvictToCapacity freed %d nodes, want >= 1", freed)
	}
	if tr.OverCapacity() {
		t.Fatalf("still over capacity after drain: Len=%d MaxNodes=2", tr.Len())
	}

	// Merge-on-evict: build [1,2,3] and [1,2,4]; this splits at [1,2].
	// Evicting the [4] leaf leaves [1,2] with a single child [3] — the parent
	// must merge into [1,2,3] so no dangling single-child internal node remains.
	mt := radix.New(radix.Config{MaxNodes: 16})
	mt.Insert(toks(1, 2, 3), "x")
	four := mt.Insert(toks(1, 2, 4), "y")
	before := mt.Len() // [1,2] + [3] + [4] = 3

	// Make [4] the LRU victim, evict it explicitly via its leaf.
	mt.Match(toks(1, 2, 3)) // freshen the survivor
	if got := mt.EvictNode(four); !got {
		t.Fatal("EvictNode([1,2,4]) returned false, want true")
	}
	// [4] gone AND [1,2]+[3] merged into one node → net minus 2 from before.
	if got := mt.Len(); got != before-2 {
		t.Fatalf("post-merge Len = %d, want %d (leaf removed + parent merged)", got, before-2)
	}
	// The merged survivor is still exactly findable with its value intact.
	n, m := mt.Match(toks(1, 2, 3))
	if m != 3 || n.Value != "x" {
		t.Fatalf("merged survivor = %d/%v, want 3/x", m, nodeValue(n))
	}

	// Merge that must re-parent grandchildren: [1,2,3], [1,2,4,5], [1,2,4,6]
	// build [1,2] -> {[3], [1,2,4] -> {[5],[6]}}. Evicting the [3] leaf leaves
	// [1,2] with one child [1,2,4] that has its OWN children — the merge fuses
	// [1,2]+[4] into [1,2,4] and must re-home [5] and [6] under it.
	gt := radix.New(radix.Config{MaxNodes: 16})
	gt.Insert(toks(1, 2, 3), "three")
	gt.Insert(toks(1, 2, 4, 5), "five")
	gt.Insert(toks(1, 2, 4, 6), "six")
	three, _ := gt.Match(toks(1, 2, 3))
	if !gt.EvictNode(three) {
		t.Fatal("EvictNode([1,2,3]) = false, want true")
	}
	// Grandchildren survived the re-parent and remain exactly findable.
	if n, m := gt.Match(toks(1, 2, 4, 5)); m != 4 || n.Value != "five" {
		t.Fatalf("regrandchild [5] = %d/%v, want 4/five", m, nodeValue(n))
	}
	if n, m := gt.Match(toks(1, 2, 4, 6)); m != 4 || n.Value != "six" {
		t.Fatalf("regrandchild [6] = %d/%v, want 4/six", m, nodeValue(n))
	}
}

// --- guards, refcount edges, capacity, snapshot ---------------------------

// TestRadix_Guards covers the defensive no-ops: Acquire/Release on nil, and
// EvictNode's rejection of nil, root-adjacent, non-leaf, and referenced nodes.
func TestRadix_Guards(t *testing.T) {
	tr := radix.New(radix.Config{MaxNodes: 16})

	// nil Acquire/Release must not panic and must not affect anything.
	tr.Acquire(nil)
	tr.Release(nil)

	// Parent() on a nil receiver is a safe nil.
	var none *radix.Node
	if none.Parent() != nil {
		t.Fatal("nil.Parent() != nil")
	}

	tr.Insert(toks(1, 2, 3), "x")
	internalPath := tr.Insert(toks(1, 2, 9), "y") // forces split at [1,2]

	// EvictNode(nil) → false.
	if tr.EvictNode(nil) {
		t.Fatal("EvictNode(nil) = true, want false")
	}

	// EvictNode on an internal (non-leaf) node → false. Reach the [1,2] split
	// via the parent chain of a leaf.
	internal := parentOf(internalPath)
	if internal == nil {
		t.Fatal("expected an internal split parent")
	}
	if tr.EvictNode(internal) {
		t.Fatal("EvictNode(internal) = true, want false (not a leaf)")
	}

	// EvictNode on a referenced leaf → false.
	tr.Acquire(internalPath)
	if tr.EvictNode(internalPath) {
		t.Fatal("EvictNode(referenced leaf) = true, want false")
	}
	tr.Release(internalPath)

	// Release below zero is clamped — a second Release after balance is a no-op
	// that leaves the leaf evictable.
	tr.Release(internalPath)
	if !tr.EvictNode(internalPath) {
		t.Fatal("EvictNode(unreferenced leaf) = false, want true after clamped release")
	}
}

// TestRadix_Capacity_Unbounded covers MaxNodes<=0: never over capacity and
// EvictToCapacity is a no-op, plus EvictToCapacity stopping early when the only
// over-capacity leaves are all referenced.
func TestRadix_Capacity_Unbounded(t *testing.T) {
	// Unbounded tree — capacity helpers are inert.
	ub := radix.New(radix.Config{MaxNodes: 0})
	ub.Insert(toks(1), "a")
	ub.Insert(toks(2), "b")
	if ub.OverCapacity() {
		t.Fatal("unbounded tree reports over capacity")
	}
	if freed := ub.EvictToCapacity(); freed != 0 {
		t.Fatalf("unbounded EvictToCapacity freed %d, want 0", freed)
	}

	// Bounded tree over capacity but every leaf referenced → drain stalls at >0.
	bt := radix.New(radix.Config{MaxNodes: 1})
	a := bt.Insert(toks(1), "a")
	b := bt.Insert(toks(2), "b")
	bt.Acquire(a)
	bt.Acquire(b)
	if !bt.OverCapacity() {
		t.Fatal("bounded tree not over capacity with 2 nodes, MaxNodes 1")
	}
	freed := bt.EvictToCapacity()
	if freed != 0 {
		t.Fatalf("EvictToCapacity freed %d with all leaves pinned, want 0", freed)
	}
	if !bt.OverCapacity() {
		t.Fatal("tree should remain over capacity when nothing is evictable")
	}
	// Plain Evict also returns nil when every leaf is referenced.
	if v := bt.Evict(); v != nil {
		t.Fatalf("Evict with all leaves pinned = %v, want nil", nodeValue(v))
	}
}

// TestRadix_NoMergeOnValuedParent covers the merge guard: evicting a leaf whose
// parent both has a remaining child AND terminates a key of its own must NOT
// merge — the parent's value would be lost. Sequences [1,2], [1,2,3], [1,2,4]
// give a [1,2] node that holds a value and has two children; dropping [4]
// leaves it valued with one child, so it stays put.
func TestRadix_NoMergeOnValuedParent(t *testing.T) {
	tr := radix.New(radix.Config{MaxNodes: 16})
	tr.Insert(toks(1, 2), "mid") // [1,2] terminates a key
	tr.Insert(toks(1, 2, 3), "leaf3")
	four := tr.Insert(toks(1, 2, 4), "leaf4")
	before := tr.Len()

	if !tr.EvictNode(four) {
		t.Fatal("EvictNode([1,2,4]) = false, want true")
	}
	// Only the leaf is gone — the valued [1,2] parent is NOT merged away.
	if got := tr.Len(); got != before-1 {
		t.Fatalf("post-evict Len = %d, want %d (no merge of valued parent)", got, before-1)
	}
	if n, m := tr.Match(toks(1, 2)); m != 2 || n.Value != "mid" {
		t.Fatalf("valued parent lost: %d/%v, want 2/mid", m, nodeValue(n))
	}
	if n, m := tr.Match(toks(1, 2, 3)); m != 3 || n.Value != "leaf3" {
		t.Fatalf("surviving child lost: %d/%v, want 3/leaf3", m, nodeValue(n))
	}
}

// TestRadix_Snapshot covers the Result convention: an under-capacity tree yields
// OK with Stats, an over-capacity tree yields a failed Result carrying the
// scoped error.
func TestRadix_Snapshot(t *testing.T) {
	tr := radix.New(radix.Config{MaxNodes: 2})
	tr.Insert(toks(1), "a")

	r := tr.Snapshot()
	if !r.OK {
		t.Fatalf("under-capacity Snapshot not OK: %v", r.Error())
	}
	s := r.Value.(radix.Stats)
	if s.Nodes != 1 || s.Capacity != 2 || s.Over {
		t.Fatalf("Stats = %+v, want Nodes 1 / Capacity 2 / Over false", s)
	}

	tr.Insert(toks(2), "b")
	tr.Insert(toks(3), "c") // 3 > MaxNodes 2 → over capacity
	r = tr.Snapshot()
	if r.OK {
		t.Fatal("over-capacity Snapshot OK, want failed Result")
	}
	if r.Error() == "" {
		t.Fatal("over-capacity Snapshot carries no error message")
	}
}

// TestRadix_LongestPrefix_Good is the cross-conversation share Match cannot do:
// one cached sequence lives on a single edge, and a second query sharing its
// leading run — but diverging mid-edge — must still report the shared length and
// hand back the cached value. Match returns 0 here (no whole edge walked);
// LongestPrefix returns the true shared run.
func TestRadix_LongestPrefix_Good(t *testing.T) {
	tr := radix.New(radix.Config{MaxNodes: 16})
	leaf := tr.Insert(toks(1, 2, 3, 4, 5), "bundle-a")

	// Match discards the mid-edge run — the contrast this method exists for.
	if _, m := tr.Match(toks(1, 2, 3, 9)); m != 0 {
		t.Fatalf("Match mid-edge = %d, want 0 (whole-edge only) — LongestPrefix contrast is void", m)
	}

	node, matched, ok := tr.LongestPrefix(toks(1, 2, 3, 9))
	if !ok {
		t.Fatal("LongestPrefix ok=false, want true for a shared leading run")
	}
	if matched != 3 {
		t.Fatalf("LongestPrefix matched = %d, want 3 (shared [1,2,3])", matched)
	}
	if node != leaf || node.Value != "bundle-a" {
		t.Fatalf("LongestPrefix node value = %v, want the cached bundle-a leaf", nodeValue(node))
	}

	// Exact and over-length queries agree with Match on whole-edge lengths.
	if _, matched, _ = tr.LongestPrefix(toks(1, 2, 3, 4, 5)); matched != 5 {
		t.Fatalf("exact LongestPrefix = %d, want 5", matched)
	}
	if _, matched, _ = tr.LongestPrefix(toks(1, 2, 3, 4, 5, 6)); matched != 5 {
		t.Fatalf("over-length LongestPrefix = %d, want 5 (cached prefix only)", matched)
	}
}

// TestRadix_LongestPrefix_Bad walks past a valueless split node to a
// representative value: a query landing exactly on an internal split point (no
// value of its own) still resolves to a leaf beneath it, because every leaf in
// that subtree shares the matched prefix byte-for-byte.
func TestRadix_LongestPrefix_Bad(t *testing.T) {
	tr := radix.New(radix.Config{MaxNodes: 16})
	tr.Insert(toks(1, 2, 3, 4), "a") // [1,2] split parent forms when the
	tr.Insert(toks(1, 2, 9, 9), "b") // second insert diverges at index 2

	// Query ends on the valueless split node [1,2]; a representative value from
	// the subtree must still come back so the shared span is reusable.
	node, matched, ok := tr.LongestPrefix(toks(1, 2))
	if !ok || matched != 2 {
		t.Fatalf("LongestPrefix([1,2]) = (matched %d, ok %v), want (2, true)", matched, ok)
	}
	if node == nil || node.Value == nil {
		t.Fatalf("split-node LongestPrefix returned no representative value: %v", nodeValue(node))
	}
	if v := node.Value; v != "a" && v != "b" {
		t.Fatalf("representative value = %v, want one of the subtree leaves a/b", v)
	}

	// A query diverging just past the split still resolves to length 2.
	if _, matched, ok = tr.LongestPrefix(toks(1, 2, 5)); !ok || matched != 2 {
		t.Fatalf("LongestPrefix([1,2,5]) = (matched %d, ok %v), want (2, true)", matched, ok)
	}
}

// TestRadix_LongestPrefix_Ugly rejects the empty query and a first-token miss —
// the caller must fall back to a fresh prefill, never a phantom graft.
func TestRadix_LongestPrefix_Ugly(t *testing.T) {
	tr := radix.New(radix.Config{MaxNodes: 16})
	tr.Insert(toks(1, 2, 3), "a")

	if _, matched, ok := tr.LongestPrefix(nil); ok || matched != 0 {
		t.Fatalf("LongestPrefix(nil) = (matched %d, ok %v), want (0, false)", matched, ok)
	}
	if _, matched, ok := tr.LongestPrefix(toks(9, 9)); ok || matched != 0 {
		t.Fatalf("LongestPrefix(first-token miss) = (matched %d, ok %v), want (0, false)", matched, ok)
	}
	if _, _, ok := tr.LongestPrefix(toks(1, 2, 3)); !ok {
		t.Fatal("LongestPrefix on empty tree branch regressed for a valid key")
	}
}

// parentOf reaches a leaf's internal split parent for the EvictNode non-leaf
// rejection test.
func parentOf(n *radix.Node) *radix.Node { return n.Parent() }

// nodeValue is a nil-safe accessor for failure messages.
func nodeValue(n *radix.Node) any {
	if n == nil {
		return "<nil node>"
	}
	return n.Value
}
