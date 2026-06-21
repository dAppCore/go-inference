// SPDX-Licence-Identifier: EUPL-1.2

// Package radix is the token-sequence radix tree behind cross-request KV
// prefix sharing (RFC — prefix cache). When two requests share a leading
// run of tokens they can share the KV blocks computed for that run; this tree
// is the index that finds the shared run. It maps a token prefix to an opaque
// Value (the execution engine maps that Value to its KV blocks) and exposes the
// length of the longest cached prefix for an incoming sequence — the cache-hit
// length the scheduler skips recomputing.
//
// The tree is a classic radix (compressed) trie over []int token keys: each
// edge holds a run of tokens rather than a single token, and inserting a key
// that diverges mid-edge SPLITS that edge into a shared parent and two
// branches. It is pure index logic — it never touches KV memory, never loads a
// model — and it is deterministic: recency for LRU is a monotonic tick, never
// the wall clock.
//
//	tr := radix.New(radix.Config{MaxNodes: 4096})
//	tr.Insert([]int{1, 2, 3, 4}, blockA)        // cache a prefix → KV handle
//	node, hit := tr.Match([]int{1, 2, 3, 4, 5}) // hit == 4: reuse 4 tokens' KV
//	tr.Acquire(node)                            // protect it while a request runs
//	defer tr.Release(node)
//	for tr.OverCapacity() {                      // reclaim under memory pressure
//	    if tr.Evict() == nil { break }          // nil → nothing evictable
//	}
//
// Capacity metric: node count. MaxNodes bounds the number of nodes in the tree
// excluding the always-present root; OverCapacity / EvictToCapacity reclaim
// against it by evicting least-recently-used unreferenced leaves. (A node-count
// bound, not a token-count bound — one node per cached branch point is the
// natural unit of the index, and the execution engine accounts KV bytes
// separately via the opaque Value.)
package radix

import (
	"sync"

	core "dappco.re/go"
)

// Config tunes one tree. MaxNodes is the capacity bound used by OverCapacity
// and EvictToCapacity; a value <= 0 means unbounded (OverCapacity is always
// false and EvictToCapacity is a no-op).
//
//	cfg := radix.Config{MaxNodes: 4096}
type Config struct {
	MaxNodes int // capacity bound on node count (excludes root); <=0 == unbounded
}

// Node is one vertex of the radix tree. edge is the run of tokens on the
// in-edge from the parent (empty only for the root). Value is the opaque
// payload for the full prefix ending at this node — nil on the root and on
// internal split points that no key terminates at. Callers read Value; the tree
// owns everything else.
//
//	if node.Value != nil { kvHandle := node.Value.(KVHandle) }
type Node struct {
	edge     []int         // tokens on the edge into this node
	Value    any           // opaque payload for the prefix ending here (nil if none)
	children map[int]*Node // keyed by first token of each child's edge
	parent   *Node
	refs     int    // Acquire/Release count — >0 protects from eviction
	tick     uint64 // last-used recency (LRU key; higher == more recent)
}

// Tree is a token-prefix radix tree. Construct with New. Safe for concurrent
// use — every public method takes the tree lock.
type Tree struct {
	mu       sync.Mutex
	root     *Node
	maxNodes int
	count    int    // nodes excluding root
	tick     uint64 // monotonic recency source
}

// New builds an empty tree with the given capacity bound.
//
//	tr := radix.New(radix.Config{MaxNodes: 4096})
func New(cfg Config) *Tree {
	return &Tree{
		root:     &Node{children: map[int]*Node{}},
		maxNodes: cfg.MaxNodes,
	}
}

// nextTick advances and returns the recency counter. Caller holds mu.
func (t *Tree) nextTick() uint64 {
	t.tick++
	return t.tick
}

// commonPrefix returns the length of the shared leading run of a and b.
//
//	commonPrefix([]int{1, 2, 9}, []int{1, 2, 3}) == 2
func commonPrefix(a, b []int) int {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	i := 0
	for i < n && a[i] == b[i] {
		i++
	}
	return i
}

// Match walks the tree along tokens, returning the deepest node reached and how
// many tokens matched — the cache-hit length. A full match lands on the node
// whose accumulated edges equal tokens; a partial match stops at the deepest
// node fully consumed before divergence (an in-edge that only partly matches
// does NOT advance into that child, so matchedLen counts only whole edges
// walked). Match marks every node on the walked path as used (LRU) so a hit
// protects its prefix from being the next eviction victim. On any miss — empty
// tokens, empty tree, or a first token with no child — it returns the root and
// 0.
//
//	node, hit := tr.Match([]int{1, 2, 3, 4, 5}) // hit == 4 → reuse 4 tokens' KV
func (t *Tree) Match(tokens []int) (node *Node, matchedLen int) {
	t.mu.Lock()
	defer t.mu.Unlock()

	cur := t.root
	cur.tick = t.nextTick()
	matched := 0
	for matched < len(tokens) {
		child, ok := cur.children[tokens[matched]]
		if !ok {
			break
		}
		want := tokens[matched:]
		k := commonPrefix(child.edge, want)
		if k == len(child.edge) {
			// Whole edge consumed — descend and keep walking.
			matched += k
			cur = child
			cur.tick = t.nextTick()
			continue
		}
		// Partial edge match — the hit stops here; do not enter the child.
		break
	}
	return cur, matched
}

// Insert adds tokens to the tree, attaching value to the node for the full key,
// and returns that node. It reuses any existing shared prefix and SPLITS an
// existing edge when tokens diverge mid-edge (the classic radix split: the edge
// breaks into a shared parent plus the original tail and the new tail).
// Re-inserting an existing key updates its Value in place and returns the same
// node — no new node is created. Inserting an empty (or nil) sequence is a
// no-op that returns the root. Insert marks the path used (LRU).
//
//	leaf := tr.Insert([]int{1, 2, 3}, kvHandle) // leaf.Value == kvHandle
func (t *Tree) Insert(tokens []int, value any) *Node {
	t.mu.Lock()
	defer t.mu.Unlock()

	if len(tokens) == 0 {
		t.root.tick = t.nextTick()
		return t.root
	}

	cur := t.root
	cur.tick = t.nextTick()
	rest := tokens
	// rest is non-empty on entry and strictly shrinks each iteration that does
	// not return, so the loop always exits via a return — no trailing statement.
	for {
		child, ok := cur.children[rest[0]]
		if !ok {
			// No child starts here — hang the whole remaining run as a new leaf.
			leaf := &Node{edge: cloneTokens(rest), Value: value, children: map[int]*Node{}, parent: cur}
			leaf.tick = t.nextTick()
			cur.children[rest[0]] = leaf
			t.count++
			return leaf
		}

		k := commonPrefix(child.edge, rest)
		if k == len(child.edge) {
			// Edge fully matched — descend and consume it.
			cur = child
			cur.tick = t.nextTick()
			rest = rest[k:]
			if len(rest) == 0 {
				// Exact existing key — update value in place.
				cur.Value = value
				return cur
			}
			continue
		}
		// Mid-edge divergence (k < len(child.edge)) — split child.edge at k.
		cur = t.splitChild(cur, child, k)
		rest = rest[k:]
		if len(rest) == 0 {
			// New key ends exactly at the split point — it owns the value.
			cur.Value = value
			return cur
		}
	}
}

// splitChild breaks child's in-edge at offset k (0 < k < len(child.edge)),
// inserting a new shared-prefix node between parent and child. The new node
// carries no value; the original child keeps its value and its subtree. Returns
// the new shared node. Caller holds mu.
//
//	// edge [1,2,3,4] split at k=2 → shared [1,2] -> child [3,4]
func (t *Tree) splitChild(parent, child *Node, k int) *Node {
	shared := &Node{
		edge:     cloneTokens(child.edge[:k]),
		children: map[int]*Node{},
		parent:   parent,
	}
	shared.tick = t.nextTick()

	// Re-root the original child under shared with its edge trimmed by k.
	child.edge = cloneTokens(child.edge[k:])
	child.parent = shared
	shared.children[child.edge[0]] = child

	// Replace child with shared in the parent's child map.
	parent.children[shared.edge[0]] = shared
	t.count++ // one new internal node
	return shared
}

// Parent returns the node one step up the prefix path, or nil for the root.
// Exposed so a caller (or a diagnostic walk) can climb from a matched leaf back
// through the shared internal nodes of its prefix.
//
//	for n := leaf; n != nil; n = n.Parent() { … }
func (n *Node) Parent() *Node {
	if n == nil {
		return nil
	}
	return n.parent
}

// Acquire pins node (and, transitively, the prefix path to it) so eviction
// skips it while a request that depends on its KV is in flight. Balance every
// Acquire with a Release. Acquire on a nil node is a no-op.
//
//	tr.Acquire(node); defer tr.Release(node)
func (t *Tree) Acquire(node *Node) {
	if node == nil {
		return
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	for n := node; n != nil && n != t.root; n = n.parent {
		n.refs++
	}
}

// Release undoes one Acquire on node's path, returning it to eviction
// eligibility once its ref count reaches zero. Release on a nil node, or below
// zero, is clamped to a no-op so a stray Release can't corrupt the count.
//
//	tr.Release(node)
func (t *Tree) Release(node *Node) {
	if node == nil {
		return
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	for n := node; n != nil && n != t.root; n = n.parent {
		if n.refs > 0 {
			n.refs--
		}
	}
}

// Len reports the number of nodes excluding the root — the value bounded by
// MaxNodes.
//
//	if tr.Len() > 1000 { tr.EvictToCapacity() }
func (t *Tree) Len() int {
	t.mu.Lock()
	defer t.mu.Unlock()
	return t.count
}

// OverCapacity reports whether the node count exceeds MaxNodes. Always false
// when MaxNodes <= 0 (unbounded).
//
//	for tr.OverCapacity() { tr.Evict() }
func (t *Tree) OverCapacity() bool {
	t.mu.Lock()
	defer t.mu.Unlock()
	return t.maxNodes > 0 && t.count > t.maxNodes
}

// Evict removes the single least-recently-used UNREFERENCED leaf and returns
// it, or nil when no leaf is evictable (every leaf is referenced, or the tree
// is empty). Removing a leaf whose parent is left an unreferenced internal node
// with exactly one remaining child merges that parent back into the child, so
// the tree never keeps a redundant single-child split. Evict does not change
// recency of survivors.
//
//	if victim := tr.Evict(); victim != nil { engine.Free(victim.Value) }
func (t *Tree) Evict() *Node {
	t.mu.Lock()
	defer t.mu.Unlock()
	return t.evictLocked()
}

// EvictNode removes a specific leaf, applying the same parent-merge as Evict.
// It reports whether the node was removed; a non-leaf, referenced, nil, root,
// or detached node is not removed and returns false. Useful when the caller
// already holds the victim (for example to drop a known-cold prefix).
//
//	if tr.EvictNode(leaf) { engine.Free(leaf.Value) }
func (t *Tree) EvictNode(node *Node) bool {
	t.mu.Lock()
	defer t.mu.Unlock()
	if node == nil || node == t.root || node.parent == nil {
		return false
	}
	if len(node.children) != 0 || node.refs > 0 {
		return false
	}
	t.removeLeaf(node)
	return true
}

// EvictToCapacity evicts least-recently-used leaves until the node count is
// within MaxNodes (or nothing more is evictable), returning how many nodes were
// removed (including any merged parents). A no-op when unbounded or already
// within capacity.
//
//	freed := tr.EvictToCapacity()
func (t *Tree) EvictToCapacity() int {
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.maxNodes <= 0 {
		return 0
	}
	freed := 0
	for t.count > t.maxNodes {
		before := t.count
		if t.evictLocked() == nil {
			break // nothing left to evict
		}
		freed += before - t.count
	}
	return freed
}

// evictLocked finds and removes the LRU unreferenced leaf, returning it (or
// nil). Caller holds mu.
func (t *Tree) evictLocked() *Node {
	victim := t.lruLeaf()
	if victim == nil {
		return nil
	}
	t.removeLeaf(victim)
	return victim
}

// lruLeaf returns the least-recently-used unreferenced leaf, or nil if none.
// Caller holds mu.
func (t *Tree) lruLeaf() *Node {
	var best *Node
	t.walkLeaves(t.root, func(leaf *Node) {
		if leaf.refs > 0 {
			return
		}
		if best == nil || leaf.tick < best.tick {
			best = leaf
		}
	})
	return best
}

// walkLeaves visits every leaf under node (the root itself is never a leaf
// candidate). Caller holds mu.
func (t *Tree) walkLeaves(node *Node, visit func(*Node)) {
	if len(node.children) == 0 {
		if node != t.root {
			visit(node)
		}
		return
	}
	for _, c := range node.children {
		t.walkLeaves(c, visit)
	}
}

// removeLeaf detaches a leaf from its parent and applies the single-child
// parent merge. Caller holds mu and has verified leaf is a real, childless
// node.
func (t *Tree) removeLeaf(leaf *Node) {
	parent := leaf.parent
	delete(parent.children, leaf.edge[0])
	t.count--
	t.maybeMerge(parent)
}

// maybeMerge collapses an internal node that has been left with exactly one
// child into that child, concatenating their edges. Only valueless, unpinned,
// non-root internals are merged — a node that terminates a key, is referenced,
// or is the root keeps its identity. Caller holds mu.
//
//	// parent [1,2] with sole child [3] -> merged [1,2,3]
func (t *Tree) maybeMerge(node *Node) {
	if node == nil || node == t.root {
		return
	}
	if len(node.children) != 1 || node.Value != nil || node.refs > 0 {
		return
	}
	// Pull up the lone child into node, fusing the edges.
	var only *Node
	for _, c := range node.children {
		only = c
	}
	merged := make([]int, 0, len(node.edge)+len(only.edge))
	merged = append(merged, node.edge...)
	merged = append(merged, only.edge...)
	node.edge = merged
	node.Value = only.Value
	node.children = only.children
	for _, gc := range node.children {
		gc.parent = node
	}
	// node keeps its slot in its parent (edge[0] unchanged); the lone child
	// node is absorbed, so the live node count drops by one.
	t.count--
}

// cloneTokens copies a token run so the tree never aliases caller slices (an
// insert must not be mutated by a later caller reslice of the same backing
// array).
//
//	edge := cloneTokens(rest)
func cloneTokens(s []int) []int {
	out := make([]int, len(s))
	copy(out, s)
	return out
}

// Stats is a read-only snapshot of tree size for diagnostics and the result
// convention. Capacity is the configured MaxNodes (0 == unbounded).
//
//	s := tr.Stats(); core.Print(s.Nodes, "/", s.Capacity)
type Stats struct {
	Nodes    int
	Capacity int
	Over     bool
}

// Snapshot returns current size as a Core Result for callers that branch on
// r.OK — OK is false (carrying a scoped core.E) only when the tree is over
// capacity, so a watchdog can treat "over budget" as a failed result and
// trigger reclamation, otherwise it carries the Stats value.
//
//	if r := tr.Snapshot(); !r.OK { tr.EvictToCapacity() }
func (t *Tree) Snapshot() core.Result {
	t.mu.Lock()
	defer t.mu.Unlock()
	s := Stats{Nodes: t.count, Capacity: t.maxNodes, Over: t.maxNodes > 0 && t.count > t.maxNodes}
	if s.Over {
		return core.Fail(core.E("radix", "prefix tree over capacity", nil))
	}
	return core.Ok(s)
}
