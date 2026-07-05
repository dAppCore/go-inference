// SPDX-Licence-Identifier: EUPL-1.2

// Eviction policy: the LRU choice of which resident adapter to drop at capacity.

package lora

import "sync"

// EvictionPolicy decides which resident adapter to drop when the Pool is full. It
// tracks recency (MarkUsed), picks a victim restricted to the supplied evictable
// candidates (SelectVictim), and forgets an adapter once removed (Remove). It
// holds no adapter state beyond recency — the Pool owns residency and pinning and
// only ever offers genuinely evictable ids as candidates.
//
//	pol := lora.NewLRUEvictionPolicy()
//	pol.MarkUsed(id)
//	victim, ok := pol.SelectVictim(evictableIDs)
type EvictionPolicy interface {
	// MarkUsed records that an adapter was just accessed (most-recent). The empty
	// id is ignored.
	MarkUsed(id string)
	// SelectVictim returns the policy's choice of which candidate to evict, or
	// ok=false when no candidate is eligible. The candidate set is the Pool's set
	// of evictable (resident, unreferenced, unpinned) ids.
	SelectVictim(candidates []string) (id string, ok bool)
	// Remove drops an adapter from the policy's tracking (after it is evicted or
	// unregistered). The empty / unknown id is a no-op.
	Remove(id string)
}

// lruEvictionPolicy is the least-recently-used EvictionPolicy. Recency is a
// monotonic counter (not wall-clock), so victim selection is deterministic and
// reproducible in tests — the same access sequence always yields the same victim,
// matching the recency model in pkg/residency.
type lruEvictionPolicy struct {
	mu   sync.Mutex
	tick uint64
	used map[string]uint64 // id → last-use tick (higher == more recent)
}

// NewLRUEvictionPolicy builds an empty LRU policy ready to track adapter usage.
//
//	pol := lora.NewLRUEvictionPolicy()
func NewLRUEvictionPolicy() EvictionPolicy {
	return &lruEvictionPolicy{used: make(map[string]uint64)}
}

// MarkUsed stamps the adapter with the next monotonic tick, making it the
// most-recently-used. The empty id is ignored (mirrors SGLang's None handling).
func (p *lruEvictionPolicy) MarkUsed(id string) {
	if id == "" {
		return
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	p.tick++
	p.used[id] = p.tick
}

// SelectVictim returns the candidate with the lowest recency tick (the LRU),
// considering only candidates the policy has actually seen. An empty candidate
// set, or one containing no tracked id, returns ok=false.
func (p *lruEvictionPolicy) SelectVictim(candidates []string) (string, bool) {
	p.mu.Lock()
	defer p.mu.Unlock()
	var victim string
	var victimTick uint64
	found := false
	for _, id := range candidates {
		if id == "" {
			continue
		}
		t, seen := p.used[id]
		if !seen {
			continue
		}
		if !found || t < victimTick {
			victim, victimTick, found = id, t, true
		}
	}
	return victim, found
}

// Remove forgets an adapter's recency. The empty / unknown id is a no-op.
func (p *lruEvictionPolicy) Remove(id string) {
	if id == "" {
		return
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	delete(p.used, id)
}
