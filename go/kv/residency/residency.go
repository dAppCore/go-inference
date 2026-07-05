// SPDX-Licence-Identifier: EUPL-1.2

// Package residency is the per-device model residency policy from RFC
// §6.16. Local memory is finite — the 16 GB GPU and the 96 GB M3 Ultra (RFC
// §6.2) hold only a few models at once — so each local runtime runs a Policy
// that loads a model on first request, keeps it resident, and evicts the
// least-recently-used non-pinned model under budget or concurrency pressure.
//
// The package is pure logic over model ids and byte sizes: it records WHICH
// models are resident and decides what to evict, but never loads a model or
// touches a device. The caller (the local runtime endpoint) owns go-mlx Close
// and the actual load — residency just tells it what to do.
//
//	p := residency.New(residency.Policy{
//		Device: "local-gpu", BudgetBytes: 16 << 30, ConcurrentCap: 4,
//		Warm: []residency.WarmModel{{ID: "gemma-e4b", SizeBytes: 4 << 30}},
//	})
//	d := p.Touch("qwen-q4", 8<<30) // load on first touch, LRU-evict to fit
//	if !d.Admitted { return d.Err() }
//	for _, id := range d.Evicted { runtime.Close(id) } // free GPU memory first
//	if d.Loaded { runtime.Load("qwen-q4") }
//
// Pinned / warm models (RFC §6.16 warm pool & pinning) are never evicted; an
// admission that cannot fit even after evicting every non-pinned model is
// rejected (Decision.Admitted == false) rather than touching the pinned set,
// so the caller falls out to another device or provider (RFC §6.2).
package residency

import (
	"cmp"
	"slices"
	"sync"

	core "dappco.re/go"
)

// Reason explains a Decision that did NOT admit a model. The zero value is
// ReasonNone — set on every admitted decision.
type Reason string

const (
	// ReasonNone is the reason on an admitted decision (model is resident).
	ReasonNone Reason = ""
	// ReasonTooLarge: the model exceeds the device budget even on an empty
	// device — it can never fit here, route it elsewhere (RFC §6.2 device-fit).
	ReasonTooLarge Reason = "too_large"
	// ReasonNoEvictableSpace: the model would fit an empty device, but the
	// resident pinned/warm set leaves too little budget (or too few cap slots)
	// and nothing non-pinned is evictable. Queue behind a load or fall back.
	ReasonNoEvictableSpace Reason = "no_evictable_space"
)

// WarmModel is a model pinned resident at construction (RFC §6.16 warm pool):
// the default Gemma 4 / Qwen are warmed at startup so the first request doesn't
// pay a load. A warm model that overflows the budget is skipped — the policy
// never holds a model the device can't budget for.
type WarmModel struct {
	ID        string
	SizeBytes int64
}

// Policy configures one device's residency rules. A device is a single local
// runtime endpoint with its own memory budget and quant profile (RFC §6.2):
// go-mlx on the M3 Ultra, or the CUDA/ROCm runtime on the 16 GB GPU.
type Policy struct {
	Device        string      // device / runtime label, for diagnostics
	BudgetBytes   int64       // resident set never exceeds this (clamped ≥ 0)
	ConcurrentCap int         // max models resident together (clamped ≥ 0)
	Warm          []WarmModel // pinned + resident from startup (warm pool)
}

// Decision is the outcome of a Touch: whether the model was admitted, whether a
// load is required, and which models the caller must Close to make room.
type Decision struct {
	ModelID  string // the touched model
	Admitted bool   // true → the model is resident after this Touch
	Loaded   bool   // true → caller must load it (first touch / reload). A
	// resident-hit re-touch is Admitted but not Loaded.
	Evicted []string // models to Close, in eviction (LRU-first) order
	Reason  Reason   // why not admitted (ReasonNone when Admitted)
}

// Err turns a Decision into the Core result convention (RFC.md §7 — core.E /
// core.Result). An admitted decision is core.Ok(d.ModelID); a rejection is a
// failed Result wrapping a scoped core.E so callers can branch on r.OK.
//
//	d := p.Touch(id, size)
//	if r := d.Err(); !r.OK { return r } // not admitted — fall back to provider
func (d Decision) Err() core.Result {
	if d.Admitted {
		return core.Ok(d.ModelID)
	}
	return core.Fail(core.E("ai", "model not admitted: "+d.ModelID+" ("+string(d.Reason)+")", nil))
}

// resident is one model held in the device's working set, with its size and the
// recency tick of its last touch (the LRU key — higher == more recent).
type resident struct {
	id     string
	size   int64
	pinned bool
	tick   uint64
}

// Policy state — guarded by mu so a runtime can Touch from multiple request
// goroutines (RFC §6.16 concurrency). LRU recency is a monotonic counter, so
// the policy is deterministic with no wall-clock dependency.
type policyState struct {
	mu     sync.Mutex
	budget int64
	cap    int
	tick   uint64
	models map[string]*resident
}

// Policy is opaque to callers; New returns *PolicyImpl behind the Policy config.
// (Kept as a distinct type so the config struct and the runtime aren't the same
// value — New consumes Policy, returns the running policy.)

// New builds a running residency policy from a Policy config, warming and
// pinning any Warm models that fit the budget and cap.
//
//	p := residency.New(residency.Policy{Device: "local-gpu", BudgetBytes: 16<<30, ConcurrentCap: 4})
func New(cfg Policy) *Manager {
	budget := cfg.BudgetBytes
	if budget < 0 {
		budget = 0
	}
	capN := cfg.ConcurrentCap
	if capN < 0 {
		capN = 0
	}
	m := &Manager{policyState{
		budget: budget,
		cap:    capN,
		models: make(map[string]*resident),
	}}
	// Warm the pool: pin + admit each warm model that fits within the running
	// budget and cap. A warm model that would overflow is skipped (RFC §6.16:
	// never hold a model the device can't budget for).
	for _, w := range cfg.Warm {
		if w.SizeBytes > m.s.budget {
			continue
		}
		if len(m.s.models) >= m.s.cap {
			continue
		}
		if m.s.used()+w.SizeBytes > m.s.budget {
			continue
		}
		m.s.tick++
		m.s.models[w.ID] = &resident{id: w.ID, size: w.SizeBytes, pinned: true, tick: m.s.tick}
	}
	return m
}

// Manager runs one device's residency policy. Construct with New. Safe to share
// across goroutines.
type Manager struct{ s policyState }

// used is the current resident byte total. Caller holds mu.
func (s *policyState) used() int64 {
	var total int64
	for _, r := range s.models {
		total += r.size
	}
	return total
}

// Touch marks modelID used at sizeBytes. If the model is already resident it is
// a hit — recency is bumped, no load, no eviction. Otherwise the policy admits
// it: it evicts the least-recently-used NON-pinned models (RFC §6.16 lazy load,
// LRU evict) until the new model fits both the byte budget and the concurrency
// cap, records it resident, and returns Loaded=true. If the model can't fit even
// on an empty device it is rejected ReasonTooLarge; if it would fit empty but
// the pinned/warm set leaves no evictable room it is rejected
// ReasonNoEvictableSpace — in both cases nothing resident is disturbed.
//
//	d := p.Touch("qwen-q4", 8<<30)
//	for _, id := range d.Evicted { runtime.Close(id) }
//	if d.Loaded { runtime.Load(d.ModelID) }
func (m *Manager) Touch(modelID string, sizeBytes int64) Decision {
	if sizeBytes < 0 {
		sizeBytes = 0
	}
	m.s.mu.Lock()
	defer m.s.mu.Unlock()

	// Hit: already resident → bump recency, update size, no load/evict.
	if r, ok := m.s.models[modelID]; ok {
		m.s.tick++
		r.tick = m.s.tick
		r.size = sizeBytes
		return Decision{ModelID: modelID, Admitted: true, Loaded: false}
	}

	// Can it ever fit this device? (RFC §6.2 device-fit gate.)
	if sizeBytes > m.s.budget {
		return Decision{ModelID: modelID, Admitted: false, Reason: ReasonTooLarge}
	}
	// A non-zero model can never sit on a zero-slot device.
	if m.s.cap == 0 {
		return Decision{ModelID: modelID, Admitted: false, Reason: ReasonNoEvictableSpace}
	}

	// Plan eviction: walk non-pinned residents LRU-first, marking models for
	// eviction until BOTH constraints are satisfiable for the newcomer.
	evicted := m.s.planEviction(sizeBytes)
	if evicted == nil {
		// nil (not empty) → constraints can't be met without evicting a pinned
		// model. Reject; leave the resident set untouched.
		return Decision{ModelID: modelID, Admitted: false, Reason: ReasonNoEvictableSpace}
	}

	// Commit the plan: remove the evicted models, then admit the newcomer.
	for _, id := range evicted {
		delete(m.s.models, id)
	}
	m.s.tick++
	m.s.models[modelID] = &resident{id: modelID, size: sizeBytes, pinned: false, tick: m.s.tick}
	return Decision{ModelID: modelID, Admitted: true, Loaded: true, Evicted: evicted}
}

// planEviction returns the LRU-ordered ids to evict so that a model of size
// `incoming` fits the budget and leaves a free cap slot. Pinned models are never
// candidates. Returns an empty (non-nil) slice when no eviction is needed, and
// nil when the constraints cannot be met without evicting a pinned model. Caller
// holds mu.
func (s *policyState) planEviction(incoming int64) []string {
	// Already room on both axes? No eviction needed.
	if s.used()+incoming <= s.budget && len(s.models) < s.cap {
		return []string{}
	}

	// Eviction candidates: non-pinned residents, LRU-first (lowest tick).
	candidates := make([]*resident, 0, len(s.models))
	for _, r := range s.models {
		if !r.pinned {
			candidates = append(candidates, r)
		}
	}
	slices.SortFunc(candidates, func(a, b *resident) int { return cmp.Compare(a.tick, b.tick) })

	pinnedBytes := int64(0)
	pinnedCount := 0
	for _, r := range s.models {
		if r.pinned {
			pinnedBytes += r.size
			pinnedCount++
		}
	}

	// Evict LRU-first until the newcomer fits memory AND a cap slot is free.
	// After evicting k candidates, residents = pinned + (len(candidates)-k),
	// and that must be < cap to leave room for the newcomer.
	evicted := make([]string, 0, len(candidates))
	freedBytes := int64(0)
	for i := 0; ; i++ {
		remainingCount := pinnedCount + (len(candidates) - len(evicted))
		usedBytes := pinnedBytes + (s.nonPinnedBytes(candidates) - freedBytes)
		memOK := usedBytes+incoming <= s.budget
		capOK := remainingCount < s.cap
		if memOK && capOK {
			return evicted
		}
		if i >= len(candidates) {
			// Exhausted every non-pinned model and still can't fit → only the
			// pinned set blocks it. Signal rejection (nil, not empty).
			return nil
		}
		victim := candidates[i]
		evicted = append(evicted, victim.id)
		freedBytes += victim.size
	}
}

// nonPinnedBytes totals the sizes of the candidate (non-pinned) residents.
// Caller holds mu.
func (s *policyState) nonPinnedBytes(candidates []*resident) int64 {
	var total int64
	for _, r := range candidates {
		total += r.size
	}
	return total
}

// Resident returns the ids currently held in the working set, sorted for
// deterministic output.
//
//	for _, id := range p.Resident() { … }
func (m *Manager) Resident() []string {
	m.s.mu.Lock()
	defer m.s.mu.Unlock()
	ids := make([]string, 0, len(m.s.models))
	for id := range m.s.models {
		ids = append(ids, id)
	}
	slices.Sort(ids)
	return ids
}

// IsResident reports whether modelID is currently held resident.
func (m *Manager) IsResident(modelID string) bool {
	m.s.mu.Lock()
	defer m.s.mu.Unlock()
	_, ok := m.s.models[modelID]
	return ok
}

// Pin marks a resident model as never-evict (RFC §6.16 pinning). Pinning a model
// that isn't resident is a no-op — the warm pool is the way to admit-and-pin at
// startup; Pin only protects something already loaded.
//
//	p.Touch("gemma-e4b", 4<<30); p.Pin("gemma-e4b") // keep it resident
func (m *Manager) Pin(modelID string) {
	m.s.mu.Lock()
	defer m.s.mu.Unlock()
	if r, ok := m.s.models[modelID]; ok {
		r.pinned = true
	}
}

// Unpin returns a model to normal LRU eviction eligibility. No-op if absent.
func (m *Manager) Unpin(modelID string) {
	m.s.mu.Lock()
	defer m.s.mu.Unlock()
	if r, ok := m.s.models[modelID]; ok {
		r.pinned = false
	}
}

// IsPinned reports whether a resident model is currently pinned.
func (m *Manager) IsPinned(modelID string) bool {
	m.s.mu.Lock()
	defer m.s.mu.Unlock()
	r, ok := m.s.models[modelID]
	return ok && r.pinned
}
