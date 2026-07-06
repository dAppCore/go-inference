// SPDX-Licence-Identifier: EUPL-1.2

// Pool: load-on-demand adapter serving over a capacity bound, plus the Loader boundary.

package lora

import (
	"context"
	"sort"
	"sync"

	core "dappco.re/go"
)

// Loader is the go-mlx apply/unload boundary. The Pool calls Load when an adapter
// must become resident on the base model and Unload when it is evicted or
// unregistered. The real implementation applies / detaches the LoRA delta on the
// device; this package never does, so it stays pure logic and the Loader is faked
// in tests.
//
//	type mlxLoader struct{ … }
//	func (l mlxLoader) Load(ctx context.Context, ref lora.AdapterRef) error { … }
//	func (l mlxLoader) Unload(ctx context.Context, id string) error          { … }
type Loader interface {
	// Load applies the adapter to the base model. A non-nil error aborts
	// admission — the adapter is not recorded resident.
	Load(ctx context.Context, ref AdapterRef) error
	// Unload detaches a previously loaded adapter by id.
	Unload(ctx context.Context, id string) error
}

// Config builds a Pool: the go-mlx Loader, the EvictionPolicy, and the Capacity
// (maximum adapters resident on the base model at once). Capacity is clamped to
// ≥ 0; a zero-capacity pool admits nothing.
type Config struct {
	Loader   Loader
	Policy   EvictionPolicy
	Capacity int
}

// Pool is the adapter serving manager. It composes a Registry (catalogue +
// ref-counts), a Loader (go-mlx apply/unload), and an EvictionPolicy over a
// capacity bound, exposing load-on-demand selection (Use), pinning, and residency
// queries. Safe for concurrent use.
//
//	pool := lora.NewPool(lora.Config{Loader: l, Policy: lora.NewLRUEvictionPolicy(), Capacity: 8})
type Pool struct {
	mu       sync.Mutex
	reg      *Registry
	loader   Loader
	policy   EvictionPolicy
	capacity int
	resident map[string]string // id → name, the working set on the base model
	pinned   map[string]bool   // id → pinned (never-evict)
}

// NewPool builds a serving pool from a Config.
//
//	pool := lora.NewPool(cfg)
func NewPool(cfg Config) *Pool {
	capN := max(cfg.Capacity, 0)
	return &Pool{
		reg:      NewRegistry(),
		loader:   cfg.Loader,
		policy:   cfg.Policy,
		capacity: capN,
		resident: make(map[string]string),
		pinned:   make(map[string]bool),
	}
}

// Register adds an adapter to the pool's catalogue (delegates to the Registry).
//
//	pool.Register(lora.AdapterRef{Name: "a", Path: "/x", BaseModel: "gemma-e4b"})
func (p *Pool) Register(ref AdapterRef) error { return p.reg.Register(ref) }

// Unregister removes a free adapter. If it is currently resident it is unloaded
// and dropped from the working set first; an in-flight adapter cannot be
// unregistered.
//
//	pool.Unregister("a")
func (p *Pool) Unregister(name string) error {
	id, ok := p.reg.idByName(name)
	if !ok {
		return core.E("ai", "lora: unknown adapter: "+name, nil)
	}

	p.mu.Lock()
	// Refuse before unloading if the adapter is in flight — keeps the catalogue
	// and the working set consistent.
	if p.reg.InUse(id) {
		p.mu.Unlock()
		return core.E("ai", "lora: adapter in use, cannot unregister: "+name, nil)
	}
	wasResident := p.resident[id] != ""
	if wasResident {
		delete(p.resident, id)
		delete(p.pinned, id)
		p.policy.Remove(id)
	}
	p.mu.Unlock()

	if wasResident {
		_ = p.loader.Unload(context.Background(), id)
	}
	return p.reg.Unregister(name)
}

// Use resolves the adapter named, ensures it is resident on the base model
// (loading it on demand, evicting the LRU evictable adapter when at capacity),
// takes an in-flight ref, and returns the adapter id plus a release closure. The
// adapter cannot be evicted between this call and release.
//
//	id, release, err := pool.Use(ctx, "support-tone")
//	if err != nil { return err }
//	defer release()
//
// Errors: an unknown name (registry error); an empty pool that still can't fit
// the adapter — Capacity 0 — yields a CannotFit error (see IsCannotFit); a full
// pool where every resident adapter is referenced or pinned yields a CannotAdmit
// error (see IsCannotAdmit); a Loader failure is surfaced verbatim and leaves
// nothing resident.
func (p *Pool) Use(ctx context.Context, name string) (string, func(), error) {
	// Resolve the ref once (its error is the unknown-name path), then take an
	// in-flight ref so the adapter is fenced against eviction for this call.
	ref, err := p.reg.Get(name)
	if err != nil {
		return "", nil, err
	}
	id, err := p.reg.Acquire(name)
	if err != nil {
		return "", nil, err
	}
	release := p.releaser(id)

	p.mu.Lock()

	// Resident hit: bump recency, return without reloading.
	if p.resident[id] != "" {
		p.policy.MarkUsed(id)
		p.mu.Unlock()
		return id, release, nil
	}

	// Capacity 0 → the adapter can never fit, even on an empty pool.
	if p.capacity == 0 {
		p.mu.Unlock()
		release()
		return "", nil, errCannotFit(name)
	}

	// At capacity → must evict an evictable adapter before loading.
	if len(p.resident) >= p.capacity {
		victim, ok := p.policy.SelectVictim(p.evictable())
		if !ok {
			// Everything resident is referenced or pinned — admission impossible.
			p.mu.Unlock()
			release()
			return "", nil, errCannotAdmit(name)
		}
		delete(p.resident, victim)
		delete(p.pinned, victim)
		p.policy.Remove(victim)
		p.mu.Unlock()

		// Unload the victim outside the lock (it is no longer resident, so no
		// concurrent Use can pick it).
		_ = p.loader.Unload(ctx, victim)

		p.mu.Lock()
	}

	// Reserve the slot before the (possibly slow) load so a concurrent Use sees
	// the adapter as resident and does not double-load it. On load failure the
	// reservation is rolled back.
	p.resident[id] = name
	p.policy.MarkUsed(id)
	p.mu.Unlock()

	if lerr := p.loader.Load(ctx, ref); lerr != nil {
		// Roll the reservation back so the slot is reusable and the failed
		// adapter is not reported resident.
		p.mu.Lock()
		delete(p.resident, id)
		delete(p.pinned, id)
		p.policy.Remove(id)
		p.mu.Unlock()
		release()
		return "", nil, lerr
	}

	return id, release, nil
}

// releaser returns an idempotent closure that drops exactly one in-flight ref for
// id. Calling it more than once is harmless (the Registry clamps at zero), but it
// only decrements on the first call to avoid releasing a ref it did not take.
func (p *Pool) releaser(id string) func() {
	var once sync.Once
	return func() {
		once.Do(func() { p.reg.Release(id) })
	}
}

// evictable returns the ids of resident adapters that may be evicted: resident,
// not pinned, and not in flight. The incoming adapter is never resident at the
// eviction point (a resident hit returns from Use before eviction), so it needs
// no special-casing here. Caller holds mu.
func (p *Pool) evictable() []string {
	out := make([]string, 0, len(p.resident))
	for id := range p.resident {
		if p.pinned[id] {
			continue
		}
		if p.reg.InUse(id) {
			continue
		}
		out = append(out, id)
	}
	return out
}

// Pin marks a resident adapter as never-evict. Pinning an adapter that is not
// resident is a no-op — pin protects something already loaded, mirroring
// residency.Pin.
//
//	pool.Use(ctx, "a"); pool.Pin("a") // keep a resident
func (p *Pool) Pin(name string) {
	id, ok := p.reg.idByName(name)
	if !ok {
		return
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.resident[id] != "" {
		p.pinned[id] = true
	}
}

// Unpin returns an adapter to normal eviction eligibility. No-op if absent or not
// resident.
//
//	pool.Unpin("a")
func (p *Pool) Unpin(name string) {
	id, ok := p.reg.idByName(name)
	if !ok {
		return
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	delete(p.pinned, id)
}

// IsResident reports whether the adapter named is currently loaded on the base
// model.
func (p *Pool) IsResident(name string) bool {
	id, ok := p.reg.idByName(name)
	if !ok {
		return false
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.resident[id] != ""
}

// Resident returns the names of the adapters currently loaded on the base model,
// sorted for deterministic output.
//
//	for _, name := range pool.Resident() { … }
func (p *Pool) Resident() []string {
	p.mu.Lock()
	defer p.mu.Unlock()
	names := make([]string, 0, len(p.resident))
	for _, name := range p.resident {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}
