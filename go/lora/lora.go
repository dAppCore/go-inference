// SPDX-Licence-Identifier: EUPL-1.2

// Package lora is the adapter-level multi-LoRA serving pool for the inference stack. One base
// model (held resident by the model-level pkg/residency policy) serves many LoRA
// adapters at once: each request selects an adapter by name, the Pool loads it on
// demand via a go-mlx Loader, keeps a bounded set resident, and evicts the
// least-recently-used adapter that is neither in-flight nor pinned when it hits
// capacity.
//
// Where pkg/residency reasons over MODELS and byte budgets — which whole models
// fit a 16 GB GPU / 96 GB M3 Ultra — this package reasons over ADAPTERS and a
// count cap: adapters are small (LoRA deltas), so the binding constraint is how
// many can be applied to the live base model at once, not their bytes. The two
// compose: residency keeps the base model loaded, this pool swaps adapters on top
// of it. Neither package touches a device; the caller injects the real go-mlx
// apply/unload behind the Loader interface and this package only decides what to
// load, what to evict, and which adapter is safe to evict.
//
//	pool := lora.NewPool(lora.Config{
//		Loader:   mlxLoader,                  // real go-mlx apply/unload
//		Policy:   lora.NewLRUEvictionPolicy(),
//		Capacity: 8,                          // max adapters resident at once
//	})
//	pool.Register(lora.AdapterRef{Name: "support-tone", Path: "/adapters/support", BaseModel: "gemma-e4b"})
//	id, release, err := pool.Use(ctx, "support-tone") // load-on-demand, ref-counted
//	if err != nil { return err }
//	defer release()                                    // drop the in-flight ref
//	// … run inference on the base model with adapter `id` applied …
//
// Ref-counting guarantees an adapter serving an in-flight request is never
// evicted: Use takes a ref, the returned release drops it, and only adapters with
// a zero ref-count (and not pinned) are eviction candidates.
package lora

import (
	"context"
	"slices"
	"sort"
	"sync"

	core "dappco.re/go"
)

// AdapterRef identifies one LoRA adapter: a human Name (the request-side selector
// and registry key), the Path the Loader applies from, and the BaseModel the
// adapter was trained against. The triple yields a stable ID — see ID.
//
//	r := lora.AdapterRef{Name: "support-tone", Path: "/adapters/support", BaseModel: "gemma-e4b"}
type AdapterRef struct {
	Name      string
	Path      string
	BaseModel string
}

// ID is the deterministic adapter id derived from Name and Path. Like SGLang's
// LoRARef.deterministic_id, it is stable across processes and machines for the
// same Name+Path so every node minting refs from the same --adapter-paths agrees
// on the id (a uuid4-style random id would diverge per process). The id is a
// content hash, so a re-pathed adapter of the same name is a distinct id.
//
//	lora.AdapterRef{Name: "a", Path: "/x"}.ID() // stable for ("a","/x")
func (r AdapterRef) ID() string {
	return core.SHA256HexString(deterministicSeed(r.Name, r.Path))
}

// deterministicSeed joins name and path with a NUL so ("ab","c") and ("a","bc")
// never collide. Caller-free helper, used only by ID.
func deterministicSeed(name, path string) string {
	return name + "\x00" + path
}

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

// entry is one registered adapter: its ref, the cached deterministic id, its
// current ref-count (outstanding Use leases), and whether it is currently
// resident on the base model. id is AdapterRef.ID() — a SHA-256 + hex of the
// immutable Name+Path — computed once at Register so the hot name→id paths
// (Acquire, IsResident, Pin) never re-hash.
type entry struct {
	ref      AdapterRef
	id       string
	refs     int
	resident bool
}

// Registry is the catalogue of known adapters with ref-counted leases. Register
// adds an adapter (keyed by Name), Acquire/Release fence in-flight use so the Pool
// never evicts an adapter mid-request, and Unregister removes a free adapter. It
// is safe for concurrent use.
//
//	reg := lora.NewRegistry()
//	reg.Register(lora.AdapterRef{Name: "a", Path: "/x"})
//	id, _ := reg.Acquire("a"); defer reg.Release(id)
type Registry struct {
	mu     sync.Mutex
	byName map[string]*entry
	byID   map[string]*entry
}

// NewRegistry builds an empty adapter registry.
//
//	reg := lora.NewRegistry()
func NewRegistry() *Registry {
	return &Registry{
		byName: make(map[string]*entry),
		byID:   make(map[string]*entry),
	}
}

// Register records a new adapter under its Name. A missing Name, or a Name that
// is already registered, returns a typed core error.
//
//	if err := reg.Register(ref); err != nil { … }
func (r *Registry) Register(ref AdapterRef) error {
	if ref.Name == "" {
		return core.E("ai", "lora: adapter name is required", nil)
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if _, ok := r.byName[ref.Name]; ok {
		return core.E("ai", "lora: adapter already registered: "+ref.Name, nil)
	}
	id := ref.ID()
	e := &entry{ref: ref, id: id}
	r.byName[ref.Name] = e
	r.byID[id] = e
	return nil
}

// Unregister removes a free adapter by Name. An unknown name, or an adapter with
// outstanding refs (an in-flight request), returns a typed core error so the Pool
// can never lose an adapter from under a live request.
func (r *Registry) Unregister(name string) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	e, ok := r.byName[name]
	if !ok {
		return core.E("ai", "lora: unknown adapter: "+name, nil)
	}
	if e.refs > 0 {
		return core.E("ai", "lora: adapter in use, cannot unregister: "+name, nil)
	}
	delete(r.byName, name)
	delete(r.byID, e.id)
	return nil
}

// Get returns the adapter ref registered under name, or a typed error if unknown.
//
//	ref, err := reg.Get("a")
func (r *Registry) Get(name string) (AdapterRef, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	e, ok := r.byName[name]
	if !ok {
		return AdapterRef{}, core.E("ai", "lora: unknown adapter: "+name, nil)
	}
	return e.ref, nil
}

// List returns every registered adapter ref, sorted by Name for deterministic
// output.
//
//	for _, ref := range reg.List() { … }
func (r *Registry) List() []AdapterRef {
	r.mu.Lock()
	defer r.mu.Unlock()
	out := make([]AdapterRef, 0, len(r.byName))
	for _, e := range r.byName {
		out = append(out, e.ref)
	}
	slices.SortFunc(out, func(a, b AdapterRef) int {
		switch {
		case a.Name < b.Name:
			return -1
		case a.Name > b.Name:
			return 1
		default:
			return 0
		}
	})
	return out
}

// Acquire bumps the ref-count for the adapter named and returns its id. While the
// count is non-zero the adapter is in use and the Pool will not evict it. An
// unknown name returns a typed error. Balance every Acquire with a Release.
//
//	id, err := reg.Acquire("a"); defer reg.Release(id)
func (r *Registry) Acquire(name string) (string, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	e, ok := r.byName[name]
	if !ok {
		return "", core.E("ai", "lora: unknown adapter: "+name, nil)
	}
	e.refs++
	return e.id, nil
}

// Release drops one ref for the adapter id. It clamps at zero, so an over-release
// or a release of an unknown id is a harmless no-op.
//
//	reg.Release(id)
func (r *Registry) Release(id string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	e, ok := r.byID[id]
	if !ok {
		return
	}
	if e.refs > 0 {
		e.refs--
	}
}

// RefCount reports the number of outstanding leases on the adapter id. An unknown
// id reports 0.
func (r *Registry) RefCount(id string) int {
	r.mu.Lock()
	defer r.mu.Unlock()
	if e, ok := r.byID[id]; ok {
		return e.refs
	}
	return 0
}

// InUse reports whether the adapter id has any outstanding lease (and is thus
// ineligible for eviction).
func (r *Registry) InUse(id string) bool {
	return r.RefCount(id) > 0
}

// idByName resolves name to its cached deterministic id, with ok=false for an
// unknown name. It lets the Pool's residency/pin paths get an id without
// recomputing AdapterRef.ID() (a SHA-256 + hex) on every call.
//
//	id, ok := reg.idByName("a")
func (r *Registry) idByName(name string) (string, bool) {
	r.mu.Lock()
	defer r.mu.Unlock()
	e, ok := r.byName[name]
	if !ok {
		return "", false
	}
	return e.id, true
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
	capN := cfg.Capacity
	if capN < 0 {
		capN = 0
	}
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

// fitError is the typed admission failure. Kind distinguishes a structural
// impossibility (CannotFit — the pool is too small even when empty) from a
// transient one (CannotAdmit — full of referenced/pinned adapters, retry once a
// lease is released). Test with IsCannotFit / IsCannotAdmit.
type fitError struct {
	kind string
	name string
}

const (
	kindCannotFit   = "cannot_fit"
	kindCannotAdmit = "cannot_admit"
)

// Error renders the admission failure via the Core error convention.
func (e *fitError) Error() string {
	switch e.kind {
	case kindCannotFit:
		return "lora: adapter cannot fit pool (capacity too small): " + e.name
	default:
		return "lora: cannot admit adapter, no evictable slot: " + e.name
	}
}

func errCannotFit(name string) error {
	return core.E("ai", (&fitError{kind: kindCannotFit, name: name}).Error(), &fitError{kind: kindCannotFit, name: name})
}

func errCannotAdmit(name string) error {
	return core.E("ai", (&fitError{kind: kindCannotAdmit, name: name}).Error(), &fitError{kind: kindCannotAdmit, name: name})
}

// IsCannotFit reports whether err is the structural "adapter can never fit this
// pool" failure (Capacity too small even when empty). The caller routes the
// request elsewhere rather than retrying.
//
//	if lora.IsCannotFit(err) { … route to another node … }
func IsCannotFit(err error) bool { return fitKind(err) == kindCannotFit }

// IsCannotAdmit reports whether err is the transient "no evictable slot" failure
// (the pool is full of in-flight or pinned adapters). The caller may retry once a
// lease is released.
//
//	if lora.IsCannotAdmit(err) { … backoff and retry … }
func IsCannotAdmit(err error) bool { return fitKind(err) == kindCannotAdmit }

// fitKind finds the kind of a fitError in err's chain via core.As (which walks
// the Core error tree, including the Cause of a core.E). Returns "" when err is
// not an admission failure.
func fitKind(err error) string {
	var fe *fitError
	if core.As(err, &fe) {
		return fe.kind
	}
	return ""
}
