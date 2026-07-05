// SPDX-Licence-Identifier: EUPL-1.2

// Package kvtier is the hierarchical KV-cache tiering policy for local
// inference. The attention KV cache is the memory hog of long-context
// generation — on the 16 GB GPU (RFC §6.2) only a slice of it fits — so
// this policy keeps the HOT KV blocks on the GPU within a byte budget and spills
// cold blocks down the hierarchy GPU → CPU → Disk, promoting a block back to the
// GPU the moment it is touched again.
//
// The package is pure placement logic over block ids and byte sizes. It records
// WHICH tier each block sits in and decides what to move, but never copies a
// byte: the real offload/reload is an injected Store. A runtime wires its
// CUDA/host/mmap copier behind Store; the tests wire a fake. This is the KV-cache
// sibling of the whole-model `residency` policy in the same module.
//
//	fs := myRuntimeStore{}                       // real GPU<->CPU<->disk copier
//	m := kvtier.New(kvtier.Budget{
//		GPU:  16 << 30, // bytes of KV cache the GPU will hold
//		CPU:  64 << 30,
//		Disk: 512 << 30,
//	}, fs)
//	if err := m.Put(ctx, kvtier.Block{ID: "seq42:layer0", SizeBytes: 8 << 20}); err != nil {
//		return err // block bigger than the GPU itself — route elsewhere
//	}
//	_ = m.Access(ctx, "seq42:layer0") // touched again → promote back to GPU
//
// Placement is deterministic: recency is a monotonic tick (the LRU key), so the
// same sequence of operations always produces the same tier layout, with no
// wall-clock dependency. Pinned blocks are never demoted off the GPU.
package kvtier

import (
	"cmp"
	"context"
	"slices"
	"sync"

	core "dappco.re/go"
)

// Tier names a level of the KV-cache hierarchy, ordered hot → cold. TierNone is
// the zero value and means "not tracked / not resident" — TierOf returns it for
// an unknown block. Lower numeric value == hotter (GPU < CPU < Disk).
type Tier int

const (
	// TierNone is the zero value: the block is not held in any tier.
	TierNone Tier = iota
	// TierGPU is the hot tier — KV blocks the GPU is actively attending over.
	TierGPU
	// TierCPU is the warm spill tier — host RAM, a copy away from the GPU.
	TierCPU
	// TierDisk is the cold backstop — mmap'd / on-disk KV, assumed large.
	TierDisk
)

// String renders a Tier for diagnostics and move logs.
//
//	core.Println(kvtier.TierGPU.String()) // "gpu"
func (t Tier) String() string {
	switch t {
	case TierGPU:
		return "gpu"
	case TierCPU:
		return "cpu"
	case TierDisk:
		return "disk"
	case TierNone:
		return "none"
	default:
		return "unknown"
	}
}

// Block is a unit of KV cache the policy places: an opaque id and its byte size.
// The id is whatever the runtime keys its cache on (e.g. "seq:layer:page").
//
//	b := kvtier.Block{ID: "seq42:layer0", SizeBytes: 8 << 20}
type Block struct {
	ID        string
	SizeBytes int64
}

// Store performs the real movement of a KV block between tiers — the GPU↔host
// copy or the host↔disk offload. The policy calls Move once per hop it decides
// on; a returned error aborts the operation and the policy rolls its in-memory
// accounting back so a half-applied move never corrupts the tier map. `to` ==
// TierNone means "drop the block from `from`" (an evict/remove).
//
//	func (s runtimeStore) Move(ctx context.Context, id string, from, to kvtier.Tier) error {
//		return s.copy(ctx, id, from, to) // cudaMemcpy / pwrite / free
//	}
type Store interface {
	Move(ctx context.Context, blockID string, from, to Tier) error
}

// Budget is the per-tier byte ceiling. The GPU and CPU tiers are bounded; Disk is
// the backstop and is treated as effectively unbounded — a non-positive Disk
// budget is taken to mean "no limit". Negative budgets are floored to 0.
//
//	kvtier.Budget{GPU: 16 << 30, CPU: 64 << 30, Disk: 512 << 30}
type Budget struct {
	GPU  int64
	CPU  int64
	Disk int64
}

// Typed errors. Callers branch with errors.Is — the descriptive forms returned
// by the manager wrap these sentinels so the id-carrying message and the typed
// identity travel together.
//
//	if err := m.Put(ctx, b); errors.Is(err, kvtier.ErrTooLarge) { … }
var (
	// ErrTooLarge: the block exceeds the GPU budget even on an empty GPU, so it
	// can never be placed in the hot tier — route it elsewhere.
	ErrTooLarge = core.E("ai", "kv block exceeds gpu budget", nil)
	// ErrUnknownBlock: Access was asked to promote a block the manager has never
	// tracked.
	ErrUnknownBlock = core.E("ai", "kv block not found", nil)
	// ErrStore: the injected Store failed to move a block; the manager rolled its
	// accounting back to the pre-operation state.
	ErrStore = core.E("ai", "kv store move failed", nil)
)

// entry is one tracked KV block: its id (== its map key, carried so the planner
// can build moves and sort demotion candidates without a parallel id slice), its
// size, current tier, pin state, and the recency tick of its last touch (the LRU
// key — higher == more recent). proj is transient scratch: planRebalance projects
// each block's tier into it while building a plan, so a block demoted GPU→CPU can
// be re-considered for CPU→Disk in the same pass — without copying the whole tier
// map. It is meaningful only inside a single locked planRebalance call.
type entry struct {
	id     string
	size   int64
	tick   uint64
	tier   Tier
	proj   Tier
	pinned bool
}

// Manager runs one device's KV-cache tiering policy. Construct with New. Safe to
// share across goroutines — every operation takes the manager lock so concurrent
// request goroutines see a consistent tier map.
type Manager struct {
	mu     sync.Mutex
	store  Store
	budget Budget
	tick   uint64
	blocks map[string]*entry
	// Planning scratch, reused under mu so the per-token rebalance path allocates
	// nothing after warmup. cand collects the LRU demotion candidates to sort;
	// plan is the move plan handed to execute (Access prepends its promote hop into
	// it). Neither is read or retained outside a locked planRebalance/rebalance/
	// Access, so reusing the backing arrays across calls is safe.
	cand []*entry
	plan []plannedMove
}

// New builds a tiering manager over a per-tier byte Budget and an injected Store.
// Negative budgets are floored to 0.
//
//	m := kvtier.New(kvtier.Budget{GPU: 16 << 30, CPU: 64 << 30, Disk: 512 << 30}, store)
func New(b Budget, store Store) *Manager {
	if b.GPU < 0 {
		b.GPU = 0
	}
	if b.CPU < 0 {
		b.CPU = 0
	}
	if b.Disk < 0 {
		b.Disk = 0
	}
	return &Manager{
		store:  store,
		budget: b,
		blocks: make(map[string]*entry),
	}
}

// limitOf returns the enforced byte ceiling for the two bounded tiers, GPU and
// CPU. Disk is the backstop — it has no enforced ceiling (the spec assumes it is
// unbounded or large), so rebalance never treats Disk as an overflow source and
// limitOf is only ever asked about GPU and CPU.
func (m *Manager) limitOf(t Tier) int64 {
	if t == TierGPU {
		return m.budget.GPU
	}
	return m.budget.CPU // the only other source rebalance passes is TierCPU
}

// plannedMove is one hop the policy intends to apply: move id from→to. A move
// with to == TierNone drops the block. Plans are built fully before any Store
// call so a failure can be rolled back cleanly.
type plannedMove struct {
	id   string
	from Tier
	to   Tier
}

// Put places a new KV block on the GPU, demoting least-recently-used blocks down
// the hierarchy (GPU→CPU, and CPU→Disk if the CPU tier overflows) until every
// bounded tier is within budget. Re-Put of an existing id updates its size and
// recency in place and re-balances. A block larger than the GPU budget even on an
// empty GPU is rejected with ErrTooLarge and nothing is moved.
//
//	if err := m.Put(ctx, kvtier.Block{ID: "seq:l0", SizeBytes: 8 << 20}); err != nil { … }
func (m *Manager) Put(ctx context.Context, b Block) error {
	size := b.SizeBytes
	if size < 0 {
		size = 0
	}
	m.mu.Lock()
	defer m.mu.Unlock()

	// Can it ever sit in the hot tier? (Empty-GPU fit gate.)
	if size > m.budget.GPU {
		return core.Wrap(ErrTooLarge, "ai", "put: "+b.ID)
	}

	m.tick++
	if e, ok := m.blocks[b.ID]; ok {
		// Re-Put: refresh size + recency, pull back to GPU, then re-balance.
		e.size = size
		e.tick = m.tick
		e.tier = TierGPU
	} else {
		m.blocks[b.ID] = &entry{id: b.ID, size: size, tier: TierGPU, tick: m.tick}
	}

	if err := m.rebalance(ctx); err != nil {
		// rebalance rolled the tier map back; undo this Put's bookkeeping too.
		if e, ok := m.blocks[b.ID]; ok && e.tick == m.tick {
			delete(m.blocks, b.ID)
		}
		return err
	}
	return nil
}

// Access promotes blockID to the GPU (demoting other GPU blocks down the
// hierarchy as needed), marks it most-recently-used, and returns nil. A block
// already on the GPU is a hit: recency is bumped, nothing moves. An unknown id
// returns ErrUnknownBlock.
//
//	if err := m.Access(ctx, "seq:l0"); errors.Is(err, kvtier.ErrUnknownBlock) { … }
func (m *Manager) Access(ctx context.Context, blockID string) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	e, ok := m.blocks[blockID]
	if !ok {
		return core.Wrap(ErrUnknownBlock, "ai", "access: "+blockID)
	}

	m.tick++
	e.tick = m.tick
	if e.tier == TierGPU {
		return nil // hit — already hot, recency bumped.
	}
	from := e.tier
	// Mark the block hot (and newest, above) so the demotion planner spares it,
	// then build ONE atomic plan: the promote hop first, then any demotions it
	// forces. Sharing a plan keeps promote+demote all-or-nothing.
	e.tier = TierGPU
	m.plan = append(m.plan[:0], plannedMove{id: blockID, from: from, to: TierGPU})
	m.plan = m.planRebalance(m.plan)
	if err := m.execute(ctx, m.plan); err != nil {
		e.tier = from // roll the in-memory promotion back; execute undid the rest.
		return err
	}
	return nil
}

// rebalance demotes least-recently-used UNPINNED blocks down the hierarchy until
// every bounded tier (GPU, CPU) is within budget, cascading GPU→CPU→Disk. It is
// the placement step after a Put marks a newcomer on the GPU. Caller holds mu.
func (m *Manager) rebalance(ctx context.Context) error {
	m.plan = m.planRebalance(m.plan[:0])
	return m.execute(ctx, m.plan)
}

// execute runs a move plan through the Store and only then commits the tier
// changes in memory. A Store failure on any hop rolls back the hops already
// applied (in reverse) and returns ErrStore, so the manager's accounting never
// reflects a move that did not happen. An empty plan is a no-op. Caller holds mu.
func (m *Manager) execute(ctx context.Context, plan []plannedMove) error {
	if len(plan) == 0 {
		return nil
	}
	for i, p := range plan {
		if err := m.store.Move(ctx, p.id, p.from, p.to); err != nil {
			m.rollback(ctx, plan[:i])
			return core.Wrap(ErrStore, "ai", "move: "+p.id)
		}
	}
	for _, p := range plan {
		if e, ok := m.blocks[p.id]; ok {
			e.tier = p.to
		}
	}
	return nil
}

// planRebalance walks GPU then CPU, and for each over-budget tier selects its
// LRU unpinned blocks to demote one tier colder until the tier fits (or no more
// unpinned blocks remain — pinned blocks are immovable backstops). Demotion hops
// are appended to plan (so Access can pass a slice already holding its promote
// hop, sharing one buffer); the result is in execution order (coldest cascade
// resolved as we descend). The projected tier is tracked on each entry's transient
// proj field rather than a per-call map copy, and candidates use the reused m.cand
// scratch — both keep the per-token path allocation-free. Caller holds mu.
func (m *Manager) planRebalance(plan []plannedMove) []plannedMove {
	// Seed each block's projected tier so a block demoted GPU→CPU can be
	// re-considered for CPU→Disk in the same pass.
	for _, e := range m.blocks {
		e.proj = e.tier
	}

	for _, src := range [2]Tier{TierGPU, TierCPU} {
		dst := src + 1 // GPU→CPU, CPU→Disk
		limit := m.limitOf(src)
		// Bytes currently projected in src.
		used := int64(0)
		for _, e := range m.blocks {
			if e.proj == src {
				used += e.size
			}
		}
		if used <= limit {
			continue
		}
		// Candidates: unpinned blocks projected in src, LRU-first.
		cand := m.cand[:0]
		for _, e := range m.blocks {
			if e.proj == src && !e.pinned {
				cand = append(cand, e)
			}
		}
		slices.SortFunc(cand, func(a, b *entry) int {
			return cmp.Compare(a.tick, b.tick)
		})
		m.cand = cand
		for _, e := range cand {
			if used <= limit {
				break
			}
			plan = append(plan, plannedMove{id: e.id, from: src, to: dst})
			e.proj = dst
			used -= e.size
		}
		// If still over budget after evicting every unpinned block, the pinned
		// set legitimately holds the tier above budget — leave it (pinned wins).
	}
	return plan
}

// rollback reverses the already-applied Store hops after a mid-plan failure, in
// reverse order, on a best-effort basis (the in-memory tiers were not committed,
// so only the Store side needs undoing). Caller holds mu.
func (m *Manager) rollback(ctx context.Context, applied []plannedMove) {
	for i := len(applied) - 1; i >= 0; i-- {
		p := applied[i]
		_ = m.store.Move(ctx, p.id, p.to, p.from)
	}
}

// Evict drops blockID from whatever tier holds it, calling the Store to free the
// underlying memory (a Move to TierNone). Unknown id is a no-op. Evict is the
// explicit cousin of the automatic demotion in Put/Access.
//
//	_ = m.Evict(ctx, "seq:l0") // free this block's KV everywhere
func (m *Manager) Evict(ctx context.Context, blockID string) error {
	return m.Remove(ctx, blockID)
}

// Remove forgets blockID entirely, freeing its memory via the Store. Unknown id
// is a quiet no-op so callers can remove defensively.
//
//	_ = m.Remove(ctx, "seq:l0")
func (m *Manager) Remove(ctx context.Context, blockID string) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	e, ok := m.blocks[blockID]
	if !ok {
		return nil
	}
	if err := m.store.Move(ctx, blockID, e.tier, TierNone); err != nil {
		return core.Wrap(ErrStore, "ai", "remove: "+blockID)
	}
	delete(m.blocks, blockID)
	return nil
}

// Pin marks a resident block as never-demote: it stays on the GPU through any
// number of Put/Access pressure rounds. Pinning an unknown block is a no-op.
//
//	m.Pin("seq:l0") // keep this sequence's KV hot
func (m *Manager) Pin(blockID string) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if e, ok := m.blocks[blockID]; ok {
		e.pinned = true
	}
}

// Unpin returns a block to normal LRU demotion eligibility. No-op if unknown.
//
//	m.Unpin("seq:l0")
func (m *Manager) Unpin(blockID string) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if e, ok := m.blocks[blockID]; ok {
		e.pinned = false
	}
}

// IsPinned reports whether a tracked block is currently pinned.
func (m *Manager) IsPinned(blockID string) bool {
	m.mu.Lock()
	defer m.mu.Unlock()
	e, ok := m.blocks[blockID]
	return ok && e.pinned
}

// TierOf reports which tier holds blockID, or TierNone if it is not tracked.
//
//	if m.TierOf("seq:l0") == kvtier.TierGPU { … }
func (m *Manager) TierOf(blockID string) Tier {
	m.mu.Lock()
	defer m.mu.Unlock()
	if e, ok := m.blocks[blockID]; ok {
		return e.tier
	}
	return TierNone
}

// IsResident reports whether blockID is tracked in any tier.
func (m *Manager) IsResident(blockID string) bool {
	m.mu.Lock()
	defer m.mu.Unlock()
	_, ok := m.blocks[blockID]
	return ok
}

// Resident lists the block ids held in a tier, sorted for deterministic output.
// An empty or unknown tier returns an empty (non-nil) slice.
//
//	for _, id := range m.Resident(kvtier.TierGPU) { … }
func (m *Manager) Resident(t Tier) []string {
	m.mu.Lock()
	defer m.mu.Unlock()
	// Size the result exactly (one count pass, no alloc) so the returned slice is
	// a single allocation with no geometric regrow.
	n := 0
	for _, e := range m.blocks {
		if e.tier == t {
			n++
		}
	}
	ids := make([]string, 0, n)
	for id, e := range m.blocks {
		if e.tier == t {
			ids = append(ids, id)
		}
	}
	slices.Sort(ids)
	return ids
}

// Len reports the total number of blocks tracked across every tier.
func (m *Manager) Len() int {
	m.mu.Lock()
	defer m.mu.Unlock()
	return len(m.blocks)
}
