// SPDX-Licence-Identifier: EUPL-1.2

// serve_multimodel.go is the multi-model serving registry: it holds several
// models resident at once, routes an OpenAI-API `model` field (or a
// `model:profile` id) to the right one, loads on demand, and evicts under a
// byte budget by LRU + idle TTL. It is the oMLX-parity multi-model lifecycle for
// go-inference serve; the single-model hotSwapResolver (serve_resolver.go) is
// untouched and stays the default — RunServe only builds this when multi-model
// models are declared.
//
// multiModelResolver implements the same openai.Resolver seam the chat handler
// already consumes (ResolveModel), so nothing downstream changes: the request
// names a model, and here that name finally routes rather than being ignored.
//
// Memory ceiling: a model's residency cost is estimated from its on-disk pack
// size (weights dominate the resident footprint; go-inference exposes no
// engine-neutral live-memory read). Before a load, LRU-unpinned residents are
// evicted until the incoming model fits; a model that cannot fit even after
// evicting every unpinned resident, or one larger than the whole ceiling, is
// refused. Pinned models are exempt from both LRU and idle-TTL eviction.
//
// Eviction Closes the victim to reclaim GPU memory — a memory ceiling is only
// real if the bytes actually come back. This diverges from the hot-swap reload
// drain policy (which relies on GC): eviction is automatic and frequent, so it
// cannot wait for GC to free the device. LRU/idle victim selection targets the
// least-active model, which is the one least likely to be mid-stream; eager
// close under an active long-running stream is the residual caveat (a per-entry
// in-flight refcount with deferred close is the follow-on).

package serving

import (
	"context"
	"io"
	"slices"
	"sync"
	"time"

	core "dappco.re/go"
	"dappco.re/go/inference"
	openai "dappco.re/go/inference/serving/provider/openai"
)

// ModelSpec declares one model in the multi-model registry.
type ModelSpec struct {
	// ID is the canonical request name; "" derives from the Path basename.
	ID string
	// Path is the on-disk model directory.
	Path string
	// Aliases are alternate request names that route to this model.
	Aliases []string
	// Pinned exempts the model from LRU and idle-TTL eviction.
	Pinned bool
	// EstBytes overrides the residency-budget cost; 0 measures the pack size
	// from Path on disk.
	EstBytes uint64
	// LoadOptions are forwarded to the loader (context length, adapter, …).
	LoadOptions []inference.LoadOption
	// Profiles are named generation-option presets, exposed as `id:profile` ids
	// in /v1/models and selectable in a request's model field.
	Profiles map[string]ProfileConfig
}

// MultiModelOptions tunes the registry's residency budget and idle policy.
type MultiModelOptions struct {
	// MemoryCeiling caps the total estimated bytes of all resident models; 0 is
	// unbounded (models stay resident until idle-TTL or an explicit unload).
	MemoryCeiling uint64
	// IdleTTL evicts an unpinned model idle longer than this; 0 never idle-evicts.
	IdleTTL time.Duration
	// SweepInterval is the background idle-sweep tick; 0 lets RunServe default it.
	SweepInterval time.Duration
}

// modelEntry is one registry slot. model is nil until the first resolve (lazy
// load); lastUsed drives both LRU victim selection and idle-TTL eviction.
type modelEntry struct {
	id       string
	path     string
	pinned   bool
	estBytes uint64
	loadOpts []inference.LoadOption
	profiles map[string][]inference.GenerateOption
	model    inference.TextModel
	lastUsed time.Time
}

// multiModelResolver holds the registry and satisfies openai.Resolver plus the
// admin Reloader seam. A single mutex guards every field: the hot path (resolve
// of a resident model) and the mutating paths (load, evict, pin, sweep) all take
// it, so the resolver is race-free by construction. Loads run under the lock —
// first-loads of distinct models serialise; that is the naive-correct baseline,
// with per-entry load locks a later optimisation.
type multiModelResolver struct {
	mu        sync.Mutex
	entries   map[string]*modelEntry
	aliases   map[string]string // lower(id) and lower(alias) → canonical id
	order     []string          // id insertion order, for stable listing
	defaultID string            // unknown/blank names resolve here

	opts   MultiModelOptions
	loader ModelLoader
	sizer  func(path string) uint64
	onLoad func(inference.TextModel)
	now    func() time.Time
	log    io.Writer
}

// newMultiModelResolver builds the registry from specs. It validates that at
// least one spec is present, every spec carries a Path, and ids and aliases do
// not collide. The default model (unknown/blank request names route to it) is
// the first pinned spec, or the first spec when none is pinned.
func newMultiModelResolver(specs []ModelSpec, opts MultiModelOptions) (*multiModelResolver, error) {
	if len(specs) == 0 {
		return nil, core.E("serving.newMultiModelResolver", "no models declared", nil)
	}
	r := &multiModelResolver{
		entries: make(map[string]*modelEntry, len(specs)),
		aliases: make(map[string]string, len(specs)),
		opts:    opts,
		loader:  metalTextModelLoader,
		sizer:   estimateModelBytes,
		now:     time.Now,
	}
	for _, spec := range specs {
		if err := r.register(spec); err != nil {
			return nil, err
		}
	}
	r.defaultID = r.order[0]
	for _, id := range r.order {
		if r.entries[id].pinned {
			r.defaultID = id
			break
		}
	}
	return r, nil
}

// register adds one spec to the registry, deriving the id from the path basename
// when unset, sizing the pack, pre-building profiles, and wiring the alias map.
func (r *multiModelResolver) register(spec ModelSpec) error {
	path := core.Trim(spec.Path)
	if path == "" {
		return core.E("serving.multiModelResolver", "model spec has no path", nil)
	}
	id := core.Trim(spec.ID)
	if id == "" {
		id = core.PathBase(path)
	}
	key := core.Lower(id)
	if _, dup := r.aliases[key]; dup {
		return core.E("serving.multiModelResolver", core.Sprintf("duplicate model id/alias %q", id), nil)
	}
	estBytes := spec.EstBytes
	if estBytes == 0 && r.sizer != nil {
		estBytes = r.sizer(path)
	}
	profiles := make(map[string][]inference.GenerateOption, len(spec.Profiles))
	for name, cfg := range spec.Profiles {
		profiles[core.Lower(core.Trim(name))] = cfg.Options()
	}
	entry := &modelEntry{id: id, path: path, pinned: spec.Pinned, estBytes: estBytes, loadOpts: spec.LoadOptions, profiles: profiles}
	r.entries[id] = entry
	r.order = append(r.order, id)
	r.aliases[key] = id
	for _, alias := range spec.Aliases {
		ak := core.Lower(core.Trim(alias))
		if ak == "" {
			continue
		}
		if _, dup := r.aliases[ak]; dup {
			return core.E("serving.multiModelResolver", core.Sprintf("duplicate model id/alias %q", alias), nil)
		}
		r.aliases[ak] = id
	}
	return nil
}

// setLoader swaps the model loader (nil-safe: keeps the default metal loader).
func (r *multiModelResolver) setLoader(loader ModelLoader) {
	if loader != nil {
		r.loader = loader
	}
}

// setOnLoad registers a hook run after every successful load, so per-model
// wiring (conversation continuity) re-attaches to each loaded model.
func (r *multiModelResolver) setOnLoad(hook func(inference.TextModel)) {
	r.onLoad = hook
}

// setLog directs eviction + load notices to w.
func (r *multiModelResolver) setLog(w io.Writer) {
	r.log = w
}

// ResolveModel routes name to a resident model, loading (and evicting to fit)
// when required. name is the OpenAI-API `model` field: an exact id/alias match
// wins; a `model:profile` id resolves the model and applies the named preset; an
// unknown name falls back to the default model (the friendly single-model UX —
// a client echoing an arbitrary name still gets served). A profile named against
// a known model that does not define it is an error.
func (r *multiModelResolver) ResolveModel(ctx context.Context, name string) (inference.TextModel, error) {
	if ctx != nil {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
		}
	}
	r.mu.Lock()
	defer r.mu.Unlock()

	entry, preset, err := r.route(name)
	if err != nil {
		return nil, err
	}
	model, err := r.ensureResident(entry)
	if err != nil {
		return nil, err
	}
	entry.lastUsed = r.now()
	return wrapProfile(model, preset), nil
}

// route maps a request name to its entry and (when a profile is named) the
// resolved preset options. An exact id/alias match is tried first so an id that
// itself contains a colon is never mis-split. Otherwise the name is cut on the
// first colon into model + profile.
func (r *multiModelResolver) route(name string) (*modelEntry, []inference.GenerateOption, error) {
	if id, ok := r.aliases[core.Lower(core.Trim(name))]; ok {
		return r.entries[id], nil, nil
	}
	modelPart, profilePart, hasColon := core.Cut(name, ":")
	if hasColon {
		if id, ok := r.aliases[core.Lower(core.Trim(modelPart))]; ok {
			entry := r.entries[id]
			preset, ok := entry.profiles[core.Lower(core.Trim(profilePart))]
			if !ok {
				return nil, nil, core.E("serving.multiModelResolver", core.Sprintf("model %q has no profile %q", entry.id, core.Trim(profilePart)), nil)
			}
			return entry, preset, nil
		}
	}
	// Unknown model name → default. A colon in an unknown name is cosmetic and
	// carries no profile (only a known model's declared profiles resolve).
	return r.entries[r.defaultID], nil, nil
}

// ensureResident returns the entry's loaded model, loading it (after making room
// within the budget) when it is not yet resident.
func (r *multiModelResolver) ensureResident(entry *modelEntry) (inference.TextModel, error) {
	if entry.model != nil {
		return entry.model, nil
	}
	if err := r.makeRoom(entry); err != nil {
		return nil, err
	}
	model, err := r.loader(entry.path, entry.loadOpts...)
	if err != nil {
		return nil, err
	}
	if r.onLoad != nil {
		r.onLoad(model)
	}
	entry.model = model
	printServe(r.log, "serve: loaded model %s (~%d bytes, resident ~%d/%d)", entry.id, entry.estBytes, r.residentBytes(), r.opts.MemoryCeiling)
	return model, nil
}

// makeRoom evicts LRU-unpinned residents until incoming fits under the ceiling.
// A model larger than the whole ceiling, or one that cannot fit even after every
// unpinned resident is evicted, is refused rather than loaded over budget.
func (r *multiModelResolver) makeRoom(incoming *modelEntry) error {
	ceiling := r.opts.MemoryCeiling
	if ceiling == 0 {
		return nil // unbounded
	}
	if incoming.estBytes > ceiling {
		return core.E("serving.multiModelResolver", core.Sprintf("model %q (~%d bytes) exceeds the memory ceiling (%d bytes)", incoming.id, incoming.estBytes, ceiling), nil)
	}
	for r.residentBytes()+incoming.estBytes > ceiling {
		victim := r.lruUnpinnedResident(incoming.id)
		if victim == nil {
			return core.E("serving.multiModelResolver", core.Sprintf("cannot fit model %q (~%d bytes) within the memory ceiling (%d bytes) — resident ~%d bytes, no unpinned model left to evict", incoming.id, incoming.estBytes, ceiling, r.residentBytes()), nil)
		}
		r.evict(victim, "budget")
	}
	return nil
}

// residentBytes sums the estimated bytes of every currently-loaded model.
func (r *multiModelResolver) residentBytes() uint64 {
	var total uint64
	for _, id := range r.order {
		if e := r.entries[id]; e.model != nil {
			total += e.estBytes
		}
	}
	return total
}

// lruUnpinnedResident returns the loaded, unpinned entry (other than excludeID)
// with the oldest lastUsed — the LRU eviction victim. nil when none qualifies.
func (r *multiModelResolver) lruUnpinnedResident(excludeID string) *modelEntry {
	var victim *modelEntry
	for _, id := range r.order {
		e := r.entries[id]
		if e.model == nil || e.pinned || e.id == excludeID {
			continue
		}
		if victim == nil || e.lastUsed.Before(victim.lastUsed) {
			victim = e
		}
	}
	return victim
}

// evict drops the entry's resident model and Closes it to reclaim GPU memory.
// The spec stays registered, so a later resolve reloads it from disk.
func (r *multiModelResolver) evict(e *modelEntry, reason string) {
	m := e.model
	e.model = nil
	if m != nil {
		m.Close()
	}
	printServe(r.log, "serve: evicted model %s (%s), ~%d bytes freed", e.id, reason, e.estBytes)
}

// sweepIdle evicts every unpinned resident idle at least IdleTTL as of now, and
// returns how many were evicted. It is called on a background tick and directly
// from tests with a controlled clock. Zero IdleTTL disables idle eviction.
func (r *multiModelResolver) sweepIdle(now time.Time) int {
	r.mu.Lock()
	defer r.mu.Unlock()
	ttl := r.opts.IdleTTL
	if ttl <= 0 {
		return 0
	}
	evicted := 0
	for _, id := range r.order {
		e := r.entries[id]
		if e.model == nil || e.pinned {
			continue
		}
		if now.Sub(e.lastUsed) >= ttl {
			r.evict(e, "idle-ttl")
			evicted++
		}
	}
	return evicted
}

// startSweeper runs the idle-TTL sweep on a ticker until ctx is cancelled. It is
// a no-op when idle eviction is off. The goroutine stops on ctx.Done — it can
// never outlive the serve it belongs to.
func (r *multiModelResolver) startSweeper(ctx context.Context, interval time.Duration) {
	if r.opts.IdleTTL <= 0 || interval <= 0 {
		return
	}
	go func() {
		ticker := time.NewTicker(interval)
		defer ticker.Stop()
		for {
			select {
			case <-ctx.Done():
				return
			case <-ticker.C:
				r.sweepIdle(r.now())
			}
		}
	}()
}

// --- Runtime control-plane operations (the admin load/unload/pin/list surface) ---

// loadSpec registers spec (if new) and loads it now, returning the entry id. An
// already-registered id reloads through ensureResident. It is the mechanism
// behind both /v1/admin/models/load and the legacy /v1/admin/serve/reload.
func (r *multiModelResolver) loadSpec(spec ModelSpec) (string, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	id := core.Trim(spec.ID)
	if id == "" {
		id = core.PathBase(core.Trim(spec.Path))
	}
	if _, ok := r.entries[id]; !ok {
		if err := r.register(spec); err != nil {
			return "", err
		}
	}
	entry := r.entries[id]
	if _, err := r.ensureResident(entry); err != nil {
		return "", err
	}
	entry.lastUsed = r.now()
	return id, nil
}

// unloadModel force-evicts a model by id, freeing its memory. Unload is an
// explicit operator action, so it overrides the pin exemption; the spec stays
// registered and reloads on the next resolve.
func (r *multiModelResolver) unloadModel(id string) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	entry, ok := r.entries[r.canonical(id)]
	if !ok {
		return core.E("serving.multiModelResolver", core.Sprintf("unknown model %q", id), nil)
	}
	if entry.model == nil {
		return nil // already not resident
	}
	r.evict(entry, "unload")
	return nil
}

// setPinned toggles a model's eviction exemption.
func (r *multiModelResolver) setPinned(id string, pinned bool) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	entry, ok := r.entries[r.canonical(id)]
	if !ok {
		return core.E("serving.multiModelResolver", core.Sprintf("unknown model %q", id), nil)
	}
	entry.pinned = pinned
	return nil
}

// canonical maps an id or alias to its canonical id, or returns the input
// lowered when unknown (so callers get a clean "unknown model" from the lookup).
func (r *multiModelResolver) canonical(id string) string {
	if cid, ok := r.aliases[core.Lower(core.Trim(id))]; ok {
		return cid
	}
	return core.Trim(id)
}

// ModelStatus is the per-model snapshot for /v1/admin/models and /v1/models.
type ModelStatus struct {
	ID           string   `json:"id"`
	Path         string   `json:"path"`
	Resident     bool     `json:"resident"`
	Pinned       bool     `json:"pinned"`
	EstBytes     uint64   `json:"est_bytes"`
	Profiles     []string `json:"profiles,omitempty"`
	LastUsedUnix int64    `json:"last_used_unix,omitempty"`
}

// list returns a per-model status snapshot in registration order.
func (r *multiModelResolver) list() []ModelStatus {
	r.mu.Lock()
	defer r.mu.Unlock()
	out := make([]ModelStatus, 0, len(r.order))
	for _, id := range r.order {
		e := r.entries[id]
		status := ModelStatus{
			ID:       e.id,
			Path:     e.path,
			Resident: e.model != nil,
			Pinned:   e.pinned,
			EstBytes: e.estBytes,
			Profiles: sortedProfileNames(e.profiles),
		}
		if !e.lastUsed.IsZero() {
			status.LastUsedUnix = e.lastUsed.Unix()
		}
		out = append(out, status)
	}
	return out
}

// listedModelIDs returns every servable id — each model plus its `id:profile`
// combinations — in registration order, for the /v1/models list callback.
func (r *multiModelResolver) listedModelIDs() []string {
	r.mu.Lock()
	defer r.mu.Unlock()
	out := []string{}
	for _, id := range r.order {
		e := r.entries[id]
		out = append(out, e.id)
		for _, name := range sortedProfileNames(e.profiles) {
			out = append(out, e.id+":"+name)
		}
	}
	return out
}

// residentModelIDs returns the ids of the currently-loaded models, for the
// /v1/health probe.
func (r *multiModelResolver) residentModelIDs() []string {
	r.mu.Lock()
	defer r.mu.Unlock()
	out := []string{}
	for _, id := range r.order {
		if r.entries[id].model != nil {
			out = append(out, id)
		}
	}
	return out
}

// CurrentPath returns the default model's path — the admin serve/status seam.
func (r *multiModelResolver) CurrentPath() string {
	r.mu.Lock()
	defer r.mu.Unlock()
	if e, ok := r.entries[r.defaultID]; ok {
		return e.path
	}
	return ""
}

// ReloadModel loads newPath as a pinned model and makes it the default — the
// multi-model mapping of the legacy /v1/admin/serve/reload verb. It returns the
// previous default's path and the new one.
func (r *multiModelResolver) ReloadModel(newPath string, newOpts []inference.LoadOption) (prevPath, newActive string, err error) {
	r.mu.Lock()
	prev := ""
	if e, ok := r.entries[r.defaultID]; ok {
		prev = e.path
	}
	r.mu.Unlock()

	id, err := r.loadSpec(ModelSpec{Path: newPath, Pinned: true, LoadOptions: newOpts})
	if err != nil {
		return "", "", err
	}
	r.mu.Lock()
	r.defaultID = id
	newActive = r.entries[id].path
	r.mu.Unlock()
	return prev, newActive, nil
}

// openaiResolver returns r as an openai.Resolver for wire-up sites that keep the
// interface narrow.
func (r *multiModelResolver) openaiResolver() Resolver {
	return openai.ResolverFunc(r.ResolveModel)
}

// sortedProfileNames returns a profile map's keys sorted, for stable listing.
func sortedProfileNames(profiles map[string][]inference.GenerateOption) []string {
	if len(profiles) == 0 {
		return nil
	}
	names := make([]string, 0, len(profiles))
	for name := range profiles {
		names = append(names, name)
	}
	slices.Sort(names)
	return names
}

// estimateModelBytes sums the regular-file sizes under a model directory — the
// on-disk pack size, the residency-budget proxy for a model's resident
// footprint (weights dominate; go-inference exposes no engine-neutral live GPU
// read). An unreadable path yields 0: the budget cannot gate what it cannot
// size, so the load proceeds rather than being falsely refused.
func estimateModelBytes(path string) uint64 {
	if core.Trim(path) == "" {
		return 0
	}
	var total uint64
	_ = core.WalkDir(core.DirFS(path), ".", func(_ string, d core.FsDirEntry, err error) error {
		if err != nil || d == nil || d.IsDir() {
			return nil
		}
		info, ierr := d.Info()
		if ierr != nil || info == nil {
			return nil
		}
		if sz := info.Size(); sz > 0 {
			total += uint64(sz)
		}
		return nil
	})
	return total
}
