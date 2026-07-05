// SPDX-Licence-Identifier: EUPL-1.2

// Package probe is the go-mlx event-vocabulary for first-class
// observability of inference and training. Backends emit typed Events
// through a Sink; Bus fans events out to multiple sinks, Recorder stores
// them in memory for tests and reproducible probes.
//
//	recorder := probe.NewRecorder()
//	bus := probe.NewBus(recorder, callerSink)
//	bus.EmitProbe(probe.Event{Kind: probe.KindToken, Token: &probe.Token{ID: 7}})
//	events := recorder.Events()
package probe

import (
	"sync"
	"sync/atomic"

	core "dappco.re/go"
)

// Kind names the typed payload carried by a probe event.
type Kind string

// Phase identifies where the event was emitted in the runtime.
type Phase string

const (
	KindToken           Kind = "token"
	KindLogits          Kind = "logits"
	KindEntropy         Kind = "entropy"
	KindSelectedHeads   Kind = "selected_heads"
	KindLayerCoherence  Kind = "layer_coherence"
	KindRouterDecision  Kind = "router_decision"
	KindExpertResidency Kind = "expert_residency"
	KindResidual        Kind = "residual_summary"
	KindCachePressure   Kind = "cache_pressure"
	KindMemoryPressure  Kind = "memory_pressure"
	KindTraining        Kind = "training"
	KindScore           Kind = "score"

	PhasePrefill  Phase = "prefill"
	PhaseDecode   Phase = "decode"
	PhaseTraining Phase = "training"
)

// Event is the first-class event envelope for inference and training probes.
type Event struct {
	Kind            Kind              `json:"kind"`
	Phase           Phase             `json:"phase,omitempty"`
	Step            int               `json:"step"`
	Token           *Token            `json:"token,omitempty"`
	Logits          *Logits           `json:"logits,omitempty"`
	Entropy         *Entropy          `json:"entropy,omitempty"`
	SelectedHeads   *HeadSelection    `json:"selected_heads,omitempty"`
	LayerCoherence  *LayerCoherence   `json:"layer_coherence,omitempty"`
	RouterDecision  *RouterDecision   `json:"router_decision,omitempty"`
	ExpertResidency *ExpertResidency  `json:"expert_residency,omitempty"`
	Residual        *ResidualSummary  `json:"residual,omitempty"`
	Cache           *CachePressure    `json:"cache,omitempty"`
	Memory          *MemoryPressure   `json:"memory,omitempty"`
	Training        *Training         `json:"training,omitempty"`
	Score           *Score            `json:"score,omitempty"`
	Meta            map[string]string `json:"meta,omitempty"`
}

// Token records a selected token and local decode position.
type Token struct {
	ID              int32  `json:"id"`
	Text            string `json:"text,omitempty"`
	PromptTokens    int    `json:"prompt_tokens,omitempty"`
	GeneratedTokens int    `json:"generated_tokens,omitempty"`
}

// Logit records one high-scoring token from a logit vector.
type Logit struct {
	TokenID     int32   `json:"token_id"`
	Logit       float32 `json:"logit"`
	Probability float64 `json:"probability,omitempty"`
}

// Logits records a compact summary of a logit vector.
type Logits struct {
	Shape      []int32           `json:"shape,omitempty"`
	VocabSize  int               `json:"vocab_size,omitempty"`
	MaxTokenID int32             `json:"max_token_id"`
	MaxLogit   float32           `json:"max_logit"`
	MinTokenID int32             `json:"min_token_id"`
	MinLogit   float32           `json:"min_logit"`
	MeanLogit  float64           `json:"mean_logit"`
	Top        []Logit           `json:"top,omitempty"`
	Values     []float32         `json:"values,omitempty"`
	Meta       map[string]string `json:"meta,omitempty"`
}

// Entropy records the Shannon entropy of a probability distribution.
type Entropy struct {
	Value float64 `json:"value"`
	Unit  string  `json:"unit,omitempty"`
}

// HeadSelection records attention heads selected for a probe or analysis pass.
type HeadSelection struct {
	Layer  int       `json:"layer,omitempty"`
	Heads  []int     `json:"heads,omitempty"`
	Scores []float64 `json:"scores,omitempty"`
}

// LayerCoherence records per-layer K/V and residual posture metrics.
type LayerCoherence struct {
	Layer          int     `json:"layer,omitempty"`
	KeyCoherence   float64 `json:"key_coherence,omitempty"`
	ValueCoherence float64 `json:"value_coherence,omitempty"`
	CrossAlignment float64 `json:"cross_alignment,omitempty"`
	KVCoupling     float64 `json:"kv_coupling,omitempty"`
	HeadEntropy    float64 `json:"head_entropy,omitempty"`
	PhaseLock      float64 `json:"phase_lock,omitempty"`
}

// RouterDecision records MoE or routing decisions when the architecture exposes them.
type RouterDecision struct {
	Layer       int       `json:"layer,omitempty"`
	TokenID     int32     `json:"token_id,omitempty"`
	ExpertIDs   []int     `json:"expert_ids,omitempty"`
	Weights     []float32 `json:"weights,omitempty"`
	Temperature float32   `json:"temperature,omitempty"`
}

// ExpertResidencyAction names probe-visible expert residency transitions.
type ExpertResidencyAction string

const (
	ExpertResidencyActionStartup ExpertResidencyAction = "startup"
	ExpertResidencyActionPageIn  ExpertResidencyAction = "page_in"
	ExpertResidencyActionEvict   ExpertResidencyAction = "evict"
	ExpertResidencyActionHit     ExpertResidencyAction = "hit"
)

// ExpertResidency records MoE expert paging and residency transitions.
type ExpertResidency struct {
	Action             ExpertResidencyAction `json:"action"`
	Layer              int                   `json:"layer,omitempty"`
	ExpertIDs          []int                 `json:"expert_ids,omitempty"`
	ResidentExperts    int                   `json:"resident_experts,omitempty"`
	MaxResidentExperts int                   `json:"max_resident_experts,omitempty"`
	LoadedBytes        uint64                `json:"loaded_bytes,omitempty"`
	EvictedBytes       uint64                `json:"evicted_bytes,omitempty"`
	Duration           int64                 `json:"duration,omitempty"`
}

// ResidualSummary records compact residual-stream statistics.
type ResidualSummary struct {
	Layer    int     `json:"layer,omitempty"`
	Mean     float64 `json:"mean,omitempty"`
	Variance float64 `json:"variance,omitempty"`
	RMS      float64 `json:"rms,omitempty"`
	L2Norm   float64 `json:"l2_norm,omitempty"`
	MaxAbs   float64 `json:"max_abs,omitempty"`
}

// CachePressure records KV cache posture for local memory-aware runs.
type CachePressure struct {
	PromptTokens    int     `json:"prompt_tokens,omitempty"`
	GeneratedTokens int     `json:"generated_tokens,omitempty"`
	LayerCount      int     `json:"layer_count,omitempty"`
	CacheTokens     int     `json:"cache_tokens,omitempty"`
	ProcessedTokens int     `json:"processed_tokens,omitempty"`
	MaxCacheTokens  int     `json:"max_cache_tokens,omitempty"`
	Utilization     float64 `json:"utilization,omitempty"`
	Rotating        bool    `json:"rotating,omitempty"`
}

// MemoryPressure records MLX allocator pressure.
type MemoryPressure struct {
	ActiveBytes uint64 `json:"active_bytes,omitempty"`
	PeakBytes   uint64 `json:"peak_bytes,omitempty"`
	CacheBytes  uint64 `json:"cache_bytes,omitempty"`
}

// Training loss-curve lanes. LossTypeTrain is the optimizer-step loss;
// LossTypeVal is the no-grad validation forward — the two curves whose
// amplitude oscillation is the cascade read on a training run. The names
// match the v0 LEM instrument's loss_type tag verbatim so downstream
// dashboards work unchanged.
const (
	LossTypeTrain = "train"
	LossTypeVal   = "val"
)

// Training records training-loop scalars.
type Training struct {
	Step         int     `json:"step,omitempty"`
	Epoch        int     `json:"epoch,omitempty"`
	Loss         float64 `json:"loss,omitempty"`
	LearningRate float64 `json:"learning_rate,omitempty"`
	GradNorm     float64 `json:"grad_norm,omitempty"`
	// LossType separates the curves: LossTypeTrain (default when empty)
	// or LossTypeVal. Tokens is the token count consumed by the step,
	// letting a sink derive tokens_per_sec without clocking the loop.
	LossType string `json:"loss_type,omitempty"`
	Tokens   int    `json:"tokens,omitempty"`
}

// Score records semantic-quality readings at a training step — the
// lem-scorer's per-pass aggregates riding the same iteration clock as the
// loss curves, so quality patterns and loss-amplitude patterns are
// inspectable side by side. Values keys are dimension names (lek,
// composite, hostility, echo, …).
type Score struct {
	Label  string             `json:"label,omitempty"`
	Values map[string]float64 `json:"values,omitempty"`
}

// Sink consumes typed probe events.
type Sink interface {
	EmitProbe(Event)
}

// ownedEventSink is implemented by sinks that accept an unshared
// event without the Bus pre-cloning it. By implementing this
// interface, the sink declares that the Bus may deliver the event
// directly (no fanout-side CloneEvent) and that the sink may defer
// any defensive cloning to read time. Implementing this interface
// lets the Bus skip its own defensive CloneEvent when fanning out
// to that sink and the sink itself can skip the on-emit clone if
// it has a read-side deep-clone (e.g., Recorder.Events()).
//
// In exchange, the bus caller must not mutate the event (or any
// payload pointer the event aliases) after the Bus.EmitProbe call
// returns — the Bus's existing contract for owned sinks is that
// the caller has transferred ownership, and the on-emit clone
// elision rests on that promise.
//
// Sinks that don't implement this interface still receive the
// standard pre-cloned Event so the public Sink contract is
// unchanged.
type ownedEventSink interface {
	emitProbeOwned(Event)
}

// SinkFunc adapts a function into a Sink.
type SinkFunc func(Event)

// EmitProbe emits an event to the wrapped function.
//
//	probe.SinkFunc(func(e probe.Event) { … }).EmitProbe(event)
func (f SinkFunc) EmitProbe(event Event) {
	if f != nil {
		f(event)
	}
}

// Bus fans probe events out to one or more sinks.
//
// The sinks slice is published through an atomic.Pointer so EmitProbe
// reads the snapshot lock-free — the prior RWMutex paid for every
// emit, even on empty buses, dominating the no-sink hot loop. Add
// installs a fresh slice under a writer mutex so a concurrent Add
// remains race-free; readers always observe a complete snapshot.
type Bus struct {
	addMu sync.Mutex
	sinks atomic.Pointer[[]Sink]
}

// NewBus creates a fanout sink.
//
//	bus := probe.NewBus(sink1, sink2)
func NewBus(sinks ...Sink) *Bus {
	bus := &Bus{}
	if len(sinks) == 0 {
		return bus
	}
	// Build the initial sink slice directly — Add takes the mutex
	// per call, so building N sinks via Add was N lock/unlock pairs
	// before any caller could observe the bus. The constructor owns
	// the only reference so the slice growth is safe lock-free.
	initial := make([]Sink, 0, len(sinks))
	for _, sink := range sinks {
		if sink != nil {
			initial = append(initial, sink)
		}
	}
	bus.sinks.Store(&initial)
	return bus
}

// Add appends a sink to the bus. Nil receivers and nil sinks are ignored.
//
//	bus.Add(sink)
func (b *Bus) Add(sink Sink) {
	if b == nil || sink == nil {
		return
	}
	// Publish-once semantics: build the new slice, then atomic-store
	// the pointer so EmitProbe readers see the existing slice through
	// the previous pointer until the swap commits. The addMu only
	// serialises concurrent Add callers so they don't lose each
	// other's appends. Manual Unlock (no defer) keeps the path
	// branch-light — there's no panic surface inside the critical
	// section.
	b.addMu.Lock()
	var current []Sink
	if cur := b.sinks.Load(); cur != nil {
		current = *cur
	}
	next := make([]Sink, len(current)+1)
	copy(next, current)
	next[len(current)] = sink
	b.sinks.Store(&next)
	b.addMu.Unlock()
}

// EmitProbe emits an event to every sink.
//
//	bus.EmitProbe(event)
func (b *Bus) EmitProbe(event Event) {
	if b == nil {
		return
	}
	// Atomic snapshot — concurrent Add publishes through Store, so
	// the slice header we read is stable for the duration of the
	// fanout (the backing array is never mutated in place; Add
	// installs a fresh slice).
	snap := b.sinks.Load()
	if snap == nil {
		return
	}
	sinks := *snap
	// Fast-path for the common one-sink bus — keeps the OneSink
	// path branch-light and avoids the range-loop overhead the
	// multi-sink path pays.
	if len(sinks) == 1 {
		sink := sinks[0]
		if sink == nil {
			return
		}
		if owned, ok := sink.(ownedEventSink); ok {
			owned.emitProbeOwned(event)
			return
		}
		sink.EmitProbe(CloneEvent(event))
		return
	}
	for _, sink := range sinks {
		if sink == nil {
			continue
		}
		if owned, ok := sink.(ownedEventSink); ok {
			owned.emitProbeOwned(event)
			continue
		}
		sink.EmitProbe(CloneEvent(event))
	}
}

// Recorder stores probe events in memory for tests, reproducible probes,
// or artifacts.
type Recorder struct {
	mu     sync.Mutex
	events []Event
}

// NewRecorder returns a recorder sink.
//
//	r := probe.NewRecorder()
func NewRecorder() *Recorder {
	return &Recorder{}
}

// EmitProbe records an event.
//
//	r.EmitProbe(event)
func (r *Recorder) EmitProbe(event Event) {
	if r == nil {
		return
	}
	// CloneEvent (the deep copy) runs outside the lock — only the
	// slice append needs serialising. Multiple bus-driven emitters
	// can now clone in parallel and only contend on the append.
	cloned := CloneEvent(event)
	r.mu.Lock()
	r.events = append(r.events, cloned)
	r.mu.Unlock()
}

// emitProbeOwned satisfies ownedEventSink. The Bus invokes this
// method when it has already verified the caller transferred event
// ownership — the bus-side fanout no longer clones, and the
// recorder can store the value by value without a second defensive
// clone because Events() always returns a fresh deep-clone snapshot
// on read. Direct callers must use EmitProbe (which still defends
// against post-emit caller mutation); only the Bus's owned-sink
// fast-path may bypass the on-emit clone.
//
// emitProbeOwned must be called only from the same package as
// ownedEventSink; the unexported interface guarantees that
// external callers cannot satisfy it and therefore cannot invoke
// this method directly.
func (r *Recorder) emitProbeOwned(event Event) {
	if r == nil {
		return
	}
	r.mu.Lock()
	r.events = append(r.events, event)
	r.mu.Unlock()
}

// Events returns recorded events without aliasing recorder storage.
//
//	events := r.Events()
func (r *Recorder) Events() []Event {
	if r == nil {
		return nil
	}
	r.mu.Lock()
	// Snapshot the slice header — append-only growth means the
	// existing backing array is stable for snapshot[i] reads until
	// the recorder is garbage-collected, so the deep clone can
	// happen outside the lock. Holding the mutex through 128
	// CloneEvent calls otherwise serialised every concurrent
	// EmitProbe against the read.
	snapshot := r.events
	r.mu.Unlock()
	if len(snapshot) == 0 {
		return nil
	}
	out := make([]Event, len(snapshot))
	// Batch-allocate scratches for every event in a single slice — each
	// snapshot[i] gets its own scratch slot to back its payload pointers,
	// so the cloned events still don't alias each other. The previous
	// shape allocated one heap-bound pointer per non-nil payload (Token,
	// Logits, Entropy, ...) per event; with 128 events × ~5-11 pointer
	// allocs that compounded to >700 allocs from payload pointers alone.
	// One slice make absorbs them all.
	scratches := make([]cloneScratch, len(snapshot))
	for i := range snapshot {
		out[i] = cloneEventInto(snapshot[i], &scratches[i])
	}
	return out
}

// CloneEvent returns a deep copy of an Event so emitters can safely
// share immutable references downstream.
//
//	out := probe.CloneEvent(event)
//
// Each non-nil payload is cloned through its own pointer allocation so
// the per-payload alloc cost matches the per-payload size. Callers that
// batch many clones (Recorder.Events) should reach for cloneEventInto
// with a pre-allocated []cloneScratch — there a single slice make
// absorbs every payload-pointer allocation across the batch.
func CloneEvent(event Event) Event {
	out := event
	if event.Token != nil {
		token := *event.Token
		out.Token = &token
	}
	if event.Logits != nil {
		logits := *event.Logits
		// logits is a value copy of *event.Logits, so its slice headers
		// alias the same backing arrays; cloning through the local copy
		// avoids re-dereferencing event.Logits four times.
		logits.Shape = core.SliceClone(logits.Shape)
		logits.Top = core.SliceClone(logits.Top)
		logits.Values = core.SliceClone(logits.Values)
		logits.Meta = cloneMeta(logits.Meta)
		out.Logits = &logits
	}
	if event.Entropy != nil {
		entropy := *event.Entropy
		out.Entropy = &entropy
	}
	if event.SelectedHeads != nil {
		heads := *event.SelectedHeads
		heads.Heads = core.SliceClone(heads.Heads)
		heads.Scores = core.SliceClone(heads.Scores)
		out.SelectedHeads = &heads
	}
	if event.LayerCoherence != nil {
		coherence := *event.LayerCoherence
		out.LayerCoherence = &coherence
	}
	if event.RouterDecision != nil {
		router := *event.RouterDecision
		router.ExpertIDs = core.SliceClone(router.ExpertIDs)
		router.Weights = core.SliceClone(router.Weights)
		out.RouterDecision = &router
	}
	if event.ExpertResidency != nil {
		residency := *event.ExpertResidency
		residency.ExpertIDs = core.SliceClone(residency.ExpertIDs)
		out.ExpertResidency = &residency
	}
	if event.Residual != nil {
		residual := *event.Residual
		out.Residual = &residual
	}
	if event.Cache != nil {
		cache := *event.Cache
		out.Cache = &cache
	}
	if event.Memory != nil {
		memory := *event.Memory
		out.Memory = &memory
	}
	if event.Training != nil {
		training := *event.Training
		out.Training = &training
	}
	if event.Score != nil {
		scoreCopy := *event.Score
		scoreCopy.Values = cloneScoreValues(scoreCopy.Values)
		out.Score = &scoreCopy
	}
	out.Meta = cloneMeta(event.Meta)
	return out
}

// cloneScratch holds every payload value inline so a single heap
// allocation backs every payload pointer of a cloned Event. Used by
// Recorder.Events to amortise per-event payload-pointer allocations
// across a batch — one slice make backs N events' worth of payload
// storage instead of paying ~5-11 individual pointer allocs per event.
type cloneScratch struct {
	token           Token
	logits          Logits
	entropy         Entropy
	selectedHeads   HeadSelection
	layerCoherence  LayerCoherence
	routerDecision  RouterDecision
	expertResidency ExpertResidency
	residual        ResidualSummary
	cache           CachePressure
	memory          MemoryPressure
	training        Training
	score           Score
}

// cloneEventInto deep-copies event into out, using scratch to back the
// payload pointers. The caller owns scratch — typically one slot of a
// pre-allocated []cloneScratch — so the returned Event's payload
// pointers all alias storage inside scratch. Mutating out's payloads
// only affects scratch (which the caller controls), never the source.
func cloneEventInto(event Event, scratch *cloneScratch) Event {
	out := event
	if event.Token != nil {
		scratch.token = *event.Token
		out.Token = &scratch.token
	}
	if event.Logits != nil {
		scratch.logits = *event.Logits
		scratch.logits.Shape = core.SliceClone(scratch.logits.Shape)
		scratch.logits.Top = core.SliceClone(scratch.logits.Top)
		scratch.logits.Values = core.SliceClone(scratch.logits.Values)
		scratch.logits.Meta = cloneMeta(scratch.logits.Meta)
		out.Logits = &scratch.logits
	}
	if event.Entropy != nil {
		scratch.entropy = *event.Entropy
		out.Entropy = &scratch.entropy
	}
	if event.SelectedHeads != nil {
		scratch.selectedHeads = *event.SelectedHeads
		scratch.selectedHeads.Heads = core.SliceClone(scratch.selectedHeads.Heads)
		scratch.selectedHeads.Scores = core.SliceClone(scratch.selectedHeads.Scores)
		out.SelectedHeads = &scratch.selectedHeads
	}
	if event.LayerCoherence != nil {
		scratch.layerCoherence = *event.LayerCoherence
		out.LayerCoherence = &scratch.layerCoherence
	}
	if event.RouterDecision != nil {
		scratch.routerDecision = *event.RouterDecision
		scratch.routerDecision.ExpertIDs = core.SliceClone(scratch.routerDecision.ExpertIDs)
		scratch.routerDecision.Weights = core.SliceClone(scratch.routerDecision.Weights)
		out.RouterDecision = &scratch.routerDecision
	}
	if event.ExpertResidency != nil {
		scratch.expertResidency = *event.ExpertResidency
		scratch.expertResidency.ExpertIDs = core.SliceClone(scratch.expertResidency.ExpertIDs)
		out.ExpertResidency = &scratch.expertResidency
	}
	if event.Residual != nil {
		scratch.residual = *event.Residual
		out.Residual = &scratch.residual
	}
	if event.Cache != nil {
		scratch.cache = *event.Cache
		out.Cache = &scratch.cache
	}
	if event.Memory != nil {
		scratch.memory = *event.Memory
		out.Memory = &scratch.memory
	}
	if event.Training != nil {
		scratch.training = *event.Training
		out.Training = &scratch.training
	}
	if event.Score != nil {
		scratch.score = *event.Score
		scratch.score.Values = cloneScoreValues(scratch.score.Values)
		out.Score = &scratch.score
	}
	out.Meta = cloneMeta(event.Meta)
	return out
}

func cloneMeta(meta map[string]string) map[string]string {
	if len(meta) == 0 {
		return nil
	}
	return core.MapClone(meta)
}

func cloneScoreValues(values map[string]float64) map[string]float64 {
	if len(values) == 0 {
		return nil
	}
	return core.MapClone(values)
}
