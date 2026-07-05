// SPDX-Licence-Identifier: EUPL-1.2

// Package obs is the observability run-tree and feedback model for the
// inference stack (RFC.inference-stack §3.7). Every inference — local or remote
// — emits a run; tool calls and fusion-panel members (the inference stack §6.9) are child
// runs forming a tree. A run carries its inputs, outputs, model, token usage,
// status and timing; feedback (a score or label) attaches to a run by id from
// the LEK scorer, an evaluator, or a human (RFC.inference-stack §3.7).
//
// This is the pure-Go model. Runs and feedback are emitted to a Sink; the
// durable landing — go-store rows, go-log OTEL export, InfluxDB time-series,
// OpenBrain recall (RFC.inference-stack §3.7) — is a concrete Sink the host
// supplies. MemorySink here is the test/in-process implementation. The run-tree
// is the EU AI Act audit trail (RFC.inference-stack §3.8): inputs, model,
// provenance and decisions, recorded per policy.
//
//	tree := obs.NewRunTree(obs.MintIDs(), time.Now)
//	tree.Emit(sink)
//	root := tree.StartRun("chat", map[string]any{"prompt": prompt})
//	span := tree.Child(root, "tool:search", map[string]any{"q": q})
//	tree.Finish(span, map[string]any{"hits": hits}, usage)
//	tree.Finish(root, map[string]any{"reply": reply}, usage)
//	tree.Record(obs.Feedback{RunID: root.ID, Key: "quality", Score: 0.8, Source: "human"})
package obs

import (
	"sync"
	"time"

	core "dappco.re/go"
)

// Status is a run's lifecycle state (RFC.inference-stack §3.7 — a run carries a
// status).
type Status string

const (
	// StatusRunning is a run that has started and not yet finished or failed.
	StatusRunning Status = "running"
	// StatusCompleted is a run that finished successfully (Finish was called).
	StatusCompleted Status = "completed"
	// StatusFailed is a run that errored (Fail was called).
	StatusFailed Status = "failed"
)

// Run is one node in the run-tree (RFC.inference-stack §3.7). A request is a
// root run; tool calls and fusion-panel members are children, linked by
// ParentID. The run records its inputs, outputs, the model / endpoint that
// served it, token usage (any — the inference stack §6.6 usage shape), status, and
// timing; Err holds the failure message when Status is failed.
type Run struct {
	ID        string         `json:"id"`
	ParentID  string         `json:"parent_id,omitempty"`
	Name      string         `json:"name"`
	Inputs    map[string]any `json:"inputs"`
	Outputs   map[string]any `json:"outputs"`
	Model     string         `json:"model,omitempty"`
	Usage     any            `json:"usage,omitempty"`
	Status    Status         `json:"status"`
	StartedAt time.Time      `json:"started_at"`
	EndedAt   time.Time      `json:"ended_at"`
	Err       string         `json:"err,omitempty"`
}

// Feedback is a score or label attached to a run by id (RFC.inference-stack
// §3.7). Source records who produced it — "human" (annotation queue),
// "evaluator" (go-ml), or "heuristic" (the LEK scorer, go-mlx pkg/score).
type Feedback struct {
	RunID   string  `json:"run_id"`
	Key     string  `json:"key"`
	Score   float64 `json:"score"`
	Comment string  `json:"comment,omitempty"`
	Source  string  `json:"source,omitempty"`
}

// Sink is where runs and feedback land (RFC.inference-stack §3.7 — "emit &
// land"). The durable implementation writes to go-store / go-log / InfluxDB;
// MemorySink is the in-process one. Implementations must be safe for concurrent
// use — RunTree may emit from many goroutines.
type Sink interface {
	// Run records a run (on Finish or Fail).
	Run(Run)
	// Feedback records a feedback entry (on Record).
	Feedback(Feedback)
}

// MemorySink is a goroutine-safe in-memory Sink that keeps every run and
// feedback entry it is given. Used in tests and for in-process inspection.
//
//	sink := obs.NewMemorySink()
//	tree.Emit(sink)
//	... ; runs := sink.Runs()
type MemorySink struct {
	mu       sync.Mutex
	runs     []Run
	feedback []Feedback
}

// NewMemorySink returns an empty MemorySink ready to receive runs and feedback.
func NewMemorySink() *MemorySink { return &MemorySink{} }

// Run records a run.
func (m *MemorySink) Run(r Run) {
	m.mu.Lock()
	m.runs = append(m.runs, r)
	m.mu.Unlock()
}

// Feedback records a feedback entry.
func (m *MemorySink) Feedback(f Feedback) {
	m.mu.Lock()
	m.feedback = append(m.feedback, f)
	m.mu.Unlock()
}

// Runs returns a copy of the recorded runs in emission order.
func (m *MemorySink) Runs() []Run {
	m.mu.Lock()
	defer m.mu.Unlock()
	out := make([]Run, len(m.runs))
	copy(out, m.runs)
	return out
}

// FeedbackEntries returns a copy of the recorded feedback in record order.
// (The Sink method Feedback(Feedback) is the writer; this is the reader — Go
// won't let one type spell both with the same name.)
func (m *MemorySink) FeedbackEntries() []Feedback {
	m.mu.Lock()
	defer m.mu.Unlock()
	out := make([]Feedback, len(m.feedback))
	copy(out, m.feedback)
	return out
}

// IDGen mints run ids. Injectable so tests get deterministic ids (UUIDs in
// production, a sequence in tests).
type IDGen func() string

// Clock returns the current time. Injectable so tests get a fixed clock.
type Clock func() time.Time

// MintIDs is the production IDGen — a unique run id per call via core.ID
// (e.g. "id-1-a3f2b1").
//
//	tree := obs.NewRunTree(obs.MintIDs(), time.Now)
func MintIDs() IDGen { return func() string { return core.ID() } }

// RunTree builds and tracks a run-tree (RFC.inference-stack §3.7). It mints ids
// and timestamps from injected generators, maintains the parent→children
// index, records feedback by run id, and emits runs / feedback to the Sink set
// by Emit. Safe for concurrent use.
type RunTree struct {
	mu       sync.Mutex
	id       IDGen
	clock    Clock
	sink     Sink
	children map[string][]*Run
	feedback map[string][]Feedback
}

// NewRunTree constructs a RunTree over an id generator and a clock. With no
// Emit, runs are tracked in-memory only.
//
//	tree := obs.NewRunTree(obs.MintIDs(), time.Now)
func NewRunTree(id IDGen, clock Clock) *RunTree {
	return &RunTree{
		id:       id,
		clock:    clock,
		children: map[string][]*Run{},
		feedback: map[string][]Feedback{},
	}
}

// Emit sets the Sink that receives runs (on Finish / Fail) and feedback (on
// Record). Call before starting runs.
//
//	tree.Emit(obs.NewMemorySink())
func (t *RunTree) Emit(sink Sink) {
	t.mu.Lock()
	t.sink = sink
	t.mu.Unlock()
}

// StartRun opens a root run — a request (RFC.inference-stack §3.7). The run is
// minted with a fresh id, the running status, and a start time; nil inputs
// become an empty map so callers never read a nil.
//
//	root := tree.StartRun("chat", map[string]any{"prompt": prompt})
func (t *RunTree) StartRun(name string, inputs map[string]any) *Run {
	return t.start("", name, inputs)
}

// Child opens a sub-run under parent — a tool call or fusion-panel member
// (RFC.inference-stack §3.7). A nil parent promotes the run to a root (no
// parent id), so a detached span never panics.
//
//	span := tree.Child(root, "tool:search", map[string]any{"q": q})
func (t *RunTree) Child(parent *Run, name string, inputs map[string]any) *Run {
	parentID := ""
	if parent != nil {
		parentID = parent.ID
	}
	return t.start(parentID, name, inputs)
}

// start mints a run, indexes it under its parent, and returns it.
func (t *RunTree) start(parentID, name string, inputs map[string]any) *Run {
	if inputs == nil {
		inputs = map[string]any{}
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	run := &Run{
		ID:        t.id(),
		ParentID:  parentID,
		Name:      name,
		Inputs:    inputs,
		Outputs:   map[string]any{},
		Status:    StatusRunning,
		StartedAt: t.clock(),
	}
	if parentID != "" {
		t.children[parentID] = append(t.children[parentID], run)
	}
	return run
}

// Finish closes a run successfully: it records outputs and usage, marks the run
// completed, stamps the end time, and emits the run to the Sink. nil outputs
// become an empty map. Finishing a nil run is a no-op.
//
//	tree.Finish(root, map[string]any{"reply": reply}, usage)
func (t *RunTree) Finish(run *Run, outputs map[string]any, usage any) {
	if run == nil {
		return
	}
	if outputs == nil {
		outputs = map[string]any{}
	}
	t.mu.Lock()
	run.Outputs = outputs
	run.Usage = usage
	run.Status = StatusCompleted
	run.EndedAt = t.clock()
	sink := t.sink
	snapshot := *run
	t.mu.Unlock()
	if sink != nil {
		sink.Run(snapshot)
	}
}

// Fail closes a run as failed: it marks the run failed, captures the error
// message (RFC.inference-stack §3.7 — status), stamps the end time, and emits
// the run. A nil error leaves an empty message; failing a nil run is a no-op.
//
//	tree.Fail(root, core.E("obs", "model unavailable", cause))
func (t *RunTree) Fail(run *Run, err error) {
	if run == nil {
		return
	}
	msg := ""
	if err != nil {
		msg = err.Error()
	}
	t.mu.Lock()
	run.Status = StatusFailed
	run.Err = msg
	run.EndedAt = t.clock()
	sink := t.sink
	snapshot := *run
	t.mu.Unlock()
	if sink != nil {
		sink.Run(snapshot)
	}
}

// Children returns a copy of the sub-runs recorded under a run id, in start
// order. An unknown id yields an empty slice.
//
//	for _, c := range tree.Children(root.ID) { ... }
func (t *RunTree) Children(runID string) []*Run {
	t.mu.Lock()
	defer t.mu.Unlock()
	kids := t.children[runID]
	out := make([]*Run, len(kids))
	copy(out, kids)
	return out
}

// Record attaches feedback to a run by id (RFC.inference-stack §3.7). It is
// stored for aggregation and emitted to the Sink. Feedback for an unknown run
// id is kept too — aggregation is by id, so it simply never rolls up under a
// different run.
//
//	tree.Record(obs.Feedback{RunID: root.ID, Key: "quality", Score: 0.8, Source: "human"})
func (t *RunTree) Record(f Feedback) {
	t.mu.Lock()
	t.feedback[f.RunID] = append(t.feedback[f.RunID], f)
	sink := t.sink
	t.mu.Unlock()
	if sink != nil {
		sink.Feedback(f)
	}
}

// MeanByKey returns the mean feedback score per key for a run id
// (RFC.inference-stack §3.7 — rolled-up insights). A run with no feedback
// yields an empty (non-nil) map.
//
//	means := tree.MeanByKey(root.ID) // map[key]meanScore
func (t *RunTree) MeanByKey(runID string) map[string]float64 {
	t.mu.Lock()
	entries := t.feedback[runID]
	snapshot := make([]Feedback, len(entries))
	copy(snapshot, entries)
	t.mu.Unlock()

	sum := map[string]float64{}
	count := map[string]int{}
	for _, f := range snapshot {
		sum[f.Key] += f.Score
		count[f.Key]++
	}
	out := map[string]float64{}
	for key, total := range sum {
		out[key] = total / float64(count[key])
	}
	return out
}
