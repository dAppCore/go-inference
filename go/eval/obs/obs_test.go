// SPDX-Licence-Identifier: EUPL-1.2

package obs

import (
	"sync"
	"time"

	core "dappco.re/go"
)

// fixedClock is a deterministic clock: every Now advances by one second from a
// fixed epoch, so StartedAt/EndedAt are predictable in tests.
//
//	tree := NewRunTree(seqIDs(), (&fixedClock{}).Now)
type fixedClock struct {
	mu   sync.Mutex
	tick int
}

func (c *fixedClock) Now() time.Time {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.tick++
	return time.Date(2026, 6, 14, 0, 0, c.tick, 0, time.UTC)
}

// seqIDs returns an injectable id generator minting run-1, run-2, … so tree
// shape is assertable without random ids.
func seqIDs() func() string {
	var mu sync.Mutex
	n := 0
	return func() string {
		mu.Lock()
		defer mu.Unlock()
		n++
		return "run-" + core.Itoa(n)
	}
}

func TestObs_MintIDs_Good(t *core.T) {
	// The production IDGen mints a fresh, non-empty id each call — the default
	// when a caller doesn't inject a sequence.
	gen := MintIDs()
	a := gen()
	b := gen()
	core.AssertTrue(t, a != "", "id is non-empty")
	core.AssertTrue(t, a != b, "ids are unique per call")

	// It drives a RunTree end-to-end with the real clock.
	tree := NewRunTree(MintIDs(), time.Now)
	root := tree.StartRun("chat", nil)
	core.AssertTrue(t, root.ID != "", "root has a minted id")
}

func TestObs_RunTree_Good(t *core.T) {
	// A request is a root run; a tool call is a child. Finishing the root sets
	// outputs, usage, completed status, an end time after the start, and emits
	// the run to the sink.
	sink := NewMemorySink()
	tree := NewRunTree(seqIDs(), (&fixedClock{}).Now)
	tree.Emit(sink)

	root := tree.StartRun("chat", map[string]any{"prompt": "hi"})
	core.AssertEqual(t, "run-1", root.ID)
	core.AssertEqual(t, "", root.ParentID)
	core.AssertEqual(t, StatusRunning, root.Status)
	core.AssertEqual(t, "hi", root.Inputs["prompt"])
	core.AssertFalse(t, root.StartedAt.IsZero(), "root has a start time")

	child := tree.Child(root, "tool:search", map[string]any{"q": "weather"})
	core.AssertEqual(t, "run-2", child.ID)
	core.AssertEqual(t, "run-1", child.ParentID)

	tree.Finish(child, map[string]any{"hits": 3}, map[string]any{"tokens": 12})
	core.AssertEqual(t, StatusCompleted, child.Status)
	core.AssertEqual(t, 3, child.Outputs["hits"])
	core.AssertEqual(t, 12, child.Usage.(map[string]any)["tokens"])
	core.AssertTrue(t, child.EndedAt.After(child.StartedAt), "end after start")

	tree.Finish(root, map[string]any{"reply": "sunny"}, map[string]any{"tokens": 30})
	core.AssertEqual(t, StatusCompleted, root.Status)
	core.AssertEqual(t, "sunny", root.Outputs["reply"])

	// Both runs reached the sink, child before root's final emit.
	runs := sink.Runs()
	core.AssertEqual(t, 2, len(runs))
	core.AssertEqual(t, "run-2", runs[0].ID)
	core.AssertEqual(t, "run-1", runs[1].ID)
	core.AssertEqual(t, StatusCompleted, runs[1].Status)

	// Children are tracked under the parent in the tree.
	kids := tree.Children(root.ID)
	core.AssertEqual(t, 1, len(kids))
	core.AssertEqual(t, "run-2", kids[0].ID)
}

func TestObs_RunTree_Bad(t *core.T) {
	// The fail path: a run that errors is marked failed, carries the message,
	// gets an end time, and is emitted to the sink.
	sink := NewMemorySink()
	tree := NewRunTree(seqIDs(), (&fixedClock{}).Now)
	tree.Emit(sink)

	root := tree.StartRun("chat", map[string]any{"prompt": "boom"})
	tree.Fail(root, core.E("obs", "model unavailable", nil))

	core.AssertEqual(t, StatusFailed, root.Status)
	core.AssertTrue(t, core.Contains(root.Err, "model unavailable"), "error message captured")
	core.AssertFalse(t, root.EndedAt.IsZero(), "failed run has an end time")

	runs := sink.Runs()
	core.AssertEqual(t, 1, len(runs))
	core.AssertEqual(t, StatusFailed, runs[0].Status)

	// A nil error fails the run without panicking and leaves an empty message.
	other := tree.StartRun("chat", nil)
	tree.Fail(other, nil)
	core.AssertEqual(t, StatusFailed, other.Status)
	core.AssertEqual(t, "", other.Err)
}

func TestObs_RunTree_Ugly(t *core.T) {
	// Edge shapes must not panic. Finishing/failing a nil run is a no-op; a
	// child of nil becomes a root; an unknown parent id still parents by id.
	sink := NewMemorySink()
	tree := NewRunTree(seqIDs(), (&fixedClock{}).Now)
	tree.Emit(sink)

	// nil run is inert.
	tree.Finish(nil, map[string]any{"x": 1}, nil)
	tree.Fail(nil, core.E("obs", "ignored", nil))
	core.AssertEqual(t, 0, len(sink.Runs()), "nil runs never emit")

	// Child of nil parent is promoted to a root (no parent id).
	orphan := tree.Child(nil, "detached", nil)
	core.AssertEqual(t, "", orphan.ParentID)
	core.AssertEqual(t, StatusRunning, orphan.Status)
	core.AssertEqual(t, 0, len(orphan.Inputs), "nil inputs become an empty map")

	// Finishing with nil outputs leaves an empty (non-nil) output map.
	tree.Finish(orphan, nil, nil)
	core.AssertEqual(t, 0, len(orphan.Outputs), "nil outputs become an empty map")
	core.AssertEqual(t, StatusCompleted, orphan.Status)

	// A tree with no emit sink still runs without panicking.
	silent := NewRunTree(seqIDs(), (&fixedClock{}).Now)
	r := silent.StartRun("solo", nil)
	silent.Finish(r, nil, nil)
	core.AssertEqual(t, StatusCompleted, r.Status)
}

func TestObs_Feedback_Good(t *core.T) {
	// Feedback attaches scores to a run by id; MeanByKey averages each key over
	// every recorded score for that run.
	sink := NewMemorySink()
	tree := NewRunTree(seqIDs(), (&fixedClock{}).Now)
	tree.Emit(sink)

	root := tree.StartRun("chat", nil)
	tree.Finish(root, nil, nil)

	tree.Record(Feedback{RunID: root.ID, Key: "quality", Score: 0.8, Source: "human"})
	tree.Record(Feedback{RunID: root.ID, Key: "quality", Score: 0.6, Comment: "ok", Source: "evaluator"})
	tree.Record(Feedback{RunID: root.ID, Key: "ethics", Score: 1.0, Source: "heuristic"})

	// Sink recorded all three feedback entries.
	core.AssertEqual(t, 3, len(sink.FeedbackEntries()))

	means := tree.MeanByKey(root.ID)
	core.AssertEqual(t, 2, len(means))
	core.AssertEqual(t, 0.7, means["quality"]) // (0.8 + 0.6) / 2
	core.AssertEqual(t, 1.0, means["ethics"])
}

func TestObs_Feedback_Bad(t *core.T) {
	// Feedback for an unknown run id records to the sink but contributes no
	// means for any other run; querying a run with no feedback yields an empty
	// (non-nil) map.
	sink := NewMemorySink()
	tree := NewRunTree(seqIDs(), (&fixedClock{}).Now)
	tree.Emit(sink)

	root := tree.StartRun("chat", nil)
	tree.Finish(root, nil, nil)

	tree.Record(Feedback{RunID: "ghost", Key: "quality", Score: 0.5, Source: "human"})

	// The known run has no feedback of its own.
	means := tree.MeanByKey(root.ID)
	core.AssertEqual(t, 0, len(means))

	// The ghost id still aggregates its own recorded feedback.
	ghost := tree.MeanByKey("ghost")
	core.AssertEqual(t, 0.5, ghost["quality"])

	// It did land in the sink regardless.
	core.AssertEqual(t, 1, len(sink.FeedbackEntries()))
}

func TestObs_Feedback_Ugly(t *core.T) {
	// Empty / degenerate cases must not panic. Feedback with an empty key still
	// aggregates under "". A tree with no sink records means in-memory only.
	silent := NewRunTree(seqIDs(), (&fixedClock{}).Now)
	r := silent.StartRun("solo", nil)
	silent.Finish(r, nil, nil)

	silent.Record(Feedback{RunID: r.ID, Score: 0.25, Source: "heuristic"})
	silent.Record(Feedback{RunID: r.ID, Score: 0.75, Source: "heuristic"})
	means := silent.MeanByKey(r.ID)
	core.AssertEqual(t, 0.5, means[""]) // empty key folds together

	// Mean over a never-seen id is empty, not a panic.
	none := silent.MeanByKey("never")
	core.AssertEqual(t, 0, len(none))

	// MemorySink is safe under concurrent writers.
	sink := NewMemorySink()
	var wg sync.WaitGroup
	for range 50 {
		wg.Go(func() {
			sink.Run(Run{ID: "x"})
			sink.Feedback(Feedback{RunID: "x", Score: 1})
		})
	}
	wg.Wait()
	core.AssertEqual(t, 50, len(sink.Runs()))
	core.AssertEqual(t, 50, len(sink.FeedbackEntries()))
}
