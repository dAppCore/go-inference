// SPDX-Licence-Identifier: EUPL-1.2

package probe

import (
	"sync"
	"testing"
)

// fullPayloadEvent builds an Event carrying every payload pointer set —
// the fixture both CloneEvent and (through Recorder.Events) cloneEventInto
// must deep-copy without aliasing. Score.Values exercises cloneScoreValues.
func fullPayloadEvent() Event {
	return Event{
		Kind: KindLogits, Step: 1,
		Token:           &Token{ID: 1, Text: "x"},
		Logits:          &Logits{Shape: []int32{1, 2}, Top: []Logit{{TokenID: 1}}, Values: []float32{0.1}, Meta: map[string]string{"k": "v"}},
		Entropy:         &Entropy{Value: 1.2, Unit: "nats"},
		SelectedHeads:   &HeadSelection{Heads: []int{0, 1}, Scores: []float64{0.5}},
		LayerCoherence:  &LayerCoherence{Layer: 3, KeyCoherence: 0.5, KVCoupling: 0.7},
		RouterDecision:  &RouterDecision{ExpertIDs: []int{0, 1}, Weights: []float32{0.5, 0.5}},
		ExpertResidency: &ExpertResidency{Action: ExpertResidencyActionPageIn, ExpertIDs: []int{0}},
		Residual:        &ResidualSummary{Layer: 3, Mean: 0.01, RMS: 0.15},
		Cache:           &CachePressure{LayerCount: 2, CacheTokens: 16},
		Memory:          &MemoryPressure{ActiveBytes: 1 << 20, PeakBytes: 2 << 20},
		Training:        &Training{Step: 1, Loss: 0.25, LearningRate: 3e-4},
		Score:           &Score{Label: "kernel", Values: map[string]float64{"lek": 61.5}},
		Meta:            map[string]string{"prompt": "p"},
	}
}

// assertFullPayloadDetached mutates every slice/map/scalar in src and
// asserts none of the mutations surface in out — out must be a full deep
// copy. Shared by the CloneEvent and Recorder.Events round-trip tests.
func assertFullPayloadDetached(t *testing.T, src Event, out Event) {
	t.Helper()
	src.Token.Text = "mutated"
	src.Logits.Shape[0] = 99
	src.Logits.Top[0].TokenID = 99
	src.Logits.Values[0] = 9
	src.Logits.Meta["k"] = "z"
	src.Entropy.Value = 99
	src.SelectedHeads.Heads[0] = 99
	src.SelectedHeads.Scores[0] = 99
	src.LayerCoherence.KVCoupling = 99
	src.RouterDecision.ExpertIDs[0] = 99
	src.RouterDecision.Weights[0] = 99
	src.ExpertResidency.ExpertIDs[0] = 99
	src.Residual.RMS = 99
	src.Cache.CacheTokens = 99
	src.Memory.ActiveBytes = 99
	src.Training.Loss = 99
	src.Score.Values["lek"] = 99
	src.Meta["prompt"] = "mutated"

	if out.Token.Text != "x" {
		t.Fatal("shared Token")
	}
	if out.Logits.Shape[0] != 1 || out.Logits.Top[0].TokenID != 1 || out.Logits.Values[0] != 0.1 || out.Logits.Meta["k"] != "v" {
		t.Fatalf("shared Logits internals: %+v", out.Logits)
	}
	if out.Entropy.Value != 1.2 {
		t.Fatalf("shared Entropy: %+v", out.Entropy)
	}
	if out.SelectedHeads.Heads[0] != 0 || out.SelectedHeads.Scores[0] != 0.5 {
		t.Fatalf("shared SelectedHeads: %+v", out.SelectedHeads)
	}
	if out.LayerCoherence.KVCoupling != 0.7 {
		t.Fatalf("shared LayerCoherence: %+v", out.LayerCoherence)
	}
	if out.RouterDecision.ExpertIDs[0] != 0 || out.RouterDecision.Weights[0] != 0.5 {
		t.Fatalf("shared RouterDecision: %+v", out.RouterDecision)
	}
	if out.ExpertResidency.ExpertIDs[0] != 0 {
		t.Fatalf("shared ExpertResidency: %+v", out.ExpertResidency)
	}
	if out.Residual.RMS != 0.15 {
		t.Fatalf("shared Residual: %+v", out.Residual)
	}
	if out.Cache.CacheTokens != 16 {
		t.Fatalf("shared Cache: %+v", out.Cache)
	}
	if out.Memory.ActiveBytes != 1<<20 {
		t.Fatalf("shared Memory: %+v", out.Memory)
	}
	if out.Training.Loss != 0.25 {
		t.Fatalf("shared Training: %+v", out.Training)
	}
	if out.Score.Values["lek"] != 61.5 {
		t.Fatalf("shared Score.Values: %+v", out.Score)
	}
	if out.Meta["prompt"] != "p" {
		t.Fatalf("shared Meta: %+v", out.Meta)
	}
}

// --- SinkFunc.EmitProbe ---

// Good: a SinkFunc dispatches the event to the wrapped function verbatim.
func TestProbe_SinkFunc_EmitProbe_Good(t *testing.T) {
	var got Event
	f := SinkFunc(func(e Event) { got = e })
	f.EmitProbe(Event{Kind: KindRouterDecision, RouterDecision: &RouterDecision{Layer: 2}})
	if got.Kind != KindRouterDecision || got.RouterDecision == nil || got.RouterDecision.Layer != 2 {
		t.Fatalf("got = %+v", got)
	}
}

// Bad: SinkFunc passes the event through unfiltered — it does not validate
// payload/kind agreement, so a Kind with a mismatched (nil) payload is still
// delivered as-is. EmitProbe is a pure adapter, not a gatekeeper.
func TestProbe_SinkFunc_EmitProbe_Bad(t *testing.T) {
	var got Event
	delivered := false
	f := SinkFunc(func(e Event) { got = e; delivered = true })
	// KindToken with no Token payload — malformed, but EmitProbe forwards it.
	f.EmitProbe(Event{Kind: KindToken})
	if !delivered {
		t.Fatal("SinkFunc.EmitProbe dropped a malformed event; it must forward verbatim")
	}
	if got.Token != nil {
		t.Fatalf("SinkFunc.EmitProbe synthesised a payload: %+v", got.Token)
	}
}

// Ugly: a nil SinkFunc must no-op silently on dispatch, and an explicitly
// nil-valued SinkFunc(nil) behaves identically — neither panics, and a
// real wrapped function still dispatches afterwards, proving the nil guard
// is per-value, not a permanently poisoned type.
func TestProbe_SinkFunc_EmitProbe_Ugly(t *testing.T) {
	var f SinkFunc
	f.EmitProbe(Event{Kind: KindToken})             // zero value — must not panic
	SinkFunc(nil).EmitProbe(Event{Kind: KindToken}) // explicit nil — must not panic
	dispatched := false
	live := SinkFunc(func(Event) { dispatched = true })
	live.EmitProbe(Event{Kind: KindToken})
	if !dispatched {
		t.Fatal("a non-nil SinkFunc stopped dispatching after nil EmitProbe calls")
	}
}

// --- NewBus ---

// Good: NewBus over two recorders fans a single emit to both, and the
// concurrent-emit case still lands every event exactly once.
func TestProbe_NewBus_Good(t *testing.T) {
	t.Run("FansOutToAllSinks", func(t *testing.T) {
		rec1 := NewRecorder()
		rec2 := NewRecorder()
		bus := NewBus(rec1, rec2)
		bus.EmitProbe(Event{Kind: KindToken, Token: &Token{ID: 1}})
		if len(rec1.Events()) != 1 || len(rec2.Events()) != 1 {
			t.Fatalf("fanout = rec1:%d rec2:%d, want 1 each", len(rec1.Events()), len(rec2.Events()))
		}
	})
	t.Run("ConcurrentSafe", func(t *testing.T) {
		bus := NewBus()
		rec := NewRecorder()
		bus.Add(rec)
		var wg sync.WaitGroup
		for range 100 {
			wg.Go(func() {
				bus.EmitProbe(Event{Kind: KindToken})
			})
		}
		wg.Wait()
		if got := len(rec.Events()); got != 100 {
			t.Fatalf("concurrent emit count = %d, want 100", got)
		}
	})
}

// Bad: NewBus filters nil sinks out of the variadic argument list, so a bus
// built from only nil sinks holds none and a subsequent emit reaches nobody
// (and must not panic). The constructor never stores a nil-interface sink.
func TestProbe_NewBus_Bad(t *testing.T) {
	bus := NewBus(nil, nil)
	rec := NewRecorder()
	bus.Add(rec)
	bus.EmitProbe(Event{Kind: KindToken, Token: &Token{ID: 1}})
	// Only the later-added real recorder receives the event; the nil sinks
	// were dropped at construction rather than emitted to.
	if len(rec.Events()) != 1 {
		t.Fatalf("rec.Events() = %d, want 1 (nil ctor sinks must be ignored)", len(rec.Events()))
	}
}

// Ugly: NewBus() with no sinks yields an empty, usable bus — emitting to it
// is a clean no-op (nil stored snapshot), never a panic.
func TestProbe_NewBus_Ugly(t *testing.T) {
	bus := NewBus()
	bus.EmitProbe(Event{Kind: KindToken, Token: &Token{ID: 1}}) // must not panic
	rec := NewRecorder()
	bus.Add(rec)
	bus.EmitProbe(Event{Kind: KindToken})
	if len(rec.Events()) != 1 {
		t.Fatalf("empty NewBus stayed usable: rec.Events() = %d, want 1", len(rec.Events()))
	}
}

// --- Bus.Add ---

// Good: Add to a bus that already holds a sink copies the existing slice and
// grows it — the grow path (the constructor seeds the first sink).
func TestProbe_Bus_Add_Good(t *testing.T) {
	rec1 := NewRecorder()
	rec2 := NewRecorder()
	bus := NewBus(rec1) // bus already has a stored slice
	bus.Add(rec2)       // copy current + append
	bus.EmitProbe(Event{Kind: KindToken, Token: &Token{ID: 1}})
	if len(rec1.Events()) != 1 || len(rec2.Events()) != 1 {
		t.Fatalf("after grow = rec1:%d rec2:%d, want 1 each", len(rec1.Events()), len(rec2.Events()))
	}
}

// Bad: Add(nil) is ignored — a nil sink is never appended, so a real sink
// added afterwards is the only one that receives events.
func TestProbe_Bus_Add_Bad(t *testing.T) {
	bus := NewBus()
	bus.Add(nil) // must not panic; no sink added
	rec := NewRecorder()
	bus.Add(rec)
	bus.EmitProbe(Event{Kind: KindToken})
	if len(rec.Events()) != 1 {
		t.Fatalf("rec.Events() len = %d, want 1 (Add(nil) must be ignored)", len(rec.Events()))
	}
}

// Ugly: Add on a nil *Bus receiver must no-op (not panic), and a real bus is
// unaffected — it still accepts an Add and delivers to the added sink. The
// nil guard is on the receiver only; live buses keep working.
func TestProbe_Bus_Add_Ugly(t *testing.T) {
	var b *Bus
	b.Add(NewRecorder()) // nil receiver — must not panic
	live := NewBus()
	rec := NewRecorder()
	live.Add(rec) // a real bus still grows
	live.EmitProbe(Event{Kind: KindToken})
	if len(rec.Events()) != 1 {
		t.Fatalf("real bus Add stopped working after nil-receiver Add: %d, want 1", len(rec.Events()))
	}
}

// --- Bus.EmitProbe ---

// Good: every fanout branch — all-owned, non-owned (pre-cloned), and mixed —
// delivers a fully detached event to each sink.
func TestProbe_Bus_EmitProbe_Good(t *testing.T) {
	t.Run("OwnedSinkDeepClonedOnRead", func(t *testing.T) {
		// The owned-sink path: the Bus skips on-emit cloning, but
		// Recorder.Events() returns deep-cloned events so consumers can
		// never alias storage across reads.
		rec := NewRecorder()
		bus := NewBus(rec)
		bus.EmitProbe(Event{
			Kind:  KindToken,
			Token: &Token{ID: 7, Text: "answer"},
			Meta:  map[string]string{"k": "v"},
		})
		first := rec.Events()
		second := rec.Events()
		if len(first) != 1 || len(second) != 1 {
			t.Fatalf("events len first=%d second=%d, want 1 each", len(first), len(second))
		}
		if first[0].Token == second[0].Token {
			t.Fatal("Events() returned aliased Token pointers across calls")
		}
		first[0].Token.ID = 99
		first[0].Meta["k"] = "mutated"
		if second[0].Token.ID != 7 {
			t.Fatalf("second snapshot Token.ID = %d, want 7 (snapshots aliased)", second[0].Token.ID)
		}
		if second[0].Meta["k"] != "v" {
			t.Fatalf("second snapshot Meta[k] = %q, want v (snapshots aliased)", second[0].Meta["k"])
		}
	})
	t.Run("NonOwnedSinkReceivesClone", func(t *testing.T) {
		// A SinkFunc does not implement ownedEventSink, so the Bus takes
		// the non-owned path and pre-clones the event before delivery —
		// the single-sink CloneEvent branch.
		var got Event
		bus := NewBus(SinkFunc(func(e Event) { got = e }))
		src := Event{Kind: KindToken, Token: &Token{ID: 7, Text: "x"}, Meta: map[string]string{"k": "v"}}
		bus.EmitProbe(src)
		src.Token.Text = "mutated"
		src.Meta["k"] = "mutated"
		if got.Token == nil || got.Token.Text != "x" {
			t.Fatalf("non-owned sink got aliased Token: %+v", got.Token)
		}
		if got.Meta["k"] != "v" {
			t.Fatalf("non-owned sink got aliased Meta: %+v", got.Meta)
		}
	})
	t.Run("MixedOwnedAndFuncSinks", func(t *testing.T) {
		// One owned sink (Recorder) and one non-owned (SinkFunc)
		// exercises the multi-sink owned-continue and non-owned-clone
		// branches in a single emit.
		rec := NewRecorder()
		var got Event
		bus := NewBus(rec, SinkFunc(func(e Event) { got = e }))
		bus.EmitProbe(Event{Kind: KindToken, Token: &Token{ID: 7, Text: "x"}})
		if len(rec.Events()) != 1 {
			t.Fatalf("owned sink events = %d, want 1", len(rec.Events()))
		}
		if got.Token == nil || got.Token.ID != 7 {
			t.Fatalf("func sink got = %+v", got)
		}
	})
}

// Bad: a SinkFunc panics inside EmitProbe — the Bus does not recover, so the
// panic propagates to the emitter. This pins the contract that the Bus is a
// transparent fanout, not a panic firewall around its sinks.
func TestProbe_Bus_EmitProbe_Bad(t *testing.T) {
	bus := NewBus(SinkFunc(func(Event) { panic("sink blew up") }))
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("Bus.EmitProbe swallowed a sink panic; it must propagate")
		}
	}()
	bus.EmitProbe(Event{Kind: KindToken})
}

// Ugly: the degenerate receivers — nil *Bus, empty bus, and a typed-nil
// owned sink — must all no-op on EmitProbe without panicking.
func TestProbe_Bus_EmitProbe_Ugly(t *testing.T) {
	t.Run("NilReceiver", func(t *testing.T) {
		var b *Bus
		b.EmitProbe(Event{}) // must not panic
	})
	t.Run("EmptyBusNoOp", func(t *testing.T) {
		bus := NewBus()
		bus.EmitProbe(Event{Kind: KindToken, Token: &Token{ID: 1}}) // snap==nil early return
	})
	t.Run("TypedNilOwnedSink", func(t *testing.T) {
		// A typed-nil sink (a nil *Recorder boxed in a Sink) is not a nil
		// interface, so NewBus stores it and the owned fast-path calls
		// emitProbeOwned on a nil receiver — which must no-op, not panic.
		var nilRec *Recorder
		bus := NewBus(nilRec)
		bus.EmitProbe(Event{Kind: KindToken, Token: &Token{ID: 1}}) // must not panic
	})
}

// --- NewRecorder ---

// Good: NewRecorder returns a fresh, usable recorder that starts empty and
// records the events emitted to it.
func TestProbe_NewRecorder_Good(t *testing.T) {
	rec := NewRecorder()
	if got := rec.Events(); got != nil {
		t.Fatalf("fresh NewRecorder().Events() = %v, want nil", got)
	}
	rec.EmitProbe(Event{Kind: KindToken, Token: &Token{ID: 1}})
	if len(rec.Events()) != 1 {
		t.Fatalf("NewRecorder did not record: Events() = %d, want 1", len(rec.Events()))
	}
}

// Bad: NewRecorder records every event it is given without de-duplication or
// validation — emitting the same malformed (nil-payload) event twice yields
// two recorded entries, not one. The recorder is a faithful log, not a filter.
func TestProbe_NewRecorder_Bad(t *testing.T) {
	rec := NewRecorder()
	bad := Event{Kind: KindToken} // KindToken with no Token payload
	rec.EmitProbe(bad)
	rec.EmitProbe(bad)
	got := rec.Events()
	if len(got) != 2 {
		t.Fatalf("recorder de-duplicated or dropped events: Events() = %d, want 2", len(got))
	}
	if got[0].Token != nil {
		t.Fatalf("recorder synthesised a payload: %+v", got[0].Token)
	}
}

// Ugly: a nil *Recorder built WITHOUT NewRecorder must no-op on both
// EmitProbe and Events rather than panic, while a sibling recorder from
// NewRecorder records normally — the constructor is what makes the receiver
// live, and the nil path degrades silently instead of corrupting it.
func TestProbe_NewRecorder_Ugly(t *testing.T) {
	var r *Recorder                                           // never went through NewRecorder
	r.EmitProbe(Event{Kind: KindToken, Token: &Token{ID: 1}}) // must not panic
	if got := r.Events(); got != nil {
		t.Fatalf("nil Recorder.Events() = %v, want nil", got)
	}
	// A real NewRecorder instance is unaffected by the nil sibling's emit.
	live := NewRecorder()
	live.EmitProbe(Event{Kind: KindToken, Token: &Token{ID: 2}})
	if len(live.Events()) != 1 {
		t.Fatalf("NewRecorder instance did not record beside the nil path: %d, want 1", len(live.Events()))
	}
}

// --- Recorder.EmitProbe ---

// Good: EmitProbe records a defensive copy — mutating the caller's payloads
// after the emit never surfaces in the recorded event.
func TestProbe_Recorder_EmitProbe_Good(t *testing.T) {
	recorder := NewRecorder()
	event := Event{
		Kind:  KindLogits,
		Phase: PhaseDecode,
		Step:  3,
		Token: &Token{
			ID: 7, Text: "answer", PromptTokens: 11, GeneratedTokens: 2,
		},
		Logits: &Logits{
			Shape: []int32{1, 4}, VocabSize: 4,
			MaxTokenID: 7, MaxLogit: 4.5,
			Top: []Logit{{TokenID: 7, Logit: 4.5, Probability: 0.75}},
		},
		Cache: &CachePressure{
			LayerCount: 2, CacheTokens: 16, ProcessedTokens: 18,
		},
		Meta: map[string]string{"prompt_id": "abc"},
	}
	recorder.EmitProbe(event)
	// Mutate caller-side payloads — should not surface in recorded copy.
	event.Token.Text = "mutated"
	event.Logits.Top[0].Probability = 0.0
	event.Cache.ProcessedTokens = 99
	event.Meta["prompt_id"] = "changed"
	events := recorder.Events()
	if len(events) != 1 {
		t.Fatalf("Events() len = %d, want 1", len(events))
	}
	got := events[0]
	if got.Token.Text != "answer" {
		t.Fatalf("Token.Text = %q, want answer (defensive copy)", got.Token.Text)
	}
	if got.Logits.Top[0].Probability != 0.75 {
		t.Fatalf("Logits.Top probability = %v, want 0.75 (defensive copy)", got.Logits.Top[0].Probability)
	}
	if got.Cache.ProcessedTokens != 18 {
		t.Fatalf("Cache.ProcessedTokens = %d, want 18 (defensive copy)", got.Cache.ProcessedTokens)
	}
	if got.Meta["prompt_id"] != "abc" {
		t.Fatalf("Meta[prompt_id] = %q, want abc (defensive copy)", got.Meta["prompt_id"])
	}
}

// Bad: EmitProbe with an empty (zero-value) Event still records an entry —
// the recorder does not reject events that carry no payload. It logs what it
// is handed, even when that is nothing useful.
func TestProbe_Recorder_EmitProbe_Bad(t *testing.T) {
	rec := NewRecorder()
	rec.EmitProbe(Event{}) // zero Kind, no payloads
	got := rec.Events()
	if len(got) != 1 {
		t.Fatalf("empty event was dropped: Events() = %d, want 1", len(got))
	}
	if got[0].Kind != "" || got[0].Token != nil {
		t.Fatalf("recorder mutated an empty event: %+v", got[0])
	}
}

// Ugly: EmitProbe on a nil *Recorder must no-op silently rather than panic.
func TestProbe_Recorder_EmitProbe_Ugly(t *testing.T) {
	var r *Recorder
	r.EmitProbe(Event{Kind: KindToken, Token: &Token{ID: 1}}) // must not panic
	if got := r.Events(); got != nil {
		t.Fatalf("nil Recorder after EmitProbe: Events() = %v, want nil", got)
	}
}

// --- Recorder.Events ---

// Good: Events returns a fully detached deep clone of a rich event (the
// cloneEventInto batch path), and successive reads do not alias each other.
func TestProbe_Recorder_Events_Good(t *testing.T) {
	src := fullPayloadEvent()
	rec := NewRecorder()
	rec.EmitProbe(src)
	events := rec.Events()
	if len(events) != 1 {
		t.Fatalf("Events() len = %d, want 1", len(events))
	}
	assertFullPayloadDetached(t, src, events[0])
}

// TestProbe_Recorder_Events_BatchScratchIsPerEvent proves Events' batch-clone
// path (cloneEventInto against a pre-allocated []cloneScratch — see the doc
// comment on Events) gives EACH event in the batch its own scratch slot: a
// recorder holding N distinct rich events must clone each into independent
// storage, not share one slot across the batch. A shared/aliased scratch (for
// example, a stray index bug that always wrote into scratches[0]) would make
// every returned event's payload pointers alias the LAST event cloned into
// that slot — invisible to a length-1-batch test, so this drives a batch of
// three distinct events and cross-checks every pair.
func TestProbe_Recorder_Events_BatchScratchIsPerEvent(t *testing.T) {
	rec := NewRecorder()
	rec.EmitProbe(Event{Kind: KindToken, Token: &Token{ID: 1, Text: "first"}, Meta: map[string]string{"k": "1"}})
	rec.EmitProbe(Event{Kind: KindToken, Token: &Token{ID: 2, Text: "second"}, Meta: map[string]string{"k": "2"}})
	rec.EmitProbe(Event{Kind: KindToken, Token: &Token{ID: 3, Text: "third"}, Meta: map[string]string{"k": "3"}})

	events := rec.Events()
	if len(events) != 3 {
		t.Fatalf("Events() len = %d, want 3", len(events))
	}
	// Each event must still carry its OWN values — a shared scratch slot
	// would collapse them all to the last-cloned event's values.
	wantText := []string{"first", "second", "third"}
	wantMeta := []string{"1", "2", "3"}
	for i, e := range events {
		if e.Token.ID != int32(i+1) || e.Token.Text != wantText[i] {
			t.Fatalf("events[%d].Token = %+v, want ID %d / text %q", i, e.Token, i+1, wantText[i])
		}
		if e.Meta["k"] != wantMeta[i] {
			t.Fatalf("events[%d].Meta[k] = %q, want %q", i, e.Meta["k"], wantMeta[i])
		}
	}
	// Mutating one event's clone must not alter any sibling's clone — proves
	// the payload pointers do not alias a shared backing scratch.
	events[0].Token.Text = "mutated"
	events[0].Meta["k"] = "mutated"
	if events[1].Token.Text != "second" || events[1].Meta["k"] != "2" {
		t.Fatalf("events[1] changed after mutating events[0]: token=%+v meta=%+v — scratch slots are aliased", events[1].Token, events[1].Meta)
	}
	if events[2].Token.Text != "third" || events[2].Meta["k"] != "3" {
		t.Fatalf("events[2] changed after mutating events[0]: token=%+v meta=%+v — scratch slots are aliased", events[2].Token, events[2].Meta)
	}
}

// Bad: Events is a read-only snapshot — mutating the returned slice's
// payloads must not corrupt the recorder's stored copy, so a second Events()
// call still returns the original values. A caller cannot scribble through
// the read API back into recorder state.
func TestProbe_Recorder_Events_Bad(t *testing.T) {
	rec := NewRecorder()
	rec.EmitProbe(Event{Kind: KindToken, Token: &Token{ID: 7, Text: "answer"}})
	first := rec.Events()
	first[0].Token.Text = "mutated" // scribble on the returned snapshot
	first[0].Token.ID = 99
	second := rec.Events()
	if second[0].Token.Text != "answer" || second[0].Token.ID != 7 {
		t.Fatalf("mutation through Events() leaked into recorder state: %+v", second[0].Token)
	}
}

// Ugly: Events on a non-nil recorder that never recorded returns nil (not an
// empty allocated slice), and on a nil *Recorder it also returns nil.
func TestProbe_Recorder_Events_Ugly(t *testing.T) {
	t.Run("EmptyReturnsNil", func(t *testing.T) {
		rec := NewRecorder()
		if got := rec.Events(); got != nil {
			t.Fatalf("empty Recorder.Events() = %v, want nil", got)
		}
	})
	t.Run("NilReceiverReturnsNil", func(t *testing.T) {
		var r *Recorder
		if got := r.Events(); got != nil {
			t.Fatalf("nil Recorder.Events() = %v, want nil", got)
		}
	})
}

// --- CloneEvent ---

// Good: CloneEvent deep-copies every payload pointer of a rich event so the
// clone is fully detached from the source.
func TestProbe_CloneEvent_Good(t *testing.T) {
	src := fullPayloadEvent()
	out := CloneEvent(src)
	assertFullPayloadDetached(t, src, out)
}

// Bad: CloneEvent must NOT manufacture payloads that the source lacks — a
// score event whose Values map is empty clones to a Score with a nil Values
// map, not an empty allocated one (the cloneScoreValues empty short-circuit).
// Treating "absent" as "present-but-empty" would be a silent data change.
func TestProbe_CloneEvent_Bad(t *testing.T) {
	src := Event{Kind: KindScore, Score: &Score{Label: "x"}}
	out := CloneEvent(src)
	if out.Score == nil || out.Score.Label != "x" {
		t.Fatalf("CloneEvent lost Score: %+v", out.Score)
	}
	if out.Score.Values != nil {
		t.Fatalf("CloneEvent allocated empty Score.Values: %+v", out.Score.Values)
	}
}

// Ugly: CloneEvent on an event with no payload pointers preserves the scalar
// fields and leaves every payload pointer nil — no phantom allocations.
func TestProbe_CloneEvent_Ugly(t *testing.T) {
	src := Event{Kind: KindToken, Step: 1}
	out := CloneEvent(src)
	if out.Kind != KindToken || out.Step != 1 {
		t.Fatalf("CloneEvent lost scalar fields: %+v", out)
	}
	if out.Token != nil || out.Logits != nil || out.Entropy != nil {
		t.Fatalf("CloneEvent created phantom payload pointers: %+v", out)
	}
}

// --- Bonus coverage: exported constants ---
// These assert the typed string constants the wire schema depends on; they
// carry no public function/method symbol of their own but pin the vocabulary
// downstream dashboards read.

func TestProbe_ExpertResidencyActionConstants_AreStrings(t *testing.T) {
	cases := []struct {
		got, want ExpertResidencyAction
	}{
		{ExpertResidencyActionStartup, "startup"},
		{ExpertResidencyActionPageIn, "page_in"},
		{ExpertResidencyActionEvict, "evict"},
		{ExpertResidencyActionHit, "hit"},
	}
	for _, c := range cases {
		if c.got != c.want {
			t.Fatalf("constant = %q, want %q", c.got, c.want)
		}
	}
}

func TestProbe_KindAndPhaseConstants_StringValues(t *testing.T) {
	if KindToken != "token" || KindTraining != "training" || PhasePrefill != "prefill" {
		t.Fatal("constants do not have expected string values")
	}
}
