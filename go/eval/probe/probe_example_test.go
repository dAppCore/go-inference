// SPDX-Licence-Identifier: EUPL-1.2

package probe_test

import (
	core "dappco.re/go"
	"dappco.re/go/inference/eval/probe"
)

// ExampleNewBus builds a fanout bus over a recorder and emits one event,
// then reads it back.
func ExampleNewBus() {
	recorder := probe.NewRecorder()
	bus := probe.NewBus(recorder)
	bus.EmitProbe(probe.Event{Kind: probe.KindToken, Token: &probe.Token{ID: 7}})
	core.Println(len(recorder.Events()))
	// Output: 1
}

// ExampleBus_Add attaches a sink after construction, then emits.
func ExampleBus_Add() {
	bus := probe.NewBus()
	recorder := probe.NewRecorder()
	bus.Add(recorder)
	bus.EmitProbe(probe.Event{Kind: probe.KindToken})
	core.Println(len(recorder.Events()))
	// Output: 1
}

// ExampleBus_EmitProbe fans a single event out to two sinks.
func ExampleBus_EmitProbe() {
	a := probe.NewRecorder()
	b := probe.NewRecorder()
	bus := probe.NewBus(a, b)
	bus.EmitProbe(probe.Event{Kind: probe.KindEntropy, Entropy: &probe.Entropy{Value: 1.5}})
	core.Println(len(a.Events()), len(b.Events()))
	// Output: 1 1
}

// ExampleNewRecorder records an event and reads the in-memory copy.
func ExampleNewRecorder() {
	recorder := probe.NewRecorder()
	recorder.EmitProbe(probe.Event{Kind: probe.KindToken, Token: &probe.Token{ID: 42, Text: "hi"}})
	core.Println(recorder.Events()[0].Token.Text)
	// Output: hi
}

// ExampleRecorder_EmitProbe records a defensive copy: mutating the caller's
// payload after the emit does not change the stored event.
func ExampleRecorder_EmitProbe() {
	recorder := probe.NewRecorder()
	event := probe.Event{Kind: probe.KindToken, Token: &probe.Token{ID: 1, Text: "answer"}}
	recorder.EmitProbe(event)
	event.Token.Text = "mutated" // caller-side mutation after emit
	core.Println(recorder.Events()[0].Token.Text)
	// Output: answer
}

// ExampleRecorder_Events returns recorded events without aliasing storage —
// two reads return distinct payload pointers.
func ExampleRecorder_Events() {
	recorder := probe.NewRecorder()
	recorder.EmitProbe(probe.Event{Kind: probe.KindToken, Token: &probe.Token{ID: 7}})
	first := recorder.Events()
	second := recorder.Events()
	core.Println(first[0].Token == second[0].Token)
	// Output: false
}

// ExampleSinkFunc_EmitProbe adapts a plain function into a Sink.
func ExampleSinkFunc_EmitProbe() {
	var seen probe.Kind
	sink := probe.SinkFunc(func(e probe.Event) { seen = e.Kind })
	sink.EmitProbe(probe.Event{Kind: probe.KindLogits})
	core.Println(string(seen))
	// Output: logits
}

// ExampleCloneEvent deep-copies an event so the clone is detached from the
// source's payload pointers.
func ExampleCloneEvent() {
	src := probe.Event{Kind: probe.KindToken, Token: &probe.Token{ID: 1, Text: "x"}}
	clone := probe.CloneEvent(src)
	src.Token.Text = "mutated"
	core.Println(clone.Token.Text)
	// Output: x
}
