// SPDX-Licence-Identifier: EUPL-1.2

package inference

import core "dappco.re/go"

func ExampleProbeSinkFunc() {
	sink := ProbeSinkFunc(func(event ProbeEvent) {
		core.Println(event.Kind, event.Token.Text)
	})

	sink.EmitProbe(ProbeEvent{
		Kind:  ProbeEventToken,
		Token: &ProbeToken{Text: "hello"},
	})
	// Output: token hello
}

func ExampleProbeSinkFunc_EmitProbe() {
	sink := ProbeSinkFunc(func(event ProbeEvent) {
		core.Println(event.Kind)
	})

	sink.EmitProbe(ProbeEvent{Kind: ProbeEventTraining})
	// Output: training
}

func ExampleNewProbeBus() {
	var seen int
	bus := NewProbeBus(ProbeSinkFunc(func(ProbeEvent) { seen++ }))

	bus.EmitProbe(ProbeEvent{Kind: ProbeEventEntropy})

	core.Println(seen)
	// Output: 1
}

func ExampleProbeBus() {
	var seen int
	bus := NewProbeBus(
		ProbeSinkFunc(func(ProbeEvent) { seen++ }),
		ProbeSinkFunc(func(ProbeEvent) { seen++ }),
	)

	bus.EmitProbe(ProbeEvent{Kind: ProbeEventEntropy})

	core.Println(seen)
	// Output: 2
}

func ExampleProbeBus_Add() {
	var seen int
	bus := NewProbeBus()
	bus.Add(ProbeSinkFunc(func(ProbeEvent) { seen++ }))

	bus.EmitProbe(ProbeEvent{Kind: ProbeEventResidual})

	core.Println(seen)
	// Output: 1
}

func ExampleProbeBus_EmitProbe() {
	var kind ProbeEventKind
	bus := NewProbeBus(ProbeSinkFunc(func(event ProbeEvent) {
		kind = event.Kind
	}))

	bus.EmitProbe(ProbeEvent{Kind: ProbeEventCachePressure})

	core.Println(kind)
	// Output: cache_pressure
}
