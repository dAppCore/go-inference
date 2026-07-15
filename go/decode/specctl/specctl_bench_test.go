// SPDX-Licence-Identifier: EUPL-1.2

// Allocation contracts for the adaptive speculative-length controller (AX-11).
// Record and NextLength are the per-decode hot path — both fold or read scalar
// state under a mutex, with no slice/map/string work — so they must not
// allocate. These benches pin that to zero.
//
// Run: go test -bench=. -benchmem -run='^$' ./specctl/
package specctl_test

import (
	"testing"

	"dappco.re/go/inference/decode/specctl"
)

// Package sinks defeat dead-code elimination.
var (
	sinkInt   int
	sinkFloat float64
	sinkCtrl  *specctl.Adaptive
)

func BenchmarkAdaptive_Record(b *testing.B) {
	c := specctl.New(specctl.Controller{Min: 1, Max: 8, Window: 8})
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		c.Record(8, i%9) // proposed 8, accepted cycles 0..8
	}
}

func BenchmarkAdaptive_NextLength(b *testing.B) {
	c := specctl.New(specctl.Controller{Min: 1, Max: 8, Window: 8})
	c.Record(8, 5)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sinkInt = c.NextLength()
	}
}

func BenchmarkAdaptive_AcceptRate(b *testing.B) {
	c := specctl.New(specctl.Controller{Min: 1, Max: 8, Window: 8})
	c.Record(8, 5)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sinkFloat = c.AcceptRate()
	}
}

// New is the once-per-request constructor — it allocates the single running
// controller struct (the one unavoidable alloc); this pins that it is the only
// one, and that config clamping stays scalar.
func BenchmarkAdaptive_New(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sinkCtrl = specctl.New(specctl.Controller{Min: 1, Max: 8, Window: 8})
	}
}

// Reset clears the learned rate back to cold-start under the mutex — a per-request
// scalar write that must not allocate.
func BenchmarkAdaptive_Reset(b *testing.B) {
	c := specctl.New(specctl.Controller{Min: 1, Max: 8, Window: 8})
	c.Record(8, 5)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		c.Reset()
	}
}
