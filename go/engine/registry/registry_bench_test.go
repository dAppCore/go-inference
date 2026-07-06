// SPDX-Licence-Identifier: EUPL-1.2

package registry_test

import (
	"strconv"
	"testing"

	registry "dappco.re/go/inference/engine/registry"
)

// Package sinks keep the compiler from eliminating the benchmarked work.
var (
	entriesSink []registry.Entry
	okSink      bool
	valSink     any
)

// benchN is a realistic catalogue size — a few dozen model variants.
const benchN = 24

// benchEntry builds a populated catalogue entry with deterministic, varied
// content for the i-th model — two aliases, a mix of capabilities and statuses.
//
//	e := benchEntry(7)
func benchEntry(i int) registry.Entry {
	s := strconv.Itoa(i)
	e := registry.Entry{
		ID:            "lthn/model-" + s,
		Aliases:       []string{"m" + s, "alias-" + s},
		Architecture:  "gemma4",
		Params:        4_500_000_000,
		ContextLength: 131072,
		Quantisation:  "Q4_K_M",
		Format:        registry.FormatGGUF,
		MemoryBytes:   uint64(2_000_000_000 + i*1_000_000_000),
		DeviceFit:     []string{"metal", "cuda"},
		Capabilities: registry.Capabilities{
			Tools:     true,
			Vision:    i%2 == 0,
			Grammar:   true,
			Streaming: true,
		},
		Source: registry.Source{LocalPath: "/models/model-" + s},
		Status: registry.StatusReady,
	}
	if i%5 == 0 {
		e.Status = registry.StatusDraft
	}
	return e
}

// benchRegistry returns a Registry seeded with benchN realistic entries.
//
//	r := benchRegistry(b)
func benchRegistry(tb testing.TB) *registry.Registry {
	tb.Helper()
	r := registry.New()
	for i := range benchN {
		if pr := r.Put(benchEntry(i)); !pr.OK {
			tb.Fatalf("seed put %d: %v", i, pr.Error())
		}
	}
	return r
}

func BenchmarkRegistry_Resolve_ID(b *testing.B) {
	r := benchRegistry(b)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		res := r.Resolve("lthn/model-7")
		okSink = res.OK
		valSink = res.Value
	}
}

func BenchmarkRegistry_Resolve_Alias(b *testing.B) {
	r := benchRegistry(b)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		res := r.Resolve("alias-7")
		okSink = res.OK
		valSink = res.Value
	}
}

func BenchmarkRegistry_Get(b *testing.B) {
	r := benchRegistry(b)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		res := r.Get("lthn/model-7")
		okSink = res.OK
		valSink = res.Value
	}
}

func BenchmarkRegistry_Put_Update(b *testing.B) {
	r := benchRegistry(b)
	e := benchEntry(7) // existing id → update in place, no map growth
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		res := r.Put(e)
		okSink = res.OK
	}
}

func BenchmarkRegistry_List(b *testing.B) {
	r := benchRegistry(b)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		entriesSink = r.List()
	}
}

func BenchmarkRegistry_Filter(b *testing.B) {
	r := benchRegistry(b)
	f := registry.Filter{Tools: true, ReadyOnly: true}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		entriesSink = r.Filter(f)
	}
}

func BenchmarkRegistry_FitsDevice(b *testing.B) {
	r := benchRegistry(b)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		entriesSink = r.FitsDevice(96 << 30)
	}
}

func BenchmarkRegistry_FitsDeviceWith(b *testing.B) {
	r := benchRegistry(b)
	f := registry.Filter{Vision: true, ReadyOnly: true}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		entriesSink = r.FitsDeviceWith(96<<30, f)
	}
}

func BenchmarkRegistry_SetCard(b *testing.B) {
	r := benchRegistry(b)
	card := registry.ModelCard{
		IntendedUse:        "Ethical instruction following on device.",
		TrainingProvenance: "Gemma 4 + LEM ethics adapter.",
		EvalSummary:        "8-PAC unanimous pass.",
		Limitations:        "English-first.",
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		res := r.SetCard("lthn/model-7", card)
		okSink = res.OK
	}
}

func BenchmarkRegistry_GetCard(b *testing.B) {
	r := benchRegistry(b)
	if sr := r.SetCard("lthn/model-7", registry.ModelCard{IntendedUse: "x"}); !sr.OK {
		b.Fatalf("seed card: %v", sr.Error())
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		res := r.GetCard("lthn/model-7")
		okSink = res.OK
		valSink = res.Value
	}
}

func BenchmarkMemStore_List(b *testing.B) {
	s := registry.NewMemStore()
	for i := range benchN {
		if pr := s.Put(benchEntry(i)); !pr.OK {
			b.Fatalf("seed put %d: %v", i, pr.Error())
		}
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		entriesSink = s.List()
	}
}
