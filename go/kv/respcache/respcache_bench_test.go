// SPDX-Licence-Identifier: EUPL-1.2

package respcache_test

import (
	"testing"
	"time"

	"dappco.re/go/inference/kv/respcache"
)

// benchRequest is the realistic "two-message chat with a stop list" used across
// the cache benchmarks. The stop list is intentionally out of order so Key
// exercises the canonicalising sort on every call (the hot path).
func benchRequest() respcache.Request {
	return respcache.Request{
		Model: "gemma-4-e4b",
		Messages: []respcache.Message{
			{Role: "system", Content: "you are a helpful assistant"},
			{Role: "user", Content: "what is the capital of france?"},
		},
		Temperature: 0.2,
		TopP:        0.9,
		MaxTokens:   256,
		Seed:        42,
		Stop:        []string{"END", "\n\n", "STOP"},
	}
}

// benchRequestNoStop is the same shape with no stop list, isolating the
// key-building cost when the sort path is skipped.
func benchRequestNoStop() respcache.Request {
	r := benchRequest()
	r.Stop = nil
	return r
}

// Package sinks keep the optimiser from eliding the benchmarked calls.
var (
	sinkKey  string
	sinkComp respcache.Completion
	sinkHit  bool
)

// BenchmarkKey measures the canonical key derivation on the stop-present path —
// the per-request allocation hotspot (copy + sort + JSON + hash).
func BenchmarkKey(b *testing.B) {
	req := benchRequest()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sinkKey = respcache.Key(req)
	}
}

// BenchmarkKey_NoStop measures key derivation when the stop list is empty, so
// the copy+sort branch is skipped (isolates the JSON+hash floor).
func BenchmarkKey_NoStop(b *testing.B) {
	req := benchRequestNoStop()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sinkKey = respcache.Key(req)
	}
}

// BenchmarkKey_SortedStop measures key derivation when the stop list is already
// in sorted order (the common single-element / pre-sorted case): the defensive
// copy is unnecessary because the list is never mutated, so it is skipped.
func BenchmarkKey_SortedStop(b *testing.B) {
	req := benchRequest()
	req.Stop = []string{"\n\n", "END", "STOP"} // ascending: needs no copy
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sinkKey = respcache.Key(req)
	}
}

// BenchmarkGet measures a cache hit: Key derivation plus the store lookup and
// TTL check — the read side of the serving hot path.
func BenchmarkGet(b *testing.B) {
	c := respcache.New(nil)
	req := benchRequest()
	c.Set(req, respcache.Completion{Text: "paris", Model: "gemma-4-e4b", FinishReason: "stop"}, time.Hour)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sinkComp, sinkHit = c.Get(req)
	}
}

// BenchmarkSet measures a cache store: Key derivation plus the map write — the
// write side of the serving hot path.
func BenchmarkSet(b *testing.B) {
	c := respcache.New(nil)
	req := benchRequest()
	out := respcache.Completion{Text: "paris", Model: "gemma-4-e4b", FinishReason: "stop"}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		c.Set(req, out, time.Hour)
	}
}
