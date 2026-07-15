// SPDX-Licence-Identifier: EUPL-1.2

// Allocation contracts for the residency policy (AX-11). A resident-hit Touch
// (the common re-touch) bumps recency under the lock and must not allocate. The
// eviction path plans an LRU order and is the one that does real buffering work,
// so it is benched under continuous churn to keep the eviction planner hot.
//
// Run: go test -bench=. -benchmem -run='^$' ./residency/
package residency

import "testing"

var (
	benchDecision Decision
	benchIDs      []string
)

func BenchmarkManager_Touch_Hit(b *testing.B) {
	m := New(Policy{Device: "bench", BudgetBytes: 64 << 30, ConcurrentCap: 8})
	m.Touch("model-a", 4<<30) // resident; every Touch below is a hit
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchDecision = m.Touch("model-a", 4<<30)
	}
}

func BenchmarkManager_Touch_Evict(b *testing.B) {
	// cap 4 on a budget that holds exactly four of these models; touching an
	// eight-model rotation forces an LRU eviction (and a plan + sort) every call.
	const size = 4 << 30
	m := New(Policy{Device: "bench", BudgetBytes: 16 << 30, ConcurrentCap: 4})
	ids := []string{"m0", "m1", "m2", "m3", "m4", "m5", "m6", "m7"}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchDecision = m.Touch(ids[i%len(ids)], size)
	}
}

func BenchmarkManager_Resident(b *testing.B) {
	m := New(Policy{Device: "bench", BudgetBytes: 64 << 30, ConcurrentCap: 8})
	for _, id := range []string{"m0", "m1", "m2", "m3", "m4"} {
		m.Touch(id, 4<<30)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchIDs = m.Resident()
	}
}
