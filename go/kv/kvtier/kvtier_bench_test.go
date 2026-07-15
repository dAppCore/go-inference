// SPDX-Licence-Identifier: EUPL-1.2

package kvtier_test

import (
	"context"
	"strconv"
	"testing"

	"dappco.re/go/inference/kv/kvtier"
)

// noopStore is a zero-cost Store for benchmarks: the policy decides the moves,
// the real copier is irrelevant to the placement allocations under test, so the
// fake does nothing and allocates nothing.
type noopStore struct{}

func (noopStore) Move(context.Context, string, kvtier.Tier, kvtier.Tier) error { return nil }

// benchManager builds a manager pre-loaded with total 8 MB blocks where the GPU
// holds gpuBlocks of them and the remainder sit on CPU — the steady state of a
// long-context run: a small hot GPU set over a large warm CPU pool. ids are
// returned in put order (oldest first) so a round-robin access hits the CPU pool.
func benchManager(b *testing.B, gpuBlocks, total int) (*kvtier.Manager, []string) {
	b.Helper()
	const blk = 8 << 20 // 8 MB per block
	m := kvtier.New(kvtier.Budget{
		GPU:  int64(gpuBlocks) * blk,
		CPU:  int64(total) * blk, // CPU holds every non-GPU block, no disk cascade
		Disk: 1 << 50,
	}, noopStore{})
	ids := make([]string, total)
	ctx := context.Background()
	for i := range total {
		id := "seq:l" + strconv.Itoa(i)
		ids[i] = id
		if err := m.Put(ctx, kvtier.Block{ID: id, SizeBytes: blk}); err != nil {
			b.Fatalf("setup put %s: %v", id, err)
		}
	}
	return m, ids
}

var (
	sinkErr  error
	sinkSlc  []string
	sinkTier kvtier.Tier
)

// BenchmarkManager_Access_Hit measures the pure-hit fast path: a GPU-resident
// block accessed again bumps recency and moves nothing — the per-token best case
// that should stay at the allocation floor.
func BenchmarkManager_Access_Hit(b *testing.B) {
	m, ids := benchManager(b, 4, 4) // all 4 fit the GPU, every access is a hit
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sinkErr = m.Access(ctx, ids[i&3])
	}
}

// BenchmarkManager_Access_Promote measures the hot per-token promote path: a
// small GPU over a large CPU pool, accessed round-robin so every touch promotes
// a CPU block to the GPU and demotes the GPU LRU back to CPU (one rebalance hop).
func BenchmarkManager_Access_Promote(b *testing.B) {
	m, ids := benchManager(b, 2, 32)
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sinkErr = m.Access(ctx, ids[i%len(ids)])
	}
}

// BenchmarkManager_Access_Promote_Large is the same promote path over a much
// larger tracked set, surfacing the O(blocks) cost of the per-call projection
// map that planRebalance rebuilds on every rebalance.
func BenchmarkManager_Access_Promote_Large(b *testing.B) {
	m, ids := benchManager(b, 4, 256)
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sinkErr = m.Access(ctx, ids[i%len(ids)])
	}
}

// BenchmarkManager_Put_Refresh measures re-Put of an existing block (the runtime
// refreshing a page's size/recency): pulls it back to the GPU and rebalances,
// demoting the LRU — the per-request placement churn without growing the map.
func BenchmarkManager_Put_Refresh(b *testing.B) {
	m, ids := benchManager(b, 2, 32)
	ctx := context.Background()
	const blk = 8 << 20
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sinkErr = m.Put(ctx, kvtier.Block{ID: ids[i%len(ids)], SizeBytes: blk})
	}
}

// BenchmarkManager_Resident measures the diagnostic lister: it allocates a result
// slice and sorts it. Included so the query-path allocation is visible alongside
// the placement path.
func BenchmarkManager_Resident(b *testing.B) {
	m, _ := benchManager(b, 4, 32)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sinkSlc = m.Resident(kvtier.TierCPU)
	}
}

// BenchmarkManager_TierOf measures the single-block tier query (a map read under
// the lock) — the floor reference for the read-only accessors.
func BenchmarkManager_TierOf(b *testing.B) {
	m, ids := benchManager(b, 4, 32)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sinkTier = m.TierOf(ids[i%len(ids)])
	}
}
