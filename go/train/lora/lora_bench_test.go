// SPDX-Licence-Identifier: EUPL-1.2

package lora_test

import (
	"context"
	"strconv"
	"testing"

	"dappco.re/go/inference/train/lora"
)

// Package sinks defeat dead-code elimination so the benchmarked work is real.
var (
	sinkString string
	sinkBool   bool
	sinkInt    int
	sinkID     string
	sinkRel    func()
	sinkRefs   []lora.AdapterRef
	sinkNames  []string
	sinkVictim string
	sinkOK     bool
)

// benchRef builds a realistic adapter ref (name + filesystem path + base model),
// matching the shape the Pool registers and the Loader applies from.
func benchRef(i int) lora.AdapterRef {
	n := "support-tone-" + strconv.Itoa(i)
	return lora.AdapterRef{Name: n, Path: "/adapters/" + n + "/adapter_model.safetensors", BaseModel: "gemma-e4b"}
}

// benchNoopLoader is a zero-work Loader so Pool benchmarks measure the pool
// logic (residency, ref-counting, eviction), not a fake's book-keeping allocs.
type benchNoopLoader struct{}

func (benchNoopLoader) Load(context.Context, lora.AdapterRef) error { return nil }
func (benchNoopLoader) Unload(context.Context, string) error        { return nil }

// BenchmarkAdapterRef_ID measures the deterministic id derivation (seed concat +
// SHA-256 + hex) that the Registry/Pool drive on most name→id resolutions.
func BenchmarkAdapterRef_ID(b *testing.B) {
	r := benchRef(0)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sinkID = r.ID()
	}
}

// BenchmarkRegistry_Acquire measures the per-lease acquire path (name lookup +
// id resolution), the work every Pool.Use performs to fence an adapter.
func BenchmarkRegistry_Acquire(b *testing.B) {
	r := lora.NewRegistry()
	_ = r.Register(benchRef(0))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		id, _ := r.Acquire("support-tone-0")
		sinkID = id
		r.Release(id)
	}
}

// BenchmarkRegistry_Register measures cataloguing an adapter (entry alloc + two
// map inserts + one id derivation).
func BenchmarkRegistry_Register(b *testing.B) {
	refs := make([]lora.AdapterRef, b.N)
	for i := range refs {
		refs[i] = benchRef(i)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		r := lora.NewRegistry()
		_ = r.Register(refs[i])
	}
}

// BenchmarkRegistry_Get measures pure name→ref resolution (no id work).
func BenchmarkRegistry_Get(b *testing.B) {
	r := lora.NewRegistry()
	_ = r.Register(benchRef(0))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ref, _ := r.Get("support-tone-0")
		sinkString = ref.Name
	}
}

// BenchmarkRegistry_List measures the sorted snapshot of all registered refs.
func BenchmarkRegistry_List(b *testing.B) {
	r := lora.NewRegistry()
	for i := 0; i < 8; i++ {
		_ = r.Register(benchRef(i))
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sinkRefs = r.List()
	}
}

// BenchmarkPool_Use_ResidentHit measures the hottest serving path: a Use of an
// already-resident adapter — the per-request steady state (Get + Acquire +
// recency bump + release).
func BenchmarkPool_Use_ResidentHit(b *testing.B) {
	p := lora.NewPool(lora.Config{Loader: benchNoopLoader{}, Policy: lora.NewLRUEvictionPolicy(), Capacity: 8})
	_ = p.Register(benchRef(0))
	ctx := context.Background()
	// Make it resident once up-front.
	_, rel, _ := p.Use(ctx, "support-tone-0")
	rel()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		id, release, _ := p.Use(ctx, "support-tone-0")
		sinkID = id
		release()
	}
}

// BenchmarkPool_Use_Evict measures the at-capacity path: every iteration forces
// the LRU eviction + load of a fresh adapter (capacity 1, two adapters).
func BenchmarkPool_Use_Evict(b *testing.B) {
	p := lora.NewPool(lora.Config{Loader: benchNoopLoader{}, Policy: lora.NewLRUEvictionPolicy(), Capacity: 1})
	_ = p.Register(benchRef(0))
	_ = p.Register(benchRef(1))
	ctx := context.Background()
	names := [2]string{"support-tone-0", "support-tone-1"}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		id, release, _ := p.Use(ctx, names[i&1])
		sinkID = id
		release()
	}
}

// BenchmarkPool_IsResident measures the residency query (Get + id + map lookup).
func BenchmarkPool_IsResident(b *testing.B) {
	p := lora.NewPool(lora.Config{Loader: benchNoopLoader{}, Policy: lora.NewLRUEvictionPolicy(), Capacity: 8})
	_ = p.Register(benchRef(0))
	_, rel, _ := p.Use(context.Background(), "support-tone-0")
	rel()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sinkBool = p.IsResident("support-tone-0")
	}
}

// BenchmarkPool_Resident measures the sorted resident-name snapshot at a typical
// working-set size (8 adapters).
func BenchmarkPool_Resident(b *testing.B) {
	p := lora.NewPool(lora.Config{Loader: benchNoopLoader{}, Policy: lora.NewLRUEvictionPolicy(), Capacity: 8})
	ctx := context.Background()
	for i := 0; i < 8; i++ {
		_ = p.Register(benchRef(i))
		_, rel, _ := p.Use(ctx, "support-tone-"+strconv.Itoa(i))
		rel()
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sinkNames = p.Resident()
	}
}

// BenchmarkPool_Pin measures pin of a resident adapter (Get + id + map set).
func BenchmarkPool_Pin(b *testing.B) {
	p := lora.NewPool(lora.Config{Loader: benchNoopLoader{}, Policy: lora.NewLRUEvictionPolicy(), Capacity: 8})
	_ = p.Register(benchRef(0))
	_, rel, _ := p.Use(context.Background(), "support-tone-0")
	rel()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		p.Pin("support-tone-0")
	}
}

// BenchmarkEviction_SelectVictim measures LRU victim selection over a full
// working set of candidate ids.
func BenchmarkEviction_SelectVictim(b *testing.B) {
	pol := lora.NewLRUEvictionPolicy()
	cands := make([]string, 8)
	for i := range cands {
		cands[i] = benchRef(i).ID()
		pol.MarkUsed(cands[i])
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		v, ok := pol.SelectVictim(cands)
		sinkVictim, sinkOK = v, ok
	}
}

// BenchmarkEviction_MarkUsed measures the recency stamp on the LRU policy.
func BenchmarkEviction_MarkUsed(b *testing.B) {
	pol := lora.NewLRUEvictionPolicy()
	id := benchRef(0).ID()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		pol.MarkUsed(id)
	}
	sinkInt = 0
}
