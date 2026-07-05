// SPDX-Licence-Identifier: EUPL-1.2

package memory_test

import (
	"testing"

	"dappco.re/go/inference/kv/memory"
	mp "dappco.re/go/inference/modelpack"
)

// TestMemory_NewPlan_ContextDerivedFromMemory proves the plan derives context
// length from truth — the model's declared maximum bounded by what the machine
// actually holds — instead of pinning it at a per-RAM-class magic baseline that
// could only ever cap DOWN. A 256K-capable model on a big machine rises toward
// its declared max; the same model on a starved machine is bounded below it by
// the real memory budget.
func TestMemory_NewPlan_ContextDerivedFromMemory(t *testing.T) {
	model := func(weight uint64) *mp.ModelPack {
		return &mp.ModelPack{
			Architecture:  "gemma4_text",
			ContextLength: 262144, // model declares 256K
			NumLayers:     28,
			HiddenSize:    2048,
			WeightBytes:   weight,
			QuantBits:     6,
		}
	}

	big := memory.NewPlan(memory.Input{
		Device: memory.DeviceInfo{Architecture: "apple", MemorySize: 512 * memory.GiB, MaxRecommendedWorkingSetSize: 480 * memory.GiB},
		Pack:   model(8 * memory.GiB),
	})
	if big.ContextLength <= 131072 {
		t.Fatalf("big-RAM ContextLength = %d, want > 131072 (must rise above the old RAM-bucket cap toward the model's 256K)", big.ContextLength)
	}
	if big.ContextLength > 262144 {
		t.Fatalf("big-RAM ContextLength = %d, want <= 262144 (never exceed the model's declared maximum)", big.ContextLength)
	}

	small := memory.NewPlan(memory.Input{
		Device: memory.DeviceInfo{Architecture: "apple", MemorySize: 16 * memory.GiB, MaxRecommendedWorkingSetSize: 14 * memory.GiB},
		Pack:   model(4 * memory.GiB),
	})
	if small.ContextLength <= 0 {
		t.Fatalf("small-RAM ContextLength = %d, want > 0", small.ContextLength)
	}
	if small.ContextLength >= big.ContextLength {
		t.Fatalf("small-RAM ContextLength = %d, want < big-RAM %d (context bounded by device memory)", small.ContextLength, big.ContextLength)
	}
}

// TestMemory_NewPlan_ContextUsesRealKVWidth proves the derivation sizes the KV
// cache from the model's true grouped-query width (num_kv_heads * head_dim),
// not hidden_size: a model that declares its KV dims fits MORE context than the
// same model where the planner must fall back to the hidden-size over-estimate.
func TestMemory_NewPlan_ContextUsesRealKVWidth(t *testing.T) {
	dev := memory.DeviceInfo{Architecture: "apple", MemorySize: 96 * memory.GiB, MaxRecommendedWorkingSetSize: 80 * memory.GiB}
	base := func() *mp.ModelPack {
		return &mp.ModelPack{Architecture: "gemma4_text", ContextLength: 262144, NumLayers: 48, HiddenSize: 5120, WeightBytes: 12 * memory.GiB, QuantBits: 6}
	}

	// No KV dims declared → planner falls back to hidden_size (over-counts KV).
	fallback := memory.NewPlan(memory.Input{Device: dev, Pack: base()})

	// Real GQA width: 8 kv-heads x 256 head_dim = 2048, far below hidden 5120.
	gqa := base()
	gqa.NumKVHeads = 8
	gqa.HeadDim = 256
	real := memory.NewPlan(memory.Input{Device: dev, Pack: gqa})

	if real.ContextLength <= fallback.ContextLength {
		t.Fatalf("real-KV-width ContextLength = %d, want > hidden-fallback %d (GQA KV is smaller, so more context fits)", real.ContextLength, fallback.ContextLength)
	}
}

// TestMemory_NewPlan_SlotsBatchDeriveNoInversion proves the concurrency capacity
// is derived from truth — the count of full model-context windows the machine's
// post-weights KV budget holds — and is monotonic in memory. The old per-class
// slot baseline (96GB→2, 64GB→1) made a LARGER machine divide its KV budget
// harder than the extra RAM grew it, so a 96GB box could derive a SMALLER
// context than a 64GB one. A derived capacity cannot invert: more RAM never
// yields fewer slots, and so never a smaller per-slot context. Batch tracks
// slots — one capacity drives both the concurrency semaphore and the decode
// batch, keeping fitContextLength's ÷slots coherent with the KV ×batch estimate.
func TestMemory_NewPlan_SlotsBatchDeriveNoInversion(t *testing.T) {
	// 28-layer GQA model: kv width = 4 heads x 256 head_dim = 1024, far below
	// the 2048 hidden size, and weights heavy enough that 64GB cannot cap at
	// the model max — so the raw budget÷slots division is what gets compared.
	model := func() *mp.ModelPack {
		return &mp.ModelPack{
			Architecture: "gemma4_text", ContextLength: 262144,
			NumLayers: 28, HiddenSize: 2048, NumKVHeads: 4, HeadDim: 256,
			WeightBytes: 20 * memory.GiB, QuantBits: 6,
		}
	}
	plan := func(mem, ws uint64) memory.Plan {
		return memory.NewPlan(memory.Input{
			Device: memory.DeviceInfo{Architecture: "apple", MemorySize: mem, MaxRecommendedWorkingSetSize: ws},
			Pack:   model(),
		})
	}
	p64 := plan(64*memory.GiB, 60*memory.GiB)
	p96 := plan(96*memory.GiB, 90*memory.GiB)
	p512 := plan(512*memory.GiB, 480*memory.GiB)

	// Context never shrinks as memory grows — the inversion is impossible.
	if !(p64.ContextLength <= p96.ContextLength && p96.ContextLength <= p512.ContextLength) {
		t.Fatalf("context not monotonic in RAM: 64GB=%d 96GB=%d 512GB=%d (a larger machine must never derive a smaller context)", p64.ContextLength, p96.ContextLength, p512.ContextLength)
	}
	// Slots never shrink as memory grows.
	if !(p64.ParallelSlots <= p96.ParallelSlots && p96.ParallelSlots <= p512.ParallelSlots) {
		t.Fatalf("slots not monotonic in RAM: 64GB=%d 96GB=%d 512GB=%d", p64.ParallelSlots, p96.ParallelSlots, p512.ParallelSlots)
	}
	// One derived capacity drives both: batch == slots on every machine.
	for _, p := range []memory.Plan{p64, p96, p512} {
		if p.BatchSize != p.ParallelSlots {
			t.Fatalf("batch %d != slots %d — the two must be the one derived capacity", p.BatchSize, p.ParallelSlots)
		}
	}
}

// TestMemory_NewPlan_SlotsScaleWithCapacity proves slots are the real count of
// full-context windows that fit, not a capped per-class guess. A large machine
// running a model whose context window is a small fraction of its KV budget
// derives many concurrent slots (well past the old baseline cap of 2), each
// still holding the model's full declared context; a starved machine running a
// model that barely fits derives a single slot.
func TestMemory_NewPlan_SlotsScaleWithCapacity(t *testing.T) {
	big := memory.NewPlan(memory.Input{
		Device: memory.DeviceInfo{Architecture: "apple", MemorySize: 512 * memory.GiB, MaxRecommendedWorkingSetSize: 480 * memory.GiB},
		Pack: &mp.ModelPack{
			Architecture: "gemma4_text", ContextLength: 32768,
			NumLayers: 28, HiddenSize: 2048, NumKVHeads: 4, HeadDim: 256,
			WeightBytes: 8 * memory.GiB, QuantBits: 6,
		},
	})
	if big.ParallelSlots <= 2 {
		t.Fatalf("big-box small-model ParallelSlots = %d, want > 2 (derived capacity, not the old per-class cap)", big.ParallelSlots)
	}
	if big.ContextLength != 32768 {
		t.Fatalf("big-box ContextLength = %d, want the model's full 32768 held in every slot", big.ContextLength)
	}

	starved := memory.NewPlan(memory.Input{
		Device: memory.DeviceInfo{Architecture: "apple", MemorySize: 16 * memory.GiB, MaxRecommendedWorkingSetSize: 14 * memory.GiB},
		Pack: &mp.ModelPack{
			Architecture: "gemma4_text", ContextLength: 262144,
			NumLayers: 48, HiddenSize: 5120, NumKVHeads: 8, HeadDim: 256,
			WeightBytes: 8 * memory.GiB, QuantBits: 6,
		},
	})
	if starved.ParallelSlots != 1 {
		t.Fatalf("starved-box big-model ParallelSlots = %d, want 1 (only one window fits)", starved.ParallelSlots)
	}
}

// TestMemory_NewPlan_SlotsBatchColdStartDefault proves that with no model to
// derive from, the plan reports the honest local default — one foreground slot,
// batch one — for EVERY machine class, instead of a per-RAM-class guess at a
// concurrency it cannot know without the model. Real capacity is derived only
// once a model's footprint is known.
func TestMemory_NewPlan_SlotsBatchColdStartDefault(t *testing.T) {
	for _, mem := range []uint64{16, 64, 96, 128, 512} {
		p := memory.NewPlan(memory.Input{
			Device: memory.DeviceInfo{Architecture: "apple", MemorySize: mem * memory.GiB, MaxRecommendedWorkingSetSize: (mem - 4) * memory.GiB},
		})
		if p.ParallelSlots != 1 || p.BatchSize != 1 {
			t.Fatalf("%dGB cold-start slots/batch = %d/%d, want 1/1 (no model → honest local default)", mem, p.ParallelSlots, p.BatchSize)
		}
	}
}
